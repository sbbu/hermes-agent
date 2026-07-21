"""Supervisor for the dashboard compute-host child process.

The dashboard process owns sockets and JSON-RPC dispatch.  When
``dashboard.turn_isolation`` is enabled, agent turns move behind one persistent
``python -m tui_gateway.compute_host`` child so compute-heavy agent threads do
not contend with the serving process' event loop for the same GIL.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)
_Thread = threading.Thread

MUTATOR_ROUTE_TABLE: dict[str, str] = {
    "prompt.submit": "turn-path",
    "session.interrupt": "turn-path",
    "reload.mcp": "run-concurrent",
    "session.save": "run-concurrent",
    "session.compress": "idle-gated",
    "prompt.submit.truncate": "idle-gated",
    "slash.model": "idle-gated",
    "slash.personality": "idle-gated",
    "slash.prompt": "idle-gated",
    "slash.compress": "idle-gated",
    "session.reset": "idle-gated",
    "session.history.reload": "idle-gated",
    "slash.retry": "idle-gated",
}

_REGISTRY_NAME = "dashboard-compute-host.json"
_RESPAWN_WINDOW_SECS = 300.0
_SHUTDOWN_TIMEOUT_SECS = 10.0
_LEGACY_REGISTRY_START_TOLERANCE_S = 30.0
_REGISTRY_LOCK_TIMEOUT_SECS = 20.0


def append_log_record(path: str | Path, record: str) -> None:
    """Append one log record using O_APPEND and exactly one os.write call."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = record if record.endswith("\n") else f"{record}\n"
    data = text.encode("utf-8", errors="replace")
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def _default_registry_path() -> Path:
    return get_hermes_home() / "state" / _REGISTRY_NAME


@contextmanager
def _registry_file_lock(registry_path: Path):
    """Serialize registry reconcile/write/remove across dashboard processes."""
    from gateway.status import _release_file_lock, _try_acquire_file_lock

    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + _REGISTRY_LOCK_TIMEOUT_SECS
    try:
        while not _try_acquire_file_lock(handle):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring compute-host registry lock: {lock_path}")
            time.sleep(0.05)
        try:
            yield
        finally:
            _release_file_lock(handle)
    finally:
        handle.close()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception:
        # Failure must be conservative on Windows: treating an unknown PID as
        # dead would discard the registry and spawn a second compute host.
        if sys.platform == "win32":
            return True
    try:
        os.kill(pid, 0)  # windows-footgun: ok — guarded by the win32 branch
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _pid_start_time(pid: int) -> int | None:
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _pid_create_time(pid: int) -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _pid_command(pid: int) -> str:
    if pid <= 0:
        return ""
    if sys.platform == "win32":
        try:
            import psutil  # type: ignore

            return " ".join(str(part) for part in psutil.Process(pid).cmdline())
        except Exception:
            return ""
    # Linux fast path.
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        data = proc_cmdline.read_bytes()
        if data:
            return data.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except Exception:
        pass
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def is_compute_host_identity(pid: int) -> bool:
    cmd = _pid_command(pid)
    return "tui_gateway.compute_host" in cmd


class HostSupervisor:
    """Own one persistent compute-host child and relay its frames."""

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        argv: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        rpc_sink: Callable[[dict], None] | None = None,
        respawn_max: int = 3,
        heartbeat_secs: int = 15,
        expected_build_sha: str | None = None,
        expected_hermes_home: str | None = None,
        autostart: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path) if registry_path is not None else _default_registry_path()
        self.argv = argv or [sys.executable, "-m", "tui_gateway.compute_host"]
        self.cwd = Path(cwd) if cwd is not None else _repo_root()
        self.env = env
        self.rpc_sink = rpc_sink or (lambda _obj: None)
        self.respawn_max = max(0, int(respawn_max))
        self.heartbeat_secs = max(1, int(heartbeat_secs))
        self.expected_build_sha = expected_build_sha if expected_build_sha is not None else _build_sha()
        self.expected_hermes_home = expected_hermes_home if expected_hermes_home is not None else str(get_hermes_home())

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._wait_thread: threading.Thread | None = None
        self._hello_event = threading.Event()
        self._hello: dict[str, Any] = {}
        self._closing = False
        self._stopped_respawning = False
        self._restart_times: list[float] = []
        self._pending_turns: dict[
            str,
            tuple[
                subprocess.Popen[str] | None,
                str,
                Callable[[dict], None] | None,
            ],
        ] = {}
        self._pending_controls: dict[
            str,
            tuple[subprocess.Popen[str] | None, queue.Queue[dict]],
        ] = {}
        self._stderr_tail: list[str] = []
        self._last_progress_counter = 0
        self._owned_registry_identity: tuple[int, int | None, str] | None = None
        self._owned_registry_proc: subprocess.Popen[str] | None = None

        if autostart:
            self.start()

    @property
    def pid(self) -> int:
        proc = self._proc
        return int(proc.pid or 0) if proc is not None else 0

    @property
    def hello(self) -> dict[str, Any]:
        return dict(self._hello)

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None and not self._stopped_respawning

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._closing = False
            self._start_locked(reason="startup")

    def _start_locked(self, *, reason: str) -> None:
        with _registry_file_lock(self.registry_path):
            reconciliation = self._reconcile_startup_orphan_locked()
            if reconciliation in {
                "identity-unverified",
                "termination-failed",
                "owned-by-live-supervisor",
                "owner-unverified",
            }:
                raise RuntimeError(
                    "compute host startup blocked because the registered host "
                    "identity could not be verified or terminated, or another "
                    "live supervisor owns it"
                )
            self._spawn_locked(reason=reason, registry_lock_held=True)

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                self._send_frame({"type": "shutdown", "request_id": f"shutdown-{uuid.uuid4().hex}"})
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
        except Exception:
            self._terminate_process(proc)
        finally:
            self._remove_registry_for_proc(proc)

    def reconcile_startup_orphan(self) -> str:
        """Terminate a stale registered host, guarding against PID reuse."""
        with _registry_file_lock(self.registry_path):
            return self._reconcile_startup_orphan_locked()

    def _reconcile_startup_orphan_locked(self) -> str:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "none"
        except Exception:
            self._remove_registry_locked()
            return "invalid-registry"

        try:
            owner_pid = int(data.get("owner_pid") or 0)
        except (TypeError, ValueError):
            owner_pid = 0
        if owner_pid > 0 and owner_pid != os.getpid() and _pid_alive(owner_pid):
            owner_start = data.get("owner_start_time")
            current_owner_start = _pid_start_time(owner_pid)
            try:
                owner_start = int(owner_start) if owner_start is not None else None
            except (TypeError, ValueError):
                owner_start = None
            if owner_start is None or current_owner_start is None:
                return "owner-unverified"
            if owner_start == current_owner_start:
                return "owned-by-live-supervisor"

        try:
            pid = int(data.get("host_pid") or 0)
        except Exception:
            pid = 0
        if pid <= 0 or not _pid_alive(pid):
            self._remove_registry_locked()
            return "not-running"

        recorded_start = data.get("process_start_time")
        current_start = _pid_start_time(pid)
        try:
            recorded_start = (
                int(recorded_start) if recorded_start is not None else None
            )
        except (TypeError, ValueError):
            recorded_start = None

        expected_start: int | None = None
        if recorded_start is not None:
            if current_start is not None and current_start != recorded_start:
                self._remove_registry_locked()
                return "pid-reuse-ignored"
            if current_start is not None:
                expected_start = recorded_start
        else:
            # Pre-fingerprint registries still record when the host completed
            # startup.  Compare that wall time to the live process creation
            # time, then capture the current platform-native fingerprint for
            # psutil's guarded termination path.
            try:
                registry_started_at = float(data.get("started_at"))
            except (TypeError, ValueError):
                registry_started_at = None
            process_created_at = _pid_create_time(pid)
            if registry_started_at is not None and process_created_at is not None:
                startup_delta = registry_started_at - process_created_at
                if not (
                    0.0
                    <= startup_delta
                    <= _LEGACY_REGISTRY_START_TOLERANCE_S
                ):
                    self._remove_registry_locked()
                    return "pid-reuse-ignored"
                if current_start is not None:
                    expected_start = current_start

        command = _pid_command(pid)
        if not command:
            return "identity-unverified"
        if "tui_gateway.compute_host" not in command:
            self._remove_registry_locked()
            return "pid-reuse-ignored"
        if expected_start is None:
            # A matching live PID without a stable instance fingerprint is
            # ambiguous. Keep the registry and refuse to spawn beside it.
            return "identity-unverified"

        terminated = self._terminate_pid(
            pid,
            timeout=_SHUTDOWN_TIMEOUT_SECS,
            expected_start_time=expected_start,
        )
        if not terminated:
            return "termination-failed"
        self._remove_registry_locked()
        return "terminated"

    def submit_turn(
        self,
        frame: dict[str, Any],
        *,
        on_complete: Callable[[dict], None] | None = None,
    ) -> str:
        self.start()
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        sid = str(frame.get("sid") or "")
        payload = dict(frame)
        payload["type"] = "turn.start"
        payload["request_id"] = request_id
        try:
            with self._lock:
                self._pending_turns[request_id] = (self._proc, sid, on_complete)
                self._send_frame(payload)
        except Exception as exc:
            with self._lock:
                self._pending_turns.pop(request_id, None)
            err = {
                "type": "turn.error",
                "sid": sid,
                "request_id": request_id,
                "reason": "send_failed",
                "message": str(exc),
            }
            if on_complete is not None:
                on_complete(err)
            raise
        return request_id

    def interrupt(self, sid: str, *, request_id: str | None = None) -> None:
        self.start()
        self._send_frame({"type": "interrupt", "sid": sid, "request_id": request_id or uuid.uuid4().hex})

    def force_release(
        self,
        sid: str,
        *,
        timeout: float = 30.0,
        wait: bool = True,
        clear_queued_prompt: bool = False,
    ) -> dict:
        """Rebuild one stuck host session without restarting the shared host."""
        self.start()
        request_id = uuid.uuid4().hex
        q: queue.Queue[dict] | None = queue.Queue(maxsize=1) if wait else None
        try:
            with self._lock:
                if q is not None:
                    self._pending_controls[request_id] = (self._proc, q)
                self._send_frame(
                    {
                        "type": "force_release",
                        "sid": sid,
                        "request_id": request_id,
                        "clear_queued_prompt": clear_queued_prompt,
                    }
                )
            if q is None:
                return {"status": "sent", "request_id": request_id}
            return q.get(timeout=timeout)
        finally:
            if q is not None:
                with self._lock:
                    self._pending_controls.pop(request_id, None)

    def reload_mcp(self, sid: str, *, request_id: str | None = None) -> dict:
        response = self.control(
            sid,
            route_name="reload.mcp",
            payload={"type": "reload_mcp", "sid": sid, "request_id": request_id or uuid.uuid4().hex},
            wait=True,
        )
        if response.get("type") in {"control.error", "error"}:
            raise RuntimeError(str(response.get("message") or "compute-host reload failed"))
        nested_response = response.get("response")
        if isinstance(nested_response, dict) and nested_response.get("error"):
            nested_error = nested_response["error"]
            message = (
                nested_error.get("message")
                if isinstance(nested_error, dict)
                else nested_error
            )
            raise RuntimeError(str(message or "compute-host reload failed"))
        return response

    def control(
        self,
        sid: str,
        *,
        route_name: str,
        payload: dict[str, Any] | None = None,
        wait: bool = True,
        timeout: float = 30.0,
    ) -> dict:
        if route_name not in MUTATOR_ROUTE_TABLE:
            raise ValueError(f"unclassified host mutator route: {route_name}")
        self.start()
        request_id = str((payload or {}).get("request_id") or uuid.uuid4().hex)
        frame = dict(payload or {})
        frame.setdefault("type", "control")
        frame["sid"] = sid
        frame["route_name"] = route_name
        frame["request_id"] = request_id
        q: queue.Queue[dict] | None = None
        if wait:
            q = queue.Queue(maxsize=1)
        try:
            with self._lock:
                if q is not None:
                    self._pending_controls[request_id] = (self._proc, q)
                self._send_frame(frame)
            if not wait or q is None:
                return {"status": "sent", "request_id": request_id}
            return q.get(timeout=timeout)
        finally:
            if q is not None:
                with self._lock:
                    self._pending_controls.pop(request_id, None)

    def _spawn_locked(
        self,
        *,
        reason: str,
        registry_lock_held: bool = False,
    ) -> None:
        if self._stopped_respawning:
            raise RuntimeError("compute host respawn disabled after crash loop")
        self._hello_event.clear()
        self._hello = {}
        env = hermes_subprocess_env(inherit_credentials=True)
        env.update(os.environ)
        if self.env:
            env.update(self.env)
        env["HERMES_COMPUTE_HOST_HEARTBEAT_SECS"] = str(self.heartbeat_secs)
        env.setdefault("PYTHONPATH", str(_repo_root()))
        if str(_repo_root()) not in env["PYTHONPATH"].split(os.pathsep):
            env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env["PYTHONPATH"]
        proc = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._proc = proc
        stdout_thread = _Thread(
            target=self._drain_stdout,
            args=(proc,),
            name="compute-host-stdout",
            daemon=True,
        )
        stderr_thread = _Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="compute-host-stderr",
            daemon=True,
        )
        wait_thread = _Thread(
            target=self._wait_for_exit,
            args=(proc, stdout_thread),
            name="compute-host-wait",
            daemon=True,
        )
        self._stdout_thread = stdout_thread
        self._stderr_thread = stderr_thread
        self._wait_thread = wait_thread
        stdout_thread.start()
        stderr_thread.start()
        wait_thread.start()
        try:
            if not self._hello_event.wait(timeout=10.0):
                raise RuntimeError(
                    f"compute host did not send hello; stderr={self._stderr_tail[-5:]}"
                )
            self._validate_hello()
            self._persist_registry(lock_held=registry_lock_held)
        except Exception:
            # Detach before termination so the exit waiter cannot treat this
            # rejected child as the active host or schedule a crash respawn.
            if self._proc is proc:
                self._proc = None
            self._terminate_process(proc)
            raise
        logger.info("compute host started pid=%s reason=%s", proc.pid, reason)

    def _validate_hello(self) -> None:
        hello = self._hello
        if not hello:
            raise RuntimeError("compute host missing hello")
        got_home = str(hello.get("hermes_home") or "")
        if got_home and got_home != self.expected_hermes_home:
            raise RuntimeError(f"compute host HERMES_HOME mismatch: {got_home} != {self.expected_hermes_home}")
        got_sha = str(hello.get("build_sha") or "")
        if self.expected_build_sha != "unknown" and got_sha not in {"", "unknown", self.expected_build_sha}:
            raise RuntimeError(f"compute host build mismatch: {got_sha} != {self.expected_build_sha}")

    def _persist_registry(self, *, lock_held: bool = False) -> None:
        if lock_held:
            self._persist_registry_locked()
            return
        with _registry_file_lock(self.registry_path):
            self._persist_registry_locked()

    def _persist_registry_locked(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        host_pid = self.pid
        process_start_time = _pid_start_time(host_pid)
        boot_id = str(self._hello.get("boot_id") or "")
        payload = {
            "host_pid": host_pid,
            "process_start_time": process_start_time,
            "owner_pid": os.getpid(),
            "owner_start_time": _pid_start_time(os.getpid()),
            "boot_id": boot_id,
            "build_sha": self._hello.get("build_sha") or "",
            "started_at": time.time(),
            "argv": self.argv,
        }
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)
        self._owned_registry_identity = (host_pid, process_start_time, boot_id)
        self._owned_registry_proc = self._proc

    def _remove_registry(self) -> None:
        proc = self._owned_registry_proc
        if proc is not None:
            self._remove_registry_for_proc(proc)

    def _remove_registry_for_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            if self._owned_registry_proc is not proc:
                return
            identity = self._owned_registry_identity
        if identity is None:
            return
        with _registry_file_lock(self.registry_path):
            self._remove_registry_locked(expected_identity=identity)
        with self._lock:
            if (
                self._owned_registry_proc is proc
                and self._owned_registry_identity == identity
            ):
                self._owned_registry_identity = None
                self._owned_registry_proc = None

    def _remove_registry_locked(
        self,
        *,
        expected_identity: tuple[int, int | None, str] | None = None,
    ) -> bool:
        if expected_identity is not None:
            try:
                current = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return True
            except Exception:
                return False
            current_identity = (
                int(current.get("host_pid") or 0),
                current.get("process_start_time"),
                str(current.get("boot_id") or ""),
            )
            if current_identity != expected_identity:
                return False
        try:
            self.registry_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except Exception:
            logger.debug("failed to remove compute host registry", exc_info=True)
            return False

    def _send_frame(self, frame: dict[str, Any]) -> None:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise RuntimeError("compute host is not running")
            proc.stdin.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=False) + "\n")
            proc.stdin.flush()

    def _drain_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("compute host emitted invalid json: %r", raw[:200])
                continue
            if isinstance(frame, dict):
                self._handle_host_frame(frame, proc=proc)

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            text = raw.rstrip("\n")
            if text:
                self._stderr_tail = (self._stderr_tail + [text])[-80:]
                logger.warning("compute host stderr: %s", text)

    def _handle_host_frame(
        self,
        frame: dict[str, Any],
        *,
        proc: subprocess.Popen[str] | None = None,
    ) -> None:
        ftype = str(frame.get("type") or "")
        if ftype == "hello":
            self._hello = dict(frame)
            self._hello_event.set()
            return
        if ftype == "hb":
            self._last_progress_counter = int(frame.get("progress_counter") or self._last_progress_counter)
            logger.debug("compute host heartbeat: %s", frame)
            return
        if ftype == "rpc":
            message = frame.get("message")
            if isinstance(message, dict):
                self.rpc_sink(message)
            return
        if ftype in {"turn.end", "turn.error"}:
            self._complete_turn(frame, proc=proc)
            return
        if ftype in {
            "control.ack",
            "control.error",
            "force_release.ack",
            "interrupt.ack",
            "reload_mcp.ack",
            "shutdown.ack",
        }:
            request_id = str(frame.get("request_id") or "")
            with self._lock:
                pending = self._pending_controls.get(request_id)
                if pending is not None and (proc is None or pending[0] is proc):
                    self._pending_controls.pop(request_id, None)
                    q = pending[1]
                else:
                    q = None
            if q is not None:
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass
            return
        if ftype == "error" and frame.get("request_id"):
            request_id = str(frame.get("request_id") or "")
            with self._lock:
                pending = self._pending_controls.get(request_id)
                if pending is not None and (proc is None or pending[0] is proc):
                    self._pending_controls.pop(request_id, None)
                    q = pending[1]
                else:
                    q = None
            if q is not None:
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

    def _complete_turn(
        self,
        frame: dict[str, Any],
        *,
        proc: subprocess.Popen[str] | None = None,
    ) -> None:
        request_id = str(frame.get("request_id") or "")
        with self._lock:
            pending = self._pending_turns.get(request_id)
            if pending is not None and (proc is None or pending[0] is proc):
                self._pending_turns.pop(request_id, None)
            else:
                pending = None
        if pending is None:
            return
        _owner_proc, _sid, cb = pending
        if cb is not None:
            try:
                cb(frame)
            except Exception:
                logger.exception("compute host turn completion callback failed")

    def _wait_for_exit(
        self,
        proc: subprocess.Popen[str],
        stdout_thread: threading.Thread | None = None,
    ) -> None:
        code = proc.wait()
        # A child can flush its terminal response immediately before exit. Let
        # that process' stdout reader claim completed requests before synthesizing
        # crash errors for whatever remains unresolved. This is deliberately an
        # unbounded join: frame handling is ordered, and the production websocket
        # sink has its own bounded write timeout; timing out here can overtake a
        # terminal ack queued behind a slow event and falsely fail the request.
        if stdout_thread is not None and stdout_thread is not threading.current_thread():
            stdout_thread.join()

        pending_turns: dict[str, tuple[str, Callable[[dict], None] | None]] = {}
        pending_controls: dict[str, queue.Queue[dict]] = {}
        with self._lock:
            current_process = self._proc is proc
            if current_process:
                self._proc = None
            closing = self._closing
            for request_id, (owner_proc, sid, callback) in list(
                self._pending_turns.items()
            ):
                if owner_proc is proc:
                    pending_turns[request_id] = (sid, callback)
                    self._pending_turns.pop(request_id, None)
            for request_id, (owner_proc, response_queue) in list(
                self._pending_controls.items()
            ):
                if owner_proc is proc:
                    pending_controls[request_id] = response_queue
                    self._pending_controls.pop(request_id, None)

        if current_process:
            self._remove_registry_for_proc(proc)
        reason = "shutdown" if closing else "crash"
        message = (
            f"compute host shut down with code {code}"
            if closing
            else f"compute host exited with code {code}"
        )
        # Wake bounded control RPCs before invoking arbitrary turn callbacks.
        self._fail_pending_controls(
            pending_controls,
            reason=reason,
            message=message,
        )
        try:
            self._fail_pending_turns(
                pending_turns,
                reason=reason,
                message=message,
            )
        finally:
            if current_process and not closing:
                self._maybe_respawn_after_crash()

    def _fail_pending_turns(
        self,
        pending: dict[str, tuple[str, Callable[[dict], None] | None]],
        *,
        reason: str,
        message: str,
    ) -> None:
        for request_id, (sid, cb) in pending.items():
            frame = {
                "type": "turn.error",
                "sid": sid,
                "request_id": request_id,
                "reason": reason,
                "message": message,
            }
            try:
                self.rpc_sink(
                    {
                        "jsonrpc": "2.0",
                        "method": "event",
                        "params": {
                            "type": "error",
                            "session_id": sid,
                            "payload": {"message": message, "reason": reason},
                        },
                    }
                )
            except Exception:
                logger.exception("compute host crash event sink failed")
            if cb is not None:
                try:
                    cb(frame)
                except Exception:
                    logger.exception("compute host error callback failed")

    def _fail_pending_controls(
        self,
        pending: dict[str, queue.Queue[dict]],
        *,
        reason: str,
        message: str,
    ) -> None:
        """Wake synchronous control callers when their child exits."""
        for request_id, response_queue in pending.items():
            try:
                response_queue.put_nowait(
                    {
                        "type": "control.error",
                        "request_id": request_id,
                        "reason": reason,
                        "message": message,
                    }
                )
            except queue.Full:
                # A terminal response won the race with process exit.
                pass

    def _maybe_respawn_after_crash(self) -> None:
        now = time.monotonic()
        self._restart_times = [t for t in self._restart_times if now - t <= _RESPAWN_WINDOW_SECS]
        if len(self._restart_times) >= self.respawn_max:
            self._stopped_respawning = True
            logger.error("compute host crash loop: max %s restarts per 5min reached; not respawning", self.respawn_max)
            return
        self._restart_times.append(now)
        # Small bounded backoff; tests and first recovery stay quick.
        delay = min(5.0, 0.25 * (2 ** max(0, len(self._restart_times) - 1)))

        def _respawn() -> None:
            time.sleep(delay)
            with self._lock:
                if self._closing or self._stopped_respawning or self._proc is not None:
                    return
                try:
                    self._start_locked(reason="crash")
                except Exception:
                    logger.exception("compute host respawn failed")

        _Thread(target=_respawn, name="compute-host-respawn", daemon=True).start()

    def _terminate_pid(
        self,
        pid: int,
        *,
        timeout: float = _SHUTDOWN_TIMEOUT_SECS,
        expected_start_time: int | None = None,
    ) -> bool:
        """Terminate one verified process instance without crossing PID reuse."""
        if expected_start_time is None:
            return False
        try:
            import psutil  # type: ignore
        except ImportError:
            logger.debug("psutil unavailable; refusing unguarded PID termination")
            return False

        try:
            proc = psutil.Process(pid)
            if _pid_start_time(pid) != expected_start_time:
                return False
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                # psutil's Process object retains PID + creation-time identity;
                # kill() refuses a recycled PID before signaling it.
                proc.kill()
                proc.wait(timeout=2)
            return True
        except psutil.NoSuchProcess:
            return True
        except Exception:
            logger.debug(
                "failed to terminate verified compute host pid=%s",
                pid,
                exc_info=True,
            )
            return False

    def _terminate_process(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


__all__ = [
    "MUTATOR_ROUTE_TABLE",
    "HostSupervisor",
    "append_log_record",
    "is_compute_host_identity",
]
