import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    MUTATOR_ROUTE_TABLE,
    HostSupervisor,
    append_log_record,
)


def _json_lines(out: io.StringIO) -> list[dict]:
    frames = []
    for line in out.getvalue().splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def _wait_for_frame(out: io.StringIO, predicate, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _json_lines(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for frame; saw={_json_lines(out)}")


def test_compute_host_workers_inherit_tui_pool_env_or_8(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_RPC_POOL_WORKERS", raising=False)
    monkeypatch.delenv("HERMES_COMPUTE_HOST_WORKERS", raising=False)
    assert _default_workers() == 8

    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "11")
    assert _default_workers() == 11

    # Dead-RC tombstone: malformed env falls back to 8, not the old except-branch 4.
    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "not-an-int")
    assert _default_workers() == 8


def test_compute_host_frame_protocol_round_trip():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=2, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-1",
                "prompt": "hello",
                "delta_count": 3,
                "delay_s": 0,
            }
        )

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-1")
        assert end["history_version"] == 1
        frames = _json_lines(out)
        assert [f["type"] for f in frames if f.get("request_id") == "turn-1"] == [
            "turn.started",
            "delta",
            "delta",
            "delta",
            "turn.end",
        ]
    finally:
        host.close()


def test_compute_host_interrupt_control_is_not_queued_behind_turn():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-slow",
                "prompt": "hello",
                "delta_count": 200,
                "delay_s": 0.01,
            }
        )
        _wait_for_frame(out, lambda f: f.get("type") == "delta" and f.get("request_id") == "turn-slow")

        host.handle_frame({"type": "interrupt", "sid": "alpha", "request_id": "stop-1"})
        ack = _wait_for_frame(out, lambda f: f.get("type") == "interrupt.ack" and f.get("request_id") == "stop-1")
        assert ack["applied"] is True

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-slow")
        assert end["interrupted"] is True
        typed = [f["type"] for f in _json_lines(out)]
        assert typed.index("interrupt.ack") < typed.index("turn.end")
    finally:
        host.close()


def test_compute_host_force_release_rebuilds_only_the_stuck_session(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)

    class _OldAgent:
        def __init__(self):
            self._session_db = object()
            self.interrupted = False

        def interrupt(self, *_args, **_kwargs):
            self.interrupted = True

    old_agent = _OldAgent()
    replacement = object()
    ready = threading.Event()
    ready.set()
    session = {
        "agent": old_agent,
        "agent_ready": ready,
        "agent_build_started": True,
        "agent_build_generation": 1,
        "session_key": "key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "run_generation": 1,
        "running": True,
        "inflight_turn": {"user": "stuck"},
        "slash_worker": None,
    }
    server._sessions["real-sid"] = session

    def _build(_sid, current):
        current["agent"] = replacement
        current["agent_ready"].set()

    monkeypatch.setattr(server, "_start_agent_build", _build)
    try:
        host.handle_frame(
            {
                "type": "force_release",
                "sid": "real-sid",
                "request_id": "release-1",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "force_release.ack",
        )
    finally:
        server._sessions.pop("real-sid", None)
        host.close()

    assert ack["request_id"] == "release-1"
    assert ack["applied"] is True
    assert old_agent.interrupted is True
    assert old_agent._session_db is None
    assert session["agent"] is replacement
    assert session["running"] is False
    assert "real-sid" in host._force_release_bypass


def test_force_released_turn_bypasses_exhausted_executor(monkeypatch):
    host = ComputeHost(stdout=io.StringIO(), max_workers=1, heartbeat_secs=0)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    replacement_started = threading.Event()

    def _block_worker():
        blocker_started.set()
        release_blocker.wait(timeout=2)

    host._executor.submit(_block_worker)
    assert blocker_started.wait(timeout=1)
    monkeypatch.setattr(
        host,
        "_run_real_turn",
        lambda _frame: replacement_started.set(),
    )
    with host._force_release_lock:
        host._force_release_bypass.add("wedged")
    try:
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "wedged",
                "request_id": "replacement",
            }
        )
        assert replacement_started.wait(timeout=1)
    finally:
        release_blocker.set()
        host.close()


def test_supervisor_force_release_waits_for_matching_ack(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    sent = []

    monkeypatch.setattr(supervisor, "start", lambda: None)

    def _send(frame):
        sent.append(frame)
        supervisor._handle_host_frame(
            {
                "type": "force_release.ack",
                "sid": frame["sid"],
                "request_id": frame["request_id"],
                "applied": True,
            }
        )

    monkeypatch.setattr(supervisor, "_send_frame", _send)

    ack = supervisor.force_release("sid")

    assert ack["applied"] is True
    assert sent[0]["type"] == "force_release"
    assert sent[0]["sid"] == "sid"

    sent.clear()
    result = supervisor.force_release(
        "sid-no-wait",
        wait=False,
        clear_queued_prompt=True,
    )
    assert result["status"] == "sent"
    assert sent[0]["clear_queued_prompt"] is True


def test_compute_host_flushes_sessions_on_orphan_shutdown(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    session = {"session_key": "key"}
    calls: list[tuple[dict, str]] = []
    server._sessions["flush-sid"] = session
    monkeypatch.setattr(
        server,
        "_finalize_session",
        lambda sess, end_reason="tui_close": calls.append((sess, end_reason)),
    )
    try:
        host.flush_all_sessions(reason="orphan")
        assert calls == [(session, "compute_host_orphan")]
    finally:
        server._sessions.pop("flush-sid", None)
        host.close()


def test_compute_host_parent_guard_exits_when_parent_pid_changes(monkeypatch):
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    host._parent_pid = 111
    monkeypatch.setattr(os, "getppid", lambda: 222)

    def _exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _exit)

    with pytest.raises(SystemExit) as exc_info:
        host._parent_guard_loop()

    assert exc_info.value.code == 0
    orphan = next(frame for frame in _json_lines(out) if frame.get("type") == "orphan")
    assert orphan["old_ppid"] == 111
    assert orphan["ppid"] == 222
    assert isinstance(orphan["host_ns"], int)


def test_mutator_route_table_matches_prd_inventory():
    assert MUTATOR_ROUTE_TABLE == {
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


def test_compute_host_compress_control_runs_identity_guard_in_host(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)

    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        context_compressor = None

    session = {
        "agent": _Agent(),
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }
    calls: dict[str, object] = {}

    def _compress(sess, focus_topic=None, **_kwargs):
        assert sess is session
        calls["compress_focus"] = focus_topic
        with sess["history_lock"]:
            sess["history"] = [{"role": "summary", "content": "compressed in host"}]
            sess["history_version"] = 3

    def _sync(sid, sess):
        assert sess is session
        calls["sync"] = sid
        sess["session_key"] = "after-key"

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _sync)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {
            "model": "host-model",
            "provider": "host-provider",
            "usage": {"total": 2},
        },
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-1",
                "route_name": "slash.compress",
                "command": "/compress focus",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == {"compress_focus": "focus", "sync": "sid"}
    assert ack["route_name"] == "slash.compress"
    assert ack["session_key"] == "after-key"
    assert ack["history_version"] == 3
    assert ack["message_count"] == 1
    assert ack["session_info"]["model"] == "host-model"


def test_compute_host_session_compress_returns_structured_result(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    session = {
        "agent": None,
        "session_key": "host-key",
        "history": [{"role": "user", "content": "preserved"}],
        "history_lock": threading.Lock(),
        "history_version": 3,
        "running": False,
    }
    calls: list[dict] = []

    def compress_handler(_rid, params):
        calls.append(params)
        return {
            "result": {
                "status": "aborted",
                "messages": [{"role": "user", "content": "preserved"}],
                "summary": {"aborted": True, "headline": "Compression aborted"},
            }
        }

    server._sessions["sid"] = session
    monkeypatch.setitem(server._methods, "session.compress", compress_handler)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {"model": "host-model"})

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-structured",
                "route_name": "session.compress",
                "command": "/compress auth",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "control.ack" and frame.get("request_id") == "compress-structured",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == [{"session_id": "sid", "focus_topic": "auth"}]
    assert ack["result"]["status"] == "aborted"
    assert ack["result"]["summary"]["aborted"] is True
    assert ack["session_key"] == "host-key"
    assert ack["history_version"] == 3
    assert ack["message_count"] == 1
    assert ack["session_info"] == {"model": "host-model"}


def test_append_log_record_single_write_lines(tmp_path):
    path = tmp_path / "agent.log"

    def writer(i: int) -> None:
        append_log_record(path, f"line-{i:03d}-" + ("x" * 2000))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert sorted(line.split("-", 2)[1] for line in lines) == [f"{i:03d}" for i in range(32)]
    assert all(line.endswith("x" * 2000) for line in lines)


def test_compute_host_windows_pid_probes_use_psutil_without_signaling(monkeypatch):
    from gateway import status as gateway_status
    from tui_gateway import host_supervisor

    process_calls: list[int] = []
    pid_exists_calls: list[int] = []

    class _Process:
        def __init__(self, pid):
            process_calls.append(pid)

        def cmdline(self):
            return [r"C:\Python311\python.exe", "-m", "tui_gateway.compute_host"]

    class _Psutil:
        Process = _Process

    monkeypatch.setattr(host_supervisor.sys, "platform", "win32")
    monkeypatch.setattr(
        gateway_status,
        "_pid_exists",
        lambda pid: pid_exists_calls.append(pid) or True,
    )
    monkeypatch.setitem(sys.modules, "psutil", _Psutil)
    monkeypatch.setattr(
        host_supervisor.os,
        "kill",
        lambda *_args: pytest.fail("Windows PID probes must not call os.kill"),
    )
    monkeypatch.setattr(
        host_supervisor.subprocess,
        "check_output",
        lambda *_args, **_kwargs: pytest.fail("Windows PID probes must not call ps"),
    )

    assert host_supervisor._pid_alive(4321) is True
    assert host_supervisor.is_compute_host_identity(4321) is True
    assert pid_exists_calls == [4321]
    assert process_calls == [4321]


def test_compute_host_windows_pid_probe_failure_is_fail_closed(monkeypatch):
    from gateway import status as gateway_status
    from tui_gateway import host_supervisor

    monkeypatch.setattr(host_supervisor.sys, "platform", "win32")
    monkeypatch.setattr(
        gateway_status,
        "_pid_exists",
        lambda _pid: (_ for _ in ()).throw(OSError("probe unavailable")),
    )
    monkeypatch.setattr(
        host_supervisor.os,
        "kill",
        lambda *_args: pytest.fail("Windows fallback must not call os.kill"),
    )

    assert host_supervisor._pid_alive(4321) is True


def test_compute_host_termination_preserves_process_identity(monkeypatch, tmp_path):
    from tui_gateway import host_supervisor

    events: list[object] = []

    class _TimeoutExpired(Exception):
        pass

    class _NoSuchProcess(Exception):
        pass

    class _Process:
        def __init__(self, pid):
            events.append(("process", pid))

        def create_time(self):
            return 12.34

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append(("wait", timeout))
            if events.count("terminate") and "kill" not in events:
                raise _TimeoutExpired()

        def kill(self):
            events.append("kill")

    class _Psutil:
        Process = _Process
        TimeoutExpired = _TimeoutExpired
        NoSuchProcess = _NoSuchProcess

    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setitem(sys.modules, "psutil", _Psutil)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 1234)

    assert supervisor._terminate_pid(
        4321,
        timeout=0,
        expected_start_time=1234,
    ) is True
    assert events == [
        ("process", 4321),
        "terminate",
        ("wait", 0),
        "kill",
        ("wait", 2),
    ]


def _write_host_registry(
    path: Path,
    *,
    process_start_time: int | None,
    host_pid: int | None = None,
    started_at: float = 100.0,
    owner_pid: int | None = None,
    owner_start_time: int | None = None,
    boot_id: str = "stale",
) -> None:
    path.write_text(
        json.dumps(
            {
                "host_pid": os.getpid() if host_pid is None else host_pid,
                "boot_id": boot_id,
                "process_start_time": process_start_time,
                "started_at": started_at,
                "owner_pid": owner_pid,
                "owner_start_time": owner_start_time,
            }
        ),
        encoding="utf-8",
    )


def test_supervisor_reconcile_preserves_live_owner(tmp_path, monkeypatch):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    owner_pid = os.getpid() + 100000
    _write_host_registry(
        registry,
        process_start_time=111,
        owner_pid=owner_pid,
        owner_start_time=777,
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)

    def _start_time(pid):
        if pid == owner_pid:
            return 777
        pytest.fail("live owner must short-circuit child reconciliation")

    monkeypatch.setattr(host_supervisor, "_pid_start_time", _start_time)

    assert supervisor.reconcile_startup_orphan() == "owned-by-live-supervisor"
    assert registry.exists()


def test_supervisor_owned_registry_removal_preserves_replacement(tmp_path):
    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(
        registry,
        host_pid=222,
        process_start_time=2222,
        boot_id="replacement",
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    owned_proc: Any = object()
    supervisor._owned_registry_identity = (111, 1111, "old")
    supervisor._owned_registry_proc = owned_proc

    supervisor._remove_registry_for_proc(owned_proc)

    assert json.loads(registry.read_text(encoding="utf-8"))["host_pid"] == 222
    assert supervisor._owned_registry_identity is None
    assert supervisor._owned_registry_proc is None


def test_supervisor_stale_waiter_cannot_clear_replacement_ownership(tmp_path):
    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(
        registry,
        host_pid=222,
        process_start_time=2222,
        boot_id="replacement",
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    stale_proc: Any = object()
    replacement_proc: Any = object()
    replacement_identity = (222, 2222, "replacement")
    supervisor._owned_registry_identity = replacement_identity
    supervisor._owned_registry_proc = replacement_proc

    supervisor._remove_registry_for_proc(stale_proc)

    assert registry.exists()
    assert supervisor._owned_registry_identity == replacement_identity
    assert supervisor._owned_registry_proc is replacement_proc


def test_supervisor_reaps_child_rejected_after_hello(tmp_path, monkeypatch):
    from tui_gateway import host_supervisor

    created: list[Any] = []
    real_popen = host_supervisor.subprocess.Popen

    def _popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(host_supervisor.subprocess, "Popen", _popen)
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        expected_build_sha="definitely-not-the-current-build",
        autostart=False,
    )

    with pytest.raises(RuntimeError, match="build mismatch"):
        supervisor.start()

    assert len(created) == 1
    assert created[0].poll() is not None
    assert supervisor._proc is None
    assert not supervisor.registry_path.exists()


def test_supervisor_startup_reconcile_pid_reuse_guard(tmp_path, monkeypatch):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(registry, process_start_time=111)
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 222)
    monkeypatch.setattr(
        host_supervisor,
        "_pid_command",
        lambda _pid: pytest.fail("start-time mismatch must short-circuit command probing"),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda *_args, **_kwargs: pytest.fail("reused PID must not be signaled"),
    )

    assert supervisor.reconcile_startup_orphan() == "pid-reuse-ignored"
    assert not registry.exists()


def test_supervisor_legacy_registry_clears_unrelated_reused_pid(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(registry, process_start_time=None)
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        host_supervisor,
        "_pid_command",
        lambda _pid: "python unrelated_service.py",
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda *_args, **_kwargs: pytest.fail("unrelated PID must not be signaled"),
    )

    assert supervisor.reconcile_startup_orphan() == "pid-reuse-ignored"
    assert not registry.exists()


def test_supervisor_legacy_registry_terminates_matching_process_instance(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(
        registry,
        process_start_time=None,
        started_at=100.0,
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    calls: list[int | None] = []
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 111)
    monkeypatch.setattr(host_supervisor, "_pid_create_time", lambda _pid: 99.0)
    monkeypatch.setattr(
        host_supervisor,
        "_pid_command",
        lambda _pid: "python -m tui_gateway.compute_host",
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda _pid, *, timeout, expected_start_time: calls.append(
            expected_start_time
        )
        or True,
    )

    assert supervisor.reconcile_startup_orphan() == "terminated"
    assert calls == [111]
    assert not registry.exists()


def test_supervisor_legacy_registry_rejects_process_created_after_registry(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(
        registry,
        process_start_time=None,
        started_at=100.0,
    )
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 111)
    monkeypatch.setattr(host_supervisor, "_pid_create_time", lambda _pid: 100.1)
    monkeypatch.setattr(
        host_supervisor,
        "_pid_command",
        lambda _pid: pytest.fail("reused PID must short-circuit command probing"),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda *_args, **_kwargs: pytest.fail("reused PID must not be signaled"),
    )

    assert supervisor.reconcile_startup_orphan() == "pid-reuse-ignored"
    assert not registry.exists()


def test_supervisor_startup_reconcile_fails_closed_when_identity_unknown(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(registry, process_start_time=111)
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 111)
    monkeypatch.setattr(host_supervisor, "_pid_command", lambda _pid: "")
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda *_args, **_kwargs: pytest.fail("unverified PID must not be signaled"),
    )

    assert supervisor.reconcile_startup_orphan() == "identity-unverified"
    assert registry.exists()
    monkeypatch.setattr(supervisor, "reconcile_startup_orphan", lambda: "identity-unverified")
    monkeypatch.setattr(
        supervisor,
        "_spawn_locked",
        lambda **_kwargs: pytest.fail("startup must not spawn beside an unverified host"),
    )
    with pytest.raises(RuntimeError, match="identity could not be verified"):
        supervisor.start()


def test_supervisor_startup_reconcile_terminates_matching_process_instance(
    tmp_path,
    monkeypatch,
):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    _write_host_registry(registry, process_start_time=111)
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    calls: list[tuple[int, float, int | None]] = []
    monkeypatch.setattr(host_supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 111)
    monkeypatch.setattr(
        host_supervisor,
        "_pid_command",
        lambda _pid: "python -m tui_gateway.compute_host",
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_pid",
        lambda pid, *, timeout, expected_start_time: calls.append(
            (pid, timeout, expected_start_time)
        )
        or True,
    )

    assert supervisor.reconcile_startup_orphan() == "terminated"
    assert calls == [(os.getpid(), 10.0, 111)]
    assert not registry.exists()


def test_supervisor_registry_persists_process_identity(tmp_path, monkeypatch):
    from tui_gateway import host_supervisor

    registry = tmp_path / "dashboard-compute-host.json"
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    proc: Any = type("_Proc", (), {"pid": 4321})()
    supervisor._proc = proc
    monkeypatch.setattr(host_supervisor, "_pid_start_time", lambda _pid: 111)

    supervisor._persist_registry()

    assert json.loads(registry.read_text(encoding="utf-8"))["process_start_time"] == 111


def test_supervisor_crash_releases_pending_control_waiter(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )

    class _ExitedProcess:
        def wait(self):
            return 7

    proc: Any = _ExitedProcess()
    supervisor._proc = proc
    frame_sent = threading.Event()
    result: dict[str, Any] = {}

    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda _frame: frame_sent.set())
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)

    def _wait_for_control() -> None:
        try:
            result["frame"] = supervisor.control(
                "sid",
                route_name="session.compress",
                timeout=1.0,
            )
        except Exception as exc:
            result["error"] = exc

    waiter = threading.Thread(target=_wait_for_control)
    waiter.start()
    assert frame_sent.wait(timeout=1)

    supervisor._wait_for_exit(proc)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert "error" not in result
    assert result["frame"]["type"] == "control.error"
    assert result["frame"]["request_id"]
    assert result["frame"]["reason"] == "crash"
    assert result["frame"]["message"] == "compute host exited with code 7"
    assert supervisor._pending_controls == {}


def test_supervisor_crash_does_not_fail_replacement_controls(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )

    class _ExitedProcess:
        def wait(self):
            return 7

    old_proc: Any = _ExitedProcess()
    new_proc: Any = object()
    supervisor._proc = old_proc
    sent_frames: list[dict] = []
    first_sent = threading.Event()
    second_sent = threading.Event()
    results: dict[str, Any] = {}

    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)

    def _send(frame: dict) -> None:
        sent_frames.append(frame)
        (first_sent if len(sent_frames) == 1 else second_sent).set()

    monkeypatch.setattr(supervisor, "_send_frame", _send)

    def _control(name: str) -> None:
        results[name] = supervisor.control(
            "sid",
            route_name="session.compress",
            timeout=1.0,
        )

    old_waiter = threading.Thread(target=_control, args=("old",))
    old_waiter.start()
    assert first_sent.wait(timeout=1)

    with supervisor._lock:
        supervisor._proc = new_proc
    new_waiter = threading.Thread(target=_control, args=("new",))
    new_waiter.start()
    assert second_sent.wait(timeout=1)

    supervisor._wait_for_exit(old_proc)
    old_waiter.join(timeout=1)
    assert results["old"]["type"] == "control.error"
    assert new_waiter.is_alive(), "old host crash failed a replacement-host control"

    supervisor._handle_host_frame(
        {
            "type": "control.ack",
            "request_id": sent_frames[1]["request_id"],
        },
        proc=new_proc,
    )
    new_waiter.join(timeout=1)

    assert not new_waiter.is_alive()
    assert results["new"]["type"] == "control.ack"
    assert supervisor._pending_controls == {}


def test_supervisor_drains_terminal_control_ack_before_crash_error(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )

    class _ExitedProcess:
        def wait(self):
            return 7

    proc: Any = _ExitedProcess()
    supervisor._proc = proc
    sent: dict[str, Any] = {}
    frame_sent = threading.Event()
    result: dict[str, Any] = {}

    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_remove_registry", lambda: None)
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)

    def _send(frame: dict) -> None:
        sent.update(frame)
        frame_sent.set()

    monkeypatch.setattr(supervisor, "_send_frame", _send)

    waiter = threading.Thread(
        target=lambda: result.update(
            response=supervisor.control(
                "sid",
                route_name="session.compress",
                timeout=1.0,
            )
        )
    )
    waiter.start()
    assert frame_sent.wait(timeout=1)

    class _StdoutDrain:
        def join(self, *args, **kwargs) -> None:
            result["join_call"] = (args, kwargs)
            supervisor._handle_host_frame(
                {"type": "control.ack", "request_id": sent["request_id"]},
                proc=proc,
            )

    stdout_thread: Any = _StdoutDrain()
    supervisor._wait_for_exit(proc, stdout_thread)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert result["join_call"] == ((), {})
    assert result["response"]["type"] == "control.ack"
    assert supervisor._pending_controls == {}


def test_supervisor_reload_mcp_raises_on_host_control_error(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(
        supervisor,
        "control",
        lambda *_args, **_kwargs: {
            "type": "control.error",
            "message": "compute host exited",
        },
    )

    with pytest.raises(RuntimeError, match="compute host exited"):
        supervisor.reload_mcp("sid")


def test_supervisor_reload_mcp_raises_on_nested_rpc_error(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "dashboard-compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    monkeypatch.setattr(
        supervisor,
        "control",
        lambda *_args, **_kwargs: {
            "type": "reload_mcp.ack",
            "response": {
                "jsonrpc": "2.0",
                "error": {"code": 5015, "message": "MCP discovery failed"},
            },
        },
    )

    with pytest.raises(RuntimeError, match="MCP discovery failed"):
        supervisor.reload_mcp("sid")


def test_supervisor_crash_emits_turn_error_and_respawns(tmp_path):
    script = tmp_path / "fake_host.py"
    script.write_text(
        """
import json, os, sys
print(json.dumps({'type':'hello','host_pid':os.getpid(),'boot_id':'boot-1','build_sha':'test','hermes_home':os.environ.get('HERMES_HOME','')}), flush=True)
for raw in sys.stdin:
    frame=json.loads(raw)
    if frame.get('type') == 'shutdown':
        print(json.dumps({'type':'shutdown.ack','request_id':frame.get('request_id')}), flush=True)
        break
    if frame.get('type') == 'turn.start':
        print(json.dumps({'type':'turn.started','sid':frame.get('sid'),'request_id':frame.get('request_id')}), flush=True)
        sys.stdout.flush()
        os._exit(7)
""".strip(),
        encoding="utf-8",
    )
    registry = tmp_path / "dashboard-compute-host.json"
    completions: list[dict] = []
    rpc_events: list[dict] = []
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, str(script)],
        rpc_sink=rpc_events.append,
        respawn_max=2,
        heartbeat_secs=1,
        expected_build_sha="test",
        autostart=False,
    )
    try:
        supervisor.start()
        supervisor.submit_turn(
            {"type": "turn.start", "sid": "sid-1", "request_id": "turn-1", "text": "hello"},
            on_complete=completions.append,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not completions:
            time.sleep(0.02)
        assert completions, "host crash did not complete pending turn"
        assert completions[0]["type"] == "turn.error"
        assert completions[0]["reason"] == "crash"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not supervisor.is_running():
            time.sleep(0.02)
        assert supervisor.is_running()
    finally:
        supervisor.shutdown()


def _make_compress_host_session(events: list) -> dict:
    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        session_id = "rotated-id"

    agent = _Agent()
    agent.context_compressor = type("ContextEngineStub", (), {})()
    agent.context_compressor.on_session_start = (
        lambda *_args, **_kwargs: events.append("notify")
    )
    return {
        "agent": agent,
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }


def test_compute_host_compress_control_notifies_engine_after_commit(monkeypatch):
    """The compute-host slash.compress route must fire the context-engine
    boundary hook exactly once, and only AFTER the host commits the compressed
    history + session-key sync (salvaged #65670, extended to this route)."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
        finalize_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)

    def _compress(sess, focus_topic=None, **_kwargs):
        # Simulate agent._compress_context(defer_context_engine_notification=True)
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )
        with sess["history_lock"]:
            sess["history"] = [{"role": "summary", "content": "compressed"}]
            sess["history_version"] = 3

    def _sync(sid, sess):
        events.append("sync")
        sess["session_key"] = "after-key"

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _sync)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-1",
                "route_name": "slash.compress",
                "command": "/compress",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    # Exactly one notification, after the session-key commit.
    assert events == ["sync", "notify"]
    assert ack["session_key"] == "after-key"
    # Nothing pending leaks onto the agent for a later compress to misfire.
    assert (
        finalize_context_engine_compression_notification(
            session["agent"], committed=True
        )
        is False
    )


def test_compute_host_compress_control_failure_discards_notification(monkeypatch):
    """When the host-side compress mirror fails after compression queued the
    boundary notification, the pending hook must be discarded — never left to
    fire against a boundary the host rejected."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
        finalize_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)

    def _compress(sess, focus_topic=None, **_kwargs):
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic host commit failure")

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _boom)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-2",
                "route_name": "slash.compress",
                "command": "/compress",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-2",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert events == []
    assert "live session sync failed" in str(ack.get("output") or "")
    # The pending notification was discarded, not left on the agent.
    assert (
        finalize_context_engine_compression_notification(
            session["agent"], committed=True
        )
        is False
    )
    assert events == []


def test_compute_host_compact_alias_routes_to_compress_mirror(monkeypatch):
    """slash.compress control frames forward the user's raw alias verbatim;
    /compact must reach the compress mirror (and its deferred-notification
    finalize wiring), not silently no-op."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)
    calls: dict[str, object] = {}

    def _compress(sess, focus_topic=None, **_kwargs):
        calls["focus"] = focus_topic
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a: events.append("sync")
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compact-1",
                "route_name": "slash.compress",
                "command": "/compact focus topic",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compact-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == {"focus": "focus topic"}
    assert events == ["sync", "notify"]
    assert ack["route_name"] == "slash.compress"
