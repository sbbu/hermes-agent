import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tui_gateway import compute_host, server
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


def test_force_release_invalidates_turn_queued_before_child_registration(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    ensure_calls = []

    def _block_worker():
        blocker_started.set()
        release_blocker.wait(timeout=2)

    host._executor.submit(_block_worker)
    assert blocker_started.wait(timeout=1)
    monkeypatch.setattr(
        host,
        "_ensure_server_session",
        lambda *_args: ensure_calls.append(True),
    )
    server._sessions.pop("queued-real-sid", None)
    try:
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "queued-real-sid",
                "request_id": "queued-turn",
            }
        )
        host.handle_frame(
            {
                "type": "force_release",
                "sid": "queued-real-sid",
                "request_id": "release-queued",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda frame: frame.get("request_id") == "release-queued",
        )
        release_blocker.set()
        error = _wait_for_frame(
            out,
            lambda frame: frame.get("request_id") == "queued-turn",
        )
    finally:
        release_blocker.set()
        server._sessions.pop("queued-real-sid", None)
        host.close()

    assert ack["applied"] is True
    assert ack["reason"] == "pending_turn_cancelled"
    assert error["reason"] == "force_released_before_start"
    assert ensure_calls == []
    assert "queued-real-sid" in host._force_release_bypass


def test_real_turn_exception_does_not_clear_replacement_session(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    old_session = {
        "agent": object(),
        "session_key": "old-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "inflight_turn": None,
    }
    replacement = {
        "running": True,
        "inflight_turn": {"user": "replacement"},
    }
    sid = "reused-real-sid"
    host._real_turn_epochs[sid] = 1
    server._sessions[sid] = old_session
    monkeypatch.setattr(host, "_ensure_server_session", lambda *_args: old_session)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)

    def _raise_after_replacement(*_args, **_kwargs):
        server._sessions[sid] = replacement
        raise RuntimeError("old worker failed late")

    monkeypatch.setattr(server, "_run_prompt_submit", _raise_after_replacement)
    try:
        host._run_real_turn(
            {
                "type": "turn.start",
                "sid": sid,
                "request_id": "old-turn",
                "session_key": "old-key",
                "_host_turn_epoch": 1,
            }
        )
    finally:
        server._sessions.pop(sid, None)
        host.close()

    assert replacement["running"] is True
    assert replacement["inflight_turn"] == {"user": "replacement"}


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


def test_supervisor_turn_correlation_is_unique_across_client_request_ids(
    tmp_path, monkeypatch
):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "compute-host.json",
        argv=[sys.executable, "-c", ""],
        autostart=False,
    )
    sent = []
    callbacks = []
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(supervisor, "_send_frame", lambda frame: sent.append(dict(frame)))

    first_id = supervisor.submit_turn(
        {"sid": "session-a", "request_id": "1"},
        on_complete=lambda frame: callbacks.append(("a", frame["sid"])),
    )
    second_id = supervisor.submit_turn(
        {"sid": "session-b", "request_id": "1"},
        on_complete=lambda frame: callbacks.append(("b", frame["sid"])),
    )

    assert first_id != second_id
    assert sent[0]["client_request_id"] == "1"
    assert sent[1]["client_request_id"] == "1"
    supervisor._complete_turn(
        {"type": "turn.end", "sid": "session-a", "request_id": first_id},
    )
    supervisor._complete_turn(
        {"type": "turn.end", "sid": "session-b", "request_id": second_id},
    )

    assert callbacks == [("a", "session-a"), ("b", "session-b")]
    assert supervisor._pending_turns == {}


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


def _record_finalize(monkeypatch, events: list[str], *sids: str) -> None:
    """Give ``flush_all_sessions`` sessions and record which ones finalize."""
    keys = sids or ("s1",)
    monkeypatch.setattr(
        server,
        "_sessions",
        {sid: {"session_key": sid} for sid in keys},
        raising=False,
    )
    monkeypatch.setattr(
        server,
        "_finalize_session",
        lambda _session, end_reason="tui_close": events.append(
            f"finalize:{_session['session_key']}:{end_reason}"
        ),
        raising=False,
    )


def _register_turn(host: ComputeHost, fn, sid: str = "s1") -> None:
    """Submit a turn exactly the way ``_handle_turn_start`` does."""
    host._track_turn_future(host._executor.submit(fn), sid)


def test_shutdown_drains_in_flight_turn_before_finalizing_sessions(monkeypatch):
    events: list[str] = []
    _record_finalize(monkeypatch, events)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    running = threading.Event()

    def _turn() -> None:
        running.set()
        time.sleep(0.3)
        events.append("turn_end")

    _register_turn(host, _turn, sid="s1")
    assert running.wait(timeout=5.0)

    host.shutdown(reason="sigterm", wait=3.0)

    # ``_finalize_session`` latches on ``session["_finalized"]``, so its single
    # run has to observe the finished turn or the tail is unpersistable. A turn
    # that *did* drain must still finalize — the live-turn skip must not
    # over-reach into sessions whose work is done.
    assert events == ["turn_end", "finalize:s1:compute_host_sigterm"]

    # The done-callback still has to remove the entry now that the container is
    # a dict: ``set.discard`` was a valid bare callback, ``dict.pop`` is not.
    deadline = time.monotonic() + 2.0
    while host._turn_futures and time.monotonic() < deadline:
        time.sleep(0.01)
    assert host._turn_futures == {}, "in-flight turns must not accumulate"


def test_shutdown_retains_a_live_turns_session_when_the_drain_deadline_expires(monkeypatch):
    wait = 1.0
    events: list[str] = []
    _record_finalize(monkeypatch, events, "live", "idle")

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        started = time.monotonic()
        host.shutdown(reason="sigterm", wait=wait)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    # ``_finalize_session`` is one-shot, and the ``shutdown(wait=False)`` that
    # follows does not join the turn. Spending "live"'s single latch mid-turn
    # would leave it permanently un-finalizable and release its active-session
    # lease out from under running work — the same lifecycle race the drain
    # exists to close, just moved past the deadline. It is retained unfinalized
    # for recovery instead. A turn outliving the window must not cost the flush
    # for anyone else, so "idle" still finalizes in the same pass.
    assert events == ["finalize:idle:compute_host_sigterm"]
    assert elapsed < wait


def test_shutdown_retains_live_sessions_within_the_stdin_closed_budget(monkeypatch):
    """The tightest real budget any caller uses is ``wait=2.0``.

    ``run_host`` finalizes through ``host.shutdown(reason="stdin_closed",
    wait=2.0)``, which is where the reserve — ``wait`` minus
    ``min(_FLUSH_RESERVE_SECS, wait / 2)`` — has the least room to work with.
    The retain-live-sessions rule must hold there without costing the flush for
    idle sessions and without pushing the call past the budget the supervisor's
    kill escalation is timed against.
    """
    wait = 2.0
    drain_budget = wait - min(compute_host._FLUSH_RESERVE_SECS, wait / 2.0)

    events: list[str] = []
    _record_finalize(monkeypatch, events, "live", "idle")

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        started = time.monotonic()
        host.shutdown(reason="stdin_closed", wait=wait)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert events == ["finalize:idle:compute_host_stdin_closed"]
    assert elapsed >= drain_budget - 1e-6, "the drain must use its full window"
    assert elapsed < wait


def test_shutdown_drain_sleep_never_overshoots_the_reserve(monkeypatch):
    """The drain's per-tick sleep must be bounded by the time left to it.

    A flat tick overshoots the drain deadline by up to one tick, eating the
    reserve held back for ``flush_all_sessions``; for a small ``wait`` that is
    the whole reserve. Asserting on the *requested* sleep totals rather than on
    wall-clock keeps this deterministic: each sleep is clamped to the remaining
    time, so the sum can never exceed the drain budget however the scheduler
    interleaves.
    """
    wait = 0.34
    drain_budget = wait - min(compute_host._FLUSH_RESERVE_SECS, wait / 2.0)

    events: list[str] = []
    _record_finalize(monkeypatch, events, "idle")

    slept: list[float] = []
    real_sleep = time.sleep

    def _recording_sleep(seconds: float) -> None:
        slept.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(compute_host.time, "sleep", _recording_sleep)

    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    release = threading.Event()
    running = threading.Event()

    def _stuck_turn() -> None:
        running.set()
        release.wait(timeout=30.0)

    _register_turn(host, _stuck_turn, sid="live")
    assert running.wait(timeout=5.0)

    try:
        host.shutdown(reason="sigterm", wait=wait)
    finally:
        release.set()

    assert events == ["finalize:idle:compute_host_sigterm"]
    assert slept, "the drain loop should have ticked at least once"
    assert sum(slept) <= drain_budget + 1e-6
