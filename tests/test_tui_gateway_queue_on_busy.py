"""A prompt that lands mid-turn is redirected or queued, never dropped.

Before this, ``prompt.submit`` on a running session returned ``session busy``,
forcing clients into a deadline-bounded busy-retry. When turn teardown outlived
the deadline — e.g. a slow, non-interruptible tool (``web_search``) still
running when the user hit stop — the resubmitted message was silently dropped
("it just doesn't listen"). The gateway now applies the ``busy_input_mode``
policy: redirect the live turn by default, with the legacy interrupt + queue
path retained as a compatibility fallback.
"""

import threading
import time
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


# ── _enqueue_prompt ────────────────────────────────────────────────────────

def test_enqueue_pins_text_and_transport():
    session = _session()
    server._enqueue_prompt(session, "hello", "ws-1")
    assert session["queued_prompt"] == {"text": "hello", "transport": "ws-1"}


def test_enqueue_merges_second_arrival_losslessly():
    session = _session()
    server._enqueue_prompt(session, "first", "ws-1")
    server._enqueue_prompt(session, "second", "ws-2")
    assert session["queued_prompt"]["text"] == "first\n\nsecond"
    # Latest transport wins so the drain streams to the most recent client.
    assert session["queued_prompt"]["transport"] == "ws-2"


# ── _handle_busy_submit (policy) ───────────────────────────────────────────

def test_busy_interrupt_mode_redirects_active_turn(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("redirect must not hard-interrupt")
        ),
    )
    session = _session(agent=agent, running=True)
    session["inflight_turn"] = {"user": "original request", "assistant": "partial reply"}

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert seen == ["redirect"]
    assert session["inflight_turn"]["user"] == "redirect"
    assert session.get("queued_prompt") is None


def test_busy_interrupt_mode_falls_back_for_legacy_agent(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "redirect", "ws-1")

    assert resp["result"]["status"] == "queued"
    deadline = time.monotonic() + 1
    while calls["interrupt"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls["interrupt"] == 1
    assert session["queued_prompt"]["text"] == "redirect"


def test_busy_queue_mode_queues_without_interrupting(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1))
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "later", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert calls["interrupt"] == 0
    assert session["queued_prompt"]["text"] == "later"


def test_busy_steer_mode_injects_when_accepted(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: True, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "steered"
    assert session.get("queued_prompt") is None


def test_busy_steer_mode_falls_back_to_queue_when_rejected(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent = types.SimpleNamespace(steer=lambda text: False, interrupt=lambda *a, **k: None)
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, "nudge", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "nudge"


def test_busy_interrupt_does_not_hold_history_lock_or_delay_queue(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    interrupt_started = threading.Event()
    release_interrupt = threading.Event()

    def blocking_interrupt():
        interrupt_started.set()
        release_interrupt.wait(timeout=2)

    session = _session(
        agent=types.SimpleNamespace(interrupt=blocking_interrupt),
        running=True,
    )

    started = time.monotonic()
    resp = server._handle_busy_submit("r1", "sid", session, "keep this", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert time.monotonic() - started < 0.25
    assert session["queued_prompt"]["text"] == "keep this"
    assert interrupt_started.wait(timeout=1)
    assert session["history_lock"].acquire(timeout=0.25)
    session["history_lock"].release()
    release_interrupt.set()


def test_busy_helper_retries_when_turn_finished(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    session = _session(running=False)

    assert server._handle_busy_submit("r1", "sid", session, "run now", "ws-1") is None
    assert session.get("queued_prompt") is None


def test_busy_interrupt_mode_normalizes_rich_text_before_redirect(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)
    rich = [{"type": "text", "text": "  redirect me  "}]

    resp = server._handle_busy_submit(
        "r1",
        "sid",
        session,
        rich,
        "ws-1",
    )

    assert resp["result"]["status"] == "redirected"
    assert seen == ["redirect me"]
    assert session.get("queued_prompt") is None


def test_busy_queue_fallback_preserves_original_structured_text(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    rich = [{"type": "text", "text": "  keep me  "}]
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: False,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == rich


def test_busy_interrupt_mode_queues_multimodal_payload_instead_of_redirect(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    seen = []
    rich = [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
        interrupt=lambda *a, **k: None,
    )
    session = _session(agent=agent, running=True)

    resp = server._handle_busy_submit("r1", "sid", session, rich, "ws-1")

    assert resp["result"]["status"] == "queued"
    assert seen == []
    assert session["queued_prompt"]["text"] == rich


def test_prompt_submit_claim_is_atomic_across_concurrent_clients(monkeypatch):
    """Only one idle submit may claim the run; the racer is queued."""

    real_thread = threading.Thread
    first_release = threading.Barrier(2)

    class _GateLock:
        def __init__(self):
            self._lock = threading.Lock()
            self._seen: set[int] = set()
            self._seen_lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            ident = threading.get_ident()
            with self._seen_lock:
                first = ident not in self._seen
                self._seen.add(ident)
            self._lock.release()
            if first:
                first_release.wait(timeout=2)

    class _NoopThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            return None

    session = _session(history_lock=_GateLock())
    server._sessions["sid"] = session
    responses = {}

    def _submit(rid, text):
        responses[rid] = server.handle_request(
            {
                "id": rid,
                "method": "prompt.submit",
                "params": {"session_id": "sid", "text": text},
            }
        )

    callers = [
        real_thread(target=_submit, args=("one", "first")),
        real_thread(target=_submit, args=("two", "second")),
    ]
    try:
        monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")
        monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
        monkeypatch.setattr(server.threading, "Thread", _NoopThread)

        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=2)

        assert all(not caller.is_alive() for caller in callers)
        statuses = sorted(resp["result"]["status"] for resp in responses.values())
        assert statuses == ["queued", "streaming"]
        assert session["run_generation"] == 1
        assert session["queued_prompt"]["text"] in {"first", "second"}
        assert session["inflight_turn"]["user"] in {"first", "second"}
        assert session["queued_prompt"]["text"] != session["inflight_turn"]["user"]
    finally:
        server._sessions.pop("sid", None)
# ── _drain_queued_prompt ───────────────────────────────────────────────────

def test_drain_fires_queued_prompt_and_claims_running(monkeypatch):
    fired = {}
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda rid, sid, session, text: fired.update(rid=rid, sid=sid, text=text),
    )
    session = _session(queued_prompt={"text": "go", "transport": "ws-9"})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    assert fired == {"rid": "r1", "sid": "sid", "text": "go"}
    assert session["running"] is True
    assert session["queued_prompt"] is None
    assert session["transport"] == "ws-9"


def test_drain_noop_when_nothing_queued(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session()
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["running"] is False


def test_drain_noop_when_session_already_running(monkeypatch):
    """A fresh turn that claimed the session beats a stale queued entry —
    the drain leaves it for that turn's own tail."""
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fire")))
    session = _session(running=True, queued_prompt={"text": "go", "transport": None})
    assert server._drain_queued_prompt("r1", "sid", session) is False
    assert session["queued_prompt"]["text"] == "go"


def test_drain_releases_running_on_dispatch_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("dispatch failed")
    monkeypatch.setattr(server, "_run_prompt_submit", _boom)
    session = _session(queued_prompt={"text": "go", "transport": None})

    assert server._drain_queued_prompt("r1", "sid", session) is True
    # Failure must not leave the session wedged as running.
    assert session["running"] is False
