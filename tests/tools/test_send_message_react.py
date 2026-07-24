"""Tests for send_message action='react'/'unreact' dispatch.

Kept separate from ``test_send_message_tool.py`` because that module skips
wholesale when optional Telegram dependencies are not installed.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import tools.send_message_tool as smt


class _FakePhotonAdapter:
    """Adapter exposing add_reaction/remove_reaction coroutines."""

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("add", chat_id, emoji, message_id))
        return {"success": True, "emoji": emoji}

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("remove", chat_id, message_id))
        return {"success": True}


class _NoReactionAdapter:
    """Adapter with no reaction support at all."""


def _runner_with(adapter):
    from gateway.config import Platform

    return SimpleNamespace(adapters={Platform("photon"): adapter})


def _call(args):
    return json.loads(smt.send_message_tool(args))


def test_react_dispatches_to_add_reaction():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "❤️"}
        )
    assert result["success"] is True
    assert adapter.calls == [("add", "+15551234567", "❤️", None)]


def test_unreact_dispatches_to_remove_reaction():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {
                "action": "unreact",
                "target": "photon:+15551234567",
                "message_id": "msg-9",
            }
        )
    assert result["success"] is True
    assert adapter.calls == [("remove", "+15551234567", "msg-9")]


def test_reaction_directory_thread_target_uses_parent_chat_id(tmp_path):
    adapter = _FakePhotonAdapter()
    cache_file = tmp_path / "channel_directory.json"
    cache_file.write_text(json.dumps({
        "updated_at": "2026-01-01T00:00:00",
        "platforms": {
            "photon": [
                {
                    "id": "space-a:thread-b",
                    "name": "Family / topic thread-b",
                    "type": "group",
                    "thread_id": "thread-b",
                }
            ]
        },
    }))

    with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), patch(
        "gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)
    ):
        friendly_result = _call(
            {
                "action": "react",
                "target": "photon:name=Family / topic thread-b (group)",
                "emoji": "👍",
            }
        )
        raw_result = _call(
            {
                "action": "react",
                "target": "photon:space-a:thread-b",
                "emoji": "✅",
            }
        )

    assert friendly_result["success"] is True
    assert raw_result["success"] is True
    assert adapter.calls == [
        ("add", "space-a", "👍", None),
        ("add", "space-a", "✅", None),
    ]


def test_react_requires_emoji():
    result = _call({"action": "react", "target": "photon:+15551234567"})
    assert result.get("success") is not True
    assert "emoji" in json.dumps(result)


def test_unreact_does_not_require_emoji():
    adapter = _FakePhotonAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call({"action": "unreact", "target": "photon:+15551234567"})
    assert result["success"] is True
    assert adapter.calls == [("remove", "+15551234567", None)]


def test_react_unsupported_platform_adapter():
    adapter = _NoReactionAdapter()
    with patch("gateway.run._gateway_runner_ref", lambda: _runner_with(adapter)):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "does not support" in json.dumps(result)


def test_react_without_live_gateway():
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        result = _call(
            {"action": "react", "target": "photon:+15551234567", "emoji": "👍"}
        )
    assert result.get("success") is not True
    assert "live" in json.dumps(result)
