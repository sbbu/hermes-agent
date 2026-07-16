"""Security regression tests for the TUI local shell RPC."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_shell_exec")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        yield importlib.import_module("tui_gateway.server")


def _safe_command_checks():
    return patch.multiple(
        "tools.approval",
        detect_hardline_command=MagicMock(return_value=(False, "")),
        detect_dangerous_command=MagicMock(return_value=(False, None, "")),
    )


def test_shell_exec_sanitizes_inherited_provider_credentials(server):
    secret = "sk-provider-secret-value"
    completed = SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": secret}),
        _safe_command_checks(),
        patch.object(server.subprocess, "run", return_value=completed) as run,
    ):
        response = server._methods["shell.exec"]("request-1", {"command": "printf ok"})

    assert response["result"]["stdout"] == "ok\n"
    child_env = run.call_args.kwargs["env"]
    assert "OPENROUTER_API_KEY" not in child_env
    assert secret not in child_env.values()


def test_shell_exec_redacts_secrets_printed_by_command(server, monkeypatch):
    secret = "sk-output-secret-value"
    completed = SimpleNamespace(
        stdout=f"token={secret}\n",
        stderr=f"failed with {secret}\n",
        returncode=1,
    )
    from agent import redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    with (
        _safe_command_checks(),
        patch.object(server.subprocess, "run", return_value=completed),
    ):
        response = server._methods["shell.exec"]("request-2", {"command": "print-token"})

    assert secret not in response["result"]["stdout"]
    assert secret not in response["result"]["stderr"]
    assert response["result"]["stdout"] != completed.stdout


def test_shell_exec_recovers_when_process_cwd_was_deleted(server, monkeypatch, tmp_path):
    completed = SimpleNamespace(stdout="ok\n", stderr="", returncode=0)
    deleted_cwd_path = tmp_path / "deleted" / "workspace"

    def deleted_cwd():
        raise FileNotFoundError("process cwd was deleted")

    monkeypatch.setattr(server.os, "getcwd", deleted_cwd)
    monkeypatch.setenv("TERMINAL_CWD", str(deleted_cwd_path))
    with (
        _safe_command_checks(),
        patch.object(server.subprocess, "run", return_value=completed) as run,
    ):
        response = server._methods["shell.exec"]("request-3", {"command": "printf ok"})

    assert response["result"]["stdout"] == "ok\n"
    assert run.call_args.kwargs["cwd"] == str(tmp_path)


def test_default_session_cwd_recovers_deleted_local_terminal_cwd(server, monkeypatch, tmp_path):
    deleted_cwd_path = tmp_path / "deleted" / "workspace"

    def deleted_cwd():
        raise FileNotFoundError("process cwd was deleted")

    monkeypatch.setattr(server.os, "getcwd", deleted_cwd)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(deleted_cwd_path))
    with patch.object(server, "_launch_configured_cwd", return_value=None):
        assert server._default_session_cwd() == str(tmp_path)


def test_default_session_cwd_preserves_remote_terminal_cwd(server, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "/remote-only/workspace")
    with patch.object(server, "_launch_configured_cwd", return_value=None):
        assert server._default_session_cwd() == "/remote-only/workspace"
