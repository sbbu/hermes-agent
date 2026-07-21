"""Regression tests for one canonical Codex auth owner across profiles."""

import json
import threading

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from hermes_cli import auth as A

PROVIDER = "openai-codex"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "worker" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return profile_path, root_path


def _entry(access="access-0", refresh="refresh-0", source="manual:device_code"):
    return PooledCredential(
        provider=PROVIDER,
        id="codex-shared",
        label="shared",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access,
        refresh_token=refresh,
    )


def test_profile_pool_reads_and_writes_global_owner(profile_and_root):
    profile_path, root_path = profile_and_root
    root_entry = _entry().to_dict()
    stale_profile_entry = {
        **root_entry,
        "access_token": "stale-profile-access",
        "refresh_token": "consumed-profile-refresh",
    }
    _write(root_path, {"version": 1, "credential_pool": {PROVIDER: [root_entry]}})
    _write(profile_path, {"version": 1, "credential_pool": {PROVIDER: [stale_profile_entry]}})

    assert A.read_credential_pool(PROVIDER) == [root_entry]
    assert A.read_credential_pool(None)[PROVIDER] == [root_entry]

    fresh = {**root_entry, "access_token": "fresh-access", "refresh_token": "fresh-refresh"}
    assert A.write_credential_pool(PROVIDER, [fresh]) == root_path
    assert A.read_credential_pool(PROVIDER) == [fresh]
    assert _read(profile_path)["credential_pool"][PROVIDER] == [stale_profile_entry]


def test_profile_singleton_reads_and_writes_global_owner(profile_and_root):
    profile_path, root_path = profile_and_root
    _write(root_path, {"version": 1, "providers": {PROVIDER: {"tokens": {"access_token": "root-access", "refresh_token": "root-refresh"}}}})
    _write(profile_path, {"version": 1, "providers": {PROVIDER: {"tokens": {"access_token": "stale-access", "refresh_token": "consumed-refresh"}}}})

    state = A.get_provider_auth_state(PROVIDER)
    assert state is not None
    assert state["tokens"]["refresh_token"] == "root-refresh"

    A._save_codex_tokens(
        {"access_token": "fresh-access", "refresh_token": "fresh-refresh"},
        last_refresh="2026-01-01T00:00:00Z",
    )
    assert _read(root_path)["providers"][PROVIDER]["tokens"]["refresh_token"] == "fresh-refresh"
    assert _read(profile_path)["providers"][PROVIDER]["tokens"]["refresh_token"] == "consumed-refresh"


def test_device_code_sync_targets_global_owner(profile_and_root):
    profile_path, root_path = profile_and_root
    _write(root_path, {"version": 1, "providers": {PROVIDER: {"tokens": {"access_token": "root-old", "refresh_token": "root-old-refresh"}}}})
    _write(profile_path, {"version": 1, "providers": {PROVIDER: {"tokens": {"access_token": "stale", "refresh_token": "consumed-refresh"}}}})

    CredentialPool(PROVIDER, [])._sync_device_code_entry_to_auth_store(
        _entry("root-new", "root-new-refresh", source="device_code")
    )
    assert _read(root_path)["providers"][PROVIDER]["tokens"]["refresh_token"] == "root-new-refresh"
    assert _read(profile_path)["providers"][PROVIDER]["tokens"]["refresh_token"] == "consumed-refresh"


def test_profiles_refresh_one_canonical_manual_chain(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    entry = _entry()
    _write(root_path, {"version": 1, "credential_pool": {PROVIDER: [entry.to_dict()]}})
    _write(profile_path, {"version": 1, "credential_pool": {PROVIDER: [{**entry.to_dict(), "access_token": "stale", "refresh_token": "consumed"}]}})

    current = {"refresh": "refresh-0"}
    calls = []
    guard = threading.Lock()

    def fake_refresh(access_token, refresh_token, **kwargs):
        del access_token, kwargs
        with guard:
            assert refresh_token == current["refresh"]
            calls.append(refresh_token)
            generation = len(calls)
            current["refresh"] = f"refresh-{generation}"
            return {
                "access_token": f"access-{generation}",
                "refresh_token": current["refresh"],
                "last_refresh": f"2026-01-01T00:00:0{generation}Z",
            }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    pools = [CredentialPool(PROVIDER, [entry]), CredentialPool(PROVIDER, [entry])]
    barrier = threading.Barrier(3)
    results = [None, None]

    def refresh(index):
        barrier.wait()
        results[index] = pools[index]._refresh_entry(entry, force=True)

    threads = [threading.Thread(target=refresh, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert all(result is not None for result in results)
    assert calls == ["refresh-0", "refresh-1"]
    assert _read(root_path)["credential_pool"][PROVIDER][0]["refresh_token"] == "refresh-2"
    assert _read(profile_path)["credential_pool"][PROVIDER][0]["refresh_token"] == "consumed"


def test_pool_only_runtime_fallbacks_read_global_owner(profile_and_root):
    profile_path, root_path = profile_and_root
    root_entry = {
        **_entry("root-access", "root-refresh").to_dict(),
        "last_status": "exhausted",
        "last_error_code": 429,
        "last_error_reason": "usage_limit",
        "last_error_message": "quota exhausted",
        "last_error_reset_at": 4_102_444_800,
    }
    _write(root_path, {"version": 1, "credential_pool": {PROVIDER: [root_entry]}})
    _write(
        profile_path,
        {
            "version": 1,
            "credential_pool": {
                PROVIDER: [
                    {
                        **root_entry,
                        "access_token": "stale-profile-access",
                        "last_status": None,
                    }
                ]
            },
        },
    )

    status = A._codex_pool_rate_limit_status()
    assert status is not None
    assert status["reason"] == "usage_limit"

    root_entry["last_status"] = None
    root_entry["last_error_reset_at"] = None
    _write(root_path, {"version": 1, "credential_pool": {PROVIDER: [root_entry]}})
    assert A._pool_codex_access_token() == "root-access"


def test_suppression_and_logout_target_global_owner(profile_and_root):
    profile_path, root_path = profile_and_root
    _write(root_path, {
        "version": 1,
        "providers": {PROVIDER: {"tokens": {"access_token": "a", "refresh_token": "r"}}},
        "credential_pool": {PROVIDER: [_entry().to_dict()]},
    })
    _write(profile_path, {"version": 1, "providers": {}, "active_provider": PROVIDER})

    A.suppress_credential_source(PROVIDER, "device_code")
    assert A.is_source_suppressed(PROVIDER, "device_code") is True
    assert _read(root_path)["suppressed_sources"][PROVIDER] == ["device_code"]
    assert A.unsuppress_credential_source(PROVIDER, "device_code") is True

    assert A.clear_provider_auth(PROVIDER) is True
    root = _read(root_path)
    assert PROVIDER not in root["providers"]
    assert PROVIDER not in root["credential_pool"]
    assert _read(profile_path)["active_provider"] is None
