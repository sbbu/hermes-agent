"""Regression tests for one canonical Codex auth owner across profiles."""

import base64
import json
import threading
from contextlib import contextmanager

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from agent.credential_sources import _clear_auth_store_provider, find_removal_step
from hermes_cli import auth as A

PROVIDER = "openai-codex"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _jwt(account_id: str) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": account_id}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"e30.{payload}.signature"


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


def test_legacy_profile_auth_is_adopted_when_global_owner_is_empty(profile_and_root):
    profile_path, root_path = profile_and_root
    entry = _entry().to_dict()
    singleton = {
        "tokens": {
            "access_token": "legacy-access",
            "refresh_token": "legacy-refresh",
        }
    }
    _write(root_path, {"version": 1, "providers": {"nous": {"access_token": "keep"}}})
    _write(
        profile_path,
        {
            "version": 1,
            "active_provider": PROVIDER,
            "providers": {PROVIDER: singleton},
            "credential_pool": {PROVIDER: [entry]},
            "suppressed_sources": {PROVIDER: ["env:OPENAI_API_KEY"]},
        },
    )

    assert A.get_provider_auth_state(PROVIDER) == singleton
    assert A.read_credential_pool(PROVIDER) == [entry]

    root = _read(root_path)
    profile = _read(profile_path)
    assert root["providers"]["nous"] == {"access_token": "keep"}
    assert root["providers"][PROVIDER] == singleton
    assert root["credential_pool"][PROVIDER] == [entry]
    assert root["suppressed_sources"][PROVIDER] == ["env:OPENAI_API_KEY"]
    assert PROVIDER not in profile["providers"]
    assert PROVIDER not in profile["credential_pool"]
    assert PROVIDER not in profile["suppressed_sources"]
    assert profile["active_provider"] == PROVIDER


def test_empty_root_error_state_does_not_block_valid_profile_adoption(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    entry = _entry(source="device_code")
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {},
                    "last_auth_error": "expired refresh token",
                }
            },
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [entry.to_dict()]},
        },
    )
    live_empty = CredentialPool(PROVIDER, [], owner_generation=0)

    state = A.get_provider_auth_state(PROVIDER)

    assert state is not None
    assert state["tokens"]["access_token"] == entry.access_token
    assert _read(root_path)["shared_auth_owners"][PROVIDER]["generation"] == 1
    assert [credential.id for credential in live_empty.entries()] == [entry.id]


def test_credentialless_profile_cannot_claim_owner_before_valid_sibling(
    profile_and_root,
    monkeypatch,
):
    empty_profile_path, root_path = profile_and_root
    valid_profile_path = empty_profile_path.parent.parent / "valid" / "auth.json"
    entry = _entry(source="device_code")
    _write(root_path, {"version": 1, "providers": {}})
    _write(
        empty_profile_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {"tokens": {}, "last_auth_error": "expired"}
            },
            "credential_pool": {
                PROVIDER: [
                    {
                        "id": "metadata-only",
                        "label": "expired",
                        "auth_type": "oauth",
                        "source": "device_code",
                        "last_status": "dead",
                    }
                ]
            },
        },
    )
    _write(
        valid_profile_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [entry.to_dict()]},
        },
    )

    assert A.get_provider_auth_state(PROVIDER) is None
    assert PROVIDER not in _read(root_path).get("shared_auth_owners", {})

    monkeypatch.setattr(A, "_auth_file_path", lambda: valid_profile_path)
    state = A.get_provider_auth_state(PROVIDER)
    assert state is not None
    assert state["tokens"]["access_token"] == entry.access_token
    assert _read(root_path)["shared_auth_owners"][PROVIDER]["generation"] == 1


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
    A.read_credential_pool(PROVIDER)
    assert _read(profile_path)["credential_pool"][PROVIDER][0]["refresh_token"] == (
        "consumed"
    )


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
        "active_provider": PROVIDER,
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
    assert root["active_provider"] == PROVIDER
    assert _read(profile_path)["active_provider"] is None


def test_logout_never_nests_profile_and_global_store_locks(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    _write(
        root_path,
        {
            "version": 1,
            "providers": {PROVIDER: {"tokens": {"access_token": "a"}}},
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "active_provider": PROVIDER,
            "providers": {PROVIDER: {"tokens": {"access_token": "stale"}}},
        },
    )

    real_file_lock = A._file_lock
    depth = 0
    max_depth = 0

    @contextmanager
    def recording_file_lock(lock_path, holder, timeout_seconds, timeout_message):
        nonlocal depth, max_depth
        with real_file_lock(lock_path, holder, timeout_seconds, timeout_message):
            depth += 1
            max_depth = max(max_depth, depth)
            try:
                yield
            finally:
                depth -= 1

    monkeypatch.setattr(A, "_file_lock", recording_file_lock)

    assert A.clear_provider_auth(PROVIDER) is True
    assert max_depth == 1
    assert PROVIDER not in _read(profile_path)["providers"]
    assert PROVIDER not in _read(root_path)["providers"]


def test_logout_tombstone_blocks_stale_sibling_profile_re_adoption(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    sibling_home = profile_path.parent.parent / "other"
    sibling_path = sibling_home / "auth.json"
    _write(
        root_path,
        {
            "version": 1,
            "providers": {PROVIDER: {"tokens": {"access_token": "canonical"}}},
        },
    )
    _write(
        profile_path,
        {"version": 1, "active_provider": PROVIDER},
    )
    _write(
        sibling_path,
        {
            "version": 1,
            "active_provider": PROVIDER,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": "stale-sibling",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
            "credential_pool": {
                PROVIDER: [
                    {
                        **_entry().to_dict(),
                        "id": "stale-manual",
                        "access_token": "stale-manual-access",
                        "refresh_token": "stale-manual-refresh",
                    }
                ]
            },
        },
    )

    assert A.clear_provider_auth(PROVIDER) is True
    root = _read(root_path)
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 1
    assert marker["deleted"] is True
    assert marker["legacy_migration_closed"] is True
    assert PROVIDER not in root["providers"]

    monkeypatch.setenv("HERMES_HOME", str(sibling_home))
    assert A.get_provider_auth_state(PROVIDER) is None
    assert A.read_credential_pool(PROVIDER) == []
    assert PROVIDER not in _read(root_path)["providers"]
    assert PROVIDER not in _read(root_path).get("credential_pool", {})
    assert _read(sibling_path)["providers"][PROVIDER]["tokens"]["access_token"] == (
        "stale-sibling"
    )

    A._save_codex_tokens(
        {"access_token": "fresh-after-logout", "refresh_token": "fresh-refresh"},
    )
    state = A.get_provider_auth_state(PROVIDER)
    assert state is not None
    assert state["tokens"]["access_token"] == "fresh-after-logout"
    fresh_pool = A.read_credential_pool(PROVIDER)
    assert len(fresh_pool) == 1
    assert fresh_pool[0]["access_token"] == "fresh-after-logout"
    assert "stale-manual" in {
        entry["id"] for entry in _read(sibling_path)["credential_pool"][PROVIDER]
    }


def test_tombstoned_owner_hides_rows_from_every_read_path(profile_and_root):
    profile_path, root_path = profile_and_root
    stale_entry = _entry().to_dict()
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
            "credential_pool": {PROVIDER: [stale_entry]},
            "shared_auth_owners": {
                PROVIDER: {
                    "generation": 4,
                    "deleted": True,
                    "legacy_migration_closed": True,
                }
            },
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})

    assert A.get_provider_auth_state(PROVIDER) is None
    assert A.read_credential_pool(PROVIDER) == []
    assert PROVIDER not in A.read_credential_pool()
    assert A._pool_codex_access_token() == ""
    assert A._codex_pool_rate_limit_status() is None

    A.suppress_credential_source(PROVIDER, "device_code")
    marker = _read(root_path)["shared_auth_owners"][PROVIDER]
    assert marker["deleted"] is True
    assert marker["legacy_migration_closed"] is True


def test_logout_generation_fences_live_stale_pool_writer(profile_and_root):
    profile_path, root_path = profile_and_root
    entry = _entry()
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [entry.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    live_pool = CredentialPool(PROVIDER, [entry], owner_generation=0)
    live_pool._current_id = entry.id

    assert A.clear_provider_auth(PROVIDER) is True
    assert live_pool.current() is None
    assert A.write_credential_pool(PROVIDER, [entry.to_dict()]) is None
    assert live_pool.select() is None
    assert live_pool._persist() is False
    assert live_pool.has_credentials() is False

    root = _read(root_path)
    assert PROVIDER not in root["providers"]
    assert PROVIDER not in root["credential_pool"]
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 1
    assert marker["deleted"] is True
    assert marker["legacy_migration_closed"] is True


def test_tombstoned_pool_allows_explicit_add_reactivation(profile_and_root):
    profile_path, root_path = profile_and_root
    old_entry = _entry(source="device_code")
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": old_entry.access_token,
                        "refresh_token": old_entry.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [old_entry.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    pool = CredentialPool(PROVIDER, [old_entry], owner_generation=0)
    assert A.clear_provider_auth(PROVIDER) is True
    old_writer = _read(root_path)
    old_writer.setdefault("providers", {})[PROVIDER] = {
        "tokens": {
            "access_token": "revived-stale",
            "refresh_token": "revived-stale-refresh",
        }
    }
    _write(root_path, old_writer)
    fresh_entry = PooledCredential(
        provider=PROVIDER,
        id="fresh-add",
        label="fresh",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token="fresh-access",
        refresh_token="fresh-refresh",
    )

    assert pool.add_entry(fresh_entry) is not None

    root = _read(root_path)
    assert root["shared_auth_owners"][PROVIDER]["deleted"] is False
    assert PROVIDER not in root.get("providers", {})
    assert [entry["id"] for entry in root["credential_pool"][PROVIDER]] == [
        "fresh-add"
    ]


def test_auth_remove_clears_canonical_singleton_and_fences_profile_copy(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": "stale-profile",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
        },
    )

    assert _clear_auth_store_provider(PROVIDER) is True
    assert A.get_provider_auth_state(PROVIDER) is None
    root = _read(root_path)
    assert PROVIDER not in root["providers"]
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 1
    assert marker["deleted"] is False
    assert _read(profile_path)["providers"][PROVIDER]["tokens"]["access_token"] == (
        "stale-profile"
    )


def test_reauth_advances_logout_generation(profile_and_root):
    profile_path, root_path = profile_and_root
    old_entry = PooledCredential(
        provider=PROVIDER,
        id="old-device",
        label="old",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
            "credential_pool": {PROVIDER: [old_entry.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    live_pool = CredentialPool(PROVIDER, [old_entry], owner_generation=0)
    assert A.clear_provider_auth(PROVIDER) is True

    A._save_codex_tokens(
        {"access_token": "new-access", "refresh_token": "new-refresh"},
    )

    root = _read(root_path)
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 2
    assert marker["deleted"] is False
    assert marker["legacy_migration_closed"] is True
    assert root["providers"][PROVIDER]["tokens"]["access_token"] == "new-access"
    assert [entry.access_token for entry in live_pool.entries()] == ["new-access"]

    A._save_codex_tokens(
        {"access_token": "newer-access", "refresh_token": "newer-refresh"},
    )
    root = _read(root_path)
    assert root["shared_auth_owners"][PROVIDER]["generation"] == 3
    assert root["providers"][PROVIDER]["tokens"]["refresh_token"] == "newer-refresh"


def test_suppression_only_owner_blocks_legacy_profile_adoption(profile_and_root):
    profile_path, root_path = profile_and_root
    _write(root_path, {"version": 1, "providers": {}})
    _write(
        profile_path,
        {
            "version": 1,
            "providers": {},
            "suppressed_sources": {PROVIDER: ["device_code"]},
        },
    )

    assert A.get_provider_auth_state(PROVIDER) is None
    root = _read(root_path)
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 1
    assert marker["deleted"] is True
    assert root["suppressed_sources"][PROVIDER] == ["device_code"]
    assert PROVIDER not in _read(profile_path).get("suppressed_sources", {})


def test_pool_mutation_advances_generation_and_fences_stale_remove(profile_and_root):
    profile_path, root_path = profile_and_root
    first = _entry()
    second = PooledCredential(
        provider=PROVIDER,
        id="codex-2",
        label="second",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token="access-2",
        refresh_token="refresh-2",
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": first.access_token,
                        "refresh_token": first.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [first.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    writer = CredentialPool(PROVIDER, [first], owner_generation=0)
    stale_remover = CredentialPool(PROVIDER, [first], owner_generation=0)

    assert writer.add_entry(second) is not None
    assert stale_remover.remove_index(1) is None

    root = _read(root_path)
    marker = root["shared_auth_owners"][PROVIDER]
    assert marker["generation"] == 1
    assert marker["deleted"] is False
    assert {entry["id"] for entry in root["credential_pool"][PROVIDER]} == {
        "codex-shared",
        "codex-2",
    }
    assert root["providers"][PROVIDER]["tokens"]["access_token"] == first.access_token

    A._save_codex_tokens(
        {"access_token": "refreshed-access", "refresh_token": "refreshed-refresh"},
    )
    selected = stale_remover.select()
    assert selected is not None
    assert selected.id == first.id
    assert selected.refresh_token == "refreshed-refresh"
    assert _read(root_path)["shared_auth_owners"][PROVIDER]["generation"] == 2


def test_seeded_remove_cleanup_cannot_delete_later_reauth(profile_and_root):
    profile_path, root_path = profile_and_root
    entry = _entry(source="device_code")
    alias = PooledCredential(
        provider=PROVIDER,
        id="legacy-alias",
        label="legacy alias",
        auth_type=AUTH_TYPE_OAUTH,
        priority=1,
        source="manual:device_code",
        access_token=entry.access_token,
        refresh_token=entry.refresh_token,
    )
    independent = PooledCredential(
        provider=PROVIDER,
        id="independent",
        label="independent",
        auth_type=AUTH_TYPE_OAUTH,
        priority=2,
        source="manual:device_code",
        access_token="other-access",
        refresh_token="other-refresh",
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": entry.access_token,
                        "refresh_token": entry.refresh_token,
                    }
                }
            },
            "credential_pool": {
                PROVIDER: [
                    entry.to_dict(),
                    alias.to_dict(),
                    independent.to_dict(),
                ]
            },
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    pool = CredentialPool(
        PROVIDER,
        [entry, alias, independent],
        owner_generation=0,
    )

    removed = pool.remove_index(1)
    assert removed is not None
    after_remove = _read(root_path)
    assert PROVIDER not in after_remove["providers"]
    assert after_remove["suppressed_sources"][PROVIDER] == ["device_code"]
    assert [item["id"] for item in after_remove["credential_pool"][PROVIDER]] == [
        "independent"
    ]

    A._save_codex_tokens(
        {"access_token": "reauth-access", "refresh_token": "reauth-refresh"},
    )
    step = find_removal_step(PROVIDER, removed.source)
    assert step is not None
    assert step.remove_fn(PROVIDER, removed).suppress is False

    state = A.get_provider_auth_state(PROVIDER)
    assert state is not None
    assert state["tokens"]["access_token"] == "reauth-access"
    assert PROVIDER not in _read(root_path).get("suppressed_sources", {})


def test_seeded_remove_revocation_blocks_untouched_sibling_alias(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    sibling_path = profile_path.parent.parent / "sibling" / "auth.json"
    seeded = _entry(source="device_code")
    alias = PooledCredential(
        provider=PROVIDER,
        id="sibling-alias",
        label="sibling alias",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token=seeded.access_token,
        refresh_token=seeded.refresh_token,
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": seeded.access_token,
                        "refresh_token": seeded.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [seeded.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    _write(
        sibling_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [alias.to_dict()]},
        },
    )

    pool = CredentialPool(PROVIDER, [seeded], owner_generation=0)
    assert pool.remove_index(1) is not None

    monkeypatch.setattr(A, "_auth_file_path", lambda: sibling_path)
    assert A.read_credential_pool(PROVIDER) == []
    assert _read(root_path).get("credential_pool", {}).get(PROVIDER, []) == []
    assert PROVIDER not in _read(sibling_path).get("credential_pool", {})


def test_quota_clear_targets_global_owner_and_refreshes_live_pool(profile_and_root):
    profile_path, root_path = profile_and_root
    entry = _entry().to_dict()
    entry.update(
        {
            "last_status": "exhausted",
            "last_status_at": 1.0,
            "last_error_code": 429,
            "last_error_reason": "usage_limit_reached",
            "last_error_message": "quota exhausted",
            "last_error_reset_at": 2.0,
        }
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [entry]},
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    live_pool = CredentialPool(
        PROVIDER,
        [PooledCredential.from_dict(PROVIDER, entry)],
        owner_generation=0,
    )

    assert A.clear_codex_pool_quota_cooldowns() == 1
    root = _read(root_path)
    assert root["credential_pool"][PROVIDER][0]["last_status"] is None
    assert root["shared_auth_owners"][PROVIDER]["generation"] == 1
    reloaded = live_pool.entries()
    assert len(reloaded) == 1
    assert reloaded[0].last_status is None


def test_named_profile_codex_save_does_not_change_root_active_provider(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    _write(
        root_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {"anthropic": {"api_key": "keep"}},
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "active_provider": "nous",
            "providers": {"nous": {"access_token": "keep"}},
        },
    )

    A._save_codex_tokens(
        {"access_token": "codex-access", "refresh_token": "codex-refresh"},
    )

    assert _read(root_path)["active_provider"] == "anthropic"
    assert _read(profile_path)["active_provider"] == "nous"


def test_named_profile_codex_refresh_activates_unset_local_profile(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    _write(
        root_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {
                "anthropic": {"api_key": "keep"},
                PROVIDER: {
                    "tokens": {
                        "access_token": "old-codex",
                        "refresh_token": "old-refresh",
                    }
                },
            },
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})

    A._save_codex_tokens(
        {"access_token": "new-codex", "refresh_token": "new-refresh"},
    )

    assert _read(profile_path)["active_provider"] == PROVIDER
    assert _read(root_path)["active_provider"] == "anthropic"


def test_named_profile_auth_add_activates_with_existing_global_pool(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    existing = _entry().to_dict()
    _write(
        root_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {"anthropic": {"api_key": "keep"}},
            "credential_pool": {PROVIDER: [existing]},
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    monkeypatch.setattr(
        A,
        "_codex_device_code_login",
        lambda: {
            "tokens": {
                "access_token": "added-access",
                "refresh_token": "added-refresh",
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-07-23T00:00:00Z",
        },
    )
    from hermes_cli.auth_commands import auth_add_command

    class Args:
        provider = PROVIDER
        auth_type = "oauth"
        api_key = None
        label = "added"

    auth_add_command(Args())

    assert _read(profile_path)["active_provider"] == PROVIDER
    assert _read(root_path)["active_provider"] == "anthropic"


def test_initialized_owner_merges_jwt_distinct_device_account_with_id_collision(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    account_a = _jwt("account-a")
    account_b = _jwt("account-b")
    root_entry = _entry(access=account_a, refresh="refresh-a", source="device_code")
    profile_entry = {
        **_entry(access=account_b, refresh="refresh-b", source="device_code").to_dict(),
        "id": root_entry.id,
    }
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": account_a,
                        "refresh_token": "refresh-a",
                    }
                }
            },
            "credential_pool": {PROVIDER: [root_entry.to_dict()]},
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": account_b,
                        "refresh_token": "refresh-b",
                    }
                }
            },
            "credential_pool": {PROVIDER: [profile_entry]},
        },
    )

    merged = A.read_credential_pool(PROVIDER)

    assert {entry["access_token"] for entry in merged} == {account_a, account_b}
    assert len({entry["id"] for entry in merged}) == 2
    assert next(entry for entry in merged if entry["access_token"] == account_b)[
        "source"
    ] == "manual:device_code"
    profile = _read(profile_path)
    assert PROVIDER not in profile.get("providers", {})
    assert PROVIDER not in profile.get("credential_pool", {})


def test_initialized_owner_merges_later_profile_manual_accounts_and_suppressions(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    first = _entry().to_dict()
    second = {**first, "id": "profile-second", "access_token": "second-access"}
    _write(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [first]},
        },
    )
    _write(
        profile_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [second]},
            "suppressed_sources": {PROVIDER: ["device_code"]},
        },
    )

    entries = A.read_credential_pool(PROVIDER)
    assert {entry["id"] for entry in entries} == {first["id"], "profile-second"}
    assert PROVIDER not in _read(profile_path).get("credential_pool", {})
    assert PROVIDER not in _read(profile_path).get("suppressed_sources", {})
    assert A.list_suppressed_credential_sources(PROVIDER) == ["device_code"]
    assert A.unsuppress_credential_source(PROVIDER, "device_code") is True
    assert A.list_suppressed_credential_sources(PROVIDER) == []


def test_stale_status_writer_cannot_revive_same_id(profile_and_root):
    profile_path, root_path = profile_and_root
    entry = _entry()
    _write(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [entry.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    writer = CredentialPool(PROVIDER, [entry], owner_generation=0)
    stale = CredentialPool(PROVIDER, [entry], owner_generation=0)
    exhausted_payload = entry.to_dict()
    exhausted_payload.update(
        {
            "last_status": "dead",
            "last_status_at": 1.0,
            "last_error_code": 401,
            "last_error_reason": "token_invalidated",
        }
    )
    writer._entries = [PooledCredential.from_dict(PROVIDER, exhausted_payload)]

    assert writer._persist() is True
    assert stale._persist() is False

    root_entry = _read(root_path)["credential_pool"][PROVIDER][0]
    assert root_entry["last_status"] == "dead"
    assert stale.entries()[0].last_status == "dead"


def test_exhaustion_retries_after_concurrent_owner_generation_write(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    first = _entry()
    second = PooledCredential(
        provider=PROVIDER,
        id="codex-sibling",
        label="sibling",
        auth_type=AUTH_TYPE_OAUTH,
        priority=1,
        source="manual:device_code",
        access_token=first.access_token,
        refresh_token="sibling-refresh",
    )
    original_payloads = [first.to_dict(), second.to_dict()]
    _write(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: original_payloads},
            "shared_auth_owners": {
                PROVIDER: {"generation": 0, "deleted": False}
            },
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    pool = CredentialPool(PROVIDER, [first, second], owner_generation=0)
    real_persist = pool._persist
    persist_calls = 0

    def persist_after_concurrent_write(**kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            # Another process advances the owner epoch after this pool marks
            # the failed key in memory but before its CAS reaches disk.
            assert A.write_credential_pool(PROVIDER, original_payloads) == root_path
        return real_persist(**kwargs)

    monkeypatch.setattr(pool, "_persist", persist_after_concurrent_write)

    selected = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint=first.runtime_api_key,
    )

    assert selected is None
    assert persist_calls == 2
    assert {entry.last_status for entry in pool.entries()} == {"exhausted"}
    persisted = _read(root_path)["credential_pool"][PROVIDER]
    assert {entry.get("last_status") for entry in persisted} == {"exhausted"}


def test_refresh_generation_change_stops_iteration_over_stale_snapshot(
    profile_and_root,
    monkeypatch,
):
    profile_path, root_path = profile_and_root
    first = _entry(access="first-access", refresh="first-refresh", source="device_code")
    second = PooledCredential(
        provider=PROVIDER,
        id="second",
        label="second",
        auth_type=AUTH_TYPE_OAUTH,
        priority=1,
        source="manual:device_code",
        access_token="second-access",
        refresh_token="second-refresh",
    )
    _write(
        root_path,
        {
            "version": 1,
            "providers": {
                PROVIDER: {
                    "tokens": {
                        "access_token": first.access_token,
                        "refresh_token": first.refresh_token,
                    }
                }
            },
            "credential_pool": {PROVIDER: [first.to_dict(), second.to_dict()]},
        },
    )
    _write(profile_path, {"version": 1, "active_provider": PROVIDER})
    pool = CredentialPool(PROVIDER, [first, second], owner_generation=0)
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda entry: entry.id == first.id)

    def logout_during_refresh(entry, *, force=False):
        assert A.clear_provider_auth(PROVIDER) is True
        pool._shared_owner_generation_is_current(continue_after_reload=True)
        return None

    monkeypatch.setattr(pool, "_refresh_entry", logout_during_refresh)

    assert pool.select() is None
    assert pool.entries() == []


def test_stale_401_cannot_mark_concurrently_reauthed_entry_dead(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    old = _entry()
    fresh_payload = {
        **old.to_dict(),
        "access_token": "fresh-token",
        "refresh_token": "fresh-refresh",
    }
    _write(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {PROVIDER: [fresh_payload]},
            "shared_auth_owners": {
                PROVIDER: {"generation": 1, "deleted": False}
            },
        },
    )
    _write(profile_path, {"version": 1, "providers": {}})
    pool = CredentialPool(PROVIDER, [old], owner_generation=0)

    assert pool._shared_owner_generation_is_current() is False
    selected = pool.mark_exhausted_and_rotate(
        status_code=401,
        error_context={"error": "token invalidated"},
        api_key_hint=old.runtime_api_key,
    )

    assert selected is not None
    assert selected.access_token == "fresh-token"
    persisted = _read(root_path)["credential_pool"][PROVIDER][0]
    assert persisted.get("last_status") is None


def test_preserved_marker_old_writer_is_detected_by_state_digest(profile_and_root):
    profile_path, root_path = profile_and_root
    original = _entry()
    _write(root_path, {"version": 1, "providers": {}, "credential_pool": {}})
    _write(profile_path, {"version": 1, "providers": {}})
    assert A.write_credential_pool(PROVIDER, [original.to_dict()]) == root_path
    marker = _read(root_path)["shared_auth_owners"][PROVIDER]
    initial_generation = marker["generation"]
    stale_pool = CredentialPool(
        PROVIDER,
        [original],
        owner_generation=initial_generation,
    )

    old_writer_store = _read(root_path)
    old_writer_store["credential_pool"][PROVIDER][0]["refresh_token"] = (
        "old-writer-refresh"
    )
    _write(root_path, old_writer_store)
    unrelated_save = A._load_auth_store(root_path)
    unrelated_save.setdefault("providers", {})["anthropic"] = {"api_key": "other"}
    A._save_auth_store(unrelated_save, target_path=root_path)

    assert stale_pool._persist() is False
    root = _read(root_path)
    assert root["shared_auth_owners"][PROVIDER]["generation"] == (
        initial_generation + 1
    )
    assert root["credential_pool"][PROVIDER][0]["refresh_token"] == (
        "old-writer-refresh"
    )

    assert A.clear_provider_auth(PROVIDER) is True
    tombstoned = _read(root_path)
    tombstoned["credential_pool"][PROVIDER] = [original.to_dict()]
    _write(root_path, tombstoned)
    stale_pool._current_id = original.id
    assert stale_pool.current() is None
    assert A.read_credential_pool(PROVIDER) == []

    A._save_codex_tokens(
        {"access_token": "reactivated", "refresh_token": "reactivated-refresh"},
    )
    visible = A.read_credential_pool(PROVIDER)
    assert len(visible) == 1
    assert visible[0]["access_token"] == "reactivated"
