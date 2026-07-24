"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)


class AmbiguousChannelName(ValueError):
    """Raised when a friendly target name maps to multiple destinations."""

DIRECTORY_PATH = get_hermes_home() / "channel_directory.json"
# Throttle window for repeated Slack channel-directory refresh failures.
# The directory rebuilds on a timer, so a persistent workspace error (e.g.
# missing scope, revoked token) would otherwise re-log the same warning on
# every refresh. Warn once per (team, error detail) per interval; repeats
# drop to DEBUG.
_SLACK_DIRECTORY_WARNING_INTERVAL_SECONDS = 3600
_slack_directory_warning_last: Dict[tuple[str, str], float] = {}

# User-maintained friendly-name overlay. The directory is fully regenerated
# from live adapters + session data on a timer, so hand-edits to
# channel_directory.json don't survive. Aliases declared here are re-applied
# on every build AND every load, giving durable human-friendly names (and
# letting you pre-name a chat before it has produced any traffic).
# Format: {"<platform>": {"<chat_id>": "<friendly name>", ...}, ...}
CHANNEL_ALIASES_PATH = get_hermes_home() / "channel_aliases.json"


def _load_channel_aliases() -> Dict[str, Dict[str, str]]:
    if not CHANNEL_ALIASES_PATH.exists():
        return {}
    try:
        with open(CHANNEL_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_channel_aliases(platforms: Dict[str, Any]) -> None:
    """Overlay friendly names onto directory entries by chat_id.

    Renames matching entries in place; injects a placeholder entry for an
    aliased id that hasn't been discovered yet (so a freshly-created group is
    addressable by name before its first message). Mutates *platforms*.
    """
    aliases = _load_channel_aliases()
    for plat_name, id_map in aliases.items():
        if not isinstance(id_map, dict):
            continue
        entries = platforms.setdefault(plat_name, [])
        if not isinstance(entries, list):
            continue
        for chat_id, friendly in id_map.items():
            if not isinstance(friendly, str) or not friendly.strip():
                continue
            chat_id = str(chat_id)
            friendly = friendly.strip()
            matched = False
            for e in entries:
                if isinstance(e, dict) and e.get("id") == chat_id:
                    e["name"] = friendly
                    matched = True
            if not matched:
                entries.append({
                    "id": chat_id,
                    "name": friendly,
                    "type": "group" if str(chat_id).endswith("@g.us") else "dm",
                    "thread_id": None,
                })


def _normalize_channel_query(value: str) -> str:
    return value.lstrip("#").strip().lower()


def _channel_target_name(platform_name: str, channel: Dict[str, Any]) -> str:
    """Return the human-facing target label shown to users for a channel entry."""
    name = channel["name"]
    if platform_name == "discord" and channel.get("guild"):
        return f"#{name}"
    if platform_name != "discord" and channel.get("type"):
        return f"{name} ({channel['type']})"
    return name


def _unique_channel_id(channels: List[Dict[str, Any]]) -> Optional[str]:
    """Return the sole distinct target ID in *channels*, else ``None``.

    Directory builders can legitimately contribute duplicate rows for the
    same target (for example, Discord enumeration plus session history), so
    uniqueness is about destination IDs rather than row count.
    """
    ids = {str(ch["id"]) for ch in channels if ch.get("id") is not None}
    if len(ids) == 1:
        return next(iter(ids))
    return None


def _channel_query_id_index(
    platform_name: str,
    channels: List[Dict[str, Any]],
) -> Dict[str, set[str]]:
    """Index every accepted exact/qualified query by destination ID."""
    query_ids: Dict[str, set[str]] = {}
    for channel in channels:
        channel_id = str(channel["id"])
        label = _channel_target_name(platform_name, channel)
        queries = {
            _normalize_channel_query(channel["name"]),
            _normalize_channel_query(label),
        }
        guild = channel.get("guild")
        if platform_name == "discord" and guild:
            queries.add(
                _normalize_channel_query(f"{guild}/{channel['name']}")
            )
        for query in queries:
            query_ids.setdefault(query, set()).add(channel_id)
    return query_ids


def _target_ref_round_trips(
    platform_name: str,
    target_ref: str,
    channel_id: str,
) -> bool:
    """Return whether downstream target parsing preserves this destination.

    Some valid human names also look like explicit IDs (for example a Discord
    display name of ``123``). Those refs bypass directory resolution entirely,
    so they are safe to advertise only when the parser identifies the same
    destination. If the parser cannot be imported, fail closed to the raw ID.
    """
    try:
        from tools.send_message_tool import _parse_target_ref

        parsed_chat_id, parsed_thread_id, is_explicit = _parse_target_ref(
            platform_name, target_ref
        )
    except Exception:
        return False

    if is_explicit:
        parsed_id = str(parsed_chat_id)
        if parsed_thread_id is not None:
            parsed_id = f"{parsed_id}:{parsed_thread_id}"
        return parsed_id == channel_id

    try:
        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        return False
    return str(resolved) == channel_id


def _display_target_refs(
    platform_name: str,
    channels: List[Dict[str, Any]],
) -> Dict[int, str]:
    """Map directory-row identities to friendly, unambiguous target refs.

    Duplicate human names are common across Discord guilds and Slack
    workspaces. Showing the same friendly ref for several IDs invites a send
    to the wrong destination. Qualify Discord names by guild when that is
    sufficient; otherwise expose the unambiguous raw ID. Indexes are built
    once so formatting a large workspace remains linear rather than quadratic.
    """
    query_ids = _channel_query_id_index(platform_name, channels)
    raw_query_ids: Dict[str, set[str]] = defaultdict(set)
    for candidate in channels:
        raw_query_ids[_normalize_channel_query(str(candidate["id"]))].add(
            str(candidate["id"])
        )

    def safe_ref(candidate: str, channel_id: str) -> Optional[str]:
        key = _normalize_channel_query(candidate)
        if (
            query_ids[key] | raw_query_ids[key]
        ) == {channel_id} and _target_ref_round_trips(
            platform_name, candidate, channel_id
        ):
            return candidate

        # Explicit name intent avoids both opaque-ID parsing and raw-ID
        # precedence in ``resolve_channel_name``.
        forced = f"name={candidate}"
        if query_ids[key] == {channel_id} and _target_ref_round_trips(
            platform_name, forced, channel_id
        ):
            return forced
        return None

    refs: Dict[int, str] = {}
    for channel in channels:
        channel_id = str(channel["id"])
        label = _channel_target_name(platform_name, channel)
        target_ref = safe_ref(label, channel_id)
        if target_ref is not None:
            refs[id(channel)] = target_ref
            continue

        guild = channel.get("guild")
        if platform_name == "discord" and guild:
            qualified = f"{guild}/{channel['name']}"
            target_ref = safe_ref(qualified, channel_id)
            if target_ref is not None:
                refs[id(channel)] = target_ref
                continue

        raw_ref = channel_id
        if not _target_ref_round_trips(platform_name, raw_ref, channel_id):
            raw_ref = f"id={channel_id}"
        refs[id(channel)] = raw_ref

    return refs


def _session_entry_id(origin: Dict[str, Any]) -> Optional[str]:
    chat_id = origin.get("chat_id")
    if not chat_id:
        return None
    thread_id = origin.get("thread_id")
    if thread_id:
        return f"{chat_id}:{thread_id}"
    return str(chat_id)


def _session_entry_name(origin: Dict[str, Any]) -> str:
    base_name = origin.get("chat_name") or origin.get("user_name") or str(origin.get("chat_id"))
    thread_id = origin.get("thread_id")
    if not thread_id:
        return base_name

    topic_label = origin.get("chat_topic") or f"topic {thread_id}"
    return f"{base_name} / {topic_label}"


def _warn_slack_directory(team_id: str, detail: str) -> None:
    """Warn once per team/error per interval for recurring Slack refresh failures."""
    key = (str(team_id), str(detail))
    now = time.monotonic()
    last = _slack_directory_warning_last.get(key)
    if last is None or now - last >= _SLACK_DIRECTORY_WARNING_INTERVAL_SECONDS:
        _slack_directory_warning_last[key] = now
        logger.warning(
            "Channel directory: failed to list Slack channels for team %s: %s",
            team_id,
            detail,
        )
    else:
        logger.debug(
            "Channel directory: suppressed repeated Slack channel list failure "
            "for team %s: %s",
            team_id,
            detail,
        )


# ---------------------------------------------------------------------------
# Build / refresh
# ---------------------------------------------------------------------------

async def build_channel_directory(adapters: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Build a channel directory from connected platform adapters and session data.

    Returns the directory dict and writes it to DIRECTORY_PATH.
    """
    from gateway.config import Platform

    platforms: Dict[str, List[Dict[str, str]]] = {}

    for platform, adapter in adapters.items():
        try:
            if platform == Platform.DISCORD:
                platforms["discord"] = await asyncio.to_thread(_build_discord, adapter)
            elif platform == Platform.SLACK:
                platforms["slack"] = await _build_slack(adapter)
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Platforms that don't support direct channel enumeration get session-based
    # discovery automatically, but only for platforms connected in THIS gateway
    # process. Historical session origins for disabled/decommissioned platforms
    # must not be resurrected into the active send-target directory (stale
    # targets make send_message route to platforms that can no longer deliver).
    _SKIP_SESSION_DISCOVERY = frozenset({"local", "api_server", "webhook"})
    adapter_platform_names = {getattr(p, "value", str(p)) for p in adapters}
    for plat in Platform:
        plat_name = plat.value
        if (
            plat_name in _SKIP_SESSION_DISCOVERY
            or plat_name in platforms
            or plat_name not in adapter_platform_names
        ):
            continue
        platforms[plat_name] = await asyncio.to_thread(_build_from_sessions, plat_name)

    # Include plugin-registered platforms (dynamic enum members aren't in
    # Platform.__members__, so the loop above misses them). Same
    # connected-only rule: don't expose stale session targets for plugins
    # that are not loaded.
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if (
                entry.name not in _SKIP_SESSION_DISCOVERY
                and entry.name not in platforms
                and entry.name in adapter_platform_names
            ):
                platforms[entry.name] = await asyncio.to_thread(_build_from_sessions, entry.name)
    except Exception:
        pass

    # Overlay user-maintained friendly names before persisting.
    _apply_channel_aliases(platforms)

    directory = {
        "updated_at": datetime.now().isoformat(),
        "platforms": platforms,
    }

    try:
        atomic_json_write(DIRECTORY_PATH, directory)
    except Exception as e:
        logger.warning("Channel directory: failed to write: %s", e)

    return directory


def _build_discord(adapter) -> List[Dict[str, str]]:
    """Enumerate all text channels and forum channels the Discord bot can see."""
    channels = []
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord  # noqa: F401 — SDK presence check
    except ImportError:
        return channels

    for guild in client.guilds:
        for ch in guild.text_channels:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "channel",
            })
        # Forum channels (type 15) — creating a message auto-spawns a thread post.
        forums = getattr(guild, "forum_channels", None) or []
        for ch in forums:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "forum",
            })
        # Also include DM-capable users we've interacted with is not
        # feasible via guild enumeration; those come from sessions.

    # Merge any DMs from session history
    channels.extend(_build_from_sessions("discord"))
    return channels


def _slack_api_error_code(error: Exception) -> Optional[str]:
    """Return Slack Web API error code from SlackApiError-like exceptions."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        value = response.get("error")
        return str(value) if value else None
    if response is not None:
        try:
            value = response.get("error")
            return str(value) if value else None
        except Exception:
            pass
    return None


async def _build_slack(adapter) -> List[Dict[str, Any]]:
    """List Slack channels the bot has joined across all workspaces.

    Uses ``users.conversations`` against each workspace's web client. Pulls
    public + private channels the bot is a member of, then merges in DMs
    discovered from session history (IMs aren't useful to enumerate
    proactively). If the Slack app lacks channels:read, fall back to session
    history quietly instead of logging a recurring warning every refresh.
    """
    team_clients = getattr(adapter, "_team_clients", None) or {}
    if not team_clients:
        return await asyncio.to_thread(_build_from_sessions, "slack")

    channels: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for team_id, client in team_clients.items():
        try:
            cursor: Optional[str] = None
            for _page in range(20):  # safety cap on pagination
                response = await client.users_conversations(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=200,
                    cursor=cursor,
                )
                if not response.get("ok"):
                    error_code = response.get("error", "unknown")
                    if error_code == "missing_scope":
                        logger.debug(
                            "Channel directory: Slack team %s lacks channels:read; using session history only",
                            team_id,
                        )
                    else:
                        detail = f"users.conversations not ok: {error_code}"
                        _warn_slack_directory(team_id, detail)
                    break
                for ch in response.get("channels", []):
                    cid = ch.get("id")
                    name = ch.get("name")
                    if not cid or not name or cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    channels.append({
                        "id": cid,
                        "name": name,
                        "type": "private" if ch.get("is_private") else "channel",
                    })
                cursor = (response.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        except Exception as e:
            if _slack_api_error_code(e) == "missing_scope":
                logger.debug(
                    "Channel directory: Slack team %s lacks channels:read; using session history only",
                    team_id,
                )
            else:
                _warn_slack_directory(team_id, str(e))
            continue

    # Merge in DM/group entries discovered from session history.
    # Build a lookup from API-discovered channels so we can enrich session entries.
    api_name_lookup = {ch["id"]: ch["name"] for ch in channels}

    for entry in await asyncio.to_thread(_build_from_sessions, "slack"):
        eid = entry.get("id")
        if eid not in seen_ids:
            # If the entry name is still a raw Slack ID (e.g. C0xxx / D0xxx),
            # try to resolve it from the API lookup first.
            if entry.get("name", "").startswith(("C0", "D0", "G0")):
                if eid in api_name_lookup:
                    entry["name"] = api_name_lookup[eid]
            channels.append(entry)
            seen_ids.add(eid)

    # Resolve remaining raw-ID entries (DMs, private channels not in bot scope)
    # by calling conversations.info + users.info for each.
    unresolved = [ch for ch in channels if ch.get("name", "").startswith(("C0", "D0", "G0"))]
    if unresolved and team_clients:
        client = next(iter(team_clients.values()))
        for entry in unresolved:
            try:
                resp = await client.conversations_info(channel=entry["id"])
                if not resp.get("ok"):
                    continue
                ch_info = resp.get("channel", {})
                if ch_info.get("is_im"):
                    peer_user = ch_info.get("user", "")
                    if peer_user:
                        user_resp = await client.users_info(user=peer_user)
                        if user_resp.get("ok"):
                            u = user_resp["user"]
                            entry["name"] = (
                                u.get("profile", {}).get("display_name")
                                or u.get("real_name")
                                or u.get("name")
                                or entry["id"]
                            )
                            entry["type"] = "dm"
                else:
                    entry["name"] = ch_info.get("name") or ch_info.get("name_normalized") or entry["id"]
            except Exception as e:
                logger.debug("Channel directory: failed to resolve %s: %s", entry["id"], e)
                continue

    return channels


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from gateway session origin data.

    state.db is the primary source (#9006): gateway session rows persist
    origin_json.  Falls back to sessions.json for pre-migration databases.
    """
    entries = _build_from_sessions_db(platform_name)
    if entries:
        return entries
    return _build_from_sessions_json(platform_name)


def _build_from_sessions_db(platform_name: str) -> List[Dict[str, str]]:
    """Pull channels/contacts from state.db gateway session rows."""
    entries: List[Dict[str, str]] = []
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            lister = getattr(db, "list_gateway_sessions", None)
            if not callable(lister):
                return []
            rows = lister(platform=platform_name, active_only=False)
        finally:
            db.close()

        seen_ids = set()
        for row in rows:
            origin: Dict[str, Any] = {}
            if row.get("origin_json"):
                try:
                    parsed = json.loads(row["origin_json"])
                    if isinstance(parsed, dict):
                        origin = parsed
                except (TypeError, ValueError):
                    pass
            if not origin:
                origin = {
                    "chat_id": row.get("chat_id"),
                    "thread_id": row.get("thread_id"),
                    "chat_name": row.get("display_name"),
                }
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": row.get("chat_type") or "dm",
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug(
            "Channel directory: state.db session read failed for %s: %s",
            platform_name, e,
        )
    return entries


def _build_from_sessions_json(platform_name: str) -> List[Dict[str, str]]:
    """Legacy fallback: pull channels/contacts from sessions.json origin data."""
    sessions_path = get_hermes_home() / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
            # Skip documentation/metadata sentinels (keys starting with "_",
            # e.g. the gateway's "_README" note) — not session entries.
            if str(_key).startswith("_") or not isinstance(session, dict):
                continue
            origin = session.get("origin") or {}
            if origin.get("platform") != platform_name:
                continue
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": session.get("chat_type", "dm"),
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug("Channel directory: failed to read sessions for %s: %s", platform_name, e)

    return entries


# ---------------------------------------------------------------------------
# Read / resolve
# ---------------------------------------------------------------------------

def load_directory() -> Dict[str, Any]:
    """Load the cached channel directory from disk."""
    if not DIRECTORY_PATH.exists():
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base
    try:
        with open(DIRECTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Re-apply aliases on read so friendly names take effect immediately,
        # even between timed rebuilds and for brand-new alias entries.
        _apply_channel_aliases(data.setdefault("platforms", {}))
        return data
    except Exception:
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base


def lookup_channel_type(platform_name: str, chat_id: str) -> Optional[str]:
    """Return the channel ``type`` string (e.g. ``"channel"``, ``"forum"``) for *chat_id*, or *None* if unknown."""
    directory = load_directory()
    for ch in directory.get("platforms", {}).get(platform_name, []):
        if ch.get("id") == chat_id:
            return ch.get("type")
    return None


def split_channel_target_id(
    platform_name: str,
    target_id: str,
) -> tuple[str, Optional[str]]:
    """Recover chat/thread parts stored on a directory entry.

    Session-derived entries retain ``thread_id`` alongside their legacy
    ``<chat_id>:<thread_id>`` composite ID. This metadata is authoritative for
    platforms whose opaque IDs cannot be split by the generic target parser.
    """
    raw_target = str(target_id)
    directory = load_directory()
    for channel in directory.get("platforms", {}).get(platform_name, []):
        if str(channel.get("id")) != raw_target:
            continue
        thread_id = channel.get("thread_id")
        if thread_id is None:
            break
        thread = str(thread_id)
        suffix = f":{thread}"
        if raw_target.endswith(suffix):
            chat_id = raw_target[: -len(suffix)]
            if chat_id:
                return chat_id, thread
        break
    return raw_target, None


def resolve_channel_name(platform_name: str, name: str) -> Optional[str]:
    """
    Resolve a human-friendly channel name to a numeric ID.

    Matching strategy (case-insensitive, unique matches only):
    - Discord: "bot-home", "#bot-home", "GuildName/bot-home"
    - Telegram: display name or group name
    - Slack: "engineering", "#engineering"

    Ambiguous names raise :class:`AmbiguousChannelName` rather than selecting
    an arbitrary target; callers can disambiguate with a guild-qualified
    Discord name or raw ID.
    """
    directory = load_directory()
    channels = directory.get("platforms", {}).get(platform_name, [])
    if not channels:
        return None

    raw = name.strip()
    force_name = raw.lower().startswith("name=")
    if force_name:
        raw = raw[len("name=") :].strip()
        if not raw:
            return None

    # 0. Exact ID match — case-sensitive, no normalization. Lets callers pass
    # raw platform IDs (e.g. Slack "C0B0QV5434G") even when the format guard
    # in _parse_target_ref hasn't recognized them as explicit. ``name=`` opts
    # out of this precedence when a human label is intentionally ID-shaped.
    if not force_name:
        for ch in channels:
            if ch.get("id") == raw:
                return ch["id"]

    query = _normalize_channel_query(raw)

    # 1. Exact names, display labels, and Discord guild-qualified names share
    # one index. Unioning those interpretations prevents a DM/thread label
    # containing "/" from shadowing (or being shadowed by) a guild channel.
    exact_ids = _channel_query_id_index(platform_name, channels).get(query, set())
    if len(exact_ids) > 1:
        raise AmbiguousChannelName(
            f"Target '{name}' on {platform_name} matches multiple destinations"
        )
    if exact_ids:
        return next(iter(exact_ids))

    # 2. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if _normalize_channel_query(ch["name"]).startswith(query)]
    matched_id = _unique_channel_id(matches)
    if matches and matched_id is None:
        raise AmbiguousChannelName(
            f"Target '{name}' on {platform_name} matches multiple destinations"
        )
    return matched_id


def format_directory_for_display(platform_filter: Optional[str] = None) -> str:
    """Return a human-readable list of available messaging targets."""
    directory = load_directory()
    platforms = directory.get("platforms", {})
    if platform_filter:
        key = platform_filter.strip().lower()
        platforms = {
            name: channels
            for name, channels in platforms.items()
            if name.lower() == key
        }

    if not any(platforms.values()):
        return "No messaging platforms connected or no channels discovered yet."

    lines = ["Available messaging targets:\n"]

    for plat_name, channels in sorted(platforms.items()):
        if not channels:
            continue
        target_refs = _display_target_refs(plat_name, channels)

        # Group Discord channels by guild
        if plat_name == "discord":
            guilds: Dict[str, List] = {}
            dms: List = []
            for ch in channels:
                guild = ch.get("guild")
                if guild:
                    guilds.setdefault(guild, []).append(ch)
                else:
                    dms.append(ch)

            for guild_name, guild_channels in sorted(guilds.items()):
                lines.append(f"Discord ({guild_name}):")
                for ch in sorted(guild_channels, key=lambda c: c["name"]):
                    lines.append(f"  discord:{target_refs[id(ch)]}")
            if dms:
                lines.append("Discord (DMs):")
                for ch in dms:
                    lines.append(f"  discord:{target_refs[id(ch)]}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                lines.append(f"  {plat_name}:{target_refs[id(ch)]}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
