"""Schema for the generic `computer_use` tool.

Model-agnostic. Any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more
reliable than pixel coordinates. Pixel coordinates remain supported for
models that were trained on them (e.g. Claude's computer-use RL).
"""

from __future__ import annotations

from typing import Any, Dict


# One consolidated tool with an `action` discriminator. Keeps the schema
# compact and the per-turn token cost low.
COMPUTER_USE_SCHEMA: Dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive desktop apps through cua-driver without taking the user's real cursor/focus. "
        "Prefer capture(mode='som') then act by element index; use coordinates only when needed. "
        "Supports native apps and typed browser-page actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture",
                    "click",
                    "double_click",
                    "right_click",
                    "middle_click",
                    "drag",
                    "scroll",
                    "type",
                    "key",
                    "set_value",
                    "wait",
                    "list_apps",
                    "list_windows",
                    "focus_app",
                    "cua_browser_state",
                    "cua_browser_prepare",
                    "cua_browser_navigate",
                    "cua_browser_click",
                    "cua_browser_type",
                    "cua_browser_pointer",
                    "cua_browser_dialog",
                    "cua_browser_set_input_files",
                    "cua_browser_download",
                ],
                "description": (
                    "Action to perform. Capture is read-only; mutations require approval unless "
                    "auto-approved. Prefer set_value for popups/sliders."
                ),
            },
            # ── capture ────────────────────────────────────────────
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "description": (
                    "Capture mode: som (default) = screenshot + numbered elements + AX; "
                    "vision = plain screenshot; ax = accessibility tree only."
                ),
            },
            "app": {
                "type": "string",
                "description": (
                    "App name/bundle ID; omit for frontmost window. Use screen/desktop for the "
                    "OS shell. Captures cover one window/display at a time."
                ),
            },
            "pid": {
                "type": "integer",
                "description": (
                    "Optional exact process target for action='capture'. Pair "
                    "with window_id when discovery cannot resolve an X11 app."
                ),
            },
            "window_id": {
                "type": "integer",
                "description": (
                    "Optional exact native window target for action='capture'. "
                    "Pair with pid when an external cua-driver list_windows "
                    "lookup has already identified the window."
                ),
            },
            "max_elements": {
                "type": "integer",
                "description": (
                    "AX element cap (default 100, max 1000). If truncated, narrow app scope or "
                    "raise it; screenshot-backed som/vision captures are unaffected."
                ),
                "default": 100,
                "minimum": 1,
                "maximum": 1000,
            },
            # ── click / drag / scroll targeting ────────────────────
            "element": {
                "type": "integer",
                "description": (
                    "1-based element index from the latest som capture; preferred over coordinates."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Window-relative [x,y] from the capture; use only without an element index."
                ),
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button. Defaults to left.",
            },
            "modifiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "cmd", "shift", "option", "alt", "ctrl", "fn",
                        "win", "windows", "super", "meta",
                    ],
                },
                "description": "Modifier keys held during the action.",
            },
            # ── drag ───────────────────────────────────────────────
            "from_element": {"type": "integer",
                              "description": "Source element index (drag)."},
            "to_element": {"type": "integer",
                            "description": "Target element index (drag)."},
            "from_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Source [x,y] (drag; use when no element available).",
            },
            "to_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Target [x,y] (drag; use when no element available).",
            },
            # ── scroll ─────────────────────────────────────────────
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction.",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll wheel ticks. Default 3.",
            },
            # ── set_value ──────────────────────────────────────────
            "value": {
                "type": "string",
                "description": (
                    "For action='set_value': the value to set on the element. "
                    "For AXPopUpButton / select dropdowns, pass the option's "
                    "display label (e.g. 'Blue'). For sliders and other "
                    "AXValue-settable elements, pass the numeric or string value."
                ),
            },
            # ── type / key / wait ──────────────────────────────────
            "text": {
                "type": "string",
                "description": "Text to type (respects the current layout).",
            },
            "keys": {
                "type": "string",
                "description": (
                    "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', "
                    "'escape', 'tab'. Use '+' to combine."
                ),
            },
            "seconds": {
                "type": "number",
                "description": "Seconds to wait. Max 30.",
            },
            # ── focus_app ──────────────────────────────────────────
            "raise_window": {
                "type": "boolean",
                "description": (
                    "focus_app only. True visibly raises the window; default false routes in background."
                ),
            },
            # ── delivery (verify → escalate ladder) ────────────────
            "delivery_mode": {
                "type": "string",
                "enum": ["background", "foreground"],
                "description": (
                    "Input route; background is default. A confirmed effect is done; for unverifiable, "
                    "inspect fresh state before retrying. Escalate only after suspected_noop/refusal. "
                    "Foreground visibly fronts/restores the app and needs separate approval."
                ),
            },
            "bring_to_front": {
                "type": "boolean",
                "description": (
                    "foreground only: persistently raise before input; separate approval, default false."
                ),
            },
            # ── cua-driver typed browser route ─────────────────────
            "tab_id": {
                "type": "string",
                "description": "Opaque tab capability returned by cua_browser_state.",
            },
            "ref": {
                "type": "string",
                "description": "Current semantic ref from the latest cua_browser_state snapshot.",
            },
            "destination_ref": {
                "type": "string",
                "description": "Current destination ref for a typed pointer action.",
            },
            "url": {"type": "string", "description": "URL for cua_browser_navigate."},
            "input_route": {
                "type": "string",
                "enum": ["trusted", "dom_event"],
                "description": (
                    "Typed-browser trust class. Defaults to trusted. dom_event "
                    "is an explicit downgrade and is never selected silently."
                ),
            },
            "snapshot_format": {
                "type": "string",
                "enum": ["semantic_v2", "dom_refs_v1"],
                "description": "Typed-browser snapshot format; semantic_v2 is the default.",
            },
            "query": {"type": "string", "description": "Optional browser-state query."},
            "scope_ref": {"type": "string", "description": "Optional current ref to scope a snapshot."},
            "continuation": {"type": "string", "description": "Continuation minted by the current snapshot."},
            "profile_mode": {
                "type": "string",
                "enum": ["isolated_new", "isolated_named", "existing_profile"],
                "description": (
                    "Browser preparation mode; existing_profile availability is permission-enforced."
                ),
            },
            "profile_name": {"type": "string", "description": "Name for isolated_named setup."},
            "allow_launch": {
                "type": "boolean",
                "description": "Explicitly allow launch of a driver-owned isolated browser.",
            },
            "browser_pointer_action": {
                "type": "string",
                "enum": ["hover", "right_click", "double_click", "scroll", "drag"],
                "description": "Operation for cua_browser_pointer.",
            },
            "browser_dialog_action": {
                "type": "string",
                "enum": ["inspect", "accept", "dismiss"],
                "description": "Page JavaScript dialog action; native prompts stay on the native ladder.",
            },
            "browser_type_mode": {
                "type": "string",
                "enum": ["insert_text", "keystrokes"],
                "description": "Delivery form for cua_browser_type; defaults to insert_text.",
            },
            "dialog_id": {"type": "string", "description": "Opaque page-dialog capability."},
            "prompt_text": {"type": "string", "description": "Optional text for a page prompt dialog."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit paths for cua_browser_set_input_files.",
            },
            "destination_root": {
                "type": "string",
                "description": "Approved destination root for cua_browser_download.",
            },
            "delta_x": {"type": "number", "description": "Typed pointer horizontal delta."},
            "delta_y": {"type": "number", "description": "Typed pointer vertical delta."},
            "x": {"type": "number", "description": "Typed browser viewport x coordinate."},
            "y": {"type": "number", "description": "Typed browser viewport y coordinate."},
            "to_x": {"type": "number", "description": "Typed browser drag destination x."},
            "to_y": {"type": "number", "description": "Typed browser drag destination y."},
            # ── return shape ───────────────────────────────────────
            "capture_after": {
                "type": "boolean",
                "description": (
                    "If true, take a follow-up capture after the action "
                    "and include it in the response. Saves a round-trip "
                    "when you need to verify an action's effect."
                ),
            },
        },
        "required": ["action"],
    },
}


def get_computer_use_schema() -> Dict[str, Any]:
    """Return the generic OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
