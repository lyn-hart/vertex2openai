"""
Anthropic Messages API ↔ Gemini conversion and SSE helpers.

Converts Anthropic `/v1/messages` requests to Gemini Content + generation config,
and maps Gemini responses / streams back to Anthropic event shapes so Claude Code
can talk to this adapter with native Gemini model names.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from message_processing import (
    _build_function_call_part,
    _build_function_response_part,
    _decode_tool_call_id_thought_signature,
    _encode_tool_call_id_with_thought_signature,
    parse_gemini_response_for_reasoning_and_content,
)


# ── text / content helpers ──────────────────────────────────────────────────


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = (block.get("type") or "").lower()
                if btype in ("text", "input_text", "output_text") and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif btype == "thinking" and isinstance(block.get("thinking"), str):
                    # Do not inject thinking into system/user text by default
                    continue
                elif btype == "tool_result":
                    parts.append(_tool_result_to_text(block))
        return "\n".join(parts)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _tool_result_to_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return _as_text(c)
    if c is None:
        return ""
    try:
        return json.dumps(c, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(c)


def _tool_result_to_dict(block: dict) -> Dict[str, Any]:
    text = _tool_result_to_text(block)
    if not text:
        return {"result": ""}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"result": text}


def _image_to_gemini_part(block: dict) -> Optional[types.Part]:
    source = block.get("source") or {}
    if not isinstance(source, dict):
        return None
    stype = (source.get("type") or "").lower()
    if stype == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        if not data:
            return None
        try:
            import base64

            raw = base64.b64decode(data)
            return types.Part.from_bytes(data=raw, mime_type=media)
        except Exception as e:
            print(f"Warning: Failed to decode Anthropic base64 image: {e}")
            return None
    if stype == "url":
        url = source.get("url") or ""
        if url.startswith("data:") and ";base64," in url:
            try:
                import base64

                header, b64 = url.split(";base64,", 1)
                media = header.split(":", 1)[1] if ":" in header else "image/png"
                raw = base64.b64decode(b64)
                return types.Part.from_bytes(data=raw, mime_type=media)
            except Exception as e:
                print(f"Warning: Failed to decode data-URL image: {e}")
                return None
        # Remote URLs not fetched here; pass as text reference
        if url:
            return types.Part(text=f"[image: {url}]")
    return None


def flatten_system_instruction(system: Any) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        text = system.strip()
        return text or None
    text = _as_text(system).strip()
    return text or None


# ── request conversion ──────────────────────────────────────────────────────


def create_anthropic_gemini_contents(
    messages: List[Any],
    *,
    tool_name_by_id: Optional[Dict[str, str]] = None,
) -> List[types.Content]:
    """
    Convert Anthropic messages to Gemini Content list.

    tool_name_by_id is optional pre-seeded map of tool_use_id → function name
    for tool_result blocks that omit the name (Claude Code always uses tool_use_id).
    """
    print("Converting Anthropic messages to Gemini format...")
    gemini_messages: List[types.Content] = []
    name_map: Dict[str, str] = dict(tool_name_by_id or {})
    pending_function_response_parts: List[types.Part] = []

    def flush_pending_function_response_parts():
        nonlocal pending_function_response_parts
        if pending_function_response_parts:
            gemini_messages.append(
                types.Content(role="function", parts=pending_function_response_parts)
            )
            pending_function_response_parts = []

    for idx, raw in enumerate(messages or []):
        if not isinstance(raw, dict):
            continue
        role = (raw.get("role") or "user").lower()
        content = raw.get("content")

        if role == "user":
            # Split tool_result blocks into Gemini function responses
            if isinstance(content, list):
                pending_user_parts: List[types.Part] = []
                for block in content:
                    if isinstance(block, dict) and (block.get("type") or "").lower() == "tool_result":
                        # Flush non-tool user content before function responses
                        if pending_user_parts:
                            flush_pending_function_response_parts()
                            gemini_messages.append(
                                types.Content(role="user", parts=pending_user_parts)
                            )
                            pending_user_parts = []

                        tool_use_id = (
                            block.get("tool_use_id")
                            or block.get("tool_call_id")
                            or block.get("id")
                            or ""
                        )
                        raw_id, _ = _decode_tool_call_id_thought_signature(str(tool_use_id))
                        function_name = (
                            block.get("name")
                            or name_map.get(str(tool_use_id))
                            or name_map.get(raw_id)
                            or "tool"
                        )
                        tool_output = _tool_result_to_dict(block)
                        if block.get("is_error"):
                            tool_output = {"error": tool_output.get("result", tool_output)}
                        part = _build_function_response_part(
                            function_name=function_name,
                            tool_output_data=tool_output,
                            tool_call_id=raw_id,
                        )
                        pending_function_response_parts.append(part)
                    elif isinstance(block, dict) and (block.get("type") or "").lower() == "image":
                        img_part = _image_to_gemini_part(block)
                        if img_part is not None:
                            pending_user_parts.append(img_part)
                    elif isinstance(block, dict) and (block.get("type") or "").lower() in (
                        "text",
                        "input_text",
                    ):
                        text = block.get("text") or ""
                        if text:
                            pending_user_parts.append(types.Part(text=text))
                    elif isinstance(block, str) and block:
                        pending_user_parts.append(types.Part(text=block))
                    elif isinstance(block, dict):
                        # document / other: best-effort text
                        t = block.get("text") or block.get("title") or _as_text(block)
                        if t:
                            pending_user_parts.append(types.Part(text=str(t)))

                if pending_function_response_parts:
                    flush_pending_function_response_parts()
                if pending_user_parts:
                    gemini_messages.append(types.Content(role="user", parts=pending_user_parts))
            else:
                flush_pending_function_response_parts()
                text = _as_text(content)
                if text:
                    gemini_messages.append(
                        types.Content(role="user", parts=[types.Part(text=text)])
                    )

        elif role == "assistant":
            flush_pending_function_response_parts()
            parts: List[types.Part] = []
            if isinstance(content, str):
                if content:
                    parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        if isinstance(block, str) and block:
                            parts.append(types.Part(text=block))
                        continue
                    btype = (block.get("type") or "text").lower()
                    if btype in ("text", "output_text"):
                        t = block.get("text") or ""
                        if t:
                            parts.append(types.Part(text=t))
                    elif btype == "thinking":
                        # Gemini history does not need thinking blocks; skip
                        continue
                    elif btype == "tool_use":
                        name = block.get("name") or ""
                        tool_id = block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                        raw_id, thought_sig = _decode_tool_call_id_thought_signature(str(tool_id))
                        if name and raw_id:
                            name_map[str(tool_id)] = name
                            name_map[raw_id] = name
                        inp = block.get("input")
                        if isinstance(inp, str):
                            try:
                                parsed_args = json.loads(inp) if inp else {}
                            except json.JSONDecodeError:
                                parsed_args = {"raw": inp}
                        elif isinstance(inp, dict):
                            parsed_args = inp
                        else:
                            parsed_args = {}
                        parts.append(
                            _build_function_call_part(
                                function_name=name,
                                parsed_arguments=parsed_args if isinstance(parsed_args, dict) else {},
                                tool_call_id=raw_id,
                                thought_signature=thought_sig,
                            )
                        )
            else:
                text = _as_text(content)
                if text:
                    parts.append(types.Part(text=text))

            if parts:
                gemini_messages.append(types.Content(role="model", parts=parts))
            else:
                print(f"Skipping empty assistant message at index {idx}")

        elif role in ("system", "developer"):
            # Prefer top-level system; if mid-conversation system appears, treat as user
            flush_pending_function_response_parts()
            text = _as_text(content)
            if text.strip():
                gemini_messages.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )
        else:
            flush_pending_function_response_parts()
            text = _as_text(content)
            if text:
                gemini_messages.append(
                    types.Content(role="user", parts=[types.Part(text=text)])
                )

    flush_pending_function_response_parts()

    if not gemini_messages:
        print("Warning: No Anthropic messages converted. Using placeholder user prompt.")
        return [
            types.Content(
                role="user",
                parts=[types.Part(text="Placeholder prompt: No valid input messages provided.")],
            )
        ]

    print(f"Converted to {len(gemini_messages)} Gemini messages")
    return gemini_messages


def _sanitize_json_schema_for_gemini(schema: Any) -> Any:
    """Drop keys that only confuse Gemini while keeping full JSON Schema shape.

    Claude Code tools often include draft-07 fields (propertyNames, exclusiveMinimum,
    $schema, etc.). Gemini's typed `Schema` rejects many of these, so we pass the
    cleaned object via FunctionDeclaration.parameters_json_schema instead.
    """
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("$schema", "$id", "$comment", "definitions"):
            continue
        if key == "$defs":
            out["$defs"] = (
                {k: _sanitize_json_schema_for_gemini(v) for k, v in value.items()}
                if isinstance(value, dict)
                else value
            )
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _sanitize_json_schema_for_gemini(v) for k, v in value.items()}
            continue
        if key in ("items", "additionalProperties", "not") and isinstance(value, dict):
            out[key] = _sanitize_json_schema_for_gemini(value)
            continue
        if key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [_sanitize_json_schema_for_gemini(v) for v in value]
            continue
        out[key] = value
    return out


def anthropic_tools_to_gemini(tools: Optional[List[Any]]) -> List[types.FunctionDeclaration]:
    """Convert Anthropic tools to Gemini FunctionDeclaration list.

    Uses parameters_json_schema so Claude Code's richer JSON Schemas (propertyNames,
    exclusiveMinimum, additionalProperties objects, etc.) are accepted by the SDK.
    """
    if not tools:
        return []
    declarations: List[types.FunctionDeclaration] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # OpenAI-shaped tool passthrough
        if isinstance(t.get("function"), dict):
            func_def = t["function"]
            name = func_def.get("name")
            if not name:
                continue
            parameters = func_def.get("parameters")
            kwargs: Dict[str, Any] = {"name": name}
            if func_def.get("description") is not None:
                kwargs["description"] = func_def.get("description")
            if isinstance(parameters, dict):
                kwargs["parameters_json_schema"] = _sanitize_json_schema_for_gemini(parameters)
            elif parameters is not None:
                kwargs["parameters_json_schema"] = parameters
            declarations.append(types.FunctionDeclaration(**kwargs))
            continue

        # Anthropic custom tool: name + input_schema
        name = t.get("name")
        if not name:
            # Built-in Anthropic tools without schemas (bash_*, web_search_*) — skip for Gemini
            continue
        schema = t.get("input_schema") if t.get("input_schema") is not None else t.get("parameters")
        kwargs = {"name": name}
        if t.get("description") is not None:
            kwargs["description"] = t.get("description")
        if isinstance(schema, dict):
            kwargs["parameters_json_schema"] = _sanitize_json_schema_for_gemini(schema)
        elif schema is not None:
            kwargs["parameters_json_schema"] = schema
        declarations.append(types.FunctionDeclaration(**kwargs))
    return declarations


def anthropic_tool_choice_to_gemini(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None
    mode = None
    allowed_functions = None
    if isinstance(tool_choice, str):
        low = tool_choice.lower()
        if low == "none":
            mode = "NONE"
        elif low == "auto":
            mode = "AUTO"
        elif low in ("any", "required"):
            mode = "ANY"
    elif isinstance(tool_choice, dict):
        t = (tool_choice.get("type") or "").lower()
        if t == "auto":
            mode = "AUTO"
        elif t == "any":
            mode = "ANY"
        elif t == "none":
            mode = "NONE"
        elif t == "tool":
            name = tool_choice.get("name") or ""
            if name:
                mode = "ANY"
                allowed_functions = [name]
        elif t == "function":
            name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
            if name:
                mode = "ANY"
                allowed_functions = [name]
    if not mode:
        return None
    config_dict: Dict[str, Any] = {"mode": mode}
    if allowed_functions:
        config_dict["allowed_function_names"] = allowed_functions
    return {"function_calling_config": config_dict}


# Claude Code / Anthropic effort labels → Gemini thinking_budget tokens.
# Gemini only exposes thinking_budget (int) + include_thoughts, not effort tiers.
_EFFORT_TO_THINKING_BUDGET: Dict[str, int] = {
    "none": 0,
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 32768,
    # aliases
    "ultrathink": 32768,
    "ultra": 32768,
}


def _effort_label_to_budget(label: Any) -> Optional[int]:
    """Map effort-like string labels to a Gemini thinking_budget, or None if unknown."""
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    if not key:
        return None
    if key in _EFFORT_TO_THINKING_BUDGET:
        return _EFFORT_TO_THINKING_BUDGET[key]
    # Accept "effort=high" / "thinking=high" style leftovers
    if "=" in key:
        return _effort_label_to_budget(key.split("=", 1)[-1])
    return None


def _thinking_budget_from_anthropic(thinking: Any) -> Optional[int]:
    """Return thinking budget tokens if explicitly configured, else None.

    Accepts Anthropic/Claude Code shapes:
      - true / false
      - "enabled" / "disabled" / "adaptive"
      - effort labels: low | medium | high | xhigh | max | minimal | none
      - {"type": "enabled", "budget_tokens": N}
      - {"type": "enabled", "effort": "high"}  (Claude Code style)
      - {"type": "adaptive"} / {"type": "disabled"}
    """
    if thinking is None:
        return None
    if isinstance(thinking, bool):
        return _EFFORT_TO_THINKING_BUDGET["medium"] if thinking else 0
    if isinstance(thinking, (int, float)) and not isinstance(thinking, bool):
        try:
            return max(0, int(thinking))
        except (TypeError, ValueError):
            return None
    if isinstance(thinking, str):
        low = thinking.strip().lower()
        if low in ("disabled", "none", "false", "off"):
            return 0
        if low in ("enabled", "true", "on", "adaptive"):
            return None  # use model defaults with include_thoughts
        effort_budget = _effort_label_to_budget(low)
        if effort_budget is not None:
            return effort_budget
        return None
    if isinstance(thinking, dict):
        ttype = (thinking.get("type") or "").lower()
        if ttype in ("disabled", "none", "off"):
            return 0
        if ttype == "adaptive":
            # Prefer explicit effort/budget if present; otherwise leave default.
            pass

        # Explicit numeric budget wins.
        budget = thinking.get("budget_tokens")
        if budget is None:
            budget = thinking.get("budget")
        if budget is not None:
            try:
                return max(0, int(budget))
            except (TypeError, ValueError):
                pass

        # Effort label on the thinking object (Claude Code / extended thinking).
        for key in ("effort", "level", "thinking_effort", "reasoning_effort"):
            if key in thinking:
                effort_budget = _effort_label_to_budget(thinking.get(key))
                if effort_budget is not None:
                    return effort_budget

        if ttype in ("enabled", "true", "on", ""):
            # enabled without budget/effort → medium default
            if ttype in ("enabled", "true", "on"):
                return _EFFORT_TO_THINKING_BUDGET["medium"]
            return None
        if ttype == "adaptive":
            return None
        # Unknown type but may still carry effort-only payload handled above
        return None
    return None


def _effort_from_request(request: Any) -> Optional[int]:
    """Pick up top-level effort fields Claude Code / clients may send outside thinking."""
    for attr in (
        "effort",
        "thinking_effort",
        "reasoning_effort",
        "output_config",
    ):
        val = getattr(request, attr, None)
        if val is None and hasattr(request, "model_extra") and isinstance(request.model_extra, dict):
            val = request.model_extra.get(attr)
        if val is None:
            continue
        if isinstance(val, dict):
            # e.g. output_config: {effort: "high"} or {thinking: {type, budget_tokens}}
            for key in ("effort", "level", "thinking_effort", "reasoning_effort"):
                if key in val:
                    budget = _effort_label_to_budget(val.get(key))
                    if budget is not None:
                        return budget
            if "thinking" in val:
                budget = _thinking_budget_from_anthropic(val.get("thinking"))
                if budget is not None:
                    return budget
            if "budget_tokens" in val or "budget" in val:
                budget = _thinking_budget_from_anthropic(val)
                if budget is not None:
                    return budget
            continue
        if isinstance(val, str):
            budget = _effort_label_to_budget(val)
            if budget is not None:
                return budget
            # allow full thinking-string forms
            budget = _thinking_budget_from_anthropic(val)
            if budget is not None:
                return budget
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                continue
    return None


def create_anthropic_generation_config(
    request: Any,
    *,
    base_model_name: str,
    is_grounded_search: bool = False,
    is_nothinking_model: bool = False,
    is_max_thinking_model: bool = False,
) -> Dict[str, Any]:
    """Build Gemini generation config dict from Anthropic Messages request."""
    config: Dict[str, Any] = {}

    if getattr(request, "temperature", None) is not None:
        config["temperature"] = request.temperature
    if getattr(request, "max_tokens", None) is not None:
        config["max_output_tokens"] = request.max_tokens
    if getattr(request, "top_p", None) is not None:
        config["top_p"] = request.top_p
    if getattr(request, "top_k", None) is not None:
        config["top_k"] = request.top_k
    if getattr(request, "stop_sequences", None):
        config["stop_sequences"] = request.stop_sequences

    safety_threshold = "BLOCK_NONE"
    config["safety_settings"] = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_UNSPECIFIED", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold=safety_threshold),
        types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold),
    ]

    system_text = flatten_system_instruction(getattr(request, "system", None))
    if system_text:
        config["system_instruction"] = system_text

    function_declarations = anthropic_tools_to_gemini(getattr(request, "tools", None))
    tools_list: List[Any] = []
    if function_declarations:
        tools_list.append(types.Tool(function_declarations=function_declarations))
    if is_grounded_search:
        tools_list.append(types.Tool(google_search=types.GoogleSearch()))
    if tools_list:
        config["tools"] = tools_list

    tool_config = anthropic_tool_choice_to_gemini(getattr(request, "tool_choice", None))
    if tool_config:
        config["tool_config"] = tool_config

    # Thinking config
    config["thinking_config"] = {"include_thoughts": True}
    if "gemini-2.5-flash-lite" in base_model_name or "image" in base_model_name:
        config["thinking_config"]["include_thoughts"] = False

    # Priority: request.thinking → top-level effort fields → model suffix overrides
    anth_budget = _thinking_budget_from_anthropic(getattr(request, "thinking", None))
    if anth_budget is None:
        anth_budget = _effort_from_request(request)
    if anth_budget is not None:
        config["thinking_config"]["thinking_budget"] = anth_budget
        if anth_budget == 0:
            config["thinking_config"]["include_thoughts"] = False
        else:
            config["thinking_config"]["include_thoughts"] = True

    # Model suffix overrides take precedence
    if is_nothinking_model or is_max_thinking_model:
        if is_nothinking_model:
            budget = 128 if ("gemini-2.5-pro" in base_model_name or "gemini-3-pro" in base_model_name) else 0
        else:
            budget = 32768 if ("gemini-2.5-pro" in base_model_name or "gemini-3-pro" in base_model_name) else 24576
        config["thinking_config"]["thinking_budget"] = budget
        if budget == 0:
            config["thinking_config"]["include_thoughts"] = False
        else:
            config["thinking_config"]["include_thoughts"] = True

    return config


# ── response conversion ─────────────────────────────────────────────────────


def map_gemini_finish_to_stop_reason(
    raw_finish: Any,
    *,
    has_tool_calls: bool = False,
) -> str:
    if has_tool_calls:
        return "tool_use"
    if raw_finish is None:
        return "end_turn"
    if hasattr(raw_finish, "name"):
        reason = raw_finish.name.upper()
    else:
        reason = str(raw_finish).upper()
    if reason in ("TOOL_CODE", "FUNCTION_CALL"):
        return "tool_use"
    if reason == "MAX_TOKENS":
        return "max_tokens"
    if reason == "SAFETY":
        return "refusal"
    if reason == "STOP":
        return "end_turn"
    return "end_turn"


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"raw": raw}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {"result": str(raw)}


def _extract_usage(gemini_response_obj: Any) -> Dict[str, int]:
    usage = {"input_tokens": 0, "output_tokens": 0}
    um = getattr(gemini_response_obj, "usage_metadata", None)
    if um is None:
        return usage
    if hasattr(um, "prompt_token_count") and um.prompt_token_count is not None:
        usage["input_tokens"] = int(um.prompt_token_count)
    if hasattr(um, "candidates_token_count") and um.candidates_token_count is not None:
        usage["output_tokens"] = int(um.candidates_token_count)
    elif hasattr(um, "total_token_count") and um.total_token_count is not None:
        total = int(um.total_token_count)
        if usage["input_tokens"] and total > usage["input_tokens"]:
            usage["output_tokens"] = total - usage["input_tokens"]
        else:
            usage["input_tokens"] = total
    return usage


def gemini_response_to_anthropic(
    gemini_response_obj: Any,
    *,
    model: str,
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a full Gemini generate_content response to Anthropic message."""
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    content_blocks: List[Dict[str, Any]] = []
    stop_reason = "end_turn"
    has_tools = False

    candidates = getattr(gemini_response_obj, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        raw_finish = getattr(candidate, "finish_reason", None)

        # Collect tool calls first
        if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    has_tools = True
                    raw_tool_id = (
                        getattr(fc, "id", None)
                        or f"toolu_{uuid.uuid4().hex[:24]}"
                    )
                    tool_id = _encode_tool_call_id_with_thought_signature(
                        raw_tool_id, getattr(part, "thought_signature", None)
                    )
                    args = getattr(fc, "args", None) or {}
                    if not isinstance(args, dict):
                        try:
                            args = dict(args)
                        except (TypeError, ValueError):
                            args = _parse_tool_arguments(args)
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": getattr(fc, "name", "") or "",
                            "input": args,
                        }
                    )

        reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
        # Insert thinking/text before tool_use for more natural order
        text_blocks: List[Dict[str, Any]] = []
        if reasoning_str:
            text_blocks.append({"type": "thinking", "thinking": reasoning_str})
        if normal_content_str:
            text_blocks.append({"type": "text", "text": normal_content_str})
        # Rebuild: thinking/text then tool_use
        tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        content_blocks = text_blocks + tool_blocks

        stop_reason = map_gemini_finish_to_stop_reason(raw_finish, has_tool_calls=has_tools)
    elif hasattr(gemini_response_obj, "text") and gemini_response_obj.text is not None:
        content_blocks.append({"type": "text", "text": str(gemini_response_obj.text)})
    else:
        content_blocks.append({"type": "text", "text": ""})

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = _extract_usage(gemini_response_obj)
    if usage["output_tokens"] <= 0:
        approx = 0
        for b in content_blocks:
            if b.get("type") == "text":
                approx += max(1, (len(b.get("text") or "") + 3) // 4)
            elif b.get("type") == "thinking":
                approx += max(1, (len(b.get("thinking") or "") + 3) // 4)
            elif b.get("type") == "tool_use":
                try:
                    approx += max(1, (len(json.dumps(b.get("input") or {}, ensure_ascii=False)) + 3) // 4)
                except (TypeError, ValueError):
                    pass
        usage["output_tokens"] = approx

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
        },
    }


# ── errors ──────────────────────────────────────────────────────────────────


def anthropic_error(
    message: str,
    *,
    status: int = 500,
    err_type: str = "api_error",
) -> Dict[str, Any]:
    if status == 401:
        err_type = "authentication_error"
    elif status == 403:
        err_type = "permission_error"
    elif status == 404:
        err_type = "not_found_error"
    elif status == 429:
        err_type = "rate_limit_error"
    elif status == 400:
        err_type = "invalid_request_error"
    return {
        "type": "error",
        "error": {
            "type": err_type,
            "message": message,
        },
    }


# ── count_tokens ────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def count_tokens_for_request(request: Any) -> Dict[str, Any]:
    total = 0
    if getattr(request, "system", None) is not None:
        total += estimate_tokens(_as_text(request.system))
    for m in getattr(request, "messages", None) or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        total += estimate_tokens(_as_text(content))
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    total += estimate_tokens(str(b.get("name") or ""))
                    try:
                        total += estimate_tokens(json.dumps(b.get("input") or {}, ensure_ascii=False))
                    except (TypeError, ValueError):
                        pass
    for t in getattr(request, "tools", None) or []:
        if isinstance(t, dict):
            total += estimate_tokens(str(t.get("name") or ""))
            total += estimate_tokens(str(t.get("description") or ""))
            schema = t.get("input_schema") or t.get("parameters") or {}
            try:
                total += estimate_tokens(json.dumps(schema, ensure_ascii=False))
            except (TypeError, ValueError):
                pass
    return {"input_tokens": total}


# ── SSE helpers ─────────────────────────────────────────────────────────────


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def anthropic_stream_message_start(
    *, message_id: str, model: str, input_tokens: int = 0
) -> str:
    return _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                },
            },
        },
    )


def anthropic_stream_block_start_text(index: int) -> str:
    return _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )


def anthropic_stream_block_start_thinking(index: int) -> str:
    return _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    )


def anthropic_stream_block_start_tool(index: int, *, tool_id: str, name: str) -> str:
    return _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        },
    )


def anthropic_stream_text_delta(index: int, text: str) -> str:
    return _sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def anthropic_stream_thinking_delta(index: int, text: str) -> str:
    return _sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    )


def anthropic_stream_input_json_delta(index: int, partial_json: str) -> str:
    return _sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def anthropic_stream_block_stop(index: int) -> str:
    return _sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


def anthropic_stream_message_delta(
    *,
    stop_reason: str,
    output_tokens: int = 0,
    input_tokens: Optional[int] = None,
) -> str:
    usage: Dict[str, Any] = {"output_tokens": int(output_tokens or 0)}
    if input_tokens is not None:
        usage["input_tokens"] = int(input_tokens or 0)
    return _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": usage,
        },
    )


def anthropic_stream_message_stop() -> str:
    return _sse_event("message_stop", {"type": "message_stop"})


def anthropic_stream_error(message: str, err_type: str = "api_error") -> str:
    return _sse_event(
        "error",
        {
            "type": "error",
            "error": {"type": err_type, "message": message},
        },
    )


def anthropic_stream_ping() -> str:
    return _sse_event("ping", {"type": "ping"})


# ── stream assembler ────────────────────────────────────────────────────────


class GeminiAnthropicStreamAssembler:
    """
    Stateful converter: Gemini stream chunks → Anthropic SSE event strings.

    Emits content_block open/delta/close for thinking, text, and tool_use as
    chunks arrive. Call finish() after the stream ends for message_delta/stop.
    """

    def __init__(self, *, message_id: str, model: str, input_tokens: int = 0):
        self.message_id = message_id
        self.model = model
        self.input_tokens = input_tokens
        self.next_index = 0
        self.open_kind: Optional[str] = None  # "thinking" | "text" | "tool_use"
        self.open_index: Optional[int] = None
        self.stop_reason = "end_turn"
        self.has_tool_calls = False
        self.output_tokens = 0
        self._started = False
        self._tool_ids_emitted: set = set()

    def start_events(self) -> List[str]:
        self._started = True
        return [
            anthropic_stream_message_start(
                message_id=self.message_id,
                model=self.model,
                input_tokens=self.input_tokens,
            ),
            anthropic_stream_ping(),
        ]

    def _close_open(self) -> List[str]:
        events: List[str] = []
        if self.open_kind is not None and self.open_index is not None:
            events.append(anthropic_stream_block_stop(self.open_index))
            self.open_kind = None
            self.open_index = None
        return events

    def _ensure_block(self, kind: str) -> Tuple[List[str], int]:
        events: List[str] = []
        if self.open_kind == kind and self.open_index is not None:
            return events, self.open_index
        events.extend(self._close_open())
        idx = self.next_index
        self.next_index += 1
        self.open_kind = kind
        self.open_index = idx
        if kind == "thinking":
            events.append(anthropic_stream_block_start_thinking(idx))
        elif kind == "text":
            events.append(anthropic_stream_block_start_text(idx))
        return events, idx

    def process_chunk(self, chunk: Any) -> List[str]:
        events: List[str] = []
        if not self._started:
            events.extend(self.start_events())

        # Update usage if present on chunk
        usage = _extract_usage(chunk)
        if usage["input_tokens"]:
            self.input_tokens = usage["input_tokens"]
        if usage["output_tokens"]:
            self.output_tokens = usage["output_tokens"]

        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            return events
        candidate = candidates[0]

        raw_finish = getattr(candidate, "finish_reason", None)
        if raw_finish is not None:
            self.stop_reason = map_gemini_finish_to_stop_reason(
                raw_finish, has_tool_calls=self.has_tool_calls
            )

        # Tool calls
        if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    events.extend(self._close_open())
                    self.has_tool_calls = True
                    self.stop_reason = "tool_use"
                    raw_tool_id = (
                        getattr(fc, "id", None)
                        or f"toolu_{uuid.uuid4().hex[:24]}"
                    )
                    tool_id = _encode_tool_call_id_with_thought_signature(
                        raw_tool_id, getattr(part, "thought_signature", None)
                    )
                    if tool_id in self._tool_ids_emitted:
                        continue
                    self._tool_ids_emitted.add(tool_id)
                    idx = self.next_index
                    self.next_index += 1
                    name = getattr(fc, "name", "") or ""
                    events.append(
                        anthropic_stream_block_start_tool(idx, tool_id=tool_id, name=name)
                    )
                    args = getattr(fc, "args", None)
                    if args is not None:
                        try:
                            if isinstance(args, dict):
                                args_json = json.dumps(args, ensure_ascii=False)
                            else:
                                args_json = json.dumps(dict(args), ensure_ascii=False)
                        except (TypeError, ValueError):
                            args_json = json.dumps(_parse_tool_arguments(args), ensure_ascii=False)
                        events.append(anthropic_stream_input_json_delta(idx, args_json))
                    events.append(anthropic_stream_block_stop(idx))
                    # tool block is fully closed; leave open_kind unset
                    continue

        reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
        if reasoning_str:
            open_events, idx = self._ensure_block("thinking")
            events.extend(open_events)
            events.append(anthropic_stream_thinking_delta(idx, reasoning_str))
            self.output_tokens += max(1, (len(reasoning_str) + 3) // 4)
        if normal_content_str:
            open_events, idx = self._ensure_block("text")
            events.extend(open_events)
            events.append(anthropic_stream_text_delta(idx, normal_content_str))
            self.output_tokens += max(1, (len(normal_content_str) + 3) // 4)

        return events

    def finish(self) -> List[str]:
        events: List[str] = []
        if not self._started:
            events.extend(self.start_events())
        events.extend(self._close_open())
        if self.has_tool_calls:
            self.stop_reason = "tool_use"
        events.append(
            anthropic_stream_message_delta(
                stop_reason=self.stop_reason,
                output_tokens=self.output_tokens,
                input_tokens=self.input_tokens,
            )
        )
        events.append(anthropic_stream_message_stop())
        return events

    def error_close(self, message: str, err_type: str = "api_error") -> List[str]:
        """Emit error then close the stream with message_delta/stop if started."""
        events: List[str] = []
        if not self._started:
            events.extend(self.start_events())
        events.append(anthropic_stream_error(message, err_type=err_type))
        events.extend(self.finish())
        return events
