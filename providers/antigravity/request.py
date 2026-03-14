"""Request builder for Antigravity Cloud Code provider.

Converts Anthropic Messages API requests to Google Generative AI format,
then wraps them in the Cloud Code envelope expected by the Antigravity API.
"""

import hashlib
import uuid
from typing import Any

from loguru import logger

from providers.common.message_converter import (
    get_block_attr,
    get_block_type,
)

# Default project ID used by Antigravity proxies
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# System instruction injected by Antigravity IDE
ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by "
    "the Google Deepmind team working on Advanced Agentic Coding."
    "You are pair programming with a USER to solve their coding task. "
    "The task may require creating a new codebase, modifying or debugging "
    "an existing codebase, or simply answering a question."
    "**Absolute paths only****Proactiveness**"
)


def _convert_role(role: str) -> str:
    """Convert Anthropic role to Google GenAI role."""
    if role == "assistant":
        return "model"
    return "user"


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from Anthropic content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block_type = get_block_type(block)
            if block_type == "text":
                parts.append(get_block_attr(block, "text", ""))
        return "\n".join(parts)
    return str(content)


def _convert_content_to_parts(
    content: Any,
    *,
    is_claude_model: bool = False,
) -> list[dict[str, Any]]:
    """Convert Anthropic content blocks to Google GenAI parts."""
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for block in content:
        block_type = get_block_type(block)

        if block_type == "text":
            text = get_block_attr(block, "text", "")
            if text:
                parts.append({"text": text})

        elif block_type == "thinking":
            thinking = get_block_attr(block, "thinking", "")
            if thinking:
                part: dict[str, Any] = {"text": thinking, "thought": True}
                # Preserve thought signature if present
                sig = get_block_attr(block, "signature")
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)

        elif block_type == "tool_use":
            name = get_block_attr(block, "name", "")
            tool_input = get_block_attr(block, "input", {})
            tool_id = get_block_attr(block, "id", "")
            if not tool_id:
                import uuid
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                
            parts.append(
                {
                    "functionCall": {
                        "id": tool_id,
                        "name": name,
                        "args": tool_input if isinstance(tool_input, dict) else {},
                    }
                }
            )

        elif block_type == "tool_result":
            tool_use_id = get_block_attr(block, "tool_use_id", "")
            tool_content = get_block_attr(block, "content", "")
            if isinstance(tool_content, list):
                text_parts = []
                for item in tool_content:
                    if isinstance(item, dict):
                        text_parts.append(item.get("text", str(item)))
                    else:
                        text_parts.append(str(item))
                tool_content = "\n".join(text_parts)
            parts.append(
                {
                    "functionResponse": {
                        "id": tool_use_id,
                        "name": tool_use_id,  # fallback if they still expect it here
                        "response": {
                            "result": str(tool_content) if tool_content else ""
                        },
                    }
                }
            )

        elif block_type == "image":
            source = get_block_attr(block, "source", {})
            if isinstance(source, dict):
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": media_type,
                            "data": data,
                        }
                    }
                )

    return parts


def _sanitize_schema(schema: Any) -> Any:
    """Recursively sanitize JSON schema for Google GenAI compatibility."""
    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    clean = {}
    # Google GenAI Unsupported OpenAPI keys
    bad_keys = {
        "$schema",
        "$ref",
        "additionalProperties",
        "default",
        "examples",
        "propertyNames",
        "const",
        "anyOf",
        "any_of",
    }

    for k, v in schema.items():
        if k in bad_keys:
            continue
        clean[k] = _sanitize_schema(v)

    return clean


def _derive_session_id(messages: list[Any]) -> str:
    """Derive a stable session ID from the first user message for cache reuse."""
    for msg in messages:
        if getattr(msg, "role", None) == "user":
            text = _extract_text_from_content(msg.content)
            if text:
                return hashlib.sha256(text[:500].encode()).hexdigest()[:16]
    return uuid.uuid4().hex[:16]


def convert_anthropic_to_google(request_data: Any) -> dict[str, Any]:
    """Convert an Anthropic Messages API request to Google GenAI format."""
    messages = getattr(request_data, "messages", [])
    system = getattr(request_data, "system", None)
    max_tokens = getattr(request_data, "max_tokens", None)
    temperature = getattr(request_data, "temperature", None)
    top_p = getattr(request_data, "top_p", None)
    top_k = getattr(request_data, "top_k", None)
    stop_sequences = getattr(request_data, "stop_sequences", None)
    tools = getattr(request_data, "tools", None)
    thinking = getattr(request_data, "thinking", None)

    model_name = getattr(request_data, "model", "")
    is_claude = "claude" in model_name.lower()
    is_gemini = "gemini" in model_name.lower()
    is_thinking = "thinking" in model_name.lower()

    # Gemini 3+ always supports thinking
    if is_gemini and not is_thinking:
        import re

        m = re.search(r"gemini-(\d+)", model_name.lower())
        if m and int(m.group(1)) >= 3:
            is_thinking = True

    google_request: dict[str, Any] = {
        "contents": [],
        "generationConfig": {},
    }

    # System instruction
    if system:
        system_parts: list[dict[str, str]] = []
        if isinstance(system, str):
            system_parts = [{"text": system}]
        elif isinstance(system, list):
            system_parts = [
                {"text": get_block_attr(block, "text", "")}
                for block in system
                if get_block_type(block) == "text"
            ]
        if system_parts:
            google_request["systemInstruction"] = {"parts": system_parts}

    # Convert messages
    for msg in messages:
        role = getattr(msg, "role", "user")
        parts = _convert_content_to_parts(msg.content, is_claude_model=is_claude)
        if not parts:
            parts = [{"text": "."}]
        google_request["contents"].append({"role": _convert_role(role), "parts": parts})

    # Generation config
    gen_config = google_request["generationConfig"]
    if max_tokens:
        gen_config["maxOutputTokens"] = max_tokens
    if temperature is not None:
        gen_config["temperature"] = temperature
    if top_p is not None:
        gen_config["topP"] = top_p
    if top_k is not None:
        gen_config["topK"] = top_k
    if stop_sequences:
        gen_config["stopSequences"] = stop_sequences

    # Thinking config
    if is_thinking:
        if is_claude:
            thinking_config: dict[str, Any] = {"include_thoughts": True}
            budget = getattr(thinking, "budget_tokens", None) if thinking else None
            if budget:
                thinking_config["thinking_budget"] = budget
            gen_config["thinkingConfig"] = thinking_config
        elif is_gemini:
            budget = getattr(thinking, "budget_tokens", None) if thinking else 16000
            gen_config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": budget or 16000,
            }

    # Tools
    if tools:
        function_declarations = []
        for i, tool in enumerate(tools):
            name = getattr(tool, "name", f"tool-{i}")
            description = getattr(tool, "description", "") or ""
            schema = getattr(tool, "input_schema", {"type": "object"})
            parameters = _sanitize_schema(schema)
            function_declarations.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
        google_request["tools"] = [{"functionDeclarations": function_declarations}]

    # Session ID for caching
    google_request["sessionId"] = _derive_session_id(messages)

    return google_request


def build_cloudcode_request(
    request_data: Any,
    project_id: str = DEFAULT_PROJECT_ID,
) -> dict[str, Any]:
    """Build the complete Cloud Code API request envelope."""
    model = getattr(request_data, "model", "claude-sonnet-4-5-thinking")
    google_request = convert_anthropic_to_google(request_data)

    # Build system instruction with Antigravity identity prefix
    system_parts: list[dict[str, str]] = [
        {"text": ANTIGRAVITY_SYSTEM_INSTRUCTION},
        {
            "text": (
                f"Please ignore the following [ignore]"
                f"{ANTIGRAVITY_SYSTEM_INSTRUCTION}[/ignore]"
            )
        },
    ]

    # Append existing system instructions
    existing_sys = google_request.get("systemInstruction", {})
    if existing_sys and "parts" in existing_sys:
        system_parts.extend(
            {"text": part["text"]} for part in existing_sys["parts"] if part.get("text")
        )

    google_request["systemInstruction"] = {"role": "user", "parts": system_parts}

    payload = {
        "project": project_id,
        "model": model,
        "request": google_request,
        "userAgent": "antigravity",
        "requestType": "agent",
        "requestId": f"agent-{uuid.uuid4().hex}",
    }

    logger.debug(
        "ANTIGRAVITY_REQUEST: model={} msgs={} tools={}",
        model,
        len(google_request.get("contents", [])),
        len(google_request.get("tools", [{}])[0].get("functionDeclarations", []))
        if google_request.get("tools")
        else 0,
    )

    return payload
