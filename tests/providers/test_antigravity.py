"""Tests for Antigravity Cloud Code provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providers.antigravity import AntigravityProvider
from providers.antigravity.request import (
    ANTIGRAVITY_SYSTEM_INSTRUCTION,
    DEFAULT_PROJECT_ID,
    build_cloudcode_request,
    convert_anthropic_to_google,
)
from providers.base import ProviderConfig


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name="read_file", description="Read a file", input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }


class MockThinking:
    def __init__(self, budget_tokens=16000):
        self.budget_tokens = budget_tokens
        self.enabled = True


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "claude-sonnet-4-5-thinking"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 4096
        self.temperature = 0.5
        self.top_p = 0.9
        self.top_k = None
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MockThinking()
        for k, v in kwargs.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────
# Request converter tests
# ──────────────────────────────────────────────────────────────────────


def test_convert_anthropic_to_google_basic():
    """Basic conversion produces correct Google GenAI structure."""
    req = MockRequest()
    result = convert_anthropic_to_google(req)

    assert "contents" in result
    assert "generationConfig" in result
    assert len(result["contents"]) == 1
    assert result["contents"][0]["role"] == "user"
    assert result["contents"][0]["parts"] == [{"text": "Hello"}]


def test_convert_anthropic_to_google_system_instruction():
    """System prompt is converted to systemInstruction."""
    req = MockRequest(system="Be helpful")
    result = convert_anthropic_to_google(req)

    assert "systemInstruction" in result
    assert result["systemInstruction"]["parts"] == [{"text": "Be helpful"}]


def test_convert_anthropic_to_google_system_list():
    """System prompt as list of blocks is converted."""
    req = MockRequest(
        system=[{"type": "text", "text": "Rule 1"}, {"type": "text", "text": "Rule 2"}]
    )
    result = convert_anthropic_to_google(req)

    assert len(result["systemInstruction"]["parts"]) == 2


def test_convert_anthropic_to_google_generation_config():
    """Generation config params are mapped correctly."""
    req = MockRequest(max_tokens=2048, temperature=0.7, top_p=0.95, top_k=40)
    result = convert_anthropic_to_google(req)

    gen = result["generationConfig"]
    assert gen["maxOutputTokens"] == 2048
    assert gen["temperature"] == 0.7
    assert gen["topP"] == 0.95
    assert gen["topK"] == 40


def test_convert_anthropic_to_google_thinking_claude():
    """Thinking config for Claude models uses snake_case fields."""
    req = MockRequest(
        model="claude-sonnet-4-5-thinking", thinking=MockThinking(budget_tokens=8000)
    )
    result = convert_anthropic_to_google(req)

    thinking_config = result["generationConfig"]["thinkingConfig"]
    assert thinking_config["include_thoughts"] is True
    assert thinking_config["thinking_budget"] == 8000


def test_convert_anthropic_to_google_thinking_gemini():
    """Thinking config for Gemini models uses camelCase fields."""
    req = MockRequest(
        model="gemini-3-flash", thinking=MockThinking(budget_tokens=12000)
    )
    result = convert_anthropic_to_google(req)

    thinking_config = result["generationConfig"]["thinkingConfig"]
    assert thinking_config["includeThoughts"] is True
    assert thinking_config["thinkingBudget"] == 12000


def test_convert_anthropic_to_google_gemini3_auto_thinking():
    """Gemini 3+ automatically enables thinking even without 'thinking' in name."""
    req = MockRequest(model="gemini-3-flash", thinking=None)
    result = convert_anthropic_to_google(req)

    assert "thinkingConfig" in result["generationConfig"]
    assert result["generationConfig"]["thinkingConfig"]["includeThoughts"] is True


def test_convert_anthropic_to_google_tools():
    """Tools are converted to functionDeclarations format."""
    tools = [MockTool(name="read_file", description="Read file")]
    req = MockRequest(tools=tools)
    result = convert_anthropic_to_google(req)

    assert "tools" in result
    assert len(result["tools"]) == 1
    func_decls = result["tools"][0]["functionDeclarations"]
    assert len(func_decls) == 1
    assert func_decls[0]["name"] == "read_file"
    assert func_decls[0]["description"] == "Read file"


def test_convert_anthropic_to_google_role_mapping():
    """Assistant role is mapped to 'model' for Google GenAI."""
    req = MockRequest(
        messages=[
            MockMessage("user", "Hi"),
            MockMessage("assistant", "Hello!"),
            MockMessage("user", "Bye"),
        ]
    )
    result = convert_anthropic_to_google(req)

    assert result["contents"][0]["role"] == "user"
    assert result["contents"][1]["role"] == "model"
    assert result["contents"][2]["role"] == "user"


def test_convert_anthropic_to_google_session_id():
    """Session ID is derived from first user message."""
    req = MockRequest()
    result = convert_anthropic_to_google(req)

    assert "sessionId" in result
    assert isinstance(result["sessionId"], str)
    assert len(result["sessionId"]) == 16


def test_convert_content_thinking_blocks():
    """Thinking blocks in content are converted with thought=True."""
    content = [
        {"type": "thinking", "thinking": "Let me think...", "signature": "abc123"},
        {"type": "text", "text": "The answer is 42."},
    ]
    req = MockRequest(messages=[MockMessage("assistant", content)])
    result = convert_anthropic_to_google(req)

    parts = result["contents"][0]["parts"]
    assert parts[0]["thought"] is True
    assert parts[0]["text"] == "Let me think..."
    assert parts[0]["thoughtSignature"] == "abc123"
    assert parts[1]["text"] == "The answer is 42."


def test_convert_content_tool_use():
    """Tool use blocks are converted to functionCall."""
    tool_use_content = [
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "read_file",
            "input": {"path": "test.py"},
        },
    ]
    tool_result_content = [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "result",
        },
    ]
    req = MockRequest(
        messages=[
            MockMessage("assistant", tool_use_content),
            MockMessage("user", tool_result_content),
        ]
    )
    result = convert_anthropic_to_google(req)

    parts = result["contents"][0]["parts"]
    assert "functionCall" in parts[0]
    assert parts[0]["functionCall"]["name"] == "read_file"
    assert parts[0]["functionCall"]["args"] == {"path": "test.py"}


def test_convert_content_tool_result():
    """Tool result blocks are converted to functionResponse."""
    tool_use_content = [
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "read_file",
            "input": {"path": "test.py"},
        },
    ]
    tool_result_content = [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "File contents here",
        },
    ]
    req = MockRequest(
        messages=[
            MockMessage("assistant", tool_use_content),
            MockMessage("user", tool_result_content),
        ]
    )
    result = convert_anthropic_to_google(req)

    parts = result["contents"][1]["parts"]
    assert "functionResponse" in parts[0]
    assert parts[0]["functionResponse"]["name"] == "toolu_123"
    assert parts[0]["functionResponse"]["response"]["result"] == "File contents here"


# ──────────────────────────────────────────────────────────────────────
# Cloud Code envelope tests
# ──────────────────────────────────────────────────────────────────────


def test_build_cloudcode_request_envelope():
    """Cloud Code request has correct envelope structure."""
    req = MockRequest()
    payload = build_cloudcode_request(req)

    assert payload["project"] == DEFAULT_PROJECT_ID
    assert payload["model"] == "claude-sonnet-4-5-thinking"
    assert payload["userAgent"] == "antigravity"
    assert payload["requestType"] == "agent"
    assert payload["requestId"].startswith("agent-")
    assert "request" in payload


def test_build_cloudcode_request_system_instruction_prefix():
    """System instruction is prefixed with Antigravity identity."""
    req = MockRequest(system="Custom system prompt")
    payload = build_cloudcode_request(req)

    sys_parts = payload["request"]["systemInstruction"]["parts"]
    # First part is Antigravity identity
    assert ANTIGRAVITY_SYSTEM_INSTRUCTION in sys_parts[0]["text"]
    # Second part is the [ignore] wrapper
    assert "[ignore]" in sys_parts[1]["text"]
    # Third part is the user's original system prompt
    assert sys_parts[2]["text"] == "Custom system prompt"


def test_build_cloudcode_request_custom_project_id():
    """Custom project ID is used when provided."""
    req = MockRequest()
    payload = build_cloudcode_request(req, project_id="custom-project-123")

    assert payload["project"] == "custom-project-123"


# ──────────────────────────────────────────────────────────────────────
# Provider tests
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def antigravity_config():
    return ProviderConfig(
        api_key="ya29.test_oauth_token",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter."""
    with patch("providers.antigravity.client.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _slot():
            yield

        instance.concurrency_slot = _slot
        yield instance


@pytest.fixture
def provider(antigravity_config, tmp_path):
    return AntigravityProvider(
        antigravity_config, accounts_path=tmp_path / "no_accounts.json"
    )


def test_provider_init(antigravity_config):
    """Provider initializes with OAuth token from config."""
    provider = AntigravityProvider(antigravity_config)
    assert provider._api_key == "ya29.test_oauth_token"


def test_provider_headers(provider):
    """Provider headers include OAuth token and Antigravity-specific headers."""
    headers = provider._build_headers("ya29.test_oauth_token")
    assert headers["Authorization"] == "Bearer ya29.test_oauth_token"
    assert "X-Goog-Api-Client" in headers
    assert "Client-Metadata" in headers
    assert "antigravity" in headers["User-Agent"]


def _make_sse_line(parts, finish_reason=None, usage=None):
    """Helper to build a Google GenAI SSE data line."""
    data = {
        "response": {
            "candidates": [
                {
                    "content": {"parts": parts},
                    **({"finishReason": finish_reason} if finish_reason else {}),
                }
            ],
            **({"usageMetadata": usage} if usage else {}),
        }
    }
    return f"data: {json.dumps(data)}"


class MockAsyncLineIterator:
    """Mock httpx streaming response."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


@pytest.mark.asyncio
async def test_stream_response_text(provider):
    """Streaming text response produces correct Anthropic SSE events."""
    req = MockRequest()

    lines = [
        _make_sse_line([{"text": "Hello "}]),
        _make_sse_line(
            [{"text": "World"}],
            finish_reason="STOP",
            usage={"promptTokenCount": 100, "candidatesTokenCount": 10},
        ),
    ]

    mock_response = MockAsyncLineIterator(lines)

    with patch.object(
        provider, "_try_stream_with_rotation", new_callable=AsyncMock
    ) as mock_try:
        mock_try.return_value = mock_response
        events = [e async for e in provider.stream_response(req)]

    assert any("message_start" in e for e in events)
    assert any("Hello " in e for e in events)
    assert any("World" in e for e in events)
    assert any("message_delta" in e for e in events)
    assert any("message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_thinking(provider):
    """Streaming thinking blocks produces thinking_delta events."""
    req = MockRequest()

    lines = [
        _make_sse_line([{"text": "Let me think...", "thought": True}]),
        _make_sse_line([{"text": "The answer is 42."}], finish_reason="STOP"),
    ]

    mock_response = MockAsyncLineIterator(lines)

    with patch.object(
        provider, "_try_stream_with_rotation", new_callable=AsyncMock
    ) as mock_try:
        mock_try.return_value = mock_response
        events = [e async for e in provider.stream_response(req)]

    assert any("thinking_delta" in e and "Let me think" in e for e in events)
    assert any("text_delta" in e and "42" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_tool_call(provider):
    """Streaming tool calls produces tool_use blocks."""
    req = MockRequest()

    lines = [
        _make_sse_line(
            [{"functionCall": {"name": "read_file", "args": {"path": "test.py"}}}],
            finish_reason="STOP",
        ),
    ]

    mock_response = MockAsyncLineIterator(lines)

    with patch.object(
        provider, "_try_stream_with_rotation", new_callable=AsyncMock
    ) as mock_try:
        mock_try.return_value = mock_response
        events = [e async for e in provider.stream_response(req)]

    tool_events = [e for e in events if "tool_use" in e]
    assert len(tool_events) > 0
    assert any("read_file" in e for e in tool_events)


@pytest.mark.asyncio
async def test_stream_response_error(provider):
    """Connection errors produce error events and complete the stream."""
    req = MockRequest()

    with patch.object(
        provider, "_try_stream_with_rotation", new_callable=AsyncMock
    ) as mock_try:
        mock_try.side_effect = ConnectionError("Failed to connect")
        events = [e async for e in provider.stream_response(req)]

    assert any("message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_empty_produces_space(provider):
    """Empty stream still produces at least one text block with space."""
    req = MockRequest()

    mock_response = MockAsyncLineIterator([])

    with patch.object(
        provider, "_try_stream_with_rotation", new_callable=AsyncMock
    ) as mock_try:
        mock_try.return_value = mock_response
        events = [e async for e in provider.stream_response(req)]

    # Should have message_start -> text_block -> message_delta -> message_stop
    assert any("message_start" in e for e in events)
    assert any("message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_cleanup(antigravity_config):
    """Cleanup closes the HTTP client."""
    p = AntigravityProvider(antigravity_config)
    with patch.object(p._client, "aclose", new_callable=AsyncMock) as mock_close:
        await p.cleanup()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_rotation_raises_when_all_rate_limited(tmp_path, antigravity_config):
    """When all accounts are rate-limited the loop immediately raises."""
    provider = AntigravityProvider(
        antigravity_config, accounts_path=tmp_path / "no.json"
    )
    mgr = MagicMock()
    mgr.has_accounts = True
    mgr.account_count = 1

    # pick_account returns None indicating all accounts rate limited
    mgr.pick_account.return_value = None
    mgr.get_min_wait_seconds.return_value = 2.0

    provider._account_manager = mgr

    with pytest.raises(RuntimeError, match="All accounts rate-limited for claude"):
        await provider._try_stream_with_rotation({"project": "p"}, "claude")


@pytest.mark.asyncio
async def test_rotation_max_attempts_raises(tmp_path, antigravity_config):
    """After max_attempts the loop gives up with a descriptive error."""
    provider = AntigravityProvider(
        antigravity_config, accounts_path=tmp_path / "no.json"
    )

    mgr = MagicMock()
    mgr.has_accounts = True
    mgr.account_count = 2

    mock_account = MagicMock()
    mock_account.email = "test@example.com"
    mock_account.project_id = ""
    mock_account.get_access_token = AsyncMock(return_value="tok")

    mgr.pick_account.return_value = mock_account
    mgr.get_min_wait_seconds.return_value = 30.0

    provider._account_manager = mgr

    with patch.object(provider, "_try_endpoints", new_callable=AsyncMock) as mock_try:
        # always raise 429
        err = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=MagicMock(status_code=429, headers={}, text=""),
        )
        mock_try.side_effect = err

        with pytest.raises(RuntimeError, match="gave up after 5 attempts"):
            await provider._try_stream_with_rotation({"project": "p"}, "claude")

        assert mock_try.call_count == 5
