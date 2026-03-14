"""Antigravity Cloud Code provider implementation.

Proxies requests to Google's Antigravity Cloud Code API, which provides
access to Claude and Gemini models via the Antigravity IDE backend.
The API uses Google Generative AI format wrapped in a Cloud Code envelope
and streams responses via SSE.

Supports multi-account mode with automatic rotation on rate limits
(via accounts JSON managed by manage_accounts.py), or single-token
mode via ANTIGRAVITY_OAUTH_TOKEN for backward compatibility.

When ANTHROPIC_API_KEY is set, exhausted accounts automatically fall back
to the real Anthropic API (Claude Pro subscription passthrough).
"""

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.common import (
    SSEBuilder,
    append_request_id,
    get_user_facing_error_message,
    map_error,
)
from providers.rate_limit import GlobalRateLimiter

from .account_manager import DEFAULT_ACCOUNTS_PATH, AccountManager
from .request import build_cloudcode_request

# Anthropic API endpoint for Claude Pro fallback
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

# Antigravity Cloud Code API endpoints (in fallback order)
ANTIGRAVITY_ENDPOINTS = [
    "https://daily-cloudcode-pa.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

# Version string matching real Cloud Code extension
_PLUGIN_VERSION = "0.1"


def _build_client_metadata(project_id: str = "") -> str:
    """Build Client-Metadata JSON matching the real Cloud Code extension.

    The extension's buildClientMetadata() method returns:
    {ideType, ideVersion, platform, pluginVersion, updateChannel,
     duetProject, pluginType, ideName}
    """
    return json.dumps(
        {
            "ideType": "VSCODE",
            "ideVersion": "1.100.0",
            "platform": "WINDOWS_AMD64",
            "pluginVersion": _PLUGIN_VERSION,
            "updateChannel": "",
            "duetProject": project_id,
            "pluginType": "CLOUD_CODE",
            "ideName": "Visual Studio Code",
        }
    )


class AntigravityProvider(BaseProvider):
    """Antigravity Cloud Code provider using direct HTTP streaming.

    Unlike OpenAI-compatible providers, Antigravity uses Google's
    Generative AI format wrapped in a Cloud Code envelope. Responses
    are streamed as SSE events containing `candidates[0].content.parts`.

    Supports two modes:
    - Multi-account: loads accounts from JSON, rotates on rate limits
    - Single-token: uses ANTIGRAVITY_OAUTH_TOKEN from config (backward compat)
    """

    def __init__(
        self,
        config: ProviderConfig,
        accounts_path: str | Path = DEFAULT_ACCOUNTS_PATH,
        anthropic_api_key: str = "",
    ):
        super().__init__(config)
        self._api_key = config.api_key  # Fallback single OAuth2 token
        self._anthropic_api_key = anthropic_api_key
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

        # Multi-account manager
        self._account_manager = AccountManager(accounts_path)
        self._account_manager.load()

        if self._account_manager.has_accounts:
            logger.info(
                "ANTIGRAVITY: Multi-account mode ({} accounts)",
                self._account_manager.account_count,
            )
        else:
            logger.info("ANTIGRAVITY: Single-token mode")

        if self._anthropic_api_key:
            logger.info("ANTIGRAVITY: Claude Pro fallback ENABLED")
        else:
            logger.info(
                "ANTIGRAVITY: Claude Pro fallback disabled (no ANTHROPIC_API_KEY)"
            )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()

    def _build_headers(
        self, token: str, accept: str = "text/event-stream", project_id: str = ""
    ) -> dict[str, str]:
        """Build request headers with OAuth token."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"antigravity/{_PLUGIN_VERSION} proxy/free-claude-code",
            "X-Goog-Api-Client": f"google-cloud-sdk vscode_cloudshelleditor/{_PLUGIN_VERSION}",
            "Client-Metadata": _build_client_metadata(project_id),
        }
        if accept != "application/json":
            headers["Accept"] = accept
        return headers

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        api_key: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id, api_key
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        api_key: str | None = None,
    ) -> AsyncIterator[str]:
        """Core streaming implementation with multi-account retry.

        Falls back to direct Anthropic API if all accounts are exhausted
        and an Anthropic API key is available (either from request or config).
        """
        tag = "ANTIGRAVITY"
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        fallback_key = api_key or self._anthropic_api_key

        payload = build_cloudcode_request(request)
        model = payload.get("model", "unknown")
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "{}_STREAM:{} model={} msgs={}",
            tag,
            req_tag,
            model,
            len(payload.get("request", {}).get("contents", [])),
        )

        yield sse.message_start()

        error_occurred = False
        error_message = ""
        finish_reason = "end_turn"
        usage_output_tokens = 0

        async with self._global_rate_limiter.concurrency_slot():
            try:
                try:
                    response = await self._try_stream_with_rotation(payload, model)
                except RuntimeError as e:
                    if not fallback_key:
                        raise
                    # Fallback to Claude Pro API
                    logger.warning(
                        "ANTIGRAVITY_FALLBACK:{} all accounts exhausted, "
                        "switching to direct Anthropic API: {}",
                        req_tag,
                        e,
                    )
                    for event in sse.close_all_blocks():
                        yield event
                    async for line in self._stream_anthropic_fallback(
                        request, fallback_key
                    ):
                        yield line
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    json_text = line[5:].strip()
                    if not json_text:
                        continue

                    try:
                        data = json.loads(json_text)
                    except json.JSONDecodeError:
                        logger.warning(
                            "{}_PARSE_ERROR:{} invalid JSON: {}",
                            tag,
                            req_tag,
                            json_text[:100],
                        )
                        continue

                    # Unwrap Cloud Code envelope
                    inner = data.get("response", data)

                    # Extract usage
                    usage = inner.get("usageMetadata")
                    if usage:
                        usage_output_tokens = usage.get(
                            "candidatesTokenCount", usage_output_tokens
                        )

                    candidates = inner.get("candidates", [])
                    if not candidates:
                        continue

                    first_candidate = candidates[0]
                    content = first_candidate.get("content", {})
                    parts = content.get("parts", [])

                    # Check finish reason
                    candidate_finish = first_candidate.get("finishReason")
                    if candidate_finish:
                        if candidate_finish == "STOP":
                            finish_reason = "end_turn"
                        elif candidate_finish == "MAX_TOKENS":
                            finish_reason = "max_tokens"
                        else:
                            finish_reason = "end_turn"

                    for part in parts:
                        if part.get("thought") is True:
                            text = part.get("text", "")
                            if text:
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(text)

                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            tool_name = fc.get("name", "")
                            tool_args = fc.get("args", {})
                            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"

                            for event in sse.close_content_blocks():
                                yield event

                            block_idx = sse.blocks.allocate_index()

                            if tool_name == "Task" and isinstance(tool_args, dict):
                                tool_args["run_in_background"] = False

                            yield sse.content_block_start(
                                block_idx,
                                "tool_use",
                                id=tool_id,
                                name=tool_name,
                            )
                            yield sse.content_block_delta(
                                block_idx,
                                "input_json_delta",
                                json.dumps(tool_args),
                            )
                            yield sse.content_block_stop(block_idx)

                        elif "text" in part:
                            text = part.get("text", "")
                            if text:
                                for event in sse.ensure_text_block():
                                    yield event
                                yield sse.emit_text_delta(text)

            except Exception as e:
                logger.error("{}_ERROR:{} {}: {}", tag, req_tag, type(e).__name__, e)
                mapped_e = map_error(e)
                error_occurred = True
                error_message = append_request_id(
                    get_user_facing_error_message(
                        mapped_e, read_timeout_s=self._config.http_read_timeout
                    ),
                    request_id,
                )
                for event in sse.close_content_blocks():
                    yield event
                for event in sse.emit_error(error_message):
                    yield event

        # Ensure at least one content block exists
        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_states
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            usage_output_tokens if usage_output_tokens else sse.estimate_output_tokens()
        )
        yield sse.message_delta(finish_reason, output_tokens)
        yield sse.message_stop()

    async def _stream_anthropic_fallback(
        self, request: Any, fallback_key: str
    ) -> AsyncIterator[str]:
        """Fall back to the real Anthropic API (Claude Pro subscription).

        Since the proxy input is already in Anthropic format and the
        Anthropic API returns Anthropic SSE, this is a simple passthrough:
        serialize the original request → POST to api.anthropic.com →
        yield SSE lines unchanged.
        """
        # Exclude internal/proxy fields from the payload
        body = request.model_dump(
            exclude_none=True,
            exclude={"extra_body", "original_model", "resolved_provider_model"},
            by_alias=True,
        )

        # Ensure tools are formatted correctly
        if hasattr(request, "tools") and request.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in request.tools
            ]

        # Thinking / extended thinking
        if hasattr(request, "thinking") and request.thinking:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.thinking.budget_tokens,
            }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": fallback_key,
            "Anthropic-Version": ANTHROPIC_API_VERSION,
            "Accept": "text/event-stream",
        }

        logger.info(
            "ANTHROPIC_FALLBACK: model={} msgs={}",
            request.model,
            len(body["messages"]),
        )

        response = await self._client.send(
            self._client.build_request(
                "POST",
                ANTHROPIC_API_URL,
                headers=headers,
                content=json.dumps(body),
            ),
            stream=True,
        )

        if response.status_code != 200:
            error_body = await response.aread()
            error_text = error_body.decode("utf-8", errors="replace")
            logger.error(
                "ANTHROPIC_FALLBACK_ERROR: status={} error={}",
                response.status_code,
                error_text[:300],
            )
            raise RuntimeError(
                f"Anthropic API error ({response.status_code}): {error_text[:200]}"
            )

        # Pipe SSE stream directly — already in Anthropic format
        async for line in response.aiter_lines():
            yield line + "\n"

    async def _try_stream_with_rotation(
        self, payload: dict, model: str
    ) -> httpx.Response:
        """Try to stream and rotate through accounts on failure.

        Failure cases like 429/503/401 automatically rotate to the next account.
        If all accounts are currently rate-limited, immediately falls back.

        Single-token mode: tries the single token directly (no rotation).
        """
        if not self._account_manager.has_accounts:
            return await self._try_endpoints(self._api_key, payload)

        max_attempts = max(5, self._account_manager.account_count * 2)

        for attempt in range(1, max_attempts + 1):
            account = self._account_manager.pick_account(model)

            if account is None:
                # All accounts are currently rate-limited or invalid.
                wait = self._account_manager.get_min_wait_seconds(model)
                logger.warning(
                    "ANTIGRAVITY: all accounts rate-limited for {}, "
                    "next available in {:.0f}s",
                    model,
                    wait,
                )
                raise RuntimeError(
                    f"All accounts rate-limited for {model} — "
                    f"Next account available in {wait:.0f}s."
                )

            if account.project_id:
                payload["project"] = account.project_id
            else:
                from .request import DEFAULT_PROJECT_ID

                payload["project"] = DEFAULT_PROJECT_ID

            try:
                token = await account.get_access_token()
            except Exception:
                # get_access_token already called mark_invalid() — just rotate
                logger.warning(
                    "ANTIGRAVITY: {} token refresh failed, rotating account",
                    account.email,
                )
                continue

            try:
                return await self._try_endpoints(token, payload)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 503):
                    reset_s = self._parse_reset_seconds(e)
                    self._account_manager.mark_rate_limited(
                        account.email, model, reset_s
                    )
                    logger.info(
                        "ANTIGRAVITY_{}: {} rate-limited, rotating account "
                        "(attempt={})",
                        status,
                        account.email,
                        attempt,
                    )
                    continue
                if status == 401:
                    self._account_manager.mark_invalid(
                        account.email, f"Auth failed: {status}"
                    )
                    logger.warning(
                        "ANTIGRAVITY_401: {} auth failed, rotating account",
                        account.email,
                    )
                    continue
                raise

        # Max attempts exhausted
        wait = self._account_manager.get_min_wait_seconds(model)
        raise RuntimeError(
            f"All accounts rate-limited for {model} — "
            f"gave up after {max_attempts} attempts. "
            f"Next account available in {wait:.0f}s."
        )

    async def _try_endpoints(self, token: str, payload: dict) -> httpx.Response:
        """Try each endpoint in fallback order, return first successful stream.

        Rate-limit (429/503) and auth (401) errors are per-account, not
        per-endpoint, so they are raised immediately for account rotation.
        Only connection errors and non-critical HTTP errors (e.g. 400, 500)
        trigger fallback to the next endpoint.
        """
        headers = self._build_headers(token, project_id=payload.get("project", ""))
        last_error: Exception | None = None

        for endpoint in ANTIGRAVITY_ENDPOINTS:
            url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
            try:
                response = await self._client.send(
                    self._client.build_request(
                        "POST",
                        url,
                        headers=headers,
                        content=json.dumps(payload),
                    ),
                    stream=True,
                )
            except Exception as e:
                # Connection-level failure — try next endpoint
                logger.warning(
                    "ANTIGRAVITY_CONNECT_ERROR: {} {}: {}",
                    endpoint,
                    type(e).__name__,
                    e,
                )
                last_error = e
                continue

            if response.status_code == 200:
                return response

            body = await response.aread()
            error_text = body.decode("utf-8", errors="replace")
            logger.warning(
                "ANTIGRAVITY_ENDPOINT_ERROR: {} status={} error={}",
                endpoint,
                response.status_code,
                error_text[:200],
            )

            last_error = httpx.HTTPStatusError(
                f"API error ({response.status_code}): {error_text[:200]}",
                request=response.request,
                response=response,
            )

            # Rate-limit (429/503) and auth (401) errors are per-account,
            # not per-endpoint — raise immediately so account rotation in
            # _try_stream_with_rotation can handle them properly.
            if response.status_code in (401, 429, 503):
                raise last_error

            # Other HTTP errors (400, 500, etc.) — try next endpoint

        if last_error:
            raise last_error
        raise ConnectionError("Failed to connect to any Antigravity endpoint")

    @staticmethod
    def _parse_reset_seconds(error: httpx.HTTPStatusError) -> float:
        """Parse rate limit reset time from error response. Default 60s."""
        try:
            body = error.response.text
            if "retry-after" in error.response.headers:
                return float(error.response.headers["retry-after"])
            # Try to parse from JSON body
            data = json.loads(body)
            if "error" in data and "retryDelay" in str(data):
                return 60.0
        except Exception:
            pass
        return 60.0
