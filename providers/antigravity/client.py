"""Antigravity Cloud Code provider implementation.

Proxies requests to Google's Antigravity Cloud Code API, which provides
access to Claude and Gemini models via the Antigravity IDE backend.
The API uses Google Generative AI format wrapped in a Cloud Code envelope
and streams responses via SSE.

Supports multi-account mode with automatic rotation on rate limits
(via accounts JSON managed by manage_accounts.py), or single-token
mode via ANTIGRAVITY_OAUTH_TOKEN for backward compatibility.
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

# Antigravity Cloud Code API endpoints (in fallback order)
ANTIGRAVITY_ENDPOINTS = [
    "https://daily-cloudcode-pa.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

# Required headers for Antigravity API
_ANTIGRAVITY_HEADERS = {
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    ),
}

# Max retry attempts across accounts
MAX_ACCOUNT_RETRIES = 5


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
    ):
        super().__init__(config)
        self._api_key = config.api_key  # Fallback single OAuth2 token
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

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()

    def _build_headers(
        self, token: str, accept: str = "text/event-stream"
    ) -> dict[str, str]:
        """Build request headers with OAuth token."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity/1.11.5 proxy/free-claude-code",
            **_ANTIGRAVITY_HEADERS,
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
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(
                request, input_tokens, request_id
            ):
                yield event

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
    ) -> AsyncIterator[str]:
        """Core streaming implementation with multi-account retry."""
        tag = "ANTIGRAVITY"
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

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
                response = await self._try_stream_with_rotation(payload, model)

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

    async def _try_stream_with_rotation(
        self, payload: dict, model: str
    ) -> httpx.Response:
        """Try to stream with account rotation on rate limits.

        Multi-account mode: picks accounts, retries on 429/401.
        Single-token mode: tries the single token directly.
        """
        if not self._account_manager.has_accounts:
            # Single-token fallback
            return await self._try_endpoints(self._api_key, payload)

        max_attempts = min(MAX_ACCOUNT_RETRIES, self._account_manager.account_count + 1)

        for attempt in range(max_attempts):
            account = self._account_manager.pick_account(model)

            if account is None:
                if self._account_manager.all_rate_limited(model):
                    wait = self._account_manager.get_min_wait_seconds(model)
                    raise RuntimeError(
                        f"All accounts rate-limited for {model}. "
                        f"Next available in {wait:.0f}s"
                    )
                raise RuntimeError("No accounts available")

            try:
                token = await account.get_access_token()
                return await self._try_endpoints(token, payload)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    reset_s = self._parse_reset_seconds(e)
                    self._account_manager.mark_rate_limited(
                        account.email, model, reset_s
                    )
                    logger.info(
                        "ANTIGRAVITY_429: {} rate-limited, trying next account "
                        "(attempt {}/{})",
                        account.email,
                        attempt + 1,
                        max_attempts,
                    )
                    continue
                if status == 401:
                    self._account_manager.mark_invalid(
                        account.email, f"Auth failed: {status}"
                    )
                    logger.warning(
                        "ANTIGRAVITY_401: {} auth failed, trying next account",
                        account.email,
                    )
                    continue
                raise

        raise RuntimeError(
            f"Exhausted all {max_attempts} retry attempts across accounts"
        )

    async def _try_endpoints(self, token: str, payload: dict) -> httpx.Response:
        """Try each endpoint in fallback order, return first successful stream."""
        headers = self._build_headers(token)
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

                # For 429/401, raise immediately to trigger account rotation
                if response.status_code in (401, 429):
                    raise last_error

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                logger.warning(
                    "ANTIGRAVITY_CONNECT_ERROR: {} {}: {}",
                    endpoint,
                    type(e).__name__,
                    e,
                )
                last_error = e

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
