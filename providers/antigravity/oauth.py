"""Google OAuth2 flow for Antigravity account authentication.

Implements the authorization code flow:
1. Opens browser to Google consent screen
2. Receives callback with authorization code on local HTTP server
3. Exchanges code for access + refresh tokens
4. Fetches user email for account identification
"""

import asyncio
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from loguru import logger

# Public OAuth client credentials (same as all Antigravity proxies)
OAUTH_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
OAUTH_CLIENT_SECRET = "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"
OAUTH_CALLBACK_PORT = 51121
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}/oauth-callback"

OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"

OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]


def _build_auth_url(state: str) -> str:
    """Build the Google OAuth2 authorization URL."""
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{OAUTH_AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> dict[str, Any]:
    """Exchange authorization code for access and refresh tokens."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI,
            },
        )
        response.raise_for_status()
        return response.json()


async def refresh_access_token(refresh_token: str) -> dict[str, str]:
    """Use a refresh token to get a new access token.

    Returns dict with 'access_token' and optionally 'refresh_token'
    if Google rotated the refresh token.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        data = response.json()
        result = {"access_token": data["access_token"]}
        # Google may rotate refresh tokens
        if "refresh_token" in data:
            result["refresh_token"] = data["refresh_token"]
        return result


async def fetch_user_email(access_token: str) -> str:
    """Fetch the authenticated user's email address."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            OAUTH_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("email", "unknown@gmail.com")


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    auth_code: str | None = None
    received_state: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/oauth-callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        if "error" in params:
            _OAuthCallbackHandler.error = params["error"][0]
            self._send_html(
                "Authentication Failed",
                f"Error: {_OAuthCallbackHandler.error}",
                success=False,
            )
        elif "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            _OAuthCallbackHandler.received_state = params.get("state", [None])[0]
            self._send_html(
                "Authentication Successful!",
                "You can close this window and return to the terminal.",
                success=True,
            )
        else:
            _OAuthCallbackHandler.error = "no_code"
            self._send_html(
                "Authentication Failed",
                "No authorization code received.",
                success=False,
            )

    def _send_html(self, title: str, message: str, *, success: bool) -> None:
        icon = "✅" if success else "❌"
        color = "#4CAF50" if success else "#f44336"
        html = f"""<!DOCTYPE html>
<html><head><title>{title}</title></head>
<body style="font-family:system-ui;display:flex;justify-content:center;
align-items:center;min-height:100vh;margin:0;background:#1a1a2e;color:#eee">
<div style="text-align:center;padding:2rem;border-radius:12px;
background:#16213e;box-shadow:0 4px 20px rgba(0,0,0,.3)">
<h1 style="color:{color}">{icon} {title}</h1>
<p>{message}</p>
</div></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP server logging."""


def run_oauth_flow() -> dict[str, str]:
    """Run the full OAuth flow synchronously (blocking).

    Opens browser, waits for callback, exchanges code for tokens.

    Returns:
        dict with 'access_token', 'refresh_token', and 'email'

    Raises:
        RuntimeError: If authentication fails
    """
    state = secrets.token_urlsafe(32)

    # Reset handler state
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.received_state = None
    _OAuthCallbackHandler.error = None

    server = HTTPServer(("localhost", OAUTH_CALLBACK_PORT), _OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout

    auth_url = _build_auth_url(state)
    logger.info("Opening browser for Google OAuth login...")
    webbrowser.open(auth_url)

    # Wait for the callback
    while (
        _OAuthCallbackHandler.auth_code is None and _OAuthCallbackHandler.error is None
    ):
        server.handle_request()

    server.server_close()

    if _OAuthCallbackHandler.error:
        raise RuntimeError(
            f"OAuth authentication failed: {_OAuthCallbackHandler.error}"
        )

    if _OAuthCallbackHandler.received_state != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF attack")

    code = _OAuthCallbackHandler.auth_code
    if not code:
        raise RuntimeError("No authorization code received")

    # Exchange code for tokens
    token_data = asyncio.run(exchange_code_for_tokens(code))
    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")

    if not access_token or not refresh_token:
        raise RuntimeError("Token exchange failed — no tokens received")

    # Get user email
    email = asyncio.run(fetch_user_email(access_token))

    logger.success("Authenticated as: {}", email)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "email": email,
    }
