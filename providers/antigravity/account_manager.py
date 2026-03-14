"""Multi-account manager for Antigravity provider.

Handles account storage, token refresh, rate-limit tracking,
and automatic rotation between Google accounts.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .oauth import refresh_access_token

# Default storage location
DEFAULT_ACCOUNTS_PATH = Path.home() / ".freecc" / "antigravity_accounts.json"

# Token refresh interval (5 minutes)
TOKEN_REFRESH_INTERVAL_S = 5 * 60

# Default cooldown for rate-limited accounts (60 seconds)
DEFAULT_COOLDOWN_S = 60

# Maximum wait before throwing error (2 minutes)
MAX_WAIT_BEFORE_ERROR_S = 120


class Account:
    """Represents a single Google account with its state."""

    def __init__(self, data: dict[str, Any]):
        self.email: str = data.get("email", "unknown")
        self.source: str = data.get("source", "oauth")
        self.refresh_token: str = data.get("refresh_token", "")
        self.api_key: str = data.get("api_key", "")
        self.project_id: str = data.get("project_id", "")
        self.added_at: str = data.get("added_at", "")
        self.is_invalid: bool = False
        self.invalid_reason: str | None = None
        self.rate_limits: dict[str, dict[str, Any]] = {}
        self.last_used: str | None = data.get("last_used")

        # In-memory token cache
        self._cached_token: str | None = None
        self._token_fetched_at: float = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        d: dict[str, Any] = {
            "email": self.email,
            "source": self.source,
            "added_at": self.added_at,
            "last_used": self.last_used,
            "rate_limits": self.rate_limits,
        }
        if self.source == "oauth" and self.refresh_token:
            d["refresh_token"] = self.refresh_token
        if self.source == "manual" and self.api_key:
            d["api_key"] = self.api_key
        if self.project_id:
            d["project_id"] = self.project_id
        return d

    def is_rate_limited(self, model: str | None = None) -> bool:
        """Check if account is rate-limited for a specific model (or any)."""
        now = time.time()
        if model and model in self.rate_limits:
            rl = self.rate_limits[model]
            return rl.get("reset_at", 0) > now
        if not model:
            return any(rl.get("reset_at", 0) > now for rl in self.rate_limits.values())
        return False

    def mark_rate_limited(
        self, model: str, reset_seconds: float = DEFAULT_COOLDOWN_S
    ) -> None:
        """Mark this account as rate-limited for a model."""
        self.rate_limits[model] = {
            "reset_at": time.time() + reset_seconds,
            "marked_at": time.time(),
        }
        logger.warning(
            "ACCOUNT_RATE_LIMITED: {} model={} reset_in={:.0f}s",
            self.email,
            model,
            reset_seconds,
        )

    def mark_invalid(self, reason: str) -> None:
        """Mark this account as invalid (needs re-auth)."""
        self.is_invalid = True
        self.invalid_reason = reason
        logger.warning("ACCOUNT_INVALID: {} reason={}", self.email, reason)

    def clear_expired_limits(self) -> int:
        """Remove expired rate limits. Returns count cleared."""
        now = time.time()
        expired = [
            m for m, rl in self.rate_limits.items() if rl.get("reset_at", 0) <= now
        ]
        for m in expired:
            del self.rate_limits[m]
        return len(expired)

    def is_available(self, model: str | None = None) -> bool:
        """Check if this account is available (not invalid, not rate-limited)."""
        if self.is_invalid:
            return False
        return not self.is_rate_limited(model)

    async def get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        now = time.time()

        # Return cached token if still fresh
        if (
            self._cached_token
            and (now - self._token_fetched_at) < TOKEN_REFRESH_INTERVAL_S
        ):
            return self._cached_token

        # Manual/raw token accounts
        if self.source == "manual" and self.api_key:
            self._cached_token = self.api_key
            self._token_fetched_at = now
            return self.api_key

        # OAuth accounts — refresh the token
        if self.source == "oauth" and self.refresh_token:
            try:
                result = await refresh_access_token(self.refresh_token)
                self._cached_token = result["access_token"]
                self._token_fetched_at = now
                # Handle refresh token rotation
                if (
                    "refresh_token" in result
                    and result["refresh_token"] != self.refresh_token
                ):
                    logger.info("OAUTH_TOKEN_ROTATED: {}", self.email)
                    self.refresh_token = result["refresh_token"]
                # Clear invalid flag on success
                if self.is_invalid:
                    self.is_invalid = False
                    self.invalid_reason = None
                return self._cached_token
            except Exception as e:
                self.mark_invalid(str(e))
                raise

        raise RuntimeError(f"Account {self.email} has no valid credentials")

    def get_wait_seconds(self, model: str | None = None) -> float:
        """Get seconds until this account becomes available for a model."""
        now = time.time()
        if model and model in self.rate_limits:
            return max(0, self.rate_limits[model].get("reset_at", 0) - now)
        if not model and self.rate_limits:
            return max(
                0, max(rl.get("reset_at", 0) for rl in self.rate_limits.values()) - now
            )
        return 0


class AccountManager:
    """Manages multiple Antigravity accounts with rotation and rate-limit tracking."""

    def __init__(self, accounts_path: str | Path = DEFAULT_ACCOUNTS_PATH):
        self._path = Path(accounts_path)
        self._accounts: list[Account] = []
        self._active_index: int = 0
        self._loaded = False

    def load(self) -> None:
        """Load accounts from the JSON file."""
        if not self._path.exists():
            logger.info("ACCOUNT_MANAGER: No accounts file at {}", self._path)
            self._loaded = True
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            raw_accounts = data.get("accounts", [])
            self._accounts = [Account(a) for a in raw_accounts]
            self._active_index = min(
                data.get("active_index", 0), max(0, len(self._accounts) - 1)
            )
            self._loaded = True
            # Clear stale rate limits on load
            for acc in self._accounts:
                acc.clear_expired_limits()
            logger.info(
                "ACCOUNT_MANAGER: Loaded {} account(s) from {}",
                len(self._accounts),
                self._path,
            )
        except Exception as e:
            logger.error("ACCOUNT_MANAGER: Failed to load accounts: {}", e)
            self._loaded = True

    def save(self) -> None:
        """Persist accounts to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "accounts": [a.to_dict() for a in self._accounts],
            "active_index": self._active_index,
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def account_count(self) -> int:
        return len(self._accounts)

    @property
    def has_accounts(self) -> bool:
        return len(self._accounts) > 0

    def add_account(
        self,
        email: str,
        refresh_token: str,
        source: str = "oauth",
    ) -> None:
        """Add a new account. Replaces existing account with same email."""
        # Remove existing with same email
        self._accounts = [a for a in self._accounts if a.email != email]
        acc = Account(
            {
                "email": email,
                "source": source,
                "refresh_token": refresh_token,
                "added_at": datetime.now(UTC).isoformat(),
            }
        )
        self._accounts.append(acc)
        self.save()
        logger.info("ACCOUNT_ADDED: {} (total: {})", email, len(self._accounts))

    def remove_account(self, index: int) -> str | None:
        """Remove account by index. Returns removed email or None."""
        if 0 <= index < len(self._accounts):
            removed = self._accounts.pop(index)
            if self._active_index >= len(self._accounts):
                self._active_index = 0
            self.save()
            return removed.email
        return None

    def get_accounts_status(self) -> list[dict[str, Any]]:
        """Get status of all accounts for display."""
        result = []
        for acc in self._accounts:
            acc.clear_expired_limits()
            wait = acc.get_wait_seconds()
            status = (
                "invalid"
                if acc.is_invalid
                else ("rate-limited" if wait > 0 else "available")
            )
            result.append(
                {
                    "email": acc.email,
                    "source": acc.source,
                    "status": status,
                    "wait_seconds": wait,
                    "invalid_reason": acc.invalid_reason,
                    "added_at": acc.added_at,
                }
            )
        return result

    def pick_account(self, model: str | None = None) -> Account | None:
        """Pick the best available account for a request.

        Strategy: sticky selection — prefers the current account for cache
        continuity, only switches when rate-limited or invalid.
        """
        if not self._accounts:
            return None

        # Clear expired limits first
        for acc in self._accounts:
            acc.clear_expired_limits()

        # Try current account first (sticky)
        if self._active_index < len(self._accounts):
            current = self._accounts[self._active_index]
            if current.is_available(model):
                current.last_used = datetime.now(UTC).isoformat()
                return current

        # Current is unavailable — find next available
        for i in range(len(self._accounts)):
            idx = (self._active_index + 1 + i) % len(self._accounts)
            candidate = self._accounts[idx]
            if candidate.is_available(model):
                self._active_index = idx
                candidate.last_used = datetime.now(UTC).isoformat()
                logger.info(
                    "ACCOUNT_SWITCHED: {} -> {} (model={})",
                    self._accounts[(idx - 1) % len(self._accounts)].email
                    if len(self._accounts) > 1
                    else "none",
                    candidate.email,
                    model or "any",
                )
                self.save()
                return candidate

        # All accounts are rate-limited or invalid
        return None

    def mark_rate_limited(
        self, email: str, model: str, reset_seconds: float = DEFAULT_COOLDOWN_S
    ) -> None:
        """Mark an account as rate-limited for a model."""
        for acc in self._accounts:
            if acc.email == email:
                acc.mark_rate_limited(model, reset_seconds)
                self.save()
                return

    def mark_invalid(self, email: str, reason: str) -> None:
        """Mark an account as invalid."""
        for acc in self._accounts:
            if acc.email == email:
                acc.mark_invalid(reason)
                self.save()
                return

    def get_min_wait_seconds(self, model: str | None = None) -> float:
        """Get minimum wait time until any account becomes available."""
        waits = []
        for acc in self._accounts:
            if not acc.is_invalid:
                w = acc.get_wait_seconds(model)
                if w > 0:
                    waits.append(w)
        return min(waits) if waits else 0

    def all_rate_limited(self, model: str | None = None) -> bool:
        """Check if all accounts are rate-limited or invalid."""
        if not self._accounts:
            return False
        return all(not a.is_available(model) for a in self._accounts)
