"""Tests for Antigravity multi-account manager."""

import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from providers.antigravity.account_manager import (
    DEFAULT_COOLDOWN_S,
    Account,
    AccountManager,
)

# ──────────────────────────────────────────────────────────────────────
# Account unit tests
# ──────────────────────────────────────────────────────────────────────


def test_account_from_dict():
    """Account initializes from dict correctly."""
    acc = Account(
        {
            "email": "test@gmail.com",
            "source": "oauth",
            "refresh_token": "1//abc",
            "added_at": "2026-01-01T00:00:00Z",
        }
    )
    assert acc.email == "test@gmail.com"
    assert acc.source == "oauth"
    assert acc.refresh_token == "1//abc"
    assert not acc.is_invalid
    assert acc.is_available()


def test_account_to_dict_oauth():
    """OAuth account serializes with refresh_token."""
    acc = Account({"email": "a@b.com", "source": "oauth", "refresh_token": "tok123"})
    d = acc.to_dict()
    assert d["refresh_token"] == "tok123"
    assert "api_key" not in d


def test_account_to_dict_manual():
    """Manual account serializes with api_key."""
    acc = Account({"email": "a@b.com", "source": "manual", "api_key": "ya29.xyz"})
    d = acc.to_dict()
    assert d["api_key"] == "ya29.xyz"
    assert "refresh_token" not in d


def test_account_rate_limit():
    """Rate limiting marks account unavailable for the specified model."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    acc.mark_rate_limited("claude-sonnet-4-5-thinking", 30)

    assert acc.is_rate_limited("claude-sonnet-4-5-thinking")
    assert not acc.is_rate_limited("gemini-3-flash")
    assert not acc.is_available("claude-sonnet-4-5-thinking")
    assert acc.is_available("gemini-3-flash")


def test_account_rate_limit_any():
    """is_rate_limited(None) checks any model."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    acc.mark_rate_limited("claude-sonnet-4-5-thinking", 30)
    assert acc.is_rate_limited()


def test_account_clear_expired_limits():
    """Expired rate limits are cleared."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    # Set a rate limit in the past
    acc.rate_limits["some-model"] = {
        "reset_at": time.time() - 1,
        "marked_at": time.time() - 60,
    }
    cleared = acc.clear_expired_limits()
    assert cleared == 1
    assert not acc.rate_limits


def test_account_mark_invalid():
    """Invalid accounts are not available."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    acc.mark_invalid("auth failed")
    assert acc.is_invalid
    assert acc.invalid_reason == "auth failed"
    assert not acc.is_available()


def test_account_get_wait_seconds():
    """Wait seconds reflects time until rate limit resets."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    acc.mark_rate_limited("test-model", 30)
    wait = acc.get_wait_seconds("test-model")
    assert 25 < wait <= 30


def test_account_get_wait_seconds_no_limit():
    """Wait is 0 when not rate-limited."""
    acc = Account({"email": "a@b.com", "source": "oauth"})
    assert acc.get_wait_seconds("test-model") == 0


@pytest.mark.asyncio
async def test_account_get_access_token_manual():
    """Manual accounts return their api_key directly."""
    acc = Account({"email": "a@b.com", "source": "manual", "api_key": "ya29.test"})
    token = await acc.get_access_token()
    assert token == "ya29.test"


@pytest.mark.asyncio
async def test_account_get_access_token_oauth():
    """OAuth accounts refresh their token."""
    acc = Account({"email": "a@b.com", "source": "oauth", "refresh_token": "1//abc"})

    with patch(
        "providers.antigravity.account_manager.refresh_access_token",
        new_callable=AsyncMock,
    ) as mock_refresh:
        mock_refresh.return_value = {"access_token": "ya29.fresh"}
        token = await acc.get_access_token()

    assert token == "ya29.fresh"
    mock_refresh.assert_called_once_with("1//abc")


@pytest.mark.asyncio
async def test_account_get_access_token_cached():
    """Cached token is returned without refresh if still fresh."""
    acc = Account({"email": "a@b.com", "source": "manual", "api_key": "ya29.test"})
    # First call caches
    await acc.get_access_token()
    # Change the api_key — should still return cached
    acc.api_key = "ya29.changed"
    token = await acc.get_access_token()
    assert token == "ya29.test"


@pytest.mark.asyncio
async def test_account_get_access_token_oauth_marks_invalid_on_error():
    """OAuth refresh failure marks account invalid."""
    acc = Account({"email": "a@b.com", "source": "oauth", "refresh_token": "1//bad"})

    with patch(
        "providers.antigravity.account_manager.refresh_access_token",
        new_callable=AsyncMock,
    ) as mock_refresh:
        mock_refresh.side_effect = Exception("token revoked")
        with pytest.raises(Exception, match="token revoked"):
            await acc.get_access_token()

    assert acc.is_invalid


# ──────────────────────────────────────────────────────────────────────
# AccountManager tests
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def accounts_file(tmp_path):
    """Create a temporary accounts JSON file."""
    return tmp_path / "accounts.json"


@pytest.fixture
def manager_with_accounts(accounts_file):
    """Create a manager with two pre-loaded accounts."""
    data = {
        "accounts": [
            {
                "email": "user1@gmail.com",
                "source": "oauth",
                "refresh_token": "1//token1",
                "added_at": "2026-01-01T00:00:00Z",
            },
            {
                "email": "user2@gmail.com",
                "source": "oauth",
                "refresh_token": "1//token2",
                "added_at": "2026-01-02T00:00:00Z",
            },
        ],
        "active_index": 0,
    }
    accounts_file.write_text(json.dumps(data))
    manager = AccountManager(accounts_file)
    manager.load()
    return manager


def test_manager_load_empty(accounts_file):
    """Manager loads fine with no file."""
    manager = AccountManager(accounts_file)
    manager.load()
    assert manager.account_count == 0
    assert not manager.has_accounts


def test_manager_load_accounts(manager_with_accounts):
    """Manager loads accounts from JSON."""
    assert manager_with_accounts.account_count == 2
    assert manager_with_accounts.has_accounts


def test_manager_add_account(accounts_file):
    """Adding an account creates the file and persists."""
    manager = AccountManager(accounts_file)
    manager.load()
    manager.add_account("new@gmail.com", "1//new_token")
    assert manager.account_count == 1

    # Verify persistence
    data = json.loads(accounts_file.read_text())
    assert len(data["accounts"]) == 1
    assert data["accounts"][0]["email"] == "new@gmail.com"


def test_manager_add_account_replaces_existing(manager_with_accounts, accounts_file):
    """Adding account with same email replaces the existing one."""
    manager_with_accounts.add_account("user1@gmail.com", "1//updated_token")
    assert manager_with_accounts.account_count == 2

    data = json.loads(accounts_file.read_text())
    tokens = [a["refresh_token"] for a in data["accounts"]]
    assert "1//updated_token" in tokens
    assert "1//token1" not in tokens


def test_manager_remove_account(manager_with_accounts):
    """Remove account by index."""
    email = manager_with_accounts.remove_account(0)
    assert email == "user1@gmail.com"
    assert manager_with_accounts.account_count == 1


def test_manager_remove_invalid_index(manager_with_accounts):
    """Remove with invalid index returns None."""
    assert manager_with_accounts.remove_account(99) is None


def test_manager_pick_account_sticky(manager_with_accounts):
    """Pick prefers the current (sticky) account."""
    acc = manager_with_accounts.pick_account()
    assert acc is not None
    assert acc.email == "user1@gmail.com"

    # Picking again returns same account (sticky)
    acc2 = manager_with_accounts.pick_account()
    assert acc2 is not None
    assert acc2.email == "user1@gmail.com"


def test_manager_pick_account_rotates_on_rate_limit(manager_with_accounts):
    """Pick rotates to next account when current is rate-limited."""
    manager_with_accounts.mark_rate_limited(
        "user1@gmail.com", "test-model", DEFAULT_COOLDOWN_S
    )
    acc = manager_with_accounts.pick_account("test-model")
    assert acc is not None
    assert acc.email == "user2@gmail.com"


def test_manager_pick_account_none_when_all_limited(manager_with_accounts):
    """Pick returns None when all accounts are rate-limited."""
    manager_with_accounts.mark_rate_limited("user1@gmail.com", "test-model", 60)
    manager_with_accounts.mark_rate_limited("user2@gmail.com", "test-model", 60)
    acc = manager_with_accounts.pick_account("test-model")
    assert acc is None


def test_manager_pick_account_skips_invalid(manager_with_accounts):
    """Pick skips invalid accounts."""
    manager_with_accounts.mark_invalid("user1@gmail.com", "auth failed")
    acc = manager_with_accounts.pick_account()
    assert acc is not None
    assert acc.email == "user2@gmail.com"


def test_manager_all_rate_limited(manager_with_accounts):
    """all_rate_limited returns True when all accounts are limited."""
    assert not manager_with_accounts.all_rate_limited("test-model")
    manager_with_accounts.mark_rate_limited("user1@gmail.com", "test-model", 60)
    assert not manager_with_accounts.all_rate_limited("test-model")
    manager_with_accounts.mark_rate_limited("user2@gmail.com", "test-model", 60)
    assert manager_with_accounts.all_rate_limited("test-model")


def test_manager_get_min_wait(manager_with_accounts):
    """get_min_wait_seconds returns shortest wait."""
    manager_with_accounts.mark_rate_limited("user1@gmail.com", "m", 120)
    manager_with_accounts.mark_rate_limited("user2@gmail.com", "m", 30)
    wait = manager_with_accounts.get_min_wait_seconds("m")
    assert 25 < wait <= 30


def test_manager_get_accounts_status(manager_with_accounts):
    """Status shows available/rate-limited/invalid states."""
    manager_with_accounts.mark_rate_limited("user2@gmail.com", "m", 60)
    statuses = manager_with_accounts.get_accounts_status()
    assert len(statuses) == 2
    assert statuses[0]["status"] == "available"
    assert statuses[1]["status"] == "rate-limited"


def test_manager_save_and_reload(accounts_file):
    """Accounts survive save/reload cycle."""
    m1 = AccountManager(accounts_file)
    m1.load()
    m1.add_account("a@b.com", "1//tok")

    m2 = AccountManager(accounts_file)
    m2.load()
    assert m2.account_count == 1
    assert m2._accounts[0].email == "a@b.com"
