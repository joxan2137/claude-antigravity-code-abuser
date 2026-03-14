#!/usr/bin/env python3
"""Antigravity Account Manager — manage Google accounts for the proxy.

Run with: uv run manage_accounts.py

Provides an interactive CLI to add, remove, and view Google accounts
used by the Antigravity provider for multi-account rotation.
"""

import sys
from pathlib import Path

from loguru import logger

from providers.antigravity.account_manager import DEFAULT_ACCOUNTS_PATH, AccountManager

BANNER = """
\033[1;36m═══════════════════════════════════════════════════\033[0m
\033[1;37m  Antigravity Account Manager\033[0m
\033[1;36m═══════════════════════════════════════════════════\033[0m"""


def show_status(manager: AccountManager) -> None:
    """Display account status."""
    statuses = manager.get_accounts_status()
    total = len(statuses)
    available = sum(1 for s in statuses if s["status"] == "available")

    print(
        f"\n  Accounts: \033[1m{total}\033[0m total, \033[32m{available}\033[0m available\n"
    )

    if not statuses:
        print("  \033[33mNo accounts configured.\033[0m")
        print("  Use [A] to add your first Google account.\n")
        return

    for i, s in enumerate(statuses, 1):
        email = s["email"]
        status = s["status"]
        if status == "available":
            icon = "\033[32m✅ available\033[0m"
        elif status == "rate-limited":
            wait = s["wait_seconds"]
            minutes = int(wait // 60)
            seconds = int(wait % 60)
            time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
            icon = f"\033[33m⏳ rate-limited (resets in {time_str})\033[0m"
        else:
            reason = s.get("invalid_reason", "unknown")
            icon = f"\033[31m❌ invalid ({reason})\033[0m"
        print(f"  {i}. {email}  {icon}")

    print()


def add_account(manager: AccountManager) -> None:
    """Add a Google account via OAuth."""
    print("\n  \033[1mOpening browser for Google login...\033[0m")
    print("  Complete the sign-in in your browser.\n")

    try:
        from providers.antigravity.oauth import run_oauth_flow

        result = run_oauth_flow()
        manager.add_account(
            email=result["email"],
            refresh_token=result["refresh_token"],
            source="oauth",
        )
        print(f"\n  \033[32m✅ Added account: {result['email']}\033[0m\n")
    except Exception as e:
        print(f"\n  \033[31m❌ Failed to add account: {e}\033[0m\n")


def remove_account(manager: AccountManager) -> None:
    """Remove an account by number."""
    statuses = manager.get_accounts_status()
    if not statuses:
        print("\n  \033[33mNo accounts to remove.\033[0m\n")
        return

    show_status(manager)
    choice = input("  Enter account number to remove (or 'c' to cancel): ").strip()

    if choice.lower() == "c":
        return

    try:
        idx = int(choice) - 1
        email = manager.remove_account(idx)
        if email:
            print(f"\n  \033[32m✅ Removed: {email}\033[0m\n")
        else:
            print("\n  \033[31m❌ Invalid account number.\033[0m\n")
    except ValueError:
        print("\n  \033[31m❌ Invalid input.\033[0m\n")


def add_manual_token(manager: AccountManager) -> None:
    """Add an account with a raw OAuth token (no refresh)."""
    print("\n  \033[1mManual Token Entry\033[0m")
    print("  This token will expire and cannot be auto-refreshed.")
    print("  For persistent accounts, use [A] (OAuth login) instead.\n")

    email = input("  Email (for identification): ").strip()
    if not email:
        print("\n  \033[31m❌ Email required.\033[0m\n")
        return

    token = input("  OAuth access token: ").strip()
    if not token:
        print("\n  \033[31m❌ Token required.\033[0m\n")
        return

    # Store as manual account
    from providers.antigravity.account_manager import Account

    manager._accounts = [a for a in manager._accounts if a.email != email]
    acc = Account(
        {
            "email": email,
            "source": "manual",
            "api_key": token,
        }
    )
    manager._accounts.append(acc)
    manager.save()
    print(f"\n  \033[32m✅ Added manual token for: {email}\033[0m\n")


def main() -> None:
    """Run the interactive account manager."""
    # Suppress loguru noise for interactive CLI
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    accounts_path = DEFAULT_ACCOUNTS_PATH

    # Allow custom path via CLI arg
    if len(sys.argv) > 1:
        accounts_path = Path(sys.argv[1])

    manager = AccountManager(accounts_path)
    manager.load()

    print(BANNER)
    print(f"  Config: {accounts_path}")

    while True:
        show_status(manager)

        print("  \033[1m[A]\033[0m Add account (Google OAuth login)")
        print("  \033[1m[M]\033[0m Add manual token")
        print("  \033[1m[R]\033[0m Remove account")
        print("  \033[1m[Q]\033[0m Quit")
        print()

        choice = input("  > ").strip().upper()

        if choice == "A":
            add_account(manager)
        elif choice == "M":
            add_manual_token(manager)
        elif choice == "R":
            remove_account(manager)
        elif choice == "Q":
            print("\n  Bye!\n")
            break
        else:
            print("\n  \033[33mInvalid option.\033[0m\n")


if __name__ == "__main__":
    main()
