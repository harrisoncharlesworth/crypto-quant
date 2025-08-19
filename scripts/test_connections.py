#!/usr/bin/env python3
"""
Test script to validate Binance API and email connections.
Run this after configuring your .env file.
"""

import os
import sys
import asyncio
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.notifications.email import notifier

# Load environment variables
load_dotenv()


async def test_binance_connection():
    """Test Binance API connection."""
    print("🔗 Testing Binance Connection...")

    try:
        # Get credentials from environment
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")
        sandbox = os.getenv("BINANCE_SANDBOX", "true").lower() == "true"

        if not api_key or not secret:
            print("❌ Binance credentials not found in .env file")
            print("   Please set BINANCE_API_KEY and BINANCE_SECRET")
            return False

        # Create exchange instance
        exchange_config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        }

        if sandbox:
            print("📋 Using Binance TESTNET (test mode)")
            exchange_config["test"] = True  # Use alternative testnet flag
        else:
            exchange_config["sandbox"] = sandbox

        exchange = ccxt.binance(exchange_config)

        # Test public endpoint (no auth required)
        print("   Testing public data access...")
        ticker = exchange.fetch_ticker("BTC/USDT")
        print(f"   ✅ Public API works - BTC/USDT price: ${ticker['last']:,.2f}")

        # Test private endpoint (requires valid API keys)
        print("   Testing authenticated access...")
        balance = exchange.fetch_balance()
        print("   ✅ Private API works - Account connected successfully")

        # Show balances if any
        non_zero_balances = {
            asset: balance for asset, balance in balance["total"].items() if balance > 0
        }
        if non_zero_balances:
            print("   💰 Non-zero balances:")
            for asset, amount in non_zero_balances.items():
                print(f"      {asset}: {amount:,.4f}")
        else:
            print("   💰 Account has zero balances (normal for testnet)")

        return True

    except ccxt.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("   Check your API key and secret in .env file")
        return False
    except ccxt.NetworkError as e:
        print(f"❌ Network error: {e}")
        print("   Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def test_email_notifications():
    """Test email notification system."""
    print("\n📧 Testing Email Notifications...")

    try:
        # Check if email is configured (support both naming conventions)
        email_from = os.getenv("EMAIL_FROM")
        email_password = os.getenv("EMAIL_PASSWORD")
        email_to = os.getenv("EMAIL_TO")

        # Fallback to new variable names
        smtp_user = email_from or os.getenv("SMTP_USER")
        smtp_password = email_password or os.getenv("SMTP_PASSWORD")
        notification_email = email_to or os.getenv("NOTIFICATION_EMAIL")

        if not all([smtp_user, smtp_password, notification_email]):
            print("❌ Email credentials not found in .env file")
            print("   Please set EMAIL_FROM, EMAIL_PASSWORD, and EMAIL_TO")
            print("   (or SMTP_USER, SMTP_PASSWORD, and NOTIFICATION_EMAIL)")
            return False

        print(f"   Using SMTP user: {smtp_user}")
        print(f"   Sending to: {notification_email}")

        # Test basic email
        print("   Sending test email...")
        success = await notifier.send_email(
            subject="🤖 Crypto Quant Bot - Connection Test",
            body=f"""
Test email from your crypto quantitative trading bot!

✅ Email notifications are working correctly
🕐 Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 System: Ready for live trading

If you received this email, your notification system is properly configured.
            """.strip(),
        )

        if success:
            print("   ✅ Email sent successfully!")
            print(f"   📬 Check your inbox at {notification_email}")
            return True
        else:
            print("   ❌ Failed to send email")
            return False

    except Exception as e:
        print(f"❌ Email error: {e}")
        return False


async def test_trading_simulation():
    """Test a simulated trading signal and notification."""
    print("\n🎯 Testing Trading Signal Simulation...")

    try:
        # Simulate a trading signal
        await notifier.send_trade_alert(
            symbol="BTCUSDT",
            action="buy",
            price=45000.0,
            size=0.001,
            reason="Momentum signal triggered (TEST MODE)",
        )
        print("   ✅ Trade alert email sent")

        await notifier.send_risk_alert(
            message="Portfolio exposure at 15% (within limits)", severity="INFO"
        )
        print("   ✅ Risk alert email sent")

        return True

    except Exception as e:
        print(f"❌ Trading simulation error: {e}")
        return False


async def main():
    """Run all connection tests."""
    print("🚀 Crypto Quant Bot - Connection Test")
    print("=" * 50)

    # Test results
    tests_passed = 0
    total_tests = 3

    # Test Binance
    if await test_binance_connection():
        tests_passed += 1

    # Test Email
    if await test_email_notifications():
        tests_passed += 1

    # Test Trading Simulation
    if await test_trading_simulation():
        tests_passed += 1

    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All systems working! Ready for paper trading.")
        print("\nNext steps:")
        print("1. Ensure DRY_RUN=true in your .env file")
        print("2. Run: python3 -m scripts.backtest --symbol BTCUSDT --signal momentum")
        print("3. Monitor email notifications for 24 hours")
        print("4. When confident, set DRY_RUN=false for live trading")
    else:
        print("⚠️  Some tests failed. Fix the issues above before proceeding.")

        if tests_passed == 0:
            print("\n🔧 Common fixes:")
            print("- Check .env file has correct credentials")
            print("- Verify Binance testnet API keys are enabled")
            print("- Confirm Gmail app password (not regular password)")
            print("- Check firewall/VPN isn't blocking connections")


if __name__ == "__main__":
    asyncio.run(main())
