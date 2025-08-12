#!/usr/bin/env python3
"""
Test script to verify email notification system is working.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantbot.notifications.email import EmailNotifier


async def test_email_system():
    """Test the email notification system."""
    print("🧪 Testing Email Notification System...")
    
    # Initialize the email notifier
    notifier = EmailNotifier()
    
    # Check if credentials are configured
    if not notifier.smtp_user or not notifier.smtp_password or not notifier.notification_email:
        print("❌ Email credentials not fully configured!")
        print(f"   SMTP User: {notifier.smtp_user}")
        print(f"   SMTP Password: {'*' * len(notifier.smtp_password) if notifier.smtp_password else 'None'}")
        print(f"   Notification Email: {notifier.notification_email}")
        return False
    
    print("✅ Email credentials configured")
    print(f"   From: {notifier.smtp_user}")
    print(f"   To: {notifier.notification_email}")
    print(f"   SMTP Server: {notifier.smtp_host}:{notifier.smtp_port}")
    
    # Test 1: Simple email
    print("\n📧 Sending test email...")
    success = await notifier.send_email(
        subject="🧪 Crypto Quant Bot - Email Test",
        body="""
🚀 CRYPTO QUANT BOT - EMAIL TEST

This is a test email to verify the notification system is working correctly.

✅ Email System Status:
   • SMTP Connection: Working
   • Authentication: Successful
   • Recipient: Configured
   • Daily Reports: Enabled

📊 Bot Status:
   • Trading: Active
   • Signals: Generating
   • Risk Management: Conservative Settings
   • Daily Digest: 6:00 PM AEST

🔔 You should now receive:
   • Trade execution alerts
   • Daily portfolio summaries
   • Risk alerts (if needed)
   • Signal generation notifications

---
🤖 Crypto Quant Bot Trading System
📧 Daily Reports: 6:00 PM AEST
🌐 Railway Cloud Deployment
        """
    )
    
    if success:
        print("✅ Test email sent successfully!")
        
        # Test 2: Trade alert
        print("\n📈 Sending test trade alert...")
        trade_success = await notifier.send_enhanced_trade_alert(
            symbol="BTCUSD",
            action="buy",
            price=45000.00,
            size=0.001,
            reason="Test Signal: Momentum + Mean Reversion",
            confidence=0.75,
            signal_strength=0.8,
            account_balance=200000.00
        )
        
        if trade_success:
            print("✅ Test trade alert sent successfully!")
        else:
            print("❌ Test trade alert failed!")
            
        return True
    else:
        print("❌ Test email failed!")
        return False


async def main():
    """Main function."""
    try:
        success = await test_email_system()
        if success:
            print("\n🎉 Email notification system is working correctly!")
            print("   You should receive test emails shortly.")
        else:
            print("\n💥 Email notification system has issues.")
            print("   Check the configuration and try again.")
    except Exception as e:
        print(f"❌ Error testing email system: {e}")


if __name__ == "__main__":
    asyncio.run(main())
