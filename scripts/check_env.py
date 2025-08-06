#!/usr/bin/env python3
"""
Quick check of .env file configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Environment Variable Check")
print("=" * 40)

# Check Binance credentials
api_key = os.getenv('BINANCE_API_KEY')
secret = os.getenv('BINANCE_SECRET')
sandbox = os.getenv('BINANCE_SANDBOX')

print(f"BINANCE_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
if api_key:
    print(f"  Length: {len(api_key)} characters")
    print(f"  Starts with: {api_key[:8]}...")

print(f"BINANCE_SECRET: {'✅ Set' if secret else '❌ Missing'}")
if secret:
    print(f"  Length: {len(secret)} characters")
    print(f"  Starts with: {secret[:8]}...")

print(f"BINANCE_SANDBOX: {sandbox}")

# Check email credentials (support both variable naming conventions)
email_from = os.getenv('EMAIL_FROM')
email_password = os.getenv('EMAIL_PASSWORD')
email_to = os.getenv('EMAIL_TO')
smtp_server = os.getenv('SMTP_SERVER')

# Fallback to new variable names
smtp_user = email_from or os.getenv('SMTP_USER')
smtp_password = email_password or os.getenv('SMTP_PASSWORD')
notification_email = email_to or os.getenv('NOTIFICATION_EMAIL')

print(f"\nEmail Configuration:")
print(f"EMAIL_FROM: {'✅ Set' if email_from else '❌ Missing'}")
print(f"EMAIL_PASSWORD: {'✅ Set' if email_password else '❌ Missing'}")
print(f"EMAIL_TO: {'✅ Set' if email_to else '❌ Missing'}")
print(f"SMTP_SERVER: {'✅ Set' if smtp_server else '❌ Missing'}")

print("\n📋 Expected formats:")
print("BINANCE_API_KEY=64-character string (letters and numbers)")
print("BINANCE_SECRET=64-character string (letters and numbers)")
print("SMTP_USER=your_email@gmail.com")
print("SMTP_PASSWORD=16-character app password (xxxx xxxx xxxx xxxx)")
