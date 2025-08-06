#!/usr/bin/env python3
"""
Debug email configuration with detailed error information
"""

import os
import asyncio
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

async def test_email_detailed():
    """Test email with detailed debugging."""
    
    print("📧 Detailed Email Debug Test")
    print("=" * 40)
    
    # Get credentials
    email_from = os.getenv('EMAIL_FROM')
    email_password = os.getenv('EMAIL_PASSWORD') 
    email_to = os.getenv('EMAIL_TO')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    
    print(f"From: {email_from}")
    print(f"To: {email_to}")
    print(f"Server: {smtp_server}:{smtp_port}")
    print(f"Password length: {len(email_password) if email_password else 0}")
    
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = email_from
        message["To"] = email_to
        message["Subject"] = "🤖 Crypto Bot Test"
        
        body = "Test email from crypto quant bot!"
        message.attach(MIMEText(body, "plain"))
        
        print("\n🔗 Connecting to SMTP server...")
        
        # Connect with detailed error handling
        async with aiosmtplib.SMTP(
            hostname=smtp_server,
            port=smtp_port,
            start_tls=True,
            timeout=30
        ) as server:
            print("✅ Connected to server")
            
            print("🔐 Logging in...")
            await server.login(email_from, email_password)
            print("✅ Login successful")
            
            print("📤 Sending message...")
            await server.send_message(message)
            print("✅ Email sent successfully!")
            
        return True
        
    except aiosmtplib.SMTPAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Check your Gmail app password is correct")
        print("💡 Make sure 2-factor authentication is enabled")
        return False
    except aiosmtplib.SMTPConnectError as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Other error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_email_detailed())
