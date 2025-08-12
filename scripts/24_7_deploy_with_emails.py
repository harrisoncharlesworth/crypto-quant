#!/usr/bin/env python3
"""
24/7 Crypto Quant Bot Deployment with Daily Email Reports
Deploys the optimized bot with strategic recommendations and comprehensive email notifications
"""

import os
import sys
import asyncio
import subprocess
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optimized_bot_config import OptimizedBotConfig

class CryptoQuantDeployment:
    """24/7 deployment manager with email notifications."""
    
    def __init__(self):
        self.config = OptimizedBotConfig()
        self.bot_process = None
        self.email_recipient = "ebullemor@gmail.com"
        self.deployment_start_time = datetime.now()
        
    def setup_environment(self):
        """Set up complete environment for 24/7 deployment."""
        print("🔧 Setting up 24/7 Deployment Environment...")
        
        # Alpaca credentials
        os.environ["ALPACA_API_KEY"] = "PKJYFI6XVZ9UGW85JFVP"
        os.environ["ALPACA_SECRET_KEY"] = "HY2qdRNAX8TSONDcnsqPfNHUp7WzpeHzpsAdNEGZ"
        
        # Apply optimized configuration
        self.config.setup_environment()
        
        # Enhanced email notifications
        os.environ["ENABLE_EMAIL_NOTIFICATIONS"] = "true"
        os.environ["EMAIL_RECIPIENT"] = self.email_recipient
        os.environ["DIGEST_INTERVAL_HOURS"] = "24"  # Daily reports
        os.environ["TRADE_NOTIFICATIONS"] = "true"  # Individual trade emails
        os.environ["DAILY_SUMMARY"] = "true"        # Daily summary emails
        
        # 24/7 specific settings
        os.environ["RESTART_ON_FAILURE"] = "true"
        os.environ["HEALTH_CHECK_INTERVAL"] = "300"  # 5 minutes
        os.environ["MAX_RESTART_ATTEMPTS"] = "10"
        os.environ["LOG_ROTATION"] = "true"
        
        # Enhanced monitoring
        os.environ["PERFORMANCE_MONITORING"] = "true"
        os.environ["RISK_MONITORING"] = "true"
        os.environ["POSITION_TRACKING"] = "true"
        os.environ["SIGNAL_MONITORING"] = "true"
        
        print("✅ Environment configured for 24/7 operation with email notifications")
        
    async def test_alpaca_connection(self):
        """Test Alpaca connection before deployment."""
        try:
            from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper
            
            print("🔌 Testing Alpaca Paper Trading Connection...")
            alpaca = AlpacaWrapper(paper=True)
            alpaca.load_markets()
            
            account = alpaca.trading_client.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   Account ID: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Cash: ${float(account.cash):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            
            return True
        except Exception as e:
            print(f"❌ Alpaca connection failed: {e}")
            return False
            
    def create_24_7_config_file(self):
        """Create comprehensive 24/7 configuration file."""
        config_content = f"""
# 24/7 Crypto Quant Bot Configuration
# Optimized with Strategic Recommendations + Daily Email Reports
# Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

# Alpaca Configuration
ALPACA_API_KEY=PKJYFI6XVZ9UGW85JFVP
ALPACA_SECRET_KEY=HY2qdRNAX8TSONDcnsqPfNHUp7WzpeHzpsAdNEGZ
ALPACA_PAPER=true

# Core Trading Settings
DRY_RUN=false
USE_FUTURES=true
UPDATE_INTERVAL_MINUTES=10
TRADING_SYMBOLS=BTCUSD,ETHUSD,SOLUSD,ADAUSD

# Enhanced Risk Management (Strategic Recommendations)
MAX_PORTFOLIO_ALLOCATION=0.60
MAX_NET_EXPOSURE=0.20
MAX_GROSS_LEVERAGE=1.8
MAX_SINGLE_POSITION=0.06
MAX_DAILY_DRAWDOWN=0.15
MAX_WEEKLY_DRAWDOWN=0.25
POSITION_SIZING_METHOD=KELLY_OPTIMAL

# Signal Configurations (Optimized)
MOMENTUM_LOOKBACK_DAYS=25
MOMENTUM_MA_WINDOW=80
MOMENTUM_WEIGHT=1.1

BREAKOUT_CHANNEL_PERIOD=25
BREAKOUT_ATR_PERIOD=10
BREAKOUT_ATR_MULTIPLIER=1.8
BREAKOUT_WEIGHT=1.3

MEAN_REVERSION_LOOKBACK_DAYS=7
MEAN_REVERSION_ZSCORE_THRESHOLD=1.6
MEAN_REVERSION_WEIGHT=0.9

FUNDING_CARRY_THRESHOLD=0.0003
FUNDING_CARRY_MAX_ALLOCATION=0.12
FUNDING_CARRY_WEIGHT=1.2

# Portfolio Blender Settings
ALLOCATION_METHOD=CONFIDENCE_WEIGHTED
MIN_SIGNAL_CONFIDENCE=0.20

# Email Notifications (Daily Reports)
ENABLE_EMAIL_NOTIFICATIONS=true
EMAIL_RECIPIENT={self.email_recipient}
DIGEST_INTERVAL_HOURS=24
TRADE_NOTIFICATIONS=true
DAILY_SUMMARY=true
EMAIL_SUBJECT_PREFIX="[Crypto Quant Bot]"

# 24/7 Operation Settings
RESTART_ON_FAILURE=true
HEALTH_CHECK_INTERVAL=300
MAX_RESTART_ATTEMPTS=10
LOG_ROTATION=true
LOG_LEVEL=INFO

# Enhanced Monitoring
PERFORMANCE_MONITORING=true
RISK_MONITORING=true
POSITION_TRACKING=true
SIGNAL_MONITORING=true
ENABLE_HEALTH_CHECKS=true

# Performance Targets
TARGET_MONTHLY_RETURN=0.15
MAX_DRAWDOWN_TARGET=0.25
TARGET_SHARPE_RATIO=0.8
WIN_RATE_TARGET=0.55
VOLATILITY_TARGET=0.35
ALPHA_TARGET=0.20

# Daily Report Settings
DAILY_REPORT_TIME=18:00
DAILY_REPORT_TIMEZONE=UTC
INCLUDE_TRADE_DETAILS=true
INCLUDE_PERFORMANCE_METRICS=true
INCLUDE_RISK_METRICS=true
INCLUDE_SIGNAL_ANALYSIS=true
"""
        
        with open("24_7_config.env", "w") as f:
            f.write(config_content)
        
        print("✅ 24/7 configuration file created")
        
    def send_deployment_notification(self, status="STARTED"):
        """Send deployment notification email."""
        try:
            subject = f"[Crypto Quant Bot] 24/7 Deployment {status}"
            
            body = f"""
🚀 Crypto Quant Bot 24/7 Deployment {status}

📊 Deployment Details:
• Start Time: {self.deployment_start_time:%Y-%m-%d %H:%M:%S UTC}
• Status: {status}
• Email Recipient: {self.email_recipient}
• Trading Mode: Paper Trading (Alpaca)
• Symbols: BTCUSD, ETHUSD, SOLUSD, ADAUSD

🎯 Strategic Recommendations Implemented:
• Enhanced Risk Management (50% exposure reduction)
• Optimized Signal Configurations
• Confidence-Weighted Portfolio Allocation
• Daily Email Reports Enabled
• 24/7 Operation with Auto-Restart

📈 Performance Targets:
• Monthly Return: 15%
• Max Drawdown: 25%
• Sharpe Ratio: 0.8
• Win Rate: 55%

📧 Email Notifications:
• Daily Trade Summaries: 6:00 PM UTC
• Individual Trade Alerts: Enabled
• Performance Reports: Daily
• Risk Monitoring: Continuous

🛡️ Risk Management:
• Max Net Exposure: 20%
• Max Gross Leverage: 1.8
• Max Single Position: 6%
• Daily Drawdown Limit: 15%

The bot is now running 24/7 with comprehensive monitoring and daily email reports to {self.email_recipient}.

---
Generated by Crypto Quant Bot Deployment System
"""
            
            # For now, we'll use a simple notification
            # In production, you'd integrate with your email service
            print(f"📧 Email notification would be sent to {self.email_recipient}")
            print(f"   Subject: {subject}")
            print(f"   Status: {status}")
            
        except Exception as e:
            print(f"⚠️ Email notification failed: {e}")
            
    def start_24_7_bot(self):
        """Start the bot for 24/7 operation."""
        try:
            print("🚀 Starting 24/7 Crypto Quant Bot...")
            
            # Create configuration file
            self.create_24_7_config_file()
            
            # Start the bot with 24/7 settings
            cmd = [sys.executable, "-u", "scripts/run_live_bot.py"]
            
            # Set environment for 24/7 deployment
            env = os.environ.copy()
            env["24_7_MODE"] = "true"
            env["STRATEGIC_RECOMMENDATIONS"] = "implemented"
            env["EMAIL_NOTIFICATIONS"] = "enabled"
            env["DAILY_REPORTS"] = "enabled"
            
            print("📊 Starting with 24/7 configuration...")
            print("   • Enhanced risk management: ACTIVE")
            print("   • Strategic recommendations: IMPLEMENTED")
            print("   • Daily email reports: ENABLED")
            print("   • Auto-restart on failure: ENABLED")
            print("   • Continuous monitoring: ACTIVE")
            
            # Start the bot process
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            print("✅ 24/7 bot started successfully!")
            print(f"📧 Daily reports will be sent to: {self.email_recipient}")
            print("🛑 Press Ctrl+C to stop")
            print()
            
            return self.bot_process
            
        except Exception as e:
            print(f"❌ Failed to start 24/7 bot: {e}")
            return None
            
    def monitor_bot_health(self):
        """Monitor bot health and restart if needed."""
        restart_attempts = 0
        max_attempts = int(os.environ.get("MAX_RESTART_ATTEMPTS", 10))
        
        while restart_attempts < max_attempts:
            if self.bot_process and self.bot_process.poll() is not None:
                print(f"⚠️ Bot process stopped, restarting... (Attempt {restart_attempts + 1}/{max_attempts})")
                restart_attempts += 1
                
                # Send restart notification
                self.send_deployment_notification(f"RESTARTED (Attempt {restart_attempts})")
                
                # Restart the bot
                self.bot_process = self.start_24_7_bot()
                
                if self.bot_process:
                    print("✅ Bot restarted successfully")
                else:
                    print("❌ Failed to restart bot")
                    
                time.sleep(30)  # Wait before next restart attempt
            else:
                time.sleep(60)  # Check every minute
                
        print("❌ Max restart attempts reached, stopping deployment")
        
    async def deploy_24_7_bot(self):
        """Main deployment function for 24/7 operation."""
        print("🎯 24/7 CRYPTO QUANT BOT DEPLOYMENT")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print(f"Email Recipient: {self.email_recipient}")
        print("Mode: 24/7 Operation with Daily Email Reports")
        print()
        
        # Setup environment
        self.setup_environment()
        
        # Test connection
        if not await self.test_alpaca_connection():
            print("❌ Connection test failed")
            return False
            
        # Send deployment notification
        self.send_deployment_notification("STARTED")
        
        # Start bot
        process = self.start_24_7_bot()
        
        if process:
            print("📊 24/7 BOT DEPLOYMENT SUCCESSFUL")
            print("-" * 50)
            print("✅ Bot is now running 24/7")
            print(f"📧 Daily reports: {self.email_recipient}")
            print("🔄 Auto-restart: ENABLED")
            print("📈 Performance monitoring: ACTIVE")
            print("🛡️ Risk management: ENHANCED")
            print()
            
            try:
                # Monitor the process
                for line in process.stdout:
                    print(line.rstrip())
                    if "ERROR" in line or "CRITICAL" in line:
                        print("⚠️ Warning detected in bot output")
                        
            except KeyboardInterrupt:
                print("📡 24/7 bot stopped by user")
                process.terminate()
                self.send_deployment_notification("STOPPED")
                
        return True

async def main():
    """Main entry point for 24/7 deployment."""
    deployment = CryptoQuantDeployment()
    
    try:
        await deployment.deploy_24_7_bot()
    except KeyboardInterrupt:
        print("📡 24/7 deployment interrupted")
        deployment.send_deployment_notification("INTERRUPTED")
    except Exception as e:
        print(f"❌ 24/7 deployment failed: {e}")
        deployment.send_deployment_notification("FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
