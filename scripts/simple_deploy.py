#!/usr/bin/env python3
"""
Simple deployment script for Alpaca Paper Trading Bot
"""

import os
import sys
import asyncio
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def setup_environment():
    """Set up environment for Alpaca paper trading."""
    print("🔧 Setting up Alpaca Paper Trading Environment...")
    
    # Core settings
    os.environ["ALPACA_PAPER"] = "true"
    os.environ["DRY_RUN"] = "false"
    os.environ["USE_FUTURES"] = "true"
    os.environ["UPDATE_INTERVAL_MINUTES"] = "5"
    
    # Trading pairs
    os.environ["TRADING_SYMBOLS"] = "BTCUSD,ETHUSD,SOLUSD,ADAUSD,LTCUSD,XRPUSD"
    os.environ["MAX_PORTFOLIO_ALLOCATION"] = "0.80"
    
    # Risk management
    os.environ["MAX_NET_EXPOSURE"] = "0.30"
    os.environ["MAX_GROSS_LEVERAGE"] = "2.5"
    os.environ["MAX_SINGLE_POSITION"] = "0.10"
    
    # Notifications
    os.environ["ENABLE_EMAIL_NOTIFICATIONS"] = "true"
    os.environ["DIGEST_INTERVAL_HOURS"] = "24"
    
    # Logging
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["ENABLE_HEALTH_CHECKS"] = "true"
    
    print("✅ Environment configured")

def check_credentials():
    """Check if Alpaca credentials are set."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return False
    
    print("✅ Alpaca credentials found")
    return True

async def test_connection():
    """Test Alpaca connection."""
    try:
        from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper
        
        print("🔌 Testing Alpaca connection...")
        alpaca = AlpacaWrapper(paper=True)
        alpaca.load_markets()
        
        account = alpaca.trading_client.get_account()
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"   Account: {account.id}")
        print(f"   Status: {account.status}")
        
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def start_bot():
    """Start the trading bot."""
    try:
        print("🚀 Starting Crypto Quant Bot...")
        
        # Import and run the bot
        import subprocess
        import sys
        
        # Run the bot directly
        cmd = [sys.executable, "scripts/run_live_bot.py"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("📡 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")

async def simple_deploy():
    """Main deployment function."""
    print("🚀 SIMPLE DEPLOYMENT - ALPACA PAPER TRADING")
    print("=" * 50)
    print(f"Started: {datetime.now()}")
    print("Mode: 24/7 Paper Trading")
    print()
    
    # Setup
    setup_environment()
    
    # Check credentials
    if not check_credentials():
        return False
    
    # Test connection
    if not await test_connection():
        return False
    
    print("✅ Ready to start bot!")
    print("📊 Bot will run 24/7 on Alpaca Paper Trading")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    # Start bot
    await start_bot()
    
    return True

def main():
    """Main entry point."""
    try:
        asyncio.run(simple_deploy())
    except KeyboardInterrupt:
        print("📡 Deployment interrupted")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
