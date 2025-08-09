#!/usr/bin/env python3
"""
Quick deployment script for Alpaca Paper Trading Bot
Immediate setup and launch for 24/7 operation
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment for Alpaca paper trading."""
    logger.info("🔧 Setting up Alpaca Paper Trading Environment...")
    
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
    
    logger.info("✅ Environment configured")

def check_credentials():
    """Check if Alpaca credentials are set."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        logger.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        logger.info("💡 Please set your credentials:")
        logger.info("   export ALPACA_API_KEY='your_api_key'")
        logger.info("   export ALPACA_SECRET_KEY='your_secret_key'")
        return False
    
    logger.info("✅ Alpaca credentials found")
    return True

async def test_connection():
    """Test Alpaca connection."""
    try:
        from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper
        
        logger.info("🔌 Testing Alpaca connection...")
        alpaca = AlpacaWrapper(paper=True)
        alpaca.load_markets()
        
        account = alpaca.trading_client.get_account()
        logger.info(f"✅ Connected to Alpaca Paper Trading")
        logger.info(f"   Account: {account.id}")
        logger.info(f"   Status: {account.status}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

async def start_bot():
    """Start the trading bot."""
    try:
        logger.info("🚀 Starting Crypto Quant Bot...")
        
        # Import and run the bot
        from scripts.run_live_bot import main
        await main()
        
    except KeyboardInterrupt:
        logger.info("📡 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")

async def quick_deploy():
    """Main deployment function."""
    logger.info("🚀 QUICK DEPLOYMENT - ALPACA PAPER TRADING")
    logger.info("=" * 50)
    logger.info(f"Started: {datetime.now()}")
    logger.info("Mode: 24/7 Paper Trading")
    logger.info()
    
    # Setup
    setup_environment()
    
    # Check credentials
    if not check_credentials():
        return False
    
    # Test connection
    if not await test_connection():
        return False
    
    logger.info("✅ Ready to start bot!")
    logger.info("📊 Bot will run 24/7 on Alpaca Paper Trading")
    logger.info("🛑 Press Ctrl+C to stop")
    logger.info()
    
    # Start bot
    await start_bot()
    
    return True

def main():
    """Main entry point."""
    try:
        asyncio.run(quick_deploy())
    except KeyboardInterrupt:
        logger.info("📡 Deployment interrupted")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
