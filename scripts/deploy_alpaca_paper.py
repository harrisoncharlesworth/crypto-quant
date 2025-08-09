#!/usr/bin/env python3
"""
Deployment script for 24/7 Alpaca Paper Trading Bot
Sets up environment variables and launches the bot for continuous operation
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import signal
import subprocess
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("deployment.log")
    ]
)
logger = logging.getLogger(__name__)

class AlpacaPaperDeployment:
    """Deployment manager for Alpaca paper trading bot."""
    
    def __init__(self):
        self.running = True
        self.bot_process = None
        self.restart_count = 0
        self.max_restarts = 10
        
        # Set deployment environment variables
        self.setup_environment()
        
    def setup_environment(self):
        """Set up environment variables for Alpaca paper trading."""
        logger.info("🔧 Setting up Alpaca Paper Trading Environment")
        
        # Core Alpaca settings
        os.environ["ALPACA_PAPER"] = "true"
        os.environ["DRY_RUN"] = "false"  # Paper trading is not dry run
        os.environ["USE_FUTURES"] = "true"
        os.environ["UPDATE_INTERVAL_MINUTES"] = "5"
        
        # Trading configuration
        os.environ["TRADING_SYMBOLS"] = "BTCUSD,ETHUSD,SOLUSD,ADAUSD,LTCUSD,XRPUSD"
        os.environ["MAX_PORTFOLIO_ALLOCATION"] = "0.80"
        
        # Risk management
        os.environ["MAX_NET_EXPOSURE"] = "0.30"
        os.environ["MAX_GROSS_LEVERAGE"] = "2.5"
        os.environ["MAX_SINGLE_POSITION"] = "0.10"
        
        # Notification settings
        os.environ["ENABLE_EMAIL_NOTIFICATIONS"] = "true"
        os.environ["DIGEST_INTERVAL_HOURS"] = "24"
        
        # Logging and monitoring
        os.environ["LOG_LEVEL"] = "INFO"
        os.environ["ENABLE_HEALTH_CHECKS"] = "true"
        
        logger.info("✅ Environment variables configured")
        
    def check_alpaca_credentials(self):
        """Check if Alpaca credentials are properly configured."""
        logger.info("🔑 Checking Alpaca credentials...")
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            logger.error("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
            logger.info("💡 Please set your Alpaca credentials:")
            logger.info("   export ALPACA_API_KEY='your_api_key'")
            logger.info("   export ALPACA_SECRET_KEY='your_secret_key'")
            return False
        
        logger.info("✅ Alpaca credentials found")
        return True
    
    async def test_alpaca_connection(self):
        """Test connection to Alpaca paper trading."""
        logger.info("🔌 Testing Alpaca paper trading connection...")
        
        try:
            # Test Alpaca wrapper
            alpaca = AlpacaWrapper(paper=True)
            alpaca.load_markets()
            
            # Test account access
            account = alpaca.trading_client.get_account()
            logger.info(f"✅ Connected to Alpaca Paper Trading")
            logger.info(f"   Account ID: {account.id}")
            logger.info(f"   Status: {account.status}")
            logger.info(f"   Currency: {account.currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {e}")
            return False
    
    def start_bot(self):
        """Start the trading bot process."""
        logger.info("🚀 Starting Alpaca Paper Trading Bot...")
        
        try:
            # Start the bot as a subprocess
            cmd = [
                sys.executable, "-u", 
                os.path.join(os.path.dirname(__file__), "run_live_bot.py")
            ]
            
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            logger.info(f"✅ Bot started with PID: {self.bot_process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
    
    def monitor_bot(self):
        """Monitor the bot process and restart if needed."""
        logger.info("👀 Starting bot monitoring...")
        
        while self.running and self.restart_count < self.max_restarts:
            if self.bot_process is None or self.bot_process.poll() is not None:
                # Bot has stopped or crashed
                if self.bot_process:
                    exit_code = self.bot_process.poll()
                    logger.warning(f"⚠️  Bot stopped with exit code: {exit_code}")
                
                if self.restart_count < self.max_restarts:
                    self.restart_count += 1
                    logger.info(f"🔄 Restarting bot (attempt {self.restart_count}/{self.max_restarts})")
                    
                    # Wait before restart
                    time.sleep(30)
                    
                    if not self.start_bot():
                        logger.error("❌ Failed to restart bot")
                        break
                else:
                    logger.error("❌ Max restart attempts reached")
                    break
            
            # Monitor bot output
            if self.bot_process and self.bot_process.stdout:
                try:
                    line = self.bot_process.stdout.readline()
                    if line:
                        print(f"[BOT] {line.strip()}")
                except:
                    pass
            
            time.sleep(1)
    
    def stop_bot(self):
        """Stop the trading bot."""
        logger.info("🛑 Stopping bot...")
        self.running = False
        
        if self.bot_process:
            try:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=30)
                logger.info("✅ Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Force killing bot...")
                self.bot_process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"📡 Received signal {signum}, shutting down...")
        self.stop_bot()
        sys.exit(0)
    
    async def deploy(self):
        """Main deployment process."""
        logger.info("🚀 ALPACA PAPER TRADING DEPLOYMENT")
        logger.info("=" * 50)
        logger.info(f"Deployment started: {datetime.now()}")
        logger.info("Mode: 24/7 Paper Trading")
        logger.info("Platform: Alpaca")
        logger.info()
        
        # Check credentials
        if not self.check_alpaca_credentials():
            logger.error("❌ Deployment failed: Missing credentials")
            return False
        
        # Test connection
        if not await self.test_alpaca_connection():
            logger.error("❌ Deployment failed: Connection test failed")
            return False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start bot
        if not self.start_bot():
            logger.error("❌ Deployment failed: Could not start bot")
            return False
        
        logger.info("✅ Deployment successful!")
        logger.info("📊 Bot is now running 24/7 on Alpaca Paper Trading")
        logger.info("📈 Monitor logs for trading activity")
        logger.info("🛑 Press Ctrl+C to stop")
        logger.info()
        
        # Monitor bot
        self.monitor_bot()
        
        return True

def main():
    """Main entry point."""
    deployment = AlpacaPaperDeployment()
    
    try:
        asyncio.run(deployment.deploy())
    except KeyboardInterrupt:
        logger.info("📡 Deployment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
