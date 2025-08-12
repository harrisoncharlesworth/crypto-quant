#!/usr/bin/env python3
"""
Optimized Deployment Script for Crypto Quant Bot
Implements strategic recommendations with enhanced risk management
"""

import os
import sys
import asyncio
import subprocess
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optimized_bot_config import OptimizedBotConfig

class OptimizedDeployment:
    """Optimized deployment with strategic recommendations implemented."""
    
    def __init__(self):
        self.config = OptimizedBotConfig()
        self.bot = None
        
    def setup_environment(self):
        """Set up optimized environment for deployment."""
        print("🔧 Setting up Optimized Deployment Environment...")
        
        # Set Alpaca credentials (from previous deployment)
        os.environ["ALPACA_API_KEY"] = "PKJYFI6XVZ9UGW85JFVP"
        os.environ["ALPACA_SECRET_KEY"] = "HY2qdRNAX8TSONDcnsqPfNHUp7WzpeHzpsAdNEGZ"
        
        # Apply optimized configuration
        self.config.setup_environment()
        
        print("✅ Environment configured with strategic optimizations")
    
    async def test_connection(self):
        """Test Alpaca connection with optimized settings."""
        try:
            from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper
            
            print("🔌 Testing Optimized Alpaca Connection...")
            alpaca = AlpacaWrapper(paper=True)
            alpaca.load_markets()
            
            account = alpaca.trading_client.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   Account: {account.id}")
            print(f"   Status: {account.status}")
            print(f"   Cash: ${float(account.cash):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def create_optimized_bot(self):
        """Create the optimized trading bot."""
        print("🚀 Creating Optimized Trading Bot...")
        
        self.bot = self.config.create_optimized_bot()
        
        print("✅ Optimized bot created with:")
        print(f"   • {len(self.bot['signals'])} enhanced signals")
        print(f"   • Risk limits: {self.bot['risk_limits']}")
        print(f"   • Allocation method: {self.bot['config'].allocation_method}")
        
        return self.bot
    
    def start_optimized_bot(self):
        """Start the optimized bot with enhanced monitoring."""
        try:
            print("🚀 Starting Optimized Crypto Quant Bot...")
            
            # Create optimized configuration file
            self.create_optimized_config_file()
            
            # Start the bot with optimized settings
            cmd = [sys.executable, "-u", "scripts/run_live_bot.py"]
            
            # Set environment for optimized deployment
            env = os.environ.copy()
            env["OPTIMIZED_MODE"] = "true"
            env["STRATEGIC_RECOMMENDATIONS"] = "implemented"
            
            print("📊 Starting with optimized configuration...")
            print("   • Enhanced risk management: ACTIVE")
            print("   • Breakout focus: MAINTAINED")
            print("   • Funding carry: EXPANDED")
            print("   • New risk controls: IMPLEMENTED")
            
            # Start the bot process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            print("✅ Optimized bot started successfully!")
            print("📈 Monitoring enhanced performance metrics...")
            
            # Monitor the process
            for line in process.stdout:
                print(line.rstrip())
                if "ERROR" in line or "CRITICAL" in line:
                    print("⚠️  Warning detected in bot output")
            
            return process
            
        except Exception as e:
            print(f"❌ Failed to start optimized bot: {e}")
            return None
    
    def create_optimized_config_file(self):
        """Create optimized configuration file for the bot."""
        config_content = """
# Optimized Crypto Quant Bot Configuration
# Implements strategic recommendations from 6-month performance analysis

# Core Settings
ALPACA_PAPER=true
DRY_RUN=false
USE_FUTURES=true
UPDATE_INTERVAL_MINUTES=10

# Trading Symbols (Focused on high-liquidity)
TRADING_SYMBOLS=BTCUSD,ETHUSD,SOLUSD,ADAUSD

# Enhanced Risk Management (STRATEGIC RECOMMENDATION #1)
MAX_PORTFOLIO_ALLOCATION=0.60
MAX_NET_EXPOSURE=0.20
MAX_GROSS_LEVERAGE=1.8
MAX_SINGLE_POSITION=0.06

# New Risk Controls
MAX_DAILY_DRAWDOWN=0.15
MAX_WEEKLY_DRAWDOWN=0.25
POSITION_SIZING_METHOD=KELLY_OPTIMAL

# Signal Configurations
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
CONFLICT_RESOLUTION=CONFIDENCE_BASED
REBALANCE_FREQUENCY=DAILY

# Monitoring and Alerts
ENABLE_EMAIL_NOTIFICATIONS=true
DIGEST_INTERVAL_HOURS=12
LOG_LEVEL=INFO
ENABLE_HEALTH_CHECKS=true
PERFORMANCE_MONITORING=true

# Performance Targets
TARGET_MONTHLY_RETURN=0.15
MAX_DRAWDOWN_TARGET=0.25
TARGET_SHARPE_RATIO=0.8
WIN_RATE_TARGET=0.55
VOLATILITY_TARGET=0.35
ALPHA_TARGET=0.20
"""
        
        with open("optimized_config.env", "w") as f:
            f.write(config_content)
        
        print("✅ Optimized configuration file created")
    
    def get_performance_monitoring(self):
        """Set up performance monitoring dashboard."""
        print("📊 Setting up Performance Monitoring Dashboard...")
        
        targets = self.config.get_performance_targets()
        metrics = self.config.get_monitoring_metrics()
        
        print("🎯 PERFORMANCE TARGETS")
        print("-" * 40)
        for key, value in targets.items():
            if 'return' in key or 'drawdown' in key or 'rate' in key or 'alpha' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.1f}")
        
        print("\n📈 MONITORING METRICS")
        print("-" * 40)
        for period, metric_list in metrics.items():
            print(f"{period.replace('_', ' ').title()}:")
            for metric in metric_list:
                print(f"  • {metric.replace('_', ' ').title()}")
        
        print("\n✅ Performance monitoring configured")
    
    async def deploy_optimized_bot(self):
        """Main deployment function with all optimizations."""
        print("🎯 OPTIMIZED DEPLOYMENT - STRATEGIC RECOMMENDATIONS")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print("Mode: Enhanced Risk Management + Strategic Optimizations")
        print()
        
        # Setup
        self.setup_environment()
        
        # Test connection
        if not await self.test_connection():
            print("❌ Connection test failed")
            return False
        
        # Create optimized bot
        self.create_optimized_bot()
        
        # Setup monitoring
        self.get_performance_monitoring()
        
        print("✅ Ready to start optimized bot!")
        print("📊 Bot will run with strategic recommendations implemented")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        # Start bot
        process = self.start_optimized_bot()
        
        if process:
            try:
                process.wait()
            except KeyboardInterrupt:
                print("📡 Optimized bot stopped by user")
                process.terminate()
        
        return True

async def main():
    """Main entry point for optimized deployment."""
    deployment = OptimizedDeployment()
    
    try:
        await deployment.deploy_optimized_bot()
    except KeyboardInterrupt:
        print("📡 Optimized deployment interrupted")
    except Exception as e:
        print(f"❌ Optimized deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
