#!/usr/bin/env python3
"""
Live crypto trading bot runner for Railway deployment.
Runs continuously with proper error handling and monitoring.
"""

import sys
import os
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, Any
import traceback
import threading
from flask import Flask, jsonify

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from quantbot.signals.momentum import TimeSeriesMomentumSignal, MomentumConfig
from quantbot.signals.mean_reversion import ShortTermMeanReversionSignal, MeanReversionConfig
from quantbot.signals.funding_carry import PerpFundingCarrySignal, FundingCarryConfig
from quantbot.portfolio.blender_v2 import PortfolioBlenderV2, BlenderConfigV2, AllocationMethod
from quantbot.notifications.email import notifier
import ccxt
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/trading_bot.log') if os.path.exists('/tmp') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Health check server
app = Flask(__name__)

@app.route('/health')
def health():
    """Health check endpoint for Railway."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

class CryptoTradingBot:
    """Main trading bot for Railway deployment."""
    
    def __init__(self):
        self.running = True
        self.exchange = None
        self.signals = {}
        self.blender = None
        self.last_signal_time = None
        self.main_task = None
        
        # Trading configuration
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.update_interval = int(os.getenv('UPDATE_INTERVAL_MINUTES', '15'))  # 15 minutes default
        self.symbols = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,LTCUSDT,MATICUSDT,XRPUSDT').split(',')
        
        logger.info(f"Bot initialized - DRY_RUN: {self.dry_run}, Interval: {self.update_interval}min")
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")
    
    async def setup_exchange(self):
        """Setup Binance exchange connection."""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            secret = os.getenv('BINANCE_SECRET')
            sandbox = os.getenv('BINANCE_SANDBOX', 'true').lower() == 'true'
            
            if not api_key or not secret:
                raise ValueError("Binance API credentials not configured")
            
            config = {
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
            }
            
            if sandbox:
                config['test'] = True
                logger.info("Using Binance testnet")
            
            self.exchange = ccxt.binance(config)
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info("✅ Exchange connection established")
            
            # Send startup notification
            await notifier.send_email(
                subject="🚀 Crypto Bot Deployed on Railway",
                body=f"""
Your crypto quantitative trading bot is now live on Railway!

🤖 Status: {"Paper Trading" if self.dry_run else "Live Trading"}
📊 Symbols: {', '.join(self.symbols)}
⏰ Update Interval: {self.update_interval} minutes
🏦 Exchange: Binance {"Testnet" if sandbox else "Live"}
💰 USDT Balance: {balance.get('USDT', {}).get('free', 0):.2f}

The bot will now monitor markets and generate trading signals automatically.
                """.strip()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup exchange: {e}")
            await notifier.send_risk_alert(
                message=f"Exchange setup failed: {e}",
                severity="CRITICAL"
            )
            return False
    
    def setup_signals(self):
        """Setup trading signals."""
        try:
            # Momentum signal
            momentum_config = MomentumConfig(
                lookback_days=30,
                skip_recent_days=7,
                ma_window=50,
                weight=1.0
            )
            self.signals['momentum'] = TimeSeriesMomentumSignal(momentum_config)
            
            # Mean reversion signal
            mr_config = MeanReversionConfig(
                lookback_days=3,
                zscore_threshold=2.0,
                min_liquidity_volume=1_000_000,
                weight=0.8
            )
            self.signals['mean_reversion'] = ShortTermMeanReversionSignal(mr_config)
            
            # Funding carry signal (if not in dry run)
            if not self.dry_run:
                funding_config = FundingCarryConfig(
                    funding_threshold=0.0007,  # 0.07%
                    max_allocation=0.20,
                    weight=1.5
                )
                self.signals['funding_carry'] = PerpFundingCarrySignal(funding_config)
            
            # Setup portfolio blender
            blender_config = BlenderConfigV2(
                allocation_method=AllocationMethod.RISK_PARITY,
                max_net_exposure=0.30,
                correlation_threshold=0.8
            )
            self.blender = PortfolioBlenderV2(blender_config)
            
            logger.info(f"✅ Signals setup complete: {list(self.signals.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup signals: {e}")
            return False
    
    async def get_market_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch market data from exchange with exponential backoff."""
        for retry in range(5):
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe='1h', 
                    limit=days * 24
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
                
            except ccxt.NetworkError as e:
                if retry < 4:
                    wait_time = 2 ** retry
                    logger.warning(f"Network error for {symbol}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch data for {symbol} after 5 retries: {e}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                return pd.DataFrame()
    
    async def generate_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate signals for a symbol."""
        try:
            # Get market data
            data = await self.get_market_data(symbol)
            if data.empty:
                return {}
            
            # Generate signals
            signal_results = {}
            for signal_name, signal_obj in self.signals.items():
                try:
                    result = await signal_obj.generate(data, symbol)
                    signal_results[signal_name] = result
                except Exception as e:
                    logger.error(f"Signal {signal_name} failed for {symbol}: {e}")
            
            # Blend signals
            if signal_results:
                blended = self.blender.blend_signals(signal_results, symbol)
                return {
                    'symbol': symbol,
                    'final_position': blended.final_position,
                    'confidence': blended.confidence,
                    'individual_signals': {name: result.value for name, result in signal_results.items()},
                    'metadata': blended.metadata
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return {}
    
    async def execute_trades(self, signals: Dict[str, Any]):
        """Execute trades based on signals."""
        try:
            for symbol_data in signals.values():
                symbol = symbol_data.get('symbol')
                position = symbol_data.get('final_position', 0)
                confidence = symbol_data.get('confidence', 0)
                
                if abs(position) < 0.1:  # Neutral position
                    continue
                
                # Calculate position size
                max_position_size = float(os.getenv('MAX_POSITION_SIZE', '100'))
                position_size = abs(position) * confidence * max_position_size
                
                action = "BUY" if position > 0 else "SELL"
                
                if self.dry_run:
                    logger.info(f"DRY RUN: {action} {position_size:.2f} USDT of {symbol}")
                    
                    # Send notification
                    await notifier.send_trade_alert(
                        symbol=symbol,
                        action=action.lower(),
                        price=0.0,  # We'd get this from ticker
                        size=position_size,
                        reason=f"Signal strength: {position:.3f}, confidence: {confidence:.3f} (DRY RUN)"
                    )
                else:
                    # TODO: Implement actual trading logic
                    logger.info(f"LIVE TRADE: {action} {position_size:.2f} USDT of {symbol}")
        
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            await notifier.send_risk_alert(
                message=f"Trade execution error: {e}",
                severity="ERROR"
            )
    
    async def watchdog(self):
        """Internal watchdog to ensure trading loop is running."""
        while self.running:
            try:
                # Check if trading loop is stuck
                if (self.last_signal_time and 
                    datetime.utcnow() - self.last_signal_time > timedelta(minutes=2 * self.update_interval)):
                    logger.error("⚠️ Trading loop stuck - restarting...")
                    
                    # Cancel and restart main task
                    if self.main_task and not self.main_task.done():
                        self.main_task.cancel()
                    
                    self.main_task = asyncio.create_task(self.trading_loop())
                    
                    await notifier.send_risk_alert(
                        message="Trading loop was stuck and has been restarted",
                        severity="WARNING"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(60)

    async def trading_loop(self):
        """Main trading loop."""
        logger.info("🤖 Starting trading loop...")
        
        while self.running:
            try:
                logger.info("📊 Generating trading signals...")
                
                # Generate signals for all symbols
                all_signals = {}
                for symbol in self.symbols:
                    signal_data = await self.generate_signals(symbol)
                    if signal_data:
                        all_signals[symbol] = signal_data
                
                # Execute trades
                if all_signals:
                    await self.execute_trades(all_signals)
                    self.last_signal_time = datetime.utcnow()
                
                # Send daily summary (if it's been 24 hours)
                if (self.last_signal_time and 
                    datetime.utcnow() - self.last_signal_time > timedelta(hours=24)):
                    await notifier.send_daily_summary(
                        pnl=0.0,  # TODO: Calculate actual P&L
                        trades=len(all_signals),
                        signals=list(all_signals.keys())
                    )
                
                # Wait for next iteration
                logger.info(f"💤 Sleeping for {self.update_interval} minutes...")
                await asyncio.sleep(self.update_interval * 60)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.error(traceback.format_exc())
                
                await notifier.send_risk_alert(
                    message=f"Trading loop error: {e}",
                    severity="ERROR"
                )
                
                # Wait before retrying
                await asyncio.sleep(60)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def run(self):
        """Run the trading bot."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            logger.info("🚀 Starting Crypto Quant Trading Bot on Railway...")
            
            # Setup exchange
            if not await self.setup_exchange():
                return
            
            # Setup signals
            if not self.setup_signals():
                return
            
            # Start trading loop and watchdog
            self.main_task = asyncio.create_task(self.trading_loop())
            watchdog_task = asyncio.create_task(self.watchdog())
            
            # Wait for either task to complete
            await asyncio.gather(self.main_task, watchdog_task, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            logger.error(traceback.format_exc())
            
            await notifier.send_risk_alert(
                message=f"Bot crashed: {e}",
                severity="CRITICAL"
            )
        
        finally:
            logger.info("🛑 Trading bot stopped")
            await notifier.send_email(
                subject="🛑 Crypto Bot Stopped",
                body="Your crypto trading bot has stopped running on Railway."
            )

def run_health_server():
    """Run Flask health server in background."""
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

async def main():
    """Main entry point."""
    # Start health server in background thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    logger.info(f"🌐 Health server started on port {os.getenv('PORT', 8080)}")
    
    # Start trading bot
    bot = CryptoTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
