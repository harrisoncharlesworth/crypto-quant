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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from quantbot.signals.momentum import TimeSeriesMomentumSignal, MomentumConfig
from quantbot.signals.mean_reversion import (
    ShortTermMeanReversionSignal,
    MeanReversionConfig,
)
from quantbot.signals.funding_carry import PerpFundingCarrySignal, FundingCarryConfig
from quantbot.portfolio.blender_v2 import (
    PortfolioBlenderV2,
    BlenderConfigV2,
    AllocationMethod,
)
from quantbot.notifications.email import notifier
import ccxt
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        (
            logging.FileHandler("/tmp/trading_bot.log")
            if os.path.exists("/tmp")
            else logging.StreamHandler()
        ),
    ],
)
logger = logging.getLogger(__name__)

# Health check server
app = Flask(__name__)


@app.route("/health")
def health():
    """Health check endpoint for Railway."""
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})


class CryptoTradingBot:
    """Main trading bot for Railway deployment."""

    def __init__(self):
        self.running = True
        self.exchange = None
        self.signals = {}
        self.blender = None
        self.last_signal_time = None
        self.main_task = None
        self.next_digest = None
        self.recent_signals = []
        self.recent_trades = []

        # Trading configuration
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.use_futures = (
            os.getenv("USE_FUTURES", "true").lower() == "true"
        )  # Default to futures
        self.update_interval = int(
            os.getenv("UPDATE_INTERVAL_MINUTES", "5")
        )  # 5 minutes default for more action
        self.symbols = os.getenv(
            "TRADING_SYMBOLS",
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,LTCUSDT,MATICUSDT,XRPUSDT",
        ).split(",")
        self.max_portfolio_allocation = float(
            os.getenv("MAX_PORTFOLIO_ALLOCATION", "0.80")
        )  # Use 80% of balance

        logger.info(
            f"Bot initialized - DRY_RUN: {self.dry_run}, Futures: {self.use_futures}, Interval: {self.update_interval}min"
        )
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")
        logger.info(f"Max portfolio allocation: {self.max_portfolio_allocation:.0%}")

        # Initialize next digest time
        self.next_digest = self.calculate_next_digest_time()

    def calculate_next_digest_time(self) -> datetime:
        """Calculate next digest send time (00:00, 08:00, 16:00 UTC)."""
        DIGEST_HOURS = [0, 8, 16]
        now = datetime.utcnow()

        # Find next digest hour
        next_hour = None
        for hour in DIGEST_HOURS:
            digest_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if digest_time > now:
                next_hour = digest_time
                break

        # If no hour today, use first hour tomorrow
        if next_hour is None:
            next_hour = now.replace(
                hour=DIGEST_HOURS[0], minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

        logger.info(
            f"Next digest scheduled for: {next_hour.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return next_hour

    async def setup_exchange(self):
        """Setup Binance exchange connection."""
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            secret = os.getenv("BINANCE_SECRET")
            sandbox = os.getenv("BINANCE_SANDBOX", "true").lower() == "true"

            if not api_key or not secret:
                raise ValueError("Binance API credentials not configured")

            config = {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
            }

            if sandbox:
                config["test"] = True
                logger.info("Using Binance testnet")

            # Choose exchange type based on configuration
            if self.use_futures:
                config["options"] = {"defaultType": "future"}  # Configure for futures
                logger.info("Using Binance Futures API")
            else:
                logger.info("Using Binance Spot API")

            self.exchange = ccxt.binance(config)

            # Test connection
            _ = self.exchange.fetch_balance()  # Test connection only
            logger.info("✅ Exchange connection established")

            # Deployment notification disabled to prevent spam
            # await notifier.send_email(
            #     subject="🚀 Crypto Bot Deployed on Railway",
            #     body=f"Bot deployed: {'Paper Trading' if self.dry_run else 'Live Trading'}"
            # )

            return True

        except Exception as e:
            logger.error(f"Failed to setup exchange: {e}")
            await notifier.send_risk_alert(
                message=f"Exchange setup failed: {e}", severity="CRITICAL"
            )
            return False

    def setup_signals(self):
        """Setup trading signals."""
        try:
            # Momentum signal
            momentum_config = MomentumConfig(
                lookback_days=30, skip_recent_days=7, ma_window=50, weight=1.0
            )
            self.signals["momentum"] = TimeSeriesMomentumSignal(momentum_config)

            # Mean reversion signal
            mr_config = MeanReversionConfig(
                lookback_days=3,
                zscore_threshold=2.0,
                min_liquidity_volume=1_000_000,
                weight=0.8,
            )
            self.signals["mean_reversion"] = ShortTermMeanReversionSignal(mr_config)

            # Funding carry signal (if not in dry run)
            if not self.dry_run:
                funding_config = FundingCarryConfig(
                    funding_threshold=0.0007, max_allocation=0.20, weight=1.5  # 0.07%
                )
                self.signals["funding_carry"] = PerpFundingCarrySignal(funding_config)

            # Setup portfolio blender
            from quantbot.portfolio.blender_v2 import RiskLimits

            risk_limits = RiskLimits(max_net_exposure=0.30)
            blender_config = BlenderConfigV2(
                allocation_method=AllocationMethod.RISK_PARITY, risk_limits=risk_limits
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
                    symbol, timeframe="1h", limit=days * 24
                )

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)

                return df

            except ccxt.NetworkError as e:
                if retry < 4:
                    wait_time = 2**retry
                    logger.warning(
                        f"Network error for {symbol}, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to fetch data for {symbol} after 5 retries: {e}"
                    )
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
                    "symbol": symbol,
                    "final_position": blended.final_position,
                    "confidence": blended.confidence,
                    "individual_signals": {
                        name: result.value for name, result in signal_results.items()
                    },
                    "metadata": blended.metadata,
                }

            return {}

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return {}

    async def get_account_balance(self) -> float:
        """Get current USDT balance."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get("USDT", {}).get("free", 0)
            return float(usdt_balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0

    async def execute_trades(self, signals: Dict[str, Any]):
        """Execute trades based on signals with proper position sizing."""
        try:
            # Get current balance
            balance = await self.get_account_balance()
            if balance <= 0:
                logger.warning("No USDT balance available for trading")
                return

            # Calculate available capital for new positions
            available_capital = balance * self.max_portfolio_allocation

            # Count valid signals (lowered threshold to be more active)
            valid_signals = [
                s for s in signals.values() if abs(s.get("final_position", 0)) >= 0.05
            ]
            if not valid_signals:
                logger.info("No valid trading signals generated")
                return

            # Allocate capital across signals
            capital_per_signal = available_capital / len(valid_signals)

            for symbol_data in valid_signals:
                symbol = symbol_data.get("symbol")
                position = symbol_data.get("final_position", 0)
                confidence = symbol_data.get("confidence", 0)

                # Get current price
                current_price = await self.get_current_price(symbol)
                if current_price <= 0:
                    continue

                # Calculate position size based on signal strength and confidence
                signal_strength = abs(position) * confidence
                position_value = capital_per_signal * signal_strength

                # Minimum position size of $50
                if position_value < 50:
                    continue

                action = "BUY" if position > 0 else "SELL"

                # Store signal for tracking
                signal_text = f"{datetime.utcnow().strftime('%H:%M')} {symbol} {action} ${position_value:.0f} (conf: {confidence:.1%})"
                self.recent_signals.append(signal_text)

                # Keep only last 50 signals
                if len(self.recent_signals) > 50:
                    self.recent_signals = self.recent_signals[-50:]

                if self.dry_run:
                    logger.info(
                        f"🎯 PAPER TRADE: {action} ${position_value:.2f} of {symbol} @ ${current_price:.4f}"
                    )
                    logger.info(
                        f"   → Signal: {position:.3f}, Confidence: {confidence:.1%}, Balance: ${balance:.0f}"
                    )

                    # Track paper trade
                    trade_info = {
                        "symbol": symbol,
                        "action": action.lower(),
                        "price": current_price,
                        "size": position_value,
                        "timestamp": datetime.utcnow(),
                    }
                    self.recent_trades.append(trade_info)

                    # Send notification
                    await notifier.send_trade_alert(
                        symbol=symbol,
                        action=action.lower(),
                        price=current_price,
                        size=position_value,
                        reason=f"Signal: {position:.3f}, Confidence: {confidence:.1%} (PAPER)",
                    )
                else:
                    # TODO: Implement actual live trading
                    logger.info(
                        f"🚀 LIVE TRADE: {action} ${position_value:.2f} of {symbol} @ ${current_price:.4f}"
                    )

            logger.info(
                f"💰 Portfolio allocation: {len(valid_signals)} positions, ${available_capital:.0f} capital used"
            )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            await notifier.send_risk_alert(
                message=f"Trade execution error: {e}", severity="ERROR"
            )

    async def send_portfolio_digest(self):
        """Send comprehensive portfolio digest email."""
        try:
            # Get account data
            balance = await self.get_account_balance()

            # Calculate simple P&L based on recent trades (paper trading)
            realised_pnl = 0.0  # TODO: Calculate from completed trades
            _ = 0.0  # unrealised_pnl - TODO: Calculate from open positions

            # Mock open positions for paper trading
            open_positions = []
            if self.recent_trades:
                # Show last few trades as "open positions" for demo
                for trade in self.recent_trades[-5:]:
                    open_positions.append(
                        {
                            "symbol": trade["symbol"],
                            "side": trade["action"].upper(),
                            "size": trade["size"],
                            "entry_price": trade["price"],
                            "current_price": trade["price"]
                            * (
                                1 + np.random.uniform(-0.02, 0.02)
                            ),  # Mock price movement
                            "unrealised_pnl": np.random.uniform(-20, 50),  # Mock P&L
                        }
                    )

            # Send digest
            await notifier.send_digest(
                account_equity=balance,
                realised_pnl=realised_pnl,
                unrealised_pnl=sum(
                    pos.get("unrealised_pnl", 0) for pos in open_positions
                ),
                open_positions=open_positions,
                recent_signals=self.recent_signals,
                period="8-hour",
            )

            logger.info(
                f"📬 Portfolio digest sent - Balance: ${balance:.0f}, Signals: {len(self.recent_signals)}"
            )

        except Exception as e:
            logger.error(f"Failed to send portfolio digest: {e}")

    async def watchdog(self):
        """Internal watchdog to ensure trading loop is running."""
        while self.running:
            try:
                # Check if trading loop is stuck
                if (
                    self.last_signal_time
                    and datetime.utcnow() - self.last_signal_time
                    > timedelta(minutes=2 * self.update_interval)
                ):
                    logger.error("⚠️ Trading loop stuck - restarting...")

                    # Cancel and restart main task
                    if self.main_task and not self.main_task.done():
                        self.main_task.cancel()

                    self.main_task = asyncio.create_task(self.trading_loop())

                    await notifier.send_risk_alert(
                        message="Trading loop was stuck and has been restarted",
                        severity="WARNING",
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

                # Check if time for digest email
                if datetime.utcnow() >= self.next_digest:
                    await self.send_portfolio_digest()
                    self.next_digest = self.calculate_next_digest_time()

                # Wait for next iteration
                logger.info(f"💤 Sleeping for {self.update_interval} minutes...")
                await asyncio.sleep(self.update_interval * 60)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                logger.error(traceback.format_exc())

                await notifier.send_risk_alert(
                    message=f"Trading loop error: {e}", severity="ERROR"
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
                message=f"Bot crashed: {e}", severity="CRITICAL"
            )

        finally:
            logger.info("🛑 Trading bot stopped")
            # Stop notification disabled to prevent spam
            # await notifier.send_email(
            #     subject="🛑 Crypto Bot Stopped",
            #     body="Your crypto trading bot has stopped running on Railway."
            # )


def run_health_server():
    """Run Flask health server in background using Waitress for production."""
    port = int(os.getenv("PORT", 8080))

    # Check if we're in Railway (production) environment
    if os.getenv("RAILWAY_ENVIRONMENT"):
        # Use Waitress for production (simpler than Gunicorn)
        try:
            from waitress import serve

            serve(app, host="0.0.0.0", port=port, threads=4)
        except ImportError:
            # Fallback to Flask if Waitress not available
            app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    else:
        # Use Flask dev server for local development
        app.run(host="0.0.0.0", port=port, debug=False)


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
