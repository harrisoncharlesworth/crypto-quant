#!/usr/bin/env python3
"""
Simple backtesting script for crypto quant signals.
Usage: python -m scripts.backtest --symbol BTCUSDT --signal momentum
"""

import asyncio
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals.momentum import TimeSeriesMomentumSignal, MomentumConfig
from quantbot.signals.breakout import DonchianBreakoutSignal, BreakoutConfig
from quantbot.signals.mean_reversion import (
    ShortTermMeanReversionSignal,
    MeanReversionConfig,
)
from quantbot.portfolio.blender import (
    PortfolioBlender,
    BlenderConfig,
    ConflictResolution,
)
from quantbot.config import load_config


def generate_sample_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for backtesting."""
    np.random.seed(42)  # Reproducible results

    # Start from a reasonable crypto price
    start_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="H")

    # Generate realistic price series with trend and volatility
    returns = np.random.normal(0.0002, 0.02, len(dates))  # Hourly returns
    prices = [start_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Prevent negative prices

    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Simulate intraday price action
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": max(open_price, high, close),
                "low": min(open_price, low, close),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def create_signal(signal_name: str, symbol: str):
    """Create and configure a trading signal."""

    if signal_name == "momentum":
        config = MomentumConfig(
            lookback_days=90, skip_recent_days=7, ma_window=200, weight=1.0
        )
        return TimeSeriesMomentumSignal(config)

    elif signal_name == "breakout":
        config = BreakoutConfig(
            channel_period=55, atr_period=14, atr_multiplier=2.0, weight=1.0
        )
        return DonchianBreakoutSignal(config)

    elif signal_name == "mean_reversion":
        config = MeanReversionConfig(
            lookback_days=3,
            zscore_threshold=2.0,
            min_liquidity_volume=1_000_000,
            weight=1.0,
        )
        return ShortTermMeanReversionSignal(config)

    elif signal_name == "multi":
        # Return all signals for blending
        return {
            "momentum": create_signal("momentum", symbol),
            "breakout": create_signal("breakout", symbol),
            "mean_reversion": create_signal("mean_reversion", symbol),
        }

    else:
        raise ValueError(f"Unknown signal: {signal_name}")


async def run_single_signal_backtest(symbol: str, signal_name: str, data: pd.DataFrame):
    """Run backtest for a single signal."""

    signal_obj = create_signal(signal_name, symbol)
    if isinstance(signal_obj, dict):
        raise ValueError(f"Expected single signal for {signal_name}, got dictionary")
    signal = signal_obj

    # Run signal on different time windows
    results = []
    test_points = 50  # Test every N periods

    print(f"⚡ Running {signal_name} signal calculations...")
    for i in range(len(data) // test_points, len(data), test_points):
        window_data = data.iloc[:i]
        if len(window_data) < 100:  # Need minimum data
            continue

        result = await signal.generate(window_data, symbol)
        results.append(
            {
                "timestamp": window_data.index[-1],
                "price": window_data.iloc[-1]["close"],
                "signal_value": result.value,
                "confidence": result.confidence,
                "signal_name": signal_name,
                "metadata": result.metadata,
            }
        )

    return pd.DataFrame(results)


async def run_multi_signal_backtest(symbol: str, data: pd.DataFrame):
    """Run backtest with portfolio blender combining all signals."""

    # Create signals
    signals_dict = create_signal("multi", symbol)
    if not isinstance(signals_dict, dict):
        raise ValueError("Expected dictionary of signals for multi-signal mode")
    signals = signals_dict

    # Create portfolio blender
    blender_config = BlenderConfig(
        conflict_resolution=ConflictResolution.CONFIDENCE_WEIGHTED,
        max_position_size=1.0,
        min_signal_confidence=0.3,
        risk_budget_per_signal={
            "momentum": 0.4,
            "breakout": 0.3,
            "mean_reversion": 0.3,
        },
    )
    blender = PortfolioBlender(blender_config)

    # Run signals and blending
    results = []
    test_points = 50  # Test every N periods

    print("⚡ Running multi-signal portfolio blending...")
    for i in range(len(data) // test_points, len(data), test_points):
        window_data = data.iloc[:i]
        if len(window_data) < 100:  # Need minimum data
            continue

        # Get signals from all sources
        signal_results = {}
        for signal_name, signal in signals.items():
            try:
                signal_results[signal_name] = await signal.generate(window_data, symbol)
            except Exception as e:
                print(f"Error in {signal_name} signal: {e}")
                continue

        # Blend signals
        if signal_results:
            blended = blender.blend_signals(signal_results, symbol)

            results.append(
                {
                    "timestamp": window_data.index[-1],
                    "price": window_data.iloc[-1]["close"],
                    "signal_value": blended.final_position,
                    "confidence": blended.confidence,
                    "signal_name": "blended",
                    "individual_signals": {
                        name: result.value for name, result in signal_results.items()
                    },
                    "signal_contributions": blended.signal_contributions,
                    "metadata": blended.metadata,
                }
            )

    return pd.DataFrame(results)


async def run_backtest(symbol: str, signal_name: str = "momentum"):
    """Run backtest for specified signal(s) and symbol."""

    print(f"🚀 Starting backtest for {symbol} using {signal_name} signal(s)")

    # Load configuration
    load_config()

    # Generate sample data
    print("📊 Generating sample market data...")
    data = generate_sample_data(symbol, days=365)
    print(
        f"   Generated {len(data)} data points from {data.index[0]} to {data.index[-1]}"
    )

    # Run appropriate backtest
    if signal_name == "multi":
        results_df = await run_multi_signal_backtest(symbol, data)
    else:
        results_df = await run_single_signal_backtest(symbol, signal_name, data)

    if results_df.empty:
        print("❌ No results generated")
        return

    # Set index for analysis
    results_df.set_index("timestamp", inplace=True)

    # Display backtest statistics
    print_backtest_results(results_df, signal_name)


def print_backtest_results(results_df: pd.DataFrame, signal_name: str):
    """Print comprehensive backtest results."""

    # Basic signal statistics
    print(f"\n📈 Backtest Results for {signal_name}:")
    print(f"   Total signals: {len(results_df)}")
    print(f"   Long signals: {len(results_df[results_df['signal_value'] > 0.1])}")
    print(f"   Short signals: {len(results_df[results_df['signal_value'] < -0.1])}")
    print(
        f"   Neutral signals: {len(results_df[abs(results_df['signal_value']) <= 0.1])}"
    )

    # Calculate performance metrics
    results_df["position"] = np.where(
        results_df["signal_value"] > 0.1,
        1,
        np.where(results_df["signal_value"] < -0.1, -1, 0),
    )
    results_df["returns"] = results_df["price"].pct_change()
    results_df["strategy_returns"] = (
        results_df["position"].shift(1) * results_df["returns"]
    )

    # Performance metrics
    try:
        strategy_cum_returns = (1 + results_df["strategy_returns"].fillna(0)).prod()
        total_return = (
            float(strategy_cum_returns - 1) if pd.notna(strategy_cum_returns) else 0.0
        )

        price_ratio = results_df["price"].iloc[-1] / results_df["price"].iloc[0]
        buy_hold_return = float(price_ratio - 1) if pd.notna(price_ratio) else 0.0
    except Exception:
        total_return = 0.0
        buy_hold_return = 0.0

    # Calculate Sharpe ratio (simplified)
    strategy_returns = results_df["strategy_returns"].fillna(0)
    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        )  # Annualized
    else:
        sharpe_ratio = 0.0

    print("\n💰 Performance:")
    print(f"   Strategy Return: {total_return:.2%}")
    print(f"   Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"   Alpha: {total_return - buy_hold_return:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")

    # Signal-specific information
    if signal_name == "multi" and "individual_signals" in results_df.columns:
        print("\n🔄 Signal Contributions (last 5 periods):")
        for _, row in results_df.tail(5).iterrows():
            if pd.notna(row["individual_signals"]):
                individual = row["individual_signals"]
                print(
                    f"   {row.name}: Momentum: {individual.get('momentum', 0):.3f}, "
                    f"Breakout: {individual.get('breakout', 0):.3f}, "
                    f"Mean Rev: {individual.get('mean_reversion', 0):.3f}"
                )

    # Show recent signals
    print("\n🎯 Recent Signals (last 5):")
    for _, row in results_df.tail(5).iterrows():
        direction = (
            "LONG"
            if row["signal_value"] > 0.1
            else "SHORT" if row["signal_value"] < -0.1 else "NEUTRAL"
        )
        timestamp_str = (
            row.name.strftime("%Y-%m-%d %H:%M")
            if hasattr(row.name, "strftime")
            else str(row.name)
        )
        print(
            f"   {timestamp_str} | ${row['price']:,.2f} | {direction} ({row['signal_value']:.3f}, conf: {row['confidence']:.2f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Run crypto signal backtest")
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--signal", default="momentum", help="Signal to test (default: momentum)"
    )

    args = parser.parse_args()

    asyncio.run(run_backtest(args.symbol, args.signal))


if __name__ == "__main__":
    main()
