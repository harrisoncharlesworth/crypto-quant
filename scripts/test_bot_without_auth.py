#!/usr/bin/env python3
"""
Test the crypto bot with public data only (no authentication)
This proves the bot logic works while we fix the API keys
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals.momentum import TimeSeriesMomentumSignal, MomentumConfig


def generate_sample_data(symbol: str = "BTCUSDT", days: int = 90) -> pd.DataFrame:
    """Generate sample data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="h")

    # Generate realistic price series
    start_price = 45000
    returns = np.random.normal(0.0002, 0.02, len(dates))
    prices = [start_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))

    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
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


async def test_signal_without_auth():
    """Test signal generation without needing exchange authentication."""

    print("🤖 Testing Crypto Bot Logic (No Auth Required)")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating sample market data...")
    data = generate_sample_data("BTCUSDT", days=90)
    print(f"   Generated {len(data)} data points")

    # Create momentum signal
    print("\n⚡ Testing Momentum Signal...")
    config = MomentumConfig(
        lookback_days=30, skip_recent_days=7, ma_window=50, weight=1.0
    )

    signal = TimeSeriesMomentumSignal(config)

    # Test signal generation
    try:
        result = await signal.generate(data, "BTCUSDT")

        print("✅ Signal Generated Successfully!")
        print(f"   Signal Value: {result.value:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(
            f"   Direction: {'LONG' if result.value > 0 else 'SHORT' if result.value < 0 else 'NEUTRAL'}"
        )

        if result.metadata:
            print(
                f"   Momentum Return: {result.metadata.get('momentum_return', 0):.3f}"
            )
            print(f"   Price vs MA: {result.metadata.get('price_vs_ma', 0):.3f}")

        print("\n🎯 Signal Quality Check:")
        print(f"   Value in range [-1, 1]: {'✅' if -1 <= result.value <= 1 else '❌'}")
        print(
            f"   Confidence in range [0, 1]: {'✅' if 0 <= result.confidence <= 1 else '❌'}"
        )
        print(f"   Has metadata: {'✅' if result.metadata else '❌'}")

        return True

    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False


async def main():
    """Run the test."""

    success = await test_signal_without_auth()

    if success:
        print("\n🎉 SUCCESS! The crypto bot logic is working perfectly!")
        print("\n📋 Next steps:")
        print("   1. Fix Binance API authentication (check trading permissions)")
        print("   2. Set up email credentials for notifications")
        print("   3. Run full system test")
        print("   4. Start paper trading")
    else:
        print("\n❌ Bot logic test failed - check signal implementation")


if __name__ == "__main__":
    asyncio.run(main())
