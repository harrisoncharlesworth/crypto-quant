#!/usr/bin/env python3
"""
Demo script for the Perp Funding Carry signal.

Demonstrates usage and integration with the portfolio blender.
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals.funding_carry import PerpFundingCarrySignal, FundingCarryConfig


def create_sample_data(hours: int = 24) -> pd.DataFrame:
    """Create sample OHLCV data for demo."""
    dates = pd.date_range(start="2023-01-01", periods=hours, freq="h")

    data = []
    base_price = 50000

    for i, date in enumerate(dates):
        price = base_price + np.random.normal(0, 200)
        data.append(
            {
                "datetime": date,
                "open": price,
                "high": price + abs(np.random.normal(0, 100)),
                "low": price - abs(np.random.normal(0, 100)),
                "close": price,
                "volume": np.random.uniform(50, 200),
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


async def demo_funding_carry_signal():
    """Demonstrate the funding carry signal with various scenarios."""

    print("🚀 Perp Funding Carry Signal Demo")
    print("=" * 50)

    # Create signal with demo configuration
    config = FundingCarryConfig(
        funding_threshold=0.0007,  # 0.07% threshold
        max_allocation=0.20,  # 20% max allocation
        reversal_stop_hours=6,
        confidence_multiplier=10.0,
    )

    signal = PerpFundingCarrySignal(config)

    # Create sample market data
    data = create_sample_data(24)

    print(f"📊 Market Data: {len(data)} hours of OHLCV data")
    print(f"💰 Price Range: ${data['close'].min():.0f} - ${data['close'].max():.0f}")
    print()

    # Test scenarios with different funding rates
    scenarios = [
        (
            "🟢 Deeply Negative Funding",
            -0.0015,
            "ETHUSDT",
            "Should generate LONG signal",
        ),
        ("🔴 High Positive Funding", 0.0012, "ADAUSDT", "Should generate SHORT signal"),
        ("⚪ Neutral Funding", 0.0003, "BTCUSDT", "Should generate NEUTRAL signal"),
        ("🟡 Extreme Negative", -0.0025, "DOTUSDT", "Strong LONG with risk management"),
        ("🟠 Emergency Level", 0.025, "RISKUSDT", "Emergency exit scenario"),
    ]

    results = []

    for scenario_name, funding_rate, symbol, expected in scenarios:
        print(f"{scenario_name}")
        print(f"Symbol: {symbol}")
        print(f"Funding Rate: {funding_rate:.4f} ({funding_rate*100:.2f}%)")
        print(f"Expected: {expected}")

        # Mock the funding rate for this scenario
        original_method = signal._get_funding_rate

        async def mock_funding_rate(symbol_param):
            return funding_rate

        signal._get_funding_rate = mock_funding_rate

        # Generate signal
        result = await signal.generate(data, symbol)

        # Restore original method
        signal._get_funding_rate = original_method

        # Display results
        print(f"📈 Signal Value: {result.value:.4f}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"📊 Direction: {result.metadata['signal_direction']}")
        print(f"⚖️ Risk Adjustment: {result.metadata['risk_adjustment']:.3f}")
        print()

        results.append(
            {
                "scenario": scenario_name,
                "symbol": symbol,
                "funding_rate": funding_rate,
                "signal_value": result.value,
                "confidence": result.confidence,
                "direction": result.metadata["signal_direction"],
            }
        )

    # Summary table
    print("📋 SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Scenario':<25} {'Symbol':<10} {'Funding%':<10} {'Signal':<8} {'Conf':<6} {'Direction':<8}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['scenario']:<25} {r['symbol']:<10} {r['funding_rate']*100:>7.2f}% "
            f"{r['signal_value']:>7.3f} {r['confidence']:>5.2f} {r['direction']:<8}"
        )

    print()

    # Test strategy metrics
    print("📊 STRATEGY METRICS DEMO")
    print("=" * 30)

    # Simulate some trading history
    funding_history = [-0.0008, -0.0006, -0.0004, 0.0002, 0.0008, -0.0003]

    for rate in funding_history:
        signal._update_funding_history("DEMO", rate)

    metrics = signal.get_strategy_metrics("DEMO")

    print(f"Current Funding: {metrics.get('current_funding', 'N/A')}")
    print(f"24h Avg Funding: {metrics.get('avg_funding_24h', 'N/A')}")
    print(f"Funding Volatility: {metrics.get('funding_volatility', 'N/A')}")
    print(f"Direction Changes: {metrics.get('funding_direction_changes', 'N/A')}")
    print(f"Extreme Periods: {metrics.get('extreme_funding_periods', 'N/A')}")

    print()
    print("✅ Demo completed successfully!")
    print("💡 This signal is ready for integration with the portfolio blender.")


if __name__ == "__main__":
    asyncio.run(demo_funding_carry_signal())
