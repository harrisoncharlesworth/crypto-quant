#!/usr/bin/env python3
"""
Test script for momentum signal parameter grid search.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals.momentum import TimeSeriesMomentumSignal


def generate_test_data(days: int = 180) -> pd.DataFrame:
    """Generate test data for grid search."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="h")

    # Generate realistic price series
    returns = np.random.normal(0.0002, 0.02, len(dates))
    prices = [50000.0]

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


def main():
    print("🔍 Testing Momentum Signal Parameter Grid Search")

    # Generate test data
    data = generate_test_data(days=180)
    print(f"Generated {len(data)} data points")

    # Run grid search
    print("⚡ Running parameter grid search...")
    results = TimeSeriesMomentumSignal.grid_search_parameters(data, "BTCUSDT")

    print("\n📊 Grid Search Results:")
    if results["best_params"]:
        best = results["best_params"]
        print("✅ Best Parameters:")
        print(f"   Lookback Days: {best['lookback_days']}")
        print(f"   Skip Recent Days: {best['skip_recent_days']}")
        print(f"   MA Window: {best['ma_window']}")
        print(f"   Score: {best['score']:.4f}")

        print("\n🏆 Top 5 Parameter Combinations:")
        sorted_results = sorted(
            results["all_results"], key=lambda x: x["score"], reverse=True
        )
        for i, result in enumerate(sorted_results[:5]):
            print(
                f"   {i+1}. Lookback: {result['lookback_days']}, Skip: {result['skip_recent_days']}, "
                f"MA: {result['ma_window']}, Score: {result['score']:.4f}"
            )
    else:
        print("❌ No valid parameter combinations found")


if __name__ == "__main__":
    main()
