#!/usr/bin/env python3
"""
Demo script for Portfolio Blender v2 with all 12 crypto signals.

Tests enhanced blending logic, risk management, and performance attribution
across directional, market-neutral, and overlay signal types.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.quantbot.signals.base import SignalResult, SignalConfig
from src.quantbot.portfolio.blender_v2 import (
    PortfolioBlenderV2, BlenderConfigV2, SignalType, AllocationMethod,
    RiskLimits, SignalTypeConfig
)


class MockSignalGenerator:
    """Generate realistic mock signals for all 12 signal types."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.price_history = self._generate_price_series()
        self.current_idx = 0
    
    def _generate_price_series(self, length: int = 1000) -> pd.Series:
        """Generate realistic crypto price series."""
        returns = np.random.normal(0.0001, 0.03, length)  # Daily returns
        returns[::50] += np.random.normal(0, 0.1, len(returns[::50]))  # Add volatility clusters
        
        prices = 40000 * np.exp(np.cumsum(returns))  # Start at $40k BTC
        return pd.Series(prices)
    
    def get_current_price(self) -> float:
        """Get current price for signal generation."""
        return self.price_history.iloc[min(self.current_idx, len(self.price_history) - 1)]
    
    def advance_time(self):
        """Advance time for next signal generation."""
        self.current_idx += 1
    
    def generate_all_signals(self, symbol: str = "BTCUSDT") -> Dict[str, SignalResult]:
        """Generate all 12 signals with realistic correlations and behaviors."""
        current_price = self.get_current_price()
        prev_prices = self.price_history.iloc[max(0, self.current_idx-30):self.current_idx+1]
        
        signals = {}
        
        # 1. Time-Series Momentum (Directional)
        momentum_value = self._calculate_momentum(prev_prices)
        signals["time_series_momentum"] = SignalResult(
            symbol=symbol,
            value=momentum_value,
            confidence=0.7 + 0.2 * abs(momentum_value),
            timestamp=datetime.utcnow(),
            metadata={
                "lookback_days": 30,
                "realized_vol": 0.12,
                "signal_type": "trend_following"
            }
        )
        
        # 2. Donchian Breakout (Directional)
        breakout_value = self._calculate_breakout(prev_prices)
        signals["donchian_breakout"] = SignalResult(
            symbol=symbol,
            value=breakout_value,
            confidence=0.8 if abs(breakout_value) > 0.3 else 0.4,
            timestamp=datetime.utcnow(),
            metadata={
                "channel_period": 55,
                "realized_vol": 0.15,
                "atr_multiple": 2.0
            }
        )
        
        # 3. Short-Term Mean Reversion (Directional) 
        mean_revert_value = self._calculate_mean_reversion(prev_prices)
        signals["short_term_mean_reversion"] = SignalResult(
            symbol=symbol,
            value=mean_revert_value,
            confidence=0.6 + 0.3 * abs(mean_revert_value),
            timestamp=datetime.utcnow(),
            metadata={
                "zscore_threshold": 2.0,
                "realized_vol": 0.08,
                "reversion_speed": 0.3
            }
        )
        
        # 4. Perp Funding Carry (Market-Neutral)
        funding_rate = np.random.normal(0.01, 0.05)  # 1% avg, 5% vol annualized
        funding_value = -np.sign(funding_rate) * min(abs(funding_rate * 10), 0.8)
        signals["perp_funding_carry"] = SignalResult(
            symbol=symbol,
            value=funding_value,
            confidence=0.9 if abs(funding_rate) > 0.02 else 0.5,
            timestamp=datetime.utcnow(),
            metadata={
                "funding_rate": funding_rate,
                "market_neutral": True,
                "realized_vol": 0.06
            }
        )
        
        # 5. OI/Price Divergence (Directional)
        oi_divergence = np.random.normal(0, 0.3)
        price_momentum = (prev_prices.iloc[-1] / prev_prices.iloc[-5] - 1) if len(prev_prices) >= 5 else 0
        divergence_signal = -0.5 * oi_divergence if abs(price_momentum) > 0.02 else 0
        signals["oi_price_divergence"] = SignalResult(
            symbol=symbol,
            value=divergence_signal,
            confidence=0.7 if abs(oi_divergence) > 0.2 else 0.3,
            timestamp=datetime.utcnow(),
            metadata={
                "oi_change": oi_divergence,
                "price_momentum": price_momentum,
                "realized_vol": 0.10
            }
        )
        
        # 6. Alt/BTC Cross-Sectional (Market-Neutral)
        alt_momentum = np.random.normal(0, 0.4)
        btc_momentum = momentum_value
        relative_momentum = alt_momentum - btc_momentum
        signals["alt_btc_cross_sectional"] = SignalResult(
            symbol=symbol,
            value=relative_momentum * 0.6,  # Scale down
            confidence=0.8 if abs(relative_momentum) > 0.3 else 0.4,
            timestamp=datetime.utcnow(),
            metadata={
                "alt_momentum": alt_momentum,
                "btc_momentum": btc_momentum,
                "market_neutral": True,
                "realized_vol": 0.07
            }
        )
        
        # 7. Cash-and-Carry Basis (Market-Neutral)
        basis_rate = np.random.normal(0.08, 0.12)  # 8% avg basis with 12% vol
        carry_signal = np.sign(basis_rate) * min(abs(basis_rate * 5), 0.7)
        signals["cash_carry_basis"] = SignalResult(
            symbol=symbol,
            value=carry_signal,
            confidence=0.9 if abs(basis_rate) > 0.10 else 0.6,
            timestamp=datetime.utcnow(),
            metadata={
                "basis_rate": basis_rate,
                "futures_premium": basis_rate,
                "market_neutral": True,
                "realized_vol": 0.05
            }
        )
        
        # 8. Cross-Exchange Funding (Market-Neutral)
        funding_spread = np.random.normal(0, 0.03)
        spread_signal = np.sign(funding_spread) * min(abs(funding_spread * 15), 0.6)
        signals["cross_exchange_funding"] = SignalResult(
            symbol=symbol,
            value=spread_signal,
            confidence=0.8 if abs(funding_spread) > 0.02 else 0.4,
            timestamp=datetime.utcnow(),
            metadata={
                "funding_spread": funding_spread,
                "exchange_1": "binance",
                "exchange_2": "bybit",
                "market_neutral": True,
                "realized_vol": 0.04
            }
        )
        
        # 9. Options Vol-Risk Premium (Market-Neutral)
        implied_vol = np.random.normal(0.6, 0.2)  # 60% IV avg
        realized_vol = np.random.normal(0.4, 0.15)  # 40% RV avg
        vol_premium = implied_vol - realized_vol
        vol_signal = -np.sign(vol_premium) * min(abs(vol_premium * 3), 0.8)
        signals["options_vol_risk_premium"] = SignalResult(
            symbol=symbol,
            value=vol_signal,
            confidence=0.85 if abs(vol_premium) > 0.15 else 0.5,
            timestamp=datetime.utcnow(),
            metadata={
                "implied_vol": implied_vol,
                "realized_vol": realized_vol,
                "vol_premium": vol_premium,
                "market_neutral": True
            }
        )
        
        # 10. 25Δ Skew Whipsaw (Directional)
        skew = np.random.normal(0.1, 0.15)  # 10% avg skew
        skew_signal = -np.sign(skew) * min(abs(skew * 4), 0.6) if abs(skew) > 0.15 else 0
        signals["delta_skew_whipsaw"] = SignalResult(
            symbol=symbol,
            value=skew_signal,
            confidence=0.7 if abs(skew) > 0.20 else 0.3,
            timestamp=datetime.utcnow(),
            metadata={
                "skew_25d": skew,
                "skew_threshold": 0.15,
                "realized_vol": 0.09
            }
        )
        
        # 11. Stablecoin Supply Ratio (Overlay)
        ssr = np.random.normal(0.15, 0.05)  # 15% avg SSR
        ssr_zscore = (ssr - 0.15) / 0.05
        ssr_signal = np.clip(ssr_zscore * 0.3, -0.8, 0.8)
        signals["stablecoin_supply_ratio"] = SignalResult(
            symbol=symbol,
            value=ssr_signal,
            confidence=0.6 + 0.3 * abs(ssr_zscore),
            timestamp=datetime.utcnow(),
            metadata={
                "ssr": ssr,
                "ssr_zscore": ssr_zscore,
                "signal_type": "overlay",
                "realized_vol": 0.03
            }
        )
        
        # 12. MVRV Z-Score (Overlay)
        mvrv_zscore = np.random.normal(1.5, 2.0)  # Centered around 1.5
        mvrv_signal = 0
        if mvrv_zscore > 7:  # Euphoric
            mvrv_signal = -0.7
        elif mvrv_zscore < -1:  # Value territory
            mvrv_signal = 0.7
        elif abs(mvrv_zscore - 1.5) > 3:  # Moderate deviation
            mvrv_signal = -np.sign(mvrv_zscore - 1.5) * 0.3
            
        signals["mvrv_zscore"] = SignalResult(
            symbol=symbol,
            value=mvrv_signal,
            confidence=0.8 if abs(mvrv_zscore) > 3 else 0.4,
            timestamp=datetime.utcnow(),
            metadata={
                "mvrv_zscore": mvrv_zscore,
                "regime": "euphoric" if mvrv_zscore > 7 else "value" if mvrv_zscore < -1 else "neutral",
                "signal_type": "overlay",
                "realized_vol": 0.02
            }
        )
        
        return signals
    
    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate momentum signal value."""
        if len(prices) < 20:
            return 0
        
        short_ma = prices.rolling(5).mean().iloc[-1]
        long_ma = prices.rolling(20).mean().iloc[-1]
        momentum = (short_ma / long_ma - 1) * 5  # Scale to [-1, 1] range
        return np.clip(momentum, -0.9, 0.9)
    
    def _calculate_breakout(self, prices: pd.Series) -> float:
        """Calculate breakout signal value."""
        if len(prices) < 20:
            return 0
        
        highest = prices.rolling(20).max().iloc[-1]
        lowest = prices.rolling(20).min().iloc[-1]
        current = prices.iloc[-1]
        
        if current >= highest:
            return 0.8
        elif current <= lowest:
            return -0.8
        else:
            # Position within channel
            position = (current - lowest) / (highest - lowest) - 0.5
            return position * 0.4  # Scaled signal
    
    def _calculate_mean_reversion(self, prices: pd.Series) -> float:
        """Calculate mean reversion signal."""
        if len(prices) < 10:
            return 0
        
        mean_price = prices.rolling(10).mean().iloc[-1]
        std_price = prices.rolling(10).std().iloc[-1]
        current = prices.iloc[-1]
        
        if std_price == 0:
            return 0
        
        zscore = (current - mean_price) / std_price
        # Mean reversion: fade extreme moves
        if abs(zscore) > 2:
            return -np.sign(zscore) * 0.7
        elif abs(zscore) > 1:
            return -np.sign(zscore) * 0.3
        else:
            return 0


def run_blender_demo():
    """Run comprehensive demo of Portfolio Blender v2."""
    
    print("🚀 Portfolio Blender v2 Demo - 12 Signal Integration")
    print("=" * 60)
    
    # Initialize signal generator
    signal_gen = MockSignalGenerator(seed=42)
    
    # Test different allocation methods
    allocation_methods = [
        AllocationMethod.RISK_PARITY,
        AllocationMethod.CONFIDENCE_WEIGHTED,
        AllocationMethod.KELLY_OPTIMAL,
        AllocationMethod.PERFORMANCE_WEIGHTED
    ]
    
    results = {}
    
    for method in allocation_methods:
        print(f"\n📊 Testing Allocation Method: {method.value.upper()}")
        print("-" * 40)
        
        # Configure blender
        config = BlenderConfigV2(
            allocation_method=method,
            min_signal_confidence=0.3,
            risk_limits=RiskLimits(
                max_net_exposure=0.30,
                max_gross_leverage=2.5,
                max_single_position=0.10
            )
        )
        
        blender = PortfolioBlenderV2(config)
        method_results = []
        
        # Simulate 100 time steps
        for step in range(100):
            # Generate signals
            signals = signal_gen.generate_all_signals()
            
            # Blend signals
            result = blender.blend_signals(signals, "BTCUSDT", {
                "realized_vol": 0.12,
                "momentum": signals["time_series_momentum"].value
            })
            
            # Update performance (mock)
            for signal_name in signals.keys():
                performance = np.random.normal(0.01, 0.05)  # Mock daily returns
                blender.update_performance(signal_name, performance)
            
            method_results.append(result)
            signal_gen.advance_time()
            
            # Print progress every 20 steps
            if (step + 1) % 20 == 0:
                print(f"   Step {step + 1}/100: Net={result.final_position:.3f}, "
                      f"Gross={result.gross_exposure:.3f}, Conf={result.confidence:.3f}")
        
        results[method] = method_results
        
        # Print method summary
        final_result = method_results[-1]
        avg_net = np.mean([r.final_position for r in method_results])
        avg_gross = np.mean([r.gross_exposure for r in method_results])
        avg_conf = np.mean([r.confidence for r in method_results])
        
        print(f"\n   Summary for {method.value}:")
        print(f"   ├─ Average Net Exposure: {avg_net:.3f}")
        print(f"   ├─ Average Gross Leverage: {avg_gross:.3f}")
        print(f"   ├─ Average Confidence: {avg_conf:.3f}")
        print(f"   ├─ Active Signals: {final_result.risk_metrics['signals_active']}")
        print(f"   └─ Max Correlation: {final_result.risk_metrics['max_correlation']:.3f}")
        
        # Check risk violations
        violations = blender.check_risk_limits()
        if violations:
            print(f"   ⚠️  Risk Violations: {len(violations)}")
            for violation in violations:
                print(f"      - {violation['type']}: {violation['current']:.3f} > {violation['limit']:.3f}")
        else:
            print("   ✅ No Risk Violations")
    
    print(f"\n🎯 DETAILED ANALYSIS - Final Snapshot")
    print("=" * 60)
    
    # Analyze the best performing method (last one tested)
    best_method = AllocationMethod.RISK_PARITY
    final_result = results[best_method][-1]
    
    print(f"\nMethod: {best_method.value}")
    print(f"Final Position: {final_result.final_position:.4f}")
    print(f"Gross Exposure: {final_result.gross_exposure:.4f}")
    print(f"Confidence: {final_result.confidence:.4f}")
    
    print(f"\n📈 SIGNAL BREAKDOWN:")
    print(f"├─ Directional Position: {final_result.directional_position:.4f}")
    print(f"├─ Market-Neutral Position: {final_result.market_neutral_position:.4f}")
    print(f"└─ Overlay Adjustments: {final_result.overlay_adjustments}")
    
    print(f"\n⚖️  SIGNAL CONTRIBUTIONS:")
    sorted_contributions = sorted(
        final_result.signal_contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    for signal_name, contribution in sorted_contributions[:8]:  # Top 8
        signal_type = config.signal_classifications.get(signal_name, SignalType.DIRECTIONAL)
        print(f"├─ {signal_name:<25} {contribution:>8.4f} ({signal_type.value})")
    
    print(f"\n🎲 RISK METRICS:")
    risk_metrics = final_result.risk_metrics
    print(f"├─ Net Exposure: {risk_metrics['net_exposure']:.4f}")
    print(f"├─ Leverage Utilization: {risk_metrics['leverage_utilization']:.2%}")
    print(f"├─ Max Correlation: {risk_metrics['max_correlation']:.4f}")
    print(f"├─ Signal Concentration: {risk_metrics['signal_concentration']:.4f}")
    print(f"└─ Confidence-Weighted Exposure: {risk_metrics['confidence_weighted_exposure']:.4f}")
    
    print(f"\n📊 PORTFOLIO STATISTICS:")
    stats = blender.get_portfolio_statistics()
    summary = stats["summary"]
    print(f"├─ Snapshots Recorded: {summary['snapshots_count']}")
    print(f"├─ Signals Tracked: {summary['signals_tracked']}")
    print(f"├─ Avg Net Exposure: {summary['avg_net_exposure']:.4f}")
    print(f"├─ Avg Gross Leverage: {summary['avg_gross_leverage']:.4f}")
    print(f"└─ Max Leverage Used: {summary['max_leverage_used']:.4f}")
    
    print(f"\n🏆 SIGNAL TYPE ANALYSIS:")
    type_exposures = final_result.portfolio_snapshot.type_exposures
    for signal_type, exposure in type_exposures.items():
        config_limit = config.signal_type_configs[signal_type].max_allocation
        utilization = exposure / config_limit if config_limit > 0 else 0
        print(f"├─ {signal_type.value:<15}: {exposure:.4f} ({utilization:.1%} of limit)")
    
    print(f"\n🔬 CORRELATION ANALYSIS:")
    correlation_risks = final_result.portfolio_snapshot.correlation_risks
    if correlation_risks:
        print("   High correlation pairs:")
        for pair, risk in sorted(correlation_risks.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   ├─ {pair}: {risk:.4f}")
    else:
        print("   ✅ No significant correlation risks detected")
    
    print(f"\n📏 PERFORMANCE COMPARISON:")
    print("   Method Comparison (Final Position):")
    for method, method_results in results.items():
        final_pos = method_results[-1].final_position
        avg_conf = np.mean([r.confidence for r in method_results])
        print(f"   ├─ {method.value:<20}: {final_pos:>8.4f} (conf: {avg_conf:.3f})")
    
    print(f"\n🎉 Demo Complete!")
    print("=" * 60)
    
    return results


def test_signal_conflicts():
    """Test signal conflict resolution scenarios."""
    print(f"\n🥊 SIGNAL CONFLICT RESOLUTION TEST")
    print("=" * 40)
    
    config = BlenderConfigV2(allocation_method=AllocationMethod.RISK_PARITY)
    blender = PortfolioBlenderV2(config)
    
    # Create conflicting signals
    conflicting_signals = {
        "strong_long_momentum": SignalResult(
            symbol="BTCUSDT",
            value=0.8, confidence=0.9, timestamp=datetime.utcnow(),
            metadata={"signal_type": "momentum"}
        ),
        "strong_short_breakout": SignalResult(
            symbol="BTCUSDT",
            value=-0.7, confidence=0.85, timestamp=datetime.utcnow(),
            metadata={"signal_type": "breakout"}
        ),
        "weak_long_mean_revert": SignalResult(
            symbol="BTCUSDT",
            value=0.3, confidence=0.4, timestamp=datetime.utcnow(),
            metadata={"signal_type": "mean_reversion"}
        ),
        "neutral_funding": SignalResult(
            symbol="BTCUSDT",
            value=0.1, confidence=0.7, timestamp=datetime.utcnow(),
            metadata={"market_neutral": True}
        )
    }
    
    result = blender.blend_signals(conflicting_signals, "BTCUSDT")
    
    print(f"Conflict Resolution Result:")
    print(f"├─ Final Position: {result.final_position:.4f}")
    print(f"├─ Confidence: {result.confidence:.4f}")
    print(f"├─ Directional Component: {result.directional_position:.4f}")
    print(f"└─ Market-Neutral Component: {result.market_neutral_position:.4f}")
    
    print(f"\nSignal Contributions:")
    for signal_name, contribution in result.signal_contributions.items():
        original_value = conflicting_signals[signal_name].value
        print(f"├─ {signal_name}: {original_value:.3f} → {contribution:.4f}")


if __name__ == "__main__":
    try:
        # Run main demo
        results = run_blender_demo()
        
        # Test conflict resolution
        test_signal_conflicts()
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
