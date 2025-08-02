import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import itertools

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class MomentumConfig(SignalConfig):
    """Configuration for momentum signal."""

    lookback_days: int = 90  # 3-month momentum (30-90 days typical)
    skip_recent_days: int = 7  # Skip last week to de-noise
    ma_window: int = 200  # Moving average filter
    min_periods: int = 100
    volatility_target: float = 0.15  # 15% annualized volatility target
    volatility_lookback: int = 30  # Days for volatility estimation
    enable_vol_targeting: bool = True  # Enable volatility targeting


class TimeSeriesMomentumSignal(SignalBase):
    """
    Time-Series Momentum (3-12M) - Signal #1 from your list.

    Works clean on BTC & majors; ignores last week to de-noise news spikes.
    Uses moving average trend filter for additional confirmation.
    """

    def __init__(self, config: MomentumConfig):
        super().__init__(config)
        self.config: MomentumConfig = config

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate momentum signal from price data."""

        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # Calculate momentum return (skip recent days to de-noise)
            end_idx = len(data) - self.config.skip_recent_days
            start_idx = end_idx - self.config.lookback_days

            if start_idx < 0:
                start_idx = 0

            if end_idx <= start_idx:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Insufficient lookback period"},
                )

            # Calculate momentum return
            price_start = data.iloc[start_idx]["close"]
            price_end = data.iloc[end_idx]["close"]
            momentum_return = (price_end - price_start) / price_start

            # Moving average trend filter
            ma_values = data["close"].rolling(window=self.config.ma_window).mean()
            current_price = data.iloc[-1]["close"]
            current_ma = (
                ma_values.iloc[-1] if not pd.isna(ma_values.iloc[-1]) else current_price
            )

            # MA slope (trend strength)
            ma_slope = 0.0
            if len(ma_values) >= 20:
                recent_ma = ma_values.iloc[-20:].values
                if not np.any(pd.isna(recent_ma)):
                    ma_slope = (
                        np.polyfit(range(20), recent_ma.astype(float), 1)[0]
                        / current_ma
                    )

            # Price above/below MA
            price_vs_ma = (current_price - current_ma) / current_ma

            # Combine momentum with trend filter
            base_signal = np.tanh(momentum_return * 2)  # Scale and bound to [-1,1]

            # Apply trend filter: strengthen signal if price above rising MA
            if current_price > current_ma and ma_slope > 0:
                trend_boost = min(0.5, abs(price_vs_ma) + ma_slope * 100)
                signal_value = base_signal * (1 + trend_boost)
            elif current_price < current_ma and ma_slope < 0:
                trend_boost = min(0.5, abs(price_vs_ma) + abs(ma_slope) * 100)
                signal_value = base_signal * (1 + trend_boost)
            else:
                # Counter-trend: reduce signal strength
                signal_value = base_signal * 0.5

            # Apply volatility targeting if enabled
            vol_adjustment = 1.0
            realized_vol = None
            if self.config.enable_vol_targeting:
                vol_adjustment, realized_vol = self._calculate_volatility_adjustment(
                    data
                )
                signal_value *= vol_adjustment

            # Bound final signal
            signal_value = max(-1.0, min(1.0, signal_value))

            # Calculate confidence based on consistency and strength
            momentum_strength = abs(momentum_return)
            trend_consistency = (
                1.0
                if (base_signal > 0 and price_vs_ma > 0)
                or (base_signal < 0 and price_vs_ma < 0)
                else 0.5
            )

            confidence = min(1.0, momentum_strength * 10 * trend_consistency)

            # Metadata for debugging and analysis
            metadata = {
                "momentum_return": momentum_return,
                "ma_slope": ma_slope,
                "price_vs_ma": price_vs_ma,
                "lookback_days": self.config.lookback_days,
                "skip_days": self.config.skip_recent_days,
                "trend_boost": trend_boost if "trend_boost" in locals() else 0.0,
                "vol_adjustment": vol_adjustment,
                "realized_vol": realized_vol,
            }

            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=signal_value,
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _calculate_volatility_adjustment(
        self, data: pd.DataFrame
    ) -> Tuple[float, Optional[float]]:
        """Calculate volatility adjustment factor for position sizing."""
        try:
            # Calculate returns for volatility estimation
            returns = data["close"].pct_change().dropna()

            # Use lookback window for volatility calculation
            vol_window = min(
                len(returns), self.config.volatility_lookback * 24
            )  # Daily to hourly
            recent_returns = returns.tail(vol_window)

            if len(recent_returns) < 10:  # Need minimum data
                return 1.0, None

            # Calculate annualized volatility (assuming hourly data)
            realized_vol = float(recent_returns.std() * np.sqrt(24 * 365))  # Annualize

            if realized_vol <= 0:
                return 1.0, realized_vol

            # Volatility adjustment: target_vol / realized_vol
            vol_adjustment = self.config.volatility_target / realized_vol

            # Cap adjustment to reasonable bounds
            vol_adjustment = max(0.25, min(4.0, vol_adjustment))

            return vol_adjustment, realized_vol

        except Exception:
            return 1.0, None

    @classmethod
    def grid_search_parameters(cls, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Perform grid search to find optimal parameters for momentum signal.

        Returns best parameters based on simple performance metrics.
        """
        # Parameter ranges for grid search
        lookback_ranges = [30, 60, 90]  # 1-3 months as per research specs
        skip_days_ranges = [5, 7, 10]  # Skip last 5-10 days
        ma_window_ranges = [100, 200, 300]  # Various MA periods

        best_params = None
        best_score = -np.inf
        results = []

        for lookback, skip_days, ma_window in itertools.product(
            lookback_ranges, skip_days_ranges, ma_window_ranges
        ):
            try:
                # Create config with current parameters
                config = MomentumConfig(
                    lookback_days=lookback,
                    skip_recent_days=skip_days,
                    ma_window=ma_window,
                    enable_vol_targeting=True,
                )

                signal = cls(config)

                # Test on multiple time windows
                score = cls._backtest_parameters(signal, data, symbol)

                result = {
                    "lookback_days": lookback,
                    "skip_recent_days": skip_days,
                    "ma_window": ma_window,
                    "score": score,
                }
                results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = result

            except Exception:
                continue

        return {
            "best_params": best_params,
            "all_results": results,
            "best_score": best_score,
        }

    @staticmethod
    def _backtest_parameters(
        signal: "TimeSeriesMomentumSignal", data: pd.DataFrame, symbol: str
    ) -> float:
        """Simple backtesting for parameter optimization."""
        try:
            if len(data) < 200:  # Need sufficient data
                return -np.inf

            # Test on last portion of data
            test_periods = min(50, len(data) // 4)
            returns = []

            for i in range(len(data) - test_periods, len(data), 5):  # Every 5 periods
                window_data = data.iloc[:i]
                if len(window_data) < signal.config.min_periods:
                    continue

                # Get signal (synchronous version for grid search)
                result = signal._generate_sync(window_data, symbol)
                if result.confidence < 0.5:  # Only high confidence signals
                    continue

                # Calculate forward return (simplified)
                if i + 5 < len(data):
                    current_price = window_data.iloc[-1]["close"]
                    future_price = data.iloc[i + 5]["close"]
                    forward_return = (future_price - current_price) / current_price

                    # Align signal with return
                    signal_aligned_return = result.value * forward_return
                    returns.append(signal_aligned_return)

            if len(returns) < 5:  # Need minimum signals
                return -np.inf

            # Simple scoring: Sharpe-like ratio
            avg_return = np.mean(returns)
            volatility = np.std(returns) if len(returns) > 1 else 1.0

            if volatility == 0:
                return 0.0

            return float(avg_return / volatility)

        except Exception:
            return -np.inf

    def _generate_sync(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Synchronous version of generate for grid search."""
        # Copy the main logic without async
        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # Calculate momentum return (skip recent days to de-noise)
            end_idx = len(data) - self.config.skip_recent_days
            start_idx = end_idx - self.config.lookback_days

            if start_idx < 0:
                start_idx = 0

            if end_idx <= start_idx:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Insufficient lookback period"},
                )

            # Calculate momentum return
            price_start = data.iloc[start_idx]["close"]
            price_end = data.iloc[end_idx]["close"]
            momentum_return = (price_end - price_start) / price_start

            # Simple signal calculation for grid search
            base_signal = np.tanh(momentum_return * 2)

            # Apply volatility targeting if enabled
            vol_adjustment = 1.0
            if self.config.enable_vol_targeting:
                vol_adjustment, _ = self._calculate_volatility_adjustment(data)
                base_signal *= vol_adjustment

            signal_value = max(-1.0, min(1.0, base_signal))

            # Simple confidence calculation
            confidence = min(1.0, abs(momentum_return) * 10)

            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=signal_value,
                confidence=confidence,
                metadata={"momentum_return": momentum_return},
            )

        except Exception as e:
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )


class CrossSectionalMomentumSignal(SignalBase):
    """
    Alt/BTC Cross-Sectional Momentum - Signal #2 from your list.

    Ranks alts by 6M performance vs BTC; long top decile, short bottom.
    Requires multiple assets to rank against each other.
    """

    def __init__(self, config: SignalConfig):
        super().__init__(config)
        self.lookback_days = 180  # 6 months
        self.ranking_cache: Dict[str, float] = {}
        self.last_ranking_time: datetime = datetime.min

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate cross-sectional momentum signal."""

        # For now, return neutral - requires multi-asset ranking system
        # TODO: Implement when portfolio manager supports multiple assets

        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={
                "status": "not_implemented",
                "reason": "requires_multi_asset_ranking",
            },
        )
