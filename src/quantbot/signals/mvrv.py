"""
MVRV Z-Score Signal Implementation

MVRV (Market Value to Realized Value) Z-Score signal for regime detection
and local top/bottom prediction in cryptocurrency markets.

Key Features:
- Value zone detection (<-1 Z-score) → contrarian long bias
- Euphoric zone detection (>7 Z-score) → contrarian short bias
- Weekly signal frequency to avoid whipsaws
- Regime filter for other signals
- Historical calibration back to 2013 patterns

Signal Interpretation:
- MVRV Z <-1: Value zone → contrarian long positioning
- MVRV Z >7: Euphoric zone → contrarian short positioning
- Neutral range: Normal market conditions
- Extreme readings → turn off/reverse directional strategies
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from .base import SignalBase, SignalResult, SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class MVRVConfig(SignalConfig):
    """Configuration for MVRV Z-Score signal."""

    # Z-Score thresholds
    value_threshold: float = -1.0  # Value zone threshold
    euphoric_threshold: float = 7.0  # Euphoric zone threshold

    # Historical parameters
    lookback_periods: int = 365 * 4  # 4 years for Z-score calculation
    signal_frequency_days: int = 7  # Weekly rebalancing

    # MVRV calculation parameters
    realized_cap_multiplier: float = 0.7  # Mock realized cap adjustment
    volatility_window: int = 30  # Days for volatility normalization

    # Regime detection
    regime_filter_enabled: bool = True
    extreme_zone_damping: float = 0.5  # Reduce signal strength in extremes


class MVRVSignal(SignalBase):
    """
    MVRV Z-Score Signal for regime detection and local top/bottom prediction.

    The MVRV ratio compares current market capitalization to realized
    capitalization, providing insight into market valuation extremes.

    Z-Score normalization enables consistent threshold-based regime detection
    across different market cycles and volatility environments.
    """

    def __init__(self, config: MVRVConfig):
        super().__init__(config)
        self.config: MVRVConfig = config
        self._last_signal_date: Optional[datetime] = None
        self._mvrv_history: pd.DataFrame = pd.DataFrame()

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """
        Generate MVRV Z-Score signal.

        Args:
            data: OHLCV dataframe with datetime index
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            SignalResult with regime classification and directional bias
        """
        if not self.validate_data(data, min_periods=self.config.lookback_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=data.index[-1],
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data for MVRV calculation"},
            )

        # Check weekly frequency constraint
        current_date = data.index[-1]
        if self._should_skip_signal(current_date):
            return self._get_previous_signal(symbol, current_date)

        # Calculate MVRV ratio and Z-score
        mvrv_ratio = self._calculate_mvrv_ratio(data)
        mvrv_zscore = self._calculate_zscore(mvrv_ratio, data)

        # Generate regime-aware signal
        signal_value, confidence, regime = self._generate_regime_signal(
            mvrv_zscore, data
        )

        # Update signal history
        self._last_signal_date = current_date
        self._update_history(current_date, mvrv_ratio, mvrv_zscore, regime)

        metadata = {
            "mvrv_ratio": float(mvrv_ratio.iloc[-1]),
            "mvrv_zscore": float(mvrv_zscore.iloc[-1]),
            "regime": regime,
            "is_value_zone": mvrv_zscore.iloc[-1] < self.config.value_threshold,
            "is_euphoric_zone": mvrv_zscore.iloc[-1] > self.config.euphoric_threshold,
            "weekly_frequency": True,
            "signal_type": "regime_filter",
        }

        logger.info(
            f"MVRV Signal - {symbol}: Z-Score={mvrv_zscore.iloc[-1]:.2f}, "
            f"Regime={regime}, Signal={signal_value:.2f}"
        )

        return SignalResult(
            symbol=symbol,
            timestamp=current_date,
            value=signal_value,
            confidence=confidence,
            metadata=metadata,
        )

    def _calculate_mvrv_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MVRV ratio using market cap and realized cap estimates.

        Since we don't have real on-chain data, we estimate:
        - Market Cap = Price * Circulating Supply (estimated)
        - Realized Cap = Market Cap * realized_cap_multiplier

        Args:
            data: OHLCV dataframe

        Returns:
            MVRV ratio time series
        """
        # Estimate circulating supply growth (simplified model)
        days_since_start = (data.index - data.index[0]).days
        estimated_supply = 19_000_000 + (
            days_since_start * 900 / 365
        )  # BTC supply model

        # Market capitalization
        market_cap = data["close"] * estimated_supply

        # Realized cap estimate (simplified as percentage of market cap)
        # In reality, this would be the sum of each UTXO's value at time of last movement
        price_ma = data["close"].rolling(window=365, min_periods=30).mean()
        # Fill initial NaN values with early price data
        price_ma = price_ma.fillna(data["close"].expanding().mean())
        realized_cap = price_ma * estimated_supply * self.config.realized_cap_multiplier

        # MVRV ratio
        mvrv_ratio = market_cap / realized_cap

        return mvrv_ratio.ffill()

    def _calculate_zscore(self, mvrv_ratio: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Z-score of MVRV ratio for normalization.

        Args:
            mvrv_ratio: MVRV ratio time series
            data: OHLCV dataframe for alignment

        Returns:
            MVRV Z-score time series
        """
        # Rolling statistics for Z-score calculation
        window = min(self.config.lookback_periods, len(mvrv_ratio))

        rolling_mean = mvrv_ratio.rolling(window=window, min_periods=window // 4).mean()

        rolling_std = mvrv_ratio.rolling(window=window, min_periods=window // 4).std()

        # Z-score calculation
        mvrv_zscore = (mvrv_ratio - rolling_mean) / rolling_std

        return mvrv_zscore.fillna(0)

    def _generate_regime_signal(
        self, mvrv_zscore: pd.Series, data: pd.DataFrame
    ) -> tuple[float, float, str]:
        """
        Generate regime-aware signal based on MVRV Z-score.

        Args:
            mvrv_zscore: MVRV Z-score time series
            data: OHLCV dataframe

        Returns:
            Tuple of (signal_value, confidence, regime)
        """
        current_zscore = mvrv_zscore.iloc[-1]

        # Determine regime
        if current_zscore < self.config.value_threshold:
            regime = "value_zone"
            # Contrarian long bias in value zone
            signal_strength = min(1.0, abs(current_zscore) / 2.0)
            signal_value = signal_strength
            confidence = min(0.9, abs(current_zscore) / 3.0)

        elif current_zscore > self.config.euphoric_threshold:
            regime = "euphoric_zone"
            # Contrarian short bias in euphoric zone
            signal_strength = min(
                1.0, (current_zscore - self.config.euphoric_threshold) / 3.0
            )
            signal_value = -signal_strength
            confidence = min(0.9, signal_strength)

        else:
            regime = "neutral"
            # Neutral signal in normal range
            signal_value = 0.0
            confidence = 0.3

        # Apply extreme zone damping if enabled
        if self.config.regime_filter_enabled and regime != "neutral":
            signal_value *= self.config.extreme_zone_damping

        return signal_value, confidence, regime

    def _should_skip_signal(self, current_date: datetime) -> bool:
        """Check if signal should be skipped due to frequency constraint."""
        if self._last_signal_date is None:
            return False

        days_since_last = (current_date - self._last_signal_date).days
        return days_since_last < self.config.signal_frequency_days

    def _get_previous_signal(self, symbol: str, current_date: datetime) -> SignalResult:
        """Return previous signal when skipping due to frequency constraint."""
        if self._mvrv_history.empty:
            return SignalResult(
                symbol=symbol,
                timestamp=current_date,
                value=0.0,
                confidence=0.0,
                metadata={"status": "no_previous_signal"},
            )

        last_row = self._mvrv_history.iloc[-1]

        return SignalResult(
            symbol=symbol,
            timestamp=current_date,
            value=0.0,  # No new signal
            confidence=0.1,
            metadata={
                "status": "frequency_skip",
                "last_mvrv_zscore": float(last_row["mvrv_zscore"]),
                "last_regime": last_row["regime"],
                "days_until_next": self.config.signal_frequency_days
                - (current_date - (self._last_signal_date or current_date)).days,
            },
        )

    def _update_history(
        self,
        timestamp: datetime,
        mvrv_ratio: pd.Series,
        mvrv_zscore: pd.Series,
        regime: str,
    ) -> None:
        """Update internal signal history."""
        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "mvrv_ratio": [mvrv_ratio.iloc[-1]],
                "mvrv_zscore": [mvrv_zscore.iloc[-1]],
                "regime": [regime],
            }
        )

        self._mvrv_history = pd.concat([self._mvrv_history, new_row], ignore_index=True)

        # Keep only recent history to manage memory
        max_history = 1000
        if len(self._mvrv_history) > max_history:
            self._mvrv_history = self._mvrv_history.tail(max_history)

    def get_regime_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current regime status for use by other signals.

        Returns:
            Dictionary with regime information or None if no signal generated
        """
        if self._mvrv_history.empty:
            return None

        last_row = self._mvrv_history.iloc[-1]

        return {
            "regime": last_row["regime"],
            "mvrv_zscore": last_row["mvrv_zscore"],
            "is_extreme": last_row["regime"] != "neutral",
            "should_dampen_directional": last_row["regime"]
            in ["value_zone", "euphoric_zone"],
            "contrarian_bias": (
                "long"
                if last_row["regime"] == "value_zone"
                else "short" if last_row["regime"] == "euphoric_zone" else "neutral"
            ),
        }

    def get_historical_extremes(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Get historical extreme readings for analysis.

        Args:
            lookback_days: Days to look back for extremes

        Returns:
            Dictionary with extreme value statistics
        """
        if self._mvrv_history.empty:
            return {}

        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = self._mvrv_history[
            self._mvrv_history["timestamp"] > cutoff_date
        ]

        if recent_history.empty:
            return {}

        return {
            "min_zscore": float(recent_history["mvrv_zscore"].min()),
            "max_zscore": float(recent_history["mvrv_zscore"].max()),
            "value_zone_count": int((recent_history["regime"] == "value_zone").sum()),
            "euphoric_zone_count": int(
                (recent_history["regime"] == "euphoric_zone").sum()
            ),
            "neutral_count": int((recent_history["regime"] == "neutral").sum()),
            "avg_zscore": float(recent_history["mvrv_zscore"].mean()),
            "zscore_volatility": float(recent_history["mvrv_zscore"].std()),
        }
