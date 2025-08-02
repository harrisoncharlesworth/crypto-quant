import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class SSRConfig(SignalConfig):
    """Configuration for Stablecoin Supply Ratio signal."""

    zscore_window: int = 252  # 1-year rolling window for Z-score (weekly data)
    rebalance_frequency: str = "weekly"  # Weekly signal generation
    long_boost_threshold: float = -1.0  # Z-score threshold for long boost
    short_reduce_threshold: float = 1.0  # Z-score threshold for short reduction
    allocation_adjustment: float = 0.25  # 25% allocation adjustment
    min_periods: int = 52  # Minimum 1 year of weekly data
    signal_type: str = "overlay"  # Overlay filter vs standalone strategy
    confidence_scaling: float = 2.0  # Z-score to confidence conversion factor


class StablecoinSupplyRatioSignal(SignalBase):
    """
    Stablecoin Supply Ratio (SSR) Signal - Market-Neutral Overlay Strategy.

    Calculates SSR = Stablecoin Market Cap / Total Crypto Market Cap
    Uses Z-score normalization to detect accumulation phases via dry powder metrics.

    Signal interpretation:
    - SSR <-1σ: Boost long allocation (high dry powder = bullish medium-term)
    - SSR >+1σ: Reduce long allocation (low dry powder = bearish medium-term)
    - Neutral range: No allocation adjustment

    Risk management:
    - Weekly rebalancing reduces overtrading
    - Z-score bounds prevent extreme allocations
    - Overlay nature limits direct market exposure
    - Historical calibration for regime stability
    """

    def __init__(self, config: SSRConfig):
        super().__init__(config)
        self.config: SSRConfig = config
        self._cached_zscore_data: Optional[pd.Series] = None
        self._last_zscore_update: Optional[datetime] = None

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate SSR overlay signal from market data."""

        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data for SSR calculation"},
            )

        try:
            # Get mock on-chain data (in production, would fetch from Glassnode/CryptoQuant)
            ssr_data = self._get_ssr_data(data)

            if len(ssr_data) < self.config.min_periods:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Insufficient SSR historical data"},
                )

            # Calculate Z-score normalized SSR
            current_ssr = ssr_data.iloc[-1]
            ssr_zscore = self._calculate_zscore(ssr_data, current_ssr)

            # Generate overlay signal based on Z-score thresholds
            signal_value, signal_type = self._generate_overlay_signal(ssr_zscore)

            # Calculate confidence based on Z-score extremity
            confidence = self._calculate_confidence(ssr_zscore)

            # Check for weekly rebalancing timing
            is_rebalance_time = self._is_rebalance_time(data.index[-1])

            # Apply frequency filter - only generate signals on rebalance schedule
            if not is_rebalance_time:
                signal_value *= 0.1  # Reduce signal strength off-schedule
                confidence *= 0.5

            # Metadata for debugging and analysis
            metadata = {
                "ssr_current": float(current_ssr),
                "ssr_zscore": float(ssr_zscore),
                "signal_type": signal_type,
                "is_rebalance_time": is_rebalance_time,
                "allocation_adjustment": self._get_allocation_adjustment(ssr_zscore),
                "dry_powder_level": self._interpret_dry_powder(ssr_zscore),
                "lookback_window": self.config.zscore_window,
                "regime_detection": self._detect_regime(ssr_data),
                "ssr_trend": self._calculate_ssr_trend(ssr_data),
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
                metadata={"error": f"SSR calculation failed: {str(e)}"},
            )

    def _get_ssr_data(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Generate mock SSR data structure.
        In production, would fetch from Glassnode/CryptoQuant APIs.
        """
        # Create weekly timestamps aligned with market data
        weekly_dates = pd.date_range(
            start=market_data.index[0], end=market_data.index[-1], freq="W"
        )

        # Mock SSR calculation with realistic dynamics
        # SSR typically ranges from 0.05 (5%) to 0.25 (25%)
        # Higher SSR = more stablecoins relative to crypto market cap

        np.random.seed(42)  # Consistent mock data

        # Base SSR level with trend and cyclical components
        base_ssr = 0.12  # 12% baseline SSR

        # Create realistic SSR time series
        ssr_values = []
        for i, date in enumerate(weekly_dates):
            # Trend component (gradual increase over time)
            trend = 0.02 * (i / len(weekly_dates))

            # Cyclical component (market cycles)
            cycle = 0.03 * np.sin(2 * np.pi * i / 52)  # Annual cycle

            # Market stress component (based on volatility)
            if i < len(market_data):
                # Higher volatility = higher SSR (flight to stablecoins)
                market_vol = (
                    market_data["close"]
                    .pct_change()
                    .rolling(7)
                    .std()
                    .iloc[min(i * 7, len(market_data) - 1)]
                )
                stress = (
                    0.02 * min(market_vol * 50, 1.0) if not pd.isna(market_vol) else 0
                )
            else:
                stress = 0

            # Random noise
            noise = 0.01 * np.random.normal(0, 1)

            ssr = base_ssr + trend + cycle + stress + noise
            ssr = max(0.03, min(0.30, ssr))  # Bound SSR to realistic range
            ssr_values.append(ssr)

        return pd.Series(ssr_values, index=weekly_dates, name="ssr")

    def _calculate_zscore(self, ssr_data: pd.Series, current_value: float) -> float:
        """Calculate Z-score for current SSR value using rolling window."""
        try:
            # Use rolling window for Z-score calculation
            window_size = min(len(ssr_data), self.config.zscore_window)
            recent_data = ssr_data.tail(window_size)

            mean_ssr = recent_data.mean()
            std_ssr = recent_data.std()

            if std_ssr == 0 or pd.isna(std_ssr):
                return 0.0

            zscore = (current_value - mean_ssr) / std_ssr
            return float(zscore)

        except Exception:
            return 0.0

    def _generate_overlay_signal(self, zscore: float) -> tuple[float, str]:
        """Generate overlay signal based on Z-score thresholds."""

        if zscore <= self.config.long_boost_threshold:
            # Low SSR = High dry powder = Boost long allocation
            signal_strength = min(1.0, abs(zscore) / 2.0)  # Scale by Z-score magnitude
            signal_value = signal_strength * self.config.allocation_adjustment
            signal_type = "long_boost"

        elif zscore >= self.config.short_reduce_threshold:
            # High SSR = Low dry powder = Reduce long allocation
            signal_strength = min(1.0, abs(zscore) / 2.0)
            signal_value = -signal_strength * self.config.allocation_adjustment
            signal_type = "long_reduce"

        else:
            # Neutral range - no allocation adjustment
            signal_value = 0.0
            signal_type = "neutral"

        return signal_value, signal_type

    def _calculate_confidence(self, zscore: float) -> float:
        """Calculate confidence based on Z-score extremity."""
        # Higher absolute Z-score = higher confidence
        # Scale to [0, 1] range using tanh function
        raw_confidence = abs(zscore) / self.config.confidence_scaling
        confidence = float(np.tanh(raw_confidence))

        return min(1.0, max(0.0, confidence))

    def _is_rebalance_time(self, current_time: pd.Timestamp) -> bool:
        """Check if current time aligns with rebalancing frequency."""
        if self.config.rebalance_frequency == "weekly":
            # Rebalance on Sundays (weekday 6)
            return current_time.weekday() == 6
        elif self.config.rebalance_frequency == "daily":
            return True
        else:
            # Default to weekly
            return current_time.weekday() == 6

    def _get_allocation_adjustment(self, zscore: float) -> Dict[str, Any]:
        """Calculate specific allocation adjustments for portfolio manager."""

        if zscore <= self.config.long_boost_threshold:
            return {
                "long_boost": self.config.allocation_adjustment,
                "short_reduce": 0.0,
                "action": "increase_long_exposure",
            }
        elif zscore >= self.config.short_reduce_threshold:
            return {
                "long_boost": 0.0,
                "short_reduce": self.config.allocation_adjustment,
                "action": "decrease_long_exposure",
            }
        else:
            return {
                "long_boost": 0.0,
                "short_reduce": 0.0,
                "action": "maintain_exposure",
            }

    def _interpret_dry_powder(self, zscore: float) -> str:
        """Interpret dry powder levels from SSR Z-score."""
        if zscore <= -2.0:
            return "very_high_dry_powder"
        elif zscore <= -1.0:
            return "high_dry_powder"
        elif zscore <= 0.0:
            return "above_average_dry_powder"
        elif zscore <= 1.0:
            return "below_average_dry_powder"
        elif zscore <= 2.0:
            return "low_dry_powder"
        else:
            return "very_low_dry_powder"

    def _detect_regime(self, ssr_data: pd.Series) -> str:
        """Detect market regime based on SSR trends."""
        if len(ssr_data) < 12:  # Need at least 3 months of weekly data
            return "insufficient_data"

        # Calculate trend over different timeframes
        short_term_trend = self._calculate_trend(ssr_data.tail(4))  # 1 month
        medium_term_trend = self._calculate_trend(ssr_data.tail(12))  # 3 months

        if short_term_trend > 0.005 and medium_term_trend > 0.002:
            return "accumulation_regime"  # Rising SSR = accumulating stablecoins
        elif short_term_trend < -0.005 and medium_term_trend < -0.002:
            return "deployment_regime"  # Falling SSR = deploying dry powder
        elif abs(short_term_trend) < 0.002 and abs(medium_term_trend) < 0.001:
            return "stable_regime"  # Stable SSR
        else:
            return "transitional_regime"  # Mixed signals

    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate trend slope for SSR data."""
        if len(data) < 2:
            return 0.0

        try:
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data.values.astype(float), 1)
            return float(slope)
        except Exception:
            return 0.0

    def _calculate_ssr_trend(self, ssr_data: pd.Series) -> Dict[str, float]:
        """Calculate SSR trends over multiple timeframes."""
        trends = {}

        timeframes = {
            "1m": 4,  # 4 weeks
            "3m": 12,  # 12 weeks
            "6m": 26,  # 26 weeks
            "1y": 52,  # 52 weeks
        }

        for label, periods in timeframes.items():
            if len(ssr_data) >= periods:
                trend_data = ssr_data.tail(periods)
                trends[f"trend_{label}"] = self._calculate_trend(trend_data)
            else:
                trends[f"trend_{label}"] = 0.0

        return trends

    @classmethod
    def create_default_config(cls) -> SSRConfig:
        """Create default SSR configuration."""
        return SSRConfig(
            enabled=True,
            weight=1.0,
            min_confidence=0.3,  # Lower threshold for overlay signals
            zscore_window=252,
            rebalance_frequency="weekly",
            long_boost_threshold=-1.0,
            short_reduce_threshold=1.0,
            allocation_adjustment=0.25,
            signal_type="overlay",
            confidence_scaling=2.0,
        )

    def get_allocation_overlay(self, signal_result: SignalResult) -> Dict[str, Any]:
        """
        Extract allocation overlay instructions for portfolio manager.

        This method provides the interface for portfolio allocation adjustments
        based on SSR signal state.
        """
        if not signal_result.metadata:
            return {"action": "no_adjustment", "multiplier": 1.0}

        allocation_adj = signal_result.metadata.get("allocation_adjustment", {})
        action = allocation_adj.get("action", "maintain_exposure")

        if action == "increase_long_exposure":
            multiplier = 1.0 + allocation_adj.get("long_boost", 0.0)
        elif action == "decrease_long_exposure":
            multiplier = 1.0 - allocation_adj.get("short_reduce", 0.0)
        else:
            multiplier = 1.0

        return {
            "action": action,
            "multiplier": multiplier,
            "confidence": signal_result.confidence,
            "ssr_zscore": signal_result.metadata.get("ssr_zscore", 0.0),
            "dry_powder_level": signal_result.metadata.get(
                "dry_powder_level", "unknown"
            ),
            "regime": signal_result.metadata.get("regime_detection", "unknown"),
        }
