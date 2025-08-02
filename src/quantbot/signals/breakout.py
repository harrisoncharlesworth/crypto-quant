import pandas as pd
from datetime import datetime
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class BreakoutConfig(SignalConfig):
    """Configuration for Donchian breakout signal."""

    channel_period: int = 55  # 55-day high/low channels
    atr_period: int = 14  # ATR calculation period
    atr_multiplier: float = 2.0  # ATR multiplier for stops
    min_periods: int = 100  # Minimum periods required
    choppy_threshold: float = 0.3  # ADX threshold for choppy market detection
    adx_period: int = 14  # ADX period for trend strength


class DonchianBreakoutSignal(SignalBase):
    """
    55-Day Donchian Breakout + ATR Stop Signal.

    Generates long signals on breakouts above 55-day high and short signals
    on breakdowns below 55-day low. Uses ATR-based position sizing and stops.
    Includes choppy market detection to reduce false signals.
    """

    def __init__(self, config: BreakoutConfig):
        super().__init__(config)
        self.config: BreakoutConfig = config

    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_adx(self, data: pd.DataFrame, period: int) -> float:
        """Calculate ADX for trend strength (simplified version)."""
        high = data["high"]
        low = data["low"]

        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True range
        atr = self.calculate_atr(data, period)

        # Directional indicators
        plus_di = (plus_dm.rolling(window=period).mean() / atr * 100).fillna(0)
        minus_di = (minus_dm.rolling(window=period).mean() / atr * 100).fillna(0)

        # ADX calculation
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0

    def detect_choppy_market(self, data: pd.DataFrame) -> bool:
        """Detect if market is in choppy/sideways condition."""
        try:
            adx = self.calculate_adx(data, self.config.adx_period)
            return (
                adx < self.config.choppy_threshold * 100
            )  # Convert threshold to ADX scale
        except Exception:
            return False

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate Donchian breakout signal from price data."""

        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # Calculate Donchian channels
            high_channel = data["high"].rolling(window=self.config.channel_period).max()
            low_channel = data["low"].rolling(window=self.config.channel_period).min()

            # Current price and channel levels
            current_price = data.iloc[-1]["close"]
            current_high_channel = high_channel.iloc[-1]
            current_low_channel = low_channel.iloc[-1]

            # Previous channel levels for breakout detection
            prev_high_channel = (
                high_channel.iloc[-2] if len(data) > 1 else current_high_channel
            )
            prev_low_channel = (
                low_channel.iloc[-2] if len(data) > 1 else current_low_channel
            )

            # Calculate ATR for position sizing and stops
            atr = self.calculate_atr(data, self.config.atr_period)
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.01

            # Detect choppy market conditions
            is_choppy = self.detect_choppy_market(data)

            # Initialize signal
            signal_value = 0.0
            confidence = 0.0

            # Breakout detection
            breakout_type = "none"

            # Long breakout: price breaks above 55-day high
            if (
                current_price > prev_high_channel
                and current_price >= current_high_channel
            ):
                breakout_strength = (current_price - current_high_channel) / current_atr
                signal_value = min(
                    1.0, breakout_strength * 0.5
                )  # Scale breakout strength
                breakout_type = "long"

            # Short breakout: price breaks below 55-day low
            elif (
                current_price < prev_low_channel
                and current_price <= current_low_channel
            ):
                breakout_strength = (current_low_channel - current_price) / current_atr
                signal_value = max(
                    -1.0, -breakout_strength * 0.5
                )  # Scale breakout strength
                breakout_type = "short"

            # Position within channel (no breakout)
            else:
                channel_width = current_high_channel - current_low_channel
                if channel_width > 0:
                    # Position within channel (-1 at low, +1 at high)
                    channel_position = (
                        2 * (current_price - current_low_channel) / channel_width - 1
                    )
                    signal_value = (
                        channel_position * 0.3
                    )  # Reduced signal for non-breakouts

            # Reduce signal strength in choppy markets
            if is_choppy:
                signal_value *= 0.5

            # Calculate confidence based on breakout strength and market conditions
            if breakout_type != "none":
                breakout_strength_normalized = abs(signal_value)
                trend_strength = 1.0 - (0.5 if is_choppy else 0.0)
                confidence = min(1.0, breakout_strength_normalized * trend_strength)
            else:
                confidence = min(0.5, abs(signal_value))

            # ATR-based stop levels
            atr_stop_long = current_price - (current_atr * self.config.atr_multiplier)
            atr_stop_short = current_price + (current_atr * self.config.atr_multiplier)

            # Channel statistics for position sizing
            channel_width = current_high_channel - current_low_channel
            channel_width_pct = (
                (channel_width / current_price) * 100 if current_price > 0 else 0
            )

            # Metadata for debugging and risk management
            metadata = {
                "breakout_type": breakout_type,
                "high_channel": current_high_channel,
                "low_channel": current_low_channel,
                "current_atr": current_atr,
                "atr_stop_long": atr_stop_long,
                "atr_stop_short": atr_stop_short,
                "channel_width": channel_width,
                "channel_width_pct": channel_width_pct,
                "is_choppy": is_choppy,
                "adx": self.calculate_adx(data, self.config.adx_period),
                "channel_period": self.config.channel_period,
                "atr_multiplier": self.config.atr_multiplier,
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
