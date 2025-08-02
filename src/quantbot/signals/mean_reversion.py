import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class MeanReversionConfig(SignalConfig):
    """Configuration for mean reversion signal."""

    lookback_days: int = 3  # Short-term lookback for crypto volatility
    zscore_threshold: float = 2.0  # ±2σ threshold for entry signals
    min_liquidity_volume: float = 1_000_000  # USD volume filter for liquidity
    max_spread_bps: float = 50  # Max spread in basis points
    liquidation_volume_multiple: float = (
        3.0  # Volume spike threshold for liquidation detection
    )
    funding_rate_neutral_threshold: float = 0.01  # Placeholder for funding rate filter
    min_periods: int = 10  # Minimum data points required


class ShortTermMeanReversionSignal(SignalBase):
    """
    Short-Term Mean Reversion Signal for Crypto Markets.

    Enhanced mean reversion strategy specifically designed for crypto perps:
    - 3-day lookback with ±2σ threshold detection
    - Funding rate neutral filter (placeholder for Phase 2)
    - Liquidation overshoot detection via volume spikes
    - Large-cap/high-liquidity filter to avoid slippage
    - Z-score based contrarian entry/exit signals
    - Confidence scoring based on deviation magnitude and market conditions

    Strategy Logic:
    - Fade extreme moves beyond 2σ from 3-day rolling mean
    - Higher confidence on larger deviations and clean market conditions
    - Filter out moves during extreme funding rate environments
    - Detect potential liquidation cascades for opportunistic entries
    """

    def __init__(self, config: MeanReversionConfig):
        super().__init__(config)
        self.config: MeanReversionConfig = config

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate mean reversion signal from price data."""

        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # Calculate 3-day rolling statistics
            window = self.config.lookback_days * 24  # Convert days to hours for 1h data
            rolling_mean = (
                data["close"].rolling(window=window, min_periods=window // 2).mean()
            )
            rolling_std = (
                data["close"].rolling(window=window, min_periods=window // 2).std()
            )

            # Current price and latest statistics
            current_price = data.iloc[-1]["close"]
            latest_mean = rolling_mean.iloc[-1]
            latest_std = rolling_std.iloc[-1]

            if pd.isna(latest_mean) or pd.isna(latest_std) or latest_std == 0:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Invalid rolling statistics"},
                )

            # Calculate Z-score
            z_score = (current_price - latest_mean) / latest_std

            # Liquidity filter - check recent volume
            recent_volume_usd = self._estimate_volume_usd(data, current_price)
            liquidity_pass = recent_volume_usd >= self.config.min_liquidity_volume

            # Liquidation detection - volume spike analysis
            liquidation_detected = self._detect_liquidation_cascade(data, z_score)

            # Funding rate filter (placeholder - will be enhanced in Phase 2)
            funding_neutral = self._check_funding_neutrality()

            # Generate contrarian signal based on Z-score
            signal_value = self._calculate_signal_value(z_score, liquidation_detected)

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                z_score=z_score,
                liquidity_pass=liquidity_pass,
                funding_neutral=funding_neutral,
                liquidation_detected=liquidation_detected,
                data_quality=len(data) >= self.config.min_periods * 2,
            )

            # Apply liquidity filter - zero out signals for illiquid pairs
            if not liquidity_pass:
                signal_value = 0.0
                confidence = 0.0

            # Comprehensive metadata for analysis and debugging
            metadata = {
                "z_score": float(z_score),
                "rolling_mean": float(latest_mean),
                "rolling_std": float(latest_std),
                "current_price": float(current_price),
                "volume_usd_estimate": float(recent_volume_usd),
                "liquidity_pass": liquidity_pass,
                "liquidation_detected": liquidation_detected,
                "funding_neutral": funding_neutral,
                "lookback_hours": window,
                "threshold_used": self.config.zscore_threshold,
                "deviation_magnitude": abs(float(z_score)),
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

    def _calculate_signal_value(
        self, z_score: float, liquidation_detected: bool
    ) -> float:
        """Calculate contrarian signal value based on Z-score."""

        # Base contrarian signal - fade the move
        if abs(z_score) < self.config.zscore_threshold:
            return 0.0  # No signal if within normal range

        # Contrarian signal strength increases with deviation magnitude
        deviation_magnitude = abs(z_score) - self.config.zscore_threshold
        base_strength = min(
            1.0, deviation_magnitude / 2.0
        )  # Scale additional deviation

        # Contrarian signal: negative Z-score (oversold) -> buy signal (+)
        # Positive Z-score (overbought) -> sell signal (-)
        signal_direction = -np.sign(z_score)

        # Boost signal if liquidation cascade detected (more extreme mean reversion opportunity)
        liquidation_boost = 1.5 if liquidation_detected else 1.0

        signal_value = signal_direction * base_strength * liquidation_boost

        # Ensure signal stays within bounds
        return max(-1.0, min(1.0, signal_value))

    def _calculate_confidence(
        self,
        z_score: float,
        liquidity_pass: bool,
        funding_neutral: bool,
        liquidation_detected: bool,
        data_quality: bool,
    ) -> float:
        """Calculate confidence based on signal strength and market conditions."""

        if abs(z_score) < self.config.zscore_threshold:
            return 0.0

        # Base confidence from deviation magnitude
        deviation_factor = min(1.0, (abs(z_score) - self.config.zscore_threshold) / 3.0)
        base_confidence = 0.3 + (deviation_factor * 0.4)  # 0.3 to 0.7 range

        # Confidence modifiers
        confidence_multiplier = 1.0

        # Liquidity boost
        if liquidity_pass:
            confidence_multiplier *= 1.2
        else:
            confidence_multiplier *= 0.3  # Heavy penalty for illiquid pairs

        # Funding environment boost (placeholder)
        if funding_neutral:
            confidence_multiplier *= 1.1
        else:
            confidence_multiplier *= 0.9

        # Liquidation cascade boost - higher confidence in extreme dislocations
        if liquidation_detected:
            confidence_multiplier *= 1.3

        # Data quality factor
        if data_quality:
            confidence_multiplier *= 1.0
        else:
            confidence_multiplier *= 0.8

        final_confidence = base_confidence * confidence_multiplier

        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, final_confidence))

    def _estimate_volume_usd(self, data: pd.DataFrame, current_price: float) -> float:
        """Estimate recent USD volume for liquidity assessment."""

        # Use last 24 hours of volume data
        recent_hours = min(24, len(data))
        recent_data = data.tail(recent_hours)

        # Estimate USD volume using average of high/low as price proxy
        avg_prices = (recent_data["high"] + recent_data["low"]) / 2
        volume_usd = (recent_data["volume"] * avg_prices).sum()

        return float(volume_usd)

    def _detect_liquidation_cascade(self, data: pd.DataFrame, z_score: float) -> bool:
        """Detect potential liquidation cascades via volume and price action."""

        if len(data) < 6:  # Need at least 6 periods for analysis
            return False

        # Compare recent volume to historical average
        recent_volume = data.iloc[-3:]["volume"].mean()  # Last 3 hours
        historical_volume = data.iloc[-24:-3]["volume"].mean()  # Previous 21 hours

        if historical_volume == 0:
            return False

        volume_ratio = recent_volume / historical_volume

        # Liquidation signals:
        # 1. High volume spike (3x+ normal)
        # 2. Extreme price movement (|z_score| > 2.5)
        # 3. Recent volatility spike
        recent_volatility = data.iloc[-6:]["close"].std()
        historical_volatility = data.iloc[-24:-6]["close"].std()
        volatility_ratio = (
            recent_volatility / historical_volatility
            if historical_volatility > 0
            else 1.0
        )

        liquidation_conditions = [
            volume_ratio >= self.config.liquidation_volume_multiple,
            abs(z_score) >= 2.5,
            volatility_ratio >= 2.0,
        ]

        # Need at least 2 of 3 conditions for liquidation detection
        return sum(liquidation_conditions) >= 2

    def _check_funding_neutrality(self) -> bool:
        """
        Check if funding rate environment is neutral for mean reversion.

        Placeholder implementation for Phase 2 enhancement.
        Will integrate with exchange APIs to fetch current funding rates.

        Returns:
            bool: True if funding environment is neutral/favorable for mean reversion
        """

        # TODO: Implement actual funding rate checks
        # - Fetch current funding rate from exchange
        # - Check if absolute funding rate < threshold
        # - Consider funding rate trend (not extremely negative/positive)
        # - Factor in cross-exchange funding spreads

        # For now, assume neutral funding environment
        return True
