import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class FundingCarryConfig(SignalConfig):
    """Configuration for funding carry signal."""

    funding_threshold: float = 0.0007  # 0.07% threshold from Kaiko research
    max_allocation: float = 0.20  # 20% max allocation per signal
    reversal_stop_hours: int = 6  # Stop if funding reverses for 6+ hours
    confidence_multiplier: float = 10.0  # Scale factor for confidence calculation
    min_funding_magnitude: float = 0.0001  # Minimum funding to consider
    max_position_hours: int = 24  # Maximum position hold time

    # Risk management
    max_funding_exposure: float = 2.0  # Maximum funding rate exposure multiplier
    emergency_exit_threshold: float = 0.02  # 2% adverse funding rate change


class PerpFundingCarrySignal(SignalBase):
    """
    Perp Funding Carry Signal - Market-Neutral Strategy

    Strategy: Long assets with deeply negative funding, short positive funding
    Classification: Market-Neutral (M-N)

    Based on Kaiko research showing funding ±0.07% predicts spot mean-reversion & OI washes.
    Runs hourly; caps PnL to funding income to avoid directional bleed.

    Key Features:
    - Long when funding < -0.07%, short when funding > +0.07%
    - Market-neutral delta exposure
    - Risk management for funding rate reversals
    - Confidence scoring based on funding magnitude
    - 20% max allocation with 6-hour reversal stop
    """

    def __init__(self, config: FundingCarryConfig):
        super().__init__(config)
        self.config: FundingCarryConfig = config
        self.funding_history: Dict[str, list] = {}  # Track funding rate history
        self.position_entry_time: Optional[datetime] = None
        self.last_funding_direction: Optional[int] = (
            None  # 1 for positive, -1 for negative
        )

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate funding carry signal from market data and funding rates."""

        if not self.validate_data(data, min_periods=1):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient market data"},
            )

        try:
            # Get current funding rate (mock implementation for now)
            current_funding = await self._get_funding_rate(symbol)
            if current_funding is None:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Funding rate unavailable"},
                )

            # Update funding history for reversal detection
            self._update_funding_history(symbol, current_funding)

            # Check for funding reversal stop condition
            if self._should_stop_for_reversal(symbol):
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        "reason": "funding_reversal_stop",
                        "current_funding": current_funding,
                        "reversal_hours": self._count_reversal_hours(symbol),
                    },
                )

            # Check position time limits
            if self._is_position_time_exceeded():
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        "reason": "max_position_time_exceeded",
                        "current_funding": current_funding,
                        "hours_in_position": self._hours_in_position(),
                    },
                )

            # Generate market-neutral signal based on funding threshold
            signal_value = self._calculate_signal_value(current_funding)
            confidence = self._calculate_confidence(current_funding)

            # Apply risk management scaling
            risk_adjusted_signal = self._apply_risk_management(
                signal_value, current_funding
            )

            # Metadata for analysis and debugging
            metadata = {
                "current_funding": current_funding,
                "funding_threshold": self.config.funding_threshold,
                "signal_direction": (
                    "long"
                    if signal_value > 0
                    else "short" if signal_value < 0 else "neutral"
                ),
                "funding_magnitude": abs(current_funding),
                "confidence_raw": confidence,
                "risk_adjustment": (
                    risk_adjusted_signal / signal_value if signal_value != 0 else 1.0
                ),
                "funding_history_length": len(self.funding_history.get(symbol, [])),
                "reversal_hours": self._count_reversal_hours(symbol),
                "position_hours": self._hours_in_position(),
                "max_allocation": self.config.max_allocation,
                "strategy_type": "market_neutral_carry",
            }

            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=risk_adjusted_signal,
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

    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for the symbol.

        Mock implementation - in production this would query exchange APIs.
        Returns annualized funding rate as decimal (e.g., 0.0007 = 0.07%).
        """
        # Mock funding rates for testing
        # In production: query Binance/Bybit/OKX funding rate APIs

        mock_funding_rates = {
            "BTCUSDT": 0.0005,  # Slightly positive
            "ETHUSDT": -0.0008,  # Negative (should trigger long)
            "ADAUSDT": 0.0012,  # High positive (should trigger short)
            "DOTUSDT": -0.0015,  # Very negative (strong long signal)
            "SOLUSDT": 0.0002,  # Low positive
        }

        # Add some time-based variation to simulate real funding rates
        base_rate = mock_funding_rates.get(symbol, 0.0)

        # Simulate funding rate volatility
        import random

        random.seed(int(datetime.now().timestamp()) % 1000)
        variation = random.uniform(-0.0003, 0.0003)

        return base_rate + variation

    def _update_funding_history(self, symbol: str, funding_rate: float) -> None:
        """Update funding rate history for reversal detection."""
        if symbol not in self.funding_history:
            self.funding_history[symbol] = []

        # Store tuple of (timestamp, funding_rate)
        self.funding_history[symbol].append((datetime.utcnow(), funding_rate))

        # Keep only last 24 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.funding_history[symbol] = [
            (ts, rate) for ts, rate in self.funding_history[symbol] if ts > cutoff_time
        ]

    def _should_stop_for_reversal(self, symbol: str) -> bool:
        """Check if funding has reversed direction for too long."""
        if symbol not in self.funding_history or len(self.funding_history[symbol]) < 2:
            return False

        reversal_hours = self._count_reversal_hours(symbol)
        return reversal_hours >= self.config.reversal_stop_hours

    def _count_reversal_hours(self, symbol: str) -> int:
        """Count consecutive hours of funding reversal."""
        if symbol not in self.funding_history or len(self.funding_history[symbol]) < 2:
            return 0

        history = self.funding_history[symbol]
        if not history:
            return 0

        # Get initial direction when position was likely entered
        initial_rate = history[0][1]
        initial_direction = (
            1
            if initial_rate > self.config.funding_threshold
            else -1 if initial_rate < -self.config.funding_threshold else 0
        )

        if initial_direction == 0:
            return 0

        # Count consecutive hours where funding moved against our position
        reversal_count = 0
        for i, (timestamp, rate) in enumerate(reversed(history)):
            current_direction = (
                1
                if rate > self.config.funding_threshold
                else -1 if rate < -self.config.funding_threshold else 0
            )

            # If direction has reversed, count it
            if current_direction != 0 and current_direction != initial_direction:
                reversal_count += 1
            else:
                break  # Stop counting at first non-reversal

        return reversal_count

    def _is_position_time_exceeded(self) -> bool:
        """Check if maximum position time has been exceeded."""
        if self.position_entry_time is None:
            return False

        hours_in_position = (
            datetime.utcnow() - self.position_entry_time
        ).total_seconds() / 3600
        return hours_in_position > self.config.max_position_hours

    def _hours_in_position(self) -> float:
        """Get hours in current position."""
        if self.position_entry_time is None:
            return 0.0

        return (datetime.utcnow() - self.position_entry_time).total_seconds() / 3600

    def _calculate_signal_value(self, funding_rate: float) -> float:
        """Calculate base signal value based on funding rate thresholds."""

        # Check if funding rate meets minimum magnitude requirement
        if abs(funding_rate) < self.config.min_funding_magnitude:
            return 0.0

        # Market-neutral carry strategy:
        # Long when funding < -threshold (we collect funding)
        # Short when funding > +threshold (we pay funding but expect mean reversion)

        if funding_rate < -self.config.funding_threshold:
            # Negative funding: Long position (collect funding)
            signal_strength = min(
                1.0, abs(funding_rate) / (self.config.funding_threshold * 3)
            )
            return signal_strength

        elif funding_rate > self.config.funding_threshold:
            # Positive funding: Short position (pay funding but expect reversion)
            signal_strength = min(
                1.0, funding_rate / (self.config.funding_threshold * 3)
            )
            return -signal_strength

        else:
            # Within neutral range
            return 0.0

    def _calculate_confidence(self, funding_rate: float) -> float:
        """Calculate confidence based on funding rate magnitude and consistency."""

        # Base confidence from funding magnitude (more aggressive scaling)
        magnitude_confidence = min(
            1.0, abs(funding_rate) * self.config.confidence_multiplier
        )

        # Strong boost for funding rates above threshold
        if abs(funding_rate) >= self.config.funding_threshold:
            # Scale confidence more aggressively for threshold-crossing rates
            threshold_multiple = abs(funding_rate) / self.config.funding_threshold
            magnitude_confidence = min(1.0, 0.5 + (0.4 * threshold_multiple))

        # Additional boost for very extreme rates
        if abs(funding_rate) >= self.config.funding_threshold * 2:
            magnitude_confidence = min(1.0, magnitude_confidence * 1.3)

        # Reduce confidence if we're near reversal stop
        reversal_penalty = 1.0
        reversal_hours = self._count_reversal_hours(
            ""
        )  # Approximate for confidence calc
        if reversal_hours > 0:
            reversal_penalty = max(
                0.3, 1.0 - (reversal_hours / self.config.reversal_stop_hours)
            )

        # Reduce confidence for positions held too long
        time_penalty = 1.0
        hours_in_pos = self._hours_in_position()
        if hours_in_pos > 0:
            time_penalty = max(
                0.5, 1.0 - (hours_in_pos / self.config.max_position_hours)
            )

        final_confidence = magnitude_confidence * reversal_penalty * time_penalty
        return max(0.0, min(1.0, final_confidence))

    def _apply_risk_management(self, signal_value: float, funding_rate: float) -> float:
        """Apply risk management scaling to the signal."""

        if signal_value == 0.0:
            return 0.0

        # Scale by max allocation
        risk_adjusted = signal_value * self.config.max_allocation

        # Additional scaling for extreme funding rates to prevent over-leverage
        if abs(funding_rate) > self.config.funding_threshold * 2:
            # Reduce position size for very extreme funding
            extreme_scale = min(
                1.0, self.config.funding_threshold * 2 / abs(funding_rate)
            )
            risk_adjusted *= extreme_scale

        # Emergency exit scaling
        if abs(funding_rate) > self.config.emergency_exit_threshold:
            risk_adjusted *= 0.1  # Drastically reduce exposure

        return max(
            -self.config.max_allocation, min(self.config.max_allocation, risk_adjusted)
        )

    def reset_position_tracking(self) -> None:
        """Reset position tracking - call when position is closed."""
        self.position_entry_time = None
        self.last_funding_direction = None

    def set_position_entry(self, direction: int) -> None:
        """Set position entry time and direction."""
        self.position_entry_time = datetime.utcnow()
        self.last_funding_direction = direction

    def get_strategy_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get strategy-specific metrics for monitoring."""
        history = self.funding_history.get(symbol, [])

        if not history:
            return {"status": "no_data"}

        recent_rates = [rate for _, rate in history[-24:]]  # Last 24 hours

        return {
            "current_funding": recent_rates[-1] if recent_rates else None,
            "avg_funding_24h": np.mean(recent_rates) if recent_rates else None,
            "funding_volatility": (
                np.std(recent_rates) if len(recent_rates) > 1 else None
            ),
            "reversal_hours": self._count_reversal_hours(symbol),
            "position_hours": self._hours_in_position(),
            "extreme_funding_periods": sum(
                1 for rate in recent_rates if abs(rate) > self.config.funding_threshold
            ),
            "funding_direction_changes": self._count_direction_changes(recent_rates),
            "max_funding_24h": max(recent_rates) if recent_rates else None,
            "min_funding_24h": min(recent_rates) if recent_rates else None,
        }

    def _count_direction_changes(self, rates: list) -> int:
        """Count how many times funding direction changed in the period."""
        if len(rates) < 2:
            return 0

        changes = 0
        for i in range(1, len(rates)):
            prev_sign = 1 if rates[i - 1] > 0 else -1 if rates[i - 1] < 0 else 0
            curr_sign = 1 if rates[i] > 0 else -1 if rates[i] < 0 else 0

            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                changes += 1

        return changes
