import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class OIDivergenceConfig(SignalConfig):
    """Configuration for Open Interest / Price Divergence signal."""

    # Core parameters based on Kaiko research
    oi_momentum_window: int = 24  # Hours for OI momentum calculation
    price_momentum_window: int = 24  # Hours for price momentum calculation
    divergence_threshold: float = 0.15  # Minimum divergence strength threshold
    min_venues: int = 3  # Minimum venues for cross-venue confirmation

    # Volume-weighted OI analysis
    volume_weight_enabled: bool = True
    volume_lookback_hours: int = 168  # 7 days for volume analysis

    # Liquidation cascade detection
    volume_spike_threshold: float = (
        2.5  # Volume spike multiplier for liquidation detection
    )
    liquidation_lookback_hours: int = 6  # Hours to look back for liquidation signals

    # Signal classification (2-S symmetric directional)
    signal_smoothing_window: int = 3  # Hours for signal smoothing
    confidence_decay_hours: int = 12  # Hours for confidence decay

    # Risk management
    max_signal_strength: float = 1.0  # Maximum signal value
    min_confidence: float = 0.7  # Minimum confidence threshold from requirements
    venue_weight_method: str = "volume"  # "equal" or "volume" weighting

    # Early warning system
    oi_flush_threshold: float = 0.3  # OI decrease threshold for flush detection
    price_flatness_threshold: float = (
        0.02  # Price movement threshold for "flat" detection
    )
    warning_lookahead_hours: int = 6  # Early warning time horizon


class OIPriceDivergenceSignal(SignalBase):
    """
    Open Interest / Price Divergence Signal - Directional Strategy

    Strategy: Detect divergences between OI momentum and price momentum
    Classification: Symmetric Directional (2-S)

    Key Patterns:
    - Rising OI + falling price → build bearish signal
    - OI flush + price flat → prep long signal
    - Volume-weighted OI analysis for accuracy
    - Cross-venue confirmation to avoid spoof noise

    Based on Kaiko research showing OI/price divergences preceded March '25 15% flush.
    Implements liquidation cascade prediction and early warning capabilities.

    Features:
    - Multi-venue OI confirmation (≥3 venues required)
    - Volume-weighted OI momentum calculation
    - Liquidation cascade detection via volume spikes
    - Early warning system for market preparation
    - Precision-recall >0.7 targeting
    - Symmetric directional signals (-1 to +1)
    """

    def __init__(self, config: OIDivergenceConfig):
        super().__init__(config)
        self.config: OIDivergenceConfig = config

        # Multi-venue OI tracking
        self.oi_history: Dict[str, Dict[str, List[Tuple[datetime, float, float]]]] = (
            {}
        )  # symbol -> venue -> [(timestamp, oi, volume)]
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = (
            {}
        )  # symbol -> [(timestamp, price)]

        # Signal state tracking
        self.last_signals: Dict[str, List[Tuple[datetime, float, float]]] = (
            {}
        )  # symbol -> [(timestamp, signal, confidence)]
        self.warning_states: Dict[str, Dict[str, Any]] = (
            {}
        )  # symbol -> warning metadata

        # Mock venue setup (structure for real implementation)
        self.venues = ["binance", "bybit", "okx", "coinbase", "ftx"]

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate OI/Price divergence signal from market data."""

        if not self.validate_data(
            data,
            min_periods=max(
                self.config.oi_momentum_window, self.config.price_momentum_window
            ),
        ):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient market data"},
            )

        try:
            # Get multi-venue OI data (mock implementation for now)
            venue_oi_data = await self._get_multi_venue_oi_data(symbol)
            if len(venue_oi_data) < self.config.min_venues:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        "error": "Insufficient venue coverage",
                        "venues_available": len(venue_oi_data),
                        "min_required": self.config.min_venues,
                    },
                )

            # Update price history
            self._update_price_history(symbol, data)

            # Calculate OI momentum across venues
            oi_momentum = self._calculate_oi_momentum(symbol, venue_oi_data)
            if oi_momentum is None:
                return self._create_no_signal_result(
                    symbol, "oi_momentum_calculation_failed"
                )

            # Calculate price momentum
            price_momentum = self._calculate_price_momentum(symbol, data)
            if price_momentum is None:
                return self._create_no_signal_result(
                    symbol, "price_momentum_calculation_failed"
                )

            # Detect liquidation cascades
            liquidation_signal = self._detect_liquidation_cascade(
                symbol, data, venue_oi_data
            )

            # Calculate divergence strength
            divergence_strength = self._calculate_divergence_strength(
                oi_momentum, price_momentum
            )

            # Generate base signal from divergence patterns
            base_signal = self._generate_base_signal(
                oi_momentum, price_momentum, divergence_strength
            )

            # Apply cross-venue confirmation
            venue_confidence = self._calculate_venue_confidence(
                venue_oi_data, oi_momentum
            )

            # Apply volume weighting if enabled
            if self.config.volume_weight_enabled:
                volume_adjustment = self._calculate_volume_adjustment(
                    symbol, venue_oi_data
                )
                base_signal *= volume_adjustment

            # Apply signal smoothing
            smoothed_signal = self._apply_signal_smoothing(symbol, base_signal)

            # Calculate final confidence score
            confidence = self._calculate_final_confidence(
                divergence_strength, venue_confidence, liquidation_signal, symbol
            )

            # Early warning system
            warning_metadata = self._generate_early_warning(
                symbol, oi_momentum, price_momentum, data
            )

            # Apply risk management bounds
            final_signal = max(
                -self.config.max_signal_strength,
                min(self.config.max_signal_strength, smoothed_signal),
            )

            # Store signal for smoothing and analysis
            self._store_signal_history(symbol, final_signal, confidence)

            # Comprehensive metadata
            metadata = {
                "oi_momentum": oi_momentum,
                "price_momentum": price_momentum,
                "divergence_strength": divergence_strength,
                "venue_count": len(venue_oi_data),
                "venue_confidence": venue_confidence,
                "liquidation_signal": liquidation_signal,
                "base_signal": base_signal,
                "smoothed_signal": smoothed_signal,
                "volume_adjustment": (
                    volume_adjustment if self.config.volume_weight_enabled else 1.0
                ),
                "signal_pattern": self._classify_signal_pattern(
                    oi_momentum, price_momentum
                ),
                "early_warning": warning_metadata,
                "venues_data": {
                    venue: {
                        "oi_momentum": data["oi_momentum"],
                        "volume_ratio": data["volume_ratio"],
                    }
                    for venue, data in venue_oi_data.items()
                },
                "strategy_type": "oi_price_divergence",
                "classification": "symmetric_directional",
                "cascade_risk": liquidation_signal > 0.5,
                "signal_strength_raw": abs(final_signal),
                "confidence_components": {
                    "divergence": min(
                        1.0, divergence_strength / self.config.divergence_threshold
                    ),
                    "venue": venue_confidence,
                    "liquidation": liquidation_signal,
                    "temporal": self._calculate_temporal_confidence(symbol),
                },
            }

            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=final_signal,
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

    async def _get_multi_venue_oi_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get multi-venue open interest data.

        Mock implementation - in production this would query multiple exchange APIs.
        Returns dict mapping venue to OI data including momentum and volume metrics.
        """
        venue_data = {}

        # Mock implementation with realistic OI patterns
        for i, venue in enumerate(self.venues):
            if venue not in self.oi_history.get(symbol, {}):
                if symbol not in self.oi_history:
                    self.oi_history[symbol] = {}
                self.oi_history[symbol][venue] = []

            # Simulate venue-specific OI data with realistic patterns
            base_oi = 100000000 + (i * 20000000)  # Different base OI per venue

            # Add time-based variation and patterns
            import random

            current_time = datetime.utcnow()
            random.seed(int(current_time.timestamp()) % 1000 + i)

            # Simulate realistic OI patterns:
            # - Rising OI + falling price pattern (bearish)
            # - OI flush + flat price pattern (bullish setup)
            if random.random() < 0.3:  # 30% chance of divergence pattern
                if random.random() < 0.5:
                    # Rising OI pattern (potential bearish divergence)
                    oi_change = random.uniform(0.05, 0.15)  # 5-15% increase
                    volume_multiplier = random.uniform(1.2, 2.0)
                else:
                    # OI flush pattern (potential bullish setup)
                    oi_change = random.uniform(-0.25, -0.10)  # 10-25% decrease
                    volume_multiplier = random.uniform(2.0, 3.5)
            else:
                # Normal market conditions
                oi_change = random.uniform(-0.05, 0.05)  # ±5% normal variation
                volume_multiplier = random.uniform(0.8, 1.2)

            current_oi = base_oi * (1 + oi_change)
            current_volume = base_oi * 0.1 * volume_multiplier  # Volume as % of OI

            # Store in history
            self.oi_history[symbol][venue].append(
                (current_time, current_oi, current_volume)
            )

            # Keep only recent history
            cutoff_time = current_time - timedelta(
                hours=self.config.volume_lookback_hours
            )
            self.oi_history[symbol][venue] = [
                (ts, oi, vol)
                for ts, oi, vol in self.oi_history[symbol][venue]
                if ts > cutoff_time
            ]

            # Calculate OI momentum for this venue
            oi_momentum = self._calculate_venue_oi_momentum(symbol, venue)
            volume_ratio = self._calculate_venue_volume_ratio(symbol, venue)

            venue_data[venue] = {
                "current_oi": current_oi,
                "current_volume": current_volume,
                "oi_momentum": oi_momentum,
                "volume_ratio": volume_ratio,
                "data_quality": random.uniform(0.8, 1.0),  # Mock data quality score
            }

        # Remove venues with insufficient data
        return {
            venue: data
            for venue, data in venue_data.items()
            if data["oi_momentum"] is not None and data["volume_ratio"] is not None
        }

    def _update_price_history(self, symbol: str, data: pd.DataFrame) -> None:
        """Update price history for momentum calculation."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        current_price = float(data.iloc[-1]["close"])
        current_time = datetime.utcnow()

        self.price_history[symbol].append((current_time, current_price))

        # Keep only recent history
        cutoff_time = current_time - timedelta(
            hours=max(self.config.price_momentum_window, 168)
        )
        self.price_history[symbol] = [
            (ts, price) for ts, price in self.price_history[symbol] if ts > cutoff_time
        ]

    def _calculate_venue_oi_momentum(self, symbol: str, venue: str) -> Optional[float]:
        """Calculate OI momentum for a specific venue."""
        if symbol not in self.oi_history or venue not in self.oi_history[symbol]:
            return None

        history = self.oi_history[symbol][venue]
        if len(history) < 2:
            return None

        # Get OI values within momentum window
        cutoff_time = datetime.utcnow() - timedelta(
            hours=self.config.oi_momentum_window
        )
        recent_history = [(ts, oi, vol) for ts, oi, vol in history if ts > cutoff_time]

        if len(recent_history) < 2:
            return None

        # Calculate momentum as percentage change
        start_oi = recent_history[0][1]
        end_oi = recent_history[-1][1]

        if start_oi == 0:
            return None

        momentum = (end_oi - start_oi) / start_oi
        return momentum

    def _calculate_venue_volume_ratio(self, symbol: str, venue: str) -> Optional[float]:
        """Calculate volume ratio for a specific venue (current vs average)."""
        if symbol not in self.oi_history or venue not in self.oi_history[symbol]:
            return None

        history = self.oi_history[symbol][venue]
        if len(history) < 5:  # Need minimum history
            return None

        # Current volume
        current_volume = history[-1][2]

        # Average volume over lookback period
        volumes = [vol for _, _, vol in history[:-1]]  # Exclude current
        avg_volume = np.mean(volumes) if volumes else current_volume

        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def _calculate_oi_momentum(
        self, symbol: str, venue_oi_data: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate aggregated OI momentum across venues."""
        if not venue_oi_data:
            return None

        if self.config.venue_weight_method == "volume":
            # Volume-weighted OI momentum
            total_weight = 0
            weighted_momentum = 0

            for venue, data in venue_oi_data.items():
                if data["oi_momentum"] is not None and data["volume_ratio"] is not None:
                    weight = data["current_volume"]
                    weighted_momentum += data["oi_momentum"] * weight
                    total_weight += weight

            return weighted_momentum / total_weight if total_weight > 0 else None
        else:
            # Equal-weighted OI momentum
            momentums = [
                data["oi_momentum"]
                for data in venue_oi_data.values()
                if data["oi_momentum"] is not None
            ]
            return np.mean(momentums) if momentums else None

    def _calculate_price_momentum(
        self, symbol: str, data: pd.DataFrame
    ) -> Optional[float]:
        """Calculate price momentum over the specified window."""
        if len(data) < self.config.price_momentum_window:
            return None

        # Use the last N periods for momentum calculation
        start_price = float(data.iloc[-self.config.price_momentum_window]["close"])
        end_price = float(data.iloc[-1]["close"])

        if start_price == 0:
            return None

        momentum = (end_price - start_price) / start_price
        return momentum

    def _calculate_divergence_strength(
        self, oi_momentum: float, price_momentum: float
    ) -> float:
        """Calculate the strength of OI/Price divergence."""
        # Divergence occurs when OI and price move in opposite directions
        # Strength is measured by the magnitude of opposite movements

        # Normalize momentums to similar scales
        oi_norm = np.tanh(oi_momentum * 10)  # Scale OI momentum
        price_norm = np.tanh(price_momentum * 5)  # Scale price momentum

        # Calculate divergence as negative correlation strength
        divergence = -oi_norm * price_norm  # Negative correlation

        # Only consider positive divergences (opposite movements)
        divergence = max(0, float(divergence))

        # Add magnitude component for stronger divergences
        magnitude_bonus = (abs(oi_norm) + abs(price_norm)) / 2

        return divergence + (magnitude_bonus * 0.3)

    def _generate_base_signal(
        self, oi_momentum: float, price_momentum: float, divergence_strength: float
    ) -> float:
        """Generate base signal from OI/Price patterns."""

        # Pattern 1: Rising OI + falling price → bearish signal
        if oi_momentum > 0.05 and price_momentum < -0.02:
            signal_strength = min(1.0, divergence_strength * 2)
            return -signal_strength  # Negative (short signal)

        # Pattern 2: OI flush + price flat → bullish signal
        elif (
            oi_momentum < -0.1
            and abs(price_momentum) < self.config.price_flatness_threshold
        ):
            signal_strength = min(1.0, abs(oi_momentum) * 3)
            return signal_strength  # Positive (long signal)

        # Pattern 3: Strong divergence patterns (lowered threshold)
        elif divergence_strength > self.config.divergence_threshold * 0.8:
            # Use OI direction for signal direction (contrarian)
            if oi_momentum > 0 and price_momentum < 0:
                return -min(
                    1.0, divergence_strength * 2
                )  # High OI + falling price → bearish
            elif oi_momentum < 0 and abs(price_momentum) < 0.03:
                return min(1.0, abs(oi_momentum) * 2)  # Low OI + flat price → bullish

        return 0.0

    def _detect_liquidation_cascade(
        self, symbol: str, data: pd.DataFrame, venue_oi_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Detect liquidation cascade signals via volume spikes."""

        # Check for volume spikes in market data
        if len(data) < self.config.liquidation_lookback_hours:
            return 0.0

        recent_data = data.tail(self.config.liquidation_lookback_hours)
        current_volume = float(recent_data.iloc[-1]["volume"])
        avg_volume = float(recent_data["volume"].mean())

        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Check for OI volume spikes across venues
        venue_volume_spikes = []
        for venue, data_dict in venue_oi_data.items():
            if data_dict["volume_ratio"] is not None:
                venue_volume_spikes.append(data_dict["volume_ratio"])

        max_venue_spike = max(venue_volume_spikes) if venue_volume_spikes else 1.0

        # Combine market and OI volume signals
        combined_spike = (volume_spike + max_venue_spike) / 2

        # Normalize to 0-1 range
        if combined_spike >= self.config.volume_spike_threshold:
            return min(
                1.0, (combined_spike - 1.0) / (self.config.volume_spike_threshold - 1.0)
            )

        return 0.0

    def _calculate_venue_confidence(
        self, venue_oi_data: Dict[str, Dict[str, Any]], oi_momentum: float
    ) -> float:
        """Calculate confidence based on cross-venue confirmation."""

        if len(venue_oi_data) < self.config.min_venues:
            return 0.0

        # Check consistency across venues
        venue_momentums = [
            data["oi_momentum"]
            for data in venue_oi_data.values()
            if data["oi_momentum"] is not None
        ]

        if len(venue_momentums) < self.config.min_venues:
            return 0.0

        # Calculate momentum agreement (how consistent directions are)
        positive_count = sum(1 for m in venue_momentums if m > 0.01)
        negative_count = sum(1 for m in venue_momentums if m < -0.01)
        total_count = len(venue_momentums)

        # High confidence if most venues agree on direction
        agreement_ratio = max(positive_count, negative_count) / total_count

        # Bonus for data quality
        avg_quality = np.mean([data["data_quality"] for data in venue_oi_data.values()])

        # Venue coverage bonus
        coverage_bonus = min(1.0, len(venue_oi_data) / 5)  # Max bonus at 5 venues

        return agreement_ratio * avg_quality * coverage_bonus

    def _calculate_volume_adjustment(
        self, symbol: str, venue_oi_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate volume-based signal adjustment."""

        volume_ratios = [
            data["volume_ratio"]
            for data in venue_oi_data.values()
            if data["volume_ratio"] is not None
        ]

        if not volume_ratios:
            return 1.0

        # Higher volume = stronger signal
        avg_volume_ratio = np.mean(volume_ratios)

        # Scale adjustment: normal volume (1.0) = no change, high volume = boost
        adjustment = 0.5 + (0.5 * min(2.0, avg_volume_ratio))  # Range: 0.5 to 1.5

        return adjustment

    def _apply_signal_smoothing(self, symbol: str, base_signal: float) -> float:
        """Apply temporal smoothing to reduce noise."""

        if symbol not in self.last_signals:
            return base_signal

        recent_signals = [
            sig
            for ts, sig, conf in self.last_signals[symbol]
            if (datetime.utcnow() - ts).total_seconds() / 3600
            <= self.config.signal_smoothing_window
        ]

        if not recent_signals:
            return base_signal

        # Weighted average with current signal having highest weight
        weights = [0.5]  # Current signal weight
        signals = [base_signal]

        for i, sig in enumerate(reversed(recent_signals[-3:])):  # Last 3 signals
            weight = 0.3 / (i + 1)  # Decreasing weights
            weights.append(weight)
            signals.append(sig)

        total_weight = sum(weights)
        weighted_sum = sum(w * s for w, s in zip(weights, signals))

        return weighted_sum / total_weight

    def _calculate_final_confidence(
        self,
        divergence_strength: float,
        venue_confidence: float,
        liquidation_signal: float,
        symbol: str,
    ) -> float:
        """Calculate final confidence score with all components."""

        # Base confidence from divergence strength
        divergence_conf = min(
            1.0, divergence_strength / self.config.divergence_threshold
        )

        # Temporal consistency bonus
        temporal_conf = self._calculate_temporal_confidence(symbol)

        # Liquidation risk component (adds confidence if cascade detected)
        liquidation_conf = (
            liquidation_signal * 0.3
        )  # Modest boost from liquidation signals

        # Combine components with weights
        final_confidence = (
            divergence_conf * 0.4  # 40% from divergence strength
            + venue_confidence * 0.35  # 35% from venue confirmation
            + temporal_conf * 0.15  # 15% from temporal consistency
            + liquidation_conf * 0.1  # 10% from liquidation detection
        )

        return max(0.0, min(1.0, final_confidence))

    def _calculate_temporal_confidence(self, symbol: str) -> float:
        """Calculate confidence based on signal temporal consistency."""

        if symbol not in self.last_signals or len(self.last_signals[symbol]) < 2:
            return 0.5  # Neutral confidence for new signals

        # Check recent signal consistency
        recent_signals = [
            sig
            for ts, sig, conf in self.last_signals[symbol]
            if (datetime.utcnow() - ts).total_seconds() / 3600
            <= self.config.confidence_decay_hours
        ]

        if len(recent_signals) < 2:
            return 0.5

        # Calculate directional consistency
        positive_count = sum(1 for sig in recent_signals if sig > 0.1)
        negative_count = sum(1 for sig in recent_signals if sig < -0.1)
        total_count = len(recent_signals)

        consistency_ratio = max(positive_count, negative_count) / total_count

        return consistency_ratio

    def _generate_early_warning(
        self, symbol: str, oi_momentum: float, price_momentum: float, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate early warning system metadata."""

        warnings = {}

        # Check for OI flush + price flat pattern (bullish setup)
        if (
            oi_momentum < -self.config.oi_flush_threshold
            and abs(price_momentum) < self.config.price_flatness_threshold
        ):
            warnings["oi_flush_detected"] = {
                "type": "bullish_setup",
                "oi_flush_magnitude": abs(oi_momentum),
                "price_stability": abs(price_momentum),
                "estimated_timeframe": f"{self.config.warning_lookahead_hours}h",
                "confidence": min(
                    1.0, abs(oi_momentum) / self.config.oi_flush_threshold
                ),
            }

        # Check for rising OI + falling price (bearish warning)
        if oi_momentum > 0.05 and price_momentum < -0.02:
            warnings["bearish_divergence_building"] = {
                "type": "bearish_warning",
                "oi_buildup": oi_momentum,
                "price_weakness": price_momentum,
                "cascade_risk": "high" if oi_momentum > 0.15 else "medium",
                "estimated_impact": "15%+" if oi_momentum > 0.2 else "5-15%",
            }

        # Volume analysis for early warning
        if len(data) >= 24:
            recent_volume = float(data.tail(24)["volume"].mean())
            historical_volume = float(data["volume"].mean())
            volume_trend = (
                recent_volume / historical_volume if historical_volume > 0 else 1.0
            )

            if volume_trend > 1.5:
                warnings["volume_buildup"] = {
                    "type": "activity_increase",
                    "volume_ratio": volume_trend,
                    "interpretation": "increased market interest",
                }

        return warnings

    def _classify_signal_pattern(
        self, oi_momentum: float, price_momentum: float
    ) -> str:
        """Classify the type of signal pattern detected."""

        if oi_momentum > 0.05 and price_momentum < -0.02:
            return "rising_oi_falling_price"
        elif oi_momentum < -0.1 and abs(price_momentum) < 0.02:
            return "oi_flush_flat_price"
        elif oi_momentum > 0.1 and price_momentum > 0.05:
            return "oi_price_alignment_bullish"
        elif oi_momentum < -0.05 and price_momentum < -0.05:
            return "oi_price_alignment_bearish"
        elif (
            abs(oi_momentum) > 0.1
            and abs(price_momentum) > 0.05
            and np.sign(oi_momentum) != np.sign(price_momentum)
        ):
            return "strong_divergence"
        else:
            return "neutral_or_weak"

    def _store_signal_history(
        self, symbol: str, signal: float, confidence: float
    ) -> None:
        """Store signal history for smoothing and analysis."""

        if symbol not in self.last_signals:
            self.last_signals[symbol] = []

        current_time = datetime.utcnow()
        self.last_signals[symbol].append((current_time, signal, confidence))

        # Keep only recent history
        cutoff_time = current_time - timedelta(hours=24)
        self.last_signals[symbol] = [
            (ts, sig, conf)
            for ts, sig, conf in self.last_signals[symbol]
            if ts > cutoff_time
        ]

    def _create_no_signal_result(self, symbol: str, reason: str) -> SignalResult:
        """Helper to create no-signal result with reason."""
        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={
                "reason": reason,
                "strategy_type": "oi_price_divergence",
                "classification": "symmetric_directional",
            },
        )

    def get_strategy_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get strategy-specific metrics for monitoring."""

        metrics = {"strategy": "oi_price_divergence"}

        # Venue coverage metrics
        if symbol in self.oi_history:
            venue_count = len(self.oi_history[symbol])
            metrics["venue_coverage"] = {
                "active_venues": venue_count,
                "coverage_ratio": venue_count / len(self.venues),
                "venues": list(self.oi_history[symbol].keys()),
            }

        # Signal history metrics
        if symbol in self.last_signals:
            recent_signals = self.last_signals[symbol][-10:]  # Last 10 signals
            if recent_signals:
                signals = [sig for _, sig, _ in recent_signals]
                confidences = [conf for _, _, conf in recent_signals]

                metrics["signal_history"] = {
                    "recent_signal_avg": np.mean(signals),
                    "signal_volatility": np.std(signals) if len(signals) > 1 else 0.0,
                    "avg_confidence": np.mean(confidences),
                    "signal_count_24h": len(recent_signals),
                }

        # Early warning status
        if symbol in self.warning_states:
            metrics["early_warnings"] = self.warning_states[symbol]

        return metrics

    def reset_state(self, symbol: Optional[str] = None) -> None:
        """Reset signal state for testing or restart."""
        if symbol:
            # Reset specific symbol
            if symbol in self.oi_history:
                del self.oi_history[symbol]
            if symbol in self.price_history:
                del self.price_history[symbol]
            if symbol in self.last_signals:
                del self.last_signals[symbol]
            if symbol in self.warning_states:
                del self.warning_states[symbol]
        else:
            # Reset all state
            self.oi_history.clear()
            self.price_history.clear()
            self.last_signals.clear()
            self.warning_states.clear()
