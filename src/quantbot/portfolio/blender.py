"""
Portfolio blender for combining multiple trading signals intelligently.

Handles signal conflicts, risk budgeting, and position normalization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..signals.base import SignalResult


class ConflictResolution(Enum):
    """Methods for resolving conflicting signals."""

    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STRONGEST_SIGNAL = "strongest_signal"
    RISK_PARITY = "risk_parity"


@dataclass
class BlenderConfig:
    """Configuration for portfolio signal blender."""

    conflict_resolution: ConflictResolution = ConflictResolution.CONFIDENCE_WEIGHTED
    max_position_size: float = 1.0  # Maximum absolute position (-1 to +1)
    min_signal_confidence: float = 0.1  # Filter out low confidence signals
    correlation_lookback: int = 100  # Periods for signal correlation estimation
    risk_budget_per_signal: Dict[str, float] = None  # Signal-specific risk budgets
    decay_factor: float = 0.95  # Exponential decay for historical signal performance

    def __post_init__(self):
        if self.risk_budget_per_signal is None:
            self.risk_budget_per_signal = {}


@dataclass
class BlendedSignal:
    """Result from blending multiple signals."""

    symbol: str
    timestamp: datetime
    final_position: float  # Combined position size [-1, +1]
    confidence: float  # Combined confidence [0, 1]
    individual_signals: Dict[str, SignalResult]  # Original signals
    signal_contributions: Dict[str, float]  # How much each signal contributed
    metadata: Dict[str, Any]


class PortfolioBlender:
    """
    Intelligent signal blender for crypto trading strategies.

    Combines momentum, breakout, and mean reversion signals with:
    - Conflict resolution when signals disagree
    - Risk budgeting across signal types
    - Confidence-weighted position sizing
    - Signal correlation and performance tracking
    """

    def __init__(self, config: BlenderConfig):
        self.config = config
        self.signal_history: Dict[str, List[SignalResult]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def blend_signals(
        self, signals: Dict[str, SignalResult], symbol: str
    ) -> BlendedSignal:
        """
        Blend multiple signals into a single trading position.

        Args:
            signals: Dictionary of {signal_name: SignalResult}
            symbol: Trading symbol

        Returns:
            BlendedSignal with combined position and metadata
        """

        # Filter signals by minimum confidence
        filtered_signals = self._filter_signals_by_confidence(signals)

        if not filtered_signals:
            return BlendedSignal(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                final_position=0.0,
                confidence=0.0,
                individual_signals=signals,
                signal_contributions={},
                metadata={"reason": "no_signals_above_confidence_threshold"},
            )

        # Calculate signal weights based on configuration and performance
        signal_weights = self._calculate_signal_weights(filtered_signals)

        # Resolve conflicts and blend signals
        blended_position, blend_confidence = self._resolve_signal_conflicts(
            filtered_signals, signal_weights
        )

        # Calculate individual signal contributions
        contributions = self._calculate_signal_contributions(
            filtered_signals, signal_weights, blended_position
        )

        # Apply position sizing constraints
        final_position = self._apply_position_constraints(blended_position)

        # Update signal history for future correlation analysis
        self._update_signal_history(filtered_signals)

        # Comprehensive metadata
        metadata = {
            "conflict_resolution": self.config.conflict_resolution.value,
            "signal_weights": signal_weights,
            "signals_used": list(filtered_signals.keys()),
            "signals_filtered": list(
                set(signals.keys()) - set(filtered_signals.keys())
            ),
            "correlation_data": self._get_correlation_metadata(),
            "position_scaling": (
                final_position / blended_position if blended_position != 0 else 1.0
            ),
        }

        return BlendedSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            final_position=final_position,
            confidence=blend_confidence,
            individual_signals=signals,
            signal_contributions=contributions,
            metadata=metadata,
        )

    def _filter_signals_by_confidence(
        self, signals: Dict[str, SignalResult]
    ) -> Dict[str, SignalResult]:
        """Filter out signals below minimum confidence threshold."""
        return {
            name: signal
            for name, signal in signals.items()
            if signal.confidence >= self.config.min_signal_confidence
        }

    def _calculate_signal_weights(
        self, signals: Dict[str, SignalResult]
    ) -> Dict[str, float]:
        """Calculate weights for each signal based on config and performance."""

        weights = {}
        total_weight = 0.0

        for signal_name, signal in signals.items():
            # Base weight from signal configuration
            base_weight = signal.config.weight if hasattr(signal, "config") else 1.0

            # Risk budget weight from configuration
            risk_budget = self.config.risk_budget_per_signal.get(signal_name, 1.0)

            # Confidence weight
            confidence_weight = signal.confidence

            # Historical performance weight (if available)
            performance_weight = self._get_performance_weight(signal_name)

            # Combined weight
            combined_weight = (
                base_weight * risk_budget * confidence_weight * performance_weight
            )
            weights[signal_name] = combined_weight
            total_weight += combined_weight

        # Normalize weights to sum to 1
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights if all weights are zero
            equal_weight = 1.0 / len(signals) if signals else 0.0
            weights = {name: equal_weight for name in signals.keys()}

        return weights

    def _resolve_signal_conflicts(
        self, signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Resolve conflicts between signals and compute blended position."""

        if not signals:
            return 0.0, 0.0

        if self.config.conflict_resolution == ConflictResolution.WEIGHTED_AVERAGE:
            return self._weighted_average_blend(signals, weights)

        elif self.config.conflict_resolution == ConflictResolution.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_blend(signals, weights)

        elif self.config.conflict_resolution == ConflictResolution.STRONGEST_SIGNAL:
            return self._strongest_signal_blend(signals)

        elif self.config.conflict_resolution == ConflictResolution.RISK_PARITY:
            return self._risk_parity_blend(signals, weights)

        else:
            # Default to confidence weighted
            return self._confidence_weighted_blend(signals, weights)

    def _weighted_average_blend(
        self, signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Simple weighted average of signal values."""

        weighted_sum = sum(
            signal.value * weights[name] for name, signal in signals.items()
        )

        weighted_confidence = sum(
            signal.confidence * weights[name] for name, signal in signals.items()
        )

        return weighted_sum, weighted_confidence

    def _confidence_weighted_blend(
        self, signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Blend signals with additional confidence weighting."""

        total_confidence_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0

        for name, signal in signals.items():
            confidence_weight = signal.confidence * weights[name]
            weighted_sum += signal.value * confidence_weight
            confidence_sum += signal.confidence * confidence_weight
            total_confidence_weight += confidence_weight

        if total_confidence_weight == 0:
            return 0.0, 0.0

        final_position = weighted_sum / total_confidence_weight
        final_confidence = confidence_sum / total_confidence_weight

        return final_position, final_confidence

    def _strongest_signal_blend(
        self, signals: Dict[str, SignalResult]
    ) -> Tuple[float, float]:
        """Use the strongest signal by absolute value × confidence."""

        strongest_signal = max(
            signals.values(), key=lambda s: abs(s.value) * s.confidence
        )

        return strongest_signal.value, strongest_signal.confidence

    def _risk_parity_blend(
        self, signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Risk parity approach - scale signals by their volatility."""

        # For now, use simple risk parity based on signal magnitude
        # In a full implementation, this would use historical signal volatilities

        signal_vols = {}
        for name, signal in signals.items():
            # Estimate signal "volatility" from metadata if available
            vol = 1.0  # Default
            if signal.metadata and "realized_vol" in signal.metadata:
                vol = signal.metadata["realized_vol"] or 1.0
            signal_vols[name] = vol

        # Risk parity weights (inverse volatility)
        risk_weights = {name: 1.0 / vol for name, vol in signal_vols.items()}
        total_risk_weight = sum(risk_weights.values())

        if total_risk_weight > 0:
            risk_weights = {
                name: w / total_risk_weight for name, w in risk_weights.items()
            }

        # Combine with original weights
        combined_weights = {
            name: weights[name] * risk_weights[name] for name in signals.keys()
        }

        return self._weighted_average_blend(signals, combined_weights)

    def _calculate_signal_contributions(
        self,
        signals: Dict[str, SignalResult],
        weights: Dict[str, float],
        final_position: float,
    ) -> Dict[str, float]:
        """Calculate how much each signal contributed to final position."""

        contributions = {}

        if final_position == 0:
            return {name: 0.0 for name in signals.keys()}

        for name, signal in signals.items():
            # Contribution = (signal_value * weight) / final_position
            signal_contribution = signal.value * weights[name]
            contribution_ratio = (
                signal_contribution / final_position if final_position != 0 else 0
            )
            contributions[name] = contribution_ratio

        return contributions

    def _apply_position_constraints(self, position: float) -> float:
        """Apply position sizing constraints."""

        # Bound to maximum position size
        bounded_position = max(
            -self.config.max_position_size, min(self.config.max_position_size, position)
        )

        return bounded_position

    def _get_performance_weight(self, signal_name: str) -> float:
        """Get historical performance weight for a signal."""

        if signal_name not in self.performance_history:
            return 1.0  # Default weight for new signals

        history = self.performance_history[signal_name]
        if len(history) < 5:  # Need minimum history
            return 1.0

        # Calculate exponentially weighted performance
        weights = [self.config.decay_factor**i for i in range(len(history))]
        weighted_perf = sum(p * w for p, w in zip(history, weights)) / sum(weights)

        # Convert to weight (clamp between 0.1 and 2.0)
        performance_weight = max(0.1, min(2.0, 1.0 + weighted_perf))

        return performance_weight

    def _update_signal_history(self, signals: Dict[str, SignalResult]):
        """Update signal history for correlation and performance tracking."""

        for name, signal in signals.items():
            if name not in self.signal_history:
                self.signal_history[name] = []

            self.signal_history[name].append(signal)

            # Keep only recent history
            if len(self.signal_history[name]) > self.config.correlation_lookback:
                self.signal_history[name] = self.signal_history[name][
                    -self.config.correlation_lookback :
                ]

    def _get_correlation_metadata(self) -> Dict[str, Any]:
        """Get correlation metadata for signals."""

        if len(self.signal_history) < 2:
            return {"correlations": "insufficient_data"}

        # Calculate pairwise correlations
        correlations = {}
        signal_names = list(self.signal_history.keys())

        for i, name1 in enumerate(signal_names):
            for name2 in signal_names[i + 1 :]:
                if (
                    len(self.signal_history[name1]) >= 10
                    and len(self.signal_history[name2]) >= 10
                ):

                    values1 = [s.value for s in self.signal_history[name1][-10:]]
                    values2 = [s.value for s in self.signal_history[name2][-10:]]

                    if len(values1) == len(values2):
                        corr = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(corr):
                            correlations[f"{name1}_{name2}"] = float(corr)

        return {"correlations": correlations}

    def update_performance(self, signal_name: str, performance: float):
        """Update performance history for a signal."""

        if signal_name not in self.performance_history:
            self.performance_history[signal_name] = []

        self.performance_history[signal_name].append(performance)

        # Keep only recent performance history
        if len(self.performance_history[signal_name]) > 50:
            self.performance_history[signal_name] = self.performance_history[
                signal_name
            ][-50:]

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about signal performance and correlations."""

        stats = {
            "signal_count": len(self.signal_history),
            "signals_tracked": list(self.signal_history.keys()),
            "correlation_metadata": self._get_correlation_metadata(),
            "performance_summary": {},
        }

        # Performance summary for each signal
        for name, history in self.performance_history.items():
            if len(history) >= 5:
                stats["performance_summary"][name] = {
                    "avg_performance": np.mean(history),
                    "performance_volatility": np.std(history),
                    "recent_performance": (
                        np.mean(history[-5:]) if len(history) >= 5 else None
                    ),
                    "sample_count": len(history),
                }

        return stats
