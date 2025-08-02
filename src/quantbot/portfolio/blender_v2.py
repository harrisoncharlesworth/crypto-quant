"""
Enhanced Portfolio Blender v2 for 12-signal crypto portfolio management.

Handles market-neutral vs directional signal buckets, risk parity allocation,
dynamic correlation tracking, and comprehensive risk management across all signal types.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..signals.base import SignalResult


class SignalType(Enum):
    """Signal classification for bucketing and risk management."""
    DIRECTIONAL = "directional"          # Symmetric long/short signals
    MARKET_NEUTRAL = "market_neutral"    # Beta-neutral strategies
    OVERLAY = "overlay"                  # Filters/regime indicators
    

class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMAL = "kelly_optimal"
    PERFORMANCE_WEIGHTED = "performance_weighted"


@dataclass
class SignalTypeConfig:
    """Configuration for each signal type bucket."""
    max_allocation: float = 0.2  # Max allocation per signal type
    volatility_target: float = 0.15  # Annualized volatility target
    turnover_limit: float = 0.15  # Monthly turnover limit
    correlation_threshold: float = 0.7  # Max correlation before position reduction


@dataclass 
class RiskLimits:
    """Comprehensive risk limits for portfolio."""
    max_net_exposure: float = 0.30  # Maximum net directional exposure
    max_gross_leverage: float = 3.0  # Maximum gross leverage
    min_leverage: float = 1.0  # Minimum leverage floor
    max_single_position: float = 0.10  # Max position per individual signal
    max_correlated_exposure: float = 0.25  # Max exposure to highly correlated signals


@dataclass
class BlenderConfigV2:
    """Enhanced configuration for Portfolio Blender v2."""
    
    # Core allocation settings
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    min_signal_confidence: float = 0.3
    correlation_lookback: int = 100
    
    # Signal type configurations
    signal_type_configs: Dict[SignalType, SignalTypeConfig] = field(default_factory=lambda: {
        SignalType.DIRECTIONAL: SignalTypeConfig(max_allocation=0.15, volatility_target=0.10),
        SignalType.MARKET_NEUTRAL: SignalTypeConfig(max_allocation=0.20, volatility_target=0.08),
        SignalType.OVERLAY: SignalTypeConfig(max_allocation=0.05, volatility_target=0.05)
    })
    
    # Risk management
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    
    # Signal classification
    signal_classifications: Dict[str, SignalType] = field(default_factory=lambda: {
        # Directional signals (2-S)
        "time_series_momentum": SignalType.DIRECTIONAL,
        "donchian_breakout": SignalType.DIRECTIONAL,
        "short_term_mean_reversion": SignalType.DIRECTIONAL,
        "oi_price_divergence": SignalType.DIRECTIONAL,
        "delta_skew_whipsaw": SignalType.DIRECTIONAL,
        
        # Market-neutral signals (M-N)
        "perp_funding_carry": SignalType.MARKET_NEUTRAL,
        "alt_btc_cross_sectional": SignalType.MARKET_NEUTRAL,
        "cash_carry_basis": SignalType.MARKET_NEUTRAL,
        "cross_exchange_funding": SignalType.MARKET_NEUTRAL,
        "options_vol_risk_premium": SignalType.MARKET_NEUTRAL,
        
        # Overlay/filter signals
        "stablecoin_supply_ratio": SignalType.OVERLAY,
        "mvrv_zscore": SignalType.OVERLAY
    })
    
    # Performance tracking
    decay_factor: float = 0.95
    performance_lookback: int = 50


@dataclass
class PortfolioSnapshot:
    """Real-time portfolio snapshot for risk monitoring."""
    timestamp: datetime
    net_exposure: float
    gross_leverage: float
    signal_exposures: Dict[str, float]
    type_exposures: Dict[SignalType, float]
    correlation_risks: Dict[str, float]
    kelly_fractions: Dict[str, float]
    risk_budget_utilization: Dict[SignalType, float]


@dataclass
class BlendedSignalV2:
    """Enhanced result from 12-signal blending."""
    symbol: str
    timestamp: datetime
    
    # Final positions
    final_position: float  # Net position [-1, +1]
    gross_exposure: float  # Total gross exposure
    confidence: float
    
    # Signal breakdown
    directional_position: float
    market_neutral_position: float
    overlay_adjustments: Dict[str, float]
    
    # Individual signals
    individual_signals: Dict[str, SignalResult]
    signal_contributions: Dict[str, float]
    signal_weights: Dict[str, float]
    
    # Risk metrics
    portfolio_snapshot: PortfolioSnapshot
    risk_metrics: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any]


class PortfolioBlenderV2:
    """
    Enhanced Portfolio Blender v2 for 12-signal crypto portfolio management.
    
    Features:
    - Separate buckets for directional vs market-neutral signals
    - Risk parity allocation with Kelly optimization
    - Real-time correlation tracking and risk monitoring
    - Signal conflict resolution across all 12 signals
    - Comprehensive performance attribution
    - Dynamic leverage and exposure management
    """
    
    def __init__(self, config: BlenderConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Historical tracking
        self.signal_history: Dict[str, List[SignalResult]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        # Risk monitoring
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        self.risk_alerts: List[Dict[str, Any]] = []
        
        self.logger.info("Initialized PortfolioBlenderV2 with 12-signal support")
    
    def blend_signals(
        self, 
        signals: Dict[str, SignalResult], 
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None
    ) -> BlendedSignalV2:
        """
        Enhanced signal blending with 12-signal support and risk management.
        
        Args:
            signals: Dictionary of {signal_name: SignalResult}
            symbol: Trading symbol
            market_data: Optional market data for risk calculations
            
        Returns:
            BlendedSignalV2 with comprehensive portfolio information
        """
        
        # Filter signals by confidence
        filtered_signals = self._filter_signals_by_confidence(signals)
        
        if not filtered_signals:
            return self._create_empty_result(symbol, signals, "no_confident_signals")
        
        # Classify signals by type
        signal_buckets = self._classify_signals_by_type(filtered_signals)
        
        # Calculate correlation-adjusted weights
        signal_weights = self._calculate_enhanced_weights(filtered_signals, signal_buckets)
        
        # Blend signals within each bucket
        directional_position = self._blend_directional_signals(
            signal_buckets.get(SignalType.DIRECTIONAL, {}), signal_weights
        )
        
        market_neutral_position = self._blend_market_neutral_signals(
            signal_buckets.get(SignalType.MARKET_NEUTRAL, {}), signal_weights
        )
        
        # Apply overlay adjustments
        overlay_adjustments = self._apply_overlay_signals(
            signal_buckets.get(SignalType.OVERLAY, {}), signal_weights
        )
        
        # Combine positions with overlay adjustments
        net_position = self._combine_positions(
            directional_position, market_neutral_position, overlay_adjustments
        )
        
        # Apply risk limits and constraints
        final_position, gross_exposure = self._apply_risk_limits(
            net_position, directional_position, market_neutral_position, signal_weights
        )
        
        # Calculate portfolio metrics
        portfolio_snapshot = self._create_portfolio_snapshot(
            filtered_signals, signal_weights, final_position, gross_exposure
        )
        
        # Update tracking data
        self._update_tracking_data(filtered_signals, portfolio_snapshot)
        
        # Calculate comprehensive confidence
        blend_confidence = self._calculate_blended_confidence(
            filtered_signals, signal_weights, portfolio_snapshot
        )
        
        # Calculate signal contributions
        contributions = self._calculate_signal_contributions(
            filtered_signals, signal_weights, final_position
        )
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(
            portfolio_snapshot, filtered_signals, market_data
        )
        
        # Comprehensive metadata
        metadata = self._build_metadata(
            signal_buckets, signal_weights, overlay_adjustments, risk_metrics
        )
        
        return BlendedSignalV2(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            final_position=final_position,
            gross_exposure=gross_exposure,
            confidence=blend_confidence,
            directional_position=directional_position,
            market_neutral_position=market_neutral_position,
            overlay_adjustments=overlay_adjustments,
            individual_signals=signals,
            signal_contributions=contributions,
            signal_weights=signal_weights,
            portfolio_snapshot=portfolio_snapshot,
            risk_metrics=risk_metrics,
            metadata=metadata
        )
    
    def _filter_signals_by_confidence(
        self, signals: Dict[str, SignalResult]
    ) -> Dict[str, SignalResult]:
        """Filter signals by minimum confidence threshold."""
        return {
            name: signal
            for name, signal in signals.items()
            if signal.confidence >= self.config.min_signal_confidence
        }
    
    def _classify_signals_by_type(
        self, signals: Dict[str, SignalResult]
    ) -> Dict[SignalType, Dict[str, SignalResult]]:
        """Classify signals into directional, market-neutral, and overlay buckets."""
        buckets = {signal_type: {} for signal_type in SignalType}
        
        for name, signal in signals.items():
            signal_type = self.config.signal_classifications.get(name, SignalType.DIRECTIONAL)
            buckets[signal_type][name] = signal
        
        return buckets
    
    def _calculate_enhanced_weights(
        self, 
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Calculate correlation-adjusted weights using the selected allocation method."""
        
        if self.config.allocation_method == AllocationMethod.RISK_PARITY:
            return self._calculate_risk_parity_weights(signals, signal_buckets)
        elif self.config.allocation_method == AllocationMethod.KELLY_OPTIMAL:
            return self._calculate_kelly_weights(signals, signal_buckets)
        elif self.config.allocation_method == AllocationMethod.PERFORMANCE_WEIGHTED:
            return self._calculate_performance_weights(signals, signal_buckets)
        elif self.config.allocation_method == AllocationMethod.CONFIDENCE_WEIGHTED:
            return self._calculate_confidence_weights(signals, signal_buckets)
        else:  # EQUAL_WEIGHT
            return self._calculate_equal_weights(signals, signal_buckets)
    
    def _calculate_risk_parity_weights(
        self,
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Calculate risk parity weights within each signal type bucket."""
        weights = {}
        
        for signal_type, bucket_signals in signal_buckets.items():
            if not bucket_signals:
                continue
                
            # Get volatility estimates for signals in this bucket
            bucket_vols = {}
            for name, signal in bucket_signals.items():
                vol = self._get_signal_volatility(name, signal)
                bucket_vols[name] = vol
            
            # Risk parity within bucket (inverse volatility)
            bucket_risk_weights = {name: 1.0 / vol for name, vol in bucket_vols.items()}
            total_risk_weight = sum(bucket_risk_weights.values())
            
            # Normalize and apply bucket allocation
            bucket_allocation = self.config.signal_type_configs[signal_type].max_allocation
            
            for name in bucket_signals.keys():
                normalized_weight = bucket_risk_weights[name] / total_risk_weight
                weights[name] = normalized_weight * bucket_allocation
        
        return weights
    
    def _calculate_kelly_weights(
        self,
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Calculate Kelly-optimal weights based on expected returns and volatilities."""
        weights = {}
        
        for signal_type, bucket_signals in signal_buckets.items():
            if not bucket_signals:
                continue
            
            for name, signal in bucket_signals.items():
                # Get expected return and volatility
                expected_return = self._get_expected_return(name, signal)
                volatility = self._get_signal_volatility(name, signal)
                
                # Kelly fraction: f = μ / σ²
                if volatility > 0:
                    kelly_fraction = expected_return / (volatility ** 2)
                    # Clamp between min and max leverage
                    kelly_fraction = max(
                        self.config.risk_limits.min_leverage / len(bucket_signals),
                        min(self.config.risk_limits.max_gross_leverage / len(bucket_signals), kelly_fraction)
                    )
                else:
                    kelly_fraction = self.config.risk_limits.min_leverage / len(bucket_signals)
                
                weights[name] = kelly_fraction * signal.confidence
        
        return self._normalize_weights(weights)
    
    def _calculate_performance_weights(
        self,
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Weight signals based on historical performance."""
        weights = {}
        
        for name, signal in signals.items():
            performance_weight = self._get_performance_weight(name)
            confidence_weight = signal.confidence
            weights[name] = performance_weight * confidence_weight
        
        return self._normalize_weights(weights)
    
    def _calculate_confidence_weights(
        self,
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Weight signals based on confidence scores."""
        weights = {name: signal.confidence for name, signal in signals.items()}
        return self._normalize_weights(weights)
    
    def _calculate_equal_weights(
        self,
        signals: Dict[str, SignalResult],
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]]
    ) -> Dict[str, float]:
        """Equal weight allocation within signal type buckets."""
        weights = {}
        
        for signal_type, bucket_signals in signal_buckets.items():
            if not bucket_signals:
                continue
            
            bucket_allocation = self.config.signal_type_configs[signal_type].max_allocation
            signal_weight = bucket_allocation / len(bucket_signals)
            
            for name in bucket_signals.keys():
                weights[name] = signal_weight
        
        return weights
    
    def _blend_directional_signals(
        self, directional_signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> float:
        """Blend directional signals with conflict resolution."""
        if not directional_signals:
            return 0.0
        
        # Check for conflicts (opposing signals)
        positive_signals = {name: sig for name, sig in directional_signals.items() if sig.value > 0}
        negative_signals = {name: sig for name, sig in directional_signals.items() if sig.value < 0}
        
        if positive_signals and negative_signals:
            # Resolve conflict using confidence and weights
            pos_strength = sum(sig.value * sig.confidence * weights.get(name, 0) 
                             for name, sig in positive_signals.items())
            neg_strength = sum(abs(sig.value) * sig.confidence * weights.get(name, 0) 
                             for name, sig in negative_signals.items())
            
            # Net directional position
            net_position = pos_strength - neg_strength
        else:
            # No conflict, simple weighted average
            net_position = sum(sig.value * weights.get(name, 0) 
                             for name, sig in directional_signals.items())
        
        return net_position
    
    def _blend_market_neutral_signals(
        self, market_neutral_signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> float:
        """Blend market-neutral signals (should sum to near zero net exposure)."""
        if not market_neutral_signals:
            return 0.0
        
        # Market-neutral signals contribute to gross exposure but should net to ~0
        weighted_sum = sum(sig.value * weights.get(name, 0) 
                          for name, sig in market_neutral_signals.items())
        
        return weighted_sum
    
    def _apply_overlay_signals(
        self, overlay_signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply overlay signals as position adjustments/filters."""
        adjustments = {}
        
        for name, signal in overlay_signals.items():
            if name == "stablecoin_supply_ratio":
                # SSR as market sentiment overlay
                adjustments["ssr_adjustment"] = signal.value * weights.get(name, 0)
            elif name == "mvrv_zscore":
                # MVRV as regime filter
                adjustments["mvrv_filter"] = signal.value * weights.get(name, 0)
        
        return adjustments
    
    def _combine_positions(
        self, 
        directional_pos: float, 
        market_neutral_pos: float, 
        overlay_adjustments: Dict[str, float]
    ) -> float:
        """Combine positions from different buckets with overlay adjustments."""
        
        base_position = directional_pos + market_neutral_pos
        
        # Apply overlay adjustments
        for adjustment_name, adjustment_value in overlay_adjustments.items():
            if adjustment_name == "ssr_adjustment":
                # SSR adjusts overall position sizing
                base_position *= (1.0 + adjustment_value * 0.1)  # Max 10% adjustment
            elif adjustment_name == "mvrv_filter":
                # MVRV can reduce position in extreme conditions
                if adjustment_value < -0.5:  # Oversold - reduce shorts
                    base_position = max(0, base_position)
                elif adjustment_value > 0.5:  # Overbought - reduce longs
                    base_position = min(0, base_position)
        
        return base_position
    
    def _apply_risk_limits(
        self,
        net_position: float,
        directional_pos: float,
        market_neutral_pos: float,
        signal_weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Apply comprehensive risk limits to final position."""
        
        # Net exposure limit
        final_position = max(
            -self.config.risk_limits.max_net_exposure,
            min(self.config.risk_limits.max_net_exposure, net_position)
        )
        
        # Calculate gross exposure
        gross_exposure = abs(directional_pos) + abs(market_neutral_pos)
        
        # Gross leverage limit
        if gross_exposure > self.config.risk_limits.max_gross_leverage:
            scale_factor = self.config.risk_limits.max_gross_leverage / gross_exposure
            final_position *= scale_factor
            gross_exposure = self.config.risk_limits.max_gross_leverage
        
        # Minimum leverage floor
        if gross_exposure < self.config.risk_limits.min_leverage:
            if gross_exposure > 0:
                scale_factor = self.config.risk_limits.min_leverage / gross_exposure
                final_position *= scale_factor
                gross_exposure = self.config.risk_limits.min_leverage
        
        return final_position, gross_exposure
    
    def _create_portfolio_snapshot(
        self,
        signals: Dict[str, SignalResult],
        weights: Dict[str, float],
        final_position: float,
        gross_exposure: float
    ) -> PortfolioSnapshot:
        """Create real-time portfolio snapshot for risk monitoring."""
        
        # Calculate exposures by signal
        signal_exposures = {
            name: signal.value * weights.get(name, 0) for name, signal in signals.items()
        }
        
        # Calculate exposures by type
        type_exposures = {signal_type: 0.0 for signal_type in SignalType}
        for name, exposure in signal_exposures.items():
            signal_type = self.config.signal_classifications.get(name, SignalType.DIRECTIONAL)
            type_exposures[signal_type] += abs(exposure)
        
        # Calculate correlation risks
        correlation_risks = self._calculate_correlation_risks(signals, weights)
        
        # Calculate Kelly fractions
        kelly_fractions = {
            name: self._calculate_kelly_fraction(name, signal) 
            for name, signal in signals.items()
        }
        
        # Risk budget utilization
        risk_budget_utilization = {
            signal_type: type_exposures[signal_type] / config.max_allocation
            for signal_type, config in self.config.signal_type_configs.items()
        }
        
        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            net_exposure=final_position,
            gross_leverage=gross_exposure,
            signal_exposures=signal_exposures,
            type_exposures=type_exposures,
            correlation_risks=correlation_risks,
            kelly_fractions=kelly_fractions,
            risk_budget_utilization=risk_budget_utilization
        )
    
    def _calculate_blended_confidence(
        self,
        signals: Dict[str, SignalResult],
        weights: Dict[str, float],
        portfolio_snapshot: PortfolioSnapshot
    ) -> float:
        """Calculate blended confidence incorporating risk factors."""
        
        # Base confidence from signals
        weighted_confidence = sum(
            signal.confidence * weights.get(name, 0) for name, signal in signals.items()
        )
        
        # Risk adjustment factors
        correlation_penalty = max(portfolio_snapshot.correlation_risks.values()) if portfolio_snapshot.correlation_risks else 0
        concentration_penalty = max(portfolio_snapshot.risk_budget_utilization.values()) - 1.0
        
        # Adjust confidence based on risk factors
        risk_adjusted_confidence = weighted_confidence * (1.0 - correlation_penalty * 0.2) * (1.0 - max(0, concentration_penalty * 0.3))
        
        return max(0.0, min(1.0, risk_adjusted_confidence))
    
    def _calculate_signal_contributions(
        self,
        signals: Dict[str, SignalResult],
        weights: Dict[str, float],
        final_position: float
    ) -> Dict[str, float]:
        """Calculate how much each signal contributed to final position."""
        contributions = {}
        
        if abs(final_position) < 1e-10:
            return {name: 0.0 for name in signals.keys()}
        
        for name, signal in signals.items():
            signal_contribution = signal.value * weights.get(name, 0)
            contribution_ratio = signal_contribution / final_position if final_position != 0 else 0
            contributions[name] = contribution_ratio
        
        return contributions
    
    def _calculate_risk_metrics(
        self,
        portfolio_snapshot: PortfolioSnapshot,
        signals: Dict[str, SignalResult],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        
        risk_metrics = {
            "net_exposure": portfolio_snapshot.net_exposure,
            "gross_leverage": portfolio_snapshot.gross_leverage,
            "leverage_utilization": portfolio_snapshot.gross_leverage / self.config.risk_limits.max_gross_leverage,
            "max_correlation": max(portfolio_snapshot.correlation_risks.values()) if portfolio_snapshot.correlation_risks else 0,
            "signal_concentration": max(portfolio_snapshot.risk_budget_utilization.values()),
            "signals_active": len([s for s in signals.values() if abs(s.value) > 0.01]),
            "confidence_weighted_exposure": sum(
                abs(signal.value * signal.confidence) for signal in signals.values()
            )
        }
        
        # Add market context if available
        if market_data:
            risk_metrics["market_volatility"] = market_data.get("realized_vol", 0)
            risk_metrics["market_momentum"] = market_data.get("momentum", 0)
        
        return risk_metrics
    
    # Helper methods
    def _get_signal_volatility(self, signal_name: str, signal: SignalResult) -> float:
        """Get volatility estimate for a signal."""
        if signal_name in self.volatility_estimates:
            return self.volatility_estimates[signal_name]
        
        # Extract from metadata if available
        if signal.metadata and "realized_vol" in signal.metadata:
            vol = signal.metadata["realized_vol"] or 0.15
        else:
            # Default volatility based on signal type
            signal_type = self.config.signal_classifications.get(signal_name, SignalType.DIRECTIONAL)
            vol = self.config.signal_type_configs[signal_type].volatility_target
        
        self.volatility_estimates[signal_name] = vol
        return vol
    
    def _get_expected_return(self, signal_name: str, signal: SignalResult) -> float:
        """Get expected return for Kelly calculation."""
        if signal_name in self.performance_history and len(self.performance_history[signal_name]) >= 10:
            recent_returns = self.performance_history[signal_name][-10:]
            return float(np.mean(recent_returns))
        return 0.02  # Default 2% expected return
    
    def _get_performance_weight(self, signal_name: str) -> float:
        """Get performance-based weight for a signal."""
        if signal_name not in self.performance_history or len(self.performance_history[signal_name]) < 5:
            return 1.0
        
        history = self.performance_history[signal_name]
        # Exponentially weighted performance
        weights = [self.config.decay_factor**i for i in range(len(history))]
        weighted_perf = sum(p * w for p, w in zip(history, weights)) / sum(weights)
        
        return max(0.1, min(2.0, 1.0 + weighted_perf))
    
    def _calculate_correlation_risks(
        self, signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate correlation risk between signals."""
        correlation_risks = {}
        
        if len(signals) < 2:
            return correlation_risks
        
        signal_names = list(signals.keys())
        for i, name1 in enumerate(signal_names):
            for name2 in signal_names[i+1:]:
                corr = self._get_signal_correlation(name1, name2)
                if abs(corr) > self.config.signal_type_configs[SignalType.DIRECTIONAL].correlation_threshold:
                    combined_weight = weights.get(name1, 0) + weights.get(name2, 0)
                    correlation_risks[f"{name1}_{name2}"] = abs(corr) * combined_weight
        
        return correlation_risks
    
    def _get_signal_correlation(self, signal1: str, signal2: str) -> float:
        """Get correlation between two signals."""
        if (signal1 not in self.signal_history or signal2 not in self.signal_history or
            len(self.signal_history[signal1]) < 10 or len(self.signal_history[signal2]) < 10):
            return 0.0
        
        values1 = [s.value for s in self.signal_history[signal1][-20:]]
        values2 = [s.value for s in self.signal_history[signal2][-20:]]
        
        if len(values1) == len(values2) and len(values1) >= 10:
            corr = np.corrcoef(values1, values2)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        
        return 0.0
    
    def _calculate_kelly_fraction(self, signal_name: str, signal: SignalResult) -> float:
        """Calculate Kelly fraction for individual signal."""
        expected_return = self._get_expected_return(signal_name, signal)
        volatility = self._get_signal_volatility(signal_name, signal)
        
        if volatility > 0:
            kelly = expected_return / (volatility ** 2)
            return max(0.01, min(0.25, kelly))  # Clamp between 1% and 25%
        return 0.01
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {name: weight / total_weight for name, weight in weights.items()}
        return {name: 1.0 / len(weights) for name in weights.keys()} if weights else {}
    
    def _update_tracking_data(
        self, signals: Dict[str, SignalResult], portfolio_snapshot: PortfolioSnapshot
    ):
        """Update historical tracking data."""
        # Update signal history
        for name, signal in signals.items():
            if name not in self.signal_history:
                self.signal_history[name] = []
            
            self.signal_history[name].append(signal)
            if len(self.signal_history[name]) > self.config.correlation_lookback:
                self.signal_history[name] = self.signal_history[name][-self.config.correlation_lookback:]
        
        # Update portfolio snapshots
        self.portfolio_snapshots.append(portfolio_snapshot)
        if len(self.portfolio_snapshots) > 1000:  # Keep last 1000 snapshots
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
    
    def _create_empty_result(
        self, symbol: str, signals: Dict[str, SignalResult], reason: str
    ) -> BlendedSignalV2:
        """Create empty result when no signals are available."""
        empty_snapshot = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            net_exposure=0.0,
            gross_leverage=0.0,
            signal_exposures={},
            type_exposures={t: 0.0 for t in SignalType},
            correlation_risks={},
            kelly_fractions={},
            risk_budget_utilization={t: 0.0 for t in SignalType}
        )
        
        return BlendedSignalV2(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            final_position=0.0,
            gross_exposure=0.0,
            confidence=0.0,
            directional_position=0.0,
            market_neutral_position=0.0,
            overlay_adjustments={},
            individual_signals=signals,
            signal_contributions={},
            signal_weights={},
            portfolio_snapshot=empty_snapshot,
            risk_metrics={"reason": reason},
            metadata={"empty_result": True, "reason": reason}
        )
    
    def _build_metadata(
        self,
        signal_buckets: Dict[SignalType, Dict[str, SignalResult]],
        signal_weights: Dict[str, float],
        overlay_adjustments: Dict[str, float],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive metadata for the blended result."""
        return {
            "allocation_method": self.config.allocation_method.value,
            "signal_counts": {
                signal_type.value: len(signals) for signal_type, signals in signal_buckets.items()
            },
            "weight_distribution": {
                signal_type.value: sum(
                    signal_weights.get(name, 0) for name in signals.keys()
                ) for signal_type, signals in signal_buckets.items()
            },
            "overlay_adjustments": overlay_adjustments,
            "risk_budget_utilization": risk_metrics.get("signal_concentration", 0),
            "correlation_max": risk_metrics.get("max_correlation", 0),
            "signals_active": risk_metrics.get("signals_active", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Public methods for monitoring and analysis
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics."""
        if not self.portfolio_snapshots:
            return {"error": "No portfolio history available"}
        
        recent_snapshots = self.portfolio_snapshots[-100:]  # Last 100 snapshots
        
        return {
            "summary": {
                "snapshots_count": len(self.portfolio_snapshots),
                "signals_tracked": len(self.signal_history),
                "avg_net_exposure": np.mean([s.net_exposure for s in recent_snapshots]),
                "avg_gross_leverage": np.mean([s.gross_leverage for s in recent_snapshots]),
                "max_leverage_used": max([s.gross_leverage for s in recent_snapshots]),
            },
            "risk_metrics": {
                "max_correlation_risk": max([
                    max(s.correlation_risks.values()) if s.correlation_risks else 0 
                    for s in recent_snapshots
                ]),
                "avg_signal_concentration": np.mean([
                    max(s.risk_budget_utilization.values()) if s.risk_budget_utilization else 0
                    for s in recent_snapshots
                ])
            },
            "performance_summary": {
                name: {
                    "sample_count": len(history),
                    "avg_performance": np.mean(history) if len(history) >= 5 else None,
                    "volatility": np.std(history) if len(history) >= 5 else None
                } for name, history in self.performance_history.items()
            }
        }
    
    def update_performance(self, signal_name: str, performance: float):
        """Update performance history for a signal."""
        if signal_name not in self.performance_history:
            self.performance_history[signal_name] = []
        
        self.performance_history[signal_name].append(performance)
        
        # Keep only recent performance history
        if len(self.performance_history[signal_name]) > self.config.performance_lookback:
            self.performance_history[signal_name] = self.performance_history[signal_name][-self.config.performance_lookback:]
    
    def check_risk_limits(self) -> List[Dict[str, Any]]:
        """Check current risk limits and return any violations."""
        if not self.portfolio_snapshots:
            return []
        
        latest_snapshot = self.portfolio_snapshots[-1]
        violations = []
        
        # Net exposure check
        if abs(latest_snapshot.net_exposure) > self.config.risk_limits.max_net_exposure:
            violations.append({
                "type": "net_exposure_violation",
                "current": latest_snapshot.net_exposure,
                "limit": self.config.risk_limits.max_net_exposure,
                "severity": "high"
            })
        
        # Gross leverage check
        if latest_snapshot.gross_leverage > self.config.risk_limits.max_gross_leverage:
            violations.append({
                "type": "gross_leverage_violation", 
                "current": latest_snapshot.gross_leverage,
                "limit": self.config.risk_limits.max_gross_leverage,
                "severity": "high"
            })
        
        # Correlation risk check
        if latest_snapshot.correlation_risks:
            max_corr_risk = max(latest_snapshot.correlation_risks.values())
            if max_corr_risk > 0.5:  # 50% correlation risk threshold
                violations.append({
                    "type": "correlation_risk",
                    "current": max_corr_risk,
                    "limit": 0.5,
                    "severity": "medium"
                })
        
        return violations
