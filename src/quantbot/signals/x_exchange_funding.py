import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class XExchangeFundingConfig(SignalConfig):
    """Configuration for cross-exchange funding dispersion signal."""

    # Dispersion thresholds (in basis points)
    entry_threshold_bps: float = 20.0  # Enter when dispersion >20 bps
    exit_threshold_bps: float = 5.0  # Exit when dispersion <5 bps

    # Inventory shuffling costs
    inventory_cost_bps: float = 2.0  # Budget 2 bps for inventory transfer

    # Position management
    max_allocation: float = 0.20  # 20% max allocation per signal
    max_position_hours: int = 48  # Max 48h position (convergence window)

    # Risk management
    reversal_stop_bps: float = 30.0  # Stop if spread moves against us by 30 bps
    latency_buffer_ms: int = 500  # Account for execution latency
    min_liquidity_threshold: float = 1000000  # Min $1M liquidity per side

    # Exchange configuration
    supported_exchanges: Optional[List[str]] = None
    primary_venue: str = "binance"  # Primary venue for hedging

    # Confidence parameters
    confidence_multiplier: float = 2.0
    persistence_weight: float = 0.3  # Weight for dispersion persistence
    volume_weight: float = 0.2  # Weight for venue volume ratio

    def __post_init__(self):
        if self.supported_exchanges is None:
            self.supported_exchanges = ["binance", "bybit", "okx"]


class XExchangeFundingDispersionSignal(SignalBase):
    """
    Cross-Exchange Funding Dispersion Arbitrage Signal - Market-Neutral Strategy

    Strategy: Long funding-cheap venue, short funding-expensive venue
    Classification: Market-Neutral (M-N) with low-latency inventory shuffling

    Based on Kaiko research: Funding dispersion >20 bps converges in 24-48h window.
    Requires sophisticated inventory management and cross-venue execution.

    Key Features:
    - Multi-venue funding rate tracking (Binance, Bybit, OKX)
    - Dispersion-based entry/exit thresholds (20 bps / 5 bps)
    - Inventory shuffling cost modeling (2 bps budget)
    - Market-neutral delta exposure across venues
    - Low-latency execution considerations
    - Convergence timing prediction (24-48h window)
    - Risk management for spread reversals and execution delays

    Risk Management:
    - Inventory transfer cost allocation (2 bps)
    - Spread reversal detection (30 bps stop)
    - Exchange connectivity monitoring
    - Position auto-close mechanisms
    - Liquidity threshold enforcement
    """

    def __init__(self, config: XExchangeFundingConfig):
        super().__init__(config)
        self.config: XExchangeFundingConfig = config

        # Multi-venue data tracking
        self.funding_rates: Dict[str, Dict[str, float]] = (
            {}
        )  # {symbol: {exchange: rate}}
        self.funding_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}

        # Position tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}  # {symbol: position_info}
        self.position_entry_times: Dict[str, datetime] = {}

        # Exchange connectivity status
        self.exchange_status: Dict[str, bool] = {
            exchange: True for exchange in (self.config.supported_exchanges or [])
        }

        # Execution latency tracking
        self.latency_history: Dict[str, List[float]] = {}  # {exchange: [latencies]}

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate cross-exchange funding dispersion signal."""

        if not self.validate_data(data, min_periods=1):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient market data"},
            )

        try:
            # Get multi-venue funding rates
            venue_funding = await self._fetch_multi_venue_funding(symbol)
            if not venue_funding or len(venue_funding) < 2:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Insufficient venue funding data"},
                )

            # Update funding history
            self._update_funding_history(symbol, venue_funding)

            # Check exchange connectivity
            connectivity_ok = self._check_exchange_connectivity()
            if not connectivity_ok:
                return self._generate_emergency_exit_signal(
                    symbol, "connectivity_issues"
                )

            # Check existing position status
            existing_position = self.active_positions.get(symbol)
            if existing_position:
                return await self._handle_existing_position(symbol, venue_funding, data)

            # Calculate funding dispersion
            dispersion_data = self._calculate_funding_dispersion(venue_funding)
            if dispersion_data is None:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={"error": "Failed to calculate dispersion"},
                )

            # Check entry threshold
            if abs(dispersion_data["dispersion_bps"]) < self.config.entry_threshold_bps:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        "reason": "below_entry_threshold",
                        "dispersion_bps": dispersion_data["dispersion_bps"],
                        "entry_threshold": self.config.entry_threshold_bps,
                        "venue_rates": venue_funding,
                    },
                )

            # Check liquidity and execution feasibility
            execution_feasible = await self._check_execution_feasibility(
                symbol, dispersion_data
            )
            if not execution_feasible:
                return SignalResult(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    value=0.0,
                    confidence=0.0,
                    metadata={
                        "reason": "execution_not_feasible",
                        "dispersion_bps": dispersion_data["dispersion_bps"],
                    },
                )

            # Generate arbitrage signal
            signal_value = self._calculate_arbitrage_signal(dispersion_data)
            confidence = self._calculate_arbitrage_confidence(
                symbol, dispersion_data, venue_funding
            )

            # Apply risk management
            risk_adjusted_signal = self._apply_arbitrage_risk_management(
                signal_value, dispersion_data, venue_funding
            )

            # Store position entry info
            if abs(risk_adjusted_signal) > 0.01:  # Any meaningful signal
                self._record_position_entry(
                    symbol, dispersion_data, venue_funding, risk_adjusted_signal
                )

            # Comprehensive metadata
            metadata = {
                "strategy_type": "cross_exchange_funding_arbitrage",
                "dispersion_bps": dispersion_data["dispersion_bps"],
                "entry_threshold": self.config.entry_threshold_bps,
                "cheap_venue": dispersion_data["cheap_venue"],
                "expensive_venue": dispersion_data["expensive_venue"],
                "cheap_funding": dispersion_data["cheap_funding"],
                "expensive_funding": dispersion_data["expensive_funding"],
                "venue_funding_rates": venue_funding,
                "inventory_cost_bps": self.config.inventory_cost_bps,
                "expected_profit_bps": abs(dispersion_data["dispersion_bps"])
                - self.config.inventory_cost_bps,
                "convergence_window_hours": self.config.max_position_hours,
                "signal_direction": (
                    "long_cheap_short_expensive" if signal_value > 0 else "neutral"
                ),
                "position_size": abs(risk_adjusted_signal),
                "risk_adjustments": {
                    "inventory_cost_impact": self.config.inventory_cost_bps,
                    "latency_buffer": self.config.latency_buffer_ms,
                    "liquidity_check": execution_feasible,
                },
                "exchange_status": self.exchange_status.copy(),
                "dispersion_persistence": self._calculate_dispersion_persistence(
                    symbol
                ),
                "venue_volume_ratios": await self._get_venue_volume_ratios(symbol),
                "estimated_convergence_hours": self._estimate_convergence_time(
                    dispersion_data
                ),
                "max_allocation": self.config.max_allocation,
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
                metadata={"error": f"Signal generation failed: {str(e)}"},
            )

    async def _fetch_multi_venue_funding(
        self, symbol: str
    ) -> Optional[Dict[str, float]]:
        """
        Fetch funding rates from multiple exchanges.

        Mock implementation - in production this would query multiple exchange APIs.
        Returns {exchange: funding_rate} dict.
        """

        # Mock funding rates with realistic cross-venue spreads
        base_rates = {
            "BTCUSDT": {"binance": 0.0005, "bybit": 0.0008, "okx": 0.0003},
            "ETHUSDT": {"binance": -0.0012, "bybit": -0.0008, "okx": -0.0015},
            "ADAUSDT": {"binance": 0.0025, "bybit": 0.0018, "okx": 0.0030},
            "DOTUSDT": {"binance": -0.0020, "bybit": -0.0035, "okx": -0.0018},
            "SOLUSDT": {"binance": 0.0015, "bybit": 0.0040, "okx": 0.0012},
        }

        if symbol not in base_rates:
            # Default rates for unknown symbols
            import random

            random.seed(int(datetime.now().timestamp()) % 1000)

            base_rate = random.uniform(-0.003, 0.003)
            return {
                exchange: base_rate + random.uniform(-0.001, 0.001)
                for exchange in (self.config.supported_exchanges or [])
            }

        # Add time-based variation and cross-venue dispersion
        venue_rates = {}
        for exchange in self.config.supported_exchanges or []:
            if exchange in base_rates[symbol]:
                base_rate = base_rates[symbol][exchange]

                # Add realistic variation and occasional extreme dispersions
                import random

                random.seed(int(datetime.now().timestamp() + hash(exchange)) % 1000)

                # Occasional large dispersions (5% of the time)
                if random.random() < 0.05:
                    # Large dispersion event
                    dispersion_factor = random.uniform(1.5, 3.0)
                    if exchange == "bybit":  # Make bybit expensive sometimes
                        base_rate *= dispersion_factor
                    elif exchange == "okx":  # Make okx cheap sometimes
                        base_rate /= dispersion_factor

                # Normal small variations
                variation = random.uniform(-0.0002, 0.0002)
                venue_rates[exchange] = base_rate + variation

        return venue_rates if len(venue_rates) >= 2 else None

    def _update_funding_history(
        self, symbol: str, venue_funding: Dict[str, float]
    ) -> None:
        """Update funding rate history for dispersion tracking."""
        if symbol not in self.funding_history:
            self.funding_history[symbol] = []

        # Store timestamped venue funding snapshot
        self.funding_history[symbol].append((datetime.utcnow(), venue_funding.copy()))

        # Keep only last 48 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        self.funding_history[symbol] = [
            (ts, rates)
            for ts, rates in self.funding_history[symbol]
            if ts > cutoff_time
        ]

    def _calculate_funding_dispersion(
        self, venue_funding: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Calculate funding rate dispersion across venues."""
        if len(venue_funding) < 2:
            return None

        rates = list(venue_funding.values())
        venues = list(venue_funding.keys())

        # Find min and max funding rates
        min_rate = min(rates)
        max_rate = max(rates)

        # Calculate dispersion in basis points
        dispersion_bps = (max_rate - min_rate) * 10000

        # Identify cheap and expensive venues
        cheap_venue = venues[rates.index(min_rate)]
        expensive_venue = venues[rates.index(max_rate)]

        return {
            "dispersion_bps": dispersion_bps,
            "cheap_venue": cheap_venue,
            "expensive_venue": expensive_venue,
            "cheap_funding": min_rate,
            "expensive_funding": max_rate,
            "venue_spread": max_rate - min_rate,
            "mean_funding": np.mean(rates),
            "funding_std": np.std(rates),
            "venue_count": len(venue_funding),
        }

    async def _handle_existing_position(
        self, symbol: str, venue_funding: Dict[str, float], data: pd.DataFrame
    ) -> SignalResult:
        """Handle existing arbitrage position - check for exit conditions."""

        position = self.active_positions[symbol]
        entry_time = self.position_entry_times[symbol]

        # Calculate current dispersion
        dispersion_data = self._calculate_funding_dispersion(venue_funding)
        if dispersion_data is None:
            return self._generate_emergency_exit_signal(
                symbol, "dispersion_calculation_failed"
            )

        # Check exit conditions

        # 1. Dispersion below exit threshold
        if abs(dispersion_data["dispersion_bps"]) <= self.config.exit_threshold_bps:
            self._close_position(symbol)
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={
                    "reason": "dispersion_converged",
                    "current_dispersion_bps": dispersion_data["dispersion_bps"],
                    "exit_threshold": self.config.exit_threshold_bps,
                    "position_duration_hours": (
                        datetime.utcnow() - entry_time
                    ).total_seconds()
                    / 3600,
                    "entry_dispersion_bps": position.get("entry_dispersion_bps"),
                    "profit_loss_estimate": self._calculate_pnl_estimate(
                        symbol, dispersion_data
                    ),
                },
            )

        # 2. Maximum position time exceeded
        position_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
        if position_hours >= self.config.max_position_hours:
            self._close_position(symbol)
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={
                    "reason": "max_time_exceeded",
                    "position_hours": position_hours,
                    "max_hours": self.config.max_position_hours,
                    "current_dispersion_bps": dispersion_data["dispersion_bps"],
                    "profit_loss_estimate": self._calculate_pnl_estimate(
                        symbol, dispersion_data
                    ),
                },
            )

        # 3. Spread reversal stop loss
        entry_dispersion = position.get("entry_dispersion_bps", 0)
        current_dispersion = dispersion_data["dispersion_bps"]

        # Check if spread has moved significantly against us
        # Original position was based on entry dispersion magnitude
        dispersion_reduction = abs(entry_dispersion) - abs(current_dispersion)

        if dispersion_reduction > self.config.reversal_stop_bps:
            self._close_position(symbol)
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={
                    "reason": "spread_reversal_stop",
                    "dispersion_reduction_bps": dispersion_reduction,
                    "reversal_stop_bps": self.config.reversal_stop_bps,
                    "current_dispersion_bps": dispersion_data["dispersion_bps"],
                    "entry_dispersion_bps": entry_dispersion,
                    "profit_loss_estimate": self._calculate_pnl_estimate(
                        symbol, dispersion_data
                    ),
                },
            )

        # Position still valid - maintain current exposure
        current_signal = position.get("signal_value", 0.0)

        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=current_signal,
            confidence=min(
                1.0,
                abs(dispersion_data["dispersion_bps"])
                / self.config.entry_threshold_bps,
            ),
            metadata={
                "reason": "maintaining_position",
                "position_hours": position_hours,
                "current_dispersion_bps": dispersion_data["dispersion_bps"],
                "entry_dispersion_bps": entry_dispersion,
                "dispersion_reduction_bps": dispersion_reduction,
                "estimated_pnl": self._calculate_pnl_estimate(symbol, dispersion_data),
                "time_to_max": self.config.max_position_hours - position_hours,
            },
        )

    async def _check_execution_feasibility(
        self, symbol: str, dispersion_data: Dict[str, Any]
    ) -> bool:
        """Check if arbitrage execution is feasible given current market conditions."""

        # Mock liquidity check - in production would query order books
        cheap_venue = dispersion_data["cheap_venue"]
        expensive_venue = dispersion_data["expensive_venue"]

        # Simulate liquidity availability
        liquidity_sufficient = True

        # Check minimum liquidity threshold
        mock_liquidity = {
            "binance": 5000000,  # $5M
            "bybit": 3000000,  # $3M
            "okx": 2000000,  # $2M
        }

        for venue in [cheap_venue, expensive_venue]:
            if venue in mock_liquidity:
                if mock_liquidity[venue] < self.config.min_liquidity_threshold:
                    liquidity_sufficient = False
                    break

        # Check exchange connectivity
        connectivity_ok = self.exchange_status.get(
            cheap_venue, False
        ) and self.exchange_status.get(expensive_venue, False)

        # Check if dispersion after costs is still profitable
        net_profit_bps = (
            abs(dispersion_data["dispersion_bps"]) - self.config.inventory_cost_bps
        )
        profitable = net_profit_bps > 5.0  # Minimum 5 bps profit after costs

        return liquidity_sufficient and connectivity_ok and profitable

    def _calculate_arbitrage_signal(self, dispersion_data: Dict[str, Any]) -> float:
        """Calculate arbitrage signal strength based on dispersion magnitude."""

        dispersion_bps = abs(dispersion_data["dispersion_bps"])

        # Only generate signal if above entry threshold
        if dispersion_bps <= self.config.entry_threshold_bps:
            return 0.0

        # Signal strength scales with dispersion magnitude above threshold
        excess_dispersion = dispersion_bps - self.config.entry_threshold_bps

        # Normalize to 0-1 range, with 1.0 at 50 bps dispersion
        max_expected_dispersion = 50.0
        signal_strength = min(
            1.0,
            excess_dispersion
            / (max_expected_dispersion - self.config.entry_threshold_bps),
        )

        # Always positive for market-neutral arbitrage (direction handled in position allocation)
        return signal_strength

    def _calculate_arbitrage_confidence(
        self,
        symbol: str,
        dispersion_data: Dict[str, Any],
        venue_funding: Dict[str, float],
    ) -> float:
        """Calculate confidence in arbitrage opportunity."""

        # Base confidence from dispersion magnitude
        dispersion_bps = abs(dispersion_data["dispersion_bps"])
        magnitude_confidence = min(
            1.0, dispersion_bps / (self.config.entry_threshold_bps * 2)
        )

        # Persistence factor - higher confidence if dispersion has been stable
        persistence_factor = self._calculate_dispersion_persistence(symbol)

        # Volume balance factor - higher confidence if venues have balanced volume
        volume_balance = self._calculate_venue_volume_balance(
            list(venue_funding.keys())
        )

        # Net profit factor after costs
        net_profit_bps = dispersion_bps - self.config.inventory_cost_bps
        profit_factor = min(
            1.0, max(0.0, net_profit_bps / 20.0)
        )  # 20 bps = full confidence

        # Combine factors
        confidence = (
            magnitude_confidence * 0.4
            + persistence_factor * self.config.persistence_weight
            + volume_balance * self.config.volume_weight
            + profit_factor * 0.1
        )

        return max(0.0, min(1.0, confidence))

    def _calculate_dispersion_persistence(self, symbol: str) -> float:
        """Calculate how persistent the current dispersion has been."""
        history = self.funding_history.get(symbol, [])
        if len(history) < 3:
            return 0.5  # Neutral for insufficient data

        # Look at last few hours of dispersion data
        recent_dispersions = []
        for _, venue_rates in history[-6:]:  # Last 6 data points
            if len(venue_rates) >= 2:
                rates = list(venue_rates.values())
                dispersion = (max(rates) - min(rates)) * 10000  # Convert to bps
                recent_dispersions.append(dispersion)

        if len(recent_dispersions) < 2:
            return 0.5

        # Higher persistence if dispersion has been consistently above threshold
        above_threshold_count = sum(
            1 for d in recent_dispersions if d >= self.config.entry_threshold_bps
        )
        persistence = above_threshold_count / len(recent_dispersions)

        return persistence

    def _calculate_venue_volume_balance(self, venues: List[str]) -> float:
        """Calculate volume balance factor across venues (mock implementation)."""
        # Mock implementation - in production would use actual volume data
        # Higher score for more balanced volume distribution
        return 0.8  # Assume reasonable balance

    async def _get_venue_volume_ratios(self, symbol: str) -> Dict[str, float]:
        """Get volume ratios across venues (mock implementation)."""
        # Mock volume distribution
        return {"binance": 0.45, "bybit": 0.35, "okx": 0.20}

    def _apply_arbitrage_risk_management(
        self,
        signal_value: float,
        dispersion_data: Dict[str, Any],
        venue_funding: Dict[str, float],
    ) -> float:
        """Apply risk management scaling to arbitrage signal."""

        if signal_value == 0.0:
            return 0.0

        # Base allocation scaling
        risk_adjusted = signal_value * self.config.max_allocation

        # Reduce for insufficient profit margin
        net_profit_bps = (
            abs(dispersion_data["dispersion_bps"]) - self.config.inventory_cost_bps
        )
        if net_profit_bps < 10.0:  # Less than 10 bps net profit
            profit_scale = max(0.3, net_profit_bps / 10.0)
            risk_adjusted *= profit_scale

        # Reduce for venue concentration risk
        venue_count = len(venue_funding)
        if venue_count < 3:
            risk_adjusted *= 0.8  # Reduce exposure with fewer venues

        # Ensure within allocation limits
        return max(0.0, min(self.config.max_allocation, risk_adjusted))

    def _record_position_entry(
        self,
        symbol: str,
        dispersion_data: Dict[str, Any],
        venue_funding: Dict[str, float],
        signal_value: float,
    ) -> None:
        """Record position entry for tracking."""
        self.position_entry_times[symbol] = datetime.utcnow()
        self.active_positions[symbol] = {
            "entry_time": datetime.utcnow(),
            "entry_dispersion_bps": dispersion_data["dispersion_bps"],
            "cheap_venue": dispersion_data["cheap_venue"],
            "expensive_venue": dispersion_data["expensive_venue"],
            "entry_venue_rates": venue_funding.copy(),
            "signal_value": signal_value,  # Store the actual signal strength
        }

    def _close_position(self, symbol: str) -> None:
        """Close position and clean up tracking."""
        self.active_positions.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)

    def _calculate_pnl_estimate(
        self, symbol: str, current_dispersion_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate P&L for current position."""
        position = self.active_positions.get(symbol)
        if not position:
            return {"estimated_pnl_bps": 0.0}

        entry_dispersion = position["entry_dispersion_bps"]
        current_dispersion = current_dispersion_data["dispersion_bps"]

        # P&L = convergence of spread minus costs
        convergence_profit = abs(entry_dispersion) - abs(current_dispersion)
        net_pnl = convergence_profit - self.config.inventory_cost_bps

        return {
            "estimated_pnl_bps": net_pnl,
            "entry_dispersion_bps": entry_dispersion,
            "current_dispersion_bps": current_dispersion,
            "convergence_profit_bps": convergence_profit,
            "inventory_cost_bps": self.config.inventory_cost_bps,
        }

    def _estimate_convergence_time(self, dispersion_data: Dict[str, Any]) -> float:
        """Estimate convergence time based on dispersion magnitude."""
        dispersion_bps = abs(dispersion_data["dispersion_bps"])

        # Linear model: larger dispersions take longer to converge
        # Based on Kaiko research: 24-48h convergence window
        base_hours = 24.0
        max_hours = 48.0

        # Scale linearly with dispersion magnitude
        if dispersion_bps <= self.config.entry_threshold_bps:
            return base_hours

        # Higher dispersions take longer
        excess_ratio = (
            dispersion_bps - self.config.entry_threshold_bps
        ) / 30.0  # 30 bps reference
        estimated_hours = base_hours + (max_hours - base_hours) * min(1.0, excess_ratio)

        return estimated_hours

    def _check_exchange_connectivity(self) -> bool:
        """Check if critical exchanges are connected."""
        critical_exchanges = (self.config.supported_exchanges or [])[
            :2
        ]  # First 2 exchanges are critical
        return all(
            self.exchange_status.get(exchange, False) for exchange in critical_exchanges
        )

    def _generate_emergency_exit_signal(self, symbol: str, reason: str) -> SignalResult:
        """Generate emergency exit signal."""
        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={
                "reason": f"emergency_exit_{reason}",
                "emergency": True,
                "action_required": "immediate_position_close",
            },
        )

    # Utility methods for monitoring and testing

    def get_arbitrage_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive arbitrage metrics for monitoring."""

        position = self.active_positions.get(symbol)
        history = self.funding_history.get(symbol, [])

        metrics = {
            "position_active": symbol in self.active_positions,
            "exchange_status": self.exchange_status.copy(),
            "supported_exchanges": self.config.supported_exchanges,
        }

        if position:
            entry_time = self.position_entry_times[symbol]
            position_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600

            metrics.update(
                {
                    "position_duration_hours": position_hours,
                    "entry_dispersion_bps": position["entry_dispersion_bps"],
                    "cheap_venue": position["cheap_venue"],
                    "expensive_venue": position["expensive_venue"],
                    "time_remaining_hours": max(
                        0, self.config.max_position_hours - position_hours
                    ),
                }
            )

        if history:
            recent_dispersions = []
            for _, venue_rates in history[-12:]:  # Last 12 hours
                if len(venue_rates) >= 2:
                    rates = list(venue_rates.values())
                    dispersion = (max(rates) - min(rates)) * 10000
                    recent_dispersions.append(dispersion)

            if recent_dispersions:
                metrics.update(
                    {
                        "avg_dispersion_12h_bps": np.mean(recent_dispersions),
                        "max_dispersion_12h_bps": max(recent_dispersions),
                        "min_dispersion_12h_bps": min(recent_dispersions),
                        "dispersion_volatility": np.std(recent_dispersions),
                        "opportunities_count": sum(
                            1
                            for d in recent_dispersions
                            if d >= self.config.entry_threshold_bps
                        ),
                    }
                )

        return metrics

    def update_exchange_status(self, exchange: str, status: bool) -> None:
        """Update exchange connectivity status."""
        if exchange in (self.config.supported_exchanges or []):
            self.exchange_status[exchange] = status

    def force_close_position(
        self, symbol: str, reason: str = "manual_override"
    ) -> None:
        """Force close position (for emergency or manual intervention)."""
        if symbol in self.active_positions:
            self._close_position(symbol)

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all active arbitrage positions."""
        return {
            "active_positions_count": len(self.active_positions),
            "symbols": list(self.active_positions.keys()),
            "total_allocation": len(self.active_positions) * self.config.max_allocation,
            "oldest_position_hours": (
                max(
                    (datetime.utcnow() - entry_time).total_seconds() / 3600
                    for entry_time in self.position_entry_times.values()
                )
                if self.position_entry_times
                else 0
            ),
        }
