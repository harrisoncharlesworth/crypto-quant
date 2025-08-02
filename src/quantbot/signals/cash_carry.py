import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import SignalBase, SignalResult, SignalConfig


class FuturesExpiry(Enum):
    """Futures contract expiry types."""

    QUARTERLY = "quarterly"
    PERPETUAL = "perpetual"
    MONTHLY = "monthly"


@dataclass
class FuturesContract:
    """Mock futures contract data structure."""

    symbol: str
    expiry_date: Optional[datetime]
    contract_type: FuturesExpiry
    price: float
    volume: float
    open_interest: float
    margin_requirement: float
    funding_rate: Optional[float] = None  # For perpetuals


@dataclass
class CashCarryConfig(SignalConfig):
    """Configuration for Cash-and-Carry Basis arbitrage signal."""

    # Strategy thresholds
    basis_threshold_min: float = 0.08  # 8% annualized basis minimum
    basis_threshold_max: float = 0.10  # 10% annualized basis for max signal
    max_allocation: float = 0.20  # 20% max allocation per arbitrage

    # Risk management
    margin_spike_threshold: float = 0.5  # 50% increase in margin = reduce size
    margin_emergency_threshold: float = 1.0  # 100% increase = emergency exit
    fee_change_threshold: float = 0.002  # 0.2% fee increase threshold
    max_basis_decay_days: int = 30  # Max days to convergence

    # Liquidity requirements
    min_volume_24h: float = 1000000  # $1M minimum 24h volume
    min_spot_depth: float = 100000  # $100k minimum order book depth
    min_futures_oi: float = 50000000  # $50M minimum open interest

    # Position management
    position_size_scale: float = 1.0  # Scale factor for position sizing
    basis_stability_periods: int = 24  # Hours to check basis stability
    confidence_basis_multiplier: float = 5.0  # Confidence scaling

    # Convergence timing
    min_days_to_expiry: int = 7  # Minimum days before expiry
    max_days_to_expiry: int = 90  # Maximum days for quarterly contracts


class CashCarryArbitrageSignal(SignalBase):
    """
    Cash-and-Carry Basis Arbitrage Signal - Market-Neutral Strategy

    Strategy: Exploit pricing discrepancies between spot and futures markets
    Classification: Market-Neutral (M-N)

    Key Mechanics:
    - When futures > spot (contango): Long spot + Short futures
    - When spot > futures (backwardation): Short spot + Long futures
    - Harvest annualized basis >8-10% on high-liquidity pairs
    - Target 10% avg excess return, Sharpe ≈ 0.6

    Risk Management:
    - Monitor margin requirement changes (CME spikes can nuke arbitrage)
    - Track fee structure changes
    - Maintain 1:1 hedge ratio (delta-neutral)
    - Size according to basis magnitude and liquidity
    - Emergency exits for margin spikes >100%
    """

    def __init__(self, config: CashCarryConfig):
        super().__init__(config)
        self.config: CashCarryConfig = config
        self.basis_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.margin_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.position_entries: Dict[str, Dict] = {}
        self.carry_income_tracker: Dict[str, float] = {}

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate cash-carry arbitrage signal from spot and futures data."""

        if not self.validate_data(data, min_periods=1):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient market data"},
            )

        try:
            # Get current spot price
            spot_price = data.iloc[-1]["close"]

            # Get futures contracts data (mock implementation)
            futures_contracts = await self._get_futures_contracts(symbol)
            if not futures_contracts:
                return self._no_signal_result(symbol, "No futures contracts available")

            # Calculate basis for each contract
            arbitrage_opportunities = []

            for contract in futures_contracts:
                basis_data = self._calculate_basis(spot_price, contract)
                if basis_data and self._meets_liquidity_requirements(contract, data):
                    arbitrage_opportunities.append((contract, basis_data))

            if not arbitrage_opportunities:
                return self._no_signal_result(
                    symbol, "No valid arbitrage opportunities"
                )

            # Select best opportunity
            best_contract, best_basis = self._select_best_opportunity(
                arbitrage_opportunities
            )

            # Check risk management conditions
            risk_check = await self._check_risk_conditions(symbol, best_contract)
            if not risk_check["allowed"]:
                return self._no_signal_result(
                    symbol, f"Risk check failed: {risk_check['reason']}"
                )

            # Generate signal
            signal_value = self._calculate_signal_value(best_basis)

            # If no signal, return early with zero confidence
            if signal_value == 0.0:
                return self._no_signal_result(
                    symbol,
                    f"Basis {best_basis['annualized_basis']:.1%} below threshold {self.config.basis_threshold_min:.1%}",
                )

            confidence = self._calculate_confidence(symbol, best_basis, best_contract)

            # Apply position sizing
            position_size = self._calculate_position_size(
                best_basis, best_contract, confidence
            )
            final_signal = signal_value * position_size

            # Update tracking
            self._update_basis_history(symbol, best_basis["annualized_basis"])
            self._update_margin_history(symbol, best_contract.margin_requirement)

            # Metadata
            metadata = {
                "spot_price": spot_price,
                "futures_price": best_contract.price,
                "annualized_basis": best_basis["annualized_basis"],
                "days_to_expiry": best_basis["days_to_expiry"],
                "contract_type": best_contract.contract_type.value,
                "margin_requirement": best_contract.margin_requirement,
                "expected_return": self._estimate_carry_income(best_basis),
                "position_size": position_size,
                "risk_factors": risk_check,
                "liquidity_check": self._get_liquidity_metrics(best_contract, data),
                "basis_stability": self._calculate_basis_stability(symbol),
                "convergence_timing": self._estimate_convergence_timing(best_basis),
                "strategy_type": "cash_carry_arbitrage",
                "hedge_ratio": 1.0,  # 1:1 delta-neutral
                "max_allocation": self.config.max_allocation,
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

    async def _get_futures_contracts(self, symbol: str) -> List[FuturesContract]:
        """
        Get available futures contracts for the symbol.
        Mock implementation - in production would query CME/Binance APIs.
        """
        # Mock futures data for major crypto pairs
        base_symbol = symbol.replace("USDT", "").replace("USD", "")

        contracts = []

        # Mock quarterly contracts
        quarterly_expiries = [
            datetime.now() + timedelta(days=30),  # Monthly
            datetime.now() + timedelta(days=90),  # Quarterly
            datetime.now() + timedelta(days=180),  # Semi-annual
        ]

        # Mock spot price (would come from data)
        spot_price = (
            45000 if "BTC" in base_symbol else 3000 if "ETH" in base_symbol else 100
        )

        for i, expiry in enumerate(quarterly_expiries):
            # Create realistic futures pricing with slight contango
            time_to_expiry = (expiry - datetime.now()).days / 365.25
            contango_premium = 0.05 * time_to_expiry  # 5% annualized contango
            futures_price = spot_price * (1 + contango_premium)

            # Simulate volume and OI based on time to expiry
            volume_base = 500000000 if "BTC" in base_symbol else 200000000
            oi_base = 2000000000 if "BTC" in base_symbol else 800000000

            volume = volume_base * (1 - i * 0.3)  # Nearer contracts have more volume
            open_interest = oi_base * (1 - i * 0.4)

            # Mock margin requirements (CME-style)
            margin_req = 0.05 + (i * 0.01)  # 5-7% margin requirement

            contracts.append(
                FuturesContract(
                    symbol=f"{base_symbol}-{expiry.strftime('%m%y')}",
                    expiry_date=expiry,
                    contract_type=FuturesExpiry.QUARTERLY,
                    price=futures_price,
                    volume=volume,
                    open_interest=open_interest,
                    margin_requirement=margin_req,
                )
            )

        # Mock perpetual contract
        perp_price = spot_price * 1.001  # Small premium
        contracts.append(
            FuturesContract(
                symbol=f"{base_symbol}-PERP",
                expiry_date=None,
                contract_type=FuturesExpiry.PERPETUAL,
                price=perp_price,
                volume=volume_base * 1.5,
                open_interest=oi_base * 2,
                margin_requirement=0.02,  # Lower margin for perps
                funding_rate=0.0001,  # Mock funding rate
            )
        )

        return contracts

    def _calculate_basis(
        self, spot_price: float, contract: FuturesContract
    ) -> Optional[Dict[str, Any]]:
        """Calculate annualized basis for futures contract."""

        if contract.contract_type == FuturesExpiry.PERPETUAL:
            # For perpetuals, use funding rate as basis
            if contract.funding_rate is None:
                return None

            # Annualize funding rate (assuming 8-hour funding)
            annualized_basis = contract.funding_rate * 3 * 365

            return {
                "basis_points": (contract.price - spot_price) / spot_price,
                "annualized_basis": annualized_basis,
                "days_to_expiry": None,
                "is_perpetual": True,
                "funding_rate": contract.funding_rate,
            }

        else:
            # For dated futures
            if contract.expiry_date is None:
                return None

            days_to_expiry = (contract.expiry_date - datetime.now()).days

            if days_to_expiry < self.config.min_days_to_expiry:
                return None  # Too close to expiry

            if days_to_expiry > self.config.max_days_to_expiry:
                return None  # Too far from expiry

            # Calculate annualized basis
            basis_points = (contract.price - spot_price) / spot_price
            time_fraction = days_to_expiry / 365.25
            annualized_basis = basis_points / time_fraction if time_fraction > 0 else 0

            return {
                "basis_points": basis_points,
                "annualized_basis": annualized_basis,
                "days_to_expiry": days_to_expiry,
                "is_perpetual": False,
                "time_fraction": time_fraction,
            }

    def _meets_liquidity_requirements(
        self, contract: FuturesContract, spot_data: pd.DataFrame
    ) -> bool:
        """Check if contract meets minimum liquidity requirements."""

        # Check futures volume and open interest
        if contract.volume < self.config.min_volume_24h:
            return False

        if contract.open_interest < self.config.min_futures_oi:
            return False

        # Check spot volume (from data)
        if len(spot_data) > 0:
            recent_volume = spot_data.iloc[-1]["volume"]
            if recent_volume < self.config.min_volume_24h:
                return False

        return True

    def _select_best_opportunity(
        self, opportunities: List[Tuple[FuturesContract, Dict]]
    ) -> Tuple[FuturesContract, Dict]:
        """Select the best arbitrage opportunity based on risk-adjusted return."""

        def score_opportunity(contract, basis_data):
            # Higher basis = better
            basis_score = abs(basis_data["annualized_basis"])

            # Penalize very short or very long expiries
            if basis_data["is_perpetual"]:
                time_score = 1.0  # Perpetuals get neutral time score
            else:
                days = basis_data["days_to_expiry"]
                ideal_days = 45  # Prefer ~6 week contracts
                time_score = 1.0 - abs(days - ideal_days) / 90
                time_score = max(0.1, time_score)

            # Prefer higher liquidity
            liquidity_score = min(
                1.0, contract.open_interest / self.config.min_futures_oi
            )

            # Penalize high margin requirements
            margin_score = max(0.1, 1.0 - contract.margin_requirement)

            return basis_score * time_score * liquidity_score * margin_score

        scored_opportunities = [
            (contract, basis_data, score_opportunity(contract, basis_data))
            for contract, basis_data in opportunities
        ]

        # Sort by score (highest first)
        scored_opportunities.sort(key=lambda x: x[2], reverse=True)

        best_contract, best_basis, _ = scored_opportunities[0]
        return best_contract, best_basis

    async def _check_risk_conditions(
        self, symbol: str, contract: FuturesContract
    ) -> Dict[str, Any]:
        """Check risk management conditions."""

        # Check margin spike conditions
        margin_history = self.margin_history.get(symbol, [])
        if margin_history:
            recent_margins = [
                m for t, m in margin_history if t > datetime.now() - timedelta(hours=24)
            ]
            if recent_margins:
                avg_margin = np.mean(recent_margins)
                margin_increase = (
                    contract.margin_requirement - avg_margin
                ) / avg_margin

                if margin_increase > self.config.margin_emergency_threshold:
                    return {
                        "allowed": False,
                        "reason": f"Emergency margin spike: {margin_increase:.1%}",
                        "margin_increase": margin_increase,
                    }

                if margin_increase > self.config.margin_spike_threshold:
                    return {
                        "allowed": True,
                        "reason": "Margin spike detected - reduced sizing",
                        "margin_increase": margin_increase,
                        "size_reduction": 0.5,
                    }

        # Check if we're already in a position (avoid over-concentration)
        if symbol in self.position_entries:
            existing_pos = self.position_entries[symbol]
            time_in_position = (datetime.now() - existing_pos["entry_time"]).days

            if time_in_position > self.config.max_basis_decay_days:
                return {
                    "allowed": False,
                    "reason": f"Position held too long: {time_in_position} days",
                    "time_in_position": time_in_position,
                }

        return {
            "allowed": True,
            "reason": "All risk checks passed",
            "margin_change": 0.0,
        }

    def _calculate_signal_value(self, basis_data: Dict[str, Any]) -> float:
        """Calculate signal value based on basis magnitude."""

        annualized_basis = basis_data["annualized_basis"]

        # Check if basis meets minimum threshold
        if abs(annualized_basis) < self.config.basis_threshold_min:
            return 0.0

        # Calculate signal strength
        if annualized_basis > 0:
            # Contango: Long spot, Short futures
            signal_strength = min(
                1.0, annualized_basis / self.config.basis_threshold_max
            )
            return signal_strength
        else:
            # Backwardation: Short spot, Long futures
            signal_strength = min(
                1.0, abs(annualized_basis) / self.config.basis_threshold_max
            )
            return -signal_strength

    def _calculate_confidence(
        self, symbol: str, basis_data: Dict[str, Any], contract: FuturesContract
    ) -> float:
        """Calculate confidence based on basis stability and market conditions."""

        # Base confidence from basis magnitude
        basis_magnitude = abs(basis_data["annualized_basis"])
        magnitude_confidence = min(
            1.0, basis_magnitude * self.config.confidence_basis_multiplier
        )

        # Boost for strong basis above max threshold
        if basis_magnitude >= self.config.basis_threshold_max:
            magnitude_confidence = min(1.0, magnitude_confidence * 1.5)

        # Basis stability factor
        stability_factor = self._calculate_basis_stability(symbol)

        # Liquidity confidence
        liquidity_confidence = min(
            1.0, contract.open_interest / (self.config.min_futures_oi * 2)
        )

        # Time to expiry factor (for dated futures)
        time_factor = 1.0
        if not basis_data["is_perpetual"]:
            days = basis_data["days_to_expiry"]
            # Prefer contracts with 2-8 weeks to expiry
            if 14 <= days <= 56:
                time_factor = 1.0
            else:
                time_factor = max(0.3, 1.0 - abs(days - 35) / 60)

        # Margin requirement penalty
        margin_factor = max(0.1, 1.0 - contract.margin_requirement)

        final_confidence = (
            magnitude_confidence
            * stability_factor
            * liquidity_confidence
            * time_factor
            * margin_factor
        )

        return max(0.0, min(1.0, final_confidence))

    def _calculate_position_size(
        self, basis_data: Dict[str, Any], contract: FuturesContract, confidence: float
    ) -> float:
        """Calculate position size based on basis strength and risk factors."""

        # Base size from max allocation
        base_size = self.config.max_allocation

        # Scale by confidence
        confidence_scaled = base_size * confidence

        # Scale by basis strength relative to threshold
        basis_strength = (
            abs(basis_data["annualized_basis"]) / self.config.basis_threshold_max
        )
        basis_scaled = confidence_scaled * min(1.0, basis_strength)

        # Reduce size for high margin requirements
        margin_penalty = max(0.1, 1.0 - contract.margin_requirement)
        margin_scaled = basis_scaled * margin_penalty

        # Apply global position scaling
        final_size = margin_scaled * self.config.position_size_scale

        return max(0.0, min(self.config.max_allocation, final_size))

    def _update_basis_history(self, symbol: str, basis: float) -> None:
        """Update basis history for stability tracking."""
        if symbol not in self.basis_history:
            self.basis_history[symbol] = []

        self.basis_history[symbol].append((datetime.now(), basis))

        # Keep only recent history
        cutoff = datetime.now() - timedelta(
            hours=self.config.basis_stability_periods * 2
        )
        self.basis_history[symbol] = [
            (t, b) for t, b in self.basis_history[symbol] if t > cutoff
        ]

    def _update_margin_history(self, symbol: str, margin: float) -> None:
        """Update margin requirement history."""
        if symbol not in self.margin_history:
            self.margin_history[symbol] = []

        self.margin_history[symbol].append((datetime.now(), margin))

        # Keep last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        self.margin_history[symbol] = [
            (t, m) for t, m in self.margin_history[symbol] if t > cutoff
        ]

    def _calculate_basis_stability(self, symbol: str) -> float:
        """Calculate basis stability factor."""
        if symbol not in self.basis_history:
            return 0.5  # Neutral for no history

        history = self.basis_history[symbol]
        if len(history) < 2:
            return 0.5

        recent_basis = [b for t, b in history[-self.config.basis_stability_periods :]]

        if len(recent_basis) < 2:
            return 0.5

        # Calculate coefficient of variation (lower = more stable)
        cv = (
            np.std(recent_basis) / abs(np.mean(recent_basis))
            if np.mean(recent_basis) != 0
            else 1.0
        )

        # Convert to stability score (higher = more stable)
        stability = max(0.1, 1.0 - min(1.0, cv * 2))

        return stability

    def _estimate_carry_income(self, basis_data: Dict[str, Any]) -> float:
        """Estimate expected carry income from the arbitrage."""

        annualized_basis = abs(basis_data["annualized_basis"])

        if basis_data["is_perpetual"]:
            # For perpetuals, estimate based on funding rate
            # Assume average 30-day holding period
            holding_period = 30 / 365.25
            return annualized_basis * holding_period
        else:
            # For dated futures, full basis should be captured
            return annualized_basis * basis_data["time_fraction"]

    def _estimate_convergence_timing(
        self, basis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate when basis will converge to zero."""

        if basis_data["is_perpetual"]:
            return {
                "type": "perpetual",
                "expected_days": None,
                "convergence_certainty": 0.0,  # Funding rates don't converge
            }
        else:
            days_to_expiry = basis_data["days_to_expiry"]
            return {
                "type": "expiry_convergence",
                "expected_days": days_to_expiry,
                "convergence_certainty": 0.95,  # High certainty at expiry
            }

    def _get_liquidity_metrics(
        self, contract: FuturesContract, spot_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get liquidity assessment metrics."""

        return {
            "futures_volume": contract.volume,
            "futures_oi": contract.open_interest,
            "spot_volume": spot_data.iloc[-1]["volume"] if len(spot_data) > 0 else 0,
            "volume_ratio": (
                contract.volume / max(1, spot_data.iloc[-1]["volume"])
                if len(spot_data) > 0
                else 0
            ),
            "liquidity_score": min(
                1.0, contract.open_interest / self.config.min_futures_oi
            ),
        }

    def _no_signal_result(self, symbol: str, reason: str) -> SignalResult:
        """Helper to create no-signal result."""
        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={"reason": reason, "strategy_type": "cash_carry_arbitrage"},
        )

    def track_carry_income(self, symbol: str, income: float) -> None:
        """Track actual carry income for performance monitoring."""
        if symbol not in self.carry_income_tracker:
            self.carry_income_tracker[symbol] = 0.0
        self.carry_income_tracker[symbol] += income

    def get_strategy_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get strategy-specific metrics for monitoring."""

        basis_history = self.basis_history.get(symbol, [])
        margin_history = self.margin_history.get(symbol, [])

        metrics = {
            "total_carry_income": self.carry_income_tracker.get(symbol, 0.0),
            "basis_stability": self._calculate_basis_stability(symbol),
            "basis_history_length": len(basis_history),
            "margin_history_length": len(margin_history),
        }

        if basis_history:
            recent_basis = [b for t, b in basis_history[-24:]]  # Last 24 periods
            metrics.update(
                {
                    "avg_basis_24h": np.mean(recent_basis),
                    "basis_volatility": np.std(recent_basis),
                    "max_basis_24h": max(recent_basis),
                    "min_basis_24h": min(recent_basis),
                }
            )

        if margin_history:
            recent_margins = [m for t, m in margin_history[-24:]]
            metrics.update(
                {
                    "avg_margin_24h": np.mean(recent_margins),
                    "margin_volatility": np.std(recent_margins),
                    "max_margin_24h": max(recent_margins),
                    "min_margin_24h": min(recent_margins),
                }
            )

        return metrics
