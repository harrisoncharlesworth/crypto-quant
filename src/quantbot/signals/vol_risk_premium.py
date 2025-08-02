import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm

from .base import SignalBase, SignalResult, SignalConfig


class OptionType(Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


class StraddleType(Enum):
    """Straddle type enumeration."""

    ATM = "atm"
    OTM = "otm"


@dataclass
class OptionsContract:
    """Mock Deribit options contract data structure."""

    symbol: str
    underlying: str
    strike: float
    expiry_date: datetime
    option_type: OptionType
    price: float
    bid: float
    ask: float
    implied_volatility: float
    volume: float
    open_interest: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    mark_price: float
    index_price: float  # Underlying spot price


@dataclass
class StraddlePosition:
    """Straddle position representation."""

    call_contract: OptionsContract
    put_contract: OptionsContract
    quantity: float
    entry_price: float
    entry_time: datetime
    premium_collected: float
    delta_hedge_units: float
    hedge_history: List[Dict] = None

    def __post_init__(self):
        if self.hedge_history is None:
            self.hedge_history = []


@dataclass
class WingProtection:
    """Tail protection wings (25Δ puts/calls)."""

    call_wing: OptionsContract  # 25Δ call
    put_wing: OptionsContract  # 25Δ put
    total_cost: float
    cost_ratio: float  # Cost as % of premium collected


@dataclass
class VolRiskPremiumConfig(SignalConfig):
    """Configuration for Options Vol-Risk Premium signal."""

    # Strategy thresholds
    vrp_threshold_min: float = 0.05  # 5% minimum VRP to trigger
    vrp_threshold_max: float = 0.15  # 15% VRP for max signal
    iv_percentile_threshold: float = 0.70  # IV above 70th percentile

    # Expiry preferences
    min_days_to_expiry: int = 7  # Minimum 7 days
    max_days_to_expiry: int = 14  # Maximum 14 days
    target_days_to_expiry: int = 10  # Optimal 10 days

    # Position sizing and risk
    max_vega_allocation: float = 0.20  # 20% max vega allocation
    vol_target: float = 0.15  # 15% volatility target
    max_position_count: int = 3  # Max concurrent straddle positions

    # Delta hedging
    delta_hedge_threshold: float = 0.10  # Hedge when |delta| > 10%
    hedge_frequency_hours: int = 4  # Hedge every 4 hours
    hedge_slippage: float = 0.001  # 0.1% slippage assumption

    # Tail protection (disaster wings)
    wing_delta: float = 0.25  # 25Δ wings
    max_wing_cost_ratio: float = 0.15  # ≤15% of premium collected
    enable_tail_protection: bool = True

    # Liquidity requirements
    min_volume_24h: float = 50000  # $50k minimum 24h volume
    min_open_interest: float = 100000  # $100k minimum OI
    max_bid_ask_spread: float = 0.05  # 5% max bid-ask spread

    # Risk management
    max_gamma_exposure: float = 100000  # Max gamma per position
    pnl_stop_loss: float = -0.30  # -30% stop loss
    early_close_profit: float = 0.50  # Close at 50% profit

    # Volatility calculation
    realized_vol_window: int = 30  # 30-day realized vol window
    vol_ema_alpha: float = 0.1  # EMA smoothing for vol estimates


class VolatilityRiskPremiumSignal(SignalBase):
    """
    Options Vol-Risk Premium Signal - Market-Neutral Volatility Strategy

    Strategy: Sell ATM straddles when implied vol > realized vol (positive VRP)
    Classification: Market-Neutral (M-N)

    Key Mechanics:
    - Sell 7-14D ATM straddles when VRP > threshold
    - Delta-hedge every 4 hours to maintain market neutrality
    - Hedge tail risk with 25Δ protective wings (cost ≤15% premium)
    - Vega position sizing with volatility targeting
    - Positive VRP across regimes (based on arXiv 2024 research)

    Options Greeks Management:
    - Delta: Hedged to zero every 4 hours
    - Gamma: Monitored for excessive exposure
    - Vega: Primary risk factor, sized via vol-targeting
    - Theta: Primary P&L source (time decay)

    Risk Management:
    - Tail protection via 25Δ wings
    - Position sizing based on vega limits
    - Early profit taking at 50% of premium
    - Stop loss at -30% of premium
    """

    def __init__(self, config: VolRiskPremiumConfig):
        super().__init__(config)
        self.config: VolRiskPremiumConfig = config
        self.active_positions: Dict[str, List[StraddlePosition]] = {}
        self.hedge_schedule: Dict[str, datetime] = {}
        self.vol_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.iv_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.pnl_tracker: Dict[str, Dict] = {}

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate vol-risk premium signal from options and spot data."""

        if not self.validate_data(data, min_periods=self.config.realized_vol_window):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient market data"},
            )

        try:
            # Calculate realized volatility
            realized_vol = self._calculate_realized_volatility(data)

            # Get options data (mock implementation)
            options_data = await self._get_options_data(symbol)
            if not options_data:
                return self._no_signal_result(symbol, "No options data available")

            # Find suitable ATM straddle opportunities
            straddle_opportunities = self._find_straddle_opportunities(
                options_data, data.iloc[-1]["close"]
            )

            if not straddle_opportunities:
                return self._no_signal_result(
                    symbol, "No suitable straddle opportunities"
                )

            # Calculate VRP for each opportunity
            vrp_signals = []
            for call_opt, put_opt in straddle_opportunities:
                vrp_data = self._calculate_vrp(call_opt, put_opt, realized_vol)
                if vrp_data and self._meets_liquidity_requirements(call_opt, put_opt):
                    vrp_signals.append((call_opt, put_opt, vrp_data))

            if not vrp_signals:
                return self._no_signal_result(
                    symbol, "No VRP opportunities above threshold"
                )

            # Select best opportunity
            best_call, best_put, best_vrp = self._select_best_vrp_opportunity(
                vrp_signals
            )

            # Check risk management conditions
            risk_check = await self._check_risk_conditions(symbol, best_call, best_put)
            if not risk_check["allowed"]:
                return self._no_signal_result(
                    symbol, f"Risk check failed: {risk_check['reason']}"
                )

            # Check for tail protection availability
            tail_protection = None
            if self.config.enable_tail_protection:
                tail_protection = await self._get_tail_protection(
                    symbol, best_call, best_put, best_vrp["premium_collected"]
                )

            # Calculate signal value and confidence
            signal_value = self._calculate_signal_value(best_vrp)
            confidence = self._calculate_confidence(
                symbol, best_vrp, best_call, best_put
            )

            # Calculate vega-weighted position size
            position_size = self._calculate_vega_position_size(
                best_call, best_put, confidence
            )
            final_signal = signal_value * position_size

            # Update tracking
            self._update_vol_history(symbol, realized_vol)
            self._update_iv_history(symbol, best_vrp["implied_vol"])

            # Metadata
            metadata = {
                "realized_vol": realized_vol,
                "implied_vol": best_vrp["implied_vol"],
                "vrp": best_vrp["vrp"],
                "premium_collected": best_vrp["premium_collected"],
                "days_to_expiry": best_vrp["days_to_expiry"],
                "atm_strike": best_call.strike,
                "spot_price": data.iloc[-1]["close"],
                "call_delta": best_call.delta,
                "put_delta": best_put.delta,
                "net_delta": best_call.delta + best_put.delta,
                "total_vega": best_call.vega + best_put.vega,
                "total_gamma": best_call.gamma + best_put.gamma,
                "total_theta": best_call.theta + best_put.theta,
                "position_size": position_size,
                "tail_protection": (
                    tail_protection.__dict__ if tail_protection else None
                ),
                "strategy_type": "vol_risk_premium",
                "hedge_schedule": "4_hours",
                "max_wing_cost": self.config.max_wing_cost_ratio,
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

    def _calculate_realized_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized realized volatility using log returns."""

        # Calculate log returns
        returns = np.log(data["close"] / data["close"].shift(1)).dropna()

        # Use rolling window
        if len(returns) > self.config.realized_vol_window:
            returns = returns.tail(self.config.realized_vol_window)

        # Annualized volatility (assuming daily data)
        vol = returns.std() * np.sqrt(365)

        return vol

    async def _get_options_data(self, symbol: str) -> List[OptionsContract]:
        """
        Get options data from Deribit (mock implementation).
        In production, this would query Deribit API.
        """

        # Mock options data for major crypto pairs
        base_symbol = symbol.replace("USDT", "").replace("USD", "")

        if base_symbol not in ["BTC", "ETH"]:
            return []

        # Mock current spot price
        spot_price = 45000 if base_symbol == "BTC" else 3000

        options = []

        # Generate mock option chains for different expiries
        expiry_days = [7, 10, 14, 21, 28]

        for days in expiry_days:
            expiry = datetime.now() + timedelta(days=days)

            # Generate strikes around ATM
            strikes = [
                spot_price * (1 + i * 0.05) for i in range(-4, 5)  # ±20% strikes
            ]

            for strike in strikes:
                # Mock implied volatility with smile
                moneyness = strike / spot_price
                base_iv = 0.80 if base_symbol == "BTC" else 0.90  # BTC lower vol

                # Add volatility smile
                if moneyness < 0.95:  # ITM puts / OTM calls
                    iv_adjustment = (0.95 - moneyness) * 0.3
                elif moneyness > 1.05:  # OTM puts / ITM calls
                    iv_adjustment = (moneyness - 1.05) * 0.2
                else:
                    iv_adjustment = 0

                iv = base_iv + iv_adjustment

                # Calculate Greeks using Black-Scholes approximation
                time_to_expiry = days / 365.25
                call_greeks = self._calculate_bs_greeks(
                    spot_price, strike, time_to_expiry, 0.05, iv, OptionType.CALL
                )
                put_greeks = self._calculate_bs_greeks(
                    spot_price, strike, time_to_expiry, 0.05, iv, OptionType.PUT
                )

                # Mock prices and market data
                call_price = call_greeks["price"]
                put_price = put_greeks["price"]

                # Mock volume and OI based on moneyness and time
                volume_base = 1000000 if abs(moneyness - 1.0) < 0.05 else 100000
                oi_base = 5000000 if abs(moneyness - 1.0) < 0.05 else 500000

                volume = volume_base * (
                    1 - (days - 7) / 21
                )  # Shorter expiry = more volume
                oi = oi_base * (1 - (days - 7) / 28)

                # Call option
                options.append(
                    OptionsContract(
                        symbol=f"{base_symbol}-{expiry.strftime('%d%b%y')}-{int(strike)}-C",
                        underlying=base_symbol,
                        strike=strike,
                        expiry_date=expiry,
                        option_type=OptionType.CALL,
                        price=call_price,
                        bid=call_price * 0.98,
                        ask=call_price * 1.02,
                        implied_volatility=iv,
                        volume=volume,
                        open_interest=oi,
                        delta=call_greeks["delta"],
                        gamma=call_greeks["gamma"],
                        vega=call_greeks["vega"],
                        theta=call_greeks["theta"],
                        rho=call_greeks["rho"],
                        mark_price=call_price,
                        index_price=spot_price,
                    )
                )

                # Put option
                options.append(
                    OptionsContract(
                        symbol=f"{base_symbol}-{expiry.strftime('%d%b%y')}-{int(strike)}-P",
                        underlying=base_symbol,
                        strike=strike,
                        expiry_date=expiry,
                        option_type=OptionType.PUT,
                        price=put_price,
                        bid=put_price * 0.98,
                        ask=put_price * 1.02,
                        implied_volatility=iv,
                        volume=volume,
                        open_interest=oi,
                        delta=put_greeks["delta"],
                        gamma=put_greeks["gamma"],
                        vega=put_greeks["vega"],
                        theta=put_greeks["theta"],
                        rho=put_greeks["rho"],
                        mark_price=put_price,
                        index_price=spot_price,
                    )
                )

        return options

    def _calculate_bs_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> Dict[str, float]:
        """Calculate Black-Scholes option price and Greeks."""

        if T <= 0:
            return {
                "price": (
                    max(0, S - K) if option_type == OptionType.CALL else max(0, K - S)
                ),
                "delta": 1.0 if (option_type == OptionType.CALL and S > K) else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r
            * K
            * np.exp(-r * T)
            * (norm.cdf(d2) if option_type == OptionType.CALL else norm.cdf(-d2))
        ) / 365  # Per day

        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    def _find_straddle_opportunities(
        self, options: List[OptionsContract], spot_price: float
    ) -> List[Tuple[OptionsContract, OptionsContract]]:
        """Find suitable ATM straddle opportunities."""

        straddles = []

        # Group by expiry
        expiry_groups = {}
        for opt in options:
            expiry_key = opt.expiry_date.date()
            if expiry_key not in expiry_groups:
                expiry_groups[expiry_key] = {"calls": [], "puts": []}

            if opt.option_type == OptionType.CALL:
                expiry_groups[expiry_key]["calls"].append(opt)
            else:
                expiry_groups[expiry_key]["puts"].append(opt)

        # Find ATM straddles for each valid expiry
        for expiry_date, options_by_type in expiry_groups.items():
            days_to_expiry = (
                datetime.combine(expiry_date, datetime.min.time()) - datetime.now()
            ).days

            # Check expiry within target range
            if not (
                self.config.min_days_to_expiry
                <= days_to_expiry
                <= self.config.max_days_to_expiry
            ):
                continue

            calls = options_by_type["calls"]
            puts = options_by_type["puts"]

            # Find ATM strike (closest to spot)
            strikes = set([opt.strike for opt in calls + puts])
            atm_strike = min(strikes, key=lambda x: abs(x - spot_price))

            # Get ATM options
            atm_call = next((c for c in calls if c.strike == atm_strike), None)
            atm_put = next((p for p in puts if p.strike == atm_strike), None)

            if atm_call and atm_put:
                straddles.append((atm_call, atm_put))

        return straddles

    def _calculate_vrp(
        self, call_opt: OptionsContract, put_opt: OptionsContract, realized_vol: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate volatility risk premium for straddle."""

        # Average implied volatility of straddle
        implied_vol = (call_opt.implied_volatility + put_opt.implied_volatility) / 2

        # VRP = IV - RV (positive when IV > RV)
        vrp = implied_vol - realized_vol

        # Check minimum VRP threshold
        if vrp < self.config.vrp_threshold_min:
            return None

        # Calculate premium collected (short straddle)
        premium_collected = call_opt.price + put_opt.price

        # Days to expiry
        days_to_expiry = (call_opt.expiry_date - datetime.now()).days

        return {
            "vrp": vrp,
            "implied_vol": implied_vol,
            "realized_vol": realized_vol,
            "premium_collected": premium_collected,
            "days_to_expiry": days_to_expiry,
            "time_to_expiry": days_to_expiry / 365.25,
        }

    def _meets_liquidity_requirements(
        self, call_opt: OptionsContract, put_opt: OptionsContract
    ) -> bool:
        """Check if options meet minimum liquidity requirements."""

        # Check volume requirements
        if (
            call_opt.volume < self.config.min_volume_24h
            or put_opt.volume < self.config.min_volume_24h
        ):
            return False

        # Check open interest
        if (
            call_opt.open_interest < self.config.min_open_interest
            or put_opt.open_interest < self.config.min_open_interest
        ):
            return False

        # Check bid-ask spreads
        call_spread = (call_opt.ask - call_opt.bid) / call_opt.price
        put_spread = (put_opt.ask - put_opt.bid) / put_opt.price

        if (
            call_spread > self.config.max_bid_ask_spread
            or put_spread > self.config.max_bid_ask_spread
        ):
            return False

        return True

    def _select_best_vrp_opportunity(
        self, opportunities: List[Tuple[OptionsContract, OptionsContract, Dict]]
    ) -> Tuple[OptionsContract, OptionsContract, Dict]:
        """Select best VRP opportunity based on risk-adjusted metrics."""

        def score_opportunity(call_opt, put_opt, vrp_data):
            # Higher VRP = better
            vrp_score = vrp_data["vrp"] / self.config.vrp_threshold_max

            # Prefer target expiry
            days = vrp_data["days_to_expiry"]
            time_score = 1.0 - abs(days - self.config.target_days_to_expiry) / 7
            time_score = max(0.1, time_score)

            # Higher premium = better
            premium_score = min(1.0, vrp_data["premium_collected"] / 1000)  # Normalize

            # Lower gamma risk = better
            total_gamma = abs(call_opt.gamma + put_opt.gamma)
            gamma_score = max(0.1, 1.0 - total_gamma / self.config.max_gamma_exposure)

            # Higher liquidity = better
            min_volume = min(call_opt.volume, put_opt.volume)
            liquidity_score = min(1.0, min_volume / self.config.min_volume_24h)

            return (
                vrp_score * time_score * premium_score * gamma_score * liquidity_score
            )

        scored_opportunities = [
            (
                call_opt,
                put_opt,
                vrp_data,
                score_opportunity(call_opt, put_opt, vrp_data),
            )
            for call_opt, put_opt, vrp_data in opportunities
        ]

        # Sort by score (highest first)
        scored_opportunities.sort(key=lambda x: x[3], reverse=True)

        best_call, best_put, best_vrp, _ = scored_opportunities[0]
        return best_call, best_put, best_vrp

    async def _check_risk_conditions(
        self, symbol: str, call_opt: OptionsContract, put_opt: OptionsContract
    ) -> Dict[str, Any]:
        """Check risk management conditions."""

        # Check position count limits
        active_positions = self.active_positions.get(symbol, [])
        if len(active_positions) >= self.config.max_position_count:
            return {
                "allowed": False,
                "reason": f"Max position count reached: {len(active_positions)}",
            }

        # Check gamma exposure
        total_gamma = abs(call_opt.gamma + put_opt.gamma)
        if total_gamma > self.config.max_gamma_exposure:
            return {
                "allowed": False,
                "reason": f"Gamma exposure too high: {total_gamma:.0f}",
            }

        # Check existing vega exposure
        current_vega = sum(
            pos.call_contract.vega + pos.put_contract.vega for pos in active_positions
        )
        position_vega = call_opt.vega + put_opt.vega

        if (
            abs(current_vega + position_vega) > self.config.max_vega_allocation * 10000
        ):  # Scale
            return {
                "allowed": False,
                "reason": f"Vega limit exceeded: {abs(current_vega + position_vega):.0f}",
            }

        return {
            "allowed": True,
            "reason": "All risk checks passed",
            "current_positions": len(active_positions),
            "current_vega": current_vega,
            "position_gamma": total_gamma,
        }

    async def _get_tail_protection(
        self,
        symbol: str,
        call_opt: OptionsContract,
        put_opt: OptionsContract,
        premium_collected: float,
    ) -> Optional[WingProtection]:
        """Get 25Δ wing protection options."""

        # Mock implementation - find 25Δ options
        base_symbol = symbol.replace("USDT", "").replace("USD", "")
        spot_price = call_opt.index_price

        # Target 25Δ strikes (approximate)
        call_wing_strike = spot_price * 1.15  # ~25Δ call
        put_wing_strike = spot_price * 0.85  # ~25Δ put

        # Mock wing contracts
        wing_call = OptionsContract(
            symbol=f"{base_symbol}-{call_opt.expiry_date.strftime('%d%b%y')}-{int(call_wing_strike)}-C",
            underlying=base_symbol,
            strike=call_wing_strike,
            expiry_date=call_opt.expiry_date,
            option_type=OptionType.CALL,
            price=spot_price * 0.02,  # 2% of spot
            bid=spot_price * 0.019,
            ask=spot_price * 0.021,
            implied_volatility=call_opt.implied_volatility * 1.1,  # Higher IV for wings
            volume=call_opt.volume * 0.3,
            open_interest=call_opt.open_interest * 0.3,
            delta=0.25,
            gamma=call_opt.gamma * 0.5,
            vega=call_opt.vega * 0.3,
            theta=call_opt.theta * 0.2,
            rho=call_opt.rho * 0.3,
            mark_price=spot_price * 0.02,
            index_price=spot_price,
        )

        wing_put = OptionsContract(
            symbol=f"{base_symbol}-{put_opt.expiry_date.strftime('%d%b%y')}-{int(put_wing_strike)}-P",
            underlying=base_symbol,
            strike=put_wing_strike,
            expiry_date=put_opt.expiry_date,
            option_type=OptionType.PUT,
            price=spot_price * 0.02,  # 2% of spot
            bid=spot_price * 0.019,
            ask=spot_price * 0.021,
            implied_volatility=put_opt.implied_volatility * 1.1,
            volume=put_opt.volume * 0.3,
            open_interest=put_opt.open_interest * 0.3,
            delta=-0.25,
            gamma=put_opt.gamma * 0.5,
            vega=put_opt.vega * 0.3,
            theta=put_opt.theta * 0.2,
            rho=put_opt.rho * 0.3,
            mark_price=spot_price * 0.02,
            index_price=spot_price,
        )

        # Calculate wing cost
        wing_cost = wing_call.price + wing_put.price
        cost_ratio = wing_cost / premium_collected

        # Check cost threshold
        if cost_ratio > self.config.max_wing_cost_ratio:
            return None

        return WingProtection(
            call_wing=wing_call,
            put_wing=wing_put,
            total_cost=wing_cost,
            cost_ratio=cost_ratio,
        )

    def _calculate_signal_value(self, vrp_data: Dict[str, Any]) -> float:
        """Calculate signal value based on VRP magnitude."""

        vrp = vrp_data["vrp"]

        # Signal strength based on VRP relative to thresholds
        if vrp >= self.config.vrp_threshold_max:
            return -1.0  # Max short signal (sell vol)
        elif vrp >= self.config.vrp_threshold_min:
            # Scale between min and max thresholds
            strength = vrp / self.config.vrp_threshold_max
            return -min(1.0, strength)  # Negative = sell vol
        else:
            return 0.0  # No signal

    def _calculate_confidence(
        self,
        symbol: str,
        vrp_data: Dict[str, Any],
        call_opt: OptionsContract,
        put_opt: OptionsContract,
    ) -> float:
        """Calculate confidence based on VRP strength and market conditions."""

        # Base confidence from VRP magnitude
        vrp_strength = vrp_data["vrp"] / self.config.vrp_threshold_max
        magnitude_confidence = min(1.0, vrp_strength)

        # Time to expiry factor (prefer target expiry)
        days = vrp_data["days_to_expiry"]
        time_factor = 1.0 - abs(days - self.config.target_days_to_expiry) / 7
        time_factor = max(0.3, time_factor)

        # Liquidity confidence
        min_volume = min(call_opt.volume, put_opt.volume)
        liquidity_factor = min(1.0, min_volume / (self.config.min_volume_24h * 2))

        # VRP persistence factor
        vrp_persistence = self._calculate_vrp_persistence(symbol, vrp_data["vrp"])

        # Bid-ask spread penalty
        call_spread = (call_opt.ask - call_opt.bid) / call_opt.price
        put_spread = (put_opt.ask - put_opt.bid) / put_opt.price
        avg_spread = (call_spread + put_spread) / 2
        spread_factor = max(0.1, 1.0 - avg_spread / self.config.max_bid_ask_spread)

        final_confidence = (
            magnitude_confidence
            * time_factor
            * liquidity_factor
            * vrp_persistence
            * spread_factor
        )

        return max(0.0, min(1.0, final_confidence))

    def _calculate_vega_position_size(
        self, call_opt: OptionsContract, put_opt: OptionsContract, confidence: float
    ) -> float:
        """Calculate position size based on vega exposure and vol targeting."""

        # Total vega per straddle
        total_vega = call_opt.vega + put_opt.vega

        # Target vega based on vol targeting
        target_vega = (
            self.config.max_vega_allocation * 10000
        )  # Scale for realistic values

        # Base position size from vega limits
        base_size = min(1.0, target_vega / abs(total_vega)) if total_vega != 0 else 0

        # Scale by confidence
        confidence_scaled = base_size * confidence

        # Apply volatility targeting adjustment
        implied_vol = (call_opt.implied_volatility + put_opt.implied_volatility) / 2
        vol_adjustment = (
            self.config.vol_target / implied_vol if implied_vol > 0 else 1.0
        )
        vol_adjusted = confidence_scaled * min(1.5, vol_adjustment)  # Cap adjustment

        return max(0.0, min(1.0, vol_adjusted))

    def _calculate_vrp_persistence(self, symbol: str, current_vrp: float) -> float:
        """Calculate VRP persistence factor for confidence."""

        vrp_history = [(datetime.now(), current_vrp)]  # Add current

        if symbol in self.vol_history and symbol in self.iv_history:
            vol_hist = self.vol_history[symbol]
            iv_hist = self.iv_history[symbol]

            # Calculate historical VRP
            for (vol_time, rv), (iv_time, iv) in zip(vol_hist[-10:], iv_hist[-10:]):
                if abs((vol_time - iv_time).total_seconds()) < 3600:  # Within 1 hour
                    historical_vrp = iv - rv
                    vrp_history.append((vol_time, historical_vrp))

        if len(vrp_history) < 3:
            return 0.5  # Neutral for insufficient history

        # Check if VRP has been consistently positive
        recent_vrp = [vrp for _, vrp in vrp_history[-5:]]
        positive_count = sum(1 for vrp in recent_vrp if vrp > 0)
        persistence = positive_count / len(recent_vrp)

        return max(0.1, persistence)

    def _update_vol_history(self, symbol: str, realized_vol: float) -> None:
        """Update realized volatility history."""
        if symbol not in self.vol_history:
            self.vol_history[symbol] = []

        self.vol_history[symbol].append((datetime.now(), realized_vol))

        # Keep last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        self.vol_history[symbol] = [
            (t, v) for t, v in self.vol_history[symbol] if t > cutoff
        ]

    def _update_iv_history(self, symbol: str, implied_vol: float) -> None:
        """Update implied volatility history."""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []

        self.iv_history[symbol].append((datetime.now(), implied_vol))

        # Keep last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        self.iv_history[symbol] = [
            (t, v) for t, v in self.iv_history[symbol] if t > cutoff
        ]

    def _no_signal_result(self, symbol: str, reason: str) -> SignalResult:
        """Helper to create no-signal result."""
        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={"reason": reason, "strategy_type": "vol_risk_premium"},
        )

    async def execute_delta_hedge(
        self, symbol: str, position: StraddlePosition
    ) -> Dict[str, Any]:
        """Execute delta hedging for position (mock implementation)."""

        current_time = datetime.now()

        # Calculate current net delta
        net_delta = position.call_contract.delta + position.put_contract.delta

        # Check if hedging is needed
        if abs(net_delta) < self.config.delta_hedge_threshold:
            return {
                "hedged": False,
                "reason": "Delta within threshold",
                "net_delta": net_delta,
            }

        # Calculate hedge quantity (opposite direction)
        hedge_quantity = -net_delta * position.quantity

        # Mock execution (in production would place actual orders)
        hedge_cost = (
            abs(hedge_quantity)
            * position.call_contract.index_price
            * self.config.hedge_slippage
        )

        # Update position tracking
        hedge_record = {
            "timestamp": current_time,
            "net_delta": net_delta,
            "hedge_quantity": hedge_quantity,
            "hedge_cost": hedge_cost,
            "spot_price": position.call_contract.index_price,
        }

        position.hedge_history.append(hedge_record)
        position.delta_hedge_units += hedge_quantity

        return {
            "hedged": True,
            "net_delta": net_delta,
            "hedge_quantity": hedge_quantity,
            "hedge_cost": hedge_cost,
            "total_hedge_cost": sum(h["hedge_cost"] for h in position.hedge_history),
        }

    def calculate_position_pnl(
        self, position: StraddlePosition, current_spot: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive P&L attribution for position."""

        # Time elapsed
        time_elapsed = (
            datetime.now() - position.entry_time
        ).total_seconds() / 86400  # Days

        # Current option values (mock recalculation)
        call_current = self._calculate_bs_greeks(
            current_spot,
            position.call_contract.strike,
            max(
                0.001,
                (position.call_contract.expiry_date - datetime.now()).days / 365.25,
            ),
            0.05,
            position.call_contract.implied_volatility,
            OptionType.CALL,
        )

        put_current = self._calculate_bs_greeks(
            current_spot,
            position.put_contract.strike,
            max(
                0.001,
                (position.put_contract.expiry_date - datetime.now()).days / 365.25,
            ),
            0.05,
            position.put_contract.implied_volatility,
            OptionType.PUT,
        )

        # P&L components
        premium_pnl = position.premium_collected  # Premium collected upfront
        option_pnl = (
            -(call_current["price"] + put_current["price"]) * position.quantity
        )  # Short position
        hedge_costs = sum(h["hedge_cost"] for h in position.hedge_history)

        # Delta hedge P&L (approximate)
        hedge_pnl = position.delta_hedge_units * (
            current_spot - position.call_contract.index_price
        )

        total_pnl = premium_pnl + option_pnl + hedge_pnl - hedge_costs

        return {
            "total_pnl": total_pnl,
            "premium_collected": premium_pnl,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "hedge_costs": hedge_costs,
            "days_held": time_elapsed,
            "theta_capture": position.call_contract.theta + position.put_contract.theta,
            "current_spot": current_spot,
            "entry_spot": position.call_contract.index_price,
        }

    def get_strategy_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get strategy-specific metrics for monitoring."""

        active_positions = self.active_positions.get(symbol, [])

        metrics = {
            "active_positions": len(active_positions),
            "total_vega": sum(
                pos.call_contract.vega + pos.put_contract.vega
                for pos in active_positions
            ),
            "total_gamma": sum(
                abs(pos.call_contract.gamma + pos.put_contract.gamma)
                for pos in active_positions
            ),
            "total_theta": sum(
                pos.call_contract.theta + pos.put_contract.theta
                for pos in active_positions
            ),
            "total_premium_collected": sum(
                pos.premium_collected for pos in active_positions
            ),
        }

        # Vol history metrics
        if symbol in self.vol_history:
            recent_vol = [v for _, v in self.vol_history[symbol][-10:]]
            metrics.update(
                {
                    "avg_realized_vol": np.mean(recent_vol) if recent_vol else 0,
                    "vol_volatility": np.std(recent_vol) if len(recent_vol) > 1 else 0,
                }
            )

        if symbol in self.iv_history:
            recent_iv = [v for _, v in self.iv_history[symbol][-10:]]
            metrics.update(
                {
                    "avg_implied_vol": np.mean(recent_iv) if recent_iv else 0,
                    "iv_volatility": np.std(recent_iv) if len(recent_iv) > 1 else 0,
                }
            )

        return metrics
