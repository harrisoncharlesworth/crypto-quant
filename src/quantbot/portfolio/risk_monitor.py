"""
Portfolio Risk Monitor for ATR-based position sizing and heat management.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class RiskMetrics:
    """Enhanced real-time risk metrics for portfolio monitoring."""

    # Core metrics
    portfolio_heat: float
    max_heat_limit: float
    heat_utilization_pct: float
    individual_position_risks: Dict[str, float]
    atr_values: Dict[str, float]
    position_sizes: Dict[str, float]
    nav: float
    timestamp: datetime

    # Enhanced accuracy metrics
    correlation_adjusted_heat: float = 0.0
    var_95_1day: float = 0.0  # 95% 1-day Value at Risk
    expected_shortfall: float = 0.0  # Expected Shortfall (Conditional VaR)
    projected_heat_6h: float = 0.0  # 6-hour ahead projection

    # Fee and funding metrics
    total_funding_pnl_24h: float = 0.0
    estimated_fees_24h: float = 0.0
    slippage_adjusted_ear: Dict[str, float] = field(default_factory=dict)

    # Market context
    market_conditions: Dict[str, Any] = field(default_factory=dict)

    # Performance attribution
    signal_attribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sector_exposures: Dict[str, float] = field(default_factory=dict)


class PortfolioRiskMonitor:
    """
    Real-time portfolio risk monitoring with ATR-based heat calculations.

    Tracks:
    - Portfolio heat (total risk if all stops hit)
    - Individual position risks
    - ATR values for position sizing
    - Risk limit compliance
    """

    def __init__(
        self,
        nav: float = 200000.0,
        max_heat_pct: float = 0.08,  # 8% max portfolio heat
        equity_at_risk_pct: float = 0.005,  # 0.5% per trade
        atr_multiplier: float = 1.2,
        single_position_cap_pct: float = 0.05,  # 5% max per position
    ):
        self.nav = nav
        self.max_heat_pct = max_heat_pct
        self.equity_at_risk_pct = equity_at_risk_pct
        self.atr_multiplier = atr_multiplier
        self.single_position_cap_pct = single_position_cap_pct

        self.max_heat_limit = nav * max_heat_pct
        self.single_position_limit = nav * single_position_cap_pct

        # Historical tracking
        self.risk_history: List[RiskMetrics] = []
        self.atr_cache: Dict[str, Tuple[float, datetime]] = {}

        logging.info(
            f"Portfolio Risk Monitor initialized - NAV: ${nav:,.0f}, Max Heat: ${self.max_heat_limit:,.0f}"
        )

    def calculate_portfolio_heat(
        self,
        positions: Dict[str, float],  # symbol -> position size
        market_data: Dict[str, pd.DataFrame],  # symbol -> OHLCV data
    ) -> RiskMetrics:
        """Calculate current portfolio heat and risk metrics."""

        total_heat = 0.0
        individual_risks = {}
        atr_values = {}

        for symbol, position_size in positions.items():
            if abs(position_size) < 0.0001:  # Skip tiny positions
                continue

            try:
                # Calculate ATR for this symbol
                atr = self._calculate_atr(symbol, market_data.get(symbol))
                atr_values[symbol] = atr

                if atr > 0:
                    # Position risk = position_size * ATR * multiplier
                    position_risk = abs(position_size) * atr * self.atr_multiplier
                    individual_risks[symbol] = position_risk
                    total_heat += position_risk
                else:
                    individual_risks[symbol] = 0.0

            except Exception as e:
                logging.warning(f"Risk calculation failed for {symbol}: {e}")
                individual_risks[symbol] = 0.0
                atr_values[symbol] = 0.0

        heat_utilization = (
            (total_heat / self.max_heat_limit) * 100 if self.max_heat_limit > 0 else 0
        )

        metrics = RiskMetrics(
            portfolio_heat=total_heat,
            max_heat_limit=self.max_heat_limit,
            heat_utilization_pct=heat_utilization,
            individual_position_risks=individual_risks,
            atr_values=atr_values,
            position_sizes=positions.copy(),
            nav=self.nav,
            timestamp=datetime.utcnow(),
        )

        # Store in history
        self.risk_history.append(metrics)
        if len(self.risk_history) > 1000:  # Keep last 1000 records
            self.risk_history = self.risk_history[-1000:]

        # Calculate enhanced metrics
        metrics.correlation_adjusted_heat = self._calculate_correlation_adjusted_heat(
            positions, market_data
        )
        metrics.var_95_1day = self._calculate_var_95(positions, market_data)
        metrics.expected_shortfall = self._calculate_expected_shortfall(
            positions, market_data
        )
        metrics.projected_heat_6h = self._calculate_projected_heat(
            positions, market_data
        )
        metrics.market_conditions = self._get_market_conditions()
        metrics.sector_exposures = self._calculate_sector_exposures(positions)

        return metrics

    def check_new_position_risk(
        self,
        symbol: str,
        proposed_size: float,
        current_positions: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position would exceed risk limits.

        Returns:
            (allowed, reason, max_allowed_size)
        """

        # Calculate current heat
        current_metrics = self.calculate_portfolio_heat(current_positions, market_data)

        # Calculate ATR for new position
        try:
            atr = self._calculate_atr(symbol, market_data.get(symbol))
            if atr <= 0:
                return False, "No ATR data available", 0.0
        except Exception as e:
            return False, f"ATR calculation failed: {e}", 0.0

        # Calculate risk of proposed position
        proposed_risk = abs(proposed_size) * atr * self.atr_multiplier

        # Check single position limit
        current_price = self._get_current_price(symbol, market_data.get(symbol))
        if current_price is not None:
            proposed_notional = abs(proposed_size) * current_price
            if proposed_notional > self.single_position_limit:
                max_size_by_cap = self.single_position_limit / current_price
                return (
                    False,
                    f"Exceeds single position cap (${proposed_notional:,.0f} > ${self.single_position_limit:,.0f})",
                    max_size_by_cap,
                )

        # Check portfolio heat limit
        new_total_heat = current_metrics.portfolio_heat + proposed_risk
        if new_total_heat > self.max_heat_limit:
            remaining_heat = self.max_heat_limit - current_metrics.portfolio_heat
            max_size_by_heat = (
                remaining_heat / (atr * self.atr_multiplier)
                if (atr * self.atr_multiplier) > 0
                else 0
            )
            return (
                False,
                f"Would exceed portfolio heat limit (${new_total_heat:,.0f} > ${self.max_heat_limit:,.0f})",
                max_size_by_heat,
            )

        return True, "Position approved", proposed_size

    def calculate_optimal_position_size(
        self,
        symbol: str,
        signal_strength: float,  # -1 to 1
        market_data: Dict[str, pd.DataFrame],
    ) -> float:
        """Calculate optimal position size using ATR-based methodology."""

        try:
            # Get ATR
            atr = self._calculate_atr(symbol, market_data.get(symbol))
            if atr <= 0:
                return 0.0

            # Get current price
            current_price = self._get_current_price(symbol, market_data.get(symbol))
            if current_price is None or current_price <= 0:
                return 0.0

            # Calculate position size based on equity at risk
            ear = self.nav * self.equity_at_risk_pct
            base_position_size = ear / (atr * self.atr_multiplier)

            # Apply signal strength scaling
            scaled_position = base_position_size * abs(signal_strength)

            # Apply single position cap
            max_notional = self.nav * self.single_position_cap_pct
            max_size_by_cap = max_notional / current_price

            final_size = min(scaled_position, max_size_by_cap)

            # Apply direction
            if signal_strength < 0:
                final_size = -final_size

            return final_size

        except Exception as e:
            logging.warning(
                f"Optimal position size calculation failed for {symbol}: {e}"
            )
            return 0.0

    def _calculate_atr(
        self, symbol: str, market_data: Optional[pd.DataFrame], period: int = 10
    ) -> float:
        """Calculate Average True Range with caching."""

        # Check cache first (valid for 1 hour)
        if symbol in self.atr_cache:
            cached_atr, cached_time = self.atr_cache[symbol]
            if datetime.utcnow() - cached_time < timedelta(hours=1):
                return cached_atr

        if market_data is None or len(market_data) < period:
            return 0.0

        try:
            # Get OHLC data
            df = market_data.copy()
            if len(df) < period:
                return 0.0

            # Ensure we have the required columns
            required_cols = ["high", "low", "close"]
            if not all(col in df.columns for col in required_cols):
                return 0.0

            # Take last period rows
            df = df.tail(period)

            # Calculate True Range components
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["close"].shift(1))
            df["tr3"] = abs(df["low"] - df["close"].shift(1))

            # True Range is the maximum of the three
            df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

            # Average True Range
            atr = df["true_range"].mean()

            # Cache the result
            if not pd.isna(atr) and atr > 0:
                self.atr_cache[symbol] = (float(atr), datetime.utcnow())
                return float(atr)

            return 0.0

        except Exception as e:
            logging.warning(f"ATR calculation failed for {symbol}: {e}")
            return 0.0

    def _get_current_price(
        self, symbol: str, market_data: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """Get current price from market data."""
        if market_data is None or len(market_data) == 0:
            return None

        try:
            if "close" in market_data.columns:
                return float(market_data["close"].iloc[-1])
        except Exception:
            pass

        return None

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary for monitoring dashboard."""
        if not self.risk_history:
            return {"status": "No risk data available"}

        latest = self.risk_history[-1]

        return {
            "portfolio_heat": latest.portfolio_heat,
            "heat_limit": latest.max_heat_limit,
            "heat_utilization_pct": latest.heat_utilization_pct,
            "positions_count": len(
                [p for p in latest.position_sizes.values() if abs(p) > 0.0001]
            ),
            "largest_position_risk": (
                max(latest.individual_position_risks.values())
                if latest.individual_position_risks
                else 0
            ),
            "nav": latest.nav,
            "status": (
                "HEALTHY"
                if latest.heat_utilization_pct < 80
                else "WARNING" if latest.heat_utilization_pct < 100 else "CRITICAL"
            ),
            "timestamp": latest.timestamp.isoformat(),
        }

    def update_nav(self, new_nav: float):
        """Update NAV and recalculate limits."""
        self.nav = new_nav
        self.max_heat_limit = new_nav * self.max_heat_pct
        self.single_position_limit = new_nav * self.single_position_cap_pct

        logging.info(
            f"NAV updated to ${new_nav:,.0f}, Max Heat: ${self.max_heat_limit:,.0f}"
        )

    def _calculate_correlation_adjusted_heat(
        self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate correlation-adjusted portfolio heat using 90-day rolling correlations."""
        try:
            if len(positions) < 2:
                return sum(
                    self._calculate_atr(symbol, market_data.get(symbol))
                    * abs(size)
                    * self.atr_multiplier
                    for symbol, size in positions.items()
                )

            symbols = list(positions.keys())
            returns_matrix = []

            # Calculate 30-day returns for correlation (simplified)
            for symbol in symbols:
                data = market_data.get(symbol)
                if data is not None and len(data) >= 30:
                    returns = data["close"].pct_change().dropna().tail(30)
                    returns_matrix.append(returns.values)

            if len(returns_matrix) < 2:
                return 0.0

            # Calculate correlation matrix
            import numpy as np

            returns_df = pd.DataFrame(returns_matrix).T
            corr_matrix = returns_df.corr().fillna(0).values

            # Weight vector (position sizes * ATR * multiplier)
            weights = []
            for symbol in symbols:
                atr = self._calculate_atr(symbol, market_data.get(symbol))
                weight = abs(positions[symbol]) * atr * self.atr_multiplier
                weights.append(weight)

            weights = np.array(weights)

            # Portfolio variance: w^T * Σ * w
            portfolio_variance = np.dot(weights, np.dot(corr_matrix, weights))

            return float(np.sqrt(portfolio_variance)) if portfolio_variance > 0 else 0.0

        except Exception as e:
            logging.warning(f"Correlation-adjusted heat calculation failed: {e}")
            return 0.0

    def _calculate_var_95(
        self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate 95% 1-day Value at Risk using parametric method."""
        try:
            import numpy as np

            total_portfolio_value = 0.0
            portfolio_volatility = 0.0

            for symbol, size in positions.items():
                if abs(size) < 0.0001:
                    continue

                data = market_data.get(symbol)
                if data is None or len(data) < 30:
                    continue

                # Calculate 30-day volatility
                returns = data["close"].pct_change().dropna().tail(30)
                if len(returns) < 10:
                    continue

                daily_vol = returns.std()
                current_price = data["close"].iloc[-1] if len(data) > 0 else 0
                position_value = abs(size) * current_price

                total_portfolio_value += position_value
                portfolio_volatility += (position_value * daily_vol) ** 2

            if total_portfolio_value == 0:
                return 0.0

            portfolio_volatility = np.sqrt(portfolio_volatility)
            portfolio_vol_pct = (
                portfolio_volatility / total_portfolio_value
                if total_portfolio_value > 0
                else 0
            )

            # 95% VaR = 1.645 * σ * Portfolio Value
            var_95 = 1.645 * portfolio_vol_pct * total_portfolio_value

            return float(var_95)

        except Exception as e:
            logging.warning(f"VaR calculation failed: {e}")
            return 0.0

    def _calculate_expected_shortfall(
        self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            var_95 = self._calculate_var_95(positions, market_data)
            # ES ≈ VaR / 0.95 for normal distribution (simplified)
            return var_95 * 1.05 if var_95 > 0 else 0.0
        except Exception as e:
            logging.warning(f"Expected Shortfall calculation failed: {e}")
            return 0.0

    def _calculate_projected_heat(
        self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate projected portfolio heat 6 hours ahead."""
        try:
            import numpy as np

            projected_heat = 0.0
            sqrt_6h = np.sqrt(6 / 24)  # 6 hours as fraction of day

            for symbol, size in positions.items():
                if abs(size) < 0.0001:
                    continue

                data = market_data.get(symbol)
                if data is None or len(data) < 10:
                    continue

                # Use ATR as volatility proxy
                atr = self._calculate_atr(symbol, data)
                if atr <= 0:
                    continue

                # Project ATR forward by 6 hours with vol scaling
                projected_atr = atr * sqrt_6h
                projected_risk = abs(size) * projected_atr * self.atr_multiplier
                projected_heat += projected_risk

            return projected_heat

        except Exception as e:
            logging.warning(f"Projected heat calculation failed: {e}")
            return 0.0

    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions and volatility metrics."""
        try:
            # Simplified market conditions - can be enhanced with external APIs
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "market_session": self._get_market_session(),
                "volatility_regime": "normal",  # Can be enhanced with VIX-like indicators
                "funding_environment": "neutral",  # Can be enhanced with funding rate data
            }
        except Exception as e:
            logging.warning(f"Market conditions fetch failed: {e}")
            return {}

    def _get_market_session(self) -> str:
        """Determine current market session."""
        utc_hour = datetime.utcnow().hour
        if 0 <= utc_hour < 8:
            return "asia"
        elif 8 <= utc_hour < 16:
            return "europe"
        else:
            return "americas"

    def _calculate_sector_exposures(
        self, positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate exposure by crypto sectors."""
        try:
            # Simplified sector mapping - can be enhanced with external data
            sector_map = {
                "BTC": "store_of_value",
                "ETH": "smart_contract",
                "BNB": "exchange",
                "SOL": "layer1",
                "ADA": "layer1",
                "DOT": "interoperability",
                "AVAX": "layer1",
                "LINK": "oracle",
                "UNI": "defi",
                "DOGE": "meme",
                "SHIB": "meme",
                "MATIC": "layer2",
                "OP": "layer2",
                "ARB": "layer2",
            }

            sector_exposures = {}
            total_exposure = sum(abs(size) for size in positions.values())

            if total_exposure == 0:
                return {}

            for symbol, size in positions.items():
                # Extract base symbol (remove USD suffix)
                base_symbol = symbol.replace("USD", "").replace("USDT", "")
                sector = sector_map.get(base_symbol, "other")

                exposure_pct = abs(size) / total_exposure * 100
                sector_exposures[sector] = (
                    sector_exposures.get(sector, 0) + exposure_pct
                )

            return sector_exposures

        except Exception as e:
            logging.warning(f"Sector exposure calculation failed: {e}")
            return {}
