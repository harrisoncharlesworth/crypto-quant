import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class CrossSectionalConfig(SignalConfig):
    """Configuration for cross-sectional momentum signal."""

    lookback_days: int = 30  # 30-day lookback for ranking
    hold_days: int = 7  # 7-day rebalancing period
    ranking_period: int = 180  # 6-month performance ranking (6M as per research)
    min_periods: int = 200  # Minimum data required
    universe_size: int = 10  # Number of alts in universe
    decile_threshold: float = 0.2  # Top/bottom 20% (decile approximation)
    beta_hedge_ratio: float = 1.0  # BTC hedge ratio for market neutrality
    min_spread: float = 0.05  # Minimum spread for confidence (5%)
    correlation_threshold: float = 0.7  # Correlation breakdown threshold
    position_cap: float = 0.1  # Maximum position size per asset (10%)
    rebalance_threshold: float = 0.05  # Minimum change to trigger rebalance


class AltBTCCrossSectionalSignal(SignalBase):
    """
    Alt/BTC Cross-Sectional Momentum Signal - Market Neutral Strategy.

    Ranks altcoins by their 6-month performance relative to BTC, goes long
    top-decile performers and short bottom-decile. Hedges market beta with
    BTC perpetual short to maintain market neutrality.

    Key Features:
    - 30-day lookback with 7-day rebalancing
    - Alt/BTC ratio momentum ranking
    - Top vs bottom decile spread targeting >20% p.a.
    - Market-neutral via BTC hedge
    - Momentum persistence detection
    - Risk management for correlation breakdowns
    """

    def __init__(self, config: CrossSectionalConfig):
        super().__init__(config)
        self.config: CrossSectionalConfig = config
        self.universe = self._get_altcoin_universe()
        self.rankings_cache: Dict[str, Dict] = {}
        self.last_rebalance: datetime = datetime.min
        self.position_cache: Dict[str, float] = {}

    def _get_altcoin_universe(self) -> List[str]:
        """Define the universe of major altcoins for ranking."""
        # Major altcoins with good liquidity and BTC correlation
        return [
            "ETHUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "MATICUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "BCHUSDT",
            "XRPUSDT",
            "AVAXUSDT",
        ]

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate cross-sectional momentum signal for given symbol."""

        if not self.validate_data(data, self.config.min_periods):
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # For BTC, return hedge signal (opposite of alt exposure)
            if symbol in ["BTCUSDT", "BTCUSD"]:
                return await self._generate_btc_hedge_signal(symbol)

            # For alts, generate ranking-based signal
            if symbol in self.universe:
                return await self._generate_alt_signal(data, symbol)

            # For non-universe assets, return neutral
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"status": "not_in_universe"},
            )

        except Exception as e:
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def _generate_alt_signal(
        self, data: pd.DataFrame, symbol: str
    ) -> SignalResult:
        """Generate signal for altcoin based on cross-sectional ranking."""

        # Mock multi-asset data for demonstration (in production, get from data layer)
        multi_asset_data = self._mock_multi_asset_data(data, symbol)

        # Calculate alt/BTC ratios and performance
        alt_btc_performance = self._calculate_alt_btc_performance(
            multi_asset_data, symbol
        )

        # Perform cross-sectional ranking
        rankings = self._perform_cross_sectional_ranking(multi_asset_data)

        # Get signal based on ranking
        signal_value, confidence = self._calculate_ranking_signal(
            symbol, rankings, alt_btc_performance
        )

        # Apply momentum persistence filter
        signal_value = self._apply_momentum_filter(signal_value, alt_btc_performance)

        # Apply correlation breakdown risk management
        signal_value, confidence = self._apply_correlation_filter(
            signal_value, confidence, multi_asset_data, symbol
        )

        # Calculate position size with risk management
        position_size = self._calculate_position_size(
            signal_value, confidence, alt_btc_performance
        )

        # Update position cache for BTC hedging
        self.position_cache[symbol] = position_size

        # Metadata for analysis and debugging
        metadata = {
            "alt_btc_performance": alt_btc_performance,
            "ranking": rankings.get(symbol, {}),
            "position_size": position_size,
            "universe_size": len(self.universe),
            "rebalance_due": self._is_rebalance_due(),
            "lookback_days": self.config.lookback_days,
            "ranking_period": self.config.ranking_period,
        }

        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=position_size,  # Use position-sized signal
            confidence=confidence,
            metadata=metadata,
        )

    async def _generate_btc_hedge_signal(self, symbol: str) -> SignalResult:
        """Generate BTC hedge signal for market neutrality."""

        # Calculate net alt exposure to determine hedge requirement
        net_alt_exposure = sum(self.position_cache.values())

        # Hedge with opposite BTC position (market neutral)
        btc_hedge_signal = -net_alt_exposure * self.config.beta_hedge_ratio

        # Apply position cap
        btc_hedge_signal = max(
            -self.config.position_cap, min(self.config.position_cap, btc_hedge_signal)
        )

        # Confidence based on alt exposure magnitude
        confidence = min(1.0, abs(net_alt_exposure) * 2)

        metadata = {
            "hedge_type": "market_neutral",
            "net_alt_exposure": net_alt_exposure,
            "beta_hedge_ratio": self.config.beta_hedge_ratio,
            "hedge_signal": btc_hedge_signal,
        }

        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=btc_hedge_signal,
            confidence=confidence,
            metadata=metadata,
        )

    def _mock_multi_asset_data(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Mock multi-asset data structure for demonstration.
        In production, this would come from the data infrastructure.
        """
        # Create mock data based on the provided data with correlation patterns
        np.random.seed(42)  # For reproducible results

        multi_data = {"BTCUSDT": data.copy()}

        for alt_symbol in self.universe:
            if alt_symbol == symbol:
                multi_data[alt_symbol] = data.copy()
            else:
                # Generate correlated price series for mock data
                btc_returns = data["close"].pct_change().fillna(0)

                # Different correlation levels for different alts
                correlation = 0.6 + np.random.normal(0, 0.1)  # 0.5-0.7 correlation
                alt_specific = np.random.normal(
                    0, 0.02, len(data)
                )  # Alt-specific noise

                alt_returns = (
                    correlation * btc_returns + (1 - correlation) * alt_specific
                )

                # Construct price series
                alt_prices = data["close"].iloc[0] * (1 + alt_returns).cumprod()

                alt_data = data.copy()
                alt_data["close"] = alt_prices
                alt_data["high"] = alt_prices * 1.02  # Mock high
                alt_data["low"] = alt_prices * 0.98  # Mock low

                multi_data[alt_symbol] = alt_data

        return multi_data

    def _calculate_alt_btc_performance(
        self, multi_data: Dict[str, pd.DataFrame], symbol: str
    ) -> Dict[str, float]:
        """Calculate alt/BTC relative performance metrics."""

        btc_data = multi_data.get("BTCUSDT")
        alt_data = multi_data.get(symbol)

        if btc_data is None or alt_data is None:
            return {"6m_return": 0.0, "30d_return": 0.0, "momentum_score": 0.0}

        # Calculate performance periods
        periods = {
            "6m_return": min(self.config.ranking_period, len(alt_data)),
            "30d_return": min(self.config.lookback_days, len(alt_data)),
        }

        performance = {}

        for period_name, period_days in periods.items():
            if period_days < 5:  # Need minimum data
                performance[period_name] = 0.0
                continue

            # Alt and BTC returns over period
            alt_start = alt_data.iloc[-period_days]["close"]
            alt_end = alt_data.iloc[-1]["close"]
            alt_return = (alt_end - alt_start) / alt_start

            btc_start = btc_data.iloc[-period_days]["close"]
            btc_end = btc_data.iloc[-1]["close"]
            btc_return = (btc_end - btc_start) / btc_start

            # Relative performance (alt vs BTC)
            relative_performance = alt_return - btc_return
            performance[period_name] = relative_performance

        # Momentum score combining both periods
        momentum_score = 0.7 * performance.get(
            "6m_return", 0.0
        ) + 0.3 * performance.get("30d_return", 0.0)
        performance["momentum_score"] = momentum_score

        return performance

    def _perform_cross_sectional_ranking(
        self, multi_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Perform cross-sectional ranking of all universe assets."""

        rankings = {}
        momentum_scores = {}

        # Calculate momentum scores for all assets
        for alt_symbol in self.universe:
            if alt_symbol in multi_data:
                performance = self._calculate_alt_btc_performance(
                    multi_data, alt_symbol
                )
                momentum_scores[alt_symbol] = performance["momentum_score"]

        # Rank assets by momentum score
        sorted_assets = sorted(
            momentum_scores.items(), key=lambda x: x[1], reverse=True
        )

        n_assets = len(sorted_assets)
        if n_assets == 0:
            return rankings

        # Calculate decile thresholds
        top_decile_size = max(1, int(n_assets * self.config.decile_threshold))
        bottom_decile_size = max(1, int(n_assets * self.config.decile_threshold))

        for i, (symbol, score) in enumerate(sorted_assets):
            rank_percentile = (
                (n_assets - i - 1) / (n_assets - 1) if n_assets > 1 else 0.5
            )

            # Determine decile classification
            if i < top_decile_size:
                decile = "top"
                decile_strength = 1.0 - (i / top_decile_size)
            elif i >= n_assets - bottom_decile_size:
                decile = "bottom"
                decile_strength = (
                    i - (n_assets - bottom_decile_size)
                ) / bottom_decile_size
            else:
                decile = "middle"
                decile_strength = 0.0

            rankings[symbol] = {
                "rank": i + 1,
                "percentile": rank_percentile,
                "momentum_score": score,
                "decile": decile,
                "decile_strength": decile_strength,
            }

        return rankings

    def _calculate_ranking_signal(
        self, symbol: str, rankings: Dict[str, Dict], performance: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate signal value and confidence based on ranking."""

        ranking_info = rankings.get(symbol, {})

        if not ranking_info:
            return 0.0, 0.0

        decile = ranking_info.get("decile", "middle")
        decile_strength = ranking_info.get("decile_strength", 0.0)
        momentum_score = ranking_info.get("momentum_score", 0.0)

        # Base signal from decile classification
        if decile == "top":
            signal_value = decile_strength  # Long top performers
        elif decile == "bottom":
            signal_value = -decile_strength  # Short bottom performers
        else:
            signal_value = 0.0  # Neutral for middle decile

        # Scale by momentum magnitude
        signal_value *= min(1.0, abs(momentum_score) * 10)

        # Calculate confidence
        spread_magnitude = abs(momentum_score)
        ranking_clarity = abs(decile_strength)
        confidence = min(1.0, spread_magnitude * 5 + ranking_clarity * 0.5)

        # Boost confidence if spread is significant (>5%)
        if spread_magnitude > self.config.min_spread:
            confidence *= 1.5
            confidence = min(1.0, confidence)

        return signal_value, confidence

    def _apply_momentum_filter(
        self, signal_value: float, performance: Dict[str, float]
    ) -> float:
        """Apply momentum persistence filter."""

        # Check for momentum persistence across timeframes
        short_term_momentum = performance.get("30d_return", 0.0)
        long_term_momentum = performance.get("6m_return", 0.0)

        # Reduce signal if short and long term momentum diverge
        momentum_consistency = 1.0
        if short_term_momentum * long_term_momentum < 0:  # Opposite signs
            momentum_consistency = 0.5  # Reduce signal strength

        return signal_value * momentum_consistency

    def _apply_correlation_filter(
        self,
        signal_value: float,
        confidence: float,
        multi_data: Dict[str, pd.DataFrame],
        symbol: str,
    ) -> Tuple[float, float]:
        """Apply correlation breakdown risk management."""

        try:
            btc_data = multi_data.get("BTCUSDT")
            alt_data = multi_data.get(symbol)

            if btc_data is None or alt_data is None:
                return signal_value, confidence

            # Calculate recent correlation
            lookback = min(30, len(alt_data))
            btc_returns = btc_data["close"].pct_change().tail(lookback)
            alt_returns = alt_data["close"].pct_change().tail(lookback)

            correlation = btc_returns.corr(alt_returns)

            # Reduce signal if correlation breaks down (too high or negative)
            if abs(correlation) > self.config.correlation_threshold:
                # High correlation reduces cross-sectional opportunity
                signal_value *= 0.5
                confidence *= 0.7
            elif correlation < 0:
                # Negative correlation is unusual, reduce confidence
                confidence *= 0.8

            return signal_value, confidence

        except Exception:
            return signal_value, confidence

    def _calculate_position_size(
        self, signal_value: float, confidence: float, performance: Dict[str, float]
    ) -> float:
        """Calculate position size with risk management."""

        # Base position from signal and confidence
        base_position = signal_value * confidence

        # Scale by momentum magnitude (higher conviction for stronger moves)
        momentum_magnitude = abs(performance.get("momentum_score", 0.0))
        momentum_scaling = min(1.5, 1.0 + momentum_magnitude * 2)

        position = base_position * momentum_scaling

        # Apply position cap
        position = max(
            -self.config.position_cap, min(self.config.position_cap, position)
        )

        # Update position cache for BTC hedging (use symbol as key, not signal_value)
        # Note: This should be called from the main generate() method with symbol

        return position

    def _is_rebalance_due(self) -> bool:
        """Check if rebalancing is due based on time and threshold."""

        time_since_rebalance = datetime.utcnow() - self.last_rebalance
        return time_since_rebalance.days >= self.config.hold_days

    def calculate_portfolio_metrics(
        self, rankings: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics for monitoring."""

        if not rankings:
            return {}

        # Extract momentum scores
        scores = [r["momentum_score"] for r in rankings.values()]

        if not scores:
            return {}

        # Top vs bottom decile spread
        sorted_scores = sorted(scores, reverse=True)
        n_decile = max(1, len(sorted_scores) // 10)

        top_decile_avg = np.mean(sorted_scores[:n_decile]) if n_decile > 0 else 0
        bottom_decile_avg = np.mean(sorted_scores[-n_decile:]) if n_decile > 0 else 0

        spread = top_decile_avg - bottom_decile_avg
        spread_annual = spread * np.sqrt(252)  # Annualized

        return {
            "top_decile_avg": float(top_decile_avg),
            "bottom_decile_avg": float(bottom_decile_avg),
            "spread": float(spread),
            "spread_annual": float(spread_annual),
            "universe_size": float(len(rankings)),
            "momentum_std": float(np.std(scores)),
        }


@dataclass
class CrossSectionalMomentumConfig(SignalConfig):
    """Enhanced configuration for cross-sectional momentum with multi-timeframe analysis."""

    # Core parameters
    ranking_lookback: int = 180  # 6-month ranking period
    signal_lookback: int = 30  # 30-day signal lookback
    rebalance_frequency: int = 7  # 7-day rebalancing

    # Universe definition
    min_universe_size: int = 8  # Minimum assets for ranking
    max_universe_size: int = 15  # Maximum assets to avoid overfit

    # Signal generation
    top_decile_pct: float = 0.2  # Top 20%
    bottom_decile_pct: float = 0.2  # Bottom 20%
    min_ranking_spread: float = 0.1  # 10% minimum spread for signal

    # Risk management
    max_position_size: float = 0.15  # 15% max per position
    correlation_cap: float = 0.8  # Max correlation for independence
    volatility_target: float = 0.12  # 12% annual volatility target

    # Market neutral parameters
    hedge_ratio: float = 1.0  # Full hedge ratio
    hedge_rebalance_threshold: float = 0.02  # 2% drift threshold


class EnhancedCrossSectionalSignal(SignalBase):
    """
    Enhanced cross-sectional momentum with multi-timeframe analysis and
    improved risk management for production use.
    """

    def __init__(self, config: CrossSectionalMomentumConfig):
        super().__init__(config)
        self.config: CrossSectionalMomentumConfig = config

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Enhanced signal generation with multi-timeframe ranking."""

        # Implementation would include:
        # 1. Multi-timeframe momentum calculation
        # 2. Risk-adjusted ranking (Sharpe-based)
        # 3. Dynamic universe selection
        # 4. Volatility targeting
        # 5. Enhanced market-neutral hedging

        return SignalResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            value=0.0,
            confidence=0.0,
            metadata={"status": "enhanced_implementation_placeholder"},
        )
