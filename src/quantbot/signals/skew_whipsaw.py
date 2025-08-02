import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from .base import SignalBase, SignalResult, SignalConfig


@dataclass
class SkewWhipsawConfig(SignalConfig):
    """Configuration for 25Δ Skew Whipsaw signal."""

    skew_threshold: float = 15.0  # +15 skew threshold for signal activation
    vol_peak_lookback: int = 24  # Hours to look back for vol peak detection
    max_iv_exposure: float = 0.5  # Max 50% of current IV for position sizing
    spread_width_pct: float = 0.05  # 5% OTM for vertical spread width
    min_confidence_iv: float = 0.20  # Minimum IV for signal confidence
    max_confidence_iv: float = 0.80  # Maximum IV for signal confidence
    skew_mean_reversion_period: int = 48  # Hours for skew mean reversion
    volume_spike_threshold: float = 2.0  # 2x average volume for ETF headlines
    min_time_to_expiry: int = 168  # Minimum 7 days to expiry for options


class SkewWhipsawSignal(SignalBase):
    """
    25Δ Skew Whipsaw Signal - Options volatility strategy.

    Fades extreme skew (>+15) via contrarian positioning after vol peaks.
    Uses vertical spreads to bound loss and size positions relative to IV.

    Strategy Logic:
    1. Monitor 25Δ put-call skew for extreme readings (>+15)
    2. Detect volatility peaks via volume spikes and IV expansion
    3. Generate contrarian signal to fade skew back to mean
    4. Use vertical spreads for position construction with bounded loss
    5. Size positions up to 50% of current IV levels
    """

    def __init__(self, config: SkewWhipsawConfig):
        super().__init__(config)
        self.config: SkewWhipsawConfig = config
        self.skew_history: List[Tuple[datetime, float]] = []
        self.vol_spike_cache: Dict[str, float] = {}

    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """Generate skew whipsaw signal from options market data."""

        if not self.validate_data(data, 48):  # Need 48 hours minimum
            return SignalResult(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                value=0.0,
                confidence=0.0,
                metadata={"error": "Insufficient data"},
            )

        try:
            # Generate mock options data (replace with real Deribit data)
            options_data = self._generate_mock_options_data(data, symbol)

            # Calculate 25Δ skew
            skew_25d = self._calculate_25delta_skew(options_data)

            # Detect volatility peak
            vol_peak_score = self._detect_volatility_peak(data, options_data)

            # Check for ETF headline events (volume spikes)
            headline_score = self._detect_headline_events(data)

            # Calculate skew mean reversion tendency
            skew_reversion_score = self._calculate_skew_mean_reversion(skew_25d)

            # Generate contrarian signal if skew > threshold
            signal_value = 0.0
            if abs(skew_25d) > self.config.skew_threshold:
                # Contrarian signal: fade the skew
                raw_signal = -np.sign(skew_25d) * (abs(skew_25d) / 100.0)

                # Enhance signal strength after vol peaks
                vol_enhancement = 1.0 + (vol_peak_score * 0.5)

                # Enhance signal during headline events
                headline_enhancement = 1.0 + (headline_score * 0.3)

                # Apply mean reversion tendency
                reversion_factor = 1.0 + skew_reversion_score

                signal_value = (
                    raw_signal
                    * vol_enhancement
                    * headline_enhancement
                    * reversion_factor
                )

            # Apply volatility-based position sizing
            current_iv = options_data.get("atm_iv", 0.5)
            vol_sizing_factor = min(1.0, current_iv / self.config.max_confidence_iv)
            signal_value *= vol_sizing_factor

            # Bound signal to [-1, 1]
            signal_value = max(-1.0, min(1.0, signal_value))

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                skew_25d, current_iv, vol_peak_score, headline_score, options_data
            )

            # Create vertical spread recommendation
            spread_recommendation = self._design_vertical_spread(
                signal_value, current_iv, options_data, symbol
            )

            # Position sizing based on IV exposure limits
            position_size = self._calculate_position_size(current_iv, signal_value)

            # Update skew history for mean reversion analysis
            self._update_skew_history(skew_25d)

            metadata = {
                "skew_25d": skew_25d,
                "atm_iv": current_iv,
                "vol_peak_score": vol_peak_score,
                "headline_score": headline_score,
                "skew_reversion_score": skew_reversion_score,
                "vol_sizing_factor": vol_sizing_factor,
                "spread_recommendation": spread_recommendation,
                "position_size": position_size,
                "time_to_expiry": options_data.get("time_to_expiry", 0),
                "strategy_type": "skew_mean_reversion",
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

    def _generate_mock_options_data(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """Generate realistic mock options data with skew patterns."""
        try:
            # Current price for options calculations
            current_price = data.iloc[-1]["close"]

            # Calculate realized volatility for IV simulation
            returns = data["close"].pct_change().dropna()
            realized_vol = float(
                returns.tail(168).std() * np.sqrt(24 * 365)
            )  # Annualized

            # Simulate IV with regime-dependent behavior
            base_iv = max(0.3, min(1.5, realized_vol * 1.2))  # IV typically > RV

            # Add volatility clustering and spikes
            vol_spike = 1.0
            recent_returns = returns.tail(24).abs()
            if (
                len(recent_returns) > 0
                and recent_returns.mean() > returns.tail(168).abs().mean() * 2
            ):
                vol_spike = 1.5  # IV spikes during stress

            atm_iv = base_iv * vol_spike

            # Generate realistic skew patterns
            # During stress: put skew increases (fear premium)
            stress_factor = (
                min(2.0, recent_returns.mean() * 100) if len(recent_returns) > 0 else 0
            )

            # Base skew: usually slightly positive (put skew)
            base_skew = 5.0 + stress_factor * 3.0

            # Add cyclical and mean-reverting component
            time_factor = (len(data) % 168) / 168.0  # Weekly cycle
            cyclical_skew = 5.0 * np.sin(time_factor * 2 * np.pi)

            # Random noise
            noise = np.random.normal(0, 2.0)

            skew_25d = base_skew + cyclical_skew + noise

            # Ensure extreme skews occasionally for testing
            if np.random.random() < 0.1:  # 10% chance of extreme skew
                skew_25d *= 2.0

            # Time to expiry (typically use near-term options 7-30 days)
            time_to_expiry = 14 * 24  # 14 days in hours

            # Strike prices for vertical spreads
            otm_call_strike = current_price * (1 + self.config.spread_width_pct)
            otm_put_strike = current_price * (1 - self.config.spread_width_pct)

            return {
                "current_price": current_price,
                "atm_iv": atm_iv,
                "skew_25d": skew_25d,
                "call_25d_iv": atm_iv - (skew_25d / 100.0),
                "put_25d_iv": atm_iv + (skew_25d / 100.0),
                "time_to_expiry": time_to_expiry,
                "otm_call_strike": otm_call_strike,
                "otm_put_strike": otm_put_strike,
                "realized_vol": realized_vol,
                "vol_spike_factor": vol_spike,
            }

        except Exception:
            # Fallback values
            return {
                "current_price": 50000.0,
                "atm_iv": 0.5,
                "skew_25d": 8.0,
                "call_25d_iv": 0.48,
                "put_25d_iv": 0.52,
                "time_to_expiry": 168,
                "otm_call_strike": 52500.0,
                "otm_put_strike": 47500.0,
                "realized_vol": 0.4,
                "vol_spike_factor": 1.0,
            }

    def _calculate_25delta_skew(self, options_data: Dict[str, Any]) -> float:
        """Calculate 25-delta put-call implied volatility skew."""
        put_25d_iv = options_data.get("put_25d_iv", 0.5)
        call_25d_iv = options_data.get("call_25d_iv", 0.5)

        # Skew = Put IV - Call IV (in vol points, convert to percentage)
        skew = (put_25d_iv - call_25d_iv) * 100

        return skew

    def _detect_volatility_peak(
        self, data: pd.DataFrame, options_data: Dict[str, Any]
    ) -> float:
        """Detect volatility peaks that precede skew mean reversion."""
        try:
            # Use recent price volatility as proxy for vol peak
            returns = data["close"].pct_change().dropna()

            if len(returns) < self.config.vol_peak_lookback:
                return 0.0

            # Recent volatility vs historical average
            recent_vol = returns.tail(self.config.vol_peak_lookback).std()
            historical_vol = returns.tail(168).std()  # 1 week baseline

            if historical_vol <= 0:
                return 0.0

            vol_ratio = recent_vol / historical_vol

            # Peak detection: recent vol significantly higher than average
            peak_score = max(
                0.0, min(1.0, (vol_ratio - 1.0) / 1.0)
            )  # Normalize to [0,1]

            # Enhanced by IV spike
            iv_spike = options_data.get("vol_spike_factor", 1.0)
            if iv_spike > 1.2:
                peak_score *= 1.5

            return peak_score

        except Exception:
            return 0.0

    def _detect_headline_events(self, data: pd.DataFrame) -> float:
        """Detect ETF headline events via volume spike analysis."""
        try:
            if len(data) < 48:
                return 0.0

            # Recent volume vs average
            recent_volume = data["volume"].tail(12).mean()  # Last 12 hours
            avg_volume = data["volume"].tail(168).mean()  # Weekly average

            if avg_volume <= 0:
                return 0.0

            volume_ratio = recent_volume / avg_volume

            # Headline events typically show 2x+ volume spikes
            headline_score = max(
                0.0, min(1.0, (volume_ratio - self.config.volume_spike_threshold) / 2.0)
            )

            return headline_score

        except Exception:
            return 0.0

    def _calculate_skew_mean_reversion(self, current_skew: float) -> float:
        """Calculate skew mean reversion tendency."""
        try:
            # Historical mean skew (typically around 8-12 for crypto)
            historical_mean = 10.0

            # Distance from mean
            skew_deviation = abs(current_skew - historical_mean)

            # Mean reversion tendency: stronger for larger deviations
            reversion_score = min(1.0, skew_deviation / 20.0)

            # Check recent skew history for momentum
            if len(self.skew_history) >= 5:
                recent_skews = [skew for _, skew in self.skew_history[-5:]]
                skew_momentum = np.polyfit(range(5), recent_skews, 1)[0]

                # If skew is moving away from mean, reversion more likely
                if (current_skew > historical_mean and skew_momentum > 0) or (
                    current_skew < historical_mean and skew_momentum < 0
                ):
                    reversion_score *= 1.3

            return reversion_score

        except Exception:
            return 0.5

    def _calculate_confidence(
        self,
        skew_25d: float,
        current_iv: float,
        vol_peak_score: float,
        headline_score: float,
        options_data: Dict[str, Any],
    ) -> float:
        """Calculate signal confidence based on multiple factors."""
        try:
            # Base confidence from skew extremity
            skew_confidence = min(1.0, abs(skew_25d) / 30.0)  # Max at ±30 skew

            # IV level confidence (prefer moderate IV levels)
            iv_confidence = 1.0
            if current_iv < self.config.min_confidence_iv:
                iv_confidence = current_iv / self.config.min_confidence_iv
            elif current_iv > self.config.max_confidence_iv:
                iv_confidence = self.config.max_confidence_iv / current_iv

            # Time to expiry confidence (avoid very short expiry)
            time_to_expiry = options_data.get("time_to_expiry", 168)
            time_confidence = min(1.0, time_to_expiry / self.config.min_time_to_expiry)

            # Volatility peak enhances confidence
            vol_confidence = 0.7 + (vol_peak_score * 0.3)

            # Headline events enhance confidence
            headline_confidence = 0.8 + (headline_score * 0.2)

            # Combined confidence (geometric mean for balanced factors)
            confidence = (
                skew_confidence
                * iv_confidence
                * time_confidence
                * vol_confidence
                * headline_confidence
            ) ** (1 / 5)

            return min(1.0, confidence)

        except Exception:
            return 0.5

    def _design_vertical_spread(
        self,
        signal_value: float,
        current_iv: float,
        options_data: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        """Design vertical spread position for bounded loss."""
        try:
            current_price = options_data.get("current_price", 50000)
            spread_width = current_price * self.config.spread_width_pct

            if signal_value > 0:  # Bullish: call spread
                strategy = "bull_call_spread"
                long_strike = current_price  # ATM
                short_strike = current_price + spread_width  # OTM
                max_profit = spread_width

                # Estimate spread cost (simplified)
                # atm_iv = current_iv
                # otm_iv = current_iv * 0.9  # Typical vol smile
                spread_cost = max_profit * 0.3  # Rough estimate

            else:  # Bearish: put spread
                strategy = "bear_put_spread"
                long_strike = current_price  # ATM
                short_strike = current_price - spread_width  # OTM
                max_profit = spread_width

                # Put spreads typically cost more due to skew
                spread_cost = max_profit * 0.4

            max_loss = spread_cost
            risk_reward = max_profit / max_loss if max_loss > 0 else 0

            return {
                "strategy": strategy,
                "long_strike": long_strike,
                "short_strike": short_strike,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "risk_reward": risk_reward,
                "estimated_cost": spread_cost,
                "expiry_days": options_data.get("time_to_expiry", 168) / 24,
            }

        except Exception:
            return {
                "strategy": "undefined",
                "error": "spread_design_failed",
            }

    def _calculate_position_size(
        self, current_iv: float, signal_value: float
    ) -> Dict[str, Any]:
        """Calculate position size limited to IV exposure."""
        try:
            # Base position size as percentage of portfolio
            base_size = abs(signal_value) * 0.05  # 5% max base allocation

            # IV adjustment: reduce size for high IV
            iv_adjustment = min(1.0, self.config.max_iv_exposure / current_iv)

            # Final position size
            position_size = base_size * iv_adjustment

            # Risk metrics
            max_iv_exposure = position_size * current_iv

            return {
                "position_size_pct": position_size,
                "iv_exposure": max_iv_exposure,
                "iv_adjustment": iv_adjustment,
                "base_size": base_size,
                "max_allowed_iv_exposure": self.config.max_iv_exposure,
            }

        except Exception:
            return {
                "position_size_pct": 0.01,
                "iv_exposure": 0.005,
                "error": "position_sizing_failed",
            }

    def _update_skew_history(self, skew_25d: float) -> None:
        """Update skew history for mean reversion analysis."""
        current_time = datetime.utcnow()
        self.skew_history.append((current_time, skew_25d))

        # Keep only recent history (last 48 hours)
        cutoff_time = current_time - timedelta(
            hours=self.config.skew_mean_reversion_period
        )
        self.skew_history = [
            (timestamp, skew)
            for timestamp, skew in self.skew_history
            if timestamp > cutoff_time
        ]
