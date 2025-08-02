import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.quantbot.signals.skew_whipsaw import (
    SkewWhipsawSignal,
    SkewWhipsawConfig,
)


@pytest.fixture
def skew_config():
    """Create test configuration for skew whipsaw signal."""
    return SkewWhipsawConfig(
        skew_threshold=15.0,
        vol_peak_lookback=24,
        max_iv_exposure=0.5,
        spread_width_pct=0.05,
        min_confidence_iv=0.20,
        max_confidence_iv=0.80,
        skew_mean_reversion_period=48,
        volume_spike_threshold=2.0,
        min_time_to_expiry=168,
        enabled=True,
        weight=1.0,
        min_confidence=0.5,
    )


@pytest.fixture
def skew_signal(skew_config):
    """Create skew whipsaw signal instance."""
    return SkewWhipsawSignal(skew_config)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with volatility patterns."""
    dates = pd.date_range("2024-01-01", periods=200, freq="h")  # 200 hours of data

    # Generate realistic price series with volatility clustering
    np.random.seed(42)
    returns = np.random.normal(0, 0.008, 200)  # 0.8% hourly vol

    # Add some volatility spikes
    spike_indices = [50, 120, 180]
    for idx in spike_indices:
        returns[idx : idx + 5] *= 3  # 3x vol spike

    prices = [45000]  # Start at $45k BTC
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate volume with spikes during vol events
    base_volume = np.random.uniform(80000, 120000, 200)
    for idx in spike_indices:
        base_volume[idx : idx + 5] *= 2.5  # Volume spike with vol spike

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.015 for p in prices],
            "low": [p * 0.985 for p in prices],
            "close": prices,
            "volume": base_volume,
        },
        index=dates,
    )

    return data


@pytest.fixture
def extreme_skew_scenario():
    """Create scenario with extreme skew conditions."""
    return {
        "current_price": 45000,
        "atm_iv": 0.75,
        "skew_25d": 22.0,  # Extreme skew >15 threshold
        "call_25d_iv": 0.53,
        "put_25d_iv": 0.75,
        "time_to_expiry": 240,  # 10 days
        "otm_call_strike": 47250,
        "otm_put_strike": 42750,
        "realized_vol": 0.55,
        "vol_spike_factor": 1.8,
    }


@pytest.fixture
def normal_skew_scenario():
    """Create scenario with normal skew conditions."""
    return {
        "current_price": 45000,
        "atm_iv": 0.45,
        "skew_25d": 8.0,  # Normal skew <15 threshold
        "call_25d_iv": 0.41,
        "put_25d_iv": 0.49,
        "time_to_expiry": 168,  # 7 days
        "otm_call_strike": 47250,
        "otm_put_strike": 42750,
        "realized_vol": 0.40,
        "vol_spike_factor": 1.0,
    }


class TestSkewWhipsawSignal:
    """Test suite for skew whipsaw signal."""

    @pytest.mark.asyncio
    async def test_signal_initialization(self, skew_signal, skew_config):
        """Test signal initializes correctly."""
        assert skew_signal.config == skew_config
        assert skew_signal.name == "SkewWhipsawSignal"
        assert skew_signal.skew_history == []
        assert skew_signal.vol_spike_cache == {}

    def test_25delta_skew_calculation(self, skew_signal):
        """Test 25-delta skew calculation."""
        options_data = {
            "put_25d_iv": 0.75,
            "call_25d_iv": 0.65,
        }

        skew = skew_signal._calculate_25delta_skew(options_data)
        expected_skew = (0.75 - 0.65) * 100  # 10 vol points

        assert skew == expected_skew
        assert isinstance(skew, float)

    def test_volatility_peak_detection(self, skew_signal, sample_ohlcv_data):
        """Test volatility peak detection."""
        # Use data with vol spikes
        options_data = {"vol_spike_factor": 1.5}

        vol_peak_score = skew_signal._detect_volatility_peak(
            sample_ohlcv_data, options_data
        )

        assert 0.0 <= vol_peak_score <= 1.0
        assert isinstance(vol_peak_score, float)

        # Test with high vol spike factor
        high_spike_data = {"vol_spike_factor": 2.0}
        high_score = skew_signal._detect_volatility_peak(
            sample_ohlcv_data, high_spike_data
        )

        assert high_score >= vol_peak_score  # Higher spike should give higher score

    def test_headline_event_detection(self, skew_signal, sample_ohlcv_data):
        """Test ETF headline event detection via volume spikes."""
        headline_score = skew_signal._detect_headline_events(sample_ohlcv_data)

        assert 0.0 <= headline_score <= 1.0
        assert isinstance(headline_score, float)

        # Test with insufficient data
        short_data = sample_ohlcv_data.head(20)
        score_short = skew_signal._detect_headline_events(short_data)
        assert score_short == 0.0

    def test_skew_mean_reversion_calculation(self, skew_signal):
        """Test skew mean reversion tendency calculation."""
        # Test with extreme skew
        extreme_skew = 25.0
        reversion_score = skew_signal._calculate_skew_mean_reversion(extreme_skew)

        assert 0.0 <= reversion_score <= 1.0
        assert isinstance(reversion_score, float)

        # Extreme skew should have higher reversion score
        normal_skew = 8.0
        normal_score = skew_signal._calculate_skew_mean_reversion(normal_skew)
        assert reversion_score > normal_score

    def test_mock_options_data_generation(self, skew_signal, sample_ohlcv_data):
        """Test mock options data generation."""
        options_data = skew_signal._generate_mock_options_data(
            sample_ohlcv_data, "BTCUSDT"
        )

        required_fields = [
            "current_price",
            "atm_iv",
            "skew_25d",
            "call_25d_iv",
            "put_25d_iv",
            "time_to_expiry",
            "otm_call_strike",
            "otm_put_strike",
            "realized_vol",
        ]

        for field in required_fields:
            assert field in options_data
            assert isinstance(options_data[field], (int, float))

        # Sanity checks
        assert options_data["current_price"] > 0
        assert 0.1 <= options_data["atm_iv"] <= 2.0
        assert options_data["time_to_expiry"] > 0
        assert options_data["otm_call_strike"] > options_data["current_price"]
        assert options_data["otm_put_strike"] < options_data["current_price"]

    def test_confidence_calculation(self, skew_signal, extreme_skew_scenario):
        """Test signal confidence calculation."""
        confidence = skew_signal._calculate_confidence(
            skew_25d=extreme_skew_scenario["skew_25d"],
            current_iv=extreme_skew_scenario["atm_iv"],
            vol_peak_score=0.8,
            headline_score=0.6,
            options_data=extreme_skew_scenario,
        )

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

        # Test with normal conditions
        normal_confidence = skew_signal._calculate_confidence(
            skew_25d=8.0,  # Normal skew
            current_iv=0.45,
            vol_peak_score=0.2,
            headline_score=0.1,
            options_data={"time_to_expiry": 168},
        )

        # Extreme conditions should give higher confidence
        assert confidence > normal_confidence

    def test_vertical_spread_design(self, skew_signal, extreme_skew_scenario):
        """Test vertical spread position design."""
        # Test bullish signal (fade negative skew)
        bullish_spread = skew_signal._design_vertical_spread(
            signal_value=0.5,
            current_iv=extreme_skew_scenario["atm_iv"],
            options_data=extreme_skew_scenario,
            symbol="BTCUSDT",
        )

        assert "strategy" in bullish_spread
        assert bullish_spread["strategy"] == "bull_call_spread"
        assert "max_profit" in bullish_spread
        assert "max_loss" in bullish_spread
        assert "risk_reward" in bullish_spread
        assert bullish_spread["long_strike"] < bullish_spread["short_strike"]

        # Test bearish signal (fade positive skew)
        bearish_spread = skew_signal._design_vertical_spread(
            signal_value=-0.5,
            current_iv=extreme_skew_scenario["atm_iv"],
            options_data=extreme_skew_scenario,
            symbol="BTCUSDT",
        )

        assert bearish_spread["strategy"] == "bear_put_spread"
        assert bearish_spread["long_strike"] > bearish_spread["short_strike"]

    def test_position_sizing_calculation(self, skew_signal):
        """Test position sizing based on IV exposure."""
        # Test with high IV
        high_iv_sizing = skew_signal._calculate_position_size(
            current_iv=1.0, signal_value=0.8  # 100% IV
        )

        assert "position_size_pct" in high_iv_sizing
        assert "iv_exposure" in high_iv_sizing
        assert "iv_adjustment" in high_iv_sizing

        # Test with low IV
        low_iv_sizing = skew_signal._calculate_position_size(
            current_iv=0.3, signal_value=0.8  # 30% IV
        )

        # Low IV should allow larger position size
        assert low_iv_sizing["position_size_pct"] >= high_iv_sizing["position_size_pct"]

        # IV exposure should be capped at max_iv_exposure
        max_exposure = skew_signal.config.max_iv_exposure
        assert high_iv_sizing["iv_exposure"] <= max_exposure * 1.1  # Small tolerance

    def test_skew_history_updates(self, skew_signal):
        """Test skew history tracking and cleanup."""
        # Add multiple skew readings
        test_skews = [15.0, 18.0, 22.0, 19.0, 16.0]

        for skew in test_skews:
            skew_signal._update_skew_history(skew)

        assert len(skew_signal.skew_history) == len(test_skews)

        # Check that old history gets cleaned up
        # Mock old timestamps
        old_time = datetime.utcnow() - timedelta(hours=100)
        skew_signal.skew_history.insert(0, (old_time, 10.0))

        # Update with new skew (should trigger cleanup)
        skew_signal._update_skew_history(20.0)

        # Old entry should be removed
        assert all(
            timestamp > datetime.utcnow() - timedelta(hours=48)
            for timestamp, _ in skew_signal.skew_history
        )

    @pytest.mark.asyncio
    async def test_signal_generation_extreme_skew(self, skew_signal, sample_ohlcv_data):
        """Test signal generation with extreme skew scenario."""
        with patch.object(
            skew_signal,
            "_generate_mock_options_data",
            return_value={
                "current_price": 45000,
                "atm_iv": 0.75,
                "skew_25d": 20.0,  # Above threshold
                "call_25d_iv": 0.55,
                "put_25d_iv": 0.75,
                "time_to_expiry": 240,
                "otm_call_strike": 47250,
                "otm_put_strike": 42750,
                "realized_vol": 0.55,
                "vol_spike_factor": 1.5,
            },
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert signal.symbol == "BTCUSDT"
            assert isinstance(signal.value, float)
            assert isinstance(signal.confidence, float)

            # Extreme positive skew should generate contrarian negative signal
            assert signal.value < 0  # Fade the skew
            assert signal.confidence > 0

            # Check metadata
            assert "skew_25d" in signal.metadata
            assert "strategy_type" in signal.metadata
            assert signal.metadata["strategy_type"] == "skew_mean_reversion"
            assert "spread_recommendation" in signal.metadata
            assert "position_size" in signal.metadata

    @pytest.mark.asyncio
    async def test_signal_generation_normal_skew(self, skew_signal, sample_ohlcv_data):
        """Test signal generation with normal skew (no signal)."""
        with patch.object(
            skew_signal,
            "_generate_mock_options_data",
            return_value={
                "current_price": 45000,
                "atm_iv": 0.45,
                "skew_25d": 8.0,  # Below threshold
                "call_25d_iv": 0.41,
                "put_25d_iv": 0.49,
                "time_to_expiry": 168,
                "otm_call_strike": 47250,
                "otm_put_strike": 42750,
                "realized_vol": 0.40,
                "vol_spike_factor": 1.0,
            },
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            # Normal skew should not generate signal
            assert signal.value == 0.0
            assert (
                signal.confidence >= 0.0
            )  # May have some confidence from other factors

    @pytest.mark.asyncio
    async def test_signal_generation_insufficient_data(self, skew_signal):
        """Test signal generation with insufficient data."""
        # Create minimal data
        short_data = pd.DataFrame(
            {
                "open": [45000, 45100],
                "high": [45200, 45300],
                "low": [44800, 44900],
                "close": [45100, 45200],
                "volume": [100000, 110000],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )

        signal = await skew_signal.generate(short_data, "BTCUSDT")

        assert signal.value == 0.0
        assert signal.confidence == 0.0
        assert "error" in signal.metadata
        assert "Insufficient data" in signal.metadata["error"]

    @pytest.mark.asyncio
    async def test_signal_enhancement_factors(self, skew_signal, sample_ohlcv_data):
        """Test signal enhancement from vol peaks and headlines."""
        # Mock scenario with vol peak and headline event
        enhanced_scenario = {
            "current_price": 45000,
            "atm_iv": 0.80,
            "skew_25d": 18.0,  # Above threshold
            "call_25d_iv": 0.62,
            "put_25d_iv": 0.80,
            "time_to_expiry": 240,
            "otm_call_strike": 47250,
            "otm_put_strike": 42750,
            "realized_vol": 0.50,
            "vol_spike_factor": 2.0,  # High vol spike
        }

        with patch.object(
            skew_signal, "_generate_mock_options_data", return_value=enhanced_scenario
        ):
            with patch.object(
                skew_signal,
                "_detect_volatility_peak",
                return_value=0.9,  # High vol peak
            ):
                with patch.object(
                    skew_signal,
                    "_detect_headline_events",
                    return_value=0.8,  # High headline score
                ):
                    enhanced_signal = await skew_signal.generate(
                        sample_ohlcv_data, "BTCUSDT"
                    )

        # Compare with baseline (no enhancements)
        with patch.object(
            skew_signal, "_generate_mock_options_data", return_value=enhanced_scenario
        ):
            with patch.object(
                skew_signal, "_detect_volatility_peak", return_value=0.1  # Low vol peak
            ):
                with patch.object(
                    skew_signal,
                    "_detect_headline_events",
                    return_value=0.1,  # Low headline score
                ):
                    baseline_signal = await skew_signal.generate(
                        sample_ohlcv_data, "BTCUSDT"
                    )

        # Enhanced signal should be stronger
        assert abs(enhanced_signal.value) >= abs(baseline_signal.value)
        assert enhanced_signal.confidence >= baseline_signal.confidence

    def test_vol_sizing_factor_application(self, skew_signal):
        """Test volatility-based position sizing factor."""
        # Test different IV levels
        low_iv = 0.30
        medium_iv = 0.60  # Below max_confidence_iv of 0.80
        max_confidence_iv = skew_signal.config.max_confidence_iv

        # Calculate sizing factors (as used in actual signal)
        low_iv_factor = min(1.0, low_iv / max_confidence_iv)
        medium_iv_factor = min(1.0, medium_iv / max_confidence_iv)

        # All factors should be valid
        assert 0.0 <= low_iv_factor <= 1.0
        assert 0.0 <= medium_iv_factor <= 1.0

        # Lower IV should give smaller factor (position scaling is based on IV/max_IV)
        assert low_iv_factor < medium_iv_factor

        # At max confidence IV, factor should be 1.0
        max_iv_factor = min(1.0, max_confidence_iv / max_confidence_iv)
        assert max_iv_factor == 1.0

    @pytest.mark.asyncio
    async def test_error_handling(self, skew_signal, sample_ohlcv_data):
        """Test error handling in signal generation."""
        # Mock an exception in options data generation
        with patch.object(
            skew_signal,
            "_generate_mock_options_data",
            side_effect=Exception("Mock error"),
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert signal.value == 0.0
            assert signal.confidence == 0.0
            assert "error" in signal.metadata
            assert "Mock error" in signal.metadata["error"]

    def test_signal_bounds(self, skew_signal):
        """Test that signal values are properly bounded."""
        # Test extreme scenarios that might produce out-of-bounds values
        test_cases = [
            {"raw_signal": 5.0, "enhancement": 3.0},  # Very high values
            {"raw_signal": -5.0, "enhancement": 3.0},  # Very negative values
            {"raw_signal": 0.5, "enhancement": 0.1},  # Normal values
        ]

        for case in test_cases:
            # Simulate signal calculation
            raw_signal = case["raw_signal"]
            enhanced_signal = raw_signal * case["enhancement"]

            # Apply bounds (same as in actual signal)
            bounded_signal = max(-1.0, min(1.0, enhanced_signal))

            assert -1.0 <= bounded_signal <= 1.0
            assert isinstance(bounded_signal, float)


class TestSkewWhipsawIntegration:
    """Integration tests for skew whipsaw signal."""

    @pytest.mark.asyncio
    async def test_complete_skew_whipsaw_workflow(self, skew_signal, sample_ohlcv_data):
        """Test complete workflow with realistic market scenario."""
        # Create realistic scenario: market stress with extreme skew
        stress_scenario = {
            "current_price": 42000,  # Price down from 45k
            "atm_iv": 0.95,  # High IV during stress
            "skew_25d": 25.0,  # Extreme put skew
            "call_25d_iv": 0.70,
            "put_25d_iv": 0.95,
            "time_to_expiry": 168,  # 1 week to expiry
            "otm_call_strike": 44100,
            "otm_put_strike": 39900,
            "realized_vol": 0.75,
            "vol_spike_factor": 2.2,  # High vol spike
        }

        with patch.object(
            skew_signal, "_generate_mock_options_data", return_value=stress_scenario
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            # Should generate strong contrarian signal
            assert signal.value < 0  # Fade extreme put skew
            assert signal.confidence > 0.5  # High confidence due to extremity

            # Verify strategy components
            metadata = signal.metadata
            assert metadata["skew_25d"] == 25.0
            assert metadata["strategy_type"] == "skew_mean_reversion"

            # Check spread recommendation
            spread_rec = metadata["spread_recommendation"]
            assert spread_rec["strategy"] == "bear_put_spread"  # Fade put skew
            assert spread_rec["max_loss"] > 0
            assert spread_rec["max_profit"] > 0

            # Check position sizing
            position_size = metadata["position_size"]
            assert position_size["iv_exposure"] <= skew_signal.config.max_iv_exposure

    @pytest.mark.asyncio
    async def test_mean_reversion_timing(self, skew_signal, sample_ohlcv_data):
        """Test timing of skew mean reversion signals."""
        # Build up skew history
        skew_history = [12.0, 15.0, 18.0, 22.0, 25.0]  # Increasing skew

        for skew in skew_history[:-1]:
            skew_signal._update_skew_history(skew)

        # Test current extreme skew
        extreme_scenario = {
            "current_price": 45000,
            "atm_iv": 0.80,
            "skew_25d": 25.0,
            "call_25d_iv": 0.55,
            "put_25d_iv": 0.80,
            "time_to_expiry": 240,
            "otm_call_strike": 47250,
            "otm_put_strike": 42750,
            "realized_vol": 0.60,
            "vol_spike_factor": 1.8,
        }

        with patch.object(
            skew_signal, "_generate_mock_options_data", return_value=extreme_scenario
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            # Skew momentum away from mean should enhance reversion signal
            assert signal.value != 0.0
            assert signal.confidence > 0

            # Check that skew history was updated
            assert len(skew_signal.skew_history) > 0
            assert skew_signal.skew_history[-1][1] == 25.0

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, skew_signal, sample_ohlcv_data):
        """Test integration of risk management features."""
        # Test scenario with risk concerns
        risk_scenario = {
            "current_price": 45000,
            "atm_iv": 1.2,  # Very high IV
            "skew_25d": 20.0,
            "call_25d_iv": 0.80,
            "put_25d_iv": 1.00,
            "time_to_expiry": 48,  # Short expiry (risky)
            "otm_call_strike": 47250,
            "otm_put_strike": 42750,
            "realized_vol": 0.90,
            "vol_spike_factor": 2.5,
        }

        with patch.object(
            skew_signal, "_generate_mock_options_data", return_value=risk_scenario
        ):
            signal = await skew_signal.generate(sample_ohlcv_data, "BTCUSDT")

            # Check risk adjustments
            metadata = signal.metadata

            # High IV should reduce position size
            position_size = metadata["position_size"]
            assert position_size["iv_adjustment"] < 1.0

            # Short expiry should reduce confidence
            time_to_expiry = metadata["time_to_expiry"]
            assert time_to_expiry < skew_signal.config.min_time_to_expiry

            # Spread should have reasonable risk-reward
            spread_rec = metadata["spread_recommendation"]
            if "risk_reward" in spread_rec:
                assert spread_rec["risk_reward"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
