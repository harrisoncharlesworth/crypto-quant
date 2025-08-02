import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.quantbot.signals.ssr import StablecoinSupplyRatioSignal, SSRConfig
from src.quantbot.signals.base import SignalResult


class TestStablecoinSupplyRatioSignal:
    """Test suite for Stablecoin Supply Ratio signal."""

    @pytest.fixture
    def default_config(self):
        """Create default SSR configuration for testing."""
        return SSRConfig(
            enabled=True,
            weight=1.0,
            min_confidence=0.3,
            zscore_window=52,  # Shorter window for testing
            rebalance_frequency="weekly",
            long_boost_threshold=-1.0,
            short_reduce_threshold=1.0,
            allocation_adjustment=0.25,
            min_periods=20,  # Reduced for testing
            signal_type="overlay",
            confidence_scaling=2.0,
        )

    @pytest.fixture
    def ssr_signal(self, default_config):
        """Create SSR signal instance."""
        return StablecoinSupplyRatioSignal(default_config)

    @pytest.fixture
    def market_data(self):
        """Create mock market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="H")

        # Create realistic price data with trends and volatility
        np.random.seed(42)
        prices = []
        base_price = 50000

        for i in range(len(dates)):
            # Add trend, cycles, and noise
            trend = 0.1 * (i / len(dates))  # Slight upward trend
            cycle = 0.05 * np.sin(2 * np.pi * i / (24 * 30))  # Monthly cycle
            noise = 0.02 * np.random.normal()

            price_mult = 1 + trend + cycle + noise
            price = base_price * price_mult
            prices.append(price)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + 0.01 * abs(np.random.normal())) for p in prices],
                "low": [p * (1 - 0.01 * abs(np.random.normal())) for p in prices],
                "close": prices,
                "volume": [1000 + 500 * abs(np.random.normal()) for _ in prices],
            },
            index=dates,
        )

        return data

    @pytest.mark.asyncio
    async def test_signal_generation_success(self, ssr_signal, market_data):
        """Test successful SSR signal generation."""
        result = await ssr_signal.generate(market_data, "BTCUSDT")

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"
        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.metadata is not None
        assert "ssr_current" in result.metadata
        assert "ssr_zscore" in result.metadata
        assert "signal_type" in result.metadata

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, ssr_signal):
        """Test handling of insufficient data."""
        # Create minimal data
        dates = pd.date_range(start="2023-01-01", periods=5, freq="H")
        data = pd.DataFrame(
            {
                "open": [50000] * 5,
                "high": [51000] * 5,
                "low": [49000] * 5,
                "close": [50000] * 5,
                "volume": [1000] * 5,
            },
            index=dates,
        )

        result = await ssr_signal.generate(data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_ssr_data_generation(self, ssr_signal, market_data):
        """Test SSR data generation and structure."""
        ssr_data = ssr_signal._get_ssr_data(market_data)

        assert isinstance(ssr_data, pd.Series)
        assert len(ssr_data) > 0
        assert ssr_data.name == "ssr"
        assert all(0.0 <= val <= 1.0 for val in ssr_data)  # SSR should be ratio
        assert ssr_data.index.freq == "W"  # Weekly frequency

    def test_zscore_calculation(self, ssr_signal):
        """Test Z-score calculation for SSR values."""
        # Create test SSR data
        ssr_values = [0.10, 0.12, 0.11, 0.15, 0.09, 0.13, 0.08, 0.16]
        ssr_data = pd.Series(ssr_values)

        # Test normal case
        zscore = ssr_signal._calculate_zscore(ssr_data, 0.20)
        assert isinstance(zscore, float)
        assert not np.isnan(zscore)
        assert zscore > 0  # 0.20 is above mean of test data

        # Test edge case - zero standard deviation
        constant_data = pd.Series([0.10] * 10)
        zscore_constant = ssr_signal._calculate_zscore(constant_data, 0.10)
        assert zscore_constant == 0.0

    def test_overlay_signal_generation(self, ssr_signal):
        """Test overlay signal generation from Z-scores."""
        # Test long boost threshold
        signal_val, signal_type = ssr_signal._generate_overlay_signal(-1.5)
        assert signal_val > 0  # Positive signal for long boost
        assert signal_type == "long_boost"

        # Test short reduce threshold
        signal_val, signal_type = ssr_signal._generate_overlay_signal(1.5)
        assert signal_val < 0  # Negative signal for long reduction
        assert signal_type == "long_reduce"

        # Test neutral range
        signal_val, signal_type = ssr_signal._generate_overlay_signal(0.0)
        assert signal_val == 0.0
        assert signal_type == "neutral"

    def test_confidence_calculation(self, ssr_signal):
        """Test confidence calculation from Z-scores."""
        # Test various Z-score values
        test_zscores = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for zscore in test_zscores:
            confidence = ssr_signal._calculate_confidence(zscore)
            assert 0.0 <= confidence <= 1.0

            # Higher absolute Z-score should give higher confidence
            if abs(zscore) > 1.0:
                assert confidence > 0.3

    def test_rebalance_timing(self, ssr_signal):
        """Test weekly rebalancing logic."""
        # Test different weekdays
        sunday = pd.Timestamp("2023-01-01")  # Sunday
        monday = pd.Timestamp("2023-01-02")  # Monday

        assert ssr_signal._is_rebalance_time(sunday)  # Sunday = rebalance
        assert not ssr_signal._is_rebalance_time(monday)  # Monday = no rebalance

    def test_allocation_adjustment_calculation(self, ssr_signal):
        """Test allocation adjustment calculations."""
        # Test long boost scenario
        adj_boost = ssr_signal._get_allocation_adjustment(-1.5)
        assert adj_boost["action"] == "increase_long_exposure"
        assert adj_boost["long_boost"] == 0.25
        assert adj_boost["short_reduce"] == 0.0

        # Test long reduce scenario
        adj_reduce = ssr_signal._get_allocation_adjustment(1.5)
        assert adj_reduce["action"] == "decrease_long_exposure"
        assert adj_reduce["long_boost"] == 0.0
        assert adj_reduce["short_reduce"] == 0.25

        # Test neutral scenario
        adj_neutral = ssr_signal._get_allocation_adjustment(0.0)
        assert adj_neutral["action"] == "maintain_exposure"
        assert adj_neutral["long_boost"] == 0.0
        assert adj_neutral["short_reduce"] == 0.0

    def test_dry_powder_interpretation(self, ssr_signal):
        """Test dry powder level interpretation."""
        # Test extreme cases
        assert ssr_signal._interpret_dry_powder(-2.5) == "very_high_dry_powder"
        assert ssr_signal._interpret_dry_powder(-1.5) == "high_dry_powder"
        assert ssr_signal._interpret_dry_powder(0.0) == "above_average_dry_powder"
        assert ssr_signal._interpret_dry_powder(1.5) == "low_dry_powder"
        assert ssr_signal._interpret_dry_powder(2.5) == "very_low_dry_powder"

    def test_regime_detection(self, ssr_signal):
        """Test market regime detection."""
        # Create trending SSR data
        dates = pd.date_range(start="2023-01-01", periods=20, freq="W")

        # Rising trend (accumulation regime)
        rising_ssr = pd.Series(np.linspace(0.10, 0.20, 20), index=dates)
        regime_rising = ssr_signal._detect_regime(rising_ssr)
        assert regime_rising == "accumulation_regime"

        # Falling trend (deployment regime)
        falling_ssr = pd.Series(np.linspace(0.20, 0.10, 20), index=dates)
        regime_falling = ssr_signal._detect_regime(falling_ssr)
        assert regime_falling == "deployment_regime"

        # Stable trend
        stable_ssr = pd.Series([0.15] * 20, index=dates)
        regime_stable = ssr_signal._detect_regime(stable_ssr)
        assert regime_stable == "stable_regime"

    def test_trend_calculation(self, ssr_signal):
        """Test trend calculation functionality."""
        # Rising trend
        rising_data = pd.Series([1, 2, 3, 4, 5])
        trend_rising = ssr_signal._calculate_trend(rising_data)
        assert trend_rising > 0

        # Falling trend
        falling_data = pd.Series([5, 4, 3, 2, 1])
        trend_falling = ssr_signal._calculate_trend(falling_data)
        assert trend_falling < 0

        # Flat trend
        flat_data = pd.Series([3, 3, 3, 3, 3])
        trend_flat = ssr_signal._calculate_trend(flat_data)
        assert abs(trend_flat) < 0.01  # Essentially zero

    def test_allocation_overlay_interface(self, ssr_signal):
        """Test allocation overlay interface for portfolio manager."""
        # Create mock signal result
        result = SignalResult(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            value=0.2,
            confidence=0.8,
            metadata={
                "allocation_adjustment": {
                    "action": "increase_long_exposure",
                    "long_boost": 0.25,
                    "short_reduce": 0.0,
                },
                "ssr_zscore": -1.5,
                "dry_powder_level": "high_dry_powder",
                "regime_detection": "deployment_regime",
            },
        )

        overlay = ssr_signal.get_allocation_overlay(result)

        assert overlay["action"] == "increase_long_exposure"
        assert overlay["multiplier"] == 1.25  # 1.0 + 0.25 boost
        assert overlay["confidence"] == 0.8
        assert overlay["ssr_zscore"] == -1.5

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = StablecoinSupplyRatioSignal.create_default_config()

        assert isinstance(config, SSRConfig)
        assert config.enabled
        assert config.signal_type == "overlay"
        assert config.rebalance_frequency == "weekly"
        assert config.allocation_adjustment == 0.25

    @pytest.mark.asyncio
    async def test_error_handling(self, ssr_signal):
        """Test error handling in signal generation."""
        # Test with None data
        result = await ssr_signal.generate(None, "BTCUSDT")
        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata

        # Test with invalid data structure
        invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
        result = await ssr_signal.generate(invalid_data, "BTCUSDT")
        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_signal_metadata_completeness(self, ssr_signal, market_data):
        """Test that signal metadata contains all expected fields."""
        result = await ssr_signal.generate(market_data, "BTCUSDT")

        expected_fields = [
            "ssr_current",
            "ssr_zscore",
            "signal_type",
            "is_rebalance_time",
            "allocation_adjustment",
            "dry_powder_level",
            "lookback_window",
            "regime_detection",
            "ssr_trend",
        ]

        for field in expected_fields:
            assert field in result.metadata, f"Missing metadata field: {field}"

    def test_signal_bounds(self, ssr_signal):
        """Test that signal values are properly bounded."""
        # Test extreme Z-scores don't break bounds
        extreme_zscores = [-5.0, -3.0, 0.0, 3.0, 5.0]

        for zscore in extreme_zscores:
            signal_val, _ = ssr_signal._generate_overlay_signal(zscore)
            assert (
                -1.0 <= signal_val <= 1.0
            ), f"Signal {signal_val} out of bounds for Z-score {zscore}"

    def test_weekly_frequency_consistency(self, ssr_signal, market_data):
        """Test that SSR data maintains weekly frequency."""
        ssr_data = ssr_signal._get_ssr_data(market_data)

        # Check that all intervals are approximately 7 days
        if len(ssr_data) > 1:
            intervals = ssr_data.index.to_series().diff().dt.days
            intervals = intervals.dropna()

            # Most intervals should be 7 days (weekly)
            assert all(
                6 <= interval <= 8 for interval in intervals
            ), "SSR data not properly weekly"
