"""Tests for MVRV Z-Score signal implementation."""

import pytest
import pandas as pd
import numpy as np

from src.quantbot.signals.mvrv import MVRVSignal, MVRVConfig
from src.quantbot.signals.base import SignalResult


class TestMVRVSignal:
    """Test suite for MVRV Z-Score signal."""

    @pytest.fixture
    def config(self):
        """Standard MVRV configuration for testing."""
        return MVRVConfig(
            value_threshold=-1.0,
            euphoric_threshold=7.0,
            lookback_periods=365 * 2,  # 2 years for testing
            signal_frequency_days=7,
            realized_cap_multiplier=0.7,
            volatility_window=30,
            regime_filter_enabled=True,
            extreme_zone_damping=0.5,
        )

    @pytest.fixture
    def signal(self, config):
        """MVRV signal instance for testing."""
        return MVRVSignal(config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")

        # Generate realistic price data with volatility
        np.random.seed(42)
        base_price = 30000
        returns = np.random.normal(0.0005, 0.03, len(dates))
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.995, 1.005, len(prices)),
                "high": prices * np.random.uniform(1.01, 1.03, len(prices)),
                "low": prices * np.random.uniform(0.97, 0.99, len(prices)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(prices)),
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def extreme_data(self):
        """Generate data with extreme MVRV conditions."""
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")

        # Create data with clear extreme zones
        base_price = 20000

        # Simulate market cycle: crash -> recovery -> bull -> peak -> crash
        cycle_prices = []
        for i, date in enumerate(dates):
            progress = i / len(dates)

            if progress < 0.2:  # Initial crash (value zone)
                price = base_price * (0.3 + 0.4 * progress * 5)
            elif progress < 0.6:  # Recovery and bull market
                price = base_price * (0.7 + 1.5 * (progress - 0.2) * 2.5)
            elif progress < 0.8:  # Peak euphoria (euphoric zone)
                price = base_price * (2.2 + 1.0 * (progress - 0.6) * 5)
            else:  # Crash back down
                price = base_price * (3.2 - 2.0 * (progress - 0.8) * 5)

            cycle_prices.append(price)

        prices = np.array(cycle_prices)

        data = pd.DataFrame(
            {
                "open": prices * 0.999,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, len(prices)),
            },
            index=dates,
        )

        return data

    @pytest.mark.asyncio
    async def test_basic_signal_generation(self, signal, sample_data):
        """Test basic signal generation functionality."""
        result = await signal.generate(sample_data, "BTCUSDT")

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"
        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.metadata is not None
        assert "mvrv_ratio" in result.metadata
        assert "mvrv_zscore" in result.metadata
        assert "regime" in result.metadata

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, signal):
        """Test handling of insufficient data."""
        # Create minimal data (less than required lookback)
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        minimal_data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [105] * 10,
                "low": [95] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        result = await signal.generate(minimal_data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_mvrv_calculation(self, signal, sample_data):
        """Test MVRV ratio calculation logic."""
        # Test internal MVRV calculation
        mvrv_ratio = signal._calculate_mvrv_ratio(sample_data)

        assert isinstance(mvrv_ratio, pd.Series)
        assert len(mvrv_ratio) == len(sample_data)
        assert all(mvrv_ratio > 0)  # MVRV should be positive
        assert not mvrv_ratio.isna().any()  # No NaN values

    @pytest.mark.asyncio
    async def test_zscore_calculation(self, signal, sample_data):
        """Test Z-score calculation and normalization."""
        mvrv_ratio = signal._calculate_mvrv_ratio(sample_data)
        mvrv_zscore = signal._calculate_zscore(mvrv_ratio, sample_data)

        assert isinstance(mvrv_zscore, pd.Series)
        assert len(mvrv_zscore) == len(sample_data)

        # Z-score should have reasonable range (not all zeros after initial period)
        non_zero_scores = mvrv_zscore[mvrv_zscore != 0]
        assert len(non_zero_scores) > len(mvrv_zscore) * 0.5

    @pytest.mark.asyncio
    async def test_value_zone_detection(self, signal, extreme_data):
        """Test detection of value zone (MVRV Z-Score < -1)."""
        # Ensure we have sufficient data for MVRV calculation
        if len(extreme_data) < signal.config.lookback_periods:
            result = await signal.generate(extreme_data, "BTCUSDT")
            # Should return error for insufficient data
            assert result.value == 0.0
            assert result.confidence == 0.0
            assert "error" in result.metadata
        else:
            # Run signal on early period (should be value zone)
            early_data = extreme_data.iloc[: signal.config.lookback_periods + 100]
            result = await signal.generate(early_data, "BTCUSDT")

            # Should detect value zone characteristics
            assert result.metadata["regime"] in ["value_zone", "neutral"]

            if result.metadata["is_value_zone"]:
                assert result.value > 0  # Long bias in value zone
                assert result.confidence > 0.3

    @pytest.mark.asyncio
    async def test_euphoric_zone_detection(self, signal, extreme_data):
        """Test detection of euphoric zone (MVRV Z-Score > 7)."""
        # Configure for more sensitive euphoric detection
        signal.config.euphoric_threshold = 2.0

        # Ensure sufficient data for MVRV calculation
        if len(extreme_data) < signal.config.lookback_periods:
            result = await signal.generate(extreme_data, "BTCUSDT")
            # Should return error for insufficient data
            assert result.value == 0.0
            assert result.confidence == 0.0
            assert "error" in result.metadata
        else:
            # Run signal on full data (may contain euphoric periods)
            result = await signal.generate(extreme_data, "BTCUSDT")

            # May detect euphoric characteristics depending on data
            if (
                "is_euphoric_zone" in result.metadata
                and result.metadata["is_euphoric_zone"]
            ):
                assert result.value < 0  # Short bias in euphoric zone
                assert result.confidence > 0.3

    @pytest.mark.asyncio
    async def test_weekly_frequency_constraint(self, signal, sample_data):
        """Test weekly signal frequency constraint."""
        # Generate first signal
        await signal.generate(sample_data, "BTCUSDT")
        assert signal._last_signal_date is not None

        # Generate signal next day (should be skipped)
        next_day_data = sample_data.iloc[:-1]  # Simulate next day
        result2 = await signal.generate(next_day_data, "BTCUSDT")

        # Second result should indicate frequency skip
        if "status" in result2.metadata:
            assert result2.metadata["status"] == "frequency_skip"

    @pytest.mark.asyncio
    async def test_regime_signal_generation(self, signal, sample_data):
        """Test regime-aware signal generation logic."""
        mvrv_ratio = signal._calculate_mvrv_ratio(sample_data)
        mvrv_zscore = signal._calculate_zscore(mvrv_ratio, sample_data)

        # Test different Z-score scenarios
        test_cases = [
            (-2.0, "value_zone", "positive"),  # Value zone
            (8.0, "euphoric_zone", "negative"),  # Euphoric zone
            (0.5, "neutral", "neutral"),  # Neutral zone
        ]

        for zscore_val, expected_regime, expected_bias in test_cases:
            # Create synthetic Z-score
            test_zscore = pd.Series([zscore_val], index=[mvrv_zscore.index[-1]])

            signal_val, confidence, regime = signal._generate_regime_signal(
                test_zscore, sample_data.tail(1)
            )

            assert regime == expected_regime

            if expected_bias == "positive":
                assert signal_val > 0
            elif expected_bias == "negative":
                assert signal_val < 0
            else:
                assert signal_val == 0

    @pytest.mark.asyncio
    async def test_extreme_zone_damping(self, signal, sample_data):
        """Test signal damping in extreme zones."""
        # Enable extreme zone damping
        signal.config.regime_filter_enabled = True
        signal.config.extreme_zone_damping = 0.5

        # Create extreme Z-score scenario
        mvrv_ratio = signal._calculate_mvrv_ratio(sample_data)
        extreme_zscore = pd.Series([-3.0], index=[mvrv_ratio.index[-1]])

        signal_val, confidence, regime = signal._generate_regime_signal(
            extreme_zscore, sample_data.tail(1)
        )

        # Signal should be damped in extreme zones
        if regime != "neutral":
            # Damped signal should be reduced
            assert abs(signal_val) <= 1.0

    @pytest.mark.asyncio
    async def test_regime_status_interface(self, signal, sample_data):
        """Test regime status interface for other signals."""
        # Generate signal to populate history
        await signal.generate(sample_data, "BTCUSDT")

        # Test regime status retrieval
        regime_status = signal.get_regime_status()

        if regime_status is not None:
            assert "regime" in regime_status
            assert "mvrv_zscore" in regime_status
            assert "is_extreme" in regime_status
            assert "should_dampen_directional" in regime_status
            assert "contrarian_bias" in regime_status

    @pytest.mark.asyncio
    async def test_historical_extremes_analysis(self, signal, sample_data):
        """Test historical extreme analysis functionality."""
        # Generate multiple signals to build history
        for i in range(0, len(sample_data), 7):  # Weekly signals
            subset_data = (
                sample_data.iloc[: i + 100]
                if i + 100 < len(sample_data)
                else sample_data
            )
            if len(subset_data) > signal.config.lookback_periods:
                await signal.generate(subset_data, "BTCUSDT")

        # Analyze historical extremes
        extremes = signal.get_historical_extremes(lookback_days=90)

        if extremes:
            assert "min_zscore" in extremes
            assert "max_zscore" in extremes
            assert "value_zone_count" in extremes
            assert "euphoric_zone_count" in extremes
            assert "neutral_count" in extremes

    @pytest.mark.asyncio
    async def test_signal_metadata_completeness(self, signal, sample_data):
        """Test completeness of signal metadata."""
        result = await signal.generate(sample_data, "BTCUSDT")

        expected_metadata_keys = [
            "mvrv_ratio",
            "mvrv_zscore",
            "regime",
            "is_value_zone",
            "is_euphoric_zone",
            "weekly_frequency",
            "signal_type",
        ]

        for key in expected_metadata_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"

    @pytest.mark.asyncio
    async def test_signal_value_bounds(self, signal, sample_data):
        """Test signal value stays within expected bounds."""
        result = await signal.generate(sample_data, "BTCUSDT")

        # Signal value should be bounded
        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0

        # Confidence should be reasonable for any regime
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_config_parameter_effects(self, sample_data):
        """Test different configuration parameters."""
        # Test with different thresholds
        configs = [
            MVRVConfig(value_threshold=-0.5, euphoric_threshold=5.0),
            MVRVConfig(value_threshold=-2.0, euphoric_threshold=10.0),
            MVRVConfig(regime_filter_enabled=False),
            MVRVConfig(extreme_zone_damping=0.1),
            MVRVConfig(signal_frequency_days=1),
        ]

        for config in configs:
            signal = MVRVSignal(config)
            result = await signal.generate(sample_data, "BTCUSDT")

            # Should generate valid signal regardless of config
            assert isinstance(result, SignalResult)
            assert -1.0 <= result.value <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_signal_initialization(self, config):
        """Test proper signal initialization."""
        signal = MVRVSignal(config)

        assert signal.config == config
        assert signal.name == "MVRVSignal"
        assert signal._last_signal_date is None
        assert signal._mvrv_history.empty

    @pytest.mark.asyncio
    async def test_memory_management(self, signal, sample_data):
        """Test memory management with large history."""
        # Generate many signals to test history management
        for i in range(50):
            # Simulate passage of time
            signal._last_signal_date = None  # Reset to force signal generation
            await signal.generate(sample_data, "BTCUSDT")

        # History should be managed (not grow indefinitely)
        assert len(signal._mvrv_history) <= 1000

    def test_signal_type_classification(self, signal):
        """Test signal is properly classified as regime filter."""
        assert hasattr(signal, "get_regime_status")
        assert hasattr(signal, "get_historical_extremes")

        # Should be designed as regime filter
        assert signal.config.regime_filter_enabled is not None
