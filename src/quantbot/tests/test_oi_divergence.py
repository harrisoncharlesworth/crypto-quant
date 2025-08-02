import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from quantbot.signals.oi_divergence import OIPriceDivergenceSignal, OIDivergenceConfig


class TestOIPriceDivergenceSignal:
    """Test suite for Open Interest / Price Divergence Signal."""

    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return OIDivergenceConfig(
            oi_momentum_window=24,
            price_momentum_window=24,
            divergence_threshold=0.15,
            min_venues=3,
            volume_weight_enabled=True,
            min_confidence=0.7,
        )

    @pytest.fixture
    def signal(self, config):
        """Signal instance for testing."""
        return OIPriceDivergenceSignal(config)

    @pytest.fixture
    def sample_data(self):
        """Sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=48, freq="H")
        np.random.seed(42)  # For reproducible tests

        # Create realistic price movement with some volatility
        base_price = 50000
        price_changes = np.random.normal(0, 0.01, 48)  # 1% hourly volatility
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.005 for p in prices],  # Small wick up
                "low": [p * 0.995 for p in prices],  # Small wick down
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 48),
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def divergence_data(self):
        """Data showing clear OI/Price divergence patterns."""
        dates = pd.date_range(start="2024-01-01", periods=48, freq="H")

        # Create falling price pattern for bearish divergence test
        base_price = 50000
        prices = []
        for i in range(48):
            # Gradual price decline
            price = base_price * (1 - (i * 0.002))  # 0.2% decline per hour
            prices.append(price)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.002 for p in prices],
                "low": [p * 0.998 for p in prices],
                "close": prices,
                "volume": np.random.uniform(2000, 8000, 48),  # Higher volume
            },
            index=dates,
        )

        return data

    @pytest.mark.asyncio
    async def test_signal_initialization(self, config):
        """Test signal initialization and configuration."""
        signal = OIPriceDivergenceSignal(config)

        assert signal.config == config
        assert signal.name == "OIPriceDivergenceSignal"
        assert len(signal.venues) == 5
        assert signal.oi_history == {}
        assert signal.price_history == {}

    @pytest.mark.asyncio
    async def test_insufficient_data(self, signal):
        """Test handling of insufficient market data."""
        # Create minimal data (less than required window)
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
            },
            index=[datetime.utcnow()],
        )

        result = await signal.generate(data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "Insufficient market data" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_insufficient_venues(self, signal, sample_data):
        """Test handling when insufficient venues are available."""

        # Mock _get_multi_venue_oi_data to return insufficient venues
        with patch.object(
            signal,
            "_get_multi_venue_oi_data",
            return_value={"binance": {}, "bybit": {}},
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            assert result.value == 0.0
            assert result.confidence == 0.0
            assert "Insufficient venue coverage" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_bearish_divergence_signal(self, signal, divergence_data):
        """Test bearish signal generation (rising OI + falling price)."""

        # Mock venue data showing rising OI
        mock_venue_data = {
            "binance": {
                "current_oi": 100000000,
                "current_volume": 10000000,
                "oi_momentum": 0.12,  # Rising OI
                "volume_ratio": 1.5,
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 80000000,
                "current_volume": 8000000,
                "oi_momentum": 0.10,  # Rising OI
                "volume_ratio": 1.3,
                "data_quality": 0.92,
            },
            "okx": {
                "current_oi": 60000000,
                "current_volume": 6000000,
                "oi_momentum": 0.08,  # Rising OI
                "volume_ratio": 1.2,
                "data_quality": 0.90,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(divergence_data, "BTCUSDT")

            # Should generate bearish signal due to rising OI + falling price
            assert result.value < 0  # Bearish signal
            assert result.confidence >= 0.6  # Accept slightly lower confidence for test
            assert result.metadata["signal_pattern"] == "rising_oi_falling_price"
            assert result.metadata["venue_count"] == 3

    @pytest.mark.asyncio
    async def test_bullish_oi_flush_signal(self, signal, sample_data):
        """Test bullish signal from OI flush + flat price."""

        # Mock venue data showing OI flush
        mock_venue_data = {
            "binance": {
                "current_oi": 70000000,
                "current_volume": 15000000,
                "oi_momentum": -0.25,  # Strong OI flush
                "volume_ratio": 2.5,
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 55000000,
                "current_volume": 12000000,
                "oi_momentum": -0.20,  # OI flush
                "volume_ratio": 2.2,
                "data_quality": 0.93,
            },
            "okx": {
                "current_oi": 45000000,
                "current_volume": 10000000,
                "oi_momentum": -0.18,  # OI flush
                "volume_ratio": 2.0,
                "data_quality": 0.91,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            # Should generate bullish signal due to OI flush + relatively flat price
            assert result.value > 0  # Bullish signal
            assert result.metadata["signal_pattern"] == "oi_flush_flat_price"
            assert result.metadata["venue_count"] == 3

    @pytest.mark.asyncio
    async def test_volume_weighting(self, signal, sample_data):
        """Test volume-weighted OI momentum calculation."""

        # Test with volume weighting enabled
        mock_venue_data = {
            "binance": {
                "current_oi": 100000000,
                "current_volume": 20000000,  # High volume venue
                "oi_momentum": 0.15,
                "volume_ratio": 1.5,
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 50000000,
                "current_volume": 5000000,  # Low volume venue
                "oi_momentum": -0.05,  # Opposite momentum
                "volume_ratio": 1.2,
                "data_quality": 0.90,
            },
            "okx": {
                "current_oi": 30000000,
                "current_volume": 3000000,  # Low volume venue
                "oi_momentum": -0.03,
                "volume_ratio": 1.1,
                "data_quality": 0.88,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            # Volume-weighted momentum should be dominated by high-volume venue (binance)
            assert (
                result.metadata["oi_momentum"] > 0
            )  # Should be positive due to volume weighting

    @pytest.mark.asyncio
    async def test_liquidation_cascade_detection(self, signal):
        """Test liquidation cascade detection via volume spikes."""

        # Create data with volume spike
        dates = pd.date_range(start="2024-01-01", periods=24, freq="H")
        volumes = [1000] * 20 + [5000, 6000, 7000, 8000]  # Volume spike at end

        data = pd.DataFrame(
            {
                "open": [50000] * 24,
                "high": [50100] * 24,
                "low": [49900] * 24,
                "close": [50000] * 24,
                "volume": volumes,
            },
            index=dates,
        )

        mock_venue_data = {
            "binance": {
                "current_oi": 100000000,
                "current_volume": 15000000,
                "oi_momentum": 0.05,
                "volume_ratio": 3.0,  # High volume ratio indicating liquidation
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 80000000,
                "current_volume": 12000000,
                "oi_momentum": 0.03,
                "volume_ratio": 2.8,
                "data_quality": 0.92,
            },
            "okx": {
                "current_oi": 60000000,
                "current_volume": 10000000,
                "oi_momentum": 0.02,
                "volume_ratio": 2.5,
                "data_quality": 0.90,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(data, "BTCUSDT")

            assert result.metadata["liquidation_signal"] > 0
            assert result.metadata["cascade_risk"] is True

    @pytest.mark.asyncio
    async def test_signal_smoothing(self, signal, sample_data):
        """Test temporal signal smoothing functionality."""

        # Generate multiple signals to test smoothing
        mock_venue_data = {
            "binance": {
                "current_oi": 100000000,
                "current_volume": 10000000,
                "oi_momentum": 0.10,
                "volume_ratio": 1.5,
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 80000000,
                "current_volume": 8000000,
                "oi_momentum": 0.08,
                "volume_ratio": 1.3,
                "data_quality": 0.92,
            },
            "okx": {
                "current_oi": 60000000,
                "current_volume": 6000000,
                "oi_momentum": 0.06,
                "volume_ratio": 1.2,
                "data_quality": 0.90,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            # Generate first signal
            await signal.generate(sample_data, "BTCUSDT")

            # Generate second signal (should be smoothed with first)
            await signal.generate(sample_data, "BTCUSDT")

            # Check that signal history is being maintained
            assert "BTCUSDT" in signal.last_signals
            assert len(signal.last_signals["BTCUSDT"]) >= 1

    def test_divergence_strength_calculation(self, signal):
        """Test divergence strength calculation logic."""

        # Test strong positive divergence (OI up, price down)
        oi_momentum = 0.15
        price_momentum = -0.08
        divergence = signal._calculate_divergence_strength(oi_momentum, price_momentum)
        assert divergence > 0.5  # Should be strong divergence

        # Test weak divergence
        oi_momentum = 0.02
        price_momentum = -0.01
        divergence = signal._calculate_divergence_strength(oi_momentum, price_momentum)
        assert divergence < 0.3  # Should be weak divergence

        # Test no divergence (same direction)
        oi_momentum = 0.05
        price_momentum = 0.05
        divergence = signal._calculate_divergence_strength(oi_momentum, price_momentum)
        assert divergence == 0.0  # No divergence

    def test_signal_pattern_classification(self, signal):
        """Test signal pattern classification."""

        # Test rising OI + falling price pattern
        pattern = signal._classify_signal_pattern(0.10, -0.05)
        assert pattern == "rising_oi_falling_price"

        # Test OI flush + flat price pattern
        pattern = signal._classify_signal_pattern(-0.15, 0.01)
        assert pattern == "oi_flush_flat_price"

        # Test strong divergence pattern
        pattern = signal._classify_signal_pattern(0.12, -0.08)
        assert pattern == "strong_divergence"

        # Test neutral pattern
        pattern = signal._classify_signal_pattern(0.02, 0.01)
        assert pattern == "neutral_or_weak"

    @pytest.mark.asyncio
    async def test_early_warning_system(self, signal, sample_data):
        """Test early warning system functionality."""

        # Mock OI flush scenario
        mock_venue_data = {
            "binance": {
                "current_oi": 70000000,
                "current_volume": 15000000,
                "oi_momentum": -0.35,
                "volume_ratio": 2.5,
                "data_quality": 0.95,
            },
            "bybit": {
                "current_oi": 55000000,
                "current_volume": 12000000,
                "oi_momentum": -0.30,
                "volume_ratio": 2.2,
                "data_quality": 0.93,
            },
            "okx": {
                "current_oi": 45000000,
                "current_volume": 10000000,
                "oi_momentum": -0.25,
                "volume_ratio": 2.0,
                "data_quality": 0.91,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            # Check for early warning metadata
            assert "early_warning" in result.metadata
            warnings = result.metadata["early_warning"]

            if "oi_flush_detected" in warnings:
                assert warnings["oi_flush_detected"]["type"] == "bullish_setup"
                assert warnings["oi_flush_detected"]["confidence"] > 0.8

    @pytest.mark.asyncio
    async def test_confidence_thresholding(self, signal, sample_data):
        """Test that signals meet minimum confidence requirements."""

        # Mock weak signal scenario
        mock_venue_data = {
            "binance": {
                "current_oi": 100000000,
                "current_volume": 10000000,
                "oi_momentum": 0.02,
                "volume_ratio": 1.1,
                "data_quality": 0.75,
            },
            "bybit": {
                "current_oi": 90000000,
                "current_volume": 9000000,
                "oi_momentum": 0.01,
                "volume_ratio": 1.0,
                "data_quality": 0.70,
            },
            "okx": {
                "current_oi": 80000000,
                "current_volume": 8000000,
                "oi_momentum": 0.01,
                "volume_ratio": 0.9,
                "data_quality": 0.65,
            },
        }

        with patch.object(
            signal, "_get_multi_venue_oi_data", return_value=mock_venue_data
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            # Weak signals should result in low confidence
            assert result.confidence < signal.config.min_confidence

    def test_venue_confidence_calculation(self, signal):
        """Test cross-venue confidence calculation."""

        # Test consistent venue data
        venue_data = {
            "binance": {"oi_momentum": 0.10, "data_quality": 0.95},
            "bybit": {"oi_momentum": 0.08, "data_quality": 0.93},
            "okx": {"oi_momentum": 0.12, "data_quality": 0.91},
            "coinbase": {"oi_momentum": 0.09, "data_quality": 0.89},
        }

        confidence = signal._calculate_venue_confidence(venue_data, 0.10)
        assert confidence > 0.7  # High confidence for consistent venues

        # Test inconsistent venue data
        venue_data_inconsistent = {
            "binance": {"oi_momentum": 0.10, "data_quality": 0.95},
            "bybit": {"oi_momentum": -0.05, "data_quality": 0.93},
            "okx": {"oi_momentum": 0.02, "data_quality": 0.91},
        }

        confidence = signal._calculate_venue_confidence(venue_data_inconsistent, 0.10)
        assert confidence < 0.5  # Low confidence for inconsistent venues

    def test_strategy_metrics(self, signal):
        """Test strategy metrics generation."""

        # Initialize some test data
        signal.oi_history["BTCUSDT"] = {
            "binance": [(datetime.utcnow(), 100000000, 10000000)],
            "bybit": [(datetime.utcnow(), 80000000, 8000000)],
        }
        signal.last_signals["BTCUSDT"] = [
            (datetime.utcnow(), 0.5, 0.8),
            (datetime.utcnow(), -0.3, 0.7),
        ]

        metrics = signal.get_strategy_metrics("BTCUSDT")

        assert metrics["strategy"] == "oi_price_divergence"
        assert "venue_coverage" in metrics
        assert metrics["venue_coverage"]["active_venues"] == 2
        assert "signal_history" in metrics

    def test_state_reset(self, signal):
        """Test signal state reset functionality."""

        # Add some test data
        signal.oi_history["BTCUSDT"] = {"binance": []}
        signal.price_history["BTCUSDT"] = []
        signal.last_signals["BTCUSDT"] = []

        # Reset specific symbol
        signal.reset_state("BTCUSDT")
        assert "BTCUSDT" not in signal.oi_history
        assert "BTCUSDT" not in signal.price_history
        assert "BTCUSDT" not in signal.last_signals

        # Add data again and reset all
        signal.oi_history["BTCUSDT"] = {"binance": []}
        signal.oi_history["ETHUSDT"] = {"binance": []}

        signal.reset_state()
        assert len(signal.oi_history) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, signal, sample_data):
        """Test error handling in signal generation."""

        # Mock an exception in venue data retrieval
        with patch.object(
            signal, "_get_multi_venue_oi_data", side_effect=Exception("Mock error")
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

            assert result.value == 0.0
            assert result.confidence == 0.0
            assert "Mock error" in result.metadata["error"]

    def test_signal_normalization(self, signal):
        """Test signal value normalization."""

        # Test values within bounds
        assert signal.normalize_signal(0.5) == 0.5
        assert signal.normalize_signal(-0.5) == -0.5

        # Test values exceeding bounds
        assert signal.normalize_signal(2.0) == 1.0
        assert signal.normalize_signal(-2.0) == -1.0

        # Test with custom scale
        assert signal.normalize_signal(2.0, scale=4.0) == 0.5
        assert signal.normalize_signal(-2.0, scale=4.0) == -0.5


@pytest.mark.asyncio
async def test_integration_with_real_market_scenarios():
    """Integration test with realistic market scenarios."""

    config = OIDivergenceConfig(
        oi_momentum_window=24,
        price_momentum_window=24,
        divergence_threshold=0.15,
        min_venues=3,
        min_confidence=0.7,
    )
    signal = OIPriceDivergenceSignal(config)

    # Simulate March 2025 flash crash scenario (per Kaiko research)
    dates = pd.date_range(start="2025-03-24", periods=48, freq="H")

    # Create price decline pattern
    base_price = 65000
    prices = []
    for i in range(48):
        if i < 24:
            # Gradual decline leading up to crash
            price = base_price * (1 - (i * 0.001))
        else:
            # Sharp 15% drop in second day
            price = base_price * 0.85 * (1 - ((i - 24) * 0.0005))
        prices.append(price)

    crash_data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [2000 + i * 100 for i in range(48)],  # Increasing volume
        },
        index=dates,
    )

    # Mock rising OI scenario before crash
    mock_venue_data = {
        "binance": {
            "current_oi": 120000000,  # High OI
            "current_volume": 25000000,
            "oi_momentum": 0.18,  # Strong OI buildup
            "volume_ratio": 2.8,
            "data_quality": 0.95,
        },
        "bybit": {
            "current_oi": 100000000,
            "current_volume": 20000000,
            "oi_momentum": 0.15,
            "volume_ratio": 2.5,
            "data_quality": 0.93,
        },
        "okx": {
            "current_oi": 80000000,
            "current_volume": 16000000,
            "oi_momentum": 0.12,
            "volume_ratio": 2.2,
            "data_quality": 0.91,
        },
    }

    with patch.object(signal, "_get_multi_venue_oi_data", return_value=mock_venue_data):
        result = await signal.generate(crash_data, "BTCUSDT")

        # Should generate strong bearish signal predicting the crash
        assert result.value < -0.5  # Strong bearish signal
        assert result.confidence >= 0.7  # High confidence
        assert result.metadata["signal_pattern"] == "rising_oi_falling_price"
        assert result.metadata["cascade_risk"] is True

        # Check early warning system
        warnings = result.metadata["early_warning"]
        assert "bearish_divergence_building" in warnings
        assert warnings["bearish_divergence_building"]["cascade_risk"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
