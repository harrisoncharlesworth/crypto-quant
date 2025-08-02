import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta

from src.quantbot.signals.cross_sectional import (
    AltBTCCrossSectionalSignal,
    CrossSectionalConfig,
    CrossSectionalMomentumConfig,
    EnhancedCrossSectionalSignal,
)
from src.quantbot.signals.base import SignalResult


class TestCrossSectionalSignal:
    """Test suite for Alt/BTC Cross-Sectional Momentum Signal."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return CrossSectionalConfig(
            lookback_days=30,
            hold_days=7,
            ranking_period=90,  # Shorter for testing
            universe_size=5,  # Smaller universe for testing
            decile_threshold=0.4,  # 40% for smaller universe
            min_periods=50,  # Shorter minimum for testing
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create trending price data
        base_price = 1000
        trend = np.linspace(0, 0.3, 100)  # 30% uptrend
        noise = np.random.normal(0, 0.02, 100)  # 2% daily noise

        prices = base_price * np.exp(trend + noise.cumsum())

        data = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def signal_instance(self, sample_config):
        """Create signal instance for testing."""
        return AltBTCCrossSectionalSignal(sample_config)

    def test_signal_initialization(self, sample_config):
        """Test signal initialization and configuration."""
        signal = AltBTCCrossSectionalSignal(sample_config)

        assert signal.config == sample_config
        assert len(signal.universe) > 0
        assert "ETHUSDT" in signal.universe
        assert isinstance(signal.rankings_cache, dict)
        assert isinstance(signal.position_cache, dict)

    def test_universe_definition(self, signal_instance):
        """Test altcoin universe definition."""
        universe = signal_instance._get_altcoin_universe()

        assert len(universe) >= 5
        assert "ETHUSDT" in universe
        assert "ADAUSDT" in universe
        assert "SOLUSDT" in universe
        assert all(symbol.endswith("USDT") for symbol in universe)

    def test_insufficient_data_handling(self, signal_instance):
        """Test handling of insufficient data."""
        # Create insufficient data
        short_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )

        result = asyncio.run(signal_instance.generate(short_data, "ETHUSDT"))

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata
        assert "Insufficient data" in result.metadata["error"]

    def test_mock_multi_asset_data_generation(self, signal_instance, sample_data):
        """Test mock multi-asset data generation."""
        multi_data = signal_instance._mock_multi_asset_data(sample_data, "ETHUSDT")

        assert "BTCUSDT" in multi_data
        assert "ETHUSDT" in multi_data
        assert len(multi_data) >= 2

        # Test data structure
        for symbol, data in multi_data.items():
            assert isinstance(data, pd.DataFrame)
            assert len(data) == len(sample_data)
            assert all(
                col in data.columns
                for col in ["open", "high", "low", "close", "volume"]
            )

    def test_alt_btc_performance_calculation(self, signal_instance, sample_data):
        """Test alt/BTC relative performance calculation."""
        multi_data = signal_instance._mock_multi_asset_data(sample_data, "ETHUSDT")
        performance = signal_instance._calculate_alt_btc_performance(
            multi_data, "ETHUSDT"
        )

        assert "6m_return" in performance
        assert "30d_return" in performance
        assert "momentum_score" in performance

        assert isinstance(performance["momentum_score"], float)
        assert -1.0 <= performance["momentum_score"] <= 1.0  # Reasonable bounds

    def test_cross_sectional_ranking(self, signal_instance, sample_data):
        """Test cross-sectional ranking functionality."""
        multi_data = signal_instance._mock_multi_asset_data(sample_data, "ETHUSDT")
        rankings = signal_instance._perform_cross_sectional_ranking(multi_data)

        assert len(rankings) > 0

        for symbol, ranking in rankings.items():
            assert "rank" in ranking
            assert "percentile" in ranking
            assert "momentum_score" in ranking
            assert "decile" in ranking
            assert "decile_strength" in ranking

            assert ranking["decile"] in ["top", "middle", "bottom"]
            assert 0.0 <= ranking["percentile"] <= 1.0
            assert 0.0 <= ranking["decile_strength"] <= 1.0

    def test_ranking_signal_calculation(self, signal_instance):
        """Test signal calculation from rankings."""
        # Mock ranking data
        rankings = {
            "ETHUSDT": {
                "rank": 1,
                "percentile": 0.9,
                "momentum_score": 0.15,
                "decile": "top",
                "decile_strength": 0.8,
            }
        }

        performance = {"momentum_score": 0.15}

        signal_value, confidence = signal_instance._calculate_ranking_signal(
            "ETHUSDT", rankings, performance
        )

        assert signal_value > 0  # Should be positive for top decile
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should have decent confidence for strong signal

    def test_momentum_filter(self, signal_instance):
        """Test momentum persistence filter."""
        # Consistent momentum
        performance_consistent = {"30d_return": 0.1, "6m_return": 0.2}

        filtered_consistent = signal_instance._apply_momentum_filter(
            1.0, performance_consistent
        )
        assert filtered_consistent == 1.0  # No reduction for consistent momentum

        # Divergent momentum
        performance_divergent = {"30d_return": 0.1, "6m_return": -0.1}

        filtered_divergent = signal_instance._apply_momentum_filter(
            1.0, performance_divergent
        )
        assert filtered_divergent < 1.0  # Reduced for divergent momentum

    def test_correlation_filter(self, signal_instance, sample_data):
        """Test correlation breakdown risk management."""
        multi_data = signal_instance._mock_multi_asset_data(sample_data, "ETHUSDT")

        signal_value, confidence = signal_instance._apply_correlation_filter(
            1.0, 0.8, multi_data, "ETHUSDT"
        )

        assert 0.0 <= confidence <= 1.0
        assert -1.0 <= signal_value <= 1.0

    def test_position_size_calculation(self, signal_instance):
        """Test position size calculation with risk management."""
        performance = {"momentum_score": 0.1}

        position = signal_instance._calculate_position_size(0.5, 0.8, performance)

        assert abs(position) <= signal_instance.config.position_cap
        assert isinstance(position, float)

    def test_btc_hedge_signal_generation(self, signal_instance):
        """Test BTC hedge signal for market neutrality."""
        # Set up some alt positions
        signal_instance.position_cache = {
            "ETHUSDT": 0.1,
            "ADAUSDT": 0.05,
            "SOLUSDT": -0.08,
        }

        result = asyncio.run(signal_instance._generate_btc_hedge_signal("BTCUSDT"))

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"
        assert result.metadata is not None
        assert "hedge_type" in result.metadata
        assert result.metadata["hedge_type"] == "market_neutral"

        # Signal should be opposite of net alt exposure
        net_exposure = sum(signal_instance.position_cache.values())
        expected_hedge = -net_exposure * signal_instance.config.beta_hedge_ratio
        assert abs(result.value - expected_hedge) < 0.001

    @pytest.mark.asyncio
    async def test_alt_signal_generation(self, signal_instance, sample_data):
        """Test altcoin signal generation."""
        result = await signal_instance.generate(sample_data, "ETHUSDT")

        assert isinstance(result, SignalResult)
        assert result.symbol == "ETHUSDT"
        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0

        # Check metadata
        assert result.metadata is not None
        assert "alt_btc_performance" in result.metadata
        assert "ranking" in result.metadata
        assert "position_size" in result.metadata

    @pytest.mark.asyncio
    async def test_btc_signal_generation(self, signal_instance, sample_data):
        """Test BTC hedge signal generation."""
        result = await signal_instance.generate(sample_data, "BTCUSDT")

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"
        assert result.metadata is not None
        assert "hedge_type" in result.metadata

    @pytest.mark.asyncio
    async def test_non_universe_asset(self, signal_instance, sample_data):
        """Test handling of non-universe assets."""
        result = await signal_instance.generate(sample_data, "RANDOMUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert result.metadata["status"] == "not_in_universe"

    def test_portfolio_metrics_calculation(self, signal_instance):
        """Test portfolio-level metrics calculation."""
        # Mock rankings
        rankings = {
            "ETHUSDT": {"momentum_score": 0.15},
            "ADAUSDT": {"momentum_score": 0.08},
            "SOLUSDT": {"momentum_score": -0.05},
            "DOTUSDT": {"momentum_score": -0.12},
        }

        metrics = signal_instance.calculate_portfolio_metrics(rankings)

        assert "top_decile_avg" in metrics
        assert "bottom_decile_avg" in metrics
        assert "spread" in metrics
        assert "spread_annual" in metrics
        assert "universe_size" in metrics

        assert metrics["spread"] > 0  # Top should outperform bottom
        assert metrics["universe_size"] == len(rankings)

    def test_rebalance_timing(self, signal_instance):
        """Test rebalancing timing logic."""
        # Fresh signal (no rebalance needed)
        signal_instance.last_rebalance = datetime.utcnow()
        assert not signal_instance._is_rebalance_due()

        # Old signal (rebalance needed)
        signal_instance.last_rebalance = datetime.utcnow() - timedelta(days=10)
        assert signal_instance._is_rebalance_due()

    def test_error_handling(self, signal_instance):
        """Test error handling in signal generation."""
        # Test with malformed data
        bad_data = pd.DataFrame({"bad_column": [1, 2, 3]})

        result = asyncio.run(signal_instance.generate(bad_data, "ETHUSDT"))

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_signal_bounds(self, signal_instance, sample_data):
        """Test that signals are properly bounded."""
        result = asyncio.run(signal_instance.generate(sample_data, "ETHUSDT"))

        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CrossSectionalConfig()
        signal = AltBTCCrossSectionalSignal(config)
        assert signal.config == config

        # Config with extreme values
        extreme_config = CrossSectionalConfig(
            lookback_days=1000,
            position_cap=2.0,  # 200% position
            decile_threshold=0.9,  # 90% decile
        )
        signal_extreme = AltBTCCrossSectionalSignal(extreme_config)
        assert signal_extreme.config.lookback_days == 1000


class TestEnhancedCrossSectionalSignal:
    """Test suite for enhanced cross-sectional signal."""

    def test_enhanced_config(self):
        """Test enhanced configuration."""
        config = CrossSectionalMomentumConfig(
            ranking_lookback=180, signal_lookback=30, rebalance_frequency=7
        )

        assert config.ranking_lookback == 180
        assert config.signal_lookback == 30
        assert config.rebalance_frequency == 7

        signal = EnhancedCrossSectionalSignal(config)
        assert signal.config == config

    @pytest.mark.asyncio
    async def test_enhanced_signal_placeholder(self):
        """Test enhanced signal placeholder implementation."""
        config = CrossSectionalMomentumConfig()
        signal = EnhancedCrossSectionalSignal(config)

        data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )

        result = await signal.generate(data, "ETHUSDT")

        assert isinstance(result, SignalResult)
        assert result.metadata is not None
        assert "enhanced_implementation_placeholder" in result.metadata["status"]


# Integration and performance tests
class TestCrossSectionalIntegration:
    """Integration tests for cross-sectional signal."""

    @pytest.fixture
    def large_dataset(self):
        """Create larger dataset for performance testing."""
        dates = pd.date_range(start="2022-01-01", periods=365, freq="H")

        # Create realistic crypto price data
        returns = np.random.normal(0, 0.02, len(dates))
        prices = 1000 * np.exp(returns.cumsum())

        return pd.DataFrame(
            {
                "open": prices * np.random.uniform(0.995, 1.005, len(dates)),
                "high": prices * np.random.uniform(1.005, 1.02, len(dates)),
                "low": prices * np.random.uniform(0.98, 0.995, len(dates)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )

    @pytest.mark.asyncio
    async def test_signal_performance(self, large_dataset):
        """Test signal performance with large dataset."""
        config = CrossSectionalConfig(min_periods=100)
        signal = AltBTCCrossSectionalSignal(config)

        import time

        start_time = time.time()

        result = await signal.generate(large_dataset, "ETHUSDT")

        end_time = time.time()
        execution_time = end_time - start_time

        assert execution_time < 2.0  # Should complete within 2 seconds
        assert isinstance(result, SignalResult)

    def test_market_neutral_portfolio(self):
        """Test market-neutral portfolio construction."""
        config = CrossSectionalConfig()
        signal = AltBTCCrossSectionalSignal(config)

        # Simulate portfolio with alt positions
        alt_positions = {
            "ETHUSDT": 0.1,
            "ADAUSDT": 0.05,
            "SOLUSDT": -0.03,
            "LINKUSDT": 0.08,
            "DOTUSDT": -0.02,
        }

        signal.position_cache = alt_positions

        # Calculate net exposure
        net_exposure = sum(alt_positions.values())

        # BTC hedge should offset net exposure
        btc_hedge = -net_exposure * config.beta_hedge_ratio

        # Total portfolio exposure should be near zero
        total_exposure = net_exposure + btc_hedge
        assert abs(total_exposure) < 0.001  # Market neutral

    def test_ranking_consistency(self):
        """Test ranking consistency across multiple runs."""
        config = CrossSectionalConfig()
        signal = AltBTCCrossSectionalSignal(config)

        # Create deterministic data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(100, 200, 100),
                "low": np.random.uniform(100, 200, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # Generate rankings multiple times
        rankings1 = signal._perform_cross_sectional_ranking(
            signal._mock_multi_asset_data(data, "ETHUSDT")
        )

        rankings2 = signal._perform_cross_sectional_ranking(
            signal._mock_multi_asset_data(data, "ETHUSDT")
        )

        # Rankings should be consistent with same seed
        for symbol in rankings1:
            if symbol in rankings2:
                assert rankings1[symbol]["rank"] == rankings2[symbol]["rank"]


if __name__ == "__main__":
    pytest.main([__file__])
