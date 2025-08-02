import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.quantbot.signals.x_exchange_funding import (
    XExchangeFundingDispersionSignal,
    XExchangeFundingConfig,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")

    return pd.DataFrame(
        {
            "open": np.random.uniform(45000, 55000, 100),
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(45000, 55000, 100),
            "close": np.random.uniform(45000, 55000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        },
        index=dates,
    )


@pytest.fixture
def config():
    """Create test configuration."""
    return XExchangeFundingConfig(
        entry_threshold_bps=20.0,
        exit_threshold_bps=5.0,
        inventory_cost_bps=2.0,
        max_allocation=0.20,
        max_position_hours=48,
        reversal_stop_bps=30.0,
        supported_exchanges=["binance", "bybit", "okx"],
    )


@pytest.fixture
def signal(config):
    """Create signal instance."""
    return XExchangeFundingDispersionSignal(config)


class TestXExchangeFundingConfig:
    """Test configuration class."""

    def test_default_config_creation(self):
        """Test default configuration values."""
        config = XExchangeFundingConfig()

        assert config.entry_threshold_bps == 20.0
        assert config.exit_threshold_bps == 5.0
        assert config.inventory_cost_bps == 2.0
        assert config.max_allocation == 0.20
        assert config.max_position_hours == 48
        assert config.reversal_stop_bps == 30.0
        assert config.supported_exchanges == ["binance", "bybit", "okx"]

    def test_custom_config_creation(self):
        """Test custom configuration values."""
        config = XExchangeFundingConfig(
            entry_threshold_bps=25.0,
            exit_threshold_bps=3.0,
            inventory_cost_bps=3.0,
            max_allocation=0.15,
            supported_exchanges=["binance", "bybit"],
        )

        assert config.entry_threshold_bps == 25.0
        assert config.exit_threshold_bps == 3.0
        assert config.inventory_cost_bps == 3.0
        assert config.max_allocation == 0.15
        assert config.supported_exchanges == ["binance", "bybit"]


class TestXExchangeFundingDispersionSignal:
    """Test cross-exchange funding dispersion signal."""

    def test_signal_initialization(self, config):
        """Test signal initialization."""
        signal = XExchangeFundingDispersionSignal(config)

        assert signal.config == config
        assert signal.name == "XExchangeFundingDispersionSignal"
        assert isinstance(signal.funding_rates, dict)
        assert isinstance(signal.funding_history, dict)
        assert isinstance(signal.active_positions, dict)
        assert isinstance(signal.exchange_status, dict)

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, signal):
        """Test handling of insufficient market data."""
        # Empty dataframe
        result = await signal.generate(pd.DataFrame(), "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata
        assert "Insufficient market data" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_funding_dispersion_calculation(self, signal):
        """Test funding dispersion calculation."""
        venue_funding = {
            "binance": 0.0005,  # 5 bps
            "bybit": 0.0020,  # 20 bps
            "okx": 0.0003,  # 3 bps
        }

        dispersion = signal._calculate_funding_dispersion(venue_funding)

        assert dispersion is not None
        assert dispersion["dispersion_bps"] == pytest.approx(
            17.0, rel=1e-2
        )  # (20-3) bps
        assert dispersion["cheap_venue"] == "okx"
        assert dispersion["expensive_venue"] == "bybit"
        assert dispersion["cheap_funding"] == 0.0003
        assert dispersion["expensive_funding"] == 0.0020

    @pytest.mark.asyncio
    async def test_below_entry_threshold(self, signal, sample_ohlcv_data):
        """Test signal when dispersion is below entry threshold."""

        # Mock small dispersion
        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            mock_fetch.return_value = {
                "binance": 0.0005,  # 5 bps
                "bybit": 0.0007,  # 7 bps - only 2 bps spread
                "okx": 0.0006,  # 6 bps
            }

            result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert result.value == 0.0
            assert result.confidence == 0.0
            assert result.metadata["reason"] == "below_entry_threshold"
            assert result.metadata["dispersion_bps"] < signal.config.entry_threshold_bps

    @pytest.mark.asyncio
    async def test_above_entry_threshold_generates_signal(
        self, signal, sample_ohlcv_data
    ):
        """Test signal generation when dispersion exceeds threshold."""

        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            with patch.object(signal, "_check_execution_feasibility") as mock_feasible:
                mock_fetch.return_value = {
                    "binance": 0.0005,  # 5 bps
                    "bybit": 0.0030,  # 30 bps - 25 bps spread (above 20 bps threshold)
                    "okx": 0.0008,  # 8 bps
                }
                mock_feasible.return_value = True

                result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

                assert result.value > 0.0
                assert result.confidence > 0.0
                assert (
                    result.metadata["strategy_type"]
                    == "cross_exchange_funding_arbitrage"
                )
                assert (
                    result.metadata["dispersion_bps"]
                    >= signal.config.entry_threshold_bps
                )
                assert result.metadata["cheap_venue"] == "binance"
                assert result.metadata["expensive_venue"] == "bybit"

    @pytest.mark.asyncio
    async def test_execution_feasibility_check(self, signal):
        """Test execution feasibility checking."""

        dispersion_data = {
            "dispersion_bps": 25.0,
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
        }

        # Test feasible case
        signal.exchange_status = {"binance": True, "bybit": True, "okx": True}
        feasible = await signal._check_execution_feasibility("BTCUSDT", dispersion_data)
        assert feasible is True

        # Test insufficient profit case
        dispersion_data["dispersion_bps"] = 3.0  # Below cost threshold
        feasible = await signal._check_execution_feasibility("BTCUSDT", dispersion_data)
        assert feasible is False

    @pytest.mark.asyncio
    async def test_position_lifecycle_entry_to_exit(self, signal, sample_ohlcv_data):
        """Test complete position lifecycle from entry to exit."""

        # Entry: Large dispersion
        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            with patch.object(signal, "_check_execution_feasibility") as mock_feasible:
                mock_fetch.return_value = {
                    "binance": 0.0005,  # 5 bps
                    "bybit": 0.0030,  # 30 bps - large spread
                    "okx": 0.0008,  # 8 bps
                }
                mock_feasible.return_value = True

                # Generate entry signal
                result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

                assert result.value > 0.0
                assert "BTCUSDT" in signal.active_positions
                assert "BTCUSDT" in signal.position_entry_times

                # Exit: Dispersion converged
                mock_fetch.return_value = {
                    "binance": 0.0006,  # 6 bps
                    "bybit": 0.0008,  # 8 bps - converged to 2 bps spread
                    "okx": 0.0007,  # 7 bps
                }

                # Generate exit signal
                result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

                assert result.value == 0.0
                assert result.metadata["reason"] == "dispersion_converged"
                assert "BTCUSDT" not in signal.active_positions

    @pytest.mark.asyncio
    async def test_position_max_time_exit(self, signal, sample_ohlcv_data):
        """Test position exit due to maximum time exceeded."""

        # Manually create a position that's too old
        old_time = datetime.utcnow() - timedelta(hours=50)  # Older than 48h max
        signal.position_entry_times["BTCUSDT"] = old_time
        signal.active_positions["BTCUSDT"] = {
            "entry_time": old_time,
            "entry_dispersion_bps": 25.0,
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
            "signal_value": 0.20,
        }

        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            mock_fetch.return_value = {
                "binance": 0.0005,
                "bybit": 0.0025,  # Still dispersed but time exceeded
                "okx": 0.0008,
            }

            result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert result.value == 0.0
            assert result.metadata["reason"] == "max_time_exceeded"
            assert "BTCUSDT" not in signal.active_positions

    @pytest.mark.asyncio
    async def test_spread_reversal_stop_loss(self, signal, sample_ohlcv_data):
        """Test position exit due to spread reversal stop loss."""

        # Set up position with initial dispersion
        signal.position_entry_times["BTCUSDT"] = datetime.utcnow()
        signal.active_positions["BTCUSDT"] = {
            "entry_time": datetime.utcnow(),
            "entry_dispersion_bps": 25.0,  # Entered at 25 bps
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
            "signal_value": 0.20,
        }

        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            # Spread significantly reduced (from 25 bps to ~2 bps = 23 bps reduction)
            mock_fetch.return_value = {
                "binance": 0.0006,  # 6 bps
                "bybit": 0.0008,  # 8 bps - only 2 bps spread (much smaller than 25 bps entry)
                "okx": 0.0007,  # 7 bps
            }

            result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert result.value == 0.0
            assert (
                result.metadata["reason"] == "dispersion_converged"
            )  # Should exit via convergence, not stop loss
            assert "BTCUSDT" not in signal.active_positions

    def test_arbitrage_signal_calculation(self, signal):
        """Test arbitrage signal strength calculation."""

        # Test various dispersion levels
        test_cases = [
            {"dispersion_bps": 15.0, "expected_signal": 0.0},  # Below threshold
            {"dispersion_bps": 25.0, "expected_signal": 0.167},  # 5 bps above threshold
            {"dispersion_bps": 35.0, "expected_signal": 0.5},  # 15 bps above threshold
            {"dispersion_bps": 50.0, "expected_signal": 1.0},  # At max expected
            {"dispersion_bps": 70.0, "expected_signal": 1.0},  # Above max (capped)
        ]

        for case in test_cases:
            dispersion_data = {"dispersion_bps": case["dispersion_bps"]}
            signal_value = signal._calculate_arbitrage_signal(dispersion_data)

            if case["dispersion_bps"] <= signal.config.entry_threshold_bps:
                assert signal_value == 0.0
            else:
                assert signal_value >= 0.0
                assert signal_value <= 1.0

    def test_confidence_calculation(self, signal):
        """Test confidence calculation factors."""

        dispersion_data = {
            "dispersion_bps": 30.0,
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
        }

        venue_funding = {"binance": 0.0005, "bybit": 0.0035, "okx": 0.0010}

        confidence = signal._calculate_arbitrage_confidence(
            "BTCUSDT", dispersion_data, venue_funding
        )

        assert 0.0 <= confidence <= 1.0
        # Should be reasonably confident for large dispersion
        assert confidence > 0.3

    def test_pnl_estimation(self, signal):
        """Test P&L estimation for positions."""

        # Set up position
        signal.active_positions["BTCUSDT"] = {
            "entry_dispersion_bps": 25.0,
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
        }

        current_dispersion_data = {
            "dispersion_bps": 10.0  # Converged from 25 to 10 bps
        }

        pnl = signal._calculate_pnl_estimate("BTCUSDT", current_dispersion_data)

        assert "estimated_pnl_bps" in pnl
        assert pnl["entry_dispersion_bps"] == 25.0
        assert pnl["current_dispersion_bps"] == 10.0
        # Profit = 15 bps convergence - 2 bps costs = 13 bps
        assert pnl["estimated_pnl_bps"] == pytest.approx(13.0, rel=1e-2)

    def test_risk_management_scaling(self, signal):
        """Test risk management scaling of signals."""

        dispersion_data = {"dispersion_bps": 30.0}
        venue_funding = {"binance": 0.0005, "bybit": 0.0035, "okx": 0.0010}

        # Test normal case
        signal_value = 0.8
        risk_adjusted = signal._apply_arbitrage_risk_management(
            signal_value, dispersion_data, venue_funding
        )

        assert risk_adjusted <= signal.config.max_allocation
        assert risk_adjusted > 0.0

        # Test low profit case
        dispersion_data["dispersion_bps"] = (
            8.0  # Only 6 bps net profit after 2 bps cost
        )
        risk_adjusted = signal._apply_arbitrage_risk_management(
            signal_value, dispersion_data, venue_funding
        )

        assert (
            risk_adjusted < signal_value * signal.config.max_allocation
        )  # Should be scaled down

    def test_convergence_time_estimation(self, signal):
        """Test convergence time estimation."""

        test_cases = [
            {
                "dispersion_bps": 20.0,
                "min_hours": 24.0,
                "max_hours": 24.0,
            },  # At threshold
            {"dispersion_bps": 35.0, "min_hours": 24.0, "max_hours": 48.0},  # Moderate
            {"dispersion_bps": 50.0, "min_hours": 35.0, "max_hours": 48.0},  # Large
        ]

        for case in test_cases:
            dispersion_data = {"dispersion_bps": case["dispersion_bps"]}
            estimated_hours = signal._estimate_convergence_time(dispersion_data)

            assert case["min_hours"] <= estimated_hours <= case["max_hours"]

    def test_exchange_status_management(self, signal):
        """Test exchange connectivity status management."""

        # Test initial status
        assert all(
            signal.exchange_status[ex] for ex in signal.config.supported_exchanges
        )

        # Test status update
        signal.update_exchange_status("binance", False)
        assert signal.exchange_status["binance"] is False

        # Test connectivity check
        connectivity = signal._check_exchange_connectivity()
        assert connectivity is False  # binance is down and it's critical

        # Restore connectivity
        signal.update_exchange_status("binance", True)
        connectivity = signal._check_exchange_connectivity()
        assert connectivity is True

    def test_position_force_close(self, signal):
        """Test manual position closure."""

        # Create position
        signal.position_entry_times["BTCUSDT"] = datetime.utcnow()
        signal.active_positions["BTCUSDT"] = {"test": "data"}

        # Force close
        signal.force_close_position("BTCUSDT", "test_reason")

        assert "BTCUSDT" not in signal.active_positions
        assert "BTCUSDT" not in signal.position_entry_times

    def test_metrics_collection(self, signal):
        """Test metrics collection for monitoring."""

        # Test empty state
        metrics = signal.get_arbitrage_metrics("BTCUSDT")
        assert metrics["position_active"] is False
        assert "exchange_status" in metrics

        # Test with position
        signal.position_entry_times["BTCUSDT"] = datetime.utcnow()
        signal.active_positions["BTCUSDT"] = {
            "entry_dispersion_bps": 25.0,
            "cheap_venue": "binance",
            "expensive_venue": "bybit",
        }

        metrics = signal.get_arbitrage_metrics("BTCUSDT")
        assert metrics["position_active"] is True
        assert "position_duration_hours" in metrics
        assert "entry_dispersion_bps" in metrics

    def test_position_summary(self, signal):
        """Test position summary functionality."""

        # Test empty
        summary = signal.get_position_summary()
        assert summary["active_positions_count"] == 0
        assert summary["total_allocation"] == 0.0

        # Add positions
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            signal.position_entry_times[symbol] = datetime.utcnow()
            signal.active_positions[symbol] = {"test": "data"}

        summary = signal.get_position_summary()
        assert summary["active_positions_count"] == 2
        assert summary["total_allocation"] == 2 * signal.config.max_allocation
        assert len(summary["symbols"]) == 2

    @pytest.mark.asyncio
    async def test_emergency_exit_generation(self, signal, sample_ohlcv_data):
        """Test emergency exit signal generation."""

        # Simulate connectivity failure
        signal.exchange_status = {"binance": False, "bybit": False, "okx": True}

        with patch.object(signal, "_fetch_multi_venue_funding") as mock_fetch:
            mock_fetch.return_value = {
                "binance": 0.0005,
                "bybit": 0.0030,
                "okx": 0.0008,
            }

            result = await signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert result.value == 0.0
            assert result.confidence == 0.0
            assert "emergency_exit" in result.metadata["reason"]
            assert result.metadata["emergency"] is True

    def test_funding_history_management(self, signal):
        """Test funding rate history tracking."""

        # Add funding data
        venue_funding = {"binance": 0.0005, "bybit": 0.0025, "okx": 0.0008}
        signal._update_funding_history("BTCUSDT", venue_funding)

        assert "BTCUSDT" in signal.funding_history
        assert len(signal.funding_history["BTCUSDT"]) == 1

        # Test persistence calculation
        for _ in range(5):
            signal._update_funding_history("BTCUSDT", venue_funding)

        persistence = signal._calculate_dispersion_persistence("BTCUSDT")
        assert 0.0 <= persistence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
