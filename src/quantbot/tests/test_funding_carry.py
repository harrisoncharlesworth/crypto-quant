import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

from quantbot.signals.funding_carry import (
    PerpFundingCarrySignal,
    FundingCarryConfig,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=24, freq="H")

    data = []
    base_price = 50000

    for i, date in enumerate(dates):
        price = base_price + np.random.normal(0, 100)
        data.append(
            {
                "datetime": date,
                "open": price,
                "high": price + abs(np.random.normal(0, 50)),
                "low": price - abs(np.random.normal(0, 50)),
                "close": price,
                "volume": np.random.uniform(50, 200),
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


@pytest.fixture
def funding_signal():
    """Create funding carry signal with test configuration."""
    config = FundingCarryConfig(
        funding_threshold=0.0007,  # 0.07%
        max_allocation=0.20,  # 20%
        reversal_stop_hours=6,
        confidence_multiplier=10.0,
        min_funding_magnitude=0.0001,
    )
    return PerpFundingCarrySignal(config)


@pytest.fixture
def high_threshold_signal():
    """Create signal with higher threshold for testing edge cases."""
    config = FundingCarryConfig(
        funding_threshold=0.0015,  # 0.15% - higher threshold
        max_allocation=0.15,
        reversal_stop_hours=4,
    )
    return PerpFundingCarrySignal(config)


@pytest.mark.asyncio
async def test_signal_initialization(funding_signal):
    """Test that the signal initializes correctly."""
    assert funding_signal.config.funding_threshold == 0.0007
    assert funding_signal.config.max_allocation == 0.20
    assert funding_signal.config.reversal_stop_hours == 6
    assert funding_signal.name == "PerpFundingCarrySignal"
    assert funding_signal.funding_history == {}
    assert funding_signal.position_entry_time is None


@pytest.mark.asyncio
async def test_insufficient_data(funding_signal):
    """Test signal behavior with insufficient market data."""
    empty_data = pd.DataFrame()

    result = await funding_signal.generate(empty_data, "BTCUSDT")

    assert result.value == 0.0
    assert result.confidence == 0.0
    assert "error" in result.metadata
    assert "Insufficient market data" in result.metadata["error"]


@pytest.mark.asyncio
async def test_long_signal_negative_funding(funding_signal, sample_ohlcv_data):
    """Test long signal generation for deeply negative funding."""
    # Mock strongly negative funding rate
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = -0.0010  # -0.10% (below -0.07% threshold)

        result = await funding_signal.generate(sample_ohlcv_data, "ETHUSDT")

        # Should generate long signal
        assert (
            result.value > 0
        ), f"Expected positive signal for negative funding, got {result.value}"
        assert result.confidence > 0, "Expected positive confidence"
        assert result.metadata["signal_direction"] == "long"
        assert result.metadata["current_funding"] == -0.0010
        assert result.metadata["strategy_type"] == "market_neutral_carry"


@pytest.mark.asyncio
async def test_short_signal_positive_funding(funding_signal, sample_ohlcv_data):
    """Test short signal generation for high positive funding."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = 0.0012  # 0.12% (above 0.07% threshold)

        result = await funding_signal.generate(sample_ohlcv_data, "ADAUSDT")

        # Should generate short signal
        assert (
            result.value < 0
        ), f"Expected negative signal for positive funding, got {result.value}"
        assert result.confidence > 0, "Expected positive confidence"
        assert result.metadata["signal_direction"] == "short"
        assert result.metadata["current_funding"] == 0.0012


@pytest.mark.asyncio
async def test_neutral_signal_within_threshold(funding_signal, sample_ohlcv_data):
    """Test neutral signal when funding is within threshold range."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = 0.0003  # 0.03% (within ±0.07% threshold)

        result = await funding_signal.generate(sample_ohlcv_data, "BTCUSDT")

        # Should generate neutral signal
        assert (
            result.value == 0.0
        ), f"Expected neutral signal for low funding, got {result.value}"
        assert result.confidence >= 0, "Confidence should be non-negative"
        assert result.metadata["signal_direction"] == "neutral"


@pytest.mark.asyncio
async def test_funding_unavailable(funding_signal, sample_ohlcv_data):
    """Test behavior when funding rate is unavailable."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = None

        result = await funding_signal.generate(sample_ohlcv_data, "UNKNOWNUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata
        assert "Funding rate unavailable" in result.metadata["error"]


@pytest.mark.asyncio
async def test_confidence_calculation_extreme_funding(
    funding_signal, sample_ohlcv_data
):
    """Test confidence calculation for extreme funding rates."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        # Very extreme negative funding
        mock_funding.return_value = -0.0020  # -0.20%

        result = await funding_signal.generate(sample_ohlcv_data, "EXTREMEUSDT")

        # Should have high confidence for extreme rates
        assert (
            result.confidence > 0.5
        ), f"Expected high confidence for extreme funding, got {result.confidence}"
        assert result.value > 0, "Should be long for extreme negative funding"


@pytest.mark.asyncio
async def test_signal_strength_scaling(funding_signal, sample_ohlcv_data):
    """Test that signal strength scales with funding magnitude."""

    # Test different funding levels
    funding_levels = [-0.0008, -0.0015, -0.0030]  # Increasingly negative
    signal_values = []

    for funding_rate in funding_levels:
        with patch.object(
            funding_signal, "_get_funding_rate", new_callable=AsyncMock
        ) as mock_funding:
            mock_funding.return_value = funding_rate
            result = await funding_signal.generate(sample_ohlcv_data, "TESTUSDT")
            signal_values.append(result.value)

    # Signal strength should generally increase with funding magnitude
    # (though bounded by risk management)
    assert signal_values[0] > 0, "Should be positive for negative funding"
    assert (
        signal_values[1] > signal_values[0] or signal_values[1] > 0.15
    ), "Stronger funding should give stronger signal (or hit risk limit)"


@pytest.mark.asyncio
async def test_max_allocation_enforcement(funding_signal, sample_ohlcv_data):
    """Test that signals don't exceed maximum allocation."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        # Extreme funding that would generate large signal
        mock_funding.return_value = -0.0050  # -0.50%

        result = await funding_signal.generate(sample_ohlcv_data, "EXTREMEUSDT")

        # Should be capped at max allocation
        assert (
            abs(result.value) <= funding_signal.config.max_allocation
        ), f"Signal {result.value} exceeds max allocation {funding_signal.config.max_allocation}"


@pytest.mark.asyncio
async def test_minimum_funding_magnitude_filter(funding_signal, sample_ohlcv_data):
    """Test filtering of very small funding rates."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        # Very small funding below minimum threshold
        mock_funding.return_value = 0.00005  # 0.005%

        result = await funding_signal.generate(sample_ohlcv_data, "SMALLUSDT")

        # Should generate no signal for tiny funding
        assert result.value == 0.0, "Should ignore very small funding rates"
        assert result.metadata["signal_direction"] == "neutral"


@pytest.mark.asyncio
async def test_funding_history_tracking(funding_signal, sample_ohlcv_data):
    """Test funding rate history tracking."""
    funding_rates = [-0.0008, -0.0006, -0.0004, 0.0002]

    for i, rate in enumerate(funding_rates):
        with patch.object(
            funding_signal, "_get_funding_rate", new_callable=AsyncMock
        ) as mock_funding:
            mock_funding.return_value = rate
            await funding_signal.generate(sample_ohlcv_data, "TRACKUSDT")

    # Should have history for the symbol
    assert "TRACKUSDT" in funding_signal.funding_history
    assert len(funding_signal.funding_history["TRACKUSDT"]) == len(funding_rates)

    # Check that rates were stored correctly
    stored_rates = [rate for _, rate in funding_signal.funding_history["TRACKUSDT"]]
    assert stored_rates == funding_rates


@pytest.mark.asyncio
async def test_reversal_stop_condition(funding_signal, sample_ohlcv_data):
    """Test funding reversal stop condition."""

    # Simulate funding reversal scenario
    # Start with strong negative funding, then reverse to positive
    funding_sequence = [-0.0010] * 3 + [0.0008] * 7  # 7 hours of reversal

    for rate in funding_sequence:
        with patch.object(
            funding_signal, "_get_funding_rate", new_callable=AsyncMock
        ) as mock_funding:
            mock_funding.return_value = rate
            result = await funding_signal.generate(sample_ohlcv_data, "REVERSALUSDT")

    # Last signal should be stopped due to reversal
    assert result.value == 0.0, "Signal should be stopped after reversal"
    assert result.confidence == 0.0
    assert result.metadata["reason"] == "funding_reversal_stop"
    assert (
        result.metadata["reversal_hours"] >= funding_signal.config.reversal_stop_hours
    )


@pytest.mark.asyncio
async def test_position_time_limits(funding_signal, sample_ohlcv_data):
    """Test maximum position time enforcement."""

    # Set position entry time to simulate long-held position
    funding_signal.set_position_entry(1)  # Long position
    funding_signal.position_entry_time = datetime.utcnow() - timedelta(
        hours=25
    )  # 25 hours ago

    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = -0.0010  # Strong negative funding

        result = await funding_signal.generate(sample_ohlcv_data, "TIMEUSDT")

    # Should be stopped due to time limit
    assert result.value == 0.0, "Signal should be stopped after max time"
    assert result.metadata["reason"] == "max_position_time_exceeded"
    assert (
        result.metadata["hours_in_position"] > funding_signal.config.max_position_hours
    )


@pytest.mark.asyncio
async def test_risk_management_extreme_rates(funding_signal, sample_ohlcv_data):
    """Test risk management for extremely high funding rates."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        # Emergency exit threshold rate
        mock_funding.return_value = 0.025  # 2.5% (above 2% emergency threshold)

        result = await funding_signal.generate(sample_ohlcv_data, "EMERGENCYUSDT")

        # Should drastically reduce exposure
        assert (
            abs(result.value) < 0.05
        ), "Should have very small signal for emergency rates"


@pytest.mark.asyncio
async def test_strategy_metrics(funding_signal, sample_ohlcv_data):
    """Test strategy metrics calculation."""

    # Generate some funding history
    funding_rates = [-0.0008, -0.0006, 0.0004, 0.0008, -0.0002]

    for rate in funding_rates:
        with patch.object(
            funding_signal, "_get_funding_rate", new_callable=AsyncMock
        ) as mock_funding:
            mock_funding.return_value = rate
            await funding_signal.generate(sample_ohlcv_data, "METRICSUSDT")

    metrics = funding_signal.get_strategy_metrics("METRICSUSDT")

    # Check metrics are calculated
    assert "current_funding" in metrics
    assert "avg_funding_24h" in metrics
    assert "funding_volatility" in metrics
    assert "reversal_hours" in metrics
    assert "extreme_funding_periods" in metrics
    assert "funding_direction_changes" in metrics

    assert metrics["current_funding"] == funding_rates[-1]
    assert isinstance(metrics["avg_funding_24h"], float)
    assert metrics["funding_direction_changes"] > 0  # Should detect direction changes


@pytest.mark.asyncio
async def test_position_tracking_methods(funding_signal):
    """Test position tracking utility methods."""

    # Initially no position
    assert funding_signal._hours_in_position() == 0.0
    assert not funding_signal._is_position_time_exceeded()

    # Set position entry
    funding_signal.set_position_entry(1)

    assert funding_signal.position_entry_time is not None
    assert funding_signal.last_funding_direction == 1
    assert funding_signal._hours_in_position() >= 0

    # Reset position
    funding_signal.reset_position_tracking()

    assert funding_signal.position_entry_time is None
    assert funding_signal.last_funding_direction is None
    assert funding_signal._hours_in_position() == 0.0


@pytest.mark.asyncio
async def test_signal_bounds_enforcement(funding_signal, sample_ohlcv_data):
    """Test that all signals are properly bounded."""

    # Test with various extreme funding rates
    extreme_rates = [-0.01, -0.005, 0.005, 0.01, 0.03, -0.03]

    for rate in extreme_rates:
        with patch.object(
            funding_signal, "_get_funding_rate", new_callable=AsyncMock
        ) as mock_funding:
            mock_funding.return_value = rate
            result = await funding_signal.generate(sample_ohlcv_data, "BOUNDSUSDT")

            # Check bounds
            assert (
                -1.0 <= result.value <= 1.0
            ), f"Signal {result.value} outside bounds for funding {rate}"
            assert (
                0.0 <= result.confidence <= 1.0
            ), f"Confidence {result.confidence} outside bounds"


@pytest.mark.asyncio
async def test_metadata_completeness(funding_signal, sample_ohlcv_data):
    """Test that signal metadata contains all expected fields."""
    with patch.object(
        funding_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = -0.0010

        result = await funding_signal.generate(sample_ohlcv_data, "METAUSDT")

    required_metadata = [
        "current_funding",
        "funding_threshold",
        "signal_direction",
        "funding_magnitude",
        "confidence_raw",
        "risk_adjustment",
        "funding_history_length",
        "reversal_hours",
        "position_hours",
        "max_allocation",
        "strategy_type",
    ]

    for key in required_metadata:
        assert key in result.metadata, f"Missing metadata key: {key}"

    # Check specific values
    assert result.metadata["strategy_type"] == "market_neutral_carry"
    assert result.metadata["max_allocation"] == funding_signal.config.max_allocation


@pytest.mark.asyncio
async def test_error_handling(funding_signal, sample_ohlcv_data):
    """Test error handling in signal generation."""

    # Force an exception in funding rate calculation
    with patch.object(
        funding_signal, "_get_funding_rate", side_effect=Exception("Test error")
    ):
        result = await funding_signal.generate(sample_ohlcv_data, "ERRORUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "error" in result.metadata
        assert "Test error" in result.metadata["error"]


@pytest.mark.asyncio
async def test_different_threshold_configurations(
    high_threshold_signal, sample_ohlcv_data
):
    """Test signal behavior with different threshold configurations."""

    # Test with funding that's above original threshold but below new one
    with patch.object(
        high_threshold_signal, "_get_funding_rate", new_callable=AsyncMock
    ) as mock_funding:
        mock_funding.return_value = -0.0010  # -0.10%

        result = await high_threshold_signal.generate(sample_ohlcv_data, "THRESHUSDT")

        # Should be neutral with higher threshold (0.15% vs 0.07%)
        assert result.value == 0.0, "Should be neutral with higher threshold"
        assert result.metadata["signal_direction"] == "neutral"


@pytest.mark.asyncio
async def test_mock_funding_rate_variation():
    """Test the mock funding rate implementation."""
    config = FundingCarryConfig()
    signal = PerpFundingCarrySignal(config)

    # Test different symbols return different base rates
    btc_rate = await signal._get_funding_rate("BTCUSDT")
    eth_rate = await signal._get_funding_rate("ETHUSDT")

    assert btc_rate is not None
    assert eth_rate is not None
    assert isinstance(btc_rate, float)
    assert isinstance(eth_rate, float)

    # Rates should be in reasonable range
    assert -0.01 < btc_rate < 0.01  # Within ±1%
    assert -0.01 < eth_rate < 0.01
