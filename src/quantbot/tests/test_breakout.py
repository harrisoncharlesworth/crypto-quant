import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantbot.signals.breakout import DonchianBreakoutSignal, BreakoutConfig


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

    # Create trending price data with some volatility
    np.random.seed(42)
    base_price = 100
    trend = np.linspace(0, 30, 200)  # Uptrend
    noise = np.random.normal(0, 2, 200)

    close_prices = base_price + trend + noise

    # Generate OHLCV with realistic relationships
    data = []
    for i, close in enumerate(close_prices):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = close + np.random.normal(0, 0.5)
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "datetime": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


@pytest.fixture
def breakout_signal():
    """Create a DonchianBreakoutSignal instance with test configuration."""
    config = BreakoutConfig(
        channel_period=55, atr_period=14, atr_multiplier=2.0, min_periods=60
    )
    return DonchianBreakoutSignal(config)


@pytest.mark.asyncio
async def test_breakout_signal_initialization(breakout_signal):
    """Test that the signal initializes correctly."""
    assert breakout_signal.config.channel_period == 55
    assert breakout_signal.config.atr_period == 14
    assert breakout_signal.config.atr_multiplier == 2.0
    assert breakout_signal.name == "DonchianBreakoutSignal"


@pytest.mark.asyncio
async def test_insufficient_data(breakout_signal):
    """Test signal behavior with insufficient data."""
    # Create very small dataset
    small_data = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
        }
    )

    result = await breakout_signal.generate(small_data, "BTCUSDT")

    assert result.value == 0.0
    assert result.confidence == 0.0
    assert "error" in result.metadata


@pytest.mark.asyncio
async def test_atr_calculation(breakout_signal, sample_data):
    """Test ATR calculation."""
    atr = breakout_signal.calculate_atr(sample_data, 14)

    # ATR should be positive and not NaN for sufficient data
    assert len(atr) == len(sample_data)
    assert not pd.isna(atr.iloc[-1])
    assert atr.iloc[-1] > 0


@pytest.mark.asyncio
async def test_signal_generation(breakout_signal, sample_data):
    """Test basic signal generation."""
    result = await breakout_signal.generate(sample_data, "BTCUSDT")

    # Basic checks
    assert result.symbol == "BTCUSDT"
    assert isinstance(result.timestamp, datetime)
    assert -1.0 <= result.value <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.metadata is not None


@pytest.mark.asyncio
async def test_breakout_detection(breakout_signal):
    """Test breakout detection logic."""
    # Create data with clear breakout pattern
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Sideways market followed by breakout
    prices = [100] * 60  # Sideways for 60 days
    prices.extend([100 + i for i in range(1, 41)])  # Strong uptrend

    data = []
    for i, close in enumerate(prices):
        data.append(
            {
                "datetime": dates[i],
                "open": close + np.random.normal(0, 0.1),
                "high": close + abs(np.random.normal(0, 0.5)),
                "low": close - abs(np.random.normal(0, 0.5)),
                "close": close,
                "volume": 1000,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)

    result = await breakout_signal.generate(df, "BTCUSDT")

    # Should detect upward breakout
    assert result.value > 0
    assert "breakout_type" in result.metadata
    assert "high_channel" in result.metadata
    assert "current_atr" in result.metadata


@pytest.mark.asyncio
async def test_choppy_market_detection(breakout_signal, sample_data):
    """Test choppy market detection."""
    # Create choppy/sideways market data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Oscillating prices (choppy market)
    base_price = 100
    choppy_prices = [
        base_price + 5 * np.sin(i * 0.3) + np.random.normal(0, 1) for i in range(100)
    ]

    data = []
    for i, close in enumerate(choppy_prices):
        data.append(
            {
                "datetime": dates[i],
                "open": close + np.random.normal(0, 0.1),
                "high": close + abs(np.random.normal(0, 0.5)),
                "low": close - abs(np.random.normal(0, 0.5)),
                "close": close,
                "volume": 1000,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)

    result = await breakout_signal.generate(df, "BTCUSDT")

    # Check that choppy market detection works
    assert "is_choppy" in result.metadata
    assert isinstance(result.metadata["is_choppy"], (bool, np.bool_))


@pytest.mark.asyncio
async def test_metadata_completeness(breakout_signal, sample_data):
    """Test that all expected metadata is present."""
    result = await breakout_signal.generate(sample_data, "BTCUSDT")

    expected_keys = [
        "breakout_type",
        "high_channel",
        "low_channel",
        "current_atr",
        "atr_stop_long",
        "atr_stop_short",
        "channel_width",
        "channel_width_pct",
        "is_choppy",
        "adx",
        "channel_period",
        "atr_multiplier",
    ]

    for key in expected_keys:
        assert key in result.metadata, f"Missing metadata key: {key}"


@pytest.mark.asyncio
async def test_signal_bounds(breakout_signal, sample_data):
    """Test that signal values are properly bounded."""
    result = await breakout_signal.generate(sample_data, "BTCUSDT")

    # Signal should be bounded between -1 and 1
    assert -1.0 <= result.value <= 1.0
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_error_handling(breakout_signal):
    """Test error handling with malformed data."""
    # Missing required columns
    bad_data = pd.DataFrame({"price": [100, 101, 102], "vol": [1000, 1100, 1200]})

    result = await breakout_signal.generate(bad_data, "BTCUSDT")

    assert result.value == 0.0
    assert result.confidence == 0.0
    assert "error" in result.metadata
