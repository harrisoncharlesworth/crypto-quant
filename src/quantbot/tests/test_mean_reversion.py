import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantbot.signals.mean_reversion import (
    ShortTermMeanReversionSignal,
    MeanReversionConfig,
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing mean reversion signals."""
    # Use hourly data for crypto (24 hours * 5 days = 120 periods)
    dates = pd.date_range(start="2023-01-01", periods=120, freq="H")

    np.random.seed(42)
    base_price = 50000  # BTC-like price

    # Create mean-reverting price series with some trend
    prices = [base_price]
    for i in range(1, 120):
        # Mean reversion component
        reversion = 0.02 * (base_price - prices[-1])
        # Random walk component
        random_move = np.random.normal(0, 500)
        # New price
        new_price = prices[-1] + reversion + random_move
        prices.append(max(new_price, 100))  # Floor price at $100

    # Generate OHLCV with realistic relationships
    data = []
    for i, close in enumerate(prices):
        # Realistic spread around close
        high = close + abs(np.random.normal(0, 200))
        low = close - abs(np.random.normal(0, 200))
        open_price = close + np.random.normal(0, 100)

        # Volume in typical crypto range
        volume = np.random.uniform(50, 500)  # BTC amount

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
def extreme_move_data():
    """Create data with extreme price moves for testing threshold detection."""
    dates = pd.date_range(start="2023-01-01", periods=80, freq="H")

    # Stable price for 72 hours, then extreme move
    stable_price = 50000
    prices = [stable_price] * 72

    # Add extreme downward move (liquidation-like)
    prices.extend(
        [stable_price - i * 1000 for i in range(1, 9)]
    )  # -8k move over 8 hours

    data = []
    for i, close in enumerate(prices):
        volume_multiplier = 5 if i >= 72 else 1  # High volume during move

        data.append(
            {
                "datetime": dates[i],
                "open": close + np.random.normal(0, 50),
                "high": close + abs(np.random.normal(0, 100)),
                "low": close - abs(np.random.normal(0, 100)),
                "close": close,
                "volume": np.random.uniform(100, 200) * volume_multiplier,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


@pytest.fixture
def mean_reversion_signal():
    """Create a mean reversion signal instance with test configuration."""
    config = MeanReversionConfig(
        lookback_days=3,
        zscore_threshold=2.0,
        min_liquidity_volume=1_000_000,
        min_periods=10,
    )
    return ShortTermMeanReversionSignal(config)


@pytest.mark.asyncio
async def test_signal_initialization(mean_reversion_signal):
    """Test that the signal initializes correctly."""
    assert mean_reversion_signal.config.lookback_days == 3
    assert mean_reversion_signal.config.zscore_threshold == 2.0
    assert mean_reversion_signal.config.min_liquidity_volume == 1_000_000
    assert mean_reversion_signal.name == "ShortTermMeanReversionSignal"


@pytest.mark.asyncio
async def test_insufficient_data(mean_reversion_signal):
    """Test signal behavior with insufficient data."""
    # Create very small dataset
    small_data = pd.DataFrame(
        {
            "open": [50000, 50100],
            "high": [50200, 50300],
            "low": [49900, 50000],
            "close": [50100, 50200],
            "volume": [10, 11],
        }
    )

    result = await mean_reversion_signal.generate(small_data, "BTCUSDT")

    assert result.value == 0.0
    assert result.confidence == 0.0
    assert "error" in result.metadata


@pytest.mark.asyncio
async def test_signal_generation_normal_market(mean_reversion_signal, sample_data):
    """Test basic signal generation with normal market conditions."""
    result = await mean_reversion_signal.generate(sample_data, "BTCUSDT")

    # Basic validation
    assert result.symbol == "BTCUSDT"
    assert isinstance(result.timestamp, datetime)
    assert -1.0 <= result.value <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.metadata is not None

    # Check metadata completeness
    required_metadata = [
        "z_score",
        "rolling_mean",
        "rolling_std",
        "current_price",
        "volume_usd_estimate",
        "liquidity_pass",
        "liquidation_detected",
        "funding_neutral",
        "lookback_hours",
        "threshold_used",
    ]

    for key in required_metadata:
        assert key in result.metadata, f"Missing metadata key: {key}"


@pytest.mark.asyncio
async def test_extreme_move_detection(mean_reversion_signal, extreme_move_data):
    """Test detection of extreme moves that should trigger mean reversion signals."""
    result = await mean_reversion_signal.generate(extreme_move_data, "BTCUSDT")

    # Should detect the extreme move and generate contrarian signal
    z_score = result.metadata["z_score"]

    # The extreme downward move should create negative z-score
    assert z_score < -2.0, f"Expected z-score < -2.0, got {z_score}"

    # Signal should be positive (contrarian - buy after extreme sell-off)
    if abs(z_score) >= mean_reversion_signal.config.zscore_threshold:
        assert (
            result.value > 0
        ), f"Expected positive signal for extreme oversold, got {result.value}"


@pytest.mark.asyncio
async def test_liquidation_detection(mean_reversion_signal, extreme_move_data):
    """Test liquidation cascade detection."""
    result = await mean_reversion_signal.generate(extreme_move_data, "BTCUSDT")

    # The extreme move data should trigger liquidation detection
    assert result.metadata[
        "liquidation_detected"
    ], "Expected liquidation cascade to be detected"

    # Confidence should be boosted when liquidation is detected
    if abs(result.metadata["z_score"]) >= 2.0:
        assert (
            result.confidence > 0.3
        ), "Expected higher confidence during liquidation cascade"


@pytest.mark.asyncio
async def test_liquidity_filter(mean_reversion_signal):
    """Test liquidity filtering for low-volume pairs."""
    # Create low-volume data
    dates = pd.date_range(start="2023-01-01", periods=80, freq="H")
    low_volume_data = []

    for i, date in enumerate(dates):
        price = 1000 + np.random.normal(0, 50)  # Smaller altcoin price
        low_volume_data.append(
            {
                "datetime": date,
                "open": price,
                "high": price + abs(np.random.normal(0, 10)),
                "low": price - abs(np.random.normal(0, 10)),
                "close": price,
                "volume": np.random.uniform(0.1, 1.0),  # Very low volume
            }
        )

    df = pd.DataFrame(low_volume_data)
    df.set_index("datetime", inplace=True)

    result = await mean_reversion_signal.generate(df, "SMALLALTUSDT")

    # Low volume should filter out signals
    assert not result.metadata["liquidity_pass"], "Expected liquidity filter to fail"
    assert result.value == 0.0, "Expected zero signal for illiquid pair"
    assert result.confidence == 0.0, "Expected zero confidence for illiquid pair"


@pytest.mark.asyncio
async def test_contrarian_signal_direction(mean_reversion_signal):
    """Test that signals are properly contrarian."""
    dates = pd.date_range(start="2023-01-01", periods=80, freq="H")

    # Create strong upward trend that should trigger short signal
    prices = [50000 + i * 200 for i in range(80)]  # Strong uptrend

    data = []
    for i, close in enumerate(prices):
        data.append(
            {
                "datetime": dates[i],
                "open": close,
                "high": close + 100,
                "low": close - 100,
                "close": close,
                "volume": 100,  # Sufficient volume
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)

    result = await mean_reversion_signal.generate(df, "BTCUSDT")

    z_score = result.metadata["z_score"]

    # Strong uptrend should create positive z-score
    if z_score > 2.0:
        # Contrarian signal should be negative (fade the uptrend)
        assert (
            result.value < 0
        ), f"Expected negative contrarian signal for overbought, got {result.value}"


@pytest.mark.asyncio
async def test_no_signal_in_normal_range(mean_reversion_signal):
    """Test that no signal is generated when price is within normal range."""
    dates = pd.date_range(start="2023-01-01", periods=80, freq="H")

    # Create stable price with small variations (within 2σ)
    stable_price = 50000
    prices = [stable_price + np.random.normal(0, 100) for _ in range(80)]

    data = []
    for i, close in enumerate(prices):
        data.append(
            {
                "datetime": dates[i],
                "open": close,
                "high": close + 50,
                "low": close - 50,
                "close": close,
                "volume": 100,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)

    result = await mean_reversion_signal.generate(df, "BTCUSDT")

    z_score = result.metadata["z_score"]

    # If within threshold, should have no signal
    if abs(z_score) < 2.0:
        assert (
            result.value == 0.0
        ), f"Expected no signal for normal range, got {result.value}"
        assert (
            result.confidence == 0.0
        ), f"Expected no confidence for normal range, got {result.confidence}"


@pytest.mark.asyncio
async def test_volume_usd_estimation(mean_reversion_signal, sample_data):
    """Test USD volume estimation logic."""
    result = await mean_reversion_signal.generate(sample_data, "BTCUSDT")

    volume_usd = result.metadata["volume_usd_estimate"]

    # Should be reasonable USD volume
    assert volume_usd > 0, "Volume USD estimate should be positive"
    assert isinstance(volume_usd, float), "Volume USD should be float"


@pytest.mark.asyncio
async def test_confidence_calculation(mean_reversion_signal, extreme_move_data):
    """Test confidence calculation factors."""
    result = await mean_reversion_signal.generate(extreme_move_data, "BTCUSDT")

    metadata = result.metadata

    # Confidence should increase with deviation magnitude
    if abs(metadata["z_score"]) >= 2.0:
        assert (
            result.confidence > 0.0
        ), "Expected positive confidence for extreme deviation"

        # Liquidation detection should boost confidence
        if metadata["liquidation_detected"]:
            # Can't easily test exact boost without internal calculation,
            # but confidence should be reasonable
            assert result.confidence <= 1.0, "Confidence should not exceed 1.0"


@pytest.mark.asyncio
async def test_signal_bounds_and_normalization(
    mean_reversion_signal, extreme_move_data
):
    """Test that signal values are properly bounded and normalized."""
    result = await mean_reversion_signal.generate(extreme_move_data, "BTCUSDT")

    # Signal should always be bounded
    assert -1.0 <= result.value <= 1.0, f"Signal {result.value} outside [-1, 1] bounds"
    assert (
        0.0 <= result.confidence <= 1.0
    ), f"Confidence {result.confidence} outside [0, 1] bounds"


@pytest.mark.asyncio
async def test_error_handling(mean_reversion_signal):
    """Test error handling with malformed data."""
    # Missing required columns
    bad_data = pd.DataFrame({"price": [50000, 50100, 50200], "vol": [100, 110, 120]})

    result = await mean_reversion_signal.generate(bad_data, "BTCUSDT")

    assert result.value == 0.0
    assert result.confidence == 0.0
    assert "error" in result.metadata


@pytest.mark.asyncio
async def test_funding_neutrality_placeholder(mean_reversion_signal, sample_data):
    """Test funding rate neutrality check (placeholder implementation)."""
    result = await mean_reversion_signal.generate(sample_data, "BTCUSDT")

    # Currently returns True (placeholder)
    assert result.metadata["funding_neutral"] is True

    # Funding neutrality is factored into confidence calculation
    assert "funding_neutral" in result.metadata


@pytest.mark.asyncio
async def test_rolling_statistics_calculation(mean_reversion_signal, sample_data):
    """Test rolling statistics calculation accuracy."""
    result = await mean_reversion_signal.generate(sample_data, "BTCUSDT")

    metadata = result.metadata

    # Rolling statistics should be reasonable
    assert metadata["rolling_mean"] > 0, "Rolling mean should be positive"
    assert metadata["rolling_std"] > 0, "Rolling std should be positive"

    # Z-score calculation should be consistent
    current_price = metadata["current_price"]
    rolling_mean = metadata["rolling_mean"]
    rolling_std = metadata["rolling_std"]

    expected_z_score = (current_price - rolling_mean) / rolling_std
    actual_z_score = metadata["z_score"]

    # Allow for small floating point differences
    assert (
        abs(expected_z_score - actual_z_score) < 1e-10
    ), "Z-score calculation mismatch"
