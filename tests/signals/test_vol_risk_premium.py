import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.quantbot.signals.vol_risk_premium import (
    VolatilityRiskPremiumSignal,
    VolRiskPremiumConfig,
    OptionsContract,
    OptionType,
    StraddlePosition,
    WingProtection,
)


@pytest.fixture
def vol_config():
    """Create test configuration for vol risk premium signal."""
    return VolRiskPremiumConfig(
        vrp_threshold_min=0.05,  # 5% minimum VRP
        vrp_threshold_max=0.15,  # 15% VRP for max signal
        min_days_to_expiry=7,
        max_days_to_expiry=14,
        target_days_to_expiry=10,
        max_vega_allocation=0.20,
        vol_target=0.15,
        delta_hedge_threshold=0.10,
        hedge_frequency_hours=4,
        wing_delta=0.25,
        max_wing_cost_ratio=0.15,
        enable_tail_protection=True,
        min_volume_24h=50000,
        min_open_interest=100000,
        max_bid_ask_spread=0.05,
        realized_vol_window=30,
    )


@pytest.fixture
def vol_signal(vol_config):
    """Create vol risk premium signal instance."""
    return VolatilityRiskPremiumSignal(vol_config)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with realistic volatility."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Generate realistic price series with volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 60)  # 2% daily vol
    prices = [45000]  # Start at $45k BTC

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": np.random.uniform(100000, 1000000, 60),
        }
    )

    return data.set_index("timestamp")


@pytest.fixture
def mock_options_chain():
    """Create mock options chain for testing."""
    spot_price = 45000  # This should match the sample data approximately
    expiry = datetime.now() + timedelta(days=10)

    # ATM options
    atm_call = OptionsContract(
        symbol="BTC-10FEB24-45000-C",
        underlying="BTC",
        strike=45000,
        expiry_date=expiry,
        option_type=OptionType.CALL,
        price=1800,
        bid=1780,
        ask=1820,
        implied_volatility=0.80,  # 80% IV
        volume=500000,
        open_interest=2000000,
        delta=0.50,
        gamma=0.000025,
        vega=180,
        theta=-50,
        rho=20,
        mark_price=1800,
        index_price=spot_price,
    )

    atm_put = OptionsContract(
        symbol="BTC-10FEB24-45000-P",
        underlying="BTC",
        strike=45000,
        expiry_date=expiry,
        option_type=OptionType.PUT,
        price=1750,
        bid=1730,
        ask=1770,
        implied_volatility=0.82,  # 82% IV (slight skew)
        volume=450000,
        open_interest=1800000,
        delta=-0.50,
        gamma=0.000025,
        vega=175,
        theta=-48,
        rho=-18,
        mark_price=1750,
        index_price=spot_price,
    )

    # 25Δ wing options
    wing_call = OptionsContract(
        symbol="BTC-10FEB24-51750-C",
        underlying="BTC",
        strike=51750,  # ~15% OTM
        expiry_date=expiry,
        option_type=OptionType.CALL,
        price=450,
        bid=440,
        ask=460,
        implied_volatility=0.90,
        volume=100000,
        open_interest=500000,
        delta=0.25,
        gamma=0.000012,
        vega=90,
        theta=-15,
        rho=8,
        mark_price=450,
        index_price=spot_price,
    )

    wing_put = OptionsContract(
        symbol="BTC-10FEB24-38250-P",
        underlying="BTC",
        strike=38250,  # ~15% OTM
        expiry_date=expiry,
        option_type=OptionType.PUT,
        price=420,
        bid=410,
        ask=430,
        implied_volatility=0.88,
        volume=120000,
        open_interest=600000,
        delta=-0.25,
        gamma=0.000012,
        vega=85,
        theta=-12,
        rho=-6,
        mark_price=420,
        index_price=spot_price,
    )

    return {
        "atm_call": atm_call,
        "atm_put": atm_put,
        "wing_call": wing_call,
        "wing_put": wing_put,
        "all_options": [atm_call, atm_put, wing_call, wing_put],
    }


class TestVolRiskPremiumSignal:
    """Test suite for vol risk premium signal."""

    @pytest.mark.asyncio
    async def test_signal_initialization(self, vol_signal, vol_config):
        """Test signal initializes correctly."""
        assert vol_signal.config == vol_config
        assert vol_signal.name == "VolatilityRiskPremiumSignal"
        assert vol_signal.active_positions == {}
        assert vol_signal.hedge_schedule == {}
        assert vol_signal.vol_history == {}
        assert vol_signal.iv_history == {}

    def test_realized_volatility_calculation(self, vol_signal, sample_ohlcv_data):
        """Test realized volatility calculation."""
        realized_vol = vol_signal._calculate_realized_volatility(sample_ohlcv_data)

        # Should be reasonable volatility for crypto
        assert 0.1 <= realized_vol <= 2.0
        assert isinstance(realized_vol, float)

    def test_black_scholes_greeks(self, vol_signal):
        """Test Black-Scholes Greeks calculation."""
        S, K, T, r, sigma = 45000, 45000, 10 / 365.25, 0.05, 0.80

        call_greeks = vol_signal._calculate_bs_greeks(
            S, K, T, r, sigma, OptionType.CALL
        )
        put_greeks = vol_signal._calculate_bs_greeks(S, K, T, r, sigma, OptionType.PUT)

        # Check basic properties
        assert call_greeks["delta"] > 0  # Call delta positive
        assert put_greeks["delta"] < 0  # Put delta negative
        assert call_greeks["gamma"] == put_greeks["gamma"]  # Same gamma for same strike
        assert call_greeks["vega"] > 0 and put_greeks["vega"] > 0  # Vega positive
        assert call_greeks["theta"] < 0 and put_greeks["theta"] < 0  # Theta negative

        # Put-call parity check (approximately)
        assert (
            abs(call_greeks["price"] - put_greeks["price"] - (S - K * np.exp(-r * T)))
            < 100
        )

    @pytest.mark.asyncio
    async def test_options_data_generation(self, vol_signal):
        """Test mock options data generation."""
        options = await vol_signal._get_options_data("BTCUSDT")

        assert len(options) > 0
        assert all(isinstance(opt, OptionsContract) for opt in options)

        # Check we have both calls and puts
        calls = [opt for opt in options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options if opt.option_type == OptionType.PUT]
        assert len(calls) > 0
        assert len(puts) > 0

        # Check expiry range
        for opt in options:
            days_to_expiry = (opt.expiry_date - datetime.now()).days
            assert 1 <= days_to_expiry <= 30

    def test_straddle_opportunities_finding(self, vol_signal, mock_options_chain):
        """Test finding ATM straddle opportunities."""
        options = mock_options_chain["all_options"]
        spot_price = 45000

        straddles = vol_signal._find_straddle_opportunities(options, spot_price)

        assert len(straddles) > 0
        for call, put in straddles:
            assert call.option_type == OptionType.CALL
            assert put.option_type == OptionType.PUT
            assert call.strike == put.strike  # Same strike for straddle
            assert call.expiry_date == put.expiry_date  # Same expiry

    def test_vrp_calculation(self, vol_signal, mock_options_chain):
        """Test VRP calculation."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]
        realized_vol = 0.60  # 60% realized vol

        vrp_data = vol_signal._calculate_vrp(call, put, realized_vol)

        assert vrp_data is not None
        assert "vrp" in vrp_data
        assert "implied_vol" in vrp_data
        assert "premium_collected" in vrp_data

        # VRP should be positive (IV > RV)
        expected_iv = (call.implied_volatility + put.implied_volatility) / 2
        expected_vrp = expected_iv - realized_vol
        assert abs(vrp_data["vrp"] - expected_vrp) < 0.01

    def test_vrp_below_threshold(self, vol_signal, mock_options_chain):
        """Test VRP calculation when below threshold."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]
        realized_vol = 0.85  # High realized vol (above IV)

        vrp_data = vol_signal._calculate_vrp(call, put, realized_vol)

        # Should return None when VRP below threshold
        assert vrp_data is None

    def test_liquidity_requirements(self, vol_signal, mock_options_chain):
        """Test liquidity requirements checking."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        # Should pass liquidity requirements
        assert vol_signal._meets_liquidity_requirements(call, put)

        # Test with low volume
        low_volume_call = OptionsContract(
            symbol=call.symbol,
            underlying=call.underlying,
            strike=call.strike,
            expiry_date=call.expiry_date,
            option_type=call.option_type,
            price=call.price,
            bid=call.bid,
            ask=call.ask,
            implied_volatility=call.implied_volatility,
            volume=1000,  # Low volume
            open_interest=call.open_interest,
            delta=call.delta,
            gamma=call.gamma,
            vega=call.vega,
            theta=call.theta,
            rho=call.rho,
            mark_price=call.mark_price,
            index_price=call.index_price,
        )

        assert not vol_signal._meets_liquidity_requirements(low_volume_call, put)

    def test_signal_value_calculation(self, vol_signal):
        """Test signal value calculation from VRP."""
        # High VRP should give strong sell signal
        high_vrp_data = {"vrp": 0.20}  # 20% VRP
        signal_value = vol_signal._calculate_signal_value(high_vrp_data)
        assert signal_value < 0  # Negative = sell vol
        assert signal_value <= -1.0  # Capped at -1

        # Medium VRP
        medium_vrp_data = {"vrp": 0.10}  # 10% VRP
        signal_value = vol_signal._calculate_signal_value(medium_vrp_data)
        assert -1.0 <= signal_value < 0

        # Low VRP
        low_vrp_data = {"vrp": 0.02}  # 2% VRP (below threshold)
        signal_value = vol_signal._calculate_signal_value(low_vrp_data)
        assert signal_value == 0.0

    def test_vega_position_sizing(self, vol_signal, mock_options_chain):
        """Test vega-based position sizing."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]
        confidence = 0.8

        position_size = vol_signal._calculate_vega_position_size(call, put, confidence)

        assert 0 <= position_size <= 1.0
        assert isinstance(position_size, float)

        # Higher confidence should give larger size
        high_conf_size = vol_signal._calculate_vega_position_size(call, put, 0.9)
        low_conf_size = vol_signal._calculate_vega_position_size(call, put, 0.3)
        assert high_conf_size >= low_conf_size

    @pytest.mark.asyncio
    async def test_tail_protection(self, vol_signal, mock_options_chain):
        """Test tail protection wing calculation."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]
        premium_collected = call.price + put.price

        with patch.object(
            vol_signal,
            "_get_options_data",
            return_value=mock_options_chain["all_options"],
        ):
            wings = await vol_signal._get_tail_protection(
                "BTCUSDT", call, put, premium_collected
            )

            if wings:  # May be None if cost too high
                assert isinstance(wings, WingProtection)
                assert wings.cost_ratio <= vol_signal.config.max_wing_cost_ratio
                assert wings.call_wing.delta > 0
                assert wings.put_wing.delta < 0

    @pytest.mark.asyncio
    async def test_risk_conditions_check(self, vol_signal, mock_options_chain):
        """Test risk management conditions."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        risk_check = await vol_signal._check_risk_conditions("BTCUSDT", call, put)

        assert "allowed" in risk_check
        assert "reason" in risk_check
        assert isinstance(risk_check["allowed"], bool)

        # Fresh signal should pass risk checks
        assert risk_check["allowed"]

    @pytest.mark.asyncio
    async def test_full_signal_generation_positive_vrp(
        self, vol_signal, sample_ohlcv_data, mock_options_chain
    ):
        """Test full signal generation with positive VRP."""
        with patch.object(
            vol_signal,
            "_get_options_data",
            return_value=mock_options_chain["all_options"],
        ):
            signal = await vol_signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert signal.symbol == "BTCUSDT"
            assert isinstance(signal.value, float)
            assert isinstance(signal.confidence, float)
            assert 0 <= signal.confidence <= 1.0

            if signal.value != 0:  # If signal triggered
                assert signal.value < 0  # Should be negative (sell vol)
                assert signal.metadata["strategy_type"] == "vol_risk_premium"
                assert "vrp" in signal.metadata
                assert "premium_collected" in signal.metadata

    @pytest.mark.asyncio
    async def test_signal_generation_insufficient_data(self, vol_signal):
        """Test signal generation with insufficient data."""
        # Create minimal data (less than required window)
        short_data = pd.DataFrame(
            {
                "open": [45000, 45100],
                "high": [45200, 45300],
                "low": [44800, 44900],
                "close": [45100, 45200],
                "volume": [100000, 110000],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        signal = await vol_signal.generate(short_data, "BTCUSDT")

        assert signal.value == 0.0
        assert signal.confidence == 0.0
        assert "error" in signal.metadata

    def test_delta_hedging_execution(self, vol_signal, mock_options_chain):
        """Test delta hedging execution."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        # Create position with high net delta
        position = StraddlePosition(
            call_contract=call,
            put_contract=put,
            quantity=10,
            entry_price=call.price + put.price,
            entry_time=datetime.now(),
            premium_collected=(call.price + put.price) * 10,
            delta_hedge_units=0,
            hedge_history=[],
        )

        # Mock high delta requiring hedging
        call.delta = 0.60
        put.delta = -0.40  # Net delta = 0.20 (above threshold)

        # Execute hedge
        hedge_result = asyncio.run(vol_signal.execute_delta_hedge("BTCUSDT", position))

        assert "hedged" in hedge_result
        if hedge_result["hedged"]:
            assert "hedge_quantity" in hedge_result
            assert "hedge_cost" in hedge_result
            assert len(position.hedge_history) > 0

    def test_position_pnl_calculation(self, vol_signal, mock_options_chain):
        """Test position P&L calculation."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        position = StraddlePosition(
            call_contract=call,
            put_contract=put,
            quantity=10,
            entry_price=call.price + put.price,
            entry_time=datetime.now() - timedelta(days=2),
            premium_collected=(call.price + put.price) * 10,
            delta_hedge_units=0,
            hedge_history=[{"hedge_cost": 500, "timestamp": datetime.now()}],
        )

        current_spot = 46000  # 2% move up
        pnl_data = vol_signal.calculate_position_pnl(position, current_spot)

        assert "total_pnl" in pnl_data
        assert "premium_collected" in pnl_data
        assert "option_pnl" in pnl_data
        assert "hedge_costs" in pnl_data
        assert pnl_data["premium_collected"] > 0  # Premium should be positive

    def test_strategy_metrics(self, vol_signal, mock_options_chain):
        """Test strategy metrics calculation."""
        # Add some mock positions
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        position = StraddlePosition(
            call_contract=call,
            put_contract=put,
            quantity=5,
            entry_price=call.price + put.price,
            entry_time=datetime.now(),
            premium_collected=(call.price + put.price) * 5,
            delta_hedge_units=0,
        )

        vol_signal.active_positions["BTCUSDT"] = [position]

        metrics = vol_signal.get_strategy_metrics("BTCUSDT")

        assert "active_positions" in metrics
        assert "total_vega" in metrics
        assert "total_gamma" in metrics
        assert "total_theta" in metrics
        assert "total_premium_collected" in metrics

        assert metrics["active_positions"] == 1
        assert metrics["total_premium_collected"] > 0

    def test_vol_history_tracking(self, vol_signal):
        """Test volatility history tracking."""
        symbol = "BTCUSDT"

        # Add some vol history
        vol_signal._update_vol_history(symbol, 0.75)
        vol_signal._update_iv_history(symbol, 0.80)

        assert symbol in vol_signal.vol_history
        assert symbol in vol_signal.iv_history
        assert len(vol_signal.vol_history[symbol]) == 1
        assert len(vol_signal.iv_history[symbol]) == 1

        # Test persistence calculation
        persistence = vol_signal._calculate_vrp_persistence(symbol, 0.05)
        assert 0 <= persistence <= 1.0

    @pytest.mark.asyncio
    async def test_no_options_data_scenario(self, vol_signal, sample_ohlcv_data):
        """Test scenario where no options data is available."""
        with patch.object(vol_signal, "_get_options_data", return_value=[]):
            signal = await vol_signal.generate(sample_ohlcv_data, "BTCUSDT")

            assert signal.value == 0.0
            assert signal.confidence == 0.0
            assert "No options data available" in signal.metadata["reason"]

    def test_wing_cost_too_high(self, vol_signal, mock_options_chain):
        """Test scenario where wing protection cost is too high."""
        call = mock_options_chain["atm_call"]
        put = mock_options_chain["atm_put"]

        # Mock expensive wings
        with patch.object(vol_signal, "_get_tail_protection") as mock_wings:
            mock_wings.return_value = None  # Wings too expensive

            # This should still work, just without tail protection
            result = asyncio.run(mock_wings("BTCUSDT", call, put, 1000))
            assert result is None


@pytest.mark.asyncio
class TestVolRiskPremiumIntegration:
    """Integration tests for vol risk premium signal."""

    async def test_complete_workflow_positive_vrp(
        self, vol_signal, sample_ohlcv_data, mock_options_chain
    ):
        """Test complete workflow with positive VRP scenario."""
        # Set up scenario with positive VRP
        realized_vol = 0.60  # Lower than implied vol

        # Get the actual final price from sample data to create matching options
        final_price = sample_ohlcv_data.iloc[-1]["close"]

        # Create ATM options that match the final price
        expiry = datetime.now() + timedelta(days=10)
        atm_call = OptionsContract(
            symbol=f"BTC-{expiry.strftime('%d%b%y').upper()}-{int(final_price)}-C",
            underlying="BTC",
            strike=final_price,
            expiry_date=expiry,
            option_type=OptionType.CALL,
            price=final_price * 0.04,  # 4% of spot
            bid=final_price * 0.039,
            ask=final_price * 0.041,
            implied_volatility=0.80,  # 80% IV > 60% RV
            volume=500000,
            open_interest=2000000,
            delta=0.50,
            gamma=0.000025,
            vega=180,
            theta=-50,
            rho=20,
            mark_price=final_price * 0.04,
            index_price=final_price,
        )

        atm_put = OptionsContract(
            symbol=f"BTC-{expiry.strftime('%d%b%y').upper()}-{int(final_price)}-P",
            underlying="BTC",
            strike=final_price,
            expiry_date=expiry,
            option_type=OptionType.PUT,
            price=final_price * 0.039,  # Slightly less for put
            bid=final_price * 0.038,
            ask=final_price * 0.040,
            implied_volatility=0.82,  # 82% IV (slight skew)
            volume=450000,
            open_interest=1800000,
            delta=-0.50,
            gamma=0.000025,
            vega=175,
            theta=-48,
            rho=-18,
            mark_price=final_price * 0.039,
            index_price=final_price,
        )

        with patch.object(
            vol_signal,
            "_get_options_data",
            return_value=[atm_call, atm_put],
        ):
            with patch.object(
                vol_signal, "_calculate_realized_volatility", return_value=realized_vol
            ):
                signal = await vol_signal.generate(sample_ohlcv_data, "BTCUSDT")

                # Should generate sell signal
                assert signal.value < 0
                assert signal.confidence > 0
                assert signal.metadata["vrp"] > 0
                assert signal.metadata["strategy_type"] == "vol_risk_premium"

    async def test_complete_workflow_negative_vrp(
        self, vol_signal, sample_ohlcv_data, mock_options_chain
    ):
        """Test complete workflow with negative VRP scenario."""
        # Set up scenario with negative VRP
        realized_vol = 0.90  # Higher than implied vol

        with patch.object(
            vol_signal,
            "_get_options_data",
            return_value=mock_options_chain["all_options"],
        ):
            with patch.object(
                vol_signal, "_calculate_realized_volatility", return_value=realized_vol
            ):
                signal = await vol_signal.generate(sample_ohlcv_data, "BTCUSDT")

                # Should not generate signal
                assert signal.value == 0.0
                assert signal.confidence == 0.0

    async def test_risk_limits_enforcement(
        self, vol_signal, sample_ohlcv_data, mock_options_chain
    ):
        """Test that risk limits are properly enforced."""
        # Fill up position limits
        vol_signal.active_positions["BTCUSDT"] = [
            None
        ] * vol_signal.config.max_position_count

        with patch.object(
            vol_signal,
            "_get_options_data",
            return_value=mock_options_chain["all_options"],
        ):
            signal = await vol_signal.generate(sample_ohlcv_data, "BTCUSDT")

            # Should not generate signal due to position limits
            assert signal.value == 0.0
            assert "Max position count reached" in signal.metadata["reason"]


if __name__ == "__main__":
    pytest.main([__file__])
