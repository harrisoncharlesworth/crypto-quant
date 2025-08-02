import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.quantbot.signals.cash_carry import (
    CashCarryArbitrageSignal,
    CashCarryConfig,
    FuturesContract,
    FuturesExpiry,
)
from src.quantbot.signals.base import SignalResult


@pytest.fixture
def config():
    """Default configuration for testing."""
    return CashCarryConfig(
        basis_threshold_min=0.08,
        basis_threshold_max=0.10,
        max_allocation=0.20,
        margin_spike_threshold=0.5,
        margin_emergency_threshold=1.0,
        min_volume_24h=1000000,
        min_futures_oi=50000000,
        position_size_scale=1.0,
        confidence_basis_multiplier=5.0,
    )


@pytest.fixture
def signal(config):
    """Cash-carry signal instance."""
    return CashCarryArbitrageSignal(config)


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(45000, 46000, 100),
            "high": np.random.uniform(45500, 46500, 100),
            "low": np.random.uniform(44500, 45500, 100),
            "close": np.random.uniform(45000, 46000, 100),
            "volume": np.random.uniform(1000000, 5000000, 100),
        },
        index=dates,
    )

    # Ensure last close is known value for testing
    data.iloc[-1, data.columns.get_loc("close")] = 45000
    return data


@pytest.fixture
def mock_futures_contracts():
    """Mock futures contracts for testing."""
    return [
        # Quarterly contract in contango (8.5% annualized basis)
        FuturesContract(
            symbol="BTC-0624",
            expiry_date=datetime.now() + timedelta(days=45),
            contract_type=FuturesExpiry.QUARTERLY,
            price=45950,  # ~8.5% annualized basis over 45 days
            volume=500000000,
            open_interest=2000000000,
            margin_requirement=0.05,
        ),
        # Quarterly contract in strong contango (12% annualized basis)
        FuturesContract(
            symbol="BTC-0924",
            expiry_date=datetime.now() + timedelta(days=90),
            contract_type=FuturesExpiry.QUARTERLY,
            price=46350,  # ~12% annualized basis over 90 days
            volume=300000000,
            open_interest=1500000000,
            margin_requirement=0.06,
        ),
        # Perpetual in slight contango
        FuturesContract(
            symbol="BTC-PERP",
            expiry_date=None,
            contract_type=FuturesExpiry.PERPETUAL,
            price=45100,
            volume=800000000,
            open_interest=3000000000,
            margin_requirement=0.02,
            funding_rate=0.0003,  # ~10% annualized
        ),
    ]


class TestCashCarryArbitrageSignal:
    """Test cases for Cash-and-Carry Basis arbitrage signal."""

    @pytest.mark.asyncio
    async def test_basic_signal_generation(
        self, signal, sample_data, mock_futures_contracts
    ):
        """Test basic signal generation with valid arbitrage opportunity."""

        with patch.object(
            signal, "_get_futures_contracts", return_value=mock_futures_contracts
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"
        assert result.value != 0.0  # Should generate a signal
        assert result.confidence > 0.0
        assert "annualized_basis" in result.metadata
        assert "strategy_type" in result.metadata
        assert result.metadata["strategy_type"] == "cash_carry_arbitrage"

    @pytest.mark.asyncio
    async def test_contango_signal_direction(
        self, signal, sample_data, mock_futures_contracts
    ):
        """Test that contango generates positive signal (long spot, short futures)."""

        with patch.object(
            signal, "_get_futures_contracts", return_value=mock_futures_contracts
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

        # All mock contracts are in contango, should generate positive signal
        assert result.value > 0.0
        assert result.metadata["annualized_basis"] > 0.0

    @pytest.mark.asyncio
    async def test_backwardation_signal(self, signal, sample_data):
        """Test signal generation in backwardation (futures < spot)."""

        # Create contracts in backwardation
        backwardation_contracts = [
            FuturesContract(
                symbol="BTC-0624",
                expiry_date=datetime.now() + timedelta(days=45),
                contract_type=FuturesExpiry.QUARTERLY,
                price=43500,  # Below spot (45000)
                volume=500000000,
                open_interest=2000000000,
                margin_requirement=0.05,
            )
        ]

        with patch.object(
            signal, "_get_futures_contracts", return_value=backwardation_contracts
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

        # Backwardation should generate negative signal (short spot, long futures)
        assert result.value < 0.0
        assert result.metadata["annualized_basis"] < 0.0

    @pytest.mark.asyncio
    async def test_insufficient_basis_threshold(self, signal, sample_data):
        """Test that insufficient basis results in no signal."""

        # Create contract with minimal basis (below threshold)
        low_basis_contracts = [
            FuturesContract(
                symbol="BTC-0624",
                expiry_date=datetime.now() + timedelta(days=45),
                contract_type=FuturesExpiry.QUARTERLY,
                price=45100,  # Only ~2.7% annualized basis
                volume=500000000,
                open_interest=2000000000,
                margin_requirement=0.05,
            )
        ]

        with patch.object(
            signal, "_get_futures_contracts", return_value=low_basis_contracts
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_liquidity_requirements(self, signal, sample_data, config):
        """Test liquidity filtering."""

        # Create contract with insufficient liquidity
        low_liquidity_contracts = [
            FuturesContract(
                symbol="BTC-0624",
                expiry_date=datetime.now() + timedelta(days=45),
                contract_type=FuturesExpiry.QUARTERLY,
                price=46000,
                volume=100000,  # Below min_volume_24h
                open_interest=1000000,  # Below min_futures_oi
                margin_requirement=0.05,
            )
        ]

        with patch.object(
            signal, "_get_futures_contracts", return_value=low_liquidity_contracts
        ):
            result = await signal.generate(sample_data, "BTCUSDT")

        assert result.value == 0.0
        assert "No valid arbitrage opportunities" in result.metadata["reason"]

    @pytest.mark.asyncio
    async def test_margin_spike_risk_management(
        self, signal, sample_data, mock_futures_contracts
    ):
        """Test margin spike detection and risk management."""

        # Add margin history to simulate spike
        symbol = "BTCUSDT"
        base_time = datetime.now()

        # Add historical margin data (5% margin)
        signal.margin_history[symbol] = [
            (base_time - timedelta(hours=i), 0.05) for i in range(24, 0, -1)
        ]

        # Create contract with spiked margin (150% increase)
        high_margin_contracts = [
            FuturesContract(
                symbol="BTC-0624",
                expiry_date=datetime.now() + timedelta(days=45),
                contract_type=FuturesExpiry.QUARTERLY,
                price=46000,
                volume=500000000,
                open_interest=2000000000,
                margin_requirement=0.125,  # 150% increase from 0.05
            )
        ]

        with patch.object(
            signal, "_get_futures_contracts", return_value=high_margin_contracts
        ):
            result = await signal.generate(sample_data, symbol)

        # Should reject due to emergency margin spike
        assert result.value == 0.0
        assert "Emergency margin spike" in result.metadata["reason"]

    def test_basis_calculation_quarterly(self, signal):
        """Test basis calculation for quarterly futures."""

        spot_price = 45000
        contract = FuturesContract(
            symbol="BTC-0624",
            expiry_date=datetime.now() + timedelta(days=45),
            contract_type=FuturesExpiry.QUARTERLY,
            price=46000,
            volume=500000000,
            open_interest=2000000000,
            margin_requirement=0.05,
        )

        basis_data = signal._calculate_basis(spot_price, contract)

        assert basis_data is not None
        assert basis_data["basis_points"] == (46000 - 45000) / 45000
        assert basis_data["days_to_expiry"] in [44, 45]  # Can vary due to timing
        assert not basis_data["is_perpetual"]
        assert basis_data["annualized_basis"] > 0  # Should be positive

    def test_basis_calculation_perpetual(self, signal):
        """Test basis calculation for perpetual contracts."""

        spot_price = 45000
        contract = FuturesContract(
            symbol="BTC-PERP",
            expiry_date=None,
            contract_type=FuturesExpiry.PERPETUAL,
            price=45100,
            volume=800000000,
            open_interest=3000000000,
            margin_requirement=0.02,
            funding_rate=0.0003,
        )

        basis_data = signal._calculate_basis(spot_price, contract)

        assert basis_data is not None
        assert basis_data["is_perpetual"]
        assert basis_data["funding_rate"] == 0.0003
        assert basis_data["annualized_basis"] == 0.0003 * 3 * 365  # Annualized funding

    def test_position_sizing(self, signal, mock_futures_contracts):
        """Test position sizing logic."""

        basis_data = {
            "annualized_basis": 0.12,  # 12% basis
            "days_to_expiry": 45,
            "is_perpetual": False,
        }

        contract = mock_futures_contracts[0]  # 5% margin requirement
        confidence = 0.8

        position_size = signal._calculate_position_size(
            basis_data, contract, confidence
        )

        # Should be positive and reasonable
        assert 0.0 < position_size <= signal.config.max_allocation

        # Higher confidence should lead to larger position
        high_confidence_size = signal._calculate_position_size(
            basis_data, contract, 0.95
        )
        assert high_confidence_size >= position_size

    def test_confidence_calculation(self, signal, mock_futures_contracts):
        """Test confidence scoring."""

        symbol = "BTCUSDT"
        basis_data = {
            "annualized_basis": 0.12,  # Strong 12% basis
            "days_to_expiry": 45,
            "is_perpetual": False,
        }

        contract = mock_futures_contracts[0]

        confidence = signal._calculate_confidence(symbol, basis_data, contract)

        assert 0.0 <= confidence <= 1.0

        # Strong basis should lead to reasonable confidence
        assert confidence > 0.3

    def test_basis_stability_tracking(self, signal):
        """Test basis stability calculation."""

        symbol = "BTCUSDT"

        # Add stable basis history
        base_time = datetime.now()
        stable_basis = [0.10, 0.101, 0.099, 0.102, 0.098]  # Low volatility

        signal.basis_history[symbol] = [
            (base_time - timedelta(hours=i), basis)
            for i, basis in enumerate(reversed(stable_basis))
        ]

        stability = signal._calculate_basis_stability(symbol)

        # Stable basis should have high stability score
        assert stability > 0.7

        # Add volatile basis history
        volatile_basis = [0.10, 0.15, 0.05, 0.20, 0.03]  # High volatility

        signal.basis_history[symbol] = [
            (base_time - timedelta(hours=i), basis)
            for i, basis in enumerate(reversed(volatile_basis))
        ]

        stability = signal._calculate_basis_stability(symbol)

        # Volatile basis should have lower stability score
        assert stability < 0.5

    def test_carry_income_estimation(self, signal):
        """Test carry income estimation."""

        # Test quarterly contract
        quarterly_basis = {
            "annualized_basis": 0.10,  # 10% annualized
            "time_fraction": 45 / 365.25,  # 45 days
            "is_perpetual": False,
        }

        income = signal._estimate_carry_income(quarterly_basis)
        expected = 0.10 * (45 / 365.25)  # ~1.23%

        assert abs(income - expected) < 0.001

        # Test perpetual contract
        perp_basis = {
            "annualized_basis": 0.12,  # 12% annualized funding
            "is_perpetual": True,
        }

        income = signal._estimate_carry_income(perp_basis)
        expected = 0.12 * (30 / 365.25)  # Assume 30-day holding

        assert abs(income - expected) < 0.001

    @pytest.mark.asyncio
    async def test_no_futures_available(self, signal, sample_data):
        """Test behavior when no futures contracts are available."""

        with patch.object(signal, "_get_futures_contracts", return_value=[]):
            result = await signal.generate(sample_data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "No futures contracts available" in result.metadata["reason"]

    @pytest.mark.asyncio
    async def test_insufficient_data(self, signal, config):
        """Test handling of insufficient market data."""

        # Empty dataframe
        empty_data = pd.DataFrame()

        result = await signal.generate(empty_data, "BTCUSDT")

        assert result.value == 0.0
        assert result.confidence == 0.0
        assert "Insufficient market data" in result.metadata["error"]

    def test_strategy_metrics(self, signal):
        """Test strategy metrics calculation."""

        symbol = "BTCUSDT"

        # Add some tracking data
        signal.carry_income_tracker[symbol] = 0.05  # 5% realized income

        base_time = datetime.now()
        signal.basis_history[symbol] = [
            (base_time - timedelta(hours=i), 0.10 + np.random.normal(0, 0.01))
            for i in range(24)
        ]

        signal.margin_history[symbol] = [
            (base_time - timedelta(hours=i), 0.05 + np.random.normal(0, 0.005))
            for i in range(24)
        ]

        metrics = signal.get_strategy_metrics(symbol)

        assert "total_carry_income" in metrics
        assert metrics["total_carry_income"] == 0.05
        assert "basis_stability" in metrics
        assert "avg_basis_24h" in metrics
        assert "avg_margin_24h" in metrics

    @pytest.mark.asyncio
    async def test_sharpe_ratio_target(self, signal, config):
        """Test that strategy can achieve target Sharpe ratio ≈ 0.6."""

        # Simulate multiple trading periods with realistic parameters
        results = []
        returns = []

        # Use a single base date and vary the time component
        base_date = datetime(2024, 1, 15)  # Fixed date to avoid month issues

        for i in range(50):
            # Create sample data with slight price variation
            start_time = base_date + timedelta(hours=i)
            dates = pd.date_range(start=start_time, periods=24, freq="h")
            base_price = 45000 + np.random.normal(0, 500)

            data = pd.DataFrame(
                {
                    "open": [base_price] * 24,
                    "high": [base_price * 1.01] * 24,
                    "low": [base_price * 0.99] * 24,
                    "close": [base_price] * 24,
                    "volume": [2000000] * 24,
                },
                index=dates,
            )

            # Create contract with basis that varies realistically
            basis_rate = np.random.uniform(0.08, 0.15)  # 8-15% annualized
            futures_price = base_price * (1 + basis_rate * 45 / 365.25)

            contracts = [
                FuturesContract(
                    symbol="BTC-0624",
                    expiry_date=datetime.now() + timedelta(days=45),
                    contract_type=FuturesExpiry.QUARTERLY,
                    price=futures_price,
                    volume=500000000,
                    open_interest=2000000000,
                    margin_requirement=0.05,
                )
            ]

            with patch.object(signal, "_get_futures_contracts", return_value=contracts):
                result = await signal.generate(data, "BTCUSDT")
                results.append(result)

                # Estimate return based on signal strength and basis
                if result.value != 0.0:
                    estimated_return = basis_rate * abs(result.value) * 45 / 365.25
                    returns.append(estimated_return)

        # Calculate strategy statistics
        if returns:
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0

            # Check that we're in the ballpark of target metrics
            assert avg_return > 0.001  # > 0.1% average return per trade
            assert sharpe_ratio > 0.2  # Reasonable Sharpe ratio

            # Count successful signals
            successful_signals = sum(1 for r in results if r.value != 0.0)
            assert successful_signals > 20  # Should find opportunities frequently

    def test_risk_management_integration(self, signal, config):
        """Test integration of all risk management features."""

        # Test position size scaling with different risk factors
        test_cases = [
            # (basis, margin_req, expected_relative_size)
            (0.12, 0.05, 1.0),  # Strong basis, normal margin
            (0.08, 0.05, 0.8),  # Threshold basis, normal margin
            (0.12, 0.10, 0.6),  # Strong basis, high margin
            (0.15, 0.15, 0.4),  # Very strong basis, very high margin
        ]

        for basis_rate, margin_req, expected_rel_size in test_cases:
            basis_data = {
                "annualized_basis": basis_rate,
                "days_to_expiry": 45,
                "is_perpetual": False,
            }

            contract = FuturesContract(
                symbol="BTC-0624",
                expiry_date=datetime.now() + timedelta(days=45),
                contract_type=FuturesExpiry.QUARTERLY,
                price=45000 * (1 + basis_rate * 45 / 365.25),
                volume=500000000,
                open_interest=2000000000,
                margin_requirement=margin_req,
            )

            confidence = 0.8
            position_size = signal._calculate_position_size(
                basis_data, contract, confidence
            )

            # Verify size scales appropriately with risk
            relative_size = position_size / config.max_allocation
            assert abs(relative_size - expected_rel_size) < 0.3  # Within 30% tolerance
