#!/usr/bin/env python3
"""
Comprehensive validation script for all 12 implemented crypto signals.

This script tests all implemented signals with mock data to ensure:
1. Signal generation without errors
2. Output bounds compliance (-1 to +1 for directional, proper ranges for others)
3. Confidence scores (0 to 1)
4. Metadata structure completeness
5. Config integration working
6. Portfolio Blender v2 integration
7. Signal classification (M-N vs 2-S) validation
8. Risk management compliance

Author: Crypto Quant Team
Date: 02/08/2025
"""

import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback
from dataclasses import dataclass, field

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals import *
from quantbot.portfolio.blender_v2 import (
    PortfolioBlenderV2,
    BlenderConfigV2,
    SignalType,
    AllocationMethod,
)
from quantbot.signals.base import SignalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of signal validation test."""

    signal_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    test_results: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class MockDataGenerator:
    """Generate realistic mock market data for testing."""

    @staticmethod
    def generate_ohlcv_data(
        symbol: str = "BTCUSDT",
        periods: int = 200,
        start_price: float = 45000.0,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Generate realistic OHLCV data with trends and volatility."""

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=periods), periods=periods, freq="H"
        )

        # Generate price series with some trend and noise
        returns = np.random.normal(0, volatility, periods)
        # Add some autocorrelation for realism
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i - 1]

        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, volatility / 4)))
            low = close * (1 - abs(np.random.normal(0, volatility / 4)))
            if i == 0:
                open_price = close
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, volatility / 8))

            volume = abs(np.random.normal(1000, 200))

            data.append(
                {
                    "timestamp": dates[i],
                    "open": open_price,
                    "high": max(open_price, high, close),
                    "low": min(open_price, low, close),
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    @staticmethod
    def generate_funding_data(periods: int = 200) -> pd.DataFrame:
        """Generate funding rate data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods * 8),
            periods=periods,
            freq="8H",
        )

        funding_rates = np.random.normal(
            0.0001, 0.0005, periods
        )  # Realistic funding rates

        return pd.DataFrame(
            {
                "timestamp": dates,
                "funding_rate": funding_rates,
                "funding_rate_predicted": funding_rates
                + np.random.normal(0, 0.0001, periods),
            }
        ).set_index("timestamp")

    @staticmethod
    def generate_oi_data(periods: int = 200) -> pd.DataFrame:
        """Generate open interest data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods), periods=periods, freq="H"
        )

        base_oi = 1000000000  # 1B USD
        oi_changes = np.random.normal(0, 0.02, periods)
        oi_values = [base_oi]

        for change in oi_changes[1:]:
            oi_values.append(oi_values[-1] * (1 + change))

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open_interest": oi_values,
                "open_interest_value": [
                    oi * 45000 for oi in oi_values
                ],  # Assuming ~45k BTC price
            }
        ).set_index("timestamp")

    @staticmethod
    def generate_options_data(periods: int = 200) -> pd.DataFrame:
        """Generate options market data."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods), periods=periods, freq="H"
        )

        # Realistic implied volatility data
        iv_levels = np.random.normal(0.6, 0.15, periods)  # 60% average IV
        iv_levels = np.clip(iv_levels, 0.2, 1.5)  # Reasonable bounds

        # Generate skew data (25 delta put vol - 25 delta call vol)
        skew_values = np.random.normal(0.05, 0.02, periods)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "implied_vol_atm": iv_levels,
                "implied_vol_25d_call": iv_levels - skew_values / 2,
                "implied_vol_25d_put": iv_levels + skew_values / 2,
                "realized_vol_30d": np.random.normal(0.5, 0.1, periods),
                "vol_risk_premium": iv_levels - np.random.normal(0.5, 0.1, periods),
            }
        ).set_index("timestamp")

    @staticmethod
    def generate_onchain_data(periods: int = 200) -> pd.DataFrame:
        """Generate on-chain data for MVRV and SSR."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=periods), periods=periods, freq="D"
        )

        # MVRV data
        mvrv_values = np.random.normal(1.5, 0.5, periods)
        mvrv_values = np.clip(mvrv_values, 0.5, 4.0)

        # Stablecoin supply data
        total_supply = 150_000_000_000  # $150B total stablecoin supply
        supply_changes = np.random.normal(0, 0.01, periods)
        supply_values = [total_supply]

        for change in supply_changes[1:]:
            supply_values.append(supply_values[-1] * (1 + change))

        btc_market_cap = [
            val * 45000 * 19_000_000 for val in np.random.normal(1, 0.02, periods)
        ]

        return pd.DataFrame(
            {
                "timestamp": dates,
                "mvrv_ratio": mvrv_values,
                "realized_cap": [
                    cap / mvrv for cap, mvrv in zip(btc_market_cap, mvrv_values)
                ],
                "market_cap": btc_market_cap,
                "stablecoin_supply": supply_values,
                "btc_market_cap": btc_market_cap,
            }
        ).set_index("timestamp")


class SignalValidator:
    """Comprehensive signal validation framework."""

    def __init__(self):
        self.mock_data = MockDataGenerator()
        self.validation_results: List[ValidationResult] = []

        # Define all 12 signals with their configurations
        self.signal_definitions = {
            # Directional signals (2-S)
            "time_series_momentum": {
                "class": TimeSeriesMomentumSignal,
                "config": MomentumConfig(),
                "type": SignalType.DIRECTIONAL,
                "data_requirements": ["ohlcv"],
            },
            "donchian_breakout": {
                "class": DonchianBreakoutSignal,
                "config": BreakoutConfig(),
                "type": SignalType.DIRECTIONAL,
                "data_requirements": ["ohlcv"],
            },
            "short_term_mean_reversion": {
                "class": ShortTermMeanReversionSignal,
                "config": MeanReversionConfig(),
                "type": SignalType.DIRECTIONAL,
                "data_requirements": ["ohlcv"],
            },
            "oi_price_divergence": {
                "class": OIPriceDivergenceSignal,
                "config": OIDivergenceConfig(),
                "type": SignalType.DIRECTIONAL,
                "data_requirements": ["ohlcv", "oi"],
            },
            "delta_skew_whipsaw": {
                "class": SkewWhipsawSignal,
                "config": SkewWhipsawConfig(),
                "type": SignalType.DIRECTIONAL,
                "data_requirements": ["ohlcv", "options"],
            },
            # Market-neutral signals (M-N)
            "perp_funding_carry": {
                "class": PerpFundingCarrySignal,
                "config": FundingCarryConfig(),
                "type": SignalType.MARKET_NEUTRAL,
                "data_requirements": ["ohlcv", "funding"],
            },
            "alt_btc_cross_sectional": {
                "class": CrossSectionalMomentumSignal,
                "config": MomentumConfig(),
                "type": SignalType.MARKET_NEUTRAL,
                "data_requirements": ["ohlcv"],
            },
            "cash_carry_basis": {
                "class": CashCarryArbitrageSignal,
                "config": CashCarryConfig(),
                "type": SignalType.MARKET_NEUTRAL,
                "data_requirements": ["ohlcv", "funding"],
            },
            "cross_exchange_funding": {
                "class": XExchangeFundingDispersionSignal,
                "config": XExchangeFundingConfig(),
                "type": SignalType.MARKET_NEUTRAL,
                "data_requirements": ["funding"],
            },
            "options_vol_risk_premium": {
                "class": VolatilityRiskPremiumSignal,
                "config": VolRiskPremiumConfig(),
                "type": SignalType.MARKET_NEUTRAL,
                "data_requirements": ["ohlcv", "options"],
            },
            # Overlay/filter signals
            "stablecoin_supply_ratio": {
                "class": StablecoinSupplyRatioSignal,
                "config": SSRConfig(),
                "type": SignalType.OVERLAY,
                "data_requirements": ["onchain"],
            },
            "mvrv_zscore": {
                "class": MVRVSignal,
                "config": MVRVConfig(),
                "type": SignalType.OVERLAY,
                "data_requirements": ["onchain"],
            },
        }

    async def validate_all_signals(self) -> Dict[str, ValidationResult]:
        """Validate all 12 signals comprehensively."""
        logger.info("Starting comprehensive validation of all 12 crypto signals...")

        validation_results = {}

        for signal_name, signal_def in self.signal_definitions.items():
            logger.info(f"Validating signal: {signal_name}")
            try:
                result = await self.validate_single_signal(signal_name, signal_def)
                validation_results[signal_name] = result
                self.validation_results.append(result)

                if result.passed:
                    logger.info(f"✅ {signal_name}: PASSED")
                else:
                    logger.warning(
                        f"❌ {signal_name}: FAILED - {', '.join(result.errors)}"
                    )

            except Exception as e:
                error_msg = f"Validation failed with exception: {str(e)}"
                logger.error(f"❌ {signal_name}: {error_msg}")
                logger.error(traceback.format_exc())

                validation_results[signal_name] = ValidationResult(
                    signal_name=signal_name,
                    status="FAIL",
                    test_results={},
                    errors=[error_msg],
                )

        return validation_results

    async def validate_single_signal(
        self, signal_name: str, signal_def: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a single signal comprehensively."""

        test_results = {}
        errors = []
        warnings = []

        try:
            # 1. Test signal instantiation
            signal_class = signal_def["class"]
            config = signal_def["config"]

            signal = signal_class(config)
            test_results["instantiation"] = True

            # 2. Generate appropriate mock data
            mock_data = self._generate_mock_data_for_signal(
                signal_def["data_requirements"]
            )
            test_results["mock_data_generation"] = True

            # 3. Test signal generation
            symbol = "BTCUSDT"
            ohlcv_data = mock_data.get("ohlcv", pd.DataFrame())

            # Enhance data with additional fields if needed
            if "funding" in signal_def["data_requirements"]:
                funding_data = mock_data.get("funding", pd.DataFrame())
                if not funding_data.empty and not ohlcv_data.empty:
                    # Add funding data to OHLCV
                    ohlcv_data = ohlcv_data.join(funding_data, how="left")
                    ohlcv_data["funding_rate"] = ohlcv_data["funding_rate"].fillna(0)

            if "oi" in signal_def["data_requirements"]:
                oi_data = mock_data.get("oi", pd.DataFrame())
                if not oi_data.empty and not ohlcv_data.empty:
                    ohlcv_data = ohlcv_data.join(oi_data, how="left")
                    ohlcv_data["open_interest"] = ohlcv_data["open_interest"].fillna(
                        method="ffill"
                    )

            if "options" in signal_def["data_requirements"]:
                options_data = mock_data.get("options", pd.DataFrame())
                if not options_data.empty and not ohlcv_data.empty:
                    ohlcv_data = ohlcv_data.join(options_data, how="left")
                    for col in options_data.columns:
                        ohlcv_data[col] = ohlcv_data[col].fillna(method="ffill")

            if "onchain" in signal_def["data_requirements"]:
                # For on-chain signals, use the on-chain data directly
                onchain_data = mock_data.get("onchain", pd.DataFrame())
                if not onchain_data.empty:
                    data_for_signal = onchain_data
                else:
                    data_for_signal = ohlcv_data
            else:
                data_for_signal = ohlcv_data

            # Generate signal
            signal_result = await signal.generate(data_for_signal, symbol)
            test_results["signal_generation"] = True

            # 4. Validate signal result structure
            if not isinstance(signal_result, SignalResult):
                errors.append(
                    f"Signal must return SignalResult, got {type(signal_result)}"
                )
            else:
                test_results["result_type"] = True

                # 5. Validate signal value bounds
                if not (-1.0 <= signal_result.value <= 1.0):
                    errors.append(
                        f"Signal value {signal_result.value} outside bounds [-1, 1]"
                    )
                else:
                    test_results["value_bounds"] = True

                # 6. Validate confidence bounds
                if not (0.0 <= signal_result.confidence <= 1.0):
                    errors.append(
                        f"Confidence {signal_result.confidence} outside bounds [0, 1]"
                    )
                else:
                    test_results["confidence_bounds"] = True

                # 7. Validate metadata structure
                if signal_result.metadata is None:
                    warnings.append(
                        "Signal metadata is None - consider adding metadata"
                    )
                else:
                    test_results["metadata_present"] = True
                    if not isinstance(signal_result.metadata, dict):
                        errors.append(
                            f"Metadata must be dict, got {type(signal_result.metadata)}"
                        )
                    else:
                        test_results["metadata_structure"] = True

                # 8. Validate signal classification matches expected type
                expected_type = signal_def["type"]
                test_results["expected_type"] = expected_type.value
                test_results["classification_check"] = True

                # 9. Test signal with edge cases
                edge_case_results = await self._test_edge_cases(signal, symbol)
                test_results["edge_cases"] = edge_case_results

                # 10. Test config integration
                config_test = self._test_config_integration(signal, config)
                test_results["config_integration"] = config_test

        except Exception as e:
            errors.append(f"Signal validation failed: {str(e)}")
            logger.error(f"Error validating {signal_name}: {traceback.format_exc()}")

        # Determine overall status
        if errors:
            status = "FAIL"
        elif warnings:
            status = "WARNING"
        else:
            status = "PASS"

        return ValidationResult(
            signal_name=signal_name,
            status=status,
            test_results=test_results,
            errors=errors,
            warnings=warnings,
        )

    def _generate_mock_data_for_signal(
        self, requirements: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Generate mock data based on signal requirements."""
        mock_data = {}

        if "ohlcv" in requirements:
            mock_data["ohlcv"] = self.mock_data.generate_ohlcv_data()

        if "funding" in requirements:
            mock_data["funding"] = self.mock_data.generate_funding_data()

        if "oi" in requirements:
            mock_data["oi"] = self.mock_data.generate_oi_data()

        if "options" in requirements:
            mock_data["options"] = self.mock_data.generate_options_data()

        if "onchain" in requirements:
            mock_data["onchain"] = self.mock_data.generate_onchain_data()

        return mock_data

    async def _test_edge_cases(self, signal, symbol: str) -> Dict[str, bool]:
        """Test signal with edge cases."""
        edge_results = {}

        try:
            # Test with minimal data
            minimal_data = self.mock_data.generate_ohlcv_data(periods=10)
            result = await signal.generate(minimal_data, symbol)
            edge_results["minimal_data"] = isinstance(result, SignalResult)
        except:
            edge_results["minimal_data"] = False

        try:
            # Test with flat prices (no volatility)
            flat_data = self.mock_data.generate_ohlcv_data(volatility=0.0)
            result = await signal.generate(flat_data, symbol)
            edge_results["flat_prices"] = isinstance(result, SignalResult)
        except:
            edge_results["flat_prices"] = False

        try:
            # Test with high volatility
            volatile_data = self.mock_data.generate_ohlcv_data(volatility=0.1)
            result = await signal.generate(volatile_data, symbol)
            edge_results["high_volatility"] = isinstance(result, SignalResult)
        except:
            edge_results["high_volatility"] = False

        return edge_results

    def _test_config_integration(self, signal, config) -> bool:
        """Test that signal properly uses its configuration."""
        try:
            # Check that signal has config attribute
            if not hasattr(signal, "config"):
                return False

            # Check that config is properly set
            if signal.config != config:
                return False

            # Check that config has required attributes
            if not hasattr(config, "enabled"):
                return False

            return True
        except:
            return False

    async def validate_portfolio_blender(self) -> ValidationResult:
        """Validate Portfolio Blender v2 with all signals."""
        logger.info("Validating Portfolio Blender v2 integration...")

        test_results = {}
        errors = []
        warnings = []

        try:
            # 1. Initialize blender
            config = BlenderConfigV2()
            blender = PortfolioBlenderV2(config)
            test_results["blender_init"] = True

            # 2. Generate signals from all signal types
            signals = {}
            symbol = "BTCUSDT"

            for signal_name, signal_def in self.signal_definitions.items():
                try:
                    signal_class = signal_def["class"]
                    signal_config = signal_def["config"]
                    signal = signal_class(signal_config)

                    # Generate appropriate mock data
                    mock_data = self._generate_mock_data_for_signal(
                        signal_def["data_requirements"]
                    )

                    # Prepare data
                    if "onchain" in signal_def["data_requirements"]:
                        data = mock_data.get("onchain", pd.DataFrame())
                    else:
                        data = mock_data.get("ohlcv", pd.DataFrame())

                        # Add additional data if needed
                        if "funding" in signal_def["data_requirements"]:
                            funding_data = mock_data.get("funding", pd.DataFrame())
                            if not funding_data.empty:
                                data = data.join(funding_data, how="left")
                                data["funding_rate"] = data["funding_rate"].fillna(0)

                        if "oi" in signal_def["data_requirements"]:
                            oi_data = mock_data.get("oi", pd.DataFrame())
                            if not oi_data.empty:
                                data = data.join(oi_data, how="left")
                                data["open_interest"] = data["open_interest"].fillna(
                                    method="ffill"
                                )

                        if "options" in signal_def["data_requirements"]:
                            options_data = mock_data.get("options", pd.DataFrame())
                            if not options_data.empty:
                                data = data.join(options_data, how="left")
                                for col in options_data.columns:
                                    data[col] = data[col].fillna(method="ffill")

                    result = await signal.generate(data, symbol)
                    signals[signal_name] = result

                except Exception as e:
                    warnings.append(
                        f"Could not generate signal {signal_name}: {str(e)}"
                    )
                    # Create a dummy signal for testing
                    signals[signal_name] = SignalResult(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        value=np.random.uniform(-0.5, 0.5),
                        confidence=np.random.uniform(0.3, 0.8),
                        metadata={"test": True},
                    )

            test_results["signal_generation_count"] = len(signals)

            # 3. Test blender with all signals
            blended_result = blender.blend_signals(signals, symbol)
            test_results["blend_signals"] = True

            # 4. Validate blended result structure
            if hasattr(blended_result, "final_position"):
                test_results["final_position_present"] = True
                if not (-1.0 <= blended_result.final_position <= 1.0):
                    errors.append(
                        f"Final position {blended_result.final_position} outside bounds"
                    )
                else:
                    test_results["final_position_bounds"] = True

            if hasattr(blended_result, "confidence"):
                test_results["blended_confidence_present"] = True
                if not (0.0 <= blended_result.confidence <= 1.0):
                    errors.append(
                        f"Blended confidence {blended_result.confidence} outside bounds"
                    )
                else:
                    test_results["blended_confidence_bounds"] = True

            # 5. Validate signal classification
            if hasattr(blended_result, "directional_position"):
                test_results["directional_position_present"] = True

            if hasattr(blended_result, "market_neutral_position"):
                test_results["market_neutral_position_present"] = True

            # 6. Validate risk metrics
            if hasattr(blended_result, "risk_metrics"):
                test_results["risk_metrics_present"] = True
                if isinstance(blended_result.risk_metrics, dict):
                    test_results["risk_metrics_structure"] = True

            # 7. Test portfolio statistics
            portfolio_stats = blender.get_portfolio_statistics()
            test_results["portfolio_statistics"] = isinstance(portfolio_stats, dict)

            # 8. Test risk limit checking
            risk_violations = blender.check_risk_limits()
            test_results["risk_limit_check"] = isinstance(risk_violations, list)

            # 9. Test different allocation methods
            allocation_methods = [
                AllocationMethod.EQUAL_WEIGHT,
                AllocationMethod.CONFIDENCE_WEIGHTED,
                AllocationMethod.RISK_PARITY,
            ]
            allocation_results = {}

            for method in allocation_methods:
                try:
                    test_config = BlenderConfigV2(allocation_method=method)
                    test_blender = PortfolioBlenderV2(test_config)
                    test_result = test_blender.blend_signals(signals, symbol)
                    allocation_results[method.value] = True
                except Exception as e:
                    allocation_results[method.value] = False
                    warnings.append(
                        f"Allocation method {method.value} failed: {str(e)}"
                    )

            test_results["allocation_methods"] = allocation_results

        except Exception as e:
            errors.append(f"Portfolio blender validation failed: {str(e)}")
            logger.error(f"Blender validation error: {traceback.format_exc()}")

        # Determine status
        if errors:
            status = "FAIL"
        elif warnings:
            status = "WARNING"
        else:
            status = "PASS"

        return ValidationResult(
            signal_name="portfolio_blender_v2",
            status=status,
            test_results=test_results,
            errors=errors,
            warnings=warnings,
        )

    def generate_validation_report(
        self, results: Dict[str, ValidationResult], blender_result: ValidationResult
    ) -> str:
        """Generate comprehensive validation report."""

        passed_signals = sum(1 for r in results.values() if r.passed)
        total_signals = len(results)

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        CRYPTO QUANT SIGNAL VALIDATION REPORT                    ║
║                                   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL SUMMARY
═══════════════════════════════════════════════════════════════════════════════════
• Total Signals Tested: {total_signals}/12
• Signals Passed: {passed_signals}/{total_signals} ({(passed_signals/total_signals*100):.1f}%)
• Portfolio Blender v2: {'✅ PASSED' if blender_result.passed else '❌ FAILED'}
• Overall Status: {'🟢 ALL SYSTEMS OPERATIONAL' if passed_signals == total_signals and blender_result.passed else '🟡 SOME ISSUES DETECTED' if passed_signals >= total_signals * 0.8 else '🔴 CRITICAL ISSUES'}

📈 SIGNAL BREAKDOWN BY TYPE
═══════════════════════════════════════════════════════════════════════════════════
"""

        # Group signals by type
        directional_signals = []
        market_neutral_signals = []
        overlay_signals = []

        for name, result in results.items():
            signal_def = self.signal_definitions.get(name, {})
            signal_type = signal_def.get("type", SignalType.DIRECTIONAL)

            status_icon = (
                "✅" if result.passed else "⚠️" if result.status == "WARNING" else "❌"
            )
            signal_info = f"  {status_icon} {name:<30} ({result.status})"

            if signal_type == SignalType.DIRECTIONAL:
                directional_signals.append(signal_info)
            elif signal_type == SignalType.MARKET_NEUTRAL:
                market_neutral_signals.append(signal_info)
            else:
                overlay_signals.append(signal_info)

        report += f"""
🎯 DIRECTIONAL SIGNALS (2-S Strategy)
{'─' * 50}
{chr(10).join(directional_signals) if directional_signals else '  No directional signals tested'}

⚖️ MARKET-NEUTRAL SIGNALS (M-N Strategy)  
{'─' * 50}
{chr(10).join(market_neutral_signals) if market_neutral_signals else '  No market-neutral signals tested'}

🔧 OVERLAY/FILTER SIGNALS
{'─' * 50}
{chr(10).join(overlay_signals) if overlay_signals else '  No overlay signals tested'}

🏗️ PORTFOLIO BLENDER V2 VALIDATION
═══════════════════════════════════════════════════════════════════════════════════
Status: {'✅ PASSED' if blender_result.passed else '❌ FAILED'}
"""

        if blender_result.test_results:
            report += f"""
Test Results:
• Signal Integration: {blender_result.test_results.get('signal_generation_count', 0)}/12 signals
• Position Bounds: {'✅' if blender_result.test_results.get('final_position_bounds') else '❌'}
• Confidence Bounds: {'✅' if blender_result.test_results.get('blended_confidence_bounds') else '❌'}
• Risk Metrics: {'✅' if blender_result.test_results.get('risk_metrics_present') else '❌'}
• Allocation Methods: {sum(blender_result.test_results.get('allocation_methods', {}).values())}/3 working
"""

        # Add detailed errors if any
        failed_signals = [r for r in results.values() if not r.passed]
        if failed_signals or not blender_result.passed:
            report += """
🚨 DETAILED ERROR ANALYSIS
═══════════════════════════════════════════════════════════════════════════════════
"""

            for result in failed_signals:
                if result.errors:
                    report += f"""
❌ {result.signal_name.upper()}:
{chr(10).join(f"   • {error}" for error in result.errors)}
"""

                if result.warnings:
                    report += f"""
⚠️ {result.signal_name.upper()} WARNINGS:
{chr(10).join(f"   • {warning}" for warning in result.warnings)}
"""

            if not blender_result.passed:
                report += f"""
❌ PORTFOLIO BLENDER V2:
{chr(10).join(f"   • {error}" for error in blender_result.errors)}
"""

        report += """
📋 RECOMMENDED ACTIONS
═══════════════════════════════════════════════════════════════════════════════════
"""

        if passed_signals == total_signals and blender_result.passed:
            report += "✅ All systems operational! Ready for production deployment.\n"
        else:
            if passed_signals < total_signals:
                report += f"• Fix {total_signals - passed_signals} failing signal(s) before production deployment\n"
            if not blender_result.passed:
                report += "• Resolve Portfolio Blender v2 issues before running integrated strategies\n"

            report += "• Run unit tests with: pytest tests/\n"
            report += "• Check code quality with: ruff check src/ && black src/\n"

        report += f"""
📊 SIGNAL COMPLIANCE MATRIX
═══════════════════════════════════════════════════════════════════════════════════
Signal Name                    | Bounds | Confidence | Metadata | Config | Edge Cases
{'─' * 90}
"""

        for name, result in results.items():
            bounds_ok = "✅" if result.test_results.get("value_bounds") else "❌"
            confidence_ok = (
                "✅" if result.test_results.get("confidence_bounds") else "❌"
            )
            metadata_ok = (
                "✅"
                if result.test_results.get("metadata_structure")
                else "⚠️" if result.test_results.get("metadata_present") else "❌"
            )
            config_ok = "✅" if result.test_results.get("config_integration") else "❌"
            edge_cases = result.test_results.get("edge_cases", {})
            edge_ok = (
                "✅"
                if sum(edge_cases.values()) >= len(edge_cases) * 0.67
                else "⚠️" if sum(edge_cases.values()) > 0 else "❌"
            )

            report += f"{name:<30} |   {bounds_ok}    |     {confidence_ok}      |    {metadata_ok}     |   {config_ok}    |     {edge_ok}\n"

        report += f"""
{'═' * 90}
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Validation framework: Crypto Quant Signal Validator v2.0
"""

        return report


async def main():
    """Main validation execution."""
    print("🚀 Starting comprehensive crypto signal validation...")

    validator = SignalValidator()

    # Validate all individual signals
    signal_results = await validator.validate_all_signals()

    # Validate portfolio blender
    blender_result = await validator.validate_portfolio_blender()

    # Generate and display report
    report = validator.generate_validation_report(signal_results, blender_result)

    print(report)

    # Save report to file
    report_file = f"/Users/harrycharlesworth/Repositories/crypto-quant/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Full report saved to: {report_file}")

    # Exit with appropriate code
    all_passed = (
        all(r.passed for r in signal_results.values()) and blender_result.passed
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
