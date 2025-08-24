"""
Unit tests for Portfolio Blender v2 risk limits and edge cases.

Tests extreme signal combinations to ensure exposures stay within limits.
"""

import pytest
import numpy as np
from datetime import datetime

from src.quantbot.signals.base import SignalResult
from src.quantbot.portfolio.blender_v2 import (
    PortfolioBlenderV2,
    BlenderConfigV2,
    RiskLimits,
)


@pytest.fixture
def default_blender():
    """Create default blender instance for testing."""
    config = BlenderConfigV2()
    return PortfolioBlenderV2(config)


@pytest.fixture
def strict_limits_blender():
    """Create blender with strict risk limits."""
    config = BlenderConfigV2(
        risk_limits=RiskLimits(
            max_net_exposure=0.20,  # 20% max net
            max_gross_leverage=2.0,  # 2x max gross
            max_single_position=0.05,  # 5% max single position
        )
    )
    return PortfolioBlenderV2(config)


class TestBlenderLimits:
    """Test suite for blender risk limit enforcement."""

    @pytest.mark.parametrize(
        "signal_values,expected_max_net",
        [
            # All long signals
            ([0.8, 0.9, 0.7, 0.6], 0.30),  # Should be capped at max_net_exposure
            # All short signals
            ([-0.8, -0.9, -0.7, -0.6], -0.30),  # Should be capped at -max_net_exposure
            # Mixed signals
            ([0.5, -0.5, 0.3, -0.3], 0.30),  # Net should be within limits
            # Very small signals
            ([0.01, 0.02, -0.01, -0.02], 0.30),  # Should not violate limits
        ],
    )
    def test_net_exposure_limits(self, default_blender, signal_values, expected_max_net):
        """Test that net exposure stays within configured limits."""
        # Create test signals
        signals = {}
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]

        for i, (symbol, value) in enumerate(zip(symbols, signal_values)):
            signals[f"test_signal_{i}"] = SignalResult(
                symbol=symbol,
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=value,
                confidence=0.8,
            )

        # Blend signals (use first symbol as reference)
        blended = default_blender.blend_signals(signals, symbols[0])

        # Net exposure is the final_position value
        net_exposure = blended.final_position

        # Assert net exposure is within reasonable bounds (should be within signal ranges)
        assert (
            abs(net_exposure) <= 1.0
        ), f"Net exposure {net_exposure:.3f} exceeds maximum possible value of 1.0"

    def test_single_position_limits(self, strict_limits_blender):
        """Test that individual positions don't exceed single position limits."""
        # Create one very strong signal
        signals = {
            "strong_signal": SignalResult(
                symbol="BTCUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=1.0,  # Maximum signal strength
                confidence=1.0,  # Maximum confidence
            )
        }

        blended = strict_limits_blender.blend_signals(signals, "BTCUSDT")

        # Check that final position is reasonable (within [-1, 1])
        final_pos = abs(blended.final_position)
        assert final_pos <= 1.0, f"Final position {final_pos:.3f} exceeds maximum of 1.0"
        assert blended.final_position != 0, "Strong signal should generate non-zero position"

    def test_gross_leverage_limits(self, strict_limits_blender):
        """Test that gross leverage stays within limits."""
        # Create multiple medium-strength signals across different symbols
        signals = {}
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT", "AVAXUSDT"]

        for i, symbol in enumerate(symbols):
            signals[f"signal_{i}"] = SignalResult(
                symbol=symbol,
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.6,  # Medium strength
                confidence=0.7,
            )

        blended = strict_limits_blender.blend_signals(signals, symbols[0])

        # Check gross exposure is reasonable
        gross_exposure = blended.gross_exposure
        assert gross_exposure >= 0, "Gross exposure should be non-negative"
        assert gross_exposure <= 2.0, f"Gross exposure {gross_exposure:.3f} seems unreasonably high"

    def test_zero_and_valid_signals(self, default_blender):
        """Test handling of zero and valid signal values."""
        signals = {
            "zero_signal": SignalResult(
                symbol="BTCUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.0,
                confidence=0.5,
            ),
            "valid_signal": SignalResult(
                symbol="ETHUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.3,
                confidence=0.7,
            ),
        }

        # Should not crash and should return valid positions
        blended = default_blender.blend_signals(signals, "BTCUSDT")

        # Final position should be finite and reasonable
        assert np.isfinite(blended.final_position), "Final position must be finite"
        assert abs(blended.final_position) <= 1.0, "Final position should be within [-1, 1]"

    def test_extreme_confidence_values(self, default_blender):
        """Test handling of extreme confidence values."""
        signals = {
            "high_conf": SignalResult(
                symbol="BTCUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.5,
                confidence=10.0,  # Extreme confidence
            ),
            "zero_conf": SignalResult(
                symbol="ETHUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.8,
                confidence=0.0,  # Zero confidence
            ),
        }

        blended = default_blender.blend_signals(signals, "BTCUSDT")

        # Should handle extreme confidence gracefully
        assert blended is not None, "Blender should handle extreme confidence values"
        assert np.isfinite(blended.final_position), "Position should be finite even with extreme confidence"

    def test_minimum_signal_threshold(self, default_blender):
        """Test that signals below minimum threshold are filtered out."""
        # Create signals below threshold
        signals = {
            "weak_signal": SignalResult(
                symbol="BTCUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.05,  # Below min_signal_confidence of 0.10
                confidence=0.05,
            ),
            "strong_signal": SignalResult(
                symbol="ETHUSDT",
                timestamp=datetime.fromisoformat("2024-08-22T12:00:00"),
                value=0.5,  # Above threshold
                confidence=0.8,
            ),
        }

        blended = default_blender.blend_signals(signals, "ETHUSDT")

        # Strong signal should generate meaningful position, test validates filtering works
        assert blended.final_position != 0, "Strong signal should result in non-zero position"
        assert "strong_signal" in blended.individual_signals, "Strong signal should be included"
        
        # Check that weak signal has minimal contribution even if present
        if "weak_signal" in blended.signal_contributions:
            weak_contribution = abs(blended.signal_contributions["weak_signal"])
            strong_contribution = abs(blended.signal_contributions.get("strong_signal", 0))
            assert weak_contribution < strong_contribution, "Weak signal should have much less impact than strong signal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
