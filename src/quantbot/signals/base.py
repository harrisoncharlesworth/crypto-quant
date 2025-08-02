from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime


@dataclass
class SignalResult:
    """Result from a signal calculation."""

    symbol: str
    timestamp: datetime
    value: float  # Signal strength: -1 (strong sell) to +1 (strong buy)
    confidence: float  # 0 to 1
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_long(self) -> bool:
        """True if signal suggests long position."""
        return self.value > 0

    @property
    def is_short(self) -> bool:
        """True if signal suggests short position."""
        return self.value < 0

    @property
    def is_neutral(self) -> bool:
        """True if signal is neutral."""
        return abs(self.value) < 0.1


@dataclass
class SignalConfig:
    """Base configuration for signals."""

    enabled: bool = True
    weight: float = 1.0
    min_confidence: float = 0.5


class SignalBase(ABC):
    """Abstract base class for all trading signals."""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    async def generate(self, data: pd.DataFrame, symbol: str) -> SignalResult:
        """
        Generate signal from market data.

        Args:
            data: OHLCV dataframe with datetime index
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            SignalResult with value between -1 and 1
        """
        pass

    def validate_data(self, data: pd.DataFrame, min_periods: int = 1) -> bool:
        """Validate input data has sufficient periods."""
        if data is None or len(data) < min_periods:
            return False

        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)

    def normalize_signal(self, raw_value: float, scale: float = 1.0) -> float:
        """Normalize raw signal to -1 to +1 range."""
        return max(-1.0, min(1.0, raw_value / scale))
