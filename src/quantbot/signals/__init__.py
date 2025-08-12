from .base import SignalBase, SignalResult, SignalConfig
from .momentum import (
    TimeSeriesMomentumSignal,
    CrossSectionalMomentumSignal,
    MomentumConfig,
)
from .breakout import DonchianBreakoutSignal, BreakoutConfig
from .mean_reversion import ShortTermMeanReversionSignal, MeanReversionConfig
from .funding_carry import PerpFundingCarrySignal, FundingCarryConfig
from .cash_carry import CashCarryArbitrageSignal, CashCarryConfig
from .oi_divergence import OIPriceDivergenceSignal, OIDivergenceConfig
from .x_exchange_funding import (
    XExchangeFundingDispersionSignal,
    XExchangeFundingConfig,
)

# Handle optional dependencies gracefully
try:
    from .vol_risk_premium import (
        VolatilityRiskPremiumSignal,
        VolRiskPremiumConfig,
    )
    VOL_RISK_PREMIUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: vol_risk_premium not available: {e}")
    VOL_RISK_PREMIUM_AVAILABLE = False
    VolatilityRiskPremiumSignal = None
    VolRiskPremiumConfig = None

try:
    from .skew_whipsaw import SkewWhipsawSignal, SkewWhipsawConfig
    SKEW_WHIPSAW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: skew_whipsaw not available: {e}")
    SKEW_WHIPSAW_AVAILABLE = False
    SkewWhipsawSignal = None
    SkewWhipsawConfig = None

try:
    from .ssr import StablecoinSupplyRatioSignal, SSRConfig
    SSR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ssr not available: {e}")
    SSR_AVAILABLE = False
    StablecoinSupplyRatioSignal = None
    SSRConfig = None

try:
    from .mvrv import MVRVSignal, MVRVConfig
    MVRV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: mvrv not available: {e}")
    MVRV_AVAILABLE = False
    MVRVSignal = None
    MVRVConfig = None

# Build __all__ list dynamically based on available modules
__all__ = [
    # Base classes
    "SignalBase",
    "SignalResult",
    "SignalConfig",
    # Momentum signals
    "TimeSeriesMomentumSignal",
    "CrossSectionalMomentumSignal",
    "MomentumConfig",
    # Breakout signals
    "DonchianBreakoutSignal",
    "BreakoutConfig",
    # Mean reversion signals
    "ShortTermMeanReversionSignal",
    "MeanReversionConfig",
    # Funding carry signals
    "PerpFundingCarrySignal",
    "FundingCarryConfig",
    # Cash-carry arbitrage signals
    "CashCarryArbitrageSignal",
    "CashCarryConfig",
    # OI divergence signals
    "OIPriceDivergenceSignal",
    "OIDivergenceConfig",
    # Cross-exchange funding signals
    "XExchangeFundingDispersionSignal",
    "XExchangeFundingConfig",
]

# Add optional signals if available
if VOL_RISK_PREMIUM_AVAILABLE:
    __all__.extend(["VolatilityRiskPremiumSignal", "VolRiskPremiumConfig"])

if SKEW_WHIPSAW_AVAILABLE:
    __all__.extend(["SkewWhipsawSignal", "SkewWhipsawConfig"])

if SSR_AVAILABLE:
    __all__.extend(["StablecoinSupplyRatioSignal", "SSRConfig"])

if MVRV_AVAILABLE:
    __all__.extend(["MVRVSignal", "MVRVConfig"])
