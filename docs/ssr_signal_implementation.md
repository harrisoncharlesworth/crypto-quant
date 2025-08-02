# Stablecoin Supply Ratio (SSR) Signal Implementation

## Overview
The Stablecoin Supply Ratio signal is a market-neutral overlay strategy that detects accumulation phases by measuring "dry powder" levels in the crypto ecosystem. It acts as a portfolio allocation filter rather than a standalone trading signal.

## Key Mechanics

### Signal Calculation
- **SSR Formula**: `SSR = Stablecoin Market Cap / Total Crypto Market Cap`
- **Normalization**: Z-score over 1-year rolling window (252 weekly periods)
- **Frequency**: Weekly rebalancing (Sundays)

### Overlay Strategy Logic
- **SSR < -1σ**: Boost long allocation by 25% (high dry powder = medium-term bullish)
- **SSR > +1σ**: Reduce long allocation by 25% (low dry powder = medium-term bearish)  
- **Neutral range**: No allocation adjustment

### Dry Powder Interpretation
- **Low SSR** = High stablecoin/crypto ratio = More dry powder available = Bullish positioning
- **High SSR** = Low stablecoin/crypto ratio = Less dry powder = Bearish positioning
- **Falling SSR trend** = Dry powder deployment = Accumulation phase detection

## Implementation Features

### Core Components
1. **StablecoinSupplyRatioSignal**: Main signal class inheriting from SignalBase
2. **SSRConfig**: Configuration dataclass with overlay-specific parameters
3. **Mock Data Generation**: Simulates realistic on-chain data structure
4. **Z-score Normalization**: 1-year rolling window with proper regime handling

### Risk Management
- Weekly rebalancing prevents overtrading
- Z-score bounds prevent extreme allocations (±25% max adjustment)
- Overlay classification limits direct market exposure
- Historical calibration maintains regime stability

### Signal Metadata
Rich metadata includes:
- Current SSR value and Z-score
- Allocation adjustment instructions
- Dry powder level interpretation
- Market regime detection
- Multi-timeframe trend analysis

## Usage Integration

### Portfolio Manager Interface
```python
# Get allocation overlay instructions
overlay = signal.get_allocation_overlay(signal_result)
multiplier = overlay["multiplier"]  # 0.75, 1.0, or 1.25
action = overlay["action"]  # increase/decrease/maintain exposure
```

### Configuration Options
- `zscore_window`: Rolling window for normalization (default: 252 weeks)
- `allocation_adjustment`: Max adjustment percentage (default: 25%)
- `long_boost_threshold`: Z-score for long boost (default: -1.0)
- `short_reduce_threshold`: Z-score for reduction (default: +1.0)

## Testing Coverage
Comprehensive test suite covers:
- Signal generation success/failure scenarios
- Z-score calculation accuracy
- Overlay mechanics validation
- Confidence scoring
- Weekly rebalancing logic
- Regime detection algorithms
- Error handling and edge cases
- Allocation adjustment interface

## Research Foundation
Based on Phase 5 requirements:
- Market-neutral (M-N) classification
- Medium-term positioning bias (weeks to months)
- Accumulation phase detection via stablecoin flows
- Overlay filter strategy for existing signals
- Weekly frequency reduces noise and overtrading

## Files Created
- `src/quantbot/signals/ssr.py`: Main implementation
- `tests/test_ssr_signal.py`: Comprehensive test suite
- Updated `src/quantbot/signals/__init__.py`: Package integration

The SSR signal provides sophisticated dry powder monitoring with proper overlay mechanics for portfolio allocation modulation while maintaining risk controls and regime awareness.
