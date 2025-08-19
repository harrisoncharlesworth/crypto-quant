# Portfolio Expansion & Scaling Plan

## Overview
Expand from 8 to 38 trading pairs and scale position sizing to utilize $200k equity effectively.

## Phase 1: Ticker Expansion (30 Additional Pairs)

### New Trading Pairs
Based on Oracle analysis - liquid, diversified crypto universe:

**Layer 1 Protocols:**
- DOT, AVAX, TRX, ATOM, NEAR, ALGO, BCH, ETC, ICP, EGLD, APT, HBAR, FTM

**DeFi & Interoperability:**
- LINK, UNI, CRV, GRT, QNT

**Layer 2 Solutions:**
- OP, ARB

**Meme/Community:**
- DOGE, SHIB

**Gaming/Metaverse:**
- SAND, AXS, MANA

**Specialized:**
- XLM, VET, FIL, THETA, GMX

### Selection Criteria Met:
- ≥$40M daily volume
- ≤20bp median spread at $25k size
- Market cap >$900M
- Available on Binance spot
- Correlation with BTC <0.80 for 60% of time

## Phase 2: Position Sizing Overhaul

### New Risk Framework
- **Equity-at-Risk (EaR)**: 0.5% of NAV per trade ($1,000)
- **Position sizing**: EaR / (10-day ATR × 1.2)
- **Max portfolio heat**: 8% NAV ($16,000)
- **Single pair cap**: min(5% NAV, 20% ADV) = $10,000
- **Sector concentration**: max 25% NAV per sector

### Scaling Timeline
1. **Week 1-2**: Dry run at 25% sizing
2. **Week 3-10**: Live at 0.5% EaR, 8% heat
3. **After 60 days**: If drawdown <5% and Sharpe >1, scale to 1% EaR

## Phase 3: Implementation Steps

### Code Changes Required
1. **Update ticker list** in configuration
2. **Implement ATR-based sizing** in Portfolio Blender
3. **Add portfolio heat monitoring**
4. **Implement sector caps**
5. **Add ADV/liquidity checks**

### Risk Controls
- Real-time portfolio heat calculation
- Position size validation before orders
- Circuit breakers for gap risk
- Per-pair trading halt logic

### Testing & Validation
- Backtest 38-pair universe with current signals
- Validate slippage assumptions
- Test risk control mechanisms
- Monitor correlation changes

## Expected Outcomes
- **Capital utilization**: Increase from ~20% to 80-90%
- **Diversification**: Reduce concentration risk
- **Risk-adjusted returns**: Maintain Sharpe >1.0
- **Max drawdown**: Keep <15%

## Success Metrics
- Portfolio heat stays <8% NAV
- No single position >5% NAV
- Weekly risk review compliance
- Gradual scaling based on performance

## Timeline
- **Day 1**: Implement configuration changes
- **Day 2-3**: Code ATR sizing and risk controls
- **Day 4-7**: Testing and validation
- **Day 8+**: Phased live deployment
