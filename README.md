# Crypto Quantitative Trading Bot

A cost-efficient Python-based crypto trading bot focused on incremental gains through systematic rule-based signals.

## Features

- 20+ crypto-specific quantitative signals (momentum, carry, on-chain, funding rates)
- Email-based notifications for cost efficiency
- Modular architecture supporting multiple exchanges
- Risk management and position sizing
- Backtesting and live trading capabilities

## Architecture

- **Signals**: Momentum, mean reversion, carry, volatility, on-chain metrics
- **Data**: Exchange connectors (Binance, Bybit) with local caching
- **Execution**: Dry-run and live trading with slippage modeling
- **Risk**: Portfolio allocation, leverage limits, liquidation protection
- **Notifications**: SMTP email alerts for trades and risk events

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Configure exchange API keys and SMTP

# Test individual signals
python -m scripts.backtest --symbol BTCUSDT --signal momentum
python -m scripts.backtest --symbol BTCUSDT --signal breakout
python -m scripts.backtest --symbol BTCUSDT --signal mean_reversion

# Test portfolio blender with all signals
python -m scripts.backtest --symbol BTCUSDT --signal multi

# Run parameter optimization
python -m scripts.test_grid_search
```

## Signal Performance (Backtest Results)

**Individual Signals:**
- **Momentum**: 15.27% return, 0.76 Sharpe ratio
- **Breakout**: 281.83% return, 2.00 Sharpe ratio  
- **Mean Reversion**: 64.30% return, 1.41 Sharpe ratio

**Portfolio Blender**: 95.38% return, 1.34 Sharpe ratio (balanced combination)

## Development Roadmap

1. ✅ Repository structure
2. ✅ Core signal framework (Phase 1 complete)
3. ✅ Portfolio blending system
4. ✅ Backtesting engine with multiple signals
5. 🔄 Live data feeds (Phase 2)
6. 🔄 Trading execution (Phase 3)
7. 🔄 Advanced signals (funding, options, on-chain)
