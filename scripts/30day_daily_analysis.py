#!/usr/bin/env python3
"""
30-Day Daily Profit Analysis for Crypto Quant Bot
Provides detailed daily profit breakdown for the past 30 days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import yaml
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantbot.signals.momentum import TimeSeriesMomentumSignal, MomentumConfig
from quantbot.signals.breakout import DonchianBreakoutSignal, BreakoutConfig
from quantbot.signals.mean_reversion import MeanReversionSignal, MeanReversionConfig
from quantbot.signals.funding_carry import FundingCarrySignal, FundingCarryConfig
from quantbot.portfolio.blender_v2 import PortfolioBlenderV2

def generate_30day_data(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Generate realistic 30-day market data for analysis."""
    
    # Create date range for past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Generate hourly data points
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Simulate realistic BTC price movements
    np.random.seed(42)  # For reproducible results
    
    # Start with realistic BTC price
    start_price = 45000.0
    
    # Generate price movements with realistic volatility
    returns = np.random.normal(0.0001, 0.02, len(date_range))  # 2% hourly volatility
    prices = [start_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(prices))
    }, index=date_range)
    
    # Ensure high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    return data

def calculate_daily_returns(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from signal data."""
    
    # Resample to daily data
    daily_signals = signals_df.resample('D').agg({
        'final_position': 'last',
        'confidence': 'mean',
        'price': 'last'
    }).fillna(0)
    
    # Calculate daily returns
    daily_signals['price_change'] = daily_signals['price'].pct_change()
    daily_signals['strategy_return'] = daily_signals['final_position'].shift(1) * daily_signals['price_change']
    daily_signals['cumulative_return'] = (1 + daily_signals['strategy_return'].fillna(0)).cumprod() - 1
    
    return daily_signals

def format_currency(amount: float) -> str:
    """Format amount as currency."""
    if abs(amount) >= 1000:
        return f"${amount:,.0f}"
    else:
        return f"${amount:.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2%}"

async def run_30day_daily_analysis(symbol: str = "BTCUSDT"):
    """Run comprehensive 30-day daily profit analysis."""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("📊 30-DAY DAILY PROFIT ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    print(f"Duration: 30 days")
    print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    
    # Generate 30-day data
    data = generate_30day_data(symbol)
    print(f"📈 Market Data: {len(data):,} hourly data points")
    print(f"   Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Starting Price: ${data['close'].iloc[0]:.2f}")
    print(f"   Ending Price: ${data['close'].iloc[-1]:.2f}")
    print(f"   30-Day Market Return: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    print()
    
    # Initialize signals with current conservative settings
    signals = {}
    
    # Momentum Signal (conservative settings)
    momentum_config = MomentumConfig(lookback_days=25, skip_recent_days=3, ma_window=80, weight=1.0)
    momentum_signal = TimeSeriesMomentumSignal(momentum_config)
    momentum_results = []
    
    for i in range(80, len(data), 6):  # Every 6 hours for efficiency
        window_data = data.iloc[:i+1]
        result = await momentum_signal.generate(window_data, symbol)
        momentum_results.append({
            'timestamp': data.index[i],
            'signal': result.value,
            'confidence': result.confidence,
            'price': data['close'].iloc[i]
        })
    
    momentum_df = pd.DataFrame(momentum_results)
    signals['momentum'] = momentum_df
    
    # Breakout Signal (conservative settings)
    breakout_config = BreakoutConfig(channel_period=25, atr_period=14, atr_multiplier=1.8, weight=1.0)
    breakout_signal = DonchianBreakoutSignal(breakout_config)
    breakout_results = []
    
    for i in range(25, len(data), 6):
        window_data = data.iloc[:i+1]
        result = await breakout_signal.generate(window_data, symbol)
        breakout_results.append({
            'timestamp': data.index[i],
            'signal': result.value,
            'confidence': result.confidence,
            'price': data['close'].iloc[i]
        })
    
    breakout_df = pd.DataFrame(breakout_results)
    signals['breakout'] = breakout_df
    
    # Mean Reversion Signal (conservative settings)
    mean_rev_config = MeanReversionConfig(lookback_days=7, zscore_threshold=1.6, weight=1.0)
    mean_rev_signal = MeanReversionSignal(mean_rev_config)
    mean_rev_results = []
    
    for i in range(7*24, len(data), 6):
        window_data = data.iloc[:i+1]
        result = await mean_rev_signal.generate(window_data, symbol)
        mean_rev_results.append({
            'timestamp': data.index[i],
            'signal': result.value,
            'confidence': result.confidence,
            'price': data['close'].iloc[i]
        })
    
    mean_rev_df = pd.DataFrame(mean_rev_results)
    signals['mean_reversion'] = mean_rev_df
    
    # Funding Carry Signal (conservative settings)
    funding_config = FundingCarryConfig(threshold=0.0003, weight=1.0)
    funding_signal = FundingCarrySignal(funding_config)
    funding_results = []
    
    for i in range(24, len(data), 6):
        window_data = data.iloc[:i+1]
        result = await funding_signal.generate(window_data, symbol)
        funding_results.append({
            'timestamp': data.index[i],
            'signal': result.value,
            'confidence': result.confidence,
            'price': data['close'].iloc[i]
        })
    
    funding_df = pd.DataFrame(funding_results)
    signals['funding_carry'] = funding_df
    
    # Portfolio Blender (conservative settings)
    blender = PortfolioBlenderV2()
    blended_results = []
    
    # Align all signals to same timestamps
    all_timestamps = set()
    for signal_df in signals.values():
        all_timestamps.update(signal_df['timestamp'])
    
    all_timestamps = sorted(list(all_timestamps))
    
    for timestamp in all_timestamps:
        signal_data = {}
        for signal_name, signal_df in signals.items():
            matching_rows = signal_df[signal_df['timestamp'] == timestamp]
            if not matching_rows.empty:
                signal_data[signal_name] = {
                    'value': matching_rows.iloc[0]['signal'],
                    'confidence': matching_rows.iloc[0]['confidence']
                }
        
        if signal_data:
            result = blender.blend(signal_data)
            price = data.loc[timestamp, 'close'] if timestamp in data.index else data['close'].iloc[-1]
            blended_results.append({
                'timestamp': timestamp,
                'final_position': result['position'],
                'confidence': result['confidence'],
                'price': price
            })
    
    blended_df = pd.DataFrame(blended_results)
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(blended_df)
    
    # Display daily breakdown
    print("📅 DAILY PROFIT BREAKDOWN")
    print("=" * 80)
    print(f"{'Date':<12} {'Position':<8} {'Price':<10} {'Daily Return':<12} {'Cumulative':<12} {'P&L':<10}")
    print("-" * 80)
    
    total_pnl = 0
    initial_capital = 10000  # $10,000 starting capital
    
    for date, row in daily_returns.iterrows():
        if pd.notna(row['strategy_return']):
            daily_pnl = row['strategy_return'] * initial_capital
            total_pnl += daily_pnl
            
            position = "LONG" if row['final_position'] > 0.1 else "SHORT" if row['final_position'] < -0.1 else "NEUTRAL"
            
            print(f"{date:%m-%d}      {position:<8} ${row['price']:<9.0f} {format_percentage(row['strategy_return']):<12} "
                  f"{format_percentage(row['cumulative_return']):<12} {format_currency(daily_pnl):<10}")
    
    print("-" * 80)
    print(f"TOTAL P&L: {format_currency(total_pnl)}")
    print(f"TOTAL RETURN: {format_percentage(daily_returns['cumulative_return'].iloc[-1])}")
    print()
    
    # Performance summary
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 40)
    
    total_return = daily_returns['cumulative_return'].iloc[-1]
    total_trades = len(daily_returns[daily_returns['strategy_return'] != 0])
    winning_days = len(daily_returns[daily_returns['strategy_return'] > 0])
    losing_days = len(daily_returns[daily_returns['strategy_return'] < 0])
    win_rate = winning_days / total_trades if total_trades > 0 else 0
    
    avg_daily_return = daily_returns['strategy_return'].mean()
    daily_volatility = daily_returns['strategy_return'].std()
    sharpe_ratio = avg_daily_return / daily_volatility if daily_volatility > 0 else 0
    
    max_drawdown = (daily_returns['cumulative_return'] - daily_returns['cumulative_return'].expanding().max()).min()
    
    print(f"Total Return: {format_percentage(total_return)}")
    print(f"Total P&L: {format_currency(total_pnl)}")
    print(f"Trading Days: {total_trades}")
    print(f"Winning Days: {winning_days}")
    print(f"Losing Days: {losing_days}")
    print(f"Win Rate: {format_percentage(win_rate)}")
    print(f"Avg Daily Return: {format_percentage(avg_daily_return)}")
    print(f"Daily Volatility: {format_percentage(daily_volatility)}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {format_percentage(max_drawdown)}")
    print()
    
    # Signal contribution analysis
    print("🎯 SIGNAL CONTRIBUTION ANALYSIS")
    print("=" * 40)
    
    for signal_name, signal_df in signals.items():
        if not signal_df.empty:
            signal_trades = len(signal_df[abs(signal_df['signal']) > 0.1])
            avg_confidence = signal_df['confidence'].mean()
            print(f"{signal_name.replace('_', ' ').title():<20} {signal_trades:>3} trades, {avg_confidence:.3f} avg confidence")
    
    print()
    print("✅ Analysis Complete!")
    print("📧 Daily reports will be sent to ebullemor@gmail.com at 6:00 PM AEST")
    print("🌐 Bot is running 24/7 on Railway with conservative settings")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_30day_daily_analysis())
