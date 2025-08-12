#!/usr/bin/env python3
"""
Price Evolution Analysis: Shows exact price changes from 2022 to 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show_price_evolution():
    """Show the exact price evolution from 2022 to 2025."""
    
    print("💰 PRICE EVOLUTION ANALYSIS")
    print("=" * 50)
    print("Showing how BTC price evolves from 2022 to 2025")
    print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    
    # Generate 3-year data (same as 3-year report)
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    dates = pd.date_range(start=start_date, end=end_date, freq="h")
    
    start_price = 35000  # August 2022 price
    returns = np.random.normal(0.0002, 0.025, len(dates))
    
    # Add market cycles
    total_len = len(returns)
    year1_end = total_len // 3
    year2_end = 2 * total_len // 3
    
    # Year 1 (2022-2023): Bear market
    returns[:year1_end] += np.random.normal(-0.0003, 0.030, year1_end)
    
    # Year 2 (2023-2024): Bull market
    returns[year1_end:year2_end] += np.random.normal(0.0008, 0.020, year2_end - year1_end)
    
    # Year 3 (2024-2025): Consolidation
    returns[year2_end:] += np.random.normal(0.0004, 0.018, total_len - year2_end)
    
    # Calculate prices
    prices = [start_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices
    })
    df.set_index('timestamp', inplace=True)
    
    print("📈 PRICE EVOLUTION TIMELINE")
    print("-" * 50)
    
    # Key dates to show
    key_dates = [
        ('2022-08-11', 'Start (August 2022)'),
        ('2023-01-01', 'January 2023'),
        ('2023-08-11', '1 Year Later'),
        ('2024-01-01', 'January 2024'),
        ('2024-08-11', '2 Years Later'),
        ('2025-01-01', 'January 2025'),
        ('2025-08-10', 'Current (August 2025)')
    ]
    
    print(f"{'Date':<12} {'Price':<12} {'Change':<15} {'Period':<20}")
    print("-" * 60)
    
    start_price_actual = None
    for date_str, description in key_dates:
        try:
            date = pd.to_datetime(date_str)
            if date in df.index:
                price = df.loc[date, 'price']
                if start_price_actual is None:
                    start_price_actual = price
                    change = "0.00%"
                else:
                    change_pct = ((price / start_price_actual) - 1) * 100
                    change = f"{change_pct:+.2f}%"
                
                print(f"{date_str:<12} ${price:<11.2f} {change:<15} {description:<20}")
        except:
            # Find closest date
            closest_date = df.index[df.index.get_indexer([date], method='nearest')[0]]
            price = df.loc[closest_date, 'price']
            if start_price_actual is None:
                start_price_actual = price
                change = "0.00%"
            else:
                change_pct = ((price / start_price_actual) - 1) * 100
                change = f"{change_pct:+.2f}%"
            
            print(f"{closest_date.strftime('%Y-%m-%d'):<12} ${price:<11.2f} {change:<15} {description:<20}")
    
    print()
    
    # Yearly breakdown
    print("📊 YEARLY PRICE SUMMARY")
    print("-" * 50)
    
    for year in [2022, 2023, 2024, 2025]:
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            year_start = year_data.iloc[0]['price']
            year_end = year_data.iloc[-1]['price']
            year_return = ((year_end / year_start) - 1) * 100
            
            print(f"{year}: ${year_start:.2f} → ${year_end:.2f} ({year_return:+.2f}%)")
    
    print()
    
    # The problem explained
    print("🔍 THE PRICE PROBLEM EXPLAINED")
    print("-" * 50)
    print("1. **3-Year Report**:")
    print(f"   • Starts at ${start_price:,.2f} in August 2022")
    print(f"   • Evolves to ${df[df.index.year == 2025].iloc[0]['price']:.2f} in January 2025")
    print(f"   • This is the '3-Year 2025 Starting Price' I referenced")
    print()
    print("2. **YTD Report**:")
    print(f"   • Starts fresh at $45,000 in January 2025")
    print(f"   • This is the 'YTD 2025 Starting Price' I referenced")
    print()
    print("3. **The Discrepancy**:")
    print(f"   • 3-Year January 2025: ${df[df.index.year == 2025].iloc[0]['price']:.2f}")
    print(f"   • YTD January 2025: $45,000.00")
    print(f"   • Difference: ${abs(45000 - df[df.index.year == 2025].iloc[0]['price']):.2f}")
    print()
    print("4. **Why This Happens**:")
    print("   • 3-Year report: Continuous price evolution from 2022")
    print("   • YTD report: Fresh start with realistic 2025 price")
    print("   • Different market cycle modeling approaches")
    print()
    
    # Show the exact comparison
    print("📊 EXACT COMPARISON")
    print("-" * 50)
    
    # Get 2025 data from 3-year generation
    df_2025 = df[df.index.year == 2025]
    
    print("**3-Year Report 2025 Data:**")
    print(f"   Starting Price (Jan 2025): ${df_2025.iloc[0]['price']:.2f}")
    print(f"   Ending Price (Aug 2025): ${df_2025.iloc[-1]['price']:.2f}")
    print(f"   2025 Return: {((df_2025.iloc[-1]['price'] / df_2025.iloc[0]['price']) - 1) * 100:.2f}%")
    print()
    print("**YTD Report 2025 Data:**")
    print(f"   Starting Price (Jan 2025): $45,000.00")
    print(f"   Ending Price (Aug 2025): $24,357.61 (from comparison)")
    print(f"   2025 Return: -45.87%")
    print()
    print("**The Prices I Referenced:**")
    print(f"   • YTD 2025 Starting Price: $45,000.00")
    print(f"   • 3-Year 2025 Starting Price: ${df_2025.iloc[0]['price']:.2f}")
    print(f"   • 3-Year 2022 Starting Price: ${start_price:,.2f}")

if __name__ == "__main__":
    show_price_evolution()
