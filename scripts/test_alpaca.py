#!/usr/bin/env python3
"""
Test script for Alpaca crypto trading integration.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from quantbot.exchanges.alpaca_wrapper import AlpacaWrapper

# Load environment variables
load_dotenv()

async def test_alpaca_connection():
    """Test Alpaca connection and basic functionality."""
    print("🧪 Testing Alpaca Integration")
    print("=" * 50)
    
    try:
        # Initialize Alpaca wrapper
        print("1. Initializing Alpaca wrapper...")
        alpaca = AlpacaWrapper(paper=True)
        print("✅ Alpaca wrapper initialized")
        
        # Load markets
        print("\n2. Loading available markets...")
        alpaca.load_markets()
        print(f"✅ Loaded {len(alpaca.markets)} markets")
        
        # Show available symbols
        symbols = list(alpaca.markets.keys())[:5]  # Show first 5
        print(f"📊 Available symbols (first 5): {symbols}")
        
        # Test balance
        print("\n3. Testing account balance...")
        balance = alpaca.fetch_balance()
        if balance:
            usd_balance = balance.get('USD', {}).get('free', 0)
            print(f"✅ USD Balance: ${usd_balance:,.2f}")
            
            # Show crypto balances
            crypto_balances = {k: v for k, v in balance.items() if k != 'USD' and v.get('total', 0) > 0}
            if crypto_balances:
                print("💰 Crypto positions:")
                for symbol, bal in crypto_balances.items():
                    print(f"   {symbol}: {bal['total']:.6f}")
        else:
            print("❌ Failed to fetch balance")
        
        # Test ticker data
        print("\n4. Testing ticker data...")
        test_symbol = 'BTCUSD'
        ticker = alpaca.fetch_ticker(test_symbol)
        if ticker:
            print(f"✅ {test_symbol} ticker:")
            print(f"   Last: ${ticker.get('last', 0):,.2f}")
            print(f"   Bid: ${ticker.get('bid', 0):,.2f}")
            print(f"   Ask: ${ticker.get('ask', 0):,.2f}")
        else:
            print(f"❌ Failed to fetch ticker for {test_symbol}")
        
        # Test historical data
        print("\n5. Testing historical data...")
        ohlcv = alpaca.fetch_ohlcv(test_symbol, '1h', 24)
        if ohlcv:
            print(f"✅ Fetched {len(ohlcv)} hours of {test_symbol} data")
            if ohlcv:
                latest = ohlcv[-1]
                print(f"   Latest close: ${latest[4]:,.2f}")
                print(f"   Volume: {latest[5]:,.0f}")
        else:
            print(f"❌ Failed to fetch OHLCV for {test_symbol}")
        
        # Test paper trading order (if we have balance)
        if balance and balance.get('USD', {}).get('free', 0) > 50:
            print("\n6. Testing paper trading order...")
            try:
                # Small test order
                order = alpaca.create_market_buy_order(test_symbol, 0.001)  # $50-100 worth
                if order:
                    print(f"✅ Paper trade order created:")
                    print(f"   Order ID: {order.get('id')}")
                    print(f"   Symbol: {order.get('symbol')}")
                    print(f"   Amount: {order.get('amount')}")
                    print(f"   Status: {order.get('status')}")
                else:
                    print("❌ Failed to create test order")
            except Exception as e:
                print(f"⚠️ Paper trading test failed: {e}")
        else:
            print("\n6. Skipping paper trading test (insufficient balance)")
        
        print("\n🎉 Alpaca integration test completed!")
        
    except Exception as e:
        print(f"❌ Alpaca test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_alpaca_connection())
