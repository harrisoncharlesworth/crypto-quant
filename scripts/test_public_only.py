#!/usr/bin/env python3
"""
Test just public data access (no authentication required)
"""

import ccxt
import asyncio

async def test_public_data():
    """Test public Binance endpoints only."""
    print("🔍 Testing Public Data Access...")
    
    try:
        # Test mainnet public data
        print("\n📊 Mainnet Public Data:")
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"   BTC/USDT Price: ${ticker['last']:,.2f}")
        print(f"   24h Volume: {ticker['baseVolume']:,.2f} BTC")
        
        # Test testnet public data  
        print("\n🧪 Testnet Public Data:")
        testnet_exchange = ccxt.binance({
            'sandbox': True,
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            }
        })
        
        testnet_ticker = testnet_exchange.fetch_ticker('BTC/USDT')
        print(f"   BTC/USDT Price: ${testnet_ticker['last']:,.2f}")
        print(f"   24h Volume: {testnet_ticker['baseVolume']:,.2f} BTC")
        
        # Test order book
        orderbook = testnet_exchange.fetch_order_book('BTC/USDT', limit=5)
        print(f"   Best Bid: ${orderbook['bids'][0][0]:,.2f}")
        print(f"   Best Ask: ${orderbook['asks'][0][0]:,.2f}")
        
        print("\n✅ Public data access working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_public_data())
