#!/usr/bin/env python3
"""
Debug Binance authentication with detailed error information
"""

import os
import ccxt
from dotenv import load_dotenv
import time

load_dotenv()

def test_auth_detailed():
    """Test authentication with detailed error reporting."""
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')
    
    print("🔐 Detailed Authentication Test")
    print("=" * 40)
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    print(f"Secret: {secret[:10]}...{secret[-10:]}")
    
    try:
        # Create exchange with detailed config
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': True,
            'enableRateLimit': True,
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            },
            'options': {
                'defaultType': 'spot',
            }
        })
        
        print("\n🧪 Testing different endpoints...")
        
        # Test 1: Server time
        try:
            server_time = exchange.fetch_time()
            print(f"✅ Server time: {server_time}")
        except Exception as e:
            print(f"❌ Server time error: {e}")
        
        # Test 2: Account status (simple)
        try:
            status = exchange.fetch_status()
            print(f"✅ Exchange status: {status}")
        except Exception as e:
            print(f"❌ Status error: {e}")
            
        # Test 3: Balance (requires auth)
        try:
            balance = exchange.fetch_balance()
            print(f"✅ Balance fetch successful!")
            print(f"   Free USDT: {balance.get('USDT', {}).get('free', 0)}")
            return True
        except ccxt.AuthenticationError as e:
            print(f"❌ Authentication Error: {e}")
            print("\n🔧 Possible solutions:")
            print("1. Wait 10-15 minutes for API key to activate")
            print("2. Check 'Enable Spot & Margin Trading' is checked")
            print("3. Create a new API key if this one is old")
            print("4. Verify IP restrictions aren't blocking access")
            return False
        except Exception as e:
            print(f"❌ Other error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    test_auth_detailed()
