#!/usr/bin/env python3
"""
Alternative Binance testnet connection test with different configurations
"""

import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

def test_binance_different_configs():
    """Try different Binance testnet configurations."""
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')
    
    if not api_key or not secret:
        print("❌ API credentials not found")
        return
    
    print("🧪 Testing Different Binance Configurations")
    print("=" * 50)
    
    # Configuration 1: Standard testnet
    print("\n1️⃣ Testing Standard Testnet Config...")
    try:
        exchange1 = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': True,
            'enableRateLimit': True,
        })
        balance1 = exchange1.fetch_balance()
        print("✅ Config 1 SUCCESS!")
        print(f"   USDT Balance: {balance1.get('USDT', {}).get('free', 0)}")
        return True
    except Exception as e:
        print(f"❌ Config 1 failed: {e}")
    
    # Configuration 2: Manual testnet URLs
    print("\n2️⃣ Testing Manual URL Config...")
    try:
        exchange2 = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': True,
            'enableRateLimit': True,
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                    'sapi': 'https://testnet.binance.vision/sapi',
                    'sapiV2': 'https://testnet.binance.vision/sapi/v2',
                }
            }
        })
        balance2 = exchange2.fetch_balance()
        print("✅ Config 2 SUCCESS!")
        print(f"   USDT Balance: {balance2.get('USDT', {}).get('free', 0)}")
        return True
    except Exception as e:
        print(f"❌ Config 2 failed: {e}")
    
    # Configuration 3: Different testnet approach
    print("\n3️⃣ Testing Alternative Testnet...")
    try:
        exchange3 = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'test': True,  # Alternative sandbox flag
        })
        balance3 = exchange3.fetch_balance()
        print("✅ Config 3 SUCCESS!")
        print(f"   USDT Balance: {balance3.get('USDT', {}).get('free', 0)}")
        return True
    except Exception as e:
        print(f"❌ Config 3 failed: {e}")
    
    print("\n🔧 All configurations failed. Possible issues:")
    print("1. API key needs 'Enable Spot & Margin Trading' permission")
    print("2. API key might be too new (wait 15 minutes)")
    print("3. IP restrictions might be blocking access")
    print("4. API key might need to be recreated")
    
    return False

if __name__ == "__main__":
    test_binance_different_configs()
