#!/usr/bin/env python3
"""
Test script to verify Railway deployment is working with Alpaca.
"""

import requests
import json
from datetime import datetime

def test_railway_deployment(railway_url):
    """Test Railway deployment health and functionality."""
    print("🧪 Testing Railway Deployment")
    print("=" * 50)
    
    if not railway_url:
        print("❌ No Railway URL provided")
        return False
    
    # Ensure URL has http/https
    if not railway_url.startswith(('http://', 'https://')):
        railway_url = f"https://{railway_url}"
    
    try:
        # Test health endpoint
        print(f"1. Testing health endpoint: {railway_url}/health")
        response = requests.get(f"{railway_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Railway deployment: {e}")
        return False

def check_deployment_logs():
    """Provide instructions for checking logs."""
    print("\n📋 Next Steps - Check Railway Logs:")
    print("1. Go to your Railway dashboard")
    print("2. Click on your crypto-quant service")
    print("3. Click 'Deployments' tab")
    print("4. Look for these log messages:")
    print("   ✅ '🚀 Starting Crypto Quant Trading Bot on Railway...'")
    print("   ✅ '✅ Alpaca exchange connection established'")
    print("   ✅ '✅ Signals setup complete'")
    print("   ✅ '🤖 Starting trading loop...'")
    print("\n📧 Email Notifications:")
    print("   You should receive startup email if configured")

if __name__ == "__main__":
    # You can test with your Railway URL
    railway_url = input("Enter your Railway URL (or press Enter to skip): ").strip()
    
    if railway_url:
        success = test_railway_deployment(railway_url)
        if success:
            print("\n🎉 Railway deployment is healthy!")
        else:
            print("\n❌ Railway deployment has issues")
    else:
        print("⏭️ Skipping URL test")
    
    check_deployment_logs()
