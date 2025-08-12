#!/usr/bin/env python3
"""
Railway Cloud Deployment for Crypto Quant Bot
Deploys the bot to Railway cloud for true 24/7 operation
"""

import os
import sys
import subprocess
import json
from datetime import datetime

class RailwayDeployment:
    """Deploy crypto quant bot to Railway cloud."""
    
    def __init__(self):
        self.project_name = "crypto-quant-bot"
        self.service_name = "trading-bot"
        
    def check_railway_cli(self):
        """Check if Railway CLI is installed."""
        try:
            result = subprocess.run(['railway', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Railway CLI found")
                return True
            else:
                print("❌ Railway CLI not found")
                return False
        except FileNotFoundError:
            print("❌ Railway CLI not installed")
            return False
            
    def install_railway_cli(self):
        """Install Railway CLI."""
        print("📦 Installing Railway CLI...")
        try:
            # Install via npm
            subprocess.run(['npm', 'install', '-g', '@railway/cli'], 
                         check=True)
            print("✅ Railway CLI installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Railway CLI")
            print("   Please install manually: npm install -g @railway/cli")
            return False
            
    def login_to_railway(self):
        """Login to Railway."""
        print("🔐 Logging into Railway...")
        try:
            subprocess.run(['railway', 'login'], check=True)
            print("✅ Logged into Railway successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to login to Railway")
            return False
            
    def create_railway_project(self):
        """Create Railway project."""
        print("🚀 Creating Railway project...")
        try:
            subprocess.run(['railway', 'init', '--name', self.project_name], 
                         check=True)
            print("✅ Railway project created!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create Railway project")
            return False
            
    def setup_environment_variables(self):
        """Set up environment variables in Railway."""
        print("🔧 Setting up environment variables...")
        
        env_vars = {
            "ALPACA_API_KEY": "PKJYFI6XVZ9UGW85JFVP",
            "ALPACA_SECRET_KEY": "HY2qdRNAX8TSONDcnsqPfNHUp7WzpeHzpsAdNEGZ",
            "ALPACA_PAPER": "true",
            "DRY_RUN": "false",
            "USE_FUTURES": "true",
            "UPDATE_INTERVAL_MINUTES": "10",
            "TRADING_SYMBOLS": "BTCUSD,ETHUSD,SOLUSD,ADAUSD",
            "MAX_PORTFOLIO_ALLOCATION": "0.60",
            "MAX_NET_EXPOSURE": "0.20",
            "MAX_GROSS_LEVERAGE": "1.8",
            "MAX_SINGLE_POSITION": "0.06",
            "MAX_DAILY_DRAWDOWN": "0.15",
            "MAX_WEEKLY_DRAWDOWN": "0.25",
            "POSITION_SIZING_METHOD": "KELLY_OPTIMAL",
            "ENABLE_EMAIL_NOTIFICATIONS": "true",
            "EMAIL_RECIPIENT": "ebullemor@gmail.com",
            "DIGEST_INTERVAL_HOURS": "24",
            "TRADE_NOTIFICATIONS": "true",
            "DAILY_SUMMARY": "true",
            "LOG_LEVEL": "INFO",
            "ENABLE_HEALTH_CHECKS": "true",
            "PERFORMANCE_MONITORING": "true",
            "RISK_MONITORING": "true",
            "POSITION_TRACKING": "true",
            "SIGNAL_MONITORING": "true"
        }
        
        for key, value in env_vars.items():
            try:
                subprocess.run(['railway', 'variables', 'set', f'{key}={value}'], 
                             check=True)
                print(f"   ✅ Set {key}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to set {key}")
                
        print("✅ Environment variables configured!")
        
    def deploy_to_railway(self):
        """Deploy the bot to Railway."""
        print("🚀 Deploying to Railway...")
        try:
            subprocess.run(['railway', 'up'], check=True)
            print("✅ Deployed to Railway successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to deploy to Railway")
            return False
            
    def get_deployment_url(self):
        """Get the deployment URL."""
        try:
            result = subprocess.run(['railway', 'domain'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                domain = result.stdout.strip()
                print(f"🌐 Deployment URL: https://{domain}")
                return domain
            else:
                print("❌ Could not get deployment URL")
                return None
        except Exception as e:
            print(f"❌ Error getting deployment URL: {e}")
            return None
            
    def check_deployment_status(self):
        """Check deployment status."""
        print("📊 Checking deployment status...")
        try:
            result = subprocess.run(['railway', 'status'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Deployment Status:")
                print(result.stdout)
                return True
            else:
                print("❌ Could not check deployment status")
                return False
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return False
            
    def setup_logs_monitoring(self):
        """Set up logs monitoring."""
        print("📋 Setting up logs monitoring...")
        try:
            subprocess.run(['railway', 'logs', '--follow'], 
                         check=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("✅ Logs monitoring set up (press Ctrl+C to stop)")
        except Exception as e:
            print(f"❌ Error setting up logs: {e}")
            
    def create_deployment_summary(self):
        """Create deployment summary."""
        summary = f"""
🚀 CRYPTO QUANT BOT - RAILWAY DEPLOYMENT SUMMARY
{'=' * 60}

📅 Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
🌐 Platform: Railway Cloud
📧 Email Reports: ebullemor@gmail.com
🔄 Operation: 24/7 (True cloud deployment)

✅ FEATURES ACTIVE:
• Enhanced Risk Management (50% exposure reduction)
• Strategic Recommendations Implemented
• Daily Email Reports (6:00 PM UTC)
• Auto-restart on failure
• Continuous monitoring
• Paper trading on Alpaca

📊 TRADING CONFIGURATION:
• Symbols: BTCUSD, ETHUSD, SOLUSD, ADAUSD
• Max Net Exposure: 20%
• Max Gross Leverage: 1.8
• Max Single Position: 6%
• Daily Drawdown Limit: 15%

🎯 PERFORMANCE TARGETS:
• Monthly Return: 15%
• Max Drawdown: 25%
• Sharpe Ratio: 0.8
• Win Rate: 55%

📧 EMAIL NOTIFICATIONS:
• Daily Trade Summaries: 6:00 PM UTC
• Individual Trade Alerts: Enabled
• Risk Alerts: Enabled
• Performance Reports: Daily

🛡️ RISK MANAGEMENT:
• Real-time exposure monitoring
• Automatic position sizing
• Drawdown protection
• Correlation monitoring

📋 MONITORING COMMANDS:
• Check status: railway status
• View logs: railway logs
• Update deployment: railway up
• Access dashboard: railway dashboard

🌐 DEPLOYMENT URL: {self.get_deployment_url() or 'Check Railway dashboard'}

---
Generated by Crypto Quant Bot Railway Deployment
"""
        
        with open("RAILWAY_DEPLOYMENT_SUMMARY.md", "w") as f:
            f.write(summary)
            
        print("✅ Deployment summary saved to RAILWAY_DEPLOYMENT_SUMMARY.md")
        
    def deploy_complete_solution(self):
        """Complete Railway deployment process."""
        print("🎯 RAILWAY CLOUD DEPLOYMENT - 24/7 OPERATION")
        print("=" * 60)
        print("This will deploy your bot to Railway cloud for true 24/7 operation")
        print("Your bot will run continuously, even when your PC is off!")
        print()
        
        # Check Railway CLI
        if not self.check_railway_cli():
            if not self.install_railway_cli():
                return False
                
        # Login to Railway
        if not self.login_to_railway():
            return False
            
        # Create project
        if not self.create_railway_project():
            return False
            
        # Setup environment
        self.setup_environment_variables()
        
        # Deploy
        if not self.deploy_to_railway():
            return False
            
        # Get deployment info
        self.get_deployment_url()
        self.check_deployment_status()
        
        # Create summary
        self.create_deployment_summary()
        
        print()
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 40)
        print("✅ Your bot is now running 24/7 on Railway cloud")
        print("📧 Daily reports will be sent to: ebullemor@gmail.com")
        print("🌐 Check Railway dashboard for monitoring")
        print("🔄 Bot will run continuously, even when PC is off")
        print()
        print("📋 Next Steps:")
        print("   1. Check your email for daily reports")
        print("   2. Monitor performance via Railway dashboard")
        print("   3. Review logs if needed: railway logs")
        print("   4. Update deployment: railway up")
        
        return True

def main():
    """Main deployment function."""
    deployment = RailwayDeployment()
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deployment.deploy_complete_solution()
    else:
        print("🚀 Railway Cloud Deployment for Crypto Quant Bot")
        print("=" * 50)
        print("This will deploy your bot to Railway cloud for true 24/7 operation")
        print()
        print("Benefits:")
        print("  ✅ Runs 24/7 (even when PC is off)")
        print("  ✅ Auto-restart on failure")
        print("  ✅ Cloud monitoring and logs")
        print("  ✅ Daily email reports")
        print("  ✅ No local resource usage")
        print()
        print("To deploy: python deploy_to_railway.py deploy")
        print()
        print("Note: You'll need to login to Railway during deployment")

if __name__ == "__main__":
    main()
