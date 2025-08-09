# 🚀 Crypto Quant Bot - 24/7 Alpaca Paper Trading Deployment Guide

## Overview
This guide will help you deploy the crypto quantitative trading bot to run 24/7 on your Alpaca paper trading account.

## 📋 Prerequisites

### 1. Alpaca Account Setup
- [ ] Create an Alpaca account at [alpaca.markets](https://alpaca.markets)
- [ ] Enable paper trading
- [ ] Generate API keys (Paper Trading)
- [ ] Note your API Key and Secret Key

### 2. System Requirements
- [ ] Ubuntu 20.04+ or similar Linux distribution
- [ ] Python 3.11+
- [ ] 2GB+ RAM
- [ ] Stable internet connection
- [ ] 24/7 uptime capability

## 🔧 Installation Steps

### Step 1: Clone and Setup Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-quant-EB.git
cd crypto-quant-EB

# Install dependencies
pip3 install -r requirements-railway.txt

# Create logs directory
mkdir -p logs
```

### Step 2: Configure Alpaca Credentials
```bash
# Set your Alpaca credentials
export ALPACA_API_KEY="your_paper_trading_api_key"
export ALPACA_SECRET_KEY="your_paper_trading_secret_key"

# Test connection
python3 scripts/test_alpaca.py
```

### Step 3: Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Alpaca Configuration
ALPACA_API_KEY=your_paper_trading_api_key
ALPACA_SECRET_KEY=your_paper_trading_secret_key
ALPACA_PAPER=true

# Trading Configuration
DRY_RUN=false
USE_FUTURES=true
UPDATE_INTERVAL_MINUTES=5
TRADING_SYMBOLS=BTCUSD,ETHUSD,SOLUSD,ADAUSD,LTCUSD,XRPUSD
MAX_PORTFOLIO_ALLOCATION=0.80

# Risk Management
MAX_NET_EXPOSURE=0.30
MAX_GROSS_LEVERAGE=2.5
MAX_SINGLE_POSITION=0.10

# Notifications
ENABLE_EMAIL_NOTIFICATIONS=true
DIGEST_INTERVAL_HOURS=24

# Logging
LOG_LEVEL=INFO
ENABLE_HEALTH_CHECKS=true
```

## 🚀 Deployment Options

### Option 1: Direct Deployment (Recommended for Testing)
```bash
# Run the deployment script
python3 scripts/deploy_alpaca_paper.py
```

### Option 2: Systemd Service (Recommended for Production)
```bash
# Copy service file
sudo cp deployment/crypto-quant-bot.service /etc/systemd/system/

# Edit the service file with your credentials
sudo nano /etc/systemd/system/crypto-quant-bot.service

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable crypto-quant-bot
sudo systemctl start crypto-quant-bot

# Check status
sudo systemctl status crypto-quant-bot
```

### Option 3: Railway Deployment (Cloud)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy
railway up
```

## 📊 Monitoring and Management

### Check Bot Status
```bash
# If using systemd
sudo systemctl status crypto-quant-bot

# Check logs
sudo journalctl -u crypto-quant-bot -f

# Check bot logs
tail -f logs/trading_bot.log
```

### Bot Controls
```bash
# Start bot
sudo systemctl start crypto-quant-bot

# Stop bot
sudo systemctl stop crypto-quant-bot

# Restart bot
sudo systemctl restart crypto-quant-bot

# Enable auto-start on boot
sudo systemctl enable crypto-quant-bot
```

### Health Monitoring
```bash
# Check if bot is responding
curl http://localhost:8080/health

# Monitor trading activity
tail -f logs/trading_bot.log | grep "TRADE"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Alpaca Connection Failed
```bash
# Check credentials
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# Test connection manually
python3 scripts/test_alpaca.py
```

#### 2. Bot Not Starting
```bash
# Check Python dependencies
pip3 list | grep alpaca

# Check logs
tail -f logs/trading_bot.log

# Check system resources
htop
df -h
```

#### 3. Trading Not Working
```bash
# Check account status
python3 scripts/check_env.py

# Verify paper trading is enabled
python3 -c "import os; print('PAPER:', os.getenv('ALPACA_PAPER'))"
```

### Log Analysis
```bash
# View recent trades
grep "TRADE" logs/trading_bot.log | tail -20

# View errors
grep "ERROR" logs/trading_bot.log | tail -20

# View signal generation
grep "SIGNAL" logs/trading_bot.log | tail -20
```

## 📈 Performance Monitoring

### Daily Performance Check
```bash
# Generate daily report
python3 scripts/performance_report.py

# Check portfolio status
python3 scripts/check_env.py
```

### Weekly Performance Report
```bash
# Generate weekly report
python3 scripts/ytd_performance_report.py
```

## 🔒 Security Considerations

### API Key Security
- [ ] Use paper trading keys only
- [ ] Never commit API keys to git
- [ ] Use environment variables
- [ ] Regularly rotate keys

### System Security
- [ ] Keep system updated
- [ ] Use firewall
- [ ] Monitor system logs
- [ ] Regular backups

## 📞 Support

### Emergency Stop
```bash
# Immediate stop
sudo systemctl stop crypto-quant-bot

# Kill all Python processes (if needed)
pkill -f "run_live_bot.py"
```

### Contact Information
- GitHub Issues: [Repository Issues](https://github.com/yourusername/crypto-quant-EB/issues)
- Email: your-email@example.com

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Alpaca account created and verified
- [ ] Paper trading enabled
- [ ] API keys generated
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Environment configured

### Deployment
- [ ] Credentials configured
- [ ] Connection tested
- [ ] Bot started successfully
- [ ] Health checks passing
- [ ] Monitoring setup

### Post-Deployment
- [ ] First trade executed
- [ ] Notifications working
- [ ] Logs being generated
- [ ] Performance monitoring active
- [ ] Backup strategy implemented

## 🎯 Next Steps

1. **Monitor Performance**: Watch the bot's performance for the first week
2. **Adjust Parameters**: Fine-tune trading parameters based on results
3. **Scale Up**: Consider adding more trading pairs
4. **Live Trading**: When ready, transition to live trading (with real money)

## ⚠️ Important Notes

- **Paper Trading Only**: This deployment uses Alpaca paper trading (no real money)
- **Risk Management**: The bot includes built-in risk management features
- **24/7 Operation**: The bot will run continuously and restart automatically
- **Monitoring Required**: Regular monitoring is recommended
- **Backup Strategy**: Implement regular backups of configuration and logs

---

**🚀 Your crypto quant bot is now ready for 24/7 Alpaca paper trading!**
