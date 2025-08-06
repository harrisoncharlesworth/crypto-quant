# 🚂 Railway Deployment Guide

## Quick Deploy to Railway

### 1. **Push to GitHub**
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 2. **Deploy on Railway**
1. Go to [railway.app](https://railway.app)
2. Sign up/in with GitHub
3. Click "Deploy from GitHub repo"
4. Select your `crypto-quant` repository
5. Railway will auto-detect and deploy

### 3. **Set Environment Variables**
In Railway dashboard → Variables tab, add:

```bash
# Binance API (use your existing values)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_SECRET=your_testnet_secret_here
BINANCE_SANDBOX=true

# Email (use your existing values)
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Trading Configuration
DRY_RUN=true
MAX_POSITION_SIZE=100
UPDATE_INTERVAL_MINUTES=15
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
RISK_LIMIT=0.02
```

### 4. **Deploy and Monitor**
- Railway will automatically build and deploy
- Check logs in Railway dashboard
- You'll receive email notifications when bot starts

## 🔧 **Configuration Options**

| Variable | Description | Default |
|----------|-------------|---------|
| `DRY_RUN` | Paper trading mode | `true` |
| `UPDATE_INTERVAL_MINUTES` | Signal generation frequency | `15` |
| `TRADING_SYMBOLS` | Comma-separated symbols | `BTCUSDT,ETHUSDT` |
| `MAX_POSITION_SIZE` | Max USD per trade | `100` |
| `RISK_LIMIT` | Max risk per trade (%) | `0.02` |

## 📊 **Monitoring**

**Email Notifications:**
- ✅ Bot startup/shutdown
- 📈 Trade signals (every 15 min)
- ⚠️ Risk alerts and errors
- 📊 Daily performance summary

**Railway Logs:**
- Real-time trading decisions
- Signal generation details
- System health monitoring

## 🚀 **Production Checklist**

### **Before Going Live:**
- [ ] Test deployment with `DRY_RUN=true`
- [ ] Verify email notifications work
- [ ] Monitor for 24-48 hours
- [ ] Check all signals generating correctly

### **Going Live:**
- [ ] Set `DRY_RUN=false`
- [ ] Use real Binance API keys (not testnet)
- [ ] Set `BINANCE_SANDBOX=false`
- [ ] Start with small `MAX_POSITION_SIZE`
- [ ] Monitor closely first week

## 💰 **Railway Costs**

**Estimated Monthly Cost:** $5-10
- Hobby plan: $5/month
- Pro plan: $20/month (if you need more resources)
- No usage charges for compute time

**Cost Efficiency:**
- Runs 24/7 automatically
- No server management needed
- Scales automatically
- Much cheaper than VPS hosting

## 🛠️ **Troubleshooting**

**Common Issues:**
1. **Build fails**: Check requirements-railway.txt
2. **Bot doesn't start**: Check environment variables
3. **No emails**: Verify EMAIL_* variables
4. **API errors**: Check Binance API permissions

**Log Commands:**
```bash
# View logs in Railway dashboard or via CLI
railway logs
```

## 🔄 **Updates**

To update your bot:
```bash
git add .
git commit -m "Update trading logic"
git push origin main
```

Railway will automatically redeploy with zero downtime.
