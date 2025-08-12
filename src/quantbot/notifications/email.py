import os
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Cost-efficient email notification system using SMTP."""

    def __init__(self):
        # Support both old and new variable names for compatibility
        self.smtp_host = os.getenv("SMTP_SERVER", os.getenv("SMTP_HOST", "smtp.gmail.com"))
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("EMAIL_FROM", os.getenv("SMTP_USER"))
        self.smtp_password = os.getenv("EMAIL_PASSWORD", os.getenv("SMTP_PASSWORD"))
        self.notification_email = os.getenv("EMAIL_RECIPIENT", os.getenv("EMAIL_TO", os.getenv("NOTIFICATION_EMAIL")))

        if not all([self.smtp_user, self.smtp_password, self.notification_email]):
            logger.warning("Email credentials not fully configured")

    async def send_trade_alert(
        self, symbol: str, action: str, price: float, size: float, reason: str
    ) -> bool:
        """Send comprehensive trade execution alert."""
        # Determine trade type and emoji
        trade_emoji = "🚀" if action.lower() == "buy" else "📉" if action.lower() == "sell" else "⚡"
        action_upper = action.upper()
        
        subject = f"{trade_emoji} Trade Executed: {action_upper} {symbol} @ ${price:,.2f}"

        # Calculate position value
        position_value = size * price
        
        # Get current time in AEST
        from datetime import timezone, timedelta
        aest_tz = timezone(timedelta(hours=10))  # AEST is UTC+10
        current_time = datetime.now(aest_tz).strftime('%Y-%m-%d %H:%M:%S AEST')

        body = f"""
🚀 CRYPTO QUANT BOT - TRADE ALERT

{trade_emoji} TRADE EXECUTED
{'='*50}

📊 Trade Details:
   Symbol: {symbol}
   Action: {action_upper}
   Price: ${price:,.2f}
   Size: {size:,.4f}
   Position Value: ${position_value:,.2f}
   Time: {current_time}

🎯 Signal Information:
   {reason}

💰 Portfolio Impact:
   • Trade Type: {'LONG' if action.lower() == 'buy' else 'SHORT'} Position
   • Position Size: {size:,.4f} {symbol}
   • Market Value: ${position_value:,.2f}

📈 Risk Management:
   • Conservative Settings Active
   • Enhanced Risk Limits Applied
   • Position Sizing: Kelly Optimal
   • Max Exposure: 20%

🔔 Next Steps:
   • Monitor position performance
   • Check daily report at 6:00 PM AEST
   • Review risk metrics in daily summary

---
🤖 Crypto Quant Bot Trading System
📧 Daily Reports: 6:00 PM AEST
🌐 Railway Cloud Deployment
"""

        return await self.send_email(subject, body)

    async def send_enhanced_trade_alert(
        self, 
        symbol: str, 
        action: str, 
        price: float, 
        size: float, 
        reason: str,
        confidence: float = 0.0,
        signal_strength: float = 0.0,
        account_balance: float = 0.0
    ) -> bool:
        """Send enhanced trade alert with additional context."""
        trade_emoji = "🚀" if action.lower() == "buy" else "📉" if action.lower() == "sell" else "⚡"
        action_upper = action.upper()
        
        subject = f"{trade_emoji} LIVE TRADE: {action_upper} {symbol} @ ${price:,.2f}"
        
        # Calculate metrics
        position_value = size * price
        portfolio_percentage = (position_value / account_balance * 100) if account_balance > 0 else 0
        
        # Get current time in AEST
        from datetime import timezone, timedelta
        aest_tz = timezone(timedelta(hours=10))
        current_time = datetime.now(aest_tz).strftime('%Y-%m-%d %H:%M:%S AEST')
        
        # Determine confidence level
        confidence_level = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
        confidence_emoji = "🔥" if confidence > 0.7 else "⚡" if confidence > 0.4 else "💡"

        body = f"""
🚀 CRYPTO QUANT BOT - LIVE TRADE ALERT

{trade_emoji} TRADE EXECUTED SUCCESSFULLY
{'='*60}

📊 TRADE DETAILS:
   Symbol: {symbol}
   Action: {action_upper}
   Price: ${price:,.2f}
   Size: {size:,.4f}
   Position Value: ${position_value:,.2f}
   Portfolio %: {portfolio_percentage:.2f}%
   Time: {current_time}

🎯 SIGNAL ANALYSIS:
   {reason}
   Signal Strength: {signal_strength:.3f}
   {confidence_emoji} Confidence: {confidence:.1%} ({confidence_level})

💰 PORTFOLIO STATUS:
   Account Balance: ${account_balance:,.2f}
   Position Value: ${position_value:,.2f}
   Available Capital: ${(account_balance - position_value):,.2f}

📈 RISK MANAGEMENT:
   ✅ Conservative Settings Active
   ✅ Enhanced Risk Limits (20% max exposure)
   ✅ Kelly Optimal Position Sizing
   ✅ Daily Drawdown Protection (15%)
   ✅ Real-time Monitoring Enabled

🔔 IMMEDIATE ACTIONS:
   • Trade executed successfully
   • Position monitoring active
   • Risk limits maintained
   • Daily report scheduled for 6:00 PM AEST

📧 NOTIFICATIONS:
   • Trade alerts: ✅ ENABLED
   • Daily reports: ✅ 6:00 PM AEST
   • Risk alerts: ✅ ENABLED
   • Performance tracking: ✅ ACTIVE

---
🤖 Crypto Quant Bot Trading System
📧 Email: ebullemor@gmail.com
🌐 Railway Cloud Deployment
⏰ AEST Timezone (UTC+10)
"""

        return await self.send_email(subject, body)

    async def send_risk_alert(self, message: str, severity: str = "WARNING") -> bool:
        """Send risk management alert."""
        subject = f"⚠️ Risk Alert: {severity}"

        body = f"""
        Risk Alert:
        
        Severity: {severity}
        Message: {message}
        Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        Please review your positions and risk parameters.
        """

        return await self.send_email(subject, body)

    async def send_daily_summary(
        self, pnl: float, trades: int, signals: List[str]
    ) -> bool:
        """Send daily trading summary."""
        subject = f"📊 Daily Summary - PnL: ${pnl:+.2f}"

        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"

        body = f"""
        Daily Trading Summary:
        
        {pnl_emoji} P&L: ${pnl:+,.2f}
        📊 Trades Executed: {trades}
        🎯 Active Signals: {len(signals)}
        
        Signal Activity:
        {chr(10).join(f'  • {signal}' for signal in signals)}
        
        Date: {datetime.utcnow().strftime('%Y-%m-%d')}
        """

        return await self.send_email(subject, body)

    async def send_digest(
        self, 
        account_equity: float,
        realised_pnl: float, 
        unrealised_pnl: float,
        open_positions: List[dict],
        recent_signals: List[str],
        period: str = "8-hour"
    ) -> bool:
        """Send rich digest email with comprehensive portfolio status."""
        total_pnl = realised_pnl + unrealised_pnl
        pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➡️"
        
        subject = f"📬 {period.title()} Digest - Total PnL: ${total_pnl:+.2f}"
        
        # Create HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
                .content {{ padding: 20px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 0.9em; color: #6c757d; margin-top: 5px; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .neutral {{ color: #6c757d; }}
                .positions-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .positions-table th, .positions-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e9ecef; }}
                .positions-table th {{ background-color: #f8f9fa; font-weight: 600; }}
                .signals-list {{ background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 15px 0; }}
                .footer {{ background: #f8f9fa; padding: 15px; text-align: center; font-size: 0.9em; color: #6c757d; border-radius: 0 0 8px 8px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Crypto Quant Bot</h1>
                    <h2>{period.title()} Portfolio Digest</h2>
                    <p>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>
                
                <div class="content">
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">${account_equity:,.2f}</div>
                            <div class="metric-label">💰 Account Equity</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if realised_pnl >= 0 else 'negative'}">${realised_pnl:+,.2f}</div>
                            <div class="metric-label">💸 Realised P&L</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if unrealised_pnl >= 0 else 'negative'}">${unrealised_pnl:+,.2f}</div>
                            <div class="metric-label">📊 Unrealised P&L</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {'positive' if total_pnl >= 0 else 'negative'}">{pnl_emoji} ${total_pnl:+,.2f}</div>
                            <div class="metric-label">🎯 Total P&L</div>
                        </div>
                    </div>

                    <h3>📈 Open Positions</h3>
                    {self._format_positions_table(open_positions)}

                    <h3>🎯 Recent Signals</h3>
                    <div class="signals-list">
                        {self._format_signals_list(recent_signals)}
                    </div>
                    
                    <h3>📊 Market Analysis</h3>
                    <p><strong>Active Trading Pairs:</strong> BTC, ETH, BNB, SOL, ADA, LTC, MATIC, XRP</p>
                    <p><strong>Risk Management:</strong> Portfolio Blender V2 with risk parity allocation</p>
                    <p><strong>Signal Count:</strong> 12 evidence-based quantitative signals</p>
                </div>
                
                <div class="footer">
                    <p>🚀 Powered by Crypto Quant Bot | Running on Railway</p>
                    <p>Next digest in 8 hours</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return await self.send_html_email(subject, html_body)

    def _format_positions_table(self, positions: List[dict]) -> str:
        """Format positions as HTML table."""
        if not positions:
            return "<p>No open positions</p>"
        
        table_rows = ""
        for pos in positions:
            side_class = "positive" if pos.get('side', '').upper() == 'LONG' else "negative"
            pnl_class = "positive" if pos.get('unrealised_pnl', 0) >= 0 else "negative"
            
            table_rows += f"""
            <tr>
                <td><strong>{pos.get('symbol', 'N/A')}</strong></td>
                <td><span class="{side_class}">{pos.get('side', 'N/A').upper()}</span></td>
                <td>${pos.get('size', 0):,.2f}</td>
                <td>${pos.get('entry_price', 0):,.2f}</td>
                <td>${pos.get('current_price', 0):,.2f}</td>
                <td><span class="{pnl_class}">${pos.get('unrealised_pnl', 0):+,.2f}</span></td>
            </tr>
            """
        
        return f"""
        <table class="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """

    def _format_signals_list(self, signals: List[str]) -> str:
        """Format signals as HTML list."""
        if not signals:
            return "<p>No recent signals</p>"
        
        recent_signals = signals[-10:]  # Last 10 signals
        signal_items = "".join([f"<li>{signal}</li>" for signal in recent_signals])
        
        return f"<ul>{signal_items}</ul>"

    async def send_signal_alert(
        self, symbol: str, signal_name: str, strength: float, confidence: float
    ) -> bool:
        """Send signal generation alert."""
        subject = f"🎯 Signal: {signal_name} on {symbol}"

        direction = "LONG" if strength > 0 else "SHORT"
        strength_emoji = (
            "🔥" if abs(strength) > 0.7 else "⚡" if abs(strength) > 0.3 else "💡"
        )

        body = f"""
        Signal Generated:
        
        {strength_emoji} Signal: {signal_name}
        📈 Symbol: {symbol}
        🎯 Direction: {direction}
        💪 Strength: {strength:.2f}
        🎯 Confidence: {confidence:.1%}
        Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        return await self.send_email(subject, body)

    async def send_email(
        self, subject: str, body: str, to_email: Optional[str] = None
    ) -> bool:
        """Send email via SMTP."""
        if not self.smtp_user or not self.smtp_password:
            logger.error("SMTP credentials not configured")
            return False

        to_email = to_email or self.notification_email
        if not to_email:
            logger.error("No recipient email configured")
            return False

        try:
            message = MIMEMultipart()
            message["From"] = self.smtp_user
            message["To"] = to_email
            message["Subject"] = subject

            message.attach(MIMEText(body, "plain"))

            async with aiosmtplib.SMTP(
                hostname=self.smtp_host, port=self.smtp_port, start_tls=True
            ) as server:
                await server.login(self.smtp_user, self.smtp_password)
                await server.send_message(message)

            logger.info(f"Email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def send_html_email(
        self, subject: str, html_body: str, to_email: Optional[str] = None
    ) -> bool:
        """Send HTML email via SMTP."""
        if not self.smtp_user or not self.smtp_password:
            logger.error("SMTP credentials not configured")
            return False

        to_email = to_email or self.notification_email
        if not to_email:
            logger.error("No recipient email configured")
            return False

        try:
            message = MIMEMultipart("alternative")
            message["From"] = self.smtp_user
            message["To"] = to_email
            message["Subject"] = subject

            # Create both plain text and HTML versions
            text_body = self._html_to_text(html_body)
            text_part = MIMEText(text_body, "plain")
            html_part = MIMEText(html_body, "html")

            message.attach(text_part)
            message.attach(html_part)

            async with aiosmtplib.SMTP(
                hostname=self.smtp_host, port=self.smtp_port, start_tls=True
            ) as server:
                await server.login(self.smtp_user, self.smtp_password)
                await server.send_message(message)

            logger.info(f"HTML email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send HTML email: {e}")
            return False

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text for fallback."""
        import re
        # Simple HTML to text conversion
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# Global instance
notifier = EmailNotifier()
