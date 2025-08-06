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
        self.notification_email = os.getenv("EMAIL_TO", os.getenv("NOTIFICATION_EMAIL"))

        if not all([self.smtp_user, self.smtp_password, self.notification_email]):
            logger.warning("Email credentials not fully configured")

    async def send_trade_alert(
        self, symbol: str, action: str, price: float, size: float, reason: str
    ) -> bool:
        """Send trade execution alert."""
        subject = f"🤖 Trade Alert: {action.upper()} {symbol}"

        body = f"""
        Trade Executed:
        
        Symbol: {symbol}
        Action: {action.upper()}
        Price: ${price:,.2f}
        Size: {size:,.2f}
        Reason: {reason}
        Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
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


# Global instance
notifier = EmailNotifier()
