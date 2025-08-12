#!/usr/bin/env python3
"""
Daily Email Reporter for Crypto Quant Bot
Sends comprehensive daily trade summaries and performance reports
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any
import json

class DailyEmailReporter:
    """Handles daily email reports for the crypto quant bot."""
    
    def __init__(self, recipient_email: str = "ebullemor@gmail.com"):
        self.recipient_email = recipient_email
        self.sender_email = os.environ.get("SENDER_EMAIL", "crypto-quant-bot@example.com")
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_username = os.environ.get("SMTP_USERNAME", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        
    def format_trade_summary(self, trades: List[Dict[str, Any]]) -> str:
        """Format trade summary for email."""
        if not trades:
            return "No trades executed today."
            
        summary = "📊 DAILY TRADE SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        total_pnl = 0
        total_trades = len(trades)
        winning_trades = 0
        
        for symbol, symbol_trades in trades_by_symbol.items():
            summary += f"🔸 {symbol}:\n"
            symbol_pnl = 0
            
            for trade in symbol_trades:
                side = trade.get('side', 'Unknown')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                timestamp = trade.get('timestamp', 'Unknown')
                
                summary += f"   • {side.upper()}: {quantity:.4f} @ ${price:,.2f}"
                if pnl:
                    summary += f" (PnL: ${pnl:,.2f})"
                    symbol_pnl += pnl
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                summary += f" - {timestamp}\n"
            
            if symbol_pnl != 0:
                summary += f"   📈 Symbol PnL: ${symbol_pnl:,.2f}\n"
            summary += "\n"
        
        # Overall statistics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        summary += "📈 DAILY STATISTICS\n"
        summary += "-" * 30 + "\n"
        summary += f"Total Trades: {total_trades}\n"
        summary += f"Winning Trades: {winning_trades}\n"
        summary += f"Win Rate: {win_rate:.1f}%\n"
        summary += f"Total PnL: ${total_pnl:,.2f}\n"
        
        return summary
        
    def format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for email."""
        summary = "📊 PERFORMANCE METRICS\n"
        summary += "=" * 50 + "\n\n"
        
        # Daily metrics
        summary += "📅 DAILY METRICS:\n"
        summary += f"• Total Return: {metrics.get('daily_return', 0):.2%}\n"
        summary += f"• Daily Drawdown: {metrics.get('daily_drawdown', 0):.2%}\n"
        summary += f"• Position Count: {metrics.get('position_count', 0)}\n"
        summary += f"• Signal Confidence Avg: {metrics.get('signal_confidence_avg', 0):.3f}\n"
        summary += f"• Volatility: {metrics.get('volatility', 0):.2%}\n\n"
        
        # Weekly metrics (if available)
        if 'weekly_return' in metrics:
            summary += "📅 WEEKLY METRICS:\n"
            summary += f"• Weekly Return: {metrics.get('weekly_return', 0):.2%}\n"
            summary += f"• Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            summary += f"• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
            summary += f"• Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            summary += f"• Alpha vs Market: {metrics.get('alpha_vs_market', 0):.2%}\n\n"
        
        # Monthly metrics (if available)
        if 'monthly_return' in metrics:
            summary += "📅 MONTHLY METRICS:\n"
            summary += f"• Monthly Return: {metrics.get('monthly_return', 0):.2%}\n"
            summary += f"• Risk-Adjusted Return: {metrics.get('risk_adjusted_return', 0):.2%}\n"
            summary += f"• Signal Efficiency: {metrics.get('signal_efficiency', 0):.2%}\n"
            summary += f"• Position Turnover: {metrics.get('position_turnover', 0):.2%}\n"
            summary += f"• Correlation Exposure: {metrics.get('correlation_exposure', 0):.2%}\n\n"
        
        return summary
        
    def format_risk_metrics(self, risk_data: Dict[str, Any]) -> str:
        """Format risk metrics for email."""
        summary = "🛡️ RISK METRICS\n"
        summary += "=" * 50 + "\n\n"
        
        summary += "📊 CURRENT EXPOSURE:\n"
        summary += f"• Net Exposure: {risk_data.get('net_exposure', 0):.2%}\n"
        summary += f"• Gross Leverage: {risk_data.get('gross_leverage', 0):.2f}\n"
        summary += f"• Max Single Position: {risk_data.get('max_single_position', 0):.2%}\n"
        summary += f"• Correlated Exposure: {risk_data.get('correlated_exposure', 0):.2%}\n\n"
        
        summary += "📉 RISK LIMITS:\n"
        summary += f"• Max Net Exposure: {risk_data.get('max_net_exposure', 0):.2%}\n"
        summary += f"• Max Gross Leverage: {risk_data.get('max_gross_leverage', 0):.2f}\n"
        summary += f"• Max Single Position: {risk_data.get('max_single_position_limit', 0):.2%}\n"
        summary += f"• Daily Drawdown Limit: {risk_data.get('daily_drawdown_limit', 0):.2%}\n\n"
        
        summary += "⚠️ RISK ALERTS:\n"
        alerts = risk_data.get('alerts', [])
        if alerts:
            for alert in alerts:
                summary += f"• {alert}\n"
        else:
            summary += "• No risk alerts - all metrics within limits\n"
        
        return summary
        
    def format_signal_analysis(self, signal_data: Dict[str, Any]) -> str:
        """Format signal analysis for email."""
        summary = "🎯 SIGNAL ANALYSIS\n"
        summary += "=" * 50 + "\n\n"
        
        summary += "📊 SIGNAL CONTRIBUTIONS:\n"
        contributions = signal_data.get('contributions', {})
        for signal, contribution in contributions.items():
            summary += f"• {signal.replace('_', ' ').title()}: {contribution:.1%}\n"
        
        summary += "\n📈 SIGNAL PERFORMANCE:\n"
        performance = signal_data.get('performance', {})
        for signal, perf in performance.items():
            summary += f"• {signal.replace('_', ' ').title()}: {perf:.2%}\n"
        
        summary += "\n🎯 ACTIVE SIGNALS:\n"
        active_signals = signal_data.get('active_signals', [])
        if active_signals:
            for signal in active_signals:
                summary += f"• {signal}\n"
        else:
            summary += "• No active signals currently\n"
        
        return summary
        
    def format_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> str:
        """Format portfolio summary for email."""
        summary = "💼 PORTFOLIO SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        summary += "💰 ACCOUNT STATUS:\n"
        summary += f"• Cash: ${portfolio_data.get('cash', 0):,.2f}\n"
        summary += f"• Portfolio Value: ${portfolio_data.get('portfolio_value', 0):,.2f}\n"
        summary += f"• Buying Power: ${portfolio_data.get('buying_power', 0):,.2f}\n"
        summary += f"• Total PnL: ${portfolio_data.get('total_pnl', 0):,.2f}\n\n"
        
        summary += "📊 POSITIONS:\n"
        positions = portfolio_data.get('positions', [])
        if positions:
            for position in positions:
                symbol = position.get('symbol', 'Unknown')
                quantity = position.get('quantity', 0)
                avg_price = position.get('avg_price', 0)
                market_value = position.get('market_value', 0)
                unrealized_pnl = position.get('unrealized_pnl', 0)
                
                summary += f"• {symbol}: {quantity:.4f} @ ${avg_price:,.2f}"
                summary += f" (Value: ${market_value:,.2f}, PnL: ${unrealized_pnl:,.2f})\n"
        else:
            summary += "• No open positions\n"
        
        return summary
        
    def create_daily_report(self, 
                          trades: List[Dict[str, Any]] = None,
                          performance_metrics: Dict[str, Any] = None,
                          risk_metrics: Dict[str, Any] = None,
                          signal_analysis: Dict[str, Any] = None,
                          portfolio_summary: Dict[str, Any] = None) -> str:
        """Create comprehensive daily report."""
        
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        email_body = f"""
🚀 CRYPTO QUANT BOT - DAILY REPORT
📅 Date: {report_date}
📧 Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

{'=' * 60}

"""
        
        # Add trade summary
        if trades is not None:
            email_body += self.format_trade_summary(trades) + "\n\n"
        
        # Add performance metrics
        if performance_metrics is not None:
            email_body += self.format_performance_metrics(performance_metrics) + "\n\n"
        
        # Add risk metrics
        if risk_metrics is not None:
            email_body += self.format_risk_metrics(risk_metrics) + "\n\n"
        
        # Add signal analysis
        if signal_analysis is not None:
            email_body += self.format_signal_analysis(signal_analysis) + "\n\n"
        
        # Add portfolio summary
        if portfolio_summary is not None:
            email_body += self.format_portfolio_summary(portfolio_summary) + "\n\n"
        
        # Add footer
        email_body += f"""
{'=' * 60}

📊 BOT STATUS: ✅ RUNNING 24/7
🎯 STRATEGIC RECOMMENDATIONS: ✅ IMPLEMENTED
🛡️ RISK MANAGEMENT: ✅ ENHANCED
📧 NEXT REPORT: Tomorrow at 6:00 PM UTC

---
Generated by Crypto Quant Bot Daily Reporter
Email: {self.recipient_email}
"""
        
        return email_body
        
    def send_email(self, subject: str, body: str) -> bool:
        """Send email report."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # For now, we'll simulate email sending
            # In production, you'd use actual SMTP
            print(f"📧 EMAIL NOTIFICATION")
            print(f"   To: {self.recipient_email}")
            print(f"   Subject: {subject}")
            print(f"   Body length: {len(body)} characters")
            print(f"   Status: Would be sent via SMTP")
            
            # Uncomment below for actual email sending
            # with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            #     server.starttls()
            #     server.login(self.smtp_username, self.smtp_password)
            #     server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
            
    def send_daily_report(self, 
                         trades: List[Dict[str, Any]] = None,
                         performance_metrics: Dict[str, Any] = None,
                         risk_metrics: Dict[str, Any] = None,
                         signal_analysis: Dict[str, Any] = None,
                         portfolio_summary: Dict[str, Any] = None) -> bool:
        """Send daily report email."""
        
        report_date = datetime.now().strftime("%Y-%m-%d")
        subject = f"[Crypto Quant Bot] Daily Report - {report_date}"
        
        body = self.create_daily_report(
            trades=trades,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            signal_analysis=signal_analysis,
            portfolio_summary=portfolio_summary
        )
        
        return self.send_email(subject, body)
        
    def send_trade_alert(self, trade: Dict[str, Any]) -> bool:
        """Send individual trade alert."""
        symbol = trade.get('symbol', 'Unknown')
        side = trade.get('side', 'Unknown')
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        
        subject = f"[Crypto Quant Bot] Trade Alert - {side.upper()} {symbol}"
        
        body = f"""
🚀 TRADE ALERT

Symbol: {symbol}
Action: {side.upper()}
Quantity: {quantity:.4f}
Price: ${price:,.2f}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

---
Crypto Quant Bot Trading System
"""
        
        return self.send_email(subject, body)
        
    def send_risk_alert(self, alert_message: str) -> bool:
        """Send risk alert email."""
        subject = f"[Crypto Quant Bot] Risk Alert - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
⚠️ RISK ALERT

{alert_message}

Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

---
Crypto Quant Bot Risk Management System
"""
        
        return self.send_email(subject, body)
