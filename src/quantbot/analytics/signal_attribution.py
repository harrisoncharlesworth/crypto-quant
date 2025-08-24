"""
Signal Attribution and Performance Analytics

Tracks performance by signal, sector, and time period for comprehensive attribution analysis.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SignalPerformance:
    """Performance metrics for individual signals."""

    signal_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    hit_rate: float = 0.0
    total_pnl: float = 0.0
    average_pnl_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_time_hours: float = 0.0
    last_signal_time: Optional[datetime] = None
    confidence_vs_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Individual trade record for attribution tracking."""

    timestamp: datetime
    symbol: str
    signal_name: str
    signal_strength: float
    signal_confidence: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    position_size: float = 0.0
    pnl: Optional[float] = None
    fees: float = 0.0
    funding_pnl: float = 0.0
    slippage: float = 0.0
    is_closed: bool = False


class SignalAttributionTracker:
    """
    Comprehensive signal attribution and performance tracking system.

    Tracks:
    - Performance by individual signal
    - Attribution by asset and sector
    - Time-based performance analysis
    - Risk-adjusted returns
    - Trade quality metrics
    """

    def __init__(self, max_history_days: int = 90):
        self.max_history_days = max_history_days
        self.trade_records: List[TradeRecord] = []
        self.signal_performance: Dict[str, SignalPerformance] = {}
        self.daily_pnl: Dict[str, Dict[str, float]] = defaultdict(
            dict
        )  # date -> signal -> pnl

        # Initialize performance tracking for known signals
        self.known_signals = [
            "time_series_momentum",
            "donchian_breakout",
            "short_term_mean_reversion",
            "oi_price_divergence",
            "delta_skew_whipsaw",
            "perp_funding_carry",
            "alt_btc_cross_sectional",
            "cash_carry_basis",
            "cross_exchange_funding",
            "options_vol_risk_premium",
            "stablecoin_supply_ratio",
            "mvrv_zscore",
        ]

        for signal in self.known_signals:
            self.signal_performance[signal] = SignalPerformance(signal_name=signal)

        logger.info(
            f"Signal Attribution Tracker initialized with {len(self.known_signals)} signals"
        )

    def record_trade_entry(
        self,
        symbol: str,
        signal_name: str,
        signal_strength: float,
        signal_confidence: float,
        entry_price: float,
        position_size: float,
    ) -> str:
        """Record a new trade entry."""
        trade_id = f"{symbol}_{signal_name}_{datetime.utcnow().timestamp()}"

        trade = TradeRecord(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            signal_name=signal_name,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            entry_price=entry_price,
            position_size=position_size,
        )

        self.trade_records.append(trade)

        # Update signal performance tracking
        if signal_name not in self.signal_performance:
            self.signal_performance[signal_name] = SignalPerformance(
                signal_name=signal_name
            )

        self.signal_performance[signal_name].last_signal_time = datetime.utcnow()

        logger.debug(f"Recorded trade entry: {signal_name} on {symbol}")
        return trade_id

    def record_trade_exit(
        self,
        symbol: str,
        signal_name: str,
        exit_price: float,
        fees: float = 0.0,
        funding_pnl: float = 0.0,
        slippage: float = 0.0,
    ) -> bool:
        """Record trade exit and calculate performance."""
        # Find the most recent open trade for this signal/symbol
        trade = None
        for t in reversed(self.trade_records):
            if t.symbol == symbol and t.signal_name == signal_name and not t.is_closed:
                trade = t
                break

        if not trade:
            logger.warning(f"No open trade found for {signal_name} on {symbol}")
            return False

        # Update trade record
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.utcnow()
        trade.fees = fees
        trade.funding_pnl = funding_pnl
        trade.slippage = slippage
        trade.is_closed = True

        # Calculate P&L
        price_pnl = (exit_price - trade.entry_price) * trade.position_size
        if trade.position_size < 0:  # Short position
            price_pnl = -price_pnl

        total_pnl = price_pnl + funding_pnl - fees - slippage
        trade.pnl = total_pnl

        # Update signal performance
        self._update_signal_performance(trade)

        # Update daily P&L tracking
        date_key = trade.exit_timestamp.date().isoformat()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = {}

        if signal_name not in self.daily_pnl[date_key]:
            self.daily_pnl[date_key][signal_name] = 0.0

        self.daily_pnl[date_key][signal_name] += total_pnl

        logger.debug(
            f"Recorded trade exit: {signal_name} on {symbol}, P&L: ${total_pnl:.2f}"
        )
        return True

    def _update_signal_performance(self, trade: TradeRecord):
        """Update performance metrics for a completed trade."""
        signal_perf = self.signal_performance[trade.signal_name]

        signal_perf.total_trades += 1
        signal_perf.total_pnl += trade.pnl or 0.0

        if trade.pnl > 0:
            signal_perf.winning_trades += 1
        else:
            signal_perf.losing_trades += 1

        signal_perf.hit_rate = (
            signal_perf.winning_trades / signal_perf.total_trades
            if signal_perf.total_trades > 0
            else 0.0
        )

        signal_perf.average_pnl_per_trade = (
            signal_perf.total_pnl / signal_perf.total_trades
            if signal_perf.total_trades > 0
            else 0.0
        )

        # Calculate hold time
        if trade.exit_timestamp and trade.timestamp:
            hold_time = (trade.exit_timestamp - trade.timestamp).total_seconds() / 3600
            current_avg = signal_perf.avg_hold_time_hours * (
                signal_perf.total_trades - 1
            )
            signal_perf.avg_hold_time_hours = (
                current_avg + hold_time
            ) / signal_perf.total_trades

        # Track confidence vs performance
        confidence_bucket = (
            f"{int(trade.signal_confidence * 10) * 10}%"  # 0%, 10%, 20%, etc.
        )
        if confidence_bucket not in signal_perf.confidence_vs_performance:
            signal_perf.confidence_vs_performance[confidence_bucket] = 0.0
        signal_perf.confidence_vs_performance[confidence_bucket] += trade.pnl or 0.0

    def get_signal_attribution_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive signal attribution report."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Filter recent trades
        recent_trades = [
            t for t in self.trade_records if t.timestamp >= cutoff_date and t.is_closed
        ]

        # Calculate attribution by signal
        signal_attribution = {}
        for signal_name in self.known_signals:
            signal_trades = [t for t in recent_trades if t.signal_name == signal_name]

            if not signal_trades:
                signal_attribution[signal_name] = {
                    "total_pnl": 0.0,
                    "trade_count": 0,
                    "hit_rate": 0.0,
                    "avg_pnl": 0.0,
                    "contribution_pct": 0.0,
                }
                continue

            total_pnl = sum(t.pnl for t in signal_trades if t.pnl is not None)
            winning_trades = sum(1 for t in signal_trades if t.pnl and t.pnl > 0)
            hit_rate = winning_trades / len(signal_trades) if signal_trades else 0.0
            avg_pnl = total_pnl / len(signal_trades) if signal_trades else 0.0

            signal_attribution[signal_name] = {
                "total_pnl": total_pnl,
                "trade_count": len(signal_trades),
                "hit_rate": hit_rate,
                "avg_pnl": avg_pnl,
                "contribution_pct": 0.0,  # Will calculate after total
            }

        # Calculate contribution percentages
        total_portfolio_pnl = sum(
            attr["total_pnl"] for attr in signal_attribution.values()
        )
        if total_portfolio_pnl != 0:
            for attr in signal_attribution.values():
                attr["contribution_pct"] = (
                    attr["total_pnl"] / total_portfolio_pnl
                ) * 100

        # Top contributors and detractors
        sorted_signals = sorted(
            signal_attribution.items(), key=lambda x: x[1]["total_pnl"], reverse=True
        )

        top_contributors = sorted_signals[:5]
        top_detractors = sorted_signals[-5:] if len(sorted_signals) > 5 else []

        return {
            "period_days": days,
            "total_trades": len(recent_trades),
            "total_pnl": total_portfolio_pnl,
            "signal_attribution": signal_attribution,
            "top_contributors": [{"signal": s[0], **s[1]} for s in top_contributors],
            "top_detractors": [{"signal": s[0], **s[1]} for s in top_detractors],
            "overall_hit_rate": (
                sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
                / len(recent_trades)
                if recent_trades
                else 0.0
            ),
        }

    def get_sector_attribution(self, days: int = 30) -> Dict[str, float]:
        """Calculate P&L attribution by crypto sectors."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_trades = [
            t for t in self.trade_records if t.timestamp >= cutoff_date and t.is_closed
        ]

        # Sector mapping (simplified)
        sector_map = {
            "BTC": "store_of_value",
            "ETH": "smart_contract",
            "BNB": "exchange",
            "SOL": "layer1",
            "ADA": "layer1",
            "DOT": "interoperability",
            "AVAX": "layer1",
            "LINK": "oracle",
            "UNI": "defi",
            "DOGE": "meme",
            "SHIB": "meme",
            "MATIC": "layer2",
            "OP": "layer2",
            "ARB": "layer2",
        }

        sector_pnl = defaultdict(float)

        for trade in recent_trades:
            base_symbol = trade.symbol.replace("USD", "").replace("USDT", "")
            sector = sector_map.get(base_symbol, "other")
            sector_pnl[sector] += trade.pnl or 0.0

        return dict(sector_pnl)

    def get_trade_quality_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate trade execution quality metrics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_trades = [
            t for t in self.trade_records if t.timestamp >= cutoff_date and t.is_closed
        ]

        if not recent_trades:
            return {"no_data": True}

        total_fees = sum(t.fees for t in recent_trades)
        total_slippage = sum(t.slippage for t in recent_trades)
        total_notional = sum(
            abs(t.position_size * t.entry_price) for t in recent_trades
        )

        avg_slippage_bps = (
            (total_slippage / total_notional * 10000) if total_notional > 0 else 0
        )
        fee_rate_bps = (
            (total_fees / total_notional * 10000) if total_notional > 0 else 0
        )

        # Calculate average hold times by signal
        hold_times_by_signal = defaultdict(list)
        for trade in recent_trades:
            if trade.exit_timestamp:
                hold_time = (
                    trade.exit_timestamp - trade.timestamp
                ).total_seconds() / 3600
                hold_times_by_signal[trade.signal_name].append(hold_time)

        avg_hold_times = {
            signal: np.mean(times) for signal, times in hold_times_by_signal.items()
        }

        return {
            "total_trades": len(recent_trades),
            "avg_slippage_bps": avg_slippage_bps,
            "fee_rate_bps": fee_rate_bps,
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "avg_hold_times": avg_hold_times,
            "pnl_per_dollar_traded": (
                sum(t.pnl or 0 for t in recent_trades) / total_notional
                if total_notional > 0
                else 0.0
            ),
        }

    def cleanup_old_records(self):
        """Remove old trade records to manage memory."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.max_history_days)

        initial_count = len(self.trade_records)
        self.trade_records = [
            t for t in self.trade_records if t.timestamp >= cutoff_date
        ]

        # Clean up daily P&L data
        cutoff_date_str = cutoff_date.date().isoformat()
        old_dates = [date for date in self.daily_pnl.keys() if date < cutoff_date_str]
        for date in old_dates:
            del self.daily_pnl[date]

        removed_count = initial_count - len(self.trade_records)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old trade records")
