"""
Market Conditions and Volatility Dashboard

Fetches real-time market conditions, volatility metrics, and funding data for comprehensive market context.
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MarketConditions:
    """Market conditions snapshot."""
    timestamp: datetime
    btc_price: float = 0.0
    eth_price: float = 0.0
    btc_24h_change: float = 0.0
    eth_24h_change: float = 0.0
    
    # Volatility metrics
    btc_realized_vol_7d: float = 0.0
    eth_realized_vol_7d: float = 0.0
    btc_implied_vol: float = 0.0
    eth_implied_vol: float = 0.0
    vol_risk_premium_btc: float = 0.0
    vol_risk_premium_eth: float = 0.0
    
    # Funding and basis
    btc_funding_rate: float = 0.0
    eth_funding_rate: float = 0.0
    btc_perp_basis: float = 0.0
    eth_perp_basis: float = 0.0
    
    # Market breadth
    crypto_fear_greed_index: int = 50
    btc_dominance: float = 0.0
    total_market_cap: float = 0.0
    market_cap_24h_change: float = 0.0
    coins_above_ma20: float = 0.0
    
    # Liquidity metrics
    btc_orderbook_depth_1pct: float = 0.0
    eth_orderbook_depth_1pct: float = 0.0
    
    # Macro correlations (30-day rolling)
    btc_spx_correlation: float = 0.0
    btc_dxy_correlation: float = 0.0
    
    # Market session and regime
    market_session: str = "unknown"
    volatility_regime: str = "normal"  # low, normal, high, extreme
    funding_environment: str = "neutral"  # bullish, neutral, bearish


class MarketConditionsProvider:
    """
    Fetches and provides comprehensive market conditions data.
    
    Uses multiple data sources for redundancy:
    - CoinGecko for basic price and market data
    - Binance for funding rates and orderbook depth  
    - Fear & Greed Index API
    - Optional: External macro data sources
    """
    
    def __init__(self):
        self.cache_duration_minutes = 15  # Cache market data for 15 minutes
        self.last_update: Optional[datetime] = None
        self.cached_conditions: Optional[MarketConditions] = None
        
        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.binance_base = "https://api.binance.com/api/v3"
        self.fear_greed_api = "https://api.alternative.me/fng/"
        
        logger.info("Market Conditions Provider initialized")
    
    async def get_market_conditions(self, force_refresh: bool = False) -> MarketConditions:
        """Get current market conditions with caching."""
        now = datetime.utcnow()
        
        # Check cache
        if (not force_refresh and 
            self.cached_conditions and 
            self.last_update and
            (now - self.last_update).total_seconds() < self.cache_duration_minutes * 60):
            return self.cached_conditions
        
        try:
            conditions = await self._fetch_market_conditions()
            self.cached_conditions = conditions
            self.last_update = now
            return conditions
        except Exception as e:
            logger.error(f"Failed to fetch market conditions: {e}")
            # Return cached data if available, otherwise empty conditions
            if self.cached_conditions:
                return self.cached_conditions
            return MarketConditions(timestamp=now)
    
    async def _fetch_market_conditions(self) -> MarketConditions:
        """Fetch all market conditions data from various sources."""
        conditions = MarketConditions(timestamp=datetime.utcnow())
        
        # Fetch data from multiple sources concurrently
        tasks = [
            self._fetch_coingecko_data(conditions),
            self._fetch_binance_data(conditions),
            self._fetch_fear_greed_data(conditions),
            self._calculate_volatility_metrics(conditions),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Determine market regime
        self._determine_market_regime(conditions)
        
        return conditions
    
    async def _fetch_coingecko_data(self, conditions: MarketConditions):
        """Fetch basic market data from CoinGecko."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get BTC and ETH prices with 24h change
                url = f"{self.coingecko_base}/simple/price"
                params = {
                    'ids': 'bitcoin,ethereum',
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_market_cap': 'true'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'bitcoin' in data:
                            conditions.btc_price = data['bitcoin']['usd']
                            conditions.btc_24h_change = data['bitcoin'].get('usd_24h_change', 0.0)
                        
                        if 'ethereum' in data:
                            conditions.eth_price = data['ethereum']['usd']
                            conditions.eth_24h_change = data['ethereum'].get('usd_24h_change', 0.0)
                
                # Get global market data
                global_url = f"{self.coingecko_base}/global"
                async with session.get(global_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data:
                            global_data = data['data']
                            conditions.btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0.0)
                            conditions.total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0.0)
                            conditions.market_cap_24h_change = global_data.get('market_cap_change_percentage_24h_usd', 0.0)
        
        except Exception as e:
            logger.warning(f"CoinGecko data fetch failed: {e}")
    
    async def _fetch_binance_data(self, conditions: MarketConditions):
        """Fetch funding rates and orderbook data from Binance."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get funding rates
                funding_url = f"{self.binance_base}/premiumIndex"
                
                for symbol in ['BTCUSDT', 'ETHUSDT']:
                    params = {'symbol': symbol}
                    async with session.get(funding_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            funding_rate = float(data.get('lastFundingRate', 0)) * 100  # Convert to %
                            
                            if symbol == 'BTCUSDT':
                                conditions.btc_funding_rate = funding_rate * 365 * 3  # Annualized
                            else:
                                conditions.eth_funding_rate = funding_rate * 365 * 3
                
                # Get orderbook depth (simplified - top 20 levels)
                depth_url = f"{self.binance_base}/depth"
                
                for symbol in ['BTCUSDT', 'ETHUSDT']:
                    params = {'symbol': symbol, 'limit': 20}
                    async with session.get(depth_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Calculate 1% depth
                            if symbol == 'BTCUSDT':
                                current_price = conditions.btc_price
                            else:
                                current_price = conditions.eth_price
                            
                            if current_price > 0:
                                depth = self._calculate_orderbook_depth(data, current_price, 0.01)
                                if symbol == 'BTCUSDT':
                                    conditions.btc_orderbook_depth_1pct = depth
                                else:
                                    conditions.eth_orderbook_depth_1pct = depth
        
        except Exception as e:
            logger.warning(f"Binance data fetch failed: {e}")
    
    async def _fetch_fear_greed_data(self, conditions: MarketConditions):
        """Fetch Fear & Greed Index data."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_api) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            conditions.crypto_fear_greed_index = int(data['data'][0]['value'])
        
        except Exception as e:
            logger.warning(f"Fear & Greed index fetch failed: {e}")
    
    async def _calculate_volatility_metrics(self, conditions: MarketConditions):
        """Calculate realized volatility and vol risk premium."""
        try:
            # This would require historical price data - simplified for now
            # In production, you'd fetch OHLC data and calculate realized vol
            
            # Placeholder values based on typical ranges
            # BTC realized vol typically 40-80%
            conditions.btc_realized_vol_7d = 60.0  # Would be calculated from price data
            conditions.eth_realized_vol_7d = 70.0  # Would be calculated from price data
            
            # Implied vol would come from options data (Deribit, etc.)
            conditions.btc_implied_vol = 65.0  # Would fetch from options APIs
            conditions.eth_implied_vol = 75.0
            
            # Vol risk premium = Implied - Realized
            conditions.vol_risk_premium_btc = conditions.btc_implied_vol - conditions.btc_realized_vol_7d
            conditions.vol_risk_premium_eth = conditions.eth_implied_vol - conditions.eth_realized_vol_7d
        
        except Exception as e:
            logger.warning(f"Volatility metrics calculation failed: {e}")
    
    def _calculate_orderbook_depth(self, orderbook_data: Dict, current_price: float, depth_pct: float) -> float:
        """Calculate orderbook depth within specified percentage of current price."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            depth_range = current_price * depth_pct
            min_price = current_price - depth_range
            max_price = current_price + depth_range
            
            bid_depth = sum(float(qty) for price, qty in bids if float(price) >= min_price)
            ask_depth = sum(float(qty) for price, qty in asks if float(price) <= max_price)
            
            return (bid_depth + ask_depth) * current_price
        
        except Exception as e:
            logger.warning(f"Orderbook depth calculation failed: {e}")
            return 0.0
    
    def _determine_market_regime(self, conditions: MarketConditions):
        """Determine current market regime based on conditions."""
        # Market session based on UTC time
        utc_hour = datetime.utcnow().hour
        if 0 <= utc_hour < 8:
            conditions.market_session = "asia"
        elif 8 <= utc_hour < 16:
            conditions.market_session = "europe"
        else:
            conditions.market_session = "americas"
        
        # Volatility regime based on realized vol
        avg_vol = (conditions.btc_realized_vol_7d + conditions.eth_realized_vol_7d) / 2
        if avg_vol < 40:
            conditions.volatility_regime = "low"
        elif avg_vol > 80:
            conditions.volatility_regime = "high"
        elif avg_vol > 120:
            conditions.volatility_regime = "extreme"
        else:
            conditions.volatility_regime = "normal"
        
        # Funding environment
        avg_funding = (abs(conditions.btc_funding_rate) + abs(conditions.eth_funding_rate)) / 2
        if avg_funding > 20:  # >20% annualized
            if conditions.btc_funding_rate > 0:
                conditions.funding_environment = "bullish_extreme"
            else:
                conditions.funding_environment = "bearish_extreme"
        elif avg_funding > 10:  # 10-20%
            if conditions.btc_funding_rate > 0:
                conditions.funding_environment = "bullish"
            else:
                conditions.funding_environment = "bearish"
        else:
            conditions.funding_environment = "neutral"
    
    def get_market_summary(self, conditions: MarketConditions) -> Dict[str, Any]:
        """Generate a summary of market conditions for display."""
        return {
            "prices": {
                "btc": f"${conditions.btc_price:,.0f} ({conditions.btc_24h_change:+.1f}%)",
                "eth": f"${conditions.eth_price:,.0f} ({conditions.eth_24h_change:+.1f}%)"
            },
            "volatility": {
                "btc_realized_7d": f"{conditions.btc_realized_vol_7d:.1f}%",
                "eth_realized_7d": f"{conditions.eth_realized_vol_7d:.1f}%",
                "btc_vol_risk_premium": f"{conditions.vol_risk_premium_btc:+.1f}%",
                "eth_vol_risk_premium": f"{conditions.vol_risk_premium_eth:+.1f}%"
            },
            "funding": {
                "btc_funding_annualized": f"{conditions.btc_funding_rate:+.1f}%",
                "eth_funding_annualized": f"{conditions.eth_funding_rate:+.1f}%",
                "environment": conditions.funding_environment
            },
            "market": {
                "fear_greed_index": conditions.crypto_fear_greed_index,
                "btc_dominance": f"{conditions.btc_dominance:.1f}%",
                "market_cap_change_24h": f"{conditions.market_cap_24h_change:+.1f}%",
                "session": conditions.market_session,
                "volatility_regime": conditions.volatility_regime
            },
            "liquidity": {
                "btc_depth_1pct": f"${conditions.btc_orderbook_depth_1pct:,.0f}",
                "eth_depth_1pct": f"${conditions.eth_orderbook_depth_1pct:,.0f}"
            }
        }


# Global instance for easy access
market_provider = MarketConditionsProvider()
