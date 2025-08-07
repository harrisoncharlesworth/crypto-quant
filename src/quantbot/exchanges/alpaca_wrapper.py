"""
Alpaca exchange wrapper to provide CCXT-like interface for the quantbot.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class AlpacaWrapper:
    """Alpaca wrapper that provides CCXT-like interface."""
    
    def __init__(self, paper: bool = True):
        """Initialize Alpaca clients."""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        
        # Trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper
        )
        
        # Data client (always uses live data)
        self.data_client = CryptoHistoricalDataClient()
        
        self.paper = paper
        self.markets = {}
        self.symbols = {}
        
    def _to_alpaca_timeframe(self, tf: str) -> TimeFrame:
        """Convert CCXT-style timeframe strings to Alpaca TimeFrame instance."""
        try:
            amount = int(tf[:-1])
            unit_char = tf[-1].lower()

            if unit_char == 'm':
                unit = TimeFrameUnit.Minute
            elif unit_char == 'h':
                unit = TimeFrameUnit.Hour
            elif unit_char == 'd':
                unit = TimeFrameUnit.Day
            else:
                # Default to hour for unsupported timeframes
                unit = TimeFrameUnit.Hour
                amount = 1

            return TimeFrame(amount, unit)
        except:
            # Fallback to 1 hour
            return TimeFrame(1, TimeFrameUnit.Hour)
        
    def load_markets(self):
        """Load available crypto markets."""
        try:
            # Pre-define common crypto markets since API filtering is complex
            common_cryptos = [
                'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'LTC/USD', 'XRP/USD',
                'DOGE/USD', 'LINK/USD', 'MATIC/USD', 'AVAX/USD'
            ]
            
            for crypto_symbol in common_cryptos:
                symbol = crypto_symbol.replace('/', '')  # BTCUSD format
                self.markets[symbol] = {
                    'id': symbol,
                    'symbol': symbol,
                    'base': crypto_symbol.split('/')[0],
                    'quote': crypto_symbol.split('/')[1],
                    'active': True,
                    'precision': {'amount': 8, 'price': 8},
                    'limits': {'amount': {'min': 0.0001}},
                    'info': {'symbol': crypto_symbol}
                }
                self.symbols[symbol] = crypto_symbol  # Store original format
                
            print(f"Loaded {len(self.markets)} predefined crypto markets")
        except Exception as e:
            print(f"Error loading markets: {e}")
            
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data for a symbol."""
        try:
            # Get latest quote
            from alpaca.data.requests import CryptoLatestQuoteRequest
            alpaca_symbol = self.symbols.get(symbol, f"{symbol[:3]}/{symbol[3:]}")
            
            request = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
            quotes = self.data_client.get_crypto_latest_quote(request)
            
            if alpaca_symbol in quotes:
                quote = quotes[alpaca_symbol]
                return {
                    'symbol': symbol,
                    'bid': float(quote.bid_price) if quote.bid_price else None,
                    'ask': float(quote.ask_price) if quote.ask_price else None,
                    'last': float(quote.ask_price) if quote.ask_price else None,
                    'timestamp': quote.timestamp.timestamp() * 1000,
                    'datetime': quote.timestamp.isoformat(),
                    'info': quote
                }
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            
        return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Fetch OHLCV data."""
        try:
            alpaca_symbol = self.symbols.get(symbol, f"{symbol[:3]}/{symbol[3:]}")
            
            # Convert timeframe using helper method
            alpaca_tf = self._to_alpaca_timeframe(timeframe)
            
            # Calculate start time - get more data than needed
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit + 24)  # Extra buffer
            
            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=alpaca_tf,
                start=start_time,
                end=end_time
            )
            
            bars = self.data_client.get_crypto_bars(request)
            
            ohlcv_data = []
            if alpaca_symbol in bars.data:
                for bar in bars.data[alpaca_symbol]:
                    ohlcv_data.append([
                        int(bar.timestamp.timestamp() * 1000),  # timestamp
                        float(bar.open),    # open
                        float(bar.high),    # high
                        float(bar.low),     # low
                        float(bar.close),   # close
                        float(bar.volume)   # volume
                    ])
            
            return ohlcv_data[-limit:] if ohlcv_data else []
            
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
    def fetch_balance(self) -> Dict:
        """Fetch account balance."""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            balance = {
                'USD': {
                    'free': float(account.buying_power),
                    'used': 0.0,
                    'total': float(account.portfolio_value)
                }
            }
            
            # Add crypto positions
            for position in positions:
                if position.asset_class == AssetClass.CRYPTO:
                    symbol = position.symbol.split('/')[0]  # Get base currency
                    qty = float(position.qty)
                    market_value = float(position.market_value) if position.market_value else 0.0
                    
                    balance[symbol] = {
                        'free': qty,
                        'used': 0.0,
                        'total': qty
                    }
            
            return balance
            
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return {}
    
    def create_market_buy_order(self, symbol: str, amount: float) -> Dict:
        """Create a market buy order."""
        return self._create_order(symbol, 'market', 'buy', amount, None)
    
    def create_market_sell_order(self, symbol: str, amount: float) -> Dict:
        """Create a market sell order."""
        return self._create_order(symbol, 'market', 'sell', amount, None)
    
    def create_limit_buy_order(self, symbol: str, amount: float, price: float) -> Dict:
        """Create a limit buy order."""
        return self._create_order(symbol, 'limit', 'buy', amount, price)
    
    def create_limit_sell_order(self, symbol: str, amount: float, price: float) -> Dict:
        """Create a limit sell order."""
        return self._create_order(symbol, 'limit', 'sell', amount, price)
    
    def _create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Internal method to create orders."""
        try:
            alpaca_symbol = self.symbols.get(symbol, f"{symbol[:3]}/{symbol[3:]}")
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            
            if order_type == 'market':
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=amount,
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )
            elif order_type == 'limit':
                order_request = LimitOrderRequest(
                    symbol=alpaca_symbol,
                    qty=amount,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            order = self.trading_client.submit_order(order_request)
            
            return {
                'id': order.id,
                'symbol': symbol,
                'amount': float(order.qty),
                'price': float(order.limit_price) if order.limit_price else None,
                'side': side,
                'type': order_type,
                'status': order.status.value,
                'timestamp': order.created_at.timestamp() * 1000,
                'info': order
            }
            
        except Exception as e:
            print(f"Error creating {order_type} {side} order for {symbol}: {e}")
            return {}
    
    def fetch_order(self, order_id: str) -> Dict:
        """Fetch order details."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol.replace('/', ''),
                'amount': float(order.qty),
                'filled': float(order.filled_qty) if order.filled_qty else 0.0,
                'price': float(order.limit_price) if order.limit_price else None,
                'side': 'buy' if order.side == OrderSide.BUY else 'sell',
                'type': order.order_type.value.lower(),
                'status': order.status.value.lower(),
                'timestamp': order.created_at.timestamp() * 1000,
                'info': order
            }
        except Exception as e:
            print(f"Error fetching order {order_id}: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        try:
            order = self.trading_client.cancel_order_by_id(order_id)
            return {'id': order_id, 'status': 'canceled'}
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return {}
    
    @property
    def id(self) -> str:
        """Exchange identifier."""
        return 'alpaca'
    
    @property
    def name(self) -> str:
        """Exchange name."""
        return 'Alpaca'
