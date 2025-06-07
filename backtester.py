import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

class Backtester:
    """
    Kelas untuk backtesting strategi trading
    """
    
    def __init__(self, initial_balance: float = 1000000):  # 1 juta IDR
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.trade_costs = 0.001  # 0.1% trading fee
        
    def reset(self):
        """Reset backtester untuk test baru"""
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
    
    def calculate_portfolio_value(self, current_prices: Dict) -> float:
        """Menghitung total nilai portfolio"""
        total_value = self.balance
        
        for asset, amount in self.positions.items():
            if asset in current_prices:
                total_value += amount * current_prices[asset]
        
        return total_value
    
    def execute_trade(self, timestamp: datetime, pair: str, action: str, 
                     price: float, amount: float, reason: str = "") -> bool:
        """
        Eksekusi trade dalam backtesting
        """
        try:
            asset = pair.replace('idr', '').upper()
            trade_value = amount * price
            trading_cost = trade_value * self.trade_costs
            
            if action.lower() == 'buy':
                # Beli asset
                required_balance = trade_value + trading_cost
                if self.balance >= required_balance:
                    self.balance -= required_balance
                    if asset not in self.positions:
                        self.positions[asset] = 0
                    self.positions[asset] += amount
                    
                    # Log trade
                    self.trades.append({
                        'timestamp': timestamp,
                        'pair': pair,
                        'action': action,
                        'price': price,
                        'amount': amount,
                        'value': trade_value,
                        'cost': trading_cost,
                        'balance_after': self.balance,
                        'reason': reason
                    })
                    return True
                else:
                    return False  # Insufficient balance
                    
            elif action.lower() == 'sell':
                # Jual asset
                if asset in self.positions and self.positions[asset] >= amount:
                    self.positions[asset] -= amount
                    received_amount = trade_value - trading_cost
                    self.balance += received_amount
                    
                    # Hapus posisi jika kosong
                    if self.positions[asset] == 0:
                        del self.positions[asset]
                    
                    # Log trade
                    self.trades.append({
                        'timestamp': timestamp,
                        'pair': pair,
                        'action': action,
                        'price': price,
                        'amount': amount,
                        'value': trade_value,
                        'cost': trading_cost,
                        'balance_after': self.balance,
                        'reason': reason
                    })
                    return True
                else:
                    return False  # Insufficient asset
            
            return False
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def run_backtest(self, data: pd.DataFrame, strategy_func, **strategy_params) -> Dict:
        """
        Menjalankan backtesting dengan strategi yang diberikan
        """
        self.reset()
        
        if data.empty:
            return self.get_backtest_results()
        
        print(f"Starting backtest with {len(data)} data points...")
        
        for i, row in data.iterrows():
            try:
                timestamp = pd.to_datetime(row['timestamp']) if 'timestamp' in row else datetime.now()
                current_prices = {'BTC': row['close']}  # Simplified
                
                # Hitung nilai portfolio saat ini
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_values.append({
                    'timestamp': timestamp,
                    'value': portfolio_value,
                    'balance': self.balance,
                    'positions': self.positions.copy()
                })
                
                # Jalankan strategi
                if i >= strategy_params.get('min_periods', 50):  # Need enough data for indicators
                    historical_data = data.iloc[:i+1]
                    signals = strategy_func(historical_data, **strategy_params)
                    
                    # Eksekusi trade berdasarkan sinyal
                    if signals.get('action') in ['buy', 'sell']:
                        pair = 'btcidr'
                        price = row['close']
                        
                        if signals['action'] == 'buy' and self.balance > 10000:  # Min balance check
                            # Beli dengan persentase tertentu dari balance
                            trade_percent = strategy_params.get('position_size', 0.1)  # 10% default
                            trade_amount_idr = self.balance * trade_percent
                            amount = trade_amount_idr / price
                            
                            self.execute_trade(
                                timestamp=timestamp,
                                pair=pair,
                                action='buy',
                                price=price,
                                amount=amount,
                                reason=signals.get('reason', 'Strategy signal')
                            )
                            
                        elif signals['action'] == 'sell' and 'BTC' in self.positions:
                            # Jual semua atau sebagian BTC
                            sell_percent = strategy_params.get('sell_percent', 1.0)  # 100% default
                            amount = self.positions['BTC'] * sell_percent
                            
                            if amount > 0:
                                self.execute_trade(
                                    timestamp=timestamp,
                                    pair=pair,
                                    action='sell',
                                    price=price,
                                    amount=amount,
                                    reason=signals.get('reason', 'Strategy signal')
                                )
                        
            except Exception as e:
                print(f"Error in backtest iteration {i}: {e}")
                continue
        
        return self.get_backtest_results()
    
    def get_backtest_results(self) -> Dict:
        """
        Mendapatkan hasil backtesting
        """
        if not self.portfolio_values:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'summary': "No trades executed"
            }
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # Trading metrics
        total_trades = len(self.trades)
        winning_trades = 0
        losing_trades = 0
        
        if not trades_df.empty:
            # Calculate P&L for each trade pair
            buy_trades = trades_df[trades_df['action'] == 'buy'].copy()
            sell_trades = trades_df[trades_df['action'] == 'sell'].copy()
            
            for _, sell_trade in sell_trades.iterrows():
                # Find corresponding buy trade (simplified)
                corresponding_buys = buy_trades[buy_trades['timestamp'] < sell_trade['timestamp']]
                if not corresponding_buys.empty:
                    last_buy = corresponding_buys.iloc[-1]
                    pnl = (sell_trade['price'] - last_buy['price']) * sell_trade['amount']
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = (winning_trades / max(winning_trades + losing_trades, 1)) * 100
        
        # Max drawdown
        portfolio_df['peak'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Sharpe ratio (simplified)
        if len(portfolio_df) > 1:
            returns = portfolio_df['value'].pct_change().dropna()
            if returns.std() != 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_history': portfolio_df,
            'trades_history': trades_df,
            'summary': f"Return: {total_return:.2f}%, Trades: {total_trades}, Win Rate: {win_rate:.1f}%"
        }
    
    def create_backtest_plots(self, results: Dict):
        """
        Membuat plot hasil backtesting
        """
        try:
            portfolio_df = results.get('portfolio_history', pd.DataFrame())
            trades_df = results.get('trades_history', pd.DataFrame())
            
            if portfolio_df.empty:
                return None
            
            # Portfolio value plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=portfolio_df['timestamp'],
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add trade markers
            if not trades_df.empty:
                buy_trades = trades_df[trades_df['action'] == 'buy']
                sell_trades = trades_df[trades_df['action'] == 'sell']
                
                if not buy_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_trades['timestamp'],
                        y=[results['initial_balance']] * len(buy_trades),  # Simplified
                        mode='markers',
                        name='Buy',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ))
                
                if not sell_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_trades['timestamp'],
                        y=[results['initial_balance']] * len(sell_trades),  # Simplified
                        mode='markers',
                        name='Sell',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    ))
            
            fig.update_layout(
                title='Backtest Results - Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (IDR)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating backtest plots: {e}")
            return None

def simple_ma_strategy(data: pd.DataFrame, short_window: int = 20, long_window: int = 50, **kwargs) -> Dict:
    """
    Simple Moving Average Strategy untuk backtesting
    """
    try:
        if len(data) < long_window:
            return {'action': 'hold', 'reason': 'Not enough data'}
        
        # Calculate moving averages
        data['ma_short'] = data['close'].rolling(window=short_window).mean()
        data['ma_long'] = data['close'].rolling(window=long_window).mean()
        
        # Get current and previous values
        current = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else current
        
        # Generate signals
        if (current['ma_short'] > current['ma_long'] and 
            previous['ma_short'] <= previous['ma_long']):
            return {'action': 'buy', 'reason': 'MA bullish crossover'}
        elif (current['ma_short'] < current['ma_long'] and 
              previous['ma_short'] >= previous['ma_long']):
            return {'action': 'sell', 'reason': 'MA bearish crossover'}
        else:
            return {'action': 'hold', 'reason': 'No clear signal'}
            
    except Exception as e:
        print(f"Error in MA strategy: {e}")
        return {'action': 'hold', 'reason': 'Strategy error'}

def rsi_strategy(data: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, **kwargs) -> Dict:
    """
    RSI Strategy untuk backtesting
    """
    try:
        if len(data) < rsi_period + 1:
            return {'action': 'hold', 'reason': 'Not enough data'}
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Generate signals
        if current_rsi < oversold:
            return {'action': 'buy', 'reason': f'RSI oversold ({current_rsi:.1f})'}
        elif current_rsi > overbought:
            return {'action': 'sell', 'reason': f'RSI overbought ({current_rsi:.1f})'}
        else:
            return {'action': 'hold', 'reason': f'RSI neutral ({current_rsi:.1f})'}
            
    except Exception as e:
        print(f"Error in RSI strategy: {e}")
        return {'action': 'hold', 'reason': 'Strategy error'}
