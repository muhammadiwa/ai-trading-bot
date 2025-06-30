"""
Backtesting framework for strategy evaluation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import structlog

from core.database import get_db, PriceHistory, AISignal
from ai.signal_generator import SignalGenerator
from ai.lstm_predictor import LSTMPredictor

logger = structlog.get_logger(__name__)

class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self):
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_balance: float = 0.0
        self.final_balance: float = 0.0
        self.total_return: float = 0.0
        self.total_return_percent: float = 0.0
        self.max_drawdown: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.win_rate: float = 0.0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

class Backtester:
    """Backtesting framework for trading strategies"""
    
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.lstm_predictor = LSTMPredictor()
    
    async def run_backtest(self, 
                          pair_id: str,
                          start_date: datetime,
                          end_date: datetime,
                          initial_balance: float = 1000000.0,
                          strategy: str = "ai_signals",
                          min_confidence: float = 0.6) -> BacktestResult:
        """Run backtest for a specific strategy"""
        try:
            logger.info("Starting backtest", 
                       pair_id=pair_id, 
                       start_date=start_date.date(),
                       end_date=end_date.date(),
                       strategy=strategy)
            
            # Get historical data
            historical_data = await self._get_historical_data(pair_id, start_date, end_date)
            
            if historical_data is None or len(historical_data) < 10:
                raise Exception("Insufficient historical data for backtesting")
            
            # Initialize backtest result
            result = BacktestResult()
            result.start_date = start_date
            result.end_date = end_date
            result.initial_balance = initial_balance
            
            # Run strategy-specific backtest
            if strategy == "ai_signals":
                await self._backtest_ai_signals(result, historical_data, pair_id, min_confidence)
            elif strategy == "lstm_prediction":
                await self._backtest_lstm_prediction(result, historical_data, pair_id)
            elif strategy == "buy_and_hold":
                await self._backtest_buy_and_hold(result, historical_data)
            elif strategy == "dca":
                await self._backtest_dca(result, historical_data, initial_balance)
            else:
                raise Exception(f"Unknown strategy: {strategy}")
            
            # Calculate final metrics
            self._calculate_metrics(result)
            
            logger.info("Backtest completed", 
                       pair_id=pair_id,
                       strategy=strategy,
                       total_return=f"{result.total_return_percent:.2f}%",
                       max_drawdown=f"{result.max_drawdown:.2f}%",
                       win_rate=f"{result.win_rate:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error("Failed to run backtest", 
                       pair_id=pair_id, strategy=strategy, error=str(e))
            raise
    
    async def _get_historical_data(self, pair_id: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for backtesting"""
        try:
            db = get_db()
            try:
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= start_date,
                    PriceHistory.timestamp <= end_date
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                if not price_data:
                    logger.warning("No historical data found in database", pair_id=pair_id)
                    return None
                
                data = []
                for record in price_data:
                    data.append({
                        'timestamp': record.timestamp,
                        'open': record.open_price,
                        'high': record.high_price,
                        'low': record.low_price,
                        'close': record.close_price,
                        'volume': record.volume
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                return df
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get historical data", pair_id=pair_id, error=str(e))
            return None
    
    async def _backtest_ai_signals(self, result: BacktestResult, 
                                  historical_data: pd.DataFrame, 
                                  pair_id: str, min_confidence: float):
        """Backtest AI signal-based strategy"""
        try:
            balance = result.initial_balance
            position = 0.0  # Amount of crypto held
            position_cost = 0.0  # Cost basis of position
            
            # Simulate trading based on historical signals
            for i in range(30, len(historical_data)):  # Start after 30 days for signal generation
                current_date = historical_data.index[i]
                current_price = historical_data.iloc[i]['close']
                
                # Get data up to current date for signal generation
                historical_subset = historical_data.iloc[:i+1]
                
                # Generate signal (simplified - in reality this would use the actual signal generator)
                signal = await self._simulate_signal_generation(historical_subset, pair_id)
                
                if signal and signal['confidence'] >= min_confidence:
                    trade_amount = balance * 0.1  # Risk 10% per trade
                    
                    if signal['signal_type'] == 'buy' and balance >= trade_amount:
                        # Buy signal
                        quantity = trade_amount / current_price
                        position += quantity
                        position_cost += trade_amount
                        balance -= trade_amount
                        
                        # Record trade
                        trade = {
                            'date': current_date,
                            'type': 'buy',
                            'price': current_price,
                            'quantity': quantity,
                            'amount': trade_amount,
                            'balance': balance,
                            'confidence': signal['confidence']
                        }
                        result.trades.append(trade)
                        
                    elif signal['signal_type'] == 'sell' and position > 0:
                        # Sell signal
                        sell_value = position * current_price
                        pnl = sell_value - position_cost
                        
                        balance += sell_value
                        
                        # Record trade
                        trade = {
                            'date': current_date,
                            'type': 'sell',
                            'price': current_price,
                            'quantity': position,
                            'amount': sell_value,
                            'balance': balance,
                            'pnl': pnl,
                            'confidence': signal['confidence']
                        }
                        result.trades.append(trade)
                        
                        # Reset position
                        position = 0.0
                        position_cost = 0.0
                
                # Record equity curve
                current_equity = balance + (position * current_price)
                result.equity_curve.append({
                    'date': current_date,
                    'equity': current_equity,
                    'balance': balance,
                    'position_value': position * current_price
                })
            
            # Close any remaining position
            if position > 0:
                final_price = historical_data.iloc[-1]['close']
                sell_value = position * final_price
                balance += sell_value
                
                trade = {
                    'date': historical_data.index[-1],
                    'type': 'sell',
                    'price': final_price,
                    'quantity': position,
                    'amount': sell_value,
                    'balance': balance,
                    'pnl': sell_value - position_cost,
                    'confidence': 1.0
                }
                result.trades.append(trade)
            
            result.final_balance = balance
            
        except Exception as e:
            logger.error("Failed to backtest AI signals", error=str(e))
            result.final_balance = result.initial_balance
    
    async def _backtest_buy_and_hold(self, result: BacktestResult, historical_data: pd.DataFrame):
        """Backtest simple buy and hold strategy"""
        try:
            initial_price = historical_data.iloc[0]['close']
            final_price = historical_data.iloc[-1]['close']
            
            # Buy at the beginning
            quantity = result.initial_balance / initial_price
            
            # Sell at the end
            result.final_balance = quantity * final_price
            
            # Record trades
            result.trades = [
                {
                    'date': historical_data.index[0],
                    'type': 'buy',
                    'price': initial_price,
                    'quantity': quantity,
                    'amount': result.initial_balance,
                    'balance': 0.0
                },
                {
                    'date': historical_data.index[-1],
                    'type': 'sell',
                    'price': final_price,
                    'quantity': quantity,
                    'amount': result.final_balance,
                    'balance': result.final_balance,
                    'pnl': result.final_balance - result.initial_balance
                }
            ]
            
            # Create equity curve
            for i, (date, row) in enumerate(historical_data.iterrows()):
                current_equity = quantity * row['close']
                result.equity_curve.append({
                    'date': date,
                    'equity': current_equity,
                    'balance': 0.0,
                    'position_value': current_equity
                })
            
        except Exception as e:
            logger.error("Failed to backtest buy and hold", error=str(e))
            result.final_balance = result.initial_balance
    
    async def _backtest_dca(self, result: BacktestResult, historical_data: pd.DataFrame, monthly_investment: float = 100000.0):
        """Backtest Dollar Cost Averaging strategy"""
        try:
            balance = 0.0
            total_invested = 0.0
            position = 0.0
            
            # DCA every 30 days
            dca_interval = 30
            last_dca_index = 0
            
            for i in range(len(historical_data)):
                current_date = historical_data.index[i]
                current_price = historical_data.iloc[i]['close']
                
                # Check if it's time for DCA
                if i - last_dca_index >= dca_interval or i == 0:
                    # Buy with monthly investment
                    quantity = monthly_investment / current_price
                    position += quantity
                    total_invested += monthly_investment
                    
                    # Record trade
                    trade = {
                        'date': current_date,
                        'type': 'dca_buy',
                        'price': current_price,
                        'quantity': quantity,
                        'amount': monthly_investment,
                        'total_invested': total_invested
                    }
                    result.trades.append(trade)
                    
                    last_dca_index = i
                
                # Record equity curve
                current_equity = position * current_price
                result.equity_curve.append({
                    'date': current_date,
                    'equity': current_equity,
                    'balance': 0.0,
                    'position_value': current_equity,
                    'total_invested': total_invested
                })
            
            # Final balance is the value of all accumulated crypto
            final_price = historical_data.iloc[-1]['close']
            result.final_balance = position * final_price
            result.initial_balance = total_invested  # For DCA, "initial" is total invested
            
        except Exception as e:
            logger.error("Failed to backtest DCA", error=str(e))
            result.final_balance = result.initial_balance
    
    async def _simulate_signal_generation(self, historical_data: pd.DataFrame, pair_id: str) -> Optional[Dict[str, Any]]:
        """Simulate signal generation for backtesting"""
        try:
            # This is a simplified version - in practice, you'd use the actual signal generator
            # but with historical data only up to the current point
            
            if len(historical_data) < 20:
                return None
            
            # Calculate simple indicators
            close_prices = historical_data['close']
            rsi = self._calculate_rsi(close_prices, 14)
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = close_prices.rolling(50).mean()
            
            if len(rsi) < 1 or pd.isna(rsi.iloc[-1]):
                return None
            
            current_rsi = rsi.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            # Simple signal logic
            if current_rsi < 30 and current_sma_20 > current_sma_50:
                return {
                    'signal_type': 'buy',
                    'confidence': 0.7,
                    'indicators': {'rsi': current_rsi, 'sma_20': current_sma_20, 'sma_50': current_sma_50}
                }
            elif current_rsi > 70 and current_sma_20 < current_sma_50:
                return {
                    'signal_type': 'sell',
                    'confidence': 0.7,
                    'indicators': {'rsi': current_rsi, 'sma_20': current_sma_20, 'sma_50': current_sma_50}
                }
            else:
                return {
                    'signal_type': 'hold',
                    'confidence': 0.5,
                    'indicators': {'rsi': current_rsi, 'sma_20': current_sma_20, 'sma_50': current_sma_50}
                }
            
        except Exception as e:
            logger.error("Failed to simulate signal generation", error=str(e))
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate performance metrics"""
        try:
            # Basic metrics
            result.total_return = result.final_balance - result.initial_balance
            result.total_return_percent = (result.total_return / result.initial_balance) * 100
            
            # Trade statistics
            result.total_trades = len(result.trades)
            
            if result.total_trades > 0:
                # Calculate wins/losses
                wins = [t for t in result.trades if t.get('pnl', 0) > 0]
                losses = [t for t in result.trades if t.get('pnl', 0) < 0]
                
                result.winning_trades = len(wins)
                result.losing_trades = len(losses)
                result.win_rate = (result.winning_trades / result.total_trades) * 100
                
                if result.winning_trades > 0:
                    result.avg_win = sum(t['pnl'] for t in wins) / result.winning_trades
                
                if result.losing_trades > 0:
                    result.avg_loss = sum(t['pnl'] for t in losses) / result.losing_trades
            
            # Calculate max drawdown
            if result.equity_curve:
                equity_values = [point['equity'] for point in result.equity_curve]
                running_max = np.maximum.accumulate(equity_values)
                drawdown = (equity_values - running_max) / running_max * 100
                result.max_drawdown = abs(min(drawdown)) if drawdown else 0.0
                
                # Calculate Sharpe ratio (simplified)
                if len(equity_values) > 1:
                    returns = pd.Series(equity_values).pct_change().dropna()
                    if returns.std() != 0:
                        result.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            
            # Additional metrics
            result.metrics = {
                'profit_factor': abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0,
                'trades_per_month': result.total_trades / max(1, ((result.end_date - result.start_date).days / 30)),
                'avg_trade_duration': self._calculate_avg_trade_duration(result.trades),
                'best_trade': max([t.get('pnl', 0) for t in result.trades], default=0),
                'worst_trade': min([t.get('pnl', 0) for t in result.trades], default=0)
            }
            
        except Exception as e:
            logger.error("Failed to calculate metrics", error=str(e))
    
    def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade duration in days"""
        try:
            if len(trades) < 2:
                return 0.0
            
            durations = []
            buy_date = None
            
            for trade in trades:
                if trade['type'] in ['buy', 'dca_buy']:
                    buy_date = trade['date']
                elif trade['type'] == 'sell' and buy_date:
                    duration = (trade['date'] - buy_date).days
                    durations.append(duration)
                    buy_date = None
            
            return sum(durations) / len(durations) if durations else 0.0
            
        except Exception as e:
            logger.error("Failed to calculate average trade duration", error=str(e))
            return 0.0
    
    async def compare_strategies(self, 
                               pair_id: str,
                               start_date: datetime,
                               end_date: datetime,
                               initial_balance: float = 1000000.0) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        try:
            strategies = ["ai_signals", "buy_and_hold", "dca"]
            results = {}
            
            for strategy in strategies:
                try:
                    result = await self.run_backtest(
                        pair_id, start_date, end_date, initial_balance, strategy
                    )
                    results[strategy] = result
                except Exception as e:
                    logger.error(f"Failed to backtest {strategy}", error=str(e))
                    continue
            
            return results
            
        except Exception as e:
            logger.error("Failed to compare strategies", error=str(e))
            return {}
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a formatted backtest report"""
        try:
            report = f"""
=== BACKTEST REPORT ===
Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
Initial Balance: {result.initial_balance:,.0f} IDR
Final Balance: {result.final_balance:,.0f} IDR

=== PERFORMANCE ===
Total Return: {result.total_return:,.0f} IDR ({result.total_return_percent:.2f}%)
Max Drawdown: {result.max_drawdown:.2f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}

=== TRADE STATISTICS ===
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.2f}%
Average Win: {result.avg_win:,.0f} IDR
Average Loss: {result.avg_loss:,.0f} IDR

=== ADDITIONAL METRICS ===
Profit Factor: {result.metrics.get('profit_factor', 0):.2f}
Best Trade: {result.metrics.get('best_trade', 0):,.0f} IDR
Worst Trade: {result.metrics.get('worst_trade', 0):,.0f} IDR
Avg Trade Duration: {result.metrics.get('avg_trade_duration', 0):.1f} days
Trades per Month: {result.metrics.get('trades_per_month', 0):.1f}
"""
            return report
            
        except Exception as e:
            logger.error("Failed to generate report", error=str(e))
            return "Failed to generate report"
