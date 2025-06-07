import os
import asyncio
import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from dotenv import load_dotenv

# Import custom modules
from indodax_api import IndodaxAPI
from technical_analysis import TechnicalAnalysis
from lstm_predictor import LSTMPredictor
from notification_manager import NotificationManager
from transaction_logger import TransactionLogger
from risk_manager import RiskManager
from backtester import Backtester, simple_ma_strategy, rsi_strategy

class AITradingBot:
    """
    AI Trading Bot utama yang mengintegrasikan semua komponen
    """
    
    def __init__(self, config_file: str = ".env"):
        # Load configuration
        load_dotenv(config_file)
        
        # Initialize components
        self.api = IndodaxAPI(
            api_key=os.getenv('INDODAX_API_KEY', ''),
            secret_key=os.getenv('INDODAX_SECRET_KEY', '')
        )
        
        self.technical_analyzer = TechnicalAnalysis()
        self.lstm_predictor = LSTMPredictor()
        self.logger = TransactionLogger()
        
        # Risk management configuration
        risk_config = {
            'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', 5)),
            'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', 10)),
            'max_daily_trades': int(os.getenv('MAX_DAILY_TRADES', 10)),
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'risk_per_trade': 0.02
        }
        self.risk_manager = RiskManager(risk_config)
        
        # Notification manager
        self.notification_manager = NotificationManager(
            telegram_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            twilio_sid=os.getenv('TWILIO_ACCOUNT_SID'),
            twilio_token=os.getenv('TWILIO_AUTH_TOKEN'),
            whatsapp_from=os.getenv('TWILIO_WHATSAPP_FROM'),
            whatsapp_to=os.getenv('TWILIO_WHATSAPP_TO')
        )
        
        # Trading parameters
        self.trading_pairs = ['btcidr', 'ethidr']
        self.is_running = False
        self.auto_trading_enabled = False
        self.historical_data = {}
        self.current_positions = {}
        
        # Performance tracking
        self.daily_stats = {
            'trades_executed': 0,
            'profit_loss': 0,
            'start_balance': 0
        }
        
        print("AI Trading Bot initialized successfully!")
    
    def fetch_historical_data(self, pair: str = 'btcidr', limit: int = 1000) -> pd.DataFrame:
        """
        Mengambil data historis dari Indodax
        """
        try:
            print(f"Fetching historical data for {pair}...")
            
            # Get recent trades (as proxy for historical data)  
            trades = self.api.get_trades(pair)
            
            if not trades or len(trades) == 0:
                print(f"No trade data available for {pair}")
                return pd.DataFrame()
            
            # Handle different response formats
            if isinstance(trades, dict) and 'return' in trades:
                trades_data = trades['return']
            elif isinstance(trades, list):
                trades_data = trades
            else:
                print(f"Unexpected trade data format for {pair}")
                return pd.DataFrame()
            
            if not trades_data:
                print(f"Empty trade data for {pair}")
                return pd.DataFrame()
            
            # Convert to DataFrame with error handling
            try:
                df = pd.DataFrame(trades_data)
            except Exception as df_error:
                print(f"Error creating DataFrame: {df_error}")
                return pd.DataFrame()
            
            if df.empty:
                print(f"Empty DataFrame for {pair}")
                return pd.DataFrame()
            
            # Process the data with safe column handling
            try:
                # Ensure we have required columns with safe defaults
                if 'price' in df.columns:
                    df['close'] = pd.to_numeric(df['price'], errors='coerce')
                else:
                    df['close'] = 0
                    
                if 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'], unit='s', errors='coerce')
                elif 'tid' in df.columns:
                    # Use trade ID as proxy for time ordering
                    base_time = pd.to_datetime('now')
                    df['timestamp'] = base_time - pd.to_timedelta(range(len(df), 0, -1), unit='m')
                else:
                    # Generate timestamps
                    base_time = pd.to_datetime('now')
                    df['timestamp'] = base_time - pd.to_timedelta(range(len(df), 0, -1), unit='m')
                
                # Add OHLCV data (simplified - using close price)
                df['open'] = df['close']
                df['high'] = df['close'] * 1.001  # Slight variation
                df['low'] = df['close'] * 0.999
                
                if 'amount' in df.columns:
                    df['volume'] = pd.to_numeric(df['amount'], errors='coerce').fillna(1)
                else:
                    df['volume'] = 1
                
                # Remove rows with invalid data
                df = df.dropna(subset=['close', 'timestamp'])
                
                if df.empty:
                    print(f"No valid data after processing for {pair}")
                    return pd.DataFrame()
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Keep only required columns
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = df[required_columns]
                
                print(f"Retrieved {len(df)} data points for {pair}")
                return df.tail(limit)  # Return last N records
                
            except Exception as process_error:
                print(f"Error processing data for {pair}: {process_error}")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching historical data for {pair}: {e}")
            self.logger.log_error("DATA_FETCH_ERROR", str(e), "fetch_historical_data", pair)
            return pd.DataFrame()
    
    def analyze_market(self, pair: str = 'btcidr') -> Dict:
        """
        Analisis pasar lengkap
        """
        try:
            # Get fresh data
            df = self.fetch_historical_data(pair)
            
            if df.empty:
                return {'error': 'No data available'}
            
            # Technical analysis
            df_analyzed = self.technical_analyzer.analyze_data(df, pair)
            
            # Get trading signals
            signals = self.technical_analyzer.get_signals(df_analyzed)
            
            # AI prediction
            ai_prediction = None
            ai_confidence = 0
            
            try:
                if self.lstm_predictor.is_trained or self.lstm_predictor.load_model():
                    predictions = self.lstm_predictor.predict(df_analyzed, steps_ahead=1)
                    if len(predictions) > 0:
                        ai_prediction = predictions[0]
                        # Simple confidence based on recent price stability
                        recent_volatility = df_analyzed['close'].tail(10).std() / df_analyzed['close'].tail(10).mean()
                        ai_confidence = max(50, 100 - (recent_volatility * 1000))
            except Exception as e:
                print(f"AI prediction error: {e}")
            
            # Current market data
            ticker = self.api.get_ticker(pair)
            current_price = float(ticker.get('last', 0)) if ticker else df_analyzed['close'].iloc[-1]
            
            # Compile analysis results
            analysis = {
                'pair': pair.upper(),
                'current_price': current_price,
                'timestamp': datetime.now(),
                
                # Technical indicators
                'rsi_value': df_analyzed['rsi'].iloc[-1] if 'rsi' in df_analyzed.columns else 0,
                'rsi_signal': signals.get('rsi_signal', 'HOLD'),
                'macd_signal': signals.get('macd_signal', 'HOLD'),
                'sma_signal': signals.get('sma_signal', 'HOLD'),
                'bb_signal': signals.get('bb_signal', 'HOLD'),
                'overall_signal': signals.get('overall_signal', 'HOLD'),
                
                # AI prediction
                'predicted_price': ai_prediction,
                'ai_confidence': ai_confidence,
                'predicted_direction': 'UP' if ai_prediction and ai_prediction > current_price else 'DOWN',
                
                # Risk management
                'stop_loss': self.risk_manager.stop_loss_percent,
                'take_profit': self.risk_manager.take_profit_percent,
                
                # Raw data for further analysis
                'technical_data': df_analyzed.tail(1).to_dict('records')[0] if not df_analyzed.empty else {}
            }
            
            # Log signal
            self.logger.log_signal(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error in market analysis: {e}")
            self.logger.log_error("ANALYSIS_ERROR", str(e), "analyze_market", pair)
            return {'error': str(e)}
    
    async def execute_trade(self, pair: str, action: str, analysis: Dict) -> Dict:
        """
        Eksekusi trading berdasarkan analisis
        """
        try:
            # Get account info
            account_info = self.api.get_account_info()
            if not account_info or 'return' not in account_info:
                return {'success': False, 'message': 'Cannot get account info'}
            
            balance_info = account_info['return']['balance']
            idr_balance = float(balance_info.get('idr', 0))
            
            # Risk management check
            current_price = analysis.get('current_price', 0)
            
            trade_data = {
                'pair': pair,
                'order_type': action,
                'price': current_price
            }
            
            risk_check = self.risk_manager.should_execute_trade(trade_data, idr_balance)
            
            if not risk_check['should_trade']:
                return {
                    'success': False,
                    'message': f"Risk management block: {risk_check['reason']}"
                }
            
            # Calculate trade amount
            if action.lower() == 'buy':
                # Use recommended position size from risk manager
                position_size_idr = risk_check['recommended_position_size'] * current_price
                trade_amount_idr = min(position_size_idr, idr_balance * 0.1)  # Max 10% of balance
                amount = trade_amount_idr / current_price
                
                if trade_amount_idr < 10000:  # Minimum trade amount
                    return {'success': False, 'message': 'Trade amount too small'}
                
                # Execute buy order
                result = self.api.place_buy_order(pair, amount, current_price)
                
            else:  # sell
                # Get asset balance
                asset_name = pair.replace('idr', '')
                asset_balance = float(balance_info.get(asset_name, 0))
                
                if asset_balance < 0.0001:  # Minimum amount to sell
                    return {'success': False, 'message': 'Insufficient asset balance'}
                
                amount = asset_balance * 0.9  # Sell 90% of holdings
                
                # Execute sell order
                result = self.api.place_sell_order(pair, amount, current_price)
            
            # Process result
            if result and 'success' in result and result['success']:
                # Log successful trade
                trade_log = {
                    'pair': pair,
                    'order_type': action,
                    'amount': amount,
                    'price': current_price,
                    'total': amount * current_price,
                    'order_id': result.get('return', {}).get('order_id', ''),
                    'status': 'SUCCESS',
                    'reason': f"Signal: {analysis.get('overall_signal', 'N/A')}",
                    'rsi': analysis.get('rsi_value', 0),
                    'macd_signal': analysis.get('macd_signal', ''),
                    'sma_signal': analysis.get('sma_signal', ''),
                    'predicted_price': analysis.get('predicted_price', 0),
                    'ai_confidence': analysis.get('ai_confidence', 0),
                    'stop_loss': risk_check.get('stop_loss_price', 0),
                    'take_profit': risk_check.get('take_profit_price', 0)
                }
                
                self.logger.log_trade(trade_log)
                
                # Update risk manager
                self.risk_manager.update_trade_result({'pnl': 0})  # PnL will be calculated later
                
                # Send notification
                await self.notification_manager.send_trade_execution_notification(trade_log)
                
                return {
                    'success': True,
                    'message': f"{action.capitalize()} order executed successfully",
                    'trade_data': trade_log
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'API call failed'
                self.logger.log_error("TRADE_EXECUTION_ERROR", error_msg, "execute_trade", pair)
                return {'success': False, 'message': error_msg}
                
        except Exception as e:
            print(f"Error executing trade: {e}")
            self.logger.log_error("TRADE_ERROR", str(e), "execute_trade", pair)
            return {'success': False, 'message': str(e)}
    
    async def trading_cycle(self):
        """
        Siklus trading utama
        """
        try:
            print(f"[{datetime.now()}] Running trading cycle...")
            
            for pair in self.trading_pairs:
                if not self.is_running:
                    break
                
                # Analyze market
                analysis = self.analyze_market(pair)
                
                if 'error' in analysis:
                    continue
                
                # Send signal notification
                await self.notification_manager.send_trading_signal_notification(analysis)
                
                # Auto trading logic
                if self.auto_trading_enabled:
                    overall_signal = analysis.get('overall_signal', 'HOLD')
                    ai_confidence = analysis.get('ai_confidence', 0)
                    
                    # Only trade if confidence is high enough
                    if ai_confidence >= 70 and overall_signal in ['BUY', 'SELL']:
                        result = await self.execute_trade(pair, overall_signal.lower(), analysis)
                        print(f"Auto trade result for {pair}: {result.get('message', 'Unknown')}")
                
                # Small delay between pairs
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"Error in trading cycle: {e}")
            self.logger.log_error("TRADING_CYCLE_ERROR", str(e), "trading_cycle")
    
    def train_ai_model(self, pair: str = 'btcidr', retrain: bool = False):
        """
        Melatih model AI LSTM
        """
        try:
            print(f"Training AI model for {pair}...")
            
            # Get more historical data for training
            df = self.fetch_historical_data(pair, limit=2000)
            
            if len(df) < 100:
                print("Not enough data for training")
                return False
            
            # Train model
            if retrain or not self.lstm_predictor.load_model():
                training_results = self.lstm_predictor.train(
                    data=df,
                    epochs=int(os.getenv('LSTM_EPOCHS', 50)),
                    batch_size=int(os.getenv('LSTM_BATCH_SIZE', 32))
                )
                
                print(f"Training completed. Validation loss: {training_results.get('val_loss', 'N/A')}")
                
                # Save model
                self.lstm_predictor.save_model()
                
                return True
            else:
                print("Model loaded from saved file")
                return True
                
        except Exception as e:
            print(f"Error training AI model: {e}")
            self.logger.log_error("AI_TRAINING_ERROR", str(e), "train_ai_model", pair)
            return False
    
    def run_backtest(self, pair: str = 'btcidr', strategy: str = 'ma_cross', days: int = 30) -> Dict:
        """
        Menjalankan backtesting
        """
        try:
            print(f"Running backtest for {pair} using {strategy} strategy...")
            
            # Get historical data
            df = self.fetch_historical_data(pair, limit=days * 24)  # Assuming hourly data
            
            if df.empty:
                return {'error': 'No data for backtesting'}
            
            # Initialize backtester
            backtester = Backtester(initial_balance=1000000)  # 1M IDR
            
            # Select strategy
            if strategy == 'ma_cross':
                strategy_func = simple_ma_strategy
                strategy_params = {'short_window': 10, 'long_window': 30, 'position_size': 0.1}
            elif strategy == 'rsi':
                strategy_func = rsi_strategy
                strategy_params = {'rsi_period': 14, 'oversold': 30, 'overbought': 70, 'position_size': 0.1}
            else:
                return {'error': 'Unknown strategy'}
            
            # Run backtest
            results = backtester.run_backtest(df, strategy_func, **strategy_params)
            
            print(f"Backtest completed: {results.get('summary', 'No summary')}")
            return results
            
        except Exception as e:
            print(f"Error in backtesting: {e}")
            self.logger.log_error("BACKTEST_ERROR", str(e), "run_backtest", pair)
            return {'error': str(e)}
    
    def start_bot(self, auto_trading: bool = False):
        """
        Memulai bot trading
        """
        self.is_running = True
        self.auto_trading_enabled = auto_trading
        
        print(f"🚀 AI Trading Bot started! Auto trading: {'ON' if auto_trading else 'OFF'}")
        
        # Schedule trading cycles
        schedule.every(5).minutes.do(lambda: asyncio.run(self.trading_cycle()))
        
        # Schedule daily reports
        schedule.every().day.at("09:00").do(lambda: asyncio.run(self.send_daily_report()))
        
        # Main loop
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping bot...")
                self.stop_bot()
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                self.logger.log_error("MAIN_LOOP_ERROR", str(e), "start_bot")
                time.sleep(10)  # Wait before retrying
    
    def stop_bot(self):
        """
        Menghentikan bot trading
        """
        self.is_running = False
        self.auto_trading_enabled = False
        print("🛑 AI Trading Bot stopped!")
    
    async def send_daily_report(self):
        """
        Mengirim laporan harian
        """
        try:
            # Get account info
            account_info = self.api.get_account_info()
            
            if account_info and 'return' in account_info:
                balance_info = account_info['return']['balance']
                
                portfolio_data = {
                    'idr_balance': float(balance_info.get('idr', 0)),
                    'btc_balance': float(balance_info.get('btc', 0)),
                    'eth_balance': float(balance_info.get('eth', 0)),
                    'total_value': 0,  # Will be calculated
                    'daily_pnl': self.daily_stats.get('profit_loss', 0),
                    'daily_pnl_percent': 0,
                    'trades_today': self.daily_stats.get('trades_executed', 0),
                    'win_rate': 75.0  # Placeholder
                }
                
                # Calculate total value (simplified)
                btc_price = self.api.get_ticker('btcidr')
                eth_price = self.api.get_ticker('ethidr')
                
                portfolio_data['total_value'] = (
                    portfolio_data['idr_balance'] +
                    portfolio_data['btc_balance'] * float(btc_price.get('last', 0)) +
                    portfolio_data['eth_balance'] * float(eth_price.get('last', 0))
                )
                
                await self.notification_manager.send_portfolio_update_notification(portfolio_data)
                self.logger.log_portfolio(portfolio_data)
                
        except Exception as e:
            print(f"Error sending daily report: {e}")
            self.logger.log_error("DAILY_REPORT_ERROR", str(e), "send_daily_report")
    
    async def test_notifications(self):
        """
        Test sistem notifikasi
        """
        print("Testing notification system...")
        results = self.notification_manager.test_notifications()
        
        for platform, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{platform.capitalize()}: {status}")
        
        return results

# Main execution
if __name__ == "__main__":
    # Create bot instance
    bot = AITradingBot()
    
    # Test notifications
    print("Testing notifications...")
    notification_results = asyncio.run(bot.test_notifications())
    
    # Train AI model (optional)
    print("Training AI model...")
    bot.train_ai_model('btcidr')
    
    # Run backtest (optional)
    print("Running backtest...")
    backtest_results = bot.run_backtest('btcidr', 'ma_cross', 7)
    print(f"Backtest summary: {backtest_results.get('summary', 'No results')}")
    
    # Start bot
    try:
        bot.start_bot(auto_trading=False)  # Set to True for auto trading
    except KeyboardInterrupt:
        print("\nExiting...")
        bot.stop_bot()
