import pandas as pd
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

class TransactionLogger:
    """
    Kelas untuk logging transaksi ke file CSV dan JSON
    """
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        self.ensure_log_directory()
        
        # File paths
        self.trades_csv = os.path.join(log_directory, "trades.csv")
        self.signals_csv = os.path.join(log_directory, "signals.csv")
        self.portfolio_csv = os.path.join(log_directory, "portfolio.csv")
        self.errors_csv = os.path.join(log_directory, "errors.csv")
        
        # Initialize CSV files if they don't exist
        self.initialize_csv_files()
    
    def ensure_log_directory(self):
        """Memastikan direktori log ada"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
    
    def initialize_csv_files(self):
        """Inisialisasi file CSV dengan header"""
        
        # Trades CSV
        if not os.path.exists(self.trades_csv):
            trades_headers = [
                'timestamp', 'pair', 'order_type', 'amount', 'price', 'total',
                'order_id', 'status', 'reason', 'rsi', 'macd_signal', 'sma_signal',
                'predicted_price', 'ai_confidence', 'stop_loss', 'take_profit'
            ]
            with open(self.trades_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(trades_headers)
        
        # Signals CSV
        if not os.path.exists(self.signals_csv):
            signals_headers = [
                'timestamp', 'pair', 'current_price', 'rsi_value', 'rsi_signal',
                'macd_signal', 'sma_signal', 'bb_signal', 'overall_signal',
                'predicted_price', 'predicted_direction', 'ai_confidence'
            ]
            with open(self.signals_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(signals_headers)
        
        # Portfolio CSV
        if not os.path.exists(self.portfolio_csv):
            portfolio_headers = [
                'timestamp', 'idr_balance', 'btc_balance', 'eth_balance',
                'total_value', 'daily_pnl', 'daily_pnl_percent',
                'total_pnl', 'total_pnl_percent', 'trades_today', 'win_rate'
            ]
            with open(self.portfolio_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(portfolio_headers)
        
        # Errors CSV
        if not os.path.exists(self.errors_csv):
            errors_headers = [
                'timestamp', 'error_type', 'error_message', 'function_name',
                'pair', 'additional_info'
            ]
            with open(self.errors_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(errors_headers)
    
    def log_trade(self, trade_data: Dict):
        """Log transaksi trading"""
        try:
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp,
                trade_data.get('pair', ''),
                trade_data.get('order_type', ''),
                trade_data.get('amount', 0),
                trade_data.get('price', 0),
                trade_data.get('total', 0),
                trade_data.get('order_id', ''),
                trade_data.get('status', ''),
                trade_data.get('reason', ''),
                trade_data.get('rsi', 0),
                trade_data.get('macd_signal', ''),
                trade_data.get('sma_signal', ''),
                trade_data.get('predicted_price', 0),
                trade_data.get('ai_confidence', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0)
            ]
            
            with open(self.trades_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"Trade logged: {trade_data.get('order_type', 'Unknown')} {trade_data.get('pair', 'Unknown')}")
            
        except Exception as e:
            print(f"Error logging trade: {e}")
    
    def log_signal(self, signal_data: Dict):
        """Log sinyal trading"""
        try:
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp,
                signal_data.get('pair', ''),
                signal_data.get('current_price', 0),
                signal_data.get('rsi_value', 0),
                signal_data.get('rsi_signal', ''),
                signal_data.get('macd_signal', ''),
                signal_data.get('sma_signal', ''),
                signal_data.get('bb_signal', ''),
                signal_data.get('overall_signal', ''),
                signal_data.get('predicted_price', 0),
                signal_data.get('predicted_direction', ''),
                signal_data.get('confidence', 0)
            ]
            
            with open(self.signals_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
        except Exception as e:
            print(f"Error logging signal: {e}")
    
    def log_portfolio(self, portfolio_data: Dict):
        """Log status portfolio"""
        try:
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp,
                portfolio_data.get('idr_balance', 0),
                portfolio_data.get('btc_balance', 0),
                portfolio_data.get('eth_balance', 0),
                portfolio_data.get('total_value', 0),
                portfolio_data.get('daily_pnl', 0),
                portfolio_data.get('daily_pnl_percent', 0),
                portfolio_data.get('total_pnl', 0),
                portfolio_data.get('total_pnl_percent', 0),
                portfolio_data.get('trades_today', 0),
                portfolio_data.get('win_rate', 0)
            ]
            
            with open(self.portfolio_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
        except Exception as e:
            print(f"Error logging portfolio: {e}")
    
    def log_error(self, error_type: str, error_message: str, function_name: str = "",
                  pair: str = "", additional_info: str = ""):
        """Log error/exception"""
        try:
            timestamp = datetime.now().isoformat()
            
            row = [
                timestamp,
                error_type,
                error_message,
                function_name,
                pair,
                additional_info
            ]
            
            with open(self.errors_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"Error logged: {error_type} - {error_message}")
            
        except Exception as e:
            print(f"Error logging error: {e}")
    
    def get_trades_history(self, limit: int = 100) -> pd.DataFrame:
        """Mendapatkan riwayat trading"""
        try:
            if os.path.exists(self.trades_csv):
                df = pd.read_csv(self.trades_csv)
                return df.tail(limit)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading trades history: {e}")
            return pd.DataFrame()
    
    def get_signals_history(self, limit: int = 100) -> pd.DataFrame:
        """Mendapatkan riwayat sinyal"""
        try:
            if os.path.exists(self.signals_csv):
                df = pd.read_csv(self.signals_csv)
                return df.tail(limit)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading signals history: {e}")
            return pd.DataFrame()
    
    def get_portfolio_history(self, limit: int = 100) -> pd.DataFrame:
        """Mendapatkan riwayat portfolio"""
        try:
            if os.path.exists(self.portfolio_csv):
                df = pd.read_csv(self.portfolio_csv)
                return df.tail(limit)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading portfolio history: {e}")
            return pd.DataFrame()
    
    def get_errors_history(self, limit: int = 100) -> pd.DataFrame:
        """Mendapatkan riwayat error"""
        try:
            if os.path.exists(self.errors_csv):
                df = pd.read_csv(self.errors_csv)
                return df.tail(limit)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading errors history: {e}")
            return pd.DataFrame()
    
    def get_trading_stats(self, days: int = 30) -> Dict:
        """Mendapatkan statistik trading"""
        try:
            df = self.get_trades_history(1000)  # Get more data for analysis
            
            if df.empty:
                return {}
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by days
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            df_filtered = df[df['timestamp'] >= cutoff_date]
            
            if df_filtered.empty:
                return {}
            
            # Calculate statistics
            total_trades = len(df_filtered)
            buy_trades = len(df_filtered[df_filtered['order_type'] == 'buy'])
            sell_trades = len(df_filtered[df_filtered['order_type'] == 'sell'])
            
            # Calculate P&L (simplified)
            total_buy_value = df_filtered[df_filtered['order_type'] == 'buy']['total'].sum()
            total_sell_value = df_filtered[df_filtered['order_type'] == 'sell']['total'].sum()
            
            stats = {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_buy_value': total_buy_value,
                'total_sell_value': total_sell_value,
                'estimated_pnl': total_sell_value - total_buy_value,
                'avg_trade_size': df_filtered['total'].mean(),
                'days_analyzed': days
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating trading stats: {e}")
            return {}
    
    def export_data_to_json(self, filename: str = None) -> str:
        """Export semua data ke JSON"""
        try:
            if filename is None:
                filename = f"trading_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'trades': self.get_trades_history(1000).to_dict('records') if not self.get_trades_history(1000).empty else [],
                'signals': self.get_signals_history(1000).to_dict('records') if not self.get_signals_history(1000).empty else [],
                'portfolio': self.get_portfolio_history(1000).to_dict('records') if not self.get_portfolio_history(1000).empty else [],
                'errors': self.get_errors_history().to_dict('records') if not self.get_errors_history().empty else [],
                'stats': self.get_trading_stats()
            }
            
            filepath = os.path.join(self.log_directory, filename)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return ""
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Membersihkan log lama"""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            for csv_file in [self.trades_csv, self.signals_csv, self.portfolio_csv]:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df_filtered = df[df['timestamp'] >= cutoff_date]
                        df_filtered.to_csv(csv_file, index=False)
            
            print(f"Cleaned up logs older than {days_to_keep} days")
            
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
