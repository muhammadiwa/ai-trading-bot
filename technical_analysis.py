import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
import requests
from datetime import datetime, timedelta

class TechnicalAnalysis:
    """
    Kelas untuk analisis teknikal
    """
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_sma(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Menghitung Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.Series, window: int = 20) -> pd.Series:
        """Menghitung Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Menghitung Relative Strength Index"""
        return ta.momentum.RSIIndicator(data, window=window).rsi()
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Menghitung MACD"""
        macd_indicator = ta.trend.MACD(data, window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            'macd': macd_indicator.macd(),
            'macd_signal': macd_indicator.macd_signal(),
            'macd_histogram': macd_indicator.macd_diff()
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Menghitung Bollinger Bands"""
        bb_indicator = ta.volatility.BollingerBands(data, window=window, window_dev=std_dev)
        return {
            'bb_upper': bb_indicator.bollinger_hband(),
            'bb_middle': bb_indicator.bollinger_mavg(),
            'bb_lower': bb_indicator.bollinger_lband()
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Menghitung Stochastic Oscillator"""
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=k_window, smooth_window=d_window)
        return {
            'stoch_k': stoch_indicator.stoch(),
            'stoch_d': stoch_indicator.stoch_signal()
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Menghitung Average True Range"""
        return ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()
    
    def calculate_volume_indicators(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Menghitung indikator volume"""
        return {
            'vwap': ta.volume.VolumeSMAIndicator(close, volume, window=20).volume_sma(),
            'mfi': ta.volume.MFIIndicator(close, close, close, volume, window=14).money_flow_index(),
            'obv': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        }
    
    def get_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Mendapatkan sinyal trading berdasarkan analisis teknikal
        """
        signals = {
            'rsi_signal': 'HOLD',
            'macd_signal': 'HOLD',
            'sma_signal': 'HOLD',
            'bb_signal': 'HOLD',
            'overall_signal': 'HOLD'
        }
        
        if len(df) < 50:  # Need enough data for analysis
            return signals
        
        latest_data = df.iloc[-1]
        prev_data = df.iloc[-2]
        
        # RSI Signal
        if latest_data['rsi'] < 30:
            signals['rsi_signal'] = 'BUY'  # Oversold
        elif latest_data['rsi'] > 70:
            signals['rsi_signal'] = 'SELL'  # Overbought
        
        # MACD Signal
        if (latest_data['macd'] > latest_data['macd_signal'] and 
            prev_data['macd'] <= prev_data['macd_signal']):
            signals['macd_signal'] = 'BUY'  # Bullish crossover
        elif (latest_data['macd'] < latest_data['macd_signal'] and 
              prev_data['macd'] >= prev_data['macd_signal']):
            signals['macd_signal'] = 'SELL'  # Bearish crossover
        
        # Moving Average Signal
        if latest_data['close'] > latest_data['sma_20']:
            signals['sma_signal'] = 'BUY'
        elif latest_data['close'] < latest_data['sma_20']:
            signals['sma_signal'] = 'SELL'
        
        # Bollinger Bands Signal
        if latest_data['close'] <= latest_data['bb_lower']:
            signals['bb_signal'] = 'BUY'  # Price touching lower band
        elif latest_data['close'] >= latest_data['bb_upper']:
            signals['bb_signal'] = 'SELL'  # Price touching upper band
        
        # Overall Signal (majority voting)
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        
        if buy_signals > sell_signals:
            signals['overall_signal'] = 'BUY'
        elif sell_signals > buy_signals:
            signals['overall_signal'] = 'SELL'
        
        return signals
    
    def analyze_data(self, df: pd.DataFrame, pair: str = "BTCIDR") -> pd.DataFrame:
        """
        Melakukan analisis teknikal lengkap pada data
        """
        if len(df) < 50:
            print(f"Not enough data for technical analysis. Need at least 50 data points, got {len(df)}")
            return df
        
        try:
            # Moving Averages
            df['sma_20'] = self.calculate_sma(df['close'], 20)
            df['sma_50'] = self.calculate_sma(df['close'], 50)
            df['ema_12'] = self.calculate_ema(df['close'], 12)
            df['ema_26'] = self.calculate_ema(df['close'], 26)
            
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # MACD
            macd_data = self.calculate_macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['macd_signal']
            df['macd_histogram'] = macd_data['macd_histogram']
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_data['bb_upper']
            df['bb_middle'] = bb_data['bb_middle']
            df['bb_lower'] = bb_data['bb_lower']
            
            # Stochastic (if OHLC data available)
            if all(col in df.columns for col in ['high', 'low']):
                stoch_data = self.calculate_stochastic(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch_data['stoch_k']
                df['stoch_d'] = stoch_data['stoch_d']
                
                # ATR
                df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
            
            # Volume indicators (if volume data available)
            if 'volume' in df.columns:
                volume_data = self.calculate_volume_indicators(df['close'], df['volume'])
                df['vwap'] = volume_data['vwap']
                df['mfi'] = volume_data['mfi']
                df['obv'] = volume_data['obv']
            
            print(f"Technical analysis completed for {pair}")
            return df
            
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return df
