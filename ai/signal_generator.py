"""
AI-powered trading signal generator using technical analysis and machine learning
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from typing import Dict, Optional, List, Any
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import structlog

from core.indodax_api import indodax_api
from core.database import get_db, AISignal, PriceHistory

logger = structlog.get_logger(__name__)

class SignalGenerator:
    """AI-powered trading signal generator"""
    
    def __init__(self):
        self.model_path = "ai/models/"
        self.scaler = StandardScaler()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists(self.model_path):
                for file in os.listdir(self.model_path):
                    if file.endswith('.pkl'):
                        pair_id = file.replace('.pkl', '')
                        model_file = os.path.join(self.model_path, file)
                        with open(model_file, 'rb') as f:
                            self.models[pair_id] = pickle.load(f)
                        logger.info("Loaded model for pair", pair_id=pair_id)
        except Exception as e:
            logger.warning("Failed to load models", error=str(e))
    
    async def generate_signal(self, pair_id: str) -> Optional[AISignal]:
        """Generate trading signal for a specific pair"""
        try:
            logger.info("Generating signal", pair_id=pair_id)
            
            # Get historical data
            historical_data = await self._get_historical_data(pair_id)
            
            if historical_data is None or len(historical_data) < 50:
                logger.warning("Insufficient data for signal generation", pair_id=pair_id)
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(historical_data)
            
            # Generate signal using AI model or fallback to technical analysis
            if pair_id in self.models:
                signal = await self._generate_ml_signal(pair_id, indicators)
            else:
                signal = await self._generate_technical_signal(pair_id, indicators)
            
            # Save signal to database
            if signal:
                await self._save_signal(signal)
                logger.info("Signal generated successfully", 
                           pair_id=pair_id, 
                           signal_type=signal.signal_type,
                           confidence=signal.confidence)
            
            return signal
            
        except Exception as e:
            logger.error("Failed to generate signal", pair_id=pair_id, error=str(e))
            return None
    
    async def _get_historical_data(self, pair_id: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data for analysis"""
        try:
            db = get_db()
            try:
                # Get data from last 30 days
                cutoff_date = datetime.now() - timedelta(days=days)
                
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= cutoff_date
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                if not price_data:
                    # Fallback: get recent data from API
                    return await self._fetch_recent_data(pair_id)
                
                # Convert to DataFrame
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
                return df
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get historical data", pair_id=pair_id, error=str(e))
            return None
    
    async def _fetch_recent_data(self, pair_id: str) -> Optional[pd.DataFrame]:
        """Fetch recent data from Indodax API as fallback"""
        try:
            # Get OHLC data for last 7 days
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (7 * 24 * 60 * 60)  # 7 days ago
            
            ohlc_data = await indodax_api.get_ohlc_history(
                pair_id.upper(), start_time, end_time, "15"
            )
            
            if not ohlc_data:
                return None
            
            # Convert to DataFrame
            data = []
            for candle in ohlc_data:
                data.append({
                    'timestamp': datetime.fromtimestamp(candle['Time']),
                    'open': float(candle['Open']),
                    'high': float(candle['High']),
                    'low': float(candle['Low']),
                    'close': float(candle['Close']),
                    'volume': float(candle['Volume'])
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error("Failed to fetch recent data", pair_id=pair_id, error=str(e))
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators"""
        try:
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            indicators['macd'] = ta.trend.macd_diff(df['close'])
            indicators['macd_signal'] = ta.trend.macd_signal(df['close'])
            
            # RSI
            indicators['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Bollinger Bands
            indicators['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            indicators['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            indicators['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            
            # Stochastic
            indicators['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            indicators['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Volume indicators
            indicators['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
            indicators['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
            
            # Support and Resistance levels
            indicators['support'] = df['low'].rolling(window=20).min()
            indicators['resistance'] = df['high'].rolling(window=20).max()
            
            return indicators
            
        except Exception as e:
            logger.error("Failed to calculate technical indicators", error=str(e))
            return {}
    
    async def _generate_ml_signal(self, pair_id: str, indicators: Dict[str, Any]) -> Optional[AISignal]:
        """Generate signal using machine learning model"""
        try:
            model = self.models[pair_id]
            
            # Prepare features for prediction
            features = self._prepare_features(indicators)
            
            if features is None:
                return None
            
            # Make prediction
            prediction = model.predict_proba([features])[0]
            
            # Determine signal type and confidence
            buy_prob = prediction[2] if len(prediction) > 2 else 0  # Buy probability
            sell_prob = prediction[0] if len(prediction) > 0 else 0  # Sell probability
            hold_prob = prediction[1] if len(prediction) > 1 else 0  # Hold probability
            
            max_prob = max(buy_prob, sell_prob, hold_prob)
            
            if max_prob == buy_prob and buy_prob > 0.6:
                signal_type = "buy"
            elif max_prob == sell_prob and sell_prob > 0.6:
                signal_type = "sell"
            else:
                signal_type = "hold"
            
            # Create AI signal
            signal = AISignal(
                pair_id=pair_id,
                signal_type=signal_type,
                confidence=float(max_prob),
                price_prediction=self._predict_price(indicators),
                indicators=self._get_current_indicators(indicators),
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            return signal
            
        except Exception as e:
            logger.error("Failed to generate ML signal", pair_id=pair_id, error=str(e))
            return None
    
    async def _generate_technical_signal(self, pair_id: str, indicators: Dict[str, Any]) -> Optional[AISignal]:
        """Generate signal using technical analysis rules"""
        try:
            current_indicators = self._get_current_indicators(indicators)
            
            # Initialize signal strength
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            
            # RSI signals
            if 'rsi' in current_indicators:
                rsi = current_indicators['rsi']
                if rsi < 30:  # Oversold
                    buy_signals += 2
                elif rsi > 70:  # Overbought
                    sell_signals += 2
                total_signals += 2
            
            # MACD signals
            if 'macd' in current_indicators and 'macd_signal' in current_indicators:
                macd = current_indicators['macd']
                macd_signal = current_indicators['macd_signal']
                if macd > macd_signal:  # Bullish crossover
                    buy_signals += 2
                elif macd < macd_signal:  # Bearish crossover
                    sell_signals += 2
                total_signals += 2
            
            # Moving Average signals
            if 'sma_20' in current_indicators and 'sma_50' in current_indicators:
                sma_20 = current_indicators['sma_20']
                sma_50 = current_indicators['sma_50']
                if sma_20 > sma_50:  # Short MA above long MA
                    buy_signals += 1
                elif sma_20 < sma_50:  # Short MA below long MA
                    sell_signals += 1
                total_signals += 1
            
            # Bollinger Bands signals
            if all(k in current_indicators for k in ['bb_upper', 'bb_lower', 'close']):
                close = current_indicators['close']
                bb_upper = current_indicators['bb_upper']
                bb_lower = current_indicators['bb_lower']
                
                if close < bb_lower:  # Price below lower band
                    buy_signals += 1
                elif close > bb_upper:  # Price above upper band
                    sell_signals += 1
                total_signals += 1
            
            # Stochastic signals
            if 'stoch_k' in current_indicators and 'stoch_d' in current_indicators:
                stoch_k = current_indicators['stoch_k']
                stoch_d = current_indicators['stoch_d']
                
                if stoch_k < 20 and stoch_d < 20:  # Oversold
                    buy_signals += 1
                elif stoch_k > 80 and stoch_d > 80:  # Overbought
                    sell_signals += 1
                total_signals += 1
            
            # Determine final signal
            if total_signals == 0:
                return None
            
            buy_confidence = buy_signals / total_signals
            sell_confidence = sell_signals / total_signals
            
            if buy_confidence >= 0.6:
                signal_type = "buy"
                confidence = buy_confidence
            elif sell_confidence >= 0.6:
                signal_type = "sell"
                confidence = sell_confidence
            else:
                signal_type = "hold"
                confidence = 1 - max(buy_confidence, sell_confidence)
            
            # Create AI signal
            signal = AISignal(
                pair_id=pair_id,
                signal_type=signal_type,
                confidence=confidence,
                price_prediction=self._predict_price(indicators),
                indicators=current_indicators,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            return signal
            
        except Exception as e:
            logger.error("Failed to generate technical signal", pair_id=pair_id, error=str(e))
            return None
    
    def _prepare_features(self, indicators: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare features for ML model"""
        try:
            current_indicators = self._get_current_indicators(indicators)
            
            # Define feature list (should match training features)
            feature_names = [
                'rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50',
                'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle',
                'stoch_k', 'stoch_d', 'volume_sma'
            ]
            
            features = []
            for feature in feature_names:
                if feature in current_indicators and current_indicators[feature] is not None:
                    features.append(float(current_indicators[feature]))
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error("Failed to prepare features", error=str(e))
            return None
    
    def _get_current_indicators(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Get current (latest) values of indicators"""
        current = {}
        
        for key, values in indicators.items():
            if hasattr(values, 'iloc') and len(values) > 0:
                # Get last value from pandas Series
                current[key] = float(values.iloc[-1]) if not pd.isna(values.iloc[-1]) else 0.0
            elif isinstance(values, (list, np.ndarray)) and len(values) > 0:
                # Get last value from list/array
                current[key] = float(values[-1]) if not np.isnan(values[-1]) else 0.0
            elif isinstance(values, (int, float)):
                current[key] = float(values)
        
        return current
    
    def _predict_price(self, indicators: Dict[str, Any]) -> float:
        """Simple price prediction based on indicators"""
        try:
            current = self._get_current_indicators(indicators)
            
            if 'close' in current:
                # Simple prediction: use EMA trend
                if 'ema_12' in current and 'ema_26' in current:
                    ema_12 = current['ema_12']
                    ema_26 = current['ema_26']
                    
                    # If short EMA > long EMA, predict slight increase
                    if ema_12 > ema_26:
                        return current['close'] * 1.02  # 2% increase
                    else:
                        return current['close'] * 0.98  # 2% decrease
                
                return current['close']  # No change prediction
            
            return 0.0
            
        except Exception as e:
            logger.error("Failed to predict price", error=str(e))
            return 0.0
    
    async def _save_signal(self, signal: AISignal):
        """Save generated signal to database"""
        try:
            db = get_db()
            try:
                db.add(signal)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to save signal", error=str(e))
