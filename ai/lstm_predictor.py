"""
LSTM-based price prediction model for cryptocurrency trading
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import joblib
import os
import structlog

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("TensorFlow not available, LSTM predictor will use fallback methods")

from core.database import get_db, PriceHistory
from core.indodax_api import IndodaxAPI

logger = structlog.get_logger(__name__)

class LSTMPredictor:
    """LSTM-based cryptocurrency price predictor"""
    
    def __init__(self):
        self.model_path = "ai/models/lstm/"
        self.scalers_path = "ai/models/scalers/"
        self.lookback_days = 60  # Use 60 days of historical data
        self.models = {}
        self.scalers = {}
        self.api = IndodaxAPI()
        
        # Create directories if they don't exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.scalers_path, exist_ok=True)
        
        if TENSORFLOW_AVAILABLE:
            self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing trained models and scalers"""
        try:
            for file in os.listdir(self.model_path):
                if file.endswith('.h5'):
                    pair_id = file.replace('.h5', '')
                    model_path = os.path.join(self.model_path, file)
                    scaler_path = os.path.join(self.scalers_path, f"{pair_id}_scaler.pkl")
                    
                    if os.path.exists(scaler_path):
                        self.models[pair_id] = load_model(model_path)
                        self.scalers[pair_id] = joblib.load(scaler_path)
                        logger.info("Loaded LSTM model", pair_id=pair_id)
        except Exception as e:
            logger.error("Failed to load existing models", error=str(e))
    
    async def predict_price(self, pair_id: str, days_ahead: int = 1) -> Dict[str, Any]:
        """Predict future price for a cryptocurrency pair"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return await self._fallback_prediction(pair_id, days_ahead)
            
            # Check if model exists for this pair
            if pair_id not in self.models:
                logger.info("Training new LSTM model", pair_id=pair_id)
                await self._train_model(pair_id)
            
            if pair_id not in self.models:
                logger.warning("Model not available, using fallback", pair_id=pair_id)
                return await self._fallback_prediction(pair_id, days_ahead)
            
            # Get recent price data
            recent_data = await self._get_recent_price_data(pair_id, self.lookback_days)
            
            if recent_data is None or len(recent_data) < self.lookback_days:
                logger.warning("Insufficient data for prediction", pair_id=pair_id)
                return await self._fallback_prediction(pair_id, days_ahead)
            
            # Prepare data for prediction
            prediction_data = self._prepare_prediction_data(recent_data, pair_id)
            
            if prediction_data is None:
                return await self._fallback_prediction(pair_id, days_ahead)
            
            # Make prediction
            model = self.models[pair_id]
            scaler = self.scalers[pair_id]
            
            predictions = []
            current_data = prediction_data.copy()
            
            for _ in range(days_ahead):
                # Predict next day
                pred_scaled = model.predict(current_data.reshape(1, self.lookback_days, 1), verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(pred_price)
                
                # Update current_data for next prediction
                current_data = np.roll(current_data, -1)
                current_data[-1] = pred_scaled[0][0]
            
            current_price = recent_data['close'].iloc[-1]
            predicted_price = predictions[0] if predictions else current_price
            
            # Calculate confidence based on recent model performance
            confidence = await self._calculate_confidence(pair_id, recent_data)
            
            return {
                "pair_id": pair_id,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change": predicted_price - current_price,
                "price_change_percent": ((predicted_price - current_price) / current_price) * 100,
                "confidence": confidence,
                "predictions": predictions,
                "method": "lstm",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to predict price", pair_id=pair_id, error=str(e))
            return await self._fallback_prediction(pair_id, days_ahead)
    
    async def _train_model(self, pair_id: str) -> bool:
        """Train LSTM model for a specific pair"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return False
            
            logger.info("Starting LSTM model training", pair_id=pair_id)
            
            # Get historical data for training (6 months)
            training_data = await self._get_training_data(pair_id, days=180)
            
            if training_data is None or len(training_data) < 100:
                logger.warning("Insufficient training data", pair_id=pair_id)
                return False
            
            # Prepare training data
            X_train, y_train, scaler = self._prepare_training_data(training_data)
            
            if X_train is None:
                return False
            
            # Create LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.lookback_days, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                shuffle=False
            )
            
            # Save model and scaler
            model_file = os.path.join(self.model_path, f"{pair_id}.h5")
            scaler_file = os.path.join(self.scalers_path, f"{pair_id}_scaler.pkl")
            
            model.save(model_file)
            joblib.dump(scaler, scaler_file)
            
            # Store in memory
            self.models[pair_id] = model
            self.scalers[pair_id] = scaler
            
            logger.info("LSTM model trained successfully", 
                       pair_id=pair_id,
                       final_loss=history.history['loss'][-1])
            
            return True
            
        except Exception as e:
            logger.error("Failed to train LSTM model", pair_id=pair_id, error=str(e))
            return False
    
    async def _get_training_data(self, pair_id: str, days: int = 180) -> Optional[pd.DataFrame]:
        """Get historical data for model training"""
        try:
            db = get_db()
            try:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= cutoff_date
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                if len(price_data) < 50:
                    # Try to fetch from API
                    return await self._fetch_historical_from_api(pair_id, days)
                
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
            logger.error("Failed to get training data", pair_id=pair_id, error=str(e))
            return None
    
    async def _get_recent_price_data(self, pair_id: str, days: int) -> Optional[pd.DataFrame]:
        """Get recent price data for prediction"""
        try:
            db = get_db()
            try:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= cutoff_date
                ).order_by(PriceHistory.timestamp.desc()).limit(days).all()
                
                if len(price_data) < days:
                    return await self._fetch_historical_from_api(pair_id, days)
                
                data = []
                for record in reversed(price_data):  # Reverse to get chronological order
                    data.append({
                        'timestamp': record.timestamp,
                        'close': record.close_price
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                return df
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get recent price data", pair_id=pair_id, error=str(e))
            return None
    
    async def _fetch_historical_from_api(self, pair_id: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from Indodax API as fallback"""
        try:
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (days * 24 * 60 * 60)
            
            # Use 1-day candles for longer periods
            period = "1D" if days > 30 else "1H"
            
            ohlc_data = await self.api.get_ohlc_history(
                pair_id.upper(), start_time, end_time, period
            )
            
            if not ohlc_data:
                return None
            
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
            logger.error("Failed to fetch historical data from API", pair_id=pair_id, error=str(e))
            return None
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[MinMaxScaler]]:
        """Prepare data for LSTM training"""
        try:
            # Use close prices for training
            prices = df['close'].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(self.lookback_days, len(scaled_prices)):
                X.append(scaled_prices[i-self.lookback_days:i, 0])
                y.append(scaled_prices[i, 0])
            
            if len(X) < 10:  # Need at least 10 samples
                return None, None, None
            
            return np.array(X), np.array(y), scaler
            
        except Exception as e:
            logger.error("Failed to prepare training data", error=str(e))
            return None, None, None
    
    def _prepare_prediction_data(self, df: pd.DataFrame, pair_id: str) -> Optional[np.ndarray]:
        """Prepare recent data for prediction"""
        try:
            if pair_id not in self.scalers:
                return None
            
            # Get last 60 days of close prices
            prices = df['close'].tail(self.lookback_days).values.reshape(-1, 1)
            
            if len(prices) < self.lookback_days:
                return None
            
            # Scale using existing scaler
            scaler = self.scalers[pair_id]
            scaled_prices = scaler.transform(prices)
            
            return scaled_prices.flatten()
            
        except Exception as e:
            logger.error("Failed to prepare prediction data", error=str(e))
            return None
    
    async def _calculate_confidence(self, pair_id: str, recent_data: pd.DataFrame) -> float:
        """Calculate prediction confidence based on recent performance"""
        try:
            # Simple confidence calculation based on volatility
            if len(recent_data) < 10:
                return 0.5
            
            # Calculate recent volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Lower volatility = higher confidence
            if volatility < 0.02:  # Less than 2% daily volatility
                confidence = 0.8
            elif volatility < 0.05:  # Less than 5% daily volatility
                confidence = 0.6
            elif volatility < 0.1:  # Less than 10% daily volatility
                confidence = 0.4
            else:
                confidence = 0.2
            
            return min(max(confidence, 0.1), 0.9)  # Clamp between 0.1 and 0.9
            
        except Exception as e:
            logger.error("Failed to calculate confidence", error=str(e))
            return 0.5
    
    async def _fallback_prediction(self, pair_id: str, days_ahead: int = 1) -> Dict[str, Any]:
        """Fallback prediction using simple moving average"""
        try:
            # Get recent data
            recent_data = await self._get_recent_price_data(pair_id, 30)
            
            if recent_data is None or len(recent_data) < 10:
                return {
                    "pair_id": pair_id,
                    "current_price": 0.0,
                    "predicted_price": 0.0,
                    "price_change": 0.0,
                    "price_change_percent": 0.0,
                    "confidence": 0.1,
                    "predictions": [],
                    "method": "fallback",
                    "error": "Insufficient data"
                }
            
            current_price = recent_data['close'].iloc[-1]
            
            # Simple moving average prediction
            ma_7 = recent_data['close'].tail(7).mean()
            ma_14 = recent_data['close'].tail(14).mean()
            
            # Trend-based prediction
            if ma_7 > ma_14:
                predicted_price = current_price * 1.01  # Slight upward trend
            else:
                predicted_price = current_price * 0.99  # Slight downward trend
            
            return {
                "pair_id": pair_id,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change": predicted_price - current_price,
                "price_change_percent": ((predicted_price - current_price) / current_price) * 100,
                "confidence": 0.3,
                "predictions": [predicted_price],
                "method": "moving_average",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed fallback prediction", pair_id=pair_id, error=str(e))
            return {
                "pair_id": pair_id,
                "current_price": 0.0,
                "predicted_price": 0.0,
                "price_change": 0.0,
                "price_change_percent": 0.0,
                "confidence": 0.1,
                "predictions": [],
                "method": "error",
                "error": str(e)
            }
    
    async def retrain_model(self, pair_id: str) -> bool:
        """Retrain model with latest data"""
        try:
            logger.info("Retraining LSTM model", pair_id=pair_id)
            
            # Remove existing model
            if pair_id in self.models:
                del self.models[pair_id]
            if pair_id in self.scalers:
                del self.scalers[pair_id]
            
            # Train new model
            return await self._train_model(pair_id)
            
        except Exception as e:
            logger.error("Failed to retrain model", pair_id=pair_id, error=str(e))
            return False
