"""
LSTM-based price prediction model for cryptocurrency trading
with ensemble learning capabilities
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import os
import structlog

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = None

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    Adam = None

# Check for additional ML dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from core.database import get_db, PriceHistory
from core.indodax_api import IndodaxAPI

logger = structlog.get_logger(__name__)

class LSTMPredictor:
    """LSTM-based cryptocurrency price predictor with ensemble learning"""
    
    def __init__(self):
        self.model_path = "ai/models/lstm/"
        self.scalers_path = "ai/models/scalers/"
        self.ensemble_path = "ai/models/ensemble/"
        self.lookback_days = 60  # Use 60 days of historical data
        self.models = {}
        self.scalers = {}
        self.ensemble_models = {}
        self.api = IndodaxAPI()
        
        # Create directories if they don't exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.scalers_path, exist_ok=True)
        os.makedirs(self.ensemble_path, exist_ok=True)
        
        # Check available ML frameworks
        self.available_frameworks = {
            'tensorflow': TENSORFLOW_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'prophet': PROPHET_AVAILABLE,
            'pmdarima': PMDARIMA_AVAILABLE,
            'statsmodels': STATSMODELS_AVAILABLE
        }
        
        # Load models based on available frameworks
        self._load_existing_models()
        
        # Log available frameworks
        logger.info(
            "ML Frameworks availability",
            tensorflow=TENSORFLOW_AVAILABLE,
            sklearn=SKLEARN_AVAILABLE,
            xgboost=XGBOOST_AVAILABLE,
            lightgbm=LIGHTGBM_AVAILABLE,
            prophet=PROPHET_AVAILABLE,
            pmdarima=PMDARIMA_AVAILABLE,
            statsmodels=STATSMODELS_AVAILABLE
        )
    
    def _load_existing_models(self):
        """Load existing trained models and scalers"""
        try:
            if not TENSORFLOW_AVAILABLE or not SKLEARN_AVAILABLE:
                return
                
            for file in os.listdir(self.model_path):
                if file.endswith('.h5'):
                    pair_id = file.replace('.h5', '')
                    model_path = os.path.join(self.model_path, file)
                    scaler_path = os.path.join(self.scalers_path, f"{pair_id}_scaler.pkl")
                    
                    if os.path.exists(scaler_path) and load_model is not None:
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
    
    async def predict_ensemble(self, pair_id: str, lookback_days: int = 60) -> Dict[str, Any]:
        """Generate prediction using ensemble of models"""
        try:
            # Check if we have the necessary libraries
            available_models = []
            
            if TENSORFLOW_AVAILABLE and pair_id in self.models:
                available_models.append("lstm")
            
            if XGBOOST_AVAILABLE:
                available_models.append("xgboost")
                
            if LIGHTGBM_AVAILABLE:
                available_models.append("lightgbm")
                
            if PROPHET_AVAILABLE:
                available_models.append("prophet")
                
            if PMDARIMA_AVAILABLE or STATSMODELS_AVAILABLE:
                available_models.append("arima")
            
            if not available_models:
                logger.warning("No ML models available for ensemble prediction")
                return {
                    "status": "error",
                    "message": "No ML models available for prediction",
                    "prediction": 0,
                    "confidence": 0
                }
                
            # Get historical data
            historical_data = await self._get_historical_data(pair_id, lookback_days)
            
            if historical_data is None or len(historical_data) < lookback_days:
                logger.warning(f"Insufficient historical data for {pair_id}")
                return {
                    "status": "error",
                    "message": "Insufficient historical data",
                    "prediction": 0,
                    "confidence": 0
                }
            
            # Generate features
            df_features = self.generate_technical_features(historical_data)
            
            # Make predictions with each available model
            predictions = {}
            confidences = {}
            
            # LSTM prediction if available
            if "lstm" in available_models:
                lstm_result = await self._predict_lstm(pair_id, df_features)
                if lstm_result:
                    predictions["lstm"] = lstm_result["prediction"]
                    confidences["lstm"] = lstm_result["confidence"]
            
            # XGBoost prediction
            if "xgboost" in available_models:
                xgb_result = self._predict_xgboost(pair_id, df_features)
                if xgb_result:
                    predictions["xgboost"] = xgb_result["prediction"]
                    confidences["xgboost"] = xgb_result["confidence"]
            
            # LightGBM prediction
            if "lightgbm" in available_models:
                lgbm_result = self._predict_lightgbm(pair_id, df_features)
                if lgbm_result:
                    predictions["lightgbm"] = lgbm_result["prediction"]
                    confidences["lightgbm"] = lgbm_result["confidence"]
            
            # Prophet prediction
            if "prophet" in available_models:
                prophet_result = self._predict_prophet(pair_id, historical_data)
                if prophet_result:
                    predictions["prophet"] = prophet_result["prediction"]
                    confidences["prophet"] = prophet_result["confidence"]
            
            # ARIMA prediction
            if "arima" in available_models:
                arima_result = self._predict_arima(pair_id, historical_data)
                if arima_result:
                    predictions["arima"] = arima_result["prediction"]
                    confidences["arima"] = arima_result["confidence"]
            
            # If no predictions were made, return error
            if not predictions:
                logger.warning(f"No valid predictions for {pair_id}")
                return {
                    "status": "error",
                    "message": "No valid predictions",
                    "prediction": 0,
                    "confidence": 0
                }
            
            # Calculate ensemble prediction (weighted average)
            weighted_sum = 0
            total_weight = 0
            
            for model, pred in predictions.items():
                conf = confidences.get(model, 0.5)
                weighted_sum += pred * conf
                total_weight += conf
            
            if total_weight > 0:
                ensemble_prediction = weighted_sum / total_weight
            else:
                ensemble_prediction = 0
                
            # Calculate confidence as weighted average of individual confidences
            ensemble_confidence = sum(confidences.values()) / len(confidences) if confidences else 0.5
            
            # Calculate consistency between models
            consistency = 1.0
            if len(predictions) > 1:
                # Check if all predictions have same direction
                directions = [pred > 0 for pred in predictions.values()]
                if all(directions) or not any(directions):
                    consistency = 1.0
                else:
                    # Calculate standard deviation of normalized predictions
                    normalized_preds = []
                    for pred in predictions.values():
                        normalized_preds.append(pred)
                    
                    std_dev = np.std(normalized_preds)
                    # Higher std dev means less consistent (lower consistency)
                    consistency = max(0.3, 1.0 - std_dev)
            
            # Adjust confidence based on consistency
            adjusted_confidence = ensemble_confidence * consistency
            
            # Create detailed response
            result = {
                "status": "success",
                "prediction": ensemble_prediction,
                "confidence": adjusted_confidence,
                "predictions": predictions,
                "confidences": confidences,
                "consistency": consistency,
                "models_used": list(predictions.keys())
            }
            
            logger.info(f"Ensemble prediction for {pair_id}", 
                       prediction=f"{ensemble_prediction:.6f}", 
                       confidence=f"{adjusted_confidence:.4f}",
                       models_used=len(predictions))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "prediction": 0,
                "confidence": 0
            }
    
    async def _train_model(self, pair_id: str) -> bool:
        """Train LSTM model for a specific pair"""
        try:
            if not TENSORFLOW_AVAILABLE or not SKLEARN_AVAILABLE:
                logger.warning("TensorFlow or Scikit-learn not available for training", pair_id=pair_id)
                return False
            
            if Sequential is None or LSTM is None or Dense is None or Dropout is None or Adam is None:
                logger.warning("TensorFlow components not available for training", pair_id=pair_id)
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
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
        """Prepare data for LSTM training"""
        try:
            # Use close prices for training
            prices = np.array(df['close'].values).reshape(-1, 1)
            
            # Scale data
            if not SKLEARN_AVAILABLE or MinMaxScaler is None:
                return None, None, None
                
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
            prices = np.array(df['close'].tail(self.lookback_days).values).reshape(-1, 1)
            
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
    
    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical features for prediction models"""
        try:
            # Create a copy of dataframe to avoid modifying the original
            df_features = df.copy()
            
            # Basic price features
            df_features['returns'] = df_features['close'].pct_change()
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
            
            # Volatility features
            df_features['volatility_5d'] = df_features['returns'].rolling(window=5).std()
            df_features['volatility_10d'] = df_features['returns'].rolling(window=10).std()
            df_features['volatility_20d'] = df_features['returns'].rolling(window=20).std()
            
            # Price momentum at different timeframes
            for window in [1, 3, 5, 10, 20]:
                df_features[f'momentum_{window}d'] = df_features['close'].pct_change(periods=window)
            
            # Simple moving averages
            for window in [5, 10, 20, 50, 100]:
                df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
                
                # SMA ratio to current price
                df_features[f'price_sma_{window}_ratio'] = df_features['close'] / df_features[f'sma_{window}']
            
            # Exponential moving averages
            for window in [5, 10, 20, 50]:
                df_features[f'ema_{window}'] = df_features['close'].ewm(span=window, adjust=False).mean()
            
            # Moving average crossovers
            df_features['sma_5_10_cross'] = df_features['sma_5'] - df_features['sma_10']
            df_features['sma_10_20_cross'] = df_features['sma_10'] - df_features['sma_20']
            df_features['sma_20_50_cross'] = df_features['sma_20'] - df_features['sma_50']
            
            # MACD
            df_features['ema_12'] = df_features['close'].ewm(span=12, adjust=False).mean()
            df_features['ema_26'] = df_features['close'].ewm(span=26, adjust=False).mean()
            df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
            df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
            df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
            
            # Volume features
            df_features['volume_sma_5'] = df_features['volume'].rolling(window=5).mean()
            df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']
            df_features['volume_change'] = df_features['volume'].pct_change()
            
            # Price and volume relationship - calculate correlation after computing rolling values
            returns_10d = df_features['returns'].rolling(window=10)
            volume_change_10d = df_features['volume_change'].rolling(window=10)
            # Calculate correlation between the two series using pairwise method
            df_features['price_volume_corr'] = df_features['returns'].rolling(window=10).corr(df_features['volume_change'])
            
            # Try to use TA library if available
            try:
                import ta
                from ta.momentum import RSIIndicator, StochasticOscillator
                from ta.volatility import BollingerBands, AverageTrueRange
                from ta.trend import ADXIndicator
                
                # RSI
                rsi_indicator = RSIIndicator(df_features['close'], window=14)
                df_features['rsi_14'] = rsi_indicator.rsi()
                
                # Bollinger Bands
                bollinger_bands = BollingerBands(df_features['close'], window=20, window_dev=2)
                df_features['bb_high'] = bollinger_bands.bollinger_hband()
                df_features['bb_low'] = bollinger_bands.bollinger_lband()
                df_features['bb_mid'] = bollinger_bands.bollinger_mavg()
                df_features['bb_width'] = (df_features['bb_high'] - df_features['bb_low']) / df_features['bb_mid']
                
                # Stochastic Oscillator
                stoch_indicator = StochasticOscillator(
                    df_features['high'], df_features['low'], df_features['close'], window=14, smooth_window=3
                )
                df_features['stoch_k'] = stoch_indicator.stoch()
                df_features['stoch_d'] = stoch_indicator.stoch_signal()
                
                # ATR
                atr_indicator = AverageTrueRange(
                    df_features['high'], df_features['low'], df_features['close'], window=14
                )
                df_features['atr'] = atr_indicator.average_true_range()
                
                # ADX
                adx_indicator = ADXIndicator(
                    df_features['high'], df_features['low'], df_features['close'], window=14
                )
                df_features['adx'] = adx_indicator.adx()
                
                # VWAP
                df_features['vwap_daily'] = (df_features['volume'] * df_features['close']).cumsum() / df_features['volume'].cumsum()
                
                logger.info("Using TA library for technical indicators")
                
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                logger.warning(f"TA library not available or error: {str(e)}, using manual calculations")
                
                # Simplified RSI calculation
                delta = df_features['close'].diff()
                
                # Calculate gains and losses without using boolean indexing
                gain = delta.apply(lambda x: x if x > 0 else 0)
                loss = delta.apply(lambda x: abs(x) if x < 0 else 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # Avoid division by zero
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df_features['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Simple ATR calculation
                high_low = df_features['high'] - df_features['low']
                high_close = (df_features['high'] - df_features['close'].shift()).abs()
                low_close = (df_features['low'] - df_features['close'].shift()).abs()
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df_features['atr'] = true_range.rolling(window=14).mean()
            
            # Add day of week if timestamp is available
            if 'timestamp' in df_features.columns and pd.api.types.is_datetime64_any_dtype(df_features['timestamp']):
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            
            # Fill NaN values using backfill then forward fill
            df_features = df_features.bfill().ffill()
            
            # If any NaN values remain, replace with 0
            if df_features.isna().sum().sum() > 0:
                df_features = df_features.fillna(0)
                
            return df_features
            
        except Exception as e:
            logger.error(f"Error generating technical features: {str(e)}")
            
            # Return the original DataFrame with basic features if there's an error
            df_features = df.copy()
            df_features['returns'] = df_features['close'].pct_change().fillna(0)
            return df_features
    
    async def _predict_lstm(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using LSTM model"""
        if not TENSORFLOW_AVAILABLE or pair_id not in self.models:
            return None
            
        try:
            # Get model and scaler
            model = self.models[pair_id]
            scaler = self.scalers[pair_id]
            
            # Prepare data for LSTM
            sequence_length = 60
            
            if len(df_features) < sequence_length:
                logger.warning(f"Not enough data for LSTM prediction on {pair_id}")
                return None
                
            # Extract relevant features for LSTM
            features = [
                'close', 'returns', 'volatility_10d', 'sma_10', 'sma_20',
                'ema_12', 'macd', 'rsi_14'
            ]
            
            # Use only available features
            available_features = [f for f in features if f in df_features.columns]
            
            if len(available_features) < 5:
                logger.warning("Not enough features for LSTM prediction")
                return None
                
            # Scale features
            data = df_features[available_features].values
            scaled_data = scaler.transform(data)
            
            # Create sequence for LSTM
            x_test = []
            x_test.append(scaled_data[-sequence_length:])
            x_test = np.array(x_test)
            
            # Make prediction
            prediction = model.predict(x_test, verbose=0)
            
            # Get predicted price change
            last_price = df_features['close'].iloc[-1]
            predicted_change = prediction[0][0]
            
            # Calculate confidence based on model loss history
            if hasattr(model, 'history') and 'val_loss' in model.history:
                val_loss = model.history['val_loss'][-1]
                confidence = max(0.5, min(0.9, 1.0 - val_loss))
            else:
                # Default confidence
                confidence = 0.7
            
            return {
                "prediction": predicted_change,
                "confidence": confidence,
                "last_price": last_price
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            return None
    
    def _predict_xgboost(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return None
            
        try:
            import xgboost as xgb
            
            # Check if we have a trained model
            model_path = os.path.join(self.ensemble_path, f"{pair_id}_xgboost.json")
            
            if not os.path.exists(model_path):
                # Try to create a simple model on-the-fly for testing
                return self._create_simple_xgboost_model(pair_id, df_features)
            
            # Load the model
            model = xgb.Booster()
            model.load_model(model_path)
            
            # Prepare features
            features = ['returns', 'volatility_10d', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi_14']
            
            # Use only available features
            available_features = [f for f in features if f in df_features.columns]
            
            if len(available_features) < 4:
                logger.warning("Not enough features for XGBoost prediction")
                return None
                
            # Get latest data point
            latest_data = df_features[available_features].iloc[-1:].values
            
            # Create DMatrix
            dtest = xgb.DMatrix(latest_data)
            
            # Make prediction
            prediction = model.predict(dtest)[0]
            
            # Last price
            last_price = df_features['close'].iloc[-1]
            
            # Fixed confidence for now
            confidence = 0.65
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "last_price": last_price
            }
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {str(e)}")
            return None
    
    def _create_simple_xgboost_model(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create a simple XGBoost model for simulation"""
        if not XGBOOST_AVAILABLE:
            return None
            
        try:
            import xgboost as xgb
            
            # Check if we have enough data
            if len(df_features) < 100:
                return None
                
            # Calculate target: next day return
            df_features['target'] = df_features['close'].shift(-1) / df_features['close'] - 1
            
            # Drop NaN
            df_features.dropna(inplace=True)
            
            # Features for simple model
            features = ['returns', 'volatility_10d', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi_14']
            
            # Use only available features
            available_features = [f for f in features if f in df_features.columns]
            
            # Split data: use last 20% for validation
            split_idx = int(len(df_features) * 0.8)
            X_train = df_features[available_features].values[:split_idx]
            y_train = df_features['target'].values[:split_idx]
            
            # Create simple model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            params = {
                'objective': 'reg:squarederror',
                'eta': 0.1,
                'max_depth': 3,
                'subsample': 0.7,
                'colsample_bytree': 0.7
            }
            
            # Train a very simple model
            model = xgb.train(params, dtrain, num_boost_round=20)
            
            # Get latest data point
            latest_data = df_features[available_features].iloc[-1:].values
            dtest = xgb.DMatrix(latest_data)
            
            # Make prediction
            prediction = model.predict(dtest)[0]
            
            # Save model for future use
            os.makedirs(self.ensemble_path, exist_ok=True)
            model_path = os.path.join(self.ensemble_path, f"{pair_id}_xgboost.json")
            model.save_model(model_path)
            
            # Last price
            last_price = df_features['close'].iloc[-1]
            
            # Lower confidence for simple model
            confidence = 0.55
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "last_price": last_price
            }
            
        except Exception as e:
            logger.error(f"Error creating simple XGBoost model: {str(e)}")
            return None
    
    def _predict_lightgbm(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            return None
            
        try:
            import lightgbm as lgb
            
            # Check if we have a trained model
            model_path = os.path.join(self.ensemble_path, f"{pair_id}_lightgbm.txt")
            
            if not os.path.exists(model_path):
                # Try to create a simple model on-the-fly
                return self._create_simple_lightgbm_model(pair_id, df_features)
                
            # Load model
            model = lgb.Booster(model_file=model_path)
            
            # Prepare features
            features = ['returns', 'volatility_10d', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi_14']
            
            # Use only available features
            available_features = [f for f in features if f in df_features.columns]
            
            if len(available_features) < 4:
                logger.warning("Not enough features for LightGBM prediction")
                return None
                
            # Get latest data point - ensure proper float64 data type
            latest_data = df_features[available_features].iloc[-1:].values.astype(np.float64)
            
            # Make prediction
            prediction = model.predict(latest_data)[0]
            
            # Last price
            last_price = df_features['close'].iloc[-1]
            
            # Fixed confidence
            confidence = 0.65
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "last_price": last_price
            }
            
        except Exception as e:
            logger.error(f"Error in LightGBM prediction: {str(e)}")
            return None
    
    def _create_simple_lightgbm_model(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create a simple LightGBM model for simulation"""
        if not LIGHTGBM_AVAILABLE:
            return None
            
        try:
            import lightgbm as lgb
            
            # Check if we have enough data
            if len(df_features) < 100:
                return None
                
            # Calculate target: next day return
            df_features['target'] = df_features['close'].shift(-1) / df_features['close'] - 1
            
            # Drop NaN
            df_features.dropna(inplace=True)
            
            # Features for simple model
            features = ['returns', 'volatility_10d', 'sma_10', 'sma_20', 'ema_12', 'macd', 'rsi_14']
            
            # Use only available features
            available_features = [f for f in features if f in df_features.columns]
            
            # Split data
            split_idx = int(len(df_features) * 0.8)
            X_train = df_features[available_features].values[:split_idx].astype(np.float64)
            y_train = df_features['target'].values[:split_idx].astype(np.float64)
            
            # Create dataset - explicitly convert to numpy arrays and proper types
            train_data = lgb.Dataset(X_train, label=y_train.tolist())
            
            # Simple parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
            
            # Train model
            model = lgb.train(params, train_data, num_boost_round=20)
            
            # Save model
            os.makedirs(self.ensemble_path, exist_ok=True)
            model_path = os.path.join(self.ensemble_path, f"{pair_id}_lightgbm.txt")
            model.save_model(model_path)
            
            # Make prediction - ensure proper data type for prediction
            latest_data = df_features[available_features].iloc[-1:].values.astype(np.float64)
            prediction = model.predict(latest_data)[0]
            
            # Last price
            last_price = df_features['close'].iloc[-1]
            
            # Lower confidence for simple model
            confidence = 0.55
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "last_price": last_price
            }
            
        except Exception as e:
            logger.error(f"Error creating simple LightGBM model: {str(e)}")
            return None
    def _predict_arima(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using ARIMA model"""
        if not PMDARIMA_AVAILABLE and not STATSMODELS_AVAILABLE:
            logger.warning("Neither pmdarima nor statsmodels are available for ARIMA modeling")
            return None
            
        try:
            # Check if we have enough data
            if len(df_features) < 60:
                logger.warning(f"Not enough data for ARIMA prediction: {len(df_features)} < 60 points")
                return None
                
            # Get price data - we'll work with closing prices
            prices = df_features['close'].values
            last_price = df_features['close'].iloc[-1]
            
            # Try to load cached model first
            model_path = os.path.join(self.ensemble_path, f"{pair_id}_arima.pkl")
            cached_model = None
            
            if os.path.exists(model_path):
                try:
                    cached_model = joblib.load(model_path)
                    logger.info(f"Loaded cached ARIMA model for {pair_id}")
                except Exception as e:
                    logger.warning(f"Could not load cached ARIMA model: {str(e)}")
            
            predicted_price = None
            forecast_error = None
            model_quality = 0.6  # Default model quality
            
            if PMDARIMA_AVAILABLE:
                from pmdarima import auto_arima
                
                try:
                    arima_model = None
                    
                    if cached_model is not None:
                        arima_model = cached_model
                    else:
                        # Fit new model
                        logger.info(f"Fitting new ARIMA model for {pair_id} using auto_arima")
                        arima_model = auto_arima(
                            prices, 
                            start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                            start_P=0, seasonal=False, d=1, D=0, trace=False,
                            error_action='ignore', suppress_warnings=True, 
                            stepwise=True,
                            max_order=5  # Limit total order to prevent overfitting
                        )
                        
                        # Save model for future use
                        os.makedirs(self.ensemble_path, exist_ok=True)
                        joblib.dump(arima_model, model_path)
                    
                    if arima_model is not None:
                        # Get model order for logging
                        order = getattr(arima_model, 'order', None)
                        if order:
                            logger.info(f"ARIMA model order for {pair_id}: {order}")
                        
                        # Forecast
                        forecast, conf_int = arima_model.predict(n_periods=1, return_conf_int=True)
                        predicted_price = forecast[0]
                        
                        # Calculate forecast error from confidence interval width
                        forecast_error = (conf_int[0][1] - conf_int[0][0]) / (2 * last_price)
                        
                        # Get AIC for model quality assessment if available
                        if hasattr(arima_model, 'aic'):
                            try:
                                model_quality = 0.7 - min(0.2, arima_model.aic() / 100000)
                            except:
                                pass
                except Exception as e:
                    logger.error(f"Error in pmdarima prediction: {str(e)}")
                    # Continue to try statsmodels if pmdarima fails
            
            # If pmdarima failed or isn't available, try statsmodels
            if predicted_price is None and STATSMODELS_AVAILABLE:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    
                    # Try a few different orders if needed
                    orders = [(1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1)]
                    
                    best_aic = float('inf')
                    best_model = None
                    best_order = None
                    
                    for order in orders:
                        try:
                            # Fit model with this order
                            model = ARIMA(prices, order=order)
                            model_fit = model.fit()
                            
                            # Check if this is the best model
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_model = model_fit
                                best_order = order
                        except:
                            continue
                    
                    if best_model is not None:
                        logger.info(f"Best ARIMA order for {pair_id}: {best_order} with AIC: {best_aic:.2f}")
                        
                        # Forecast
                        forecast = best_model.forecast(steps=1)
                        predicted_price = forecast[0]
                        
                        # Calculate model quality from AIC
                        model_quality = 0.6 - min(0.1, best_aic / 100000)
                    else:
                        logger.warning(f"Could not fit any ARIMA model for {pair_id}")
                        return None
                except Exception as e:
                    logger.error(f"Error in statsmodels ARIMA prediction: {str(e)}")
                    return None
                
            if predicted_price is None:
                logger.warning(f"Failed to get ARIMA prediction for {pair_id}")
                return None
                
            # Calculate predicted change
            predicted_change = (predicted_price / last_price) - 1
            
            # Base confidence calculation
            confidence = model_quality  # Start with model quality score
            
            # Adjust confidence based on forecast error if available
            if forecast_error is not None:
                # Higher error = lower confidence
                error_factor = max(0.3, min(1.0, 1.0 - (forecast_error * 5)))
                confidence = (confidence + error_factor) / 2
            
            # Adjust confidence based on data quality
            if len(df_features) > 200:
                confidence += 0.05
                
            # Check for stationarity of returns
            if 'returns' in df_features:
                returns = df_features['returns'].dropna()
                if len(returns) > 30:
                    mean = returns.mean()
                    std = returns.std()
                    if abs(mean) < 0.01 and std < 0.05:
                        # More stationary data = higher confidence
                        confidence += 0.05
                    
            # Check recent volatility - higher volatility = lower confidence
            if 'volatility_10d' in df_features:
                recent_vol = df_features['volatility_10d'].iloc[-1]
                if recent_vol > 0.02:
                    confidence -= min(0.1, recent_vol)
                
            return {
                "prediction": predicted_change,
                "confidence": min(max(0.4, confidence), 0.8),  # Clamp between 0.4 and 0.8
                "last_price": last_price,
                "predicted_price": predicted_price
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {str(e)}")
            return None
    
    def _predict_prophet(self, pair_id: str, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using Facebook Prophet model"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet library not available for prediction")
            return None
            
        try:
            from prophet import Prophet
            
            # Check if we have enough data
            if len(df_features) < 30:
                logger.warning(f"Not enough data for Prophet prediction: {len(df_features)} < 30 points")
                return None
                
            # Prepare data for Prophet
            prophet_df = pd.DataFrame()
            
            # Handle timestamp column - Prophet needs 'ds' column
            if df_features.index.name == 'timestamp':
                prophet_df['ds'] = df_features.index
            elif 'timestamp' in df_features.columns:
                prophet_df['ds'] = pd.to_datetime(df_features['timestamp'])
            else:
                # Create synthetic dates if no timestamp column exists
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(df_features))
                prophet_df['ds'] = pd.date_range(start=start_date, periods=len(df_features), freq='D')
                
            # Prophet needs 'y' column for target values
            prophet_df['y'] = df_features['close']
            
            # Make sure there are no NaN values
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 30:
                logger.warning("Not enough valid data points for Prophet after cleaning")
                return None
                
            # Create and fit model with settings appropriate for crypto
            model = Prophet(
                changepoint_prior_scale=0.05,  # Flexibility of the trend
                seasonality_prior_scale=0.1,   # Flexibility of the seasonality
                changepoint_range=0.9,         # Proportion of history in which trend changes are allowed
                interval_width=0.95            # Width of uncertainty intervals
            )
            
            # Add seasonalities with appropriate Fourier orders
            model.add_seasonality(name='daily', period=1, fourier_order=10)
            model.add_seasonality(name='weekly', period=7, fourier_order=5)
            model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
            
            # Add additional seasonality that might be relevant to crypto
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            # Fit the model
            try:
                model.fit(prophet_df)
            except Exception as fit_error:
                logger.warning(f"Error fitting Prophet model: {str(fit_error)}")
                # Try a simpler model if the first attempt fails
                try:
                    model = Prophet(
                        changepoint_prior_scale=0.05,
                        interval_width=0.95
                    )
                    model.fit(prophet_df)
                    logger.info("Using simplified Prophet model")
                except Exception as e:
                    logger.error(f"Failed to fit simplified Prophet model: {str(e)}")
                    return None
            
            # Make forecast
            future = model.make_future_dataframe(periods=7)  # Predict up to 7 days ahead
            forecast = model.predict(future)
            
            # Get predictions
            last_price = df_features['close'].iloc[-1]
            tomorrow_forecast = forecast[forecast['ds'] > prophet_df['ds'].iloc[-1]].iloc[0]
            predicted_price = tomorrow_forecast['yhat']
            predicted_change = (predicted_price / last_price) - 1
            
            # Calculate confidence
            lower_bound = tomorrow_forecast['yhat_lower']
            upper_bound = tomorrow_forecast['yhat_upper']
            
            # Width of prediction interval affects confidence
            price_range = upper_bound - lower_bound
            price_range_pct = price_range / last_price
            
            # Narrower range = higher confidence
            confidence_from_range = max(0.4, 1.0 - (price_range_pct * 5))
            
            # Base confidence
            base_confidence = 0.65
            
            # Historical performance - check if model could have predicted the last few days correctly
            if len(forecast) > 5 and len(prophet_df) > 5:
                last_predictions = forecast['yhat'].iloc[-len(prophet_df):-1]
                last_actuals = prophet_df['y'].iloc[1:]
                
                # Calculate directional accuracy (up/down)
                pred_directions = np.sign(np.diff(last_predictions))
                actual_directions = np.sign(np.diff(last_actuals))
                
                # Count matches
                matches = sum(pred_directions == actual_directions)
                direction_accuracy = matches / len(pred_directions) if len(pred_directions) > 0 else 0.5
                
                # Adjust base confidence based on historical performance
                historical_confidence = 0.5 + (direction_accuracy - 0.5) * 2  # Scale to 0.5-1.0
                base_confidence = (base_confidence + historical_confidence) / 2
            
            # Final confidence
            confidence = (base_confidence + confidence_from_range) / 2
            
            result = {
                "prediction": predicted_change,
                "confidence": min(confidence, 0.85),  # Cap at 0.85
                "last_price": last_price,
                "predicted_price": predicted_price,
                "forecast_range": {
                    "lower": lower_bound,
                    "upper": upper_bound
                }
            }
            
            # Store additional forecast data for possible visualization
            week_ahead = forecast[forecast['ds'] > prophet_df['ds'].iloc[-1]].iloc[:7]
            result["extended_forecast"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in week_ahead['ds']],
                "values": week_ahead['yhat'].tolist(),
                "lower_bounds": week_ahead['yhat_lower'].tolist(),
                "upper_bounds": week_ahead['yhat_upper'].tolist()
            }
            
            logger.info(f"Prophet prediction for {pair_id}: {predicted_change:.4%} change, confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {str(e)}")
            return None
    
    async def _get_historical_data(self, pair_id: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Get historical data for prediction"""
        try:
            # Try to get data from database first
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)  # Get more data than needed
            
            from core.database import get_db, PriceHistory
            
            db = get_db()
            try:
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= start_date,
                    PriceHistory.timestamp <= end_date
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                if len(price_data) >= lookback_days:
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
                    df = df.sort_index()
                    return df
            finally:
                db.close()
            
            # If not enough data in database, try API
            logger.info(f"Fetching historical data from API for {pair_id}")
            
            # Convert pair_id format (btc_idr -> BTCIDR)
            symbol = pair_id.replace("_", "").upper()
            
            # Get data from API
            api_data = await self.api.get_historical_data(
                symbol=symbol,
                from_timestamp=int(start_date.timestamp()),
                to_timestamp=int(end_date.timestamp()),
                timeframe="1D"
            )
            
            if not api_data:
                logger.warning(f"No data returned from API for {pair_id}")
                return None
                
            # Convert to DataFrame
            data = []
            for record_item in api_data:
                try:
                    timestamp = None
                    open_price = 0.0
                    high_price = 0.0
                    low_price = 0.0
                    close_price = 0.0
                    volume = 0.0
                    
                    # Handle dictionary type response
                    if isinstance(record_item, dict):
                        # Handle different possible key formats for timestamp
                        timestamp_key = None
                        for key in ['timestamp', 'time', 'date', 'Time']:
                            if key in record_item:
                                timestamp_key = key
                                break
                                
                        if timestamp_key:
                            ts_value = record_item[timestamp_key]
                            if isinstance(ts_value, (int, float)):
                                timestamp = datetime.fromtimestamp(ts_value / 1000 if ts_value > 1e10 else ts_value)
                            else:
                                try:
                                    timestamp = pd.to_datetime(ts_value)
                                except:
                                    pass
                        
                        # Get price data with fallbacks for different key formats
                        for key in ['open', 'Open']:
                            if key in record_item and record_item[key] is not None:
                                try:
                                    open_price = float(record_item[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                                    
                        for key in ['high', 'High']:
                            if key in record_item and record_item[key] is not None:
                                try:
                                    high_price = float(record_item[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                                    
                        for key in ['low', 'Low']:
                            if key in record_item and record_item[key] is not None:
                                try:
                                    low_price = float(record_item[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                                    
                        for key in ['close', 'Close']:
                            if key in record_item and record_item[key] is not None:
                                try:
                                    close_price = float(record_item[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                                    
                        for key in ['volume', 'Volume']:
                            if key in record_item and record_item[key] is not None:
                                try:
                                    volume = float(record_item[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                    
                    # Handle list type response - use a safer approach without direct indexing
                    elif isinstance(record_item, list):
                        values = []
                        for i, val in enumerate(record_item):
                            if i >= 6:  # We only need the first 6 values
                                break
                            values.append(val)
                            
                        # Pad with None if the list is too short
                        while len(values) < 6:
                            values.append(None)
                            
                        # Now process the values
                        try:
                            if values[0] is not None and isinstance(values[0], (int, float)):
                                timestamp = datetime.fromtimestamp(values[0] / 1000 if values[0] > 1e10 else values[0])
                        except (IndexError, TypeError, ValueError):
                            pass
                            
                        try:
                            if values[1] is not None:
                                open_price = float(values[1])
                        except (IndexError, TypeError, ValueError):
                            pass
                            
                        try:
                            if values[2] is not None:
                                high_price = float(values[2])
                        except (IndexError, TypeError, ValueError):
                            pass
                            
                        try:
                            if values[3] is not None:
                                low_price = float(values[3])
                        except (IndexError, TypeError, ValueError):
                            pass
                            
                        try:
                            if values[4] is not None:
                                close_price = float(values[4])
                        except (IndexError, TypeError, ValueError):
                            pass
                            
                        try:
                            if values[5] is not None:
                                volume = float(values[5])
                        except (IndexError, TypeError, ValueError):
                            pass
                    
                    # Skip if we couldn't parse a timestamp
                    if timestamp is None:
                        logger.debug("Skipping record without valid timestamp")
                        continue
                        
                    # Ensure we have valid price data (at least close price)
                    if close_price <= 0:
                        logger.debug("Skipping record with invalid close price")
                        continue
                        
                    data.append({
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })
                except Exception as e:
                    logger.warning(f"Error parsing API record: {str(e)}")
                    continue
                    
            if not data:
                logger.warning(f"No valid records from API for {pair_id}")
                return None
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
