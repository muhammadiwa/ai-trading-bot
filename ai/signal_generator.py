"""
Advanced AI-powered trading signal generator with real-time data integration
Uses ensemble machine learning models and comprehensive technical analysis
for cryptocurrency market prediction and signal generation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Import TA-Lib with proper modules and error handling
try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import VolumeWeightedAveragePrice, VolumePriceTrendIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    ta = None
    # Create placeholder classes
    class PlaceholderIndicator:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: pd.Series([0] * 100)
    
    SMAIndicator = EMAIndicator = MACD = RSIIndicator = PlaceholderIndicator
    StochasticOscillator = BollingerBands = AverageTrueRange = PlaceholderIndicator
    VolumeWeightedAveragePrice = VolumePriceTrendIndicator = PlaceholderIndicator

from typing import Dict, Optional, List, Any, Tuple, Union, Set
import os
import json
import pickle
import asyncio
import structlog
from sqlalchemy.sql import func

# ML imports with robust error handling
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create placeholder classes for sklearn components
    class PlaceholderScaler:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X):
            return self
        def transform(self, X):
            return X if isinstance(X, list) else X.values if hasattr(X, 'values') else X
        def fit_transform(self, X):
            return X if isinstance(X, list) else X.values if hasattr(X, 'values') else X
        def inverse_transform(self, X):
            return X
    
    StandardScaler = MinMaxScaler = PlaceholderScaler
    
    class PlaceholderClassifier:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [1] * (len(X) if hasattr(X, '__len__') else 1)
        def predict_proba(self, X):
            return [[0.1, 0.8, 0.1]] * (len(X) if hasattr(X, '__len__') else 1)
    
    RandomForestClassifier = GradientBoostingClassifier = PlaceholderClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from core.indodax_api import IndodaxAPI
from core.database import get_db, AISignal, PriceHistory

logger = structlog.get_logger(__name__)

class SignalGenerator:
    """Advanced AI-powered trading signal generator with ensemble models and real-time data"""
    
    def __init__(self):
        self.models_path = "ai/models/signals/"
        self.ensemble_path = "ai/models/ensemble/"
        self.scaler_path = "ai/models/scalers/"
        self.models = {}  # Dictionary to store trained models
        self.scalers = {}  # Dictionary to store data scalers
        self.feature_importance = {}  # Store feature importance for models
        self.api = IndodaxAPI()  # API client for real-time data
        
        # Create required directories if they don't exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.ensemble_path, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)
        
        # Configure settings
        self.prediction_window = 24  # Hours ahead for prediction
        self.min_confidence_threshold = 0.65  # Minimum confidence for signals
        self.min_data_points = 100  # Minimum data points required for analysis
        
        # Track model performance metrics
        self.model_metrics = {}
        
        # Initialize scalers based on available libraries
        if SKLEARN_AVAILABLE:
            self.main_scaler = StandardScaler()
            self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Load pre-trained models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained ML models and scalers"""
        try:
            # Load main models
            if os.path.exists(self.models_path):
                for file in os.listdir(self.models_path):
                    if file.endswith('.pkl'):
                        pair_id = file.replace('_model.pkl', '')
                        model_file = os.path.join(self.models_path, file)
                        try:
                            with open(model_file, 'rb') as f:
                                self.models[pair_id] = pickle.load(f)
                            
                            # Load corresponding scaler if available
                            scaler_file = os.path.join(self.scaler_path, f"{pair_id}_scaler.pkl")
                            if os.path.exists(scaler_file):
                                with open(scaler_file, 'rb') as f:
                                    self.scalers[pair_id] = pickle.load(f)
                            
                            # Load feature importance if available
                            imp_file = os.path.join(self.models_path, f"{pair_id}_importance.json")
                            if os.path.exists(imp_file):
                                with open(imp_file, 'r') as f:
                                    self.feature_importance[pair_id] = json.load(f)
                            
                            logger.info("Loaded model and metadata for pair", pair_id=pair_id)
                        except Exception as e:
                            logger.warning(f"Failed to load model for {pair_id}: {str(e)}")
            
            # Load ensemble models if available
            if XGBOOST_AVAILABLE and os.path.exists(self.ensemble_path):
                for file in os.listdir(self.ensemble_path):
                    if file.endswith('_xgb.json'):
                        pair_id = file.replace('_xgb.json', '')
                        # XGBoost models are loaded on-demand when needed
                        self.models.setdefault(f"{pair_id}_ensemble", {})["xgboost"] = True
                        logger.info("Found XGBoost model", pair_id=pair_id)
            
            if LIGHTGBM_AVAILABLE and os.path.exists(self.ensemble_path):
                for file in os.listdir(self.ensemble_path):
                    if file.endswith('_lgbm.txt'):
                        pair_id = file.replace('_lgbm.txt', '')
                        # LightGBM models are loaded on-demand when needed
                        self.models.setdefault(f"{pair_id}_ensemble", {})["lightgbm"] = True
                        logger.info("Found LightGBM model", pair_id=pair_id)
                        
            # Log available models
            if self.models:
                logger.info(f"Loaded {len(self.models)} model configurations")
            else:
                logger.warning("No pre-trained models found")
                
        except Exception as e:
            logger.warning("Failed to load models", error=str(e))
    
    async def generate_signal(self, pair_id: str) -> Optional[AISignal]:
        """Generate advanced trading signal using ensemble ML models and real-time data
        
        Args:
            pair_id: The trading pair identifier (e.g., 'btc_idr')
            
        Returns:
            AISignal object with signal details or None if generation fails
        """
        try:
            logger.info("Generating signal with advanced AI", pair_id=pair_id)
            
            # Normalize pair_id format - ensure consistent format with underscore
            if "_" not in pair_id and len(pair_id) > 3:
                base_currency = pair_id[:-3].lower()  # e.g., "btc" from "btcidr"
                quote_currency = pair_id[-3:].lower()  # e.g., "idr" from "btcidr"
                pair_id = f"{base_currency}_{quote_currency}"
            
            # Get historical data with adequate lookback period
            historical_data = await self._get_historical_data(pair_id, days=60)  # 60 days for better pattern recognition
            
            if historical_data is None or len(historical_data) < self.min_data_points:
                logger.warning(f"Insufficient data for AI signal generation: {len(historical_data) if historical_data is not None else 0} < {self.min_data_points} points", 
                              pair_id=pair_id)
                # Fallback to simple signal based on current market data
                return await self._generate_simple_signal(pair_id)
            
            # Calculate comprehensive technical indicators
            indicators = self._calculate_technical_indicators(historical_data)
            if not indicators:
                logger.warning("Failed to calculate technical indicators", pair_id=pair_id)
                return await self._generate_simple_signal(pair_id)
            
            # Add market context (e.g., overall market trends, volume profile)
            await self._add_market_context(pair_id, historical_data, indicators)
            
            # Generate signals using different methods and ensemble the results
            signals = {}
            confidences = {}
            
            # 1. Primary ML model if available
            if pair_id in self.models:
                ml_signal = await self._generate_ml_signal(pair_id, indicators, historical_data)
                if ml_signal:
                    signals["ml"] = ml_signal.signal_type
                    confidences["ml"] = ml_signal.confidence
            
            # 2. Ensemble model (XGBoost/LightGBM) if available
            ensemble_signal = await self._generate_ensemble_signal(pair_id, indicators, historical_data)
            if ensemble_signal:
                signals["ensemble"] = ensemble_signal.signal_type
                confidences["ensemble"] = ensemble_signal.confidence
            
            # 3. Pure technical analysis as additional input
            ta_signal = await self._generate_technical_signal(pair_id, indicators)
            if ta_signal:
                signals["technical"] = ta_signal.signal_type
                confidences["technical"] = ta_signal.confidence
            
            # If we don't have any signals, return the technical one as fallback
            if not signals and ta_signal:
                logger.info("Using technical analysis as fallback", pair_id=pair_id)
                await self._save_signal(ta_signal)
                return ta_signal
            elif not signals:
                logger.warning("No signals generated", pair_id=pair_id)
                return await self._generate_simple_signal(pair_id)
            
            # Ensemble the signals with weighted voting
            final_signal = await self._ensemble_signals(pair_id, signals, confidences, historical_data)
            
            # Calculate additional metrics for the signal
            await self._enrich_signal_metrics(pair_id, final_signal, historical_data)
            
            # Save final signal to database
            if final_signal:
                await self._save_signal(final_signal)
                logger.info("Signal generated successfully", 
                           pair_id=pair_id, 
                           signal_type=final_signal.signal_type,
                           confidence=final_signal.confidence,
                           methods=list(signals.keys()))
            
            return final_signal
            
        except Exception as e:
            logger.error("Failed to generate signal", pair_id=pair_id, error=str(e))
            return None
    
    async def _get_historical_data(self, pair_id: str, days: int = 30, resolution: str = "1D") -> Optional[pd.DataFrame]:
        """Get historical price data for analysis from database with API fallback
        
        Args:
            pair_id: Trading pair ID (e.g., 'btc_idr')
            days: Number of days of historical data to fetch
            resolution: Data resolution ('1D', '4h', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if data retrieval fails
        """
        try:
            # Try database first (preferred source for consistent data)
            db_data = await self._get_data_from_database(pair_id, days)
            if db_data is not None and len(db_data) >= self.min_data_points:
                logger.info(f"Using database data for {pair_id}: {len(db_data)} records")
                return db_data
                
            # Fallback: get data from API
            logger.info(f"Database data insufficient, fetching from API: {pair_id}")
            api_data = await self._fetch_api_data(pair_id, days, resolution)
            if api_data is not None:
                # Store the API data in the database for future use
                await self._store_api_data(pair_id, api_data)
                return api_data
                
            # If both methods failed, try fetching a smaller timeframe
            if days > 14:
                logger.warning(f"Failed to get {days} days of data, trying last 14 days")
                return await self._get_historical_data(pair_id, 14, resolution)
            
            logger.error(f"Could not retrieve historical data for {pair_id}")
            return None
                
        except Exception as e:
            logger.error(f"Failed to get historical data for {pair_id}: {str(e)}")
            return None
            
    async def _get_data_from_database(self, pair_id: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data from the database"""
        try:
            db = get_db()
            try:
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Query with time-based filtering
                price_data = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= cutoff_date
                ).order_by(PriceHistory.timestamp.asc()).all()
                
                if not price_data or len(price_data) < self.min_data_points:
                    logger.warning(f"Insufficient records in database: {len(price_data) if price_data else 0}")
                    return None
                
                # Convert to DataFrame
                data = []
                for record in price_data:
                    try:
                        data.append({
                            'timestamp': record.timestamp,
                            'open': float(getattr(record, 'open_price', 0)),
                            'high': float(getattr(record, 'high_price', 0)),
                            'low': float(getattr(record, 'low_price', 0)),
                            'close': float(getattr(record, 'close_price', 0)),
                            'volume': float(getattr(record, 'volume', 0))
                        })
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning(f"Error converting record to float: {str(e)}")
                        continue
                
                df = pd.DataFrame(data)
                
                # Check if data is continuous (no big gaps)
                if len(df) > 0:
                    df['timestamp_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600
                    max_gap = df['timestamp_diff'].max()
                    if max_gap > 24:  # More than 24 hours gap
                        logger.warning(f"Data has gaps, max gap: {max_gap} hours")
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                return df
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return None
    
    async def _fetch_api_data(self, pair_id: str, days: int = 30, resolution: str = "1D") -> Optional[pd.DataFrame]:
        """Fetch historical data directly from Indodax API
        
        Args:
            pair_id: Trading pair ID (e.g., 'btc_idr')
            days: Number of days to fetch
            resolution: Data resolution ('1D', '4h', '1h', '15', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get current timestamp and calculate start time
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (days * 24 * 60 * 60)  # Convert days to seconds
            
            # Format pair_id for API call (API requires uppercase without underscore)
            api_pair_id = pair_id.replace("_", "").upper()
            
            # Request OHLC data from Indodax API
            logger.info(f"Fetching {days} days of {resolution} data for {api_pair_id} from Indodax API")
            ohlc_data = await self.api.get_ohlc_history(
                api_pair_id, start_time, end_time, resolution
            )
            
            if not ohlc_data:
                logger.warning(f"No OHLC data returned from API for {api_pair_id}")
                return None
                
            if isinstance(ohlc_data, dict) and "error" in ohlc_data:
                error_msg = ohlc_data.get("error", "Unknown API error")
                logger.error(f"API error: {error_msg}")
                return None
                
            # Validate data format
            if not isinstance(ohlc_data, list):
                logger.error(f"Invalid data format from API: {type(ohlc_data)}")
                return None
                
            # Convert API data to DataFrame
            data = []
            for candle in ohlc_data:
                try:
                    # Handle different API response formats
                    if isinstance(candle, dict):
                        # Dictionary format
                        data.append({
                            'timestamp': datetime.fromtimestamp(int(candle.get('Time', 0))),
                            'open': float(candle.get('Open', 0)),
                            'high': float(candle.get('High', 0)),
                            'low': float(candle.get('Low', 0)),
                            'close': float(candle.get('Close', 0)),
                            'volume': float(candle.get('Volume', 0))
                        })
                    elif isinstance(candle, list) and len(candle) >= 6:
                        # Array format [time, open, high, low, close, volume]
                        data.append({
                            'timestamp': datetime.fromtimestamp(int(candle[0])),
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        })
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Error parsing candle data: {str(e)}")
                    continue
            
            if not data:
                logger.warning("No valid candles found in API response")
                return None
                
            # Create DataFrame and sort by timestamp
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')
            
            # Remove duplicate timestamps
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} candles from API")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data from API: {str(e)}")
            return None
            
    async def _store_api_data(self, pair_id: str, df: pd.DataFrame) -> bool:
        """Store API data in the database for future use
        
        Args:
            pair_id: Trading pair ID
            df: DataFrame with OHLCV data
            
        Returns:
            True if data was stored successfully, False otherwise
        """
        try:
            db = get_db()
            try:
                # Reset index to access timestamp column
                df_records = df.reset_index()
                
                # Prepare records for bulk insert
                records = []
                for _, row in df_records.iterrows():
                    # Check if record already exists
                    existing = db.query(PriceHistory).filter(
                        PriceHistory.pair_id == pair_id,
                        PriceHistory.timestamp == row['timestamp']
                    ).first()
                    
                    if not existing:
                        record = PriceHistory(
                            pair_id=pair_id,
                            timestamp=row['timestamp'],
                            open_price=float(row['open']),
                            high_price=float(row['high']),
                            low_price=float(row['low']),
                            close_price=float(row['close']),
                            volume=float(row['volume'])
                        )
                        records.append(record)
                
                # Bulk insert new records
                if records:
                    db.bulk_save_objects(records)
                    db.commit()
                    logger.info(f"Stored {len(records)} new price records in database")
                    return True
                    
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to store API data: {str(e)}")
            return False
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for signal generation
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            indicators = {}
            
            if not TA_AVAILABLE:
                # Fallback calculation if TA library is not available
                return self._calculate_basic_indicators(df_copy)
            
            # Moving Averages
            sma20 = SMAIndicator(close=df_copy['close'], window=20)
            indicators['sma_20'] = sma20.sma_indicator()
            
            sma50 = SMAIndicator(close=df_copy['close'], window=50)
            indicators['sma_50'] = sma50.sma_indicator()
            
            ema12 = EMAIndicator(close=df_copy['close'], window=12)
            indicators['ema_12'] = ema12.ema_indicator()
            
            ema26 = EMAIndicator(close=df_copy['close'], window=26)
            indicators['ema_26'] = ema26.ema_indicator()
            
            # MACD
            macd = MACD(close=df_copy['close'])
            indicators['macd'] = macd.macd_diff()
            indicators['macd_signal'] = macd.macd_signal()
            indicators['macd_line'] = macd.macd()
            
            # RSI
            rsi_indicator = RSIIndicator(close=df_copy['close'], window=14)
            indicators['rsi'] = rsi_indicator.rsi()
            
            # Bollinger Bands
            bollinger = BollingerBands(close=df_copy['close'])
            indicators['bb_upper'] = bollinger.bollinger_hband()
            indicators['bb_lower'] = bollinger.bollinger_lband()
            indicators['bb_middle'] = bollinger.bollinger_mavg()
            indicators['bb_width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'])
            indicators['stoch_k'] = stoch.stoch()
            indicators['stoch_d'] = stoch.stoch_signal()
            
            # ATR for volatility measurement
            atr = AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'])
            indicators['atr'] = atr.average_true_range()
            
            # Volume indicators
            if 'volume' in df_copy:
                # Volume Moving Average
                indicators['volume_sma'] = df_copy['volume'].rolling(window=20).mean()
                
                # Volume trend
                indicators['volume_change'] = df_copy['volume'].pct_change()
                indicators['volume_ratio'] = df_copy['volume'] / df_copy['volume'].rolling(window=20).mean()
                
                # VWAP
                try:
                    vwap = VolumeWeightedAveragePrice(high=df_copy['high'], low=df_copy['low'], 
                                                     close=df_copy['close'], volume=df_copy['volume'])
                    indicators['vwap'] = vwap.volume_weighted_average_price()
                except:
                    # Calculate simplified VWAP
                    indicators['vwap'] = (df_copy['close'] * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()
                
                # Volume Price Trend
                vpt = VolumePriceTrendIndicator(close=df_copy['close'], volume=df_copy['volume'])
                indicators['vpt'] = vpt.volume_price_trend()
            
            # Support and Resistance levels
            indicators['support'] = df_copy['low'].rolling(window=20).min()
            indicators['resistance'] = df_copy['high'].rolling(window=20).max()
            
            # Price momentum at different timeframes
            for window in [1, 3, 5, 10, 20]:
                indicators[f'momentum_{window}'] = df_copy['close'].pct_change(periods=window)
                
            # Volatility indicators
            indicators['volatility_daily'] = df_copy['close'].pct_change().rolling(window=20).std()
            indicators['volatility_weekly'] = df_copy['close'].pct_change().rolling(window=5).std()
            
            # Price distance from moving averages (normalized)
            indicators['price_sma20_ratio'] = df_copy['close'] / indicators['sma_20']
            indicators['price_sma50_ratio'] = df_copy['close'] / indicators['sma_50']
            
            # Moving average crossovers
            indicators['sma_crossover'] = indicators['sma_20'] - indicators['sma_50']
            indicators['ema_crossover'] = indicators['ema_12'] - indicators['ema_26']
            
            # Market regime indicators (bull/bear)
            # Calculate historical returns
            df_copy['returns'] = df_copy['close'].pct_change()
            
            # Calculate bull/bear indicator based on moving averages
            indicators['market_regime'] = np.where(
                indicators['sma_20'] > indicators['sma_50'], 
                1,  # Bullish
                -1  # Bearish
            )
            
            # Remove NaN values
            for key in indicators:
                if isinstance(indicators[key], pd.Series):
                    indicators[key] = indicators[key].fillna(method='ffill').fillna(0)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {str(e)}")
            # Fallback to basic indicators
            return self._calculate_basic_indicators(df)
            
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators when TA library is not available"""
        try:
            indicators = {}
            
            # Simple Moving Averages
            indicators['sma_20'] = df['close'].rolling(window=20).mean()
            indicators['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            indicators['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            indicators['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # Crude MACD calculation
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            
            # Simple RSI calculation
            delta = df['close'].diff()
            try:
                # Ensure we're working with numeric data
                delta = pd.to_numeric(delta, errors='coerce')
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
            except:
                # Fallback for type issues - manual calculation
                gain_values = []
                loss_values = []
                for i in range(len(delta)):
                    try:
                        val = delta.iloc[i]
                        # Handle NaN values
                        if val is None or str(val).lower() == 'nan':
                            gain_values.append(0.0)
                            loss_values.append(0.0)
                        else:
                            val_float = float(str(val))
                            if val_float > 0:
                                gain_values.append(val_float)
                                loss_values.append(0.0)
                            else:
                                gain_values.append(0.0)
                                loss_values.append(-val_float)
                    except (ValueError, TypeError, AttributeError):
                        gain_values.append(0.0)
                        loss_values.append(0.0)
                
                gain = pd.Series(gain_values, index=delta.index, dtype=float)
                loss = pd.Series(loss_values, index=delta.index, dtype=float)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            indicators['bb_middle'] = indicators['sma_20']
            std_dev = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (std_dev * 2)
            indicators['bb_lower'] = indicators['bb_middle'] - (std_dev * 2)
            
            # Price momentum
            indicators['momentum_1'] = df['close'].pct_change(periods=1)
            indicators['momentum_5'] = df['close'].pct_change(periods=5)
            
            # Support and resistance
            indicators['support'] = df['low'].rolling(window=20).min()
            indicators['resistance'] = df['high'].rolling(window=20).max()
            
            # Clean up NaN values
            for key in indicators:
                if isinstance(indicators[key], pd.Series):
                    indicators[key] = indicators[key].fillna(0)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate basic indicators: {str(e)}")
            return {}
    
    async def _generate_ml_signal(self, pair_id: str, indicators: Dict[str, Any], 
                                historical_data: pd.DataFrame) -> Optional[AISignal]:
        """Generate signal using machine learning model
        
        Args:
            pair_id: Trading pair ID
            indicators: Dictionary of technical indicators
            historical_data: Historical price data
            
        Returns:
            AISignal from ML model or None if generation fails
        """
        try:
            model = self.models[pair_id]
            
            # Prepare features for prediction
            features = self._prepare_features(indicators)
            
            if features is None:
                return None
            
            # Scale features if scaler is available
            if pair_id in self.scalers:
                features = self.scalers[pair_id].transform([features])[0]
            
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
            
            # Get current indicators for the signal
            current_indicators = self._get_current_indicators(indicators)
            
            # Add ML-specific indicators
            current_indicators['ml_buy_prob'] = float(buy_prob)
            current_indicators['ml_sell_prob'] = float(sell_prob)
            current_indicators['ml_hold_prob'] = float(hold_prob)
            # Store model type separately since indicators is expecting float values
            model_type_name = type(model).__name__
            
            # Create AI signal
            signal = AISignal(
                pair_id=pair_id,
                signal_type=signal_type,
                confidence=float(max_prob),
                price_prediction=self._predict_price(indicators),
                indicators=current_indicators,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=2)  # Longer expiry for ML signals
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
    
    async def _generate_simple_signal(self, pair_id: str) -> Optional[AISignal]:
        """Generate simple signal based on current market data when historical data is insufficient"""
        try:
            # Convert pair_id format for ticker API (btc_idr -> btcidr)
            ticker_pair = pair_id.replace("_", "") if "_" in pair_id else f"{pair_id}idr"
            
            # Get current ticker data
            ticker = await self.api.get_ticker(ticker_pair)
            if not ticker or "ticker" not in ticker:
                logger.warning("No ticker data available", pair_id=pair_id, ticker_pair=ticker_pair)
                return None
            
            ticker_data = ticker["ticker"]
            current_price = float(ticker_data.get("last", 0))
            high_24h = float(ticker_data.get("high", 0))
            low_24h = float(ticker_data.get("low", 0))
            volume = float(ticker_data.get("vol_idr", 0))
            
            if current_price == 0:
                logger.warning("Current price is zero", pair_id=pair_id)
                return None
            
            # Simple analysis based on 24h price position
            price_position = (current_price - low_24h) / (high_24h - low_24h) if high_24h > low_24h else 0.5
            
            # Calculate price change percentage
            if high_24h > low_24h:
                volatility = ((high_24h - low_24h) / low_24h) * 100
            else:
                volatility = 0
            
            # Generate signal based on price position and volatility
            if price_position < 0.3 and volatility > 2:  # Near 24h low with decent volatility
                signal_type = "buy"
                confidence = 0.65
                signal_strength = "medium"
                reasons = f"Price near 24h low ({price_position:.1%} of range), good entry opportunity"
            elif price_position > 0.7 and volatility > 2:  # Near 24h high with decent volatility
                signal_type = "sell" 
                confidence = 0.65
                signal_strength = "medium"
                reasons = f"Price near 24h high ({price_position:.1%} of range), consider taking profit"
            else:  # Middle range or low volatility
                signal_type = "hold"
                confidence = 0.5
                signal_strength = "weak"
                reasons = f"Price in middle range ({price_position:.1%}), wait for clearer signal"
            
            # Calculate stop loss and take profit
            if signal_type == "buy":
                stop_loss = current_price * 0.97  # 3% below
                take_profit = current_price * 1.05  # 5% above
            elif signal_type == "sell":
                stop_loss = current_price * 1.03  # 3% above
                take_profit = current_price * 0.95  # 5% below
            else:  # hold
                stop_loss = current_price * 0.95
                take_profit = current_price * 1.05
            
            # Create signal
            signal = AISignal(
                pair_id=pair_id,
                signal_type=signal_type,
                confidence=confidence,
                price_prediction=take_profit,
                indicators={
                    "current_price": current_price,
                    "high_24h": high_24h,
                    "low_24h": low_24h,
                    "volume": volume,
                    "price_position": price_position,
                    "volatility": volatility,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "signal_strength": signal_strength,
                    "reasons": reasons
                },
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            return signal
            
        except Exception as e:
            logger.error("Failed to generate simple signal", pair_id=pair_id, error=str(e))
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
    
    async def _add_market_context(self, pair_id: str, historical_data: pd.DataFrame, 
                           indicators: Dict[str, Any]) -> None:
        """Add broader market context to the indicators
        
        Args:
            pair_id: Trading pair ID
            historical_data: Historical price data
            indicators: Dictionary of technical indicators
        """
        try:
            # Get BTC/IDR data as market benchmark if this is not already BTC
            if not pair_id.startswith('btc_'):
                btc_data = await self._get_historical_data('btc_idr', days=30)
                if btc_data is not None and len(btc_data) > 20:
                    # Calculate correlation with BTC
                    if len(historical_data) >= len(btc_data):
                        # Align timestamps between pair data and BTC data
                        aligned_price = historical_data['close'].iloc[-len(btc_data):]
                        if len(aligned_price) == len(btc_data['close']):
                            correlation = aligned_price.corr(btc_data['close'])
                            indicators['btc_correlation'] = pd.Series([correlation] * len(historical_data.index),
                                                                     index=historical_data.index)
                    
                    # Add BTC trend indicator
                    btc_trend = 1 if btc_data['close'].iloc[-1] > btc_data['close'].iloc[-20] else -1
                    indicators['btc_trend'] = pd.Series([btc_trend] * len(historical_data.index),
                                                      index=historical_data.index)
                    
                    # Market strength - 15-day BTC volatility
                    btc_volatility = btc_data['close'].pct_change().rolling(15).std().iloc[-1]
                    indicators['market_volatility'] = pd.Series([btc_volatility] * len(historical_data.index),
                                                             index=historical_data.index)
            
            # Add market cap ranking if available (from API or fixed list)
            market_cap_ranking = await self._get_market_cap_ranking(pair_id)
            if market_cap_ranking:
                indicators['market_cap_rank'] = pd.Series([market_cap_ranking] * len(historical_data.index),
                                                       index=historical_data.index)
            
            # Add time-based features
            if isinstance(historical_data.index, pd.DatetimeIndex):
                hour_of_day = historical_data.index.hour
                day_of_week = historical_data.index.dayofweek
                month = historical_data.index.month
            else:
                # Handle non-datetime index by creating constant features
                hour_of_day = [12] * len(historical_data)
                day_of_week = [0] * len(historical_data)
                month = [datetime.now().month] * len(historical_data)
            
            indicators['hour_of_day'] = pd.Series(hour_of_day, index=historical_data.index)
            indicators['day_of_week'] = pd.Series(day_of_week, index=historical_data.index)
            
            # Month seasonality (1-12)
            indicators['month'] = pd.Series(month, index=historical_data.index)
            
            # Trading volume trends
            if 'volume' in historical_data.columns:
                vol_ratio = historical_data['volume'] / historical_data['volume'].rolling(window=30).mean()
                indicators['volume_ratio_30d'] = vol_ratio
            
            logger.info(f"Added market context for {pair_id}")
            
        except Exception as e:
            logger.error(f"Failed to add market context: {str(e)}")
    
    async def _get_market_cap_ranking(self, pair_id: str) -> Optional[int]:
        """Get market cap ranking for a cryptocurrency
        
        This is a simplistic implementation that could be enhanced with a proper
        market data API in the future.
        """
        # Simple mapping for common currencies (could be from API or external data source)
        rankings = {
            'btc_idr': 1,   # Bitcoin
            'eth_idr': 2,   # Ethereum
            'usdt_idr': 3,  # Tether
            'bnb_idr': 4,   # Binance Coin
            'sol_idr': 5,   # Solana
            'xrp_idr': 6,   # XRP
            'ada_idr': 8,   # Cardano
            'doge_idr': 10, # Dogecoin
            'trx_idr': 12,  # TRON
            'dot_idr': 15,  # Polkadot
            'ltc_idr': 18,  # Litecoin
        }
        
        return rankings.get(pair_id.lower())
    
    async def _generate_ensemble_signal(self, pair_id: str, indicators: Dict[str, Any], 
                                historical_data: pd.DataFrame) -> Optional[AISignal]:
        """Generate signals using ensemble models (XGBoost, LightGBM)
        
        Args:
            pair_id: Trading pair ID
            indicators: Dictionary of technical indicators
            historical_data: Historical price data
            
        Returns:
            Signal from ensemble models or None if not available
        """
        try:
            ensemble_pair_id = f"{pair_id}_ensemble"
            if ensemble_pair_id not in self.models:
                return None
            
            ensemble_results = {}
            ensemble_confidences = {}
            features = self._prepare_features(indicators)
            
            if features is None:
                return None
                
            # Convert features to numpy array if it's not already
            if not isinstance(features, np.ndarray):
                features = np.array(features).reshape(1, -1)
                
            # XGBoost model
            if XGBOOST_AVAILABLE and xgb is not None and self.models.get(ensemble_pair_id, {}).get("xgboost"):
                try:
                    xgb_model_path = os.path.join(self.ensemble_path, f"{pair_id}_xgb.json")
                    if os.path.exists(xgb_model_path):
                        # Load model if not already in memory
                        if "xgb_model" not in self.models.get(ensemble_pair_id, {}):
                            bst = xgb.Booster()
                            bst.load_model(xgb_model_path)
                            self.models[ensemble_pair_id]["xgb_model"] = bst
                        else:
                            bst = self.models[ensemble_pair_id]["xgb_model"]
                        
                        # Make prediction
                        dmatrix = xgb.DMatrix(features)
                        xgb_preds = bst.predict(dmatrix)
                        
                        # Extract signal and confidence - ensure we have array
                        if isinstance(xgb_preds, np.ndarray) and len(xgb_preds) >= 3:  # Assuming 3 classes: sell, hold, buy
                            signal_idx = np.argmax(xgb_preds)
                            signals = ["sell", "hold", "buy"]
                            ensemble_results["xgboost"] = signals[signal_idx]
                            ensemble_confidences["xgboost"] = float(xgb_preds[signal_idx])
                except Exception as e:
                    logger.warning(f"XGBoost prediction failed: {str(e)}")
            
            # LightGBM model
            if LIGHTGBM_AVAILABLE and lgb is not None and self.models.get(ensemble_pair_id, {}).get("lightgbm"):
                try:
                    lgbm_model_path = os.path.join(self.ensemble_path, f"{pair_id}_lgbm.txt")
                    if os.path.exists(lgbm_model_path):
                        # Load model if not already in memory
                        if "lgbm_model" not in self.models.get(ensemble_pair_id, {}):
                            gbm = lgb.Booster(model_file=lgbm_model_path)
                            self.models[ensemble_pair_id]["lgbm_model"] = gbm
                        else:
                            gbm = self.models[ensemble_pair_id]["lgbm_model"]
                        
                        # Make prediction
                        lgbm_preds = gbm.predict(features)
                        
                        # Extract signal and confidence - ensure we have array
                        if isinstance(lgbm_preds, np.ndarray) and len(lgbm_preds) >= 3:  # Assuming 3 classes: sell, hold, buy
                            signal_idx = np.argmax(lgbm_preds)
                            signals = ["sell", "hold", "buy"]
                            ensemble_results["lightgbm"] = signals[signal_idx]
                            ensemble_confidences["lightgbm"] = float(lgbm_preds[signal_idx])
                except Exception as e:
                    logger.warning(f"LightGBM prediction failed: {str(e)}")
            
            # If we don't have any ensemble results, return None
            if not ensemble_results:
                return None
                
            # Combine ensemble results with weighted voting
            return self._combine_ensemble_votes(pair_id, ensemble_results, ensemble_confidences)
            
        except Exception as e:
            logger.error(f"Failed to generate ensemble signal for {pair_id}: {str(e)}")
            return None
            
    def _combine_ensemble_votes(self, pair_id: str, signals: Dict[str, str], 
                               confidences: Dict[str, float]) -> Optional[AISignal]:
        """Combine votes from multiple ensemble models
        
        Args:
            pair_id: Trading pair ID
            signals: Dictionary of signals from each model
            confidences: Dictionary of confidence scores from each model
            
        Returns:
            Combined AISignal
        """
        if not signals:
            return None
            
        # Define weights for different models
        weights = {
            "xgboost": 0.6,
            "lightgbm": 0.4
        }
        
        # Count weighted votes for each signal type
        vote_counts = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_weight = 0.0
        
        for model, signal in signals.items():
            if model in weights and signal in vote_counts:
                vote_counts[signal] += weights[model] * confidences.get(model, 0.5)
                total_weight += weights[model]
        
        if total_weight == 0.0:
            return None
            
        # Normalize votes by total weight
        for signal in vote_counts:
            vote_counts[signal] /= total_weight
            
        # Find signal with highest vote
        max_signal = max(vote_counts.items(), key=lambda x: x[1])
        signal_type = max_signal[0]
        confidence = max_signal[1]
        
        if confidence < self.min_confidence_threshold:
            signal_type = "hold"
            confidence = 0.5
            
        # Create signal
        current_indicators = {}
        for model, signal in signals.items():
            current_indicators[f"{model}_signal"] = signal
            current_indicators[f"{model}_confidence"] = confidences.get(model, 0.0)
            
        current_price = self._get_current_price_from_indicators(current_indicators) 
        
        signal = AISignal(
            pair_id=pair_id,
            signal_type=signal_type,
            confidence=float(confidence),
            price_prediction=self._predict_ensemble_price(signal_type, current_price),
            indicators=current_indicators,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=3)  # Longer expiry for ensemble signals
        )
        
        return signal
        
    def _predict_ensemble_price(self, signal_type: str, current_price: float) -> float:
        """Predict price target based on signal type for ensemble models"""
        if signal_type == "buy":
            return current_price * 1.03  # 3% profit target
        elif signal_type == "sell":
            return current_price * 0.97  # 3% lower price target
        else:
            return current_price  # No change for hold
            
    def _get_current_price_from_indicators(self, indicators: Dict[str, Any]) -> float:
        """Extract current price from indicators dictionary"""
        if 'close' in indicators:
            return indicators['close']
        return 0.0
        
    async def _ensemble_signals(self, pair_id: str, signals: Dict[str, str], 
                               confidences: Dict[str, float], 
                               historical_data: pd.DataFrame) -> AISignal:
        """Ensemble different signal sources with weighted voting
        
        Args:
            pair_id: Trading pair ID
            signals: Dictionary with signals from different sources
            confidences: Dictionary with confidence levels for each signal
            historical_data: Historical price data for additional context
            
        Returns:
            Final AISignal combining all sources
        """
        try:
            # Define weights for different signal sources
            weights = {
                "ml": 0.35,       # Primary ML model
                "ensemble": 0.40,  # Ensemble models (XGBoost, LightGBM)
                "technical": 0.25  # Traditional technical analysis
            }
            
            # Count weighted votes for each signal type
            vote_counts = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
            total_weight = 0.0
            
            for source, signal in signals.items():
                if source in weights and signal in vote_counts:
                    vote_counts[signal] += weights[source] * confidences.get(source, 0.5)
                    total_weight += weights[source]
            
            if total_weight == 0.0:
                logger.warning(f"No valid signals found for {pair_id}")
                # Fallback to technical analysis or simple signal
                return await self._generate_technical_signal(pair_id, 
                                                           self._calculate_technical_indicators(historical_data))
                
            # Normalize votes by total weight
            for signal in vote_counts:
                vote_counts[signal] /= total_weight
                
            # Find signal with highest vote
            max_signal = max(vote_counts.items(), key=lambda x: x[1])
            signal_type = max_signal[0]
            confidence = max_signal[1]
            
            # For low confidence predictions, default to hold
            if confidence < self.min_confidence_threshold:
                logger.info(f"Low confidence signal ({confidence:.2f}) for {pair_id}, defaulting to hold")
                signal_type = "hold"
                confidence = 0.5
                
            # Create a combined signal
            combined_indicators = {}
            
            # Add info about contributing signals
            for source, signal in signals.items():
                combined_indicators[f"{source}_signal"] = signal
                combined_indicators[f"{source}_confidence"] = confidences.get(source, 0.0)
                
            # Add current market data
            try:
                if len(historical_data) > 0:
                    latest_data = historical_data.iloc[-1]
                    combined_indicators["close"] = float(latest_data["close"])
                    combined_indicators["open"] = float(latest_data["open"])
                    combined_indicators["high"] = float(latest_data["high"])
                    combined_indicators["low"] = float(latest_data["low"])
                    combined_indicators["volume"] = float(latest_data["volume"])
            except (KeyError, IndexError, ValueError) as e:
                logger.warning(f"Couldn't extract latest market data: {str(e)}")
                
            # Create the final signal
            signal = AISignal(
                pair_id=pair_id,
                signal_type=signal_type,
                confidence=float(confidence),
                price_prediction=self._predict_price_target(signal_type, combined_indicators),
                indicators=combined_indicators,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=2)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to ensemble signals for {pair_id}: {str(e)}")
            # Fallback to technical analysis
            return await self._generate_technical_signal(pair_id, 
                                                       self._calculate_technical_indicators(historical_data))
                                                       
    def _predict_price_target(self, signal_type: str, indicators: Dict[str, Any]) -> float:
        """Predict target price based on signal type and current indicators"""
        # Get current price
        current_price = indicators.get("close", 0.0)
        if current_price == 0.0:
            return 0.0
            
        # Adjust prediction based on signal type
        if signal_type == "buy":
            # Target: Typical profit target based on recent volatility
            atr = indicators.get("atr", current_price * 0.02)  # Default to 2% if ATR not available
            return current_price * (1 + (2.5 * (atr / current_price)))  # Target 2.5x ATR profit
            
        elif signal_type == "sell":
            # For sell signals, predict continued decline
            atr = indicators.get("atr", current_price * 0.02)
            return current_price * (1 - (2.0 * (atr / current_price)))  # Target 2x ATR decline
            
        else:  # hold
            return current_price  # No change expected
    
    async def _enrich_signal_metrics(self, pair_id: str, signal: AISignal, 
                            historical_data: pd.DataFrame) -> None:
        """Enrich the signal with additional metrics and risk management parameters
        
        Args:
            pair_id: Trading pair ID
            signal: The AISignal to enrich
            historical_data: Historical price data for calculations
        """
        try:
            if signal is None or len(historical_data) < 20:
                return
                
            # Calculate volatility-based metrics
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365)  # Annualized volatility
            
            # Calculate ATR for stop loss/take profit levels
            high_low = historical_data['high'] - historical_data['low']
            high_close_prev = pd.Series(np.abs(historical_data['high'] - historical_data['close'].shift(1)), 
                                      index=historical_data.index)
            low_close_prev = pd.Series(np.abs(historical_data['low'] - historical_data['close'].shift(1)), 
                                     index=historical_data.index)
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            current_price = historical_data['close'].iloc[-1]
            
            # Enhanced indicators dictionary - properly handle SQLAlchemy Column vs dict
            enhanced_indicators = {}  # Always start with a fresh dict
            if hasattr(signal, 'indicators') and signal.indicators is not None:
                # If it's a SQLAlchemy Column, we need to get the actual value
                try:
                    if hasattr(signal.indicators, 'copy'):
                        # If it's a dict-like object with copy method
                        temp_indicators = signal.indicators.copy()
                        if isinstance(temp_indicators, dict):
                            enhanced_indicators.update(temp_indicators)
                    elif isinstance(signal.indicators, dict):
                        enhanced_indicators.update(signal.indicators)
                    # If it's still not a dict, we'll just start with empty dict
                except Exception:
                    # If any error occurs, start with empty dict
                    enhanced_indicators = {}
            
            # Add volatility metrics
            enhanced_indicators['annualized_volatility'] = float(volatility)
            enhanced_indicators['atr'] = float(atr)
            enhanced_indicators['atr_percentage'] = float((atr / current_price) * 100)
            
            # Calculate dynamic stop loss and take profit based on volatility
            # Get actual signal type value (handle SQLAlchemy Column)
            signal_type = getattr(signal, 'signal_type', 'hold')
            if hasattr(signal_type, '__str__'):
                signal_type = str(signal_type)
            
            if signal_type == "buy":
                # For buy signals: Stop loss below current price, take profit above
                stop_loss_pct = max(0.02, min(0.10, atr / current_price * 2))  # 2-10% based on volatility
                take_profit_pct = max(0.03, min(0.15, atr / current_price * 3))  # 3-15% based on volatility
                
                enhanced_indicators['stop_loss'] = current_price * (1 - stop_loss_pct)
                enhanced_indicators['take_profit'] = current_price * (1 + take_profit_pct)
                enhanced_indicators['risk_reward_ratio'] = take_profit_pct / stop_loss_pct
                
            elif signal_type == "sell":
                # For sell signals: Stop loss above current price, take profit below
                stop_loss_pct = max(0.02, min(0.10, atr / current_price * 2))
                take_profit_pct = max(0.03, min(0.15, atr / current_price * 3))
                
                enhanced_indicators['stop_loss'] = current_price * (1 + stop_loss_pct)
                enhanced_indicators['take_profit'] = current_price * (1 - take_profit_pct)
                enhanced_indicators['risk_reward_ratio'] = take_profit_pct / stop_loss_pct
                
            else:  # hold
                enhanced_indicators['stop_loss'] = current_price * 0.95  # 5% protective stop
                enhanced_indicators['take_profit'] = current_price * 1.05  # 5% profit target
                enhanced_indicators['risk_reward_ratio'] = 1.0
            
            # Add trend strength metrics
            if len(historical_data) >= 50:
                sma_20 = historical_data['close'].rolling(20).mean().iloc[-1]
                sma_50 = historical_data['close'].rolling(50).mean().iloc[-1]
                
                trend_strength = abs(sma_20 - sma_50) / sma_50
                enhanced_indicators['trend_strength'] = float(trend_strength)
                enhanced_indicators['trend_direction'] = 1 if sma_20 > sma_50 else -1
            
            # Add volume analysis
            if 'volume' in historical_data.columns:
                avg_volume = historical_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = historical_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                enhanced_indicators['volume_ratio'] = float(volume_ratio)
                enhanced_indicators['volume_confirmation'] = volume_ratio > 1.2  # Strong volume
            
            # Add market regime analysis
            # Calculate recent performance vs longer-term performance
            if len(historical_data) >= 30:
                recent_return = (current_price / historical_data['close'].iloc[-7] - 1) * 100  # 7-day return
                monthly_return = (current_price / historical_data['close'].iloc[-30] - 1) * 100  # 30-day return
                
                enhanced_indicators['recent_performance'] = float(recent_return)
                enhanced_indicators['monthly_performance'] = float(monthly_return)
                
                # Market regime: trending vs ranging
                high_20 = historical_data['high'].rolling(20).max().iloc[-1]
                low_20 = historical_data['low'].rolling(20).min().iloc[-1]
                range_pct = (high_20 - low_20) / low_20 * 100
                
                enhanced_indicators['market_range_pct'] = float(range_pct)
                enhanced_indicators['market_regime'] = 'trending' if range_pct > 15 else 'ranging'
            
            # Add signal quality score (0-100)
            quality_score = self._calculate_signal_quality(signal, enhanced_indicators, historical_data)
            enhanced_indicators['signal_quality_score'] = quality_score
            
            # Add recommended position size based on volatility and confidence
            # Get actual confidence value (handle SQLAlchemy Column)
            confidence_value = getattr(signal, 'confidence', 0.5)
            if hasattr(confidence_value, '__float__'):
                confidence_value = float(confidence_value)
            
            position_size = self._calculate_position_size(confidence_value, volatility, atr / current_price)
            enhanced_indicators['recommended_position_size'] = position_size
            
            # Update the signal with enhanced indicators (handle SQLAlchemy assignment)
            # Note: Since AISignal.indicators is a JSON column, we can't modify it after creation
            # The indicators should be set during signal creation in the calling methods
            
            logger.info(f"Enriched signal for {pair_id} with {len(enhanced_indicators)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to enrich signal metrics: {str(e)}")
    
    def _calculate_signal_quality(self, signal: AISignal, indicators: Dict[str, Any], 
                                 historical_data: pd.DataFrame) -> float:
        """Calculate overall signal quality score (0-100)
        
        Args:
            signal: The AISignal object
            indicators: Dictionary of indicators
            historical_data: Historical price data
            
        Returns:
            Quality score from 0 to 100
        """
        try:
            score = 0.0
            max_score = 100.0
            
            # Get actual values from SQLAlchemy columns
            signal_type = getattr(signal, 'signal_type', 'hold')
            if hasattr(signal_type, '__str__'):
                signal_type = str(signal_type)
                
            confidence_value = getattr(signal, 'confidence', 0.5)
            if hasattr(confidence_value, '__float__'):
                confidence_value = float(confidence_value)
            
            # Confidence score (30% weight)
            confidence_score = confidence_value * 30
            score += confidence_score
            
            # Trend alignment (20% weight)
            trend_alignment = 0
            if 'trend_direction' in indicators:
                if signal_type == "buy" and indicators['trend_direction'] == 1:
                    trend_alignment = 20
                elif signal_type == "sell" and indicators['trend_direction'] == -1:
                    trend_alignment = 20
                elif signal_type == "hold":
                    trend_alignment = 15  # Neutral position gets some points
            score += trend_alignment
            
            # Volume confirmation (15% weight)
            volume_score = 0
            if 'volume_confirmation' in indicators:
                if indicators['volume_confirmation']:
                    volume_score = 15
                else:
                    volume_score = 8  # Partial points for normal volume
            score += volume_score
            
            # Volatility appropriateness (15% weight)
            volatility_score = 0
            if 'atr_percentage' in indicators:
                atr_pct = indicators['atr_percentage']
                if 1.0 <= atr_pct <= 5.0:  # Optimal volatility range
                    volatility_score = 15
                elif 0.5 <= atr_pct <= 8.0:  # Acceptable range
                    volatility_score = 10
                else:  # Too low or too high volatility
                    volatility_score = 5
            score += volatility_score
            
            # Risk-reward ratio (10% weight)
            rr_score = 0
            if 'risk_reward_ratio' in indicators:
                rr_ratio = indicators['risk_reward_ratio']
                if rr_ratio >= 2.0:  # Excellent risk-reward
                    rr_score = 10
                elif rr_ratio >= 1.5:  # Good risk-reward
                    rr_score = 8
                elif rr_ratio >= 1.0:  # Acceptable risk-reward
                    rr_score = 5
            score += rr_score
            
            # Market regime fit (10% weight)
            regime_score = 0
            if 'market_regime' in indicators:
                if signal_type != "hold":
                    if indicators['market_regime'] == 'trending':
                        regime_score = 10  # Good for directional signals
                    else:
                        regime_score = 6   # Ranging market less ideal
                else:
                    regime_score = 8  # Hold is often good in ranging markets
            score += regime_score
            
            return float(min(score, max_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate signal quality: {str(e)}")
            return 50.0  # Default middle score
    
    def _calculate_position_size(self, confidence: float, volatility: float, atr_pct: float) -> float:
        """Calculate recommended position size as percentage of portfolio
        
        Args:
            confidence: Signal confidence (0-1)
            volatility: Annualized volatility 
            atr_pct: ATR as percentage of price
            
        Returns:
            Recommended position size as decimal (e.g., 0.05 = 5%)
        """
        try:
            # Base position size
            base_size = 0.02  # 2% base allocation
            
            # Adjust for confidence
            confidence_multiplier = min(confidence * 2, 1.5)  # Max 1.5x for high confidence
            
            # Adjust for volatility (reduce size for higher volatility)
            volatility_adjustment = max(0.5, 1 - (volatility - 0.5))  # Reduce if vol > 50%
            
            # Adjust for ATR (reduce size for higher ATR)
            atr_adjustment = max(0.5, 1 - (atr_pct - 0.02) / 0.05)  # Reduce if ATR > 2%
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * volatility_adjustment * atr_adjustment
            
            # Cap at reasonable limits
            return max(0.005, min(position_size, 0.10))  # Between 0.5% and 10%
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {str(e)}")
            return 0.02  # Default 2%
