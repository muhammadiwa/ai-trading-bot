import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    """
    Kelas untuk prediksi harga menggunakan LSTM Deep Learning
    """
    
    def __init__(self, sequence_length: int = 60, prediction_days: int = 1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Mempersiapkan data untuk training LSTM
        """
        # Pilih kolom yang akan digunakan sebagai fitur
        feature_columns = ['close', 'volume', 'high', 'low']
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if not available_columns:
            available_columns = [target_column]
        
        # Menggunakan data yang tersedia
        dataset = data[available_columns].values
        
        # Normalisasi data
        dataset_scaled = self.scaler.fit_transform(dataset)
        
        # Membuat sequences untuk LSTM
        X, y = [], []
        for i in range(self.sequence_length, len(dataset_scaled) - self.prediction_days + 1):
            X.append(dataset_scaled[i-self.sequence_length:i])
            y.append(dataset_scaled[i:i+self.prediction_days, 0])  # Prediksi harga close
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Membangun model LSTM
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.prediction_days)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.2, target_column: str = 'close') -> dict:
        """
        Melatih model LSTM
        """
        print("Preparing data for training...")
        X, y = self.prepare_data(data, target_column)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data untuk training dan validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        print("Starting training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        self.is_trained = True
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        training_results = {
            'train_loss': train_loss[0],
            'train_mae': train_loss[1],
            'val_loss': val_loss[0],
            'val_mae': val_loss[1],
            'epochs_completed': len(history.history['loss']),
            'history': history.history
        }
        
        print(f"Training completed. Final validation loss: {val_loss[0]:.6f}")
        return training_results
    
    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> np.ndarray:
        """
        Membuat prediksi harga
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Menggunakan data terakhir untuk prediksi
        feature_columns = ['close', 'volume', 'high', 'low']
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if len(available_columns) == 0:
            available_columns = ['close']
        
        # Ambil data terakhir sesuai sequence_length
        last_data = data[available_columns].tail(self.sequence_length).values
        last_data_scaled = self.scaler.transform(last_data)
        
        predictions = []
        current_sequence = last_data_scaled.copy()
        
        for _ in range(steps_ahead):
            # Reshape untuk input LSTM
            X_pred = current_sequence.reshape(1, self.sequence_length, len(available_columns))
            
            # Prediksi
            pred_scaled = self.model.predict(X_pred, verbose=0)
            
            # Denormalisasi hanya untuk harga close (kolom pertama)
            pred_dummy = np.zeros((1, len(available_columns)))
            pred_dummy[0, 0] = pred_scaled[0, 0]
            pred_price = self.scaler.inverse_transform(pred_dummy)[0, 0]
            
            predictions.append(pred_price)
            
            # Update sequence untuk prediksi berikutnya
            # Tambahkan prediksi ke sequence dan hapus data paling lama
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled[0, 0]  # Update harga close dengan prediksi
            current_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
        
        return np.array(predictions)
    
    def load_model(self, model_path: str = 'models/best_lstm_model.h5', 
                   scaler_path: str = 'models/scaler.pkl') -> bool:
        """
        Memuat model yang sudah dilatih
        """
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("Model loaded successfully")
                return True
            else:
                print("Model files not found")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str = 'models/lstm_model.h5',
                   scaler_path: str = 'models/scaler.pkl'):
        """
        Menyimpan model yang sudah dilatih
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"Model saved to {model_path}")
    
    def evaluate_predictions(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """
        Mengevaluasi akurasi prediksi
        """
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # Menghitung akurasi arah (apakah prediksi naik/turun benar)
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }
    
    def get_model_summary(self) -> str:
        """
        Mendapatkan ringkasan model
        """
        if self.model is not None:
            return self.model.summary()
        else:
            return "Model has not been built yet"
