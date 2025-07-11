#!/usr/bin/env python3
"""
Future Stock Price Prediction
Generate sequences for future prediction based on stock symbol and datetime
"""

import pandas as pd
import numpy as np
from colors import *
import os
import pickle
import glob
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class FuturePredictionGenerator:
    """Generate sequences for future price prediction"""
    
    def __init__(self):
        self.metadata = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_cols = None
        self.stock_to_id = {}
        self.sequence_length = 60
        self.prediction_steps = 480  # Default prediction steps
        
    def load_metadata(self, data_dir='time_normalized_data'):
        """Load metadata from processed data"""
        try:
            # Find latest metadata file
            metadata_files = glob.glob(os.path.join(data_dir, 'metadata_*.pkl'))
            if not metadata_files:
                raise FileNotFoundError("No metadata file found")
                
            latest_metadata_file = max(metadata_files, key=os.path.getctime)
            
            with open(latest_metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.feature_scaler = self.metadata['feature_scaler']
            self.target_scaler = self.metadata['target_scaler']
            self.feature_cols = self.metadata['feature_cols']
            self.stock_to_id = self.metadata['stock_to_id']
            
            print_green(f"✓ Loaded metadata from {latest_metadata_file}")
            print_cyan(f"  Available stocks: {len(self.stock_to_id)}")
            print_cyan(f"  Feature columns: {len(self.feature_cols)}")
            
            return True
            
        except Exception as e:
            print_red(f"Error loading metadata: {e}")
            return False
    
    def load_latest_stock_data(self, stock_symbol, data_dir='data/feature'):
        """Load the latest data for a specific stock"""
        try:
            stock_file = os.path.join(data_dir, f'{stock_symbol.lower()}.csv')
            
            if not os.path.exists(stock_file):
                print_red(f"Stock data file not found: {stock_file}")
                return None
                
            df = pd.read_csv(stock_file)
            df['stock_symbol'] = stock_symbol.lower()
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Apply same feature engineering as in training
            df = self._create_robust_features(df)
            df = self._clean_data(df)
            
            # Sort by datetime
            df = df.sort_values('Datetime').reset_index(drop=True)
            
            print_green(f"✓ Loaded {len(df)} records for {stock_symbol}")
            print_cyan(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
            
            return df
            
        except Exception as e:
            print_red(f"Error loading stock data for {stock_symbol}: {e}")
            return None
    
    def _create_robust_features(self, df):
        """Create robust features (same as megaData_normalizer.py)"""
        df_processed = df.copy()
        
        # Time features
        df_processed['Hour'] = df_processed['Datetime'].dt.hour
        df_processed['DayOfWeek'] = df_processed['Datetime'].dt.dayofweek
        df_processed['Month'] = df_processed['Datetime'].dt.month
        
        # Cyclical encoding
        df_processed['Hour_sin'] = np.sin(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['Hour_cos'] = np.cos(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed['Month'] / 12)
        df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed['Month'] / 12)
        
        # Price features
        df_processed['Price_Range'] = df_processed['High'] - df_processed['Low']
        df_processed['Price_Range_Pct'] = self._safe_division(
            df_processed['Price_Range'], df_processed['Close'], 0.0
        )
        df_processed['Open_Close_Ratio'] = self._safe_division(
            df_processed['Open'], df_processed['Close'], 1.0
        )
        df_processed['High_Close_Ratio'] = self._safe_division(
            df_processed['High'], df_processed['Close'], 1.0
        )
        df_processed['Low_Close_Ratio'] = self._safe_division(
            df_processed['Low'], df_processed['Close'], 1.0
        )
        
        # Volume features
        if 'Volume' in df_processed.columns:
            df_processed['Volume'] = np.maximum(df_processed['Volume'], 0)
            df_processed['Volume_Log'] = np.log1p(df_processed['Volume'])
            df_processed['Volume_MA5'] = df_processed['Volume'].rolling(window=5, min_periods=1).mean()
            df_processed['Volume_Ratio'] = self._safe_division(
                df_processed['Volume'], df_processed['Volume_MA5'], 1.0
            )
        
        # Technical indicators
        if 'RSI' in df_processed.columns:
            df_processed['RSI'] = np.clip(df_processed['RSI'], 0, 100)
            df_processed['RSI_Normalized'] = df_processed['RSI'] / 100.0
            df_processed['RSI_Overbought'] = (df_processed['RSI'] > 70).astype(float)
            df_processed['RSI_Oversold'] = (df_processed['RSI'] < 30).astype(float)
        
        # MACD features
        if 'MACD' in df_processed.columns:
            df_processed['MACD_Signal_Diff'] = df_processed['MACD'] - df_processed.get('MACD_Signal', 0)
            df_processed['MACD_Positive'] = (df_processed['MACD'] > 0).astype(float)
        
        # Moving average features
        ma_cols = [col for col in df_processed.columns if col.startswith('MA')]
        for ma_col in ma_cols:
            if ma_col in df_processed.columns:
                ratio_col = f'{ma_col}_Price_Ratio'
                df_processed[ratio_col] = self._safe_division(
                    df_processed['Close'], df_processed[ma_col], 1.0
                )
        
        # Volatility features
        df_processed['Returns'] = df_processed['Close'].pct_change().fillna(0)
        df_processed['Returns_Abs'] = np.abs(df_processed['Returns'])
        df_processed['Volatility_5'] = df_processed['Returns'].rolling(window=5, min_periods=1).std().fillna(0)
        
        return df_processed
    
    def _safe_division(self, numerator, denominator, fill_value=0.0):
        """Safe division to avoid NaN"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, fill_value)
            return result
    
    def _clean_data(self, df):
        """Clean data thoroughly"""
        df_clean = df.copy()
        
        # Handle numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Replace infinite values
            df_clean[col] = np.where(np.isinf(df_clean[col]), np.nan, df_clean[col])
            
            # Handle extreme outliers
            if df_clean[col].notna().sum() > 0:
                q_low = df_clean[col].quantile(0.001)
                q_high = df_clean[col].quantile(0.999)
                df_clean[col] = np.clip(df_clean[col], q_low, q_high)
        
        # Fill NaN values
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Final check
        remaining_nans = df_clean.isnull().sum().sum()
        if remaining_nans > 0:
            print_red(f"Warning: {remaining_nans} NaN values remaining, replacing with 0")
            df_clean = df_clean.fillna(0.0)
        
        return df_clean
    
    def generate_future_sequence(self, stock_symbol, target_datetime, data_dir='data/feature'):
        """Generate sequence for future prediction"""
        print_green(f"Generating future sequence for {stock_symbol} at {target_datetime}")
        
        # Load stock data
        stock_data = self.load_latest_stock_data(stock_symbol, data_dir)
        if stock_data is None:
            return None
        
        # Convert target datetime
        target_dt = pd.to_datetime(target_datetime)
        
        # Find the closest available data point before target datetime
        available_data = stock_data[stock_data['Datetime'] <= target_dt]
        
        if len(available_data) < self.sequence_length:
            print_red(f"Insufficient historical data. Need at least {self.sequence_length} records, got {len(available_data)}")
            return None
        
        # Get the most recent sequence_length records
        recent_data = available_data.tail(self.sequence_length).copy()
        
        # Extract features
        try:
            features = recent_data[self.feature_cols].values
            
            # Normalize features using the same scaler from training
            features_normalized = self.feature_scaler.transform(features)
            
            # Get current price info
            current_price = recent_data['Close'].iloc[-1]
            current_open = recent_data['Open'].iloc[-1]
            current_datetime = recent_data['Datetime'].iloc[-1]
            
            # Create sequence ready for model prediction
            sequence = features_normalized.reshape(1, self.sequence_length, -1)
            
            prediction_info = {
                'sequence': sequence,
                'current_price': current_price,
                'current_open': current_open,
                'current_datetime': current_datetime,
                'target_datetime': target_dt,
                'stock_symbol': stock_symbol,
                'stock_id': self.stock_to_id.get(stock_symbol.lower(), -1),
                'sequence_shape': sequence.shape,
                'feature_names': self.feature_cols
            }
            
            print_green(f"✓ Generated sequence successfully")
            print_cyan(f"  Sequence shape: {sequence.shape}")
            print_cyan(f"  Current price: ${current_price:.2f}")
            print_cyan(f"  Current open: ${current_open:.2f}")
            print_cyan(f"  Data as of: {current_datetime}")
            print_cyan(f"  Predicting for: {target_dt}")
            
            return prediction_info
            
        except Exception as e:
            print_red(f"Error generating sequence: {e}")
            return None
    
    def predict_future_prices(self, model, stock_symbol, target_datetime, data_dir='data/feature'):
        """Complete pipeline: generate sequence and predict future prices"""
        print_green("=" * 80)
        print_green(f"FUTURE PRICE PREDICTION FOR {stock_symbol.upper()}")
        print_green("=" * 80)
        
        # Generate sequence
        prediction_info = self.generate_future_sequence(stock_symbol, target_datetime, data_dir)
        if prediction_info is None:
            return None
        
        # Make prediction
        try:
            print_cyan("Making prediction...")
            normalized_predictions = model.predict(prediction_info['sequence'], verbose=0)
            
            # Denormalize predictions
            predictions_flat = normalized_predictions.flatten().reshape(-1, 1)
            predictions_original = self.target_scaler.inverse_transform(predictions_flat).flatten()
            
            # Convert relative changes to actual prices
            current_price = prediction_info['current_price']
            predicted_prices = current_price * (1 + predictions_original)
            
            # Generate future timestamps (assuming 30-minute intervals)
            future_timestamps = []
            base_time = prediction_info['current_datetime']
            for i in range(len(predicted_prices)):
                future_timestamps.append(base_time + timedelta(minutes=30*(i+1)))
            
            # Create results
            results = {
                'stock_symbol': stock_symbol,
                'current_datetime': prediction_info['current_datetime'],
                'target_datetime': prediction_info['target_datetime'],
                'current_price': current_price,
                'current_open': prediction_info['current_open'],
                'predicted_prices': predicted_prices,
                'future_timestamps': future_timestamps,
                'prediction_changes': predictions_original,
                'prediction_count': len(predicted_prices)
            }
            
            print_green(f"✓ Prediction completed successfully")
            print_cyan(f"  Generated {len(predicted_prices)} future price predictions")
            print_cyan(f"  Price range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
            
            return results
            
        except Exception as e:
            print_red(f"Error making prediction: {e}")
            return None
    
    def save_predictions(self, results, output_file=None):
        """Save predictions to file"""
        if results is None:
            return
            
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"future_predictions_{results['stock_symbol']}_{timestamp}.csv"
        
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'datetime': results['future_timestamps'],
                'predicted_price': results['predicted_prices'],
                'price_change_pct': results['prediction_changes'] * 100
            })
            
            df.to_csv(output_file, index=False)
            print_green(f"✓ Predictions saved to {output_file}")
            
        except Exception as e:
            print_red(f"Error saving predictions: {e}")
    
    def display_predictions(self, results):
        """Display predictions in a formatted way"""
        if results is None:
            return
            
        print_green("\n" + "=" * 80)
        print_green("PREDICTION RESULTS")
        print_green("=" * 80)
        
        print_cyan(f"Stock: {results['stock_symbol'].upper()}")
        print_cyan(f"Current Price: ${results['current_price']:.2f}")
        print_cyan(f"Current Open: ${results['current_open']:.2f}")
        print_cyan(f"Data as of: {results['current_datetime']}")
        print_cyan(f"Predicting for: {results['target_datetime']}")
        
        print_green(f"\nFirst 10 Future Price Predictions:")
        for i in range(min(10, len(results['predicted_prices']))):
            timestamp = results['future_timestamps'][i]
            price = results['predicted_prices'][i]
            change_pct = results['prediction_changes'][i] * 100
            
            color_func = print_green if change_pct >= 0 else print_red
            color_func(f"  {timestamp}: ${price:.2f} ({change_pct:+.2f}%)")
        
        if len(results['predicted_prices']) > 10:
            print_cyan(f"\n... and {len(results['predicted_prices']) - 10} more predictions")

def main():
    """Example usage"""
    print_green("=" * 80)
    print_green("FUTURE STOCK PRICE PREDICTION SYSTEM")
    print_green("=" * 80)
    
    # Initialize predictor
    predictor = FuturePredictionGenerator()
    
    # Load metadata
    if not predictor.load_metadata():
        print_red("Failed to load metadata. Make sure you have run the data processing first.")
        return
    
    # Example usage
    stock_symbol = "AAPL"  # Example stock
    target_datetime = "2025-07-11 15:30:00"  # Example target time
    
    # Note: You would need to load your trained model here
    # model = load_model('your_trained_model.h5')
    
    print_yellow("Note: This is a template. You need to:")
    print_yellow("1. Load your trained model")
    print_yellow("2. Call predictor.predict_future_prices(model, stock_symbol, target_datetime)")
    print_yellow("3. Use predictor.display_predictions(results) to show results")
    
    # Show available stocks
    print_green(f"\nAvailable stocks ({len(predictor.stock_to_id)}):")
    for i, stock in enumerate(list(predictor.stock_to_id.keys())[:10]):
        print_cyan(f"  {stock.upper()}")
    if len(predictor.stock_to_id) > 10:
        print_cyan(f"  ... and {len(predictor.stock_to_id) - 10} more")

if __name__ == "__main__":
    main()
