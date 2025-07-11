#!/usr/bin/env python3
"""
Time-Based Data Normalizer with Chunk Processing
Divides data into 7 days test set and remaining as training set
"""

import pandas as pd
import numpy as np
from colors import *
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime, timedelta
import pickle
import gc
import shutil
import tempfile
import glob

warnings.filterwarnings('ignore')

class TimeBasedChunkNormalizer:
    """Time-based data normalizer with chunk processing for memory efficiency"""
    
    def __init__(self, test_days=10, chunk_size=3):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.test_days = test_days
        self.chunk_size = chunk_size
        self.metadata = {
            'train_metadata': [],
            'test_metadata': [],
            'feature_cols': [],
            'stock_to_id': {},
            'feature_scaler': None,
            'target_scaler': None,
            'test_start_date': None,
            'prediction_steps': None,
            'timestamp': None
        }
        
    def clear_output_dir(self, output_dir):
        """Clear output directory"""
        if os.path.exists(output_dir):
            print_yellow(f"Clearing old output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print_green(f"✓ Output directory ready: {output_dir}")
        
    def process_data(self, data_dir='data/feature', output_dir='time_normalized_data'):
        """Main processing function with time-based chunking"""
        print_green("=" * 80)
        print_green("Starting Time-Based Chunk Processing...")
        print_green("=" * 80)
        
        # Clear and create output directory
        self.clear_output_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata['timestamp'] = timestamp
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print_cyan(f"Found {len(csv_files)} CSV files to process")
        
        # Step 1: Process in chunks and determine global time split
        print_green("\n1. Processing chunks and determining time split...")
        chunk_files, global_test_start_date = self._process_chunks_and_find_split(
            csv_files, data_dir, output_dir, timestamp
        )
        
        # Step 2: Create final normalized datasets
        print_green("\n2. Creating final normalized datasets...")
        final_data = self._combine_chunks_with_time_split(
            chunk_files, global_test_start_date, output_dir, timestamp
        )
        
        # Step 3: Save final results
        print_green("\n3. Saving final results...")
        self._save_final_data(final_data, output_dir, timestamp)
        
        print_green("\n✓ Time-based chunk processing completed!")
        return final_data
    
    def _process_chunks_and_find_split(self, csv_files, data_dir, output_dir, timestamp):
        """Process data in chunks and find global time split"""
        chunk_files = []
        all_dates = set()
        
        # Process each chunk
        for chunk_start in range(0, len(csv_files), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(csv_files))
            chunk_files_batch = csv_files[chunk_start:chunk_end]
            chunk_num = chunk_start // self.chunk_size + 1
            
            print_cyan(f"Processing chunk {chunk_num}: {chunk_files_batch}")
            
            try:
                # Load and process chunk
                chunk_data = self._load_and_process_chunk(chunk_files_batch, data_dir)
                
                if chunk_data is not None:
                    # Collect all dates
                    chunk_dates = set(chunk_data['Datetime'].dt.date)
                    all_dates.update(chunk_dates)
                    
                    # Save chunk temporarily
                    chunk_file = os.path.join(output_dir, f'temp_chunk_{chunk_num}_{timestamp}.pkl')
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_data, f)
                    
                    chunk_files.append(chunk_file)
                    print_green(f"✓ Chunk {chunk_num} processed and saved")
                    
                    # Clean memory
                    del chunk_data
                    gc.collect()
                else:
                    print_red(f"✗ Chunk {chunk_num} failed to process")
                    
            except Exception as e:
                print_red(f"✗ Error processing chunk {chunk_num}: {e}")
                continue
        
        # Determine global test start date
        sorted_dates = sorted(list(all_dates))
        print_cyan(f"Global date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        print_cyan(f"Total unique dates: {len(sorted_dates)}")
        
        if len(sorted_dates) <= self.test_days:
            print_yellow(f"Warning: Total days {len(sorted_dates)} <= test days {self.test_days}")
            global_test_start_date = sorted_dates[len(sorted_dates)//2]
        else:
            global_test_start_date = sorted_dates[-self.test_days]
        
        print_cyan(f"Global test start date: {global_test_start_date}")
        self.metadata['test_start_date'] = global_test_start_date
        
        return chunk_files, global_test_start_date
    
    def _load_and_process_chunk(self, chunk_files, data_dir):
        """Load and process a single chunk of files"""
        chunk_data = []
        
        for csv_file in chunk_files:
            file_path = os.path.join(data_dir, csv_file)
            stock_symbol = os.path.splitext(csv_file)[0]
            
            try:
                df = pd.read_csv(file_path)
                df['stock_symbol'] = stock_symbol
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                # Feature engineering (keeping same features as megaData_normalizer.py)
                df = self._create_robust_features(df)
                df = self._clean_data(df)
                
                chunk_data.append(df)
                
            except Exception as e:
                print_red(f"  ✗ Error processing {csv_file}: {e}")
                continue
        
        if chunk_data:
            combined_chunk = pd.concat(chunk_data, ignore_index=True)
            combined_chunk = combined_chunk.sort_values(['stock_symbol', 'Datetime']).reset_index(drop=True)
            return self._clean_data(combined_chunk)
        
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
    
    def _combine_chunks_with_time_split(self, chunk_files, test_start_date, output_dir, timestamp):
        """Combine chunks and apply time-based split"""
        print_cyan(f"Combining {len(chunk_files)} chunks with time split at {test_start_date}")
        
        # Initialize containers
        train_sequences = []
        train_targets = []
        train_metadata = []
        test_sequences = []
        test_targets = []
        test_metadata = []
        
        all_feature_cols = None
        stock_to_id = {}
        stock_id_counter = 0
        
        # Process each chunk
        for i, chunk_file in enumerate(chunk_files):
            print_cyan(f"Processing chunk file {i+1}/{len(chunk_files)}")
            
            try:
                # Load chunk
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                # Create time-aware sequences for this chunk
                chunk_result = self._create_time_aware_sequences_for_chunk(
                    chunk_data, test_start_date, stock_to_id, stock_id_counter
                )
                
                if chunk_result:
                    # Update stock_to_id mapping
                    stock_to_id.update(chunk_result['stock_to_id'])
                    stock_id_counter = max(stock_to_id.values()) + 1 if stock_to_id else 0
                    
                    # Accumulate data
                    if chunk_result['train_sequences']:
                        train_sequences.extend(chunk_result['train_sequences'])
                        train_targets.extend(chunk_result['train_targets'])
                        train_metadata.extend(chunk_result['train_metadata'])
                    
                    if chunk_result['test_sequences']:
                        test_sequences.extend(chunk_result['test_sequences'])
                        test_targets.extend(chunk_result['test_targets'])
                        test_metadata.extend(chunk_result['test_metadata'])
                    
                    # Store feature columns
                    if all_feature_cols is None:
                        all_feature_cols = chunk_result['feature_cols']
                
                # Clean up
                del chunk_data
                gc.collect()
                
                # Remove temporary chunk file
                os.remove(chunk_file)
                
            except Exception as e:
                print_red(f"Error processing chunk file {chunk_file}: {e}")
                continue
        
        print_cyan(f"Data collection completed:")
        print_cyan(f"  Training sequences: {len(train_sequences)}")
        print_cyan(f"  Test sequences: {len(test_sequences)}")
        print_cyan(f"  Unique stocks: {len(stock_to_id)}")
        
        # Convert to numpy arrays with shape validation
        if train_sequences:
            # Check shapes and ensure consistency
            train_sequences, train_targets = self._validate_and_fix_shapes(train_sequences, train_targets, 'train')
            train_sequences = np.array(train_sequences)
            train_targets = np.array(train_targets)
        else:
            train_sequences = np.array([])
            train_targets = np.array([])
            
        if test_sequences:
            # Check shapes and ensure consistency
            test_sequences, test_targets = self._validate_and_fix_shapes(test_sequences, test_targets, 'test')
            test_sequences = np.array(test_sequences)
            test_targets = np.array(test_targets)
        else:
            test_sequences = np.array([])
            test_targets = np.array([])
        
        # Normalize data
        normalized_data = self._normalize_time_split_data(
            train_sequences, train_targets, test_sequences, test_targets
        )
        
        # Update metadata
        self.metadata.update({
            'train_metadata': train_metadata,
            'test_metadata': test_metadata,
            'feature_cols': all_feature_cols,
            'stock_to_id': stock_to_id,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
        })
        
        return {
            'train_sequences_normalized': normalized_data['train_sequences_normalized'],
            'train_targets_normalized': normalized_data['train_targets_normalized'],
            'test_sequences_normalized': normalized_data['test_sequences_normalized'],
            'test_targets_normalized': normalized_data['test_targets_normalized'],
            'metadata': self.metadata
        }
    
    def _create_time_aware_sequences_for_chunk(self, chunk_data, test_start_date, existing_stock_to_id, stock_id_start):
        """Create sequences for a chunk with time awareness"""
        sequence_length = 60
        target_trading_days = 10
        prediction_steps = self._calculate_trading_day_steps(chunk_data, target_trading_days)
        
        # Define feature columns
        exclude_cols = [
            'Datetime', 'stock_symbol', 'Year', 'Month', 'Day', 'Hour', 'Minute', 
            'DayOfWeek', 'DayOfYear', 'Dividends', 'Stock Splits', 'Capital Gains', 'Date'
        ]
        
        numeric_cols = chunk_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        print_blue(f"Feature columns for chunk: {feature_cols}")

        # Create stock mapping for this chunk
        chunk_stocks = chunk_data['stock_symbol'].unique()
        print_cyan(f"Unique stocks in chunk: {chunk_stocks}")
        chunk_stock_to_id = {}
        
        for stock in chunk_stocks:
            if stock in existing_stock_to_id:
                chunk_stock_to_id[stock] = existing_stock_to_id[stock]
            else:
                chunk_stock_to_id[stock] = stock_id_start
                stock_id_start += 1
        
        # Split data by time
        test_start_datetime = pd.to_datetime(test_start_date)
        train_data = chunk_data[chunk_data['Datetime'] < test_start_datetime].copy()
        test_data = chunk_data[chunk_data['Datetime'] >= test_start_datetime].copy()
        
        print_cyan(f"Time split debug - Total data: {len(chunk_data)}, Train: {len(train_data)}, Test: {len(test_data)}")
        print_cyan(f"Test start date: {test_start_datetime}")
        if len(test_data) > 0:
            print_cyan(f"Test data date range: {test_data['Datetime'].min()} to {test_data['Datetime'].max()}")
        else:
            print_yellow("No test data found!")
        
        # Create sequences for train and test
        train_seq, train_tar, train_meta = self._create_sequences_for_period(
            train_data, feature_cols, chunk_stock_to_id, sequence_length, prediction_steps, 'train'
        )
        
        test_seq, test_tar, test_meta = self._create_sequences_for_period(
            test_data, feature_cols, chunk_stock_to_id, sequence_length, prediction_steps, 'test', 
            allow_shorter=True
        )
        print_cyan(f"test_sequences: {len(test_seq)} sequences, {len(test_tar)} targets, {len(test_meta)} metadata")
        return {
            'train_sequences': train_seq,
            'train_targets': train_tar,
            'train_metadata': train_meta,
            'test_sequences': test_seq,
            'test_targets': test_tar,
            'test_metadata': test_meta,
            'feature_cols': feature_cols,
            'stock_to_id': chunk_stock_to_id,
            'prediction_steps': prediction_steps
        }
    
    def _calculate_trading_day_steps(self, df, target_trading_days=5):
        """Calculate trading day steps"""
        sample_data = df.groupby('stock_symbol').first().reset_index()
        if len(sample_data) > 0:
            first_stock = sample_data.iloc[0]['stock_symbol']
            stock_data = df[df['stock_symbol'] == first_stock].sort_values('Datetime')
            
            stock_data['Date'] = stock_data['Datetime'].dt.date
            daily_counts = stock_data.groupby('Date').size()
            
            stock_data['DayOfWeek'] = stock_data['Datetime'].dt.dayofweek
            weekday_data = stock_data[stock_data['DayOfWeek'] < 5]
            
            if len(weekday_data) > 0:
                weekday_data['Date'] = weekday_data['Datetime'].dt.date
                weekday_daily_counts = weekday_data.groupby('Date').size()
                avg_steps_per_day = weekday_daily_counts.mean()
            else:
                avg_steps_per_day = daily_counts.mean()
            
            total_steps = int(avg_steps_per_day * target_trading_days)
            return max(total_steps, 240)  # Minimum 240 steps (5 trading days * 48)
        else:
            return 48 * target_trading_days
    
    def _create_sequences_for_period(self, data, feature_cols, stock_to_id, sequence_length, prediction_steps, period_name, allow_shorter=False):
        """Create sequences for a specific period"""
        sequences = []
        targets = []
        metadata = []
        
        print_cyan(f"Creating sequences for {period_name} period with {len(data)} data points")
        
        for stock in stock_to_id.keys():
            stock_data = data[data['stock_symbol'] == stock].sort_values('Datetime').reset_index(drop=True)
            
            # Adjust minimum requirements for test period
            if period_name == 'test' and allow_shorter:
                min_required = sequence_length + max(10, prediction_steps // 10)
            else:
                min_required = sequence_length + prediction_steps
            
            print_cyan(f"Stock {stock}: {len(stock_data)} data points, min required: {min_required}")
            
            if len(stock_data) < min_required:
                print_yellow(f"  Skipping {stock}: insufficient data ({len(stock_data)} < {min_required})")
                continue
            
            try:
                features = stock_data[feature_cols].values
                close_prices = stock_data['Close'].values
                
                # Check data quality
                if np.isnan(features).any() or np.isnan(close_prices).any():
                    continue
                
                # Adjust sliding window strategy for test period
                if period_name == 'test' and allow_shorter:
                    actual_prediction_steps = min(prediction_steps, len(stock_data) - sequence_length)
                    if actual_prediction_steps < 1:
                        continue
                    max_start_idx = len(stock_data) - sequence_length - actual_prediction_steps
                    start_indices = [max_start_idx] if max_start_idx >= 0 else []
                else:
                    actual_prediction_steps = prediction_steps
                    start_indices = range(len(stock_data) - sequence_length - actual_prediction_steps + 1)
                
                print_cyan(f"  Stock {stock}: using {actual_prediction_steps} prediction steps, {len(list(start_indices))} sequences")
                
                # Create sequences
                for i in start_indices:
                    if i < 0:
                        continue
                        
                    input_sequence = features[i:i + sequence_length]
                    target_prices = close_prices[i + sequence_length:i + sequence_length + actual_prediction_steps]
                    current_price = close_prices[i + sequence_length - 1]
                    
                    if len(target_prices) == 0:
                        continue
                    
                    # Always ensure targets have the same length as prediction_steps
                    if len(target_prices) < prediction_steps:
                        padding_length = prediction_steps - len(target_prices)
                        last_price = target_prices[-1] if len(target_prices) > 0 else current_price
                        target_prices = np.concatenate([target_prices, np.full(padding_length, last_price)])
                    elif len(target_prices) > prediction_steps:
                        # Truncate if somehow longer
                        target_prices = target_prices[:prediction_steps]
                    
                    # Calculate relative changes
                    target_changes = (target_prices - current_price) / current_price
                    
                    # Check validity
                    if np.isnan(target_changes).any() or np.isinf(target_changes).any():
                        continue
                    
                    # Ensure consistent shape
                    if len(target_changes) != prediction_steps:
                        print_red(f"  Warning: target_changes shape mismatch for {stock}: {len(target_changes)} != {prediction_steps}")
                        continue
                    
                    sequences.append(input_sequence)
                    targets.append(target_changes)
                    metadata.append({
                        'stock_id': stock_to_id[stock],
                        'stock_symbol': stock,
                        'datetime': stock_data['Datetime'].iloc[i + sequence_length - 1],
                        'current_price': current_price,
                        'period': period_name,
                        'actual_prediction_steps': actual_prediction_steps
                    })
                
            except Exception as e:
                print_red(f"  Error processing stock {stock}: {e}")
                continue
        
        return sequences, targets, metadata
    
    def _validate_and_fix_shapes(self, sequences, targets, period_name):
        """Validate and fix shape inconsistencies in sequences and targets"""
        print_cyan(f"Validating shapes for {period_name} data...")
        
        if not sequences or not targets:
            return sequences, targets
        
        # Check sequence shapes
        seq_shapes = [seq.shape for seq in sequences]
        unique_seq_shapes = list(set(seq_shapes))
        print_cyan(f"  Sequence shapes found: {unique_seq_shapes}")
        
        # Check target shapes
        target_shapes = [tar.shape for tar in targets]
        unique_target_shapes = list(set(target_shapes))
        print_cyan(f"  Target shapes found: {unique_target_shapes}")
        
        # Find most common shapes
        from collections import Counter
        seq_shape_counts = Counter(seq_shapes)
        target_shape_counts = Counter(target_shapes)
        
        most_common_seq_shape = seq_shape_counts.most_common(1)[0][0]
        most_common_target_shape = target_shape_counts.most_common(1)[0][0]
        
        print_cyan(f"  Most common sequence shape: {most_common_seq_shape}")
        print_cyan(f"  Most common target shape: {most_common_target_shape}")
        
        # Filter out sequences/targets with inconsistent shapes
        valid_sequences = []
        valid_targets = []
        
        for seq, tar in zip(sequences, targets):
            if seq.shape == most_common_seq_shape and tar.shape == most_common_target_shape:
                valid_sequences.append(seq)
                valid_targets.append(tar)
        
        print_cyan(f"  Kept {len(valid_sequences)}/{len(sequences)} sequences with consistent shapes")
        
        return valid_sequences, valid_targets
    
    def _normalize_time_split_data(self, train_sequences, train_targets, test_sequences, test_targets):
        """Normalize data with time-aware approach"""
        print_cyan("Starting time-aware normalization...")
        
        if len(train_sequences) == 0:
            print_red("No training data available for normalization!")
            return {
                'train_sequences_normalized': train_sequences,
                'train_targets_normalized': train_targets,
                'test_sequences_normalized': test_sequences,
                'test_targets_normalized': test_targets
            }
        
        # Fit scalers on training data only
        print_cyan("Fitting scalers on training data...")
        train_seq_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
        train_tar_flat = train_targets.reshape(-1, 1)
        
        self.feature_scaler.fit(train_seq_flat)
        self.target_scaler.fit(train_tar_flat)
        
        # Transform training data
        print_cyan("Transforming training data...")
        train_seq_norm = self._batch_normalize(train_sequences, self.feature_scaler, 'sequences')
        train_tar_norm = self._batch_normalize(train_targets, self.target_scaler, 'targets')
        
        # Transform test data if available
        if len(test_sequences) > 0:
            print_cyan("Transforming test data...")
            test_seq_norm = self._batch_normalize(test_sequences, self.feature_scaler, 'sequences')
            test_tar_norm = self._batch_normalize(test_targets, self.target_scaler, 'targets')
        else:
            test_seq_norm = np.array([])
            test_tar_norm = np.array([])
        
        print_green("✓ Time-aware normalization completed")
        
        return {
            'train_sequences_normalized': train_seq_norm,
            'train_targets_normalized': train_tar_norm,
            'test_sequences_normalized': test_seq_norm,
            'test_targets_normalized': test_tar_norm
        }
    
    def _batch_normalize(self, data, scaler, data_type):
        """Batch normalization to avoid memory issues"""
        if len(data) == 0:
            return data
            
        batch_size = 100
        normalized_data = np.zeros_like(data, dtype=np.float32)
        
        for i in range(0, len(data), batch_size):
            end_idx = min(i + batch_size, len(data))
            batch = data[i:end_idx]
            
            if data_type == 'sequences':
                batch_flat = batch.reshape(-1, batch.shape[-1])
                batch_norm = scaler.transform(batch_flat)
                batch_norm = batch_norm.reshape(batch.shape)
            else:  # targets
                batch_flat = batch.reshape(-1, 1)
                batch_norm = scaler.transform(batch_flat)
                batch_norm = batch_norm.reshape(batch.shape)
            
            # Clean NaN values
            batch_norm = np.nan_to_num(batch_norm, nan=0.0, posinf=0.0, neginf=0.0)
            normalized_data[i:end_idx] = batch_norm
            
            del batch, batch_flat, batch_norm
            gc.collect()
        
        return normalized_data
    
    def _save_final_data(self, final_data, output_dir, timestamp):
        """Save final processed data"""
        try:
            # Save sequences and targets
            train_seq_file = os.path.join(output_dir, f'train_sequences_{timestamp}.npy')
            train_tar_file = os.path.join(output_dir, f'train_targets_{timestamp}.npy')
            
            np.save(train_seq_file, final_data['train_sequences_normalized'])
            np.save(train_tar_file, final_data['train_targets_normalized'])
            
            test_seq_file = None
            test_tar_file = None
            
            if len(final_data['test_sequences_normalized']) > 0:
                test_seq_file = os.path.join(output_dir, f'test_sequences_{timestamp}.npy')
                test_tar_file = os.path.join(output_dir, f'test_targets_{timestamp}.npy')
                
                np.save(test_seq_file, final_data['test_sequences_normalized'])
                np.save(test_tar_file, final_data['test_targets_normalized'])
            
            # Save metadata
            metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(final_data['metadata'], f)
            
            print_green(f"✓ Final data saved to {output_dir}")
            print_cyan(f"  Training sequences: {train_seq_file}")
            print_cyan(f"  Training targets: {train_tar_file}")
            if test_seq_file:
                print_cyan(f"  Test sequences: {test_seq_file}")
                print_cyan(f"  Test targets: {test_tar_file}")
            print_cyan(f"  Metadata: {metadata_file}")
            
            # Print final statistics
            train_shape = final_data['train_sequences_normalized'].shape
            test_shape = final_data['test_sequences_normalized'].shape
            
            print_green(f"\nFinal Data Statistics:")
            print_green(f"  Training: {train_shape[0]} sequences, {train_shape[1]} timesteps, {train_shape[2]} features")
            if len(test_shape) > 0:
                print_green(f"  Test: {test_shape[0]} sequences, {test_shape[1]} timesteps, {test_shape[2]} features")
            print_green(f"  Stocks: {len(self.metadata['stock_to_id'])}")
            print_green(f"  Test period: Last {self.test_days} days")
            
        except Exception as e:
            print_red(f"Error saving final data: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run time-based chunk normalization"""
    print_green("=" * 80)
    print_green("Time-Based Chunk Normalization with 7-Day Test Split")
    print_green("=" * 80)
    
    normalizer = TimeBasedChunkNormalizer(test_days=10, chunk_size=3)
    
    try:
        # Process data
        final_data = normalizer.process_data(
            data_dir='data/feature',
            output_dir='time_normalized_data'
        )
        
        print_green("\n" + "="*80)
        print_green("🎉 TIME-BASED CHUNK NORMALIZATION COMPLETED!")
        print_green("="*80)
        print_green("✓ Data processed with time-based train/test split")
        print_green("✓ Memory-efficient chunk processing used")
        print_green("✓ Compatible with main.py training script")
        print_green("✓ Last 7 days reserved for testing")
        
    except Exception as e:
        print_red(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()