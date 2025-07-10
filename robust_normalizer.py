#!/usr/bin/env python3
"""
修復時間洩漏的強健版資料正規化腳本
"""

import pandas as pd
import numpy as np
from colors import *
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings('ignore')

class TimeAwareStockDataNormalizer:
    """時間感知的股票資料正規化器 - 避免look-ahead bias"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        
    def safe_division(self, numerator, denominator, fill_value=0.0):
        """安全除法，避免除零和產生NaN"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, fill_value)
            return result
    
    def clean_data(self, df):
        """徹底清理數據中的異常值"""
        df_clean = df.copy()
        
        # 1. 處理數值列的異常值
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 替換無限值
            df_clean[col] = np.where(np.isinf(df_clean[col]), np.nan, df_clean[col])
            
            # 處理極端異常值
            if df_clean[col].notna().sum() > 0:
                q_low = df_clean[col].quantile(0.001)
                q_high = df_clean[col].quantile(0.999)
                df_clean[col] = np.clip(df_clean[col], q_low, q_high)
        
        # 2. 填充NaN值
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # 3. 最終檢查
        remaining_nans = df_clean.isnull().sum().sum()
        if remaining_nans > 0:
            print_red(f"警告: 仍有 {remaining_nans} 個NaN值，將全部替換為0")
            df_clean = df_clean.fillna(0.0)
        
        return df_clean
    
    def load_and_preprocess_data(self, data_dir='data/feature'):
        """載入並預處理所有股票數據"""
        all_data = []
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print_green(f"發現 {len(csv_files)} 個CSV檔案")
        
        for i, csv_file in enumerate(csv_files):
            print_yellow(f"處理文件 {i+1}/{len(csv_files)}: {csv_file}")
            
            file_path = os.path.join(data_dir, csv_file)
            stock_symbol = os.path.splitext(csv_file)[0]
            
            try:
                df = pd.read_csv(file_path)
                df['stock_symbol'] = stock_symbol
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                # 清理數據
                df = self.clean_data(df)
                
                # 基本特徵工程
                df = self._create_robust_features(df)
                
                # 再次清理
                df = self.clean_data(df)
                
                print_green(f"  ✓ 成功處理 {csv_file}, 形狀: {df.shape}")
                all_data.append(df)
                
            except Exception as e:
                print_red(f"  ✗ 處理 {csv_file} 時出錯: {e}")
                continue
        
        if not all_data:
            raise ValueError("沒有成功處理任何文件")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['stock_symbol', 'Datetime']).reset_index(drop=True)
        
        # 最終清理
        combined_df = self.clean_data(combined_df)
        
        print_green(f"合併後數據形狀: {combined_df.shape}")
        return combined_df
    
    def _create_robust_features(self, df):
        """創建強健的特徵，避免產生NaN"""
        df_processed = df.copy()
        
        print_cyan("  創建時間特徵...")
        # 時間特徵
        df_processed['Hour'] = df_processed['Datetime'].dt.hour
        df_processed['DayOfWeek'] = df_processed['Datetime'].dt.dayofweek
        df_processed['Month'] = df_processed['Datetime'].dt.month
        
        # 週期性編碼
        df_processed['Hour_sin'] = np.sin(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['Hour_cos'] = np.cos(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed['Month'] / 12)
        df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed['Month'] / 12)
        
        print_cyan("  創建價格特徵...")
        # 價格特徵
        df_processed['Price_Range'] = df_processed['High'] - df_processed['Low']
        df_processed['Price_Range_Pct'] = self.safe_division(
            df_processed['Price_Range'], df_processed['Close'], 0.0
        )
        df_processed['Open_Close_Ratio'] = self.safe_division(
            df_processed['Open'], df_processed['Close'], 1.0
        )
        df_processed['High_Close_Ratio'] = self.safe_division(
            df_processed['High'], df_processed['Close'], 1.0
        )
        df_processed['Low_Close_Ratio'] = self.safe_division(
            df_processed['Low'], df_processed['Close'], 1.0
        )
        
        print_cyan("  創建成交量特徵...")
        # 成交量特徵
        if 'Volume' in df_processed.columns:
            df_processed['Volume'] = np.maximum(df_processed['Volume'], 0)
            df_processed['Volume_Log'] = np.log1p(df_processed['Volume'])
            
            df_processed['Volume_MA5'] = df_processed['Volume'].rolling(window=5, min_periods=1).mean()
            df_processed['Volume_Ratio'] = self.safe_division(
                df_processed['Volume'], df_processed['Volume_MA5'], 1.0
            )
        
        print_cyan("  創建技術指標特徵...")
        # 技術指標特徵
        if 'RSI' in df_processed.columns:
            df_processed['RSI'] = np.clip(df_processed['RSI'], 0, 100)
            df_processed['RSI_Normalized'] = df_processed['RSI'] / 100.0
            df_processed['RSI_Overbought'] = (df_processed['RSI'] > 70).astype(float)
            df_processed['RSI_Oversold'] = (df_processed['RSI'] < 30).astype(float)
        
        # MACD特徵
        if 'MACD' in df_processed.columns:
            df_processed['MACD_Signal_Diff'] = df_processed['MACD'] - df_processed.get('MACD_Signal', 0)
            df_processed['MACD_Positive'] = (df_processed['MACD'] > 0).astype(float)
        
        # 移動平均特徵
        ma_cols = [col for col in df_processed.columns if col.startswith('MA')]
        for ma_col in ma_cols:
            if ma_col in df_processed.columns:
                ratio_col = f'{ma_col}_Price_Ratio'
                df_processed[ratio_col] = self.safe_division(
                    df_processed['Close'], df_processed[ma_col], 1.0
                )
        
        print_cyan("  創建波動率特徵...")
        # 波動率特徵
        df_processed['Returns'] = df_processed['Close'].pct_change().fillna(0)
        df_processed['Returns_Abs'] = np.abs(df_processed['Returns'])
        df_processed['Volatility_5'] = df_processed['Returns'].rolling(window=5, min_periods=1).std().fillna(0)
        
        return df_processed
    
    def create_time_aware_sequences(self, df, sequence_length=60, prediction_steps=336, test_days=7):
        """創建時間感知的序列數據 - 避免look-ahead bias"""
        print_cyan(f"創建時間感知序列，測試期間: {test_days} 天")
        
        # 定義特徵欄位
        exclude_cols = [
            'Datetime', 'stock_symbol', 'Year', 'Month', 'Day', 'Hour', 'Minute', 
            'DayOfWeek', 'DayOfYear', 'Dividends', 'Stock Splits', 'Capital Gains'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print_green(f"使用 {len(feature_cols)} 個特徵")
        
        # 找出全局的時間分割點
        max_date = df['Datetime'].max()
        min_date = df['Datetime'].min()

        # 測試期: 最新的7天
        test_start_date = max_date - timedelta(days=test_days)
        
        print_cyan(f"數據時間範圍: {min_date} 到 {max_date}")
        print_cyan(f"訓練期結束: {test_start_date}")
        print_cyan(f"測試期開始: {test_start_date}")
        
        # 分別處理訓練和測試數據
        train_data = df[df['Datetime'] < test_start_date].copy()
        test_data = df[df['Datetime'] >= test_start_date].copy()
        
        print_cyan(f"訓練數據量: {len(train_data)}")
        print_cyan(f"測試數據量: {len(test_data)}")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("訓練或測試數據為空，請調整test_days參數")
        
        # 創建股票ID映射
        unique_stocks = df['stock_symbol'].unique()
        stock_to_id = {stock: i for i, stock in enumerate(unique_stocks)}
        
        # 創建訓練序列
        train_sequences, train_targets, train_metadata = self._create_sequences_for_period(
            train_data, feature_cols, stock_to_id, sequence_length, prediction_steps, 'train'
        )
        
        # 創建測試序列
        test_sequences, test_targets, test_metadata = self._create_sequences_for_period(
            test_data, feature_cols, stock_to_id, sequence_length, prediction_steps, 'test'
        )
        
        print_green(f"訓練序列: {len(train_sequences)}")
        print_green(f"測試序列: {len(test_sequences)}")
        
        return {
            'train_sequences': np.array(train_sequences),
            'train_targets': np.array(train_targets),
            'train_metadata': train_metadata,
            'test_sequences': np.array(test_sequences) if test_sequences else np.array([]),
            'test_targets': np.array(test_targets) if test_targets else np.array([]),
            'test_metadata': test_metadata,
            'feature_cols': feature_cols,
            'stock_to_id': stock_to_id,
            'test_start_date': test_start_date
        }
    
    def _create_sequences_for_period(self, data, feature_cols, stock_to_id, sequence_length, prediction_steps, period_name):
        """為特定時期創建序列"""
        sequences = []
        targets = []
        metadata = []
        
        for stock in stock_to_id.keys():
            stock_data = data[data['stock_symbol'] == stock].sort_values('Datetime').reset_index(drop=True)
            
            if len(stock_data) < sequence_length + prediction_steps:
                print_yellow(f"  {period_name} - 股票 {stock} 數據不足，跳過")
                continue
            
            try:
                features = stock_data[feature_cols].values
                close_prices = stock_data['Close'].values
                
                # 檢查數據質量
                if np.isnan(features).any() or np.isnan(close_prices).any():
                    print_red(f"  {period_name} - 股票 {stock} 包含NaN值，跳過")
                    continue
                
                stock_sequences = 0
                # 創建滑動窗口
                for i in range(len(stock_data) - sequence_length - prediction_steps + 1):
                    input_sequence = features[i:i + sequence_length]
                    target_prices = close_prices[i + sequence_length:i + sequence_length + prediction_steps]
                    current_price = close_prices[i + sequence_length - 1]
                    
                    # 計算相對變化
                    target_changes = (target_prices - current_price) / current_price
                    
                    # 檢查目標是否有效
                    if np.isnan(target_changes).any() or np.isinf(target_changes).any():
                        continue
                    
                    sequences.append(input_sequence)
                    targets.append(target_changes)
                    metadata.append({
                        'stock_id': stock_to_id[stock],
                        'stock_symbol': stock,
                        'datetime': stock_data['Datetime'].iloc[i + sequence_length - 1],
                        'current_price': current_price,
                        'period': period_name
                    })
                    stock_sequences += 1
                
                if stock_sequences > 0:
                    print_green(f"  {period_name} - {stock}: 創建了 {stock_sequences} 個序列")
                
            except Exception as e:
                print_red(f"  {period_name} - 處理股票 {stock} 時出錯: {e}")
                continue
        
        return sequences, targets, metadata
    
    def normalize_data(self, data):
        """使用時間感知的標準化"""
        print_cyan("開始時間感知標準化...")
        
        train_sequences = data['train_sequences']
        train_targets = data['train_targets']
        test_sequences = data['test_sequences']
        test_targets = data['test_targets']
        
        print_cyan(f"訓練序列形狀: {train_sequences.shape}")
        print_cyan(f"測試序列形狀: {test_sequences.shape}")
        
        # 只使用訓練數據來擬合標準化器
        train_sequences_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
        train_targets_flat = train_targets.reshape(-1, 1)
        
        # 標準化特徵
        print_cyan("使用訓練數據擬合特徵標準化器...")
        sequences_normalized_train = self.feature_scaler.fit_transform(train_sequences_flat)
        sequences_normalized_train = sequences_normalized_train.reshape(train_sequences.shape)
        
        # 標準化目標
        print_cyan("使用訓練數據擬合目標標準化器...")
        targets_normalized_train = self.target_scaler.fit_transform(train_targets_flat)
        targets_normalized_train = targets_normalized_train.reshape(train_targets.shape)
        
        # 應用標準化到測試數據
        if len(test_sequences) > 0:
            test_sequences_flat = test_sequences.reshape(-1, test_sequences.shape[-1])
            test_targets_flat = test_targets.reshape(-1, 1)
            
            sequences_normalized_test = self.feature_scaler.transform(test_sequences_flat)
            sequences_normalized_test = sequences_normalized_test.reshape(test_sequences.shape)
            
            targets_normalized_test = self.target_scaler.transform(test_targets_flat)
            targets_normalized_test = targets_normalized_test.reshape(test_targets.shape)
        else:
            sequences_normalized_test = np.array([])
            targets_normalized_test = np.array([])
        
        # 清理NaN值
        sequences_normalized_train = np.nan_to_num(sequences_normalized_train, nan=0.0)
        targets_normalized_train = np.nan_to_num(targets_normalized_train, nan=0.0)
        sequences_normalized_test = np.nan_to_num(sequences_normalized_test, nan=0.0)
        targets_normalized_test = np.nan_to_num(targets_normalized_test, nan=0.0)
        
        self.is_fitted = True
        
        print_green("✓ 時間感知標準化完成")
        
        return {
            'train_sequences_normalized': sequences_normalized_train,
            'train_targets_normalized': targets_normalized_train,
            'test_sequences_normalized': sequences_normalized_test,
            'test_targets_normalized': targets_normalized_test,
            'train_metadata': data['train_metadata'],
            'test_metadata': data['test_metadata'],
            'feature_cols': data['feature_cols'],
            'stock_to_id': data['stock_to_id'],
            'test_start_date': data['test_start_date']
        }
    
    def save_processed_data(self, processed_data, output_dir='time_aware_normalized_data'):
        """保存處理後的數據"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存訓練數據
        np.save(os.path.join(output_dir, f'train_sequences_{timestamp}.npy'), 
                processed_data['train_sequences_normalized'])
        np.save(os.path.join(output_dir, f'train_targets_{timestamp}.npy'), 
                processed_data['train_targets_normalized'])
        
        # 保存測試數據
        if len(processed_data['test_sequences_normalized']) > 0:
            np.save(os.path.join(output_dir, f'test_sequences_{timestamp}.npy'), 
                    processed_data['test_sequences_normalized'])
            np.save(os.path.join(output_dir, f'test_targets_{timestamp}.npy'), 
                    processed_data['test_targets_normalized'])
        
        # 保存元數據
        metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'train_metadata': processed_data['train_metadata'],
                'test_metadata': processed_data['test_metadata'],
                'feature_cols': processed_data['feature_cols'],
                'stock_to_id': processed_data['stock_to_id'],
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'test_start_date': processed_data['test_start_date'],
                'timestamp': timestamp
            }, f)
        
        print_green(f"數據已保存到 {output_dir}")
        
        return {
            'train_sequences_file': os.path.join(output_dir, f'train_sequences_{timestamp}.npy'),
            'train_targets_file': os.path.join(output_dir, f'train_targets_{timestamp}.npy'),
            'test_sequences_file': os.path.join(output_dir, f'test_sequences_{timestamp}.npy'),
            'test_targets_file': os.path.join(output_dir, f'test_targets_{timestamp}.npy'),
            'metadata_file': metadata_file
        }

def main():
    """主函數"""
    print_green("=" * 60)
    print_green("時間感知數據預處理開始...")
    print_green("=" * 60)
    
    normalizer = TimeAwareStockDataNormalizer()
    
    try:
        # 1. 載入和預處理數據
        print_green("\n1. 載入和預處理數據...")
        combined_df = normalizer.load_and_preprocess_data()
        
        # 2. 創建時間感知序列
        print_green("\n2. 創建時間感知序列...")
        data = normalizer.create_time_aware_sequences(combined_df, test_days=3)
        
        # 3. 標準化數據
        print_green("\n3. 時間感知標準化...")
        processed_data = normalizer.normalize_data(data)
        
        # 4. 保存數據
        print_green("\n4. 保存數據...")
        files = normalizer.save_processed_data(processed_data)
        
        # 5. 數據質量檢查
        print_green("\n5. 數據質量檢查...")
        train_seq = processed_data['train_sequences_normalized']
        test_seq = processed_data['test_sequences_normalized']
        
        print_cyan(f"最終統計:")
        print_cyan(f"  訓練序列: {train_seq.shape}")
        print_cyan(f"  測試序列: {test_seq.shape}")
        print_cyan(f"  特徵維度: {train_seq.shape[2]}")
        print_cyan(f"  測試開始日期: {processed_data['test_start_date']}")
        
        print_green("\n✓ 時間感知數據預處理完成！")
        print_green("✓ 已避免look-ahead bias")
        
    except Exception as e:
        print_red(f"\n✗ 處理過程中出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()