#!/usr/bin/env python3
"""
更強健的資料正規化腳本 - 徹底解決NaN問題
"""

import pandas as pd
import numpy as np
from colors import *
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

class RobustStockDataNormalizer:
    """強健版股票資料正規化器"""
    
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
            
            # 處理極端異常值 (超過99.9%分位數或低於0.1%分位數)
            if df_clean[col].notna().sum() > 0:
                q_low = df_clean[col].quantile(0.001)
                q_high = df_clean[col].quantile(0.999)
                df_clean[col] = np.clip(df_clean[col], q_low, q_high)
        
        # 2. 填充NaN值
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # 使用中位數填充
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
        # 價格特徵 - 使用安全除法
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
            # 確保Volume非負
            df_processed['Volume'] = np.maximum(df_processed['Volume'], 0)
            df_processed['Volume_Log'] = np.log1p(df_processed['Volume'])
            
            # 成交量移動平均
            df_processed['Volume_MA5'] = df_processed['Volume'].rolling(window=5, min_periods=1).mean()
            df_processed['Volume_Ratio'] = self.safe_division(
                df_processed['Volume'], df_processed['Volume_MA5'], 1.0
            )
        
        print_cyan("  創建技術指標特徵...")
        # 技術指標特徵
        if 'RSI' in df_processed.columns:
            # 確保RSI在合理範圍內
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
                # MA與當前價格的比率
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
    
    def create_sequences_and_targets(self, df, sequence_length=60, prediction_steps=336):
        """創建序列數據和目標變數"""
        all_sequences = []
        all_targets = []
        all_metadata = []
        
        # 定義特徵欄位 - 排除非數值和標識列
        exclude_cols = [
            'Datetime', 'stock_symbol', 'Year', 'Month', 'Day', 'Hour', 'Minute', 
            'DayOfWeek', 'DayOfYear', 'Dividends', 'Stock Splits',
            'Capital Gains'  # 添加這個可能存在的列
        ]
        
        # 只選擇數值型特徵
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print_green(f"使用 {len(feature_cols)} 個特徵")
        print_cyan(f"特徵列表前10個: {feature_cols[:10]}")
        
        unique_stocks = df['stock_symbol'].unique()
        stock_to_id = {stock: i for i, stock in enumerate(unique_stocks)}
        
        successful_sequences = 0
        
        for stock in unique_stocks:
            stock_data = df[df['stock_symbol'] == stock].sort_values('Datetime').reset_index(drop=True)
            
            if len(stock_data) < sequence_length + prediction_steps:
                print_yellow(f"股票 {stock} 數據不足 ({len(stock_data)} < {sequence_length + prediction_steps})，跳過")
                continue
            
            # 提取特徵和Close價格
            try:
                features = stock_data[feature_cols].values
                close_prices = stock_data['Close'].values
                
                # 檢查數據質量
                if np.isnan(features).any() or np.isnan(close_prices).any():
                    print_red(f"股票 {stock} 包含NaN值，跳過")
                    continue
                
                if np.isinf(features).any() or np.isinf(close_prices).any():
                    print_red(f"股票 {stock} 包含無限值，跳過")
                    continue
                
                # 創建滑動窗口
                stock_sequences = 0
                for i in range(len(stock_data) - sequence_length - prediction_steps + 1):
                    # 輸入序列
                    input_sequence = features[i:i + sequence_length]
                    
                    # 目標價格
                    target_prices = close_prices[i + sequence_length:i + sequence_length + prediction_steps]
                    current_price = close_prices[i + sequence_length - 1]
                    
                    # 計算相對變化 (百分比)
                    target_changes = (target_prices - current_price) / current_price
                    
                    # 檢查目標是否有效
                    if np.isnan(target_changes).any() or np.isinf(target_changes).any():
                        continue
                    
                    all_sequences.append(input_sequence)
                    all_targets.append(target_changes)
                    
                    all_metadata.append({
                        'stock_id': stock_to_id[stock],
                        'stock_symbol': stock,
                        'datetime': stock_data['Datetime'].iloc[i + sequence_length - 1],
                        'current_price': current_price
                    })
                    
                    stock_sequences += 1
                
                successful_sequences += stock_sequences
                print_green(f"  ✓ {stock}: 創建了 {stock_sequences} 個序列")
                
            except Exception as e:
                print_red(f"  ✗ 處理股票 {stock} 時出錯: {e}")
                continue
        
        print_green(f"總共創建了 {successful_sequences} 個有效序列")
        
        if successful_sequences == 0:
            raise ValueError("沒有創建任何有效序列")
        
        return {
            'sequences': np.array(all_sequences),
            'targets': np.array(all_targets),
            'metadata': all_metadata,
            'feature_cols': feature_cols,
            'stock_to_id': stock_to_id
        }
    
    def normalize_data(self, data):
        """正規化特徵和目標，徹底避免NaN"""
        sequences = data['sequences']
        targets = data['targets']
        
        print_cyan("開始數據標準化...")
        print_cyan(f"標準化前 - 序列形狀: {sequences.shape}, 目標形狀: {targets.shape}")
        
        # 檢查輸入數據
        if np.isnan(sequences).any():
            print_red("警告: 輸入序列包含NaN值，將被替換為0")
            sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(targets).any():
            print_red("警告: 輸入目標包含NaN值，將被替換為0")
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 重塑數據進行標準化
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        
        # 標準化特徵
        print_cyan("標準化特徵...")
        try:
            sequences_normalized = self.feature_scaler.fit_transform(sequences_flat)
            sequences_normalized = sequences_normalized.reshape(original_shape)
        except Exception as e:
            print_red(f"特徵標準化失敗: {e}")
            # 手動標準化
            mean = np.nanmean(sequences_flat, axis=0)
            std = np.nanstd(sequences_flat, axis=0)
            std = np.where(std == 0, 1, std)  # 避免除零
            sequences_normalized = (sequences_flat - mean) / std
            sequences_normalized = sequences_normalized.reshape(original_shape)
        
        # 標準化目標變數
        print_cyan("標準化目標...")
        targets_flat = targets.reshape(-1, 1)
        try:
            targets_normalized = self.target_scaler.fit_transform(targets_flat)
            targets_normalized = targets_normalized.reshape(targets.shape)
        except Exception as e:
            print_red(f"目標標準化失敗: {e}")
            # 手動標準化
            mean = np.nanmean(targets_flat)
            std = np.nanstd(targets_flat)
            if std == 0:
                std = 1
            targets_normalized = (targets_flat - mean) / std
            targets_normalized = targets_normalized.reshape(targets.shape)
        
        # 最終清理
        sequences_normalized = np.nan_to_num(sequences_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        targets_normalized = np.nan_to_num(targets_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.is_fitted = True
        
        # 最終檢查
        if np.isnan(sequences_normalized).any() or np.isnan(targets_normalized).any():
            raise ValueError("標準化後仍有NaN值，這不應該發生")
        
        print_green("✓ 數據標準化完成")
        print_green(f"特徵數據形狀: {sequences_normalized.shape}")
        print_green(f"目標數據形狀: {targets_normalized.shape}")
        
        return {
            'sequences_normalized': sequences_normalized,
            'targets_normalized': targets_normalized,
            'metadata': data['metadata'],
            'feature_cols': data['feature_cols'],
            'stock_to_id': data['stock_to_id']
        }
    
    def save_processed_data(self, processed_data, output_dir='robust_normalized_data'):
        """保存處理後的數據"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存為numpy格式
        sequences_file = os.path.join(output_dir, f'sequences_{timestamp}.npy')
        targets_file = os.path.join(output_dir, f'targets_{timestamp}.npy')
        
        np.save(sequences_file, processed_data['sequences_normalized'])
        np.save(targets_file, processed_data['targets_normalized'])
        
        # 保存元數據
        metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': processed_data['metadata'],
                'feature_cols': processed_data['feature_cols'],
                'stock_to_id': processed_data['stock_to_id'],
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'timestamp': timestamp
            }, f)
        
        print_green(f"數據已保存:")
        print_green(f"  序列: {sequences_file}")
        print_green(f"  目標: {targets_file}")
        print_green(f"  元數據: {metadata_file}")
        
        return {
            'sequences_file': sequences_file,
            'targets_file': targets_file,
            'metadata_file': metadata_file
        }

def main():
    """主函數"""
    print_green("=" * 60)
    print_green("強健版數據預處理開始...")
    print_green("=" * 60)
    
    normalizer = RobustStockDataNormalizer()
    
    try:
        # 1. 載入和預處理數據
        print_green("\n1. 載入和預處理數據...")
        combined_df = normalizer.load_and_preprocess_data()
        
        # 2. 創建序列和目標
        print_green("\n2. 創建序列和目標...")
        data = normalizer.create_sequences_and_targets(combined_df)
        
        # 3. 標準化數據
        print_green("\n3. 標準化數據...")
        processed_data = normalizer.normalize_data(data)
        
        # 4. 保存數據
        print_green("\n4. 保存數據...")
        files = normalizer.save_processed_data(processed_data)
        
        # 5. 數據質量檢查
        print_green("\n5. 數據質量檢查...")
        sequences = processed_data['sequences_normalized']
        targets = processed_data['targets_normalized']
        
        print_cyan(f"最終統計:")
        print_cyan(f"  序列數量: {sequences.shape[0]}")
        print_cyan(f"  特徵維度: {sequences.shape[2]}")
        print_cyan(f"  序列長度: {sequences.shape[1]}")
        print_cyan(f"  預測步數: {targets.shape[1]}")
        
        print_cyan(f"特徵統計:")
        print_cyan(f"  平均值: {sequences.mean():.6f}")
        print_cyan(f"  標準差: {sequences.std():.6f}")
        print_cyan(f"  最小值: {sequences.min():.6f}")
        print_cyan(f"  最大值: {sequences.max():.6f}")
        
        print_cyan(f"目標統計:")
        print_cyan(f"  平均值: {targets.mean():.6f}")
        print_cyan(f"  標準差: {targets.std():.6f}")
        print_cyan(f"  最小值: {targets.min():.6f}")
        print_cyan(f"  最大值: {targets.max():.6f}")
        
        print_green("\n✓ 強健版數據預處理完成！")
        
    except Exception as e:
        print_red(f"\n✗ 處理過程中出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()