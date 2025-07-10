#!/usr/bin/env python3
"""
修復時間洩漏的強健版資料正規化腳本 - 考慮交易日曆
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
    
    def calculate_trading_day_steps(self, df, target_trading_days=5):
        """計算實際交易日對應的時間步數"""
        print_cyan(f"計算 {target_trading_days} 個交易日對應的時間步數...")
        
        # 分析數據的時間間隔
        sample_data = df.groupby('stock_symbol').first().reset_index()
        if len(sample_data) > 0:
            # 使用第一支股票分析時間模式
            first_stock = sample_data.iloc[0]['stock_symbol']
            stock_data = df[df['stock_symbol'] == first_stock].sort_values('Datetime')
            
            # 計算每天的數據點數
            stock_data['Date'] = stock_data['Datetime'].dt.date
            daily_counts = stock_data.groupby('Date').size()
            
            # 排除週末（假設週末數據很少或沒有）
            stock_data['DayOfWeek'] = stock_data['Datetime'].dt.dayofweek
            weekday_data = stock_data[stock_data['DayOfWeek'] < 5]  # 0-4 是週一到週五
            
            if len(weekday_data) > 0:
                weekday_data['Date'] = weekday_data['Datetime'].dt.date
                weekday_daily_counts = weekday_data.groupby('Date').size()
                avg_steps_per_day = weekday_daily_counts.mean()
            else:
                avg_steps_per_day = daily_counts.mean()
            
            # 計算目標交易日的步數
            total_steps = int(avg_steps_per_day * target_trading_days)
            
            print_cyan(f"  平均每個交易日步數: {avg_steps_per_day:.1f}")
            print_cyan(f"  {target_trading_days} 個交易日總步數: {total_steps}")
            
            return total_steps
        else:
            # 默認值：假設每天48步（30分鐘間隔，24小時），5個交易日
            return 48 * target_trading_days
        
    def create_time_aware_sequences(self, df, sequence_length=60, target_trading_days=5, test_trading_days=10):
        """創建時間感知的序列數據 - 考慮交易日曆"""
        print_cyan(f"創建時間感知序列，測試期間: {test_trading_days} 個交易日")
        
        # 計算實際的預測步數
        prediction_steps = self.calculate_trading_day_steps(df, target_trading_days)
        
        # 定義特徵欄位
        exclude_cols = [
            'Datetime', 'stock_symbol', 'Year', 'Month', 'Day', 'Hour', 'Minute', 
            'DayOfWeek', 'DayOfYear', 'Dividends', 'Stock Splits', 'Capital Gains', 'Date'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print_green(f"使用 {len(feature_cols)} 個特徵")
        print_green(f"序列長度: {sequence_length}")
        print_green(f"預測步數: {prediction_steps} (對應 {target_trading_days} 個交易日)")
        
        # 找出全局的時間分割點 - 基於交易日
        max_date = df['Datetime'].max()
        min_date = df['Datetime'].min()
        
        # 分析交易日分佈
        df['Date'] = df['Datetime'].dt.date
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        
        # 找出所有交易日（有數據的日期）- 只考慮工作日
        trading_dates = []
        for date in sorted(df['Date'].unique()):
            date_data = df[df['Date'] == date]
            # 檢查是否為工作日且有足夠的數據
            if date_data['DayOfWeek'].iloc[0] < 5 and len(date_data) > 10:  # 週一到週五且有足夠數據
                trading_dates.append(date)
        
        print_cyan(f"有效交易日數: {len(trading_dates)}")
        
        # 動態調整測試期間
        if len(trading_dates) < test_trading_days:
            adjusted_test_days = max(1, len(trading_dates) // 4)  # 使用1/4的數據作為測試
            print_yellow(f"調整測試期間從 {test_trading_days} 到 {adjusted_test_days} 個交易日")
            test_trading_days = adjusted_test_days
        
        # 取最後幾個交易日作為測試期
        test_start_date = trading_dates[-test_trading_days]
        test_start_datetime = pd.to_datetime(test_start_date)
        
        print_cyan(f"數據時間範圍: {min_date} 到 {max_date}")
        print_cyan(f"有效交易日數: {len(trading_dates)}")
        print_cyan(f"測試期開始日期: {test_start_date}")
        print_cyan(f"實際測試期交易日數: {test_trading_days}")
        
        # 分別處理訓練和測試數據
        train_data = df[df['Datetime'] < test_start_datetime].copy()
        test_data = df[df['Datetime'] >= test_start_datetime].copy()
        
        print_cyan(f"訓練數據量: {len(train_data)}")
        print_cyan(f"測試數據量: {len(test_data)}")
        
        if len(train_data) == 0:
            raise ValueError("訓練數據為空，請調整test_trading_days參數")
            
        # 創建股票ID映射
        unique_stocks = df['stock_symbol'].unique()
        stock_to_id = {stock: i for i, stock in enumerate(unique_stocks)}
        
        # 創建訓練序列
        train_sequences, train_targets, train_metadata = self._create_sequences_for_period(
            train_data, feature_cols, stock_to_id, sequence_length, prediction_steps, 'train'
        )
        
        # 創建測試序列 - 使用更寬鬆的條件
        test_sequences, test_targets, test_metadata = self._create_sequences_for_period(
            test_data, feature_cols, stock_to_id, sequence_length, prediction_steps, 'test', 
            allow_shorter=True
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
            'test_start_date': test_start_datetime,
            'prediction_steps': prediction_steps
        }
        
    def _create_sequences_for_period(self, data, feature_cols, stock_to_id, sequence_length, prediction_steps, period_name, allow_shorter=False):
        """為特定時期創建序列 - 改進版本"""
        sequences = []
        targets = []
        metadata = []
        
        for stock in stock_to_id.keys():
            stock_data = data[data['stock_symbol'] == stock].sort_values('Datetime').reset_index(drop=True)
            
            # 根據期間調整最小數據要求
            if period_name == 'test' and allow_shorter:
                # 測試期允許更短的序列 - 大幅降低要求
                min_required = sequence_length + max(10, prediction_steps // 10)  # 至少需要1/10的預測步數
                print_cyan(f"  {period_name} - 股票 {stock}: 最小需求 {min_required}, 實際 {len(stock_data)}")
            else:
                min_required = sequence_length + prediction_steps
            
            if len(stock_data) < min_required:
                print_yellow(f"  {period_name} - 股票 {stock} 數據不足 ({len(stock_data)} < {min_required})，跳過")
                continue
            
            try:
                features = stock_data[feature_cols].values
                close_prices = stock_data['Close'].values
                
                # 檢查數據質量
                if np.isnan(features).any() or np.isnan(close_prices).any():
                    print_red(f"  {period_name} - 股票 {stock} 包含NaN值，跳過")
                    continue
                
                stock_sequences = 0
                
                # 根據期間調整滑動窗口策略
                if period_name == 'test' and allow_shorter:
                    # 測試期：使用實際可用的數據長度
                    actual_prediction_steps = min(prediction_steps, len(stock_data) - sequence_length)
                    if actual_prediction_steps < 1:
                        print_yellow(f"  {period_name} - 股票 {stock} 預測步數不足，跳過")
                        continue
                    
                    # 只創建一個序列（最新的）
                    max_start_idx = len(stock_data) - sequence_length - actual_prediction_steps
                    start_indices = [max_start_idx] if max_start_idx >= 0 else []
                else:
                    # 訓練期：使用完整的滑動窗口
                    actual_prediction_steps = prediction_steps
                    start_indices = range(len(stock_data) - sequence_length - actual_prediction_steps + 1)
                
                # 創建序列
                for i in start_indices:
                    if i < 0:
                        continue
                        
                    input_sequence = features[i:i + sequence_length]
                    target_prices = close_prices[i + sequence_length:i + sequence_length + actual_prediction_steps]
                    current_price = close_prices[i + sequence_length - 1]
                    
                    # 檢查目標價格是否有效
                    if len(target_prices) == 0:
                        continue
                    
                    # 如果目標長度不足，進行填充
                    if len(target_prices) < prediction_steps:
                        # 使用最後一個價格進行填充
                        padding_length = prediction_steps - len(target_prices)
                        last_price = target_prices[-1] if len(target_prices) > 0 else current_price
                        target_prices = np.concatenate([target_prices, np.full(padding_length, last_price)])
                    
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
                        'period': period_name,
                        'actual_prediction_steps': actual_prediction_steps
                    })
                    stock_sequences += 1
                
                if stock_sequences > 0:
                    print_green(f"  {period_name} - {stock}: 創建了 {stock_sequences} 個序列")
                
            except Exception as e:
                print_red(f"  {period_name} - 處理股票 {stock} 時出錯: {e}")
                continue
        
        return sequences, targets, metadata
    
    def save_processed_data(self, processed_data, output_dir='normalized_data'):
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
                'prediction_steps': processed_data['prediction_steps'],
                'timestamp': timestamp
            }, f)
        
        print_green(f"數據已保存到 {output_dir}")
        print_cyan(f"預測步數: {processed_data['prediction_steps']}")
        
        return {
            'train_sequences_file': os.path.join(output_dir, f'train_sequences_{timestamp}.npy'),
            'train_targets_file': os.path.join(output_dir, f'train_targets_{timestamp}.npy'),
            'test_sequences_file': os.path.join(output_dir, f'test_sequences_{timestamp}.npy'),
            'test_targets_file': os.path.join(output_dir, f'test_targets_{timestamp}.npy'),
            'metadata_file': metadata_file
        }
    def save_processed_chunk_data(self, processed_data, output_dir='normalized_data_temp', chunk_index=0):
        """保存處理後的數據"""
        os.makedirs(output_dir, exist_ok=True)
    
        
        # 保存訓練數據
        np.save(os.path.join(output_dir, f'train_sequences_{chunk_index}.npy'), 
                processed_data['train_sequences_normalized'])
        np.save(os.path.join(output_dir, f'train_targets_{chunk_index}.npy'), 
                processed_data['train_targets_normalized'])
        
        # 保存測試數據
        if len(processed_data['test_sequences_normalized']) > 0:
            np.save(os.path.join(output_dir, f'test_sequences_{chunk_index}.npy'), 
                    processed_data['test_sequences_normalized'])
            np.save(os.path.join(output_dir, f'test_targets_{chunk_index}.npy'), 
                    processed_data['test_targets_normalized'])
        
        # 保存元數據
        # metadata_file = os.path.join(output_dir, f'metadata_{chunk_index}.pkl')
        # with open(metadata_file, 'wb') as f:
        #     pickle.dump({
        #         'train_metadata': processed_data['train_metadata'],
        #         'test_metadata': processed_data['test_metadata'],
        #         'feature_cols': processed_data['feature_cols'],
        #         'stock_to_id': processed_data['stock_to_id'],
        #         'feature_scaler': self.feature_scaler,
        #         'target_scaler': self.target_scaler,
        #         'test_start_date': processed_data['test_start_date'],
        #         'prediction_steps': processed_data['prediction_steps'],
        #         'timestamp': timestamp
        #     }, f)
        
        print_green(f"數據已保存到 {output_dir}")
        print_cyan(f"預測步數: {processed_data['prediction_steps']}")
        
        return {
            'train_sequences_file': os.path.join(output_dir, f'train_sequences_{chunk_index}.npy'),
            'train_targets_file': os.path.join(output_dir, f'train_targets_{chunk_index}.npy'),
            'test_sequences_file': os.path.join(output_dir, f'test_sequences_{chunk_index}.npy'),
            'test_targets_file': os.path.join(output_dir, f'test_targets_{chunk_index}.npy'),
            # 'metadata_file': metadata_file
        }
        
    def process_in_chunks(self, data_dir='data/feature', chunk_size=5):
        """分塊處理數據"""
        print_green("開始分塊處理超大數據集...")
        
        # 只處理部分股票
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print_cyan(f"總共 {len(csv_files)} 個文件，分塊處理")
        
        # 創建輸出目錄
        output_dir = 'normalized_data'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 分塊處理
        for chunk_start in range(0, len(csv_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(csv_files))
            chunk_files = csv_files[chunk_start:chunk_end]
            
            print_cyan(f"處理塊 {chunk_start//chunk_size + 1}: {chunk_files}")
            
            # 處理當前塊
            chunk_data = self._process_chunk(chunk_files, data_dir , chunk_start//chunk_size + 1)
            
            if chunk_data is not None:
                # 保存當前塊
                chunk_file = os.path.join(output_dir, f'chunk_{chunk_start//chunk_size + 1}_{timestamp}.pkl')
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_data, f)
                
                print_green(f"塊 {chunk_start//chunk_size + 1} 處理完成")
                
                # 清理內存
                del chunk_data
                gc.collect()
        

        # merge all processed chunks
        print_cyan("合併所有處理過的塊...")
        process_data = dict()
        process_data['train_metadata'] = []
        process_data['test_metadata'] = []
        process_data['stock_to_id'] = []
        process_data['feature_cols'] = []
        process_data['feature_scaler'] = self.feature_scaler
        process_data['target_scaler'] = self.target_scaler
        process_data['test_start_date'] = None
        process_data['prediction_steps'] = None
        process_data['timestamp'] = timestamp
        for chunk_file in os.listdir(output_dir+'_temp'):
            if chunk_file.endswith('.npy'):
                output_file_name = ''
                if 'train_sequences' in chunk_file:
                    output_file_name = f'train_sequences_file_{timestamp}.npy'
                elif 'train_targets' in chunk_file:
                    output_file_name = f'train_targets_file_{timestamp}.npy'
                elif 'test_sequences' in chunk_file:
                    output_file_name = f'test_sequences_file_{timestamp}.npy'
                elif 'test_targets' in chunk_file:
                    output_file_name = f'test_targets_file_{timestamp}.npy'
                    
                with open(os.path.join(output_dir, chunk_file), 'rb') as f:
                    chunk_data = pickle.load(f)
                    process_data['train_metadata'].append(chunk_data)
                    process_data['test_metadata'].append(chunk_data)
                    process_data['stock_to_id'].append(chunk_data)
                    if process_data['feature_cols']:
                        process_data['feature_cols'] = chunk_data['feature_cols']
                with open(os.path.join(output_dir, output_file_name), 'ab') as f:
                    pickle.dump(chunk_data, f)
        
        # 保存元數據
        print_green("\\4. 保存元數據...")
        metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'train_metadata': process_data['train_metadata'],
                'test_metadata': process_data['test_metadata'],
                'feature_cols': process_data['feature_cols'],
                'stock_to_id': process_data['stock_to_id'],
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'test_start_date': process_data['test_start_date'],
                'prediction_steps': process_data['prediction_steps'],
                'timestamp': timestamp
            }, f)

        print_green("所有塊處理完成！")

    def _process_chunk(self, files, data_dir , chunk_id):
        """處理單個數據塊"""
        all_data = []
        
        for csv_file in files:
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
                
                all_data.append(df)
                
            except Exception as e:
                print_red(f"處理 {csv_file} 時出錯: {e}")
                continue
        
        if not all_data:
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['stock_symbol', 'Datetime']).reset_index(drop=True)
        
        # 最終清理
        combined_df = self.clean_data(combined_df)
        
        print_green(f"合併後數據形狀: {combined_df.shape}")
        
        # 創建時間感知序列
        data = self.create_time_aware_sequences(
            combined_df, sequence_length=60, target_trading_days=5, test_trading_days=10
        )
        
        try:
            processed_data = self.normalize_data(data)
            print_green("標準化成功完成！")
        except Exception as e:
            print_red(f"標準化失敗: {e}")
            raise


        print_green("\n4. 保存數據...")
        files = self.save_processed_chunk_data(processed_data , chunk_index=chunk_id)

    def normalize_data(self, data):
        """使用時間感知的標準化 - 改進版本"""
        print_cyan("開始時間感知標準化...")
        
        train_sequences = data['train_sequences']
        train_targets = data['train_targets']
        test_sequences = data['test_sequences']
        test_targets = data['test_targets']
        
        print_cyan(f"訓練序列形狀: {train_sequences.shape}")
        print_cyan(f"測試序列形狀: {test_sequences.shape}")
        
        # 檢查內存使用情況
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print_cyan(f"當前內存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
        
        # 計算需要的內存
        train_flat_size = train_sequences.shape[0] * train_sequences.shape[1] * train_sequences.shape[2]
        estimated_memory_gb = train_flat_size * 8 / (1024**3)  # 假設float64
        print_cyan(f"預估標準化需要內存: {estimated_memory_gb:.1f} GB")
        
        if estimated_memory_gb > 4:  # 如果超過4GB，使用批量處理
            print_yellow("數據量過大，使用批量標準化...")
            return self._normalize_data_batch(data)
        
        try:
            # 只使用訓練數據來擬合標準化器
            print_cyan("重塑訓練序列數據...")
            train_sequences_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
            train_targets_flat = train_targets.reshape(-1, 1)
            
            print_cyan(f"重塑後訓練序列形狀: {train_sequences_flat.shape}")
            print_cyan(f"重塑後訓練目標形狀: {train_targets_flat.shape}")
            
            # 標準化特徵
            print_cyan("使用訓練數據擬合特徵標準化器...")
            sequences_normalized_train = self.feature_scaler.fit_transform(train_sequences_flat)
            print_cyan("特徵標準化完成，重塑回原始形狀...")
            sequences_normalized_train = sequences_normalized_train.reshape(train_sequences.shape)
            
            # 標準化目標
            print_cyan("使用訓練數據擬合目標標準化器...")
            targets_normalized_train = self.target_scaler.fit_transform(train_targets_flat)
            print_cyan("目標標準化完成，重塑回原始形狀...")
            targets_normalized_train = targets_normalized_train.reshape(train_targets.shape)
            
            # 清理中間變量
            del train_sequences_flat, train_targets_flat
            import gc
            gc.collect()
            
            # 應用標準化到測試數據
            if len(test_sequences) > 0:
                print_cyan("標準化測試數據...")
                test_sequences_flat = test_sequences.reshape(-1, test_sequences.shape[-1])
                test_targets_flat = test_targets.reshape(-1, 1)
                
                sequences_normalized_test = self.feature_scaler.transform(test_sequences_flat)
                sequences_normalized_test = sequences_normalized_test.reshape(test_sequences.shape)
                
                targets_normalized_test = self.target_scaler.transform(test_targets_flat)
                targets_normalized_test = targets_normalized_test.reshape(test_targets.shape)
                
                # 清理中間變量
                del test_sequences_flat, test_targets_flat
                gc.collect()
            else:
                sequences_normalized_test = np.array([])
                targets_normalized_test = np.array([])
            
            # 清理NaN值
            print_cyan("清理NaN值...")
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
                'test_start_date': data['test_start_date'],
                'prediction_steps': data['prediction_steps']
            }
            
        except MemoryError as e:
            print_red(f"內存不足: {e}")
            print_yellow("切換到批量標準化...")
            return self._normalize_data_batch(data)
        except Exception as e:
            print_red(f"標準化過程中出錯: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _normalize_data_batch(self, data):
        """批量標準化數據 - 終極保守版本"""
        print_cyan("開始批量標準化...")
        
        train_sequences = data['train_sequences']
        train_targets = data['train_targets']
        test_sequences = data['test_sequences']
        test_targets = data['test_targets']
        
        batch_size = 100  # 進一步減少批量大小
        
        # 第一步：使用極少量數據擬合標準化器
        print_cyan("使用極少量數據擬合標準化器...")
        sample_size = min(500, len(train_sequences))  # 只使用500個樣本
        sample_indices = np.random.choice(len(train_sequences), sample_size, replace=False)
        
        # 分批次擬合標準化器
        print_cyan("分批次擬合標準化器...")
        feature_samples = []
        target_samples = []
        
        for i in range(0, sample_size, 50):  # 每次只處理50個樣本
            end_idx = min(i + 50, sample_size)
            batch_indices = sample_indices[i:end_idx]
            
            batch_sequences = train_sequences[batch_indices]
            batch_targets = train_targets[batch_indices]
            
            # 重塑並收集
            batch_seq_flat = batch_sequences.reshape(-1, batch_sequences.shape[-1])
            batch_tar_flat = batch_targets.reshape(-1, 1)
            
            feature_samples.append(batch_seq_flat)
            target_samples.append(batch_tar_flat)
            
            # 清理
            del batch_sequences, batch_targets, batch_seq_flat, batch_tar_flat
            import gc
            gc.collect()
        
        # 合併樣本並擬合
        print_cyan("合併樣本並擬合標準化器...")
        all_feature_samples = np.vstack(feature_samples)
        all_target_samples = np.vstack(target_samples)
        
        self.feature_scaler.fit(all_feature_samples)
        self.target_scaler.fit(all_target_samples)
        
        # 清理
        del feature_samples, target_samples, all_feature_samples, all_target_samples
        gc.collect()
        
        print_green("✓ 標準化器擬合完成")
        
        # 第二步：直接保存到文件而不是內存
        print_cyan("直接保存標準化結果到文件...")
        
        # 創建輸出目錄
        output_dir = 'normalized_data'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 直接保存訓練數據
        train_file = os.path.join(output_dir, f'train_sequences_{timestamp}.npy')
        train_target_file = os.path.join(output_dir, f'train_targets_{timestamp}.npy')
        
        print_cyan("直接批量處理並保存訓練數據...")
        
        # 創建空的npy文件
        np.save(train_file, np.array([]))
        np.save(train_target_file, np.array([]))
        
        # 用append模式寫入
        train_results = []
        train_target_results = []
        
        total_batches = (len(train_sequences) + batch_size - 1) // batch_size
        
        for i in range(0, len(train_sequences), batch_size):
            end_idx = min(i + batch_size, len(train_sequences))
            batch_num = i // batch_size + 1
            
            if batch_num % 50 == 0:  # 每50批輸出一次
                print_cyan(f"訓練數據批次 {batch_num}/{total_batches}")
            
            try:
                batch_seq = train_sequences[i:end_idx]
                batch_tar = train_targets[i:end_idx]
                
                batch_seq_flat = batch_seq.reshape(-1, batch_seq.shape[-1])
                batch_tar_flat = batch_tar.reshape(-1, 1)
                
                batch_seq_norm = self.feature_scaler.transform(batch_seq_flat)
                batch_tar_norm = self.target_scaler.transform(batch_tar_flat)
                
                # 重塑並清理NaN
                batch_seq_reshaped = batch_seq_norm.reshape(batch_seq.shape)
                batch_tar_reshaped = batch_tar_norm.reshape(batch_tar.shape)
                
                # 逐步清理NaN
                batch_seq_clean = np.nan_to_num(batch_seq_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                batch_tar_clean = np.nan_to_num(batch_tar_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                
                train_results.append(batch_seq_clean)
                train_target_results.append(batch_tar_clean)
                
                # 清理
                del batch_seq, batch_tar, batch_seq_flat, batch_tar_flat
                del batch_seq_norm, batch_tar_norm, batch_seq_reshaped, batch_tar_reshaped
                del batch_seq_clean, batch_tar_clean
                gc.collect()
                
                # 每累積一定數量就保存一次
                if len(train_results) >= 10 or batch_num == total_batches:
                    print_cyan(f"保存累積的批次數據...")
                    
                    # 合併並保存
                    if len(train_results) > 0:
                        combined_seq = np.vstack(train_results)
                        combined_tar = np.vstack(train_target_results)
                        
                        # 如果是第一次保存，直接保存；否則追加
                        if i < batch_size * 10:  # 第一次
                            np.save(train_file, combined_seq)
                            np.save(train_target_file, combined_tar)
                        else:  # 追加
                            existing_seq = np.load(train_file)
                            existing_tar = np.load(train_target_file)
                            
                            new_seq = np.vstack([existing_seq, combined_seq])
                            new_tar = np.vstack([existing_tar, combined_tar])
                            
                            np.save(train_file, new_seq)
                            np.save(train_target_file, new_tar)
                            
                            del existing_seq, existing_tar, new_seq, new_tar
                        
                        del combined_seq, combined_tar
                        train_results = []
                        train_target_results = []
                        gc.collect()
                    
            except Exception as e:
                print_red(f"批次 {batch_num} 處理出錯: {e}")
                continue
        
        # 處理測試數據
        if len(test_sequences) > 0:
            print_cyan("直接批量處理並保存測試數據...")
            
            test_file = os.path.join(output_dir, f'test_sequences_{timestamp}.npy')
            test_target_file = os.path.join(output_dir, f'test_targets_{timestamp}.npy')
            
            test_results = []
            test_target_results = []
            
            for i in range(0, len(test_sequences), batch_size):
                end_idx = min(i + batch_size, len(test_sequences))
                
                try:
                    batch_seq = test_sequences[i:end_idx]
                    batch_tar = test_targets[i:end_idx]
                    
                    batch_seq_flat = batch_seq.reshape(-1, batch_seq.shape[-1])
                    batch_tar_flat = batch_tar.reshape(-1, 1)
                    
                    batch_seq_norm = self.feature_scaler.transform(batch_seq_flat)
                    batch_tar_norm = self.target_scaler.transform(batch_tar_flat)
                    
                    batch_seq_reshaped = batch_seq_norm.reshape(batch_seq.shape)
                    batch_tar_reshaped = batch_tar_norm.reshape(batch_tar.shape)
                    
                    batch_seq_clean = np.nan_to_num(batch_seq_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                    batch_tar_clean = np.nan_to_num(batch_tar_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    test_results.append(batch_seq_clean)
                    test_target_results.append(batch_tar_clean)
                    
                    del batch_seq, batch_tar, batch_seq_flat, batch_tar_flat
                    del batch_seq_norm, batch_tar_norm, batch_seq_reshaped, batch_tar_reshaped
                    del batch_seq_clean, batch_tar_clean
                    gc.collect()
                    
                except Exception as e:
                    print_red(f"測試批次處理出錯: {e}")
                    continue
            
            # 保存測試數據
            if len(test_results) > 0:
                combined_test_seq = np.vstack(test_results)
                combined_test_tar = np.vstack(test_target_results)
                
                np.save(test_file, combined_test_seq)
                np.save(test_target_file, combined_test_tar)
                
                del combined_test_seq, combined_test_tar
                test_results = []
                test_target_results = []
                gc.collect()
        
        # 載入最終結果
        print_cyan("載入最終結果...")
        train_seq_final = np.load(train_file)
        train_tar_final = np.load(train_target_file)
        
        if len(test_sequences) > 0:
            test_seq_final = np.load(test_file)
            test_tar_final = np.load(test_target_file)
        else:
            test_seq_final = np.array([])
            test_tar_final = np.array([])
        
        self.is_fitted = True
        
        print_green("✓ 極保守批量標準化完成")
        
        return {
            'train_sequences_normalized': train_seq_final,
            'train_targets_normalized': train_tar_final,
            'test_sequences_normalized': test_seq_final,
            'test_targets_normalized': test_tar_final,
            'train_metadata': data['train_metadata'],
            'test_metadata': data['test_metadata'],
            'feature_cols': data['feature_cols'],
            'stock_to_id': data['stock_to_id'],
            'test_start_date': data['test_start_date'],
            'prediction_steps': data['prediction_steps']
        }
def main():
    """主函數"""
    print_green("=" * 60)
    print_green("時間感知數據預處理開始（考慮交易日曆）...")
    print_green("=" * 60)
    
    normalizer = TimeAwareStockDataNormalizer()
    
    try:
        # 1. 載入和預處理數據
        print_green("\n1. 載入和預處理數據...")
        combined_df = normalizer.load_and_preprocess_data()
        
        # 2. 創建時間感知序列
        print_green("\n2. 創建時間感知序列...")
        data = normalizer.create_time_aware_sequences(
            combined_df, 
            sequence_length=60,
            target_trading_days=5,      # 預測5個交易日
            test_trading_days=10        # 增加測試期到10個交易日
        )
        
        # 內存檢查
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print_cyan(f"序列創建後內存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
        
        # 3. 標準化數據
        print_green("\n3. 時間感知標準化...")
        try:
            processed_data = normalizer.normalize_data(data)
            print_green("標準化成功完成！")
        except Exception as e:
            print_red(f"標準化失敗: {e}")
            raise
        
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
        print_cyan(f"  預測步數: {processed_data['prediction_steps']}")
        print_cyan(f"  測試開始日期: {processed_data['test_start_date']}")
        
        # 最終內存檢查
        memory_info = process.memory_info()
        print_cyan(f"完成後內存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
        
        print_green("\n✓ 時間感知數據預處理完成！")
        print_green("✓ 已避免look-ahead bias")
        print_green("✓ 已考慮交易日曆限制")
        
    except KeyboardInterrupt:
        print_red("\n✗ 用戶中斷程序")
    except MemoryError as e:
        print_red(f"\n✗ 內存不足: {e}")
        print_yellow("建議：")
        print_yellow("1. 減少序列長度 (sequence_length)")
        print_yellow("2. 減少預測步數 (target_trading_days)")
        print_yellow("3. 增加系統內存")
    except Exception as e:
        print_red(f"\n✗ 處理過程中出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()