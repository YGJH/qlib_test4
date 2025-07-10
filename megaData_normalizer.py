#!/usr/bin/env python3
"""
超大數據集處理版本 - 最小內存使用
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

warnings.filterwarnings('ignore')

class MegaDataNormalizer:
    """超大數據集正規化器"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.metadata = {}
        self.init_metadata()
        
    def init_metadata(self):
        """初始化元數據"""
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
        """清空輸出目錄"""
        if os.path.exists(output_dir):
            print_yellow(f"清空舊的輸出目錄: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print_green(f"✓ 輸出目錄已準備: {output_dir}")
        
    def save_metadata(self, output_dir='normalized_data', timestamp=None):
        """保存處理後的元數據"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 更新元數據
        self.metadata['feature_scaler'] = self.feature_scaler
        self.metadata['target_scaler'] = self.target_scaler
        self.metadata['timestamp'] = timestamp
        
        metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.pkl')
        
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            print_green(f"✓ 元數據已保存: {metadata_file}")
        except Exception as e:
            print_red(f"✗ 保存元數據失敗: {e}")

    def process_in_chunks(self, data_dir='data/feature', chunk_size=5):
        """分塊處理數據並累積結果"""
        print_green("開始分塊處理超大數據集...")
        
        # 創建輸出目錄並清空
        output_dir = 'mega_normalized_data'
        self.clear_output_dir(output_dir)
        
        # 獲取所有CSV文件
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print_cyan(f"總共 {len(csv_files)} 個文件，分塊處理")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata['timestamp'] = timestamp
        
        # 第一步：預先計算全局預測步數
        print_green("第一步：分析數據以確定統一的預測步數...")
        global_prediction_steps = self._calculate_global_prediction_steps(csv_files, data_dir)
        print_green(f"全局預測步數確定為: {global_prediction_steps}")
        
        # 累積所有處理結果
        all_train_sequences = []
        all_train_targets = []
        all_test_sequences = []
        all_test_targets = []
        all_train_metadata = []
        all_test_metadata = []
        combined_stock_to_id = {}
        
        # 分塊處理
        for chunk_start in range(0, len(csv_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(csv_files))
            chunk_files = csv_files[chunk_start:chunk_end]
            
            print_cyan(f"處理塊 {chunk_start//chunk_size + 1}: {chunk_files}")
            
            # 處理當前塊 - 傳入全局預測步數
            chunk_data = self._process_chunk(chunk_files, data_dir, timestamp, global_prediction_steps)
            
            if chunk_data is not None:
                # 驗證數據維度一致性
                current_train_shape = chunk_data['train_sequences_normalized'].shape
                current_target_shape = chunk_data['train_targets_normalized'].shape
                
                if all_train_sequences:
                    expected_train_shape = all_train_sequences[0].shape
                    expected_target_shape = all_train_targets[0].shape
                    
                    if (current_train_shape[1:] != expected_train_shape[1:] or 
                        current_target_shape[1:] != expected_target_shape[1:]):
                        print_red(f"維度不匹配！跳過當前塊")
                        print_red(f"  期望訓練形狀: {expected_train_shape}, 實際: {current_train_shape}")
                        print_red(f"  期望目標形狀: {expected_target_shape}, 實際: {current_target_shape}")
                        continue
                
                # 累積數據
                all_train_sequences.append(chunk_data['train_sequences_normalized'])
                all_train_targets.append(chunk_data['train_targets_normalized'])
                all_train_metadata.extend(chunk_data['train_metadata'])
                
                # 累積測試數據
                if len(chunk_data['test_sequences_normalized']) > 0:
                    all_test_sequences.append(chunk_data['test_sequences_normalized'])
                    all_test_targets.append(chunk_data['test_targets_normalized'])
                    all_test_metadata.extend(chunk_data['test_metadata'])
                
                # 更新 stock_to_id 映射
                combined_stock_to_id.update(chunk_data['stock_to_id'])
                
                # 保存當前塊的詳細信息（用於調試）
                chunk_file = os.path.join(output_dir, f'chunk_{chunk_start//chunk_size + 1}_{timestamp}.pkl')
                try:
                    chunk_summary = {
                        'files': chunk_files,
                        'train_shape': chunk_data['train_sequences_normalized'].shape,
                        'test_shape': chunk_data['test_sequences_normalized'].shape,
                        'stocks': list(chunk_data['stock_to_id'].keys()),
                        'prediction_steps': chunk_data['prediction_steps']
                    }
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_summary, f)
                    print_green(f"✓ 塊 {chunk_start//chunk_size + 1} 處理完成")
                except Exception as e:
                    print_red(f"✗ 保存塊摘要失敗: {e}")
                
                # 清理內存
                del chunk_data
                gc.collect()
        
        # 合併所有數據
        print_green("合併所有塊的數據...")
        if all_train_sequences:
            print_cyan(f"準備合併 {len(all_train_sequences)} 個訓練塊")
            
            # 打印每個塊的維度信息
            for i, (seq, tar) in enumerate(zip(all_train_sequences, all_train_targets)):
                print_cyan(f"  塊 {i+1}: 序列 {seq.shape}, 目標 {tar.shape}")
            
            final_train_sequences = np.concatenate(all_train_sequences, axis=0)
            final_train_targets = np.concatenate(all_train_targets, axis=0)
            
            if all_test_sequences:
                print_cyan(f"準備合併 {len(all_test_sequences)} 個測試塊")
                for i, (seq, tar) in enumerate(zip(all_test_sequences, all_test_targets)):
                    print_cyan(f"  測試塊 {i+1}: 序列 {seq.shape}, 目標 {tar.shape}")
                
                final_test_sequences = np.concatenate(all_test_sequences, axis=0)
                final_test_targets = np.concatenate(all_test_targets, axis=0)
            else:
                final_test_sequences = np.array([])
                final_test_targets = np.array([])
            
            # 創建最終的合併數據
            combined_data = {
                'train_sequences_normalized': final_train_sequences,
                'train_targets_normalized': final_train_targets,
                'test_sequences_normalized': final_test_sequences,
                'test_targets_normalized': final_test_targets,
                'train_metadata': all_train_metadata,
                'test_metadata': all_test_metadata,
                'stock_to_id': combined_stock_to_id,
                'feature_cols': self.metadata['feature_cols'],
                'test_start_date': self.metadata['test_start_date'],
                'prediction_steps': global_prediction_steps
            }
            
            # 保存最終合併的數據
            print_green("保存最終合併的數據...")
            self.save_processed_data(combined_data, output_dir, timestamp)
            for chunk_file in os.listdir(output_dir):
                if chunk_file.startswith('chunk_'):
                    chunk_path = os.path.join(output_dir, chunk_file)
                    os.remove(chunk_path)


            print_green(f"✓ 最終數據統計:")
            print_green(f"  總訓練序列: {final_train_sequences.shape}")
            print_green(f"  總測試序列: {final_test_sequences.shape}")
            print_green(f"  總股票數: {len(combined_stock_to_id)}")
            print_green(f"  特徵維度: {final_train_sequences.shape[2] if len(final_train_sequences.shape) > 2 else 0}")
            print_green(f"  預測步數: {global_prediction_steps}")
            
            # 清理中間數據
            del all_train_sequences, all_train_targets, all_test_sequences, all_test_targets
            gc.collect()
        
        # 保存最終元數據
        self.save_metadata(output_dir, timestamp)
        print_green("所有塊處理完成！")
        
    def save_processed_data(self, processed_data, output_dir='normalized_data', timestamp=None):
        """保存處理後的數據"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 保存訓練數據
            train_seq_file = os.path.join(output_dir, f'train_sequences_{timestamp}.npy')
            train_tar_file = os.path.join(output_dir, f'train_targets_{timestamp}.npy')
            
            np.save(train_seq_file, processed_data['train_sequences_normalized'])
            np.save(train_tar_file, processed_data['train_targets_normalized'])
            
            # 保存測試數據
            test_seq_file = None
            test_tar_file = None
            
            if len(processed_data['test_sequences_normalized']) > 0:
                test_seq_file = os.path.join(output_dir, f'test_sequences_{timestamp}.npy')
                test_tar_file = os.path.join(output_dir, f'test_targets_{timestamp}.npy')
                
                np.save(test_seq_file, processed_data['test_sequences_normalized'])
                np.save(test_tar_file, processed_data['test_targets_normalized'])
            
            # 更新元數據
            self.metadata['timestamp'] = timestamp
            self.metadata['train_metadata'].extend(processed_data['train_metadata'])
            self.metadata['test_metadata'].extend(processed_data['test_metadata'])
            self.metadata['feature_cols'] = processed_data['feature_cols']
            
            # 合併 stock_to_id 字典
            if isinstance(processed_data['stock_to_id'], dict):
                self.metadata['stock_to_id'].update(processed_data['stock_to_id'])
            
            # 修復 test_start_date 問題
            if isinstance(processed_data['test_start_date'], (pd.Timestamp, datetime)):
                self.metadata['test_start_date'] = processed_data['test_start_date']
            
            self.metadata['prediction_steps'] = processed_data['prediction_steps']
            self.metadata['feature_scaler'] = self.feature_scaler
            self.metadata['target_scaler'] = self.target_scaler
            
            print_green(f"✓ 數據已保存到 {output_dir}")
            print_cyan(f"  訓練序列: {train_seq_file}")
            print_cyan(f"  訓練目標: {train_tar_file}")
            if test_seq_file:
                print_cyan(f"  測試序列: {test_seq_file}")
                print_cyan(f"  測試目標: {test_tar_file}")
            print_cyan(f"  預測步數: {processed_data['prediction_steps']}")
            
            return {
                'train_sequences_file': train_seq_file,
                'train_targets_file': train_tar_file,
                'test_sequences_file': test_seq_file,
                'test_targets_file': test_tar_file,
                'timestamp': timestamp
            }
            
        except Exception as e:
            print_red(f"✗ 保存數據失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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

    def safe_division(self, numerator, denominator, fill_value=0.0):
        """安全除法，避免除零和產生NaN"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, fill_value)
            return result

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
 
    def _process_chunk(self, files, data_dir, timestamp, global_prediction_steps):
        """處理單個數據塊 - 返回處理後的數據而不直接保存"""
        try:
            # 載入數據
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
                    
                    print_green(f"  ✓ 成功處理 {csv_file}, 形狀: {df.shape}")
                    all_data.append(df)
                    
                except Exception as e:
                    print_red(f"  ✗ 處理 {csv_file} 時出錯: {e}")
                    continue

            if not all_data:
                print_red("沒有成功處理任何文件")
                return None
                
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['stock_symbol', 'Datetime']).reset_index(drop=True)
            
            # 最終清理
            combined_df = self.clean_data(combined_df)
            
            print_green(f"合併後數據形狀: {combined_df.shape}")
            
            print_green("\n2. 創建時間感知序列...")
            data = self.create_time_aware_sequences(
                combined_df, 
                sequence_length=60,
                target_trading_days=5,      # 預測5個交易日
                test_trading_days=10,       # 增加測試期到10個交易日
                fixed_prediction_steps=global_prediction_steps  # 使用全局預測步數
            )
                
            # 內存檢查
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            print_cyan(f"序列創建後內存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
            
            # 3. 標準化數據
            print_green("\n3. 時間感知標準化...")
            try:
                processed_data = self.normalize_data(data)
                print_green("標準化成功完成！")
            except Exception as e:
                print_red(f"標準化失敗: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 4. 數據質量檢查
            print_green("\n4. 數據質量檢查...")
            train_seq = processed_data['train_sequences_normalized']
            test_seq = processed_data['test_sequences_normalized']
            
            print_cyan(f"當前塊統計:")
            print_cyan(f"  訓練序列: {train_seq.shape}")
            print_cyan(f"  測試序列: {test_seq.shape}")
            print_cyan(f"  特徵維度: {train_seq.shape[2] if len(train_seq.shape) > 2 else 0}")
            print_cyan(f"  預測步數: {processed_data['prediction_steps']}")
            print_cyan(f"  測試開始日期: {processed_data['test_start_date']}")
            
            # 更新全局元數據
            self.metadata['feature_cols'] = processed_data['feature_cols']
            self.metadata['test_start_date'] = processed_data['test_start_date']
            self.metadata['prediction_steps'] = processed_data['prediction_steps']
            
            # 返回處理後的數據
            return processed_data
            
        except Exception as e:
            print_red(f"處理塊時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ...existing code... (保持 create_time_aware_sequences, _create_sequences_for_period, normalize_data, _normalize_data_batch 等方法不變)
    
    def create_time_aware_sequences(self, df, sequence_length=60, target_trading_days=5, test_trading_days=10, fixed_prediction_steps=None):
        """創建時間感知的序列數據 - 考慮交易日曆"""
        print_cyan(f"創建時間感知序列，測試期間: {test_trading_days} 個交易日")
        
        # 使用固定的預測步數（如果提供）或計算新的
        if fixed_prediction_steps is not None:
            prediction_steps = fixed_prediction_steps
            print_cyan(f"使用固定預測步數: {prediction_steps}")
        else:
            prediction_steps = self.calculate_trading_day_steps(df, target_trading_days)
            print_cyan(f"計算得到預測步數: {prediction_steps}")
        
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
        
        # 創建訓練序列 - 傳入固定預測步數
        train_sequences, train_targets, train_metadata = self._create_sequences_for_period(
            train_data, feature_cols, stock_to_id, sequence_length, prediction_steps, 'train'
        )
        
        # 創建測試序列 - 使用更寬鬆的條件，傳入固定預測步數
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
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            print_cyan(f"當前內存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
        except:
            print_cyan("無法獲取內存信息")
        
        # 計算需要的內存
        train_flat_size = train_sequences.shape[0] * train_sequences.shape[1] * train_sequences.shape[2]
        estimated_memory_gb = train_flat_size * 8 / (1024**3)  # 假設float64
        print_cyan(f"預估標準化需要內存: {estimated_memory_gb:.1f} GB")
        
        if estimated_memory_gb > 2:  # 降低閾值到2GB
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


    def _batch_normalize_sequences(self, sequences, batch_size):
        """批量標準化序列數據"""
        normalized_sequences = np.zeros_like(sequences)
        
        for i in range(0, len(sequences), batch_size):
            end_idx = min(i + batch_size, len(sequences))
            batch = sequences[i:end_idx]
            
            batch_flat = batch.reshape(-1, batch.shape[-1])
            batch_norm = self.feature_scaler.transform(batch_flat)
            batch_norm = batch_norm.reshape(batch.shape)
            batch_norm = np.nan_to_num(batch_norm, nan=0.0)
            
            normalized_sequences[i:end_idx] = batch_norm
            
            del batch, batch_flat, batch_norm
            gc.collect()
        
        return normalized_sequences

    def _batch_normalize_targets(self, targets, batch_size):
        """批量標準化目標數據"""
        normalized_targets = np.zeros_like(targets)
        
        for i in range(0, len(targets), batch_size):
            end_idx = min(i + batch_size, len(targets))
            batch = targets[i:end_idx]
            
            batch_flat = batch.reshape(-1, 1)
            batch_norm = self.target_scaler.transform(batch_flat)
            batch_norm = batch_norm.reshape(batch.shape)
            batch_norm = np.nan_to_num(batch_norm, nan=0.0)
            
            normalized_targets[i:end_idx] = batch_norm
            
            del batch, batch_flat, batch_norm
            gc.collect()
        
        return normalized_targets

    def _calculate_global_prediction_steps(self, csv_files, data_dir, target_trading_days=5):
        """預先計算全局統一的預測步數"""
        print_cyan("分析數據以確定統一的預測步數...")
        
        # 取前幾個文件作為樣本
        sample_files = csv_files[:min(5, len(csv_files))]
        all_sample_data = []
        
        for csv_file in sample_files:
            try:
                file_path = os.path.join(data_dir, csv_file)
                stock_symbol = os.path.splitext(csv_file)[0]
                
                df = pd.read_csv(file_path)
                df['stock_symbol'] = stock_symbol
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                all_sample_data.append(df)
                print_cyan(f"  樣本文件: {csv_file}")
                
            except Exception as e:
                print_yellow(f"  跳過樣本文件 {csv_file}: {e}")
                continue
        
        if not all_sample_data:
            print_yellow("無法載入樣本數據，使用默認預測步數")
            return 48 * target_trading_days  # 默認值
        
        # 合併樣本數據
        combined_sample = pd.concat(all_sample_data, ignore_index=True)
        combined_sample = combined_sample.sort_values(['stock_symbol', 'Datetime']).reset_index(drop=True)
        
        # 計算預測步數
        prediction_steps = self.calculate_trading_day_steps(combined_sample, target_trading_days)
        
        print_green(f"✓ 全局預測步數計算完成: {prediction_steps}")
        return prediction_steps

def main():
    normalizer = MegaDataNormalizer()
    normalizer.process_in_chunks(chunk_size=1)  # 每次只處理1個股票，最安全

if __name__ == "__main__":
    main()