
#!/usr/bin/env python3
"""
股票資料正規化腳本
Stock Data Normalization Script for Machine Learning

將股票CSV資料轉換成適合機器學習模型訓練的格式
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

class StockDataNormalizer:
    """股票資料正規化器"""
    
    def __init__(self, scaling_method='standard'):
        """
        初始化正規化器
        
        Args:
            scaling_method: 正規化方法 ('standard', 'minmax', 'robust')
        """
        self.scaling_method = scaling_method
        self.scalers = {}
        self.feature_stats = {}
        self.is_fitted = False
        
        # 初始化scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
    
    def load_data(self, file_path):
        """載入CSV資料"""
        try:
            df = pd.read_csv(file_path)
            
            # 從檔案路徑提取股票代碼
            stock_symbol = os.path.splitext(os.path.basename(file_path))[0]
            df['stock_symbol'] = stock_symbol
            
            print(f"成功載入資料: {file_path}")
            print(f"   股票代碼: {stock_symbol}")
            print(f"   資料形狀: {df.shape}")
            print(f"   時間範圍: {df['Datetime'].iloc[0]} 到 {df['Datetime'].iloc[-1]}")
            return df
        except Exception as e:
            print(f"載入資料失敗: {str(e)}")
            return None
    
    def analyze_data_quality(self, df):
        """分析資料品質"""
        print("\n資料品質分析:")
        print("=" * 50)
        
        # 缺失值分析
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        print("缺失值統計:")
        for col in df.columns:
            if missing_data[col] > 0:
                print(f"   {col}: {missing_data[col]} ({missing_pct[col]:.2f}%)")
        
        # 資料類型
        print(f"\n資料類型:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        # 基本統計
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n數值欄位統計 ({len(numeric_cols)} 個欄位):")
        stats = df[numeric_cols].describe()
        
        return {
            'missing_data': missing_data,
            'missing_pct': missing_pct,
            'stats': stats,
            'numeric_cols': numeric_cols
        }
    
    def preprocess_datetime(self, df):
        """處理時間欄位"""
        df_processed = df.copy()
        
        # 轉換時間格式
        df_processed['Datetime'] = pd.to_datetime(df_processed['Datetime'])
        
        # 提取時間特徵
        df_processed['Year'] = df_processed['Datetime'].dt.year
        df_processed['Month'] = df_processed['Datetime'].dt.month
        df_processed['Day'] = df_processed['Datetime'].dt.day
        df_processed['Hour'] = df_processed['Datetime'].dt.hour
        df_processed['Minute'] = df_processed['Datetime'].dt.minute
        df_processed['DayOfWeek'] = df_processed['Datetime'].dt.dayofweek
        df_processed['DayOfYear'] = df_processed['Datetime'].dt.dayofyear
        
        # 創建週期性特徵（正弦餘弦編碼）
        df_processed['Hour_sin'] = np.sin(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['Hour_cos'] = np.cos(2 * np.pi * df_processed['Hour'] / 24)
        df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
        df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed['Month'] / 12)
        df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed['Month'] / 12)
        
        print("時間特徵工程完成")
        return df_processed
    
    def handle_missing_values(self, df):
        """處理缺失值"""
        df_processed = df.copy()
        
        # 對於技術指標，使用前向填充和後向填充
        technical_indicators = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 
                               'MACD_Histogram', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                               'Volume_MA', 'Volume_Ratio', 'Volatility']
        
        for col in technical_indicators:
            if col in df_processed.columns:
                # 先前向填充，再後向填充
                df_processed[col] = df_processed[col].fillna(method='ffill').fillna(method='bfill')
                
                # 如果還有缺失值，用該欄位的中位數填充
                if df_processed[col].isnull().any():
                    median_value = df_processed[col].median()
                    df_processed[col].fillna(median_value, inplace=True)
        
        # 對於Price_Change_Pct，第一個值通常是NaN，設為0
        if 'Price_Change_Pct' in df_processed.columns:
            df_processed['Price_Change_Pct'].fillna(0, inplace=True)
        
        print("缺失值處理完成")
        return df_processed
    
    def create_features(self, df):
        """創建額外特徵"""
        df_processed = df.copy()
        
        # 價格相關特徵
        df_processed['Price_Range'] = df_processed['High'] - df_processed['Low']
        df_processed['Price_Range_Pct'] = (df_processed['Price_Range'] / df_processed['Close']) * 100
        df_processed['Open_Close_Ratio'] = df_processed['Open'] / df_processed['Close']
        df_processed['High_Close_Ratio'] = df_processed['High'] / df_processed['Close']
        df_processed['Low_Close_Ratio'] = df_processed['Low'] / df_processed['Close']
        
        # 成交量相關特徵
        if 'Volume' in df_processed.columns:
            df_processed['Volume_Log'] = np.log1p(df_processed['Volume'])
            
            # 成交量移動平均
            df_processed['Volume_MA5'] = df_processed['Volume'].rolling(window=5).mean()
            df_processed['Volume_MA10'] = df_processed['Volume'].rolling(window=10).mean()
            df_processed['Volume_Ratio_5'] = df_processed['Volume'] / df_processed['Volume_MA5']
            df_processed['Volume_Ratio_10'] = df_processed['Volume'] / df_processed['Volume_MA10']
        
        # 技術分析特徵
        if all(col in df_processed.columns for col in ['MA5', 'MA10', 'MA20']):
            df_processed['MA_Trend_5_10'] = (df_processed['MA5'] > df_processed['MA10']).astype(int)
            df_processed['MA_Trend_10_20'] = (df_processed['MA10'] > df_processed['MA20']).astype(int)
            df_processed['Price_Above_MA5'] = (df_processed['Close'] > df_processed['MA5']).astype(int)
            df_processed['Price_Above_MA20'] = (df_processed['Close'] > df_processed['MA20']).astype(int)
        
        # RSI相關特徵
        if 'RSI' in df_processed.columns:
            df_processed['RSI_Overbought'] = (df_processed['RSI'] > 70).astype(int)
            df_processed['RSI_Oversold'] = (df_processed['RSI'] < 30).astype(int)
            df_processed['RSI_Neutral'] = ((df_processed['RSI'] >= 30) & (df_processed['RSI'] <= 70)).astype(int)
        
        # 布林通道相關特徵
        if all(col in df_processed.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            df_processed['BB_Position'] = (df_processed['Close'] - df_processed['BB_Lower']) / (df_processed['BB_Upper'] - df_processed['BB_Lower'])
            df_processed['BB_Squeeze'] = (df_processed['BB_Upper'] - df_processed['BB_Lower']) / df_processed['Close']
        
        # 價格動量特徵
        for window in [3, 5, 10]:
            df_processed[f'Return_{window}d'] = df_processed['Close'].pct_change(window)
            df_processed[f'Volatility_{window}d'] = df_processed['Close'].rolling(window=window).std()
        
        print("特徵工程完成")
        return df_processed
    
    def create_target_variables(self, df):
        """創建目標變數（用於預測）"""
        df_processed = df.copy()
        
        # 下一期價格變化
        df_processed['Next_Close'] = df_processed['Close'].shift(-1)
        df_processed['Next_Return'] = df_processed['Close'].pct_change(1).shift(-1)
        
        # 價格方向預測（二元分類）
        df_processed['Price_Direction'] = (df_processed['Next_Return'] > 0).astype(int)
        
        # 價格變化幅度分類（多元分類）
        df_processed['Return_Category'] = pd.cut(df_processed['Next_Return'], 
                                               bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                               labels=['大跌', '小跌', '持平', '小漲', '大漲'])
        
        # 將類別編碼為數字
        if 'Return_Category' in df_processed.columns:
            le = LabelEncoder()
            df_processed['Return_Category_Encoded'] = le.fit_transform(df_processed['Return_Category'].fillna('持平'))
        
        print("目標變數創建完成")
        return df_processed
    
    def normalize_features(self, df, fit=True):
        """正規化特徵"""
        df_processed = df.copy()
        
        # 定義不需要正規化的欄位
        exclude_cols = ['Datetime', 'stock_symbol', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 
                       'DayOfYear', 'Dividends', 'Stock Splits', 'Price_Direction',
                       'Return_Category', 'Return_Category_Encoded']
        
        # 獲取需要正規化的數值欄位
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if fit:
            # 擬合並轉換
            df_processed[cols_to_normalize] = self.scaler.fit_transform(df_processed[cols_to_normalize])
            self.is_fitted = True
            
            # 儲存統計資訊
            self.feature_stats = {
                'mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'feature_names': cols_to_normalize
            }
        else:
            if not self.is_fitted:
                raise ValueError("Scaler has not been fitted yet. Call with fit=True first.")
            # 只轉換
            df_processed[cols_to_normalize] = self.scaler.transform(df_processed[cols_to_normalize])
        
        print(f"特徵正規化完成 (方法: {self.scaling_method})")
        print(f"   正規化欄位數: {len(cols_to_normalize)}")
        
        return df_processed
    
    def prepare_for_ml(self, df):
        """準備機器學習格式"""
        df_processed = df.copy()
        
        # 移除最後一行（因為沒有下一期的目標值）
        df_processed = df_processed[:-1]
        
        # 移除仍有缺失值的行
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        
        if dropped_rows > 0:
            print(f"移除了 {dropped_rows} 行包含缺失值的資料")
        
        # 分離特徵和目標變數
        target_cols = ['Next_Close', 'Next_Return', 'Price_Direction', 'Return_Category_Encoded']
        # 保留 Datetime 和 stock_symbol
        preserve_cols = ['Datetime', 'stock_symbol']
        feature_cols = [col for col in df_processed.columns 
                       if col not in target_cols + ['Return_Category'] + preserve_cols]
        
        X = df_processed[feature_cols]
        y_regression = df_processed['Next_Close'] # <--- 主要目標
        y_secondary_regression = df_processed['Next_Return']
        y_classification = df_processed['Price_Direction']
        y_multiclass = df_processed['Return_Category_Encoded']
        
        print("機器學習格式準備完成")
        print(f"   特徵維度: {X.shape}")
        print(f"   特徵數量: {len(feature_cols)}")
        
        return {
            'features': X,
            'target_regression': y_regression,
            'target_secondary_regression': y_secondary_regression,
            'target_classification': y_classification,
            'target_multiclass': y_multiclass,
            'datetime': df_processed['Datetime'],
            'stock_symbol': df_processed['stock_symbol'],
            'feature_names': feature_cols,
            'target_names': target_cols,
            'time_step': 10 # <--- 在此處定義 time_step
        }
    
    def save_normalized_data(self, data, output_dir='normalized_data'):
        """儲存正規化後的資料"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 儲存特徵資料（包含datetime和stock_symbol）
        features_file = os.path.join(output_dir, f'combined_features_{timestamp}.csv')
        features_with_metadata = pd.concat([
            data['datetime'].reset_index(drop=True),
            data['stock_symbol'].reset_index(drop=True), 
            data['features'].reset_index(drop=True)
        ], axis=1)
        features_with_metadata.to_csv(features_file, index=False)
        print(f"特徵資料已儲存: {features_file}")
        
        # 儲存目標變數
        targets_file = os.path.join(output_dir, f'combined_targets_{timestamp}.csv')
        target_df = pd.DataFrame({
            'datetime': data['datetime'],
            'stock_symbol': data['stock_symbol'],
            'target_regression': data['target_regression'],
            'target_classification': data['target_classification'],
            'target_multiclass': data['target_multiclass']
        })
        target_df.to_csv(targets_file, index=False)
        print(f"目標變數已儲存: {targets_file}")
        
        # 儲存scaler
        scaler_file = os.path.join(output_dir, f'combined_scaler_{timestamp}.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler已儲存: {scaler_file}")
        
        # 儲存特徵名稱和統計資訊
        metadata_file = os.path.join(output_dir, f'combined_metadata_{timestamp}.pkl')
        metadata = {
            'feature_names': data['feature_names'],
            'target_names': data['target_names'],
            'scaling_method': self.scaling_method,
            'feature_stats': self.feature_stats,
            'total_stocks': len(csv_files),
            'total_samples': len(combined_features),
            'time_step': ml_data['time_step'], # <--- 儲存 time_step
            'target_col': 'Next_Close' # <--- 儲存目標欄位名稱
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"元數據已儲存: {metadata_file}")
        
        return {
            'features_file': features_file,
            'targets_file': targets_file,
            'scaler_file': scaler_file,
            'metadata_file': metadata_file
        }


def main():
    """主函數 - 執行完整的資料正規化流程"""
    print("股票資料正規化開始...")
    print("=" * 60)
    
    DATA_DIR = 'data/feature'
    
    # 收集所有處理過的資料
    all_processed_data = []
    
    # 初始化正規化器
    normalizer = StockDataNormalizer(scaling_method='standard')
    
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"發現 {len(csv_files)} 個CSV檔案: {csv_files}")
    
    # 第一階段：預處理所有檔案但不正規化
    print("\n第一階段：預處理所有檔案...")
    for i, csv_file in enumerate(csv_files):
        csv_file_path = os.path.join(DATA_DIR, csv_file)
        print(f"\n處理檔案 [{i+1}/{len(csv_files)}]: {csv_file}")
        
        if not os.path.exists(csv_file_path):
            print(f"檔案不存在: {csv_file_path}")
            continue
    
        # 載入資料
        df = normalizer.load_data(csv_file_path)
        if df is None:
            continue
    
        # 分析資料品質
        quality_analysis = normalizer.analyze_data_quality(df)
        
        # 預處理步驟（不包含正規化）
        print("\n開始資料預處理...")
        df = normalizer.preprocess_datetime(df)
        df = normalizer.handle_missing_values(df)
        df = normalizer.create_features(df)
        df = normalizer.create_target_variables(df)
        
        # 儲存預處理後的資料
        all_processed_data.append(df)
    
    if not all_processed_data:
        print("沒有成功處理任何檔案！")
        return
    
    # 第二階段：合併所有資料並統一正規化
    print(f"\n第二階段：合併 {len(all_processed_data)} 個股票的資料...")
    combined_df = pd.concat(all_processed_data, ignore_index=True)
    print(f"合併後資料形狀: {combined_df.shape}")
    
    # 對合併後的資料進行正規化
    print("\n進行統一正規化...")
    combined_df = normalizer.normalize_features(combined_df, fit=True)
    
    # 準備機器學習格式
    print("\n準備機器學習格式...")
    ml_data = normalizer.prepare_for_ml(combined_df)
    
    # 準備最終的特徵和目標資料
    features_with_meta = pd.concat([
        ml_data['datetime'].reset_index(drop=True),
        ml_data['stock_symbol'].reset_index(drop=True),
        ml_data['features'].reset_index(drop=True)
    ], axis=1)
    
    targets_with_meta = pd.DataFrame({
        'datetime': ml_data['datetime'],
        'stock_symbol': ml_data['stock_symbol'],
        'target_regression': ml_data['target_regression'],
        'target_classification': ml_data['target_classification'],
        'target_multiclass': ml_data['target_multiclass']
    })
    
    if all_processed_data:
        # 合併所有資料
        # print(f"\n合併 {len(all_features)} 個股票的資料...")
        # combined_features = pd.concat(all_features, ignore_index=True)
        # combined_targets = pd.concat(all_targets, ignore_index=True)
        combined_features = features_with_meta
        combined_targets = targets_with_meta
        
        # 儲存合併後的資料
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'normalized_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # 儲存特徵資料
        features_file = os.path.join(output_dir, f'combined_features_{timestamp}.csv')
        combined_features.to_csv(features_file, index=False)
        print(f"合併特徵資料已儲存: {features_file}")
        
        # 儲存目標資料
        targets_file = os.path.join(output_dir, f'combined_targets_{timestamp}.csv')
        combined_targets.to_csv(targets_file, index=False)
        print(f"合併目標資料已儲存: {targets_file}")
        
        # 儲存scaler和metadata
        scaler_file = os.path.join(output_dir, f'combined_scaler_{timestamp}.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(normalizer.scaler, f)
        print(f"Scaler已儲存: {scaler_file}")
        
        metadata_file = os.path.join(output_dir, f'combined_metadata_{timestamp}.pkl')
        metadata = {
            'feature_names': ml_data['feature_names'],
            'target_names': ml_data['target_names'],
            'scaling_method': normalizer.scaling_method,
            'feature_stats': normalizer.feature_stats,
            'total_stocks': len(csv_files),
            'total_samples': len(combined_features),
            'time_step': ml_data['time_step'], # <--- 儲存 time_step
            'target_col': 'Next_Close' # <--- 儲存目標欄位名稱
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"元數據已儲存: {metadata_file}")
        
        # 顯示總結
        print("\n" + "=" * 60)
        print("資料正規化完成！")
        print(f"最終資料形狀: {combined_features.shape}")
        print(f"特徵數量: {len(ml_data['feature_names'])}")
        print(f"股票數量: {len(csv_files)}")
        print(f"總樣本數: {len(combined_features)}")
        print(f"目標變數: {len(ml_data['target_names'])}")
        
        print("\n已生成的檔案:")
        print(f"   features_file: {features_file}")
        print(f"   targets_file: {targets_file}")
        print(f"   scaler_file: {scaler_file}")
        print(f"   metadata_file: {metadata_file}")
        
        print("\n使用建議:")
        print("   1. 使用 combined_features.csv 作為模型輸入")
        print("   2. datetime 和 stock_symbol 已保留在特徵中")
        print("   3. 使用 combined_targets.csv 中的目標變數進行訓練")
        print("   4. 回歸任務: target_regression")
        print("   5. 二元分類: target_classification")
        print("   6. 多元分類: target_multiclass")
        print("   7. 載入 combined_scaler.pkl 對新資料進行相同的正規化")
    else:
        print("沒有成功處理任何檔案！")


if __name__ == "__main__":
    main()
