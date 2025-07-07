import pandas as pd
import numpy as np
from colors import *
import warnings
import os

warnings.filterwarnings('ignore')

def debug_nan_issues():
    """診斷NaN值問題"""
    print_cyan("開始診斷NaN值問題...")
    
    data_dir = 'data/feature'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_nan_info = {}
    
    for i, csv_file in enumerate(csv_files):  # 先檢查所有文件
        print_yellow(f"\n檢查文件 {i+1}: {csv_file}")
        file_path = os.path.join(data_dir, csv_file)
        
        df = pd.read_csv(file_path)
        print_cyan(f"  原始數據形狀: {df.shape}")
        
        # 檢查每一列的NaN情況
        nan_cols = []
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                nan_cols.append((col, nan_count))
                print_red(f"    {col}: {nan_count} 個NaN值")
        
        if not nan_cols:
            print_green("    ✓ 此文件無NaN值")
        
        # 檢查無限值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_cols.append((col, inf_count))
                print_red(f"    {col}: {inf_count} 個無限值")
        
        if not inf_cols:
            print_green("    ✓ 此文件無無限值")
        
        # 檢查數據類型
        print_cyan("    數據類型:")
        for col in df.columns:
            print_cyan(f"      {col}: {df[col].dtype}")
        
        all_nan_info[csv_file] = {
            'nan_cols': nan_cols,
            'inf_cols': inf_cols,
            'shape': df.shape
        }
    
    return all_nan_info

def test_feature_engineering():
    """測試特徵工程過程"""
    print_cyan("\n測試特徵工程過程...")
    
    # 載入一個文件進行詳細測試
    data_dir = 'data/feature'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    test_file = csv_files[0]
    
    print_yellow(f"使用測試文件: {test_file}")
    
    df = pd.read_csv(os.path.join(data_dir, test_file))
    df['stock_symbol'] = os.path.splitext(test_file)[0]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    print_cyan(f"處理前數據形狀: {df.shape}")
    print_cyan(f"處理前NaN數量: {df.isnull().sum().sum()}")
    
    # 逐步進行特徵工程
    df_processed = df.copy()
    
    # 1. 處理缺失值
    print_yellow("\n1. 處理缺失值...")
    before_fillna = df_processed.isnull().sum().sum()
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    after_fillna = df_processed.isnull().sum().sum()
    print_cyan(f"   fillna前: {before_fillna}, fillna後: {after_fillna}")
    
    # 2. 時間特徵
    print_yellow("\n2. 創建時間特徵...")
    df_processed['Hour'] = df_processed['Datetime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['Datetime'].dt.dayofweek
    
    # 檢查是否產生NaN
    print_cyan(f"   Hour NaN: {df_processed['Hour'].isnull().sum()}")
    print_cyan(f"   DayOfWeek NaN: {df_processed['DayOfWeek'].isnull().sum()}")
    
    # 三角函數特徵
    df_processed['Hour_sin'] = np.sin(2 * np.pi * df_processed['Hour'] / 24)
    df_processed['Hour_cos'] = np.cos(2 * np.pi * df_processed['Hour'] / 24)
    df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
    df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
    
    # 檢查三角函數特徵
    trig_features = ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']
    for feat in trig_features:
        nan_count = df_processed[feat].isnull().sum()
        inf_count = np.isinf(df_processed[feat]).sum()
        print_cyan(f"   {feat} - NaN: {nan_count}, Inf: {inf_count}")
    
    # 3. 價格特徵
    print_yellow("\n3. 創建價格特徵...")
    df_processed['Price_Range'] = df_processed['High'] - df_processed['Low']
    df_processed['Price_Range_Pct'] = df_processed['Price_Range'] / df_processed['Close']
    df_processed['Open_Close_Ratio'] = df_processed['Open'] / df_processed['Close']
    
    price_features = ['Price_Range', 'Price_Range_Pct', 'Open_Close_Ratio']
    for feat in price_features:
        nan_count = df_processed[feat].isnull().sum()
        inf_count = np.isinf(df_processed[feat]).sum()
        zero_count = (df_processed[feat] == 0).sum()
        print_cyan(f"   {feat} - NaN: {nan_count}, Inf: {inf_count}, Zero: {zero_count}")
        
        # 檢查除零問題
        if feat == 'Price_Range_Pct':
            zero_close = (df_processed['Close'] == 0).sum()
            print_red(f"     Close為0的數量: {zero_close}")
        if feat == 'Open_Close_Ratio':
            zero_close = (df_processed['Close'] == 0).sum()
            print_red(f"     Close為0的數量: {zero_close}")
    
    # 4. 成交量特徵
    print_yellow("\n4. 創建成交量特徵...")
    if 'Volume' in df_processed.columns:
        # 檢查Volume是否有負值或0值
        negative_volume = (df_processed['Volume'] < 0).sum()
        zero_volume = (df_processed['Volume'] == 0).sum()
        print_cyan(f"   Volume負值: {negative_volume}, 零值: {zero_volume}")
        
        df_processed['Volume_Log'] = np.log1p(df_processed['Volume'])
        
        nan_count = df_processed['Volume_Log'].isnull().sum()
        inf_count = np.isinf(df_processed['Volume_Log']).sum()
        print_cyan(f"   Volume_Log - NaN: {nan_count}, Inf: {inf_count}")
    
    # 5. 技術指標特徵
    print_yellow("\n5. 創建技術指標特徵...")
    if 'RSI' in df_processed.columns:
        rsi_nan = df_processed['RSI'].isnull().sum()
        print_cyan(f"   RSI NaN: {rsi_nan}")
        
        df_processed['RSI_Overbought'] = (df_processed['RSI'] > 70).astype(int)
        df_processed['RSI_Oversold'] = (df_processed['RSI'] < 30).astype(int)
        
        # 檢查新特徵
        for feat in ['RSI_Overbought', 'RSI_Oversold']:
            nan_count = df_processed[feat].isnull().sum()
            print_cyan(f"   {feat} NaN: {nan_count}")
    
    # 總體檢查
    print_yellow(f"\n最終檢查:")
    print_cyan(f"   處理後數據形狀: {df_processed.shape}")
    print_cyan(f"   總NaN數量: {df_processed.isnull().sum().sum()}")
    print_cyan(f"   總Inf數量: {np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()}")
    
    # 找出包含NaN的列
    nan_columns = df_processed.columns[df_processed.isnull().any()].tolist()
    if nan_columns:
        print_red(f"   包含NaN的列: {nan_columns}")
        for col in nan_columns:
            print_red(f"     {col}: {df_processed[col].isnull().sum()} 個NaN")
    
    return df_processed

if __name__ == "__main__":
    # 診斷原始數據
    nan_info = debug_nan_issues()
    
    # 測試特徵工程
    processed_df = test_feature_engineering()