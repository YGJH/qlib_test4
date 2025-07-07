import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import glob
import re
from colors import *

warnings.filterwarnings('ignore')
def load_fixed_data():
    """載入修復後的數據"""
    data_dir = 'robust_normalized_data'
    
    # 找到最新的檔案
    npy_files = glob.glob(os.path.join(data_dir, 'sequences_*.npy'))
    if not npy_files:
        raise FileNotFoundError("找不到處理後的數據檔案")
    
    latest_seq_file = max(npy_files, key=os.path.getctime)
    timestamp_match = re.search(r'_(\d{8}_\d{6})\.npy', latest_seq_file)
    timestamp = timestamp_match.group(1)
    
    # 載入數據
    sequences = np.load(latest_seq_file)
    targets = np.load(os.path.join(data_dir, f'targets_{timestamp}.npy'))
    
    with open(os.path.join(data_dir, f'metadata_{timestamp}.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print_green(f"載入數據: 序列 {sequences.shape}, 目標 {targets.shape}")
    return sequences, targets, metadata

def inspect_normalized_data():
    """檢查正規化後的資料質量"""
    
    from pathlib import Path 
    font_path = Path(__file__).parent / "ttc" / "GenKiGothic2JP-B-03.ttf"
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = [prop.get_name()]
    
    
    # 載入資料
    sequences, targets, metadata = load_fixed_data()


    stock_to_id = metadata['stock_to_id']
    feature_scaler = metadata['feature_scaler']
    target_scaler = metadata['target_scaler']
    feature_cols = metadata['feature_cols']

    print("=" * 80)
    print("資料檢查報告")
    print("=" * 80)
    
    # 1. 基本資訊
    print(f"\n1. 基本資訊:")
    print(f"   特徵資料形狀: {sequences.shape}")
    print(f"   目標資料形狀: {targets.shape}")
    print(f"   特徵數量: {len(metadata['feature_names'])}")
    print(f"   股票數量: {feature_cols['stock_symbol'].nunique()}")
    print(f"   股票清單: {list(feature_cols['stock_symbol'].unique())}")
    
    # 2. 時間範圍檢查
    print(f"\n2. 時間範圍:")
    sequences['Datetime'] = pd.to_datetime(sequences['Datetime'])
    targets['datetime'] = pd.to_datetime(targets['datetime'])
    print(f"   特徵時間範圍: {sequences['Datetime'].min()} 到 {sequences['Datetime'].max()}")
    print(f"   目標時間範圍: {targets['datetime'].min()} 到 {targets['datetime'].max()}")
    
    # 3. 檢查標準化後的數值範圍
    print(f"\n3. 標準化後數值檢查:")
    feature_cols = metadata['feature_names']
    numeric_features = sequences[feature_cols]
    
    print(f"   特徵值統計:")
    print(f"     平均值範圍: {numeric_features.mean().min():.4f} 到 {numeric_features.mean().max():.4f}")
    print(f"     標準差範圍: {numeric_features.std().min():.4f} 到 {numeric_features.std().max():.4f}")
    print(f"     最小值範圍: {numeric_features.min().min():.4f} 到 {numeric_features.min().max():.4f}")
    print(f"     最大值範圍: {numeric_features.max().min():.4f} 到 {numeric_features.max().max():.4f}")
    
    # 4. 目標變數檢查
    print(f"\n4. 目標變數檢查:")
    print(f"   target_regression 統計:")
    print(f"     平均值: {targets['target_regression'].mean():.4f}")
    print(f"     標準差: {targets['target_regression'].std():.4f}")
    print(f"     最小值: {targets['target_regression'].min():.4f}")
    print(f"     最大值: {targets['target_regression'].max():.4f}")
    print(f"     缺失值: {targets['target_regression'].isnull().sum()}")

    # 5. 檢查數據連續性
    print(f"\n5. 數據連續性檢查:")
    for stock in sequences['stock_symbol'].unique()[:3]:  # 檢查前3隻股票
        stock_data = sequences[sequences['stock_symbol'] == stock].sort_values('Datetime')
        time_diffs = stock_data['Datetime'].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0]
        print(f"   {stock}: 最常見時間間隔 = {most_common_diff}")
    
    # 6. 檢查是否有數據洩漏
    print(f"\n6. 數據洩漏檢查:")
    sample_stock = sequences['stock_symbol'].iloc[0]
    sample_data = sequences[sequences['stock_symbol'] == sample_stock].head(10)
    sample_targets = targets[targets['stock_symbol'] == sample_stock].head(10)

    print(f"   範例股票: {sample_stock}")
    print(f"   特徵時間: {sample_data['Datetime'].iloc[0]} 到 {sample_data['Datetime'].iloc[-1]}")
    print(f"   目標時間: {sample_targets['datetime'].iloc[0]} 到 {sample_targets['datetime'].iloc[-1]}")
    
    # 7. 檢查scaler資訊
    print(f"\n7. Scaler資訊:")
    if hasattr(target_scaler, 'mean_'):
        print(f"   類型: StandardScaler")
        print(f"   特徵平均值範圍: {target_scaler.mean_.min():.4f} 到 {target_scaler.mean_.max():.4f}")
        print(f"   特徵縮放範圍: {target_scaler.scale_.min():.4f} 到 {target_scaler.scale_.max():.4f}")
        
        # 檢查Close價格的標準化參數
        if 'Close' in feature_cols:
            close_idx = feature_cols.index('Close')
            print(f"   Close價格 - 平均值: {target_scaler.mean_[close_idx]:.4f}")
            print(f"   Close價格 - 縮放值: {target_scaler.scale_[close_idx]:.4f}")
    
    # 8. 可視化部分數據
    print(f"\n8. 生成可視化圖表...")
    
    # 選擇一支股票進行詳細分析
    sample_stock = sequences['stock_symbol'].unique()[0]
    stock_features = sequences[sequences['stock_symbol'] == sample_stock].sort_values('Datetime')
    stock_targets = targets[targets['stock_symbol'] == sample_stock].sort_values('datetime')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    



    # 標準化後的Close價格
    if 'Close' in feature_cols:
        axes[0,0].plot(stock_features['Datetime'][:100], stock_features['Close'][:100])
        axes[0,0].set_title(f'{sample_stock} - 標準化後Close價格')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # 目標變數分布
    axes[0,1].hist(stock_targets['target_regression'], bins=50, alpha=0.7)
    axes[0,1].set_title('目標變數分布')
    axes[0,1].set_xlabel('target_regression')
    
    # 特徵相關性熱圖 (前20個特徵)
    corr_features = stock_features.iloc[:, :20].corr()
    sns.heatmap(corr_features, ax=axes[1,0], cmap='coolwarm', center=0)
    axes[1,0].set_title('特徵相關性熱圖 (前20個)')
    
    # 不同股票的目標變數分布
    stock_target_stats = stock_targets.groupby('stock_symbol')['target_regression'].agg(['mean', 'std'])
    axes[1,1].scatter(stock_target_stats['mean'], stock_target_stats['std'])
    axes[1,1].set_xlabel('目標變數平均值')
    axes[1,1].set_ylabel('目標變數標準差')
    axes[1,1].set_title('各股票目標變數統計')
    
    plt.tight_layout()
    plt.savefig('data_inspection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'sequences': sequences,
        'targets': targets,
        'scaler': target_scaler,
        'metadata': metadata,
        'feature_cols': feature_cols
    }

if __name__ == "__main__":
    data_info = inspect_normalized_data()