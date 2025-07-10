import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import glob
import re
import os
from datetime import datetime, timedelta
import warnings
from colors import *
import psutil
import GPUtil

warnings.filterwarnings('ignore')

from stock_transformer import StockTransformer



def load_time_normalized_data():
    """Load time-normalized data - compatible with main.py"""
    data_dir = 'time_normalized_data'
    
    # Find latest files
    train_seq_files = glob.glob(os.path.join(data_dir, 'train_sequences_*.npy'))
    if not train_seq_files:
        raise FileNotFoundError("No time-normalized training data found")
    
    latest_train_file = max(train_seq_files, key=os.path.getctime)
    timestamp_match = re.search(r'_(\d{8}_\d{6})\.npy', latest_train_file)
    timestamp = timestamp_match.group(1)
    
    # Load all data
    train_sequences = np.load(os.path.join(data_dir, f'train_sequences_{timestamp}.npy'))
    train_targets = np.load(os.path.join(data_dir, f'train_targets_{timestamp}.npy'))
    
    # Check if test data exists
    test_seq_file = os.path.join(data_dir, f'test_sequences_{timestamp}.npy')
    test_tar_file = os.path.join(data_dir, f'test_targets_{timestamp}.npy')
    
    if os.path.exists(test_seq_file) and os.path.exists(test_tar_file):
        test_sequences = np.load(test_seq_file)
        test_targets = np.load(test_tar_file)
    else:
        print_yellow("No test data found, creating empty arrays")
        test_sequences = np.array([])
        test_targets = np.array([])
    
    # Load metadata
    with open(os.path.join(data_dir, f'metadata_{timestamp}.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print_green(f"Loaded time-normalized data:")
    print_green(f"  Training: {train_sequences.shape}")
    print_green(f"  Test: {test_sequences.shape}")
    print_green(f"  Test period: {metadata.get('test_start_date', 'N/A')}")
    
    # Create metadata objects for train and test
    train_metadata = metadata['train_metadata']
    test_metadata = metadata['test_metadata']
    
    return {
        'train_sequences': train_sequences,
        'test_sequences': test_sequences,
        'train_targets': train_targets,
        'test_targets': test_targets,
        'train_metadata': train_metadata,
        'test_metadata': test_metadata,
        'full_metadata': metadata
    }


def time_based_split(sequences, targets, metadata, test_days=6):
    """基於時間的數據分割"""
    print_cyan(f"\n按時間分割數據，測試期: 最新 {test_days} 天")
    
    # 提取所有時間戳
    all_datetimes = []
    for i, item in enumerate(metadata['metadata']):
        datetime_val = item['datetime']
        if isinstance(datetime_val, str):
            datetime_val = pd.to_datetime(datetime_val)
        all_datetimes.append((i, datetime_val))
    
    # 按時間排序
    all_datetimes.sort(key=lambda x: x[1])
    
    # 找出所有唯一的日期
    unique_dates = sorted(list(set([dt.date() for _, dt in all_datetimes])))
    print_cyan(f"數據時間範圍: {unique_dates[0]} 到 {unique_dates[-1]}")
    print_cyan(f"總共 {len(unique_dates)} 天的數據")
    
    # 確定測試期開始日期
    if len(unique_dates) <= test_days:
        print_yellow(f"警告: 總天數 {len(unique_dates)} 小於等於測試天數 {test_days}")
        test_start_date = unique_dates[len(unique_dates)//2]  # 使用後一半作為測試
        print_yellow(f"調整測試開始日期為: {test_start_date}")
    else:
        test_start_date = unique_dates[-test_days]
    
    print_cyan(f"測試期開始日期: {test_start_date}")
    print_cyan(f"訓練期: {unique_dates[0]} 到 {test_start_date - timedelta(days=1)}")
    print_cyan(f"測試期: {test_start_date} 到 {unique_dates[-1]}")
    
    # 分割索引
    train_indices = []
    test_indices = []
    
    for i, datetime_val in all_datetimes:
        if datetime_val.date() < test_start_date:
            train_indices.append(i)
        else:
            test_indices.append(i)
    
    print_cyan(f"訓練樣本數: {len(train_indices)}")
    print_cyan(f"測試樣本數: {len(test_indices)}")
    
    if len(test_indices) == 0:
        raise ValueError("測試集為空，請調整 test_days 參數")
    
    # 分割數據
    train_sequences = sequences[train_indices]
    test_sequences = sequences[test_indices]
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]
    
    # 分割元數據
    train_metadata = [metadata['metadata'][i] for i in train_indices]
    test_metadata = [metadata['metadata'][i] for i in test_indices]
    
    # 檢查各股票在訓練集和測試集的分佈
    train_stocks = set([item['stock_symbol'] for item in train_metadata])
    test_stocks = set([item['stock_symbol'] for item in test_metadata])
    
    print_cyan(f"\n股票分佈:")
    print_cyan(f"  訓練集股票數: {len(train_stocks)}")
    print_cyan(f"  測試集股票數: {len(test_stocks)}")
    print_cyan(f"  共同股票數: {len(train_stocks & test_stocks)}")
    
    if len(train_stocks & test_stocks) == 0:
        print_red("警告: 訓練集和測試集沒有共同股票！")
    
    return train_sequences, test_sequences, train_targets, test_targets, train_metadata, test_metadata

class FocalMSELoss(nn.Module):
    """改進的損失函數 - 專注於難預測的樣本"""
    def __init__(self, alpha=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # 使用絕對誤差作為權重，難預測的樣本權重更高
        weights = torch.abs(pred - target) ** self.alpha
        focal_mse = weights * mse
        
        if self.reduction == 'mean':
            return focal_mse.mean()
        elif self.reduction == 'sum':
            return focal_mse.sum()
        else:
            return focal_mse

class WarmupCosineScheduler:
    """帶預熱的余弦退火學習率調度器"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 預熱階段
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火階段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def evaluate_model(model, data_loader, target_scaler, device):
    """評估模型性能 - 修復MAPE計算"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets, batch_stock_ids in data_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_stock_ids = batch_stock_ids.squeeze().to(device, non_blocking=True)
            
            with autocast():
                outputs = model(batch_features, batch_stock_ids)
            
            pred = outputs['predictions'].cpu().numpy()
            actual = batch_targets.cpu().numpy()
            
            predictions.append(pred)
            actuals.append(actual)
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # 反標準化
    pred_flat = predictions.reshape(-1, 1)
    actual_flat = actuals.reshape(-1, 1)
    
    pred_denorm = target_scaler.inverse_transform(pred_flat).reshape(predictions.shape)
    actual_denorm = target_scaler.inverse_transform(actual_flat).reshape(actuals.shape)
    
    # 修復數值範圍 - 限制在合理範圍內
    pred_denorm = np.clip(pred_denorm, -1.0, 1.0)  # 限制在-100%到+100%之間
    actual_denorm = np.clip(actual_denorm, -1.0, 1.0)
    
    # 計算指標
    mse = mean_squared_error(actual_denorm.flatten(), pred_denorm.flatten())
    mae = mean_absolute_error(actual_denorm.flatten(), pred_denorm.flatten())
    rmse = np.sqrt(mse)
    
    # 修復MAPE計算 - 避免除零和極端值
    actual_abs = np.abs(actual_denorm)
    mask = actual_abs > 1e-6  # 只計算絕對值大於1e-6的樣本
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual_denorm[mask] - pred_denorm[mask]) / actual_denorm[mask])) * 100
        mape = min(mape, 1000.0)  # 限制MAPE最大值為1000%
    else:
        mape = 0.0
    
    # 計算方向準確率 (短期和長期)
    pred_direction_1d = (pred_denorm[:, 47] > 0).astype(int)  # 1天後
    actual_direction_1d = (actual_denorm[:, 47] > 0).astype(int)
    direction_accuracy_1d = (pred_direction_1d == actual_direction_1d).mean()
    
    pred_direction_7d = (pred_denorm[:, -1] > 0).astype(int)  # 7天後
    actual_direction_7d = (actual_denorm[:, -1] > 0).astype(int)
    direction_accuracy_7d = (pred_direction_7d == actual_direction_7d).mean()
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'direction_accuracy_1d': float(direction_accuracy_1d),
        'direction_accuracy_7d': float(direction_accuracy_7d),
        'predictions': pred_denorm,
        'actuals': actual_denorm
    }

def predict_stock_future(model, metadata, device, days_ahead=7, sequences=None):
    """預測所有股票的未來價格 - 修復累積計算"""
    model.eval()
    
    stock_to_id = metadata['stock_to_id']
    target_scaler = metadata['target_scaler']
    feature_scaler = metadata['feature_scaler']
    
    predictions = {}
    
    print_cyan(f"\n預測未來 {days_ahead} 天的股價變化...")
    
    with torch.no_grad():
        # 為每支股票預測
        for stock_symbol, stock_id in stock_to_id.items():
            # 獲取該股票的最新序列
            stock_sequences = []
            stock_metadata = []
            
            for i, meta in enumerate(metadata['metadata']):
                if meta['stock_id'] == stock_id:
                    stock_sequences.append(sequences[i])
                    stock_metadata.append(meta)
            
            if not stock_sequences:
                continue
            
            # 使用最新的序列進行預測
            latest_sequence = stock_sequences[-1]  # 最新的60個時間步
            latest_meta = stock_metadata[-1]
            
            # 轉換為tensor
            input_tensor = torch.FloatTensor(latest_sequence).unsqueeze(0).to(device)
            stock_id_tensor = torch.LongTensor([stock_id]).to(device)
            
            # 預測
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_tensor, stock_id_tensor)
            
            predicted_changes = outputs['predictions'].cpu().numpy()[0]  # [336]
            
            # 反標準化
            predicted_changes_denorm = target_scaler.inverse_transform(
                predicted_changes.reshape(-1, 1)
            ).flatten()
            
            # 限制預測變化在合理範圍內
            predicted_changes_denorm = np.clip(predicted_changes_denorm, -0.2, 0.2)  # 限制單步變化在±20%
            
            # 修復未來價格計算 - 使用對數累積避免指數爆炸
            current_price = latest_meta['current_price']
            
            # 方法1: 直接累積（有限制）
            future_prices = []
            current_price_temp = current_price
            
            for i, change in enumerate(predicted_changes_denorm):
                # 使用小變化累積，避免指數增長
                new_price = current_price_temp * (1 + change * 0.1)  # 縮小變化幅度
                new_price = max(new_price, current_price * 0.5)  # 不能低於原價50%
                new_price = min(new_price, current_price * 2.0)   # 不能高於原價200%
                future_prices.append(new_price)
                current_price_temp = new_price
            
            # 生成時間戳
            last_datetime = latest_meta['datetime']
            if isinstance(last_datetime, str):
                last_datetime = pd.to_datetime(last_datetime)
            
            future_times = []
            for i in range(len(predicted_changes_denorm)):
                future_time = last_datetime + timedelta(minutes=30*(i+1))
                future_times.append(future_time)
            
            # 計算關鍵統計
            price_1d = future_prices[47] if len(future_prices) > 47 else future_prices[-1]  # 1天後
            price_7d = future_prices[-1]  # 7天後
            
            change_1d = ((price_1d - current_price) / current_price) * 100
            change_7d = ((price_7d - current_price) / current_price) * 100
            
            # 確保變化在合理範圍內
            change_1d = np.clip(change_1d, -50, 50)  # 限制在±50%
            change_7d = np.clip(change_7d, -80, 80)  # 限制在±80%
            
            # 計算波動率 - 使用更小的時間窗口
            if len(future_prices) >= 48:
                price_returns = []
                for i in range(1, min(48, len(future_prices))):
                    ret = (future_prices[i] - future_prices[i-1]) / future_prices[i-1]
                    price_returns.append(ret)
                volatility = np.std(price_returns) * 100 * np.sqrt(48)  # 年化波動率
                volatility = min(volatility, 100.0)  # 限制最大波動率
            else:
                volatility = 5.0  # 默認值
            
            predictions[stock_symbol] = {
                'current_price': float(current_price),
                'current_datetime': last_datetime,
                'future_times': future_times,
                'future_prices': future_prices,
                'price_1d': float(price_1d),
                'price_7d': float(price_7d),
                'change_1d': float(change_1d),
                'change_7d': float(change_7d),
                'volatility': float(volatility),
                'trend': 'bullish' if change_7d > 2 else 'bearish' if change_7d < -2 else 'neutral'
            }
    
    return predictions

def visualize_training_results(train_losses, val_losses, metrics):
    """可視化訓練結果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 訓練曲線
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # 2. 預測vs實際 (樣本)
    sample_idx = np.random.choice(len(metrics['predictions']), 3)
    for i, idx in enumerate(sample_idx):
        axes[0, 1].plot(metrics['actuals'][idx][:48*3], alpha=0.7, label=f'Actual {i+1}')
        axes[0, 1].plot(metrics['predictions'][idx][:48*3], alpha=0.7, linestyle='--', label=f'Predicted {i+1}')
    axes[0, 1].set_xlabel('Time Steps (30min intervals)')
    axes[0, 1].set_ylabel('Price Change (%)')
    axes[0, 1].set_title('Sample Predictions vs Actuals (3 days)')
    axes[0, 1].legend()
    
    # 3. 誤差分布
    errors = (metrics['predictions'] - metrics['actuals']).flatten()
    axes[0, 2].hist(errors, bins=50, alpha=0.7, density=True)
    axes[0, 2].set_xlabel('Prediction Error')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].axvline(0, color='red', linestyle='--')
    
    # 4. 散點圖 (1天後預測)
    pred_1d = metrics['predictions'][:, 47]  # 1天後預測
    actual_1d = metrics['actuals'][:, 47]    # 1天後實際
    axes[1, 0].scatter(actual_1d, pred_1d, alpha=0.5, s=1)
    axes[1, 0].plot([actual_1d.min(), actual_1d.max()], [actual_1d.min(), actual_1d.max()], 'r--')
    axes[1, 0].set_xlabel('Actual (1 day)')
    axes[1, 0].set_ylabel('Predicted (1 day)')
    axes[1, 0].set_title('1-Day Prediction Accuracy')
    
    # 5. 散點圖 (7天後預測)
    pred_7d = metrics['predictions'][:, -1]  # 7天後預測
    actual_7d = metrics['actuals'][:, -1]    # 7天後實際
    axes[1, 1].scatter(actual_7d, pred_7d, alpha=0.5, s=1)
    axes[1, 1].plot([actual_7d.min(), actual_7d.max()], [actual_7d.min(), actual_7d.max()], 'r--')
    axes[1, 1].set_xlabel('Actual (7 days)')
    axes[1, 1].set_ylabel('Predicted (7 days)')
    axes[1, 1].set_title('7-Day Prediction Accuracy')
    
    # 6. 方向準確率隨時間變化
    direction_accuracy_over_time = []
    for t in range(0, min(metrics['predictions'].shape[1], 48*7), 12):  # 每6小時一個點
        pred_dir = (metrics['predictions'][:, t] > 0).astype(int)
        actual_dir = (metrics['actuals'][:, t] > 0).astype(int)
        acc = (pred_dir == actual_dir).mean()
        direction_accuracy_over_time.append(acc)
    
    time_points = np.arange(0, len(direction_accuracy_over_time)) * 6  # 6小時間隔
    axes[1, 2].plot(time_points, direction_accuracy_over_time, marker='o')
    axes[1, 2].set_xlabel('Hours Ahead')
    axes[1, 2].set_ylabel('Direction Accuracy')
    axes[1, 2].set_title('Direction Accuracy Over Time')
    axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_stock_predictions(predictions, num_stocks=6):
    """可視化股票預測結果"""
    from pathlib import Path 
    font_path = Path(__file__).parent / "ttc" / "GenKiGothic2JP-B-03.ttf"
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = [prop.get_name()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    if num_stocks == None:
        stocks_to_plot = list(predictions.keys())
    else:
        stocks_to_plot = list(predictions.keys())[:num_stocks]

    for i, stock in enumerate(stocks_to_plot):
        if i >= len(axes):
            break
        
        pred_data = predictions[stock]
        
        # 繪製7天價格預測
        days_to_show = min(7, len(pred_data['future_times']) // 48)
        end_idx = days_to_show * 48
        
        times = pred_data['future_times'][:end_idx]
        prices = pred_data['future_prices'][:end_idx]
        
        axes[i].plot(times, prices, linewidth=2, label='預測價格')
        
        # 添加當前價格線
        axes[i].axhline(y=pred_data['current_price'], color='red', linestyle='--', alpha=0.7,
                       label=f'當前: ${pred_data["current_price"]:.2f}')
        
        # 添加1天和7天預測點
        if len(times) > 48:
            axes[i].scatter(times[47], pred_data['price_1d'], color='orange', s=50, 
                           label=f'1天後: ${pred_data["price_1d"]:.2f} ({pred_data["change_1d"]:+.1f}%)')
        
        axes[i].scatter(times[-1], pred_data['price_7d'], color='green', s=50,
                       label=f'7天後: ${pred_data["price_7d"]:.2f} ({pred_data["change_7d"]:+.1f}%)')
        
        axes[i].set_title(f'{stock} - {pred_data["trend"].upper()} (波動率: {pred_data["volatility"]:.1f}%)')
        axes[i].set_xlabel('時間')
        axes[i].set_ylabel('價格 ($)')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    # 移除多餘的子圖
    for i in range(len(stocks_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('股票未來7天價格預測', fontsize=16)
    plt.tight_layout()
    plt.savefig('stock_future_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_prediction_summary(predictions):
    """打印預測摘要 - 修復極端數值"""
    print_green("\n" + "="*80)
    print_green("📈 股票預測摘要報告")
    print_green("="*80)
    
    # 統計分析
    changes_1d = [pred['change_1d'] for pred in predictions.values()]
    changes_7d = [pred['change_7d'] for pred in predictions.values()]
    volatilities = [pred['volatility'] for pred in predictions.values()]
    
    print_cyan(f"\n整體市場預測:")
    print_cyan(f"  平均1天變化: {np.mean(changes_1d):+.2f}% (標準差: {np.std(changes_1d):.2f}%)")
    print_cyan(f"  平均7天變化: {np.mean(changes_7d):+.2f}% (標準差: {np.std(changes_7d):.2f}%)")
    print_cyan(f"  平均波動率: {np.mean(volatilities):.2f}%")
    
    # 按趨勢分類
    bullish = [s for s, p in predictions.items() if p['trend'] == 'bullish']
    bearish = [s for s, p in predictions.items() if p['trend'] == 'bearish']
    neutral = [s for s, p in predictions.items() if p['trend'] == 'neutral']
    
    print_cyan(f"\n趨勢分布:")
    print_green(f"  看漲 (>+2%): {len(bullish)} 支股票")
    print_red(f"  看跌 (<-2%): {len(bearish)} 支股票")
    print_yellow(f"  中性 (-2%~+2%): {len(neutral)} 支股票")
    
    # 詳細預測 - 只保存關鍵信息
    print_cyan(f"\n詳細預測結果:")
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['change_7d'], reverse=True)
    result = dict()
    
    for stock, pred in sorted_predictions:
        result[stock] = {
            'current_price': pred['current_price'],
            'price_1d': pred['price_1d'],
            'change_1d': pred['change_1d'],
            'price_7d': pred['price_7d'],
            'change_7d': pred['change_7d'],
            'volatility': pred['volatility'],
            'trend': pred['trend']
        }
    
    # 保存預測摘要
    import json
    with open('predictions_summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def train_optimized_model():
    """訓練優化後的模型 - 使用時間分割"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"使用設備: {device}")
    
    # 載入按時間分割的數據
    data_dict = load_time_normalized_data()
    
    train_sequences = data_dict['train_sequences']
    test_sequences = data_dict['test_sequences']
    train_targets = data_dict['train_targets']
    test_targets = data_dict['test_targets']
    train_metadata = data_dict['train_metadata']
    test_metadata = data_dict['test_metadata']
    full_metadata = data_dict['full_metadata']
    
    # 提取元數據
    stock_to_id = full_metadata['stock_to_id']
    feature_scaler = full_metadata['feature_scaler']
    target_scaler = full_metadata['target_scaler']
    feature_cols = full_metadata['feature_cols']
    
    # 創建股票ID數組
    train_stock_ids = np.array([item['stock_id'] for item in train_metadata])
    test_stock_ids = np.array([item['stock_id'] for item in test_metadata])
    
    print_cyan(f"\n時間分割後的數據統計:")
    print_cyan(f"  訓練序列數量: {len(train_sequences)}")
    print_cyan(f"  測試序列數量: {len(test_sequences)}")
    print_cyan(f"  特徵維度: {train_sequences.shape[-1]}")
    print_cyan(f"  預測步數: {train_targets.shape[-1]}")
    print_cyan(f"  股票數量: {len(stock_to_id)}")
    
    # 進一步分割訓練集為訓練/驗證 (80/20)
    # train_seq, val_seq, train_tar, val_tar, train_ids, val_ids = train_test_split(
    #     train_sequences, train_targets, train_stock_ids, 
    #     test_size=0.2, random_state=42, stratify=train_stock_ids
    # )

    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    # 這裡 groups=train_stock_ids，確保同一 group (股票) 不會被分到不同集
    train_idx, val_idx = next(gss.split(train_sequences, train_targets, groups=train_stock_ids))

    train_seq, val_seq = train_sequences[train_idx], train_sequences[val_idx]
    train_tar, val_tar = train_targets[train_idx],   train_targets[val_idx]
    train_ids,  val_ids  = train_stock_ids[train_idx], train_stock_ids[val_idx]

    
    print_cyan(f"\n最終數據分割:")
    print_cyan(f"  訓練集: {len(train_seq)} 個序列")
    print_cyan(f"  驗證集: {len(val_seq)} 個序列")
    print_cyan(f"  測試集: {len(test_sequences)} 個序列")
    
    # 創建數據加載器
    train_dataset = TensorDataset(
        torch.FloatTensor(train_seq),
        torch.FloatTensor(train_tar),
        torch.LongTensor(train_ids)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_seq),
        torch.FloatTensor(val_tar),
        torch.LongTensor(val_ids)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_sequences),
        torch.FloatTensor(test_targets),
        torch.LongTensor(test_stock_ids)
    )
    
    # 優化的數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 創建優化的模型
    model = StockTransformer(
        input_dim=train_sequences.shape[-1],
        num_stocks=len(stock_to_id),
        d_model=256,
        nhead=8,
        num_layers=24,  # 減少層數
        dropout=0.1,
        prediction_horizon=train_targets.shape[-1]
    ).to(device)
    
    # 編譯模型
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print_magenta(f"模型參數總數: {total_params/1e6:.1f}M")

    # 優化的訓練設置
    criterion = FocalMSELoss(alpha=1.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 使用預熱學習率調度
    num_epochs = 100
    warmup_epochs = 10
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, 1e-4, 1e-6)
    
    # 混合精度訓練
    scaler = torch.amp.GradScaler()
    
    # 訓練循環
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print_green(f"\n開始訓練 (共 {num_epochs} 個epoch)...")
    
    for epoch in range(num_epochs):
        # 更新學習率
        current_lr = scheduler.step(epoch)
        
        # 訓練階段
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_features, batch_targets, batch_stock_ids in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_stock_ids = batch_stock_ids.squeeze().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(batch_features, batch_stock_ids)
                loss = criterion(outputs['predictions'], batch_targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
        
        # 驗證階段
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets, batch_stock_ids in val_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_stock_ids = batch_stock_ids.squeeze().to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(batch_features, batch_stock_ids)
                    loss = criterion(outputs['predictions'], batch_targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 早停和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'full_metadata': full_metadata,
                'train_metadata': train_metadata,
                'test_metadata': test_metadata,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'best_time_split_model.pth')
        else:
            patience_counter += 1
        
        # 打印進度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print_cyan(f"Epoch [{epoch+1}/{num_epochs}]")
            print_cyan(f"  Train Loss: {avg_train_loss:.6f}")
            print_cyan(f"  Val Loss: {avg_val_loss:.6f}")
            print_cyan(f"  LR: {current_lr:.7f}")
            print_cyan(f"  Best Val Loss: {best_val_loss:.6f}")
        
        # 早停
        if patience_counter >= patience:
            print_yellow(f"早停在第 {epoch+1} 個epoch (patience: {patience})")
            break
    
    # 刪除不用的記憶體
    del train_loader, optimizer, scheduler, scaler, model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 載入最佳模型進行評估
    checkpoint = torch.load('best_time_split_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 評估模型在驗證集和測試集上的性能
    print_magenta("\n評估模型性能...")
    
    # 驗證集評估
    val_metrics = evaluate_model(model, val_loader, target_scaler, device)
    print_blue(f"\n驗證集性能:")
    print_blue(f"  RMSE: {val_metrics['rmse']:.4f}")
    print_blue(f"  MAE: {val_metrics['mae']:.4f}")
    print_blue(f"  MAPE: {val_metrics['mape']:.2f}%")
    print_blue(f"  1天方向準確率: {val_metrics['direction_accuracy_1d']:.4f}")
    print_blue(f"  7天方向準確率: {val_metrics['direction_accuracy_7d']:.4f}")
    
    # 測試集評估 (真正的未見過數據)
    test_metrics = evaluate_model(model, test_loader, target_scaler, device)
    print_green(f"\n測試集性能 (未見過的最新6天數據):")
    print_green(f"  RMSE: {test_metrics['rmse']:.4f}")
    print_green(f"  MAE: {test_metrics['mae']:.4f}")
    print_green(f"  MAPE: {test_metrics['mape']:.2f}%")
    print_green(f"  1天方向準確率: {test_metrics['direction_accuracy_1d']:.4f}")
    print_green(f"  7天方向準確率: {test_metrics['direction_accuracy_7d']:.4f}")
    
    # 保存評估指標
    metrics_summary = {
        'validation': {
            'rmse': val_metrics['rmse'],
            'mae': val_metrics['mae'],
            'mape': val_metrics['mape'],
            'direction_accuracy_1d': val_metrics['direction_accuracy_1d'],
            'direction_accuracy_7d': val_metrics['direction_accuracy_7d']
        },
        'test': {
            'rmse': test_metrics['rmse'],
            'mae': test_metrics['mae'],
            'mape': test_metrics['mape'],
            'direction_accuracy_1d': test_metrics['direction_accuracy_1d'],
            'direction_accuracy_7d': test_metrics['direction_accuracy_7d']
        }
    }
    
    import json
    with open('time_split_model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4)
    
    # 可視化訓練結果 - 使用測試集指標
    print_cyan("\n生成訓練結果可視化...")
    visualize_training_results(train_losses, val_losses, test_metrics)
    
    # 預測未來股價 - 使用完整數據集
    print_cyan("\n開始預測未來股價...")
    all_sequences = np.concatenate([train_sequences, test_sequences], axis=0)
    predictions = predict_stock_future(model, full_metadata, device, days_ahead=7, sequences=all_sequences)
    
    # 打印預測摘要
    print_prediction_summary(predictions)
    
    # 可視化預測結果
    print_cyan("\n生成預測結果可視化...")
    visualize_stock_predictions(predictions)
    
    print_green("\n🎉 時間分割訓練和預測流程完成！")
    print_green("📁 生成的文件:")
    print_green("  - best_time_split_model.pth (最佳時間分割模型)")
    print_green("  - time_split_model_metrics.json (驗證集和測試集評估指標)")
    print_green("  - predictions_summary.json (預測摘要)")
    print_green("  - comprehensive_training_results.png (訓練結果)")
    print_green("  - stock_future_predictions.png (預測結果)")
    
    return model, full_metadata, val_metrics, test_metrics, predictions

if __name__ == "__main__":
    model, metadata, val_metrics, test_metrics, predictions = train_optimized_model()
    print_green("✅ 時間分割訓練流程完成！")