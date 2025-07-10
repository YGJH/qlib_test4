import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import glob
import re
import os
from datetime import datetime, timedelta
import warnings
from colors import *
import gc

warnings.filterwarnings('ignore')

from stock_transformer import StockTransformer

def load_time_aware_data():
    """載入時間感知的數據"""
    data_dir = 'mega_normalized_data'
    
    # 找到最新的檔案
    train_files = glob.glob(os.path.join(data_dir, 'train_sequences_*.npy'))
    if not train_files:
        raise FileNotFoundError("找不到時間感知的訓練數據檔案")
    
    latest_train_file = max(train_files, key=os.path.getctime)
    timestamp_match = re.search(r'_(\d{8}_\d{6})\.npy', latest_train_file)
    timestamp = timestamp_match.group(1)
    
    # 載入訓練數據
    train_sequences = np.load(latest_train_file)
    train_targets = np.load(os.path.join(data_dir, f'train_targets_{timestamp}.npy'))
    
    # 載入測試數據
    test_sequences_file = os.path.join(data_dir, f'test_sequences_{timestamp}.npy')
    test_targets_file = os.path.join(data_dir, f'test_targets_{timestamp}.npy')
    
    if os.path.exists(test_sequences_file):
        test_sequences = np.load(test_sequences_file)
        test_targets = np.load(test_targets_file)
    else:
        test_sequences = np.array([])
        test_targets = np.array([])
    
    # 載入元數據
    with open(os.path.join(data_dir, f'metadata_{timestamp}.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print_green(f"載入時間感知數據:")
    print_green(f"  訓練序列: {train_sequences.shape}")
    print_green(f"  測試序列: {test_sequences.shape}")
    print_green(f"  測試開始日期: {metadata['test_start_date']}")
    
    return {
        'train_sequences': train_sequences,
        'train_targets': train_targets,
        'test_sequences': test_sequences,
        'test_targets': test_targets,
        'metadata': metadata
    }

def create_validation_split(train_sequences, train_targets, train_metadata, val_ratio=0.2):
    """從訓練數據中創建驗證集 - 時間感知"""
    # 確保序列數量與元數據匹配
    min_length = min(len(train_sequences), len(train_targets), len(train_metadata))
    
    if min_length != len(train_sequences):
        print_yellow(f"警告: 序列數量不匹配，調整為 {min_length}")
        train_sequences = train_sequences[:min_length]
        train_targets = train_targets[:min_length]
        train_metadata = train_metadata[:min_length]
    
    # 按時間排序
    sorted_indices = sorted(range(len(train_metadata)), key=lambda i: train_metadata[i]['datetime'])
    
    # 取最後val_ratio的數據作為驗證集
    val_start_idx = int(len(sorted_indices) * (1 - val_ratio))
    
    # 分割索引
    train_indices = sorted_indices[:val_start_idx]
    val_indices = sorted_indices[val_start_idx:]
    
    # 創建分割
    train_seq_split = train_sequences[train_indices]
    train_tar_split = train_targets[train_indices]
    val_seq_split = train_sequences[val_indices]
    val_tar_split = train_targets[val_indices]
    
    print_cyan(f"驗證集分割:")
    print_cyan(f"  總序列數: {len(train_sequences)}")
    print_cyan(f"  訓練: {len(train_indices)} 個序列")
    print_cyan(f"  驗證: {len(val_indices)} 個序列")
    
    return train_seq_split, train_tar_split, val_seq_split, val_tar_split

def evaluate_model(model, data_loader, target_scaler, device, period_name=""):
    """評估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets, batch_stock_ids in data_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_stock_ids = batch_stock_ids.squeeze().to(device, non_blocking=True)
            
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
    
    # 限制數值範圍
    pred_denorm = np.clip(pred_denorm, -1.0, 1.0)
    actual_denorm = np.clip(actual_denorm, -1.0, 1.0)
    
    # 計算指標
    mse = mean_squared_error(actual_denorm.flatten(), pred_denorm.flatten())
    mae = mean_absolute_error(actual_denorm.flatten(), pred_denorm.flatten())
    rmse = np.sqrt(mse)
    
    # 安全的MAPE計算
    actual_abs = np.abs(actual_denorm)
    mask = actual_abs > 1e-6
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual_denorm[mask] - pred_denorm[mask]) / actual_denorm[mask])) * 100
        mape = min(mape, 1000.0)
    else:
        mape = 0.0
    
    # 方向準確率 - 不同時間點
    direction_accuracies = {}
    time_points = [47, 95, 143, 191, 239, 287, -1]  # 1天, 2天, 3天, 4天, 5天, 6天, 7天
    time_labels = ['1天', '2天', '3天', '4天', '5天', '6天', '7天']
    
    for i, (tp, label) in enumerate(zip(time_points, time_labels)):
        if tp == -1:
            tp = pred_denorm.shape[1] - 1
        
        if tp < pred_denorm.shape[1]:
            pred_dir = (pred_denorm[:, tp] > 0).astype(int)
            actual_dir = (actual_denorm[:, tp] > 0).astype(int)
            direction_accuracies[label] = (pred_dir == actual_dir).mean()
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'direction_accuracies': direction_accuracies,
        'predictions': pred_denorm,
        'actuals': actual_denorm,
        'period': period_name
    }


def predict_stock_future(model, metadata, device, days_ahead=7, sequences=None):
    """預測股票未來價格"""
    model.eval()
    
    stock_to_id = metadata['stock_to_id']
    target_scaler = metadata['target_scaler']
    train_metadata = metadata.get('train_metadata', [])
    
    predictions = {}
    
    print_cyan(f"\n預測未來 {days_ahead} 天的股價變化...")
    
    # 如果沒有提供序列，創建一個簡單的預測
    if sequences is None or len(sequences) == 0 or len(train_metadata) == 0:
        print_yellow("沒有足夠的數據進行預測，返回空結果")
        return predictions
    
    with torch.no_grad():
        for stock_symbol, stock_id in stock_to_id.items():
            print_cyan(f"預測股票: {stock_symbol}")
            
            try:
                # 獲取該股票的最新序列
                stock_sequences = []
                stock_metadata = []
                
                for i, meta in enumerate(train_metadata):
                    if i < len(sequences) and meta['stock_id'] == stock_id:
                        stock_sequences.append(sequences[i])
                        stock_metadata.append(meta)
                
                if not stock_sequences:
                    continue
                
                # 使用最新的序列
                latest_sequence = stock_sequences[-1]
                latest_meta = stock_metadata[-1]
                
                # 轉換為tensor
                input_tensor = torch.FloatTensor(latest_sequence).unsqueeze(0).to(device)
                stock_id_tensor = torch.LongTensor([stock_id]).to(device)
                
                # 預測
                outputs = model(input_tensor, stock_id_tensor)
                predicted_changes = outputs['predictions'].cpu().numpy()[0]
                
                # 反標準化
                predicted_changes_denorm = target_scaler.inverse_transform(
                    predicted_changes.reshape(-1, 1)
                ).flatten()
                
                # 限制變化範圍
                predicted_changes_denorm = np.clip(predicted_changes_denorm, -0.1, 0.1)
                
                # 計算未來價格
                current_price = latest_meta.get('current_price', 100.0)  # 默認價格
                future_prices = []
                current_price_temp = current_price
                
                for i, change in enumerate(predicted_changes_denorm):
                    new_price = current_price_temp * (1 + change * 0.05)
                    new_price = max(new_price, current_price * 0.7)
                    new_price = min(new_price, current_price * 1.5)
                    future_prices.append(new_price)
                    current_price_temp = new_price
                
                # 關鍵統計
                price_1d = future_prices[47] if len(future_prices) > 47 else future_prices[-1]
                price_7d = future_prices[-1]
                
                change_1d = ((price_1d - current_price) / current_price) * 100
                change_7d = ((price_7d - current_price) / current_price) * 100
                
                change_1d = np.clip(change_1d, -20, 20)
                change_7d = np.clip(change_7d, -30, 30)
                
                # 波動率
                if len(future_prices) >= 24:
                    price_returns = []
                    for i in range(1, min(24, len(future_prices))):
                        ret = (future_prices[i] - future_prices[i-1]) / future_prices[i-1]
                        price_returns.append(ret)
                    volatility = np.std(price_returns) * 100 * np.sqrt(24)
                    volatility = min(volatility, 50.0)
                else:
                    volatility = 5.0
                
                predictions[stock_symbol] = {
                    'current_price': float(current_price),
                    'price_1d': float(price_1d),
                    'price_7d': float(price_7d),
                    'change_1d': float(change_1d),
                    'change_7d': float(change_7d),
                    'volatility': float(volatility),
                    'trend': 'bullish' if change_7d > 1 else 'bearish' if change_7d < -1 else 'neutral'
                }
                
            except Exception as e:
                print_red(f"預測股票 {stock_symbol} 時出錯: {e}")
                continue
    
    return predictions


def train_time_aware_model():
    """訓練時間感知模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"使用設備: {device}")
    
    try:
        # 載入時間感知數據
        data = load_time_aware_data()
        
        train_sequences = data['train_sequences']
        train_targets = data['train_targets']
        test_sequences = data['test_sequences']
        test_targets = data['test_targets']
        metadata = data['metadata']
        
        # 提取元數據
        stock_to_id = metadata['stock_to_id']
        target_scaler = metadata['target_scaler']
        train_metadata = metadata['train_metadata']
        test_metadata = metadata['test_metadata']
        
        # 創建驗證集
        train_seq_split, train_tar_split, val_seq_split, val_tar_split = create_validation_split(
            train_sequences, train_targets, train_metadata
        )
        
        # 創建股票ID數組
        train_stock_ids = np.array([meta['stock_id'] for meta in train_metadata])
        train_ids_split = train_stock_ids[:len(train_seq_split)]
        val_ids_split = train_stock_ids[len(train_seq_split):]
        
        # 創建數據加載器
        train_dataset = TensorDataset(
            torch.FloatTensor(train_seq_split),
            torch.FloatTensor(train_tar_split),
            torch.LongTensor(train_ids_split)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_seq_split),
            torch.FloatTensor(val_tar_split),
            torch.LongTensor(val_ids_split)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                                  num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                 num_workers=8, pin_memory=True)

        # 創建測試數據加載器
        test_loader = None
        if len(test_sequences) > 0:
            test_stock_ids = np.array([meta['stock_id'] for meta in test_metadata])
            test_dataset = TensorDataset(
                torch.FloatTensor(test_sequences),
                torch.FloatTensor(test_targets),
                torch.LongTensor(test_stock_ids)
            )
            test_loader = DataLoader(test_dataset, batch_size=32, 
                            shuffle=False, num_workers=0, pin_memory=True)
        
        # 創建模型
        model = StockTransformer(
            input_dim=train_sequences.shape[-1],
            num_stocks=len(stock_to_id),
            d_model=256,
            nhead=8,
            num_layers=32,
            dropout=0.1,
            prediction_horizon=train_targets.shape[-1]
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print_magenta(f"模型參數總數: {total_params/1e6:.1f}M")
        
        # 訓練設置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 訓練循環
        num_epochs = 2
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print_green(f"\n開始時間感知訓練 (共 {num_epochs} 個epoch)...")
        
        for epoch in range(num_epochs):
            # 訓練階段
            model.train()
            train_loss = 0
            for batch_features, batch_targets, batch_stock_ids in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                batch_stock_ids = batch_stock_ids.squeeze().to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features, batch_stock_ids)
                loss = criterion(outputs['predictions'], batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 驗證階段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets, batch_stock_ids in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    batch_stock_ids = batch_stock_ids.squeeze().to(device)
                    
                    outputs = model(batch_features, batch_stock_ids)
                    loss = criterion(outputs['predictions'], batch_targets)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, 'best_time_aware_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print_cyan(f"Epoch [{epoch+1}/{num_epochs}]")
                print_cyan(f"  Train Loss: {avg_train_loss:.6f}")
                print_cyan(f"  Val Loss: {avg_val_loss:.6f}")
                print_cyan(f"  Best Val Loss: {best_val_loss:.6f}")
                print_cyan(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
            
            if patience_counter >= patience:
                print_yellow(f"早停在第 {epoch+1} 個epoch")
                break
        
        # 載入最佳模型
        checkpoint = torch.load('best_time_aware_model.pth', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 評估模型
        print_magenta("\n=== 模型評估 ===")
        
        # 驗證集評估
        print_cyan("\n1. 驗證集評估:")
        val_metrics = evaluate_model(model, val_loader, target_scaler, device, "驗證集")
        
        print_blue(f"  RMSE: {val_metrics['rmse']:.4f}")
        print_blue(f"  MAE: {val_metrics['mae']:.4f}")
        print_blue(f"  MAPE: {val_metrics['mape']:.2f}%")
        print_blue("  方向準確率:")
        for period, acc in val_metrics['direction_accuracies'].items():
            print_blue(f"    {period}: {acc:.4f}")
        
        # 測試集評估
        test_metrics = None
        if test_loader is not None:
            print_cyan("\n2. 測試集評估 (未來3天數據):")
            test_metrics = evaluate_model(model, test_loader, target_scaler, device, "測試集")
            
            print_blue(f"  RMSE: {test_metrics['rmse']:.4f}")
            print_blue(f"  MAE: {test_metrics['mae']:.4f}")
            print_blue(f"  MAPE: {test_metrics['mape']:.2f}%")
            print_blue("  方向準確率:")
            for period, acc in test_metrics['direction_accuracies'].items():
                print_blue(f"    {period}: {acc:.4f}")
        
        # 保存評估結果
        results = {
            'validation_metrics': {
                'rmse': val_metrics['rmse'],
                'mae': val_metrics['mae'],
                'mape': val_metrics['mape'],
                'direction_accuracies': val_metrics['direction_accuracies']
            }
        }
        
        if test_metrics:
            results['test_metrics'] = {
                'rmse': test_metrics['rmse'],
                'mae': test_metrics['mae'],
                'mape': test_metrics['mape'],
                'direction_accuracies': test_metrics['direction_accuracies']
            }
        
        # 保存結果
        
        print_green("\n🎉 時間感知模型訓練完成！")
        print_green("📁 生成的文件:")
        print_green("  - best_time_aware_model.pth")
        print_green("  - time_aware_model_results.json")
        
        # 分析結果
        print_cyan("\n=== 結果分析 ===")
        if test_metrics:
            print_cyan("驗證集 vs 測試集比較:")
            print_cyan(f"  RMSE: 驗證集 {val_metrics['rmse']:.4f} vs 測試集 {test_metrics['rmse']:.4f}")
            print_cyan(f"  1天方向準確率: 驗證集 {val_metrics['direction_accuracies']['1天']:.4f} vs 測試集 {test_metrics['direction_accuracies']['1天']:.4f}")
            print_cyan(f"  7天方向準確率: 驗證集 {val_metrics['direction_accuracies']['7天']:.4f} vs 測試集 {test_metrics['direction_accuracies']['7天']:.4f}")
        with open('model_metrics.txt', 'w', encoding='utf-8') as f:
            f.write("=== 模型評估指標 ===\n")
            f.write(f"RMSE: {val_metrics['rmse']:.4f}\n")
            f.write(f"MAE: {val_metrics['mae']:.4f}\n")
            f.write(f"MAPE: {val_metrics['mape']:.2f}%\n")
            f.write("方向準確率:\n")
            for period, acc in val_metrics['direction_accuracies'].items():
                f.write(f"  {period}: {acc:.4f}\n")

        # 預測未來股價
        print_cyan("開始預測未來股價...")
        # 使用驗證集的最後一部分數據進行預測
        predictions = predict_stock_future(model, metadata, device, days_ahead=7, sequences=val_seq_split)
        
        # 保存預測結果
        predictions_summary = {}
        for stock, pred in predictions.items():
            predictions_summary[stock] = {
                'current_price': pred['current_price'],
                'price_1d': pred['price_1d'],
                'change_1d': pred['change_1d'],
                'price_7d': pred['price_7d'],
                'change_7d': pred['change_7d'],
                'volatility': pred['volatility'],
                'trend': pred['trend']
            }
        import json
        with open('predictions_summary.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_summary, f, ensure_ascii=False, indent=4)
        
        print_green("\n🎉 訓練和預測完成！")
        print_green("📁 生成的文件:")
        print_green("  - best_robust_model.pth")
        print_green("  - model_metrics.json")
        print_green("  - predictions_summary.json")
    
        return model, metadata, val_metrics, test_metrics
    
    except Exception as e:
        print_red(f"訓練過程中發生錯誤: {e.with_traceback()}")
        return None, None, None, None
    finally:
        # 清理內存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print_green("開始時間感知模型訓練...")
    result = train_time_aware_model()
    
    if result[0] is not None:
        print_green("✅ 時間感知訓練成功完成！")
        print_green("✅ 已避免look-ahead bias問題")
    else:
        print_red("❌ 訓練失敗")