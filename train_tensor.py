import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import glob
from colors import *
def load_time_normalized_data():
    """Load time-normalized data - compatible with main.py"""
    data_dir = 'time_normalized_data'
    import glob
    import os
    import re
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

def visualize_predictions(predictions_original, test_targets_original, test_metadata, num_samples=10):
    """Visualize predictions vs actual values for a few samples"""
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Stock Price Predictions vs Actual Values', fontsize=16)
    
    # Select random samples to visualize
    sample_indices = np.random.choice(len(test_metadata), min(num_samples, len(test_metadata)), replace=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        row = idx // 5
        col = idx % 5
        
        meta = test_metadata[sample_idx]
        current_price = meta['current_price']
        
        # Convert relative changes back to actual prices
        predicted_prices = current_price * (1 + predictions_original[sample_idx])
        actual_prices = current_price * (1 + test_targets_original[sample_idx])
        
        # Create time steps for x-axis
        time_steps = range(len(predicted_prices))
        
        # Plot
        ax = axes[row, col]
        ax.plot(time_steps, predicted_prices, 'b-', label='Predicted', linewidth=2)
        ax.plot(time_steps, actual_prices, 'r-', label='Actual', linewidth=2)
        ax.axhline(y=current_price, color='g', linestyle='--', alpha=0.7, label='Current Price')
        
        ax.set_title(f'{meta["stock_symbol"]} (${current_price:.2f})', fontsize=10)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price ($)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(sample_indices), 10):
        row = idx // 5
        col = idx % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('stock_predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("\n" + "="*50)
    print("PREDICTION STATISTICS")
    print("="*50)
    
    for sample_idx in sample_indices[:5]:  # Show stats for first 5 samples
        meta = test_metadata[sample_idx]
        current_price = meta['current_price']
        
        predicted_prices = current_price * (1 + predictions_original[sample_idx])
        actual_prices = current_price * (1 + test_targets_original[sample_idx])
        
        # Calculate metrics
        mse = np.mean((predicted_prices - actual_prices) ** 2)
        mae = np.mean(np.abs(predicted_prices - actual_prices))
        mape = np.mean(np.abs((predicted_prices - actual_prices) / actual_prices)) * 100
        
        print(f"\nStock: {meta['stock_symbol']}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
        print(f"Actual Range: ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")


def build_model(input_shape, prediction_steps):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(prediction_steps, activation='linear')  # 预测未来多个时间步
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def main():
    # 使用示例
    data_dict = load_time_normalized_data()

    train_sequences = data_dict['train_sequences']
    test_sequences = data_dict['test_sequences']
    train_targets = data_dict['train_targets']
    test_targets = data_dict['test_targets']
    train_metadata = data_dict['train_metadata']
    test_metadata = data_dict['test_metadata']
    full_metadata = data_dict['full_metadata']
    
    print_green(f"Data shapes:")
    print_green(f"  Train sequences: {train_sequences.shape}")
    print_green(f"  Train targets: {train_targets.shape}")
    print_green(f"  Test sequences: {test_sequences.shape}")
    print_green(f"  Test targets: {test_targets.shape}")
    if test_metadata:
        print_green(f"  First test sample: {test_metadata[0]['stock_symbol']} at {test_metadata[0]['datetime']}")
        print_green(f"  Test samples: {len(test_metadata)}")
    
    print_green(f"  Test period: {full_metadata.get('test_start_date', 'N/A')}")
    input_shape = (train_sequences.shape[1], train_sequences.shape[2])  # (60, 48)
    prediction_steps = train_targets.shape[1]  # 预测步数

    model = build_model(input_shape, prediction_steps)
    
    # 训练模型
    print_green("\nTraining model...")
    history = model.fit(
        train_sequences, train_targets,
        epochs=50,  # Reduced epochs for faster training
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 评估模型
    test_loss, test_mae = model.evaluate(test_sequences, test_targets)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

    # 进行预测
    print("\nMaking predictions...")
    
    predictions = model.predict(test_sequences)

    # 反标准化（重要！）
    target_scaler = full_metadata['target_scaler']
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    test_targets_original = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).reshape(test_targets.shape)

    # 确保索引不会越界
    max_samples = min(len(test_metadata), len(predictions_original), len(test_targets_original))
    
    print_green(f"\nProcessing {max_samples} samples for visualization...")
    
    # 可视化预测结果
    try:
        visualize_predictions(
            predictions_original[:max_samples], 
            test_targets_original[:max_samples], 
            test_metadata[:max_samples], 
            num_samples=min(10, max_samples)
        )
    except Exception as e:
        print_red(f"Visualization failed: {e}")
        print_yellow("Falling back to simple text output...")
        
        # 简单的文本输出作为后备
        for i in range(min(5, max_samples)):
            meta = test_metadata[i]
            current_price = meta['current_price']
            predicted_prices = current_price * (1 + predictions_original[i])
            actual_prices = current_price * (1 + test_targets_original[i])
            
            print_green(f"\nStock: {meta['stock_symbol']}")
            print_green(f"Current Price: ${current_price:.2f}")
            print_green(f"Predicted Range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
            print_green(f"Actual Range: ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")
    
    # 保存训练好的模型
    model_filename = f"stock_prediction_model_{data_dict['full_metadata']['timestamp']}.h5"
    model.save(model_filename)
    print_green(f"\nModel saved as: {model_filename}")
    
    # 示例：使用模型进行未来预测
    print_green("\n" + "="*80)
    print_green("FUTURE PREDICTION EXAMPLE")
    print_green("="*80)
    
    # 导入我们创建的未来预测模块
    try:
        from predict_future import FuturePredictionGenerator
        
        # 初始化预测器
        predictor = FuturePredictionGenerator()
        
        # 加载元数据
        if predictor.load_metadata():
            # 获取一个可用的股票
            available_stocks = list(predictor.stock_to_id.keys())
            if available_stocks:
                stock_symbol = available_stocks[0]  # 使用第一个可用股票
                target_datetime = "2025-07-11 16:00:00"  # 目标预测时间
                
                print_cyan(f"Demonstrating future prediction for {stock_symbol.upper()}...")
                
                # 生成未来预测序列
                prediction_info = predictor.generate_future_sequence(stock_symbol, target_datetime)
                
                if prediction_info is not None:
                    # 使用训练好的模型进行预测
                    future_predictions = model.predict(prediction_info['sequence'], verbose=0)
                    
                    # 反标准化
                    future_predictions_original = target_scaler.inverse_transform(
                        future_predictions.reshape(-1, 1)
                    ).reshape(future_predictions.shape)
                    
                    # 转换为实际价格
                    current_price = prediction_info['current_price']
                    predicted_prices = current_price * (1 + future_predictions_original[0])
                    
                    # 显示前10个预测
                    print_green(f"\nFuture price predictions for {stock_symbol.upper()}:")
                    print_green(f"Current price: ${current_price:.2f}")
                    print_green(f"Prediction time: {target_datetime}")
                    print_green(f"First 10 predicted prices:")
                    
                    for i in range(min(10, len(predicted_prices))):
                        change_pct = ((predicted_prices[i] - current_price) / current_price) * 100
                        color_func = print_green if change_pct >= 0 else print_red
                        color_func(f"  Step {i+1}: ${predicted_prices[i]:.2f} ({change_pct:+.2f}%)")
                    
                    print_cyan(f"\nGenerated {len(predicted_prices)} future price predictions!")
                    
                    # 保存预测结果
                    results = {
                        'stock_symbol': stock_symbol,
                        'current_price': current_price,
                        'predicted_prices': predicted_prices,
                        'prediction_changes': future_predictions_original[0],
                        'target_datetime': target_datetime
                    }
                    
                    predictor.save_predictions(results)
                    
                else:
                    print_red("Failed to generate future prediction sequence")
            else:
                print_red("No available stocks for prediction")
        else:
            print_red("Failed to load metadata for future prediction")
            
    except ImportError:
        print_yellow("predict_future module not found. Future prediction example skipped.")
    except Exception as e:
        print_red(f"Future prediction example failed: {e}")
        
    print_green("\n" + "="*80)
    print_green("TRAINING AND PREDICTION COMPLETED!")
    print_green("="*80)

if __name__ == "__main__":
    main()