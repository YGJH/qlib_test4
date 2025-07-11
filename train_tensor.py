import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 加载数据
def load_processed_data(data_dir, timestamp):
    train_sequences = np.load(f'{data_dir}/train_sequences_{timestamp}.npy')
    train_targets = np.load(f'{data_dir}/train_targets_{timestamp}.npy')
    test_sequences = np.load(f'{data_dir}/test_sequences_{timestamp}.npy')
    test_targets = np.load(f'{data_dir}/test_targets_{timestamp}.npy')
    
    with open(f'{data_dir}/metadata_{timestamp}.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return train_sequences, train_targets, test_sequences, test_targets, metadata


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
    train_sequences, train_targets, test_sequences, test_targets, metadata = load_processed_data('time_normalized_data', '20250711_XXXXXX')

    input_shape = (train_sequences.shape[1], train_sequences.shape[2])  # (60, 48)
    prediction_steps = train_targets.shape[1]  # 预测步数

    model = build_model(input_shape, prediction_steps)
    # 训练模型
    history = model.fit(
        train_sequences, train_targets,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    # 评估模型
    test_loss, test_mae = model.evaluate(test_sequences, test_targets)
    print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

        # 进行预测
    predictions = model.predict(test_sequences)

    # 反标准化（重要！）
    target_scaler = metadata['target_scaler']
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    test_targets_original = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).reshape(test_targets.shape)

    # 转换回实际价格
    test_metadata = metadata['test_metadata']
    for i, meta in enumerate(test_metadata):
        current_price = meta['current_price']
        # 预测的是相对变化率，转换回实际价格
        predicted_prices = current_price * (1 + predictions_original[i])
        actual_prices = current_price * (1 + test_targets_original[i])
        
        print(f"Stock: {meta['stock_symbol']}")
        print(f"Current Price: {current_price}")
        print(f"Predicted Prices: {predicted_prices}")
        print(f"Actual Prices: {actual_prices}")

if __name__ == "__main__":
    main()