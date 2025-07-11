# 股票未来价格预测系统使用指南

## 概述
这个系统可以根据给定的股票代码和目标时间预测未来的股票价格。系统包含三个主要组件：

1. **数据处理** (`megaData_normalizer.py`) - 处理历史数据并创建训练/测试集
2. **模型训练** (`train_tensor.py`) - 训练LSTM模型并进行预测
3. **未来预测** (`predict_future.py`) - 根据股票代码和时间生成未来价格预测

## 使用流程

### 1. 数据处理
首先运行数据处理脚本：
```bash
python megaData_normalizer.py
```

这会创建以下文件：
- `time_normalized_data/train_sequences_TIMESTAMP.npy` - 训练序列
- `time_normalized_data/train_targets_TIMESTAMP.npy` - 训练目标
- `time_normalized_data/test_sequences_TIMESTAMP.npy` - 测试序列
- `time_normalized_data/test_targets_TIMESTAMP.npy` - 测试目标
- `time_normalized_data/metadata_TIMESTAMP.pkl` - 元数据（包含标准化器等）

### 2. 模型训练
运行训练脚本：
```bash
python train_tensor.py
```

这会：
- 加载处理后的数据
- 训练LSTM模型
- 在测试集上评估模型
- 保存训练好的模型 (`stock_prediction_model_TIMESTAMP.h5`)
- 演示未来预测功能

### 3. 未来价格预测
使用训练好的模型进行未来预测：

```python
from predict_future import FuturePredictionGenerator
from tensorflow.keras.models import load_model

# 初始化预测器
predictor = FuturePredictionGenerator()

# 加载元数据
predictor.load_metadata()

# 加载训练好的模型
model = load_model('stock_prediction_model_TIMESTAMP.h5')

# 进行预测
results = predictor.predict_future_prices(
    model=model,
    stock_symbol="AAPL",
    target_datetime="2025-07-11 16:00:00"
)

# 显示结果
predictor.display_predictions(results)
```

## 数据结构说明

### 输入数据
- **序列长度**: 60个时间步（约30分钟间隔）
- **特征数量**: 48个特征（价格、技术指标、时间特征等）
- **预测步数**: 480步（约10个交易日）

### 特征包括：
- 基础价格数据：Open, High, Low, Close, Volume
- 技术指标：RSI, MACD, 移动平均线等
- 时间特征：小时、星期、月份（正弦/余弦编码）
- 衍生特征：价格比率、波动率、成交量指标等

### 预测输出
- 预测的是**相对价格变化率**（相对于当前价格）
- 需要转换为实际价格：`predicted_price = current_price * (1 + predicted_change)`

## 核心功能

### 1. 时间序列预测
- 基于历史60个时间步的数据
- 预测未来480个时间步的价格变化
- 使用LSTM神经网络

### 2. 数据标准化
- 特征和目标都进行了标准化
- 训练时只使用训练集数据进行标准化
- 预测时需要使用相同的标准化器

### 3. 时间意识分割
- 使用时间序列的最后10天作为测试集
- 确保没有未来信息泄露

## 使用示例

### 预测单个股票
```python
# 预测AAPL在特定时间的价格
results = predictor.predict_future_prices(
    model, "AAPL", "2025-07-11 15:30:00"
)

if results:
    print(f"当前价格: ${results['current_price']:.2f}")
    print(f"预测价格范围: ${results['predicted_prices'].min():.2f} - ${results['predicted_prices'].max():.2f}")
```

### 批量预测多个股票
```python
stocks = ["AAPL", "GOOGL", "MSFT"]
target_time = "2025-07-11 16:00:00"

for stock in stocks:
    results = predictor.predict_future_prices(model, stock, target_time)
    if results:
        avg_price = np.mean(results['predicted_prices'])
        print(f"{stock}: ${results['current_price']:.2f} → ${avg_price:.2f}")
```

## 注意事项

1. **数据要求**: 确保股票数据文件存在于`data/feature/`目录中
2. **时间格式**: 使用ISO格式的时间字符串（"YYYY-MM-DD HH:MM:SS"）
3. **模型兼容性**: 使用相同的数据处理流程训练的模型
4. **内存管理**: 大量数据时使用批处理来避免内存问题
5. **预测解释**: 预测结果是概率性的，应谨慎用于实际交易

## 文件结构
```
qlib_test4/
├── megaData_normalizer.py      # 数据处理
├── train_tensor.py             # 模型训练
├── predict_future.py           # 未来预测核心
├── predict_future_example.py   # 使用示例
├── colors.py                   # 颜色输出工具
├── data/feature/               # 原始股票数据
├── time_normalized_data/       # 处理后的数据
└── stock_prediction_model_*.h5 # 训练好的模型
```

## 扩展功能

### 1. 可视化
系统包含可视化功能，可以生成：
- 预测价格vs实际价格的对比图
- 价格变化百分比图表
- 多股票预测比较图

### 2. 评估指标
- MAE (Mean Absolute Error)
- MSE (Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

### 3. 自定义预测
可以修改预测参数：
- 序列长度
- 预测步数
- 特征选择

这个系统提供了完整的股票价格预测流程，从数据处理到模型训练再到未来预测，可以根据实际需求进行调整和扩展。
