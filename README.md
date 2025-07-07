# 股票資料抓取器 Stock Data Fetcher

這是一個專為data scientist設計的股票資料抓取工具，使用最普遍的Python套件和方法來獲取股票的即時資料，支援1分鐘間隔的高頻資料抓取。

## 功能特色 Features

### 📊 資料抓取功能
- **多市場支援**: 美股、台股、加密貨幣
- **高頻資料**: 支援1分鐘間隔的即時資料
- **豐富資訊**: 價格、成交量、技術指標、財務資料、選擇權資料
- **熱門股票**: 預設包含熱門股票清單

### 📈 技術分析指標
- 移動平均線 (MA5, MA10, MA20)
- 相對強弱指標 (RSI)
- MACD指標
- 布林通道 (Bollinger Bands)
- 成交量分析
- 波動率計算

### 💾 資料儲存格式
- **CSV**: 價格資料，方便Excel開啟
- **Pickle**: 完整資料結構
- **JSON**: 摘要資訊
- **TXT**: 分析報告

### 📊 視覺化功能
- 價格走勢圖
- 技術指標圖表
- 多股票對比圖
- 儀表板式總覽

## 安裝說明 Installation

### 1. 安裝Python套件
```bash
pip install -r requirements.txt
```

### 2. 主要套件說明
- `yfinance`: 最受歡迎的股票資料API
- `pandas`: 資料處理和分析
- `matplotlib`: 資料視覺化
- `numpy`: 數值計算
- `seaborn`: 進階圖表

## 使用方法 Usage

### 快速開始
```bash
python quick_example.py
```

### 完整功能
```bash
python stock_data_fetcher.py
```

### 自定義股票
執行程式後，可以輸入自己想要的股票代碼：
```
請輸入要抓取的股票代碼 (用逗號分隔): AAPL,MSFT,GOOGL,2330.TW
```

## 程式碼範例 Code Examples

### 基本使用
```python
from stock_data_fetcher import StockDataFetcher

# 創建抓取器
fetcher = StockDataFetcher()

# 抓取資料
symbols = ['AAPL', 'MSFT', '2330.TW']
data = fetcher.fetch_realtime_data(symbols, period="1d", interval="1m")

# 儲存資料
fetcher.save_data(data)
```

### 進階功能
```python
# 獲取市場總覽
market_summary = fetcher.get_market_summary()

# 生成分析報告
report = fetcher.generate_report(data)

# 創建視覺化
fetcher.create_visualization(data)
```

## 支援的股票代碼格式 Supported Ticker Formats

### 美股 US Stocks
- `AAPL` - Apple Inc.
- `MSFT` - Microsoft Corporation
- `GOOGL` - Alphabet Inc.
- `TSLA` - Tesla Inc.

### 台股 Taiwan Stocks
- `2330.TW` - 台積電
- `2317.TW` - 鴻海
- `2454.TW` - 聯發科

### 加密貨幣 Cryptocurrencies
- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum
- `BNB-USD` - Binance Coin

### 指數 Indices
- `^GSPC` - S&P 500
- `^DJI` - Dow Jones
- `^TWII` - 台灣加權指數

## 參數設定 Parameters

### 時間範圍 Period
- `1d` - 1天
- `5d` - 5天
- `1mo` - 1個月
- `3mo` - 3個月
- `6mo` - 6個月
- `1y` - 1年
- `2y` - 2年
- `5y` - 5年
- `10y` - 10年
- `ytd` - 年初至今
- `max` - 所有可用資料

### 資料間隔 Interval
- `1m` - 1分鐘 (最多7天資料)
- `2m` - 2分鐘
- `5m` - 5分鐘
- `15m` - 15分鐘
- `30m` - 30分鐘
- `1h` - 1小時
- `1d` - 1天
- `1wk` - 1週
- `1mo` - 1個月

## 輸出檔案說明 Output Files

執行後會產生以下檔案：

1. **`[timestamp]_[symbol]_prices.csv`** - 個別股票的價格資料
2. **`[timestamp]_complete.pkl`** - 完整的資料結構（包含所有資訊）
3. **`[timestamp]_summary.json`** - 摘要資訊
4. **`stock_report_[timestamp].txt`** - 分析報告
5. **`stock_analysis_[timestamp].png`** - 視覺化圖表

## 常見問題 FAQ

### Q: 為什麼有些股票抓不到資料？
A: 可能原因：
- 股票代碼格式錯誤
- 該股票在指定時間無交易
- API限制或網路問題

### Q: 1分鐘資料的限制？
A: Yahoo Finance對1分鐘資料有限制，通常只能取得最近7天的資料。

### Q: 如何提高抓取成功率？
A: 
- 使用正確的股票代碼格式
- 避免頻繁請求（程式已內建延遲）
- 檢查網路連線

### Q: 資料準確度如何？
A: 資料來源為Yahoo Finance，一般用於研究和分析足夠準確，但不建議用於實際交易決策。

## 技術細節 Technical Details

### 使用的API
- **Yahoo Finance API** (透過yfinance套件)
- 免費使用，無需API Key
- 支援全球主要交易所

### 資料處理流程
1. 抓取原始資料
2. 計算技術指標
3. 資料清理和驗證
4. 多格式儲存
5. 生成分析報告

### 錯誤處理
- 自動重試機制
- 詳細的錯誤記錄
- 失敗股票清單追蹤

## 擴展建議 Extensions

### 可以加入的功能
1. **更多技術指標**: KD指標、威廉指標等
2. **預警系統**: 價格突破、成交量異常等
3. **資料庫儲存**: PostgreSQL、MongoDB等
4. **即時推送**: Telegram、Email通知
5. **Web介面**: Flask/Django網頁版
6. **API服務**: RESTful API提供資料

### 進階分析
1. **機器學習預測**: 價格趨勢預測
2. **情緒分析**: 新聞、社群媒體情緒
3. **關聯性分析**: 股票間相關性
4. **投資組合優化**: 現代投資組合理論

## 授權 License

此專案為開源專案，歡迎使用和修改。請注意：
- 僅供學習和研究使用
- 投資有風險，請謹慎決策
- 遵守相關法規和API使用條款

## 聯絡資訊 Contact

如有問題或建議，歡迎提出 Issue 或 Pull Request。

---

**免責聲明**: 本工具僅供教育和研究用途，不構成投資建議。投資有風險，請謹慎評估。
