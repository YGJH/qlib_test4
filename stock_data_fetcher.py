#!/usr/bin/env python3
"""
Stock Data Fetcher - 股票資料抓取腳本
使用最普遍的data science方式抓取股票資訊，interval為1分鐘
獲取盡可能多的股票資訊
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
import time
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

class StockDataFetcher:
    """股票資料抓取器"""
    
    def __init__(self):
        """初始化"""
        self.session = requests.Session()
        self.DATA_DIR = 'data'
        self.FEATURE_DIR = os.path.join(self.DATA_DIR, 'feature')
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        if not os.path.exists(self.FEATURE_DIR):
            os.makedirs(self.FEATURE_DIR)

    def get_popular_stocks(self) -> List[str]:
        """獲取熱門股票代碼"""
        with open('symbols.txt', 'r') as f:
            custom_stocks = f.read().strip().split('\n')
        return custom_stocks

    def _fetch_data_in_chunks(self, stock, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """分塊抓取資料以避免API限制"""
        all_data = []
        current_start = start_date
        while current_start < end_date:
            # yfinance's end date is exclusive, so we can aim for 60 days.
            current_end = current_start + timedelta(days=59)
            if current_end > end_date:
                current_end = end_date
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')
            
            logger.info(f"  正在抓取 {stock.ticker} 從 {start_str} 到 {end_str} 的資料...")
            
            try:
                hist_data = stock.history(start=start_str, end=end_str, interval="30m")
                if not hist_data.empty:
                    all_data.append(hist_data)
            except Exception as e:
                logger.error(f"抓取 {stock.ticker} 在 {start_str} 到 {end_str} 區間時失敗: {e}")

            current_start = current_end + timedelta(days=1)
            time.sleep(0.2)  # Be respectful to the API provider
            
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data)

    def fetch_realtime_data(self, symbols: List[str], start_date: str, end_date: str, 
                           interval: str = "30m") -> Dict[str, pd.DataFrame]:
        """
        抓取即時股票資料，並與現有資料合併
        
        Args:
            symbols: 股票代碼列表
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            interval: 資料間隔
        
        Returns:
            包含所有股票資料的字典
        """
        stock_data = {}
        failed_symbols = []
        
        logger.info(f"開始處理 {len(symbols)} 支股票的資料，目標區間: {start_date} to {end_date}...")
        
        user_start_date = pd.to_datetime(start_date)
        user_end_date = pd.to_datetime(end_date)

        for symbol in symbols:
            try:
                logger.info(f"正在處理 {symbol}...")
                stock = yf.Ticker(symbol)
                
                fetch_start_date = user_start_date
                existing_data = None
                symbol_csv_path = os.path.join(self.FEATURE_DIR, f'{symbol.lower()}.csv')

                if os.path.exists(symbol_csv_path):
                    logger.info(f"找到 {symbol} 的現有資料，正在讀取...")
                    existing_data = pd.read_csv(symbol_csv_path, index_col=0, parse_dates=True)
                    if not existing_data.empty:
                        existing_data.index = pd.to_datetime(existing_data.index).tz_localize(None)
                        last_timestamp = existing_data.index.max()
                        logger.info(f"  最新資料時間戳: {last_timestamp}")
                        fetch_start_date = last_timestamp + timedelta(minutes=30) # Start from next interval

                if fetch_start_date >= user_end_date:
                    logger.info(f"{symbol}: 資料已是最新 ({existing_data.index.max().strftime('%Y-%m-%d')})，無需下載。")
                    hist_data = existing_data
                else:
                    logger.info(f"{symbol}: 將從 {fetch_start_date.strftime('%Y-%m-%d %H:%M:%S')} 開始抓取新資料...")
                    
                    total_days = (user_end_date - fetch_start_date).days
                    if interval == "30m" and total_days > 60:
                        logger.info(f"時間範圍 ({total_days} 天) 超過60天，將分塊抓取。")
                        new_data = self._fetch_data_in_chunks(stock, fetch_start_date, user_end_date, interval)
                    else:
                        new_data = stock.history(start=fetch_start_date, end=user_end_date, interval=interval)

                    if new_data.empty:
                        logger.warning(f"{symbol}: 未能獲取新的歷史資料。")
                        hist_data = existing_data if existing_data is not None else pd.DataFrame()
                    else:
                        new_data.index = pd.to_datetime(new_data.index).tz_convert('UTC').tz_localize(None)
                        if existing_data is not None:
                            hist_data = pd.concat([existing_data, new_data])
                            hist_data = hist_data[~hist_data.index.duplicated(keep='last')]
                            hist_data.sort_index(inplace=True)
                        else:
                            hist_data = new_data
                
                if hist_data.empty:
                    logger.warning(f"{symbol}: 最終無可用資料。")
                    failed_symbols.append(symbol)
                    continue
                
                info = stock.info
                hist_data = self.calculate_technical_indicators(hist_data)
                
                stock_data[symbol] = {
                    'price_data': hist_data,
                    'info': info,
                    'last_update': datetime.now()
                }
                
                logger.info(f"{symbol}: 成功處理，總共有 {len(hist_data)} 筆資料。")
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"{symbol}: 處理失敗 - {str(e)}", exc_info=True)
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"以下股票處理失敗: {failed_symbols}")
        
        logger.info(f"成功處理 {len(stock_data)} 支股票的資料。")
        return stock_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        if df.empty or 'Close' not in df.columns:
            return df
        
        try:
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            df['BB_Middle'] = df['MA20']
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            df['Price_Change_Pct'] = df['Close'].pct_change() * 100
            
            if 'Volume' in df.columns:
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
        except Exception as e:
            logger.warning(f"計算技術指標時發生錯誤: {str(e)}")
        
        return df
    
    def get_market_summary(self) -> Dict:
        """獲取市場總覽"""
        # This function can be simplified or removed if not central to the new logic
        return {}

    def save_data(self, stock_data: Dict, filename: str = None):
        """儲存資料"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"stock_data_{timestamp}"
        
        try:
            # 1. 儲存價格資料為CSV到 feature 資料夾
            for symbol, data in stock_data.items():
                price_df = data['price_data']
                if not price_df.empty:
                    csv_filename = Path(self.FEATURE_DIR) / f"{symbol.lower()}.csv"
                    price_df.to_csv(csv_filename)
                    logger.info(f"已更新/儲存 {symbol} 價格資料至 {csv_filename}")
            
            # 2. 儲存完整資料為pickle
            pickle_filename = Path(self.DATA_DIR) / f"{filename}_complete.pkl"
            # For pickle, let's not store the full history, just the latest update info
            # to avoid huge files. The CSV is the source of truth for prices.
            summary_data_for_pickle = {s: {k: v for k, v in d.items() if k != 'price_data'} for s, d in stock_data.items()}
            pd.to_pickle(summary_data_for_pickle, pickle_filename)
            logger.info(f"已儲存中繼資料至 {pickle_filename}")
            
            # 3. 儲存摘要資訊為JSON
            summary = {}
            for symbol, data in stock_data.items():
                try:
                    latest_price = data['price_data'].iloc[-1] if not data['price_data'].empty else None
                    summary[symbol] = {
                        'latest_price': float(latest_price['Close']) if latest_price is not None else None,
                        'volume': float(latest_price['Volume']) if latest_price is not None and 'Volume' in latest_price else None,
                        'data_points': len(data['price_data']),
                        'last_update': data['last_update'].isoformat()
                    }
                except Exception:
                    summary[symbol] = {'error': 'Failed to process data'}

            json_filename = Path(self.DATA_DIR) / f"{filename}_summary.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"已儲存摘要資訊至 {json_filename}")
            
        except Exception as e:
            logger.error(f"儲存資料失敗: {str(e)}")
    
    def create_visualization(self, stock_data: Dict, symbols: List[str] = None):
        """創建視覺化圖表"""
        if not symbols:
            symbols = list(stock_data.keys())[:6]
        
        try:
            font_path = Path(__file__).parent / "ttc" / "GenKiGothic2JP-B-03.ttf"
            if font_path.exists():
                from matplotlib import font_manager
                font_manager.fontManager.addfont(str(font_path))
                prop = font_manager.FontProperties(fname=str(font_path))
                plt.rcParams["font.family"] = [prop.get_name()]
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('股票資料分析儀表板', fontsize=16, fontweight='bold')
            
            for i, symbol in enumerate(symbols[:6]):
                ax = axes.flatten()[i]
                if symbol in stock_data and not stock_data[symbol]['price_data'].empty:
                    df = stock_data[symbol]['price_data']
                    ax.plot(df.index, df['Close'], label='收盤價', linewidth=1)
                    if 'MA20' in df.columns:
                        ax.plot(df.index, df['MA20'], label='20日均線', alpha=0.7, linestyle='--')
                    ax.set_title(f"{symbol} ({stock_data[symbol]['info'].get('shortName', 'N/A')})", fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, f'{symbol}\n無資料', ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'stock_analysis_{timestamp}.png', dpi=300)
            logger.info(f"已儲存分析圖表至 stock_analysis_{timestamp}.png")
            plt.show()
            
        except Exception as e:
            logger.error(f"創建視覺化失敗: {str(e)}")
    
    def generate_report(self, stock_data: Dict) -> str:
        """生成分析報告"""
        report = [f"股票資料分析報告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60]
        
        for symbol, data in stock_data.items():
            try:
                df = data['price_data']
                if df.empty: continue
                latest = df.iloc[-1]
                info = data.get('info', {})
                report.append(f"\n代碼: {symbol} ({info.get('longName', 'N/A')})")
                report.append(f"  最新價格: ${latest['Close']:.2f}")
                if 'Price_Change_Pct' in df.columns:
                    report.append(f"  最近變動: {latest.get('Price_Change_Pct', 0):.2f}%")
                if 'RSI' in df.columns and not pd.isna(latest['RSI']):
                    rsi_status = "超買" if latest['RSI'] > 70 else "超賣" if latest['RSI'] < 30 else "中性"
                    report.append(f"  RSI (14): {latest['RSI']:.2f} ({rsi_status})")
            except Exception as e:
                report.append(f"\n分析 {symbol} 失敗: {e}")
        
        report_text = "\n".join(report)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'stock_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        return report_text

def main():
    """主函數"""
    print("=== 股票資料抓取與分析腳本 ===")
    fetcher = StockDataFetcher()
    
    try:
        stocks = fetcher.get_popular_stocks()
        if not stocks:
            raise ValueError("未提供任何股票代碼！請檢查 symbols.txt 檔案。")
        
        # 定義要抓取的日期範圍
        end_date = datetime.now()
        start_date = end_date - timedelta(days=59) # 預設抓取最近兩年的資料

        print(f"\n將更新/抓取 {len(stocks)} 支股票的資料...")
        print(f"時間範圍: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        stock_data = fetcher.fetch_realtime_data(
            symbols=stocks,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval="30m"
        )
        
        if not stock_data:
            print("未能獲取或處理任何股票資料！")
            return

        print(f"\n成功處理 {len(stock_data)} 支股票。")
        
        print("\n正在儲存資料...")
        fetcher.save_data(stock_data)
        
        print("\n生成分析報告...")
        report = fetcher.generate_report(stock_data)
        print(report)
        
        try:
            print("\n創建視覺化圖表...")
            fetcher.create_visualization(stock_data)
        except Exception as e:
            print(f"視覺化創建失敗: {e}")
            
        print("\n=== 任務完成 ===")

    except Exception as e:
        logger.error(f"主程序發生錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    main()