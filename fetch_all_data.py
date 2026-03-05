import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import concurrent.futures
from tqdm import tqdm

def fetch_single_stock(symbol, start_str, end_str, save_dir):
    """单独下载某只股票的数据并保存为 CSV"""
    file_path = os.path.join(save_dir, f"{symbol}.csv")
    # 支持断点续传：如果文件已经存在，就跳过
    if os.path.exists(file_path):
        return symbol, True
    
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        if not df.empty:
            # 统一字段列名和增加计算字段
            df['昨收盘价'] = df['收盘'] - df['涨跌额']
            
            mapping = {
                '日期': '交易日期', '开盘': '开盘价', '最高': '最高价', '最低': '最低价',
                '收盘': '收盘价', '成交量': '成交量(手)', '涨跌额': '涨跌额'
            }
            df.rename(columns=mapping, inplace=True)
            
            # 统一列顺序
            cols = ['交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量(手)', '昨收盘价', '涨跌额']
            # 注意：此脚本下载时不含“名称”，如果需要可以后续通过名称列表补齐，但为了基础回测够用了
            df = df[[c for c in cols if c in df.columns]]
            
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return symbol, True
    except Exception as e:
        # 可能遇到新股、停牌或者请求错误
        return symbol, False

def fetch_all_a_shares_past_year():
    print("正在获取全市场 A 股股票列表...")
    stock_info = ak.stock_info_a_code_name()
    symbols = stock_info['code'].tolist()
    
    # 过滤掉以 8 和 4 开头的北交所/三板股票 (选择性过滤，这里以常用的沪深A股为主：60, 00, 30, 688)
    # 不过为了满足你“所有”的需求，这里我们不去进行强过滤，全部抓取。
    print(f"共获取到 {len(symbols)} 只股票代码。")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    save_dir = "all_stock_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"数据将保存在当前目录下的 '{save_dir}' 文件夹中。")
    print(f"时间范围：{start_str} 至 {end_str}")
    print("开始多线程下载... (预计需要1~3分钟)")
    
    failed_symbols = []
    
    # 使用 10 个线程加速下载 (不建议开太高以防 IP 被封)
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_stock, sym, start_str, end_str, save_dir): sym for sym in symbols}
        
        # tqdm 打印进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="下载进度"):
            sym, success = future.result()
            if not success:
                failed_symbols.append(sym)
                
    print(f"\n全市场数据下载完成！CSV 稳妥保存在 '{save_dir}' 目录下。")
    if failed_symbols:
        print(f"有 {len(failed_symbols)} 只股票下载失败或无数据（多数为刚上市/尚未上市/长期停牌）。")

if __name__ == "__main__":
    fetch_all_a_shares_past_year()
