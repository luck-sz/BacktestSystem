import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm
import time
import random

# 映射 backtest.py 中的 RENAME_MAP：akshare 的原生列名 -> 所需列名
COL_MAPPING = {
    '日期': '交易日期',
    '开盘': '开盘价',
    '最高': '最高价',
    '最低': '最低价',
    '收盘': '收盘价',
    '成交量': '成交量(手)',
}

def update_single_stock(symbol, name, start_str, end_str, file_path):
    """获取单只股票数据并保存/更新，增加重试机制"""
    retries = 2
    for attempt in range(retries + 1):
        try:
            # 随机短暂延迟，模拟人工请求
            time.sleep(random.uniform(0.1, 0.5))
            
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                    start_date=start_str, end_date=end_str, 
                                    adjust="qfq")
            
            if df.empty:
                return symbol, False
                
            df['名称'] = name
            df['日期'] = df['日期'].str.replace('-', '')
            df.rename(columns=COL_MAPPING, inplace=True)
            
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            return symbol, True
        except Exception as e:
            if attempt < retries:
                time.sleep(random.uniform(1, 3)) # 失败后多等一会儿
                continue
            return symbol, False

def run_update():
    print("=== 开始更新本地股票数据 (温和模式) ===")
    
    # 1. 定位现有文件夹
    data_dir = "all_stock_data"
    existing_files = {}
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".csv") and len(f) == 10: # xxxxxx.csv
                    symbol = f[:6]
                    existing_files[symbol] = os.path.join(root, f)
    
    print(f"DONE: 在本地发现 {len(existing_files)} 个现有数据文件。")

    # 2. 获取股票列表 (如果接口还是封死，就仅用本地列表)
    stock_map = {}
    print("正在尝试获取最新股票名单...")
    try:
        stock_info = ak.stock_zh_a_spot_em()
        if not stock_info.empty:
            stock_map = dict(zip(stock_info['代码'], stock_info['名称']))
            print(f"DONE: 获取到实时列表，共 {len(stock_map)} 只股票。")
    except Exception as e:
        print(f"WARN: 获取网络列表受阻 ({e})，将仅更新本地已有 {len(existing_files)} 只股票。")

    # 合并任务
    symbols_to_fetch = set(existing_files.keys()).union(set(stock_map.keys()))
    if not symbols_to_fetch:
        print("ERROR: 无更新任务，请确认数据目录是否存在。")
        return

    # 3. 准备时间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    print(f"DATE: 更新段: {start_str} 至 {end_str}")

    # 4. 设置安全并发数
    max_workers = 3 # 从 10 降到 3，极显著降低封禁概率
    print(f"START: Starting parallel download (Max Workers: {max_workers})...")
    
    failed = []
    
    def get_save_path(sym):
        return existing_files.get(sym, os.path.join(data_dir, f"{sym}.csv"))

    def get_name(sym):
        return stock_map.get(sym, sym)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 分批提交或直接提交，Executor 会控制并发
        futures = {executor.submit(update_single_stock, sym, get_name(sym), start_str, end_str, get_save_path(sym)): sym for sym in symbols_to_fetch}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols_to_fetch), desc="Progress"):
            sym, success = future.result()
            if not success:
                failed.append(sym)

    print(f"\nALL FINISHED! Success: {len(symbols_to_fetch) - len(failed)} | Failed: {len(failed)}")
    if failed:
        print(f"INFO: {len(failed)} stocks failed. If the IP is blocked, try again in 30 minutes.")

if __name__ == "__main__":
    run_update()
