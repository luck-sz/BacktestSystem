"""
append_akshare.py — 使用 AKShare 增量补充最新几天数据

逻辑：
1. 扫描所有 CSV，读取最后一行的交易日期
2. 仅拉取该日期之后的数据（20260228 ~ 20260304）
3. 列与原始 CSV 对齐：有数据的列填值，其余字段留空
4. 以追加模式写入文件末尾，不破坏历史数据
"""
import akshare as ak
import pandas as pd
import os
import time
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 配置 ───
DATA_DIR     = r'all_stock_data'
FETCH_START  = '20260228'   # 稍早一天，确保不遗漏
FETCH_END    = '20260304'   # 今天
MAX_WORKERS  = 3            # 并发线程，盘后可稍微高一点
BASE_DELAY   = 0.4          # 每次请求基础延迟(秒)

# 原始 CSV 的 34 列顺序（不可改变）
TARGET_COLUMNS = [
    '股票代码', '名称', '所属行业', '地域', '上市日期', 'TS代码', '交易日期',
    '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '涨跌额', '涨跌幅(%)',
    '成交量(手)', '成交额(千元)', '换手率(%)', '换手率(自由流通股)', '量比',
    '市盈率', '市盈率(TTM)', '市净率', '市销率', '市销率(TTM)',
    '股息率(%)', '股息率(TTM)(%)', '总股本(万股)', '流通股本(万股)',
    '自由流通股本(万股)', '总市值(万元)', '流通市值(万元)',
    '今日涨停价', '今日跌停价', '复权因子'
]

# AKShare stock_zh_a_hist 返回列 -> 目标列 映射
AK_MAPPING = {
    '日期':    '交易日期',
    '开盘':    '开盘价',
    '最高':    '最高价',
    '最低':    '最低价',
    '收盘':    '收盘价',
    '成交量':  '成交量(手)',
    '成交额':  '成交额(千元)',
    '振幅':    '',   # 无对应，丢弃
    '涨跌幅':  '涨跌幅(%)',
    '涨跌额':  '涨跌额',
    '换手率':  '换手率(%)',
}

def get_last_date(file_path: str) -> str:
    """读取 CSV 末尾，返回最后一条记录的交易日期（8 位数字字符串）"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(-2048, 2)
            tail = f.read().decode('utf-8-sig', errors='ignore')
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) > 6:
                date_str = parts[6].strip()
                if date_str.isdigit() and len(date_str) == 8:
                    return date_str
    except Exception:
        pass
    return ''

def process_single(file_path: str) -> str:
    symbol = os.path.basename(file_path)[:6]
    try:
        # 1. 检查已有最新日期，避免重复
        last_date = get_last_date(file_path)
        if last_date >= FETCH_END:
            return f'SKIP:{symbol}'

        # 2. 拉取增量数据（前复权）
        time.sleep(BASE_DELAY + random.uniform(0, 0.3))
        df = ak.stock_zh_a_hist(
            symbol=symbol, period='daily',
            start_date=FETCH_START, end_date=FETCH_END,
            adjust='qfq'
        )
        if df is None or df.empty:
            return f'NODATA:{symbol}'

        # 3. 过滤已有的日期行
        df['日期'] = df['日期'].astype(str).str.replace('-', '')
        if last_date:
            df = df[df['日期'] > last_date]
        if df.empty:
            return f'SKIP:{symbol}'

        # 4. 列映射 & 对齐
        df.rename(columns=AK_MAPPING, inplace=True)
        df['股票代码'] = symbol
        # 其余目标列全部填空
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        df_out = df[TARGET_COLUMNS]
        df_out.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        return f'OK:{symbol}:{len(df_out)}rows'

    except Exception as e:
        return f'ERROR:{symbol}:{str(e)[:80]}'

def main():
    print('=== AKShare 增量补充（盘后模式）===')
    print(f'目标日期范围: {FETCH_START} ~ {FETCH_END}')

    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv') and f[:6].isdigit():
                all_files.append(os.path.join(root, f))

    print(f'共发现 {len(all_files)} 个文件，{MAX_WORKERS} 线程并发开始...\n')

    ok = skip = nodata = error = 0
    error_sample = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, fp): fp for fp in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files), desc='追加进度'):
            r = future.result()
            if r.startswith('OK'):
                ok += 1
            elif r.startswith('SKIP'):
                skip += 1
            elif r.startswith('NODATA'):
                nodata += 1
            else:
                error += 1
                if len(error_sample) < 10:
                    error_sample.append(r)

    print(f'\n=== 完成 ===')
    print(f'  成功追加: {ok} 只')
    print(f'  已是最新: {skip} 只')
    print(f'  无新数据: {nodata} 只（停牌/退市等）')
    print(f'  失败错误: {error} 只')
    if error_sample:
        print('异常样本:')
        for e in error_sample:
            print(f'  {e}')

if __name__ == '__main__':
    main()
