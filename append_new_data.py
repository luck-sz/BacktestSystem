"""
append_new_data.py — 增量补充数据脚本（修复版）

主要修复：
1. ts_code 推断规则修正：920/8/4 开头 -> .BJ，6/5 开头 -> .SH，其余 -> .SZ
2. 多线程并发（不同于 IP 限制的单线程），适度加速
3. 详细错误日志，便于排查失败原因
"""
import tushare as ts
import pandas as pd
import os
import time
from tqdm import tqdm
import concurrent.futures

# ─── 配置 ───
TOKEN = '8cd72c7ab2a53ee9cdbb8cff0c23778c691e1aecc63d0ad6b692e708'
DATA_DIR = r'all_stock_data'
# 请求日期范围（只补充缺失的交易日）
APPEND_START = "20260228"  # 从 0228 开始确保不遗漏
APPEND_END   = "20260304"  # 今天的日期（如有数据会含 0304）
MAX_WORKERS  = 5           # 并发线程数，TuShare Pro 支持每分钟500次，5线程完全安全
DELAY        = 0.05        # 每个线程内最小延迟（秒）

ts.set_token(TOKEN)
TS_API = ts.pro_api()

# 目标列结构（与原始 CSV 表头保持完全一致）
TARGET_COLUMNS = [
    '股票代码', '名称', '所属行业', '地域', '上市日期', 'TS代码', '交易日期',
    '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '涨跌额', '涨跌幅(%)',
    '成交量(手)', '成交额(千元)', '换手率(%)', '换手率(自由流通股)', '量比',
    '市盈率', '市盈率(TTM)', '市净率', '市销率', '市销率(TTM)',
    '股息率(%)', '股息率(TTM)(%)', '总股本(万股)', '流通股本(万股)',
    '自由流通股本(万股)', '总市值(万元)', '流通市值(万元)',
    '今日涨停价', '今日跌停价', '复权因子'
]

# TuShare daily 列 -> 目标列映射
MAPPING_DAILY = {
    'ts_code':   'TS代码',
    'trade_date':'交易日期',
    'open':      '开盘价',
    'high':      '最高价',
    'low':       '最低价',
    'close':     '收盘价',
    'pre_close': '前收盘价',
    'change':    '涨跌额',
    'pct_chg':   '涨跌幅(%)',
    'vol':       '成交量(手)',
    'amount':    '成交额(千元)',
}

def sym_to_tscode(symbol: str) -> str:
    """★ 修复版：根据股票代码前缀正确推断交易所后缀"""
    if symbol.startswith(('6', '5')):
        return symbol + '.SH'
    elif symbol.startswith(('4', '8', '9')):
        return symbol + '.BJ'   # 北交所 / 三板
    else:
        return symbol + '.SZ'

def get_last_date(file_path: str) -> str:
    """快速读取 CSV 文件末尾，返回最后一条记录的交易日期字符串"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(-2048, 2)
            tail = f.read().decode('utf-8-sig', errors='ignore')
        for line in reversed(tail.splitlines()):
            if line.strip():
                parts = line.split(',')
                # 交易日期在第7列（索引6）
                if len(parts) > 6 and parts[6].strip().isdigit() and len(parts[6].strip()) == 8:
                    return parts[6].strip()
    except Exception:
        pass
    return ''

def process_single(file_path: str) -> str:
    """处理单只股票的增量追加，返回状态字符串"""
    symbol = os.path.basename(file_path)[:6]
    try:
        # 1. 检查末尾日期，已有 APPEND_END 数据则跳过
        last_date = get_last_date(file_path)
        if last_date >= APPEND_END:
            return f'SKIP:{symbol}'

        # 2. 确定实际需要拉取的起始日期（从 last_date 的下一个交易日）
        # 保险起见从 APPEND_START 拉，重复行由检查避免
        ts_code = sym_to_tscode(symbol)

        # 3. 从 TuShare 拉取增量数据
        time.sleep(DELAY)
        df = TS_API.daily(ts_code=ts_code, start_date=APPEND_START, end_date=APPEND_END)
        if df is None or df.empty:
            return f'NODATA:{symbol}'

        # 4. 过滤掉已存在的日期
        if last_date:
            df = df[df['trade_date'] > last_date]
        if df.empty:
            return f'SKIP:{symbol}'

        # 5. 处理列映射
        df.rename(columns=MAPPING_DAILY, inplace=True)
        df['股票代码'] = symbol
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        # 6. 时间正序排列后追加
        df_out = df[TARGET_COLUMNS].sort_values('交易日期')
        df_out.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        return f'OK:{symbol}:{len(df_out)}rows'

    except Exception as e:
        return f'ERROR:{symbol}:{str(e)[:60]}'

def main():
    print("=== 增量数据补充（修复版）===")
    print(f"目标日期范围: {APPEND_START} ~ {APPEND_END}")

    # 扫描所有 CSV 文件
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv') and f[:6].isdigit():
                all_files.append(os.path.join(root, f))
    print(f"共发现 {len(all_files)} 个文件，使用 {MAX_WORKERS} 并发线程开始处理...\n")

    ok = skip = nodata = error = 0
    error_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, fp): fp for fp in all_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc="进度"):
            result = future.result()
            if result.startswith('OK'):
                ok += 1
            elif result.startswith('SKIP'):
                skip += 1
            elif result.startswith('NODATA'):
                nodata += 1
            else:
                error += 1
                error_list.append(result)

    print(f"\n=== 完成 ===")
    print(f"  成功追加: {ok} 只")
    print(f"  已是最新: {skip} 只（无需更新）")
    print(f"  无数据  : {nodata} 只（停牌/退市等）")
    print(f"  异常错误: {error} 只")
    if error_list:
        print("错误样本（前10条）:")
        for e in error_list[:10]:
            print(f"  {e}")

if __name__ == '__main__':
    main()
