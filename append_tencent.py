"""
append_tencent_v3.py — 增量补充数据（腾讯接口，插入头部，补齐静态字段）

规则：
1. 从文件现有数据提取静态字段（名称/行业/地域/上市日期/TS代码）
2. 从腾讯接口获取最新数据（从 last_date+1 至今）
3. 新行插到文件头部（表头下方），整体保持时间倒序
4. 所有静态字段直接复用原始行数据，不留空白
"""
import requests
import json
import pandas as pd
import os
import time
import random
from tqdm import tqdm

# ─── 配置 ───
DATA_DIR    = r'all_stock_data'
FETCH_END   = '20260304'   # 今天（含）
FETCH_N     = 10           # 每次取最新N条（覆盖最近约2周，足够）
DELAY_MIN   = 0.2
DELAY_MAX   = 0.5
MAX_RETRY   = 3

# 34列完整表头
TARGET_COLUMNS = [
    '股票代码', '名称', '所属行业', '地域', '上市日期', 'TS代码', '交易日期',
    '开盘价', '最高价', '最低价', '收盘价', '前收盘价', '涨跌额', '涨跌幅(%)',
    '成交量(手)', '成交额(千元)', '换手率(%)', '换手率(自由流通股)', '量比',
    '市盈率', '市盈率(TTM)', '市净率', '市销率', '市销率(TTM)',
    '股息率(%)', '股息率(TTM)(%)', '总股本(万股)', '流通股本(万股)',
    '自由流通股本(万股)', '总市值(万元)', '流通市值(万元)',
    '今日涨停价', '今日跌停价', '复权因子'
]

# 静态字段（不会随交易日变化）
STATIC_COLS = ['名称', '所属行业', '地域', '上市日期', 'TS代码']

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Referer":    "https://gu.qq.com/",
})


def sym_to_qq(symbol: str) -> str:
    if symbol.startswith(('6', '5')):
        return 'sh' + symbol
    elif symbol.startswith(('4', '8', '9')):
        return 'bj' + symbol
    else:
        return 'sz' + symbol


def fetch_latest(symbol: str) -> pd.DataFrame | None:
    """从腾讯接口获取最新N条前复权日K"""
    qq_code = sym_to_qq(symbol)
    url = "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
    params = {
        "_var": f"k_{qq_code}",
        "param": f"{qq_code},day,,,{FETCH_N},qfq",
    }
    for attempt in range(MAX_RETRY):
        try:
            r = SESSION.get(url, params=params, timeout=10)
            text = r.text
            if '=' not in text:
                return None
            json_str = text[text.index('=') + 1:]
            data = json.loads(json_str)
            stock_data = data.get('data', {}).get(qq_code, {})
            klines = stock_data.get('qfqday', stock_data.get('day', []))
            if not klines:
                return None
            records = []
            for k in klines:
                # 腾讯格式: [日期, 开, 收, 高, 低, 成交量, {}, 换手率?, 成交额?, ...]
                date_str = str(k[0]).replace('-', '')
                records.append({
                    '交易日期':    date_str,
                    '开盘价':      k[1],
                    '收盘价':      k[2],
                    '最高价':      k[3],
                    '最低价':      k[4],
                    '成交量(手)':  k[5],
                    '成交额(千元)': k[8] if len(k) > 8 else '',
                })
            return pd.DataFrame(records)
        except Exception:
            if attempt < MAX_RETRY - 1:
                time.sleep(random.uniform(2, 5))
    return None


def get_file_meta(file_path: str) -> dict:
    """
    读取 CSV 文件：
    - 返回 last_date（第一行数据的交易日期，即当前最新日期，因为是倒序）
    - 返回 static_info（从已有数据提取的静态字段值）
    """
    try:
        # 只读前几行，获取静态字段和最新日期
        df_head = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig', dtype=str)
        
        # 最新日期：第一行数据的交易日期（文件是倒序的）
        last_date = ''
        if '交易日期' in df_head.columns and not df_head.empty:
            last_date = str(df_head['交易日期'].iloc[0]).strip()
        
        # 静态字段：从第一行有效数据提取
        static_info = {}
        for col in STATIC_COLS:
            if col in df_head.columns:
                val = df_head[col].iloc[0] if not df_head.empty else ''
                static_info[col] = '' if pd.isna(val) else str(val)
        
        return {'last_date': last_date, 'static': static_info}
    except Exception:
        return {'last_date': '', 'static': {}}


def process_file(file_path: str):
    symbol = os.path.basename(file_path)[:6]

    # 1. 读取文件元信息
    meta = get_file_meta(file_path)
    last_date = meta['last_date']
    static_info = meta['static']

    # 如果已经有最新数据，跳过
    if last_date >= FETCH_END:
        return 'skip', symbol

    # 2. 从腾讯接口拉取最新数据
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    df_new = fetch_latest(symbol)

    if df_new is None or df_new.empty:
        return 'nodata', symbol

    # 3. 只保留 last_date 之后的新数据，且不超过 FETCH_END
    if last_date:
        df_new = df_new[df_new['交易日期'] > last_date]
    df_new = df_new[df_new['交易日期'] <= FETCH_END]

    if df_new.empty:
        return 'skip', symbol

    # 4. 填充静态字段（直接复用原文件中的值，不留空白）
    df_new['股票代码'] = symbol
    for col in STATIC_COLS:
        df_new[col] = static_info.get(col, '')

    # 补齐所有目标列（其他动态列暂时留空）
    for col in TARGET_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ''

    # 5. 新行按时间"倒序"排列（最新在最前）
    df_new = df_new[TARGET_COLUMNS].sort_values('交易日期', ascending=False)
    new_rows_count = len(df_new)

    # 6. 读取原始文件（含表头），将新行插入在表头之后
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            original_content = f.read()

        # 分离表头和数据
        lines = original_content.splitlines(keepends=True)
        header_line = lines[0]
        data_lines = lines[1:]

        # 将新行转为 CSV 文本
        new_csv = df_new.to_csv(index=False, header=False, encoding='utf-8-sig')

        # 重新组合：表头 + 新行 + 原始数据行
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(header_line)
            f.write(new_csv)
            f.writelines(data_lines)

    except Exception as e:
        return 'error', f"{symbol}:{str(e)[:60]}"

    return 'ok', f"{symbol}(+{new_rows_count}行)"


def main():
    print("=== 腾讯行情增量补充（头部插入 + 静态字段补齐）===")
    print(f"补充至: {FETCH_END}，新行插入文件头部，时间倒序\n")

    # 接口验证
    print(">> 接口验证...")
    test = fetch_latest('000001')
    if test is None or test.empty:
        print("[FAIL] 腾讯行情接口无法访问，退出。")
        return
    print(f"[OK] 000001 获取 {len(test)} 条，最新={test['交易日期'].max()}")
    print()

    # 扫描所有文件
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv') and f[:6].isdigit():
                all_files.append(os.path.join(root, f))

    # 预分类
    need_update = []
    already_done = 0
    for fp in all_files:
        meta = get_file_meta(fp)
        if meta['last_date'] >= FETCH_END:
            already_done += 1
        else:
            need_update.append(fp)

    print(f"总文件: {len(all_files)} | 已是最新: {already_done} | 需更新: {len(need_update)}\n")

    ok_count = skip_count = nodata_count = error_count = 0
    errors = []

    for fp in tqdm(need_update, desc="处理进度"):
        status, info = process_file(fp)
        if status == 'ok':
            ok_count += 1
        elif status == 'skip':
            skip_count += 1
        elif status == 'nodata':
            nodata_count += 1
        else:
            error_count += 1
            errors.append(info)

    total = max(len(need_update), 1)
    print(f"\n{'='*50}")
    print(f" ✅ 成功（头部插入） : {ok_count:5d} 只  ({ok_count/total*100:.1f}%)")
    print(f" ⏭️  已是最新（跳过）: {already_done + skip_count:5d} 只")
    print(f" 📭 无新数据（停牌等）: {nodata_count:5d} 只")
    print(f" ❌ 最终失败         : {error_count:5d} 只  ({error_count/total*100:.1f}%)")
    print(f"{'='*50}")
    if errors:
        print("错误样本:")
        for e in errors[:10]:
            print(f"  {e}")

    # 验证 000560 结果
    print("\n>> 验证 000560.csv 前5行:")
    try:
        df_check = pd.read_csv(
            r'all_stock_data\1-500\000560.csv',
            nrows=5, encoding='utf-8-sig', dtype=str
        )
        print(df_check[['股票代码','名称','交易日期','开盘价','收盘价']].to_string(index=False))
    except Exception as e:
        print(f"验证失败: {e}")


if __name__ == '__main__':
    main()
