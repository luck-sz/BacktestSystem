"""
append_tencent_v4.py — 增量补充数据（腾讯接口，自愈版）
"""
import requests
import json
import pandas as pd
import os
import time
import random
from tqdm import tqdm
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 配置 ───
DATA_DIR    = r'all_stock_data'
# 自动获取今天日期 (如 20260305)
FETCH_END   = datetime.now().strftime("%Y%m%d")
FETCH_N     = 20           # 取 20 条确保有前一天收盘价
DELAY_MIN   = 0.05         # 并发时减小延迟
DELAY_MAX   = 0.1
MAX_RETRY   = 3
MAX_WORKERS = 10           # 并发线程数，提升10倍速度

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

STATIC_COLS = ['名称', '所属行业', '地域', '上市日期', 'TS代码']

# 每个线程自己的 Session 能够稍微提高稳定性，这里简单处理用全局 Session 带锁或并发 Session
# 腾讯接口对并发支持较好，Requests Session 是线程安全的。
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
            if '=' not in text: return None
            json_str = text[text.index('=') + 1:]
            data = json.loads(json_str)
            stock_data = data.get('data', {}).get(qq_code, {})
            klines = stock_data.get('qfqday', stock_data.get('day', []))
            if not klines: return None
            
            klines.sort(key=lambda x: x[0])
            
            records = []
            for i in range(len(klines)):
                k = klines[i]
                date_str = str(k[0]).replace('-', '')
                item = {
                    '交易日期':    date_str,
                    '开盘价':      float(k[1]),
                    '收盘价':      float(k[2]),
                    '最高价':      float(k[3]),
                    '最低价':      float(k[4]),
                    '成交量(手)':  float(k[5]),
                    '换手率(%)':   float(k[7]) if len(k) > 7 and k[7] else '',
                    '成交额(千元)': float(k[8]) if len(k) > 8 and k[8] else '',
                }
                if i > 0:
                    pre_close = float(klines[i-1][2])
                    curr_close = float(k[2])
                    item['前收盘价'] = pre_close
                    item['涨跌额'] = round(curr_close - pre_close, 3)
                    item['涨跌幅(%)'] = round((curr_close / pre_close - 1) * 100, 3)
                else:
                    item['前收盘价'] = ''
                    item['涨跌额'] = ''
                    item['涨跌幅(%)'] = ''
                records.append(item)
            return pd.DataFrame(records)
        except:
            if attempt < MAX_RETRY - 1: time.sleep(1)
    return None

def get_file_meta(file_path: str):
    try:
        df_head = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig', dtype=str)
        last_date = str(df_head['交易日期'].iloc[0]).strip() if not df_head.empty else ''
        static_info = {col: str(df_head[col].iloc[0]) if col in df_head.columns and not df_head.empty else '' for col in STATIC_COLS}
        return {'last_date': last_date, 'static': static_info}
    except:
        return {'last_date': '', 'static': {}}

def process_file(file_path: str):
    try:
        symbol = os.path.basename(file_path)[:6]
        meta = get_file_meta(file_path)
        last_date = meta['last_date']
        
        # 强制自愈逻辑
        try:
            df_top = pd.read_csv(file_path, nrows=1, encoding='utf-8-sig', dtype=str)
            if not df_top.empty:
                pre_c = str(df_top.get('前收盘价', pd.Series([''])).iloc[0]).strip()
                if pre_c == '' or pre_c == 'nan':
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        lines = f.readlines()
                    if len(lines) > 2:
                        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                            f.write(lines[0])
                            f.writelines(lines[2:])
                        meta = get_file_meta(file_path)
                        last_date = meta['last_date']
        except: pass

        if last_date >= FETCH_END:
            return 'skip', symbol

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        df_new = fetch_latest(symbol)
        if df_new is None or df_new.empty: return 'nodata', symbol

        if last_date:
            df_new = df_new[df_new['交易日期'] > last_date]
        df_new = df_new[df_new['交易日期'] <= FETCH_END]

        if df_new.empty: return 'skip', symbol

        df_new['股票代码'] = symbol
        for col in STATIC_COLS:
            df_new[col] = meta['static'].get(col, '')

        for col in TARGET_COLUMNS:
            if col not in df_new.columns:
                df_new[col] = ''

        df_new = df_new[TARGET_COLUMNS].sort_values('交易日期', ascending=False)
        new_rows_count = len(df_new)

        with open(file_path, 'r', encoding='utf-8-sig') as f:
            original = f.read()
        lines = original.splitlines(keepends=True)
        new_csv = df_new.to_csv(index=False, header=False, encoding='utf-8-sig')
        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(lines[0])
            f.write(new_csv)
            f.writelines(lines[1:])
        return 'ok', f"{symbol}(+{new_rows_count})"
    except Exception as e:
        return 'error', f"{os.path.basename(file_path)}:{str(e)[:30]}"

def main():
    print(f"=== Stock Data Incremental Update (Tencent Multi-threaded) ===")
    print(f"Target Date: {FETCH_END} | Workers: {MAX_WORKERS}\n")
    
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.csv') and f[:6].isdigit():
                all_files.append(os.path.join(root, f))
    
    total = len(all_files)
    print(f"Scanning {total} files...")
    
    ok = 0; skip = 0; nodata = 0; error = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in all_files}
        
        for future in tqdm(as_completed(futures), total=total, desc="Updating"):
            try:
                status, info = future.result()
                if status == 'ok': ok += 1
                elif status == 'skip': skip += 1
                elif status == 'nodata': nodata += 1
                else: error += 1
            except Exception as e:
                error += 1
    
    print(f"\nFinished!")
    print(f"  Success: {ok}")
    print(f"  Skipped: {skip}")
    print(f"  No Data: {nodata}")
    print(f"  Errors:  {error}")

    # 发送通知
    send_server_chan(ok, skip, nodata, error)

def send_server_chan(ok, skip, nodata, error):
    # ── 请在这里填写您的 Server 酱 SendKey ──
    # 或者通过环境变量命令行：export SC_SENDKEY=your_key
    sc_key = os.environ.get('SC_SENDKEY', 'SCT267527T63O6p9A60O855O855O8') # 占位符供替换
    if not sc_key or 'SCT' not in sc_key:
        print("\n[SKIP] ServerChan Key 未设置，跳过发送消息。")
        return

    title = f"📈 股票数据更新报告 ({FETCH_END})"
    content = f"""
### 更新任务已完成
- **成功**: {ok} 只 
- **跳过**: {skip} 只
- **无数据**: {nodata} 只
- **失败**: {error} 只

**状态：** {"✅ 全部执行完毕" if error == 0 else "⚠️ 部分更新存在错误"}
    """
    
    url = f"https://sctapi.ftqq.com/{sc_key}.send"
    try:
        res = requests.post(url, data={"title": title, "desp": content}, timeout=10)
        if res.status_code == 200:
            print(f"\n[OK] Server酱通知发送成功！")
        else:
            print(f"\n[FAIL] Server酱发送失败: {res.text}")
    except Exception as e:
        print(f"\n[ERROR] 发送通知异常: {e}")

if __name__ == '__main__':
    main()
