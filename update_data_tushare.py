import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# 配置区
TOKEN = '8cd72c7ab2a53ee9cdbb8cff0c23778c691e1aecc63d0ad6b692e708'
DATA_DIR = r'all_stock_data'
TS_API = ts.pro_api(TOKEN)

# 映射与回测引擎匹配
# TuShare pro_bar/daily 列名 -> 项目所需列名
COL_MAPPING = {
    'trade_date': '交易日期',
    'open': '开盘价',
    'high': '最高价',
    'low': '最低价',
    'close': '收盘价',
    'vol': '成交量(手)',
}

def get_ts_code_map():
    """获取所有股票代码映射: 000001 -> 000001.SZ"""
    print("正在从 TuShare 获取股票列表映射...")
    try:
        # 尝试 API 获取
        df = TS_API.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
        return {row['symbol']: (row['ts_code'], row['name']) for _, row in df.iterrows()}
    except Exception as e:
        print(f"API 获取股票列表失败 (频率限制)，尝试从本地数据目录恢复映射...")
        # 兜底方案：根据常见规则拼接代码 (6开头.SH, 其余.SZ)
        mapping = {}
        if os.path.exists(DATA_DIR):
            for root, _, files in os.walk(DATA_DIR):
                for f in files:
                    if f.endswith(".csv") and len(f) == 10:
                        symbol = f[:6]
                        # 简单规则：6开头是上海，其他是深圳（北交所除外，但作为回测初步够用）
                        ts_code = f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"
                        mapping[symbol] = (ts_code, symbol) # 名字暂用代码代替
        return mapping

def update_stock_tushare(symbol, ts_code, name, start_str, end_str, save_path):
    """抓取单只股票详情并保存"""
    # 尝试使用 pro_bar 获取前复权数据
    # 注意：pro_bar 某些积分级别可能无法使用，如果失败则尝试 daily (不复权)
    try:
        # 增加延迟防封
        time.sleep(0.5) 
        
        # 尝试使用 ts.pro_bar
        df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_str, end_date=end_str)
        
        if df is None or df.empty:
            # 备选：尝试通用 daily 接口
            df = TS_API.daily(ts_code=ts_code, start_date=start_str, end_date=end_str)
            if df is None or df.empty:
                return False
                
        df['名称'] = name
        df.rename(columns=COL_MAPPING, inplace=True)
        
        # 保持与项目一致的列顺和内容
        # 筛选出项目 backtest.py 需要的列
        cols = ['名称', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '成交量(手)']
        # 检查是否存在，不存在的列补空 (TuShare 可能少某些列)
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        
        # 排序：从旧到新 (backtest.py 通常需要)
        df = df.sort_values('交易日期')
        
        df[cols].to_csv(save_path, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        if "抱歉，您没有权限访问该接口" in str(e) or "权限" in str(e):
            # 如果 pro_bar 没权限，就退而求次使用普通 daily 但无法自动复权
            # 实际生产中建议用户积攒积分，这里提供一个反馈
            # print(f"\n[ERROR] 股票 {symbol} 更新失败: 积分不足无法使用复权接口。")
            pass
        return False

def main():
    print("=== 开始使用 TuShare 方案更新数据 ===")
    
    # 1. 建立映射
    code_map = get_ts_code_map()
    if not code_map:
        print("无法建立代码映射，程序退出。")
        return

    # 2. 定位本地文件
    existing_files = {}
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".csv") and len(f) == 10:
                symbol = f[:6]
                existing_files[symbol] = os.path.join(root, f)
    
    print(f"本地发现 {len(existing_files)} 只股票需要更新。")

    # 3. 设置时间范围 (最近一年)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    print(f"更新范围：{start_str} 至 {end_str}")

    # 4. 循环更新 (TuShare 需要控制频率)
    success_count = 0
    fail_count = 0
    
    # 为了演示建议先跑一小批，确认 token 有效性
    symbols = list(existing_files.keys())
    
    # 增加一个简单的断点续传检测：如果文件是今天更新的，可以跳过 (可选)
    
    for sym in tqdm(symbols, desc="TuShare 同步进度"):
        if sym not in code_map:
            fail_count += 1
            continue
            
        ts_code, name = code_map[sym]
        save_path = existing_files[sym]
        
        if update_stock_tushare(sym, ts_code, name, start_str, end_str, save_path):
            success_count += 1
        else:
            fail_count += 1
            # 如果连续失败多次，可能是 Token 流量用完或被限制
            if fail_count > 10 and success_count == 0:
                print("\n[CRITICAL] 连续多次请求失败，请检查 TuShare Token 积分是否足够。")
                break
        
        # TuShare 普通账号频率限制通常为 20-50次/分钟
        # 每抓取一只停一下
        time.sleep(0.5) 

    print(f"\n=== 同步完成 ===")
    print(f"成功: {success_count} | 失败: {fail_count}")

if __name__ == "__main__":
    main()
