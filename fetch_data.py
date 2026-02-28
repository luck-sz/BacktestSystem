import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def fetch_past_year_data(symbol="000001", adjust="qfq"):
    """
    获取过去一年的日K线数据
    :param symbol: 股票代码 (默认 000001 平安银行)
    :param adjust: 复权类型 (默认 qfq 前复权)
    """
    print(f"正在获取股票 {symbol} 的数据...")
    
    # 获取今天的时间
    end_date = datetime.now()
    # 获取一年前的时间
    start_date = end_date - timedelta(days=365)
    
    # 格式化时间为 YYYYMMDD (akshare 要求)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    print(f"时间范围: {start_str} 至 {end_str}")
    
    # 获取历史日线数据
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust=adjust)
        
        if df.empty:
            print(f"未获取到 {symbol} 的数据，请检查股票代码或网络。")
            return
        
        # 将数据保存为 CSV
        filename = f"history_data_{symbol}_{start_str}_{end_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"获取成功！共 {len(df)} 条数据。")
        print(f"数据已保存到当前目录下的: {filename}")
        
        # 打印前 5 行预览
        print("\n数据预览 (前5行):")
        print(df.head())
        
    except Exception as e:
        print(f"数据获取失败: {e}")

if __name__ == "__main__":
    # 你可以修改这里的代码来获取其他的股票，例如茅台：fetch_past_year_data("600519")
    fetch_past_year_data("000001")
