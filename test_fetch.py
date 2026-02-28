import akshare as ak
import traceback

try:
    df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date='20250228', end_date='20260228', adjust='qfq')
    print(df.head())
except Exception as e:
    traceback.print_exc()
