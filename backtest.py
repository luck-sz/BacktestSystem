import os
import pandas as pd
import numpy as np
import asyncio

def run_backtest(data_dir):
    asyncio.run(_run_backtest_wrapper(data_dir))

async def _run_backtest_wrapper(data_dir):
    async def print_callback(level, msg):
        print(f"[{level}] {msg}")
    await run_backtest_async(data_dir, print_callback)

async def run_backtest_async(data_dir, log_callback, start_date="2026-01-02", end_date="2026-02-27", strategy="rsv", initial_capital=100000.0, max_positions=2):
    """
    通用平台横截面回测引擎
    支持：RSV背离+缩量轮动策略、5日强平动量策略
    """
    if strategy == "ma":
        await log_callback("INFO", f"=== [策略启动] 5日强平动量策略 ({start_date} 到 {end_date}) ===")
    elif strategy == "brick":
        await log_callback("INFO", f"=== [策略启动] 砖型图打分策略 ({start_date} 到 {end_date}) ===")
    else:
        await log_callback("INFO", f"=== [策略启动] RSV背离+缩量轮动策略 ({start_date} 到 {end_date}) ===")
    
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        if "大盘" in root:
            continue
        for f in files:
            if f.endswith('.csv') and f[:6].isdigit():
                csv_files.append(os.path.join(root, f))
    
    import datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    await log_callback("INFO", f"开始加载数据: 共发现 {len(csv_files)} 只个股数据，准备全市场回测...")
    target_files = csv_files
    total_files = len(target_files)
    
    all_dates = set()
    stock_data = {}
    stock_names = {}
    
    for idx, file_path in enumerate(target_files):
        if idx % 100 == 0 and idx > 0:
            await log_callback("DEBUG", f"仍在处理中: 已预加载并计算指标 {idx}/{total_files} 只股票...")
            await asyncio.sleep(0.01)
            
        stock_code = os.path.basename(file_path).replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df) < 25:
                continue
                
            # 适配新的数据结构
            rename_map = {
                '名称': 'name',
                '交易日期': 'date',
                '开盘价': 'open',
                '最高价': 'high',
                '最低价': 'low',
                '收盘价': 'close',
                '成交量(手)': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
            
            if 'name' in df.columns:
                stock_names[stock_code] = df['name'].iloc[0]
            
            df['date'] = pd.to_datetime(df['date'].astype(str))
            df = df.sort_values(by='date').reset_index(drop=True)
            
            params_ok = False
            if strategy == "rsv":
                # RSV Long (21)
                low_21 = df['low'].rolling(window=21, min_periods=1).min()
                high_close_21 = df['close'].rolling(window=21, min_periods=1).max()
                df['rsv_long'] = (df['close'] - low_21) / (high_close_21 - low_21 + 1e-9) * 100.0
                
                # RSV Short (3)
                low_3 = df['low'].rolling(window=3, min_periods=1).min()
                high_close_3 = df['close'].rolling(window=3, min_periods=1).max()
                df['rsv_short'] = (df['close'] - low_3) / (high_close_3 - low_3 + 1e-9) * 100.0
                
                df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
                params_ok = True
            elif strategy == "ma":
                df['sma5'] = df['close'].rolling(window=5).mean()
                df['sma20'] = df['close'].rolling(window=20).mean()
                df['buy_signal'] = (df['sma5'] > df['sma20']) & (df['sma5'].shift(1) <= df['sma20'].shift(1))
                params_ok = True
            elif strategy == "brick":
                df['ma60'] = df['close'].rolling(window=60).mean()
                
                low_9 = df['low'].rolling(window=9).min()
                high_9 = df['high'].rolling(window=9).max()
                diff_9 = (high_9 - low_9).replace(0, 0.001)
                rsv = (df['close'] - low_9) / diff_9 * 100
                k = rsv.ewm(alpha=1/3, adjust=False).mean()
                d = k.ewm(alpha=1/3, adjust=False).mean()
                df['j'] = 3 * k - 2 * d
                
                hhv4 = df['high'].rolling(window=4).max()
                llv4 = df['low'].rolling(window=4).min()
                diff4 = (hhv4 - llv4).replace(0, 0.001)
                
                var1a = (hhv4 - df['close']) / diff4 * 100 - 90
                var2a = var1a.ewm(alpha=1/4, adjust=False).mean() + 100
                
                var3a = (df['close'] - llv4) / diff4 * 100
                var4a = var3a.ewm(alpha=1/6, adjust=False).mean()
                var5a = var4a.ewm(alpha=1/6, adjust=False).mean() + 100
                
                v6a = (var5a - var2a).fillna(0)
                df['brick'] = np.where(v6a > 4, v6a - 4, 0.0)
                
                b_t0 = df['brick']
                b_t1 = df['brick'].shift(1)
                b_t2 = df['brick'].shift(2)
                
                color_ok = (b_t1 < b_t2) & (b_t0 > b_t1)
                trend_ok = df['close'] > df['ma60']
                oversold_ok = df['j'].shift(1).rolling(window=5).min() < 0
                
                red_len = b_t0 - b_t1
                green_len = b_t2 - b_t1
                length_ok = red_len > green_len
                
                df['brick_score'] = red_len / (green_len + 0.001)
                df['buy_signal'] = color_ok & trend_ok & oversold_ok & length_ok
                params_ok = True
                
            if not params_ok:
                continue
                
            # Filter by date range AFTER computing rolling indicators
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                continue
                
            df.set_index('date', inplace=True)
            stock_data[stock_code] = df
            all_dates.update(df.index)
        except Exception as e:
            pass

    sorted_dates = sorted(list(all_dates))
    
    trades = []
    held_stocks = {} 
    buy_list = []

    daily_equity_curve = {}
    partial_equity_updates = {}
    current_equity = 1.0
    
    if len(sorted_dates) == 0:
        await log_callback("WARN", "在指定的日期范围内没有可用数据...")
        return None
        
    await log_callback("INFO", "开始进行日级别横截面撮合与仿真...")
    
    for i in range(len(sorted_dates)):
        current_date = sorted_dates[i]
        date_str = current_date.strftime("%Y-%m-%d")
        
        if i % 10 == 0:
            await log_callback("DEBUG", f"回测进行中: 正在撮合 {date_str} 的交易 ({i}/{len(sorted_dates)}天)...")
            await asyncio.sleep(0.01)
        
        # 1. 开盘/集合竞价买入
        bought_today = []
        for stock in buy_list:
            if len(held_stocks) >= max_positions:
                break
            if stock not in held_stocks:
                df = stock_data[stock]
                if current_date in df.index:
                    open_price = float(df.loc[current_date, 'open']) if strategy in ('rsv', 'brick') else float(df.loc[current_date, 'close'])
                    
                    if strategy == 'brick':
                        prev_date = sorted_dates[i-1] if i > 0 else None
                        if prev_date and prev_date in df.index:
                            prev_close = float(df.loc[prev_date, 'close'])
                            if (open_price / prev_close - 1) > 0.005:
                                continue

                    if not np.isnan(open_price) and open_price > 0:
                        # 初始资金为 initial_capital，随着收益曲线复利
                        pos_value = (initial_capital * current_equity) / max_positions
                        shares = int(pos_value / (open_price * 100)) * 100
                        if shares == 0:
                            shares = 100
                        actual_cost = shares * open_price
                        
                        held_stocks[stock] = {
                            'buy_date': current_date,
                            'buy_price': open_price,
                            'start_idx': i,
                            'shares': shares,
                            'buy_amount': actual_cost
                        }
                        bought_today.append(stock)
        buy_list = []
        
        # 记录今日净值情况 (当日包含未卖出的老持仓和早盘新买入的新持仓)
        daily_ret_sum = 0.0
        for stock in held_stocks:
            df = stock_data[stock]
            if current_date in df.index:
                curr_close = float(df.loc[current_date, 'close'])
                if stock in bought_today:
                    buy_p = held_stocks[stock]['buy_price']
                    ret = (curr_close - buy_p) / buy_p
                else:
                    prev_date = sorted_dates[i-1]
                    if prev_date in df.index:
                        prev_close = float(df.loc[prev_date, 'close'])
                        ret = (curr_close - prev_close) / prev_close
                    else:
                        ret = 0.0
                daily_ret_sum += ret
        
        # 为了防除数0和保持基准，取目前实际持仓数量或者最大仓数量的占卜
        active_pos = max(len(held_stocks), 1) if strategy == 'ma' else max_positions
        port_daily_ret = daily_ret_sum / active_pos
        current_equity *= (1 + port_daily_ret)
        rounded_eq = round((current_equity - 1) * 100, 4)
        daily_equity_curve[date_str] = rounded_eq
        partial_equity_updates[date_str] = rounded_eq
        
        if len(partial_equity_updates) >= 20 or i == len(sorted_dates) - 1:
            await log_callback("EQUITY_UPDATE", partial_equity_updates)
            partial_equity_updates = {}
            
        # 2. 尾盘卖出判定
        to_remove = []
        for stock, info in held_stocks.items():
            hold_days = i - info['start_idx']
            df = stock_data[stock]
            if current_date in df.index:
                sell_price = float(df.loc[current_date, 'close'])
                profit_pct = (sell_price - info['buy_price']) / info['buy_price'] * 100
                
                sell_reason = None
                if strategy == "rsv":
                    if hold_days >= 2:
                        sell_reason = "持仓满2天尾盘卖出"
                elif strategy == "ma":
                    if profit_pct > 0.2:
                        sell_reason = "盈利卖出(模拟14:55)"
                    elif hold_days >= 5:
                        sell_reason = "持仓满5天强制卖出"
                elif strategy == "brick":
                    if profit_pct > 0:
                        sell_reason = "盈利卖出"
                    elif hold_days >= 5:
                        sell_reason = "持仓满5天强制卖出"
                
                if sell_reason:
                    trade_data = {
                        '代码': stock,
                        '名称': stock_names.get(stock, stock),
                        '买入日期': info['buy_date'].strftime("%Y-%m-%d"),
                        '买入价': round(info['buy_price'], 2),
                        '购买股数': info.get('shares', 0),
                        '购买金额': round(info.get('buy_amount', 0), 2),
                        '卖出日期': date_str,
                        '卖出价': round(sell_price, 2),
                        '持仓天数': hold_days if hold_days > 0 else 1,
                        '收益率(%)': round(profit_pct, 2)
                    }
                    trades.append(trade_data)
                    await log_callback("TRADE_RECORD", trade_data)
                    to_remove.append(stock)
        
        for stock in to_remove:
            del held_stocks[stock]
            
        # 3. 收盘后选股 盘后计算买入名单 (待次日买入)
        available_slots = max_positions - len(held_stocks)
        if available_slots > 0:
            candidates = []
            for stock, df in stock_data.items():
                if stock in held_stocks:
                    continue
                if current_date in df.index:
                    row = df.loc[current_date]
                    if strategy == "rsv":
                        if pd.notnull(row['rsv_long']) and pd.notnull(row['rsv_short']) and pd.notnull(row['vol_ratio']):
                            if row['rsv_long'] >= 80 and row['rsv_short'] <= 20 and row['vol_ratio'] > 0:
                                candidates.append({
                                    'stock': stock,
                                    'vol_ratio': row['vol_ratio']
                                })
                    elif strategy == "ma":
                        if pd.notnull(row.get('buy_signal')) and row['buy_signal']:
                            candidates.append({
                                'stock': stock,
                                'vol_ratio': 1.0 # 随便给个权重
                            })
                    elif strategy == "brick":
                        if pd.notnull(row.get('buy_signal')) and row['buy_signal']:
                            candidates.append({
                                'stock': stock,
                                'score': float(row.get('brick_score', 0))
                            })
            
            if candidates:
                if strategy == "rsv":
                    candidates.sort(key=lambda x: x['vol_ratio'])
                    buy_list = [c['stock'] for c in candidates[:available_slots]]
                elif strategy == "brick":
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    buy_list = [c['stock'] for c in candidates] # Keep all. Next morning's opening price gap check may filter some out.
                else:
                    buy_list = [c['stock'] for c in candidates[:available_slots]]

    await log_callback("INFO", "=== 回测结束，正在汇总统计报告 ===")

    
    if not trades:
        await log_callback("WARN", "没有产生足够的触发信号导致交易...")
        return None
        
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df['收益率(%)'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    avg_profit = trades_df['收益率(%)'].mean()
    
    await log_callback("INFO", f"总交易次数: {total_trades} 次")
    await log_callback("INFO", f"交易胜率: {win_rate:.2f}%")
    
    return {
        'initial_capital': initial_capital,
        'final_capital': round(initial_capital * current_equity, 2),
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': round(win_rate, 2),
        'avg_profit': round(avg_profit, 2),
        'daily_equity': daily_equity_curve
    }

if __name__ == "__main__":
    data_dir = "e:/PythonProject/BacktestSystem/all_stock_data"
    run_backtest(data_dir)
