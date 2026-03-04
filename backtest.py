"""
backtest.py — 通用 A股横截面回测引擎

支持策略：
  - rsv   : RSV背离 + 缩量轮动（短期超跌反弹）
  - brick : 砖型图打分（KDJ超卖 + MA60趋势 + 砖型柱反转）
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Coroutine

import numpy as np
import pandas as pd

# ─────────────────────── 常量 ───────────────────────
RENAME_MAP = {
    "名称": "name",
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(手)": "volume",
}

STRATEGY_NAMES = {
    "rsv":   "RSV背离+缩量轮动策略",
    "brick": "砖型图打分策略",
    "small_bank": "小市值+银行轮动(实盘优化版)",
}

MIN_ROWS = 25  # 数据行数过少则跳过

# ─────────────────────── 指标计算 ───────────────────────

def _calc_rsv(df: pd.DataFrame) -> pd.DataFrame:
    """RSV背离策略指标"""
    low21   = df["low"].rolling(21, min_periods=1).min()
    high21  = df["close"].rolling(21, min_periods=1).max()
    df["rsv_long"]  = (df["close"] - low21)  / (high21 - low21  + 1e-9) * 100

    low3    = df["low"].rolling(3, min_periods=1).min()
    high3   = df["close"].rolling(3, min_periods=1).max()
    df["rsv_short"] = (df["close"] - low3)   / (high3  - low3   + 1e-9) * 100

    df["vol_ratio"] = df["volume"] / df["volume"].shift(1).replace(0, np.nan)
    return df


def _calc_brick(df: pd.DataFrame) -> pd.DataFrame:
    """砖型图打分策略指标（通达信风格近似）"""
    df["ma60"] = df["close"].rolling(60).mean()

    low9  = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    diff9 = (high9 - low9).clip(lower=0.001)
    rsv   = (df["close"] - low9) / diff9 * 100
    k     = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d     = k.ewm(alpha=1 / 3, adjust=False).mean()
    df["j"] = 3 * k - 2 * d

    hhv4   = df["high"].rolling(4).max()
    llv4   = df["low"].rolling(4).min()
    diff4  = (hhv4 - llv4).clip(lower=0.001)

    var1a  = (hhv4 - df["close"]) / diff4 * 100 - 90
    var2a  = var1a.ewm(alpha=1 / 4, adjust=False).mean() + 100
    var3a  = (df["close"] - llv4) / diff4 * 100
    var4a  = var3a.ewm(alpha=1 / 6, adjust=False).mean()
    var5a  = var4a.ewm(alpha=1 / 6, adjust=False).mean() + 100

    v6a         = (var5a - var2a).fillna(0)
    df["brick"] = np.where(v6a > 4, v6a - 4, 0.0)

    b0 = df["brick"]
    b1 = df["brick"].shift(1)
    b2 = df["brick"].shift(2)

    # color_ok: 要求指标出现 V 型反向，且回调深度超过 3%（避免 000913 这种微小抖动）
    color_ok    = (b1 < b2) & (b0 > b1) & (b1 < b2 * 0.97)
    trend_ok    = df["close"] > df["ma60"]
    oversold_ok = df["j"].shift(1).rolling(5).min() < 0
    red_len     = b0 - b1
    green_len   = b2 - b1
    length_ok   = red_len > green_len

    df["brick_score"] = red_len / (green_len + 0.001)
    df["buy_signal"]  = color_ok & trend_ok & oversold_ok & length_ok
    return df


def _calc_small_bank(df: pd.DataFrame) -> pd.DataFrame:
    """小市值+银行轮动混合策略指标"""
    df["turnover"] = df["close"] * df["volume"]
    df["avg_turnover20"] = df["turnover"].rolling(20, min_periods=1).mean()
    df["max_vol120"] = df["volume"].rolling(120, min_periods=1).max()
    # days_listed 由外层统筹提供，避免在此处受截断影响
    if "days_listed" not in df.columns:
        df["days_listed"] = np.arange(len(df))
    df["prev_close"] = df["close"].shift(1)
    return df


_INDICATOR_FUNCS = {
    "rsv":   _calc_rsv,
    "brick": _calc_brick,
    "small_bank": _calc_small_bank,
}

def _is_bank_period(dt: pd.Timestamp) -> bool:
    mmdd = dt.strftime('%m-%d')
    return ('12-15' <= mmdd <= '12-31') or ('01-01' <= mmdd <= '01-30') or ('04-04' <= mmdd <= '04-28')

# ─────────────────────── 单文件加载（同步，用于 executor） ───────────────────────

def _load_single_file(args: tuple) -> tuple[str, str, pd.DataFrame | None]:
    """
    同步地加载并预处理单只股票 CSV。
    返回 (stock_code, stock_name, df_or_None)
    """
    file_path, strategy, start_ts, end_ts, exclude_boards = args
    stock_code = os.path.basename(file_path).replace(".csv", "")
    calc_fn    = _INDICATOR_FUNCS.get(strategy)
    if calc_fn is None:
        return stock_code, "", None

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in RENAME_MAP)
        if df.empty or len(df) < MIN_ROWS:
            return stock_code, "", None

        df.rename(columns=RENAME_MAP, inplace=True)
        stock_name = str(df["name"].iloc[0]) if "name" in df.columns else stock_code

        # 检查是否由于名字带 ST/退 等情况被排除
        if exclude_boards and "st" in exclude_boards:
            name_upper = stock_name.upper()
            if "ST" in name_upper or "退" in name_upper:
                return stock_code, stock_name, None

        # 记录全历史交易日数（在截取前），用于次新股过滤
        total_trading_days = len(df)

        # ====== 核心性能优化：预先过滤无关历史数据 ======
        # 预留 500 天(自然日)作为指标滚动（如MA120, EWM）预热空间
        warmup_days = pd.Timedelta(days=500)
        min_date_int = int((start_ts - warmup_days).strftime("%Y%m%d"))
        
        # 将原始 date 列作为数字快速剔除远古数据，大幅减少随后 to_datetime 及指标计算的耗时
        df["_date_num"] = pd.to_numeric(df["date"], errors="coerce")
        df = df[df["_date_num"] >= min_date_int].copy()
        df.drop(columns=["_date_num"], inplace=True)

        if df.empty or len(df) < MIN_ROWS:
            return stock_code, stock_name, None

        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        df.drop_duplicates(subset=["date"], keep="last", inplace=True)  # 防止重复日期导致 loc 返回 DataFrame
        df.reset_index(drop=True, inplace=True)

        # 修复指标需要的 days_listed (保留其在全量数据中的近似绝对序号)
        df["days_listed"] = total_trading_days - len(df) + np.arange(len(df))

        df = calc_fn(df)

        # 必须在计算指标之后再做精确的日期截取，保证滚动窗口完整且最终输出期间精准
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if df.empty:
            return stock_code, stock_name, None

        # 写入全历史天数，供 _select_candidates 过滤次新股
        df["total_trading_days"] = total_trading_days

        df.set_index("date", inplace=True)
        return stock_code, stock_name, df.reset_index().to_dict("list")

    except Exception:
        return stock_code, "", None


# ─────────────────────── 主回测引擎 ───────────────────────

LogCallback = Callable[[str, Any], Coroutine]


async def run_backtest_async(
    data_dir:        str,
    log_callback:    LogCallback,
    start_date:      str   = "2026-01-02",
    end_date:        str   = "2026-02-27",
    strategy:        str   = "rsv",
    initial_capital: float = 100_000.0,
    max_positions:   int   = 2,
    exclude_boards:  list[str] = None
) -> dict | None:
    """
    异步横截面回测主函数。

    参数
    ----
    data_dir        : 股票 CSV 数据目录
    log_callback    : async (level: str, message: Any) -> None
    start_date      : 回测开始日期 YYYY-MM-DD
    end_date        : 回测结束日期 YYYY-MM-DD
    strategy        : 策略名称 rsv / ma / brick
    initial_capital : 初始资金（元）
    max_positions   : 最大持仓数量（只）
    """
    strategy_label = STRATEGY_NAMES.get(strategy, strategy)
    await log_callback("INFO", f"=== [策略启动] {strategy_label} ({start_date} 到 {end_date}) ===")

    # ── 1. 扫描 CSV 文件列表 ──
    csv_files: list[str] = []
    
    exclude_boards = exclude_boards or []
    
    for root, _, files in os.walk(data_dir):
        if "大盘" in root:
            continue
        for f in files:
            if f.endswith(".csv") and f[:6].isdigit():
                code = f[:6]
                # 板块过滤（在文件扫描阶段快速排除，减少 I/O）
                if "kc"  in exclude_boards and code.startswith("688"):
                    continue
                if "cy"  in exclude_boards and (code.startswith("300") or code.startswith("301")):
                    continue
                if "bse" in exclude_boards and (code.startswith("4") or code.startswith("8")):
                    continue
                # ST/退市 通过名字过滤，在 _load_single_file 内处理
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        await log_callback("WARN", f"数据目录中未找到任何 CSV 文件：{data_dir}")
        return None

    total_files = len(csv_files)
    await log_callback("INFO", f"开始加载数据: 共发现 {total_files} 只个股数据，准备全市场回测...")

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    args_list = [(fp, strategy, start_ts, end_ts, exclude_boards) for fp in csv_files]

    # ── 2. 并发预加载 & 指标计算（使用进程池避免 GIL） ──
    stock_data:  dict[str, pd.DataFrame] = {}
    stock_names: dict[str, str]          = {}
    all_dates:   set                     = set()

    loop = asyncio.get_event_loop()
    BATCH = 200  # 每批处理数量，兼顾内存与进度反馈

    with ProcessPoolExecutor() as executor:
        for batch_start in range(0, total_files, BATCH):
            batch = args_list[batch_start: batch_start + BATCH]
            results = await loop.run_in_executor(
                executor,
                _batch_load,
                batch,
            )
            for code, name, df_dict in results:
                if df_dict is not None:
                    df = pd.DataFrame(df_dict)
                    df.set_index("date", inplace=True)
                    stock_data[code]  = df
                    stock_names[code] = name
                    all_dates.update(df.index)

            loaded = min(batch_start + BATCH, total_files)
            await log_callback("DEBUG", f"仍在处理中: 已预加载并计算指标 {loaded}/{total_files} 只股票...")
            await asyncio.sleep(0)  # 让事件循环有机会推送 SSE

    sorted_dates = sorted(all_dates)
    if not sorted_dates:
        await log_callback("WARN", "在指定的日期范围内没有可用数据，请检查数据目录或日期范围。")
        return None

    await log_callback("INFO", f"数据加载完毕，共 {len(stock_data)} 只股票参与回测，回测日历 {len(sorted_dates)} 个交易日。")
    await log_callback("INFO", "开始进行日级别横截面撮合与仿真...")

    # ── 3. 逐日撮合 ──
    trades:               list[dict] = []
    held_stocks:          dict[str, dict] = {}
    buy_list:             list[str] = []
    daily_equity_curve:   dict[str, float] = {}
    partial_equity_updates: dict[str, float] = {}
    current_equity = 1.0

    total_days = len(sorted_dates)

    for i, current_date in enumerate(sorted_dates):
        date_str = current_date.strftime("%Y-%m-%d")

        if i % 10 == 0:
            await log_callback("DEBUG", f"回测进行中: 正在撮合 {date_str} 的交易 ({i}/{total_days} 天)...")
            await asyncio.sleep(0)

        # ── 3a. 开盘买入 ──
        bought_today: list[str] = []
        for stock in buy_list:
            if len(held_stocks) >= max_positions:
                break
            if stock in held_stocks:
                continue
            df = stock_data.get(stock)
            if df is None or current_date not in df.index:
                continue

            row = df.loc[current_date]
            open_price = float(row.get("open", np.nan))

            # 砖型图策略：过滤跳空高开 > 0.5%
            if strategy == "brick" and i > 0:
                prev_date = sorted_dates[i - 1]
                if prev_date in df.index:
                    prev_close = float(df.loc[prev_date, "close"])
                    if prev_close > 0 and (open_price / prev_close - 1) > 0.005:
                        continue

            if np.isnan(open_price) or open_price <= 0:
                continue

            pos_value   = (initial_capital * current_equity) / max_positions
            shares      = max(int(pos_value / (open_price * 100)) * 100, 100)
            actual_cost = shares * open_price

            held_stocks[stock] = {
                "buy_date":   current_date,
                "buy_price":  open_price,
                "start_idx":  i,
                "shares":     shares,
                "buy_amount": actual_cost,
            }
            bought_today.append(stock)

        buy_list = []

        # ── 3b. 当日净值更新 ──
        daily_ret_sum = 0.0
        n_positions   = len(held_stocks)

        for stock, info in held_stocks.items():
            df = stock_data.get(stock)
            if df is None or current_date not in df.index:
                continue
            curr_close = float(df.loc[current_date, "close"])
            if stock in bought_today:
                ret = (curr_close - info["buy_price"]) / info["buy_price"]
            elif i > 0:
                prev_date = sorted_dates[i - 1]
                if prev_date in df.index:
                    prev_close = float(df.loc[prev_date, "close"])
                    ret = (curr_close - prev_close) / prev_close if prev_close > 0 else 0.0
                else:
                    ret = 0.0
            else:
                ret = 0.0
            daily_ret_sum += ret

        # 用实际持仓数量做分母（若空仓则分母为 max_positions 保证基准稳定）
        divisor = n_positions if n_positions > 0 else max_positions
        current_equity *= (1 + daily_ret_sum / divisor)

        rounded_eq = round((current_equity - 1) * 100, 4)
        daily_equity_curve[date_str]    = rounded_eq
        partial_equity_updates[date_str] = rounded_eq

        if len(partial_equity_updates) >= 20 or i == total_days - 1:
            await log_callback("EQUITY_UPDATE", partial_equity_updates)
            partial_equity_updates = {}

        # ── 3c. 尾盘卖出判定 ──
        to_remove: list[str] = []
        is_bp = False
        bank_stocks = ["601988", "601398", "601288"]
        if strategy == "small_bank":
            is_bp = _is_bank_period(current_date)
            
        for stock, info in held_stocks.items():
            df = stock_data.get(stock)
            if df is None or current_date not in df.index:
                continue

            sell_price = float(df.loc[current_date, "close"])
            hold_days  = i - info["start_idx"]
            profit_pct = (sell_price - info["buy_price"]) / info["buy_price"] * 100

            sell_reason = None
            if strategy == "small_bank":
                if is_bp:
                    if stock not in bank_stocks:
                        sell_reason = "进入银行时段，清仓非银行股"
                else:
                    if stock in bank_stocks:
                        sell_reason = "进入小市值时段，清仓银行股"
                    elif profit_pct < -9.0:
                        sell_reason = "个股止损触发(跌幅>-9%)"
                    elif hold_days > 0 and i > 0:
                        # 放量抛售
                        prev_dt = sorted_dates[i-1]
                        if prev_dt in df.index:
                            max_vol = float(df.loc[prev_dt, "max_vol120"])
                            curr_vol = float(df.loc[current_date, "volume"])
                            if max_vol > 0 and curr_vol > max_vol * 0.9:
                                sell_reason = f"放量卖出(放量至新高90%以上)"
                    elif hold_days >= 5:
                        sell_reason = "持仓满5天无条件清仓(时间止损)"
            elif strategy == "brick":
                if hold_days >= 1:
                    open_p = float(df.loc[current_date, "open"])
                    open_profit = (open_p - info["buy_price"]) / info["buy_price"] * 100
                    if open_profit > 1.0:
                        sell_price = open_p
                        profit_pct = open_profit
                        sell_reason = "次日早盘开盘盈利超1%止盈"
                    else:
                        sell_reason = "次日尾盘14:55无条件清仓"
                elif hold_days >= 5:
                    sell_reason = "持仓满5天强制清仓(风控)"
            else:
                sell_reason = _judge_sell(strategy, hold_days, profit_pct)
                if not sell_reason and hold_days >= 5:
                    sell_reason = "持仓满5天兜底强制清仓"
            if sell_reason:
                trade_data = {
                    "代码":     stock,
                    "名称":     stock_names.get(stock, stock),
                    "买入日期": info["buy_date"].strftime("%Y-%m-%d"),
                    "买入价":   round(info["buy_price"], 2),
                    "购买股数": info.get("shares", 0),
                    "购买金额": round(info.get("buy_amount", 0), 2),
                    "卖出日期": date_str,
                    "卖出价":   round(sell_price, 2),
                    "持仓天数": max(hold_days, 1),
                    "收益率(%)": round(profit_pct, 2),
                }
                trades.append(trade_data)
                await log_callback("TRADE_RECORD", trade_data)
                to_remove.append(stock)

        for stock in to_remove:
            del held_stocks[stock]

        # ── 3d. 盘后选股（待次日买入） ──
        available_slots = max_positions - len(held_stocks)
        if available_slots > 0:
            buy_list = _select_candidates(
                strategy, stock_data, held_stocks, current_date, available_slots,
                exclude_boards,
            )

    # ── 4. 汇总统计 ──
    await log_callback("INFO", "=== 回测结束，正在汇总统计报告 ===")

    if not trades:
        await log_callback("WARN", "没有产生任何有效交易，请检查策略参数或扩大回测时间范围。")
        return None

    trades_df   = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_trades  = int((trades_df["收益率(%)"] > 0).sum())
    loss_trades = total_trades - win_trades
    win_rate    = win_trades / total_trades * 100
    avg_profit  = float(trades_df["收益率(%)"].mean())
    max_profit  = float(trades_df["收益率(%)"].max())
    max_loss    = float(trades_df["收益率(%)"].min())

    await log_callback("INFO", f"总交易次数: {total_trades} 次 | 胜率: {win_rate:.2f}% | 平均收益: {avg_profit:.2f}%")

    return {
        "initial_capital": initial_capital,
        "final_capital":   round(initial_capital * current_equity, 2),
        "total_trades":    total_trades,
        "win_trades":      win_trades,
        "loss_trades":     loss_trades,
        "win_rate":        round(win_rate, 2),
        "avg_profit":      round(avg_profit, 2),
        "max_profit":      round(max_profit, 2),
        "max_loss":        round(max_loss, 2),
        "daily_equity":    daily_equity_curve,
    }


# ─────────────────────── 辅助函数 ───────────────────────

def _batch_load(args_list: list[tuple]) -> list[tuple]:
    """进程池调用的批量加载入口（必须是顶层函数）"""
    return [_load_single_file(a) for a in args_list]


def _judge_sell(strategy: str, hold_days: int, profit_pct: float) -> str | None:
    """根据策略和持仓情况判断是否触发卖出，返回卖出原因或 None"""
    if strategy == "rsv":
        if hold_days >= 2:
            return "持仓满2天尾盘卖出"
    return None


def _select_candidates(
    strategy:       str,
    stock_data:     dict[str, pd.DataFrame],
    held_stocks:    dict[str, dict],
    current_date,
    available_slots: int,
    exclude_boards: list[str] | None = None,
) -> list[str]:
    """盘后选股，返回次日计划买入的股票列表"""
    exclude_boards = exclude_boards or []

    # ── 解析带参数的过滤条件 ──
    new_min_days     = None
    lowprice_limit   = None
    highprice_limit  = None
    lowvol_limit     = None   # 万元
    mktcap_min_limit = None   # 亿元
    mktcap_max_limit = None   # 亿元

    for item in exclude_boards:
        if item.startswith("new:"):
            try: new_min_days = int(item.split(":")[1])
            except: new_min_days = 365
        elif item.startswith("lowprice:"):
            try: lowprice_limit = float(item.split(":")[1])
            except: lowprice_limit = 3.0
        elif item.startswith("highprice:"):
            try: highprice_limit = float(item.split(":")[1])
            except: highprice_limit = 200.0
        elif item.startswith("lowvol:"):
            try: lowvol_limit = float(item.split(":")[1]) * 10000  # 万元转元
            except: lowvol_limit = 2000 * 10000
        elif item.startswith("mktcap_min:"):
            try: mktcap_min_limit = float(item.split(":")[1])
            except: mktcap_min_limit = 20.0
        elif item.startswith("mktcap_max:"):
            try: mktcap_max_limit = float(item.split(":")[1])
            except: mktcap_max_limit = 200.0

    filter_suspended = "suspended" in exclude_boards

    candidates: list[dict] = []

    for stock, df in stock_data.items():
        if stock in held_stocks or current_date not in df.index:
            continue
        row = df.loc[current_date]

        # ── 通用过滤（对所有策略生效）──
        # 停牌过滤
        if filter_suspended:
            vol = row.get("volume", 1)
            if pd.notnull(vol) and float(vol) == 0:
                continue

        # 次新股过滤（使用全历史交易日数）
        if new_min_days is not None:
            total_days = row.get("total_trading_days", row.get("days_listed", 9999))
            if pd.notnull(total_days) and float(total_days) < new_min_days:
                continue

        # 低价股过滤
        if lowprice_limit is not None:
            close = row.get("close", 0)
            if pd.notnull(close) and float(close) < lowprice_limit:
                continue

        # 高价股过滤
        if highprice_limit is not None:
            close = row.get("close", 0)
            if pd.notnull(close) and float(close) > highprice_limit:
                continue
        
        # 获取日均成交额用于流动性和市值过滤
        avg_t = row.get("avg_turnover20")
        if avg_t is None:
            close = float(row.get("close", 0) or 0)
            vol   = float(row.get("volume", 0) or 0)
            avg_t = close * vol # 估算

        # 低流动性过滤（日均成交额)
        if lowvol_limit is not None:
            if pd.notnull(avg_t) and float(avg_t) < lowvol_limit:
                continue

        # 市值过滤（市值 = 日均成交额 / 0.02）
        # 换算关系：threshold_yi * 1e8 * 0.02 = threshold_yi * 2e6
        if pd.notnull(avg_t):
            if mktcap_min_limit is not None:
                if float(avg_t) < mktcap_min_limit * 2_000_000:
                    continue
            if mktcap_max_limit is not None:
                if float(avg_t) > mktcap_max_limit * 2_000_000:
                    continue

        # ── 策略专属选股逻辑 ──
        if strategy == "rsv":
            rl = row.get("rsv_long");  rs = row.get("rsv_short"); vr = row.get("vol_ratio")
            if pd.notnull(rl) and pd.notnull(rs) and pd.notnull(vr):
                if rl >= 80 and rs <= 20 and vr > 0:
                    candidates.append({"stock": stock, "score": float(vr)})

        elif strategy == "brick":
            sig = row.get("buy_signal")
            if pd.notnull(sig) and sig:
                candidates.append({"stock": stock, "score": float(row.get("brick_score", 0))})

        elif strategy == "small_bank":
            # small_bank 自带板块过滤（北交所/科创/创业等已在顶层过滤，此处保留兜底）
            if stock.startswith(("4", "8", "9", "68", "30")):
                continue
            # 次新股已由通用过滤处理；若用户未勾选，仍执行策略内置的 375-day 过滤
            if new_min_days is None and row.get("days_listed", 0) < 375:
                continue
            turnover = row.get("avg_turnover20")
            if pd.notnull(turnover) and turnover > 0:
                candidates.append({"stock": stock, "score": float(turnover)})

    if strategy == "small_bank":
        bank_stocks = ["601988", "601398", "601288"]
        if _is_bank_period(current_date):
            bank_cands = []
            for b in bank_stocks:
                b_df = stock_data.get(b)
                if b_df is not None and current_date in b_df.index:
                    b_row = b_df.loc[current_date]
                    if b_row["close"] > 0:
                        prev_close = float(b_row.get("prev_close", 0.0))
                        if prev_close > 0:
                            ratio = float(b_row["close"]) / prev_close
                            bank_cands.append({"stock": b, "score": ratio})
            if bank_cands:
                bank_cands.sort(key=lambda x: x["score"])
                if len(bank_cands) > 1 and (bank_cands[-1]["score"] - bank_cands[0]["score"] > 0.005):
                    return [bank_cands[0]["stock"]]
                else:
                    holding_bank = [s for s in held_stocks if s in bank_stocks]
                    return holding_bank if holding_bank else [bank_cands[0]["stock"]]
            return []
        else:
            if not candidates: return []
            candidates.sort(key=lambda x: x["score"])
            return [c["stock"] for c in candidates[:available_slots]]

    if not candidates:
        return []

    if strategy == "rsv":
        # RSV 策略：取成交量放大倍数最小的（缩量）
        candidates.sort(key=lambda x: x["score"])
        return [c["stock"] for c in candidates[:available_slots]]
    else:
        # Brick：按得分从高到低排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return [c["stock"] for c in candidates[:available_slots]]



async def select_stocks_async(
    data_dir:       str,
    target_date:    str,
    strategy:       str = "brick",
    exclude_boards: list[str] = None,
    log_callback:   LogCallback = None
) -> list[dict]:
    """
    每日选股入口：根据指定日期，筛选出符合策略信号的公开股票。
    """
    async def log(level: str, msg: Any):
        if log_callback:
            await log_callback(level, msg)

    target_ts = pd.Timestamp(target_date)
    exclude_boards = exclude_boards or []

    await log("INFO", f"开始处理 {target_date} 的选股任务，策略：{strategy}，排除板块：{exclude_boards}...")

    # 1. 扫描文件
    csv_files: list[str] = []
    for root, _, files in os.walk(data_dir):
        if "大盘" in root: continue
        for f in files:
            if f.endswith(".csv") and f[:6].isdigit():
                code = f[:6]
                if "kc" in exclude_boards and code.startswith("688"): continue
                if "cy" in exclude_boards and (code.startswith("300") or code.startswith("301")): continue
                if "bse" in exclude_boards and (code.startswith("4") or code.startswith("8") or code.startswith("9")): continue
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        await log("WARN", "未找到有效的股票数据文件。")
        return []

    total_files = len(csv_files)
    await log("INFO", f"共扫描到 {total_files} 只个股，准备计算技术指标...")

    # 2. 并发加载 & 计算指标
    start_ts = target_ts - pd.Timedelta(days=120) 
    args_list = [(fp, strategy, start_ts, target_ts, exclude_boards) for fp in csv_files]

    stock_data:  dict[str, pd.DataFrame] = {}
    stock_names: dict[str, str]          = {}

    BATCH = 300
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        for batch_start in range(0, total_files, BATCH):
            batch = args_list[batch_start: batch_start + BATCH]
            results = await loop.run_in_executor(executor, _batch_load, batch)
            
            for code, name, df_dict in results:
                if df_dict is not None:
                    df = pd.DataFrame(df_dict)
                    df.set_index("date", inplace=True)
                    stock_data[code]  = df
                    stock_names[code] = name
            
            loaded = min(batch_start + BATCH, total_files)
            progress = int(loaded / total_files * 100)
            await log("DEBUG", f"指标计算进度: {loaded}/{total_files} ({progress}%)")
            await asyncio.sleep(0)

    await log("INFO", f"数据处理完成，正在执行全市场跨截面筛选...")

    # 3. 选股筛选
    selected_codes = _select_candidates(
        strategy, stock_data, {}, target_ts, available_slots=100,
        exclude_boards=exclude_boards
    )

    # 4. 封装结果
    final_results = []
    for code in selected_codes:
        df = stock_data[code]
        if target_ts not in df.index:
            continue
        row = df.loc[target_ts]
        final_results.append({
            "code": code,
            "name": stock_names.get(code, code),
            "close": round(float(row.get("close", 0)), 2),
            "score": round(float(row.get("brick_score", 0)) if strategy == "brick" else 0, 2),
            "signal": "买入"
        })

    await log("INFO", f"选股任务结束，共筛选出 {len(final_results)} 只符合条件的个股。")
    return final_results


# ─────────────────────── CLI 入口 ───────────────────────

def run_backtest(data_dir: str) -> None:
    asyncio.run(_run_backtest_wrapper(data_dir))


async def _run_backtest_wrapper(data_dir: str) -> None:
    async def print_callback(level: str, msg: Any) -> None:
        print(f"[{level}] {msg}")
    await run_backtest_async(data_dir, print_callback)


if __name__ == "__main__":
    run_backtest("e:/PythonProject/BacktestSystem/all_stock_data")
