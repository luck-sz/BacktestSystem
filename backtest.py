"""
backtest.py — 通用 A股横截面回测引擎

支持策略：
  - rsv   : RSV背离 + 缩量轮动（短期超跌反弹）
  - brick : 砖型图打分（KDJ超卖 + MA60趋势 + 砖型柱反转）
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import numpy as np
import pandas as pd

from columnar_store import resolve_stock_files
from factor_store import get_factor_path

# ─────────────────────── 常量 ───────────────────────
RENAME_MAP = {
    "名称": "name",
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(手)": "volume",
    "涨跌幅(%)": "pct_chg",
    "前收盘价": "prev_close",
    "所属行业": "industry",
    "地域": "region",
}

STRATEGY_NAMES = {
    "rsv": "RSV超卖",
    "brick": "超跌反弹",
    "trend_surfer": "趋势龙头",
    "surge_relay": "放量回踩续涨",
}

BRICK_STRATEGIES = {"brick"}

MIN_ROWS = 25  # 数据行数过少则跳过
CACHE_DIRNAME = ".frame_cache"
CACHE_VERSION = 1
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "pct_chg", "prev_close")
LOAD_BATCH_SIZE = 400
MAP_CHUNK_SIZE = 8
POOL_WORKERS = max(1, min(os.cpu_count() or 1, 8))


def _cache_path_for(file_path: str) -> Path:
    src = Path(file_path)
    data_root = next(
        (parent for parent in src.parents if parent.name == "all_stock_data"), None
    )
    if data_root is None:
        return src.with_suffix(".pkl")
    return data_root / CACHE_DIRNAME / src.relative_to(data_root).with_suffix(".pkl")


def _read_cached_stock_frame(file_path: str) -> pd.DataFrame:
    src = Path(file_path)
    stat = src.stat()
    signature = {
        "version": CACHE_VERSION,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }
    cache_path = _cache_path_for(file_path)

    if cache_path.exists():
        try:
            payload = pd.read_pickle(cache_path)
            if payload.get("meta") == signature and isinstance(
                payload.get("data"), pd.DataFrame
            ):
                return payload["data"].copy()
        except Exception:
            pass

    if src.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
        if df.empty:
            return df
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df = pd.read_csv(file_path, usecols=lambda c: c in RENAME_MAP)
        if df.empty:
            return df

        df.rename(columns=RENAME_MAP, inplace=True)

        if "date" not in df.columns:
            return pd.DataFrame()

        raw_date = pd.to_numeric(df["date"], errors="coerce")
        df = df[raw_date.notna()].copy()
        if df.empty:
            return df

        df["date"] = pd.to_datetime(
            raw_date.loc[df.index].astype("int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

    df.dropna(subset=["date"], inplace=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in ("name", "industry", "region"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.to_pickle({"meta": signature, "data": df}, cache_path)
    except Exception:
        pass

    return df.copy()


def _read_factor_frame(file_path: str) -> pd.DataFrame | None:
    src = Path(file_path)
    data_root = next(
        (parent for parent in src.parents if parent.name == "all_stock_data"), None
    )
    if data_root is None:
        return None

    factor_path = get_factor_path(str(src), str(data_root))
    if not factor_path:
        return None

    try:
        return pd.read_parquet(factor_path)
    except Exception:
        return None


def _has_custom_brick_params(exclude_boards: list[str] | None) -> bool:
    return any(item.startswith("brick_") for item in (exclude_boards or []))


def _get_strategy_param(
    exclude_boards: list[str] | None,
    key: str,
    default: Any,
    caster: Callable[[str], Any],
    *,
    absolute: bool = False,
) -> Any:
    for item in exclude_boards or []:
        if not item.startswith(f"{key}:"):
            continue
        _, raw_value = item.split(":", 1)
        try:
            value = caster(raw_value)
            return abs(value) if absolute else value
        except Exception:
            return default
    return default


def _apply_precomputed_factors(
    df: pd.DataFrame,
    file_path: str,
    strategy: str,
    exclude_boards: list[str] | None = None,
) -> pd.DataFrame | None:
    if strategy in BRICK_STRATEGIES:
        return None

    factors = _read_factor_frame(file_path)
    if factors is None or factors.empty:
        return None

    merged = df.merge(factors, on="date", how="left", copy=False)

    if strategy == "rsv":
        return merged
    return None


def _map_batch_parallel(executor, batch: list[tuple]) -> list[tuple]:
    return list(executor.map(_load_single_file, batch, chunksize=MAP_CHUNK_SIZE))


def _create_selection_executor():
    """Windows 下避免 ProcessPool 拉起大量控制台窗口。"""
    if os.name == "nt":
        return ThreadPoolExecutor(max_workers=POOL_WORKERS)
    return ProcessPoolExecutor(max_workers=POOL_WORKERS)


# ─────────────────────── 指标计算 ───────────────────────


def _calc_rsv(
    df: pd.DataFrame, exclude_boards: Optional[list[str]] = None
) -> pd.DataFrame:
    """RSV背离策略指标"""
    low21 = df["low"].rolling(21, min_periods=1).min()
    high21 = df["close"].rolling(21, min_periods=1).max()
    df["rsv_long"] = (df["close"] - low21) / (high21 - low21 + 1e-9) * 100

    low3 = df["low"].rolling(3, min_periods=1).min()
    high3 = df["close"].rolling(3, min_periods=1).max()
    df["rsv_short"] = (df["close"] - low3) / (high3 - low3 + 1e-9) * 100

    df["vol_ratio"] = df["volume"] / df["volume"].shift(1).replace(0, np.nan)
    return df


def _calc_brick(
    df: pd.DataFrame, exclude_boards: Optional[list[str]] = None
) -> pd.DataFrame:
    """砖型图打分策略指标（通达信风格近似）"""
    exclude_boards = exclude_boards or []
    ma_short = _get_strategy_param(exclude_boards, "brick_ma_short", 6, int)
    ma_long = _get_strategy_param(exclude_boards, "brick_ma_long", 40, int)
    below_days = _get_strategy_param(exclude_boards, "brick_below_days", 5, int)
    callback_depth = _get_strategy_param(
        exclude_boards, "brick_callback", 1.0, float, absolute=True
    )
    explosive_power = _get_strategy_param(exclude_boards, "brick_power", 1.0, float)
    kdj_limit = _get_strategy_param(exclude_boards, "brick_kdj_limit", 0.0, float)
    breakout_buffer = _get_strategy_param(
        exclude_boards, "brick_breakout_buffer", 0.0, float
    )

    callback_mult = max(0.0, 1.0 - callback_depth / 100.0)

    df["ma5"] = df["close"].rolling(ma_short).mean()
    df["ma60"] = df["close"].rolling(ma_long).mean()

    low9 = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    diff9 = (high9 - low9).clip(lower=0.001)
    rsv = (df["close"] - low9) / diff9 * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    df["j"] = 3 * k - 2 * d

    hhv4 = df["high"].rolling(4).max()
    llv4 = df["low"].rolling(4).min()
    diff4 = (hhv4 - llv4).clip(lower=0.001)

    var1a = (hhv4 - df["close"]) / diff4 * 100 - 90
    var2a = var1a.ewm(alpha=1 / 4, adjust=False).mean() + 100
    var3a = (df["close"] - llv4) / diff4 * 100
    var4a = var3a.ewm(alpha=1 / 6, adjust=False).mean()
    var5a = var4a.ewm(alpha=1 / 6, adjust=False).mean() + 100

    v6a = (var5a - var2a).fillna(0)
    df["brick"] = np.where(v6a > 4, v6a - 4, 0.0)

    b0 = df["brick"]
    b1 = df["brick"].shift(1)
    b2 = df["brick"].shift(2)

    # 使用长度来判断爆发力，对齐通达信视觉效果
    red_len = b0 - b1
    green_len = b2 - b1

    color_ok = (b1 < b2) & (b0 > b1) & (b1 <= b2 * callback_mult)
    power_ok = red_len > (green_len * explosive_power)
    trend_ok = df["close"] > df["ma60"]
    oversold_ok = df["j"].shift(1).rolling(5).min() < kdj_limit

    is_below_ma5 = df["close"] < df["ma5"]
    below_ma5_5days = is_below_ma5.shift(1).rolling(below_days).sum() == below_days

    is_breakout = df["close"] >= df["ma5"] * (1 + breakout_buffer / 100.0)

    df["brick_score"] = red_len / (green_len + 0.001)
    df["buy_signal"] = color_ok & trend_ok & power_ok & oversold_ok & below_ma5_5days
    if breakout_buffer > 0:
        df["buy_signal"] = df["buy_signal"] & is_breakout
    return df


def _calc_trend_surfer(
    df: pd.DataFrame, exclude_boards: Optional[list[str]] = None
) -> pd.DataFrame:
    """趋势突破策略指标（中期趋势 + 新高突破 + 放量确认）"""
    exclude_boards = exclude_boards or []
    fast_ma = _get_strategy_param(exclude_boards, "trend_fast_ma", 20, int)
    mid_ma = _get_strategy_param(exclude_boards, "trend_mid_ma", 60, int)
    slow_ma = _get_strategy_param(exclude_boards, "trend_slow_ma", 120, int)
    breakout_lookback = _get_strategy_param(
        exclude_boards, "trend_breakout_days", 55, int
    )
    strong_window = _get_strategy_param(exclude_boards, "trend_strong_days", 20, int)
    short_mom_min = _get_strategy_param(
        exclude_boards, "trend_short_mom_min", 0.12, float
    )
    long_mom_min = _get_strategy_param(
        exclude_boards, "trend_long_mom_min", 0.25, float
    )
    volume_mult = _get_strategy_param(exclude_boards, "trend_volume_mult", 1.2, float)
    volume_window = _get_strategy_param(exclude_boards, "trend_volume_window", 15, int)
    up_down_ratio_min = _get_strategy_param(
        exclude_boards, "trend_up_down_ratio", 1.1, float
    )
    pullback_volume_max = _get_strategy_param(
        exclude_boards, "trend_pullback_volume_max", 0.95, float
    )
    buffer_pct = _get_strategy_param(
        exclude_boards, "trend_breakout_buffer", 0.0, float
    )

    df["ma_fast"] = df["close"].rolling(fast_ma).mean()
    df["ma_mid"] = df["close"].rolling(mid_ma).mean()
    df["ma_slow"] = df["close"].rolling(slow_ma).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["high_breakout"] = df["high"].shift(1).rolling(breakout_lookback).max()
    df["low_trail"] = df["low"].shift(1).rolling(10).min()
    df["high_strong"] = df["high"].rolling(strong_window).max()

    prev_close = df["close"].shift(1)
    day_ret = df["close"] / prev_close - 1
    df["mom_20"] = df["close"] / df["close"].shift(20) - 1
    df["mom_60"] = df["close"] / df["close"].shift(60) - 1
    df["pullback_from_high"] = df["close"] / df["high_strong"] - 1
    df["volume_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    up_volume = df["volume"].where(day_ret > 0, np.nan)
    down_volume = df["volume"].where(day_ret < 0, np.nan)
    df["up_volume_avg"] = up_volume.rolling(volume_window, min_periods=3).mean()
    df["down_volume_avg"] = down_volume.rolling(volume_window, min_periods=3).mean()
    df["up_down_volume_ratio"] = df["up_volume_avg"] / df["down_volume_avg"].replace(
        0, np.nan
    )
    recent_pullback = (day_ret.shift(1).rolling(5, min_periods=3).min() < 0) & (
        df["close"] >= df["ma_fast"]
    )
    df["pullback_volume_ratio"] = down_volume.rolling(5, min_periods=2).mean() / df[
        "vol_ma20"
    ].replace(0, np.nan)

    trend_ok = (
        (df["ma_fast"] > df["ma_mid"])
        & (df["ma_mid"] > df["ma_slow"])
        & (df["close"] > df["ma_fast"])
        & (df["ma_mid"] > df["ma_mid"].shift(5))
    )
    breakout_ok = df["close"] >= df["high_breakout"] * (1 + buffer_pct / 100.0)
    momentum_ok = (df["mom_20"] >= short_mom_min) & (df["mom_60"] >= long_mom_min)
    strength_ok = df["close"] >= df["high_strong"] * 0.97
    volume_ok = df["volume_ratio"] >= volume_mult
    structure_ok = df["up_down_volume_ratio"].fillna(0) >= up_down_ratio_min
    pullback_ok = (~recent_pullback) | (
        df["pullback_volume_ratio"].fillna(0) <= pullback_volume_max
    )
    limit_up_ok = (df["close"] / prev_close - 1).fillna(0) < 0.095

    df["buy_signal"] = (
        trend_ok
        & breakout_ok
        & momentum_ok
        & strength_ok
        & volume_ok
        & structure_ok
        & pullback_ok
        & limit_up_ok
    )
    df["trend_score"] = (
        df["mom_60"].fillna(0) * 100
        + df["mom_20"].fillna(0) * 80
        + df["volume_ratio"].fillna(0) * 10
        + df["up_down_volume_ratio"].fillna(0) * 15
        - df["pullback_volume_ratio"].fillna(0) * 10
        + df["pullback_from_high"].fillna(-1) * 50
    )
    return df


def _calc_surge_relay(
    df: pd.DataFrame, exclude_boards: Optional[list[str]] = None
) -> pd.DataFrame:
    """三日放量回踩续涨：阳线走强 -> 阴线回踩 -> 阳线放量续涨"""
    df["ma5"] = df["close"].rolling(5).mean()

    day1_vol = df["volume"].shift(2)
    day2_vol = df["volume"].shift(1)
    day3_vol = df["volume"]

    prev_close_day1 = df["close"].shift(3)
    prev_close_day2 = df["close"].shift(2)
    prev_close_day3 = df["close"].shift(1)

    day1_close = df["close"].shift(2)
    day2_close = df["close"].shift(1)
    day3_close = df["close"]
    day1_open = df["open"].shift(2)
    day2_open = df["open"].shift(1)

    three_day_vol = day1_vol + day2_vol + day3_vol
    day2_vol_share = (day2_vol / three_day_vol.replace(0, np.nan)).fillna(1)

    cond_day1 = (day1_close > day1_open) & (day1_close > prev_close_day1)
    cond_day2 = (
        (day2_close < day2_open)
        & (day2_close < prev_close_day2)
        & (day2_vol < day1_vol)
    )
    cond_day3 = (
        (day3_close > df["open"])
        & (day3_close > prev_close_day3)
        & (day3_vol > day1_vol)
        & (day3_close > df["ma5"])
    )

    df["buy_signal"] = cond_day1 & cond_day2 & cond_day3
    df["relay_score"] = 1 - day2_vol_share
    df["relay_defense"] = df["low"].shift(1).rolling(3).min()
    return df


_INDICATOR_FUNCS = {
    "rsv": _calc_rsv,
    "brick": _calc_brick,
    "trend_surfer": _calc_trend_surfer,
    "surge_relay": _calc_surge_relay,
}


# ─────────────────────── 单文件加载（同步，用于 executor） ───────────────────────


def _load_single_file(args: tuple) -> tuple[str, str, pd.DataFrame | None]:
    """
    同步地加载并预处理单只股票 CSV。
    返回 (stock_code, stock_name, df_or_None)
    """
    file_path, strategy, start_ts, end_ts, exclude_boards = args
    stock_code = Path(file_path).stem[:6]
    calc_fn = _INDICATOR_FUNCS.get(strategy)
    if calc_fn is None:
        return stock_code, "", None

    try:
        df = _read_cached_stock_frame(file_path)
        if df.empty or len(df) < MIN_ROWS:
            return stock_code, "", None

        stock_name = str(df["name"].iloc[-1]) if "name" in df.columns else stock_code

        if exclude_boards and "st" in exclude_boards:
            name_upper = stock_name.upper()
            if "ST" in name_upper or "退" in name_upper:
                return stock_code, stock_name, None

        total_trading_days = len(df)

        warmup_start = start_ts - pd.Timedelta(days=500)
        df = df[df["date"] >= warmup_start].copy()

        if df.empty or len(df) < MIN_ROWS:
            return stock_code, stock_name, None

        df["days_listed"] = total_trading_days - len(df) + np.arange(len(df))

        precomputed_df = _apply_precomputed_factors(
            df, file_path, strategy, exclude_boards
        )
        if precomputed_df is not None:
            df = precomputed_df
        else:
            df = calc_fn(df, exclude_boards)

        # 必须在计算指标之后再做精确的日期截取，保证滚动窗口完整且最终输出期间精准
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if df.empty:
            return stock_code, stock_name, None

        # 写入全历史天数，供 _select_candidates 过滤次新股
        df["total_trading_days"] = total_trading_days

        df.set_index("date", inplace=True)
        return stock_code, stock_name, df

    except Exception:
        return stock_code, "", None


# ─────────────────────── 主回测引擎 ───────────────────────

LogCallback = Callable[[str, Any], Coroutine]


async def run_backtest_async(
    data_dir: str,
    log_callback: LogCallback,
    start_date: str = "2026-01-02",
    end_date: str = "2026-02-27",
    strategy: str = "rsv",
    initial_capital: float = 100_000.0,
    max_positions: int = 2,
    exclude_boards: list[str] = None,
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
    await log_callback(
        "INFO", f"=== [策略启动] {strategy_label} ({start_date} 到 {end_date}) ==="
    )

    # ── 1. 扫描数据文件列表 ──
    exclude_boards = exclude_boards or []
    candidate_files = resolve_stock_files(data_dir, prefer_parquet=True)
    csv_files: list[str] = []
    for file_path in candidate_files:
        code = Path(file_path).stem[:6]
        if "kc" in exclude_boards and code.startswith("688"):
            continue
        if "cy" in exclude_boards and (
            code.startswith("300") or code.startswith("301")
        ):
            continue
        if "bse" in exclude_boards and (
            code.startswith("4") or code.startswith("8") or code.startswith("9")
        ):
            continue
        csv_files.append(file_path)

    if not csv_files:
        await log_callback("WARN", f"数据目录中未找到任何 CSV 文件：{data_dir}")
        return None

    total_files = len(csv_files)
    await log_callback(
        "INFO", f"开始加载数据: 共发现 {total_files} 只个股数据，准备全市场回测..."
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    args_list = [(fp, strategy, start_ts, end_ts, exclude_boards) for fp in csv_files]
    brick_gap_limit = _get_strategy_param(
        exclude_boards, "brick_gap_limit", 0.3, float, absolute=True
    )
    brick_take_profit = _get_strategy_param(
        exclude_boards, "brick_take_profit", 12.0, float, absolute=True
    )
    brick_stop_loss = _get_strategy_param(
        exclude_boards, "brick_stop_loss", 6.0, float, absolute=True
    )
    brick_max_hold_days = _get_strategy_param(
        exclude_boards, "brick_max_hold_days", 10, int, absolute=True
    )
    trend_gap_limit = _get_strategy_param(
        exclude_boards, "trend_gap_limit", 4.0, float, absolute=True
    )
    trend_fast_exit_drawdown = _get_strategy_param(
        exclude_boards, "trend_fast_exit_drawdown", 6.0, float, absolute=True
    )
    trend_trail_profit_trigger = _get_strategy_param(
        exclude_boards, "trend_trail_profit_trigger", 999.0, float, absolute=True
    )
    trend_trail_drawdown = _get_strategy_param(
        exclude_boards, "trend_trail_drawdown", 999.0, float, absolute=True
    )
    trend_take_profit = _get_strategy_param(
        exclude_boards, "trend_take_profit", 999.0, float, absolute=True
    )
    trend_stop_loss = _get_strategy_param(
        exclude_boards, "trend_stop_loss", 8.0, float, absolute=True
    )
    trend_max_hold_days = _get_strategy_param(
        exclude_boards, "trend_max_hold_days", 40, int, absolute=True
    )
    main_gap_limit = _get_strategy_param(
        exclude_boards, "main_gap_limit", 5.0, float, absolute=True
    )
    main_take_profit = _get_strategy_param(
        exclude_boards, "main_take_profit", 35.0, float, absolute=True
    )
    main_stop_loss = _get_strategy_param(
        exclude_boards, "main_stop_loss", 7.0, float, absolute=True
    )
    main_trail_trigger = _get_strategy_param(
        exclude_boards, "main_trail_trigger", 15.0, float, absolute=True
    )
    main_trail_drawdown = _get_strategy_param(
        exclude_boards, "main_trail_drawdown", 7.0, float, absolute=True
    )
    main_max_hold_days = _get_strategy_param(
        exclude_boards, "main_max_hold_days", 30, int, absolute=True
    )
    relay_gap_limit = _get_strategy_param(
        exclude_boards, "relay_gap_limit", 4.0, float, absolute=True
    )
    relay_take_profit = _get_strategy_param(
        exclude_boards, "relay_take_profit", 12.0, float, absolute=True
    )
    relay_stop_loss = _get_strategy_param(
        exclude_boards, "relay_stop_loss", 5.0, float, absolute=True
    )
    relay_max_hold_days = _get_strategy_param(
        exclude_boards, "relay_max_hold_days", 8, int, absolute=True
    )

    # ── 2. 并发预加载 & 指标计算（使用进程池避免 GIL） ──
    stock_data: dict[str, pd.DataFrame] = {}
    stock_names: dict[str, str] = {}
    all_dates: set = set()

    batch_size = LOAD_BATCH_SIZE

    with _create_selection_executor() as executor:
        for batch_start in range(0, total_files, batch_size):
            batch = args_list[batch_start : batch_start + batch_size]
            results = await asyncio.to_thread(_map_batch_parallel, executor, batch)
            for code, name, df_data in results:
                if df_data is not None:
                    stock_data[code] = df_data
                    stock_names[code] = name
                    all_dates.update(df_data.index)

            loaded = min(batch_start + batch_size, total_files)
            await log_callback(
                "DEBUG",
                f"仍在处理中: 已预加载并计算指标 {loaded}/{total_files} 只股票...",
            )
            await asyncio.sleep(0)  # 让事件循环有机会推送 SSE

    sorted_dates = sorted(all_dates)
    if not sorted_dates:
        await log_callback(
            "WARN", "在指定的日期范围内没有可用数据，请检查数据目录或日期范围。"
        )
        return None

    await log_callback(
        "INFO",
        f"数据加载完毕，共 {len(stock_data)} 只股票参与回测，回测日历 {len(sorted_dates)} 个交易日。",
    )
    await log_callback("INFO", "开始进行日级别横截面撮合与仿真...")

    # ── 3. 逐日撮合 ──
    trades: list[dict] = []
    held_stocks: dict[str, dict] = {}
    buy_list: list[str] = []
    daily_equity_curve: dict[str, float] = {}
    partial_equity_updates: dict[str, float] = {}
    current_equity = 1.0

    total_days = len(sorted_dates)

    for i, current_date in enumerate(sorted_dates):
        date_str = current_date.strftime("%Y-%m-%d")

        if i % 10 == 0:
            await log_callback(
                "DEBUG",
                f"回测进行中: 正在撮合 {date_str} 的交易 ({i}/{total_days} 天)...",
            )
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

            # 砖型图策略：过滤次日高开过猛，降低追高回撤
            if strategy in BRICK_STRATEGIES and i > 0:
                prev_date = sorted_dates[i - 1]
                if prev_date in df.index:
                    prev_close = float(df.loc[prev_date, "close"])
                    if prev_close > 0 and (open_price / prev_close - 1) > (
                        brick_gap_limit / 100.0
                    ):
                        continue

            if strategy == "trend_surfer" and i > 0:
                prev_date = sorted_dates[i - 1]
                if prev_date in df.index:
                    prev_close = float(df.loc[prev_date, "close"])
                    if prev_close > 0 and (open_price / prev_close - 1) > (
                        trend_gap_limit / 100.0
                    ):
                        continue

            if strategy == "surge_relay" and i > 0:
                prev_date = sorted_dates[i - 1]
                if prev_date in df.index:
                    prev_close = float(df.loc[prev_date, "close"])
                    if prev_close > 0 and (open_price / prev_close - 1) > (
                        relay_gap_limit / 100.0
                    ):
                        continue

            if np.isnan(open_price) or open_price <= 0:
                continue

            pos_value = (initial_capital * current_equity) / max_positions
            shares = max(int(pos_value / (open_price * 100)) * 100, 100)
            actual_cost = shares * open_price

            held_stocks[stock] = {
                "buy_date": current_date,
                "buy_price": open_price,
                "start_idx": i,
                "shares": shares,
                "buy_amount": actual_cost,
                "peak_close": open_price,
            }
            bought_today.append(stock)

        buy_list = []

        # ── 3b. 当日净值更新 ──
        daily_ret_sum = 0.0
        n_positions = len(held_stocks)

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
                    ret = (
                        (curr_close - prev_close) / prev_close
                        if prev_close > 0
                        else 0.0
                    )
                else:
                    ret = 0.0
            else:
                ret = 0.0
            daily_ret_sum += ret

        # 用实际持仓数量做分母（若空仓则分母为 max_positions 保证基准稳定）
        divisor = n_positions if n_positions > 0 else max_positions
        current_equity *= 1 + daily_ret_sum / divisor

        rounded_eq = round((current_equity - 1) * 100, 4)
        daily_equity_curve[date_str] = rounded_eq
        partial_equity_updates[date_str] = rounded_eq

        if len(partial_equity_updates) >= 20 or i == total_days - 1:
            await log_callback("EQUITY_UPDATE", partial_equity_updates)
            partial_equity_updates = {}

        # ── 3c. 尾盘卖出判定 ──
        to_remove: list[str] = []
        for stock, info in held_stocks.items():
            df = stock_data.get(stock)
            if df is None or current_date not in df.index:
                continue

            sell_price = float(df.loc[current_date, "close"])
            hold_days = i - info["start_idx"]
            profit_pct = (sell_price - info["buy_price"]) / info["buy_price"] * 100
            info["peak_close"] = max(
                float(info.get("peak_close", sell_price)), sell_price
            )

            sell_reason = None
            if strategy in BRICK_STRATEGIES:
                if hold_days >= 1:
                    close_p = float(df.loc[current_date, "close"])
                    ma_short = (
                        float(df.loc[current_date, "ma5"])
                        if "ma5" in df.columns
                        else 0.0
                    )

                    if profit_pct >= brick_take_profit:
                        sell_price = close_p
                        profit_pct = (
                            (sell_price - info["buy_price"]) / info["buy_price"] * 100
                        )
                        sell_reason = f"波段止盈触发(盈利>={brick_take_profit:.1f}%)"
                    elif profit_pct <= -brick_stop_loss:
                        sell_price = close_p
                        profit_pct = (
                            (sell_price - info["buy_price"]) / info["buy_price"] * 100
                        )
                        sell_reason = f"波段止损触发(亏损>={brick_stop_loss:.1f}%)"
                    elif hold_days >= brick_max_hold_days:
                        sell_price = close_p
                        profit_pct = (
                            (sell_price - info["buy_price"]) / info["buy_price"] * 100
                        )
                        sell_reason = f"持仓满{brick_max_hold_days}天波段了结"
                    elif ma_short > 0 and close_p < ma_short:
                        sell_price = close_p
                        profit_pct = (
                            (sell_price - info["buy_price"]) / info["buy_price"] * 100
                        )
                        sell_reason = "尾盘跌破短期均线(5日线)卖出"
            elif strategy == "trend_surfer":
                if hold_days >= 1:
                    close_p = float(df.loc[current_date, "close"])
                    ma_fast = float(df.loc[current_date, "ma_fast"] or 0.0)
                    ma_mid = float(df.loc[current_date, "ma_mid"] or 0.0)
                    low_trail = float(df.loc[current_date, "low_trail"] or 0.0)
                    peak_close = float(info.get("peak_close", close_p))
                    drawdown_pct = (
                        (close_p - peak_close) / peak_close * 100
                        if peak_close > 0
                        else 0.0
                    )

                    if profit_pct >= trend_take_profit:
                        sell_reason = f"趋势止盈触发(盈利>={trend_take_profit:.1f}%)"
                    elif profit_pct <= -trend_stop_loss:
                        sell_reason = f"趋势止损触发(亏损>={trend_stop_loss:.1f}%)"
                    elif hold_days >= trend_max_hold_days:
                        sell_reason = f"持仓满{trend_max_hold_days}天趋势了结"
                    elif (
                        profit_pct >= trend_trail_profit_trigger
                        and drawdown_pct <= -trend_trail_drawdown
                    ):
                        sell_reason = (
                            f"盈利达到{trend_trail_profit_trigger:.1f}%后回撤"
                            f">={trend_trail_drawdown:.1f}%离场"
                        )
                    elif (
                        ma_fast > 0
                        and close_p < ma_fast
                        and drawdown_pct <= -trend_fast_exit_drawdown
                    ):
                        sell_reason = "跌破快线且自高点回撤扩大"
                    elif low_trail > 0 and close_p < low_trail:
                        sell_reason = "跌破10日低点离场"
                    elif ma_mid > 0 and close_p < ma_mid:
                        sell_reason = "跌破中期趋势线离场"
            elif strategy == "surge_relay":
                if hold_days >= 1:
                    close_p = float(df.loc[current_date, "close"])
                    ma5 = float(df.loc[current_date, "ma5"] or 0.0)
                    defense = float(df.loc[current_date, "relay_defense"] or 0.0)

                    if profit_pct >= relay_take_profit:
                        sell_reason = (
                            f"续涨策略止盈触发(盈利>={relay_take_profit:.1f}%)"
                        )
                    elif profit_pct <= -relay_stop_loss:
                        sell_reason = f"续涨策略止损触发(亏损>={relay_stop_loss:.1f}%)"
                    elif hold_days >= relay_max_hold_days:
                        sell_reason = f"持仓满{relay_max_hold_days}天离场"
                    elif defense > 0 and close_p < defense:
                        sell_reason = "跌破三日防守位离场"
                    elif ma5 > 0 and close_p < ma5:
                        sell_reason = "跌破5日线离场"
            else:
                sell_reason = _judge_sell(strategy, hold_days, profit_pct)
                if not sell_reason and hold_days >= 5:
                    sell_reason = "持仓满5天兜底强制清仓"
            if sell_reason:
                trade_data = {
                    "代码": stock,
                    "名称": stock_names.get(stock, stock),
                    "买入日期": info["buy_date"].strftime("%Y-%m-%d"),
                    "买入价": round(info["buy_price"], 2),
                    "购买股数": info.get("shares", 0),
                    "购买金额": round(info.get("buy_amount", 0), 2),
                    "卖出日期": date_str,
                    "卖出价": round(sell_price, 2),
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
                strategy,
                stock_data,
                held_stocks,
                current_date,
                available_slots,
                exclude_boards,
            )

    # ── 4. 汇总统计 ──
    await log_callback("INFO", "=== 回测结束，正在汇总统计报告 ===")

    if not trades:
        await log_callback(
            "WARN", "没有产生任何有效交易，请检查策略参数或扩大回测时间范围。"
        )
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_trades = int((trades_df["收益率(%)"] > 0).sum())
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100
    avg_profit = float(trades_df["收益率(%)"].mean())
    max_profit = float(trades_df["收益率(%)"].max())
    max_loss = float(trades_df["收益率(%)"].min())

    await log_callback(
        "INFO",
        f"总交易次数: {total_trades} 次 | 胜率: {win_rate:.2f}% | 平均收益: {avg_profit:.2f}%",
    )

    return {
        "initial_capital": initial_capital,
        "final_capital": round(initial_capital * current_equity, 2),
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": round(win_rate, 2),
        "avg_profit": round(avg_profit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "daily_equity": daily_equity_curve,
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
    strategy: str,
    stock_data: dict[str, pd.DataFrame],
    held_stocks: dict[str, dict],
    current_date,
    available_slots: int,
    exclude_boards: list[str] | None = None,
) -> list[str]:
    """盘后选股，返回次日计划买入的股票列表"""
    exclude_boards = exclude_boards or []

    # ── 解析带参数的过滤条件 ──
    new_min_days = None
    lowprice_limit = None
    highprice_limit = None
    lowvol_limit = None  # 万元
    mktcap_min_limit = None  # 亿元
    mktcap_max_limit = None  # 亿元

    for item in exclude_boards:
        if item.startswith("new:"):
            try:
                new_min_days = int(item.split(":")[1])
            except:
                new_min_days = 365
        elif item.startswith("lowprice:"):
            try:
                lowprice_limit = float(item.split(":")[1])
            except:
                lowprice_limit = 3.0
        elif item.startswith("highprice:"):
            try:
                highprice_limit = float(item.split(":")[1])
            except:
                highprice_limit = 200.0
        elif item.startswith("lowvol:"):
            try:
                lowvol_limit = float(item.split(":")[1]) * 10000  # 万元转元
            except:
                lowvol_limit = 2000 * 10000
        elif item.startswith("mktcap_min:"):
            try:
                mktcap_min_limit = float(item.split(":")[1])
            except:
                mktcap_min_limit = 20.0
        elif item.startswith("mktcap_max:"):
            try:
                mktcap_max_limit = float(item.split(":")[1])
            except:
                mktcap_max_limit = 200.0

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
            vol = float(row.get("volume", 0) or 0)
            avg_t = close * vol  # 估算

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
            rl = row.get("rsv_long")
            rs = row.get("rsv_short")
            vr = row.get("vol_ratio")
            if pd.notnull(rl) and pd.notnull(rs) and pd.notnull(vr):
                if rl >= 80 and rs <= 20 and vr > 0:
                    candidates.append({"stock": stock, "score": float(vr)})

        elif strategy in BRICK_STRATEGIES:
            sig = row.get("buy_signal")
            if pd.notnull(sig) and sig:
                candidates.append(
                    {"stock": stock, "score": float(row.get("brick_score", 0))}
                )

        elif strategy == "trend_surfer":
            sig = row.get("buy_signal")
            if pd.notnull(sig) and sig:
                candidates.append(
                    {"stock": stock, "score": float(row.get("trend_score", 0))}
                )

        elif strategy == "surge_relay":
            sig = row.get("buy_signal")
            if pd.notnull(sig) and sig:
                candidates.append(
                    {"stock": stock, "score": float(row.get("relay_score", 0))}
                )

    if not candidates:
        return []

    if strategy == "rsv":
        # RSV 策略：取成交量放大倍数最小的（缩量）
        candidates.sort(key=lambda x: x["score"])
        return [c["stock"] for c in candidates[:available_slots]]
    elif strategy in BRICK_STRATEGIES:
        # Brick：暂无特定排序或随机取，可以按 score 从高到低
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return [c["stock"] for c in candidates[:available_slots]]
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return [c["stock"] for c in candidates[:available_slots]]


async def select_stocks_async(
    data_dir: str,
    target_date: str,
    strategy: str = "brick",
    exclude_boards: list[str] = None,
    log_callback: LogCallback = None,
) -> list[dict]:
    """
    每日选股入口：根据指定日期，筛选出符合策略信号的公开股票。
    """

    async def log(level: str, msg: Any):
        if log_callback:
            await log_callback(level, msg)

    target_ts = pd.Timestamp(target_date)
    exclude_boards = exclude_boards or []

    await log(
        "INFO",
        f"开始处理 {target_date} 的选股任务，策略：{strategy}，排除板块：{exclude_boards}...",
    )

    # 1. 扫描文件
    candidate_files = resolve_stock_files(data_dir, prefer_parquet=False)
    csv_files: list[str] = []
    for file_path in candidate_files:
        code = Path(file_path).stem[:6]
        if "kc" in exclude_boards and code.startswith("688"):
            continue
        if "cy" in exclude_boards and (
            code.startswith("300") or code.startswith("301")
        ):
            continue
        if "bse" in exclude_boards and (
            code.startswith("4") or code.startswith("8") or code.startswith("9")
        ):
            continue
        csv_files.append(file_path)

    if not csv_files:
        await log("WARN", "未找到有效的股票数据文件。")
        return []

    total_files = len(csv_files)
    await log("INFO", f"共扫描到 {total_files} 只个股，准备计算技术指标...")

    # 2. 并发加载 & 计算指标
    start_ts = target_ts - pd.Timedelta(days=500)
    args_list = [
        (fp, strategy, start_ts, target_ts, exclude_boards) for fp in csv_files
    ]

    stock_data: dict[str, pd.DataFrame] = {}
    stock_names: dict[str, str] = {}

    batch_size = LOAD_BATCH_SIZE
    with _create_selection_executor() as executor:
        for batch_start in range(0, total_files, batch_size):
            batch = args_list[batch_start : batch_start + batch_size]
            results = await asyncio.to_thread(_map_batch_parallel, executor, batch)

            for code, name, df_data in results:
                if df_data is not None:
                    stock_data[code] = df_data
                    stock_names[code] = name

            loaded = min(batch_start + batch_size, total_files)
            progress = int(loaded / total_files * 100)
            await log("DEBUG", f"指标计算进度: {loaded}/{total_files} ({progress}%)")
            await asyncio.sleep(0)

    await log("INFO", f"数据处理完成，正在执行全市场跨截面筛选...")

    # 3. 选股筛选
    selected_codes = _select_candidates(
        strategy,
        stock_data,
        {},
        target_ts,
        available_slots=100,
        exclude_boards=exclude_boards,
    )

    # 4. 封装结果
    final_results = []
    for code in selected_codes:
        df = stock_data[code]
        if target_ts not in df.index:
            continue
        row = df.loc[target_ts]
        # 获取涨跌幅（优先取 pct_chg 列，若无则尝试动态计算）
        pct_val = row.get("pct_chg")
        if pd.isna(pct_val) or pct_val == 0 or pct_val == "":
            close_p = float(row.get("close", 0))
            prev_p = float(row.get("prev_close", 0))
            if prev_p > 0:
                pct_val = (close_p / prev_p - 1) * 100
            else:
                pct_val = 0

        if strategy in BRICK_STRATEGIES:
            score_value = float(row.get("brick_score", 0))
        elif strategy == "trend_surfer":
            score_value = float(row.get("trend_score", 0))
        elif strategy == "surge_relay":
            score_value = float(row.get("relay_score", 0))
        else:
            score_value = 0.0

        final_results.append(
            {
                "code": code,
                "name": stock_names.get(code, code),
                "close": round(float(row.get("close", 0)), 2),
                "pct_chg": round(float(pct_val), 2),
                "score": round(score_value, 2),
                "signal": "买入",
                "industry": str(row.get("industry", "")),
                "region": str(row.get("region", "")),
            }
        )

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
