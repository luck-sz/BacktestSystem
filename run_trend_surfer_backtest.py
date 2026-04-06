from __future__ import annotations

import asyncio
import json
import os
import uuid

from main import BASE_DIR, _run_backtest_with_cache, _save_history


START_DATE = "2021-03-24"
END_DATE = "2026-03-24"
INITIAL_CAPITAL = 100000.0
MAX_POSITIONS = 1

EXCLUDES = [
    "st",
    "kc",
    "bse",
    "suspended",
    "new:180",
    "lowprice:4",
    "highprice:160",
    "lowvol:5000",
    "mktcap_min:20",
    "mktcap_max:1000",
    "trend_fast_ma:10",
    "trend_mid_ma:30",
    "trend_slow_ma:90",
    "trend_breakout_days:30",
    "trend_strong_days:15",
    "trend_short_mom_min:0.08",
    "trend_long_mom_min:0.15",
    "trend_volume_mult:1.0",
    "trend_volume_window:10",
    "trend_up_down_ratio:1.0",
    "trend_pullback_volume_max:0.9",
    "trend_breakout_buffer:0.0",
    "trend_gap_limit:5.0",
    "trend_fast_exit_drawdown:0.0",
    "trend_trail_profit_trigger:20.0",
    "trend_trail_drawdown:10.0",
    "trend_take_profit:50.0",
    "trend_stop_loss:5.0",
    "trend_max_hold_days:45",
]


async def silent_log(level: str, message):
    return None


async def main() -> None:
    data_dir = os.path.join(BASE_DIR, "all_stock_data")
    result, trades, cache_hit, _, _ = await _run_backtest_with_cache(
        data_dir=data_dir,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy="trend_surfer",
        initial_capital=INITIAL_CAPITAL,
        max_positions=MAX_POSITIONS,
        exclude_boards=EXCLUDES,
        log_callback=silent_log,
    )
    if not result:
        print("No backtest result")
        return

    history_id = uuid.uuid4().hex
    _save_history(
        history_id=history_id,
        strategy="trend_surfer",
        start_date=START_DATE,
        end_date=END_DATE,
        max_positions=MAX_POSITIONS,
        initial_capital=INITIAL_CAPITAL,
        exclude_boards=EXCLUDES,
        result=result,
        trades=trades,
        history_meta={
            "source": "manual_run",
            "script": "run_trend_surfer_backtest.py",
            "cache_hit": cache_hit,
        },
    )

    payload = {
        "history_id": history_id,
        "cache_hit": cache_hit,
        "strategy_return": round(
            (result["final_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2
        ),
        "stats": result,
        "trade_count": len(trades),
        "sample_trades": trades[:10],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
