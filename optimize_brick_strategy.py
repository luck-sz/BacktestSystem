from __future__ import annotations

import asyncio
import json
import os
import uuid
from itertools import product

from main import BASE_DIR, _run_backtest_with_cache, _save_history


DATA_DIR = os.path.join(BASE_DIR, "all_stock_data")
START_DATE = "2025-03-22"
END_DATE = "2026-03-20"
INITIAL_CAPITAL = 100000.0
TOP_N = 20

BASE_FILTERS = [
    "st",
    "kc",
    "cy",
    "bse",
    "suspended",
]


def build_excludes(combo: dict) -> list[str]:
    filters = list(BASE_FILTERS)
    filters.extend(
        [
            f"new:{combo['new_days']}",
            f"lowprice:{combo['lowprice']}",
            f"highprice:{combo['highprice']}",
            f"lowvol:{combo['lowvol']}",
            f"mktcap_min:{combo['mktcap_min']}",
            f"mktcap_max:{combo['mktcap_max']}",
            f"brick_ma_short:{combo['ma_short']}",
            f"brick_ma_long:{combo['ma_long']}",
            f"brick_below_days:{combo['below_days']}",
            f"brick_callback:{combo['callback']}",
            f"brick_power:{combo['power']}",
            f"brick_kdj_limit:{combo['kdj_limit']}",
            f"brick_gap_limit:{combo['gap_limit']}",
            f"brick_take_profit:{combo['take_profit']}",
            f"brick_stop_loss:{combo['stop_loss']}",
            f"brick_max_hold_days:{combo['max_hold_days']}",
        ]
    )
    if combo.get("breakout_buffer", 0):
        filters.append(f"brick_breakout_buffer:{combo['breakout_buffer']}")
    return filters


async def silent_log(level: str, message):
    return None


def score_result(item: dict) -> tuple:
    return (
        item["strategy_return"],
        item["win_rate"],
        item["avg_profit"],
        -item["max_loss"],
    )


async def run_combo(combo: dict, index: int, total: int) -> dict | None:
    excludes = build_excludes(combo)
    print(f"[{index}/{total}] testing {json.dumps(combo, ensure_ascii=False)}")
    result, trades, cache_hit, _, _ = await _run_backtest_with_cache(
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy="brick",
        initial_capital=INITIAL_CAPITAL,
        max_positions=combo["max_positions"],
        exclude_boards=excludes,
        log_callback=silent_log,
    )
    if not result:
        print("  -> no trades")
        return None

    history_id = uuid.uuid4().hex
    _save_history(
        history_id=history_id,
        strategy="brick",
        start_date=START_DATE,
        end_date=END_DATE,
        max_positions=combo["max_positions"],
        initial_capital=INITIAL_CAPITAL,
        exclude_boards=excludes,
        result=result,
        trades=trades,
        history_meta={
            "source": "parameter_sweep",
            "optimizer": "optimize_brick_strategy.py",
            "cache_hit": cache_hit,
        },
    )
    strategy_return = round(
        (result["final_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        2,
    )
    item = {
        "history_id": history_id,
        "cache_hit": cache_hit,
        "params": combo,
        "strategy_return": strategy_return,
        "win_rate": result.get("win_rate", 0),
        "avg_profit": result.get("avg_profit", 0),
        "max_profit": result.get("max_profit", 0),
        "max_loss": result.get("max_loss", 0),
        "total_trades": result.get("total_trades", 0),
        "final_capital": result.get("final_capital", 0),
    }
    print(
        "  -> "
        f"return={item['strategy_return']}% win={item['win_rate']}% "
        f"trades={item['total_trades']} max_loss={item['max_loss']}% history={item['history_id']}"
    )
    return item


async def run_stage(name: str, combos: list[dict], results: list[dict]) -> list[dict]:
    print(f"\n=== {name} | {len(combos)} combos ===")
    total = len(combos)
    for idx, combo in enumerate(combos, start=1):
        item = await run_combo(combo, idx, total)
        if item:
            results.append(item)
    results.sort(key=score_result, reverse=True)
    print("top 5 after stage:")
    for item in results[:5]:
        print(item["strategy_return"], item["params"], item["history_id"])
    return results


async def main() -> None:
    results: list[dict] = []

    stage1 = []
    for max_positions, ma_short, ma_long, below_days, callback, power in product(
        [4, 6, 8],
        [5, 6],
        [40, 60],
        [3, 5],
        [0.5, 1.0],
        [0.8, 1.0],
    ):
        stage1.append(
            {
                "max_positions": max_positions,
                "new_days": 250,
                "lowprice": 3,
                "highprice": 60,
                "lowvol": 3000,
                "mktcap_min": 20,
                "mktcap_max": 300,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "below_days": below_days,
                "callback": callback,
                "power": power,
                "kdj_limit": 0,
                "gap_limit": 0.5,
                "take_profit": 12,
                "stop_loss": 6,
                "max_hold_days": 12,
                "breakout_buffer": 0,
            }
        )
    await run_stage("Stage 1 Entry Structure", stage1, results)

    if not results:
        print("No valid results")
        return

    best = dict(results[0]["params"])

    stage2 = []
    for lowprice, lowvol, mktcap_min, mktcap_max, gap_limit in product(
        [3, 4, 5],
        [2000, 3000, 5000],
        [20, 30, 50],
        [200, 300, 500],
        [0.3, 0.5, 1.0],
    ):
        combo = dict(best)
        combo.update(
            {
                "lowprice": lowprice,
                "lowvol": lowvol,
                "mktcap_min": mktcap_min,
                "mktcap_max": mktcap_max,
                "gap_limit": gap_limit,
            }
        )
        stage2.append(combo)
    await run_stage("Stage 2 Market Filters", stage2, results)

    best = dict(results[0]["params"])
    stage3 = []
    for take_profit, stop_loss, max_hold_days, kdj_limit in product(
        [8, 12, 16, 24],
        [4, 6, 8],
        [6, 8, 12, 16],
        [-10, 0],
    ):
        combo = dict(best)
        combo.update(
            {
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold_days": max_hold_days,
                "kdj_limit": kdj_limit,
            }
        )
        stage3.append(combo)
    await run_stage("Stage 3 Exit Tuning", stage3, results)

    best = dict(results[0]["params"])
    stage4 = []
    for callback, power, breakout_buffer, new_days in product(
        [0.5, 1.0, 1.5],
        [0.8, 1.0, 1.2],
        [0, 0.3, 0.5],
        [180, 250, 365],
    ):
        combo = dict(best)
        combo.update(
            {
                "callback": callback,
                "power": power,
                "breakout_buffer": breakout_buffer,
                "new_days": new_days,
            }
        )
        stage4.append(combo)
    await run_stage("Stage 4 Fine Tune", stage4, results)

    results.sort(key=score_result, reverse=True)
    print("\n=== Final Top Results ===")
    for item in results[:TOP_N]:
        print(
            item["strategy_return"],
            item["win_rate"],
            item["avg_profit"],
            item["max_loss"],
            item["total_trades"],
            item["params"],
            item["history_id"],
        )


if __name__ == "__main__":
    asyncio.run(main())
