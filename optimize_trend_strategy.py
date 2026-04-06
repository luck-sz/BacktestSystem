from __future__ import annotations

import asyncio
import json
import os
import uuid
from itertools import product

from main import BASE_DIR, _run_backtest_with_cache, _save_history


DATA_DIR = os.path.join(BASE_DIR, "all_stock_data")
START_DATE = "2025-03-24"
END_DATE = "2026-03-24"
INITIAL_CAPITAL = 100000.0
TOP_N = 20

BASE_FILTERS = [
    "st",
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
            f"trend_fast_ma:{combo['fast_ma']}",
            f"trend_mid_ma:{combo['mid_ma']}",
            f"trend_slow_ma:{combo['slow_ma']}",
            f"trend_breakout_days:{combo['breakout_days']}",
            f"trend_strong_days:{combo['strong_days']}",
            f"trend_short_mom_min:{combo['short_mom_min']}",
            f"trend_long_mom_min:{combo['long_mom_min']}",
            f"trend_volume_mult:{combo['volume_mult']}",
            f"trend_volume_window:{combo['volume_window']}",
            f"trend_up_down_ratio:{combo['up_down_ratio']}",
            f"trend_pullback_volume_max:{combo['pullback_volume_max']}",
            f"trend_breakout_buffer:{combo['breakout_buffer']}",
            f"trend_gap_limit:{combo['gap_limit']}",
            f"trend_take_profit:{combo['take_profit']}",
            f"trend_stop_loss:{combo['stop_loss']}",
            f"trend_max_hold_days:{combo['max_hold_days']}",
        ]
    )
    if combo.get("exclude_kc"):
        filters.append("kc")
    if combo.get("exclude_cy"):
        filters.append("cy")
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
        strategy="trend_surfer",
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
        strategy="trend_surfer",
        start_date=START_DATE,
        end_date=END_DATE,
        max_positions=combo["max_positions"],
        initial_capital=INITIAL_CAPITAL,
        exclude_boards=excludes,
        result=result,
        trades=trades,
        history_meta={
            "source": "parameter_sweep",
            "optimizer": "optimize_trend_strategy.py",
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


async def main() -> None:
    results: list[dict] = []

    combos = []
    for (
        max_positions,
        breakout_days,
        long_mom_min,
        volume_mult,
        gap_limit,
        stop_loss,
        max_hold_days,
        volume_window,
        up_down_ratio,
        pullback_volume_max,
        exclude_kc,
    ) in product(
        [1, 2],
        [40, 55],
        [0.25, 0.35],
        [1.0, 1.3],
        [3.0, 5.0],
        [6.0, 8.0],
        [30, 45],
        [10, 15],
        [1.0, 1.15],
        [0.9, 1.0],
        [False, True],
    ):
        combos.append(
            {
                "max_positions": max_positions,
                "new_days": 250,
                "lowprice": 4,
                "highprice": 120,
                "lowvol": 5000,
                "mktcap_min": 20,
                "mktcap_max": 800,
                "fast_ma": 20,
                "mid_ma": 60,
                "slow_ma": 120,
                "breakout_days": breakout_days,
                "strong_days": 20,
                "short_mom_min": 0.10,
                "long_mom_min": long_mom_min,
                "volume_mult": volume_mult,
                "volume_window": volume_window,
                "up_down_ratio": up_down_ratio,
                "pullback_volume_max": pullback_volume_max,
                "breakout_buffer": 0.0,
                "gap_limit": gap_limit,
                "take_profit": 999.0,
                "stop_loss": stop_loss,
                "max_hold_days": max_hold_days,
                "exclude_kc": exclude_kc,
                "exclude_cy": False,
            }
        )

    total = len(combos)
    print(f"\n=== Trend Surfer Sweep | {total} combos ===")
    for idx, combo in enumerate(combos, start=1):
        item = await run_combo(combo, idx, total)
        if item:
            results.append(item)
            results.sort(key=score_result, reverse=True)
            print("top 5 so far:")
            for top in results[:5]:
                print(top["strategy_return"], top["params"], top["history_id"])

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
