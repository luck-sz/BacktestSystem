from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import uuid

from main import BASE_DIR, _run_backtest_with_cache, _save_history


DATA_DIR = os.path.join(BASE_DIR, "all_stock_data")
START_DATE = "2021-03-24"
END_DATE = "2026-03-24"
INITIAL_CAPITAL = 100000.0
TOP_N = 15
RANDOM_SEED = 42
SAMPLE_SIZE = 60

BASE_FILTERS = ["st", "bse", "suspended"]


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
            f"trend_fast_exit_drawdown:{combo.get('fast_exit_drawdown', 6.0)}",
            f"trend_trail_profit_trigger:{combo.get('trail_profit_trigger', 999.0)}",
            f"trend_trail_drawdown:{combo.get('trail_drawdown', 999.0)}",
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
        item["final_capital"],
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
            "optimizer": "optimize_trend_surfer_5y.py",
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


def build_combos() -> list[dict]:
    curated = [
        {
            "max_positions": 1,
            "new_days": 250,
            "lowprice": 4,
            "highprice": 120,
            "lowvol": 5000,
            "mktcap_min": 20,
            "mktcap_max": 800,
            "fast_ma": 20,
            "mid_ma": 60,
            "slow_ma": 120,
            "breakout_days": 40,
            "strong_days": 20,
            "short_mom_min": 0.10,
            "long_mom_min": 0.25,
            "volume_mult": 1.0,
            "volume_window": 15,
            "up_down_ratio": 1.15,
            "pullback_volume_max": 0.9,
            "breakout_buffer": 0.0,
            "gap_limit": 5.0,
            "take_profit": 999.0,
            "stop_loss": 6.0,
            "max_hold_days": 45,
            "exclude_kc": True,
            "exclude_cy": False,
        },
        {
            "max_positions": 1,
            "new_days": 180,
            "lowprice": 3,
            "highprice": 200,
            "lowvol": 3000,
            "mktcap_min": 10,
            "mktcap_max": 1500,
            "fast_ma": 10,
            "mid_ma": 30,
            "slow_ma": 60,
            "breakout_days": 20,
            "strong_days": 10,
            "short_mom_min": 0.05,
            "long_mom_min": 0.12,
            "volume_mult": 0.8,
            "volume_window": 10,
            "up_down_ratio": 0.95,
            "pullback_volume_max": 1.0,
            "breakout_buffer": 0.0,
            "gap_limit": 8.0,
            "take_profit": 999.0,
            "stop_loss": 8.0,
            "max_hold_days": 60,
            "exclude_kc": False,
            "exclude_cy": False,
        },
        {
            "max_positions": 2,
            "new_days": 120,
            "lowprice": 4,
            "highprice": 160,
            "lowvol": 2000,
            "mktcap_min": 10,
            "mktcap_max": 800,
            "fast_ma": 10,
            "mid_ma": 30,
            "slow_ma": 90,
            "breakout_days": 30,
            "strong_days": 15,
            "short_mom_min": 0.08,
            "long_mom_min": 0.15,
            "volume_mult": 0.8,
            "volume_window": 10,
            "up_down_ratio": 1.0,
            "pullback_volume_max": 1.0,
            "breakout_buffer": 0.0,
            "gap_limit": 8.0,
            "take_profit": 999.0,
            "stop_loss": 8.0,
            "max_hold_days": 45,
            "exclude_kc": False,
            "exclude_cy": False,
        },
    ]

    rng = random.Random(RANDOM_SEED)
    sampled = []
    while len(sampled) < SAMPLE_SIZE:
        combo = {
            "max_positions": rng.choice([1, 2]),
            "new_days": rng.choice([120, 180, 250]),
            "lowprice": rng.choice([3, 4, 6]),
            "highprice": rng.choice([120, 160, 220]),
            "lowvol": rng.choice([2000, 5000, 10000]),
            "mktcap_min": rng.choice([10, 20]),
            "mktcap_max": rng.choice([500, 800, 1500]),
            "fast_ma": rng.choice([10, 20]),
            "mid_ma": rng.choice([30, 60]),
            "slow_ma": rng.choice([90, 120]),
            "breakout_days": rng.choice([20, 30, 40]),
            "strong_days": rng.choice([10, 15, 20]),
            "short_mom_min": rng.choice([0.05, 0.1]),
            "long_mom_min": rng.choice([0.12, 0.2, 0.3]),
            "volume_mult": rng.choice([0.8, 1.0, 1.2]),
            "volume_window": rng.choice([10, 15]),
            "up_down_ratio": rng.choice([0.95, 1.0, 1.15]),
            "pullback_volume_max": rng.choice([0.85, 0.95, 1.05]),
            "breakout_buffer": 0.0,
            "gap_limit": rng.choice([5.0, 8.0]),
            "take_profit": 999.0,
            "stop_loss": rng.choice([6.0, 8.0, 10.0]),
            "max_hold_days": rng.choice([30, 45, 60]),
            "exclude_kc": rng.choice([False, True]),
            "exclude_cy": rng.choice([False, True]),
        }
        if not (combo["fast_ma"] < combo["mid_ma"] < combo["slow_ma"]):
            continue
        if combo["breakout_days"] < combo["strong_days"]:
            continue
        sampled.append(combo)
    seen = set()
    combos = []
    for combo in curated + sampled:
        key = json.dumps(combo, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)
    return combos


async def main() -> None:
    results: list[dict] = []
    combos = build_combos()
    total = len(combos)
    print(f"\n=== Trend Surfer 5Y Sweep | {total} combos ===")

    for idx, combo in enumerate(combos, start=1):
        item = await run_combo(combo, idx, total)
        if not item:
            continue
        results.append(item)
        results.sort(key=score_result, reverse=True)
        best = results[0]
        print(
            f"best_so_far={best['strategy_return']}% trades={best['total_trades']} history={best['history_id']}"
        )

    results.sort(key=score_result, reverse=True)
    print("\n=== Final Top Results ===")
    for item in results[:TOP_N]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
