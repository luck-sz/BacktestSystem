from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

from main import BASE_DIR, _run_backtest_with_cache, _save_history


START_DATE = "2021-03-24"
END_DATE = "2026-03-24"
INITIAL_CAPITAL = 100000.0


def build_excludes(combo: dict) -> list[str]:
    filters = ["st", "bse", "suspended"]
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


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_trend_surfer_combo.py '<json combo>'")

    combo = json.loads(sys.argv[1])
    data_dir = os.path.join(BASE_DIR, "all_stock_data")
    excludes = build_excludes(combo)
    result, trades, cache_hit, _, _ = await _run_backtest_with_cache(
        data_dir=data_dir,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy="trend_surfer",
        initial_capital=INITIAL_CAPITAL,
        max_positions=combo["max_positions"],
        exclude_boards=excludes,
        log_callback=silent_log,
    )
    if not result:
        print(json.dumps({"ok": False, "combo": combo}, ensure_ascii=False))
        return

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
            "source": "manual_combo",
            "script": "run_trend_surfer_combo.py",
            "cache_hit": cache_hit,
        },
    )
    payload = {
        "ok": True,
        "history_id": history_id,
        "cache_hit": cache_hit,
        "strategy_return": round(
            (result["final_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2
        ),
        "final_capital": round(result["final_capital"], 2),
        "total_trades": result.get("total_trades", 0),
        "win_rate": result.get("win_rate", 0),
        "avg_profit": result.get("avg_profit", 0),
        "max_profit": result.get("max_profit", 0),
        "max_loss": result.get("max_loss", 0),
        "combo": combo,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
