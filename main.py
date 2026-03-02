"""
main.py — FastAPI 后端服务

提供：
  - GET /               首页
  - GET /report         绩效报告页
  - GET /history        回测历史页
  - GET /api/start_backtest       启动回测 (Server-Sent Events)
  - GET /api/get_report_data      获取上次回测结果
  - GET /api/history              获取历史回测列表
  - GET /api/history/{history_id} 获取某次历史回测详情
  - DELETE /api/history/{history_id} 删除某条历史记录
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backtest import run_backtest_async, STRATEGY_NAMES

app = FastAPI(title="A股回测分析工作台", version="1.0.0")

# ── 常量 ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
HISTORY_DIR = os.path.join(BASE_DIR, "backtest_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ── 全局状态 ──────────────────────────────────────────────
_last_result: dict = {"stats": {}, "trades": []}
_is_running   = False   # 防止并发重复提交


# ── 历史记录工具函数 ──────────────────────────────────────

def _save_history(
    history_id: str,
    strategy: str,
    start_date: str,
    end_date: str,
    max_positions: int,
    initial_capital: float,
    exclude_boards: list[str],
    result: dict,
    trades: list[dict],
) -> None:
    """将一次完整的回测结果写入 JSON 文件。"""
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    # 计算回测天数（自然日）
    try:
        d0 = datetime.date.fromisoformat(start_date)
        d1 = datetime.date.fromisoformat(end_date)
        period_days = (d1 - d0).days
    except Exception:
        period_days = 0

    payload = {
        "id":              history_id,
        "strategy":        strategy,
        "strategy_name":   STRATEGY_NAMES.get(strategy, strategy),
        "created_at":      created_at,
        "start_date":      start_date,
        "end_date":        end_date,
        "period_days":     period_days,
        "max_positions":   max_positions,
        "initial_capital": initial_capital,
        "exclude_boards":  exclude_boards,
        "stats":           result,
        "trades":          trades,
    }
    path = os.path.join(HISTORY_DIR, f"{history_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _list_history() -> list[dict]:
    """扫描历史目录，返回按创建时间倒序排列的摘要列表（不含 trades 明细）。"""
    records = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(HISTORY_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 只返回摘要字段，不含 trades（减少传输量）
            stats = data.get("stats", {})
            records.append({
                "id":              data.get("id"),
                "strategy":        data.get("strategy"),
                "strategy_name":   data.get("strategy_name"),
                "created_at":      data.get("created_at"),
                "start_date":      data.get("start_date"),
                "end_date":        data.get("end_date"),
                "period_days":     data.get("period_days"),
                "max_positions":   data.get("max_positions"),
                "initial_capital": data.get("initial_capital"),
                "exclude_boards":  data.get("exclude_boards", []),
                "total_trades":    stats.get("total_trades", 0),
                "win_rate":        stats.get("win_rate", 0),
                "avg_profit":      stats.get("avg_profit", 0),
                "max_profit":      stats.get("max_profit", 0),
                "max_loss":        stats.get("max_loss", 0),
                "final_capital":   stats.get("final_capital", 0),
                # 策略总收益率 = (final - initial) / initial * 100
                "strategy_return": round(
                    (stats.get("final_capital", data.get("initial_capital", 1))
                     - data.get("initial_capital", 1))
                    / max(data.get("initial_capital", 1), 1) * 100, 2
                ),
            })
        except Exception:
            continue
    records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return records


# ── 页面路由 ──────────────────────────────────────────────

def _read_html(filename: str) -> str:
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return _read_html("index.html")


@app.get("/report", response_class=HTMLResponse)
async def report():
    return _read_html("report.html")


@app.get("/history", response_class=HTMLResponse)
async def history_page():
    return _read_html("history.html")


# ── API 路由 ──────────────────────────────────────────────

@app.get("/api/get_report_data")
async def get_report_data():
    """返回最近一次回测的汇总统计与交易流水。"""
    if not _last_result["stats"]:
        raise HTTPException(status_code=404, detail="尚未运行任何回测，无数据可展示")
    return _last_result


@app.get("/api/history")
async def api_list_history():
    """返回所有历史回测的摘要列表（按创建时间倒序）。"""
    return JSONResponse(content=_list_history())


@app.get("/api/history/{history_id}")
async def api_get_history(history_id: str):
    """返回指定 ID 的历史回测完整数据（含 trades）。"""
    fpath = os.path.join(HISTORY_DIR, f"{history_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="历史记录不存在")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.delete("/api/history/{history_id}")
async def api_delete_history(history_id: str):
    """删除指定 ID 的历史回测记录。"""
    fpath = os.path.join(HISTORY_DIR, f"{history_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="历史记录不存在")
    os.remove(fpath)
    return {"ok": True}


@app.get("/api/start_backtest")
async def start_backtest(request: Request):
    """启动回测并通过 SSE 实时推送进度。"""
    global _is_running

    if _is_running:
        raise HTTPException(status_code=409, detail="当前已有回测任务正在运行，请稍后再试")

    # ── 解析参数 ──
    p = request.query_params
    start_date = p.get("start_date", "2026-01-02")
    end_date   = p.get("end_date",   "2026-02-27")
    strategy   = p.get("strategy",   "rsv").lower()

    default_max = {"rsv": 2, "ma": 5, "brick": 10}
    try:
        max_positions = int(p.get("max_positions", default_max.get(strategy, 5)))
    except (ValueError, TypeError):
        max_positions = default_max.get(strategy, 5)

    try:
        initial_capital = float(p.get("initial_capital", 100_000))
    except (ValueError, TypeError):
        initial_capital = 100_000.0

    exclude_boards_str = p.get("exclude", "")
    exclude_boards = [x.strip() for x in exclude_boards_str.split(",") if x.strip()]

    # 为本次回测生成唯一 ID
    history_id = uuid.uuid4().hex

    # ── SSE 消息队列 ──
    queue: asyncio.Queue = asyncio.Queue()

    async def log_callback(level: str, message) -> None:
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        await queue.put({"time": time_str, "level": level, "message": message})
        if level == "TRADE_RECORD":
            _last_result["trades"].append(message)

    async def backtest_task() -> None:
        global _is_running
        _is_running = True
        data_dir = os.path.join(BASE_DIR, "all_stock_data")
        try:
            _last_result["trades"] = []
            _last_result["stats"]  = {}

            result = await run_backtest_async(
                data_dir, log_callback,
                start_date, end_date, strategy,
                initial_capital, max_positions,
                exclude_boards
            )
            if result:
                _last_result["stats"] = result
                # ── 保存历史记录 ──
                _save_history(
                    history_id=history_id,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    max_positions=max_positions,
                    initial_capital=initial_capital,
                    exclude_boards=exclude_boards,
                    result=result,
                    trades=list(_last_result["trades"]),
                )
                await queue.put({
                    "time":    datetime.datetime.now().strftime("%H:%M:%S"),
                    "level":   "RESULT",
                    "message": json.dumps({**result, "history_id": history_id}, ensure_ascii=False),
                })
        except Exception as exc:
            await queue.put({
                "time":    datetime.datetime.now().strftime("%H:%M:%S"),
                "level":   "ERROR",
                "message": f"回测引擎异常：{exc}",
            })
        finally:
            _is_running = False
            await queue.put({"level": "DONE"})

    async def event_generator():
        asyncio.create_task(backtest_task())
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if msg.get("level") == "DONE":
                yield "data: DONE\n\n"
                break

            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 本地启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
