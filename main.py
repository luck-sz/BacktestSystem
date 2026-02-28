"""
main.py — FastAPI 后端服务

提供：
  - GET /               首页
  - GET /report         绩效报告页
  - GET /api/start_backtest   启动回测 (Server-Sent Events)
  - GET /api/get_report_data  获取上次回测结果
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from backtest import run_backtest_async

app = FastAPI(title="A股回测分析工作台", version="1.0.0")

# ── 全局状态 ──────────────────────────────────────────────
_last_result: dict = {"stats": {}, "trades": []}
_is_running   = False   # 防止并发重复提交


# ── 页面路由 ──────────────────────────────────────────────

def _read_html(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return _read_html("index.html")


@app.get("/report", response_class=HTMLResponse)
async def report():
    return _read_html("report.html")


# ── API 路由 ──────────────────────────────────────────────

@app.get("/api/get_report_data")
async def get_report_data():
    """返回最近一次回测的汇总统计与交易流水。"""
    if not _last_result["stats"]:
        raise HTTPException(status_code=404, detail="尚未运行任何回测，无数据可展示")
    return _last_result


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
    strategy   = p.get("strategy",   "rsv")

    default_max = {"rsv": 2, "ma": 5, "brick": 10}
    try:
        max_positions = int(p.get("max_positions", default_max.get(strategy, 5)))
    except (ValueError, TypeError):
        max_positions = default_max.get(strategy, 5)

    try:
        initial_capital = float(p.get("initial_capital", 100_000))
    except (ValueError, TypeError):
        initial_capital = 100_000.0

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
        data_dir = os.path.join(os.path.dirname(__file__), "all_stock_data")
        try:
            _last_result["trades"] = []
            _last_result["stats"]  = {}

            result = await run_backtest_async(
                data_dir, log_callback,
                start_date, end_date, strategy,
                initial_capital, max_positions,
            )
            if result:
                _last_result["stats"] = result
                await queue.put({
                    "time":    datetime.datetime.now().strftime("%H:%M:%S"),
                    "level":   "RESULT",
                    "message": json.dumps(result, ensure_ascii=False),
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
                # SSE 心跳，防止浏览器超时断连
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
            "X-Accel-Buffering": "no",   # 禁止 Nginx 缓冲，确保实时推送
        },
    )


# ── 本地启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
