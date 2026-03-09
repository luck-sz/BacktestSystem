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
import sys

from typing import Any, Callable, Coroutine
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from backtest import run_backtest_async, select_stocks_async, STRATEGY_NAMES

app = FastAPI(title="A股回测分析工作台", version="1.0.0")

# ── 常量 ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
HISTORY_DIR = os.path.join(BASE_DIR, "backtest_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ── 全局状态 ──────────────────────────────────────────────
_is_running   = False   # 防止并发重复提交
_last_selection_results = {}
_last_result = {"trades": [], "stats": {}}


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


@app.get("/select", response_class=HTMLResponse)
async def select_page():
    return _read_html("select.html")


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


@app.get("/api/select_stocks")
async def api_select_stocks(request: Request):
    """获取最后一次选股结果 API。"""
    return JSONResponse(content=_last_selection_results)


@app.get("/api/start_select_stocks")
async def api_start_select_stocks(request: Request):
    """开启选股任务并推送进度 (SSE)。"""
    p = request.query_params
    target_date = p.get("date", "2026-03-02")
    strategy = p.get("strategy", "brick").lower()
    exclude_boards_str = p.get("exclude", "")
    exclude_boards = [x.strip() for x in exclude_boards_str.split(",") if x.strip()]

    async def event_generator():
        # 用于接收日志回调的对列
        queue = asyncio.Queue()

        async def log_cb(level: str, msg: Any):
            await queue.put({"level": level, "message": msg})

        # 在后台启动选股逻辑
        select_task = asyncio.create_task(select_stocks_async(
            os.path.join(BASE_DIR, "all_stock_data"),
            target_date,
            strategy,
            exclude_boards,
            log_cb
        ))

        while True:
            try:
                # 尝试从队列中获取消息，并设置超时，避免死循环造成 CPU 占用过高
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                pass

            if select_task.done():
                try:
                    results = await select_task
                    # 保存到全局状态供前端后续拉取
                    _last_selection_results.clear()
                    _last_selection_results.update({
                        "date": target_date,
                        "strategy": strategy,
                        "count": len(results),
                        "data": results,
                        "status": "success"
                    })
                    yield "data: DONE\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'level': 'ERROR', 'message': f'选股异常: {str(e)}'}, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stock_kline/{stock_code}")
async def api_stock_kline(stock_code: str):
    """获取指定股票的 K 线数据。"""
    data_dir = os.path.join(BASE_DIR, "all_stock_data")
    target_file = None
    
    # 查找文件
    for root, _, files in os.walk(data_dir):
        if f"{stock_code}.csv" in files:
            target_file = os.path.join(root, f"{stock_code}.csv")
            break
            
    if not target_file:
        raise HTTPException(status_code=404, detail=f"未找到股票 {stock_code} 的数据文件")
        
    try:
        from backtest import RENAME_MAP
        
        # 读取最近 200 天的数据用于展示
        df = pd.read_csv(target_file, usecols=lambda c: c in RENAME_MAP)
        df.rename(columns=RENAME_MAP, inplace=True)
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime('%Y-%m-%d')
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        
        # 1. 首先计算全量数据的指标，确保“预热”充足
        low_9 = df["low"].rolling(window=9).min()
        high_9 = df["high"].rolling(window=9).max()
        rsv = (df["close"] - low_9) / (high_9 - low_9).replace(0, 1e-9) * 100
        
        # KDJ 标准计算
        df["k"] = rsv.ewm(com=2, adjust=False).mean()
        df["d"] = df["k"].ewm(com=2, adjust=False).mean()
        df["j"] = 3 * df["k"] - 2 * df["d"]
        
        # 均线计算
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()
        
        # Brick 砖型指标计算
        hhv4   = df["high"].rolling(4).max()
        llv4   = df["low"].rolling(4).min()
        diff4  = (hhv4 - llv4).clip(lower=0.001)

        import numpy as np
        var1a  = (hhv4 - df["close"]) / diff4 * 100 - 90
        var2a  = var1a.ewm(alpha=1 / 4, adjust=False).mean() + 100
        var3a  = (df["close"] - llv4) / diff4 * 100
        var4a  = var3a.ewm(alpha=1 / 6, adjust=False).mean()
        var5a  = var4a.ewm(alpha=1 / 6, adjust=False).mean() + 100

        v6a         = (var5a - var2a).fillna(0)
        df["brick"] = np.where(v6a > 4, v6a - 4, 0.0)
        df["prev_brick"] = df["brick"].shift(1).fillna(0)
        
        # 2. 截取最近的 300 条
        df = df.tail(300)
        
        # 3. 填充计算初期产生的空值
        df = df.fillna(0)
        
        # 转换为列表 [date, open, close, low, high, volume, k, d, j, ma5, ma60, brick, prev_brick]
        chart_data = df[["date", "open", "close", "low", "high", "volume", "k", "d", "j", "ma5", "ma60", "brick", "prev_brick"]].values.tolist()
        
        if df.empty:
            return {"code": stock_code, "name": stock_code, "data": []}

        return {
            "code": stock_code,
            "name": str(df["name"].iloc[0]) if "name" in df.columns else stock_code,
            "data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析数据失败: {str(e)}")


# ── API 自动更新与检查 ──────────────────────────────────────

@app.get("/api/check_and_update")
async def api_check_and_update():
    """检查股票数据是否为最新，如果不是则运行 append_tencent.py 并推送进度"""
    from datetime import datetime
    import re
    
    now = datetime.now()
    target_date = now.strftime("%Y%m%d")
    data_dir = os.path.join(BASE_DIR, "all_stock_data")
    needs_update = False
    
    # 检查是否为工作日的 8:30 到 15:10 之间
    is_trading_hours = False
    if now.weekday() < 5:  # 0-4 分别是周一到周五
        if (now.hour == 8 and now.minute >= 30) or (9 <= now.hour <= 14) or (now.hour == 15 and now.minute <= 10):
            is_trading_hours = True
    
    lock_file = os.path.join(data_dir, ".update_lock")
    last_check_date = ""
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                last_check_date = f.read().strip()
        except:
            pass

    # 如果今天已经完整的跑过一次更新，或者正处于工作日盘中交易时段，则不触发自动刷新机制
    if last_check_date == target_date or is_trading_hours:
        needs_update = False
    else:
        sample_file = None
        if os.path.exists(data_dir):
            # 查找任意一个 csv 文件（可能在子目录下）
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.csv'):
                        sample_file = os.path.join(root, f)
                        break
                if sample_file:
                    break
                    
        if sample_file:
            try:
                df = pd.read_csv(sample_file, nrows=1, dtype=str)
                if not df.empty and '交易日期' in df.columns:
                    last_date = str(df['交易日期'].iloc[0]).strip()
                    if last_date < target_date:
                        needs_update = True
            except:
                needs_update = True
        else:
            needs_update = True

    async def event_generator():
        if not needs_update:
            yield f"data: {json.dumps({'status': 'updated'})}\\n\\n"
            return
            
        yield f"data: {json.dumps({'status': 'updating', 'progress': 0})}\\n\\n"
        
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "append_tencent.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=BASE_DIR
        )
        
        # tqdm 默认使用 \r 刷新，因此不能用 readline()
        async def read_until_cr_or_lf(stream):
            res = bytearray()
            while True:
                chunk = await stream.read(1)
                if not chunk:
                    break
                if chunk in (b'\\r', b'\\n'):
                    break
                res.extend(chunk)
            return res.decode('utf-8', errors='ignore')
            
        import re
        while True:
            line = await read_until_cr_or_lf(process.stdout)
            if not line:
                if process.stdout.at_eof():
                    break
                else:
                    await asyncio.sleep(0.01)
                    continue
            
            # 从 line 中解析类似 " 25%|" 的进度
            match = re.search(r'(\\d+)%', line)
            if match:
                pct = int(match.group(1))
                yield f"data: {json.dumps({'status': 'updating', 'progress': pct})}\\n\\n"
                
        await process.wait()
        
        # 记录今天已经检查/更新过
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(lock_file, "w") as f:
                f.write(target_date)
        except:
            pass
            
        yield f"data: {json.dumps({'status': 'done', 'progress': 100})}\\n\\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 本地启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
