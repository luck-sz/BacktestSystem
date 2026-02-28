from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
import json
import datetime
import os
from backtest import run_backtest_async

app = FastAPI()

# 全局变量缓存最近一次的回测结果，用于发送给绩效报告页面
last_backtest_result = {
    "stats": {},
    "trades": []
}

# 挂载首页路由
@app.get("/")
async def read_index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# 新增：挂载绩效报告页路由
@app.get("/report")
async def read_report():
    path = os.path.join(os.path.dirname(__file__), "report.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# 新增：报告页用来索要数据的接口
@app.get("/api/get_report_data")
async def get_report_data():
    if not last_backtest_result["stats"]:
        return {"error": "尚未运行任何回测，无数据可展示"}
    return last_backtest_result

# 触发回测的后端接口 (SSE)
@app.get("/api/start_backtest")
async def start_backtest(request: Request):
    start_date = request.query_params.get("start_date", "2026-01-02")
    end_date = request.query_params.get("end_date", "2026-02-27")
    strategy = request.query_params.get("strategy", "rsv")
    
    try:
        max_positions = int(request.query_params.get("max_positions", "2" if strategy == "rsv" else "5"))
    except:
        max_positions = 2 if strategy == "rsv" else 5
    
    try:
        initial_capital = float(request.query_params.get("initial_capital", "100000"))
    except:
        initial_capital = 100000.0

    queue = asyncio.Queue()
    
    async def log_callback(level, message):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        await queue.put({"time": time_str, "level": level, "message": message})
        
        # 为了保留所有的流水记录发送给 report 页面，如果属于流水数据，则静默保存
        if level == "TRADE_RECORD":
            last_backtest_result["trades"].append(message)

    async def backtest_task():
        data_dir = os.path.join(os.path.dirname(__file__), "all_stock_data")
        try:
            # 清空上一轮的缓存
            last_backtest_result["trades"] = []
            last_backtest_result["stats"] = {}
            
            result = await run_backtest_async(data_dir, log_callback, start_date, end_date, strategy, initial_capital, max_positions)
            if result:
                last_backtest_result["stats"] = result
                await queue.put({"time": datetime.datetime.now().strftime("%H:%M:%S"), "level": "RESULT", "message": json.dumps(result)})
        except Exception as e:
            await queue.put({"time": datetime.datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": str(e)})
        finally:
            await queue.put({"level": "DONE"})

    async def event_generator():
        task = asyncio.create_task(backtest_task())
        while True:
            try:
                # 最多等5秒，超时就发一个keepalive心跳包，防止浏览器断线
                msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # 发送注释行作为心跳（SSE规范：以冒号开头的行为注释，浏览器会忽略）
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止Nginx缓冲，确保实时推送
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
