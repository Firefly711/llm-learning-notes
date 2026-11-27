# -*- coding: utf-8 -*-
"""
W2-D5 · 把真实模型装进 FastAPI（异步 + 超时重试 + 错误映射）
==============================================================
昨天（D4）真实调用跑在控制台里，用的是同步 httpx.Client。
今天把它搬进服务，要解决四件工程问题——每一件都是面试会问的：

  1. 阻塞     同步 httpx 写在 async 路由里，会把整个事件循环卡住 → 用 httpx.AsyncClient
  2. 复用     每次请求新建客户端 = 每次重做 TCP/TLS 握手 → 用 lifespan 建一个全局 client
  3. 重试     429 / 5xx / 超时是"瞬时故障"要退避重试；401 / 402 / 400 是"确定性错误"重试没用
  4. 错误映射 上游的 401 不能直接透传给前端（那是你和供应商之间的事）→ 换成 502/503/504

本地无 key 也能跑：LLM_MODE 默认 mock，mock 里用 await asyncio.sleep(1.5) 模拟模型延迟，
所以并发实验照样能做出效果。

接口
----
  GET  /health  服务状态：用哪个模型、有没有 key、工具数量
  GET  /tools   工具声明
  POST /chat    对话（auto_execute 双模式，协议和 D3 完全一样）

跑法
----
  python -m uvicorn w2d5_fastapi_real_llm_demo:app --port 8020 --app-dir "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat"
  然后打开 http://127.0.0.1:8020/docs

环境变量
--------
  DEEPSEEK_API_KEY   有 key 且 LLM_MODE=real 才会真调模型
  DEEPSEEK_BASE_URL  默认 https://api.deepseek.com/v1
  DEEPSEEK_MODEL     默认 deepseek-chat
  LLM_MODE           real | mock，默认 mock
  MOCK_BLOCKING      1 = 用 time.sleep 模拟延迟（故意阻塞，用来对比并发退化）

【今天的核心实验】
  两个请求同时打进来：
    - 异步版（await asyncio.sleep）→ 总耗时 ≈ 单个耗时
    - 阻塞版（time.sleep）        → 总耗时 ≈ 两个加起来
  这就是"异步"在生产里的真实价值：不是快，是"等待时不占着人"。
"""
import asyncio
import json
import math
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===========================================================================
# 0. 配置
# ===========================================================================
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_MODE = os.getenv("LLM_MODE", "mock").lower()
MOCK_BLOCKING = os.getenv("MOCK_BLOCKING", "0") == "1"
MOCK_DELAY = float(os.getenv("MOCK_DELAY", "1.5"))
TIMEOUT = 30.0
MAX_RETRIES = 2
BACKOFF = 0.8

# 没有 key 时自动降级，别让服务起不来
if LLM_MODE == "real" and not KEY:
    print("⚠ LLM_MODE=real 但没有 DEEPSEEK_API_KEY，已降级为 mock。")
    LLM_MODE = "mock"


# ===========================================================================
# 1. 工具声明（和 D3/D4 一致：description 是模型唯一的说明书）
# ===========================================================================
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气，返回温度与天气状况。仅当用户明确询问天气时调用；未提供城市时必须先向用户确认，不要猜。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京、济南"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间。用户问当前时间、几点时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "做数学计算。用户给出算式时调用，如 2+3*4。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，只含数字和 + - * / ( ) ."}
                },
                "required": ["expression"],
            },
        },
    },
]


# ===========================================================================
# 2. 请求 / 响应模型
# ===========================================================================
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="消息角色")
    content: Optional[str] = Field(None, description="文本内容")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="assistant 的工具申请单")
    tool_call_id: Optional[str] = Field(None, description="role=tool 时必带")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    auto_execute: bool = Field(False, description="true=服务端跑完整个循环；false=只返回这一轮决定")
    max_turns: int = Field(5, ge=1, le=10)


class ToolCallOut(BaseModel):
    id: str
    type: Literal["function"] = "function"
    name: str
    arguments: Dict[str, Any]


class ChatResponse(BaseModel):
    # 注意：max_turns_exceeded 是 D3 自检第 3 题的标准答案——
    # 失败要显式失败，不能用"看起来正常的答案"掩盖
    type: Literal["tool_calls", "answer", "max_turns_exceeded"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallOut]] = None
    turns: int = 0
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    tokens: Optional[int] = Field(None, description="本次请求累计消耗的 token（真实模式才有）")
    model_mode: str = Field("mock", description="这一轮实际用的模型模式")


# ===========================================================================
# 3. 工具执行 —— 改成 async
#    真实项目里工具常常是"查数据库 / 调内部 API / 读文件"，都是 IO。
#    改成 async 之后，同一轮里的多个工具可以真正并行（asyncio.gather）。
# ===========================================================================
async def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    await asyncio.sleep(0.05)  # 模拟一次网络/DB 开销，让并行和串行的差别在日志里看得见

    if name == "get_weather":
        fake_weather = {"北京": "晴 22°C", "济南": "多云 21°C", "上海": "小雨 24°C", "成都": "阴 19°C"}
        city = arguments.get("city") or "未知"
        return json.dumps(
            {"city": city, "weather": fake_weather.get(city, "暂无数据"), "_source": "mock"},
            ensure_ascii=False,
        )

    if name == "get_current_time":
        return json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)

    if name == "calculator":
        expr = str(arguments.get("expression", "")).replace(" ", "")
        if not expr or not all(c in "0123456789+-*/()." for c in expr):
            return json.dumps({"error": "表达式含非法字符"}, ensure_ascii=False)
        try:
            value = eval(expr, {"__builtins__": {}}, {"math": math})  # 仅教学
            return json.dumps({"expression": expr, "result": round(float(value), 4)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"计算失败：{e}"}, ensure_ascii=False)

    return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)


# ===========================================================================
# 4. Mock 模型（无 key 可用）—— 关键：用 await asyncio.sleep 模拟延迟
# ===========================================================================
async def mock_step(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if MOCK_BLOCKING:
        time.sleep(MOCK_DELAY)   # ← 故意阻塞：事件循环被卡死，并发请求排队
    else:
        await asyncio.sleep(MOCK_DELAY)  # ← 正确：让出控制权，别的请求能进来

    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content") or ""
            break

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    if tool_msgs:
        parts = []
        for m in tool_msgs:
            try:
                data = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                data = {}
            if "weather" in data:
                parts.append(f"{data.get('city')}今天{data.get('weather')}")
            elif "result" in data:
                parts.append(f"{data.get('expression')} = {data.get('result')}")
            elif "time" in data:
                parts.append(f"现在是 {data.get('time')}")
            elif "error" in data:
                parts.append(f"工具报错：{data.get('error')}")
        return {"type": "answer", "content": ("；".join(parts) + "。") if parts else "没有拿到有效结果。"}

    calls: List[Dict[str, Any]] = []
    if "天气" in last_user:
        city = "济南"
        for c in ["北京", "上海", "广州", "深圳", "成都", "杭州"]:
            if c in last_user:
                city = c
                break
        calls.append({"name": "get_weather", "arguments": {"city": city}})
    if "时间" in last_user or "几点" in last_user:
        calls.append({"name": "get_current_time", "arguments": {}})
    if any(c in last_user for c in "+-*/"):
        expr = last_user
        for prefix in ("帮我算一下", "帮我算", "计算一下", "计算", "等于多少", "是多少"):
            expr = expr.replace(prefix, "").strip()
        if expr:
            calls.append({"name": "calculator", "arguments": {"expression": expr}})

    if calls:
        return {
            "type": "tool_calls",
            "tool_calls": [
                {"id": f"call_{i}", "name": c["name"], "arguments": c["arguments"]}
                for i, c in enumerate(calls)
            ],
        }
    return {"type": "answer", "content": f"（mock 直答）你说了：{last_user}。我没用到工具。"}


# ===========================================================================
# 5. 真实模型 —— 超时 + 重试 + 错误分类
# ===========================================================================
class UpstreamError(Exception):
    """上游调用失败。status 是上游返回码（0 表示网络/超时）。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def explain_status(code: int, body: str) -> str:
    if code == 401:
        return "认证失败（key 无效/过期）"
    if code == 402:
        return "余额不足"
    if code == 429:
        return "触发限流"
    if code == 400:
        return f"上游拒绝请求（多半是我们的 payload 有错）：{body[:200]}"
    if 500 <= code < 600:
        return f"上游服务异常 {code}"
    return f"HTTP {code}: {body[:200]}"


# 只有"瞬时故障"才值得重试。401/402/400 重试一百次也是错。
RETRYABLE = {429, 500, 502, 503, 504}


async def call_with_retry(client: httpx.AsyncClient, payload: dict) -> dict:
    """指数退避：0.8s → 1.6s，最多 2 次重试（总共 3 次尝试）。"""
    last: Optional[UpstreamError] = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
        except httpx.TimeoutException:
            last = UpstreamError(0, f"上游超时（>{TIMEOUT}s）")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF * (2 ** attempt))
                continue
            raise last
        except httpx.HTTPError as e:
            last = UpstreamError(0, f"网络错误：{e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF * (2 ** attempt))
                continue
            raise last

        if resp.status_code == 200:
            return resp.json()

        last = UpstreamError(resp.status_code, explain_status(resp.status_code, resp.text))
        if resp.status_code not in RETRYABLE:
            raise last            # 确定性错误，立刻失败
        if attempt < MAX_RETRIES:
            await asyncio.sleep(BACKOFF * (2 ** attempt) + random.uniform(0, 0.3))

    raise last or UpstreamError(0, "未知错误")


def clean_assistant_message(msg: dict) -> dict:
    """
    把上游返回的 assistant 消息"洗"成可以回传的最小结构。
    DeepSeek 可能带 reasoning_content，原样回传会被判非法字段 → 任务 400。
    """
    return {
        "role": "assistant",
        "content": msg.get("content"),
        "tool_calls": [
            {
                "id": tc["id"],
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],  # 保持字符串原样
                },
            }
            for tc in msg.get("tool_calls") or []
        ],
    }


async def real_step(client: httpx.AsyncClient, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
    data = await call_with_retry(client, payload)

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = (data.get("usage") or {}).get("total_tokens")

    if msg.get("tool_calls"):
        calls = []
        for tc in msg["tool_calls"]:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                # 关键：解析失败也要回传，让模型自己纠错，不能吞掉
                args = {"__parse_error__": raw}
            calls.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": args})
        return {
            "type": "tool_calls",
            "tool_calls": calls,
            "assistant_message": clean_assistant_message(msg),
            "usage": usage,
            "finish_reason": choice.get("finish_reason"),
        }

    return {
        "type": "answer",
        "content": msg.get("content") or "（模型没有返回内容）",
        "usage": usage,
        "finish_reason": choice.get("finish_reason"),
    }


async def model_step(client: Optional[httpx.AsyncClient], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统一入口：mock 和 real 返回同一种结构，上层循环完全不用改。"""
    if LLM_MODE == "real" and client is not None:
        return await real_step(client, messages)
    return await mock_step(messages)


# ===========================================================================
# 6. 错误映射 —— 你的 401 不是用户的 401
# ===========================================================================
def to_http_status(status: int) -> int:
    """上游错误 → 本服务的错误码。前端只该关心"我们这边出了什么问题"。"""
    if status in (401, 403):
        return 502   # 上游认证失败 = 我方网关与供应商之间的问题
    if status == 402:
        return 503   # 服务暂时不可用（供应商没钱了）
    if status == 429:
        return 503
    if status == 0:
        return 504   # 超时 / 网络
    if status == 400:
        return 500   # payload 是我们拼的，构造错了就是我们自己的 bug
    return 502


# ===========================================================================
# 7. Agent 循环（异步版）
# ===========================================================================
async def agent_loop(client: Optional[httpx.AsyncClient], history: List[Dict[str, Any]], max_turns: int) -> ChatResponse:
    steps: List[Dict[str, Any]] = []
    total_tokens = 0

    for turn in range(max_turns):
        decision = await model_step(client, history)
        total_tokens += decision.get("usage") or 0

        if decision["type"] == "answer":
            return ChatResponse(
                type="answer",
                content=decision["content"],
                turns=turn + 1,
                steps=steps,
                tokens=total_tokens if LLM_MODE == "real" else None,
                model_mode=LLM_MODE,
            )

        tcs = decision["tool_calls"]

        # ① 申请单进历史（真实模式用上游原样返回的消息；mock 自己拼）
        if decision.get("assistant_message"):
            history.append(decision["assistant_message"])
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                            },
                        }
                        for tc in tcs
                    ],
                }
            )

        # ② 并行执行：同一轮的多个工具真的同时跑
        results = await asyncio.gather(*[run_tool(tc["name"], tc["arguments"]) for tc in tcs])

        # ③ 逐条回传，各带各的 tool_call_id
        for tc, result in zip(tcs, results):
            steps.append(
                {
                    "turn": turn,
                    "tool": tc["name"],
                    "arguments": tc["arguments"],
                    "result": result,
                }
            )
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # 轮数用尽：显式失败，不伪装成答案
    return ChatResponse(
        type="max_turns_exceeded",
        content=f"达到最大轮数 {max_turns}，模型仍未收敛，已停止（防止死循环与继续烧 token）。",
        turns=max_turns,
        steps=steps,
        tokens=total_tokens if LLM_MODE == "real" else None,
        model_mode=LLM_MODE,
    )


# ===========================================================================
# 8. 应用与生命周期 —— client 建一次，全服务复用
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 整个进程只建一个 AsyncClient：连接池复用，避免每次请求重做 TCP/TLS 握手
    client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        timeout=TIMEOUT,
    ) if (LLM_MODE == "real" and KEY) else None

    app.state.client = client
    print(f"✓ 服务启动：LLM_MODE={LLM_MODE}  model={MODEL if LLM_MODE == 'real' else 'mock'}")
    yield
    if client is not None:
        await client.aclose()
    print("✓ 服务关闭：AsyncClient 已释放")


app = FastAPI(
    title="Real LLM Chat API",
    description="W2-D5：真实模型 + 异步 + 超时重试 + 错误映射",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", summary="服务状态")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "llm_mode": LLM_MODE,
        "model": MODEL if LLM_MODE == "real" else "mock",
        "has_key": bool(KEY),
        "tools": len(TOOLS),
        "timeout": TIMEOUT,
        "max_retries": MAX_RETRIES,
        "mock_blocking": MOCK_BLOCKING,
    }


@app.get("/tools", summary="工具声明")
def list_tools() -> Dict[str, Any]:
    return {"count": len(TOOLS), "tools": TOOLS}


@app.post("/chat", response_model=ChatResponse, summary="对话（真实模型 / mock）")
async def chat(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump(exclude_none=True) for m in req.messages]
    client: Optional[httpx.AsyncClient] = app.state.client

    try:
        if req.auto_execute:
            return await agent_loop(client, history, req.max_turns)

        decision = await model_step(client, history)
        if decision["type"] == "answer":
            return ChatResponse(
                type="answer",
                content=decision["content"],
                turns=1,
                tokens=decision.get("usage") if LLM_MODE == "real" else None,
                model_mode=LLM_MODE,
            )
        return ChatResponse(
            type="tool_calls",
            tool_calls=[ToolCallOut(**tc) for tc in decision["tool_calls"]],
            turns=1,
            tokens=decision.get("usage") if LLM_MODE == "real" else None,
            model_mode=LLM_MODE,
        )
    except UpstreamError as e:
        # 把上游错误翻译成本服务的错误码，并把真实原因写进日志/detail
        raise HTTPException(status_code=to_http_status(e.status), detail=f"上游调用失败：{e.message}")


# ===========================================================================
# 9. 自检（先自己答，答完再对答案）
# ===========================================================================
# Q1  为什么路由必须写成 async def、客户端必须用 httpx.AsyncClient？
#     如果在 async 路由里用 httpx.Client（同步），并发两个请求会发生什么？
#
# Q2  AsyncClient 为什么放在 lifespan 里建一次，而不是每个请求 new 一个？
#     至少说两个理由。
#
# Q3  哪些错误该重试、哪些绝对不该？把 429 / 401 / 402 / 500 / 超时 分个类。
#
# Q4  上游返回 401 时，为什么本服务要返回 502 而不是原样 401？
#     如果原样透传，前端会误判成什么？
#
# Q5  mock 里把 await asyncio.sleep 改成 time.sleep 之后，两个并发请求的
#     总耗时从 ~1.5s 变成 ~3s。用"事件循环在干什么"解释这个现象。
#
# 运行：
#   python -m uvicorn w2d5_fastapi_real_llm_demo:app --port 8020 --app-dir "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat"
