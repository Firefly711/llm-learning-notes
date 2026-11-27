# -*- coding: utf-8 -*-
"""
W2-D6 · 给接口上锁：API Key 鉴权 + 限流
=========================================
D5 的接口已经能调真实模型了，但它现在是个"裸奔"的服务：
任何人只要知道 http://127.0.0.1:8020 就能让它替你调模型、烧你的余额。

今天加两道闸门：

  1. 鉴权（Authentication）—— 你是谁？
     没有 key → 401；key 不对 → 401。用 X-API-Key 头。
     比较用 secrets.compare_digest（防时序攻击），不是 == 。

  2. 限流（Rate Limiting）—— 你能用多少？
     每个 key 一个滑动窗口，超了 → 429 + Retry-After。
     按 key 限流而不是按 IP：IP 会变、会共享，key 才是"配额主体"。

【今天最重要的设计认知】
鉴权是"横切关注点"——它不该改业务逻辑。
所以本文件 **import 了 D5 的实现**，业务代码一行没动，
只在路由签名里加了一个 `key: str = Depends(guard)`。

跑法
----
  python -m uvicorn w2d6_auth_ratelimit_demo:app --port 8024 --app-dir "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat"
  打开 http://127.0.0.1:8024/docs ，右上角有 Authorize 按钮，填 API Key 后再试。

环境变量
--------
  APP_API_KEYS   逗号分隔的合法 key，默认 "dev-key-001"
  RATE_LIMIT     窗口内最多几次，默认 5
  RATE_WINDOW    窗口秒数，默认 60
  LLM_MODE       同 D5：real | mock（由 D5 模块读取）

【面试要点】
  - 401 是"你没资格"，429 是"你有资格但先歇会儿"，别混用
  - 单进程内存限流在 --workers 4 下会失效（每个 worker 各算各的）→ 生产用 Redis
"""
import asyncio
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

# 业务逻辑直接复用 D5：鉴权不该重写业务代码
import w2d5_fastapi_real_llm_demo as core

# ===========================================================================
# 1. 配置
# ===========================================================================
API_KEYS = [k.strip() for k in os.getenv("APP_API_KEYS", "dev-key-001").split(",") if k.strip()]
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))


def mask(key: str) -> str:
    """日志里绝不打印明文 key。生产上连掩码都要谨慎，这里保留头尾便于人工确认。"""
    if not key:
        return "-"
    return key[:4] + "…" + key[-3:] if len(key) > 8 else "***"


# ===========================================================================
# 2. 鉴权 + 限流
# ===========================================================================
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,              # 自己抛 401，才能带上 WWW-Authenticate 头
    description="在下方填入你的 API Key（本地默认 dev-key-001）",
)

_hits: Dict[str, deque] = {}
_lock = asyncio.Lock()


async def check_rate(key: str) -> None:
    """滑动窗口：每个 key 记录最近 RATE_WINDOW 秒内的请求时间戳。"""
    now = time.monotonic()
    async with _lock:                       # 计数器是共享状态，必须加锁
        dq = _hits.setdefault(key, deque())
        while dq and now - dq[0] >= RATE_WINDOW:
            dq.popleft()                    # 滑出窗口的旧请求作废

        if len(dq) >= RATE_LIMIT:
            retry_after = int(RATE_WINDOW - (now - dq[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"超出限额：{RATE_WINDOW} 秒内最多 {RATE_LIMIT} 次，请 {retry_after} 秒后重试",
                headers={"Retry-After": str(retry_after)},
            )
        dq.append(now)


async def guard(key: Optional[str] = Security(api_key_header)) -> str:
    """所有受保护路由的统一入口：先认证，再限流。"""
    if not key:
        raise HTTPException(
            status_code=401,
            detail="缺少 X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # 常量时间比较：避免通过响应时间差猜出 key 的前几位
    if not any(secrets.compare_digest(key, valid) for valid in API_KEYS):
        raise HTTPException(
            status_code=401,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    await check_rate(key)
    return key


# ===========================================================================
# 3. 应用
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = (
        httpx.AsyncClient(
            headers={"Authorization": f"Bearer {core.KEY}", "Content-Type": "application/json"},
            timeout=core.TIMEOUT,
        )
        if (core.LLM_MODE == "real" and core.KEY)
        else None
    )
    app.state.client = client
    print(
        f"✓ 服务启动：LLM_MODE={core.LLM_MODE}  鉴权=开（{len(API_KEYS)} 个 key）  "
        f"限流={RATE_LIMIT} 次/{RATE_WINDOW}s"
    )
    yield
    if client is not None:
        await client.aclose()
        print("✓ 服务关闭：AsyncClient 已释放")


app = FastAPI(
    title="Secured Chat API",
    description="W2-D6：API Key 鉴权 + 滑动窗口限流",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request, call_next):
    """访问日志：时间、状态、耗时、key 掩码。明文 key 永远不进日志。"""
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    key = request.headers.get("x-api-key")
    print(f"  {request.method:4s} {request.url.path:10s} → {resp.status_code}  {dt:6.0f}ms  key={mask(key)}")
    return resp


@app.get("/health", summary="服务状态（不需要鉴权）")
def health() -> dict:
    return {
        "status": "ok",
        "llm_mode": core.LLM_MODE,
        "model": core.MODEL if core.LLM_MODE == "real" else "mock",
        "auth": "enabled",
        "keys_configured": len(API_KEYS),
        "rate_limit": {"limit": RATE_LIMIT, "window_seconds": RATE_WINDOW},
    }


@app.get("/tools", summary="工具声明（需要鉴权）")
def list_tools(key: str = Depends(guard)) -> dict:
    return {"count": len(core.TOOLS), "tools": core.TOOLS}


@app.post("/chat", response_model=core.ChatResponse, summary="对话（需要鉴权）")
async def chat(req: core.ChatRequest, key: str = Depends(guard)) -> core.ChatResponse:
    history = [m.model_dump(exclude_none=True) for m in req.messages]
    client: Optional[httpx.AsyncClient] = app.state.client

    try:
        if req.auto_execute:
            return await core.agent_loop(client, history, req.max_turns)

        decision = await core.model_step(client, history)
        if decision["type"] == "answer":
            return core.ChatResponse(
                type="answer",
                content=decision["content"],
                turns=1,
                tokens=decision.get("usage") if core.LLM_MODE == "real" else None,
                model_mode=core.LLM_MODE,
            )
        return core.ChatResponse(
            type="tool_calls",
            tool_calls=[core.ToolCallOut(**tc) for tc in decision["tool_calls"]],
            turns=1,
            tokens=decision.get("usage") if core.LLM_MODE == "real" else None,
            model_mode=core.LLM_MODE,
        )
    except core.UpstreamError as e:
        raise HTTPException(status_code=core.to_http_status(e.status), detail=f"上游调用失败：{e.message}")


# ===========================================================================
# 4. 自检（先自己答）
# ===========================================================================
# Q1  /health 为什么故意不鉴权？如果它也要 key，会带来什么麻烦？
#
# Q2  比较 key 为什么用 secrets.compare_digest，而不是 key == valid？
#     举一个"用 == 会漏什么"的具体场景。
#
# Q3  这套内存限流，在 `uvicorn --workers 4` 启动时会怎样？
#     为什么？生产上应该怎么改？
#
# Q4  401 和 429 对客户端来说，重试策略应该有什么区别？
#     为什么 401 不能重试、429 应该重试？
#
# Q5  访问日志里为什么不能打印明文 key？即使"只有我自己看日志"也不行？
