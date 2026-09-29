# -*- coding: utf-8 -*-
"""
Day 1 · FastAPI 入门：路由 + Pydantic
=====================================
今日主线A任务。跟着注释敲一遍，跑起来，再回答文末 3 个问题。
运行方式：  uvicorn day1_fastapi_demo:app --reload
打开文档：  http://127.0.0.1:8000/docs   （FastAPI 自动生成的 Swagger 交互文档）
"""
from datetime import date
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(
    title="Day1 学习接口",
    description="FastAPI 路由 + Pydantic 校验入门示例（AI 应用工程师方向）",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# 一、Pydantic 模型：声明"数据长什么样"，FastAPI 自动做校验 / 文档 / 序列化
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    """单条对话消息。字段类型 + 约束 = 自动校验规则。"""
    role: str = Field(..., description="system / user / assistant / tool")
    content: str = Field(..., min_length=1, max_length=4096)


class ChatRequest(BaseModel):
    """POST /chat 的请求体：校验不通过会直接返回 422，无需手写判断。"""
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="采样温度 0~2")
    max_tokens: int = Field(1024, ge=1, le=8192)
    tools: Optional[List[dict]] = Field(None, description="function calling 工具 schema 列表")


class ChatResponse(BaseModel):
    """响应模型：只暴露想暴露的字段，避免内部结构泄漏。"""
    reply: str
    tool_calls: List[str] = []
    token_hint: int


# ---------------------------------------------------------------------------
# 二、路由：路径参数 + 查询参数 + 请求体，三种传参方式各就各位
# ---------------------------------------------------------------------------
@app.get("/")
def health() -> dict:
    """GET /：健康检查。"""
    return {"status": "ok", "today": str(date.today())}


@app.get("/items/{item_id}")
def get_item(item_id: int, q: Optional[str] = Query(None, max_length=50)):
    """
    路径参数 item_id 是 int → 传非数字自动 422；
    查询参数 q 可选，带长度校验。返回 dict 会被自动序列化成 JSON。
    """
    return {"item_id": item_id, "q": q, "note": "路径参数 + 查询参数示例"}


# ---------------------------------------------------------------------------
# 三、业务接口：把 Pydantic 模型当参数，请求体校验、响应校验自动完成
# ---------------------------------------------------------------------------
def _mock_llm(messages: List[ChatMessage], temperature: float) -> str:
    """占位用的"模型"：真实项目中替换成调用 Claude/GPT/DeepSeek API。"""
    last = messages[-1].content
    if "天气" in last:
        return "（示例回复）我查一下天气再告诉你 —— 接入真实 LLM 后这里会调用 function calling。"
    return f"已收到你的消息（{len(messages)} 条上下文，temperature={temperature}）：{last[:40]}"


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    今日重点：POST + 请求体 + 响应模型三件套。
    1) req 由 FastAPI 按 ChatRequest 自动解析、校验（422 自带错误详情）；
    2) 返回对象按 ChatResponse 校验并过滤字段；
    3) 在线文档 /docs 里可以直接点 Try it out 测试。
    """
    reply = _mock_llm(req.messages, req.temperature)
    used_tools = [t.get("function", {}).get("name", "?") for t in (req.tools or [])]
    return ChatResponse(reply=reply, tool_calls=used_tools, token_hint=max(10, len(reply) * 2))


# ---------------------------------------------------------------------------
# 四、课后自检（回答完这 3 题，今天 FastAPI 部分就算过）
# ---------------------------------------------------------------------------
# 1. 把 /items/{item_id} 改成 /items/{item_id:str} 会怎样？为什么要限定类型？
# 2. ChatRequest 里 temperature 加了 ge=0.0/le=2.0，传 3 会发生什么？错误码是多少？
# 3. response_model=ChatResponse 的作用是什么？如果不加会有什么风险？
#
# 试运行：先 GET http://127.0.0.1:8000/items/5?q=hello
#        再 POST http://127.0.0.1:8000/chat  {"messages":[{"role":"user","content":"今天济南天气怎么样"}]}
