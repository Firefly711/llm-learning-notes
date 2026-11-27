# -*- coding: utf-8 -*-
"""
W2-D3 · 把 Function Calling 装进 FastAPI
==========================================
今天的核心：昨天那条链路（tool_calls → 执行 → role=tool 回传 → 总结）
不再是控制台里的 print，而是一个真正的 HTTP 服务。

两个接口：
  GET  /tools   查看服务端声明了哪些工具（模型能看到的"说明书"）
  POST /chat    对话入口，有两个模式：
      auto_execute = false  → 无状态模式：只返回模型这一轮的决定
                              （tool_calls 或最终回答），工具由客户端执行。
                              这是 OpenAI / Anthropic 的真实产品形态。
      auto_execute = true   → 托管模式：服务端自己跑完整个循环并执行工具，
                              直接返回最终答案 + 每一步轨迹。方便测试和演示。

跑法（注意 --app-dir，避免加载错目录的同名文件）：
  python -m uvicorn w2d3_fastapi_tools_demo:app --port 8000 --app-dir "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat"
  然后打开 http://127.0.0.1:8000/docs

自检要点见文件末尾注释。
"""
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Function Calling API",
    description="W2-D3 练习：把模型决策 + 工具执行 + 结果回传做成 HTTP 接口",
    version="0.1.0",
)

# ===========================================================================
# 1. 工具声明 —— 模型唯一能看到的东西（和昨天完全一样，照搬即可）
# ===========================================================================
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气。用户问天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京、济南"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间。用户问时间、几点时调用。",
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
# 2. 请求 / 响应模型 —— 这里的每个 Literal 都是一道校验
# ===========================================================================
class Message(BaseModel):
    """一条对话消息。兼容 OpenAI 的 messages 结构。"""

    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="消息角色")
    content: Optional[str] = Field(None, description="文本内容；assistant 只申请工具时可为空")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="assistant 消息里模型返回的工具申请单"
    )
    tool_call_id: Optional[str] = Field(None, description="role=tool 时必须带上，对应哪张申请单")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, description="完整对话历史")
    auto_execute: bool = Field(
        False, description="true=服务端自动执行工具并跑完循环；false=只返回模型这一轮的决定"
    )
    max_turns: int = Field(5, ge=1, le=10, description="最大循环轮数，防止死循环烧钱")


class ToolCallOut(BaseModel):
    id: str
    type: Literal["function"] = "function"
    name: str
    arguments: Dict[str, Any]


class ChatResponse(BaseModel):
    type: Literal["tool_calls", "answer"] = Field(..., description="本轮模型的决定类型")
    content: Optional[str] = Field(None, description="type=answer 时是最终回答")
    tool_calls: Optional[List[ToolCallOut]] = Field(None, description="type=tool_calls 时的申请单")
    turns: int = Field(0, description="实际循环了几轮（auto_execute=false 时恒为 1）")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="工具执行轨迹，便于排查")


# ===========================================================================
# 3. 工具执行 —— 和昨天一模一样：模型进不来
# ===========================================================================
def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "get_weather":
        fake_weather = {"北京": "晴 22°C", "济南": "多云 21°C", "上海": "小雨 24°C", "成都": "阴 19°C"}
        city = arguments.get("city") or "未知"
        weather = fake_weather.get(city, "暂无数据")
        return json.dumps({"city": city, "weather": weather}, ensure_ascii=False)

    if name == "get_current_time":
        return json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)

    if name == "calculator":
        expr = str(arguments.get("expression", "")).replace(" ", "")
        if not expr or not all(c in "0123456789+-*/()." for c in expr):
            return json.dumps({"error": "表达式含非法字符"}, ensure_ascii=False)
        try:
            value = eval(expr, {"__builtins__": {}}, {"math": math})  # 仅教学，生产用表达式解析库
            return json.dumps({"expression": expr, "result": round(float(value), 4)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"计算失败：{e}"}, ensure_ascii=False)

    return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)


# ===========================================================================
# 4. Mock 模型 —— 和昨天相比唯一的升级：支持"并行工具调用"
# ===========================================================================
def mock_model(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """返回 {"type":"tool_calls","tool_calls":[...]} 或 {"type":"answer","content":...}"""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content") or ""
            break

    # ① 已有工具结果 → 汇总成最终回答（真实模型也是在拿到 role=tool 后才总结）
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
        return {"type": "answer", "content": "；".join(parts) + "。" if parts else "没有拿到有效结果。"}

    # ② 没有工具结果 → 判断这一轮要调哪些工具（可以多个 = 并行调用）
    calls: List[Dict[str, Any]] = []

    if "天气" in last_user:
        city = "济南"  # 注意：这是 mock 的默认值，真实模型可能编造或反问
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
                {"id": f"call_{i}", "type": "function", "name": c["name"], "arguments": c["arguments"]}
                for i, c in enumerate(calls)
            ],
        }

    # ③ 不需要工具 → 直接回答
    return {"type": "answer", "content": f"（模型直接回答）你说了：{last_user}。我没用到工具。"}


# ===========================================================================
# 5. Agent 循环 —— 昨天控制台里的那个 for，现在成了服务端的核心逻辑
# ===========================================================================
def agent_loop(history: List[Dict[str, Any]], max_turns: int) -> ChatResponse:
    steps: List[Dict[str, Any]] = []

    for turn in range(max_turns):
        decision = mock_model(history)

        if decision["type"] == "answer":
            return ChatResponse(type="answer", content=decision["content"], turns=turn + 1, steps=steps)

        tcs = decision["tool_calls"]

        # 关键：先把"申请单"写进历史（assistant + tool_calls）
        history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for tc in tcs
                ],
            }
        )

        # 并行调用：遍历数组，逐个执行、逐个回传（每个都有自己的 tool_call_id）
        for tc in tcs:
            result = run_tool(tc["name"], tc["arguments"])
            steps.append({"turn": turn, "tool": tc["name"], "arguments": tc["arguments"], "result": result})
            history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # 轮数用尽仍未收敛 → 明确报错，绝不能假装成功
    return ChatResponse(
        type="answer",
        content=f"达到最大轮数 {max_turns}，模型仍未给出最终回答（已停止，避免死循环）。",
        turns=max_turns,
        steps=steps,
    )


# ===========================================================================
# 6. 路由
# ===========================================================================
@app.get("/tools", summary="查看工具声明")
def list_tools() -> Dict[str, Any]:
    """模型能看到的工具清单。写工具描述时，把自己当成在给新同事交代任务。"""
    return {"count": len(TOOLS), "tools": TOOLS}


@app.post("/chat", response_model=ChatResponse, summary="对话（支持工具调用）")
def chat(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump(exclude_none=True) for m in req.messages]

    if req.auto_execute:
        return agent_loop(history, req.max_turns)

    # 无状态模式：只跑一轮，把决定返回给客户端，由客户端执行工具
    decision = mock_model(history)
    if decision["type"] == "answer":
        return ChatResponse(type="answer", content=decision["content"], turns=1, steps=[])

    return ChatResponse(
        type="tool_calls",
        tool_calls=[ToolCallOut(**tc) for tc in decision["tool_calls"]],
        turns=1,
        steps=[],
    )


# ===========================================================================
# 7. 自检（明天的题就从这里出，先自己想想）
# ===========================================================================
# Q1  auto_execute=true 和 false，分别对应"服务端托管"和"客户端执行"两种产品形态。
#     真实场景里，为什么大多数厂商选 false（只下发 tool_calls）？
# Q2  并行调用时，两条工具结果用同一个 tool_call_id 回传会怎样？
# Q3  max_turns 用尽后，服务端返回的是什么？为什么不能返回一个"看起来很正常的答案"？
# Q4  mock_model 里 "济南" 是默认值。真实项目里应该怎么处理"用户没说城市"？
# Q5  /tools 返回的 description，和模型的行为之间是什么关系？举一个写坏了的例子。
