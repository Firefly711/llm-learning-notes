# -*- coding: utf-8 -*-
"""
Day 1 · Function Calling 深度实践
=================================
今日主线B任务。重点不是"会用 SDK"，而是讲清"一轮工具调用背后发生了什么"。

本文件默认用 MOCK 模式运行（不需要 API key），把整条链路真实跑给你看：
    用户消息 → 模型决定调用工具(tool_call) → 程序执行工具 → 工具结果回传
    → 模型基于结果生成最终回答

跑法：  python day1_function_calling_demo.py

想切真实模型：设环境变量 OPENAI_API_KEY（DeepSeek/智谱等 OpenAI 兼容 API 也行）
并改 MODEL / BASE_URL，见文末 REAL_API 说明。
"""
import json
import math
import os
from datetime import datetime

# ===========================================================================
# 第 1 步：把"工具"定义成 JSON Schema —— 模型看到的是这份声明，不是函数本身
# ===========================================================================
TOOLS = [
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
            "description": "获取当前时间。用户问时间/几点时调用。",
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
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        },
    },
]

# ===========================================================================
# 第 2 步：真实执行工具的函数（这是"我们自己"的代码，模型碰不到）
# ===========================================================================
def run_tool(name: str, arguments: dict) -> str:
    """根据模型给出的工具名和参数，实际执行并返回字符串结果。"""
    if name == "get_weather":
        # 教学示例用固定数据；真实项目这里换成天气 API 调用
        fake_weather = {"北京": "晴 22°C", "济南": "多云 21°C", "上海": "小雨 24°C"}
        city = arguments.get("city", "济南")
        return json.dumps({"city": city, "weather": fake_weather.get(city, "未知 20°C")}, ensure_ascii=False)

    if name == "get_current_time":
        return json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)

    if name == "calculator":
        expr = arguments.get("expression", "").replace(" ", "")
        # 只允许白名单字符，防注入（生产项目用表达式解析库，绝不 eval 用户输入）
        if not expr or not all(c in "0123456789+-*/()." for c in expr):
            return json.dumps({"error": "表达式含非法字符"}, ensure_ascii=False)
        try:
            value = eval(expr, {"__builtins__": {}}, {"math": math})  # 仅教学演示，生产勿用 eval
            return json.dumps({"expression": expr, "result": round(float(value), 4)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)

# ===========================================================================
# 第 3 步：MOCK"模型"—— 模拟真实 LLM 的两个行为：
#   ① 决定要不要调工具、调哪个、传什么参数（返回 tool_calls）
#   ② 拿到工具结果后，生成最终回答（返回 assistant 消息）
# 真实接入时，这两步分别是一次 API 调用，见文末。
# ===========================================================================
def mock_model_step(messages):
    """启发式模拟：已有工具结果→生成最终回答；否则按关键词返回 tool_call。"""
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break

    # ① 已有 role="tool" 的结果 → 模拟模型"看到结果后总结"（真实模型也是这一轮出最终回答）
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    if tool_msgs:
        last_tool = json.loads(tool_msgs[-1]["content"])
        if "result" in last_tool:
            return f"计算结果：{last_tool['expression']} = {last_tool['result']}。"
        if "weather" in last_tool:
            return f"{last_tool['city']}今天的天气是：{last_tool['weather']}。"
        if "time" in last_tool:
            return f"现在是 {last_tool['time']}。"
        return f"根据查询结果：{json.dumps(last_tool, ensure_ascii=False)}。"

    # ② 没有工具结果 → 判断这轮要不要调工具
    if "天气" in last_user:
        return [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "济南"}}}]
    if "时间" in last_user or "几点" in last_user:
        return [{"type": "function", "function": {"name": "get_current_time", "arguments": {}}}]
    if any(c in last_user for c in "+-*/"):
        expr = last_user
        for prefix in ("帮我算一下", "帮我算", "计算一下", "计算", "等于多少", "是多少"):
            expr = expr.replace(prefix, "").strip()
        return [{"type": "function", "function": {"name": "calculator", "arguments": {"expression": expr}}}]

    # ③ 不需要工具 → 直接回答
    return f"（模型直接回答）你说了：{last_user}。我没用到工具。"

# ===========================================================================
# 第 4 步：完整的 Function Calling 循环（真实项目中唯一不变的主干逻辑）
# ===========================================================================
def run_conversation(user_text: str) -> None:
    messages = [{"role": "system", "content": "你是乐于助人的助手，必要时使用工具获取实时信息。"},
                {"role": "user", "content": user_text}]

    print(f"\n用户: {user_text}")
    # 最多循环 5 轮，防止模型反复调用工具形成死循环（这是 Agent 工程的常见坑）
    for turn in range(5):
        step = mock_model_step(messages)  # ← 真实项目：requests 调 /v1/chat/completions
        if isinstance(step, str):
            # 模型已拿到工具结果，生成最终回答
            print(f"助手: {step}\n")
            return
        tool_calls = step

        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"  → 模型请求调用工具: {name}({json.dumps(args, ensure_ascii=False)})")

            result = run_tool(name, args)        # ← 我们执行，模型只有"声明"，没有执行权
            print(f"  ← 工具返回: {result}")

            # 关键一步：把工具结果以 role="tool" 回传，并带上 tool_call_id 关联
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})
            messages.append({"role": "tool", "tool_call_id": f"call_{turn}", "content": result})

    print("  ⚠ 达到最大循环次数，强制结束（防止死循环）\n")

# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Function Calling 深度实践（MOCK 模式，无需 API key）")
    print("=" * 60)
    for text in ["今天济南天气怎么样？", "现在几点了？", "帮我算一下 (2+3)*4-1", "你好，简单介绍一下你自己"]:
        run_conversation(text)

    print("-" * 60)
    print("""
【换真实模型只需 3 步】本文件的教学核心 = messages 的拼接与循环，与 SDK 无关：
  1) 设环境变量: OPENAI_API_KEY
  2) 把 mock_model_step 替换为一次 HTTP 调用（OpenAI 兼容格式）:
       POST {BASE_URL}/chat/completions
       Authorization: Bearer {key}
       body: {"model": MODEL, "messages": messages, "tools": TOOLS}
  3) 响应里 choices[0].message 若带 tool_calls → 执行工具回传后再次调用；
     不带 → content 就是最终回答。
  常见 BASE_URL: https://api.openai.com/v1 (OpenAI)
                  https://api.deepseek.com/v1 (DeepSeek, 国内可直连)
【面试加分点】能讲清这三件事：
  ① 模型不"执行"工具，只"声明"要调用的工具与参数（tool_calls）；
  ② 工具结果必须带 role="tool" + tool_call_id 回传，模型才能"看到"结果；
  ③ 必须有最大轮数保护，否则 agent 会死循环烧钱。
""")
