# -*- coding: utf-8 -*-
"""
W2-D4 · 把 mock 换成真实模型（DeepSeek）
========================================
昨天和前天，tool_calls 是我们自己造的。今天换成真实模型返回的。

真实模型和 mock 的差别只有两件事：
  1. 谁生成 tool_calls —— 从"关键词匹配"变成"模型语义判断"
  2. arguments 变成 JSON 字符串 —— 必须 json.loads，不能直接当 dict 用

其余环节（执行、回传、循环、轮数上限）一行都不用改。
这正好验证一件事：协议不依赖具体模型。

用法
----
1) 先设置 key（在 PowerShell 里，当前窗口有效）：
     $env:DEEPSEEK_API_KEY = "你的key"
2) 跑：
     python "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat/w2d4_real_llm_tools_demo.py"
3) 只想看请求体、不发请求：
     python ".../w2d4_real_llm_tools_demo.py" --check

可选环境变量：
  DEEPSEEK_BASE_URL  默认 https://api.deepseek.com/v1
  DEEPSEEK_MODEL     默认 deepseek-chat
"""
import json
import math
import os
import sys
from datetime import datetime

import httpx

BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
KEY = os.getenv("DEEPSEEK_API_KEY")
MAX_TURNS = 5
TIMEOUT = 30.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的当前天气。用户问天气时调用。",
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
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    if name == "get_weather":
        fake = {"北京": "晴 22°C", "济南": "多云 21°C", "上海": "小雨 24°C", "成都": "阴 19°C"}
        city = arguments.get("city") or "未知"
        return json.dumps({"city": city, "weather": fake.get(city, "暂无数据")}, ensure_ascii=False)
    if name == "get_current_time":
        return json.dumps({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)
    if name == "calculator":
        expr = str(arguments.get("expression", "")).replace(" ", "")
        if not expr or not all(c in "0123456789+-*/()." for c in expr):
            return json.dumps({"error": "表达式含非法字符"}, ensure_ascii=False)
        try:
            value = eval(expr, {"__builtins__": {}}, {"math": math})
            return json.dumps({"expression": expr, "result": round(float(value), 4)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"计算失败：{e}"}, ensure_ascii=False)
    return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)


def build_payload(messages: list) -> dict:
    return {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}


def explain_status(code: int, body_text: str) -> str:
    """把上游 HTTP 错误翻译成人能看懂的话。生产项目里这段应该进日志和监控。"""
    if code == 401:
        return "401 认证失败：key 无效、过期或没带上。检查 $env:DEEPSEEK_API_KEY 是否设置正确。"
    if code == 402:
        return "402 余额不足：账户没钱了。到 DeepSeek 控制台充值（几元就够跑完整个 W2），再重试。"
    if code == 429:
        return "429 触发限流：请求太频繁。等几秒再试，生产上要做指数退避重试。"
    if code == 400:
        return f"400 请求非法：多半是 messages 结构不对或工具 schema 不被接受。上游返回：{body_text[:200]}"
    if 500 <= code < 600:
        return f"{code} 上游服务异常：不是你的问题，属于供应商故障，生产上要重试 + 降级。"
    return f"HTTP {code}：{body_text[:200]}"


def chat_once(client: httpx.Client, messages: list) -> dict:
    payload = build_payload(messages)
    resp = client.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(explain_status(resp.status_code, resp.text))
    return resp.json()


def run_conversation(client: httpx.Client, question: str) -> None:
    messages = [
        {"role": "system", "content": "你是乐于助人的助手。需要实时信息时使用工具，不要编造。"},
        {"role": "user", "content": question},
    ]
    print(f"\n用户: {question}")

    total_tokens = 0
    for turn in range(MAX_TURNS):
        try:
            data = chat_once(client, messages)
        except RuntimeError as e:
            print(f"  ✗ 调用失败：{e}")
            return
        except httpx.TimeoutException:
            print(f"  ✗ 请求超时（{TIMEOUT}s）：网络慢或上游拥堵，生产上要重试或降级。")
            return
        except httpx.ConnectError as e:
            print(f"  ✗ 连不上 {BASE_URL}：检查网络 / 代理。{e}")
            return

        usage = data.get("usage") or {}
        total_tokens += usage.get("total_tokens", 0)
        choice = data["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason")

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            print(f"助手: {msg.get('content')}")
            print(f"  [本轮 finish_reason={finish}, 累计 tokens={total_tokens}]")
            return

        print(f"  → 模型申请调用 {len(tool_calls)} 个工具（finish_reason={finish}）")
        # 关键：原样把 assistant 消息写回历史（含 tool_calls），不要自己重建
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            # 真实模型的 arguments 是 JSON 字符串，必须解析，且要防它给出非法 JSON
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                print(f"  ⚠ 模型给了非法 JSON 参数：{raw_args!r}，回传错误让它纠正")
                args = None

            if args is None:
                result = json.dumps({"error": "arguments 不是合法 JSON"}, ensure_ascii=False)
            else:
                print(f"     参数: {json.dumps(args, ensure_ascii=False)}")
                result = run_tool(name, args)

            print(f"     工具 {name} 返回: {result}")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        print("  ↻ 已回传工具结果，再次请求模型…")

    print(f"  ⚠ 达到最大轮数 {MAX_TURNS}，已停止（防止死循环烧钱），累计 tokens={total_tokens}")


def main() -> None:
    if "--check" in sys.argv:
        demo_messages = [
            {"role": "system", "content": "你是乐于助人的助手。"},
            {"role": "user", "content": "济南天气怎么样？现在几点了？"},
        ]
        print("将要发送的请求体（不发请求，key 不打印）：")
        print(json.dumps(build_payload(demo_messages), ensure_ascii=False, indent=2))
        print(f"\n目标地址: {BASE_URL}/chat/completions  模型: {MODEL}")
        return

    if not KEY:
        print("✗ 没有找到 DEEPSEEK_API_KEY。")
        print("  在当前 PowerShell 窗口执行：")
        print('     $env:DEEPSEEK_API_KEY = "你的key"')
        print("  然后重新运行本脚本。")
        print("  只想看请求体结构：加 --check 参数。")
        return

    print("=" * 64)
    print(f"真实模型测试 | {MODEL} @ {BASE_URL}")
    print("=" * 64)

    with httpx.Client(headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}) as client:
        # 四组问题，覆盖：单工具 / 并行工具 / 纯计算 / 不该调工具
        run_conversation(client, "济南天气怎么样？现在几点了？")
        run_conversation(client, "帮我算一下 (12+8)*3/5")
        run_conversation(client, "你好，用一句话介绍你自己")

    print("\n提示：如果看到 402，先去充值；余额正常但模型没调工具，说明工具 description 写得不够清楚。")


if __name__ == "__main__":
    main()
