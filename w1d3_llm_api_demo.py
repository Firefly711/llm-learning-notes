# -*- coding: utf-8 -*-
"""
W1 Day3 · LLM API 首次调用 + 流式输出（streaming）
==================================================
今天核心就一件事：看清"调用大模型"到底发生了什么，以及 streaming 为什么是标配。

双模式：
  1) MOCK 模式（默认，无需 key）：模拟一个"LLM 服务端"，完整演示 非流式 vs 流式 的差别；
  2) 真实模式（有 key 才启用）：设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，
     用 DeepSeek（OpenAI 兼容接口）真实调一次。

运行：python w1d3_llm_api_demo.py
"""
import asyncio
import os
import time

# ===========================================================================
# 一、先看"API 调用"的本质
#   调用 LLM = 向服务器发一个 HTTP POST 请求：
#     POST {base_url}/chat/completions
#     body: { "model": "...", "messages": [...], "stream": true/false }
#   服务器返回的 messages 结构（OpenAI 兼容，各家通用）：
#     [{"role":"system","content":"你是助手"},   ← 设定人设/规则，只出现一次
#      {"role":"user","content":"你好"},          ← 用户说的话
#      {"role":"assistant","content":"你好！"}]   ← 模型的回复
# ===========================================================================

# 模拟回复内容（一段话，方便看流式效果）
MOCK_REPLY = "你好！我是模拟的大模型。今天我们来学习如何调用 LLM API 和流式输出。流式输出的好处是：第一个字很快就能出现，不用等整段生成完。这对用户体验非常重要！"


# ===========================================================================
# 二、MOCK 服务端：假装我们是一台真实的模型服务器
# ===========================================================================

class MockServer:
    """模拟一个 LLM 服务端：处理需要 1.5 秒，之后一次性返回或逐字吐出。"""

    @staticmethod
    async def complete_once(prompt: str) -> str:
        """非流式：攒完整段，一次性返回。"""
        await asyncio.sleep(1.5)  # 模拟服务器计算时间
        return MOCK_REPLY

    @staticmethod
    async def complete_stream(prompt: str):
        """流式：把整段拆成"字"，每隔 0.05 秒吐一个字（模拟网络分块传输）。"""
        await asyncio.sleep(1.5)
        for ch in MOCK_REPLY:
            await asyncio.sleep(0.05)
            yield ch


# ===========================================================================
# 三、非流式调用 vs 流式调用：用时间戳看清差别
# ===========================================================================

async def demo_non_stream():
    print("── 非流式调用（stream=False）────────────────────────")
    t0 = time.time()
    reply = await MockServer.complete_once("你好")
    t_full = time.time() - t0
    print(f"  等待 {t_full:.1f} 秒后，一次性打印全文：")
    print(f"  {reply}\n")
    return t_full


async def demo_stream():
    print("── 流式调用（stream=True）──────────────────────────")
    t0 = time.time()
    first_shown = False
    pieces = []
    async for chunk in MockServer.complete_stream("你好"):
        if not first_shown:
            # 关键指标：首字延迟 = 用户等第一个字的时间
            print(f"  首字出现时间：{time.time() - t0:.2f} 秒")
            print(f"  输出过程：", end="", flush=True)
            first_shown = True
        print(chunk, end="", flush=True)  # flush=True 强制立即打印（流式的灵魂）
        pieces.append(chunk)
    t_done = time.time() - t0
    print(f"\n  全部输出完成：{t_done:.2f} 秒（但用户 1.6 秒就开始看到内容了）\n")
    return t_done


# ===========================================================================
# 四、真实 API 调用（有 key 才跑；没有 key 就跳过，不影响今天的学习）
#   DeepSeek 兼容 OpenAI 格式：POST https://api.deepseek.com/v1/chat/completions
# ===========================================================================

async def real_call():
    import httpx
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = "https://api.deepseek.com/v1" if os.environ.get("DEEPSEEK_API_KEY") else "https://api.openai.com/v1"
    model = "deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY") else "gpt-4o-mini"

    if not key:
        print("  跳过真实调用：未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY 环境变量")
        print("  （有 key 后：在 PowerShell 执行  $env:DEEPSEEK_API_KEY='sk-...'  再运行本文件即可）\n")
        return

    print(f"── 真实调用 {model}（stream=True）────────────────")
    async with httpx.AsyncClient(timeout=60) as client:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个简洁的助手。"},
                {"role": "user", "content": "用一句话介绍你自己"},
            ],
            "stream": True,
        }
        t0 = time.time()
        first = True
        async with client.stream("POST", f"{base}/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"},
                                 json=payload) as resp:
            if resp.status_code != 200:
                print(f"  ⚠ HTTP {resp.status_code}：{await resp.aread()}")
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                # 简化解析：只取 delta.content（教学版，生产用官方 SDK 或 json 解析）
                if '"content"' in data:
                    content = data.split('"content":"', 1)[1].split('"', 1)[0]
                    if first:
                        print(f"  真实首字延迟：{time.time() - t0:.2f} 秒")
                        first = False
                    print(content, end="", flush=True)
        print(f"\n  真实流式输出完成，总耗时 {time.time() - t0:.2f} 秒\n")


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W1 Day3 · LLM API 首次调用 + streaming")
    print("=" * 60)
    asyncio.run(demo_non_stream())
    asyncio.run(demo_stream())
    asyncio.run(real_call())

    print("=" * 60)
    print("自检（回答这 3 题，W1 Day3 算过）")
    print("=" * 60)
    print("Q1: 流式和一次性返回，用户感知上最大的区别是什么？（用首字延迟回答）")
    print("Q2: 为什么生产环境的 LLM 应用几乎都用流式？至少说两点理由。")
    print("Q3: messages 数组里的 role 有哪几种？system 消息是用来做什么的？")
