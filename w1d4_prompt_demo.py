# -*- coding: utf-8 -*-
"""
W1 Day4 · Prompt Engineering 入门
=================================
核心认知：提示词（Prompt）工程 = 学会"组装 messages 数组"。

今天学三招（面试高频）：
  1) system prompt —— 给模型设角色、定规则
  2) few-shot —— 给几个示例，模型照葫芦画瓢（学格式/风格）
  3) CoT（思维链）—— 让模型"一步步想"，提升推理准确率

运行：python w1d4_prompt_demo.py
"""
import asyncio
import json
import os

# ===========================================================================
# 一、通用调用函数：不管 mock 还是真实，都是"组装 messages → 发请求"
#    这就是昨天学的 payload 结构，今天换不同 messages 反复用
# ===========================================================================

def build_payload(messages: list, stream: bool = False) -> dict:
    """把 messages 组装成 API 请求体（昨天学的固定格式）。"""
    return {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": stream,
    }


# ===========================================================================
# 二、演示 1：system prompt 定角色 —— 同一个问题，不同人设不同回答
# ===========================================================================

async def p1_system_prompt():
    print("─" * 60)
    print("演示1：system prompt 定角色")
    print("─" * 60)
    question = "我该不该用 Python 写这个自动化脚本？"

    # 人设 A：严谨工程师
    messages_a = [
        {"role": "system", "content": "你是一位严谨的软件工程师。回答时必须先分析利弊，再给结论。"},
        {"role": "user", "content": question},
    ]
    # 人设 B：毒舌段子手
    messages_b = [
        {"role": "system", "content": "你是一位毒舌的程序员段子手，回答要幽默、犀利、简短。"},
        {"role": "user", "content": question},
    ]

    print("\n📦 给模型 A 的 messages（注意 system 那行）：")
    print(json.dumps(messages_a, ensure_ascii=False, indent=2))
    reply_a = await mock_reply("A严谨工程师", question)
    print(f"\n💬 模型 A 回答：{reply_a}\n")

    print("📦 给模型 B 的 messages：")
    print(json.dumps(messages_b, ensure_ascii=False, indent=2))
    reply_b = await mock_reply("B毒舌段子手", question)
    print(f"\n💬 模型 B 回答：{reply_b}\n")

    print("✅ 结论：同一个问题，只改 system 一行，输出风格完全不同——")
    print("   system prompt = 给模型“装脑子”，定角色、定规则、定格式。\n")


# ===========================================================================
# 三、演示 2：few-shot —— 给示例，模型学会"照格式输出"
#    任务：把用户评价分类为 正面/负面，并输出 JSON
# ===========================================================================

async def p2_few_shot():
    print("─" * 60)
    print("演示2：few-shot 用示例教格式")
    print("─" * 60)

    # 零示例（zero-shot）：直接问
    zero_shot = [
        {"role": "system", "content": "把用户评价分类为正面或负面，输出 JSON。"},
        {"role": "user", "content": "物流太慢了，等了一周才到"},
    ]

    # 两个示例（few-shot）：先给模型"标准答案长什么样"
    few_shot = [
        {"role": "system", "content": "把用户评价分类为正面或负面，输出 JSON，格式和示例一致。"},
        {"role": "user", "content": "这个手机拍照太清晰了，爱不释手"},
        {"role": "assistant", "content": '{"sentiment": "正面", "keywords": ["拍照清晰"]}'},
        {"role": "user", "content": "屏幕用了三天就出现坏点"},
        {"role": "assistant", "content": '{"sentiment": "负面", "keywords": ["屏幕坏点"]}'},
        {"role": "user", "content": "物流太慢了，等了一周才到"},
    ]

    print("\n📦 zero-shot 只给了 3 条 messages：")
    print(json.dumps(zero_shot, ensure_ascii=False))
    print(f"💬 模型输出（容易格式跑偏）：{await mock_reply('zero-shot', '分类')}\n")

    print("📦 few-shot 给了 5 条 messages（注意中间多出的 2 个示例对）：")
    print(json.dumps(few_shot, ensure_ascii=False))
    print(f"💬 模型输出（看了示例，跟着输出 JSON）：{await mock_reply('few-shot', '分类')}\n")

    print("✅ 结论：示例对 = user(问题)+assistant(标准答案)。模型会模仿示例的格式。")
    print("   想让模型稳定输出 JSON/表格/指定风格 → 给 few-shot 示例是最有效的办法。\n")


# ===========================================================================
# 四、演示 3：CoT（思维链）—— 让模型"先想再答"，提升正确率
# ===========================================================================

async def p3_chain_of_thought():
    print("─" * 60)
    print("演示3：CoT 思维链")
    print("─" * 60)
    question = "一个农夫有 12 只鸡，卖掉 3 只，又买了 5 只，现在有几只鸡？"

    direct = [
        {"role": "user", "content": question},
    ]
    cot = [
        {"role": "user", "content": f"{question}\n请一步一步思考，最后再给出答案。"},
    ]

    print("\n📦 直接问（不带思考要求）：")
    print(f"💬 模型回答：{await mock_reply('直接', question)}\n")

    print("📦 CoT 问法（加了一句'请一步一步思考'）：")
    print(f"💬 模型回答：{await mock_reply('CoT', question)}\n")

    print("✅ 结论：复杂推理/数学/逻辑题，让模型把推理过程写出来，准确率显著提升。")
    print("   底层原因：模型是“一步步生成”的，写下中间步骤相当于给它“草稿纸”。\n")


# ===========================================================================
# MOCK 回答（无 key 时用）：模拟三种 prompt 的输出效果
# ===========================================================================

MOCK_ANSWERS = {
    "A严谨工程师": "建议写。理由：1) 任务重复且规则明确，适合自动化；2) 脚本一次投入、长期复用。风险：需预留异常处理和日志，避免跑挂了无人知。结论：值得做，但先写清需求边界。",
    "B毒舌段子手": "写啊！你手敲一小时，脚本跑一辈子。不写？那你就是全公司最贵的 Ctrl+C/V 机器。",
    "zero-shot": '{"分类": "负面"}',
    "few-shot": '{"sentiment": "负面", "keywords": ["物流慢"]}',
    "直接": "9只。",
    "CoT": "最初有12只 → 卖掉3只：12-3=9 → 又买5只：9+5=14 → 所以现在有14只鸡。",
}


async def mock_reply(kind: str, question: str) -> str:
    await asyncio.sleep(0.6)  # 模拟一点延迟
    return MOCK_ANSWERS[kind]


# ===========================================================================
# 真实调用（有 key 才跑）：用同一个 payload，看看真模型对三种 prompt 的反应
# ===========================================================================

async def real_quick(messages: list, label: str):
    import httpx
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=build_payload(messages),
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"  [{label}] {content}\n")
        else:
            print(f"  [{label}] HTTP {resp.status_code}（余额不足？请充值后重试）\n")


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W1 Day4 · Prompt Engineering 入门")
    print("=" * 60)
    asyncio.run(p1_system_prompt())
    asyncio.run(p2_few_shot())
    asyncio.run(p3_chain_of_thought())

    if os.environ.get("DEEPSEEK_API_KEY"):
        print("─" * 60)
        print("真实调用：同一个问题，三种提示词，真模型的表现")
        print("─" * 60)
        asyncio.run(real_quick([
            {"role": "system", "content": "用一句话回答，不超过20字。"},
            {"role": "user", "content": "什么是 RAG？"}], "system版"))
        asyncio.run(real_quick([
            {"role": "system", "content": "回答用 JSON 格式。"},
            {"role": "user", "content": "苹果、香蕉、橙子分别是什么颜色？"},
            {"role": "assistant", "content": '{"苹果": "红色", "香蕉": "黄色"}'},
            {"role": "user", "content": "葡萄是什么颜色？"}], "few-shot版"))

    print("=" * 60)
    print("自检（回答这 3 题，W1 Day4 算过）")
    print("=" * 60)
    print("Q1: system prompt 和 user 消息有什么区别？各举一个用途。")
    print("Q2: 模型输出的 JSON 格式总是不稳定，你会怎么解决？（至少说一种）")
    print("Q3: CoT 为什么能提升推理正确率？")
