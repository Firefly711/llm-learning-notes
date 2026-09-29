  # -*- coding: utf-8 -*-
"""
W1 Day2 · Python asyncio 异步基础
=================================
为什么 AI 应用工程师必须会这个：
  1) 调 LLM API 是"网络 IO"——等待模型回答常常要好几秒，同步代码会卡死整个服务；
  2) 真实项目里要同时发多个请求（多路并发），异步是唯一的高效姿势；
  3) FastAPI、OpenAI 客户端、LangChain 全是异步的，看不懂 async 就没法干活。

今天任务：跑实验（看时间对比）→ 看懂三件事 → 回答自检题。
运行：python w1d2_asyncio_demo.py
"""
import asyncio
import time

# ===========================================================================
# 一、模拟一次 LLM API 调用：真实场景里这里是"发请求给模型，等它回答"
#    await asyncio.sleep(2) 就相当于"网络等待 2 秒"
# ===========================================================================

async def mock_llm_call(name: str, delay: float) -> str:
    print(f"  [开始] {name}：请求发出，等待 {delay} 秒…")
    await asyncio.sleep(delay)          # 关键：await = 让出 CPU，去干别的
    print(f"  [完成] {name}：收到回复 ✅")
    return f"{name} 的结果"


# ===========================================================================
# 二、同步版：一个等完再发下一个（最笨的写法，但新手都会这么想）
# ===========================================================================

async def run_sync():
    print("--- 同步版：三个请求挨个等 ---")
    t0 = time.time()
    r1 = await mock_llm_call("请求A", 2)
    r2 = await mock_llm_call("请求B", 2)
    r3 = await mock_llm_call("请求C", 2)
    print(f"--- 同步版总耗时：{time.time() - t0:.1f} 秒（明显 ≈ 2×3）\n")
    return [r1, r2, r3]


# ===========================================================================
# 三、异步版：三个请求同时发出，一起等（asyncio.gather = 并发调度）
# ===========================================================================

async def run_concurrent():
    print("--- 异步版：三个请求同时发出 ---")
    t0 = time.time()
    results = await asyncio.gather(
        mock_llm_call("请求A", 1),
        mock_llm_call("请求B", 2),
        mock_llm_call("请求C", 3),
    )
    print(f"--- 异步版总耗时：{time.time() - t0:.1f} 秒（明显 ≈ 2，而不是 6）\n")
    return results


# ===========================================================================
# 四、超时保护：真实项目必须给每次调用加超时，防止模型卡死烧钱
# ===========================================================================

async def run_timeout():
    print("--- 超时保护：这个请求太慢，2 秒还没回来就放弃 ---")
    try:
        result = await asyncio.wait_for(mock_llm_call("慢请求", 5), timeout=2)
        print(f"  结果：{result}")
    except asyncio.TimeoutError:
        print("  ⚠ 请求超时，已放弃（真实项目这里会做重试或降级）\n")


# ===========================================================================
# 入口：asyncio.run() 就是"启动事件循环"——异步程序唯一的入口
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W1 Day2 · asyncio 异步基础（模拟 LLM API 调用）")
    print("=" * 60)
    asyncio.run(run_sync())
    asyncio.run(run_concurrent())
    asyncio.run(run_timeout())

    print("=" * 60)
    print("自检（回答这 3 题，W1 Day2 算过）")
    print("=" * 60)
    print("Q1: 为什么 LLM API 调用要用异步，而不是同步？")
    print("Q2: 在普通函数（def）里能直接写 await 吗？会报什么错？为什么？")
    print("Q3: asyncio.gather(a(), b()) 和 先 await a() 再 await b() 有什么区别？")
