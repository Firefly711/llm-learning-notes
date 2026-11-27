# -*- coding: utf-8 -*-
"""
W2-D5 实验 · 并发 vs 串行
==========================
目的：证明 async 路由的价值——"等待时不占着人"。

做法：
  1) 预热一次（把 TLS 握手、建连的开销排除在外，否则第一次会显得特别慢）
  2) 并发：asyncio.gather 同时发 N 个请求，看总耗时
  3) 串行：for 循环一个一个发，看总耗时

预期（每个请求 2 轮模型调用，单请求约 4~5 秒）：
  并发 3 个 ≈ 一个请求的时间（4~5s）
  串行 3 个 ≈ 三个请求相加（12~15s）

跑法（服务必须已经启动在 8020）：
  python w2d5_concurrency_experiment.py
  换端口/改请求数：
  python w2d5_concurrency_experiment.py --base http://127.0.0.1:8022 --n 3
"""
import argparse
import asyncio
import time

import httpx

PAYLOAD = {
    "messages": [{"role": "user", "content": "济南天气怎么样？现在几点了？"}],
    "auto_execute": True,
    "max_turns": 5,
}


async def one(client: httpx.AsyncClient, base: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{base}/chat", json=PAYLOAD, timeout=60)
        dt = time.perf_counter() - t0
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "code": r.status_code,
            "dt": dt,
            "tokens": body.get("tokens"),
            "detail": body.get("detail"),
        }
    except Exception as e:
        return {"code": 0, "dt": time.perf_counter() - t0, "tokens": None, "detail": repr(e)}


def show(tag: str, results: list) -> None:
    for i, r in enumerate(results, 1):
        if r["code"] == 200:
            print(f"    第 {i} 个: {r['dt']:.2f}s  tokens={r['tokens']}")
        else:
            print(f"    第 {i} 个: 失败  code={r['code']}  {str(r['detail'])[:100]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    async with httpx.AsyncClient() as client:
        print(f"目标: {args.base}   每个请求都是 2 轮模型调用\n")
        print("【预热】排除建连开销 …")
        warm = await one(client, args.base)
        print(f"    预热耗时: {warm['dt']:.2f}s  code={warm['code']}\n")

        print(f"【并发】{args.n} 个请求同时发出 …")
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(client, args.base) for _ in range(args.n)])
        total_c = time.perf_counter() - t0
        show("并发", results)
        print(f"    总耗时: {total_c:.2f}s\n")

        print(f"【串行】{args.n} 个请求排队执行 …")
        t0 = time.perf_counter()
        results2 = []
        for _ in range(args.n):
            results2.append(await one(client, args.base))
        total_s = time.perf_counter() - t0
        show("串行", results2)
        print(f"    总耗时: {total_s:.2f}s\n")

        print("-" * 56)
        if total_c > 0:
            print(f"结论: 并发 {total_c:.2f}s  vs  串行 {total_s:.2f}s  →  快 {total_s / total_c:.1f} 倍")
        ok = sum(1 for r in results if r["code"] == 200)
        print(f"成功 {ok}/{args.n}。如果看到 503，说明触发了限流——这本身也是错误映射生效的证据。")
        print("注意: 并发没有让模型变快，它只是让'等待'不再互相排队。")


if __name__ == "__main__":
    asyncio.run(main())
