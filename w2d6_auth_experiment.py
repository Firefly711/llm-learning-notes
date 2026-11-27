# -*- coding: utf-8 -*-
r"""
W2-D6 · 鉴权与限流实验：把 401 / 429 亲手打出来
=================================================
前提：8025 端口的服务已经在另一个窗口跑着
  python -m uvicorn w2d6_auth_ratelimit_demo:app --port 8025 --app-dir "C:\Users\李赟坤\Doubao\chats\2026-09-20\new-chat"

跑法（本窗口）：
  cd C:\Users\李赟坤\Doubao\chats\2026-09-20\new-chat
  python w2d6_auth_experiment.py

想省时间可加 --skip-wait，跳过最后约 60 秒的窗口恢复验证。
"""
import argparse
import time

import httpx

GOOD_KEY = "dev-key-001"
BAD_KEY = "sk-this-key-is-wrong"
BODY = {"messages": [{"role": "user", "content": "你好"}]}


def one_line(r):
    """把一次响应压成一行，便于对照"""
    try:
        j = r.json()
    except Exception:
        return f"{r.status_code}  {r.text[:60]}"
    if "detail" in j:
        return f"{r.status_code}  detail={j['detail']}"
    if j.get("type") == "answer":
        return f"{r.status_code}  type=answer  「{j.get('content', '')[:28]}」"
    return f"{r.status_code}  {str(j)[:60]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8025")
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()

    with httpx.Client(base_url=args.base, timeout=20) as c:
        health = c.get("/health")
        if health.status_code != 200:
            print(f"× 连不上 {args.base}/health（返回 {health.status_code}）")
            print("  先在另一个窗口启动 w2d6_auth_ratelimit_demo，端口 8025")
            return
        h = health.json()
        rl = h["rate_limit"]
        limit, window = rl["limit"], rl["window_seconds"]

        print("=" * 70)
        print("W2-D6 鉴权与限流实验")
        print(f"服务 {args.base}    模式 {h['llm_mode']}    限额 {limit} 次 / {window}s")
        print("=" * 70)

        # ---------- 0. 清场 + 拿第 1 次正常响应 ----------
        first = c.post("/chat", json=BODY, headers={"X-API-Key": GOOD_KEY})
        cool = 0
        while first.status_code == 429 and cool < 3:
            wait = int(first.headers.get("Retry-After", window))
            print(f"\n[0] 上一轮配额还没滑出窗口，等 {wait} 秒…")
            time.sleep(wait + 1)
            first = c.post("/chat", json=BODY, headers={"X-API-Key": GOOD_KEY})
            cool += 1
        if first.status_code == 429:
            print("× 始终被限流，请 Ctrl+C 重启服务后再跑")
            return
        print(f"\n[0] 正常响应（这一次已经吃掉 1 个配额）：{one_line(first)}")

        # ---------- 1. 不带 key ----------
        print("\n[1] 不带 key —— 三道门分别试")
        r = c.get("/health")
        print(f"  GET  /health    {one_line(r)}   ← 刻意不设防")
        r = c.get("/tools")
        print(f"  GET  /tools     {one_line(r)}")
        print(f"     响应头  WWW-Authenticate: {r.headers.get('www-authenticate')}")
        r = c.post("/chat", json=BODY)
        print(f"  POST /chat      {one_line(r)}")

        # ---------- 2. key 不对 ----------
        print("\n[2] 带了 key 但值不对")
        r = c.post("/chat", json=BODY, headers={"X-API-Key": BAD_KEY})
        print(f"  POST /chat      {one_line(r)}")
        print("     ↑ 和第 [1] 段的 detail 不同：一个是「没给」，一个是「给了但错」")

        # ---------- 3. 正确 key 连打，直到撞墙 ----------
        print(f"\n[3] 正确 key 连续请求，共 {limit + 1} 次（第 [0] 段算第 1 次）")
        last = first
        rows = [(1, first)]
        for i in range(2, limit + 2):
            t0 = time.perf_counter()
            last = c.post("/chat", json=BODY, headers={"X-API-Key": GOOD_KEY})
            ms = (time.perf_counter() - t0) * 1000
            rows.append((i, last, ms))
            time.sleep(0.05)

        for row in rows:
            i, r = row[0], row[1]
            ms = f"{row[2]:5.0f}ms" if len(row) > 2 else "      -"
            mark = "  ← 被限流" if r.status_code == 429 else ""
            print(f"  第 {i} 次  {one_line(r)}  {ms}{mark}")
        if last.status_code == 429:
            print(f"     响应头  Retry-After: {last.headers.get('retry-after')} 秒")

        # ---------- 4. 窗口自动恢复 ----------
        if args.skip_wait:
            print("\n[4] 已跳过等待（--skip-wait）")
            return
        retry = int(last.headers.get("Retry-After", window))
        print(f"\n[4] 等 {retry + 1} 秒，看窗口自己滑过去（这期间所有请求都会被拒）")
        time.sleep(retry + 1)
        r = c.post("/chat", json=BODY, headers={"X-API-Key": GOOD_KEY})
        print(f"  冷却结束后再发 1 次  {one_line(r)}")

        # ---------- 5. 结论 ----------
        print("\n" + "=" * 70)
        print("四条结论（面试可以直接说）")
        print("  1. 没 key 和 key 错都是 401，但 detail 可区分；这类错误不能重试，只能换凭据")
        print(f"  2. 前 {limit} 次正常，第 {limit + 1} 次 429 + Retry-After：配额用尽")
        print("  3. 429 不是故障，等窗口滑过去自动恢复，不需要改配置或重启")
        print("  4. 配额是按 key 记的，不是按 IP：换一个 key 就是一份新额度")
        print("=" * 70)


if __name__ == "__main__":
    main()
