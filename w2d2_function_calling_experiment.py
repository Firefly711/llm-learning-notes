# -*- coding: utf-8 -*-
"""
W2-D2 实验：观察 Function Calling 的三个真实行为
================================================
跑法：python "C:/Users/李赟坤/Doubao/chats/2026-09-20/new-chat/w2d2_function_calling_experiment.py"

不需要 API key，用的是 demo 文件里的 mock 模型。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day1_function_calling_demo import run_conversation

print("\n" + "=" * 60)
print("实验 1：一次问两件事（天气 + 时间）")
print("=" * 60)
run_conversation("济南今天天气怎么样？现在几点了？")

print("\n" + "=" * 60)
print("实验 2：没给城市（信息缺失）")
print("=" * 60)
run_conversation("今天天气怎么样？")

print("\n" + "=" * 60)
print("实验 3：复杂算式")
print("=" * 60)
run_conversation("帮我算一下 (12+8)*3/5")
