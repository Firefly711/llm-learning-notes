# -*- coding: utf-8 -*-
"""
W1 Day1 · Python 类型提示（Type Hints）
======================================
为什么 AI 应用工程师必须会这个：
  1) FastAPI 靠类型注解自动生成请求校验和 API 文档（你后面会用到）；
  2) LLM 返回的数据是"不稳定的 JSON"，用类型提示 + Pydantic 才能安全接住；
  3) 大模型应用的代码库通常很绕（工具调用、Agent 状态流转），类型提示是防走神最好的护栏。

今天的任务：看输出 → 改代码 → 回答自检题。
运行：python w1d1_type_hints_demo.py
"""
from typing import Callable, Dict, List, Literal, Optional, TypedDict, Union

# ===========================================================================
# 一、基础语法：给变量、函数参数、返回值"贴标签"
# ===========================================================================

def add(a: int, b: int) -> int:
    """参数 a、b 是 int，返回值是 int。写错类型 IDE 立刻划线，但运行时不一定报错。"""
    return a + b


def greet(name: str, greeting: str = "你好") -> str:
    """默认值 + 类型注解可以共存。"""
    return f"{greeting}，{name}！"


# ===========================================================================
# 二、复合类型：一个变量可能有好几种形态
# ===========================================================================

# Optional[int] == int 或 None（例如"没查到的结果"）
def find_user_age(uid: str) -> Optional[int]:
    if uid == "u001":
        return 21
    return None  # 查不到就返回 None，调用方必须处理这种情况


# Union[int, str] == int 或 str（例如 LLM 可能返回数字也可能返回文字）
def normalize_answer(raw: Union[int, str]) -> str:
    return str(raw)


# Literal：只允许这几个值（例如限定 LLM 输出的固定枚举）
def set_agent_mode(mode: Literal["auto", "manual"]) -> str:
    return f"Agent 模式已切换为 {mode}"


# List / Dict：容器里的元素类型
def count_tokens_by_role(messages: List[Dict[str, str]]) -> Dict[str, int]:
    """输入：如 [{'role': 'user', 'content': 'hi'}]；输出：各 role 的条数。"""
    result: Dict[str, int] = {}
    for m in messages:
        role = m["role"]
        result[role] = result.get(role, 0) + 1
    return result


# ===========================================================================
# 三、AI 场景专用：TypedDict（给字典定结构）—— LLM 返回 JSON 时最好用
# ===========================================================================

class LLMUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class LLMResponse(TypedDict):
    content: str
    usage: LLMUsage


def parse_llm_response(data: LLMResponse) -> str:
    """TypedDict 让"字典有哪些字段"一目了然，写错键 IDE 直接提示。"""
    return f"回复内容：{data['content']}（共消耗 {data['usage']['prompt_tokens']} + {data['usage']['completion_tokens']} tokens）"


# ===========================================================================
# 四、函数作为参数：Callable —— Agent 里把工具函数传给调度器的标准姿势
# ===========================================================================

def execute_with_retry(worker: Callable[[str], str], payload: str, retries: int = 2) -> str:
    """Callable[[str], str] 表示：这个参数是个函数，输入 str、输出 str。"""
    for attempt in range(retries + 1):
        try:
            return worker(payload)
        except Exception:
            if attempt == retries:
                raise
    return ""


def fake_llm_call(prompt: str) -> str:
    return f"[模型回复] {prompt}"


# ===========================================================================
# 运行区：看这些输出，对照上面的注释理解每个类型
# ===========================================================================
def summarize_messages(messages: List[Dict[str, str]]) -> str:
    """统计对话里各角色的条数，返回一句总结。"""
    stats = count_tokens_by_role(messages)
    parts = [f"{role} {n} 条" for role, n in stats.items()]
    return "共" + str(len(messages)) + "条消息（" + "，".join(parts) + "）"


if __name__ == "__main__":
    print("1)", add(2, 3))
    print("2)", greet("小明"))
    print("3) find_user_age('u001') =", find_user_age("u001"), "| 查不到时 =", find_user_age("x"))
    print("4)", normalize_answer(42), "|", normalize_answer("四十"))
    print("5)", set_agent_mode("auto"))
    print("6)", count_tokens_by_role([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": "hello"}]))
    print("7)", parse_llm_response({"content": "今天天气不错",
                                     "usage": {"prompt_tokens": 120, "completion_tokens": 30}}))
    print("8)", execute_with_retry(fake_llm_call, "请介绍你自己"))
    print("9)", summarize_messages([{"role": "user", "content": "你好"},
                                    {"role": "assistant", "content": "你好！"},
                                    {"role": "user", "content": "继续"}]))
    print("\n" + "=" * 56)
    print("自检（回答这 3 题，W1 Day1 算过）")
    print("=" * 56)
    print("Q1: Optional[int] 和 int 有什么区别？为什么 LLM 应用里 Optional 很常见？")
    print("Q2: 把 set_agent_mode('auto') 改成 set_agent_mode(123)，程序能运行成功吗？")
    print("    （提示：试试，再想想为什么——类型提示只是'提示'，不是运行时强制）")
    print("Q3: 上面的 LLMResponse 如果少了 usage 字段，运行 parse_llm_response 会怎样？")
