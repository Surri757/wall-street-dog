# -*- coding: utf-8 -*-
"""数据源子包共享的解析工具（零依赖叶子模块，pools/concepts 共同下沉点）。"""
from __future__ import annotations


def num(v) -> float:
    """宽松转 float；NaN/非法返回 NaN（东财池字段大量 'null'/缺位）。"""
    try:
        f = float(v)
        return f if f == f else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def to_int(v) -> int:
    f = num(v)
    return int(f) if f == f else 0


def to_secid(code: str) -> str:
    """6 位代码 → 东财 secid（沪 1. / 深 0. 前缀）。

    此前 kline/concepts/pools 共 5 处手抄同一表达式，改口径漏一处即静默漂移
    （拆包时就漏过 _num/_to_int 导入致解析必崩——tests/test_logic.py 有记录）。
    """
    return ("1." if code.startswith(("6", "9")) else "0.") + code


def to_tx_symbol(code: str) -> str:
    """6 位代码 → 腾讯/新浪日K symbol（sh/sz 前缀）。"""
    return ("sh" if code.startswith(("6", "9")) else "sz") + code
