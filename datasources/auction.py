# -*- coding: utf-8 -*-
"""腾讯分时（近 RETENTION_DAYS 个交易日分钟线）：集合竞价点 / 当日总量 / 执行日路径。

一次请求返回最近 5 个交易日（新→旧）、完整交易日约 267 个点，每点为**累计口径**：
    HHMM 价格 累计成交量(手) 累计成交额(元)
  - 当日第一个点（0930）= 开盘集合竞价：价格=开盘价、量=竞价量、额=竞价额；
  - 当日最后一个点 = 当日总量 / 总额。
实测（2026-11-23/24，5 只 × 5 日）：首点价=日线开盘、末点价=日线收盘（逐位相等），
末点量/日线量 = 1.0000~1.0005。date 参数不改变返回范围（恒为最近 5 个交易日），
故历史竞价数据**不可回补**，只能前向采集（见 auction 子包）。

点数不足 COMPLETE_POINTS 的日子标记 complete=False（盘中/截断），调用方不得当完整
交易日落盘。
"""
from __future__ import annotations

from .http import EmptyDataError, TX_MINUTE_URL, _http_get_json, assert_public_url
from .util import to_tx_symbol

RETENTION_DAYS = 5   # 上游保留的交易日数（实测）
COMPLETE_POINTS = 240  # 完整交易日实测 267 点；不足视为盘中/截断


def parse_minute_days(node: dict) -> list[dict]:
    """纯函数：腾讯分时节点 → [{date: 'YYYYMMDD', complete: bool,
    points: [(hhmm, 价, 累计量, 累计额)]}]。保持上游新→旧顺序；坏行跳过，
    无有效点的日子不返回。"""
    out = []
    for day in node.get("data") or []:
        pts = []
        for raw in day.get("data") or []:
            p = str(raw).split()
            if len(p) < 4:
                continue
            try:
                price, vol, amt = float(p[1]), float(p[2]), float(p[3])
            except ValueError:
                continue
            if price > 0 and vol >= 0 and amt >= 0:
                pts.append((p[0], price, vol, amt))
        if pts:
            out.append({"date": str(day.get("date") or ""), "points": pts,
                        "complete": len(pts) >= COMPLETE_POINTS})
    return out


def fetch_minute_days(code: str) -> list[dict]:
    """个股近 5 个交易日分时（新→旧）。带工程统一重试（_http_get_json）。

    上游对该代码无任何数据（长期停牌/代码无效）→ 返回 []，不是错误；
    网络/解析类瞬时失败重试耗尽后向上抛，由调用方决定是否改日再采。
    """
    assert_public_url(TX_MINUTE_URL)
    symbol = to_tx_symbol(code)
    try:
        data = _http_get_json(TX_MINUTE_URL, {"code": symbol},
                              referer="https://gu.qq.com/")
    except EmptyDataError:
        return []
    return parse_minute_days(data.get(symbol) or {})
