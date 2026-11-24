# -*- coding: utf-8 -*-
"""日K获取层：三级路径（当日缓存 → 昨日缓存增量拼接 → 全量拉取）+ 连板近似。

增量拼接精度保障：快照昨收与缓存末根收盘一致性校验（容差 max(1分, 0.1%)），
除权/停牌/缺口自动触发全量重拉——实测增量与全量逐位一致。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from config import LIMIT_UP_APPROX, LIMIT_UP_APPROX_ST, MIN_BARS
from data_source import fetch_daily


def incremental_bars(cand: dict, cached, trade_date) -> pd.DataFrame | None:
    """昨日缓存日K + 今日快照 OHLCV 增量拼接（零网络）。

    仅当缓存末根日期早于复盘日、且快照昨收与缓存末根收盘一致时拼接，
    否则返回 None 走全量重拉。
    """
    if cached is None or len(cached) < 2:
        return None
    last = cached.iloc[-1]
    last_date = last["date"]
    if not isinstance(last_date, dt.date) or last_date >= trade_date:
        return None
    prev_close = cand.get("prev_close")
    if prev_close is None or prev_close != prev_close or prev_close <= 0:
        return None
    if abs(float(prev_close) - float(last["close"])) > max(0.011, 0.001 * float(prev_close)):
        return None  # 除权/送转/口径变化：不能直接拼
    vals = {k: cand.get(k) for k in ("open", "high", "low", "close", "volume")}
    if any(x is None or x != x for x in vals.values()):
        return None  # 快照缺当日字段 → 全量
    row = pd.DataFrame([{"date": trade_date, **{k: float(x) for k, x in vals.items()}}])
    return pd.concat([cached.reset_index(drop=True), row], ignore_index=True)


def drop_intraday_last(bars_cache: dict, trade_date) -> int:
    """丢弃「复盘日当天」那根日K（返回处理只数），用于上一轮是盘中运行的情形。

    盘中运行时末根由快照增量拼接或盘中接口产生：量不完整、价未定。此前
    收盘后重跑看到缓存末根日期 == 复盘日就直接复用——封板股的量比系统性
    偏低，而「盘中预览、收盘后再跑一次」正是文档推荐的工作流。砍掉末根后
    acquire_bars 会用收盘快照重新拼接（零网络），除权/缺口则自动全量重拉。
    """
    n = 0
    for code, b in list(bars_cache.items()):
        if (b is not None and len(b) >= 3
                and b["date"].iloc[-1] == trade_date):
            bars_cache[code] = b.iloc[:-1].reset_index(drop=True)
            n += 1
    return n


def acquire_bars(cand: dict, code: str, trade_date,
                  bars_cache: dict | None) -> tuple[pd.DataFrame | None, str]:
    """日K获取三级路径。返回 (bars, skip_reason)；skip_reason 非空表示不可用。"""
    cached = bars_cache.get(code) if bars_cache is not None else None
    bars = None
    if cached is not None and len(cached) >= 2 and cached["date"].iloc[-1] == trade_date:
        bars = cached                                  # 已含复盘日（如重跑），直接用
    elif cached is not None:
        bars = incremental_bars(cand, cached, trade_date)  # 昨日缓存 + 今日快照拼接
        if bars is not None and bars_cache is not None:
            bars_cache[code] = bars
    if bars is None:
        bars = fetch_daily(code, trade_date)
        if bars is not None and bars_cache is not None:
            bars_cache[code] = bars  # 新拉的 bars 写入缓存，升级算法时复用
    if bars is None:
        return None, "日K获取失败"
    if len(bars) < MIN_BARS:
        return None, f"次新(仅{len(bars)}根K线)"
    if bars["date"].iloc[-1] != trade_date:
        return None, "日K未覆盖复盘日(停牌/源异常)"      # 防止把旧K线当成当日数据
    return bars, ""


def approx_lianban(bars: pd.DataFrame, name: str) -> int:
    """日K近似连板数（连续收盘涨幅≥涨停近似阈值），涨停池连板缺失时的降级。"""
    close_arr = bars["close"].astype(float).to_numpy()
    up_th = LIMIT_UP_APPROX_ST if "ST" in str(name).upper() else LIMIT_UP_APPROX
    lb = 0
    for i in range(len(close_arr) - 1, max(len(close_arr) - 12, 1) - 1, -1):
        if float(close_arr[i] / close_arr[i - 1] - 1.0) >= up_th:
            lb += 1
        else:
            break
    return lb
