# -*- coding: utf-8 -*-
"""交易日历与复盘日期解析。

交易日历是全管线的地基（窗口滚动 / 缓存清理 / 旁路模块的「是否交易日」
都靠它），所以这里做两件事，2026-11-24 补：

- **进程内 memo**：同一进程多次调用只拉一次。此前 main 与 vane.daily_sync
  各拉一次，每次都是一次 akshare 网络请求。
- **本地副本兜底**：每次成功拉取都落一份 out/trade_calendar.txt；拉取失败
  回退副本。此前失败即返回空日历，调用方各自退化（vane 甚至退化成
  「非空池=交易日」——历史池对非交易日回声最后一个交易日，周五的板会被
  数三次，再写进不可改写的预注册台账）。

空日历（[]）= 「不知道哪些天是交易日」，调用方必须显式处理，不得默认今天。
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import fstore
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

CALENDAR_FILE = os.path.join("out", "trade_calendar.txt")

_MEMO: Optional[list[dt.date]] = None
_MEMO_DAY: Optional[dt.date] = None


# ---------------------------------------------------------------- 交易日历

def _fetch() -> list[dt.date]:
    if ak is None:
        return []
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(cal["trade_date"]).dt.date
        return sorted(set(dates))
    except Exception as e:                                    # noqa: BLE001
        print(f"  [警告] 交易日历拉取失败: {type(e).__name__}: {e}")
        return []


def _load_copy() -> list[dt.date]:
    try:
        with open(CALENDAR_FILE, encoding="utf-8") as f:
            return [dt.date.fromisoformat(x) for x in f.read().split()]
    except (OSError, ValueError):
        return []


def trade_calendar(refresh: bool = False) -> list[dt.date]:
    """全量交易日历（升序）。memo → 网络拉取 → 本地副本 → []。

    refresh=True 强制重拉（长驻进程跨零点后自动按日期失效，无需手动）。
    """
    global _MEMO, _MEMO_DAY
    today = dt.date.today()
    if _MEMO and _MEMO_DAY == today and not refresh:
        return list(_MEMO)
    cal = _fetch()
    if cal:
        _MEMO, _MEMO_DAY = cal, today
        try:
            fstore.write_text(CALENDAR_FILE, "\n".join(f"{d:%Y-%m-%d}" for d in cal))
        except OSError as e:                                  # noqa: BLE001 副本失败不影响本次
            print(f"  [提示] 交易日历副本落盘失败: {e}")
        return list(cal)
    cached = _load_copy()
    if cached:
        tail = ("——副本未覆盖今天，交易日判断不可靠，请检查网络后重跑"
                if cached[-1] < today else "")
        print(f"  [警告] 交易日历拉取失败，使用本地副本"
              f"（{cached[0]} ~ {cached[-1]}）{tail}")
    return cached


def resolve_trade_date(arg: Optional[str],
                       calendar: Optional[list[dt.date]] = None) -> dt.date:
    """解析复盘日期：未指定时自动取最近一个已收盘交易日。

    calendar 可传入已拉取的交易日历，避免重复请求。
    """
    today = dt.date.today()
    if arg:
        return dt.datetime.strptime(arg, "%Y%m%d").date()

    cal = calendar if calendar is not None else trade_calendar()
    if cal:
        if today in cal:
            now = dt.datetime.now().time()
            from config import AFTER_CLOSE_BUFFER_MINUTES
            if now >= dt.time(15, AFTER_CLOSE_BUFFER_MINUTES):
                return today
        past = [d for d in cal if d < today]
        if past:
            if (today - past[-1]).days > 10:
                # 日历陈旧（本地副本过期）：past[-1] 不是「最近交易日」而是
                # 「日历知道的最后一天」——以它为锚滚动窗口会清掉更新的缓存
                print(f"  [警告] 交易日历截至 {past[-1]}，已落后今天 {today} "
                      f"{(today - past[-1]).days} 天：请先恢复网络再复盘")
            print(f"  [提示] 未到收盘确认时间，复盘最近交易日: {past[-1]}")
            return past[-1]
    print("  [提示] 交易日历不可用，默认使用今天（非交易日接口可能返回空）")
    return today
