# -*- coding: utf-8 -*-
"""本轮换手率取数（换手门槛用，2026-11-25）。

涨停日的换手率就在当日涨停池行里（东财 hs）；断板日该票不在涨停池，只能取日K。
用户要求「一天都不可以少」——断板日换手率直接决定一只票进不进名单，不能指望
哪个接口当天心情好。所以是「本地库优先 + 多源互备 + 取到即入库」：

    ① 本地换手库  out/research/vane/turnover/<日期>.csv（code, 换手率%）
       每次运行写入：当天全市场快照的换手率（主流程本来就会下载全部股票的换手）、
       涨停池各行换手率、以及下面各源取到的数。零网络，先查这里。
    ② 在线两源互备：东财日K → 新浪日K（data_source.fetch_turnover）
    ③ baostock（turn；盘后约 17:35 入库）
    ④ 紫阳日K缓存、主工程近几日全市场快照（只读）

全部取不到的日子返回里就没有——调用方按「缺数据」如实标注并报警，
不猜数、不当成 0。

来由（2026-11-25 实测）：东财各主机整晚主动断连、baostock 登录超时，全筑股份的
两个断板日被判「缺数据」而挡在门槛外，而它那两天实际换手 13.9% / 15.8%，按规则
本该过线；同一时刻新浪日K可用，且数值与东财、baostock 逐日一致。
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import socket
from pathlib import Path

import fstore

from . import config as C

STORE_KEEP_DAYS = 80      # 本地换手库保留的文件数（≈ 4 个月交易日）

_MEMO: dict = {}          # (code, start, end, need) → {date: 换手率}，进程内有效
_DEAD: set = set()        # 本进程已判定不可用的在线源：一次失败即跳过，免得每只票白等
_ZIYANG_BARS = ...        # 紫阳日K缓存惰性加载一次：None=不可用；否则为 dict
_SPOT_MEMO: dict = {}     # 日期 → {code: 换手率%}（主工程快照，进程内有效）


# ---------------------------------------------------------------- 本地换手库
def _day_path(day: str) -> str:
    return os.path.join(C.TURNOVER_DIR, f"{day}.csv")


def _read_day(day: str) -> dict:
    """某日的本地换手库 {code: %}；文件不存在或读坏返回 {}。"""
    try:
        text = Path(_day_path(day)).read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    out: dict = {}
    for r in csv.DictReader(io.StringIO(text)):
        try:
            out[str(r["code"]).zfill(6)] = float(r["turnover"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def record(day: str, mapping: dict) -> None:
    """把一批 {code: 换手率%} 并入本地换手库的某一日（原子写、幂等）。空批不写。"""
    add = {}
    for code, v in (mapping or {}).items():
        x = _num(v)
        if x is not None:
            add[str(code).zfill(6)] = x
    if not add:
        return
    merged = {**_read_day(day), **add}
    fstore.write_csv(_day_path(day), ["code", "turnover"],
                     [{"code": c, "turnover": v} for c, v in sorted(merged.items())])


def record_snapshot(day: str) -> None:
    """把主工程当日全市场快照里的换手率入库（主流程已下载，零额外网络）。"""
    got = _snapshot(day)
    if got:
        record(day, got)


def prune(keep: int = STORE_KEEP_DAYS) -> None:
    """本地换手库滚动清理：只留最近 keep 个交易日的文件。"""
    try:
        files = sorted(f for f in os.listdir(C.TURNOVER_DIR)
                       if f.endswith(".csv") and len(f) == 14)
    except OSError:
        return
    for f in files[:-keep] if len(files) > keep else []:
        try:
            os.remove(os.path.join(C.TURNOVER_DIR, f))
        except OSError:
            pass


# ---------------------------------------------------------------- 取数
def daily_turnover(code: str, start: dt.date, end: dt.date,
                   need: list | None = None) -> dict:
    """补齐个股在 need 这些交易日（ISO 日期列表；缺省=区间内取到的全部）的换手率（%）。

    依次过：本地换手库 → 在线两源 → baostock → 紫阳缓存/近几日快照；每过一层
    都把新取到的数写回本地库，need 全部补齐即停。返回 {date: %}（可能仍缺）。
    """
    key = (code, start, end, tuple(need or ()))
    if key in _MEMO:
        return _MEMO[key]
    got: dict = {}
    # 各源签名不同，这里统一成无参调用（need 只对本地库有意义：它按日存、查得快）
    chain = (("store", lambda: _from_store(code, need)),
             ("online", lambda: _online_turnover(code, start, end)),
             ("baostock", lambda: _baostock_turnover(code, start, end)),
             ("local", lambda: _local_turnover(code, start, end)))
    for name, src in chain:
        if name in _DEAD:
            continue
        part = src()
        if not part:
            if name in ("online", "baostock"):
                _DEAD.add(name)      # 在线源整源不可用：本进程不再逐只白等
            continue
        new = {d: v for d, v in part.items() if d not in got}
        for d, v in new.items():
            record(d, {code: v})
        got.update(new)
        if need is not None and all(d in got for d in need):
            break
    _MEMO[key] = got
    return got


def _from_store(code: str, need) -> dict:
    out: dict = {}
    for d in need or ():
        v = _read_day(d).get(code)
        if v is not None:
            out[d] = v
    return out


def _online_turnover(code: str, start: dt.date, end: dt.date) -> dict:
    try:
        from data_source import fetch_turnover
        return fetch_turnover(code, start, end)
    except Exception:                          # noqa: BLE001 两源都不通 → 试下一层
        return {}


def _baostock_turnover(code: str, start: dt.date, end: dt.date) -> dict:
    """baostock 日换手率。库自身请求不带超时：登录时临时设 socket 默认超时
    （连接继承之），登录后立即恢复默认值——不改宿主进程的全局设置。"""
    try:
        import baostock as bs
    except ImportError:
        return {}
    prev = socket.getdefaulttimeout()
    if prev is None:
        socket.setdefaulttimeout(20)
    try:
        if bs.login().error_code != "0":
            return {}
    finally:
        socket.setdefaulttimeout(prev)
    sym = ("sh." if code.startswith("6") else "sz.") + code   # 本模块只看主板
    try:
        rs = bs.query_history_k_data_plus(
            sym, "date,turn", start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"), frequency="d", adjustflag="3")
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
    except Exception:                          # noqa: BLE001 读失败按取不到
        return {}
    out: dict = {}
    for r in rows:
        if len(r) >= 2:
            try:
                out[r[0]] = float(r[1])
            except (TypeError, ValueError):
                continue
    return out


def _local_turnover(code: str, start: dt.date, end: dt.date) -> dict:
    """本地已有的换手率（零网络）。① 紫阳日K缓存——它的候选宇宙基本覆盖强势股、
    历史长（换手率为百分数）；② 主工程近几日全市场快照——滚动窗口只留最近 3 个
    交易日（换手率为小数）。只读，不触发任何拉取。"""
    global _ZIYANG_BARS
    out: dict = {}
    if _ZIYANG_BARS is ...:
        try:
            from ziyang.seatdata import load_bars_cache
            _ZIYANG_BARS = load_bars_cache() or {}
        except Exception:                      # noqa: BLE001 紫阳未装/缓存损坏
            _ZIYANG_BARS = None
    df = (_ZIYANG_BARS or {}).get(code)
    if df is not None and len(df) and "turnover" in df.columns:
        for d, v in zip(df["date"], df["turnover"]):
            x = _num(v)
            if x is not None:
                out[str(d)[:10]] = x
    d = start
    while d <= end:
        snap = _snapshot(d.isoformat())
        x = snap.get(code)
        if x is not None:
            out.setdefault(d.isoformat(), x)
        d += dt.timedelta(days=1)
    return {d: v for d, v in out.items()
            if start.isoformat() <= d <= end.isoformat()}


def _snapshot(day: str) -> dict:
    """主工程某日快照的全市场换手率 {code: %}（只有最近 3 个交易日留得住）。"""
    if day not in _SPOT_MEMO:
        try:
            import cache_store
            spot = cache_store.load("out", dt.date.fromisoformat(day)).get("spot")
            rows = (spot[["code", "turnover"]].to_dict("records")
                    if spot is not None and len(spot) and "turnover" in spot.columns
                    else [])
            _SPOT_MEMO[day] = {
                str(r["code"]).zfill(6): 100.0 * float(r["turnover"])
                for r in rows if _num(r.get("turnover")) is not None}
        except Exception:                      # noqa: BLE001 快照缺失/损坏 → 无
            _SPOT_MEMO[day] = {}
    return _SPOT_MEMO[day]


def _num(v) -> float | None:
    try:
        return float(v) if v is not None and v == v else None
    except (TypeError, ValueError):
        return None
