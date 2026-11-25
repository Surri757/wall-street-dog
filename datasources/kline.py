# -*- coding: utf-8 -*-
"""个股日K：东财直连 → 腾讯直连 → 新浪（节流）三源降级；含熔断与增量取数。"""
from __future__ import annotations

import contextlib
import datetime as dt
import threading
import time
from typing import Optional

import pandas as pd

from config import HISTORY_DAYS
from .http import (EM_KLINE_URL, EmptyDataError, TX_KLINE_URL, _http_get_json)
from .util import to_secid, to_tx_symbol

try:
    import akshare as ak
except ImportError:
    ak = None


# ---------------------------------------------------------------- 日K

def _daily_em_direct(code: str, end_date: dt.date) -> pd.DataFrame:
    secid = to_secid(code)
    start = (end_date - dt.timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": "1",  # 日线 + 前复权
        "beg": start, "end": end_date.strftime("%Y%m%d"),
    }
    data = _http_get_json(EM_KLINE_URL, params,
                          referer=f"https://quote.eastmoney.com/{code}.html")
    klines = data.get("klines") or []
    rows = []
    for line in klines:
        p = str(line).split(",")
        if len(p) >= 6:
            rows.append({"date": p[0], "open": p[1], "close": p[2],
                         "high": p[3], "low": p[4], "volume": p[5]})
    return _finalize_bars(pd.DataFrame(rows))


def _turnover_em(code: str, start: dt.date, end: dt.date) -> dict:
    """东财日K 的 f61 字段（与 _daily_em_direct 同一端点、同一套表头）。"""
    params = {
        "secid": to_secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": "1",
        "beg": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    }
    try:
        data = _http_get_json(EM_KLINE_URL, params,
                              referer=f"https://quote.eastmoney.com/{code}.html",
                              retries=1)
    except Exception:                          # noqa: BLE001 补充字段：失败即空
        return {}
    out: dict = {}
    for line in data.get("klines") or []:
        p = str(line).split(",")
        if len(p) >= 11:
            try:
                out[p[0]] = float(p[10])
            except ValueError:                 # 停牌日等 "-" 占位
                continue
    return out


def fetch_turnover(code: str, start: dt.date, end: dt.date) -> dict:
    """个股区间内每日换手率（百分数）→ {"YYYY-MM-DD": 换手率}；取不到返回 {}。

    两源互备：东财日K → 新浪日K。东财会整晚主动断连（2026-11-25 实测，编号主机
    亦然），新浪是主流程日K降级链一直在用的源；两源数值逐日一致（同日实测全筑
    股份 11-20 / 11-23：13.9% / 15.8%）。只服务少数几只票的补充字段，不重试，
    失败返回空由调用方继续降级。
    """
    return _turnover_em(code, start, end) or _turnover_sina(code, start, end)


def _daily_tx_direct(code: str, end_date: dt.date) -> pd.DataFrame:
    """直连腾讯日K（前复权），列序: date, open, close, high, low, ..."""
    symbol = to_tx_symbol(code)
    start = (end_date - dt.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    params = {"param": f"{symbol},day,{start},{end},{HISTORY_DAYS + 60},qfq"}
    # 走 _http_get_json：此前是单发 get 无重试，一次抖动就白掉一个源
    data = _http_get_json(TX_KLINE_URL, params, referer="https://gu.qq.com/",
                          retries=2)
    node = data.get(symbol) or {}
    klines = node.get("qfqday") or node.get("day") or []
    rows = []
    for p in klines:
        if len(p) >= 6:
            rows.append({"date": p[0], "open": p[1], "close": p[2],
                         "high": p[3], "low": p[4], "volume": p[5]})
    return _finalize_bars(pd.DataFrame(rows))


_SINA_LOCK = threading.Lock()  # 新浪日K大量抓取易封 IP：串行 + 节流
_SINA_CALL_TIMEOUT = 20.0      # 秒：akshare 内部请求不带超时，靠外层兜底挂死


@contextlib.contextmanager
def _sina_guard():
    """调 akshare 新浪接口时的锁与超时保护（日K、换手率共用）。

    锁与请求都限时：此前在持锁状态下调用无超时的 akshare，连接一悬挂，
    锁永不释放，8 个工作线程全部卡死在 acquire 上。
    """
    if not _SINA_LOCK.acquire(timeout=_SINA_CALL_TIMEOUT):
        raise TimeoutError("新浪日K源忙（上一请求未返回），跳过")
    try:
        import socket
        prev = socket.getdefaulttimeout()
        # requests 未显式设超时时沿用 socket 默认值——只在这一个调用的窗口内
        # 临时生效，finally 恢复，不改宿主进程的全局默认
        socket.setdefaulttimeout(_SINA_CALL_TIMEOUT)
        try:
            yield
        finally:
            socket.setdefaulttimeout(prev)
    finally:
        _SINA_LOCK.release()


def _daily_sina(code: str, end_date: dt.date) -> pd.DataFrame:
    if ak is None:
        raise ImportError("akshare 未安装")
    symbol = to_tx_symbol(code)
    start = end_date - dt.timedelta(days=HISTORY_DAYS)
    with _sina_guard():
        raw = ak.stock_zh_a_daily(symbol=symbol,
                                  start_date=start.strftime("%Y%m%d"),
                                  end_date=end_date.strftime("%Y%m%d"),
                                  adjust="qfq")
        time.sleep(0.2)  # 节流，避免触发新浪封 IP
    if isinstance(raw, pd.DataFrame) and "volume" in raw.columns:
        # 新浪日K成交量单位是股，东财/腾讯是手：统一为手（跨源拼接/比值才一致）
        raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce") / 100.0
    return _finalize_bars(raw)


def _turnover_sina(code: str, start: dt.date, end: dt.date) -> dict:
    """新浪日K 的换手率列（=成交量/流通股本，小数口径）→ {date: %}；失败 {}。"""
    if ak is None:
        return {}
    try:
        with _sina_guard():
            raw = ak.stock_zh_a_daily(symbol=to_tx_symbol(code),
                                      start_date=start.strftime("%Y%m%d"),
                                      end_date=end.strftime("%Y%m%d"), adjust="")
            time.sleep(0.2)
    except Exception:                          # noqa: BLE001 节流/封锁/解析失败即空
        return {}
    if not isinstance(raw, pd.DataFrame) or "turnover" not in raw.columns:
        return {}
    out: dict = {}
    for d, v in zip(raw["date"], raw["turnover"]):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v == v:
            out[str(d)[:10]] = 100.0 * v
    return out


def _finalize_bars(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    need = ["date", "open", "close", "high", "low"]
    if raw is None or raw.empty or not all(c in raw.columns for c in need):
        # EmptyDataError：fetch_daily 据此区分「该股在此源无数据」与「源故障」
        raise EmptyDataError("日K数据为空或字段缺失")
    out = raw[need].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for c in ("open", "close", "high", "low"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "volume" in raw.columns:
        out["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        out["volume"] = float("nan")  # 二波检测需要量能；缺失时按 NaN 处理
    out = out.dropna(subset=["close", "high", "low"])
    out = out.sort_values("date").reset_index(drop=True)
    if len(out) < 2:
        raise EmptyDataError("日K有效数据不足")
    return out


_DAILY_OK_FN = None          # 记住首个成功的日K源，后续优先使用
_DAILY_OK_SINCE = 0.0        # 记忆起点（秒）：备用源记忆带 TTL，到期重探首选源
_SOURCE_MEMORY_TTL = 120.0   # 秒：东财/腾讯各瞬时失败一次就永久锁死新浪（最慢、
                             # 最易封 IP）；到期把首选源放回队首重试一次
_EM_DAILY_FAILS = 0          # 东财日K连续失败计数，>=3 后本次运行不再尝试
_EM_DAILY_MAX_FAILS = 3
_TX_DAILY_FAILS = 0          # 腾讯日K连续失败计数（同东财，避免每只白等超时）
_TX_DAILY_MAX_FAILS = 3
_DAILY_LOCK = threading.Lock()  # 上面的状态被并发线程读写，必须持锁变更


def _pick_daily_sources() -> list:
    with _DAILY_LOCK:
        ordered = [_daily_em_direct, _daily_tx_direct, _daily_sina]
        fresh = (_DAILY_OK_FN is not None
                 and (_DAILY_OK_FN is _daily_em_direct
                      or time.time() - _DAILY_OK_SINCE < _SOURCE_MEMORY_TTL))
        if fresh:
            ordered = [_DAILY_OK_FN] + [f for f in ordered if f is not _DAILY_OK_FN]
        return [f for f in ordered
                if not (f is _daily_em_direct and _EM_DAILY_FAILS >= _EM_DAILY_MAX_FAILS)
                and not (f is _daily_tx_direct and _TX_DAILY_FAILS >= _TX_DAILY_MAX_FAILS)]


def _daily_report_failure(fn) -> None:
    global _EM_DAILY_FAILS, _TX_DAILY_FAILS
    with _DAILY_LOCK:
        if fn is _daily_em_direct:
            _EM_DAILY_FAILS += 1
        if fn is _daily_tx_direct:
            _TX_DAILY_FAILS += 1


_DAILY_FAIL_LOGGED: set = set()  # 已打印过失败讯息的源（每源只报首次，防刷屏）


def _daily_report_success(fn) -> None:
    global _DAILY_OK_FN, _DAILY_OK_SINCE, _EM_DAILY_FAILS, _TX_DAILY_FAILS
    with _DAILY_LOCK:
        if fn is _daily_em_direct:
            _EM_DAILY_FAILS = 0
        if fn is _daily_tx_direct:
            _TX_DAILY_FAILS = 0
        if fn is not _DAILY_OK_FN:
            label = {_daily_em_direct: "东财直连(带UA)",
                     _daily_tx_direct: "腾讯直连",
                     _daily_sina: "新浪akshare"}[fn]
            if fn is _daily_sina:
                print(f"  [数据源] 日K切换至: {label}"
                      f"（新浪串行节流防封IP，预计每只约1.5秒）")
            else:
                print(f"  [数据源] 日K切换至: {label}")
            _DAILY_OK_FN = fn
            _DAILY_OK_SINCE = time.time()


def fetch_daily(code: str, end_date: dt.date) -> Optional[pd.DataFrame]:
    """单只股票前复权日K，降级链：东财直连 → 腾讯直连 → 新浪。

    首个成功源被记住（备用源的记忆带 TTL，到期重探首选源）；东财/腾讯连续
    失败 3 次后本次运行直接跳过（避免每只白等超时）——**只计源故障**，
    「该股在此源无数据」（退市/未上市/代码不存在）不计熔断。
    失败返回 None（由上层跳过）。每源首次失败打一行日志（此前整条链零日志，
    三源全败无声——对照 spot.py 的 [尝试]/[错误] 输出）。
    """
    for fn in _pick_daily_sources():
        try:
            df = fn(code, end_date)
        except EmptyDataError:
            continue        # 该股在此源无数据：换源，不计源故障、不刷日志
        except Exception as e:
            _daily_report_failure(fn)
            with _DAILY_LOCK:
                first = fn not in _DAILY_FAIL_LOGGED
                _DAILY_FAIL_LOGGED.add(fn)
            if first:
                label = {_daily_em_direct: "东财直连(带UA)",
                         _daily_tx_direct: "腾讯直连",
                         _daily_sina: "新浪akshare"}.get(fn, str(fn))
                print(f"  [尝试] 日K源 {label} 失败（{type(e).__name__}），降级下一源")
            continue
        _daily_report_success(fn)
        return df
    return None

