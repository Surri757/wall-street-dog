# -*- coding: utf-8 -*-
"""概念题材：板块快照/个股概念/板块日K；以及区间日K取数（校准/回灌前向收益用）。"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Optional

import pandas as pd

from .http import (EM_HEADERS, EM_KLINE_URL, EM_SPOT_HOSTS, EM_SPOT_PATH,
                   TX_KLINE_URL, _http_get_json, _SESSION)
from .util import num as _num, to_int as _to_int, to_secid, to_tx_symbol


# ---------------------------------------------------------------- 概念题材

def fetch_concept_boards() -> Optional[dict]:
    """东财概念板块行情快照（fs=m:90 t:3）：一次分页拉全量（约500个）。

    返回 {板块名: {"pct": 涨跌幅, "up": 上涨家数, "down": 下跌家数,
                  "lead": 领涨股名, "bk": 板块代码}}；失败返回 None（上层回退行业口径）。
    """
    base = {
        "pn": 1, "pz": 100, "po": 1, "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": 2, "invt": 2,
        "fid": "f3", "fs": "m:90 t:3 f:!50",
        "fields": "f12,f14,f3,f104,f105,f128",
    }
    headers = dict(EM_HEADERS, Referer="https://data.eastmoney.com/bkzj/hy.html")
    rows: list[dict] = []
    for host in EM_SPOT_HOSTS:
        try:
            page, total = 1, None
            while True:
                data = _http_get_json(f"https://{host}{EM_SPOT_PATH}",
                                      dict(base, pn=page), referer=headers["Referer"],
                                      timeout=8)
                diff = data.get("diff")
                if not isinstance(diff, list) or not diff:
                    break
                rows.extend(diff)
                total = int(data.get("total") or 0)
                if not total or page * base["pz"] >= total:
                    break
                page += 1
                time.sleep(0.12)
            if rows:
                break
        except Exception:
            rows = []
            continue
    if not rows:
        return None
    out = {}
    for r in rows:
        try:
            name = str(r.get("f14", "")).strip()
            if not name:
                continue
            out[name] = {
                "pct": _num(r.get("f3")),
                "up": _to_int(r.get("f104")),
                "down": _to_int(r.get("f105")),
                "lead": str(r.get("f128", "")).strip(),
                "bk": str(r.get("f12", "")).strip(),
            }
        except Exception:
            continue
    return out or None


def fetch_stock_concepts(code: str, concept_names: set) -> list[str]:
    """个股所属板块（slist spt=3）∩ 当日概念板块名单 → 该股的概念列表。

    用名单求交过滤掉行业/地域板块，规避接口类型标记的口径不确定性。
    失败返回空列表（上层回退行业口径）。
    """
    if not concept_names:
        return []
    secid = to_secid(code)
    params = {"spt": 3, "fltt": 2, "invt": 2,
              "fields": "f12,f13,f14,f3", "secid": secid,
              "ut": "bd1d9ddb04089700cf9c27f6f7426281"}
    for host in EM_SPOT_HOSTS:
        try:
            data = _http_get_json(f"https://{host}/api/qt/slist/get", params,
                                  referer=f"https://quote.eastmoney.com/{code}.html",
                                  timeout=6)
            diff = data.get("diff")
            if not isinstance(diff, list):
                continue
            names = [str(r.get("f14", "")).strip() for r in diff
                     if str(r.get("f14", "")).strip() in concept_names]
            return sorted(set(names))
        except Exception:
            continue
    return []


def fetch_board_closes(bk: str, days: int = 6) -> Optional[list[float]]:
    """概念板块近 N 根日K收盘（secid=90.BKxxxx），用于题材发酵度（今日 vs 前几日均）。"""
    params = {
        "secid": f"90.{bk}", "fields1": "f1,f2,f3", "fields2": "f51,f53",
        "klt": "101", "fqt": "1", "lmt": str(days), "end": "20500101",
    }
    try:
        data = _http_get_json(EM_KLINE_URL, params,
                              referer="https://quote.eastmoney.com/center/boardlist.html")
        klines = data.get("klines") or []
        closes = [float(str(k).split(",")[1]) for k in klines if len(str(k).split(",")) >= 2]
        return closes if len(closes) >= 2 else None
    except Exception:
        return None


_RANGE_OK = None   # 源记忆：记住首个成功源直接复用。东财集群断连时，
                   # 每股先白等 ~20s（3重试×6s超时+休眠）才降级，3048 只=数小时。
_RANGE_LOCK = threading.Lock()   # _RANGE_OK 无锁读写防护：research_scan 以
                                 # ThreadPoolExecutor(6) 并发调用本函数
                                 # （kline.py 对同类状态用 _DAILY_LOCK，同款）


def fetch_close_range(code: str, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
    """某股 [start, end] 区间日K（date/open/close/high/low），供回测校准取前向收益。

    降级链：东财 kline → 腾讯 fqkline；带源记忆——首次探测东财单次快试
    （retries=1），失败即锁定腾讯，后续零无效重试。锁定源**本次调用**失败
    （断连/限流）时降级试另一源一次再放弃：此前锁定后源异常直接返回 None、
    腾讯永不再试（与 docstring 宣称的 fetch_daily 逐次换源模式不符）。
    """
    global _RANGE_OK
    with _RANGE_LOCK:
        ok = _RANGE_OK
    if ok is not None:
        try:
            df = ok(code, start, end)
        except Exception:
            df = None
        if df is None:
            alt = _close_range_tx if ok is _close_range_em else _close_range_em
            try:
                df = alt(code, start, end)
            except Exception:
                df = None
        return df
    df = _close_range_em(code, start, end, retries=1)
    if df is not None:
        with _RANGE_LOCK:
            _RANGE_OK = _close_range_em
        return df
    df = _close_range_tx(code, start, end)
    if df is not None:
        with _RANGE_LOCK:
            _RANGE_OK = _close_range_tx
    return df


def _close_range_em(code: str, start: dt.date, end: dt.date,
                    retries: int = None) -> Optional[pd.DataFrame]:
    secid = to_secid(code)
    params = {
        "secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "klt": "101", "fqt": "1",
        "beg": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
    }
    try:
        data = _http_get_json(EM_KLINE_URL, params,
                              referer=f"https://quote.eastmoney.com/{code}.html",
                              retries=retries)
        rows = []
        for line in data.get("klines") or []:
            p = str(line).split(",")
            if len(p) >= 5:
                row = {"date": p[0], "open": float(p[1]), "close": float(p[2]),
                       "high": float(p[3]), "low": float(p[4])}
                if len(p) >= 6:
                    row["volume"] = float(p[5])   # 手（缩量因子用）
                rows.append(row)
        return _finalize_close_range(rows)
    except Exception:
        return None


def _close_range_tx(code: str, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
    """腾讯日K（前复权）：列序 date, open, close, high, low, volume。"""
    symbol = to_tx_symbol(code)
    params = {"param": f"{symbol},day,{start:%Y-%m-%d},{end:%Y-%m-%d},320,qfq"}
    try:
        r = _SESSION.get(TX_KLINE_URL, params=params,
                         headers=dict(EM_HEADERS, Referer="https://gu.qq.com/"),
                         timeout=6)
        r.raise_for_status()
        node = (r.json().get("data") or {}).get(symbol, {})
        klines = node.get("qfqday") or node.get("day") or []
        rows = []
        for p in klines:
            if isinstance(p, (list, tuple)) and len(p) >= 5:
                row = {"date": str(p[0]), "open": float(p[1]), "close": float(p[2]),
                       "high": float(p[3]), "low": float(p[4])}
                if len(p) >= 6:
                    row["volume"] = float(p[5])   # 手（缩量因子用）
                rows.append(row)
        return _finalize_close_range(rows)
    except Exception:
        return None


def _finalize_close_range(rows: list[dict]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    return df.reset_index(drop=True)
