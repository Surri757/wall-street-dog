# -*- coding: utf-8 -*-
"""候选池构建：全市场快照 → 主板候选（向量化，纯数据变换，零网络）。

口径：仅沪深主板（ST 剔除）；候选池 = 曾涨停（含炸板）/ 曾跌停 / 振幅≥7%，
仅供情绪速览统计；二板判定只针对「收盘封板」股。
候选额外携带当日 open/volume（bars 增量拼接用；缺失按 NaN 容忍）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    AMP_THRESHOLD, EXCLUDE_ST, LIMIT_NORMAL, LIMIT_ST,
    MAIN_BOARD_PREFIXES,
)
from trade_rules import limit_down_price, limit_price

EPS = 1e-6


def is_main_board(code: str) -> bool:
    return code.startswith(MAIN_BOARD_PREFIXES)


def limit_prices_vec(prev: np.ndarray, ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化涨跌停价（交易所四舍五入到分）。与 trade_rules.limit_price 同一个
    整数分值式子 `(分值×百分比+50)//100`——两条路径的恒等由
    test_limit_price_matches_exchange 在全价位网格上强制。"""
    cents = np.rint(prev * 100.0).astype(np.int64)          # 价格本身两位小数
    m_up = np.rint((1.0 + ratio) * 100.0).astype(np.int64)   # 110 / 105
    m_dn = np.rint((1.0 - ratio) * 100.0).astype(np.int64)   # 90 / 95
    return ((cents * m_up + 50) // 100) / 100.0, ((cents * m_dn + 50) // 100) / 100.0


def limit_prices(prev_close: float, name: str) -> tuple[float, float]:
    """(涨停价, 跌停价)：委托 trade_rules 的精确实现，本模块不留第二份舍入。"""
    ratio = LIMIT_ST if "ST" in str(name).upper() else LIMIT_NORMAL
    return (limit_price(prev_close, ratio), limit_down_price(prev_close, ratio))


def build_candidates(spot: pd.DataFrame) -> list[dict]:
    """从全市场快照筛出主板候选池（向量化，5400 行毫秒级）。"""
    if spot is None or spot.empty:
        return []

    def _col(name: str) -> pd.Series:
        if name in spot.columns:
            return pd.to_numeric(spot[name], errors="coerce")
        return pd.Series(np.nan, index=spot.index)

    code = spot["code"].astype(str).str.strip().str.zfill(6)
    # 东财把 3 字股票名补成「南 京 港」这种**内部带空格**的形式，strip() 去不掉。
    # 内部空格会被算进显示宽度，破坏所有按 _pad 对齐的名称列，且会污染归档 CSV。
    name = spot["name"].astype(str).map(lambda x: "".join(x.split()))
    prev_close = _col("prev_close")
    high, low, close = _col("high"), _col("low"), _col("close")

    st = name.str.upper().str.contains("ST", na=False).to_numpy()
    # to_numpy() 在 pandas CoW（2.x 默认开启）下可能返回只读视图，后面 ok &= ...
    # 需要原地写入——首取即 copy，阻断 read-only 崩溃
    ok = code.str[:3].isin(set(MAIN_BOARD_PREFIXES)).to_numpy().copy()
    ok &= ~(prev_close.isna() | high.isna() | low.isna() | close.isna()).to_numpy()
    ok &= (prev_close > 0).to_numpy() & (close > 0).to_numpy()
    ok &= (high > 0).to_numpy() & (low > 0).to_numpy()
    if EXCLUDE_ST:
        ok &= ~st

    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return []
    prev = prev_close.to_numpy()[idx]
    ratio = np.where(st[idx], LIMIT_ST, LIMIT_NORMAL)
    limit_up, limit_down = limit_prices_vec(prev, ratio)
    hi, lo, cl = high.to_numpy()[idx], low.to_numpy()[idx], close.to_numpy()[idx]
    touched_up = hi >= limit_up - EPS
    touched_down = lo <= limit_down + EPS
    sealed_up = cl >= limit_up - EPS
    sealed_down = cl <= limit_down + EPS
    amp = (hi - lo) / prev * 100.0
    sel = touched_up | touched_down | (amp >= AMP_THRESHOLD - EPS)

    o = _col("open").to_numpy()[idx]
    v = _col("volume").to_numpy()[idx]
    pct = _col("pct").to_numpy()[idx]
    amount = _col("amount").to_numpy()[idx]
    turn = _col("turnover").to_numpy()[idx]
    code_a, name_a = code.to_numpy()[idx], name.to_numpy()[idx]

    out: list[dict] = []
    for i in np.nonzero(sel)[0]:
        pools: list[str] = []
        if touched_up[i]:
            pools.append("涨停")
        if touched_down[i]:
            pools.append("跌停")
        if amp[i] >= AMP_THRESHOLD - EPS and not touched_up[i] and not touched_down[i]:
            pools.append("振幅")
        out.append({
            "code": code_a[i], "name": name_a[i], "pools": pools,
            "zt_status": "封板" if sealed_up[i] else ("炸板" if touched_up[i] else ""),
            "dt_status": "封死" if sealed_down[i] else ("翘板" if touched_down[i] else ""),
            "close": cl[i], "prev_close": prev[i],
            "high": hi[i], "low": lo[i],
            "limit_up": limit_up[i], "limit_down": limit_down[i],
            "amp": amp[i], "open": o[i], "volume": v[i],
            "pct": float(pct[i]) if pct[i] == pct[i] else float("nan"),
            "amount": float(amount[i]) if amount[i] == amount[i] else 0.0,
            "turnover": float(turn[i]) if turn[i] == turn[i] else 0.0,
        })
    return out
