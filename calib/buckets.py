# -*- coding: utf-8 -*-
"""分数桶：分段方案 + bucket_stats.json 的统计构造（单一实现）。

此前分桶数学存在两份：sections._table 渲染一份、cli main() 尾部落盘 JSON 又
算一份——两处各自维护，改口径时极易只改一半。本模块是分桶统计的唯一实现，
渲染（sections）与落盘（cli）都从这里取数。
"""
from __future__ import annotations

import datetime as dt

from calib.forward import MAIN_METRIC
from calib.stats import cluster_mean

BUCKETS = [(0, 59), (60, 69), (70, 79), (80, 100)]


def bucket_segments(rows: list[dict], score_key: str = "board_score") -> list:
    """按 BUCKETS 切分 → [(label, seg), ...]；空段也保留（与表行一一对应）。"""
    out = []
    for lo, hi in BUCKETS:
        seg = [r for r in rows if lo <= (r.get(score_key) or 0) <= hi]
        out.append((f"{lo}-{hi}", seg))
    return out


def bucket_stats(board_rows: list[dict]) -> dict:
    """各分数桶的历史前向分布 → bucket_stats.json 内容（供报告「同档实测」标注）。

    信息口径键（open/close/win/promo）被 report._detail 与历史对照依赖，不得删；
    可成交口径（net/fillable_rate 等）自 2026-09-10 起为主口径。
    """
    stats = {}
    for label, seg in bucket_segments(board_rows):
        if len(seg) >= 5:
            n = len(seg)
            fill = [r for r in seg if r.get("fillable")]
            st = cluster_mean(fill, MAIN_METRIC) if fill else None
            stats[label] = {
                "n": n,
                # 信息口径（保留：report._detail 与历史对照都依赖这几个键，不得删）
                "open": round(sum(r["next_open_pct"] for r in seg) / n, 2),
                "close": round(sum(r["next_close_pct"] for r in seg) / n, 2),
                "win": round(sum(1 for r in seg if r["next_open_pct"] > 0) / n, 3),
                "promo": round(sum(r["promoted"] for r in seg) / n, 3),
                # 可成交口径（2026-09-10 起为主口径）：net 扣双边成本，
                # fillable_rate 直接告诉读者「这一档有多大比例你根本买不到」
                "fillable_n": len(fill),
                "fillable_rate": round(len(fill) / n, 3),
                "tradable": (round(sum(r["tradable_ret"] for r in fill) / len(fill), 2)
                             if fill else None),
                "net": round(st["mean"], 2) if st and st["mean"] is not None else None,
                "net_t": round(st["t"], 2) if st and st["t"] is not None else None,
                "days": st["clusters"] if st else 0,
                "updated": f"{dt.date.today():%Y-%m-%d}",
            }
    return stats
