# -*- coding: utf-8 -*-
"""多席位画像挖掘 CLI：python -m ziyang.mine_seat {席位key}

对 SEATS 注册表中的任意席位执行与紫阳东路同标准的画像挖掘：
席位明细全量拉取 → 快照 → build_profile（12 节画像，日K缓存市场级共享）
→ 纯文本画像报告落盘。

注意：非默认席位**只做画像挖掘**，不进预判/台账/主流程（主流程恒用默认席位
紫阳东路）。给某席位上预判是独立决策，须另行预注册。
"""
from __future__ import annotations

import sys

from . import config as C


def mine(key: str) -> dict:
    C.use_seat(key)
    from .profile import build_profile
    from .report import render, save
    from .seatdata import (fetch_snapshot_any, float_shares_map, industry_map,
                           load_bars_cache, save_bars_cache, update_seat_trades)

    print(f"[1/3] 拉取席位明细（{C.SEAT_SHORT} {C.SEAT_CODE}）…")
    seat = update_seat_trades()
    print(f"      {len(seat)} 个 (日,股) 事件，最新 {seat['交易日期'].max()}")
    print("[2/3] 全市场快照（含量比/流通市值/行业，东财→新浪兜底）…")
    snap = fetch_snapshot_any()
    print(f"      {len(snap)} 条（{snap.attrs.get('source', '东财扩展快照')}）")
    print("[3/3] 挖掘画像（前夜特征/市值力度/卖出纪律/环境节奏/时点，日K共享缓存）…")
    cache = load_bars_cache()
    p = build_profile(seat, cache, float_shares_map(snap), industry_map(snap))
    save_bars_cache(cache)

    save(render())
    eve = p.get("eve", {})
    print(f"      画像完成：点火事件 {eve.get('n_events', 0)} / "
          f"{eve.get('n_days', 0)} 交易日 / {eve.get('n_stocks', 0)} 只 → {C.PROFILE_JSON}")
    print(f"      报告 → {C.REPORT_TXT}")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        print("已注册席位：", ", ".join(f"{k}={v['short']}" for k, v in C.SEATS.items()))
        return 2
    mine(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
