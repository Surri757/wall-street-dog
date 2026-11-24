# -*- coding: utf-8 -*-
"""CLI 入口：python -m ziyang {update|predict|verify|daily|report}

管道流：update（席位明细+快照+画像）→ verify（核验到期预判）→ predict（明日预判+登记）。
daily 链把快照从 update 传给 predict，全链只拉一次快照。
"""
from __future__ import annotations

import sys

from . import config as C


def _cmd_update(snapshot=None):
    from .profile import build_profile
    from .seatdata import (fetch_snapshot_any, float_shares_map, industry_map,
                           load_bars_cache, save_bars_cache, update_seat_trades)
    print(f"[1/3] 拉取席位明细（{C.SEAT_SHORT} {C.SEAT_CODE}）…")
    seat = update_seat_trades()
    print(f"      {len(seat)} 个 (日,股) 事件，最新 {seat['交易日期'].max()}")
    if snapshot is None:
        print("[2/3] 全市场快照（含量比/流通市值/行业，东财→新浪兜底）…")
        snapshot = fetch_snapshot_any()
    print(f"      {len(snapshot)} 条（{snapshot.attrs.get('source', '东财扩展快照')}）")
    print("[3/3] 挖掘画像（点火前夜特征，需逐股拉日K，首次较慢）…")
    cache = load_bars_cache()
    p = build_profile(seat, cache, float_shares_map(snapshot), industry_map(snapshot))
    save_bars_cache(cache)
    eve = p.get("eve", {})
    print(f"      画像完成：点火事件 {eve.get('n_events', 0)} / "
          f"{eve.get('n_days', 0)} 交易日 / {eve.get('n_stocks', 0)} 只"
          f" → {C.PROFILE_JSON}")
    return p


def _cmd_predict(snapshot=None, dry_run: bool = False):
    from . import ledger
    from .report import render, save
    from .screen import predict
    pred = predict(snapshot=snapshot)   # trade_date 由交易日历锚定（跨零点安全）
    msg = ("（dry-run 不登记）" if dry_run
           else ledger.register(pred["trade_date"], pred["predictions"],
                                regen_same_day=True, ranked=pred["all"]))
    print(msg)
    txt = render(pred=pred)
    save(txt)
    print(txt)


def _cmd_verify():
    import datetime as _dt
    from . import ledger
    from .report import render, save
    from .seatdata import load_seat_trades
    seat = load_seat_trades()
    print(ledger.verify(_dt.date.today(), seat))

    # 召回对照：最近已核验日，席位真实主板净买 vs 台账 topN（漏判清单）
    surprises = None
    rows = [r for r in ledger._load() if r["status"] == "verified"]
    if rows:
        vday = max(r["verify_date"] for r in rows)
        pday = next(r["predict_date"] for r in rows if r["verify_date"] == vday)
        reg = {r["code"] for r in rows if r["predict_date"] == pday}
        surprises = ledger.seat_surprises(_dt.date.fromisoformat(vday), reg, seat)
    # 漏判归因：所有已核验且已落排名快照的预判日
    for d in sorted({r["predict_date"] for r in rows}):
        ledger.autopsy(_dt.date.fromisoformat(d), seat)

    txt = render(accuracy=ledger.accuracy_summary(), surprises=surprises)
    save(txt)
    print(txt)


def _cmd_daily():
    import time as _time

    from .seatdata import fetch_snapshot_any
    t: dict = {}
    t0 = _time.time(); snap = fetch_snapshot_any(); t["快照"] = _time.time() - t0   # 全链只拉一次快照
    t0 = _time.time(); _cmd_update(snapshot=snap); t["更新"] = _time.time() - t0
    t0 = _time.time(); _cmd_verify(); t["核验"] = _time.time() - t0
    t0 = _time.time(); _cmd_predict(snapshot=snap); t["预判"] = _time.time() - t0
    brk = "/".join("%s%.0fs" % (k, v) for k, v in t.items())
    print("daily 耗时 %.0fs（%s）" % (sum(t.values()), brk))


def _cmd_report():
    from . import ledger
    from .report import render, save
    txt = render(accuracy=ledger.accuracy_summary())
    save(txt)
    print(txt)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "daily"
    if cmd == "update":
        _cmd_update()
    elif cmd == "predict":
        _cmd_predict(dry_run="--dry-run" in argv)
    elif cmd == "verify":
        _cmd_verify()
    elif cmd == "daily":
        _cmd_daily()
    elif cmd == "report":
        _cmd_report()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
