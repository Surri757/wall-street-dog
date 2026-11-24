# -*- coding: utf-8 -*-
"""校准 CLI：装配 forward/features/sections/ledger_health/buckets 并落盘。

前向缓存 _forward.csv 的写入走防覆写保护（2026-09-15 起）：
- 本次重拉失败的旧缓存行**原样回写**（保留旧值、下次再试），不得从文件中消失；
- 输出行数相对旧文件大幅缩水（疑信号装载故障）时拒绝覆写。
这是 research_scan._safe_save 同族保护——2026-09-04 限流事故里 klines.pkl
曾因无保护被 1 只成功的新缓存覆盖。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

from concurrent.futures import ThreadPoolExecutor

import fstore
import pandas as pd

from termstyle import weekday_cn

from calib.buckets import bucket_stats
from calib.features import analyze_dims
from calib.forward import FWD_KEYS, fetch_forward, load_forward_cache, load_signals
from calib.ledger_health import ledger_section
from calib.sections import (_source_of, oos_section, source_section, summarize)


def _safe_write_forward(path: str, rows: list[dict], fwd_df, failed_keys: set) -> bool:
    """防覆写落盘 _forward.csv；返回是否写入。

    两层保护：
    1. 旧行保底回写——failed_keys 里的 (code, date) 若在旧文件中有行（stale 缺列
       或本次网络失败），把旧行原样拼回去：旧值可能不完整，但**存在**才能下次重试，
       消失即永久丢失（旧行既是 stale 又重拉失败是此前唯一漏数路径）。
    2. 总量下限——新表行数 < 旧表一半且旧表 >20 行时，判定装载异常（如 signals
       glob 失配），拒绝覆盖，旧文件原样保留。
    """
    carry = []
    if fwd_df is not None and not fwd_df.empty and failed_keys:
        for _, r in fwd_df.iterrows():
            k = (str(r.get("code", "")).zfill(6), str(r.get("date", ""))[:10])
            if k in failed_keys:
                carry.append({c: r.get(c) for c in fwd_df.columns})
    if carry:
        print(f"  [保护] {len(carry)} 行本次重拉失败，回写旧值待下次重试（不丢历史）")
    out = pd.DataFrame(rows + carry)
    old_n = 0 if (fwd_df is None or fwd_df.empty) else len(fwd_df)
    if old_n > 20 and len(out) < old_n * 0.5:
        print(f"  [保护] 新表仅 {len(out)} 行 < 旧表 {old_n} 行的 50%，"
              f"疑似信号装载故障——保留旧 {path} 不覆盖，请检查 archive/signals_*.csv")
        return False
    # 原子写（+ .bak）：截断的 _forward.csv 会让下次 read_csv 失败、fwd_df 清空，
    # _safe_write_forward 的旧行保底与缩水保护随之全部失效（依赖旧文件可读）
    fstore.write_with_bak(
        path, lambda tp: pd.DataFrame(out).to_csv(tp, index=False,
                                                  encoding="utf-8-sig"))
    return True


def main():
    ap = argparse.ArgumentParser(description="二板定龙头评分回测校准")
    ap.add_argument("--out", default="out", help="输出目录（默认 out）")
    ap.add_argument("--refresh", action="store_true", help="前向数据全部重拉")
    args = ap.parse_args()

    df = load_signals(args.out)
    if df.empty:
        print("out/archive/ 中暂无入池信号样本。请先在每个交易日收盘后运行 main.py 积累归档。")
        return

    fwd_path = os.path.join(args.out, "archive", "_forward.csv")
    fwd_df = pd.DataFrame()
    if os.path.exists(fwd_path) and not args.refresh:
        try:
            fwd_df = pd.read_csv(fwd_path, dtype={"code": str})
        except Exception:
            fwd_df = pd.DataFrame()

    cache, stale = load_forward_cache(fwd_df)
    if stale:
        print(f"  [缓存] {stale}/{len(fwd_df)} 行缺新口径列"
              f"（{'/'.join(FWD_KEYS[4:])}），本次将重新拉取")

    rows, pending, failed, skipped = [], 0, 0, 0
    failed_keys: set = set()
    # 解析串行、网络预取并行：每日新增 pending 约 10-25 条曾串行白等 5-30s。
    # fetch_close_range 自带 _RANGE_LOCK 线程安全（research_scan 同款先例）；
    # cache 命中行在线程里即时返回，不产生额外网络。
    parsed = []
    for _, r in df.iterrows():
        try:
            code = str(r["code"]).zfill(6)
            d = dt.datetime.strptime(str(r["trade_date"]), "%Y-%m-%d").date()
            score = int(r["board_score"])
            base = float(r["close"])
        except Exception:
            skipped += 1        # 坏行计数（_forward 样本本就少，静默跳行会让桶悄悄缩水）
            continue
        parsed.append((code, d, score, base, r))

    def _fetch(p):
        code, d, base = p[0], p[1], p[3]
        try:
            return fetch_forward(code, d, base, cache)
        except Exception:
            return ("fail", None)

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = list(pool.map(_fetch, parsed))

    for (code, d, score, base, r), (status, m) in zip(parsed, fetched):
        if status == "ok":
            rows.append({"code": code, "date": f"{d:%Y-%m-%d}", "board_score": score,
                         "name": r.get("name", ""),
                         "weekday": r.get("weekday", "") or weekday_cn(d),
                         "close": base, "ltsz": r.get("ltsz", ""),
                         "turnover": r.get("turnover", ""),
                         "seal_ratio": r.get("seal_ratio", ""),
                         "zhaban_times": r.get("zhaban_times", ""),
                         "board_note": r.get("board_note", ""),
                         "cycle_stage": r.get("cycle_stage", ""),
                         "vol_ratio": r.get("vol_ratio", ""),
                         "source": _source_of(r), **m})
        elif status == "nodata":
            pending += 1
        else:
            failed += 1
            failed_keys.add((code, f"{d:%Y-%m-%d}"))

    if rows:
        # 全量覆写（带防覆写保护）：失败旧行由 _safe_write_forward 回写保底。
        if failed:
            print(f"  ⚠ {failed} 条前向行情拉取失败，受保护回写待下次重试")
        _safe_write_forward(fwd_path, rows, fwd_df, failed_keys)

    print(f"信号样本(含未过线): {len(df)} 条；已实现前向数据: {len(rows)} 条；"
          f"待信号日后市: {pending} 条；行情拉取失败(下次自动重试): {failed} 条"
          + (f"；坏行跳过: {skipped} 条" if skipped else "") + "\n")
    if not rows:
        if failed:
            print(f"注意: {failed} 条信号的前向行情拉取失败（东财与腾讯均不可达），"
                  f"下次运行会自动重试。")
        if pending:
            print("暂无已实现的前向数据（信号日之后至少需 1 个交易日）。"
                  "请继续每日运行积累，之后执行 python calibrate.py")
        return
    # 二板表只用 board_score>0 的样本（N 字入池的首板行不得污染二板口径）
    board_rows = [r for r in rows if r["board_score"] > 0]
    report = summarize(board_rows)
    print(report)
    dims = analyze_dims(board_rows)
    print("\n【维度区分度分析】")
    print(dims)
    oos = oos_section(board_rows)
    print(oos)
    ssec = source_section(board_rows)
    print(ssec)
    lsec = ledger_section(args.out)
    print(lsec)

    # 经验预期定价：各分数桶的历史前向分布 → JSON，供每日报告标注「同档历史预期」
    # （bucket_stats.json 截断会让报告的「同档实测」标注静默消失——原子写）
    import json
    stats = bucket_stats(board_rows)
    if stats:
        fstore.write_text(os.path.join(args.out, "archive", "bucket_stats.json"),
                          json.dumps(stats, ensure_ascii=False, indent=1))

    txt_path = os.path.join(args.out, "calibration_report.txt")
    fstore.write_text(
        txt_path,
        f"二板定龙头评分校准报告  生成: {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
        + report + "\n\n【维度区分度分析】\n" + dims + "\n" + oos + "\n"
        + ssec + "\n" + lsec + "\n")
    print(f"\n已写入: {txt_path}")
