# -*- coding: utf-8 -*-
"""历史样本回灌：重建过去交易日的二板定龙头评分，写入 out/archive/ 供校准。

数据源：东财历史涨停/炸板池（push2ex，免费接口约保留最近 3 周）+ 腾讯日K
（区间任意取）。两源在东财行情集群断连期间依然可用。

与当日实跑的口径差异（跨日一致，分数桶间可比）：
- 赚钱效应不可重建（需昨日涨停股当日涨幅的全量数据）→ 统一按缺失计分并标注；
- 概念口径依赖行情集群 → A维自动走行业兜底；
- 只回灌「无官方归档」且池数据仍保留的日期，不覆盖正式运行的结果。

用法:
    python backfill.py                # 默认回看 21 个自然日内的可回灌交易日
    python backfill.py --days 30
    python backfill.py --dry-run      # 只列可回灌日期，不写文件
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time

import fstore
import pandas as pd

import data_source
import mood
from archive_schema import SIGNALS_CSV_COLUMNS
from config import EXCLUDE_ST, MAIN_BOARD_PREFIXES
from second_board import detect_second_board
from termstyle import weekday_cn


def _mb(code: str) -> bool:
    return code.startswith(MAIN_BOARD_PREFIXES)


def collect_trade_days(end_date: dt.date, calendar_span: int, calendar=None):
    """返回 [(date, zt_rows)]（新→旧），只含池数据仍在保留窗口内的交易日。

    交易日判定用交易日历——直接按自然日探测会把周末/节假日误判为交易日：
    历史池对非交易日会「回声」最后一个交易日的数据（实测 8/29 周六与 8/28
    池完全一致），回声日会被写成一份 signals 归档并混进 calibrate 样本池。
    日历不可用一律拒跑（此前「退化为逐日探测」正是这条事故的入口）。
    """
    cal = calendar if calendar is not None else data_source.trade_calendar()
    if not cal:
        print("  [失败] 交易日历不可用：无日历无法排除周末回声，拒绝回灌"
              "（请恢复网络后重跑）")
        return []
    tdays = [d for d in cal if d <= end_date][-40:]
    tdays = [d for d in tdays if d >= end_date - dt.timedelta(days=calendar_span)]
    days = []
    for d in sorted(tdays, reverse=True):   # 新 → 旧
        rows = data_source.parse_pool("zt", data_source.pool_raw("zt", f"{d:%Y%m%d}"))
        if rows:
            days.append((d, rows))
        time.sleep(0.1)
    return days


def ctx_from_pools(zt_rows, zb_rows, prev_zt_rows):
    """由三个池的解析行构建 detect_second_board 所需的 (ctx, extras)。

    纯数据变换，可离线单测。口径与 fetch_pool_extras 对齐：
    炸板率/晋级率/梯队按主板范围，题材家数按全市场。
    """
    industry_count: dict[str, int] = {}
    industry_max: dict[str, int] = {}
    ladder: dict[int, int] = {}
    today_2plus = 0
    sealed_mb = broken_mb = 0
    extras: dict[str, dict] = {}
    ind_mb: dict[str, int] = {}      # 今日主板行业家数（持续题材）
    ind_y_mb: dict[str, int] = {}    # 昨日主板行业家数

    for r in zt_rows:
        code = r["code"]
        slot = extras.setdefault(code, {})
        slot["lianban"] = r["lianban"]
        slot["fbt"] = r["fbt"]
        slot["industry"] = r["industry"]
        if r["fund"] == r["fund"]:
            slot["fund"] = r["fund"]
        if r["ltsz"] == r["ltsz"]:
            slot["ltsz"] = r["ltsz"]
        if r["industry"]:
            industry_count[r["industry"]] = industry_count.get(r["industry"], 0) + 1
            industry_max[r["industry"]] = max(industry_max.get(r["industry"], 0), r["lianban"])
        if _mb(code):
            sealed_mb += 1
            if r["industry"]:
                ind_mb[r["industry"]] = ind_mb.get(r["industry"], 0) + 1
            if r["lianban"] >= 2:
                today_2plus += 1
                ladder[r["lianban"]] = ladder.get(r["lianban"], 0) + 1

    for r in zb_rows:
        slot = extras.setdefault(r["code"], {})
        slot["zhaban_times"] = r["zhaban"]
        if r["lianban"]:
            slot["lianban"] = r["lianban"]
        if _mb(r["code"]):
            broken_mb += 1

    y_fbt: dict[str, str] = {}
    y_turn: dict[str, float] = {}
    y_seal: dict[str, float] = {}
    ind_y: dict[str, int] = {}
    prev_mb = 0
    for r in prev_zt_rows or []:
        if r["fbt"]:
            y_fbt[r["code"]] = r["fbt"]
        if r["turnover"] == r["turnover"]:
            y_turn[r["code"]] = r["turnover"]
        if r["fund"] == r["fund"] and r["amount"] > 0:
            y_seal[r["code"]] = round(r["fund"] / r["amount"], 2)
        if r["industry"]:
            ind_y[r["industry"]] = ind_y.get(r["industry"], 0) + 1
        if _mb(r["code"]):
            prev_mb += 1
            if r["industry"]:
                ind_y_mb[r["industry"]] = ind_y_mb.get(r["industry"], 0) + 1

    ctx = {
        "industry_count": industry_count,
        "industry_max_lianban": industry_max,
        "industry_yesterday_count": ind_y,
        "today_2plus_count": today_2plus or None,
        "promotion_rate": (today_2plus / prev_mb) if (today_2plus and prev_mb) else None,
        "yesterday_fbt": y_fbt,
        "yesterday_turnover": y_turn,
        "yesterday_seal": y_seal,
        "persistent_industries": {k for k, v in ind_mb.items()
                                  if v >= 2 and ind_y_mb.get(k, 0) >= 2} or None,
        "market_max_lianban": max(ladder) if ladder else None,
        "lianban_ladder": dict(sorted(ladder.items(), reverse=True)) or None,
        "break_rate": (broken_mb / (sealed_mb + broken_mb))
                      if (sealed_mb + broken_mb) else None,
        "zt_pool_ok": bool(zt_rows),
        "zb_pool_ok": bool(zb_rows),
    }
    return ctx, extras


def backfill_day(day: dt.date, zt_rows, prev_zt_rows, out_dir: str,
                 stage: str = "未知", dry_run: bool = False) -> tuple[int, int]:
    """回灌单个交易日：评分 2 板股并写 signals_{date}.csv。返回 (候选数, 入池数)。

    stage：该日情绪周期阶段（由 main 预计算序列判定），进 D 维周期槽，保持与实盘口径一致。
    """
    zb_rows = data_source.parse_pool("zb", data_source.pool_raw("zb", f"{day:%Y%m%d}"))
    ctx, extras = ctx_from_pools(zt_rows, zb_rows, prev_zt_rows)
    ctx["cycle_stage"] = stage

    targets = [r for r in zt_rows
               if r["lianban"] == 2 and _mb(r["code"])
               and not (EXCLUDE_ST and "ST" in r["name"].upper())]
    rows = []
    for r in targets:
        code, name = r["code"], r["name"]
        bars = data_source.fetch_close_range(
            code, day - dt.timedelta(days=90), day)
        if bars is None or bars["date"].iloc[-1] != day:
            rows.append({"trade_date": f"{day:%Y-%m-%d}", "weekday": weekday_cn(day),
                         "code": code, "name": name,
                         "zt_status": "封板", "close": r["close"],
                         "board_pass": 0, "board_score": 0, "board_note": "",
                         "skip_reason": "日K未覆盖当日",
                         "market_max_lianban": ctx["market_max_lianban"] or "",
                         "cycle_stage": stage, "source": "backfill"})
            continue
        ex = extras.get(code, {})
        cand = {"code": code, "name": name, "zt_status": "封板",
                "close": r["close"], "turnover": r["turnover"],
                "amount": r["amount"], "pct": r["pct"]}
        res = detect_second_board(cand, bars, ex, ctx)
        seal = round(ex.get("fund", 0) / r["amount"], 2) \
            if (ex.get("fund") and r["amount"] > 0) else ""
        rows.append({
            "trade_date": f"{day:%Y-%m-%d}", "weekday": weekday_cn(day),
            "code": code, "name": name,
            "zt_status": "封板", "close": r["close"], "pct": r["pct"],
            "turnover": r["turnover"], "amount": r["amount"],
            "industry": ex.get("industry", ""), "ltsz": ex.get("ltsz", ""),
            "lianban": ex.get("lianban", 2), "zhaban_times": ex.get("zhaban_times", ""),
            "fbt": ex.get("fbt", ""),
            "fund": ex.get("fund", ""), "seal_ratio": seal,
            "board_pass": int(res["board_pass"]), "board_score": res["board_score"],
            "board_note": res["board_note"], "skip_reason": "",
            "market_max_lianban": ctx["market_max_lianban"] or "",
            "cycle_stage": stage, "source": "backfill",
        })

    if dry_run:
        return len(targets), sum(x["board_pass"] for x in rows)
    # 与实跑归档同一张列清单（archive_schema）：回灌日没有的列留空，不再写 23 列子集
    # ——同名 signals_*.csv 此前因此有三种形态，读方只能各自防御
    df = pd.DataFrame(rows, columns=SIGNALS_CSV_COLUMNS)
    arch = os.path.join(out_dir, "archive")
    os.makedirs(arch, exist_ok=True)
    # 原子写：回灌窗口仅 3 周，截断 CSV = 该日样本永久丢失且会混入
    # calibrate 样本池污染分桶
    fstore.write_dataframe_csv(os.path.join(arch, f"signals_{day:%Y%m%d}.csv"), df)
    return len(targets), sum(x["board_pass"] for x in rows)


def main():
    ap = argparse.ArgumentParser(description="二板定龙头历史样本回灌")
    ap.add_argument("--days", type=int, default=21,
                    help="回看的自然日跨度（默认21；池接口只保留约3周）")
    ap.add_argument("--out", default="out", help="输出目录（默认 out）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = ap.parse_args()

    end = dt.date.today() - dt.timedelta(days=1)
    print(f"探测最近 {args.days} 个自然日的可回灌交易日（东财历史池）...")
    days = collect_trade_days(end, args.days)
    if not days:
        print("历史池无数据（可能已超出接口保留窗口或网络不可达）。")
        return
    days.sort(key=lambda x: x[0])   # 旧 → 新
    # 「已完成」判定不能只看 exists：截断的半截 CSV（历史崩溃残留）会被永久
    # 跳过回灌。要求存在且至少 1 行数据。
    def _archived(d: dt.date) -> bool:
        p = os.path.join(args.out, "archive", f"signals_{d:%Y%m%d}.csv")
        if not os.path.exists(p):
            return False
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                return any(csv.DictReader(f))
        except Exception:                      # noqa: BLE001 读不出=视为未归档
            return False

    todo = [(d, rows) for d, rows in days if not _archived(d)]
    print(f"交易日 {len(days)} 个（{days[0][0]} ~ {days[-1][0]}），"
          f"其中可回灌（无官方归档）{len(todo)} 个")
    if args.dry_run:
        for d, _ in todo:
            print(f"  可回灌: {d}")
        return

    # 预计算各日情绪指标（用已拉取的池，避免重复请求），逐日判定周期阶段
    zb_cache: dict = {}
    metrics_by_day: dict = {}
    for i, (d, rows) in enumerate(days):
        zb = zb_cache.get(d)
        if zb is None:
            zb = data_source.parse_pool("zb", data_source.pool_raw("zb", f"{d:%Y%m%d}"))
            zb_cache[d] = zb
        prev_rows = days[i - 1][1] if i > 0 else None
        m = mood.day_metrics(rows, prev_rows, zb)
        m["date"] = f"{d:%Y%m%d}"
        metrics_by_day[d] = m
    day_list = [d for d, _ in days]

    def stage_of(d: dt.date) -> str:
        i = day_list.index(d)
        hist = [metrics_by_day[dd] for dd in day_list[max(0, i - 3):i + 1]]
        return mood.classify_stage(hist)[0]

    total_c = total_p = 0
    for i, (d, rows) in enumerate(todo):
        prev_rows = None
        for pd_, pr in reversed(days):
            if pd_ < d:
                prev_rows = pr    # 取紧邻的前一个交易日（days 为升序，从新往旧找）
                break
        n, p = backfill_day(d, rows, prev_rows, args.out, stage=stage_of(d))
        total_c += n
        total_p += p
        print(f"  [{i + 1}/{len(todo)}] {d}: 2板股 {n} 只，入池 {p} 只")
        time.sleep(0.2)
    print(f"\n回灌完成：{len(todo)} 个交易日，2板股 {total_c} 只，入池 {total_p} 只。")
    print("下一步：python calibrate.py 产出分数桶校准报告（回灌样本的前向数据已全部可实现）。")


if __name__ == "__main__":
    main()
