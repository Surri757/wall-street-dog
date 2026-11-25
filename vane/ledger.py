# -*- coding: utf-8 -*-
"""台账层：情绪风向标预注册账（先标签后计分纪律——只记录晋级结果，不做预测）。

生命周期：scan 日登记（date+code，一经写入不得改写，当日重跑除外）→
下一交易日核验回填 promoted/next_pct → summary 汇总累计晋级率。

核验口径：
- promoted：复盘日后首根日K收盘涨幅 ≥ PROMOTE_APPROX(9.5%) 即晋级成功
  （主工程晋级判定单一口径）；日K拉不到时降级用次日涨停池在池判定
  （在池=Y，next_pct 留空）。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path

import fstore
from data_source import parse_pool, pool_raw

from . import config as C

LEDGER_COLS = ["date", "rank", "code", "name", "group", "label", "B", "span",
               "breaks", "seal_ratio", "fbt", "industry", "close",
               "model_version", "model_score", "shadow_rank", "score100",
               "round_dates",
               "status", "verify_date", "promoted", "next_pct", "note",
               # 换手门槛留痕（2026-11-25）：本轮交易日数 / 换手≥10% 天数 /
               # 单日最高换手% / 逐日换手%（"|" 分隔，缺=—）
               "turn_days", "turn_hi", "turn_max", "turn_seq"]

# 晋级判定阈值（百分数口径，来源 config.PROMOTE_APPROX 单一事实源）
_PROMOTE_PCT = C.PROMOTE_APPROX * 100.0


def _load() -> list[dict]:
    if not os.path.exists(C.LEDGER_CSV):
        return []
    with open(C.LEDGER_CSV, encoding="utf-8-sig", newline="") as f:
        return [{c: r.get(c, "") for c in LEDGER_COLS} for r in csv.DictReader(f)]


def _save(ledger: list[dict]) -> None:
    # 原子写（fstore tmp+replace）：预注册台账是纪律证据，截断=证据损坏
    fstore.write_csv(C.LEDGER_CSV, LEDGER_COLS, ledger, quoting=csv.QUOTE_NONNUMERIC)


def rows_for(trade_date: dt.date) -> list[dict]:
    """某复盘日的登记行（按 rank 排序）——渲染层只读入口，零网络。"""
    key = trade_date.isoformat()
    rows = [r for r in _load() if r["date"] == key]
    return sorted(rows, key=lambda r: int(r["rank"]) if str(r["rank"]).isdigit() else 99)


def register(trade_date: dt.date, picks: list[dict],
             regen_same_day: bool = False) -> str:
    """登记当日记选（预注册纪律；与 ziyang.ledger.register 同款守则：
    跨日冻结，当日重跑允许重登——收盘后池数据偶有延迟补全）。"""
    ledger = _load()
    key = trade_date.isoformat()
    if any(r["date"] == key and r["code"] for r in ledger):
        if regen_same_day and trade_date == dt.date.today():
            ledger = [r for r in ledger if r["date"] != key]
        else:
            return f"{trade_date} 已登记，拒绝覆写（预注册纪律）"
    for p in picks:
        ledger.append({
            "date": key, "rank": p["rank"], "code": p["code"], "name": p["name"],
            "group": p["group"], "label": p["label"], "B": p["B"],
            "span": p["span"], "breaks": p["breaks"],
            "seal_ratio": p["seal_ratio"] if p.get("seal_ratio") is not None else "",
            "fbt": p.get("fbt", ""), "industry": p.get("industry", ""),
            "close": p.get("close", ""),
            "model_version": p.get("model_version", ""),
            "model_score": (round(p["model_score"], 4)
                            if p.get("model_score") is not None else ""),
            "shadow_rank": p.get("shadow_rank", ""),
            "score100": p.get("score100", ""),
            "round_dates": "|".join(p.get("round_dates") or []),
            "status": "pending", "verify_date": "", "promoted": "",
            "next_pct": "", "note": "",
            "turn_days": p.get("turn_days", ""), "turn_hi": p.get("turn_hi", ""),
            "turn_max": p.get("turn_max", ""), "turn_seq": p.get("turn_seq", ""),
        })
    ledger.sort(key=lambda r: (r["date"], str(r["rank"]).zfill(2)))
    _save(ledger)
    return f"{trade_date} 登记 {len(picks)} 条风向标"


def mark_scanned(trade_date: dt.date, n_picks: int, miss_days: int,
                 gated: list | None = None) -> None:
    """scan 跑过即留痕（区分「今日无候选」与「模块没跑」；section_lines 依据）。

    gated：被换手门槛挡下的候选摘要列表，供主报告区块如实展示「谁被挡、为什么」。
    """
    mark = {"date": trade_date.isoformat(), "n_picks": n_picks,
            "miss_days": miss_days, "gated": gated or []}
    fstore.write_text(C.LAST_SCAN_JSON, json.dumps(mark, ensure_ascii=False))


def scanned_on(trade_date: dt.date) -> dict | None:
    """该日是否跑过 scan（last_scan 只存最新一次，日期不匹配即视为该日未跑）。"""
    try:
        mark = json.loads(Path(C.LAST_SCAN_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):                  # 无痕 / 读坏按未跑处理
        return None
    if not isinstance(mark, dict):
        return None
    return mark if mark.get("date") == trade_date.isoformat() else None


def _next_pct_from_bars(code: str, base_iso: str, today: dt.date):
    """复盘日后首根日K → (核验日iso, 次日收盘涨幅%)；(None, None)=数据未就绪。"""
    from data_source import fetch_daily
    bars = fetch_daily(code, today)
    if bars is None or not len(bars):
        return None, None                           # 拉不到，保持 pending
    dates = [str(x)[:10] for x in bars["date"]]
    j = next((i for i, d in enumerate(dates) if d > base_iso), None)
    if j is None or j < 1:
        return None, None                           # T+1 日K未就绪（停牌/滞后）
    close = bars["close"].astype(float)
    return dates[j], round((close.iloc[j] / close.iloc[j - 1] - 1.0) * 100.0, 2)


def verify(today: dt.date | None = None) -> str:
    """核验所有 pending 且 T+1 已到的登记行（纯记录，不预测）。"""
    today = today or dt.date.today()
    ledger = _load()
    changed = 0
    for r in ledger:
        if r["status"] != "pending":
            continue
        base = dt.date.fromisoformat(r["date"])
        if base >= today:
            continue                                # T+1 未到
        vdate, pct = _next_pct_from_bars(r["code"], r["date"], today)
        if pct is None:
            continue                                # 数据未就绪，保持 pending
        if pct >= _PROMOTE_PCT:
            promoted = "Y"
        else:
            # 日K近似口径之外，用次日涨停池复核（在池=封板=晋级）
            raw = pool_raw("zt", vdate.replace("-", ""))
            if raw is None:
                # 拉取失败 ≠ 不在池：写成 "N" 是永久标签，保持 pending 明日重核
                continue
            pool = parse_pool("zt", raw)
            promoted = "Y" if any(x["code"] == r["code"] for x in pool) else "N"
        r.update({"status": "verified", "verify_date": vdate,
                  "promoted": promoted, "next_pct": pct})
        changed += 1
    if changed:
        _save(ledger)
    return f"核验回填 {changed} 行"


def summary() -> dict:
    """累计晋级率（观察口径：先标签后计分，不做判读结论）。

    by_model_version：模型 vs 启发式的持续实证——model_version 空串=启发式
    排序时代的登记行（模型主排上线前/模型不可用日），单列对照。
    """
    rows = [r for r in _load() if r["status"] == "verified"]
    if not rows:
        return {"n": 0}
    n_y = sum(1 for r in rows if r["promoted"] == "Y")
    pcts = [float(r["next_pct"]) for r in rows
            if r["next_pct"] not in ("", None)]
    out = {"n": len(rows), "n_days": len({r["date"] for r in rows}),
           "promoted": n_y,
           "promote_rate": round(100.0 * n_y / len(rows), 1),
           "next_pct_mean": round(sum(pcts) / len(pcts), 2) if pcts else None}
    by_ver: dict[str, list] = {}
    for r in rows:
        by_ver.setdefault(r.get("model_version") or "", []).append(r)
    out["by_model_version"] = {
        v or "启发式": {"n": len(g),
                        "promoted": sum(1 for r in g if r["promoted"] == "Y"),
                        "promote_rate": round(
                            100.0 * sum(1 for r in g if r["promoted"] == "Y")
                            / len(g), 1)}
        for v, g in sorted(by_ver.items())}
    return out
