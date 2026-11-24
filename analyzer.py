# -*- coding: utf-8 -*-
"""分析编排：候选 → 池富集 → 日K获取 → 概念挂载 → 五维评分（并发）。

单股流程 _analyze_one 四步纯职责（init/enrich/bars/concepts），
编排 analyze_all 负责并发、情绪环境注入与运行内缓存回写。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

import numpy as np

from bars import acquire_bars, approx_lianban
from data_source import fetch_board_closes, fetch_stock_concepts
from second_board import detect_second_board

_BOARD_KLINE_LOCK = threading.Lock()   # _board_kline check-then-fetch 防重拉


def _init_result(cand: dict) -> dict:
    result = {**cand}
    result.update({
        "ok": False, "skip_reason": "",
        "lianban": 0, "zhaban_times": 0, "industry": "", "fbt": "",
        "board_pass": False, "board_score": 0, "board_note": "",
        "fund": float("nan"), "seal_ratio": float("nan"),
        "ltsz": float("nan"), "fbt_y": "", "seal_ratio_y": float("nan"),
        "concepts": "",
    })
    return result


def _pool_enrich(result: dict, code: str, extras: dict, context: dict | None) -> dict:
    """把涨停池/昨日池的附加字段写进 result（连板/炸板/行业/封单/流通盘/昨日口径）。"""
    ex = extras.get(code, {})
    result["lianban"] = ex.get("lianban", 0)
    result["zhaban_times"] = ex.get("zhaban_times", 0)
    result["industry"] = ex.get("industry", "")
    result["fbt"] = ex.get("fbt", "")
    v = ex.get("ltsz")
    if v is not None:
        result["ltsz"] = float(v)
    if context is not None:
        result["fbt_y"] = str((context.get("yesterday_fbt") or {}).get(code, "") or "")
        sy = float((context.get("yesterday_seal") or {}).get(code, 0.0) or 0.0)
        if sy > 0:
            result["seal_ratio_y"] = sy
    fund_v = float(ex.get("fund") or 0.0)
    amount_v = float(result.get("amount") or 0.0)
    if fund_v > 0:
        result["fund"] = fund_v
        if amount_v > 0:
            result["seal_ratio"] = round(fund_v / amount_v, 2)
    return ex


def _attach_concepts(code: str, result: dict, ex: dict, context: dict | None) -> None:
    """概念题材挂载（仅 2 板候选）：个股概念 + 最热概念板块K线（发酵度原料）。

    结果缓存在 ctx（_stock_concepts/_board_kline）并随滚动缓存持久化。
    """
    boards = (context or {}).get("concept_boards") or {}
    if result["lianban"] != 2 or not boards:
        return
    cmap = context.setdefault("_stock_concepts", {})
    if code in cmap:
        concepts = cmap[code]
    else:
        concepts = fetch_stock_concepts(code, set(boards))
        cmap[code] = concepts
    if not concepts:
        return
    ex["concepts"] = concepts
    result["concepts"] = "、".join(concepts)
    best = max((n for n in concepts if n in boards),
               key=lambda n: float(boards[n].get("pct") or -99))
    ex["best_concept"] = best     # 供 second_board A 维直接消费（免重复求最热）
    bk = str(boards[best].get("bk") or "")
    if bk:
        # 双检锁：多只 2 板股同板块时并发 miss 会各拉一次（check-then-fetch）
        kmap = context.setdefault("_board_kline", {})
        with _BOARD_KLINE_LOCK:
            if bk not in kmap:
                kmap[bk] = fetch_board_closes(bk)


def _analyze_one(cand: dict, trade_date, extras: dict[str, dict],
                 context: dict | None = None,
                 bars_cache: dict | None = None) -> dict:
    code = cand["code"]
    result = _init_result(cand)
    ex = _pool_enrich(result, code, extras, context)

    if cand.get("zt_status") != "封板":
        # 非收盘封板股不可能成为二板候选：不拉日K，直接返回（省网络）
        result["ok"] = True
        return result

    bars, skip = acquire_bars(cand, code, trade_date, bars_cache)
    if skip:
        result["skip_reason"] = skip
        return result

    # 量比 = 今日成交量 / 前5日均量（明日关注清单的核心筛选因子）
    vol_arr = bars["volume"].astype(float).to_numpy() if "volume" in bars.columns else None
    if vol_arr is not None and len(vol_arr) >= 6:
        base = np.nanmean(vol_arr[-6:-1])
        if base > 0:
            result["vol_ratio"] = round(float(vol_arr[-1]) / base, 2)

    result["lianban"] = int(result["lianban"] or approx_lianban(bars, cand.get("name", "")))
    ex = dict(ex)
    if not ex.get("lianban"):
        ex["lianban"] = result["lianban"]

    _attach_concepts(code, result, ex, context)
    result.update(detect_second_board(cand, bars, ex, context))
    result["ok"] = True
    return result


def analyze_all(cands: list[dict], trade_date, extras: dict[str, dict],
                workers: int = 8, progress=print,
                context: dict | None = None,
                bars_cache: dict | None = None) -> tuple[list[dict], list[dict]]:
    """并发分析候选（仅封板股需要日K），返回 (结果列表, 跳过列表)。

    bars_cache: {code: 日K DataFrame}，命中直接复用（不碰网络），
    新拉取的 bars 会写回该 dict。
    """
    # 市场情绪环境注入（二板评分用炸板率/赚钱效应；用全体候选口径算炸板率）
    ctx = dict(context or {})
    if ctx.get("break_rate") is None:
        up = [c for c in cands if "涨停" in c["pools"]]
        sealed = sum(1 for c in up if c["zt_status"] == "封板")
        if up:
            ctx["break_rate"] = (len(up) - sealed) / len(up)
    if ctx.get("yzt_ret") is None and ctx.get("yzt_today_pct"):
        ctx["yzt_ret"] = sum(ctx["yzt_today_pct"]) / len(ctx["yzt_today_pct"])
    results: list[dict] = []
    skipped: list[dict] = []
    total = len(cands)
    done = 0
    sealed_n = sum(1 for c in cands if c.get("zt_status") == "封板")
    progress(f"分析 {total} 只候选（其中封板 {sealed_n} 只需日K，其余零网络），"
             f"并发 {workers}...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_analyze_one, c, trade_date, extras, ctx, bars_cache): c
                for c in cands}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r["ok"]:
                results.append(r)
            else:
                skipped.append(r)
            if done % 50 == 0 or done == total:
                progress(f"  进度 {done}/{total}")
    # 概念/板块K线的运行内缓存回写原 ctx，随滚动缓存持久化（跨日复用，概念名单基本稳定）
    if isinstance(context, dict):
        for k in ("_stock_concepts", "_board_kline"):
            if k in ctx:
                context[k] = ctx[k]
    results.sort(key=lambda x: (-x.get("board_score", 0), -x.get("amp", 0)))
    return results, skipped
