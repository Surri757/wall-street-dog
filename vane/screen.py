# -*- coding: utf-8 -*-
"""编排层：拉窗口涨停池 → engine 回溯 → 附封板质量/梯队上下文 → 产出名单。

数据通道全走 data_source 门面现成导出（pool_raw/parse_pool/trade_calendar），
每天 1 次请求、6 秒快速失败、进程内 memo；窗口内某日拉取失败按「该日无板」
处理（板数保守低估，miss_days 落盘注明），复盘日当日涨停池拉失败则整体放弃；
交易日历不可用同样整体放弃（历史池对非交易日回声最后交易日，不能拿池试探
代替日历）。

脏数据单点收敛：池字段的 NaN（东财 null/缺位）在本层统一清成 None，
下游 engine/report/ledger 只判 None，不再各自散写 v==v 判断。
"""
from __future__ import annotations

import datetime as dt

from data_source import parse_pool, pool_raw, trade_calendar

from . import config as C
from .engine import label, select, streak


def _is_main_board(code: str) -> bool:
    # 不复用 candidates.is_main_board：旁路子包自持惯例（同 ziyang），
    # 前缀口径经 vane.config 转发自主 config 单一事实源
    return code.startswith(C.MAIN_BOARD_PREFIXES)


def _num_or_none(v) -> float | None:
    """池数值字段 NaN → None（缺失的统一表达；正常值原样透传）。"""
    return v if v is not None and v == v else None


def _ymd(d: dt.date) -> str:
    return f"{d:%Y%m%d}"


def _window_days(trade_date: dt.date, window: int) -> tuple[list[dt.date], bool]:
    """窗口交易日（旧→新，含 trade_date）。只用交易日历；日历不可用或未覆盖
    复盘日返回 ([], False)，由 build_rows 判为本次放弃。

    不做「非空池=交易日」的池试探降级：历史池对非交易日会回声最后一个交易日
    的数据（backfill.py 实测 8/29 周六与 8/28 池完全一致）——回声日会让周五
    的板被数三次，B/span 虚高，而结果会写进不可改写的预注册台账、并进入
    训练样本。日历坏了就放弃当天，明日重跑。
    """
    cal = trade_calendar()
    past = [d for d in cal if d <= trade_date][-window:]
    if past and past[-1] == trade_date:
        return past, True
    return [], False


def scan(trade_date: dt.date, window: int = C.WINDOW,
         order_key=None) -> dict:
    """跑一次完整筛选。返回 dict（picks/bench/ladder/miss_days/...），不写任何盘。

    order_key 缺省=启发式排序；模型主排时由 daily_sync 传入 model_rank_key
    （先 apply_ranking 出分，再以模型键截断名额）。
    """
    built = build_rows(trade_date, window)
    if built.get("error"):
        return built
    picks, bench = select(built["rows"], top_n=C.TOP_N,
                          board_range=C.BOARD_RANGE, key=order_key)
    for i, r in enumerate(picks, 1):
        r["rank"] = i
        r["group"] = f"{r['B']}to{r['B'] + 1}"
    built["picks"], built["bench"] = picks, bench
    return built


def build_rows(trade_date: dt.date, window: int = C.WINDOW) -> dict:
    """纯数据层：拉窗口池 → 今日候选行（不选名额）。scan 的分解步骤，
    供模型评分链复用（daily_sync：build_rows → apply_ranking → select）。"""
    past, cal_ok = _window_days(trade_date, window)
    if not cal_ok or len(past) < 2:
        return {"error": f"{trade_date} 无法确定窗口交易日（非交易日，或交易日历"
                         "不可用/未覆盖该日）——本次筛选放弃，不登记"}
    cal_str = [f"{d:%Y-%m-%d}" for d in past]

    # 1) 拉窗口内每日涨停池（收盘封板口径），聚合出每票涨停日集合
    zt_by_day: dict[str, dict[str, dict]] = {}       # iso日 → {code: 归一行}
    miss_days: list[str] = []
    for d, iso in zip(past, cal_str):
        rows = parse_pool("zt", pool_raw("zt", _ymd(d)))
        if not rows:
            miss_days.append(iso)                    # 拉失败或空池 → 按该日无板
            continue
        zt_by_day[iso] = {r["code"]: r for r in rows}
    today_iso = cal_str[-1]
    today_pool = zt_by_day.get(today_iso)
    if today_pool is None:
        return {"error": f"{today_iso} 涨停池拉取失败，本次筛选放弃（明日重跑）"}

    zt_dates_of: dict[str, list[str]] = {}           # code → 窗口内涨停日
    for iso, pool in zt_by_day.items():
        for code in pool:
            zt_dates_of.setdefault(code, []).append(iso)

    # 2) 今日封板的主板非 ST 票逐只回溯本轮板数
    rows: list[dict] = []
    ladder: dict[int, int] = {}                      # 本轮板数 B ≥ 2 → 家数（主板）
    for code, prow in today_pool.items():
        if not _is_main_board(code):
            continue
        if C.EXCLUDE_ST and "ST" in prow["name"].upper():
            continue
        st = streak(zt_dates_of.get(code, []), today_iso, cal_str, C.MAX_GAP)
        if st is None:
            continue
        amount = prow["amount"]
        fund = prow["fund"]
        rows.append({
            "code": code, "name": prow["name"],
            "B": st["B"], "span": st["span"], "breaks": st["breaks"],
            "round_dates": st["round_dates"], "label": label(st),
            "close": prow["close"], "pct": prow["pct"],
            "fbt": prow["fbt"], "industry": prow["industry"],
            "ltsz": _num_or_none(prow["ltsz"]),
            "seal_ratio": (round(fund / amount, 2)
                           if fund == fund and amount > 0 else None),
        })
        if st["B"] >= 2:
            ladder[st["B"]] = ladder.get(st["B"], 0) + 1

    return {
        "trade_date": today_iso,
        "window": len(cal_str),
        "miss_days": miss_days,
        "rows": rows,                      # 候选行（不选名额——名额在 scan/模型链）
        "ladder": dict(sorted(ladder.items(), reverse=True)),
        "n_today_mb_zt": len(rows),        # 今日主板封板数（非ST）
    }
