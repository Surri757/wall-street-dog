# -*- coding: utf-8 -*-
"""二板定龙头（二进三候选）检测 v2 —— 五维联合评分。

调研口径（2026-08：雪球连板统计 / 财联社情绪复盘 / 打板量化研究）：
  基础比率：一进二≈15.7%、二进三≈34.7%；评分≥60 应显著高于 35%。

  A 题材地位（25）：板块涨停家数10 + 板内身位8 + 题材发酵度7
     发酵度 = 同行业今日涨停家数 vs 昨日（升高=发酵中，降低=退潮）
  B 首板质量（25）：昨日封板时间8 + 昨日实体板6 + 昨日换手5 + 昨日封成比6
     （全部真实昨日口径：昨日涨停池重建；封成比≥2倍=首板封单强）
  C 二板强度（25）：今日高开区间10 + 今日换手8 + 炸板次数7
  D 情绪环境（15）：主板炸板率5 + 晋级率5 + 昨涨停今表现（赚钱效应）5
  E 辨识度（10）：流通市值6（<30亿最优）+ 股价4（低价股人气足）

硬条件：主板收盘封板 + 2连板（外部涨停池连板数；缺失时日K近似）。
数据缺失：对应项给中性分并计 miss，最终每项 -BOARD_MISS_PENALTY 分
（防中性分叠满混过门槛），board_note 标注缺失项数。
无未来函数。
"""
from __future__ import annotations

import pandas as pd

from config import (BOARD_FBT_EARLY, BOARD_FBT_MID,
                    BOARD_LTSZ_MID, BOARD_LTSZ_SMALL, BOARD_MISS_PENALTY,
                    BOARD_PRICE_LOW, BOARD_PRICE_VLOW, BOARD_SCORE_MIN,
                    BOARD_TURN_BEST, LIMIT_UP_APPROX, LIMIT_UP_APPROX_ST)
from board_rules import gap_score
from mood import STAGE_D_SCORE


def _base() -> dict:
    return {"board_pass": False, "board_score": 0, "board_note": ""}


def _is_limit_close(pct: float, name: str) -> bool:
    return pct >= LIMIT_UP_APPROX or ("ST" in str(name).upper() and pct >= LIMIT_UP_APPROX_ST)


def detect_second_board(cand: dict, bars: pd.DataFrame,
                        extras: dict, ctx: dict | None) -> dict:
    ctx = ctx or {}
    name = str(cand.get("name", ""))
    code = str(cand.get("code", ""))
    # ---- 硬条件：收盘封板 + 2连板 ----
    if cand.get("zt_status") != "封板":
        return _base()
    lianban = int(extras.get("lianban", 0) or 0)
    if lianban > 0:
        is_two_board = (lianban == 2)  # 外部涨停池连板数可信时以它为准
    else:
        is_two_board = False
        if bars is not None and len(bars) >= 3:
            close = bars["close"].astype(float).to_numpy()
            pct_today = float(close[-1] / close[-2] - 1.0)
            pct_yesterday = float(close[-2] / close[-3] - 1.0)
            if _is_limit_close(pct_today, name) and _is_limit_close(pct_yesterday, name):
                is_two_board = True  # 池数据缺失时用日K近似
    if not is_two_board:
        return _base()

    score = 0
    dims: dict[str, list[str]] = {}   # 维度 → 明细标签（分组展示）

    def dim(key: str, label: str) -> None:
        dims.setdefault(key, [label])

    def add(key: str, s: int, note: str) -> None:
        nonlocal score
        score += s
        if note:
            dims[key].append(note)

    miss = 0   # 数据缺失项数：中性分按「无信息」给，但整票要打折

    # ---- A. 题材地位 25：家数10 + 身位8 + 发酵7（概念口径优先，行业兜底）----
    dim("A", "题材")
    industry = extras.get("industry", "") or ""
    concepts = extras.get("concepts") or []
    boards = ctx.get("concept_boards") or {}
    mine = [boards[n] for n in concepts if n in boards]
    if mine:
        # 概念口径：取该股所属概念中最热的一个（analyzer 富集时已算好则直接取，
        # 免对同一股同日重复求 max；缺失时回退自算——second_board 须可独立运行）
        best_name = extras.get("best_concept") or max(
            (n for n in concepts if n in boards),
            key=lambda n: float(boards[n].get("pct") or -99))
        best = boards[best_name]
        pct = float(best.get("pct") or 0.0)
        up, dn = int(best.get("up") or 0), int(best.get("down") or 0)
        breadth = up / (up + dn) if (up + dn) > 0 else 0.0
        short = best_name[:6]
        if pct >= 3.0 and breadth >= 0.7:
            add("A", 10, f"{short}{pct:+.1f}%热")
        elif pct >= 1.5:
            add("A", 7, f"{short}{pct:+.1f}%")
        elif pct > 0:
            add("A", 4, f"{short}{pct:+.1f}%弱")
        else:
            add("A", 2, f"{short}绿")
        # 身位：该股是否为任一所属概念的领涨股（题材辨识度核心）
        if any(str(b.get("lead") or "") == name for b in mine):
            add("A", 8, "概念领涨")
        else:
            add("A", 4, "")
        # 发酵：最热概念今日涨幅 vs 前几日均（板块K线，增强项，无数据不给 miss）
        kline = (ctx.get("_board_kline") or {}).get(str(best.get("bk") or ""))
        rets = [kline[i] / kline[i - 1] - 1.0
                for i in range(1, len(kline))] if kline else []
        if len(rets) >= 2:
            prior = rets[:-1][-4:]
            base_r = sum(prior) / len(prior) if prior else 0.0
            if rets[-1] > 0 and rets[-1] >= base_r * 1.5 + 0.002:
                add("A", 7, "加速")
            elif rets[-1] > 0:
                add("A", 4, "")
            else:
                add("A", 2, "退潮")
        else:
            add("A", 4, "")
    else:
        # 行业口径（概念接口不可用时的兜底，同 v2）
        ind_count = (ctx.get("industry_count") or {}).get(industry, 0) if industry else 0
        if industry in (ctx.get("persistent_industries") or ()):
            dims["A"].append("持续题材")   # 实证标签：连续两日≥2家，次日收+3.08% vs 非持续+1.55%（不计分待OOS）
        if ind_count >= 3:
            add("A", 10, f"{industry}{ind_count}家")
        elif ind_count == 2:
            add("A", 7, f"{industry}2家")
        elif ind_count == 1:
            add("A", 4, "独苗")
        else:
            add("A", 5, "数据缺失")
            miss += 1
        ind_max = (ctx.get("industry_max_lianban") or {}).get(industry, 0) if industry else 0
        if ind_max and lianban >= ind_max:
            add("A", 8, "最高板")
        elif ind_max:
            add("A", 4, "跟风板")
        else:
            add("A", 5, "")
            miss += 1
        ind_y = (ctx.get("industry_yesterday_count") or {})
        if industry and ind_y:
            y_cnt = ind_y.get(industry, 0)
            if ind_count > y_cnt:
                add("A", 7, "发酵" if y_cnt else "新题材")
            elif ind_count == y_cnt:
                add("A", 4, "")
            else:
                add("A", 2, "退潮")
        else:
            add("A", 4, "")
            miss += 1

    # ---- B. 首板质量 25（真实昨日口径）：封板时间8 + 实体6 + 换手5 + 封成比6 ----
    dim("B", "首板")
    fbt1 = str((ctx.get("yesterday_fbt") or {}).get(code, "") or "")
    if fbt1 and fbt1 <= BOARD_FBT_EARLY:
        add("B", 8, f"{fbt1}封板")
    elif fbt1 and fbt1 <= BOARD_FBT_MID:
        add("B", 6, f"{fbt1}封板")
    elif fbt1:
        add("B", 4, f"{fbt1}晚封")
    else:
        add("B", 4, "封板时间未知")
        miss += 1
    if bars is not None and len(bars) >= 3:
        open_arr = bars["open"].astype(float).to_numpy()
        close_arr = bars["close"].astype(float).to_numpy()
        gap_yesterday = float(open_arr[-2] / close_arr[-3] - 1.0)
        if gap_yesterday >= 0.05:  # 昨日一字/接近一字开盘：锁仓强（1414样本：晋级53% vs 23%）
            add("B", 6, "一字")
        else:
            add("B", 3, "实体")
    else:
        add("B", 4, "")
        miss += 1
    turn1 = float((ctx.get("yesterday_turnover") or {}).get(code, 0.0) or 0.0)
    if 0.05 <= turn1 <= 0.25:
        add("B", 5, "")
    elif turn1 > 0:
        add("B", 3, f"昨换{turn1:.0%}")
    else:
        add("B", 4, "")
        miss += 1
    seal_y = float((ctx.get("yesterday_seal") or {}).get(code, 0.0) or 0.0)
    if seal_y >= 2:
        add("B", 6, f"昨封{seal_y:.1f}")
    elif seal_y >= 0.5:
        add("B", 4, f"昨封{seal_y:.1f}")
    elif seal_y > 0:
        add("B", 2, f"昨封{seal_y:.1f}")
    else:
        add("B", 4, "")
        miss += 1

    # ---- C. 二板强度 25：高开10 + 换手8 + 炸板7 ----
    # 2026-08-30 回灌校准(106样本)：准一字开盘晋级率58.8%远超高开3-7%的18.2%，
    # 缩量换手(<5%)次日+4.39%优于充分换手的+1.37%——按数据反转了原调研口径分档。
    # 样本仅3周窗口，≥300样本后需复核（防单窗口过拟合）。
    dim("C", "二板")
    if bars is not None and len(bars) >= 2:
        open_arr = bars["open"].astype(float).to_numpy()
        close_arr = bars["close"].astype(float).to_numpy()
        gap_today = float(open_arr[-1] / close_arr[-2] - 1.0)
        pts, lab = gap_score(gap_today)
        add("C", pts, lab)
    else:
        add("C", 5, "")
        miss += 1
    turnover_t = float(cand.get("turnover", 0.0) or 0.0)
    lo, hi = BOARD_TURN_BEST
    if 0 < turnover_t < lo:
        add("C", 8, "缩量板")              # 数据：缩量二板次日+4.39%，强于充分换手
    elif lo <= turnover_t <= hi:
        add("C", 6, "")
    elif 0.20 < turnover_t <= 0.30:
        add("C", 5, f"换手{turnover_t:.0%}分歧")
    elif turnover_t > 0.30:
        add("C", 3, f"死亡换手{turnover_t:.0%}")
    else:
        add("C", 5, "")
        miss += 1
    zb = int(extras.get("zhaban_times", 0) or 0)
    if not ctx.get("zb_pool_ok"):
        add("C", 5, "炸板未知")
        miss += 1
    elif zb == 0:
        add("C", 7, "无炸板")
    elif zb == 1:
        add("C", 5, "炸1回封")
    else:
        add("C", 2, f"炸{zb}次")

    # ---- D. 情绪环境 15：炸板率4 + 晋级率4 + 赚钱效应4 + 情绪周期3 ----
    dim("D", "情绪")
    break_rate = float(ctx.get("break_rate", 0.40) or 0.40)
    if break_rate <= 0.25:
        add("D", 4, f"炸板{break_rate:.0%}")
    elif break_rate <= 0.40:
        add("D", 3, f"炸板{break_rate:.0%}")
    else:
        add("D", 2, f"退潮{break_rate:.0%}")
    promo = ctx.get("promotion_rate")
    if promo is not None:
        if promo >= 0.40:
            add("D", 4, f"晋级{promo:.0%}")
        elif promo >= 0.20:
            add("D", 3, f"晋级{promo:.0%}")
        else:
            add("D", 2, f"晋级{promo:.0%}低")
    else:
        add("D", 2, "晋级未知")
        miss += 1
    yzt_ret = ctx.get("yzt_ret")
    if yzt_ret is not None:
        if yzt_ret >= 2.0:
            add("D", 4, f"昨涨停{yzt_ret:+.1f}%")
        elif yzt_ret >= 0:
            add("D", 3, f"昨涨停{yzt_ret:+.1f}%")
        elif yzt_ret >= -2.0:
            add("D", 2, f"昨涨停{yzt_ret:+.1f}%")
        else:
            add("D", 1, f"亏钱{yzt_ret:+.1f}%")
    else:
        add("D", 2, "赚钱效应未知")
        miss += 1
    # 情绪周期阶段（mood.py 判定；缺序列时按未知中性 + miss）
    stage = str(ctx.get("cycle_stage") or "未知")
    stage_s = STAGE_D_SCORE.get(stage, 1)
    if stage != "未知":
        add("D", stage_s, f"{stage}期")
    else:
        add("D", stage_s, "")
        miss += 1

    # ---- E. 辨识度 10：流通市值6 + 低价4 ----
    dim("E", "辨识")
    ltsz = float(extras.get("ltsz") or 0.0)
    if ltsz > 0:
        if ltsz < BOARD_LTSZ_SMALL:
            add("E", 6, f"流通{ltsz/1e8:.0f}亿小盘")
        elif ltsz < BOARD_LTSZ_MID:
            add("E", 4, f"流通{ltsz/1e8:.0f}亿")
        else:
            add("E", 2, f"流通{ltsz/1e8:.0f}亿")
    else:
        add("E", 3, "")
        miss += 1
    try:
        px = float(cand.get("close", 0) or 0)
    except (TypeError, ValueError):
        px = 0.0
    if px > 0:
        if px < BOARD_PRICE_VLOW:
            add("E", 4, f"{px:.2f}元深低价")
        elif px < BOARD_PRICE_LOW:
            add("E", 3, "低价")
        else:
            add("E", 2, "")
    else:
        add("E", 3, "")
        miss += 1

    # 数据缺失惩罚：每缺一项 -BOARD_MISS_PENALTY 分
    if miss:
        score = max(0, score - miss * BOARD_MISS_PENALTY)
        dims.setdefault("X", ["数据"]).append(f"缺失{miss}项(-{miss * BOARD_MISS_PENALTY})")

    note = " | ".join(
        f"{labels[0]}:{'·'.join(labels[1:])}" for labels in dims.values() if len(labels) > 1)
    return {
        "board_pass": bool(score >= BOARD_SCORE_MIN),
        "board_score": score,
        "board_note": note,
    }
