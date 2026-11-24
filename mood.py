# -*- coding: utf-8 -*-
"""情绪周期引擎：多日序列判定短线情绪所处阶段（自主创新模块）。

调研口径（2026-08 综合：知乎/雪球/闽发/腾讯云情绪周期框架）：
  连板高度=空间尺子、晋级率=接力意愿、炸板率=分歧强度、赚钱效应=溢价扩散，
  四者结合定位周期阶段；**退潮期技术分析普遍失效、防御优先**。

阶段与量化特征（判定顺序即优先级）：
  退潮：高度较近端高点断板 ≥2 级（高点≥4），或亏钱效应 <-2%
  冰点：高度 ≤3 且晋级率 <15%
  高潮：高度创序列新高且 ≥5，炸板率 ≤35%（一致性最强）
  发酵：高度抬升（高于前几日均值）且晋级率 ≥25%、赚钱效应非负
  回暖：赚钱效应转正但高度/晋级未确认抬升
  分歧：其余情形（高度滞涨、信号矛盾）

阶段 → D 维「周期槽」得分（0~3）与仓位建议。
"""
from __future__ import annotations

import datetime as dt

import data_source
from config import MAIN_BOARD_PREFIXES

STAGE_D_SCORE = {"发酵": 3, "高潮": 3, "回暖": 2, "分歧": 1, "冰点": 0,
                 "退潮": 0, "未知": 1}
STAGE_ADVICE = {"发酵": "正常仓位", "高潮": "正常·警惕一致性过热",
                "回暖": "试探仓位", "分歧": "轻仓",
                "冰点": "观望等启动", "退潮": "防御优先(空仓)", "未知": "—"}
# 量化仓位（0~1，占总资金）：报告展示用；排板模式下单票仓位 = 阶段仓位 / 同时持有上限
STAGE_POSITION = {"高潮": 0.6, "发酵": 0.5, "回暖": 0.3, "分歧": 0.2,
                  "冰点": 0.0, "退潮": 0.0, "未知": 0.3}


def classify_stage(hist: list[dict]) -> tuple[str, str]:
    """按多日指标序列判定当前阶段。hist 最旧→最新，每项含
    height/promo/break_rate/money（promo、break_rate、money 可为 None）。

    返回 (阶段, 判定依据描述)。纯函数，可离线单测。

    ⚠ 没有 height 键的交易日 = 池取数失败，**整天剔除**、不参与判定——缺失 ≠
    0 高度。此前缺失日的空指标被 `or 0` 读成「高度 0」，一次池接口失败就足以
    触发「高度 N→0 断板 → 退潮期 → 空仓」（2026-11-24 修复）。
    """
    full = [x for x in hist if x.get("height") is not None]
    if len(full) < 2:
        return "未知", ("序列不足" if len(full) == len(hist) else "序列不足（池数据缺失）")
    hist = full
    h = [x["height"] for x in hist]
    p_last = hist[-1].get("promo")
    b_last = hist[-1].get("break_rate")
    m_last = hist[-1].get("money")

    H, Hmax_prev = h[-1], max(h[:-1])
    fmt = lambda v: f"{v:.0%}" if isinstance(v, float) else "?"

    # 退潮：高度断板（近端高点≥4，回落≥2级）或明确亏钱效应
    if Hmax_prev >= 4 and Hmax_prev - H >= 2:
        return "退潮", f"高度{Hmax_prev}→{H}断板"
    if m_last is not None and m_last < -2.0:
        return "退潮", f"亏钱效应{m_last:+.1f}%"
    # 冰点：高度压缩且接力意愿极低
    if H <= 3 and (p_last is None or p_last < 0.15):
        return "冰点", f"高度{H}·晋级{fmt(p_last) if p_last is not None else '缺'}"
    # 高潮：高度创阶段新高（≥5）且分歧弱（炸板≤35%）
    if H >= max(h) and H >= 5 and (b_last is None or b_last <= 0.35):
        return "高潮", f"高度{H}创新高·炸板{fmt(b_last)}"
    # 发酵：高度抬升 + 接力意愿强 + 溢价非负
    prev_mean = sum(h[:-1]) / len(h[:-1])
    if (H > prev_mean and p_last is not None and p_last >= 0.25
            and (m_last is None or m_last >= 0)):
        return "发酵", f"高度{H}抬升(均{prev_mean:.1f})·晋级{fmt(p_last)}"
    # 回暖：赚钱效应修复但结构未确认
    if m_last is not None and m_last > 0:
        return "回暖", f"赚钱效应{m_last:+.1f}%·高度{H}未确认"
    return "分歧", f"高度{H}滞涨·信号矛盾"


def day_metrics(zt_rows, prev_zt_rows, zb_rows) -> dict:
    """单日情绪指标：高度/主板封板数/晋级率/炸板率（纯数据变换）。"""
    mb = lambda r: r["code"].startswith(MAIN_BOARD_PREFIXES)
    height = max((r["lianban"] for r in zt_rows if mb(r)), default=0)
    sealed_mb = sum(1 for r in zt_rows if mb(r))
    two_plus = sum(1 for r in zt_rows if mb(r) and r["lianban"] >= 2)
    broken_mb = sum(1 for r in zb_rows if mb(r))
    prev_mb = sum(1 for r in prev_zt_rows or [] if mb(r))
    return {
        "height": height,
        "zt_count": sealed_mb,
        "promo": (two_plus / prev_mb) if prev_mb else None,
        "break_rate": (broken_mb / (sealed_mb + broken_mb))
                      if (sealed_mb + broken_mb) else None,
    }


def build_series(trade_date: dt.date, calendar, lookback: int = 3) -> list[dict]:
    """取 [T-lookback, T] 各交易日指标（旧→新）。今日 money 由上层补入。

    代价：每回看日 1 次涨停池 + 1 次炸板池 + 前日涨停池（推算晋级率）。
    首个回看日的「昨日」= 窗口前一交易日（cal[-lookback-1]）；日历不足以
    再前推时该日晋级率置 None——此前误取 cal[-1]（未来日），序列首日 promo
    系统性错位并污染归档 cycle_series.csv（2026-11-20 修复）。
    """
    cal = [d for d in (calendar or []) if d < trade_date]
    tdays = cal[-lookback:] if lookback else []
    dates = tdays + [trade_date]
    series = []
    zt_cache: dict = {}
    for i, d in enumerate(dates):
        raw_zt = data_source.pool_raw("zt", f"{d:%Y%m%d}")
        zt = data_source.parse_pool("zt", raw_zt)
        zb = data_source.parse_pool("zb", data_source.pool_raw("zb", f"{d:%Y%m%d}"))
        prev_d = (dates[i - 1] if i > 0
                  else (cal[-lookback - 1] if len(cal) > lookback else None))
        if prev_d is None:
            prev_zt = []
        elif prev_d in zt_cache:
            prev_zt = zt_cache[prev_d]
        else:
            prev_zt = data_source.parse_pool(
                "zt", data_source.pool_raw("zt", f"{prev_d:%Y%m%d}"))
            zt_cache[prev_d] = prev_zt
        zt_cache[d] = zt
        # raw_zt is None = 取数失败（≠ 取数成功的空池 []）：该日不给任何指标。
        m = day_metrics(zt, prev_zt, zb) if raw_zt is not None else {}
        m["date"] = f"{d:%Y-%m-%d}"
        series.append(m)
    return series


def stage_for_today(trade_date: dt.date, calendar, ctx: dict) -> tuple[str, str, list]:
    """整合：构建序列 → 补入今日赚钱效应 → 判定阶段。"""
    series = build_series(trade_date, calendar)
    rets = ctx.get("yzt_today_pct") or []
    if series and rets:
        series[-1]["money"] = sum(rets) / len(rets)
    elif series:
        series[-1]["money"] = ctx.get("yzt_ret")
    if series and series[-1].get("height") is None:
        # 今日涨停池取数失败：如实报「未知」，不拿前几日的阶段冒充今日，
        # 更不把缺失当 0 高度判成退潮/冰点
        return "未知", "今日涨停池取数失败（不以缺失当 0 高度）", series
    stage, why = classify_stage(series)
    return stage, why, series
