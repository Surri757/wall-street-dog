# -*- coding: utf-8 -*-
"""维度区分度分析：归档行 → 布尔特征 → 按日聚类的组间对照 + BH-FDR。"""
from __future__ import annotations

import datetime as dt
import re

from calib.forward import MAIN_METRIC, MIN_INFER_DAYS
from calib.stats import bh_fdr, cluster_diff, two_sided_p
from termstyle import weekday_cn


def _note_val(note: str, pattern: str):
    m = re.search(pattern, note or "")
    return m.group(1) if m else None


def build_features(r: dict) -> dict[str, bool]:
    """从归档行提取维度布尔特征（数值列直取 + board_note 正则）。"""
    note = r.get("board_note") or ""
    f: dict[str, bool] = {}
    try:
        ltsz = float(r.get("ltsz") or 0)
        f["小盘<30亿"] = 0 < ltsz < 3e9
        f["中大盘≥30亿"] = ltsz >= 3e9
    except (TypeError, ValueError):
        pass
    try:
        px = float(r.get("close") or 0)
        f["低价<10元"] = 0 < px < 10
        f["≥10元"] = px >= 10
    except (TypeError, ValueError):
        pass
    fbt_y = _note_val(note, r"首板:(\d{4})")
    if fbt_y:
        f["首板早封≤1000"] = fbt_y <= "1000"
        f["首板晚封>1130"] = fbt_y > "1130"
    seal_y = _note_val(note, r"昨封([0-9.]+)")
    if seal_y is not None:
        v = float(seal_y)
        f["昨封强≥2"] = v >= 2
        f["昨封弱<0.5"] = 0 < v < 0.5
    f["高开3-7%优"] = "优" in note and "高开" in note
    f["准一字开盘"] = "准一字" in note
    f["高开弱"] = "高开" in note and "弱" in note
    try:
        turn = float(r.get("turnover") or 0)
        f["换手5-20%"] = 0.05 <= turn <= 0.20
        f["换手<5%"] = 0 < turn < 0.05
    except (TypeError, ValueError):
        pass
    try:
        zb = int(float(r.get("zhaban_times") or 0))
        f["无炸板"] = zb == 0
        f["有炸板"] = zb > 0
    except (TypeError, ValueError):
        pass
    f["持续题材"] = "持续题材" in note
    # 首板形态（首板段的实体/一字，注意与二板段的"准一字"区分）
    seg = _note_val(note, r"首板:([^|]*)")
    if seg is not None:
        f["首板实体"] = "实体" in seg
        f["首板一字"] = "一字" in seg
    # 二板日封成比分档（实证：≥0.5 次日+6.5%/晋级~66%；与准一字/缩量高度共线≈同一
    # 「筹码锁定」信号，故不重复计分，仅在此追踪独立增量的区分度）
    try:
        sr = float(r.get("seal_ratio") or 0)
        if sr > 0:
            f["今封≥0.5"] = sr >= 0.5
    except (TypeError, ValueError):
        pass
    f["题材发酵"] = ("发酵" in note) or ("新题材" in note) or ("加速" in note)
    f["题材退潮"] = "退潮" in note
    # 情绪周期阶段（优先取归档列，旧样本回退 note 正则；六阶段互斥布尔，保证对照组非空）
    stage = str(r.get("cycle_stage") or "").strip()
    if not stage:
        m_note = _note_val(note, r"(高潮|发酵|回暖|分歧|冰点|退潮)期")
        stage = m_note or ""
    if stage:
        for st in ("高潮", "发酵", "回暖", "分歧", "冰点", "退潮"):
            f[f"周期·{st}"] = (st == stage)
    m = _note_val(note, r"缺失(\d+)项")
    if m:
        f["数据缺失≥3"] = int(m) >= 3
    # 星期效应（实证：周中后段强于周一，窗口内混杂周期阶段，仅观察不计分）
    wd = str(r.get("weekday") or "").strip()
    if not wd and r.get("date"):
        try:
            wd = weekday_cn(dt.datetime.strptime(str(r["date"]), "%Y-%m-%d"))
        except ValueError:
            wd = ""
    if wd:
        for w in ("周一", "周二", "周三", "周四", "周五"):
            f[w] = (wd == w)
    return {k: v for k, v in f.items() if v is not None}


COLLINEAR_GROUPS = [
    # build_features 注释里承认：今封≥0.5 与准一字/缩量高度共线，
    # ≈同一个「筹码锁定」信号。共线特征不是独立检验，合并计一个 m，否则会高估
    # 检验族规模、把 BH 校正做得过度保守。
    {"准一字开盘", "今封≥0.5", "换手<5%"},
]


def analyze_dims(rows: list[dict], min_n: int = 5) -> str:
    """维度级区分度：每个特征「成立组 vs 不成立组」在**可成交口径**下的对比 + BH-FDR。

    此前两个缺陷：指标是 next_close_pct/promoted（不可成交，买不到的收益）；
    29 个特征零多重比较校正（@0.05 下 P(≥1 假阳性)=1−0.95^29≈77%）。
    渲染必须在收集完所有 t/p 之后做——旧实现在循环体内直接拼字符串，
    t/p 从未被收集，因此结构上就无法校正。
    """
    analyzed = [(r, build_features(r)) for r in rows]
    all_keys = sorted({k for _, f in analyzed for k in f})
    recs = []
    for key in all_keys:
        inside_all = [r for r, f in analyzed if f.get(key) is True]
        outside_all = [r for r, f in analyzed if f.get(key) is False]
        if len(inside_all) < min_n or len(outside_all) < min_n:
            continue
        inside = [r for r in inside_all if r.get("fillable")]
        outside = [r for r in outside_all if r.get("fillable")]
        if len(inside) < min_n or len(outside) < min_n:
            continue                      # 可成交子样本不足 → 不做检验（不计入 m）
        d = cluster_diff(inside, outside, MAIN_METRIC, "date")
        mi = sum(r[MAIN_METRIC] for r in inside) / len(inside)
        mo = sum(r[MAIN_METRIC] for r in outside) / len(outside)
        pi = sum(r.get("promoted") or 0 for r in inside_all) / len(inside_all)
        recs.append({"key": key, "n": len(inside_all), "n_fill": len(inside),
                     "in": mi, "out": mo,
                     # Δ 与 t 同源：此前 Δ 用未配对均值差 mi−mo、t/p/q 用按日配对，
                     # 两者可以异号（表里出现 Δ>0 而 t<0）
                     "delta": d["diff"] if d["diff"] is not None else mi - mo,
                     "t": d["t"], "days": d["days"],
                     "p": two_sided_p(d["t"]), "promo": pi})
    if not recs:
        return (f"【维度区分度分析】可成交子样本不足（每组需 ≥{min_n} 条），"
                f"暂无可判读的维度。")
    # 共线簇合并 → 有效检验数 m_eff（BH 用它，不用原始特征数，否则校正过度保守）
    tested = {r["key"] for r in recs}
    merged = sum(max(0, len(g & tested) - 1) for g in COLLINEAR_GROUPS)
    m_eff = max(1, len(recs) - merged)
    for r, q in zip(recs, bh_fdr([r["p"] for r in recs], m_eff)):
        r["q"] = q
    recs.sort(key=lambda r: -abs(r["t"]) if r["t"] is not None else 0.0)

    lines = [
        f"{'特征':<12}{'n':>4}{'可成交':>7}{'组内净%':>9}{'组外净%':>9}"
        f"{'Δ':>7}{'聚类t':>8}{'p':>7}{'q(BH)':>7}{'[参考]晋级':>11}",
        "-" * 84,
    ]
    for r in recs:
        t_s = f"{r['t']:+.2f}" if r["t"] is not None else "单簇"
        p_s = f"{r['p']:.3f}" if r["p"] is not None else "—"
        q_s = f"{r['q']:.3f}" if r["q"] is not None else "—"
        sig = r["q"] is not None and r["q"] < 0.10
        # 配对日不足门槛不打 ★（此前只要 q<0.10 就标，每组 ≥5 行即出 ★）
        mark = (" ★" if sig and r["days"] >= MIN_INFER_DAYS
                else " (配对日不足)" if sig else "")
        lines.append(f"{r['key']:<12}{r['n']:>4}{r['n_fill']:>7}"
                     f"{r['in']:>9.2f}{r['out']:>9.2f}{r['delta']:>+7.2f}"
                     f"{t_s:>8}{p_s:>7}{q_s:>7}{r['promo']:>10.1%}{mark}")
    m_eff = len(recs) - merged
    lines.append("-" * 84)
    lines.append(f"（主口径＝可成交净收益 {MAIN_METRIC}，仅 fillable 子样本；"
                 f"每组最少 {min_n} 条；按交易日聚类配对，按 |t| 降序；"
                 f"Δ=按日配对均值差，与 t 同源）")
    lines.append(f"（多重比较：原始 {len(recs)} 个检验，共线簇合并 {merged} 个 → "
                 f"**m_eff={m_eff}**；★ = BH q<0.10 且配对日≥{MIN_INFER_DAYS} 才算过关，"
                 f"(配对日不足) = q<0.10 但配对日<{MIN_INFER_DAYS}，不判读。"
                 f"零校正时 @0.05 下 P(≥1 假阳性)=1−0.95^{len(recs)}≈"
                 f"{1 - 0.95 ** len(recs):.0%}）")
    lines.append("（p 值为正态近似，簇数<30 时偏乐观；[参考]晋级 是信息口径，买不到）")
    return "\n".join(lines)
