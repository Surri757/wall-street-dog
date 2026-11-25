# -*- coding: utf-8 -*-
"""premium11 验证纯函数模块：逐日配对、状态隔离、合同指标、NW-HAC、部署门。

零网络、零 IO；阈值由 gate.py 从 ``contract.json`` 注入，模块内不另立业务默认值。
口径依据：``out/research/vane/premium11/contract.json``（冻结合同优先于
validation_research.md）。

臂：
- A  新溢价模型 Top3（固定三席，每席目标资金 1/3，未成交留现、不重分配）
- B1 旧晋级模型 Top3（同 A 席位规则）
- B2 同宇宙 T+1 实际可成交全体等权
- B0 现有启发式兜底/影子对照，不替代 B1/B2 硬基线
"""
from __future__ import annotations

import math
from collections import defaultdict

from config import COST_SLIPPAGE  # 成本单一事实源，禁止另写字面量

# ── 统一状态词表（entry / exit 两组；旧词表别名在 normalize_record 归一） ──────
ENTRY_FILLED = "filled"
ENTRY_PARTIAL = "partial"
ENTRY_UNFILLED = "unfilled"
ENTRY_NODATA = "nodata"
ENTRY_FAIL = "fail"

EXIT_EXITED = "exited"
EXIT_PENDING = "pending"
EXIT_CARRY = "carry"
EXIT_NODATA = "nodata"
EXIT_FAIL = "fail"
EXIT_UNFILLED = "unfilled"

SETTLED_ENTRY = {ENTRY_FILLED, ENTRY_PARTIAL}
UNRESOLVED_EXIT = {EXIT_PENDING, EXIT_CARRY, EXIT_NODATA, EXIT_FAIL,
                    EXIT_UNFILLED}

# 旧状态词表 → 统一词表别名（历史台账兼容，不改写原始行）
_ENTRY_ALIAS = {
    "unfilled_one_word": ENTRY_UNFILLED,
    "halted_no_data": ENTRY_NODATA,
    "nodata_fail": ENTRY_FAIL,
}
_EXIT_ALIAS = {
    "pending_limit_down": EXIT_PENDING,
    "pending_suspension": EXIT_PENDING,
    "unresolved_censored": EXIT_UNFILLED,
}

ARMS = ("A", "B1", "B2")
SHADOW_ARMS = ("B0",)

# 近似层永远只能 SCREENING；真实层才允许过门
LAYER_MINUTE20 = "minute20"
LAYER_APPROX = "approx_open"


# ── 严格解析（CSV 数值/布尔均为字符串，空串/None 保持缺失） ──────────────────

def _as_float(x):
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _as_bool(x) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def _finite(x) -> bool:
    return _as_float(x) is not None


def normalize_record(rec: dict) -> dict:
    """归一状态别名、数值与布尔；返回浅拷贝，不改动入参。"""
    out = dict(rec)
    out["entry_status"] = _ENTRY_ALIAS.get(
        str(out.get("entry_status") or "").strip(),
        str(out.get("entry_status") or "").strip()) or None
    out["exit_status"] = _EXIT_ALIAS.get(
        str(out.get("exit_status") or "").strip(),
        str(out.get("exit_status") or "").strip()) or None
    for k in ("net_ret", "gross_ret", "model_score", "old_promo_p",
              "filled_ratio", "weight", "cash_weight", "mkt_height", "B", "rank"):
        if k in out:
            out[k] = _as_float(out[k])
    for k in ("unresolved_flag", "carry_flag", "complete"):
        if k in out:
            out[k] = _as_bool(out[k])
    return out


def is_unresolved(rec: dict) -> bool:
    """未决/顺延/数据缺失：未实际退出或拉取失败，不计已结算样本。"""
    if _as_bool(rec.get("unresolved_flag")):
        return True
    return rec.get("exit_status") in UNRESOLVED_EXIT


def is_settled(rec: dict) -> bool:
    """已结算往返：成交入场、已退出、无未决标记、净收益为有限小数。"""
    return (rec.get("entry_status") in SETTLED_ENTRY
            and rec.get("exit_status") == EXIT_EXITED
            and not is_unresolved(rec)
            and _finite(rec.get("net_ret")))


def _filled_ratio(rec: dict) -> float:
    fr = _as_float(rec.get("filled_ratio"))
    if fr is None:
        return 1.0
    return min(max(fr, 0.0), 1.0)


# ── Newey-West HAC（合同：Bartlett，lag=max(1, floor(n^(1/3)))，手写无第三方依赖） ──

def hac_lag(n: int) -> int:
    if n <= 1:
        return 0
    return max(1, int(math.floor(n ** (1.0 / 3.0))))


def nw_hac_mean_ci(x, z: float = 1.96) -> dict:
    """均值的 Newey-West HAC 标准误与双侧 95% CI。

    返回 ``{n, mean, se, lag, ci_low, ci_high}``；空序列/单样本给出退化结果，
    调用方须自行按样本门拒绝。同日多票已在上游聚合成日簇，这里只对日序列估计。
    """
    arr = [v for v in (_as_float(t) for t in x) if v is not None]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "lag": None,
                "ci_low": None, "ci_high": None}
    mean = sum(arr) / n
    if n == 1:
        return {"n": 1, "mean": mean, "se": 0.0, "lag": 0,
                "ci_low": mean, "ci_high": mean}
    lag = min(hac_lag(n), n - 1)
    e = [v - mean for v in arr]
    gamma0 = sum(v * v for v in e) / n
    omega = gamma0
    for l in range(1, lag + 1):
        gamma_l = sum(e[i] * e[i + l] for i in range(n - l)) / n
        omega += 2.0 * (1.0 - l / (lag + 1)) * gamma_l
    omega = max(omega, 0.0)  # 数值保护，HAC 方差半正定
    se = math.sqrt(omega / n)
    return {"n": n, "mean": mean, "se": se, "lag": lag,
            "ci_low": mean - z * se, "ci_high": mean + z * se}


# ── 逐日组合收益（日簇口径，现金留现不重分配） ──────────────────────────────

def _arm_rows(records: list[dict], arm: str) -> list[dict]:
    return [r for r in records if r.get("arm") == arm]


# Top3 固定席位臂：每席目标资金 1/3，未成交留现且不重分配；B0 为影子 Top3，
# 席位规则与 A/B1 一致。B2 为同宇宙实际可成交全体等权基线。
FIXED_SEAT_ARMS = ("A", "B1", "B0")


def cohort_day(records: list[dict], arm: str) -> dict:
    """单臂单信号日组合净收益。

    - A/B1/B0：固定三席，每席目标 1/3；不足 3 只有多少列多少，每席仍 1/3，
      空余留现；未成交席现金贡献 0，不把资金重分配给其余成交票。
    - B2：同宇宙实际可成交全体等权；成交部分按权结算，未成交部分留现。
    """
    rows = _arm_rows(records, arm)
    unresolved = any(is_unresolved(r) for r in rows)
    settled = [r for r in rows if is_settled(r)]

    if arm in FIXED_SEAT_ARMS:
        target = 1.0 / 3.0
        weights = [target] * len(rows)
    else:
        k = len(settled)  # 实际可成交且成交的只数
        target = 1.0 / k if k else 0.0
        weights = [target] * len(rows)

    contribution = 0.0
    cash_weight = 0.0
    for r, w in zip(rows, weights):
        if is_settled(r):
            fr = _filled_ratio(r)
            invested = w * fr
            contribution += invested * float(r["net_ret"])
            cash_weight += w * (1.0 - fr)
        else:
            cash_weight += w

    return {"day_ret": contribution, "cash_weight": cash_weight,
            "n_rows": len(rows), "n_settled": len(settled),
            "n_unresolved": sum(is_unresolved(r) for r in rows),
            "unresolved": unresolved}


def paired_cohorts(records: list[dict], arms: tuple[str, ...] = ARMS):
    """按 signal_date 配对三臂；缺臂或任一臂存在未决头寸的 cohort 剔除并披露。

    返回 ``(paired_dates, excluded)``：paired_dates 升序；excluded 逐条写明原因。
    """
    by_date: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_date[r["signal_date"]][r.get("arm")].append(r)

    paired, excluded = [], []
    for d in sorted(by_date):
        arms_day = by_date[d]
        missing = [a for a in arms if a not in arms_day]
        if missing:
            excluded.append({"signal_date": d, "reason": "arm_missing",
                             "missing": missing, "arms": sorted(arms_day)})
            continue
        bad = [a for a in arms if any(is_unresolved(x) for x in arms_day[a])]
        if bad:
            excluded.append({"signal_date": d, "reason": "unresolved_position",
                             "arms_with_unresolved": bad})
            continue
        paired.append(d)
    return paired, excluded


def day_series(records: list[dict], arm: str, dates: list[str]) -> list[float]:
    """对齐配对日序的臂日净收益；缺失日由 paired_cohorts 保证同表。"""
    by = {d: cohort_day([r for r in records if r["signal_date"] == d], arm)
          for d in dates}
    return [by[d]["day_ret"] for d in dates]


# ── 合同指标 ───────────────────────────────────────────────────────────────

def compound_nav(day_rets: list[float]) -> tuple[float, float]:
    """复利净值与最大回撤。

    MDD 以**正值回撤幅度**返回（0 表示无回撤，0.10 = 10% 回撤）；风险门按
    ``mdd <= 0.10`` 比较，深回撤不会被负号方向放过。
    """
    nav, peak, mdd = 1.0, 1.0, 0.0
    for r in day_rets:
        nav *= 1.0 + r
        peak = max(peak, nav)
        if peak > 0:
            mdd = max(mdd, 1.0 - nav / peak)
    return nav, mdd


def monthly_returns(day_rets: list[float], dates: list[str]) -> dict[str, float]:
    """按月复利的月度净收益；返回 {YYYY-MM: 月内复利净收益}。"""
    by_month: dict[str, list[float]] = defaultdict(list)
    for d, r in zip(dates, day_rets):
        by_month[d[:7]].append(r)
    return {m: math.prod(1.0 + v for v in rs) - 1.0
            for m, rs in sorted(by_month.items())}


def concentration(settled_net: list[float]) -> dict:
    """盈利集中度：毛利润、Top3 盈利贡献占比、剔最大 3 笔后剩余均值。"""
    pos = [x for x in settled_net if x > 0]
    gross = sum(pos)
    top3 = sorted(pos, reverse=True)[:3]
    top3_share = (sum(top3) / gross) if gross > 0 else None
    rest = sorted(settled_net)[:max(0, len(settled_net) - 3)]  # 去尾：剔除最大 3 笔
    rest_mean = (sum(rest) / len(rest)) if rest else None
    return {"gross_profit": gross, "n_trades": len(settled_net),
            "top3_profit_share": top3_share, "drop_best3_mean": rest_mean}


def stress_day_returns(records: list[dict], dates: list[str],
                       arm: str) -> list[float]:
    """成本压力档：滑点翻倍——在原往返成本上再加一份 COST_SLIPPAGE。"""
    out = []
    for d in dates:
        rows = [r for r in records if r["signal_date"] == d and r.get("arm") == arm]
        settled = [r for r in rows if is_settled(r)]
        if arm in FIXED_SEAT_ARMS:
            w = 1.0 / 3.0
        else:
            k = len(settled)
            w = 1.0 / k if k else 0.0
        contrib = 0.0
        for r in rows:
            if is_settled(r):
                fr = _filled_ratio(r)
                stress_net = float(r["net_ret"]) - COST_SLIPPAGE
                contrib += w * fr * stress_net
        out.append(contrib)
    return out


def settled_nets(records: list[dict], arm: str = "A",
                 dates: list[str] | None = None) -> list[float]:
    date_set = set(dates) if dates else None
    return [float(r["net_ret"]) for r in _arm_rows(records, arm)
            if is_settled(r) and (date_set is None or r["signal_date"] in date_set)]


# ── 部署门评估（阈值来自 contract.json；不在模块内移动阈值） ────────────────

def _gate(passed: bool, value, reason: str, **extra) -> dict:
    g = {"pass": bool(passed), "value": value, "reason": reason}
    g.update(extra)
    return g


def _regime_map(records: list[dict]) -> dict[str, str]:
    """日 regime：优先记录自带 ``regime``；其次 mkt_height 验证期中位数高低分组。"""
    by_date: dict[str, list] = defaultdict(list)
    for r in records:
        by_date[r["signal_date"]].append(r)
    heights = [_as_float(r.get("mkt_height")) for rs in by_date.values()
               for r in rs if _as_float(r.get("mkt_height")) is not None]
    median_h = sorted(heights)[len(heights) // 2] if heights else None
    out: dict[str, str] = {}
    for d, rs in by_date.items():
        tag = next((r.get("regime") for r in rs if r.get("regime")), None)
        if tag:
            out[d] = tag
        elif median_h is not None:
            h = next((_as_float(r.get("mkt_height")) for r in rs
                      if _as_float(r.get("mkt_height")) is not None), None)
            out[d] = "high" if h is not None and h >= median_h else "low"
        else:
            out[d] = "unknown"
    return out


def evaluate_gates(records: list[dict], thresholds: dict,
                   layer: str, regime_by_date: dict[str, str] | None = None) -> dict:
    """按合同逐门评估；返回完整门报告。

    thresholds 键见 contract.json ``sample_gate_minute20_only`` 与
    ``deployment_gate_all_required``。任何未决/数据不足/近似层都不得 promote。
    """
    records = [normalize_record(r) for r in records]
    regime_by_date = regime_by_date or _regime_map(records)
    report: dict = {"layer": layer, "promote": False, "gates": {},
                    "sample": {}, "excluded_cohorts": [], "arms": {}}

    paired, excluded = paired_cohorts(records)
    report["excluded_cohorts"] = excluded
    if not paired:
        report["sample"] = {"n_signal_days": 0, "n_months": 0, "n_regimes": 0,
                            "n_settled_trips": 0, "unresolved_records":
                                sum(is_unresolved(r) for r in records)}
        report["gates"]["sample_gate"] = _gate(False, 0, "无配对已结算 cohort")
        report["decision"] = "no_pairable_cohort"
        return report

    a_days = day_series(records, "A", paired)
    b1_days = day_series(records, "B1", paired)
    b2_days = day_series(records, "B2", paired)
    report["arms"] = {
        "A": {"n_days": len(paired), "day_returns": [round(x, 8) for x in a_days]},
        "B1": {"day_returns": [round(x, 8) for x in b1_days]},
        "B2": {"day_returns": [round(x, 8) for x in b2_days]},
    }

    # ── 样本门（仅 minute20 可计数；approx_open 整层降级 SCREENING） ──
    months = sorted({d[:7] for d in paired})
    regime_tags = {regime_by_date.get(d, "unknown") for d in paired}
    known_regimes = {t for t in regime_tags if t and t != "unknown"}
    n_trips = len(settled_nets(records, "A", paired))
    n_unresolved = sum(is_unresolved(r) for r in records)
    report["sample"] = {"n_signal_days": len(paired), "n_months": len(months),
                         "n_regimes": len(known_regimes),
                         "n_regimes_with_unknown": len(regime_tags),
                         "n_settled_trips": n_trips,
                         "unresolved_records": n_unresolved,
                         "dates": paired, "months": months,
                         "regimes": sorted(regime_tags)}

    smin = thresholds.get("sample_gate_minute20_only", {})
    sample_reasons = []
    if layer != LAYER_MINUTE20:
        sample_reasons.append(f"layer={layer} 非 minute20，样本门不计入")
    if len(paired) < smin.get("independent_signal_days_min", 20):
        sample_reasons.append(f"独立信号日 {len(paired)} < "
                              f"{smin.get('independent_signal_days_min', 20)}")
    if len(months) < smin.get("calendar_months_min", 2):
        sample_reasons.append(f"覆盖月份 {len(months)} < {smin.get('calendar_months_min', 2)}")
    if len(known_regimes) < smin.get("regimes_min", 2):
        sample_reasons.append(f"有效 regime {len(known_regimes)} < "
                              f"{smin.get('regimes_min', 2)}（unknown 不计）")
    if n_trips < smin.get("settled_round_trips_min", 30):
        sample_reasons.append(f"已结算往返 {n_trips} < "
                              f"{smin.get('settled_round_trips_min', 30)}")
    if n_unresolved and not smin.get("unresolved_allowed_for_promotion", False):
        sample_reasons.append(f"存在未决头寸 {n_unresolved} 笔，不允许 promote")
    sample_pass = not sample_reasons
    report["gates"]["sample_gate"] = _gate(sample_pass, report["sample"],
                                           "样本充分" if sample_pass else "；".join(sample_reasons))

    # ── 盈利门 ──
    a_ci = nw_hac_mean_ci(a_days)
    cum_a = sum(a_days)
    nav_a, mdd_a = compound_nav(a_days)
    profit_ok = (cum_a > 0 and a_ci["mean"] is not None and a_ci["mean"] > 0
                 and a_ci["ci_low"] is not None and a_ci["ci_low"] > 0)
    report["gates"]["profit"] = _gate(
        profit_ok, {"cumulative": round(cum_a, 8), "mean": a_ci["mean"],
                    "hac_se": a_ci["se"], "lag": a_ci["lag"],
                    "ci_low": a_ci["ci_low"], "nav_final": nav_a},
        "累计/日均/HAC 下界均 >0" if profit_ok else
        f"累计 {cum_a:.4f} 或均值 {a_ci['mean']} 或 CI 下界 {a_ci['ci_low']} 未 >0")

    # ── 基线击败门（配对差，日簇口径） ──
    d1 = [a - b for a, b in zip(a_days, b1_days)]
    d2 = [a - b for a, b in zip(a_days, b2_days)]
    d1_ci, d2_ci = nw_hac_mean_ci(d1), nw_hac_mean_ci(d2)
    beats1 = d1_ci["mean"] is not None and d1_ci["mean"] > 0 and d1_ci["ci_low"] > 0
    beats2 = d2_ci["mean"] is not None and d2_ci["mean"] > 0 and d2_ci["ci_low"] > 0
    report["gates"]["beats_baselines"] = _gate(
        beats1 and beats2,
        {"A_minus_B1": {"mean": d1_ci["mean"], "ci_low": d1_ci["ci_low"],
                        "lag": d1_ci["lag"]},
         "A_minus_B2": {"mean": d2_ci["mean"], "ci_low": d2_ci["ci_low"],
                        "lag": d2_ci["lag"]}},
        "A 同时击败 B1/B2（配对差均值与 HAC 下界 >0）" if beats1 and beats2
        else "未同时击败两条硬基线")

    # ── 跨时期门 ──
    monthly = monthly_returns(a_days, paired)
    pos_months = sum(1 for v in monthly.values() if v > 0)
    frac_pos = pos_months / len(monthly) if monthly else 0.0
    best_month = max(monthly.values()) if monthly else 0.0
    cum_drop_best = sum(v for m, v in monthly.items() if v != best_month) \
        if len(monthly) > 1 else sum(monthly.values())
    cp = thresholds.get("cross_period", {})
    cross_ok = (frac_pos >= cp.get("month_positive_frac", 0.60)
                and cum_drop_best > 0)
    report["gates"]["cross_period"] = _gate(
        cross_ok, {"monthly": monthly, "positive_frac": frac_pos,
                   "drop_best_month_cum": cum_drop_best},
        "≥60% 月份为正且剔最佳月累计 >0" if cross_ok else
        f"正收益月占比 {frac_pos:.2f} 或剔最佳月累计 {cum_drop_best:.4f} 不达标")

    # ── 集中度门 ──
    conc = concentration(settled_nets(records, "A", paired))
    top3_ok = conc["top3_profit_share"] is not None and conc["top3_profit_share"] < 0.50
    drop3_ok = conc["drop_best3_mean"] is not None and conc["drop_best3_mean"] > 0
    conc_ok = conc["gross_profit"] > 0 and top3_ok and drop3_ok
    report["gates"]["concentration"] = _gate(
        conc_ok, conc,
        "Top3 盈利占比 <50% 且剔最佳 3 笔均值 >0" if conc_ok else
        "毛利润 ≤0 或集中度/剔最佳笔不达标")

    # ── 风险门（MDD 为正值回撤幅度） ──
    worst_day = min(a_days) if a_days else None
    risk_ok = mdd_a <= 0.10 and worst_day is not None and worst_day >= -0.03
    report["gates"]["risk"] = _gate(
        risk_ok, {"mdd": mdd_a, "worst_day": worst_day, "nav_final": nav_a},
        "MDD ≤10% 且最差日 ≥−3%" if risk_ok else
        f"MDD {mdd_a:.4f} 或最差日 {worst_day} 越线")

    # ── 折级/日级非劣门 ──
    min_base = [min(b1, b2) for b1, b2 in zip(b1_days, b2_days)]
    noninf = [a - mb >= -0.005 for a, mb in zip(a_days, min_base)]
    frac_noninf = sum(noninf) / len(noninf) if noninf else 0.0
    any_bad = worst_day is not None and worst_day < -0.03
    ni_ok = frac_noninf >= 2.0 / 3.0 and not any_bad
    report["gates"]["fold_noninferiority"] = _gate(
        ni_ok, {"noninferior_frac": frac_noninf,
                "days_below_min_base": sum(not x for x in noninf),
                "worst_day": worst_day},
        "≥2/3 日非劣于 min(B1,B2)−0.5pp 且无单日 <−3%" if ni_ok else
        f"非劣占比 {frac_noninf:.2f} 或存在单日越线")

    # ── 成本压力门（保护门：滑点翻倍后日均仍 >0） ──
    stress = stress_day_returns(records, paired, "A")
    stress_ci = nw_hac_mean_ci(stress)
    stress_ok = stress_ci["mean"] is not None and stress_ci["mean"] > 0
    report["gates"]["cost_stress"] = _gate(
        stress_ok, {"stress_mean": stress_ci["mean"], "stress_ci_low": stress_ci["ci_low"]},
        "滑点翻倍后日均净收益仍 >0" if stress_ok else "滑点翻倍后日均转负")

    # ── 决策：硬门全过 + 样本门过 + 真实层 + 无未决，才允许 promote ──
    hard = ["profit", "beats_baselines", "cross_period", "concentration",
            "risk", "fold_noninferiority", "cost_stress"]
    all_hard = all(report["gates"][h]["pass"] for h in hard)
    sample_pass = report["gates"]["sample_gate"]["pass"]
    layer_ok = layer == LAYER_MINUTE20
    # 近似层无论数字多好一律 screening，禁止 promote
    report["promote"] = bool(all_hard and sample_pass and layer_ok)
    report["decision"] = ("promote" if report["promote"] else
                          ("screening_only" if not layer_ok else "shadow_only"))
    report["hard_gates_all_pass"] = all_hard
    return report
