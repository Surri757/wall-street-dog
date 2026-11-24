# -*- coding: utf-8 -*-
"""聚类推断统计核心（纯函数，无项目依赖）。

样本高度按日聚类：同一交易日的票共享市场情绪与连板高度，不是独立观测。
_forward.csv 177 条只跨 23 个独立交易日，朴素 i.i.d. SE 低估约 2.5 倍
（过线 vs 未过线晋级率差 29.4pp：朴素 z=4.32，按日聚类后 t=1.72、p≈0.10）。
因此本项目所有均值/差值判读一律走本模块，不得再直接比点估计。

本模块是 calibrate 时代拆出的**公共统计库**：research_scan 的因子对照、
calib 的分桶/维度分析、prereg 研究脚本共用同一实现——此前 research_scan
自带一套裸均值对比，恰好违反了这里编码的纪律。
"""
from __future__ import annotations

from math import erf, exp, lgamma, log, sqrt


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔的连分式（Numerical Recipes betacf，修正 Lentz 法）。"""
    MAXIT, EPS, FPMIN = 300, 3e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b)
             + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sided_p(t, df: int | None = None) -> float | None:
    """双侧 p 值。缺省正态近似（stdlib erf）；传 df（簇数−1）按 Student t。

    簇数少时正态近似偏乐观：df=7 的 5% 临界值是 2.36 而非 1.96。df 参数供
    调用方按配对日数选用；不传时与旧版逐位一致（既有判读口径不变）。
    """
    if t is None:
        return None
    x = abs(float(t))
    if df is None or df >= 30:
        return max(0.0, min(1.0, 1.0 - erf(x / sqrt(2.0))))
    if df < 1:
        return None
    if x == 0.0:
        return 1.0
    df = int(df)
    return _betai(df / 2.0, 0.5, df / (df + x * x))


def cluster_mean(rows: list[dict], val_key: str,
                 cluster_key: str = "date") -> dict:
    """单组的按簇稳健均值推断。

    SE 用 CR0 + 小样本校正：u_i = y_i − mean，按簇求和得 A_g，
    SE = sqrt( G/(G−1) · Σ A_g² ) / n。

    **均值与聚类分开过滤**：mean 只依赖值本身，缺簇键的行仍计入均值；
    SE/t/ci 才依赖簇键。样本为空或**单簇**（G<2，SE 无定义）时 t/ci 为 None——
    调用方必须降级为点估计并明确标注，不能把 None 当 0 或当显著。
    """
    def _ok(v):
        return v is not None and v == v
    vals = [r.get(val_key) for r in rows if _ok(r.get(val_key))]
    n = len(vals)
    if not n:
        return {"n": 0, "clusters": 0, "mean": None, "se": None, "t": None,
                "ci": (None, None)}
    mean = sum(vals) / n
    sums: dict = {}
    for r in rows:
        c, v = r.get(cluster_key), r.get(val_key)
        if not c or not _ok(v):
            continue
        sums[c] = sums.get(c, 0.0) + (v - mean)
    g = len(sums)
    if g < 2:
        return {"n": n, "clusters": g, "mean": mean, "se": None, "t": None,
                "ci": (None, None)}
    se = (sum(a * a for a in sums.values()) * g / (g - 1)) ** 0.5 / n
    t = mean / se if se > 0 else None
    ci = (mean - 1.96 * se, mean + 1.96 * se) if se > 0 else (None, None)
    return {"n": n, "clusters": g, "mean": mean, "se": se, "t": t, "ci": ci}


def cluster_diff(a_rows: list[dict], b_rows: list[dict], val_key: str,
                 cluster_key: str = "date") -> dict:
    """两组均值之差的推断，用**按日配对**设计。

    对每个同时含 A、B 两组样本的日期，算「组内均值之差」，再对这些日度差做
    单样本 t。比直接给差值套 cluster-robust 公式更稳健也更好解释（每个日度差
    已天然消除了当天的市场共同项），代价是只有单边样本的日期被丢弃 → 返回里
    报 dropped 数，丢弃过多时必须提示结论不可靠。
    """
    def by_day(rows):
        d: dict = {}
        for r in rows:
            c, v = r.get(cluster_key), r.get(val_key)
            if c and v is not None and v == v:
                d.setdefault(c, []).append(v)
        return d
    da, db = by_day(a_rows), by_day(b_rows)
    days = sorted(set(da) & set(db))
    dropped = len(set(da) ^ set(db))
    if not days:
        return {"n_a": len(a_rows), "n_b": len(b_rows), "days": 0,
                "dropped": dropped, "diff": None, "se": None, "t": None,
                "ci": (None, None)}
    diffs = [sum(da[d]) / len(da[d]) - sum(db[d]) / len(db[d]) for d in days]
    g = len(diffs)
    mean = sum(diffs) / g
    if g < 2:
        return {"n_a": len(a_rows), "n_b": len(b_rows), "days": g,
                "dropped": dropped, "diff": mean, "se": None, "t": None,
                "ci": (None, None)}
    var = sum((x - mean) ** 2 for x in diffs) / (g - 1)
    se = (var / g) ** 0.5
    t = mean / se if se > 0 else None
    ci = (mean - 1.96 * se, mean + 1.96 * se) if se > 0 else (None, None)
    return {"n_a": len(a_rows), "n_b": len(b_rows), "days": g,
            "dropped": dropped, "diff": mean, "se": se, "t": t, "ci": ci}


# 兼容别名：calibrate 时代的私有名已被 out/research/prereg/* 研究脚本导入，
# 属事实公共 API，拆包后必须继续可用。
_two_sided_p = two_sided_p


def bh_fdr(pvals: list, m: int | None = None) -> list:
    """Benjamini-Hochberg q 值。返回与输入同序；None（无法推断）原样保留。

    m 可显式传入以反映**有效**检验数（共线特征合并后小于原始特征数）；
    缺省取有效 p 值个数。
    """
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None]
    m = m or len(valid)
    q: list = [None] * len(pvals)
    if not valid or m <= 0:
        return q
    order = sorted(valid, key=lambda ip: ip[1])
    running = 1.0
    for rank in range(len(order), 0, -1):
        i, p = order[rank - 1]
        running = min(running, p * m / rank)
        q[i] = min(1.0, running)
    return q


def fmt_t(stat: dict) -> str:
    """渲染聚类 t；单簇/空样本时返回「单簇」占位，不能显示成 0 或显著。"""
    t = stat.get("t")
    return f"{t:+.2f}" if t is not None else "单簇"


def fmt_mean(stat: dict | None, width: int = 10) -> str:
    """渲染均值单元格；无样本/无均值时用破折号占位，不能显示成 0.00。"""
    if not stat or stat.get("mean") is None:
        return f"{'—':>{width}}"
    return f"{stat['mean']:>{width}.2f}"
