# -*- coding: utf-8 -*-
"""信号健康度判读：台账实盘成绩 vs config.EXPECT 预注册期望。"""
from __future__ import annotations

from calib.forward import MIN_INFER_DAYS
from calib.stats import cluster_mean
from config import EXPECT

# 判读阈值（预注册，研究日志五P）：期望值取自 config.EXPECT 的 **prereg**
# 字段——判读基准必须是开跑前预注册的数字，换成事后修订的 current 就是移动球门。
# current（五R 终审等最新实测）只用于展示，不参与预警判读。
LEDGER_EXPECT = {
    k: {"mean": EXPECT[k]["prereg"][0], "win": EXPECT[k]["prereg"][1],
        "label": EXPECT[k]["label"]}
    for k in ("open_pool_t0", "slam")
}
LEDGER_WARN_PP = 2.0      # 单笔均值低于史测该 pp 数 → 预警
# 判读门槛复用 MIN_INFER_DAYS（独立交易日簇数）。旧的 LEDGER_MIN_N=8 数的是**笔数**：
# 实测 59 笔已平仓只跨 4 个 signal_date，其中 30 笔挤在同一天——同日并发持仓共享
# 当天市场状态，不是独立样本，笔数再多也攒不出推断力。


def _row_pair(r) -> tuple | None:
    """台账 CSV 行 → (signal_date, pnl 百分点)；未平仓/无 pnl/脏值返回 None。

    pnl 优先取 `pnl_net`（扣双边成本），缺失回退毛 `pnl`——旧台账行没有该列，
    判读必须在新旧两种 schema 下都能工作。此规则只此一份，open_pool 分组与
    slam 整表都走它，避免两处各写一遍导致口径分叉。
    """
    v = r.get("pnl_net")
    if v in ("", None):
        v = r.get("pnl")
    if v in ("", None):
        return None
    try:
        return (str(r.get("signal_date") or ""), float(v) * 100.0)
    except (TypeError, ValueError):
        return None


def _ledger_rows(path):
    """读台账已平仓行 → [(signal_date, pnl_pct), ...]；文件缺失/损坏返回 []。"""
    import csv as _csv
    import os as _os
    if not _os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return []
    return [p for p in (_row_pair(r) for r in rows) if p is not None]


def _judge(label: str, pairs: list, exp_mean: float) -> str:
    """按预注册阈值判读单信号：预警 / 正常 / 簇数不足。

    偏离的显著性用 dev/se（se 为按 signal_date 聚类的均值标准误）——这才是
    「实盘是否真的弱于史测」的检验量；旧实现只比两个点估计。
    """
    if not pairs:
        return f"  · {label}: 台账无已平仓记录"
    pnls = [v for _, v in pairs]
    mean = sum(pnls) / len(pnls)
    win = sum(1 for v in pnls if v > 0) / len(pnls) * 100
    st = cluster_mean([{"date": d or "unknown", "v": v} for d, v in pairs], "v", "date")
    dev = mean - exp_mean
    base = (f"{label}: {len(pnls)}笔/{st['clusters']}个交易日 "
            f"均值{mean:+.2f}% (史测{exp_mean:+.2f}%) 胜率{win:.0f}%")
    if st["clusters"] < MIN_INFER_DAYS:
        return (f"  · {base} → ⚠ **仅 {st['clusters']} 个交易日**（<{MIN_INFER_DAYS}），"
                f"无法推断。同日并发持仓不是独立样本，笔数攒再多也不算够；"
                f"当前偏离 {dev:+.2f}pp 只是点估计")
    t_dev = dev / st["se"] if st["se"] else None
    t_s = f"，偏离t={t_dev:+.2f}" if t_dev is not None else ""
    if dev <= -LEDGER_WARN_PP:
        return (f"  ⚠ {base} → 实盘弱于史测 {dev:+.2f}pp（阈值 −{LEDGER_WARN_PP}pp）"
                f"{t_s}，建议降仓/停用")
    return f"  ✓ {base} → 偏离 {dev:+.2f}pp 在阈值内{t_s}，正常"


def ledger_section(out_dir: str) -> str:
    """【信号健康度】：台账实盘成绩 vs 史测期望，自动判读偏离（预注册阈值）。"""
    import csv as _csv
    import os as _os
    lines = ["\n【信号健康度（台账实盘 vs 史测期望，预注册判读）】"]
    any_data = False

    p = _os.path.join(out_dir, "archive", "open_pool_ledger.csv")
    if _os.path.exists(p):
        any_data = True
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            rows = []
        groups = {"0": [], "12": [], "legacy": []}
        for r in rows:
            pair = _row_pair(r)
            if pair is None:
                continue
            t = str(r.get("tier", ""))
            key = "0" if t == "0" else ("12" if t in ("1", "2") else "legacy")
            groups[key].append(pair)
        exp = LEDGER_EXPECT["open_pool_t0"]
        for key, lab in (("0", exp["label"]), ("12", "开盘池观察档"),
                         ("legacy", "开盘池旧规则期")):
            if groups[key]:
                lines.append(_judge(lab, groups[key], exp["mean"]))

    p2 = _os.path.join(out_dir, "archive", "slam_ledger.csv")
    if _os.path.exists(p2):
        any_data = True
        exp = LEDGER_EXPECT["slam"]
        lines.append(_judge(exp["label"], _ledger_rows(p2), exp["mean"]))

    if not any_data:
        lines.append("  （台账尚未建立——开盘池/打板画像信号出现后自动记账）")
    return "\n".join(lines)
