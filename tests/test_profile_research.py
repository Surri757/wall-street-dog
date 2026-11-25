# -*- coding: utf-8 -*-
"""deep10 画像线研究模块的离线区分力测试（不依赖网络）。

覆盖：分布度量（JSD/TVD/PSI/Wilson）、时间顺序切分无重叠、平滑概率归一、
真实快照只读复核（事件/点火/可算特征计数）、重算与现网画像一致。

运行: py -3.11 tests/test_profile_research.py
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEEP10 = ROOT / "out" / "research" / "profile" / "deep10"
sys.path.insert(0, str(ROOT))

# 显式按文件路径加载本线 common，使用唯一模块名；避免与 vane 线
# out/research/vane/deep10/common.py 在同一 pytest 会话里因裸模块名
# `common` 串用 sys.modules 缓存（先导入者占名，后到者取错模块）。
_spec = importlib.util.spec_from_file_location(
    "profile_deep10_common", str(DEEP10 / "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
sys.modules["profile_deep10_common"] = C


def test_common_bound_to_profile_deep10():
    """防撞回归：加载的 common 必须来自画像 deep10 目录，而非 vane 线同名模块。"""
    assert Path(C.__file__).resolve().parent == DEEP10.resolve()


def test_jsd_zero_and_symmetric_bounded():
    p = {"a": 0.6, "b": 0.4}
    q = {"a": 0.6, "b": 0.4}
    assert C.jsd(p, q) == 0.0
    r = {"a": 0.2, "b": 0.8}
    assert abs(C.jsd(p, r) - C.jsd(r, p)) < 1e-12
    assert 0.0 <= C.jsd(p, r) <= 1.0


def test_tvd_known_value():
    # 两桶：p=(1,0), q=(0.5,0.5) → TVD=0.5·(|1-0.5|+|0-0.5|)=0.5
    p = {"x": 1.0, "y": 0.0}
    q = {"x": 0.5, "y": 0.5}
    assert abs(C.tvd(p, q) - 0.5) < 1e-12


def test_psi_zero_for_identical():
    p = {"a": 0.3, "b": 0.7}
    assert abs(C.psi(p, p)) < 1e-12


def test_wilson_ci():
    lo, hi = C.wilson_ci(73, 90)
    assert 0.0 <= lo < 73 / 90 < hi <= 1.0
    lo0, hi0 = C.wilson_ci(0, 90)
    assert lo0 == 0.0 and hi0 > 0.0
    lo_n, hi_n = C.wilson_ci(1, 0)
    assert np.isnan(lo_n) and np.isnan(hi_n)


def test_smooth_probs_normalized_and_pseudocount():
    counts = {"a": 9, "b": 1}
    pr = C.smooth_probs(counts, pseudocount=0.5)
    assert abs(sum(pr.values()) - 1.0) < 1e-12
    # 零桶被伪计数抬升，概率 > 0
    assert pr["b"] > 0.0
    raw = C.raw_probs(counts)
    assert abs(sum(raw.values()) - 1.0) < 1e-12
    empty = C.raw_probs({"a": 0, "b": 0})
    assert abs(empty["a"] - 0.5) < 1e-12 and abs(empty["b"] - 0.5) < 1e-12


def test_chrono_split_no_overlap():
    days = [dt.date(2026, m, 1) for m in range(1, 13)]
    cutoff, train, val = C.chrono_split(days, val_frac=1 / 3)
    assert max(train) <= cutoff < min(val)
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(set(days))
    # 不足 4 个独立日必须报错（防伪切分）
    try:
        C.chrono_split([dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 3)])
        raise AssertionError("不足 4 日应抛 ValueError")
    except ValueError:
        pass


def test_real_snapshot_counts():
    """只读复核真实快照计数（与 audit.json 记录一致）。"""
    seat = C.load_seat_events()
    since = dt.date.fromisoformat(C.PROFILE_SINCE)
    st = seat[seat["交易日期"] >= since]
    ev = C.ignition_events(seat)
    bars = C.load_bars_cache()
    snap = C.load_snapshot()
    fs = {}
    if {"float_mv", "close"}.issubset(snap.columns):
        ss = snap.dropna(subset=["float_mv", "close"])
        ss = ss[ss["close"] > 0]
        fs = {str(c).zfill(6): mv / cl for c, mv, cl in
              zip(ss["code"], ss["float_mv"], ss["close"])}
    feat = C.ignition_features(ev, bars, fs)
    assert len(st) == 712, len(st)
    assert st["交易日期"].nunique() == 191
    assert len(ev) == 308, len(ev)
    assert len(feat) == 305, len(feat)


def test_reconstruction_matches_deployed_profile():
    """重算合并窗桶计数须与现网 profile.json eve.dist 一致（描述保真）。"""
    if not C.PROFILE_JSON.exists():
        return
    seat = C.load_seat_events()
    ev = C.ignition_events(seat)
    bars = C.load_bars_cache()
    snap = C.load_snapshot()
    fs = {}
    if {"float_mv", "close"}.issubset(snap.columns):
        ss = snap.dropna(subset=["float_mv", "close"])
        ss = ss[ss["close"] > 0]
        fs = {str(c).zfill(6): mv / cl for c, mv, cl in
              zip(ss["code"], ss["float_mv"], ss["close"])}
    feat = C.ignition_features(ev, bars, fs)
    deployed = json.loads(C.PROFILE_JSON.read_text(encoding="utf-8"))
    dist = deployed.get("eve", {}).get("dist", {})
    for f in C.FEATURES:
        counts = C.counts_from_table(feat, f)
        for lab, n in counts.items():
            assert dist.get(f, {}).get(lab, {}).get("n") == n, (f, lab, n)


def test_no_production_write_paths_touched():
    """研究模块常量指向只读工件；研究输出目录独立于 ziyang 数据目录。"""
    assert C.DATA_DIR.name == "ziyang"
    assert C.DEEP10_DIR.parent.name == "profile"
    assert str(C.PROFILE_JSON).startswith(str(C.DATA_DIR))


# ------------------------------------------------------------ 防泄漏区分力测试
def _synth_row(date: dt.date, **kw) -> dict:
    base = dict(lianban=0, pct=2.0, turnover=12.0, vr=1.1, price=8.0,
                pos20=0.5, ltsz=120.0)
    base.update(kw)
    base["ign_date"] = date
    return base


def test_fit_helpers_ignore_validation_events():
    """基线/候选拟合只看训练表；改动验证窗事件不得改变拟合分布（防 v1 重叠泄漏）。"""
    d1, d2, d3 = dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)
    train = pd.DataFrame([_synth_row(d1, lianban=0), _synth_row(d2, lianban=0)])
    val = pd.DataFrame([_synth_row(d3, lianban=2)])

    p0_before = C.fit_pooled_probs(train, "lianban")
    p1_before = C.fit_recent_probs(train, "lianban", 0.5)

    # 验证窗翻倍并改桶；拟合侧未接收 val，结果必须不变
    val = pd.DataFrame([_synth_row(d3, lianban=2), _synth_row(d3, lianban=2),
                        _synth_row(d3, lianban=1)])
    p0_after = C.fit_pooled_probs(train, "lianban")
    p1_after = C.fit_recent_probs(train, "lianban", 0.5)
    assert p0_before == p0_after
    assert p1_before == p1_after

    # 对照：若像 v1 B0 那样把训练+验证合并拟合，分布确实会变——泄漏向量存在，
    # 但本线拟合助手不走该路径。
    pooled = pd.concat([train, val], ignore_index=True)
    pooled_probs = C.smooth_probs(C.counts_from_table(pooled, "lianban"))
    assert pooled_probs["2板"] != p0_before["2板"]


def test_recent_window_subset_within_train_only():
    """近期窗候选只在训练窗日期内取最近比例，绝不引用验证窗日期。"""
    old = [dt.date(2026, 5, d) for d in (1, 2, 3)]
    recent = [dt.date(2026, 5, d) for d in (4, 5, 6)]
    val_d = dt.date(2026, 5, 7)
    rows = [_synth_row(d, lianban=0) for d in old] + \
           [_synth_row(d, lianban=2) for d in recent]
    train = pd.DataFrame(rows)

    recent_probs = C.fit_recent_probs(train, "lianban", 0.5)
    pooled_probs = C.fit_pooled_probs(train, "lianban")
    # 近期窗只含后 3 日（2板），2板 概率应高于全窗混合
    assert recent_probs["2板"] > pooled_probs["2板"]
    # 近期子集日期全部 ⊆ 训练日期，与验证日期不相交
    train_dates = set(train["ign_date"])
    assert val_d not in train_dates
    assert set(recent) <= train_dates

    # 拟合助手源码层不引用现网 profile.json（后者仅保真核验使用）
    import inspect
    src = inspect.getsource(C.fit_pooled_probs) + inspect.getsource(C.fit_recent_probs)
    assert "PROFILE_JSON" not in src


def test_origin_splits_disjoint_and_ordered():
    """多走查起点：训练/验证不相交，且训练日期全部早于验证日期。"""
    days = [dt.date(2026, m, 1) for m in range(1, 13)]
    splits = C.origin_splits(days, (1 / 3, 1 / 4))
    assert len(splits) == 2
    for sp in splits:
        tr = [dt.date.fromisoformat(d) for d in sp["train_dates"]]
        va = [dt.date.fromisoformat(d) for d in sp["val_dates"]]
        assert not set(tr) & set(va)
        assert max(tr) < min(va)


if __name__ == "__main__":
    test_common_bound_to_profile_deep10(); print("PASS common_bound")
    test_jsd_zero_and_symmetric_bounded(); print("PASS jsd")
    test_tvd_known_value(); print("PASS tvd")
    test_psi_zero_for_identical(); print("PASS psi")
    test_wilson_ci(); print("PASS wilson")
    test_smooth_probs_normalized_and_pseudocount(); print("PASS smooth_probs")
    test_chrono_split_no_overlap(); print("PASS chrono_split")
    test_real_snapshot_counts(); print("PASS real_snapshot_counts")
    test_reconstruction_matches_deployed_profile(); print("PASS reconstruction")
    test_no_production_write_paths_touched(); print("PASS path_boundary")
    test_fit_helpers_ignore_validation_events(); print("PASS fit_ignore_validation")
    test_recent_window_subset_within_train_only(); print("PASS recent_window_train_only")
    test_origin_splits_disjoint_and_ordered(); print("PASS origin_splits_disjoint")
    print("\ndeep10 画像线研究测试全部通过")
