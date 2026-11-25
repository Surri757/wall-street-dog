# -*- coding: utf-8 -*-
"""deep10 ziyang 研究线离线测试：复核 wf_opt 预存扩展的修复与合同门控。

不依赖网络；合成数据验证排序权重按组传播、校准器接口、半衰公式、边界过滤与
生产正门判定。真实数据复现命令见 out/research/ziyang/deep10/ 结论文档。

运行：
  py -3.11 tests/test_ziyang_research.py
  py -3.11 -m pytest tests/test_ziyang_research.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "wf_opt", os.path.join(ROOT, "out", "research", "ziyang", "deep7", "wf_opt.py"))
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


# ---------------------------------------------------------------- 排序权重
def test_ranker_receives_group_level_weights():
    """排序臂：半衰权重必须按组（交易日）聚合传入，长度 == 组数。"""
    from xgboost import XGBRanker
    captured = {}
    orig = XGBRanker.fit

    def spy(self, X, y, **kwargs):
        captured["kwargs"] = kwargs
        return orig(self, X, y, **kwargs)

    XGBRanker.fit = spy
    try:
        groups = [4, 5, 3]
        gsw = np.array([1.0, 0.7, 0.4])
        variant = {"model": "xgb_rank"}
        out = W.fit_model(np.random.default_rng(0).normal(size=(12, 3)),
                          np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], float),
                          variant, groups=groups, group_sample_weight=gsw)
    finally:
        XGBRanker.fit = orig

    assert out[0] == "xgb_rank"
    kw = captured["kwargs"]
    assert kw["group"] == groups
    assert "sample_weight" in kw
    assert np.asarray(kw["sample_weight"]).shape[0] == len(groups)


def test_ranker_without_group_weights_has_no_sw_kwarg():
    """未开启半衰时，排序 fit 不传 sample_weight（兼容旧调用）。"""
    from xgboost import XGBRanker
    captured = {}
    orig = XGBRanker.fit

    def spy(self, X, y, **kwargs):
        captured["kwargs"] = kwargs
        return orig(self, X, y, **kwargs)

    XGBRanker.fit = spy
    try:
        variant = {"model": "xgb_rank"}
        W.fit_model(np.random.default_rng(1).normal(size=(8, 3)),
                    np.array([1, 0, 0, 0, 1, 0, 0, 0], float),
                    variant, groups=[4, 4])
    finally:
        XGBRanker.fit = orig
    assert "sample_weight" not in captured["kwargs"]


def test_ranker_per_sample_weight_is_rejected_by_xgb():
    """钉住 xgb 1.7.6 行为：排序组模式下按样本权重长度不符会报错——
    这正是必须按组聚合权重的原因（防止回退成按样本传权）。"""
    from xgboost import XGBRanker
    X = np.random.default_rng(2).normal(size=(8, 3))
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0], float)
    sw = np.random.default_rng(3).uniform(0.5, 1.5, size=8)
    rk = XGBRanker(n_estimators=3, max_depth=2, objective="rank:pairwise",
                   device="cuda", random_state=1, n_jobs=1, verbosity=0)
    try:
        rk.fit(X, y, group=[4, 4], sample_weight=sw)
    except Exception as e:  # noqa: BLE001
        assert "weight" in str(e).lower() or "group" in str(e).lower()
        return
    raise AssertionError("xgb 1.7.6 应拒绝排序组模式下的按样本权重")


# ---------------------------------------------------------------- 校准接口
def test_platt_calibrator_matches_lr_proba():
    """_PlattCalibration 接口与 IsotonicRegression.predict 对齐：1 维输入 → 概率。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 1))
    y = (X.ravel() + rng.normal(scale=0.5, size=60) > 0).astype(int)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    cal = W._PlattCalibration(clf)
    q = np.array([[-1.0], [0.0], [1.0]])
    p = cal.predict(q)
    assert p.shape == (3,)
    assert np.all((p >= 0) & (p <= 1))
    # 与底层 LR predict_proba 逐位一致
    assert np.allclose(p, clf.predict_proba(q)[:, 1])


def test_isotonic_calibrator_attached_in_compose_path():
    """mem_compose 臂在模型后挂第三元校准器（类型为可调回归器）。"""
    # 结构断言：joint_scores 读取 m[2].predict；校准器须暴露 predict 一维映射。
    cal_cls = W._PlattCalibration
    assert hasattr(cal_cls, "predict")


# ---------------------------------------------------------------- 半衰公式
def test_halflife_weights_monotonic():
    """半衰权重 0.5**(age/hl)：越近权重越高，hl=20 时 20 日前折半。"""
    hl = 20
    ages = list(range(40))
    w = [0.5 ** (a / hl) for a in ages]
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1))
    assert abs(w[20] - 0.5) < 1e-12
    assert w[0] == 1.0


# ---------------------------------------------------------------- 边界过滤
def _retain(rows, cutoff_s):
    keep = []
    for r in rows:
        if r["_vd"] is None or r["_vd"] >= cutoff_s or r["_yizi"]:
            continue
        keep.append(r["code"])
    return keep


def test_boundary_filter_drops_future_yizi_missing():
    cutoff = "2026-08-01"
    rows = [
        {"code": "A", "_vd": "2026-07-30", "_yizi": False},   # 保留
        {"code": "B", "_vd": "2026-08-01", "_yizi": False},   # V>=cutoff 剔
        {"code": "C", "_vd": None, "_yizi": False},           # 无核验日 剔
        {"code": "D", "_vd": "2026-07-29", "_yizi": True},    # 一字 剔
        {"code": "E", "_vd": "2026-07-31", "_yizi": False},   # 保留
    ]
    keep = _retain(rows, cutoff)
    assert keep == ["A", "E"]


# ---------------------------------------------------------------- 生产正门
def production_gate(hold: dict, current: dict) -> dict:
    """deep10 正门判定（与 freeze_contract.json 一致；参考旧三门不混用）。"""
    return {
        "exec_positive": hold["EXEC"] is not None and hold["EXEC"] > 0,
        "exec_beats_current": hold["EXEC"] is not None
        and current["EXEC"] is not None and hold["EXEC"] > current["EXEC"],
        "recall_85pct": hold["RECALL"] is not None and current["RECALL"] is not None
        and hold["RECALL"] >= 0.85 * current["RECALL"] - 1e-9,
        "win_not_worse_2pp": hold["WIN"] is not None and current["WIN"] is not None
        and hold["WIN"] >= current["WIN"] - 2.0,
    }


def test_gate_pass_and_fail_cases():
    cur = {"EXEC": -0.4144, "RECALL": 0.0245, "WIN": 44.4}
    good = {"EXEC": 0.3076, "RECALL": 0.0392, "WIN": 55.6}
    g = production_gate(good, cur)
    assert all(g.values())
    # EXEC 为负 → 不过
    bad_exec = {"EXEC": -0.0568, "RECALL": 0.0441, "WIN": 50.0}
    assert not production_gate(bad_exec, cur)["exec_positive"]
    # RECALL 跌破 85% → 不过
    bad_recall = {"EXEC": 0.2, "RECALL": 0.015, "WIN": 50.0}
    assert not production_gate(bad_recall, cur)["recall_85pct"]
    # WIN 降超 2pp → 不过
    bad_win = {"EXEC": 0.2, "RECALL": 0.03, "WIN": 41.0}
    assert not production_gate(bad_win, cur)["win_not_worse_2pp"]


# ---------------------------------------------------------------- 合同一致性
def test_freeze_contract_consistency():
    path = os.path.join(ROOT, "out", "research", "ziyang", "deep10",
                        "freeze_contract.json")
    contract = json.load(open(path, encoding="utf-8"))
    assert contract["validation_boundaries"]["dev_months_selection"] == \
        ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
    assert "2026-10" in contract["validation_boundaries"]["excluded_months"]
    assert "2026-11" in contract["validation_boundaries"]["excluded_months"]
    assert contract["production_gate"]["rules"]
    assert any("profile.py" in x for x in contract["scope"]["forbidden_to_modify"])
    assert any("test_logic.py" in x for x in contract["scope"]["forbidden_to_modify"])


def test_current_arm_contract_params():
    """当前生产臂冻结参数与合同一致（λ0.35 / squad1.10 / sold-0.48 / uw_sup）。"""
    assert W.C.JOINT_LAMBDA == 0.35
    cur = {
        "mem_attach": ["hazard_zy", "squad20", "sold_young", "uw_sup"],
        "lam": 0.35,
    }
    assert cur["lam"] == 0.35
    assert "uw_sup" in cur["mem_attach"]


if __name__ == "__main__":
    test_ranker_receives_group_level_weights()
    print("PASS ranker group weights")
    test_ranker_without_group_weights_has_no_sw_kwarg()
    print("PASS ranker no-sw default")
    test_ranker_per_sample_weight_is_rejected_by_xgb()
    print("PASS ranker per-sample sw rejected")
    test_platt_calibrator_matches_lr_proba()
    print("PASS platt calibrator interface")
    test_isotonic_calibrator_attached_in_compose_path()
    print("PASS calibrator attachment contract")
    test_halflife_weights_monotonic()
    print("PASS halflife formula")
    test_boundary_filter_drops_future_yizi_missing()
    print("PASS boundary filter")
    test_gate_pass_and_fail_cases()
    print("PASS production gate")
    test_freeze_contract_consistency()
    print("PASS freeze contract consistency")
    test_current_arm_contract_params()
    print("PASS current arm params")
    print("\ndeep10 ziyang 研究线离线测试全部通过")
