# -*- coding: utf-8 -*-
"""deep10 vane 线研究测试：冻结快照复现、工件一致性、refit meta 一致性回归、
校准单调性与时序边界。离线合成数据 + 只读真实部署工件；不写生产目录。

运行（项目根目录）:
    py -3.11 tests/test_vane_research.py
或: py -3.11 -m pytest tests/test_vane_research.py -q
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEEP10 = os.path.join(ROOT, "out", "research", "vane", "deep10")

# 显式按文件路径加载本线 common，使用唯一模块名；避免与画像线
# out/research/profile/deep10/common.py 在同一 pytest 会话里因裸模块名
# `common` 串用 sys.modules 缓存（先导入者占名，后到者取错模块）。
_spec = importlib.util.spec_from_file_location(
    "vane_deep10_common", os.path.join(DEEP10, "common.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
sys.modules["vane_deep10_common"] = common

import vane.config as VC  # noqa: E402
import vane.model as VM  # noqa: E402
import vane.screen as VS  # noqa: E402
import data_source  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "out", "research", "vane", "model")


# ---------- 合成环境夹具（独立于 test_logic.py，不引用其私有函数） ----------

def _raw(code, name, lb=1, amt=8e7):
    return {"c": code, "n": name, "p": 12340, "zdp": 9.97, "amount": amt,
            "hs": 500.0, "lbc": lb, "fbt": 93100, "fund": 1.6e8,
            "hybk": "测试业", "ltsz": 5e9}


def _make_env():
    """返回 (tmp, dates, pools_ymd, restore)。补丁 vane/screen 的数据源与路径到 tmp。"""
    tmp = tempfile.mkdtemp()
    dates = ([dt.date(2026, 11, x) for x in range(2, 7)]
             + [dt.date(2026, 11, x) for x in range(9, 15)])
    iso = [d.isoformat() for d in dates]
    pools = {}
    for i, d in enumerate(iso):
        rows = [_raw("600001", "甲一", min(i + 1, 6)),
                _raw("600002", "乙二", min(max(i - 2, 1), 5)),
                _raw("600003", "丙三", min(max(i - 4, 1), 4))]
        if i % 2 == 0:
            rows.append(_raw("600004", "丁四", 2))
        pools[d] = rows
    pools_ymd = {k.replace("-", ""): v for k, v in pools.items()}

    orig = {
        "pool_raw": VM.pool_raw, "parse_pool": VM.parse_pool,
        "_window_days": VM._window_days, "xgb_ready": VM._xgb_ready,
        "vs_pool_raw": VS.pool_raw, "vs_parse_pool": VS.parse_pool,
        "vs_cal": VS.trade_calendar,
    }
    saved_cfg = {k: getattr(VC, k) for k in (
        "DATA_DIR", "MODEL_DIR", "MODEL_JSON", "MODEL_LGR_JSON",
        "MODEL_XGB_JSON", "MODEL_PREV_JSON", "ISO_JSON",
        "RETRAIN_DIFF_JSON", "LEDGER_CSV", "SAMPLES_MIN_ROWS",
        "SAMPLES_MIN_POS")}

    VM.pool_raw = lambda k, d: pools_ymd.get(d) if k == "zt" else []
    VM.parse_pool = data_source.parse_pool
    VM._window_days = lambda td, w=VC.WINDOW: (dates, True)
    VM._xgb_ready = lambda: False  # 测试强制 logreg 路径
    VS.pool_raw = VM.pool_raw
    VS.parse_pool = VM.parse_pool
    VS.trade_calendar = lambda: dates

    VC.DATA_DIR = tmp
    VC.MODEL_DIR = tmp
    VC.SAMPLES_MIN_ROWS = 8
    VC.SAMPLES_MIN_POS = 2
    for k in ("MODEL_JSON", "MODEL_LGR_JSON", "MODEL_XGB_JSON",
              "MODEL_PREV_JSON", "ISO_JSON", "RETRAIN_DIFF_JSON",
              "LEDGER_CSV"):
        setattr(VC, k, os.path.join(tmp, k))

    def restore():
        VM.pool_raw, VM.parse_pool, VM._window_days = (
            orig["pool_raw"], orig["parse_pool"], orig["_window_days"])
        VM._xgb_ready = orig["xgb_ready"]
        VS.pool_raw, VS.parse_pool, VS.trade_calendar = (
            orig["vs_pool_raw"], orig["vs_parse_pool"], orig["vs_cal"])
        for k, v in saved_cfg.items():
            setattr(VC, k, v)
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, dates, pools_ymd, restore


# ---------- 0. 模块防撞回归 ----------

def test_common_bound_to_vane_deep10():
    """加载的 common 必须来自 vane deep10 目录，而非画像线同名模块。"""
    assert os.path.dirname(os.path.abspath(common.__file__)) == os.path.abspath(DEEP10)


# ---------- 1. 冻结快照与时序折型 ----------

def test_snapshot_frozen_and_folds():
    df = common.load_snapshot()
    assert len(df) == 200 and int(df.y.sum()) == 58 and df.date.nunique() == 13
    plan = common.fold_plan(df)
    assert [(len(t), len(e)) for _, t, e in plan] == [(6, 4), (9, 4)]
    for _, tr, te in plan:
        assert all(t < e for t in tr for e in te)  # 训练日严格早于验证日
    overlap = set(plan[0][2]) & set(plan[1][2])
    assert overlap == {"2026-11-18"}
    unique = set(plan[0][2]) | set(plan[1][2])
    assert len(unique) == 7
    print("PASS snapshot_frozen_and_folds")


# ---------- 2. 冻结快照复现锚点（logreg 路径确定性强，须近全等） ----------

def test_reproduce_logreg_anchors():
    df = common.load_snapshot()
    rows = df.to_dict("records")
    r0 = common.evaluate(rows, {"kind": "logreg", "C": 0.5,
                                "calibration": "platt"})
    assert r0["pooled"]["top3"] == 0.4286
    assert r0["pooled"]["logloss"] == 0.7587
    assert r0["pooled"]["unique_ratio"] == 1.0
    r1 = common.evaluate(rows, {"kind": "logreg", "C": 1.0,
                                "calibration": "platt"})
    assert r1["pooled"]["top3"] == 0.4286
    assert r1["pooled"]["logloss"] == 0.7779
    r3 = common.evaluate(rows, {"kind": "logreg", "C": 0.5,
                                "calibration": "iso"})
    # 小样本 isotonic 并列退化：去重率显著下降（与 sources.md A1 一致）
    assert r3["pooled"]["unique_ratio"] < 0.5
    print("PASS reproduce_logreg_anchors")


# ---------- 3. 真实部署工件加载/评分/读写一致性（只读） ----------

def test_deployed_artifact_consistency():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.json")):
        print("SKIP deployed artifact 缺失")
        return
    meta = json.loads(open(os.path.join(MODEL_DIR, "model.json"),
                           encoding="utf-8").read())
    lgr = json.loads(open(os.path.join(MODEL_DIR, "model_lgr.json"),
                          encoding="utf-8").read())
    iso = json.loads(open(os.path.join(MODEL_DIR, "iso.json"),
                          encoding="utf-8").read())
    assert meta["version"] == VC.MODEL_VERSION
    assert meta["kind"] == "logreg" and meta["calibration"] == "platt"
    assert meta["features"] == list(VC.FEATURES)
    assert meta["coef"] == lgr["coef"]
    assert meta["intercept"] == lgr["intercept"]
    assert np.allclose(meta["mu"], lgr["mu"])
    assert np.allclose(meta["sd"], lgr["sd"])
    assert meta["medians"] == lgr["medians"]

    df = common.load_snapshot()
    X, _ = VM._matrix(df.to_dict("records"), medians=meta.get("medians"))
    p = VM._predict_proba(meta, os.path.join(MODEL_DIR, "model_lgr.json"), X)
    q = VM._iso_apply(p, iso)
    p2 = VM._predict_proba(meta, os.path.join(MODEL_DIR, "model_lgr.json"), X)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert np.allclose(p, p2)  # 确定性复跑
    assert len(np.unique(np.round(q, 8))) == len(q)
    # 手动复算一致性
    z = (X - np.array(lgr["mu"], float)) / np.array(lgr["sd"], float)
    raw = 1 / (1 + np.exp(-(z @ np.array(lgr["coef"], float)
                             + lgr["intercept"])))
    assert np.allclose(p, raw, atol=1e-6)
    print("PASS deployed_artifact_consistency")


# ---------- 4. refit 后 MODEL_JSON 与 MODEL_LGR_JSON 一致性（回归缺陷） ----------

def test_refit_meta_consistency_regression():
    _, dates, _, restore = _make_env()
    try:
        today = dates[-1]
        m1 = VM.train(today)
        assert m1["ok"]
        # 第二套新鲜样本：翻转 600001 标签，制造与首训不同的模型参数
        fresh_samples, fresh_days, miss = VM.build_samples(today)
        for s in fresh_samples:
            if s["code"] == "600001":
                s["y"] = 1 - s["y"]
        orig_bs = VM.build_samples
        VM.build_samples = lambda *a, **k: (fresh_samples, fresh_days, miss)
        msg = VM.refit_today(today)
        VM.build_samples = orig_bs
        assert "完成" in msg, msg

        meta = json.loads(open(VC.MODEL_JSON, encoding="utf-8").read())
        lgr = json.loads(open(VC.MODEL_LGR_JSON, encoding="utf-8").read())
        # 修复后：meta 必须反映新鲜重算结果，与 lgr 工件一致
        assert meta["coef"] == lgr["coef"]
        assert meta["intercept"] == lgr["intercept"]
        assert meta["n_rows"] == len(fresh_samples)
        assert bool(meta.get("refit_at")) and meta.get("finetuned") is True
    finally:
        restore()
    print("PASS refit_meta_consistency_regression")


# ---------- 5. 校准层单调性与 iso 兼容 ----------

def test_platt_monotonic_and_iso_compat():
    rng = np.linspace(0.05, 0.95, 25)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1,
                  0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    platt = VM._platt_fit(rng, y)
    q = VM._iso_apply(rng, platt)
    diffs = np.diff(q)
    assert (diffs >= -1e-12).all()  # Platt 严格单调，保持排序
    legacy = {"x": [0.0, 1.0], "y": [0.2, 0.8]}
    q2 = VM._iso_apply(rng, legacy)
    assert len(q2) == len(rng) and np.isfinite(q2).all()
    print("PASS platt_monotonic_and_iso_compat")


# ---------- 6. 时序边界：标准化/中位数仅来自训练折 ----------

def test_no_leakage_standardization():
    df = common.load_snapshot()
    rows = df.to_dict("records")
    res = common.evaluate(rows, {"kind": "logreg", "C": 0.5,
                                 "calibration": "platt"})
    plan = common.fold_plan(df)
    for fm, (_, tr_days, _) in zip(res["folds"], plan):
        train = [r for r in rows if r["date"] in set(tr_days)]
        assert fm["train_rows"] == len(train)
        assert fm["train_pos"] == sum(r["y"] for r in train)
        assert not fm["constant_train"] and not fm["constant_test"]
    # 校验 R0 折内标准化可复算（直接对 fold1 训练折比对 common 拟合产物）
    tr = [r for r in rows if r["date"] in set(plan[0][1])]
    art = common.fit_base(tr, "logreg", C=0.5, calibration="platt")
    Xtr, _ = VM._matrix(tr)
    assert np.allclose(art["mu"], Xtr.mean(axis=0))
    assert np.allclose(art["sd"], np.where(Xtr.std(axis=0) > 1e-9,
                                            Xtr.std(axis=0), 1.0))
    print("PASS no_leakage_standardization")


def main():
    test_common_bound_to_vane_deep10()
    test_snapshot_frozen_and_folds()
    test_reproduce_logreg_anchors()
    test_deployed_artifact_consistency()
    test_refit_meta_consistency_regression()
    test_platt_monotonic_and_iso_compat()
    test_no_leakage_standardization()
    print("\ndeep10 vane 研究测试全部通过")


if __name__ == "__main__":
    main()
