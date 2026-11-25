# -*- coding: utf-8 -*-
"""premium11 溢价影子模型区分力测试（离线合成 + tmp 工件；不写生产目录）。

覆盖合同硬边界：
- 目标是净收益回归（不是晋级标签）；
- 时序 purge/embargo：exit_date < 验证 signal_date；
- 旧概率边界：缺 old_promo_p / 非有限值不进训练；
- Ridge 排序方向；样本门 not_ready；
- 生产失败隔离：数据层缺失/异常不阻断旧 B1/B0 榜；
- minute20 / approx_open 物理隔离，proxy 醒目标 SCREENING；
- 工件 roundtrip 一致 + 坏件回退；影子 CSV 全历史幂等重建。

运行（项目根目录）:
    D:\\Miniconda3\\python.exe tests/test_vane_premium_model.py
或: D:\\Miniconda3\\python.exe -m pytest tests/test_vane_premium_model.py -q
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vane.config as VC  # noqa: E402
import vane.premium_model as PM  # noqa: E402

VALIDATION = "2026-11-20"
FEAT_DEFAULTS = {"B": 4, "span": 4, "breaks": 0, "seal_ratio": 0.3,
                 "fbt_min": 570, "log_close": 1.1, "log_ltsz": 1.5,
                 "mkt_height": 5, "mkt_two_plus": 10, "ind_zt_count": 2,
                 "ind_max_b": 5, "height_trend": 0.5}


def _rec(code, day, exit_day, net, old_p=0.6, layer="minute20", arm="U", **over):
    r = {"signal_date": day, "exit_date": exit_day, "code": code, "name": code,
         "B": 4, "layer": layer, "arm": arm, "entry_status": "filled",
         "exit_status": "exited", "net_ret": net,
         "old_promo_p": old_p, "unresolved_flag": 0, "complete": 1,
         "old_model_version": VC.MODEL_VERSION}
    r.update(FEAT_DEFAULTS)
    r.update(over)
    return r


def _days(n=12, start=dt.date(2026, 11, 2)):
    """n 个工作日序列（跳过周末），返回 (signal_iso, exit_iso) 列表。"""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            nxt = d + dt.timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += dt.timedelta(days=1)
            out.append((d.isoformat(), nxt.isoformat()))
        d += dt.timedelta(days=1)
    return out


def _make_records(n_days=12, per_day=4, layer="minute20", seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i, (sig, exi) in enumerate(_days(n_days)):
        for j in range(per_day):
            code = f"600{100 + i * 10 + j}"
            seal = 0.1 + 0.08 * j + rng.uniform(-0.02, 0.02)
            net = 0.02 * seal - 0.001 + rng.uniform(-0.002, 0.002)
            recs.append(_rec(code, sig, exi, round(float(net), 6),
                             seal_ratio=round(float(seal), 4), layer=layer))
    return recs


def _patch_paths(tmp):
    saved = {k: getattr(VC, k) for k in
             ("PREMIUM_MODEL_JSON", "PREMIUM_SHADOW_JSON", "PREMIUM_SHADOW_CSV")}
    VC.PREMIUM_MODEL_JSON = os.path.join(tmp, "premium_model.json")
    VC.PREMIUM_SHADOW_JSON = os.path.join(tmp, "shadow_daily.json")
    VC.PREMIUM_SHADOW_CSV = os.path.join(tmp, "shadow_daily.csv")
    return saved


def _restore_paths(saved):
    for k, v in saved.items():
        setattr(VC, k, v)


def _install_stub(records_by_layer=None, raise_load=False, sync_fn=None):
    mod = types.ModuleType("vane.premium_data")

    def load_labels(layer):
        if raise_load:
            raise RuntimeError("boom")
        return (records_by_layer or {}).get(layer, [])

    mod.record_signal = lambda *a, **k: None
    mod.collect_due = lambda *a, **k: None
    mod.load_universe = lambda d: []
    mod.load_labels = load_labels
    if sync_fn is not None:
        mod.sync_forward_ledger = sync_fn
    sys.modules["vane.premium_data"] = mod
    # 同步包属性：`from . import premium_data` 会优先取包属性，仅换 sys.modules
    # 会让 stub 被真实模块绕过（跨文件共跑时隔离失效）。
    import vane as _vane
    setattr(_vane, "premium_data", mod)
    return mod


def _uninstall_stub():
    sys.modules.pop("vane.premium_data", None)
    import vane as _vane
    if hasattr(_vane, "premium_data"):
        delattr(_vane, "premium_data")


def _today_rows(n=5):
    rows = []
    for j in range(n):
        r = {"code": f"6009{j:02d}", "name": f"票{j}", "B": 4,
             "model_score": float(j - 2), "seal_ratio": 0.2 + 0.1 * j,
             "fbt": "0931", "close": 12.0, "ltsz": 5e9, "industry": "测试"}
        r.update(FEAT_DEFAULTS)
        rows.append(r)
    return rows


# ---------- 1. 目标是净收益回归，不是晋级标签 ----------

def test_target_is_regression_not_promotion():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        recs = _make_records()
        art = PM.fit(recs, VALIDATION, kind="P1", layer="minute20")
        assert art["ok"], art
        assert art["features"] == list(VC.PREMIUM_FEATURES)
        assert "y" not in art["features"] and "promoted" not in art["features"]
        assert art["alpha"] == VC.PREMIUM_ALPHA == 10.0
        # 排序方向：高 seal（高净收益）行溢价分应更高
        rows = [dict(r) for r in recs[-4:]]
        scored = PM.score_rows(rows, art)
        order = [r["code"] for r in sorted(
            scored, key=lambda r: -r["premium_score"])]
        top = max(scored, key=lambda r: r["premium_score"])
        bottom = min(scored, key=lambda r: r["premium_score"])
        assert top["seal_ratio"] >= bottom["seal_ratio"]
        assert len(order) == 4
        print("PASS target_is_regression_not_promotion")
    finally:
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 2. 时序 purge / embargo 边界 ----------

def test_purge_embargo_boundary():
    recs = _make_records(n_days=12)
    # 边界样本：exit_date == 验证日、exit_date > 验证日，均必须剔除
    recs.append(_rec("600998", "2026-11-19", VALIDATION, 0.05,
                     exit_status="exited"))
    recs.append(_rec("600999", "2026-11-19", "2026-11-21", 0.05,
                     exit_status="exited"))
    train = PM._train_subset(recs, VALIDATION, "minute20")
    codes = {r["code"] for r in train}
    assert "600998" not in codes and "600999" not in codes
    # 正常前驱行保留
    assert len(train) >= 30
    print("PASS purge_embargo_boundary")


# ---------- 3. 旧概率边界 + 有限性校验 ----------

def test_old_promo_and_finite_boundary():
    recs = _make_records(n_days=12)
    # 缺 old_promo_p
    recs.append(_rec("600990", "2026-11-18", "2026-11-19", 0.03, old_p=None))
    # old_promo_p 非有限
    recs.append(_rec("600991", "2026-11-18", "2026-11-19", 0.03, old_p=np.nan))
    # 特征非有限
    recs.append(_rec("600992", "2026-11-18", "2026-11-19", 0.03,
                     seal_ratio=np.nan))
    train = PM._train_subset(recs, VALIDATION, "minute20")
    clean, dropped, reasons = PM._clean_train(train)
    codes = {r["code"] for r in clean}
    # 缺 old_promo_p 在子集阶段即剔除；非有限两行在 clean 阶段剔除
    assert "600990" not in {r["code"] for r in train}
    assert "600991" not in codes and "600992" not in codes
    assert dropped == 2
    assert reasons
    print("PASS old_promo_and_finite_boundary")


# ---------- 4. 已结算状态解析（entry filled/partial、exit exited、CSV 字符串） ----------

def test_settled_status_parsing():
    base = dict(signal_date="2026-11-18", exit_date="2026-11-19",
                entry_status="filled", exit_status="exited",
                net_ret=0.01, old_promo_p=0.5, complete=1)
    assert PM._is_settled({**base})
    assert PM._is_settled({**base, "entry_status": "partial"})
    assert not PM._is_settled({**base, "exit_status": "filled"})
    assert not PM._is_settled({**base, "entry_status": "unfilled"})
    assert not PM._is_settled({**base, "exit_status": "pending"})
    assert not PM._is_settled({**base, "unresolved_flag": "1"})
    assert not PM._is_settled({**base, "unresolved_flag": "True"})
    assert not PM._is_settled({**base, "complete": "0"})
    assert not PM._is_settled({**base, "complete": "false"})
    assert not PM._is_settled({**base, "net_ret": None})
    assert not PM._is_settled({**base, "net_ret": ""})
    assert PM._is_settled({**base, "complete": None})  # 未知不算显式假
    print("PASS settled_status_parsing")


# ---------- 5. 样本门 not_ready：A 臂为空，不冒充模型 ----------

def test_not_ready_gate():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        recs = _make_records(n_days=4, per_day=4)  # 4 日 / 16 行，双门不过
        art = PM.fit(recs, VALIDATION, layer="minute20")
        assert not art["ok"] and art["status"] == "insufficient_train"
        arms = PM.rank_arms(_today_rows(), None)
        assert arms["A_top3"] == [] and arms["A_picks"] == []
        assert arms["B1_top3"] and arms["B0_top3"]   # 旧臂/启发式照常
        assert arms["overlap_A_B1"] == 0
        print("PASS not_ready_gate")
    finally:
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 6. 生产失败隔离：数据层缺失/异常不阻断旧榜 ----------

def test_production_failure_isolation():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    orig_layer = PM._data_layer
    try:
        # 场景 1：数据层接缝缺失（模拟 premium_data 未交付）→ no_data_layer，旧榜照常
        PM._data_layer = lambda: None
        rec = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert rec["status"] == "no_data_layer"
        assert rec["A_top3"] == []
        assert rec["B1_top3"] and rec["B0_top3"]
        assert "no_data_layer" in rec["note"]

        # 场景 2：数据层抛错 → load_labels_error，A 空、B1/B0 不阻断
        _install_stub(raise_load=True)
        PM._data_layer = orig_layer
        rec2 = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert rec2["status"].startswith("load_labels_error")
        assert rec2["A_top3"] == [] and rec2["B1_top3"]
    finally:
        PM._data_layer = orig_layer
        _uninstall_stub()
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS production_failure_isolation")


# ---------- 7. 层隔离：minute20 / approx_open 不混训，proxy 标 SCREENING ----------

def test_layer_isolation_and_screening_flag():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        minute = _make_records(n_days=12, layer="minute20", seed=1)
        proxy = _make_records(n_days=12, layer="approx_open", seed=2)
        art_m = PM.fit(minute, VALIDATION, layer="minute20")
        art_p = PM.fit(proxy, VALIDATION, layer="approx_open")
        assert art_m["ok"] and art_p["ok"]
        assert art_m["layer"] == "minute20" and art_p["layer"] == "approx_open"
        # minute 工件不得被 approx_open 记录复用（层不兼容）
        assert not PM._artifact_ready(art_p) or art_p["layer"] != art_m["layer"]

        _install_stub({"approx_open": proxy})
        rec = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert rec["layer"] == "approx_open" and rec["screening"] is True
        assert "SCREENING" in rec["note"]
        # proxy 训练集里不得混入 minute20 行
        train_p = PM._train_subset(proxy + minute, VALIDATION, "approx_open")
        assert all(r["layer"] == "approx_open" for r in train_p)
    finally:
        _uninstall_stub()
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS layer_isolation_and_screening_flag")


# ---------- 8. 工件 roundtrip 一致 + 坏件回退 ----------

def test_artifact_roundtrip_and_fallback():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        recs = _make_records(n_days=12)
        art = PM.fit(recs, VALIDATION, kind="P1", layer="minute20")
        PM.save_artifact(art)
        loaded = PM.load_artifact()
        assert PM._artifact_ready(loaded)
        rows = _today_rows()
        s1 = PM.score_rows([dict(r) for r in rows], art)
        s2 = PM.score_rows([dict(r) for r in rows], loaded)
        assert np.allclose([r["premium_score"] for r in s1],
                           [r["premium_score"] for r in s2])

        # 坏件 JSON → load 返回 None → A 臂回退为空，不抛错
        with open(VC.PREMIUM_MODEL_JSON, "w", encoding="utf-8") as f:
            f.write("{corrupt")
        assert PM.load_artifact() is None
        arms = PM.rank_arms(_today_rows(), None)
        assert arms["A_top3"] == [] and arms["B1_top3"]
    finally:
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS artifact_roundtrip_and_fallback")


# ---------- 9. P2 截面去均值：y_mean 回加，分数有限 ----------

def test_p2_demean_fit():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        recs = _make_records(n_days=12)
        art = PM.fit(recs, VALIDATION, kind="P2", layer="minute20")
        assert art["ok"] and art["kind"] == "P2"
        assert abs(art["y_mean"] - np.mean([r["net_ret"] for r in recs])) < 1e-6
        scored = PM.score_rows(_today_rows(), art)
        assert all(np.isfinite(r["premium_score"]) for r in scored)
    finally:
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS p2_demean_fit")


# ---------- 10. 影子 CSV 全历史幂等重建 ----------

def test_shadow_csv_history_rebuild():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    try:
        _install_stub({"minute20": _make_records(n_days=12)})
        d1 = dt.date(2026, 11, 19)
        d2 = dt.date(2026, 11, 20)
        PM.daily_shadow(d1, _today_rows(), None)
        PM.daily_shadow(d2, _today_rows(), None)
        import csv as _csv
        with open(VC.PREMIUM_SHADOW_CSV, encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
        days = {r["signal_date"] for r in rows}
        assert days == {"2026-11-19", "2026-11-20"}
        arms = {(r["signal_date"], r["arm"]) for r in rows}
        assert ("2026-11-19", "B1") in arms and ("2026-11-20", "B0") in arms
        # 当日重跑覆写不产生重复
        PM.daily_shadow(d2, _today_rows(), None)
        with open(VC.PREMIUM_SHADOW_CSV, encoding="utf-8-sig") as f:
            rows2 = list(_csv.DictReader(f))
        assert len(rows2) == len(rows)
    finally:
        _uninstall_stub()
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS shadow_csv_history_rebuild")


def test_report_render_shadow_block():
    """影子区块随 render 落盘：SCREENING 标识与三臂代码可见；无结果整节省略。"""
    tmp = tempfile.mkdtemp()
    saved_paths = _patch_paths(tmp)
    saved_report = VC.REPORT_TXT
    VC.REPORT_TXT = os.path.join(tmp, "report.txt")
    try:
        from vane import report as VR
        ps = {"status": "insufficient_train", "layer": "approx_open",
              "screening": True, "A_top3": [], "B1_top3": ["600901"],
              "B0_top3": ["600902"], "overlap_A_B1": 0, "B2_n": 5}
        txt = VR.render({"trade_date": "2026-11-20", "picks": [], "bench": [],
                         "premium_shadow": ps})
        assert "溢价影子" in txt and "SCREENING" in txt
        assert "600901" in txt and "600902" in txt
        txt2 = VR.render({"trade_date": "2026-11-20", "picks": [], "bench": []})
        assert "溢价影子" not in txt2
    finally:
        VC.REPORT_TXT = saved_report
        _restore_paths(saved_paths)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS report_render_shadow_block")


# ---------- arm 过滤：训练只用全宇宙 U 行，执行臂复制行不重复训练 ----------

def test_train_only_universe_arm():
    days = _days(12)
    rows = []
    for i, (sig, exi) in enumerate(days):
        for j in range(4):
            rows.append(_rec(f"600{200 + i * 10 + j}", sig, exi, 0.01, arm="U"))
        code = f"600{200 + i * 10}"
        for arm in ("A", "B1", "B0", "B2"):
            rows.append(_rec(code, sig, exi, 0.02, arm=arm))
    subset = PM._train_subset(rows, VALIDATION, "minute20")
    assert len(subset) == 48
    assert all(r["arm"] == "U" for r in subset)
    art = PM.fit(rows, VALIDATION, layer="minute20")
    assert art["ok"] and art["n_rows"] == 48
    print("PASS train_only_universe_arm")


# ---------- sync_forward_ledger：成功触发 / 失败隔离留痕 / 缺接口静默 ----------

def test_sync_forward_ledger_isolation():
    tmp = tempfile.mkdtemp()
    saved = _patch_paths(tmp)
    calls = []

    def sync_ok():
        calls.append(1)

    try:
        _install_stub({}, sync_fn=sync_ok)
        rec = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert len(calls) == 1                      # 台账落盘后触发同步
        assert "sync_forward_ledger" not in rec["note"]

        def sync_boom():
            raise RuntimeError("ledger boom")

        _install_stub({}, sync_fn=sync_boom)
        rec2 = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert "sync_forward_ledger:RuntimeError" in rec2["note"]
        assert rec2["B1_top3"] and rec2["B0_top3"]  # 旧臂对照不阻断

        _install_stub({})                           # 数据层无该接口
        rec3 = PM.daily_shadow(dt.date(2026, 11, 20), _today_rows(), None)
        assert "sync_forward_ledger" not in rec3["note"]
    finally:
        _uninstall_stub()
        _restore_paths(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS sync_forward_ledger_isolation")


def main():
    test_target_is_regression_not_promotion()
    test_purge_embargo_boundary()
    test_old_promo_and_finite_boundary()
    test_settled_status_parsing()
    test_not_ready_gate()
    test_production_failure_isolation()
    test_layer_isolation_and_screening_flag()
    test_artifact_roundtrip_and_fallback()
    test_p2_demean_fit()
    test_shadow_csv_history_rebuild()
    test_report_render_shadow_block()
    test_train_only_universe_arm()
    test_sync_forward_ledger_isolation()
    print("\npremium11 溢价影子模型测试全部通过")


if __name__ == "__main__":
    main()
