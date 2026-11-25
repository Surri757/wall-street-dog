# -*- coding: utf-8 -*-
"""premium11 验证门测试：HAC 已知例子、严格解析、资金口径、各过门/失败场景、
proxy 数据可行性。离线合成数据 + 只读归档；不写生产目录、不训练。

运行（项目根目录）:
    D:\\Miniconda3\\python.exe tests/test_vane_premium_gate.py
或: D:\\Miniconda3\\python.exe -m pytest tests/test_vane_premium_gate.py -q
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vane import premium_validation as pv  # noqa: E402

PREMIUM11 = os.path.join(ROOT, "out", "research", "vane", "premium11")
_spec = importlib.util.spec_from_file_location(
    "premium11_gate", os.path.join(PREMIUM11, "gate.py"))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


# ── 夹具 ────────────────────────────────────────────────────────────────────

def _rec(signal_date, arm, rank, code, net, *, entry="filled", exit_status="exited",
         layer=pv.LAYER_MINUTE20, regime="high", filled_ratio=1.0,
         unresolved=False, old_promo_p=None):
    return {
        "signal_date": signal_date, "exec_date": None, "exit_date": None,
        "arm": arm, "rank": rank, "code": code, "name": code, "B": 4,
        "model_version": "test", "model_score": None,
        "old_model_version": None, "old_promo_p": old_promo_p,
        "layer": layer, "entry_status": entry, "filled_ratio": filled_ratio,
        "buy_vwap": None, "exit_status": exit_status, "sell_vwap_20m": None,
        "gross_ret": None, "net_ret": net, "weight": None, "cash_weight": None,
        "carry_flag": False, "unresolved_flag": unresolved,
        "data_source": "test", "fetched_at": None, "tz": "Asia/Shanghai",
        "adj": "raw", "complete": True, "note": "", "regime": regime,
    }


def _dates(n=21):
    sep = [f"2026-09-{10 + i:02d}" for i in range(min(11, n))]
    oct_n = max(0, n - len(sep))
    oct_ = [f"2026-10-{8 + i:02d}" for i in range(oct_n)]
    return sep + oct_


def _pass_scenario(layer=pv.LAYER_MINUTE20, a_base=0.004, b1_base=0.001,
                   b2_base=0.0005):
    """三臂配对合成场景：A 稳定小胜基线，各硬门可达标。"""
    records = []
    for i, d in enumerate(_dates()):
        regime = "high" if i % 2 == 0 else "low"
        for j, net in enumerate((a_base + 0.002, a_base, a_base - 0.002)):
            records.append(_rec(d, "A", j + 1, f"60000{j}", net, regime=regime))
        for j in range(3):
            records.append(_rec(d, "B1", j + 1, f"00200{j}", b1_base, regime=regime))
        for j in range(4):
            records.append(_rec(d, "B2", j + 1, f"00300{j}", b2_base, regime=regime))
    return records


# ── 1. HAC：常数序列 / 独立复算 / lag 公式 ─────────────────────────────────

def test_hac_constant_series():
    r = pv.nw_hac_mean_ci([0.01] * 20)
    assert r["n"] == 20 and abs(r["mean"] - 0.01) < 1e-12
    assert r["se"] < 1e-12
    assert abs(r["ci_low"] - 0.01) < 1e-10 and abs(r["ci_high"] - 0.01) < 1e-10
    print("PASS hac_constant_series")


def test_hac_lag_formula():
    assert pv.hac_lag(1) == 0
    assert pv.hac_lag(5) == 1
    assert pv.hac_lag(25) == 2
    assert pv.hac_lag(100) == 4
    print("PASS hac_lag_formula")


def test_hac_matches_manual_independent():
    x = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, 0.008, -0.004, 0.011,
         -0.007, 0.006, 0.003, -0.012, 0.009, 0.002, -0.005, 0.007, -0.001, 0.004]
    n = len(x)
    mean = sum(x) / n
    lag = max(1, int(math.floor(n ** (1 / 3))))
    e = [v - mean for v in x]
    omega = sum(v * v for v in e) / n
    for l in range(1, lag + 1):
        gamma = sum(e[i] * e[i + l] for i in range(n - l)) / n
        omega += 2 * (1 - l / (lag + 1)) * gamma
    se = math.sqrt(max(omega, 0.0) / n)

    r = pv.nw_hac_mean_ci(x)
    assert r["n"] == n and r["lag"] == lag
    assert abs(r["mean"] - mean) < 1e-15
    assert abs(r["se"] - se) < 1e-15
    assert abs(r["ci_low"] - (mean - 1.96 * se)) < 1e-12
    print("PASS hac_matches_manual_independent")


def test_hac_empty_and_single():
    assert pv.nw_hac_mean_ci([])["n"] == 0
    one = pv.nw_hac_mean_ci([0.02])
    assert abs(one["mean"] - 0.02) < 1e-12 and one["se"] == 0.0
    print("PASS hac_empty_and_single")


# ── 2. 严格解析与状态归一 ──────────────────────────────────────────────────

def test_normalize_string_types_and_aliases():
    raw = {"entry_status": "unfilled_one_word", "exit_status": "exited",
           "net_ret": "0.012", "unresolved_flag": "0", "filled_ratio": "1",
           "B": "4"}
    n = pv.normalize_record(raw)
    assert n["entry_status"] == "unfilled"
    assert n["net_ret"] == 0.012
    assert n["unresolved_flag"] is False
    assert n["filled_ratio"] == 1.0
    assert n["B"] == 4.0
    assert pv.is_settled(n) is False  # 未成交入场不算已结算

    raw2 = {"entry_status": "filled", "exit_status": "pending_limit_down",
            "net_ret": "0.01", "unresolved_flag": "1"}
    n2 = pv.normalize_record(raw2)
    assert n2["exit_status"] == "pending"
    assert pv.is_unresolved(n2) is True
    assert pv.is_settled(n2) is False
    print("PASS normalize_string_types_and_aliases")


def test_bool_string_zero_is_false():
    """回归：CSV '0' 不能被当 True（旧实现裸判真值导致全部误判未决）。"""
    n = pv.normalize_record({"unresolved_flag": "0", "carry_flag": "false"})
    assert n["unresolved_flag"] is False and n["carry_flag"] is False
    print("PASS bool_string_zero_is_false")


# ── 3. 资金口径：留现不重分配 / B2 等权 ───────────────────────────────────

def test_cash_weight_no_redistribution():
    records = [
        _rec("2026-10-01", "A", 1, "600001", 0.03),
        _rec("2026-10-01", "A", 2, "600002", None, entry="unfilled"),
        _rec("2026-10-01", "A", 3, "600003", -0.01),
    ]
    day = pv.cohort_day(records, "A")
    # 成交两席各 1/3；未成交席留现，不把 1/3 重分配给其余两席
    assert abs(day["day_ret"] - (0.03 - 0.01) / 3) < 1e-12
    assert abs(day["cash_weight"] - 1 / 3) < 1e-12
    assert day["n_settled"] == 2
    print("PASS cash_weight_no_redistribution")


def test_b0_fixed_seat_cash_no_redistribution():
    """B0 同为 Top3 固定席位：未成交席留现 1/3，成交席仍各 1/3，不重分配。"""
    records = [
        _rec("2026-10-01", "B0", 1, "600101", 0.03),
        _rec("2026-10-01", "B0", 2, "600102", None, entry="unfilled"),
        _rec("2026-10-01", "B0", 3, "600103", -0.01),
    ]
    day = pv.cohort_day(records, "B0")
    assert abs(day["day_ret"] - (0.03 - 0.01) / 3) < 1e-12
    assert abs(day["cash_weight"] - 1 / 3) < 1e-12
    assert day["n_settled"] == 2
    # 未成交席组合贡献必须为 0，不把席位资金并入成交票
    unfilled = [r for r in records if r["entry_status"] == "unfilled"]
    assert pv.is_settled(unfilled[0]) is False
    print("PASS b0_fixed_seat_cash_no_redistribution")


def test_unfilled_contributes_zero_no_redistribution():
    """仅一只成交、两只未成交：成交票不得按 1/2 或全仓重分配。"""
    records = [
        _rec("2026-10-02", "A", 1, "600201", 0.06),
        _rec("2026-10-02", "A", 2, "600202", None, entry="unfilled"),
        _rec("2026-10-02", "A", 3, "600203", None, entry="unfilled"),
    ]
    day = pv.cohort_day(records, "A")
    assert abs(day["day_ret"] - 0.06 / 3) < 1e-12
    assert abs(day["cash_weight"] - 2 / 3) < 1e-12
    print("PASS unfilled_contributes_zero_no_redistribution")


def test_b2_equal_weight_executable():
    records = [_rec("2026-10-01", "B2", j + 1, f"00300{j}", 0.006)
               for j in range(4)]
    day = pv.cohort_day(records, "B2")
    assert abs(day["day_ret"] - 0.006) < 1e-12
    assert day["cash_weight"] == 0.0
    print("PASS b2_equal_weight_executable")


# ── 4. 指标回归：MDD 正值 / 剔最佳 3 笔 ───────────────────────────────────

def test_mdd_positive_magnitude():
    nav, mdd = pv.compound_nav([0.10, -0.20, 0.05])
    # 1.10 → 0.88 → 0.924；回撤幅度 = 1 − 0.88/1.10 = 0.20
    assert abs(mdd - 0.20) < 1e-12
    assert mdd >= 0
    print("PASS mdd_positive_magnitude")


def test_concentration_drop_largest_three():
    nets = [0.10, 0.09, 0.08, 0.01, -0.02]
    c = pv.concentration(nets)
    gross = 0.10 + 0.09 + 0.08 + 0.01
    assert abs(c["gross_profit"] - gross) < 1e-12
    assert abs(c["top3_profit_share"] - (0.27 / gross)) < 1e-12
    # 剔最大 3 笔（0.10/0.09/0.08），剩余 [0.01, -0.02] 均值 −0.005
    assert abs(c["drop_best3_mean"] - (-0.005)) < 1e-12
    print("PASS concentration_drop_largest_three")


# ── 5. 部署门场景 ──────────────────────────────────────────────────────────

def _contract_thresholds():
    return gate.load_contract()


def test_pass_scenario_minute20_promotes():
    records = _pass_scenario()
    report = pv.evaluate_gates(records, _contract_thresholds(),
                               pv.LAYER_MINUTE20)
    assert report["promote"] is True, report["gates"]
    assert report["decision"] == "promote"
    for g in ("sample_gate", "profit", "beats_baselines", "cross_period",
              "concentration", "risk", "fold_noninferiority", "cost_stress"):
        assert report["gates"][g]["pass"], (g, report["gates"][g]["reason"])
    assert report["sample"]["n_settled_trips"] >= 30
    assert report["sample"]["n_regimes"] == 2
    print("PASS pass_scenario_minute20_promotes")


def test_approx_layer_never_promotes_even_with_good_numbers():
    records = _pass_scenario(layer=pv.LAYER_APPROX)
    report = pv.evaluate_gates(records, _contract_thresholds(),
                               pv.LAYER_APPROX)
    assert report["promote"] is False
    assert report["decision"] == "screening_only"
    assert report["gates"]["sample_gate"]["pass"] is False
    print("PASS approx_layer_never_promotes_even_with_good_numbers")


def test_unresolved_cohort_excluded_and_blocks_promotion():
    records = _pass_scenario()
    # 末一日 A 一席顺延未决 → 该 cohort 剔除，且存在未决记录
    last = records[-1]
    records.append(_rec(last["signal_date"], "A", 4, "600099", None,
                        entry="filled", exit_status="pending", unresolved=True))
    report = pv.evaluate_gates(records, _contract_thresholds(),
                               pv.LAYER_MINUTE20)
    assert report["promote"] is False
    assert any(e["reason"] == "unresolved_position" for e in report["excluded_cohorts"])
    assert report["sample"]["unresolved_records"] >= 1
    print("PASS unresolved_cohort_excluded_and_blocks_promotion")


def test_insufficient_sample_blocks_promotion():
    short = _pass_scenario()[:0]
    dates = _dates(10)
    for d in dates:
        for arm, base in (("A", 0.004), ("B1", 0.001), ("B2", 0.0005)):
            n_slots = 3 if arm != "B2" else 4
            for j in range(n_slots):
                short.append(_rec(d, arm, j + 1, f"{arm}{j}{d[-2:]}", base))
    report = pv.evaluate_gates(short, _contract_thresholds(), pv.LAYER_MINUTE20)
    assert report["promote"] is False
    assert report["gates"]["sample_gate"]["pass"] is False
    print("PASS insufficient_sample_blocks_promotion")


def test_beats_baselines_fails_when_a_loses():
    records = _pass_scenario(a_base=0.0005, b1_base=0.003, b2_base=0.002)
    report = pv.evaluate_gates(records, _contract_thresholds(),
                               pv.LAYER_MINUTE20)
    assert report["gates"]["beats_baselines"]["pass"] is False
    assert report["promote"] is False
    print("PASS beats_baselines_fails_when_a_loses")


def test_risk_gate_flags_deep_drawdown():
    records = []
    for i, d in enumerate(_dates()):
        net = -0.06 if i == 5 else 0.004
        for j in range(3):
            records.append(_rec(d, "A", j + 1, f"60000{j}", net))
        for j in range(3):
            records.append(_rec(d, "B1", j + 1, f"00200{j}", 0.001))
        for j in range(4):
            records.append(_rec(d, "B2", j + 1, f"00300{j}", 0.0005))
    report = pv.evaluate_gates(records, _contract_thresholds(),
                               pv.LAYER_MINUTE20)
    assert report["gates"]["risk"]["pass"] is False
    print("PASS risk_gate_flags_deep_drawdown")


# ── 6. CSV 台账解析（字符串字段走 normalize） ──────────────────────────────

def test_load_records_csv_coerces_types():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "ledger.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal_date", "arm", "code", "entry_status", "exit_status",
                     "net_ret", "unresolved_flag", "layer"])
        w.writerow(["2026-10-01", "A", "600001", "filled", "exited", "0.015", "0",
                     "minute20"])
        w.writerow(["2026-10-01", "A", "600002", "unfilled_one_word", "exited",
                    "", "0", "minute20"])
    rows = gate.load_records(path)
    assert len(rows) == 2
    assert rows[0]["net_ret"] == 0.015
    assert rows[0]["unresolved_flag"] is False
    assert rows[1]["entry_status"] == "unfilled"
    assert rows[1]["net_ret"] is None
    print("PASS load_records_csv_coerces_types")


# ── 7. proxy 合同宇宙重建与归档只读核对（归档缺失则 skip） ──────────────────

def test_proxy_archive_coverage_readonly():
    archive = os.path.join(ROOT, "out", "archive")
    if not os.path.isdir(archive):
        print("SKIP proxy archive 缺失")
        return
    spec2 = importlib.util.spec_from_file_location(
        "premium11_proxy", os.path.join(PREMIUM11, "proxy_screening.py"))
    proxy = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(proxy)
    signals, schema = proxy.load_signals()
    assert len(signals) > 0 and len(schema) > 0
    bars = proxy._load_pkl(proxy.BARS_PKL)
    cal = proxy.load_calendar()

    events, excluded, _samples, win_excl = proxy.reconstruct_events(
        signals, bars, cal)
    # 合同宇宙：重建 B ∈ {4,5}，近似层，旧概率留空不编造
    assert all(e["B"] in (4, 5) for e in events)
    assert all(e["layer"] == pv.LAYER_APPROX for e in events)
    assert all(e["old_promo_p"] is None for e in events)
    # 合同硬约束：approx 层日 K 开盘只进 exec_open/exit_open，
    # VWAP 成交价字段必须 None；臂统一 U，不得再造 UNIVERSE 臂名。
    assert all(e["arm"] == "U" for e in events)
    assert not any(e["arm"] == "UNIVERSE" for e in events)
    assert all(e["buy_vwap"] is None and e["sell_vwap_20m"] is None
               for e in events)
    for e in events:
        if e["exec_open"] is not None:
            assert e["exec_open"] > 0
            assert e["exit_open"] is not None
        assert "_sort" not in e
    executable = [e for e in events if e["net_ret"] is not None]
    assert len(executable) > 0

    # 窗口缺池日整组排除；排除原因分类落账
    assert excluded.get("window_incomplete", 0) > 0
    assert len(win_excl) > 0
    assert all(w["missing_pool_days"] for w in win_excl)

    # 合同宇宙内近似基线：B0 启发式 Top3 / B2 全体等权；B1 历史不可得
    bases = proxy.baseline_summaries(events)
    assert bases["B0"]["status"] == "proxy_heuristic_top3"
    assert bases["B2"]["status"] == "proxy_universe_equal_weight"
    assert bases["B1"]["status"] == "unavailable"

    # 旧 lianban 口径仅描述性对照，不得作为合同基线
    legacy = proxy.legacy_descriptive(signals, bars, cal)
    assert legacy["contract_baseline_eligible"] is False
    assert legacy["rows"] > len(events)
    print(f"PASS proxy_archive_coverage_readonly (recon={len(events)}, "
          f"executable={len(executable)}, legacy={legacy['rows']})")


# ── 8. 前向台账集成：U 训练层排除 / A/B1/B2 进门控 / 缺臂无配对 ──────────────

def _forward_rows(dates, arms=("A", "B1", "B2"), bases=None):
    bases = bases or {"A": 0.004, "B1": 0.001, "B2": 0.0005}
    rows = []
    for d in dates:
        for arm in arms:
            slots = 3 if arm != "B2" else 4
            for j in range(slots):
                rows.append(_rec(d, arm, j + 1, f"{arm}{j}{d[-2:]}",
                                  bases[arm], layer=pv.LAYER_MINUTE20))
    return rows


def test_train_arm_u_excluded_from_gate():
    """U 训练层宇宙行不得被当作 A 臂；剔除后无评估臂 → no_pairable_cohort。"""
    rows = [_rec(d, "U", 1, "600001", 0.01, layer=pv.LAYER_MINUTE20)
            for d in _dates(5)]
    report = gate.run(rows, gate.load_contract())
    assert report["decision"] == "no_pairable_cohort"
    assert report["promote"] is False
    assert report["dropped_train_arm_rows"] == len(rows)
    assert report["arms"] == {}
    print("PASS train_arm_u_excluded_from_gate")


def test_forward_arms_enter_gate_pairing():
    """前向 A/B1/B2 可进入配对评估；样本不足只过不了样本门，不是无配对。"""
    rows = _forward_rows(_dates(8))
    report = gate.run(rows, gate.load_contract(), layer=pv.LAYER_MINUTE20)
    assert report["decision"] != "no_pairable_cohort"
    assert set(report["arms"]) == {"A", "B1", "B2"}
    assert report["gates"]["sample_gate"]["pass"] is False
    assert report["promote"] is False
    print("PASS forward_arms_enter_gate_pairing")


def test_missing_arm_still_no_pairable():
    """缺 A 臂（仅 B1/B2）→ 配对不齐，no_pairable_cohort。"""
    rows = _forward_rows(_dates(8), arms=("B1", "B2"))
    report = gate.run(rows, gate.load_contract(), layer=pv.LAYER_MINUTE20)
    assert report["decision"] == "no_pairable_cohort"
    assert report["promote"] is False
    print("PASS missing_arm_still_no_pairable")


def test_explicit_records_filters_u():
    """显式台账（--records）归一后剔除 U 训练臂，只留评估臂。"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "forward_ledger.json")
    payload = {"records": [
        {"signal_date": "2026-10-01", "arm": "U", "code": "600001",
         "net_ret": "0.01", "entry_status": "filled", "exit_status": "exited",
         "layer": "minute20"},
        {"signal_date": "2026-10-01", "arm": "A", "code": "600002",
         "net_ret": "0.015", "entry_status": "filled", "exit_status": "exited",
         "layer": "minute20"},
    ]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    rows = gate.load_records(path)
    assert len(rows) == 1
    assert rows[0]["arm"] == "A"
    assert rows[0]["net_ret"] == 0.015
    print("PASS explicit_records_filters_u")


def test_empty_ledger_layer_unknown_no_pairable():
    """空台账：层别标 unknown，不冒充 approx/minute20；无配对 cohort。"""
    report = gate.run([], gate.load_contract())
    assert report["layer"] == "unknown"
    assert report["decision"] == "no_pairable_cohort"
    assert report["promote"] is False
    assert report["arms"] == {}
    assert report["sample"]["n_settled_trips"] == 0
    print("PASS empty_ledger_layer_unknown_no_pairable")


def test_load_forward_delegates_to_data_layer():
    """默认前向台账委托 premium_data.load_forward；返回行不得含 U 训练臂。"""
    rows = gate.load_forward()
    assert all(str(r.get("arm")) != "U" for r in rows)
    print("PASS load_forward_delegates_to_data_layer")


def main():
    test_hac_constant_series()
    test_hac_lag_formula()
    test_hac_matches_manual_independent()
    test_hac_empty_and_single()
    test_normalize_string_types_and_aliases()
    test_bool_string_zero_is_false()
    test_cash_weight_no_redistribution()
    test_b0_fixed_seat_cash_no_redistribution()
    test_unfilled_contributes_zero_no_redistribution()
    test_b2_equal_weight_executable()
    test_mdd_positive_magnitude()
    test_concentration_drop_largest_three()
    test_pass_scenario_minute20_promotes()
    test_approx_layer_never_promotes_even_with_good_numbers()
    test_unresolved_cohort_excluded_and_blocks_promotion()
    test_insufficient_sample_blocks_promotion()
    test_beats_baselines_fails_when_a_loses()
    test_risk_gate_flags_deep_drawdown()
    test_load_records_csv_coerces_types()
    test_proxy_archive_coverage_readonly()
    test_train_arm_u_excluded_from_gate()
    test_forward_arms_enter_gate_pairing()
    test_missing_arm_still_no_pairable()
    test_explicit_records_filters_u()
    test_empty_ledger_layer_unknown_no_pairable()
    test_load_forward_delegates_to_data_layer()
    print("\npremium11 验证门测试全部通过")


if __name__ == "__main__":
    main()
