# -*- coding: utf-8 -*-
"""premium11 数据底座测试：纯函数 VWAP/校验、宇宙快照幂等、采集状态机、
顺延推进、proxy 层隔离、真实 pkl 只读回放。离线合成 + 临时目录；不写生产目录。

运行（项目根目录）:
    D:\\Miniconda3\\python.exe tests/test_vane_premium_data.py
或: D:\\Miniconda3\\python.exe -m pytest tests/test_vane_premium_data.py -q
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import vane.premium_data as PD  # noqa: E402

CAL = [dt.date(2026, 11, 20), dt.date(2026, 11, 23),
       dt.date(2026, 11, 24), dt.date(2026, 11, 25)]
SIG = "2026-11-20"
EXEC = "2026-11-23"
EXIT = "2026-11-24"
NEXT = "2026-11-25"


# ---------- 合成夹具 ----------

def _labels(n: int) -> list[str]:
    head = ["0930"] + [f"{m:04d}" for m in range(931, 951)]
    rest = [f"{m:04d}" for m in range(1000, 1000 + n - len(head))]
    return (head + rest)[:n]


def day_points(price: float = 10.0, n: int = 267, vol_inc: float = 100.0,
               zero: bool = False) -> list:
    """累计口径分钟点：vol 手、amt 元；amt=vol×100×price（单位自洽）。"""
    pts, vol = [], 0.0
    for hh in _labels(n):
        vol += 0.0 if zero else vol_inc
        amt = 0.0 if zero else vol * 100.0 * price
        pts.append((hh, price, vol, amt))
    return pts


def _row(code: str, rank: int, B: int = 4, arm: str = "A", **feat) -> dict:
    row = {"code": code, "name": f"测试{code}", "B": B, "group": f"{B}to{B+1}",
           "rank": rank, "arm": arm, "close": 10.0,
           "model_version": "premium-shadow", "model_score": 0.6,
           "old_model_version": "vane-3d-2-lr05-platt", "old_promo_p": 0.42}
    row.update(feat)
    return row


def _make_env():
    tmp = tempfile.mkdtemp()
    orig = {k: getattr(PD, k) for k in
            ("PREMIUM_DIR", "UNIVERSE_DIR", "MINUTE_DIR", "LABELS_CSV",
             "AUDIT_JSON", "AUDIT_LOG", "SHADOW_JSON", "LEDGER_JSON")}
    PD.PREMIUM_DIR = tmp
    PD.UNIVERSE_DIR = os.path.join(tmp, "universe")
    PD.MINUTE_DIR = os.path.join(tmp, "minute")
    PD.LABELS_CSV = os.path.join(tmp, "labels.csv")
    PD.AUDIT_JSON = os.path.join(tmp, "data_audit.json")
    PD.AUDIT_LOG = os.path.join(tmp, "data_audit.log")
    PD.SHADOW_JSON = os.path.join(tmp, "shadow_daily.json")
    PD.LEDGER_JSON = os.path.join(tmp, "forward_ledger.json")

    def restore():
        for k, v in orig.items():
            setattr(PD, k, v)
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, restore


def _fake_minute(day_map: dict):
    def _fetch(code: str):
        return [{"date": ymd, "complete": len(pts) >= PD.COMPLETE_POINTS,
                 "points": pts} for ymd, pts in day_map.items()]
    return _fetch


def _fake_bars(opens: dict):
    def _fetch(code: str, today=None):
        dates = [SIG, EXEC, EXIT, NEXT]
        return pd.DataFrame({
            "date": dates,
            "open": [opens.get(d, 10.0) for d in dates],
            "close": [10.0] * len(dates),
            "high": [10.5] * len(dates),
            "low": [9.5] * len(dates),
            "volume": [10000.0] * len(dates),
        })
    return _fetch


# ---------- 1. 纯函数：VWAP 与校验 ----------

def test_vwap_full_day_and_window_basic():
    pts = day_points(price=10.0)
    assert PD.full_day_vwap(pts) == 10.0
    win = PD.window_vwap(pts)
    assert win["vwap"] == 10.0
    assert win["n_labels"] == PD.EXIT_LABELS == 21
    assert win["issues"] == []
    print("PASS vwap_full_day_and_window_basic")


def test_validate_detects_corruption():
    ok = day_points()
    assert PD.validate_minute_points(ok)["complete"] is True

    dup = day_points()
    dup[5] = dup[4]                      # 重复 hhmm
    assert PD.validate_minute_points(dup)["ok"] is False

    dec = day_points()
    dec[10] = (dec[10][0], dec[10][1], dec[9][2] - 50, dec[9][3] - 50)
    assert PD.validate_minute_points(dec)["monotone"] is False

    zero = day_points(zero=True)
    v = PD.validate_minute_points(zero)
    assert v["zero_volume"] is True and v["complete"] is False

    bad_price = day_points()
    bad_price[3] = (bad_price[3][0], 0.0, bad_price[3][2], bad_price[3][3])
    assert PD.validate_minute_points(bad_price)["ok"] is False

    trunc = day_points(n=100)
    assert PD.validate_minute_points(trunc)["complete"] is False
    print("PASS validate_detects_corruption")


def test_zero_trade_judged_by_endpoint_cumulative():
    # 零成交：0950 末点累计量/额为 0；不得对累计量跨点求和
    pts = day_points(price=9.0, zero=True)
    blocked = PD.detect_exit_blocked(pts, prev_close=10.0)
    assert blocked["blocked"] is True and blocked["reason"] == "zero_trade"
    print("PASS zero_trade_judged_by_endpoint_cumulative")


# ---------- 2. 宇宙快照：幂等/冻结/特征留存 ----------

def test_record_signal_roundtrip_freeze_and_features():
    tmp, restore = _make_env()
    try:
        rows = [_row("600001", 1, span=2, breaks=0, seal_ratio=0.31,
                     fbt_min=93, log_close=2.3, log_ltsz=20.1, mkt_height=5,
                     mkt_two_plus=12, ind_zt_count=3, ind_max_b=4,
                     height_trend=1.0),
                _row("600002", 2, B=5, span=1, breaks=0),
                _row("600003", 3, B=2)]          # B=2 出宇宙
        r1 = PD.record_signal(SIG, rows, now=dt.date(2026, 11, 20))
        assert r1["status"] == "recorded" and r1["n_rows"] == 2
        uni = PD.load_universe(SIG)
        assert {x["code"] for x in uni} == {"600001", "600002"}
        a = next(x for x in uni if x["code"] == "600001")
        assert a["old_promo_p"] == 0.42 and a["old_model_version"]
        assert a["raw"]["seal_ratio"] == 0.31

        # 跨日冻结：不带 now 接缝 → 拒绝覆写，原行不变
        r2 = PD.record_signal(SIG, [_row("600009", 1)])
        assert r2["status"] == "frozen" and r2["n_rows"] == 2
        assert {x["code"] for x in PD.load_universe(SIG)} == {"600001", "600002"}

        # 当日重跑接缝允许替换
        r3 = PD.record_signal(SIG, [_row("600008", 1, B=4)],
                              now=dt.date(2026, 11, 20))
        assert r3["status"] == "recorded" and r3["n_rows"] == 1
    finally:
        restore()
    print("PASS record_signal_roundtrip_freeze_and_features")


# ---------- 3. 采集结算：正常成交 + proxy 层隔离 ----------

def test_collect_due_settled_flow_and_proxy_isolation():
    tmp, restore = _make_env()
    try:
        PD.record_signal(SIG, [_row("600001", 1), _row("600002", 2),
                               _row("600003", 3)],
                         now=dt.date(2026, 11, 20))
        day_map = {"20261123": day_points(10.0), "20261124": day_points(10.5)}
        stats = PD.collect_due(
            today=dt.date(2026, 11, 25), calendar=CAL,
            fetch_minute=_fake_minute(day_map),
            fetch_daily=_fake_bars({EXEC: 10.0, EXIT: 10.5}))
        assert stats["records"] == 3 and stats["filled"] == 3
        assert stats["proxy_rows"] == 3

        m = PD.load_labels("minute20")
        assert len(m) == 3
        for r in m:
            assert r["entry_status"] == "filled"
            assert r["exit_status"] == "exited"
            assert str(r["unresolved_flag"]) == "0"
            assert r["complete"] in (True, "True")
            assert r["buy_vwap"] and r["sell_vwap_20m"]
            assert r["net_ret"] not in (None, "")
            # 特征已复制到顶层
            assert "seal_ratio" in r and "fbt_min" in r

        p = PD.load_labels("approx_open")
        assert len(p) == 3
        for r in p:
            assert r["arm"] == "U"                    # 训练层统一 U，执行臂由台账派生
            assert r["layer"] == "approx_open"
            assert r["entry_status"] == "filled"
            assert r["exit_status"] == "exited"
            assert str(r["unresolved_flag"]) == "0"
            assert r["buy_vwap"] in (None, "")
            assert r["sell_vwap_20m"] in (None, "")
            assert r["exec_open"] and r["exit_open"]
            assert r["net_ret"] not in (None, "")

        # 分钟原料落盘（T+1 全日 + T+2 窗口）
        assert os.path.isdir(os.path.join(tmp, "minute", "20261123"))
        assert os.path.isdir(os.path.join(tmp, "minute", "20261124"))
    finally:
        restore()
    print("PASS collect_due_settled_flow_and_proxy_isolation")


# ---------- 4. 一字买不进：现金席，未决=0，不重分配 ----------

def test_one_word_unfilled_is_settled_cash_slot():
    tmp, restore = _make_env()
    try:
        PD.record_signal(SIG, [_row("600001", 1)],
                         now=dt.date(2026, 11, 20))
        one_word = day_points(price=11.0)     # 涨停价 11.0（昨收 10.0）
        day_map = {"20261123": one_word, "20261124": day_points(10.5)}
        PD.collect_due(today=dt.date(2026, 11, 25), calendar=CAL,
                       fetch_minute=_fake_minute(day_map),
                       fetch_daily=_fake_bars({}))
        m = PD.load_labels("minute20")
        assert len(m) == 1
        r = m[0]
        assert r["entry_status"] == "unfilled"
        assert r["exit_status"] == "unfilled"
        assert str(r["unresolved_flag"]) == "0"      # 已处理现金席
        assert r["net_ret"] in (None, "")            # 单笔净收益为空
        assert abs(float(r["cash_weight"]) - 1.0 / 3.0) < 1e-6
        assert float(r["filled_ratio"]) == 0.0
    finally:
        restore()
    print("PASS one_word_unfilled_is_settled_cash_slot")


# ---------- 5. 退出受阻：carry 顺延，后续交易日继续采集/结算 ----------

def test_exit_blocked_carries_then_advances_next_day():
    tmp, restore = _make_env()
    try:
        PD.record_signal(SIG, [_row("600001", 1)],
                         now=dt.date(2026, 11, 20))
        # 第一轮：T+2 跌停锁死（9.0 = 10.0×0.9）→ carry 顺延
        blocked = day_points(price=9.0)
        day_map1 = {"20261123": day_points(10.0), "20261124": blocked}
        PD.collect_due(today=dt.date(2026, 11, 24), calendar=CAL,
                       fetch_minute=_fake_minute(day_map1),
                       fetch_daily=_fake_bars({}))
        m1 = PD.load_labels("minute20")
        assert len(m1) == 1
        r1 = m1[0]
        assert r1["exit_status"] == "carry"
        assert str(r1["carry_flag"]) == "1"
        assert str(r1["unresolved_flag"]) == "1"
        assert r1["net_ret"] in (None, "")
        assert r1["exit_date"] == EXIT

        # 第二轮：未决头寸推进到下一交易日 11-25，当日窗口可卖 → exited
        day_map2 = {"20261123": day_points(10.0), "20261125": day_points(10.5)}
        PD.collect_due(today=dt.date(2026, 11, 25), calendar=CAL,
                       fetch_minute=_fake_minute(day_map2),
                       fetch_daily=_fake_bars({}))
        m2 = PD.load_labels("minute20")
        assert len(m2) == 1
        r2 = m2[0]
        assert r2["exit_date"] == NEXT
        assert r2["exit_status"] == "exited"
        assert str(r2["unresolved_flag"]) == "0"
        assert r2["net_ret"] not in (None, "")
    finally:
        restore()
    print("PASS exit_blocked_carries_then_advances_next_day")


# ---------- 6. 故障/截断：不写伪完整数据 ----------

def test_fetch_failure_no_pseudo_write():
    tmp, restore = _make_env()
    try:
        PD.record_signal(SIG, [_row("600001", 1)],
                         now=dt.date(2026, 11, 20))

        def _boom(code):
            raise RuntimeError("network down")

        PD.collect_due(today=dt.date(2026, 11, 25), calendar=CAL,
                       fetch_minute=_boom, fetch_daily=_fake_bars({}))
        m = PD.load_labels("minute20")
        assert len(m) == 1
        assert m[0]["entry_status"] == "fail"
        assert str(m[0]["unresolved_flag"]) == "1"
        # 无分钟原料落盘
        assert not os.path.isdir(os.path.join(tmp, "minute"))
    finally:
        restore()
    print("PASS fetch_failure_no_pseudo_write")


def test_truncated_minute_not_persisted():
    tmp, restore = _make_env()
    try:
        PD.record_signal(SIG, [_row("600001", 1)],
                         now=dt.date(2026, 11, 20))
        day_map = {"20261123": day_points(n=100)}     # 截断
        PD.collect_due(today=dt.date(2026, 11, 25), calendar=CAL,
                       fetch_minute=_fake_minute(day_map),
                       fetch_daily=_fake_bars({}))
        m = PD.load_labels("minute20")
        assert m[0]["entry_status"] == "nodata"
        assert not os.path.isdir(os.path.join(tmp, "minute"))
    finally:
        restore()
    print("PASS truncated_minute_not_persisted")


# ---------- 7. 真实 pkl 只读回放（缺配对 T+2，不伪造标签） ----------

def test_real_pkl_replay_readonly():
    report = PD.replay_real_minute()
    if report["codes"] == 0:
        print("SKIP 既有竞价分钟 pkl 目录为空")
        return
    assert report["monotone_ok"] == report["codes"]
    for day in report["days"]:
        for s in day["samples"]:
            assert s["n_points"] >= PD.COMPLETE_POINTS
            assert s["full_day_vwap"] is not None
            assert s["vwap_0930_0950"] is not None
    assert "未伪造历史 minute20 标签" in report["note"]
    print(f"PASS real_pkl_replay_readonly codes={report['codes']}")


# ---------- 8. 执行台账：U 训练层 + forward_ledger 臂派生 ----------

def _u_row(sd, code, layer="minute20", entry="filled", exit="exited", net=0.01,
           model_score=0.5, old_promo_p=0.4, unresolved=0, premium=None, rank=""):
    return {
        "signal_date": sd, "exec_date": "", "exit_date": "", "arm": "U",
        "rank": rank, "code": code, "name": f"测试{code}", "B": 4,
        "model_version": "vane-3d-2-lr05-platt", "model_score": model_score,
        "premium_score": premium, "old_model_version": "vane-3d-2-lr05-platt",
        "old_promo_p": old_promo_p, "layer": layer, "entry_status": entry,
        "filled_ratio": 1.0 if entry == "filled" else 0.0, "buy_vwap": 10.0,
        "exit_status": exit, "sell_vwap_20m": 10.1, "gross_ret": net,
        "net_ret": net, "weight": 0.0, "cash_weight": 0.0,
        "carry_flag": 1 if exit == "carry" else 0, "unresolved_flag": unresolved,
        "data_source": "tx-minute", "fetched_at": "2026-11-25 00:00:00",
        "tz": "Asia/Shanghai", "adj": "raw", "complete": True, "note": "",
        "exec_open": None, "exit_open": None,
        "span": 4, "breaks": 0, "seal_ratio": 0.3, "fbt_min": 570,
        "log_close": 1.1, "log_ltsz": 1.5, "mkt_height": 5, "mkt_two_plus": 10,
        "ind_zt_count": 2, "ind_max_b": 5, "height_trend": 0.5,
    }


def test_forward_ledger_arms_idempotent():
    import fstore
    tmp, restore = _make_env()
    try:
        u_rows = [
            # 日1 minute20：成交 / 一字现金 / 未决顺延
            _u_row("2026-11-20", "600001", model_score=0.5, premium=1.2),
            _u_row("2026-11-20", "600002", model_score=0.4),
            _u_row("2026-11-20", "600003", model_score=0.3),
            _u_row("2026-11-20", "600004", entry="unfilled", exit="unfilled",
                   net=None, unresolved=0),
            _u_row("2026-11-20", "600005", exit="carry", net=None, unresolved=1),
            # 日2 minute20
            _u_row("2026-11-23", "600011", model_score=0.8),
            _u_row("2026-11-23", "600012", model_score=0.7),
            _u_row("2026-11-23", "600013", model_score=0.6),
            # 日1 proxy 层
            _u_row("2026-11-20", "600001", layer="approx_open", net=0.02),
            _u_row("2026-11-20", "600002", layer="approx_open", net=0.01),
        ]
        PD._save_labels(u_rows)

        shadow = [
            {"signal_date": "2026-11-20", "layer": "minute20",
             "model_version": "prem-1",
             "A_picks": [{"rank": 1, "code": "600001", "score": 1.2},
                         {"rank": 2, "code": "600002", "score": 0.9},
                         {"rank": 3, "code": "600004", "score": 0.4}],
             "B1_picks": [{"rank": 1, "code": "600002", "score": 0.7},
                          {"rank": 2, "code": "600003", "score": 0.6},
                          {"rank": 3, "code": "600001", "score": 0.5}],
             "B0_picks": [{"rank": 1, "code": "600003", "score": None}]},
            {"signal_date": "2026-11-23", "layer": "minute20",
             "model_version": "prem-1",
             "A_picks": [],
             "B1_picks": [{"rank": 1, "code": "600011", "score": 0.8},
                          {"rank": 2, "code": "600012", "score": 0.7}],
             "B0_picks": [{"rank": 1, "code": "600013", "score": None}]},
            {"signal_date": "2026-11-20", "layer": "approx_open",
             "model_version": "prem-1",
             "A_picks": [{"rank": 1, "code": "600001", "score": 1.1}],
             "B1_picks": [], "B0_picks": []},
        ]
        fstore.write_json(str(PD.SHADOW_JSON), shadow)

        stats = PD.sync_forward_ledger()
        assert stats["records"] == len(PD.load_forward())

        def fwd(sd, code, arm, layer="minute20"):
            return next((r for r in PD.load_forward(layer)
                         if r["signal_date"] == sd and r["code"] == code
                         and r["arm"] == arm), None)

        # B2 = 全宇宙 U 行（含现金/未决）
        b2 = PD.load_forward()
        assert sum(1 for r in b2 if r["arm"] == "B2") == len(u_rows)

        # A：日1 三只（含一字买不进的 600004，臂行保留）；日2 A 榜为空 → 无 A 行
        a1 = {r["code"] for r in PD.load_forward("minute20")
              if r["signal_date"] == "2026-11-20" and r["arm"] == "A"}
        assert a1 == {"600001", "600002", "600004"}
        assert not [r for r in PD.load_forward("minute20")
                    if r["signal_date"] == "2026-11-23" and r["arm"] == "A"]

        # B1 / B0 名单与 A 不同
        b1_1 = {r["code"] for r in PD.load_forward("minute20")
                if r["signal_date"] == "2026-11-20" and r["arm"] == "B1"}
        assert b1_1 == {"600002", "600003", "600001"}
        b1_2 = {r["code"] for r in PD.load_forward("minute20")
                if r["signal_date"] == "2026-11-23" and r["arm"] == "B1"}
        assert b1_2 == {"600011", "600012"}
        b0_1 = {r["code"] for r in PD.load_forward("minute20")
                if r["signal_date"] == "2026-11-20" and r["arm"] == "B0"}
        assert b0_1 == {"600003"}
        b0_2 = {r["code"] for r in PD.load_forward("minute20")
                if r["signal_date"] == "2026-11-23" and r["arm"] == "B0"}
        assert b0_2 == {"600013"}

        # Top3 席位权重 1/3；B2 占位 0；臂得分落对应字段
        a_row = fwd("2026-11-20", "600001", "A")
        assert abs(float(a_row["weight"]) - 1 / 3) < 1e-6
        assert float(a_row["premium_score"]) == 1.2
        b1_row = fwd("2026-11-20", "600002", "B1")
        assert abs(float(b1_row["weight"]) - 1 / 3) < 1e-6
        assert float(b1_row["model_score"]) == 0.7
        b2_row = fwd("2026-11-20", "600001", "B2")
        assert float(b2_row["weight"]) == 0.0

        # 一字现金席：T 晚入选 A 后即使 T+1 买不进，A 臂行必须保留——
        # entry/exit=unfilled、net_ret 空、weight/cash_weight=1/3、unresolved=0，不重分配
        cash_a = fwd("2026-11-20", "600004", "A")
        assert cash_a and cash_a["entry_status"] == "unfilled"
        assert cash_a["exit_status"] == "unfilled"
        assert cash_a["net_ret"] in (None, "")
        assert abs(float(cash_a["weight"]) - 1 / 3) < 1e-6
        assert abs(float(cash_a["cash_weight"]) - 1 / 3) < 1e-6
        assert str(cash_a["unresolved_flag"]) == "0"
        cash_b2 = fwd("2026-11-20", "600004", "B2")
        assert cash_b2 and cash_b2["entry_status"] == "unfilled"
        # 未决顺延：未被任何 Top3 臂选中，仅 B2 可见，unresolved=1 保留
        carry = fwd("2026-11-20", "600005", "B2")
        assert carry and str(carry["unresolved_flag"]) == "1"
        assert carry["net_ret"] in (None, "")
        assert not fwd("2026-11-20", "600005", "A")

        # proxy 层台账独立存在
        pa = fwd("2026-11-20", "600001", "A", "approx_open")
        assert pa and float(pa["premium_score"]) == 1.1
        assert sum(1 for r in PD.load_forward("approx_open") if r["arm"] == "B2") == 2

        # 训练层 U 无重复键
        labels = PD.load_labels()
        keys = [(r["signal_date"], r["code"], r["layer"]) for r in labels]
        assert len(keys) == len(set(keys))
        assert all(r["arm"] == "U" for r in labels)

        # 幂等：二次同步台账记录一致
        before = {(r["signal_date"], r["code"], r["arm"], r["layer"]): r
                  for r in PD.load_forward()}
        PD.sync_forward_ledger()
        after = {(r["signal_date"], r["code"], r["arm"], r["layer"]): r
                 for r in PD.load_forward()}
        assert set(before) == set(after)
        for k in before:
            assert before[k] == after[k]
    finally:
        restore()
    print("PASS forward_ledger_arms_idempotent")


def main():
    test_vwap_full_day_and_window_basic()
    test_validate_detects_corruption()
    test_zero_trade_judged_by_endpoint_cumulative()
    test_record_signal_roundtrip_freeze_and_features()
    test_collect_due_settled_flow_and_proxy_isolation()
    test_one_word_unfilled_is_settled_cash_slot()
    test_exit_blocked_carries_then_advances_next_day()
    test_fetch_failure_no_pseudo_write()
    test_truncated_minute_not_persisted()
    test_forward_ledger_arms_idempotent()
    test_real_pkl_replay_readonly()
    print("\npremium11 数据底座测试全部通过")


if __name__ == "__main__":
    main()
