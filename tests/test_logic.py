# -*- coding: utf-8 -*-
"""离线逻辑测试：不依赖网络，用合成数据验证筛选与二板定龙头口径。

运行: python tests/test_logic.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_store
import ledger
import data_source
import pipeline
import report as rpt
import screener as scr
import second_board

import shutil


def test_main_board():
    for c in ["600000", "601398", "603288", "605117", "000001", "002594", "003816"]:
        assert scr.is_main_board(c), c
    for c in ["300750", "301236", "688981", "689009", "830799", "920002", "430047"]:
        assert not scr.is_main_board(c), c
    print("PASS is_main_board")


def test_limit_prices():
    assert scr.limit_prices(10.00, "浦发银行") == (11.00, 9.00)
    assert scr.limit_prices(10.00, "ST某某") == (10.50, 9.50)
    assert scr.limit_prices(10.03, "普通股") == (11.03, 9.03)
    assert scr.limit_prices(7.27, "*ST某某") == (7.63, 6.91)  # 7.27*1.05=7.6335 -> 7.63
    print("PASS limit_prices")


def test_build_candidates():
    spot = pd.DataFrame([
        # 主板普通，收盘封板：昨收10，涨停价11
        dict(code="600000", name="封板股", close=11.00, high=11.00, low=10.10, prev_close=10.0,
             pct=10.0, amount=1e8, turnover=5.0),
        # 主板普通，曾涨停炸板：最高触及11，收盘10.5
        dict(code="600001", name="炸板股", close=10.50, high=11.00, low=10.00, prev_close=10.0,
             pct=5.0, amount=1e8, turnover=5.0),
        # ST 5% 封板：10.5 即涨停
        dict(code="600002", name="ST测试", close=10.50, high=10.50, low=10.00, prev_close=10.0,
             pct=5.0, amount=1e8, turnover=5.0),
        # 曾跌停但收盘翘板
        dict(code="600003", name="翘板股", close=9.40, high=9.80, low=9.00, prev_close=10.0,
             pct=-6.0, amount=1e8, turnover=5.0),
        # 振幅 7% 未触涨跌停：(10.6-9.9)/10=7%
        dict(code="600004", name="振幅股", close=10.30, high=10.60, low=9.90, prev_close=10.0,
             pct=3.0, amount=1e8, turnover=5.0),
        # 振幅 6.9% 不入选
        dict(code="600005", name="平凡股", close=10.20, high=10.59, low=9.90, prev_close=10.0,
             pct=2.0, amount=1e8, turnover=5.0),
        # 创业板涨停，应被主板过滤
        dict(code="300001", name="创业板", close=11.00, high=11.00, low=10.10, prev_close=10.0,
             pct=10.0, amount=1e8, turnover=5.0),
    ])
    cands = scr.build_candidates(spot)
    codes = {c["code"]: c for c in cands}
    assert set(codes) == {"600000", "600001", "600003", "600004"}, set(codes)
    assert "600002" not in codes  # ST 股被剔除
    assert codes["600000"]["pools"] == ["涨停"] and codes["600000"]["zt_status"] == "封板"
    assert codes["600001"]["zt_status"] == "炸板"
    assert codes["600003"]["pools"] == ["跌停"] and codes["600003"]["dt_status"] == "翘板"
    assert codes["600004"]["pools"] == ["振幅"]
    print("PASS build_candidates")


def test_cache_is_valid():
    """缓存有效性：版本/日期/时间戳 + 「复盘日盘中写的一律作废」（过零点回归）。

    回归点：盘中作废判定此前包在 trade_date == today 里——前一天 14:xx 的
    盘中快照过了零点就被永久判成收盘数据。
    """
    V = cache_store.CACHE_VERSION
    today = dt.date.today()
    yday = today - dt.timedelta(days=1)
    close_ts = dt.datetime.combine(today, dt.time(15, 30)).timestamp()   # 收盘缓冲后
    noon_ts = dt.datetime.combine(today, dt.time(14, 0)).timestamp()     # 盘中

    def mk(date, ts, version=V):
        return {"date": date, "ts": ts, "version": version}

    assert data_source.cache_is_valid(mk(today, close_ts), today)
    assert not data_source.cache_is_valid(mk(yday, close_ts), today)     # 日期不符
    assert not data_source.cache_is_valid({"date": today, "ts": close_ts}, today)  # 缺 version
    assert not data_source.cache_is_valid(mk(today, close_ts, V - 1), today)  # 旧版本
    assert not data_source.cache_is_valid(mk(today, 0), today)           # 无时间戳
    # 复盘日盘中写的缓存一律作废：与今天几号、现在几点都无关
    y_noon = dt.datetime.combine(yday, dt.time(14, 0)).timestamp()
    assert not data_source.cache_is_valid(mk(today, noon_ts), today)
    assert not data_source.cache_is_valid(mk(yday, y_noon), yday)        # 过零点后仍作废
    assert cache_store.is_intraday_cache(mk(yday, y_noon), yday)
    assert not cache_store.is_intraday_cache(mk(yday, close_ts - 86400), yday)
    print("PASS cache_is_valid")


def test_window_dates():
    d = {x: dt.date(2026, 8, x) for x in (17, 18, 19, 20, 21, 24)}
    cal = [d[17], d[18], d[19], d[20], d[21], d[24]]  # 17周一 ... 21周五, 24下周一
    # 周一~周四：周内滚动，最多保留最近 3 个
    assert cache_store.window_dates(cal, d[17]) == [d[17]]
    assert cache_store.window_dates(cal, d[18]) == [d[17], d[18]]
    assert cache_store.window_dates(cal, d[19]) == [d[17], d[18], d[19]]
    assert cache_store.window_dates(cal, d[20]) == [d[18], d[19], d[20]]  # 周四，周一滚出
    # 周五：最近 3 个交易日（周三/四/五）；周末悬停等价
    assert cache_store.window_dates(cal, d[21]) == [d[19], d[20], d[21]]
    # 下周一：清空上一轮，窗口=[下周一]
    assert cache_store.window_dates(cal, d[24]) == [d[24]]
    print("PASS window_dates")


def test_cache_store():
    tmp = "out_cache_test"
    d = {x: dt.date(2026, 8, x) for x in (17, 18, 19, 20, 21, 24)}
    cal = [d[17], d[18], d[19], d[20], d[21], d[24]]

    for x in (17, 18, 19, 20):
        cache_store.save_spot(tmp, d[x], pd.DataFrame({"code": [f"60000{x}"]}))
    cache_store.save_result(tmp, d[20], [{"ok": True}], [], [{"ok": True}])

    got = cache_store.load(tmp, d[20])
    assert got["spot"] is not None and got["results"]

    # 周四(20)：窗口=[二,三,四]，周一滑出清除
    cache_store.prune(tmp, cal, d[20], keep=3)
    files = sorted(os.listdir(os.path.join(tmp, "cache")))
    assert files == ["20260818.pkl", "20260819.pkl", "20260820.pkl"], files

    # 周五(21)收盘后：窗口=周三/四/五，周一周二滑出清除
    cache_store.prune(tmp, cal, d[21], keep=3)
    files = sorted(os.listdir(os.path.join(tmp, "cache")))
    assert files == ["20260819.pkl", "20260820.pkl"], files

    # 下周一(24)：清空上一轮
    cache_store.prune(tmp, cal, d[24], keep=3)
    files = sorted(os.listdir(os.path.join(tmp, "cache")))
    assert files == [], files

    # 旧版单文件缓存清理
    open(os.path.join(tmp, "cache_spot.pkl"), "wb").close()
    cache_store.prune(tmp, cal, d[24], keep=3)
    assert not os.path.exists(os.path.join(tmp, "cache_spot.pkl"))
    shutil.rmtree(tmp)
    print("PASS cache_store")


def test_pool_candidates():
    zt = [["600000", 0, "测试甲", 11000, 10.0, 1e8, 0, 0, 5.0, 2,
           93000, 93000, 5e7, 0, "行业", {"days": 1, "ct": 1}],
          ["300001", 0, "创业板", 11000, 10.0, 1e8, 0, 0, 5.0, 1,
           93000, 93000, 5e7, 0, "行业", {"days": 1, "ct": 1}]]
    zb = [["600001", 0, "测试乙", 10500, 11000, 5.0, 1e8, 0, 0, 5.0,
           93000, 2, 9.5, 1.0, {"days": 1, "ct": 1}, "行业"]]
    cands, extras = data_source.build_candidates_from_pools(zt, zb)
    m = {c["code"]: c for c in cands}
    assert set(m) == {"600000", "600001"}
    assert m["600000"]["zt_status"] == "封板" and m["600000"]["pools"] == ["涨停"]
    assert abs(m["600000"]["close"] - 11.0) < 1e-9
    assert extras["600000"]["lianban"] == 2
    assert m["600001"]["zt_status"] == "炸板" and extras["600001"]["zhaban_times"] == 2
    assert abs(m["600001"]["amp"] - 9.5) < 1e-9
    assert all(c["fill"] for c in cands)
    assert data_source.build_candidates_from_pools([], []) == (None, None)
    print("PASS pool candidates")


def test_pool_candidates_dict():
    """东财池 2026-08 起返回字段名字典行，必须与旧位置数组同样可解析。"""
    zt = [{"c": "600000", "m": 1, "n": "测试甲", "p": 11000, "zdp": 10.0,
           "amount": 1e8, "hs": 5.0, "lbc": 2, "fbt": 92500, "zbc": 0,
           "fund": 8e7, "hybk": "化学制品", "ltsz": 4.3e9,
           "zttj": {"days": 2, "ct": 2}},
          {"c": "300001", "m": 0, "n": "创业板", "p": 11000, "zdp": 10.0,
           "amount": 1e8, "hs": 5.0, "lbc": 1, "fbt": 92500, "zbc": 0,
           "fund": 5e7, "hybk": "化学制品",
           "zttj": {"days": 14, "ct": 7}}]          # 14天7板，非连续
    zb = [{"c": "600001", "m": 1, "n": "测试乙", "p": 10500, "ztp": 11000,
           "zdp": 5.0, "amount": 1e8, "hs": 5.0, "fbt": 93000, "zbc": 2,
           "zf": 9.5, "zs": 1.0, "zttj": {"days": 8, "ct": 5},
           "hybk": "化学制品"}]                      # 8天5板（实测北纬科技）
    cands, extras = data_source.build_candidates_from_pools(zt, zb)
    m = {c["code"]: c for c in cands}
    assert set(m) == {"600000", "600001"}, set(m)
    assert m["600000"]["zt_status"] == "封板"
    assert abs(m["600000"]["close"] - 11.0) < 1e-9
    assert abs(m["600000"]["turnover"] - 0.05) < 1e-9   # hs 百分数 → 小数
    assert extras["600000"]["lianban"] == 2
    assert extras["600000"]["fund"] == 8e7                # 封单额透传
    assert extras["600000"]["ltsz"] == 4.3e9              # 流通市值透传
    assert extras["600000"]["fbt"] == "0925"             # 92500 → "0925"
    assert m["600001"]["zt_status"] == "炸板" and extras["600001"]["zhaban_times"] == 2
    assert abs(m["600001"]["amp"] - 9.5) < 1e-9
    # 连板数只认 lbc：zttj 的「天数」不是连板数（此前 14天7板 → 14、8天5板 → 8）
    assert "lianban" not in extras["600001"]             # 炸板池无 lbc，不产出
    row14 = data_source.norm_pool_row(zt[1], "zt")
    assert row14["lianban"] == 1 and row14["zttj_days"] == 14 and row14["zttj_ct"] == 7
    row8 = data_source.norm_pool_row(zb[0], "zb")
    assert row8["lianban"] == 0 and row8["zttj_days"] == 8 and row8["zttj_ct"] == 5
    # 池外显式标了封板的票不被炸板池改写状态
    both = data_source.build_candidates_from_pools(zt[:1], [dict(zb[0], c="600000")])
    assert both[0][0]["zt_status"] == "封板" and both[1]["600000"]["lianban"] == 2
    assert both[1]["600000"]["zhaban_times"] == 2
    # 单行解析：dict 行缺代码返回 None
    assert data_source.norm_pool_row({"n": "无代码"}, "zt") is None
    print("PASS pool candidates (dict format)")


def test_norm_pool_row_units():
    """池换手率统一换算小数、封板时间 HHMMSS→HHMM、无法识别行返回 None。"""
    row = data_source.norm_pool_row(
        {"c": "600519", "n": "X", "p": 15000, "hs": 1.69,
         "lbc": 3, "fbt": 92500, "hybk": "白酒"}, "zt")
    assert row["turnover"] == 0.0169 and row["fbt"] == "0925"
    assert row["lianban"] == 3 and row["close"] == 15.0
    assert data_source.norm_pool_row("garbage", "zt") is None
    # 0 行解析告警：_parse_pool 有数据但全不可解析时返回空列表
    assert data_source.parse_pool("zt", ["bad", 42]) == []
    print("PASS norm pool row units")


def test_incremental_bars():
    """增量拼接精度：昨日缓存 + 今日快照行 == 全量日K；除权被拦截。"""
    n = 60
    x = np.arange(n)
    close = 10 + 0.03 * x + np.sin(x / 3.0) * 0.8
    vol = 1000 + 300 * np.abs(np.sin(x / 2.0))
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=int(i)) for i in x]
    full = pd.DataFrame({"date": dates, "open": close - 0.05, "close": close,
                         "high": close + 0.12, "low": close - 0.12, "volume": vol})
    trade_date, seed = dates[-1], full.iloc[:-1].reset_index(drop=True)
    last = full.iloc[-1]
    cand = {"prev_close": float(full["close"].iloc[-2]),
            "open": float(last["open"]), "high": float(last["high"]),
            "low": float(last["low"]), "close": float(last["close"]),
            "volume": float(last["volume"])}

    merged = scr.incremental_bars(cand, seed, trade_date)
    assert merged is not None and len(merged) == n
    pd.testing.assert_frame_equal(
        merged[["date", "open", "close", "high", "low", "volume"]],
        full[["date", "open", "close", "high", "low", "volume"]])

    # 除权（昨收对不上）→ 拒绝拼接
    bad = dict(cand, prev_close=cand["prev_close"] * 0.95)
    assert scr.incremental_bars(bad, seed, trade_date) is None
    # 快照缺 open → 拒绝
    miss = {k: v for k, v in cand.items() if k != "open"}
    assert scr.incremental_bars(miss, seed, trade_date) is None
    # 缓存已含复盘日 → 拒绝（由调用方直接复用）
    assert scr.incremental_bars(cand, full, trade_date) is None
    print("PASS incremental bars")


def test_second_board():
    """五维评分 v2：题材30+首板25+二板25+情绪15+辨识10，缺失惩罚 -2/项。"""
    n = 10
    close = [10.0] * n
    close[-3], close[-2], close[-1] = 10.0, 11.0, 12.1   # 昨日+10%涨停，今日+10%涨停
    opn = [c - 0.1 for c in close]
    opn[-2], opn[-1] = 10.4, 11.55                       # 昨日实体开盘，今日高开5%
    bars = pd.DataFrame({
        "date": [dt.date(2025, 6, 1) + dt.timedelta(days=i) for i in range(n)],
        "open": opn, "close": close,
        "high": [c + 0.05 for c in close], "low": [c - 0.05 for c in close],
        "volume": 1000.0,
    })
    cand = {"code": "600001", "name": "测试龙头", "zt_status": "封板",
            "close": 12.1, "turnover": 0.10}
    extras = {"lianban": 2, "fbt": "0930", "industry": "PCB", "zhaban_times": 0,
              "ltsz": 2.5e9}
    ctx = {"industry_count": {"PCB": 3}, "industry_max_lianban": {"PCB": 2},
           "industry_yesterday_count": {"PCB": 1},
           "persistent_industries": {"PCB"},
           "break_rate": 0.20, "promotion_rate": 0.45, "yzt_ret": 2.5,
           "yesterday_fbt": {"600001": "0925"},
           "yesterday_turnover": {"600001": 0.12},
           "yesterday_seal": {"600001": 2.5},
           "cycle_stage": "发酵", "zb_pool_ok": True}
    r = second_board.detect_second_board(cand, bars, extras, ctx)
    assert r["board_pass"], r
    # A=10+8+7(发酵) B=8+3(实体)+5+6(昨封2.5) C=7(高开优)+6(换手10%)+7(无炸板)
    # D=4+4+4+3(周期发酵) E=6+2 = 90（B维已按 n=1414 翻转：一字6/实体3）
    assert r["board_score"] == 90, r
    assert "持续题材" in r["board_note"]            # 实证标签（不计分，分数仍93）
    for key in ("题材", "首板", "二板", "情绪", "辨识"):
        assert f"{key}:" in r["board_note"], key
    assert "发酵" in r["board_note"] and "昨封2.5" in r["board_note"]
    assert "昨涨停+2.5%" in r["board_note"]              # 赚钱效应入评分
    assert "发酵期" in r["board_note"]                    # 情绪周期入 D 维
    assert "流通25亿小盘" in r["board_note"]              # 2.5e9 → 小盘

    # 满分护栏：低价深盘 + 实体首板 + 全维度拉满 = 92；一字首板同配置 = 95
    r_max = second_board.detect_second_board(
        dict(cand, close=4.5), bars, extras, ctx)
    assert r_max["board_score"] == 92, r_max
    assert "深低价" in r_max["board_note"]
    bars_w = bars.copy()
    bars_w.loc[bars_w.index[-2], "open"] = float(bars["close"].iloc[-3]) * 1.095  # 首板一字
    r_w = second_board.detect_second_board(cand, bars_w, extras, ctx)
    assert r_w["board_score"] == 93 and "一字" in r_w["board_note"], r_w  # 90+6-3
    r_wmax = second_board.detect_second_board(dict(cand, close=4.5), bars_w, extras, ctx)
    assert r_wmax["board_score"] == 95, r_wmax  # B维翻转后的真满配

    # 数据驱动分档：准一字 + 缩量 = C维满分 25（强于高开优+充分换手的 20）
    bars2 = bars.copy()
    bars2.loc[bars2.index[-1], "open"] = float(bars["close"].iloc[-2]) * 1.095  # 高开9.5%≈准一字
    r_lock = second_board.detect_second_board(
        dict(cand, turnover=0.03), bars2, extras, ctx)
    assert "准一字" in r_lock["board_note"] and "缩量板" in r_lock["board_note"]
    assert r_lock["board_score"] == r["board_score"] + 5, r_lock

    # 昨日数据缺失：中性分 + 缺失惩罚，显著低于全数据票
    ctx2 = {k: v for k, v in ctx.items()
            if k not in ("yesterday_fbt", "yesterday_turnover", "yesterday_seal",
                         "yzt_ret")}
    r2 = second_board.detect_second_board(cand, bars, extras, ctx2)
    assert "封板时间未知" in r2["board_note"]
    assert "缺失" in r2["board_note"]
    assert r2["board_score"] < r["board_score"] - 10

    # 全维度数据缺失：中性分叠加 + 缺失惩罚后不得过 60 线
    empty = second_board.detect_second_board(
        cand, bars, {}, {"industry_count": {}, "industry_max_lianban": {}})
    assert empty["board_score"] < 60, empty
    assert "缺失" in empty["board_note"]

    # 概念口径 A 维：最热概念热度 + 领涨身位 + 板块加速
    ctx_c = dict(ctx)
    ctx_c["concept_boards"] = {
        "固态电池": {"pct": 4.2, "up": 70, "down": 10, "lead": "测试龙头", "bk": "BK0001"},
        "储能": {"pct": 0.8, "up": 40, "down": 30, "lead": "别的股", "bk": "BK0002"},
    }
    ctx_c["_board_kline"] = {"BK0001": [100.0, 101.0, 102.0, 103.0, 106.0]}
    ex_c = dict(extras, concepts=["固态电池", "储能"])
    rc = second_board.detect_second_board(cand, bars, ex_c, ctx_c)
    note = rc["board_note"]
    assert "固态电池" in note and "概念领涨" in note and "加速" in note
    # A=10(热)+8(领涨)+7(加速)=25，其余维度同 r（行业兜底被概念覆盖后不应更低）
    assert rc["board_score"] >= r["board_score"], (rc["board_score"], r["board_score"])

    # 概念领涨缺失时的身位分
    ctx_c2 = dict(ctx_c)
    ctx_c2["concept_boards"] = {
        "固态电池": {"pct": 4.2, "up": 70, "down": 10, "lead": "他人", "bk": "BK0001"}}
    rc2 = second_board.detect_second_board(cand, bars, ex_c, ctx_c2)
    assert "概念领涨" not in rc2["board_note"]
    assert rc2["board_score"] < rc["board_score"]

    # 非二连板（lianban=1 且昨日非涨停）→ 拒绝
    bad_cand = {"code": "600002", "name": "首板股", "zt_status": "封板",
                "close": 11.0, "turnover": 0.10}
    bad_extras = {"lianban": 1}
    assert not second_board.detect_second_board(
        bad_cand, bars, bad_extras, {})["board_pass"]
    print("PASS second board")


def test_gap_score_single_source():
    """C 维高开缺口打分必须是单一实现（board_rules.gap_score）。

    历史教训：second_board 曾与已移除的 N 字评分器各自解析同一个
    BOARD_GAP_BEST 却给出相反极性，用户同一晚会拿到两套矛盾的高开指令。
    """
    import io as _io
    import board_rules
    # 1) 档位与极性（准一字最优 = 2026-08-30 重校后的现行口径）
    assert board_rules.gap_score(0.10) == (10, "准一字")
    assert board_rules.gap_score(0.095) == (10, "准一字")
    assert board_rules.gap_score(0.05)[0] == 7      # 最优区 3~7%
    assert board_rules.gap_score(0.03)[0] == 7
    assert board_rules.gap_score(0.07)[0] == 7
    assert board_rules.gap_score(0.02)[0] == 6      # 其余高开
    assert board_rules.gap_score(0.08)[0] == 6
    assert board_rules.gap_score(0.005)[0] == 3     # <1% 视为弱
    assert board_rules.gap_score(-0.02) == (3, "低开-2.0%")
    # 2) 源码级守卫：评分器不得内联解析缺口档位
    for mod in ("second_board",):
        src = _io.open(mod + ".py", encoding="utf-8").read()
        assert "BOARD_GAP_BEST" not in src, \
            f"{mod} 又自己解析缺口档位了——必须走 board_rules.gap_score，否则两处会再次分叉"
        assert "gap_score(" in src, f"{mod} 未调用共用实现"
    print("PASS gap score single source")


def test_board_report():
    """报告只保留二板定龙头：评分≥70 才进信号清单；CSV 带日期与关键列。"""
    tmp = "out_board_test"
    d = dt.date(2025, 6, 10)
    cands = [{"code": "600001", "name": "二板龙头", "pools": ["涨停"],
              "zt_status": "封板", "dt_status": "", "close": 12.1, "amp": 10.0,
              "pct": 10.0, "amount": 2e8, "turnover": 0.1},
             {"code": "600002", "name": "未入选封板", "pools": ["涨停"],
              "zt_status": "封板", "dt_status": "", "close": 11.0, "amp": 10.0,
              "pct": 10.0, "amount": 1e8, "turnover": 0.05},
             {"code": "600003", "name": "翘板股", "pools": ["跌停"],
              "zt_status": "", "dt_status": "翘板", "close": 9.0, "amp": 8.0,
              "pct": -5.0, "amount": 1e8, "turnover": 5.0}]
    base = {"ok": True, "skip_reason": "", "lianban": 2, "zhaban_times": 0,
            "industry": "PCB", "fbt": "0931", "fund": 1e8, "seal_ratio": 0.5,
            "ltsz": 4e9, "board_pass": False, "board_score": 0, "board_note": ""}
    results = [
        {**cands[0], **base, "board_pass": True, "board_score": 82,
         "board_note": "热门题材(PCB,3家涨停) | 首板封板早(0925)"},
        {**cands[1], **base, "lianban": 1, "board_score": 45, "board_note": "题材独苗"},
        {**cands[2], **base},   # 翘板股：非封板，零日K路径，无评分
    ]
    ctx = {"lianban_ladder": {2: 3, 3: 1}, "market_max_lianban": 3,
           "yzt_today_pct": [1.0, -2.0, 3.0], "cycle_stage": "高潮",
           "cycle_note": "高度7创新高"}
    txt, csv, _ = rpt.write_report(results, [], cands, d, tmp,
                                signal_only=True, ctx=ctx)
    body = open(txt, encoding="utf-8").read()
    # 核心输出必须渲染：此前 board 列表算出来后唯一用途是 len(board) 计数，
    # _block 只在 --all 下调用，默认报告从不显示名单，而本测试只断言「未入选的
    # 不在 body」，于是缺陷长期全绿。
    assert "600001" in body, "过线候选未渲染——核心输出不可见"
    assert "二板龙头" in body
    assert "二板定龙头 · 信息口径" in body          # 双轨分离：明示非买入指令
    assert "82分" in body and "82%" not in body      # 序数加权和不得印成概率
    assert "明日可买" in body                        # 次日可成交性标记
    assert "600002" not in body and "600003" not in body       # 未入选不展示
    assert "今日买入候选" not in body and "KDJ" not in body     # 区块已按需求移除
    assert "连板高度: 3 板（3板×1 2板×3）" in body
    assert "红盘 2/3" in body and "常态约+2%~4%" in body
    assert "情绪周期: 高潮期（高度7创新高）" in body            # 周期行
    assert "总仓位 [██████░░░░] 60%" in body                      # 量化仓位条
    cdf = pd.read_csv(csv, dtype={"code": str})
    assert set(cdf["trade_date"]) == {"2025-06-10"}
    for col in ("board_score", "seal_ratio", "ltsz", "fbt", "zt_money_effect",
                "cycle_stage", "weekday"):
        assert col in cdf.columns, col
    assert set(cdf["cycle_stage"].dropna()) == {"高潮"}    # 周期列写入归档
    assert set(cdf["weekday"].dropna()) == {"周二"}        # 2025-06-10 是周二
    assert "kdj_cross" not in cdf.columns and "wave_pass" not in cdf.columns
    # 归档存在
    arch = os.path.join(tmp, "archive")
    assert os.path.exists(os.path.join(arch, "signals_20250610.csv"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS board report")


def test_calibrate_summarize():
    """校准器分桶汇总：可成交主口径 + 聚类推断 + 日簇不足降级。"""
    import calibrate

    def mk(score, date, ow, net, promo, open_pct, close_pct):
        return {"board_score": score, "date": date, "next_open_pct": open_pct,
                "next_close_pct": close_pct, "max3_pct": close_pct + 2.0,
                "promoted": promo, "one_word": int(ow), "fillable": int(not ow),
                "tradable_ret": net + 0.152, "net_ret": net}

    d1, d2, d3 = "2026-08-10", "2026-08-11", "2026-08-12"
    rows = [
        mk(85, d1, False, 2.0, 1, 4.0, 6.0),    # 80-100，可成交
        mk(82, d2, True, -1.0, 1, 9.8, 8.0),    # 80-100，**一字买不到** → 应被排除
        mk(75, d1, False, -0.5, 1, 3.0, 2.5),   # 70-79，可成交
        mk(72, d3, False, 0.3, 0, 2.0, 2.3),    # 70-79，可成交
        mk(65, d2, False, 0.8, 0, 1.0, 1.8),    # 60-69，可成交
        mk(62, d3, False, -0.2, 0, 0.5, 0.3),   # 60-69，可成交
    ]
    out = calibrate.summarize(rows)
    # 新表头：日簇 / 可成交 / 一字% / 净开→收% / 聚类t
    for col in ("日簇", "可成交", "一字%", "净开→收%", "聚类t", "[参考]次收%"):
        assert col in out, col
    assert "80-100" in out and "70-79" in out and "60-69" in out and "基线(全部)" in out
    # 一字样本必须被排除在可成交统计外：80-100 桶 2 条，可成交只剩 1 条、一字 50%
    line80 = [l for l in out.splitlines() if l.startswith("80-100")][0]
    assert "50%" in line80, line80
    # 外部锚点与口径说明保留
    assert "34.7%" in out
    assert "可成交净收益超额" in out and "晋级率超额(信息口径)" in out
    # 过线可成交组日期 {d1,d3}、未过线 {d2,d3} → 仅 1 个配对日 < MIN_INFER_DAYS
    assert "无法推断" in out, "日簇不足时必须明说无法推断，不能给看似有效的 t"
    assert "只是点估计" in out
    # 单调性判据已换轨到可成交口径
    assert "可成交净收益**单调性" in out or "可成交净收益单调性" in out.replace("**", "")
    print("PASS calibrate summarize")


def test_cluster_inference():
    """聚类推断纯函数：CR0 标准误、按日配对差值、单簇降级、BH-FDR。"""
    import calibrate as cal

    # 单簇 → t/ci 必须为 None（不能当 0，也不能当显著）
    one = [{"date": "2026-08-10", "v": 1.0}, {"date": "2026-08-10", "v": 3.0}]
    st = cal.cluster_mean(one, "v")
    assert st["n"] == 2 and st["clusters"] == 1
    assert st["mean"] == 2.0 and st["t"] is None and st["ci"] == (None, None)
    assert cal.fmt_t(st) == "单簇"

    # 缺簇键的行仍计入均值，只是算不出 SE（均值与聚类分开过滤）
    nodate = [{"v": 1.0}, {"v": 3.0}]
    st = cal.cluster_mean(nodate, "v")
    assert st["mean"] == 2.0 and st["clusters"] == 0 and st["t"] is None

    # 两簇手算核对：值 [0,0]（d1）与 [4,4]（d2），mean=2
    # u = [-2,-2,+2,+2]；A_d1=-4, A_d2=+4 → Σ A_g² = 32
    # CR0 小样本校正 G/(G-1)=2 → SE = sqrt(32*2)/4 = sqrt(64)/4 = 2.0；t = 2/2 = 1.0
    two = [{"date": "d1", "v": 0.0}, {"date": "d1", "v": 0.0},
           {"date": "d2", "v": 4.0}, {"date": "d2", "v": 4.0}]
    st = cal.cluster_mean(two, "v")
    assert abs(st["mean"] - 2.0) < 1e-9
    assert abs(st["se"] - 2.0) < 1e-9, st
    assert abs(st["t"] - 1.0) < 1e-9, st

    # 按日配对差值：d1 组内差 1-0=1，d2 差 3-2=1 → 差恒为 1、SE=0 → t 无定义
    a = [{"date": "d1", "v": 1.0}, {"date": "d2", "v": 3.0}]
    b = [{"date": "d1", "v": 0.0}, {"date": "d2", "v": 2.0}]
    d = cal.cluster_diff(a, b, "v")
    assert d["days"] == 2 and d["dropped"] == 0 and abs(d["diff"] - 1.0) < 1e-9
    assert d["se"] == 0.0 and d["t"] is None      # 零方差不得报成显著

    # 单边日丢弃计数
    d = cal.cluster_diff(a, [{"date": "d1", "v": 0.0}], "v")
    assert d["days"] == 1 and d["dropped"] == 1

    # 两组无共同日期 → diff 为 None（调用方须降级，不能崩）
    d = cal.cluster_diff([{"date": "d1", "v": 1.0}], [{"date": "d9", "v": 1.0}], "v")
    assert d["days"] == 0 and d["diff"] is None and d["t"] is None

    # BH-FDR：q 单调不减于排序后的 p，且 None 原样保留
    q = cal.bh_fdr([0.01, 0.04, None, 0.03])
    assert q[2] is None
    assert abs(q[0] - 0.01 * 3 / 1) < 1e-9, q     # m=3（有效 p 个数）
    assert q[1] == q[3] or q[1] >= q[3]           # 单调性
    # 显式传入更小的有效检验数（共线合并）→ q 变小
    assert cal.bh_fdr([0.01, 0.04, 0.03], m=2)[0] < q[0] + 1e-12

    # 双侧 p：t=1.96 → ≈0.05；t=None → None
    assert abs(cal._two_sided_p(1.959964) - 0.05) < 1e-4
    assert cal._two_sided_p(None) is None
    print("PASS cluster inference")


def test_calibrate_forward_states():
    """前向数据三态：ok（含缓存命中）/ nodata（信号日后暂无交易日）/ fail（行情不可达）。

    另覆盖复权口径：bars 是前复权序列（东财 fqt=1 / 腾讯 qfq），而调用方传入的
    base_close 是归档 CSV 里的**原始**收盘。两者不等时必须以 bars 内信号日自身
    收盘为基准，否则跨除权日的前向收益系统性偏负、promoted 误判。

    行情源 monkeypatch，不联网。"""
    import calibrate
    import pandas as pd

    d0 = dt.date(2026, 8, 28)
    ok_bars = pd.DataFrame({
        "date": [d0, dt.date(2026, 8, 31)],
        "open": [50.0, 55.0], "close": [53.31, 58.64],
        "high": [54.0, 58.64], "low": [49.9, 54.5]})
    nodata_bars = ok_bars.iloc[:1].copy()          # 只有信号日本身
    shifted_bars = ok_bars.iloc[1:].copy()         # 不含信号日（停牌/退市/源漂移）

    orig = calibrate.data_source.fetch_close_range
    calls = {"n": 0}

    def fake(code, start, end):
        calls["n"] += 1
        if code == "600001" or code == "600004":
            return ok_bars
        if code == "600002":
            return nodata_bars
        if code == "600005":
            return shifted_bars
        return None

    calibrate.data_source.fetch_close_range = fake
    try:
        cache = {}
        st, m = calibrate.fetch_forward("600001", d0, 53.31, cache)
        assert st == "ok" and m is not None
        assert abs(m["next_open_pct"] - (55.0 / 53.31 - 1) * 100) < 1e-6
        assert abs(m["next_close_pct"] - (58.64 / 53.31 - 1) * 100) < 1e-6
        assert m["promoted"] == 1                    # +10% 晋级
        n_after_first = calls["n"]
        st2, m2 = calibrate.fetch_forward("600001", d0, 53.31, cache)
        assert st2 == "ok" and m2 == m
        assert calls["n"] == n_after_first           # 缓存命中不再请求

        # 复权混用回归：传入的原始收盘 99.99 与前复权序列内收盘 53.31 不一致。
        # 旧实现直接拿 99.99 当基准，会算出 -45% 的假暴跌且 promoted=0。
        st, m = calibrate.fetch_forward("600004", d0, 99.99, cache)
        assert st == "ok"
        assert abs(m["next_open_pct"] - (55.0 / 53.31 - 1) * 100) < 1e-6, \
            "基准用了传入的原始收盘而非前复权序列内收盘——复权口径混用"
        assert abs(m["next_close_pct"] - (58.64 / 53.31 - 1) * 100) < 1e-6
        assert m["promoted"] == 1

        # 信号日不在返回区间内 → fail 且不缓存；绝不回退到混用口径
        st, m = calibrate.fetch_forward("600005", d0, 10.0, cache)
        assert st == "fail" and m is None
        assert ("600005", d0) not in cache

        st, m = calibrate.fetch_forward("600002", d0, 10.0, cache)
        assert st == "nodata" and m is None

        st, m = calibrate.fetch_forward("600003", d0, 10.0, cache)
        assert st == "fail" and m is None
        assert ("600003", d0) not in cache           # 失败不缓存，下次自动重试
    finally:
        calibrate.data_source.fetch_close_range = orig
    print("PASS calibrate forward states")


def test_forward_tradable_metrics():
    """fetch_forward 的可成交口径：tradable_ret / net_ret / one_word 边界。"""
    import calibrate
    import trade_rules as tr
    from config import COST_ROUND_TRIP

    d0 = dt.date(2026, 8, 28)
    d1 = dt.date(2026, 8, 31)

    def bars_of(open1, close1=58.64, base=53.31):
        return pd.DataFrame({
            "date": [d0, d1], "open": [50.0, open1],
            "close": [base, close1], "high": [54.0, max(open1, close1)],
            "low": [49.9, min(open1, close1)]})

    orig = calibrate.data_source.fetch_close_range
    cases = {"600001": bars_of(55.0), "600002": bars_of(58.65),
             "600003": bars_of(58.40)}
    calibrate.data_source.fetch_close_range = lambda c, s, e: cases.get(c)
    try:
        limit = tr.limit_price(53.31)          # 手算对照，不复制取整公式
        # 53.31×1.10=58.641，交易所四舍五入到分 → 58.64。旧公式（round(v*100+0.5)
        # 双重舍入）曾给 58.65，2026-09-15 修正为 floor(v*100+0.5) 与 candidates.round2 同语义
        assert abs(limit - 58.64) < 1e-9, limit

        # (1) 次日开盘 55.0 远低于涨停价 → 可成交
        st, m = calibrate.fetch_forward("600001", d0, 53.31, {})
        assert st == "ok"
        assert m["one_word"] == 0 and m["fillable"] == 1
        exp_tr = (58.64 / 55.0 - 1.0) * 100.0
        assert abs(m["tradable_ret"] - exp_tr) < 1e-9, m
        assert abs(m["net_ret"] - (exp_tr - COST_ROUND_TRIP * 100.0)) < 1e-9, m
        # tradable_ret 的基准是**次日开盘**，与传入的 base_close 无关 →
        # 这就是它天然免疫前复权/原始价混用的原因
        st2, m2 = calibrate.fetch_forward("600001", d0, 99.99, {})
        assert abs(m2["tradable_ret"] - exp_tr) < 1e-9, m2
        # 信息口径的基准同样取 bars 内信号日自身收盘（上一轮修的复权混用），
        # 因此也不随传入的 base_close 变化——base_close 只是有效性哨兵
        assert abs(m2["next_close_pct"] - (58.64 / 53.31 - 1) * 100) < 1e-6

        # (2) 次日开盘恰等于涨停价 → 一字，排队买不到
        st, m = calibrate.fetch_forward("600002", d0, 53.31, {})
        assert m["one_word"] == 1 and m["fillable"] == 0, m

        # (3) 开盘略低于涨停价但在 LIMIT_EPS 容差内 → 仍判一字
        st, m = calibrate.fetch_forward("600003", d0, 53.31, {})
        assert 58.40 < limit and m["one_word"] == 1 and m["fillable"] == 0, m
        # net_ret 对一字样本仍然计算（语义＝「若成交则如何」），不置 0
        assert m["net_ret"] != 0.0
    finally:
        calibrate.data_source.fetch_close_range = orig
    print("PASS forward tradable metrics")


def test_forward_cache_schema():
    """load_forward_cache：缺新口径列的旧行必须判 stale 并重拉，不得静默丢弃。

    这是本轮最高风险的数据丢失路径——旧行被丢 + main() 尾部 to_csv **全量覆写**
    = 历史样本永久消失（_forward.csv 只有 199 行 / 23 个交易日，丢不起）。
    """
    import calibrate

    full = {"code": "600001", "date": "2026-08-28",
            "next_open_pct": 3.0, "next_close_pct": 5.0, "max3_pct": 8.0,
            "promoted": 1, "one_word": 0, "fillable": 1,
            "tradable_ret": 1.94, "net_ret": 1.79}
    old = {"code": "600002", "date": "2026-08-28",          # --refresh 之前的旧 schema
           "next_open_pct": 3.0, "next_close_pct": 5.0, "max3_pct": 8.0,
           "promoted": 1}
    nan_row = dict(full, code="600003", net_ret=float("nan"))

    cache, stale = calibrate.load_forward_cache(pd.DataFrame([full, old, nan_row]))
    key1 = ("600001", dt.date(2026, 8, 28))
    key2 = ("600002", dt.date(2026, 8, 28))
    key3 = ("600003", dt.date(2026, 8, 28))
    assert key1 in cache and cache[key1]["net_ret"] == 1.79
    assert cache[key1]["promoted"] == 1 and isinstance(cache[key1]["promoted"], int)
    assert key2 not in cache, "旧 schema 行必须判缓存未命中以触发重拉"
    assert key3 not in cache, "NaN 同样必须判未命中，不能当成 0"
    assert stale == 2, stale

    # 空表不崩
    cache, stale = calibrate.load_forward_cache(pd.DataFrame())
    assert cache == {} and stale == 0
    cache, stale = calibrate.load_forward_cache(None)
    assert cache == {} and stale == 0
    print("PASS forward cache schema")


def test_source_stratification():
    """来源分层：_source_of 归一化 + source_section 分层表与门槛敏感性。"""
    import calibrate

    # _source_of：pandas 缺列给的是 NaN，而 NaN 是**真值**，必须显式判
    assert calibrate._source_of({"source": "backfill"}) == "backfill"
    assert calibrate._source_of({"source": "live"}) == "live"
    assert calibrate._source_of({"source": None}) == "live*"
    assert calibrate._source_of({}) == "live*"
    assert calibrate._source_of({"source": float("nan")}) == "live*"
    assert calibrate._source_of({"source": "BACKFILL"}) == "backfill"

    def mk(score, date, src, net, ow=False):
        return {"board_score": score, "date": date, "source": src,
                "next_open_pct": 1.0, "next_close_pct": 2.0, "max3_pct": 4.0,
                "promoted": 1, "one_word": int(ow), "fillable": int(not ow),
                "tradable_ret": net + 0.152, "net_ret": net,
                "board_note": "题材:独苗" + ("·缺失1项" if src == "backfill" else "")}

    days = [f"2026-09-{i:02d}" for i in range(1, 10)]
    rows = []
    for i, d in enumerate(days):
        rows.append(mk(75, d, "live", 1.0 + (i % 3) * 0.3))
        rows.append(mk(65, d, "backfill", -0.5 + (i % 2) * 0.2))
    out = calibrate.source_section(rows)
    assert "backfill" in out and "live" in out
    assert "缺失罚" in out and "不可直接比较" in out
    # 门槛敏感性：覆盖 66~74，现行门槛 70 必须标出
    for th in ("66", "68", "70", "72", "74"):
        assert th in out, th
    assert "←现行" in out
    # 旧 schema（无 fillable/net_ret）→ 降级为破折号，不崩、不当成 0
    legacy = [{"board_score": s, "date": d, "next_open_pct": 1.0,
               "next_close_pct": 2.0, "max3_pct": 4.0, "promoted": 1}
              for s, d in ((75, "2026-09-01"), (65, "2026-09-02"))]
    out2 = calibrate.source_section(legacy)
    assert "—" in out2 and "live*" in out2, out2
    print("PASS source stratification")


def test_calibrate_features():
    """维度特征提取：数值列 + board_note 正则（离线）。"""
    import calibrate
    r = {"ltsz": 2.7e9, "close": 3.63, "turnover": 0.08, "zhaban_times": 0,
         "board_note": "题材:软件开发5家·最高板·发酵 | 首板:0938封板·实体·昨换4%·昨封0.2 | "
                       "二板:准一字·无炸板 | 情绪:炸板19%·晋级25%·赚钱效应未知 | 辨识:流通27亿小盘·3.63元深低价"}
    f = calibrate.build_features(r)
    assert f["小盘<30亿"] and f["低价<10元"] and f["换手5-20%"] and f["无炸板"]
    assert f["首板早封≤1000"] and not f["首板晚封>1130"]
    assert f["准一字开盘"] and f["题材发酵"] and not f["题材退潮"]
    assert f["昨封弱<0.5"] and not f["昨封强≥2"]
    # note 变体：高开优 / 退潮 / 缺失
    r2 = {"ltsz": "", "close": 12.0, "turnover": 0.03, "zhaban_times": 2,
          "board_note": "题材:种植业3家·跟风板·退潮 | 首板:1436晚封 | 二板:高开+4.7%优·炸2次 | 数据:缺失5项(-10)"}
    f2 = calibrate.build_features(r2)
    assert f2["题材退潮"] and not f2["题材发酵"]
    assert f2["高开3-7%优"] and f2["换手<5%"] and f2["有炸板"]
    assert f2["首板晚封>1130"] and f2["数据缺失≥3"]
    # 情绪周期特征：优先取列，列缺时回退 note；六阶段互斥（对照组非空）
    f3 = calibrate.build_features(dict(r2, cycle_stage="高潮"))
    assert f3["周期·高潮"] and not f3["周期·分歧"] and "周期·冰点" in f3
    f4 = calibrate.build_features(dict(r2, board_note="情绪:炸板19%·高潮期"))
    assert f4["周期·高潮"]
    # 二板日封成比分档（CSV 列直取；0/缺失不出特征）
    assert calibrate.build_features(dict(r2, seal_ratio=2.4))["今封≥0.5"]
    assert not calibrate.build_features(dict(r2, seal_ratio=0.2))["今封≥0.5"]
    assert "今封≥0.5" not in calibrate.build_features(dict(r2))
    # 星期特征：列直取 / 按日期推导，五日互斥
    f5 = calibrate.build_features(dict(r2, weekday="周四"))
    assert f5["周四"] and not f5["周一"]
    f6 = calibrate.build_features(dict(r2, date="2026-08-28"))   # 周五推导
    assert f6["周五"] and not f6["周四"]

    # _pool_raw 进程内记忆化：同 (kind,date) 二次调用不再发请求
    import datasources.pools as pools_mod
    calls = {"n": 0}
    orig_http = pools_mod._http_get_json

    def fake_http(url, params, referer, timeout=6, retries=None):
        calls["n"] += 1
        return {"pool": [{"c": "600000", "n": "x", "p": 10000, "hs": 1, "lbc": 2,
                          "fbt": 93000, "hybk": "T", "zdp": 10, "amount": 1}]}

    pools_mod._http_get_json = fake_http
    pools_mod._POOL_MEMO.clear()
    try:
        a1 = pools_mod._pool_raw("zt", "20260901")
        a2 = pools_mod._pool_raw("zt", "20260901")
        assert a1 is a2 and calls["n"] == 1          # 命中记忆，未重复请求
        pools_mod._pool_raw("zt", "20260902")
        assert calls["n"] == 2
    finally:
        pools_mod._http_get_json = orig_http
        pools_mod._POOL_MEMO.clear()
    print("PASS calibrate features")


def test_calibrate_oos():
    """样本外检验：四态判读（✔保持 / ✘消失 / ◐正但不显著 / ⚠簇数不足）+ 等待。

    旧判据是 `keep = pool > base` 两个点估计比大小，OOS 只有 8 个独立交易日时
    把「不显著」硬判成保持或消失都是错的。新判据主口径＝可成交净收益，且必须
    过 MIN_INFER_DAYS 个**配对交易日**才给结论。
    """
    import calibrate

    def row(d, s, p, net, ow=False):
        return {"date": d, "board_score": s, "promoted": p,
                "next_open_pct": 1.0, "next_close_pct": 2.0, "max3_pct": 4.0,
                "one_word": int(ow), "fillable": int(not ow),
                "tradable_ret": net + 0.152, "net_ret": net}

    days = [f"2026-09-{i:02d}" for i in range(1, 11)]      # 10 个 OOS 交易日 > 8 门槛
    # ⚠ 陷阱：日度差值若恒定则方差 0 → se=0 → t=None，会掉进「簇数不足」而不是
    # 预期的 ✔/✘。每组净收益必须带日间波动。
    keep_rows = [row("2026-08-20", 75, 1, 1.0)]            # 拟合窗口内，应被排除
    for i, d in enumerate(days):
        keep_rows.append(row(d, 75, 1, 2.0 + (i % 3) * 0.2))
        keep_rows.append(row(d, 65, 0, 0.0 + (i % 2) * 0.1))
    out = calibrate.oos_section(keep_rows)
    assert "样本外检验" in out and "2026-08-27" in out      # 拟合窗口标注
    assert "10 个独立交易日" in out, out
    assert "✔ 保持" in out, out

    # ✘ 消失：过线组可成交净收益低于未过线组 → 差 ≤ 0，按预注册应回退
    gone = []
    for i, d in enumerate(days):
        gone.append(row(d, 75, 0, -1.0 + (i % 3) * 0.2))
        gone.append(row(d, 65, 1, 0.5 + (i % 2) * 0.1))
    out2 = calibrate.oos_section(gone)
    assert "✘ 消失" in out2 and "回退" in out2, out2

    # ◐ 方向为正但不显著。构造可手算：过线组净收益在 0/1 间交替（均值 0.5），
    # 未过线组恒定 0.45 → 日度差 −0.45/+0.55 交替，均值 +0.05、sd≈0.527、
    # se≈0.167、t≈0.30，方向为正但远不显著。
    weak = []
    for i, d in enumerate(days):
        weak.append(row(d, 75, 1, 0.0 if i % 2 else 1.0))
        weak.append(row(d, 65, 0, 0.45))
    out3 = calibrate.oos_section(weak)
    assert "◐" in out3 and "不显著" in out3, out3

    # ⚠ 簇数不足：只有 2 个 OOS 交易日 → 不得给 ✔/✘/◐ 任何结论
    thin = [row("2026-09-01", 75, 1, 2.0), row("2026-09-02", 65, 0, 0.0)]
    out4 = calibrate.oos_section(thin)
    assert "簇数不足" in out4, out4
    for bad in ("✔ 保持", "✘ 消失", "◐"):
        assert bad not in out4, f"簇数不足时不该出现 {bad}"

    # 无样本外数据 → 等待提示，不崩溃
    out5 = calibrate.oos_section([row("2026-08-20", 75, 1, 1.0)])
    assert "尚无样本外" in out5
    print("PASS calibrate oos")


def test_mood_stages():
    """情绪周期判定：六阶段 + 序列不足/未知路径（离线纯函数）。"""
    import mood

    def d(h, promo=None, brk=None, money=None):
        return {"height": h, "promo": promo, "break_rate": brk, "money": money}

    # 退潮：高度 7→4 断板
    s, w = mood.classify_stage([d(7), d(6), d(4)])
    assert s == "退潮" and "断板" in w
    # 冰点：高度2 且晋级率 10%
    s, _ = mood.classify_stage([d(3, 0.3), d(2, 0.10)])
    assert s == "冰点"
    # 高潮：高度 8 创新高、炸板 25%
    s, w = mood.classify_stage([d(6, 0.3, 0.3), d(8, 0.4, 0.25)])
    assert s == "高潮" and "创新高" in w
    # 发酵：高度自低位抬升（5 > 前4.5均）但未创序列新高（前高6）
    s, w = mood.classify_stage([d(6, 0.2, 0.3), d(3, 0.2, 0.3),
                                d(5, 0.30, 0.3, 1.0)])
    assert s == "发酵" and "抬升" in w
    # 回暖：赚钱转正但高度未抬升
    s, _ = mood.classify_stage([d(4, 0.2, 0.4), d(4, 0.2, 0.4, 1.5)])
    assert s == "回暖"
    # 分歧：其余（高度滞涨、无赚钱数据）
    s, _ = mood.classify_stage([d(4, 0.2, 0.4), d(4, 0.2, 0.4)])
    assert s == "分歧"
    # 退潮（亏钱效应路径）：高度稳但 -3%
    s, _ = mood.classify_stage([d(5, 0.2, 0.3), d(5, 0.2, 0.3, -3.0)])
    assert s == "退潮"
    # 序列不足 → 未知
    s, w = mood.classify_stage([d(5)])
    assert s == "未知" and "序列不足" in w
    # 阶段→D槽分/仓位建议映射完整
    for st in ("发酵", "高潮", "回暖", "分歧", "冰点", "退潮", "未知"):
        assert st in mood.STAGE_D_SCORE and st in mood.STAGE_ADVICE

    # build_series 首日「昨日池」锚点：须取窗口前一交易日（cal[-lookback-1]），
    # 此前误取 cal[-1]（未来日）→ 序列首日 promo 分母错位污染 cycle_series.csv
    dates = [dt.date(2026, 9, d) for d in (7, 8, 9, 10, 11, 14, 15, 16, 17, 18)]
    asked: list[str] = []

    def fake_pool_raw(kind, ymd):
        asked.append((kind, ymd))
        return []                       # 全空池 → day_metrics 走默认，只验日期

    orig = (mood.data_source.pool_raw, mood.data_source.parse_pool)
    mood.data_source.pool_raw = fake_pool_raw
    mood.data_source.parse_pool = lambda kind, raw: raw or []
    try:
        mood.build_series(dt.date(2026, 9, 18), dates, lookback=3)
        zt_days = [ymd for k, ymd in asked if k == "zt"]
        # 首个回看日 09-15 的「昨日」必须是 09-14（窗口前一交易日），而非 09-17
        assert "20260914" in zt_days, zt_days
    finally:
        mood.data_source.pool_raw, mood.data_source.parse_pool = orig
    print("PASS mood stages")


def test_concept_boards_parse():
    """概念板块快照解析（mock 行情源）：拆包曾漏 _num/_to_int 导入致解析必崩。"""
    import datasources.concepts as con

    def fake_http(url, params, referer, timeout=8, retries=None):
        # 首页返回2个板块 + total=2（不分页）；第二次调用（slist 等）不会发生
        return {"total": 2, "diff": [
            {"f12": "BK0001", "f14": "固态电池", "f3": 4.21, "f104": 70,
             "f105": 10, "f128": "测试龙头"},
            {"f12": "BK0002", "f14": "储能", "f3": 0.8, "f104": 40,
             "f105": 30, "f128": "别的股"},
        ]}

    orig = con._http_get_json
    con._http_get_json = fake_http
    try:
        boards = con.fetch_concept_boards()
    finally:
        con._http_get_json = orig
    assert boards and len(boards) == 2
    b = boards["固态电池"]
    assert b["pct"] == 4.21 and b["up"] == 70 and b["down"] == 10
    assert b["lead"] == "测试龙头" and b["bk"] == "BK0001"
    print("PASS concept boards parse")


def test_ledger_health():
    """信号健康度判读：✓正常 / ⚠偏离预警 / ⚠簇数不足 / 空台账 + pnl_net 优先。"""
    import calibrate as cal
    tmp = "out_health_test"
    os.makedirs(os.path.join(tmp, "archive"), exist_ok=True)
    cols = ["signal_date", "code", "name", "tier", "buy_date", "buy_open",
            "status", "sell_date", "sell_open", "pnl"]

    def write(rows, columns=None):
        with open(os.path.join(tmp, "archive", "open_pool_ledger.csv"),
                  "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns or cols)
            w.writeheader()
            w.writerows(rows)

    def mk(n, pnl, same_day=False, extra=None):
        """n 笔已平仓；same_day=True 时全部挤在同一个 signal_date。"""
        out = []
        for i in range(n):
            r = {"signal_date": "2025-06-01" if same_day else f"2025-06-{i + 1:02d}",
                 "code": f"6000{i:02d}", "name": "A", "tier": "0",
                 "buy_date": "2025-06-02", "buy_open": "10", "status": "bought",
                 "sell_date": "2025-06-03", "sell_open": "10.3", "pnl": pnl}
            r.update(extra or {})
            out.append(r)
        return out

    # ✓ 正常：8 个**不同** signal_date，均值 +2.9% 高于史测 0.31%
    write(mk(8, 0.029))
    body = cal.ledger_section(tmp)
    assert "✓" in body and "正常" in body, body
    assert "8笔/8个交易日" in body, body

    # ⚠ 偏离预警：同样 8 个日期，均值 -3% → dev=-3.31pp ≤ -2pp 阈值
    write(mk(8, -0.03))
    body = cal.ledger_section(tmp)
    assert "⚠" in body and "降仓/停用" in body, body

    # ⚠ 簇数不足：8 笔但全在同一天。旧实现数够 8 笔就判 ✓/⚠——
    # 实测台账 59 笔只跨 4 个 signal_date、其中 30 笔挤在同一天，笔数不是样本量。
    write(mk(8, -0.03, same_day=True))
    body = cal.ledger_section(tmp)
    assert "无法推断" in body and "8笔/1个交易日" in body, body
    assert "不是独立样本" in body, body
    assert "降仓/停用" not in body, "簇数不足时不得下达降仓指令"

    # pnl_net 优先：同时有两列时必须用扣成本的那个（0.029 → 净 -0.03 应触发预警）
    write(mk(8, 0.029, extra={"pnl_net": -0.03}), cols + ["pnl_net"])
    body = cal.ledger_section(tmp)
    assert "均值-3.00%" in body, body
    assert "降仓/停用" in body, "未优先使用 pnl_net（扣成本口径）"

    # 仅 2 笔 → 仍是簇数不足，且不得出现已废弃的「还需N笔」文案
    write(mk(2, -0.03))
    body = cal.ledger_section(tmp)
    assert "无法推断" in body and "还需" not in body, body
    # 空目录
    shutil.rmtree(tmp, ignore_errors=True)
    tmp2 = "out_health_empty"
    os.makedirs(tmp2, exist_ok=True)
    assert "台账尚未建立" in cal.ledger_section(tmp2)
    shutil.rmtree(tmp2, ignore_errors=True)
    print("PASS ledger health")


def test_report_plain_file():
    """落盘纯文本 / 返回值彩色：同一份报告两种形态，ANSI 码不得写进文件。

    必须强制开启颜色才测得到——测试环境下 stdout 不是 tty，termstyle 默认关闭着色，
    不 monkeypatch 就永远测不出「ANSI 被写进文件」这个回归（实测 143 行里 94 行带 ESC）。
    """
    import termstyle

    def mk(code, name, score, note=""):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": 10.1, "prev_close": 10.0, "close": 11.0,
                "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08,
                "ok": True, "skip_reason": "", "lianban": 2, "zhaban_times": 0,
                "industry": "PCB", "fbt": "0931", "fund": 1e8, "seal_ratio": 0.5,
                "ltsz": 4e9, "board_pass": score >= 70, "board_score": score,
                "board_note": note, "vol_ratio": 1.0}

    results = [mk("600001", "二板龙头", 82, "题材:独苗 | 首板:0931封板 | 二板:高开+5.0%优")]
    tmp = "out_plain_test"
    orig = termstyle._ENABLED
    termstyle._ENABLED = True
    try:
        txt_path, _, colored = rpt.write_report(
            results, [], results, dt.date(2025, 6, 10), tmp,
            signal_only=True, ctx={"cycle_stage": "发酵"})
        body = open(txt_path, encoding="utf-8").read()
        assert "\x1b" not in body, "报告文件不得含 ANSI 转义码（编辑器/grep 不可读）"
        assert "\x1b" in colored, "返回值应保留彩色供终端打印"
        assert termstyle.strip_ansi(colored) == body, "剥码后必须与落盘内容逐字符相等"
        arch = open(os.path.join(tmp, "archive", "report_20250610.txt"),
                    encoding="utf-8").read()
        assert "\x1b" not in arch, "归档报告同样不得含 ANSI 码"
        # 表头框线四边等显示宽度：中文按 2 列显示、按 1 字符计数，
        # 旧版手工数空格导致「生成 22:19    ║」突出框外
        # 注意用元组精确匹配：`l[:1] in "╔║╠╚"` 对空行也成立（空串是任意串的子串）
        box = [l for l in body.splitlines() if l[:1] in ("╔", "║", "╠", "╚")]
        assert len(box) == 5, box
        widths = {termstyle.disp_width(l) for l in box}
        assert len(widths) == 1, f"框线不等宽: {[(l, termstyle.disp_width(l)) for l in box]}"
    finally:
        termstyle._ENABLED = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS report plain file")


def test_name_normalization():
    """股票名内部空格归一：东财把 3 字名存成「南 京 港」。

    内部空格会被算进显示宽度，破坏所有按 pad 对齐的名称列，还会写进归档 CSV
    长期污染数据。两条路径都要覆盖：新数据在 candidates 入口归一
    （strip() 去不掉内部空格），历史脏数据在 ledger._load 读取入口归一
    （一处修好即覆盖全部 8 个渲染点）。
    """
    import candidates

    spot = pd.DataFrame([
        dict(code="600000", name="南 京 港", close=11.00, high=11.00, low=10.10,
             prev_close=10.0, pct=10.0, amount=1e8, turnover=5.0),
        dict(code="600001", name="  远 望 谷 ", close=11.00, high=11.00, low=10.10,
             prev_close=10.0, pct=10.0, amount=1e8, turnover=5.0),
        dict(code="600002", name="浦发银行", close=11.00, high=11.00, low=10.10,
             prev_close=10.0, pct=10.0, amount=1e8, turnover=5.0),
    ])
    got = {c["code"]: c["name"] for c in candidates.build_candidates(spot)}
    assert got.get("600000") == "南京港", got
    assert got.get("600001") == "远望谷", got        # 首尾与内部空格一并去掉
    assert got.get("600002") == "浦发银行", got      # 正常名称不受影响

    tmp = "out_name_test"
    os.makedirs(os.path.join(tmp, "archive"), exist_ok=True)
    cols = ["signal_date", "code", "name", "tier", "buy_date", "buy_open",
            "status", "sell_date", "sell_open", "pnl"]
    p = os.path.join(tmp, "archive", "open_pool_ledger.csv")
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"signal_date": "2026-09-01", "code": "002040",
                    "name": "南 京 港", "tier": "0", "buy_date": "2026-09-02",
                    "buy_open": "10", "status": "bought", "sell_date": "",
                    "sell_open": "", "pnl": ""})
    rows = ledger._load(p, cols)
    assert rows and rows[0]["name"] == "南京港", rows
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS name normalization")


def test_wrap_helpers():
    """折行工具回归守卫：不得丢字、不得重复、不得劈开词组、不得超宽。"""
    import termstyle as ts

    assert ts.wrap("短文本", 96) == ["短文本"]          # 不超宽原样返回

    txt = ("旧规则期 35笔/3日 均值-1.11% 中位-5.14% 胜26% 净值-5.91% 回撤-7.4% "
           "另附一段很长的说明文字用来触发折行，确认不会丢字也不会重复。")
    out = ts.wrap(txt, 60, "  ")
    assert len(out) > 1, out
    for ln in out:
        assert ts.disp_width(ln) <= 60, (ts.disp_width(ln), ln)
    # 去掉缩进与空格后必须还原原文的非空白字符序列——既查丢失也查**重复**。
    # 曾经的 bug：空格断行分支 append 了整个 cur（已含尾巴），尾巴又被带到下一行，
    # 于是「按日等权净值」印了两遍。财务报告里重复数字是严重错误。
    joined = "".join(out).replace(" ", "")
    assert joined == txt.replace(" ", ""), (joined, txt)
    assert joined.count("旧规则期") == 1, "折行不得重复文字"

    # wrap_segs：尽量不在段中间断开
    segs = ["题材:持续题材·电力5家·最高板·发酵", "首板:1054封板·实体·昨封0.1",
            "二板:高开+2.6%·无炸板", "辨识:流通58亿"]
    out2 = ts.wrap_segs(segs, 60, "└ ", "     ")
    assert all(ts.disp_width(l) <= 60 for l in out2), out2
    flat = "".join(out2)
    for sg in segs:
        assert sg in flat, f"段被劈开: {sg}"
    # 首行前缀必须保留，且第一个段前面**不能**多出分隔符
    # （曾用 cur.strip() 判断是否已装段，传了 first_prefix 就两个 bug 一起犯：
    #   先是首段前多一个「│」，改成 added 标志后又把前缀整个丢了）
    assert out2[0].startswith("└ " + segs[0]), out2[0]
    assert "└  │" not in flat and "└ │" not in flat, flat

    # dim_wrap：着色不得改变文本内容，且同样不超宽
    long_note = "数学终审（五R）：12 个月 t=-0.19 对称硬币，130 个条件格无一通过多重比较"
    out3 = ts.dim_wrap(long_note, "  ", 60)
    assert all(ts.disp_width(ts.strip_ansi(l)) <= 60 for l in out3), out3
    assert "".join(ts.strip_ansi(l) for l in out3).replace(" ", "") \
        == long_note.replace(" ", "")
    print("PASS wrap helpers")


def test_stats_line_daily_equal_weight():
    """_stats_line 按日等权，不得把同日并发仓当串行全仓复利。"""
    import termstyle

    # 同日 2 只 +5%/−5% → 当日组合 0%，净值 0%。
    # 旧版逐笔连乘会得 1.05×0.95−1 = −0.25%（103 笔时输出过「复合−80.21%」）
    same_day = [{"pnl": 0.05, "sell_date": "2026-09-01"},
                {"pnl": -0.05, "sell_date": "2026-09-01"}]
    # 长标签 + 多日：用来验证折行后仍不超宽（单行实测可达 126 列）
    cross_day_wide = [{"pnl": 0.05 + i * 0.001, "sell_date": f"2026-09-{i+1:02d}"}
                      for i in range(9)]
    out = "\n".join(ledger._stats_line(same_day, "测试"))
    assert "2笔/1日" in out, out
    assert "均值+0.00%" in out and "中位+0.00%" in out, out
    assert "净值+0.00%" in out, out
    assert "回撤0.0%" in out, out
    assert "复合" not in out, "不得再出现把并发仓当串行复利的「复合」"
    # 返回值必须是**已折行**的列表：单行实测 106~126 列，会撑破 78 列版面
    assert isinstance(ledger._stats_line(same_day, "测试"), list)
    for ln in ledger._stats_line(cross_day_wide, "旧规则期累计（tier0-only 之前）"):
        assert termstyle.disp_width(ln) <= 100, (termstyle.disp_width(ln), ln)

    # 跨日才应复利：D1 +5%、D2 −5% → 1.05×0.95−1 = −0.25%，峰值 1.05 → 回撤 −5.0%
    cross_day = [{"pnl": 0.05, "sell_date": "2026-09-01"},
                 {"pnl": -0.05, "sell_date": "2026-09-02"}]
    out2 = "\n".join(ledger._stats_line(cross_day, "测试"))
    assert "2笔/2日" in out2, out2
    assert "净值-0.25%" in out2, out2
    assert "回撤-5.0%" in out2, out2
    # 右偏分布必须同时给中位数：均值被少数大赚拉高（fills.csv 均值+0.39%/中位−1.27%）
    skewed = [{"pnl": 0.10, "sell_date": "2026-09-01"},
              {"pnl": -0.02, "sell_date": "2026-09-02"},
              {"pnl": -0.02, "sell_date": "2026-09-03"}]
    out3 = "\n".join(ledger._stats_line(skewed, "测试"))
    assert "均值+2.00%" in out3 and "中位-2.00%" in out3, out3
    # 缩进统一：三组统计不得一组有缩进、另两组顶格（旧版 .strip() 造成的层级错觉）
    for ln in ledger._stats_line(cross_day, "观察档累计", style=termstyle.dim):
        assert termstyle.strip_ansi(ln).startswith("  "), ln
    print("PASS stats line daily equal weight")


def test_report_summary():
    """报告头部决策摘要：三行速读与区块数据一致。"""
    def mk(code, name, open_, prev, close, vr, lb=1):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": open_, "prev_close": prev, "close": close,
                "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08,
                "ok": True, "skip_reason": "", "lianban": lb, "zhaban_times": 0,
                "industry": "", "fbt": "", "fund": float("nan"),
                "seal_ratio": float("nan"), "ltsz": float("nan"),
                "board_pass": False, "board_score": 0, "board_note": "",
                "vol_ratio": vr}
    results = [mk("600001", "池票A", 10.1, 10.0, 11.0, 1.0, lb=2)]
    tmp = "out_summary_test"
    txt, _, _ = rpt.write_report(results, [], results, dt.date(2025, 6, 10), tmp,
                              signal_only=True, ctx={"cycle_stage": "发酵"})
    body = open(txt, encoding="utf-8").read()
    i_sum = body.index("明日决策摘要")
    i_speed = body.index("市场情绪速览")
    assert i_sum < i_speed                          # 摘要在速览之前
    seg = body[i_sum:i_speed]
    # 双轨分离：信息轨（评分识别龙头）与行动轨（可执行清单）必须分开出现
    assert "信息轨：二板定龙头" in seg
    assert "行动轨：打板画像" in seg
    assert "非建仓指令" in seg
    assert "五R 判死" in seg                       # 开盘池已降级，摘要须如实标注
    assert "发酵期·建议仓位 50%" in seg
    # 版面守卫：报告分隔线 78 列，任何行都不该严重溢出。
    # 用 disp_width 而非 len——中文占 2 列。瘦身前实测最宽 220 列（五维明细行）。
    import termstyle as _ts
    wide = [(n, _ts.disp_width(l), l[:40]) for n, l in enumerate(body.splitlines(), 1)
            if _ts.disp_width(l) > 100]
    assert not wide, f"有 {len(wide)} 行超过 100 显示列: {wide[:3]}"
    # 回归守卫：旧摘要把打板画像写成「槽位①(主)…+4.2%/84%」，该数字与 config 的
    # +4.69%、board_rules 的 +4.43%/+4.7% 互相矛盾。期望值一律走 config.EXPECT，
    # 渲染层不得再写字面量。
    assert "+4.2%/84%" not in seg
    assert "槽位①" not in seg
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS report summary")


def test_no_lookahead_fields():
    """无未来函数守卫：决策规则函数只准访问收盘可知字段（白名单）。

    白名单 = 信号日收盘时点已确定的数据。任何新字段加入规则前必须先确认
    其在当日 15:00 收盘时可知，否则属于未来函数（回测虚高、实盘亏损）。
    """
    import ast as _ast
    import io as _io
    ALLOWED = {
        # 当日收盘可知：行情/池/评分/环境
        "zt_status", "vol_ratio", "open", "prev_close", "close", "pct", "amp",
        "turnover", "amount", "lianban", "zhaban_times", "fbt", "fbt_y",
        "fund", "seal_ratio", "seal_ratio_y", "ltsz", "industry", "name",
        "code", "board_pass", "board_score", "board_note", "concepts", "skip_reason", "pools",
        # 台账内部字段（登记/结算时写入，非行情；结算用的 T+1/T+2 行情
        # 在对应交易日运行时才读——与 sell_date/sell_open 同类）
        "signal_date", "buy_date", "buy_open", "buy_price", "status",
        "sell_date", "sell_open", "pnl", "tier", "vr",
        "fill", "fill_date", "pnl_net",
        # 市场环境键（今日收盘可算：池统计/昨池对照/周期判定）
        "break_rate", "promotion_rate", "yzt_ret", "cycle_stage",
        "industry_count", "industry_max_lianban", "industry_yesterday_count",
        "persistent_industries", "concept_boards", "yesterday_fbt",
        "yesterday_turnover", "yesterday_seal", "zb_pool_ok", "zt_pool_ok",
        "_board_kline", "_stock_concepts",
        # 概念板块快照内部键（今日盘后）
        "up", "down", "lead", "pct", "bk",
        "yzt_today_pct", "cycle_note", "market_max_lianban",
        "lianban_ladder", "today_2plus_count", "yesterday_zt_count",
        "height", "zt_count", "promo", "money", "date",
        "fill",   # 历史补录标记（build_candidates_from_pools 写入，非行情）
        "source",  # 归档行血缘标记（live 实跑 / backfill 回灌），供 calibrate 分层，
                   # 由写入方生成该行时确定，非行情、不参与选股判定
        "trade_date", "weekday",  # 归档行标识列（记录性，非行情）
        "high", "low", "volume",   # 今日 spot 的 OHLCV（收盘可知）
        "results", "spot", "bars", # 滚动缓存结构键（数据装配，非决策字段）
        "_slam_boost", "_otier", "_wtier",  # 运行时标注键（由收盘可知字段派生）
        "best_concept",  # analyzer 富集时算好的最热概念名（概念板块当日涨幅派生，
                         # 收盘可知；second_board 消费以免对同一股重复求 max）
        # 紫阳甄选台账键（ziyang 模块，2026-09-17 接入关注清单）：甄选行由
        # 信号日**收盘数据 + 当晚龙虎榜**生成（全部在 T+1 开盘前可知），
        # 只参与次日关注排序，不含任何 T+1 信息——非未来函数
        "zy_rank", "zy_tags", "zy_score",
        "rank", "score", "eve_tags",   # ziyang 台账原始列（同上，盘后生成）
        # 静态统计表 config.EXPECT 与 out/archive/bucket_stats.json 的**模式键**：
        # 它们是表结构，不是行情字段也不是决策输入。EXPECT 在 config.py 里硬编码、
        # 收盘前后完全一样；bucket_stats 由 calibrate 基于**历史已实现样本**算出，
        # report._detail 只拿它做「同档实测」标注，不参与任何当日选股判定，
        # 因此与本守卫要防的「用当日收盘不可知的数据做决策」无关。
        # 显式归类而非改用下标语法绕过守卫——绕过会让下次新增读取时重复误报。
        # config.EXPECT
        "prereg", "current", "verdict", "fill_rate", "median", "t", "rule_split",
        "days", "net", "fillable_rate",                    # bucket_stats.json
    }
    TARGETS = {"board_rules.py": None, "second_board.py": None,
               "report.py": None, "ledger.py": None}
    for fname in TARGETS:
        tree = _ast.parse(_io.open(fname, encoding="utf-8").read())
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.FunctionDef):
                continue
            for n in _ast.walk(node):
                if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                        and n.func.attr == "get" and n.args
                        and isinstance(n.args[0], _ast.Constant)
                        and isinstance(n.args[0].value, str)
                        and n.args[0].value not in ALLOWED):
                    raise AssertionError(
                        f"{fname}:{node.name} 访问了未知字段 "
                        f"'{n.args[0].value}'——若为收盘可知请加入白名单并注明来源，"
                        f"否则即未来函数")


def test_module_contracts():
    """模块连接契约：门面导出完整性 + ctx 键生产/消费匹配 + 子包无横向私有依赖。

    这是「数据源管道与模块连接是否流畅」的常驻自动化验证：
    - 任何业务模块经 data_source 用到的名字必须由门面导出；
    - second_board 消费的每一个 ctx 键必须有生产者（防键名漂移导致静默中性分）；
    - datasources 子包内部禁止横向私有导入（concepts→pools._x 类耦合禁止复发）；
    - 业务层禁止绕过门面直接 import datasources 子模块。
    """
    import ast
    import io
    import os
    import re

    import data_source

    # 1) 门面导出完整性
    used = set()
    for f in ("main.py", "work.py", "screener.py", "mood.py", "backfill.py",
              "calibrate.py", "report.py", "second_board.py",
              "board_rules.py", "ledger.py",
              "pipeline.py", "candidates.py", "bars.py", "analyzer.py"):
        src = io.open(f, encoding="utf-8").read()
        used |= set(re.findall(r"data_source\.([A-Za-z_][A-Za-z0-9_]*)", src))
    missing = [n for n in sorted(used) if not hasattr(data_source, n)]
    assert not missing, f"门面缺失导出: {missing}"

    # 2) ctx 契约：second_board 消费的键 ⊆ 生产者产出的键
    producers = set()
    for f in ("datasources/pools.py", "mood.py", "main.py", "screener.py",
              "pipeline.py", "analyzer.py"):
        src = io.open(f, encoding="utf-8").read()
        producers |= set(re.findall(r'ctx\["([A-Za-z_][A-Za-z0-9_]*)"\]\s*=\s', src))
        producers |= set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)":', src))
    for mod in ("second_board",):
        m_src = io.open(mod + ".py", encoding="utf-8").read()
        consumers = set(re.findall(r'ctx\.get\("([A-Za-z_][A-Za-z0-9_]*)"\)', m_src))
        consumers |= set(re.findall(r'ctx\["([A-Za-z_][A-Za-z0-9_]*)"\]', m_src))
        leaked = consumers - producers - {"_stock_concepts", "_board_kline"}  # 运行时缓存键由 screener 注入
        assert not leaked, f"{mod} 消费了无人生产的 ctx 键: {leaked}"

    # 3) datasources 子包禁止横向私有导入（http/util 是允许的底层）
    for mod in os.listdir("datasources"):
        if not mod.endswith(".py") or mod == "__init__.py":
            continue
        tree = ast.parse(io.open(os.path.join("datasources", mod), encoding="utf-8").read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and node.module not in ("http", "util", "config", "cache_store")):
                for a in node.names:
                    assert not a.name.startswith("_"),                         f"datasources/{mod} 横向私有导入 {node.module}.{a.name}"

    # 4) 业务层禁止绕过门面
    for f in ("main.py", "work.py", "screener.py", "mood.py", "backfill.py",
              "calibrate.py", "report.py", "second_board.py",
              "pipeline.py", "candidates.py", "bars.py", "analyzer.py"):
        src = io.open(f, encoding="utf-8").read()
        assert "from datasources" not in src and "import datasources." not in src,             f"{f} 绕过 data_source 门面直接依赖子包"

    # 5) 分层方向：只允许向低层导入（域内白名单除外）
    LAYER = {"candidates": 1, "bars": 1,                 # 纯数据层
             "mood": 2, "second_board": 2,             # 领域层
             "board_rules": 2,                           # 选股规则层
             "analyzer": 3, "report": 3, "ledger": 3,    # 编排/展示/台账层
             "pipeline": 4, "main": 5, "work": 5}        # 流水线/入口层
    SAME_LAYER_OK = {("second_board", "mood"),           # 评分读取周期分值表
                     # C 维高开缺口打分的单一实现归 board_rules（历史教训：
                     # 两处各自实现同一常量曾给出相反极性）
                     ("second_board", "board_rules")}
    for mod, layer in LAYER.items():
        if mod == "screener":
            continue
        src = io.open(mod + ".py", encoding="utf-8").read()
        for dep in re.findall(r"from (pipeline|analyzer|candidates|bars|second_board|mood|board_rules|report|main|work) import", src):
            if dep == mod:
                continue
            assert LAYER.get(dep, 0) < layer or (mod, dep) in SAME_LAYER_OK,                 f"{mod}(L{layer}) -> {dep}(L{LAYER.get(dep, '?')}) 违反分层单向规则"
    # 6) calib 包（横切研究工具）：禁止依赖生产 L3+ 模块与绕过门面访问子包。
    #    calib 只准用 data_source 门面 / trade_rules / config / calib 内部——
    #    它是 calibrate 的实现体，不是生产链一环，混进 report/ledger/pipeline
    #    依赖会让研究工具与生产展示层双向纠缠。
    for modname in os.listdir("calib"):
        if not modname.endswith(".py") or modname == "__init__.py":
            continue
        csrc = io.open(os.path.join("calib", modname), encoding="utf-8").read()
        assert "from datasources" not in csrc and "import datasources." not in csrc,                 f"calib/{modname} 绕过 data_source 门面直接依赖子包"
        for dep in re.findall(r"from (pipeline|analyzer|report|ledger|main|work) import", csrc):
            raise AssertionError(f"calib/{modname} 依赖生产 L3+ 模块 {dep}（研究工具不得进生产链）")
    print("PASS module contracts")


def test_trade_rules():
    """排板成交模型 + 卖出规则（纯函数，合成K线）。"""
    import trade_rules as tr
    n = 12
    close = [10.0] * n
    close[8], close[9] = 11.0, 12.1          # 8=首板, 9=二板(事件日)
    close[10] = 13.31                        # T+1 封死(12.1*1.1)
    close[11] = 14.64                        # T+2 再封板 → R1 持有到 T+3 开盘卖
    opn = [9.9] * n
    opn[10] = 12.5                           # T+1 高开但非一字
    opn[11] = 14.64                          # T+2 一字开盘
    high = [c + 0.1 for c in close]
    high[10] = 13.31                         # T+1 盘中触涨停(回封成交)
    df = pd.DataFrame({
        "date": [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(n)],
        "open": opn, "close": close, "high": high, "low": [c - 0.1 for c in close],
        "volume": 1000.0,
    })
    fill = tr.simulate_fill(df, 9)
    assert fill["fill"] and not fill["one_word_open"] and fill["sealed_close"]
    ex = tr.simulate_exit(df, 9, fill)
    assert ex["rule"] == "R1封板持有"
    assert abs(ex["ret"] - (14.64 / 13.31 - 1)) < 1e-9 or ex["exit_idx"] == 11

    # R2：T+1 炸板 → T+1 不可卖（A 股 T+1），T+2 开盘卖
    df2 = df.copy()
    df2.loc[10, "close"] = 12.5
    df2.loc[11, "open"] = 11.8                     # T+2 低开
    fill2 = tr.simulate_fill(df2, 9)
    assert fill2["fill"] and not fill2["sealed_close"]
    ex2 = tr.simulate_exit(df2, 9, fill2)
    assert ex2["rule"] == "R2炸板T+2卖"
    assert abs(ex2["ret"] - (11.8 / tr.limit_price(12.1) - 1)) < 1e-9

    # R3：一字开盘不成交
    df3 = df.copy()
    df3.loc[10, "open"] = 13.31
    fill3 = tr.simulate_fill(df3, 9)
    assert not fill3["fill"] and fill3["one_word_open"]
    assert tr.simulate_exit(df3, 9, fill3)["rule"] == "R3未成交"

    # STAGE_POSITION 完整性
    import mood
    for st in ("高潮", "发酵", "回暖", "分歧", "冰点", "退潮", "未知"):
        assert 0 <= mood.STAGE_POSITION[st] <= 1
    print("PASS trade rules")


def test_end_to_end_report():
    tmp_dir = "out_test"
    # mock 掉网络：合成 60 根上行日K（昨日+今日双涨停结构）
    def fake_daily(code, end_date):
        n = 20
        close = np.full(n, 10.0)
        close[-2], close[-1] = 11.0, 12.1
        return pd.DataFrame({
            "date": [end_date - dt.timedelta(days=int(n - 1 - i)) for i in range(n)],
            "open": np.r_[np.full(n - 2, 9.95), 10.4, 11.55], "close": close,
            "high": close + 0.05, "low": close - 0.05,
            "volume": 1000.0,
        })

    import bars as bars_mod
    orig_fetch = bars_mod.fetch_daily
    bars_mod.fetch_daily = fake_daily   # 补丁真实消费方（bars.acquire_bars 的模块全局）

    spot = pd.DataFrame([
        dict(code="600000", name="测试甲", close=11.00, high=11.00, low=10.10,
             prev_close=10.0, pct=10.0, amount=1e8, turnover=5.0),
        dict(code="600003", name="测试乙", close=9.40, high=9.80, low=9.00,
             prev_close=10.0, pct=-6.0, amount=1e8, turnover=5.0),
    ])
    cands = scr.build_candidates(spot)
    try:
        results, skipped = scr.analyze_all(cands, dt.date(2025, 6, 10),
                                           {"600000": {"lianban": 2, "fund": 5e7}},
                                           workers=2)
    finally:
        bars_mod.fetch_daily = orig_fetch
    assert len(results) == 2 and not skipped
    by = {r["code"]: r for r in results}
    assert by["600003"]["ok"] and not by["600003"]["board_pass"]   # 翘板股零日K
    assert by["600000"]["board_pass"] or by["600000"]["board_score"] > 0
    assert by["600000"]["seal_ratio"] == 0.5                       # 5e7/1e8

    txt, csv, _ = rpt.write_report(results, skipped, cands, dt.date(2025, 6, 10),
                                tmp_dir, signal_only=False)
    body = open(txt, encoding="utf-8").read()
    assert "市场情绪速览" in body and "曾涨停" in body and "生成" in body
    assert "附、全部候选（--all）" in body and "600000" in body
    cdf = pd.read_csv(csv, dtype={"code": str})
    assert set(cdf["code"]) == {"600000", "600003"}
    assert set(cdf["trade_date"]) == {"2025-06-10"}
    shutil.rmtree(tmp_dir, ignore_errors=True)            # 含 archive 子目录一并清理
    print("PASS end-to-end report")


def test_em_spot_units():
    """东财快照单位契约：turnover 百分数→小数（C维小数阈值口径），volume 手，pct 百分数。"""
    from datasources.spot import _em_rows_to_frame
    rows = [{"f12": "600000", "f14": "浦发", "f2": 10.0, "f15": 10.5, "f16": 9.9,
             "f17": 10.1, "f18": 9.8, "f3": 2.04, "f5": 123456.0, "f6": 1.2e8,
             "f8": 2.53}]
    df = _em_rows_to_frame(rows)
    r = df.iloc[0]
    assert abs(float(r["turnover"]) - 0.0253) < 1e-9      # 2.53% → 0.0253
    assert float(r["volume"]) == 123456.0                 # 手，不换算
    assert abs(float(r["pct"]) - 2.04) < 1e-9             # 百分数保持
    assert str(r["code"]) == "600000"
    print("PASS em spot units")


def test_watch_tier():
    """明日清单可执行期望分档：一字剔除、实体0.5~0.8与高开≤0.3为★★★、按档位排序。"""
    def mk(code, name, open_, prev, close, vr):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": open_, "prev_close": prev, "close": close,
                "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08,
                "ok": True, "skip_reason": "", "lianban": 1, "zhaban_times": 0,
                "industry": "", "fbt": "", "fund": float("nan"),
                "seal_ratio": float("nan"), "ltsz": float("nan"),
                "board_pass": False, "board_score": 0, "board_note": "",
                "vol_ratio": vr}
    # 实体0.65(★★★粮仓) / 高开0.25(★★★稀有) / 高开0.60(★★☆) / 实体0.40(★☆☆) / 一字(剔除)
    results = [
        mk("600001", "实体粮仓", 10.1, 10.0, 11.0, 0.65),
        mk("600002", "高开金矿", 10.5, 10.0, 11.0, 0.25),
        mk("600003", "高开次优", 10.5, 10.0, 11.0, 0.60),
        mk("600004", "实体末位", 10.1, 10.0, 11.0, 0.40),
        mk("600005", "一字剔除", 11.0, 10.0, 11.0, 0.30),
    ]
    tmp = "out_watch_test"
    txt, _, _ = rpt.write_report(results, [], results, dt.date(2025, 6, 10), tmp,
                              signal_only=True, ctx={})
    body = open(txt, encoding="utf-8").read()
    assert "附录·早盘+5%备选" not in body               # 附录已删，排序断言随之移除
    assert "一字剔除" in body.split("今日一字/准一字")[1]    # 只出现在备注行
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS watch tier")


def test_open_pool():
    """开盘买入池主区块：档位排序、一字/缩量/高位放量排除、期望与风险标注。"""
    def mk(code, name, open_, prev, close, vr, lb=1):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": open_, "prev_close": prev, "close": close,
                "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08,
                "ok": True, "skip_reason": "", "lianban": lb, "zhaban_times": 0,
                "industry": "", "fbt": "", "fund": float("nan"),
                "seal_ratio": float("nan"), "ltsz": float("nan"),
                "board_pass": False, "board_score": 0, "board_note": "",
                "vol_ratio": vr}
    results = [
        mk("600001", "三板温和", 10.1, 10.0, 11.0, 1.2, lb=3),   # ★★★ +1.5%/62%
        mk("600002", "二板实体", 10.1, 10.0, 11.0, 1.0, lb=2),   # ★★★ +0.6%/51%
        mk("600003", "首板备选", 10.1, 10.0, 11.0, 1.0),         # ★☆☆ 观察名单
        mk("600004", "缩量票", 10.1, 10.0, 11.0, 0.6),           # 排除→附录
        mk("600005", "一字票", 11.0, 10.0, 11.0, 1.2),           # 排除(一字)
        mk("600006", "高位放量", 10.1, 10.0, 11.0, 2.0, lb=3),   # 排除(3板vr>1.5)
    ]
    tmp = "out_open_pool_test"
    txt, _, _ = rpt.write_report(results, [], results, dt.date(2025, 6, 10), tmp,
                              signal_only=True, ctx={"cycle_stage": "发酵"})
    body = open(txt, encoding="utf-8").read()
    assert "封板温和量观察" in body                   # 已降级为观察名单
    assert "历史开→开+1.5%/62%" in body and "历史开→开+0.6%/51%" in body
    assert "不推荐买入" in body
    assert body.index("三板温和") < body.index("二板实体")
    assert "数学终审" in body                          # t=-0.19 声明行
    assert "首板备选" in body                          # 数据行保留
    assert "缩量票 1 只不入池" in body
    assert "今日一字/准一字 1 只未列" in body and "：一字票" in body
    assert "高位放量（3板+·vr>1.5）1 只排除" in body
    assert "t=-0.19" in body                           # 数学终审声明
    assert "附录·早盘+5%备选" not in body               # 附录已按需求删除
    # 空仓警示路径：无 tier0 时
    txt2, _, _ = rpt.write_report(results[2:], [], results[2:], dt.date(2025, 6, 10),
                               tmp, signal_only=True, ctx={})
    body2 = open(txt2, encoding="utf-8").read()
    assert "封板温和量观察" in body2
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS open pool")


def test_slam_ledger():
    """打板画像台账（2026-09-15 起接成交模型，README 待办#5）：
    D1 登记涨停价 → D2 按 T+1 行情判定成交（一字排队不成交/触板回封成交）
    → D3 开盘卖出核收益；pnl_net 扣双边成本。"""
    def mk(code, name, open_, prev, close, vr, lb=1):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": open_, "prev_close": prev, "close": close,
                "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08,
                "ok": True, "skip_reason": "", "lianban": lb, "zhaban_times": 0,
                "industry": "", "fbt": "", "fund": float("nan"),
                "seal_ratio": float("nan"), "ltsz": float("nan"),
                "board_pass": False, "board_score": 0, "board_note": "",
                "vol_ratio": vr}
    from config import COST_ROUND_TRIP
    results_d1 = [mk("600010", "打板票A", 10.1, 10.0, 11.0, 0.65),   # 画像内
                  mk("600011", "深缩排除", 10.1, 10.0, 11.0, 0.3),    # vr≤0.5 排除
                  mk("600012", "秒板排除", 10.6, 10.0, 11.0, 0.65)]   # 开盘>3% 排除
    tmp = "out_slam_test"
    spot1 = pd.DataFrame([{"code": "600010", "pct": 4.5, "open": 11.2,
                           "high": 11.6, "low": 11.0, "close": 11.5,
                           "prev_close": 11.0}])
    txt, _, _ = rpt.write_report(results_d1, [], results_d1, dt.date(2025, 6, 10),
                              tmp, signal_only=True, ctx={}, spot=spot1)
    body = open(txt, encoding="utf-8").read()
    assert "打板台账核账" in body                        # 区块已删，台账照常核账
    assert "打板价(涨停)=11.00" not in body              # 渲染区块确认已删
    assert "今日画像·登记" in body and "打11.0" in body   # 登记中：明晚才判定成交
    led0 = pd.read_csv(os.path.join(tmp, "archive", "slam_ledger.csv"), dtype={"code": str})
    a0 = led0[led0["code"] == "600010"].iloc[0]          # 画像票已登记
    assert abs(float(a0["buy_price"]) - 11.0) < 1e-9      # slam台账买价=涨停价登记
    assert pd.isna(a0["fill"])                            # 登记晚 T+1 不存在，待判定
    assert led0[led0["code"] == "600011"].empty          # 深缩量排除
    assert led0[led0["code"] == "600012"].empty          # 秒板排除
    # D2：600010 非一字开盘 11.55、盘中回封触板 12.10 → 以涨停价成交（fill=1）；
    #     600013 新画像票登记，次日再判
    results_d2 = [mk("600013", "新票B", 5.05, 5.0, 5.5, 0.7)]
    spot2 = pd.DataFrame([{"code": "600010", "pct": 9.1, "open": 11.55,
                           "high": 12.10, "low": 11.4, "close": 12.0,
                           "prev_close": 11.0},
                          {"code": "600013", "pct": 10.0, "open": 5.05,
                           "high": 5.5, "low": 5.0, "close": 5.5,
                           "prev_close": 5.0}])
    txt2, _, _ = rpt.write_report(results_d2, [], results_d2, dt.date(2025, 6, 11),
                               tmp, signal_only=True, ctx={}, spot=spot2)
    body2 = open(txt2, encoding="utf-8").read()
    assert "今晨判定" in body2 and "成交 1 只" in body2
    led2 = pd.read_csv(os.path.join(tmp, "archive", "slam_ledger.csv"),
                       dtype={"code": str})
    a2 = led2[led2["code"] == "600010"].iloc[0]
    assert int(a2["fill"]) == 1 and a2["fill_date"] == "2025-06-11"
    assert pd.isna(a2["sell_open"])                       # 成交后 T+2 开盘才卖
    # D3：600010 开盘 12.71 卖出 → +15.55%；600013 一字开盘 6.05 → 排队不成交
    spot3 = pd.DataFrame([{"code": "600010", "pct": -2.0, "open": 12.71,
                           "high": 12.8, "low": 12.4, "close": 12.5,
                           "prev_close": 12.0},
                          {"code": "600013", "pct": 10.0, "open": 6.05,
                           "high": 6.05, "low": 6.05, "close": 6.05,
                           "prev_close": 5.5}])
    txt3, _, _ = rpt.write_report([], [], [], dt.date(2025, 6, 12),
                               tmp, signal_only=True, ctx={}, spot=spot3)
    body3 = open(txt3, encoding="utf-8").read()
    assert "已打板·卖出" in body3 and "打11.0→卖12.71" in body3 and "+15.6%" in body3
    assert "未成交 1 只" in body3                          # 600013 一字：fill=0 终态
    assert "打板画像 1笔/1日" in body3, body3
    assert "均值+15.55%" in body3 and "中位+15.55%" in body3
    assert "净值+15.55%" in body3 and "复合" not in body3
    led3 = pd.read_csv(os.path.join(tmp, "archive", "slam_ledger.csv"),
                       dtype={"code": str})
    a3 = led3[led3["code"] == "600010"].iloc[0]
    assert abs(float(a3["pnl"]) - round(12.71 / 11.0 - 1, 4)) < 1e-9
    assert abs(float(a3["pnl_net"])
               - round(round(12.71 / 11.0 - 1, 4) - COST_ROUND_TRIP, 4)) < 1e-9
    b3 = led3[led3["code"] == "600013"].iloc[0]
    assert int(b3["fill"]) == 0 and pd.isna(b3["pnl"])    # 未成交不计损益
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS slam ledger")


def test_open_pool_tracking():
    """开盘买入池台账核账：D1登记买入 → D2开盘卖出闭环 + 累计统计 + 新买入。"""
    tmp = "out_open_track_test"
    os.makedirs(os.path.join(tmp, "archive"), exist_ok=True)
    base_row = {"zt_status": "封板"}
    pd.DataFrame([{**base_row, "trade_date": "2025-06-09", "code": "600010",
                   "name": "池票A", "vol_ratio": 1.0, "open": 10.1,
                   "prev_close": 10.0, "close": 11.0, "lianban": 2},
                  {**base_row, "trade_date": "2025-06-09", "code": "600012",
                   "name": "观察C", "vol_ratio": 1.0, "open": 10.1,
                   "prev_close": 10.0, "close": 10.5, "lianban": 1}]).to_csv(
        os.path.join(tmp, "archive", "signals_20250609.csv"),
        index=False, encoding="utf-8-sig")
    dummy = [{"code": "600999", "name": "占位", "pools": ["涨停"], "zt_status": "",
              "dt_status": "", "close": 5.0, "open": 5.0, "prev_close": 5.0,
              "pct": 0.0, "amp": 1.0, "amount": 1e8, "turnover": 0.01,
              "ok": True, "skip_reason": "", "lianban": 0, "zhaban_times": 0,
              "industry": "", "fbt": "", "fund": float("nan"),
              "seal_ratio": float("nan"), "ltsz": float("nan"),
              "board_pass": False, "board_score": 0, "board_note": ""}]
    # D1（06-10）：A 开盘买入 @11.2
    spot1 = pd.DataFrame([{"code": "600010", "pct": 4.5, "open": 11.2,
                           "high": 11.6, "low": 11.0, "close": 11.5,
                           "prev_close": 11.0},
                          {"code": "600012", "pct": 1.0, "open": 10.6,
                           "high": 10.7, "low": 10.5, "close": 10.6,
                           "prev_close": 10.5}])
    txt, _, _ = rpt.write_report(dummy, [], dummy, dt.date(2025, 6, 10), tmp,
                              signal_only=True, ctx={}, spot=spot1)
    body = open(txt, encoding="utf-8").read()
    assert "开盘买入池核账" in body and "今晨开盘买入" in body and "买11.2" in body
    # 观察登记已折叠为 1 行计数：逐笔买价不再印在报告里（该策略五R 已判死，
    # 逐笔列 24 只会把核心输出挤成配角），但必须留下 CSV 台账出处，折叠≠丢审计
    assert "观察登记" in body and "仅记账验证 1 只" in body
    assert "买10.6" not in body, "观察登记应折叠为计数，不再逐笔列买价"
    assert "open_pool_ledger.csv" in body
    # D2（06-11）：A 开盘卖出 @11.8（+5.36%），B（06-10 归档池票）开盘买入 @10.3
    pd.DataFrame([{**base_row, "trade_date": "2025-06-10", "code": "600011",
                   "name": "池票B", "vol_ratio": 1.0, "open": 10.1,
                   "prev_close": 10.0, "close": 10.0, "lianban": 2}]).to_csv(
        os.path.join(tmp, "archive", "signals_20250610.csv"),
        index=False, encoding="utf-8-sig")
    spot2 = pd.DataFrame([
        {"code": "600010", "pct": 2.7, "open": 11.8, "high": 11.9,
         "low": 11.6, "close": 11.8, "prev_close": 11.5},
        {"code": "600011", "pct": 3.0, "open": 10.3, "high": 10.4,
         "low": 10.1, "close": 10.3, "prev_close": 10.0},
        {"code": "600012", "pct": -1.0, "open": 10.4, "high": 10.5,
         "low": 10.3, "close": 10.4, "prev_close": 10.6}])
    txt, _, _ = rpt.write_report(dummy, [], dummy, dt.date(2025, 6, 11), tmp,
                              signal_only=True, ctx={}, spot=spot2)
    body = open(txt, encoding="utf-8").read()
    assert "今晨开盘卖出（轮动完成 2 只）" in body   # A(买)+C(观察) 都完成轮动
    assert "买11.2→卖11.8" in body and "+5.4%" in body
    assert "买10.3" in body                          # B 新买入
    # _stats_line 新口径：标签缩写以便单行放得下，带交易日数、中位数与净值，
    # 且**不得**再出现「复合」——旧版把同日并发仓当串行全仓连乘，
    # 103 笔得出「复合-80.21%」这种无意义又刺眼的数字
    assert "★★★档 1笔/1日" in body, body
    assert "均值+5.36%" in body and "中位+5.36%" in body
    assert "净值+5.36%" in body and "回撤0.0%" in body
    assert "复合" not in body, "同日并发仓不得按复利连乘"
    assert "观察档 1笔/1日" in body   # C(观察)分账统计
    led_df = pd.read_csv(os.path.join(tmp, "archive", "open_pool_ledger.csv"),
                         dtype={"code": str})
    a = led_df[led_df["code"] == "600010"].iloc[0]
    assert a["status"] == "bought" and str(a["tier"]) == "0"
    assert abs(float(a["pnl"]) - (11.8 / 11.2 - 1)) < 1e-4   # 台账存4位小数
    c = led_df[led_df["code"] == "600012"].iloc[0]
    assert c["status"] == "watch" and str(c["tier"]) == "2"
    assert abs(float(c["pnl"]) - (10.4 / 10.6 - 1)) < 1e-4
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS open pool tracking")


def test_cycle_series():
    """周期序列归档：追加/同日覆盖幂等，stage 只属于复盘日。"""
    tmp = "out_cycle_test"
    s1 = [{"date": "2025-06-09", "height": 3, "zt_count": 30, "promo": 0.2,
           "break_rate": 0.3, "money": None},
          {"date": "2025-06-10", "height": 5, "zt_count": 40, "promo": 0.3,
           "break_rate": 0.25, "money": 2.5}]
    pipeline.archive_cycle_series(tmp, s1, "发酵")
    path = os.path.join(tmp, "archive", "cycle_series.csv")
    df1 = pd.read_csv(path)
    assert list(df1["date"]) == ["2025-06-09", "2025-06-10"]
    assert df1.iloc[-1]["stage"] == "发酵" and pd.isna(df1.iloc[0]["stage"])
    # 同日重跑：覆盖该日行（指标刷新+stage更新），前日不动
    s2 = [{"date": "2025-06-10", "height": 5, "zt_count": 41, "promo": 0.3,
           "break_rate": 0.25, "money": 3.1}]
    pipeline.archive_cycle_series(tmp, s2, "高潮")
    df2 = pd.read_csv(path)
    assert list(df2["date"]) == ["2025-06-09", "2025-06-10"]
    assert df2.iloc[-1]["stage"] == "高潮"
    assert int(df2.iloc[-1]["zt_count"]) == 41
    assert abs(float(df2.iloc[-1]["money"]) - 3.1) < 1e-9
    assert int(df2.iloc[0]["zt_count"]) == 30      # 前日未被覆盖
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS cycle series")


def test_doctor():
    """数据体检：量纲回归/数量级漂移/K线口径/量比分布四类告警 + 健康样本零告警。"""
    d = dt.date(2025, 6, 10)
    n = 300
    spot = pd.DataFrame({"code": ["600000"] * n, "pct": [1.0] * n,
                         "close": [10.0] * n, "turnover": [0.02] * n})
    spot.loc[spot.index[:40], "pct"] = 10.0        # 40 只主板封板
    results = ([{"code": "600000", "lianban": 2, "vol_ratio": 1.0}] * 40
               + [{"code": "600001", "lianban": 0, "vol_ratio": 0.6}] * 60)
    bars = {"600000": pd.DataFrame(
        {"date": [dt.date(2025, 6, 9), d], "close": [9.0, 10.0]})}
    assert pipeline.doctor_check(spot, results, bars, d) == []
    # 昨日播种的旧 bars（末根早于复盘日，非封板股未分析）不参与口径对比
    stale = {"600009": pd.DataFrame(
        {"date": [dt.date(2025, 6, 6), dt.date(2025, 6, 9)], "close": [9.0, 12.0]})}
    assert pipeline.doctor_check(spot, results, stale, d) == []
    assert any("换手率" in w for w in
               pipeline.doctor_check(spot.assign(turnover=2.5), results, bars, d))
    assert any("封板" in w for w in
               pipeline.doctor_check(spot.assign(pct=1.0), results, bars, d))
    assert any("不一致" in w for w in pipeline.doctor_check(
        spot, results, {"600000": pd.DataFrame(
            {"date": [dt.date(2025, 6, 9), d], "close": [9.0, 12.0]})}, d))
    assert any("量比" in w for w in pipeline.doctor_check(
        spot, [{"code": "600000", "lianban": 1, "vol_ratio": 30.0}] * 20, bars, d))
    print("PASS doctor")


def test_limit_price_matches_exchange():
    """涨跌停价取整单一语义：标量路径 ≡ 候选池向量化路径（全价位网格）。

    2026-09-15 修正的 bug：旧 round(v*100+0.5)/100 双重舍入，v*100 小数落在
    (0,0.5) 时高估一分（4.62→5.09，交易所口径 5.08）。
    2026-11-24 修正的 bug：候选池向量化那一行是无容差的 `floor(v*100+0.5)/100`，
    与标量路径**并不同源**——0.01~300 元网格上 190 个价位低一分（涨停 33 /
    跌停 157，全是昨收尾数 5 分的价位）。跌停价偏低会让真正触及跌停的票不被
    识别（一字跌停振幅为 0，连候选池都进不了）。此测试强制两条路径在涨跌停
    两侧全部恒等。
    """
    import trade_rules as tr
    from candidates import limit_prices, limit_prices_vec
    # bug 区已知值（旧实现分别给 5.09 / 3.67 / 6.09）
    assert tr.limit_price(4.62) == 5.08
    assert tr.limit_price(3.33) == 3.66
    assert tr.limit_price(5.535) == 6.09
    # 向量化旧实现的已知错值：一律低一分
    assert limit_prices_vec(np.array([17.15]), np.array([0.10]))[0][0] == 18.87
    assert limit_prices_vec(np.array([2.65]), np.array([0.10]))[1][0] == 2.39
    assert tr.limit_down_price(17.15) == 15.44 and tr.limit_down_price(2.65) == 2.39
    # 网格恒等（0.01 步长扫 0.01~300 元，覆盖主板全部价位）× 普通/ST 两档
    cents = np.arange(1, 30001)
    pc = cents / 100.0
    for ratio in (0.10, 0.05):
        up, dn = limit_prices_vec(pc, np.full(len(pc), ratio))
        assert (up == np.array([tr.limit_price(p, ratio) for p in pc])).all()
        assert (dn == np.array([tr.limit_down_price(p, ratio) for p in pc])).all()
    name = "普通股"
    assert all(limit_prices(p, name) == (tr.limit_price(p), tr.limit_down_price(p))
               for p in (10.0, 17.15, 2.65, 33.05, 64.85))
    print("PASS limit price matches exchange")


def test_calib_facade():
    """calib 包门面契约：calibrate.py 重导出拆分前的全部事实公共 API。

    calibrate.cluster_mean 等被 tests 与 out/research/prereg/* 研究脚本导入
    （含 _two_sided_p 私有名）；tests 还在 calibrate.data_source 上打补丁——
    门面必须保持这些访问点可用，拆分才能对调用方零成本。
    """
    import calibrate
    import calib
    for name in ("cluster_mean", "cluster_diff", "two_sided_p", "_two_sided_p",
                 "bh_fdr", "fmt_t", "fmt_mean", "build_features", "analyze_dims",
                 "COLLINEAR_GROUPS", "load_signals", "fetch_forward",
                 "load_forward_cache", "FWD_KEYS", "MAIN_METRIC", "MIN_INFER_DAYS",
                 "BUCKETS", "bucket_stats", "_table", "_contrast_line", "summarize",
                 "oos_section", "source_section", "ledger_section",
                 "LEDGER_EXPECT", "main"):
        assert hasattr(calibrate, name), f"calibrate 门面缺 {name}"
        assert hasattr(calib, name), f"calib 包缺 {name}"
    assert calibrate.MAIN_METRIC == calib.forward.MAIN_METRIC == "net_ret"
    assert calibrate.MIN_INFER_DAYS == calib.forward.MIN_INFER_DAYS == 8
    # 补丁点：patch calibrate.data_source.X 必须被 calib.forward 看到（同一模块对象）
    assert calibrate.data_source is calib.forward.data_source
    print("PASS calib facade")


def test_forward_safe_write():
    """_forward.csv 防覆写：失败旧行回写保底 + 行数骤减拒写。

    旧口径是 to_csv 全量覆写——旧行既是 stale 又重拉失败时该历史样本永久
    消失（_forward.csv 只有 ~200 行 / 23 个交易日，丢不起）。"""
    import calib.cli

    tmp = "out_fwd_safe_test"
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "_forward.csv")
    old = pd.DataFrame([
        {"code": "600001", "date": "2026-08-10", "next_open_pct": 1.0,
         "net_ret": 0.5, "promoted": 1, "one_word": 0, "fillable": 1,
         "tradable_ret": 0.65, "next_close_pct": 2.0, "max3_pct": 3.0},
        {"code": "600002", "date": "2026-08-11", "next_open_pct": 2.0,
         "net_ret": 1.5, "promoted": 0, "one_word": 0, "fillable": 1,
         "tradable_ret": 1.65, "next_close_pct": 3.0, "max3_pct": 4.0},
    ])
    old.to_csv(path, index=False, encoding="utf-8-sig")
    new_rows = [{"code": "600001", "date": "2026-08-10", "net_ret": 0.5}]
    # (1) 600002 重拉失败（在 failed_keys）→ 旧行必须回写，不得消失
    assert calib.cli._safe_write_forward(
        path, new_rows, old, {("600002", "2026-08-11")})
    got = pd.read_csv(path, dtype={"code": str})
    assert set(zip(got["code"], got["date"])) == {("600001", "2026-08-10"),
                                                  ("600002", "2026-08-11")}
    # (2) 行数骤减（<旧表一半且旧表>20 行）→ 拒写，旧文件原样保留
    big_old = pd.concat([old] * 11, ignore_index=True)
    big_old.to_csv(path, index=False, encoding="utf-8-sig")
    assert not calib.cli._safe_write_forward(path, new_rows, big_old, set())
    got2 = pd.read_csv(path, dtype={"code": str})
    assert len(got2) == len(big_old), "拒写时旧文件必须原样保留"
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS forward safe write")


def test_research_scan_window_and_lianban():
    """research_scan：连板数/量比/事件日缺口新列 + 窗口冻结锚点（待办#6）。"""

    import research_scan as rs

    tmp = "out_rscan_test"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    meta = os.path.join(tmp, "window.json")
    orig_meta, rs.WINDOW_META = rs.WINDOW_META, meta
    orig_events, rs.EVENTS = rs.EVENTS, os.path.join(tmp, "events.csv")
    try:
        # (1) 窗口冻结：首次计算落锚 → 之后传再大的 months 也返回锚点
        d0 = dt.date(2025, 9, 5)
        a1 = rs.window_start(1)
        assert a1 == dt.date.today() - dt.timedelta(days=31)
        a2 = rs.window_start(12)
        assert a2 == a1, "锚点已冻结，months 再大也不得滑动"
        assert json.load(open(meta, encoding="utf-8"))["start"] == f"{a1:%Y-%m-%d}"
        # (2) 显式重锚
        a3 = rs.window_start(12, refreeze=True)
        assert a3 == dt.date.today() - dt.timedelta(days=372)

        # (3) scan_events 新列：构造 3 连板序列（第 5~7 日连续 +10% 涨停，
        #     事件日 = 第 7 日；扫描循环从 i>=6 起，事件必须落在其后）
        def row(d, o, c, v):
            return {"date": d, "open": o, "close": c, "volume": v}
        days = [d0 + dt.timedelta(days=i) for i in range(12)]
        px = 10.0
        rows = []
        for i, d in enumerate(days):
            if 5 <= i <= 7:                     # 三连板
                o, c, v = round(px * 1.02, 3), round(px * 1.10, 3), 5.0e6
            else:
                o, c, v = px, round(px * 0.99, 3), 1.0e7
            rows.append(row(d, o, c, v))
            px = c
        df = pd.DataFrame(rows)
        evs = rs.scan_events(df, d0)
        ev = [e for e in evs if e["date"] == days[7]]
        assert ev, "第7日应为事件日（第3个涨停）"
        e = ev[0]
        assert e["lianban"] == 3, "三连板事件 lianban 必须为 3"
        assert e["gap_event"] == round(float(rows[7]["open"] / rows[6]["close"] - 1), 4)
        base = df["volume"].iloc[2:7].mean()    # 前5日均量（i=7 的窗口 [2,7)）
        assert e["vr"] == round(float(df["volume"].iloc[7] / base), 2)
        # 事件检测未变：gap_next/one_word_next 语义保持
        assert e["one_word_next"] is False or e["gap_next"] >= 0.09
    finally:
        rs.WINDOW_META, rs.EVENTS = orig_meta, orig_events
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS research scan window & lianban")


def test_ledger_backfill_settlement():
    """台账跳日结算（待办#5）：按预期 T+1 日开盘价结算，而非下次运行日开盘。

    场景：周二 06-10 登记买入；周三 06-11 未运行（spot 由缓存补）；
    周四 06-12 运行 → 必须以**周三**开盘价结算（sell_date=06-11），并在报告
    标注「补结」。旧实现会拿周四开盘价悄悄记成 T+1 卖。"""
    tmp = "out_settle_test"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "archive"), exist_ok=True)
    base_row = {"zt_status": "封板"}
    pd.DataFrame([{**base_row, "trade_date": "2025-06-09", "code": "600010",
                   "name": "池票A", "vol_ratio": 1.0, "open": 10.1,
                   "prev_close": 10.0, "close": 11.0, "lianban": 2}]).to_csv(
        os.path.join(tmp, "archive", "signals_20250609.csv"),
        index=False, encoding="utf-8-sig")
    dummy = [{"code": "600999", "name": "占位", "pools": ["涨停"], "zt_status": "",
              "dt_status": "", "close": 5.0, "open": 5.0, "prev_close": 5.0,
              "pct": 0.0, "amp": 1.0, "amount": 1e8, "turnover": 0.01,
              "ok": True, "skip_reason": "", "lianban": 0, "zhaban_times": 0,
              "industry": "", "fbt": "", "fund": float("nan"),
              "seal_ratio": float("nan"), "ltsz": float("nan"),
              "board_pass": False, "board_score": 0, "board_note": ""}]
    from config import COST_ROUND_TRIP
    # 周二（06-10）：登记买入 @11.2
    spot_tue = pd.DataFrame([{"code": "600010", "pct": 4.5, "open": 11.2,
                              "high": 11.6, "low": 11.0, "close": 11.5,
                              "prev_close": 11.0}])
    rpt.write_report(dummy, [], dummy, dt.date(2025, 6, 10), tmp,
                     signal_only=True, ctx={}, spot=spot_tue)
    led0 = pd.read_csv(os.path.join(tmp, "archive", "open_pool_ledger.csv"),
                       dtype={"code": str})
    a0 = led0[led0["code"] == "600010"].iloc[0]
    assert abs(float(a0["buy_open"]) - 11.2) < 1e-9
    # 周三（06-11）未运行：把周三 spot 存进滚动缓存（结算数据源）
    cache_store.save_spot(tmp, dt.date(2025, 6, 11), pd.DataFrame(
        [{"code": "600010", "pct": -1.0, "open": 11.34, "high": 11.4,
          "low": 11.1, "close": 11.3, "prev_close": 11.5}]))
    # 周四（06-12）运行：必须以周三开盘 11.34 结算（T+1），不是周四开盘
    spot_thu = pd.DataFrame([{"code": "600010", "pct": -3.0, "open": 11.0,
                              "high": 11.1, "low": 10.9, "close": 11.0,
                              "prev_close": 11.3}])
    txt, _, _ = rpt.write_report(dummy, [], dummy, dt.date(2025, 6, 12), tmp,
                                 signal_only=True, ctx={}, spot=spot_thu)
    body = open(txt, encoding="utf-8").read()
    assert "补结 1 笔" in body, body[-2000:]
    led = pd.read_csv(os.path.join(tmp, "archive", "open_pool_ledger.csv"),
                      dtype={"code": str})
    a = led[led["code"] == "600010"].iloc[0]
    assert a["sell_date"] == "2025-06-11", "结算日必须是预期 T+1（周三），非运行日"
    assert abs(float(a["sell_open"]) - 11.34) < 1e-9
    assert abs(float(a["pnl"]) - round(11.34 / 11.2 - 1, 4)) < 1e-9
    assert abs(float(a["pnl_net"])
               - round(round(11.34 / 11.2 - 1, 4) - COST_ROUND_TRIP, 4)) < 1e-9
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS ledger backfill settlement")


def test_watchlist_block():
    """明日关注清单：三模式合并去重、评分降序、硬性 ≤5 只、置于决策摘要之后。

    用户诉求：不要「这个模式一种、那个模式一种」散在各区块——同一张名单
    按评分排序，最多 5 只。开盘池（五R 判死）不得混入。"""
    def mk(code, name, open_, prev, close, vr, lb, score, note=""):
        return {"code": code, "name": name, "pools": ["涨停"], "zt_status": "封板",
                "dt_status": "", "open": open_, "prev_close": prev,
                "close": close, "pct": 10.0, "amp": 8.0, "amount": 1e8,
                "turnover": 0.08, "ok": True, "skip_reason": "",
                "lianban": lb, "zhaban_times": 0, "industry": "", "fbt": "",
                "fund": float("nan"), "seal_ratio": float("nan"),
                "ltsz": float("nan"),
                "board_pass": score >= 70, "board_score": score,
                "board_note": note or "题材:X │ 首板:实体 │ 二板:高开+2.6%",
                "vol_ratio": vr}
    tmp = "out_watchlist_test"
    shutil.rmtree(tmp, ignore_errors=True)
    results = [
        mk("600001", "龙头甲", 10.2, 10.0, 11.0, 0.65, 2, 82),   # 龙头+打板(vr0.65·开+2%)
        mk("600002", "龙头乙", 10.5, 10.0, 11.0, 1.50, 2, 76),   # 仅龙头(量比超画像与盯盘)
        mk("600003", "龙头丙", 10.8, 10.0, 11.0, 0.30, 2, 71),   # 龙头+5%盯盘(vr0.3非打板)
        mk("600004", "三板丁", 10.1, 10.0, 11.0, 0.70, 3, 0),    # 仅打板(3板无评分)
        mk("600005", "首板戊", 10.2, 10.0, 11.0, 0.50, 1, 0),    # 仅+5%盯盘(首板缩量)
    ]
    spot = pd.DataFrame([{"code": f"60000{i}", "pct": 1.0, "open": 11.0,
                          "high": 11.2, "low": 10.9, "close": 11.0,
                          "prev_close": 11.0} for i in range(1, 6)])
    txt, _, _ = rpt.write_report(results, [], results, dt.date(2025, 6, 10),
                                 tmp, signal_only=True, ctx={}, spot=spot)
    body = open(txt, encoding="utf-8").read()
    # 区块存在且紧跟决策摘要、在情绪速览之前
    i_sum = body.find("明日决策摘要")
    i_watch = body.find("明日关注清单")
    i_mood = body.find("市场情绪速览")
    assert 0 < i_sum < i_watch < i_mood, (i_sum, i_watch, i_mood)
    # 恰好 5 只、按评分降序、无分票按连板数排在有分票后
    import re as _re
    rows = _re.findall(r"^  (\d)\. ", body[body.find("明日关注清单"):
                                           body.find("▎市场情绪速览")], _re.M)
    assert len(rows) == 5, rows
    order = [(body.find("82分"), body.find("76分"), body.find("71分"))
             for _ in [0]][0]
    assert order[0] < order[1] < order[2]
    # 去重与标签叠加：龙头甲同时是打板画像票 → 一行两个标签
    assert "龙头·打板" in body and "龙头·+5%盯盘" in body
    # 打板票不重复叠 +5%盯盘 标签
    assert "打板·+5%盯盘" not in body
    # 排板价提示（无评分打板票）
    assert "排板≈" in body
    # 截断：7 只过线 → 只列 5 只并注明
    results2 = [mk(f"60001{i}", f"票{i}", 10.2, 10.0, 11.0, 1.5, 2, 85 - i)
                for i in range(7)]
    spot2 = pd.DataFrame([{"code": f"60001{i}", "pct": 1.0, "open": 11.0,
                           "high": 11.2, "low": 10.9, "close": 11.0,
                           "prev_close": 11.0} for i in range(7)])
    txt2, _, _ = rpt.write_report(results2, [], results2, dt.date(2025, 6, 11),
                                  tmp, signal_only=True, ctx={}, spot=spot2)
    body2 = open(txt2, encoding="utf-8").read()
    seg2 = body2[body2.find("明日关注清单"):body2.find("▎市场情绪速览")]
    rows2 = _re.findall(r"^  (\d)\. ", seg2, _re.M)
    assert len(rows2) == 5 and "另有 2 只" in seg2, seg2
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS watchlist block")


def test_watchlist_unified_score():
    """统一关注分：三模式一张榜、档位锚点、共振加分、排序=看好度。

    用户要求：全部模式的票用统一评分体系打分取前五，分数越高越看好。
    档位锚点见 report._WATCH_MODE_BASE 注释（每个数字对应项目自己的
    实测/判定状态）。N字/十字星已全量移除（五U终止/五N打板子集）。"""
    def mk(code, name, score=0, vr=None, lb=1, close=11.0, note=""):
        return {"code": code, "name": name, "zt_status": "封板",
                "board_pass": score >= 70, "board_score": score,
                "board_note": note or "题材:X │ 首板:实体 │ 二板:高开+2.6%",
                "vol_ratio": vr, "lianban": lb, "close": close, "open": 10.5,
                "prev_close": 10.0}
    board = [mk("600100", "龙头A", 86, vr=0.65, lb=2)]          # 86×0.7+30 = 90
    slam = [mk("600200", "打板B", vr=0.70, lb=3)]               # 56+26 = 82
    watch = [mk("600300", "盯盘C", vr=0.50, lb=1)]              # 52+24 = 76
    lines = rpt._watchlist_lines(board, slam, watch)
    body = "\n".join(lines)
    # 三个模式各有一行，排序 = 统一关注分降序 = 看好度
    pos = [body.find(c) for c in ("600100", "600200", "600300")]
    assert all(p > 0 for p in pos) and pos == sorted(pos), pos
    assert "90分" in body and "82分" in body and "76分" in body
    for tag in ("[龙头]", "[打板]", "[+5%盯盘]"):
        assert tag in body, tag
    # 难成交龙头可买性降档：84分龙头准一字 → 84×0.7+12 = 71，排在打板票之后
    board2 = [mk("600110", "难丁", 84, vr=0.65, lb=2,
                 note="题材:X │ 首板:实体 │ 二板:准一字")]
    body2 = "\n".join(rpt._watchlist_lines(board2, slam, watch))
    assert "71分" in body2
    assert body2.find("600200") < body2.find("600110"), "可执行的打板票应排在买不到的龙头前"
    # 共振加分：龙头同时命中打板画像 → max(86×0.7, 56)+4+30 = 94
    lines3 = rpt._watchlist_lines(board, [dict(slam[0], code="600100", vr=0.65)],
                                  [])
    body3 = "\n".join(lines3)
    assert "94分" in body3 and "龙头·打板" in body3
    print("PASS watchlist unified score")


def test_ziyang_progress():
    """紫阳模块进度条：百分比两位小数、空任务零输出、非 tty 每 25 步一行。

    背景：2026-09-17 用户被「[画像] 日K 150/168 假死」卡住后设立——所有
    慢循环必须全程可见，末行定格 100.00% + 用时。
    """
    import io as _io

    import ziyang.progress as zp

    # 两位小数（四舍五入方向随 Python round-half-even 由格式化决定，此处锁行为）
    pg = zp.Progress("测试", 168)
    pg._tty = False
    assert pg._line(150).split()[-1] == "89.29%", pg._line(150)
    assert pg._line(1).split()[-1] == "0.60%", pg._line(1)
    assert pg._line(168).split()[-1] == "100.00%", pg._line(168)
    pg7 = zp.Progress("七分之一", 7)
    pg7._tty = False
    assert pg7._line(1).split()[-1] == "14.29%", pg7._line(1)
    assert pg7._line(3).split()[-1] == "42.86%", pg7._line(3)

    class _Cap:                                   # 捕获 stdout 的假非 tty 流
        def __init__(self):
            self.buf = _io.StringIO()
        def isatty(self):
            return False
        def write(self, s):
            return self.buf.write(s)
        def flush(self):
            pass

    real, cap = sys.stdout, _Cap()
    sys.stdout = cap
    try:
        # 空任务零输出
        p0 = zp.Progress("空", 0)
        for i in range(3):
            p0.update(i)
        assert p0.done() >= 0
        assert cap.buf.getvalue() == "", cap.buf.getvalue()
        # 非 tty：50 步只打 25/50 中间行 + done 收尾行（末步不重复打）
        p1 = zp.Progress("批", 50)
        for i in range(1, 51):
            p1.update(i, extra="600000")
        p1.done()
        lines = [l for l in cap.buf.getvalue().splitlines() if l.strip()]
    finally:
        sys.stdout = real
    assert len(lines) == 2, lines
    assert "25/50" in lines[0] and "50.00%" in lines[0], lines
    assert "100.00%" in lines[1] and "（" in lines[1], lines
    print("PASS ziyang progress bar")


def test_ziyang_bars_batch():
    """紫阳日K批量并发拉取：空批零输出；缓存命中时线程池路径零网络、不改数据。

    性能改造（2026-09-17）的回归守卫：批量段只做网络加速，逐只口径与
    get_bars 串行版一致（增量拼接+复权锚校验在 get_bars 内部不变）。
    """
    import io as _io

    import pandas as pd

    import ziyang.seatdata as zs

    def _bars(n=40):
        days = [dt.date(2026, 8, 1) + dt.timedelta(days=i) for i in range(n)]
        return pd.DataFrame({
            "date": [d.isoformat() for d in days],
            "open": [10.0] * n, "close": [10.0] * n, "high": [10.0] * n,
            "low": [10.0] * n, "volume": [100.0] * n, "amount": [1e6] * n,
            "pct": [0.0] * n, "turnover": [1.0] * n})

    class _Cap:
        def __init__(self):
            self.buf = _io.StringIO()
        def isatty(self):
            return False
        def write(self, s):
            return self.buf.write(s)
        def flush(self):
            pass

    real, cap = sys.stdout, _Cap()
    sys.stdout = cap
    try:
        zs.fetch_bars_batch({}, {}, tag="空批")          # 空批：零输出零线程
        cache = {"600000": _bars()}
        before = cache["600000"].copy()
        # end=末根日期 → get_bars 走缓存快速路径（trim），线程池全程零网络
        zs.fetch_bars_batch(cache, {"600000": dt.date(2026, 9, 9)}, tag="批")
        out = cap.buf.getvalue()
    finally:
        sys.stdout = real
    assert "空批" not in out, out                        # 空批零输出
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines and "100.00%" in lines[-1], lines       # 批次完成有定格行
    assert len(cache["600000"]) == len(before)           # 命中缓存不改数据
    # 记忆化：返回拷贝，调用方改返回值不污染记忆
    m1 = zs.archive_industry_map()
    n1 = len(m1)
    m1["000000"] = "被污染"
    m2 = zs.archive_industry_map()
    assert "000000" not in m2 and len(m2) == n1
    zs._IND_MAP_MEMO = None                             # 复位，不泄漏给后续测试
    print("PASS ziyang bars batch")


def test_ziyang_bars_batch_retry():
    """批量拉取失败救回（2026-09-17 二轮迭代）：空手而归的码批末串行重试一轮，
    瞬时失败救回、持续失败记账可见（「失败 N（重试救回 M）」）。零网络。"""
    import io as _io

    import ziyang.seatdata as zs

    calls, cache = {}, {}
    real_get = zs.get_bars
    real_ready = zs._bs_ready

    def fake_get(c, code, end, min_bars=30, sleep=None):
        calls[code] = calls.get(code, 0) + 1
        if code == "600003" or (code == "600002" and calls[code] == 1):
            return None                        # 600003 持续失败；600002 首次瞬时失败
        c[code] = "OK"
        return c[code]

    class _Cap:
        def __init__(self):
            self.buf = _io.StringIO()
        def isatty(self):
            return False
        def write(self, s):
            return self.buf.write(s)
        def flush(self):
            pass

    real_out, cap = sys.stdout, _Cap()
    zs.get_bars = fake_get
    zs._bs_ready = lambda end, now=None: False     # 钉死线程池路径（重试语义所在）
    sys.stdout = cap
    try:
        end = dt.date(2026, 9, 17)
        zs.fetch_bars_batch(cache, {"600001": end, "600002": end, "600003": end},
                            tag="批", workers=2)
        out = cap.buf.getvalue()
    finally:
        sys.stdout = real_out
        zs.get_bars = real_get
        zs._bs_ready = real_ready
    assert calls["600002"] == 2, calls          # 瞬时失败被重试
    assert calls["600003"] == 2, calls          # 持续失败也重试过一轮
    assert cache.get("600002") == "OK"          # 救回
    assert cache.get("600003") is None          # 仍空手 → 走中性桶
    assert "救回 1" in out and "失败 1 只" in out, out
    print("PASS ziyang bars batch retry")


def test_ziyang_snapshot_fresh():
    """新鲜度探测 v2（比涨幅，2026-09-17）：复权基准无关 + 跨零点保护不回退。

    回归背景：v1 比绝对收盘价，快照(未复权) vs 日K(前复权)在除权股上必假阴性，
    且随缓存最后写入源翻转（实测同晚间宇宙 477↔610 抖动）。
    """
    import pandas as pd

    import ziyang.screen as zscr

    day = dt.date(2026, 9, 17)

    def bars(closes):
        n = len(closes)
        base = day - dt.timedelta(days=n - 1)
        return pd.DataFrame({"date": [(base + dt.timedelta(days=i)).isoformat()
                                      for i in range(n)],
                             "close": closes})

    closes = [10.0] * 30
    closes[-1] = 10.5                                  # 昨收10 → 今日 +5.0%
    cache = {c: bars(list(closes)) for c in ("000001", "600519", "601318")}

    def snap(pcts, closes_=(10.5,) * 3):
        return pd.DataFrame({"code": ["000001", "600519", "601318"],
                             "close": list(closes_), "pct": list(pcts)})

    # 1) 涨幅一致 → 新鲜
    assert zscr.snapshot_is_fresh(snap((5.0, 5.0, 5.0)), cache, day) is True
    # 2) 跨零点：快照滚动为次日盘前（涨幅归零）→ 不新鲜（保护不回退）
    assert zscr.snapshot_is_fresh(snap((0.0, 0.0, 0.0)), cache, day) is False
    # 3) 除权回归例：快照未复权价比日K前复权高 20%，但涨幅一致 → 仍新鲜
    #    （v1 口径此例判 False，导致强势通道被误关）
    assert zscr.snapshot_is_fresh(snap((5.0, 5.0, 5.0), (12.6, 12.6, 12.6)),
                                  cache, day) is True
    # 4) 容差边界：|差|≤0.5 个百分点算一致；3 只中 2 只一致才判新鲜
    assert zscr.snapshot_is_fresh(snap((5.2, 5.4, 5.4)), cache, day) is True
    assert zscr.snapshot_is_fresh(snap((5.2, 5.6, 0.0)), cache, day) is False
    print("PASS ziyang snapshot fresh")


def test_ziyang_baostock_layer():
    """baostock 盘后主力源（2026-09-18）：入库时间门槛 + 行转换量纲契约。"""
    import pandas as pd

    import ziyang.seatdata as zs

    # 入库门槛：end 日 17:35 起视为可用；此前 baostock 无当日K，不得入链
    assert zs._bs_ready("20260917", dt.datetime(2026, 9, 17, 17, 35)) is True
    assert zs._bs_ready("20260917", dt.datetime(2026, 9, 17, 17, 34)) is False
    assert zs._bs_ready("20260917", dt.datetime(2026, 9, 18, 9, 0)) is True   # 次日亦可用
    assert zs._bs_ready("20260918", dt.datetime(2026, 9, 18, 9, 0)) is False  # 当日未入库

    # 行转换：volume 股→手；停牌 turn 空串→None；close 缺失行丢弃
    rows = [
        ["2026-09-17", "10.0", "10.5", "9.9", "10.4", "1755380", "2217338282.75", "0.71"],
        ["2026-09-10", "10.0", "10.0", "10.0", "10.0", "0", "", ""],          # 停牌日
        ["2026-09-09", "", "", "", "", "", "", ""],                            # 无效行
    ]
    df = zs._bs_rows_to_df(rows)
    assert len(df) == 2, df
    r = df.iloc[0]
    assert r["date"] == "2026-09-17" and abs(r["volume"] - 17553.80) < 0.01   # 股→手
    assert r["turnover"] == 0.71 and r["pct"] is None
    susp = df.iloc[1]
    assert pd.isna(susp["turnover"]) and susp["volume"] == 0.0
    print("PASS ziyang baostock layer")


def test_ziyang_judgement_baseline_recall():
    """预注册判读：基线（T日主板封板股·席位次日净买率）与召回的机械计算。

    只建裁判席不碰打分口径（config 台账纪律段预注册的指标）。合成数据零网络。
    """
    import tempfile

    import ziyang.config as zc
    import ziyang.ledger as zl

    day, vday = dt.date(2026, 9, 16), dt.date(2026, 9, 17)
    tmp = tempfile.mkdtemp()
    old = {k: getattr(zc, k) for k in
           ("LEDGER_CSV", "SEAT_TRADES_CSV", "ROOT")}
    try:
        zc.ROOT = tmp
        zc.LEDGER_CSV = os.path.join(tmp, "ledger.csv")
        zc.SEAT_TRADES_CSV = os.path.join(tmp, "seat.csv")
        # 归档：T 日主板封板 600100/000100，炸板 600200（不计），创业板封板 300999（不计）
        arc = os.path.join(tmp, "out", "archive")
        os.makedirs(arc, exist_ok=True)
        pd.DataFrame({"code": ["600100", "000100", "600200", "300999"],
                      "zt_status": ["封板", "封板", "炸板", "封板"]}).to_csv(
            os.path.join(arc, "signals_20260916.csv"), index=False, encoding="utf-8-sig")
        # 席位次日：买 600100（基线命中+召回命中）与 600300（非封板，仅召回分母）
        pd.DataFrame({"交易日期": [vday] * 2, "股票代码": ["600100", "600300"],
                      "股票名称": ["甲", "乙"], "涨跌幅": [10.0, 3.0],
                      "买入金额": [1e8, 1e8], "卖出金额": [0.0, 0.0],
                      "上榜原因": ["x", "x"]}).to_csv(
            zc.SEAT_TRADES_CSV, index=False, encoding="utf-8-sig")
        # 台账：T 日已核验两条，600100 在册
        with open(zc.LEDGER_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=zl.LEDGER_COLS)
            w.writeheader()
            w.writerow({"predict_date": day.isoformat(), "code": "600100", "name": "甲",
                        "status": "verified", "verify_date": vday.isoformat(),
                        "seat_bought": "Y", "outcome": "ignited", "rank": "1"})
            w.writerow({"predict_date": day.isoformat(), "code": "600900", "name": "丙",
                        "status": "verified", "verify_date": vday.isoformat(),
                        "seat_bought": "N", "outcome": "miss", "rank": "2"})
        jr = zl.baseline_and_recall()
        assert jr["n_days"] == 1, jr
        assert jr["baseline"]["n"] == 2 and jr["baseline"]["hit"] == 1, jr
        assert jr["baseline"]["pct"] == 50.0
        assert jr["recall"]["n"] == 2 and jr["recall"]["caught"] == 1, jr
        assert jr["recall"]["pct"] == 50.0
        assert jr["by_day"][0]["sealed_n"] == 2 and jr["by_day"][0]["ignited"] == 1
        # 席位明细缺文件 → None（报告层跳过，不触网）
        os.remove(zc.SEAT_TRADES_CSV)
        assert zl.baseline_and_recall() is None
    finally:
        for k, v in old.items():
            setattr(zc, k, v)
    print("PASS ziyang judgement baseline/recall")


def test_ziyang_snap_reuse_and_calendar():
    """快照当日复用窗口 + 交易日历口径（纯函数真值表，零网络）。"""
    import ziyang.seatdata as zs

    anchor = dt.date(2026, 9, 17)
    mk = dt.datetime
    # 复用：当日 22:00 拉取、23:00 重跑（同晚间）；00:30 跨零点重跑仍在窗口
    assert zs._snap_reusable(mk(2026, 9, 17, 22, 0), mk(2026, 9, 17, 23, 0), anchor) is True
    assert zs._snap_reusable(mk(2026, 9, 17, 22, 0), mk(2026, 9, 18, 0, 30), anchor) is True
    # 不复用：盘前 14:00 拉的（盘中数据）；次晨 09:15 后（开盘/已滚动）；跨锚日
    assert zs._snap_reusable(mk(2026, 9, 17, 14, 0), mk(2026, 9, 17, 22, 0), anchor) is False
    assert zs._snap_reusable(mk(2026, 9, 17, 22, 0), mk(2026, 9, 18, 10, 0), anchor) is False
    assert zs._snap_reusable(mk(2026, 9, 18, 22, 0), mk(2026, 9, 18, 23, 0), anchor) is False

    # 日历：周五盘后=周五；周六/周日→周五；周五盘中→周四；长假连跨
    cal = {d.isoformat(): d.weekday() < 5
           for d in [dt.date(2026, 9, i) for i in range(7, 20)]}
    cal[dt.date(2026, 9, 10).isoformat()] = False          # 中秋休市日
    fri, sat, sun, thu = (dt.date(2026, 9, d) for d in (11, 12, 13, 10))
    assert zs._last_trade_from_cal(cal, fri, dt.time(16, 0)) == fri
    assert zs._last_trade_from_cal(cal, sat, dt.time(10, 0)) == fri
    assert zs._last_trade_from_cal(cal, sun, dt.time(10, 0)) == fri
    assert zs._last_trade_from_cal(cal, fri, dt.time(10, 0)) == dt.date(2026, 9, 9)  # 周五盘中越过休市周四
    assert zs._last_trade_from_cal(cal, dt.date(2026, 9, 11), dt.time(10, 0)).isoformat() == "2026-09-09"
    print("PASS ziyang snapshot reuse & calendar")


def test_vane_streak():
    """风向标：本轮板数回溯口径（连板/断板/久远断档重置/MAX_GAP 边界）。"""
    import vane.engine as VE
    # 10 个连续交易日坐标（周一 11-02 起，跳过周末）
    cal = [d.isoformat() for d in
           [dt.date(2026, 11, x) for x in range(2, 7)]          # 11-02 ~ 11-06
           + [dt.date(2026, 11, x) for x in range(9, 14)]]      # 11-09 ~ 11-13
    t = cal[-1]                                                  # 复盘日 11-13
    G = 3                                                        # MAX_GAP 口径
    # 4 连板：D7~D10
    st = VE.streak([cal[6], cal[7], cal[8], cal[9]], t, cal, G)
    assert (st["B"], st["span"], st["breaks"]) == (4, 4, 0)
    # 5天4板断1：D6,D7,D8 + D10（D8→D10 隔1个交易日）
    st = VE.streak([cal[5], cal[6], cal[7], cal[9]], t, cal, G)
    assert (st["B"], st["span"], st["breaks"]) == (4, 5, 1)
    # 断 3 个交易日仍是同一轮（MAX_GAP=3 边界）：D6 + D10
    st = VE.streak([cal[5], cal[9]], t, cal, G)
    assert (st["B"], st["span"], st["breaks"]) == (2, 5, 1)
    # 久远断档重置：D1,D2 与今日隔 7 个交易日 → 本轮只有今日 1 板
    st = VE.streak([cal[0], cal[1], cal[9]], t, cal, G)
    assert (st["B"], st["span"], st["breaks"]) == (1, 1, 0)
    # 断而不散：隔日板 D1,3,5,7,9,10 → 10天6板断4
    st = VE.streak([cal[i] for i in (0, 2, 4, 6, 8, 9)], t, cal, G)
    assert (st["B"], st["span"], st["breaks"]) == (6, 10, 4)
    # 今日未封板无从谈晋级
    assert VE.streak([cal[8]], t, cal, G) is None
    # 标签口径
    assert VE.label({"span": 5, "B": 4, "breaks": 1}) == "5天4板断1"
    assert VE.label({"span": 4, "B": 4, "breaks": 0}) == "4天4板"
    print("PASS vane streak")


def test_vane_ranking():
    """风向标排序：板高 > 断板少 > 跨度紧凑 > 封成比；B∉{4,5} 不入围。"""
    import vane.engine as VE
    rows = [
        {"code": "600002", "B": 4, "span": 4, "breaks": 0, "seal_ratio": 2.0, "fbt": "0930"},   # 4连板
        {"code": "600003", "B": 5, "span": 6, "breaks": 1, "seal_ratio": 1.5, "fbt": "1000"},   # 6天5板断1
        {"code": "600004", "B": 5, "span": 5, "breaks": 0, "seal_ratio": 0.5, "fbt": "1400"},   # 5连板（封成比最差）
        {"code": "600005", "B": 4, "span": 8, "breaks": 2, "seal_ratio": 3.0, "fbt": "0925"},   # 8天4板断2
        {"code": "600006", "B": 6, "span": 6, "breaks": 0, "seal_ratio": 9.0, "fbt": "0930"},   # 6板：不占名额
        {"code": "600007", "B": 3, "span": 3, "breaks": 0, "seal_ratio": 9.0, "fbt": "0930"},   # 3板：不入围
        {"code": "600008", "B": 4, "span": 5, "breaks": 0, "seal_ratio": 0.1, "fbt": "1455"},   # 同板同断比 span 的对照
        {"code": "600009", "B": 5, "span": 6, "breaks": 1, "seal_ratio": None, "fbt": "1000"},  # 封成比缺失 → 垫底
    ]
    picks, bench = VE.select(rows, top_n=3, board_range=(4, 5))
    # 期望序：5连板 > 6天5板断1(有封成比) > 6天5板断1(封成比None垫底) > 4连板 > …
    assert [r["code"] for r in picks] == ["600004", "600003", "600009"]
    assert [r["code"] for r in bench] == ["600002", "600008", "600005"]
    # 6板与3板既不占名额也不进落选
    assert {r["code"] for r in picks + bench} == {"600002", "600003", "600004",
                                                  "600005", "600008", "600009"}
    picks2, _ = VE.select(rows, top_n=2, board_range=(4, 5))
    assert [r["code"] for r in picks2] == ["600004", "600003"]
    print("PASS vane ranking")


def test_vane_scan_pipeline():
    """风向标编排：池数据 → 筛选（主板/ST 过滤、名额、梯队、不足3只场景）。
    在真实消费模块 vane.screen 上打补丁，池行用东财 dict 原始格式喂真 parse_pool。"""
    import vane.screen as VS
    dates = ([dt.date(2026, 11, x) for x in range(2, 7)]        # 11-02 ~ 11-06
             + [dt.date(2026, 11, x) for x in range(9, 14)])    # 11-09 ~ 11-13
    iso = [d.isoformat() for d in dates]                        # 10 个交易日

    def raw(code, name, lb=1):
        return {"c": code, "n": name, "p": 12340, "zdp": 9.97, "amount": 8e7,
                "hs": 500.0, "lbc": lb, "fbt": 93100, "fund": 1.6e8,
                "hybk": "测试业", "ltsz": 5e9}

    # 板链设计（按 iso 索引）：
    #   甲一 [4,5,6,8,9]  → B5 span6 breaks1「6天5板断1」（晋6组第1）
    #   乙二 [6,7,8,9]    → B4 span4 breaks0「4天4板」（晋5组第1）
    #   丁四 [0,2,4,6,8,9]→ B6 span10 breaks4（更高标：不占名额，只进梯队）
    #   庚七 [0,1,2,3]    → 今日断板 → 无从谈晋级，整体跳过
    #   ST戊/己创(300开头)→ 主板与 ST 口径剔除
    pools = {
        iso[0]: [raw("600004", "丁四"), raw("600007", "庚七")],
        iso[1]: [raw("600007", "庚七", 2)],
        iso[2]: [raw("600004", "丁四", 2), raw("600007", "庚七", 3)],
        iso[3]: [raw("600007", "庚七", 4)],
        iso[4]: [raw("600001", "甲一"), raw("600003", "丙三"), raw("600004", "丁四", 3)],
        iso[5]: [raw("600001", "甲一", 2), raw("600003", "丙三", 2)],
        iso[6]: [raw("600001", "甲一", 3), raw("600002", "乙二"),
                 raw("600003", "丙三", 3), raw("600004", "丁四", 4)],
        iso[7]: [raw("600002", "乙二", 2)],
        iso[8]: [raw("600001", "甲一", 4), raw("600002", "乙二", 3),
                 raw("600004", "丁四", 6)],
        iso[9]: [raw("600001", "甲一", 2), raw("600002", "乙二", 4),
                 raw("600004", "丁四", 2), raw("600005", "ST戊"),
                 raw("300006", "己创", 4)],
    }
    orig = (VS.pool_raw, VS.trade_calendar)
    pools_ymd = {k.replace("-", ""): v for k, v in pools.items()}   # 键转 YYYYMMDD
    VS.trade_calendar = lambda: dates
    VS.pool_raw = lambda kind, d: (pools_ymd.get(d) if kind == "zt" else [])
    try:
        res = VS.scan(dt.date(2026, 11, 13))
        assert "error" not in res, res.get("error")
        picks = res["picks"]
        # 甲一5板组第1、乙二4板组第2；丁四B=6不占名额、庚七今日断板 → 名额不足3如实少列
        assert [p["code"] for p in picks] == ["600001", "600002"]
        assert picks[0]["group"] == "5to6" and picks[0]["label"] == "6天5板断1"
        assert picks[1]["group"] == "4to5" and picks[1]["label"] == "4天4板"
        assert picks[0]["rank"] == 1 and picks[1]["rank"] == 2
        assert res["bench"] == []
        assert res["ladder"] == {6: 1, 5: 1, 4: 1}
        assert res["miss_days"] == []                # 全窗口池数据齐备
        assert res["window"] == 10                   # 日历只有 10 日 → 全量用上
        assert res["n_today_mb_zt"] == 3             # 甲一/乙二/丁四（戊/己被剔）
        assert picks[0]["seal_ratio"] == 2.0         # 池 fund/amount 正常换算
    finally:
        VS.pool_raw, VS.trade_calendar = orig
    print("PASS vane scan pipeline")


def test_vane_report_guard():
    """风向标报告守卫：error 结果只返回文本不落盘（保住上次好报告）；
    缺失字段（seal_ratio/ltsz/fbt=None）安全渲染为 —。"""
    import tempfile
    from pathlib import Path as _P
    import vane.config as VC
    from vane import report as VR
    tmp = tempfile.mkdtemp()
    orig = (VC.DATA_DIR, VC.REPORT_TXT, VC.LEDGER_CSV)
    VC.DATA_DIR = tmp
    VC.REPORT_TXT = str(_P(tmp) / "report_latest.txt")
    VC.LEDGER_CSV = str(_P(tmp) / "ledger.csv")     # 不存在 → summary n=0（无台账节）
    try:
        good = {"trade_date": "2026-11-18",
                "picks": [{"rank": 1, "code": "600001", "name": "甲一",
                           "B": 5, "span": 6, "breaks": 1, "label": "6天5板断1",
                           "round_dates": ["2026-11-11", "2026-11-13"],
                           "seal_ratio": None, "fbt": "", "ltsz": None}],
                "bench": [], "ladder": {5: 1}, "miss_days": []}
        txt = VR.render(good)
        assert _P(VC.REPORT_TXT).exists()
        assert "封成比— · —封" in txt and "11-11→11-13" in txt
        # error 结果不覆盖已落盘的好报告
        before = _P(VC.REPORT_TXT).read_text(encoding="utf-8")
        assert "boom" in VR.render({"error": "boom"})
        assert _P(VC.REPORT_TXT).read_text(encoding="utf-8") == before
    finally:
        VC.DATA_DIR, VC.REPORT_TXT, VC.LEDGER_CSV = orig
    print("PASS vane report guard")


def test_vane_section():
    """风向标主报告开头区块（用户约定：单独列出、放报告开头、显眼展示）：
    有票/无候选/未跑三态 + 星级/板链/模型行 + 缺失字段安全渲染 + 版面守卫。"""
    from unittest import mock

    import termstyle as _ts
    import vane
    import vane.ledger as LG
    D = dt.date(2026, 11, 18)
    rows = [
        {"rank": "1", "code": "002148", "name": "北纬科技", "label": "6天5板断1",
         "B": "5", "seal_ratio": "0.04", "fbt": "0930", "industry": "通信服务",
         "score100": "80.0", "model_score": "-1.2", "shadow_rank": "2",
         "round_dates": "2026-11-11|2026-11-13"},
        {"rank": "2", "code": "002685", "name": "华东重机", "label": "4天4板",
         "B": "4", "seal_ratio": "", "fbt": "", "industry": "",
         "score100": "", "model_score": "", "shadow_rank": "1",
         "round_dates": ""},
    ]
    with mock.patch.object(LG, "rows_for", return_value=rows), \
         mock.patch.object(LG, "summary", return_value={"n": 5, "promoted": 2}):
        block = vane.section_lines(D)
    joined = _ts.strip_ansi("\n".join(block))
    assert block[0].startswith("▎情绪风向标") and block[-1] == ""
    assert "★★★" in joined and "#1 002148" in joined          # 星级=score100 分档
    assert "6天5板断1(晋6)" in joined and "模型分-1.20" in joined
    assert "封成比0.04 · 09:30封" in joined and "通信服务" in joined
    assert "板链 11-11→11-13" in joined                        # 竖线链还原箭头链
    assert "封成比— · —封" in joined                           # 缺失安全渲染
    assert "影子#1" in joined
    assert "台账:晋级 2/5" in joined and "python -m vane report" in joined
    with mock.patch.object(LG, "rows_for", return_value=[]), \
         mock.patch.object(LG, "summary", return_value={"n": 0}), \
         mock.patch.object(LG, "scanned_on",
                           return_value={"date": D.isoformat()}):
        empty_block = _ts.strip_ansi("\n".join(vane.section_lines(D)))
    assert "如实空缺" in empty_block
    with mock.patch.object(LG, "rows_for", return_value=[]), \
         mock.patch.object(LG, "scanned_on", return_value=None):
        assert vane.section_lines(D) == []                     # 未跑整节省略
    wide = [l for l in empty_block.splitlines()
            if _ts.disp_width(l) > 100]
    assert not wide, wide                                      # 版面守卫
    # 集成：主报告中 vane 区块位于报告开头（表头框之后、决策摘要之前）
    with mock.patch.object(LG, "rows_for", return_value=rows), \
         mock.patch.object(LG, "summary",
                           return_value={"n": 5, "promoted": 2,
                                         "promote_rate": 40.0}), \
         mock.patch.object(LG, "scanned_on",
                           return_value={"date": D.isoformat()}):
        tmp = "out_vane_section_test"
        txt, _, _ = rpt.write_report([], [], [], D, tmp, signal_only=True,
                                     ctx={"cycle_stage": "发酵"})
        body = _ts.strip_ansi(open(txt, encoding="utf-8").read())
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
    i_vane = body.index("▎情绪风向标")
    i_sum = body.index("▎明日决策摘要")
    i_watch = body.index("▎明日关注清单")
    i_speed = body.index("▎市场情绪速览")
    assert body.rindex("╚") < i_vane < i_sum < i_watch < i_speed  # 开头+顺序
    wide = [l for l in body.splitlines() if _ts.disp_width(l) > 100]
    assert not wide, wide                                      # 整份版面守卫
    print("PASS vane section")


def _vane_model_env(monkey):
    """模型测试公共夹具：合成池窗口 + config 路径重定向 tmp。

    返回 (VM, VC, tmp, dates, orig, saved)。池设计：
      甲一 600001 板链 idx[4,5,6,8,9] → 6天5板断1
      乙二 600002 板链 idx[6,7,8,9]   → 4连板
      丁四 600004 板链 idx[0,2,4,6,8] → 9天5板断4（隔日板）
      庚七 600007 板链 idx[0,1,2]     → 3连板后消失
    """
    import tempfile
    import vane.config as VC
    import vane.model as VM
    import vane.screen as VS
    tmp = tempfile.mkdtemp()
    dates = ([dt.date(2026, 11, x) for x in range(2, 7)]
             + [dt.date(2026, 11, x) for x in range(9, 14)])   # 10 个交易日
    iso = [d.isoformat() for d in dates]

    def raw(code, name, lb=1, amt=8e7):
        return {"c": code, "n": name, "p": 12340, "zdp": 9.97, "amount": amt,
                "hs": 500.0, "lbc": lb, "fbt": 93100, "fund": 1.6e8,
                "hybk": "测试业", "ltsz": 5e9}

    pools = {
        iso[0]: [raw("600004", "丁四"), raw("600007", "庚七")],
        iso[1]: [raw("600007", "庚七", 2)],
        iso[2]: [raw("600004", "丁四", 2), raw("600007", "庚七", 3)],
        iso[3]: [raw("600008", "辛八")],                 # 孤立首板（B=1 被滤）
        iso[4]: [raw("600001", "甲一"), raw("600004", "丁四", 3)],
        iso[5]: [raw("600001", "甲一", 2), raw("600002", "乙二")],
        iso[6]: [raw("600001", "甲一", 3), raw("600002", "乙二", 2),
                 raw("600004", "丁四", 4)],
        iso[7]: [raw("600002", "乙二", 3)],
        iso[8]: [raw("600001", "甲一", 4), raw("600002", "乙二", 4),
                 raw("600004", "丁四", 5)],
        iso[9]: [raw("600001", "甲一", 2), raw("600002", "乙二", 5)],
    }
    orig = (VM.pool_raw, VM.parse_pool, VM._window_days,
            VS.pool_raw, VS.parse_pool, VS.trade_calendar)
    pools_ymd = {k.replace("-", ""): v for k, v in pools.items()}
    VM.pool_raw = lambda kind, d: (pools_ymd.get(d) if kind == "zt" else [])
    VM.parse_pool = data_source.parse_pool
    VM._window_days = lambda td, w=VC.WINDOW: (dates, True)
    VS.pool_raw = VM.pool_raw
    VS.parse_pool = VM.parse_pool
    VS.trade_calendar = lambda: dates
    saved = {k: getattr(VC, k) for k in
             ("DATA_DIR", "MODEL_DIR", "MODEL_JSON", "MODEL_LGR_JSON",
              "MODEL_XGB_JSON", "MODEL_PREV_JSON", "ISO_JSON",
              "RETRAIN_DIFF_JSON", "LEDGER_CSV", "SAMPLES_MIN_ROWS")}
    VC.DATA_DIR = tmp
    VC.MODEL_DIR = tmp
    VC.SAMPLES_MIN_ROWS = 8                      # 合成宇宙 12 行过样本门
    for k in ("MODEL_JSON", "MODEL_LGR_JSON", "MODEL_XGB_JSON",
              "MODEL_PREV_JSON", "ISO_JSON", "RETRAIN_DIFF_JSON",
              "LEDGER_CSV"):
        setattr(VC, k, os.path.join(tmp, k))
    return VM, VC, tmp, dates, orig, saved


def _vane_model_restore(VM, VC, orig, saved, tmp):
    import shutil
    import vane.screen as VS
    (VM.pool_raw, VM.parse_pool, VM._window_days,
     VS.pool_raw, VS.parse_pool, VS.trade_calendar) = orig
    for k, v in saved.items():
        setattr(VC, k, v)
    shutil.rmtree(tmp, ignore_errors=True)


def test_vane_model_samples():
    """三日模型样本构建：宇宙=B≥2 全体、标签=次日在池、双日纪律零泄漏。"""
    VM, VC, tmp, dates, orig, saved = _vane_model_env({})
    try:
        samples, days, miss = VM.build_samples(dt.date(2026, 11, 13))
        assert len(days) == 10 and miss == []
        assert all(s["y"] in (0, 1) for s in samples)
        assert all(s["date"] != "2026-11-13" for s in samples)  # 末日无标签
        m = {(s["code"], s["date"]): s["y"] for s in samples}
        assert m[("600001", "2026-11-10")] == 0        # 次日(11-11)断板
        assert m[("600001", "2026-11-09")] == 1        # 次日(11-10)晋级
        assert m[("600002", "2026-11-10")] == 1        # 乙二次日晋级
        bmap = {(s["code"], s["date"]): s["B"] for s in samples}
        assert bmap[("600001", "2026-11-12")] == 4     # 甲一 4板断1
        assert bmap[("600004", "2026-11-12")] == 5     # 丁四 5板断4
        assert ("600007", "2026-11-03") in m           # B=2 亦入宇宙
        assert ("600008", "2026-11-05") not in m       # 孤立首板 B=1 被滤
        rep = VM.samples_report(dt.date(2026, 11, 13))
        assert "样本" in rep and "基率" in rep
    finally:
        _vane_model_restore(VM, VC, orig, saved, tmp)
    print("PASS vane model samples")


def test_vane_model_train_score():
    """三日模型训练与评分：meta 字段/logreg 工件/主排与影子/缺模型回落。"""
    VM, VC, tmp, dates, orig, saved = _vane_model_env({})
    try:
        import vane.screen as VS
        T_DAY = dt.date(2026, 11, 13)
        meta = VM.train(T_DAY)
        assert meta["ok"] and meta["kind"] == "logreg"  # 合成窗口 < xgb 门
        for k in ("version", "cutoff", "n_rows", "n_pos", "features",
                  "trained_at", "base_rate"):
            assert k in meta, k
        assert os.path.exists(VC.MODEL_JSON)
        assert os.path.exists(VC.MODEL_LGR_JSON)
        assert meta["cutoff"] == "2026-11-12"           # 窗口倒数第二日
        meta2, note2 = VM.ensure_model(T_DAY)
        assert meta2["ok"] and "复用" in note2           # 三日门：当日训练复用
        built = VS.build_rows(T_DAY)
        rows = VM.apply_ranking(built["rows"], T_DAY, meta2)
        assert all(r.get("shadow_rank") for r in rows
                   if int(r["B"]) in VC.BOARD_RANGE)
        scored = [r for r in rows if r.get("model_score") is not None]
        assert scored, "模型分缺失"
        order_key = VM.model_rank_key()
        picks, _ = VS.select(rows, top_n=VC.TOP_N,
                             board_range=VC.BOARD_RANGE, key=order_key)
        ss = [p["model_score"] for p in picks if p["model_score"] is not None]
        assert ss == sorted(ss, reverse=True)           # 主排确定性
        rows0 = VM.apply_ranking(built["rows"], T_DAY, None)
        assert all(r.get("model_score") is None for r in rows0)
        assert all(r.get("shadow_rank") for r in rows0
                   if int(r["B"]) in VC.BOARD_RANGE)    # 缺模型影子照登
    finally:
        _vane_model_restore(VM, VC, orig, saved, tmp)
    print("PASS vane model train/score")


def test_vane_model_ensure():
    """三日门判定：过期重训 / 坏件隔离自愈 / 重训失败同口径回退护栏。"""
    from pathlib import Path as _P
    VM, VC, tmp, dates, orig, saved = _vane_model_env({})
    try:
        T_DAY = dt.date(2026, 11, 13)
        meta = VM.train(T_DAY)
        # ① 过期：trained_at 改回窗口首日 → 距今 9 个交易日 ≥3 → 重训
        stale = dict(meta, trained_at="2026-11-02 10:00:00")
        _P(VC.MODEL_JSON).write_text(json.dumps(stale, ensure_ascii=False))
        meta2, note2 = VM.ensure_model(T_DAY)
        assert meta2["ok"] and "重训" in note2
        # ② 坏件自愈：伪造成 xgb+坏 booster → 可导入环境隔离 .corrupt 重训；
        #   xgboost 缺失环境（如本测试默认解释器）按设计跳过隔离，直接重训回落
        bad = dict(meta2, kind="xgb", trained_at="2026-11-13 09:00:00")
        _P(VC.MODEL_XGB_JSON).write_bytes(b"{L\x00\x00bad")
        _P(VC.MODEL_JSON).write_text(json.dumps(bad, ensure_ascii=False))
        meta3, note3 = VM.ensure_model(T_DAY)
        if VM._xgb_ready():
            assert os.path.exists(VC.MODEL_XGB_JSON + ".corrupt")
        assert meta3["kind"] == "logreg"
        # ③ 回退护栏：置过期 + 重训失败（样本门不过）→ 沿用同口径旧模型
        stale3 = dict(meta3, trained_at="2026-11-02 10:00:00")
        _P(VC.MODEL_JSON).write_text(json.dumps(stale3, ensure_ascii=False))
        orig_bs = VM.build_samples
        VM.build_samples = lambda *a, **k: ([], dates, [])
        try:
            meta4, note4 = VM.ensure_model(T_DAY)
            assert meta4["ok"] and "沿用旧模型" in note4
        finally:
            VM.build_samples = orig_bs
        # ④ 无模型且训练失败 → None（daily_sync 回落启发式）
        os.remove(VC.MODEL_JSON)
        VM.build_samples = lambda *a, **k: ([], dates, [])
        try:
            meta5, note5 = VM.ensure_model(T_DAY)
            assert meta5 is None and "启发式" in note5
        finally:
            VM.build_samples = orig_bs
    finally:
        _vane_model_restore(VM, VC, orig, saved, tmp)
    print("PASS vane model ensure")


def test_auction_guard_and_parse():
    """竞价采集底座：出站 URL 守卫（只放行公网 http/https）+ 腾讯分时解析（离线）。"""
    import socket

    from datasources import auction as DS_A
    from datasources.auction import parse_minute_days
    from datasources.http import assert_public_url

    assert data_source.fetch_minute_days is DS_A.fetch_minute_days     # 门面导出即实现
    # 字面量内网/环回/非 http(s)：数值 IP 不触发 DNS，离线可测
    for bad in ("ftp://web.ifzq.gtimg.cn/x", "http://localhost/x",
                "http://a.local/x", "http://127.0.0.1:8000/x",
                "http://10.0.0.5/x", "http://192.168.1.1/x"):
        try:
            assert_public_url(bad)
        except ValueError:
            continue
        raise AssertionError(f"守卫未拦截: {bad}")
    real = socket.getaddrinfo
    try:
        socket.getaddrinfo = lambda host, *a, **k: [
            (None, None, None, "", ("220.181.38.148", 0))]
        assert_public_url("https://web.ifzq.gtimg.cn/appstock/app/day/query?code=sh600000")
        socket.getaddrinfo = lambda host, *a, **k: [
            (None, None, None, "", ("10.1.2.3", 0))]
        try:
            assert_public_url("https://evil.example.com/x")
        except ValueError:
            pass
        else:
            raise AssertionError("公网域名解析到内网地址未被拦截（重绑定）")

        def _boom(host, *a, **k):
            raise socket.gaierror(8, "nodename nor servname")
        socket.getaddrinfo = _boom
        try:
            assert_public_url("https://nowhere.example.com/x")
        except ValueError:
            pass
        else:
            raise AssertionError("无法解析的主机未被拒绝")
    finally:
        socket.getaddrinfo = real

    node = {"data": [
        {"date": "20261123", "data": ["0930 4.46 1753 781838.00",
                                      "0931 4.47 8869 3978361.00", "bad",
                                      "1530 4.76 202705 93771080.07"]},
        {"date": "20261120", "data": []},
        {"date": "20261119", "data": ["0930 4.85 404 195940.00",
                                      "1530 4.80 243157 118235196.00"]},
    ]}
    days = parse_minute_days(node)
    assert [d["date"] for d in days] == ["20261123", "20261119"]   # 空日剔除、新→旧保持
    assert days[0]["points"][0] == ("0930", 4.46, 1753.0, 781838.0)
    assert days[0]["points"][-1] == ("1530", 4.76, 202705.0, 93771080.07)
    assert len(days[0]["points"]) == 3                            # 坏行跳过
    assert parse_minute_days({}) == []
    assert days[0]["complete"] is False                           # 3 点不算完整交易日
    labels = ([f"09{m:02d}" for m in range(30, 60)]
              + [f"{h:02d}{m:02d}" for h in (10, 11) for m in range(60)]
              + [f"13{m:02d}" for m in range(1, 31)]
              + [f"{h:02d}{m:02d}" for h in (14, 15) for m in range(60)])   # 300 点
    full = parse_minute_days({
        "data": [{"date": "20261123", "data": [
            f"{lab} {4.5 + i * 1e-4} {1000 * (i + 1)} {450000 * (i + 1)}"
            for i, lab in enumerate(labels)]}]})
    assert len(full[0]["points"]) == len(labels) and full[0]["complete"] is True
    bad = parse_minute_days({"data": [{"date": "20261120", "data": [
        "0930 0 100 1000", "0931 -1.5 200 3000", "0932 4.4 -3 4000",
        "0933 4.4 300 -1", "0934 4.4 400 4400"]}]})
    assert [p[0] for p in bad[0]["points"]] == ["0934"]        # 价≤0 / 量额为负 的点剔除
    # 上游无该代码数据（EmptyDataError）→ [] 而不是抛错；瞬时网络失败照抛（交给重试/改日再采）
    from datasources.http import EmptyDataError
    guard, get_json = DS_A.assert_public_url, DS_A._http_get_json
    DS_A.assert_public_url = lambda url: None                   # 离线：跳过 DNS 解析

    def _no_data(*a, **k):
        raise EmptyDataError("响应无 data 字段")

    def _conn_err(*a, **k):
        raise ConnectionError("boom")
    try:
        DS_A._http_get_json = _no_data
        assert DS_A.fetch_minute_days("600000") == []
        DS_A._http_get_json = lambda *a, **k: {"sh600000": node}
        assert [d["date"] for d in DS_A.fetch_minute_days("600000")] == ["20261123", "20261119"]
        DS_A._http_get_json = _conn_err
        try:
            DS_A.fetch_minute_days("600000")
        except ConnectionError:
            pass
        else:
            raise AssertionError("瞬时网络失败被吞掉了")
    finally:
        DS_A.assert_public_url, DS_A._http_get_json = guard, get_json
    print("PASS auction guard and parse")


def test_auction_collect_offline():
    """竞价采集：注入假分时（不联网）→ 台账口径 + 幂等补采 + 守卫跳过 + 读数不判读。"""
    import tempfile

    import auction
    from auction import config as AC

    tmp = tempfile.mkdtemp()
    orig = (AC.DATA_DIR, AC.LEDGER_DIR, AC.MINUTE_DIR, AC.ARCHIVE_DIR)
    AC.DATA_DIR, AC.ARCHIVE_DIR = tmp, os.path.join(tmp, "archive")
    AC.LEDGER_DIR = os.path.join(tmp, "ledger")
    AC.MINUTE_DIR = os.path.join(tmp, "minute")
    AC._ensure_in_data_dir(AC.LEDGER_DIR, AC.MINUTE_DIR)      # 路径守卫对新路径仍成立
    os.makedirs(AC.ARCHIVE_DIR)
    orig_fetch = data_source.fetch_minute_days

    # 时间线：11-19/11-20 两个信号日（归档），执行日 11-20 / 11-23
    d18, t1, t2, x2 = (dt.date(2026, 11, 18), dt.date(2026, 11, 19),
                       dt.date(2026, 11, 20), dt.date(2026, 11, 23))
    cal = [d18, t1, t2, x2]
    arc = pd.DataFrame([
        dict(code="600001", name="甲股", zt_status="封板", close=10.00, prev_close=9.09,
             lianban=1, vol_ratio=0.7, board_score=72),
        dict(code="000002", name="乙股", zt_status="封板", close=20.00, prev_close=18.18,
             lianban=2, vol_ratio=1.6, board_score=55),
        dict(code="000003", name="丙股", zt_status="炸板", close=5.00, prev_close=4.80,
             lianban=0, vol_ratio=2.0, board_score=0),
        dict(code="000004", name="丁股", zt_status="封板", close=8.80, prev_close=8.00,
             lianban=1, vol_ratio=0.6, board_score=0),
        dict(code="000005", name="戊股", zt_status=np.nan, close=7.00, prev_close=6.60,
             lianban=0, vol_ratio=1.0, board_score=0),
    ])
    for t in (t1, t2):
        arc.to_csv(os.path.join(AC.ARCHIVE_DIR, f"signals_{t:%Y%m%d}.csv"),
                   index=False, encoding="utf-8-sig")
    # (竞价价, 竞价量, 竞价额, 盘中价, 收盘, 当日总量, 当日总额)；信号日总量恒 100000 手
    spec = {
        "600001": (10.40, 3500.0, 3.64e7, 11.00, 11.00, 150000.0, 1.55e9),  # +4.0% · 3.5% · 0945 封死
        "000002": (19.60, 2000.0, 3.92e7, 19.40, 19.00, 90000.0, 1.80e9),   # -2.0% · 2.0% · 未触板
        "000003": (5.20, 1000.0, 5.2e6, 5.25, 5.10, 50000.0, 2.5e8),        # 炸板组 · 开盘/涨停价 98.5%
    }
    close_t = {"600001": 10.00, "000002": 20.00, "000003": 5.00}

    def _day(d, pts):
        return {"date": d, "points": pts}

    labels = ([f"09{m:02d}" for m in range(30, 60)]
              + [f"{h:02d}{m:02d}" for h in (10, 11) for m in range(60)]
              + [f"13{m:02d}" for m in range(1, 31)]
              + [f"{h:02d}{m:02d}" for h in (14, 15) for m in range(60)])   # 300 点 ≥ 240

    def _series(o, av, aa, mid, cx, tv, ta):
        """完整日分时：0930 竞价点 → 0945 起为触板价 mid → 末点收盘 cx；量额线性累计。"""
        pts = []
        for i, lab in enumerate(labels):
            px = o if i < 15 else (mid if i < len(labels) - 1 else cx)
            frac = i / (len(labels) - 1)
            pts.append((lab, px, av + (tv - av) * frac, aa + (ta - aa) * frac))
        return pts

    def fake(code):
        if code == "000004":            # 两个执行日均停牌：只有更早的日子 → 不入账
            return [dict(_day("20261118", [("0930", 8.2, 100.0, 8.2e5)]), complete=False)]
        o, av, aa, mid, cx, tv, ta = spec[code]
        ct = close_t[code]
        return [dict(_day("20261123", _series(o, av, aa, mid, cx, tv, ta)), complete=True),
                dict(_day("20261120", _series(ct, 500.0, 5e6, ct, ct, 100000.0, 9.5e8)),
                     complete=True)]

    day23 = os.path.join(AC.LEDGER_DIR, "2026-11-23.csv")
    data_source.fetch_minute_days = fake
    try:
        res = auction.collect(x2, t2)
        assert res["rows"] == 3 and res["codes"] == 4 and res["skipped"] == 1, res
        led = pd.read_csv(day23, dtype={"code": str, "first_limit_time": str})
        assert list(led.columns) == list(AC.COLS)
        assert set(led["code"]) == {"600001", "000002", "000003"}     # 振幅股与停牌股不入账
        a = led[led["code"] == "600001"].iloc[0]
        b = led[led["code"] == "000002"].iloc[0]
        c = led[led["code"] == "000003"].iloc[0]
        assert abs(a["auction_vol_ratio"] - 0.035) < 1e-12            # 竞价量/信号日总量
        assert abs(a["gap"] - 0.04) < 1e-12 and abs(a["ret_oc"] - (11.0 / 10.4 - 1)) < 1e-12
        assert bool(a["touch_limit"]) and bool(a["sealed"]) and a["first_limit_time"] == "09:45"
        assert a["lianban"] == 1 and a["group"] == "封板" and a["exec_date"] == "2026-11-23"
        assert abs(a["limit_t"] - 10.00) < 1e-9                       # 涨停价走 trade_rules
        assert abs(b["auction_vol_ratio"] - 0.02) < 1e-12 and abs(b["gap"] + 0.02) < 1e-12
        assert not bool(b["touch_limit"]) and not bool(b["sealed"])
        assert b["first_limit_time"] != b["first_limit_time"]         # 未触板：时间为空
        assert c["group"] == "炸板" and abs(c["limit_t"] - 5.28) < 1e-9
        assert abs(c["gap_vs_limit"] - (5.2 / 5.28 - 1)) < 1e-12
        assert os.path.exists(os.path.join(AC.MINUTE_DIR, "20261123.pkl"))
        auction.collect(x2, t2)                                        # 重跑 = 整日分片替换
        assert len(pd.read_csv(day23)) == 3
        assert os.listdir(AC.LEDGER_DIR) == ["2026-11-23.csv"]         # 按日分片，无总表

        # 整天成功才落盘：分时截断 / 请求失败 → 不写盘，已落盘分片原样保留
        data_source.fetch_minute_days = lambda code: (
            [dict(fake(code)[0], points=fake(code)[0]["points"][:100], complete=False)]
            if code == "600001" else fake(code))
        res_bad = auction.collect(x2, t2)
        assert res_bad.get("error") and res_bad["truncated"] == 1, res_bad
        assert len(pd.read_csv(day23)) == 3

        def _flaky(code):
            if code == "000002":
                raise ConnectionError("boom")                          # 瞬时网络失败
            return fake(code)
        data_source.fetch_minute_days = _flaky
        res_bad = auction.collect(x2, t2)
        assert res_bad.get("error") and res_bad["failed"] == 1, res_bad
        assert len(pd.read_csv(day23)) == 3
        data_source.fetch_minute_days = fake

        # 守卫（vane 同款）：历史日 / 盘中 / 无日历 → 跳过且台账不变
        assert "非最新" in auction.daily_sync(t2, calendar=cal,
                                              now=dt.datetime(2026, 11, 23, 16, 0))
        assert "盘中" in auction.daily_sync(x2, calendar=cal,
                                            now=dt.datetime(2026, 11, 23, 10, 0))
        assert "交易日历不可用" in auction.daily_sync(x2, calendar=[])
        assert len(auction._load()) == 3
        # 保留期内漏跑的执行日自动补采（11-20 的信号日 11-19 有归档）
        msg = auction.daily_sync(x2, calendar=cal, now=dt.datetime(2026, 11, 23, 16, 0))
        assert "11-20 3行" in msg and "11-23" not in msg, msg
        led = pd.DataFrame(auction._load())
        assert len(led) == 6 and set(led["exec_date"]) == {"2026-11-20", "2026-11-23"}
        # 补采日 11-20 的信号日 11-19 已无分时（保留窗外）→ 占比留空，其余照记
        assert led.loc[led["exec_date"] == "2026-11-20", "auction_vol_ratio"].isna().all()
        assert led.loc[led["exec_date"] == "2026-11-23", "auction_vol_ratio"].notna().all()
        msg2 = auction.daily_sync(x2, calendar=cal, now=dt.datetime(2026, 11, 23, 16, 0))
        assert "无待采" in msg2 and len(auction._load()) == 6
        # 读数（零网络）：样本不足只计数，不判读
        txt = auction.report()
        assert txt.count("样本积累中") == 3, txt
        assert "A3 弱转强(炸板组 开盘/涨停价98~109%): 样本积累中（组内 1 / 组外 1" in txt, txt
        assert all(f"｜{v}" not in txt for v in ("过线", "追踪", "否决")), txt
        # 昨收缺失的炸板行（历史池补录日）无信号日涨停价 → 整行排除出 A3 宇宙，不进对照组
        uni = auction._tests([dict(group="炸板", gap_vs_limit=np.nan, ret_oc=0.01),
                              dict(group="炸板", gap_vs_limit=0.05, ret_oc=0.02)])[2][1]
        assert len(uni) == 1 and uni[0]["gap_vs_limit"] == 0.05
        # 采集失败的日子：daily_sync 告警留痕、不写盘；恢复后下次运行自动补上
        os.remove(day23)
        data_source.fetch_minute_days = _flaky
        msg3 = auction.daily_sync(x2, calendar=cal, now=dt.datetime(2026, 11, 23, 16, 0))
        assert "不落盘" in msg3 and "⚠" in msg3 and not os.path.exists(day23), msg3
        data_source.fetch_minute_days = fake
        msg4 = auction.daily_sync(x2, calendar=cal, now=dt.datetime(2026, 11, 23, 16, 0))
        assert "11-23 3行" in msg4 and len(pd.read_csv(day23)) == 3, msg4
    finally:
        data_source.fetch_minute_days = orig_fetch
        AC.DATA_DIR, AC.LEDGER_DIR, AC.MINUTE_DIR, AC.ARCHIVE_DIR = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS auction collect offline")


def test_fstore_text_contract():
    """文件仓写入契约（2026-11-24）：每行恰好一个行终止符；DataFrame CSV 带 BOM。

    根因：文本模式写入把 csv / to_csv 自带的 "\\r\\n" 再翻译一次 → "\\r\\r\\n"，
    全部台账与归档 CSV 每行多一个冗余字节；另有 to_csv 返回字符串时 encoding
    是空操作，`write_text(path, df.to_csv(encoding="utf-8-sig"))` 写出的是无 BOM 文件。
    """
    from pathlib import Path
    import tempfile

    import fstore

    tmp = Path(tempfile.mkdtemp())
    try:
        p = tmp / "a.csv"
        fstore.write_csv(str(p), ["a", "b"], [{"a": "x", "b": "y"},
                                              {"a": "南 京 港", "b": "1"}])
        raw = p.read_bytes()
        assert b"\r\r\n" not in raw and raw.count(b"\r\n") == 3, raw   # 表头 + 2 行
        assert raw.startswith(b"\xef\xbb\xbf")                        # utf-8-sig
        df = pd.read_csv(p, encoding="utf-8-sig")
        assert list(df.columns) == ["a", "b"] and len(df) == 2 and df.loc[1, "a"] == "南 京 港"

        q = tmp / "df.csv"
        fstore.write_dataframe_csv(str(q), pd.DataFrame({"x": [1, 2], "y": ["甲", "乙"]}))
        rq = q.read_bytes()
        assert rq.startswith(b"\xef\xbb\xbf") and b"\r\r\n" not in rq
        assert rq.count(b"\r\n") == 3 and len(pd.read_csv(q)) == 2

        t = tmp / "t.txt"
        fstore.write_text(str(t), "a\nb\n")             # 纯 LF 文本：字节行为不变
        assert t.read_bytes() == (b"a\r\nb\r\n" if os.linesep == "\r\n" else b"a\nb\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS fstore text contract")


def test_mood_missing_pool_day():
    """池取数失败日不参与周期判定（缺失 ≠ 0 高度）；今日失败 → 如实报未知。

    旧实现把失败日的空指标 `or 0` 读成高度 0：一次池接口失败足以触发
    「高度 N→0 断板 → 退潮期 → 空仓」（2026-11-24 修复）。
    """
    import mood

    def d(h):
        return {"height": h, "promo": 0.3, "break_rate": 0.3, "money": 1.0}

    stage, why = mood.classify_stage([d(7), {"date": "x"}, d(7)])
    assert stage != "退潮", (stage, why)                     # 7→(失败)→7 不是断板
    assert mood.classify_stage([{"date": "x"}, {"date": "y"}]) == \
        ("未知", "序列不足（池数据缺失）")
    assert mood.classify_stage([d(7), d(0)])[0] == "退潮"      # 真实的 0 高度仍算断板

    def row(code, lb):
        return {"code": code, "name": "x", "lianban": lb, "zhaban": 0,
                "fbt": "0930", "industry": "", "fund": float("nan"),
                "ltsz": float("nan"), "turnover": float("nan"), "amount": 1.0}
    pools = {("zt", "20261119"): [row("600001", 6), row("600002", 1)],
             ("zt", "20261120"): [row("600003", 6), row("600004", 1)]}
    dates = [dt.date(2026, 11, 17), dt.date(2026, 11, 18),
             dt.date(2026, 11, 19), dt.date(2026, 11, 20)]
    orig = (mood.data_source.pool_raw, mood.data_source.parse_pool)

    def _ok(kind, ymd):
        return pools.get((kind, ymd), [])                # 取数成功的空池 = 高度 0
    mood.data_source.pool_raw, mood.data_source.parse_pool = _ok, lambda kind, raw: raw or []
    try:
        s, w, series = mood.stage_for_today(dt.date(2026, 11, 20), dates, {})
        assert [x.get("height") for x in series] == [0, 0, 6, 6] and s == "高潮", (s, w)

        def _today_fails(kind, ymd):                    # 今日池取数失败 → None
            return None if ymd == "20261120" else pools.get((kind, ymd), [])
        mood.data_source.pool_raw = _today_fails
        s, w, series = mood.stage_for_today(dt.date(2026, 11, 20), dates, {})
        assert s == "未知" and "取数失败" in w and "height" not in series[-1], (s, w)
    finally:
        mood.data_source.pool_raw, mood.data_source.parse_pool = orig
    print("PASS mood missing pool day")


def test_history_pool_source_label():
    """历史池补录的候选自带 source=backfill 并原样落进归档；未标注的仍是 live。

    calibrate 按 backfill/live 分层（两者口径差 2~4 分），此前补录路径缺这一项
    → report 兜底记成 live，补录日（无昨收/振幅）混进了实跑层（2026-11-24 修复）。
    """
    from pathlib import Path
    zt = [["600000", 0, "测试甲", 11000, 10.0, 1e8, 0, 0, 5.0, 2,
           93000, 93000, 5e7, 0, "行业", {"days": 1, "ct": 1}]]
    cands, _ = data_source.build_candidates_from_pools(zt, [])
    assert cands and all(c["source"] == "backfill" and c["fill"] for c in cands)

    def mk(code, **extra):
        row = {"code": code, "name": "测试", "pools": ["涨停"], "zt_status": "封板",
               "dt_status": "", "open": 10.1, "prev_close": 10.0, "close": 11.0,
               "pct": 10.0, "amp": 8.0, "amount": 1e8, "turnover": 0.08, "ok": True,
               "skip_reason": "", "lianban": 1, "zhaban_times": 0, "industry": "",
               "fbt": "0931", "fund": 1e8, "seal_ratio": 0.5, "ltsz": 4e9,
               "board_pass": False, "board_score": 0, "board_note": "", "vol_ratio": 1.0}
        row.update(extra)
        return row
    rows = [mk("600001", source="backfill"), mk("600002")]
    tmp = Path("out_source_test")
    try:
        rpt.write_report(rows, [], rows, dt.date(2025, 11, 19), str(tmp),
                         signal_only=True, ctx={})
        df = pd.read_csv(tmp / "archive" / "signals_20251119.csv", dtype={"code": str})
        assert df["source"].tolist() == ["backfill", "live"], df["source"].tolist()
        assert list(df.columns) == rpt.CSV_COLUMNS
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS history pool source label")


def test_ledger_tracking_protection():
    """台账核账区（2026-11-24）：读失败不清空历史；锁被占跳过；单本账异常不拖垮其余。

    旧实现 _load 对任何异常返回 []，随后「追加当日行→整体写回」把空账写回——
    一次读故障清空全部历史；且更新排在报告落盘之前、无锁无兜底。
    """
    from pathlib import Path
    import tempfile

    import fstore
    import termstyle as T

    tmp = Path(tempfile.mkdtemp())
    orig = (ledger._load, ledger.open_pool_tracking, ledger.slam_ledger,
            fstore.time.sleep)
    fstore.time.sleep = lambda s: None                 # 锁冲突重试不真等
    empty = pd.DataFrame({"code": ["600001"], "open": [10.0], "close": [10.0],
                          "prev_close": [9.0]})
    try:
        (tmp / "archive").mkdir()
        p = tmp / "archive" / "open_pool_ledger.csv"
        p.write_text("signal_date,code,name\n2026-11-19,600001,甲\n", encoding="utf-8-sig")

        def _denied(path, cols):
            raise PermissionError(13, "used by another process")
        ledger._load = _denied
        out = ledger.tracking_sections(str(tmp), dt.date(2026, 11, 20), empty, [])
        assert any("核账失败" in x for x in out), out
        assert "甲" in p.read_text(encoding="utf-8-sig")        # 历史一行未丢
        ledger._load = orig[0]

        (tmp / "cache").mkdir(exist_ok=True)             # run_lock 已建好锁目录
        lock = tmp / "cache" / ".run.lock"
        lock.touch()                                     # 另一复盘进程持锁
        out = ledger.tracking_sections(str(tmp), dt.date(2026, 11, 20), empty, [])
        assert len(out) == 1 and "跳过" in T.strip_ansi(out[0]), out
        lock.unlink()

        def _slam_boom(*a, **k):
            raise RuntimeError("boom")
        ledger.open_pool_tracking = lambda *a, **k: ["OPEN"]
        ledger.slam_ledger = _slam_boom
        out = ledger.tracking_sections(str(tmp), dt.date(2026, 11, 20), empty, [])
        assert out[0] == "OPEN" and any("打板" in x for x in out), out
    finally:
        (ledger._load, ledger.open_pool_tracking, ledger.slam_ledger,
         fstore.time.sleep) = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS ledger tracking protection")


# ================================================================ 第三轮质量（2026-11-24 全项目审查）
# 每个函数对应一组审查发现；断言旁注明修的是什么，回归时按此定位。

def test_round3_fstore_calendar_prune():
    """fstore 读侧/序列化助手；交易日历 memo + 本地副本兜底；prune 不删锚点之后的缓存。"""
    from pathlib import Path
    import tempfile

    import fstore
    from datasources import calendar as CAL

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a" / "b.json"
        fstore.write_json(p, {"x": 1, "名": "中文"}, indent=2)
        assert fstore.read_json(p) == {"x": 1, "名": "中文"}
        assert fstore.read_json(Path(td) / "nope.json") is None
        assert fstore.read_json(Path(td) / "nope.json", default=[]) == []
        bad = Path(td) / "bad.json"
        bad.write_text("{broken", encoding="utf-8")
        assert fstore.read_json(bad, default={"d": 1}) == {"d": 1}
        q = Path(td) / "x.csv"
        fstore.write_via(q, lambda tmp: Path(tmp).write_text("k,v\n1,2\n", encoding="utf-8"))
        assert q.read_text(encoding="utf-8") == "k,v\n1,2\n"
        assert not [x for x in Path(td).rglob("*") if ".tmp" in x.name]   # 无临时文件残留
        fstore.write_with_bak(q, lambda tmp: Path(tmp).write_text("new", encoding="utf-8"))
        assert (Path(td) / "x.csv.bak").read_text(encoding="utf-8") == "k,v\n1,2\n"
        assert q.read_text(encoding="utf-8") == "new"

    # 交易日历：同进程只拉一次；拉取失败回退本地副本；都没有 = 显式的 []
    cal = [dt.date(2026, 1, 5), dt.date(2026, 11, 20)]
    calls = []
    saved = (CAL._fetch, CAL._MEMO, CAL._MEMO_DAY, CAL.CALENDAR_FILE)
    try:
        with tempfile.TemporaryDirectory() as td:
            CAL.CALENDAR_FILE = os.path.join(td, "trade_calendar.txt")
            CAL._fetch = lambda: (calls.append(1), list(cal))[1]
            CAL._MEMO = None
            assert CAL.trade_calendar() == cal and CAL.trade_calendar() == cal
            assert len(calls) == 1                        # 进程内 memo
            assert Path(CAL.CALENDAR_FILE).exists()       # 成功即落副本
            CAL._MEMO = None                              # 新进程 + 网络失败
            CAL._fetch = lambda: []
            assert CAL.trade_calendar() == cal            # 回退本地副本
            CAL._MEMO = None
            Path(CAL.CALENDAR_FILE).unlink()
            assert CAL.trade_calendar() == []
    finally:
        (CAL._fetch, CAL._MEMO, CAL._MEMO_DAY, CAL.CALENDAR_FILE) = saved

    # prune：日历陈旧（回退到过期副本）时锚点落后于现有缓存，锚点之后的文件不删
    with tempfile.TemporaryDirectory() as td:
        days = [dt.date(2026, 11, 18), dt.date(2026, 11, 19), dt.date(2026, 11, 20)]
        for d in days:
            cache_store.save_result(td, d, [{"code": "600000"}], [], [{"code": "600000"}])
        cache_store.prune(td, [days[0]], days[0], keep=3)
        assert sorted(os.listdir(os.path.join(td, "cache"))) == \
            [f"{d:%Y%m%d}.pkl" for d in days]
        cache_store.prune(td, days, days[-1], keep=1)     # 正常路径照常清窗口外
        assert os.listdir(os.path.join(td, "cache")) == ["20261120.pkl"]
    print("PASS round3 fstore/calendar/prune")


def test_round3_cache_and_bars_guards():
    """save_spot 作废旧结果 / 盘中末根丢弃 / 盘中缓存不播种 / 残缺快照不落缓存。"""
    import tempfile

    import bars as B
    import fstore
    from datasources import spot as S

    day, prev, day2 = (dt.date(2026, 11, 20), dt.date(2026, 11, 19),
                       dt.date(2026, 11, 18))
    hist = pd.DataFrame({"date": pd.bdate_range("2026-11-02", periods=15).date,
                         "close": 1.0, "open": 1.0, "high": 1.0, "low": 1.0,
                         "volume": 1.0})
    with tempfile.TemporaryDirectory() as td:
        cache_store.save_result(td, day, [{"code": "600000"}], [], [{"code": "600000"}])
        assert cache_store.load(td, day).get("results")
        cache_store.save_spot(td, day, pd.DataFrame({"code": ["600000"]}))
        c = cache_store.load(td, day)
        assert c.get("spot") is not None
        # 新快照落地即作废旧结果：此前 ts 刷新而 results 留着，分析阶段一崩，
        # 下次就把「新时间戳 + 旧（可能盘中的）结果」当完整结果复用
        assert not any(k in c for k in ("results", "skipped", "cands", "ctx"))

        cache = {"a": hist, "b": hist.iloc[:-1]}
        assert B.drop_intraday_last(cache, day) == 1
        assert len(cache["a"]) == 14 and cache["a"]["date"].iloc[-1] < day
        assert len(cache["b"]) == 14                        # 本就没有复盘日那根

        # 上一交易日的缓存写于盘中 → 不播种其日K（末根是盘中拼的）
        cache_store.save_result(td, prev, [{"code": "600000"}], [], [{"code": "600000"}],
                                bars={"600000": hist})
        forged = cache_store.load(td, prev)             # 走模块自己的读路径
        forged["ts"] = dt.datetime.combine(prev, dt.time(14, 0)).timestamp()
        fstore.write_pkl(cache_store.file_path(td, prev), forged)
        assert cache_store.is_intraday_cache(cache_store.load(td, prev), prev)
        assert pipeline.seed_bars_from_prev(td, day, [prev, day], {}) == {}

        # 残缺快照：本次可用但不落缓存；完整快照才落
        fake = pd.DataFrame({"code": [f"6000{i:02d}" for i in range(200)],
                             "name": ["测"] * 200, "close": 1.0, "high": 1.0,
                             "low": 1.0, "prev_close": 1.0})
        fake.attrs["partial"] = True
        keep = (S._spot_em_direct, S._spot_sina)
        try:
            S._spot_em_direct = lambda: fake
            S._spot_sina = lambda: (_ for _ in ()).throw(ValueError("不该走到新浪"))
            assert S.fetch_spot(day2, cache_dir=td, force=True) is not None
            assert cache_store.load(td, day2).get("spot") is None
            full = fake.copy()
            full.attrs["partial"] = False
            S._spot_em_direct = lambda: full
            S.fetch_spot(day2, cache_dir=td, force=True)
            assert cache_store.load(td, day2).get("spot") is not None
        finally:
            S._spot_em_direct, S._spot_sina = keep
    print("PASS round3 cache/bars guards")


def test_round3_kline_sources():
    """日K 降级链：无数据不计熔断 / 备用源记忆带 TTL / 新浪锁与请求限时。"""
    import time

    from datasources import kline as K
    from datasources.http import EmptyDataError

    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        return False

    good = pd.DataFrame({"date": [dt.date(2026, 11, 19), dt.date(2026, 11, 20)],
                         "open": 1.0, "close": 1.0, "high": 1.0, "low": 1.0,
                         "volume": 1.0})
    names = ("_daily_em_direct", "_daily_tx_direct", "_daily_sina")
    state = ("_DAILY_OK_FN", "_DAILY_OK_SINCE", "_EM_DAILY_FAILS",
             "_TX_DAILY_FAILS", "_DAILY_FAIL_LOGGED", "_SINA_CALL_TIMEOUT")
    keep = {n: getattr(K, n) for n in names + state}

    def _empty(code, end):
        raise EmptyDataError("该股无数据")

    try:
        K._daily_em_direct = _empty
        K._daily_tx_direct = _empty
        K._daily_sina = lambda code, end: good
        K._EM_DAILY_FAILS = K._TX_DAILY_FAILS = 0
        K._DAILY_OK_FN = None
        K._DAILY_FAIL_LOGGED = set()
        assert K.fetch_daily("600000", dt.date(2026, 11, 20)) is good
        # 该股在前两源无数据 ≠ 源故障：不进熔断计数
        assert K._EM_DAILY_FAILS == 0 and K._TX_DAILY_FAILS == 0
        assert K._pick_daily_sources()[0] is K._daily_sina          # 记忆生效
        K._DAILY_OK_SINCE = time.time() - K._SOURCE_MEMORY_TTL - 1
        assert K._pick_daily_sources()[0] is K._daily_em_direct      # TTL 到期重探首选源
        K._DAILY_OK_FN = None
        assert raises(lambda: K._finalize_bars(pd.DataFrame()), EmptyDataError)
        assert K._SINA_LOCK.acquire(timeout=1)
        K._SINA_CALL_TIMEOUT = 0.5
        try:
            t0 = time.time()
            # 调原函数（上面的替身绕过了锁）
            assert raises(lambda: keep["_daily_sina"]("600000", dt.date(2026, 11, 20)),
                          TimeoutError)
            assert time.time() - t0 < 5                 # 锁被占时不无限等待
        finally:
            K._SINA_LOCK.release()
    finally:
        for n, v in keep.items():
            setattr(K, n, v)
    print("PASS round3 kline sources")


def test_round3_pools_backfill_research_stats():
    """补录涨停池失败即弃 / 回灌无日历拒跑 / 首板一字口径 / Student-t p 值。"""
    from math import erf, sqrt

    import backfill
    import research_scan as R
    from calib.stats import two_sided_p
    from datasources import pools as P

    day = dt.date(2026, 11, 20)
    keep = P.pool_raw
    try:
        P.pool_raw = lambda kind, d: (None if kind == "zt"
                                      else [{"c": "600001", "n": "乙", "p": 10000}])
        # 涨停池失败不再「部分成功也返回」零封板候选
        assert P.fetch_history_pool_candidates(day) == (None, None)
        P.pool_raw = lambda kind, d: (
            [{"c": "600001", "n": "甲", "p": 11000, "zdp": 10.0, "lbc": 1}]
            if kind == "zt" else None)
        cands, extras = P.fetch_history_pool_candidates(day)
        assert cands[0]["zt_status"] == "封板" and "zhaban_times" not in extras["600001"]
    finally:
        P.pool_raw = keep

    keep = data_source.trade_calendar
    data_source.trade_calendar = lambda: []
    try:
        assert backfill.collect_trade_days(day, 21) == []        # 无日历拒跑（防周末回声）
    finally:
        data_source.trade_calendar = keep

    # 首板一字 = 首板日开盘缺口（i-1）；事件日（i）缺口另列 event_word
    d = pd.bdate_range("2026-09-01", periods=12).date
    close = [10.0] * 12
    close[7], close[8], close[9] = 11.0, 12.1, 12.5
    open_ = list(close)
    open_[7], open_[8], open_[9] = 11.0, 11.5, 12.2
    ev = R.scan_events(pd.DataFrame({"date": d, "open": open_, "close": close,
                                     "volume": [100.0] * 12}), dt.date(2026, 8, 1))
    assert len(ev) == 1
    assert ev[0]["first_word"] is True and abs(ev[0]["gap_first"] - 0.10) < 1e-9
    assert ev[0]["event_word"] is False and abs(ev[0]["gap_event"] - 0.0455) < 1e-9

    # p 值：缺省口径逐位不变；给 df 走 Student t（小簇数更保守）
    assert abs(two_sided_p(2.0) - (1 - erf(2.0 / sqrt(2)))) < 1e-15
    assert two_sided_p(2.0, df=7) > two_sided_p(2.0)
    assert abs(two_sided_p(2.365, df=7) - 0.05) < 6e-4
    assert abs(two_sided_p(2.0, df=7) - 0.0857) < 2e-3
    assert two_sided_p(None, df=7) is None and two_sided_p(0.0, df=5) == 1.0
    assert two_sided_p(2.0, df=0) is None
    assert abs(two_sided_p(2.0, df=30) - two_sided_p(2.0)) < 1e-15
    print("PASS round3 pools/backfill/research/stats")


def test_round3_auction_report_gates():
    """竞价报告：缺失值不进对照组；配对日不足只报样本量、不落否决；m=3。"""
    import auction as A

    rows = ([{"group": "封板", "auction_vol_ratio": r, "gap": 0.02, "lianban": 1,
              "ret_oc": v, "exec_date": d}
             for d in ("2026-11-19", "2026-11-20")
             for r, v in [(0.05, 0.02)] * 15 + [(0.01, -0.02)] * 15]
            + [{"group": "封板", "auction_vol_ratio": None, "gap": None, "lianban": 1,
                "ret_oc": -0.5, "exec_date": "2026-11-19"}] * 5)
    uni = {name: u for name, u, _ in A._tests(rows)}
    assert all(r["auction_vol_ratio"] is not None for r in uni["A1 竞价量占比≥3%"])
    assert all(r["gap"] is not None for r in uni["A2 黄金分割高开3.82~6.18%"])
    assert uni["A3 弱转强(炸板组 开盘/涨停价98~109%)"] == []
    keep = A._load
    A._load = lambda: rows
    try:
        txt = A.report()
    finally:
        A._load = keep
    body = txt.splitlines()[:-1]                       # 末行是口径图例（含「过线须」字样）
    assert not any(v in ln for ln in body for v in ("否决", "过线", "追踪")), txt
    assert "配对日 2<8，不判读" in txt and txt.count("样本积累中") == 3, txt
    print("PASS round3 auction report gates")


def test_round3_vane_model_space():
    """vane：推理必须走训练时的标准化空间；缺标准化参数的 meta 不可复用；
    日历不可用不登记；训练样本只取缺池日之后的连续覆盖尾段。"""
    import xgboost as xgb
    from vane import screen as VS

    VM, VC, tmp, dates, orig, saved = _vane_model_env({})
    try:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        mu, sd = X.mean(axis=0), X.std(axis=0)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=300) > 0).astype(int)
        ref = xgb.train({"max_depth": 2, "verbosity": 0},
                        xgb.DMatrix((X - mu) / sd, label=y), num_boost_round=20)
        art = os.path.join(tmp, "fake_xgb.json")
        ref.save_model(art)
        VM._BOOTER_CACHE.clear()
        meta = {"kind": "xgb", "mu": mu.tolist(), "sd": sd.tolist()}
        p = VM._predict_proba(meta, art, X)
        assert np.allclose(p, ref.predict(xgb.DMatrix((X - mu) / sd)), atol=1e-6)
        assert not np.allclose(p, ref.predict(xgb.DMatrix(X)))    # 旧喂法确实不同
        assert not VM._same_version({"ok": True, "version": VC.MODEL_VERSION,
                                     "features": list(VC.FEATURES)})

        VS.trade_calendar = lambda: []
        built = VS.build_rows(dt.date(2026, 11, 13))
        assert built.get("error") and "交易日历" in built["error"]
        VS.trade_calendar = lambda: dates

        base_raw = VM.pool_raw
        miss_key = f"{dates[7]:%Y%m%d}"
        VM.pool_raw = lambda kind, d: (None if d == miss_key else base_raw(kind, d))
        samples, sdays, miss = VM.build_samples(dt.date(2026, 11, 13))
        assert f"{dates[7]:%Y-%m-%d}" in miss
        assert sdays == dates[8:] and samples == []        # 尾段不足 3 日 → 无样本，不落伪负例
        VM.pool_raw = base_raw
    finally:
        _vane_model_restore(VM, VC, orig, saved, tmp)
    print("PASS round3 vane model space")


def test_round3_ziyang_guards():
    """ziyang：续训标签已发布水位；登记被拒不落榜单快照。"""
    import tempfile
    from pathlib import Path

    import ziyang.config as ZC
    import ziyang.ledger as ZL
    from ziyang import scorer as ZS

    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "seat.csv", Path(td) / "zcl.csv"
        a.write_text("交易日期,股票代码,买入金额,卖出金额\n2026-11-20,600000,1,0\n",
                     encoding="utf-8-sig")
        b.write_text("交易日期,股票代码,买入金额,卖出金额\n2026-11-18,600000,1,0\n",
                     encoding="utf-8-sig")
        keep = (ZC.SEAT_TRADES_CSV, ZS._ZCL_CSV)
        ZC.SEAT_TRADES_CSV, ZS._ZCL_CSV = str(a), str(b)
        try:
            assert ZS._labels_covered_through() == "2026-11-18"    # 两份里较早的那份
            b.write_text("broken", encoding="utf-8")
            assert ZS._labels_covered_through() == ""              # 一份读坏 = 标签不可信
        finally:
            ZC.SEAT_TRADES_CSV, ZS._ZCL_CSV = keep

        snaps = []
        keep = (ZL.persist_ranking, ZC.LEDGER_CSV, ZL.now_cn)
        ZL.persist_ranking = lambda d, ranked: snaps.append((d, len(ranked)))
        ZC.LEDGER_CSV = str(Path(td) / "ledger.csv")
        try:
            pred = {"code": "600001", "name": "甲", "score": 1.0}
            assert ZL.register(dt.date(2026, 11, 20), [pred],
                               ranked=[pred]).endswith("登记 1 条预判")
            assert len(snaps) == 1
            # 跨零点重跑：不是「当日」，登记被拒 → 台账与榜单都不动
            ZL.now_cn = lambda: dt.datetime(2026, 11, 21, 10, 0)
            msg = ZL.register(dt.date(2026, 11, 20), [dict(pred, code="600002")],
                              regen_same_day=True, ranked=[pred])
            assert "拒绝覆写" in msg and len(snaps) == 1
        finally:
            ZL.persist_ranking, ZC.LEDGER_CSV, ZL.now_cn = keep

    # 日K增量拼接：全空列不得触发 pandas 弃用警告（2026-11-24 一次运行刷 283 条，
    # 占日志 98%）。触发形态=新拉片段带同名的 object 型全空列，旧段同列是实值
    import warnings
    from ziyang import seatdata as ZSD
    from ziyang import scorer as ZS2
    have = pd.DataFrame({"date": ["2026-11-10", "2026-11-11"], "close": 1.0,
                         "amount": 2.0, "turnover": 3.0, "pct": np.nan})
    have["pct"] = have["pct"].astype(object)
    add = pd.DataFrame({"date": ["2026-11-11", "2026-11-12"], "close": 1.0,
                        "amount": None, "turnover": None})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = ZSD._splice_bars(have, add)
    assert not caught, [str(w.message)[:60] for w in caught]
    assert list(merged.columns) == list(have.columns) and len(merged) == 4
    assert merged["amount"].dtype == np.float64 and merged["pct"].isna().all()
    # 上一代 booster 归档名保留 .json 扩展名（xgboost 按扩展名选加载格式）
    assert ZS2._prev_xgb_path().endswith(".prev.json")
    print("PASS round3 ziyang guards")


if __name__ == "__main__":
    test_main_board()
    test_limit_prices()
    test_build_candidates()
    test_cache_is_valid()
    test_window_dates()
    test_cache_store()
    test_pool_candidates()
    test_pool_candidates_dict()
    test_norm_pool_row_units()
    test_incremental_bars()
    test_second_board()
    test_gap_score_single_source()
    test_em_spot_units()
    test_watch_tier()
    test_open_pool()
    test_open_pool_tracking()
    test_slam_ledger()
    test_cycle_series()
    test_doctor()
    test_board_report()
    test_calibrate_summarize()
    test_cluster_inference()
    test_calibrate_forward_states()
    test_forward_tradable_metrics()
    test_forward_cache_schema()
    test_source_stratification()
    test_calibrate_features()
    test_calibrate_oos()
    test_ledger_health()
    test_report_plain_file()
    test_name_normalization()
    test_wrap_helpers()
    test_stats_line_daily_equal_weight()
    test_report_summary()
    test_no_lookahead_fields()
    test_module_contracts()
    test_limit_price_matches_exchange()
    test_calib_facade()
    test_forward_safe_write()
    test_research_scan_window_and_lianban()
    test_ledger_backfill_settlement()
    test_watchlist_block()
    test_watchlist_unified_score()
    test_ziyang_progress()
    test_ziyang_bars_batch()
    test_ziyang_bars_batch_retry()
    test_ziyang_snapshot_fresh()
    test_ziyang_baostock_layer()
    test_ziyang_judgement_baseline_recall()
    test_ziyang_snap_reuse_and_calendar()
    test_vane_streak()
    test_vane_ranking()
    test_vane_scan_pipeline()
    test_vane_report_guard()
    test_vane_section()
    test_vane_model_samples()
    test_vane_model_train_score()
    test_vane_model_ensure()
    test_auction_guard_and_parse()
    test_auction_collect_offline()
    test_fstore_text_contract()
    test_mood_missing_pool_day()
    test_history_pool_source_label()
    test_ledger_tracking_protection()
    test_round3_fstore_calendar_prune()
    test_round3_cache_and_bars_guards()
    test_round3_kline_sources()
    test_round3_pools_backfill_research_stats()
    test_round3_auction_report_gates()
    test_round3_vane_model_space()
    test_round3_ziyang_guards()
    test_trade_rules()
    test_mood_stages()
    test_concept_boards_parse()
    test_end_to_end_report()
    print("\n全部离线测试通过")
