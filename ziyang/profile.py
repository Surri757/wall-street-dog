# -*- coding: utf-8 -*-
"""画像层：席位交易画像挖掘（2026-09-20 五路深挖后扩为多维画像）。

画像结构（profile.json 各节）：
1. style      —— 行为统计（买卖比/主板占比/买入日涨停率/后市表现/活跃度）
2. eve        —— 点火前夜六维分布（连板/涨幅/换手/量比/价位/20日位置）
3. eve_split  —— 前夜画像按连板细分（0板回调点火 vs 1板接力点火两种打法）
4. ltsz       —— 市值维度（前夜流通市值档；股本取当前值近似，档宽下可接受）
5. power      —— 点火力度（净买/当日成交；引导型 vs 锁筹型的判别证据）
6. sell       —— 卖出纪律（持仓周期/隔日兑现/卖后走势/做T痕迹）
7. regime     —— 环境与节奏（出勤率/情绪阶段偏好/连买/一日多票）
8. timing     —— 封板时点（主工程归档交集，覆盖有限如实标注）
9. theme      —— 题材偏好（近60日行业分布）
10. deep_refs —— 静态深挖产物指针（联动席位/网络情报，非每日刷新）

预判层（screen.py）用 eve 分布对当日候选打分；台账（ledger.py）裁判准确性。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
from pathlib import Path

import fstore
import pandas as pd

from . import config as C
from . import features as F
from .progress import Progress
from .seatdata import (feats_from_bars, fetch_bars_batch, idx_on,
                       save_bars_cache)


# ---------------------------------------------------------------- 前夜特征
def mine_ignition_events(seat_trades: pd.DataFrame, bars_cache: dict,
                         float_shares: dict | None = None) -> pd.DataFrame:
    """点火事件 → 前夜特征明细表（out 不落盘，由 build_profile 汇总入 profile.json）。

    前夜特征 = feats_from_bars(bars, 点火日, back=1)——与预判层共用同一实现
    （seatdata.feats_from_bars），保证「挖掘口径=打分口径」。日K批量并发拉取
    （fetch_bars_batch，逐只含增量拼接+复权锚校验，checkpoint 落盘）。
    """
    ev = seat_trades[seat_trades["主板"] & (seat_trades["净买额"] > C.IGNITION_MIN_NET_BUY)
                     & (seat_trades["涨跌幅"] >= C.LIMIT_UP_APPROX)
                     & (seat_trades["交易日期"] >= dt.date.fromisoformat(C.PROFILE_SINCE))]
    ev = ev.copy()
    ev["交易日期"] = pd.to_datetime(ev["交易日期"]).dt.date
    # 每只股票拉到其最大事件日（一次请求服务该股全部事件）
    ends = dict(ev.groupby("股票代码")["交易日期"].max())
    fetch_bars_batch(bars_cache, ends, tag="画像·日K")
    save_bars_cache(bars_cache)

    rows = []
    pg = Progress("画像·前夜特征", len(ev))
    for k, (_, r) in enumerate(ev.iterrows(), 1):
        bars = bars_cache.get(r["股票代码"])
        if bars is not None and len(bars):
            row = feats_from_bars(bars, r["交易日期"], back=1,
                                  float_shares=(float_shares or {}).get(r["股票代码"]))
            if row is not None:
                # 市值（亿）＝前夜收盘×当前流通股本（送转会漂移，30/50/100/200
                # 亿档宽下可接受；deep/timing_power 归档 D 日实测口径交叉验证）；
                # D 日成交额（亿）供点火力度分母，腾讯行缺额时 OHLC 均价近似
                shares = (float_shares or {}).get(r["股票代码"])
                row["ltsz"] = shares * row["price"] / 1e8 if shares else None
                i = idx_on(bars, r["交易日期"])
                amt = bars["amount"].iloc[i] if i is not None and "amount" in bars.columns else None
                if (amt is None or pd.isna(amt)) and i is not None:
                    b2 = bars.iloc[i]
                    amt = b2["volume"] * 100 * (b2["open"] + b2["high"]
                                                + b2["low"] + b2["close"]) / 4
                row["d_amount_yi"] = (float(amt) / 1e8
                                      if amt is not None and not pd.isna(amt) else None)
                row.update({"ign_date": r["交易日期"], "code": r["股票代码"],
                            "name": r["名称"], "net_buy": float(r["净买额"]),
                            "ign_pct": float(r["涨跌幅"])})
                rows.append(row)
        pg.update(k)
    pg.done()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 深度画像辅助
POWER_LABELS = ("<3%", "3~5%", "5~10%", "10~20%", ">20%")
POWER_EDGES = (0, 0.03, 0.05, 0.10, 0.20, math.inf)
SELL_PCT_LABELS = ("跌停/深跌", "-5~0", "0~5", "5~9.5", "涨停/大涨")
SELL_PCT_EDGES = (-math.inf, -5, 0, 5, 9.5, math.inf)
FBT_BUCKETS = ((0, 1000, "早盘<10:00"), (1000, 1130, "10:00~11:30"),
               (1130, 1400, "11:30~14:00"), (1400, 2400, "14:00后"))


def _bucketize(vals, edges, labels) -> dict:
    s = pd.Series(list(vals)).dropna()
    if not len(s):
        return {}
    cut = pd.cut(s.astype(float), bins=list(edges), labels=list(labels), right=False)
    vc = cut.value_counts()
    return {lab: {"n": int(vc.get(lab, 0)),
                  "pct": round(100.0 * float(vc.get(lab, 0)) / len(s), 1)}
            for lab in labels if int(vc.get(lab, 0))}


def _sell_profile(st: pd.DataFrame, bars_cache: dict) -> dict:
    """卖出纪律：持仓周期/隔日兑现/卖出日状态/卖后走势/做T痕迹。

    口径：st=(日,股)去重净事件；配对=净买→同股下一个净卖；持有收益为
    收盘价近似；卖后走势取东财后视列（仅覆盖上榜披露行）。
    """
    st = st.sort_values(["股票代码", "交易日期"])
    dates_seq = sorted(st["交易日期"].unique())
    pos = {d: i for i, d in enumerate(dates_seq)}
    per_code: dict[str, list] = {}
    for tup in st[["股票代码", "交易日期", "净买额"]].itertuples(index=False):
        per_code.setdefault(tup[0], []).append((tup[1], float(tup[2])))

    holding, pairs, next_day_sell, buys_n = [], [], 0, 0
    for code, evs in per_code.items():
        for j, (d, net) in enumerate(evs):
            if net <= 0:
                continue
            buys_n += 1
            # 配对口径：每个净买事件 → 同股其后**首个**净卖事件（跳过中间的
            # 加仓买点，连续买入日的每一笔都各自配对——与深挖基准 n=190 一致）
            nxt_sell = next((evs[k] for k in range(j + 1, len(evs)) if evs[k][1] < 0), None)
            if nxt_sell is not None:
                gap = pos[nxt_sell[0]] - pos[d]
                holding.append(gap)
                pairs.append((code, d, nxt_sell[0]))
                if gap == 1:
                    next_day_sell += 1

    rets = []
    for code, d1, d2 in pairs:
        b = bars_cache.get(code)
        if b is None:
            continue
        i1, i2 = idx_on(b, d1), idx_on(b, d2)
        if i1 is not None and i2 is not None and i1 < i2:
            c1 = float(b["close"].iloc[i1])
            if c1 > 0:
                rets.append((float(b["close"].iloc[i2]) / c1 - 1.0) * 100.0)

    # 做T痕迹与卖后走势：回到原始双边行（净事件看不到日内双边）
    raw = pd.read_csv(C.SEAT_TRADES_CSV, dtype={"股票代码": str}, encoding="utf-8-sig")
    raw["交易日期"] = pd.to_datetime(raw["交易日期"]).dt.date
    raw = raw[raw["交易日期"] >= dt.date.fromisoformat(C.PROFILE_SINCE)]
    g = raw.groupby(["交易日期", "股票代码"]).agg(买=("买入金额", "max"), 卖=("卖出金额", "max"))
    t_days = int(((g["买"] > 5e7) & (g["卖"] > 5e7)).sum())

    sells = st[st["净买额"] < 0]
    fwd = {}
    if len(sells):
        for k, col in (("d1", "1日后涨跌幅"), ("d3", "3日后涨跌幅"), ("d5", "5日后涨跌幅")):
            v = pd.to_numeric(raw.merge(
                sells[["交易日期", "股票代码"]].assign(_k=1),
                on=["交易日期", "股票代码"], how="inner"
            ).groupby(["交易日期", "股票代码"], as_index=False)[col].max()[col],
                errors="coerce").dropna()
            if len(v):
                fwd[k] = {"n": int(len(v)), "median": round(float(v.median()), 2),
                          "mean": round(float(v.mean()), 2),
                          "win": round(100.0 * float((v > 0).mean()), 1)}

    hd, rs = pd.Series(holding), pd.Series(rets)
    return {
        "n_pairs": len(pairs),
        "buys_no_later_sell": buys_n - len(pairs),
        "holding_dist": {"T+1": int((hd == 1).sum()),
                         "2~5日": int(((hd >= 2) & (hd <= 5)).sum()),
                         "6~10日": int(((hd >= 6) & (hd <= 10)).sum()),
                         ">10日": int((hd > 10).sum()),
                         "median_days": float(hd.median()) if len(hd) else None},
        "next_day_sell_pct": round(100.0 * next_day_sell / buys_n, 1) if buys_n else None,
        "sell_day_pct_dist": _bucketize(sells["涨跌幅"], SELL_PCT_EDGES, SELL_PCT_LABELS),
        "after_sell": fwd,
        "holding_ret": {"n": len(rets), "median": round(float(rs.median()), 2),
                        "mean": round(float(rs.mean()), 2),
                        "win": round(100.0 * float((rs > 0).mean()), 1)} if len(rets) else {},
        "t_trade_days": t_days,
        "note": "净额口径（单边榜未披露侧按0）；持有收益为收盘价近似",
    }


def _regime_profile(st: pd.DataFrame) -> dict:
    """环境与节奏：出勤率/情绪阶段偏好（归档窗口）/连买/一日多票。

    cycle_series 覆盖随主工程归档增长，阶段结论窗口受限时如实标注。
    """
    dates_seq = sorted(st["交易日期"].unique())
    pos = {d: i for i, d in enumerate(dates_seq)}
    buys = st[st["净买额"] > 0]
    active_days = buys["交易日期"].nunique()

    buy_keys = set(zip(buys["股票代码"], buys["交易日期"]))
    repeat = sum(1 for (c, d) in buy_keys
                 if pos.get(d, -1) + 1 < len(dates_seq)
                 and (c, dates_seq[pos[d] + 1]) in buy_keys)
    max_streak, streak_case = 0, None
    per_code: dict[str, list] = {}
    for (c, d) in buy_keys:
        per_code.setdefault(c, []).append(d)
    for c, ds in per_code.items():
        ds.sort()
        run, best = 1, None
        for i in range(1, len(ds)):
            run = run + 1 if pos[ds[i]] == pos[ds[i - 1]] + 1 else 1
            if run > max_streak:
                max_streak, best = run, (c, ds[i - run + 1], ds[i])
                streak_case = best
    daily_n = buys.groupby("交易日期").size()

    stage, stage_window = {}, ""
    cs_path = os.path.join(C.ROOT, "out", "archive", "cycle_series.csv")
    if os.path.exists(cs_path):
        try:
            cs = pd.read_csv(cs_path)
            cs["date"] = pd.to_datetime(cs["date"]).dt.date
            cs = cs.dropna(subset=["stage"])
            if len(cs):
                stage_window = f"{min(cs['date'])}~{max(cs['date'])}"
                buys_d = buys[["交易日期"]].merge(
                    cs, left_on="交易日期", right_on="date", how="inner")
                day_share = cs["stage"].value_counts(normalize=True)
                for s, cnt in buys_d["stage"].value_counts().items():
                    days_s = int((cs["stage"] == s).sum())
                    stage[s] = {"events": int(cnt), "days": days_s,
                                "events_per_day": round(float(cnt) / days_s, 2) if days_s else None,
                                "baseline_day_share_pct": round(100.0 * float(day_share[s]), 1)}
        except Exception:
            stage = {}
    return {
        "attendance_pct": round(100.0 * active_days / len(dates_seq), 1) if dates_seq else None,
        "active_days": int(active_days), "total_days": len(dates_seq),
        "stage_pref": stage, "stage_window": stage_window,
        "repeat_buy_pct": round(100.0 * repeat / len(buy_keys), 1) if buy_keys else None,
        "max_streak": ({"days": max_streak, "code": streak_case[0],
                        "from": str(streak_case[1]), "to": str(streak_case[2])}
                       if streak_case and max_streak >= 2 else None),
        "daily_tickets": {"p50": int(daily_n.median()), "max": int(daily_n.max()),
                          "multi_ticket_day_pct": round(100.0 * float((daily_n >= 2).mean()), 1)}
        if len(daily_n) else {},
    }


def _timing_from_archive(ign: pd.DataFrame) -> dict:
    """封板时点：主工程归档（涨停池含 fbt/zhaban）× 点火事件交集。覆盖有限，如实标注。"""
    from .seatdata import archive_signal_files
    fbts, zhbs = [], []
    for d, f in archive_signal_files():
        day_rows = ign[ign["ign_date"] == d]
        if not len(day_rows):
            continue
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            continue
        df["code"] = df["code"].str.zfill(6)
        m = df[df["code"].isin(set(day_rows["code"]))]
        for _, r in m.iterrows():
            try:
                fbts.append(int(float(r.get("fbt"))))
            except (TypeError, ValueError):
                pass
            try:
                zhbs.append(int(float(r.get("zhaban_times"))))
            except (TypeError, ValueError):
                pass
    s, zs = pd.Series(fbts).dropna(), pd.Series(zhbs).dropna()
    fbt_bucket = {lab: int(((s >= lo) & (s < hi)).sum()) for lo, hi, lab in FBT_BUCKETS}
    med = "%02d:%02d" % (int(s.median()) // 100, int(s.median()) % 100) if len(s) else None
    return {"matched": int(len(s)),
            "note": "主工程归档窗口内的交集样本，覆盖有限（方向性参考）",
            "fbt_dist": fbt_bucket, "fbt_median": med,
            "zhaban_0_pct": round(100.0 * float((zs == 0).mean()), 1) if len(zs) else None}


# ---------------------------------------------------------------- 画像汇总
def _dist(series: pd.Series, labels) -> dict:
    """分类桶分布：{label: {n, pct}}（缺失桶不计入分母）。"""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {}
    vc = s.value_counts()
    return {lab: {"n": int(vc.get(lab, 0)),
                  "pct": round(100.0 * vc.get(lab, 0) / n, 1)} for lab in labels}


def _fwd_stats(raw: pd.DataFrame) -> dict:
    """席位净买入事件的后市表现（东财自带 1/3/5/10 日后涨跌幅，%）。"""
    raw = raw[pd.to_datetime(raw["交易日期"]).dt.date >= dt.date.fromisoformat(C.PROFILE_SINCE)]
    g = raw.groupby(["交易日期", "股票代码"], as_index=False).agg(
        买入=("买入金额", "max"), 卖出=("卖出金额", "max"),
        d1=("1日后涨跌幅", "max"), d3=("3日后涨跌幅", "max"),
        d5=("5日后涨跌幅", "max"), d10=("10日后涨跌幅", "max"))
    g = g[(g["买入"] - g["卖出"].fillna(0)) > 0]
    out = {}
    for k, col in (("d1", "d1"), ("d3", "d3"), ("d5", "d5"), ("d10", "d10")):
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        out[k] = {"n": int(len(v)), "mean": round(float(v.mean()), 2),
                  "median": round(float(v.median()), 2),
                  "win": round(100.0 * float((v > 0).mean()), 0)} if len(v) else {}
    return out


def _evolve(st: pd.DataFrame, ign: pd.DataFrame) -> tuple[dict, dict]:
    """画像每日迭代细描（2026-11-22 用户定向）：7日 vs 30日 滚动对比。

    画像随 daily_sync 每日重建是既有行为；本函数补上「迭代是否在收敛」的
    可观测面——关键行为指标按近 7 日 / 近 30 日双窗对比，漂移即细描方向的
    证据（如 0板点火占比抬升=回调点火风格强化）。返回 (trend, hist_row)。
    """
    today = dt.date.today()

    def _win(days: int) -> dict:
        lo = today - dt.timedelta(days=days)
        b = st[(st["交易日期"] >= lo) & (st["净买额"] > 0)]
        ig = ign[ign["ign_date"] >= lo]
        pw = (ig["net_buy"] / (ig["d_amount_yi"] * 1e8)).dropna()
        return {
            "days": int(b["交易日期"].nunique()),
            "buy_events": int(len(b)),
            "ignitions": int(len(ig)),
            "ignition_rate_pct": (round(100.0 * len(ig) / len(b), 1)
                                  if len(b) else None),
            "zero_board_share_pct": (round(100.0 * float((ig["lianban"] == 0).mean()), 1)
                                     if len(ig) else None),
            "power_median_pct": (round(100.0 * float(pw.median()), 1)
                                 if len(pw) else None),
            "net_buy_yi_median": (round(float(b["净买额"].median()) / 1e8, 3)
                                  if len(b) else None),
        }

    w7, w30 = _win(7), _win(30)
    delta = {}
    for k in ("ignition_rate_pct", "zero_board_share_pct", "power_median_pct",
              "net_buy_yi_median"):
        a, b = w7.get(k), w30.get(k)
        delta[k] = round(a - b, 1) if (a is not None and b is not None) else None
    trend = {"w7": w7, "w30": w30, "delta": delta,
             "note": "7日 vs 30日滚动对比，随 daily_sync 每日迭代（profile_history.csv 留轨迹）"}
    hist = {"date": today.isoformat()}
    for pre, w in (("w7", w7), ("w30", w30)):
        for k, v in w.items():
            hist[f"{pre}_{k}"] = v
    return trend, hist


HIST_COLS = ["date", "w7_days", "w7_buy_events", "w7_ignitions",
             "w7_ignition_rate_pct", "w7_zero_board_share_pct",
             "w7_power_median_pct", "w7_net_buy_yi_median",
             "w30_days", "w30_buy_events", "w30_ignitions",
             "w30_ignition_rate_pct", "w30_zero_board_share_pct",
             "w30_power_median_pct", "w30_net_buy_yi_median"]


def _save_history(hist: dict) -> None:
    """画像每日快照追加（按日幂等 upsert；原子写）。"""
    import io
    import os
    path = C.PROFILE_HISTORY_CSV
    rows: dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = {r["date"]: r for r in csv.DictReader(f)}
        except Exception as e:                     # noqa: BLE001 读坏不覆写
            # 读失败按空起步再覆写 = 一次瞬时读失败清空整份画像轨迹
            print(f"[紫阳] 画像历史 {os.path.basename(path)} 读取失败"
                  f"（{type(e).__name__}），本次放弃追加、不覆写旧档")
            return
    rows[hist["date"]] = {c: ("" if hist.get(c) is None else hist.get(c))
                          for c in HIST_COLS}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HIST_COLS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(rows[k] for k in sorted(rows))
    fstore.write_text(path, buf.getvalue(), encoding="utf-8-sig")


def build_profile(seat_trades: pd.DataFrame, bars_cache: dict,
                  float_shares: dict | None = None,
                  industry_map: dict | None = None) -> dict:
    """挖掘并落盘 profile.json，返回画像 dict。"""
    ign = mine_ignition_events(seat_trades, bars_cache, float_shares)
    since = dt.date.fromisoformat(C.PROFILE_SINCE)
    st = seat_trades.copy()
    st["交易日期"] = pd.to_datetime(st["交易日期"]).dt.date
    st = st[st["交易日期"] >= since]
    buys = st[st["净买额"] > 0]

    # style：整体行为统计
    daily_net = buys.groupby("交易日期")["净买额"].sum()
    style = {
        "window": {"since": C.PROFILE_SINCE,
                   "days": int(st["交易日期"].nunique()),
                   "stocks": int(st["股票代码"].nunique()),
                   "events": int(len(st))},
        "buy_events": int(len(buys)),
        "sell_events": int(len(st[st["净买额"] < 0])),
        "buy_mainboard_pct": round(100.0 * float(buys["主板"].mean()), 1) if len(buys) else None,
        "buy_day_limit_pct": round(100.0 * float((buys["涨跌幅"] >= C.LIMIT_UP_APPROX).mean()), 1) if len(buys) else None,
        "daily_netbuy_yi": {"mean": round(float(daily_net.mean()) / 1e8, 2),
                            "p50": round(float(daily_net.median()) / 1e8, 2),
                            "max": round(float(daily_net.max()) / 1e8, 2)} if len(daily_net) else {},
        "fwd_after_netbuy": _fwd_stats(pd.read_csv(C.SEAT_TRADES_CSV, dtype={"股票代码": str},
                                                   encoding="utf-8-sig")),
        "active_recent20": int(buys[buys["交易日期"] >= dt.date.today() - dt.timedelta(days=28)]
                               ["交易日期"].nunique()),
    }

    # eve：点火前夜分布（与 features.py 分桶一致，预判层直接消费）
    eve = {}
    if len(ign):
        feats = ign.apply(lambda r: F.eve_features_of_row(r), axis=1, result_type="expand")
        label_attrs = {"lianban": "LB_LABELS", "pct": "PCT_LABELS",
                       "turnover": "TURN_LABELS", "vr": "VR_LABELS",
                       "price": "PRICE_LABELS", "pos20": "POS_LABELS", "ltsz": "LTSZ_LABELS"}
        eve = {
            "n_events": int(len(ign)),
            "n_days": int(ign["ign_date"].nunique()),
            "n_stocks": int(ign["code"].nunique()),
            "dist": {col: _dist(feats[col], getattr(F, label_attrs[col]))
                     for col in feats.columns if col in label_attrs},
            "net_buy_yi_median": round(float(ign["net_buy"].median()) / 1e8, 3),
        }

    # theme：题材偏好（近 60 自然日净买股的行业分布；行业映射来自最新快照）
    theme = {}
    if industry_map:
        recent = buys[buys["交易日期"] >= dt.date.today() - dt.timedelta(days=60)]
        inds = recent["股票代码"].map(industry_map).dropna()
        if len(inds):
            vc = inds.value_counts()
            theme = {"window_days": 60,
                     "n": int(len(inds)),
                     "top": [(k, int(v)) for k, v in vc.head(12).items()]}

    # eve_split：两种打法细分（0板=回调点火 73% vs 1板=高位接力 23%）
    eve_split = {}
    if len(ign):
        for lb, lab in ((0, "0板·回调点火"), (1, "1板·接力点火")):
            g = ign[ign["lianban"] == lb]
            if not len(g):
                continue
            feats = g.apply(lambda r: F.eve_features_of_row(r), axis=1, result_type="expand")
            label_attrs = {"lianban": "LB_LABELS", "pct": "PCT_LABELS",
                           "turnover": "TURN_LABELS", "vr": "VR_LABELS",
                           "price": "PRICE_LABELS", "pos20": "POS_LABELS", "ltsz": "LTSZ_LABELS"}
            eve_split[lab] = {
                "n": int(len(g)),
                "dist": {col: _dist(feats[col], getattr(F, label_attrs[col]))
                         for col in feats.columns if col in label_attrs},
                "net_buy_yi_median": round(float(g["net_buy"].median()) / 1e8, 3),
            }

    # ltsz / power：市值档与点火力度（全量口径，成交额腾讯行 OHLC 近似）
    ltsz, power = {}, {}
    if len(ign):
        ltsz = {"dist": _bucketize(ign["ltsz"], F.LTSZ_EDGES, F.LTSZ_LABELS),
                "note": "前夜流通市值=前夜收盘×当前股本（送转漂移近似）"}
        pw = (ign["net_buy"] / (ign["d_amount_yi"] * 1e8)).dropna()
        if len(pw):
            power = {"dist": _bucketize(pw, POWER_EDGES, POWER_LABELS),
                     "median_pct": round(100.0 * float(pw.median()), 1),
                     "note": "净买额/当日成交额（引导型<10% vs 锁筹型>20%）"}

    # sell / regime / timing：深挖维度（本地可复算，随 daily 全量刷新）
    sell = _sell_profile(st, bars_cache)
    regime = _regime_profile(st)
    timing = _timing_from_archive(ign) if len(ign) else {}

    # trend/history：画像每日迭代细描（7日 vs 30日 双窗对比 + 快照轨迹）
    trend, hist = _evolve(st, ign)

    # deep_refs：静态深挖产物指针（按席位指向各自的 deep/ 目录；重跑 mine_seat
    # 前可能尚未生成，渲染层判空跳过）。紫阳的联动采样摘要与网络情报要点
    # 已固化于 ziyang/RESEARCH.md 3.8 节，此处只留路径指针
    _deep_dir = os.path.join("out", "research", os.path.basename(C.DATA_DIR), "deep")
    deep_refs = {
        "allies": {"source": os.path.join(_deep_dir, "ally_seats.json"),
                   "refreshable": False},
        "web_intel": {"source": os.path.join(_deep_dir, "web_research.md"),
                      "refreshable": False},
    }

    profile = {
        "seat": {"code": C.SEAT_CODE, "name": C.SEAT_NAME, "short": C.SEAT_SHORT},
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "style": style, "eve": eve, "eve_split": eve_split,
        "ltsz": ltsz, "power": power,
        "sell": sell, "regime": regime, "timing": timing,
        "theme": theme, "trend": trend, "deep_refs": deep_refs,
    }
    # 原子写 + .bak 一代副本：画像虽可再生，但重算依赖当日 LHB 披露窗口——
    # 坏一次要等隔日数据，保留上一代自救；快照轨迹落 profile_history.csv
    # （每日 upsert，细描迭代轨迹可回溯）
    fstore.write_with_bak(C.PROFILE_JSON, lambda tp: Path(tp).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"))
    _save_history(hist)
    return profile


def load_profile() -> dict | None:
    """读画像。主件损坏回退上一代 .bak（write_with_bak 每次重算都留一份——
    此前只写不读）；两份都不可读才返回 None（调用方按「无画像」处理，
    下次重算自愈）。"""
    for path in (C.PROFILE_JSON, C.PROFILE_JSON + ".bak"):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            print(f"[紫阳] 画像 {os.path.basename(path)} 读取失败"
                  f"（{type(e).__name__}）" + ("，回退 .bak" if path == C.PROFILE_JSON else ""))
    return None
