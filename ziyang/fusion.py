# -*- coding: utf-8 -*-
"""双席位→军团融合层 v2：同日共击原生建模（2026-09-20 用户定向大改）。

v1（已废弃）＝紫阳画像分 + 开源画像分的独立似然加总——接力地图证实两席位
是「同日共击型」资金（游资接力链不存在），独立交集建模方向错误。
v2 直接以**共击事件**为挖掘对象：

1. 军团（SQUAD）＝紫阳东路的同日共击盟友（relay_map 实测 ≥4 次同日的游资型
   营业部；高盛/总部类通道型资金剔除——被动承接无预测信息）。
2. 共击事件 ＝ 紫阳点火（主板净买>0 且涨停）∧ ≥1 军团席位同日净买。
3. 共击前夜画像 ＝ 共击事件 D-1 的七维特征分布（与主甄选同分桶同平滑）。
4. 共振甄选 ＝ 当日候选对「共击前夜画像」的对数提升度 + 三个手工先验
   （军团动量/紫阳复买/共击题材）。**建议层，不写台账、不动主甄选冻结口径**。

CLI：python -m ziyang.fusion
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import fstore
import pandas as pd

from . import config as C
from . import features as F

FUSION_DIR = os.path.join(C.ROOT, "out", "research", "fusion")
FUSION_JSON = os.path.join(FUSION_DIR, "fusion_profile.json")
FUSION_REPORT = os.path.join(FUSION_DIR, "report_latest.txt")
PARTNER_DIR = os.path.join(FUSION_DIR, "partner_trades")

_MB = C.MAIN_BOARD_PREFIXES   # 主板口径（包内单一源，此前 4 处手抄）
_SINCE = dt.date(2026, 1, 1)

# 军团兜底注册表：legion.json（自动进化引擎）缺失时回退此名单。
# 正役成员以 active_squad() 为准（2026-11-22 起动态化——调研实证盟友关系
# 非稳态：太华路×8 全在 8 月、近三月共击 0，静态名单必然过期；首版调研
# 证据=out/research/fusion/relay_map_fresh.json + relay_fresh.py 可复算）。
# 开源西大街数据在 out/research/kaiyuan/，其余四席在 partner_trades/。
DEFAULT_SQUAD = {
    "开源西大街": {"code": "10046745", "csv": None},           # kaiyuan/seat_trades.csv
    "国新北京": {"code": "10485107", "csv": "10485107.csv"},
    "知春路": {"code": "10025390", "csv": "10025390.csv"},
    "江苏路": {"code": "10025080", "csv": "10025080.csv"},
    "开源太华路": {"code": "10477800", "csv": "10477800.csv"},
}


def active_squad() -> dict[str, dict]:
    """现役军团（legion.json 动态成员源；无档案回退 DEFAULT_SQUAD）。"""
    from .legion import active_squad as _as
    return _as()


def squad_csv_paths() -> list[str]:
    """军团成员明细 CSV 绝对路径（scorer squad20 特征消费；缺失档跳过）。"""
    paths = []
    for s in active_squad().values():
        p = (os.path.join(C.ROOT, "out", "research", "kaiyuan", "seat_trades.csv")
             if s["csv"] is None else os.path.join(PARTNER_DIR, s["csv"]))
        if os.path.exists(p):
            paths.append(p)
    return paths


# ---------------------------------------------------------------- 数据
def _dedupe_net_buys(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"股票代码": str}, encoding="utf-8-sig")
    df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
    g = (df.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(买=("买入金额", "max"), 卖=("卖出金额", "max")))
    g["净"] = g["买"] - g["卖"].fillna(0)
    return g[(g["净"] > 0) & g["股票代码"].str[:3].isin(_MB)
             & (g["交易日期"] >= _SINCE)]


def ensure_partner_trades() -> None:
    """军团席位明细落盘（幂等：判据=数据末日期≥最近已收盘交易日——路线图#2，
    旧判据「文件存在即跳过」使 +0.6 共击先验的 20 日记忆窗常态盲窗 3~8 天）。

    拉取后判空不落盘（与 update_seat_trades 同纪律），失败打印不静默。
    """
    os.makedirs(PARTNER_DIR, exist_ok=True)
    from .seatdata import last_trade_date
    lt = last_trade_date().isoformat()
    for name, s in active_squad().items():
        if s["csv"] is None:
            continue
        path = os.path.join(PARTNER_DIR, s["csv"])
        if os.path.exists(path):
            try:
                have = pd.read_csv(path, usecols=["交易日期"], encoding="utf-8-sig")
                if str(pd.to_datetime(have["交易日期"]).max())[:10] >= lt:
                    continue
            except Exception:                      # noqa: BLE001 损坏文件直接重拉
                pass
        print(f"[军团] 拉取 {name}（{s['code']}）全量明细…")
        try:
            import akshare as ak
            import io as _io
            from contextlib import redirect_stderr
            buf = _io.StringIO()
            with redirect_stderr(buf):
                df = ak.stock_lhb_yyb_detail_em(symbol=s["code"])
            if df is None or not len(df):
                print(f"[军团] {name} 拉取为空，保留旧档")
                continue
            fstore.write_dataframe_csv(path, df)
        except Exception as e:                      # noqa: BLE001
            print(f"[军团] {name} 刷新失败（{type(e).__name__}），沿用旧档")


def squad_keys() -> dict[tuple, list[str]]:
    """{(日, 股): [同日净买的军团席位名]}。"""
    out: dict[tuple, list[str]] = {}
    for name, s in active_squad().items():
        csv = (os.path.join(C.ROOT, "out", "research", "kaiyuan", "seat_trades.csv")
               if s["csv"] is None else os.path.join(PARTNER_DIR, s["csv"]))
        if not os.path.exists(csv):
            continue                       # 动态成员：档未就位即跳过（不崩）
        ev = _dedupe_net_buys(csv)
        for r in ev.itertuples(index=False):
            out.setdefault((r.交易日期, r.股票代码), []).append(name)
    return out


def _zy_ignitions() -> pd.DataFrame:
    """紫阳点火事件（与画像同口径）。"""
    df = pd.read_csv(os.path.join(C.ROOT, "out", "research", "ziyang",
                                  "seat_trades.csv"),
                     dtype={"股票代码": str}, encoding="utf-8-sig")
    df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
    g = (df.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(名称=("股票名称", "last"), 幅=("涨跌幅", "max"),
              买=("买入金额", "max"), 卖=("卖出金额", "max"),
              d1=("1日后涨跌幅", "max"), d3=("3日后涨跌幅", "max"),
              d5=("5日后涨跌幅", "max")))
    g["净"] = g["买"] - g["卖"].fillna(0)
    return g[(g["净"] > 0) & (g["幅"] >= C.LIMIT_UP_APPROX)
             & g["股票代码"].str[:3].isin(_MB) & (g["交易日期"] >= _SINCE)]


# ---------------------------------------------------------------- 共击画像
def _fwd(v) -> dict:
    v = pd.to_numeric(pd.Series(list(v)), errors="coerce").dropna()
    return {"n": int(len(v)), "median": round(float(v.median()), 2),
            "win": round(100.0 * float((v > 0).mean()), 0)} if len(v) else {}


def co_attack_stats() -> dict:
    """共击事件统计 + 后市对照 + 前夜画像（全部落盘 fusion_profile.json）。"""
    ign = _zy_ignitions()
    sq = squad_keys()
    rows = []
    for r in ign.itertuples(index=False):
        partners = sq.get((r.交易日期, r.股票代码), [])
        rows.append({"date": r.交易日期, "code": r.股票代码, "name": r.名称,
                     "net_zy_yi": r.净 / 1e8, "pct": r.幅,
                     "partners": partners, "n_partners": len(partners),
                     "d1": r.d1, "d3": r.d3, "d5": r.d5,
                     "co": int(bool(partners))})
    ev = pd.DataFrame(rows)
    co = ev[ev["co"] == 1]
    solo = ev[ev["co"] == 0]

    # 前夜画像（共击事件 D-1 七维；bars 复用共享缓存，紫阳点火股全覆盖）
    from .seatdata import feats_from_bars, load_bars_cache
    cache = load_bars_cache()
    from .seatdata import fetch_snapshot_any, float_shares_map
    try:
        shares = float_shares_map(fetch_snapshot_any())
    except Exception:
        shares = {}
    eve_rows = []
    for r in co.itertuples(index=False):
        b = cache.get(r.code)
        if b is None or not len(b):
            continue
        f = feats_from_bars(b, r.date, back=1, float_shares=shares.get(r.code))
        if f is None:
            continue
        sh = shares.get(r.code)
        f["ltsz"] = sh * f["price"] / 1e8 if sh else None
        f["code"], f["date"], f["n_partners"] = r.code, r.date, r.n_partners
        eve_rows.append(f)
    eve_df = pd.DataFrame(eve_rows)
    label_attrs = {"lianban": "LB_LABELS", "pct": "PCT_LABELS",
                   "turnover": "TURN_LABELS", "vr": "VR_LABELS",
                   "price": "PRICE_LABELS", "pos20": "POS_LABELS",
                   "ltsz": "LTSZ_LABELS"}
    dists = {}
    if len(eve_df):
        feats = eve_df.apply(lambda r: F.eve_features_of_row(r), axis=1,
                             result_type="expand")
        for col, attr in label_attrs.items():
            s = feats[col].dropna()
            if len(s):
                vc = s.value_counts()
                dists[col] = {lab: {"n": int(vc.get(lab, 0)),
                                    "pct": round(100.0 * vc.get(lab, 0) / len(s), 1)}
                              for lab in getattr(F, attr) if vc.get(lab, 0)}

    partner_counts = {}
    for plist in co["partners"]:
        for p in plist:
            partner_counts[p] = partner_counts.get(p, 0) + 1

    # 伙伴质量（2026-11-22 用户定向·军团作用优化）：各席位共击时紫阳点火股的
    # 后市分布——军团成员不只看「来没来」，更看「来了之后肉厚不厚」；
    # 按共击次数降序，是军团 5 席代表性排序的量化依据。
    partner_quality = {}
    for r in co.itertuples(index=False):
        for p in r.partners:
            q = partner_quality.setdefault(
                p, {"co_n": 0, "d1": [], "d5": []})
            q["co_n"] += 1
            if r.d1 == r.d1:
                q["d1"].append(float(r.d1))
            if r.d5 == r.d5:
                q["d5"].append(float(r.d5))
    for q in partner_quality.values():
        d1s, d5s = q.pop("d1"), q.pop("d5")
        q["d1_median"] = round(sum(d1s) / len(d1s), 2) if d1s else None
        q["d5_median"] = round(sum(d5s) / len(d5s), 2) if d5s else None

    out = {
        "version": 2, "direction": "同日共击（接力链证伪后的定向重构）",
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "squad": {k: v["code"] for k, v in active_squad().items()},
        "n_ignition": int(len(ev)), "n_co": int(len(co)),
        "co_pct": round(100.0 * len(co) / len(ev), 1) if len(ev) else None,
        "co_days": int(co["date"].nunique()) if len(co) else 0,
        "partner_counts": dict(sorted(partner_counts.items(), key=lambda x: -x[1])),
        "partner_quality": dict(sorted(
            partner_quality.items(), key=lambda kv: -kv[1]["co_n"])),
        "co_fwd": {k: _fwd(co[k]) for k in ("d1", "d3", "d5")},
        "solo_fwd": {k: _fwd(solo[k]) for k in ("d1", "d3", "d5")},
        "eve": {"n": int(len(eve_df)), "dist": dists,
                "net_zy_median_yi": round(float(co["net_zy_yi"].median()), 2)
                if len(co) else None},
        "events": [{"date": str(r.date), "code": r.code, "name": r.name,
                    "partners": r.partners} for r in co.itertuples(index=False)],
        "note": "共击=紫阳点火∧军团同日净买；军团=relay_map 实测同日≥4次的游资型席位",
    }
    fstore.write_json(FUSION_JSON, out, indent=2)
    return out


# ---------------------------------------------------------------- 共振甄选 v2
def _loglift(feat: str, bucket, dists: dict, base: dict) -> float | None:
    d = dists.get(feat) or {}
    n_ign = sum(v["n"] for v in d.values())
    k = len(d)
    cnt, tot = base.get(feat, ({}, 0))
    if bucket is None or not d or tot == 0:
        return None
    p_ign = max((d.get(bucket, {}).get("n", 0) + 0.5) / (n_ign + 0.5 * k), 0.005)
    p_base = max((cnt.get(bucket, 0) + 0.5) / (tot + 0.5 * k), 0.01)
    return math.log(p_ign / p_base)


def resonance(trade_date: dt.date | None = None) -> dict:
    """共击共振甄选：对当日候选宇宙按「共击前夜画像」打对数提升度 + 三先验。"""
    from .screen import predict
    from .seatdata import last_trade_date, load_seat_trades

    trade_date = trade_date or last_trade_date()
    with open(FUSION_JSON, encoding="utf-8") as f:
        fj = json.load(f)
    dists = (fj.get("eve") or {}).get("dist") or {}
    if not dists:
        raise RuntimeError("共击画像为空：先跑 co_attack_stats()")

    base_pref = predict(trade_date)
    universe = base_pref["all"]

    # 宇宙基线（同 v1.3 口径）
    base = {}
    for f in C.FEATURES:
        cnt, tot = {}, 0
        for r in universe:
            b = F.eve_features_of_row(r).get(f)
            if b is not None:
                cnt[b] = cnt.get(b, 0) + 1
                tot += 1
        base[f] = (cnt, tot)

    # 先验素材
    co_events = fj.get("events") or []
    co_dates = sorted({dt.date.fromisoformat(e["date"]) for e in co_events})
    dates_sorted = sorted(co_dates)
    momentum = bool(dates_sorted
                    and trade_date - dates_sorted[-1] <= dt.timedelta(days=5))
    seat = load_seat_trades()
    seat["交易日期"] = pd.to_datetime(seat["交易日期"]).dt.date
    recent_zy = seat[(seat["交易日期"] >= trade_date - dt.timedelta(days=20))
                     & (seat["净买额"] > 0)]
    zy_codes = set(recent_zy["股票代码"])
    from .seatdata import industry_map
    imap = industry_map()
    co_themes = {imap[e["code"]] for e in co_events
                 if dt.date.fromisoformat(e["date"])
                 >= trade_date - dt.timedelta(days=20) and imap.get(e["code"])}

    out = []
    for r in universe:
        feats = F.eve_features_of_row(r)
        total, contrib = 0.0, []
        for f in C.FEATURES:
            ll = _loglift(f, feats.get(f), dists, base)
            if ll is not None:
                total += ll
                contrib.append((f, feats.get(f), round(ll, 2)))
        if momentum:
            # 军团动量按成员质量加权（legion 证据分派生；样本不足回 1.0——
            # 先标签后计分）：近期共击的伙伴质量均值决定加成幅度
            from .legion import seat_weights
            wmap = seat_weights()
            recent_partners = [p for e in co_events
                               if dt.date.fromisoformat(e["date"])
                               >= trade_date - dt.timedelta(days=20)
                               for p in e["partners"]]
            w = (sum(wmap.get(p, 1.0) for p in recent_partners)
                 / len(recent_partners)) if recent_partners else 1.0
            bonus = round(0.5 * max(0.5, min(1.5, w)), 2)
            total += bonus
            contrib.append(("军团动量",
                            f"近5日有共击·伙伴质量{w:.2f}", bonus))
        if r["code"] in zy_codes:
            total += 0.5
            contrib.append(("紫阳复买", "近20日已买", 0.5))
        if r.get("industry") and r["industry"] in co_themes:
            total += 0.7
            contrib.append(("共击题材", r["industry"], 0.7))
        out.append(dict(r, score_co=round(total, 3), contrib_co=contrib))
    out.sort(key=lambda x: (-x["score_co"], x["code"]))
    for i, r in enumerate(out, 1):
        r["rank_co"] = i
    return {"trade_date": base_pref["trade_date"], "universe_n": base_pref["universe_n"],
            "zy_top3": base_pref["predictions"], "co_top3": out[:C.TOP_N],
            "momentum": momentum, "n_co_events": fj.get("n_co"),
            "all": out}


# ---------------------------------------------------------------- 渲染
def render(fj: dict | None = None, res: dict | None = None) -> str:
    from .report import _w
    L = ["─" * 78, "紫阳军团 · 同日共击融合（v2）", "─" * 78]
    if fj:
        L += ["", "▎共击结构（v2 画像·fusion_profile.json）", ""]
        L += _w("  紫阳点火 %d 事件中，军团同日共击 %d 次（%s%%，%d 个交易日）；"
                "军团=%s（高盛/总部类通道资金已剔除）"
                % (fj["n_ignition"], fj["n_co"], fj["co_pct"], fj["co_days"],
                   "、".join(fj["partner_counts"])))
        L += _w("  伙伴频次：" + "、".join(f"{k}×{v}" for k, v in fj["partner_counts"].items()))
        pq = fj.get("partner_quality") or {}
        if pq:
            L += _w("  伙伴质量（共击时点火股后市）："
                    + "、".join(f"{k}×{v['co_n']}(d1 {v.get('d1_median')}%/d5 "
                                f"{v.get('d5_median')}%)"
                                for k, v in list(pq.items())[:6]))
        cf, sf = fj["co_fwd"], fj["solo_fwd"]
        L += _w("  共击 vs 单买后市：d1 中位 %s%% vs %s%%｜d5 中位 %s%% vs %s%%"
                % (cf["d1"].get("median"), sf["d1"].get("median"),
                   cf["d5"].get("median"), sf["d5"].get("median")))
        d = (fj.get("eve") or {}).get("dist") or {}
        if d:
            L += _w("  共击前夜画像（n=%d）：%s ｜ %s"
                    % ((fj.get("eve") or {}).get("n"),
                       _dd(d.get("lianban")), _dd(d.get("ltsz"))))
    if res:
        L += ["", "▎共击共振甄选（建议层·不进台账）", ""]
        for r in res["co_top3"]:
            top_c = sorted(r.get("contrib_co", []), key=lambda x: -abs(x[2]))[:3]
            det = "、".join(f"{a}={b}({v:+})" for a, b, v in top_c)
            L += _w("  #%d %s %s  共击分%.2f  %s板·%+.1f%%·%.0f亿"
                    % (r["rank_co"], r["code"], r["name"], r["score_co"],
                       r.get("lianban"), r["pct"], r.get("ltsz") or 0))
            L += _w("      " + det)
        zy_set = {x["code"] for x in res["zy_top3"]}
        overlap = [r for r in res["co_top3"] if r["code"] in zy_set]
        L += _w("  与主甄选重合 %d/3｜军团动量=%s（近5日共击事件）｜共击画像 n=%s"
                % (len(overlap), "在线" if res["momentum"] else "沉寂",
                   res["n_co_events"]))
    L += ["", "─" * 78, "建议层：不写台账、不改主甄选冻结口径；不构成投资建议。", "─" * 78]
    return "\n".join(L)


def _dd(d: dict | None) -> str:
    if not d:
        return "-"
    return " ".join(f"{k}:{v['pct']}%" for k, v in d.items() if v.get("n"))


def main() -> int:
    ensure_partner_trades()
    fj = co_attack_stats()
    res = resonance()
    txt = render(fj, res)
    fstore.write_text(FUSION_REPORT, txt)
    print(txt)
    print(f"\n→ {FUSION_JSON}\n→ {FUSION_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
