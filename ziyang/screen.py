# -*- coding: utf-8 -*-
"""预判层：用点火前夜画像对当日主板候选打分，输出「明日紫阳东路可能点火拉板」清单。

打分口径（v1.2，带概率地板的对数提升度 + Laplace 平滑）：
    score(x) = Σ_f log[ P平滑(f桶(x) | 点火前夜画像) / P平滑(f桶(x) | 当日候选宇宙) ]
               + 0.7×题材共振 + 0.5×复买
排序目标 = 贝叶斯后验 P(它点火|x)：分母（当日宇宙基线）提供区分度，两个概率
地板治分母噪声（P画像桶≥0.5%、P宇宙桶≥1%——今天宇宙只有 1~2 只的桶不再
产生虚高 lift，它几乎不碰的 3板+ 稀有形态可以加分但不可能登顶）。
（v1 无地板→稀有桶登顶；v1.1 纯似然→退化为选最平庸票。均已废弃，
见 score docstring 版本记录。）

数据纪律（2026-09-17 跨零点事故后确立）：
- **特征以 trade_date 锚定的日K为唯一事实源**（涨跌幅/换手/量比/价位/位置/连板），
  实时快照只在「新鲜度校验通过」时用于强势通道取票，结构性字段（名称/流通市值）
  任何时候可用——新浪列表跨零点会滚动为次日盘前数据，直接取其涨跌幅会产出废单。

候选宇宙两条通道（对应它的两类点火模式）：
1. 强势通道：当日涨幅 ≥ CANDIDATE_PCT_MIN（接力/打加速段）；
2. 高低切换通道：近 CANDIDATE_RECENT_LIMIT_DAYS 内曾封板（主工程归档，零网络）。

先标签后计分：本层权重不拟合任何未来数据，全部来自画像分布与两个手工
先验（题材共振 / 席位近10日重复买入），由台账独立裁判。
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

import fstore

from . import config as C
from . import features as F
from .progress import Progress
from .seatdata import (archive_pool_pct, archive_signal_files, archive_union_codes,
                       cache_last_date, feats_from_bars, fetch_bars_batch,
                       get_bars, idx_on, load_bars_cache, save_bars_cache)

_PROBES = ("000001", "600519", "601318")   # 快照新鲜度探测股（大盘流动股）


def recent_sealed_codes(trade_date: dt.date, days: int) -> set[str]:
    """主工程归档里近 N 个自然日内封板过的股票（高低切换通道候选）。"""
    out: set[str] = set()
    for d, f in archive_signal_files():
        if not (trade_date - dt.timedelta(days=days) <= d <= trade_date):
            continue
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            continue
        if "zt_status" in df.columns:
            out |= set(df.loc[df["zt_status"] == "封板", "code"].dropna()
                       .astype(str).str.zfill(6))
    return out


def snapshot_is_fresh(snapshot: pd.DataFrame | None, bars_cache: dict,
                      trade_date: dt.date) -> bool:
    """快照是否仍描述 trade_date：3 只探测股「当日涨幅」与日K一致（容差 0.5 个百分点）。

    v2（2026-09-17）：改比涨幅而非绝对收盘价。快照是未复权价，日K是前复权
    （东财/腾讯）或未复权（新浪第三源）——绝对价差的复权基准不固定，除权股
    会假阴性，且探测结果随「哪家源最后写入缓存」翻转（实测同一晚间宇宙
    477↔610 抖动）。涨幅是各序列内部比值，与复权基准无关。跨零点保护语义
    不变：次日盘前快照涨幅归零/滚动，与昨日日K涨幅不符 → 判不新鲜。
    """
    if snapshot is None:
        return False
    snap = snapshot.set_index("code") if "code" in snapshot.columns else None
    if snap is None:
        return False
    ok = 0
    for code in _PROBES:
        b = get_bars(bars_cache, code, trade_date)
        if b is None:
            continue
        i = idx_on(b, trade_date)
        if i is None or i < 1 or code not in snap.index:
            continue
        sp = snap.loc[code, "pct"]
        if pd.isna(sp):
            continue
        bc, bp = float(b["close"].iloc[i]), float(b["close"].iloc[i - 1])
        if bp <= 0:
            continue
        b_pct = (bc / bp - 1.0) * 100.0
        if abs(float(sp) - b_pct) <= 0.5:
            ok += 1
    return ok >= 2


def build_universe(snapshot: pd.DataFrame | None, bars_cache: dict,
                   trade_date: dt.date, fresh: bool | None = None,
                   stats: dict | None = None) -> list[dict]:
    """当日候选宇宙 + 逐只特征（日K驱动；快照只供名称/流通市值/新鲜时的强势通道）。

    fresh 由调用方算好传入（predict 里行业映射也要用，避免重复探测）。
    stats 为可选出参字典：回填扫描基监控指标（改动四②：并集规模/补拉量/
    补拉成功率——呈现于报告，验收=触发日 100% 在报告呈现）。
    """
    if fresh is None:
        fresh = snapshot_is_fresh(snapshot, bars_cache, trade_date)
    iso = trade_date.isoformat()
    # ---- v2.5 改动四①（条件部署，SCAN_BASE_SUPERSET 门控）：扫描基超集 ----
    # 旧基=缓存+四通道；超集=缓存全部主板 code ∪ 近 N 自然日归档 signals code
    # 并集，对缓存末日期 < trade_date 的陈旧票先补拉再走四通道。∪基按构造
    # ⊇ 理想旧基；补拉直接对冲 9 月式横截面塌缩（新鲜横截面 1145→540 时，
    # 归档 20 自然日并集仍有 ~1054 只）。部署门槛（预注册，config 注释/协议）
    # 未过前维持旧基，仅保留监控指标（告警形态）。
    scan = {"superset": bool(C.SCAN_BASE_SUPERSET), "trade_date": iso,
            "stale_n": sum(1 for c in bars_cache
                           if str(c)[:3] in C.MAIN_BOARD_PREFIXES
                           and (cache_last_date(bars_cache, c) or "") < iso)}
    if C.SCAN_BASE_SUPERSET:
        base = {str(c).zfill(6) for c in bars_cache
                if str(c)[:3] in C.MAIN_BOARD_PREFIXES}
        uni = archive_union_codes(trade_date, C.SCAN_BASE_ARCHIVE_DAYS)
        base |= uni
        stale = {c: trade_date for c in sorted(base)
                 if (cache_last_date(bars_cache, c) or "") < iso}
        if stale:
            fetch_bars_batch(bars_cache, stale, tag="预判·扫描基补拉")
        ok = sum(1 for c in stale if (cache_last_date(bars_cache, c) or "") >= iso)
        scan.update({"union_n": len(uni), "base_n": len(base), "stale_n": len(stale),
                     "refill_ok": ok,
                     "refill_rate": round(100.0 * ok / len(stale), 1) if stale else None})
    # 主板非ST池只过滤一次：结构字段（名称/流通市值）与强势通道共用
    structural: dict[str, dict] = {}
    snap_mb = pd.DataFrame()
    if snapshot is not None:
        snap_mb = snapshot[snapshot["code"].map(
            lambda c: str(c)[:3] in C.MAIN_BOARD_PREFIXES)]
        snap_mb = snap_mb[~snap_mb["name"].map(
            lambda n: "ST" in str(n).upper() or "退" in str(n))]
        for _, r in snap_mb.iterrows():
            structural[str(r["code"])] = {
                "name": r.get("name"),
                "float_mv": float(r["float_mv"]) if pd.notna(r.get("float_mv")) else None,
            }
    sealed = recent_sealed_codes(trade_date, C.CANDIDATE_RECENT_LIMIT_DAYS)

    codes: set[str] = set(sealed)
    # 强势通道一：当日归档池（主工程 15:10 已落盘，跨零点安全；池本身即主板非ST）
    for code, pct in archive_pool_pct(trade_date).items():
        if pct is not None and not (isinstance(pct, float) and math.isnan(pct)) \
                and pct >= C.CANDIDATE_PCT_MIN:
            codes.add(code)
    # 强势通道二：新鲜快照的当日涨幅（快照可用时覆盖归档缺失的交易日）
    if fresh and len(snap_mb):
        s = snap_mb.dropna(subset=["pct"])
        s = s[(s["pct"] >= C.CANDIDATE_PCT_MIN) & (s["volume"].fillna(0) > 0)]
        codes |= set(s["code"].astype(str))

    # 动量通道（v1.4）：5日涨幅 ≥ CANDIDATE_M5_MIN——上涨趋势内的回调票
    # （0板回调点火前夜 m5 中位 +13%，当日涨幅却可能 <3%，此前是召回盲区）。
    # 快照新鲜用 f109；否则扫日K缓存的既有票补算（只覆盖缓存内股票，保守）。
    m5_map: dict[str, float] = {}
    if fresh and len(snap_mb) and "m5" in snap_mb.columns:
        s5 = snap_mb.dropna(subset=["m5"])
        s5 = s5[s5["m5"] >= C.CANDIDATE_M5_MIN]
        for _, r in s5.iterrows():
            m5_map[str(r["code"])] = float(r["m5"])
    else:
        iso = trade_date.isoformat()
        for code, b in bars_cache.items():
            if b is None or not len(b) or str(code)[:3] not in C.MAIN_BOARD_PREFIXES:
                continue
            i = idx_on(b, trade_date)
            if i is None or i < 5:
                continue
            c = b["close"]
            m5 = (float(c.iloc[i]) / float(c.iloc[i - 5]) - 1.0) * 100.0
            if m5 >= C.CANDIDATE_M5_MIN:
                m5_map[str(code).zfill(6)] = m5
    codes |= set(m5_map)

    # 抬头通道（v1.4）：横盘基座 + 近5日单日涨幅 ≥2%（基座里先动了的那天）。
    # 依据：全体点火前夜的 11% 属「m5<5 的横盘抬头」模式（华天科技 09-17
    # 漏检实例：十几天横盘+09-16 抬头3%），加本通道后宇宙召回 87%→98%。
    # 快照无对应字段，仅扫日K缓存（保守）；阈值敏感性 1.5~2.5% 均覆盖 12~15%。
    stir_map: dict[str, float] = {}
    iso = trade_date.isoformat()
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in C.MAIN_BOARD_PREFIXES:
            continue
        i = idx_on(b, trade_date)
        if i is None or i < 6:
            continue
        c = b["close"]
        if (float(c.iloc[i]) / float(c.iloc[i - 5]) - 1.0) * 100.0 >= C.CANDIDATE_M5_MIN:
            continue                      # 已由动量通道覆盖
        pc = b["close"].astype(float)
        stir = max((pc.iloc[j] / pc.iloc[j - 1] - 1.0) * 100.0
                   for j in range(i - 4, i + 1))
        if stir >= C.CANDIDATE_STIR_MIN:
            stir_map[str(code).zfill(6)] = round(stir, 1)
    codes |= set(stir_map)

    # 网络段与计算段分离：先并发把全部候选日K补到 trade_date（含增量拼接/
    # 复权锚校验，见 fetch_bars_batch），再串行走特征与过滤——口径与逐只拉取
    # 完全一致（串行段 get_bars 全程命中缓存，零网络）。
    codes = sorted(codes)
    fetch_bars_batch(bars_cache, {c: trade_date for c in codes}, tag="预判·候选日K")

    rows = []
    pg = Progress("预判·特征", len(codes))
    for k, code in enumerate(codes, 1):
        pg.update(k, extra=code)
        info = structural.get(code, {})
        name = str(info.get("name") or code)
        if "ST" in name.upper() or "退" in name:
            continue
        b = get_bars(bars_cache, code, trade_date)
        if b is None:
            continue
        fmv = info.get("float_mv")
        shares = float_shares_for(fmv, b, trade_date)
        f = feats_from_bars(b, trade_date, back=0, float_shares=shares)
        if f is None:
            continue
        # 股性特征（v2.0）：近60日涨停基因（≥9.5 近似）次数与最大连板
        i_f = idx_on(b, trade_date)
        zt60 = maxlb60 = 0
        if i_f is not None and i_f >= 1:
            cl = b["close"].astype(float)
            pc = (cl / cl.shift(1) - 1.0) * 100.0
            is_zt = pc >= 9.5
            lb, cnt = [], 0
            for ok in is_zt:
                cnt = cnt + 1 if ok else 0
                lb.append(cnt)
            zt60 = int(is_zt.iloc[max(0, i_f - 59): i_f + 1].sum())
            maxlb60 = int(max(lb[max(0, i_f - 59): i_f + 1])) if lb else 0
        strong = f["pct"] is not None and f["pct"] >= C.CANDIDATE_PCT_MIN
        switch = code in sealed and not strong
        m5 = m5_map.get(code)
        if m5 is None:
            i5 = idx_on(b, trade_date)
            if i5 is not None and i5 >= 5:
                c5 = b["close"]
                m5 = (float(c5.iloc[i5]) / float(c5.iloc[i5 - 5]) - 1.0) * 100.0
        momentum = (m5 is not None and m5 >= C.CANDIDATE_M5_MIN
                    and not strong and not switch)
        stir = stir_map.get(code)
        base_stir = (stir is not None and not strong and not switch and not momentum)
        if not (strong or switch or momentum or base_stir):
            continue
        rows.append(dict(code=code, name=name, industry=None,
                         float_mv_yi=round(fmv / 1e8, 1) if fmv else None,
                         ltsz=(fmv / 1e8 if fmv else None),
                         m5=round(m5, 1) if m5 is not None else None,
                         zt60=zt60, maxlb60=maxlb60,
                         via_switch=switch, via_m5=momentum, via_stir=base_stir, **f))
    save_bars_cache(bars_cache)
    pg.done(f"入宇宙 {len(rows)} 只")
    if stats is not None:
        stats.update(scan)
    return rows


def float_shares_for(fmv: float | None, bars: pd.DataFrame,
                     trade_date: dt.date) -> float | None:
    """流通股本 = 流通市值 / 当日收盘（供换手率近似）。"""
    if not fmv or fmv <= 0:
        return None
    i = idx_on(bars, trade_date)
    if i is None:
        return None
    price = float(bars["close"].iloc[i])
    return fmv / price if price > 0 else None


def score(universe: list[dict], profile: dict,
          recent_industries: set, recent_codes: set) -> list[dict]:
    """对候选逐只打分（v1.2 带概率地板的对数提升度），降序返回。

    score = Σ_f log[ P平滑(f桶|点火前夜画像) / P平滑(f桶|当日候选宇宙) ]
            + 题材先验 + 复买先验

    排序目标是贝叶斯后验 P(点火|今晚特征) ∝ P(特征|点火)/P(特征|今天)——
    分母（当日宇宙基线）必须保留，否则纯似然会退化为「选最平庸票」
    （0板桶占画像 72%，几百只 0 板票毫无区分度，v1.1 实测教训）。

    两个地板治 v1 的病根（分母噪声）：
    - P(画像桶) 下限 0.5%：画像没点过的桶不再给 -inf 级负分；
    - P(宇宙桶) 下限 1%：今天宇宙里只出现 1~2 只的桶（如 3板+）分母不可信，
      不再产生 +1.5 级虚高 lift——它 274 次点火里 3板+ 仅 4 次（1.5%），
      该桶 lift 封顶约 log(1.6%/1%)≈+0.5，稀有形态可以加分但不可能登顶。
    v1.2 于台账首轮核验前定版（零核验记录），此后冻结，由台账裁判。
    """
    dists = profile.get("eve", {}).get("dist", {})
    if not dists:
        raise RuntimeError("画像 eve 分布为空：先运行 update 挖掘画像")
    # 基线：当日候选宇宙的各特征桶频率
    base = {}
    for f in C.FEATURES:
        if f == "theme":
            continue
        cnt: dict[str, int] = {}
        tot = 0
        for r in universe:
            b = F.eve_features_of_row(r).get(f)
            if b is not None:
                cnt[b] = cnt.get(b, 0) + 1
                tot += 1
        base[f] = (cnt, tot)

    out = []
    for r in universe:
        feats = F.eve_features_of_row(r)
        total, contrib, misses = 0.0, [], 0
        for f in C.FEATURES:
            if f == "theme":
                continue
            b = feats.get(f)
            d = dists.get(f) or {}
            n_ign = sum(v["n"] for v in d.values())
            k = len(d)                          # 桶数（Laplace 平滑用）
            cnt, tot = base.get(f, ({}, 0))
            if b is None or not d or tot == 0:
                misses += 1
                continue
            p_ign = max((d.get(b, {}).get("n", 0) + 0.5) / (n_ign + 0.5 * k), 0.005)
            p_base = max((cnt.get(b, 0) + 0.5) / (tot + 0.5 * k), 0.01)
            ll = math.log(p_ign / p_base)
            total += ll
            contrib.append((f, b, round(ll, 2)))
        # 手工先验一：题材共振（席位近10个交易日净买过的行业）
        ind = r.get("industry")
        if ind and str(ind) in recent_industries:
            total += 0.7
            contrib.append(("theme", str(ind), 0.7))
        # 手工先验二：席位近10个交易日买过该股本身（连日跟踪/滚动做T风格）
        if r["code"] in recent_codes:
            total += 0.5
            contrib.append(("repeat", "席位近期已买", 0.5))
        out.append(dict(r, score=round(total, 3), contrib=contrib, misses=misses))
    out.sort(key=lambda x: (-x["score"], x["code"]))
    # 展示分：当日宇宙内 0~100 归一（仅可视化；台账与判读一律用原始 score/rank）
    lo = out[-1]["score"] if out else 0.0
    hi = out[0]["score"] if out else 1.0
    span = (hi - lo) or 1.0
    for i, r in enumerate(out):
        r["rank"] = i + 1
        r["score100"] = round(100.0 * (r["score"] - lo) / span, 0)
    return out


def money_prior(r: dict, repeat_set: set) -> int:
    """改动三 money_prior 三档（0~3，全部 D-晚可知、无缺口列）：
    lianban≥2 连板档 / repeat01 记忆档（紫阳∪知春路 [D-20,D) 半开窗，与
    训练侧同口径）/ 股性档（近60日涨停基因：zt60≥1 或 maxlb60≥2）。
    先验依据（干净窗主板前缀+LIVE+V≤08-31）：2板+ 48.0%（n=1018）vs 首板
    44.3%（n=4329）——见 config 注释与预注册协议。"""
    return (int((r.get("lianban") or 0) >= 2) + int(r["code"] in repeat_set)
            + int((r.get("zt60") or 0) >= 1 or (r.get("maxlb60") or 0) >= 2))


def predict(trade_date: dt.date | None = None, top_n: int | None = None,
            snapshot: pd.DataFrame | None = None) -> dict:
    """编排：画像 + 日K + 快照 → 候选宇宙 → 打分 → top N。不写台账（见 ledger.register）。"""
    from .profile import load_profile
    from .seatdata import (fetch_snapshot_any, industry_map, last_trade_date,
                           load_seat_trades)

    top_n = top_n or C.TOP_N
    # 交易日锚定：最近已收盘交易日（跨零点/盘前运行时快照仍是上一交易日数据）
    trade_date = trade_date or last_trade_date()
    profile = load_profile()
    if profile is None:
        raise RuntimeError("profile.json 不存在：先运行 python -m ziyang update")

    if snapshot is None:
        snapshot = fetch_snapshot_any()      # 结构字段跨零点仍可用；涨跌幅经新鲜度校验
    bars_cache = load_bars_cache()
    fresh = snapshot_is_fresh(snapshot, bars_cache, trade_date)
    scan: dict = {}
    universe = build_universe(snapshot, bars_cache, trade_date, fresh, stats=scan)

    # 行业映射：归档（池内）∪ 快照（新鲜时全市场；新浪兜底无行业列，自然退化为纯归档）
    imap = industry_map(snapshot if fresh else None)
    for r in universe:
        if imap.get(r["code"]):
            r["industry"] = imap[r["code"]]

    seat = load_seat_trades().copy()
    seat["交易日期"] = pd.to_datetime(seat["交易日期"]).dt.date
    since = trade_date - dt.timedelta(days=C.THEME_LOOKBACK_DAYS * 2)
    recent = seat[(seat["交易日期"] >= since) & (seat["净买额"] > 0)]
    recent_inds = {imap[c] for c in recent["股票代码"] if c in imap}
    recent_codes = set(recent["股票代码"])   # 影子 v1.4 语义冻结：紫阳、含当日

    ranked = score(universe, profile, recent_inds, recent_codes)

    # ---- v2.0 主分：日内分位 logistic（deep2 四路深讨定稿）----
    # v1.4 log-lift 降为影子（shadow_score/shadow_rank 落排名快照，前向 A/B）。
    from . import scorer
    from .seatdata import float_shares_map
    shares_map = float_shares_map(snapshot)
    name_map = ({str(r["code"]).zfill(6): str(r["name"])
                 for _, r in snapshot.iterrows()}
                if snapshot is not None else {})
    # v2.1 交互特征：当日涨停触及家数 × 个股涨幅分位（唯一过 CI 的特征增量）
    scorer.attach_it_ztc_pct(ranked, scorer.market_zt_touch_count(bars_cache, trade_date))
    # v2.4 题材特征：themed 特征表查表注入（无涨停史=0 中性；表缺失时全 0 可回滚）
    from .themed import attach_theme_features
    attach_theme_features(ranked, trade_date)
    # v2.6 分时轨迹特征（deep7 Round1）：tp_touch/tp_am，≤D 最近池上轨迹
    from .themed import attach_tp_features
    attach_tp_features(ranked, trade_date)
    # v3.1 记忆特征行键（hazard_zy/squad20/sold_young/uw_lr）——score_rows 在
    # 校准 logit 上以冻结 log-LR 合成（废除手工后处理 ±0.6/-0.65，全换拟合值）。
    # closes=买入日收盘锚（浮亏维度：两个地板的深套票 uw_lr=-1.31/-2.34 压制）
    _cl = {}
    for _c, _b in bars_cache.items():
        if _b is not None and len(_b):
            _cl[str(_c).zfill(6)] = dict(zip(_b["date"], _b["close"]))
    scorer.attach_memory_features(ranked, trade_date, closes=_cl)
    # v3.0 own-pick 后处理惩罚（deep8 实证席 #1）——标记先算，
    # 惩罚在模型分落位后统一应用（防止被 L441 的 score_v2 赋值覆写）。
    # 窗口=半开 [D-5, D)（只含此前交易日的选择）——v3.0.3 修复：此前含当日，
    # 同日重跑会惩罚上一跑的 topN 造成甄选两组票间振荡、登记结果取决于
    # 重跑次数（2026-09-28 实测复现）。与 repeat01/卖出记忆的半开窗口同构。
    # 语义=槽位机会成本规则（deep8：65% 槽位被重复旧宠吃掉、累计 -90%），
    # 不是概率参数；幅度待台账≥8 独立日后按实测重拟。
    pick_hist = scorer.load_pick_history()
    own_lo = (trade_date - dt.timedelta(days=5)).isoformat()
    own_hi = trade_date.isoformat()
    for r in ranked:
        hist = pick_hist.get(r["code"], [])
        recent_picks = [d for d in hist if own_lo <= d < own_hi]
        r["own_pick_penalty"] = 0.5 * len(recent_picks) if recent_picks else 0.0
        r["own_pick_count_5d"] = len(recent_picks)
    # repeat01 记忆集（v2.3 路线图#1 修复点③）：紫阳∪知春路、[D-20,D) 半开窗
    # ——与训练语义完全一致（消除 15:10 首跑/晚间重跑的特征漂移与运行时刻依赖）
    repeat_set_v2 = scorer.union_repeat_set(trade_date)
    meta = scorer.ensure_model(bars_cache, name_map, shares_map)
    v2_scores, v2_contribs = scorer.score_rows(ranked, repeat_set_v2)
    for r, s, cc in zip(ranked, v2_scores, v2_contribs):
        r["shadow_score"], r["shadow_rank"] = r["score"], r["rank"]
        # v3.1：s 已是校准合成分（logit(iso(P)) + 冻结 log-LR），own_pick 是
        # 唯一保留的后处理项（槽位规则，见上注）
        r["score_v2"] = round(float(s) - r["own_pick_penalty"], 3)
        if r["own_pick_penalty"] > 0:
            cc = cc + [("own_pick", -round(r["own_pick_penalty"], 2))]
        r["contrib"] = cc                    # 特征贡献+记忆合成项（score_rows 并入）
    ranked.sort(key=lambda x: (-x["score_v2"], x["code"]))
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["score"] = r.pop("score_v2")       # score 即 v2（台账/下游统一）
    if ranked:
        lo2 = ranked[-1]["score"]
        hi2 = ranked[0]["score"]
        span2 = (hi2 - lo2) or 1.0
        for r in ranked:
            r["score100"] = round(100.0 * (r["score"] - lo2) / span2, 0)

    # ---- v2.5 改动三：λ 联合分整形（候选口径，增列并行）----
    # score_joint = rank(score_v2) + λ·rank(money_prior)。主 rank 语义不变——
    # 切换为生产主口径由判定窗两臂门控（映射(1)/(3)/(4)）决定；joint_rank 与
    # money_prior 以增列落台账与排名快照。λ 冻结值=config.JOINT_LAMBDA（选定
    # 表=out/research/ziyang/deep6/wf_profit.json，≤08-31 walk-forward 众数）。
    if ranked:
        sv = pd.Series([r["score"] for r in ranked], dtype=float)
        mv = pd.Series([money_prior(r, repeat_set_v2) for r in ranked], dtype=float)
        joint = sv.rank(pct=True) + C.JOINT_LAMBDA * mv.rank(pct=True)
        order = sorted(range(len(ranked)),
                       key=lambda k: (-float(joint.iloc[k]), ranked[k]["code"]))
        for j, k in enumerate(order):
            ranked[k]["joint_rank"] = j + 1
            ranked[k]["score_joint"] = round(float(joint.iloc[k]), 4)
        for r, m in zip(ranked, mv):
            r["money_prior"] = int(m)

    # ---- v2.5 改动五：出勤三层标注（deep5/attendance.md 预注册；只标注不改
    # 排序，不做硬门控）。low_att 照常出票+标注+建议仓位减半；反向监控见报告。
    stratum, att_info = "normal", {}
    try:
        from .ledger import attendance_stratum
        stratum, att_info = attendance_stratum(trade_date, seat_trades=seat,
                                               bars_cache=bars_cache)
    except Exception as _e:                    # noqa: BLE001 标注失败不阻断预判
        att_info = {"error": type(_e).__name__}
    for r in ranked:
        r["stratum"] = stratum

    # ---- v2.5 改动四④：月度重训首日新旧 top3 diff（度量/呈现层）----
    retrain_diff = scorer.retrain_top3_diff(trade_date, ranked, repeat_set_v2)

    # ---- v3.0 在线续训（deep8 #2：每日小步吸收新标签）----
    try:
        ft = scorer.online_finetune(bars_cache, name_map, shares_map)
        if ft.get("ok") and not ft.get("skipped"):
            meta["finetuned_today"] = True
            if ft.get("n_label_pending"):
                print(f"[紫阳] 续训：{ft['n_label_pending']} 个训练日的标签日龙虎榜"
                      "尚未发布，已跳过（不记负例）")
        elif not ft.get("ok"):
            print(f"[紫阳] 续训跳过（{ft.get('reason')}），沿用月度模型")
    except Exception as e:                                # noqa: BLE001 续训失败不阻断
        # 不静默：续训坏一天 = 模型少吃一天标签，且此前这里连报错都没有
        print(f"[紫阳] 续训失败（{type(e).__name__}: {e}），沿用月度模型")

    # ---- v3.0.2 相对低分弃权（2026-09-28 现场"选出这玩意"后设立）----
    # v3.0 在线续训后分数尺度漂移（旧 3~5 → 现 ~1.2），绝对阈值失效；改为
    # 滚动自校准：当日 top1 < ABSTAIN_REL × 同模型版本近 20 日 top1 中位 →
    # 弃权。历史 < ABSTAIN_MIN_HIST 日不开门（防冷启动误杀）。弃权日
    # predictions 置空但全量排名快照照落（记录"今天选择了不出票"）。
    # sr-v1 弃权门不启用——代理分数训练导致校准失真（day_prob 0.99 vs
    # 票级 0.001），只用它做 sr_day_prob 标注；重训后可并入双门。
    abstain = False
    abstain_reason = ""
    try:
        hist = fstore.read_json(C.ABSTAIN_HIST_FILE)
        if not isinstance(hist, dict):      # 缺失/损坏/非对象 = 无历史
            hist = {}
        mver = str(meta.get("version", "unknown"))
        day_scores = hist.get(mver, {})
        if not isinstance(day_scores, dict):
            day_scores = {}
        # 「近 20 日」= 按日期排序后的最后 20 个。此前按分数排序取最大的 20 个：
        # 历史一超过 20 日，门槛就变成「历史高分位的中位数」，弃权系统性偏多
        prior = [(d, float(v)) for d, v in sorted(day_scores.items())
                 if d < trade_date.isoformat()]
        if len(prior) >= C.ABSTAIN_MIN_HIST and ranked:
            med = float(np.median([v for _, v in prior[-20:]]))
            if ranked[0]["score"] < C.ABSTAIN_REL * med:
                abstain = True
                abstain_reason = (f"low_score(top1={ranked[0]['score']:.2f}"
                                  f"<{C.ABSTAIN_REL}×med{med:.2f})")
        # 登记当日 top1（弃权日也登记——弱日本身就是分布信息）
        if ranked:
            hist.setdefault(mver, {})[trade_date.isoformat()] = \
                round(float(ranked[0]["score"]), 4)
        if abstain:
            # _abstain 与版本键平级：selection_line 据此在主报告显示弃权行
            # （否则空台账会让甄选行整行消失，用户无从区分"弃权"与"没跑"）
            hist.setdefault("_abstain", {})[trade_date.isoformat()] = abstain_reason
        elif "_abstain" in hist:
            hist["_abstain"].pop(trade_date.isoformat(), None)  # 复跑翻案清标记
        if ranked or abstain:
            fstore.write_json(C.ABSTAIN_HIST_FILE, hist, indent=1)
    except Exception as e:                           # noqa: BLE001 弃权门失败照常出票
        # 不静默：弃权门坏一天=丢一个独立样本日（2026-11-20 复盘发现 np 未导入
        # 曾让整段静默死亡、top1 历史冻结——失败必须留痕）
        print(f"[紫阳] 弃权门失败（{type(e).__name__}: {e}），照常出票")
    try:
        from .poolmodel import predict_success_rate
        ztc = scorer.market_zt_touch_count(bars_cache, trade_date)
        seat5 = len(recent)
        sv = [r["score"] for r in ranked[:3]] if len(ranked) >= 3 else [0]
        day_feats = [len(universe), ztc, seat5, sv[0],
                     float(np.std(sv)) if len(sv) > 1 else 0]
        tick_feats = [[i, r["score"], r["score"] - sv[0],
                       1.0 if r.get("via_switch") else 0,
                       1.0 if r.get("via_m5") else 0,
                       float(r.get("theme_max_b") or 0)]
                      for i, r in enumerate(ranked[:3], 1)]
        sr = predict_success_rate(day_feats, tick_feats)
        day_prob = sr.get("day_prob", 1.0)
        for r in ranked:
            r["sr_day_prob"] = day_prob
            r["sr_ticket_prob"] = None
        if "ticket_probs" in sr:
            for r, tp in zip(ranked[:3], sr["ticket_probs"]):
                r["sr_ticket_prob"] = tp
    except Exception:                                # noqa: BLE001 标注失败照常出票
        pass

    return {"trade_date": trade_date, "universe_n": len(universe),
            "model": meta,
            "predictions": [] if abstain else ranked[:top_n],
            "all": ranked, "abstain": abstain, "abstain_reason": abstain_reason,
            "scanbase": scan, "stratum": stratum, "attendance": att_info,
            "retrain_diff": retrain_diff}
