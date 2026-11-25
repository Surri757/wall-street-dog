# -*- coding: utf-8 -*-
"""题材特征（路线图#11，deep5 题材席 txb30 方案的生产化）。

两列特征（进 v2 特征集，日内秩变换对齐其余特征）：
- theme_max_b：个股近 90 日涨停标签的**最大收缩 lift**（lift=n/(n+k) 向1收缩，
  k=5；无涨停史=0 中性）
- theme30_b：近 30 日标签 lift 均值（同上口径）

标签 lift 用**训练窗内**紫阳∪知春路点火的标签频次 vs 涨停池基线计算
（生产=union 标签，deep5 验证时是紫阳-only——故上线前须 union 重跑确认）。
特征表落盘 ths_feat_daily.csv 供训练/推理共用，幂等重算。
"""
from __future__ import annotations

import datetime as dt
import os
from collections import Counter

import fstore
import pandas as pd

from . import config as C

EVENTS_CSV = os.path.join(C.DATA_DIR, "ths_events.csv")
FEAT_CSV = os.path.join(C.DATA_DIR, "ths_feat_daily.csv")
_MB = C.MAIN_BOARD_PREFIXES   # 主板口径（包内单一源，此前 4 处手抄）
_K = 5                    # 贝叶斯收缩强度（deep5 题材席标定）
_WINDOWS = (90, 30)


def _build_ths_events() -> pd.DataFrame:
    """合并 deep4 历史 + 生产 ths_pool 目录 → 去重事件表（幂等）。"""
    import glob
    import re
    frames = []
    for g in (os.path.join(C.ROOT, "out", "research", "ziyang", "deep4", "ths_pool_*.csv"),
              os.path.join(C.ROOT, "out", "research", "ziyang", "ths_pool", "*.csv")):
        for f in sorted(glob.glob(g)):
            try:
                df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
            except Exception:
                continue
            if "date" not in df.columns:
                m = re.search(r"(\d{8})", os.path.basename(f))
                if not m:
                    continue
                df["date"] = pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
            have = [c for c in ("date", "code", "reason_type") if c in df.columns]
            if "date" not in have or "code" not in have:
                continue
            frames.append(df[have])
    if not frames:
        raise RuntimeError("无涨停池文件可合并（ths_events 重建失败）")
    ev = pd.concat(frames, ignore_index=True)
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    ev = ev.drop_duplicates(subset=["date", "code"], keep="last")
    return ev.sort_values("date").reset_index(drop=True)


def _union_ignition_codes() -> dict[str, set]:
    """紫阳∪知春路点火（净买>0+涨停）日历 {date: set(code)}（标签源=union）。"""
    from .scorer import _ZCL_CSV
    frames = []
    for path in (C.SEAT_TRADES_CSV, _ZCL_CSV):
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.strftime("%Y-%m-%d")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(幅=("涨跌幅", "max"), 买=("买入金额", "max"), 卖=("卖出金额", "max")))
    g["净"] = g["买"] - g["卖"].fillna(0)
    g = g[(g["净"] > 0) & (g["幅"] >= C.LIMIT_UP_APPROX)
          & g["股票代码"].str[:3].isin(_MB)]
    out: dict[str, set] = {}
    for d, grp in g.groupby("交易日期"):
        out[d] = set(grp["股票代码"])
    return out


def _rolling_tag_unions(ev: pd.DataFrame) -> dict:
    """(date, code, win) → 当日以前 win 自然日内的历史标签并集。

    原实现为「每个交易日 × 每个窗口 × 每只当日股票」反复过滤整张事件表，
    复杂度随事件数平方增长。按 code 分组、按日期升序遍历并用 Counter 增量维护
    标签覆盖，每个历史事件只进入/离开窗口一次；输出键与值保持原口径。
    """
    out = {}
    for code, grp in ev.groupby("code", sort=False):
        dates = grp["date"].tolist()
        tags = grp["tags"].tolist()
        for win in _WINDOWS:
            counts = Counter()
            left = 0
            for i, d in enumerate(dates):
                if i:
                    for tag in tags[i - 1]:
                        counts[tag] += 1
                lo = (dt.date.fromisoformat(d) - dt.timedelta(days=win)).isoformat()
                while left < i and dates[left] < lo:
                    for tag in tags[left]:
                        counts[tag] -= 1
                        if counts[tag] <= 0:
                            del counts[tag]
                    left += 1
                out[(d, code, win)] = set(counts)
    return out


def build_features() -> str:
    """全量重算题材特征表（训练/推理共读；重训前调用一次）。"""
    ev = _build_ths_events()
    fstore.write_dataframe_csv(EVENTS_CSV, ev)
    ign = _union_ignition_codes()

    # 标签 lift 图：date → {tag: 收缩 lift}，只用 date 以前窗口内的点火标签频次
    ev["tags"] = ev["reason_type"].fillna("").map(
        lambda s: [t for t in str(s).split("+") if t]) if "reason_type" in ev else [[]] * len(ev)
    history_tags = _rolling_tag_unions(ev)
    rows = []
    for d, de in ev.groupby("date", sort=True):
        # 基线：当日涨停池全体标签频
        base: dict[str, int] = {}
        for tags in de["tags"]:
            for t in tags:
                base[t] = base.get(t, 0) + 1
        n_base = len(de)
        # 点火标签频（当日）
        ign_codes = ign.get(d, set())
        pos: dict[str, int] = {}
        n_pos = 0
        for r in de.itertuples(index=False):
            if r.code in ign_codes:
                n_pos += 1
                for t in r.tags:
                    pos[t] = pos.get(t, 0) + 1
        lifts = {t: (pos.get(t, 0) + 0.0) / (base[t] * max(n_pos, 1) / max(n_base, 1) + _K)
                 for t in base} if n_pos else {}
        # 滚动窗口特征（只往前看）：标签并集已按 code 增量预计算，查表即可。
        for win in _WINDOWS:
            for code in de["code"]:
                tags90 = history_tags.get((d, code, win), set())
                if not tags90:
                    rows.append({"date": d, "code": code, "win": win, "val": 0.0})
                    continue
                ls = [lifts.get(t, 0.0) for t in tags90]
                # 近 win 日每日 lift 是当日算的——历史 lift 需存图；简化：用当前
                # 图的 90 日均值近似（生产口径标注近似，随每日重算自愈）
                rows.append({"date": d, "code": code, "win": win,
                             "val": round(max(ls) if win == 90 else sum(ls) / len(ls), 4)})
    out = pd.DataFrame(rows).pivot_table(index=["date", "code"], columns="win",
                                         values="val").reset_index()
    out.columns = ["date", "code", "theme30_b", "theme_max_b"]
    fstore.write_dataframe_csv(FEAT_CSV, out)
    print(f"ths_feat_daily.csv: {len(out)} 行（覆盖 {out['date'].nunique()} 日）")
    return FEAT_CSV


if __name__ == "__main__":
    build_features()


def attach_theme_features(rows: list[dict], trade_date) -> None:
    """推理时给宇宙行注入 theme_max_b/theme30_b（滚动 90/30 日查表，缺行=0
    中性；表不存在时全 0——两列全 0 经日内秩变换为常数，排序不变=安全回滚）。

    修 bug：此前只查 trade_date 当日池内票（宇宙 536 只全零）；正确口径=
    个股近 90 日内**任意一次**上榜的标签特征都应可见（deep5 题材席口径）。
    """
    import datetime as _dt
    d = trade_date.isoformat()
    d90 = (trade_date - _dt.timedelta(days=90)).isoformat()
    d30 = (trade_date - _dt.timedelta(days=30)).isoformat()
    if not os.path.exists(FEAT_CSV):
        for r in rows:
            r["theme_max_b"] = r["theme30_b"] = 0.0
        return
    try:
        f = pd.read_csv(FEAT_CSV, dtype={"code": str}, encoding="utf-8-sig")
        f["code"] = f["code"].str.zfill(6)
        f = f[f["date"] < d]                        # 只用当日以前
    except Exception:                                # noqa: BLE001 表损坏不阻断
        for r in rows:
            r["theme_max_b"] = r["theme30_b"] = 0.0
        return
    for r in rows:
        code = r["code"]
        hist = f[f["code"] == code]
        if not len(hist):
            r["theme_max_b"] = r["theme30_b"] = 0.0
            continue
        h90 = hist[hist["date"] >= d90]
        h30 = hist[hist["date"] >= d30]
        r["theme_max_b"] = float(h90["theme_max_b"].max()) if len(h90) else 0.0
        r["theme30_b"] = float(h30["theme30_b"].mean()) if len(h30) else 0.0


# ================================================================ v2.6 分时轨迹特征（tp）
# deep7 Round1（2026-09-28）：deep4/ths_mining.md §四组间差异三维中取二维
# ——tp_touch（首触板点序，慢热=晚触板偏好）+ tp_am（上午 60 分钟均涨，开盘
# 不抢）。走样前验证（deep7/wf_opt.py，框架=deep6/wf_profit 逐字复刻，
# identity 自检逐位一致）：EXEC -0.1933→+0.0362，RECALL 0.0473→0.0450
# （≥0.75×基线 0.0386 门），WIN 49.6→50.4，6/6 月 EXEC 不劣于现行，DEV 窗
# （≤08-31）EXEC +0.1007 亦为全变体最优（选定无泄漏）。tp_reopen/三维版
# 或伤 RECALL 门或增益减半，不取。数据源与 themed 同池（deep4 历史 + 生产
# ths_pool/），池文件缺失时特征列全缺失→中位数填充→安全回滚。
_TP_FEATS = ("tp_touch", "tp_am")
_TP_TOUCH_TH, _TP_SEAL_TH = 9.5, 9.8   # 触板=股性口径 9.5；封板=生产 9.8


def _parse_tp(s) -> tuple | None:
    """80 点 3 分钟涨幅轨迹 → (首触板点序, 上午 60 分钟均涨)。

    首触板=首个 ≥9.5 的点（无触板=80）；上午=前 20 点（3 分钟/点）均值。
    结构核验依据 deep4：恒 80 点、末点=收盘涨幅（MAD=0）。
    """
    try:
        vals = [float(x) for x in str(s).split(";") if x != ""]
    except ValueError:
        return None
    if len(vals) < 60:
        return None
    touch = len(vals)
    for k, v in enumerate(vals):
        if v >= _TP_TOUCH_TH:
            touch = k
            break
    return float(touch), sum(vals[:20]) / 20.0


_TP_STORE: dict | None = None      # code -> [(date, (touch, am)), …] 按日升序


def _tp_store() -> dict:
    """ths_pool_*.csv（deep4 历史 + 生产目录，与 _build_ths_events 同 glob）
    → code → 按日升序的 (date, feats)。进程内缓存，只读本地。"""
    global _TP_STORE
    if _TP_STORE is not None:
        return _TP_STORE
    import glob as _glob
    import re as _re
    store: dict = {}
    seen: set = set()
    pats = (os.path.join(C.ROOT, "out", "research", "ziyang", "deep4",
                         "ths_pool_*.csv"),
            os.path.join(C.DATA_DIR, "ths_pool", "*.csv"))
    for pat in pats:
        for f in sorted(_glob.glob(pat)):
            try:
                df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
            except Exception:                      # noqa: BLE001
                continue
            if "time_preview" not in df.columns or "code" not in df.columns:
                continue
            if "date" in df.columns:
                dates = df["date"].astype(str)
            else:
                m = _re.search(r"(\d{8})", os.path.basename(f))
                if not m:
                    continue
                dates = pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
            for d, code, tp in zip(dates, df["code"].astype(str).str.zfill(6),
                                   df["time_preview"]):
                if (d, code) in seen or not isinstance(tp, str):
                    continue
                feats = _parse_tp(tp)
                if feats is None:
                    continue
                seen.add((d, code))
                store.setdefault(code, []).append((d, feats))
    for code in store:
        store[code].sort()
    _TP_STORE = store
    return store


def attach_tp_features(rows: list[dict], trade_date) -> None:
    """宇宙行注入 tp_touch/tp_am：该码 ≤D 最近一次池上轨迹（90 自然日窗）。

    D 晚可知（D 日池快照收盘后落盘；D 日自身在池=用当日轨迹）。无池史→
    不设键（logit_matrix 中位数填充，秩居中=中性）。幂等：重复 attach 同值。
    """
    import bisect as _bisect
    import datetime as _dt
    store = _tp_store()
    d = trade_date.isoformat()
    d90 = (trade_date - _dt.timedelta(days=90)).isoformat()
    for r in rows:
        hist = store.get(r["code"])
        if not hist:
            for k in _TP_FEATS:
                r.pop(k, None)
            continue
        dates = [h[0] for h in hist]
        j = _bisect.bisect_right(dates, d) - 1
        if j < 0 or dates[j] < d90:
            for k in _TP_FEATS:
                r.pop(k, None)
            continue
        r["tp_touch"], r["tp_am"] = hist[j][1]
