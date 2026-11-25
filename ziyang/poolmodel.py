# -*- coding: utf-8 -*-
"""票池建模 + 预判成功率模块（2026-09-28 用户令新建）。

两个独立模型，共享走样前框架：

1. 票池建模（pool_model）——对当日候选宇宙（紫阳甄选 + 四通道）训练
   「次日赚钱概率」logistic：标签=次日开→收 >0（可执行口径，扣一字），
   输出每只票的 P(赚) + 票池整体期望胜率。用途：从 700 只宇宙中
   挑出「明天买了大概率不亏」的票，与紫阳甄选（席位视角）互补。

2. 预判成功率模块（success_rate）——对紫阳甄选 top3 的「今天这批
   预判有几成把握」建模：日级二分类（当日至少 1 命中）+ 票级 P(命中)。
   日级特征=宇宙规模/涨停触及/席位近5日事件/分数集中度（预测日晚可知）；
   票级特征=排名/分数差/通道/题材/复买。输出：日级置信 0~1 + 每票概率。

两个模型都是建议层：不写台账、不改 v2.5 冻结口径、由前向数据独立验证。
"""
from __future__ import annotations

import datetime as dt
import json
from bisect import bisect_right
import math
import os

import fstore
import numpy as np

from . import config as C

POOL_MODEL_JSON = os.path.join(C.DATA_DIR, "model", "pool_model.json")
SR_MODEL_JSON = os.path.join(C.DATA_DIR, "model", "success_rate.json")

_MB = C.MAIN_BOARD_PREFIXES   # 主板口径（包内单一源，此前 4 处手抄）


# ================================================================ 共享工具
def _load_stocks(bars_cache: dict) -> dict:
    """bars_cache → 训练用数组包（与 scorer._stock_arrays 同口径）。"""
    from .scorer import _stock_arrays
    stocks = {}
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in _MB:
            continue
        S = _stock_arrays(b)
        if S is not None:
            stocks[str(code).zfill(6)] = S
    return stocks


def _union_netbuy() -> dict:
    from .scorer import _netbuy_calendar
    return _netbuy_calendar()


def _next_bar(S: dict, d: str) -> int | None:
    """d 之后第一根 K 线下标；日期已升序，二分替代逐行扫描。"""
    i = bisect_right(S["dates"], d)
    return i if i < len(S["dates"]) else None


# ================================================================ 1. 票池建模
def _pool_row_at(S: dict, code: str, i: int) -> dict | None:
    """从数组包取特征行（与 scorer._train_row_at 同口径，无标签）。"""
    from .scorer import _train_row_at
    return _train_row_at(S, code, i, {}, {})


def _pool_feats(rows: list[dict]) -> np.ndarray:
    """票池特征矩阵：七维连续值 + 通道哑变量（不做日内分位——票池模型
    面向全宇宙而非日截面排名，用原始值+标准差标准化）。列顺序见
    _POOL_FEAT_NAMES（price/ltsz 取 log10）。"""
    X = []
    for r in rows:
        X.append([
            float(r.get("lianban") or 0),
            float(r.get("pct") or 0),
            float(r.get("turnover") or 0),
            float(r.get("vr") or 0),
            math.log10(max(float(r.get("price") or 1), 0.01)),
            float(r.get("pos20") or 0.5),
            math.log10(max(float(r.get("ltsz") or 1), 0.01)),
            float(r.get("zt60") or 0),
            float(r.get("maxlb60") or 0),
            float(r.get("m5") or 0),
            1.0 if r.get("via_switch") else 0.0,
            1.0 if r.get("via_m5") else 0.0,
            1.0 if r.get("via_stir") else 0.0,
        ])
    return np.array(X, dtype=float)


_POOL_FEAT_NAMES = ["lianban", "pct", "turnover", "vr", "log_price", "pos20",
                    "log_ltsz", "zt60", "maxlb60", "m5",
                    "ch_switch", "ch_momentum", "ch_stir"]


def train_pool_model(bars_cache: dict) -> dict:
    """月度走样前训练「次日赚钱概率」模型（票池层）。

    标签：次日开→收 > 0（扣一字：开盘≥昨收×1.10 剔除）。
    """
    stocks = _load_stocks(bars_cache)
    cutoff = dt.date.today().replace(day=1)
    cutoff_s = cutoff.isoformat()
    Xs, ys = [], []
    for d in sorted(set(sum([S["dates"] for S in stocks.values()], []))):
        if d >= cutoff_s:
            break
        rows = []
        for code, S in stocks.items():
            i = S["idx"].get(d)
            if i is None or i < 6:
                continue
            r = _pool_row_at(S, code, i)
            if r is None:
                continue
            j = _next_bar(S, d)
            if j is None or j + 1 >= len(S["dates"]):
                continue
            # 一字剔除
            limit = round(float(S["close"][i]) * 1.10, 2)
            if float(S["open"][j]) >= limit - 0.001:
                continue
            ret = (float(S["close"][j]) / float(S["open"][j]) - 1.0) * 100.0
            r["_y"] = 1.0 if ret > 0 else 0.0
            rows.append(r)
        if len(rows) < 50:
            continue
        Xs.append(_pool_feats(rows))
        ys.extend(r["_y"] for r in rows)
    if not ys or sum(ys) < 20:
        return {"ok": False, "reason": f"正例不足: {sum(ys) if ys else 0}"}
    X = np.vstack(Xs)
    y = np.array(ys)
    # 标准化（特征量纲差异大：lianban~10 vs ltsz~1000）
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    # GPU 优先训练（用户令：跑模型全部用 GPU）
    from . import scorer as _sc; _fit_logistic = _sc._fit_logistic
    coef, intercept = _fit_logistic(Xn, y)
    meta = {
        "ok": True, "version": "pool-v1",
        "backend": _sc._fit_backend,
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cutoff": cutoff_s, "n_rows": len(y), "n_pos": int(y.sum()),
        "base_rate": round(float(y.mean()), 3),
        "features": _POOL_FEAT_NAMES,
        "coef": [round(float(v), 4) for v in coef],
        "intercept": round(float(intercept), 4),
        "mu": [round(float(v), 4) for v in mu],
        "sd": [round(float(v), 4) for v in sd],
        "label": "次日开→收>0（扣一字）",
    }
    fstore.write_json(POOL_MODEL_JSON, meta, indent=2)
    return meta


def _sig(z: float) -> float:
    """数值安全 sigmoid（裁剪 ±30 防 exp 溢出 → NaN）。"""
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


def score_pool(rows: list[dict]) -> list[dict]:
    """推理：给宇宙行打 P(次日赚钱) 概率。"""
    with open(POOL_MODEL_JSON, encoding="utf-8") as f:
        m = json.load(f)
    if not m.get("ok"):
        raise RuntimeError("票池模型不可用")
    X = np.nan_to_num(_pool_feats(rows), nan=0.0, posinf=50.0, neginf=-50.0)
    mu = np.array(m["mu"])
    sd = np.array(m["sd"])
    w = np.array(m["coef"])
    logit = ((X - mu) / sd) @ w + m["intercept"]
    prob = [_sig(float(z)) for z in logit]
    out = []
    for r, p in zip(rows, prob):
        out.append(dict(r, pool_prob=round(p, 3)))
    out.sort(key=lambda x: -x["pool_prob"])
    return out


# ================================================================ 2. 预判成功率
_SR_DAY_FEATS = ["universe_n", "ztc", "seat_5d_events", "top1_score", "score_std"]
_SR_TICKET_FEATS = ["rank", "score", "margin", "via_switch", "via_m5", "theme"]


def train_success_rate(bars_cache: dict) -> dict:
    """月度走样前训练预判成功率模型（两层）。

    日级：P(当日 top3 至少 1 命中席位净买) —— logistic on 日上下文特征。
    票级：P(该票命中) —— logistic on 票级特征。
    """
    stocks = _load_stocks(bars_cache)
    netbuy = _union_netbuy()
    cutoff = dt.date.today().replace(day=1)
    cutoff_s = cutoff.isoformat()
    all_days = sorted(set(sum([S["dates"] for S in stocks.values()], [])))

    day_X, day_y = [], []
    tick_X, tick_y = [], []
    for d in all_days:
        if d >= cutoff_s:
            break
        rows = []
        for code, S in stocks.items():
            i = S["idx"].get(d)
            if i is None or i < 6:
                continue
            r = _pool_row_at(S, code, i)
            if r is not None:
                rows.append(r)
        if len(rows) < 50:
            continue
        j = _next_bar(list(stocks.values())[0], d)  # 近似下一交易日
        if j is None:
            continue
        next_d = None
        for S in stocks.values():
            nj = _next_bar(S, d)
            if nj is not None:
                next_d = S["dates"][nj]
                break
        if next_d is None:
            continue
        labels = netbuy.get(next_d, set())
        if not labels:
            continue
        # 排名：按现有 v2.5 逻辑——简化为用 v2 分数代理（训练近似，生产用真实模型）
        # 此处用 logit 分数近似（lianban+pct+ltsz 的加权）
        scores = []
        for r in rows:
            s = (0.5 * float(r.get("lianban") or 0)
                 + 0.02 * float(r.get("pct") or 0)
                 + 0.3 * math.log10(max(float(r.get("ltsz") or 1), 1)))
            scores.append(s)
        order = sorted(range(len(rows)), key=lambda k: -scores[k])
        top3 = order[:3]
        hits = sum(1 for k in top3 if rows[k]["code"] in labels)
        # 日级特征
        ztc = sum(1 for S2 in stocks.values()
                  if (i2 := S2["idx"].get(d)) is not None
                  and np.isfinite(S2["hpct"][i2]) and S2["hpct"][i2] >= 9.5)
        d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=7)).isoformat()
        seat_5d = sum(len(v) for k, v in netbuy.items() if d0 <= k < d)
        s_vals = [scores[k] for k in top3]
        day_X.append([len(rows), ztc, seat_5d, s_vals[0] if s_vals else 0,
                      float(np.std(s_vals)) if len(s_vals) > 1 else 0])
        day_y.append(1.0 if hits > 0 else 0.0)
        # 票级特征
        top1_s = s_vals[0] if s_vals else 1.0
        for rank_i, k in enumerate(top3, 1):
            r = rows[k]
            tick_X.append([
                rank_i,
                scores[k],
                scores[k] - top1_s,
                1.0 if r.get("via_switch") else 0.0,
                1.0 if r.get("via_m5") else 0.0,
                float(r.get("theme_max_b") or 0),
            ])
            tick_y.append(1.0 if r["code"] in labels else 0.0)

    meta = {"version": "sr-v1",
            "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cutoff": cutoff_s}
    dy = np.array(day_y)
    if len(dy) >= 10 and dy.sum() >= 3:
        DX = np.array(day_X, dtype=float)
        dmu, dsd = DX.mean(axis=0), DX.std(axis=0) + 1e-9
        from .scorer import _fit_logistic as _gpu_fit
        dcoef, dinter = _gpu_fit((DX - dmu) / dsd, dy)
        meta["day"] = {"ok": True, "features": _SR_DAY_FEATS,
                       "coef": [round(float(v), 4) for v in dcoef],
                       "intercept": round(float(dinter), 4),
                       "mu": [round(float(v), 4) for v in dmu],
                       "sd": [round(float(v), 4) for v in dsd],
                       "n_days": len(dy), "base_rate": round(float(dy.mean()), 3)}
    else:
        meta["day"] = {"ok": False, "reason": f"日样本不足: {len(dy)}/{dy.sum() if len(dy) else 0}"}

    ty = np.array(tick_y)
    if len(ty) >= 30 and ty.sum() >= 5:
        TX = np.array(tick_X, dtype=float)
        tmu, tsd = TX.mean(axis=0), TX.std(axis=0) + 1e-9
        from .scorer import _fit_logistic as _gpu_fit
        tcoef, tinter = _gpu_fit((TX - tmu) / tsd, ty)
        meta["ticket"] = {"ok": True, "features": _SR_TICKET_FEATS,
                          "coef": [round(float(v), 4) for v in tcoef],
                          "intercept": round(float(tinter), 4),
                          "mu": [round(float(v), 4) for v in tmu],
                          "sd": [round(float(v), 4) for v in tsd],
                          "n": len(ty), "base_rate": round(float(ty.mean()), 3)}
    else:
        meta["ticket"] = {"ok": False, "reason": f"票样本不足: {len(ty)}/{ty.sum() if len(ty) else 0}"}

    fstore.write_json(SR_MODEL_JSON, meta, indent=2)
    return meta


def predict_success_rate(day_feats: list[float],
                         ticket_feats: list[list[float]]) -> dict:
    """推理：日级置信 + 每票命中概率。"""
    def _safe_logit(m, X):
        mu = np.array(m["mu"])
        sd = np.array(m["sd"])
        sd = np.where(np.abs(sd) < 1e-9, 1.0, sd)   # 零方差列→除零→NaN 防护
        w = np.array(m["coef"])
        raw = ((np.array(X, dtype=float) - mu) / sd) @ w + m["intercept"]
        return np.nan_to_num(raw, nan=0.0, posinf=30.0, neginf=-30.0)

    with open(SR_MODEL_JSON, encoding="utf-8") as f:
        m = json.load(f)
    out = {}
    if m.get("day", {}).get("ok"):
        z = float(_safe_logit(m["day"], [day_feats])[0])
        out["day_prob"] = round(_sig(z), 3)
    if m.get("ticket", {}).get("ok"):
        zs = _safe_logit(m["ticket"], ticket_feats)
        probs = [_sig(float(z)) for z in zs]
        out["ticket_probs"] = [round(p, 3) for p in probs]
        out["expected_hits"] = round(sum(probs), 2)
    return out


# ================================================================ CLI
def main() -> int:
    from .seatdata import load_bars_cache
    print("[票池模型] 训练…")
    bars = load_bars_cache()
    pm = train_pool_model(bars)
    print(f"  → {POOL_MODEL_JSON}: ok={pm.get('ok')} base_rate={pm.get('base_rate')}")
    print("[成功率模型] 训练…")
    sr = train_success_rate(bars)
    print(f"  → {SR_MODEL_JSON}: day.ok={sr.get('day',{}).get('ok')} "
          f"ticket.ok={sr.get('ticket',{}).get('ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
