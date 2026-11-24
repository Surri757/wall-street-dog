# -*- coding: utf-8 -*-
"""vane 三日自学习模型（口径冻结见 vane/RESEARCH.md，2026-11-22 立项）。

对齐 ziyang 月度模型功能面：walk-forward 训练 / 三日门自动重训 / 坏件自愈 /
isotonic 校准 / 贡献解释 / 每日微调 / 新旧对照 / 台账 by_version 归因。

口径（用户拍板 + 调研冻结）：
- 训练宇宙 = 每日主板非ST封板股中本轮板数 B≥2 的**全体**候选（B 作特征；
  推理只对 B∈{4,5} 应用）——样本经济与 ziyang「宇宙全体行+稀有标签」同构；
- 标签 y = code ∈ T+1 涨停池（晋级成功；纯池口径不剔一字——一字晋级也是
  情绪延续，观察标语义）；
- 训练日 ≤ cutoff（=窗口倒数第二日，标签需 T+1）、标签日 < cutoff——双日
  纪律零泄漏；
- 排序 = 模型主排 + 启发式 shadow_rank 影子登记（台账 by_version 晋级率
  形成「模型 vs 启发式」持续实证——先标签后计分纪律的落地形式）；
- 特征用 poolmodel 式标准化（mu/sd 存档）而非日内分位：池宇宙日截面仅
  5-15 行，分位排名会退化为常数。

继承的三个已知坑（ziyang 实证）：①原子写 tmp 必须保留原扩展名（xgboost
按扩展名选 UBJSON/文本——fstore 已修）；②xgb 续训必须 xgb.train(params,
num_boost_round=N, xgb_model=booster) 显式传参（update 的 iteration 是
轮次编号不是棵数）；③任何模型异常必须打印留痕不静默。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

import numpy as np

import fstore
from data_source import parse_pool, pool_raw

from . import config as C
from .engine import sort_key
from .screen import _is_main_board, _num_or_none, _window_days, _ymd


# ---------------------------------------------------------------- 池数据

def _load_pools(days: list[dt.date]) -> tuple[dict, list[str]]:
    """逐日主板非ST封板池（pool_raw 进程内 memo——scan 已拉过的日零网络）。"""
    pools, miss = {}, []
    for d in days:
        rows = parse_pool("zt", pool_raw("zt", _ymd(d)))
        if not rows:
            miss.append(f"{d:%Y-%m-%d}")
            pools[d] = {}
            continue
        pools[d] = {r["code"]: r for r in rows
                    if _is_main_board(r["code"])
                    and not (C.EXCLUDE_ST and "ST" in r["name"].upper())}
    return pools, miss


# ---------------------------------------------------------------- 特征行

def _round_state(code: str, idx: int, days: list[dt.date],
                 pools: dict) -> dict:
    """与 engine.streak 同口径的 B/span/breaks（池日集合版）。"""
    from .engine import streak
    cal = [f"{d:%Y-%m-%d}" for d in days]
    zt_dates = [cal[i] for i in range(idx + 1) if code in pools[days[i]]]
    st = streak(zt_dates, cal[idx], cal[:idx + 1], C.MAX_GAP)
    return st or {"B": 0, "span": 0, "breaks": 0}


def _feature_row(code: str, prow: dict, idx: int, days: list[dt.date],
                 pools: dict) -> dict:
    """单候选 (code, T) 的 12 列特征（全 T 收盘可知，纯池口径）。"""
    st = _round_state(code, idx, days, pools)
    amount, fund = prow["amount"], prow["fund"]
    seal = (fund / amount) if (fund == fund and amount > 0) else None
    fbt = prow.get("fbt") or ""
    fbt_min = (int(fbt[:2]) * 60 + int(fbt[2:4])) if len(fbt) == 4 else None
    ltsz = _num_or_none(prow.get("ltsz"))
    close = prow.get("close")
    ind = prow.get("industry") or ""
    pool = pools[days[idx]]
    mkt_height = max((r["lianban"] for r in pool.values()), default=0)
    ind_rows = [r for r in pool.values() if ind and r.get("industry") == ind]
    prev_h = [max((r["lianban"] for r in pools[days[j]].values()), default=0)
              for j in range(max(0, idx - 3), idx) if pools.get(days[j])]
    return {
        "date": f"{days[idx]:%Y-%m-%d}", "code": code,
        "name": prow.get("name", ""), "industry": ind,
        "B": st["B"], "span": st["span"], "breaks": st["breaks"],
        "seal_ratio": seal, "fbt_min": fbt_min,
        "log_close": math.log10(close) if close and close > 0 else None,
        "log_ltsz": math.log10(ltsz / 1e8) if ltsz and ltsz > 0 else None,
        "mkt_height": mkt_height,
        "mkt_two_plus": sum(1 for r in pool.values() if r["lianban"] >= 2),
        "ind_zt_count": len(ind_rows),
        "ind_max_b": max((r["lianban"] for r in ind_rows), default=0),
        "height_trend": (mkt_height - sum(prev_h) / len(prev_h))
                        if prev_h else None,
    }


# ---------------------------------------------------------------- 样本

def build_samples(trade_date: dt.date, window: int = C.WINDOW):
    """walk-forward 样本：窗口内每个 T（T+1 也在窗口）的 B≥2 候选行 + 晋级标签。

    返回 (samples, days, miss_days)。样本只到倒数第二日（末日无 T+1 标签）；
    训练日 ≤ cutoff=days[-2]、标签日 < cutoff——双日纪律零泄漏。

    缺池日的处理：窗口头部缺池=超出接口保留期（常态），裁掉；窗口内部/尾部
    缺池=接口抖动，以最后一个缺池日为界只保留其后连续覆盖的尾段。缺池日
    此前照常进样本——标签日拉取失败时该日全部样本被记成负例，train 又丢弃
    miss 照常落盘，伪负例模型还被三日门视为新鲜。
    """
    days, _degraded = _window_days(trade_date, window)
    if len(days) < 3:
        return [], days, []
    pools, miss = _load_pools(days)
    last_miss = max((i for i, d in enumerate(days) if not pools[d]), default=-1)
    days = days[last_miss + 1:]
    if len(days) < 3:
        return [], days, miss
    samples = []
    for i in range(len(days) - 1):
        nxt = days[i + 1]
        for code, prow in pools[days[i]].items():
            feat = _feature_row(code, prow, i, days, pools)
            if feat["B"] < 2:               # 训练宇宙 = B≥2 全体（用户拍板）
                continue
            feat["y"] = 1 if code in pools[nxt] else 0
            samples.append(feat)
    return samples, days, miss


def samples_report(trade_date: dt.date) -> str:
    """样本经济探测（python -m vane samples）：逐 B 层样本量与晋级率。"""
    samples, days, miss = build_samples(trade_date)
    span = (f"{days[0]} ~ {days[-1]}" if days else "-")
    lines = [f"窗口 {len(days)} 个交易日（{span}）"
             + (f"｜缺池 {len(miss)} 日" if miss else "")]
    if not samples:
        lines.append("样本 0 行（窗口不足或池拉取失败）")
        return "\n".join(lines)
    by_b: dict[int, list] = {}
    for s in samples:
        by_b.setdefault(s["B"], []).append(s)
    n_pos = sum(s["y"] for s in samples)
    lines.append(f"样本 {len(samples)} 行 / 正例 {n_pos}"
                 f"（基率 {100.0 * n_pos / len(samples):.0f}%）")
    for b in sorted(by_b):
        g = by_b[b]
        p = sum(s["y"] for s in g)
        lines.append(f"  {b}板: {len(g):3d} 行, 晋级 {p}/{len(g)}"
                     f" = {100.0 * p / len(g):.0f}%")
    gate = ("✓" if (n_pos >= C.SAMPLES_MIN_POS
                    and len(samples) >= C.SAMPLES_MIN_ROWS)
            else f"✗（低于样本门：正例≥{C.SAMPLES_MIN_POS} 且"
                 f"行数≥{C.SAMPLES_MIN_ROWS}）")
    lines.append(f"样本门: {gate}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 特征矩阵

def _matrix(rows: list[dict], medians: dict | None = None):
    """特征 → (X, medians)。缺失填中位数（推理透传训练中位数，小截面稳定）。"""
    import pandas as pd
    df = pd.DataFrame(rows)
    med, cols = {}, []
    for f in C.FEATURES:
        col = pd.to_numeric(df.get(f), errors="coerce") \
            if f in df.columns else pd.Series([np.nan] * len(df))
        m = (medians or {}).get(f)
        if m is None:
            m = float(col.median()) if col.notna().any() else 0.0
        med[f] = float(m)
        cols.append(col.fillna(m).to_numpy(dtype=float))
    return np.column_stack(cols), med


def _logreg_artifact(meta: dict, med: dict) -> dict:
    return {"coef": meta["coef"], "intercept": meta["intercept"],
            "medians": med, "mu": meta["mu"], "sd": meta["sd"],
            "features": list(C.FEATURES)}


# ---------------------------------------------------------------- 训练

def _fit_logreg(X, y, day_keys, sample_weight=None):
    """logreg 主路径：C 小网格按 leave-one-day-out 对数损失自动选参。

    小样本（~100-150 行）下的「内置迭代优化」——网格/折数都便宜且诚实。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    best, best_ll = None, None
    for c in C.LOGREG_GRID:
        lls = []
        for d in sorted(set(day_keys)):
            te = np.array([k == d for k in day_keys])
            tr = ~te
            if tr.sum() < 5 or y[tr].sum() < 1 or len(set(y[te])) < 2:
                continue
            clf = LogisticRegression(C=c, max_iter=2000,
                                     class_weight="balanced")
            clf.fit(X[tr], y[tr])
            lls.append(log_loss(y[te], clf.predict_proba(X[te])[:, 1],
                                labels=[0, 1]))
        ll = float(np.mean(lls)) if lls else float("inf")
        if best_ll is None or ll < best_ll:
            best, best_ll = c, ll
    w = sample_weight if sample_weight is not None else np.ones(len(y))
    clf = LogisticRegression(C=best or 1.0, max_iter=2000,
                             class_weight="balanced")
    clf.fit(X, y, sample_weight=w)
    return {"coef": [round(float(v), 6) for v in clf.coef_[0]],
            "intercept": round(float(clf.intercept_[0]), 6),
            "best_C": best, "cv_logloss": round(best_ll, 4),
            "backend": "sklearn-logreg"}


def _fit_xgb(X, y):
    """xgb 门控路径（n≥XGB_GATE_N）：cuda→cpu 兜底（对齐 ziyang 双路径）。

    xgboost 未安装/导入失败时抛出——由 train 捕获回落 logreg 主路径。
    """
    import xgboost as xgb
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    base = {"objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": 3, "learning_rate": 0.05, "tree_method": "hist",
            "scale_pos_weight": spw, "verbosity": 0}
    dtrain = xgb.DMatrix(X, label=y)
    try:
        booster = xgb.train({**base, "device": "cuda"}, dtrain,
                            num_boost_round=120)
        backend = "xgb-cuda"
    except Exception:                          # noqa: BLE001 CUDA 不可用退 CPU
        booster = xgb.train({**base, "device": "cpu"}, dtrain,
                            num_boost_round=120)
        backend = "xgb-cpu"
    return booster, backend


def _iso_fit(p, y):
    """训练集内 isotonic 校准；退化（取值过少）返回 None（meta.iso_error）。"""
    try:
        from sklearn.isotonic import IsotonicRegression
        if len(set(np.round(p, 6))) < 5:
            return None
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return {"x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_]}
    except Exception:                          # noqa: BLE001 校准失败不阻断
        return None


def train(today: dt.date | None = None) -> dict:
    """滚动窗全量训练（三日门触发/手动调用）。meta.ok=False=样本门不过。"""
    import fstore
    today = today or dt.date.today()
    samples, days, _miss = build_samples(today)
    n_pos = sum(s["y"] for s in samples)
    if n_pos < C.SAMPLES_MIN_POS or len(samples) < C.SAMPLES_MIN_ROWS:
        return {"ok": False,
                "reason": f"样本门不过（{len(samples)} 行 / 正例 {n_pos}）"}
    X, med = _matrix(samples)
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
    Xz = (X - mu) / sd
    y = np.array([s["y"] for s in samples], dtype=int)
    day_keys = [s["date"] for s in samples]
    cutoff = days[-2] if len(days) >= 2 else days[-1]
    meta = {"version": C.MODEL_VERSION, "cadence_days": C.CADENCE_DAYS,
            "cutoff": f"{cutoff:%Y-%m-%d}", "n_days": len(days),
            "n_rows": len(samples), "n_pos": int(n_pos),
            "base_rate": round(100.0 * n_pos / len(samples), 1),
            "features": list(C.FEATURES),
            # trained_at 锚定 as-of 交易日（三日门按交易日算距）；墙钟另存 saved_at
            "trained_at": f"{today:%Y-%m-%d} "
                          f"{dt.datetime.now():%H:%M:%S}",
            "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 标准化参数随 meta 落盘：推理/续训/贡献度经 _zspace 还原训练空间
            "mu": [round(float(v), 6) for v in mu],
            "sd": [round(float(v), 6) for v in sd],
            "medians": med,
            "ok": True}
    # 旧件归档：meta→model_prev.json；booster→model_xgb.prev.json；logreg 工件→.prev
    prev_version = None
    if os.path.exists(C.MODEL_JSON):
        try:
            prev = _load_json(C.MODEL_JSON)
            if prev.get("ok"):
                prev_version = prev.get("version")
                fstore.write_text(C.MODEL_PREV_JSON,
                             json.dumps(prev, ensure_ascii=False, indent=1))
                if prev.get("kind") == "xgb" and \
                        os.path.exists(C.MODEL_XGB_JSON):
                    os.replace(C.MODEL_XGB_JSON, _prev_xgb_path())
                if prev.get("kind") == "logreg" and \
                        os.path.exists(C.MODEL_LGR_JSON):
                    fstore.write_text(C.MODEL_LGR_JSON + ".prev",
                                 Path(C.MODEL_LGR_JSON).read_text(
                                     encoding="utf-8"))
        except Exception as e:                 # noqa: BLE001 归档失败留痕不阻断
            meta["prev_archive_error"] = f"{type(e).__name__}: {e}"
    meta["prev_version"] = prev_version
    if len(samples) >= C.XGB_GATE_N:
        try:
            booster, backend = _fit_xgb(Xz, y)
        except Exception as e:                 # noqa: BLE001 xgb 不可用回落 logreg
            meta["xgb_error"] = f"{type(e).__name__}: {e}"
            booster = None
        if booster is not None:
            meta.update({"kind": "xgb", "backend": backend})
            fstore.write_with_bak(C.MODEL_XGB_JSON, booster.save_model)
            # 三日门可能已把旧件载入进程缓存——不失效则下面的保序拟合与当日
            # 打分都还在用被替换掉的旧模型
            _BOOTER_CACHE.pop(C.MODEL_XGB_JSON, None)
    if meta.get("kind") != "xgb":
        # 近期权重半衰 7 日（情绪生态滚动适配）；LOD 选参不加权保持稳健
        age = np.array([(today - dt.date.fromisoformat(k)).days
                        for k in day_keys], dtype=float)
        w = 0.5 ** (age / 7.0)
        fit = _fit_logreg(Xz, y, day_keys, sample_weight=w)
        meta.update({"kind": "logreg", **fit})
        fstore.write_text(C.MODEL_LGR_JSON, json.dumps(
            _logreg_artifact({**meta, "mu": mu.tolist(), "sd": sd.tolist()},
                             med), ensure_ascii=False, indent=1))
    iso = None
    try:
        # 走推理同一条路径出训练集概率：保序表拟合的必须正是日后被打分的东西
        # （此前 logreg 分支对已标准化的 Xz 再标准化一次）
        art = (C.MODEL_XGB_JSON if meta["kind"] == "xgb"
               else C.MODEL_LGR_JSON)
        iso = _iso_fit(_predict_proba(meta, art, X), y)
    except Exception as e:                     # noqa: BLE001 校准失败留痕
        meta["iso_error"] = f"{type(e).__name__}: {e}"
    meta["iso"] = bool(iso)
    if iso:
        fstore.write_text(C.ISO_JSON, json.dumps(iso))
    elif os.path.exists(C.ISO_JSON):
        # 新模型没有校准表：旧模型的 iso 绝不能留用（那是按旧分数拟合的映射）
        os.remove(C.ISO_JSON)
    fstore.write_text(C.MODEL_JSON, json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


# ---------------------------------------------------------------- 推理

_BOOTER_CACHE: dict = {}


def _prev_xgb_path() -> str:
    """上一代 booster 的归档路径：`model_xgb.prev.json`——**保留 .json 扩展名**。

    xgboost 按扩展名选加载格式，`.prev` 结尾会报 Unknown file format 再猜 JSON。
    读取侧兼容旧命名 `.json.prev` 的存量归档。
    """
    new = os.path.splitext(C.MODEL_XGB_JSON)[0] + ".prev.json"
    old = C.MODEL_XGB_JSON + ".prev"
    return old if (not os.path.exists(new) and os.path.exists(old)) else new


def _load_booster(path: str):
    import xgboost as xgb
    if path not in _BOOTER_CACHE:
        b = xgb.Booster()
        b.load_model(path)
        _BOOTER_CACHE[path] = b
    return _BOOTER_CACHE[path]


def booster_loadable(path: str) -> bool:
    try:
        _load_booster(path)
        return True
    except Exception:                          # noqa: BLE001
        return False


def _xgb_ready() -> bool:
    """xgboost 可导入（缺失模块时不得误隔离有效工件——只影响降级路径）。"""
    try:
        import xgboost                          # noqa: F401
        return True
    except Exception:                          # noqa: BLE001
        return False


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _iso_apply(p, iso):
    if not iso:
        return p
    return np.interp(p, iso["x"], iso["y"])


def _load_json(path: str):
    """读 JSON 工件（模型 meta/logreg/保序表）。不存在或损坏照常抛错，
    由调用方决定「按无工件处理」还是留痕——两处都必须显式。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _zspace(X, meta: dict) -> np.ndarray:
    """原始特征 → 训练时的标准化空间。训练统一喂 (X-mu)/sd，所以推理、续训、
    贡献度、保序拟合都必须先过这一步。

    2026-11-24 实证：xgb 路径训练喂的是标准化特征，推理与续训却喂原始值——
    盘上现行工件对两种输入的输出秩相关仅 0.54（fbt_min 均值 622、log_ltsz 等
    特征的分裂点在原始值下恒落一侧），「模型主排」约半数排序信息失真。
    """
    mu = np.array(meta["mu"], dtype=float)
    sd = np.array(meta["sd"], dtype=float)
    return (np.asarray(X, dtype=float) - mu) / sd


def _predict_proba(meta: dict, art_path: str, X) -> np.ndarray:
    """按 meta.kind 出**未过保序校准**的原始概率（art_path 可指向 .prev 旧件）。

    当日主排、新旧对照、训练时的保序拟合共用这一个出口：三处的打分此前各抄
    一份，改了其一漏其另一就不是同口径。
    """
    if meta.get("kind") == "xgb":
        import xgboost as xgb
        return _load_booster(art_path).predict(xgb.DMatrix(_zspace(X, meta)))
    art = _load_json(art_path)
    z = _zspace(X, art)
    return 1.0 / (1.0 + np.exp(-(z @ np.array(art["coef"], dtype=float)
                                 + art["intercept"])))


def _read_iso():
    try:
        return _load_json(C.ISO_JSON)
    except (OSError, ValueError):              # 无 iso / 坏 iso = 不校准
        return None


def _contribs(meta: dict, X_row) -> str:
    """top3 贡献特征（xgb: pred_contribs 近似 SHAP；logreg: |z·coef|）。X_row 为原始特征行。"""
    try:
        row = np.asarray(X_row, dtype=float).reshape(1, -1)
        if meta.get("kind") == "xgb":
            import xgboost as xgb
            booster = _load_booster(C.MODEL_XGB_JSON)
            c = booster.predict(xgb.DMatrix(_zspace(row, meta)),
                                pred_contribs=True)[0][:-1]
        else:
            art = _load_json(C.MODEL_LGR_JSON)
            c = _zspace(row, art)[0] * np.array(art["coef"], dtype=float)
        order = np.argsort(-np.abs(c))[:3]
        return "+".join(C.FEATURES[i] for i in order)
    except Exception:                          # noqa: BLE001 解释失败不阻断
        return ""


# ---------------------------------------------------------------- 三日门

def _meta_load() -> dict | None:
    try:
        meta = _load_json(C.MODEL_JSON)
    except (OSError, ValueError):              # 无 meta / 坏 meta = 无模型
        return None
    return meta if isinstance(meta, dict) else None


def _usable(meta: dict | None, days: list[dt.date]) -> bool:
    """三日门：ok ∧ 三日新鲜 ∧ 同版本 ∧ 同特征集 ∧（xgb）工件可加载。"""
    if not _same_version(meta):
        return False
    t_str = (meta.get("trained_at") or "")[:10]
    if t_str not in {f"{d:%Y-%m-%d}" for d in days}:
        return False                           # 旧于窗口=必然过期
    idx = max(i for i, d in enumerate(days)
              if f"{d:%Y-%m-%d}" == t_str)
    if (len(days) - 1) - idx >= C.CADENCE_DAYS:
        return False
    if meta.get("kind") == "xgb" and \
            not booster_loadable(C.MODEL_XGB_JSON):
        return False
    return True


def ensure_model(today: dt.date | None = None):
    """三日门：可复用返回 meta；否则重训。失败时同口径回退，无模型返回 None。

    坏件自愈：xgb 工件不可加载 → 隔离 .corrupt 再重训（2026-11-22 教训）。
    返回 (meta|None, note)。
    """
    today = today or dt.date.today()
    days, _degraded = _window_days(today, C.WINDOW)
    meta = _meta_load()
    if _usable(meta, days):
        return meta, "三日门通过，复用现有模型"
    if meta and meta.get("kind") == "xgb" and _xgb_ready() and \
            os.path.exists(C.MODEL_XGB_JSON) and \
            not booster_loadable(C.MODEL_XGB_JSON):
        try:
            os.replace(C.MODEL_XGB_JSON, C.MODEL_XGB_JSON + ".corrupt")
            print("[风向标] 模型工件损坏，已隔离 .corrupt 并重训")
        except OSError:
            pass
    fresh = train(today)
    if fresh.get("ok"):
        return fresh, (f"三日重训完成（{fresh['n_rows']} 行 / 正例 "
                       f"{fresh['n_pos']}）")
    # 回退护栏：仅接受同版本同特征集旧模型（拒绝跨口径静默回退）
    if _usable_version_only(meta):
        return meta, f"重训失败（{fresh.get('reason')}），沿用旧模型"
    print(f"[风向标] 模型不可用：{fresh.get('reason')}——本次回退启发式排序")
    return None, f"无模型（{fresh.get('reason')}），启发式排序"


def _same_version(meta: dict | None) -> bool:
    """同版本 ∧ 同特征集 ∧ 带标准化参数（缺 mu/sd = 无法同口径推理，一律重训）。"""
    return bool(meta and meta.get("ok")
                and meta.get("version") == C.MODEL_VERSION
                and meta.get("features") == list(C.FEATURES)
                and len(meta.get("mu") or []) == len(C.FEATURES)
                and len(meta.get("sd") or []) == len(C.FEATURES))


def _usable_version_only(meta: dict | None) -> bool:
    return _same_version(meta) and (meta.get("kind") != "xgb"
                                    or booster_loadable(C.MODEL_XGB_JSON))


# ---------------------------------------------------------------- 评分主排

def apply_ranking(rows: list[dict], trade_date: dt.date,
                  meta: dict | None) -> list[dict]:
    """模型打分 + 启发式影子：返回附 model_score/shadow_rank/score100 的 rows。

    - meta 可用：对全部 B≥2 行出分（logit 尺度，按 code 关联）；
    - meta=None：启发式原样，shadow_rank 照登（先标签后计分——两套排序的
      晋级率都进台账，by_model_version 持续实证）。
    排序本身由 engine.select(key=model_rank_key) 在调用方完成。
    """
    rows = [dict(r) for r in rows]
    qual = [r for r in rows if int(r.get("B") or 0) in C.BOARD_RANGE]
    # 启发式影子（无论模型是否可用，两套排序都留证）
    for rk, r in enumerate(sorted(qual, key=sort_key), 1):
        r["shadow_rank"] = rk
    if not meta:
        for r in rows:
            r["model_score"] = None
            r["model_version"] = ""
        return rows
    # 今日截面特征（pool_raw memo：scan 已拉过的日零网络复用）
    days, _degraded = _window_days(trade_date, C.WINDOW)
    pools, _miss = _load_pools(days)
    idx = len(days) - 1
    today_pool = pools.get(days[idx], {})
    scores: dict[str, float] = {}
    contribs: dict[str, str] = {}
    codes = [r["code"] for r in rows if r["code"] in today_pool]
    if codes:
        feats = [_feature_row(c, today_pool[c], idx, days, pools) for c in codes]
        X, _ = _matrix(feats, medians=meta.get("medians"))
        art_path = (C.MODEL_XGB_JSON if meta.get("kind") == "xgb"
                    else C.MODEL_LGR_JSON)
        p = _predict_proba(meta, art_path, X)
        L = _logit(_iso_apply(p, _read_iso()))
        for c, v in zip(codes, L):
            scores[c] = float(v)
        for c, i in zip(codes, range(len(codes))):
            contribs[c] = _contribs(meta, X[i])
    qs = [scores[c] for c in codes if c in scores]
    lo, hi = (min(qs), max(qs)) if qs else (0.0, 0.0)
    span = (hi - lo) or 1.0
    for r in rows:
        c = r["code"]
        r["model_score"] = scores.get(c)
        r["model_version"] = meta.get("version", "")
        r["contribs"] = contribs.get(c, "")
        if c in scores:
            r["score100"] = round(100.0 * (scores[c] - lo) / span, 1)
    return rows


def model_rank_key():
    """select 的模型主排 key 工厂：分高优先，启发式键兜底保确定性。"""
    def key(r):
        s = r.get("model_score")
        return (-(s if s is not None else float("-inf")), sort_key(r))
    return key


# ---------------------------------------------------------------- 每日微调

def refit_today(today: dt.date | None = None) -> str:
    """每日微调（幂等）：xgb warm-start 5 棵小步树；logreg 全量重算+近期加权。

    与三日门的关系：refit 更新 refit_at 与工件，**不动 trained_at**——
    三日重训节奏不被每日微调吞掉（对齐 ziyang cutoff/trained_at 分离）。
    """
    today = today or dt.date.today()
    meta = _meta_load()
    if not (meta and meta.get("ok")):
        return "微调跳过（无可用模型）"
    if (meta.get("refit_at") or "")[:10] == f"{today:%Y-%m-%d}":
        return "微调跳过（今日已做，幂等守卫）"
    samples, days, _miss = build_samples(today)
    recent = [s for s in samples
              if s["date"] >= f"{(days[-6] if len(days) >= 6 else days[0]):%Y-%m-%d}"]
    if len(recent) < 10:
        return f"微调跳过（近窗样本仅 {len(recent)} 行）"
    X, _ = _matrix(recent, medians=meta.get("medians"))
    y = np.array([s["y"] for s in recent], dtype=int)
    if meta.get("kind") == "xgb":
        import xgboost as xgb
        booster = _load_booster(C.MODEL_XGB_JSON)
        # 显式传参（update 的 iteration 是轮次编号——2026-11-20 教训）；
        # 续训必须喂训练空间的标准化特征（同 _predict_proba）
        booster = xgb.train({"objective": "binary:logistic", "eval_metric":
                             "logloss", "max_depth": 3, "learning_rate": 0.005,
                             "verbosity": 0},
                            xgb.DMatrix(_zspace(X, meta), label=y),
                            num_boost_round=5, xgb_model=booster)
        fstore.write_with_bak(C.MODEL_XGB_JSON, booster.save_model)
        _BOOTER_CACHE.pop(C.MODEL_XGB_JSON, None)   # 失效进程缓存
        note = "xgb warm-start 5 棵"
    else:
        # logreg 无 warm-start：全量重算即微调（近期半衰加权已在 train 内）
        fresh = train(today)
        if not fresh.get("ok"):
            return f"微调失败（{fresh.get('reason')}），保留旧件"
        note = f"logreg 重算（{fresh['n_rows']} 行）"
    meta["refit_at"] = f"{today:%Y-%m-%d} {dt.datetime.now():%H:%M:%S}"
    meta["finetuned"] = True
    fstore.write_text(C.MODEL_JSON, json.dumps(meta, ensure_ascii=False, indent=1))
    return f"每日微调完成（{note}）"


# ---------------------------------------------------------------- 新旧对照

def retrain_diff(rows: list[dict], today: dt.date | None = None) -> dict | None:
    """重训后首个预判日：旧模型对当日候选重打分，输出新旧 top3 对照（幂等）。"""
    today = today or dt.date.today()
    meta = _meta_load()
    if not (meta and meta.get("prev_version")):
        return None
    diff_path = C.RETRAIN_DIFF_JSON
    if os.path.exists(diff_path):
        try:
            old = _load_json(diff_path)
            if old.get("shown_for") == f"{today:%Y-%m-%d}":
                return old                       # 幂等：当日已呈现过
        except Exception:                      # noqa: BLE001
            pass
    prev_meta_path = C.MODEL_PREV_JSON
    if not os.path.exists(prev_meta_path):
        return None
    try:
        prev_meta = _load_json(prev_meta_path)
        prev_art = (_prev_xgb_path() if prev_meta.get("kind") == "xgb"
                    else C.MODEL_LGR_JSON + ".prev")
        if not os.path.exists(prev_art):
            return None
        days, _ = _window_days(today, C.WINDOW)
        pools, _miss = _load_pools(days)
        idx = len(days) - 1
        feats = [_feature_row(r["code"], pools[days[idx]][r["code"]], idx,
                              days, pools)
                 for r in rows if r["code"] in pools[days[idx]]]
        X, _ = _matrix(feats, medians=prev_meta.get("medians"))
        # 旧件按原始概率出分、不过保序校准：对照的是模型本身，不混入校准层
        L = _logit(_predict_proba(prev_meta, prev_art, X))
        order = [feats[i]["code"] for i in np.argsort(-L)]
        new_order = [r["code"] for r in sorted(
            rows, key=lambda r: -(r.get("model_score") or float("-inf")))
            if r.get("model_score") is not None]
        top = lambda seq: seq[:C.TOP_N]
        diff = {"prev_version": prev_meta.get("version"),
                "new_version": meta.get("version"),
                "shown_for": f"{today:%Y-%m-%d}",
                "prev_top3": top(order), "new_top3": top(new_order),
                "overlap_n": len(set(top(order)) & set(top(new_order)))}
        fstore.write_text(diff_path,
                          json.dumps(diff, ensure_ascii=False, indent=1))
        return diff
    except Exception as e:                     # noqa: BLE001 对照失败留痕
        print(f"[风向标] 新旧对照失败（{type(e).__name__}: {e}）")
        return None
