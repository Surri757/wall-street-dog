# -*- coding: utf-8 -*-
"""premium11 溢价影子模型（合同: out/research/vane/premium11/contract.json）。

定位（2026-11-25 冻结）：
- 只做影子（shadow）评分与新旧榜对照；**生产名额仍由旧晋级模型/启发式决定**，
  本模块不替换任何生产排序、不写旧晋级工件。
- 目标：T+1 全日 VWAP 买入 → T+2 09:30–09:50 卖出 VWAP 的扣费净收益（decimal）；
  proxy（T+1 开盘→T+2 开盘）仅 SCREENING，物理分列，不得计入部署门。

模型（合同 §models）：
- P1：标准化 12 特征 + old_promo_p 的 Ridge(alpha=10.0)，拟合已结算净收益；
- P2：按 signal_date 截面去均值后的 Ridge(alpha=10.0)，推理加回训练折收益均值。

零泄漏边界（合同 §validation）：
- 训练行只取 exit_date < 验证 signal_date 的已结算可成交记录（一日 embargo）；
- 中位数/标准化/模型参数仅训练折拟合，验证折透传；
- old_promo_p 必须是折内旧工件口径；缺失行不进训练，不用当前旧工件回溯打分；
- 不足 10 训练信号日 / 30 条已结算记录 → not_ready，不出模型分；
- minute20 与 approx_open 两层不混训；approx_open 工件不得标记 minute20。

失败隔离：本模块任何异常由 vane.daily_sync 外层捕获留痕，不阻断旧 vane 链路。
"""
from __future__ import annotations

import datetime as dt

import numpy as np

import fstore

from . import config as C
from .engine import sort_key


# ---------------------------------------------------------------- 数据层接缝

def _data_layer():
    """惰性接入执行数据代理的 vane.premium_data；未交付/导入失败返回 None。

    约定接口（对方文件，本模块不改）：
      record_signal(trade_date, rows)   # 保存 T 晚全宇宙快照（含旧版本/旧分）
      collect_due(today)                # 触发到期分时采集
      load_universe(trade_date)         # 读回某日宇宙行
      load_labels(layer)                # 读回已结算记录（特征列 + net_ret）
    """
    try:
        from . import premium_data as PD
        return PD
    except Exception:                      # noqa: BLE001 数据层未交付=影子降级
        return None


def _load_labels(layer: str):
    """返回 (records|None, status)。数据层缺失/接口缺失/异常均显式留痕。"""
    pd = _data_layer()
    if pd is None:
        return None, "no_data_layer"
    fn = getattr(pd, "load_labels", None)
    if fn is None:
        return None, "data_layer_interface_missing"
    try:
        records = list(fn(layer) or [])
    except Exception as e:                # noqa: BLE001
        return None, f"load_labels_error:{type(e).__name__}"
    return records, "ok"


# ---------------------------------------------------------------- 样本筛选

_TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
_FALSE_TOKENS = {"0", "false", "no", "n", ""}


def _truthy(v) -> bool:
    """宽松真值解析：兼容 bool / int / CSV 字符串 '1'/'true'/'0'/'false'/空串。"""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in _TRUE_TOKENS


def _explicit_false(v) -> bool:
    """显式假值（0/'0'/'false'/空串）；None 视为未知，不算显式假。"""
    if v is None:
        return False
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v == 0
    return str(v).strip().lower() in _FALSE_TOKENS


SETTLED_ENTRY = ("filled", "partial")
SETTLED_EXIT = ("exited",)  # 与 premium_validation.EXIT_EXITED 对齐


def _is_settled(r: dict) -> bool:
    """已结算可成交：入场成交、已退出、未决假、净收益有限非空、complete 非显式假。

    口径与 premium_validation.is_settled 对齐；exit_status 兼容历史 filled 写法。
    CSV 空串/'0'/'False' 按缺失/假值严格解析。
    """
    if _truthy(r.get("unresolved_flag")):
        return False
    if str(r.get("entry_status") or "").strip() not in SETTLED_ENTRY:
        return False
    if str(r.get("exit_status") or "").strip().lower() not in SETTLED_EXIT:
        return False
    nr = r.get("net_ret")
    if nr is None or str(nr).strip() == "":
        return False
    try:
        if not np.isfinite(float(nr)):
            return False
    except (TypeError, ValueError):
        return False
    if _explicit_false(r.get("complete")):
        return False
    return True


def _row_finite(r: dict) -> tuple[bool, str]:
    """训练行逐行校验：12 特征 + old_promo_p + net_ret 必须可转有限数值。"""
    for f in C.FEATURES:
        try:
            x = float(r.get(f))
        except (TypeError, ValueError):
            return False, f"feature_{f}_nonnumeric"
        if not np.isfinite(x):
            return False, f"feature_{f}_nonfinite"
    for f in ("old_promo_p", "net_ret"):
        try:
            x = float(r.get(f))
        except (TypeError, ValueError):
            return False, f"{f}_nonnumeric"
        if not np.isfinite(x):
            return False, f"{f}_nonfinite"
    return True, ""


def _clean_train(rows: list[dict]) -> tuple[list[dict], int, dict]:
    """剔除非有限/缺特征行；返回 (clean, dropped, reason_counts)。"""
    clean, dropped, reasons = [], 0, {}
    for r in rows:
        ok, why = _row_finite(r)
        if ok:
            clean.append(r)
        else:
            dropped += 1
            reasons[why] = reasons.get(why, 0) + 1
    return clean, dropped, reasons


# 训练只取全宇宙未配对行；执行臂（A/B1/B0/B2）复制行不进训练，避免重复计数。
ARM_TRAIN_OK = ("U", "")


def _train_subset(records: list[dict], signal_date: str, layer: str) -> list[dict]:
    """walk-forward 训练子集：同层、全宇宙行(arm=U/空)、已结算、
    exit_date < 验证 signal_date、旧概率可得。"""
    out = []
    for r in records:
        if r.get("layer") != layer:
            continue
        arm = str(r.get("arm") or "").strip()
        if arm not in ARM_TRAIN_OK:
            continue
        if not _is_settled(r):
            continue
        exit_d = str(r.get("exit_date") or "")
        if not exit_d or exit_d >= signal_date:      # embargo：退出日严格早于验证日
            continue
        if r.get("old_promo_p") is None:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------- 特征矩阵

def _matrix(rows: list[dict], medians: dict | None = None):
    """12 旧特征 + old_promo_p → (X, medians)。缺失填训练中位数（推理透传）。"""
    import pandas as pd

    from . import model as VM

    X12, med = VM._matrix(rows, medians=medians)
    col = pd.to_numeric(pd.Series([r.get("old_promo_p") for r in rows]),
                        errors="coerce")
    m = (medians or {}).get("old_promo_p")
    if m is None:
        m = float(col.median()) if col.notna().any() else 0.5
    med["old_promo_p"] = float(m)
    Xp = col.fillna(m).to_numpy(dtype=float).reshape(-1, 1)
    return np.hstack([X12, Xp]), med


# ---------------------------------------------------------------- 拟合

def fit(records: list[dict], signal_date, kind: str = "P1",
        layer: str = "minute20") -> dict:
    """在已结算训练子集上拟合 Ridge；样本门不过返回 ok=False（not_ready）。"""
    if layer not in C.PREMIUM_LAYERS:
        return {"ok": False, "status": "unknown_layer", "layer": layer,
                "kind": kind, "n_days": 0, "n_rows": 0}
    sd = signal_date.isoformat() if hasattr(signal_date, "isoformat") else str(signal_date)
    raw_train = _train_subset(records, sd, layer)
    train, n_dropped, drop_reasons = _clean_train(raw_train)
    days = sorted({str(r["signal_date"]) for r in train})
    if len(days) < C.PREMIUM_MIN_TRAIN_DAYS or len(train) < C.PREMIUM_MIN_SETTLED:
        return {"ok": False, "status": "insufficient_train", "layer": layer,
                "kind": kind, "n_days": len(days), "n_rows": len(train),
                "n_dropped": n_dropped, "drop_reasons": drop_reasons,
                "min_days": C.PREMIUM_MIN_TRAIN_DAYS,
                "min_rows": C.PREMIUM_MIN_SETTLED}
    X, med = _matrix(train)
    y = np.array([float(r["net_ret"]) for r in train], dtype=float)
    day_keys = [str(r["signal_date"]) for r in train]

    if kind == "P2":
        Xd = X.copy()
        yd = y.copy()
        by_day: dict[str, list[int]] = {}
        for i, d in enumerate(day_keys):
            by_day.setdefault(d, []).append(i)
        for idxs in by_day.values():
            Xd[idxs] -= X[idxs].mean(axis=0)
            yd[idxs] -= y[idxs].mean()
        y_mean = float(y.mean())
        Xfit, yfit = Xd, yd
    else:
        y_mean = 0.0
        Xfit, yfit = X, y

    mu = Xfit.mean(axis=0)
    sd = np.where(Xfit.std(axis=0) > 1e-9, Xfit.std(axis=0), 1.0)
    Xz = (Xfit - mu) / sd

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=C.PREMIUM_ALPHA, fit_intercept=True)
    ridge.fit(Xz, yfit)

    return {
        "version": C.PREMIUM_VERSION, "kind": kind, "layer": layer,
        "features": list(C.PREMIUM_FEATURES), "alpha": C.PREMIUM_ALPHA,
        "medians": med,
        "mu": [round(float(v), 6) for v in mu],
        "sd": [round(float(v), 6) for v in sd],
        "coef": [round(float(v), 6) for v in ridge.coef_],
        "intercept": round(float(ridge.intercept_), 6),
        "y_mean": round(y_mean, 6),
        "n_days": len(days), "n_rows": len(train),
        "n_dropped": n_dropped, "drop_reasons": drop_reasons,
        "train_days": days, "cutoff": days[-1],
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ok": True,
    }


# ---------------------------------------------------------------- 工件存取

def _artifact_ready(art) -> bool:
    return bool(art and art.get("ok") and art.get("version") == C.PREMIUM_VERSION
                and art.get("features") == list(C.PREMIUM_FEATURES)
                and len(art.get("coef") or []) == len(C.PREMIUM_FEATURES)
                and len(art.get("mu") or []) == len(C.PREMIUM_FEATURES)
                and len(art.get("sd") or []) == len(C.PREMIUM_FEATURES))


def save_artifact(art: dict) -> None:
    fstore.write_json(C.PREMIUM_MODEL_JSON, art)


def load_artifact() -> dict | None:
    art = fstore.read_json(C.PREMIUM_MODEL_JSON)
    return art if isinstance(art, dict) else None


# ---------------------------------------------------------------- 推理

def _sigmoid(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-p))


def _old_prob(r: dict):
    """旧晋级概率：优先 old_promo_p 字段；缺省由旧模型 logit(model_score) 还原。"""
    v = r.get("old_promo_p")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    ms = r.get("model_score")
    if ms is not None:
        try:
            return float(_sigmoid(float(ms)))
        except (TypeError, ValueError):
            return None
    return None


def score_rows(rows: list[dict], art: dict) -> list[dict]:
    """按工件给宇宙行打溢价分；工件不可用时 premium_score 全 None（不出模型分）。"""
    rows = [dict(r) for r in rows]
    if not _artifact_ready(art):
        for r in rows:
            r["premium_score"] = None
        return rows
    X, _ = _matrix(rows, medians=art.get("medians"))
    mu = np.array(art["mu"], dtype=float)
    sd = np.array(art["sd"], dtype=float)
    Xz = (X - mu) / sd
    pred = Xz @ np.array(art["coef"], dtype=float) + float(art["intercept"]) \
        + float(art.get("y_mean", 0.0))
    for r, v in zip(rows, pred):
        r["premium_score"] = float(v)
    return rows


# ---------------------------------------------------------------- 臂排名

def _qual(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r.get("B") or 0) in C.BOARD_RANGE]


def _pick(rank: int, r: dict, score_key: str) -> dict:
    return {"rank": rank, "code": r["code"], "name": r.get("name", ""),
            "B": r.get("B"), "score": r.get(score_key),
            "old_promo_p": _old_prob(r)}


def rank_arms(rows: list[dict], art: dict | None) -> dict:
    """A=新溢价模型；B1=旧晋级模型；B0=启发式；B2=同宇宙全体等权（只记票数）。

    工件不可用（not_ready/坏件）时 A_top3/A_picks 必须为空——不得按 code 兜底
    冒充有模型；B1/B0 影子对照照常出榜。
    """
    qual = _qual(rows)
    b1_key = lambda r: (-(r.get("model_score")
                          if r.get("model_score") is not None
                          else float("-inf")), r["code"])
    B1 = sorted(qual, key=b1_key)[:C.TOP_N]
    B0 = sorted(qual, key=sort_key)[:C.TOP_N]
    b1_codes = [r["code"] for r in B1]

    if not _artifact_ready(art):
        return {
            "A_top3": [], "B1_top3": b1_codes,
            "B0_top3": [r["code"] for r in B0],
            "B2_n": len(qual), "overlap_A_B1": 0,
            "A_picks": [],
            "B1_picks": [_pick(i, r, "model_score") for i, r in enumerate(B1, 1)],
            "B0_picks": [_pick(i, r, None) for i, r in enumerate(B0, 1)],
        }

    scored = score_rows(qual, art)
    a_key = lambda r: (-(r.get("premium_score")
                         if r.get("premium_score") is not None
                         else float("-inf")), r["code"])
    A = sorted(scored, key=a_key)[:C.TOP_N]
    a_codes = [r["code"] for r in A]
    return {
        "A_top3": a_codes, "B1_top3": b1_codes,
        "B0_top3": [r["code"] for r in B0],
        "B2_n": len(qual),
        "overlap_A_B1": len(set(a_codes) & set(b1_codes)),
        "A_picks": [_pick(i, r, "premium_score") for i, r in enumerate(A, 1)],
        "B1_picks": [_pick(i, r, "model_score") for i, r in enumerate(B1, 1)],
        "B0_picks": [_pick(i, r, None) for i, r in enumerate(B0, 1)],
    }


def _enrich_today_features(trade_date, rows: list[dict]) -> list[dict]:
    """把今日宇宙行补齐 12 特征列（复用旧模型池视图口径）；池管道失败用行内字段兜底。"""
    out = [dict(r) for r in rows]
    try:
        from . import model as VM
        days, _ = VM._window_days(trade_date, C.WINDOW)
        pools, _ = VM._load_pools(days)
        view = VM._pool_view(days, pools)
        idx = len(days) - 1
        today_pool = pools.get(days[idx], {})
        by = {r["code"]: r for r in out}
        for code, prow in today_pool.items():
            if code not in by:
                continue
            feat = VM._feature_row(code, prow, idx, days, pools, view)
            by[code].update({k: feat.get(k) for k in C.FEATURES})
        return out
    except Exception:                      # noqa: BLE001 特征管道失败：中位数兜底
        return out


# ---------------------------------------------------------------- 影子台账

SHADOW_COLS = ["signal_date", "arm", "rank", "code", "name", "B",
               "model_version", "model_score", "premium_score", "old_promo_p",
               "layer", "status"]


def _shadow_rows(ledger: list[dict]) -> list[dict]:
    """从完整 JSON 台账重建全历史逐臂 picks CSV（幂等；当日覆写不影响历史行）。"""
    rows_out = []
    for rec in ledger:
        for arm, picks in (("A", rec.get("A_picks") or []),
                           ("B1", rec.get("B1_picks") or []),
                           ("B0", rec.get("B0_picks") or [])):
            for p in picks:
                rows_out.append({
                    "signal_date": rec["signal_date"], "arm": arm,
                    "rank": p["rank"], "code": p["code"], "name": p["name"],
                    "B": p.get("B"),
                    "model_version": rec.get("model_version", ""),
                    "model_score": p.get("score") if arm != "A" else None,
                    "premium_score": p.get("score") if arm == "A" else None,
                    "old_promo_p": p.get("old_promo_p"),
                    "layer": rec.get("layer", ""), "status": rec.get("status", ""),
                })
    return rows_out


def _append_shadow(record: dict) -> None:
    """JSON 主台账（按 signal_date 幂等覆写当日）+ 由全量台账重建 CSV 历史。"""
    ledger = fstore.read_json(C.PREMIUM_SHADOW_JSON, default=[])
    if not isinstance(ledger, list):
        ledger = []
    ledger = [r for r in ledger if r.get("signal_date") != record["signal_date"]]
    ledger.append(record)
    fstore.write_json(C.PREMIUM_SHADOW_JSON, ledger)
    fstore.write_csv(C.PREMIUM_SHADOW_CSV, SHADOW_COLS, _shadow_rows(ledger),
                     quoting=fstore.csv.QUOTE_NONNUMERIC)


# ---------------------------------------------------------------- 每日影子入口

def daily_shadow(trade_date, rows: list[dict],
                 old_meta: dict | None = None) -> dict:
    """每日影子：存宇宙快照 → 触发到期采集 → 选层 → 拟合/复用 → 三臂对照落盘。

    任何环节失败都降级留痕；返回对照 record（不阻断旧 vane 主排）。
    """
    sd = trade_date.isoformat() if hasattr(trade_date, "isoformat") else str(trade_date)
    notes: list[str] = []
    pd = _data_layer()

    # 1) 全宇宙快照 + 到期采集（失败留痕，不阻断）
    if pd is not None:
        old_ver = (old_meta or {}).get("version") or ""
        try:
            if hasattr(pd, "record_signal"):
                pd.record_signal(trade_date, rows, old_model_version=old_ver)
        except Exception as e:              # noqa: BLE001
            notes.append(f"record_signal:{type(e).__name__}")
        try:
            if hasattr(pd, "collect_due"):
                pd.collect_due(trade_date)
        except Exception as e:              # noqa: BLE001
            notes.append(f"collect_due:{type(e).__name__}")
    else:
        notes.append("no_data_layer")

    # 2) 选层：minute20 优先；无 minute20 已结算记录才退 approx_open（仅 SCREENING）
    records, st = _load_labels("minute20")
    layer = "minute20"
    if records is None or not records:
        records, st2 = _load_labels("approx_open")
        layer = "approx_open"
        if records is None:
            notes.append(st2)
    if records is None:
        records, status = [], st
    else:
        status = "ok"

    # 3) 拟合（walk-forward，训练行 exit_date < 今日）；门不过则尝试兼容旧件
    art = None
    if records:
        fit_res = fit(records, sd, kind="P1", layer=layer)
        if fit_res.get("ok"):
            art = fit_res
            try:
                save_artifact(art)
            except Exception as e:              # noqa: BLE001
                notes.append(f"artifact_save:{type(e).__name__}")
            status = "ready"
        else:
            status = fit_res.get("status", status)
            old = load_artifact()
            if _artifact_ready(old) and old.get("layer") == layer:
                art = old
                notes.append("reuse_artifact")
            else:
                art = None
                if old is not None:
                    notes.append("artifact_incompatible")
    else:
        status = st if st != "ok" else "no_records"

    # 4) 臂排名（A 无工件时 premium_score 全 None，A_top3 为空，不冒充有模型）
    enriched = _enrich_today_features(trade_date, rows)
    arms = rank_arms(enriched, art)

    screening = layer == "approx_open"
    if screening:
        notes.append("SCREENING：proxy(open-open) 层仅筛选对照，不得计入部署门/切生产")

    record = {
        "signal_date": sd,
        "layer": layer if records else "none",
        "screening": screening,
        "status": status,
        "model_version": art.get("version", "") if art else "",
        "model_kind": art.get("kind", "") if art else "",
        "n_train_days": art.get("n_days", 0) if art else 0,
        "n_train_rows": art.get("n_rows", 0) if art else 0,
        **arms,
        "note": "｜".join(notes),
        "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _append_shadow(record)

    # 5) 影子台账落盘后：从 shadow_daily 生成 A/B1/B0/B2 执行配对层（数据代理
    #    可选接口）；失败只记 note 回写台账，不阻断旧榜。
    if pd is not None:
        sync = getattr(pd, "sync_forward_ledger", None)
        if callable(sync):
            try:
                sync()
            except Exception as e:              # noqa: BLE001
                record["note"] = "｜".join(
                    [record.get("note") or "",
                     f"sync_forward_ledger:{type(e).__name__}"])
                try:
                    _append_shadow(record)
                except Exception:              # noqa: BLE001
                    pass
    return record
