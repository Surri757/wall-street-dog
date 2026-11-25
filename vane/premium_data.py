# -*- coding: utf-8 -*-
"""premium11 执行数据底座：真实 VWAP 标签原料 + 前向采集（数据层，不训练模型）。

冻结合同：``out/research/vane/premium11/contract.json``（vane-premium11-2026-11-25）。

口径要点（与合同逐条对齐）：
  - 宇宙：主板非 ST 封板、本轮板数 B∈{4,5}，沿用 vane 现有宇宙，不改 B 口径；
    ``record_signal`` 保存**全宇宙**及旧模型版本/分数，不只 Top3。
  - 买入价：T+1 全日市场成交 VWAP = 当日累计成交额 /（当日累计成交量手 × 100）；
    买入窗冻结 full_day，不得改成 09:30–09:50。
  - 卖出价：实际退出日 09:30–09:50 窗口 VWAP（含 0930 竞价点与 0950 点，
    21 个时间标签；窗口量额以 0 为基准到 0950 累计值）。
  - 净收益：``sell_vwap/buy_vwap - 1 - config.COST_ROUND_TRIP``；费用字段全部复用
    config 单一事实源，本模块不另写成本字面量。
  - 状态：entry_status ∈ filled/unfilled/partial/nodata/fail；
    exit_status ∈ exited/pending/carry/nodata/fail；未成交净收益为空，组合现金权重另算；
    跌停/零成交/停牌卖不出 → pending/carry 顺延，未决头寸不删除、不按 0 收益伪造。
  - 无逐笔/排单数据：成交归属标 ``market_vwap_shadow``（市场 VWAP 参考成交价），
    不得宣称个人账户真实成交率。
  - proxy：T+1 开盘 → T+2 开盘的日 K open-open 扣费收益，``layer=approx_open``，
    独立字段、独立层，只用于筛选，绝不计入 minute20 样本门/部署门。

落盘纪律（对齐 auction 子包）：
  - 按日分片、原子写（fstore tmp+replace）；截断/失败不写伪完整分钟原料；
  - 跨日已冻结的宇宙/台账不得静默覆写；当日重跑允许替换；
  - 网络故障只留 fail/nodata 状态与待采记录，不产生伪完整标签。

公开接口（供模型/引擎代理调用）：
  - ``record_signal(trade_date, rows, ...)``  保存 B=4/5 全宇宙快照
  - ``collect_due(today=None, calendar=None, ...)``  采集到期 T+1 全日 + T+2/顺延日首 20 分钟
  - ``load_universe(trade_date)``  读取某日全宇宙行
  - ``load_labels(layer=None)``  读取标签台账，可按 layer 过滤
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import fstore
from config import (COST_COMMISSION, COST_SLIPPAGE, COST_STAMP_TAX,
                    COST_TRANSFER, COST_ROUND_TRIP, LIMIT_NORMAL)
from trade_rules import LIMIT_EPS, limit_down_price, limit_price

from . import config as VC

ROOT = Path(__file__).resolve().parent.parent
PREMIUM_DIR = ROOT / "out" / "research" / "vane" / "premium11"
UNIVERSE_DIR = PREMIUM_DIR / "universe"
MINUTE_DIR = PREMIUM_DIR / "minute"
LABELS_CSV = PREMIUM_DIR / "labels.csv"
AUDIT_JSON = PREMIUM_DIR / "data_audit.json"
AUDIT_LOG = PREMIUM_DIR / "data_audit.log"
SHADOW_JSON = PREMIUM_DIR / "shadow_daily.json"
LEDGER_JSON = PREMIUM_DIR / "forward_ledger.json"

# 训练/结果层统一臂：labels.csv 每 signal+code+layer 仅一条，避免按臂重复训练。
UNIFIED_ARM = "U"
# 执行台账（forward_ledger.json）派生臂：Top3 臂 + 全体等权基线 B2。
B2_ARM = "B2"
TOP3_ARMS = ("A", "B1", "B0")
# 影子 picks 字段 → 对应臂在台账中落盘的得分字段。
ARM_SCORE_FIELD = {"A": "premium_score", "B1": "model_score", "B0": "model_score"}

TZ = "Asia/Shanghai"
ADJ = "raw"
DATA_SOURCE = "tx-minute"
EXIT_START = "0930"
EXIT_END = "0950"
EXIT_LABELS = 21                 # 0930…0950 含端点 21 个标签
MARKET_SHADOW = "market_vwap_shadow"
TOP_N = VC.TOP_N

# 合同 forward_fields（顺序即 CSV 列序）；追加 12 个 T 收盘可知特征（B 已在列）
# 与 proxy 开盘价字段。特征在结算时从 row['raw'] 复制到标签顶层，供模型训练直取。
FEATURE_FIELDS = [f for f in VC.FEATURES if f != "B"]
PROXY_OPEN_FIELDS = ["exec_open", "exit_open"]
LABEL_FIELDS = [
    "signal_date", "exec_date", "exit_date", "arm", "rank", "code", "name", "B",
    "model_version", "model_score", "premium_score", "old_model_version", "old_promo_p",
    "layer", "entry_status", "filled_ratio", "buy_vwap", "exit_status",
    "sell_vwap_20m", "gross_ret", "net_ret", "weight", "cash_weight", "carry_flag",
    "unresolved_flag", "data_source", "fetched_at", "tz", "adj", "complete", "note",
] + FEATURE_FIELDS + PROXY_OPEN_FIELDS

# 分钟完整日阈值（复用上游常量，不在本模块另立刻度）
try:
    from datasources.auction import COMPLETE_POINTS
except ImportError:                 # pragma: no cover - 保底
    COMPLETE_POINTS = 240


# ---------------------------------------------------------------- 纯工具

def _f(v):
    """宽松转 float；NaN/非法/None → None。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and math.isfinite(x) else None


def _iso(d) -> str:
    if isinstance(d, dt.date):
        return d.isoformat()
    return str(d)[:10]


def _ymd(d) -> str:
    return f"{dt.date.fromisoformat(_iso(d)):%Y%m%d}"


def _today() -> dt.date:
    return dt.date.today()


def _now() -> str:
    return f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"


def _next_td(cal: list[dt.date], d: dt.date) -> dt.date | None:
    later = [x for x in cal if x > d]
    return later[0] if later else None


# ---------------------------------------------------------------- 分钟校验/VWAP

def validate_minute_points(points: list) -> dict:
    """校验累计量(手)/额(元)分钟序列。

    返回：ok/complete/n_points/monotone/duplicates/negative_inc/zero_volume/
    issues/first/last。``complete`` 要求点数达标且校验通过；截断/坏点不得当完整日。
    """
    issues: list[str] = []
    if not points:
        return {"ok": False, "complete": False, "n_points": 0, "monotone": False,
                "duplicates": [], "negative_inc": [], "zero_volume": True,
                "issues": ["empty_points"], "first": None, "last": None}

    hhmm = [str(p[0]) for p in points]
    duplicates = sorted({h for h in hhmm if hhmm.count(h) > 1})
    if duplicates:
        issues.append(f"duplicate_hhmm:{','.join(duplicates)}")

    sorted_ok = hhmm == sorted(hhmm)
    if not sorted_ok:
        issues.append("hhmm_not_sorted")

    prev_vol = prev_amt = 0.0
    negative_inc: list[str] = []
    bad_price = 0
    monotone = True
    for p in points:
        h, price, vol, amt = str(p[0]), _f(p[1]), _f(p[2]), _f(p[3])
        if price is None or price <= 0:
            bad_price += 1
            continue
        if vol is None or amt is None or vol < 0 or amt < 0:
            monotone = False
            negative_inc.append(h)
            continue
        if vol < prev_vol - 1e-9 or amt < prev_amt - 1e-9:
            monotone = False
            negative_inc.append(h)
        prev_vol, prev_amt = vol, amt
    if bad_price:
        issues.append(f"bad_price_points:{bad_price}")
    if not monotone:
        issues.append("non_monotone_volume_or_amount")

    last_vol = _f(points[-1][2]) or 0.0
    zero_volume = last_vol <= 0
    if zero_volume:
        issues.append("zero_total_volume")

    ok = not duplicates and sorted_ok and monotone and bad_price == 0
    complete = ok and len(points) >= COMPLETE_POINTS and not zero_volume
    return {
        "ok": ok, "complete": complete, "n_points": len(points),
        "monotone": monotone, "duplicates": duplicates,
        "negative_inc": negative_inc, "zero_volume": zero_volume,
        "issues": issues, "first": tuple(points[0]), "last": tuple(points[-1]),
    }


def full_day_vwap(points: list) -> float | None:
    """全日 VWAP = 末点累计额 /（末点累计量手 × 100）。无正总量 → None。"""
    if not points:
        return None
    _, _, vol, amt = points[-1]
    vol, amt = _f(vol), _f(amt)
    if not vol or vol <= 0 or amt is None:
        return None
    return amt / (vol * 100.0)


def _point_at(points: list, hhmm: str):
    for p in points:
        if str(p[0]) == hhmm:
            return p
    return None


def window_vwap(points: list, start: str = EXIT_START, end: str = EXIT_END,
                baseline_zero: bool = True) -> dict:
    """窗口 VWAP。默认 exit 口径：0930 前基准为 0，到 0950 累计额/量（含竞价点）。

    baseline_zero=False 时用 start 前一点作基准（通用增量窗口，供其他窗对照）。
    """
    res = {"vwap": None, "vol": None, "amt": None, "n_labels": 0,
           "missing_end": False, "zero_volume": True, "issues": []}
    if not points:
        res["issues"].append("empty_points")
        return res
    end_p = _point_at(points, end)
    if end_p is None:
        res["missing_end"] = True
        res["issues"].append(f"missing_label:{end}")
        return res
    end_vol, end_amt = _f(end_p[2]), _f(end_p[3])
    if baseline_zero:
        base_vol, base_amt = 0.0, 0.0
    else:
        start_p = _point_at(points, start)
        if start_p is None:
            res["issues"].append(f"missing_label:{start}")
            base_vol = base_amt = 0.0
        else:
            base_vol, base_amt = _f(start_p[2]) or 0.0, _f(start_p[3]) or 0.0
    dvol = (end_vol or 0.0) - base_vol
    damt = (end_amt or 0.0) - base_amt
    res["vol"], res["amt"] = dvol, damt
    window = [p for p in points if start <= str(p[0]) <= end]
    res["n_labels"] = len(window)
    if dvol <= 0:
        res["zero_volume"] = True
        res["issues"].append("zero_window_volume")
        return res
    res["zero_volume"] = False
    res["vwap"] = damt / (dvol * 100.0)
    if res["n_labels"] != EXIT_LABELS:
        res["issues"].append(f"window_labels:{res['n_labels']}!={EXIT_LABELS}")
    return res


def detect_one_word_open(points: list, prev_close: float | None) -> bool:
    """T+1 开盘一字涨停（首点价 ≥ 涨停价−容差）。prev_close 缺失 → False（不臆断）。"""
    if not points or prev_close is None or prev_close <= 0:
        return False
    limit = limit_price(prev_close, LIMIT_NORMAL)
    open_px = _f(points[0][1])
    return bool(open_px is not None and open_px >= limit - limit * LIMIT_EPS)


def detect_exit_blocked(exit_points: list, prev_close: float | None) -> dict:
    """退出日窗口是否卖不出：跌停锁死 / 窗口零成交 / 停牌无点。

    零成交判定用 0950 末点累计量/额增量（合同口径：窗口基准 0 → 0950 累计值），
    **不对累计量跨点求和**（累计量求和会重复计量）。
    返回 {blocked, reason, limit_down}；无 prev_close 时只判零成交/无点。
    """
    out = {"blocked": False, "reason": None, "limit_down": None}
    if not exit_points:
        out["blocked"], out["reason"] = True, "suspension"
        return out
    win = [p for p in exit_points if EXIT_START <= str(p[0]) <= EXIT_END]
    if not win:
        out["blocked"], out["reason"] = True, "suspension"
        return out
    end_p = _point_at(exit_points, EXIT_END)
    if end_p is None:
        out["blocked"], out["reason"] = True, "suspension"
        out["issues"] = f"missing_label:{EXIT_END}"
        return out
    end_vol, end_amt = _f(end_p[2]), _f(end_p[3])
    if (end_vol or 0.0) <= 0 or (end_amt or 0.0) <= 0:
        out["blocked"], out["reason"] = True, "zero_trade"
        return out
    if prev_close and prev_close > 0:
        ld = limit_down_price(prev_close, LIMIT_NORMAL)
        out["limit_down"] = ld
        if all((_f(p[1]) or ld) <= ld + ld * LIMIT_EPS for p in win):
            out["blocked"], out["reason"] = True, "limit_down"
    return out


# ---------------------------------------------------------------- 宇宙快照（record_signal）

def _universe_path(trade_date) -> Path:
    return Path(UNIVERSE_DIR) / f"{_ymd(trade_date)}.json"


def _normalize_row(row: dict) -> dict:
    code = str(row.get("code") or "").zfill(6)
    b = row.get("B")
    try:
        b = int(b)
    except (TypeError, ValueError):
        b = None
    rank = row.get("rank")
    try:
        rank = int(rank) if rank not in (None, "") else None
    except (TypeError, ValueError):
        rank = None
    return {
        "code": code, "name": str(row.get("name") or ""), "B": b,
        "group": str(row.get("group") or ""), "rank": rank,
        "arm": str(row.get("arm") or "A"),
        "model_version": row.get("model_version") or "",
        "model_score": _f(row.get("model_score")),
        "old_model_version": row.get("old_model_version") or "",
        "old_promo_p": _f(row.get("old_promo_p")),
        "raw": {k: v for k, v in row.items() if k not in ("raw",)},
    }


def record_signal(trade_date, rows: list[dict],
                   old_model_version: str | None = None,
                   source: str = "forward",
                   fetched_at: str | None = None,
                   allow_regen_same_day: bool = True,
                   now: dt.date | None = None) -> dict:
    """保存 T 日 B∈{4,5} 全宇宙快照（含旧模型版本/分数），不只 Top3。

    纪律：当日（本地日期，`now` 为测试接缝）允许重跑替换；跨日已冻结快照
    拒绝静默覆写。返回 {"status": recorded|frozen|skipped, ...}。
    """
    td = dt.date.fromisoformat(_iso(trade_date))
    today = now or _today()
    path = _universe_path(td)

    existing = None
    if path.exists():
        existing = fstore.read_json(str(path))

    if existing and not (allow_regen_same_day and td == today):
        return {"status": "frozen", "signal_date": td.isoformat(),
                "n_rows": len(existing.get("rows", [])),
                "path": str(path), "note": "跨日快照已冻结，拒绝静默覆写"}

    kept, dropped = [], []
    for r in rows or []:
        n = _normalize_row(r)
        if n["B"] in VC.BOARD_RANGE:
            kept.append(n)
        else:
            dropped.append(n["code"])

    snapshot = {
        "signal_date": td.isoformat(),
        "source": source,
        "fetched_at": fetched_at or _now(),
        "tz": TZ,
        "adj": ADJ,
        "complete": True,
        "old_model_version": old_model_version or (existing or {}).get("old_model_version") or "",
        "n_rows": len(kept),
        "dropped_out_of_range": dropped,
        "rows": kept,
    }
    fstore.write_json(str(path), snapshot)
    return {"status": "recorded", "signal_date": td.isoformat(),
            "n_rows": len(kept), "dropped": len(dropped), "path": str(path)}


def load_universe(trade_date) -> list[dict]:
    """读取某日全宇宙行（无快照 → []）。"""
    path = _universe_path(trade_date)
    if not path.exists():
        return []
    snap = fstore.read_json(str(path))
    return list((snap or {}).get("rows", []))


def _list_universe_snapshots() -> list[dict]:
    udir = Path(UNIVERSE_DIR)
    if not udir.is_dir():
        return []
    out = []
    for p in sorted(udir.glob("*.json")):
        snap = fstore.read_json(str(p))
        if snap:
            out.append(snap)
    return out


# ---------------------------------------------------------------- 标签台账

def _load_labels() -> list[dict]:
    if not Path(LABELS_CSV).exists():
        return []
    with open(LABELS_CSV, encoding="utf-8-sig", newline="") as f:
        return [{c: r.get(c, "") for c in LABEL_FIELDS} for r in csv.DictReader(f)]


def _save_labels(rows: list[dict]) -> None:
    clean = []
    for r in rows:
        clean.append({k: ("" if r.get(k) is None else r.get(k)) for k in LABEL_FIELDS})
    fstore.write_csv(str(LABELS_CSV), LABEL_FIELDS, clean,
                     quoting=csv.QUOTE_NONNUMERIC)


def _label_key(r: dict) -> tuple[str, str, str]:
    # 训练层幂等键：信号日 + 代码 + 层；臂在训练层统一 U，执行臂由台账派生。
    return (r.get("signal_date", ""), r.get("code", ""),
            r.get("layer", "minute20"))


def load_labels(layer: str | None = None) -> list[dict]:
    """读取训练/结果层标签（仅统一臂 U）；layer ∈ minute20 / approx_open / None。"""
    rows = [r for r in _load_labels() if str(r.get("arm") or UNIFIED_ARM) == UNIFIED_ARM]
    if layer:
        rows = [r for r in rows if r.get("layer") == layer]
    return rows


# ---------------------------------------------------------------- 执行台账（forward ledger）

def _read_shadow(shadow_path: str | Path | None = None) -> list[dict]:
    """读取影子日榜 shadow_daily.json；缺失/坏件返回 []（不伪造 picks）。"""
    p = Path(shadow_path) if shadow_path else Path(SHADOW_JSON)
    if not p.exists():
        return []
    data = fstore.read_json(str(p))
    return data if isinstance(data, list) else []


def _ledger_record(u: dict, arm: str, rank, weight: float,
                   score_field: str | None, score_value,
                   model_version: str | None = None) -> dict:
    """由训练层 U 行派生一条执行台账记录；状态/结果/特征原样保留。"""
    rec = {k: u.get(k) for k in LABEL_FIELDS}
    rec["arm"] = arm
    rec["rank"] = "" if rank is None else rank
    rec["weight"] = weight
    if score_field:
        rec[score_field] = score_value
    if model_version is not None:
        rec["model_version"] = model_version
    return rec


def sync_forward_ledger(shadow_path: str | Path | None = None,
                        ledger_path: str | Path | None = None) -> dict:
    """由 labels.csv 的 U 行 + shadow_daily.json 派生执行臂台账（幂等重建）。

    - A/B1/B0：影子 picks 内的 Top3 票，写 arm/rank/weight=1/3，臂得分落对应字段；
      某日 A 榜为空则该日无 A 行，禁止按 code 兜底伪造。T 晚一旦入选，即使 T+1
      一字买不进，臂行也必须保留：entry/exit=unfilled、net_ret 空、weight/cash_weight
      =1/3、unresolved=0；该席留现贡献 0，剩余仓位不重分配、行不删除。
    - B2：当日全宇宙 U 行，arm=B2，weight=0 占位，由验证器按实际可成交等权。
    - 未决状态原样保留；不回写、不覆盖 labels.csv 的 minute20/proxy 行。
    """
    u_rows = [r for r in _load_labels() if str(r.get("arm") or UNIFIED_ARM) == UNIFIED_ARM]
    shadow = _read_shadow(shadow_path)
    u_idx = {(r.get("signal_date"), r.get("layer", "minute20"), r.get("code")): r
             for r in u_rows}
    seat = round(1.0 / TOP_N, 6)

    records: list[dict] = []

    # B2：全宇宙基线（来自训练层，不依赖影子榜）
    for u in u_rows:
        records.append(_ledger_record(u, B2_ARM, None, 0.0, None, None))

    # Top3 臂：来自影子 picks；U 行缺失则跳过，不伪造执行席位
    pick_key = {"A": "A_picks", "B1": "B1_picks", "B0": "B0_picks"}
    for rec in shadow:
        sd = rec.get("signal_date")
        layer = rec.get("layer") or "minute20"
        if not sd or layer == "none":
            continue
        model_version = rec.get("model_version", "")
        for arm in TOP3_ARMS:
            picks = rec.get(pick_key[arm]) or []
            for p in picks:
                code = p.get("code")
                u = u_idx.get((sd, layer, code))
                if not u:
                    continue
                rec_row = _ledger_record(
                    u, arm, p.get("rank"), seat,
                    ARM_SCORE_FIELD[arm], p.get("score"), model_version)
                # 一字买不进的入选票：保留臂行，席位留现（合同 unfilled 口径）
                if str(u.get("entry_status")) == "unfilled":
                    rec_row["weight"] = seat
                    rec_row["cash_weight"] = seat
                    rec_row["filled_ratio"] = 0.0
                    rec_row["unresolved_flag"] = 0
                    rec_row["net_ret"] = None
                records.append(rec_row)

    # 幂等：按 (信号日, 代码, 臂, 层) 去重，确定性重建
    by: dict[tuple, dict] = {}
    for r in records:
        by[(r.get("signal_date"), r.get("code"), r.get("arm"), r.get("layer"))] = r
    records = list(by.values())

    payload = {"generated_at": _now(), "n_u_rows": len(u_rows),
               "n_shadow_days": len(shadow), "records": records}
    fstore.write_json(str(ledger_path or LEDGER_JSON), payload)

    counts: dict[str, int] = {}
    for r in records:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    return {"status": "ok", "records": len(records), "by_arm": counts,
            "n_u_rows": len(u_rows), "n_shadow_days": len(shadow)}


def load_forward(layer: str | None = None) -> list[dict]:
    """读取配对执行层台账（forward_ledger.json）；layer ∈ minute20 / approx_open / None。"""
    p = Path(LEDGER_JSON)
    if not p.exists():
        return []
    data = fstore.read_json(str(p))
    rows = list((data or {}).get("records", []))
    if layer:
        rows = [r for r in rows if r.get("layer") == layer]
    return rows


# ---------------------------------------------------------------- 分钟原料落盘

def _minute_path(day: dt.date, code: str) -> Path:
    return Path(MINUTE_DIR) / f"{_ymd(day)}" / f"{code}.pkl"


def _write_minute(day: dt.date, code: str, role: str, points: list,
                  validation: dict, data_source: str = DATA_SOURCE) -> None:
    payload = {
        "code": code, "date": day.isoformat(), "role": role,
        "points": [list(p) for p in points],
        "complete": bool(validation.get("complete")),
        "validation": validation,
        "data_source": data_source, "fetched_at": _now(), "tz": TZ, "adj": ADJ,
    }
    fstore.write_pkl(str(_minute_path(day, code)), payload)


def _read_minute(day: dt.date, code: str) -> dict | None:
    import pickle
    p = _minute_path(day, code)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:                                   # noqa: BLE001
        return None


# ---------------------------------------------------------------- 标签结算

def _slot_weight(arm: str, rank: int | None) -> float:
    """席位权重：Top3 臂每席 1/3；B2 全体等权（结算时按可成交数，占位 0）。"""
    if arm == "B2":
        return 0.0
    return round(1.0 / TOP_N, 6) if rank is not None and rank <= TOP_N else 0.0


def settle_minute_label(row: dict, exec_points: list | None,
                        exit_points: list | None, prev_close_t: float | None,
                        exit_prev_close: float | None) -> dict:
    """根据已落盘分钟原料结算一行 minute20 标签（纯函数，不写盘）。

    row: 宇宙行归一化 dict。返回标签行 dict（未决/失败字段为空，不伪造收益）。
    特征列从 row['raw'] 的 T 收盘可知字段复制到顶层，供模型训练直取。
    """
    arm, rank, code = row["arm"], row["rank"], row["code"]
    raw = row.get("raw") or {}
    base: dict = {
        "signal_date": row.get("signal_date", ""), "exec_date": "", "exit_date": "",
        # 训练/结果层统一臂 U；执行臂由 forward_ledger 按影子 picks 派生。
        "arm": UNIFIED_ARM, "rank": rank if rank is not None else "", "code": code,
        "name": row["name"], "B": row["B"],
        "model_version": row.get("model_version", ""),
        "model_score": row.get("model_score"),
        "premium_score": raw.get("premium_score", row.get("premium_score")),
        "old_model_version": row.get("old_model_version", ""),
        "old_promo_p": row.get("old_promo_p"),
        "layer": "minute20", "entry_status": "nodata", "filled_ratio": None,
        "buy_vwap": None, "exit_status": "nodata", "sell_vwap_20m": None,
        "gross_ret": None, "net_ret": None,
        "weight": _slot_weight(arm, rank), "cash_weight": 1.0,
        "carry_flag": 0, "unresolved_flag": 1, "data_source": DATA_SOURCE,
        "fetched_at": _now(), "tz": TZ, "adj": ADJ, "complete": False, "note": "",
        "exec_open": None, "exit_open": None,
    }
    for f in VC.FEATURES:
        if f == "B":
            continue
        base[f] = raw.get(f, row.get(f))

    # ── 买入日（T+1 全日）──
    if exec_points is None:
        base["note"] = "T+1 分钟原料缺失（nodata/fail 待采）"
        return base
    v_exec = validate_minute_points(exec_points)
    if not v_exec["complete"]:
        base["entry_status"] = "nodata"
        base["note"] = f"T+1 分钟截断/未完整：{';'.join(v_exec['issues'])}"
        return base

    if detect_one_word_open(exec_points, prev_close_t):
        # 一字涨停买不进：已处理现金席——unresolved=0，per-trade net_ret 为空，
        # 组合层该席现金贡献 0；名额不重分配、行不删除。
        base["entry_status"] = "unfilled"
        base["filled_ratio"] = 0.0
        base["exit_status"] = "unfilled"
        base["cash_weight"] = base["weight"] if base["weight"] > 0 else 1.0 / TOP_N
        base["unresolved_flag"] = 0
        base["complete"] = True
        base["note"] = "一字涨停买不进；该席留现，不重分配"
        return base

    buy_vwap = full_day_vwap(exec_points)
    if buy_vwap is None:
        base["entry_status"] = "nodata"
        base["note"] = "T+1 全日量额为零，无法计算买入 VWAP"
        return base

    base["entry_status"] = "filled"
    base["filled_ratio"] = 1.0
    base["buy_vwap"] = round(buy_vwap, 4)
    base["note"] = MARKET_SHADOW

    # ── 退出日（T+2 / 顺延日首 20 分钟）──
    if exit_points is None:
        base["note"] += "；T+2/顺延日窗口原料缺失，保持 pending"
        return base
    v_exit = validate_minute_points(exit_points)
    if not v_exit["complete"]:
        base["exit_status"] = "pending"
        base["note"] += f"；退出日分钟未完整：{';'.join(v_exit['issues'])}"
        return base

    blocked = detect_exit_blocked(exit_points, exit_prev_close)
    if blocked["blocked"]:
        base["exit_status"] = "carry" if blocked["reason"] != "zero_trade" else "pending"
        base["carry_flag"] = 1
        base["note"] += f"；退出受阻（{blocked['reason']}），顺延至下一可卖窗口"
        return base

    win = window_vwap(exit_points)
    if win["vwap"] is None:
        base["exit_status"] = "pending"
        base["note"] += f"；窗口零成交：{';'.join(win['issues'])}"
        return base

    sell_vwap = win["vwap"]
    gross = sell_vwap / buy_vwap - 1.0
    net = gross - COST_ROUND_TRIP
    base["exit_status"] = "exited"
    base["sell_vwap_20m"] = round(sell_vwap, 4)
    base["gross_ret"] = round(gross, 6)
    base["net_ret"] = round(net, 6)
    base["cash_weight"] = round(base["weight"] * (1.0 - (base["filled_ratio"] or 1.0)), 6)
    base["unresolved_flag"] = 0
    base["complete"] = True
    if win["issues"]:
        base["note"] += "；窗口提示: " + ";".join(win["issues"])
    return base


def approx_open_from_bars(bars, exec_iso: str, exit_iso: str) -> dict:
    """proxy 层：T+1 开盘 → T+2 开盘日 K open-open 扣费收益（独立 approx_open 层）。

    bars: fetch_daily 返回的日 K DataFrame（date/open/...）。日 K 只准进 proxy，
    严禁进入 minute20 VWAP。缺任一日开盘 → 空结果，不伪造。
    """
    out = {"layer": "approx_open", "gross_ret": None, "net_ret": None,
           "exec_open": None, "exit_open": None, "note": ""}
    if bars is None or len(bars) == 0:
        out["note"] = "日K缺失"
        return out
    dates = [str(x)[:10] for x in bars["date"]]
    opens = bars["open"].astype(float).to_numpy()
    def _open(iso):
        j = next((i for i, d in enumerate(dates) if d == iso), None)
        return float(opens[j]) if j is not None else None
    o1, o2 = _open(exec_iso), _open(exit_iso)
    if o1 is None or o2 is None or o1 <= 0:
        out["note"] = "缺 T+1/T+2 开盘价"
        return out
    out["exec_open"], out["exit_open"] = round(o1, 4), round(o2, 4)
    out["gross_ret"] = round(o2 / o1 - 1.0, 6)
    out["net_ret"] = round(out["gross_ret"] - COST_ROUND_TRIP, 6)
    return out


# ---------------------------------------------------------------- 采集编排

def collect_due(today: dt.date | None = None,
                 calendar: list[dt.date] | None = None,
                 fetch_minute=None,
                 fetch_daily=None,
                 include_proxy: bool = True) -> dict:
    """采集到期标签原料：T+1 全日分钟 + T+2/顺延日首 20 分钟分钟。

    - 按交易日历定位 exec_date=T+1、exit_date=T+2；受阻顺延日由下次运行补齐。
    - 失败/截断不落伪完整分钟；网络异常记 fail/nodata，保持待采。
    - fetch_minute/fetch_daily 为可注入接缝（测试用）；默认走 data_source 门面。
    """
    from data_source import fetch_daily as _fd, fetch_minute_days as _fm

    fetch_minute = fetch_minute or _fm
    fetch_daily = fetch_daily or _fd
    today = today or _today()
    if calendar is None:
        from data_source import trade_calendar
        calendar = trade_calendar()
    if not calendar:
        return {"status": "fail", "note": "交易日历不可用，本次不采集", "records": 0}

    snapshots = _list_universe_snapshots()
    if not snapshots:
        return {"status": "empty", "note": "无宇宙快照，待 record_signal 落盘", "records": 0}

    labels = _load_labels()
    by_key = {_label_key(r): r for r in labels}
    stats = {"signals": 0, "records": 0, "filled": 0, "unfilled": 0,
             "pending": 0, "carry": 0, "nodata": 0, "fail": 0,
             "minute_written": 0, "proxy_rows": 0}

    def _is_unresolved(r: dict | None) -> bool:
        if not r:
            return False
        return str(r.get("unresolved_flag", "0")) == "1" and r.get(
            "exit_status") in ("carry", "pending")

    for snap in snapshots:
        sig_iso = snap["signal_date"]
        sig_d = dt.date.fromisoformat(sig_iso)
        if sig_d >= today:
            continue
        exec_d = _next_td(calendar, sig_d)
        if exec_d is None:
            continue
        stats["signals"] += 1

        for row in snap["rows"]:
            row["signal_date"] = sig_iso
            mkey = (sig_iso, row["code"], "minute20")
            existing = by_key.get(mkey)
            prev_close_t = _f(row["raw"].get("close"))

            # 顺延：既有未决头寸不固定 T+2，从已尝试退出日的下一交易日继续
            if _is_unresolved(existing) and existing.get("exit_date"):
                try:
                    base_d = dt.date.fromisoformat(existing["exit_date"])
                except ValueError:
                    base_d = exec_d
                exit_d = _next_td(calendar, base_d)
            else:
                exit_d = _next_td(calendar, exec_d)

            # ── 拉分钟（接缝；异常隔离为 fail，不中断整批）──
            try:
                days = fetch_minute(row["code"]) or []
                fetch_err = None
            except Exception as e:                               # noqa: BLE001
                days, fetch_err = [], f"{type(e).__name__}: {e}"

            by = {d["date"]: d for d in days}
            exec_day = by.get(_ymd(exec_d))
            exit_day = by.get(_ymd(exit_d)) if exit_d else None

            exec_points = exec_day["points"] if exec_day else None
            exit_points = exit_day["points"] if exit_day else None

            # 完整原料才落分钟盘；截断/失败不写伪完整文件
            v_exec = validate_minute_points(exec_points) if exec_points else None
            if exec_points and v_exec and v_exec["complete"]:
                _write_minute(exec_d, row["code"], "entry_full",
                              exec_points, v_exec)
                stats["minute_written"] += 1
            v_exit = validate_minute_points(exit_points) if exit_points else None
            if exit_points and v_exit and v_exit["complete"]:
                _write_minute(exit_d, row["code"], "exit_window",
                              exit_points, v_exit)
                stats["minute_written"] += 1

            exit_prev_close = (_f(exec_day["points"][-1][1])
                               if exec_points else None)
            label = settle_minute_label(row, exec_points, exit_points,
                                        prev_close_t, exit_prev_close)
            label["exec_date"] = exec_d.isoformat()
            label["exit_date"] = exit_d.isoformat() if exit_d else ""
            if fetch_err and exec_points is None:
                label["entry_status"] = "fail"
                label["note"] = f"分钟请求失败：{fetch_err}"

            # ── proxy approx_open（独立层，日 K 仅供筛选；臂不变，layer 区分）──
            if include_proxy:
                try:
                    bars = fetch_daily(row["code"], today)
                    px = approx_open_from_bars(
                        bars, exec_d.isoformat(),
                        exit_d.isoformat() if exit_d else "")
                    if px["net_ret"] is not None:
                        proxy_row = dict(label)
                        proxy_row.update({
                            "layer": "approx_open",
                            "arm": UNIFIED_ARM,
                            "entry_status": "filled",
                            "exit_status": "exited",
                            "unresolved_flag": 0, "carry_flag": 0,
                            "buy_vwap": None, "sell_vwap_20m": None,
                            "gross_ret": px["gross_ret"],
                            "net_ret": px["net_ret"],
                            "exec_open": px["exec_open"],
                            "exit_open": px["exit_open"],
                            "complete": True,
                            "note": "proxy open-open 仅筛选，不入 minute20 门",
                        })
                        by_key[(sig_iso, row["code"], "approx_open")] = proxy_row
                        stats["proxy_rows"] += 1
                except Exception as e:                           # noqa: BLE001
                    stats["fail"] += 1
                    label["note"] += f"；proxy 日K失败：{type(e).__name__}"

            by_key[mkey] = label
            stats["records"] += 1
            st = label["entry_status"]
            if st in stats:
                stats[st] += 1
            if label["exit_status"] == "carry":
                stats["carry"] += 1
            elif label["exit_status"] == "pending":
                stats["pending"] += 1

    _save_labels(list(by_key.values()))
    stats["status"] = "ok"
    return stats


# ---------------------------------------------------------------- 真实原料回放审计

def replay_real_minute(source_dir: str | None = None) -> dict:
    """只读回放既有竞价分钟 pkl，重算全日/首 20 分钟 VWAP 并校验单调性。

    既有 pkl 只存执行日当日点、缺配对 T+2；本回放**不生成 minute20 净收益标签**。
    """
    import pickle
    src = Path(source_dir) if source_dir else (
        ROOT / "out" / "research" / "auction" / "minute")
    report = {"source_dir": str(src), "days": [], "codes": 0, "monotone_ok": 0,
              "vwap_samples": [], "note": ""}
    if not src.is_dir():
        report["note"] = "无既有分钟原料目录"
        return report
    for p in sorted(src.glob("*.pkl")):
        with open(p, "rb") as f:
            minute = pickle.load(f)
        day_entry = {"file": p.name, "codes": len(minute), "samples": []}
        for code, payload in minute.items():
            pts = payload["points"]
            v = validate_minute_points(pts)
            if v["monotone"] and v["ok"]:
                report["monotone_ok"] += 1
            vwap_day = full_day_vwap(pts)
            win = window_vwap(pts)
            day_entry["samples"].append({
                "code": code, "n_points": v["n_points"],
                "monotone": v["monotone"], "complete": v["complete"],
                "full_day_vwap": round(vwap_day, 4) if vwap_day else None,
                "vwap_0930_0950": round(win["vwap"], 4) if win["vwap"] else None,
                "window_labels": win["n_labels"],
            })
            if len(report["vwap_samples"]) < 12:
                report["vwap_samples"].append(day_entry["samples"][-1])
        report["codes"] += len(minute)
        report["days"].append(day_entry)
    report["note"] = ("既有原料仅执行日当日点、缺配对 T+2；未伪造历史 minute20 标签。"
                       if report["days"] else "无既有分钟 pkl")
    return report


def write_audit(report: dict) -> dict:
    """落 data_audit.json + data_audit.log（原子写）。"""
    report["audited_at"] = _now()
    report["cost"] = {
        "COST_COMMISSION": COST_COMMISSION, "COST_STAMP_TAX": COST_STAMP_TAX,
        "COST_TRANSFER": COST_TRANSFER, "COST_SLIPPAGE": COST_SLIPPAGE,
        "COST_ROUND_TRIP": COST_ROUND_TRIP,
    }
    fstore.write_json(str(AUDIT_JSON), report)
    lines = [f"[{report['audited_at']}] premium11 data audit",
             f"days={len(report['days'])} codes={report['codes']} "
             f"monotone_ok={report['monotone_ok']}",
             f"note: {report['note']}"]
    for d in report["days"]:
        lines.append(f"  {d['file']} codes={d['codes']}")
    fstore.write_text(str(AUDIT_LOG), "\n".join(lines) + "\n")
    return report
