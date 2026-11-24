# -*- coding: utf-8 -*-
"""台账层：预判准确性核验账。

生命周期：predict 日登记（predict_date+code，一经写入不得改写）→ 下一交易日
核验回填 outcome → accuracy_summary 汇总判读。

核验口径：
- seat_bought：核验日席位明细出现 (code, 净买>0)（LHB 只披露触发披露条件的
  交易，未上榜≠没买——口径偏差见 RESEARCH.md）；
- next_pct / next_limit：该股 predict_date 后第一根日K（T+1 收盘涨幅 ≥9.8%）；
- outcome 四态：
    ignited    席位买入且当日涨停（点火命中，预判正确）
    bought     席位买入但未涨停（买点命中、点火未成）
    limit_only 未买但涨停（别的资金点火，预判落空）
    miss       都没有
- 判读门槛：独立 predict_date 簇数 ≥ MIN_INFER_DAYS 才输出结论性判读
  （同日多票共享市场状态，不是独立样本——与主工程校准纪律一致）。
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os

import fstore
import pandas as pd

from . import config as C
from .seatdata import (get_bars, load_bars_cache, next_bar_after, now_cn,
                       save_bars_cache)

LEDGER_COLS = ["predict_date", "code", "name", "score", "score100", "rank",
               "eve_tags", "model_version", "stratum", "status", "verify_date",
               "next_pct", "next_limit", "seat_bought", "seat_net_buy_yi",
               "outcome", "note", "exec_oo", "exec_brk5", "exec_flag",
               "joint_rank",   # v2.5 改动三增列：λ 联合分名次（主 rank 语义不变）
               "own_pick_penalty", "sell_pick_penalty", "sell_recent_yi"]
               # v3.0.2 增列：自选/卖出记忆后处理惩罚留痕（判读时可分离归因）


def _load() -> list[dict]:
    if not os.path.exists(C.LEDGER_CSV):
        return []
    with open(C.LEDGER_CSV, encoding="utf-8-sig", newline="") as f:
        return [{c: r.get(c, "") for c in LEDGER_COLS} for r in csv.DictReader(f)]


def _save(ledger: list[dict]) -> None:
    # QUOTE_NONNUMERIC：全部文本字段加引号落盘——csv/pandas/DictReader 读取端
    # 完全透明兼容（引号由 csv 模块对称解析），而文件同时构成合法的 Python
    # 字面量序列（日期/代码等前导零字段被引号保护），可被逐文件 py_compile 的
    # 门禁当源码编译（2026-09-27 数据落盘格式统一约定）。
    # 原子写：预注册台账是纪律证据，直接覆写被截断=证据损坏。
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LEDGER_COLS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(ledger)
    fstore.write_text(C.LEDGER_CSV, buf.getvalue(), encoding="utf-8-sig")


def eve_tags(pred: dict) -> str:
    feats = {
        "板": f"{pred.get('lianban')}板" if pred.get("lianban") is not None else "板?",
        "幅": f"{pred.get('pct'):+.1f}%" if pred.get("pct") is not None else "幅?",
        "换": f"{pred.get('turnover'):.0f}%" if pred.get("turnover") is not None else "换?",
        "量比": f"{pred.get('vr'):.1f}" if pred.get("vr") is not None else "量比?",
        "价": f"{pred.get('price'):.1f}" if pred.get("price") is not None else "价?",
        "值": (f"{pred.get('ltsz'):.0f}亿" if pred.get("ltsz") is not None else "值?"),
    }
    tags = "·".join(feats.values())
    if pred.get("via_switch"):
        tags += "·切换通道"
    elif pred.get("via_m5"):
        tags += "·动量通道"
    elif pred.get("via_stir"):
        tags += "·抬头通道"
    if pred.get("industry"):
        tags += f"·{pred['industry']}"
    return tags


def _current_model_version() -> str:
    """当日甄选用的模型口径（单一事实源=model.json）。"""
    mp = os.path.join(C.DATA_DIR, "model", "model.json")
    meta = fstore.read_json(mp)
    return meta.get("version", "?") if isinstance(meta, dict) else "?"


def rows_for(trade_date: dt.date) -> list[dict]:
    """某预判日的登记行（按 rank 排序）——渲染层只读入口，零网络。"""
    key = trade_date.isoformat()
    rows = [r for r in _load() if r["predict_date"] == key]
    return sorted(rows, key=lambda r: int(r["rank"]) if str(r["rank"]).isdigit() else 99)


def register(trade_date: dt.date, predictions: list[dict],
             regen_same_day: bool = False,
             ranked: list[dict] | None = None) -> str:
    """登记当日预判（预注册纪律）。

    - 该 predict_date 已登记 → 拒绝覆写；
    - 唯一例外 regen_same_day：当日（北京时间）重跑时允许重登——
      收盘后首跑（如 15:10 定时）席位当日 LHB 尚未发布（约 17~19 点），
      晚间重跑拿到完整 LHB 后刷新甄选是**核验前**的数据补全，不违反
      「不得在看到结果后改单」的纪律；跨日一律冻结。
    - ranked：当日完整榜单（漏判归因依据）。**登记成功才落快照**——被拒时
      台账没动，快照若照写，榜单与台账就对不上（此前调用方无条件
      persist_ranking，11-22 03:24 那次「拒绝覆写」仍改写了当日榜单）。
    """
    ledger = _load()
    key = trade_date.isoformat()
    if any(r["predict_date"] == key for r in ledger):
        if regen_same_day and trade_date == now_cn().date():
            # 空预测不清台账：晚间重跑若宇宙空/模型异常，把早间已登记的甄选
            # 整日抹掉等于销毁预注册证据。弃权日走 abstain 标记（screen 层），
            # 台账保持早间登记原样。
            old_n = sum(1 for r in ledger if r["predict_date"] == key)
            if old_n and not predictions:
                return (f"{trade_date} 已登记 {old_n} 条且本次预测为空——拒绝清空"
                        f"（保早间预注册，弃权日应走 abstain 标记）")
            ledger = [r for r in ledger if r["predict_date"] != key]
        else:
            return f"{trade_date} 已登记过，拒绝覆写（预注册纪律：预判一经登记不得改）"
    for p in predictions:
        ledger.append({
            "predict_date": key,
            "code": p["code"], "name": p["name"],
            "score": p.get("score", ""), "score100": p.get("score100", ""),
            "rank": p.get("rank", ""), "eve_tags": eve_tags(p),
            "status": "pending", "verify_date": "", "next_pct": "",
            "next_limit": "", "seat_bought": "", "seat_net_buy_yi": "",
            "model_version": _current_model_version(),
            "stratum": p.get("stratum", ""),   # 出勤三层标注（改动五，只标注不改排序）
            "outcome": "", "note": "",
            "exec_oo": "", "exec_brk5": "", "exec_flag": "",
            "joint_rank": p.get("joint_rank", ""),  # λ 联合分名次（改动三，增列）
            "own_pick_penalty": p.get("own_pick_penalty", ""),
            "sell_pick_penalty": p.get("sell_pick_penalty", ""),
            "sell_recent_yi": p.get("sell_recent_yi", ""),
        })
    ledger.sort(key=lambda r: (r["predict_date"], str(r["rank"])))
    _save(ledger)
    if ranked is not None:
        persist_ranking(trade_date, ranked)
    return f"{trade_date} 登记 {len(predictions)} 条预判"


# 成本口径切换日（2026-09-25）：此前核验行的 exec_oo 为毛收益，一次性迁移回填
# 已完成、迁移函数已删除；此后核验行直接扣费（见 _fill_exec）。历史口径见
# config.py 成本迁移注释块。


def _fill_exec(r: dict, bars: pd.DataFrame) -> None:
    """R5 执行模拟（扣费后净值，双边成本 EXEC_COST_ROUND_TRIP=0.152%；问题席
    deep3 定义。2026-09-26 成本口径迁移日起新核验行直接扣费；迁移日前核验的
    12 笔毛收益行由 migrate_cost_basis 一次性补扣并在 note 标注）：

    exec_oo   = 核验日开盘买 → 下一交易日开盘卖（用户隔夜主模式）——EXEC 主指标
                唯一口径（改动一②预注册冻结）：扣费后 exec_oo 的独立预测日簇
                槽位均值，一字槽两侧同剔
    exec_brk5 = 核验日 +5% 突破价买（开盘高于触发则按开盘）→ 次日开盘卖
    exec_flag = ok / 一字 / 未触发 / 待T+2 / 无数据
    一字判定=开盘价≥涨停价(昨收×1.10 四舍五入到分)-1厘。
    """
    try:
        vd = r.get("verify_date")
        if not vd:
            return
        dates = bars["date"].map(lambda s: str(s)[:10]).tolist()
        if vd not in dates:
            r["exec_flag"] = r.get("exec_flag") or "无数据"
            return
        j = dates.index(vd)
        if j + 1 >= len(bars) or j < 1:
            r["exec_flag"] = "待T+2"
            return
        vopen = float(bars["open"].iloc[j])
        vhigh = float(bars["high"].iloc[j])
        prev = float(bars["close"].iloc[j - 1])
        nopen = float(bars["open"].iloc[j + 1])
        limit = round(prev * 1.10, 2)
        flag = "ok"
        if vopen >= limit - 0.001:
            r["exec_oo"] = ""
            flag = "一字"
        elif vopen > 0:
            r["exec_oo"] = round((nopen / vopen - 1.0) * 100.0
                                 - C.EXEC_COST_ROUND_TRIP, 2)
        trig = round(prev * 1.05, 2)
        if vhigh >= trig:
            entry = max(vopen, trig)
            if entry >= limit - 0.001:
                r["exec_brk5"] = "" if flag == "一字" else r.get("exec_brk5", "")
                if flag != "一字":
                    flag = flag + "|brk一字"
            elif entry > 0:
                r["exec_brk5"] = round((nopen / entry - 1.0) * 100.0
                                       - C.EXEC_COST_ROUND_TRIP, 2)
        else:
            flag = flag + "|未触发" if flag == "ok" else flag
        r["exec_flag"] = flag
    except Exception:                            # noqa: BLE001 执行列失败不毁核验
        r["exec_flag"] = r.get("exec_flag") or "无数据"


def verify(today: dt.date | None = None,
           seat_trades: pd.DataFrame | None = None) -> str:
    """核验所有 pending 且 T+1 已到的预判行。席位明细须为最新（verify 前先 update）。"""
    today = today or dt.date.today()
    if seat_trades is None:
        from .seatdata import load_seat_trades
        seat_trades = load_seat_trades().copy()
    seat_trades["交易日期"] = pd.to_datetime(seat_trades["交易日期"]).dt.date
    # 明细覆盖断言（路线图#3）：明细末日期 < 核验日时，席位当日买入不可见，
    # 「已买」会被误判 miss 并 verified 冻结——这些行保持 pending
    detail_max = seat_trades["交易日期"].max() if len(seat_trades) else None
    ledger = _load()
    blocked = 0
    if detail_max is not None:
        for r in ledger:
            if r["status"] != "pending":
                continue
            pday = dt.date.fromisoformat(r["predict_date"])
            if pday >= today:
                continue
            # 该行的核验日未知（未算过），用 pday+1~today 窗口判：全部在明细
            # 覆盖之后才安全；只要 pday+1 超出覆盖就等明细追上再核
            if pday + dt.timedelta(days=1) > detail_max:
                blocked += 1
    if blocked:
        print(f"[核验] 席位明细末日期 {detail_max} 未覆盖 {blocked} 行的核验窗，"
              f"保持 pending（先跑 update/daily_sync 刷新明细）")
    bars_cache = load_bars_cache()
    changed = 0
    for r in ledger:
        if r["status"] != "pending":
            continue
        pday = dt.date.fromisoformat(r["predict_date"])
        if pday >= today:
            continue
        if detail_max is not None and pday + dt.timedelta(days=1) > detail_max:
            continue                      # 路线图#3：明细未覆盖核验窗，保持 pending
        bars = get_bars(bars_cache, r["code"], today)
        if bars is None:
            continue
        nxt = next_bar_after(bars, pday)
        if nxt is None or nxt[0] > today.isoformat():
            continue                      # T+1 日K未就绪（停牌/数据滞后），保持 pending
        vdate, vpct = nxt
        # 当日 LHB 约 17~19 点才发布：T+1 就是今天且未到 19:30 时先不核验，
        # 否则席位明细缺今日行会把「它买了」误判成 miss 并冻结
        if vdate == today.isoformat() and now_cn().time() < dt.time(19, 30):
            continue
        vpct = vpct if vpct is not None else float("nan")
        ev = seat_trades[(seat_trades["交易日期"] == dt.date.fromisoformat(vdate))
                         & (seat_trades["股票代码"] == r["code"])]
        net = float(ev["净买额"].iloc[0]) if len(ev) else 0.0
        bought = bool(len(ev) and net > 0)
        limit = bool(vpct >= C.LIMIT_UP_APPROX)
        outcome = ("ignited" if bought and limit else
                   "bought" if bought else
                   "limit_only" if limit else "miss")
        r.update({
            "status": "verified", "verify_date": vdate,
            "next_pct": round(vpct, 2) if vpct == vpct else "",
            "next_limit": "Y" if limit else "N",
            "seat_bought": "Y" if bought else "N",
            "seat_net_buy_yi": round(net / 1e8, 3) if bought else "",
            "outcome": outcome,
        })
        _fill_exec(r, bars)
        changed += 1
    # R5 补账：已核验但缺执行模拟或仍标「待T+2」的行（T+2 出来后回填；
    # 旧 bug：exec_flag 已写"待T+2"会被 not exec_flag 条件跳过，永不补齐）
    for r in ledger:
        if (r["status"] == "verified" and r.get("verify_date")
                and (not r.get("exec_flag") or r.get("exec_flag") == "待T+2")):
            bars2 = get_bars(bars_cache, r["code"], today)
            if bars2 is not None:
                _fill_exec(r, bars2)
                changed += 1
    save_bars_cache(bars_cache)
    if changed:
        _save(ledger)
    return f"核验回填 {changed} 行（含执行模拟补账）"


def _sealed_mainboard(trade_date: dt.date) -> set[str]:
    """某日主工程归档里的主板封板股（判读基线的分母，零网络）。"""
    from .seatdata import archive_signal_files, is_main_board
    for d, f in archive_signal_files():
        if d != trade_date:
            continue
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            return set()
        if "zt_status" not in df.columns:
            return set()
        sub = df[(df["zt_status"] == "封板") & df["code"].notna()]
        return {c.zfill(6) for c in sub["code"] if is_main_board(str(c))}
    return set()


def baseline_and_recall(seat_trades: pd.DataFrame | None = None) -> dict | None:
    """预注册判读的基线与召回（零网络：台账 + 归档 + 席位明细 CSV）。

    对每个已核验 predict_date T（verify_date 即其次日）：
    - 基线 = T 日主板封板股中，席位次日净买占比——判 precision@N 好坏的参照
      （指标在 config 台账纪律段预注册，此处只建裁判，不碰任何打分口径）；
    - 召回 = 席位次日主板净买事件中，落入当日 topN 登记的比例。
    席位明细缺失（未跑过 update）时返回 None，报告层跳过该节。
    """
    if seat_trades is None:
        if not os.path.exists(C.SEAT_TRADES_CSV):
            return None
        from .seatdata import load_seat_trades
        seat_trades = load_seat_trades()
    seat_trades = seat_trades.copy()
    seat_trades["交易日期"] = pd.to_datetime(seat_trades["交易日期"]).dt.date

    verified = [r for r in _load() if r["status"] == "verified"]
    out = {"n_days": 0, "baseline": {"n": 0, "hit": 0},
           "recall": {"n": 0, "caught": 0}, "by_day": []}
    for pred_day in sorted({r["predict_date"] for r in verified}):
        rows = [r for r in verified if r["predict_date"] == pred_day]
        vdate = next((r["verify_date"] for r in rows if r.get("verify_date")), None)
        if not vdate:
            continue
        reg = {r["code"] for r in rows}
        buys = seat_trades[(seat_trades["交易日期"] == dt.date.fromisoformat(vdate))
                           & (seat_trades["净买额"] > 0) & seat_trades["主板"]]
        bought = set(buys["股票代码"])
        sealed = _sealed_mainboard(dt.date.fromisoformat(pred_day))
        b_n, b_hit = len(sealed), len(sealed & bought)
        r_n, r_caught = len(bought), len(bought & reg)
        out["n_days"] += 1
        out["baseline"]["n"] += b_n
        out["baseline"]["hit"] += b_hit
        out["recall"]["n"] += r_n
        out["recall"]["caught"] += r_caught
        out["by_day"].append({
            "date": pred_day, "sealed_n": b_n, "baseline_hit": b_hit,
            "seat_buys_n": r_n, "recall_caught": r_caught,
            "hit": int(sum(1 for r in rows if r["seat_bought"] == "Y")),
            "ignited": int(sum(1 for r in rows if r["outcome"] == "ignited"))})
    b, r = out["baseline"], out["recall"]
    if b["n"]:
        b["pct"] = round(100.0 * b["hit"] / b["n"], 1)
    if r["n"]:
        r["pct"] = round(100.0 * r["caught"] / r["n"], 1)
    return out


def accuracy_summary() -> dict:
    """台账准确率汇总。簇数（独立 predict_date）不足判读门槛时只报数不判读。"""
    ledger = [r for r in _load() if r["status"] == "verified"]
    if not ledger:
        return {"n": 0, "note": "台账为空或全部待核验"}
    df = pd.DataFrame(ledger)
    df["next_pct"] = pd.to_numeric(df["next_pct"], errors="coerce")
    n_days = df["predict_date"].nunique()
    out = {
        "n_rows": len(df), "n_days": int(n_days),
        "gate": C.MIN_INFER_DAYS,
        "by_outcome": df["outcome"].value_counts().to_dict(),
        "precision_at_n": round(100.0 * float((df["seat_bought"] == "Y").mean()), 1),
        "ignition_hit": round(100.0 * float((df["outcome"] == "ignited").mean()), 1),
        "next_pct_mean": round(float(df["next_pct"].mean()), 2),
        "next_pct_win": round(100.0 * float((df["next_pct"] > 0).mean()), 1),
        "by_day": [],
    }
    for d, g in df.groupby("predict_date"):
        out["by_day"].append({
            "date": d, "n": len(g),
            "hit": int((g["seat_bought"] == "Y").sum()),
            "ignited": int((g["outcome"] == "ignited").sum()),
            "next_pct_mean": round(float(g["next_pct"].mean()), 2),
        })
    out["by_day"].sort(key=lambda x: x["date"])
    out["verdict"] = ("样本不足（独立预判日 %d < %d，不得下结论）" % (n_days, C.MIN_INFER_DAYS)
                      if n_days < C.MIN_INFER_DAYS else "达到判读门槛，见 by_day 分解")
    # 版本分层（路线图#8）：各口径独立 precision/日数。model_version 列于
    # 2026-09-26 落地，此前登记行无可靠归因 → 如实标注「未记录」（不猜版本）
    ver_col = df["model_version"] if "model_version" in df.columns else pd.Series(["未记录"] * len(df))
    df["_ver"] = ver_col.fillna("未记录").replace("", "未记录")
    by_ver = {}
    for ver, vg in df.groupby("_ver"):
        by_ver[ver] = {
            "n_rows": len(vg), "n_days": int(vg["predict_date"].nunique()),
            "precision_at_n": round(100.0 * float((vg["seat_bought"] == "Y").mean()), 1),
            "ignition_hit": round(100.0 * float((vg["outcome"] == "ignited").mean()), 1),
            "next_pct_mean": round(float(vg["next_pct"].mean()), 2),
        }
    out["by_version"] = by_ver
    try:
        out.update(baseline_and_recall() or {})
    except Exception as e:                     # noqa: BLE001 基线缺数据不影响主账
        out["baseline_error"] = f"{type(e).__name__}: {e}"   # 留痕不静默
    try:
        out["autopsy"] = autopsy_summary()
    except Exception as e:                     # noqa: BLE001
        out["autopsy_error"] = f"{type(e).__name__}: {e}"
    try:
        oo = [float(r["exec_oo"]) for r in ledger
              if r.get("exec_oo") not in ("", None)]
        b5 = [float(r["exec_brk5"]) for r in ledger
              if r.get("exec_brk5") not in ("", None)]
        n_skip = sum(1 for r in ledger
                     if r.get("exec_flag") and "未触发" in r["exec_flag"])
        def _st(v):
            v = [x for x in v if x == x]       # 剔 NaN：一个脏值会让中位数/胜率
            return ({"n": len(v), "median": round(sorted(v)[len(v)//2], 2),
                     "win": round(100.0*sum(1 for x in v if x > 0)/len(v), 0)}
                    if v else {})              # 整体变 nan 且报告打出 nan%
        oo_st = _st(oo)
        # 主指标口径（改动一②预注册冻结）：扣费后 exec_oo 的独立预测日簇槽位
        # 均值——每日 top3 槽位并列等权取均值再对簇取均值；一字/待T+2 槽位
        # （exec_oo 为空）剔除，不进分子分母。
        by_day: dict[str, list] = {}
        for r in ledger:
            v = r.get("exec_oo")
            if v not in ("", None):
                by_day.setdefault(r["predict_date"], []).append(float(v))
        if by_day:
            day_means = [sum(v) / len(v) for v in by_day.values()]
            oo_st["cluster_mean"] = round(sum(day_means) / len(day_means), 3)
            oo_st["cluster_days"] = len(day_means)
        out["exec"] = {"n_ok": len(oo)+len(b5), "oo": oo_st, "b5": _st(b5),
                       "n_skip": n_skip}
    except Exception as e:                     # noqa: BLE001 EXEC 是预注册主指标，
        out["exec_error"] = f"{type(e).__name__}: {e}"   # 坏值致整节消失必须留痕
    # 出勤三层双口径簿记（改动五）：主门分母=normal 层（判读口径，预注册）；
    # EV 恒等式分母=全总体（诊断口径，上方 precision_at_n / next_pct_* 保持
    # 全体口径不变）。两数并列呈现、不互换；low_att 层 seat_bought 命中率=
    # 反向监控（命中率高于 normal 层 → 降档过严）。
    strata = {}
    for layer in ("normal", "low_att", "return_day"):
        g = df[df["stratum"] == layer]
        strata[layer] = {
            "n_rows": int(len(g)),
            "n_days": int(g["predict_date"].nunique()),
            "precision_at_n": (round(100.0 * float((g["seat_bought"] == "Y").mean()), 1)
                               if len(g) else None),
        }
    out["strata"] = strata
    out["strata_untagged"] = int((df["stratum"].fillna("") == "").sum())
    return out


def seat_surprises(verify_date: dt.date, registered_codes: set[str],
                   seat_trades: pd.DataFrame) -> list[dict]:
    """核验日席位的真实主板净买事件中，台账 topN 之外的部分（漏判清单，算召回分母）。"""
    seat_trades = seat_trades.copy()
    seat_trades["交易日期"] = pd.to_datetime(seat_trades["交易日期"]).dt.date
    ev = seat_trades[(seat_trades["交易日期"] == verify_date)
                     & (seat_trades["净买额"] > 0) & seat_trades["主板"]]
    return [{"code": r["股票代码"], "name": r["名称"],
             "net_buy_yi": round(float(r["净买额"]) / 1e8, 3),
             "caught": r["股票代码"] in registered_codes}
            for _, r in ev.iterrows()]


# ---------------------------------------------------------------- 出勤三层标注
# deep5/attendance.md:91-99 预注册（2026-09-26 即日冻结，改动五）：
# - normal     无触发 → 照常出票，主 precision 口径只算这层（判读口径）；
# - low_att    在持 active10≤4（唯一 <50% 桶，P(出勤)=0.471 n=17）或 OOS
#              compact P<0.5 → 照常出票+标注+建议仓位减半，precision 分层单列；
# - return_day 连缺2日（历史 6/6 回归、强度中低）→ 照常出票+标注，不降档；
# - 不做硬门控（attendance.md:63-70 门控精度 ≤0.36，只会白丢点火事件）；
# - 反向监控：low_att 层 seat_bought 命中率（命中率高于 normal → 降档过严）；
# - 情绪维度不上（attendance.md:19）。
ATTEND_ACTIVE10_LOW = 4
ATTEND_P_LOW = 0.5
ATTEND_STREAK_RETURN = 2
ATTEND_FEATS = ("y1", "streak_abs", "ma5", "held_active10", "zt_rel")


def _seat_daily_counts(seat_trades: pd.DataFrame) -> tuple[dict, str | None]:
    """主板 (日→净买事件数) + LIVE 日（PROFILE_SINCE 起首个主板事件日；此前为
    数据管道未上线，非真实缺席——deep5/attendance.md §0 口径）。"""
    st = seat_trades[seat_trades["主板"]]
    cnt: dict[str, int] = {}
    live = None
    since = C.PROFILE_SINCE
    for d, net in zip(st["交易日期"], st["净买额"]):
        ds = d.isoformat() if isinstance(d, dt.date) else str(d)[:10]
        if ds < since:
            continue
        if net > 0:
            cnt[ds] = cnt.get(ds, 0) + 1
        if live is None:
            live = ds
    return cnt, live


def _nowcast_active10_series(seat_trades: pd.DataFrame, days: list[str],
                             max_hold: int = 10) -> dict[str, int]:
    """fusion 口径在持序列（attendance.py nowcast_series 同口径）：净买→同股
    其后首个净卖配对，未配对=在持；逐日在持张数=entry≤D<退出 且持有≤max_hold
    个交易日（unpaired 净卖为窗口外库存兑现，右删失不计）。"""
    st = seat_trades[seat_trades["主板"]].sort_values("交易日期")
    hold: dict[str, str] = {}
    episodes: list[tuple[str, str, str | None]] = []
    for d, code, net in zip(st["交易日期"], st["股票代码"], st["净买额"]):
        ds = d.isoformat() if isinstance(d, dt.date) else str(d)[:10]
        if ds < C.PROFILE_SINCE:
            continue
        if net > 0:
            if code not in hold:
                hold[code] = ds
        elif net < 0 and code in hold:
            episodes.append((code, hold.pop(code), ds))
    episodes += [(c, e, None) for c, e in hold.items()]
    pos = {d: i for i, d in enumerate(days)}
    out: dict[str, int] = {}
    for D in days:
        iD = pos[D]
        n = 0
        for c, e, x in episodes:
            pe = pos.get(e)
            if pe is None or pe > iD:
                continue
            if x is not None and pos.get(x, iD + 1) <= iD:
                continue
            if iD - pe <= max_hold:
                n += 1
        out[D] = n
    return out


def _zt_rate_series(bars_cache: dict) -> dict[str, float]:
    """主板横截面日涨停率（pct≥LIMIT_UP_APPROX 占比；attendance.py 率口径，
    横截面样本量随日波动 279→1200 不等，一律取率不取家数）。"""
    per_day: dict[str, list] = {}
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in C.MAIN_BOARD_PREFIXES:
            continue
        c = pd.to_numeric(b["close"], errors="coerce")
        pct = (c / c.shift(1) - 1.0) * 100.0
        for d, p in zip([str(x)[:10] for x in b["date"]], pct.to_numpy()):
            if p == p:
                per_day.setdefault(d, []).append(float(p))
    return {d: sum(1 for p in v if p >= C.LIMIT_UP_APPROX) / len(v)
            for d, v in per_day.items() if v}


def attendance_stratum(trade_date: dt.date,
                       seat_trades: pd.DataFrame | None = None,
                       bars_cache: dict | None = None) -> tuple[str, dict]:
    """预测日 D 的出勤层标注（特征全部 ≤ D-1 收盘可知，零网络可算）。

    compact 模型=deep5 attendance.py 的 5 特征 LR（y1/streak_abs/ma5/
    held_active10/zt_rel），按月走样前重训（训练日 < 当月 1 日，OOS 语义）。
    模型不可训（样本/类别不足）或 zt_rel 缺失时退规则档判层并如实标注。
    优先级 low_att > return_day：low_att 带仓位后果，保守优先。
    返回 (stratum, info)，info 供报告呈现触发依据。
    """
    if seat_trades is None:
        from .seatdata import load_seat_trades
        seat_trades = load_seat_trades()
    from . import seatdata as SD
    cal = None
    try:
        if os.path.exists(SD._TRADE_CAL_CSV):
            cal = SD._load_trade_cal_csv()
    except Exception:                          # noqa: BLE001
        cal = None
    if not cal:
        cal = SD._trade_cal()                  # 磁盘未覆盖时兜底（可能走网络）
    if not cal:
        return "normal", {"note": "交易日历不可用，默认 normal"}
    cnt, live = _seat_daily_counts(seat_trades)
    if live is None:
        return "normal", {"note": "PROFILE_SINCE 起无主板事件，默认 normal"}
    days = sorted(d for d, ok in cal.items()
                  if ok and live <= d < trade_date.isoformat())
    if len(days) < 10:
        return "normal", {"note": "LIVE 后预热不足 10 交易日，默认 normal"}

    def day_feats(D_idx: int) -> dict | None:
        """下标 D_idx 预测日的出勤特征（≤D-1 可知）；D_idx=len(days) 即预测日。"""
        if D_idx < 10:
            return None
        p = D_idx - 1
        D1 = days[p]
        sa = 0
        for j in range(p, -1, -1):
            if cnt.get(days[j], 0) >= 1:
                break
            sa += 1
        m5 = sum(cnt.get(days[j], 0) for j in range(D_idx - 5, D_idx)) / 5.0
        return {"D1": D1, "y1": min(cnt.get(D1, 0), 3), "streak_abs": sa,
                "ma5": m5, "held_active10": act10.get(D1, 0)}

    act10 = _nowcast_active10_series(seat_trades, days)
    fD = day_feats(len(days)) or {}
    streak_abs = fD.get("streak_abs", 0)
    ma5 = fD.get("ma5", 0.0)
    held_active10 = act10.get(days[-1], 0)
    info = {"active10": held_active10, "streak_abs": streak_abs,
            "ma5": round(ma5, 2), "live": live}
    p_att = None
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        zt = _zt_rate_series(bars_cache) if bars_cache else {}

        def day_feats_zt(D_idx: int) -> dict | None:
            f = day_feats(D_idx)
            if f is None:
                return None
            D1 = f["D1"]
            zt20 = [zt[h] for h in days[max(0, D_idx - 20):D_idx] if h in zt]
            if not zt20 or D1 not in zt:
                return None                    # bars 横截面缺该日 → 率特征缺失
            f["zt_rel"] = zt[D1] / float(np.median(zt20))
            return f

        month0 = trade_date.replace(day=1).isoformat()
        tr_X, tr_y = [], []
        for j, D in enumerate(days):
            if D >= month0:                    # 训练日 < 当月 1 日（OOS 语义）
                break
            f = day_feats_zt(j)
            if f is None:
                continue
            tr_X.append([f[k] for k in ATTEND_FEATS])
            tr_y.append(1 if cnt.get(D, 0) >= 1 else 0)
        f_now = day_feats_zt(len(days))
        if len(tr_X) >= 30 and len(set(tr_y)) == 2 and f_now is not None:
            sc2 = StandardScaler().fit(tr_X)
            clf2 = LogisticRegression(max_iter=3000, C=1.0).fit(
                sc2.transform(tr_X), tr_y)
            p_att = float(clf2.predict_proba(
                sc2.transform([[f_now[k] for k in ATTEND_FEATS]]))[0, 1])
            info["n_train_days"] = len(tr_X)
    except Exception as e:                     # noqa: BLE001 模型档缺失退规则档
        info["model_note"] = type(e).__name__
    info["p_attend"] = round(p_att, 3) if p_att is not None else None
    low_rule = held_active10 <= ATTEND_ACTIVE10_LOW
    low_model = p_att is not None and p_att < ATTEND_P_LOW
    if low_rule or low_model:
        s = "low_att"
        info["trigger"] = ("active10<=4+compact_P<0.5" if low_rule and low_model
                           else "active10<=4" if low_rule else "compact_P<0.5")
    elif streak_abs >= ATTEND_STREAK_RETURN:
        s = "return_day"
        info["trigger"] = "连缺2日"
    else:
        s = "normal"
    return s, info


# ---------------------------------------------------------------- 漏判归因
RANK_COLS = ["code", "name", "rank", "score", "shadow_rank", "shadow_score",
             "lianban", "pct", "ltsz", "m5", "zt60", "maxlb60",
             "via_switch", "via_m5", "via_stir", "industry",
             "money_prior", "score_joint", "joint_rank",
             # v3.0 增列（own-pick 惩罚 + meta-abstain 标注）
             "own_pick_penalty", "own_pick_count_5d", "sr_day_prob", "sr_ticket_prob",
             # v3.0.2 增列（卖出记忆惩罚，2026-09-28 事故实证）
             "sell_pick_penalty", "sell_recent_yi"]
AUTOPSY_COLS = ["predict_date", "verify_date", "code", "name", "net_buy_yi",
                "verdict", "rank", "channel", "miss_pct", "miss_oo", "note"]


def _rank_dir() -> str:
    return os.path.join(C.DATA_DIR, "rankings")


def persist_ranking(trade_date: dt.date, ranked: list[dict]) -> str:
    """落盘当日完整榜单快照（判读依据：冻结「我们当天看到了什么」，防数据
    漂移后无法复核漏判归因）。register 时调用。原子写防截断。"""
    d = _rank_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{trade_date:%Y%m%d}.csv")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RANK_COLS)
    w.writeheader()
    for r in ranked:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RANK_COLS})
    fstore.write_text(path, buf.getvalue(), encoding="utf-8-sig")
    return path


def load_ranking(trade_date: dt.date) -> dict[str, dict]:
    path = os.path.join(_rank_dir(), f"{trade_date:%Y%m%d}.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["code"]: r for r in csv.DictReader(f)}


def autopsy(predict_date: dt.date, seat_trades: pd.DataFrame,
            verify_date: dt.date | None = None) -> list[dict] | None:
    """漏判归因：核验日席位真实主板净买逐只分类——

    命中(top3 内) / 排序漏(在宇宙但 rank>N——打分深度问题) / 宇宙漏(榜单无此票
    ——通道召回问题)。依赖当日排名快照（无快照返回 None）；结果累计落盘
    autopsy.csv。这是 v2 决策（如按模式配额）的证据来源：排序漏多→打分问题，
    宇宙漏多→通道问题。

    机会成本对照（路线图#5）：miss_pct=核验日涨幅、miss_oo=核验日开→次日开
    （与 exec_oo 同口径），从 bars_cache 补算——「漏掉的比选中的好多少」由此
    可见（复核实测：3 只排序漏票全板 +10%、隔夜 +8.7~13.1%，而登记票中位 -2%）。
    """
    ranking = load_ranking(predict_date)
    if not ranking:
        return None
    vday = verify_date
    if vday is None:
        rows = [r for r in _load()
                if r["predict_date"] == predict_date.isoformat()
                and r["status"] == "verified"]
        if not rows:
            return None
        vday = dt.date.fromisoformat(rows[0]["verify_date"])
    st = seat_trades.copy()
    st["交易日期"] = pd.to_datetime(st["交易日期"]).dt.date
    ev = st[(st["交易日期"] == vday) & (st["净买额"] > 0) & st["主板"]]

    bars_cache = load_bars_cache()
    out = []
    for _, r in ev.iterrows():
        code = r["股票代码"]
        rk = ranking.get(code)
        if rk is None:
            verdict, rank_s, chan = "宇宙漏", "", ""
        elif int(float(rk["rank"])) <= C.TOP_N:
            verdict, rank_s, chan = "命中", rk["rank"], _channel_of(rk)
        else:
            verdict, rank_s, chan = "排序漏", rk["rank"], _channel_of(rk)
        # 机会成本对照（路线图#5）：核验日涨幅 + 隔夜开→开
        miss_pct = miss_oo = ""
        b = bars_cache.get(code)
        if b is not None and len(b):
            dates = b["date"].map(lambda s: str(s)[:10]).tolist()
            if vday.isoformat() in dates:
                j = dates.index(vday.isoformat())
                c = b["close"].astype(float)
                if j > 0:
                    miss_pct = round((c.iloc[j] / c.iloc[j - 1] - 1.0) * 100.0, 2)
                if j + 1 < len(b):
                    miss_oo = round((float(b["open"].iloc[j + 1])
                                     / float(b["open"].iloc[j]) - 1.0) * 100.0, 2)
        out.append({"predict_date": predict_date.isoformat(),
                    "verify_date": vday.isoformat(), "code": code,
                    "name": r["名称"],
                    "net_buy_yi": round(float(r["净买额"]) / 1e8, 3),
                    "verdict": verdict, "rank": rank_s, "channel": chan,
                    "miss_pct": miss_pct, "miss_oo": miss_oo, "note": ""})
    # 累计落盘（按 predict_date 幂等替换；原子写防截断）
    path = os.path.join(C.DATA_DIR, "autopsy.csv")
    old = []
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            old = [x for x in csv.DictReader(f)
                   if x["predict_date"] != predict_date.isoformat()]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=AUTOPSY_COLS)
    w.writeheader()
    w.writerows(old + out)
    fstore.write_text(path, buf.getvalue(), encoding="utf-8-sig")
    return out


def _channel_of(rk: dict) -> str:
    if rk.get("via_switch") == "True":
        return "切换通道"
    if rk.get("via_m5") == "True":
        return "动量通道"
    if rk.get("via_stir") == "True":
        return "抬头通道"
    return "强势通道"


def autopsy_summary() -> dict:
    """漏判归因汇总：席位真实买 → 命中/排序漏/宇宙漏 三分。

    机会成本对照（路线图#5）：miss_pct/miss_oo 的中位数——「漏掉的比选中的
    好多少」直接可见，是排序力的另一半证据。
    """
    path = os.path.join(C.DATA_DIR, "autopsy.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    ranks = [int(r["rank"]) for r in rows if r["verdict"] == "排序漏" and r["rank"]]
    vc = {}
    for r in rows:
        vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1

    def _med(vals):
        v = sorted(vals)
        return round(v[len(v) // 2], 2) if v else None

    miss_pcts = [float(r["miss_pct"]) for r in rows
                 if r.get("miss_pct") not in ("", None)]
    miss_oos = [float(r["miss_oo"]) for r in rows
                if r.get("miss_oo") not in ("", None)]
    return {"n_buys": len(rows), "split": vc,
            "missed_rank_median": sorted(ranks)[len(ranks) // 2] if ranks else None,
            "miss_pct_median": _med(miss_pcts),
            "miss_oo_median": _med(miss_oos),
            "miss_n": len(miss_oos),
            "by_date": {d: sum(1 for r in rows if r["predict_date"] == d)
                        for d in sorted({r["predict_date"] for r in rows})}}


def nowcast(today: dt.date | None = None,
            max_hold_days: int = 10) -> list[dict]:
    """紫阳当前在持清单（R6a，零模型产品）：净买未配对 + 持有≤N 交易日可视窗。

    deep4 落地席口径：产品中位 10 张/日；21% 处兑现风险窗（持有1~2日）。
    浮动盈亏为收盘价近似。
    """
    today = today or dt.date.today()
    st = pd.read_csv(C.SEAT_TRADES_CSV, dtype={"股票代码": str}, encoding="utf-8-sig")
    st["交易日期"] = pd.to_datetime(st["交易日期"]).dt.date
    g = (st.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(名称=("股票名称", "last"), 净=("买入金额", "max")))
    g["净"] = g["净"] - st.groupby(["交易日期", "股票代码"])["卖出金额"].max().fillna(0).values
    dates_seq = sorted(g["交易日期"].unique())
    held: dict[str, dict] = {}
    for r in g.sort_values("交易日期").itertuples(index=False):
        code = r.股票代码
        if r.净 > 0:
            held[code] = {"code": code, "name": r.名称, "buy_date": r.交易日期,
                          "net_yi": round(r.净 / 1e8, 2)}
        elif code in held and r.净 < 0:
            del held[code]
    bars_cache = load_bars_cache()
    out = []
    for code, h in held.items():
        hold_days = len([d for d in dates_seq if h["buy_date"] <= d <= today])
        if hold_days > max_hold_days:
            continue
        b = bars_cache.get(code)
        pnl = None
        if b is not None:
            dates = [str(x)[:10] for x in b["date"].tolist()]
            iso = today.isoformat()
            i = dates.index(iso) if iso in dates else (len(dates) - 1)
            j = next((k for k, d in enumerate(dates)
                      if d >= h["buy_date"].isoformat()), None)
            if j is not None and j <= i:
                c0 = float(b["close"].iloc[j - 1]) if j > 0 else None
                if c0:
                    pnl = round((float(b["close"].iloc[i]) / c0 - 1.0) * 100.0, 1)
        out.append(dict(h, hold_days=hold_days, pnl=pnl,
                        risk_window="Y" if hold_days <= 2 else ""))
    out.sort(key=lambda x: x["buy_date"])
    return out
