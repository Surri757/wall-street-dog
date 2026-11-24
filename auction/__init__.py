# -*- coding: utf-8 -*-
"""竞价采集（auction）—— 策略池「竞价族」前向验证的数据底座（旁路子包）。

背景（research_log 五Z）：外部策略池里最有辨识度的一族是「T+1 集合竞价」条件
（竞价量 ≥ 昨日总量 3%、竞价高开区间、炸板弱转强）。历史竞价数据不可回补
（腾讯分时只保留近 5 个交易日），只能前向采集——本模块每个交易日收盘后，把
**信号日（上一交易日）封板/炸板股**的当日分时落盘：
  out/research/auction/ledger/<执行日>.csv   按日分片台账：竞价点 + 执行日摘要
  out/research/auction/minute/<执行日>.pkl   执行日分钟路径原料（真实实时量比研究的备料）

落盘纪律（2026-11-24 真实数据沙箱验证后定稿）：
  - **按日分片、整天成功才落盘**：任一请求失败或分时被截断 → 该执行日不写盘，
    保留期内下次运行自动重试；不存在"某只票静默缺失"的半截台账；
  - 分片各自原子写（fstore），无"读全表→改→写全表"，并发运行不互相覆盖；
  - 执行日无分时的票（停牌）如实跳过；上游对该代码无数据的，同样跳过不报错。

定位：**只读外部行情 + 写研究台账**。不进报告、不进评分、不改明日关注清单；
假设与判定线已预登记（PREREG 七），样本达标前不做任何判读。

━━━ 已接入主工程 ━━━
    work / python main.py 复盘最新交易日时自动调用 daily_sync()：与 vane 同款守卫
    （历史日期 / 盘中跳过；失败只打一行警告，不阻断主流程）。独立 CLI:
    python -m auction collect [--date] | report
"""
from __future__ import annotations

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fstore
import pandas as pd

from trade_rules import limit_price

from . import config as C

__all__ = ["daily_sync", "collect", "report"]
__version__ = "1.1.0"


def _f(v):
    """宽松转 float；NaN/非法 → None（台账空值统一为空，不写 nan 字样）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _num(row: dict, key: str):
    """台账行取数：数值且非 NaN 才返回，否则 None。"""
    v = row.get(key)
    return v if isinstance(v, (int, float)) and v == v else None


def _archive_path(day: dt.date) -> str:
    return os.path.join(C.ARCHIVE_DIR, f"signals_{day:%Y%m%d}.csv")


def _day_path(exec_date: dt.date) -> str:
    return os.path.join(C.LEDGER_DIR, f"{exec_date:%Y-%m-%d}.csv")


def _signal_rows(signal_date: dt.date) -> pd.DataFrame:
    """信号日归档信号里的封板/炸板行（名单来源=正式归档，与报告同口径）；无归档 → 空表。"""
    path = _archive_path(signal_date)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    df = df[df["zt_status"].isin(["封板", "炸板"])].copy()
    df["group"] = df["zt_status"]
    return df


def _collected_days() -> set[str]:
    """已落盘的执行日（分片文件名即键；分片只在整天成功后写入，存在即完整）。

    只认 `YYYY-MM-DD.csv` 精确形态：fstore 的 `*.tmp.csv` 中途残留若被当成
    执行日，该日就永不重采。
    """
    if not os.path.isdir(C.LEDGER_DIR):
        return set()
    out = set()
    for n in os.listdir(C.LEDGER_DIR):
        if not n.endswith(".csv"):
            continue
        try:
            out.add(f"{dt.datetime.strptime(n[:-4], '%Y-%m-%d').date():%Y-%m-%d}")
        except ValueError:
            continue
    return out


def _load() -> list[dict]:
    """读全部执行日分片（按日期升序）。单个分片损坏只跳过并留痕，不拖垮整体读数。"""
    rows = []
    for day in sorted(_collected_days()):
        try:
            df = pd.read_csv(_day_path(dt.date.fromisoformat(day)),
                             dtype={"code": str, "first_limit_time": str},
                             encoding="utf-8-sig")
            rows += df.to_dict("records")
        except Exception as e:                                 # noqa: BLE001
            print(f"[竞价采集] 台账分片损坏，已跳过 {day}.csv（{type(e).__name__}）")
    return rows


def _clean(rec: dict) -> dict:
    """按 COLS 定列序；NaN/缺失 → 空串（csv.DictWriter 不认识 nan）。"""
    return {k: ("" if rec.get(k) is None
                or (isinstance(rec.get(k), float) and rec[k] != rec[k])
                else rec[k]) for k in C.COLS}


def _hhmm(t: str) -> str:
    """'0930' → '09:30'（不带冒号的串会被表格软件/无类型读取当成数字 930）。"""
    return f"{t[:2]}:{t[2:]}" if len(t) == 4 else t


def summarise(exec_date: dt.date, signal_date: dt.date, row: dict,
              ex: list, sg: list | None) -> dict:
    """一只票：执行日分时 ex + 信号日分时 sg（可无）→ 台账行。

    - 竞价点 = 执行日第一个点（0930）：价=开盘价、量/额=集合竞价成交量/成交额；
    - vol_t = 信号日最后一个点的累计量（手）→ auction_vol_ratio 与原帖定义
      「竞价成交量 / 昨日成交量」同口径（不用成交额近似）；信号日已出保留窗
      时（sg 为空）vol_t/占比留空，其余照记。
    - 触板/封板以**执行日**涨停价判定（基准=信号日收盘，trade_rules.limit_price）；
      gap_vs_limit 用**信号日**涨停价（A3 弱转强口径，基准=信号日昨收，缺失则留空）。
    """
    _, open_px, a_vol, a_amt = ex[0]
    vol_t, amt_t = (sg[-1][2], sg[-1][3]) if sg else (None, None)
    close_t, prev_close = _f(row.get("close")), _f(row.get("prev_close"))
    limit_t = limit_price(prev_close) if prev_close else None   # 信号日涨停价
    limit_x = limit_price(close_t) if close_t else None         # 执行日涨停价
    prices = [p[1] for p in ex]
    high_exec, close_exec = max(prices), prices[-1]
    touched = [p[0] for p in ex if limit_x and p[1] >= limit_x - 1e-9]
    lianban = _f(row.get("lianban"))
    return _clean({
        "exec_date": f"{exec_date:%Y-%m-%d}",
        "signal_date": f"{signal_date:%Y-%m-%d}",
        "code": str(row.get("code") or "").zfill(6),
        "name": str(row.get("name") or ""),
        "group": str(row.get("group") or ""),
        "lianban": int(lianban) if lianban is not None else "",
        "close_t": close_t, "prev_close": prev_close, "limit_t": limit_t,
        "vol_ratio_t": _f(row.get("vol_ratio")),
        "board_score_t": _f(row.get("board_score")),
        "open_exec": open_px,
        "gap": (open_px / close_t - 1.0) if close_t else None,
        "gap_vs_limit": (open_px / limit_t - 1.0) if limit_t else None,
        "auction_vol": a_vol, "auction_amount": a_amt,
        "vol_t": vol_t, "amount_t": amt_t,
        "auction_vol_ratio": (a_vol / vol_t) if vol_t else None,
        "high_exec": high_exec, "close_exec": close_exec,
        "pct_close": (close_exec / close_t - 1.0) if close_t else None,
        "touch_limit": bool(touched),
        "first_limit_time": _hhmm(touched[0]) if touched else "",
        "sealed": bool(limit_x and close_exec >= limit_x - 1e-9),
        "ret_oc": (close_exec / open_px - 1.0) if open_px > 0 else None,
        "fetched_at": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
    })


def collect(exec_date: dt.date, signal_date: dt.date, rows=None,
            workers: int = 4) -> dict:
    """采集一个执行日（幂等：重跑 = 整日分片替换）。

    rows 缺省取信号日归档的封板/炸板行（测试可注入）。返回计数 dict；
    含 error 键 = **未落盘**（保留期内下次运行自动重试）。
    """
    from data_source import fetch_minute_days

    if rows is None:
        rows = _signal_rows(signal_date)
    if rows is None or not len(rows):
        return {"error": f"信号日 {signal_date} 无封板/炸板归档"}
    codes = sorted({str(c).zfill(6) for c in rows["code"]})
    series: dict = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(fetch_minute_days, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                series[code] = fut.result()
            except Exception:                               # noqa: BLE001 瞬时失败：整天不落盘
                failed += 1
    ex_key, sig_key = f"{exec_date:%Y%m%d}", f"{signal_date:%Y%m%d}"
    new, minute, truncated = [], {}, 0
    for row in rows.to_dict("records"):
        code = str(row.get("code") or "").zfill(6)
        days = series.get(code)
        if not days:
            continue                       # 停牌 / 上游无该代码数据：如实跳过
        by = {d["date"]: d for d in days}
        ex = by.get(ex_key)
        if ex is None:
            continue                       # 执行日无分时（当日停牌）
        if not ex["complete"]:
            truncated += 1                 # 盘中 / 截断：不能当完整交易日
            continue
        sg = by.get(sig_key)
        new.append(summarise(exec_date, signal_date, row, ex["points"],
                             sg["points"] if sg else None))
        minute[code] = {"signal_date": f"{signal_date:%Y-%m-%d}",
                        "points": ex["points"]}
    if failed or truncated:
        return {"error": (f"{failed} 只请求失败 / {truncated} 只分时截断，"
                          f"执行日 {exec_date} 不落盘（保留期内下次自动重试）"),
                "failed": failed, "truncated": truncated, "rows": len(new)}
    fstore.write_csv(_day_path(exec_date), list(C.COLS), new)
    if minute:
        fstore.write_pkl(os.path.join(C.MINUTE_DIR, f"{exec_date:%Y%m%d}.pkl"), minute)
    return {"rows": len(new), "codes": len(codes),
            "skipped": len(codes) - len(minute)}


def daily_sync(trade_date: dt.date | None = None, calendar: list | None = None,
               now: dt.datetime | None = None) -> str:
    """当日同步：保留期内「有信号日归档但未采集」的执行日逐日补齐（幂等，可重跑）。

    守卫（vane 同款）：trade_date 非最新已收盘交易日（历史查看）或盘中
    （< 收盘缓冲）→ 跳过。calendar 建议由入口层传入（main/work 已拉过，免重复请求）。
    """
    from config import AFTER_CLOSE_BUFFER_MINUTES
    from data_source import (AUCTION_RETENTION_DAYS, resolve_trade_date,
                             trade_calendar)

    cal = calendar if calendar is not None else trade_calendar()
    if not cal:
        return "竞价采集跳过（交易日历不可用，无法确定信号日）"
    anchor = resolve_trade_date(None, calendar=cal)
    trade_date = trade_date or anchor
    if trade_date != anchor:
        return f"竞价采集跳过（{trade_date} 非最新已收盘交易日）"
    now = now or dt.datetime.now()
    if (trade_date == now.date()
            and now.time() < dt.time(15, AFTER_CLOSE_BUFFER_MINUTES)):
        return "竞价采集跳过（盘中，收盘后运行）"
    done = _collected_days()
    # 上游保留 AUCTION_RETENTION_DAYS 个交易日：取其中后 N−1 个，保证上一交易日
    # （信号日）仍在返回范围内。N 来自 datasources/auction（此前这里硬编码 4，
    # 与上游保留窗常量无关联）
    recent = [d for d in cal if d <= trade_date][-(AUCTION_RETENTION_DAYS - 1):]
    msgs, stuck = [], []
    for d in recent:
        key = f"{d:%Y-%m-%d}"
        if key in done:
            continue
        prior = [x for x in cal if x < d]
        if not prior or not os.path.exists(_archive_path(prior[-1])):
            continue
        res = collect(d, prior[-1])
        if res.get("error"):
            stuck.append(f"{d:%m-%d}（{res['error']}）")
            continue
        msgs.append(f"{d:%m-%d} {res['rows']}行")
    if stuck:
        return "竞价采集⚠ " + "｜".join(stuck) + "｜" + ("；".join(msgs) or "无新增")
    if not msgs:
        return f"竞价采集：保留期内无待采执行日（台账 {len(done)} 个执行日）"
    return f"竞价采集：{'｜'.join(msgs)}｜台账 out/research/auction/ledger/"


# ---------------------------------------------------------------- 判读（零网络）

def _tests(rows: list[dict]) -> list[tuple]:
    """预登记三个假设（PREREG 七）：(名称, 宇宙, 条件)。

    宇宙先剔除该假设的字段缺失行：昨收缺失的行（历史池补录日）无信号日
    涨停价 → 整行排除出 A3；竞价量/缺口缺失的行同样排除出 A1/A2——「未知」
    不是「不满足」，缺失行落进对照组只会稀释对照组（此前如此）。
    """
    sealed = [r for r in rows if r.get("group") == "封板"]
    sealed_ratio = [r for r in sealed
                    if _num(r, "auction_vol_ratio") is not None]
    first_board = [r for r in sealed if _num(r, "lianban") == 1
                   and _num(r, "gap") is not None]
    broken = [r for r in rows if r.get("group") == "炸板"
              and _num(r, "gap_vs_limit") is not None]
    lo, hi = C.GOLDEN_BAND
    rlo, rhi = C.RZQ_BAND
    return [
        ("A1 竞价量占比≥3%", sealed_ratio,
         lambda r: _num(r, "auction_vol_ratio") >= C.AUCTION_VOL_RATIO),
        ("A2 黄金分割高开3.82~6.18%", first_board,
         lambda r: lo <= _num(r, "gap") <= hi),
        ("A3 弱转强(炸板组 开盘/涨停价98~109%)", broken,
         lambda r: rlo <= _num(r, "gap_vs_limit") <= rhi),
    ]


def report() -> str:
    """预登记分桶读数（零网络）。未达判定线只报样本量，不做判读。"""
    from calib.stats import bh_fdr, cluster_diff, two_sided_p

    rows = [r for r in _load() if _num(r, "ret_oc") is not None]
    n_sealed = sum(1 for r in rows if r.get("group") == "封板")
    days = sorted({r.get("exec_date") for r in rows})
    L = ["━━━ 竞价采集 · 预登记读数（零网络） ━━━",
         f"台账: {len(rows)} 行 / {len(days)} 个执行日"
         + (f"（{days[0]} ~ {days[-1]}）" if days else "")
         + f"｜封板 {n_sealed} · 炸板 {len(rows) - n_sealed}"]
    if not rows:
        L.append("  尚无样本：每日盘后由 main/work 自动采集，或 python -m auction collect")
        return "\n".join(L)

    def judge(uni: list[dict], cond) -> dict:
        a = [r for r in uni if cond(r)]
        b = [r for r in uni if not cond(r)]
        rec = {"n_in": len(a), "n_out": len(b),
               "in": (sum(_num(r, "ret_oc") for r in a) / len(a)) if a else None,
               "out": (sum(_num(r, "ret_oc") for r in b) / len(b)) if b else None}
        if len(a) < C.MIN_N or len(b) < C.MIN_N:
            return rec
        d = cluster_diff(a, b, "ret_oc", "exec_date")
        # p 按 Student t（自由度=配对日−1）：预登记的判定线下配对日只有个位数，
        # 正态近似会把 2.0 的 t 读成 p=0.046，实际（df=7）是 0.08
        rec.update(delta=d["diff"], t=d["t"], days=d["days"],
                   p=two_sided_p(d["t"], df=d["days"] - 1))
        mid = sorted(r["exec_date"] for r in uni)[len(uni) // 2]
        halves = []
        for part in (lambda r: r["exec_date"] <= mid, lambda r: r["exec_date"] > mid):
            pa, pb = [r for r in a if part(r)], [r for r in b if part(r)]
            if len(pa) >= C.MIN_N and len(pb) >= C.MIN_N:
                halves.append(cluster_diff(pa, pb, "ret_oc", "exec_date")["diff"])
        rec["halves"] = halves
        return rec

    recs = [(name, judge(uni, cond)) for name, uni, cond in _tests(rows)]
    # 检验族 = 预登记的 3 个假设：m 固定为 3（此前取「有 p 的个数」，样本没攒齐
    # 的假设不占名额，校正反而偏松）
    for (_, r), q in zip(recs, bh_fdr([r.get("p") for _, r in recs], m=len(recs))):
        r["q"] = q
        # 预登记七：样本不足只报样本量、不做判读——配对日不足 MIN_DAYS 时
        # 此前照样落「否决」
        if "delta" not in r or r["days"] < C.MIN_DAYS or r["q"] is None:
            r["verdict"] = "样本积累中"
            continue
        h = r["halves"]
        stable = (len(h) == 2 and h[0] is not None and h[1] is not None
                  and h[0] * h[1] > 0)
        r["verdict"] = ("过线" if (r["q"] is not None and r["q"] < C.Q_MAX
                                   and abs(r["delta"]) >= C.MIN_DELTA
                                   and r["in"] > C.MIN_MEAN and stable
                                   and r["days"] >= C.MIN_DAYS)
                        else "追踪" if stable else "否决")
    for name, r in recs:
        if "delta" not in r:
            L.append(f"  {name}: 样本积累中（组内 {r['n_in']} / 组外 {r['n_out']}，"
                     f"各需 ≥{C.MIN_N}）")
            continue
        h = r["halves"]
        hs = "/".join(f"{x * 100:+.2f}" for x in h) if len(h) == 2 else "—"
        t_s = f"{r['t']:+.2f}" if r["t"] is not None else "单簇"
        q_s = f"{r['q']:.3f}" if r["q"] is not None else "—"
        note = ("" if r["days"] >= C.MIN_DAYS
                else f"（配对日 {r['days']}<{C.MIN_DAYS}，不判读）")
        L.append(f"  {name}: n {r['n_in']}/{r['n_out']}｜开→收 "
                 f"{r['in'] * 100:+.2f}% / {r['out'] * 100:+.2f}%｜Δ {r['delta'] * 100:+.2f}pp"
                 f"｜t {t_s}｜q {q_s}｜前后半Δ {hs}｜{r['verdict']}{note}")
    L.append(f"（口径：ret_oc=执行日开盘(竞价价)→收盘；过线须 q<{C.Q_MAX}（BH，检验族 m=3）、"
             f"|Δ|≥{C.MIN_DELTA * 100:.0f}pp、组内>{C.MIN_MEAN * 100:.1f}%、"
             f"前后半同号、配对日≥{C.MIN_DAYS}；p 按 Student t；预登记见 PREREG 七）")
    return "\n".join(L)
