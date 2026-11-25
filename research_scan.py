# -*- coding: utf-8 -*-
"""近期窗口K线事件研究：纯日K扫描近 N 个月的 2 板事件，扩充因子研究样本。

定位（研究工具，同 calibrate/backfill 家族，不进生产链）：
- 池接口只保留 ~3 周，样本天花板 106；本工具用腾讯日K（可取任意区间）在
  **近期窗口**（默认 12 个月，锚点冻结，见 window_start）扫描「连续两日收盘
  涨幅≥9.5%」事件，样本扩至数千，供因子分割验证 / IC 分析 / 权重校验。
- 特征均为决策时点可得（K线口径）：低价、准一字、高开档、缩量板、首板一字、
  连板数、事件日开盘缺口、量比、星期；前向收益用可成交口径（次日开→收，
  一字开盘不可成交样本剔除）+ 理论口径（收→收/晋级）。
- **统计纪律与 calibrate 同源**（2026-09-15 起）：因子对照一律走 calib.stats
  的按日聚类配对推断 + BH-FDR——此前本模块用裸均值对比，恰好违反了
  calibrate 文档里写明的「不得再直接比点估计」纪律。
- 与 out/archive（池口径正式归档）完全分离，结果写 out/research/events.csv。

用法:
    python research_scan.py                # 增量拉K线并扫描（缓存 out/research/klines.pkl）
    python research_scan.py --months 12    # 近期窗口（默认 12）
    python research_scan.py --stats-only   # 只重算因子统计，不拉数据
    python research_scan.py --update       # 滚动增量更新（窗口锚点冻结，只向后延伸）
    python research_scan.py --update --refreeze-window   # 显式重锚（裁旧数据，先留快照）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import data_source
import fstore
from calib.stats import bh_fdr, cluster_diff, cluster_mean, two_sided_p
from config import (CALIB_MIN_INFER_DAYS, LIMIT_UP_APPROX, MAIN_BOARD_PREFIXES,
                    ONE_WORD_GAP, PROMOTE_APPROX)
from termstyle import weekday_cn

KLINES = os.path.join("out", "research", "klines.pkl")
EVENTS = os.path.join("out", "research", "events.csv")
WINDOW_META = os.path.join("out", "research", "window.json")
LIMIT_APPROX = LIMIT_UP_APPROX      # 涨停收盘判定近似（config 单一源）
MAINBOARD = MAIN_BOARD_PREFIXES     # 主板前缀（config 单一源；曾有 3 份手抄漂移风险）


def universe() -> list[tuple[str, str]]:
    """主板非ST名单（代码, 名称）：取最近一份本地快照缓存。"""
    import cache_store
    cache_dir = os.path.join("out", "cache")
    spot = None
    if os.path.isdir(cache_dir):
        for f in sorted(os.listdir(cache_dir), reverse=True):
            if not f.endswith(".pkl"):
                continue
            try:
                day = dt.datetime.strptime(f[:-4], "%Y%m%d").date()
            except ValueError:
                continue
            c = cache_store.load("out", day)
            if c.get("spot") is not None:
                spot = c["spot"]
                break
    if spot is None:
        raise RuntimeError("无本地快照缓存（先运行一次 main.py 建立名单）")
    out = []
    for _, r in spot.iterrows():
        code = str(r.get("code", "")).zfill(6)
        name = str(r.get("name", ""))
        if not code.startswith(MAINBOARD):
            continue
        if "ST" in name.upper() or "退" in name:
            continue
        out.append((code, name))
    return out


def window_start(months: int, refreeze: bool = False) -> dt.date:
    """研究窗口起点（**冻结锚点**，README 待办#6 落地）。

    首次计算 = today − months×31 并写入 sidecar（out/research/window.json）；
    此后**永远返回锚点**——update 只向后延伸，不再每日重裁窗口。
    动机：旧行为把窗口滑到运行当日，events.csv 里的历史事件会随滚动消失，
    research_log 基于它的一切结论（n=1414/3049 两批已无法复算）随之蒸发性
    失效；冻结锚点 = holdout 冻结，是待办#4（C维重校）的前置条件。
    --refreeze-window 显式重锚（会裁掉锚外旧K线，跑前必须先留快照）。
    """
    today = dt.date.today()
    anchored = None
    if os.path.exists(WINDOW_META) and not refreeze:
        try:
            with open(WINDOW_META, encoding="utf-8") as f:
                anchored = dt.datetime.strptime(
                    json.load(f).get("start", ""), "%Y-%m-%d").date()
        except Exception:
            anchored = None
    if anchored is None:
        anchored = today - dt.timedelta(days=months * 31)
        _save_window_meta(anchored, months)
        print(f"  [窗口] 锚点冻结: {anchored}（写入 {WINDOW_META}，此后不再滑动）")
    elif refreeze:
        _save_window_meta(anchored, months)
        print(f"  [窗口] 显式重锚: {anchored}（旧锚外K线将被裁剪，确认已留快照）")
    return anchored


def _save_window_meta(start: dt.date, months: int) -> None:
    """窗口锚点 sidecar 落盘（原子写：半截 json 会让锚点读不出 → 窗口悄悄滑动）。"""
    os.makedirs(os.path.dirname(WINDOW_META), exist_ok=True)
    fstore.write_text(WINDOW_META, json.dumps(
        {"start": f"{start:%Y-%m-%d}", "anchored_on": f"{dt.date.today():%Y-%m-%d}",
         "months": months}, ensure_ascii=False, indent=1))


def scan_events(df: pd.DataFrame, start: dt.date) -> list[dict]:
    """单只股票近期 2 板事件（纯函数，可离线单测）。

    事件日 = 第二个涨停日；特征全部为事件日收盘前 + 次日开盘可得信息。
    lianban/vr/gap_event 与生产口径对齐（analyzer：vr=今量/前5日均量、
    board_rules：slam 需事件日实体开盘），使 fills 回放与 EXPECT 分层可复算。
    """
    need = ["date", "open", "close", "volume"]
    if df is None or len(df) < 12 or not all(c in df.columns for c in need):
        return []
    # 一次取出整块数值矩阵，避免每只股票各做一次 DataFrame 布尔切块、reset_index
    # 和三列独立 astype。正常日K按日期升序，直接二分窗口起点并切视图；
    # 非常规乱序输入仍沿用原「过滤但保持原行序」语义。
    dates_all = list(df["date"])
    values = df[["open", "close", "volume"]].to_numpy(dtype=float, copy=False)
    ordered = all(a <= b for a, b in zip(dates_all, dates_all[1:]))
    if ordered:
        first = bisect_left(dates_all, start)
        dates = dates_all[first:]
        values = values[first:]
    else:
        keep = [d >= start for d in dates_all]
        dates = [d for d, yes in zip(dates_all, keep) if yes]
        values = values[keep]
    if len(dates) < 8:
        return []
    o, c, v = values.T
    events = []
    for i in range(6, len(dates) - 1):      # 留5日均量窗口 + 次日交易
        if c[i - 1] <= 0 or c[i - 2] <= 0 or c[i] < 2:
            continue
        r1 = c[i - 1] / c[i - 2] - 1.0      # 昨日涨停（首板）
        r2 = c[i] / c[i - 1] - 1.0          # 今日涨停（二板日）
        if r1 < LIMIT_APPROX or r2 < LIMIT_APPROX:
            continue
        # 缺口口径（2026-11-24 勘误）：gap_first 此前写成 o[i]/c[i-1]-1——那是
        # **事件日（二板日）**的缺口，与 gap_event 同式；首板日是 i-1。于是
        # 「首板一字」列实际统计的是「二板日一字开盘」（买不到的那一档），
        # 研究日志里 n=384 的首板一字结论口径名实不符。两档分列：
        #   first_word = 首板日一字（与 second_board B 维「一字」、calib「首板一字」同名）
        #   event_word = 事件日一字（买不到，仅作可成交性标注）
        gap_event = o[i] / c[i - 1] - 1.0
        gap_first = o[i - 1] / c[i - 2] - 1.0
        gap_next = o[i + 1] / c[i] - 1.0        # 次日开盘缺口（准一字/高开档）
        # 连板数：自事件日向前数连续涨停收盘（bars.approx_lianban 同口径）
        lianban = 2
        j = i - 2
        while j >= 1 and c[j - 1] > 0 and c[j] / c[j - 1] - 1.0 >= LIMIT_APPROX:
            lianban += 1
            j -= 1
        ev = {
            "date": dates[i],
            "weekday": weekday_cn(dates[i]),
            "close": float(c[i]),
            "lianban": lianban,
            "gap_event": round(float(gap_event), 4),
            "gap_first": round(float(gap_first), 4),
            "low_price": bool(c[i] < 10),
            "gap_next": round(float(gap_next), 4),
            "one_word_next": bool(gap_next >= ONE_WORD_GAP),
            "gap_best": bool(0.03 <= gap_next < 0.07),
            "first_word": bool(gap_first >= ONE_WORD_GAP),
            "event_word": bool(gap_event >= ONE_WORD_GAP),
            "ret_oc": float(c[i + 1] / o[i + 1] - 1.0),
            "ret_cc": float(c[i + 1] / c[i] - 1.0),
        }
        base = v[max(0, i - 5):i].mean()
        ev["shrink"] = bool(base > 0 and v[i] < base * 0.6)
        ev["vr"] = round(float(v[i] / base), 2) if base > 0 else None
        events.append(ev)
    return events


def _safe_save(cache: dict, old_count: int, label: str) -> bool:
    """防覆写保护：成功数过少时拒绝覆盖缓存（2026-09-04 事故：限流致 3048 只
    仅 1 只成功，旧缓存被覆盖）。阈值：成功 < max(500, 旧缓存50%) 即拒绝。"""
    floor = max(500, int(old_count * 0.5))
    if len(cache) < floor:
        print(f"  [保护] {label}仅 {len(cache)} 只成功（阈值 {floor}），"
              f"疑似限流/断网——保留旧缓存不覆盖。稍后重试。")
        return False
    os.makedirs(os.path.dirname(KLINES), exist_ok=True)
    fstore.write_pkl(KLINES, cache)     # 39MB 资产原子写：截断=全量重拉
    print(f"  已缓存 {len(cache)} 只 → {KLINES}")
    return True


def load_klines(codes: list[str], start: dt.date, refresh: bool = False) -> dict:
    cache: dict = {}
    old_count = 0
    if os.path.exists(KLINES) and not refresh:
        try:
            with open(KLINES, "rb") as f:
                cache = pickle.load(f)
            old_count = len(cache)
        except Exception:
            cache = {}
    todo = [c for c in codes if c not in cache]
    print(f"K线缓存 {len(cache)} 只，需拉取 {len(todo)} 只（腾讯，近期窗口）...")
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(data_source.fetch_close_range, c, start,
                                dt.date.today()): c for c in todo}
            for fut in as_completed(futs):
                try:
                    df = fut.result()
                except Exception:
                    df = None
                if df is not None and len(df) >= 30:
                    cache[futs[fut]] = df
                done += 1
                if done % 400 == 0 or done == len(todo):
                    print(f"  拉取进度 {done}/{len(todo)}")
        _safe_save(cache, old_count, "全量拉取")
    return cache


def update_cache(codes: list[str], start: dt.date) -> dict:
    """滚动增量更新研究缓存（**窗口锚点冻结**：start 来自 window_start，不随日历滑动）。

    - 已缓存股票：只补末根之后的尾巴（小区间请求），拼接后剔除窗口外旧K线
    - 新上市/缺失/停牌过期：全量拉取
    - 名单外（退市/转ST）：从缓存清理
    - 已是最新（当日拉过）：仅裁剪窗口
    """
    cache: dict = {}
    if os.path.exists(KLINES):
        try:
            with open(KLINES, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    uni = set(codes)
    dropped = [c for c in list(cache) if c not in uni]
    for c in dropped:
        del cache[c]
    today = dt.date.today()
    win_start = start

    def to_date(x):
        return x if isinstance(x, dt.date) else pd.to_datetime(x).date()

    todo_full, todo_tail, fresh = [], [], []
    for c in uni:
        df = cache.get(c)
        if df is None or len(df) < 30:
            todo_full.append(c)
            continue
        last = to_date(df["date"].iloc[-1])
        if last >= today:
            fresh.append(c)                     # 今天已拉过：只裁窗口
        else:
            todo_tail.append((c, last))
    print(f"滚动更新: 补尾 {len(todo_tail)} 只，全量 {len(todo_full)} 只（新/缺），"
          f"已是最新 {len(fresh)} 只，剔除名单外 {len(dropped)} 只"
          f"（窗口锚点 {win_start}，不滑动）")

    def trim(df):
        return df[df["date"] >= win_start].reset_index(drop=True)

    for c in fresh:
        cache[c] = trim(cache[c])
    done = 0
    total = len(todo_full) + len(todo_tail)
    if total:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {}
            for c in todo_full:
                futs[pool.submit(data_source.fetch_close_range, c, win_start,
                                 today)] = (c, None)
            for c, last in todo_tail:
                futs[pool.submit(data_source.fetch_close_range, c,
                                 last + dt.timedelta(days=1),
                                 today)] = (c, last)
            for fut in as_completed(futs):
                c, last = futs[fut]
                try:
                    df = fut.result()
                except Exception:
                    df = None
                if df is not None and len(df):
                    if last is None:                    # 全量
                        if len(df) >= 30:
                            cache[c] = trim(df)
                    else:                               # 补尾拼接
                        add = df[df["date"] > last]
                        if len(add):
                            merged = pd.concat([cache[c], add], ignore_index=True)
                            cache[c] = trim(merged.sort_values("date"))
                done += 1
                if done % 400 == 0 or done == total:
                    print(f"  更新进度 {done}/{total}")
        if _safe_save(cache, max(len(cache) - done, 0), "增量更新"):
            newest = max((to_date(df["date"].iloc[-1]) for df in cache.values()
                          if df is not None and len(df)), default=None)
            print(f"  最新数据至 {newest}，共 {len(cache)} 只")
    return cache


def scan_all(klines: dict, name_of: dict, start) -> pd.DataFrame:
    """全缓存事件扫描并落盘 EVENTS。"""
    rows = []
    for code, df in klines.items():
        for ev in scan_events(df, start):
            ev["code"] = code
            ev["name"] = name_of.get(code, "")
            rows.append(ev)
    out = pd.DataFrame(rows)
    if out.empty:
        print("未扫描到事件。")
        return out
    out = out.sort_values("date").reset_index(drop=True)
    os.makedirs(os.path.dirname(EVENTS), exist_ok=True)
    fstore.write_dataframe_csv(EVENTS, out)     # 原子写：半截 events.csv 会让 --stats-only 读崩
    print(f"事件 {len(out)} 个 → {EVENTS}（{out['date'].min()} ~ {out['date'].max()}）")
    return out


def build(months: int, refresh: bool = False) -> pd.DataFrame:
    start = window_start(months)
    codes_names = universe()
    print(f"近期窗口: {start} ~ 今天（锚点冻结），名单 {len(codes_names)} 只主板非ST")
    klines = load_klines([c for c, _ in codes_names], start, refresh=refresh)
    return scan_all(klines, dict(codes_names), start)


# ---------------------------------------------------------------- 因子统计
# 2026-09-15 起接入 calib.stats：此前是裸 i.i.d. 均值对比（无聚类 SE、无 FDR），
# 而 events.csv 全窗口上千事件只跨两百多个独立交易日——正是 cluster 推断
# 要防的伪显著。输出纪律与 calibrate 一致：配对日 < CALIB_MIN_INFER_DAYS 一律
# 「无法推断」；多重比较 BH-FDR（共线特征合并 m_eff）。

_COLLINEAR = [{"one_word_next", "gap_best"}]   # 同为次日缺口档位，非独立检验


def _recs_for(df: pd.DataFrame) -> list[dict]:
    """df 子集 → cluster_diff/cluster_mean 可消费的行（date 归一 YYYY-MM-DD）。"""
    return [{"date": str(r["date"])[:10], "ret_oc": r["ret_oc"]}
            for _, r in df[["date", "ret_oc"]].iterrows()]


def stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fillable = df[~df["one_word_next"]] if "one_word_next" in df.columns else df
    days_all = df["date"].astype(str).str[:10].nunique()
    print(f"\n=== 近期窗口因子统计（n={len(df)} / {days_all} 日簇，"
          f"前向=次日开→收·可成交子样本 n={len(fillable)}，一字开盘买不到已剔除）===")
    feats = {
        "准一字": df["one_word_next"], "高开3-7%": df["gap_best"],
        "缩量板": df["shrink"], "低价<10": df["low_price"],
        "首板一字": df["first_word"],
    }
    if "lianban" in df.columns:
        feats["3板+"] = df["lianban"] >= 3
    not_one_word = (~df["one_word_next"] if "one_word_next" in df.columns
                    else pd.Series(True, index=df.index))
    recs = []
    for name, mask in feats.items():
        inside, outside = df[mask], df[~mask]
        if len(inside) < 5 or len(outside) < 5:
            print(f"  {name:<8} n={len(inside)} 不足")
            continue
        if name == "准一字":
            # 组内即「次日一字」全体，可成交口径无样本——只给信息口径
            pi = (inside["ret_cc"] >= PROMOTE_APPROX).mean()
            po = (outside["ret_cc"] >= PROMOTE_APPROX).mean()
            print(f"  {name:<8} n={len(inside):<4} [信息口径] 晋级{pi:.0%} vs 组外{po:.0%}"
                  f" —— 一字开盘排队买不到，开→收口径不适用（五T 结论）")
            continue
        fi, fo = df[mask & not_one_word], df[~mask & not_one_word]
        if len(fi) < 5 or len(fo) < 5:
            print(f"  {name:<8} n_in={len(fi)} 可成交子样本不足")
            continue
        d = cluster_diff(_recs_for(fi), _recs_for(fo), "ret_oc", "date")
        mi, mo = fi["ret_oc"].mean(), fo["ret_oc"].mean()
        pi = (inside["ret_cc"] >= PROMOTE_APPROX).mean()
        po = (outside["ret_cc"] >= PROMOTE_APPROX).mean()
        recs.append({"key": name, "n_in": len(inside), "n_fill": len(fi),
                     "in": mi, "out": mo,
                     # Δ 与 t 同源：此前 Δ 用未配对均值差 mi−mo、t/p/q 用按日配对，
                     # 两者可以异号（表里出现 Δ>0 而 t<0）
                     "delta": d["diff"] if d["diff"] is not None else mi - mo,
                     "t": d["t"], "days": d["days"], "p": two_sided_p(d["t"]),
                     "promo_i": pi, "promo_o": po})
    if recs:
        tested = {r["key"] for r in recs}
        merged = sum(max(0, len(g & tested) - 1) for g in _COLLINEAR)
        m_eff = max(1, len(recs) - merged)
        for r, q in zip(recs, bh_fdr([r["p"] for r in recs], m_eff)):
            r["q"] = q
        print(f"\n  {'特征':<8}{'n':>5}{'可成交':>7}{'组内开收%':>10}{'组外%':>8}"
              f"{'Δpp':>8}{'配对日':>7}{'聚类t':>8}{'p':>7}{'q(BH)':>7}{'[参考]晋级':>12}")
        print("  " + "-" * 88)
        for r in sorted(recs, key=lambda x: -abs(x["t"] or 0)):
            t_s = f"{r['t']:+.2f}" if r["t"] is not None else "单簇"
            p_s = f"{r['p']:.3f}" if r["p"] is not None else "—"
            q_s = f"{r['q']:.3f}" if r["q"] is not None else "—"
            # 配对日不足门槛的行不打 ★（此前只要 q<0.10 就标，与文档「无法推断」不符）
            sig = r["q"] is not None and r["q"] < 0.10
            mark = (" ★" if sig and r["days"] >= CALIB_MIN_INFER_DAYS
                    else " (配对日不足)" if sig else "")
            print(f"  {r['key']:<8}{r['n_in']:>5}{r['n_fill']:>7}"
                  f"{r['in'] * 100:>+10.2f}{r['out'] * 100:>+8.2f}"
                  f"{r['delta'] * 100:>+8.2f}{r['days']:>7}{t_s:>8}{p_s:>7}"
                  f"{q_s:>7}{r['promo_i']:>7.0%}/{r['promo_o']:<4.0%}{mark}")
        print("  " + "-" * 88)
        print(f"  （BH-FDR：原始 {len(recs)} 检验，共线合并 {merged} → m_eff={m_eff}；"
              f"Δ=按日配对均值差（与 t 同源）；★=q<0.10 且配对日≥{CALIB_MIN_INFER_DAYS}；"
              f"(配对日不足)=q<0.10 但配对日<{CALIB_MIN_INFER_DAYS}，不判读）")

    # 星期效应：按日聚类的组内均值
    if "weekday" in df.columns:
        print("\n按星期（可成交子样本，聚类t）:")
        for wd, g in fillable.groupby("weekday"):
            st = cluster_mean(_recs_for(g), "ret_oc", "date")
            t_s = f"{st['t']:+.2f}" if st["t"] is not None else "单簇"
            print(f"  {wd} n={len(g):<5} 开→收{st['mean'] * 100:+.2f}% t={t_s}"
                  f"（{st['clusters']} 日簇）")

    # 首板一字分割验证（近窗口内部再对半）——稳定性检查，不是新检验族
    if "first_word" in df.columns:
        mid = df["date"].quantile(0.5)
        for lab, seg in (("前半", df[df["date"] <= mid]), ("后半", df[df["date"] > mid])):
            i, o = seg[seg["first_word"]], seg[~seg["first_word"]]
            if len(i) < 5:
                print(f"首板一字 {lab}: n={len(i)} 不足")
                continue
            fi, fo = i[~i["one_word_next"]], o[~o["one_word_next"]]
            if len(fi) < 5 or len(fo) < 5:
                print(f"首板一字 {lab}: n_in={len(i)} 可成交子样本不足")
                continue
            d = cluster_diff(_recs_for(fi), _recs_for(fo), "ret_oc", "date")
            t_s = f"{d['t']:+.2f}" if d["t"] is not None else "单簇"
            print(f"首板一字 {lab}: n_in={len(i)} Δ开收"
                  f"{(fi['ret_oc'].mean() - fo['ret_oc'].mean()) * 100:+.2f}pp "
                  f"配对{d['days']}日 t={t_s}")


def main():
    ap = argparse.ArgumentParser(description="近期窗口K线事件研究（样本扩充）")
    ap.add_argument("--months", type=int, default=12,
                    help="窗口月数（默认12=项目研究口径；起点首次运行后冻结，不再滑动）")
    ap.add_argument("--refresh", action="store_true", help="强制重拉K线")
    ap.add_argument("--update", action="store_true",
                    help="滚动增量更新缓存（补尾/新股/退市；窗口锚点冻结只向后延伸）后重扫——每日收盘链用")
    ap.add_argument("--refreeze-window", action="store_true",
                    help="显式重锚窗口起点（会裁掉锚外旧K线，跑前先在 snapshots/ 留当日快照）")
    ap.add_argument("--stats-only", action="store_true", help="只重算统计")
    ap.add_argument("--price-check", action="store_true",
                    help="低价口径修复：取不复权K线算复权因子，修正事件日 close 后重验低价因子")
    args = ap.parse_args()
    if args.price_check:
        price_check()
        return
    if args.update:
        codes_names = universe()
        start = window_start(args.months, refreeze=args.refreeze_window)
        klines = update_cache([c for c, _ in codes_names], start)
        df = scan_all(klines, dict(codes_names), start)
        stats(df)
        return
    if args.stats_only and os.path.exists(EVENTS):
        stats(pd.read_csv(EVENTS, dtype={"code": str}, parse_dates=["date"]))
        return
    df = build(args.months, refresh=args.refresh)
    stats(df)


def price_check():
    """低价矛盾修复：前复权把除权股历史收盘压低 → close<10 误判。

    取不复权（fqt=0）K线最后一根收盘算复权因子 f = raw/qfq（窗口内近似常数），
    修正事件日 close 后重验低价因子 vs 池口径（IC +0.250*）。
    """
    import requests
    events = pd.read_csv(EVENTS, dtype={"code": str}, parse_dates=["date"])
    with open(KLINES, "rb") as f:
        klines = pickle.load(f)
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    factors = {}
    codes = sorted(set(events["code"]))
    print(f"取不复权收盘（{len(codes)} 只事件股，腾讯）...")
    for n, code in enumerate(codes):
        if code in factors:
            continue
        symbol = ("sh" if code.startswith(("6", "9")) else "sz") + code
        try:
            r = requests.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                             params={"param": f"{symbol},day,,5,5,"},
                             headers=headers, timeout=6)
            node = (r.json().get("data") or {}).get(symbol, {})
            raw_rows = node.get("day") or []
            qfq = klines.get(code)
            if not raw_rows or qfq is None or len(qfq) < 2:
                factors[code] = 1.0
            else:
                raw_last = float(raw_rows[-1][2])
                qfq_last = float(qfq["close"].iloc[-1])
                factors[code] = raw_last / qfq_last if qfq_last > 0 else 1.0
        except Exception:
            factors[code] = 1.0
        if (n + 1) % 100 == 0:
            print(f"  进度 {n + 1}/{len(codes)}")
    events["close_adj"] = events.apply(
        lambda r: r["close"] * factors.get(r["code"], 1.0), axis=1)
    raw_low = events["close_adj"] < 10
    qfq_low = events["close"] < 10
    print("")
    print("低价口径对照（n=%d）：" % len(events))
    print(f"  前复权 close<10: {qfq_low.sum()} 只（{qfq_low.mean():.0%}）")
    print(f"  不复权修正后  <10: {raw_low.sum()} 只（{raw_low.mean():.0%}）")
    print(f"  因子≠1（有除权）的股票: {sum(1 for v in factors.values() if abs(v - 1) > 0.01)} 只")
    for label, mask in (("前复权口径", qfq_low), ("不复权修正", raw_low)):
        i, o = events[mask], events[~mask]
        if len(i) < 5:
            continue
        print(f"  {label}: 低价组收→收 {i['ret_cc'].mean()*100:+.2f}% vs {o['ret_cc'].mean()*100:+.2f}% "
              f"Δ{(i['ret_cc'].mean()-o['ret_cc'].mean())*100:+.2f}pp")


if __name__ == "__main__":
    main()
