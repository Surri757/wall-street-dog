# -*- coding: utf-8 -*-
"""交易日复盘流水线：把 collect_data 的五个阶段拆为显式步骤（main/work 仅做入口编排）。

阶段（每步单一职责、可独立测试）：
  A 缓存命中检查        load_cached_result
  B 日K缓存播种          seed_bars_from_prev
  C 快照获取（含保护）    acquire_spot      —— 盘中保护 / 历史只读缓存
  D 候选池构建           candidates_from_spot
  E 市场环境装配         assemble_market_context —— 池→行业兜底→概念→赚钱效应→情绪周期
  F 分析 + 落盘          run_day（编排 A~E + analyzer.analyze_all + cache_store.save_result）
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib
import os

import pandas as pd

import cache_store
import fstore
import mood
from analyzer import analyze_all
from bars import drop_intraday_last
from candidates import build_candidates
from config import MAIN_BOARD_PREFIXES, MIN_BARS
from data_source import (cache_is_valid, fetch_concept_boards, fetch_spot,
                         fetch_pool_extras, fill_industry_local)

# 旁路子包 → (报告里的简称, 是否接收入口层已拉的交易日历)；顺序即同步顺序
_BYPASS_MODULES = (("紫阳", "ziyang", False),      # 自带 baostock 磁盘日历
                   ("风向标", "vane", True),
                   ("竞价采集", "auction", True))


def sync_bypass_modules(trade_date: dt.date, calendar=None) -> None:
    """旁路子包同步（报告渲染前）：紫阳甄选 → 情绪风向标 → 竞价采集。

    main 与 work 的**唯一挂点**：此前两个入口各手抄一份 try/except 清单，加一个
    模块就要改两处（work 曾因此漏挂 vane，风向标行在交互模式缺席）。逐个隔离：
    任一模块失败只打一行警告，不阻断主流程；「历史日期 / 盘中跳过」的守卫在
    各模块内部。入口层已拉过交易日历时转给接收方，免重复下载。
    """
    for tag, mod_name, wants_cal in _BYPASS_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            kw = {"calendar": calendar} if (wants_cal and calendar) else {}
            print(mod.daily_sync(trade_date, **kw))
        except Exception as e:                                 # noqa: BLE001
            print(f"[{tag}] 同步失败（{type(e).__name__}: {e}），主流程不受影响")


def seed_bars_from_prev(out_dir: str, trade_date: dt.date, calendar,
                        bars_cache: dict) -> dict:
    """阶段B：从上一交易日缓存播种日K（当日增量拼接零网络；只播末根连续的）。"""
    if not calendar:
        return bars_cache
    past = [d for d in calendar if d < trade_date]
    if not past:
        return bars_cache
    prev = past[-1]
    ycache = cache_store.load(out_dir, prev)
    if cache_store.is_intraday_cache(ycache, prev):
        # 上一交易日的缓存若是盘中写的，其末根是盘中拼的：收盘价未变的封板股
        # 能通过昨收校验被继续沿用，前几日量能就带着一根不完整的K线
        print(f"  [提示] {prev} 的缓存写于盘中，不播种其日K（今日按需全量拉取）")
        return bars_cache
    ybars = ycache.get("bars")
    if not isinstance(ybars, dict) or not ybars:
        return bars_cache
    seeded = 0
    for c, df in ybars.items():
        if c in bars_cache or df is None or len(df) < 2:
            continue
        if df["date"].iloc[-1] == prev:
            bars_cache[c] = df
            seeded += 1
    if seeded:
        print(f"  [增量] 从 {prev} 缓存播种 {seeded} 只日K"
              f"（今日快照增量拼接，除权/缺口自动全量重拉）")
    return bars_cache


def guard_intraday(trade_date: dt.date, calendar) -> None:
    """阶段C保护：交易日 09:15~15:05 实时快照已是今日盘中数据，拒绝新拉（防 T-1 污染）。"""
    from config import AFTER_CLOSE_BUFFER_MINUTES, INTRADAY_GUARD_START
    today = dt.date.today()
    if trade_date >= today:
        return  # 当日盘中预览允许（cache_is_valid 会在收盘后自动作废重拉）
    trading_today = (today in calendar) if calendar else (today.weekday() < 5)
    now = dt.datetime.now().time()
    if (trading_today
            and dt.time(*INTRADAY_GUARD_START) <= now
            < dt.time(15, AFTER_CLOSE_BUFFER_MINUTES)):
        raise RuntimeError(
            f"盘中保护：当前处于交易时段（{INTRADAY_GUARD_START[0]:02d}:"
            f"{INTRADAY_GUARD_START[1]:02d}~15:05），实时快照已是今日盘中数据，"
            f"不能作为 {trade_date} 的收盘数据。\n"
            f"  请 15:05 后重新运行；如已有 {trade_date} 的缓存，直接查看即可。")


def acquire_spot(trade_date: dt.date, latest: dt.date, out_dir: str,
                 refresh: bool, calendar):
    """阶段C：快照获取。历史日期只读本地缓存（live=False），最新日走实时+盘中保护。"""
    historical = trade_date < latest
    if refresh and historical:
        raise RuntimeError(f"{trade_date} 是历史日期，历史快照只能当天运行时保存，无法强制刷新。")
    if historical:
        return fetch_spot(trade_date, cache_dir=out_dir, live=False)
    guard_intraday(trade_date, calendar)
    print("拉取全市场快照（东财直连 → 新浪，当天缓存直接复用）...")
    return fetch_spot(trade_date, cache_dir=out_dir, force=refresh)


def candidates_from_spot(spot) -> list[dict]:
    """阶段D：快照 → 主板候选池（空候选视为异常日）。"""
    print(f"快照共 {len(spot)} 条，开始主板筛选（60xxxx / 00xxxx）...")
    cands = build_candidates(spot)
    print(f"主板候选: {len(cands)} 只（曾涨停 / 曾跌停 / 振幅≥7%，可重叠）")
    if not cands:
        raise RuntimeError("无候选股票。可能原因：非交易日、盘中快照为空、接口返回异常。")
    return cands


def archive_cycle_series(out_dir: str, series: list, stage: str) -> None:
    """情绪周期逐日指标长期归档（out/archive/cycle_series.csv，追加式幂等）。

    series 项含 date/height/zt_count/promo/break_rate（末项含 money）；
    stage 只属于本次复盘日。同日重跑覆盖该日行——支撑周期门控回测与
    STAGE_POSITION 仓位表重校（研究日志五H记录的张力等的就是这份数据）。
    """
    if not series:
        return
    cols = ["date", "height", "zt_count", "promo", "break_rate", "money", "stage"]
    path = os.path.join(out_dir, "archive", "cycle_series.csv")
    by_date = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                by_date = {r.get("date", ""): {c: r.get(c, "") for c in cols}
                           for r in csv.DictReader(f)}
        except Exception:
            by_date = {}
    for m in series:
        d = str(m.get("date", ""))
        if not d:
            continue
        row = by_date.get(d, {"date": d})
        for k in ("height", "zt_count", "promo", "break_rate", "money"):
            if m.get(k) is not None:
                row[k] = m[k]
        by_date[d] = row
    last_d = str(series[-1].get("date", ""))
    if last_d in by_date:
        by_date[last_d]["stage"] = stage   # 阶段只属于复盘日
    # 原子写：本档案逐日累积、窗口外池数据过东财 3 周保留期后不可再生，
    # 直接覆写被截断=永久丢历史（且写入发生在 analyze_all 前，后续崩溃殃及）
    fstore.write_csv(path, cols, [by_date[k] for k in sorted(by_date)])


def assemble_market_context(trade_date: dt.date, calendar, spot, cands,
                            out_dir: str = "out"):
    """阶段E：市场环境装配（池附加信息 → 行业兜底 → 概念快照 → 赚钱效应 → 情绪周期）。

    返回 (extras, ctx)；ctx 各键的生产者契约见 docs/architecture.md。
    周期逐日指标顺带长期归档（archive/cycle_series.csv，失败不中断）。
    """
    print("拉取涨停/炸板/跌停池补充连板信息（失败自动忽略）...")
    extras, ctx = fetch_pool_extras(trade_date, calendar=calendar)
    fill_industry_local(cands, extras, ctx)   # 池缺失时行业本地补全
    print("拉取概念板块行情（fs=m:90 t:3，失败回退行业口径）...")
    boards = fetch_concept_boards()
    if boards:
        ctx["concept_boards"] = boards
        print(f"  [数据源] 概念板块快照: {len(boards)} 个")
    else:
        print("  [提示] 概念板块接口不可用，A维按行业口径评分")
    # 昨日涨停股今日表现（赚钱效应）：昨日池代码 × 今日快照涨跌幅
    if ctx.get("_yzt_mb_codes") and spot is not None:
        pct_of = {str(r.get("code", "")).zfill(6): r.get("pct")
                  for r in spot.to_dict("records")}     # 5000 行：不用 iterrows
        rets = []
        for ycode in ctx["_yzt_mb_codes"]:
            v = pct_of.get(ycode)
            if v is not None and not pd.isna(v):
                rets.append(float(v))
        if rets:
            ctx["yzt_today_pct"] = rets
    ctx.pop("_yzt_mb_codes", None)
    stage, why, series = mood.stage_for_today(trade_date, calendar, ctx)
    ctx["cycle_stage"] = stage
    ctx["cycle_note"] = why
    try:
        archive_cycle_series(out_dir, series, stage)
    except Exception as e:
        print(f"  [提示] 周期序列归档失败（不影响本次复盘）: {e}")
    print(f"  [情绪周期] {stage}期（{why}）→ 仓位建议: {mood.STAGE_ADVICE.get(stage, '—')}")
    return extras, ctx


def doctor_check(spot, results, bars_cache, trade_date) -> list[str]:
    """数据质量自检（纯函数，返回告警列表，空=健康；2026-09-01 审计后常驻）。

    覆盖审计发现的隐患类型：字段量纲回归、跨源数量级漂移、K线与快照口径不一致。
    注意：bars_cache 含从昨日缓存播种、未被分析（非封板股）的旧 bars——
    末根早于复盘日的不参与对比。
    """
    warns = []
    # 1) 快照换手率量纲：应≈0.01~0.1 小数（东财百分数回归会到 1~10）
    if spot is not None and "turnover" in getattr(spot, "columns", []):
        t = spot["turnover"].dropna()
        t = t[t > 0]
        if len(t) > 100:
            med = float(t.median())
            if not 0.001 < med < 0.5:
                warns.append(f"快照换手率中位数 {med:.4f} 异常（应≈0.01~0.1 小数口径）"
                             f"——疑似单位回归百分数，C维换手分会全错")
    # 2) 主板封板数 vs 池连板股数量级（防池格式漂移/快照半截）
    if spot is not None and results:
        codes = spot["code"].astype(str)
        mb = codes.str.startswith(MAIN_BOARD_PREFIXES)
        sealed = int(((spot["pct"] >= 9.9) & mb).sum())
        pool_n = sum(1 for r in results if int(r.get("lianban") or 0) > 0)
        if sealed or pool_n:
            if not (pool_n and 0.3 < sealed / pool_n < 3.3):
                warns.append(f"主板快照封板 {sealed} 只 vs 池连板股 {pool_n} 只"
                             f"——疑似池接口格式漂移或快照不完整")
    # 3) K线末根收盘 vs 快照收盘抽查（只看末根=复盘日的；防复权口径/除权未重拉）
    if spot is not None and isinstance(bars_cache, dict) and bars_cache:
        close_of = {str(r["code"]).zfill(6): float(r["close"])
                    for r in spot.to_dict("records")
                    if r.get("close") == r.get("close")}
        bad = []
        for code, bars in bars_cache.items():
            if len(bad) >= 3:
                break
            if bars is None or len(bars) < 2:
                continue
            if bars["date"].iloc[-1] != trade_date:
                continue          # 昨日播种的旧 bars（非封板股未分析），不比
            px = close_of.get(str(code).zfill(6))
            if not px:
                continue
            kx = float(bars["close"].iloc[-1])
            if abs(kx - px) > max(0.011, 0.002 * px):
                bad.append(f"{code}({kx:.2f}≠{px:.2f})")
        if bad:
            warns.append("K线末根收盘与快照不一致: " + "、".join(bad)
                         + " —— 疑似复权口径/除权未重拉")
    # 4) 量比分布 sanity（防量能字段损坏导致明日清单失真）
    vrs = sorted(r["vol_ratio"] for r in (results or [])
                 if isinstance(r.get("vol_ratio"), (int, float))
                 and r["vol_ratio"] == r["vol_ratio"])
    if len(vrs) >= 10:
        med = vrs[len(vrs) // 2]
        if not 0.2 < med < 5:
            warns.append(f"量比中位数 {med} 异常（应≈0.5~2）——量能字段可能损坏")
    return warns


def run_day(trade_date: dt.date, latest: dt.date, out_dir: str, workers: int,
            refresh: bool = False, calendar=None):
    """阶段F：编排 A~E + 分析 + 落盘。失败抛 RuntimeError；返回四元组。

    进程级互斥（fstore.run_lock）：定时任务与手动 work/main 并存时，两进程
    读-改-写同一缓存 pkl 会互相吞掉当日结果——整个拉取+分析+落盘段持锁。
    """
    guard = os.path.join(out_dir, cache_store.CACHE_SUBDIR)
    with fstore.run_lock(guard):
        return _run_day(trade_date, latest, out_dir, workers,
                        refresh=refresh, calendar=calendar)


def _run_day(trade_date: dt.date, latest: dt.date, out_dir: str, workers: int,
             refresh: bool = False, calendar=None):
    from data_source import fetch_history_pool_candidates
    cache = cache_store.load(out_dir, trade_date)
    if not refresh and cache_is_valid(cache, trade_date) and cache.get("results"):
        print(f"  [缓存] 检测到 {trade_date} 的完整结果（{len(cache['results'])} 只），"
              f"跳过网络拉取（--refresh 可强制刷新）")
        # 命中路径不再回写缓存：整文件重写（含 spot 5000 行 + bars）只为刷 ts，
        # 且会把盘中 14:xx 存的缓存 ts 洗成 15:xx「已收盘」——cache_is_valid 的
        # 盘中作废逻辑反而被绕过。结构迁移由 CACHE_VERSION 兜底，无需重写。
        return (cache["results"], cache["skipped"], cache["cands"],
                cache.get("ctx") or {})

    # 日K bars 缓存：算法/版本升级重算时零网络复用（数据本身未变）+ 昨日播种
    bars_cache = dict(cache.get("bars") or {})
    if cache_store.is_intraday_cache(cache, trade_date):
        # 上一轮是盘中运行：末根是盘中拼的，砍掉后按收盘快照重拼
        n_drop = drop_intraday_last(bars_cache, trade_date)
        if n_drop:
            print(f"  [缓存] 上一轮写于盘中：{n_drop} 只日K的盘中末根已丢弃，"
                  f"按收盘快照重新拼接")
    bars_cache = seed_bars_from_prev(out_dir, trade_date, calendar, bars_cache)

    spot = acquire_spot(trade_date, latest, out_dir, refresh, calendar)
    if spot is None and trade_date < latest:
        cands, extras = fetch_history_pool_candidates(trade_date)
        ctx = {}
        if cands is None:
            raise RuntimeError(
                f"{trade_date} 无本地快照且东财历史池不可用，无法补齐。\n"
                f"  提示：历史「振幅≥7%」与「曾跌停翘板」免费接口无法重建；"
                f"建议以后当天收盘后运行本工具。")
    elif spot is None:
        raise RuntimeError("快照获取失败：请检查网络，或稍后重试。")
    else:
        cands = candidates_from_spot(spot)
        extras, ctx = assemble_market_context(trade_date, calendar, spot, cands,
                                              out_dir=out_dir)

    results, skipped = analyze_all(cands, trade_date, extras,
                                   workers=workers, context=ctx,
                                   bars_cache=bars_cache)
    print(f"分析完成: 成功 {len(results)} 只，跳过 {len(skipped)} 只"
          f"（日K失败/上市不足{MIN_BARS}根K线）")
    cache_store.save_result(out_dir, trade_date, results, skipped, cands,
                            bars=bars_cache, ctx=ctx)
    print(f"  [缓存] 完整结果与日K bars 已存入滚动缓存: {trade_date}")
    for w in doctor_check(spot, results, bars_cache, trade_date):
        print(f"  [体检⚠] {w}")
    return results, skipped, cands, ctx


def load_spot(out_dir: str, trade_date: dt.date):
    """读当日快照（报告「昨日信号今日跟踪」需要当日 spot 对照涨跌幅）。"""
    return cache_store.load(out_dir, trade_date).get("spot")


def _run_child_quiet(script: str, args: list[str], tag: str, timeout: int) -> None:
    """静默链跑同目录下的子脚本，尾部打印一行摘要；失败必须如实报出来。

    Windows 编码陷阱（此前让整条校准链静默失效）：子进程默认按控制台代码页 GBK
    写管道，而 subprocess.run(text=True) 按父进程 locale 解码 → reader 线程抛
    UnicodeDecodeError 崩掉 → 没人排空管道 → 子进程写满约 64KB 缓冲后阻塞 →
    撞 timeout 被杀。旧实现只看 stdout 尾行，stdout 为空就走 fallback 打印「完成」，
    于是**失败被报成成功**。实测后果：链跑 calibrate 时只有早期写出的 _forward.csv
    落盘，尾部的 bucket_stats.json / calibration_report.txt 从未生成——
    而报告的同档实测标注正依赖 bucket_stats.json。

    三处修正：子进程强制 UTF-8 输出；父进程 errors="replace" 不再崩；检查退出码。
    """
    import os
    import subprocess
    import sys
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        r = subprocess.run([sys.executable, path, *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, check=False)   # 退出码由下方判读
    except Exception as e:
        print(f"  [{tag}] ✘ 跳过（{e}）")
        return
    if r.returncode != 0:
        tail = [ln for ln in (r.stderr or r.stdout or "").splitlines() if ln.strip()]
        print(f"  [{tag}] ✘ 退出码 {r.returncode}"
              + (f"：{tail[-1].strip()[:120]}" if tail else "（无输出）"))
        return
    tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    print(f"  [{tag}] " + (tail[-1].strip() if tail else "完成（子进程无输出）"))


def run_research_update_quiet() -> None:
    """静默滚动研究缓存（research_scan --update，约1-3分钟）。

    可选步骤：main --with-research 或环境变量 WORK_WITH_RESEARCH=1 时触发；
    run_daily.bat 已含独立步骤，交互模式默认跳过（避免等待）。
    窗口锚点自 2026-09-15 起冻结（research_scan.window_start）：--update 只向后
    延伸，不再把研究窗口重裁到运行当日——此前那句「跑之前先留当天快照」的
    警告已过期（旧实现会滑动窗口、让历史事件蒸发）。
    """
    _run_child_quiet("research_scan.py", ["--update"], "研究缓存", 1800)


def run_calibrate_quiet() -> None:
    """静默链跑 calibrate.py（前向数据/分数桶滚动），尾部打印一行摘要。

    入口层（main/work）在报告生成后调用——用户每天只需一个动作，研究侧数据自动滚动。
    """
    _run_child_quiet("calibrate.py", [], "校准", 600)
