# -*- coding: utf-8 -*-
"""交互式复盘入口：终端进入项目文件夹后输入 work 启动。

加载并检查滚动窗口数据状态，弹出选项：
    A. 2026-08-18（周二，已更新）
    B. 2026-08-19（周三，待更新）
    C. 2026-08-20（周四，待更新）
    0. 退出
选择「待更新」日期会立即拉数据更新，更新后再次进入菜单即显示「已更新」。
窗口内完全无数据时（如周一），直接询问「是否更新今日数据？ A.是 / B.否」。
"""
from __future__ import annotations

import datetime as dt
import os

import cache_store
from config import DEFAULT_WORKERS, ROLLING_DAYS
from data_source import cache_is_valid, resolve_trade_date, trade_calendar
from pipeline import run_day as collect_data  # 入口层不互调：直接用流水线
from pipeline import sync_bypass_modules
from report import write_report
from termstyle import weekday_cn

OUT_DIR = "out"


def _show_report(trade_date: dt.date, latest: dt.date, results, skipped, cands,
                 ctx: dict | None = None, calendar: list | None = None) -> None:
    # 旁路子包同步（紫阳甄选 / 情绪风向标 / 竞价采集）：报告渲染前统一跑。挂点单一
    # 实现在 pipeline.sync_bypass_modules——此前 work 与 main 各手抄一份清单，
    # work 曾因此漏挂 vane，风向标行在交互模式缺席
    if trade_date == latest:
        sync_bypass_modules(trade_date, calendar=calendar)
    from pipeline import load_spot
    _txt, csv_path, colored = write_report(
        results, skipped, cands, trade_date,
        out_dir=OUT_DIR, signal_only=True, ctx=ctx,
        spot=load_spot(OUT_DIR, trade_date))   # 当日快照 → 昨日信号跟踪
    print("")
    print("=" * 78)
    # 打印内存里的彩色版；落盘文件是纯文本（供编辑器/grep/归档），不再读回
    print(colored)
    print(f"CSV 全量明细: {csv_path}")
    from report import calibrate_hint
    calibrate_hint(OUT_DIR, trade_date)


def _ask_yes_no(trade_date: dt.date, latest: dt.date, workers: int, cal) -> None:
    print(f"当前没有任何复盘数据。是否更新今日数据（{trade_date}）？")
    print("  A. 是")
    print("  B. 否")
    try:
        ch = input("请选择 [A/B]: ").strip().upper()
    except EOFError:
        print("（非交互终端，已退出）")
        return
    if ch != "A":
        print("未更新，退出。")
        return
    try:
        results, skipped, cands, ctx = collect_data(trade_date, latest, OUT_DIR,
                                                    workers, calendar=cal)
    except RuntimeError as e:
        print(f"更新失败: {e}")
        return
    _show_report(trade_date, latest, results, skipped, cands, ctx, calendar=cal)
    from pipeline import run_calibrate_quiet
    run_calibrate_quiet()
    if os.environ.get("WORK_WITH_RESEARCH") == "1":
        from pipeline import run_research_update_quiet
        run_research_update_quiet()


def _menu(status, latest: dt.date, workers: int, cal) -> None:
    import termstyle as T
    while True:
        print("请选择要查看/更新的日期:")
        for i, (d, updated) in enumerate(status):
            tag = "已更新" if updated else "待更新"
            c = cache_store.load(OUT_DIR, d)
            stage = (c.get("ctx") or {}).get("cycle_stage")
            extra = T.stage(f"，{stage}期") if stage else ""
            print(f"  {chr(65 + i)}. {d}（{weekday_cn(d)}，{tag}{extra}）")
        print("  0. 退出")
        try:
            ch = input("请选择: ").strip().upper()
        except EOFError:
            print("（非交互终端，已退出）")
            return
        if ch == "0":
            print("退出。")
            return
        if len(ch) != 1 or not ("A" <= ch <= "Z"):
            print("无效选择（回车=重新显示菜单，输 0 退出）。")
            continue
        idx = ord(ch) - ord("A")
        if not (0 <= idx < len(status)):
            print("无效选择。")
            continue
        trade_date, updated = status[idx]
        if not updated:
            print(f"开始更新 {trade_date}（{weekday_cn(trade_date)}）...")
        try:
            results, skipped, cands, ctx = collect_data(trade_date, latest, OUT_DIR,
                                                        workers, calendar=cal)
        except RuntimeError as e:
            print(f"更新失败: {e}")
            continue
        status[idx] = (trade_date, True)
        _show_report(trade_date, latest, results, skipped, cands, ctx, calendar=cal)
        from pipeline import run_calibrate_quiet
        run_calibrate_quiet()
        if os.environ.get("WORK_WITH_RESEARCH") == "1":
            from pipeline import run_research_update_quiet
            run_research_update_quiet()


def main():
    cal = trade_calendar()
    latest = resolve_trade_date(None, calendar=cal)
    window = cache_store.window_dates(cal, latest, keep=ROLLING_DAYS)

    status = []
    for d in window:
        c = cache_store.load(OUT_DIR, d)
        updated = bool(cache_is_valid(c, d) and c.get("results"))
        status.append((d, updated))

    cache_store.prune(OUT_DIR, cal, latest, keep=ROLLING_DAYS)

    print("=" * 60)
    print(f"  A股复盘收割器 · work 菜单   最新可复盘日: {latest}")
    print(f"  滚动窗口: {'，'.join(str(d) for d in window) or '空'}")
    print("=" * 60)

    # 周一全新（窗口只有最新一天且无数据）→ 直接询问是否更新
    if len(window) == 1 and not any(up for _, up in status):
        _ask_yes_no(latest, latest, DEFAULT_WORKERS, cal)
        return
    _menu(status, latest, DEFAULT_WORKERS, cal)


if __name__ == "__main__":
    main()
