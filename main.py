# -*- coding: utf-8 -*-
"""复盘收割器入口（非交互批量模式）—— 专注二板定龙头（2连板晋级）。

交互入口见 work.py（终端输入 work 启动菜单）。

滚动缓存机制（周内滚动，最多 3 个交易日，周末悬停，周一清空）：
- 运行前先查该日期缓存：命中直接复用，不碰网络；
- 未命中才拉数据，拉完自动存档；
- 每次运行按窗口规则清理窗口外缓存；
- 历史日期复盘优先用本地快照缓存；无快照则尝试东财历史池补录；
- --refresh 强制重拉（仅最新交易日）。

用法:
    python main.py                  # 自动复盘最近一个已收盘交易日
    python main.py --date 20250610  # 复盘窗口内的某一天
    python main.py --all            # 报告也输出未入选的封板股（默认只列评分≥60）
    python main.py --workers 4      # 调低并发（接口限流时）
    python main.py --refresh        # 强制重新拉取（仅对最新交易日有效）
    python main.py --fill           # 自动补齐滚动窗口内缺失的交易日
"""
from __future__ import annotations

import argparse
import datetime as dt

import cache_store
from config import DEFAULT_WORKERS, ROLLING_DAYS
from data_source import cache_is_valid, resolve_trade_date, trade_calendar
from report import write_report


def collect_data(trade_date: dt.date, latest: dt.date, out_dir: str,
                 workers: int, refresh: bool = False,
                 calendar: list | None = None):
    """确保某交易日有完整分析结果（流水线实现见 pipeline.py）。

    命中缓存直接返回；未命中则拉数据/补录并保存滚动缓存。
    返回 (results, skipped, cands, ctx)。失败抛 RuntimeError（带原因）。
    """
    from pipeline import run_day
    return run_day(trade_date, latest, out_dir, workers,
                   refresh=refresh, calendar=calendar)


def fill_window(latest: dt.date, cal, out_dir: str, workers: int) -> None:
    """补齐滚动窗口内 < latest 的缺失交易日。"""
    window = cache_store.window_dates(cal, latest, keep=ROLLING_DAYS)
    missing = [d for d in window if d < latest]
    for d in missing:
        c = cache_store.load(out_dir, d)
        if cache_is_valid(c, d) and c.get("results"):
            print(f"  [补录] {d} 已有完整结果，跳过")
            continue
        print(f"  [补录] 发现空缺: {d}，开始补齐...")
        try:
            collect_data(d, latest, out_dir, workers, calendar=cal)
        except RuntimeError as e:
            print(f"  [补录] {d} 补齐失败: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="A股复盘收割器（主板·二板定龙头）")
    parser.add_argument("--date", help="复盘日期 YYYYMMDD；缺省=最近一个已收盘交易日")
    parser.add_argument("--all", action="store_true",
                        help="报告也输出未入选的封板股（默认只列评分≥60 的二板候选）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"拉日K并发线程数（默认 {DEFAULT_WORKERS}）")
    parser.add_argument("--out", default="out", help="输出目录（默认 out）")
    parser.add_argument("--refresh", action="store_true",
                        help="强制忽略缓存，重新拉取数据（仅对最新交易日有效）")
    parser.add_argument("--fill", action="store_true",
                        help="自动补齐滚动窗口内缺失的交易日（历史池可补涨停/炸板/跌停）")
    parser.add_argument("--doctor", action="store_true",
                        help="只对当日缓存做数据质量自检（不拉数据不重算）")
    parser.add_argument("--with-research", action="store_true",
                        help="报告后滚动12个月研究缓存（research_scan --update，约1-3分钟）")
    args = parser.parse_args()

    cal = trade_calendar()
    latest = resolve_trade_date(None, calendar=cal)          # 最新可复盘交易日
    trade_date = resolve_trade_date(args.date, calendar=cal)  # 本次复盘日期
    print(f"复盘日期: {trade_date}")

    if args.doctor:
        from pipeline import doctor_check
        cache = cache_store.load(args.out, trade_date)
        warns = doctor_check(cache.get("spot"), cache.get("results") or [],
                             cache.get("bars") or {}, trade_date)
        if warns:
            for w in warns:
                print(f"[体检⚠] {w}")
            raise SystemExit(1)
        print("数据体检: 全部通过（换手率量纲/封板数量级/K线口径/量比分布）")
        return

    if args.fill:
        fill_window(latest, cal, args.out, args.workers)

    try:
        results, skipped, cands, ctx = collect_data(
            trade_date, latest, args.out, args.workers,
            refresh=args.refresh, calendar=cal)
    except RuntimeError as e:
        raise SystemExit(f"[失败] {e}")

    # 旁路子包同步（紫阳甄选 / 情绪风向标 / 竞价采集）：报告渲染前统一跑，
    # 挂点单一实现在 pipeline.sync_bypass_modules——历史日/盘中各自跳过，
    # 任一模块失败只打一行警告、不阻断主流程
    from pipeline import sync_bypass_modules
    sync_bypass_modules(trade_date, calendar=cal)

    # 当日快照供「昨日信号今日跟踪」对照（历史池补录日无快照则跳过该节）
    from pipeline import load_spot
    txt_path, csv_path, _colored = write_report(
        results, skipped, cands, trade_date,
        out_dir=args.out, signal_only=not args.all, ctx=ctx,
        spot=load_spot(args.out, trade_date),
    )

    # prune 以 latest（窗口最新日）为锚：--date 重放历史日时不得清掉比它新的
    # 缓存——历史日快照只能当天实跑保存，删掉即永久丢失（work.py 同款口径）
    cache_store.prune(args.out, cal, latest, keep=ROLLING_DAYS)

    print("")
    print("报告已生成（每次覆盖为最新一份）:")
    print(f"  TXT 信号清单: {txt_path}")
    print(f"  CSV 全量明细: {csv_path}")
    print(f"  信号归档: {args.out}/archive/report_{trade_date:%Y%m%d}.txt "
          f"+ signals_{trade_date:%Y%m%d}.csv（长期保留，供事后胜率校准）")
    print(f"滚动窗口: 周内滚动，最多保留最近 {ROLLING_DAYS} 个交易日，窗口外自动清理。")
    print("口径: 仅主板收盘封板股参与二板判定；2连板+五维评分≥70入池；连板数优先涨停池。")
    print("数据仅供复盘研究，不构成投资建议。")
    from pipeline import run_calibrate_quiet
    run_calibrate_quiet()
    if args.with_research:
        from pipeline import run_research_update_quiet
        run_research_update_quiet()


if __name__ == "__main__":
    main()
