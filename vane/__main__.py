# -*- coding: utf-8 -*-
"""情绪风向标独立 CLI。

用法:
    python -m vane scan [--date YYYYMMDD]   # 跑筛选+登记台账+渲染报告
                                           # （--date 指定历史日=只回看不登记）
    python -m vane verify                   # 核验到期 pending 行
    python -m vane report                   # 打印最近一份报告（零网络）
    python -m vane samples [--date]         # 三日模型样本经济探测
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="情绪风向标：4板晋5/5板晋6 三名额筛选")
    parser.add_argument("cmd", nargs="?", default="scan",
                        choices=["scan", "verify", "report", "samples"],
                        help="子命令")
    parser.add_argument("--date", help="复盘日期 YYYYMMDD（scan 可选，默认最新已收盘日）")
    args = parser.parse_args()

    if args.cmd == "scan":
        _scan(args.date)
    elif args.cmd == "samples":
        from data_source import resolve_trade_date, trade_calendar
        from vane.model import samples_report
        trade_date = resolve_trade_date(args.date,
                                        calendar=trade_calendar())
        print(samples_report(trade_date))
    elif args.cmd == "verify":
        from vane import verify_pending
        print(verify_pending())
    elif args.cmd == "report":
        import os
        from vane import config as C
        if not os.path.exists(C.REPORT_TXT):
            print("尚无报告：先跑 python -m vane scan")
            return
        with open(C.REPORT_TXT, encoding="utf-8") as f:
            print(f.read(), end="")


def _scan(date_arg: str | None) -> None:
    """scan 子命令：最新日=完整同步（守卫+登记）；--date 历史日=只回看不登记。"""
    from data_source import resolve_trade_date, trade_calendar

    from vane import config as C, daily_sync
    cal = trade_calendar()
    trade_date = resolve_trade_date(date_arg, calendar=cal)
    if date_arg and trade_date != resolve_trade_date(None, calendar=cal):
        # 历史日回看：绕过 daily_sync 守卫直接筛选，只渲染不重登台账
        from vane import ledger, report
        from vane.screen import scan
        result = scan(trade_date)
        if result.get("error"):
            print(f"[风向标] {result['error']}")
            return
        print(report.render(result))
        n = len(ledger.rows_for(trade_date))
        print(f"[风向标] 历史回看 {trade_date}（台账登记 {n} 条保持不变）")
        return
    print(daily_sync(trade_date))
    print(f"报告: {C.REPORT_TXT}")


if __name__ == "__main__":
    main()
