# -*- coding: utf-8 -*-
"""竞价采集独立 CLI。

用法:
    python -m auction collect [--date YYYYMMDD]   # 采集（最新日=落盘；--date 历史日只回看，
                                                  # 守卫会跳过，台账不变）
    python -m auction report                       # 预登记分桶读数（零网络）
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="竞价采集：策略池竞价族前向验证")
    parser.add_argument("cmd", nargs="?", default="collect",
                        choices=["collect", "report"], help="子命令")
    parser.add_argument("--date", help="执行日 YYYYMMDD（可选，默认最新已收盘日）")
    args = parser.parse_args()

    if args.cmd == "report":
        from auction import report
        print(report())
        return
    from data_source import resolve_trade_date, trade_calendar

    cal = trade_calendar()
    trade_date = resolve_trade_date(args.date, calendar=cal)
    from auction import daily_sync
    print(daily_sync(trade_date, calendar=cal))
    print("读数: python -m auction report")


if __name__ == "__main__":
    main()
