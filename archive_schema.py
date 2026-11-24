# -*- coding: utf-8 -*-
"""归档文件的模式单一源（文件仓的「表结构」；零依赖叶子模块）。

项目无数据库，数据仓 = out/ 下的 CSV / pkl / json。同一类文件的列清单必须只有
一份：此前 report.py 与 backfill.py 各手抄一份 signals_*.csv 的列定义，回灌
那份还是 23 列子集——同名归档因此有三种形态（实跑 32 列 / 回灌 23 列 /
历史池补录 32 列但大片空白），读方只能各自防御。写方一律从这里取列。
"""
from __future__ import annotations

# 日度信号归档 out/archive/signals_YYYYMMDD.csv（报告 CSV 同此列序）。
# 新列只追加在末尾、不改既有列名——历史归档是不可再生的实证资产。
SIGNALS_CSV_COLUMNS = [
    "trade_date", "weekday", "code", "name", "pools", "zt_status",
    "close", "open", "prev_close", "pct", "amp", "turnover", "amount",
    "industry", "concepts", "ltsz",
    "lianban", "zhaban_times", "fbt", "fbt_y", "fund", "seal_ratio",
    "seal_ratio_y", "vol_ratio", "board_pass", "board_score", "board_note",
    "skip_reason", "market_max_lianban", "cycle_stage", "zt_money_effect",
    "source",
]
