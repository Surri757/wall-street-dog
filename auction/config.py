# -*- coding: utf-8 -*-
"""竞价采集模块私有配置。

只放刻度参数与落盘路径；假设与判定线的**冻结版**在
out/research/prereg/2026-11-23-五Z/PREREG.md 第七节（采集开始前登记），
这里的数值必须与之一致——改这里等于改预登记，须先改登记并注明日期。
"""
from __future__ import annotations

import os

from config import CALIB_MIN_INFER_DAYS

# ── 预登记假设阈值（PREREG 七） ───────────────────────────────────────
AUCTION_VOL_RATIO = 0.03         # A1：竞价成交量 ≥ 信号日总量 3%（首板高开/追板/竞价三合一）
GOLDEN_BAND = (0.0382, 0.0618)   # A2：竞价价/信号日收盘 − 1 ∈ 3.82%~6.18%（集合竞价摸奖）
RZQ_BAND = (-0.02, 0.09)         # A3：炸板组 开盘/信号日涨停价 − 1 ∈ −2%~+9%（弱转强；
                                 #     原帖写作比值 0.98~1.09，台账统一存"比值−1"的缺口）

# ── 判定线（PREREG 七，与主检验同口径；另加"组内扣成本为正"） ──────────
MIN_N = 30                       # 每组最少样本
MIN_DAYS = CALIB_MIN_INFER_DAYS  # 按日配对最少天数
Q_MAX = 0.10                     # BH-FDR 过线线（检验族 m=3）
MIN_DELTA = 0.010                # 最小经济幅度 1.0pp
MIN_MEAN = 0.003                 # 过线还须组内 ret_oc 均值 > +0.3%（往返成本约 0.15%）

# ── 路径 ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("out", "research", "auction")
LEDGER_DIR = os.path.join(DATA_DIR, "ledger")       # 台账按执行日分片：<执行日>.csv
MINUTE_DIR = os.path.join(DATA_DIR, "minute")       # 执行日分钟路径原料（体积大，gitignore）
ARCHIVE_DIR = os.path.join("out", "archive")        # 信号日封板/炸板名单来源

# 台账列（列序即 CSV 列序）
COLS = ("exec_date", "signal_date", "code", "name", "group", "lianban",
        "close_t", "prev_close", "limit_t", "vol_ratio_t", "board_score_t",
        "open_exec", "gap", "gap_vs_limit", "auction_vol", "auction_amount",
        "vol_t", "amount_t", "auction_vol_ratio",
        "high_exec", "close_exec", "pct_close", "touch_limit", "first_limit_time",
        "sealed", "ret_oc", "fetched_at")


def _ensure_in_data_dir(*paths: str) -> None:
    """落盘路径守卫：台账路径常量必须落在 DATA_DIR 内（vane/config 同款）。"""
    root = os.path.abspath(DATA_DIR)
    for p in paths:
        if not os.path.abspath(p).startswith(root + os.sep):
            raise ValueError(f"落盘路径越界（须在 {root} 内）: {p}")


_ensure_in_data_dir(LEDGER_DIR, MINUTE_DIR)
