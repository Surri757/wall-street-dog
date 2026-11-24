# -*- coding: utf-8 -*-
"""情绪风向标模块私有配置。

刻度参数（WINDOW/MAX_GAP/TOP_N/BOARD_RANGE）只在这里调；主板/ST/晋级口径
直接复用主工程 config 的单一事实源，不另立分叉。
"""
from __future__ import annotations

import os

from config import EXCLUDE_ST, MAIN_BOARD_PREFIXES, PROMOTE_APPROX

# ── 筛选刻度 ──────────────────────────────────────────────────────────
WINDOW = 20          # 回看交易日数（复盘日往前；20 日足够覆盖 5 板+断板间隔）
MAX_GAP = 3          # 同一轮内相邻两板最大间隔（交易日，不含两端板日）；
                     # 超过即久远断档、板数重置——「前面三个板需近期发生」的量化
TOP_N = 3            # 名额：合并 4晋5 / 5晋6 两组排序取前 3（不足即少列）
BOARD_RANGE = (4, 5) # 入围本轮板数：4=明晋5组，5=明晋6组（≥6 的更高标不占名额）

# ── 路径 ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("out", "research", "vane")
LEDGER_CSV = os.path.join(DATA_DIR, "ledger.csv")
REPORT_TXT = os.path.join(DATA_DIR, "report_latest.txt")
LAST_SCAN_JSON = os.path.join(DATA_DIR, "last_scan.json")   # 当日是否跑过+票数

# ── 三日模型（2026-11-22 立项，口径冻结见 vane/RESEARCH.md） ─────────
CADENCE_DAYS = 3            # 重训节奏（交易日）：trained_at 距今 ≥3 日则重训
MODEL_DIR = os.path.join(DATA_DIR, "model")
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")          # 元数据
MODEL_LGR_JSON = os.path.join(MODEL_DIR, "model_lgr.json")  # logreg 系数+校准
MODEL_XGB_JSON = os.path.join(MODEL_DIR, "model_xgb.json")  # xgb booster
MODEL_PREV_JSON = os.path.join(MODEL_DIR, "model_prev.json")
ISO_JSON = os.path.join(MODEL_DIR, "iso.json")
RETRAIN_DIFF_JSON = os.path.join(MODEL_DIR, "retrain_diff.json")
SAMPLES_MIN_POS = 5         # 正例门：训练样本正例 <5 不落盘（ok=False）
SAMPLES_MIN_ROWS = 20       # 行数门：训练样本 <20 行不落盘（ok=False）
XGB_GATE_N = 120            # xgb 启用门：样本 ≥120 才走 xgb（否则 logreg）
LOGREG_GRID = (0.5, 1.0, 2.0)   # L2 强度小网格（leave-one-day-out 自动选参）
MODEL_VERSION = "vane-3d-1"

# 特征集（全 T 收盘可知、纯池口径；变更即触发重训——对齐 ziyang 口径门）
FEATURES = ("B", "span", "breaks", "seal_ratio", "fbt_min", "log_close",
            "log_ltsz", "mkt_height", "mkt_two_plus", "ind_zt_count",
            "ind_max_b", "height_trend")


def _ensure_in_data_dir(*paths: str) -> None:
    """路径守卫：落盘路径常量必须在 DATA_DIR 内（模块加载时一次性断言）。

    模块落盘路径全部是上方常量（非用户输入），本校验是纵深防御——
    防止常量被改动后路径静默漂移到数据目录之外。业务文件直接使用常量。
    """
    root = os.path.abspath(DATA_DIR)
    for p in paths:
        if not os.path.abspath(p).startswith(root + os.sep):
            raise ValueError(f"落盘路径越界（须在 {root} 内）: {p}")


_ensure_in_data_dir(LEDGER_CSV, REPORT_TXT, LAST_SCAN_JSON)

# ── 复用主工程口径（不复制数值，防分叉） ─────────────────────────────
__all__ = ["WINDOW", "MAX_GAP", "TOP_N", "BOARD_RANGE",
           "DATA_DIR", "LEDGER_CSV", "REPORT_TXT", "LAST_SCAN_JSON",
           "CADENCE_DAYS", "MODEL_DIR", "MODEL_JSON", "MODEL_LGR_JSON",
           "MODEL_XGB_JSON", "MODEL_PREV_JSON", "ISO_JSON",
           "RETRAIN_DIFF_JSON", "SAMPLES_MIN_POS", "SAMPLES_MIN_ROWS",
           "XGB_GATE_N",
           "LOGREG_GRID", "MODEL_VERSION", "FEATURES",
           "MAIN_BOARD_PREFIXES", "EXCLUDE_ST", "PROMOTE_APPROX"]
