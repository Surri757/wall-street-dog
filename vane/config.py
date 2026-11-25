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

# ── 换手门槛（2026-11-25 用户定） ─────────────────────────────────────
# 本轮（首板到今日：涨停日 + 断板日）里换手率 ≥ TURN_GATE_PCT% 的交易日至少
# TURN_GATE_DAYS 天，才占名额；不设上限（「有几天换手吓人也不错」）。
# 这是用户的交易逻辑，不是数据拟合出来的：research_log 五AB 的检验显示该规则
# 整体不改变晋级/开→开结果（4 板略差、5 板明显更好）；用户知情后决定采用。
TURN_GATE_DAYS = 3
TURN_GATE_PCT = 10.0   # 百分数口径（10.0 = 换手率 10%）

# ── 路径 ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("out", "research", "vane")
LEDGER_CSV = os.path.join(DATA_DIR, "ledger.csv")
REPORT_TXT = os.path.join(DATA_DIR, "report_latest.txt")
LAST_SCAN_JSON = os.path.join(DATA_DIR, "last_scan.json")   # 当日是否跑过+票数
TURNOVER_DIR = os.path.join(DATA_DIR, "turnover")  # 本地换手库：<日期>.csv（code, 换手率%）

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
XGB_GATE_N = 120            # auto 模式下样本 ≥N 才走 xgb（否则 logreg）
# Round6（2026-11-24）滚动 origin：logreg C0.5 + Platt 的 top3 42.86% vs
# 现行 xgb+iso 28.57%，logloss 0.7587 vs 4.6314；小样本强制线性主模型。
MODEL_KIND = "logreg"          # auto / xgb / logreg；小样本不再被树模型过拟合
CALIBRATION = "platt"         # platt / iso；保留 iso.json 文件名兼容旧工件
LOGREG_GRID = (0.5,)           # 滚动验证冻结；避免全量重训时又按训练内 CV 改参
MODEL_VERSION = "vane-3d-2-lr05-platt"

# 特征集（全 T 收盘可知、纯池口径；变更即触发重训——对齐 ziyang 口径门）
FEATURES = ("B", "span", "breaks", "seal_ratio", "fbt_min", "log_close",
            "log_ltsz", "mkt_height", "mkt_two_plus", "ind_zt_count",
            "ind_max_b", "height_trend")

# ── premium11 溢价影子模型（2026-11-25 合同冻结；仅影子，不切生产） ──
# 合同：out/research/vane/premium11/contract.json。新模型/样本/对照全部落本目录，
# 旧晋级工件（MODEL_DIR）保持不动；生产名额仍由旧晋级模型/启发式决定。
PREMIUM_DIR = os.path.join(DATA_DIR, "premium11")
PREMIUM_MODEL_JSON = os.path.join(PREMIUM_DIR, "premium_model.json")
PREMIUM_SHADOW_JSON = os.path.join(PREMIUM_DIR, "shadow_daily.json")
PREMIUM_SHADOW_CSV = os.path.join(PREMIUM_DIR, "shadow_daily.csv")
PREMIUM_VERSION = "premium-11-ridge10"
PREMIUM_ALPHA = 10.0          # 合同冻结 Ridge alpha；本轮不开网格
PREMIUM_MIN_TRAIN_DAYS = 10   # 样本门：≥10 个训练信号日
PREMIUM_MIN_SETTLED = 30      # 样本门：≥30 条已结算可成交记录
PREMIUM_FEATURES = list(FEATURES) + ["old_promo_p"]
PREMIUM_LAYERS = ("minute20", "approx_open")


def _ensure_in_data_dir(*paths: str) -> None:
    """路径守卫：落盘路径常量必须在 DATA_DIR 内（模块加载时一次性断言）。

    模块落盘路径全部是上方常量（非用户输入），本校验是纵深防御——
    防止常量被改动后路径静默漂移到数据目录之外。业务文件直接使用常量。
    """
    root = os.path.abspath(DATA_DIR)
    for p in paths:
        if not os.path.abspath(p).startswith(root + os.sep):
            raise ValueError(f"落盘路径越界（须在 {root} 内）: {p}")


_ensure_in_data_dir(LEDGER_CSV, REPORT_TXT, LAST_SCAN_JSON, TURNOVER_DIR,
                    PREMIUM_MODEL_JSON, PREMIUM_SHADOW_JSON, PREMIUM_SHADOW_CSV)

# ── 复用主工程口径（不复制数值，防分叉） ─────────────────────────────
__all__ = ["WINDOW", "MAX_GAP", "TOP_N", "BOARD_RANGE",
           "TURN_GATE_DAYS", "TURN_GATE_PCT", "TURNOVER_DIR",
           "DATA_DIR", "LEDGER_CSV", "REPORT_TXT", "LAST_SCAN_JSON",
           "CADENCE_DAYS", "MODEL_DIR", "MODEL_JSON", "MODEL_LGR_JSON",
           "MODEL_XGB_JSON", "MODEL_PREV_JSON", "ISO_JSON",
           "RETRAIN_DIFF_JSON", "SAMPLES_MIN_POS", "SAMPLES_MIN_ROWS",
           "XGB_GATE_N", "MODEL_KIND", "CALIBRATION",
           "LOGREG_GRID", "MODEL_VERSION", "FEATURES",
           "PREMIUM_DIR", "PREMIUM_MODEL_JSON", "PREMIUM_SHADOW_JSON",
           "PREMIUM_SHADOW_CSV", "PREMIUM_VERSION", "PREMIUM_ALPHA",
           "PREMIUM_MIN_TRAIN_DAYS", "PREMIUM_MIN_SETTLED",
           "PREMIUM_FEATURES", "PREMIUM_LAYERS",
           "MAIN_BOARD_PREFIXES", "EXCLUDE_ST", "PROMOTE_APPROX"]
