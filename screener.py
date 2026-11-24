# -*- coding: utf-8 -*-
"""筛选层门面（facade）：实现拆分在 candidates/bars/analyzer 三个单职责模块。

对外接口保持不变（tests 等既有调用方零改动）；新代码建议直接引用具体模块。
仅导出实际被使用的名字（2026-09-01 审计精简：删除无人引用的私有名转出）。

  candidates.py  快照 → 主板候选池（向量化，纯数据变换）
  bars.py        日K三级获取 + 增量拼接 + 连板近似（公开 API：
                 acquire_bars / incremental_bars / approx_lianban）
  analyzer.py    单股四步分析 + 并发编排 + 情绪环境注入
"""
from __future__ import annotations

from analyzer import analyze_all  # noqa: F401
from bars import incremental_bars  # noqa: F401
from candidates import build_candidates, is_main_board, limit_prices  # noqa: F401
