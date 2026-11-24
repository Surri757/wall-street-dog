# -*- coding: utf-8 -*-
"""评分回测校准：读 out/archive/signals_*.csv，按分数桶统计二板候选的前向表现。

**实现已拆分至 calib/ 包**（2026-09-15 降耦重构，关注点分治）：
  calib/stats 聚类推断核心 · forward 前向数据 · features 维度分析 ·
  buckets 分桶统计 · sections 报告区块 · ledger_health 台账判读 · cli 装配落盘
本文件保留为**兼容门面**：pipeline.run_calibrate_quiet / run_daily.bat 仍以
`python calibrate.py` 子进程方式调用；tests 与 out/research/prereg/* 研究脚本
导入的全部名字（含 _two_sided_p 等事实公共 API）在此重导出，签名与行为不变。

判读纪律：所有均值/差值一律走 calib.stats.cluster_mean / cluster_diff 按交易日
聚类（样本高度按日聚类，朴素 i.i.d. SE 低估约 2.5 倍），日簇不足
config.CALIB_MIN_INFER_DAYS 时输出「无法推断」而不是给一个看似有效的 t。

用法:
    python calibrate.py             # 增量：只补拉缺前向数据的信号
    python calibrate.py --refresh   # 全部重拉前向数据（换口径后必须跑一次）
前向数据缓存在 out/archive/_forward.csv（带防覆写保护），避免重复请求。

注意：信号日之后至少要有 1 个交易日才有前向数据；周六运行周五的信号会提示无数据。
"""
from __future__ import annotations

import data_source  # noqa: F401  # 兼容锚点：tests 补丁打在 calibrate.data_source 上

from calib.buckets import BUCKETS, bucket_segments, bucket_stats
from calib.cli import main
from calib.features import (COLLINEAR_GROUPS, _note_val, analyze_dims,
                            build_features)
from calib.forward import (FWD_KEYS, MAIN_METRIC, MIN_INFER_DAYS,
                           fetch_forward, load_forward_cache, load_signals)
from calib.ledger_health import (LEDGER_EXPECT, LEDGER_WARN_PP, _judge,
                                 _ledger_rows, _row_pair, ledger_section)
from calib.sections import (_contrast_line, _source_of, _table, oos_section,
                            source_section, summarize)
from calib.stats import (bh_fdr, cluster_diff, cluster_mean, fmt_mean, fmt_t,
                         two_sided_p, _two_sided_p)

__all__ = [
    # stats
    "cluster_mean", "cluster_diff", "two_sided_p", "_two_sided_p", "bh_fdr",
    "fmt_t", "fmt_mean",
    # features
    "_note_val", "build_features", "COLLINEAR_GROUPS", "analyze_dims",
    # forward
    "FWD_KEYS", "MAIN_METRIC", "MIN_INFER_DAYS", "load_signals",
    "fetch_forward", "load_forward_cache",
    # buckets
    "BUCKETS", "bucket_segments", "bucket_stats",
    # sections
    "_table", "_contrast_line", "summarize", "oos_section",
    "_source_of", "source_section",
    # ledger health
    "LEDGER_EXPECT", "LEDGER_WARN_PP", "_row_pair", "_ledger_rows", "_judge",
    "ledger_section",
    # cli
    "main",
]

if __name__ == "__main__":
    main()
