# -*- coding: utf-8 -*-
"""calib：评分校准包（2026-09-15 自 calibrate.py 拆分，六关注点分治）。

模块一览：
  stats          聚类推断统计核心（纯函数，research_scan/prereg 研究脚本共用）
  forward        信号装载 + 前向数据获取/缓存（口径常量在此）
  features       维度布尔特征 + BH-FDR 区分度分析
  buckets        分数桶方案 + bucket_stats.json 统计构造（单一实现）
  sections       报告文本区块（分桶表/汇总/OOS/来源分层）
  ledger_health  台账健康度判读（vs config.EXPECT 预注册期望）
  cli            装配与落盘（_forward.csv 防覆写保护）

兼容门面 calibrate.py 继续作为脚本入口（pipeline.run_calibrate_quiet /
run_daily.bat 以子进程方式调用），并原样重导出 calibrate 时代的全部公开名
（含 _two_sided_p 等已被研究脚本导入的事实公共 API）。

⚠ 本文件与 calibrate.py 的重导出清单互为镜像（2026-11-20 曾因双清单漂移把
死代码 _flag/_int0 多处复制）。不能让 calib/__init__ 反向 import calibrate
（calibrate 的首行 from calib.x import 会触发本包部分初始化 → ImportError），
新增导出名时两处同步改。
"""
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
