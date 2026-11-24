# vane 三日模型 · 口径冻结（2026-11-22 立项，先标签后计分纪律）

## 定位

情绪风向标的三日自学习模型（`vane/model.py`）：学习「B≥2 连板候选次日晋级」的
历史规律，对 4板晋5/5板晋6 候选出晋级概率分并**主排名额**。节奏与 ziyang 月度
模型对齐功能面（训练/重训门/自愈/校准/微调/对照/台账归因），节奏更快以适配
短线情绪生态的高频漂移。

## 冻结口径

| 项 | 口径 | 依据 |
|---|---|---|
| 训练宇宙 | 每日主板非ST封板股中本轮板数 **B≥2 全体**（B 作特征；推理只对 B∈{4,5}） | 用户拍板；样本经济（ziyang 16万行来自宇宙设计非数据量） |
| 标签 | y = code ∈ T+1 涨停池（晋级成功）；纯池口径**不剔一字**（一字晋级也是情绪延续） | 与 vane/ledger.verify 同口径 |
| 特征 | config.FEATURES 12 列，全 T 收盘可知、纯池口径（B/span/breaks/封成比/封板时间/价格/市值/市场高度/2板+家数/行业热度×2/高度趋势） | 无未来函数；vane 不引 bars |
| 标准化 | poolmodel 式 mu/sd 存档（**非日内分位**——池宇宙日截面 5-15 行，分位退化） | 分位在 inference 小截面退化为常数 |
| 双日纪律 | 训练日 ≤ cutoff（窗口倒数第二日）、标签日 < cutoff | 零泄漏 |
| 三日门 | trained_at 距今（**交易日**）≥ CADENCE_DAYS=3 → 重训；坏件隔离 .corrupt；重训失败仅同版本同特征回退 | trained_at 锚 as-of 交易日（非墙钟） |
| 每日微调 | xgb: warm-start 5 棵 lr=0.005（显式传参）；logreg: 全量重算（近期半衰 7 日加权）；幂等守卫 refit_at==today skip；**不动 trained_at**（不吞三日节奏） | ziyang cutoff/trained_at 分离先例 |
| 排序 | **模型主排**（样本门过时）+ 启发式 shadow_rank 影子登记 | 用户拍板 2026-11-22 |
| 台账归因 | model_version/model_score/shadow_rank 三列；summary().by_model_version 晋级率分层 = 「模型 vs 启发式」持续实证 | 先标签后计分 |
| 样本门 | 正例 ≥5 且 行数 ≥20（config.SAMPLES_MIN_POS/ROWS）；不过 → ok=False，启发式排序如实呈现 | 小样本防硬训 |
| xgb 门 | n ≥ 120 才启用（cuda→cpu 兜底）；xgboost 缺失 → 回落 logreg（不误隔离工件） | 小样本 xgb 过拟合 |

## 实测基线（2026-11-22 首训）

- 窗口 20 交易日（缺池 5 日）、样本 220 行 / 正例 66（基率 30%）；
- 分层晋级率：2板 29%、3板 37%、4板 29%、5板 20%、6板 50%（n 小慎读）；
- kind=xgb-cuda；首日三分模型分并列于 iso 平台 → 排序回落启发式 tiebreak
  （影子#与最终名次一致）——**已知局限**：小样本+iso 平台期区分度有限，
  随样本积累观察 by_version 实证后再评估是否上调模型权重。

## 继承的三个已知坑

1. 原子写 tmp 必须保留原扩展名（xgboost save_model 按扩展名选 UBJSON/文本
   ——2026-11-22 model_xgb.json 事故，fstore/seatdata 已修）；
2. xgb 续训必须 `xgb.train(params, num_boost_round=N, xgb_model=booster)`
   显式传参（update 的 iteration 是轮次编号不是棵数）；
3. 模型任何异常必须打印留痕不静默；verify 失败不得阻断 daily_sync（步骤隔离）。

## 待办（v2 候选，未冻结）

- bars 扩展训练历史（池仅 3 周保留；bars 可回溯 ~100 日重建涨停日序列）；
- 一字可参与性标注（T+1 开盘一字=观察强延续但不可参与，分列两口径）；
- 弃权/低置信门（相对 top1 中位，ziyang ABSTAIN 模式）——先积累 by_version
  实证再决定是否引入。
