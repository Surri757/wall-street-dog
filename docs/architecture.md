# 系统架构蓝图（2026-08-30 全面重构后；2026-09-15 阶段3：calib 拆包 + 台账结算语义）

## 分层总图

```
┌──────────────── 入口层（可执行编排，无业务逻辑）────────────────┐
│  main.py   CLI 入口：参数解析 → pipeline.run_day → 报告/清理   │
│  work.py   交互菜单入口：同 main，终端交互                      │
│  run_daily.bat  定时自动运行（收盘后 main + calibrate）         │
└───────────────────────────┬───────────────────────────────────┘
                            │ 调用
┌───────────────────────────▼───────────────────────────────────┐
│  pipeline.py  交易日流水线（五阶段显式步骤，见下）               │
└──┬───────────────┬────────────────┬─────────────────┬─────────┘
   │               │                │                 │
┌──▼──────┐  ┌─────▼─────┐  ┌───────▼──────┐  ┌──────▼────────┐
│candidates│  │ analyzer  │  │  mood.py     │  │ report.py     │
│候选池构建 │  │ 分析编排   │  │ 情绪周期引擎  │  │ 报告+归档+提示 │
│(纯变换)  │  │ (并发)     │  │ (六阶段判定)  │  │ (展示层)      │
└──┬──────┘  └──┬────┬────┘  └───────┬──────┘  └───────────────┘
   │            │    │               │
   │        ┌───▼──┐ └──────────┐    │
   │        │ bars │  二板评分   │    │
   │        │日K获取│ second_board│   │
   │        └───┬──┘  (五维)     │    │
   │            │  选股规则      │    │
   │            │  board_rules   │    │
   │            │                │    │
┌──▼────────────▼────────────────▼────▼─────────────────────────┐
│  data_source.py  数据源门面（唯一访问口）                        │
├────────────────────────────────────────────────────────────────┤
│  datasources/   http → spot/kline/pools/concepts → calendar    │
│                 util（共享解析叶子，无依赖）                     │
├────────────────────────────────────────────────────────────────┤
│  cache_store.py 滚动缓存   config.py 参数口径（唯一可调处）       │
└────────────────────────────────────────────────────────────────┘
  横切：calib/ 包（评分校准：stats 聚类推断纯函数 / forward 前向数据 / features
  维度分析 / buckets 分桶 / sections 报告区块 / ledger_health 台账判读 / cli 装配
  落盘；兼容门面 calibrate.py 保留为子进程入口与 API 重导出）与
  backfill.py（历史回灌）/ research_scan.py（近期K线事件研究，窗口锚点冻结、
  统计走 calib.stats）/ trade_rules.py（排板成交模型与卖出规则回测，
  limit_price=涨停价整数分值精确取整的单一实现）
  → 只读归档与数据源门面，不进生产链
  report.py 职责：情绪速览（含周期总仓位）+ 明日关注清单（三模式统一关注分）
  + 打板画像/+5%盯盘区块（入口层经 pipeline.load_spot 传入当日快照，对照给规则动作）
  pipeline 另含：archive_cycle_series（周期逐日指标→archive/cycle_series.csv 长期归档，
  幂等追加）+ doctor_check（数据质量自检：换手率量纲/封板数量级/K线口径/量比分布，
  run_day 成功后自动跑，main --doctor 单独触发；播种的旧 bars 不参与口径对比）
  单位契约（tests 强制）：spot/kline 的 turnover=小数、volume=手、pct=百分数——
  东财快照原生百分数换手率在 _em_rows_to_frame 归一（2026-09-01 审计修复）
```

依赖规则（test_module_contracts 常驻强制，共七条）：
1. **门面唯一**：业务层只能经 `data_source` 访问数据源，禁止直连 datasources 子模块
2. **子包单向**：datasources 内只允许依赖 http/util/config/cache_store，禁止横向私有导入
3. **分层单向**（精确层号，测试强制）：L1 纯数据 {candidates, bars} → L2 领域 {mood,
   second_board, board_rules} → L3 编排/展示/台账 {analyzer, report, ledger} →
   L4 pipeline → L5 入口 {main, work}；
   board_rules=纯规则层（池档位/打板画像/+5%档位，渲染与记账共用单一实现）；
   ledger=台账层（开盘池/打板核账，幂等 upsert，数据自己说话）；
   只允许向低层导入，同层白名单 second_board→mood（评分读周期分值表）；
   入口层互调禁止（work 曾依赖 main.collect_data，已改为直连 pipeline.run_day）
4. **ctx 契约**：second_board 消费的每个 ctx 键必须有生产者（防键名漂移→静默中性分）
5. **无未来函数守卫**（test_no_lookahead_fields）：决策规则（board_rules/
   second_board/report/ledger）只准访问收盘可知字段白名单——新字段必须
   先确认当日 15:00 收盘可知才可入白名单，否则即数据泄露（回测虚高实盘亏损）。
   已知口径近似（非泄露，已标注）：打板画像 vr 为全日量比，盘中以触板时点
   实时量比近似（缩量板封板后成交占比小）。
6. **测试离线完整性**：网络函数的 mock 补丁必须打在真实消费模块（bars.fetch_daily），
   曾因补丁点漂移导致"离线"测试实际触网（套接字封禁验证已入验收流程）
7. **calib 包独立于生产链**（2026-09-15 起）：calib/* 禁止 import 生产 L3+ 模块
   （report/ledger/analyzer/pipeline/main/work）、禁止绕过 data_source 门面访问
   datasources 子包。研究工具（research_scan/prereg 脚本）可以 import calib.stats
   共享聚类推断——此前 research_scan 自带裸均值对比，恰好违反 calibrate 文档写明
   的纪律，单一实现后该分叉不可能再发生。

## 交易日数据流（pipeline 五阶段）

```
A load_cached_result ──命中──→ 直接返回 (results, skipped, cands, ctx)
        │ 未命中
B seed_bars_from_prev  上一交易日 bars 播种（增量拼接原料）
C acquire_spot         盘中保护(09:15~15:05拒新拉) → 东财快照→新浪降级；历史日只读缓存
D candidates_from_spot 快照→主板候选（曾涨停/曾跌停/振幅≥7%，向量化 ~5ms）
E assemble_market_context
    ├ pools.fetch_pool_extras     涨停/炸板池+昨日池（连板/封单/行业/晋级率/赚钱效应原料）
    ├ fill_industry_local         池缺失时行业兜底
    ├ fetch_concept_boards        概念快照（失败→A维行业口径）
    ├ 昨日池×今日快照             赚钱效应 yzt_today_pct
    └ mood.stage_for_today        情绪周期阶段（高度/晋级/炸板/赚钱效应序列）
F analyze_all（并发）→ second_board 五维评分 → cache_store.save_result → report 归档
```

降级矩阵（每级失败的行为都是显式的）：
| 数据源 | 失败行为 |
|---|---|
| 东财快照 | → 新浪快照（open/volume 已补齐，实测与东财逐位一致） |
| 东财行情集群整体断连 | 快照走新浪；日K走腾讯；概念回退行业口径 |
| 今日池 | 评分中性分+缺失惩罚(-2/项)，报告标注 |
| 昨日池 | B维昨日口径缺失、晋级率/持续题材缺失（中性+惩罚） |
| 概念接口 | A维自动回退行业口径（打印提示） |
| 单股日K | 该股记 skip_reason，不影响其他 |

## ctx 契约表（市场环境显式契约）

| 键 | 生产者 | 消费者 |
|---|---|---|
| industry_count / industry_max_lianban / industry_yesterday_count / persistent_industries | pools.fetch_pool_extras | second_board A维 |
| today_2plus_count / yesterday_zt_count / promotion_rate | 同上 | second_board D维 |
| yesterday_fbt / yesterday_turnover / yesterday_seal | 同上 | second_board B维 + analyzer 富集 |
| market_max_lianban / lianban_ladder | 同上 | report 情绪速览 |
| concept_boards | pipeline (concepts.fetch_concept_boards) | analyzer._attach_concepts |
| yzt_today_pct / yzt_ret | pipeline / analyzer.analyze_all | mood 序列 / second_board D维 |
| cycle_stage / cycle_note | pipeline (mood.stage_for_today) | second_board D维 + report |
| break_rate | analyzer.analyze_all（全体候选口径） | second_board D维 |
| _stock_concepts / _board_kline | analyzer 运行内缓存（随滚动缓存持久化） | analyzer 概念挂载 |
| zt_pool_ok / zb_pool_ok | pools | second_board 缺失计数 |

## 关键设计决策（决策记录）

1. **门面而非直接包导入**：data_source/screener 双门面让拆分对外零成本；契约测试第4条
   防止绕过。
2. **ctx 用 dict + 契约测试，而非 dataclass**：ctx 由 pools/mood/pipeline/analyzer 四个
   生产者共同装配，dataclass 需要单一构造点、改动面大；契约测试提供同等的键漂移防护，
   且保留 dict 合并的灵活性。若未来生产者收敛为单一 MarketContext 构建器，可再升级。
3. **bars 三级获取在独立模块**：增量拼接的精度保障（昨收一致性校验）与获取策略
   是同一关注点，与候选构建/评分无关。
4. **pipeline 阶段函数化**：每阶段可独立测试（盘中保护、历史只读、环境装配），
   main/work 退化为纯入口，未来加调度器/推送只替换入口层。
5. **util 叶子模块**：pools/concepts 共享解析工具下沉，斩断子包横向私有依赖。
6. **横切工具不进生产链**：calibrate（calib 包）/backfill 只读 out/archive 与
   数据源门面，与 pipeline 无 import 关系，可随时独立演进。
7. **calib 拆包而非改写**（2026-09-15）：1018 行单文件按关注点分治为 7 个模块，
   兼容门面 calibrate.py 重导出全部旧 API——tests 与 prereg 研究脚本零改动，
   pipeline/run_daily.bat 的子进程调用不变。cluster_mean/_two_sided_p 等曾被
   研究脚本导入的私有名属事实公共 API，门面显式保留。
8. **台账按预期日结算 + 成交模型**（2026-09-15，README 待办#5）：卖出价取
   「买入日次一交易日」开盘（ledger._SpotBook 回读缓存 spot；停牌感知顺延），
   而非「下次运行日」开盘——跳日运行不再把 T+1 卖悄悄变成 T+N 卖。打板台账
   三态生命周期（登记→D2 成交判定→D3 开盘卖，R1 封板顺延一次）与
   trade_rules.simulate_fill/simulate_exit 同口径；涨停价/一字判定只经
   trade_rules.limit_price（整数分值取整；与 candidates.limit_prices 的网格
   恒等由测试强制——两份实现曾在 .5 边界各错一个方向）。
9. **旁路子包不进 LAYER 表**（2026-09 ziyang 先例，2026-11-19 vane 沿用）：
   `ziyang/`（席位预判）与 `vane/`（情绪风向标：4板晋5/5板晋6 三名额筛选，
   同轮断板间隔≤3个交易日、久远断档重置）均为旁路子包——自带 config/台账/
   CLI（`python -m vane scan|verify|report`，数据落 `out/research/vane/`），
   数据访问经 data_source 门面，但不 import 生产链业务层，故不受
   test_module_contracts 的 LAYER 约束。主流程挂点约定（与 ziyang 一致）：
   main.py 在 write_report 前 try/except 调 `daily_sync`（历史日/盘中守卫
   跳过、失败不阻断）。**vane 主报告挂点为报告开头独立区块**（2026-11-22
   用户两轮约定：「单独列出来不要融合」→「像关注清单那样显眼、放报告开头」）：
   report.py 在表头框后、决策摘要前渲染 `section_lines` 的「▎情绪风向标」
   整节（零网络只读台账，星级=模型 score100 分档、板链、模型状态行）——
   不并入「明日决策摘要」、不进明日关注清单统一关注分；全量细节在独立报告
   out/research/vane/report_latest.txt。
   **vane 三日自学习模型**（2026-11-22 立项，口径冻结见 vane/RESEARCH.md）：
   训练宇宙=B≥2 全体（标签=次日晋级，纯池口径）；Round6 后冻结为 logreg C=0.5
   + Platt 平滑校准（auto 模式仍保留 n≥120 的 xgb 门），三日门重训 + 坏件自愈 +
   每日微调（不动 trained_at）+
   retrain_diff 新旧对照；模型主排 + 启发式 shadow_rank 影子登记，
   台账 model_version 三列 + by_model_version 晋级率 = 「模型 vs 启发式」
   持续实证（先标签后计分）。
   **vane 换手门槛**（2026-11-25 用户规则，口径见 vane/RESEARCH.md）：本轮（涨停日
   +断板日）至少 3 个交易日换手 ≥10% 才占名额、不设上限；断板日换手「一天都不
   可以少」——本地换手库 `out/research/vane/turnover/<日期>.csv` 优先 → 东财日K →
   新浪日K → baostock → 紫阳日K缓存/近几日快照，在线源整源不可用即本进程熔断，
   仍缺则在同步摘要里报警；门槛外的票进落选观察并记入 last_scan，主报告区块如实列出。
   **涨停/炸板池取数稳定性**（2026-11-25）：东财历史池阵发性返回不完整的旧快照（实测 11-19
   有 20% 的请求只有 17/23 行，少的是尾盘封板的票；同一晚不同运行连板数不同）。`pool_raw`
   统一兜：行数不足响应自报总数 → 换新连接重取；每日池落本地存档 `out/archive/pools/`
   （只增不减，兼作超出在线保留期后的数据源）；互相矛盾的版本不合并只提示；当天盘中不读不写。
   **auction/（2026-11-24 立项，research_log 五Z）**：竞价采集旁路子包——外部策略池
   「竞价族」（竞价量 ≥ 昨日量 3% 等）历史不可回补（腾讯分时仅保留近 5 个交易日），
   故每日盘后前向采集。`datasources/auction.py`（`fetch_minute_days`：近 5 个交易日
   分时，首点=集合竞价、末点=当日总量）经门面导出；`auction.daily_sync` 读上一交易日
   归档 `signals_*.csv` 的封板/炸板行 → 台账 `out/research/auction/ledger/<执行日>.csv`
   （按执行日分片、原子写；**整天成功才落盘**——任一请求失败或执行日分时不完整
   则不写、下次重试；保留期内自动补采，补采窗取自门面 `AUCTION_RETENTION_DAYS`）
   + 执行日分钟路径 `minute/<执行日>.pkl`（gitignore，为将来"触发时点真实量比"
   研究留的原料）。
   **只读行情写研究台账：不进报告、不进评分、不改关注清单**——与 vane 的「独立成节」
   不同，因为它没有可展示的信号，只有积累中的样本。假设 A1~A3 与判定线预登记于
   PREREG 第七节，读数 `python -m auction report`（样本不足只计数不判读）。
   同轮新增 `http.assert_public_url`：出站 URL 守卫（只放行 http/https 且主机解析到
   公网，拒绝 localhost / 环回 / 私有 / 保留地址），`fetch_minute_days` 每次请求前调用。
   ⚠ datasources 内必须用包内相对导入（`from .http import …`）——绝对导入会命中标准库
   `http` 包，`data_source` 随之整体导入失败（2026-11-24 测试拦下）。
10. **第三轮质量：全项目审查落地（2026-11-24）。** 三个只读审查 + 逐条核实后修掉
   30 余处，全部有离线测试（`tests/test_logic.py::test_round3_*`）。沉淀成守则的几条：
   - **落盘只走 `fstore`**：vane / ziyang 自带的原子写助手（三套、行为已分叉）并入；
     fstore 补齐读侧（`read_json`）与序列化（`write_json` / `write_via` /
     `write_dataframe_csv`）。追加日志是唯一例外（with 保证句柄即可）。
   - **失败 ≠ 空**：「拉取失败 / 日历不可用 / 标签未发布」必须显式传播，不得退化成
     「该日无板」「非空池=交易日」「标签=0」。vane 日历不可用直接放弃当天（历史池对
     非交易日回声最后交易日，会把周五的板数三次并写进不可改写的台账）；训练样本只取
     最后一个缺池日之后的连续覆盖尾段；ziyang 日续训按「龙虎榜已发布水位」跳过标签
     未知日；backfill 无日历拒跑；历史补录涨停池失败即弃（不再存零封板结果）。
   - **缓存「是否最终」单一判据**：`cache_store.is_intraday_cache`——时间戳落在复盘日
     当天且早于收盘缓冲即盘中数据，与今天几号无关。新快照落地即作废旧结果；上一轮
     盘中运行留下的日K末根丢弃后按收盘快照重拼；盘中写的昨日缓存不播种。快照缺行
     超过容忍（10 只）即换源，轻微缺行可用但不落缓存。
   - **模型训练 / 推理同变换**：vane xgb 训练喂标准化特征、推理与续训却喂原始值
     （盘上工件两种输入的输出秩相关仅 0.54）——统一经 `_zspace`；重训 / 续训后旧
     booster 必须出进程缓存；新模型无校准表时清掉旧 iso。
   - **研究口径名实相符**：research_scan 的「首板一字」实为「二板日一字」（事件日
     缺口），已拆成 `first_word` / `event_word` 两列并重建 events.csv（research_log
     五AA 勘误）。连板数只认涨停池 `lbc`；炸板池「N天M板」的天数不再是连板数。
   - **推断表内部一致**：Δ 与 t 同源（按日配对），配对日不足不打 ★；`two_sided_p`
     新增 `df` 走 Student t（缺省口径逐位不变，既有判读不受影响）。
   - **ruff 基线**：`ruff.toml` 只开 F / E9 / ARG / SIM115 / PLW1510，全仓零告警。
   - **改签名必查生产调用点**：`model_rank_key` 去参后测试改了、`vane/__init__` 没改，
     TypeError 被 daily_sync 的通配捕获 → 模型主排静默退化为启发式（审查代理抓到）。
     旁路模块的「失败不阻断」通配只许包住取数，不许包住编排本身。
11. **第四轮性能调优：共享预计算 + 增量窗口，不改策略口径（2026-11-24）。**
    热点均先记录输入规模、耗时和结果指纹，再做等价重写；`test_round4_optimization_equivalences`
    常驻覆盖快速路径与原实现逐值一致。
    - `research_scan.scan_events`：日K数值列一次转成矩阵；有序日期二分区窗，乱序输入保留
      原过滤顺序。3,049 只股票 / 2,999 事件扫描约 1.30s→1.05s。
    - `vane.model._pool_view`：交易日索引、个股涨停日、市场高度和行业统计每个窗口只算一次；
      `engine.streak` 可复用位置索引，旧无缓存调用路径保留。合成 20 日 × 300 股特征基准
      0.178s→0.020s。
    - `ziyang.themed._rolling_tag_unions`：按股票维护 Counter 和窗口左界，历史事件只进入/
      离开窗口一次，替代「日期 × 窗口 × 股票」反复过滤整张事件表。17,672 行特征重建
      9.12s→1.24s，CSV 指纹不变。
    - `ziyang.poolmodel._next_bar`：有序日期用 `bisect_right`，21.6 万次调用 0.60s→0.054s；
      `seatdata.idx_on` 增加最新日快速路径并供 `scorer` / `screen` 复用，最新日市场触板统计
      约 0.35s→0.04s，历史日仍精确匹配。

## 模块规模（2026-09-15 更新）

| 模块 | 行数 | 职责 |
|---|---|---|
| pipeline.py | ~350 | 五阶段流水线 + 子进程链跑 |
| analyzer.py | ~155 | 分析编排 |
| candidates.py | ~125 | 候选池构建（round2 精确取整） |
| bars.py | ~95 | 日K获取 |
| screener.py | 20 | 门面 |
| ledger.py | ~640 | 台账（_SpotBook 预期日结算 + 打板成交模型） |
| calibrate.py | 62 | 兼容门面（API 重导出 + 子进程入口） |
| calib/stats.py | 136 | 聚类推断纯函数（公共统计库） |
| calib/forward.py | 148 | 信号装载 + 前向数据获取/缓存 |
| calib/features.py | 166 | 维度特征 + BH-FDR 分析 |
| calib/buckets.py | 57 | 分桶统计单一实现 |
| calib/sections.py | 320 | 校准报告文本区块 |
| calib/ledger_health.py | 119 | 台账健康度判读 |
| calib/cli.py | 155 | 装配落盘（_forward.csv 防覆写） |
| research_scan.py | ~510 | 事件研究（窗口冻结 + 聚类统计） |
| main.py / work.py | ~90/~110 | 入口 |
| datasources/* | 52~390 | 数据源分层 |
