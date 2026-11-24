# -*- coding: utf-8 -*-
"""席位画像/预判模块（原紫阳东路专用，2026-09-20 通用化支持多席位并行挖掘）。

模块参数唯一可调处。改口径先改这里——与项目 config.py 的纪律一致，
但不污染主配置（本模块刻意不 import 项目 config，保持可整体摘除）。
"""
from __future__ import annotations

import os

# ---------- 席位注册表 ----------
# use_seat(key) 在进程内切换活跃席位（重绑下方全部席位相关常量；日K缓存
# BARS_CACHE_PKL 为市场级共享，跨席位复用不重拉）。默认席位=ziyang，
# 主流程（work/main 的 daily_sync/甄选行/关注清单）始终用默认席位，
# 其余席位仅供 mine_seat 画像挖掘，不进预判与台账。
SEATS = {
    "ziyang": {
        "code": "10026937",
        "name": "国泰海通证券股份有限公司武汉紫阳东路证券营业部",
        "short": "紫阳东路",
        # 2026-01 起骤然活跃（2025 全年上榜成交仅约 0.8 亿）；此前席位代码被
        # 散户时代复用，老记录与游资风格无关。盘口语言 9777，疑似「消闲派」
        # 新席位（未证实，佐证见 RESEARCH.md 3.8）
        "since": "2026-01-01",
        "dir": "ziyang",
    },
    "kyxadj": {
        "code": "10046745",
        "name": "开源证券股份有限公司西安西大街证券营业部",
        "short": "开源西大街",
        # 2025-07~2026-01 高频活跃（月 300+ 行）→ 2026-02~07 蛰伏 → 2026-08
        # 起回归；与紫阳东路的头号实体联动席位（同榜 10/25，多为买3/买4
        # 跟随位）。窗口覆盖两个活跃期，蛰伏期自然零行
        "since": "2025-07-01",
        "dir": "kaiyuan",
    },
}
_ACTIVE_KEY = "ziyang"


def use_seat(key: str) -> None:
    """进程内切换活跃席位（重绑模块级常量；import 方式为 `from . import config as C`
    的调用方在运行时读取，切换即生效）。"""
    global _ACTIVE_KEY, SEAT_CODE, SEAT_NAME, SEAT_SHORT, PROFILE_SINCE
    global DATA_DIR, SEAT_TRADES_CSV, PROFILE_JSON, LEDGER_CSV, REPORT_TXT
    if key not in SEATS:
        raise KeyError(f"未注册席位 {key}（可用：{sorted(SEATS)}）")
    s = SEATS[key]
    _ACTIVE_KEY = key
    SEAT_CODE = s["code"]
    SEAT_NAME = s["name"]
    SEAT_SHORT = s["short"]
    PROFILE_SINCE = s["since"]
    DATA_DIR = os.path.join(ROOT, "out", "research", s["dir"])
    SEAT_TRADES_CSV = os.path.join(DATA_DIR, "seat_trades.csv")
    PROFILE_JSON = os.path.join(DATA_DIR, "profile.json")
    LEDGER_CSV = os.path.join(DATA_DIR, "prediction_ledger.csv")
    REPORT_TXT = os.path.join(DATA_DIR, "report_latest.txt")


# ---------- 席位身份（默认=紫阳东路；use_seat 切换） ----------
SEAT_CODE = "10026937"
SEAT_NAME = "国泰海通证券股份有限公司武汉紫阳东路证券营业部"
SEAT_SHORT = "紫阳东路"

# 画像挖掘窗口：只统计该日起的席位行为（活跃期口径，各席位起点见 SEATS 注释）。
PROFILE_SINCE = "2026-01-01"

# ---------- 数据与输出路径（相对仓库根，独立于主流程 out/cache） ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "out", "research", "ziyang")
SEAT_TRADES_CSV = os.path.join(DATA_DIR, "seat_trades.csv")       # 席位明细（全量幂等覆盖）
BARS_CACHE_PKL = os.path.join(DATA_DIR, "bars_cache.pkl")         # 个股增强日K缓存（市场级，跨席位共享）
PROFILE_JSON = os.path.join(DATA_DIR, "profile.json")             # 交易画像
PROFILE_HISTORY_CSV = os.path.join(DATA_DIR, "profile_history.csv")  # 画像每日快照（迭代细描轨迹）
LEDGER_CSV = os.path.join(DATA_DIR, "prediction_ledger.csv")      # 预判台账
REPORT_TXT = os.path.join(DATA_DIR, "report_latest.txt")          # 文本报告

# ---------- 主板口径（与主工程 config.MAIN_BOARD_PREFIXES 同口径，此处自带一份） ----------
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
LIMIT_UP_APPROX = 9.8          # 主板涨停判定（龙虎榜涨跌幅口径，百分数）
# ST/退市整理股剔除在 screen.build_universe 内按名称过滤（与主工程 EXCLUDE_ST 同义）

# ---------- 点火事件定义（画像挖掘用） ----------
# 点火＝席位净买入 > 0 且当日该股涨停（≥9.8%）。实证：2026 年其净买入事件
# 89.5% 当日涨停（374/418），「买在板上」就是它的行为模式，故以涨停日买入定义点火。
IGNITION_MIN_NET_BUY = 0.0     # 净买额下限（元），>0 即计入

# ---------- 预判层 ----------
# 候选宇宙：当日涨幅 ≥ 此值才进入打分（免费快照无全市场连板数，先用涨幅收窄宇宙，
# 再对候选拉日K算连板/量比/位置等前夜特征）。阈值依据：点火前夜画像的涨幅分布
# （见 profile.json eve.pct——多数前夜自身已涨停或大涨；高低切换型企稳票由
# 「近20日曾涨停」通道补入，见 CANDIDATE_RECENT_LIMIT_DAYS）。
CANDIDATE_PCT_MIN = 3.0
CANDIDATE_RECENT_LIMIT_DAYS = 20   # 近 N 日内曾涨停的票即使今日涨幅低也进宇宙（高低切换通道）
# 动量通道（v1.4，2026-09-20 持续优化）：当日 5 日涨幅 ≥ 此值也进宇宙。
# 依据：0板回调点火（占点火 73%）的前夜 m5 中位 +13%、p25 +7.9%——回调发生在
# 上涨趋势内；阈值 5% 覆盖 0板前夜的 83%（276 事件实测校准）。修复召回洞：
# 华天科技 09-17 漏检（当日<3% 且近20日未封板→隐形）。新浪快照无 f109 时
# 由日K缓存补算。改动发生于 v1.3 首笔登记前（零核验记录），随 v1.4 预注册。
CANDIDATE_M5_MIN = 5.0
# 抬头通道：横盘基座（m5<5%）且近5日单日涨幅 ≥ 此值——覆盖「横盘启动」模式
# （全体点火前夜的 11%，华天科技 09-17 漏检实例；加通道后宇宙召回 87%→98%，
# 纯横盘沉寂残洞仅 2%）。阈值敏感性 1.5~2.5% 稳定（12~15% 覆盖）。
CANDIDATE_STIR_MIN = 2.0
TOP_N = 5                          # 每日预判清单上限（2026-09-17 用户定 3；
                                   # 2026-09-28 用户批准扩 5——当日众泰汽车真实
                                   # 点火列 #4 被 N=3 截掉。台账 precision@N 按 5 记，
                                   # N=3 时期的历史行由 model_version 分层判读）

# 打分特征与先验（v1.3 = v1.2 对数提升度+概率地板，特征集加入 ltsz 流通市值档）。
# 依据：五路深挖实测 76% 点火在流通百亿以上、0% 在 30 亿以下（n=277），
# 头两轮 0/6 miss 的病根即缺此维度；2026-09-20 用户批准合并。版本记录见
# RESEARCH.md「打分口径」。v1.3 自合并日起冻结，台账 Accuracy 唯一裁判。
FEATURES = ("lianban", "pct", "turnover", "vr", "price", "pos20", "ltsz")

# 题材共振：席位近 N 个交易日内净买过的行业，今日同行业候选加分（游资跟题材走）。
THEME_LOOKBACK_DAYS = 10

# ---------- 台账与判读纪律（先标签后计分：预注册，不得事后移动） ----------
# - 预判一经登记（predict_date+code）不得改写；核验只回填 outcome 字段。
# - 判读门槛：独立 predict_date 簇数 ≥ 8 才输出结论性判读（同日多票共享市场
#   状态，不是独立样本——与主工程 CALIB_MIN_INFER_DAYS 同纪律）。
# - 预注册指标：precision@3（top3 中席位次日实际净买占比）、ignition 命中率
#  （top3 中「席位净买且该股次日涨停」占比）、召回（席位次日主板净买事件中
#   落入 top3 的比例）。基线对照：全部主板涨停股的次日席位净买率（台账同步统计）。
# - 8 簇是最低信息门槛，不触发判读（改动一④）：结论性判读只发生在判定窗
#   （2026-10-01 起前向）的预注册时点，由冻结重放器两臂配对输出裁判。
MIN_INFER_DAYS = 8

# 执行模拟双边成本（路线图#4：与主工程 COST_ROUND_TRIP 同口径=佣金万2.5×2+
# 印花税千0.5+过户+滑点≈0.152%，硬性约0.102%+滑点5bp）。此前 exec_oo/exec_brk5
# 为毛收益，对单笔期望零点几个百分点的策略是决定性失真。2026-09-26 起为唯一
# 口径：EXEC 主指标=扣费后 exec_oo（核验日开→次日开，一字槽两侧同剔）的独立
# 预测日簇槽位均值（3 槽并列等权）；迁移日此前核验的 12 笔已一次性补扣并在
# note 标注「成本口径迁移」（ledger.migrate_cost_basis，幂等）。
EXEC_COST_ROUND_TRIP = 0.152

# ---------- v2.5-exec 预注册常量（2026-09-26 冻结） ----------
# 协议=out/research/prereg/2026-09-26-v2.5-exec/；冻结重放器=
# out/research/ziyang/deep6/wf_profit.py（两臂同框架同宇宙，月度 expanding 重训，
# 训练/标签窗严格早于测试月）。
#
# 判读纪律（改动一④）：下方 MIN_INFER_DAYS=8 只是信息下限，不触发判读；结论性
# 判读只发生在判定窗（2026-10-01 起前向）的预注册时点。开发窗 ≤2026-08-31 只出
# 设计统计；2026-09 为锚定窗只报不门（设计过程已看过 9 月）。
#
# 改动三（λ 联合分）：score_joint = rank(score_v2) + λ·rank(money_prior)。
# money_prior 三档=既有 D-晚可知特征、无缺口列：lianban≥2 连板档 / repeat01
# 记忆档（紫阳∪知春路 [D-20,D)）/ 股性档 zt60≥1 或 maxlb60≥2。先验依据
# （干净窗主板前缀+LIVE+V≤08-31）：2板+ 48.0%（n=1018）vs 首板 44.3%（n=4329）。
# λ∈{0.2,0.5,1.0} 仅在 ≤2026-08-31 walk-forward 逐月按 recall@3 选优取众数冻结
# （选定表随 wf_profit.json 落盘；众数覆盖 <2/3 折叠月则标注不稳）。
# 已执行（deep6/wf_profit.py 2026-09-26）：逐月 best λ = 04月 0.2（三档并列取小）
# / 05月 1.0 / 06月 0.2 / 07月 0.2 / 08月 0.2 → 众数 0.2，覆盖 4/5 折叠月 ≥2/3
# =稳定。逐月 recall@3 见 out/research/ziyang/deep6/wf_profit.json。
# v2.6 重冻结（deep7 Round2，2026-09-28）：候选=v2.6-tp（tp2 特征 + GPU torch
# logistic）+ λ 扫描 {0.1,0.2,0.3,0.5,0.8}（DEV 月 ≤08-31 pooled OOS EXEC 选优，
# 全窗只吃冻结值）。选定 λ=0.1（dev EXEC +0.3468 vs λ=0.2 的 +0.2763）；全窗
# EXEC +0.1841 / RECALL 0.0573 / WIN 50.4，三门过（vs 基线 -0.3042/0.0386/40.1）。
# 同轮 sklearn C 网格（0.01/0.1/1/10 × λ）最优 C=1.0,λ=0.1（dev +0.2063）不敌
# torch 臂，生产维持 GPU torch backend。明细 deep7/round2_results.json。
JOINT_LAMBDA = 0.35
#
# v2.7.1（deep7 Round5 自由发挥，2026-09-28）：λ 细扫 0.3→0.35（DEV 月
# pooled OOS EXEC +0.4073，峰值；0.4/0.45 衰减）。全窗 EXEC +0.2639 /
# RECALL 0.0617 / WIN 53.7（三门过，优于 λ0.3 的 +0.2376）。同轮否决项：
# log_uni 分档 λ（(0.5,0.2) 等，dev +0.4049≈平手不取）、近因训练窗 60/90/120
# 日（60 日崩至 -0.42）、xgb+torch rank 集成（w0.5 -0.44 / w0.7 +0.02）。
# λ∈[0.3,0.5] 为平台区（全窗 0.24~0.26），差异在噪声量级——冻结值取 DEV 峰。
# 明细 deep7/round5_results.json。
#
# v2.7（deep7 Round3，2026-09-28）：主模型 logistic → XGBoost（coordinator
# 指定 GPU CUDA）。DEV 窗（≤08-31）超参选定：{depth 2/3/4}×{n 200/400}×
# {lr 0.1/0.05} 最优 d4/n200/lr0.1；λ 随新模型重冻结 0.3（dev EXEC +0.3770，
# 全窗 EXEC +0.2376 / RECALL 0.0617 / WIN 53.7，三门过且优于 Round2 的
# +0.1841）。CUDA 不可用时自动退 CPU（同算法）。明细 deep7/round3_results.json。
MODEL_KIND = "xgb"
XGB_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
              "seed": 20260926}
#
# 改动四①（条件部署）：扫描基超集 = 缓存全部主板 code ∪ 近 20 自然日归档
# signals_*.csv code 并集，对缓存末日期 < trade_date 的票先补拉再走四通道
# （∪基按构造 ⊇ 理想旧基，补拉直接对冲 9 月式横截面塌缩）。部署门槛（预注册）：
# 归档覆盖窗扩样重跑同一覆盖校准，∪基覆盖率 ≥ 92.3%（=理想旧基 94.3%−2pp，
# 旧基四通道确切口径见协议）且补拉成功率 ≥ REFILL_SUCCESS_MIN；不过 → 缓部署
# （维持 False=旧基，仅告警形态，映射(7)）。补拉量级：陈旧票约 450~1000 只/日，
# baostock 串行 ≈2-5 分钟（估计值，上线首周实测回填）。
SCAN_BASE_SUPERSET = False
SCAN_BASE_ARCHIVE_DAYS = 20
COVER_GATE_FLOOR = 92.3
REFILL_SUCCESS_MIN = 95.0
#
# 改动四② 管道健康锚（2026-09-26 实测 34 个归档文件）：归档日行数 median
# 111.5 / mean 129.6 / range 3..372（09-24=23 为管道级异常）；20 自然日归档
# 并集健康锚 09-17=1202（塌缩周 end-09-24 仍 1054，比缓存横截面稳；8 月不可作
# 锚——end-08-31 并集仅 493，归档 08-10 起步且早期文件个位数行）。并集不独立
# 于主工程管道健康，监控必须双指标；低于锚×HEALTH_ALERT_FRAC → uni_low /
# arch_low 告警（呈现于报告运维体检，验收=触发日 100% 在报告呈现）。
UNION_HEALTH_ANCHOR = 1202
ARCHIVE_ROWS_HEALTH_ANCHOR = 111.5
HEALTH_ALERT_FRAC = 0.7

# ---------- v3.0.2 卖出记忆惩罚 + 相对低分弃权（2026-09-28 现场反馈后实证） ----------
# 事故：09-28 甄选 #2 超声电子是席位 09-21/09-23 连续净卖 1.66 亿的票（买入记忆
# repeat01 已有、卖出记忆缺失）；当日真实点火标的众泰汽车仅列 #5 被 TOP_N 截掉。
# 实证（紫阳 1129 条上榜明细，out/research/ziyang/deep/dm1_pct_dist.csv 同目录）：
#   净卖事件后 5 自然日再买 5.7% vs 买入事件后基准 10.9% → log-LR = -0.65
# 同轮否决项：pct<-5% 硬剔除——紫阳点火前一日 ≤-5% 的占 13.6%（48/354，净买
# 8.3 亿），他确实接飞刀，硬剔除会误杀；崩跌票问题是超配不是全禁。
SELL_NET_YI = 0.3            # 净卖额下限（亿）：小额卖出是噪音不算记忆
                             # （v3.0.2 的 SELL_PENALTY=-0.65 后处理已废——对照
                             # 检验证明卖出抑制只在年轻持仓成立，收敛为 compose
                             # 里的 sold_young -0.48 冻结 log-LR）
SELL_MAX_AGE = 10            # 卖出抑制档位边界：上次买入≤10交易日（对照检验
                             # age≤10: 1.66% vs 持有 2.68%；age11~60 无差异）
# 相对低分弃权：v3.0 在线续训后分数尺度漂移（旧 3~5 → 现 ~1.2），绝对阈值全部
# 失效。改为滚动自校准：当日 top1 < ABSTAIN_REL × 同模型版本近 20 日 top1 中位
# → 弃权（predictions 置空、排名快照照落）。历史不足 ABSTAIN_MIN_HIST 日不开门。
ABSTAIN_REL = 0.60
ABSTAIN_MIN_HIST = 5
ABSTAIN_HIST_FILE = os.path.join(DATA_DIR, "top1_history.json")

# ---------- 网络 ----------
HTTP_TIMEOUT = 8
FETCH_RETRIES = 3
FETCH_SLEEP = 1.5
BARS_SLEEP = 0.15        # 单只串行拉日K的间隔（秒），防限流
# 并发批量拉取（2026-09-18 三轮迭代：时间感知四级源链）。盘后（end 日 ≥17:35）
# 批量段由 baostock 串行承担（真换手率、无 WAF、~5 只/秒）；线程池只兜底
# 失败码与盘中场景（东财→腾讯→新浪），6 线程为兜底相上限。
BARS_WORKERS = 6         # 兜底相线程数（baostock 串行相不用线程：会话非线程安全）
SNAP_WORKERS = 4         # 快照分页并发数（仅东财；新浪串行防 456）
SNAP_PROBE_TIMEOUT = 3   # 快照主机探测超时（秒）——3 台串行最坏 9s 而非 24s
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
