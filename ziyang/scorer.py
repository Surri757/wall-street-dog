# -*- coding: utf-8 -*-
"""v2.0 甄选器：日内分位 logistic（deep2/empirical.py 方案 e2 的生产化移植）。

四路深讨定稿（2026-09-20）：
- 设计席证明 v1.x log-lift 的结构缺陷（涨停证据双计分、模式混合、73% 的
  回调模式数学上不可登顶）；
- 数据席走样前验证：e2 分位 logistic 日均 recall@3 0.025→0.068（2.7 倍，
  配对自助 95%CI [+0.006,+0.084]），通道配额全部失败；
- 系数画像（跨月符号稳定）：席位次日买「低价、大市值、高换手、有涨停基因、
  买过的票」——当日涨幅与动量通道系数为负（学习出的软模式配额）。

口径纪律：
- 训练与推理共用本模块特征实现（挖掘口径=打分口径）；
- 月度滚动重训，训练日与其标签日（次日）均严格早于 cutoff，零泄漏；
- v1.4 log-lift 保留为影子并行（shadow_score 落排名快照，A/B 前向对比）；
- 模型元数据落盘 model/model.json（系数/训练窗/样本量），可复算可审计。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

import fstore

from . import config as C

MODEL_DIR = os.path.join(C.DATA_DIR, "model")
MODEL_JSON = os.path.join(MODEL_DIR, "model.json")
MODEL_PREV_JSON = os.path.join(MODEL_DIR, "model_prev.json")   # 上一次重训的旧口径存档（改动四④对照物）
MODEL_XGB_JSON = os.path.join(MODEL_DIR, "model_xgb.json")     # v2.7 xgb booster 存档（与 model.json 元数据成对）
ISO_JSON = os.path.join(MODEL_DIR, "iso.json")                 # v3.1 校准层断点（isotonic X/Y thresholds）

# 15 特征（v2.1 = e2 的 14 维 + it_ztc_pct）
# it_ztc_pct=(当日涨停触及家数/50)×rank(个股pct)：deep3 特征席唯一过 CI 的增量
# （recall@3 0.0654→0.0702，6/6 月不劣，系数 6/6 月为负=高潮日绕开高位强票）。
# 交互特征在 logit_matrix 中免二次分位（与 deep3/empirical3.py 口径一致）。
LOGIT_FEATS = ["lianban", "pct", "turnover", "vr", "log_price", "pos20",
               "log_ltsz", "m5", "zt60", "maxlb60", "repeat01",
               "ch_switch", "ch_momentum", "ch_stir", "it_ztc_pct",
               "theme_max_b", "theme30_b",
               "tp_touch", "tp_am"]
# v3.1 裁决（2026-09-28 walk-forward 保留门）：记忆三列作为 XGB 附加特征
# EXEC +0.26→-0.01 被否决（与 repeat01 共线碎片化）；改为**校准合成**——
# logit(isotonic(P)) 上直接加冻结 log-LR（hazard LUT / squad +1.10 /
# sold_young -0.48），EXEC +0.2642 / RECALL +30% / WIN 54.5 三门过且不退步。
# 模型输入保持 19 列不变；记忆参数见 attach_memory_features 与 compose_scores。
# own_pick 走后处理惩罚（screen.py），不做训练特征——避免反馈环（鸡生蛋）
_RAW_FEATS = {"it_ztc_pct"}          # 已含分位成分，直接取原值（缺失填 0）
# 口径版本：标签口径变更（如 v2.2 联合目标）与特征集变更都强制重训。
# v2.3（2026-09-26 路线图#1）：训练/推理三处口径分叉修复——①通道互斥链
# （训练 momentum/stir 此前不排除 switch，生产是全互斥）；②lianban 特征阈值
# 9.5→9.8（对齐生产 feats_from_bars；股性 zt60/maxlb60 保持 9.5=生产同口径）；
# ③repeat01 记忆集推理侧改「紫阳∪知春路、[D-20,D) 半开窗」（对齐训练语义，
# 修复见 screen.predict）。
# v2.5-exec（2026-09-26 预注册，协议=out/research/prereg/2026-09-26-v2.5-exec/，
# 改动二）：训练标签可执行化——y = 席位∪知春路 D+1 净买 ∧ 核验日 V 开盘非一字；
# 一字剔除对称化（V 开一字行正负例同删，与 EXEC 口径一致）；不设 exec_oo>0 子句
# （干净窗实测可执行席位事件 exec>0 占比 95.1% n=286，该子句只会把 ~4.9% 真席位
# 事件按无横截面结构的次日收益方向翻成负例=标签噪声）；禁纯「次日为正」标签
# （干净窗基率无结构）。标签只需 D+1 数据，cutoff 语义维持严格 <（训练日仅少
# 08-28）。采纳/回滚由判定窗两臂门控（映射(1)/(5)/(6)）。
MODEL_VERSION = "v3.1-compose"
# v3.0 四项 TIER1（deep8 七路共识）：
# ① own_pick_decay/streak（实证席：65% 槽位被重复旧宠吃掉 -90%）
# ② 每日小步续训（Kaggle Jane Street 金牌：相对 +70%）
# ③ 横截面分组（Optiver 冠军）——已由 pct_mode 日内分位实现，无需额外改
# ④ meta-abstain（理论席+实证席共识：弃低置信 30% 日）
_MB = C.MAIN_BOARD_PREFIXES   # 主板口径（包内单一源，此前 4 处手抄）
_ZT = 9.5          # 股性口径（zt60/maxlb60，生产 screen.build_universe 同为 9.5）
_LB = C.LIMIT_UP_APPROX   # lianban 特征阈值（9.8，与生产 approx_lianban_series 一致）


def _channels(strong: bool, switch: bool, m5, stir5) -> tuple[bool, bool, bool]:
    """互斥通道链——与生产 screen.build_universe 完全一致（路线图#1 修复点①）。

    生产语义：momentum 要求非强势非切换；stir 要求非强势非切换非动量且 m5 低于
    动量阈值（横盘基座）。训练此前 momentum/stir 均未排除 switch，26% 行两通道
    同真、ch_momentum 负系数特征被生产互斥链摘除，分数语义漂移。
    """
    momentum = (not strong and not switch and m5 is not None
                and m5 >= C.CANDIDATE_M5_MIN)
    stir = (not strong and not switch and not momentum and m5 is not None
            and m5 < C.CANDIDATE_M5_MIN and stir5 is not None
            and stir5 >= C.CANDIDATE_STIR_MIN)
    return switch, momentum, stir


def _feat_extractor(name: str):
    """特征名 → 行提取器（logit_matrix 按 LOGIT_FEATS 动态取列，
    使特征集变更（如 +theme 两列）无需改矩阵代码——路线图#11 前提）。"""
    if name == "log_price":
        return lambda r: math.log10(r["price"]) if r.get("price") else np.nan
    if name == "log_ltsz":
        return lambda r: math.log10(r["ltsz"]) if r.get("ltsz") else np.nan
    if name == "repeat01":
        return lambda r, _rep=None: 1.0 if r["code"] in _rep else 0.0  # placeholder
    if name in ("ch_switch", "ch_momentum", "ch_stir"):
        key = {"ch_switch": "via_switch", "ch_momentum": "via_m5",
               "ch_stir": "via_stir"}[name]
        return lambda r: 1.0 if r.get(key) else 0.0
    return lambda r: r.get(name)


def logit_matrix(rows: list[dict], repeat_set: set, pct_mode: bool = True) -> np.ndarray:
    """行 → 特征矩阵（按 LOGIT_FEATS 动态构建）。pct_mode=True 为日内分位
    （e2 口径，抗市场状态漂移）；_RAW_FEATS 免分位直接取原值。"""
    def col(vals):
        v = pd.Series(vals, dtype=float)
        if v.notna().sum() == 0:
            return np.zeros(len(v))
        v = v.fillna(v.median())
        if pct_mode:
            return v.rank(pct=True).to_numpy()
        return v.to_numpy()

    def raw(vals):
        v = pd.Series(vals, dtype=float)
        return v.fillna(0.0).to_numpy()

    columns = []
    for name in LOGIT_FEATS:
        if name == "repeat01":
            vals = [1.0 if r["code"] in repeat_set else 0.0 for r in rows]
        else:
            ex = _feat_extractor(name)
            vals = [ex(r) for r in rows]
        columns.append(raw(vals) if name in _RAW_FEATS else col(vals))
    return np.column_stack(columns)


def market_zt_touch_count(bars_cache: dict, trade_date: dt.date) -> float:
    """当日主板「触及涨停」家数（盘中最高涨幅≥9.5，含炸板）。

    用 bars_cache 截面近似（~800只主板子样本，与训练口径一致，低估绝对值
    但日间相对可比——特征只用其日间变化信息）。
    """
    iso = trade_date.isoformat()
    n = 0
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in _MB:
            continue
        dates = b["date"].tolist()
        if iso not in [str(d)[:10] for d in dates]:
            continue
        i = dates.index(next(d for d in dates if str(d)[:10] == iso))
        if i < 1:
            continue
        hi = float(b["high"].iloc[i])
        prev = float(b["close"].iloc[i - 1])
        if prev > 0 and (hi / prev - 1.0) * 100.0 >= _ZT:
            n += 1
    return float(n)


def attach_it_ztc_pct(rows: list[dict], ztc: float) -> None:
    """交互特征：(ztc/50) × 个股 pct 的宇宙内分位（deep3/empirical3 口径）。

    分位按行位置映射（pct 相同的行同分位，避免浮点键碰撞）。
    """
    idxs = [i for i, r in enumerate(rows) if r.get("pct") is not None]
    rmap = {}
    if idxs:
        s = pd.Series([rows[i]["pct"] for i in idxs], dtype=float)
        ranks = s.rank(pct=True)
        rmap = {idxs[k]: float(ranks.iloc[k]) for k in range(len(idxs))}
    for i, r in enumerate(rows):
        r["it_ztc_pct"] = (ztc / 50.0) * rmap[i] if i in rmap else None


def _squad_csvs() -> list[str]:
    """军团成员明细 CSV（动态成员源：legion.json 经 fusion.squad_csv_paths——
    2026-11-22 起军团成员由自动进化引擎月度重排，硬编码清单必然过期）。"""
    from .fusion import squad_csv_paths
    return squad_csv_paths()


def own_pick_features(rows: list[dict], pick_history: dict) -> None:
    """注入 own-pick-history 衰减特征（deep8 #1：消灭重复旧宠）。

    pick_history = {code: list[iso_date]} 最近 N 次被我方甄选上榜的日期。
    own_pick_decay = exp(-days_since_last_pick / 5)——昨天选过=1.0，
    15 天前≈0.05，从未选过=0。own_pick_streak = 连续被选天数。
    """
    for r in rows:
        hist = pick_history.get(r["code"], [])
        if not hist:
            r["own_pick_decay"] = 0.0
            r["own_pick_streak"] = 0
            continue
        last = max(hist)
        try:
            d_days = (dt.date.today() - dt.date.fromisoformat(last)).days
        except (ValueError, TypeError):
            d_days = 99
        r["own_pick_decay"] = round(math.exp(-d_days / 5.0), 4)
        # streak：从最近一天往回数连续日
        streak = 0
        check = dt.date.fromisoformat(last)
        while check.isoformat() in hist:
            streak += 1
            check -= dt.timedelta(days=1)
        r["own_pick_streak"] = streak


def load_pick_history(max_days: int = 30) -> dict:
    """从排名快照读取最近 N 日我方甄选历史 {code: [dates]}。"""
    import csv as _csv
    import glob as _glob
    out: dict[str, list[str]] = {}
    cutoff = (dt.date.today() - dt.timedelta(days=max_days)).isoformat()
    for f in sorted(_glob.glob(os.path.join(C.DATA_DIR, "rankings", "*.csv"))):
        import re
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        d = pd.to_datetime(m.group(1)).strftime("%Y-%m-%d")
        if d < cutoff:
            continue
        try:
            with open(f, encoding="utf-8-sig", newline="") as fh:
                for r in _csv.DictReader(fh):
                    try:
                        if int(float(r.get("rank", 99))) <= C.TOP_N:
                            out.setdefault(r["code"], []).append(d)
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue
    return out


# ---------------------------------------------------------------- 在线续训
def online_finetune(bars_cache: dict, name_map: dict, shares: dict,
                    recent_days: int = 5) -> dict:
    """每日小步续训（deep8 #2：Kaggle Jane Street 金牌方案，相对 +70%）。

    不替换月度模型——在现有 XGBoost booster 上继续训练少量新树（小 lr），
    吸收最近几日的新标签。等价于 warm-start fine-tune。
    """
    import xgboost as xgb
    if not os.path.exists(MODEL_XGB_JSON):
        return {"ok": False, "reason": "无 XGBoost booster 存档"}
    with open(MODEL_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("ok"):
        return {"ok": False, "reason": "基模型不可用"}
    # 只在月度重训之后、下一月度重训之前做日度 fine-tune
    last_trained = meta.get("trained_at", "")
    today = dt.datetime.now().strftime("%Y-%m-%d")
    if last_trained[:10] == today:
        return {"ok": True, "skipped": "今日已重训"}

    # 用最近 N 日数据做增量训练
    stocks = {}
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in _MB:
            continue
        S = _stock_arrays(b)
        if S is not None:
            stocks[str(code).zfill(6)] = S
    _refresh_zcl()
    netbuy = _netbuy_calendar()
    covered = _labels_covered_through()
    all_days = _trade_dates_from_bars(stocks, dt.date.fromisoformat(C.PROFILE_SINCE))
    recent = [d for d in all_days if d >= (dt.date.today() - dt.timedelta(days=recent_days)).isoformat()]
    if not recent:
        return {"ok": False, "reason": "无近期数据"}
    # 构建增量训练集
    Xs, ys = [], []
    n_label_pending = 0
    pick_hist = load_pick_history()
    for d in recent:
        rows = []
        for code, S in stocks.items():
            i = S["idx"].get(d)
            if i is None or i < 6:
                continue
            r = _train_row_at(S, code, i, name_map, shares)
            if r is not None:
                rows.append(r)
        if not rows:
            continue
        d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=20)).isoformat()
        rep = set()
        for dd, codes in netbuy.items():
            if d0 <= dd < d:
                rep |= codes
        # Bug fix（recheck B11）：标签=次根K 日净买（与 train() 的 D+1 口径一致）
        # 此前用当日净买——错位一天，续训在学错误目标
        # Bug fix（recheck A4）：_next_bar 定义在 poolmodel.py，scorer 内
        # 需要内联（此前 NameError 被 except 吞掉→续训永不生效）
        def _nb(S_, d_):
            for _i, _x in enumerate(S_["dates"]):
                if _x > d_:
                    return _i
            return None
        next_d = None
        for S2 in stocks.values():
            nj = _nb(S2, d)
            if nj is not None:
                next_d = S2["dates"][nj]
                break
        # 标签日尚无已发布明细：该日整批样本的标签未知，跳过——此前标签集
        # 取成空集、整日记 0，15:10 首跑时最近一两个训练日全是伪负例
        if next_d is None or not covered or next_d > covered:
            n_label_pending += 1
            continue
        labels = netbuy.get(next_d, set())
        ztc = sum(1.0 for S2 in stocks.values()
                  if (i2 := S2["idx"].get(d)) is not None
                  and np.isfinite(S2["hpct"][i2]) and S2["hpct"][i2] >= _ZT)
        attach_it_ztc_pct(rows, ztc)
        try:
            from .themed import attach_theme_features
            attach_theme_features(rows, dt.date.fromisoformat(d))
        except Exception:
            pass
        own_pick_features(rows, pick_hist)
        Xs.append(logit_matrix(rows, rep, pct_mode=True))
        ys.extend(1.0 if r["code"] in labels else 0.0 for r in rows)
    if not ys:
        return {"ok": False, "reason": "增量数据为空"}
    X = np.vstack(Xs)
    y = np.array(ys)
    if y.sum() < 1:
        return {"ok": False, "reason": "增量正例=0"}
    # 加载现有 booster 并继续训练（2026-11-20 修复：此前 params 构造后从未传入、
    # booster.update(iteration=5) 的 iteration 是轮次编号不是棵数——净效果是
    # 1 棵沿袭旧配置 lr=0.1 的大步树，与设计「5 棵 lr=0.005 小步树」差 20 倍步长。
    # 改为 xgb.train 显式传参 + device 兜底，与 _fit_xgb 双路径一致）
    booster = xgb.Booster()
    booster.load_model(MODEL_XGB_JSON)
    dtrain = xgb.DMatrix(X, label=y)
    ft_params = {"objective": "binary:logistic", "eval_metric": "logloss",
                 "learning_rate": 0.005, "max_depth": 4, "verbosity": 0}
    try:
        booster = xgb.train({**ft_params, "device": "cuda"}, dtrain,
                            num_boost_round=5, xgb_model=booster)
    except Exception:                              # noqa: BLE001 CUDA 不可用退 CPU
        booster = xgb.train({**ft_params, "device": "cpu"}, dtrain,
                            num_boost_round=5, xgb_model=booster)
    # 保存 fine-tuned booster（原子写 + .bak：模型是唯一真源，截断即报废）
    fstore.write_with_bak(MODEL_XGB_JSON, booster.save_model)
    _XGB_BOOSTER_CACHE.pop(MODEL_XGB_JSON, None)   # 同进程后续打分不得再用旧件
    meta["trained_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["finetuned"] = True
    fstore.write_text(MODEL_JSON,
                      json.dumps(meta, ensure_ascii=False, indent=2))
    return {"ok": True, "n_incremental": len(y), "n_pos": int(y.sum()),
            "n_label_pending": n_label_pending}


def squad_recent_codes(before: dt.date, lookback_days: int = 20) -> set:
    """军团四席位近 N 自然日净买过的票（同日共击圈层记忆，生产加分先验用）。

    依据 deep3 特征席：squad_cnt20 系数 6/6 月 +1.4~2.3（最强最稳）但行覆盖仅
    5%，入模型得不偿失——降级为手工加分先验（+0.6，未拟合，台账裁判）。
    """
    out = set()
    d0 = (before - dt.timedelta(days=lookback_days)).isoformat()
    for path in _squad_csvs():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
        except Exception:
            continue
        df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.strftime("%Y-%m-%d")
        g = (df.groupby(["交易日期", "股票代码"], as_index=False)
             .agg(买=("买入金额", "max"), 卖=("卖出金额", "max")))
        g["净"] = g["买"] - g["卖"].fillna(0)
        sel = g[(g["净"] > 0) & (g["交易日期"] >= d0)
                & (g["交易日期"] < before.isoformat())]
        out |= set(sel["股票代码"])
    return out


# ---------------------------------------------------------------- 训练侧
def _stock_arrays(bars: pd.DataFrame) -> dict | None:
    """bars → 训练用数组包（与 empirical.load_stocks 同口径，严格只读缓存）。"""
    df = pd.DataFrame({
        "date": bars["date"].map(lambda d: str(d)[:10]),
        "open": pd.to_numeric(bars.get("open"), errors="coerce"),
        "close": pd.to_numeric(bars["close"], errors="coerce"),
        "high": pd.to_numeric(bars["high"], errors="coerce"),
        "low": pd.to_numeric(bars["low"], errors="coerce"),
        "volume": pd.to_numeric(bars["volume"], errors="coerce"),
        "turnover": pd.to_numeric(bars.get("turnover"), errors="coerce"),
    }).dropna(subset=["close", "volume"]).reset_index(drop=True)
    if len(df) < 7:
        return None
    c = df["close"]
    pct = (c / c.shift(1) - 1.0) * 100.0
    is95 = (pct >= _ZT).fillna(False).astype(int)
    # lianban 特征用生产阈值 9.8（路线图#1 修复点②；股性 zt60/maxlb60 保持 9.5）
    is_lb = (pct >= _LB).fillna(False).astype(int)
    lb95, cnt = [], 0
    for ok in is95:
        cnt = cnt + 1 if ok else 0
        lb95.append(cnt)
    lb95 = pd.Series(lb95)
    lb, cnt = [], 0
    for ok in is_lb:
        cnt = cnt + 1 if ok else 0
        lb.append(cnt)
    lb = pd.Series(lb)
    hi20 = df["high"].rolling(20, min_periods=2).max()
    lo20 = df["low"].rolling(20, min_periods=2).min()
    v5m = df["volume"].shift(1).rolling(5).mean()
    hpct = (df["high"] / c.shift(1) - 1.0).to_numpy(float) * 100.0
    return {
        "dates": df["date"].tolist(),
        "idx": {d: i for i, d in enumerate(df["date"].tolist())},
        "open": df["open"].to_numpy(float),   # 核验日 V 开盘（v2.5-exec 一字判定 / exec 口径）
        "close": c.to_numpy(float), "pct": pct.to_numpy(float),
        "hpct": hpct,
        "volume": df["volume"].to_numpy(float),
        "turnover": df["turnover"].to_numpy(float),
        "lianban": lb.to_numpy(int),
        "zt20": is95.rolling(20, min_periods=1).sum().to_numpy(float),
        "zt60": is95.rolling(60, min_periods=5).sum().to_numpy(float),
        "maxlb60": lb95.rolling(60, min_periods=5).max().to_numpy(float),
        "m5": (c / c.shift(5) - 1.0).to_numpy(float) * 100.0,
        "stir5": pct.rolling(5, min_periods=5).max().to_numpy(float),
        "pos20": ((c - lo20) / (hi20 - lo20)).fillna(0.5).to_numpy(float),
        "vr": (df["volume"] / v5m).to_numpy(float),
    }


def _train_row_at(S: dict, code: str, i: int, name_map: dict, shares: dict):
    name = name_map.get(code, code)
    if "ST" in name.upper() or "退" in name:
        return None
    turn = S["turnover"][i]
    if not (isinstance(turn, float) and math.isnan(turn)):
        turn = float(turn)
    else:
        sh = shares.get(code)
        turn = S["volume"][i] * 100.0 / sh * 100.0 if sh else None
    sh = shares.get(code)
    close = float(S["close"][i])
    pct = float(S["pct"][i])
    m5 = float(S["m5"][i]) if np.isfinite(S["m5"][i]) else None
    lianban = int(S["lianban"][i])
    zt20 = float(S["zt20"][i])
    stir5 = float(S["stir5"][i]) if np.isfinite(S["stir5"][i]) else None
    # 四通道（bars 近似宇宙；互斥链与生产 screen 一致——路线图#1 修复点①；
    # 生产切换通道走归档封板，训练用 bars zt20 近似，差异已知并文档化）
    strong = pct >= C.CANDIDATE_PCT_MIN
    switch = (not strong) and zt20 >= 1
    switch, momentum, stir = _channels(strong, switch, m5, stir5)
    if not (strong or switch or momentum or stir):
        return None
    return {
        "code": code, "name": name, "lianban": lianban, "pct": pct,
        "price": close, "turnover": turn,
        "vr": float(S["vr"][i]) if np.isfinite(S["vr"][i]) else None,
        "pos20": float(S["pos20"][i]),
        "ltsz": sh * close / 1e8 if sh else None, "m5": m5,
        "zt60": int(S["zt60"][i]) if np.isfinite(S["zt60"][i]) else 0,
        "maxlb60": int(S["maxlb60"][i]) if np.isfinite(S["maxlb60"][i]) else 0,
        "via_switch": switch, "via_m5": momentum, "via_stir": stir,
    }


def _trade_dates_from_bars(stocks: dict, since: dt.date) -> list[str]:
    s = since.isoformat()
    days = set()
    for S in stocks.values():
        for d in S["dates"]:
            if d >= s:
                days.add(d)
    return sorted(days)


_ZCL_CSV = os.path.join(C.ROOT, "out", "research", "fusion",
                        "partner_trades", "10025390.csv")


def _refresh_zcl(max_age_hours: float = 20.0) -> None:
    """知春路明细刷新（联合目标的标签源，每日拉全量幂等覆盖；失败用旧档）。"""
    if os.path.exists(_ZCL_CSV) and os.path.getmtime(_ZCL_CSV) > 0:
        age_h = (dt.datetime.now().timestamp() - os.path.getmtime(_ZCL_CSV)) / 3600.0
        if age_h < max_age_hours:
            return
    try:
        import akshare as ak
        import io as _io
        from contextlib import redirect_stderr
        buf = _io.StringIO()
        with redirect_stderr(buf):
            df = ak.stock_lhb_yyb_detail_em(symbol="10025390")
        fstore.write_dataframe_csv(_ZCL_CSV, df)
        print("[v2.2] 知春路明细已刷新（联合标签源）")
    except Exception as e:                      # noqa: BLE001 刷新失败退回旧档
        print(f"[v2.2] 知春路刷新失败（{type(e).__name__}），沿用旧档")


def _netbuy_calendar() -> dict[str, set]:
    """v2.2 联合目标标签：紫阳 ∪ 知春路（同门席位行为同源，deep4 扩靶席验证：
    迁移 recall 0.083>紫阳自身 0.070，联合命中日 16→21 且 p=0.325 不可区分）。"""
    frames = []
    for path in (C.SEAT_TRADES_CSV, _ZCL_CSV):
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.strftime("%Y-%m-%d")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    g = (df.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(买=("买入金额", "max"), 卖=("卖出金额", "max")))
    g["净"] = g["买"] - g["卖"].fillna(0)
    g = g[(g["净"] > 0) & g["股票代码"].str[:3].isin(_MB)]
    out: dict[str, set] = {}
    for d, grp in g.groupby("交易日期"):
        out[d] = set(grp["股票代码"])
    return out


def _labels_covered_through() -> str:
    """标签已发布水位：各席位明细里最晚的交易日期的**最小值**（ISO，无数据=""）。

    龙虎榜按日盘后发布——晚于水位的日期「查不到净买」只说明还没发布，不能
    当负例。取最小值：联合标签=紫阳∪知春路，任一份明细没覆盖到该日，标签
    都不完整；任一份缺失/读坏则返回 ""（一律按未覆盖处理，本次续训跳过）。
    """
    latest = []
    for path in (C.SEAT_TRADES_CSV, _ZCL_CSV):
        try:
            df = pd.read_csv(path, usecols=["交易日期"], encoding="utf-8-sig")
            latest.append(str(pd.to_datetime(df["交易日期"]).max())[:10])
        except Exception:                          # noqa: BLE001 读不到=标签不可信
            return ""
    return min(latest)


def union_repeat_set(before: dt.date, lookback_days: int = 20) -> set:
    """推理侧 repeat01 记忆集（v2.3 路线图#1 修复点③）。

    紫阳∪知春路净买、自然日窗 [D-20, D) 半开——与训练侧 train() 逐字节同口径：
    不含预测日当日（消除 15:10 首跑与晚间重登的特征漂移）、含知春路
    （与 v2.2-union 系数语义匹配）。
    """
    d0 = (before - dt.timedelta(days=lookback_days)).isoformat()
    out = set()
    for dd, codes in _netbuy_calendar().items():
        if d0 <= dd < before.isoformat():
            out |= codes
    return out


# ---------------------------------------------------------------- v3.1 记忆特征
# 一次性记忆参数化（2026-09-28 用户令：加减分必须有事实依据，废除手工后处理 ±）。
# 全部数值为 ≤2026-08-31 拟合并冻结（DEV 窗纪律，生产全窗只吃冻结值）；
# 拟合明细 out/research/ziyang/deep/hazard_lut_raw.json。
_HAZARD_LUT = ((1, 1, 2.512), (2, 3, 1.665), (4, 10, 1.102),
               (11, 20, 0.306), (21, 60, -0.193), (61, None, -1.379))
# 浮亏维度（2026-09-28 深夜补充拟合，用户现场质疑"两个地板还选？"后设立）：
# 再点火率按"距上次买入收盘的浮盈亏"分层（age≤20 有效区，≤08-31 冻结，
# deep/underwater_fit.csv）：浮盈≥+5% 他加仓（1.93x, +0.66）；-5~+5% 中性；
# -5~-20% 抑制（合并桶 k=6/766, -1.31）；≤-20% 深套近乎死绝（1/352, -2.34）。
# 中间两小桶 k=3/3 各自不稳定故合并。age>60 不适用（hazard 已深冻结）。
_UW_LUT = ((0.05, None, 0.66), (-0.05, 0.05, -0.10),
           (-0.20, -0.05, -1.31), (None, -0.20, -2.34))
_MEM_CACHE: dict = {}


def _trade_calendar() -> tuple[list[str], dict[str, int]]:
    if "cal" not in _MEM_CACHE:
        df = pd.read_csv(os.path.join(C.DATA_DIR, "trade_calendar.csv"))
        days = sorted(df.loc[df["is_trade"] == 1, "date"].astype(str))
        _MEM_CACHE["cal"] = (days, {d: i for i, d in enumerate(days)})
    return _MEM_CACHE["cal"]


def _zy_events() -> pd.DataFrame:
    """紫阳本席全事件（买+卖，主板、非零净额）——hazard/卖出档只认本席。"""
    if "zy" not in _MEM_CACHE:
        df = pd.read_csv(C.SEAT_TRADES_CSV, dtype={"股票代码": str},
                         encoding="utf-8-sig")
        df["d"] = pd.to_datetime(df["交易日期"]).dt.strftime("%Y-%m-%d")
        df["code"] = df["股票代码"].str.zfill(6)
        df = df.assign(netamt=df["买入金额"].fillna(0) - df["卖出金额"].fillna(0))
        df = df[df["code"].str[:3].isin(_MB) & (df["netamt"] != 0)]
        _MEM_CACHE["zy"] = df[["code", "d", "netamt"]]
    return _MEM_CACHE["zy"]


def attach_memory_features(rows: list[dict], trade_date: dt.date,
                           closes: dict | None = None) -> None:
    """注入记忆特征（训练/推理同口径，按 D 半开窗、零未来数据）：

    hazard_zy   距上次净买的交易日龄 → 冻结 log-LR（从未买=0 中性）
    squad20     军团四席 [D-20,D) 净买（二值；拟合 +1.10）
    sold_young  上次买入≤SELL_MAX_AGE 交易日内且此后有≥SELL_NET_YI 净卖
                （二值；对照检验：仅年轻持仓档有卖出抑制效应）
    uw_lr       距上次买入的浮盈亏档 log-LR（≤08-31 冻结；closes 提供买入日
                收盘锚，当日价用行内 price——两侧同源）。缺锚=0 中性。
    """
    import bisect as _bs
    days, didx = _trade_calendar()
    d0 = trade_date.isoformat()
    if d0 not in didx:
        return
    i0 = didx[d0]
    ev = _zy_events()
    last_buy, big_sells = {}, {}
    for r in ev.itertuples():
        if r.d >= d0:
            continue
        if r.netamt > 0:
            if r.code not in last_buy or r.d > last_buy[r.code]:
                last_buy[r.code] = r.d
        elif r.netamt < -C.SELL_NET_YI * 1e8:
            big_sells.setdefault(r.code, []).append(r.d)
    sq = _MEM_CACHE.setdefault("squad_by_date", {})
    if d0 not in sq:
        sq[d0] = squad_recent_codes(trade_date)
    squad20 = sq[d0]
    for r in rows:
        c = str(r["code"]).zfill(6)
        r["squad20"] = 1.0 if c in squad20 else 0.0
        r["uw_lr"] = 0.0
        last = last_buy.get(c)
        if last is None:
            r["hazard_zy"] = 0.0
            r["sold_young"] = 0.0
            continue
        age = i0 - didx.get(last, _bs.bisect_left(days, last))
        for lo, hi, lr in _HAZARD_LUT:
            if age >= lo and (hi is None or age <= hi):
                r["hazard_zy"] = lr
                break
        else:
            r["hazard_zy"] = 0.0
        r["sold_young"] = 1.0 if (
            age <= C.SELL_MAX_AGE
            and any(last < s < d0 for s in big_sells.get(c, ()))) else 0.0
        # 浮亏档（age≤60 有效区；锚=上次买入日收盘，当日=行内 price）
        if closes and age <= 60:
            cl = closes.get(c, {}).get(last)
            px = r.get("price")
            if cl and px:
                uw = float(px) / float(cl) - 1.0
                for lo, hi, lr in _UW_LUT:
                    if (lo is None or uw >= lo) and (hi is None or uw < hi):
                        r["uw_lr"] = lr
                        break


_fit_backend = "cpu-sklearn"


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    """GPU 优先的 logistic 训练：PyTorch CUDA（RTX 4060）→ sklearn CPU 兜底。

    返回 (coef, intercept)，与 sklearn LogisticRegression(class_weight=balanced)
    语义对齐（加权交叉熵 + L2 正则）。
    """
    global _fit_backend
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        dev = torch.device("cuda")
        # 显式 float32（numpy 默认 float64 → torch 转换在某些驱动组合会静默失败）
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
        yt = torch.tensor(np.asarray(y, dtype=np.float32), device=dev)
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pos_w = max(n_neg / max(n_pos, 1), 1.0)   # balanced 等价
        w = torch.zeros(X.shape[1], dtype=torch.float32, device=dev, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.01, weight_decay=1e-4)
        pw = torch.tensor(pos_w, dtype=torch.float32, device=dev)
        for _ in range(2000):
            opt.zero_grad()
            logit = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, pos_weight=pw)
            loss.backward()
            opt.step()
        _fit_backend = "gpu-torch"
        print(f"  [GPU] logistic 训练完成（{X.shape[0]} 行 × {X.shape[1]} 维，loss={float(loss):.4f}）")
        return w.detach().cpu().numpy(), float(b.detach().cpu().item())
    except Exception:                              # noqa: BLE001 GPU 不可用退 CPU
        _fit_backend = "cpu-sklearn"
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        return clf.coef_[0], float(clf.intercept_[0])


def _fit_xgb(X: np.ndarray, y: np.ndarray):
    """v2.7 主模型：XGBoost（GPU CUDA 优先，CPU 兜底；deep7 Round3 冻结超参
    =config.XGB_PARAMS d4/n200/lr0.1，scale_pos_weight=neg/pos）。

    返回 (booster, device)。走样前验证：全窗 EXEC +0.2376 / RECALL 0.0617 /
    WIN 53.7，三门过（deep7/round3_results.json）。
    """
    import xgboost as xgb
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    p = dict(C.XGB_PARAMS)
    n_round = int(p.pop("n_estimators", 200))
    base = {"objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "scale_pos_weight": spw, "verbosity": 0}
    base.update(p)
    try:
        booster = xgb.train({**base, "device": "cuda"},
                            xgb.DMatrix(X, label=y), num_boost_round=n_round)
        return booster, "cuda"
    except Exception:                              # noqa: BLE001 CUDA 不可用退 CPU
        booster = xgb.train({**base, "device": "cpu"},
                            xgb.DMatrix(X, label=y), num_boost_round=n_round)
        return booster, "cpu"


_XGB_BOOSTER_CACHE: dict = {}


def _prev_xgb_path() -> str:
    """上一代 booster 的归档路径：`model_xgb.prev.json`——**保留 .json 扩展名**。

    xgboost 按扩展名选加载格式，`model_xgb.json.prev` 会报 Unknown file format
    再猜 JSON（每次重训首日对照都刷一条 WARNING）。读取侧仍兼容旧命名。
    """
    return os.path.splitext(MODEL_XGB_JSON)[0] + ".prev.json"


def _load_xgb_booster(path: str):
    """booster 进程内缓存加载（score_rows 每日多次调用免重复 IO）。"""
    if path not in _XGB_BOOSTER_CACHE:
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(path)
        _XGB_BOOSTER_CACHE[path] = b
    return _XGB_BOOSTER_CACHE[path]


def train(cutoff: dt.date, bars_cache: dict, name_map: dict, shares: dict,
          min_pos: int = 5, persist: bool = True) -> dict:
    """严格用 cutoff 前数据（含标签日）训练月度模型；persist=False 供走样前
    验证逐月滚动调用而不落盘（路线图#1 验证需要）。

    标签（v2.5-exec，改动二）：日 D 宇宙中，code ∈ 席位∪知春路 核验日 V 净买
    集合 ∧ V 开盘非一字（须 V < cutoff）；V=个股自身次根K（与台账 exec_oo 锚
    一致），V 开一字行正负例同删（与 EXEC 主指标「一字槽两侧同剔」对称）。
    """
    _refresh_zcl()
    since = dt.date.fromisoformat(C.PROFILE_SINCE)
    stocks = {}
    for code, b in bars_cache.items():
        if b is None or not len(b) or str(code)[:3] not in _MB:
            continue
        S = _stock_arrays(b)
        if S is not None:
            stocks[str(code).zfill(6)] = S
    days = _trade_dates_from_bars(stocks, since)
    cutoff_s = cutoff.isoformat()
    netbuy = _netbuy_calendar()

    Xs, ys = [], []
    n_yizi = 0
    for d in days:
        if d >= cutoff_s:                      # 训练日严格 < cutoff（scorer 既有语义）
            continue
        rows = []
        for code, S in stocks.items():
            i = S["idx"].get(d)
            if i is None or i < 6:
                continue
            r = _train_row_at(S, code, i, name_map, shares)
            if r is None:
                continue
            r["_y"] = None                     # 无核验日（停牌/越界）→ 不可标注，剔行
            r["_yizi"] = False
            if i + 1 < len(S["dates"]):
                vd = S["dates"][i + 1]         # 核验日 V（个股自身次根K，与 exec_oo 锚一致）
                if vd < cutoff_s:              # 标签日严格 < cutoff（零泄漏）
                    r["_y"] = 1.0 if code in netbuy.get(vd, set()) else 0.0
                    # 一字剔除对称化：V 开盘 ≥ 昨收×1.10（−1厘容差）→ 正负例同删，
                    # 与 EXEC 主指标「一字槽两侧同剔」同口径；不设 exec_oo>0 子句
                    # （真席位事件 ~4.9% 为负 exec，翻向=标签噪声）
                    r["_yizi"] = float(S["open"][i + 1]) >= round(
                        float(S["close"][i]) * 1.10, 2) - 0.001
            rows.append(r)
        if not rows:
            continue
        d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=20)).isoformat()
        rep = set()
        for dd, codes in netbuy.items():
            if d0 <= dd < d:
                rep |= codes
        ztc = 0.0
        for S in stocks.values():
            i = S["idx"].get(d)
            if i is not None and np.isfinite(S["hpct"][i]) and S["hpct"][i] >= _ZT:
                ztc += 1.0
        attach_it_ztc_pct(rows, ztc)
        # 题材特征注入（Bug2 修复：此前训练从不注入 → 模型系数恒 0，17 维实为 15）
        try:
            from .themed import attach_theme_features
            _d_date = dt.date.fromisoformat(d)
            attach_theme_features(rows, _d_date)
        except Exception:                        # noqa: BLE001 特征表缺失/损坏不阻断
            for r in rows:
                r.setdefault("theme_max_b", 0.0)
                r.setdefault("theme30_b", 0.0)
        # v2.6 分时轨迹特征注入（deep7 Round1）：tp_touch/tp_am，池文件缺失
        # → 行缺键 → logit_matrix 中位数填充=中性（安全回滚，与 theme 同模式）
        try:
            from .themed import attach_tp_features
            attach_tp_features(rows, dt.date.fromisoformat(d))
        except Exception:                        # noqa: BLE001 池文件缺失不阻断
            pass
        X = logit_matrix(rows, rep, pct_mode=True)
        # 分位变换在全宇宙上算（与推理一致：D 晚不知 V 开盘，无法按一字筛宇宙），
        # 一字掩码只决定哪些行进损失
        keep = [k for k, r in enumerate(rows) if r["_y"] is not None and not r["_yizi"]]
        if not keep:
            continue
        n_yizi += sum(1 for r in rows if r["_yizi"])
        Xs.append(X[keep])
        ys.extend(rows[k]["_y"] for k in keep)
    X = np.vstack(Xs) if Xs else np.zeros((0, len(LOGIT_FEATS)))
    y = np.array(ys)
    meta = {"version": MODEL_VERSION, "cutoff": cutoff_s, "n_train_days": len(Xs),
            "n_pos": int(y.sum()), "n_rows": int(len(y)), "n_yizi_dropped": n_yizi,
            "features": LOGIT_FEATS, "trained_at":
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if len(y) == 0 or y.sum() < min_pos:
        meta["ok"] = False
        return meta
    if getattr(C, "MODEL_KIND", "logistic") == "xgb":
        # v2.7（deep7 Round3）：XGBoost 主模型。booster 落 model_xgb.json，
        # model.json 只存元数据（kind=xgb 标记，coef 留空）。
        booster, device = _fit_xgb(X, y)
        meta.update({"ok": True, "kind": "xgb", "coef": [], "intercept": 0.0,
                     "backend": f"xgb-{device}",
                     "xgb_params": {k: v for k, v in C.XGB_PARAMS.items()}})
        # v3.1 校准层：训练集预测上拟合 isotonic，断点落 iso.json（推理侧
        # np.interp 复原，免 sklearn 依赖）。合成= logit(iso(P)) + 冻结 log-LR。
        try:
            import xgboost as _xgb
            from sklearn.isotonic import IsotonicRegression as _IR
            Ptr = booster.predict(_xgb.DMatrix(X))
            ir = _IR(y_min=1e-6, y_max=1 - 1e-6).fit(Ptr, y)
            iso = {"x": [round(float(v), 6) for v in ir.X_thresholds_],
                   "y": [round(float(v), 8) for v in ir.y_thresholds_],
                   "fitted_at": meta["trained_at"]}
        except Exception as _e:                   # noqa: BLE001 校准失败→裸 logit 桥
            iso = None
            meta["iso_error"] = type(_e).__name__
        meta["iso"] = bool(iso)
    else:
        coef, intercept = _fit_logistic(X, y)
        meta.update({"ok": True,
                     "coef": [round(float(v), 4) for v in coef],
                     "intercept": round(float(intercept), 4),
                     "backend": _fit_backend})
    if persist:
        os.makedirs(MODEL_DIR, exist_ok=True)
        # 旧口径存档（改动四④）：月度重训首日新旧 top3 diff 的对照物；
        # 对照动作在 screen.predict → retrain_top3_diff（按 trained_at 幂等）
        if os.path.exists(MODEL_JSON):
            try:
                with open(MODEL_JSON, encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("ok"):
                    fstore.write_text(MODEL_PREV_JSON,
                                      json.dumps(prev, ensure_ascii=False, indent=2))
                    if prev.get("kind") == "xgb" and os.path.exists(MODEL_XGB_JSON):
                        os.replace(MODEL_XGB_JSON, _prev_xgb_path())
                    meta["prev_version"] = prev.get("version", "?")
            except Exception:                  # noqa: BLE001 存档失败不阻断训练
                pass
        if meta.get("kind") == "xgb":
            fstore.write_with_bak(MODEL_XGB_JSON, booster.save_model)
            _XGB_BOOSTER_CACHE.pop(MODEL_XGB_JSON, None)   # 重训后旧件出缓存
            if iso is not None:
                fstore.write_text(ISO_JSON, json.dumps(iso))
        fstore.write_text(MODEL_JSON,
                          json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def _booster_loadable(path: str) -> bool:
    """模型工件可否被 xgboost 加载（损坏自愈探测）。"""
    try:
        import xgboost as xgb
        xgb.Booster().load_model(path)
        return True
    except Exception:                              # noqa: BLE001 坏件=不可加载
        return False


def ensure_model(bars_cache: dict, name_map: dict, shares: dict) -> dict:
    """月度策略：模型缺失或 cutoff 落后于本月 → 重训。返回模型元数据。"""
    month_start = dt.date.today().replace(day=1)
    meta = None
    if os.path.exists(MODEL_JSON):
        try:
            with open(MODEL_JSON, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None
    if (meta and meta.get("ok")
            and meta.get("cutoff") >= month_start.isoformat()
            and meta.get("features") == LOGIT_FEATS
            and meta.get("version") == MODEL_VERSION):  # 特征/标签口径变更均重训
        if meta.get("kind") != "xgb" or _booster_loadable(MODEL_XGB_JSON):
            return meta
        # xgb 工件损坏（如 2026-11-22 扩展名误判致 save_model 落成 UBJSON）：
        # 隔离坏件并强制重训——否则每次 daily_sync 在 score 阶段崩
        print(f"[{MODEL_VERSION}] 模型工件损坏，隔离后重训…")
        try:
            os.replace(MODEL_XGB_JSON, MODEL_XGB_JSON + ".corrupt")
        except OSError:
            pass
    print(f"[{MODEL_VERSION}] 训练月度模型（cutoff={month_start}，严格无未来数据）…")
    fresh = train(month_start, bars_cache, name_map, shares)
    if fresh.get("ok"):
        return fresh
    # 回退护栏（路线图#1）：只接受同版本/同特征集的旧模型——跨口径回退会让
    # 生产分数与系数语义脱节（此前静默回退到任意旧模型）
    if (meta and meta.get("ok")
            and meta.get("features") == LOGIT_FEATS
            and meta.get("version") == MODEL_VERSION):
        print(f"[{MODEL_VERSION}] 重训失败（{fresh.get('n_pos')} 正例），"
              f"沿用本月内旧模型（同口径）")
        return meta
    raise RuntimeError(
        f"模型不可用：重训失败（{fresh.get('n_pos')} 正例）且无同版本"
        f"（{MODEL_VERSION}）旧模型——拒绝跨口径静默回退")


def score_rows(rows: list[dict], repeat_set: set,
               model_path: str | None = None) -> tuple[list[float], list[list]]:
    """推理：日内分位变换 + 模型打分（v2.7：kind=xgb 走 booster，否则线性 logit）。

    返回 (scores, contribs)。contribs=每行 top-3 特征贡献（线性=系数×日内分位；
    xgb=pred_contribs 近似 SHAP），供报告「主贡献」行解释分数来源；只依赖已落盘
    模型。model_path 默认现行 model.json；传 model_prev.json 供重训首日对照。
    """
    with open(model_path or MODEL_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("ok"):
        raise RuntimeError("模型不可用")
    X = logit_matrix(rows, repeat_set, pct_mode=True)
    if meta.get("kind") == "xgb":
        import xgboost as xgb
        bpath = MODEL_XGB_JSON
        if model_path == MODEL_PREV_JSON:
            # 重训首日对照用旧 booster；兼容旧命名 `.json.prev` 的存量归档
            bpath = next((p for p in (_prev_xgb_path(), MODEL_XGB_JSON + ".prev")
                          if os.path.exists(p)), MODEL_XGB_JSON)
        if not os.path.exists(bpath):
            raise RuntimeError("xgb booster 缺失（model_xgb.json）")
        booster = _load_xgb_booster(bpath)
        dm = xgb.DMatrix(X)
        scores = booster.predict(dm).tolist()
        try:
            sh = booster.predict(dm, pred_contribs=True)[:, :-1]   # 末列=bias
            contribs = [[(f, round(float(v), 2)) for f, v in
                         sorted(zip(LOGIT_FEATS, row), key=lambda t: -abs(t[1]))[:3]]
                       for row in sh]
        except Exception:                          # noqa: BLE001 SHAP 失败不阻断
            contribs = [[] for _ in rows]
        # ---- v3.1 校准合成（walk-forward 2026-09-28 三门过：EXEC +0.2642 /
        # RECALL 0.0801(+30%) / WIN 54.5，不退步于 Round5 保留最优）----
        # P → iso → logit = 真对数几率；再加冻结 log-LR 记忆参数
        # （hazard LUT + squad 1.10 - sold_young 0.48，全部 ≤08-31 拟合）。
        # iso 断点缺失（旧模型）→ 裸 logit(P) 桥（同为单调真几率，合成语义不变）。
        P = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        try:
            with open(ISO_JSON, encoding="utf-8") as f:
                _iso = json.load(f)
            Pi = np.clip(np.interp(P, _iso["x"], _iso["y"]), 1e-6, 1 - 1e-6)
        except Exception:                          # noqa: BLE001
            Pi = P
        # iso 在训练 P 分布之外是平的（首断点以下全体钳到同一 y）——低概率区
        # 会大面积并列、名次退化为代码序。加 ε·raw-logit 破平局：严格单调、
        # 量级 ≤0.01（校准语义不变），并列块内部恢复模型原始序。
        L = (np.log(Pi / (1 - Pi))
             + 1e-3 * np.log(P / (1 - P)))
        composed, mem_terms = [], []
        for r, l in zip(rows, L):
            # DEV 月裁决（2026-09-28，deep7 选择集 {no-uw/full/suppress/tail}）：
            # suppress-only（只吃压制项、剔除浮盈+0.66 加分）全优——
            # DEV EXEC +0.4607/WIN 59.6%（full 因 +0.66 反而 -3.9bp 被否）。
            # 全窗冻结值：EXEC +0.3368 / WIN 57.9 / RECALL 0.0725（保留门 +7.3bp）。
            uw_eff = min(float(r.get("uw_lr") or 0.0), 0.0)
            terms = [("hazard_zy", float(r.get("hazard_zy") or 0.0)),
                     ("squad20", 1.10 * float(r.get("squad20") or 0.0)),
                     ("sold_young", -0.48 * float(r.get("sold_young") or 0.0)),
                     ("uw", uw_eff)]
            terms = [(k, round(v, 2)) for k, v in terms if v]
            composed.append(round(float(l) + sum(v for _, v in terms), 4))
            mem_terms.append(terms)
        # 贡献行并入合成项（读者须能看到"为什么是这只"）
        contribs = [c + m for c, m in zip(contribs, mem_terms)]
        return composed, contribs
    if len(meta.get("coef", [])) != len(LOGIT_FEATS):
        raise ValueError(f"模型特征数 {len(meta.get('coef', []))} 与现行特征集 "
                         f"{len(LOGIT_FEATS)} 不一致（跨口径对照拒绝）")
    w = np.array(meta["coef"])
    scores = (X @ w + meta["intercept"]).tolist()
    contribs = []
    for xi in X:
        pairs = sorted(zip(LOGIT_FEATS, (w * xi).tolist()),
                       key=lambda t: -abs(t[1]))[:3]
        contribs.append([(f, round(v, 2)) for f, v in pairs])
    return scores, contribs


def retrain_top3_diff(trade_date: dt.date, ranked: list[dict],
                      repeat_set: set) -> dict | None:
    """月度重训首日：新旧模型 top3 对照（改动四④，度量/呈现层）。

    触发=model.json 由重训产生且存档了旧口径（meta.prev_version）；对照只在
    重训后的**首个预判日**呈现一次（retrain_diff.json 记 shown_for=首次呈现的
    预判日），同日重跑复用已落盘 diff，此后不再呈现。
    """
    try:
        with open(MODEL_JSON, encoding="utf-8") as f:
            meta = json.load(f)
        if not (meta.get("ok") and meta.get("prev_version")
                and os.path.exists(MODEL_PREV_JSON)):
            return None
        diff_path = os.path.join(MODEL_DIR, "retrain_diff.json")
        stored = None
        if os.path.exists(diff_path):
            try:
                with open(diff_path, encoding="utf-8") as f:
                    stored = json.load(f)
            except Exception:                  # noqa: BLE001
                stored = None
        if stored and stored.get("trained_at") == meta.get("trained_at"):
            return None if stored.get("shown_for") else stored.get("diff")
        prev_scores, _ = score_rows(ranked, repeat_set, model_path=MODEL_PREV_JSON)
        new_top3 = [r["code"] for r in
                    sorted(ranked, key=lambda x: int(x.get("rank") or 99))[:C.TOP_N]]
        prev_top3 = [ranked[k]["code"] for k in sorted(
            range(len(ranked)),
            key=lambda k: (-prev_scores[k], ranked[k]["code"]))[:C.TOP_N]]
        diff = {"prev_version": meta.get("prev_version"),
                "new_version": meta.get("version"),
                "trained_at": meta.get("trained_at"),
                "prev_top3": prev_top3, "new_top3": new_top3,
                "overlap_n": len(set(prev_top3) & set(new_top3))}
        os.makedirs(MODEL_DIR, exist_ok=True)
        fstore.write_text(diff_path,
                          json.dumps({"trained_at": meta.get("trained_at"),
                                      "shown_for": trade_date.isoformat(),
                                      "diff": diff},
                                     ensure_ascii=False, indent=2))
        return diff
    except Exception:                          # noqa: BLE001 对照失败不影响主流程
        return None
