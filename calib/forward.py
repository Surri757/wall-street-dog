# -*- coding: utf-8 -*-
"""信号装载与前向数据获取（calibrate 的数据获取关注点）。

两套口径并存，**可成交口径为主**：
- 信息口径 next_open_pct/next_close_pct/max3_pct/promoted：基准是信号日收盘
  ——而信号日是已封死涨停的一天，收盘价买不到，只回答「会不会继续封板」。
- 可成交口径 one_word/fillable/tradable_ret/net_ret（主口径 MAIN_METRIC）：
  基准是次日开盘，一字开盘排队买不到（实测过线池占 36.6%）必须排除，
  net_ret 再扣双边成本。基准是 n1 open 而非信号日 base，天然免疫复权口径问题。
"""
from __future__ import annotations

import datetime as dt
import glob
import os

import pandas as pd

import data_source
import trade_rules
from config import (CALIB_MAIN_METRIC, CALIB_MIN_INFER_DAYS, COST_ROUND_TRIP,
                    PROMOTE_APPROX)

# 主口径：可成交（次日开盘买→次日收盘卖，扣双边成本）。
# 所有「收益」判读默认用它；next_close_pct/promoted 只作龙头识别的信息口径保留。
MAIN_METRIC = CALIB_MAIN_METRIC

# 给出任何显著性结论所需的最少**配对交易日数**（不是笔数）。
# 本项目全样本 23 个独立交易日、OOS 仅 8 个、台账仅 4 个；低于此门槛一律输出
# 「无法推断」，不给看似有效的 t。旧实现的 LEDGER_MIN_N=8 数的是笔数——
# 30 笔挤在同一天也算 30，等于没有门槛。
MIN_INFER_DAYS = CALIB_MIN_INFER_DAYS

# fetch_forward 产出的指标键。缓存装载与 CSV 读写一律以此为准——此前缓存装载
# 硬编码 4 个键，新增指标后旧行缺列会被 except 静默吞掉整行，而 cli 尾部是
# to_csv **全量覆写**，重拉一旦失败该行永久丢失。
FWD_KEYS = ("next_open_pct", "next_close_pct", "max3_pct", "promoted",
            "one_word", "fillable", "tradable_ret", "net_ret")


def load_signals(out_dir: str) -> pd.DataFrame:
    """合并全部归档信号（含未过线股票，供门槛与基线校准；剔除拉数失败的跳过行）。"""
    files = sorted(glob.glob(os.path.join(out_dir, "archive", "signals_*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype={"code": str})
        except Exception as e:
            print(f"  [跳过] 读取失败 {f}: {e}")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "board_score" not in df.columns:
        return pd.DataFrame()
    # score=0 的候选 =「未参与判定」的非 2 板股，不是校准样本
    # （N 字模式已于 2026-09-16 全量移除，历史 nshape_pass 归档行不再入选）
    keep = df["board_score"].notna() & (df["board_score"] > 0)
    df = df[keep]
    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].fillna("") == ""]
    return df


def fetch_forward(code: str, signal_date: dt.date, base_close: float,
                  cache: dict) -> tuple[str, dict | None]:
    """信号日前向指标。两套口径并存，**可成交口径为主**。

    bars 是前复权序列（东财 fqt=1 / 腾讯 qfq），而调用方传入的 base_close 是归档
    CSV 里的**原始**收盘——两者混用会让跨除权日的收益系统性偏负、promoted 误判，
    故不可成交口径的基准取 bars 内信号日自身收盘，base_close 只作有效性哨兵。
    返回 (status, metrics)：status ∈ ok / nodata（信号日后暂无交易日）/ fail（行情拉取失败）。
    仅 ok 写入 cache（失败不缓存，下次运行自动重试）。
    """
    key = (code, signal_date)
    if key in cache:
        return "ok", cache[key]
    bars = data_source.fetch_close_range(
        code, signal_date, signal_date + dt.timedelta(days=15))
    if bars is None or base_close <= 0:
        return "fail", None
    sig = bars[bars["date"] == signal_date]
    if sig.empty:
        # 信号日不在返回区间内：停牌/退市/源口径漂移。不回退到原始收盘，
        # 否则又把两种复权口径混在一起；判 fail 不缓存，下次自动重试。
        return "fail", None
    base = float(sig.iloc[0]["close"])
    if base <= 0:
        return "fail", None
    fwd = bars[bars["date"] > signal_date]
    if fwd.empty:
        return "nodata", None
    n1 = fwd.iloc[0]
    n3 = fwd.iloc[:3]
    n1_open = float(n1["open"])
    n1_close = float(n1["close"])
    # 一字判定复用 trade_rules.limit_price（涨停价取整规则只此一份，不得复制）。
    # base 是前复权价，limit 与 n1_open 同处前复权空间，比较一致；
    # LIMIT_EPS 相对容差吸收分位取整与前复权的微小偏差。
    limit = trade_rules.limit_price(base)
    one_word = n1_open >= limit - limit * trade_rules.LIMIT_EPS
    tradable = (n1_close / n1_open - 1.0) * 100.0 if n1_open > 0 else 0.0
    m = {
        "next_open_pct": (n1_open / base - 1.0) * 100.0,
        "next_close_pct": (n1_close / base - 1.0) * 100.0,
        "max3_pct": (float(n3["high"].max()) / base - 1.0) * 100.0,
        "promoted": int(n1_close / base - 1.0 >= PROMOTE_APPROX),
        "one_word": int(one_word),
        "fillable": int(not one_word),
        "tradable_ret": tradable,
        # 无条件计算，语义＝「若在次日开盘成交、收盘卖出、扣成本」。
        # 不置 0：置 0 会与「成交且打平」混淆；不可成交样本由消费方按 fillable 过滤。
        "net_ret": tradable - COST_ROUND_TRIP * 100.0,
    }
    cache[key] = m
    return "ok", m


def load_forward_cache(fwd_df) -> tuple[dict, int]:
    """_forward.csv → (cache, stale 行数)。

    按 FWD_KEYS 逐键装载，**缺任一键（含 NaN）即不进 cache** → 调用方会重拉该行。
    这是本轮最高风险的数据丢失路径：旧行若被静默丢弃，而 cli 尾部又是
    to_csv **全量覆写**，重拉一旦失败该历史样本就永久消失。stale 数必须打印出来。
    """
    cache: dict = {}
    stale = 0
    int_keys = {"promoted", "one_word", "fillable"}
    if fwd_df is None or fwd_df.empty:
        return cache, 0
    for _, r in fwd_df.iterrows():
        try:
            key = (r["code"], dt.datetime.strptime(str(r["date"]), "%Y-%m-%d").date())
            m = {}
            for k in FWD_KEYS:
                v = r.get(k)
                if v is None or v != v:          # 缺列 / NaN
                    raise KeyError(k)
                m[k] = int(float(v)) if k in int_keys else float(v)
            cache[key] = m
        except Exception:
            stale += 1
    return cache, stale
