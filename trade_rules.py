# -*- coding: utf-8 -*-
"""排板成交模型与 T+1 卖出规则回测（执行闭环核心，纯函数可单测）。

现实成交模型（对 2 板信号的次日执行）：
- T+1 开盘一字（open≈涨停价）→ 排队靠后**不成交**（最强票错过，符合排板现实）
- T+1 非一字开盘但盘中触及涨停 → 以涨停价**成交**（回封时排到队）
- T+1 全天未触涨停 → 不成交（R3 放弃，不追高）

卖出规则集（≤3 条，A 股 T+1 真实约束：买入当日不可卖）：
- R1 成交且 T+1 收盘封死 → 持有；T+2 开盘卖（若 T+2 再封板 → T+3 开盘卖，上限）
- R2 成交但 T+1 炸板收盘（收盘<涨停价）→ **当日无法卖，T+2 开盘卖**（回测 -7.88%，
  胜率 7%，深亏<-8% 占 48% —— R2 的风控在买入前筛选，不在卖出）
- R3 未成交 → 放弃

输入为 research_scan 的 klines 缓存（date/open/close/high/low/volume，前复权）。
"""
from __future__ import annotations

import pandas as pd

from config import LIMIT_NORMAL

LIMIT_EPS = 0.005      # 涨停价识别容差（前复权近似）
MAX_HOLD_DAYS = 3      # R1 连续封板最多持有到 T+3 开盘


def round_cents(cents: int, mult_pct: int) -> int:
    """交易所式「四舍五入到分」的**唯一实现**（整数运算，无浮点噪声）。

    cents × mult_pct% 取整到分：mult_pct=110 涨停价、90 跌停价（ST 为 105/95）。
    候选池的向量化路径（candidates.limit_prices_vec）与这里的标量路径用同一
    个式子——此前向量化那一行是无容差的 `floor(v*100+0.5)`，在 190 个价位上
    比交易所口径低一分（2026-11-24 价格网格实测），跌停价偏低会让真正触及
    跌停的票不被识别。
    """
    return (cents * mult_pct + 50) // 100


def limit_price(prev_close: float, ratio: float = LIMIT_NORMAL) -> float:
    """主板涨停价（默认 ±10%；ST 传 config.LIMIT_ST）。交易所四舍五入到分。

    用**整数分值**运算精确实现「P×1.1 四舍五入」——浮点直算在 .5 边界有两个
    方向的坑：旧 round(v*100+0.5)/100 双重舍入会让小数∈(0,0.5) 的价位**高估
    一分**（4.62→5.09，应为 5.08）；无容差的 floor 会让小数恰为 .5 的价位
    **低估一分**（17.15→18.86，应为 18.87）。
    test_limit_price_matches_exchange 在价位网格上强制标量/向量化两路径恒等。
    """
    return round_cents(round(prev_close * 100), round((1.0 + ratio) * 100)) / 100.0


def limit_down_price(prev_close: float, ratio: float = LIMIT_NORMAL) -> float:
    """主板跌停价（口径同 limit_price）。"""
    return round_cents(round(prev_close * 100), round((1.0 - ratio) * 100)) / 100.0


def simulate_fill(bars: pd.DataFrame, i_event: int) -> dict | None:
    """T+1 排板成交模拟。bars 为单股K线，i_event 为二板日下标。

    返回 None=无次日数据；否则 dict(fill=是否成交, fill_price, one_word_open,
    sealed_close, t1_open, t1_close)。
    """
    if i_event + 1 >= len(bars):
        return None
    o = bars["open"].astype(float).to_numpy()
    c = bars["close"].astype(float).to_numpy()
    limit = limit_price(c[i_event])
    t1 = i_event + 1
    one_word = o[t1] >= limit - limit * LIMIT_EPS
    touched = o[t1] >= limit - limit * LIMIT_EPS or bars["high"].astype(float).to_numpy()[t1] >= limit - limit * LIMIT_EPS
    sealed = c[t1] >= limit - limit * LIMIT_EPS
    fill = touched and not one_word     # 一字开盘排队靠后不成交；触板回封则成交
    return {
        "fill": bool(fill), "fill_price": limit,
        "one_word_open": bool(one_word), "sealed_close": bool(sealed),
        "t1_open": float(o[t1]), "t1_close": float(c[t1]),
    }


def simulate_exit(bars: pd.DataFrame, i_event: int, fill: dict) -> dict:
    """卖出规则模拟。返回 dict(exit_date, exit_price, rule, ret)。"""
    c = bars["close"].astype(float).to_numpy()
    o = bars["open"].astype(float).to_numpy()
    t1 = i_event + 1
    if not fill["fill"]:
        return {"rule": "R3未成交", "ret": 0.0, "exit_idx": None}
    if not fill["sealed_close"]:
        # R2：T+1 炸板 → 当日不可卖（A 股 T+1），T+2 开盘卖
        k = min(t1 + 1, len(c) - 1)
        ret = o[k] / fill["fill_price"] - 1.0
        return {"rule": "R2炸板T+2卖", "ret": ret, "exit_idx": k}
    # R1：封死持有，次日起开盘卖；再封板则顺延（上限 T+3）
    k = t1 + 1
    while k < len(bars) and k <= i_event + MAX_HOLD_DAYS:
        limit_k = limit_price(c[k - 1])
        if c[k] >= limit_k - limit_k * LIMIT_EPS and k < i_event + MAX_HOLD_DAYS:
            k += 1          # 再封板，继续持有
            continue
        break
    if k >= len(bars):
        k = len(bars) - 1  # 数据边界兜底
    ret = o[k] / fill["fill_price"] - 1.0
    return {"rule": "R1封板持有", "ret": ret, "exit_idx": k}
