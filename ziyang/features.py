# -*- coding: utf-8 -*-
"""特征分桶：画像挖掘（前夜 D-1）与预判打分（当日 T）共用同一套定义。

纪律（先标签后计分）：分桶一旦进入画像并开始登记台账预判，不得为改善
预判成绩而事后改桶——要改就另立版本并重新起账。
"""
from __future__ import annotations

import math

# 桶边界（百分数/倍数口径，见 config）
PCT_EDGES = (-math.inf, -5, 0, 3, 6, 9.5, math.inf)
PCT_LABELS = ("<-5", "-5~0", "0~3", "3~6", "6~9.5", "涨停")
TURN_EDGES = (0, 5, 10, 20, 30, math.inf)
TURN_LABELS = ("<5", "5~10", "10~20", "20~30", ">30")
VR_EDGES = (0, 0.6, 0.8, 1.0, 1.5, math.inf)
VR_LABELS = ("<0.6", "0.6~0.8", "0.8~1.0", "1.0~1.5", ">1.5")
PRICE_EDGES = (0, 5, 10, 20, 50, math.inf)
PRICE_LABELS = ("<5", "5~10", "10~20", "20~50", ">50")
POS_EDGES = (-0.01, 0.3, 0.6, 0.85, 1.01)
POS_LABELS = ("低位<0.3", "0.3~0.6", "0.6~0.85", "高位>0.85")
# 连板桶语义：0板=前夜未涨停收盘；1板=前夜恰 1 连板；2板；3板+=≥3连板。
# 边界写法与其余桶一致（左闭右开），raw 连板数为整数：0→[ -1,1 )，1→[1,2)，2→[2,3)，≥3→[3,∞)。
LB_EDGES = (-1, 1, 2, 3, math.inf)
LB_LABELS = ("0板", "1板", "2板", "3板+")
LTSZ_EDGES = (0, 30, 50, 100, 200, math.inf)
LTSZ_LABELS = ("<30亿", "30~50亿", "50~100亿", "100~200亿", ">200亿")


def _bucket(v, edges, labels):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return labels[i]
    return labels[-1] if v >= edges[-1] else None


def lb_bucket(lianban):      return _bucket(lianban, LB_EDGES, LB_LABELS)
def pct_bucket(pct):         return _bucket(pct, PCT_EDGES, PCT_LABELS)
def turn_bucket(turnover):   return _bucket(turnover, TURN_EDGES, TURN_LABELS)
def vr_bucket(vr):           return _bucket(vr, VR_EDGES, VR_LABELS)
def price_bucket(price):     return _bucket(price, PRICE_EDGES, PRICE_LABELS)
def pos_bucket(pos):         return _bucket(pos, POS_EDGES, POS_LABELS)
def ltsz_bucket(ltsz):       return _bucket(ltsz, LTSZ_EDGES, LTSZ_LABELS)


def eve_features_of_row(row: dict) -> dict:
    """统一入口：一行特征 → 分桶字典（挖掘与打分共用，保证同名同义）。

    ltsz=流通市值（亿），2026-09-20 五路深挖后加入（v1.3）：画像实测 76% 的
    点火在百亿以上、0% 在 30 亿以下，头两轮 0/6 miss 的病根即缺此维度。
    """
    return {
        "lianban": lb_bucket(row.get("lianban")),
        "pct": pct_bucket(row.get("pct")),
        "turnover": turn_bucket(row.get("turnover")),
        "vr": vr_bucket(row.get("vr")),
        "price": price_bucket(row.get("price")),
        "pos20": pos_bucket(row.get("pos20")),
        "ltsz": ltsz_bucket(row.get("ltsz")),
    }
