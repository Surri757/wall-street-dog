# -*- coding: utf-8 -*-
"""数据源门面（facade）：实现拆分在 datasources/ 包内，对外接口保持不变。

模块一览：
  datasources/http.py     共享会话 / GET+JSON 重试 / 端点常量
  datasources/spot.py     全市场快照（东财→新浪）与缓存有效性
  datasources/kline.py    个股日K三源降级 + 区间K线
  datasources/pools.py    涨停/炸板池解析与昨日池重建
  datasources/concepts.py 概念板块快照 / 个股概念 / 板块日K
  datasources/auction.py  腾讯分时（近 5 个交易日）：竞价点 / 当日总量 / 执行日路径
  datasources/calendar.py 交易日历与复盘日期解析
"""
from __future__ import annotations

from datasources.auction import RETENTION_DAYS as AUCTION_RETENTION_DAYS
from datasources.auction import fetch_minute_days
from datasources.http import (EM_HEADERS, EM_KLINE_URL, EM_POOL_BASE,
                              EM_SPOT_HOSTS, EM_SPOT_PATH, TX_KLINE_URL)
from datasources.spot import cache_is_valid, fetch_spot
from datasources.kline import fetch_daily, fetch_turnover
from datasources.pools import (build_candidates_from_pools,
                               fetch_history_pool_candidates, fetch_industry_em,
                               fetch_pool_extras, fill_industry_local, norm_pool_row,
                               parse_pool, pool_raw)
from datasources.concepts import (fetch_board_closes, fetch_close_range,
                                  fetch_concept_boards, fetch_stock_concepts)
from datasources.calendar import resolve_trade_date, trade_calendar
