# -*- coding: utf-8 -*-
"""全市场快照：东财直连（自适应分页/主机轮换）→ 新浪降级；含缓存有效性判定。"""
from __future__ import annotations

import datetime as dt
import time
from typing import Optional

import pandas as pd

import cache_store
from .http import EM_HEADERS, EM_SPOT_HOSTS, EM_SPOT_PATH, _http_get_json

# 快照完整性容忍：实得行数比 total 少超过这个数即判残缺（一页 80/100/1000 行，
# 丢一页必超）。≤ 它视为拉取途中的上市/退市漂移：可用但不落缓存。
SPOT_ROW_SLACK = 10


# ---------------------------------------------------------------- 快照（归类核心）

def _em_rows_to_frame(rows: list) -> pd.DataFrame:
    """东财快照行 → 标准列 DataFrame（纯函数，可离线单测）。

    单位契约（与新浪/涨停池对齐）：
    - turnover：东财 f8 为百分数（2.53=2.53%）→ 统一为小数（0.0253），
      消费方 second_board C 维按小数阈值比较——2026-09-01 审计修复；
    - volume：手（与腾讯日K一致）；pct/amount：百分数/元，两源本就一致。
    """
    df = pd.DataFrame(rows)
    rename = {"f12": "code", "f14": "name", "f2": "close",
              "f15": "high", "f16": "low", "f17": "open", "f18": "prev_close",
              "f3": "pct", "f5": "volume", "f6": "amount", "f8": "turnover"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for c in ("close", "high", "low", "open", "prev_close", "pct",
              "volume", "amount", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = df["turnover"] / 100.0
    return df


def _spot_em_direct() -> pd.DataFrame:
    """直连东财全市场快照（自适应页大小 + 主机轮换）。

    先按 1000/页探测；服务端若按小页（如 100）截断则自动回退小页分页。
    快照含当日 open/volume（f17/f5）——增量拼接日K的数据源。
    """
    base = {
        "po": 1, "np": 1, "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2, "invt": 2, "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f2,f3,f5,f6,f8,f12,f14,f15,f16,f17,f18",
    }
    referer = "https://quote.eastmoney.com/center/gridlist.html"

    def _page(host: str, page: int, pz: int, retries: int = None) -> dict:
        # retries=None 用默认重试；首页探测传 retries=1 快速失败（东财整体断连时
        # 3 主机 × 3 重试 × 6s 超时会白等一分多钟才降级到新浪）
        return _http_get_json(f"https://{host}{EM_SPOT_PATH}", dict(base, pn=page, pz=pz),
                              referer=referer, timeout=8, retries=retries)

    data = None
    host = EM_SPOT_HOSTS[0]
    for h in EM_SPOT_HOSTS:
        try:
            data = _page(h, 1, 1000, retries=1)   # 探测：单次快试选主机
            host = h
            break
        except Exception:
            continue
    if data is None:
        raise ValueError("东财快照各主机均不可达")

    total = int(data.get("total") or 0)
    if total <= 0:
        raise ValueError("东财快照响应缺 total：无法对账完整性，弃用（防静默缺页）")
    diff = data.get("diff")
    rows = list(diff) if isinstance(diff, list) else []
    if total > len(rows) and len(rows) < 1000:
        pz, page, rows = 100, 1, []          # 服务端截断了页大小 → 回退小页重拉
    else:
        pz, page = 1000, 2
    while len(rows) < total:
        try:
            data = _page(host, page, pz)
        except Exception:
            break
        diff = data.get("diff")
        if not isinstance(diff, list) or not diff:
            break
        rows.extend(diff)
        page += 1
        time.sleep(0.12)

    if not rows:
        raise ValueError("东财快照返回空")
    # 完整性对账（照 pools.parse_pool 先例）：截断的快照驱动候选生成会悄悄漏票，
    # 且落盘后重跑命中缓存不再告警。缺行超过容忍即抛错触发新浪降级——
    # 此前缺到 49% 也只告警放行。
    short = total - len(rows)
    if short > SPOT_ROW_SLACK:
        raise ValueError(f"东财快照分页中断：{len(rows)}/{total} 只"
                         f"（缺 {short}），超出容忍 {SPOT_ROW_SLACK} 只 → 触发新浪降级")
    df = _em_rows_to_frame(rows)
    df.attrs["partial"] = short > 0
    return df


def _spot_sina() -> pd.DataFrame:
    """新浪快照（自解析：保留换手率 turnoverratio，akshare 原版会裁剪该字段）。"""
    try:
        import requests as _requests
        from akshare.utils import demjson as _demjson
        from akshare.stock.cons import (zh_sina_a_stock_count_url,
                                        zh_sina_a_stock_payload,
                                        zh_sina_a_stock_url)
    except Exception as e:
        raise RuntimeError(f"新浪快照依赖不可用: {e}")

    count_text = _requests.get(zh_sina_a_stock_count_url, timeout=8,
                               headers=EM_HEADERS).text
    count = int(count_text.strip().strip('"') or 0)
    if count <= 0:
        raise ValueError("新浪快照总数获取失败")
    pages = (count + 79) // 80
    rows = []
    for page in range(1, pages + 1):
        payload = dict(zh_sina_a_stock_payload, page=str(page))
        r = _requests.get(zh_sina_a_stock_url, params=payload,
                          headers=EM_HEADERS, timeout=8)
        data = _demjson.decode(r.text)
        rows.extend(data if isinstance(data, list) else [])
        time.sleep(0.05)

    if not rows:
        raise ValueError("新浪快照返回空")
    # 完整性对账（与东财路径同款纪律）：残缺快照入库会静默驱动候选漏票。
    # 此前容忍 5%（约 270 只）且无重试；现缺行超过容忍即弃用（丢一页 80 行必超）。
    short = count - len(rows)
    if short > SPOT_ROW_SLACK:
        raise ValueError(
            f"新浪快照分页不完整：{len(rows)}/{count} 只（缺 {short}），"
            f"超出容忍 {SPOT_ROW_SLACK} 只，弃用残缺快照")
    raw = pd.DataFrame(rows)
    need_map = {
        "symbol": "symbol", "name": "name", "trade": "close",
        "settlement": "prev_close", "open": "open",
        "high": "high", "low": "low", "changepercent": "pct",
        "amount": "amount", "volume": "volume",
        "turnoverratio": "turnover_raw",
    }
    missing = [k for k in need_map if k not in raw.columns]
    if missing:
        raise ValueError(f"新浪快照字段变化，缺少: {missing}")
    df = pd.DataFrame({
        "code": raw["symbol"].astype(str).str[-6:],
        "name": raw["name"].astype(str),
        "close": pd.to_numeric(raw["trade"], errors="coerce"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "prev_close": pd.to_numeric(raw["settlement"], errors="coerce"),
        "pct": pd.to_numeric(raw["changepercent"], errors="coerce"),
        "amount": pd.to_numeric(raw["amount"], errors="coerce"),
        # 新浪成交量单位是股，统一为手（与东财/腾讯一致，增量拼接/量纲才可靠）
        "volume": pd.to_numeric(raw["volume"], errors="coerce") / 100.0,
    })
    # 新浪 turnoverratio 是百分数（2.53 表示 2.53%），统一换算为小数
    df["turnover"] = pd.to_numeric(raw["turnoverratio"], errors="coerce") / 100.0
    df.attrs["partial"] = count > len(rows)
    return df


def _normalize_spot(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    need = ["code", "name", "close", "high", "low", "prev_close"]
    if df is None or df.empty or not all(c in df.columns for c in need):
        return None
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    return df


def cache_is_valid(cache: dict, trade_date: dt.date) -> bool:
    """缓存有效性：版本一致 + 日期一致 + 不是复盘日盘中写的。

    「盘中写的」= 时间戳落在复盘日当天且早于收盘缓冲，判定与今天几号无关。
    此前包在 trade_date == today 里：过了零点，前一天 14:xx 的盘中快照/结果
    就永久被判成收盘数据（归档旁证：09-16/17/24/28 的 signals 封板数
    32/28/3/12，同日池口径 80/44/46/31）。
    """
    if not isinstance(cache, dict) or cache.get("date") != trade_date:
        return False
    if cache.get("version") != cache_store.CACHE_VERSION:
        return False
    if float(cache.get("ts", 0) or 0) <= 0:
        return False
    return not cache_store.is_intraday_cache(cache, trade_date)


def fetch_spot(trade_date: Optional[dt.date] = None,
               cache_dir: Optional[str] = None,
               force: bool = False,
               live: bool = True) -> Optional[pd.DataFrame]:
    """全市场快照，两级降级：东财直连(带UA) → 新浪。成功打印实际数据源。

    缓存：按交易日存于 out/cache/{date}.pkl，同一天重复运行直接复用；
    跨天滚动清理由 cache_store.prune 负责；force=True 忽略缓存强制重拉。
    live=False 时只读缓存不碰网络（历史日期复盘用，防止把实时行情错当历史数据）。
    """
    if trade_date and cache_dir and not force:
        cache = cache_store.load(cache_dir, trade_date)
        if cache_is_valid(cache, trade_date) and cache.get("spot") is not None:
            print(f"  [缓存] 复用 {trade_date} 的全市场快照"
                  f"（{len(cache['spot'])} 条），不重复拉取")
            return cache["spot"]

    if not live:
        print(f"  [提示] {trade_date} 为历史日期且无有效快照缓存，"
              f"不实时拉取（实时接口只有最新行情，无法回补历史）")
        return None

    if trade_date and trade_date == dt.date.today():
        now = dt.datetime.now().time()
        from config import AFTER_CLOSE_BUFFER_MINUTES
        if now < dt.time(15, AFTER_CLOSE_BUFFER_MINUTES):
            print("  [提示] 当前为盘中时段：本次快照是盘中数据，仅作预览；"
                  "15:05 后再次运行会自动重拉收盘数据")

    sources = (
        ("东财直连(带UA)", _spot_em_direct),
        ("新浪akshare", _spot_sina),
    )
    last_err = None
    for label, fn in sources:
        try:
            df = _normalize_spot(fn())
            if df is not None and len(df) > 100:
                partial = bool(df.attrs.get("partial"))
                print(f"  [数据源] 快照来自: {label}（{len(df)} 条）"
                      + ("［轻微缺行：本次使用、不落缓存，下次运行重拉］" if partial else ""))
                if trade_date and cache_dir and not partial:
                    try:
                        cache_store.save_spot(cache_dir, trade_date, df)
                        print(f"  [缓存] 快照已存入滚动缓存: {trade_date}")
                    except Exception as e:
                        print(f"  [提示] 快照缓存保存失败（不影响本次运行）: {e}")
                return df
        except Exception as e:
            last_err = e
            print(f"  [尝试] {label} 失败: {e}")
    print(f"  [错误] 两个快照源均不可用，最后错误: {last_err}")
    return None
