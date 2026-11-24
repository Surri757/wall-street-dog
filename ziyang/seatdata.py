# -*- coding: utf-8 -*-
"""数据层：席位明细（实时更新）+ 增强快照 + 增强日K缓存。

自包含实现，不依赖主工程 datasources（本模块可整体摘除）；唯一的跨包
依赖是零依赖叶子 fstore——全项目落盘只走它，本模块原先自带的三个
tmp+replace 助手已并入（2026-11-24）。
数据源降级纪律与主工程一致：东财直连带浏览器 UA，失败重试。
"""
from __future__ import annotations

import datetime as dt
import csv
import glob as _glob
import io
import math
import os
import pickle
import re as _re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr

import fstore
import pandas as pd
import requests

from . import config as C
from .progress import Progress

_UA = {"User-Agent": C.UA, "Referer": "https://quote.eastmoney.com/"}

# 每线程 requests.Session（keep-alive 复用连接，省掉每请求 TLS 握手；
# Session 本身跨线程不保证安全，threading.local 隔离后各线程独享）。
_tls = threading.local()


def _http() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(_UA)
        _tls.s = s
    return s


# ---------------------------------------------------------------- 基础工具
def now_cn() -> dt.datetime:
    """北京时间（naive）：全部盘中/发布守卫统一用此钟。

    此前 5 处守卫用裸 dt.datetime.now()——宿主机非北京时区（如 UTC 服务器）
    时 15:05/19:30 守卫整体平移，会造成「席位当日买入不可见→miss 冻结」
    这类守卫本要防的事故。zoneinfo 不可用时退回本地钟（行为同旧版）。
    """
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    except Exception:                              # noqa: BLE001 无 tz 数据退本地钟
        return dt.datetime.now()


def is_main_board(code: str) -> bool:
    return str(code).zfill(6)[:3] in C.MAIN_BOARD_PREFIXES


def _retry_get(url: str, params: dict, retries: int = None,
               timeout: float = None):
    retries = C.FETCH_RETRIES if retries is None else retries
    timeout = C.HTTP_TIMEOUT if timeout is None else timeout
    last = None
    for _ in range(retries):
        try:
            r = _http().get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:                     # noqa: BLE001 免费源什么都可能抛
            last = e
            time.sleep(C.FETCH_SLEEP)
    raise last


def probe_yyb_code(name: str, stock_code: str, day: str) -> str | None:
    """按席位名探东财 yyb 代码（个股龙虎榜明细 API，1 请求；legion 评审用）。

    SSRF 边界：协议/主机/路径三要素为代码常量；动态内容（股票代码/日期）
    只进查询参数并先做数字过滤；_http 会话同 seatdata 全部既有请求。
    """
    stock_code = "".join(ch for ch in str(stock_code) if ch.isdigit())[:6]
    day = "".join(ch for ch in str(day) if ch.isdigit())[:8]
    if len(stock_code) != 6 or len(day) != 8:
        return None
    scheme, host, path = ("https", "datacenter-web.eastmoney.com",
                          "/api/data/v1/get")
    url = f"{scheme}://{host}{path}"
    params = {"reportName": "RPT_BILLBOARD_DAILYDETAILSBUY", "columns": "ALL",
              "filter": (f"(SECURITY_CODE=\"{stock_code}\")"
                         f"(TRADE_DATE='{day[:4]}-{day[4:6]}-{day[6:]}')"),
              "pagesize": 50, "source": "WEB", "client": "WEB"}
    try:
        rows = _retry_get(url, params, retries=1, timeout=8).json()
        for row in (rows.get("result") or {}).get("data") or []:
            if row.get("OPERATEDEPT_NAME") == name:
                return row.get("OPERATEDEPT_CODE")
    except Exception:                          # noqa: BLE001 探码失败不阻断评审
        return None
    return None


# ---------------------------------------------------------------- 席位明细
def fetch_seat_trades(retries: int = 3) -> pd.DataFrame:
    """全量拉取席位龙虎榜历史明细（akshare → 东财数据中心，指数退避重试——
    路线图#7：此前单发无重试，一次抖动=当日全链失败）。

    返回原始行（同股同日可因多个上榜原因重复）。akshare 内部 tqdm 进度条
    重定向丢弃，保持控制台干净。
    """
    import akshare as ak
    last = None
    for attempt in range(max(1, retries)):
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                df = ak.stock_lhb_yyb_detail_em(symbol=C.SEAT_CODE)
            df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.date
            for c in ("买入金额", "卖出金额", "净额", "涨跌幅"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:                      # noqa: BLE001 免费源什么都可能抛
            last = e
            time.sleep(C.FETCH_SLEEP * (attempt + 1))
    raise last


def dedupe_events(raw: pd.DataFrame) -> pd.DataFrame:
    """(日期, 代码) 去重：同股同日多个榜单口径（日榜/3日榜）各披露一次。

    买入/卖出取各口径最大值（3日榜为窗口累计 ≥ 当日；日榜为当日），
    涨跌幅取最大值。该口径高估隔日单日金额时不影响「是否买入/当日涨停」
    两个核心判定，画像与台账只用这两个判定 + 金额做量级参考。
    """
    g = (raw.groupby(["交易日期", "股票代码"], as_index=False)
         .agg(名称=("股票名称", "last"), 涨跌幅=("涨跌幅", "max"),
              买入金额=("买入金额", "max"), 卖出金额=("卖出金额", "max"),
              榜单原因=("上榜原因", lambda s: " | ".join(sorted(set(s))))))
    g["净买额"] = g["买入金额"] - g["卖出金额"].fillna(0)
    g["主板"] = g["股票代码"].map(is_main_board)
    return g.sort_values(["交易日期", "股票代码"]).reset_index(drop=True)


def update_seat_trades() -> pd.DataFrame:
    """实时更新：全量拉取 → 幂等覆盖落盘（原子写）→ 返回去重事件表。"""
    raw = fetch_seat_trades()
    if raw.empty:
        raise RuntimeError("席位明细拉取为空（东财接口异常？），保留旧文件")
    fstore.write_dataframe_csv(C.SEAT_TRADES_CSV, raw)
    return dedupe_events(raw)


def load_seat_trades() -> pd.DataFrame:
    """读盘去重事件表；无文件时现场拉取。"""
    if os.path.exists(C.SEAT_TRADES_CSV):
        raw = pd.read_csv(C.SEAT_TRADES_CSV, dtype={"股票代码": str},
                          encoding="utf-8-sig")
        raw["交易日期"] = pd.to_datetime(raw["交易日期"]).dt.date
        return dedupe_events(raw)
    return update_seat_trades()


# ---------------------------------------------------------------- 增强快照
# 主工程快照无 量比/流通市值/行业，本模块自带扩展字段版（东财 push2 同接口多字段）。
# 行情集群单主机偶发 502/断连（主工程已踩过），主机轮换兜底。
_SPOT_FIELDS = "f2,f3,f5,f6,f8,f10,f12,f14,f15,f16,f17,f18,f20,f21,f100,f109"
_SPOT_HOSTS = ("82.push2.eastmoney.com", "push2.eastmoney.com",
               "1.push2.eastmoney.com")


def _spot_data(host: str, pn: int, retries: int = None,
               timeout: float = None) -> dict:
    """东财快照单页原始 data（含 diff/total）。失败抛出→整链走新浪兜底。"""
    base = {"po": 1, "np": 1, "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2, "invt": 2, "fid": "f12",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": _SPOT_FIELDS}
    r = _retry_get(f"https://{host}/api/qt/clist/get", dict(base, pn=pn, pz=200),
                   retries=retries, timeout=timeout)
    return r.json().get("data") or {}


def fetch_snapshot() -> pd.DataFrame:
    """全市场 A 股快照（含量比 f10、流通市值 f21、所属行业 f100）。

    主机探测短超时单发（3 台最坏 ~9s 而非 3×8s×重试）；首页即探测响应，
    不重复请求；其余分页并发拉取。
    """
    host, data = None, {}
    for h in _SPOT_HOSTS:                      # 探测可用主机（短超时快速失败）
        try:
            data = _spot_data(h, 1, retries=1, timeout=C.SNAP_PROBE_TIMEOUT)
            host = h
            break
        except Exception:
            continue
    if host is None:
        raise RuntimeError("东财快照各主机均不可达")
    rows = list(data.get("diff") or [])
    total = int(data.get("total") or 0)
    pages = list(range(2, -(-total // 200) + 1)) if total else []
    pg = Progress("快照·分页", len(pages) + 1)
    pg.update(1)
    n_done = 1
    with ThreadPoolExecutor(max_workers=C.SNAP_WORKERS) as ex:
        futs = [ex.submit(_spot_data, host, p) for p in pages]
        for f in as_completed(futs):
            rows.extend(f.result().get("diff") or [])
            n_done += 1
            pg.update(n_done)
    pg.done(f"{len(rows)} 行")
    if not rows:
        raise RuntimeError("东财快照返回空")
    df = _rows_to_snapshot(rows)
    df.attrs["source"] = "东财扩展快照"
    return df


def _spot_sina_list() -> pd.DataFrame:
    """新浪全市场列表兜底（无行业/量比——行业走归档映射，量比由日K补算）。

    新浪量纲（实测，与主工程契约一致）：volume 股→本模块统一手；amount 元；
    mktcap/nmc 万元→元；turnoverratio 百分数。新浪对并发极敏感（并发分页
    容易触发 456 限流），保持串行 + 0.1s 间隔。
    """
    def _page(p: int) -> list:
        r = _retry_get(
            "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData",
            {"page": p, "num": 100, "node": "hs_a", "symbol": "", "_s_r_a": "page"},
            retries=2)
        try:
            return r.json() or []
        except Exception:
            return []

    rows: list = []
    page, tty = 1, hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    while True:
        data = _page(page)
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
        page += 1
        if tty:                              # 兜底路径总数未知，只报行数
            print("\r  [快照·新浪兜底] 已取 %d 行" % len(rows), end="", flush=True)
        time.sleep(0.1)
    if tty and rows:
            # \r 进度行可能比最终行长（残留「行」字）：先清行再写终行
            print("\r" + " " * 44 + "\r  [快照·新浪兜底] 共 %d 行" % len(rows),
                  flush=True)
    if not rows:
        raise RuntimeError("新浪全市场列表返回空")
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "code": df["code"].astype(str).str.zfill(6),
        "name": df["name"],
        "close": pd.to_numeric(df["trade"], errors="coerce"),
        "pct": pd.to_numeric(df["changepercent"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce") / 100.0,
        "amount": pd.to_numeric(df["amount"], errors="coerce"),
        "turnover": pd.to_numeric(df["turnoverratio"], errors="coerce"),
        "vr": None,
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "prev_close": pd.to_numeric(df["settlement"], errors="coerce"),
        "total_mv": pd.to_numeric(df["mktcap"], errors="coerce") * 1e4,
        "float_mv": pd.to_numeric(df["nmc"], errors="coerce") * 1e4,
        "industry": None,
        "m5": None,          # 新浪无5日涨幅——动量通道由日K缓存补算（见 screen）
    })
    return out


def _rows_to_snapshot(rows) -> pd.DataFrame:
    """东财扩展快照行 → 标准列（量纲契约见 _spot_sina_list docstring）。"""
    df = pd.DataFrame(rows).rename(columns={
        "f12": "code", "f14": "name", "f2": "close", "f3": "pct",
        "f5": "volume", "f6": "amount", "f8": "turnover", "f10": "vr",
        "f15": "high", "f16": "low", "f17": "open", "f18": "prev_close",
        "f20": "total_mv", "f21": "float_mv", "f100": "industry",
        "f109": "m5"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in ("close", "pct", "volume", "amount", "turnover", "vr",
              "high", "low", "open", "prev_close", "total_mv", "float_mv", "m5"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 本模块量纲契约：换手率一律存百分数（东财 f8/f61 原生就是百分数），
    # 与主工程的小数口径不同——features.turn_bucket 按百分数分桶。
    return df


def _snap_reusable(mtime: dt.datetime, now: dt.datetime, anchor: dt.date) -> bool:
    """磁盘快照可否直接复用：拉取时刻与当前都落在 anchor 收盘后窗口内
    （anchor 15:05 ～ 次日 09:15）。收盘后快照是静态的，同晚间/跨零点重跑
    复用即为当日真值（跨零点新拉反而拿到次日盘前滚动数据）；开盘后
    （≥09:15）不再复用。下游 snapshot_is_fresh 仍会做涨幅探测兜底。"""
    lo = dt.datetime.combine(anchor, dt.time(15, 5))
    hi = lo + dt.timedelta(hours=18, minutes=10)
    return lo <= mtime <= now < hi


def fetch_snapshot_any() -> pd.DataFrame:
    """快照门面：当日收盘后窗口内复用磁盘快照 → 东财扩展（含行业/量比）→
    新浪列表 → 磁盘缓存快照。

    成功即落盘缓存（供复用与东财+新浪同时不可用时兜底）；磁盘快照只复用
    结构字段（名称/流通市值/行业），涨跌幅走新鲜度校验自动降级为归档口径
    （跨零点事故后的既定纪律），不产出废单。
    """
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(_SNAPSHOT_CSV))
    except Exception:
        mtime = None
    if mtime is not None and _snap_reusable(mtime, now_cn(), last_trade_date()):
        df = _load_snapshot_cache()
        if df is not None:
            df.attrs["source"] = "磁盘复用（%s 收盘后快照）" % mtime.strftime("%m-%d %H:%M")
            return df
    try:
        df = fetch_snapshot()
        _save_snapshot_cache(df)
        return df
    except Exception as e:
        try:
            df = _spot_sina_list()
        except Exception:
            df = _load_snapshot_cache()
            if df is not None:
                df.attrs["source"] = "磁盘快照缓存（东财+新浪均不可用）"
                return df
            raise
        df.attrs["source"] = f"新浪列表兜底（东财不可用：{type(e).__name__}）"
        _save_snapshot_cache(df)
        return df


_SNAPSHOT_CSV = os.path.join(C.DATA_DIR, "snapshot_cache.csv")


def _save_snapshot_cache(df: pd.DataFrame) -> None:
    try:
        fstore.write_dataframe_csv(_SNAPSHOT_CSV, df)
    except Exception:
        pass                                  # 缓存失败不影响主流程


def _load_snapshot_cache() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(_SNAPSHOT_CSV, dtype={"code": str}, encoding="utf-8-sig")
        if not len(df):
            return None
        df["code"] = df["code"].str.zfill(6)
        stamp = dt.datetime.fromtimestamp(os.path.getmtime(_SNAPSHOT_CSV))
        df.attrs["source"] = f"磁盘快照缓存（{stamp:%m-%d %H:%M}）"
        return df
    except Exception:
        return None


# ---------------------------------------------------------------- 归档读取
def archive_signal_files() -> list[tuple[dt.date, str]]:
    """主工程每日信号归档 (日期, 路径) 列表（含涨停池/炸板池等候选明细）。"""
    out = []
    for f in _glob.glob(os.path.join(C.ROOT, "out", "archive", "signals_*.csv")):
        m = _re.search(r"(\d{8})\.csv$", f)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%d").date(), f))
    return sorted(out)


_IND_MAP_MEMO: dict | None = None     # 进程内记忆化：update/predict 同跑只扫一次归档


def archive_industry_map() -> dict[str, str]:
    """code→行业，取归档中最近出现的一次（主工程行业口径）。

    进程内记忆化——同一次运行（update→predict）归档不变，二次调用零 IO；
    返回拷贝，调用方 update 快照行业不会污染记忆。
    """
    global _IND_MAP_MEMO
    if _IND_MAP_MEMO is not None:
        return dict(_IND_MAP_MEMO)
    out: dict[str, str] = {}
    for _, f in archive_signal_files():
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            continue
        if "industry" not in df.columns:
            continue
        sub = df[df["industry"].notna() & df["code"].notna()].copy()
        sub["code"] = sub["code"].str.zfill(6)
        out.update(dict(zip(sub["code"], sub["industry"])))
    _IND_MAP_MEMO = dict(out)
    return out


def archive_pool_pct(trade_date: dt.date) -> dict[str, float]:
    """主工程某日候选池（涨停/炸板/跌停/振幅池）的 {code: 当日涨幅%}。

    快照滚动到次日盘前后（跨零点运行），实时快照的涨跌幅不再描述
    trade_date——当日强势通道从此处取（主工程 15:10 已归档，零网络）。
    """
    for d, f in archive_signal_files():
        if d != trade_date:
            continue
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            return {}
        if "pct" not in df.columns:
            return {}
        sub = df[df["code"].notna()].copy()
        sub["code"] = sub["code"].str.zfill(6)
        return dict(zip(sub["code"], pd.to_numeric(sub["pct"], errors="coerce")))
    return {}


def archive_union_codes(trade_date: dt.date, days: int = 20) -> set[str]:
    """近 N 自然日主工程归档 signals_*.csv 的主板 code 并集（零网络枚举）。

    改动四①扫描基超集的第二只脚：∪基 = 缓存全部主板 code ∪ 本集合，按构造
    ⊇ 理想旧基（缓存+四通道），归档并集覆盖缓存横截面塌缩期离场的票（实测
    end-09-24 并集 1054 vs 缓存新鲜横截面 540）。注意：并集不独立于主工程管道
    健康（归档停更则并集冻结），监控须双指标（归档行数 + 并集规模）。
    """
    out: set[str] = set()
    lo = trade_date - dt.timedelta(days=days - 1)   # 恰 20 个自然日 [D-19, D]
    for d, f in archive_signal_files():
        if not (lo <= d <= trade_date):
            continue
        try:
            df = pd.read_csv(f, dtype={"code": str}, encoding="utf-8-sig")
        except Exception:
            continue
        if "code" not in df.columns:
            continue
        out |= {str(c).zfill(6) for c in df["code"].dropna() if is_main_board(str(c))}
    return out


def cache_last_date(cache: dict, code: str) -> str | None:
    """缓存中某票日K末日期（iso）；无缓存返回 None（扫描基补拉量指标用）。"""
    b = cache.get(code)
    if b is None or not len(b):
        return None
    return str(b["date"].iloc[-1])[:10]


def industry_map(snapshot: pd.DataFrame | None = None) -> dict[str, str]:
    """行业映射门面：归档（池内股票，主工程行业口径）∪ 快照行业（全市场，东财可用时）。"""
    m = archive_industry_map()
    if snapshot is not None and "industry" in getattr(snapshot, "columns", []):
        s = snapshot[snapshot["industry"].notna()]
        m.update(dict(zip(s["code"].astype(str).str.zfill(6), s["industry"])))
    return m


# ---------------------------------------------------------------- baostock 源（盘后主力）
# 免费证券宝：真换手率/成交额、前复权（与东财同基）、无 WAF；日K 17:30 入库。
# 会话是模块级单例且非线程安全 → 只做串行段（吞吐实测 ~5 只/秒）；库自身无
# 请求超时，登录前临时设 socket 默认超时兜底挂死、登录后即恢复（默认超时是
# 进程级副作用：daily_sync 先于 vane/auction/报告执行，不恢复会把宿主其后所有
# 未显式设超时的网络调用一并改成 20s）。东财 2025-04 起限流收紧（社区安全线
# ~1 次/秒），push2his 不再适合 780 只量级的批量主力，baostock 承担盘后
# （≥17:35）批量段。
_BS = None
_BS_SKIP = False          # 线程池相置 True：bs 已在串行相试过，失败码不再撞它


def _bs_ready(end: str, now: dt.datetime | None = None) -> bool:
    """baostock 是否已可能有 end 日数据（当日 17:30 入库，留 5 分钟余量）。"""
    d = dt.datetime.strptime(end, "%Y%m%d")
    return (now or now_cn()) >= d + dt.timedelta(hours=17, minutes=35)


def _bs_login():
    global _BS
    if _BS is not None:
        return _BS
    import socket as _socket

    import baostock as bs
    prev_timeout = _socket.getdefaulttimeout()
    if prev_timeout is None:
        _socket.setdefaulttimeout(20)          # baostock 自建 socket 无超时，防挂死
    try:
        # baostock 的连接在 login 内创建并继承此刻的默认超时；之后在同一连接上
        # 收发，恢复默认值不影响它
        logged_in = bs.login().error_code == "0"
    finally:
        _socket.setdefaulttimeout(prev_timeout)
    if not logged_in:
        return None
    _BS = bs
    return _BS


def _bs_rows_to_df(rows: list) -> pd.DataFrame:
    """baostock 行（date,open,high,low,close,volume,amount,turn 字符串）→
    本模块日K契约：volume 股→手，turnover 百分数原样，停牌空值→None。"""
    out = []
    for r in rows:
        v = [float(x) if x not in ("", None) else None for x in r[1:8]]
        if v[3] is None:                       # close 无效行
            continue
        out.append({"date": r[0], "open": v[0], "high": v[1], "low": v[2],
                    "close": v[3], "volume": (v[4] or 0.0) / 100.0,
                    "amount": v[5], "pct": None, "turnover": v[6]})
    return pd.DataFrame(out)


def _bs_fetch(code: str, beg: str, end: str) -> pd.DataFrame:
    """baostock 前复权日K（含真换手率）。失败抛出，由链路降级；仅串行调用。"""
    bs = _bs_login()
    if bs is None:
        raise RuntimeError("baostock 登录失败")
    sym = ("sh." if code.startswith(("6", "9")) else "sz.") + code
    beg_s = f"{beg[:4]}-{beg[4:6]}-{beg[6:]}"
    end_s = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    global _BS
    rows = []
    for _ in (1, 2):                           # 掉线重登一次
        rs = bs.query_history_k_data_plus(
            sym, "date,open,high,low,close,volume,amount,turn",
            start_date=beg_s, end_date=end_s, frequency="d", adjustflag="2")
        if rs.error_code == "0":
            while rs.next():
                rows.append(rs.get_row_data())
            break
        try:
            bs.logout()
        except Exception:
            pass
        _BS = None
        bs = _bs_login()
        if bs is None:
            break
    return _bs_rows_to_df(rows)


# ---------------------------------------------------------------- 增强日K缓存
# 主工程 fetch_daily 只有 OHLCV；本模块另需 换手率/成交额（前夜特征）。
# 降级链：东财 push2his（含换手率 f61/成交额 f57）→ 腾讯 fqkline（仅 OHLCV，
# 换手率缺失走中性桶）。东财对该端点偶发 RemoteDisconnected（README 已记录，
# 高频拉取会触发），连续 3 次失败后本次运行直接走腾讯。
_EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# 腾讯多主机：web.* 高频拉取会被 WAF 软封锁（501 跳 waf.tencent.com，
# 2026-09-17 实测约数百次后触发，封锁期撞击会续期）；裸域独立计数。失败自动
# 轮换并锁定健康主机；全主机 501 → 全局冷却 5 分钟（退避，不撞击）。
_TX_KLINE_HOSTS = ("https://web.ifzq.gtimg.cn", "https://ifzq.gtimg.cn")
_TX_PATH = "/appstock/app/fqkline/get"
# 新浪日K第三源（与新浪列表接口是不同服务，限流独立）：不复权原始价——
# 与缓存里的前复权混源时靠 get_bars 重叠锚校验兜底（漂移>0.5% 整体换锚
# 重拉，序列内部始终自洽）；volume 股→手，无 amount/turnover（同腾讯）。
_SINA_KLINE = ("https://quotes.sina.cn/cn/api/jsonp_v2.php"
               "/var _=zk/CN_MarketDataService.getKLineData")
_EM_FAILS = 0
_EM_MAX_FAILS = 3
_TX_HOST = 0
_TX_BLOCKED_UNTIL = 0.0


def _fetch_bars_range_em(code: str, beg: str, end: str) -> pd.DataFrame:
    secid = ("1." if code.startswith(("6", "9")) else "0.") + code
    params = {"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f59,f61",
              "klt": "101", "fqt": "1", "beg": beg, "end": end}
    r = _retry_get(_EM_KLINE, params, retries=1).json()
    kl = (r.get("data") or {}).get("klines") or []
    rows = []
    for line in kl:
        p = str(line).split(",")
        if len(p) >= 9:
            rows.append({"date": p[0], "open": float(p[1]), "close": float(p[2]),
                         "high": float(p[3]), "low": float(p[4]),
                         "volume": float(p[5]), "amount": float(p[6]),
                         "pct": float(p[7]), "turnover": float(p[8])})
    return pd.DataFrame(rows)


def _fetch_bars_range_tx(code: str, beg: str, end: str) -> pd.DataFrame:
    global _TX_HOST
    sym = ("sh" if code.startswith(("6", "9")) else "sz") + code
    n = max(30, (dt.datetime.strptime(end, "%Y%m%d")
                 - dt.datetime.strptime(beg, "%Y%m%d")).days + 10)
    params = {"param": f"{sym},day,,,{n},qfq"}
    rows, last = [], None
    for off in range(len(_TX_KLINE_HOSTS)):    # WAF 软封锁时自动轮换主机
        host = (_TX_HOST + off) % len(_TX_KLINE_HOSTS)
        try:
            r = _retry_get(_TX_KLINE_HOSTS[host] + _TX_PATH, params, retries=1)
            data = (r.json().get("data") or {}).get(sym) or {}
            kl = data.get("qfqday") or data.get("day") or []
            _TX_HOST = host                    # 锁定健康主机，后续直连
            break
        except Exception as e:                 # noqa: BLE001 换下一台主机
            last = e
    else:
        raise last
    for p in kl:
        # 腾讯行：[date, open, close, high, low, volume(手), ...]
        if len(p) >= 6:
            rows.append({"date": p[0], "open": float(p[1]), "close": float(p[2]),
                         "high": float(p[3]), "low": float(p[4]),
                         "volume": float(p[5]), "amount": None,
                         "pct": None, "turnover": None})
    df = pd.DataFrame(rows)
    if len(df):
        df = df[(df["date"] >= f"{beg[:4]}-{beg[4:6]}-{beg[6:]}")
                & (df["date"] <= f"{end[:4]}-{end[4:6]}-{end[6:]}")]
    return df.reset_index(drop=True)


def _fetch_bars_range_sina(code: str, beg: str, end: str) -> pd.DataFrame:
    """新浪日K第三源（JSONP 包 JSON 列表）。量纲与列对齐腾讯源（手/无换手）。"""
    import json as _json
    sym = ("sh" if code.startswith(("6", "9")) else "sz") + code
    n = max(30, (dt.datetime.strptime(end, "%Y%m%d")
                 - dt.datetime.strptime(beg, "%Y%m%d")).days + 10)
    r = _retry_get(_SINA_KLINE,
                   {"symbol": sym, "scale": 240, "ma": "no", "datalen": min(n, 250)},
                   retries=2)
    m = _re.search(r"\((.*)\)", r.text, _re.S)
    kl = _json.loads(m.group(1)) if m else []
    rows = []
    for p in kl:
        if not all(p.get(k) for k in ("day", "open", "close", "high", "low", "volume")):
            continue
        rows.append({"date": p["day"], "open": float(p["open"]),
                     "close": float(p["close"]), "high": float(p["high"]),
                     "low": float(p["low"]), "volume": float(p["volume"]) / 100.0,
                     "amount": None, "pct": None, "turnover": None})
    df = pd.DataFrame(rows)
    if len(df):
        df = df[(df["date"] >= f"{beg[:4]}-{beg[4:6]}-{beg[6:]}")
                & (df["date"] <= f"{end[:4]}-{end[4:6]}-{end[6:]}")]
    return df.reset_index(drop=True)


def _fetch_bars_range(code: str, beg: str, end: str,
                      only: str = None) -> pd.DataFrame:
    """时间感知四级降级取日K：baostock（盘后，真换手）→ 东财 → 腾讯（多主机
    +封锁退避）→ 新浪。only="bs" 时只问 baostock（批量串行相专用，快速失败）。

    全部失败返回空表（该票特征走缺失中性桶，不炸流程）。
    """
    global _EM_FAILS, _TX_BLOCKED_UNTIL, _BS_SKIP
    if only == "bs" or (not _BS_SKIP and _bs_ready(end)):
        try:
            df = _bs_fetch(code, beg, end)
            if len(df) or only == "bs":
                return df                      # only 模式：空表也如实返回，不降级
        except Exception:
            if only == "bs":
                raise                          # 串行相靠异常快速转池
    if _EM_FAILS < _EM_MAX_FAILS:
        try:
            df = _fetch_bars_range_em(code, beg, end)
            _EM_FAILS = 0
            return df
        except Exception:
            _EM_FAILS += 1
    try:
        if time.time() < _TX_BLOCKED_UNTIL:
            raise RuntimeError("腾讯冷却中（WAF 退避）")
        return _fetch_bars_range_tx(code, beg, end)
    except requests.HTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 501:
            _TX_BLOCKED_UNTIL = time.time() + 300    # WAF 封锁退避：撞击会续期
    except Exception:
        pass                                          # 网络类失败：不连坐腾讯
    try:
        return _fetch_bars_range_sina(code, beg, end)
    except Exception:
        return pd.DataFrame()


def load_bars_cache() -> dict:
    if os.path.exists(C.BARS_CACHE_PKL):
        try:
            with open(C.BARS_CACHE_PKL, "rb") as f:
                return pickle.load(f)
        except Exception as e:                      # noqa: BLE001 路线图#6：损坏要喊
            import warnings
            warnings.warn(f"bars_cache.pkl 损坏（{type(e).__name__}），本次视为空缓存"
                          "——全量重拉约5-10分钟，画像/预判成本上升", stacklevel=2)
            return {}
    return {}


def save_bars_cache(cache: dict) -> None:
    fstore.write_pkl(C.BARS_CACHE_PKL, cache)


def _norm_date(s) -> str:
    return str(s)[:10]


def _splice_bars(have: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """缓存旧段 + 新拉片段拼接；全空列不进 concat，拼完按原位置、原 dtype 补回。

    结果与直接 `pd.concat([have, add])` 逐值相同、列集合相同，只是不触发 pandas
    的弃用警告。警告的触发条件（pandas 2.2 `_concatenate_join_units`）：同一列在
    一段有值、在另一段是**另一 dtype 的全空列**，新旧 dtype 判定规则结果不同。
    本仓的实际形态：腾讯源的新拉片段带 amount/turnover 的 object 型全空列，而缓存
    旧段同名列是 float64 实值——此前只剔除「旧段没有的全空列」，这种同名的漏了，
    于是每拼一只票刷一条 FutureWarning（2026-11-24 一次运行刷了 283 条，占日志
    98%，真正的报错被淹没）。
    """
    # 新段的全空列一律不进 concat：它们对新日期只贡献 NaN，与拼接后再对齐等价
    empty_new = [c for c in add.columns if add[c].isna().all()]
    if empty_new:
        add = add.drop(columns=empty_new)
    # 两边都空的列：摘掉再补回（旧段空、新段有值的照常参与拼接）
    both_na = [(i, c, have[c].dtype) for i, c in enumerate(have.columns)
               if have[c].isna().all()
               and (c not in add.columns or add[c].isna().all())]
    if not both_na:
        return pd.concat([have, add], ignore_index=True)
    drop = [c for _, c, _ in both_na]
    merged = pd.concat([have.drop(columns=drop), add], ignore_index=True)
    for i, c, dtype in sorted(both_na):
        # 显式用 nan 填充：pd.Series(index=…, dtype=object) 补出来的是 None，
        # 与直接 concat 的结果（nan）在相等性检查里不等价
        merged.insert(i, c, pd.Series(float("nan"), index=merged.index, dtype=dtype))
    return merged


def get_bars(cache: dict, code: str, end: dt.date, min_bars: int = 30,
             sleep: float = None, only: str = None) -> pd.DataFrame | None:
    """取个股日K至 end 日（含），缓存增量拼接。返回按日期升序 DataFrame。

    前复权重锚保护：增量拼接前比对重叠日期收盘价，不一致（除权后复权锚
    漂移）则丢弃旧缓存全量重拉。腾讯源行无 pct/turnover（由消费方从收盘价
    序列统一推算涨幅；换手缺失走中性桶）。
    """
    sleep = C.BARS_SLEEP if sleep is None else sleep
    have = cache.get(code)
    if have is None or len(have) == 0:
        have = None
    if have is not None and len(have) >= min_bars:
        last = dt.datetime.strptime(_norm_date(have["date"].iloc[-1]), "%Y-%m-%d").date()
        if last >= end:
            out = have[have["date"].map(_norm_date) <= end.isoformat()]
            return out.reset_index(drop=True) if len(out) >= min_bars else have
        beg = (last - dt.timedelta(days=10)).strftime("%Y%m%d")   # 留重叠区校验
        add = _fetch_bars_range(code, beg, end.strftime("%Y%m%d"), only=only)
        if len(add):
            merged = _splice_bars(have, add)
            ov = merged.drop_duplicates(subset="date", keep="last")
            ov = ov.sort_values("date").reset_index(drop=True)
            # 重叠一致性：同日期新旧收盘差 >0.5% → 复权锚漂移，全量重拉
            chk = have.merge(add, on="date", suffixes=("_old", "_new"))
            if len(chk):
                bad = (chk["close_old"] - chk["close_new"]).abs() / chk["close_old"]
                if bad.max() > 0.005:
                    df = _fetch_bars_range(
                        code, (end - dt.timedelta(days=240)).strftime("%Y%m%d"),
                        end.strftime("%Y%m%d"), only=only)
                    time.sleep(sleep)
                    if len(df) >= min_bars:
                        cache[code] = df.sort_values("date").reset_index(drop=True)
                        return cache[code]
                    # 重拉失败不得静默返回刚校验失败的旧序列（价格桶/位置桶会在
                    # 错误复权基准上计算且无痕）——大声留痕后仍返回旧数据兜底
                    print(f"[紫阳] {code} 日K复权锚漂移且全量重拉失败"
                          f"（仅得 {len(df)} 根），沿用旧序列（价格类特征可能失真）")
                    return have
            time.sleep(sleep)
            cache[code] = ov
            return ov
        time.sleep(sleep)
        return have
    # 无缓存或条数不足：全量窗口（约自然日 240 天，覆盖 20 日位置+连板判定）
    df = _fetch_bars_range(code, (end - dt.timedelta(days=240)).strftime("%Y%m%d"),
                           end.strftime("%Y%m%d"), only=only)
    time.sleep(sleep)
    if len(df) < min_bars:
        if len(df):
            cache[code] = df
        return cache.get(code) if len(df) else None
    cache[code] = df.sort_values("date").reset_index(drop=True)
    return cache[code]


def fetch_bars_batch(cache: dict, ends: dict, tag: str,
                     workers: int = None, save_every: int = 100) -> None:
    """把多只股票日K补齐到各自 end 日（{code: end}），逐只逻辑与 get_bars
    完全一致（增量拼接 + 复权锚校验），分两相执行（2026-09-18 三轮迭代）：

    - 串行相（盘后）：全部 end 日已过 baostock 入库点（17:35）时，逐只只问
      baostock——真换手率、无 WAF、会话串行（~5 只/秒）；东财 2025-04 起
      限流收紧（安全线 ~1 次/秒），不再承担批量主力；
    - 线程池相：串行相的失败码 + 盘中场景，走 东财→腾讯→新浪（此相跳过
      baostock），波式提交、失败率 ≥1/3 退避 3s；
    - 批末串行重试一轮（0.5s 间隔，恢复 baostock 资格——中途恢复可救回）。

    失败判定 = 空手而归（返回 None 且 cache 无该码）；单只失败不炸批（该票
    特征走缺失中性桶）。线程安全：cache 只做整键赋值（GIL 原子）。
    """
    global _BS_SKIP
    if not ends:
        return
    workers = workers or C.BARS_WORKERS
    items = list(ends.items())
    pg = Progress(tag, len(items))
    n_done, todo = 0, list(items)

    if all(_bs_ready(e.strftime("%Y%m%d")) for _, e in items):
        todo = []
        for code, end in items:                # 串行相：只问 baostock，失败快速转池
            try:
                res = get_bars(cache, code, end, sleep=0, only="bs")
            except Exception:                  # noqa: BLE001 bs 缺数/掉线 → 池兜底
                res = None
            if res is None and cache.get(code) is None:
                todo.append((code, end))
            else:
                n_done += 1
                pg.update(n_done, extra=code)
            if save_every and n_done and n_done % save_every == 0:
                try:
                    save_bars_cache(dict(cache))   # 浅拷贝快照：池线程仍在写 dict，
                except Exception:                  # pickle 遍历时并发插入会炸迭代
                    pass

    if todo:
        wave_w = max(2, workers * 2)
        _BS_SKIP = True
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for i in range(0, len(todo), wave_w):
                    wave = todo[i:i + wave_w]
                    futs = {ex.submit(get_bars, cache, c, e, sleep=0.05): c
                            for c, e in wave}
                    w_fail = 0
                    for f in as_completed(futs):
                        code = futs[f]
                        try:
                            res = f.result()
                        except Exception:       # noqa: BLE001 get_bars 罕见异常
                            res = None
                        n_done += 1
                        pg.update(n_done, extra=code)
                        if save_every and n_done % save_every == 0:
                            try:
                                save_bars_cache(dict(cache))  # 同上：快照防并发迭代炸
                            except Exception:
                                pass
                        if res is None and cache.get(code) is None:
                            w_fail += 1
                    if w_fail >= max(2, len(wave) // 3):
                        time.sleep(3.0)         # 撞限流：缓口气再发下一波
        finally:
            _BS_SKIP = False

    still = [(c, e) for c, e in todo if cache.get(c) is None]
    recovered = 0
    for j, (code, end) in enumerate(still):     # 批末重试（bs 已恢复资格）
        if j:
            time.sleep(0.5)
        try:
            get_bars(cache, code, end)
        except Exception:                       # noqa: BLE001 重试仍败走中性桶
            pass
        if cache.get(code) is not None:
            recovered += 1
    n_failed = len(still) - recovered
    note = ""
    if n_failed or recovered:
        note = "失败 %d 只" % n_failed
        if recovered:
            note += "（重试救回 %d）" % recovered
        note += "，其余走中性桶"
    pg.done(note)


def approx_lianban_series(bars: pd.DataFrame) -> pd.Series:
    """前复权近似连板数序列：日涨幅 ≥ LIMIT_UP_APPROX 连续计数（百分数口径）。

    涨幅统一从收盘价序列推算（pct = close/prev_close-1），不依赖源字段——
    东财行带 f59、腾讯行缺失，混源时保持口径一致。
    """
    close = bars["close"].astype(float)
    prev = close.shift(1)
    pct = (close / prev - 1.0) * 100.0
    is_lim = pct >= C.LIMIT_UP_APPROX
    n, cnt = [], 0
    for ok in is_lim:
        cnt = cnt + 1 if ok else 0
        n.append(cnt)
    return pd.Series(n, index=bars.index)


def feats_from_bars(bars: pd.DataFrame, day: dt.date, back: int = 0,
                    float_shares: float | None = None) -> dict | None:
    """某交易日的特征行（日K为唯一事实源；画像挖掘与预判打分共用此实现）。

    back=0：day 当日 bar（预判用；停牌/未上市 → None）
    back=1：day 之前最后一根 bar（挖掘用，即点火日 D 的前夜 D-1）
    历史不足 6 根（5日均量+当日）→ None。换手率优先源字段（东财 f61 真值），
    缺失时用流通股本近似（volume手×100/股本，历史日股本取当前值，除权有
    个别漂移，桶宽下可接受）。
    """
    dates = bars["date"].map(_norm_date).tolist()
    i = None
    for j, d in enumerate(dates):
        if (d < day.isoformat() if back else d == day.isoformat()):
            i = j
            if not back:
                break
    if i is None or i < 6:
        return None
    close = float(bars["close"].iloc[i])
    v5 = bars["volume"].iloc[i - 5: i].mean()
    turn = bars["turnover"].iloc[i] if "turnover" in bars.columns else None
    if (turn is None or (isinstance(turn, float) and math.isnan(turn))) and float_shares:
        turn = float(bars["volume"].iloc[i]) * 100.0 / float_shares * 100.0
    hi20 = bars["high"].iloc[max(0, i - 19): i + 1].max()
    lo20 = bars["low"].iloc[max(0, i - 19): i + 1].min()
    return {
        "date": dates[i],
        "pct": (close / float(bars["close"].iloc[i - 1]) - 1.0) * 100.0,
        "price": close,
        "turnover": float(turn) if turn is not None and not (
            isinstance(turn, float) and math.isnan(turn)) else None,
        "vr": float(bars["volume"].iloc[i] / v5) if v5 else None,
        "pos20": float((close - lo20) / (hi20 - lo20)) if hi20 > lo20 else 0.5,
        "lianban": int(approx_lianban_series(bars.iloc[: i + 1]).iloc[-1]),
    }


def float_shares_map(snapshot: pd.DataFrame | None) -> dict[str, float]:
    """code→流通股本（股），由快照流通市值/现价折算。快照缺失时空 dict。"""
    if snapshot is None:
        return {}
    s = snapshot.dropna(subset=["float_mv", "close"])
    s = s[s["close"] > 0]
    return {str(c): mv / cl for c, mv, cl in zip(s["code"], s["float_mv"], s["close"])}


def idx_on(bars: pd.DataFrame, day: dt.date):
    """日K序列中某日的下标；无该日（停牌/未上市）返回 None。"""
    dates = bars["date"].map(_norm_date).tolist()
    try:
        return dates.index(day.isoformat())
    except ValueError:
        return None


def next_bar_after(bars: pd.DataFrame, after: dt.date) -> tuple[str, float] | None:
    """after 之后第一根日K（即下一交易日）→ (日期, 涨幅%)。无返回 None。"""
    dates = bars["date"].map(_norm_date).tolist()
    for i, d in enumerate(dates):
        if d > after.isoformat():
            pct = (bars["close"].iloc[i] / bars["close"].iloc[i - 1] - 1.0) * 100.0 \
                if i > 0 else None
            return d, (float(pct) if pct is not None else None)
    return None


_TRADE_CAL_CSV = os.path.join(C.DATA_DIR, "trade_calendar.csv")


def _trade_cal() -> dict | None:
    """iso日期 → 是否交易日。baostock 拉取 + 磁盘缓存（覆盖当日±40 天即复用，
    零网络）；baostock 不可用返回 None（调用方走旧兜底）。"""
    today = dt.date.today()
    lo = (today - dt.timedelta(days=40)).isoformat()
    hi = (today + dt.timedelta(days=40)).isoformat()
    if os.path.exists(_TRADE_CAL_CSV):
        try:
            cal = _load_trade_cal_csv()
            if cal and lo in cal and hi in cal:
                return cal
        except Exception:
            cal = None
    bs = _bs_login()
    if bs is None:
        return None
    rs = bs.query_trade_dates(
        start_date=(today - dt.timedelta(days=800)).isoformat(),
        end_date=(today + dt.timedelta(days=120)).isoformat())
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    cal = {r[0]: r[1] == "1" for r in rows}
    try:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "is_trade"])
        for k in sorted(cal):
            w.writerow([k, "1" if cal[k] else "0"])
        fstore.write_text(_TRADE_CAL_CSV, buf.getvalue(), encoding="utf-8-sig")
    except Exception:
        pass
    return cal


def _load_trade_cal_csv() -> dict:
    with open(_TRADE_CAL_CSV, encoding="utf-8-sig") as f:
        return {r["date"]: r["is_trade"] == "1" for r in csv.DictReader(f)}


def _last_trade_from_cal(cal: dict, today: dt.date, now_t: dt.time) -> dt.date:
    """日历口径的最近已收盘交易日（纯函数，供测试）：≤today 的最后一个交易
    日；若为 today 且当前 <15:05（未收盘）再退一个交易日。"""
    d = today
    for _ in range(20):                          # 最多跨 20 个自然日（春节长假）
        if cal.get(d.isoformat()):
            break
        d -= dt.timedelta(days=1)
    if d == today and now_t < dt.time(15, 5):
        d -= dt.timedelta(days=1)
        for _ in range(20):
            if cal.get(d.isoformat()):
                break
            d -= dt.timedelta(days=1)
    return d


def last_trade_date() -> dt.date:
    """最近一个**已收盘**交易日。

    交易日历优先（baostock，磁盘缓存零网络——2026-09-18 从 akshare 迁入，
    消除其对新浪日历的网络依赖与节假日误差）；日历不可用时退回粗兜底：
    today，盘前（<15:05）再减一天（容忍±假日误差，仅影响缓存效率）。
    """
    cal = _trade_cal()
    if cal:
        return _last_trade_from_cal(cal, dt.date.today(), now_cn().time())
    if now_cn().time() < dt.time(15, 5):
        return dt.date.today() - dt.timedelta(days=1)
    return dt.date.today()


_THS_POOL_DIR = os.path.join(C.DATA_DIR, "ths_pool")
_THS_URL = "https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool"
_THS_FIELDS = ("199112,10,9001,330323,330324,330325,9002,330329,"
               "133971,133970,1968584,3475914,9003,9004")


def snapshot_ths_pool(trade_date: dt.date | None = None) -> str | None:
    """同花顺涨停揭秘池每日快照（题材标签/板型/封单额）。

    该接口仅约一年滚动窗口（deep4 题材席实测 2025-09-08 起），不每日冻结
    就会蒸发——题材特征管线的历史资产保全动作，幂等（当日已存即跳过）。
    """
    trade_date = trade_date or last_trade_date()
    os.makedirs(_THS_POOL_DIR, exist_ok=True)
    path = os.path.join(_THS_POOL_DIR, f"{trade_date:%Y%m%d}.csv")
    if os.path.exists(path):
        return path
    try:
        r = _retry_get(_THS_URL, {
            "page": 1, "limit": 200, "field": _THS_FIELDS,
            "filter": "HS,GEM2STAR", "order_field": 330324,
            "order_type": 0, "date": trade_date.strftime("%Y%m%d"),
        }, retries=2)
        data = (r.json() or {}).get("data") or {}
        info = data.get("info") or []
        page, rows = 2, list(info)
        total = int(data.get("page", {}).get("total") or 0)
        while rows and len(rows) < total:
            r2 = _retry_get(_THS_URL, {
                "page": page, "limit": 200, "field": _THS_FIELDS,
                "filter": "HS,GEM2STAR", "order_field": 330324,
                "order_type": 0, "date": trade_date.strftime("%Y%m%d"),
            }, retries=1)
            d2 = (r2.json() or {}).get("data") or {}
            add = d2.get("info") or []
            if not add:
                break
            rows.extend(add)
            page += 1
            time.sleep(0.8)
        if not rows:
            return None
        fstore.write_dataframe_csv(path, pd.DataFrame(rows))
        return path
    except Exception:                            # noqa: BLE001 快照失败不影响主链
        return None
