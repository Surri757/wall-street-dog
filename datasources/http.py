# -*- coding: utf-8 -*-
"""HTTP 基础层：共享会话、通用 GET+JSON（带重试）、东财/腾讯端点常量。"""
from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urlsplit

import requests

EM_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

EM_SPOT_HOSTS = ("82.push2.eastmoney.com", "push2.eastmoney.com",
                 "1.push2.eastmoney.com")   # 行情集群单主机偶发不可达，轮换兜底
EM_SPOT_PATH = "/api/qt/clist/get"
EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TX_MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/day/query"   # 分时：近 5 个交易日
EM_POOL_BASE = "https://push2ex.eastmoney.com"


# ---------------------------------------------------------------- 出站守卫

def assert_public_url(url: str) -> None:
    """出站 URL 守卫：只放行 http/https 且主机解析到公网地址的请求。

    拒绝 localhost / 环回 / 私有 / 链路本地 / 保留 / 组播地址——校验的是
    **解析后**的地址而不只是字面量（防"公网域名解析到内网"的重绑定）。
    本项目端点全是编译期常量、无用户输入，本校验是纵深防御而非输入过滤。
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"仅允许 http/https 地址: {url[:80]}")
    host = parts.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".local", ".internal")):
        raise ValueError(f"拒绝访问本机/内网主机: {host}")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:                       # 解析失败：请求也必然失败，提前拒绝
        raise ValueError(f"主机无法解析: {host} ({e})") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"主机 {host} 解析到非公网地址 {ip}，已拒绝")


# ---------------------------------------------------------------- 基础请求

_SESSION = requests.Session()  # 连接复用：200+ 只日K逐只请求时减少握手开销
_SESSION.headers.update(EM_HEADERS)


class EmptyDataError(ValueError):
    """HTTP 成功但响应无 data 字段（上游无该标的数据，非瞬时网络故障）。

    ValueError 的子类：老调用方按 ValueError/Exception 捕获的行为不变；新调用方
    （分时采集）用它区分「这个代码没数据，跳过」与「网络抖动，重试/改日再采」。
    """


def _http_get_json(url: str, params: dict, referer: str, timeout: int = 6,
                   retries: int = None, fresh: bool = False) -> dict:
    """GET + JSON 解析，带重试（FETCH_RETRIES 次，间隔 FETCH_SLEEP 秒）。

    fresh=True 用一条全新连接发请求（不复用共享会话）：东财部分接口在不同连接
    上会返回不同版本的历史数据（见 pools.pool_raw），换连接才算一次独立观测。
    """
    from config import FETCH_RETRIES, FETCH_SLEEP
    if retries is None:
        retries = FETCH_RETRIES
    headers = dict(EM_HEADERS, Referer=referer)
    session = requests.Session() if fresh else _SESSION
    last_err = None
    for attempt in range(max(1, retries)):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            data = j.get("data")
            if data is None:
                raise EmptyDataError(f"响应无 data 字段: {str(j)[:120]}")
            return data
        except EmptyDataError:
            # 「无数据」是上游的确定答复，不是瞬时故障：重试只会每只白等
            # FETCH_RETRIES 次请求加 FETCH_SLEEP（此前落进下面的通配被重试）
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(FETCH_SLEEP)
    raise last_err
