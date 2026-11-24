# -*- coding: utf-8 -*-
"""滚动缓存：每个交易日一个缓存文件，只保留最近 N 个交易日（默认 3 天）。

目录结构:
    out/cache/20260818.pkl   当日完整数据（spot=快照，results=分析结果，可单独存在）
    out/cache/20260819.pkl
    out/cache/20260820.pkl

规则:
- 同一天内重复运行直接复用，不重复拉取；
- 周末等非交易日窗口悬停，已有数据保留、不会失效；
- 新交易日到来后窗口向前滚动，窗口外的旧日期自动清除；
- 旧版单文件缓存（cache_spot.pkl / cache_result.pkl，仅 2026-08 首两周存在）
  如有残留由 prune 清理——其结构与现行 v27 不兼容，不做迁移。
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import time
from typing import Optional

import fstore

CACHE_SUBDIR = "cache"
LEGACY_FILES = ("cache_spot.pkl", "cache_result.pkl")
CACHE_VERSION = 27  # 缓存结构版本：指标/字段变化时 +1（v27: 移除 nshape/doji 模式，结果字典去 nshape_* 键）


def file_path(out_dir: str, date: dt.date) -> str:
    return os.path.join(out_dir, CACHE_SUBDIR, f"{date:%Y%m%d}.pkl")


def _read_pkl(path: str) -> dict:
    """读缓存 pkl → dict。文件不存在 = 未缓存（静默）。

    其它失败（截断/损坏/被占用）同样按未缓存处理，但**必须留痕**：静默吞掉会让
    「为什么今天全量重拉了」无从解释，也会掩盖反复写坏缓存的真问题。
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:                                    # noqa: BLE001
        print(f"  [缓存] 读取失败，按未缓存处理（将重拉）: "
              f"{os.path.basename(path)}（{type(e).__name__}）")
        return {}
    return data if isinstance(data, dict) else {}


def _write_pkl(path: str, data: dict) -> None:
    # 原子写：此前直接覆写，中途断电/杀进程留下截断 pkl（下次 load 作废重拉，
    # 且会连带丢掉同文件里的 spot/bars）。tmp+os.replace 永无半截状态。
    fstore.write_pkl(path, data)


def load(out_dir: str, date: dt.date) -> dict:
    """读取某交易日的缓存；无文件或不可读 → {}（调用方按未缓存处理）。"""
    return _read_pkl(file_path(out_dir, date))


def is_intraday_cache(cache: dict, trade_date: dt.date) -> bool:
    """这份缓存是否写于复盘日盘中（时间戳落在复盘日当天、早于收盘缓冲）。

    盘中写的快照、结果、日K末根都不是收盘数据：有效性判定、日K 播种、
    末根重拼都以此为准（单一判据，此前散在各处各写一套）。
    """
    if not isinstance(cache, dict) or cache.get("date") != trade_date:
        return False
    ts = float(cache.get("ts", 0) or 0)
    if ts <= 0:
        return False
    from config import AFTER_CLOSE_BUFFER_MINUTES
    cached_dt = dt.datetime.fromtimestamp(ts)
    return (cached_dt.date() == trade_date
            and cached_dt.time() < dt.time(15, AFTER_CLOSE_BUFFER_MINUTES))


def save_spot(out_dir: str, date: dt.date, spot_df) -> None:
    """存当日快照。新快照落地即作废旧结果。

    results/skipped/cands/ctx 都是对着**上一份**快照算的：留着它们，分析阶段
    一旦中途崩溃（拉日K要几分钟），下一次运行会把「新时间戳 + 旧结果」当成
    完整结果直接复用——旧结果可能还是盘中的。日K bars 保留（按 code 存，与
    哪份快照无关；盘中拼的末根由 pipeline 丢弃后重拼）。
    """
    cache = load(out_dir, date)
    for key in ("results", "skipped", "cands", "ctx"):
        cache.pop(key, None)
    cache.update({"date": date, "ts": time.time(), "spot": spot_df,
                  "version": CACHE_VERSION})
    _write_pkl(file_path(out_dir, date), cache)


def save_result(out_dir: str, date: dt.date,
                results, skipped, cands, bars: Optional[dict] = None,
                ctx: Optional[dict] = None) -> None:
    cache = load(out_dir, date)
    data = {
        "date": date, "ts": time.time(),
        "results": results, "skipped": skipped, "cands": cands,
        "version": CACHE_VERSION,
    }
    if bars is not None:
        data["bars"] = bars  # 日K bars 缓存：算法升级重算时零网络复用
    elif isinstance(cache.get("bars"), dict):
        data["bars"] = cache["bars"]
    if ctx is not None:
        data["ctx"] = ctx    # 市场环境（连板梯队/晋级率/赚钱效应），复盘复用
    cache.update(data)
    _write_pkl(file_path(out_dir, date), cache)


def window_dates(calendar: list[dt.date], trade_date: dt.date,
                 keep: int = 3) -> list[dt.date]:
    """滚动窗口计算（用户口径：周内滚动，最多保留最近 keep=3 个交易日）：

    - 周一~周四：本周一至今的交易日中，取最近 keep 个（周内滚动，不跨周累积）；
      周一=[一]，周二=[一二]，周三=[一二三]，周四=[二三四]；
    - 周五：最近 keep 个交易日 = [三四五]；
    - 周末/节假日：trade_date 落在最近交易日（周五），窗口悬停不变；
    - 下周一：窗口=[周一]，上周数据全部滑出，开始新一轮。
    """
    past = sorted(d for d in calendar if d <= trade_date)
    if not past:
        return []
    wd = trade_date.weekday()  # 0=周一 ... 4=周五
    if wd == 4:
        return past[-keep:]
    monday = trade_date - dt.timedelta(days=wd)
    week_days = [d for d in past if d >= monday]
    return week_days[-keep:]


def prune(out_dir: str, calendar: list[dt.date], trade_date: dt.date,
          keep: int = 3) -> None:
    """滚动清理：只保留窗口内交易日，窗口外的缓存文件全部删除。"""
    if not calendar:
        return
    keep_set = set(window_dates(calendar, trade_date, keep=keep))
    if not keep_set:
        return

    cache_dir = os.path.join(out_dir, CACHE_SUBDIR)
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if not f.endswith(".pkl"):
                continue
            try:
                d = dt.datetime.strptime(f[:-4], "%Y%m%d").date()
            except ValueError:
                continue
            # 锚点之后的文件一律不删：trade_date 来自日历，日历陈旧（如拉取
            # 失败回退到过期的本地副本）时锚点会落后于现有缓存——当日快照
            # 不可再生，删了即永久丢失
            if d <= trade_date and d not in keep_set:
                try:
                    os.remove(os.path.join(cache_dir, f))
                    print(f"  [清理] 滚动窗口外缓存已删除: {f}")
                except OSError:
                    pass

    for f in LEGACY_FILES:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"  [清理] 旧版单文件缓存已删除: {f}")
            except OSError:
                pass
