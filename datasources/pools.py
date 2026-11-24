# -*- coding: utf-8 -*-
"""东财涨停/炸板池：双格式解析（新dict/旧数组）、连板/封单/行业/流通市值、
昨日池重建（首板口径/晋级率/赚钱效应原料）。"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import EXCLUDE_ST, MAIN_BOARD_PREFIXES
from .http import (EM_HEADERS, EM_POOL_BASE, EmptyDataError, _http_get_json,
                   _SESSION)
from .util import num as _num          # 兼容别名：模块内历史调用点
from .util import to_int as _to_int, to_secid


# ---------------------------------------------------------------- 涨跌停池解析

# 字段映射：dict 键（新版接口，2026-08 起为字段名字典）与位置索引（旧版数组）。
# 换手率统一换算为小数；价格 /1000 转元；封板时间 HHMMSS → "HHMM"。
_POOL_SPEC = {
    "zt": {  # 涨停池
        "name": (["n"], 2), "close": (["p"], 3), "pct": (["zdp"], 4),
        "amount": (["amount"], 5), "turnover": (["hs"], 8),
        "lianban": (["lbc"], 9), "fbt": (["fbt"], 10),
        "fund": (["fund"], 12), "industry": (["hybk"], 14),
        "ltsz": (["ltsz"], None),   # 流通市值（元）；仅新版 dict 格式提供
        "zttj": (["zttj"], None),   # 「N天M板」统计；仅新版 dict 格式提供
    },
    "zb": {  # 炸板池
        "name": (["n"], 2), "close": (["p"], 3), "ztp": (["ztp"], 4),
        "pct": (["zdp"], 5), "amount": (["amount"], 6), "turnover": (["hs"], 9),
        "zhaban": (["zbc"], 11), "amp": (["zf"], 12),
        "zttj": (["zttj"], 14), "industry": (["hybk"], 15),
    },
}


def _fmt_fbt(v) -> str:
    """封板时间（HHMMSS 整数/字符串）→ "HHMM"；空/非法返回空串。"""
    if v is None:
        return ""
    try:
        s = str(int(v)) if float(v) == float(v) else str(v).strip()
    except (TypeError, ValueError):
        s = str(v).strip()
    return s.zfill(6)[:4]


def norm_pool_row(p, kind: str) -> Optional[dict]:
    """单行池数据 → 统一字段 dict；无法识别返回 None（新 dict / 旧位置数组都支持）。"""
    spec = _POOL_SPEC[kind]
    if isinstance(p, dict):
        code_raw = str(p.get("c", "")).strip()

        def g(field):
            for k in spec[field][0]:
                if k in p:
                    return p[k]
            return None
    elif isinstance(p, (list, tuple)) and p:
        code_raw = str(p[0]).strip()

        def g(field):
            idx = spec[field][1]
            return p[idx] if idx is not None and len(p) > idx else None
    else:
        return None
    if not code_raw:
        return None
    code = code_raw.zfill(6)
    if len(code) != 6 or not code.isdigit():
        return None

    # 连板数只认涨停池的 lbc（连续封板数）。「N天M板」统计 zttj 不是连板数：
    # days=天数、ct=板数（实测北纬科技 8天5板 → days=8）。此前取
    # max(lbc, zttj.days)——涨停池 lbc 才是连续板数（大东方 lbc=1、14天7板 →
    # 被记成 14）；炸板池没有 lbc、直接落到 days（归档 107 条炸板行
    # lianban≥2，其中 21 条高于当日市场高度）。炸板行连板数留 0，由
    # analyzer 用日K近似；N天M板原样透传给研究用。
    lianban = _to_int(g("lianban")) if "lianban" in spec else 0
    zttj = g("zttj") if "zttj" in spec else None
    zttj = zttj if isinstance(zttj, dict) else {}
    return {
        "code": code,
        "name": str(g("name") or ""),
        "close": _num(g("close")) / 1000.0,
        "pct": _num(g("pct")),
        "amount": _num(g("amount")),
        "turnover": _num(g("turnover")) / 100.0,
        "lianban": lianban,
        "zttj_days": _to_int(zttj.get("days", 0)),   # 「N天M板」的天数（非连板数）
        "zttj_ct": _to_int(zttj.get("ct", 0)),       # 「N天M板」的板数（非连续）
        "fbt": _fmt_fbt(g("fbt")) if "fbt" in spec else "",
        "zhaban": _to_int(g("zhaban")) if "zhaban" in spec else 0,
        "fund": _num(g("fund")) if "fund" in spec else float("nan"),
        "ltsz": _num(g("ltsz")) if "ltsz" in spec else float("nan"),
        "amp": _num(g("amp")) if "amp" in spec else float("nan"),
        "industry": str(g("industry") or "").strip() if "industry" in spec else "",
    }


def parse_pool(kind: str, raw) -> list[dict]:
    """整池解析；返回非 0 行说明格式兼容。有数据但 0 行解析时大声告警（防静默降级）。

    部分行解析失败同样计数告警：接口格式半变（如一半行缺 c 键）时若静默丢弃，
    连板数/行业家数/晋级率会悄悄失真——这正是本项目要防的静默降级。
    """
    rows = []
    dropped = 0
    for p in raw or []:
        try:
            nr = norm_pool_row(p, kind)
        except Exception:
            nr = None
        if nr:
            rows.append(nr)
        else:
            dropped += 1
    if raw and dropped and not rows:
        print(f"  [错误] {kind}池返回 {len(raw)} 行、成功解析 0 行：接口格式可能已变化，"
              f"请检查 _POOL_SPEC 的键映射！本次按「池数据缺失」中性分处理。")
    elif raw and dropped:
        print(f"  [警告] {kind}池 {dropped}/{len(raw)} 行解析失败被丢弃，"
              f"统计可能偏少（接口格式半变？）")
    return rows


def build_candidates_from_pools(zt_pool, zb_pool):
    """历史补录：由东财历史涨停/炸板池原始行重建候选清单（跌停/振幅池历史无法重建）。

    返回 (cands, extras)。cands 带 fill=True 与 source="backfill" 标记
    （后者是 report 归档的来源血缘，calibrate 据此分层）。
    兼容新 dict / 旧位置数组两种行格式。
    """
    cands: dict[str, dict] = {}
    extras: dict[str, dict] = {}

    def slot(code: str) -> dict:
        # source=backfill：与 backfill.py 同一血缘标记——这类行只有池字段、没有
        # 昨收/振幅，calibrate 按 backfill/live 分层时不能把它当成实跑样本
        # （此前缺这一项，report 兜底记成 live，补录日混进了实跑层）
        return cands.setdefault(code, {
            "code": code, "name": "", "pools": [],
            "zt_status": "", "dt_status": "",
            "close": float("nan"), "prev_close": float("nan"),
            "high": float("nan"), "low": float("nan"),
            "amp": float("nan"), "pct": float("nan"),
            "amount": float("nan"), "turnover": float("nan"),
            "fill": True, "source": "backfill",
        })

    def _put(c: dict, row: dict) -> None:
        for k in ("close", "pct", "amount", "turnover"):
            v = row[k]
            if v == v:  # NaN 不覆盖
                c[k] = v
        v = row["amp"]
        if v == v:
            c["amp"] = v

    for row in parse_pool("zt", zt_pool):
        if not row["code"].startswith(MAIN_BOARD_PREFIXES):
            continue
        if EXCLUDE_ST and "ST" in row["name"].upper():
            continue
        c = slot(row["code"])
        c["name"] = row["name"]
        if "涨停" not in c["pools"]:
            c["pools"].append("涨停")
        c["zt_status"] = "封板"  # 涨停池=收盘封板
        _put(c, row)
        ex = extras.setdefault(row["code"], {})
        ex["lianban"] = row["lianban"]
        if row["fbt"]:
            ex["fbt"] = row["fbt"]
        if row["industry"]:
            ex["industry"] = row["industry"]
        if row["fund"] == row["fund"]:
            ex["fund"] = row["fund"]
        if row["ltsz"] == row["ltsz"]:
            ex["ltsz"] = row["ltsz"]

    for row in parse_pool("zb", zb_pool):
        if not row["code"].startswith(MAIN_BOARD_PREFIXES):
            continue
        if EXCLUDE_ST and "ST" in row["name"].upper():
            continue
        c = slot(row["code"])
        c["name"] = row["name"]
        if "涨停" not in c["pools"]:
            c["pools"].append("涨停")
        if c["zt_status"] != "封板":   # 涨停池成员=收盘封板；炸板池不得改写它
            c["zt_status"] = "炸板"
        _put(c, row)
        ex = extras.setdefault(row["code"], {})
        ex["zhaban_times"] = row["zhaban"]     # 炸板池无 lbc，不产出连板数

    out = list(cands.values())
    return (out, extras) if out else (None, None)


def fetch_history_pool_candidates(trade_date: dt.date):
    """拉取东财历史涨停/炸板池并重建候选（支持历史日期，免费）。

    经 pool_raw 取数：与 mood/vane 对同一日期的请求共享 _POOL_MEMO（此前自带
    一份相同参数的请求实现且绕过 memo，pipeline 两条分支对同日池会重复拉网）。
    涨停池是补录的主体：它失败即判补录失败（返回 (None, None)）——此前
    「两池都失败才算失败」，涨停池挂掉时会产出一份零封板的结果并存成完整
    缓存。炸板池只补充炸板次数，失败可以容忍。
    """
    d = trade_date.strftime("%Y%m%d")
    zt = pool_raw("zt", d)
    if zt is None:
        print(f"  [提示] {trade_date} 涨停池不可用，历史补录放弃（不落零封板结果）")
        return None, None
    return build_candidates_from_pools(zt, pool_raw("zb", d))


def fetch_industry_em(code: str) -> str:
    """东财个股资料接口补行业（f127），单只、6 秒超时；失败返回空串。"""
    secid = to_secid(code)
    params = {"secid": secid, "fields": "f127"}
    headers = dict(EM_HEADERS, Referer=f"https://quote.eastmoney.com/{code}.html")
    try:
        r = _SESSION.get("https://push2.eastmoney.com/api/qt/stock/get",
                         params=params, headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json().get("data") or {}
        return str(data.get("f127") or "").strip()
    except Exception:
        return ""


def fill_industry_local(cands: list[dict], extras: dict[str, dict],
                        ctx: dict) -> None:
    """涨停池缺失时：对封板候选拉东财 f127 补行业，并本地重建行业涨停家数。

    并行拉取（4 线程）：池退化的异常日封板 50-100 只，串行 6s 超时/只最坏
    5 分钟——恰是最需要快出报告的时候。结果照旧写入 extras（单只失败容错）。
    """
    if ctx.get("industry_count"):
        return  # 池数据已有，无需补
    print("  [补全] 涨停池行业缺失，尝试并行拉东财个股行业（f127）...")
    industry_count: dict[str, int] = {}
    sealed = [c for c in cands if c.get("zt_status") == "封板"]

    def _ind(c: dict):
        return c["code"], fetch_industry_em(c["code"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = [pair for pair in pool.map(_ind, sealed) if pair[1]]
    for code, ind in got:
        extras.setdefault(code, {})["industry"] = ind
        industry_count[ind] = industry_count.get(ind, 0) + 1
    ok = len(got)
    if ok:
        ctx["industry_count"] = industry_count
        print(f"  [补全] 行业补全 {ok}/{len(sealed)} 只封板股")
    else:
        print("  [提示] 个股行业接口不可用，题材地位维持中性分")


_POOL_MEMO: dict = {}   # 进程内池缓存：同一 (kind, date) 只拉一次。
                        # mood 序列/fetch_pool_extras/backfill 会重复请求相同日期的池。


def pool_raw(kind: str, date: str):
    key = (kind, date)
    if key in _POOL_MEMO:
        return _POOL_MEMO[key]
    paths = {"zt": "/getTopicZTPool", "zb": "/getTopicZBPool"}
    params = {
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "dpt": "wz.ztzt",
        "Pageindex": "0",
        "pagesize": "10000",
        "sort": "fbt:asc",
        "date": date,
    }
    try:
        data = _http_get_json(EM_POOL_BASE + paths[kind], params,
                              referer="https://quote.eastmoney.com/ztb/detail",
                              timeout=6)
        pool = data.get("pool") or []
        result = pool if isinstance(pool, list) else None
    except EmptyDataError:
        # 接口明确答复无该日数据（非交易日 / 超出保留期）：预期内，不刷屏
        result = None
    except Exception as e:                       # noqa: BLE001 真故障要留痕
        result = None
        print(f"  [提示] {kind}池 {date} 拉取失败（{type(e).__name__}），"
              f"该日按缺池处理")
    # 失败(None)不进 memo：瞬时断连若被进程级永久缓存，mood 多日循环中该日
    # 会永远判空且无重试机会；空池([]) 是合法交易日结果，照常缓存。
    if result is not None:
        _POOL_MEMO[key] = result
    return result


def _prev_trade_date(trade_date: dt.date,
                     calendar: Optional[list]) -> Optional[dt.date]:
    """上一个交易日：优先用交易日历；无日历时向前试探涨停池（非交易日返回空）。"""
    if calendar:
        past = [d for d in calendar if d < trade_date]
        if past:
            return past[-1]
    for back in range(1, 6):
        cand = trade_date - dt.timedelta(days=back)
        if pool_raw("zt", cand.strftime("%Y%m%d")):
            return cand
    return None


def fetch_pool_extras(trade_date: dt.date,
                      calendar: Optional[list[dt.date]] = None):
    """从东财涨停/炸板/跌停池补充连板数、封板时间、行业等（自建直连，6秒快速失败）。

    返回 (extras, ctx)：
      extras[code] = {lianban, zhaban_times, lian_dieting, fbt, industry}
      ctx = {industry_count, industry_max_lianban, yesterday_zt_count,
             today_2plus_count, promotion_rate, yesterday_fbt, yesterday_turnover}
    晋级率 = 主板今日2板+数 / 主板昨日封板数（原「昨日涨停池」接口已停用，
    改用昨日日期的涨停池重建，同时取回首板封板时间/昨日换手供二板评分用）。
    任一池失败只影响对应附加信息，不拖慢主流程。
    """
    d = trade_date.strftime("%Y%m%d")
    extra: dict[str, dict] = {}
    industry_count: dict[str, int] = {}
    industry_max: dict[str, int] = {}
    today_2plus = 0
    ladder: dict[int, int] = {}
    ind_mb: dict[str, int] = {}   # 今日主板行业涨停家数（持续题材判定用）

    zt_rows = parse_pool("zt", pool_raw("zt", d))
    if zt_rows:
        for row in zt_rows:
            code = row["code"]
            slot = extra.setdefault(code, {})
            slot["lianban"] = row["lianban"]
            slot["fbt"] = row["fbt"]
            slot["industry"] = row["industry"]
            if row["fund"] == row["fund"]:
                slot["fund"] = row["fund"]   # 封单额（封成比分子）
            if row["ltsz"] == row["ltsz"]:
                slot["ltsz"] = row["ltsz"]   # 流通市值（元）
            ind = row["industry"]
            if ind:  # 题材热度按全市场口径数家数
                industry_count[ind] = industry_count.get(ind, 0) + 1
                industry_max[ind] = max(industry_max.get(ind, 0), row["lianban"])
            if code.startswith(MAIN_BOARD_PREFIXES):
                if row["industry"]:
                    ind_mb[row["industry"]] = ind_mb.get(row["industry"], 0) + 1
                if row["lianban"] >= 2:
                    today_2plus += 1
                    ladder[row["lianban"]] = ladder.get(row["lianban"], 0) + 1
    else:
        print("  [提示] 涨停池拉取失败或为空（连板/行业数据缺失，评分中性化）")

    zb_rows = parse_pool("zb", pool_raw("zb", d))
    for row in zb_rows:
        slot = extra.setdefault(row["code"], {})
        slot["zhaban_times"] = row["zhaban"]     # 炸板池无 lbc，不产出连板数

    yesterday_fbt: dict[str, str] = {}
    yesterday_turnover: dict[str, float] = {}
    yesterday_seal: dict[str, float] = {}   # 昨日封成比（首板封单强度）
    yesterday_codes: list[str] = []
    yesterday_zt = None
    industry_yesterday: dict[str, int] = {}  # 昨日行业涨停家数（题材发酵度）
    ind_y_mb: dict[str, int] = {}            # 昨日主板行业家数（持续题材判定用）
    prev = _prev_trade_date(trade_date, calendar)
    if prev is not None:
        yrows = parse_pool("zt", pool_raw("zt", prev.strftime("%Y%m%d")))
        if yrows:
            mb_rows = [r for r in yrows
                       if r["code"].startswith(MAIN_BOARD_PREFIXES)]
            yesterday_zt = len(mb_rows)
            yesterday_codes = [r["code"] for r in mb_rows]
            for r in mb_rows:
                if r["industry"]:
                    ind_y_mb[r["industry"]] = ind_y_mb.get(r["industry"], 0) + 1
            for r in yrows:
                if r["fbt"]:
                    yesterday_fbt[r["code"]] = r["fbt"]
                if r["turnover"] == r["turnover"]:
                    yesterday_turnover[r["code"]] = r["turnover"]
                if r["fund"] == r["fund"] and r["amount"] > 0:
                    yesterday_seal[r["code"]] = round(r["fund"] / r["amount"], 2)
                if r["industry"]:
                    industry_yesterday[r["industry"]] = \
                        industry_yesterday.get(r["industry"], 0) + 1

    ctx = {
        "industry_count": industry_count,
        "industry_max_lianban": industry_max,
        "industry_yesterday_count": industry_yesterday,
        "today_2plus_count": today_2plus or None,
        "yesterday_zt_count": yesterday_zt,
        "promotion_rate": (today_2plus / yesterday_zt)
                          if (today_2plus and yesterday_zt) else None,
        "yesterday_fbt": yesterday_fbt,
        "yesterday_turnover": yesterday_turnover,
        "yesterday_seal": yesterday_seal,
        "persistent_industries": {k for k, v in ind_mb.items()
                                  if v >= 2 and ind_y_mb.get(k, 0) >= 2} or None,
        "market_max_lianban": max(ladder) if ladder else None,  # 市场连板高度
        "lianban_ladder": dict(sorted(ladder.items(), reverse=True)) or None,
        "_yzt_mb_codes": yesterday_codes,  # 供上层结合今日快照算赚钱效应
        "zt_pool_ok": bool(zt_rows),   # 二板模块：池数据可用性（缺失时对应项中性分）
        "zb_pool_ok": bool(zb_rows),
    }
    return extra, ctx


# ---- 兼容别名（历史私有名，外部旧调用点平滑过渡；新代码请用公开名）----
_norm_pool_row = norm_pool_row
_parse_pool = parse_pool
_pool_raw = pool_raw
