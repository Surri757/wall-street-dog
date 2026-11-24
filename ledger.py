# -*- coding: utf-8 -*-
"""台账层（L3）：开盘买入池核账 + 可成交打板核账。

独立于渲染的数据记账：D1 登记买入价 → D2 补卖出价 → 完整轮动损益 + 累计统计。
台账幂等（按 signal_date+code upsert）；数据自己说话——口径真实成绩每日核验。

结算语义（README 待办#5，2026-09-15 落地）：
- **按预期卖出日结算**：卖出价取「买入日次一交易日」的开盘，而不是「下次
  运行日的开盘」——旧实现 17/30 笔实际跨 2 个交易日而非 T+1，跳日运行会把
  T+1 卖悄悄变成 T+N 卖。预期日的行情从滚动缓存（out/cache/*.pkl 的 spot）
  回读；停牌感知：预期日无行情则顺延到下一有行情日开盘（停牌期间本来也卖不掉）。
- **打板接成交模型**：登记晚 T+1 行情不存在，成交判定挪到 D2 结算循环
  （trade_rules.simulate_fill 同口径：T+1 一字开盘排队不成交；非一字但盘中
  触板 → 以涨停价成交）。未成交行保留用于成交率核账，不计入收益统计。
- **pnl_net**：新平仓行同时写扣双边成本净收益（calibrate 信号健康度优先读它）。
"""
from __future__ import annotations

import contextlib
import csv as _csv
import datetime as dt
import os
import re

import pandas as pd

import board_rules
import cache_store
import fstore
import termstyle as T
import trade_rules
from termstyle import pad as _pad
from config import COST_ROUND_TRIP, EXPECT, OPEN_POOL_BUY_TIER

_RULE = "─" * 78   # 与 report.RULE_W 同宽同字符，跨模块视觉一致

_LEDGER_COLS = ["signal_date", "code", "name", "tier", "buy_date", "buy_open",
                "status", "sell_date", "sell_open", "pnl", "pnl_net"]
_SLAM_COLS = ["signal_date", "code", "name", "lianban", "vr", "buy_date",
              "buy_price", "sell_date", "sell_open", "pnl",
              "fill", "fill_date", "pnl_net"]


def _f(x) -> float:
    """宽松转 float；NaN/非法 → 0.0。"""
    try:
        v = float(x)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load(path, cols):
    """读台账（utf-8-sig，按 cols 定列）；文件不存在 → []。

    ⚠ 读取失败（被占用/损坏/编码错误）**必须抛出**，不能当成空账：调用方随后
    「追加当日行 → 整体写回」，把空账写回去等于一次读故障清空全部历史——
    台账是交易证据。抛出去由 tracking_sections 兜底：打印提示、本次不写。
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [{c: r.get(c, "") for c in cols} for r in _csv.DictReader(f)]
    # 名称归一：东财把 3 字名存成「南 京 港」，历史台账 CSV 里已经有脏数据。
    # 在**读取入口**修一次即可覆盖全部渲染点，比在 8 处各打补丁可靠。
    for r in rows:
        if r.get("name"):
            r["name"] = "".join(str(r["name"]).split())
    return rows


def _save(path, cols, ledger):
    # 原子写（fstore tmp+replace）：台账是交易证据，截断=证据丢失
    fstore.write_csv(path, cols, ledger)


def _today_move(book: "_SpotBook", trade_date: dt.date, code: str, base_price) -> str:
    """「 今收+x.x%」标注：今日快照 close vs 基准价（无快照/基准价无效返回空）。

    开盘池与打板画像两段台账渲染共用（此前两处手抄同构块）。今日行直接走
    _SpotBook 已建好的索引——不再为每本账各自 iterrows 一遍 5000 行快照。
    """
    s = book.row(trade_date, str(code).zfill(6))
    px = float(s["close"]) if (s is not None
                               and float(s.get("close") or 0) > 0) else None
    try:
        base = float(base_price)
    except (TypeError, ValueError):
        return ""
    return " 今收" + format(px / base - 1, "+.1%") if (px and base > 0) else ""


def _parse_d(s) -> dt.date | None:
    try:
        return dt.datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class _SpotBook:
    """可用日快照簿：滚动缓存内各交易日 + 今日实跑 spot，按日索引懒加载。

    台账按「预期卖出日」结算需要回读历史日开盘价——数据来源即缓存里的
    spot（3 日滚动窗口天然覆盖 T+1/T+2；跳日更久则退化到最早可用日，
    sell_date 字段如实记录实际结算日）。
    """

    def __init__(self, out_dir: str, trade_date: dt.date, today_spot):
        self._raw: dict[dt.date, object] = {}
        cache_dir = os.path.join(out_dir, "cache")
        if os.path.isdir(cache_dir):
            for f in os.listdir(cache_dir):
                m = re.match(r"(\d{8})\.pkl$", f)
                if not m:
                    continue
                d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
                if d > trade_date:
                    continue
                sp = cache_store.load(out_dir, d).get("spot")
                if sp is not None:
                    self._raw[d] = sp
        if today_spot is not None:
            self._raw[trade_date] = today_spot
        self._idx: dict[dt.date, dict] = {}

    def days(self) -> list[dt.date]:
        return sorted(self._raw)

    def row(self, day: dt.date, code: str):
        """某日某股的 spot 行（无该日数据/该股当日无行情 → None；停牌自然无行）。

        索引构建用 to_dict（向量化）：5000 行快照 iterrows≈0.5-1.5s/日，
        open_pool/slam 两本账 ×N 日缓存曾在每次报告白等数秒。
        """
        if day not in self._idx:
            sp = self._raw.get(day)
            if sp is None:
                idx = {}
            else:
                codes = sp["code"].astype(str).str.strip().str.zfill(6)
                idx = dict(zip(codes, sp.to_dict("records")))
            self._idx[day] = idx
        return self._idx[day].get(str(code).zfill(6))

    def first_open_after(self, code: str, after: dt.date):
        """after 之后首个有该股开盘价的交易日 → (date, open)；停牌感知。"""
        for d in self.days():
            if d <= after:
                continue
            r = self.row(d, code)
            if r is None:
                continue
            o = _f(r.get("open"))
            if o > 0:
                return d, o
        return None

    def first_day_after(self, code: str, after: dt.date):
        """after 之后首个有快照的交易日 → (date, row|None)（row None=该日停牌）。"""
        for d in self.days():
            if d > after:
                return d, self.row(d, code)
        return None

    def prev_close_before(self, day: dt.date, code: str) -> float:
        """day 之前最近可得收盘（封板顺延判定用）。"""
        prev = 0.0
        for x in self.days():
            if x >= day:
                break
            r = self.row(x, code)
            if r is not None and _f(r.get("close")) > 0:
                prev = _f(r.get("close"))
        return prev


def _sealed_on(book: _SpotBook, day: dt.date, code: str) -> bool:
    """day 收盘是否封板（trade_rules 同口径：收盘 ≥ 涨停价−相对容差）。"""
    r = book.row(day, code)
    if r is None:
        return False
    prev = book.prev_close_before(day, code)
    if prev <= 0:
        return False
    lim = trade_rules.limit_price(prev)
    c = _f(r.get("close"))
    return c >= lim - lim * trade_rules.LIMIT_EPS


def _stats_line(rows_, label, style=None) -> list[str]:
    """单笔统计 + 按日等权净值，返回**已折行**的行列表。

    旧版把所有已平仓 pnl 无条件复利连乘，等于把**同日并发持仓当成串行全仓**，
    于是 103 笔得出「复合 −80.21%」这种无意义数字。正确口径：先按 sell_date
    等权平均得当日组合收益，再对日收益复利，并附最大回撤。
    单笔层同时给中位数——fills.csv 就是均值 +0.39% / 中位 −1.27% 的右偏反例，
    只报均值会误导。标签带交易日数，让「笔数」与「独立日数」的差别可见。

    返回多行是因为单行实测 106~126 显示列，在 78 列分隔线的报告里严重溢出。
    style 由调用方指定（次要分组传 T.dim）；旧写法 `T.dim(_stats_line(...).strip())`
    会连首行缩进一起 strip 掉，导致三组统计缩进不一致、看着像三个不同层级。
    """
    pnls = [float(r["pnl"]) for r in rows_]
    n = len(pnls)
    mean = sum(pnls) / n
    srt = sorted(pnls)
    med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    by_day: dict = {}
    for r in rows_:
        by_day.setdefault(str(r.get("sell_date") or ""), []).append(float(r["pnl"]))
    daily = [sum(v) / len(v) for v in by_day.values()]
    cum, peak, mdd = 1.0, 1.0, 0.0
    for v in daily:
        cum *= 1 + v
        peak = max(peak, cum)
        mdd = min(mdd, cum / peak - 1.0)
    txt = (f"{label} {n}笔/{len(daily)}日 均值{mean * 100:+.2f}% "
           f"中位{med * 100:+.2f}% 胜{sum(1 for v in pnls if v > 0) / n * 100:.0f}% "
           f"净值{(cum - 1) * 100:+.2f}% 回撤{mdd * 100:.1f}%")
    fn = style or (T.green if mean > 0 else T.red)
    return [fn("  " + x) for x in T.wrap(txt, T.WRAP_W - 2, "  ")]


def open_pool_tracking(out_dir: str, trade_date: dt.date,
                        spot: pd.DataFrame) -> list[str]:
    """开盘买入池核账（开→开口径）。

    台账 out/archive/open_pool_ledger.csv（幂等 upsert，按 signal_date+code）：
    - 昨日池票今日开盘登记买入价（今日 spot open；开盘一字=竞价未成交）
    - 已买入的票按**预期卖出日**（买入日次一交易日）开盘价结算 → 完整轮动损益
    - 累计统计随台账自动更新——口径真实成绩每天自己说话
    """
    lines = []
    _arch = os.path.join(out_dir, "archive")
    prev_files = (sorted(f for f in os.listdir(_arch)
                         if f.startswith("signals_") and f.endswith(".csv"))
                 if os.path.isdir(_arch) else [])
    prev_path = None
    ydf = None
    for f in reversed(prev_files):
        try:
            d = dt.datetime.strptime(f[8:16], "%Y%m%d").date()
        except ValueError:
            continue
        if d < trade_date:
            prev_path = os.path.join(out_dir, "archive", f)
            break
    if prev_path is not None:
        try:
            ydf = pd.read_csv(prev_path, dtype={"code": str})
        except Exception:
            ydf = None

    book = _SpotBook(out_dir, trade_date, spot)     # 今日行索引 + 预期卖出日回读
    ledger_path = os.path.join(out_dir, "archive", "open_pool_ledger.csv")
    # 用模块级 _LEDGER_COLS 与 _load，不要再内联一套：此前这里自带一份逐字相同的
    # cols 定义和一份 IO，既让 _LEDGER_COLS 成了死代码，也绕过了 _load 里的
    # 名称归一（实测「南 京 港」因此仍出现在报告里）
    cols = _LEDGER_COLS
    ledger = _load(ledger_path, cols)
    changed = False
    backfilled = 0
    t_str = str(trade_date)

    # 1) 卖出完成：已买入/观察、未卖 → 按**预期卖出日**（买入日次一交易日）开盘结算
    #    watch（tier1/2 观察票）照记盈亏，单独统计——将来可回答"备选档值不值得升级"
    for r in ledger:
        if (r["status"] in ("bought", "watch") and not r.get("sell_open")
                and r["buy_date"] < t_str):
            buy_d = _parse_d(r["buy_date"])
            hit = book.first_open_after(str(r["code"]).zfill(6), buy_d) if buy_d else None
            if hit is None:
                continue      # 全窗口停牌/无数据 → 仍挂起，下次运行再试
            d, so = hit
            bo = float(r["buy_open"])
            r["sell_date"] = str(d)
            r["sell_open"] = round(so, 3)
            r["pnl"] = round(so / bo - 1.0, 4)
            r["pnl_net"] = round(float(r["pnl"]) - COST_ROUND_TRIP, 4)
            if d < trade_date:
                backfilled += 1
            changed = True

    # 2) 买入登记：昨日池票今日开盘买入
    # 空 DataFrame 保护：backfill 在当日无 2 板股时仍会写出仅表头的
    # signals_*.csv，列名齐全但 iloc[0] 会 IndexError（曾可崩整份报告）
    if (ydf is not None and len(ydf)
            and all(c in ydf.columns for c in
                    ("zt_status", "vol_ratio", "open", "prev_close"))):
        sig_date = str(ydf.iloc[0].get("trade_date", ""))
        pool, _ = board_rules.open_pool_of(ydf.to_dict("records"))
        have = {(r["signal_date"], str(r["code"]).zfill(6)) for r in ledger}
        for x in pool:
            code6 = str(x["code"]).zfill(6)
            if (sig_date, code6) in have:
                continue
            s = book.row(trade_date, code6)
            if s is None or not float(s.get("open") or 0) > 0:
                continue
            bo = round(float(s["open"]), 3)
            pc = float(s.get("prev_close") or 0)
            # 一字判定走 trade_rules.limit_price（涨停价取整规则只此一份，不得复制）
            lim = trade_rules.limit_price(pc) if pc > 0 else 0.0
            one_word = lim > 0 and bo >= lim - lim * trade_rules.LIMIT_EPS
            tier_n = int(x["_otier"][0])
            status = ("skip_1z" if one_word else
                      ("bought" if tier_n <= OPEN_POOL_BUY_TIER else "watch"))
            ledger.append({"signal_date": sig_date, "code": code6,
                           "name": str(x.get("name", "")), "tier": str(tier_n),
                           "buy_date": t_str,
                           "buy_open": "" if one_word else bo,
                           "status": status,
                           "sell_date": "", "sell_open": "", "pnl": "",
                           "pnl_net": ""})
            changed = True

    if changed:
        _save(ledger_path, cols, ledger)

    # 3) 渲染
    lines += ["", "【开盘买入池核账（台账 " + str(len(ledger)) + " 行）】", _RULE]
    bought_today = [r for r in ledger if r["buy_date"] == t_str
                    and r["status"] == "bought"]
    watch_today = [r for r in ledger if r["buy_date"] == t_str
                   and r["status"] == "watch"]
    skipped_today = [r for r in ledger if r["buy_date"] == t_str
                     and r["status"] == "skip_1z"]
    sold_today = [r for r in ledger if r.get("sell_date") == t_str
                  and r.get("pnl") not in ("", None)]
    if bought_today or skipped_today:
        n_t0 = sum(1 for r in bought_today if str(r.get("tier")) == "0")
        n_lg = len(bought_today) - n_t0
        lab = "今晨开盘买入（昨日池 " + str(len(bought_today)) + " 只"
        if n_t0 and n_lg:
            lab += f"：★★★×{n_t0}，旧规则期×{n_lg}"
        elif n_t0:
            lab += "（全部★★★）"
        lab += "）："
        lines.append(lab)
        for r in bought_today:
            fl = _today_move(book, trade_date, r["code"], r["buy_open"])
            lines.append("    " + r["code"] + " " + _pad(str(r["name"]), 8)
                         + " 买" + str(r["buy_open"]) + fl)
        if skipped_today:
            # 逐票一行且每行都是同一句「竞价一字未成交」→ 合并为一行 + 名称列表
            names = "、".join(str(r["name"]) for r in skipped_today)
            lines.append("    " + T.red(f"{len(skipped_today)} 只竞价一字未成交")
                         + T.dim(f"：{names}"))
    if watch_today:
        lines.append(T.dim(f"  观察登记（不推荐买入，仅记账验证 {len(watch_today)} 只"
                           f"·逐笔见 archive/open_pool_ledger.csv）"))
    if sold_today:
        # 该策略已被五R 判死，逐笔列 34 行只会把核心输出挤成配角；
        # 压成「汇总 + 分布两端」，逐笔明细在 CSV 台账里完整保留
        ranked = sorted(((float(r["pnl"]), r) for r in sold_today),
                        key=lambda t: -t[0])
        avg = sum(p for p, _ in ranked) / len(ranked)
        win = sum(1 for p, _ in ranked if p > 0) / len(ranked)
        lines.append(f"  今晨开盘卖出（轮动完成 {len(sold_today)} 只）："
                     f"均值{avg:+.1%} 胜率{win:.0%}"
                     f"｜逐笔见 archive/open_pool_ledger.csv")

        def _one(p, r):
            tag = T.green(f"{p:+.1%}") if p > 0 else T.red(f"{p:+.1%}")
            return (f"{r['code']} {_pad(str(r['name']), 9)}"
                    f"买{r['buy_open']}→卖{r['sell_open']} {tag}")

        ends = ranked[:2] if len(ranked) <= 4 else [ranked[0], ranked[-1]]
        lines.append(T.dim("    最佳 " + _one(*ends[0])
                           + ("" if len(ends) < 2 else "  ｜ 最差 " + _one(*ends[1]))))
    if backfilled:
        lines.append(T.dim(f"  补结 {backfilled} 笔（跳日运行：按预期 T+1 开盘价结算，"
                           f"实际结算日见台账 sell_date 列）"))
    closed = [r for r in ledger if r.get("pnl") not in ("", None)]
    buy_closed = [r for r in closed if r.get("tier") == "0"]
    watch_closed = [r for r in closed if r.get("tier") in ("1", "2")]
    legacy_closed = [r for r in closed if r.get("tier") not in ("0", "1", "2")]
    if buy_closed:
        lines += _stats_line(buy_closed, "★★★档")
    if watch_closed:
        lines += _stats_line(watch_closed, "观察档", style=T.dim)
    if legacy_closed:
        lines += _stats_line(legacy_closed, "旧规则期", style=T.dim)
    if not (bought_today or skipped_today or watch_today or sold_today):
        lines.append(T.dim("  （今日无池核账动作——昨日归档为旧格式或无池票）"))

    # 注：ydf 仍被上方开盘池登记逻辑使用，故保留其加载。
    return lines




def slam_ledger(out_dir: str, trade_date: dt.date, results: list,
                spot: pd.DataFrame) -> list[str]:
    """可成交打板台账（2026-09-15 起接成交模型，README 待办#5）。

    画像：封板 + 实体开盘(<+3%) + vr0.4~0.8（board_rules.slam_candidates 单一实现）。
    生命周期（与 trade_rules.simulate_fill/simulate_exit 及 fills_v2 回放同口径）：
      D1（信号日晚）登记：买入价 = 今日涨停价（=收盘），fill 待判定
      D2（次一交易日）判定：开盘一字 → 排队**不成交**（fill=0 终态）；
                            非一字但盘中触板 → 以涨停价成交（fill=1）
      D3（再次一交易日）卖出：开盘卖；若 D3 收盘仍封板 → 顺延一日开盘卖（R1，一次为限）
    未成交行保留：成交率要跟 config.EXPECT["slam"]["fill_rate"] 对账，
    但不进收益统计（没发生的交易没有损益）。
    """
    lines = []
    path = os.path.join(out_dir, "archive", "slam_ledger.csv")
    ledger = _load(path, _SLAM_COLS)
    changed = False
    backfilled = 0
    t_str = str(trade_date)
    book = _SpotBook(out_dir, trade_date, spot)

    for r in ledger:
        if r.get("sell_open"):
            continue                      # 已结（旧语义平仓行原样保留）
        sig_d = _parse_d(r["signal_date"])
        code = str(r["code"]).zfill(6)
        if sig_d is None or sig_d >= trade_date:
            continue
        # ① 成交判定（登记当晚 T+1 还不存在，判定挪到本循环）
        if r.get("fill") in ("", None):
            hit = book.first_day_after(code, sig_d)
            if hit is None:
                continue                  # 窗口内无可判日（缓存滚走）→ 挂起
            jd, jrow = hit
            buy = _f(r.get("buy_price"))
            lim = trade_rules.limit_price(buy) if buy > 0 else 0.0
            if jrow is None or lim <= 0:
                r["fill"] = 0             # 判定日停牌：排板单当日无成交机会
            else:
                jo = _f(jrow.get("open"))
                jh = _f(jrow.get("high"))
                one_word = jo > 0 and jo >= lim - lim * trade_rules.LIMIT_EPS
                touched = one_word or jh >= lim - lim * trade_rules.LIMIT_EPS
                r["fill"] = int(bool(touched and not one_word))
            r["fill_date"] = str(jd)
            changed = True
        if str(r["fill"]) != "1":
            continue                      # 未成交：终态，只进成交率统计
        # ② 卖出：判定日次一交易日开盘（R1：卖出日收盘仍封板则顺延一次）
        fill_d = _parse_d(r["fill_date"]) or sig_d
        hit = book.first_open_after(code, fill_d)
        if hit is None:
            continue
        d, so = hit
        offset = sum(1 for x in book.days() if sig_d < x <= d)
        nxt = book.first_open_after(code, d)
        # R1 顺延窗口上限走 trade_rules.MAX_HOLD_DAYS 单一源（此前手抄 3，
        # trade_rules 调整时台账不会跟随——docstring 自称同口径却靠人肉同步）
        if (nxt is not None and offset < trade_rules.MAX_HOLD_DAYS
                and _sealed_on(book, d, code)):
            d, so = nxt                   # R1 封板持有，顺延一日开盘卖（一次为限）
        r["sell_date"] = str(d)
        r["sell_open"] = round(so, 3)
        r["pnl"] = round(so / float(r["buy_price"]) - 1.0, 4)
        r["pnl_net"] = round(float(r["pnl"]) - COST_ROUND_TRIP, 4)
        if d < trade_date:
            backfilled += 1
        changed = True

    # 登记：今日画像票（买入价=今日涨停价=close，fill 明晚判定）
    slam = board_rules.slam_candidates(results)
    have = {(r["signal_date"], str(r["code"]).zfill(6)) for r in ledger}
    for x in slam:
        code6 = str(x["code"]).zfill(6)
        if (t_str, code6) in have:
            continue
        ledger.append({"signal_date": t_str, "code": code6,
                       "name": str(x.get("name", "")),
                       "lianban": int(x.get("lianban") or 0),
                       "vr": float(x["vol_ratio"]), "buy_date": t_str,
                       "buy_price": round(float(x.get("close") or 0), 3),
                       "sell_date": "", "sell_open": "", "pnl": "",
                       "fill": "", "fill_date": "", "pnl_net": ""})
        changed = True

    if changed:
        _save(path, _SLAM_COLS, ledger)

    # 渲染：昨日判定 + 昨日打板今卖成绩 + 持仓中 + 累计
    judged_today = [r for r in ledger if r.get("fill_date") == t_str
                    and str(r.get("fill")) in ("0", "1")]
    filled_today = [r for r in judged_today if str(r.get("fill")) == "1"]
    missed_today = [r for r in judged_today if str(r.get("fill")) == "0"]
    sold_today = [r for r in ledger if r.get("sell_date") == t_str
                  and r.get("pnl") not in ("", None)]
    holding = [r for r in ledger if r["signal_date"] == t_str]
    judged_all = [r for r in ledger if str(r.get("fill")) in ("0", "1")]
    if sold_today or holding or judged_today:
        lines += ["", "【打板台账核账】", _RULE]
        if judged_today:
            rate = (sum(1 for r in judged_all if str(r.get("fill")) == "1")
                    / len(judged_all)) if judged_all else 0.0
            exp_rate = EXPECT["slam"].get("fill_rate")
            exp_s = f"·史测 {exp_rate:.0%}" if isinstance(exp_rate, float) else ""
            lines.append(f"  昨日画像·今晨判定：成交 {len(filled_today)} 只"
                         f" / 未成交 {len(missed_today)} 只"
                         f"（窗口成交率 {rate:.0%}{exp_s}，一字排队不成交是常态）")
            if missed_today:
                names = "、".join(str(r["name"]) for r in missed_today[:6])
                lines.append(T.dim(f"    未成交：{names}"
                                   + (f" 等{len(missed_today)}只"
                                      if len(missed_today) > 6 else "")))
        if sold_today:
            lines.append("  已打板·卖出（完整轮动）：")
            for r in sold_today:
                pnl = float(r["pnl"])
                tag = (T.green(format(pnl, "+.1%")) if pnl > 0
                       else T.red(format(pnl, "+.1%")))
                lb = r.get("lianban", "")
                lines.append("    " + r["code"] + " " + _pad(str(r["name"]), 8)
                             + f" {lb}板 打{r['buy_price']}→卖{r['sell_open']} " + tag)
        if holding:
            lines.append(f"  今日画像·登记 {len(holding)} 只（买入价=今日涨停价，"
                         f"明晚按 T+1 行情判定成交，成交后大后天开盘卖核收益）：")
            for r in holding[:6]:
                fl = _today_move(book, trade_date, r["code"], r["buy_price"])
                lines.append("    " + r["code"] + " " + _pad(str(r["name"]), 8)
                             + f" {r.get('lianban', '')}板 打{r['buy_price']}" + fl)
            if len(holding) > 6:
                lines.append(f"    …另有 {len(holding) - 6} 只")
        closed = [r for r in ledger if r.get("pnl") not in ("", None)]
        if closed:
            lines += _stats_line(closed, "打板画像")
        if backfilled:
            lines.append(T.dim(f"  补结 {backfilled} 笔（跳日运行：按预期卖出日开盘价结算，"
                               f"实际结算日见台账 sell_date 列）"))
    return lines


def tracking_sections(out_dir: str, trade_date: dt.date, spot,
                      results: list) -> list[str]:
    """报告「台账核账区」的唯一入口：更新两本账并返回渲染行。

    - 两本账都是「读→改→写」，整体持 run_lock（与 pipeline.run_day 同一把进程
      互斥锁）：定时任务与手动运行撞车时不会互相覆盖对方的行；
    - 锁被占用 → 跳过并留一行提示；单本账异常 → 打印提示、返回一行占位，报告
      其余部分照常。台账按 signal_date+code 幂等 upsert，本次没记上的下次运行
      自动补。此前台账更新排在报告落盘之前且无任何保护：一次 PermissionError
      （例如台账 CSV 正被表格软件打开）就会让整份报告都写不出来。
    """
    jobs = (("开盘买入池", lambda: open_pool_tracking(out_dir, trade_date, spot)),
            ("打板", lambda: slam_ledger(out_dir, trade_date, spot=spot,
                                         results=results)))
    lines: list[str] = []
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(fstore.run_lock(
                os.path.join(out_dir, cache_store.CACHE_SUBDIR)))
        except RuntimeError as e:
            print(f"  [提示] 台账核账跳过（{e}）")
            return [T.dim("  （台账核账跳过：另一复盘进程正在运行，下次运行自动补）")]
        for name, job in jobs:
            try:
                lines += job()
            except Exception as e:                                 # noqa: BLE001
                print(f"  [提示] {name}台账核账失败（不影响本次报告；该账本次未写入，"
                      f"下次运行自动补）: {type(e).__name__}: {e}")
                lines.append(T.dim(f"  （{name}台账本次核账失败，详见运行提示）"))
    return lines


