# -*- coding: utf-8 -*-
"""报告输出：市场情绪速览 + 二板定龙头候选池。

返回 (txt 路径, csv 路径, 彩色文本)。落盘一律纯文本，彩色版只在返回值里——
调用方要终端着色就直接打印第三项，不要读回文件。
txt = 人看的信号清单（二板名单为一等区块；--all 追加全部候选）
csv = 机器可读全量（含所有分析成功的候选，供归档与事后胜率校准）
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from archive_schema import SIGNALS_CSV_COLUMNS
from config import (AMP_THRESHOLD, BOARD_SCORE_MIN, EXPECT, MIN_BARS,
                    ONE_WORD_GAP, OPEN_POOL_SHOW, OPEN_POOL_VR_MIN, SLAM_SHOW,
                    WATCHLIST_SHOW)
import board_rules
import fstore
import ledger
import termstyle as T
import trade_rules
from termstyle import pad as _pad

BOX_W = 76       # 表头框线内宽（按**显示列数**算，中文占 2 列）
RULE_W = 78      # 区块分隔线宽度；统一用 ─（U+2500），不要混 ASCII 连字符

CSV_COLUMNS = SIGNALS_CSV_COLUMNS   # 归档列的单一源在 archive_schema（backfill 同用）


def _passed(r: dict) -> bool:
    """通过口径：二板定龙头（2连板收盘封板 + 五维评分 ≥ config.BOARD_SCORE_MIN）。"""
    return bool(r.get("board_pass"))


def emotion_summary(cands: list[dict], ctx: dict | None = None) -> dict:
    ctx = ctx or {}
    up = [c for c in cands if "涨停" in c["pools"]]
    down = [c for c in cands if "跌停" in c["pools"]]
    sealed = sum(1 for c in up if c["zt_status"] == "封板")
    broken = len(up) - sealed
    sealed_down = sum(1 for c in down if c["dt_status"] == "封死")
    qiaoban = len(down) - sealed_down
    amp = sum(1 for c in cands if c["amp"] >= AMP_THRESHOLD - 1e-6)
    rets = ctx.get("yzt_today_pct") or []
    return {
        "up_total": len(up), "up_sealed": sealed, "up_broken": broken,
        "break_rate": broken / len(up) * 100 if up else 0.0,
        "down_total": len(down), "down_sealed": sealed_down, "down_qiaoban": qiaoban,
        "amp_total": amp, "cand_total": len(cands),
        "height": ctx.get("market_max_lianban") or 0,      # 市场连板高度
        "ladder": ctx.get("lianban_ladder") or {},          # 主板连板梯队
        "yzt_ret": (sum(rets) / len(rets)) if rets else None,   # 昨涨停今均价
        "yzt_n": len(rets),
        "yzt_win": sum(1 for v in rets if v > 0),               # 昨涨停今红盘数
    }


def _tag_pools(r: dict) -> str:
    tags = []
    if "涨停" in r["pools"]:
        t = r["zt_status"]
        if r.get("lianban"):
            t += f"·连板{r['lianban']}"
        if r.get("zhaban_times"):
            t += f"·炸板{r['zhaban_times']}次"
        tags.append(t)
    if "跌停" in r["pools"]:
        tags.append(r["dt_status"])
    if "振幅" in r["pools"]:
        tags.append(f"振幅{r['amp']:.1f}%")
    return " ".join(tags)


def _entry_flag(r: dict) -> tuple[str, str]:
    """次日可成交性标记 → (着色文本, 分类键 none/hard/ok)。

    今日一字 = 明日必然一字开盘，买不到；准一字 = 明日大概率排队不成。
    依据 events.csv(2999事件/241交易日)：gap_next>=0.09 与 one_word_next 是同一
    集合(n=684)，该组晋级率 78% 但可成交口径开→收 -2.11%（聚类 t=-9.22）——
    信息很强，钱赚不到。这是「信息口径」与「行动口径」的分界线。
    """
    note = r.get("board_note") or ""
    if "二板:一字" in note:
        return T.red("明日买不到"), "none"
    if "二板:准一字" in note:
        return T.yellow("明日难成交"), "hard"
    return T.green("明日可买"), "ok"


def _line(r: dict) -> str:
    """主行：评级星 + 评分 + 代码/名称/池标签 + 次日可成交性标记。"""
    score = r.get("board_score", 0)
    if r.get("board_pass"):
        # board_score 是五维序数加权和，不是概率——不能印成 %。
        # 实测 80-100 分桶红盘率 61.9%、晋级率 47.6%（bucket_stats.json），
        # 与「82%」这种读法差距很大；同档实测数字由 _detail 行给出。
        prob = _pad(f"{T.stars(score)} {score}分", 13)
    else:
        prob = " " * 13
    entry = _entry_flag(r)[0]
    name = T.bold(_pad(r["name"], 10))
    pool = _pad(_tag_pools(r), 14)
    sr = r.get("seal_ratio")
    seal = (f" {T.cyan(f'封{sr:.1f}')}") if (isinstance(sr, (int, float)) and sr == sr and sr >= 0.5) else ""
    return f"{prob} {r['code']}  {name} {pool} [{entry}]{seal}"


def _bucket_of(score) -> str:
    s = score if isinstance(score, (int, float)) else 0
    for lo, hi in ((0, 59), (60, 69), (70, 79), (80, 100)):
        if lo <= s <= hi:
            return f"{lo}-{hi}"
    return ""


def _dim_colored(note: str) -> str:
    """维度标签着色：正向关键词绿、风险词红/黄、其余默认。"""
    t = note
    for w, fn in (("发酵", T.green), ("最高板", T.green), ("新题材", T.green),
                  ("加速", T.green), ("一字", T.cyan), ("准一字", T.cyan),
                  ("缩量板", T.cyan), ("退潮", T.red), ("炸", T.yellow),
                  ("亏钱", T.red), ("死亡换手", T.red)):
        t = t.replace(w, fn(w))
    return t


DETAIL_W = T.WRAP_W                # 明细行目标显示宽度（分隔线 78，明细略宽但不破百）
DETAIL_PREFIX = "         └ "      # 明细首行前缀
DETAIL_INDENT = " " * 14           # 续行缩进，对齐到 └ 之后

# 折行实现统一在 termstyle（report 依赖 ledger，ledger 不能反向依赖 report，
# 两者共用同一套折行 → 只能放在展示层工具模块）。此处起别名保持调用点简短。
_wrap = T.wrap
_wrap_segs = T.wrap_segs
_dim_wrap = T.dim_wrap


def _detail(r: dict, stats: dict | None = None) -> list[str]:
    """明细行（可多行）：维度标签 + 同档实测。仅过线股输出，无内容返回 []。

    **不印「情绪」段**（board_note 的第 4 段）：D 维是市场级常量，同日所有票取值
    完全相同（五T 已证其横截面区分度恒为 0），逐票重印纯属重复，且「市场情绪速览」
    区块已经把炸板率/晋级率/赚钱效应/周期印过一遍——每票省下约 44 个显示列。
    """
    if not r.get("board_pass") or not r.get("board_note"):
        return []
    parts = [p.strip() for p in r["board_note"].split("|")]
    segs = [T.dim(p) for p in parts[:2]]              # 题材 / 首板
    if len(parts) > 2:
        segs.append(_dim_colored(parts[2]))           # 二板（形态关键词着色）
    # parts[3] = 情绪，市场级常量，刻意跳过（见 docstring）
    if len(parts) > 4:
        segs.append(parts[4])                         # 辨识
    stats = stats or {}
    b = stats.get(_bucket_of(r.get("board_score")))
    if b:
        # 两种执行路径的收益必须分开印，它们对应完全不同的动作：
        #   排板隔夜 = open（次日开盘 vs 信号日收盘）→ T 日挂涨停价排板**且成交**才有；
        #              实测排板成交率仅 30.7%（fills.csv 1414 事件）
        #   追买净   = net（次日开盘买→收盘卖，扣双边成本）→ T+1 开盘能买到才有，
        #              而一字/准一字票买不到
        # 此前只印 open，等于默认用户能排板成交。
        days = b.get("days")
        head = T.magenta(f"同档(n={b['n']}" + (f"·{days}日" if days else "") + ")")
        gap = (T.green(f"排板隔夜{b['open']:+.1f}%") if b["open"] >= 0
               else T.red(f"排板隔夜{b['open']:+.1f}%"))
        net = b.get("net")
        if net is None:
            net_s = T.dim(" 追买净待重算")   # 旧 bucket_stats，需跑 calibrate --refresh
        else:
            net_s = (T.green(f"追买净{net:+.1f}%") if net >= 0
                     else T.red(f"追买净{net:+.1f}%"))
        fr = b.get("fillable_rate")
        fr_s = T.dim(f" 可买{fr:.0%}") if fr is not None else ""
        segs.append("‖ " + head + " " + gap + " / " + net_s
                    + T.cyan(f" 晋级{b['promo']:.0%}") + fr_s)
    return _wrap_segs(segs, DETAIL_W, DETAIL_PREFIX, DETAIL_INDENT)


def _shrink_watch_rows(results: list) -> tuple[list, int]:
    """缩量封板 +5% 备选盯盘名单（明日非一字部分）与一字剔除数。

    单一实现：下方「+5% 备选盯盘」区块与「明日关注清单」合并口径共用——
    此前若两处各写一份筛选，票会悄悄分叉。
    """
    shrink_pool = [x for x in results
                   if x.get("zt_status") == "封板"
                   and isinstance(x.get("vol_ratio"), (int, float))
                   and x["vol_ratio"] == x["vol_ratio"]
                   and float(x["vol_ratio"]) <= OPEN_POOL_VR_MIN]
    oneword = [x for x in shrink_pool if board_rules.open_gap(x) >= ONE_WORD_GAP]
    tomorrow = [x for x in shrink_pool if board_rules.open_gap(x) < ONE_WORD_GAP]
    for x in tomorrow:
        x["_wtier"] = board_rules.watch_tier(board_rules.open_gap(x),
                                             float(x.get("vol_ratio") or 9))
    tomorrow.sort(key=lambda x: (x["_wtier"][0], x.get("vol_ratio", 9),
                                 -x.get("board_score", 0)))
    return tomorrow, len(oneword)


# ---- 明日关注清单·统一关注分（2026-09-15 起，用户要求跨模式统一打分排序）----
# 两支柱合成（0~100）：模式强度（0~70，该模式实证证据档）+ 可买性（0~30，明日能否成交）。
# 档位锚点（每个数字都对应项目自己的实测/判定，不得凭感觉调）：
#   龙头   = board_score×0.7（49~70）——唯一经 OOS 验证方向的序数刻度（五T）；
#   打板   = 56 ——fills_v2 14653 事件回放：画像+成交 n=282 均值+0.16%/胜44%（不显著但为正）；
#   +5%盯盘 = 52 ——+5% 裁决史测全胜 +9.2%（量比≤0.8）但触发频率低。
# 多模式共振每叠加 +4（模式强度封顶 70）。
_WATCH_MODE_BASE = {"打板": 56, "+5%盯盘": 52}
# 紫阳甄选=第四模式（2026-09-17 接入）：模式强度按甄选名次锚定，每名 -5：
# #1=54/#2=49/#3=44/#4=39/#5=34（TOP_N 扩 5 于 2026-09-28，#4/#5 沿用同斜率，
# 不再吃 .get 兜底的 42——那会让 #4 几乎不衰减）。
# 档位依据（两个证据的加权）：①它点火后次日 +1.69%/胜率58%（n=410，全模式中
# 最强的收益实证）→ 不该垫底；②「我们能否预判对哪只」台账零核验 → 上限必须
# 压在打板56（+26=82）之下：54+26=80。名次衰减对应甄选置信递减。
# 台账攒够独立预判日后按 precision@N 实测重锚，不得凭感觉调。
# 可买性=打板26（点火场景=挂板买）。
_WATCH_ZY_RANK_BASE = {1: 54, 2: 49, 3: 44, 4: 39, 5: 34}
_WATCH_RESONANCE = 4
_WATCH_BUYABILITY = {"ok": 30, "hard": 12, "none": 0,
                     "打板": 26, "+5%盯盘": 24}


def _watch_score(r: dict, tags: list[str], zy_rank: int | None = None) -> int:
    """统一关注分：模式强度 + 可买性。分数是排序依据，不是收益预测。"""
    base = (r.get("board_score") or 0) * 0.7 if "龙头" in tags else 0.0
    for t in ("打板", "+5%盯盘"):
        if t in tags:
            base = max(base, float(_WATCH_MODE_BASE[t]))
    if "紫阳" in tags:
        base = max(base, float(_WATCH_ZY_RANK_BASE.get(zy_rank or 3, 42)))
    strength = min(70.0, base + _WATCH_RESONANCE * (len(tags) - 1))
    if "龙头" in tags:
        buy = _WATCH_BUYABILITY[_entry_flag(r)[1]]
    elif "打板" in tags or "紫阳" in tags:
        buy = _WATCH_BUYABILITY["打板"]
    else:
        buy = _WATCH_BUYABILITY["+5%盯盘"]
    return int(round(strength + buy))


def _watchlist_lines(board: list, slam: list, watch_rows: list,
                     zy_rows: list | None = None) -> list[str]:
    """明日关注清单：**四个模式**的票合并去重 → 统一关注分降序 → 最多5只。

    解决的问题：各模式各排各的序、散在不同区块，读者要翻完整份报告才能拼出
    「明天到底盯哪几只」。合并规则：
    - 来源=四个有行动含义的模式：龙头（信息轨过线）、打板（行动轨画像）、
      +5% 盯盘（缩量备选）、紫阳甄选（预判席位点火·台账未验证）；
      开盘池（五R 判死·数据记录）不是关注对象，不入选；
    - 按 code 去重、来源标签叠加（打板的票不再重复叠 +5%盯盘——打板画像
      严格包含于盯盘条件，同票显示最强策略即可）；
    - 统一关注分 = 模式强度(0~70) + 可买性(0~30)，公式与档位锚点见上方
      常量注释——**分数越高越看好**，是全模式可比的唯一排序键；
    - 上榜票若在紫阳甄选内：代码与名称紫红标注（纯文本版以[紫阳]标签区分）；
    - 硬性截断 config.WATCHLIST_SHOW（5）：超出只报数量，完整名单在下方各区块。
    """
    rows: dict[str, dict] = {}

    def _add(r, tag):
        code = str(r.get("code", "")).zfill(6)
        e = rows.setdefault(code, {"r": r, "tags": []})
        if tag not in e["tags"]:
            e["tags"].append(tag)

    for r in board:
        _add(r, "龙头")
    for r in slam:
        _add(r, "打板")
    for r in watch_rows:
        code = str(r.get("code", "")).zfill(6)
        if code in rows and "打板" in rows[code]["tags"]:
            continue
        _add(r, "+5%盯盘")
    for z in (zy_rows or []):
        _add({"code": z.get("code"), "name": z.get("name")}, "紫阳")
        e = rows[str(z.get("code", "")).zfill(6)]
        e["zy_rank"] = int(z.get("rank") or 3)
        e["zy_tags"] = str(z.get("eve_tags") or "")
        try:
            e["zy_score"] = float(z.get("score") or 0)
        except (TypeError, ValueError):
            e["zy_score"] = 0.0

    lines = ["", T.bold("▎明日关注清单（四模式统一评分 · 降序 · 上限5只）"),
             "─" * RULE_W]
    if not rows:
        lines.append(T.dim("  （四个模式今日均无票：龙头无过线、打板画像与缩量备选为空、紫阳未同步）"))
        return lines + [""]
    for e in rows.values():
        e["score"] = _watch_score(e["r"], e["tags"], e.get("zy_rank"))
    order = sorted(rows.values(),
                   key=lambda e: (-e["score"], -(e["r"].get("lianban") or 0),
                                  e["r"].get("vol_ratio") or 9.9))
    for i, e in enumerate(order[:WATCHLIST_SHOW], 1):
        r, tags = e["r"], e["tags"]
        is_zy = "紫阳" in tags
        # 紫阳甄选票：代码与名称紫红标注（与 magenta 模式标签同色系，
        # 整行视觉上更醒目；纯文本落盘版靠 [紫阳] 标签区分）
        code_s = T.magenta(r['code']) if is_zy else r['code']
        name = T.bold(_pad(r.get("name", ""), 10))
        if is_zy:
            name = T.magenta(name)
        tag_s = T.magenta("·".join(tags))
        lb = int(r.get("lianban") or 0)
        lb_s = f"{lb}板" if lb > 1 else "首板"
        vr = r.get("vol_ratio")
        vr_s = (f"量比{vr:.2f}" if isinstance(vr, (int, float)) and vr == vr
                else "")
        close = float(r.get("close") or 0)
        # 展示位按最强模式取：龙头给可买标记、打板给排板价、盯盘给冲5%价、
        # 紫阳单列（非共振时）给甄选名次+画像标签
        if "龙头" in tags:
            flag = _entry_flag(r)[0]
            desc = "·".join(x for x in (lb_s, vr_s, f"收{close:.2f}" if close else "")
                            if x)
        elif "打板" in tags:
            flag = (T.cyan(f"排板≈{trade_rules.limit_price(close):.2f}")
                    if close > 0 else "")
            desc = "·".join(x for x in (lb_s, vr_s) if x)
        elif is_zy:
            flag = T.magenta(f"紫阳甄选#{e.get('zy_rank', 3)}"
                             f"·{e.get('zy_score', 0):.1f}")
            desc = e.get("zy_tags", "")
        else:
            flag = (T.cyan(f"冲5%≈{close * 1.05:.2f}") if close > 0 else "")
            desc = "·".join(x for x in (lb_s, vr_s) if x)
        lines.append(f"  {i}. {e['score']:>3}分 {code_s} {name} [{tag_s}] {flag} {desc}")
    if len(order) > WATCHLIST_SHOW:
        lines.append(T.dim(f"  …另有 {len(order) - WATCHLIST_SHOW} 只"
                           f"（同序排列，完整名单见下方各模式区块）"))
    # 紫阳甄选可见性保底：甄选票若被截断在名单外，单列一行提示（不占5只名额，
    # 不破坏统一评分排序——分数不够就明说，但读者必须知道它今天的判断）
    zy_out = [e for e in order[WATCHLIST_SHOW:] if "紫阳" in e["tags"]]
    zy_in = sum(1 for e in order[:WATCHLIST_SHOW] if "紫阳" in e["tags"])
    zy_all = sum(1 for e in order if "紫阳" in e["tags"])
    if zy_all and zy_out:
        names = "、".join(f"{e['r'].get('name', e['r'].get('code'))}"
                         f"#{e.get('zy_rank', 3)}({e['score']}分)"
                         for e in zy_out)
        lines.append(T.magenta(f"  紫阳甄选未入前5：{names}"
                               + (f"（已入前5 {zy_in}/{zy_all}）" if zy_in else "")))
    lines += _dim_wrap(
        "关注分(0~100)＝模式强度(0~70)＋可买性(0~30)。模式强度：龙头=board_score×0.7"
        "（唯一OOS验证序数）｜打板56（回放+0.16%/44%）｜+5%盯盘52（史测全胜+9.2%·低频）"
        "｜紫阳甄选=名次54/49/44/39/34（点火实证+1.69%但预判未验证，封顶打板之下）；"
        "多模式共振每叠加+4。可买性：可买30/难成交12/买不到0；打板与紫阳26（排板）/"
        "盯盘24（两关条件）。分数＝模式证据×可执行性的合成排序，越高越看好，"
        "**不是收益预测**。",
        indent="  ")
    return lines + [""]


def _load_bucket_stats(out_dir: str) -> dict:
    """读 calibrate.py 每日落盘的分数桶实测统计，供 _detail 标注同档数字。

    缺失/损坏一律返回空 dict——校准数据不该成为报告生成的前置依赖。
    """
    import json
    path = os.path.join(out_dir, "archive", "bucket_stats.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _block(lines: list[str], cands: list[dict], title: str, note: str = "",
           stats: dict | None = None, foot: list[str] | None = None) -> None:
    title = title.replace("▎", "")   # 统一样式：区块标题不带符号，由 _block 加
    lines.append("")
    lines.append(f"▎{title} ({len(cands)}只)" + (f"｜{note}" if note else ""))
    lines.append("─" * RULE_W)
    if not cands:
        lines.append("  （无）")
        for f in (foot or []):
            lines += _dim_wrap(f)
        return
    for r in cands:
        lines.append(_line(r))
        lines.extend(_detail(r, stats))
    for f in (foot or []):
        lines += _dim_wrap(f)


def write_report(results: list[dict], skipped: list[dict], cands: list[dict],
                 trade_date: dt.date, out_dir: str,
                 signal_only: bool = True,
                 ctx: dict | None = None,
                 spot: pd.DataFrame | None = None) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # 只保留最新一份：固定文件名，每次运行覆盖
    txt_path = os.path.join(out_dir, "report_latest.txt")
    csv_path = os.path.join(out_dir, "report_latest.csv")

    emo = emotion_summary(cands, ctx)
    # board = 过线候选，按评分降序。无条件完整渲染（曾因 signal_only 门禁只剩计数）
    board = sorted((r for r in results if _passed(r)),
                   key=lambda x: -x.get("board_score", 0))

    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    ladder_str = (" ".join(f"{n}板×{c}"
                           for n, c in sorted(emo["ladder"].items(), reverse=True))
                  if emo["ladder"] else "涨停池缺失")
    yzt_str = (f"均值 {emo['yzt_ret']:+.1f}%，红盘 {emo['yzt_win']}/{emo['yzt_n']}"
               if emo["yzt_ret"] is not None else "无数据")
    from mood import STAGE_ADVICE, STAGE_POSITION
    stage = str((ctx or {}).get("cycle_stage") or "未知")
    pos = STAGE_POSITION.get(stage, 0.3)
    filled = int(round(pos * 10))
    bar = "█" * filled + "░" * (10 - filled)
    _sfn = T.stage_fn(stage)
    bar_c = _sfn(bar) if _sfn else bar
    cycle_str = (f"{T.stage(stage + '期')}（{(ctx or {}).get('cycle_note', '')}）"
                 f"｜{STAGE_ADVICE.get(stage, '—')}｜总仓位 [{bar_c}] {pos:.0%}")
    # 表头框线：中文按 2 列显示、按 1 字符计数，手工数空格永远对不齐
    # （旧版实测「生成 22:19    ║」突出框外）→ 一律按显示宽度补齐
    def _box(s: str) -> str:
        return "║" + _pad(s, BOX_W) + "║"

    lines = [
        "╔" + "═" * BOX_W + "╗",
        _box("  A股复盘收割器 · 二板定龙头"),
        "╠" + "═" * BOX_W + "╣",
        _box(_pad(f"  复盘日期  {trade_date}（{weekdays[trade_date.weekday()]}）", 44)
             + f"生成  {dt.datetime.now():%H:%M}"),
        "╚" + "═" * BOX_W + "╝",
        "",
        "▎市场情绪速览",
        "─" * RULE_W,
        f"  曾涨停: {emo['up_total']} 只（封板 {emo['up_sealed']} / 炸板 {emo['up_broken']}，炸板率 {emo['break_rate']:.1f}%）",
        f"  连板高度: {emo['height']} 板（{ladder_str}）",
        f"  昨日涨停今日表现: {yzt_str}（常态约+2%~4%，<-2%转弱）",
        f"  情绪周期: {cycle_str}",
        f"  曾跌停: {emo['down_total']} 只（封死 {emo['down_sealed']} / 翘板 {emo['down_qiaoban']}）；"
        f"振幅≥7%: {emo['amp_total']} 只",
        f"  候选合计: {emo['cand_total']} 只；分析成功 {len(results)} 只；"
        f"二板定龙头候选: {len(board)} 只（评分≥{BOARD_SCORE_MIN}）",
    ]

    # ▎二板定龙头名单——一等区块，不受 --all 门禁。
    # 双轨分离：这是「信息口径」（评分预测次日能否继续封板＝龙头识别），不是买入指令。
    # 实测过线池 37.9% 次日开盘≥+9.5% 属排队买不到，剔除后可成交口径开→收为负，
    # 所以名单必须自带可成交性标记，行动位看下方各执行区块与末尾台账。
    stats = _load_bucket_stats(out_dir)
    n_buy = sum(1 for r in board if _entry_flag(r)[1] == "ok")
    _block(lines, board, "二板定龙头 · 信息口径（龙头识别，非买入指令）",
           note=f"评分≥{BOARD_SCORE_MIN}｜次日可买 {n_buy} 只 / 买不到·难成交 {len(board) - n_buy} 只",
           stats=stats,
           foot=["同档实测来自 out/archive/bucket_stats.json（calibrate.py 每日更新），三个数字对应三种动作：",
                 "  排板隔夜 = T 日挂涨停价排板**且成交**（实测排板成交率仅 30.7%，一字板排不进）；",
                 "  追买净 = T+1 开盘买→当日收盘卖，已扣双边成本 ≈0.15%（一字/准一字票买不到，故为负）；",
                 "  晋级 = 次日仍封板，龙头识别的信息口径，**不是收益**；可买 = 该档次日非一字的比例。",
                 "  标「明日买不到/难成交」的票信息最强但无法建仓（一字组晋级 78% 而可成交口径"
                 "开→收 −2.11%）——要看不要追。"])

    # ▎行动轨：可成交打板画像（盘中触板挂涨停价买 → 次日 9:45 前卖）
    # 该区块此前被移除后只剩计数供摘要引用，摘要称其为「槽位①(主)」却在整份报告里
    # 找不到名单——行动轨指向空气。双轨分离要求信息轨与行动轨都可见。
    slam = board_rules.slam_candidates(results)
    slam_exp = EXPECT["slam"]
    fill_rate = slam_exp.get("fill_rate") or 0.0
    slam_lab = board_rules.expect_label("slam")
    lines.append("")
    lines.append(f"▎可成交打板画像 · 行动轨（盘中触板挂涨停价买→次日9:45前卖） ({len(slam)}只)")
    lines.append("─" * RULE_W)
    if not slam:
        lines.append(T.dim("  （今日无符合画像的票：需封板 + 实体开盘<+3% + 量比 0.4~0.8）"))
    for x in slam[:SLAM_SHOW]:
        lb = int(x.get("lianban") or 0)
        lb_str = _pad(f"今日第{lb}板" if lb > 1 else "今日首板", 12)
        name = _pad(x.get("name", ""), 12)
        close = float(x.get("close") or 0)
        queue = trade_rules.limit_price(close) if close > 0 else 0.0
        boost = x.get("_slam_boost")
        lines.append(f"  {x['code']} {name} {lb_str}"
                     f"量比{_pad(str(x.get('vol_ratio', '')), 5)}"
                     f"开盘{board_rules.open_gap(x):+5.1%}  收盘{close:>7.2f}"
                     f"  排板价≈{queue:.2f}"
                     + (f"  {T.cyan(boost)}" if boost else ""))
    if len(slam) > SLAM_SHOW:
        lines.append(T.dim(f"  …另有 {len(slam) - SLAM_SHOW} 只"))
    for f in (f"史测（已接成交模型，fills_v2 14653 事件全量回放）：画像子集排板成交率 "
              f"{fill_rate:.1%}；成交单 {slam_lab}、中位 {slam_exp.get('median'):+.2f}%、"
              f"聚类 t={slam_exp.get('t'):+.2f} 不显著。",
              f"结构是二元赌局：{slam_exp.get('rule_split')}。均值被右偏掩盖而中位为负，"
              f"仓位按「可能全损」计；勿引用未接成交模型的预注册值 "
              f"{board_rules.expect_label('slam', 'prereg')}。"):
        lines += _dim_wrap(f)

    # ▎明日开盘买入池（主口径：开→开日轮动，用户模式）
    open_pool, (ow_n, shrink_n, hi_n) = board_rules.open_pool_of(results)
    open_pool.sort(key=lambda x: (x["_otier"][0], -x.get("board_score", 0)))

    lines.append("")
    lines.append(T.dim("▎封板温和量观察（数据记录，不推荐买入）"))
    lines.append("─" * RULE_W)
    if open_pool:
        lines.append(T.dim(f"  温和放量封板股（{len(open_pool)} 只，含前档）："))
        for x in open_pool[:OPEN_POOL_SHOW]:
            tier, exp_lab = x["_otier"]
            lb = x.get("lianban", 0)
            lb_str = _pad(f"今日第{lb}板" if lb > 1 else "今日首板", 12)
            name = _pad(x.get("name", ""), 12)
            close = x.get("close", 0)
            vr_s = _pad(str(x.get("vol_ratio", "")), 5)
            lines.append(T.dim(f"  {x['code']} {name} {lb_str}"
                              f"量比{vr_s} 历史开→开{exp_lab}  收盘{close:>7.2f}"))
        if len(open_pool) > OPEN_POOL_SHOW:
            lines.append(T.dim(f"  …另有 {len(open_pool) - OPEN_POOL_SHOW} 只"))
    notes = []
    if ow_n:
        ow_names = "、".join(x.get("name", "") for x in results
                            if x.get("zt_status") == "封板"
                            and isinstance(x.get("vol_ratio"), (int, float))
                            and x["vol_ratio"] == x["vol_ratio"]
                            and board_rules.open_gap(x) >= ONE_WORD_GAP)
        notes.append(f"今日一字/准一字 {ow_n} 只未列·明日竞价难成交"
                     + (f"：{ow_names}" if ow_names else ""))
    if shrink_n:
        notes.append(f"缩量票 {shrink_n} 只不入池（开盘买负期望，见附录+5%备选）")
    if hi_n:
        notes.append(f"高位放量（3板+·vr>1.5）{hi_n} 只排除（-1.4%~-2.3%）")
    if notes:
        # 每条排除说明独立成行：拼成「；」长串再折行会从中文词组中间硬断
        # （中文无空格可断），实测把「缩量票 1 只不入池」劈成两半
        for nt in notes:
            lines += _dim_wrap("· " + nt, indent="    ")
    lines += _dim_wrap("数学终审（五R）：12 个月 t=-0.19 对称硬币，130 个条件格"
                       "无一通过多重比较——本区块仅供数据记录，不构成买入依据。")

    # ▎缩量封板 +5% 备选盯盘（下一区块自带前导空行，此处不再补，否则出现连续两个空行）
    # 名单选取是单一实现（_shrink_watch_rows），与「明日关注清单」合并口径共用
    tomorrow, n_oneword = _shrink_watch_rows(results)
    lines.append("")
    # 本区块渲染的是按 vol_ratio<=0.8 筛的缩量封板票盯盘名单
    lines.append(T.bold("▎缩量封板 +5% 备选盯盘（明日冲+5%看实时量比）"))
    lines.append("─" * RULE_W)
    lines += _dim_wrap("逻辑：缩量封板→明日冲 +5% 看实时量比买入→后日开盘卖"
                       "（+9.2%/次全胜但频率低；与开盘池的放量口径相反）")
    if not tomorrow:
        lines.append(T.dim(f"  （今日缩量票全部为一字/准一字 {n_oneword} 只，无可盯名单）"))
    for x in tomorrow[:6]:
        tier, shape_lab, buy_lab = x["_wtier"]
        lb = x.get("lianban", 0)
        lb_str = _pad(f"今日第{lb}板" if lb > 1 else "今日首板", 12)
        star = T.stars3(tier)
        name = _pad(x.get("name", ""), 12)
        close = x.get("close", 0)
        vr_s = _pad(str(x.get("vol_ratio", "")), 5)
        lines.append(f"  {star} {x['code']} {name} {lb_str}"
                    f"{shape_lab}量比{vr_s} {buy_lab} "
                    f"收盘{close:>7.2f}  冲+5%≈{close * 1.05:.2f}")
    if len(tomorrow) > 6:
        lines.append(T.dim(f"  …另有 {len(tomorrow) - 6} 只期望档位较低"))
    lines.append(T.dim("  两关：①开盘<+5% ②冲+5%时实时量比≤0.8买/0.8~1半仓/>1放弃"))

    if not signal_only:
        _block(lines, results, "附、全部候选（--all）", stats=stats)

    # ▎明日决策摘要——双轨分离：信息轨（评分识别龙头）与行动轨（可执行清单）分开报。
    # 旧版摘要写「槽位①(主)：打板画像 +4.2%/84%」，该数字与 config 的 +4.69%、
    # board_rules 的 +4.43%/+4.7% 互相矛盾，且完全不提二板名单、
    # 也不提已被五R 判死却占最大版面的开盘池。期望数字一律走 config.EXPECT。
    # 紫阳甄选行：只读 ziyang 台账（零网络，渲染层无副作用）；未接入/无登记时整行省略
    # zy_rows 同时喂给明日关注清单（紫阳=第四模式，统一关注分参与排序）
    try:
        import ziyang
        zy_line = ziyang.selection_line(trade_date)
        zy_rows = ziyang.selection_rows(trade_date)
    except Exception:
        zy_line, zy_rows = None, []
    summary = [
        "▎明日决策摘要",
        "─" * RULE_W,
        *([zy_line] if zy_line else []),
        f"  信息轨：二板定龙头 {len(board)} 只（评分≥{BOARD_SCORE_MIN}）｜"
        f"次日可买 {n_buy} 只、买不到·难成交 {len(board) - n_buy} 只 → "
        + T.cyan("非建仓指令"),
        # 摘要只留最决策相关的数字，完整口径与显著性在各区块脚注里；
        # 措辞从简是为了守住 100 显示列的版面上限（有测试守卫）
        f"  行动轨：打板画像 {len(slam)} 只（成交率 {fill_rate:.0%}）"
        + f"｜开盘池 {len(open_pool)} 只" + T.red("（五R 判死）"),
        f"  周期: {stage}期·建议仓位 {pos:.0%} ｜ 空闲日空仓·不追高开 ｜ "
        f"{'⚠退潮期：新信号建议只看不做' if stage == '退潮' else '行动轨无票＝空仓'}",
    ]
    head = next(i for i, l in enumerate(lines) if l.startswith("▎市场情绪速览"))
    # 情绪风向标放报告**开头**（用户约定 2026-11-22：像关注清单那样显眼展示、
    # 单独成节）——紧跟表头框、先于决策摘要。零网络只读 vane 台账；
    # 未跑/未接入时整节省略。
    try:
        import vane
        vane_block = vane.section_lines(trade_date)
    except Exception:
        vane_block = []
    # 明日关注清单紧跟决策摘要：四个模式（龙头/打板/+5%盯盘/紫阳甄选）的票
    # 统一关注分排序合并成一张名单（≤WATCHLIST_SHOW 只）——分数越高越看好。
    watch = _watchlist_lines(board, slam, tomorrow, zy_rows)
    lines[head:head] = vane_block + summary + watch

    # ▎台账核账区（复盘末尾统一展示：买卖记录与累计成绩属事后对账）
    # 入口在 ledger.tracking_sections：持进程互斥锁更新两本账，单本账异常不拖垮报告
    if spot is not None:
        lines += ledger.tracking_sections(out_dir, trade_date, spot, results)

    lines += [
        "",
        "─" * RULE_W,
        f"跳过 {len(skipped)} 只（日K失败/上市不足{MIN_BARS}根K线）",
    ]
    lines += _wrap("口径：主板 ±10%，ST 剔除；仅收盘封板股参与二板判定；"
                   "连板数优先东财涨停池，缺失时日K近似（连续涨幅≥9.5%）。")
    lines += [
        "提示：本工具仅做数据筛选与统计，不构成任何投资建议。",
    ]
    if any(bool(c.get("fill")) for c in cands):
        lines.append("注：本报告为历史补录：候选来自东财历史涨停/炸板池；"
                     "跌停与振幅池历史数据不可补，情绪统计可能偏少。")
    txt = "\n".join(lines) + "\n"
    # 落盘一律纯文本。同一份 lines 既写文件又被 work.py 打印，而 termstyle 的颜色
    # 开关取自 sys.stdout.isatty() —— 交互运行时 ANSI 码会被写进文件（实测 143 行里
    # 94 行带 ESC），编辑器打开全是 [2m/[91m 噪音、grep 区块标题也要先剥码。
    # 彩色版通过第三个返回值交给调用方直接打印，不再读回文件。
    plain = T.strip_ansi(txt)
    fstore.write_text(txt_path, plain)

    def _csv_row(r: dict) -> dict:
        row = {c: r.get(c, "") for c in CSV_COLUMNS}
        row["trade_date"] = f"{trade_date:%Y-%m-%d}"
        row["weekday"] = T.weekday_cn(trade_date)
        row["pools"] = "+".join(r["pools"])
        row["board_pass"] = int(bool(row["board_pass"]))
        row["market_max_lianban"] = emo["height"] or ""
        row["cycle_stage"] = (ctx or {}).get("cycle_stage", "")
        # source 供 calibrate 按 backfill/live 分层：两者口径差 2~4 分，混池比较会
        # 让「60-69 桶无超额」这类结论失去意义。上游已标注则保留，否则记为 live。
        row["source"] = r.get("source") or "live"
        if emo["yzt_ret"] is not None:
            row["zt_money_effect"] = round(emo["yzt_ret"], 2)
        return row

    rows = [_csv_row(r) for r in results] + [_csv_row(r) for r in skipped]
    df_out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    fstore.write_dataframe_csv(csv_path, df_out)      # 原子写，报告中途被杀不留半截 CSV

    # 信号自动归档：滚动缓存只留 3 个交易日，历史信号必须落盘才能做事后胜率校准
    # （归档为每日证据资产，原子写防截断）
    try:
        arch_dir = os.path.join(out_dir, "archive")
        os.makedirs(arch_dir, exist_ok=True)
        fstore.write_text(os.path.join(arch_dir, f"report_{trade_date:%Y%m%d}.txt"),
                          plain)        # 归档是长期检索资产，同样不带 ANSI 码
        fstore.write_dataframe_csv(
            os.path.join(arch_dir, f"signals_{trade_date:%Y%m%d}.csv"), df_out)
    except OSError as e:
        print(f"  [提示] 信号归档写入失败（不影响本次报告）: {e}")

    # 清理历史报告：目录里只保留最新的 txt + csv 两份
    keep = {"report_latest.txt", "report_latest.csv"}
    for f in os.listdir(out_dir):
        if f.startswith("report_") and f.endswith((".txt", ".csv")) and f not in keep:
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass

    # 第三项是**彩色**文本：落盘的是纯文本，终端要彩色就直接打印它，
    # 不要像旧版那样把文件读回来（那样等于把纯文本当彩色用，白丢颜色）
    return txt_path, csv_path, txt

def calibrate_hint(out_dir: str, trade_date: dt.date) -> None:
    """收盘运行后提醒：归档中存在早于本日的信号文件 → 可跑 calibrate.py 更新校准。"""
    import glob
    import re
    older = []
    for f in glob.glob(os.path.join(out_dir, "archive", "signals_*.csv")):
        m = re.search(r"signals_(\d{8})\.csv$", os.path.basename(f))
        if m:
            d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            if d < trade_date:
                older.append(d)
    if older:
        print(f"校准提示: 归档中有 {len(older)} 个早于本日的信号文件"
              f"（最新 {max(older)}），可运行 python calibrate.py 更新前向数据与分数桶报告。")
