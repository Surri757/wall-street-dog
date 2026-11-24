# -*- coding: utf-8 -*-
"""校准报告的各文本区块（渲染层，纯函数：rows → str）。"""
from __future__ import annotations

from calib.buckets import BUCKETS
from calib.forward import MAIN_METRIC, MIN_INFER_DAYS
from calib.stats import cluster_diff, cluster_mean, fmt_mean, fmt_t
from config import COST_ROUND_TRIP


def _table(rows: list[dict], score_key: str = "board_score") -> tuple[str, dict]:
    """分数桶对照表。**主口径 = 可成交净收益**，信息口径并列但标注为参考。

    返回 (文本, stats)。stats["bucket_main"] 供 summarize 判单调性——此前用的是
    bucket_close（不可成交口径），等于拿买不到的收益给评分排序。
    旧 _forward.csv 行缺新列时该桶显示破折号，不崩溃也不当成 0。
    """
    has_new = any("fillable" in r for r in rows)
    head_note = "" if has_new else "  ⚠ 无新口径列，请先跑 calibrate.py --refresh"
    lines = [
        f"{'分数桶':<10}{'样本':>5}{'日簇':>5}{'可成交':>7}{'一字%':>7}"
        f"{'净开→收%':>10}{'聚类t':>8}{'晋级率':>8}{'[参考]次收%':>12}",
        "-" * 78,
    ]
    if head_note:
        lines.append(head_note)
    stats = {"bucket_main": [], "bucket_info": [], "bucket_promo": [],
             "bucket_n": [], "bucket_days": []}

    def _row_of(label: str, seg: list[dict]) -> str:
        n = len(seg)
        days = len({r.get("date") for r in seg if r.get("date")})
        fill = [r for r in seg if r.get("fillable")]
        st = cluster_mean(fill, MAIN_METRIC) if fill else None
        promo = sum(r.get("promoted") or 0 for r in seg) / n
        info = sum(r.get("next_close_pct") or 0 for r in seg) / n
        if st is None or st["mean"] is None:
            ow, main_s, t_s = "—", "—", "—"
        else:
            ow = f"{1 - len(fill) / n:.0%}"
            main_s = f"{st['mean']:>10.2f}"
            t_s = f"{fmt_t(st):>8}"
        return (f"{label:<10}{n:>5}{days:>5}{len(fill):>7}{ow:>7}"
                f"{main_s:>10}{t_s:>8}{promo:>8.1%}{info:>12.2f}")

    for lo, hi in BUCKETS:
        seg = [r for r in rows if lo <= (r.get(score_key) or 0) <= hi]
        label = f"{lo}-{hi}"
        if not seg:
            lines.append(f"{label:<10}{0:>5}{'—':>5}{'—':>7}{'—':>7}"
                         f"{'—':>10}{'—':>8}{'—':>8}{'—':>12}")
            stats["bucket_main"].append(None)
            stats["bucket_info"].append(None)
            stats["bucket_promo"].append(None)
            stats["bucket_n"].append(0)
            stats["bucket_days"].append(0)
            continue
        lines.append(_row_of(label, seg))
        fill = [r for r in seg if r.get("fillable")]
        st = cluster_mean(fill, MAIN_METRIC) if fill else None
        stats["bucket_main"].append(st["mean"] if st else None)
        stats["bucket_info"].append(
            sum(r.get("next_close_pct") or 0 for r in seg) / len(seg))
        stats["bucket_promo"].append(
            sum(r.get("promoted") or 0 for r in seg) / len(seg))
        stats["bucket_n"].append(len(seg))
        stats["bucket_days"].append(len({r.get("date") for r in seg if r.get("date")}))
    if rows:
        lines.append("-" * 78)
        lines.append(_row_of("基线(全部)", rows))
    lines.append("-" * 78)
    lines.append("  净开→收 = 次日开盘买→次日收盘卖，扣双边成本 "
                 f"{COST_ROUND_TRIP:.2%}，仅统计可成交样本；聚类t 按交易日聚类。")
    lines.append("  [参考]次收 = 次日收盘相对**信号日收盘**，信号日已封死涨停买不到，"
                 "属龙头识别的信息口径，不是收益。")
    return "\n".join(lines), stats


def _contrast_line(scored: list[dict], rest: list[dict], label: str,
                   val_key: str, fillable_only: bool, min_days: int = 8) -> list[str]:
    """渲染「过线 vs 未过线」的按日配对聚类推断。

    日簇不足时明确说「无法推断」，不给一个看似有效的 t——本项目 OOS 只有 8 簇、
    台账只有 4 簇，点估计比较曾长期被当成结论。
    """
    a = [r for r in scored if r.get("fillable")] if fillable_only else scored
    b = [r for r in rest if r.get("fillable")] if fillable_only else rest
    if not a or not b:
        why = "（需先跑 calibrate.py --refresh 补可成交口径列）" if fillable_only else ""
        return [f"{label}: 样本不足（过线 {len(a)} / 未过线 {len(b)}），无法对比{why}"]
    d = cluster_diff(a, b, val_key, "date")
    ma, mb = cluster_mean(a, val_key, "date"), cluster_mean(b, val_key, "date")
    if ma["mean"] is None or mb["mean"] is None:
        return [f"{label}: 缺 {val_key} 值，无法对比"]
    scale = 100.0 if val_key == "promoted" else 1.0
    unit = "pp" if val_key == "promoted" else "%"
    head = (f"{label}: 过线 {ma['mean'] * scale:+.2f}{unit}"
            f"(n={ma['n']}/{ma['clusters']}簇) vs 未过线 "
            f"{mb['mean'] * scale:+.2f}{unit}(n={mb['n']}/{mb['clusters']}簇)")
    if d["diff"] is None:
        return [head + " → ⚠ 两组无任何共同交易日，**无法配对推断**"]
    if d["t"] is None or d["days"] < min_days:
        return [head + f" → ⚠ 配对日数仅 {d['days']}（<{min_days}），**无法推断**，"
                       f"差值 {d['diff'] * scale:+.2f}{unit} 只是点估计",
                f"    （单边日丢弃 {d['dropped']} 个；样本按交易日聚类，"
                f"簇数不足时任何显著性结论都不可信）"]
    lo, hi = d["ci"]
    sig = "✔ 显著" if abs(d["t"]) >= 1.96 else "✘ 不显著"
    return [head + f" → 差 {d['diff'] * scale:+.2f}{unit}，聚类SE {d['se'] * scale:.2f}，"
                   f"t={d['t']:+.2f} {sig}，95%CI[{lo * scale:+.2f},{hi * scale:+.2f}]"
                   f"（配对 {d['days']} 日，丢弃 {d['dropped']}）"]


def summarize(rows: list[dict]) -> str:
    """全部样本的分数桶汇总 + 自动判读。**主口径 = 可成交净收益**。"""
    from config import BOARD_SCORE_MIN
    table, st = _table(rows)
    lines = [table]
    total = len(rows)
    lines.append("")
    scored = [r for r in rows if (r.get("board_score") or 0) >= BOARD_SCORE_MIN]
    rest = [r for r in rows if (r.get("board_score") or 0) < BOARD_SCORE_MIN]
    pass_rate = len(scored) / total if total else 0
    lines.append(f"当前门槛 {BOARD_SCORE_MIN}：过线率 {pass_rate:.0%}"
                 f"（过线率>80% 说明门槛无筛选力，建议提到样本分数的 P60~P75）")
    # 单调性：主判据用可成交净收益；信息口径降级为参考
    main_seq = [v for v, c in zip(st["bucket_main"], st["bucket_n"])
                if v is not None and c >= 5]
    mono = all(main_seq[i] <= main_seq[i + 1] for i in range(len(main_seq) - 1))
    lines.append(f"**可成交净收益**单调性："
                 f"{'✔ 单调（评分排序有效）' if mono else '✘ 非单调（高分桶不升反降）'}"
                 f"（仅计入样本≥5 的 {len(main_seq)} 个桶）")
    info_seq = [v for v, c in zip(st["bucket_info"], st["bucket_n"])
                if v is not None and c >= 5]
    mono_i = all(info_seq[i] <= info_seq[i + 1] for i in range(len(info_seq) - 1))
    lines.append(f"[参考] 信息口径（次日收盘，买不到）单调性：{'✔' if mono_i else '✘'}")
    lines.append("")
    lines += _contrast_line(scored, rest, "可成交净收益超额", MAIN_METRIC, True)
    lines += _contrast_line(scored, rest, "晋级率超额(信息口径)", "promoted", False)
    lines.append("注：二进三基础比率 ≈34.7%（全主板历史）。晋级率高于它只说明**会识别龙头**，"
                 "不代表能买到；能不能赚钱的判据是可成交净收益。")
    lines.append("注：晋级率高≠能赚钱。信号日已封死涨停，晋级的好处只有在 T 日就建仓"
                 "（排板）时才拿得到；T+1 开盘追买的一字票根本成交不了。")
    lines.append("注：回灌样本缺赚钱效应、A维走行业兜底，跨桶可比但绝对分与实跑口径差 2~4 分。")
    return "\n".join(lines)


def oos_section(rows: list[dict]) -> str:
    """样本外检验：拟合窗口（config.CALIB_FIT_UNTIL，含）之后的样本单独统计。

    判据（2026-09-10 起）：主口径为**可成交净收益**，且必须过日簇门槛才给结论。
    此前是 `keep = pool > base` 两个点估计比大小，而 OOS 只有 8 个独立交易日，
    把「不显著」硬判成「保持」或「消失」都是错的 → 改为四态。
    """
    from config import BOARD_SCORE_MIN, CALIB_FIT_UNTIL
    dated = [r for r in rows if r.get("date")]
    oos = [r for r in dated if r["date"] > CALIB_FIT_UNTIL]
    lines = [f"\n【样本外检验】拟合窗口截至 {CALIB_FIT_UNTIL}（含），之后为样本外"]
    if not oos:
        lines.append("尚无样本外已实现数据——等待新交易日收盘后运行 main.py 再执行 calibrate。")
        return "\n".join(lines)
    table, _st = _table(oos)
    lines.append(table)
    days = len({r["date"] for r in oos})
    scored = [r for r in oos if (r.get("board_score") or 0) >= BOARD_SCORE_MIN]
    rest = [r for r in oos if (r.get("board_score") or 0) < BOARD_SCORE_MIN]
    lines.append("")
    lines.append(f"样本外规模：{len(oos)} 条 / **{days} 个独立交易日**"
                 f"（过线 {len(scored)} / 未过线 {len(rest)}）")
    if not scored:
        lines.append("样本外暂无过线信号。")
        return "\n".join(lines)
    lines += _contrast_line(scored, rest, "样本外·可成交净收益", MAIN_METRIC, True)
    lines += _contrast_line(scored, rest, "样本外·晋级率(信息口径)", "promoted", False)
    d = cluster_diff([r for r in scored if r.get("fillable")],
                     [r for r in rest if r.get("fillable")], MAIN_METRIC, "date")
    if d["t"] is None or d["days"] < MIN_INFER_DAYS:
        lines.append(f"→ ⚠ **簇数不足**（{d['days']} 个配对日 < {MIN_INFER_DAYS}）："
                     f"本轮调参既不能判通过、也不能判回退，继续积累样本。")
    elif d["diff"] <= 0:
        lines.append("→ ✘ 消失：样本外可成交口径无正超额，按预注册应回退本轮 C 维调整。")
    elif abs(d["t"]) >= 1.96:
        lines.append("→ ✔ 保持：样本外可成交口径有显著正超额。")
    else:
        lines.append("→ ◐ 方向为正但不显著：不足以判通过，继续积累。")
    return "\n".join(lines)


def _source_of(r) -> str:
    """归档行的来源标签：backfill（历史回灌）/ live（实跑）/ live*（实跑但无标记）。

    backfill.py 一直写 `source="backfill"`（backfill.py:175,196），而 report.py
    直到 2026-09-10 才把 source 加进 CSV_COLUMNS——所以缺列/空值的历史行必然来自
    实跑，标 `live*` 以示「实跑，但归档当时没有这一列」。
    """
    v = r.get("source")
    # pandas 缺列时给的是 float('nan')，而 nan 是**真值**，必须先显式排除，
    # 不能靠 `v or ""` 兜（那样只是碰巧落到 live* 分支）
    if v is None or v != v:
        return "live*"
    s = str(v).strip().lower()
    if s in ("backfill", "live"):
        return s
    return "live*" if s in ("", "nan", "none") else s


def source_section(rows: list[dict]) -> str:
    """按 backfill/live 分层出表 + 门槛边界敏感性。

    为什么必须分层：两者口径差 2~4 分——backfill 缺赚钱效应（D维按缺失计）、
    A维走行业兜底、97/106 条带「缺失1项(−2)」、7 条带「缺失7项(−14)」，
    而 live 60/61 条无缺失。混池比较会让分桶结论失真：66-69 分挤了 32 条（19%），
    门槛 70 下 ±2 分即搬动约 10% 样本跨池，而「60-69 桶无超额」正是当初把门槛
    从 60 提到 70 的**唯一依据**。
    """
    from config import BOARD_SCORE_MIN
    lines = ["\n【来源分层：backfill（历史回灌） vs live（实跑）】"]
    if not rows:
        lines.append("  （无样本）")
        return "\n".join(lines)
    groups: dict = {}
    for r in rows:
        groups.setdefault(_source_of(r), []).append(r)
    lines.append(f"{'来源':<10}{'n':>5}{'日簇':>6}{'均分':>7}{'缺失罚':>8}"
                 f"{'过线率':>8}{'可成交n':>9}{'净开→收%':>10}{'聚类t':>8}")
    lines.append("-" * 74)
    for src in sorted(groups):
        seg = groups[src]
        days = len({r.get("date") for r in seg if r.get("date")})
        avg_score = sum(r.get("board_score") or 0 for r in seg) / len(seg)
        miss = sum(1 for r in seg if "缺失" in (r.get("board_note") or ""))
        passed = [r for r in seg if (r.get("board_score") or 0) >= BOARD_SCORE_MIN]
        fill = [r for r in seg if r.get("fillable")]
        st = cluster_mean(fill, MAIN_METRIC) if fill else None
        net_s = fmt_mean(st)
        t_s = f"{fmt_t(st):>8}" if st else f"{'—':>8}"
        lines.append(f"{src:<10}{len(seg):>5}{days:>6}{avg_score:>7.1f}"
                     f"{miss / len(seg):>8.0%}{len(passed) / len(seg):>8.0%}"
                     f"{len(fill):>9}{net_s}{t_s}")
    lines.append("-" * 74)
    lines.append("  ⚠ 两层的绝对分不可直接比较（backfill 系统性低 2~4 分）；"
                 "跨层只比**同层内**的过线 vs 未过线。")

    # 门槛敏感性：结论对 BOARD_SCORE_MIN 有多敏感
    lines.append("")
    lines.append(f"【门槛敏感性】现行 BOARD_SCORE_MIN={BOARD_SCORE_MIN}，"
                 f"主口径＝{MAIN_METRIC}（仅可成交子样本）")
    lines.append(f"{'门槛':>6}{'过线n':>7}{'日簇':>6}{'净开→收%':>10}{'聚类t':>8}"
                 f"{'未过线n':>9}{'净开→收%':>10}{'配对日':>7}{'差值t':>8}")
    lines.append("-" * 74)
    for th in (66, 68, 70, 72, 74):
        hi = [r for r in rows if (r.get("board_score") or 0) >= th]
        lo = [r for r in rows if (r.get("board_score") or 0) < th]
        hf = [r for r in hi if r.get("fillable")]
        lf = [r for r in lo if r.get("fillable")]
        sh = cluster_mean(hf, MAIN_METRIC) if hf else None
        sl = cluster_mean(lf, MAIN_METRIC) if lf else None
        d = cluster_diff(hf, lf, MAIN_METRIC, "date") if (hf and lf) else None
        mark = " ←现行" if th == BOARD_SCORE_MIN else ""
        lines.append(
            f"{th:>6}{len(hi):>7}{(sh['clusters'] if sh else 0):>6}"
            f"{fmt_mean(sh)}{(fmt_t(sh) if sh else '—'):>8}"
            f"{len(lo):>9}{fmt_mean(sl)}"
            f"{(d['days'] if d else 0):>7}{(fmt_t(d) if d else '—'):>8}{mark}")
    lines.append("-" * 74)
    lines.append("  若相邻门槛的净收益与 t 值大幅跳动，说明结论由少数边界样本驱动，"
                 "门槛本身不该被当成精确参数。")
    return "\n".join(lines)


