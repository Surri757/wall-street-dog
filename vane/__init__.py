# -*- coding: utf-8 -*-
"""情绪风向标模块（vane）。

筛选对象：下一交易日 4板晋级5板 / 5板晋级6板 的主板票，名额 TOP_N=3。
板数不要求连续（连板优先），但同一轮内相邻两板间隔 ≤ MAX_GAP 个交易日
——久远断档即重置（情绪延续性优先）。定位是**观察标**（市场空间/分歧的
锚点），不是买入清单：只挂主报告摘要行 + 独立报告，不进明日关注清单。

━━━ 已接入主工程（约定：主报告**独立区块**，不并入决策摘要/关注清单） ━━━

    work / python main.py 复盘最新交易日时自动调用 daily_sync()：
    拉窗口涨停池 → 回溯本轮板数 → 排序取名额 → 台账登记（预注册纪律）。
    历史日期查看与盘中自动跳过；失败只打一行警告，不阻断主流程。
    report.py 渲染「▎情绪风向标」独立区块（section_lines，零网络只读台账）；
    完整报告（板链/梯队/落选观察）在 out/research/vane/report_latest.txt，
    CLI: python -m vane report。

    独立运行仍有完整 CLI：
    python -m vane scan [--date YYYYMMDD] | verify | report

━━━ 程序化接口（供主工程/研究脚本调用） ━━━

    vane.daily_sync(trade_date)     # -> str  当日筛选+登记（幂等，含当日重登）
    vane.section_lines(trade_date)  # -> list[str] 主报告独立区块（零网络）
    vane.verify_pending()           # -> str  核验台账并返回摘要
"""
from __future__ import annotations

import datetime as dt

__all__ = ["daily_sync", "section_lines", "verify_pending"]
__version__ = "2.0.0"


def daily_sync(trade_date: dt.date | None = None,
               calendar: list | None = None) -> str:
    """当日同步：核验标签 → 三日门 → 模型评分主排 → 登记 → 报告 → 每日微调。

    幂等，可重复跑。守卫：trade_date 非最新已收盘交易日（历史查看）或
    盘中（<15:05）→ 跳过。模型任一环节失败回退启发式排序（留痕不静默）。
    calendar：入口层已拉的交易日历（免重复下载；None 则自取）。
    """
    from data_source import resolve_trade_date, trade_calendar

    from . import config as C
    from . import ledger, report
    from .engine import select
    from .screen import build_rows

    anchor = resolve_trade_date(None, calendar=calendar or trade_calendar())
    trade_date = trade_date or anchor
    if trade_date != anchor:
        return f"风向标跳过（{trade_date} 非最新已收盘交易日）"
    if trade_date == dt.date.today() and dt.datetime.now().time() < dt.time(15, 5):
        return "风向标跳过（盘中，收盘后运行）"

    # 0) 核验到期 pending（标签回填——模型训练数据源，先于训练）
    try:
        ledger.verify(trade_date)
    except Exception as e:                     # noqa: BLE001 核验失败不阻断
        print(f"[风向标] 核验失败（{type(e).__name__}: {e}），不影响后续")

    # 1) 三日门（自动重训/坏件自愈/回退护栏；失败→启发式排序并留痕）
    meta, mnote = None, ""
    try:
        from . import model
        meta, mnote = model.ensure_model(trade_date)
    except Exception as e:                     # noqa: BLE001
        mnote = f"模型异常（{type(e).__name__}: {e}），启发式排序"
        print(f"[风向标] {mnote}")

    # 2) 候选构建 → 模型评分主排（模型可用）→ 名额截断
    built = build_rows(trade_date)
    if built.get("error"):
        return f"风向标：{built['error']}"
    rows = built["rows"]
    if meta:
        try:
            from . import model
            rows = model.apply_ranking(rows, trade_date, meta)
            order_key = model.model_rank_key()
        except Exception as e:                 # noqa: BLE001 评分失败回退启发式
            print(f"[风向标] 评分失败（{type(e).__name__}: {e}），启发式排序")
            meta, rows = None, built["rows"]
            order_key = None
    else:
        order_key = None
    picks, bench = select(rows, top_n=C.TOP_N, board_range=C.BOARD_RANGE,
                          key=order_key)
    for i, r in enumerate(picks, 1):
        r["rank"] = i
        r["group"] = f"{r['B']}to{r['B'] + 1}"
    result = dict(built, picks=picks, bench=bench, model_meta=meta,
                  model_note=mnote)

    # 3) 登记（含 model_version/model_score/shadow_rank 归因列）+ 留痕 + 报告
    rmsg = ledger.register(trade_date, picks, regen_same_day=True)
    ledger.mark_scanned(trade_date, len(picks),
                        len(result.get("miss_days") or []))
    report.render(result)
    try:
        if meta:
            from . import model
            model.retrain_diff(rows, trade_date)
    except Exception as e:                     # noqa: BLE001 对照失败不阻断
        print(f"[风向标] 新旧对照失败（{type(e).__name__}: {e}）")

    # 4) 每日微调（影响明日；xgb warm-start / logreg 重算；幂等守卫）
    try:
        if meta:
            from . import model
            fmsg = model.refit_today(trade_date)
        else:
            fmsg = "微调跳过（无模型）"
    except Exception as e:                     # noqa: BLE001 微调失败不阻断
        fmsg = f"微调异常（{type(e).__name__}）"
        print(f"[风向标] {fmsg}")

    lad = result.get("ladder") or {}
    lad_s = "｜".join(f"{b}板×{n}" for b, n in list(lad.items())[:3])
    m_s = (f"｜模型 {meta.get('version')}·{mnote}" if meta
           else f"｜{mnote}")
    return (f"风向标同步：名额 {len(picks)}/{C.TOP_N}"
            f"｜梯队 {lad_s or '无2板+'}｜{m_s}｜{fmsg}｜{rmsg}")


def section_lines(trade_date: dt.date) -> list[str]:
    """主报告开头独立区块（零网络只读台账）：情绪风向标，关注清单同款规格。

    用户约定（2026-11-22）：单独列出来、放报告开头、显眼展示（星级+加粗，
    对齐明日关注清单视觉语言）；不并入「明日决策摘要」，也不进关注清单
    统一关注分。星级 = 模型 score100 当日归一分档（≥67★★★ / ≥34★★☆ /
    其余★☆☆）；模型未启用时不挂星（不伪造档位）。

    当日已跑但无候选 → 明示空缺（区分空间断层与模块未跑）；
    当日未跑（无台账也无 last_scan 痕迹）→ 返回 [] 整节省略。
    """
    import termstyle as T

    from . import ledger

    def _seal(v) -> str:
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return "—"

    def _fbt(v) -> str:
        v = str(v or "")
        return f"{v[:2]}:{v[2:]}" if len(v) == 4 else "—"

    def _score_num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _stars(score100):
        s = _score_num(score100)
        if s is None:
            return ""
        return T.stars3(0 if s >= 66.7 else (1 if s >= 33.4 else 2))

    rows = ledger.rows_for(trade_date)
    if not rows and not ledger.scanned_on(trade_date):
        return []
    lines = [
        T.bold("▎情绪风向标（4板晋5 / 5板晋6 · 三名额 · 明日晋级观察）"),
        "─" * 78,
    ]
    # 三日模型状态行（坏件/缺失=不显示；模型不可用日由 picks 行 model_version 空
    # 串在台账中留痕，by_model_version 对照不受影响）
    try:
        from . import model as _m
        m = _m._meta_load()
        if m and m.get("ok"):
            lines.append(
                f"  模型 {m['version']} · {m['n_rows']}行样本 · 基率 "
                f"{m['base_rate']}% · cutoff {m['cutoff']}"
                + (" · 已微调" if m.get("finetuned") else ""))
    except Exception:                          # noqa: BLE001 模型行失败不阻断
        pass
    if not rows:
        lines.append(T.dim("  （今日无 4板/5板 晋级候选——空间断层或情绪冰点，"
                           "如实空缺）"))
    for r in rows:
        stars = _stars(r.get("score100"))
        star_s = f"{stars} " if stars else ""
        ms = _score_num(r.get("model_score"))
        ms_s = f"模型分{ms:+.2f} " if ms is not None else ""
        shadow = r.get("shadow_rank")
        shadow_s = f"影子#{shadow}" if shadow else ""
        lines.append(
            f"  {star_s}#{r['rank']} {T.cyan(r['code'])} "
            f"{T.bold(r['name'])} {r['label']}(晋{int(r['B']) + 1})"
            f"  {ms_s}{shadow_s}".rstrip())
        chain = str(r.get("round_dates") or "")
        parts = chain.split("|")
        chain_s = "→".join(x[5:] if len(x) >= 10 else x for x in parts) \
            if parts and parts[0] else ""
        detail = f"封成比{_seal(r['seal_ratio'])} · {_fbt(r['fbt'])}封"
        if r.get("industry"):
            detail += f" · {r['industry']}"
        lines.append(f"     {detail}"
                     + (f" ｜ 板链 {chain_s}" if chain_s else ""))
    s = ledger.summary()
    if s.get("n"):
        rate = s.get("promote_rate")
        rate_s = f"（{rate:.0f}%）" if isinstance(rate, (int, float)) else ""
        lines.append(T.dim(f"  台账:晋级 {s['promoted']}/{s['n']}{rate_s}"
                           " · 全量详情: out/research/vane/report_latest.txt"
                           "（python -m vane report）"))
    else:
        lines.append(T.dim("  台账:待核验 · 全量详情: "
                           "out/research/vane/report_latest.txt"))
    lines.append("")                        # 与后续区块的呼吸空行
    return lines


def verify_pending(today: dt.date | None = None) -> str:
    """核验台账 pending 行（拉次日后首根日K回填晋级结果）。"""
    from . import ledger
    return ledger.verify(today)
