# -*- coding: utf-8 -*-
"""渲染层：情绪风向标独立报告（out/research/vane/report_latest.txt 纯文本）。

契约：result 为 screen.scan 的返回 dict；error 结果只返回文本不落盘——
report_latest.txt 始终保留最近一次成功筛选的完整报告。
"""
from __future__ import annotations

import fstore

from . import config as C
from . import ledger as LG

RULE_W = 68


def _fmt_seal(v) -> str:
    """封成比显示（None=缺失，上游 screen 已把 NaN 统一清成 None）。"""
    return f"{v:.2f}" if v is not None else "—"


def _ltsz_yi(v) -> str:
    """流通市值（亿元）显示。"""
    return f"{v / 1e8:.0f}亿" if v is not None else "—"


def _fbt_hhmm(v: str) -> str:
    return f"{v[:2]}:{v[2:]}" if v else "—"


def _pick_line(r: dict) -> list[str]:
    """单个名额两行：概要行 + 板链行。"""
    head = (f"  #{r['rank']} {r['code']} {r['name']:<8} {r['label']}"
            f"(晋{r['B'] + 1})  封成比{_fmt_seal(r.get('seal_ratio'))}"
            f" · {_fbt_hhmm(r.get('fbt') or '')}封"
            f" · {_ltsz_yi(r.get('ltsz'))}"
            + (f" · {r['industry']}" if r.get("industry") else ""))
    # ISO 日期取 [5:] → "MM-DD"；板链直观展示断板分布
    chain = "→".join(x[5:] for x in r["round_dates"])
    return [head, f"     板链 {chain}"]


def render(result: dict) -> str:
    """渲染完整报告文本；成功结果落盘、error 结果只返回文本（保住上次好报告）。"""
    if result.get("error"):
        return f"━━━ 情绪风向标 ━━━\n{result['error']}\n"
    d = result["trade_date"]
    meta = result.get("model_meta")
    lines = [
        "━━━ 情绪风向标 · " + d + " ━━━",
        "─" * RULE_W,
        f"口径：明晋5(4板)与明晋6(5板)合并取前{C.TOP_N}｜板=收盘封板｜同轮断板间隔"
        f"≤{C.MAX_GAP}个交易日｜仅主板非ST",
        ("排序：模型主排（三日模型）｜影子=板高 > 断板少 > 跨度紧凑 > 封成比"
         if meta else
         "排序：板高 > 断板少 > 跨度紧凑 > 封成比 > 封板时间（启发式）"),
        "",
        "▎名额（下一交易日 " + d + " 后首个交易日晋级观察）",
    ]
    # 三日模型状态行（模型不可用=启发式排序，如实呈现）
    if meta:
        lines.append(
            f"  模型 {meta.get('version')} · 三日门 · 样本 {meta.get('n_rows')} 行"
            f"（正例 {meta.get('n_pos')}·基率 {meta.get('base_rate')}%）"
            f" · cutoff {meta.get('cutoff')} · {meta.get('backend', meta.get('kind'))}"
            + (" · 已微调" if meta.get("finetuned") else ""))
    else:
        lines.append(f"  模型: 未启用（{result.get('model_note') or '样本门未过'}）"
                     "——启发式排序（shadow_rank 已登记供对照）")
    picks = result["picks"]
    if not picks:
        lines.append("  （今日无 4板/5板 晋级候选——空间断层或情绪冰点，如实空缺）")
    for r in picks:
        lines += _pick_line(r)
        if r.get("model_score") is not None:
            lines.append(f"     模型分 {r['model_score']:+.2f}"
                         + (f"（score100 {r.get('score100')}｜贡献 {r.get('contribs')}）"
                            if r.get("contribs") else "")
                         + f"  影子#{r.get('shadow_rank', '—')}")
    if result["bench"]:
        lines.append("")
        lines.append("▎落选观察（已入围但名额已满）")
        for r in result["bench"][:6]:
            lines.append(f"  · {r['code']} {r['name']:<8} {r['label']}"
                         f"(晋{r['B'] + 1})  封成比{_fmt_seal(r.get('seal_ratio'))}")
        if len(result["bench"]) > 6:
            lines.append(f"  …另有 {len(result['bench']) - 6} 只")
    ladder = result.get("ladder") or {}
    if ladder:
        lad_s = "｜".join(f"{b}板×{n}" for b, n in ladder.items())
        lines.append("")
        lines.append("▎高标梯队（本轮板数≥2，主板）  " + lad_s)
    miss = result.get("miss_days") or []
    if miss:
        lines.append(f"⚠ 窗口内 {len(miss)} 天涨停池拉取失败"
                     f"（{','.join(m[5:] for m in miss[:4])}...），板数可能被低估")
    s = LG.summary()
    if s.get("n"):
        lines.append("")
        lines.append(f"台账累计：晋级 {s['promoted']}/{s['n']}"
                     f"（{s['promote_rate']:.0f}%，{s['n_days']} 个交易日）"
                     + (f" · 次日均幅 {s['next_pct_mean']:+.2f}%"
                        if s.get("next_pct_mean") is not None else ""))
    lines.append("")
    lines.append("提示：情绪风向标为观察口径（空间/分歧锚点），不构成买入指令。")
    txt = "\n".join(lines) + "\n"
    fstore.write_text(C.REPORT_TXT, txt)    # 原子写：报告可重生成，仍防半截文件
    return txt
