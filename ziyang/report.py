# -*- coding: utf-8 -*-
"""报告层：纯文本报告（画像要点 + 明日预判清单 + 台账准确率）。

落盘纯文本不含 ANSI 码（与主工程 report 约定一致）；按显示宽度折行。
"""
from __future__ import annotations

import datetime as dt

import fstore

from . import config as C
from . import ledger as L
from .profile import load_profile

RULE = "─" * 78


def save(txt: str) -> None:
    """报告落盘（原子写）——render 之后唯一的落盘出口，各 CLI 共用。"""
    fstore.write_text(C.REPORT_TXT, txt)


def _w(s: str, width: int = 76) -> list[str]:
    """中文按 2 宽折行。"""
    out, line, w = [], "", 0
    for ch in str(s):
        cw = 2 if ord(ch) > 0x2E80 else 1
        if w + cw > width:
            out.append(line)
            line, w = "", 0
        line += ch
        w += cw
    if line:
        out.append(line)
    return out


def render(pred: dict | None = None, accuracy: dict | None = None,
           surprises: list[dict] | None = None) -> str:
    p = load_profile()
    lines = [RULE, f"{C.SEAT_SHORT}席位画像 · {dt.date.today()}", RULE]

    if p:
        st, eve = p.get("style", {}), p.get("eve", {})
        lines += ["▎席位画像（%s 生成）" % p.get("generated_at", "?"), ""]
        if st:
            daily = st.get("daily_netbuy_yi", {})
            fwd = st.get("fwd_after_netbuy", {})
            f1 = fwd.get("d1", {})
            lines += _w(
                "  %s｜画像窗口 %s 起 %d 交易日 %d 事件；净买 %d / 净卖 %d，"
                "净买主板占比 %s%%，买入日涨停率 %s%%（打板型）；"
                "单日净买均值 %s 亿（最高 %s 亿）"
                % (C.SEAT_SHORT, st.get("window", {}).get("since"),
                   st.get("window", {}).get("days", 0), st.get("window", {}).get("events", 0),
                   st.get("buy_events", 0), st.get("sell_events", 0),
                   st.get("buy_mainboard_pct"), st.get("buy_day_limit_pct"),
                   daily.get("mean"), daily.get("max")))
            if f1:
                lines += _w(
                    "  其净买入事件后市：次日 %+0.2f%%（胜率 %s%%，n=%s）、"
                    "3日 %s、5日 %s、10日 %s —— 短线资金，肉在次日"
                    % (f1.get("mean", 0), f1.get("win"), f1.get("n"),
                       _f(fwd.get("d3")), _f(fwd.get("d5")), _f(fwd.get("d10"))))
        if eve and eve.get("dist"):
            d = eve["dist"]
            for s in (
                "  点火前夜画像（n=%d 事件 / %d 日 / %d 只）：%s"
                % (eve.get("n_events", 0), eve.get("n_days", 0),
                   eve.get("n_stocks", 0), _fmt_dist(d.get("lianban"))),
                "    涨幅档：%s" % _fmt_dist(d.get("pct")),
                "    量能档：%s ｜ 换手档：%s"
                % (_fmt_dist(d.get("vr")), _fmt_dist(d.get("turnover"))),
                "    价位档：%s ｜ 20日位置：%s"
                % (_fmt_dist(d.get("price")), _fmt_dist(d.get("pos20"))),
            ):
                lines += _w(s)
        th = p.get("theme", {})
        if th.get("top"):
            lines += _w("  近60日净买行业 Top：%s"
                        % "、".join(f"{k}×{v}" for k, v in th["top"][:8]))
        # 画像迭代（2026-11-22 用户定向）：7日 vs 30日 双窗对比，随 daily_sync
        # 每日重建自动迭代——细描方向由此可观测（轨迹=profile_history.csv）
        tr = p.get("trend") or {}
        if tr.get("w7"):
            d7, d30 = tr["w7"], tr.get("w30") or {}
            dl = tr.get("delta") or {}

            def _d(v):
                return f"{v:+.1f}" if isinstance(v, (int, float)) else "—"

            lines += _w(
                "  画像迭代（7日 vs 30日）：点火率 %s%% vs %s%%（Δ%s）｜"
                "0板点火 %s%% vs %s%%（Δ%s）｜力度中位 %s%% vs %s%%"
                % (d7.get("ignition_rate_pct"), d30.get("ignition_rate_pct"),
                   _d(dl.get("ignition_rate_pct")),
                   d7.get("zero_board_share_pct"), d30.get("zero_board_share_pct"),
                   _d(dl.get("zero_board_share_pct")),
                   d7.get("power_median_pct"), d30.get("power_median_pct")))
        # 深度画像（2026-09-20 五路深挖新增；全部来自 profile.json，随 update 刷新）
        es, lz = p.get("eve_split", {}), p.get("ltsz", {}).get("dist", {})
        pw, sl = p.get("power", {}), p.get("sell", {})
        rg, tm = p.get("regime", {}), p.get("timing", {})
        if es or lz:
            lines += ["", _w("  —— 深度画像（五路深挖 · 详见 RESEARCH.md 三.3~3.8）")[0]]
            if es:
                for k, v in es.items():
                    pos = v["dist"].get("pos20", {})
                    hi = pos.get("高位>0.85", {}).get("pct", 0)
                    lines += _w("  %s：n=%d，净买中位 %.2f 亿，前夜高位占比 %s%%"
                                % (k, v.get("n", 0), v.get("net_buy_yi_median", 0) or 0, hi))
            if lz:
                lines += _w("  市值档：%s（百亿以上 %.0f%%，<30亿 %.0f%%）"
                            % (_fmt_dist(lz),
                               sum(v["pct"] for b, v in lz.items()
                                   if b in ("100~200亿", ">200亿")),
                               sum(v["pct"] for b, v in lz.items() if b == "<30亿")))
            if pw.get("dist"):
                lines += _w("  点火力度：净买/当日成交中位 %s%%（引导型点火，非锁筹）"
                            % pw.get("median_pct"))
            if sl:
                hd, hr = sl.get("holding_dist", {}), sl.get("holding_ret", {})
                a5 = (sl.get("after_sell", {}) or {}).get("d5", {})
                if hd:
                    lines += _w("  卖出纪律：持仓中位 %s 日、T+1 兑现占配对 %d/%d；"
                                "持有收益中位 %s%%/胜率 %s%%；卖后5日中位 %s%%（卖得准）"
                                % (hd.get("median_days"), hd.get("T+1", 0),
                                   sl.get("n_pairs", 0), hr.get("median"),
                                   hr.get("win"), a5.get("median")))
            if rg:
                dt_ = rg.get("daily_tickets", {})
                lines += _w("  节奏：出勤 %s%%（%s/%s 日），单日中位 %s 只·峰值 %s 只，"
                            "连买率 %s%%（一日游为主）"
                            % (rg.get("attendance_pct"), rg.get("active_days"),
                               rg.get("total_days"), dt_.get("p50"), dt_.get("max"),
                               rg.get("repeat_buy_pct")))
            if tm.get("matched"):
                lines += _w("  时点（归档交集 n=%s）：封板中位 %s，午前占 %d/%d，炸板 0 次占 %s%%"
                            % (tm.get("matched"), tm.get("fbt_median"),
                               (tm.get("fbt_dist", {}).get("早盘<10:00", 0)
                                + tm.get("fbt_dist", {}).get("10:00~11:30", 0)),
                               tm.get("matched"), tm.get("zhaban_0_pct")))
        lines.append("")

    if pred:
        preds = pred.get("predictions", [])
        lines += ["▎明日预判：%s可能点火拉板的主板票（top %d / 宇宙 %d 只）" % (C.SEAT_SHORT,
                  len(preds), pred.get("universe_n", 0)), ""]
        if not preds:
            lines.append("  （无候选）")
        for p2 in preds:
            lines += _w("  #%-2d %s %s  v2分%3d（%.2f｜影v1.4:%s名%.2f｜联#%s）  %s"
                        % (p2.get("rank", 0), p2["code"], p2["name"],
                           p2.get("score100", 0), p2.get("score", 0),
                           p2.get("shadow_rank", "-"), p2.get("shadow_score", 0.0),
                           p2.get("joint_rank", "-"),
                           L.eve_tags(p2)))
            if p2.get("contrib"):
                detail = "、".join(f"{f}({v:+})" for f, v in p2["contrib"])
                lines += _w("      主贡献：" + detail)
        att = pred.get("attendance") or {}
        if att and not att.get("error"):
            layer = pred.get("stratum") or "normal"
            tip = {"low_att": "（照常出票，建议仓位减半）",
                   "return_day": "（连缺2日回归标注：照常出票，预期强度中低）",
                   "normal": ""}.get(layer, "")
            lines += _w("  出勤层（改动五标注，不改排序）：%s%s active10=%s 连缺=%s P(出勤)=%s"
                        % (layer, tip, att.get("active10"), att.get("streak_abs"),
                           att.get("p_attend") if att.get("p_attend") is not None else "-"))
        rd = pred.get("retrain_diff")
        if rd:
            lines += _w("  月度重训首日新旧 top3 对照：旧(%s)=%s → 新(%s)=%s（重叠 %d/3）"
                        % (rd.get("prev_version"), "、".join(rd.get("prev_top3", [])),
                           rd.get("new_version"), "、".join(rd.get("new_top3", [])),
                           rd.get("overlap_n", 0)))
        sb = pred.get("scanbase") or {}
        if sb:
            refill = ("｜补拉成功率 %s%%（%s/%s）" % (sb.get("refill_rate"),
                                                     sb.get("refill_ok"),
                                                     sb.get("stale_n"))
                      if sb.get("refill_rate") is not None else "")
            lines += _w("  扫描基（改动四）：超集=%s｜陈旧票 %s 只%s"
                        % ("开" if sb.get("superset") else "关（部署门槛未过，仅监控）",
                           sb.get("stale_n"), refill))
        lines += ["  口径：候选=主板非ST且（今日涨幅≥%.0f%% 或 近20日封板 或 5日涨幅≥%.0f%% 或 横盘抬头）；分=日内分位logistic（%s·%d特征含题材/股性/复买/通道，月度滚动重训）；v1.4对数提升度为影子并行"
                  % (C.CANDIDATE_PCT_MIN, C.CANDIDATE_M5_MIN, _model_ver(), _model_feats()),
                  "  ⚠ 预判≠买入指令；龙虎榜只披露触发条件的交易，未上榜≠没买。台账裁判准确性。", ""]

    if accuracy is not None:
        lines += ["▎预判台账（先标签后计分，判读门槛=独立预判日≥%d）" % C.MIN_INFER_DAYS, ""]
        if not accuracy.get("n_rows"):
            # n_rows 是 accuracy_summary 有核验样本时的标志；"n" 键只在空账时存在
            # （2026-09-18 修复：旧条件 get("n",0)==0 让已核验账永远渲染成"暂无"）
            lines.append("  " + str(accuracy.get("note") or "暂无可核验样本"))
        else:
            bo = accuracy.get("by_outcome", {})
            lines += _w("  已核验 %d 条 / %d 个独立日：点火命中 %d、席位买入未板 %d、"
                        "他资点火 %d、全落空 %d"
                        % (accuracy.get("n_rows", 0), accuracy.get("n_days", 0),
                           bo.get("ignited", 0), bo.get("bought", 0),
                           bo.get("limit_only", 0), bo.get("miss", 0)))
            lines += _w("  precision@%d（top%d 中席位实际净买占比）=%s%%｜点火命中率=%s%%"
                        % (C.TOP_N, C.TOP_N, accuracy.get("precision_at_n"),
                           accuracy.get("ignition_hit")))
            stt = accuracy.get("strata") or {}
            if stt:
                nor, low, ret = (stt.get("normal") or {}, stt.get("low_att") or {},
                                 stt.get("return_day") or {})
                if nor.get("n_rows"):
                    lines += _w("  双口径簿记（改动五）：主门 precision（normal 层·判读口径）=%s%%（%s行/%s日）"
                                "｜全体（EV 诊断口径）=%s%%（%s行）——两数并列不互换"
                                % (nor.get("precision_at_n"), nor.get("n_rows"),
                                   nor.get("n_days"), accuracy.get("precision_at_n"),
                                   accuracy.get("n_rows")))
                if low.get("n_rows"):
                    lines += _w("  low_att 反向监控：low_att 层 seat_bought 命中率=%s%%（%s行/%s日）"
                                "——高于 normal 层说明降档过严"
                                % (low.get("precision_at_n"), low.get("n_rows"),
                                   low.get("n_days")))
                if ret.get("n_rows"):
                    lines += _w("  return_day 层：precision=%s%%（%s行/%s日，连缺2日回归标注）"
                                % (ret.get("precision_at_n"), ret.get("n_rows"),
                                   ret.get("n_days")))
            base, rec = accuracy.get("baseline") or {}, accuracy.get("recall") or {}
            if base.get("n"):                    # 预注册基线：主板封板股次日席位净买率
                lift = float(accuracy.get("precision_at_n") or 0) - float(base.get("pct") or 0)
                lines += _w("  基线：T日主板封板 %d 只·席位次日净买 %d 只（%s%%）"
                            "→ 提升 %+0.1f pp" % (base["n"], base["hit"],
                                                  base.get("pct"), lift))
            if rec.get("n"):                     # 预注册召回：席位次日净买落入 topN
                lines += _w("  召回：席位次日主板净买 %d 件·落入 top%d %d 件（%s%%）"
                            % (rec["n"], C.TOP_N, rec["caught"], rec.get("pct")))
            lines += _w("  top%d 次日平均涨幅 %+0.2f%%（胜率 %s%%）｜判读：%s"
                        % (C.TOP_N, accuracy.get("next_pct_mean", 0) or 0,
                           accuracy.get("next_pct_win"), accuracy.get("verdict")))
            ex = accuracy.get("exec") or {}
            if ex.get("n_ok"):
                oo = ex.get("oo") or {}
                cl = ("；簇槽位均值 %+0.2f%%（%s 个独立日，EXEC 主指标口径）"
                      % (oo.get("cluster_mean"), oo.get("cluster_days"))
                      if oo.get("cluster_mean") is not None else "")
                lines += _w("  执行模拟（扣费后净值，核验日开→次日开，双边 %.3f%%）：开盘口径 中位 %s%%/胜率 %s%%（n=%s）%s｜"
                            "+5%%突破口径 中位 %s%%/胜率 %s%%｜未触发/一字 %s 笔"
                            % (C.EXEC_COST_ROUND_TRIP,
                               oo.get("median"), oo.get("win"), oo.get("n"), cl,
                               ex["b5"].get("median"), ex["b5"].get("win"),
                               ex.get("n_skip")))
            bv = accuracy.get("by_version") or {}
            if len(bv) > 1:
                ver_s = "｜".join(
                    f"{k}: p@3={v['precision_at_n']}%/{v['n_days']}日"
                    for k, v in bv.items())
                lines += _w("  版本分层：" + ver_s)
            ap = accuracy.get("autopsy") or {}
            if ap.get("n_buys"):
                sp = ap.get("split", {})
                lines += _w("  漏判归因：席位真实买 %d 只 → 命中 %d｜排序漏 %d（中位第%s名）｜宇宙漏 %d"
                            % (ap["n_buys"], sp.get("命中", 0), sp.get("排序漏", 0),
                               ap.get("missed_rank_median"), sp.get("宇宙漏", 0)))
                if ap.get("miss_n"):
                    lines += _w("  机会成本：漏判票核验日中位 %s%%、隔夜开→开中位 %s%%（n=%s）"
                                "——对照登记票执行中位，排序力另一半证据"
                                % (ap.get("miss_pct_median"), ap.get("miss_oo_median"),
                                   ap.get("miss_n")))
        lines.append("")

    # 在持清单（R6a 零模型产品）+ 卖侧纪律（deep4 卖侧席实证）
    try:
        from . import ledger as _lg
        nc = _lg.nowcast()
        if nc:
            lines += ["", "▎紫阳当前在持清单（nowcast·持有≤10日可视窗）", ""]
            for h in nc[:12]:
                mark = "⚠兑现窗" if h["risk_window"] else ""
                pnl = f"{h['pnl']:+.1f}%" if h.get("pnl") is not None else "?"
                lines.append("  %s %s 持%s日 浮%s %s亿 %s"
                             % (h["code"], h["name"], h["hold_days"], pnl,
                                h["net_yi"], mark))
            if len(nc) > 12:
                lines.append(f"  …另有 {len(nc)-12} 张")
            lines += _w("  卖侧纪律（deep4 实证）：它净卖的票 77% 次日低开、持到 D+5 衰减 "
                        "-1.4%——持仓者次日竞价走；接它的卖出票至少等 D+3 后见企稳"
                        "（低点中位 D+4，D+2 反弹多为逃命波）")
    except Exception:                            # noqa: BLE001
        pass

    if surprises is not None:
        missed = [s for s in surprises if not s.get("caught")]
        caught = [s for s in surprises if s.get("caught")]
        lines += ["▎当日席位真实主板净买（召回对照）", ""]
        if not surprises:
            lines.append("  （当日无主板净买上榜）")
        for s in caught:
            lines.append("  ✓命中 %s %s  净买%.2f亿" % (s["code"], s["name"], s["net_buy_yi"]))
        for s in missed:
            lines.append("  ✗漏判 %s %s  净买%.2f亿" % (s["code"], s["name"], s["net_buy_yi"]))
        if surprises:
            lines += _w("  召回 %d/%d" % (len(caught), len(surprises)))
        lines.append("")

    # 运维体检（路线图#7：零网络，让断档自己喊出来）+ 改动四②管道健康双指标
    try:
        import pandas as _pd
        from .seatdata import (archive_signal_files, archive_union_codes,
                               last_trade_date)
        import os as _os, glob as _glob
        checks = []
        lt = last_trade_date()
        led = L._load()
        if led:
            latest_pd = max(r["predict_date"] for r in led)
            checks.append(("台账最新预判日", latest_pd,
                           "⚠落后" if latest_pd != str(lt) else "OK"))
        if _os.path.exists(C.SEAT_TRADES_CSV):
            st = _pd.read_csv(C.SEAT_TRADES_CSV, usecols=["交易日期"],
                              encoding="utf-8-sig")
            st_max = str(_pd.to_datetime(st["交易日期"]).max())[:10]
            checks.append(("席位明细末日期", st_max,
                           "⚠断更" if st_max < str(lt) else "OK"))
        sigs = archive_signal_files()
        if sigs:
            d0, f0 = sigs[-1]
            try:
                with open(f0, encoding="utf-8-sig") as _fh:
                    n_rows = sum(1 for _ in _fh) - 1
            except Exception:                  # noqa: BLE001
                n_rows = -1
            checks.append(("归档日行数(%s)" % d0, n_rows,
                           "⚠arch_low" if 0 <= n_rows < C.ARCHIVE_ROWS_HEALTH_ANCHOR
                           * C.HEALTH_ALERT_FRAC else "OK"))
        uni = archive_union_codes(lt, C.SCAN_BASE_ARCHIVE_DAYS)
        checks.append((f"{C.SCAN_BASE_ARCHIVE_DAYS}自然日归档并集", len(uni),
                       "⚠uni_low" if len(uni) < C.UNION_HEALTH_ANCHOR
                       * C.HEALTH_ALERT_FRAC else "OK"))
        ths = len(_glob.glob(_os.path.join(C.DATA_DIR, "ths_pool", "*.csv")))
        checks.append(("THS 快照累计", f"{ths} 天", "OK"))
        # 单项工件损坏只缺该项，不再让整节体检静默消失
        prof = fstore.read_json(C.PROFILE_JSON)
        if isinstance(prof, dict):
            gen = str(prof.get("generated_at", "?"))
            checks.append(("画像生成", gen[:16], "OK"))
        m = fstore.read_json(_os.path.join(C.DATA_DIR, "model", "model.json"))
        if isinstance(m, dict):
            checks.append(("模型", f'{m.get("version","?")}@{str(m.get("cutoff","?"))}',
                           "OK"))
        lines += ["", "▎运维体检（零网络）", ""]
        for name, val, flag in checks:
            lines.append(f"  {name}: {val} [{flag}]")
    except Exception:
        pass

    lines += [RULE, "紫阳东路已接入主流程（work/main 自动同步）；其余席位仅画像挖掘。不构成投资建议。", RULE]
    return "\n".join(lines)


def _model_ver() -> str:
    from .scorer import MODEL_VERSION
    return MODEL_VERSION


def _model_feats() -> int:
    from .scorer import LOGIT_FEATS
    return len(LOGIT_FEATS)


def _f(d: dict) -> str:
    return "%+0.2f%%/%s%%(n=%s)" % (d.get("mean", 0), d.get("win"), d.get("n")) if d else "-"


def _fmt_dist(d: dict | None) -> str:
    if not d:
        return "-"
    return " ".join(f"{k}:{v['pct']}%" for k, v in d.items() if v.get("n"))
