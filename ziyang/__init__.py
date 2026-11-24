# -*- coding: utf-8 -*-
"""紫阳东路席位预判模块。

预测对象：国泰海通证券武汉紫阳东路证券营业部（东财席位代码 10026937）
「隔天点火拉板」的主板票。

━━━ 已接入主工程（2026-09-17 用户批准，约定：只挂 report 层渲染 + 摘要行） ━━━

    work / python main.py 复盘最新交易日时自动调用 daily_sync()：
    席位明细全量刷新 → 画像重算 → 核验到期预判 → 生成当日 top3 甄选（台账登记）。
    历史日期查看与盘中自动跳过；失败只打一行警告，不阻断主流程。
    report.py 的「明日决策摘要」开头通过 selection_line() 渲染紫阳甄选行（零网络）。

    独立运行仍有完整 CLI：
    python -m ziyang update | predict | verify | daily | report

━━━ 程序化接口（供主工程/研究脚本调用） ━━━

    ziyang.daily_sync(trade_date)     # -> str  全模块同步（幂等，含当日重登）
    ziyang.selection_line(trade_date) # -> str|None  摘要甄选行（只读台账）
    ziyang.get_predictions()          # -> list[dict]  即时预判（纯读，不写台账）
    ziyang.verify_pending()           # -> str          核验台账并返回摘要
    ziyang.profile_summary()          # -> dict         画像摘要
    ziyang.prediction_accuracy()      # -> dict         台账准确率汇总
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import fstore

__all__ = ["daily_sync", "selection_line", "selection_rows", "get_predictions",
           "verify_pending", "profile_summary", "prediction_accuracy"]
__version__ = "1.0.0"


def daily_sync(trade_date: dt.date | None = None) -> str:
    """全模块同步：席位明细 → 画像 → 核验 → 当日甄选登记。幂等，可重复跑。

    守卫：trade_date 非最新已收盘交易日（历史查看）或盘中（<15:05）→ 跳过。
    当日重跑（晚间 LHB 发布后）会重登当日甄选（核验前刷新，纪律允许）。
    返回串尾带各阶段耗时分解（5min 预算的日常观测点）。
    """
    import time as _time

    from . import ledger
    from .profile import build_profile
    from .screen import predict
    from .seatdata import (fetch_snapshot_any, float_shares_map, industry_map,
                           last_trade_date, load_bars_cache, now_cn,
                           save_bars_cache, update_seat_trades)

    anchor = last_trade_date()
    trade_date = trade_date or anchor
    if trade_date != anchor:
        return f"紫阳同步跳过（{trade_date} 非最新已收盘交易日）"
    today = dt.date.today()
    if trade_date == today and now_cn().time() < dt.time(15, 5):
        return "紫阳同步跳过（盘中，收盘后运行）"

    t: dict = {}
    t0 = _time.time(); seat = update_seat_trades(); t["席位"] = _time.time() - t0
    t0 = _time.time(); snap = fetch_snapshot_any(); t["快照"] = _time.time() - t0
    # 步骤隔离（路线图#7）：verify 失败不得阻断预判登记——两者数据上无依赖
    # （verify 只回填台账、predict 只读 profile+seat），断一天=丢一个不可补的独立日
    cache = load_bars_cache()
    t0 = _time.time()
    try:
        profile = build_profile(seat, cache, float_shares_map(snap),
                                industry_map(snap))
    except Exception as e:                      # noqa: BLE001 画像失败不阻断预判
        # 步骤隔离同核验：predict 读的是上一版 profile.json，画像重建抛错
        # 不该连带当日预判登记一起丢（断一天=丢一个不可补的独立日）
        profile = {}
        print(f"[紫阳] 画像重建失败（{type(e).__name__}: {e}），预判沿用上一版画像")
    save_bars_cache(cache)
    t["画像"] = _time.time() - t0
    try:
        from .seatdata import snapshot_ths_pool
        snapshot_ths_pool()                     # 题材资产保全（一年滚动窗口）
    except Exception as e:                      # noqa: BLE001 快照失败不阻断主链
        print(f"[紫阳] THS 快照失败（{type(e).__name__}），次日需补拉")
    # H3 修复（deep7 资产盘点）：题材特征表须随 THS 快照同步重建——此前只在
    # 手动跑 -m ziyang.themed 时更新，daily_sync 不触发 → 特征表静默失鲜
    try:
        from .themed import build_features
        build_features()
    except Exception as e:                      # noqa: BLE001 特征失败不阻断主链
        print(f"[紫阳] 题材特征重建失败（{type(e).__name__}），特征表可能滞后")
    try:
        from .fusion import ensure_partner_trades
        ensure_partner_trades()                 # 军团 CSV 保鲜（路线图#2）
    except Exception as e:                      # noqa: BLE001 军团旧档不阻断主链
        print(f"[紫阳] 军团刷新失败（{type(e).__name__}），先验沿用旧档")
    try:
        from . import legion
        print(f"[军团] {legion.daily_tick()}")  # 月边界自动重排（评审段全离线）
    except Exception as e:                      # noqa: BLE001 进化失败沿用现名单
        print(f"[军团] 进化失败（{type(e).__name__}: {e}），沿用现有名单")
    vmsg = "核验跳过（异常）"
    try:
        t0 = _time.time(); vmsg = ledger.verify(today, seat); t["核验"] = _time.time() - t0
    except Exception as e:                      # noqa: BLE001
        print(f"[紫阳] 核验失败（{type(e).__name__}: {e}）——不阻断预判登记，明日重跑补账")
    t0 = _time.time()
    pred = predict(snapshot=snap)
    rmsg = ledger.register(pred["trade_date"], pred["predictions"],
                           regen_same_day=True, ranked=pred["all"])
    t["预判"] = _time.time() - t0
    eve = profile.get("eve", {})
    brk = "/".join("%s%.0fs" % (k, v) for k, v in t.items())
    summary = (f"紫阳同步：席位明细 {len(seat)} 事件｜画像点火 {eve.get('n_events', 0)}｜"
               f"{vmsg}｜{rmsg}｜耗时 {sum(t.values()):.0f}s（{brk}）")
    # 耗时落盘（路线图#7：只 print 不落盘=断档全静默）。追加日志不走原子写
    # ——原子写是整文件替换，追加只需要单行完整；with 保证句柄释放。
    try:
        from . import config as _C
        log = Path(_C.DATA_DIR) / "daily_sync.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{now_cn():%Y-%m-%d %H:%M:%S} {summary}\n")
    except Exception:                            # noqa: BLE001 日志失败不影响主链
        pass
    return summary


def selection_rows(trade_date: dt.date) -> list[dict]:
    """结构化甄选行（code/name/score/rank/eve_tags），供主报告关注清单消费。

    零网络只读台账；渲染层把它当作关注清单的一个「模式」接入统一关注分。
    """
    from . import ledger
    return ledger.rows_for(trade_date)


def selection_line(trade_date: dt.date) -> str | None:
    """主报告摘要行（零网络）：当日登记的甄选 + 台账累计。无登记返回 None。

    行宽约束：主报告版面上限 100 显示列（有测试守卫）。TOP_N=3 时单行放得下；
    扩 5 后（2026-09-28）自动折两行：前三只一行、其余+台账尾巴一行。
    四态明细去 ziyang report 看，摘要只给点火命中数。
    """
    from . import ledger
    rows = ledger.rows_for(trade_date)
    if not rows:
        # 弃权日：台账无登记但弃权标记存在——显示弃权行而非整行消失
        # （区分"今天选择不出票"与"模块没跑"，v3.0.2）。标记文件缺失/损坏
        # 一律按「无弃权标记」处理，不阻断主报告。
        from . import config as _C
        hist = fstore.read_json(_C.ABSTAIN_HIST_FILE)
        abstain = hist.get("_abstain") if isinstance(hist, dict) else None
        reason = (abstain or {}).get(trade_date.isoformat())
        if reason:
            return f"  紫阳甄选: 弃权·弱信号日不出票（{reason}）"
        return None
    parts = [f"{r['code']} {r['name']}·{float(r['score']):.1f}" for r in rows]
    acc = ledger.accuracy_summary()
    bo = acc.get("by_outcome", {})
    n = acc.get("n_rows", 0)
    tail = (f"（台账:点火命中{bo.get('ignited', 0)}/已核{n}）" if n
            else "（台账:待核验）")
    if len(parts) <= 3:
        return "  紫阳甄选: " + "｜".join(parts) + tail
    return ("  紫阳甄选: " + "｜".join(parts[:3]) + "\n  "
            + "｜".join(parts[3:]) + tail)


def get_predictions(trade_date: dt.date | None = None, top_n: int | None = None) -> list[dict]:
    """明日预判 topN（纯读，不写台账）。"""
    from .screen import predict
    return predict(trade_date, top_n)["predictions"]


def verify_pending(today: dt.date | None = None) -> str:
    """核验台账 pending 行（内部自动拉最新席位明细）。返回摘要文本。"""
    from . import ledger
    from .seatdata import update_seat_trades
    seat = update_seat_trades()
    return ledger.verify(today, seat)


def profile_summary() -> dict:
    """画像摘要（无画像返回空 dict，不抛异常——渲染层可直接判空跳过）。"""
    from .profile import load_profile
    return load_profile() or {}


def prediction_accuracy() -> dict:
    """台账准确率汇总（先标签后计分，簇数不足门槛时只报数不判读）。"""
    from . import ledger
    return ledger.accuracy_summary()
