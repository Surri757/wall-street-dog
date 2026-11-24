# -*- coding: utf-8 -*-
"""选股规则层（L2）：开盘买入池档位 / 可成交打板画像 / +5% 备选档位。

与渲染解耦的纯规则（输入 result dict 行 → 输出档位/清单），report 与 ledger 共用。
**无未来函数声明（2026-09-04 审计）**：所有筛选字段均为当日收盘可知
（zt_status/vol_ratio/open/prev_close/lianban/close）；
唯一口径近似：打板画像的 vr 为全日量比，盘中执行时以触板时点实时量比近似
（缩量板封板后成交占比小，误差有限——文档标注，非泄露）。
"""
from __future__ import annotations

from config import (BOARD_GAP_BEST, EXPECT, ONE_WORD_GAP,
                    OPEN_POOL_VR_MIN, SLAM_2BOARD_BOOST, SLAM_GAP_MAX,
                    SLAM_VR_MAX, SLAM_VR_MIN)


def open_gap(x) -> float:
    """今日开盘缺口（开盘/昨收-1）；数据异常返回 9.9（视同一字，保守排除）。"""
    try:
        pc = float(x.get("prev_close") or 0)
        return float(x["open"]) / pc - 1.0 if pc > 0 else 9.9
    except (TypeError, ValueError, KeyError):
        return 9.9


def gap_score(gap: float) -> tuple[int, str]:
    """高开缺口 → C 维得分与标签（单一实现，second_board 消费）。

    此前两处各自解析同一个 BOARD_GAP_BEST 却给出**相反极性**（second_board：
    两套矛盾的高开指令。收敛到这里，重校时两处同时生效。

    ⚠ 当前极性按「晋级率」校准（2026-08-30，106 条样本内）：准一字晋级率 58.8%。
    但**可成交口径下方向相反**——events.csv 2999 事件 / 241 交易日，按交易日聚类：
    准一字 开→收 −2.11%（t=−9.22）、缩量 −0.98%、高开优 −0.81%、其余 +0.23%（t=+1.33）；
    可成交子样本内给分最低的「其余」组反而 +0.57%（t=+2.86）。
    且 gap_next≥0.09 与 one_word_next 是同一集合（n=684），准一字组根本买不到。
    以可成交口径重校 C 维是待办，见 out/research/snapshots/2026-09-09/MANIFEST.md。
    """
    lo, hi = BOARD_GAP_BEST
    # 一字/准一字档统一 >= ONE_WORD_GAP（此前此处用 >，与 open_pool_of 的 >=
    # 在 gap 恰为 0.09 时结论相反——同一票打分给"高开"、开盘池却按一字剔除）
    if gap >= ONE_WORD_GAP:
        return 10, "准一字"
    if lo <= gap <= hi:
        return 7, f"高开{gap:+.1%}优"
    if 0.01 <= gap < lo or hi < gap < ONE_WORD_GAP:
        return 6, f"高开{gap:+.1%}"
    if gap > 0:
        return 3, f"高开{gap:+.1%}弱"
    return 3, f"低开{gap:+.1%}"


def expect_label(key: str, which: str = "current") -> str:
    """从 config.EXPECT 渲染史测期望标签，如 "+0.4%/38%"。

    which="current" 取最新实测（缺失时回退 prereg）；"prereg" 取预注册值。
    渲染层与判读层一律走这里——同一指标此前在 config/board_rules/report/
    calibrate/research_log 里有 3~4 个互相矛盾的版本。
    """
    e = EXPECT.get(key) or {}
    v = e.get(which) or e.get("prereg") or e.get("current")
    return f"{v[0]:+.1f}%/{v[1]:.0f}%" if v else "—"


def expect_verdict(key: str) -> str:
    """EXPECT 条目的判读：live / dead / unproven / unreproducible（未知按 unproven）。"""
    return (EXPECT.get(key) or {}).get("verdict", "unproven")


def open_pool_tier(lb: int, vr: float, gap: float):
    """开盘买入池档位（开→开 = T+1 开盘买 → T+2 开盘卖；完整12个月 n=8098）。

    返回 (档位0最优, 期望标签)；排除格返回 (None, "")：
    - 今日一字/准一字（-1.05%，由调用方先行剔除）
    - vr≤0.8 全体（缩量票开盘买 -0.8~-2.1%，归 +5% 附录）
    - 3板+·vr>1.5（高位放量过度 -1.4~-2.3%）
    各分支返回的是 12 个月网格的**分格**期望；★★★ 汇总值只在 config.EXPECT
    ["open_pool_t0"] 里维护（五R 终审已判 dead：t=-0.19 对称硬币，130 格无一
    通过多重比较），不要在渲染层另写字面量。
    """
    if lb >= 3:
        return (0, "+1.5%/62%") if 0.8 < vr <= 1.5 else (None, "")
    if lb == 2 and 0.8 < vr <= 1.5:
        return (0, "+0.6%/51%") if gap < 0.03 else (1, "+0.3%/48%")
    if lb == 2 and vr > 2.5:
        return 2, "-0.6%/36%·备选"
    if vr > 2.5:
        return 2, "+0.0%/44%"
    return 2, "-0.4%/44%·备选"


def open_pool_of(rows):
    """开盘买入池规则（主区块与台账回放共用同一口径；行可为 result dict 或归档行）。

    返回 (池列表[行+_otier], (一字数, 缩量数, 高位放量数))。行字段需含
    zt_status/vol_ratio/open/prev_close/lianban（字符串数字亦可）。
    """
    out, n_ow, n_shrink, n_hi = [], 0, 0, 0
    for r in rows:
        if r.get("zt_status") != "封板":
            continue
        try:
            vr = float(r.get("vol_ratio"))
            o_ = float(r.get("open"))
            pc = float(r.get("prev_close"))
            lb = int(float(r.get("lianban") or 0))
        except (TypeError, ValueError):
            continue
        if not (vr == vr and pc > 0):
            continue
        gap = o_ / pc - 1.0
        if gap >= ONE_WORD_GAP:
            n_ow += 1                     # 今日一字：明日竞价难成交
            continue
        if vr <= OPEN_POOL_VR_MIN:
            n_shrink += 1                 # 缩量票：归 +5% 附录
            continue
        tier = open_pool_tier(lb, vr, gap)
        if tier[0] is None:
            n_hi += 1                     # 3板+·vr>1.5
            continue
        x = dict(r)
        x["_otier"] = tier
        out.append(x)
    return out, (n_ow, n_shrink, n_hi)


def slam_candidates(rows):
    """可成交打板画像（用户前提「买得到」：实体开盘·温和缩量·盘中触板）。

    单一实现——报告区块与台账共用（此前两处重复实现，2026-09-04 审计收敛）。

    ⚠ 期望数字有两套口径，别混：
      未接成交模型（五M，12个月全体事件）：1板 +2.63%/81%、2板 +4.43%/84%、3板 +4.53%/85%
      接了成交模型（fills.csv 1414 事件，唯一可执行口径）：成交率仅 **30.7%**
        （一字买不到 17.2%），成交单 n=434 均值 +0.39%、**中位 -1.27%**、胜率 38%，
        120 日簇聚类 t=1.26（p≈0.21，不显著）。结构是二元赌局：
        封板持有 253 笔 +5.14%/65% vs 炸板止损 181 笔 -6.24%/胜率 0%。
    展示一律走 config.EXPECT["slam"]，不要在渲染层写字面量。
    """
    out = []
    for r in rows:
        if r.get("zt_status") != "封板":
            continue
        try:
            vr = float(r.get("vol_ratio"))
        except (TypeError, ValueError):
            continue
        if vr != vr:
            continue
        if SLAM_VR_MIN < vr <= SLAM_VR_MAX and open_gap(r) < SLAM_GAP_MAX:
            out.append(r)
    out.sort(key=lambda x: (-(int(x.get("lianban") or 0)), x.get("vol_ratio", 9)))
    # 2板+ 分层标记。仅作台账分层统计用：五Q 记 n=49，但 fills.csv/events.csv 都无
    # lianban 列，本地不可复算，不足以支撑「仓位加倍」这类强指令，故只标优先级。
    if SLAM_2BOARD_BOOST:
        lab = expect_label("slam_2board")
        verd = expect_verdict("slam_2board")
        tail = {"unreproducible": "n=49·本地不可复算", "unproven": "样本不足",
                "dead": "已判死"}.get(verd, "")
        for x in out:
            if int(x.get("lianban") or 0) >= 2:
                x["_slam_boost"] = f"2板+·史测{lab}" + (f"·{tail}" if tail else "")
    return out


def watch_tier(gap: float, vr: float) -> tuple[int, str, str]:
    """+5% 备选档位（缩量票口径，12个月 n=899 分档实测，研究日志五G补2）。

    返回 (档位0最优, 形态标签, 给买点标签)：
      ★★★ 实体·0.5~0.8：主力粮仓，80%给买点，全胜+9.4%的主力来源(n=500)
      ★★★ 高开3-9·≤0.3：稀有金矿，全过率11.8%全场最高(n=17，给买点仅35%)
      ★★☆ 高开3-9·0.3~0.8：次优，灰区需第二关把关
      ★☆☆ 实体·≤0.5：买得到但两关全过率≈0（86样本0次全过），末位
    """
    solid = gap < 0.03
    if solid and 0.5 < vr <= 0.8:
        return 0, "实体", "给买点~80%"
    if not solid and vr <= 0.3:
        return 0, "高开", "给买点~35%·全过率最高"
    if not solid:
        return 1, "高开", "给买点~60%"
    if vr <= 0.5:
        return 2, "实体", "给买点~74%·全过≈0"
    return 2, "实体", "给买点~57%"
