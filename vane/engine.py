# -*- coding: utf-8 -*-
"""核心纯函数：本轮板数回溯 + 入围排序。零网络零 IO，可离线单测。

口径（2026-11-19 用户拍板，预注册冻结）：
- 板 = 收盘封板（东财涨停池口径，炸板不算板）；
- 同一轮 = 相邻两板间隔 ≤ max_gap 个交易日（不含两端板日）；超过即久远
  断档、板数重置——「不一定需要连板，但前面的板必须是近期发生」的量化；
- span = 本轮首板到今日的交易日跨度（含两端），即市场惯称的「S天B板」；
- 入围：今日封板 + 主板非 ST + 本轮板数 B ∈ board_range（默认口径 4/5，
  明日晋 5 / 晋 6）；
- 排序字典序：板高降序 → 断板次数升序（同板内连板优先）→ 跨度升序
  （越紧凑越强）→ 封成比降序 → 封板时间早优先 → code 兜底保证确定性。

引擎不带业务默认值：max_gap/top_n/board_range 一律由调用方从 vane.config
传入（刻度单一事实源），测试显式传参。
"""
from __future__ import annotations


def streak(zt_dates: list[str], today: str, cal_dates: list[str],
           max_gap: int, *, pos: dict[str, int] | None = None) -> dict | None:
    """回溯个股「本轮板数」。

    zt_dates  该票窗口内全部涨停日（ISO 串，乱序/重复均可，只取 ≤ today 的部分）
    today     复盘日（ISO 串；必须本身是涨停日，否则返回 None——未封板无从谈晋级）
    cal_dates 窗口交易日（ISO 串升序，给 gap 计算提供交易日坐标）
    max_gap   同一轮内相邻两板最大间隔（交易日，不含两端板日）

    返回 {"B": 本轮板数, "span": 首板→今日交易日跨度(含两端),
          "breaks": 断板次数, "round_dates": 本轮涨停日(旧→新)}；
    今日不在涨停日集合或不在交易日坐标内时返回 None。
    """
    pos = pos if pos is not None else {d: i for i, d in enumerate(cal_dates)}
    if today not in pos:
        return None
    zs = sorted({d for d in zt_dates if d in pos and d <= today}, reverse=True)
    if not zs or zs[0] != today:
        return None
    # 从今日往前逐对回溯：相邻两板间隔 ≤ max_gap 才算同一轮，久远断档即止
    round_dates = [zs[0]]
    for i in range(1, len(zs)):
        gap = pos[zs[i - 1]] - pos[zs[i]] - 1
        if gap > max_gap:
            break
        round_dates.append(zs[i])
    round_dates.reverse()                      # 旧 → 新，首板在前
    b = len(round_dates)
    span = pos[round_dates[-1]] - pos[round_dates[0]] + 1
    breaks = sum(1 for a, nxt in zip(round_dates, round_dates[1:])
                 if pos[nxt] - pos[a] > 1)
    return {"B": b, "span": span, "breaks": breaks, "round_dates": round_dates}


def sort_key(r: dict) -> tuple:
    """排序键（合并两组统一比较；见模块 docstring 的字典序）。

    seal_ratio 约定：上游 screen 已把缺失/NaN 清成 None（脏数据单点收敛），
    此处 None → -inf 垫底；fbt 缺失 → "9999" 排最后。
    """
    sr = r.get("seal_ratio")
    sr = float(sr) if sr is not None else float("-inf")
    fbt = str(r.get("fbt") or "9999")
    return (-int(r["B"]), int(r["breaks"]), int(r["span"]), -sr, fbt, r["code"])


def select(rows: list[dict], top_n: int,
           board_range: tuple, key=None) -> tuple[list[dict], list[dict]]:
    """入围 + 换手门槛 + 排序 + 名额截断。

    key 缺省=启发式 sort_key（板高>断板>跨度>封成比）；模型主排时传入
    model.model_rank_key（分高优先、启发式兜底保确定性）。
    换手门槛（2026-11-25）：行上 turn_ok=False 的不占名额——用户的口径是
    「本轮高换手天数不足的没有参与必要」，所以它排在一切之后，进落选观察。
    没有 turn_ok 字段的行（旧数据/非 4、5 板组）按不设门槛处理。
    返回 (picks 占名额的前 top_n, bench 落选观察：名额已满的在前、门槛外的在后)。
    """
    in_range = [r for r in rows if int(r.get("B") or 0) in board_range]
    gated_out = [r for r in in_range if r.get("turn_ok") is False]
    qual = [r for r in in_range if r.get("turn_ok") is not False]
    qual.sort(key=key or sort_key)
    return qual[:top_n], qual[top_n:] + sorted(gated_out, key=sort_key)


def label(r: dict) -> str:
    """"6天4板断1" 式市场口径标签（无断板即纯「4天4板」连板形态）。"""
    s = f"{r['span']}天{r['B']}板"
    if r["breaks"]:
        s += f"断{r['breaks']}"
    return s
