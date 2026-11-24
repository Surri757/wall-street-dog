# -*- coding: utf-8 -*-
"""军团自动进化引擎（2026-11-22 用户定向：内置自动化学习更新迭代优化）。

━━━ 调研结论（relay_map_fresh 月度稳定性视图，8~11月）━━━
盟友关系**非稳态**——静态名单必然过期：
- 开源太华路 ×8 全在 8 月，9/10/11 月共击 0（代表性已过期，首月重排即降出）；
- 万豪世家 2/2/2/2 月月稳定；国信浙江互联网 11 月突现 ×5（量化散户跟随）；
- 江苏路 11 月走强 ×4；国新北京 2/7/0/6 活跃但波动。
→ 成员资格由**滚动证据分**近因指数衰减自动重排，不靠人工拍板。

━━━ 自动闭环（随 daily_sync 自驱，评审段全程离线）━━━
1. 日：伙伴 CSV 保鲜（daily_sync 既有）→ 共击证据重算（离线 join）→
   fusion 共击画像刷新 → per-seat 质量权重更新
2. 月：每月首个 daily_sync 触发 review()——90 日滚动证据分（近因半衰期
   45d）→ **在任稳定条款**：挑战者分 > 在任分 × 1.5 才换血（防成员震荡，
   换血成本=squad20 先验的统计噪声）→ 产出 legion.json（成员/证据/决策
   留痕，审计可查）；新成员无明细 CSV 时按名探码（seatdata.probe_yyb_code）
   并自动拉取
3. 季：全量 relay 重发现（网络重，relay_fresh.py 手动/定期）→ 新盟友进入
   挑战者池

━━━ 评分消费约定 ━━━
- squad20 已拟合系数（+1.10）语义不动：仍为二值特征，成员清单由
  fusion.squad_csv_paths() 从 legion.json 动态读取；
- per-seat 质量权重只进 fusion 共振**建议层**（不动主甄选冻结口径）；
  样本 <MIN_QUAL_N 时不计质量（先标签后计分纪律），权重回 1.0。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

import fstore
import pandas as pd

from . import config as C
from .seatdata import probe_yyb_code

LEGION_JSON = os.path.join(C.ROOT, "out", "research", "fusion", "legion.json")
RELAY_RAW = os.path.join(C.ROOT, "out", "research", "fusion", "_relay_fresh_raw.json")

MEMBER_N = 5              # 军团定额
EVIDENCE_WINDOW = 90      # 证据回看（自然日）
HALF_LIFE = 45.0          # 近因半衰期（天）
HYSTERESIS = 1.5          # 在任稳定条款：挑战者分 > 在任分 × 此倍数才换血
MIN_QUAL_N = 10           # 质量权重起计的最小共击样本（先标签后计分）
CHANNELS = ("深股通", "沪股通", "机构专用")
# 政策性剔除（2026-09-20 relay 研究定性 + 用户确认）：总部/高盛/自贸区=通道型
# 被动承接；拉萨/互联网分公司=量化散户跟随——均无预测信息。注意国新北京
# 虽是分公司但为游资型主力（不在剔除列）。
POLICY_EXCLUDE = ("总部", "高盛", "自贸试验区", "拉萨", "互联网分公司")

# 已知盟友全名↔短名对照（relay raw 用全名、成员册用短名；防名异码同）
KNOWN_FULL = {
    "开源证券股份有限公司西安西大街证券营业部": "开源西大街",
    "国新证券股份有限公司北京分公司": "国新北京",
    "国泰海通证券股份有限公司北京知春路证券营业部": "知春路",
    "国泰海通证券股份有限公司上海长宁区江苏路证券营业部": "江苏路",
    "开源证券股份有限公司西安太华路证券营业部": "开源太华路",
    "国泰海通证券股份有限公司湛江万豪世家证券营业部": "万豪世家",
    "国信证券股份有限公司浙江互联网分公司": "国信浙江互联网",
}


# ---------------------------------------------------------------- 状态
def load_state() -> dict:
    """读军团状态。损坏文件先隔离为 .corrupt 再按空起步。

    直接「按空起步」随后 `_save_state` 覆写，会把评审留痕（reviews）连同成员
    名单一起抹掉，事后无从追查。占用类读失败（OSError）照常抛出——调用方按
    「进化失败、沿用现有名单」处理，不得在没读到旧状态时回写。
    """
    if os.path.exists(LEGION_JSON):
        try:
            with open(LEGION_JSON, encoding="utf-8") as f:
                return json.load(f)
        except ValueError as e:            # JSON 解析/解码失败 = 内容坏了
            try:
                os.replace(LEGION_JSON, LEGION_JSON + ".corrupt")
            except OSError:
                pass
            print(f"[军团] legion.json 损坏（{type(e).__name__}），已隔离为 "
                  ".corrupt，按空起步（现役回退 DEFAULT_SQUAD）")
    return {"members": {}, "challengers": {}, "reviews": [],
            "last_review": ""}


def _save_state(state: dict) -> None:
    state["updated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fstore.write_text(LEGION_JSON,
                      json.dumps(state, ensure_ascii=False, indent=2, default=str))


# ---------------------------------------------------------------- 证据
def _zy_ignitions() -> pd.DataFrame:
    """紫阳点火 (日,股) + 次日涨幅（证据 join 右表；复用 fusion 口径）。"""
    from .fusion import _zy_ignitions as _ig
    return _ig()


def _seat_cofires(code: str, ign: pd.DataFrame) -> set[tuple]:
    """某席位（按 yyb 代码）的紫阳点火同日共击事件集合（离线 join）。

    西大街（10046745）走 kaiyuan 专属档；其余在 partner_trades/。
    """
    csv = (os.path.join(C.ROOT, "out", "research", "kaiyuan", "seat_trades.csv")
           if code == "10046745"
           else os.path.join(C.ROOT, "out", "research", "fusion",
                             "partner_trades", f"{code}.csv"))
    if not os.path.exists(csv):
        return set()
    from .fusion import _dedupe_net_buys
    try:
        ev = _dedupe_net_buys(csv)
    except Exception:                      # noqa: BLE001 坏档=无证据
        return set()
    ig_pairs = {(r.交易日期, r.股票代码) for r in ign.itertuples(index=False)}
    return {(d, s) for d, s in zip(ev["交易日期"], ev["股票代码"])
            if (d, s) in ig_pairs}


def _evidence(cofires: set[tuple], ign: pd.DataFrame, asof: dt.date,
              incumbent: bool) -> dict:
    """滚动证据分：近因指数衰减为主项（调研实证：月度分布漂移是成员过期的
    首要信号）；质量项（共击日点火股 d1 中位）样本达 MIN_QUAL_N 才呈现——
    先标签后计分，不早进评分。在任者 +0.75 稳定加成（抗证据噪声）。"""
    d1_map = {(r.交易日期, r.股票代码): r.d1 for r in ign.itertuples(index=False)
              if r.d1 == r.d1}
    lo = asof - dt.timedelta(days=EVIDENCE_WINDOW)
    score = 0.0
    recent, n90 = [], 0
    for d, s in cofires:
        if not (lo <= d <= asof):
            continue
        age = (asof - d).days
        score += math.pow(0.5, age / HALF_LIFE)
        n90 += 1
        v = d1_map.get((d, s))
        if v is not None:
            recent.append(float(v))
    recent.sort()
    d1_med = round(recent[len(recent) // 2], 2) if recent else None
    return {"co_n90": n90, "co_n_total": len(cofires),
            "score": round(score + (0.75 if incumbent else 0.0), 3),
            "raw_score": round(score, 3),
            "d1_med": d1_med,
            "last_co": (max((d for d, _ in cofires if lo <= d <= asof),
                            default=None))}


# ---------------------------------------------------------------- 拉取
def _ensure_member_csv(code: str) -> bool:
    """成员明细 CSV 就位（无则拉全量；与 fusion 拉取同纪律：空/失败不落盘）。"""
    from .fusion import PARTNER_DIR
    path = os.path.join(PARTNER_DIR, f"{code}.csv")
    if os.path.exists(path):
        return True
    try:
        import akshare as ak
        df = ak.stock_lhb_yyb_detail_em(symbol=code)
        if df is None or not len(df):
            return False
        fstore.write_dataframe_csv(path, df)
        return True
    except Exception:                      # noqa: BLE001 拉取失败下月重试
        return False


# ---------------------------------------------------------------- 评审
def review(force: bool = False, today: dt.date | None = None) -> str:
    """月度军团重排（每月首个 daily_sync 自动触发；force=True 手动）。

    在任 5 席 + 挑战者池（relay raw 非通道席位，yyb 探码入池）统一按滚动
    证据分排序；在任稳定条款（HYSTERESIS）防换血震荡；决策与依据全量留痕。
    """
    today = today or dt.date.today()
    state = load_state()
    month = today.strftime("%Y-%m")
    if not force and state.get("last_review") == month:
        return f"军团评审：{month} 已完成（跳过）"

    ign = _zy_ignitions()
    members = dict(state.get("members") or {})
    challengers = dict(state.get("challengers") or {})
    if not members:                        # 首次建档：兜底名单入册
        from .fusion import DEFAULT_SQUAD
        members = {n: {"code": v["code"]} for n, v in DEFAULT_SQUAD.items()}
    if not challengers:                    # 挑战者池种子：relay raw 非通道席位
        challengers = _seed_challengers(exclude=set(members))
    # 挑战者证据就位：按 relay 频次 Top-10 探码+拉明细（一次性，结果持久化；
    # 无码/无档的挑战者证据分记 0，不参与换血判定——不 network 轰炸）
    top_c = sorted(challengers.items(),
                   key=lambda kv: -kv[1].get("relay_n", 0))[:10]
    for n, c in top_c:
        if not c.get("code"):
            c["code"] = probe_yyb_code(n, *_probe_sample(n))
        if c.get("code"):
            _ensure_member_csv(c["code"])

    def _ev(code: str | None, incumbent: bool) -> dict:
        cofires = _seat_cofires(code, ign) if code else set()
        return _evidence(cofires, ign, today, incumbent) | {"code": code}

    ev_m = {n: _ev(m.get("code"), True) for n, m in members.items()}
    ev_c = {n: _ev(c.get("code"), False) for n, c in challengers.items()}

    def key(kv):
        return (-kv[1]["score"], -kv[1]["co_n90"], kv[0])

    ranked_m = sorted(ev_m.items(), key=key)
    ranked_c = sorted(ev_c.items(), key=key)

    decisions, demoted, promoted = [], [], []

    def _admit(name, e):
        """挑战者入册前置：探码（无码）+ 查重（名异码同防重复入册）+ 拉明细。"""
        code = e.get("code")
        if not code:
            code = probe_yyb_code(name, *_probe_sample(name))
        if not code:
            decisions.append({"date": today.isoformat(), "action": "skip",
                              "seat": name, "reason": "探码失败，留池待下月"})
            return None
        if code in {m.get("code") for m in members.values()}:
            decisions.append({"date": today.isoformat(), "action": "skip",
                              "seat": name, "reason": "名异码同（已是成员）"})
            return None
        if any(p in name for p in POLICY_EXCLUDE):
            decisions.append({"date": today.isoformat(), "action": "skip",
                              "seat": name, "reason": "通道型/量化散户，政策剔除"})
            return None
        if not _ensure_member_csv(code):
            decisions.append({"date": today.isoformat(), "action": "skip",
                              "seat": name, "reason": "CSV 拉取失败，留池待下月"})
            return None
        return code

    # ① 空位补齐：成员不足定额时按证据分直接晋升（补位≠置换，无稳定条款）
    need = MEMBER_N - len(ranked_m)
    pool = list(ranked_c)
    while need > 0 and pool:
        name, e = pool.pop(0)
        code = _admit(name, e)
        if not code:
            continue
        short = KNOWN_FULL.get(name, name)      # 已知盟友以短名入册
        promoted.append((short, code))
        decisions.append({"date": today.isoformat(), "action": "promote",
                          "seat": short, "code": code, "score": e["score"],
                          "reason": (f"空位补齐：90日证据分 {e['score']}"
                                     f"（共击 {e['co_n90']}）")})
        need -= 1

    # ② 置换换血：在任末位被最强挑战者以 HYSTERESIS 倍压制 → 1:1 末位置换
    #（每届最多 2 次，防成员剧烈震荡；换血成本=squad20 先验的统计噪声）
    pool = list(ranked_c)
    swaps = 0
    while pool and swaps < 2:
        bottom = ranked_m[len(ranked_m) - 1 - swaps]
        best = pool[0]
        if best[1]["score"] <= bottom[1]["score"] * HYSTERESIS:
            break                          # 稳定条款：不足倍差不强拆在任
        code = _admit(best[0], best[1])
        if not code:
            pool.pop(0)
            continue
        demoted.append(bottom[0])
        in_short = KNOWN_FULL.get(best[0], best[0])
        promoted.append((in_short, code))
        decisions.append({"date": today.isoformat(), "action": "swap",
                          "out": bottom[0], "out_score": bottom[1]["score"],
                          "in": in_short, "in_code": code,
                          "in_score": best[1]["score"],
                          "reason": (f"90日证据分 {best[1]['score']} > "
                                     f"{bottom[1]['score']}×{HYSTERESIS}"
                                     "（近因半衰期45d，末位置换）")})
        pool.pop(0)
        swaps += 1
    if not swaps and need <= 0:
        decisions.append({"date": today.isoformat(), "action": "hold",
                          "seat": "—",
                          "reason": (f"挑战者最强 {ranked_c[0][1]['score'] if ranked_c else '—'} "
                                     f"未达在任末位 {ranked_m[-1][1]['score'] if ranked_m else '—'} "
                                     f"的 {HYSTERESIS}×，名单维持")})

    for name in demoted:
        challengers[name] = {"code": members.pop(name)["code"]}
    for name, code in promoted:
        members[name] = {"code": code}

    # 全量证据入档（成员与挑战者统一口径）
    for n, m in members.items():
        m["evidence"] = _ev(m["code"], True)
    for n, c in challengers.items():
        c["evidence"] = _ev(c.get("code"), False)

    state.update({"members": members, "challengers": challengers,
                  "last_review": month, "member_n": MEMBER_N,
                  "policy": {"half_life_days": HALF_LIFE,
                             "hysteresis": HYSTERESIS,
                             "evidence_window_days": EVIDENCE_WINDOW}})
    state["reviews"] = (state.get("reviews") or [])[-23:] + decisions
    _save_state(state)

    pm = "、".join(n for n, _ in promoted) or "无"
    dm = "、".join(demoted) or "无"
    return f"军团重排[{month}]：晋升 {pm}｜降出 {dm}｜成员 {len(members)}/{MEMBER_N}"


def _probe_sample(name: str) -> tuple[str, str]:
    """从 relay raw 取该席位的一个 (股票代码, 日期) 样本（探码入参）。"""
    if not os.path.exists(RELAY_RAW):
        return "", ""
    df = pd.read_json(RELAY_RAW, convert_dates=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    hit = df[df["partner"] == name]
    if not len(hit):
        return "", ""
    row = hit.iloc[0]
    return row["code"], str(row["day"]).replace("-", "")


def _seed_challengers(exclude: set[str]) -> dict[str, dict]:
    """挑战者池种子：relay raw 同日榜非通道席位（剔除在任），探码入池。"""
    out: dict[str, dict] = {}
    if not os.path.exists(RELAY_RAW):
        return out
    df = pd.read_json(RELAY_RAW, convert_dates=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    z = df[(df["seat"] == "ziyang") & (df["tag"] == "cur")
           & ~df["partner"].apply(lambda n: any(c in n for c in CHANNELS))
           & ~df["partner"].apply(lambda n: any(p in n for p in POLICY_EXCLUDE))]
    known_full = set(KNOWN_FULL.values())
    for name, cnt in z["partner"].value_counts().items():
        if name in exclude or name in known_full:
            continue                       # 在任（短名）或已知盟友（另有短名条目）
        out[name] = {"code": None, "relay_n": int(cnt)}
    return out


# ---------------------------------------------------------------- 消费接口
def active_squad() -> dict[str, dict]:
    """现役军团（legion.json 为准；无档案回退 DEFAULT_SQUAD 兜底）。

    返回与 fusion.DEFAULT_SQUAD 同形：{席位名: {"code":…, "csv":…}}——消费方
    （ensure_partner_trades/squad_keys/co_attack_stats）零改动换源。
    """
    state = load_state()
    members = state.get("members") or {}
    if members:
        # 有档案即用现役成员（哪怕暂缺 1 席——回退陈旧兜底反而失真）
        from .fusion import PARTNER_DIR
        out = {}
        for n, m in members.items():
            code = m.get("code")
            if code == "10046745":          # 西大街走 kaiyuan 专属档
                out[n] = {"code": code, "csv": None}
                continue
            csv = os.path.join(PARTNER_DIR, f"{code}.csv")
            out[n] = {"code": code,
                      "csv": (f"{code}.csv" if os.path.exists(csv) else None)}
        return out
    from .fusion import DEFAULT_SQUAD
    return DEFAULT_SQUAD


def seat_weights() -> dict[str, float]:
    """per-seat 质量权重（共振建议层用；样本 <MIN_QUAL_N 回 1.0）。"""
    state = load_state()
    weights = {}
    for n, m in (state.get("members") or {}).items():
        e = m.get("evidence") or {}
        d1, n90 = e.get("d1_med"), e.get("co_n90") or 0
        if d1 is not None and n90 >= MIN_QUAL_N:
            weights[n] = round(max(0.5, min(1.5, 0.75 + d1 / 4.0)), 2)
        else:
            weights[n] = 1.0
    return weights


def daily_tick() -> str:
    """daily_sync 挂点：月边界自动评审（当月已评即跳过）；其余日只读。"""
    return f"军团tick：{review()}"


def report() -> str:
    """军团进化现状（成员证据表 + 评审决策留痕）。"""
    st = load_state()
    lines = [f"成员（{len(st.get('members') or {})}/{MEMBER_N}）："]
    for n, m in (st.get("members") or {}).items():
        e = m.get("evidence") or {}
        lines.append(f"  {n}({m.get('code')}) 90日共击{e.get('co_n90', 0)}｜"
                     f"证据分{e.get('score')}｜d1中位{e.get('d1_med')}%｜"
                     f"最近共击{e.get('last_co')}")
    ch = st.get("challengers") or {}
    if ch:
        lines.append(f"挑战者池（{len(ch)}）："
                     + "、".join(sorted(ch)[:6]) + ("…" if len(ch) > 6 else ""))
    for d in (st.get("reviews") or [])[-6:]:
        lines.append(f"  [{d.get('date')}] {d.get('action')} {d.get('seat')} "
                     f"({str(d.get('reason', ''))[:44]})")
    return "\n".join(lines)
