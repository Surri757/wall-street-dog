# -*- coding: utf-8 -*-
"""终端着色（ANSI）。Windows 终端/新版 cmd 原生支持；不支持时设 NO_COLOR=1 或
自动检测失败降级为纯文本（颜色只做强调，信息不依赖颜色）。"""
from __future__ import annotations

import os
import re
import sys

_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

# ANSI 转义序列（着色码）在宽度计算与落盘时都必须视为不可见
ESC = "\x1b"   # ANSI 转义前缀：源码里不写裸控制字符
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text) -> str:
    """去掉 ANSI 颜色码，得到可直接落盘/检索的纯文本。"""
    return _ANSI_RE.sub("", str(text))


def weekday_cn(d) -> str:
    """日期 → 周几（中文）。统一实现——此前 7 处两种手抄写法易漂移。"""
    return "周" + "一二三四五六日"[d.weekday()]


def _c(code: str, text: str) -> str:
    if not _ENABLED:
        return text
    return f"{ESC}[{code}m{text}{ESC}[0m"


def red(t):    return _c("91", t)
def green(t):  return _c("92", t)
def yellow(t): return _c("93", t)
def cyan(t):   return _c("96", t)
def bold(t):   return _c("1", t)
def dim(t):    return _c("2", t)
def magenta(t): return _c("95", t)


def stars(score: int) -> str:
    """评级着色：三星绿/两星黄/一星黄。"""
    if score >= 85:
        return green("★★★")
    if score >= 78:
        return yellow("★★☆")
    if score >= 70:
        return yellow("★☆☆")
    return dim("···")


# 情绪周期六色：高潮=品红(狂热) 发酵=绿(生长) 回暖=青 回暖、分歧=黄(警示)、
# 冰点=灰(冰冷)、退潮=红(危险)。一眼区分阶段。
STAGE_STYLE = {
    "高潮": magenta, "发酵": green, "回暖": cyan,
    "分歧": yellow, "冰点": dim, "退潮": red,
}


def stage(text: str) -> str:
    """按文本中的周期关键词整体着色（如"回暖期（赚钱效应+0.4%）"→青色）。"""
    for k, fn in STAGE_STYLE.items():
        if k in str(text):
            return fn(text)
    return str(text)


def stage_fn(stage_name: str):
    """取阶段对应的着色函数（仓位条等局部着色用）；未知阶段返回 None。"""
    for k, fn in STAGE_STYLE.items():
        if k in str(stage_name):
            return fn
    return None


def stars3(level: int) -> str:
    """档位星级三色：0(★★★)=绿（可买/最优） 1(★★☆)=黄（中档） 2(★☆☆)=灰（弱/观察）。"""
    if level <= 0:
        return green("★★★")
    if level == 1:
        return yellow("★★☆")
    return dim("★☆☆")


def _narrow_ambiguous(c: str) -> bool:
    """East Asian Ambiguous 宽度中、实际渲染为**窄**（1 列）的字符。

    `ord(c) > 127` 一律按 2 列算是错的：制表符与块元素在 Windows Terminal +
    中日韩字体下渲染 1 列。旧实现只特判了 `·`，导致报告表头框线
    `╔ + ═*76 + ╗` 被算成 156 列、而内容行 `║ + pad(s,76) + ║` 是 80 列，
    两者永远对不齐，分隔线宽度也与内容不符。

    ⚠ 不要顺手把 ★(U+2605) 等 Miscellaneous Symbols 也算窄——它们在中日韩字体下
    确实是全宽，改了会让现有的星级列对齐反而错位。
    """
    o = ord(c)
    return (c == "·"                       # U+00B7 中点
            or 0x2500 <= o <= 0x257F       # Box Drawing：─ ║ ╔ ╗ ╚ ╝ ╠ ╣
            or 0x2588 <= o <= 0x259F)      # Block Elements：█ ░ ▎


def disp_width(s) -> int:
    """终端显示宽度：中文/全角按 2 列；ambiguous 窄字符（制表符/块元素/·）按 1 列。

    先剥掉 ANSI 颜色码——着色后的字符串若不剥，转义序列会被逐字符计入宽度，
    导致 pad 出来的列在开启颜色时整体错位。
    """
    w = 0
    for c in strip_ansi(s):
        if _narrow_ambiguous(c):
            w += 1
        elif ord(c) > 127:
            w += 2
        else:
            w += 1
    return w


def pad(s, width: int) -> str:
    """按显示宽度右侧补空格（跨模块共用的对齐工具）。"""
    s = str(s)
    return s + " " * max(0, width - disp_width(s))


# ---------------------------------------------------------- 折行（跨模块共用）
# 放在 termstyle 而不是 report：report 依赖 ledger，ledger 不能反向依赖 report，
# 而两者都要同一套折行逻辑 → 共用实现只能落在展示层工具模块。

WRAP_W = 96          # 默认目标显示宽度（报告分隔线 78，正文允许略宽但不破百）
# 可在其后断行的中文标点（标点留在上一行）。含破折号与间隔号：
# 缺了它们，「…多重比较——本区块仅供数据记录，不构成买入依据。」会因搜不到断点
# 而硬断在逗号前，把「，」甩成下一行开头。
_BREAK_AFTER = "、，。；：）】」》—·"


def wrap(text, width: int = WRAP_W, indent: str = "      ") -> list[str]:
    """按**显示宽度**折行（中文占 2 列，故不能用 len）。

    断点优先级：空格之后 → 中文标点之后 → 硬断。中文没有空格，只靠硬断会把
    「涨/幅」劈开、把「，不构成买入依据。」甩成孤儿行、让续行以顿号开头。
    首行不带 indent，续行带——调用方若要整体缩进，只给首行补。

    两个易错点（都已修，改这里务必保持）：
    - 在断点处折行时只能 append 断点**之前**的部分。曾误写成 append 整个 `cur`，
      而 `cur` 已含断点之后的尾巴、尾巴又被带到下一行 → 那段文字印两遍
      （实测「…胜率100% 按日 / 按日等权净值+60.53%」，「按日」重复）。
    - 判断断点位置必须用 disp_width 而不是字符下标：中文每字符占 2 列，
      用下标比宽度会让条件几乎永不成立，退化成逐字硬断。
    """
    text = str(text)
    if disp_width(text) <= width:
        return [text]
    out, cur = [], ""
    for ch in text:
        if cur and disp_width(cur + ch) > width:
            cut = -1
            # 从后往前找断点，但不早于当前行的一半（避免首行过短）
            lo = max(0, len(cur) // 2)
            for i in range(len(cur) - 1, lo - 1, -1):
                if cur[i] == " " or cur[i] in _BREAK_AFTER:
                    cut = i
                    break
            if cut >= 0:
                head = cur[:cut + 1].rstrip()      # 空格丢弃，标点留在上一行
                tail = cur[cut + 1:] + ch
                if (disp_width(head) >= width * 0.5
                        and disp_width(indent + tail) <= width):
                    out.append(head)
                    cur = indent + tail
                    continue
            out.append(cur)
            cur = indent + ch
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def wrap_segs(segs, width: int = WRAP_W, first_prefix: str = "",
              indent: str = "      ", sep: str = " │ ") -> list[str]:
    """把「段」列表贪心装行，段间用 sep 连接，尽量不在段中间断开。

    first_prefix 用于首行前缀（如明细行的 "└ "），其宽度计入首行预算。
    是否插 sep 由 **added 标志**决定，不能用 `cur.strip()` 判断——传了 first_prefix
    时 cur 一开始就非空，会让第一个段前面也多出一个 sep（实测「└  │ 题材:…」）。
    """
    out, cur, added = [], first_prefix, False
    for s in segs:
        s = str(s).strip()
        if not s:
            continue
        # 首段要带上 first_prefix（如「└ 」），否则前缀会被整个丢掉
        cand = (first_prefix + s) if not added else cur + sep + s
        if added and disp_width(cand) > width:
            out.append(cur)
            cur = indent + s
        else:
            cur = cand
        added = True
    if added:
        out.append(cur)
    return out


def dim_wrap(text, indent: str = "  ", width: int = WRAP_W) -> list[str]:
    """暗色脚注：**先折行再逐行着色**，且只给首行补缩进。

    顺序不能反——对着色文本折行会把 ANSI 重置码切到错误的行上。
    缩进只能加一次：wrap 已经给续行加了 indent，这里若再对每行都加一遍，
    续行缩进就会翻倍（实测 4 空格变 8 空格）。
    """
    parts = wrap(str(text).strip(), width - disp_width(indent), indent)
    if not parts:
        return []
    return [dim(indent + parts[0])] + [dim(p) for p in parts[1:]]
