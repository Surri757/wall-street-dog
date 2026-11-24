# -*- coding: utf-8 -*-
"""轻量进度条：百分比精确到两位小数，tty 下 \r 行内原地刷新。

慢循环（逐股拉日K/算前夜特征/翻页取快照）全程可见，消灭「最后一行进度
停半天」的假死观感（2026-09-17 用户被 [画像] 日K 150/168 卡住后设立）。

行为约束：
- 真终端：整行原地刷新（含 ETA）；重定向/管道：每 25 步打一行，不刷屏；
- total<=0（空循环）零输出；只做显示，不碰任何数据口径。
"""
from __future__ import annotations

import sys
import time

_BAR_W = 24           # 进度条宽度（格）
_TTY_MIN_DT = 0.05    # tty 最小刷新间隔（秒），防极快循环刷爆控制台
_FILE_EVERY = 25      # 非 tty 每 N 步输出一行
_PAD = 92             # tty 帧最小宽度，覆盖上一帧残尾


def _hms(sec: float) -> str:
    sec = int(sec)
    if sec >= 3600:
        return "%d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)
    return "%02d:%02d" % (sec // 60, sec % 60)


def _dur(sec: float) -> str:
    return _hms(sec) if sec >= 10 else "%.1fs" % sec


class Progress:
    """循环进度。用法：

        pg = Progress("画像·日K", len(codes))
        for i, code in enumerate(codes, 1):
            do(code)
            pg.update(i, extra=code)
        pg.done()                      # 收尾一行 100.00% + 用时
    """

    def __init__(self, tag: str, total: int):
        self.tag = tag
        self.total = int(total)
        self._n = 0
        self._t0 = time.time()
        self._last = 0.0
        self._done = False
        self._tty = (self.total > 0 and hasattr(sys.stdout, "isatty")
                     and sys.stdout.isatty())

    def _line(self, n: int, extra: str = "") -> str:
        pct = 100.0 * n / self.total
        filled = int(round(_BAR_W * n / self.total))
        s = "  [%s] %s %d/%d %6.2f%%" % (
            self.tag, "█" * filled + "░" * (_BAR_W - filled), n, self.total, pct)
        return s + (" " + extra if extra else "")

    def _eta(self, now: float) -> str:
        el = now - self._t0
        if self._n <= 0 or el < 0.5 or self._n >= self.total:
            return ""
        return " eta %s" % _hms(el / self._n * (self.total - self._n))

    def update(self, n: int, extra: str = "") -> None:
        if self._done or self.total <= 0:
            return
        self._n = n
        now = time.time()
        if self._tty:
            if now - self._last < _TTY_MIN_DT and n != self.total:
                return
            self._last = now
            line = (self._line(n, extra) + self._eta(now)).ljust(_PAD)
            print("\r" + line, end="", flush=True)
        elif n % _FILE_EVERY == 0 and n != self.total:   # 末行留给 done()
            print(self._line(n, extra) + self._eta(now), flush=True)

    def done(self, note: str = "") -> float:
        """收尾：定格 100.00% + 用时（+ 可选备注），返回耗时秒。"""
        el = time.time() - self._t0
        if self._done or self.total <= 0:
            return el
        self._done = True
        line = self._line(self.total) + ((" " + note) if note else "") \
            + "（%s）" % _dur(el)
        print("\r" + line if self._tty else line, flush=True)
        return el
