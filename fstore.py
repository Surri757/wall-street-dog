# -*- coding: utf-8 -*-
"""文件仓读写助手（零依赖叶子模块）。

项目无 SQL 库，数据仓=CSV/pkl/json 文件；全量重写是最常见写模式
（台账/缓存/模型/归档）。直接覆写中途断电或杀进程会留下截断文件——
预注册台账被 csv.DictReader 静默读出半行、pkl 缓存作废重拉、模型文件
损坏。统一走「临时文件 + os.replace」：replace 在同一文件系统上是原子
操作，读者要么看到完整旧版、要么看到完整新版，永无半截状态。

**全项目落盘只走这一个模块**：业务代码不得再各写一套 tmp+replace
（2026-11-24 合并前 vane/ziyang/主干各有一份，行为已开始分叉——行尾、
扩展名保留、占用重试三处互不一致）。

write_with_bak 额外保留上一代 .bak，用于不可再生产物（模型/画像）——
save_model 直接覆盖唯一真源，坏一次全丢。
"""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import io
import json
import os
import pickle
import time
from pathlib import Path

LOCK_STALE_SECONDS = 1800.0   # 运行锁龄期上限：长于最长复盘会话（含 --fill）


def _tmp_of(path: Path) -> Path:
    """同目录临时名，**保留原扩展名**：xgboost.save_model 等按扩展名选格式
    的写方（.json→文本 / 其他→UBJSON 二进制）写到 `x.tmp` 会静默变成二进制
    ——2026-11-22 model_xgb.json 事故根因。命名为 `x.tmp.json`。"""
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _replace(tmp_path: str, path: str) -> None:
    # Windows：目标正被另一进程读（pickle.load/csv 打开中）时 os.replace 抛
    # PermissionError——短重试覆盖读写交叠窗口，避免毁掉整轮已完成的分析
    for i in range(3):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if i == 2:
                raise
            time.sleep(0.5)


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """原子写文本（json/txt/报告/CSV 等全量重写）。

    换行归一：文本模式写入会把 "\\n" 翻译成 os.linesep，而 csv 模块与
    DataFrame.to_csv 的输出**自带** "\\r\\n"——直接写就落成 "\\r\\r\\n"
    （2026-11-24 实测：全部台账/归档 CSV 每行多一个冗余字节 CR）。先把既有
    "\\r\\n" 归一成 "\\n" 再写：落盘后每行恰好一个行终止符（Windows=CRLF），
    纯 "\\n" 文本（报告/JSON）的字节保持不变。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(p)
    tmp.write_text(text.replace("\r\n", "\n"), encoding=encoding)
    _replace(str(tmp), str(p))


def write_dataframe_csv(path: str, df) -> None:
    """原子写 DataFrame 为 CSV（utf-8-sig，与 write_csv 的读写契约一致）。

    不要再各处手写 `write_text(path, df.to_csv(index=False, encoding="utf-8-sig"))`：
    to_csv **返回字符串时 encoding 是空操作**，那句代码写出的是无 BOM 文件，
    与它自己的意图相反（2026-11-24 实测：日度信号归档无 BOM、台账有 BOM）。
    """
    write_text(path, df.to_csv(index=False), encoding="utf-8-sig")


def write_json(path: str, obj, indent: int = 1) -> None:
    """原子写 JSON（ensure_ascii=False：中文直存，台账/模型可读可 grep）。"""
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path: str, default=None):
    """读 JSON 文件；不存在或损坏返回 default（调用方按「无该工件」处理）。

    需要区分「损坏」并留痕的调用方（如模型旧件归档）自行 read_text +
    json.loads 捕获原异常——这里只服务「没有就当没有」的读点。
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_pkl(path: str, data) -> None:
    """原子写 pickle（滚动缓存）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(p)
    tmp.write_bytes(pickle.dumps(data))
    _replace(str(tmp), str(p))


def write_csv(path: str, cols: list, rows: list[dict],
              quoting: int = csv.QUOTE_MINIMAL) -> None:
    """原子写 CSV（台账全量重写；utf-8-sig 与项目读写契约一致）。"""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, quoting=quoting)
    w.writeheader()
    w.writerows(rows)
    write_text(path, buf.getvalue(), encoding="utf-8-sig")


def _write_tmp(path: str, writer) -> tuple[str, str]:
    """写方落临时文件 → (临时文件, 目标)；由调用方决定是否备份后 replace。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_of(p)
    writer(str(tmp))
    return str(tmp), str(p)


def write_via(path: str, writer) -> None:
    """原子写，内容由 writer(tmp_path) 自行序列化（不保留 .bak）。

    写方按 tmp_path 落盘的场合用：DataFrame.to_csv(path)、pickle.dump 等
    不宜先整份序列化成字符串的大对象。
    """
    tmp, target = _write_tmp(path, writer)
    _replace(tmp, target)


def write_with_bak(path: str, writer) -> None:
    """原子写并保留上一代 .bak（不可再生产物：模型/画像）。

    writer(tmp_path) 负责把内容写到 tmp；本函数先备份旧版再 replace。
    """
    tmp, target = _write_tmp(path, writer)
    if os.path.exists(target):
        try:
            os.replace(target, target + ".bak")
        except OSError:
            pass                    # 旧版被占用/只读：放弃备份不阻断写入
    _replace(tmp, target)


@contextlib.contextmanager
def run_lock(guard_dir: str, stale_seconds: float = LOCK_STALE_SECONDS):
    """复盘运行互斥守卫：定时任务（15:10）与手动 work/main 并存时，两进程
    读-改-写同一缓存 pkl 会互相吞掉对方当日结果。

    O_EXCL 锁文件 + 龄期清理：崩溃残留的锁 stale_seconds（默认 30 分钟）后
    自动失效。刻意不做 os.kill(pid,0) 存活检测——Windows 上它会直接终止
    目标进程而不是探测。
    """
    os.makedirs(guard_dir, exist_ok=True)
    lock = os.path.join(guard_dir, ".run.lock")
    acquired = False
    for _ in range(3):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"pid={os.getpid()} started={dt.datetime.now():%H:%M:%S}")
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock)
            except OSError:
                age = stale_seconds + 1     # 锁文件被并发删掉 → 视为可抢
            if age > stale_seconds:
                try:
                    os.remove(lock)
                except OSError:
                    pass
                continue
            time.sleep(1.0)                 # 多数交叠在秒级收尾，短候重试
    if not acquired:
        raise RuntimeError(
            "另一复盘进程正在运行（缓存目录 .run.lock 新鲜）——两进程并发会互吞"
            "当日缓存；确认无并发后可删除该锁文件重跑")
    try:
        yield
    finally:
        if acquired:
            try:
                os.remove(lock)
            except OSError:
                pass
