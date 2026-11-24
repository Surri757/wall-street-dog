# -*- coding: utf-8 -*-
"""一键环境：在项目目录下创建 .venv 并安装全部依赖。

只用标准库——它跑在「依赖还没装」的那个 Python 上。依赖清单见 requirements.txt。

用法：
    Windows        双击 setup.bat（work.bat 首次运行也会自动调用）
    macOS / Linux  python3 setup_env.py

安装源顺序：环境变量 PIP_INDEX_URL（如果你设了）→ PyPI 官方源 → 清华镜像。
每个源先用 pip 自己做一次探测（限时下载 pip 自身的小包），不通或太慢就换下一个。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(ROOT, "requirements.txt")
VENV_DIR = os.path.join(ROOT, ".venv")
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
PROBE_TIMEOUT = 45          # 秒：探测下载（pip 的 wheel ≈ 2MB）的墙钟上限
PIP_COMMON = ["--disable-pip-version-check", "--timeout", "30"]


def venv_python() -> str:
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def candidate_indexes() -> list:
    """安装源候选（保序去重）；None 表示 pip 的默认源（PyPI 官方）。"""
    env_idx = os.environ.get("PIP_INDEX_URL", "").strip()
    out, seen = [], set()
    for idx in ([env_idx] if env_idx else []) + [None, MIRROR]:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def pip_cmd(py: str, index, *args: str) -> list:
    cmd = [py, "-m", "pip", *args, *PIP_COMMON]
    return cmd + ["-i", index] if index else cmd


def reachable(py: str, index) -> bool:
    """用 pip 自己做探测：限时下载一个小包。失败或超时都视为这个源不可用。"""
    with tempfile.TemporaryDirectory() as td:
        cmd = pip_cmd(py, index, "download", "--no-deps", "-q", "-d", td,
                      "--timeout", "10", "--retries", "0", "pip")
        try:
            r = subprocess.run(cmd, cwd=ROOT, timeout=PROBE_TIMEOUT, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def main() -> int:
    py = venv_python()
    if not os.path.exists(py):
        print(f"[环境] 创建虚拟环境 .venv（Python {sys.version.split()[0]}）…")
        venv.create(VENV_DIR, with_pip=True)
    for index in candidate_indexes():
        print(f"[环境] 安装源：{index or 'PyPI 官方源'}")
        if not reachable(py, index):
            print("       不通或太慢，换下一个源")
            continue
        if subprocess.run(pip_cmd(py, index, "install", "-q", "--upgrade", "pip"),
                          cwd=ROOT, check=False).returncode != 0:
            print("       pip 自身升级失败，换下一个源")
            continue
        if subprocess.run(pip_cmd(py, index, "install", "-r", REQ),
                          cwd=ROOT, check=False).returncode == 0:
            print()
            print("[环境] 依赖安装完成。")
            print("       Windows：双击 work.bat 开始复盘")
            print("       macOS / Linux：.venv/bin/python work.py")
            return 0
        print("       安装失败，换下一个源重试")
    print()
    print("[失败] 依赖安装失败：请检查网络后重新运行 setup.bat（或 python setup_env.py）。")
    return 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:                                   # noqa: BLE001
        print(f"[失败] {type(e).__name__}: {e}")
        code = 1
    if code and sys.stdin is not None and sys.stdin.isatty():
        input("按回车退出…")
    raise SystemExit(code)
