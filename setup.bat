@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem 一键环境：找到 Python 3.10+ 后交给 setup_env.py 建虚拟环境(.venv)并装好依赖
set "PYCMD="
py -3.12 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD py -3.11 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul && set "PYCMD=py -3.11"
if not defined PYCMD py -3.13 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>nul && set "PYCMD=python"
if not defined PYCMD (
  echo [失败] 没找到 Python 3.10 及以上。
  echo        请先安装：https://www.python.org/downloads/
  echo        Windows 安装时勾选 "Add python.exe to PATH"，装完重新打开本窗口再试。
  if not defined WALLSTREET_BOOTSTRAP pause
  exit /b 1
)
%PYCMD% setup_env.py
if errorlevel 1 (
  echo.
  echo 环境安装失败，见上方提示。
  if not defined WALLSTREET_BOOTSTRAP pause
  exit /b 1
)
if not defined WALLSTREET_BOOTSTRAP pause
endlocal
