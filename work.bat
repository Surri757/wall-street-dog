@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem 解释器选择：① 项目自带 .venv → ② 系统 python（依赖齐全时）→ ③ 首次运行自动装环境
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY python -c "import importlib.util as u; raise SystemExit(0 if all(u.find_spec(m) for m in 'pandas numpy requests akshare baostock xgboost sklearn'.split()) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
  echo [环境] 首次运行：正在自动创建虚拟环境并安装依赖。只需这一次，请稍候...
  set "WALLSTREET_BOOTSTRAP=1"
  call setup.bat
  if not exist ".venv\Scripts\python.exe" (
    echo [失败] 环境安装失败。也可手动执行：python setup_env.py
    pause
    exit /b 1
  )
  set "PY=.venv\Scripts\python.exe"
)
"%PY%" work.py
endlocal
