@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist out mkdir out
rem 解释器选择同 work.bat：.venv → 系统 python → 自动装环境（无人值守，不暂停）
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY python -c "import importlib.util as u; raise SystemExit(0 if all(u.find_spec(m) for m in 'pandas numpy requests akshare baostock xgboost sklearn'.split()) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
  set "WALLSTREET_BOOTSTRAP=1"
  call setup.bat
  set "PY=.venv\Scripts\python.exe"
)
rem 日志轮转：超过 1MB 转存 .old（保留上一份），防止无限增长
if exist out\auto.log (
  for %%F in (out\auto.log) do if %%~zF gtr 1048576 move /y out\auto.log out\auto.log.old
)
rem 每日收盘自动复盘 + 增量校准 + 研究缓存滚动（计划任务一键注册见 setup_task.bat）
"%PY%" main.py >> out\auto.log 2>&1
"%PY%" calibrate.py >> out\auto.log 2>&1
"%PY%" research_scan.py --update >> out\auto.log 2>&1
endlocal
