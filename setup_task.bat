@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 一键注册每交易日 15:10 自动复盘计划任务（main.py + calibrate.py + research_scan）
rem 与收盘保护对齐：15:05 后才接受收盘快照；需对本弹窗确认或以管理员运行
schtasks /create /tn "WallStreetDogDaily" /tr "\"%~dp0run_daily.bat\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:10 /f
if %errorlevel%==0 (
  echo.
  echo 注册成功：每交易日 15:10 自动复盘+校准，日志见 out\auto.log
  echo 查看: schtasks /query /tn "WallStreetDogDaily"
  echo 删除: schtasks /delete /tn "WallStreetDogDaily" /f
) else (
  echo.
  echo 注册失败：请右键本脚本，选择"以管理员身份运行"后重试
)
pause
