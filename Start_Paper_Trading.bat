@echo off
REM ============================================================
REM 直接启动纸上交易 - 避免 PowerShell 复杂性
REM ============================================================

title Paper Trading - Running

echo ============================================================
echo Starting Paper Trading (Direct Mode)
echo ============================================================
echo.
echo This version runs directly without PowerShell
echo to avoid any input waiting issues
echo.
echo Press Ctrl+C to stop the trading session
echo ============================================================
echo.

REM 直接启动 Python 程序（无缓冲模式）
python -u paper_trading.py paper_config.json

echo.
echo ============================================================
echo Trading session ended
echo ============================================================
pause