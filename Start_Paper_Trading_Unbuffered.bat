@echo off
REM ============================================================
REM 无缓冲启动纸上交易 - 解决 Windows Terminal 延迟问题
REM ============================================================

echo ============================================================
echo Starting Paper Trading (Unbuffered Output)
echo ============================================================
echo This version forces immediate output display
echo to prevent delays in Windows Terminal
echo ============================================================
echo.

REM 使用 -u 参数强制无缓冲输出
python -u paper_trading.py paper_config.json

echo.
echo ============================================================
echo Trading session ended
echo ============================================================
pause