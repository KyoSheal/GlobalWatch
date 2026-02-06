@echo off
title GlobalWatch V2.8 + Paper Trading

echo ========================================
echo Start GlobalWatch V2.8 and Paper Trading
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Starting GlobalWatch_V2.py (Streamlit)...
start "GlobalWatch V2.8" cmd /k "cd /d %~dp0 && python -m streamlit run GlobalWatch_V2.py"

echo [2/2] Starting paper_trading.py (paper_config.json)...
start "Paper Trading V2.8" cmd /k "cd /d %~dp0 && python -u paper_trading.py paper_config.json"

echo.
echo Both processes started in separate windows.
echo.
pause
