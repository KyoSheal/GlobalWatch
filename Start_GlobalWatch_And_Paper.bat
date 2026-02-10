@echo off
title GlobalWatch V3.1.2 + Paper Trading

echo ========================================
echo Start GlobalWatch V3.1.2 and Paper Trading
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Starting GlobalWatch_V2.py (Streamlit)...
start "GlobalWatch V3.1.2" cmd /k "cd /d %~dp0 && python -m streamlit run GlobalWatch_V2.py"

echo [2/2] Starting paper_trading.py (paper_config.json)...
start "Paper Trading V3.1.2" cmd /k "cd /d %~dp0 && python -u paper_trading.py paper_config.json"

echo.
echo Both processes started in separate windows.
echo.
pause

