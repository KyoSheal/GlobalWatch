@echo off
title GlobalWatch V3.1.2 + Paper Trading

echo ========================================
echo Start GlobalWatch V3.1.2 and Paper Trading
echo ========================================
echo.

cd /d "%~dp0"

rem Force-disable all one-shot debug modes for production launch
set GW_DEBUG_NEWS_OVERLAY_PHASE2=
set GW_DEBUG_SYSTEM_S1_5=
set GW_DEBUG_PLANNER_ONCE=
set GW_CHECKPOINT_ACTION=resume

echo [1/2] Starting GlobalWatch_V2.py (Streamlit)...
start "GlobalWatch V3.1.2" cmd /k "cd /d %~dp0 && python -m streamlit run GlobalWatch_V2.py"

echo [2/2] Starting paper_trading.py (paper_config.json)...
start "Paper Trading V3.1.2" cmd /k "cd /d %~dp0 && set GW_DEBUG_NEWS_OVERLAY_PHASE2= && set GW_DEBUG_SYSTEM_S1_5= && set GW_DEBUG_PLANNER_ONCE= && set GW_CHECKPOINT_ACTION=resume && python -u paper_trading.py paper_config.json"

echo.
echo Both processes started in separate windows.
echo.
pause

