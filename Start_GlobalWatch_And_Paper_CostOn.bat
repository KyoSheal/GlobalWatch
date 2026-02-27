@echo off
setlocal
title GlobalWatch + Paper Trading (Cost ON)

echo ========================================
echo Start GlobalWatch and Paper Trading (Cost ON)
echo ========================================
echo.

cd /d "%~dp0"

rem ---- pick python launcher (python or py)
set "PY=python"
where python >nul 2>&1 || set "PY=py"

rem ---- your runtime flags (optional)
set GW_DEBUG_NEWS_OVERLAY_PHASE2=
set GW_DEBUG_SYSTEM_S1_5=
set GW_DEBUG_PLANNER_ONCE=
set GW_CHECKPOINT_ACTION=resume

echo [1/2] Starting GlobalWatch_V2.py (Streamlit)...
start "GlobalWatch" cmd /k "%PY% -m streamlit run GlobalWatch_V2.py"

echo [2/2] Starting paper_trading.py with COST MODEL enabled...
start "Paper Trading (Cost ON)" cmd /k "%PY% -u paper_trading.py paper_config.json --cost-model-enabled true --cost-slippage-bps 5 --cost-fee-per-trade 0 --cost-fee-bps 0 --cost-min-fee 0"

echo.
echo Both processes started in separate windows.
echo.
pause