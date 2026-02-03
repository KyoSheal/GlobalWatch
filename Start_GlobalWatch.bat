@echo off
echo ========================================
echo GlobalWatch V2.5 Launcher
echo ========================================
echo.
echo Select mode:
echo 1. Main Application (GUI + Analysis)
echo 2. Paper Trading (Simulation Mode)
echo 3. Both (Separate Windows)
echo.
set /p choice="Enter choice (1/2/3): "

if "%choice%"=="1" goto main
if "%choice%"=="2" goto paper
if "%choice%"=="3" goto both
goto end

:main
echo.
echo Starting Main Application...
cd /d "%~dp0"
python -m streamlit run GlobalWatch_V2.py
goto end

:paper
echo.
echo Starting Paper Trading...
cd /d "%~dp0"
python paper_trading.py
goto end

:both
echo.
echo Starting both modes in separate windows...
cd /d "%~dp0"
start "GlobalWatch Main" cmd /k "python -m streamlit run GlobalWatch_V2.py"
timeout /t 2 /nobreak >nul
start "GlobalWatch Paper Trading" cmd /k "python paper_trading.py"
goto end

:end
pause