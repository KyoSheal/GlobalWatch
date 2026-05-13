@echo off
:: GlobalWatch Desktop Launcher for Windows
:: Double-click this file to start the app.

title GlobalWatch Desktop

:: ── Find Python ──────────────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python not found. Please install Python 3.10+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)

:: ── Check Python version ^(need 3.8+^) ────────────────────────────────────────
%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo Python 3.8 or newer is required. Please upgrade Python.
    pause
    exit /b 1
)

:: ── Install dependencies if missing ──────────────────────────────────────────
echo Checking dependencies...
%PYTHON% -c "import customtkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing customtkinter...
    %PYTHON% -m pip install --quiet customtkinter
)

%PYTHON% -c "import matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing matplotlib...
    %PYTHON% -m pip install --quiet matplotlib
)

:: ── Launch ───────────────────────────────────────────────────────────────────
echo Starting GlobalWatch Desktop...
cd /d "%~dp0"
%PYTHON% GlobalWatch_Desktop.py

if %errorlevel% neq 0 (
    echo.
    echo GlobalWatch exited with an error. See above for details.
    pause
)
