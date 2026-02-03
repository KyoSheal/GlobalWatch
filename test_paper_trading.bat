@echo off
echo ========================================
echo Paper Trading Quick Test
echo ========================================
echo.
echo This will run a 30-minute simulation
echo to verify the system is working correctly.
echo.
echo Press any key to start...
pause >nul

echo.
echo Starting quick test...
echo.

python paper_trading.py paper_config_quick_test.json

echo.
echo ========================================
echo Test completed!
echo.
echo Check the following files:
echo - outputs\equity_curve_quick.png
echo - outputs\paper_trades_quick.csv
echo - outputs\portfolio_snapshots_quick.jsonl
echo - outputs\paper_summary_quick.txt
echo ========================================
echo.
pause
