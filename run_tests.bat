@echo off
echo =====================================================================
echo  Running Test Suite with Pytest...
echo =====================================================================
py -m pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    python -m pytest tests/ -v
)
pause
