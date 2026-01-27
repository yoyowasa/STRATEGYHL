@echo off
setlocal EnableExtensions

set "SRC=fixtures\smoke_run"
set "RUN_ID=fixture_run_smoke_001"
if not defined OUTPUTS_DIR set "OUTPUTS_DIR=outputs_live_f15"
set "DST=%OUTPUTS_DIR%\%RUN_ID%"

if not exist "%SRC%\manifest.json" (
  echo [ERR] fixture source missing: %SRC%
  exit /b 1
)

if not exist "%OUTPUTS_DIR%" mkdir "%OUTPUTS_DIR%" >nul 2>&1
if not exist "%DST%" mkdir "%DST%" >nul 2>&1

xcopy /e /i /y "%SRC%\*" "%DST%\" >nul
if errorlevel 1 (
  echo [ERR] failed to copy fixture to %DST%
  exit /b 1
)

echo [OK] fixture ready: %DST%
exit /b 0
