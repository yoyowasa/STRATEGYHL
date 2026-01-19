@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." || (echo [ERR] failed to change to repo root. & exit /b 1)

set "OUTPUTS_DIR=outputs_live_f15"
set "REPORTS_DIR=reports_live_f15"
set "WINDOW=10"
set "EXIT_CODE=0"

if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage

if not "%~2"=="" set "WINDOW=%~2"

if not exist "%OUTPUTS_DIR%" (
  echo [ERR] outputs dir not found: "%OUTPUTS_DIR%"
  set "EXIT_CODE=1"
  goto :done
)

if "%~1"=="" (
  for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTPUTS_DIR%"') do (
    set "RUN_ID=%%D"
    goto :got_run_id
  )
  echo [ERR] no run directories found under "%OUTPUTS_DIR%".
  set "EXIT_CODE=1"
  goto :done
) else (
  set "RUN_ID=%~1"
)

:got_run_id
set "OUT_RUN=%OUTPUTS_DIR%\%RUN_ID%"
set "REP_RUN=%REPORTS_DIR%\%RUN_ID%"
set "RUN_REPORT=%REP_RUN%\run_report.json"
set "RUN_REPORT_TMP=%RUN_REPORT%.tmp"

if not exist "%OUT_RUN%" (
  echo [ERR] outputs run dir not found: "%OUT_RUN%"
  set "EXIT_CODE=1"
  goto :done
)

if not exist "%REP_RUN%" (
  mkdir "%REP_RUN%"
  if errorlevel 1 (
    echo [ERR] failed to create reports dir: "%REP_RUN%"
    set "EXIT_CODE=1"
    goto :done
  )
)

set "FORCE_REGEN=0"
if /I "%FORCE%"=="1" set "FORCE_REGEN=1"

echo [INFO] run_id  : %RUN_ID%
echo [INFO] outputs : %OUT_RUN%
echo [INFO] reports : %REP_RUN%

set "NEED_REPORT=0"
if not exist "%RUN_REPORT%" set "NEED_REPORT=1"
if "%FORCE_REGEN%"=="1" set "NEED_REPORT=1"
if "%NEED_REPORT%"=="0" (
  for %%I in ("%RUN_REPORT%") do set "RUN_REPORT_SIZE=%%~zI"
  if "%RUN_REPORT_SIZE%"=="0" (
    echo [WARN] run_report.json is empty. regenerate.
    set "NEED_REPORT=1"
  )
)

if "%FORCE_REGEN%"=="1" echo [INFO] FORCE=1 -> regenerate run_report.json

if "%NEED_REPORT%"=="1" (
  echo [INFO] generating run_report.json...
  if exist "%RUN_REPORT_TMP%" del /f /q "%RUN_REPORT_TMP%"
  python scripts\run_report.py --run-dir "%OUT_RUN%" --out "%RUN_REPORT_TMP%"
  if errorlevel 1 (
    echo [ERR] run_report failed.
    if exist "%RUN_REPORT_TMP%" del /f /q "%RUN_REPORT_TMP%"
    set "EXIT_CODE=1"
    goto :done
  )
  if not exist "%RUN_REPORT_TMP%" (
    echo [ERR] run_report output missing: "%RUN_REPORT_TMP%"
    set "EXIT_CODE=1"
    goto :done
  )
  for %%I in ("%RUN_REPORT_TMP%") do set "RUN_REPORT_TMP_SIZE=%%~zI"
  if "%RUN_REPORT_TMP_SIZE%"=="0" (
    echo [ERR] run_report output is empty: "%RUN_REPORT_TMP%"
    del /f /q "%RUN_REPORT_TMP%"
    set "EXIT_CODE=1"
    goto :done
  )
  move /y "%RUN_REPORT_TMP%" "%RUN_REPORT%" >nul
  if errorlevel 1 (
    echo [ERR] failed to finalize run_report.json.
    del /f /q "%RUN_REPORT_TMP%"
    set "EXIT_CODE=1"
    goto :done
  )
) else (
  echo [INFO] run_report.json exists. skip generation.
)

echo [INFO] running monitor...
python scripts\monitor_live.py --reports-root "%REPORTS_DIR%" --window %WINDOW%
if errorlevel 1 (
  echo [ERR] monitor failed.
  set "EXIT_CODE=1"
  goto :done
)

echo [OK] done.
echo [INFO] summary: "%REPORTS_DIR%\_monitor\summary.json"
goto :done

:usage
echo Usage: %~nx0 [RUN_ID] [WINDOW]
echo.
echo If RUN_ID is omitted, the newest directory under "%OUTPUTS_DIR%" is used.
echo WINDOW defaults to 10.
echo To force regeneration: set FORCE=1
goto :done

:done
popd
exit /b %EXIT_CODE%
