@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================================================
REM smoke_check.bat
REM   - One-command smoke checks for report/monitor loop
REM Usage:
REM   smoke_check.bat [RUN_ID]
REM =========================================================

if not defined OUTPUTS_DIR set "OUTPUTS_DIR=outputs_live_f15"
if not defined REPORTS_DIR set "REPORTS_DIR=reports_live_f15"
set "SUMMARY_JSON=%REPORTS_DIR%\_monitor\summary.json"

set "TS=%RANDOM%"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>nul') do set "TS=%%I"

set "LOGDIR=logs\smoke_%TS%"
mkdir "%LOGDIR%" 2>nul

set "LATEST="
for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTPUTS_DIR%" 2^>nul') do (
  if not defined LATEST if exist "%OUTPUTS_DIR%\%%D\manifest.json" set "LATEST=%%D"
)
if "%LATEST%"=="" (
  echo [FAIL] No manifest.json runs found under "%OUTPUTS_DIR%".
  exit /b 1
)

if "%~1"=="" (
  set "RUN_ID=%LATEST%"
) else (
  set "RUN_ID=%~1"
)

echo [INFO] RUN_ID=%RUN_ID%
echo [INFO] LATEST_MANIFEST=%LATEST%
echo [INFO] LOGDIR=%LOGDIR%

set "FAIL=0"

call :run_e2e "%RUN_ID%" "%LOGDIR%\e2e_%RUN_ID%.log"
call :get_hash "%REPORTS_DIR%\%RUN_ID%\run_report.json" HASH1
if defined HASH1 (echo [INFO] HASH1=%HASH1%) else (echo [WARN] HASH1 not set)

call :run_reuse "%RUN_ID%" "%LOGDIR%\e2e2_%RUN_ID%.log"
call :get_hash "%REPORTS_DIR%\%RUN_ID%\run_report.json" HASH2
if defined HASH2 (echo [INFO] HASH2=%HASH2%) else (echo [WARN] HASH2 not set)
if defined HASH1 if defined HASH2 if /i not "%HASH1%"=="%HASH2%" (
  echo [FAIL] Reuse hash mismatch
  set "FAIL=1"
)

call :run_force "%RUN_ID%" "%LOGDIR%\force_%RUN_ID%.log"
call :run_zerobyte "%RUN_ID%" "%LOGDIR%\zerobyte_%RUN_ID%.log"
call :run_auto "%LOGDIR%\auto_latest.log" "%LATEST%"
call :check_runs_total "%SUMMARY_JSON%"

echo.
echo ==================== SMOKE RESULT ====================
if "%FAIL%"=="0" (
  echo [PASS] all checks passed
  set "RESULT=0"
) else (
  echo [FAIL] one or more checks failed
  set "RESULT=1"
)
goto :main_end

REM =========================================================
REM subroutines
REM =========================================================

:run_e2e
set "RID=%~1"
set "LOG=%~2"
call scripts\make_report_and_monitor.bat "%RID%" > "%LOG%" 2>&1
if errorlevel 1 (
  echo [FAIL] e2e failed. see "%LOG%"
  set "FAIL=1"
  goto :eof
)
call :must_find "%LOG%" "RUN SUMMARY" "e2e RUN SUMMARY"
call :must_find "%LOG%" "EXIT_CODE=0" "e2e EXIT_CODE"
call :must_exist_nonzero "%REPORTS_DIR%\%RID%\run_report.json" "run_report.json"
call :must_not_exist "%REPORTS_DIR%\%RID%\run_report.json.tmp" "run_report.json.tmp"
echo [PASS] e2e
goto :eof

:run_reuse
set "RID=%~1"
set "LOG=%~2"
call scripts\make_report_and_monitor.bat "%RID%" > "%LOG%" 2>&1
if errorlevel 1 (
  echo [FAIL] reuse run failed. see "%LOG%"
  set "FAIL=1"
  goto :eof
)
call :must_find "%LOG%" "run_report.json exists. skip generation." "reuse skip generation"
call :must_find "%LOG%" "EXIT_CODE=0" "reuse EXIT_CODE"
echo [PASS] reuse
goto :eof

:run_force
set "RID=%~1"
set "LOG=%~2"
set "FORCE=1"
call scripts\make_report_and_monitor.bat "%RID%" > "%LOG%" 2>&1
set "FORCE="
if errorlevel 1 (
  echo [FAIL] force run failed. see "%LOG%"
  set "FAIL=1"
  goto :eof
)
call :must_find "%LOG%" "FORCE=1" "force flag"
call :must_find "%LOG%" "regenerate run_report.json" "force regenerate"
call :must_find "%LOG%" "generating run_report.json" "force generating"
call :must_find "%LOG%" "EXIT_CODE=0" "force EXIT_CODE"
echo [PASS] force
goto :eof

:run_zerobyte
set "RID=%~1"
set "LOG=%~2"
type nul > "%REPORTS_DIR%\%RID%\run_report.json"
call scripts\make_report_and_monitor.bat "%RID%" > "%LOG%" 2>&1
if errorlevel 1 (
  echo [FAIL] zerobyte run failed. see "%LOG%"
  set "FAIL=1"
  goto :eof
)
call :must_find "%LOG%" "run_report.json is empty. regenerate." "zerobyte detect"
call :must_find "%LOG%" "generating run_report.json" "zerobyte generating"
call :must_find "%LOG%" "EXIT_CODE=0" "zerobyte EXIT_CODE"
call :must_exist_nonzero "%REPORTS_DIR%\%RID%\run_report.json" "run_report.json restored"
echo [PASS] zerobyte
goto :eof

:run_auto
set "LOG=%~1"
set "EXPECTED=%~2"
call scripts\make_report_and_monitor.bat > "%LOG%" 2>&1
if errorlevel 1 (
  echo [FAIL] auto latest failed. see "%LOG%"
  set "FAIL=1"
  goto :eof
)
call :must_find "%LOG%" "RUN SUMMARY" "auto RUN SUMMARY"

set "AUTO_RID="
for /f "tokens=1* delims==" %%A in ('findstr /b /c:"RUN_ID=" "%LOG%"') do (
  set "AUTO_RID=%%B"
)

if "%AUTO_RID%"=="" (
  echo [FAIL] auto RUN_ID not found in log
  set "FAIL=1"
  goto :eof
)

if /i not "%AUTO_RID%"=="%EXPECTED%" (
  echo [FAIL] auto RUN_ID mismatch. got=%AUTO_RID% expected=%EXPECTED%
  set "FAIL=1"
  goto :eof
)

call :must_find "%LOG%" "EXIT_CODE=0" "auto EXIT_CODE"
echo [PASS] auto latest
goto :eof

:check_runs_total
set "S=%~1"
if not exist "%S%" (
  echo [FAIL] summary.json not found
  set "FAIL=1"
  goto :eof
)
set "RT="
for /f "delims=" %%A in ('powershell -NoProfile -Command "try { $raw = Get-Content -Raw -LiteralPath '%S%'; $v = (ConvertFrom-Json -InputObject $raw).runs_total; if ($null -ne $v) { Write-Output $v } } catch { }" 2^>nul') do set "RT=%%A"
if not defined RT (
  for /f "tokens=2 delims=:" %%A in ('findstr /c:"runs_total" "%S%"') do set "RT=%%A"
)
if "%RT%"=="" (
  echo [FAIL] runs_total not found in summary
  set "FAIL=1"
  goto :eof
)
set "RT=%RT:,=%"
set "RT=%RT: =%"
if "%RT%"=="0" (
  echo [FAIL] runs_total is 0
  set "FAIL=1"
  goto :eof
)
echo [PASS] runs_total=%RT%
goto :eof

:get_hash
set "FILE=%~1"
set "OUTVAR=%~2"
set "%OUTVAR%="
if not exist "%FILE%" goto :eof
for /f "tokens=1" %%H in ('certutil -hashfile "%FILE%" SHA256 ^| findstr /r "^[0-9A-F][0-9A-F]"') do (
  set "%OUTVAR%=%%H"
  goto :eof
)
goto :eof

:must_find
set "LOG=%~1"
set "PAT=%~2"
set "NAME=%~3"
findstr /i /c:"%PAT%" "%LOG%" >nul 2>&1
if errorlevel 1 (
  echo [FAIL] missing "%NAME%" in "%LOG%"
  set "FAIL=1"
)
goto :eof

:must_exist_nonzero
set "FILE=%~1"
set "NAME=%~2"
if not exist "%FILE%" (
  echo [FAIL] missing file "%NAME%" : "%FILE%"
  set "FAIL=1"
  goto :eof
)
for %%A in ("%FILE%") do set "SZ=%%~zA"
if "%SZ%"=="0" (
  echo [FAIL] file is 0 bytes "%NAME%" : "%FILE%"
  set "FAIL=1"
)
goto :eof

:must_not_exist
set "FILE=%~1"
set "NAME=%~2"
if exist "%FILE%" (
  echo [FAIL] should not exist "%NAME%" : "%FILE%"
  set "FAIL=1"
)
goto :eof

:main_end
if "%RESULT%"=="0" exit /b 0
exit /b 1
