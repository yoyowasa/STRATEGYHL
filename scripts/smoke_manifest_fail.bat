@echo off
setlocal EnableExtensions

if not defined OUTPUTS_DIR set "OUTPUTS_DIR=outputs_live_f15"
set "LOGDIR=logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
if defined MANIFEST_FAIL_LOG (
  set "LOG=%MANIFEST_FAIL_LOG%"
) else (
  set "LOG=%LOGDIR%\\smoke_manifest_fail.log"
)

set "LATEST=%~1"
if "%LATEST%"=="" (
  for /f "delims=" %%D in ('dir /b /ad /o-d "%OUTPUTS_DIR%" 2^>nul') do (
    if not defined LATEST if exist "%OUTPUTS_DIR%\\%%D\\manifest.json" set "LATEST=%%D"
  )
)
if "%LATEST%"=="" (
  echo [ERR] no manifest.json runs found under "%OUTPUTS_DIR%"
  exit /b 1
)
if not exist "%OUTPUTS_DIR%\\%LATEST%\\manifest.json" (
  echo [ERR] manifest.json not found for run: %LATEST%
  exit /b 1
)

set "MANI=%OUTPUTS_DIR%\\%LATEST%\\manifest.json"
set "BAK=%OUTPUTS_DIR%\\%LATEST%\\manifest.json.bak"
set "RENAMED=0"

ren "%MANI%" "manifest.json.bak"
if errorlevel 1 (
  echo [ERR] failed to rename manifest: "%MANI%"
  exit /b 1
)
set "RENAMED=1"

call scripts\\make_report_and_monitor.bat > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if "%RENAMED%"=="1" (
  ren "%BAK%" "manifest.json" >nul 2>&1
  if errorlevel 1 (
    copy /y "%BAK%" "%MANI%" >nul 2>&1
    if errorlevel 1 (
      echo [WARN] failed to restore manifest: "%MANI%"
    ) else (
      del /f /q "%BAK%" >nul 2>&1
    )
  )
)

if "%RC%"=="0" (
  echo [NG] expected failure but got success. see "%LOG%"
  exit /b 1
)

echo [OK] failure as expected. see "%LOG%"
exit /b 0
