@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-atom-harness-operator.ps1"
if errorlevel 1 (
  echo.
  echo Atom Harness Operator could not start.
  pause
)
endlocal
