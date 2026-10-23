@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.11 or newer from https://www.python.org/downloads/windows/
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed. Check your connection and run the README commands.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m estimation play
pause
