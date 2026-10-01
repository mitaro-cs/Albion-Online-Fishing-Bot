@echo off
setlocal
cd /d "%~dp0"
title Albion Fishing Bot

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if errorlevel 1 goto :nopython
)

".venv\Scripts\python.exe" -c "import numpy, cv2, mss, pynput, webview" >nul 2>nul
if errorlevel 1 (
  echo [setup] Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -m fishbot %*
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
pause
exit /b 1

:failed
echo.
echo Dependency installation failed - check your internet connection and try again.
pause
exit /b 1
