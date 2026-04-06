@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_CMD=python"

where %PYTHON_CMD% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10+ and make sure "python" is available.
    pause
    exit /b 1
)

echo [INFO] Project directory: %PROJECT_DIR%
echo [INFO] Checking dependencies...
%PYTHON_CMD% -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [INFO] Starting local server...
start "BacktestSystem" cmd /k "%PYTHON_CMD% -m uvicorn main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8000

echo [INFO] Server launch requested. Browser should open shortly.
exit /b 0
