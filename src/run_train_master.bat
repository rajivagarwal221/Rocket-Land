@echo off
setlocal

title Lunar Lander - Master PPO Training

REM Jump to project root (one level above src\)
cd /d "%~dp0\.."

echo ==========================================
echo       LUNAR LANDER MASTER PPO
echo ==========================================
echo.

REM Check Python environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment Python not found.
    echo Expected: .venv\Scripts\python.exe
    pause
    exit /b 1
)

REM Check training script
if not exist "src\train_master.py" (
    echo ERROR: src\train_master.py not found.
    pause
    exit /b 1
)

echo Python environment found.
echo.
echo Starting training script...
echo.

.venv\Scripts\python.exe src\train_master.py

if errorlevel 1 (
    echo.
    echo TRAINING FAILED.
    echo Please share the error message.
) else (
    echo.
    echo TRAINING SCRIPT FINISHED.
)

echo.
pause
endlocal
