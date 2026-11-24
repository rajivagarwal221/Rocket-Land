@echo off
REM One-time setup for V29 training on an NVIDIA GPU (RTX 3050).
REM Run from the project folder:  scripts\setup_v29.bat
cd /d "%~dp0.."

if not exist .venv\Scripts\python.exe (
    echo Creating .venv ...
    python -m venv .venv || goto :fail
)

echo Installing CUDA PyTorch ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :fail

echo Installing simulator packages ...
.venv\Scripts\python.exe -m pip install mujoco==4.0.1 gymnasium==1.4.0 stable-baselines3==2.9.2 tensorboard || goto :fail

echo.
echo GPU check:
.venv\Scripts\python.exe -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU - update the NVIDIA driver'); raise SystemExit(0 if ok else 1)" || goto :fail

echo.
echo Tests:
.venv\Scripts\python.exe -m unittest discover -s tests || goto :fail

echo.
echo Setup OK. Next: scripts\train_v29.bat
exit /b 0

:fail
echo.
echo Setup FAILED - copy the lines above and send them.
exit /b 1
