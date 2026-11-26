@echo off
REM Train the V29 residual policy. Re-running this RESUMES from the last
REM checkpoint in models\ (Ctrl+C saves first, so stopping is safe).
REM   scripts\train_v29.bat         resume (or start if nothing saved yet)
REM   scripts\train_v29.bat fresh   discard the saved run and start over
REM Outputs: models\master_v29_residual_*.zip, logs\v29_residual_eval.jsonl
cd /d "%~dp0.."
set V29_STEPS=2000000
set V29_NENVS=8
set V29_FRESH=0
if /i "%~1"=="fresh" set V29_FRESH=1
.venv\Scripts\python.exe src\train_v29_residual.py
