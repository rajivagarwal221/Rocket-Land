@echo off
REM Train the V29 residual policy. Re-running this RESUMES from the last
REM checkpoint in models\ (Ctrl+C saves first, so stopping is safe).
REM Outputs: models\master_v29_residual_*.zip, logs\v29_residual_eval.jsonl
cd /d "%~dp0.."
set V29_STEPS=2000000
set V29_NENVS=8
set V29_FRESH=0
.venv\Scripts\python.exe src\train_v29_residual.py
