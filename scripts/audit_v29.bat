@echo off
REM Held-out audit (seeds 90000-90999, every simulated factor on):
REM classical guidance vs. the best trained residual, same 1000 episodes.
cd /d "%~dp0.."
set ARGS=--episodes 1000 --seed 90000 --difficulty 1 --delay 2 --noise --dropout 0.005

echo === Classical guidance ===
.venv\Scripts\python.exe src\evaluate_v29.py %ARGS% --out logs\classical_heldout.csv

if not exist models\master_v29_residual_best.zip (
    echo No residual has beaten the classical baseline yet - nothing more to audit.
    exit /b 0
)
echo === Classical + trained residual ===
.venv\Scripts\python.exe src\evaluate_v29.py %ARGS% --model models\master_v29_residual_best.zip --out logs\residual_heldout.csv
echo Results: logs\classical_heldout.json and logs\residual_heldout.json
