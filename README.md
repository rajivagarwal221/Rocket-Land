# Rocket-Land — self-landing rocket with PPO + Blender

A Lunar-Lander reinforcement-learning project. A PPO policy (Stable-Baselines3)
is trained on Gymnasium's `LunarLander-v3`, then driven live into a Blender 3D
scene, or rendered offline to MP4.

## Repository layout

```
Lunar-Lander-RL/
├── src/              # training, evaluation, wrappers, comparison scripts
├── simulator/        # live HTTP backend + Blender viewer + MP4 recorder
├── blender/          # .blend scenes, previews, rendered episode videos
├── models/           # trained PPO .zip checkpoints (master_best.zip is served)
├── logs/             # SB3 Monitor CSVs, evaluation summaries, TensorBoard
├── presentation/     # slide deck + demo clip
├── .venv/            # Python virtualenv (interpreter only; gitignored)
├── .gitignore
└── README.md
```

## Setup

```powershell
# Fresh virtualenv (project's old pyvenv.cfg may point at a missing Python)
python -m venv .venv
.venv\Scripts\activate
pip install gymnasium[box2d] stable-baselines3 tensorboard
```

## Train the current master

```powershell
.venv\Scripts\python.exe src\train_master.py
# or double-click
src\run_train_master.bat
```

## Evaluate / compare candidates

```powershell
.venv\Scripts\python.exe src\evaluate_precision.py
.venv\Scripts\python.exe src\compare_master_candidate.py
.venv\Scripts\python.exe src\diagnose_master.py
```

## Live 3D demo

```powershell
# Terminal 1: run the sim backend (loads models\master_best.zip by default)
.venv\Scripts\python.exe simulator\server.py

# Blender: open blender\combined.blend, Scripting workspace,
# open simulator\blender_viewer.py, press Run.
# In the 3D viewport: N-panel -> "Lunar" tab -> Play.
```

## Offline MP4 render

```powershell
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" ^
  -b blender\combined.blend ^
  --python simulator\record_episode.py ^
  -- --frames 450 --out blender\sim_episode.mp4
```

## Stack

- **RL env:** [`gymnasium`](https://gymnasium.farama.org) (`LunarLander-v3`, Box2D physics)
- **Policy:** [`stable-baselines3`](https://stable-baselines3.readthedocs.io/) PPO + PyTorch
- **Live backend:** Python stdlib `http.server.ThreadingHTTPServer` (port 8765, JSON)
- **Frontend:** Blender 5.x via `bpy` (modal operator + N-panel)
- **Offline render:** Blender headless, PNG sequence (stitched to MP4 with OpenCV)

See `simulator/README.md` for the live-demo architecture and troubleshooting.
