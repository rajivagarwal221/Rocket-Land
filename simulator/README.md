# Lunar Lander — Live 3D Simulator

This folder bridges the trained PPO model (Stable-Baselines3 + Gymnasium
`LunarLander-v3`) with the polished 3D scene in
`../blender/combined.blend`.

```
  ┌──────────────┐    HTTP /state     ┌──────────────────────┐
  │  server.py   │ ──────────────────▶│ blender_viewer.py    │
  │  (PPO model  │                    │ (modal operator in   │
  │   steps env) │◀── /reset ─────────│  Blender, drives the │
  └──────────────┘                    │  Rocket + flames)    │
                                      └──────────────────────┘
```

## One-time setup

The project's `.venv/pyvenv.cfg` points to a Python interpreter path that
no longer exists on this machine. Fix it so `.venv/Scripts/python.exe`
works, or create a fresh venv:

```bash
# option A: edit .venv/pyvenv.cfg so `home =` points to your Python 3.13
#          install folder

# option B: new venv (recommended)
python -m venv .venv2
.venv2\Scripts\activate
pip install gymnasium[box2d] stable-baselines3
```

## Run the backend

```bash
python simulator/server.py \
    --model .venv/models/master_best.zip \
    --host 127.0.0.1 --port 8765 --hz 50
```

It loads the PPO model, steps `LunarLander-v3` in a background thread at
50 Hz, and serves JSON on `/state`, `/reset`, `/pause`, `/resume`,
`/health`.

## Drive the 3D scene

1. Open `blender/combined.blend` in Blender.
2. Switch to the **Scripting** workspace.
3. Open `simulator/blender_viewer.py` and press **Run Script**.
4. In the viewport's **N-panel → Lunar** tab you now have:
   - **Lunar Lander Live Sim** — start the live modal operator
   - **Reset Episode** — POSTs `/reset` to the backend
   - **Stop Lunar Sim** — stops the modal operator

The script is idempotent. On first run it:
- creates a `RocketRig` Empty at the rocket's centroid and parents every
  existing rocket part to it, so one transform drives the whole rocket;
- renames the existing engine cone to `MainFlame` and adds `LeftFlame` /
  `RightFlame` for the side thrusters;
- adds a `HUD` text object showing episode / step / score / status.

## Mapping between env and Blender

`LunarLander-v3` observation → `simulator.blender_viewer.py`:

| env                      | Blender                                        |
|--------------------------|------------------------------------------------|
| `x` (env units)          | `RocketRig.location.x = PAD_WORLD_X + x * 6`   |
| `y` (env units)          | `RocketRig.location.z = PAD_WORLD_Z + y * 6`   |
| `angle` (rad)            | `RocketRig.rotation_euler.y = -angle`          |
| `action` 0/1/2/3         | hides/shows `MainFlame`, `LeftFlame`, `RightFlame` |

If you move the pad to a different place in the Blender scene, update the
`PAD_WORLD_*` and `SCALE_*` constants at the top of
`blender_viewer.py`.

## Troubleshooting

- **"Backend OFFLINE" in the N-panel** — the server isn't running, or
  the host/port in `blender_viewer.py` doesn't match.
- **Rocket doesn't move** — make sure `combined.blend` is the file you
  opened (it contains a `Rocket` collection). The script needs that
  collection to build the rig.
- **Flames don't flicker on** — the backend may be between episodes
  (paused for 1.2 s after terminal) or the model is choosing action 0
  (no-op); try `Reset Episode`.
