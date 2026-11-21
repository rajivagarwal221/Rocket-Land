"""V28: first 3D policy. SAC on the MuJoCo 6-DOF lander (sim3d/).

Trained FROM SCRATCH. V27's weights cannot be reused: V27 sees 9 numbers per
frame and drives 2 engines in a plane; the 3D vehicle reports 16 numbers and
has 4 controls (throttle, two gimbal axes, roll). What carries over from
Phase 1 is the method that produced V27:

    reward        V23 recipe: potential-based shaping + terminal bonuses, no fuel term
    memory        frame stacking (4 frames = 80 ms at 50 Hz)
    capacity      [512, 512] actor and critics
    curriculum    3 phases: clean -> ramp difficulty and shrink the target -> hold
    safety net    best checkpoint chosen by PRECISE rate at the final regime

New for sim-to-real: engine and gimbal lag, servo rate limit, thrust
misalignment, mass / inertia / thrust randomization, wind with gusts, sensor
noise with per-episode bias, 40 ms observation delay.

Run
    python src/train_v28_mujoco.py                 full run
    V28_STEPS=200000 python src/train_v28_mujoco.py    wiring test
Re-running the same command RESUMES from the last checkpoint (model, replay
buffer, step count, learning-rate and curriculum position). Set V28_FRESH=1
to ignore an existing checkpoint.

Environment variables
    V28_OUT      output root (default: repo root). On Colab: a Drive folder.
    V28_STEPS    total training steps (default 6,000,000)
    V28_NENVS    parallel envs (default 8)
    V28_SUBPROC  1 = one process per env (default), 0 = all in one process
    V28_SEED     default 2828

Outputs (under V28_OUT)
    models/master_v28_latest.zip         latest checkpoint (resume point)
    models/master_v28_best.zip           best PRECISE rate at full difficulty
    models/master_v28_buffer.pkl         replay buffer
    models/master_v28_*_manifest.json    SHA-256 + hyperparameters + versions
    logs/tb_v28/                         TensorBoard
    logs/v28_eval_history.jsonl          every evaluation, one JSON per line
"""
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from sim3d.rocket_env import EnvConfig, RocketParams
from sim3d.wrappers import IMU_BIAS_STD, IMU_NOISE_STD, make_vec

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("V28_OUT", str(ROOT))).resolve()
MODELS = OUT / "models"
LOGS = OUT / "logs"
LATEST = MODELS / "master_v28_latest.zip"
BEST = MODELS / "master_v28_best.zip"
BUFFER = MODELS / "master_v28_buffer.pkl"
LATEST_MANIFEST = MODELS / "master_v28_latest_manifest.json"
BEST_MANIFEST = MODELS / "master_v28_best_manifest.json"
EVAL_HISTORY = LOGS / "v28_eval_history.jsonl"
TB_LOG = LOGS / "tb_v28"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_STEPS = int(os.environ.get("V28_STEPS", "6000000"))
N_ENVS = int(os.environ.get("V28_NENVS", "8"))
USE_SUBPROC = os.environ.get("V28_SUBPROC", "1") == "1"
SEED = int(os.environ.get("V28_SEED", "2828"))
FRESH = os.environ.get("V28_FRESH", "0") == "1"

N_STACK = 4
DELAY_STEPS = 2

LR_START = 3e-4
LR_END = 1e-5
BATCH_SIZE = 512
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 10_000
TAU = 0.005
GAMMA = 0.99
# One gradient step per 4 transitions: twice V27's update ratio.
GRADIENT_STEPS = max(1, N_ENVS // 4)
NET_ARCH = [512, 512]

# Curriculum, as fractions of TOTAL_STEPS so short runs still cover every phase.
PHASE1_END = 0.15      # clean physics, wide target: learn to land at all
PHASE2_END = 0.65      # ramp difficulty 0 -> 1, shrink target
RADIUS_START = 1.0     # m
RADIUS_FINAL = 0.35    # m, tighter than the 0.5 m ship bar for margin
SHIP_RADIUS = 0.5      # m, what evaluate_3d.py scores against

# Short runs (wiring tests) still checkpoint and evaluate a few times.
SAVE_EVERY = min(100_000, max(TOTAL_STEPS // 4, 1))
BUFFER_EVERY = 500_000
EVAL_EVERY = min(200_000, max(TOTAL_STEPS // 4, 1))
EVAL_EPISODES = 100
EVAL_ENVS = 4
EVAL_SEED = 900_000


def cosine_lr(progress_remaining: float) -> float:
    done = 1.0 - progress_remaining
    return LR_END + 0.5 * (LR_START - LR_END) * (1.0 + math.cos(math.pi * done))


def curriculum_at(step: int):
    """(difficulty, target_radius) for a given global step."""
    frac = step / max(TOTAL_STEPS, 1)
    if frac <= PHASE1_END:
        return 0.0, RADIUS_START
    if frac >= PHASE2_END:
        return 1.0, RADIUS_FINAL
    t = (frac - PHASE1_END) / (PHASE2_END - PHASE1_END)
    return t, RADIUS_START + (RADIUS_FINAL - RADIUS_START) * t


# ------------------------------------------------------------------ manifest

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(model_path: Path, manifest_path: Path, steps: int, extra=None):
    manifest = {
        "model_path": model_path.name,
        "model_sha256": sha256_of(model_path),
        "model_bytes": model_path.stat().st_size,
        "total_timesteps": int(steps),
        "seed": SEED,
        "algorithm": "SAC",
        "hparams": {
            "net_arch": NET_ARCH,
            "n_stack": N_STACK,
            "n_envs": N_ENVS,
            "vec_env": "SubprocVecEnv" if USE_SUBPROC else "DummyVecEnv",
            "lr": [LR_START, LR_END, "cosine"],
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "learning_starts": LEARNING_STARTS,
            "tau": TAU,
            "gamma": GAMMA,
            "train_freq": [1, "step"],
            "gradient_steps": GRADIENT_STEPS,
            "total_steps_target": TOTAL_STEPS,
        },
        "curriculum": {
            "phase1_end_frac": PHASE1_END,
            "phase2_end_frac": PHASE2_END,
            "radius_start_m": RADIUS_START,
            "radius_final_m": RADIUS_FINAL,
            "ship_radius_m": SHIP_RADIUS,
        },
        "vehicle": RocketParams().__dict__,
        "env_config": EnvConfig().__dict__,
        "sensors": {
            "noise_std": IMU_NOISE_STD.tolist(),
            "bias_std": IMU_BIAS_STD.tolist(),
            "delay_steps": DELAY_STEPS,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gym.__version__,
            "mujoco": mujoco.__version__,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=list), encoding="utf-8"
    )


def save_model(model, path: Path, manifest: Path, extra=None):
    """Write to a temp file first so a crash mid-save never corrupts the checkpoint."""
    tmp = path.with_name(path.stem + "_tmp.zip")
    model.save(str(tmp))
    os.replace(tmp, path)
    write_manifest(path, manifest, model.num_timesteps, extra)


# ---------------------------------------------------------------- evaluation

def evaluate(model, vec, episodes: int, seed: int) -> dict:
    """Deterministic rollouts with an equal episode quota per env (no short-episode bias)."""
    vec.seed(seed)
    obs = vec.reset()
    n = vec.num_envs
    quota = np.full(n, episodes // n)
    quota[: episodes % n] += 1
    outcomes, returns, errors = Counter(), [], []
    running = np.zeros(n)
    while quota.sum() > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec.step(action)
        running += rewards
        for i in np.flatnonzero(dones):
            if quota[i] > 0:
                quota[i] -= 1
                outcomes[infos[i]["outcome"]] += 1
                returns.append(running[i])
                if infos[i]["outcome"] in ("PRECISE", "SAFE_OUTSIDE"):
                    errors.append(infos[i]["landing_error"])
            running[i] = 0.0
    total = sum(outcomes.values())
    return {
        "episodes": total,
        "precise_rate": outcomes["PRECISE"] / total,
        "safe_rate": (outcomes["PRECISE"] + outcomes["SAFE_OUTSIDE"]) / total,
        "mean_return": float(np.mean(returns)),
        "median_landing_error_m": float(np.median(errors)) if errors else None,
        "outcomes": dict(outcomes),
    }


# ----------------------------------------------------------------- callbacks

class CurriculumCB(BaseCallback):
    """Pushes (difficulty, target_radius) to every training env."""

    def __init__(self, every: int = 500):
        super().__init__()
        self._every = every
        self._shown = None

    def _apply(self):
        difficulty, radius = curriculum_at(self.model.num_timesteps)
        self.training_env.env_method("set_curriculum", difficulty, radius)
        self.logger.record("curriculum/difficulty", difficulty)
        self.logger.record("curriculum/target_radius", radius)
        key = (round(difficulty, 1), round(radius, 1))
        if key != self._shown:
            self._shown = key
            print(
                f"[curriculum] t={self.model.num_timesteps:,} "
                f"difficulty={difficulty:.2f} target_radius={radius:.2f} m",
                flush=True,
            )

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        if self.n_calls % self._every == 0:
            self._apply()
        return True


class OutcomeLogCB(BaseCallback):
    """Rolling training-episode outcome rates in TensorBoard."""

    def _on_step(self):
        if self.n_calls % 500 == 0 and len(self.model.ep_info_buffer) > 0:
            outcomes = Counter(ep.get("outcome") for ep in self.model.ep_info_buffer)
            total = sum(outcomes.values())
            for name in ("PRECISE", "SAFE_OUTSIDE", "CRASH", "TIPPED", "TIMEOUT"):
                self.logger.record(f"outcomes/{name.lower()}", outcomes[name] / total)
        return True


class CheckpointCB(BaseCallback):
    """Latest model every SAVE_EVERY steps; replay buffer every BUFFER_EVERY."""

    def _on_training_start(self):
        self._last_model = self.model.num_timesteps
        self._last_buffer = self.model.num_timesteps

    def _on_step(self):
        t = self.model.num_timesteps
        if t - self._last_model >= SAVE_EVERY:
            self._last_model = t
            save_model(self.model, LATEST, LATEST_MANIFEST)
            if t - self._last_buffer >= BUFFER_EVERY:
                self._last_buffer = t
                self.model.save_replay_buffer(str(BUFFER))
            print(f"[checkpoint] t={t:,}", flush=True)
        return True


class EvalCB(BaseCallback):
    """Scores the policy at the FINAL regime and keeps the best by PRECISE rate."""

    def __init__(self, eval_vec):
        super().__init__()
        self._vec = eval_vec
        self._best = (-1.0, -1.0, -math.inf)
        if EVAL_HISTORY.exists() and not FRESH:
            for line in EVAL_HISTORY.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._best = max(
                    self._best,
                    (row["safe_rate"], row["precise_rate"], row["mean_return"]),
                )

    def _on_training_start(self):
        self._last = self.model.num_timesteps

    def _on_step(self):
        t = self.model.num_timesteps
        if t - self._last < EVAL_EVERY:
            return True
        self._last = t
        result = evaluate(self.model, self._vec, EVAL_EPISODES, EVAL_SEED)
        result["timesteps"] = t
        with open(EVAL_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        self.logger.record("eval/precise_rate", result["precise_rate"])
        self.logger.record("eval/safe_rate", result["safe_rate"])
        self.logger.record("eval/mean_return", result["mean_return"])
        # Rank by safe_rate first so a "lands often, sometimes off-pad" checkpoint
        # beats one that never lands at all. Precise rate breaks the tie.
        score = (result["safe_rate"], result["precise_rate"], result["mean_return"])
        is_best = score > self._best
        print(
            f"[eval] t={t:,} precise={result['precise_rate']:.1%} "
            f"safe={result['safe_rate']:.1%} return={result['mean_return']:.0f} "
            f"{result['outcomes']}{'  <- new best' if is_best else ''}",
            flush=True,
        )
        if is_best:
            self._best = score
            save_model(self.model, BEST, BEST_MANIFEST, {"eval": result})
        return True


# ---------------------------------------------------------------------- main

def main():
    for folder in (MODELS, LOGS, TB_LOG):
        folder.mkdir(parents=True, exist_ok=True)

    resume = LATEST.exists() and not FRESH
    if FRESH and EVAL_HISTORY.exists():
        EVAL_HISTORY.unlink()

    train_vec = make_vec(
        N_ENVS, SEED, 0.0, RADIUS_START, N_STACK,
        subproc=USE_SUBPROC, delay_steps=DELAY_STEPS,
    )
    eval_vec = make_vec(
        EVAL_ENVS, EVAL_SEED, 1.0, RADIUS_FINAL, N_STACK,
        subproc=False, delay_steps=DELAY_STEPS,
    )

    if resume:
        model = SAC.load(
            str(LATEST),
            env=train_vec,
            device=DEVICE,
            tensorboard_log=str(TB_LOG),
            custom_objects={"learning_rate": cosine_lr, "lr_schedule": cosine_lr},
        )
        if BUFFER.exists():
            model.load_replay_buffer(str(BUFFER))
        print(
            f"[resume] t={model.num_timesteps:,} of {TOTAL_STEPS:,}, "
            f"buffer={model.replay_buffer.size() * N_ENVS:,} transitions",
            flush=True,
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_vec,
            learning_rate=cosine_lr,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            train_freq=(1, "step"),
            gradient_steps=GRADIENT_STEPS,
            policy_kwargs=dict(
                net_arch=dict(pi=NET_ARCH, qf=NET_ARCH), activation_fn=nn.ReLU
            ),
            stats_window_size=200,
            seed=SEED,
            device=DEVICE,
            tensorboard_log=str(TB_LOG),
            verbose=1,
        )

    remaining = TOTAL_STEPS - model.num_timesteps
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    params = sum(p.numel() for p in model.policy.parameters())
    print(
        f"V28 3D on {gpu}: {TOTAL_STEPS:,} steps ({remaining:,} to go), "
        f"{N_ENVS} envs ({'subproc' if USE_SUBPROC else 'in-process'}), "
        f"obs {train_vec.observation_space.shape}, act {train_vec.action_space.shape}, "
        f"net {NET_ARCH}, {params:,} params, seed {SEED}",
        flush=True,
    )
    if remaining <= 0:
        print("Nothing to do: checkpoint already at the step target.", flush=True)
        train_vec.close()
        eval_vec.close()
        return

    callbacks = CallbackList(
        [CurriculumCB(), OutcomeLogCB(), EvalCB(eval_vec), CheckpointCB()]
    )
    started = time.time()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            log_interval=20,
            reset_num_timesteps=not resume,
            tb_log_name="v28",
        )
    except KeyboardInterrupt:
        print("[train] interrupted: saving so the run can resume.", flush=True)
    finally:
        save_model(model, LATEST, LATEST_MANIFEST)
        model.save_replay_buffer(str(BUFFER))
        train_vec.close()
        eval_vec.close()

    hours = (time.time() - started) / 3600
    print(f"V28 stopped at t={model.num_timesteps:,} after {hours:.2f} h.", flush=True)
    if BEST.exists():
        print(f"Best checkpoint: {BEST}", flush=True)


if __name__ == "__main__":
    main()
