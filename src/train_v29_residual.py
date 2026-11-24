"""V29 residual RL: learn small corrections on top of the classical guidance.

The classical controller (sim3d/guidance.py) lands 91.7% precise on held-out
seeds with every simulated factor on. Its failures are gusty strong wind.
This trains a SAC policy whose action is a bounded correction added to the
controller's command (ResidualLandingEnv, authority +-10-15% per control), so
the learned part cannot drag the vehicle far from proven guidance.

Reward is the mission's own: +100 for a precise landing, -100 for any failure,
and a small per-step cost for time, distance and tilt. No extra shaping terms:
every hand-added term in V24, V25 and V28 was exploited or regressed.

Training environment: full difficulty (wind and gusts, mass / inertia / thrust,
lag, misalignment, fuel burn, throttle floor, pad tilt, density), 2-step sensor
delay, sensor noise and bias, position-fix dropouts.

Seeds: training uses 1,000,000+; checkpoint selection uses 50000-50199;
30000+ (development) and 90000+ (held-out audit) are never touched here.

Run
    python src/train_v29_residual.py
Re-running RESUMES from the last checkpoint (model, replay buffer, step count).
V29_FRESH=1 starts over.

Environment variables
    V29_OUT            output root (default repo root; on Colab a Drive folder)
    V29_STEPS          total training steps (default 2,000,000)
    V29_NENVS          parallel envs (default 8)
    V29_SUBPROC        1 = one process per env (default), 0 = in-process
    V29_EVAL_EPISODES  selection episodes per evaluation (default 200)
    V29_SEED           default 2929

Outputs (under V29_OUT)
    models/master_v29_residual_latest.zip      resume point
    models/master_v29_residual_best.zip        best selection score, saved only
                                               when it beats the classical baseline
    models/master_v29_residual_buffer.pkl      replay buffer
    models/master_v29_residual_*_manifest.json SHA-256, settings, versions, scores
    logs/v29_residual_eval.jsonl               every evaluation
    logs/tb_v29_residual/                      TensorBoard
"""
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sim3d.mission import MissionEnv
from sim3d.mission_wrappers import mission_sensors
from sim3d.residual import ResidualLandingEnv

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("V29_OUT", str(ROOT))).resolve()
MODELS, LOGS = OUT / "models", OUT / "logs"
LATEST = MODELS / "master_v29_residual_latest.zip"
BEST = MODELS / "master_v29_residual_best.zip"
BUFFER = MODELS / "master_v29_residual_buffer.pkl"
LATEST_MANIFEST = MODELS / "master_v29_residual_latest_manifest.json"
BEST_MANIFEST = MODELS / "master_v29_residual_best_manifest.json"
EVAL_LOG = LOGS / "v29_residual_eval.jsonl"
TB_LOG = LOGS / "tb_v29_residual"
try:  # TensorBoard is optional; training must not fail because it is missing.
    import tensorboard  # noqa: F401
    TB_DIR = str(TB_LOG)
except ImportError:
    TB_DIR = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_STEPS = int(os.environ.get("V29_STEPS", "2000000"))
N_ENVS = int(os.environ.get("V29_NENVS", "8"))
USE_SUBPROC = os.environ.get("V29_SUBPROC", "1") == "1"
EVAL_EPISODES = int(os.environ.get("V29_EVAL_EPISODES", "200"))
SEED = int(os.environ.get("V29_SEED", "2929"))
FRESH = os.environ.get("V29_FRESH", "0") == "1"

# Simulated factors, identical to the held-out audit.
DIFFICULTY, DELAY, NOISE, DROPOUT = 1.0, 2, 1.0, 0.005

NET_ARCH = [256, 256]
LR = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 500_000
LEARNING_STARTS = 10_000
GAMMA = 0.99
TAU = 0.005
GRADIENT_STEPS = max(1, N_ENVS // 2)
# Low target entropy keeps exploration small, so early training does not wreck
# the baseline the residual is meant to improve.
TARGET_ENTROPY = -8.0
ENT_COEF = "auto_0.01"

TRAIN_SEED_BASE = 1_000_000
SELECT_SEED = 50_000
SAVE_EVERY = min(100_000, max(TOTAL_STEPS // 4, 1))
BUFFER_EVERY = 500_000
EVAL_EVERY = min(100_000, max(TOTAL_STEPS // 4, 1))


def make_env(rank: int = 0, seed: int = 0, monitor: bool = True):
    def _init():
        raw = MissionEnv(difficulty=DIFFICULTY)
        env = ResidualLandingEnv(mission_sensors(raw, delay=DELAY, noise_scale=NOISE,
                                                 dropout=DROPOUT))
        if monitor:
            env = Monitor(env, info_keywords=("outcome",))
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------- evaluation

def evaluate(policy, episodes: int, seed: int, n_envs: int = 8) -> dict:
    """Run episodes with exactly the seeds seed..seed+episodes-1, in lockstep batches.

    policy maps a (k, 27) observation batch to (k, 4) residual actions; the zero
    policy reproduces the classical controller exactly."""
    envs = [make_env(monitor=False)() for _ in range(n_envs)]
    queues = [list(range(seed + i, seed + episodes, n_envs)) for i in range(n_envs)]
    obs = [None] * n_envs
    for i, env in enumerate(envs):
        if queues[i]:
            obs[i], _ = env.reset(seed=queues[i].pop(0))
    outcomes, returns, running = Counter(), [], np.zeros(n_envs)
    while any(o is not None for o in obs):
        live = [i for i, o in enumerate(obs) if o is not None]
        actions = policy(np.stack([obs[i] for i in live]))
        for i, a in zip(live, actions):
            obs_i, r, te, tr, info = envs[i].step(a)
            running[i] += r
            if te or tr:
                outcomes[info["outcome"]] += 1
                returns.append(running[i])
                running[i] = 0.0
                obs[i] = envs[i].reset(seed=queues[i].pop(0))[0] if queues[i] else None
            else:
                obs[i] = obs_i
    for env in envs:
        env.close()
    n = sum(outcomes.values())
    return {"episodes": n, "precise_rate": outcomes["PRECISE"] / n,
            "mean_return": float(np.mean(returns)), "outcomes": dict(outcomes)}


def zero_policy(obs_batch):
    return np.zeros((len(obs_batch), 4), dtype=np.float32)


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
        "total_timesteps": int(steps),
        "seed": SEED,
        "algorithm": "SAC residual on V29 classical guidance",
        "factors": {"difficulty": DIFFICULTY, "delay_steps": DELAY,
                    "noise_scale": NOISE, "nav_dropout_rate": DROPOUT},
        "hparams": {"net_arch": NET_ARCH, "lr": LR, "batch_size": BATCH_SIZE,
                    "buffer_size": BUFFER_SIZE, "learning_starts": LEARNING_STARTS,
                    "gamma": GAMMA, "tau": TAU, "gradient_steps": GRADIENT_STEPS,
                    "n_envs": N_ENVS, "target_entropy": TARGET_ENTROPY,
                    "ent_coef": ENT_COEF, "total_steps_target": TOTAL_STEPS},
        "seeds": {"train_base": TRAIN_SEED_BASE, "selection": [SELECT_SEED, SELECT_SEED + EVAL_EPISODES]},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__,
                     "gymnasium": gym.__version__, "mujoco": mujoco.__version__,
                     "device": DEVICE},
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def save_buffer(model):
    """Write to a temp file first: an interrupted upload to Drive must never
    leave a truncated buffer behind (that broke resume once)."""
    tmp = BUFFER.with_name(BUFFER.stem + "_tmp.pkl")
    model.save_replay_buffer(str(tmp))
    os.replace(tmp, BUFFER)


def load_buffer(model):
    try:
        model.load_replay_buffer(str(BUFFER))
        print(f"[resume] replay buffer loaded ({model.replay_buffer.size() * N_ENVS:,} transitions)", flush=True)
    except Exception as exc:  # truncated or incompatible file
        print(f"[resume] replay buffer unreadable ({type(exc).__name__}: {exc}); "
              "continuing with an empty buffer", flush=True)


def save_model(model, path: Path, manifest: Path, extra=None):
    """Write to a temp file first so a crash mid-save never corrupts the checkpoint."""
    tmp = path.with_name(path.stem + "_tmp.zip")
    model.save(str(tmp))
    os.replace(tmp, path)
    write_manifest(path, manifest, model.num_timesteps, extra)


# ----------------------------------------------------------------- callbacks

class CheckpointCB(BaseCallback):
    def _on_training_start(self):
        self._last_model = self._last_buffer = self.model.num_timesteps

    def _on_step(self):
        t = self.model.num_timesteps
        if t - self._last_model >= SAVE_EVERY:
            self._last_model = t
            save_model(self.model, LATEST, LATEST_MANIFEST)
            if t - self._last_buffer >= BUFFER_EVERY:
                self._last_buffer = t
                save_buffer(self.model)
            print(f"[checkpoint] t={t:,}", flush=True)
        return True


class OutcomeLogCB(BaseCallback):
    def _on_step(self):
        if self.n_calls % 500 == 0 and len(self.model.ep_info_buffer) > 0:
            c = Counter(ep.get("outcome") for ep in self.model.ep_info_buffer)
            n = sum(c.values())
            for name in ("PRECISE", "SAFE_OUTSIDE", "CRASH", "TIMEOUT", "UNSETTLED"):
                self.logger.record(f"outcomes/{name.lower()}", c[name] / n)
        return True


class EvalCB(BaseCallback):
    """Selection score on fixed seeds; best saved only when it beats the baseline."""

    def __init__(self, baseline: dict):
        super().__init__()
        self._baseline = baseline
        self._best = (baseline["precise_rate"], baseline["mean_return"])
        if EVAL_LOG.exists() and not FRESH:
            for line in EVAL_LOG.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "timesteps" in row:
                    self._best = max(self._best, (row["precise_rate"], row["mean_return"]))

    def _on_training_start(self):
        self._last = self.model.num_timesteps

    def _on_step(self):
        t = self.model.num_timesteps
        if t - self._last < EVAL_EVERY:
            return True
        self._last = t
        policy = lambda o: self.model.predict(o, deterministic=True)[0]
        result = evaluate(policy, EVAL_EPISODES, SELECT_SEED)
        result["timesteps"] = t
        with open(EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        self.logger.record("eval/precise_rate", result["precise_rate"])
        self.logger.record("eval/mean_return", result["mean_return"])
        score = (result["precise_rate"], result["mean_return"])
        better = score > self._best
        print(f"[eval] t={t:,} precise={result['precise_rate']:.1%} "
              f"(baseline {self._baseline['precise_rate']:.1%}) {result['outcomes']}"
              f"{'  <- new best' if better else ''}", flush=True)
        if better:
            self._best = score
            save_model(self.model, BEST, BEST_MANIFEST,
                       {"selection_eval": result, "baseline_eval": self._baseline})
        return True


# ---------------------------------------------------------------------- main

def main():
    for folder in (MODELS, LOGS, TB_LOG):
        folder.mkdir(parents=True, exist_ok=True)
    resume = LATEST.exists() and not FRESH
    if FRESH and EVAL_LOG.exists():
        EVAL_LOG.unlink()

    # The classical baseline on the selection seeds: the number to beat.
    baseline = None
    if EVAL_LOG.exists():
        for line in EVAL_LOG.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("baseline") and row["episodes"] == EVAL_EPISODES:
                baseline = row
    if baseline is None:
        print(f"[baseline] scoring classical guidance on {EVAL_EPISODES} selection seeds...", flush=True)
        baseline = dict(evaluate(zero_policy, EVAL_EPISODES, SELECT_SEED), baseline=True)
        with open(EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(baseline) + "\n")
    print(f"[baseline] precise={baseline['precise_rate']:.1%} {baseline['outcomes']}", flush=True)

    fns = [make_env(i, TRAIN_SEED_BASE + 1000 * SEED) for i in range(N_ENVS)]
    if USE_SUBPROC and N_ENVS > 1:
        start = "spawn" if sys.platform == "win32" else "forkserver"
        train_vec = SubprocVecEnv(fns, start_method=start)
    else:
        train_vec = DummyVecEnv(fns)

    if resume:
        model = SAC.load(str(LATEST), env=train_vec, device=DEVICE, tensorboard_log=TB_DIR)
        if BUFFER.exists():
            load_buffer(model)
        print(f"[resume] t={model.num_timesteps:,} of {TOTAL_STEPS:,}", flush=True)
    else:
        model = SAC(
            "MlpPolicy", train_vec, learning_rate=LR, buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, tau=TAU, gamma=GAMMA,
            train_freq=(1, "step"), gradient_steps=GRADIENT_STEPS, ent_coef=ENT_COEF,
            target_entropy=TARGET_ENTROPY, policy_kwargs=dict(net_arch=NET_ARCH),
            stats_window_size=200, seed=SEED, device=DEVICE,
            tensorboard_log=TB_DIR, verbose=1,
        )

    remaining = TOTAL_STEPS - model.num_timesteps
    print(f"V29 residual on {DEVICE}: {TOTAL_STEPS:,} steps ({remaining:,} to go), "
          f"{N_ENVS} envs, obs {train_vec.observation_space.shape}, net {NET_ARCH}", flush=True)
    if remaining <= 0:
        print("Nothing to do: checkpoint already at the step target.", flush=True)
        train_vec.close()
        return

    started = time.time()
    try:
        model.learn(total_timesteps=remaining,
                    callback=CallbackList([OutcomeLogCB(), EvalCB(baseline), CheckpointCB()]),
                    log_interval=20, reset_num_timesteps=not resume, tb_log_name="v29_residual")
    except KeyboardInterrupt:
        print("[train] interrupted: saving so the run can resume.", flush=True)
    finally:
        save_model(model, LATEST, LATEST_MANIFEST)
        save_buffer(model)
        train_vec.close()
    print(f"V29 residual stopped at t={model.num_timesteps:,} after "
          f"{(time.time() - started) / 3600:.2f} h.", flush=True)
    print(f"Best: {BEST}" if BEST.exists() else
          "No checkpoint beat the classical baseline yet.", flush=True)


if __name__ == "__main__":
    main()
