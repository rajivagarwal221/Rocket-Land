"""V23: FINAL PRODUCTION MODEL -- precision-max with annealed sim-to-real noise.

Not a hackathon run. V23 is the deployment-candidate checkpoint. It must:
  (a) hit the Phase-1 precision bar (>=99% precise at target_limit=0.15)
  (b) survive production-grade sensor + actuator noise
  (c) ship with a reproducibility manifest for safety audit.

================================================================
GAP ANALYSIS (V22_best, shield-off, n=1000)

  PRECISE       91.0%
  SAFE_OUTSIDE   5.7%   <-- 63% of non-precise outcomes.
                           Policy lands SAFELY but drifts past +/-0.15.
                           This is the lever for 99%: tighten precision,
                           don't fix a crash tail (there almost isn't one).
  LEG_FAILURE    1.8%
  TIMEOUT        0.9%
  CRASH          0.6%

By slice:
  strong gravity: 97.2% precise  (basically done)
  weak gravity:   87.6% precise  (where the SAFE_OUTSIDE lives)

================================================================
V23 DESIGN

1. PRECISION PUSH (close the 91 -> 99 gap)
   * target_bonus         300 -> 500  (precision worth more terminally)
   * off_target_penalty    60 -> 120  (drift costs more)
   * x_shaping_coef       3.5 -> 4.5  (stronger continuous centering)
   * angular_stability_bonus 50 -> 100 (preserve both-leg landings)
   * x_shaping_fade_below  0.12       (unchanged -- anti-tipping)

2. TARGET-LIMIT CURRICULUM (the key precision lever)
   target_limit starts at 0.15 (matches v22 training regime, matches
   the EVAL bar). Anneals LINEARLY to 0.10 between step 1M and 2M,
   then holds at 0.10 for the final 1M steps.

   Mechanism: the policy is rewarded for landing inside a progressively
   SHRINKING zone. By the final 1M it is optimizing for 0.10 landings;
   when evaluated at 0.15 (the shipped success boundary) it has a
   built-in safety margin.

3. SIM-TO-REAL NOISE CURRICULUM
   Sensor noise and actuator corruption are annealed from 0 at step
   500k up to full production level at step 2M, then held full for
   the last 1M. Starting clean preserves v22's precision through the
   first phase; ramping noise gradually lets the policy adapt without
   catastrophic forgetting.

   Production noise levels (reached at scale=1.0):
     * obs noise:   per-dim Gaussian [x:0.010, y:0.010, vx:0.015, vy:0.015,
                    angle:0.009 rad, omega:0.010, legs/gravity: clean]
     * obs delay:   2 steps (approx 40 ms at 50 Hz) -- CONSTANT from step 1,
                    not annealed (delay=2 is small enough that annealing
                    adds risk without meaningful gain)
     * act noise:   Gaussian sigma=0.02 additive
     * act lag:     first-order alpha=0.5 (approx 20 ms time constant)

4. WEAK-GRAVITY EXPOSURE
   weak_gravity_prob 0.35 -> 0.55  (more training time on the lossy slice)

5. SAC HYPERPARAMETERS
   lr          linear 7e-5 -> 1e-5 over 3M steps  (sharp final convergence)
   learning_starts 5000       (warm policy, no random warmup needed)
   ent_coef    auto           (continue from v22's tuned state)
   batch_size  256
   buffer_size 1M (RESUMED from master_v22_buffer.pkl if present)
   net_arch    [256,256]      (unchanged -- bigger arch requires from-scratch)

6. BUDGET
   3 000 000 steps. On RTX 3050 approximately 6-8 h; CPU approximately 24-30 h.
   Set V23_STEPS env var to shorten (e.g. 100000 for a wiring test).

7. WARM START
   models/master_v22_best.zip  (best v22, not the final one)

8. INFRASTRUCTURE
   * Checkpoint: every 50k steps -> master_v23_production.zip
   * EvalCallback: 50 eps every 100k, with FULL noise and target_limit=0.10,
     best model to master_v23_best.zip. "Best" = best at the final
     production regime, not best at the intermediate curriculum.
   * Replay buffer: saved at each checkpoint (master_v23_buffer.pkl)
   * Reproducibility manifest: SHA-256 + software versions + hparams +
     timestamp. Written at each checkpoint, next to the zip.
   * TensorBoard: logs/tb_v23/

================================================================
EXCLUDED (deliberate -- they are post-v23 workstreams)
  * ONNX export, inference latency, action rate limiter (deployment)
  * Multi-seed ensemble (separate parallel run)
  * Larger network (requires from-scratch retrain, not fine-tune)
  * Fuel-efficiency penalty, impact-g penalty (reward-shape experiments)

================================================================
DELIVERABLES
  models/master_v23_production.zip        latest
  models/master_v23_best.zip              best under full-noise target=0.10
  models/master_v23_buffer.pkl            latest replay buffer
  models/master_v23_manifest.json
  models/master_v23_best_manifest.json
  logs/master_v23_training.monitor.csv
  logs/master_v23_eval.monitor.csv
  logs/tb_v23/
  logs/eval_v23/evaluations.npz
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper
from safety_shield import SafetyShield
from sensor_noise import (
    SensorNoise,
    ObservationDelay,
    DEFAULT_SENSOR_NOISE_STD,
)
from actuator_noise import ActuatorNoise


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
WARM_START = MODELS / "master_v22_best.zip"
WARM_BUFFER = MODELS / "master_v22_buffer.pkl"
CANDIDATE = MODELS / "master_v23_production.zip"
BUFFER_PATH = MODELS / "master_v23_buffer.pkl"
MANIFEST_PATH = MODELS / "master_v23_manifest.json"
BEST_MANIFEST_PATH = MODELS / "master_v23_best_manifest.json"
TB_LOG = LOGS / "tb_v23"
EVAL_LOG = LOGS / "eval_v23"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V23_STEPS", "3000000"))
SAVE_INTERVAL = 50_000

LR_START = 7e-5
LR_END = 1e-5
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V23_SEED", "2342"))

EVAL_EVERY = 100_000
EVAL_EPISODES = 50

# Curriculum anchors (in TOTAL_STEPS). These scale if V23_STEPS is overridden
# for a short dry-run, so the curriculum still spans the full schedule.
NOISE_ANNEAL_START_FRAC = 500_000 / 3_000_000       # 1/6
NOISE_ANNEAL_END_FRAC = 2_000_000 / 3_000_000       # 2/3
TARGET_ANNEAL_START_FRAC = 1_000_000 / 3_000_000    # 1/3
TARGET_ANNEAL_END_FRAC = 2_000_000 / 3_000_000      # 2/3

TARGET_LIMIT_START = 0.15     # matches v22 training and eval success boundary
TARGET_LIMIT_FINAL = 0.10     # production target (tighter than eval bar)
EVAL_TARGET_LIMIT = 0.10      # EvalCallback scores against final regime

# Reward knobs (precision push)
TARGET_BONUS = 500.0
OFF_TARGET_PENALTY = 120.0
TIMEOUT_PENALTY = 200.0
FAILED_LANDING_PENALTY = 500.0
X_SHAPING_COEF = 4.5
X_SHAPING_FADE_BELOW = 0.12
ANGULAR_STABILITY_BONUS = 100.0
ANGULAR_STABILITY_ANGLE = 0.15
ANGULAR_STABILITY_OMEGA = 0.05

# Sim-to-real noise (reached at scale=1.0 from the curriculum callback)
OBS_NOISE_STD = DEFAULT_SENSOR_NOISE_STD
OBS_DELAY_STEPS = 2
ACT_NOISE_STD = 0.02
ACT_LAG_ALPHA = 0.5

WEAK_GRAVITY_PROB = 0.55


# =============================================================
# REPRODUCIBILITY MANIFEST
# =============================================================

def sha256_of_file(path: Path) -> str:

    h = hashlib.sha256()
    with open(path, "rb") as f:

        for chunk in iter(lambda: f.read(1 << 20), b""):

            h.update(chunk)

    return h.hexdigest()


def hparams_dict():

    return {
        "learning_rate_start": LR_START,
        "learning_rate_end": LR_END,
        "learning_rate_schedule": "linear",
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "tau": TAU,
        "gamma": GAMMA,
        "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "total_steps_target": TOTAL_STEPS,
        "weak_gravity_prob": WEAK_GRAVITY_PROB,
        "reward": {
            "target_bonus": TARGET_BONUS,
            "off_target_penalty": OFF_TARGET_PENALTY,
            "timeout_penalty": TIMEOUT_PENALTY,
            "failed_landing_penalty": FAILED_LANDING_PENALTY,
            "x_shaping_coef": X_SHAPING_COEF,
            "x_shaping_fade_below": X_SHAPING_FADE_BELOW,
            "angular_stability_bonus": ANGULAR_STABILITY_BONUS,
            "angular_stability_angle": ANGULAR_STABILITY_ANGLE,
            "angular_stability_omega": ANGULAR_STABILITY_OMEGA,
        },
        "noise_production": {
            "obs_noise_std": OBS_NOISE_STD.tolist(),
            "obs_delay_steps": OBS_DELAY_STEPS,
            "act_noise_std": ACT_NOISE_STD,
            "act_lag_alpha": ACT_LAG_ALPHA,
        },
        "curriculum": {
            "noise_anneal_start_step": int(
                NOISE_ANNEAL_START_FRAC * TOTAL_STEPS
            ),
            "noise_anneal_end_step": int(
                NOISE_ANNEAL_END_FRAC * TOTAL_STEPS
            ),
            "target_limit_anneal_start_step": int(
                TARGET_ANNEAL_START_FRAC * TOTAL_STEPS
            ),
            "target_limit_anneal_end_step": int(
                TARGET_ANNEAL_END_FRAC * TOTAL_STEPS
            ),
            "target_limit_start": TARGET_LIMIT_START,
            "target_limit_final": TARGET_LIMIT_FINAL,
            "eval_target_limit": EVAL_TARGET_LIMIT,
        },
    }


def write_manifest(
    model_path: Path,
    manifest_path: Path,
    total_timesteps: int,
):

    manifest = {
        "model_path": model_path.name,
        "model_sha256": sha256_of_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "total_timesteps": int(total_timesteps),
        "seed": SEED,
        "hparams": hparams_dict(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gym.__version__,
            "device": DEVICE,
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if DEVICE == "cuda"
                else None
            ),
        },
        "env_wrapper_stack_inside_out": [
            "ThrustScaledLunarLander",
            "ActuatorNoise",
            "SafetyShield",
            "ObservationDelay",
            "SensorNoise",
            "TimeLimit",
            "TargetLandingWrapper",
            "Monitor",
        ],
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================
# ENV BUILDER
# =============================================================

def build_env(
    seed: int,
    noise_scale: float,
    target_limit: float,
    is_eval: bool,
):
    """Build a fully-wrapped V23 env.

    noise_scale and target_limit are the INITIAL values at construction
    time. Training-env values are updated during training by the
    curriculum callback; eval-env values are held fixed at production
    (noise_scale=1.0, target_limit=EVAL_TARGET_LIMIT).
    """

    base = ThrustScaledLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 0.8),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),
        weak_gravity_prob=WEAK_GRAVITY_PROB,
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )

    # ActuatorNoise OUTSIDE shield so shield decisions still operate on
    # the policy's INTENT -- the shield's safety claim is on what the
    # policy chose, independent of actuator corruption.
    env_inner = ActuatorNoise(
        base,
        noise_std=ACT_NOISE_STD,
        lag_alpha=ACT_LAG_ALPHA,
        clip=True,
        seed=seed + 11,
    )
    env_inner.set_scale(noise_scale)

    env_inner = SafetyShield(
        env_inner,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
    )

    # ObservationDelay then SensorNoise OUTSIDE shield. Shield already
    # saw the fresh obs through its own step(); only what the policy
    # sees is delayed / noisy.
    env_inner = ObservationDelay(env_inner, delay_steps=OBS_DELAY_STEPS)

    env_inner = SensorNoise(
        env_inner,
        noise_std=OBS_NOISE_STD,
        seed=seed + 22,
    )
    env_inner.set_scale(noise_scale)

    env_inner = gym.wrappers.TimeLimit(
        env_inner, max_episode_steps=MAX_EPISODE_STEPS
    )

    env = TargetLandingWrapper(
        env_inner,
        gamma=GAMMA,
        step_penalty=0.01,
        target_bonus=TARGET_BONUS,
        off_target_penalty=OFF_TARGET_PENALTY,
        timeout_penalty=TIMEOUT_PENALTY,
        failed_landing_penalty=FAILED_LANDING_PENALTY,
        target_limit=target_limit,
        x_shaping_coef=X_SHAPING_COEF,
        x_shaping_fade_below=X_SHAPING_FADE_BELOW,
        angular_stability_bonus=ANGULAR_STABILITY_BONUS,
        angular_stability_angle=ANGULAR_STABILITY_ANGLE,
        angular_stability_omega=ANGULAR_STABILITY_OMEGA,
    )

    log_name = "master_v23_eval" if is_eval else "master_v23_training"
    env = Monitor(env, filename=str(LOGS / log_name))

    return env


def find_wrappers(env):
    """Walk the wrapper chain and return references needed by the curriculum."""

    target_w = None
    sensor_w = None
    actuator_w = None

    e = env

    while e is not None:

        if isinstance(e, TargetLandingWrapper) and target_w is None:

            target_w = e

        if isinstance(e, SensorNoise) and sensor_w is None:

            sensor_w = e

        if isinstance(e, ActuatorNoise) and actuator_w is None:

            actuator_w = e

        e = getattr(e, "env", None)

    return target_w, sensor_w, actuator_w


# =============================================================
# CALLBACKS
# =============================================================

def _lerp(t, t_start, t_end, v_start, v_end):

    if t <= t_start:

        return v_start

    if t >= t_end:

        return v_end

    progress = (t - t_start) / (t_end - t_start)
    return v_start + (v_end - v_start) * progress


class CurriculumCallback(BaseCallback):
    """Anneals noise scale and target_limit on the TRAINING env.

    Does not touch the eval env (which is held at full-production).
    """

    def __init__(
        self,
        train_env,
        noise_start_step: int,
        noise_end_step: int,
        target_start_step: int,
        target_end_step: int,
        target_start_value: float,
        target_end_value: float,
        apply_every_n_calls: int = 1000,
    ):

        super().__init__(verbose=0)
        self._train_env = train_env
        self._noise_start = noise_start_step
        self._noise_end = noise_end_step
        self._target_start = target_start_step
        self._target_end = target_end_step
        self._target_start_v = target_start_value
        self._target_end_v = target_end_value
        self._apply_every = apply_every_n_calls

        self._target_w = None
        self._sensor_w = None
        self._actuator_w = None
        self._last_scale = None
        self._last_target = None

    def _on_training_start(self):

        self._target_w, self._sensor_w, self._actuator_w = find_wrappers(
            self._train_env
        )

        missing = []

        if self._target_w is None:

            missing.append("TargetLandingWrapper")

        if self._sensor_w is None:

            missing.append("SensorNoise")

        if self._actuator_w is None:

            missing.append("ActuatorNoise")

        if missing:

            print(
                f"[curriculum] WARNING: could not find {missing} in "
                "train_env chain; those axes will not be annealed.",
                flush=True,
            )

        self._apply(0)

    def _apply(self, t):

        scale = _lerp(t, self._noise_start, self._noise_end, 0.0, 1.0)

        target = _lerp(
            t,
            self._target_start,
            self._target_end,
            self._target_start_v,
            self._target_end_v,
        )

        if self._sensor_w is not None:

            self._sensor_w.set_scale(scale)

        if self._actuator_w is not None:

            self._actuator_w.set_scale(scale)

        if self._target_w is not None:

            self._target_w.target_limit = float(target)

        # Log transitions so training output shows curriculum progression.
        if (
            self._last_scale is None
            or abs(scale - self._last_scale) >= 0.05
            or abs(target - (self._last_target or -1)) >= 0.005
        ):

            print(
                f"[curriculum] t={t:,} "
                f"noise_scale={scale:.3f} "
                f"target_limit={target:.3f}",
                flush=True,
            )
            self._last_scale = scale
            self._last_target = target

    def _on_step(self):

        if self.n_calls % self._apply_every == 0:

            self._apply(self.model.num_timesteps)

        return True


def save_checkpoint(model, candidate_path, buffer_path, manifest_path):

    # Atomic model save
    t = candidate_path.with_name(candidate_path.stem + "_temporary")
    tz = Path(str(t) + ".zip")
    tz.unlink(missing_ok=True)
    model.save(str(t))
    os.replace(tz, candidate_path)

    try:

        model.save_replay_buffer(str(buffer_path))

    except Exception as e:

        print(f"[warn] replay buffer save failed: {e}", flush=True)

    try:

        write_manifest(
            candidate_path, manifest_path, model.num_timesteps
        )

    except Exception as e:

        print(f"[warn] manifest write failed: {e}", flush=True)


class CheckpointCB(BaseCallback):
    """Periodic latest checkpoint + replay buffer + manifest."""

    def __init__(self):

        super().__init__(verbose=1)
        self.last = 0

    def _on_training_start(self):

        self.last = self.model.num_timesteps

    def _on_step(self):

        if self.model.num_timesteps - self.last >= SAVE_INTERVAL:

            save_checkpoint(
                self.model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH
            )
            self.last = self.model.num_timesteps
            print(
                f"CHECKPOINT {self.model.num_timesteps:,}",
                flush=True,
            )

        return True


class BestManifestCB(BaseCallback):
    """Re-write the best-model manifest when EvalCallback updates it."""

    def __init__(self, best_model_path: Path):

        super().__init__(verbose=0)
        self._best_path = best_model_path
        self._last_mtime = None

    def _on_step(self):

        return True

    def _on_rollout_end(self):

        if not self._best_path.exists():

            return

        mtime = self._best_path.stat().st_mtime

        if self._last_mtime is None or mtime != self._last_mtime:

            self._last_mtime = mtime

            try:

                write_manifest(
                    self._best_path,
                    BEST_MANIFEST_PATH,
                    self.model.num_timesteps,
                )
                print(
                    "[eval] best-model manifest updated "
                    f"(t={self.model.num_timesteps:,}).",
                    flush=True,
                )

            except Exception as e:

                print(
                    f"[warn] best-model manifest failed: {e}",
                    flush=True,
                )


# =============================================================
# LR SCHEDULE
# =============================================================

def linear_schedule(start, end):

    def _s(progress_remaining):

        return end + (start - end) * progress_remaining

    return _s


# =============================================================
# MAIN
# =============================================================

def main():

    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    TB_LOG.mkdir(exist_ok=True, parents=True)
    EVAL_LOG.mkdir(exist_ok=True, parents=True)

    if not WARM_START.exists():

        raise FileNotFoundError(
            f"V23 needs V22 best as warm start, but {WARM_START} does not exist."
        )

    # Curriculum step anchors scaled to actual TOTAL_STEPS
    noise_start = int(NOISE_ANNEAL_START_FRAC * TOTAL_STEPS)
    noise_end = int(NOISE_ANNEAL_END_FRAC * TOTAL_STEPS)
    target_start = int(TARGET_ANNEAL_START_FRAC * TOTAL_STEPS)
    target_end = int(TARGET_ANNEAL_END_FRAC * TOTAL_STEPS)

    train_env = build_env(
        seed=SEED,
        noise_scale=0.0,                 # curriculum starts at scale=0
        target_limit=TARGET_LIMIT_START, # starts at 0.15
        is_eval=False,
    )

    # Eval env is held at FULL production regime so best_model is
    # picked on the regime we actually ship.
    eval_env = build_env(
        seed=SEED + 101,
        noise_scale=1.0,
        target_limit=EVAL_TARGET_LIMIT,
        is_eval=True,
    )

    try:

        lr_schedule = linear_schedule(LR_START, LR_END)

        model = SAC.load(
            str(WARM_START),
            env=train_env,
            device=DEVICE,
            seed=SEED,
            tensorboard_log=str(TB_LOG),
            custom_objects={
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
                "learning_starts": LEARNING_STARTS,
                "batch_size": BATCH_SIZE,
                "buffer_size": BUFFER_SIZE,
                "tau": TAU,
                "gamma": GAMMA,
                "train_freq": (TRAIN_FREQ, "step"),
                "gradient_steps": GRADIENT_STEPS,
            },
        )

        if WARM_BUFFER.exists():

            try:

                model.load_replay_buffer(str(WARM_BUFFER))
                print(
                    f"Loaded V22 replay buffer ({WARM_BUFFER.name}, "
                    f"size={model.replay_buffer.size():,}).",
                    flush=True,
                )

            except Exception as e:

                print(
                    f"[warn] V22 buffer load failed: {e}. "
                    "Continuing empty.",
                    flush=True,
                )

        else:

            print(
                f"[info] {WARM_BUFFER.name} not found; empty buffer.",
                flush=True,
            )

        gpu_name = (
            torch.cuda.get_device_name(0)
            if DEVICE == "cuda"
            else "CPU"
        )
        print(
            f"V23 PRODUCTION on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, "
            f"warm_start={WARM_START.name}, "
            f"reward=precision-max(bonus=500,off=120,x_coef=4.5,ang=100), "
            f"target_limit anneal {TARGET_LIMIT_START} -> "
            f"{TARGET_LIMIT_FINAL} between t={target_start:,}-{target_end:,}, "
            f"noise anneal 0 -> 1 between t={noise_start:,}-{noise_end:,}, "
            f"weak_g_prob={WEAK_GRAVITY_PROB}, "
            f"shield ON, lr {LR_START:.0e}->{LR_END:.0e} linear",
            flush=True,
        )

        # Save initial-state artifacts before learning begins.
        save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

        curriculum_cb = CurriculumCallback(
            train_env=train_env,
            noise_start_step=noise_start,
            noise_end_step=noise_end,
            target_start_step=target_start,
            target_end_step=target_end,
            target_start_value=TARGET_LIMIT_START,
            target_end_value=TARGET_LIMIT_FINAL,
        )

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(MODELS),
            log_path=str(EVAL_LOG),
            eval_freq=EVAL_EVERY,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        )

        best_manifest_cb = BestManifestCB(MODELS / "best_model.zip")
        checkpoint_cb = CheckpointCB()

        callbacks = CallbackList(
            [curriculum_cb, checkpoint_cb, eval_cb, best_manifest_cb]
        )

        try:

            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                log_interval=20,
                progress_bar=False,
                reset_num_timesteps=True,
                tb_log_name="v23",
            )

        except KeyboardInterrupt:

            save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)
            print(
                "Interrupted; saved latest + buffer + manifest.",
                flush=True,
            )
            return

        save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

        # Promote EvalCallback best -> master_v23_best.zip
        eval_best = MODELS / "best_model.zip"
        final_best = MODELS / "master_v23_best.zip"

        if eval_best.exists():

            final_best.unlink(missing_ok=True)
            os.replace(eval_best, final_best)

            try:

                write_manifest(
                    final_best,
                    BEST_MANIFEST_PATH,
                    model.num_timesteps,
                )

            except Exception as e:

                print(
                    f"[warn] final best manifest failed: {e}",
                    flush=True,
                )

            print(
                f"V23 best promoted to {final_best.name}",
                flush=True,
            )

        else:

            print(
                "[warn] EvalCallback did not produce best_model.zip.",
                flush=True,
            )

        print(f"V23 DONE: {model.num_timesteps:,}", flush=True)

    finally:

        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    sys.stdout.reconfigure(line_buffering=True)
    main()
