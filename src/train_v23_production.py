"""V23: production final training model.

THIS IS NOT A HACKATHON RUN. V23 is the model-release candidate for a
real rocket-landing control system. Every design decision is defensible
in a design review.

================================================================
BACKGROUND
V22_best achieved 90.6 percent precise, 96.3 percent safe, 1.9 percent
crash + timeout, 1.8 percent leg failures on the common-profile 1000-ep
eval (shield ON). Shield-OFF ablation scored 91.0 percent precise, 96.7
percent safe -- the policy no longer relies on the hand-written shield.
Remaining measured weakness: weak-gravity slice (g greater than -6) at
87.6 percent precise.

V22 was trained on PERFECT state and PERFECT actuators. That makes it
unfit for deployment on real hardware. V23 closes that gap.
================================================================

V23 ADDITIONS (each tied to a specific gap)

  SIM-TO-REAL REALISM
    1. SensorNoise wrapper      per-dim Gaussian noise on observation
                                (x,y=0.010; vx,vy=0.015; angle=0.009 rad;
                                 omega=0.010; legs/gravity clean).
                                Levels match a well-instrumented GPS +
                                IMU fusion state estimator.
    2. ObservationDelay wrapper 2-step buffer (approximately 40 ms at
                                50 Hz). Matches sensor-chain latency.
    3. ActuatorNoise wrapper    sigma=0.02 additive + first-order lag
                                (alpha=0.5, approximately 20 ms time
                                constant). Matches engine command-to-
                                thrust response.

  ROBUSTNESS
    4. Weak-gravity exposure    weak_gravity_prob 0.35 -> 0.50.
                                V22 did not improve on this slice; more
                                samples needed.

  INFRASTRUCTURE
    5. Replay-buffer resume     if master_v22_buffer.pkl exists, load it
                                so V23 doesn't start with an empty buffer.
    6. Reproducibility manifest written alongside every checkpoint. JSON
                                with SHA-256 of model zip, Python / SB3 /
                                gymnasium / torch versions, seed, hparams,
                                total_timesteps, timestamp. Required for
                                safety-critical audit chain.
    7. EvalCallback             50 deterministic episodes every 100k steps
                                on an evaluation env (noise ON, same seed).
                                Writes best model to master_v23_best.zip.
    8. TensorBoard log          logs/tb_v23/. Live view of all SAC losses,
                                entropy coefficient, episode reward.

  BUDGET
    2.5M steps. RTX 3050 approximately 5-7 h GPU; CPU approximately 24 h.
    Set V23_STEPS env var to shorten (e.g. 100000 for a wiring test).

  WARM START
    models/master_v22_best.zip (not v22_precision -- always warm-start
    from the best-ever checkpoint, not the final one).

  REWARD
    UNCHANGED from V22. V23 is an isolated robustness experiment: can
    we keep V22's precision while adding sensor + actuator noise?
    Mixing reward changes with env changes would make the diff impossible
    to attribute.

  SAC HYPERPARAMS
    learning_rate        constant 5e-5  (gentler than V22; we are not
                                         reshaping reward, just adding
                                         noise -- small-step updates let
                                         the policy adapt without unlearning).
    learning_starts      10 000          (longer than V22; the buffer may
                                         have been pre-filled with CLEAN
                                         transitions from V22 -- let fresh
                                         noisy transitions accumulate
                                         before training updates begin).
    ent_coef             continue auto   (V22's tuned entropy is in
                                         ballpark for the same reward
                                         surface. Autotune handles drift).
    other hparams        unchanged from V22.

  EXCLUDED FROM V23 (deliberately -- these are separate workstreams)
    * ONNX export, inference-latency measurement, action rate-limiter
      (post-training deployment steps, not training).
    * Multi-seed ensemble (separate run, launched in parallel).
    * CI eval on every commit (infra, not training).
    * Fuel-efficiency penalty, impact-g penalty
      (require reward surgery -- treat as a V24+ reward-shape experiment).
    * Target-limit annealing (destabilizing on an already-good policy).

================================================================
DELIVERABLES after V23 finishes
  models/master_v23_production.zip   latest checkpoint
  models/master_v23_best.zip         best eval-score checkpoint
  models/master_v23_buffer.pkl       latest replay buffer
  models/master_v23_manifest.json    reproducibility manifest for latest
  models/master_v23_best_manifest.json    same, for best
  logs/master_v23_training.monitor.csv
  logs/tb_v23/                       TensorBoard event files
  logs/eval_v23/                     EvalCallback evaluations.npz
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
from sensor_noise import SensorNoise, ObservationDelay, DEFAULT_SENSOR_NOISE_STD
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

TOTAL_STEPS = int(os.environ.get("V23_STEPS", "2500000"))
SAVE_INTERVAL = 50_000

LEARNING_RATE = 5e-5
LEARNING_STARTS = 10_000
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

# Sim-to-real realism levels (v23 production)
OBS_NOISE_STD = DEFAULT_SENSOR_NOISE_STD
OBS_DELAY_STEPS = 2
ACT_NOISE_STD = 0.02
ACT_LAG_ALPHA = 0.5


# =============================================================
# REPRODUCIBILITY MANIFEST
# =============================================================

def sha256_of_file(path: Path) -> str:

    h = hashlib.sha256()
    with open(path, "rb") as f:

        for chunk in iter(lambda: f.read(1 << 20), b""):

            h.update(chunk)

    return h.hexdigest()


def write_manifest(
    model_path: Path,
    manifest_path: Path,
    total_timesteps: int,
    hparams: dict,
):

    manifest = {
        "model_path": model_path.name,
        "model_sha256": sha256_of_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "total_timesteps": int(total_timesteps),
        "seed": SEED,
        "hparams": hparams,
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
        "env": {
            "base_wrappers": [
                "ThrustScaledLunarLander",
                "SafetyShield",
                "ObservationDelay",
                "SensorNoise",
                "TimeLimit",
                "TargetLandingWrapper",
                "Monitor",
            ],
            "obs_noise_std": OBS_NOISE_STD.tolist(),
            "obs_delay_steps": OBS_DELAY_STEPS,
            "act_noise_std": ACT_NOISE_STD,
            "act_lag_alpha": ACT_LAG_ALPHA,
            "max_episode_steps": MAX_EPISODE_STEPS,
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# =============================================================
# ENV BUILDER
# =============================================================

def build_env(seed: int, is_eval: bool = False):
    """Build the v23 training / evaluation env with full realism stack."""

    base = ThrustScaledLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 0.8),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),
        weak_gravity_prob=0.50,              # 0.35 -> 0.50 (close weak-g gap)
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )

    # ActuatorNoise OUTSIDE shield: policy -> actuator -> shield -> engine
    # so the shield still makes safe decisions on the policy's INTENT,
    # but what reaches the engine is noisy / laggy.
    env_inner = ActuatorNoise(
        base,
        noise_std=ACT_NOISE_STD,
        lag_alpha=ACT_LAG_ALPHA,
        clip=True,
        seed=seed + 11,
    )

    env_inner = SafetyShield(
        env_inner,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
    )

    # ObservationDelay then SensorNoise OUTSIDE the shield: shield already
    # saw the fresh obs internally, so delaying / noising what the policy
    # sees does not break the shield's safety claim.
    env_inner = ObservationDelay(env_inner, delay_steps=OBS_DELAY_STEPS)

    env_inner = SensorNoise(
        env_inner,
        noise_std=OBS_NOISE_STD,
        seed=seed + 22,
    )

    env_inner = gym.wrappers.TimeLimit(
        env_inner, max_episode_steps=MAX_EPISODE_STEPS
    )

    # Reward unchanged from V22 (proven)
    env = TargetLandingWrapper(
        env_inner,
        gamma=GAMMA,
        step_penalty=0.01,
        target_bonus=300.0,
        off_target_penalty=60.0,
        timeout_penalty=200.0,
        failed_landing_penalty=500.0,
        target_limit=0.15,
        x_shaping_coef=3.5,
        x_shaping_fade_below=0.12,
        angular_stability_bonus=50.0,
        angular_stability_angle=0.15,
        angular_stability_omega=0.05,
    )

    log_name = "master_v23_eval" if is_eval else "master_v23_training"
    env = Monitor(env, filename=str(LOGS / log_name))

    return env


# =============================================================
# CALLBACKS
# =============================================================

def hparams_dict():

    return {
        "learning_rate": LEARNING_RATE,
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "tau": TAU,
        "gamma": GAMMA,
        "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "total_steps_target": TOTAL_STEPS,
        "weak_gravity_prob": 0.50,
        "reward": {
            "target_bonus": 300.0,
            "off_target_penalty": 60.0,
            "timeout_penalty": 200.0,
            "failed_landing_penalty": 500.0,
            "x_shaping_coef": 3.5,
            "x_shaping_fade_below": 0.12,
            "angular_stability_bonus": 50.0,
        },
    }


def save_checkpoint(model, candidate_path, buffer_path, manifest_path):

    # Atomic model save
    t = candidate_path.with_name(candidate_path.stem + "_temporary")
    tz = Path(str(t) + ".zip")
    tz.unlink(missing_ok=True)
    model.save(str(t))
    os.replace(tz, candidate_path)

    # Replay buffer
    try:

        model.save_replay_buffer(str(buffer_path))

    except Exception as e:

        print(f"[warn] replay buffer save failed: {e}", flush=True)

    # Manifest
    try:

        write_manifest(
            candidate_path,
            manifest_path,
            model.num_timesteps,
            hparams_dict(),
        )

    except Exception as e:

        print(f"[warn] manifest write failed: {e}", flush=True)


class CheckpointCB(BaseCallback):
    """Periodic latest-checkpoint + replay-buffer + manifest."""

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
    """After each EvalCallback save, re-write the best-model manifest."""

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
                    hparams_dict(),
                )
                print(
                    "[eval] updated best-model manifest "
                    f"(timestep {self.model.num_timesteps:,}).",
                    flush=True,
                )

            except Exception as e:

                print(
                    f"[warn] best-model manifest failed: {e}",
                    flush=True,
                )


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

    train_env = build_env(seed=SEED, is_eval=False)
    eval_env = build_env(seed=SEED + 101, is_eval=True)

    try:

        # Load V22 weights; override hparams via custom_objects so V23
        # runs under its own regime immediately on load.
        model = SAC.load(
            str(WARM_START),
            env=train_env,
            device=DEVICE,
            seed=SEED,
            tensorboard_log=str(TB_LOG),
            custom_objects={
                "learning_rate": LEARNING_RATE,
                "lr_schedule": lambda _: LEARNING_RATE,
                "learning_starts": LEARNING_STARTS,
                "batch_size": BATCH_SIZE,
                "buffer_size": BUFFER_SIZE,
                "tau": TAU,
                "gamma": GAMMA,
                "train_freq": (TRAIN_FREQ, "step"),
                "gradient_steps": GRADIENT_STEPS,
            },
        )

        # Resume replay buffer from V22 if available.
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
                    f"[warn] could not load V22 buffer: {e}. "
                    "Continuing with empty buffer.",
                    flush=True,
                )

        else:

            print(
                f"[info] {WARM_BUFFER.name} not found; "
                "V23 starts with an empty replay buffer.",
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
            f"obs_noise=yes(2-step delay), act_noise=yes(sigma=0.02,alpha=0.5), "
            f"weak_g_prob=0.50, shield ON, lr={LEARNING_RATE}",
            flush=True,
        )

        # Save initial-state checkpoint + manifest BEFORE learn() begins
        # so an early crash still leaves a valid v23 artifact.
        save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

        # EvalCallback uses its own best_model.zip inside MODELS/.
        # We let it do its thing, then post-training rename to
        # master_v23_best.zip.
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
            [checkpoint_cb, eval_cb, best_manifest_cb]
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

        # Promote EvalCallback's best_model.zip -> master_v23_best.zip
        eval_best = MODELS / "best_model.zip"
        final_best = MODELS / "master_v23_best.zip"

        if eval_best.exists():

            final_best.unlink(missing_ok=True)
            os.replace(eval_best, final_best)

            # Re-write best manifest pointing at the final name.
            try:

                write_manifest(
                    final_best,
                    BEST_MANIFEST_PATH,
                    model.num_timesteps,
                    hparams_dict(),
                )

            except Exception as e:

                print(
                    f"[warn] final best manifest failed: {e}",
                    flush=True,
                )

            print(
                f"V23 best model promoted to {final_best.name}",
                flush=True,
            )

        else:

            print(
                "[warn] EvalCallback did not produce best_model.zip; "
                "no best-model promotion.",
                flush=True,
            )

        print(f"V23 DONE: {model.num_timesteps:,}", flush=True)

    finally:

        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    sys.stdout.reconfigure(line_buffering=True)
    main()
