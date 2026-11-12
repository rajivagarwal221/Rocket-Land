"""V25: surgical push toward 98% precise.

Context: V24 regressed from V23 (97% precise -> 80% precise) because adding
four Gaudet reward changes at once on a near-optimal policy broke the
learned safety behaviour. Key culprit: fuel_penalty_coef=0.05 taught the
policy not to fire the engine in long weak-gravity episodes.

V25 design principles:
  1. KEEP v23's entire reward unchanged (proven, 97% precise, 98.5% safe).
  2. ADD only two NON-INVASIVE defensive knobs:
       * glideslope soft penalty (gentler than v24, tighter altitude band,
         looser ratio threshold) -- targets LEG_FAILURE and CRASH cases
         where the agent arrives with lateral velocity.
       * hard attitude limit at 1.3 rad (approx 75 deg) -- only fires when
         the policy is already catastrophically off; acts as a floor, not
         a constraint on normal behaviour.
  3. DROP fuel penalty and gamma_shaping change (v24's killers).
  4. PUSH precision via tighter target-limit anneal (0.12 -> 0.08 instead
     of v23's 0.15 -> 0.10).
  5. START at FULL noise scale (v23 already adapted; no re-anneal cost).
  6. AUTO-CURRICULUM in a tight band [0.40, 0.70], only pushing weak-g
     harder when success rate >= 0.98 (very conservative).
  7. VERY GENTLE LR schedule (3e-5 -> 1e-6) for fine polish.

Target: >=98% precise, >=98.5% safe, <=0.5% crash+timeout.

If V25 regresses: V23 remains the production model.
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
from auto_curriculum import AutoCurriculumCallback


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
WARM_START = MODELS / "master_v23_best.zip"             # NOT v24
WARM_BUFFER = MODELS / "master_v23_buffer.pkl"
CANDIDATE = MODELS / "master_v25_surgical.zip"
BUFFER_PATH = MODELS / "master_v25_buffer.pkl"
MANIFEST_PATH = MODELS / "master_v25_manifest.json"
BEST_MANIFEST_PATH = MODELS / "master_v25_best_manifest.json"
TB_LOG = LOGS / "tb_v25"
EVAL_LOG = LOGS / "eval_v25"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V25_STEPS", "1500000"))
SAVE_INTERVAL = 50_000

LR_START = 3e-5
LR_END = 1e-6
LEARNING_STARTS = 3_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
GAMMA_SHAPING = 0.99                   # SAME as gamma -- v24 lesson
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V25_SEED", "2542"))

EVAL_EVERY = 100_000
EVAL_EPISODES = 50

# Target-limit anneal (tighter than v23)
TARGET_ANNEAL_START_FRAC = 300_000 / 1_500_000      # 1/5
TARGET_ANNEAL_END_FRAC = 800_000 / 1_500_000        # ~53%
TARGET_LIMIT_START = 0.12
TARGET_LIMIT_FINAL = 0.08
EVAL_TARGET_LIMIT = 0.08

# v23 reward -- UNCHANGED
TARGET_BONUS = 500.0
OFF_TARGET_PENALTY = 120.0
TIMEOUT_PENALTY = 200.0
FAILED_LANDING_PENALTY = 500.0
X_SHAPING_COEF = 4.5
X_SHAPING_FADE_BELOW = 0.12
ANGULAR_STABILITY_BONUS = 100.0
ANGULAR_STABILITY_ANGLE = 0.15
ANGULAR_STABILITY_OMEGA = 0.05

# v25 non-invasive additions
FUEL_PENALTY_COEF = 0.0              # OFF -- v24 lesson
GLIDESLOPE_PENALTY_COEF = 3.0        # soft, gentler than v24's 5.0
GLIDESLOPE_RATIO_THRESHOLD = 2.5     # looser -- only egregious lateral-dominant
GLIDESLOPE_ALTITUDE = 0.10           # tighter window -- fires less often
HARD_ATTITUDE_LIMIT = 1.3            # radians (approx 75 deg) -- floor only

# Full-production noise from step 0 (v23 already adapted during its own anneal)
OBS_NOISE_STD = DEFAULT_SENSOR_NOISE_STD
OBS_DELAY_STEPS = 2
ACT_NOISE_STD = 0.02
ACT_LAG_ALPHA = 0.5
INIT_NOISE_SCALE = 1.0

# Auto-curriculum -- conservative band
INIT_WEAK_GRAVITY_PROB = 0.55
AUTO_CURR_MIN = 0.40
AUTO_CURR_MAX = 0.70
AUTO_CURR_HIGH = 0.98
AUTO_CURR_LOW = 0.92
AUTO_CURR_STEP = 0.03


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
        "gamma_shaping": GAMMA_SHAPING,
        "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "total_steps_target": TOTAL_STEPS,
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
            "fuel_penalty_coef": FUEL_PENALTY_COEF,
            "glideslope_penalty_coef": GLIDESLOPE_PENALTY_COEF,
            "glideslope_ratio_threshold": GLIDESLOPE_RATIO_THRESHOLD,
            "glideslope_altitude": GLIDESLOPE_ALTITUDE,
            "hard_attitude_limit": HARD_ATTITUDE_LIMIT,
        },
        "noise_production": {
            "obs_noise_std": OBS_NOISE_STD.tolist(),
            "obs_delay_steps": OBS_DELAY_STEPS,
            "act_noise_std": ACT_NOISE_STD,
            "act_lag_alpha": ACT_LAG_ALPHA,
            "init_scale": INIT_NOISE_SCALE,
        },
        "curriculum": {
            "target_limit_anneal_start_step": int(
                TARGET_ANNEAL_START_FRAC * TOTAL_STEPS
            ),
            "target_limit_anneal_end_step": int(
                TARGET_ANNEAL_END_FRAC * TOTAL_STEPS
            ),
            "target_limit_start": TARGET_LIMIT_START,
            "target_limit_final": TARGET_LIMIT_FINAL,
            "eval_target_limit": EVAL_TARGET_LIMIT,
            "auto_curriculum": {
                "initial_weak_gravity_prob": INIT_WEAK_GRAVITY_PROB,
                "min_prob": AUTO_CURR_MIN,
                "max_prob": AUTO_CURR_MAX,
                "high_threshold": AUTO_CURR_HIGH,
                "low_threshold": AUTO_CURR_LOW,
                "step": AUTO_CURR_STEP,
            },
        },
        "design_notes": [
            "V23 reward unchanged.",
            "Glideslope + hard-attitude added as non-invasive defensive knobs.",
            "Fuel penalty OFF (V24 regression cause).",
            "gamma_shaping == gamma (V24 regression cause).",
            "Full noise scale from step 0 (V23 already adapted).",
            "Target-limit anneal tightened to 0.12 -> 0.08.",
            "Auto-curriculum ON with conservative thresholds.",
        ],
    }


def write_manifest(
    model_path: Path, manifest_path: Path, total_timesteps: int
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
            "ActuatorNoise (scale=1.0)",
            "SafetyShield (post_posed)",
            "ObservationDelay",
            "SensorNoise (scale=1.0)",
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
    target_limit: float,
    weak_gravity_prob: float,
    is_eval: bool,
):

    base = ThrustScaledLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 0.8),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),
        weak_gravity_prob=weak_gravity_prob,
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )

    env_inner = ActuatorNoise(
        base,
        noise_std=ACT_NOISE_STD,
        lag_alpha=ACT_LAG_ALPHA,
        clip=True,
        seed=seed + 11,
    )
    env_inner.set_scale(INIT_NOISE_SCALE)

    env_inner = SafetyShield(
        env_inner,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
        mode="post_posed",
    )

    env_inner = ObservationDelay(env_inner, delay_steps=OBS_DELAY_STEPS)
    env_inner = SensorNoise(
        env_inner, noise_std=OBS_NOISE_STD, seed=seed + 22
    )
    env_inner.set_scale(INIT_NOISE_SCALE)

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
        # v25 non-invasive additions
        gamma_shaping=GAMMA_SHAPING,
        fuel_penalty_coef=FUEL_PENALTY_COEF,
        glideslope_penalty_coef=GLIDESLOPE_PENALTY_COEF,
        glideslope_ratio_threshold=GLIDESLOPE_RATIO_THRESHOLD,
        glideslope_altitude=GLIDESLOPE_ALTITUDE,
        hard_attitude_limit=HARD_ATTITUDE_LIMIT,
    )

    log_name = "master_v25_eval" if is_eval else "master_v25_training"
    env = Monitor(
        env,
        filename=str(LOGS / log_name),
        info_keywords=("precise_landing",),
    )
    return env


def find_target_wrapper(env):

    e = env
    while e is not None:
        if isinstance(e, TargetLandingWrapper):
            return e
        e = getattr(e, "env", None)
    return None


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


class TargetLimitAnnealCallback(BaseCallback):
    """Anneals target_limit on the training env only."""

    def __init__(
        self,
        train_env,
        start_step,
        end_step,
        start_value,
        end_value,
        apply_every=1000,
    ):

        super().__init__(verbose=0)
        self._env = train_env
        self._start_step = start_step
        self._end_step = end_step
        self._start_value = start_value
        self._end_value = end_value
        self._apply_every = apply_every
        self._target_w = None
        self._last = None

    def _on_training_start(self):

        self._target_w = find_target_wrapper(self._env)
        self._apply(0)

    def _apply(self, t):

        new_limit = _lerp(
            t,
            self._start_step,
            self._end_step,
            self._start_value,
            self._end_value,
        )
        if self._target_w is not None:
            self._target_w.target_limit = float(new_limit)
        if self._last is None or abs(new_limit - self._last) >= 0.005:
            print(
                f"[target-anneal] t={t:,} target_limit={new_limit:.3f}",
                flush=True,
            )
            self._last = new_limit

    def _on_step(self):

        if self.n_calls % self._apply_every == 0:
            self._apply(self.model.num_timesteps)
        return True


def save_checkpoint(model, candidate_path, buffer_path, manifest_path):

    t = candidate_path.with_name(candidate_path.stem + "_temporary")
    tz = Path(str(t) + ".zip")
    tz.unlink(missing_ok=True)
    model.save(str(t))
    os.replace(tz, candidate_path)
    try:
        model.save_replay_buffer(str(buffer_path))
    except Exception as e:
        print(f"[warn] buffer save failed: {e}", flush=True)
    try:
        write_manifest(
            candidate_path, manifest_path, model.num_timesteps
        )
    except Exception as e:
        print(f"[warn] manifest write failed: {e}", flush=True)


class CheckpointCB(BaseCallback):

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
                    f"[warn] best manifest failed: {e}", flush=True
                )


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
            f"V25 needs V23 best as warm start, but {WARM_START} does not exist."
        )

    target_start = int(TARGET_ANNEAL_START_FRAC * TOTAL_STEPS)
    target_end = int(TARGET_ANNEAL_END_FRAC * TOTAL_STEPS)

    train_env = build_env(
        seed=SEED,
        target_limit=TARGET_LIMIT_START,
        weak_gravity_prob=INIT_WEAK_GRAVITY_PROB,
        is_eval=False,
    )

    eval_env = build_env(
        seed=SEED + 101,
        target_limit=EVAL_TARGET_LIMIT,
        weak_gravity_prob=INIT_WEAK_GRAVITY_PROB,
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
                    f"Loaded V23 buffer ({WARM_BUFFER.name}, "
                    f"size={model.replay_buffer.size():,}).",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[warn] V23 buffer load failed: {e}.", flush=True
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
            f"V25 SURGICAL on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, "
            f"warm_start={WARM_START.name}, "
            f"V23 reward UNCHANGED + glideslope={GLIDESLOPE_PENALTY_COEF}, "
            f"hard_attitude={HARD_ATTITUDE_LIMIT}, fuel=0, gamma_shaping=gamma, "
            f"target_limit anneal {TARGET_LIMIT_START} -> "
            f"{TARGET_LIMIT_FINAL} between t={target_start:,}-{target_end:,}, "
            f"auto-curriculum weak_g=[{AUTO_CURR_MIN},{AUTO_CURR_MAX}], "
            f"shield ON (post_posed), lr {LR_START:.0e}->{LR_END:.0e} linear",
            flush=True,
        )

        save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

        target_anneal_cb = TargetLimitAnnealCallback(
            train_env=train_env,
            start_step=target_start,
            end_step=target_end,
            start_value=TARGET_LIMIT_START,
            end_value=TARGET_LIMIT_FINAL,
        )

        auto_curr_cb = AutoCurriculumCallback(
            train_env=train_env,
            initial_prob=INIT_WEAK_GRAVITY_PROB,
            min_prob=AUTO_CURR_MIN,
            max_prob=AUTO_CURR_MAX,
            step_up=AUTO_CURR_STEP,
            step_down=AUTO_CURR_STEP,
            high_threshold=AUTO_CURR_HIGH,
            low_threshold=AUTO_CURR_LOW,
            rolling_window=64,
            adapt_every_n_steps=10_000,
            verbose=1,
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
            [
                target_anneal_cb,
                auto_curr_cb,
                checkpoint_cb,
                eval_cb,
                best_manifest_cb,
            ]
        )

        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callbacks,
                log_interval=20,
                progress_bar=False,
                reset_num_timesteps=True,
                tb_log_name="v25",
            )
        except KeyboardInterrupt:
            save_checkpoint(
                model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH
            )
            print(
                "Interrupted; saved latest + buffer + manifest.",
                flush=True,
            )
            return

        save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

        eval_best = MODELS / "best_model.zip"
        final_best = MODELS / "master_v25_best.zip"
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
                f"V25 best promoted to {final_best.name}", flush=True
            )
        else:
            print(
                "[warn] EvalCallback did not produce best_model.zip.",
                flush=True,
            )

        print(f"V25 DONE: {model.num_timesteps:,}", flush=True)

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    sys.stdout.reconfigure(line_buffering=True)
    main()
