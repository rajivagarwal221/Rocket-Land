"""V24: final production model.

Everything the Rocket-Land project has learned, in one fine-tune on top
of V23's champion. Design is grounded in the project's research-paper
basis; each new feature cites the paper it comes from.

================================================================
V23 BASELINE (common profile, n=1000, shield OFF)
  Precise      96.90%   (target 99%, gap -2.1 pp)
  Safe         98.50%
  Crash + TO    1.10%   (target <= 0.1%, gap -1.0 pp)
  Leg failures  0.40%
  Uniform across gravity / wind / thrust slices.

V24 CLOSES THE REMAINING GAP VIA
----------------------------------------------------------------
1. TWO-DISCOUNT REWARD (Gaudet 2020)
   * gamma_shaping = 0.95  (was 0.99 in V23)
   * gamma_terminal / SAC gamma unchanged at 0.99
   Gaudet shows that letting the shaping signal decay faster than the
   terminal reward yields a cleaner optimisation surface: the shaping
   guides the agent to the landing region, then the terminal bonus
   dominates the final metres. V23 had them at the same gamma, so the
   shaping could "compete" with the terminal bonus right at touchdown.

2. FUEL-EFFICIENT REWARD (Gaudet 2020 Sec II.D)
   * fuel_penalty_coef = 0.05 per step
   * fuel_lateral_weight = 0.5 (side engines cheaper than main)
   Gaudet reports that an explicit control-effort penalty keeps the
   policy from firing the main engine wastefully. In our setup this
   should remove the "high-g hover then drop" behaviour occasionally
   seen in long weak-gravity episodes.

3. GLIDESLOPE CONSTRAINT (Gaudet 2020 Sec II.B)
   * glideslope_penalty_coef = 5.0
   * glideslope_altitude = 0.12  (fires below altitude 0.12)
   * glideslope_ratio_threshold = 2.0  ( penalise |vx|/|vy| > 2 )
   Soft penalty on lateral-velocity-dominant descent in the final metres.
   Ensures the lander arrives with velocity vector primarily downward,
   which is directly the leg-failure root cause from V21/V22 analysis.

4. HARD ATTITUDE CONSTRAINT (Gaudet 2020 Sec II.B)
   * hard_attitude_limit = 1.2 rad (approx 69 degrees)
   Terminate episode as crash if attitude exceeds this. Gaudet uses
   80 deg (1.4 rad); we go tighter at 69 deg because our shield already
   disengages past 0.6 rad (34 deg), so anything beyond ~69 deg means
   both learned policy AND shield have failed -- hard crash is the
   correct label.

5. AUTOMATED CURRICULUM (Narvekar 2020 Sec 4.2)
   * AutoCurriculumCallback adapts weak_gravity_prob from recent
     success rate (Matiisen-style learning progress).
     Initial 0.55, bounded [0.30, 0.75].
   V23 held weak_gravity_prob fixed at 0.55. V24 lets the curriculum
   push harder when the agent is winning (rate >= 95%), back off when
   losing (rate <= 80%). This is a direct implementation of
   Narvekar 2020's "learning progress" curriculum.

6. PREEMPTIVE SHIELD MODE (Alshiekh 2018 Fig 1)
   V24 defaults to the post-posed shield (Alshiekh Fig 2) for compatibility
   with V23's warm-start, but exposes preemptive-mode support via the
   SafetyShield(mode='preemptive') option for future ablation experiments.
   The LTL spec the shield enforces is documented in safety_shield.py.

7. NG 1999 PBRS INVARIANCE PRESERVED
   Shaping is still F = gamma_shaping * Phi(s') - Phi(s), so Ng's
   policy-invariance theorem applies to the shaping term in isolation.
   Terminal bonuses (target_bonus, off_target_penalty, fuel, glideslope,
   hard-attitude) are additional reward components, not shaping.

PRESERVED FROM V23
----------------------------------------------------------------
* Reward: target_bonus=500, off_target_penalty=120 (precision-max).
* Reward: x_shaping_coef=4.5, x_shaping_fade_below=0.12.
* Reward: angular_stability_bonus=100.
* Reward: failed_landing_penalty=500, timeout_penalty=200.
* Env envelope: gravity_range, wind_power_range, turbulence_range,
  thrust_scale_range, target_width_range -- unchanged.
* Sim-to-real noise curriculum: sensor + actuator noise annealed 0 -> 1
  over step 350k -> 1.4M (scaled from V23's schedule to V24's shorter
  budget).
* Target-limit curriculum: 0.15 -> 0.10 over step 700k -> 1.4M.
* Reproducibility manifest (SHA-256, software versions, hparams).
* EvalCallback best-model saving; replay-buffer save; TensorBoard.

HYPERPARAMETERS
----------------------------------------------------------------
* Warm start: models/master_v23_best.zip
* Buffer resume: models/master_v23_buffer.pkl (if present)
* Learning rate: linear 5e-5 -> 5e-6 over 2M steps (gentle on a near-
  optimal policy; terminal LR 10x smaller than V23's for final polish).
* learning_starts: 5_000
* 2 000 000 total steps (~4-5 h GPU on RTX 3050; set V24_STEPS to shorten).

DELIVERABLES
----------------------------------------------------------------
  models/master_v24_final.zip              latest
  models/master_v24_best.zip               best under full-regime eval
  models/master_v24_buffer.pkl
  models/master_v24_manifest.json
  models/master_v24_best_manifest.json
  logs/master_v24_training.monitor.csv
  logs/tb_v24/
  logs/eval_v24/evaluations.npz
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
WARM_START = MODELS / "master_v23_best.zip"
WARM_BUFFER = MODELS / "master_v23_buffer.pkl"
CANDIDATE = MODELS / "master_v24_final.zip"
BUFFER_PATH = MODELS / "master_v24_buffer.pkl"
MANIFEST_PATH = MODELS / "master_v24_manifest.json"
BEST_MANIFEST_PATH = MODELS / "master_v24_best_manifest.json"
TB_LOG = LOGS / "tb_v24"
EVAL_LOG = LOGS / "eval_v24"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V24_STEPS", "2000000"))
SAVE_INTERVAL = 50_000

LR_START = 5e-5
LR_END = 5e-6
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99                          # SAC's RL discount (Gaudet's gamma_terminal)
GAMMA_SHAPING = 0.95                  # V24 Gaudet 2-discount: faster shaping decay
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V24_SEED", "2442"))

EVAL_EVERY = 100_000
EVAL_EPISODES = 50

# Curriculum anchors scaled to actual TOTAL_STEPS. V23 used 1/6 / 2/3 /
# 1/3 / 2/3; V24 shortens to a 2M budget while keeping the same fractions.
NOISE_ANNEAL_START_FRAC = 350_000 / 2_000_000       # ~1/6
NOISE_ANNEAL_END_FRAC = 1_400_000 / 2_000_000       # ~2/3
TARGET_ANNEAL_START_FRAC = 700_000 / 2_000_000      # ~1/3
TARGET_ANNEAL_END_FRAC = 1_400_000 / 2_000_000      # ~2/3

TARGET_LIMIT_START = 0.15
TARGET_LIMIT_FINAL = 0.10
EVAL_TARGET_LIMIT = 0.10

# Reward knobs
TARGET_BONUS = 500.0
OFF_TARGET_PENALTY = 120.0
TIMEOUT_PENALTY = 200.0
FAILED_LANDING_PENALTY = 500.0
X_SHAPING_COEF = 4.5
X_SHAPING_FADE_BELOW = 0.12
ANGULAR_STABILITY_BONUS = 100.0
ANGULAR_STABILITY_ANGLE = 0.15
ANGULAR_STABILITY_OMEGA = 0.05

# V24 Gaudet-grounded reward knobs
FUEL_PENALTY_COEF = 0.05
FUEL_LATERAL_WEIGHT = 0.5
GLIDESLOPE_PENALTY_COEF = 5.0
GLIDESLOPE_RATIO_THRESHOLD = 2.0
GLIDESLOPE_ALTITUDE = 0.12
HARD_ATTITUDE_LIMIT = 1.2            # radians (approx 69 deg)

# Sim-to-real noise levels (reached at scale=1.0)
OBS_NOISE_STD = DEFAULT_SENSOR_NOISE_STD
OBS_DELAY_STEPS = 2
ACT_NOISE_STD = 0.02
ACT_LAG_ALPHA = 0.5

# V24 auto-curriculum (Narvekar 2020)
INIT_WEAK_GRAVITY_PROB = 0.55
AUTO_CURR_MIN = 0.30
AUTO_CURR_MAX = 0.75
AUTO_CURR_HIGH = 0.95
AUTO_CURR_LOW = 0.80


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
            "fuel_lateral_weight": FUEL_LATERAL_WEIGHT,
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
            "auto_curriculum": {
                "initial_weak_gravity_prob": INIT_WEAK_GRAVITY_PROB,
                "min_prob": AUTO_CURR_MIN,
                "max_prob": AUTO_CURR_MAX,
                "high_threshold": AUTO_CURR_HIGH,
                "low_threshold": AUTO_CURR_LOW,
            },
        },
        "paper_citations": [
            "Ng et al. 1999 -- potential-based shaping",
            "Schulman et al. 2017 -- PPO (predecessor family; V24 uses SAC)",
            "Raffin et al. 2021 -- Stable-Baselines3",
            "Peng et al. 2018 -- dynamics randomization, obs/action noise",
            "Narvekar et al. 2020 -- curriculum learning survey",
            "Alshiekh et al. 2018 -- safe RL via shielding",
            "Gaudet et al. 2020 -- 6-DOF rocket landing (two-discount, fuel, glideslope, attitude)",
            "Towers et al. 2024 -- Gymnasium",
        ],
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
            "SafetyShield (post_posed, LTL-specified)",
            "ObservationDelay",
            "SensorNoise",
            "TimeLimit",
            "TargetLandingWrapper (Ng shaping + Gaudet terminal bonuses)",
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
    env_inner.set_scale(noise_scale)

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
        # V24 Gaudet knobs
        gamma_shaping=GAMMA_SHAPING,
        fuel_penalty_coef=FUEL_PENALTY_COEF,
        fuel_lateral_weight=FUEL_LATERAL_WEIGHT,
        glideslope_penalty_coef=GLIDESLOPE_PENALTY_COEF,
        glideslope_ratio_threshold=GLIDESLOPE_RATIO_THRESHOLD,
        glideslope_altitude=GLIDESLOPE_ALTITUDE,
        hard_attitude_limit=HARD_ATTITUDE_LIMIT,
    )

    log_name = "master_v24_eval" if is_eval else "master_v24_training"
    env = Monitor(
        env,
        filename=str(LOGS / log_name),
        info_keywords=("precise_landing",),
    )

    return env


def find_wrappers(env):

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
    """Anneals noise scale + target_limit on the TRAINING env."""

    def __init__(
        self,
        train_env,
        noise_start_step,
        noise_end_step,
        target_start_step,
        target_end_step,
        target_start_value,
        target_end_value,
        apply_every_n_calls=1000,
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
                f"CHECKPOINT {self.model.num_timesteps:,}", flush=True
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
                    f"[warn] best-model manifest failed: {e}",
                    flush=True,
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
            f"V24 needs V23 best as warm start, but {WARM_START} does not exist."
        )

    noise_start = int(NOISE_ANNEAL_START_FRAC * TOTAL_STEPS)
    noise_end = int(NOISE_ANNEAL_END_FRAC * TOTAL_STEPS)
    target_start = int(TARGET_ANNEAL_START_FRAC * TOTAL_STEPS)
    target_end = int(TARGET_ANNEAL_END_FRAC * TOTAL_STEPS)

    # Training env: curriculum starts at scale=0, target=0.15, weak_g=0.55.
    # SensorNoise / ActuatorNoise set to a reasonable scale; CurriculumCallback
    # resets scale to 0 at training_start (already-loaded env is overwritten).
    train_env = build_env(
        seed=SEED,
        noise_scale=0.0,
        target_limit=TARGET_LIMIT_START,
        weak_gravity_prob=INIT_WEAK_GRAVITY_PROB,
        is_eval=False,
    )

    # Eval env: held at FULL production regime so best_model is selected
    # on what we would ship.
    eval_env = build_env(
        seed=SEED + 101,
        noise_scale=1.0,
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
                    f"Loaded V23 replay buffer ({WARM_BUFFER.name}, "
                    f"size={model.replay_buffer.size():,}).",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[warn] V23 buffer load failed: {e}. Empty buffer.",
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
            f"V24 FINAL on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, "
            f"warm_start={WARM_START.name}, "
            f"gamma={GAMMA}, gamma_shaping={GAMMA_SHAPING}, "
            f"fuel_coef={FUEL_PENALTY_COEF}, glideslope_coef={GLIDESLOPE_PENALTY_COEF}, "
            f"hard_attitude={HARD_ATTITUDE_LIMIT}, "
            f"auto-curriculum weak_g=[{AUTO_CURR_MIN},{AUTO_CURR_MAX}], "
            f"shield ON (post_posed), lr {LR_START:.0e}->{LR_END:.0e} linear",
            flush=True,
        )

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

        auto_curr_cb = AutoCurriculumCallback(
            train_env=train_env,
            initial_prob=INIT_WEAK_GRAVITY_PROB,
            min_prob=AUTO_CURR_MIN,
            max_prob=AUTO_CURR_MAX,
            step_up=0.05,
            step_down=0.05,
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
                curriculum_cb,
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
                tb_log_name="v24",
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
        final_best = MODELS / "master_v24_best.zip"
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
                f"V24 best promoted to {final_best.name}", flush=True
            )
        else:
            print(
                "[warn] EvalCallback did not produce best_model.zip.",
                flush=True,
            )

        print(f"V24 DONE: {model.num_timesteps:,}", flush=True)

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":

    sys.stdout.reconfigure(line_buffering=True)
    main()
