"""V27: COLAB-SCALE PRODUCTION RUN -- the final Phase-1 push to 99% precise.

Not a surgical fine-tune. V27 is a FROM-SCRATCH training run on Colab T4
that uses the capacity the 3050 could not afford:

  - 4x VecFrameStack (8-frame temporal memory)   <- Peng 2018 recurrence gap
  - [512, 512] policy + Q                        <- 4x the capacity of v23
  - 4 parallel envs (SubprocVecEnv) on T4        <- 4x throughput
  - 8M steps (vs v23's 3M)                       <- long tail for precision
  - 3-phase curriculum (randomization -> tight)  <- Narvekar
  - batch_size 512, gradient_steps 2             <- T4 VRAM permits
  - cosine LR 3e-4 -> 1e-5                       <- final convergence

Grounds Peng 2018 (temporal memory via frame-stacking, not LSTM -- strictly
simpler, proven in OpenAI Dactyl, and avoids custom replay-buffer work),
Gaudet 2020 (long-horizon convergence via buffer size and gradient_steps),
Narvekar 2020 (3-phase curriculum on target_limit AND weak-gravity exposure),
Alshiekh 2018 (SafetyShield in the loop so the policy learns to not need it).

================================================================
V25 POST-MORTEM (which justifies v27's design)

V25 regressed v23 by 5% precise, in every slice:
  - reward additions (glideslope, hard_attitude) hurt weak-g
  - tighter target_limit=0.08 training made policy hover-defensive
  - shield-on and shield-off differ by <0.3% -> shield is not the lever;
    the policy itself is the lever

V27 removes every one of those failures:
  - reward is PURE v23 (no glideslope, no hard_attitude, no fuel)
  - target_limit ends at 0.10 for best-model selection (v23 regime)
  - shield still trains in-loop but we validate shield-off to 99%

================================================================
WHY FROM SCRATCH, NOT WARM-START

Frame-stacking changes the policy's observation dim from 9 to 72. The v23 MLP
weights cannot be loaded. The policy_net also grows from [256,256] to
[512,512]. From-scratch is the only honest option. The curriculum is designed
so the early phase (0-2M) learns v22/v23-level behavior, and the late phase
(5M-8M) exceeds v23.

================================================================
RUN ON COLAB T4

    !nvidia-smi
    !pip install -q stable-baselines3[extra]==2.3.2 gymnasium[box2d]==1.0.0 \
        sb3-contrib==2.3.0 tensorboard
    from google.colab import drive
    drive.mount('/content/drive')
    !mkdir -p /content/drive/MyDrive/rocket/models /content/drive/MyDrive/rocket/logs
    %env V27_OUT=/content/drive/MyDrive/rocket
    %env V27_STEPS=8000000
    !git clone https://github.com/<you>/Rocket.git /content/rocket
    %cd /content/rocket/src
    !python train_v27_colab.py

Wiring test (10 min):
    %env V27_STEPS=200000
    !python train_v27_colab.py

================================================================
DELIVERABLES (into V27_OUT)
  models/master_v27_production.zip   latest
  models/master_v27_best.zip         best under full-noise target=0.10
  models/master_v27_buffer.pkl       replay buffer
  models/master_v27_manifest.json
  models/master_v27_best_manifest.json
  logs/tb_v27/                       TensorBoard
  logs/eval_v27/evaluations.npz
"""
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
)

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper
from safety_shield import SafetyShield
from sensor_noise import (
    SensorNoise,
    ObservationDelay,
    DEFAULT_SENSOR_NOISE_STD,
)
from actuator_noise import ActuatorNoise


# =============================================================
# PATHS (Colab: set V27_OUT=/content/drive/MyDrive/rocket)
# =============================================================
ROOT = Path(__file__).resolve().parent.parent

OUT = Path(os.environ.get("V27_OUT", str(ROOT))).resolve()
MODELS = OUT / "models"
LOGS = OUT / "logs"
CANDIDATE = MODELS / "master_v27_production.zip"
BEST_MODEL = MODELS / "master_v27_best.zip"
BUFFER_PATH = MODELS / "master_v27_buffer.pkl"
MANIFEST_PATH = MODELS / "master_v27_manifest.json"
BEST_MANIFEST_PATH = MODELS / "master_v27_best_manifest.json"
TB_LOG = LOGS / "tb_v27"
EVAL_LOG = LOGS / "eval_v27"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# TRAIN CONFIG
# =============================================================
TOTAL_STEPS = int(os.environ.get("V27_STEPS", "8000000"))
SAVE_INTERVAL = 100_000
SEED = int(os.environ.get("V27_SEED", "7474"))

# Vec + frame-stack
N_ENVS = int(os.environ.get("V27_NENVS", "4"))
N_STACK = 8
USE_SUBPROC = os.environ.get("V27_SUBPROC", "1") == "1"

# SAC hyperparams
LR_START = 3e-4
LR_END = 1e-5
LEARNING_STARTS = 10_000
BATCH_SIZE = 512
BUFFER_SIZE = 2_000_000
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 4            # env steps between gradient blocks
GRADIENT_STEPS = 2        # gradient steps per block
MAX_EPISODE_STEPS = 1000

# Eval
EVAL_EVERY = 200_000
EVAL_EPISODES = 100

# Curriculum anchors as fractions of TOTAL_STEPS (so short dry-runs work)
PHASE1_END_FRAC = 2_000_000 / 8_000_000
PHASE2_END_FRAC = 5_000_000 / 8_000_000

TARGET_LIMIT_START = 0.15
TARGET_LIMIT_MID = 0.10
TARGET_LIMIT_END = 0.10
EVAL_TARGET_LIMIT = 0.10

WEAK_G_PROB_START = 0.35
WEAK_G_PROB_MID = 0.55
WEAK_G_PROB_END = 0.65

# Reward (v23 exactly)
TARGET_BONUS = 500.0
OFF_TARGET_PENALTY = 120.0
TIMEOUT_PENALTY = 200.0
FAILED_LANDING_PENALTY = 500.0
X_SHAPING_COEF = 4.5
X_SHAPING_FADE_BELOW = 0.12
ANGULAR_STABILITY_BONUS = 100.0
ANGULAR_STABILITY_ANGLE = 0.15
ANGULAR_STABILITY_OMEGA = 0.05

# Sim-to-real (production levels, reached at scale=1.0)
OBS_NOISE_STD = DEFAULT_SENSOR_NOISE_STD
OBS_DELAY_STEPS = 2
ACT_NOISE_STD = 0.02
ACT_LAG_ALPHA = 0.5


# =============================================================
# CURRICULUM PROXY WRAPPER
# Holds direct refs to the inner wrappers so the vec env can push
# curriculum updates through env_method("set_curriculum", ...).
# =============================================================
class CurriculumProxy(gym.Wrapper):
    def __init__(self, env, base_env, sensor_w, actuator_w, target_w):
        super().__init__(env)
        self._base_env = base_env
        self._sensor_w = sensor_w
        self._actuator_w = actuator_w
        self._target_w = target_w

    def set_curriculum(self, noise_scale: float, target_limit: float,
                       weak_g_prob: float):
        if self._sensor_w is not None:
            self._sensor_w.set_scale(float(noise_scale))
        if self._actuator_w is not None:
            self._actuator_w.set_scale(float(noise_scale))
        if self._target_w is not None:
            self._target_w.target_limit = float(target_limit)
        if self._base_env is not None and hasattr(
            self._base_env, "weak_gravity_prob"
        ):
            self._base_env.weak_gravity_prob = float(weak_g_prob)
        return True


# =============================================================
# MANIFEST
# =============================================================
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hparams_dict():
    return {
        "architecture": {
            "net_arch_pi": [512, 512],
            "net_arch_qf": [512, 512],
            "activation": "ReLU",
            "n_stack": N_STACK,
            "n_envs": N_ENVS,
            "vec_env": "SubprocVecEnv" if USE_SUBPROC else "DummyVecEnv",
        },
        "learning_rate_start": LR_START,
        "learning_rate_end": LR_END,
        "learning_rate_schedule": "cosine",
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "tau": TAU,
        "gamma": GAMMA,
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
        },
        "noise_production": {
            "obs_noise_std": OBS_NOISE_STD.tolist(),
            "obs_delay_steps": OBS_DELAY_STEPS,
            "act_noise_std": ACT_NOISE_STD,
            "act_lag_alpha": ACT_LAG_ALPHA,
        },
        "curriculum": {
            "phase1_end_frac": PHASE1_END_FRAC,
            "phase2_end_frac": PHASE2_END_FRAC,
            "target_limit_start": TARGET_LIMIT_START,
            "target_limit_mid": TARGET_LIMIT_MID,
            "target_limit_end": TARGET_LIMIT_END,
            "weak_g_prob_start": WEAK_G_PROB_START,
            "weak_g_prob_mid": WEAK_G_PROB_MID,
            "weak_g_prob_end": WEAK_G_PROB_END,
            "eval_target_limit": EVAL_TARGET_LIMIT,
        },
    }


def write_manifest(model_path: Path, manifest_path: Path, total_timesteps: int):
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
                torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
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
            "CurriculumProxy",
            "Monitor",
            ("SubprocVecEnv" if USE_SUBPROC else "DummyVecEnv"),
            "VecFrameStack(n_stack=8)",
            "VecMonitor",
        ],
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


# =============================================================
# ENV FACTORY
# =============================================================
def make_single_env(
    rank: int,
    seed: int,
    noise_scale: float,
    target_limit: float,
    weak_g_prob: float,
    is_eval: bool,
):
    def _init():
        base = ThrustScaledLunarLander(
            gravity=-10.0,
            gravity_range=(-11.5, -1.5),
            wind_power_range=(0.0, 12.0),
            turbulence_range=(0.0, 0.8),
            wind_probability=0.5,
            target_width_range=(0.10, 0.25),
            thrust_scale_range=(0.9, 1.3),
            weak_gravity_prob=weak_g_prob,
            weak_gravity_cutoff=-5.0,
            continuous=True,
        )

        act = ActuatorNoise(
            base,
            noise_std=ACT_NOISE_STD,
            lag_alpha=ACT_LAG_ALPHA,
            clip=True,
            seed=seed + 11 + rank,
        )
        act.set_scale(noise_scale)

        shielded = SafetyShield(
            act,
            y_trigger=0.45,
            vy_trigger=-0.55,
            main_thrust=1.0,
            angle_limit=0.6,
        )

        delayed = ObservationDelay(shielded, delay_steps=OBS_DELAY_STEPS)

        sens = SensorNoise(
            delayed,
            noise_std=OBS_NOISE_STD,
            seed=seed + 22 + rank,
        )
        sens.set_scale(noise_scale)

        timed = gym.wrappers.TimeLimit(sens, max_episode_steps=MAX_EPISODE_STEPS)

        target = TargetLandingWrapper(
            timed,
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

        # Proxy exposes set_curriculum() to the VecEnv.env_method path.
        proxied = CurriculumProxy(
            env=target,
            base_env=base,
            sensor_w=sens,
            actuator_w=act,
            target_w=target,
        )

        log_name = (
            "master_v27_eval" if is_eval else f"master_v27_train_w{rank}"
        )
        env = Monitor(proxied, filename=str(LOGS / log_name))
        env.reset(seed=seed + rank)
        return env

    return _init


def build_train_vec(seed: int):
    fns = [
        make_single_env(
            rank=i,
            seed=seed,
            noise_scale=0.0,
            target_limit=TARGET_LIMIT_START,
            weak_g_prob=WEAK_G_PROB_START,
            is_eval=False,
        )
        for i in range(N_ENVS)
    ]
    vec = None
    if USE_SUBPROC:
        try:
            # 'fork' works on Linux/Colab; 'spawn' is the Windows-safe fallback.
            start = "fork" if sys.platform != "win32" else "spawn"
            vec = SubprocVecEnv(fns, start_method=start)
        except Exception as e:
            print(
                f"[warn] SubprocVecEnv failed ({e}); using DummyVecEnv.",
                flush=True,
            )
    if vec is None:
        vec = DummyVecEnv(fns)
    vec = VecMonitor(vec, filename=str(LOGS / "master_v27_vec_training"))
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


def build_eval_vec(seed: int):
    fns = [
        make_single_env(
            rank=0,
            seed=seed + 101,
            noise_scale=1.0,
            target_limit=EVAL_TARGET_LIMIT,
            weak_g_prob=WEAK_G_PROB_END,
            is_eval=True,
        )
    ]
    vec = DummyVecEnv(fns)
    vec = VecMonitor(vec, filename=str(LOGS / "master_v27_vec_eval"))
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


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


class ThreePhaseCurriculum(BaseCallback):
    """3-phase curriculum pushed to every worker via env_method('set_curriculum')."""

    def __init__(
        self,
        train_vec,
        total_steps: int,
        p1_end: int,
        p2_end: int,
        apply_every_n_calls: int = 1000,
    ):
        super().__init__(verbose=0)
        self._vec = train_vec
        self._total = total_steps
        self._p1_end = p1_end
        self._p2_end = p2_end
        self._apply_every = apply_every_n_calls
        self._last_noise = None
        self._last_target = None
        self._last_weakg = None

    def _curve(self, t):
        # Noise: 0 through phase 1, 0 -> 1 across phase 2, hold 1 in phase 3
        if t <= self._p1_end:
            noise = 0.0
        elif t <= self._p2_end:
            noise = _lerp(t, self._p1_end, self._p2_end, 0.0, 1.0)
        else:
            noise = 1.0

        # Target: hold 0.15, then 0.15 -> 0.10, then hold 0.10
        if t <= self._p1_end:
            tgt = TARGET_LIMIT_START
        elif t <= self._p2_end:
            tgt = _lerp(
                t, self._p1_end, self._p2_end,
                TARGET_LIMIT_START, TARGET_LIMIT_MID,
            )
        else:
            tgt = _lerp(
                t, self._p2_end, self._total,
                TARGET_LIMIT_MID, TARGET_LIMIT_END,
            )

        # Weak-g: 0.35 -> 0.55 across phase 2, 0.55 -> 0.65 across phase 3
        if t <= self._p1_end:
            wg = WEAK_G_PROB_START
        elif t <= self._p2_end:
            wg = _lerp(
                t, self._p1_end, self._p2_end,
                WEAK_G_PROB_START, WEAK_G_PROB_MID,
            )
        else:
            wg = _lerp(
                t, self._p2_end, self._total,
                WEAK_G_PROB_MID, WEAK_G_PROB_END,
            )
        return noise, tgt, wg

    def _unwrap_to_base_vec(self):
        inner = self._vec
        # VecFrameStack -> VecMonitor -> (Subproc|Dummy)VecEnv
        while hasattr(inner, "venv"):
            inner = inner.venv
        return inner

    def _apply(self, t):
        noise, tgt, wg = self._curve(t)
        base_vec = self._unwrap_to_base_vec()
        try:
            base_vec.env_method(
                "set_curriculum", noise, tgt, wg, indices=None
            )
        except Exception as e:
            print(f"[curriculum] env_method failed: {e}", flush=True)

        if (
            self._last_noise is None
            or abs(noise - self._last_noise) >= 0.05
            or abs(tgt - (self._last_target or -1)) >= 0.005
            or abs(wg - (self._last_weakg or -1)) >= 0.02
        ):
            print(
                f"[curriculum] t={t:,} noise={noise:.3f} "
                f"target_limit={tgt:.3f} weak_g_prob={wg:.3f}",
                flush=True,
            )
            self._last_noise = noise
            self._last_target = tgt
            self._last_weakg = wg

    def _on_training_start(self):
        self._apply(0)

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
        write_manifest(candidate_path, manifest_path, model.num_timesteps)
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
            save_checkpoint(self.model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)
            self.last = self.model.num_timesteps
            print(f"CHECKPOINT {self.model.num_timesteps:,}", flush=True)
        return True


class RenameBestCB(BaseCallback):
    """EvalCallback saves as best_model.zip -- rename to master_v27_best.zip."""

    def _on_rollout_end(self):
        default_best = MODELS / "best_model.zip"
        if default_best.exists():
            try:
                default_best.replace(BEST_MODEL)
            except Exception as e:
                print(f"[warn] rename best_model failed: {e}", flush=True)
        return True

    def _on_step(self):
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
                    f"[eval] best-model manifest updated "
                    f"(t={self.model.num_timesteps:,}).",
                    flush=True,
                )
            except Exception as e:
                print(f"[warn] best-model manifest failed: {e}", flush=True)


# =============================================================
# LR SCHEDULE (cosine)
# =============================================================
def cosine_schedule(start, end):
    def _s(progress_remaining):
        frac = 1.0 - progress_remaining
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))
    return _s


# =============================================================
# MAIN
# =============================================================
def main():
    MODELS.mkdir(exist_ok=True, parents=True)
    LOGS.mkdir(exist_ok=True, parents=True)
    TB_LOG.mkdir(exist_ok=True, parents=True)
    EVAL_LOG.mkdir(exist_ok=True, parents=True)

    p1_end = int(PHASE1_END_FRAC * TOTAL_STEPS)
    p2_end = int(PHASE2_END_FRAC * TOTAL_STEPS)

    print(
        f"V27 COLAB on {DEVICE.upper()}: "
        f"{TOTAL_STEPS:,} steps, seed={SEED}, "
        f"{N_ENVS}x "
        f"{'SubprocVecEnv' if USE_SUBPROC else 'DummyVecEnv'}, "
        f"VecFrameStack(n={N_STACK}), net=[512,512], "
        f"batch={BATCH_SIZE}, buffer={BUFFER_SIZE:,}, "
        f"train_freq={TRAIN_FREQ}, grad_steps={GRADIENT_STEPS}, "
        f"lr cosine {LR_START:.0e}->{LR_END:.0e}, "
        f"phases: 0-{p1_end:,} warm | "
        f"{p1_end:,}-{p2_end:,} tighten | "
        f"{p2_end:,}-{TOTAL_STEPS:,} elite",
        flush=True,
    )

    train_env = build_train_vec(SEED)
    eval_env = build_eval_vec(SEED)

    lr_schedule = cosine_schedule(LR_START, LR_END)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512], qf=[512, 512]),
        activation_fn=nn.ReLU,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=(TRAIN_FREQ, "step"),
        gradient_steps=GRADIENT_STEPS,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log=str(TB_LOG),
    )

    gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[device] {gpu_name}", flush=True)
    print(f"[model] policy_params={total_params:,}", flush=True)

    save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)

    curriculum_cb = ThreePhaseCurriculum(
        train_vec=train_env,
        total_steps=TOTAL_STEPS,
        p1_end=p1_end,
        p2_end=p2_end,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS),
        log_path=str(EVAL_LOG),
        eval_freq=max(EVAL_EVERY // max(N_ENVS, 1), 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([
        curriculum_cb,
        eval_cb,
        RenameBestCB(),
        BestManifestCB(BEST_MODEL),
        CheckpointCB(),
    ])

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callbacks,
            log_interval=50,
            progress_bar=False,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("[train] keyboard interrupt -- saving final checkpoint.",
              flush=True)

    save_checkpoint(model, CANDIDATE, BUFFER_PATH, MANIFEST_PATH)
    # Final rename + best manifest
    default_best = MODELS / "best_model.zip"
    if default_best.exists():
        try:
            default_best.replace(BEST_MODEL)
        except Exception as e:
            print(f"[warn] final rename failed: {e}", flush=True)
    if BEST_MODEL.exists():
        try:
            write_manifest(BEST_MODEL, BEST_MANIFEST_PATH, model.num_timesteps)
        except Exception as e:
            print(f"[warn] final best manifest failed: {e}", flush=True)
    print(f"V27 done. t={model.num_timesteps:,}", flush=True)


if __name__ == "__main__":
    main()
