"""V21: aggressive-precision fine-tune of V20 safety.

Starts from models/master_v20_safety.zip (not from scratch) and reshapes
the reward to drive the precision rate from ~72% up toward 99% without
losing V20's safety floor.

Design (locked):
  * SAFETY weights UNCHANGED from V20:
      failed_landing_penalty  = 500
      timeout_penalty         = 200
  * PRECISION turned up HARD:
      target_bonus            =  20  -> 300
      off_target_penalty      =   5  ->  80
      x_shaping_coef (new)    =   0  ->   5.0
    The x-shaping term adds a per-step -k * x^2 * ground_proximity
    gradient, so the agent gets a smooth signal to converge toward the
    pad center during the whole descent, not only at touchdown.
  * Env envelope UNCHANGED from V20 (same wrappers, same ranges, shield ON).
  * SAC fine-tune hparams (standard practice for continuing an already-good
    policy on a reshaped reward):
      learning_rate   = 3e-4 -> 1e-4   (gentler updates)
      learning_starts = 20k  -> 5k     (policy is warm, less random warmup)
      ent_coef        = auto           (SB3 keeps auto-tuning from V20's state)
      batch_size, buffer_size, tau, gamma: unchanged from V20.
  * Replay buffer starts EMPTY (SB3 does not persist it with .save()).
    The warm policy still runs; buffer refills during learning_starts.
  * Budget: 500k steps (~1/3 of V20's 1.5M).

Set V21_STEPS env var to shorten for a dry run (e.g. 50000).
"""
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper
from safety_shield import SafetyShield


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
WARM_START = MODELS / "master_v20_safety.zip"
CANDIDATE = MODELS / "master_v21_precision.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V21_STEPS", "500000"))
SAVE_INTERVAL = 25_000

LEARNING_RATE = 1e-4
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V21_SEED", "2141"))


def save_model(m, d):
    t = d.with_name(d.stem + "_temporary")
    tz = Path(str(t) + ".zip")
    tz.unlink(missing_ok=True)
    m.save(str(t))
    os.replace(tz, d)


class CB(BaseCallback):
    def __init__(self):
        super().__init__(verbose=1)
        self.last = 0

    def _on_training_start(self):
        self.last = self.model.num_timesteps

    def _on_step(self):
        if self.model.num_timesteps - self.last >= SAVE_INTERVAL:
            save_model(self.model, CANDIDATE)
            self.last = self.model.num_timesteps
            print(f"CHECKPOINT {self.model.num_timesteps:,}", flush=True)
        return True


def create_env():
    base = ThrustScaledLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 0.8),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),
        weak_gravity_prob=0.25,
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )
    base = SafetyShield(
        base,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
    )
    env = gym.wrappers.TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapper(
        env,
        gamma=GAMMA,
        step_penalty=0.01,
        target_bonus=300.0,                # 20 -> 300 (precision is PRIMARY)
        off_target_penalty=80.0,           # 5 -> 80
        timeout_penalty=200.0,             # unchanged from V20
        failed_landing_penalty=500.0,      # unchanged from V20
        target_limit=0.15,
        x_shaping_coef=5.0,                # NEW: per-step -5*x^2*ground_proximity
    )
    env = Monitor(env, filename=str(LOGS / "master_v21_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    if not WARM_START.exists():
        raise FileNotFoundError(
            f"V21 needs V20 as warm start, but {WARM_START} does not exist."
        )

    env = create_env()
    try:
        # Load V20 weights with the reshaped-reward env attached.
        # custom_objects overrides the hparams baked into V20's zip so the
        # fine-tune uses the gentler LR immediately.
        model = SAC.load(
            str(WARM_START),
            env=env,
            device=DEVICE,
            seed=SEED,
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

        gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
        print(
            f"V21 PRECISION FT on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, warm_start={WARM_START.name}, "
            f"target_bonus=300, off_target=80, x_shaping=5.0, "
            f"crash=500, timeout=200, shield ON, lr={LEARNING_RATE}",
            flush=True,
        )

        save_model(model, CANDIDATE)
        cb = CB()
        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=cb,
                log_interval=20,
                progress_bar=False,
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(f"V21 DONE: {model.num_timesteps:,}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
