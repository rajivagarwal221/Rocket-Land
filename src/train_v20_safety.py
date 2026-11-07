"""V20: safety-first SAC with hard shield near the ground.

Target: 99% safe landing (CRASH + TIMEOUT < 1% combined) across the
full 9-planet / wind / variable-thrust / variable-pad randomisation.

Design changes vs V19:
  * Reward is SAFETY-DOMINANT, not precision-first:
      failed_landing_penalty  100 -> 500
      timeout_penalty         30  -> 200
      target_bonus            50  -> 20  (precision becomes a tie-breaker)
      off_target_penalty      20  -> 5
  * Physically-impossible seeds softened:
      thrust_scale_range      (0.8, 1.3) -> (0.9, 1.3)
      turbulence_range        (0.0, 1.2) -> (0.0, 0.8)
  * SafetyShield wraps the base env: whenever the lander is below y=0.45
    AND descending faster than vy=-0.55, the shield overrides the main
    engine to full thrust. Classical rocket-guidance floor beneath the
    learned policy.
  * Longer training: 1.5M steps (3x V19), buffer 1M, learning_starts
    scaled accordingly.
  * Train from scratch (not fine-tune) — the reward surface is different
    enough that warm-starting V19 would chase a different objective.

Expected runtime on RTX 3050: ~3-4 h GPU, ~12-14 h CPU.
Set V20_STEPS env var to shorten for a dry run (e.g. 200000).
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
CANDIDATE = MODELS / "master_v20_safety.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V20_STEPS", "1500000"))
SAVE_INTERVAL = 50_000
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 20_000
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000
SEED = int(os.environ.get("V20_SEED", "2040"))


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
        turbulence_range=(0.0, 0.8),       # softened from 1.2
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.9, 1.3),     # softened from 0.8
        weak_gravity_prob=0.25,
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )
    # Shield sits BETWEEN base and TimeLimit so it sees true physics obs.
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
        target_bonus=20.0,                 # precision = tie-breaker
        off_target_penalty=5.0,
        timeout_penalty=200.0,             # big terminal cost for hovering out
        failed_landing_penalty=500.0,      # crashing is CATASTROPHIC
        target_limit=0.15,
    )
    env = Monitor(env, filename=str(LOGS / "master_v20_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    env = create_env()
    try:
        policy_kwargs = {"net_arch": [256, 256]}

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            device=DEVICE,
            seed=SEED,
            verbose=0,
        )
        gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
        print(
            f"V20 SAFETY-FIRST SAC on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, seed={SEED}, "
            f"crash_penalty=500, timeout_penalty=200, shield ON",
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
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(f"V20 DONE: {model.num_timesteps:,}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
