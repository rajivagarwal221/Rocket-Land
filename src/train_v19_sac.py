"""V19: Soft Actor-Critic (SAC) with a larger policy network.

Trades PPO for SAC: an off-policy algorithm with twin critics and an
entropy bonus. On continuous-action LunarLander, SAC generally beats PPO
by several reward points due to better sample efficiency and smoother
policy updates.

V17 is the discrete-chain hand-off point; we don't warm-start weights
(SAC and PPO have different network topologies), but we DO train on the
same full-stack randomised env (random spawn + 9-planet gravity + wind +
pad tolerance + thrust scaling + target shaping).
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


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
CANDIDATE = MODELS / "master_v19_sac.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V19_STEPS", "500000"))
SAVE_INTERVAL = 25_000
LEARNING_RATE = 3e-4           # SAC typically uses ~3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 300_000
LEARNING_STARTS = 10_000       # fill buffer with random actions first
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
MAX_EPISODE_STEPS = 1000


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
        turbulence_range=(0.0, 1.2),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        thrust_scale_range=(0.8, 1.3),
        weak_gravity_prob=0.25,
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )
    env = gym.wrappers.TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapper(
        env,
        gamma=GAMMA,
        step_penalty=0.01,
        target_bonus=50.0,          # keep V17's proven scale
        off_target_penalty=20.0,
        failed_landing_penalty=100.0,
        target_limit=0.15,
    )
    env = Monitor(env, filename=str(LOGS / "master_v19_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    env = create_env()
    try:
        # Larger network to use SAC's sample efficiency well.
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
            ent_coef="auto",           # auto-tune entropy coefficient
            policy_kwargs=policy_kwargs,
            device=DEVICE,
            seed=2034,
            verbose=0,
        )
        gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
        print(
            f"V19 SAC on {DEVICE.upper()} ({gpu_name}): {TOTAL_STEPS:,} steps, "
            f"net=[256,256], lr={LEARNING_RATE}, batch={BATCH_SIZE}, "
            f"buffer={BUFFER_SIZE:,}",
            flush=True,
        )
        save_model(model, CANDIDATE)
        cb = CB()
        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=cb,
                log_interval=10,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(f"V19 DONE: {model.num_timesteps:,}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
