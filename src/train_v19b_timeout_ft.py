"""V19b: short SAC fine-tune of V19 targeting the timeout bucket.

V19 (master_v19_sac.zip) wins on precision but regressed on TIMEOUT
(4% -> 8.5%). Timeouts happen almost entirely in weak-gravity episodes
where the agent hovers instead of committing to descent.

V19b warm-starts from V19's replay buffer and policy, keeps V17's reward
scale, and turns on the commit penalty (step_penalty x2 past step 500
when still above y=0.3). Also mildly oversamples weak gravity so the
fresh samples the agent sees are the ones it is losing on.

Quick: 100k steps, ~10 min on an RTX 3050.
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
BASE = MODELS / "master_v19_sac.zip"
CANDIDATE = MODELS / "master_v19b_sac.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_STEPS = int(os.environ.get("V19B_STEPS", "100000"))
SAVE_INTERVAL = 10_000
LEARNING_RATE = 1e-4           # lower than V19's 3e-4 for stable fine-tune
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
        weak_gravity_prob=0.35,          # a bit higher than V19's 0.25
        weak_gravity_cutoff=-5.0,
        continuous=True,
    )
    env = gym.wrappers.TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapper(
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=50.0,               # keep V17/V19 scale
        off_target_penalty=20.0,
        failed_landing_penalty=100.0,
        target_limit=0.15,
        # V19b: anti-hover pressure past step 500 above y=0.3
        commit_penalty_after=500,
        commit_penalty_height=0.3,
        commit_penalty_scale=2.0,
    )
    env = Monitor(env, filename=str(LOGS / "master_v19b_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if not BASE.exists():
        raise FileNotFoundError(BASE)

    env = create_env()
    try:
        # Warm-start from V19 weights + replay buffer topology.
        model = SAC.load(str(BASE), env=env, device=DEVICE)
        model.learning_rate = LEARNING_RATE
        # SAC has separate actor / critic / entropy optimizers; patch all.
        for opt_attr in ("actor", "critic", "ent_coef_optimizer"):
            obj = getattr(model, opt_attr, None)
            opt = obj if opt_attr == "ent_coef_optimizer" else getattr(obj, "optimizer", None)
            if opt is None:
                continue
            for g in opt.param_groups:
                g["lr"] = LEARNING_RATE
        model.set_random_seed(2035)

        gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
        print(
            f"V19b SAC fine-tune on {DEVICE.upper()} ({gpu_name}): "
            f"{TOTAL_STEPS:,} steps, lr={LEARNING_RATE}, commit_penalty on, "
            f"weak_grav_prob=0.35",
            flush=True,
        )
        save_model(model, CANDIDATE)
        cb = CB()
        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=cb,
                reset_num_timesteps=False,
                log_interval=10,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(f"V19b DONE: {model.num_timesteps:,}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
