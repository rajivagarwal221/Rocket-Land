"""V14: fresh 9-dim obs policy, warm-started from V9 where shapes match.

Trains across randomized gravity in [-11.5, -1.5] with gravity as a
normalized obs dim, so the policy can specialize per planet.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gravity_obs_lander import GravityObsLunarLander
from target_landing import TargetLandingWrapper


ROOT = Path(__file__).resolve().parent.parent  # project root
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
V9 = MODELS / "master_continuous_v9_candidate.zip"
CANDIDATE = MODELS / "master_v14_gravity_aware.zip"

TOTAL_STEPS = int(os.environ.get("V14_STEPS", "300000"))
SAVE_INTERVAL = 50_000
LEARNING_RATE = 3e-4  # higher: this is effectively a near-fresh policy
MAX_EPISODE_STEPS = 1000
GRAVITY_RANGE = (-11.5, -1.5)


def save_model(model, dest):
    tmp = dest.with_name(dest.stem + "_temporary")
    tmp_zip = Path(str(tmp) + ".zip")
    tmp_zip.unlink(missing_ok=True)
    model.save(str(tmp))
    os.replace(tmp_zip, dest)


class CheckpointCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=1)
        self.last_save = 0

    def _on_training_start(self):
        self.last_save = self.model.num_timesteps

    def _on_step(self):
        if self.model.num_timesteps - self.last_save >= SAVE_INTERVAL:
            save_model(self.model, CANDIDATE)
            self.last_save = self.model.num_timesteps
            print(f"CHECKPOINT {self.model.num_timesteps:,}", flush=True)
        return True


def create_environment():
    base = GravityObsLunarLander(
        gravity=-10.0,
        gravity_range=GRAVITY_RANGE,
        render_mode=None,
        enable_wind=False,
        continuous=True,
    )
    env = gym.wrappers.TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapper(
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=50.0,
        off_target_penalty=20.0,
        timeout_penalty=30.0,
        failed_landing_penalty=100.0,
        settle_frames=15,
    )
    env = Monitor(env, filename=str(LOGS / "master_v14_training"))
    return env


def warm_start(new_model, v9_path):
    """Copy V9 weights into new 9-dim model where shapes match; pad the
    first-layer input with zeros so day-1 behaviour is identical on
    whatever gravity the fresh column defaults to (0 contribution)."""
    v9 = PPO.load(v9_path, device="cpu")
    v9_sd = v9.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    copied = padded = skipped = 0
    for k, new_v in new_sd.items():
        if k not in v9_sd:
            skipped += 1
            continue
        old_v = v9_sd[k]
        if old_v.shape == new_v.shape:
            new_sd[k] = old_v.clone()
            copied += 1
        elif old_v.dim() == 2 and new_v.dim() == 2 and \
                old_v.shape[0] == new_v.shape[0] and \
                old_v.shape[1] + 1 == new_v.shape[1]:
            # First linear: (hidden, 8) -> (hidden, 9). Pad zero column.
            padded_tensor = torch.zeros_like(new_v)
            padded_tensor[:, : old_v.shape[1]] = old_v
            new_sd[k] = padded_tensor
            padded += 1
        else:
            skipped += 1
    new_model.policy.load_state_dict(new_sd)
    print(
        f"Warm start: {copied} copied, {padded} padded, {skipped} skipped",
        flush=True,
    )


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if not V9.exists():
        raise FileNotFoundError(V9)

    env = create_environment()
    try:
        print("Building fresh 9-dim PPO...", flush=True)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device="cpu",
            seed=2028,
            verbose=0,
        )

        print("Warm-starting from V9...", flush=True)
        warm_start(model, str(V9))

        print(
            f"V14 FT: steps={TOTAL_STEPS:,} lr={LEARNING_RATE} "
            f"gravity_range={GRAVITY_RANGE} (9-dim obs)",
            flush=True,
        )
        save_model(model, CANDIDATE)
        cb = CheckpointCallback()
        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=cb,
                reset_num_timesteps=True,
                log_interval=5,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(
            f"V14 TRAINING DONE: {model.num_timesteps:,} steps", flush=True
        )
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
