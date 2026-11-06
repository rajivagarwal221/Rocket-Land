"""V11 fine-tune: targets V9's three failure modes.

Non-interactive. Loads V9 candidate, fine-tunes with V11 wrapper + harder
start-state curriculum, writes v11 candidate.
"""

import os
import sys
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from randomized_lander import RandomStartLunarLander
from target_landing_v11 import TargetLandingWrapperV11


ROOT = Path(__file__).resolve().parent.parent  # project root
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

MASTER = MODELS / "master_continuous_v9_candidate.zip"
CANDIDATE = MODELS / "master_continuous_v11_candidate.zip"

TOTAL_STEPS = int(os.environ.get("V11_STEPS", "200000"))
SAVE_INTERVAL = 25_000
LEARNING_RATE = 1e-5
TARGET_KL = 0.01
TRAINING_SEED = 2026
MAX_EPISODE_STEPS = 1000
HARD_START_PROB = 0.35  # oversample hard lateral inits


def save_model(model, destination):
    temporary = destination.with_name(destination.stem + "_temporary")
    temporary_zip = Path(str(temporary) + ".zip")
    temporary_zip.unlink(missing_ok=True)
    model.save(str(temporary))
    if not temporary_zip.exists():
        raise RuntimeError("Checkpoint creation failed.")
    os.replace(temporary_zip, destination)


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
            print(
                f"CHECKPOINT {self.model.num_timesteps:,}", flush=True
            )
        return True


def create_environment():
    base_env = RandomStartLunarLander(
        render_mode=None,
        enable_wind=False,
        hard_start_probability=HARD_START_PROB,
        continuous=True,
    )
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapperV11(
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=70.0,
        off_target_penalty=25.0,
        timeout_penalty=40.0,
        failed_landing_penalty=150.0,
        both_leg_bonus=25.0,
        settle_frames=15,
    )
    env = Monitor(env, filename=str(LOGS / "master_precision_v11_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    if not MASTER.exists():
        raise FileNotFoundError(MASTER)

    env = create_environment()
    try:
        print("Loading V9 continuous master...", flush=True)
        model = PPO.load(str(MASTER), env=env, device="cpu")
        model.tensorboard_log = None  # scrub stale path from source machine

        if not isinstance(model.action_space, gym.spaces.Box):
            raise RuntimeError("Loaded model is not continuous PPO.")

        env.env.gamma = float(model.gamma)

        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(LEARNING_RATE)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        model.target_kl = TARGET_KL
        model.set_random_seed(TRAINING_SEED)

        print(
            f"V11 FT start: steps={TOTAL_STEPS:,} lr={LEARNING_RATE} "
            f"hard_start={HARD_START_PROB}",
            flush=True,
        )

        save_model(model, CANDIDATE)
        callback = CheckpointCallback()

        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=callback,
                reset_num_timesteps=False,
                log_interval=5,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted. Saved candidate.", flush=True)
            return

        save_model(model, CANDIDATE)
        print(
            f"V11 TRAINING DONE: {model.num_timesteps:,} total steps",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
