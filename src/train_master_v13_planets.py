"""V13: fine-tune V9 with randomized gravity (planet-generalist).

Every training episode draws a fresh gravity from [-11.5, -1.5], so the
policy must learn to infer gravity from early dynamics and adapt its
thrust usage. Eval model: V9's wrapper reward (unchanged) so we do not
re-open the V11/V12 regression.
"""
import os
import sys
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from planet_lander import PlanetLunarLander
from target_landing import TargetLandingWrapper


ROOT = Path(__file__).resolve().parent.parent  # project root
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

MASTER = MODELS / "master_continuous_v9_candidate.zip"
CANDIDATE = MODELS / "master_planet_v13_candidate.zip"

TOTAL_STEPS = int(os.environ.get("V13_STEPS", "300000"))
SAVE_INTERVAL = 25_000
LEARNING_RATE = 1e-5
TARGET_KL = 0.015  # slightly looser to let gravity adaptation land
TRAINING_SEED = 2027
MAX_EPISODE_STEPS = 1000

GRAVITY_RANGE = (-11.5, -1.5)


def save_model(model, dest):
    tmp = dest.with_name(dest.stem + "_temporary")
    tmp_zip = Path(str(tmp) + ".zip")
    tmp_zip.unlink(missing_ok=True)
    model.save(str(tmp))
    if not tmp_zip.exists():
        raise RuntimeError("checkpoint failed")
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
    base = PlanetLunarLander(
        gravity=-10.0,
        gravity_range=GRAVITY_RANGE,
        render_mode=None,
        enable_wind=False,
        hard_start_probability=0.0,
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
    env = Monitor(env, filename=str(LOGS / "master_v13_planets_training"))
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
        model.tensorboard_log = None

        if not isinstance(model.action_space, gym.spaces.Box):
            raise RuntimeError("Loaded model is not continuous PPO.")

        env.env.gamma = float(model.gamma)
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(LEARNING_RATE)
        for g in model.policy.optimizer.param_groups:
            g["lr"] = LEARNING_RATE
        model.target_kl = TARGET_KL
        model.set_random_seed(TRAINING_SEED)

        print(
            f"V13 FT: steps={TOTAL_STEPS:,} lr={LEARNING_RATE} "
            f"gravity_range={GRAVITY_RANGE}",
            flush=True,
        )

        save_model(model, CANDIDATE)
        cb = CheckpointCallback()
        try:
            model.learn(
                total_timesteps=TOTAL_STEPS,
                callback=cb,
                reset_num_timesteps=False,
                log_interval=5,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            save_model(model, CANDIDATE)
            print("Interrupted; candidate saved.", flush=True)
            return

        save_model(model, CANDIDATE)
        print(
            f"V13 TRAINING DONE: {model.num_timesteps:,} total steps",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
