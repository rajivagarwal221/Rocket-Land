"""V16: fine-tune V15 with randomized target-pad width."""
import os
import sys
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from variable_pad_lander import VariablePadLunarLander
from target_landing import TargetLandingWrapper


ROOT = Path(__file__).resolve().parent.parent  # project root
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
BASE = MODELS / "master_v15_wind.zip"
CANDIDATE = MODELS / "master_v16_pad.zip"

TOTAL_STEPS = int(os.environ.get("V16_STEPS", "150000"))
SAVE_INTERVAL = 25_000
LEARNING_RATE = 5e-5
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
    base = VariablePadLunarLander(
        gravity=-10.0,
        gravity_range=(-11.5, -1.5),
        wind_power_range=(0.0, 12.0),
        turbulence_range=(0.0, 1.2),
        wind_probability=0.5,
        target_width_range=(0.10, 0.25),
        continuous=True,
    )
    env = gym.wrappers.TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    env = TargetLandingWrapper(env, gamma=0.99, failed_landing_penalty=100.0)
    env = Monitor(env, filename=str(LOGS / "master_v16_training"))
    return env


def main():
    MODELS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    env = create_env()
    try:
        model = PPO.load(str(BASE), env=env, device="cpu")
        model.tensorboard_log = None
        env.env.gamma = float(model.gamma)
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(LEARNING_RATE)
        for g in model.policy.optimizer.param_groups:
            g["lr"] = LEARNING_RATE
        model.set_random_seed(2030)
        print(f"V16 pad FT: {TOTAL_STEPS:,} steps, lr={LEARNING_RATE}", flush=True)
        save_model(model, CANDIDATE)
        cb = CB()
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
            print("Interrupted; saved.", flush=True)
            return
        save_model(model, CANDIDATE)
        print(f"V16 DONE: {model.num_timesteps:,}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
