import os
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from randomized_lander import RandomStartLunarLander
from target_landing import TargetLandingWrapper


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent  # project root (Lunar-Lander-RL/)

MODELS = ROOT / "models"
LOGS = ROOT / "logs"

MASTER = MODELS / "master_continuous_v9_candidate.zip"

CANDIDATE = MODELS / "master_continuous_v10_candidate.zip"

# ==========================================
# 2. TRAINING SETTINGS
# ==========================================

TOTAL_STEPS = 300_000

SAVE_INTERVAL = 50_000

LEARNING_RATE = 0.00001

TARGET_KL = 0.01

TRAINING_SEED = 2026

MAX_EPISODE_STEPS = 1000


# ==========================================
# 3. SAFE MODEL SAVING
# ==========================================

def save_model(model, destination):

    temporary = destination.with_name(
        destination.stem + "_temporary"
    )

    temporary_zip = Path(
        str(temporary) + ".zip"
    )

    temporary_zip.unlink(missing_ok=True)

    model.save(str(temporary))

    if not temporary_zip.exists():
        raise RuntimeError("Checkpoint creation failed.")

    os.replace(
        temporary_zip,
        destination
    )


# ==========================================
# 4. TRAINING CHECKPOINT CALLBACK
# ==========================================

class CheckpointCallback(BaseCallback):

    def __init__(self):

        super().__init__(verbose=1)

        self.last_save = 0

    def _on_training_start(self):

        self.last_save = self.model.num_timesteps

    def _on_step(self):

        if (
            self.model.num_timesteps - self.last_save
            >= SAVE_INTERVAL
        ):

            save_model(
                self.model,
                CANDIDATE
            )

            self.last_save = self.model.num_timesteps

            print(
                f"\nCHECKPOINT SAVED: "
                f"{self.model.num_timesteps:,}",
                flush=True
            )

        return True


# ==========================================
# 5. CREATE CONTINUOUS ENVIRONMENT
# ==========================================

def create_environment():

    base_env = RandomStartLunarLander(

        render_mode=None,

        enable_wind=False,

        hard_start_probability=0.0,

        continuous=True

    )

    env = gym.wrappers.TimeLimit(

        base_env,

        max_episode_steps=MAX_EPISODE_STEPS

    )

    env = TargetLandingWrapper(

        env,

        gamma=0.99,

        step_penalty=0.01,

        target_bonus=50.0,

        off_target_penalty=20.0,

        timeout_penalty=30.0,

        failed_landing_penalty=100.0,

        settle_frames=15

    )

    env = Monitor(
        env,
        filename=str(
            LOGS / "master_precision_v9_training"
        )
    )

    return env


# ==========================================
# 6. MAIN TRAINING
# ==========================================

def main():

    MODELS.mkdir(exist_ok=True)

    LOGS.mkdir(exist_ok=True)

    if not MASTER.exists():
        raise FileNotFoundError(MASTER)


    if CANDIDATE.exists():

        raise FileExistsError(
            "master_continuous_v8_candidate.zip already exists. "
            "Rename the previous candidate before continuing."
        )

    env = create_environment()

    try:

        # ----------------------------------
        # LOAD CONTINUOUS MASTER
        # ----------------------------------

        print("\nLoading V7 continuous master...")

        model = PPO.load(

            str(MASTER),

            env=env,

            device="cpu"

        )

        if not isinstance(
            model.action_space,
            gym.spaces.Box
        ):

            raise RuntimeError(
                "The loaded model is not continuous PPO."
            )

        env.env.gamma = float(model.gamma)

        print("\nV7 CONTINUOUS MASTER LOADED")

        print(
            "Previous timesteps:",
            f"{model.num_timesteps:,}"
        )

        print(
            "Action space:",
            model.action_space
        )

        # ----------------------------------
        # PPO CONFIGURATION
        # ----------------------------------

        model.learning_rate = LEARNING_RATE

        model.lr_schedule = get_schedule_fn(
            LEARNING_RATE
        )

        for group in model.policy.optimizer.param_groups:

            group["lr"] = LEARNING_RATE

        model.target_kl = TARGET_KL

        model.set_random_seed(TRAINING_SEED)

        print("\n========== TRAINING CONFIGURATION ==========")

        print("Additional steps:", TOTAL_STEPS)

        print("Learning rate:", LEARNING_RATE)

        print("Target KL:", TARGET_KL)

        print("Gamma:", model.gamma)

        print("Continuous control: ENABLED")

        print("Unified reward: V6")
        print("Two-leg contact shaping: ENABLED")
        print("Stable touchdown: 15 frames")
        print("V7 policy: STARTING POINT")

        print("Stable touchdown: 15 frames")

        print("Failed landing penalty: -100")

        print("Original champion: PROTECTED")

        # ----------------------------------
        # CONFIRMATION
        # ----------------------------------

        approval = input(
            "\nType TRAIN to begin: "
        ).strip().upper()

        if approval != "TRAIN":

            print("Training cancelled.")

            return

        # ----------------------------------
        # INITIAL CANDIDATE CHECKPOINT
        # ----------------------------------

        save_model(
            model,
            CANDIDATE
        )

        # ----------------------------------
        # TRAIN SAME CONTINUOUS POLICY
        # ----------------------------------

        callback = CheckpointCallback()

        print("\n======================================")

        print("V9 CONTINUOUS MASTER TRAINING STARTED")

        print("======================================")

        try:

            model.learn(

                total_timesteps=TOTAL_STEPS,

                callback=callback,

                reset_num_timesteps=False,

                tb_log_name="master_precision_v9",

                log_interval=1,

                progress_bar=False

            )

        except KeyboardInterrupt:

            save_model(
                model,
                CANDIDATE
            )

            print("\nTraining interrupted.")

            print("Latest candidate saved.")

            return

        # ----------------------------------
        # FINAL CHECKPOINT
        # ----------------------------------

        save_model(
            model,
            CANDIDATE
        )

        print("\n======================================")

        print("V9 TRAINING COMPLETED")

        print(
           "Final timesteps:",
           f"{model.num_timesteps:,}"
       )

        print("New candidate:", CANDIDATE)

        print("V7 starting model preserved:", MASTER)

        print("======================================")

    finally:

        env.close()


# ==========================================
# 7. RUN
# ==========================================

if __name__ == "__main__":

    main()