import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from randomized_lander import RandomStartLunarLander

import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create Stage 1 environment
env = RandomStartLunarLander(
    enable_wind=False
)

env = Monitor(
    env,
    filename="logs/curriculum_stage1.csv"
)

# Start with the original successful baseline model
model = PPO.load(
    "models/ppo_lunar_lander_baseline",
    env=env
)

print("Starting Curriculum Stage 1...")

model.learn(
    total_timesteps=1_000_000,
    reset_num_timesteps=False,
    tb_log_name="curriculum_stage1"
)

model.save(
    "models/ppo_curriculum_stage1"
)

env.close()

print("Stage 1 training completed!")