import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from randomized_lander import RandomStartLunarLander

import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create randomised environment
env = RandomStartLunarLander(
    enable_wind=False
)

env = Monitor(
    env,
    filename="logs/randomized_training.csv"
)

# Load existing baseline AI
model = PPO.load(
    "models/ppo_lunar_lander_baseline",
    env=env
)

print("Training AI with random starting conditions!")

# Train for another 2 million steps
model.learn(
    total_timesteps=2_000_000,
    reset_num_timesteps=False,
    tb_log_name="randomized_landing"
)

# Save the new model separately
model.save(
    "models/ppo_randomized_landing"
)

env.close()

print("Randomized training completed!")