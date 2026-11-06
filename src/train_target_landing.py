import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from randomized_lander import RandomStartLunarLander
from target_landing import TargetLandingWrapper

import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create randomized environment
env = RandomStartLunarLander(
    enable_wind=False
)

# Set episode limit
env = gym.wrappers.TimeLimit(
    env,
    max_episode_steps=1000
)

# Apply improved reward shaping
env = TargetLandingWrapper(
    env,
    gamma=0.99
)

env = Monitor(
    env,
    filename="logs/precision_v2.csv"
)

# Start from Stage 1 model
model = PPO.load(
    "models/ppo_curriculum_stage1",
    env=env
)

print("Starting Precision V2 Training...")

model.learn(
    total_timesteps=1_000_000,
    reset_num_timesteps=False,
    tb_log_name="precision_v2"
)

# Save V2 separately
model.save(
    "models/ppo_stage1_precision_v2"
)

env.close()

print("Precision V2 training completed!")