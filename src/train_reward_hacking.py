import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_functions import HoverBonusWrapper

import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create original environment
env = gym.make("LunarLander-v3")

# Apply custom reward wrapper
env = HoverBonusWrapper(env, bonus=1.0)

# Monitor training
env = Monitor(
    env,
    filename="logs/reward_hacking.csv"
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    tensorboard_log="./logs/tensorboard/"
)

print("Starting reward hacking experiment...")

# Train the AI
model.learn(
    total_timesteps=1_000_000,
    tb_log_name="reward_hacking"
)

# Save the model
model.save("models/ppo_reward_hacking")

env.close()

print("Reward hacking training completed!")