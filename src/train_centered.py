import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Original Lunar Lander reward
env = gym.make("LunarLander-v3")

env = Monitor(
    env,
    filename="logs/centered_training.csv"
)

# Load our existing baseline AI
model = PPO.load(
    "models/ppo_lunar_lander_baseline",
    env=env
)

print("Continuing baseline training...")

# Train for another 2 million steps
model.learn(
    total_timesteps=2_000_000,
    reset_num_timesteps=False,
    tb_log_name="centered_landing"
)

# Save the improved model separately
model.save("models/ppo_lunar_lander_centered")

env.close()

print("Centered landing training completed!")