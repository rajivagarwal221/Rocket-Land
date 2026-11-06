import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_functions import ImprovedRewardWrapper
import os

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create Lunar Lander environment
env = gym.make("LunarLander-v3")

# Apply improved reward function
env = ImprovedRewardWrapper(env, step_penalty=0.05)

# Monitor training
env = Monitor(env, filename="logs/improved_reward.csv")

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    tensorboard_log="./logs/tensorboard/"
)

print("Starting improved reward training...")

# Train for 1 million steps
model.learn(
    total_timesteps=1_000_000,
    tb_log_name="improved_reward"
)

# Save trained model
model.save("models/ppo_improved_reward")

env.close()

print("Improved reward training completed!")


