import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

# Create folders
(ROOT / "models/checkpoints").mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

# Create Lunar Lander environment
env = gym.make("LunarLander-v3")
env = Monitor(env, filename=str(ROOT / "logs/baseline_monitor.csv"))

# Save checkpoints every 100,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=str(ROOT / "models/checkpoints"),
    name_prefix="ppo_lunar_lander"
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=str(ROOT / "logs/tensorboard/"),
    seed=42
)

print("Starting PPO Training...")

# Train for 1 million steps
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback,
    tb_log_name="baseline"
)

# Save final model
model.save(str(ROOT / "models/ppo_lunar_lander_baseline"))

env.close()

print("Training completed!")
print("Model saved successfully.")