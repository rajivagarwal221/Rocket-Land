import gymnasium as gym
from stable_baselines3 import PPO
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

# Load the trained model
model = PPO.load(str(ROOT / "models/ppo_lunar_lander_baseline"))

# Open the actual Lunar Lander game
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episodes = 5

for episode in range(episodes):

    observation, info = env.reset()
    total_reward = 0

    print(f"\nEpisode {episode + 1} started!")

    while True:

        # AI chooses the best action it has learned
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Episode: {episode + 1}")
    print(f"Total Reward: {total_reward:.2f}")

    time.sleep(2)

env.close()

print("\nAI evaluation completed!")