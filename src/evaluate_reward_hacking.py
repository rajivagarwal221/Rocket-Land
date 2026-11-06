import gymnasium as gym
from stable_baselines3 import PPO
from reward_functions import HoverBonusWrapper
import time

# Load reward-hacking model
model = PPO.load("models/ppo_reward_hacking")

# Open visual game
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

# Apply the same reward wrapper
env = HoverBonusWrapper(env, bonus=1.0)

for episode in range(5):

    observation, info = env.reset()

    original_total = 0
    bonus_total = 0

    print(f"\nEpisode {episode + 1}")

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        original_total += info["original_reward"]
        bonus_total += info["hover_bonus"]

        if terminated or truncated:
            break

    print("Original reward:", round(original_total, 2))
    print("Hover bonus:", round(bonus_total, 2))
    print("Total shaped reward:", round(original_total + bonus_total, 2))

    time.sleep(2)

env.close()     