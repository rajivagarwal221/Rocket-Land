import gymnasium as gym
from stable_baselines3 import PPO
import time

model = PPO.load(
    "models/ppo_lunar_lander_centered"
)

env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episodes = 10

centered_landings = 0
safe_landings = 0

for episode in range(episodes):

    observation, info = env.reset()

    total_reward = 0
    final_reward = 0

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        final_reward = reward

        if terminated or truncated:
            break

    # Rocket's final horizontal position
    x_position = observation[0]

    # Check successful resting
    safe_landing = terminated and final_reward > 0

    # Check whether rocket is between the flags
    between_flags = abs(x_position) <= 0.20

    if safe_landing:
        safe_landings += 1

    if safe_landing and between_flags:
        centered_landings += 1
        result = "SUCCESS - LANDED BETWEEN FLAGS"

    elif safe_landing:
        result = "LANDED OUTSIDE FLAGS"

    else:
        result = "CRASH OR TIMEOUT"

    print(f"\nEpisode {episode + 1}")
    print(f"Reward: {total_reward:.2f}")
    print(f"Horizontal position: {x_position:.3f}")
    print(f"Result: {result}")

    time.sleep(2)

env.close()

print("\n========== FINAL RESULTS ==========")
print("Safe landings:", safe_landings)
print("Centered landings:", centered_landings)
print("Total episodes:", episodes)
print("Landing accuracy:", centered_landings * 10, "%")