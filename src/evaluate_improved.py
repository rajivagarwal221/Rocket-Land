import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load improved AI model
model = PPO.load("models/ppo_improved_reward")

# Open Lunar Lander game
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episodes = 10
rewards = []
successful_landings = 0

for episode in range(episodes):

    observation, info = env.reset()
    total_reward = 0

    print(f"\nEpisode {episode + 1} started!")

    while True:

        # Let AI choose an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    # Check whether both legs are touching the ground
    both_legs = (
        observation[6] > 0.5
        and observation[7] > 0.5
    )

    rewards.append(total_reward)

    if both_legs and not truncated:
        successful_landings += 1

    print(f"Episode: {episode + 1}")
    print(f"Original Reward: {total_reward:.2f}")
    print(f"Both legs touching: {both_legs}")
    print(f"Time limit reached: {truncated}")

    time.sleep(2)

average_reward = sum(rewards) / episodes

print("\n========== FINAL RESULTS ==========")
print("Average Original Reward:", round(average_reward, 2))
print("Both-leg landing episodes:", successful_landings)
print("Total Episodes:", episodes)

env.close()