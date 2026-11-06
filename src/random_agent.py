import gymnasium as gym
import time

# Launch the actual game window
env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(3):

    observation, info = env.reset()
    total_reward = 0

    print(f"Episode {episode + 1} started!")

    while True:

        # Completely random rocket control
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print("Episode:", episode + 1)
    print("Total Reward:", round(total_reward, 2))

    time.sleep(2)

env.close()