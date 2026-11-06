import gymnasium as gym
from stable_baselines3 import PPO
from randomized_lander import RandomStartLunarLander

MODELS = {
    "BEST BACKUP": "models/master_backup",
    "CURRENT MASTER": "models/ppo_stage1_precision_v2"
}

EPISODES = 100

for name, path in MODELS.items():

    print(f"\n========== {name} ==========")

    model = PPO.load(path)

    env = RandomStartLunarLander(
        enable_wind=False
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=1000
    )

    safe = 0
    centered = 0
    crashes = 0
    timeouts = 0
    rewards = []

    for episode in range(EPISODES):

        obs, info = env.reset(
            seed=1000 + episode
        )

        total_reward = 0
        final_reward = 0

        while True:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            final_reward = reward

            if terminated or truncated:
                break

        x = float(obs[0])

        both_legs = (
            obs[6] > 0.5 and obs[7] > 0.5
        )

        successful = (
            terminated and final_reward > 0
        )

        precision = (
            successful
            and both_legs
            and -0.15 < x < 0.15
        )

        if successful:
            safe += 1

        if precision:
            centered += 1

        if truncated:
            timeouts += 1

        elif terminated and not successful:
            crashes += 1

        rewards.append(total_reward)

    env.close()

    print("Episodes:", EPISODES)
    print("Safe landings:", safe)
    print("Centered landings:", centered)
    print("Crashes:", crashes)
    print("Timeouts:", timeouts)

    print(
        "Safe landing rate:",
        round(safe / EPISODES * 100, 2), "%"
    )

    print(
        "Precision accuracy:",
        round(centered / EPISODES * 100, 2), "%"
    )

    print(
        "Average reward:",
        round(sum(rewards) / EPISODES, 2)
    )