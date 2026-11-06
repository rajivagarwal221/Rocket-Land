import gymnasium as gym
from stable_baselines3 import PPO
from randomized_lander import RandomStartLunarLander
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

MODELS = {
    "SELECTED MASTER": str(ROOT / "models" / "ppo_stage1_precision_v2"),
    "PREVIOUS MASTER": str(ROOT / "models" / "current_master_archive"),
}

EPISODES = 500

TARGET_LEFT = -0.15
TARGET_RIGHT = 0.15


def evaluate(name, model_path):

    print(f"\n========== {name} ==========")

    model = PPO.load(model_path, device="cpu")

    env = RandomStartLunarLander(
        enable_wind=False
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=1000
    )

    safe = 0
    centered = 0
    outside = 0
    crashes = 0
    timeouts = 0

    rewards = []

    for episode in range(EPISODES):

        obs, info = env.reset(
            seed=2000 + episode
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

        inside_target = (
            TARGET_LEFT < x < TARGET_RIGHT
        )

        precision = (
            successful
            and both_legs
            and inside_target
        )

        if precision:
            centered += 1

        if successful:
            safe += 1

            if not precision:
                outside += 1

        elif truncated:
            timeouts += 1

        else:
            crashes += 1

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"Completed {episode + 1}/{EPISODES}"
            )

    env.close()

    print("\nFINAL RESULTS")
    print("Episodes:", EPISODES)
    print("Safe landings:", safe)
    print("Centered landings:", centered)
    print("Landings outside target:", outside)
    print("Crashes:", crashes)
    print("Timeouts:", timeouts)

    print(
        "Safe landing rate:",
        round(safe / EPISODES * 100, 2),
        "%"
    )

    print(
        "Precision accuracy:",
        round(centered / EPISODES * 100, 2),
        "%"
    )

    print(
        "Average reward:",
        round(sum(rewards) / EPISODES, 2)
    )


for name, path in MODELS.items():
    evaluate(name, path)