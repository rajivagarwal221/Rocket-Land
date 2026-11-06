import gymnasium as gym
from stable_baselines3 import PPO
from randomized_lander import RandomStartLunarLander

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

# Load selected master AI
model = PPO.load(
    str(ROOT / "models" / "ppo_stage1_precision_v2")
)

# Create environment
env = RandomStartLunarLander(
    enable_wind=False
)

env = gym.wrappers.TimeLimit(
    env,
    max_episode_steps=1000
)

EPISODES = 100

results = []

safe = 0
centered = 0
crashes = 0
timeouts = 0

for episode in range(EPISODES):

    observation, info = env.reset(
        seed=3000 + episode
    )

    initial = info.get("initial_state", {})

    total_reward = 0
    final_reward = 0
    steps = 0

    while True:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        final_reward = reward
        steps += 1

        if terminated or truncated:
            break

    # Final state
    x = float(observation[0])
    y = float(observation[1])

    vx = float(observation[2])
    vy = float(observation[3])

    angle = float(observation[4])
    omega = float(observation[5])

    both_legs = (
        observation[6] > 0.5
        and observation[7] > 0.5
    )

    successful = (
        terminated and final_reward > 0
    )

    inside_target = (
        -0.15 < x < 0.15
    )

    precision = (
        successful
        and inside_target
        and both_legs
    )

    if precision:
        result = "CENTERED"
        centered += 1
        safe += 1

    elif successful:
        result = "SAFE_OUTSIDE"
        safe += 1

    elif truncated:
        result = "TIMEOUT"
        timeouts += 1

    else:
        result = "CRASH"
        crashes += 1

    results.append({

        "episode": episode + 1,

        "initial_x": initial.get("x", 0),
        "initial_y": initial.get("y", 0),
        "initial_vx": initial.get("vx", 0),
        "initial_vy": initial.get("vy", 0),
        "initial_angle": initial.get("angle", 0),

        "final_x": x,
        "final_y": y,

        "final_vx": vx,
        "final_vy": vy,

        "final_angle": angle,
        "final_omega": omega,

        "both_legs": both_legs,
        "steps": steps,

        "reward": total_reward,
        "result": result
    })

    if (episode + 1) % 10 == 0:
        print(f"Completed {episode + 1}/{EPISODES}")


env.close()

# Save diagnostics
os.makedirs(str(ROOT / "logs"), exist_ok=True)

with open(
    str(ROOT / "logs" / "master_diagnostics.csv"),
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


print("\n========== MASTER DIAGNOSTICS ==========")

print("Episodes:", EPISODES)
print("Safe landings:", safe)
print("Centered landings:", centered)
print("Crashes:", crashes)
print("Timeouts:", timeouts)

print(
    "Precision rate:",
    round(centered / EPISODES * 100, 2),
    "%"
)

print(f"\nSaved: {ROOT / 'logs' / 'master_diagnostics.csv'}")