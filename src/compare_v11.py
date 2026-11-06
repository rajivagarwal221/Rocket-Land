"""100-episode holdout: V11 vs V9. Matches compare_master.py conventions."""
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from randomized_lander import RandomStartLunarLander

ROOT = Path(__file__).resolve().parent

MODELS = {
    "V9": str(ROOT / "models" / "master_continuous_v9_candidate"),
    "V11": str(ROOT / "models" / "master_continuous_v11_candidate"),
    "V12": str(ROOT / "models" / "master_continuous_v12_candidate"),
}

EPISODES = 100
SEED_BASE = 1000


def eval_one(name, path):
    print(f"\n========== {name} ==========", flush=True)
    model = PPO.load(path, device="cpu")

    env = RandomStartLunarLander(enable_wind=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    safe = 0
    precise = 0
    crashes = 0
    timeouts = 0
    leg_fails = 0
    rewards = []

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=SEED_BASE + ep)
        total = 0.0
        final = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            final = r
            if terminated or truncated:
                break

        x = float(obs[0])
        both_legs = obs[6] > 0.5 and obs[7] > 0.5
        successful = terminated and final > 0
        on_target = successful and both_legs and -0.15 < x < 0.15

        if successful:
            safe += 1
            if not both_legs:
                leg_fails += 1
        if on_target:
            precise += 1
        if truncated:
            timeouts += 1
        elif terminated and not successful:
            crashes += 1

        rewards.append(total)

    env.close()
    n = EPISODES
    print(f"Precise:     {precise}/{n}  ({precise/n*100:.1f}%)")
    print(f"Safe:        {safe}/{n}  ({safe/n*100:.1f}%)")
    print(f"Leg-fails:   {leg_fails}/{n}")
    print(f"Crashes:     {crashes}/{n}")
    print(f"Timeouts:    {timeouts}/{n}")
    print(f"Mean reward: {sum(rewards)/n:.2f}")
    return precise, safe, crashes, timeouts


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, path in MODELS.items():
        if only and only != name:
            continue
        eval_one(name, path)
