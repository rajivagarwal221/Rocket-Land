import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from randomized_lander import RandomStartLunarLander


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parent.parent  # project root

MODELS = {
    "ORIGINAL_MASTER":
        ROOT / "models" / "ppo_stage1_precision_v2.zip",

    "SETTLE_CANDIDATE":
        ROOT / "models" / "master_settle_candidate.zip"
}

LOG_DIR = ROOT / "logs"

SUMMARY_FILE = LOG_DIR / "master_candidate_comparison.csv"

DETAIL_FILE = LOG_DIR / "master_candidate_episodes.csv"

EPISODES = 30

SEEDS = range(100, 130)

TARGET_LIMIT = 0.15


# =====================================
# CREATE EVALUATION ENVIRONMENT
# =====================================

from target_landing import TargetLandingWrapper


def create_env():

    env = RandomStartLunarLander(
        render_mode=None,
        enable_wind=False,
        hard_start_probability=0.0
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=1000
    )

    env = TargetLandingWrapper(
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=50.0,
        off_target_penalty=20.0,
        timeout_penalty=30.0,
        settle_frames=15
    )

    return env

# =====================================
# EVALUATE ONE MODEL
# =====================================

def evaluate_model(name, model_path, env):

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = PPO.load(
        str(model_path),
        device="cpu"
    )

    results = []

    counts = {
        "PRECISE": 0,
        "SAFE_OUTSIDE": 0,
        "LEG_FAILURE": 0,
        "CRASH": 0,
        "TIMEOUT": 0
    }

    rewards = []

    print(f"\nEVALUATING {name}")
    print("-" * 50)

    for episode, seed in enumerate(SEEDS, start=1):

        obs, info = env.reset(seed=seed)

        total_reward = 0.0
        final_reward = 0.0

        terminated = False
        truncated = False

        steps = 0

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            total_reward += float(reward)

            final_reward = float(reward)

            steps += 1

        # Final landing state

        x = float(obs[0])

        vx = float(obs[2])

        vy = float(obs[3])

        angle = float(obs[4])

        both_legs = bool(
            obs[6] > 0.5 and obs[7] > 0.5
        )

        inside_target = bool(
            -TARGET_LIMIT < x < TARGET_LIMIT
        )

        safe = bool(
            terminated
            and info.get("base_reward", final_reward) > 0
            )

        precise = bool(
            info.get("precise_landing", False)
   )

        if precise:

            result = "PRECISE"

        elif safe and inside_target:

            result = "LEG_FAILURE"

        elif safe:

            result = "SAFE_OUTSIDE"

        elif truncated and not terminated:

            result = "TIMEOUT"

        else:

            result = "CRASH"

        counts[result] += 1

        rewards.append(total_reward)

        results.append({

            "model": name,

            "episode": episode,

            "seed": seed,

            "final_x": x,

            "absolute_x_error": abs(x),

            "final_vx": vx,

            "final_vy": vy,

            "final_angle": angle,

            "both_legs": both_legs,

            "inside_target": inside_target,

            "steps": steps,

            "reward": total_reward,

            "result": result

        })

        print(
            f"{episode:02d}/30 | "
            f"{result:<15} | "
            f"X={x:+.3f} | "
            f"Steps={steps}"
        )

    safe_total = (
        counts["PRECISE"]
        + counts["SAFE_OUTSIDE"]
        + counts["LEG_FAILURE"]
    )

    summary = {

        "model": name,

        "precise": counts["PRECISE"],

        "safe_outside": counts["SAFE_OUTSIDE"],

        "leg_failures": counts["LEG_FAILURE"],

        "safe_total": safe_total,

        "crashes": counts["CRASH"],

        "timeouts": counts["TIMEOUT"],

        "precision_percent": (
            counts["PRECISE"] / EPISODES * 100
        ),

        "safe_percent": (
            safe_total / EPISODES * 100
        ),

        "average_reward": (
            sum(rewards) / len(rewards)
        ),

        "mean_absolute_x": (
            sum(row["absolute_x_error"] for row in results)
            / len(results)
        )

    }

    return summary, results


# =====================================
# MAIN
# =====================================

def main():

    LOG_DIR.mkdir(exist_ok=True)

    env = create_env()

    all_summaries = []

    all_details = []

    try:

        for name, path in MODELS.items():

            summary, details = evaluate_model(
                name,
                path,
                env
            )

            all_summaries.append(summary)

            all_details.extend(details)

    finally:

        env.close()

    # Save summary CSV

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_summaries[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_summaries)

    # Save episode details CSV

    with DETAIL_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_details[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_details)

    # Print comparison

    print("\n========== FINAL COMPARISON ==========")

    for result in all_summaries:

        print(f"\n{result['model']}")

        print(
            f"Precision: "
            f"{result['precise']}/30 "
            f"({result['precision_percent']:.2f}%)"
        )

        print(
            f"Safe landings: "
            f"{result['safe_total']}/30 "
            f"({result['safe_percent']:.2f}%)"
        )

        print("Crashes:", result["crashes"])

        print("Timeouts:", result["timeouts"])

        print(
            "Average reward:",
            round(result["average_reward"], 2)
        )

        print(
            "Mean absolute landing error:",
            round(result["mean_absolute_x"], 4)
        )

    print("\nCSV FILES CREATED")

    print(SUMMARY_FILE)

    print(DETAIL_FILE)


if __name__ == "__main__":
    main()