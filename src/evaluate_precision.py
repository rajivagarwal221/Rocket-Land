import csv
from logging import info
from logging import info
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO

from randomized_lander import RandomStartLunarLander
from target_landing import TargetLandingWrapper


# ==========================================
# CONFIGURATION
# ==========================================

# ==========================================
# CONFIGURATION
# ==========================================

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent  # project root

MODELS = {
    "V9": {
        "path": ROOT / "models" / "master_continuous_v9_candidate.zip",
        "continuous": True
    },

    "V10": {
        "path": ROOT / "models" / "master_continuous_v10_candidate.zip",
        "continuous": True
    }
}

LOG_DIR = ROOT / "logs"

RESULT_FILE = LOG_DIR / "master_v10_comparison_100.csv"

EPISODES = 100

SEEDS = range(1000, 1100)

TARGET_LIMIT = 0.15

MAX_STEPS = 1000

# ==========================================
# CREATE ENVIRONMENT
# ==========================================

def create_environment(continuous):

    env = RandomStartLunarLander(
        render_mode="human",
        enable_wind=False,
        hard_start_probability=0.0,
        continuous=continuous
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


# ==========================================
# EVALUATE ONE MODEL
# ==========================================

def evaluate_model(name, config):

    model_path = config["path"]

    continuous = config["continuous"]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    env = create_environment(continuous)

    model = PPO.load(
        str(model_path),
        device="cpu"
    )

    # Validate action-space compatibility.
    if continuous:

        if not isinstance(
            model.action_space,
            gym.spaces.Box
        ):
            raise ValueError(
                "Continuous model has incompatible action space."
            )

    else:

        if not isinstance(
            model.action_space,
            gym.spaces.Discrete
        ):
            raise ValueError(
                "Discrete model has incompatible action space."
            )

    results = []

    counts = {
        "PRECISE": 0,
        "SAFE_OUTSIDE": 0,
        "REST_NO_BOTH_LEGS": 0,
        "CRASH": 0,
        "TIMEOUT": 0
    }

    rewards = []

    print("\n====================================")
    print("EVALUATING:", name)
    print("====================================")

    try:

        for episode, seed in enumerate(SEEDS, start=1):

            observation, reset_info = env.reset(
                seed=seed
            )

            initial_state = reset_info.get(
                "initial_state", {}
            )

            total_reward = 0.0

            steps = 0

            terminated = False
            truncated = False

            info = {}

            contact_trace = []

            both_contact_metrics = []

            stable_streak = 0
            
            max_stable_streak = 0

            both_contact_frames = []

            stable_streak = 0
            max_stable_streak = 0

            while not (terminated or truncated):

                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                # Discrete and continuous actions
                # must be passed in their correct formats.

                if continuous:

                    action = np.asarray(
                        action,
                        dtype=np.float32
                    ).reshape(
                        env.action_space.shape
                    )

                else:

                    action = int(
                        np.asarray(action).item()
                    )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                total_reward += float(reward)

                steps += 1
                # V6: measure stability during simultaneous leg contact

                left = observation[6] > 0.5
                right = observation[7] > 0.5

                if left and right:

                    vx_now = abs(float(observation[2]))
                    vy_now = abs(float(observation[3]))
                    angle_now = abs(float(observation[4]))
                    omega_now = abs(float(observation[5]))

                    vx_ok = vx_now < 0.025
                    vy_ok = vy_now < 0.025
                    angle_ok = angle_now < 0.20
                    omega_ok = omega_now < 0.08

                    stable_now = (
                        vx_ok and vy_ok and angle_ok and omega_ok
                    )

                    if stable_now:
                        stable_streak += 1
                    else:
                        stable_streak = 0

                    max_stable_streak = max(
                        max_stable_streak,
                        stable_streak
                    )

                    both_contact_metrics.append({
                        "step": steps,
                        "vx": vx_now,
                        "vy": vy_now,
                        "angle": angle_now,
                        "omega": omega_now,
                        "vx_ok": vx_ok,
                        "vy_ok": vy_ok,
                        "angle_ok": angle_ok,
                        "omega_ok": omega_ok,
                        "stable": stable_now
                    })

                else:
                    stable_streak = 0

                # Record stability while both legs touch the ground.

                left = observation[6] > 0.5
                right = observation[7] > 0.5

                if left and right:

                    vx_now = abs(float(observation[2]))
                    vy_now = abs(float(observation[3]))
                    angle_now = abs(float(observation[4]))
                    omega_now = abs(float(observation[5]))

                    stable_now = (
                        vx_now < 0.025
                        and vy_now < 0.025
                        and angle_now < 0.20
                        and omega_now < 0.08
                    )

                    if stable_now:
                        stable_streak += 1
                    else:
                        stable_streak = 0

                    max_stable_streak = max(
                        max_stable_streak,
                        stable_streak
                    )

                    action_values = np.asarray(action).reshape(-1)

                    both_contact_frames.append({

                        "step": steps,

                        "vx": vx_now,
                        "vy": vy_now,

                        "angle": angle_now,
                        "omega": omega_now,

                        "stable": stable_now,

                        "action_0": float(action_values[0]),
                        "action_1": float(action_values[1])

                    })

                else:

                    stable_streak = 0

                # Record only changes in leg-contact state.
                contact_state = (
                    int(observation[6] > 0.5),
                    int(observation[7] > 0.5)
                )

                if (
                    not contact_trace
                    or contact_state != contact_trace[-1]["contact"]
                ):
                    contact_trace.append({
                        "step": steps,
                        "contact": contact_state,
                        "x": round(float(observation[0]), 4),
                        "angle": round(float(observation[4]), 4),
                        "y": round(float(observation[1]), 4)
                   })

            # ----------------------------------
            # FINAL STATE
            # ----------------------------------

            final_x = float(observation[0])

            final_vx = float(observation[2])

            final_vy = float(observation[3])

            final_angle = float(observation[4])

            final_omega = float(observation[5])

            both_legs = bool(
                observation[6] > 0.5
                and observation[7] > 0.5
            )

            inside_target = bool(
                -TARGET_LIMIT < final_x < TARGET_LIMIT
            )

            base_reward = float(
                info.get("base_reward", 0.0)
            )

            safe = bool(
                terminated and base_reward > 0
            )

            precise = bool(
                safe
                and info.get("precise_landing", False)
            )

            custom_landing = bool(
                info.get("custom_stable_landing", False)
            )

            # ----------------------------------
            # CLASSIFICATION
            # ----------------------------------

            landing_status = info.get("landing_status", "")

            if truncated:
                result = "TIMEOUT"

            elif info.get("failed_rest", False) or landing_status == "FAILED_NO_BOTH_LEGS":
                result = "REST_NO_BOTH_LEGS"

            elif terminated and float(info.get("base_reward", 0.0)) > 0:
               if info.get("precise_landing", False):
                    result = "PRECISE"
               else:
                    result = "SAFE_OUTSIDE"

            else:
                result = "CRASH"

            counts[result] += 1
            if result == "REST_NO_BOTH_LEGS":

                print("\n========== STABILITY ANALYSIS ==========")

                print(
                    "Both-leg contact frames:",
                    len(both_contact_metrics)
                )

                print(
                    "Maximum consecutive stable frames:",
                    max_stable_streak
                )

                if both_contact_metrics:

                    total = len(both_contact_metrics)

                    for key, label in [
                        ("vx_ok", "Horizontal velocity"),
                        ("vy_ok", "Vertical velocity"),
                        ("angle_ok", "Landing angle"),
                        ("omega_ok", "Angular velocity")
                    ]:

                        passed = sum(
                            row[key]
                            for row in both_contact_metrics
                        )

                        print(
                            f"{label}: {passed}/{total} "
                            f"frames passed"
                        )

                    all_passed = sum(
                        row["stable"]
                        for row in both_contact_metrics
                    )

                    print(
                        "All four conditions passed:",
                        f"{all_passed}/{total}"
                    )

                print("========================================")

                print("\nCONTACT STATE TRANSITIONS:")

                for event in contact_trace:
                    print(
                        f"Step {event['step']}: "
                        f"Left/Right={event['contact']} "
                        f"X={event['x']} "
                        f"Angle={event['angle']} "
                        f"Y={event['y']}"
                    )

                print("\nBOTH-LEG STABILITY ANALYSIS")

                print(
                    "Total both-leg contact frames:",
                    len(both_contact_frames)
                )

                print(
                    "Maximum consecutive stable frames:",
                    max_stable_streak
                )

                if both_contact_frames:

                    print(
                        "Maximum horizontal velocity:",
                        max(r["vx"] for r in both_contact_frames)
                    )

                    print(
                        "Maximum vertical velocity:",
                        max(r["vy"] for r in both_contact_frames)
                    )

                    print(
                        "Maximum angle:",
                        max(r["angle"] for r in both_contact_frames)
                    )

                    print(
                        "Maximum angular velocity:",
                        max(r["omega"] for r in both_contact_frames)
                    )

                    print("\nFIRST 5 BOTH-LEG CONTACT FRAMES:")

                    for row in both_contact_frames[:5]:
                        print(row)

                for event in contact_trace:
                    print(
                    f"Step {event['step']}: "
                    f"Left/Right={event['contact']} "
                    f"X={event['x']} "
                    f"Angle={event['angle']} "
                    f"Y={event['y']}"
                    )

                print("\n--- TOUCHDOWN DIAGNOSTIC ---")
 
                print("Seed:", seed)
                print("Left leg:", observation[6])
                print("Right leg:", observation[7])
                print("Landing status:", info.get("landing_status"))
                print("Raw reward:", info.get("raw_environment_reward"))
                print("Final X:", observation[0])
                print("Angle:", observation[4])
                print("Stable contact frames:", info.get("stable_contact_steps"))

                print("----------------------------")
       
            if result == "CRASH":

               print("\n⚠️ CRASH DIAGNOSTIC")

               print("Seed:", seed)

               print("Landing status:",
               info.get("landing_status"))

               print("Failed rest:",
               info.get("failed_rest"))

               print("Raw environment reward:",
               info.get("raw_environment_reward"))

               print("Effective base reward:",
               info.get("base_reward"))

               print("Left leg:", observation[6])

               print("Right leg:", observation[7])

               print("Final X:", observation[0])

               print("Horizontal velocity:", observation[2])

               print("Vertical velocity:", observation[3])

               print("Angle:", observation[4])

               print("Angular velocity:", observation[5])

               print("Terminated:", terminated)

               print("Truncated:", truncated)

               print("========================\n")

            rewards.append(total_reward)

            results.append({

                "model": name,
                "episode": episode,
                "seed": seed,

                "initial_x": initial_state.get("x"),
                "initial_vx": initial_state.get("vx"),

                "final_x": final_x,
                "absolute_x_error": abs(final_x),

                "final_vx": final_vx,
                "final_vy": final_vy,
                "final_angle": final_angle,
                "final_omega": final_omega,

                "both_legs": both_legs,
                "left_leg_contact": float(observation[6]),
                "right_leg_contact": float(observation[7]),
  
                "inside_target": inside_target,
                "custom_stable_landing": custom_landing,

                "stable_contact_steps": info.get(
                "stable_contact_steps", 0
                ),

                "landing_status": info.get("landing_status"),
                "failed_rest": info.get("failed_rest"),
                "raw_environment_reward": info.get("raw_environment_reward"),
                "base_reward": info.get("base_reward"),

                "steps": steps,
                "reward": total_reward,
                "result": result

            })

            print(
                f"{episode:02d}/{EPISODES} | "
                f"{result:<15} | "
                f"X={final_x:+.3f} | "
                f"VX={final_vx:+.3f} | "
                f"Steps={steps}",
                flush=True
            )

    finally:

        env.close()

    # ==================================
    # SUMMARY
    # ==================================

    total_safe = (
        counts["PRECISE"]
        + counts["SAFE_OUTSIDE"]
    )

    summary = {
        "model": name,

        "precise": counts["PRECISE"],

        "safe_outside": counts["SAFE_OUTSIDE"],

        "rest_no_both_legs": counts["REST_NO_BOTH_LEGS"],

        "total_safe": total_safe,

        "crashes": counts["CRASH"],

        "timeouts": counts["TIMEOUT"],

        "precision_percent": (
            counts["PRECISE"] / EPISODES * 100
        ),

        "safe_percent": (
            total_safe / EPISODES * 100
        ),

        "average_reward": (
            sum(rewards) / len(rewards)
        ),

        "mean_absolute_x": (
            sum(
                row["absolute_x_error"]
                for row in results
            ) / len(results)
        )
    }

    return summary, results


# ==========================================
# MAIN
# ==========================================

def main():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = []

    all_results = []

    for name, config in MODELS.items():

        summary, results = evaluate_model(
            name,
            config
        )

        all_summaries.append(summary)

        all_results.extend(results)

    # ----------------------------------
    # SAVE ONE CSV
    # ----------------------------------

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()

        writer.writerows(all_results)

    # ----------------------------------
    # DISPLAY FINAL COMPARISON
    # ----------------------------------

    print("\n")
    print("=" * 65)
    print("             MASTER PPO COMPARISON")
    print("=" * 65)

    for s in all_summaries:

        print("\nMODEL:", s["model"])

        print(
            f"Precise landings: {s['precise']}/{EPISODES} "
            f"({s['precision_percent']:.2f}%)"
        )

        print(
            f"Total safe landings: {s['total_safe']}/{EPISODES} "
            f"({s['safe_percent']:.2f}%)"
        )

        print("Safe outside target:", s["safe_outside"])

        print("Rest without both legs:", s["rest_no_both_legs"])

        print("Actual crashes:", s["crashes"])

        print("Timeouts:", s["timeouts"])

        print("Average reward:", round(s["average_reward"], 2))

        print("Mean horizontal error:", round(s["mean_absolute_x"], 4))

    print("\nRESULT FILE:")
    print(RESULT_FILE)

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()