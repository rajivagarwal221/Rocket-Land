"""Unified full-stack evaluator for the Lunar-Lander master policy.

Evaluates a trained PPO model against the top of the curriculum ladder
(ThrustScaledLunarLander -> VariablePadLunarLander -> WindyPlanetLunarLander
-> GravityObsLunarLander -> PlanetLunarLander -> RandomStartLunarLander),
wrapped with TargetLandingWrapper for shaped reward.

Because GravityObsLunarLander adds gravity as a 9th obs dim, this script
requires a model trained from v14 onward (e.g. master_v17_thrust.zip).
Passing an older 8-dim model raises an obs-space mismatch; use
compare_master_candidate.py for those.

Usage
-----
    .venv\\Scripts\\python.exe src\\evaluate_all.py
    .venv\\Scripts\\python.exe src\\evaluate_all.py --model models\\master_v17_thrust.zip --episodes 200

Writes a per-episode CSV with the sampled physics for each episode and
prints an aggregate summary plus per-feature breakdowns.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper


ROOT = Path(__file__).resolve().parent.parent  # project root


# ------------------------- env builder -------------------------

def build_env(max_episode_steps: int = 1000, render: bool = False) -> gym.Env:
    # gravity_range activates per-reset gravity randomisation across planets
    # (Pluto -0.62 ... Neptune -11.15). Range clipped by env sanity bounds.
    env = ThrustScaledLunarLander(
        render_mode="human" if render else None,
        gravity_range=(-11.95, -0.5),
    )
    env = TargetLandingWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


# ------------------------- outcome classifier -------------------------

TARGET_LEFT, TARGET_RIGHT = -0.15, 0.15
SETTLE_FRAMES = 15


def classify(obs, terminated, truncated, game_over, stable_contact_steps):
    """Decide the outcome from authoritative signals.

    - game_over comes from the base Box2D LunarLander (True iff the hull hit
      the ground). This is the ground truth for CRASH.
    - stable_contact_steps is maintained by TargetLandingWrapper.
    """
    both_legs = bool(obs[6] > 0.5 and obs[7] > 0.5)
    inside = bool(TARGET_LEFT < float(obs[0]) < TARGET_RIGHT)
    stable = bool(stable_contact_steps >= SETTLE_FRAMES)
    safe_landing = bool(terminated and not game_over)

    if truncated and not terminated:
        return "TIMEOUT", both_legs, inside, stable, safe_landing
    if game_over:
        return "CRASH", both_legs, inside, stable, safe_landing
    if not both_legs:
        return "LEG_FAILURE", both_legs, inside, stable, safe_landing
    if not inside:
        return "SAFE_OUTSIDE", both_legs, inside, stable, safe_landing
    if not stable:
        return "SAFE_OUTSIDE", both_legs, inside, stable, safe_landing
    return "PRECISE", both_legs, inside, stable, safe_landing


def read_stable_contact(env: gym.Env) -> int:
    """Walk the wrapper chain to find TargetLandingWrapper's counter."""
    w = env
    while w is not getattr(w, "unwrapped", None):
        if hasattr(w, "stable_contact_steps"):
            return int(w.stable_contact_steps)
        w = getattr(w, "env", None)
        if w is None:
            break
    return 0


# ------------------------- main eval loop -------------------------

def run(model_path: Path, episodes: int, seed_offset: int, render: bool = False) -> list[dict]:
    print(f"[eval] loading model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    env = build_env(render=render)

    results = []
    for ep in range(episodes):
        seed = seed_offset + ep
        obs, info = env.reset(seed=seed)

        # Snapshot the per-episode randomisation BEFORE stepping
        base = env.unwrapped
        gravity = float(getattr(base, "current_gravity", base.gravity))
        wind_power = float(getattr(base, "wind_power", 0.0)) if getattr(base, "enable_wind", False) else 0.0
        turbulence = float(getattr(base, "turbulence_power", 0.0)) if getattr(base, "enable_wind", False) else 0.0
        target_width = float(getattr(base, "current_target_width", 0.15))
        thrust_scale = float(getattr(base, "current_thrust_scale", 1.0))

        total_reward = 0.0
        final_reward = 0.0
        steps = 0
        terminated = truncated = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            final_reward = float(reward)
            steps += 1
            if terminated or truncated:
                break

        stable_contact = read_stable_contact(env)
        game_over = bool(getattr(env.unwrapped, "game_over", False))
        outcome, both_legs, inside, stable, safe = classify(
            obs, terminated, truncated, game_over, stable_contact
        )

        results.append({
            "episode": ep + 1,
            "seed": seed,
            "gravity": round(gravity, 3),
            "wind_power": round(wind_power, 3),
            "turbulence": round(turbulence, 3),
            "target_width": round(target_width, 3),
            "thrust_scale": round(thrust_scale, 3),
            "final_x": round(float(obs[0]), 4),
            "final_vy": round(float(obs[3]), 4),
            "final_angle": round(float(obs[4]), 4),
            "steps": steps,
            "reward": round(total_reward, 2),
            "both_legs": both_legs,
            "inside_target": inside,
            "stable_settle": stable,
            "outcome": outcome,
        })

        if (ep + 1) % 25 == 0:
            print(f"[eval] {ep + 1}/{episodes}  "
                  f"last outcome: {outcome}  reward: {total_reward:+.1f}")

    env.close()
    return results


# ------------------------- reporting -------------------------

def pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}%" if den else "  n/a"


def summarise(label: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    outcomes = ["PRECISE", "SAFE_OUTSIDE", "LEG_FAILURE", "CRASH", "TIMEOUT"]
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in outcomes}
    both = sum(1 for r in rows if r["both_legs"])
    inside = sum(1 for r in rows if r["inside_target"])
    stable = sum(1 for r in rows if r["stable_settle"])
    rewards = [r["reward"] for r in rows]
    avg_r = sum(rewards) / n

    print(f"\n=== {label}  (n={n}) ===")
    for o in outcomes:
        print(f"  {o:14s} {counts[o]:4d}  {pct(counts[o], n)}")
    print(f"  {'both legs':14s} {both:4d}  {pct(both, n)}")
    print(f"  {'inside target':14s} {inside:4d}  {pct(inside, n)}")
    print(f"  {'stable settle':14s} {stable:4d}  {pct(stable, n)}")
    print(f"  {'avg reward':14s} {avg_r:7.2f}   "
          f"min {min(rewards):7.2f}   max {max(rewards):7.2f}")


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(ROOT / "models" / "master_v17_thrust.zip"),
                    help="Path to .zip model (default: master_v17_thrust.zip)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed-offset", type=int, default=5000,
                    help="Base seed; episode i uses seed-offset+i")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: logs/evaluate_all_<stem>_<N>.csv)")
    ap.add_argument("--render", action="store_true",
                    help="Show the pygame window so you can watch each landing")
    args = ap.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out) if args.out else (
        ROOT / "logs" / f"evaluate_all_{model_path.stem}_{args.episodes}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = run(model_path, args.episodes, args.seed_offset, render=args.render)

    # ---- write CSV ----
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[eval] wrote {len(rows)} rows -> {out_path}")

    # ---- overall + breakdowns ----
    summarise("OVERALL", rows)

    # Gravity bands
    low_g = [r for r in rows if r["gravity"] > -6.0]   # closer to zero = weaker gravity
    high_g = [r for r in rows if r["gravity"] <= -6.0]  # stronger gravity
    summarise("GRAVITY  weaker-than-Venus (>-6 m/s^2)", low_g)
    summarise("GRAVITY  Venus-or-stronger (<=-6 m/s^2)", high_g)

    # Wind
    wind_on = [r for r in rows if r["wind_power"] > 0.0]
    wind_off = [r for r in rows if r["wind_power"] == 0.0]
    summarise("WIND  present (power > 0)", wind_on)
    summarise("WIND  calm (power == 0)", wind_off)

    # Thrust
    low_t = [r for r in rows if r["thrust_scale"] < 1.0]
    high_t = [r for r in rows if r["thrust_scale"] >= 1.0]
    summarise("THRUST  under-powered (scale < 1.0)", low_t)
    summarise("THRUST  over-powered (scale >= 1.0)", high_t)

    # Target width
    narrow = [r for r in rows if r["target_width"] < 0.175]
    wide = [r for r in rows if r["target_width"] >= 0.175]
    summarise("TARGET  narrow (width < 0.175)", narrow)
    summarise("TARGET  wide (width >= 0.175)", wide)


if __name__ == "__main__":
    main()
