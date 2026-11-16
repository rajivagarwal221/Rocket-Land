"""Audit a 3D lander policy (or the PID baseline) over N seeded episodes.

    python src/evaluate_3d.py --model models/master_v28_best.zip --episodes 1000
    python src/evaluate_3d.py --baseline --episodes 1000

Scores at the SHIP bar by default: full difficulty (wind, lag, randomized
mass / thrust, sensor noise, delay) and a 0.5 m target radius. Episode i uses
seed (seed_offset + i), so two runs of the same command give the same table
and two policies are compared on identical conditions.
"""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from sim3d.baseline import PIDController
from sim3d.rocket_env import OBS_DIM, OUTCOMES
from sim3d.wrappers import base_env, make_vec

FIELDS = (
    "episode", "seed", "outcome", "return", "steps", "landing_error",
    "impact_speed", "impact_tilt", "wind_speed", "mass_scale", "thrust_scale",
    "start_offset", "start_alt",
)


def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def report(title: str, rows: list):
    n = len(rows)
    print(f"\n========== {title} (n={n}) ==========")
    if n == 0:
        print("  no episodes in this slice")
        return
    counts = Counter(r["outcome"] for r in rows)
    for name in OUTCOMES:
        print(f"  {name:<15}{counts[name]:>5}  {counts[name] / n:7.2%}")
    precise = counts["PRECISE"]
    safe = precise + counts["SAFE_OUTSIDE"]
    lo, hi = wilson(precise, n)
    print(f"  safe landings  {safe:>5}  {safe / n:7.2%}")
    print(f"  precise 95% CI {lo:7.2%} - {hi:.2%}")
    landed = [r for r in rows if r["outcome"] in ("PRECISE", "SAFE_OUTSIDE")]
    if landed:
        errors = np.array([r["landing_error"] for r in landed])
        impacts = np.array([r["impact_speed"] for r in landed])
        print(
            f"  landing error   median {np.median(errors):.3f} m, "
            f"95th pct {np.percentile(errors, 95):.3f} m"
        )
        print(
            f"  touchdown speed median {np.median(impacts):.2f} m/s, "
            f"max {impacts.max():.2f} m/s"
        )
    print(f"  mean return    {np.mean([r['return'] for r in rows]):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, help="SAC checkpoint (.zip)")
    ap.add_argument("--baseline", action="store_true", help="score the PID baseline instead")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed-offset", type=int, default=20000)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--target-radius", type=float, default=0.5)
    ap.add_argument("--delay-steps", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, help="write one CSV row per episode")
    args = ap.parse_args()

    if args.baseline == (args.model is not None):
        ap.error("pass exactly one of --model or --baseline")

    model, n_stack = None, 1
    if args.model is not None:
        if not args.model.exists():
            ap.error(f"checkpoint not found: {args.model}")
        from stable_baselines3 import SAC

        model = SAC.load(str(args.model), device=args.device)
        obs_dim = model.observation_space.shape[0]
        if obs_dim % OBS_DIM != 0:
            ap.error(
                f"model expects {obs_dim} observations, not a multiple of {OBS_DIM}: "
                "this is not a 3D lander checkpoint (use evaluate_all.py for V27 and older)"
            )
        n_stack = obs_dim // OBS_DIM

    vec = make_vec(
        1, args.seed_offset, args.difficulty, args.target_radius, n_stack,
        subproc=False, delay_steps=args.delay_steps,
    )
    env = base_env(vec)
    pid = PIDController()
    who = "PID baseline (true state)" if args.baseline else args.model.name

    print("\n========== ROCKET-LAND 3D EVALUATION ==========")
    print(f"[eval] controller    : {who}")
    print(f"[eval] episodes      : {args.episodes} (seeds from {args.seed_offset})")
    print(f"[eval] difficulty    : {args.difficulty}")
    print(f"[eval] target radius : {args.target_radius} m")
    print(f"[eval] frame stack   : {n_stack}, sensor delay {args.delay_steps} steps")

    rows = []
    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        vec.seed(seed)
        obs = vec.reset()
        pid.reset()
        total, steps = 0.0, 0
        while True:
            if model is None:
                action = pid(env)[None, :]
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec.step(action)
            total += float(reward[0])
            steps += 1
            if done[0]:
                break
        info = infos[0]
        rows.append(
            {
                "episode": ep, "seed": seed, "outcome": info["outcome"],
                "return": total, "steps": steps,
                **{k: info[k] for k in FIELDS[5:]},
            }
        )
        if (ep + 1) % max(1, args.episodes // 20) == 0:
            precise = sum(r["outcome"] == "PRECISE" for r in rows)
            print(
                f"[eval] {ep + 1:>5}/{args.episodes} | last={info['outcome']:<13} "
                f"| precise so far {precise / (ep + 1):6.1%}",
                flush=True,
            )
    vec.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[eval] CSV written: {args.out.resolve()}")

    report("OVERALL", rows)
    report("WIND: 3 m/s or more", [r for r in rows if r["wind_speed"] >= 3.0])
    report("WIND: under 3 m/s", [r for r in rows if r["wind_speed"] < 3.0])
    report("MASS: heavier than nominal", [r for r in rows if r["mass_scale"] >= 1.0])
    report("MASS: lighter than nominal", [r for r in rows if r["mass_scale"] < 1.0])
    report("THRUST: below nominal", [r for r in rows if r["thrust_scale"] < 1.0])
    report("THRUST: at or above nominal", [r for r in rows if r["thrust_scale"] >= 1.0])
    report("START: over 2.5 m off the pad", [r for r in rows if r["start_offset"] > 2.5])


if __name__ == "__main__":
    main()
