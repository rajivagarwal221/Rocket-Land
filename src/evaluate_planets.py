"""Evaluate a trained policy under a chosen gravity.

Usage:
    python evaluate_planets.py --model v13 --planet moon --episodes 100
    python evaluate_planets.py --model v9  --gravity -3.71 --episodes 100
    python evaluate_planets.py --model v13 --all

By default --model is v13 (planet-generalist). Pass --model v9 to compare
against the Earth-only champion.
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from planet_lander import PlanetLunarLander, PLANET_GRAVITY, resolve_gravity
from gravity_obs_lander import GravityObsLunarLander


ROOT = Path(__file__).resolve().parent

MODEL_PATHS = {
    "v9": str(ROOT / "models" / "master_continuous_v9_candidate"),
    "v11": str(ROOT / "models" / "master_continuous_v11_candidate"),
    "v12": str(ROOT / "models" / "master_continuous_v12_candidate"),
    "v13": str(ROOT / "models" / "master_planet_v13_candidate"),
    "v14": str(ROOT / "models" / "master_v14_gravity_aware"),
    "v15": str(ROOT / "models" / "master_v15_wind"),
    "v16": str(ROOT / "models" / "master_v16_pad"),
    "v17": str(ROOT / "models" / "master_v17_thrust"),
}

# Models that use 9-dim obs (gravity-aware)
NINE_DIM = {"v14", "v15", "v16", "v17"}


def run(model, gravity_val, episodes, seed_base=1000, nine_dim=False):
    EnvCls = GravityObsLunarLander if nine_dim else PlanetLunarLander
    env = EnvCls(gravity=gravity_val, enable_wind=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    safe = precise = crashes = timeouts = leg_fails = 0
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total = 0.0
        final = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, _ = env.step(action)
            total += r
            final = r
            if te or tr:
                break

        x = float(obs[0])
        both_legs = obs[6] > 0.5 and obs[7] > 0.5
        successful = te and final > 0
        on_target = successful and both_legs and -0.15 < x < 0.15

        if successful:
            safe += 1
            if not both_legs:
                leg_fails += 1
        if on_target:
            precise += 1
        if tr:
            timeouts += 1
        elif te and not successful:
            crashes += 1
        rewards.append(total)
    env.close()
    return {
        "precise": precise,
        "safe": safe,
        "leg_fails": leg_fails,
        "crashes": crashes,
        "timeouts": timeouts,
        "mean_reward": sum(rewards) / episodes,
    }


def pretty(name, g, episodes, r):
    print(
        f"{name:<12} g={g:>7.2f}  "
        f"precise {r['precise']:>3}/{episodes}  "
        f"safe {r['safe']:>3}/{episodes}  "
        f"leg-fails {r['leg_fails']:>2}  "
        f"crash {r['crashes']:>2}  "
        f"timeout {r['timeouts']:>2}  "
        f"reward {r['mean_reward']:>7.1f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="v13", choices=sorted(MODEL_PATHS))
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--planet", help="e.g. moon, mars, earth, neptune")
    grp.add_argument("--gravity", type=float, help="e.g. -3.71")
    grp.add_argument("--all", action="store_true",
                     help="sweep moon/mars/venus/earth/neptune")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    path = MODEL_PATHS[args.model]
    if not Path(path + ".zip").exists():
        print(f"ERROR: model file not found: {path}.zip", file=sys.stderr)
        sys.exit(1)
    print(f"Loading model: {args.model} -> {path}", flush=True)
    model = PPO.load(path, device="cpu")

    nine_dim = args.model in NINE_DIM

    if args.all:
        planets = ["moon", "mars", "venus", "earth", "neptune"]
        print(f"Sweeping {planets} @ {args.episodes} eps each (9-dim obs={nine_dim})", flush=True)
        for planet in planets:
            g, _ = resolve_gravity(planet)
            r = run(model, g, args.episodes, args.seed_base, nine_dim=nine_dim)
            pretty(planet, g, args.episodes, r)
        return

    if args.planet:
        g, raw = resolve_gravity(args.planet)
        label = args.planet
        if g != raw:
            print(
                f"NOTE: {args.planet} true gravity {raw:.2f} clipped to {g:.2f}",
                flush=True,
            )
    elif args.gravity is not None:
        g, _ = resolve_gravity(args.gravity)
        label = f"g={g:.2f}"
    else:
        g, _ = resolve_gravity("earth")
        label = "earth"

    r = run(model, g, args.episodes, args.seed_base, nine_dim=nine_dim)
    pretty(label, g, args.episodes, r)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
