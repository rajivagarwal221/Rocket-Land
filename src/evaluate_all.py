"""Evaluate Rocket-Land PPO/SAC checkpoints on their matching training profile.

Supported profiles: v17, v19, v19b and v20.

Examples:
    python src/evaluate_all.py --model models/master_v20_safety.zip --profile v20 --episodes 200 --shield
    python src/evaluate_all.py --model models/master_v19_sac.zip --profile v19 --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from thrust_scaled_lander import ThrustScaledLunarLander
from target_landing import TargetLandingWrapper
from safety_shield import SafetyShield


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "master_v20_safety.zip"
DEFAULT_MAX_EPISODE_STEPS = 1000


# ============================================================
# ENVIRONMENT PROFILES
# ============================================================

ENV_PROFILES: dict[str, dict[str, Any]] = {

    "v17": {
        "base": {
            "gravity": -10.0,
            "gravity_range": (-11.5, -1.5),
            "wind_power_range": (0.0, 12.0),
            "turbulence_range": (0.0, 1.2),
            "wind_probability": 0.5,
            "target_width_range": (0.10, 0.25),
            "thrust_scale_range": (0.8, 1.3),
            "continuous": True,
        },
        "shield_default": False,
        "target": {
            "gamma": 0.99,
            "failed_landing_penalty": 100.0,
        },
    },

    "v19": {
        "base": {
            "gravity": -10.0,
            "gravity_range": (-11.5, -1.5),
            "wind_power_range": (0.0, 12.0),
            "turbulence_range": (0.0, 1.2),
            "wind_probability": 0.5,
            "target_width_range": (0.10, 0.25),
            "thrust_scale_range": (0.8, 1.3),
            "weak_gravity_prob": 0.25,
            "weak_gravity_cutoff": -5.0,
            "continuous": True,
        },
        "shield_default": False,
        "target": {
            "gamma": 0.99,
            "step_penalty": 0.01,
            "target_bonus": 50.0,
            "off_target_penalty": 20.0,
            "failed_landing_penalty": 100.0,
            "target_limit": 0.15,
        },
    },

    "v19b": {
        "base": {
            "gravity": -10.0,
            "gravity_range": (-11.5, -1.5),
            "wind_power_range": (0.0, 12.0),
            "turbulence_range": (0.0, 1.2),
            "wind_probability": 0.5,
            "target_width_range": (0.10, 0.25),
            "thrust_scale_range": (0.8, 1.3),
            "weak_gravity_prob": 0.35,
            "weak_gravity_cutoff": -5.0,
            "continuous": True,
        },
        "shield_default": False,
        "target": {
            "gamma": 0.99,
            "step_penalty": 0.01,
            "target_bonus": 50.0,
            "off_target_penalty": 20.0,
            "failed_landing_penalty": 100.0,
            "target_limit": 0.15,
            "commit_penalty_after": 500,
            "commit_penalty_height": 0.3,
            "commit_penalty_scale": 2.0,
        },
    },

    "v20": {
        "base": {
            "gravity": -10.0,
            "gravity_range": (-11.5, -1.5),
            "wind_power_range": (0.0, 12.0),
            "turbulence_range": (0.0, 0.8),
            "wind_probability": 0.5,
            "target_width_range": (0.10, 0.25),
            "thrust_scale_range": (0.9, 1.3),
            "weak_gravity_prob": 0.25,
            "weak_gravity_cutoff": -5.0,
            "continuous": True,
        },
        "shield_default": True,
        "shield": {
            "y_trigger": 0.45,
            "vy_trigger": -0.55,
            "main_thrust": 1.0,
            "angle_limit": 0.6,
        },
        "target": {
            "gamma": 0.99,
            "step_penalty": 0.01,
            "target_bonus": 20.0,
            "off_target_penalty": 5.0,
            "timeout_penalty": 200.0,
            "failed_landing_penalty": 500.0,
            "target_limit": 0.15,
        },
    },
}


OUTCOMES = (
    "PRECISE",
    "SAFE_OUTSIDE",
    "LEG_FAILURE",
    "CRASH",
    "TIMEOUT",
    "OTHER_TERMINAL",
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(path: Path, device: str = "cpu"):
    """
    Automatically detect SAC or PPO from the checkpoint itself.

    Does not depend on the filename.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {path}"
        )

    failures = []

    for algorithm in (SAC, PPO):

        try:
            model = algorithm.load(
                str(path),
                device=device
            )

            print(
                f"[eval] loaded algorithm: "
                f"{algorithm.__name__}"
            )

            return model

        except Exception as exc:

            failures.append(
                f"{algorithm.__name__}: "
                f"{type(exc).__name__}: {exc}"
            )

    details = "\n".join(
        f"  - {item}" for item in failures
    )

    raise RuntimeError(
        f"Could not load {path} as SAC or PPO.\n"
        f"{details}\n"
        "Check checkpoint validity and SB3 compatibility."
    )


# ============================================================
# MODEL / ENVIRONMENT COMPATIBILITY
# ============================================================

def spaces_match(
    model_space: gym.Space,
    env_space: gym.Space,
    label: str
) -> None:

    if type(model_space) is not type(env_space):

        raise ValueError(
            f"{label} space type mismatch: "
            f"model={type(model_space).__name__}, "
            f"environment={type(env_space).__name__}"
        )

    if isinstance(model_space, gym.spaces.Box):

        if model_space.shape != env_space.shape:

            raise ValueError(
                f"{label} shape mismatch: "
                f"model={model_space.shape}, "
                f"environment={env_space.shape}"
            )

        if not np.allclose(
            model_space.low,
            env_space.low,
            equal_nan=True
        ) or not np.allclose(
            model_space.high,
            env_space.high,
            equal_nan=True
        ):

            raise ValueError(
                f"{label} Box bounds differ. "
                "Check the selected training profile."
            )

    elif model_space != env_space:

        raise ValueError(
            f"{label} spaces differ: "
            f"{model_space} vs {env_space}"
        )


def validate_model_environment(
    model,
    env: gym.Env,
    shield: bool
) -> None:

    spaces_match(
        model.observation_space,
        env.observation_space,
        "Observation"
    )

    spaces_match(
        model.action_space,
        env.action_space,
        "Action"
    )

    if not isinstance(
        env.action_space,
        gym.spaces.Box
    ):

        raise ValueError(
            "Expected continuous Box action space."
        )

    if env.action_space.shape != (2,):

        raise ValueError(
            f"Expected two engine controls, "
            f"got {env.action_space.shape}"
        )

    if env.observation_space.shape not in ((8,), (9,)):

        raise ValueError(
            f"Expected 8 or 9 observation values, "
            f"got {env.observation_space.shape}"
        )

    if shield and env.observation_space.shape[0] < 8:

        raise ValueError(
            "SafetyShield requires standard LunarLander observations."
        )


# ============================================================
# PROFILE DETECTION
# ============================================================

def infer_profile(model_path: Path) -> str:

    stem = model_path.stem.lower()

    for token, profile in (
        ("v20", "v20"),
        ("v19b", "v19b"),
        ("v19", "v19"),
        ("v17", "v17"),
    ):

        if token in stem:
            return profile

    raise ValueError(
        f"Cannot infer profile from {model_path.name}. "
        "Use --profile v17, v19, v19b or v20."
    )


# ============================================================
# ENVIRONMENT BUILDER
# ============================================================

def build_env(
    profile: str,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    render: bool = False,
    shield: Optional[bool] = None,
) -> gym.Env:

    if profile not in ENV_PROFILES:

        raise ValueError(
            f"Unknown profile: {profile}"
        )

    if max_episode_steps <= 0:

        raise ValueError(
            "max_episode_steps must be positive."
        )

    cfg = ENV_PROFILES[profile]

    use_shield = (
        cfg["shield_default"]
        if shield is None
        else shield
    )

    base_kwargs = dict(cfg["base"])

    base_kwargs["render_mode"] = (
        "human" if render else None
    )

    env: gym.Env = ThrustScaledLunarLander(
        **base_kwargs
    )

    # Wrapper order:
    # Base -> SafetyShield -> TimeLimit
    # -> TargetLandingWrapper -> Monitor

    if use_shield:

        shield_kwargs = dict(
            cfg.get("shield", {})
        )

        env = SafetyShield(
            env,
            **shield_kwargs
        )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=max_episode_steps
    )

    env = TargetLandingWrapper(
        env,
        **dict(cfg["target"])
    )

    env = Monitor(env)

    return env


# ============================================================
# WRAPPER INSPECTION
# ============================================================

def find_wrapper(
    env: gym.Env,
    wrapper_type: type
) -> Optional[gym.Wrapper]:

    current: Any = env

    while current is not None:

        if isinstance(current, wrapper_type):
            return current

        current = getattr(
            current,
            "env",
            None
        )

    return None


def seed_environment_for_episode(
    env: gym.Env,
    seed: int
) -> None:

    """
    Seed the base environment before custom wrappers
    sample episode randomisation.
    """

    base = env.unwrapped

    base.np_random = np.random.default_rng(
        seed
    )


# ============================================================
# OUTCOME CLASSIFICATION
# ============================================================

def safe_float(
    value: Any,
    default: float = float("nan")
) -> float:

    try:

        result = float(value)

        return (
            result
            if math.isfinite(result)
            else default
        )

    except (TypeError, ValueError):

        return default


def read_stable_contact(
    env: gym.Env,
    info: dict[str, Any]
) -> int:

    if "stable_contact_steps" in info:

        try:
            return int(
                info["stable_contact_steps"]
            )

        except (TypeError, ValueError):
            pass

    wrapper = find_wrapper(
        env,
        TargetLandingWrapper
    )

    return (
        int(
            getattr(
                wrapper,
                "stable_contact_steps",
                0
            )
        )
        if wrapper
        else 0
    )


def classify(
    obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    game_over: bool,
    stable_contact_steps: int,
    target_limit: float,
    settle_frames: int,
) -> dict[str, Any]:

    obs = np.asarray(
        obs,
        dtype=np.float32
    ).reshape(-1)

    if obs.size < 8:

        raise ValueError(
            f"Expected at least 8 observations, "
            f"got {obs.size}"
        )

    both_legs = bool(
        obs[6] > 0.5 and obs[7] > 0.5
    )

    inside = bool(
        -target_limit < float(obs[0]) < target_limit
    )

    stable = bool(
        stable_contact_steps >= settle_frames
    )

    landing_status = str(
        info.get("landing_status", "")
    )

    successful = bool(
        info.get("successful_landing", False)
    )

    failed_rest = bool(
        info.get("failed_rest", False)
    )

    # Fallback for older wrapper versions.

    if "successful_landing" not in info:

        successful = bool(
            (
                terminated
                and not game_over
                and both_legs
            )
            or stable
        )

    precise = bool(
        info.get(
            "precise_landing",
            info.get(
                "target_landing",
                successful and both_legs and inside
            )
        )
    )

    if game_over or landing_status == "CRASH":

        outcome = "CRASH"

    elif truncated and not terminated:

        outcome = "TIMEOUT"

    elif (
        failed_rest
        or landing_status == "FAILED_NO_BOTH_LEGS"
    ):

        outcome = "LEG_FAILURE"

    elif successful:

        outcome = (
            "PRECISE"
            if precise
            else "SAFE_OUTSIDE"
        )

    elif terminated and not both_legs:

        outcome = "LEG_FAILURE"

    elif terminated:

        outcome = "OTHER_TERMINAL"

    else:

        outcome = "OTHER_TERMINAL"

    safe_landing = bool(
        successful and not game_over
    )

    return {
        "outcome": outcome,
        "both_legs": both_legs,
        "inside_target": inside,
        "stable_settle": stable,
        "safe_landing": safe_landing,
        "precise_landing": bool(
            precise and successful
        ),
        "landing_status": landing_status,
    }


# ============================================================
# EPISODE PARAMETER READING
# ============================================================

def episode_parameter(
    info: dict[str, Any],
    base: Any,
    info_key: str,
    attr: str,
    default: float
) -> float:

    if info_key in info:

        return safe_float(
            info[info_key],
            default
        )

    return safe_float(
        getattr(base, attr, default),
        default
    )


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run(
    model_path: Path,
    episodes: int,
    seed_offset: int,
    profile: str,
    render: bool = False,
    shield: Optional[bool] = None,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device: str = "cpu",
    deterministic: bool = True,
) -> list[dict[str, Any]]:

    if episodes <= 0:

        raise ValueError(
            "Episodes must be at least 1."
        )

    cfg = ENV_PROFILES[profile]

    effective_shield = (
        cfg["shield_default"]
        if shield is None
        else shield
    )

    print("\n========== ROCKET-LAND EVALUATION ==========")

    print(f"[eval] checkpoint : {model_path}")
    print(f"[eval] profile    : {profile}")
    print(f"[eval] episodes   : {episodes}")
    print(f"[eval] shield     : {effective_shield}")
    print(f"[eval] device     : {device}")
    print(f"[eval] deterministic: {deterministic}")

    model = load_model(
        model_path,
        device=device
    )

    env = build_env(
        profile=profile,
        max_episode_steps=max_episode_steps,
        render=render,
        shield=effective_shield
    )

    results: list[dict[str, Any]] = []

    try:

        validate_model_environment(
            model,
            env,
            effective_shield
        )

        target_wrapper = find_wrapper(
            env,
            TargetLandingWrapper
        )

        if target_wrapper is None:

            raise RuntimeError(
                "TargetLandingWrapper not found."
            )

        target_limit = float(
            target_wrapper.target_limit
        )

        settle_frames = int(
            target_wrapper.settle_frames
        )

        if profile == "v20" and not effective_shield:

            print(
                "[warning] V20 was trained with SafetyShield. "
                "This is a no-shield ablation test."
            )

        if max_episode_steps != DEFAULT_MAX_EPISODE_STEPS:

            print(
                f"[warning] Episode limit is "
                f"{max_episode_steps}; default is "
                f"{DEFAULT_MAX_EPISODE_STEPS}."
            )

        base = env.unwrapped

        print(
            f"[eval] observation space: "
            f"{env.observation_space.shape}"
        )

        print(
            f"[eval] action space: "
            f"{env.action_space.shape}"
        )

        print(
            f"[eval] target limit: "
            f"{target_limit:.3f}"
        )

        print(
            f"[eval] settle frames: "
            f"{settle_frames}"
        )

        if hasattr(base, "_target_width_range"):

            print(
                "[note] Sampled target width is logged separately. "
                "The active TargetLandingWrapper limit remains "
                f"{target_limit:.3f}."
            )

        for ep in range(episodes):

            seed = seed_offset + ep

            seed_environment_for_episode(
                env,
                seed
            )

            obs, reset_info = env.reset()

            obs = np.asarray(
                obs,
                dtype=np.float32
            )

            # Record environmental randomisation.

            gravity = episode_parameter(
                reset_info,
                base,
                "gravity",
                "current_gravity",
                -10.0
            )

            wind_power = episode_parameter(
                reset_info,
                base,
                "wind_power",
                "wind_power",
                0.0
            )

            turbulence = episode_parameter(
                reset_info,
                base,
                "turbulence_power",
                "turbulence_power",
                0.0
            )

            sampled_target_width = episode_parameter(
                reset_info,
                base,
                "target_width",
                "current_target_width",
                target_limit
            )

            thrust_scale = episode_parameter(
                reset_info,
                base,
                "thrust_scale",
                "current_thrust_scale",
                1.0
            )

            total_reward = 0.0
            final_step_reward = 0.0

            steps = 0

            terminated = False
            truncated = False

            info: dict[str, Any] = dict(
                reset_info
            )

            # Run one episode.

            while not (
                terminated or truncated
            ):

                action, _ = model.predict(
                    obs,
                    deterministic=deterministic
                )

                action = np.asarray(
                    action,
                    dtype=np.float32
                ).reshape(-1)

                if not np.all(
                    np.isfinite(action)
                ):

                    raise FloatingPointError(
                        f"Non-finite action in "
                        f"episode {ep + 1}: {action}"
                    )

                action = np.clip(
                    action,
                    env.action_space.low,
                    env.action_space.high
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                obs = np.asarray(
                    obs,
                    dtype=np.float32
                )

                step_reward = float(reward)

                total_reward += step_reward

                final_step_reward = step_reward

                steps += 1

            # Episode completed.

            stable_contact_steps = read_stable_contact(
                env,
                info
            )

            game_over = bool(
                getattr(
                    env.unwrapped,
                    "game_over",
                    False
                )
            )

            classification = classify(
                obs=obs,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
                game_over=game_over,
                stable_contact_steps=stable_contact_steps,
                target_limit=target_limit,
                settle_frames=settle_frames
            )

            row = {

                "episode": ep + 1,

                "seed": seed,

                "profile": profile,

                "gravity": round(gravity, 4),

                "wind_power": round(wind_power, 4),

                "turbulence": round(turbulence, 4),

                "sampled_target_width": round(
                    sampled_target_width,
                    4
                ),

                "active_target_limit": round(
                    target_limit,
                    4
                ),

                "thrust_scale": round(
                    thrust_scale,
                    4
                ),

                "final_x": round(
                    safe_float(obs[0]),
                    5
                ),

                "final_y": round(
                    safe_float(obs[1]),
                    5
                ),

                "final_vx": round(
                    safe_float(obs[2]),
                    5
                ),

                "final_vy": round(
                    safe_float(obs[3]),
                    5
                ),

                "final_angle": round(
                    safe_float(obs[4]),
                    5
                ),

                "final_angular_velocity": round(
                    safe_float(obs[5]),
                    5
                ),

                "steps": steps,

                "episode_reward": round(
                    total_reward,
                    4
                ),

                "final_step_reward": round(
                    final_step_reward,
                    4
                ),

                "both_legs": classification["both_legs"],

                "inside_target": classification["inside_target"],

                "stable_settle": classification["stable_settle"],

                "stable_contact_steps": stable_contact_steps,

                "safe_landing": classification["safe_landing"],

                "precise_landing": classification["precise_landing"],

                "shield_fires": int(
                    info.get("shield_fires", 0)
                ),

                "landing_status": classification["landing_status"],

                "game_over": game_over,

                "terminated": bool(terminated),

                "truncated": bool(truncated),

                "outcome": classification["outcome"],
            }

            results.append(row)

            if (
                (ep + 1) % 25 == 0
                or ep + 1 == episodes
            ):

                print(
                    f"[eval] {ep + 1:>4}/{episodes} | "
                    f"last={classification['outcome']:<14} | "
                    f"reward={total_reward:+8.2f} | "
                    f"shield_fires={row['shield_fires']}"
                )

    finally:

        env.close()

    return results


# ============================================================
# STATISTICS
# ============================================================

def pct(
    num: int,
    den: int
) -> str:

    return (
        f"{100.0 * num / den:6.2f}%"
        if den
        else "   n/a"
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96
) -> tuple[float, float]:

    """Calculate a 95% Wilson confidence interval."""

    if total <= 0:

        return (
            float("nan"),
            float("nan")
        )

    p = successes / total

    z2 = z * z

    denom = 1.0 + z2 / total

    centre = (
        p + z2 / (2.0 * total)
    ) / denom

    margin = (
        z * math.sqrt(
            (
                p * (1.0 - p) / total
            )
            +
            (
                z2 / (4.0 * total * total)
            )
        )
    ) / denom

    return (
        100.0 * max(0.0, centre - margin),
        100.0 * min(1.0, centre + margin)
    )


def summarise(
    label: str,
    rows: list[dict[str, Any]]
) -> None:

    n = len(rows)

    if n == 0:

        print(
            f"\n=== {label}: no episodes ==="
        )

        return

    counts = {

        name: sum(
            row["outcome"] == name
            for row in rows
        )

        for name in OUTCOMES
    }

    precise = counts["PRECISE"]

    safe = sum(
        bool(row["safe_landing"])
        for row in rows
    )

    crashes = counts["CRASH"]

    timeouts = counts["TIMEOUT"]

    combined_crash_timeout = (
        crashes + timeouts
    )

    leg_failures = counts["LEG_FAILURE"]

    rewards = [
        float(row["episode_reward"])
        for row in rows
    ]

    steps = [
        int(row["steps"])
        for row in rows
    ]

    ci_low, ci_high = wilson_interval(
        precise,
        n
    )

    print(
        f"\n========== {label} (n={n}) =========="
    )

    for outcome in OUTCOMES:

        print(
            f"  {outcome:18s} "
            f"{counts[outcome]:5d} "
            f"{pct(counts[outcome], n)}"
        )

    print(
        f"  {'safe landings':18s} "
        f"{safe:5d} "
        f"{pct(safe, n)}"
    )

    print(
        f"  {'precise success':18s} "
        f"{precise:5d} "
        f"{pct(precise, n)}"
    )

    print(
        f"  {'precise 95% CI':18s} "
        f"{ci_low:.2f}%–{ci_high:.2f}%"
    )

    print(
        f"  {'crash + timeout':18s} "
        f"{combined_crash_timeout:5d} "
        f"{pct(combined_crash_timeout, n)}"
    )

    print(
        f"  {'leg failures':18s} "
        f"{leg_failures:5d} "
        f"{pct(leg_failures, n)}"
    )

    print(
        f"  {'average reward':18s} "
        f"{np.mean(rewards):9.2f} "
        f"min {min(rewards):9.2f} "
        f"max {max(rewards):9.2f}"
    )

    print(
        f"  {'average steps':18s} "
        f"{np.mean(steps):9.1f}"
    )


# ============================================================
# CSV EXPORT
# ============================================================

def write_csv(
    rows: list[dict[str, Any]],
    out_path: Path
) -> None:

    if not rows:

        raise ValueError(
            "No evaluation rows. "
            "Cannot write an empty CSV."
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        rows[0].keys()
    )

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(
    path_text: str,
    default_root: Path = ROOT
) -> Path:

    path = Path(
        path_text
    ).expanduser()

    if path.is_absolute():

        return path

    if path.exists():

        return path.resolve()

    return (
        default_root / path
    ).resolve()


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Rocket-Land PPO/SAC checkpoints "
            "using their matching environment profiles."
        )
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained model checkpoint."
    )

    parser.add_argument(
        "--profile",
        choices=(
            "auto",
            "v17",
            "v19",
            "v19b",
            "v20"
        ),
        default="auto"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Custom CSV output path."
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the LunarLander simulation."
    )

    shield_group = parser.add_mutually_exclusive_group()

    shield_group.add_argument(
        "--shield",
        dest="shield",
        action="store_true",
        help="Enable SafetyShield."
    )

    shield_group.add_argument(
        "--no-shield",
        dest="shield",
        action="store_false",
        help="Disable SafetyShield."
    )

    parser.set_defaults(
        shield=None
    )

    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
            "auto"
        ),
        default="cpu"
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions."
    )

    args = parser.parse_args()

    if args.episodes <= 0:

        parser.error(
            "--episodes must be greater than zero."
        )

    if args.max_episode_steps <= 0:

        parser.error(
            "--max-episode-steps must be greater than zero."
        )

    model_path = resolve_path(
        args.model
    )

    try:

        profile = (
            infer_profile(model_path)
            if args.profile == "auto"
            else args.profile
        )

    except ValueError as exc:

        parser.error(str(exc))

    # Device selection.

    if args.device == "auto":

        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = args.device

        if device == "cuda":

            import torch

            if not torch.cuda.is_available():

                parser.error(
                    "CUDA requested, but CUDA is unavailable "
                    "in this Python environment."
                )

    # Output CSV path.

    if args.out:

        out_path = resolve_path(
            args.out
        )

    else:

        out_path = (
            ROOT
            / "logs"
            / f"evaluate_{model_path.stem}_{args.episodes}.csv"
        )

    # Run evaluation.

    rows = run(

        model_path=model_path,

        episodes=args.episodes,

        seed_offset=args.seed_offset,

        profile=profile,

        render=args.render,

        shield=args.shield,

        max_episode_steps=args.max_episode_steps,

        device=device,

        deterministic=not args.stochastic,
    )

    # Save CSV.

    write_csv(
        rows,
        out_path
    )

    print(
        f"\n[eval] CSV written: {out_path}"
    )

    # Overall performance.

    summarise(
        "OVERALL",
        rows
    )

    # Gravity analysis.

    weak_gravity = [

        row for row in rows

        if row["gravity"] > -6.0
    ]

    strong_gravity = [

        row for row in rows

        if row["gravity"] <= -6.0
    ]

    summarise(
        "GRAVITY: weaker than -6 m/s^2",
        weak_gravity
    )

    summarise(
        "GRAVITY: -6 m/s^2 or stronger",
        strong_gravity
    )

    # Wind analysis.

    wind_present = [

        row for row in rows

        if row["wind_power"] > 0.0
    ]

    wind_calm = [

        row for row in rows

        if row["wind_power"] <= 0.0
    ]

    summarise(
        "WIND: present",
        wind_present
    )

    summarise(
        "WIND: calm",
        wind_calm
    )

    # Engine thrust analysis.

    underpowered = [

        row for row in rows

        if row["thrust_scale"] < 1.0
    ]

    nominal_or_overpowered = [

        row for row in rows

        if row["thrust_scale"] >= 1.0
    ]

    summarise(
        "THRUST: scale below 1.0",
        underpowered
    )

    summarise(
        "THRUST: scale at/above 1.0",
        nominal_or_overpowered
    )

    print(
        "\n[eval] Reminder: sampled_target_width is logged "
        "for traceability. The active success boundary is "
        "active_target_limit from TargetLandingWrapper."
    )

    if len(rows) < 1000:

        print(
            "\n[eval] Statistical note: this is a diagnostic sample. "
            "Use at least 1,000 episodes for model comparison and "
            "a larger independent test set (for example 10,000 episodes) "
            "before making a 98–99% performance claim."
        )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()