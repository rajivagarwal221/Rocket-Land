"""Sim-to-real layer and env factories for the 3D lander.

Stack, inside to outside:
    Rocket3DEnv        physics, actuator lag, wind, mass / thrust randomization
    ObservationDelay   sensor-to-computer latency (reused from Phase 1)
    ImuNoise           per-step sensor noise plus a per-episode bias
    SubprocVecEnv / DummyVecEnv -> VecMonitor -> VecFrameStack

ImuNoise is the outermost single-env wrapper so the vec env can reach
set_curriculum() on it directly.
"""
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
)

from sensor_noise import ObservationDelay
from sim3d.rocket_env import OBS_DIM, EnvConfig, Rocket3DEnv, RocketParams

# Per-dim Gaussian std in observation units (see the layout in rocket_env.py).
# PLACEHOLDERS: replace with the real GPS / IMU datasheet values.
IMU_NOISE_STD = np.array(
    [0.005, 0.005,            # horizontal position   (5 cm)
     0.003,                   # altitude              (3 cm)
     0.010, 0.010, 0.010,     # velocity              (5 cm/s)
     0.004, 0.004, 0.004,     # attitude              (about 0.25 deg)
     0.005, 0.005, 0.005,     # gyro                  (0.01 rad/s)
     0.0, 0.0, 0.0, 0.0],     # leg switches and actuator feedback stay exact
    dtype=np.float32,
)
# Per-episode constant offset: estimator drift and gyro bias.
IMU_BIAS_STD = np.array(
    [0.010, 0.010,
     0.003,
     0.0, 0.0, 0.0,
     0.004, 0.004, 0.0,
     0.005, 0.005, 0.005,
     0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)

INFO_KEYWORDS = ("outcome", "landing_error", "impact_speed")


class ImuNoise(gym.Wrapper):
    """Gaussian sensor noise plus a bias that is redrawn every episode."""

    def __init__(self, env, noise_std=IMU_NOISE_STD, bias_std=IMU_BIAS_STD, seed=None):
        super().__init__(env)
        self._noise_std = np.asarray(noise_std, dtype=np.float32)
        self._bias_std = np.asarray(bias_std, dtype=np.float32)
        if self._noise_std.shape != (OBS_DIM,) or self._bias_std.shape != (OBS_DIM,):
            raise ValueError(f"noise_std and bias_std must have {OBS_DIM} entries")
        self._scale = 1.0
        self._rng = np.random.default_rng(seed)
        self._bias = np.zeros(OBS_DIM, dtype=np.float32)

    def set_scale(self, scale: float):
        self._scale = float(np.clip(scale, 0.0, 1.0))

    def set_curriculum(self, difficulty: float, target_radius: float) -> bool:
        """One call the vec env uses to move every curriculum dial together."""
        self.set_scale(difficulty)
        return self.env.unwrapped.set_curriculum(difficulty, target_radius)

    def _noisy(self, obs):
        if self._scale == 0.0:
            return obs
        noise = self._rng.standard_normal(OBS_DIM).astype(np.float32)
        return obs + self._bias + noise * self._noise_std * self._scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._bias = (
            self._rng.standard_normal(OBS_DIM).astype(np.float32)
            * self._bias_std
            * self._scale
        )
        return self._noisy(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy(obs), reward, terminated, truncated, info


def make_env(
    rank: int = 0,
    seed: int = 0,
    difficulty: float = 0.0,
    target_radius: float = 0.5,
    delay_steps: int = 2,
    params: RocketParams = None,
    config: EnvConfig = None,
    render_mode: str = None,
):
    """Return a thunk that builds one fully wrapped env (picklable for subprocesses)."""

    def _init():
        env = Rocket3DEnv(
            params=params,
            config=config,
            difficulty=difficulty,
            target_radius=target_radius,
            render_mode=render_mode,
        )
        env = ObservationDelay(env, delay_steps=delay_steps)
        env = ImuNoise(env, seed=seed + 7919 * (rank + 1))
        env.set_scale(difficulty)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec(
    n_envs: int,
    seed: int,
    difficulty: float,
    target_radius: float,
    n_stack: int,
    subproc: bool = False,
    delay_steps: int = 2,
    monitor_file: str = None,
):
    fns = [
        make_env(
            rank=i,
            seed=seed,
            difficulty=difficulty,
            target_radius=target_radius,
            delay_steps=delay_steps,
        )
        for i in range(n_envs)
    ]
    if subproc and n_envs > 1:
        start = "spawn" if sys.platform == "win32" else "forkserver"
        vec = SubprocVecEnv(fns, start_method=start)
    else:
        vec = DummyVecEnv(fns)
    vec = VecMonitor(vec, filename=monitor_file, info_keywords=INFO_KEYWORDS)
    if n_stack > 1:
        vec = VecFrameStack(vec, n_stack=n_stack)
    return vec


def base_env(vec, index: int = 0) -> Rocket3DEnv:
    """The raw Rocket3DEnv behind a DummyVecEnv-based stack (viewer, baseline)."""
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    return inner.envs[index].unwrapped
