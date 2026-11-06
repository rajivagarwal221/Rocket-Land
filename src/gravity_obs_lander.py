"""Gravity-aware env: augments obs with a 9th dim = normalized gravity.

Normalization: g_obs = (gravity - GRAVITY_MEAN) / GRAVITY_SCALE.
Gravities are negative; range used for training is [-11.5, -1.5], so
GRAVITY_MEAN = -6.5, GRAVITY_SCALE = 5.0 puts obs in ~[-1, 1].

This changes the observation space from 8 to 9 dims; a fresh PPO is
required (weights can be warm-started layer by layer except for the
first input linear).
"""
import numpy as np
import gymnasium as gym

from planet_lander import PlanetLunarLander, resolve_gravity, PLANET_GRAVITY

GRAVITY_MEAN = -6.5
GRAVITY_SCALE = 5.0


def normalize_gravity(g):
    return float((g - GRAVITY_MEAN) / GRAVITY_SCALE)


class GravityObsLunarLander(PlanetLunarLander):
    """PlanetLunarLander + 9-dim obs (appends normalized gravity)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        low = np.concatenate(
            [self.observation_space.low, np.array([-3.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [self.observation_space.high, np.array([3.0], dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, obs):
        g_norm = normalize_gravity(self.current_gravity)
        return np.concatenate([obs, np.array([g_norm], dtype=np.float32)])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        return self._augment(obs), r, te, tr, info
