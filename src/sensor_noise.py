"""Sensor-side wrappers for sim-to-real realism.

A real rocket does not have perfect state access. Its state estimator
(GPS + IMU fusion) returns:

  1. Noisy position  / velocity / attitude / angular rate
  2. Delayed values (the sensor chain has latency)

These wrappers model both so the policy learns to be robust to what
the real hardware will actually hand it.

Order relative to SafetyShield: the shield must see TRUE observations
from the base env (otherwise it would fire on phantoms). So we place
these wrappers OUTSIDE the shield. The shield's internal _last_obs
stays clean; only what the policy sees is corrupted.

Wrapper stack (inside to outside):
  ThrustScaledLunarLander     (base + env randomizations)
  SafetyShield                (sees clean obs; may override action)
  ObservationDelay            (delays the obs returned up to the policy)
  SensorNoise                 (adds Gaussian noise on the delayed obs)
  TimeLimit
  TargetLandingWrapper        (reward on policy-visible obs is fine)

Noise levels are applied per-dim and configurable. Leg-contact booleans
(obs[6], obs[7]) and the normalized-gravity obs[8] must stay noise-free
(booleans + perfect-knowledge prior); their std is 0 by default.
"""
import collections

import numpy as np
import gymnasium as gym


# Per-dim Gaussian std for the 9-dim observation
# [x, y, vx, vy, angle, omega, leg1, leg2, g_norm]
DEFAULT_SENSOR_NOISE_STD = np.array(
    [
        0.010,   # x        (approx 1% viewport)
        0.010,   # y
        0.015,   # vx
        0.015,   # vy
        0.009,   # angle    (approx 0.5 deg)
        0.010,   # omega
        0.000,   # leg1     (boolean, noise would be silly)
        0.000,   # leg2
        0.000,   # g_norm   (perfect-knowledge prior from env)
    ],
    dtype=np.float32,
)


class SensorNoise(gym.Wrapper):
    """Add zero-mean Gaussian noise per observation dim.

    Noise is applied in reset() too so the first observation the policy
    sees is already noisy (consistency with later steps).
    """

    def __init__(self, env, noise_std=None, seed=None):

        super().__init__(env)

        if noise_std is None:

            noise_std = DEFAULT_SENSOR_NOISE_STD

        noise_std = np.asarray(noise_std, dtype=np.float32)

        obs_dim = int(env.observation_space.shape[0])

        if noise_std.shape[0] != obs_dim:

            raise ValueError(
                f"noise_std has {noise_std.shape[0]} dims but obs is "
                f"{obs_dim}-dim."
            )

        self._noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def _noisy(self, obs):

        noise = self._rng.normal(
            loc=0.0,
            scale=self._noise_std,
            size=self._noise_std.shape,
        ).astype(np.float32)

        out = np.asarray(obs, dtype=np.float32) + noise
        # Leg contacts are booleans represented as 0/1 floats; never drift them
        out[6] = float(obs[6])
        out[7] = float(obs[7])
        return out

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)
        return self._noisy(obs), info

    def step(self, action):

        obs, r, te, tr, info = self.env.step(action)
        return self._noisy(obs), r, te, tr, info


class ObservationDelay(gym.Wrapper):
    """Delay observations by N steps.

    At every step, returns the observation from N steps ago. The first N
    returned observations are copies of the reset observation (physically
    reasonable: at t=0 the estimator has no history and returns current state).

    The wrapper stores TRUE-obs history internally. SafetyShield is INSIDE
    this wrapper, so the shield already saw the fresh obs through its own
    step() and made its decision before this wrapper delays what the policy
    receives.
    """

    def __init__(self, env, delay_steps=2):

        super().__init__(env)
        self._delay = int(delay_steps)

        if self._delay < 0:

            raise ValueError("delay_steps must be >= 0")

        self._buffer: collections.deque = collections.deque(
            maxlen=max(1, self._delay + 1)
        )

    def _emit(self):

        # Oldest buffered obs (= the one "delay" steps behind current)
        return self._buffer[0]

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self._buffer.clear()

        for _ in range(self._delay + 1):

            self._buffer.append(
                np.asarray(obs, dtype=np.float32).copy()
            )

        return self._emit(), info

    def step(self, action):

        obs, r, te, tr, info = self.env.step(action)
        self._buffer.append(
            np.asarray(obs, dtype=np.float32).copy()
        )
        return self._emit(), r, te, tr, info
