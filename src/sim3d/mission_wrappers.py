"""V29 sensor seeding, kept separate from historical V28 wrapper behavior."""
import gymnasium as gym
import numpy as np

from sensor_noise import ObservationDelay
from sim3d.wrappers import IMU_BIAS_STD, IMU_NOISE_STD, ImuNoise

# Observation entries that come from the position fix (GPS / vision): horizontal
# position, altitude and velocity. Attitude, gyro, leg switches and actuator
# feedback come from the IMU and local hardware and keep updating.
NAV_SLICE = slice(0, 6)
BURST_STEPS = (1, 10)   # 20-200 ms at 50 Hz

# Horizontal position-fix quality (metres, 1-sigma per axis): per-step noise
# and the per-flight offset. 'gps' is the placeholder used since V28 (0.05 m
# noise, 0.10 m offset). 'rtk' is a carrier-phase RTK fix: about 1-2 cm, so
# 0.02 m for both is a conservative datasheet-style figure. Other entries
# (altitude, velocity, IMU) are unchanged.
NAV_PROFILES = {'gps': (0.05, 0.10), 'rtk': (0.02, 0.02)}
POS_UNIT = 10.0        # observation position = metres / 10


class SeededImuNoise(ImuNoise):
    def reset(self, **kwargs):
        seed = kwargs.get('seed')
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
        return super().reset(**kwargs)


class NavDropout(gym.Wrapper):
    """Position-fix dropouts: during a burst the navigation entries hold their
    last value. `rate` is the per-step chance that a burst starts."""

    def __init__(self, env, rate=0.0):
        super().__init__(env)
        if not 0 <= rate < 1:
            raise ValueError('Dropout rate must be in [0, 1)')
        self.rate = float(rate)
        self._rng = np.random.default_rng()
        self._held = None
        self._left = 0

    def reset(self, **kwargs):
        seed = kwargs.get('seed')
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
        obs, info = self.env.reset(**kwargs)
        self._held = obs[NAV_SLICE].copy()
        self._left = 0
        return obs, info

    def step(self, action):
        obs, reward, te, tr, info = self.env.step(action)
        if self._left == 0 and self.rate > 0 and self._rng.random() < self.rate:
            self._left = int(self._rng.integers(BURST_STEPS[0], BURST_STEPS[1] + 1))
        if self._left > 0:
            self._left -= 1
            obs = obs.copy()
            obs[NAV_SLICE] = self._held
        else:
            self._held = obs[NAV_SLICE].copy()
        return obs, reward, te, tr, info


def mission_sensors(raw, delay=0, noise_scale=0.0, dropout=0.0, nav='gps'):
    if delay < 0 or not 0 <= noise_scale <= 1:
        raise ValueError('Nonnegative delay and noise scale in [0,1] required')
    if nav not in NAV_PROFILES:
        raise ValueError(f'nav must be one of {sorted(NAV_PROFILES)}')
    noise, bias = IMU_NOISE_STD.copy(), IMU_BIAS_STD.copy()
    noise[:2], bias[:2] = (v / POS_UNIT for v in NAV_PROFILES[nav])
    # Same random draws for every profile, so 'gps' vs 'rtk' audits are paired.
    env = SeededImuNoise(ObservationDelay(raw, delay_steps=delay),
                         noise_std=noise, bias_std=bias)
    env.set_scale(noise_scale)
    return NavDropout(env, rate=dropout) if dropout > 0 else env
