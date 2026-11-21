"""V29 sensor seeding, kept separate from historical V28 wrapper behavior."""
import numpy as np

from sensor_noise import ObservationDelay
from sim3d.wrappers import ImuNoise


class SeededImuNoise(ImuNoise):
    def reset(self, **kwargs):
        seed = kwargs.get('seed')
        if seed is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
        return super().reset(**kwargs)


def mission_sensors(raw, delay=0, noise_scale=0.0):
    if delay < 0 or not 0 <= noise_scale <= 1:
        raise ValueError('Nonnegative delay and noise scale in [0,1] required')
    env = SeededImuNoise(ObservationDelay(raw, delay_steps=delay))
    env.set_scale(noise_scale)
    return env
