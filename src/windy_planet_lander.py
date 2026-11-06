"""Wind + turbulence + gravity + 9-dim obs."""
import numpy as np
from gravity_obs_lander import GravityObsLunarLander


class WindyPlanetLunarLander(GravityObsLunarLander):

    def __init__(
        self,
        *args,
        wind_power_range=(0.0, 15.0),
        turbulence_range=(0.0, 1.5),
        wind_probability=0.75,
        **kwargs,
    ):
        kwargs["enable_wind"] = True
        super().__init__(*args, **kwargs)
        self._wind_power_range = wind_power_range
        self._turbulence_range = turbulence_range
        self._wind_probability = wind_probability

    def reset(self, *, seed=None, options=None):
        rng = getattr(self, "np_random", None) or np.random
        if rng.random() < self._wind_probability:
            self.wind_power = float(
                rng.uniform(*self._wind_power_range)
            )
            self.turbulence_power = float(
                rng.uniform(*self._turbulence_range)
            )
            self.enable_wind = True
        else:
            self.wind_power = 0.0
            self.turbulence_power = 0.0
            self.enable_wind = False
        obs, info = super().reset(seed=seed, options=options)
        info["wind_power"] = self.wind_power
        info["turbulence_power"] = self.turbulence_power
        return obs, info
