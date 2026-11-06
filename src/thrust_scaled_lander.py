"""Thrust-scaled env: scales main / side engine power per episode.

We monkey-patch the module-level MAIN_ENGINE_POWER and SIDE_ENGINE_POWER
constants read by LunarLander.step(). The scaling factor is randomized
once per reset so the policy must infer thrust capability from early
dynamics.
"""
import numpy as np
from gymnasium.envs.box2d import lunar_lander as _ll

from variable_pad_lander import VariablePadLunarLander

_DEFAULT_MAIN = _ll.MAIN_ENGINE_POWER
_DEFAULT_SIDE = _ll.SIDE_ENGINE_POWER


class ThrustScaledLunarLander(VariablePadLunarLander):
    def __init__(self, *args, thrust_scale_range=(0.8, 1.3), **kwargs):
        super().__init__(*args, **kwargs)
        self._thrust_scale_range = thrust_scale_range
        self.current_thrust_scale = 1.0

    def reset(self, *, seed=None, options=None):
        rng = getattr(self, "np_random", None) or np.random
        self.current_thrust_scale = float(
            rng.uniform(*self._thrust_scale_range)
        )
        _ll.MAIN_ENGINE_POWER = _DEFAULT_MAIN * self.current_thrust_scale
        _ll.SIDE_ENGINE_POWER = _DEFAULT_SIDE * self.current_thrust_scale
        obs, info = super().reset(seed=seed, options=options)
        info["thrust_scale"] = self.current_thrust_scale
        return obs, info

    def close(self):
        _ll.MAIN_ENGINE_POWER = _DEFAULT_MAIN
        _ll.SIDE_ENGINE_POWER = _DEFAULT_SIDE
        return super().close()
