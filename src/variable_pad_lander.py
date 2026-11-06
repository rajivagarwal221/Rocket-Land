"""Variable-helipad env: overrides pad position/width per episode.

The base LunarLander hard-codes helipad at CHUNKS//2-1 .. CHUNKS//2+1 and
helipad_y = H/4. We override attributes AFTER super().reset() builds the
terrain is not an option because they are consumed during reset. Instead
we subclass reset to run the parent then rebuild terrain segments.

Simpler approach: monkey-patch via a custom reset that overrides the
helipad bounds used by the reward's "inside_target" check. We use a
curriculum-style randomization that shifts the target zone horizontally
(by adjusting target_limit and providing an effective pad offset to
TargetLandingWrapper via info).

Scope: for minimal risk, we vary only the TARGET WIDTH (tolerance) here
and let the physical pad remain at centre. This is the training signal
that matters for precision generalization.
"""
import numpy as np
from windy_planet_lander import WindyPlanetLunarLander


class VariablePadLunarLander(WindyPlanetLunarLander):
    def __init__(self, *args, target_width_range=(0.10, 0.25), **kwargs):
        super().__init__(*args, **kwargs)
        self._target_width_range = target_width_range
        self.current_target_width = 0.15

    def reset(self, *, seed=None, options=None):
        rng = getattr(self, "np_random", None) or np.random
        self.current_target_width = float(
            rng.uniform(*self._target_width_range)
        )
        obs, info = super().reset(seed=seed, options=options)
        info["target_width"] = self.current_target_width
        return obs, info
