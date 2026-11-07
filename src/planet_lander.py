"""PlanetLunarLander: LunarLander with configurable gravity per reset.

- `gravity`: fixed value (negative, m/s^2 in Box2D units), used at every reset.
- `gravity_range`: (low, high) tuple; a fresh value is sampled every reset.
  Use this at training time for domain randomization.

The underlying gymnasium LunarLander rebuilds its Box2D world each reset()
from self.gravity, so changing that attribute before reset is sufficient.

Gymnasium's LunarLander __init__ validates gravity in (-12, 0). We bypass
that validator so we can expose Jupiter-like gravities, but still clip to
a physically sensible training range at reset.
"""

import numpy as np

from randomized_lander import RandomStartLunarLander


# Catalogue of real-world gravities (m/s^2, negative = toward "down"). Values
# outside [-11.99, -0.3] are silently clipped at reset time.
PLANET_GRAVITY = {
    "moon": -1.62,
    "mars": -3.71,
    "mercury": -3.70,
    "pluto": -0.62,
    "venus": -8.87,
    "uranus": -8.69,
    "earth": -9.81,
    "saturn": -10.44,
    "neptune": -11.15,
    # Below are provided but clipped by Box2D-env sanity range.
    "jupiter": -24.79,
    "sun": -274.0,
}

GRAVITY_MIN = -11.95
GRAVITY_MAX = -0.5


def resolve_gravity(name_or_value):
    """Accepts a planet name (case-insensitive) or a numeric value. Returns
    a float gravity, clipped to the supported Box2D range."""
    if isinstance(name_or_value, str):
        key = name_or_value.strip().lower()
        if key not in PLANET_GRAVITY:
            raise ValueError(
                f"Unknown planet '{name_or_value}'. "
                f"Known: {sorted(PLANET_GRAVITY)}"
            )
        value = PLANET_GRAVITY[key]
    else:
        value = float(name_or_value)

    clipped = float(np.clip(value, GRAVITY_MIN, GRAVITY_MAX))
    return clipped, value


class PlanetLunarLander(RandomStartLunarLander):

    def __init__(
        self,
        gravity=-10.0,
        gravity_range=None,
        render_mode=None,
        enable_wind=False,
        hard_start_probability=0.0,
        continuous=True,
        # V18: oversample weak gravity so the agent sees more "long-fall" episodes.
        weak_gravity_prob=0.0,
        weak_gravity_cutoff=-5.0,
    ):
        # Initialise base env with a safe default, then overwrite self.gravity.
        super().__init__(
            render_mode=render_mode,
            enable_wind=enable_wind,
            hard_start_probability=hard_start_probability,
            continuous=continuous,
        )

        base_g, _ = resolve_gravity(gravity)
        self._base_gravity = base_g
        self._gravity_range = gravity_range  # (low, high) or None
        self.current_gravity = base_g
        self.gravity = base_g  # consumed by LunarLander.reset()

        # V18 extras
        self._weak_gravity_prob = float(np.clip(weak_gravity_prob, 0.0, 1.0))
        self._weak_gravity_cutoff = float(
            np.clip(weak_gravity_cutoff, GRAVITY_MIN, GRAVITY_MAX)
        )

    def _pick_gravity(self):
        if self._gravity_range is None:
            return self._base_gravity
        lo, hi = self._gravity_range
        lo = float(np.clip(lo, GRAVITY_MIN, GRAVITY_MAX))
        hi = float(np.clip(hi, GRAVITY_MIN, GRAVITY_MAX))
        if lo > hi:
            lo, hi = hi, lo
        rng = getattr(self, "np_random", None) or np.random
        if self._weak_gravity_prob > 0.0 and rng.random() < self._weak_gravity_prob:
            # Bias toward weak gravity (closer to zero). cutoff is a negative
            # number > lo; sample uniformly in [cutoff, hi] clipped to range.
            weak_lo = max(lo, self._weak_gravity_cutoff)
            weak_hi = hi
            if weak_lo <= weak_hi:
                return float(rng.uniform(weak_lo, weak_hi))
        return float(rng.uniform(lo, hi))

    def set_gravity(self, name_or_value):
        """Fix the gravity for subsequent resets (no sampling)."""
        base_g, _ = resolve_gravity(name_or_value)
        self._base_gravity = base_g
        self._gravity_range = None
        self.gravity = base_g
        self.current_gravity = base_g

    def reset(self, *, seed=None, options=None):
        g = self._pick_gravity()
        self.gravity = g  # LunarLander.reset() uses this to build Box2D world
        self.current_gravity = g

        obs, info = super().reset(seed=seed, options=options)

        info = dict(info)
        info["gravity"] = float(self.current_gravity)
        return obs, info
