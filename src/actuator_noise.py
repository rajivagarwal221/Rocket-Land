"""Actuator-side wrapper for sim-to-real realism.

A real rocket engine does not instantly produce the exact thrust the
policy commands. Two effects dominate:

  1. Multiplicative + additive noise around the commanded thrust
     (gimbal / throttle hardware is not ideal).
  2. First-order response lag: the engine chases the command with a
     time constant tau. Commanded step changes become smoothed.

This wrapper sits OUTSIDE SafetyShield in the stack:

    policy -> ActuatorNoise -> SafetyShield -> ThrustScaled -> ...

Rationale: the shield's purpose is to guarantee a safe command in
specific physical states. If we corrupt the action AFTER the shield,
the shield's safety claim is void. Placing ActuatorNoise OUTSIDE the
shield means the shield operates on the policy's intent, the shield
may override, and whatever the shield emits is then corrupted by
actuator noise + lag -- which matches reality (the shield in a real
rocket is typically a flight-computer safety law before the engine
driver, and the physical engine is still noisy / laggy).

Action space: Box(-1, 1, (2,))
    action[0] = main engine (-1 off, +1 full)
    action[1] = side engines (-1 full left, +1 full right)
"""
import numpy as np
import gymnasium as gym


class ActuatorNoise(gym.Wrapper):
    """Add Gaussian noise plus first-order lag to the commanded action."""

    def __init__(
        self,
        env,
        noise_std=0.02,                # per-dim additive Gaussian std
        lag_alpha=0.5,                 # EMA smoothing: a_eff = alpha * a_new + (1 - alpha) * a_prev
        clip=True,                     # clip final action to [-1, 1]
        seed=None,
    ):

        super().__init__(env)

        self._noise_std = float(noise_std)

        if not (0.0 < lag_alpha <= 1.0):

            raise ValueError("lag_alpha must be in (0, 1]")

        self._alpha = float(lag_alpha)
        self._clip = bool(clip)
        self._rng = np.random.default_rng(seed)

        self._prev_action = None

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        action_dim = int(np.prod(self.env.action_space.shape))
        self._prev_action = np.zeros(action_dim, dtype=np.float32)

        return obs, info

    def _corrupt(self, action):

        a = np.asarray(action, dtype=np.float32).copy()

        # Additive noise
        if self._noise_std > 0.0:

            a = a + self._rng.normal(
                loc=0.0,
                scale=self._noise_std,
                size=a.shape,
            ).astype(np.float32)

        # First-order lag (EMA); alpha=1 disables lag.
        if self._alpha < 1.0:

            a = (
                self._alpha * a
                + (1.0 - self._alpha) * self._prev_action
            )

        self._prev_action = a.copy()

        if self._clip:

            a = np.clip(a, -1.0, 1.0)

        return a

    def step(self, action):

        corrupted = self._corrupt(action)
        obs, r, te, tr, info = self.env.step(corrupted)
        info = dict(info)
        info["commanded_action"] = np.asarray(action, dtype=np.float32)
        info["effective_action"] = corrupted
        return obs, r, te, tr, info
