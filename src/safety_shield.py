"""SafetyShield: deterministic override near the ground.

A classical safety filter in the spirit of real rocket landing guidance:
whenever the lander is close to the ground AND still descending fast,
the shield forces the main engine to full thrust, overriding whatever
the policy chose. Horizontal control stays with the policy.

This is NOT a learned behaviour. It is a hand-written floor beneath the
policy so that a momentary bad action near the ground cannot turn into
a crash. For a real rocket you want both: a good policy AND a shield.

Place this wrapper BETWEEN the base env and TimeLimit/TargetLandingWrapper.
It only works on the continuous action space (Box(-1, 1, (2,))).
"""
import gymnasium as gym
import numpy as np


class SafetyShield(gym.Wrapper):

    def __init__(
        self,
        env,
        y_trigger=0.45,          # altitude below which shield engages
        vy_trigger=-0.55,        # descent speed (more negative = faster down)
        main_thrust=1.0,         # full main engine when shield fires
        angle_limit=0.6,         # radians; too tilted -> let policy right itself
        disable_on_contact=True, # once a leg touches, hand back to policy
    ):
        super().__init__(env)
        self.y_trigger = float(y_trigger)
        self.vy_trigger = float(vy_trigger)
        self.main_thrust = float(main_thrust)
        self.angle_limit = float(angle_limit)
        self.disable_on_contact = bool(disable_on_contact)
        self._last_obs = None
        self.shield_fires_this_episode = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.shield_fires_this_episode = 0
        return obs, info

    def _should_fire(self, obs):
        if obs is None:
            return False
        y = float(obs[1])
        vy = float(obs[3])
        angle = float(obs[4])
        left_leg = bool(obs[6] > 0.5)
        right_leg = bool(obs[7] > 0.5)

        if self.disable_on_contact and (left_leg or right_leg):
            return False
        if abs(angle) > self.angle_limit:
            return False  # too tilted; firing main would push sideways
        return y < self.y_trigger and vy < self.vy_trigger

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self._should_fire(self._last_obs):
            action[0] = self.main_thrust
            self.shield_fires_this_episode += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        info = dict(info)
        info["shield_fires"] = self.shield_fires_this_episode
        return obs, reward, terminated, truncated, info
