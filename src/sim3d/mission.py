"""V29 mission semantics; legacy V28 defaults remain in rocket_env.py."""
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim3d.rocket_env import EnvConfig, Rocket3DEnv


@dataclass
class MissionConfig(EnvConfig):
    max_steps: int = 1000
    max_touchdown_speed: float = 1.0
    settle_steps: int = 25
    settle_speed: float = 0.1
    settle_omega: float = 0.1
    settle_legs: int = 3
    settle_tilt: float = math.radians(10)
    post_contact_deadline: int = 150


class MissionEnv(Rocket3DEnv):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, config=config or MissionConfig(), **kwargs)

    def reset(self, **kwargs):
        self._impact_horizontal = 0.0
        self._impact_vertical = 0.0
        self._ever_hull = False
        self._tick_settle_valid = True
        self._saturation_steps = 0
        obs, info = super().reset(**kwargs)
        self._minimum_altitude = self.kinematics().alt
        return obs, info

    def _physics_step(self):
        before = self.data.qvel[:3].copy()
        mujoco.mj_step(self.model, self.data)
        legs, hull = self._leg_contacts()
        self._ever_hull |= hull
        k = self.kinematics()
        self._minimum_altitude = min(self._minimum_altitude, k.alt)
        if not (super()._is_settled(k, legs) and k.tilt <= self.cfg.settle_tilt):
            self._tick_settle_valid = False
            self._settle = 0
        if legs:
            self._touched = True
            self._impact_speed = max(self._impact_speed, float(np.linalg.norm(before)))
            self._impact_tilt = max(self._impact_tilt, k.tilt)
            self._impact_horizontal = max(self._impact_horizontal, float(np.linalg.norm(before[:2])))
            self._impact_vertical = max(self._impact_vertical, float(abs(before[2])))

    def _is_settled(self, k, legs):
        return (self._tick_settle_valid and super()._is_settled(k, legs)
                and k.tilt <= self.cfg.settle_tilt)

    def _contact_deadline_outcome(self, error):
        return 'UNSETTLED'

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Expected four finite engine controls')
        self._tick_settle_valid = True
        self._saturation_steps += int(np.any(np.abs(action) >= 0.999))
        obs, _, te, tr, info = super().step(action)
        if self._ever_hull or (self._touched and self._impact_tilt > self.cfg.settle_tilt):
            te, tr = True, False
            k = self.kinematics()
            info = self._final_info('CRASH', math.hypot(*k.pos[:2]))
        # This is a mission deadline, not an externally interrupted rollout.
        if tr:
            te, tr = True, False
        k = self.kinematics()
        # Undiscounted diagnostic mission cost: no positive per-step rewards.
        # RL reward design is intentionally deferred until the baseline gate.
        cost = self.dt_ctrl * (
            1.0 + 0.1 * min(float(np.linalg.norm(k.pos[:2])), 20.0)
            + 0.1 * min(k.tilt, math.pi)
        )
        reward = -cost
        if te:
            reward += 100.0 if info['outcome'] == 'PRECISE' else -100.0
        return obs, reward, te, tr, info

    def _final_info(self, outcome, error):
        info = super()._final_info(outcome, error)
        k = self.kinematics()
        info.update(
            mission_version='v29', final_altitude=k.alt, final_tilt=k.tilt,
            final_speed=float(np.linalg.norm(k.v_world)),
            final_angular_speed=float(np.linalg.norm(k.omega)),
            settled_seconds=self._settle * self.dt_ctrl,
            impact_horizontal=self._impact_horizontal,
            impact_vertical=self._impact_vertical,
            minimum_altitude=self._minimum_altitude,
            final_x=float(k.pos[0]), final_y=float(k.pos[1]), final_z=float(k.pos[2]),
            saturation_steps=self._saturation_steps,
        )
        return info
