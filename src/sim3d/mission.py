"""V29 mission semantics; legacy V28 defaults remain in rocket_env.py."""
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim3d.rocket_env import EnvConfig, Rocket3DEnv

G0 = 9.80665  # m/s^2, standard gravity for specific impulse


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

    # Wave 1 realism. Set any value to 0 to switch that feature off.
    # Propellant and Isp are PLACEHOLDERS until the real motor is measured.
    propellant_fraction: float = 0.17          # of wet landing mass
    isp: float = 150.0                         # s
    min_throttle: float = 0.3                  # thrust floor while the engine is lit
    pad_tilt_max: float = math.radians(3)      # scaled by difficulty
    density_var: float = 0.10                  # +/- site air density, scaled by difficulty


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
        c, m, rng, diff = self.cfg, self.model, self.np_random, self.difficulty
        self._wet_mass = float(m.body_mass[self._bid])
        self._wet_inertia = m.body_inertia[self._bid].copy()
        # Mass-dependent constants cached by mj_setConst at reset; scaled during the burn
        # because mj_setConst itself overwrites the live state.
        self._wet_cache = {name: getattr(m, name).copy() for name in
                           ('body_subtreemass', 'dof_M0', 'dof_invweight0',
                            'body_invweight0', 'actuator_acc0')}
        self._propellant = c.propellant_fraction * self._wet_mass
        if c.density_var > 0:
            self._air_density = self.params.air_density * (
                1.0 + rng.uniform(-1.0, 1.0) * c.density_var * diff)
        pad_tilt, quat = 0.0, np.array([1.0, 0.0, 0.0, 0.0])
        if c.pad_tilt_max > 0:
            pad_tilt = float(rng.uniform(0.0, c.pad_tilt_max * diff))
            bearing = rng.uniform(-math.pi, math.pi)
            mujoco.mju_axisAngle2Quat(
                quat, np.array([math.cos(bearing), math.sin(bearing), 0.0]), pad_tilt)
        m.geom_quat[self._ground] = quat
        m.geom_quat[self._pad] = quat
        mujoco.mj_forward(m, self.data)
        self.episode_meta.update(pad_tilt=pad_tilt, air_density=float(self._air_density),
                                 propellant_kg=float(self._propellant))
        info.update(self.episode_meta)
        self._minimum_altitude = self.kinematics().alt
        return obs, info

    def _burn(self):
        """Deplete propellant for one physics substep; mass and inertia follow."""
        c, d, m = self.cfg, self.data, self.model
        if c.propellant_fraction <= 0:
            return
        if self._propellant <= 0:
            d.ctrl[:3] = 0.0
            return
        force = float(np.linalg.norm(d.ctrl[:3]))
        self._propellant = max(0.0, self._propellant - force * m.opt.timestep / (c.isp * G0))
        burned = c.propellant_fraction * self._wet_mass - self._propellant
        mass = self._wet_mass - burned
        r = mass / self._wet_mass
        w = self._wet_cache
        m.body_mass[self._bid] = mass
        m.body_inertia[self._bid] = self._wet_inertia * r
        m.body_subtreemass[:] = w['body_subtreemass'] * r
        m.dof_M0[:] = w['dof_M0'] * r
        m.dof_invweight0[:] = w['dof_invweight0'] / r
        m.body_invweight0[self._bid] = w['body_invweight0'][self._bid] / r
        m.actuator_acc0[:] = w['actuator_acc0'] / r

    def _physics_step(self):
        self._burn()
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
        if self.cfg.min_throttle > 0:
            # A lit engine cannot throttle below the floor: low commands either
            # shut it down or are raised to the floor.
            action = action.copy()
            cmd = 0.5 * (float(np.clip(action[0], -1, 1)) + 1.0)
            floor = self.cfg.min_throttle
            cmd = 0.0 if cmd < 0.5 * floor else max(cmd, floor)
            action[0] = 2.0 * cmd - 1.0
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
            propellant_left_kg=float(getattr(self, '_propellant', 0.0)),
            final_mass=float(self.model.body_mass[self._bid]),
        )
        return info
