"""Nominal observation-driven guidance; no simulator or randomized-state access.

Inputs follow the 16-value V28 observation layout. The local heading frame is
handled algebraically: horizontal components rotate between heading and body
using the tilt rotation with yaw factored out. No absolute heading is needed.
"""
import math

import numpy as np

from sim3d.rocket_env import RocketParams


def residual_action(baseline, residual, authority=(0.10, 0.15, 0.15, 0.10)):
    base, delta, limits = map(np.asarray, (baseline, residual, authority))
    if any(x.shape != (4,) for x in (base, delta, limits)):
        raise ValueError('Expected four baseline, residual and authority values')
    if not all(np.isfinite(x).all() for x in (base, delta, limits)) or (limits < 0).any():
        raise ValueError('Controls must be finite; authority must be nonnegative')
    return np.clip(base + np.clip(delta, -1, 1) * limits, -1, 1).astype(np.float32)


class LandingController:
    def __init__(self, params=None, dt=0.02):
        self.params = params or RocketParams()
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.phase = 'ACQUIRE'
        self._integral = np.zeros(2)
        self._vz_integral = 0.0

    def __call__(self, observation):
        o = np.asarray(observation, dtype=float)
        if o.shape != (16,) or not np.isfinite(o).all():
            raise ValueError('Controller requires 16 finite observation values')
        p, dt = self.params, self.dt
        position, alt = o[:2] * 10, o[2] * 10
        velocity, vz, omega = o[3:5] * 5, o[5] * 5, o[9:12] * 2
        up = o[6:9] / max(np.linalg.norm(o[6:9]), 1e-6)
        # Recover the body rotation in heading coordinates using world up and
        # the definition that body-x has zero heading-y projection.
        cx = math.sqrt(max(1e-8, 1 - up[0]**2))
        R = np.array([[cx, -up[0]*up[1]/cx, -up[0]*up[2]/cx],
                      [0, up[2]/cx, -up[1]/cx], up])
        if o[12] > 0 and alt < 0.12:
            self.phase = 'SETTLE'
            return np.array([-1, 0, 0, 0], dtype=np.float32)
        error = float(np.linalg.norm(position))
        if self.phase == 'SETTLE':
            self.phase = 'ACQUIRE' if error > .45 else 'DESCEND'
        # Integrals live in the heading frame; compensate its yaw rotation.
        yaw_rate = float((R[1] @ np.cross(omega, [1., 0., 0.])) / cx)
        self._integral -= dt * yaw_rate * np.array([-self._integral[1], self._integral[0]])
        if error < 2:
            self._integral = np.clip(self._integral + position * dt, -4, 4)
        acceleration = np.clip(-0.8 * position - 1.8 * velocity - 0.3*self._integral, -2, 2)
        desired_axis = np.r_[acceleration / p.gravity, 1.0]
        desired_axis /= np.linalg.norm(desired_axis)
        # Cross product is transformed back to body coordinates for torque.
        attitude_error = R.T @ np.cross(R[:, 2], desired_axis)
        torque = p.inertia_xy * (60 * attitude_error[:2] - 14 * omega[:2])
        if self.phase == 'ACQUIRE' and error < 0.22 and np.linalg.norm(velocity) < 0.3:
            self.phase = 'DESCEND'
        elif self.phase == 'DESCEND' and error > 0.45 and alt > 0.7:
            self.phase = 'ACQUIRE'
        v_des = -float(np.clip(0.6 * max(alt, 0), 0.25, 3))
        if self.phase == 'ACQUIRE' and alt < 2.5:
            v_des = float(np.clip(0.7 * (1.5-alt), -0.7, 0.7))
        v_err = v_des - vz
        self._vz_integral = float(np.clip(self._vz_integral + v_err*dt, -1, 1))
        thrust = p.mass * (p.gravity + 4*v_err + 6*self._vz_integral) / max(up[2], 0.5)
        throttle = float(np.clip(thrust/p.max_thrust, 0, 1))
        force_arm = max(throttle*p.max_thrust*0.4, 0.1)
        gx = math.asin(float(np.clip(-torque[1]/force_arm, -1, 1))) / p.max_gimbal
        gy = math.asin(float(np.clip(torque[0]/force_arm, -1, 1))) / p.max_gimbal
        roll = -4*omega[2]*p.inertia_z/p.max_roll_torque
        return np.clip([2*throttle-1, gx, gy, roll], -1, 1).astype(np.float32)
