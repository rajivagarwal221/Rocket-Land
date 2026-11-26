"""Nominal observation-driven guidance; no simulator or randomized-state access.

Inputs follow the 16-value V28 observation layout. The local heading frame is
handled algebraically: horizontal components rotate between heading and body
using the tilt rotation with yaw factored out. No absolute heading is needed.

Wind handling (Wave 1 guidance):
- A disturbance observer compares measured horizontal velocity with what the
  thrust vector (body axis plus gimbal deflection) should produce. The
  difference -- wind, mass and thrust mismatch -- is fed forward into steering.
- A position integral removes the steady offset left by the attitude loop when
  the gimbal holds a constant deflection (centre-of-mass offset).
- Touchdown hold: below TD_ALT the descent pauses until sideways speed, offset
  and the lean needed to hold station are all small (a gust lull), then drops
  quickly so the next gust has little time to act.
- The hover gate at 1.5 m is loose on purpose: precise centring happens in the
  touchdown hold, so gusts no longer burn the mission clock at altitude.
- The hold releases only within 0.2 m of the measured centre: the per-flight
  navigation offset (~0.1 m per axis) plus drift during the drop then stays
  inside the 0.5 m pad.
- During the final drop the lean is capped, trading a few centimetres of
  drift for staying under the touchdown tilt limit.
"""
import math

import numpy as np

from sim3d.rocket_env import RocketParams

OBSERVER_BW = 2.0                  # rad/s, critically damped disturbance observer
DISTURBANCE_MAX = 3.0              # m/s^2
POSITION_INTEGRAL = 0.3            # 1/s^3
GATE_ERROR, GATE_SPEED = 0.4, 0.5              # m, m/s: ACQUIRE -> DESCEND
ABORT_ERROR, ABORT_ALT = 0.45, 0.7             # DESCEND -> ACQUIRE
MIN_DESCENT = 0.25                 # m/s
TD_ALT = 0.6                       # m, touchdown hold zone
TD_SPEED, TD_ERROR = 0.25, 0.2     # m/s, m; 0.4 let 13 of 1551 dev flights
                                   # (wind <= 4 m/s) drift past the 0.5 m pad
TD_LEAN = math.radians(6.0)        # release only when the needed lean is below this
TD_MAX_HOLD = 4.0                  # s per attempt
TD_DROP = 0.5                      # m/s final drop after release
TD_DROP_LEAN = math.radians(8.0)   # lean cap during the final drop


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
        self._disturbance = np.zeros(2)   # heading-frame horizontal accel estimate
        self._v_est = None                # observer velocity estimate
        self._integral = np.zeros(2)
        self._vz_integral = 0.0
        self._hold_time = 0.0
        self._dropping = False

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

        # Heading-frame states rotate with yaw; compensate before updating them.
        yaw_rate = float((R[1] @ np.cross(omega, [1., 0., 0.])) / cx)
        rot = lambda v: v - dt * yaw_rate * np.array([-v[1], v[0]])
        self._disturbance = rot(self._disturbance)
        self._integral = rot(self._integral)
        if error < 2:
            self._integral = np.clip(self._integral + position * dt, -4, 4)
        if self._v_est is None:
            self._v_est = velocity.copy()
        else:
            self._v_est = rot(self._v_est)
            # Thrust direction includes the gimbal deflection; otherwise the
            # engine's own side force is mistaken for wind.
            sx, sy = np.sin(o[14:16] * p.max_gimbal)
            thrust_dir = np.array([sx, sy, math.sqrt(max(0.0, 1 - sx*sx - sy*sy))])
            thrust_acc = o[13] * p.max_thrust / p.mass * (R @ thrust_dir)[:2]
            innovation = velocity - self._v_est
            self._v_est += dt * (thrust_acc + self._disturbance + 2 * OBSERVER_BW * innovation)
            self._disturbance = np.clip(self._disturbance + dt * OBSERVER_BW**2 * innovation,
                                        -DISTURBANCE_MAX, DISTURBANCE_MAX)

        acceleration = np.clip(-0.8 * position - 1.8 * velocity
                               - POSITION_INTEGRAL * self._integral, -2, 2) - self._disturbance
        lean_needed = math.atan(float(np.linalg.norm(acceleration)) / p.gravity)
        if self._dropping and lean_needed > TD_DROP_LEAN:
            # Below the hold height a late gust should cost a few centimetres of
            # drift, not a touchdown past the tilt limit.
            acceleration *= math.tan(TD_DROP_LEAN) * p.gravity / float(np.linalg.norm(acceleration))
        desired_axis = np.r_[acceleration / p.gravity, 1.0]
        desired_axis /= np.linalg.norm(desired_axis)
        # Cross product is transformed back to body coordinates for torque.
        attitude_error = R.T @ np.cross(R[:, 2], desired_axis)
        torque = p.inertia_xy * (60 * attitude_error[:2] - 14 * omega[:2])

        speed = float(np.linalg.norm(velocity))
        if self.phase == 'ACQUIRE' and error < GATE_ERROR and speed < GATE_SPEED:
            self.phase = 'DESCEND'
        elif self.phase == 'DESCEND' and error > ABORT_ERROR and alt > ABORT_ALT:
            self.phase = 'ACQUIRE'
        v_des = -float(np.clip(0.6 * max(alt, 0), MIN_DESCENT, 3))
        if self.phase == 'ACQUIRE' and alt < 2.5:
            v_des = float(np.clip(0.7 * (1.5-alt), -0.7, 0.7))
        elif self.phase == 'DESCEND' and alt < TD_ALT:
            unready = (float(np.linalg.norm(self._v_est)) > TD_SPEED
                       or error > TD_ERROR or lean_needed > TD_LEAN)
            if unready and self._hold_time < TD_MAX_HOLD:
                self._hold_time += dt
                v_des = float(np.clip(0.7 * TD_ALT - alt, -0.3, 0.3))
                self._dropping = False
            else:
                v_des = -TD_DROP
                self._dropping = True
        v_err = v_des - vz
        self._vz_integral = float(np.clip(self._vz_integral + v_err*dt, -1, 1))
        thrust = p.mass * (p.gravity + 4*v_err + 6*self._vz_integral) / max(up[2], 0.5)
        throttle = float(np.clip(thrust/p.max_thrust, 0, 1))
        force_arm = max(throttle*p.max_thrust*0.4, 0.1)
        gx = math.asin(float(np.clip(-torque[1]/force_arm, -1, 1))) / p.max_gimbal
        gy = math.asin(float(np.clip(torque[0]/force_arm, -1, 1))) / p.max_gimbal
        roll = -4*omega[2]*p.inertia_z/p.max_roll_torque
        return np.clip([2*throttle-1, gx, gy, roll], -1, 1).astype(np.float32)
