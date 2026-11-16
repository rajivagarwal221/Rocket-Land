"""Classical PID guidance for the 3D lander.

Two jobs: prove the simulated vehicle is physically landable before any
learning, and give the learned policy a number it has to beat. It reads the
true simulator state (no sensor noise or delay), so it is a reference, not a
flight controller.

Cascade: position error -> desired lean -> gimbal torque, plus a descent-rate
law on throttle that slows the vehicle as it nears the ground. Integral terms
lean the vehicle into steady wind and absorb mass / thrust mismatch; the engine
is cut at first leg contact.
"""
import math

import numpy as np

KP_POS = 0.6       # 1/s^2
KD_POS = 1.4       # 1/s
KI_POS = 0.3       # 1/s^3
INTEGRAL_ZONE = 1.0  # m, integrate only this close to the pad
GATE_ALT = 2.0       # m, below this the descent waits for the lateral error
GATE_OFFSET = 0.25   # m
INTEGRAL_MAX = 3.0   # m s
ACC_MAX = 2.0      # m/s^2 lateral
KP_ATT = 60.0      # 1/s^2
KD_ATT = 14.0      # 1/s
KV = 4.0           # 1/s on descent-rate error
KI_V = 6.0         # 1/s^2, absorbs mass and thrust mismatch
K_ROLL = 4.0       # 1/s on roll rate


class PIDController:
    def __init__(self):
        self.reset()

    def reset(self):
        self._integral = np.zeros(2)
        self._v_integral = 0.0

    def __call__(self, env) -> np.ndarray:
        e = env.unwrapped
        p = e.params
        k = e.kinematics()
        g = p.gravity

        if e._leg_contacts()[0] > 0:
            return np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Integrate position error in the world frame, only near the pad (anti-windup),
        # then express it in the heading frame like the other terms.
        if float(np.linalg.norm(k.pos[:2])) < INTEGRAL_ZONE:
            self._integral = np.clip(
                self._integral + k.pos[:2] * e.dt_ctrl, -INTEGRAL_MAX, INTEGRAL_MAX
            )
        hx = k.heading
        integ = np.array(
            [self._integral @ hx, self._integral @ np.array([-hx[1], hx[0]])]
        )

        acc = -KP_POS * k.p_h - KD_POS * k.v_h - KI_POS * integ
        ax, ay = np.clip(acc, -ACC_MAX, ACC_MAX)

        # Lean the long axis toward the direction we need to accelerate in.
        lean_err_x = ax / g - k.z_h[0]
        lean_err_y = ay / g - k.z_h[1]
        tau_y = p.inertia_xy * (KP_ATT * lean_err_x - KD_ATT * k.omega[1])
        tau_x = p.inertia_xy * (-KP_ATT * lean_err_y - KD_ATT * k.omega[0])

        v_des = -float(np.clip(0.5 * k.alt, 0.4, 4.0))
        offset = float(np.linalg.norm(k.pos[:2]))
        if k.alt < GATE_ALT and offset > GATE_OFFSET:
            v_des = max(v_des, -0.5 * (k.alt - 0.5 * GATE_ALT))  # hold above the pad until centred
        v_err = v_des - k.v_world[2]
        self._v_integral = float(np.clip(self._v_integral + v_err * e.dt_ctrl, -1.0, 1.0))
        thrust = p.mass * (g + KV * v_err + KI_V * self._v_integral) / max(k.up_body[2], 0.5)
        throttle = float(np.clip(thrust / p.max_thrust, 0.0, 1.0))

        # A side force at the engine, below the COM, turns the vehicle the other way.
        force = max(throttle * p.max_thrust, 1.0)
        arm = e._engine_arm
        gx = math.asin(float(np.clip(-tau_y / (arm * force), -1.0, 1.0)))
        gy = math.asin(float(np.clip(tau_x / (arm * force), -1.0, 1.0)))
        roll = -K_ROLL * k.omega[2] * p.inertia_z / p.max_roll_torque

        return np.clip(
            np.array(
                [2.0 * throttle - 1.0, gx / p.max_gimbal, gy / p.max_gimbal, roll],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )
