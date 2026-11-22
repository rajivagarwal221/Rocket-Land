"""3D rocket landing on MuJoCo: 6-DOF rigid body, gimballed engine, four legs.

Action  Box(-1, 1, (4,))
    [0] throttle       -1 = engine off, +1 = full thrust
    [1] gimbal x       tilts the thrust vector toward body +x
    [2] gimbal y       tilts the thrust vector toward body +y
    [3] roll torque    reaction-control torque about the long axis

Observation  Box((16,)), yaw-invariant so the policy never needs a compass
    [0:2]   horizontal offset from the pad, in the vehicle's heading frame / 10 m
    [2]     altitude of the feet above the ground / 10 m
    [3:5]   horizontal velocity in the heading frame / 5 m/s
    [5]     vertical velocity / 5 m/s
    [6:9]   world "up" expressed in the body frame (attitude)
    [9:12]  body angular velocity / 2 rad/s
    [12]    fraction of legs touching the ground
    [13]    actual throttle (after engine lag)
    [14:16] actual gimbal angles / max gimbal (after servo lag)

Episode outcomes (info["outcome"] on the last step)
    PRECISE         settled upright inside the target radius
    SAFE_OUTSIDE    settled upright outside the target radius
    CRASH           touched down too fast, or lost control in flight
    TIPPED          touched down gently and then fell over
    OUT_OF_BOUNDS   left the flight volume
    TIMEOUT         ran out of steps

Reward keeps the Phase-1 (V23) recipe: potential-based shaping on distance,
speed and attitude, plus terminal bonuses. No fuel term (the V24 lesson).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np

XML_PATH = Path(__file__).with_name("rocket.xml")

OBS_DIM = 16
ACT_DIM = 4
OUTCOMES = (
    "PRECISE",
    "SAFE_OUTSIDE",
    "CRASH",
    "TIPPED",
    "OUT_OF_BOUNDS",
    "TIMEOUT",
)


@dataclass
class RocketParams:
    """Nominal vehicle. PLACEHOLDERS until the real rocket is measured:
    replace every value here with bench data before trusting sim-to-real."""

    mass: float = 1.5               # kg, landing mass
    inertia_xy: float = 0.085       # kg m^2 about pitch / yaw axes
    inertia_z: float = 0.004        # kg m^2 about the long (roll) axis
    max_thrust: float = 30.0        # N (thrust-to-weight about 2.0)
    max_gimbal: float = 0.12        # rad (about 7 deg)
    max_roll_torque: float = 0.05   # N m
    thrust_tau: float = 0.05        # s, engine first-order response
    gimbal_tau: float = 0.03        # s, gimbal servo first-order response
    gimbal_slew: float = 3.0        # rad/s, servo rate limit
    drag_area: float = 0.064        # m^2, Cd * A for cross-flow
    cp_offset: float = 0.05         # m, centre of pressure above the COM
    air_density: float = 1.2        # kg/m^3
    gravity: float = 9.81           # m/s^2


@dataclass
class EnvConfig:
    control_hz: int = 50
    physics_hz: int = 500
    max_steps: int = 750            # 15 s
    gamma: float = 0.99

    # Start state at difficulty 1. Spreads shrink to start_floor at difficulty 0.
    alt_range: tuple = (8.0, 14.0)
    lateral_max: float = 4.0
    vz_range: tuple = (-5.0, -2.0)
    vxy_max: float = 1.5
    tilt_max: float = 0.17
    omega_max: float = 0.3
    start_floor: float = 0.4

    # Per-episode domain randomization, all scaled by difficulty.
    mass_var: float = 0.15
    inertia_var: float = 0.15
    com_offset: float = 0.01        # m, COM offset from the thrust line
    thrust_var: float = 0.10
    misalign: float = 0.01          # rad, thrust-line misalignment
    lag_var: float = 0.5
    thrust_noise: float = 0.02
    wind_prob: float = 0.7
    wind_max: float = 6.0           # m/s mean wind
    gust_frac: float = 0.3
    gust_tau: float = 1.0

    # Touchdown and safety limits.
    max_touchdown_speed: float = 1.5   # firmer: harder bounces -> CRASH instead of 'maybe retry'
    tip_angle: float = 1.05            # rad (60 deg)
    bounds_xy: float = 20.0
    bounds_z: float = 30.0
    settle_steps: int = 5              # 15 was unreachable under noise
    settle_speed: float = 0.4          # "soft touchdown" instead of "parked"
    settle_omega: float = 0.6
    settle_legs: int = 2               # legs>=3 was impossible with wind/lag
    post_contact_deadline: int = 100   # kills the bounce-forever loop
    pos_clip_m: float = 5.0            # cap distance potential so near-pad is a plateau

    # Reward.
    target_radius: float = 0.5
    w_pos: float = 100.0
    pos_scale: float = 10.0
    lateral_weight: float = 1.5
    w_vel: float = 100.0
    vel_scale: float = 5.0
    w_tilt: float = 100.0
    w_omega: float = 10.0
    w_legs: float = 0.0                # was 40; made hover-with-leg-scraping profitable
    step_penalty: float = 0.02         # mild hover tax
    action_rate_coef: float = 0.02
    descent_commit_bonus: float = 0.5  # +/step while committing: alt<2, offset<1, vz<-0.2
    landed_reward: float = 100.0
    target_bonus: float = 500.0
    off_target_penalty: float = 120.0
    soft_touchdown_bonus: float = 100.0
    crash_penalty: float = 400.0
    timeout_penalty: float = 800.0     # was 400; hover-to-timeout must be strictly worse than attempting
    oob_penalty: float = 500.0


@dataclass
class Kinematics:
    pos: np.ndarray        # world position of the body origin
    v_world: np.ndarray
    omega: np.ndarray      # body frame
    p_h: np.ndarray        # horizontal offset from the pad, heading frame
    v_h: np.ndarray        # horizontal velocity, heading frame
    alt: float             # feet above ground
    up_body: np.ndarray    # world up in the body frame
    z_h: np.ndarray        # body long axis lean, heading frame
    tilt: float            # rad from vertical
    heading: np.ndarray    # unit vector: body x axis projected on the ground


class Rocket3DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        params: Optional[RocketParams] = None,
        config: Optional[EnvConfig] = None,
        difficulty: float = 0.0,
        target_radius: Optional[float] = None,
        render_mode: Optional[str] = None,
    ):
        self.params = params or RocketParams()
        self.cfg = config or EnvConfig()
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.target_radius = float(
            self.cfg.target_radius if target_radius is None else target_radius
        )
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        m, p, cfg = self.model, self.params, self.cfg

        if cfg.physics_hz % cfg.control_hz != 0:
            raise ValueError("physics_hz must be a multiple of control_hz")
        m.opt.timestep = 1.0 / cfg.physics_hz
        m.opt.gravity[:] = (0.0, 0.0, -p.gravity)
        self.frame_skip = cfg.physics_hz // cfg.control_hz
        self.dt_ctrl = 1.0 / cfg.control_hz

        def gid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        self._bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rocket")
        self._ground = gid("ground")
        self._pad = gid("pad")
        self._flame = gid("flame")
        self._legs = frozenset(gid(f"leg{i}") for i in range(4))
        self._hull = frozenset((gid("fuselage"), gid("nose")))
        self._engine = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "engine")
        self._engine_arm = float(-m.site_pos[self._engine][2])
        self._rest_height = self._measure_rest_height()

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)

        self._renderer = None
        self._steps = 0
        self._thrust = 0.0
        self._gx = 0.0
        self._gy = 0.0
        self._prev_action = np.zeros(ACT_DIM)
        self._wind_mean = np.zeros(2)
        self._gust = np.zeros(2)
        self.episode_meta: dict = {}

    # ------------------------------------------------------------ helpers

    def _measure_rest_height(self) -> float:
        """Height of the body origin above the ground when standing on its legs."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = 0.0
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        lowest = 0.0
        for g in self._legs:
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            half = m.geom_size[g][1]
            for end in (d.geom_xpos[g] + axis * half, d.geom_xpos[g] - axis * half):
                lowest = min(lowest, end[2] - m.geom_size[g][0])
        return float(-lowest)

    def set_curriculum(self, difficulty: float, target_radius: float) -> bool:
        """Difficulty takes effect at the next reset; the radius immediately."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.target_radius = float(target_radius)
        return True

    def kinematics(self) -> Kinematics:
        d = self.data
        R = d.xmat[self._bid].reshape(3, 3)
        pos = d.xpos[self._bid]
        v_world = d.qvel[:3]
        hx = R[:2, 0]
        n = math.hypot(hx[0], hx[1])
        hx = hx / n if n > 1e-6 else np.array([1.0, 0.0])
        hy = np.array([-hx[1], hx[0]])
        e = pos[:2]
        z_axis = R[:2, 2]
        return Kinematics(
            pos=pos.copy(),
            v_world=v_world.copy(),
            omega=d.qvel[3:6].copy(),
            p_h=np.array([e @ hx, e @ hy]),
            v_h=np.array([v_world[:2] @ hx, v_world[:2] @ hy]),
            alt=float(pos[2] - self._rest_height),
            up_body=R[2, :].copy(),
            z_h=np.array([z_axis @ hx, z_axis @ hy]),
            tilt=float(math.acos(max(-1.0, min(1.0, R[2, 2])))),
            heading=hx,
        )

    def _leg_contacts(self):
        d = self.data
        legs = set()
        hull = False
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom
            if g1 == self._ground:
                other = g2
            elif g2 == self._ground:
                other = g1
            else:
                continue
            if other in self._legs:
                legs.add(int(other))
            elif other in self._hull:
                hull = True
        return len(legs), hull

    def _obs(self, k: Kinematics, legs: int) -> np.ndarray:
        p = self.params
        return np.array(
            [
                k.p_h[0] / 10.0,
                k.p_h[1] / 10.0,
                k.alt / 10.0,
                k.v_h[0] / 5.0,
                k.v_h[1] / 5.0,
                k.v_world[2] / 5.0,
                k.up_body[0],
                k.up_body[1],
                k.up_body[2],
                k.omega[0] / 2.0,
                k.omega[1] / 2.0,
                k.omega[2] / 2.0,
                legs / 4.0,
                self._thrust,
                self._gx / p.max_gimbal,
                self._gy / p.max_gimbal,
            ],
            dtype=np.float32,
        )

    def _potential(self, k: Kinematics, legs: int) -> float:
        c = self.cfg
        lat = c.lateral_weight * math.hypot(k.pos[0], k.pos[1])
        dist = min(math.hypot(lat, max(k.alt, 0.0)), c.pos_clip_m)
        speed = float(np.linalg.norm(k.v_world))
        return (
            -c.w_pos * dist / c.pos_scale
            - c.w_vel * speed / c.vel_scale
            - c.w_tilt * k.tilt
            - c.w_omega * float(np.linalg.norm(k.omega))
            + c.w_legs * legs / 4.0
        )

    # -------------------------------------------------------------- reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng, m, d, p, c = self.np_random, self.model, self.data, self.params, self.cfg
        diff = self.difficulty
        spread = c.start_floor + (1.0 - c.start_floor) * diff

        def sym(scale, size=None):
            return rng.uniform(-1.0, 1.0, size) * scale

        mass_scale = 1.0 + sym(c.mass_var * diff)
        inertia_scale = 1.0 + sym(c.inertia_var * diff, 3)
        m.body_mass[self._bid] = p.mass * mass_scale
        m.body_inertia[self._bid] = (
            np.array([p.inertia_xy, p.inertia_xy, p.inertia_z])
            * mass_scale
            * inertia_scale
        )
        # A COM that is off the thrust line is modelled by shifting the engine sideways.
        m.site_pos[self._engine][:2] = sym(c.com_offset * diff, 2)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)

        thrust_scale = 1.0 + sym(c.thrust_var * diff)
        self._max_thrust = p.max_thrust * thrust_scale
        tau_t = p.thrust_tau * (1.0 + sym(c.lag_var * diff))
        tau_g = p.gimbal_tau * (1.0 + sym(c.lag_var * diff))
        dt = m.opt.timestep
        self._k_thrust = 1.0 - math.exp(-dt / tau_t)
        self._k_gimbal = 1.0 - math.exp(-dt / tau_g)
        self._slew = p.gimbal_slew * dt
        self._gx_bias = float(sym(c.misalign * diff))
        self._gy_bias = float(sym(c.misalign * diff))
        self._thrust_noise = c.thrust_noise * diff

        wind_speed = 0.0
        if diff > 0.0 and rng.random() < c.wind_prob:
            wind_speed = float(rng.uniform(0.0, c.wind_max * diff))
        wind_dir = rng.uniform(-math.pi, math.pi)
        self._wind_mean = wind_speed * np.array([math.cos(wind_dir), math.sin(wind_dir)])
        self._gust_sigma = c.gust_frac * wind_speed
        self._gust = np.zeros(2)

        radius = c.lateral_max * spread * math.sqrt(rng.random())
        bearing = rng.uniform(-math.pi, math.pi)
        d.qpos[0] = radius * math.cos(bearing)
        d.qpos[1] = radius * math.sin(bearing)
        d.qpos[2] = rng.uniform(*c.alt_range)

        q_yaw, q_tilt, quat = np.zeros(4), np.zeros(4), np.zeros(4)
        tilt_axis = rng.uniform(-math.pi, math.pi)
        mujoco.mju_axisAngle2Quat(
            q_yaw, np.array([0.0, 0.0, 1.0]), rng.uniform(-math.pi, math.pi)
        )
        mujoco.mju_axisAngle2Quat(
            q_tilt,
            np.array([math.cos(tilt_axis), math.sin(tilt_axis), 0.0]),
            rng.uniform(0.0, c.tilt_max * spread),
        )
        mujoco.mju_mulQuat(quat, q_tilt, q_yaw)
        d.qpos[3:7] = quat

        d.qvel[0] = sym(c.vxy_max * spread)
        d.qvel[1] = sym(c.vxy_max * spread)
        d.qvel[2] = rng.uniform(*c.vz_range)
        d.qvel[3:6] = sym(c.omega_max * spread, 3)
        mujoco.mj_forward(m, d)

        self._steps = 0
        self._thrust = 0.0
        self._gx = 0.0
        self._gy = 0.0
        self._prev_action = np.zeros(ACT_DIM)
        self._touched = False
        self._impact_speed = 0.0
        self._impact_tilt = 0.0
        self._settle = 0
        self._since_touch = 0
        self._air_density = p.air_density

        self.episode_meta = {
            "mass_scale": float(mass_scale),
            "thrust_scale": float(thrust_scale),
            "wind_speed": wind_speed,
            "start_offset": float(radius),
            "start_alt": float(d.qpos[2]),
            "difficulty": diff,
        }

        k = self.kinematics()
        self._prev_phi = self._potential(k, 0)
        return self._obs(k, 0), dict(self.episode_meta)

    # --------------------------------------------------------------- step

    def _apply_wind(self):
        """Cross-flow drag from mean wind plus an Ornstein-Uhlenbeck gust."""
        c, p, d = self.cfg, self.params, self.data
        if self._gust_sigma > 0.0:
            a = self.dt_ctrl / c.gust_tau
            self._gust += -self._gust * a + self._gust_sigma * math.sqrt(
                2.0 * a
            ) * self.np_random.standard_normal(2)
        v_rel = np.array(
            [
                self._wind_mean[0] + self._gust[0] - d.qvel[0],
                self._wind_mean[1] + self._gust[1] - d.qvel[1],
                -d.qvel[2],
            ]
        )
        force = 0.5 * self._air_density * p.drag_area * np.linalg.norm(v_rel) * v_rel
        arm = d.xmat[self._bid].reshape(3, 3)[:, 2] * p.cp_offset
        d.xfrc_applied[self._bid, :3] = force
        d.xfrc_applied[self._bid, 3:] = np.cross(arm, force)

    def _physics_step(self):
        mujoco.mj_step(self.model, self.data)

    def _is_settled(self, k, legs):
        return (
            legs >= self.cfg.settle_legs
            and float(np.linalg.norm(k.v_world)) < self.cfg.settle_speed
            and float(np.linalg.norm(k.omega)) < self.cfg.settle_omega
        )

    def _contact_deadline_outcome(self, error):
        return "PRECISE" if error <= self.target_radius else "SAFE_OUTSIDE"

    def step(self, action):
        m, d, p, c = self.model, self.data, self.params, self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        throttle_cmd = 0.5 * (a[0] + 1.0)
        gx_cmd = a[1] * p.max_gimbal
        gy_cmd = a[2] * p.max_gimbal
        roll_torque = a[3] * p.max_roll_torque

        speed_before = float(np.linalg.norm(d.qvel[:3]))
        self._apply_wind()
        thrust_gain = self._max_thrust
        if self._thrust_noise > 0.0:
            thrust_gain *= 1.0 + self._thrust_noise * self.np_random.standard_normal()

        ctrl = d.ctrl
        for _ in range(self.frame_skip):
            self._thrust += (throttle_cmd - self._thrust) * self._k_thrust
            dgx = (gx_cmd - self._gx) * self._k_gimbal
            dgy = (gy_cmd - self._gy) * self._k_gimbal
            self._gx += max(-self._slew, min(self._slew, dgx))
            self._gy += max(-self._slew, min(self._slew, dgy))
            force = self._thrust * thrust_gain
            sx = math.sin(self._gx + self._gx_bias)
            sy = math.sin(self._gy + self._gy_bias)
            ctrl[0] = force * sx
            ctrl[1] = force * sy
            ctrl[2] = force * math.sqrt(max(0.0, 1.0 - sx * sx - sy * sy))
            ctrl[3] = roll_torque
            self._physics_step()

        self._steps += 1
        k = self.kinematics()
        legs, hull = self._leg_contacts()

        if legs > 0 and not self._touched:
            self._touched = True
            self._impact_speed = speed_before
            self._impact_tilt = k.tilt
        if self._touched:
            self._since_touch += 1

        outcome = None
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            outcome = "CRASH"
        elif self._touched and self._impact_speed > c.max_touchdown_speed:
            outcome = "CRASH"
        elif hull or k.tilt > c.tip_angle:
            outcome = "TIPPED" if self._touched else "CRASH"
        elif (
            abs(k.pos[0]) > c.bounds_xy
            or abs(k.pos[1]) > c.bounds_xy
            or k.pos[2] > c.bounds_z
        ):
            outcome = "OUT_OF_BOUNDS"
        else:
            settled = self._is_settled(k, legs)
            self._settle = self._settle + 1 if settled else 0
            if self._settle >= c.settle_steps:
                error = math.hypot(k.pos[0], k.pos[1])
                outcome = "PRECISE" if error <= self.target_radius else "SAFE_OUTSIDE"
            elif (
                self._touched
                and self._since_touch >= c.post_contact_deadline
            ):
                # After first contact the episode MUST resolve. Bouncing forever is not an option.
                error = math.hypot(k.pos[0], k.pos[1])
                outcome = self._contact_deadline_outcome(error)

        truncated = False
        if outcome is None and self._steps >= c.max_steps:
            outcome = "TIMEOUT"
            truncated = True
        terminated = outcome is not None and not truncated

        if outcome == "CRASH" and not np.isfinite(d.qpos).all():
            # Physics blew up: hand back a finite observation and end the episode.
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            return obs, -c.crash_penalty, True, False, self._final_info("CRASH", 0.0)

        phi = self._potential(k, legs)
        reward = c.gamma * phi - self._prev_phi - c.step_penalty
        reward -= c.action_rate_coef * float(np.mean((a - self._prev_action) ** 2))
        # Pay the policy for committing to a descent near the pad: it fights the
        # tendency to hover and collect shaping.
        offset_m = math.hypot(k.pos[0], k.pos[1])
        if k.alt < 2.0 and offset_m < 1.0 and k.v_world[2] < -0.2:
            reward += c.descent_commit_bonus
        self._prev_phi = phi
        self._prev_action = a

        info = {}
        if outcome is not None:
            error = math.hypot(k.pos[0], k.pos[1])
            if outcome in ("PRECISE", "SAFE_OUTSIDE"):
                softness = 1.0 - min(1.0, self._impact_speed / c.max_touchdown_speed)
                reward += c.landed_reward + c.soft_touchdown_bonus * softness
                reward += c.target_bonus if outcome == "PRECISE" else -c.off_target_penalty
            elif outcome == "TIMEOUT":
                reward -= c.timeout_penalty
            elif outcome == "OUT_OF_BOUNDS":
                reward -= c.oob_penalty
            else:
                reward -= c.crash_penalty
            info = self._final_info(outcome, error)

        return self._obs(k, legs), float(reward), terminated, truncated, info

    def _final_info(self, outcome: str, error: float) -> dict:
        info = dict(self.episode_meta)
        info.update(
            outcome=outcome,
            landing_error=float(error),
            impact_speed=float(self._impact_speed),
            impact_tilt=float(self._impact_tilt),
            target_radius=self.target_radius,
            precise_landing=outcome == "PRECISE",
            is_success=outcome == "PRECISE",
        )
        return info

    # ------------------------------------------------------------- render

    def sync_visuals(self):
        """Match the pad disc and flame to the current target radius and throttle."""
        self.model.geom_size[self._pad][0] = self.target_radius
        self.model.geom_rgba[self._flame][3] = 0.9 * self._thrust

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.sync_visuals()
        self._renderer.update_scene(self.data, camera="wide")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def describe(self) -> dict:
        """Vehicle and environment settings, for the reproducibility manifest."""
        return {"params": asdict(self.params), "config": asdict(self.cfg)}
