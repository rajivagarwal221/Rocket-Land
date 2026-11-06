import math
import numpy as np

from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    FPS,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
    LEG_DOWN
)


class RandomStartLunarLander(LunarLander):

    def __init__(
        self,
        render_mode=None,
        enable_wind=False,
        hard_start_probability=0.0,
        continuous=True
    ):

        if not 0.0 <= hard_start_probability <= 1.0:
            raise ValueError(
                "Hard-start probability must be between 0 and 1."
            )

        super().__init__(
            render_mode=render_mode,
            continuous=continuous,
            enable_wind=enable_wind,
            wind_power=10.0,
            turbulence_power=0.8
        )

        self.hard_start_probability = (
            hard_start_probability
        )

        self.continuous = continuous

    # =====================================
    # RANDOM HORIZONTAL POSITION
    # =====================================

    def get_random_x(self, rng):

        hard_start = (
            rng.random() < self.hard_start_probability
        )

        if hard_start:

            side = -1 if rng.random() < 0.5 else 1

            return side * rng.uniform(0.18, 0.45)

        return rng.uniform(-0.45, 0.45)

    # =====================================
    # RESET
    # =====================================

    def reset(self, *, seed=None, options=None):

        observation, info = super().reset(
            seed=seed,
            options=options
        )

        rng = self.np_random

        # Initial position
        x_obs = self.get_random_x(rng)

        y_obs = rng.uniform(0.85, 1.25)

        # Initial velocity
        vx_obs = rng.uniform(-0.50, 0.50)

        vy_obs = rng.uniform(-0.25, 0.25)

        # Initial rotation
        angle = rng.uniform(
            -math.radians(30),
            math.radians(30)
        )

        omega = rng.uniform(-0.50, 0.50)

        # Physics dimensions
        world_width = VIEWPORT_W / SCALE

        world_height = VIEWPORT_H / SCALE

        x = (
            world_width / 2
            + x_obs * world_width / 2
        )

        y = (
            self.helipad_y
            + LEG_DOWN / SCALE
            + y_obs * world_height / 2
        )

        vx = vx_obs * FPS / (world_width / 2)

        vy = vy_obs * FPS / (world_height / 2)

        # Rotate landing legs
        old_x = float(self.lander.position.x)

        old_y = float(self.lander.position.y)

        old_angle = float(self.lander.angle)

        delta = angle - old_angle

        cos_delta = math.cos(delta)

        sin_delta = math.sin(delta)

        for leg in self.legs:

            dx = float(leg.position.x) - old_x

            dy = float(leg.position.y) - old_y

            rotated_x = (
                cos_delta * dx - sin_delta * dy
            )

            rotated_y = (
                sin_delta * dx + cos_delta * dy
            )

            leg.position = (
                x + rotated_x,
                y + rotated_y
            )

            leg.angle = float(leg.angle) + delta

            leg.linearVelocity = (
                vx - omega * rotated_y,
                vy + omega * rotated_x
            )

            leg.angularVelocity = omega

            leg.ground_contact = False

            leg.awake = True

        # Main lander
        self.lander.position = (x, y)

        self.lander.angle = angle

        self.lander.linearVelocity = (vx, vy)

        self.lander.angularVelocity = omega

        self.lander.awake = True

        # Native reward shaping
        self.prev_shaping = (

            -100.0 * math.sqrt(
                x_obs**2 + y_obs**2
            )

            -100.0 * math.sqrt(
                vx_obs**2 + vy_obs**2
            )

            -100.0 * abs(angle)

        )

        observation = np.array([

            x_obs,
            y_obs,

            vx_obs,
            vy_obs,

            angle,

            20.0 * omega / FPS,

            0.0,
            0.0

        ], dtype=np.float32)

        info = dict(info)

        info["initial_state"] = {

            "x": float(x_obs),

            "y": float(y_obs),

            "vx": float(vx_obs),

            "vy": float(vy_obs),

            "angle": float(angle),

            "angular_velocity": float(
                20.0 * omega / FPS
            )

        }

        if self.render_mode == "human":
            self.render()

        return observation, info