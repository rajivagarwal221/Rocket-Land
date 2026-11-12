import gymnasium as gym
import numpy as np

from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H
)


class TargetLandingWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=50.0,
        off_target_penalty=20.0,
        timeout_penalty=30.0,
        failed_landing_penalty=100.0,
        settle_frames=15,
        target_limit=0.15,
        # --- V18 precision knobs (backward-compatible: defaults preserve V17) ---
        commit_penalty_after=0,           # step index after which step_penalty triples
        commit_penalty_height=0.3,        # ...when env obs y (altitude) exceeds this
        commit_penalty_scale=3.0,         # multiplier applied to step_penalty when active
        x_shaping_coef=0.0,
        x_shaping_fade_below=0.0,
        angular_stability_bonus=0.0,
        angular_stability_angle=0.15,
        angular_stability_omega=0.05,
        # --- V24 Gaudet-grounded knobs (backward-compatible: defaults preserve V23) ---
        gamma_shaping=None,
        fuel_penalty_coef=0.0,
        fuel_lateral_weight=0.5,
        glideslope_penalty_coef=0.0,
        glideslope_ratio_threshold=2.0,
        glideslope_altitude=0.15,
        hard_attitude_limit=None,
    ):

        super().__init__(env)

        self.gamma = gamma
        self.target_limit = target_limit

        self.step_penalty = step_penalty
        self.target_bonus = target_bonus
        self.off_target_penalty = off_target_penalty
        self.timeout_penalty = timeout_penalty
        self.failed_landing_penalty = failed_landing_penalty

        self.settle_frames = settle_frames
        self.stable_contact_steps = 0

        self.prev_potential = 0.0

        self.half_width = VIEWPORT_W / SCALE / 2
        self.half_height = VIEWPORT_H / SCALE / 2

        # V18 extras
        self.commit_penalty_after = int(commit_penalty_after)
        self.commit_penalty_height = float(commit_penalty_height)
        self.commit_penalty_scale = float(commit_penalty_scale)
        self.x_shaping_coef = float(x_shaping_coef)
        self.steps_in_episode = 0

        # V22 stability knobs
        self.x_shaping_fade_below = float(x_shaping_fade_below)
        self.angular_stability_bonus = float(angular_stability_bonus)
        self.angular_stability_angle = float(angular_stability_angle)
        self.angular_stability_omega = float(angular_stability_omega)

        # V24 Gaudet-grounded knobs
        self.gamma_shaping = (
            float(gamma_shaping) if gamma_shaping is not None else float(gamma)
        )
        self.fuel_penalty_coef = float(fuel_penalty_coef)
        self.fuel_lateral_weight = float(fuel_lateral_weight)
        self.glideslope_penalty_coef = float(glideslope_penalty_coef)
        self.glideslope_ratio_threshold = float(glideslope_ratio_threshold)
        self.glideslope_altitude = float(glideslope_altitude)
        self.hard_attitude_limit = (
            float(hard_attitude_limit)
            if hard_attitude_limit is not None
            else None
        )

        self.fuel_used_episode = 0.0
        self.hard_attitude_violation = False
        self._last_action = None

    # =====================================
    # PREDICT LANDING POSITION - V4
    # =====================================

    def predict_landing_position(self, observation):

        x = float(observation[0])
        y = float(observation[1])

        vx = float(observation[2])
        vy = float(observation[3])

        physical_vy = (
            vy * FPS / self.half_height
        )

        height = max(
            0.0,
            y * self.half_height
        )

        if vy < -0.05 and physical_vy < 0:

            time_to_ground = (
                height / max(-physical_vy, 0.10)
            )

        else:

            time_to_ground = 0.5

        time_to_ground = float(
            np.clip(time_to_ground, 0.05, 1.5)
        )

        physical_vx = (
            vx * FPS / self.half_width
        )

        predicted_x = (
            x
            + physical_vx * time_to_ground
            / self.half_width
        )

        return float(
            np.clip(predicted_x, -1.5, 1.5)
        )

    # =====================================
    # CHECK STABLE LANDING
    # =====================================

    def is_stable_touchdown(self, observation):

        both_legs = (
            observation[6] > 0.5
            and observation[7] > 0.5
        )

        vx = abs(float(observation[2]))
        vy = abs(float(observation[3]))

        angle = abs(float(observation[4]))
        omega = abs(float(observation[5]))

        return bool(
            both_legs
            and vx < 0.025
            and vy < 0.025
            and angle < 0.20
            and omega < 0.08
        )

    # =====================================
    # UNIFIED POTENTIAL - V9
    # =====================================

    def potential(self, observation):

        predicted_x = self.predict_landing_position(
            observation
        )

        vx = abs(float(observation[2]))
        vy = max(0.0, -float(observation[3]))

        angle = abs(float(observation[4]))
        omega = abs(float(observation[5]))

        height = max(
            0.0,
            float(observation[1])
        )

        ground_proximity = 1.0 - np.clip(
            height / 0.75,
            0.0,
            1.0
        )

        position_cost = 35.0 * abs(predicted_x)

        # Preserve V7 horizontal braking.
        horizontal_cost = (
            6.0 + 6.0 * ground_proximity
        ) * vx

        vertical_cost = (
            5.0 * ground_proximity * vy
        )

        rotation_cost = 3.0 * angle

        angular_cost = 1.0 * omega

        # V10: discourage one-leg contact and reward balanced touchdown.
        left_contact = bool(
            observation[6] > 0.5
        )

        right_contact = bool(
            observation[7] > 0.5
        )

        if left_contact and right_contact:

            leg_contact_cost = 0.0

        elif left_contact or right_contact:

            leg_contact_cost = (
                9.0 * ground_proximity
            )

        else:

            leg_contact_cost = (
                6.0 * ground_proximity
            )

        return float(-(
            position_cost
            + horizontal_cost
            + vertical_cost
            + rotation_cost
            + angular_cost
            + leg_contact_cost
        ))    

    # =====================================
    # RESET
    # =====================================

    def reset(self, **kwargs):

        observation, info = self.env.reset(
            **kwargs
        )

        self.stable_contact_steps = 0
        self.steps_in_episode = 0
        self.fuel_used_episode = 0.0
        self.hard_attitude_violation = False
        self._last_action = None

        self.prev_potential = self.potential(
            observation
        )

        return observation, info

    # =====================================
    # STEP
    # =====================================

    def step(self, action):

        import numpy as _np_v24
        self._last_action = _np_v24.asarray(action, dtype=float).reshape(-1)

        (
            observation,
            raw_reward,
            terminated,
            truncated,
            info
        ) = self.env.step(action)

        raw_reward = float(raw_reward)

        self.steps_in_episode += 1

        # ---------------------------------
        # OBSERVATION
        # ---------------------------------

        x = float(observation[0])
        y = float(observation[1])

        both_legs = bool(
            observation[6] > 0.5
            and observation[7] > 0.5
        )

        inside_target = bool(
            -self.target_limit < x < self.target_limit
        )

        # ---------------------------------
        # STABLE TOUCHDOWN COUNTER
        # ---------------------------------

        if self.is_stable_touchdown(observation):

            self.stable_contact_steps += 1

        else:

            self.stable_contact_steps = 0

        stable_touchdown = (
            self.stable_contact_steps >= self.settle_frames
        )

        custom_landing = bool(
            stable_touchdown and not terminated
        )

        # ---------------------------------
        # DETERMINE LANDING QUALITY
        # ---------------------------------

        native_rest = bool(
            terminated and raw_reward > 0
        )

        successful_landing = bool(
            custom_landing
            or (native_rest and both_legs)
        )

        failed_rest = bool(
            native_rest
            and not both_legs
        )

        if custom_landing:

            terminated = True
            truncated = False

        # ---------------------------------
        # EFFECTIVE BASE REWARD
        # ---------------------------------

        if failed_rest:

            base_reward = -self.failed_landing_penalty

            landing_status = "FAILED_NO_BOTH_LEGS"

        elif custom_landing:

            base_reward = 100.0

            landing_status = "STABLE_LANDING"

        elif successful_landing:

            base_reward = raw_reward

            landing_status = "LANDED_WITH_BOTH_LEGS"

        elif terminated:

            base_reward = raw_reward

            landing_status = "CRASH"

        else:

            base_reward = raw_reward

            landing_status = "IN_FLIGHT"

        # ---------------------------------
        # PRECISE LANDING
        # ---------------------------------

        precise_landing = bool(
            successful_landing
            and inside_target
            and both_legs
        )

        # ---------------------------------
        # TERMINAL REWARD
        # ---------------------------------

        terminal_bonus = 0.0

        if successful_landing:

            if precise_landing:

                terminal_bonus = self.target_bonus

            else:

                terminal_bonus = -self.off_target_penalty

            # V22: reward upright, low-rotation two-leg touchdowns so the
            # policy learns to arrive stable, not just precise.
            if self.angular_stability_bonus > 0.0:

                touchdown_angle = abs(float(observation[4]))
                touchdown_omega = abs(float(observation[5]))

                if (
                    both_legs
                    and touchdown_angle < self.angular_stability_angle
                    and touchdown_omega < self.angular_stability_omega
                ):

                    terminal_bonus += self.angular_stability_bonus

        # ---------------------------------
        # POTENTIAL SHAPING
        # ---------------------------------

        if terminated:

            next_potential = 0.0

        else:

            next_potential = self.potential(
                observation
            )

        # V24: Ng 1999 potential-based shaping with Gaudet 2020's separate discount
        # for the shaping term. F = gamma_shaping * Phi(s') - Phi(s) preserves the
        # Ng invariance theorem for the shaping component.
        shaping_reward = (
            self.gamma_shaping * next_potential
            - self.prev_potential
        )

        self.prev_potential = next_potential

        # ---------------------------------
        # TIMEOUT PENALTY
        # ---------------------------------

        timeout_reward = 0.0

        if truncated and not terminated:

            timeout_reward = -self.timeout_penalty

        # ---------------------------------
        # STEP PENALTY (with optional commit penalty)
        # ---------------------------------

        effective_step_penalty = self.step_penalty
        if (
            self.commit_penalty_after > 0
            and self.steps_in_episode > self.commit_penalty_after
            and y > self.commit_penalty_height
            and not terminated
        ):
            effective_step_penalty = (
                self.step_penalty * self.commit_penalty_scale
            )

        time_reward = -effective_step_penalty

        # ---------------------------------
        # OPTIONAL x^2 LATERAL PRESSURE
        # ---------------------------------

        x_shaping_reward = 0.0
        if self.x_shaping_coef > 0.0 and not terminated:
            ground_proximity = 1.0 - np.clip(
                max(0.0, y) / 0.75, 0.0, 1.0
            )
            # V22: fade x-shaping out very close to the ground so the policy
            # commits to a vertical touchdown instead of chasing x in the
            # last few frames (which was tipping the lander onto one leg).
            if self.x_shaping_fade_below > 0.0:
                altitude_gate = float(np.clip(
                    max(0.0, y) / self.x_shaping_fade_below, 0.0, 1.0
                ))
                ground_proximity = ground_proximity * altitude_gate
            x_shaping_reward = -(
                self.x_shaping_coef * (x ** 2) * ground_proximity
            )

        # ---------------------------------
        # V24 FUEL PENALTY (Gaudet 2020)
        # Linear cost on commanded thrust magnitude per step.
        # ---------------------------------

        fuel_reward = 0.0
        if self.fuel_penalty_coef > 0.0 and self._last_action is not None:
            a = self._last_action
            main_cost = abs(float(a[0])) if a.size >= 1 else 0.0
            side_cost = abs(float(a[1])) if a.size >= 2 else 0.0
            step_fuel = (
                main_cost + self.fuel_lateral_weight * side_cost
            )
            self.fuel_used_episode += step_fuel
            fuel_reward = -self.fuel_penalty_coef * step_fuel

        # ---------------------------------
        # V24 GLIDESLOPE PENALTY (Gaudet 2020)
        # Penalises |vx|/|vy| ratio during the final descent metres.
        # ---------------------------------

        glideslope_reward = 0.0
        if (
            self.glideslope_penalty_coef > 0.0
            and not terminated
            and 0.0 < y < self.glideslope_altitude
        ):
            vx_abs = abs(float(observation[2]))
            vy_abs = max(abs(float(observation[3])), 0.01)
            ratio = vx_abs / vy_abs
            if ratio > self.glideslope_ratio_threshold:
                glideslope_reward = -(
                    self.glideslope_penalty_coef
                    * (ratio - self.glideslope_ratio_threshold)
                )

        # ---------------------------------
        # V24 HARD ATTITUDE CONSTRAINT (Gaudet 2020)
        # |angle| > limit terminates the episode as a crash.
        # ---------------------------------

        if (
            self.hard_attitude_limit is not None
            and abs(float(observation[4])) > self.hard_attitude_limit
            and not terminated
        ):
            terminated = True
            truncated = False
            self.hard_attitude_violation = True
            base_reward = -self.failed_landing_penalty
            terminal_bonus = 0.0
            landing_status = "HARD_ATTITUDE_VIOLATION"

        # ---------------------------------
        # FINAL REWARD
        # ---------------------------------

        final_reward = (
            base_reward
            + terminal_bonus
            + shaping_reward
            + timeout_reward
            + time_reward
            + x_shaping_reward
            + fuel_reward
            + glideslope_reward
        )

        # ---------------------------------
        # INFORMATION
        # ---------------------------------

        info = dict(info)

        info["raw_environment_reward"] = raw_reward

        info["base_reward"] = float(base_reward)

        info["landing_status"] = landing_status

        info["failed_rest"] = failed_rest

        info["successful_landing"] = successful_landing

        info["precise_landing"] = precise_landing

        info["target_landing"] = precise_landing

        info["custom_stable_landing"] = custom_landing

        info["stable_contact_steps"] = self.stable_contact_steps

        info["shaping_reward"] = float(shaping_reward)

        info["terminal_bonus"] = float(terminal_bonus)

        info["timeout_reward"] = float(timeout_reward)

        info["fuel_used_episode"] = float(self.fuel_used_episode)
        info["fuel_reward"] = float(fuel_reward)
        info["glideslope_reward"] = float(glideslope_reward)
        info["hard_attitude_violation"] = bool(self.hard_attitude_violation)
        info["active_target_limit"] = float(self.target_limit)

        return (
            observation,
            float(final_reward),
            terminated,
            truncated,
            info
        )