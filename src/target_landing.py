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
        settle_frames=15
    ):

        super().__init__(env)

        self.gamma = gamma
        self.target_limit = 0.15

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

        self.prev_potential = self.potential(
            observation
        )

        return observation, info

    # =====================================
    # STEP
    # =====================================

    def step(self, action):

        (
            observation,
            raw_reward,
            terminated,
            truncated,
            info
        ) = self.env.step(action)

        raw_reward = float(raw_reward)

        # ---------------------------------
        # OBSERVATION
        # ---------------------------------

        x = float(observation[0])

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

        # ---------------------------------
        # POTENTIAL SHAPING
        # ---------------------------------

        if terminated:

            next_potential = 0.0

        else:

            next_potential = self.potential(
                observation
            )

        shaping_reward = (
            self.gamma * next_potential
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
        # STEP PENALTY
        # ---------------------------------

        time_reward = -self.step_penalty

        # ---------------------------------
        # FINAL REWARD
        # ---------------------------------

        final_reward = (
            base_reward
            + terminal_bonus
            + shaping_reward
            + timeout_reward
            + time_reward
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

        return (
            observation,
            float(final_reward),
            terminated,
            truncated,
            info
        )