import gymnasium as gym
import numpy as np

from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
)


class TargetLandingWrapperV11(gym.Wrapper):
    """V11 reward shaping targeting the three V9 failure modes:

    - SAFE_OUTSIDE (17%): steeper lateral cost near the ground
    - REST_NO_BOTH_LEGS (8%): descent-commitment penalty + first-touchdown
      bonus + harder penalty on rest-without-both-legs
    - CRASH (1%): unchanged shaping; mitigated by harder-start curriculum
    """

    def __init__(
        self,
        env,
        gamma=0.99,
        step_penalty=0.01,
        target_bonus=70.0,
        off_target_penalty=25.0,
        timeout_penalty=40.0,
        failed_landing_penalty=150.0,
        both_leg_bonus=25.0,
        settle_frames=15,
    ):
        super().__init__(env)

        self.gamma = gamma
        self.target_limit = 0.15

        self.step_penalty = step_penalty
        self.target_bonus = target_bonus
        self.off_target_penalty = off_target_penalty
        self.timeout_penalty = timeout_penalty
        self.failed_landing_penalty = failed_landing_penalty
        self.both_leg_bonus = both_leg_bonus

        self.settle_frames = settle_frames
        self.stable_contact_steps = 0
        self.both_leg_bonus_given = False
        self.low_hover_steps = 0

        self.prev_potential = 0.0

        self.half_width = VIEWPORT_W / SCALE / 2
        self.half_height = VIEWPORT_H / SCALE / 2

    # ------------------------------------------------------------------
    # LANDING PREDICTION (unchanged from V9)
    # ------------------------------------------------------------------
    def predict_landing_position(self, obs):
        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])

        physical_vy = vy * FPS / self.half_height
        height = max(0.0, y * self.half_height)

        if vy < -0.05 and physical_vy < 0:
            time_to_ground = height / max(-physical_vy, 0.10)
        else:
            time_to_ground = 0.5

        time_to_ground = float(np.clip(time_to_ground, 0.05, 1.5))
        physical_vx = vx * FPS / self.half_width
        predicted_x = x + physical_vx * time_to_ground / self.half_width
        return float(np.clip(predicted_x, -1.5, 1.5))

    def is_stable_touchdown(self, obs):
        both_legs = obs[6] > 0.5 and obs[7] > 0.5
        vx = abs(float(obs[2]))
        vy = abs(float(obs[3]))
        angle = abs(float(obs[4]))
        omega = abs(float(obs[5]))
        return bool(
            both_legs
            and vx < 0.025
            and vy < 0.025
            and angle < 0.20
            and omega < 0.08
        )

    # ------------------------------------------------------------------
    # POTENTIAL (V11: steeper lateral near ground)
    # ------------------------------------------------------------------
    def potential(self, obs):
        predicted_x = self.predict_landing_position(obs)
        vx = abs(float(obs[2]))
        vy = max(0.0, -float(obs[3]))
        angle = abs(float(obs[4]))
        omega = abs(float(obs[5]))
        height = max(0.0, float(obs[1]))

        ground_proximity = 1.0 - np.clip(height / 0.75, 0.0, 1.0)

        # V11: lateral cost grows sharply near ground.
        position_cost = (35.0 + 55.0 * ground_proximity) * abs(predicted_x)

        horizontal_cost = (6.0 + 8.0 * ground_proximity) * vx
        vertical_cost = 5.0 * ground_proximity * vy
        rotation_cost = 3.0 * angle
        angular_cost = 1.0 * omega

        left_contact = bool(obs[6] > 0.5)
        right_contact = bool(obs[7] > 0.5)

        if left_contact and right_contact:
            leg_contact_cost = 0.0
        elif left_contact or right_contact:
            leg_contact_cost = 9.0 * ground_proximity
        else:
            leg_contact_cost = 6.0 * ground_proximity

        return float(
            -(
                position_cost
                + horizontal_cost
                + vertical_cost
                + rotation_cost
                + angular_cost
                + leg_contact_cost
            )
        )

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stable_contact_steps = 0
        self.both_leg_bonus_given = False
        self.low_hover_steps = 0
        self.prev_potential = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(raw_reward)

        x = float(obs[0])
        height = max(0.0, float(obs[1]))
        vy = float(obs[3])

        left_contact = bool(obs[6] > 0.5)
        right_contact = bool(obs[7] > 0.5)
        both_legs = left_contact and right_contact

        inside_target = bool(-self.target_limit < x < self.target_limit)

        # Stable-touchdown counter.
        if self.is_stable_touchdown(obs):
            self.stable_contact_steps += 1
        else:
            self.stable_contact_steps = 0

        stable_touchdown = self.stable_contact_steps >= self.settle_frames
        custom_landing = bool(stable_touchdown and not terminated)

        native_rest = bool(terminated and raw_reward > 0)
        successful_landing = bool(custom_landing or (native_rest and both_legs))
        failed_rest = bool(terminated and raw_reward > 0 and not both_legs)

        if custom_landing:
            terminated = True
            truncated = False

        # Base reward (V11: sharper failed-rest penalty).
        if failed_rest:
            base_reward = -self.failed_landing_penalty
            landing_status = "FAILED_NO_BOTH_LEGS"
        elif custom_landing:
            base_reward = 120.0
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

        precise_landing = bool(
            successful_landing and inside_target and both_legs
        )

        # Terminal shaping.
        terminal_bonus = 0.0
        if successful_landing:
            terminal_bonus = (
                self.target_bonus
                if precise_landing
                else -self.off_target_penalty
            )

        # V11: one-shot bonus the first time both legs touch.
        touchdown_bonus = 0.0
        if both_legs and not self.both_leg_bonus_given:
            touchdown_bonus = self.both_leg_bonus
            self.both_leg_bonus_given = True

        # V11: descent-commitment penalty.
        # If we are low (<0.25), neither leg touching, and vy near zero
        # (not actively descending), accumulate a growing penalty so the
        # policy cannot park just above the pad.
        hover_penalty = 0.0
        if (
            not terminated
            and not (left_contact or right_contact)
            and height < 0.30
            and abs(vy) < 0.10
        ):
            self.low_hover_steps += 1
            hover_penalty = -0.05 * self.low_hover_steps
        else:
            self.low_hover_steps = 0

        # Potential-based shaping (gamma-consistent).
        next_potential = 0.0 if terminated else self.potential(obs)
        shaping_reward = self.gamma * next_potential - self.prev_potential
        self.prev_potential = next_potential

        # V11: timeout penalty scales with how low the lander was.
        timeout_reward = 0.0
        if truncated and not terminated:
            altitude_factor = 1.0 + 2.0 * (1.0 - min(1.0, height / 0.5))
            timeout_reward = -self.timeout_penalty * altitude_factor

        time_reward = -self.step_penalty

        final_reward = (
            base_reward
            + terminal_bonus
            + touchdown_bonus
            + hover_penalty
            + shaping_reward
            + timeout_reward
            + time_reward
        )

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
        info["touchdown_bonus"] = float(touchdown_bonus)
        info["hover_penalty"] = float(hover_penalty)
        info["timeout_reward"] = float(timeout_reward)

        return obs, float(final_reward), terminated, truncated, info
