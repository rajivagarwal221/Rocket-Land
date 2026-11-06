import gymnasium as gym


# EXPERIMENT 1: Reward Hacking
class HoverBonusWrapper(gym.Wrapper):

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)

        x = observation[0]
        y = observation[1]
        vx = observation[2]
        vy = observation[3]
        angle = observation[4]

        left_leg = observation[6]
        right_leg = observation[7]

        stable = (
            abs(angle) < 0.1
            and abs(vx) < 0.1
            and abs(vy) < 0.1
            and y > 0.2
            and left_leg == 0
            and right_leg == 0
        )

        extra_reward = 0

        if stable and not terminated and not truncated:
            extra_reward = self.bonus

        new_reward = reward + extra_reward

        info = dict(info)
        info["original_reward"] = float(reward)
        info["hover_bonus"] = float(extra_reward)

        return observation, new_reward, terminated, truncated, info


# EXPERIMENT 2: Improved Reward
class ImprovedRewardWrapper(gym.Wrapper):

    def __init__(self, env, step_penalty=0.05):
        super().__init__(env)
        self.step_penalty = step_penalty

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)

        time_penalty = -self.step_penalty

        new_reward = reward + time_penalty

        info = dict(info)
        info["original_reward"] = float(reward)
        info["time_penalty"] = float(time_penalty)

        return observation, new_reward, terminated, truncated, info