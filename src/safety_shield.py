"""SafetyShield: deterministic safety filter near the ground.

Grounded in Alshiekh et al. 2018 (Safe RL via Shielding). The shield enforces
a safety specification that in LTL reads:

    G( (y < y_trigger & vy < vy_trigger & |angle| < angle_limit)
       -> main_thrust >= main_thrust_floor )

i.e. ALWAYS, if the lander is below the altitude trigger and descending faster
than the velocity trigger with a recoverable attitude, THEN the main engine
must be at full thrust. Horizontal control stays with the policy.

Alshiekh 2018 distinguishes two shield placements:

  POST-POSED (default, Alshiekh Fig. 2):
      The shield monitors the action the policy chose and corrects it only
      if the current state would otherwise violate the spec. The policy is
      free to issue any action; the environment only sees shielded actions.
      This is what V20-V23 used.

  PREEMPTIVE (Alshiekh Fig. 1):
      Before the policy chooses, the shield restricts the admissible action
      set. The policy learns within the restricted set, which can give
      stronger convergence guarantees (Alshiekh Sec 4). V24 implementation:
      when the trigger state holds, the shield clamps the action LOW BOUND
      on the main thrust to main_thrust_floor and lets the policy choose
      within [floor, +1]. The policy still sees its own action in
      info['commanded_action']; the environment sees the clamped one.

Both modes enforce the same LTL spec. Post-posed leaves policy exploration
free; preemptive reshapes the action space but is friendlier to the
learner (no "silent override" surprise during training).

The shield is a hand-written classical rocket-guidance floor; it is NOT
a learned behaviour. Alshiekh-style formal synthesis from the LTL spec
plus an abstract Box2D dynamics model is a future-work item (see v24
research-paper notes, Novelty 1).

Place this wrapper BETWEEN the base env and TimeLimit/TargetLandingWrapper.
Continuous action space only (Box(-1, 1, (2,))).
"""
import gymnasium as gym
import numpy as np


class SafetyShield(gym.Wrapper):

    def __init__(
        self,
        env,
        y_trigger=0.45,
        vy_trigger=-0.55,
        main_thrust=1.0,
        angle_limit=0.6,
        disable_on_contact=True,
        mode="post_posed",       # V24 (Alshiekh 2018): "post_posed" or "preemptive"
        main_thrust_floor=None,  # V24: preemptive-mode floor; defaults to main_thrust
    ):
        super().__init__(env)
        self.y_trigger = float(y_trigger)
        self.vy_trigger = float(vy_trigger)
        self.main_thrust = float(main_thrust)
        self.angle_limit = float(angle_limit)
        self.disable_on_contact = bool(disable_on_contact)
        if mode not in ("post_posed", "preemptive"):
            raise ValueError(
                f"mode must be 'post_posed' or 'preemptive', got {mode!r}"
            )
        self.mode = mode
        self.main_thrust_floor = (
            float(main_thrust_floor)
            if main_thrust_floor is not None
            else float(main_thrust)
        )
        self._last_obs = None
        self.shield_fires_this_episode = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.shield_fires_this_episode = 0
        return obs, info

    def _should_fire(self, obs):
        if obs is None:
            return False
        y = float(obs[1])
        vy = float(obs[3])
        angle = float(obs[4])
        left_leg = bool(obs[6] > 0.5)
        right_leg = bool(obs[7] > 0.5)

        if self.disable_on_contact and (left_leg or right_leg):
            return False
        if abs(angle) > self.angle_limit:
            return False  # too tilted; firing main would push sideways
        return y < self.y_trigger and vy < self.vy_trigger

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32).copy()
        effective = commanded.copy()

        fire = self._should_fire(self._last_obs)
        if fire:
            if self.mode == "post_posed":
                effective[0] = self.main_thrust
            else:  # preemptive
                effective[0] = max(float(effective[0]), self.main_thrust_floor)
            self.shield_fires_this_episode += 1

        obs, reward, terminated, truncated, info = self.env.step(effective)
        self._last_obs = obs

        info = dict(info)
        info["shield_fires"] = self.shield_fires_this_episode
        info["shield_mode"] = self.mode
        info["shield_fired_this_step"] = bool(fire)
        info["commanded_main_thrust"] = float(commanded[0])
        info["effective_main_thrust"] = float(effective[0])
        return obs, reward, terminated, truncated, info
