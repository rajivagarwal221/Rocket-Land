"""Bounded residual-learning interface; wrap AFTER observation delay/noise.

The 27-value observation includes sensor state (16), the next baseline action
(4), controller integrals (3) and phase one-hot (4). This exposes controller
memory without leaking simulator truth. Supply zero actions to audit baseline.
This interface is ready for experiments; a trained V29 policy does not exist.
"""
import gymnasium as gym
import numpy as np

from sim3d.guidance import LandingController, residual_action

PHASES = ('ACQUIRE', 'DESCEND', 'SETTLE', 'ABORT')


class ResidualLandingEnv(gym.Wrapper):
    def __init__(self, env, params=None, authority=(0.1, 0.15, 0.15, 0.1)):
        super().__init__(env)
        if env.observation_space.shape != (16,) or env.action_space.shape != (4,):
            raise ValueError('Wrap the unstacked 16-observation, 4-control mission env')
        self.authority = np.asarray(authority, dtype=float)
        residual_action(np.zeros(4), np.zeros(4), self.authority)
        # These are declared nominal design values, never randomized episode truth.
        nominal = params if params is not None else env.unwrapped.params
        self.controller = LandingController(params=nominal, dt=1/env.unwrapped.cfg.control_hz)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (27,), dtype=np.float32)
        self._baseline = None

    def _prepare(self, obs):
        self._baseline = self.controller(obs)
        integrals = np.r_[self.controller._integral, self.controller._vz_integral]
        phase = [float(self.controller.phase == name) for name in PHASES]
        return np.r_[obs, self._baseline, integrals, phase].astype(np.float32)

    def reset(self, **kwargs):
        self.controller.reset()
        obs, info = self.env.reset(**kwargs)
        return self._prepare(obs), info

    def step(self, residual):
        if self._baseline is None:
            raise RuntimeError('Call reset before step')
        action = residual_action(self._baseline, residual, self.authority)
        obs, reward, te, tr, info = self.env.step(action)
        info = dict(info, baseline_action=self._baseline.copy(), applied_action=action.copy())
        if te or tr:
            # Do not advance controller memory after the mission has ended.
            phase = [float(self.controller.phase == name) for name in PHASES]
            out = np.r_[obs, self._baseline, self.controller._integral,
                        self.controller._vz_integral, phase].astype(np.float32)
            self._baseline = None
        else:
            out = self._prepare(obs)
        return out, reward, te, tr, info
