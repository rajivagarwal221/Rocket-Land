"""Automated difficulty curriculum, grounded in Narvekar 2020.

Narvekar et al. 2020 ("Curriculum Learning for Reinforcement Learning
Domains: A Framework and Survey", JMLR) classifies curriculum methods along
three axes: task generation, task sequencing, transfer. Their Section 6.1
calls out "fully automated task creation" and 6.6 "principles for curriculum
design" as open problems. Section 4.2 describes LEARNING-PROGRESS-based
curriculum selection (Matiisen et al. 2017; teacher-student).

This callback implements a lightweight automated curriculum over ONE
difficulty axis of our rocket-landing task: the weak-gravity sampling
probability.

RATIONALE
The Rocket-Land v20-v23 analysis showed that the weak-gravity slice (g
weaker than -6 m/s^2) is persistently the hardest. V22 held fixed
weak_gravity_prob=0.35; V23 bumped to 0.55 and achieved uniform ~97%
precision across slices. V24 replaces the fixed bump with an automated
adaptation:

  * If recent precision is HIGH on the current mix (agent is winning),
    increase weak_gravity_prob -> the curriculum pushes toward harder
    episodes (Matiisen 2017's "zones of proximal development" idea).
  * If recent precision is LOW (agent is losing), decrease
    weak_gravity_prob -> back off to retain learnable episodes.

This is a Narvekar-Section-4.2 "learning progress" scheduler, operating
over a single scalar difficulty dial. It is NOT full automated task
generation (Narvekar Sec 4.1) -- it selects within a fixed task family.

API
The callback finds the RandomStart/PlanetLander-style env in the training
env chain (any env with a `_weak_gravity_prob` attribute) and mutates it.

For subprocess-parallel envs, pass the reference-callable `get_envs` which
returns a list of underlying train envs; the callback updates all of them.
"""
import collections
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class AutoCurriculumCallback(BaseCallback):
    """Narvekar 2020 learning-progress curriculum over weak_gravity_prob.

    Measures rolling PRECISE-episode rate from Monitor's episode-info buffer
    (ep_info_buffer) which SB3 maintains automatically during training. When
    the rate crosses `high_threshold`, bumps weak_gravity_prob up by `step_up`.
    When it drops below `low_threshold`, backs off by `step_down`. Bounded by
    [min_prob, max_prob].

    The success signal is extracted from info['precise_landing'] written by
    TargetLandingWrapper. SB3 stores info-dict keys on episode end if they
    are registered via Monitor's info_keywords.
    """

    def __init__(
        self,
        train_env,
        get_envs=None,
        initial_prob=0.55,
        min_prob=0.30,
        max_prob=0.75,
        step_up=0.05,
        step_down=0.05,
        high_threshold=0.95,
        low_threshold=0.80,
        rolling_window=64,
        adapt_every_n_steps=10_000,
        verbose=1,
    ):

        super().__init__(verbose=verbose)
        self._train_env = train_env
        self._get_envs = get_envs
        self._initial_prob = float(initial_prob)
        self._min_prob = float(min_prob)
        self._max_prob = float(max_prob)
        self._step_up = float(step_up)
        self._step_down = float(step_down)
        self._high = float(high_threshold)
        self._low = float(low_threshold)
        self._window = int(rolling_window)
        self._adapt_every = int(adapt_every_n_steps)

        self._current_prob = float(initial_prob)
        self._last_adapt_step = 0
        self._precise_history: collections.deque = collections.deque(
            maxlen=self._window
        )

    def _find_base_envs(self):
        """Walk the env chain to find envs with a _weak_gravity_prob attr."""

        envs = []
        candidates = []

        if self._get_envs is not None:
            candidates.extend(self._get_envs())
        else:
            candidates.append(self._train_env)

        for e in candidates:
            cur = e
            while cur is not None:
                if hasattr(cur, "_weak_gravity_prob"):
                    envs.append(cur)
                    break
                cur = getattr(cur, "env", None)

        return envs

    def _set_prob(self, prob):

        prob = float(np.clip(prob, self._min_prob, self._max_prob))
        self._current_prob = prob
        for e in self._find_base_envs():
            e._weak_gravity_prob = prob

    def _on_training_start(self):

        self._set_prob(self._initial_prob)
        self._last_adapt_step = self.model.num_timesteps

    def _on_step(self):

        # Pull episode info updated by Monitor
        try:
            infos = self.model.ep_info_buffer
        except AttributeError:
            infos = None

        # Harvest precise flag from latest info dicts (SB3 emits them on
        # episode end). Monitor copies selected info_keywords into the
        # episode summary; we check both standard and info keys.
        # Fallback: check the last-emitted info from the logger if available.
        latest = getattr(self.locals, "get", lambda *a, **k: None)("infos")
        if latest is None:
            latest = self.locals.get("infos", None)

        if latest is not None:
            for info in latest:
                if not isinstance(info, dict):
                    continue
                # Monitor's "episode" key is only present at episode-end.
                ep = info.get("episode")
                if ep is None:
                    continue
                precise = info.get("precise_landing")
                if precise is not None:
                    self._precise_history.append(1.0 if precise else 0.0)

        # Adapt the curriculum every `adapt_every` steps
        if (
            self.model.num_timesteps - self._last_adapt_step
            >= self._adapt_every
        ):
            self._last_adapt_step = self.model.num_timesteps
            if len(self._precise_history) >= max(16, self._window // 2):
                rate = float(np.mean(self._precise_history))
                old_prob = self._current_prob
                if rate >= self._high:
                    self._set_prob(old_prob + self._step_up)
                elif rate <= self._low:
                    self._set_prob(old_prob - self._step_down)
                if abs(self._current_prob - old_prob) > 1e-6:
                    print(
                        f"[auto-curriculum] t={self.model.num_timesteps:,} "
                        f"recent_precise={rate:.3f} "
                        f"weak_gravity_prob {old_prob:.2f} -> "
                        f"{self._current_prob:.2f}",
                        flush=True,
                    )

        return True
