import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sim3d.mission import MissionEnv, MissionConfig
from sim3d.guidance import LandingController
from sim3d.rocket_env import Rocket3DEnv
from sim3d.rocket_env import RocketParams


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.env = MissionEnv(difficulty=0)
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def test_seed_reproducibility(self):
        a, _ = self.env.reset(seed=8)
        b, _ = self.env.reset(seed=8)
        np.testing.assert_array_equal(a, b)

    def test_unsettled_contact_cannot_succeed(self):
        self.assertEqual(self.env._contact_deadline_outcome(0.0), 'UNSETTLED')

    def test_settle_requires_upright_and_three_legs(self):
        k = self.env.kinematics()
        k.v_world[:] = 0
        k.omega[:] = 0
        k.tilt = 0
        self.assertFalse(self.env._is_settled(k, 2))
        self.assertTrue(self.env._is_settled(k, 3))
        k.tilt = np.deg2rad(11)
        self.assertFalse(self.env._is_settled(k, 4))

    def test_deadline_is_terminal_failure(self):
        self.env.cfg.max_steps = 1
        _, _, terminated, truncated, info = self.env.step([-1, 0, 0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'TIMEOUT')

    def test_rest_on_pad_succeeds_after_dwell(self):
        e = self.env
        e.data.qpos[:3] = [0, 0, e._rest_height + 0.002]
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        mujoco.mj_forward(e.model, e.data)
        for step in range(100):
            _, _, te, tr, info = e.step([-1, 0, 0, 0])
            if te or tr:
                break
        self.assertEqual(info['outcome'], 'PRECISE')
        self.assertGreaterEqual(step + 1, 25)

    def test_zero_residual_reproduces_baseline(self):
        from sim3d.guidance import residual_action
        base = np.array([0.2, -0.3, 0.4, 0.0])
        np.testing.assert_allclose(residual_action(base, np.zeros(4)), base)
        self.assertTrue(np.all(np.abs(residual_action(base, np.ones(4)*99)) <= 1))

    def test_nonterminal_reward_is_negative(self):
        _, reward, te, tr, _ = self.env.step([0, 0, 0, 0])
        self.assertFalse(te or tr)
        self.assertLess(reward, 0)

    def test_off_pad_rest_is_not_precise(self):
        e = self.env
        e.data.qpos[:3] = [1, 0, e._rest_height + 0.002]
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        mujoco.mj_forward(e.model, e.data)
        for _ in range(100):
            _, _, te, tr, info = e.step([-1, 0, 0, 0])
            if te or tr:
                break
        self.assertEqual(info['outcome'], 'SAFE_OUTSIDE')

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            self.env.step([0, 0, np.nan, 0])

    def test_residual_wrapper_zero_matches_classical_rollout(self):
        from sim3d.residual import ResidualLandingEnv
        raw = MissionEnv(difficulty=0)
        reference = MissionEnv(difficulty=0)
        wrapper = ResidualLandingEnv(raw)
        obs, _ = wrapper.reset(seed=12)
        ref_obs, _ = reference.reset(seed=12)
        controller = LandingController()
        self.assertEqual(obs.shape, (27,))
        for _ in range(20):
            obs, _, te, tr, _ = wrapper.step(np.zeros(4))
            ref_obs, _, _, _, _ = reference.step(controller(ref_obs))
            np.testing.assert_allclose(obs[:16], ref_obs, atol=1e-6)
            if te or tr:
                break
        wrapper.close()
        reference.close()

    def test_residual_uses_declared_nominal_params_and_period(self):
        from sim3d.residual import ResidualLandingEnv
        p = RocketParams(mass=2)
        e = ResidualLandingEnv(MissionEnv(params=p, config=MissionConfig(control_hz=25)))
        obs, _ = e.reset(seed=4)
        expected = LandingController(params=p, dt=0.04)(obs[:16])
        np.testing.assert_allclose(obs[16:20], expected, atol=1e-6)
        self.assertEqual(e.controller.dt, .04)
        e.close()

    def test_contact_loss_reenters_acquisition(self):
        c = LandingController()
        obs = np.zeros(16)
        obs[8] = 1
        obs[12] = .25
        c(obs)
        self.assertEqual(c.phase, 'SETTLE')
        obs[12] = 0
        obs[2] = .15
        obs[0] = .2
        c(obs)
        self.assertEqual(c.phase, 'ACQUIRE')

    def test_controller_accepts_only_observation(self):
        obs, _ = self.env.reset(seed=1)
        controller = LandingController()
        a = controller(obs)
        self.assertEqual(a.shape, (4,))
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.all(np.abs(a) <= 1))
        with self.assertRaises(ValueError):
            controller(np.zeros(9))
        obs[0] = np.nan
        with self.assertRaises(ValueError):
            controller(obs)

    def test_legacy_deadline_semantics_preserved(self):
        e = Rocket3DEnv()
        e.reset(seed=2)
        e.cfg.max_steps = 1
        _, _, te, tr, _ = e.step([-1, 0, 0, 0])
        self.assertFalse(te)
        self.assertTrue(tr)
        e.close()

    def test_engine_force_and_gimbal_torque_signs(self):
        e = self.env
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        mujoco.mj_forward(e.model, e.data)
        e.step([1, 1, 0, 0])
        self.assertGreater(e.data.qvel[0], 0)
        self.assertLess(e.data.qvel[4], 0)

    def test_controller_invariant_to_world_yaw(self):
        e = self.env
        obs, _ = e.reset(seed=7)
        a = LandingController()(obs)
        angle = 1.2
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        e.data.qpos[:2] = rot @ e.data.qpos[:2]
        e.data.qvel[:2] = rot @ e.data.qvel[:2]
        q, result = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0., 0., 1.]), angle)
        mujoco.mju_mulQuat(result, q, e.data.qpos[3:7].copy())
        e.data.qpos[3:7] = result
        mujoco.mj_forward(e.model, e.data)
        obs2 = e._obs(e.kinematics(), 0)
        np.testing.assert_allclose(LandingController()(obs2), a, atol=1e-6)

    def test_substep_detects_fast_touchdown(self):
        e = self.env
        e.data.qpos[:3] = [0, 0, e._rest_height + 0.004]
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        e.data.qvel[2] = -2
        mujoco.mj_forward(e.model, e.data)
        _, _, te, _, info = e.step([-1, 0, 0, 0])
        self.assertTrue(te)
        self.assertEqual(info['outcome'], 'CRASH')
        self.assertGreater(info['impact_vertical'], 1)

    def test_harder_second_impact_is_not_ignored(self):
        e = self.env
        e._touched = True
        e._impact_speed = 0.2
        e.data.qpos[:3] = [0, 0, e._rest_height + 0.004]
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        e.data.qvel[2] = -2
        mujoco.mj_forward(e.model, e.data)
        _, _, te, _, info = e.step([-1, 0, 0, 0])
        self.assertTrue(te)
        self.assertEqual(info['outcome'], 'CRASH')
        self.assertGreater(info['impact_speed'], 1)

    def test_noise_seed_independent_of_preceding_episode(self):
        from sim3d.mission_wrappers import mission_sensors
        e = mission_sensors(MissionEnv(), delay=0, noise_scale=1)
        first, _ = e.reset(seed=123)
        e.step([0, 0, 0, 0])
        second, _ = e.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        clean, _ = e.unwrapped.reset(seed=123)
        self.assertFalse(np.array_equal(first, clean))
        e.close()

    def test_excessive_contact_tilt_cannot_recover_to_success(self):
        e = self.env
        e._touched = True
        e._impact_tilt = np.deg2rad(20)
        _, _, te, tr, info = e.step([0, 0, 0, 0])
        self.assertTrue(te)
        self.assertFalse(tr)
        self.assertEqual(info['outcome'], 'CRASH')

    def test_transient_hull_contact_latches_failure(self):
        e = self.env
        original = e._leg_contacts
        n = 0
        def contacts():
            nonlocal n
            n += 1
            return (0, True) if n == 1 else original()
        with patch.object(e, '_leg_contacts', side_effect=contacts):
            _, _, te, _, info = e.step([0, 0, 0, 0])
        self.assertTrue(te)
        self.assertEqual(info['outcome'], 'CRASH')

    def test_single_substep_contact_loss_resets_dwell(self):
        e = self.env
        e.data.qpos[:3] = [0, 0, e._rest_height]
        e.data.qpos[3:7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        mujoco.mj_forward(e.model, e.data)
        e._settle = 24
        with patch.object(e, '_leg_contacts', side_effect=[(0, False)] + [(4, False)]*20):
            _, _, te, _, _ = e.step([-1, 0, 0, 0])
        self.assertFalse(te)
        self.assertEqual(e._settle, 0)


if __name__ == '__main__':
    unittest.main()
