"""
[ZH] 2D 仿真沙盒受控随机种子与确定性轨迹测试套件。
[EN] 2D simulation sandbox controlled random seed & deterministic trajectory test suite.

测试目标 / Verification Objectives:
    1. 验证 env.reset(seed=K) 重置位姿的完全确定性 (相同种子下小车初始坐标与朝向严格相同)；
    2. 验证不同种子能够产生多样化的有效探索起始位姿；
    3. 验证通过 options 参数可精确注入自定义初始位姿 ("init_pos", "init_theta")；
    4. 验证在固定动作序列驱动下，跨多个独立沙盒实例运行多步，其全生命周期位姿、观测、回报与碰撞状态方差严格为 0。
"""

import unittest
import numpy as np
from envs.arena2d import Arena2DEnv


class TestArena2DSeed(unittest.TestCase):
    """
    [ZH] Arena2D 沙盒受控随机流与 0 方差测试用例集合。
    [EN] Arena2D sandbox controlled random stream & zero-variance test cases.
    """

    def test_seed_reset_determinism(self):
        """
        [ZH] 验证相同种子重置环境生成完全一致的初始状态。
        """
        env = Arena2DEnv()
        obs1, info1 = env.reset(seed=42)
        pos1, theta1 = info1["position"], info1["orientation"]

        obs2, info2 = env.reset(seed=42)
        pos2, theta2 = info2["position"], info2["orientation"]

        # 断言初始位置与朝向严格一致 (0 方差)
        np.testing.assert_allclose(pos1, pos2, atol=1e-7, err_msg="相同种子两次重置位置不一致")
        self.assertAlmostEqual(theta1, theta2, places=7, msg="相同种子两次重置朝向角度不一致")
        np.testing.assert_allclose(obs1, obs2, atol=1e-7, err_msg="相同种子初始观测不一致")

    def test_different_seeds_produce_diversity(self):
        """
        [ZH] 验证不同种子能够产生不同的有效初始位姿，避免伪随机流退化为常量。
        """
        env = Arena2DEnv()
        _, info_a = env.reset(seed=101)
        _, info_b = env.reset(seed=202)

        pos_diff = np.linalg.norm(info_a["position"] - info_b["position"])
        theta_diff = abs(info_a["orientation"] - info_b["orientation"])

        self.assertGreater(pos_diff + theta_diff, 1e-3, "不同种子未产生差异化的初始位姿")

    def test_options_custom_pose_injection(self):
        """
        [ZH] 验证通过 options 参数显式注入初始位置与朝向的精确性。
        """
        env = Arena2DEnv()
        custom_pos = [2.0, 3.5]
        custom_theta = 1.25

        obs, info = env.reset(options={"init_pos": custom_pos, "init_theta": custom_theta})
        np.testing.assert_allclose(info["position"], np.array(custom_pos, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(info["orientation"], custom_theta, places=6)
        self.assertFalse(info["bumper_hit"])

    def test_zero_variance_trajectory(self):
        """
        [ZH] 验证在固定动作序列下多次独立运行产生的整条轨迹方差严格为 0。
        """
        # 设计 40 步涵盖前进、后退、左转、右转的固定测试动作序列
        action_sequence = [1, 1, 3, 1, 4, 1, 2, 1, 3, 3, 1, 4, 1, 1, 2, 0] * 3
        num_runs = 5
        fixed_seed = 2026

        all_run_positions = []
        all_run_thetas = []
        all_run_rewards = []
        all_run_observations = []

        for run_idx in range(num_runs):
            env = Arena2DEnv()
            obs, info = env.reset(seed=fixed_seed)

            run_pos = [info["position"].copy()]
            run_theta = [info["orientation"]]
            run_rew = []
            run_obs = [obs.copy()]

            for action in action_sequence:
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                run_pos.append(next_info["position"].copy())
                run_theta.append(next_info["orientation"])
                run_rew.append(reward)
                run_obs.append(next_obs.copy())

                if terminated or truncated:
                    break

            all_run_positions.append(run_pos)
            all_run_thetas.append(run_theta)
            all_run_rewards.append(run_rew)
            all_run_observations.append(run_obs)

        # 检查各次运行步数完全一致
        step_counts = [len(r) for r in all_run_positions]
        self.assertEqual(len(set(step_counts)), 1, "多次运行由于非确定性导致步数出现偏差")

        # 转换为 numpy 数组进行统计方差断言
        arr_pos = np.array(all_run_positions)  # shape: (5, num_steps, 2)
        arr_theta = np.array(all_run_thetas)    # shape: (5, num_steps)
        arr_rew = np.array(all_run_rewards)      # shape: (5, num_steps-1)
        arr_obs = np.array(all_run_observations) # shape: (5, num_steps, 8)

        # 计算并断言各维度的时间序列跨 Run 方差严格为 0
        var_pos = np.var(arr_pos, axis=0)
        var_theta = np.var(arr_theta, axis=0)
        var_rew = np.var(arr_rew, axis=0)
        var_obs = np.var(arr_obs, axis=0)

        np.testing.assert_allclose(var_pos, 0.0, atol=1e-10, err_msg="轨迹位置跨运行存在随机方差")
        np.testing.assert_allclose(var_theta, 0.0, atol=1e-10, err_msg="轨迹朝向角跨运行存在随机方差")
        np.testing.assert_allclose(var_rew, 0.0, atol=1e-10, err_msg="轨迹即时回报跨运行存在随机方差")
        np.testing.assert_allclose(var_obs, 0.0, atol=1e-10, err_msg="雷达观测读数跨运行存在随机方差")


if __name__ == "__main__":
    unittest.main()
