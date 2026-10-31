"""
[ZH] 挑战者对抗性压力测试套件 (Challenger Adversarial Stress Test Suite)。
[EN] Challenger Adversarial Stress Test Suite for Milestone M1 (Algorithm & Simulation).

本测试套件由独立审查挑战者 (Empirical Challenger) 编写并运行，旨在通过严苛的对抗性假设与极限边界输入，
实证检验 EPGM 算法与仿真核心组件的确定性、鲁棒性与接口一致性。

覆盖核心挑战领域:
1. 随机种子确定性挑战 (Seed Determinism Challenge - 50 Distinct Random Seeds):
   - 跨 50 个不同随机种子，验证每次重置及其长达 40 步的运动轨迹在观测、即时回报、智能体坐标与朝向角度上的按位完全一致性 (Bitwise Identical, 0 方差)。
   - 验证 50 个种子产生的起始分布多样性，排除伪随机流退化或常数固化。
2. 安全硬掩码对抗性欺骗挑战 (Safety Masking Adversarial Challenge):
   - 构造注入高额诱惑回报与好奇心奖励 (+10.0 ~ +100.0) 的对抗性因果图碰撞分支。
   - 验证在贪心利用 (greedy) 与极限探索 (epsilon=1.0) 模式下，安全硬掩码实现 100% 绝对物理拦截 (0 碰撞穿透)。
   - 构造四面楚歌全碰撞死角 (All-Masked Dead-End)，验证规划器优雅降级触发安全退避保底机制，杜绝异常崩溃。
3. 量化器差分与边界极限挑战 (Quantizer Differential & Boundary Challenge):
   - 浮点扰动测试: 针对 0.25 动态阈值临界点两侧 (0.2499 vs 0.2501，以及 8 维各向同性分量) 进行亚毫厘级微扰压力检验。
   - 空间分箱三等分边界测试: 验证 1/3 (0.3333) 与 2/3 (0.6667) 边界处的跳变严格符合三值区间定义。
   - 物理碰撞开关绝对优先级与观测缓存一致性测试。
"""

import unittest
import numpy as np
from envs.arena2d import Arena2DEnv
from epgm.core import EventPredictiveGraph, EPGMEdge
from epgm.planner import GraphPlanner
from epgm.quantizer import EventQuantizer


class TestSeedDeterminismChallenge(unittest.TestCase):
    """
    [ZH] 随机种子跨 50 组独立用例确定性与多样性对抗验证。
    [EN] 50-seed determinism and diversity adversarial verification suite.
    """

    def test_50_seeds_trajectory_bitwise_determinism(self):
        """
        [ZH] 针对 50 个不同随机种子，分别执行两次完整的 40 步多动作轨迹仿真，
             验证两轮执行在每一步的位置、朝向、雷达观测和即时回报上完全按位一致。
        """
        # 40 步混合运动控制指令序列: 前进(1), 后退(2), 左转(3), 右转(4), 停止(0)
        action_sequence = [1, 1, 3, 1, 4, 1, 2, 3, 1, 4, 1, 1, 2, 3, 4, 1, 0, 1, 3, 4] * 2
        seeds = [1000 + i * 17 for i in range(50)]

        for seed in seeds:
            # 第一轮执行 (Run A)
            env_a = Arena2DEnv()
            obs_a, info_a = env_a.reset(seed=seed)
            pos_history_a = [info_a["position"].copy()]
            theta_history_a = [info_a["orientation"]]
            obs_history_a = [obs_a.copy()]
            rew_history_a = []

            for act in action_sequence:
                next_obs, rew, term, trunc, next_info = env_a.step(act)
                pos_history_a.append(next_info["position"].copy())
                theta_history_a.append(next_info["orientation"])
                obs_history_a.append(next_obs.copy())
                rew_history_a.append(rew)
                if term or trunc:
                    break

            # 第二轮执行 (Run B - 完全相同种子)
            env_b = Arena2DEnv()
            obs_b, info_b = env_b.reset(seed=seed)
            pos_history_b = [info_b["position"].copy()]
            theta_history_b = [info_b["orientation"]]
            obs_history_b = [obs_b.copy()]
            rew_history_b = []

            for act in action_sequence:
                next_obs, rew, term, trunc, next_info = env_b.step(act)
                pos_history_b.append(next_info["position"].copy())
                theta_history_b.append(next_info["orientation"])
                obs_history_b.append(next_obs.copy())
                rew_history_b.append(rew)
                if term or trunc:
                    break

            # 断言轨迹步数完全相同
            self.assertEqual(
                len(pos_history_a),
                len(pos_history_b),
                f"Seed {seed}: 运行两轮步数不一致 (A={len(pos_history_a)}, B={len(pos_history_b)})",
            )

            # 断言初始位姿按位完全一致
            np.testing.assert_array_equal(
                pos_history_a[0],
                pos_history_b[0],
                err_msg=f"Seed {seed}: 初始位置产生浮点偏差",
            )
            self.assertEqual(
                theta_history_a[0],
                theta_history_b[0],
                f"Seed {seed}: 初始朝向产生偏差",
            )

            # 断言全轨迹状态、观测与回报按位完全一致 (0 容差)
            for step_idx in range(len(pos_history_a)):
                np.testing.assert_array_equal(
                    pos_history_a[step_idx],
                    pos_history_b[step_idx],
                    err_msg=f"Seed {seed} Step {step_idx}: 智能体位置跨轮次出现差异",
                )
                self.assertEqual(
                    theta_history_a[step_idx],
                    theta_history_b[step_idx],
                    f"Seed {seed} Step {step_idx}: 智能体朝向跨轮次出现差异",
                )
                np.testing.assert_array_equal(
                    obs_history_a[step_idx],
                    obs_history_b[step_idx],
                    err_msg=f"Seed {seed} Step {step_idx}: 雷达观测跨轮次出现差异",
                )

            for step_idx in range(len(rew_history_a)):
                self.assertEqual(
                    rew_history_a[step_idx],
                    rew_history_b[step_idx],
                    f"Seed {seed} Step {step_idx}: 即时回报跨轮次出现差异",
                )

    def test_50_seeds_pose_diversity(self):
        """
        [ZH] 验证 50 个不同随机种子生成的初始位姿具有统计离散度，不存在退化或常数固化。
        """
        seeds = [2000 + i * 31 for i in range(50)]
        init_positions = []
        init_thetas = []

        env = Arena2DEnv()
        for seed in seeds:
            _, info = env.reset(seed=seed)
            init_positions.append(info["position"].copy())
            init_thetas.append(info["orientation"])

        pos_arr = np.array(init_positions)  # shape: (50, 2)
        theta_arr = np.array(init_thetas)    # shape: (50,)

        # 计算并断言标准差 > 0 (具有统计多样性)
        std_x = np.std(pos_arr[:, 0])
        std_y = np.std(pos_arr[:, 1])
        std_theta = np.std(theta_arr)

        self.assertGreater(std_x, 0.05, "50 个种子采样的 X 坐标分布过于集中或退化")
        self.assertGreater(std_y, 0.05, "50 个种子采样的 Y 坐标分布过于集中或退化")
        self.assertGreater(std_theta, 0.5, "50 个种子采样的朝向角分布过于集中或退化")

        # 验证所有 50 个初始位置互不重合
        unique_positions = np.unique(np.round(pos_arr, decimals=4), axis=0)
        self.assertEqual(len(unique_positions), 50, "存在不同种子产生了重叠的初始坐标")


class TestSafetyMaskingAdversarialChallenge(unittest.TestCase):
    """
    [ZH] 安全硬掩码对抗性攻击挑战：诱惑回报陷阱与全碰撞极端死角。
    [EN] Safety masking adversarial attacks: deceptive reward baiting & all-masked dead-ends.
    """

    def setUp(self):
        self.graph = EventPredictiveGraph(action_dim=5)
        self.planner = GraphPlanner(
            self.graph,
            action_dim=5,
            curiosity_weight=1.0,  # 增大好奇心权重
            collision_conf_thresh=0.5,
            collision_min_obs=1,
            seed=1337,
        )

    def test_adversarial_deceptive_curiosity_and_reward_bait(self):
        """
        [ZH] 对抗性场景 1: 在动作 1 (前进) 上注入指向碰撞 (target=0) 的超高诱惑任务回报 (+100.0)
             和超高好奇心奖励，而动作 2 (后退) 为安全的微薄回报分支 (+0.1)。
             检验:
             - 当关闭掩码时，规划器被诱饵欺骗，贪婪选择动作 1；
             - 当启用掩码时，无论在利用还是探索模式，实现 100% 绝对拦截！
        """
        current_event = 50

        # 对抗性恶意边: 动作 1 撞墙，但谎报超级回报 +100.0，且仅观测 1 次以最大化好奇心激励
        malicious_edge = EPGMEdge(
            source_event=current_event,
            action=1,
            target_event=0,
            expected_reward=100.0,
            observation_count=1,
            confidence=0.95,
        )
        self.graph.adjacency[(current_event, 1)] = [malicious_edge]

        # 安全边缘: 动作 2 转移到安全事件 55，回报仅 0.1，观测达 50 次 (好奇心极低)
        safe_edge = EPGMEdge(
            source_event=current_event,
            action=2,
            target_event=55,
            expected_reward=0.1,
            observation_count=50,
            confidence=0.99,
        )
        self.graph.adjacency[(current_event, 2)] = [safe_edge]

        # 1. 验证对比实验: 若关闭安全硬掩码 (enable_safety_mask=False)，
        # 智能体必定经受不住 +100.0 诱惑而选择动作 1 (验证攻击载荷确实有效)
        unprotected_action, _ = self.planner.select_action(
            current_event=current_event,
            epsilon_greedy=0.0,
            enable_safety_mask=False,
        )
        self.assertEqual(
            unprotected_action,
            1,
            "攻击载荷验证失败: 在无安全掩码保护下，规划器未被高额诱饵欺骗",
        )

        # 2. 启用安全硬掩码: 贪婪模式连续 200 次决策，动作 1 必须 100% 被拦截！
        for _ in range(200):
            action, info = self.planner.select_action(
                current_event=current_event,
                epsilon_greedy=0.0,
                enable_safety_mask=True,
            )
            self.assertNotEqual(
                action,
                1,
                "安全硬掩码防御失败: 智能体在贪婪模式下被诱饵吸引选择了撞墙动作 1",
            )
            self.assertIn(1, info["masked_actions"])
            self.assertEqual(info["scores"][1], -float("inf"))

        # 3. 极端探索模式 (epsilon_greedy=1.0 纯随机): 连续 1000 次采样，动作 1 必须 0 次被选中！
        chosen_actions = []
        for _ in range(1000):
            action, _ = self.planner.select_action(
                current_event=current_event,
                epsilon_greedy=1.0,
                enable_safety_mask=True,
            )
            self.assertNotEqual(
                action,
                1,
                "安全硬掩码防御失败: 智能体在 100% 随机探索模式下违规选中了被掩码的撞墙动作 1",
            )
            chosen_actions.append(action)

        # 验证其余安全动作能够被正常探索采样
        unique_explored = set(chosen_actions)
        self.assertNotIn(1, unique_explored)
        self.assertGreater(len(unique_explored), 1, "探索模式未覆盖其他安全候选动作")

    def test_multi_branch_adversarial_minefield(self):
        """
        [ZH] 对抗性场景 2: 雷区陷阱。动作 0, 1, 3, 4 全部埋伏了碰撞边 (target=0)，
             唯有动作 2 (后退) 是唯一生路。
             验证规划器 100% 命中唯一安全动作 2。
        """
        event_id = 77
        for a in [0, 1, 3, 4]:
            self.graph.adjacency[(event_id, a)] = [
                EPGMEdge(
                    source_event=event_id,
                    action=a,
                    target_event=0,
                    expected_reward=50.0,  # 均附加高额虚假诱惑
                    observation_count=2,
                    confidence=0.85,
                )
            ]

        # 动作 2 设为安全
        self.graph.adjacency[(event_id, 2)] = [
            EPGMEdge(
                source_event=event_id,
                action=2,
                target_event=80,
                expected_reward=1.0,
                observation_count=5,
                confidence=0.9,
            )
        ]

        # 连续测试 300 次 (混合不同探索率)
        for eps in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for _ in range(60):
                action, info = self.planner.select_action(
                    current_event=event_id,
                    epsilon_greedy=eps,
                    enable_safety_mask=True,
                )
                self.assertEqual(
                    action,
                    2,
                    f"在雷区陷阱中 (eps={eps})，规划器未能选中唯一安全动作 2，选出了 {action}",
                )
                self.assertEqual(set(info["masked_actions"]), {0, 1, 3, 4})
                self.assertEqual(info["safe_actions"], [2])

    def test_extreme_dead_end_all_masked_fallback(self):
        """
        [ZH] 对抗性场景 3: 极端四面楚歌死角。所有 5 个动作 (0~4) 均被预测为撞墙！
             验证:
             1. 规划器绝不抛出任何未捕获异常 (Zero Unhandled Exceptions)；
             2. 能够正确感知并标记 fallback_triggered == True；
             3. 优雅降级并选择安全保底退避动作 (动作 2 或 0)。
        """
        event_id = 99
        for a in range(5):
            self.graph.adjacency[(event_id, a)] = [
                EPGMEdge(
                    source_event=event_id,
                    action=a,
                    target_event=0,
                    expected_reward=-20.0,
                    observation_count=10,
                    confidence=0.99,
                )
            ]

        for eps in [0.0, 0.2, 0.8, 1.0]:
            action, info = self.planner.select_action(
                current_event=event_id,
                epsilon_greedy=eps,
                enable_safety_mask=True,
            )
            self.assertTrue(
                info["fallback_triggered"],
                f"在全掩码死角下 (eps={eps})，未触发 fallback_triggered 标记",
            )
            self.assertIn(
                action,
                [0, 2],
                f"全掩码死角退避动作异常: {action} 不属于预期的退避集合 [0, 2]",
            )
            self.assertEqual(len(info["masked_actions"]), 5)

    def test_confidence_threshold_adversarial_boundary(self):
        """
        [ZH] 对抗性场景 4: 置信度浮点边界探测 (0.4999 vs 0.5000 vs 0.5001)。
        """
        event_id = 88

        # 0.4999: 低于默认 0.5 门限，不应判定为危险
        self.graph.adjacency[(event_id, 1)] = [
            EPGMEdge(source_event=event_id, action=1, target_event=0, confidence=0.4999, observation_count=10)
        ]
        self.assertFalse(self.planner.is_action_hazardous(event_id, 1))

        # 0.5000: 恰好等于 0.5 门限，必须判定为危险
        self.graph.adjacency[(event_id, 2)] = [
            EPGMEdge(source_event=event_id, action=2, target_event=0, confidence=0.5000, observation_count=1)
        ]
        self.assertTrue(self.planner.is_action_hazardous(event_id, 2))

        # 0.5001: 略微高于 0.5 门限，必须判定为危险
        self.graph.adjacency[(event_id, 3)] = [
            EPGMEdge(source_event=event_id, action=3, target_event=0, confidence=0.5001, observation_count=1)
        ]
        self.assertTrue(self.planner.is_action_hazardous(event_id, 3))


class TestQuantizerDifferentialChallenge(unittest.TestCase):
    """
    [ZH] 量化器差分跳变阈值与空间几何边界极限微扰对抗验证。
    [EN] Quantizer differential threshold & spatial boundary infinitesimal perturbation challenge.
    """

    def setUp(self):
        self.quantizer = EventQuantizer(
            num_sensors=8,
            distance_bins=3,
            delta_threshold=0.25,
            max_range=1.0,
        )

    def test_single_dimension_infinitesimal_threshold_perturbation(self):
        """
        [ZH] 单维度微扰对抗: 围绕 0.25 阈值施加 ±0.0001 扰动。
             - delta = 0.2499 -> sum_sq = 0.06245001 <= 0.0625 -> 严格处于静态区间 (1 ~ 81)
             - delta = 0.2501 -> sum_sq = 0.06255001 > 0.0625  -> 严格触发动态区间 (101 ~ 181)
        """
        base_obs = np.full(8, 0.5, dtype=np.float32)

        # 1. 注入 baseline
        self.quantizer.reset()
        e_base, _ = self.quantizer.quantize(base_obs)
        self.assertEqual(e_base, 41)

        # 2. 负微扰 (0.2499)
        pert_neg = base_obs.copy()
        pert_neg[0] += 0.2499
        e_neg, _ = self.quantizer.quantize(pert_neg)
        self.assertEqual(
            e_neg,
            41,
            f"0.2499 微小扰动误触发了动态事件, 实际输出: {e_neg}",
        )

        # 3. 重新校准 baseline
        self.quantizer.reset()
        self.quantizer.quantize(base_obs)

        # 4. 正微扰 (0.2501)
        pert_pos = base_obs.copy()
        pert_pos[0] += 0.2501
        e_pos, _ = self.quantizer.quantize(pert_pos)
        self.assertEqual(
            e_pos,
            141,
            f"0.2501 显著扰动未触发动态事件, 实际输出: {e_pos}",
        )

    def test_distributed_8d_isotropic_perturbation(self):
        """
        [ZH] 8 维各向同性各路微扰:
             欧氏范数 ||Delta s||_2 = sqrt(8 * delta_i^2) = delta_i * sqrt(8)
             临界分量 delta_crit = 0.25 / sqrt(8) ≈ 0.0883883476
             - 每路施加 0.088388 (范数 0.2499989 < 0.25) -> 静态
             - 每路施加 0.088390 (范数 0.2500045 > 0.25) -> 动态
        """
        base_obs = np.full(8, 0.5, dtype=np.float32)

        # 亚临界分量
        sub_crit = 0.088388
        self.quantizer.reset()
        self.quantizer.quantize(base_obs)
        obs_sub = base_obs + sub_crit
        e_sub, _ = self.quantizer.quantize(obs_sub)
        self.assertEqual(
            e_sub,
            41,
            f"8 维亚临界微扰误触发了动态事件: e={e_sub}",
        )

        # 超临界分量
        sup_crit = 0.088390
        self.quantizer.reset()
        self.quantizer.quantize(base_obs)
        obs_sup = base_obs + sup_crit
        e_sup, _ = self.quantizer.quantize(obs_sup)
        self.assertEqual(
            e_sup,
            141,
            f"8 维超临界微扰未能触发动态事件: e={e_sup}",
        )

    def test_spatial_tercile_boundary_transitions(self):
        """
        [ZH] 空间分箱三等分边界跳变测试 (0.3333333 与 0.6666667)。
             - s < 1/3   -> Bin 0 (危险)
             - 1/3 <= s < 2/3 -> Bin 1 (接近)
             - s >= 2/3  -> Bin 2 (安全)
        """
        eps = 1e-5
        t1 = 1.0 / 3.0
        t2 = 2.0 / 3.0

        # 全向处于 Bin 0: s = t1 - eps
        self.quantizer.reset()
        obs_bin0 = np.full(8, t1 - eps, dtype=np.float32)
        e0, _ = self.quantizer.quantize(obs_bin0)
        self.assertEqual(e0, 1, f"t1 - eps 未能稳定量化为 Bin 0 (事件 1), 实际为 {e0}")

        # 全向恰好处于 Bin 1: s = t1 + eps
        self.quantizer.reset()
        obs_bin1 = np.full(8, t1 + eps, dtype=np.float32)
        e1, _ = self.quantizer.quantize(obs_bin1)
        self.assertEqual(e1, 41, f"t1 + eps 未能稳定量化为 Bin 1 (事件 41), 实际为 {e1}")

        # 全向跨入 Bin 2: s = t2 + eps
        self.quantizer.reset()
        obs_bin2 = np.full(8, t2 + eps, dtype=np.float32)
        e2, _ = self.quantizer.quantize(obs_bin2)
        self.assertEqual(e2, 81, f"t2 + eps 未能稳定量化为 Bin 2 (事件 81), 实际为 {e2}")

    def test_bumper_override_under_extreme_dynamics_and_out_of_bounds(self):
        """
        [ZH] 极端输入对抗:
             1. 伴随剧烈动态阶跃 (0.9 -> 0.1) 的同时发生碰撞 (bumper_hit=True)，碰撞事件 0 必须享有绝对最高覆盖权；
             2. 输入异常超出量程范围 (如 -10.0 与 +999.0)，系统通过 clip 保持鲁棒不崩；
             3. 碰撞发生后，内部缓存严格记录真实观测，以便下一时刻正确计算差分。
        """
        # 1. 伴随超大阶跃的碰撞
        self.quantizer.reset()
        obs_high = np.full(8, 0.95, dtype=np.float32)
        self.quantizer.quantize(obs_high)

        # 瞬变至极贴身并且触发微动碰撞
        obs_crash = np.full(8, 0.05, dtype=np.float32)
        e_crash, _ = self.quantizer.quantize(obs_crash, bumper_hit=True)
        self.assertEqual(
            e_crash,
            0,
            f"微动碰撞开关未能在动态突变下实施绝对覆盖: e={e_crash}",
        )

        # 2. 碰撞后轻微位移 (保持在 Bin 0，无动态突变，每路位移 0.02, sum_sq=8*0.0004=0.0032 < 0.0625)
        obs_post = np.full(8, 0.07, dtype=np.float32)
        e_post, _ = self.quantizer.quantize(obs_post, bumper_hit=False)
        self.assertEqual(
            e_post,
            1,
            f"碰撞后静态观测未正确依据碰撞时位姿计算差分: e={e_post}",
        )

        # 3. 越界输入稳健性检验 (输入负数与超大数值)
        self.quantizer.reset()
        obs_underflow = np.full(8, -5.0, dtype=np.float32)
        e_under, _ = self.quantizer.quantize(obs_underflow)
        self.assertEqual(e_under, 1, "负数越界输入未能被 clip 至 Bin 0")

        self.quantizer.reset()
        obs_overflow = np.full(8, 100.0, dtype=np.float32)
        e_over, _ = self.quantizer.quantize(obs_overflow)
        self.assertEqual(e_over, 81, "超大量程输入未能被 clip 至 Bin 2")


if __name__ == "__main__":
    unittest.main()
