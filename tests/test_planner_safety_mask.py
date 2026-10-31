"""
[ZH] 决策规划器安全硬掩码 (Hard Safety Masking) 单元测试套件。
[EN] Decision planner Hard Safety Masking unit test suite.

测试目标 / Verification Objectives:
    1. 验证当预测边指向碰撞事件 (target_event == 0) 且置信度/频次达标时，规划器 100% 屏蔽该动作；
    2. 验证在高 epsilon-greedy 随机探索率下，安全硬掩码仍然生效，绝不随机抽中撞墙动作；
    3. 验证置信度与最小观测次数门限判定准则；
    4. 验证死角陷阱全被掩码时的后退/紧急避险保底逻辑；
    5. 验证未被屏蔽的安全动作分支保留正常前向评价机制。
"""

import unittest
from epgm.core import EventPredictiveGraph, EPGMEdge
from epgm.planner import GraphPlanner


class TestPlannerSafetyMask(unittest.TestCase):
    """
    [ZH] 规划器安全硬掩码核心机制测试集合。
    [EN] Planner hard safety mask core mechanism test collection.
    """

    def setUp(self):
        """
        [ZH] 初始化测试图与规划器实例。
        """
        self.graph = EventPredictiveGraph(action_dim=5)
        self.planner = GraphPlanner(
            self.graph,
            action_dim=5,
            curiosity_weight=0.5,
            collision_conf_thresh=0.5,
            collision_min_obs=1,
            seed=42,
        )

    def test_hard_safety_mask_intercepts_collision(self):
        """
        [ZH] 验证在贪心利用模式下，已知必然碰撞的动作分支被 100% 拦截并赋予负无穷打分。
        """
        # 构造场景: 在事件 10 下，动作 1 必然导致碰撞 (target_event=0, conf=0.8, obs_count=5)
        col_edge = EPGMEdge(
            source_event=10,
            action=1,
            target_event=0,
            expected_reward=-10.0,
            observation_count=5,
            confidence=0.8,
        )
        self.graph.adjacency[(10, 1)] = [col_edge]

        # 连续决策 100 次，动作 1 必须被 100% 拦截
        for _ in range(100):
            action, info = self.planner.select_action(current_event=10, epsilon_greedy=0.0)
            self.assertNotEqual(action, 1, "已知撞墙动作 1 未被安全掩码拦截")
            self.assertIn(1, info.get("masked_actions", []))
            self.assertNotIn(1, info.get("safe_actions", []))
            self.assertEqual(info["scores"][1], -float("inf"), "被掩码动作的打分未被强制置为 -inf")

    def test_safety_mask_under_high_epsilon_exploration(self):
        """
        [ZH] 验证在高探索率 (如 epsilon=0.9 与 epsilon=1.0) 下，安全硬掩码依然 100% 拦截撞墙动作。
        """
        col_edge = EPGMEdge(
            source_event=10,
            action=1,
            target_event=0,
            expected_reward=-10.0,
            observation_count=5,
            confidence=0.8,
        )
        self.graph.adjacency[(10, 1)] = [col_edge]

        # 在 90% 随机探索下采样 100 次
        for _ in range(100):
            action, info = self.planner.select_action(current_event=10, epsilon_greedy=0.9)
            self.assertNotEqual(action, 1, "随机探索模式下违规选中了危险动作 1")
            self.assertIn(1, info["masked_actions"])

        # 在 100% 纯随机探索下采样 100 次
        for _ in range(100):
            action, info = self.planner.select_action(current_event=10, epsilon_greedy=1.0)
            self.assertNotEqual(action, 1, "100% 纯随机探索下违规选中了危险动作 1")

    def test_confidence_and_observation_thresholds(self):
        """
        [ZH] 验证置信度门限 (conf_thresh) 与观测频次 (min_obs) 的精准拦截分界。
        """
        # 低置信度 (conf=0.3 < 0.5): 不应触发硬掩码
        self.graph.adjacency[(12, 1)] = [
            EPGMEdge(source_event=12, action=1, target_event=0, confidence=0.3, observation_count=5)
        ]
        self.assertFalse(self.planner.is_action_hazardous(12, 1))

        # 0 次观测 (obs_count=0 < 1): 不应触发硬掩码
        self.graph.adjacency[(12, 2)] = [
            EPGMEdge(source_event=12, action=2, target_event=0, confidence=0.9, observation_count=0)
        ]
        self.assertFalse(self.planner.is_action_hazardous(12, 2))

        # 临界达标 (conf=0.5, obs=1): 严格触发硬掩码
        self.graph.adjacency[(12, 3)] = [
            EPGMEdge(source_event=12, action=3, target_event=0, confidence=0.5, observation_count=1)
        ]
        self.assertTrue(self.planner.is_action_hazardous(12, 3))

    def test_all_masked_corner_case_fallback(self):
        """
        [ZH] 验证当所有候选动作 (0~4) 均被预测为碰撞死角时，保底机制安全降级并不抛出异常。
        """
        # 假设 5 个动作全被判定为撞墙
        for a in range(5):
            self.graph.adjacency[(20, a)] = [
                EPGMEdge(source_event=20, action=a, target_event=0, expected_reward=-10.0, confidence=0.9, observation_count=3)
            ]

        # 执行决策
        action, info = self.planner.select_action(current_event=20, epsilon_greedy=0.0)
        # 应触发保底动作 (后退 a=2 或停止 a=0)
        self.assertTrue(info.get("fallback_triggered", False), "全掩码死角未触发保底标志")
        self.assertIn(action, [0, 2], f"保底动作 {action} 不在预期集合 [0, 2] 中")

    def test_multiple_hazardous_actions_masked(self):
        """
        [ZH] 验证同时存在多个危险动作时，掩码可完整屏蔽所有危险分支。
        """
        # 动作 1 (前进) 和 动作 3 (左转) 均会撞墙
        self.graph.adjacency[(30, 1)] = [
            EPGMEdge(source_event=30, action=1, target_event=0, confidence=0.9, observation_count=2)
        ]
        self.graph.adjacency[(30, 3)] = [
            EPGMEdge(source_event=30, action=3, target_event=0, confidence=0.75, observation_count=4)
        ]

        action, info = self.planner.select_action(current_event=30, epsilon_greedy=0.0)
        self.assertNotIn(action, [1, 3])
        self.assertEqual(set(info["masked_actions"]), {1, 3})
        self.assertEqual(set(info["safe_actions"]), {0, 2, 4})

    def test_safety_mask_toggle(self):
        """
        [ZH] 验证当显式设置 enable_safety_mask=False 时，硬掩码被旁路放行。
        """
        self.graph.adjacency[(40, 1)] = [
            EPGMEdge(source_event=40, action=1, target_event=0, confidence=0.9, observation_count=2)
        ]
        _, info_enabled = self.planner.select_action(current_event=40, enable_safety_mask=True)
        self.assertIn(1, info_enabled["masked_actions"])

        _, info_disabled = self.planner.select_action(current_event=40, enable_safety_mask=False)
        self.assertEqual(len(info_disabled["masked_actions"]), 0)
        self.assertIn(1, info_disabled["safe_actions"])


if __name__ == "__main__":
    unittest.main()
