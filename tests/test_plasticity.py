"""
[ZH] 局部可塑性、拓扑生枝与修剪机制测试。
[EN] Local plasticity, topological budding and pruning mechanism tests.
"""

import unittest
from epgm.core import EventPredictiveGraph
from epgm.plasticity import TopologicalPlasticity


class TestTopologicalPlasticity(unittest.TestCase):
    def setUp(self):
        self.graph = EventPredictiveGraph(action_dim=5)
        self.plasticity = TopologicalPlasticity(self.graph, learning_rate=0.5, prune_threshold_steps=100)

    def test_edge_budding_on_surprise(self):
        """[ZH] 测试未探索边遭遇惊喜时触发的拓扑生枝 (Budding)"""
        diag = self.plasticity.step_learning(source_event=1, action=2, observed_event=3, reward=1.0, step_index=1)
        self.assertTrue(diag["is_bud"])
        self.assertFalse(diag["is_hit"])

        edge = self.graph.get_best_prediction(source_event=1, action=2)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.target_event, 3)
        self.assertEqual(edge.observation_count, 1)

    def test_synaptic_reinforcement_on_hit(self):
        """[ZH] 测试预测命中时的突触长时程增强 (LTP) 与回报平滑滤波更新"""
        # 第一次触发生枝
        self.plasticity.step_learning(source_event=1, action=2, observed_event=3, reward=0.0, step_index=1)
        # 第二次预测命中
        diag = self.plasticity.step_learning(source_event=1, action=2, observed_event=3, reward=1.0, step_index=2)
        self.assertTrue(diag["is_hit"])
        self.assertFalse(diag["is_bud"])

        edge = self.graph.get_best_prediction(source_event=1, action=2)
        self.assertEqual(edge.observation_count, 2)
        # expected_reward: 0.0 + 0.5 * (1.0 - 0.0) = 0.5
        self.assertAlmostEqual(edge.expected_reward, 0.5, places=3)

    def test_stale_edge_pruning(self):
        """[ZH] 测试休眠冗余边的自动衰减与修剪清理"""
        self.plasticity.step_learning(source_event=1, action=0, observed_event=2, reward=0.0, step_index=1)
        self.assertIn((1, 0), self.graph.adjacency)

        # 步数跳跃超过 100 步且样本少
        pruned = self.plasticity.prune_stale_edges(current_step=200)
        self.assertEqual(pruned, 1)
        self.assertNotIn((1, 0), self.graph.adjacency)


if __name__ == "__main__":
    unittest.main()
