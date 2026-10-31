"""
[ZH] EPGM 因果图结构与碎片导入导出测试。
[EN] EPGM causal graph structure and fragment export/import tests.
"""

import unittest
import numpy as np
from epgm.core import EventPredictiveGraph, EPGMEdge


class TestEventPredictiveGraph(unittest.TestCase):
    def setUp(self):
        self.graph = EventPredictiveGraph(action_dim=5)

    def test_node_addition(self):
        """[ZH] 测试节点添加与原型质心滑动平均更新"""
        proto1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        node = self.graph.add_node(event_id=10, prototype=proto1)
        self.assertEqual(node.event_id, 10)
        self.assertEqual(node.visit_count, 1)

        # 再次访问
        proto2 = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        node2 = self.graph.add_node(event_id=10, prototype=proto2)
        self.assertEqual(node2.visit_count, 2)
        self.assertAlmostEqual(float(node2.prototype_vector[0]), 0.11, places=3)

    def test_export_import_fragment(self):
        """[ZH] 测试增量图碎片序列化与反序列化融合一致性"""
        self.graph.add_node(1)
        self.graph.add_node(2)
        edge = EPGMEdge(source_event=1, action=1, target_event=2, expected_reward=0.8, observation_count=5)
        self.graph.adjacency[(1, 1)] = [edge]

        fragment = self.graph.export_fragment()
        self.assertEqual(fragment["meta"]["edge_count"], 1)

        # 在新图引擎中导入融合
        new_graph = EventPredictiveGraph(action_dim=5)
        nodes_up, edges_up = new_graph.import_fragment(fragment)
        self.assertEqual(nodes_up, 2)
        self.assertEqual(edges_up, 1)

        pred = new_graph.get_best_prediction(source_event=1, action=1)
        self.assertIsNotNone(pred)
        self.assertEqual(pred.target_event, 2)
        self.assertAlmostEqual(pred.expected_reward, 0.8, places=3)


if __name__ == "__main__":
    unittest.main()
