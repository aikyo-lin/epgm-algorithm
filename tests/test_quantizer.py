"""
[ZH] 事件量化器单元测试。
[EN] Event quantizer unit tests.
"""

import unittest
import numpy as np
from epgm.quantizer import EventQuantizer


class TestEventQuantizer(unittest.TestCase):
    def setUp(self):
        self.quantizer = EventQuantizer(num_sensors=8, distance_bins=3)

    def test_bumper_collision_priority(self):
        """[ZH] 测试微动碰撞开关的绝对优先级（固定映射为事件0）"""
        obs = np.ones(8, dtype=np.float32)
        event_id, _ = self.quantizer.quantize(obs, bumper_hit=True)
        self.assertEqual(event_id, 0)

    def test_spatial_binning_deterministic(self):
        """[ZH] 测试确定性空间感受野分箱输出一致性"""
        obs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        e1, _ = self.quantizer.quantize(obs, bumper_hit=False)
        self.quantizer.reset()
        e2, _ = self.quantizer.quantize(obs, bumper_hit=False)
        self.assertEqual(e1, e2)
        self.assertGreater(e1, 0)

    def test_dynamic_pulse_detection(self):
        """[ZH] 测试剧烈动态变化时触发的高位动态脉冲标志"""
        obs1 = np.zeros(8, dtype=np.float32)
        self.quantizer.quantize(obs1, bumper_hit=False)

        # 突变到极远距离 / Sudden jump
        obs2 = np.ones(8, dtype=np.float32)
        event_id2, _ = self.quantizer.quantize(obs2, bumper_hit=False)
        # 应该带有 >= 100 的动态标志
        self.assertGreaterEqual(event_id2, 100)


if __name__ == "__main__":
    unittest.main()
