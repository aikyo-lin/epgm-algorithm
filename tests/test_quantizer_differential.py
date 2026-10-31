"""
[ZH] 端云量化器差分动态脉冲与空间对齐单元测试套件。
[EN] Edge-cloud quantizer differential dynamic pulse & spatial alignment unit test suite.

测试目标 / Verification Objectives:
    1. 验证静态观测输入下事件 ID 严格位于 1 ~ 81 区间；
    2. 验证一阶导数剧烈变化 (||Delta s||_2 > 0.25) 时，事件 ID 准确叠加 100 偏置进入 101 ~ 181 动态脉冲区间；
    3. 验证微动碰撞开关 (bumper_hit=True) 绝对最高优先级，无条件映射为事件 0 并刷新历史缓存；
    4. 验证差分动态门限边界 (sum(Delta s_i^2) > 0.0625) 的精确数学判定；
    5. 验证 Python 算法与固件端 C 语言定点整数量化逻辑 (sum(Delta d_i^2) > 62,500) 在多步时序输入下的 100% 逐帧完全等价性。
"""

import unittest
import numpy as np
from epgm.quantizer import EventQuantizer


def c_quantizer_reference(distances_mm, bumper_hit, state):
    """
    [ZH] 纯 Python 模拟的固件端 epgm_core.c 定点整数量化参考实现。
    [EN] Pure Python reference simulation of firmware epgm_core.c fixed-point quantizer.

    Args:
        distances_mm: 8 向毫米传感器距离数组 uint16_t [0, 1000].
        bumper_hit: 物理微动碰撞标志 bool.
        state: 模拟 epgm_runtime_t 的量化器历史状态字典.
    """
    if bumper_hit:
        state["last_distances"] = list(distances_mm)
        state["has_last"] = True
        return 0

    # 1. 距离分箱: <333mm 为 0 (危险), 333~666mm 为 1 (接近), >=667mm 为 2 (安全)
    bins = [0 if d < 333 else (1 if d < 667 else 2) for d in distances_mm]

    # 2. 四象限聚合
    front = min(bins[0], bins[1], bins[7])
    right = min(bins[1], bins[2], bins[3])
    back  = min(bins[3], bins[4], bins[5])
    left  = min(bins[5], bins[6], bins[7])

    spatial_event = 1 + (front * 27 + right * 9 + back * 3 + left)

    # 3. 定点平方差动态脉冲判定 (门限 250^2 = 62,500)
    is_dynamic = False
    if state.get("has_last", False):
        sum_sq = sum((int(d) - int(ld)) ** 2 for d, ld in zip(distances_mm, state["last_distances"]))
        if sum_sq > 62500:
            is_dynamic = True

    state["last_distances"] = list(distances_mm)
    state["has_last"] = True

    return spatial_event + (100 if is_dynamic else 0)


class TestQuantizerDifferential(unittest.TestCase):
    """
    [ZH] 量化器动态差分与端端对齐测试集合。
    [EN] Quantizer dynamic differential and edge-cloud alignment test collection.
    """

    def setUp(self):
        self.quantizer = EventQuantizer(num_sensors=8, distance_bins=3, delta_threshold=0.25, max_range=1.0)

    def test_static_event_id_range(self):
        """
        [ZH] 验证平缓无突变的连续输入下，事件 ID 严格位于 1 ~ 81 静态空间。
        """
        # 全危险贴身观测 -> 1 + (0*27 + 0*9 + 0*3 + 0) = 1
        obs_danger = np.full(8, 0.1, dtype=np.float32)
        e_danger, _ = self.quantizer.quantize(obs_danger)
        self.assertEqual(e_danger, 1)

        # 全开阔安全观测 -> 1 + (2*27 + 2*9 + 2*3 + 2) = 81
        self.quantizer.reset()
        obs_safe = np.full(8, 0.9, dtype=np.float32)
        e_safe, _ = self.quantizer.quantize(obs_safe)
        self.assertEqual(e_safe, 81)

        # 中距观测 -> 1 + (1*27 + 1*9 + 1*3 + 1) = 41
        self.quantizer.reset()
        obs_mid = np.full(8, 0.5, dtype=np.float32)
        e_mid, _ = self.quantizer.quantize(obs_mid)
        self.assertEqual(e_mid, 41)

    def test_dynamic_delta_pulse_triggering(self):
        """
        [ZH] 验证突发剧烈变化触发 100 偏置动态事件，且恢复静止后平稳回归静态事件。
        """
        # 第 1 步: 初始静止观测 (全安全 81)
        obs_t0 = np.full(8, 0.9, dtype=np.float32)
        e0, _ = self.quantizer.quantize(obs_t0)
        self.assertEqual(e0, 81)

        # 第 2 步: 突变至贴身贴墙 (0.9 -> 0.1, 每路变化 0.8, ||Delta||_2 = sqrt(8 * 0.8^2) ≈ 2.26 > 0.25)
        # 应触发动态事件: 静态 ID 1 + 100 = 101
        obs_t1 = np.full(8, 0.1, dtype=np.float32)
        e1, _ = self.quantizer.quantize(obs_t1)
        self.assertEqual(e1, 101, f"突变后未产生正确的动态事件 ID, 实际得到: {e1}")

        # 第 3 步: 维持在贴身状态，导数恢复为 0
        # 应恢复为纯静态事件 1
        obs_t2 = np.full(8, 0.1, dtype=np.float32)
        e2, _ = self.quantizer.quantize(obs_t2)
        self.assertEqual(e2, 1, f"静止后未恢复为静态事件 ID, 实际得到: {e2}")

    def test_differential_threshold_mathematical_boundary(self):
        """
        [ZH] 验证差分变动量在阈值临界点两侧 (<=0.0625 与 >0.0625) 的精确分界。
        """
        base_obs = np.full(8, 0.5, dtype=np.float32)
        self.quantizer.quantize(base_obs)

        # 案例 A: 微小扰动，单路传感器变化 0.20 (sum_sq = 0.04 <= 0.0625)
        # 不应触发动态事件 (保持静态 41)
        small_change = base_obs.copy()
        small_change[0] += 0.20
        e_small, _ = self.quantizer.quantize(small_change)
        self.assertEqual(e_small, 41, "微小扰动误触发了动态事件")

        # 重置回 base_obs
        self.quantizer.quantize(base_obs)

        # 案例 B: 显著扰动，单路传感器变化 0.26 (sum_sq = 0.0676 > 0.0625)
        # 必须触发动态事件 (41 + 100 = 141)
        large_change = base_obs.copy()
        large_change[0] += 0.26
        e_large, _ = self.quantizer.quantize(large_change)
        self.assertEqual(e_large, 141, "显著扰动未触发动态事件")

    def test_bumper_collision_priority(self):
        """
        [ZH] 验证碰撞开关触发时强制返回 0，且后续观测以此碰撞帧作为差分基准。
        """
        obs = np.full(8, 0.2, dtype=np.float32)
        e_col, _ = self.quantizer.quantize(obs, bumper_hit=True)
        self.assertEqual(e_col, 0, "微动开关碰撞事件必须返回 0")

        # 碰撞后剧烈倒车至开阔区域 (0.2 -> 0.85)
        obs_retreat = np.full(8, 0.85, dtype=np.float32)
        e_retreat, _ = self.quantizer.quantize(obs_retreat, bumper_hit=False)
        # 静态 81 + 动态 100 = 181
        self.assertEqual(e_retreat, 181, "碰撞后大幅倒车未正确基于碰撞位姿计算动态脉冲")

    def test_python_and_c_reference_equivalence_sequence(self):
        """
        [ZH] 验证 Python 量化器与 C 固件定点参考模型在 10 步复杂测试序列下的逐帧 100% 一致性。
        """
        # 测试序列: (8路距离 mm, bumper_hit, 预期行为)
        test_sequence = [
            ([900] * 8, False),   # Step 0: 静态开阔 -> 81
            ([900] * 8, False),   # Step 1: 持续静态 -> 81
            ([100] * 8, False),   # Step 2: 剧烈冲向障碍 -> 101 (动态)
            ([100] * 8, False),   # Step 3: 静止贴近 -> 1
            ([100] * 8, True),    # Step 4: 发生碰撞 -> 0
            ([500] * 8, False),   # Step 5: 中距倒车脱困 -> 141 (动态)
            ([500] * 8, False),   # Step 6: 持续中距巡航 -> 41
            ([510] * 8, False),   # Step 7: 微小波动 (每路10mm, sum_sq=800 < 62500) -> 41
            ([850] * 8, False),   # Step 8: 突发加速脱离 -> 181 (动态)
            ([850] * 8, False),   # Step 9: 恢复静态 -> 81
        ]

        # 初始化量化器 (归一化参数: max_range=1000.0, delta_threshold=250.0)
        py_quantizer = EventQuantizer(num_sensors=8, distance_bins=3, delta_threshold=250.0, max_range=1000.0)
        c_state = {}

        for step_idx, (dist_mm, bump) in enumerate(test_sequence):
            dist_array = np.array(dist_mm, dtype=np.float32)
            py_id, _ = py_quantizer.quantize(dist_array, bumper_hit=bump)
            c_id = c_quantizer_reference(dist_mm, bump, c_state)

            self.assertEqual(
                py_id,
                c_id,
                f"Step {step_idx} 发生对齐失配: Python 输出 {py_id}, C 定点参考输出 {c_id}"
            )


if __name__ == "__main__":
    unittest.main()
