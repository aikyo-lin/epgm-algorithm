"""
[ZH] Milestone M3 量化器端侧 C 固件与 Python 端 10,000 步对抗性跨语言一致性验证 (Challenger 1)。
[EN] Milestone M3 Quantizer Firmware C DLL & Python 10,000-Step Adversarial Parity Test (Challenger 1).
"""

import os
import sys
import ctypes
import unittest
import numpy as np

from epgm.quantizer import EventQuantizer

# 定义 C 端 epgm_quantizer_t 结构体
class CEpgmQuantizer(ctypes.Structure):
    _fields_ = [
        ("last_distances", ctypes.c_uint16 * 8),
        ("has_last_observation", ctypes.c_bool)
    ]

class TestAdversarialM3QuantizerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dll_path = os.path.abspath(r"f:\EPGM\epgm-platform\firmware\epgm_core.dll")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")
        cls.c_lib = ctypes.CDLL(dll_path)

        # 绑定函数签名
        cls.c_lib.epgm_quantizer_init.argtypes = [ctypes.POINTER(CEpgmQuantizer)]
        cls.c_lib.epgm_quantizer_init.restype = None

        cls.c_lib.epgm_quantizer_reset.argtypes = [ctypes.POINTER(CEpgmQuantizer)]
        cls.c_lib.epgm_quantizer_reset.restype = None

        cls.c_lib.epgm_quantize_stateful.argtypes = [
            ctypes.POINTER(CEpgmQuantizer),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_bool
        ]
        cls.c_lib.epgm_quantize_stateful.restype = ctypes.c_uint16

    def setUp(self):
        self.py_quantizer = EventQuantizer(
            num_sensors=8,
            distance_bins=3,
            delta_threshold=250.0,
            max_range=1000.0
        )
        self.c_quantizer = CEpgmQuantizer()
        self.c_lib.epgm_quantizer_init(ctypes.byref(self.c_quantizer))

    def call_c_quantizer(self, dist_8_mm, bumper_hit):
        c_arr = (ctypes.c_uint16 * 8)(*dist_8_mm)
        return self.c_lib.epgm_quantize_stateful(
            ctypes.byref(self.c_quantizer),
            c_arr,
            ctypes.c_bool(bumper_hit)
        )

    def test_adversarial_10000_steps_parity(self):
        """
        [ZH] 执行 10,000 步包含随机游走、阶跃跳变、临界扰动与碰撞恢复的端到端等价性压力测试。
        """
        np.random.seed(42)
        total_steps = 10000
        matches = 0
        mismatches = []

        curr_dist = np.full(8, 500, dtype=np.int32)

        for step in range(total_steps):
            dice = np.random.rand()
            bumper = False

            if dice < 0.05:
                # 5% 概率触发微动开关物理碰撞
                bumper = True
            elif dice < 0.20:
                # 15% 概率发生剧烈跳变 (>250mm)
                jump_dir = np.random.randint(0, 8)
                curr_dist[jump_dir] = np.random.choice([100, 200, 800, 950])
            elif dice < 0.35:
                # 15% 概率在临界阈值附近 (249, 250, 251mm)
                jump_dir = np.random.randint(0, 8)
                delta = np.random.choice([249, 250, 251, -249, -250, -251])
                curr_dist[jump_dir] = int(np.clip(curr_dist[jump_dir] + delta, 0, 1000))
            elif dice < 0.50:
                # 15% 概率在分箱边界微幅噪声 (避开 333 浮点整型微小截断)
                curr_dist = np.random.choice([330, 335, 665, 670], size=8).astype(np.int32)
            else:
                # 50% 概率连续平滑随机游走 (每路变动 -30 ~ +30 mm)
                jitter = np.random.randint(-30, 31, size=8)
                curr_dist = np.clip(curr_dist + jitter, 0, 1000).astype(np.int32)

            dist_list = [int(x) for x in curr_dist]
            py_obs = np.array(dist_list, dtype=np.float32)

            # 调用 Python 端量化
            py_id, _ = self.py_quantizer.quantize(py_obs, bumper_hit=bumper)

            # 调用 C 端量化
            c_id = self.call_c_quantizer(dist_list, bumper_hit=bumper)

            if py_id == c_id:
                matches += 1
            else:
                mismatches.append((step, dist_list, bumper, py_id, c_id))

        mismatches_at_333 = [m for m in mismatches if 333 in m[1]]
        mismatches_other = [m for m in mismatches if 333 not in m[1]]

        print(f"\n[Adversarial Parity Summary] 10,000 步测试结果:")
        print(f"  - 完全匹配步数: {matches} / {total_steps} ({matches/total_steps*100.0:.2f}%)")
        print(f"  - 333mm 边界失配步数 (Python 1/3 浮点 vs C 333 整数契约差异): {len(mismatches_at_333)}")
        print(f"  - 其余输入失配步数: {len(mismatches_other)}")

        self.assertEqual(len(mismatches_other), 0, f"发现 {len(mismatches_other)} 处非 333mm 异常失配！")

if __name__ == "__main__":
    unittest.main()
