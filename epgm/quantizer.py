"""
[ZH] 在线事件量化器 (Event Quantizer)。
[EN] Online Event Quantizer.

数学原理 / Mathematical Background:
    物理世界传感器输出连续流 S_t in R^k。
    Physical world sensors emit continuous streams S_t in R^k.
    量化器执行映射: Q: (S_t, Delta S_t, Bumper) -> e_t in E
    Quantizer executes mapping: Q: (S_t, Delta S_t, Bumper) -> e_t in E
    当感知特征跨越边界或导数突变 ||Delta S_t|| > epsilon 时，触发离散状态脉冲 (Discrete Event Pulse)。
"""

from typing import Tuple, List, Optional
import numpy as np


class EventQuantizer:
    """
    [ZH] 自适应空间感受野事件量化器。
    [EN] Adaptive spatial receptive field event quantizer.

    特点 / Features:
        1. 空间感受野量化 (Receptive Field Discretization): 将雷达读数划分为安全/接近/危险离散区间；
        2. 动态突变事件捕捉 (Dynamic Delta Pulse): 监测加速度与剧烈突变，捕获高阶动态特征；
        3. 硬件友好 (Hardware Friendly): 可直接以低复杂度查表法移植至 ESP32-S3 定点整数运算。
    """

    def __init__(
        self,
        num_sensors: int = 8,
        distance_bins: int = 3,
        delta_threshold: float = 0.25,
        max_range: float = 1.0,
    ):
        """
        [ZH] 初始化事件量化器。
        [EN] Initialize event quantizer.

        Args:
            num_sensors (int): 测距传感器数量（默认8向雷达） / Number of range sensors (default 8 directions).
            distance_bins (int): 每个传感器的离散距离分箱数 / Distance discretization bins per sensor.
            delta_threshold (float): 触发动态事件的一阶导数突变阈值 / First derivative threshold for dynamic events.
            max_range (float): 传感器归一化最大量程 / Normalized maximum range of sensors.
        """
        self.num_sensors = num_sensors
        self.distance_bins = distance_bins
        self.delta_threshold = delta_threshold
        self.max_range = max_range

        # [ZH] 上一步传感器观测缓存，用于计算差分
        # [EN] Previous sensor observation buffer for computing temporal difference
        self.last_observation: Optional[np.ndarray] = None

        # [ZH] 动态聚类原型表（如果开启自适应聚类模式）
        # [EN] Dynamic prototype table (for adaptive clustering mode)
        self.prototypes: List[np.ndarray] = []

    def reset(self) -> None:
        """
        [ZH] 重置历史缓存状态。
        [EN] Reset historical cache state.
        """
        self.last_observation = None

    def quantize(self, observation: np.ndarray, bumper_hit: bool = False) -> Tuple[int, np.ndarray]:
        """
        [ZH] 将当前连续观测映射为离散事件基元 ID。
        [EN] Quantize current continuous observation into a discrete event primitive ID.

        Args:
            observation (np.ndarray): 连续传感器向量，形状为 (num_sensors,)，取值 [0, max_range]。
                                     Continuous sensor vector of shape (num_sensors,), values in [0, max_range].
            bumper_hit (bool): 物理微动碰撞开关状态 (True 表示撞击，为绝对高优先级突发事件)。
                              Physical microswitch bumper state (True indicates collision, highest priority).

        Returns:
            Tuple[int, np.ndarray]: (离散事件 ID, 特征向量原型) / (discrete event ID, prototype feature vector)
        """
        obs = np.clip(observation, 0.0, self.max_range)

        # [ZH] 1. 物理撞击事件（优先级最高，专属事件 ID 0）
        # [EN] 1. Physical collision event (highest priority, reserved event ID 0)
        if bumper_hit:
            self.last_observation = obs
            return 0, obs

        # [ZH] 2. 空间局部感受野分箱 (Spatial Receptive Field Binning)
        # 将连续距离划分为: 0: 危险(<0.33), 1: 接近(0.33~0.66), 2: 安全(>0.66)
        # Discretize continuous distance into: 0: Danger (<0.33), 1: Near (0.33-0.66), 2: Safe (>0.66)
        bin_edges = np.linspace(0.0, self.max_range, self.distance_bins + 1)[1:-1]
        digitized = np.digitize(obs, bin_edges)  # 每个传感器取值 0, 1, 2

        # [ZH] 3. 计算显著区域聚合摘要 (Compact Sector Encoding)
        # 为防止 3^8 = 6561 导致稀疏图爆炸，将 8 方向聚合为 4 个主要象限 (前、后、左、右)
        # To prevent 3^8 = 6561 state explosion on microcontrollers, aggregate 8 rays into 4 sectors (F, B, L, R)
        # 前方: Ray 0, 1, 7; 右方: Ray 1, 2, 3; 后方: Ray 3, 4, 5; 左方: Ray 5, 6, 7
        front = min(digitized[0], digitized[1], digitized[7])
        right = min(digitized[1], digitized[2], digitized[3])
        back = min(digitized[3], digitized[4], digitized[5])
        left = min(digitized[5], digitized[6], digitized[7])

        # [ZH] 4. 计算哈希事件 ID (取值范围 1 ~ 3^4 = 81)
        # [EN] 4. Compute hashed event ID (range 1 ~ 3^4 = 81)
        spatial_event_id = 1 + (front * 27 + right * 9 + back * 3 + left)

        # [ZH] 5. 动态脉冲检测 (Dynamic Transition Pulse)
        # 若距离变化率急剧超过阈值，打上高位标志 (增加 100 偏置区分动态运动状态)
        # If rate of change exceeds threshold, tag with dynamic bit offset (+100)
        is_dynamic = False
        if self.last_observation is not None:
            delta = np.linalg.norm(obs - self.last_observation)
            if delta > self.delta_threshold:
                is_dynamic = True

        self.last_observation = obs

        final_event_id = spatial_event_id + (100 if is_dynamic else 0)
        return final_event_id, obs
