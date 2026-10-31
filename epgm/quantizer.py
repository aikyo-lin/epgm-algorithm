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
        [ZH] 将当前连续观测映射为离散事件基元 ID (严格对齐端侧 C 固件量化接口契约)。
        [EN] Quantize current continuous observation into a discrete event primitive ID (aligned with C firmware).

        数学原理与接口契约 / Mathematical Background & Interface Contract:
            1. 碰撞优先级: 物理微动开关 bumper_hit 为硬件真值，触发时事件 ID 强制归 0；
            2. 空间三等分离散分箱 (Spatial Bins):
               - 危险区间 (Bin 0): s < 1/3 (约 0.3333, C 端对应 d < 333mm)
               - 接近区间 (Bin 1): 1/3 <= s < 2/3 (约 0.3333 ~ 0.6667, C 端对应 333mm <= d < 667mm)
               - 安全区间 (Bin 2): s >= 2/3 (约 0.6667, C 端对应 d >= 667mm)
            3. 紧凑四象限空间哈希 (Compact Sector Encoding):
               - 前方 (Front): min(Ray 0, Ray 1, Ray 7)
               - 右方 (Right): min(Ray 1, Ray 2, Ray 3)
               - 后方 (Back):  min(Ray 3, Ray 4, Ray 5)
               - 左方 (Left):  min(Ray 5, Ray 6, Ray 7)
               - 静态空间事件 ID: 1 + (Front * 27 + Right * 9 + Back * 3 + Left) in [1, 81]
            4. 差分动态脉冲判定 (Dynamic Transition Pulse):
               - 判定条件: ||Delta s||_2 > delta_threshold (当阈值为 0.25 时，即 sum(Delta s_i^2) > 0.0625)
               - C 端定点等价条件: sum((d_{t,i} - d_{t-1,i})^2) > 62,500 (250^2)
               - 动态事件 ID: 静态空间事件 ID + 100 in [101, 181]

        Args:
            observation (np.ndarray): 连续传感器向量，形状为 (num_sensors,)，取值 [0, max_range]。
                                     Continuous sensor vector of shape (num_sensors,), values in [0, max_range].
            bumper_hit (bool): 物理微动碰撞开关状态 (True 表示撞击，为绝对高优先级突发事件)。
                               Physical microswitch bumper state (True indicates collision, highest priority).

        Returns:
            Tuple[int, np.ndarray]: (离散事件 ID [0: 碰撞, 1~81: 静态事件, 101~181: 动态事件], 特征向量原型)
        """
        # [ZH] 输入数值清洗与 NaN/Inf 防御：消除传感器故障带来的非法浮点数，防止污染有状态历史帧 (防范 CH-05 漏洞)
        # [EN] Input sanitization: clean NaN/Inf sensor values to prevent poisoning stateful history
        clean_obs = np.nan_to_num(observation, nan=0.0, posinf=self.max_range, neginf=0.0)
        obs = np.clip(clean_obs, 0.0, self.max_range).astype(np.float32)

        # [ZH] 1. 物理微动开关撞击事件（优先级最高，专属事件 ID 0）
        # [EN] 1. Physical collision microswitch event (highest priority, reserved event ID 0)
        if bumper_hit:
            self.last_observation = obs.copy()
            return 0, obs

        # [ZH] 2. 空间局部感受野分箱 (三等分边界: 1/3 与 2/3)
        # [EN] 2. Spatial receptive field binning (tercile boundaries: 1/3 and 2/3)
        if self.distance_bins == 3:
            bin_edges = np.array([self.max_range / 3.0, 2.0 * self.max_range / 3.0], dtype=np.float32)
        else:
            bin_edges = np.linspace(0.0, self.max_range, self.distance_bins + 1)[1:-1]

        digitized = np.digitize(obs, bin_edges)  # 每个传感器取值 0, 1, 2

        # [ZH] 3. 四象限显著区域聚合 (前、右、后、左)
        # [EN] 3. Four-sector aggregation (Front, Right, Back, Left)
        front = int(min(digitized[0], digitized[1], digitized[7]))
        right = int(min(digitized[1], digitized[2], digitized[3]))
        back  = int(min(digitized[3], digitized[4], digitized[5]))
        left  = int(min(digitized[5], digitized[6], digitized[7]))

        # [ZH] 4. 计算静态空间事件 ID (取值区间 1 ~ 81)
        # [EN] 4. Compute static spatial event ID (range 1 ~ 81)
        spatial_event_id = 1 + (front * 27 + right * 9 + back * 3 + left)

        # [ZH] 5. 差分动态脉冲检测 (Dynamic Transition Pulse)
        # 判定欧式距离变化量 ||Delta s||_2 > delta_threshold, 等价于 sum(Delta s_i^2) > delta_threshold^2
        is_dynamic = False
        if self.last_observation is not None:
            diff = obs - self.last_observation
            sum_sq = float(np.sum(diff * diff))
            thresh_sq = float(self.delta_threshold * self.delta_threshold)
            # 考虑微小浮点误差，严格判定大于门限
            if sum_sq > thresh_sq:
                is_dynamic = True

        self.last_observation = obs.copy()

        # 动态事件偏移 100 偏置，区间映射为 101 ~ 181
        final_event_id = spatial_event_id + (100 if is_dynamic else 0)
        return final_event_id, obs
