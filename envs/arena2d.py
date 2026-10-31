"""
[ZH] 2D 连续空间机器人避障与导航沙盒环境 (基于 Gymnasium 标准规范)。
[EN] 2D continuous space robot obstacle avoidance & navigation sandbox environment (Gymnasium compliant).

物理与动力学说明 / Physics & Kinematics:
    小车采用双轮差速运动学模型 (Differential Drive Kinematics):
        x_{t+1} = x_t + v * cos(theta_t) * dt
        y_{t+1} = y_t + v * sin(theta_t) * dt
        theta_{t+1} = theta_t + omega * dt
    配备 8 方向等间隔射线测距雷达 (8-ray distance sensor) 与物理碰撞检测传感器。
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Arena2DEnv(gym.Env):
    """
    [ZH] 兼容 Gymnasium 的 2D 避障寻标实验沙盒。
    [EN] Gymnasium-compatible 2D obstacle avoidance & goal-seeking sandbox.

    动作空间 / Action Space:
        0: 停止 / Stop
        1: 前进 / Move Forward
        2: 后退 / Move Backward
        3: 左转 / Turn Left
        4: 右转 / Turn Right

    观测空间 / Observation Space:
        Box(0.0, 1.0, shape=(8,), float32): 8 个方向测距归一化距离 (0: 贴身障碍, 1: 开阔无碍)。
        Normalized distances from 8 radial rays (0: contact obstacle, 1: completely clear).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # [ZH] 动作与观测空间定义 / Action and Observation spaces definition
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # [ZH] 物理场地规格 (10.0m x 10.0m) / Arena physical dimensions
        self.arena_width = 10.0
        self.arena_height = 10.0
        self.robot_radius = 0.25
        self.max_ray_range = 3.5  # 最大测距距离 (米) / Maximum sensor range (meters)

        # [ZH] 障碍物列表 (x, y, radius)
        # [EN] Obstacle list (x, y, radius)
        self.obstacles: List[Tuple[float, float, float]] = [
            (2.5, 2.5, 0.8),
            (7.5, 2.5, 0.8),
            (2.5, 7.5, 0.8),
            (7.5, 7.5, 0.8),
            (5.0, 5.0, 1.2),  # 中心大障碍物 / Center large pillar
        ]

        # [ZH] 目标区域坐标与半径
        # [EN] Goal target location and radius
        self.goal_pos = np.array([8.5, 8.5], dtype=np.float32)
        self.goal_radius = 0.6

        # [ZH] 智能体本体状态: [x, y, theta]
        # [EN] Robot physical state: [x, y, theta]
        self.robot_pos = np.array([1.5, 1.5], dtype=np.float32)
        self.robot_theta = 0.0  # 弧度 / Radians

        # [ZH] 物理运动参数 / Motion parameters
        self.linear_speed = 0.25   # 前进线速度 (m/step)
        self.angular_speed = 0.35  # 转弯角速度 (rad/step)

        # [ZH] Pygame 渲染窗口句柄
        # [EN] Pygame rendering window handle
        self.screen = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        [ZH] 重置环境至初始状态，严格支持受控随机种子以消除测试方差。
        [EN] Reset environment to initial state, strictly supporting controlled random seeds to eliminate variance.

        架构设计与确定性保障 / Architecture Intent & Determinism:
            1. 基类 super().reset(seed=seed) 会初始化 Gymnasium 独立的伪随机数生成器 (self.np_random)；
            2. 彻底移除全局静态 np.random 调用，使用环境私有 self.np_random 生成初始位置与朝向；
            3. 支持通过 options 参数显式指定初始位置 ("init_pos") 与朝向角 ("init_theta")，方便基准评测与极端几何用例；
            4. 配合全确定性的运动学 step() 与解析光线投射，确保相同 seed 下整条状态与回报轨迹方差严格为 0。

        Args:
            seed (Optional[int]): 伪随机数发生器种子，若传入固定整数则完全复现初始位姿。
            options (Optional[Dict[str, Any]]): 扩展配置字典，支持:
                - "init_pos": List[float] 或 np.ndarray，小车初始坐标 [x, y]；
                - "init_theta": float，小车初始朝向角 (弧度)。

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (初始归一化观测向量 [8维], 状态附加信息字典)。
        """
        super().reset(seed=seed)
        self.current_step = 0

        # [ZH] 初始位姿设定: 优先读取 options 显式配置；若无配置则通过受控随机流 self.np_random 在安全区域采样
        # [EN] Initial pose: preferentially use options config; otherwise sample from safe zone via self.np_random
        if options is not None and "init_pos" in options:
            self.robot_pos = np.array(options["init_pos"], dtype=np.float32)
        else:
            pos_x = 1.2 + float(self.np_random.uniform(-0.3, 0.3))
            pos_y = 1.2 + float(self.np_random.uniform(-0.3, 0.3))
            self.robot_pos = np.array([pos_x, pos_y], dtype=np.float32)

        if options is not None and "init_theta" in options:
            self.robot_theta = float(options["init_theta"])
        else:
            self.robot_theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        obs, bumper = self._get_observation()
        info = {
            "bumper_hit": bumper,
            "goal_reached": False,
            "position": self.robot_pos.copy(),
            "orientation": self.robot_theta,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        [ZH] 执行一步物理仿真。
        [EN] Execute one step of physical simulation.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        v = 0.0
        w = 0.0

        if action == 1:    # 前进 / Forward
            v = self.linear_speed
        elif action == 2:  # 后退 / Backward
            v = -self.linear_speed * 0.5
        elif action == 3:  # 左转 / Turn Left
            w = self.angular_speed
        elif action == 4:  # 右转 / Turn Right
            w = -self.angular_speed

        # [ZH] 运动学更新
        # [EN] Kinematics update
        self.robot_theta = (self.robot_theta + w) % (2.0 * np.pi)
        new_pos = self.robot_pos + np.array([v * np.cos(self.robot_theta), v * np.sin(self.robot_theta)], dtype=np.float32)

        # [ZH] 碰撞检测与边界约束
        # [EN] Collision detection and boundary constraints
        bumper_hit = False

        # 1. 检查边界碰撞 / Boundary check
        if (
            new_pos[0] - self.robot_radius < 0.0
            or new_pos[0] + self.robot_radius > self.arena_width
            or new_pos[1] - self.robot_radius < 0.0
            or new_pos[1] + self.robot_radius > self.arena_height
        ):
            bumper_hit = True

        # 2. 检查障碍物圆体碰撞 / Circle obstacle check
        if not bumper_hit:
            for ox, oy, orad in self.obstacles:
                dist = np.linalg.norm(new_pos - np.array([ox, oy]))
                if dist < (self.robot_radius + orad):
                    bumper_hit = True
                    break

        if bumper_hit:
            # 碰撞时位置回退，产生反弹阻碍
            reward = -5.0
            terminated = False
        else:
            self.robot_pos = new_pos
            # 基础步进耗费 (鼓励高效探索)
            reward = -0.02
            terminated = False

        # [ZH] 检查是否抵达目标
        # [EN] Check if goal reached
        dist_to_goal = float(np.linalg.norm(self.robot_pos - self.goal_pos))
        goal_reached = dist_to_goal < (self.robot_radius + self.goal_radius)

        if goal_reached:
            reward += 15.0
            terminated = True

        # 超时截断
        truncated = self.current_step >= self.max_steps

        obs, _ = self._get_observation()
        info = {
            "bumper_hit": bumper_hit,
            "goal_reached": goal_reached,
            "dist_to_goal": dist_to_goal,
            "position": self.robot_pos.copy(),
            "orientation": self.robot_theta,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Tuple[np.ndarray, bool]:
        """
        [ZH] 执行 8 向光线投射 (Raycasting) 测距。
        [EN] Perform 8-ray casting distance measurement.
        """
        rays = 8
        obs = np.ones(rays, dtype=np.float32)

        for i in range(rays):
            angle = self.robot_theta + (i * 2.0 * np.pi / rays)
            ray_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

            min_dist = self.max_ray_range

            # 1. 与 4 个墙面求交 / Intersect with 4 walls
            if ray_dir[0] > 1e-4:
                d = (self.arena_width - self.robot_pos[0]) / ray_dir[0]
                if d > 0:
                    min_dist = min(min_dist, d)
            elif ray_dir[0] < -1e-4:
                d = (0.0 - self.robot_pos[0]) / ray_dir[0]
                if d > 0:
                    min_dist = min(min_dist, d)

            if ray_dir[1] > 1e-4:
                d = (self.arena_height - self.robot_pos[1]) / ray_dir[1]
                if d > 0:
                    min_dist = min(min_dist, d)
            elif ray_dir[1] < -1e-4:
                d = (0.0 - self.robot_pos[1]) / ray_dir[1]
                if d > 0:
                    min_dist = min(min_dist, d)

            # 2. 与各圆形障碍物求交 / Intersect with circular obstacles
            for ox, oy, orad in self.obstacles:
                oc = np.array([ox, oy], dtype=np.float32) - self.robot_pos
                tca = np.dot(oc, ray_dir)
                if tca > 0:
                    d2 = np.dot(oc, oc) - tca * tca
                    if d2 <= orad * orad:
                        thc = np.sqrt(orad * orad - d2)
                        d_hit = tca - thc
                        if 0 < d_hit < min_dist:
                            min_dist = d_hit

            # 归一化测距值在 [0, 1] 之间
            obs[i] = float(np.clip(min_dist / self.max_ray_range, 0.0, 1.0))

        # 检查本体半径内是否存在碰撞
        bumper_hit = any(obs[i] * self.max_ray_range <= self.robot_radius + 0.05 for i in range(rays))
        return obs, bumper_hit
