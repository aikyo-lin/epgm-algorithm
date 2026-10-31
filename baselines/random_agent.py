"""
[ZH] 随机探索智能体基线 (Random Agent Baseline)。
[EN] Random Agent Baseline.
"""

import numpy as np


class RandomAgent:
    """
    [ZH] 随机动作选择基线，用于验证探索效率对照。
    [EN] Random action selection baseline for verifying exploration efficiency.
    """

    def __init__(self, action_dim: int = 5):
        self.action_dim = action_dim

    def select_action(self, observation: np.ndarray) -> int:
        return int(np.random.randint(0, self.action_dim))

    def learn(self, *args, **kwargs) -> None:
        pass  # 无学习过程 / No learning process
