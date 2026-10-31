"""
[ZH] 经典深度 Q 网络基线 (DQN Baseline)。
[EN] Classic Deep Q-Network Baseline (DQN Baseline).

用于与非梯度 EPGM 在收敛样本效率、计算复杂度与探索能力上进行定量对标。
Used for quantitative benchmarking against non-gradient EPGM on sample efficiency, computational complexity and exploration.
"""

from typing import Tuple, List
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    [ZH] 2 层 MLP Q 网络逼近器。
    [EN] 2-layer MLP Q-network approximator.
    """
    def __init__(self, state_dim: int = 8, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    [ZH] 标准 DQN 智能体。
    [EN] Standard DQN Agent.
    """
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values).item())

    def store_transition(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        self.memory.append((s, a, r, s_next, done))

    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t = torch.FloatTensor(np.array(states)).to(self.device)
        a_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        d_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_eval = self.q_net(s_t).gather(1, a_t)
        with torch.no_grad():
            q_next = self.target_net(ns_t).max(1)[0].unsqueeze(1)
            q_target = r_t + (1.0 - d_t) * self.gamma * q_next

        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
