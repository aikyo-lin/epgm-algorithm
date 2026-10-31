"""
[ZH] EPGM 因果图推理与主动好奇心决策规划器。
[EN] EPGM Causal Graph Inference and Active Curiosity Decision Planner.

数学原理 / Mathematical Formulation:
    动作选择基于主动推理价值函数:
    Action selection is governed by active inference value function:
        a* = argmax_{a in A} [ Q_task(e_t, a) + beta * U_curiosity(e_t, a) ]
    其中:
    - Q_task: 基于因果预测链的预期任务累积回报 / Expected cumulative reward along causal prediction chains.
    - U_curiosity: 贝叶斯信息增益与不确定性好奇心激励 / Bayesian information gain & uncertainty curiosity bonus.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .core import EventPredictiveGraph, EPGMEdge


class GraphPlanner:
    """
    [ZH] 基于有界深度因果图搜索的实时决策器。
    [EN] Real-time decision maker based on bounded-depth causal graph search.
    """

    def __init__(
        self,
        graph: EventPredictiveGraph,
        action_dim: int = 5,
        gamma: float = 0.9,
        curiosity_weight: float = 0.4,
        planning_depth: int = 2,
    ):
        """
        [ZH] 初始化规划器。
        [EN] Initialize planner.

        Args:
            graph (EventPredictiveGraph): EPGM 拓扑因果图实例 / EPGM causal graph instance.
            action_dim (int): 动作空间维度 / Action space dimension.
            gamma (float): 时间贴现因子 / Temporal discount factor.
            curiosity_weight (float): 好奇心探索权重 beta / Curiosity exploration weight beta.
            planning_depth (int): 图前向预测展开深度 / Graph forward lookahead planning depth.
        """
        self.graph = graph
        self.action_dim = action_dim
        self.gamma = gamma
        self.curiosity_weight = curiosity_weight
        self.planning_depth = planning_depth

    def select_action(self, current_event: int, epsilon_greedy: float = 0.05) -> Tuple[int, Dict[str, float]]:
        """
        [ZH] 选择在当前事件状态下的最优动作。
        [EN] Select the optimal action given current event state.

        Args:
            current_event (int): 当前感知事件 ID / Current sensory event identifier.
            epsilon_greedy (float): 随机扰动率（打破完全平局） / Random perturbation rate (break symmetry).

        Returns:
            Tuple[int, Dict[str, float]]: (选定动作, 各动作预测价值打分字典) / (chosen_action, action_scores)
        """
        # [ZH] 小概率随机探索以打破死循环对称性
        # [EN] Small epsilon exploration to break symmetry and dead loops
        if np.random.rand() < epsilon_greedy:
            rand_action = np.random.randint(0, self.action_dim)
            return rand_action, {"random_chosen": rand_action}

        action_scores: Dict[int, float] = {}

        for a in range(self.action_dim):
            # [ZH] 计算该动作的前向预期价值
            # [EN] Compute forward anticipated value of this action
            val = self._evaluate_action_branch(current_event, a, depth=1)
            action_scores[a] = val

        # [ZH] 选取综合价值最高动作，若有多个并列最高则随机选取
        # [EN] Choose action with highest integrated score, break ties randomly
        max_score = max(action_scores.values())
        best_actions = [a for a, s in action_scores.items() if np.isclose(s, max_score, atol=1e-5)]
        chosen_action = int(np.random.choice(best_actions))

        return chosen_action, action_scores

    def _evaluate_action_branch(self, event_id: int, action: int, depth: int) -> float:
        """
        [ZH] 递归展开图预测计算分支期望值。
        [EN] Recursively unroll graph predictions to compute branch expected value.
        """
        predictions = self.graph.get_predictions(event_id, action)

        # [ZH] 1. 如果该动作从未在当前事件发生过 (未探索边)：赋予极高好奇心奖励鼓励建图
        # [EN] 1. If this action has never been executed at current event (unexplored edge): grant high curiosity bonus
        if not predictions:
            return self.curiosity_weight * 2.0

        # [ZH] 选取主导转移边 (Dominant transition edge)
        best_edge = max(predictions, key=lambda e: e.observation_count)

        # [ZH] 2. 任务即时预期回报 / Immediate expected task reward
        r_task = best_edge.expected_reward

        # [ZH] 若预测后继为碰撞事件 (event_id == 0)，给予严厉惩罚规避危险
        # [EN] If predicted target is collision event (event_id == 0), impose severe penalty
        if best_edge.target_event == 0:
            r_task -= 10.0

        # [ZH] 3. 边的不确定性好奇心奖励 / Edge uncertainty curiosity bonus
        # 观测越少，好奇心价值越高: U = 1 / sqrt(N + 1)
        r_curiosity = self.curiosity_weight / np.sqrt(best_edge.observation_count + 1)

        total_value = r_task + r_curiosity

        # [ZH] 4. 前向递归展望计算贝尔曼未来价值 / Forward lookahead computation
        if depth < self.planning_depth:
            next_event = best_edge.target_event
            future_vals = []
            for next_a in range(self.action_dim):
                next_val = self._evaluate_action_branch(next_event, next_a, depth + 1)
                future_vals.append(next_val)
            if future_vals:
                total_value += self.gamma * max(future_vals)

        return float(total_value)
