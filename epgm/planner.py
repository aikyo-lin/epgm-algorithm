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

from typing import Dict, List, Optional, Tuple, Any
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
        collision_conf_thresh: float = 0.5,
        collision_min_obs: int = 1,
        seed: Optional[int] = None,
    ):
        """
        [ZH] 初始化规划器。
        [EN] Initialize planner.

        架构设计与安全硬掩码机制 / Architecture Intent & Safety Masking:
            1. 集成有界深度因果图搜索与主动好奇心驱动机制；
            2. 引入安全硬掩码 (Hard Safety Masking): 当某动作已知以高置信度导致碰撞事件 (target_event == 0) 时，
               在利用模式赋予 -inf 强制剔除，在探索模式从候选集中剔除，根绝盲目撞墙；
            3. 配置碰撞置信度阈值 (collision_conf_thresh) 与最小观测样本数 (collision_min_obs)；
            4. 维护受控伪随机数生成器 (self.rng)，杜绝全局未控随机漂移。

        Args:
            graph (EventPredictiveGraph): EPGM 拓扑因果图实例 / EPGM causal graph instance.
            action_dim (int): 动作空间维度 / Action space dimension (默认 5: 0=停止, 1=前进, 2=后退, 3=左转, 4=右转).
            gamma (float): 时间贴现因子 / Temporal discount factor.
            curiosity_weight (float): 好奇心探索权重 beta / Curiosity exploration weight beta.
            planning_depth (int): 图前向预测展开深度 / Graph forward lookahead planning depth.
            collision_conf_thresh (float): 触发安全硬掩码的碰撞预测置信度门限 (默认 0.5).
            collision_min_obs (int): 触发安全硬掩码所需的最小碰撞观测频次 (默认 1 次确认即拦截).
            seed (Optional[int]): 规划器私有伪随机数种子，用于平局决断与安全探索采样.
        """
        self.graph = graph
        self.action_dim = action_dim
        self.gamma = gamma
        self.curiosity_weight = curiosity_weight
        self.planning_depth = planning_depth
        self.collision_conf_thresh = collision_conf_thresh
        self.collision_min_obs = collision_min_obs
        self.rng = np.random.default_rng(seed)

    def is_action_hazardous(
        self,
        event_id: int,
        action: int,
        conf_thresh: Optional[float] = None,
        min_obs: Optional[int] = None,
    ) -> bool:
        """
        [ZH] 判定在当前事件下执行指定动作是否存在高置信度碰撞危险。
        [EN] Determine if executing specified action under current event poses high-confidence collision hazard.

        判定逻辑 / Decision Logic:
            检索因果图中 (event_id, action) 对应的所有后继预测边。
            若存在任意一条预测转移指向碰撞事件 (target_event == 0)，
            且其置信度 >= conf_thresh (默认 0.5)，观测次数 >= min_obs (默认 1)，
            则判定该动作为危险动作 (Hazardous Action)，必须被安全硬掩码拦截。

        Args:
            event_id (int): 当前感知事件 ID / Current sensory event identifier.
            action (int): 待判定的候选动作 / Candidate action index.
            conf_thresh (Optional[float]): 覆盖默认阈值的碰撞置信度门限.
            min_obs (Optional[int]): 覆盖默认阈值的最小观测频次门限.

        Returns:
            bool: True 表示存在高置信度撞墙风险 (需屏蔽)，False 表示相对安全或未探索.
        """
        c_thresh = self.collision_conf_thresh if conf_thresh is None else conf_thresh
        m_obs = self.collision_min_obs if min_obs is None else min_obs

        predictions = self.graph.get_predictions(event_id, action)
        for edge in predictions:
            # 物理碰撞为绝对危险 (target_event == 0)
            if edge.target_event == 0 and edge.confidence >= c_thresh and edge.observation_count >= m_obs:
                return True
        return False

    def select_action(
        self,
        current_event: int,
        epsilon_greedy: float = 0.05,
        enable_safety_mask: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        [ZH] 选择在当前事件状态下的最优动作（支持安全硬掩码与受控探索）。
        [EN] Select optimal action given current event (supporting hard safety mask and controlled exploration).

        核心工作流程 / Core Workflow:
            1. 安全评估 (Hazard Check): 扫描所有 candidate actions，通过 is_action_hazardous 识别碰撞分支；
            2. 掩码构建 (Mask Construction): 若启用 enable_safety_mask，将危险动作移出 safe_actions 并记录到 masked_actions；
            3. 死角保底 (All-Masked Fallback): 若全部动作均被掩码，执行紧急避险动作 (优先后退 a=2 或停止 a=0)；
            4. 受控探索 (Safe Exploration): 若触发 epsilon_greedy 探索，仅在 safe_actions 内部均匀随机采样，绝不盲撞；
            5. 价值决策 (Value Decision): 计算安全动作的前向展开价值，被掩码动作赋予 -inf，在最高评分安全分支中决断。

        Args:
            current_event (int): 当前感知事件 ID / Current sensory event identifier.
            epsilon_greedy (float): 随机扰动探索率 / Random exploration rate.
            enable_safety_mask (bool): 是否启用安全硬掩码 (默认为 True 开启保护).
            seed (Optional[int]): 单步决策种子 (若提供则局部覆盖受控随机流).

        Returns:
            Tuple[int, Dict[str, Any]]: (选定动作, 附加决策信息字典包含 scores, safe_actions, masked_actions 等).
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # [ZH] 1. 计算候选动作的危险性并构建安全集合
        # [EN] 1. Evaluate candidate actions for hazards and construct safe subset
        masked_actions = []
        if enable_safety_mask:
            for a in range(self.action_dim):
                if self.is_action_hazardous(current_event, a):
                    masked_actions.append(a)

        safe_actions = [a for a in range(self.action_dim) if a not in masked_actions]

        # [ZH] 2. 全掩码极端边界保底处理 (All-Masked Corner Case Fallback)
        # 若所有动作均预测碰撞（例如陷入凹槽死角），触发紧急避险保底
        fallback_triggered = False
        if not safe_actions:
            fallback_triggered = True
            # 优先选择后退避险 (a=2)；若动作空间无后退则选择停止 (a=0)
            fallback_action = 2 if 2 < self.action_dim else 0
            safe_actions = [fallback_action]

        # [ZH] 3. 预先计算所有动作的预测价值（被硬掩码的动作赋 -inf 负无穷）
        action_scores: Dict[int, float] = {}
        for a in range(self.action_dim):
            if a in masked_actions and not fallback_triggered:
                action_scores[a] = -float('inf')
            else:
                action_scores[a] = self._evaluate_action_branch(current_event, a, depth=1)

        # [ZH] 4. 受控安全探索 (Safe Epsilon-Greedy Exploration)
        # 严格限定在 safe_actions 内部采样，确保 100% 拦截明知撞墙的危险动作
        if rng.uniform(0.0, 1.0) < epsilon_greedy:
            rand_action = int(rng.choice(safe_actions))
            info = {
                **action_scores,
                "scores": action_scores,
                "safe_actions": safe_actions,
                "masked_actions": masked_actions,
                "fallback_triggered": fallback_triggered,
                "random_chosen": rand_action,
            }
            return rand_action, info

        # [ZH] 5. 纯利用模式: 在安全集合中选取最高分动作 (平局时通过受控 rng 决断)
        valid_scores = {a: action_scores[a] for a in safe_actions}
        max_score = max(valid_scores.values())
        best_actions = [a for a, s in valid_scores.items() if np.isclose(s, max_score, atol=1e-5)]
        chosen_action = int(rng.choice(best_actions))

        info = {
            **action_scores,
            "scores": action_scores,
            "safe_actions": safe_actions,
            "masked_actions": masked_actions,
            "fallback_triggered": fallback_triggered,
        }
        return chosen_action, info

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
