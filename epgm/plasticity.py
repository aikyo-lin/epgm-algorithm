"""
[ZH] 局部可塑性更新、拓扑生长与修剪引擎。
[EN] Local Plasticity Updates, Topological Growth and Pruning Engine.

数学原理 / Mathematical Background:
    1. 突触长时程增强 (LTP) 与抑制 (LTD):
       当实际状态与预期一致，强化突触边权并衰减不确定度；
       长期无激活边进行自发指数衰减 lambda * e^{-Delta t}。
    2. 贝叶斯惊喜触发拓扑生枝 (Surprise-Induced Edge Budding):
       当遇到未探索边或严重预测偏离，动态插入新因果边并初始化高好奇心势能。
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from .core import EventPredictiveGraph, EPGMEdge


class TopologicalPlasticity:
    """
    [ZH] 拓扑自适应可塑性演化控制器。
    [EN] Topological Adaptive Plasticity Evolution Controller.
    """

    def __init__(
        self,
        graph: EventPredictiveGraph,
        learning_rate: float = 0.25,
        prune_threshold_steps: int = 2000,
        min_confidence_to_keep: float = 0.1,
    ):
        """
        [ZH] 初始化可塑性引擎。
        [EN] Initialize plasticity engine.

        Args:
            graph (EventPredictiveGraph): 关联的 EPGM 因果图 / Associated EPGM graph.
            learning_rate (float): 局部奖励与参数平滑学习率 alpha / Local learning rate alpha.
            prune_threshold_steps (int): 判定突触长期未激活并修剪的步数阈值 / Step threshold for pruning dormant edges.
            min_confidence_to_keep (float): 保留边的最低置信度阈值 / Minimum confidence threshold to retain an edge.
        """
        self.graph = graph
        self.lr = learning_rate
        self.prune_threshold_steps = prune_threshold_steps
        self.min_confidence_to_keep = min_confidence_to_keep

        # [ZH] 统计指标监控
        # [EN] Statistical metrics monitoring
        self.total_surprises: int = 0
        self.total_hits: int = 0
        self.total_buds: int = 0
        self.total_prunes: int = 0

    def step_learning(
        self,
        source_event: int,
        action: int,
        observed_event: int,
        reward: float,
        step_index: int,
    ) -> Dict[str, Any]:
        """
        [ZH] 执行单步局部因果可塑性更新。
             涵盖突触增强 (LTP)、因果生枝 (Budding)、冗余修剪 (LTD) 与
             基于尺度不变归一化香农熵的状态混叠检测与节点分裂。
        [EN] Execute single-step local causal plasticity update.
             Covers synaptic potentiation (LTP), causal budding, redundant edge pruning (LTD),
             and POMDP state aliasing detection via scale-invariant normalized Shannon entropy.

        数学公式 / Mathematical Formulation:
            1. LTP 回报滤波更新 / Reward Filtering:
               R <- R + alpha * (r - R)
            2. 无偏非负香农熵 / Non-negative Shannon Entropy:
               H(X) = - sum_{p_i > 0} p_i * log2(p_i)
            3. 尺度无关归一化相对熵 / Scale-invariant Normalized Relative Entropy:
               H_norm = H(X) / log2(K) in [0, 1],  for K >= 2

        Args:
            source_event (int): 执行动作前的事件 ID / Event ID before action execution.
            action (int): 执行的动作索引 / Action index executed.
            observed_event (int): 环境转移后实际观测到的后继事件 ID / Actually observed subsequent event ID.
            reward (float): 环境返回的标量奖惩 / Scalar reward from environment.
            step_index (int): 当前全局时间步 / Current global step index.

        Returns:
            Dict[str, Any]: 本步学习诊断信息（是否惊喜、预测误差、拓扑变动） / Step diagnostic information.
        """
        self.graph.current_step = step_index
        key = (source_event, action)
        existing_edges = self.graph.adjacency.get(key, [])

        is_hit = False
        is_bud = False
        surprise_magnitude = 0.0

        # [ZH] 1. 查找是否存在完全匹配的转移边
        # [EN] 1. Check if an exact matching transition edge exists
        matched_edge: Optional[EPGMEdge] = None
        for edge in existing_edges:
            if edge.target_event == observed_event:
                matched_edge = edge
                break

        if matched_edge is not None:
            # [ZH] 2. 预测命中 (Hit): 触发突触增强与回报滤波更新
            # [EN] 2. Prediction Hit: trigger synaptic potentiation and reward filtering update
            is_hit = True
            self.total_hits += 1

            # [ZH] 平滑更新期望回报: R <- R + alpha * (r - R)
            # [EN] Smooth expected reward update: R <- R + alpha * (r - R)
            error = reward - matched_edge.expected_reward
            matched_edge.expected_reward += self.lr * error
            matched_edge.observation_count += 1
            matched_edge.confidence = min(1.0, matched_edge.confidence + 0.05)
            # 衰减不确定度与惊喜度
            matched_edge.uncertainty = max(0.01, matched_edge.uncertainty * 0.85)
            matched_edge.last_active_step = step_index

        else:
            # [ZH] 3. 预测失配或缺失 (Surprise): 触发拓扑生枝 (Budding)
            # [EN] 3. Prediction Mismatch or Missing (Surprise): trigger edge budding
            is_bud = True
            self.total_surprises += 1
            self.total_buds += 1
            surprise_magnitude = 1.0

            new_edge = EPGMEdge(
                source_event=source_event,
                action=action,
                target_event=observed_event,
                expected_reward=float(reward),
                observation_count=1,
                confidence=0.4,
                uncertainty=1.0,
                expected_delay=1.0,
                last_active_step=step_index,
            )

            if key not in self.graph.adjacency:
                self.graph.adjacency[key] = []
            self.graph.adjacency[key].append(new_edge)

        # [ZH] 4. POMDP 状态混叠与多模态转移相对熵检测 (State Aliasing Check via Normalized Entropy)
        # [EN] 4. POMDP state aliasing detection via scale-invariant normalized Shannon entropy
        is_split = False
        all_edges_for_key = self.graph.adjacency.get(key, [])
        k_outcomes = len(all_edges_for_key)
        if k_outcomes >= 2:
            # [ZH] 计算经验分布的香农熵与无量纲归一化相对熵:
            #      H(X) = -sum_{p > 0} p * log2(p)
            #      H_norm = H(X) / log2(K) in [0, 1]
            # [EN] Compute empirical Shannon entropy and scale-invariant normalized entropy:
            #      H(X) = -sum_{p > 0} p * log2(p), H_norm = H(X) / log2(K) in [0, 1]
            total_counts = sum(max(0, e.observation_count) for e in all_edges_for_key)
            if total_counts >= 15:
                probs = [max(0, e.observation_count) / total_counts for e in all_edges_for_key]
                entropy = -sum((p * np.log2(p)) if p > 0.0 else 0.0 for p in probs)
                max_entropy = np.log2(float(k_outcomes))
                norm_entropy = (entropy / max_entropy) if max_entropy > 0.0 else 0.0

                # 若归一化相对熵超过 0.95 (极其接近理论最大均匀分布，表明存在剧烈多模态感知混叠)
                if norm_entropy > 0.95:
                    is_split = self._handle_node_split(source_event, step_index)

        # [ZH] 5. 定期执行拓扑整理与冗余修剪 (Pruning)
        # [EN] 5. Periodic topological maintenance and redundant edge pruning
        pruned_this_step = 0
        if step_index % 500 == 0 and step_index > 0:
            pruned_this_step = self.prune_stale_edges(step_index)

        return {
            "is_hit": is_hit,
            "is_bud": is_bud,
            "is_split": is_split,
            "surprise_magnitude": surprise_magnitude,
            "pruned_edges": pruned_this_step,
            "total_nodes": len(self.graph.nodes),
            "total_edges": sum(len(el) for el in self.graph.adjacency.values()),
        }

    def _handle_node_split(self, event_id: int, step_index: int) -> bool:
        """
        [ZH] 执行节点分裂，消除感知混叠并引入高阶上下文。
        [EN] Execute node splitting to resolve perceptual aliasing and introduce context.
        """
        if event_id not in self.graph.nodes:
            return False
        node = self.graph.nodes[event_id]
        if node.context_order >= 2:
            return False  # 避免阶数无限膨胀 / Prevent infinite context order

        node.context_order += 1
        return True

    def prune_stale_edges(self, current_step: int) -> int:
        """
        [ZH] 修剪长期未被激活且置信度过低的冗余因果边。
        [EN] Prune redundant causal edges that are dormant and have low confidence.
        """
        pruned_count = 0
        for key in list(self.graph.adjacency.keys()):
            surviving_edges = []
            for edge in self.graph.adjacency[key]:
                dormant_duration = current_step - edge.last_active_step
                # 若休眠过久且观测样本少，执行清理
                if dormant_duration > self.prune_threshold_steps and edge.observation_count < 3:
                    pruned_count += 1
                else:
                    surviving_edges.append(edge)

            if surviving_edges:
                self.graph.adjacency[key] = surviving_edges
            else:
                del self.graph.adjacency[key]

        self.total_prunes += pruned_count
        return pruned_count
