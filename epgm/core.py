"""
[ZH] EPGM 核心数据结构与拓扑因果图定义。
[EN] EPGM core data structures and topological causal graph definitions.

数学原理 / Mathematical Background:
    EPGM 将具身智能体感知运动经验建模为动态有向多重图 G = (V, E)。
    EPGM models the sensorimotor experience of an embodied agent as a dynamic directed multigraph G = (V, E).
    - 节点 v in V: 对应于离散化的感觉事件基元 (Sensory Event Primitive e)。
    - 有向边 e = (v_i, a) -> v_j: 对应于因果转移假说 (Causal Transition Hypothesis)，
      表示在事件 v_i 下执行动作 a，模型预测环境将跃迁至事件 v_j，并带来期望回报 R。
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EPGMEdge:
    """
    [ZH] EPGM 有向因果预测边。
    [EN] EPGM directed causal predictive edge.

    数学属性 / Mathematical Attributes:
        P(v_j | v_i, a): 转移条件概率 / Conditional transition probability.
        R(v_i, a, v_j): 预期即时标量回报 / Expected immediate scalar reward.
        U(v_i, a, v_j): 贝叶斯不确定度与惊喜度度量 / Bayesian uncertainty and surprise metric.
    """
    source_event: int
    action: int
    target_event: int
    expected_reward: float = 0.0
    observation_count: int = 1
    confidence: float = 0.5
    uncertainty: float = 1.0
    expected_delay: float = 1.0
    last_active_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        [ZH] 序列化为字典格式（便于增量上传云端与存储）。
        [EN] Serialize to dictionary format (for incremental cloud sync and storage).
        """
        return {
            "source_event": int(self.source_event),
            "action": int(self.action),
            "target_event": int(self.target_event),
            "expected_reward": round(float(self.expected_reward), 4),
            "observation_count": int(self.observation_count),
            "confidence": round(float(self.confidence), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "expected_delay": round(float(self.expected_delay), 2),
            "last_active_step": int(self.last_active_step),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EPGMEdge":
        """
        [ZH] 从字典反序列化重构边对象。
        [EN] Reconstruct edge object from dictionary.
        """
        return cls(
            source_event=data["source_event"],
            action=data["action"],
            target_event=data["target_event"],
            expected_reward=data.get("expected_reward", 0.0),
            observation_count=data.get("observation_count", 1),
            confidence=data.get("confidence", 0.5),
            uncertainty=data.get("uncertainty", 1.0),
            expected_delay=data.get("expected_delay", 1.0),
            last_active_step=data.get("last_active_step", 0),
        )


@dataclass
class EPGMNode:
    """
    [ZH] EPGM 事件节点定义。
    [EN] EPGM event node definition.

    数学属性 / Mathematical Attributes:
        v_i: 离散事件 ID / Discrete event identifier.
        c_i: 原型特征向量 (感知聚类质心) / Prototype feature vector (sensory cluster centroid).
        h_i: 上下文阶数 (用于 POMDP 混叠消除) / Contextual order (for POMDP state-aliasing disambiguation).
    """
    event_id: int
    prototype_vector: Optional[np.ndarray] = None
    visit_count: int = 1
    context_order: int = 0
    created_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        [ZH] 序列化节点数据。
        [EN] Serialize node data.
        """
        return {
            "event_id": int(self.event_id),
            "prototype_vector": self.prototype_vector.tolist() if self.prototype_vector is not None else None,
            "visit_count": int(self.visit_count),
            "context_order": int(self.context_order),
            "created_step": int(self.created_step),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EPGMNode":
        """
        [ZH] 从字典反序列化重构节点。
        [EN] Reconstruct node from dictionary.
        """
        proto = data.get("prototype_vector")
        return cls(
            event_id=data["event_id"],
            prototype_vector=np.array(proto, dtype=np.float32) if proto is not None else None,
            visit_count=data.get("visit_count", 1),
            context_order=data.get("context_order", 0),
            created_step=data.get("created_step", 0),
        )


class EventPredictiveGraph:
    """
    [ZH] 事件预测图 (EPGM) 拓扑因果图主管理类。
    [EN] Event-predictive Graph Machine (EPGM) topological causal graph main manager.

    职责 / Responsibilities:
        1. 维护节点集合 V 与邻接预测边集合 E；
           Maintain node set V and adjacency predictive edge set E;
        2. 提供 O(1) 高效因果转移查询与边置信度检索；
           Provide O(1) efficient causal transition query and edge confidence retrieval;
        3. 支持轻量级图碎片序列化 (Graph Fragment Serialization)，用于端云网络同步。
           Support lightweight graph fragment serialization for edge-cloud network sync.
    """

    def __init__(self, action_dim: int = 5, learning_rate: float = 0.2, decay_rate: float = 0.001):
        """
        [ZH] 初始化 EPGM 图引擎。
        [EN] Initialize EPGM graph engine.

        Args:
            action_dim (int): 离散动作空间大小 / Size of discrete action space.
            learning_rate (float): 局部奖励与转移概率学习率 alpha / Local reward and transition learning rate alpha.
            decay_rate (float): 突触长期抑制衰减系数 lambda / Synaptic long-term depression decay factor lambda.
        """
        self.action_dim = action_dim
        self.lr = learning_rate
        self.decay_rate = decay_rate

        # [ZH] 节点注册表: event_id -> EPGMNode
        # [EN] Node registry: event_id -> EPGMNode
        self.nodes: Dict[int, EPGMNode] = {}

        # [ZH] 邻接边查找表: (source_event, action) -> List[EPGMEdge]
        # [EN] Adjacency edge lookup table: (source_event, action) -> List[EPGMEdge]
        self.adjacency: Dict[Tuple[int, int], List[EPGMEdge]] = {}

        # [ZH] 全局演化步数计数器
        # [EN] Global evolution step counter
        self.current_step: int = 0

    def add_node(self, event_id: int, prototype: Optional[np.ndarray] = None) -> EPGMNode:
        """
        [ZH] 注册或更新一个事件节点。
        [EN] Register or update an event node.
        """
        if event_id not in self.nodes:
            self.nodes[event_id] = EPGMNode(
                event_id=event_id,
                prototype_vector=prototype,
                visit_count=1,
                created_step=self.current_step,
            )
        else:
            self.nodes[event_id].visit_count += 1
            if prototype is not None and self.nodes[event_id].prototype_vector is not None:
                # [ZH] 平滑移动质心 / Smoothly move prototype centroid
                self.nodes[event_id].prototype_vector = 0.9 * self.nodes[event_id].prototype_vector + 0.1 * prototype
        return self.nodes[event_id]

    def get_predictions(self, source_event: int, action: int) -> List[EPGMEdge]:
        """
        [ZH] 查询在当前事件与动作下的所有可能后继预测边。
        [EN] Query all possible subsequent predictive edges given current event and action.
        """
        return self.adjacency.get((source_event, action), [])

    def get_best_prediction(self, source_event: int, action: int) -> Optional[EPGMEdge]:
        """
        [ZH] 获取置信度最高的单步预测边。
        [EN] Get the single predictive edge with highest confidence / observation count.
        """
        edges = self.get_predictions(source_event, action)
        if not edges:
            return None
        return max(edges, key=lambda e: (e.observation_count, e.confidence))

    def export_fragment(self, since_step: int = 0) -> Dict[str, Any]:
        """
        [ZH] 导出自指定步数以来的增量图碎片（用于边缘端打包回传云端）。
        [EN] Export incremental graph fragment since specified step (for edge-to-cloud telemetry).
        """
        recent_edges = []
        node_ids_in_edges = set()

        for edges in self.adjacency.values():
            for edge in edges:
                if edge.last_active_step >= since_step:
                    recent_edges.append(edge.to_dict())
                    node_ids_in_edges.add(edge.source_event)
                    node_ids_in_edges.add(edge.target_event)

        recent_nodes = [
            self.nodes[nid].to_dict()
            for nid in node_ids_in_edges
            if nid in self.nodes
        ]

        return {
            "meta": {
                "step": self.current_step,
                "since_step": since_step,
                "node_count": len(recent_nodes),
                "edge_count": len(recent_edges),
            },
            "nodes": recent_nodes,
            "edges": recent_edges,
        }

    def import_fragment(self, fragment: Dict[str, Any]) -> Tuple[int, int]:
        """
        [ZH] 融合导入外部图碎片（用于云端融合或先验图下发给机器人）。
        [EN] Import and fuse external graph fragment (for cloud aggregation or prior model distribution).

        Returns:
            Tuple[int, int]: (新增/更新节点数, 新增/更新边数) / (nodes_updated, edges_updated)
        """
        nodes_updated = 0
        edges_updated = 0

        # [ZH] 1. 合并节点 / 1. Merge nodes
        for node_data in fragment.get("nodes", []):
            nid = node_data["event_id"]
            if nid not in self.nodes:
                self.nodes[nid] = EPGMNode.from_dict(node_data)
                nodes_updated += 1
            else:
                self.nodes[nid].visit_count += node_data.get("visit_count", 1)

        # [ZH] 2. 合并因果边 / 2. Merge causal edges
        for edge_data in fragment.get("edges", []):
            s = edge_data["source_event"]
            a = edge_data["action"]
            t = edge_data["target_event"]
            r = edge_data.get("expected_reward", 0.0)
            cnt = edge_data.get("observation_count", 1)

            key = (s, a)
            if key not in self.adjacency:
                self.adjacency[key] = []

            # [ZH] 查找是否已存在同向目标边
            # [EN] Check if identical target edge already exists
            found = False
            for edge in self.adjacency[key]:
                if edge.target_event == t:
                    # [ZH] 加权平均融合回报与频次
                    # [EN] Weighted average fusion of reward and counts
                    total_cnt = edge.observation_count + cnt
                    edge.expected_reward = (
                        (edge.expected_reward * edge.observation_count + r * cnt) / total_cnt
                    )
                    edge.observation_count = total_cnt
                    edge.confidence = min(1.0, edge.confidence + 0.1)
                    edge.uncertainty = max(0.01, edge.uncertainty * 0.9)
                    edge.last_active_step = max(edge.last_active_step, edge_data.get("last_active_step", 0))
                    found = True
                    break

            if not found:
                new_edge = EPGMEdge.from_dict(edge_data)
                self.adjacency[key].append(new_edge)
                edges_updated += 1

        return nodes_updated, edges_updated
