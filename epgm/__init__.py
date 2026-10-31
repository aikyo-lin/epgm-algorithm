"""
[ZH] EPGM (Event-predictive Graph Machine) 核心算法包。
[EN] EPGM (Event-predictive Graph Machine) core algorithmic package.

提供非梯度拓扑因果图的事件量化、图结构表示、主动好奇心推理与局部可塑性演化。
Provides non-gradient topological causal graph event quantization, graph representation,
active curiosity-driven inference, and local plasticity evolution.
"""

from .core import EventPredictiveGraph, EPGMEdge, EPGMNode
from .quantizer import EventQuantizer
from .planner import GraphPlanner
from .plasticity import TopologicalPlasticity

__all__ = [
    "EventPredictiveGraph",
    "EPGMEdge",
    "EPGMNode",
    "EventQuantizer",
    "GraphPlanner",
    "TopologicalPlasticity",
]

__version__ = "0.1.0"
