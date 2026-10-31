"""
[ZH] 2D 避障寻标沙盒基准评测运行脚本 (EPGM vs 基线)。
[EN] 2D obstacle avoidance and navigation benchmark runner (EPGM vs baselines).
"""

import sys
import os
import json
import time
import numpy as np

# 将算法根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.arena2d import Arena2DEnv
from epgm.core import EventPredictiveGraph
from epgm.quantizer import EventQuantizer
from epgm.planner import GraphPlanner
from epgm.plasticity import TopologicalPlasticity


def run_epgm_experiment(episodes: int = 30, max_steps_per_episode: int = 250):
    """
    [ZH] 运行 EPGM 在 2D 沙盒中的在线自组织学习实验。
    [EN] Run EPGM online self-organizing learning experiment in 2D sandbox.
    """
    print("=" * 60)
    print("[ZH] 启动 EPGM 非梯度拓扑学习基准实验")
    print("[EN] Starting EPGM Non-gradient Topological Learning Benchmark")
    print("=" * 60)

    env = Arena2DEnv(max_steps=max_steps_per_episode)
    quantizer = EventQuantizer(num_sensors=8, distance_bins=3)
    graph = EventPredictiveGraph(action_dim=5, learning_rate=0.3)
    planner = GraphPlanner(graph, action_dim=5, gamma=0.85, curiosity_weight=0.5, planning_depth=2)
    plasticity = TopologicalPlasticity(graph, learning_rate=0.3)

    global_step = 0
    history = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        quantizer.reset()
        e_id, proto = quantizer.quantize(obs, bumper_hit=info["bumper_hit"])
        graph.add_node(e_id, proto)

        total_reward = 0.0
        collisions = 0
        hits = 0
        surprises = 0

        for st in range(max_steps_per_episode):
            global_step += 1

            # [ZH] 规划器决策动作 / Planner action decision
            action, scores = planner.select_action(e_id, epsilon_greedy=0.05)

            # [ZH] 环境执行 / Environment step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward

            if next_info["bumper_hit"]:
                collisions += 1

            # [ZH] 观测事件量化 / Next event quantization
            next_e_id, next_proto = quantizer.quantize(next_obs, bumper_hit=next_info["bumper_hit"])
            graph.add_node(next_e_id, next_proto)

            # [ZH] 局部可塑性更新 / Local plasticity learning
            diag = plasticity.step_learning(e_id, action, next_e_id, reward, global_step)
            if diag["is_hit"]:
                hits += 1
            else:
                surprises += 1

            e_id = next_e_id

            if terminated or truncated:
                break

        node_count = len(graph.nodes)
        edge_count = sum(len(el) for el in graph.adjacency.values())
        goal_status = "SUCCESS (抵达目标)" if next_info["goal_reached"] else "TIMEOUT (超时)"

        print(
            f"Episode {ep:02d}/{episodes:02d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Steps: {st + 1:3d} | "
            f"Hits/Surprises: {hits:2d}/{surprises:2d} | "
            f"Nodes/Edges: {node_count:2d}/{edge_count:3d} | "
            f"Collisions: {collisions:2d} | "
            f"Status: {goal_status}"
        )

        history.append({
            "episode": ep,
            "reward": total_reward,
            "steps": st + 1,
            "collisions": collisions,
            "nodes": node_count,
            "edges": edge_count,
            "goal_reached": next_info["goal_reached"],
        })

    # [ZH] 保存实验结果与导出的图模型碎片
    # [EN] Save experiment results and exported graph model fragment
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir, "epgm_benchmark_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    graph_fragment = graph.export_fragment()
    fragment_file = os.path.join(output_dir, "epgm_graph_model.json")
    with open(fragment_file, "w", encoding="utf-8") as f:
        json.dump(graph_fragment, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[ZH] 评测数据已保存至: {result_file}")
    print(f"[ZH] 生成的 EPGM 因果拓扑图已保存至: {fragment_file}")
    print(f"[ZH] 累计构建事件节点: {len(graph.nodes)}, 拓扑预测边: {sum(len(el) for el in graph.adjacency.values())}")
    print("=" * 60)


if __name__ == "__main__":
    run_epgm_experiment(episodes=20, max_steps_per_episode=200)
