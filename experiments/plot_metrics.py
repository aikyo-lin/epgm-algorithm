"""
[ZH] 实验数据指标可视化绘图脚本。
[EN] Experiment metrics visualization and plotting script.
"""

import os
import json
import matplotlib.pyplot as plt


def plot_benchmark_results(json_path: str, save_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = [d["episode"] for d in data]
    rewards = [d["reward"] for d in data]
    collisions = [d["collisions"] for d in data]
    nodes = [d["nodes"] for d in data]
    edges = [d["edges"] for d in data]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("EPGM 2D Maze Benchmark Learning Dynamics", fontsize=14, fontweight="bold")

    # 1. 奖励曲线 / Reward
    axs[0, 0].plot(episodes, rewards, color="#2563eb", marker="o", linewidth=2)
    axs[0, 0].set_title("Episode Reward (Higher is Better)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. 碰撞次数 / Collisions
    axs[0, 1].plot(episodes, collisions, color="#dc2626", marker="s", linewidth=2)
    axs[0, 1].set_title("Collisions per Episode (Lower is Better)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Bumper Collisions")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. 节点增长 / Nodes Growth
    axs[1, 0].plot(episodes, nodes, color="#16a34a", marker="^", linewidth=2)
    axs[1, 0].set_title("Graph Nodes Discovery (Self-Organizing V)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Event Nodes")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. 因果边演化 / Edges Growth & Pruning
    axs[1, 1].plot(episodes, edges, color="#9333ea", marker="d", linewidth=2)
    axs[1, 1].set_title("Causal Edges (Growth & Plasticity Pruning E)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Predictive Edges")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[ZH] 评测指标趋势图已保存至: {save_path}")


if __name__ == "__main__":
    res_path = os.path.join(os.path.dirname(__file__), "results", "epgm_benchmark_results.json")
    out_img = os.path.join(os.path.dirname(__file__), "results", "learning_curves.png")
    if os.path.exists(res_path):
        plot_benchmark_results(res_path, out_img)
    else:
        print("Result file not found, please run run_maze_benchmark.py first.")
