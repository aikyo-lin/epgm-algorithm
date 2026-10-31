# EPGM Algorithm Lab (`epgm-algorithm`)

[ZH] EPGM 算法核心库、Gymnasium 2D 沙盒基准与对比实验。
[EN] EPGM algorithmic core, Gymnasium 2D sandbox benchmark, and comparative experiments.

## 结构说明 / Structure
- `epgm/`: Python 算法核心实现 (中英双语注释与数学原理对应)
  - `core.py`: 拓扑图与因果预测边定义
  - `quantizer.py`: 8向雷达事件量化器
  - `planner.py`: 主动好奇心推理与规划
  - `plasticity.py`: 拓扑局部可塑性、生枝与剪枝
- `envs/`: Gymnasium 兼容的 2D 连续空间仿真物理沙盒
- `baselines/`: DQN / 随机探索基线
- `experiments/`: 基准评测脚本与指标趋势图绘制

## 运行评测 / Run Benchmark
```bash
pip install -r requirements.txt
python experiments/run_maze_benchmark.py
python experiments/plot_metrics.py
```
评测生成的曲线图将保存在 `experiments/results/learning_curves.png`，导出的图模型碎片位于 `experiments/results/epgm_graph_model.json`。
