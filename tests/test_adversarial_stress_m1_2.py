"""
[ZH] Milestone M1 对抗性压力测试套件 (Challenger 2).
[EN] Milestone M1 Adversarial Stress Test Suite (Challenger 2).

测试范围 / Scope:
    1. 长程仿真确定性压力测试 (Long-Horizon Simulation Determinism, 1000 步 x 3 次重跑位姿与观测严格按位一致)
    2. 规划器动态重规划突变压力测试 (Rapid Causal Graph Mutation, Edge Pruning, Probability Inversions, Concurrency)
    3. 安全硬掩码数值完备性测试 (Safety Mask Numeric Integrity: 无 IndexError, 无 ZeroDivisionError, 无 NaN)
    4. 量化器全域不变性压力测试 (Quantizer Invariance: 10,000 次随机观测严格位于 [0, 81] 或 [101, 181])
"""

import threading
import numpy as np
import pytest

from envs.arena2d import Arena2DEnv
from epgm.core import EventPredictiveGraph, EPGMEdge
from epgm.planner import GraphPlanner
from epgm.quantizer import EventQuantizer


class TestLongHorizonDeterminism:
    """
    [ZH] 1,000 步长程仿真确定性压力测试。
    [EN] 1,000-step long-horizon simulation determinism stress test.
    """

    def test_1000_step_bitwise_determinism_cyclic_actions(self):
        """
        [ZH] 固定随机种子下，执行 1,000 步循环动作序列，独立重复 3 次，验证轨迹与观测 100% 按位一致。
        [EN] Run 1,000 steps with cyclic deterministic actions across 3 independent runs; verify 100% bitwise match.
        """
        seed = 42
        total_steps = 1000
        # 循环覆盖所有 5 个离散动作: 0=停止, 1=前进, 2=后退, 3=左转, 4=右转
        actions = [i % 5 for i in range(total_steps)]

        trajectories = []
        for _ in range(3):
            env = Arena2DEnv(max_steps=total_steps + 100)
            obs, info = env.reset(seed=seed)
            run_records = {
                "observations": [obs.copy()],
                "rewards": [],
                "positions": [info["position"].copy()],
                "orientations": [info["orientation"]],
                "bumper_hits": [info["bumper_hit"]],
                "terminated": [],
                "truncated": [],
            }

            for act in actions:
                next_obs, rew, term, trunc, next_info = env.step(act)
                run_records["observations"].append(next_obs.copy())
                run_records["rewards"].append(rew)
                run_records["positions"].append(next_info["position"].copy())
                run_records["orientations"].append(next_info["orientation"])
                run_records["bumper_hits"].append(next_info["bumper_hit"])
                run_records["terminated"].append(term)
                run_records["truncated"].append(trunc)

            trajectories.append(run_records)

        # 逐步断言 Run 0, Run 1, Run 2 之间所有观测、回报、位姿严格按位一致
        for step in range(total_steps):
            obs_0 = trajectories[0]["observations"][step]
            obs_1 = trajectories[1]["observations"][step]
            obs_2 = trajectories[2]["observations"][step]
            assert np.array_equal(obs_0, obs_1), f"Observation mismatch between run 0 and 1 at step {step}"
            assert np.array_equal(obs_1, obs_2), f"Observation mismatch between run 1 and 2 at step {step}"

            pos_0 = trajectories[0]["positions"][step]
            pos_1 = trajectories[1]["positions"][step]
            pos_2 = trajectories[2]["positions"][step]
            assert np.array_equal(pos_0, pos_1), f"Position mismatch between run 0 and 1 at step {step}"
            assert np.array_equal(pos_1, pos_2), f"Position mismatch between run 1 and 2 at step {step}"

            ori_0 = trajectories[0]["orientations"][step]
            ori_1 = trajectories[1]["orientations"][step]
            ori_2 = trajectories[2]["orientations"][step]
            assert ori_0 == ori_1, f"Orientation mismatch between run 0 and 1 at step {step}"
            assert ori_1 == ori_2, f"Orientation mismatch between run 1 and 2 at step {step}"

            rew_0 = trajectories[0]["rewards"][step]
            rew_1 = trajectories[1]["rewards"][step]
            rew_2 = trajectories[2]["rewards"][step]
            assert rew_0 == rew_1, f"Reward mismatch between run 0 and 1 at step {step}"
            assert rew_1 == rew_2, f"Reward mismatch between run 1 and 2 at step {step}"

        # 累计回报严格相等
        cum_rewards = [sum(t["rewards"]) for t in trajectories]
        assert cum_rewards[0] == cum_rewards[1] == cum_rewards[2], (
            f"Cumulative rewards not bitwise identical: {cum_rewards}"
        )

    def test_1000_step_bitwise_determinism_pseudorandom_actions(self):
        """
        [ZH] 使用固定 PRNG 生成的 1,000 步伪随机离散动作流，独立重复 3 次测试无方差。
        [EN] Run 1,000 steps with pseudorandom deterministic action stream; verify zero variance across 3 runs.
        """
        seed = 20261031
        total_steps = 1000
        action_rng = np.random.default_rng(98765)
        actions = action_rng.integers(0, 5, size=total_steps).tolist()

        trajectories = []
        for _ in range(3):
            env = Arena2DEnv(max_steps=total_steps + 100)
            obs, info = env.reset(seed=seed)
            run_records = {
                "observations": [obs.copy()],
                "rewards": [],
                "positions": [info["position"].copy()],
                "orientations": [info["orientation"]],
            }

            for act in actions:
                next_obs, rew, term, trunc, next_info = env.step(act)
                run_records["observations"].append(next_obs.copy())
                run_records["rewards"].append(rew)
                run_records["positions"].append(next_info["position"].copy())
                run_records["orientations"].append(next_info["orientation"])

            trajectories.append(run_records)

        # 验证 3 次运行严格按位一致
        for step in range(total_steps):
            assert np.array_equal(trajectories[0]["observations"][step], trajectories[1]["observations"][step])
            assert np.array_equal(trajectories[1]["observations"][step], trajectories[2]["observations"][step])
            assert np.array_equal(trajectories[0]["positions"][step], trajectories[1]["positions"][step])
            assert np.array_equal(trajectories[1]["positions"][step], trajectories[2]["positions"][step])
            assert trajectories[0]["orientations"][step] == trajectories[1]["orientations"][step]
            assert trajectories[1]["orientations"][step] == trajectories[2]["orientations"][step]
            assert trajectories[0]["rewards"][step] == trajectories[1]["rewards"][step]
            assert trajectories[1]["rewards"][step] == trajectories[2]["rewards"][step]

        final_pos_x = [t["positions"][-1][0] for t in trajectories]
        final_pos_y = [t["positions"][-1][1] for t in trajectories]
        assert final_pos_x[0] == final_pos_x[1] == final_pos_x[2]
        assert final_pos_y[0] == final_pos_y[1] == final_pos_y[2]


class TestPlannerDynamicRePlanningStress:
    """
    [ZH] 规划器动态重规划突变压力测试与数值完备性保障。
    [EN] Planner dynamic re-planning stress testing under graph mutations and numeric integrity.
    """

    def test_rapid_mutation_edge_pruning_and_inversion(self):
        """
        [ZH] 在规划决策循环中高频注入图变动（并发增边、删边、概率翻转、碰撞注入），检验健壮性。
        [EN] Rapidly mutate causal graph during decision loop (edge addition, deletion, probability inversions).
        """
        graph = EventPredictiveGraph(action_dim=5)
        planner = GraphPlanner(
            graph,
            action_dim=5,
            gamma=0.9,
            curiosity_weight=0.4,
            planning_depth=2,
            collision_conf_thresh=0.5,
            collision_min_obs=1,
            seed=42,
        )

        # 初始注入若干基础节点与边
        for s in range(1, 20):
            graph.add_node(s)
            for a in range(5):
                target = (s * 3 + a) % 20 + 1
                edge = EPGMEdge(
                    source_event=s,
                    action=a,
                    target_event=target,
                    expected_reward=0.5,
                    observation_count=2,
                    confidence=0.7,
                )
                graph.adjacency[(s, a)] = [edge]

        rng = np.random.default_rng(12345)

        for step in range(500):
            curr_event = int(rng.integers(1, 20))

            # 随机触发突变操作
            mutation_type = rng.integers(0, 4)
            target_act = int(rng.integers(0, 5))

            if mutation_type == 0:
                # 突变 0: 碰撞概率翻转 (注入指向 0 的绝对碰撞边)
                graph.adjacency[(curr_event, target_act)] = [
                    EPGMEdge(
                        source_event=curr_event,
                        action=target_act,
                        target_event=0,
                        expected_reward=-10.0,
                        observation_count=5,
                        confidence=0.95,
                    )
                ]
            elif mutation_type == 1:
                # 突变 1: 边修剪 (清空邻接表，模拟 LTD 修剪)
                graph.adjacency[(curr_event, target_act)] = []
            elif mutation_type == 2:
                # 突变 2: 概率反转与巨额虚假奖励诱骗 (测试安全硬掩码在诱惑下的抵抗力)
                graph.adjacency[(curr_event, target_act)] = [
                    EPGMEdge(
                        source_event=curr_event,
                        action=target_act,
                        target_event=0,
                        expected_reward=100.0,  # 巨额虚假期望回报诱骗
                        observation_count=10,
                        confidence=0.88,
                    )
                ]
            elif mutation_type == 3:
                # 突变 3: 恢复为安全探索边
                safe_target = int(rng.integers(1, 20))
                graph.adjacency[(curr_event, target_act)] = [
                    EPGMEdge(
                        source_event=curr_event,
                        action=target_act,
                        target_event=safe_target,
                        expected_reward=1.0,
                        observation_count=1,
                        confidence=0.6,
                    )
                ]

            # 执行规划器决策
            action, info = planner.select_action(curr_event, epsilon_greedy=0.1)

            # 断言数值有效性与安全拦截
            assert isinstance(action, int), f"Action must be int, got {type(action)}"
            assert 0 <= action < 5, f"Action out of range [0, 4]: {action}"
            assert not np.isnan(action), "Action is NaN"

            # 验证安全硬掩码在高置信度碰撞下绝对不选危险动作
            for a in range(5):
                preds = graph.get_predictions(curr_event, a)
                for edge in preds:
                    if edge.target_event == 0 and edge.confidence >= 0.5 and edge.observation_count >= 1:
                        assert a in info["masked_actions"], f"Action {a} must be masked!"
                        if not info["fallback_triggered"]:
                            assert action != a, f"Safety mask breached! Action {a} chosen despite collision hazard!"

            # 验证所有返回分数均非 NaN
            for a, score in info["scores"].items():
                assert not np.isnan(score), f"Score for action {a} is NaN!"

    def test_safety_mask_numeric_integrity_extreme_boundaries(self):
        """
        [ZH] 极限边界条件下验证安全硬掩码与规划器永不抛出 IndexError, ZeroDivisionError 或产生 NaN。
        [EN] Verify safety mask never raises IndexError, ZeroDivisionError, or NaN under extreme boundary conditions.
        """
        graph = EventPredictiveGraph(action_dim=5)
        planner = GraphPlanner(
            graph,
            action_dim=5,
            gamma=0.9,
            curiosity_weight=0.4,
            planning_depth=3,
            collision_conf_thresh=0.5,
            collision_min_obs=1,
            seed=999,
        )

        # 场景 A: 完全空图 (没有任何节点和边)
        for event in [-1, 0, 1, 99999]:
            act, info = planner.select_action(event)
            assert 0 <= act < 5
            for s in info["scores"].values():
                assert not np.isnan(s)

        # 场景 B: 观测计数为 0、置信度为 0、回报极端值 (-1e9, 1e9)
        graph.add_node(100)
        graph.adjacency[(100, 0)] = [
            EPGMEdge(source_event=100, action=0, target_event=0, expected_reward=-1e9, observation_count=0, confidence=0.0)
        ]
        graph.adjacency[(100, 1)] = [
            EPGMEdge(source_event=100, action=1, target_event=101, expected_reward=1e9, observation_count=0, confidence=0.0)
        ]
        graph.adjacency[(100, 2)] = [
            EPGMEdge(source_event=100, action=2, target_event=102, expected_reward=0.0, observation_count=0, confidence=1.0)
        ]

        act, info = planner.select_action(100, epsilon_greedy=0.0)
        assert 0 <= act < 5
        for s in info["scores"].values():
            assert not np.isnan(s)

        # 场景 C: 循环自环图 (Depth 展开不爆栈或产生除零)
        graph.adjacency[(200, 0)] = [
            EPGMEdge(source_event=200, action=0, target_event=200, expected_reward=0.0, observation_count=1, confidence=0.5)
        ]
        act, info = planner.select_action(200)
        assert 0 <= act < 5
        assert not np.isnan(info["scores"][0])

        # 场景 D: 全动作均预测碰撞 (全掩码紧急避险 fallback)
        for a in range(5):
            graph.adjacency[(300, a)] = [
                EPGMEdge(source_event=300, action=a, target_event=0, expected_reward=-5.0, observation_count=5, confidence=0.99)
            ]
        act, info = planner.select_action(300, epsilon_greedy=0.5)
        assert info["fallback_triggered"] is True
        assert act in [2, 0]  # 后退或停止保底动作
        assert not np.isnan(act)

    def test_concurrent_multithreaded_planning_mutation_stress(self):
        """
        [ZH] 多线程并发突变压力测试：读写并发交织下检验规划器健壮性。
        [EN] Multithreaded concurrent planning & mutation stress test.
        """
        graph = EventPredictiveGraph(action_dim=5)
        for i in range(1, 30):
            graph.add_node(i)
            for a in range(5):
                graph.adjacency[(i, a)] = [
                    EPGMEdge(source_event=i, action=a, target_event=(i + 1) % 30 + 1, expected_reward=0.1, observation_count=1)
                ]

        planner = GraphPlanner(graph, action_dim=5, planning_depth=2, seed=123)

        stop_event = threading.Event()
        errors = []

        def reader_task():
            rng = np.random.default_rng(2026)
            for _ in range(500):
                if stop_event.is_set():
                    break
                try:
                    ev = int(rng.integers(1, 30))
                    act, info = planner.select_action(ev, epsilon_greedy=0.1)
                    assert 0 <= act < 5
                    for s in info["scores"].values():
                        assert not np.isnan(s)
                except Exception as e:
                    errors.append(("reader", e))
                    break

        def mutator_task():
            rng = np.random.default_rng(3030)
            for _ in range(500):
                if stop_event.is_set():
                    break
                try:
                    s = int(rng.integers(1, 30))
                    a = int(rng.integers(0, 5))
                    t = int(rng.integers(0, 30))
                    # 快速更新/替换边
                    graph.adjacency[(s, a)] = [
                        EPGMEdge(
                            source_event=s,
                            action=a,
                            target_event=t,
                            expected_reward=float(rng.uniform(-5.0, 5.0)),
                            observation_count=int(rng.integers(1, 20)),
                            confidence=float(rng.uniform(0.1, 0.99)),
                        )
                    ]
                except Exception as e:
                    errors.append(("mutator", e))
                    break

        t1 = threading.Thread(target=reader_task)
        t2 = threading.Thread(target=reader_task)
        t3 = threading.Thread(target=mutator_task)

        t1.start()
        t2.start()
        t3.start()

        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        t3.join(timeout=10.0)
        stop_event.set()

        assert len(errors) == 0, f"Concurrent stress encountered errors: {errors}"


class TestQuantizerInvariance10000Observations:
    """
    [ZH] 量化器全域不变性压力测试 (10,000 次随机连续输入)。
    [EN] Quantizer invariance stress test over 10,000 random continuous observations.
    """

    def test_10000_random_observations_strict_range_invariance(self):
        """
        [ZH] 生成 10,000 组对抗性连续观测向量，断言量化器输出严格收敛于 [0, 81] 或 [101, 181]，绝无越界。
        [EN] Generate 10,000 random adversarial observations; verify event IDs strictly stay in [0, 81] or [101, 181].
        """
        quantizer = EventQuantizer(num_sensors=8, distance_bins=3, delta_threshold=0.25, max_range=1.0)
        rng = np.random.default_rng(20261031)

        total_trials = 10000
        observed_ids = set()

        # 初始化起始点
        curr_obs = rng.uniform(0.0, 1.0, size=8).astype(np.float32)

        for i in range(total_trials):
            # 对抗样本生成模式:
            # 模式 0 (40%): 平滑连续随机游走，步长 0.01 (产生稳定静态事件 1~81)
            # 模式 1 (30%): 大步长剧烈跳变 (||Delta s||_2 > 0.25)，激发动态事件 101~181
            # 模式 2 (15%): 临界分箱边界值 (0.0, 1/3, 2/3, 1.0 及微小漂移)
            # 模式 3 (10%): 超限异常输入 (负数、超大幅值，检验 np.clip 鲁棒性)
            # 模式 4 (5%): 随机物理碰撞触发 (bumper_hit=True)
            mode = i % 20

            if mode < 8:
                # 模式 0: 随机游走 (静态)
                curr_obs = np.clip(curr_obs + rng.normal(0.0, 0.01, size=8), 0.0, 1.0).astype(np.float32)
                bumper = False
            elif mode < 14:
                # 模式 1: 剧烈突变跳跃 (动态)
                curr_obs = rng.uniform(0.0, 1.0, size=8).astype(np.float32)
                bumper = False
            elif mode < 17:
                # 模式 2: 分箱边界值
                candidates = [0.0, 1.0 / 3.0 - 1e-6, 1.0 / 3.0 + 1e-6, 2.0 / 3.0 - 1e-6, 2.0 / 3.0 + 1e-6, 1.0]
                curr_obs = rng.choice(candidates, size=8).astype(np.float32)
                bumper = False
            elif mode < 19:
                # 模式 3: 超限异常观测 [-100.0, 200.0]
                curr_obs = rng.uniform(-100.0, 200.0, size=8).astype(np.float32)
                bumper = False
            else:
                # 模式 4: 硬件微动碰撞
                curr_obs = rng.uniform(0.0, 1.0, size=8).astype(np.float32)
                bumper = True

            event_id, proto = quantizer.quantize(curr_obs, bumper_hit=bumper)
            observed_ids.add(event_id)

            # 核心断言 1: 类型与非 NaN 判定
            assert isinstance(event_id, (int, np.integer)), f"Event ID must be integer, got {type(event_id)}"
            assert not np.isnan(event_id), "Event ID cannot be NaN"

            # 核心断言 2: 碰撞最高优先级约束 (必须返回 0)
            if bumper:
                assert event_id == 0, f"When bumper_hit=True, event_id must be 0, got {event_id}"

            # 核心断言 3: 区间完备不变性约束 [0, 81] 或 [101, 181]
            in_static_range = (0 <= event_id <= 81)
            in_dynamic_range = (101 <= event_id <= 181)
            assert in_static_range or in_dynamic_range, (
                f"Invariant breached! Event ID {event_id} is outside valid ranges [0, 81] and [101, 181] "
                f"at trial {i} (mode={mode}, bumper={bumper})"
            )

            # 核心断言 4: 绝无非法中间间隙 [82, 100] 与溢出 > 181
            assert not (82 <= event_id <= 100), f"Illegal event ID in forbidden gap [82, 100]: {event_id}"
            assert event_id <= 181, f"Event ID exceeded max bound 181: {event_id}"

            # 核心断言 5: 返回原型向量尺寸正确且全在 [0, 1] 归一化内
            assert proto.shape == (8,)
            assert (proto >= 0.0).all() and (proto <= 1.0).all()

        # 统计分布完备性断言: 10,000 次对抗采样充分覆盖空间
        static_count = sum(1 for eid in observed_ids if 1 <= eid <= 81)
        dynamic_count = sum(1 for eid in observed_ids if 101 <= eid <= 181)
        assert 0 in observed_ids, "Bumper collision event ID 0 was never observed"
        assert static_count >= 50, f"Static event coverage too low: {static_count}/81"
        assert dynamic_count >= 50, f"Dynamic event coverage too low: {dynamic_count}/81"

    def test_full_receptive_field_81_static_and_81_dynamic_coverage(self):
        """
        [ZH] 穷举测试 4 象限全部 3^4=81 种空间几何构型，验证所有 81 个静态 ID 与 81 个动态 ID 均可被正确生成。
        [EN] Exhaustively verify all 3^4=81 spatial sector configurations generate all 81 static & dynamic IDs.
        """
        quantizer = EventQuantizer(num_sensors=8, distance_bins=3, delta_threshold=0.25, max_range=1.0)
        static_ids = set()
        dynamic_ids = set()

        vals = [0.1, 0.5, 0.9]  # 映射到分箱 0, 1, 2
        for f in range(3):
            for r in range(3):
                for b in range(3):
                    for l in range(3):
                        # 构造四象限射线读数
                        # Front: min(ray0, ray1, ray7) = f
                        # Right: min(ray1, ray2, ray3) = r
                        # Back:  min(ray3, ray4, ray5) = b
                        # Left:  min(ray5, ray6, ray7) = l
                        obs = np.array(
                            [
                                vals[f],
                                vals[max(f, r)],
                                vals[r],
                                vals[max(r, b)],
                                vals[b],
                                vals[max(b, l)],
                                vals[l],
                                vals[max(l, f)],
                            ],
                            dtype=np.float32,
                        )

                        # 静态测试 (reset 消除历史差分)
                        quantizer.reset()
                        s_id, _ = quantizer.quantize(obs)
                        static_ids.add(s_id)

                        # 动态测试 (先输入反向极大差异向量触发突变)
                        quantizer.reset()
                        baseline = np.zeros(8, dtype=np.float32) if np.linalg.norm(obs) > 0.3 else np.ones(8, dtype=np.float32)
                        quantizer.quantize(baseline)
                        d_id, _ = quantizer.quantize(obs)
                        dynamic_ids.add(d_id)

        assert len(static_ids) == 81, f"Expected 81 static IDs, got {len(static_ids)}"
        assert min(static_ids) == 1 and max(static_ids) == 81
        assert len(dynamic_ids) == 81, f"Expected 81 dynamic IDs, got {len(dynamic_ids)}"
        assert min(dynamic_ids) == 101 and max(dynamic_ids) == 181
