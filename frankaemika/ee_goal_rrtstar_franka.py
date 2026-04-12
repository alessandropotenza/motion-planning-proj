#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# EE-goal planning for Franka Panda (7-DOF):
#   - Vanilla RRT* (C-space start + task-space goal via numerical IK)
#   - CDF-guided RRT* (task-goal termination)
#   - Pull-and-slide (task-goal termination + obstacle tangential bias)
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from franka_kinematics import fk_flange_position
from rrt_star_franka import Node, RRTStarFrankaBase, VanillaFrankaRRTStar, path_waypoint_cost
from sphere_arm_collision import SphereArmCollisionChecker
from workspace_obstacles import BoxObstacle, Obstacle, SphereObstacle, build_demo_obstacles

if torch is not None:
    from cdf_guided_rrtstar_franka import CDFGuidedFrankaRRTStar
else:  # pragma: no cover
    CDFGuidedFrankaRRTStar = None  # type: ignore[assignment]


DEFAULT_START_Q = np.array([0.0, -0.25, 0.0, -2.15, 0.0, 1.85, 0.785], dtype=np.float32)
DEFAULT_GOAL_TASK = np.array([0.55, 0.10, 0.48], dtype=np.float32)
SCENE_CHOICES = (
    "demo_table",
    "sparse",
    "pillar_and_box",
    "cluttered_gate",
    "cluttered_shelf",
    "cluttered_crossing",
)

# Hand-picked starts and task goals near obstacles for stress testing.
SCENE_DEFAULTS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "demo_table": (
        np.array([0.0, -0.25, 0.0, -2.15, 0.0, 1.85, 0.785], dtype=np.float32),
        np.array([0.55, 0.10, 0.48], dtype=np.float32),
    ),
    "sparse": (
        np.array([0.0, -0.05, 0.0, -1.90, 0.0, 1.75, 0.2], dtype=np.float32),
        np.array([0.52, 0.18, 0.42], dtype=np.float32),
    ),
    "pillar_and_box": (
        np.array([0.05, -0.35, 0.25, -2.10, 0.1, 1.90, 0.65], dtype=np.float32),
        np.array([0.42, 0.28, 0.52], dtype=np.float32),
    ),
    "cluttered_gate": (
        np.array([0.136, -0.807, 0.166, -2.002, -0.185, 2.064, 0.755], dtype=np.float32),
        np.array([0.449, 0.380, 0.574], dtype=np.float32),
    ),
    "cluttered_shelf": (
        np.array([-0.258, -0.549, -0.287, -2.308, 0.261, 2.047, 0.665], dtype=np.float32),
        np.array([1.0, 0, 0.5], dtype=np.float32),
    ),
    "cluttered_crossing": (
        np.array([0.205, -0.395, 0.173, -1.906, 0.041, 1.939, 1.097], dtype=np.float32),
        np.array([0.554, 0.307, 0.576], dtype=np.float32),
    ),
}


def _arr_str(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=3, floatmode="fixed")


def default_start_and_goal_task(scene: str) -> Tuple[np.ndarray, np.ndarray]:
    if scene in SCENE_DEFAULTS:
        s, g = SCENE_DEFAULTS[scene]
        return s.copy(), g.copy()
    return DEFAULT_START_Q.copy(), DEFAULT_GOAL_TASK.copy()


def _finite_diff_grad(fun, q: np.ndarray, eps: float = 2e-3) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(7)
    f0 = float(fun(q))
    g = np.zeros(7, dtype=np.float64)
    for i in range(7):
        q2 = q.copy()
        q2[i] += eps
        g[i] = (float(fun(q2)) - f0) / eps
    n = float(np.linalg.norm(g))
    if n < 1e-10:
        return np.zeros(7, dtype=np.float32)
    return (g / n).astype(np.float32)


def _ee_pos(q: np.ndarray) -> np.ndarray:
    return fk_flange_position(np.asarray(q, dtype=np.float64).reshape(7)).astype(np.float32)


def _ee_err_norm(q: np.ndarray, goal_task: np.ndarray) -> float:
    ee = _ee_pos(q)
    return float(np.linalg.norm(ee - goal_task))


def _spawn_obstacle_visuals(p, obstacles: List[Obstacle]) -> List[int]:
    """Create PyBullet bodies mirroring analytic obstacles (visualization only)."""
    bodies: List[int] = []
    for o in obstacles:
        if isinstance(o, SphereObstacle):
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(o.radius),
                rgbaColor=[0.85, 0.35, 0.1, 1.0],
                specularColor=[0, 0, 0, 1],
            )
            bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=o.center.tolist())
            bodies.append(bid)
        elif isinstance(o, BoxObstacle):
            he = o.half_extents.astype(float).tolist()
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=he,
                rgbaColor=[0.2, 0.45, 0.9, 1.0],
                specularColor=[0, 0, 0, 1],
            )
            bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=o.center.tolist())
            bodies.append(bid)
    return bodies


def solve_task_goal_ik(
    checker: SphereArmCollisionChecker,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    seed: int,
    n_restarts: int = 40,
    max_steps: int = 140,
    step_size: float = 0.18,
    reg_to_start: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight numerical IK for flange position:
      minimize ||fk(q) - goal_task|| + reg_to_start*||q-start_q||.
    Returns (best_q, all_candidate_qs).
    """
    rng = np.random.default_rng(seed)
    q_min = checker.q_min.astype(np.float32)
    q_max = checker.q_max.astype(np.float32)
    start_q = np.asarray(start_q, dtype=np.float32).reshape(7)
    goal_task = np.asarray(goal_task, dtype=np.float32).reshape(3)

    def objective(q: np.ndarray) -> float:
        e = _ee_err_norm(q, goal_task)
        r = reg_to_start * float(np.linalg.norm(q - start_q))
        return e + r

    seeds: List[np.ndarray] = [start_q.copy()]
    seeds.extend(rng.uniform(q_min, q_max, size=(max(0, n_restarts - 1), 7)).astype(np.float32))

    best_q = start_q.copy()
    best_obj = float("inf")
    candidates: List[np.ndarray] = []

    for q0 in seeds:
        q = q0.astype(np.float32).copy()
        for _ in range(max_steps):
            grad = _finite_diff_grad(objective, q)
            if np.linalg.norm(grad) < 1e-8:
                break
            q = q - step_size * grad
            q = np.clip(q, q_min, q_max).astype(np.float32)
        if checker.is_state_free(q):
            candidates.append(q.copy())
            obj = objective(q)
            if obj < best_obj:
                best_obj = obj
                best_q = q.copy()

    if not candidates:
        raise RuntimeError("IK failed: no collision-free IK candidate found for task goal.")

    return best_q, np.stack(candidates, axis=0)


def validate_start_and_goal(
    checker: SphereArmCollisionChecker,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure start is collision-free and task goal has at least one collision-free IK.
    Returns a feasible (goal_q, ik_solutions) pair for reuse by vanilla planner.
    """
    if not checker.is_state_free(start_q):
        raise SystemExit("Start configuration is invalid (out-of-bounds or in collision).")
    try:
        goal_q, ik_solutions = solve_task_goal_ik(
            checker,
            start_q,
            goal_task,
            seed=seed,
            n_restarts=26,
            max_steps=100,
        )
    except Exception as exc:
        raise SystemExit(
            "Task goal is not collision-feasible: no collision-free IK solution found. "
            "Adjust scene/defaults or pass --goal-task."
        ) from exc
    return goal_q, ik_solutions


if CDFGuidedFrankaRRTStar is not None:
    class TaskGoalCDFFrankaRRTStar(CDFGuidedFrankaRRTStar):
        """CDF-guided RRT* with task-space goal termination (flange position)."""

        def __init__(self, *args, ee_goal_threshold: float = 0.05, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.ee_goal_threshold = float(ee_goal_threshold)
            self.goal_task = np.zeros(3, dtype=np.float32)

        def set_goal_task(self, goal_task: np.ndarray) -> None:
            self.goal_task = np.asarray(goal_task, dtype=np.float32).reshape(3)

        def _validate_goal(self, goal: np.ndarray) -> None:
            # In task-goal mode, `goal` is a dummy placeholder for base API compatibility.
            _ = goal

        def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
            _ = goal
            return _ee_err_norm(q_new, self.goal_task) <= self.ee_goal_threshold

        def _make_goal_connection(
            self,
            new_idx: int,
            nodes: List[Node],
            goal: np.ndarray,
            goal_idx: Optional[int],
            rewires: int,
            it: int,
            t0: float,
        ) -> Tuple[Optional[int], int, Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]:
            _ = goal, it, t0
            first_goal_iteration = None
            nodes_to_first_path = None
            time_to_first_path_sec = None
            goal_event: Dict[str, Any] = {"type": "task_goal", "node_idx": new_idx}
            goal_cost = nodes[new_idx].cost
            if goal_idx is None:
                goal_idx = new_idx
                first_goal_iteration = it + 1
                nodes_to_first_path = len(nodes)
                time_to_first_path_sec = time.time() - t0
            elif goal_cost < nodes[goal_idx].cost:
                goal_idx = new_idx
                rewires += 1
            return goal_idx, rewires, first_goal_iteration, nodes_to_first_path, time_to_first_path_sec, goal_event


    class PullAndSlideFrankaRRTStar(TaskGoalCDFFrankaRRTStar):
        """
        Task-goal planner with obstacle-aware "pull and slide" sampling:
          - pull: descend task-space EE distance to goal
          - slide: when near obstacles, move tangentially to obstacle gradient while
                   still progressing toward task goal.
        """

        def __init__(self, *args, pull_steps: int = 2, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pull_steps = int(pull_steps)

        def _ee_goal_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
            q = np.asarray(q, dtype=np.float32).reshape(7)

            def f(qq: np.ndarray) -> float:
                return _ee_err_norm(qq, self.goal_task)

            d = float(f(q))
            g = _finite_diff_grad(f, q)
            return d, g

        def sample_target_with_info(
            self, goal: np.ndarray, rng: np.random.Generator
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            _ = goal  # task-goal planner uses self.goal_task instead.
            q = rng.uniform(self.q_min, self.q_max).astype(np.float32)
            d_obs, g_obs = self.get_cdf_data(q)
            d_ee, g_ee = self._ee_goal_data(q)

            if rng.random() < self.goal_bias or d_ee < max(self.goal_threshold, self.ee_goal_threshold):
                q_pull = q.copy()
                for _ in range(self.pull_steps):
                    d_tmp, g_tmp = self._ee_goal_data(q_pull)
                    q_pull = q_pull - np.clip(d_tmp, 0.0, self.step_size * 2.5) * g_tmp
                    q_pull = self.clamp_to_bounds(q_pull)
                return q_pull.astype(np.float32), {
                    "mode": "pull_goal",
                    "raw_sample": q.copy(),
                    "used_sample": q_pull.copy(),
                    "projected": True,
                    "ee_distance": float(d_ee),
                    "cdf_distance": float(d_obs),
                }

            if d_obs > self.safety_margin_c_space:
                q_goal = q - np.clip(d_ee, 0.0, self.step_size * 1.5) * g_ee
                q_goal = self.clamp_to_bounds(q_goal)
                return q_goal.astype(np.float32), {
                    "mode": "pull_free",
                    "raw_sample": q.copy(),
                    "used_sample": q_goal.copy(),
                    "projected": False,
                    "ee_distance": float(d_ee),
                    "cdf_distance": float(d_obs),
                }

            # Near obstacle: slide along obstacle boundary while preserving goal progress.
            tangent = g_ee - float(np.dot(g_ee, g_obs)) * g_obs
            n = float(np.linalg.norm(tangent))
            if n < 1e-8:
                tangent = g_ee.copy()
                n = float(np.linalg.norm(tangent))
            tangent = tangent / max(n, 1e-8)

            q_slide = q + self.step_size * tangent
            if d_obs < self.safety_margin_c_space:
                q_slide = q_slide + (self.safety_margin_c_space - d_obs) * g_obs
            q_slide = self.clamp_to_bounds(q_slide)
            return q_slide.astype(np.float32), {
                "mode": "slide_obstacle",
                "raw_sample": q.copy(),
                "used_sample": q_slide.copy(),
                "projected": True,
                "ee_distance": float(d_ee),
                "cdf_distance": float(d_obs),
            }
else:  # pragma: no cover
    class TaskGoalCDFFrankaRRTStar(RRTStarFrankaBase):
        pass

    class PullAndSlideFrankaRRTStar(TaskGoalCDFFrankaRRTStar):
        pass


@dataclass
class PlannerResult:
    planner_name: str
    path: Optional[np.ndarray]
    stats: Dict[str, Any]
    task_goal_error: float


def print_stats(title: str, stats: Dict[str, Any], path: Optional[np.ndarray], task_err: float) -> None:
    print(f"\n=== {title} ===")
    print(f"success:                 {stats['success']}")
    print(f"iterations:              {stats['iterations']}")
    print(f"accepted_nodes:          {stats['accepted_nodes']}")
    print(f"discarded_samples:       {stats['discarded_samples']}")
    print(f"rejection_rate:          {stats['rejection_rate']:.4f}")
    print(f"rewires:                 {stats['rewires']}")
    print(f"planning_time_sec:       {stats['planning_time_sec']:.4f}")
    print(f"time_to_first_path_sec:  {stats.get('time_to_first_path_sec', None)}")
    print(f"nodes_to_first_path:     {stats.get('nodes_to_first_path', None)}")
    print(f"final_path_cost:         {stats.get('final_path_cost', None)}")
    print(f"path_waypoint_cost:      {path_waypoint_cost(path)}")
    print(f"task_goal_error_m:       {task_err:.4f}")
    print(f"path_waypoints:          {0 if path is None else len(path)}")


def _spawn_task_trace(p, path_q: np.ndarray, color=(0.95, 0.1, 0.1), width: float = 2.5) -> None:
    if path_q is None or len(path_q) < 2:
        return
    ee = np.stack([_ee_pos(q) for q in path_q], axis=0)
    for i in range(len(ee) - 1):
        p.addUserDebugLine(
            lineFromXYZ=ee[i].tolist(),
            lineToXYZ=ee[i + 1].tolist(),
            lineColorRGB=list(color),
            lineWidth=float(width),
            lifeTime=0.0,
        )


def visualize_task_space_pybullet(
    path: np.ndarray,
    obstacles: List[Obstacle],
    goal_task: np.ndarray,
    hz: float = 20.0,
    hold_last_sec: float = 0.0,
    label: str = "",
) -> None:
    import pybullet as p
    import pybullet_data as pd
    from pybullet_panda_sim import PandaSim

    dt = 1.0 / hz
    p.connect(p.GUI, options="--width=1200 --height=900")
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    _spawn_obstacle_visuals(p, obstacles)

    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.035,
        rgbaColor=[0.08, 0.82, 0.18, 1.0],
        specularColor=[0.2, 0.2, 0.2, 1],
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=np.asarray(goal_task, dtype=float).tolist())

    robot = PandaSim(p, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
    if label:
        p.addUserDebugText(label, [0.15, 0.0, 1.05], textColorRGB=(1, 0.2, 0.2), textSize=1.2)

    q = np.asarray(path, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 7)

    dense: List[np.ndarray] = []
    if len(q) == 1:
        dense = [q[0].copy()]
    else:
        for i in range(len(q) - 1):
            n = max(2, int(np.linalg.norm(q[i + 1] - q[i]) / 0.04) + 1)
            for a in np.linspace(0.0, 1.0, n, endpoint=(i == len(q) - 2)):
                dense.append(((1.0 - a) * q[i] + a * q[i + 1]).astype(np.float32))
    dense_q = np.stack(dense, axis=0)

    _spawn_task_trace(p, dense_q)
    for qq in dense_q:
        robot.set_joint_positions(qq)
        p.stepSimulation()
        time.sleep(dt * 0.5)

    if hold_last_sec <= 0.0:
        # Keep the demo open until the user closes the GUI (or Ctrl+C).
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
    else:
        t_end = time.time() + hold_last_sec
        while time.time() < t_end and p.isConnected():
            p.stepSimulation()
            time.sleep(dt)
    p.disconnect()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Franka EE-goal RRT*: vanilla, cdf, pull-and-slide.")
    ap.add_argument("--planner", choices=["vanilla", "cdf", "pullandslide", "all"], default="all")
    ap.add_argument("--scene", choices=SCENE_CHOICES, default="demo_table")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--start-q", nargs=7, type=float, default=None, metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"))
    ap.add_argument("--goal-task", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--step-size", type=float, default=0.12)
    ap.add_argument("--goal-threshold", type=float, default=0.35)
    ap.add_argument("--goal-bias", type=float, default=0.08)
    ap.add_argument("--neighbor-radius", type=float, default=0.65)
    ap.add_argument("--edge-resolution", type=float, default=0.06)
    ap.add_argument("--ee-goal-threshold", type=float, default=0.06)
    ap.add_argument("--demo", action="store_true", help="Animate best successful plan in PyBullet.")
    ap.add_argument(
        "--demo-hold-sec",
        type=float,
        default=0.0,
        help="Seconds to keep PyBullet open after playback. <= 0 means keep open until window close/Ctrl+C.",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (default auto)")
    ap.add_argument("--log-every", type=int, default=0, help="Print planner progress every N iterations.")
    return ap.parse_args()


def _run_vanilla(
    checker: SphereArmCollisionChecker,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    goal_q: np.ndarray,
    ik_solutions: np.ndarray,
    args: argparse.Namespace,
) -> PlannerResult:
    print(f"\n[vanilla] IK solutions: {len(ik_solutions)}")
    print(f"[vanilla] selected goal_q = {_arr_str(goal_q)}")
    planner = VanillaFrankaRRTStar(
        checker=checker,
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=args.goal_bias,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
    )
    log_kw = {"log_every_iters": args.log_every, "log_prefix": "[vanilla] "} if args.log_every > 0 else {}
    _nodes, path, stats = planner.plan(start_q, goal_q, max_iters=args.max_iters, seed=args.seed, **log_kw)
    task_err = float("nan")
    if path is not None and len(path) > 0:
        task_err = _ee_err_norm(path[-1], goal_task)
    return PlannerResult("vanilla", path, stats, task_err)


def _run_task_goal_planner(
    planner_name: str,
    planner: TaskGoalCDFFrankaRRTStar,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    args: argparse.Namespace,
) -> PlannerResult:
    planner.set_goal_task(goal_task)
    dummy_goal = start_q.copy()
    log_kw = {"log_every_iters": args.log_every, "log_prefix": f"[{planner_name}] "} if args.log_every > 0 else {}
    _nodes, path, stats = planner.plan(start_q, dummy_goal, max_iters=args.max_iters, seed=args.seed, **log_kw)
    task_err = float("nan")
    if path is not None and len(path) > 0:
        task_err = _ee_err_norm(path[-1], goal_task)
    return PlannerResult(planner_name, path, stats, task_err)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)

    need_torch = args.planner in ("cdf", "pullandslide", "all")
    if need_torch and torch is None:
        raise SystemExit(
            "PyTorch is required for CDF/Pull-and-slide planners. "
            "Install torch or run --planner vanilla."
        )
    if torch is not None:
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    else:
        device = "cpu"

    obstacles = build_demo_obstacles(args.scene, np.random.default_rng(args.seed))
    checker = SphereArmCollisionChecker(obstacles, margin=0.02)

    d_start_q, d_goal_task = default_start_and_goal_task(args.scene)
    start_q = np.asarray(args.start_q, dtype=np.float32).reshape(7) if args.start_q is not None else d_start_q
    goal_task = np.asarray(args.goal_task, dtype=np.float32).reshape(3) if args.goal_task is not None else d_goal_task

    goal_q_feasible, ik_solutions = validate_start_and_goal(
        checker=checker,
        start_q=start_q,
        goal_task=goal_task,
        seed=args.seed,
    )

    print("=== Franka EE-goal Planning (7-DOF) ===")
    print(f"scene={args.scene}, seed={args.seed}, device={device}")
    print(f"start_q={_arr_str(start_q)}")
    print(f"goal_task(xyz)={_arr_str(goal_task)}")
    print(f"goal_q(feasible IK)={_arr_str(goal_q_feasible)}")
    print(f"goal_task_error_from_goal_q={_ee_err_norm(goal_q_feasible, goal_task):.4f} m")

    results: List[PlannerResult] = []

    if args.planner in ("vanilla", "all"):
        try:
            res_v = _run_vanilla(
                checker,
                start_q,
                goal_task,
                goal_q_feasible,
                ik_solutions,
                args,
            )
            print_stats("VanillaFrankaRRTStar (task-goal via IK)", res_v.stats, res_v.path, res_v.task_goal_error)
            results.append(res_v)
        except Exception as exc:
            print(f"\n[vanilla] failed: {exc}")

    if args.planner in ("cdf", "all"):
        cdf = TaskGoalCDFFrankaRRTStar(
            checker=checker,
            obstacles=obstacles,
            device=device,
            step_size=args.step_size,
            goal_threshold=args.goal_threshold,
            goal_bias=max(args.goal_bias, 0.10),
            neighbor_radius=args.neighbor_radius,
            edge_resolution=args.edge_resolution,
            ee_goal_threshold=args.ee_goal_threshold,
        )
        res_c = _run_task_goal_planner("cdf", cdf, start_q, goal_task, args)
        print_stats("CDFGuidedFrankaRRTStar (task-goal)", res_c.stats, res_c.path, res_c.task_goal_error)
        results.append(res_c)

    if args.planner in ("pullandslide", "all"):
        pull = PullAndSlideFrankaRRTStar(
            checker=checker,
            obstacles=obstacles,
            device=device,
            step_size=args.step_size,
            goal_threshold=args.goal_threshold,
            goal_bias=max(args.goal_bias, 0.12),
            neighbor_radius=args.neighbor_radius,
            edge_resolution=args.edge_resolution,
            ee_goal_threshold=args.ee_goal_threshold,
        )
        res_p = _run_task_goal_planner("pullandslide", pull, start_q, goal_task, args)
        print_stats("PullAndSlideFrankaRRTStar (task-goal)", res_p.stats, res_p.path, res_p.task_goal_error)
        results.append(res_p)

    success = [r for r in results if r.path is not None and bool(r.stats.get("success"))]
    if not success:
        if args.demo:
            print(
                "No planner produced a valid path; opening PyBullet scene anyway "
                "to visualize obstacle placements, start pose, and task goal."
            )
            visualize_task_space_pybullet(
                path=start_q.reshape(1, 7),
                obstacles=obstacles,
                goal_task=goal_task,
                hold_last_sec=args.demo_hold_sec,
                label="No feasible path (scene-only view)",
            )
        raise SystemExit("No planner produced a valid path. Try increasing --max-iters or relaxing scene/goal.")

    best = min(success, key=lambda r: float(r.stats.get("final_path_cost", float("inf"))))
    print(f"\nBest successful planner: {best.planner_name} (cost={best.stats.get('final_path_cost')})")

    if args.demo:
        print(f"Launching PyBullet task-space animation for planner: {best.planner_name}")
        visualize_task_space_pybullet(
            path=best.path,
            obstacles=obstacles,
            goal_task=goal_task,
            hold_last_sec=args.demo_hold_sec,
            label=f"{best.planner_name} | task-goal",
        )


if __name__ == "__main__":
    main()
