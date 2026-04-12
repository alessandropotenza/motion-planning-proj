#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""Franka Panda: vanilla RRT* vs CDF-guided RRT*, then optional PyBullet playback.

Collision checking defaults to analytic FK + sphere proxies (``sphere_arm_collision``).
With ``--collision-backend pin``, Pinocchio + hpp-fcl use URDF collision meshes instead
(see ``pin_fcl_collision.py`` and ``requirements-franka-pinocchio.txt``).

PyBullet is optional and used **only** to replay the joint-space path and draw obstacles.

Requirements:
  - ``frankaemika/model_dict.pt`` for CDF-guided mode (same as other Franka demos).
  - Run from this directory **or** set ``PYTHONPATH`` to include ``frankaemika``.

Example::

    cd frankaemika
    python plan_and_demo_franka.py --mode both --scene demo_table --demo
    python plan_and_demo_franka.py --mode vanilla --demo --demo-best-effort
    python plan_and_demo_franka.py --collision-backend pin --mode vanilla --scene demo_table
    python plan_and_demo_franka.py --mode vanilla --log-every 100 --max-iters 5000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple, cast

import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import torch

from cdf_guided_rrtstar_franka import CDFGuidedFrankaRRTStar
from rrt_star_franka import (
    Node,
    VanillaFrankaRRTStar,
    path_to_tree_node_nearest_goal,
    path_waypoint_cost,
    print_stats_franka,
)
from franka_kinematics import fk_flange_position
from sphere_arm_collision import SphereArmCollisionChecker
from workspace_obstacles import BoxObstacle, Obstacle, SphereObstacle, build_demo_obstacles


def _try_default_start_goal(scene: str) -> Tuple[np.ndarray, np.ndarray]:
    """Hand-picked joint pairs that are usually collision-free for ``demo_table``."""
    if scene == "demo_table":
        start = np.array(
            [0.0, -0.25, 0.0, -2.15, 0.0, 1.85, 0.785], dtype=np.float32
        )
        goal = np.array(
            [0.35, -0.35, 0.55, -2.35, -0.15, 1.95, 0.9], dtype=np.float32
        )
        return start, goal
    if scene == "sparse":
        start = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 1.6, 0.0], dtype=np.float32)
        goal = np.array([0.8, -0.5, 0.5, -2.0, 0.0, 2.2, 0.5], dtype=np.float32)
        return start, goal
    start = np.zeros(7, dtype=np.float32)
    goal = np.array([0.5, -0.5, 0.5, -2.0, 0.0, 2.0, 0.0], dtype=np.float32)
    return start, goal


def sample_collision_free_pair(
    checker: SphereArmCollisionChecker, rng: np.random.Generator, max_tries: int = 20000
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(max_tries):
        a = rng.uniform(checker.q_min, checker.q_max).astype(np.float32)
        b = rng.uniform(checker.q_min, checker.q_max).astype(np.float32)
        if checker.is_state_free(a) and checker.is_state_free(b):
            return a, b
    raise RuntimeError("Could not sample a collision-free start/goal pair; relax obstacles or bounds.")


def _spawn_obstacle_visuals(p, obstacles: List[Obstacle]) -> List[int]:
    """Create PyBullet bodies that *mirror* analytic obstacles (visualization only)."""
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


def _spawn_goal_marker_pybullet(p, goal_q: np.ndarray, radius: float = 0.04) -> int:
    """Green sphere at the goal end-effector (flange) position from FK (visualization only)."""
    pos = fk_flange_position(np.asarray(goal_q, dtype=np.float64).reshape(7))
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=[0.08, 0.82, 0.18, 1.0],
        specularColor=[0.2, 0.2, 0.2, 1],
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis,
        basePosition=pos.tolist(),
    )


def visualize_path_pybullet(
    path: np.ndarray,
    obstacles: List[Obstacle],
    hz: float = 20.0,
    hold_last_sec: float = 2.0,
    debug_label: Optional[str] = None,
    goal_q: Optional[np.ndarray] = None,
    goal_marker_radius: float = 0.04,
) -> None:
    import pybullet as p
    import pybullet_data as pd

    from pybullet_panda_sim import PandaSim

    dt = 1.0 / hz
    p.connect(p.GUI, options="--width=1200 --height=900")
    p.setAdditionalSearchPath(pd.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    _spawn_obstacle_visuals(p, obstacles)

    if goal_q is not None:
        _spawn_goal_marker_pybullet(p, goal_q, radius=goal_marker_radius)

    robot = PandaSim(p, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

    if debug_label:
        p.addUserDebugText(debug_label, [0.15, 0.0, 1.05], textColorRGB=(1, 0.2, 0.2), textSize=1.2)

    path = np.asarray(path, dtype=np.float32)
    if path.ndim == 1:
        path = path.reshape(1, 7)
    dense: List[np.ndarray] = []
    if len(path) == 1:
        dense = [path[0].copy()]
    else:
        for i in range(len(path) - 1):
            n = max(2, int(np.linalg.norm(path[i + 1] - path[i]) / 0.04) + 1)
            for a in np.linspace(0.0, 1.0, n, endpoint=(i == len(path) - 2)):
                dense.append(((1.0 - a) * path[i] + a * path[i + 1]).astype(np.float32))
    dense_arr = np.stack(dense, axis=0)

    for q in dense_arr:
        robot.set_joint_positions(q)
        p.stepSimulation()
        time.sleep(dt * 0.5)

    t_end = time.time() + hold_last_sec
    while time.time() < t_end:
        p.stepSimulation()
        time.sleep(dt)
    p.disconnect()


def parse_args():
    ap = argparse.ArgumentParser(description="Franka RRT* / CDF-guided RRT* + optional PyBullet demo.")
    ap.add_argument("--mode", choices=["vanilla", "cdf", "both"], default="vanilla")
    ap.add_argument("--scene",choices=["demo_table", "sparse", "pillar_and_box"], type=str, default="demo_table")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=800)
    ap.add_argument("--step-size", type=float, default=0.12)
    ap.add_argument("--goal-threshold", type=float, default=0.35)
    ap.add_argument("--goal-bias", type=float, default=0.06)
    ap.add_argument("--neighbor-radius", type=float, default=0.65)
    ap.add_argument("--edge-resolution", type=float, default=0.06)
    ap.add_argument("--demo", action="store_true", help="Open PyBullet after planning (visualization only).")
    ap.add_argument(
        "--demo-best-effort",
        action="store_true",
        help="With --demo: if no feasible path, animate the tree path to the node closest to the goal (joint L2).",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (default: auto).")
    ap.add_argument("--auto-start-goal", action="store_true", help="Random collision-free start/goal if defaults fail.")
    ap.add_argument(
        "--collision-backend",
        choices=["sphere", "pin"],
        default="sphere",
        help="sphere: analytic FK + sphere soup; pin: Pinocchio + hpp-fcl URDF collision meshes.",
    )
    ap.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Panda URDF path (only used for --collision-backend pin; default: frankaemika/panda_urdf/panda.urdf).",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=0,
        metavar="N",
        help="Print RRT* progress every N iterations (0 = disabled). Example: --log-every 100",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    obstacles = build_demo_obstacles(args.scene, rng)
    if args.collision_backend == "sphere":
        checker = SphereArmCollisionChecker(obstacles, margin=0.02)
    else:
        from pin_fcl_collision import DEFAULT_URDF, PinFclCollisionChecker

        urdf = args.urdf_path or DEFAULT_URDF
        checker = PinFclCollisionChecker(obstacles, margin=0.02, urdf_path=urdf)

    start, goal = _try_default_start_goal(args.scene)
    if args.auto_start_goal or not (checker.is_state_free(start) and checker.is_state_free(goal)):
        if not args.auto_start_goal:
            print("Default start/goal invalid for this scene; auto-sampling collision-free pair.")
        start, goal = sample_collision_free_pair(checker, rng)

    shared = dict(
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
    )

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    print("=== Franka motion planning (joint-space RRT*) ===")
    print(f"scene={args.scene}, seed={args.seed}, device={device}, collision={args.collision_backend}")
    print(f"start q: {np.array2string(start, precision=3)}")
    print(f"goal  q: {np.array2string(goal, precision=3)}")

    path_to_show: Optional[np.ndarray] = None
    path_v: Optional[np.ndarray] = None
    path_c: Optional[np.ndarray] = None
    nodes_v: Optional[List[Node]] = None
    nodes_c: Optional[List[Node]] = None

    log_kw = {}
    if args.log_every > 0:
        log_kw = dict(log_every_iters=args.log_every)

    if args.mode in ("vanilla", "both"):
        vanilla = VanillaFrankaRRTStar(checker, goal_bias=args.goal_bias, **shared)
        nodes_v, path_v, stats_v = vanilla.plan(
            start=start,
            goal=goal,
            max_iters=args.max_iters,
            seed=args.seed,
            log_prefix="[vanilla] ",
            **log_kw,
        )
        print_stats_franka(args.scene, "VanillaFrankaRRTStar", stats_v)
        print(f"path_waypoint_cost: {path_waypoint_cost(path_v)}")

    if args.mode in ("cdf", "both"):
        cdf_planner = CDFGuidedFrankaRRTStar(
            checker,
            obstacles,
            device=device,
            goal_bias=0.10,
            **shared,
        )
        nodes_c, path_c, stats_c = cdf_planner.plan(
            start=start,
            goal=goal,
            max_iters=args.max_iters,
            seed=args.seed,
            log_prefix="[cdf] ",
            **log_kw,
        )
        print_stats_franka(args.scene, "CDFGuidedFrankaRRTStar", stats_c)
        print(f"path_waypoint_cost: {path_waypoint_cost(path_c)}")

    if path_c is not None:
        path_to_show = path_c
    elif path_v is not None:
        path_to_show = path_v

    if args.demo_best_effort and not args.demo:
        print("Note: --demo-best-effort only applies together with --demo (PyBullet was not opened).")

    if args.demo:
        if path_to_show is not None:
            print("\nLaunching PyBullet visualization (feasible solution path).")
            visualize_path_pybullet(path_to_show, obstacles, debug_label="Feasible path", goal_q=goal)
        elif args.demo_best_effort:
            nodes_be = _select_tree_for_best_effort(
                cast(List[Node], nodes_v) if nodes_v is not None else None,
                cast(List[Node], nodes_c) if nodes_c is not None else None,
                goal,
                args.mode,
            )
            if nodes_be is None or len(nodes_be) == 0:
                print("No tree to visualize.")
                return
            be_path, _be_idx, be_dist = path_to_tree_node_nearest_goal(nodes_be, goal)
            print(
                f"\nNo feasible path; best-effort visualization: tree path to closest node "
                f"(joint L2 to goal = {be_dist:.4f} rad). This path is not guaranteed collision-free beyond the tree."
            )
            visualize_path_pybullet(
                be_path,
                obstacles,
                debug_label=f"Best effort (min ||q-goal|| = {be_dist:.2f})",
                goal_q=goal,
            )
        else:
            print("No feasible path to visualize; use --demo-best-effort with --demo to see the closest tree path,")
            print("or re-run with more iterations / --auto-start-goal.")


def _select_tree_for_best_effort(
    nodes_v: Optional[List[Node]],
    nodes_c: Optional[List[Node]],
    goal: np.ndarray,
    mode: str,
) -> Optional[List[Node]]:
    if mode == "vanilla":
        return nodes_v
    if mode == "cdf":
        return nodes_c
    if nodes_v is None:
        return nodes_c
    if nodes_c is None:
        return nodes_v
    _, _, dv = path_to_tree_node_nearest_goal(nodes_v, goal)
    _, _, dc = path_to_tree_node_nearest_goal(nodes_c, goal)
    return nodes_v if dv <= dc else nodes_c


if __name__ == "__main__":
    main()
