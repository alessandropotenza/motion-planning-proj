#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# -----------------------------------------------------------------------------

import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from cdf_guided_rrtstar import (
    CDF_RRTStar,
    CDF_RRTStar_EE,
    SAFETY_MARGIN_TASK_SPACE,
    SAFETY_MARGIN_C_SPACE,
    SOFTMIN_BETA,
    Vanilla_RRTStar,
)
from rrt_star_2d import make_scene
import robot_plot2D


# ---------------------------------------------------------------------------
# Scene defaults
# ---------------------------------------------------------------------------

SCENE_START_GOAL: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "scene_1": (np.array([-2.0, -1.0], dtype=np.float32), np.array([1.5,  1.2], dtype=np.float32)),
    "scene_2": (np.array([-2.4,  0.2], dtype=np.float32), np.array([2.3, -0.8], dtype=np.float32)),
    # scene_3 — "Dense Upper": navigate through tightly-packed upper obstacles.
    "scene_3": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5,  2.5], dtype=np.float32)),
    # scene_4 — "Diagonal Block": two large corner obstacles + right-side blocker.
    "scene_4": (np.array([-2.0,  1.0], dtype=np.float32), np.array([2.5,  0.5], dtype=np.float32)),
    # scene_5 — "Scattered Perimeter": five moderate perimeter obstacles.
    "scene_5": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5,  2.5], dtype=np.float32)),
    # "scene_6": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5,  2.5], dtype=np.float32)),
}

# Task-space EE goals aligned to the joint-space goals above (same target region,
# expressed as an end-effector Cartesian position).
SCENE_START_EE_GOAL: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "scene_1": (np.array([-2.0, -1.0], dtype=np.float32), np.array([-1.5, 2.8], dtype=np.float32)),
    "scene_2": (np.array([-2.4,  0.2], dtype=np.float32), np.array([-1.0, 3.3], dtype=np.float32)),
    # EE ≈ FK(goal_q) for each scene:
    #   scene_3 goal q=[0.5, 2.5] → EE ≈ (-0.22, 1.24)
    "scene_3": (np.array([-0.5, -2.5], dtype=np.float32), np.array([-0.2, 1.2], dtype=np.float32)),
    #   scene_4 goal q=[2.5, 0.5] → EE ≈ (-3.58, 1.48); start q=[-2.0, 1.0]
    "scene_4": (np.array([-2.0,  1.0], dtype=np.float32), np.array([-3.5, 1.5], dtype=np.float32)),
    #   scene_5 goal q=[0.5, 2.5] → EE ≈ (-0.22, 1.24); start q=[-0.5, -2.5]
    "scene_5": (np.array([-0.5, -2.5], dtype=np.float32), np.array([-0.2, 1.2], dtype=np.float32)),
    #   scene_6 goal q=[0.5, 2.5] → EE ≈ (-0.22, 1.24); start q=[-0.5, -2.5]
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Single-seed evaluation: Vanilla RRT* vs CDF-guided RRT*.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=2500)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5", "scene_6", "both", "all"],
        default="both",
    )

    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--goal-threshold", type=float, default=0.25)
    parser.add_argument("--goal-bias", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--edge-resolution", type=float, default=0.05)

    parser.add_argument("--safety-margin-task-space", type=float, default=SAFETY_MARGIN_TASK_SPACE)
    parser.add_argument("--safety-margin-c-space", type=float, default=SAFETY_MARGIN_C_SPACE)
    parser.add_argument("--softmin-beta", type=float, default=SOFTMIN_BETA)

    parser.add_argument("--ee-goal", nargs=2, type=float, metavar=("X", "Y"),
                        help="Task-space EE goal (overrides joint-space defaults).")
    parser.add_argument("--ee-goal-threshold", type=float, default=0.15,
                        help="EE distance threshold for task-space goal reaching.")
    parser.add_argument("--ee-model-path", type=str, default=None,
                        help="Path to trained EE CDF model (model_ee.pth).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_tree(
    ax,
    cdf: CDF2D,
    obj_list,
    nodes,
    path,
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
    hide_goal_dot: bool = False,
):
    """C-space view: obstacle CDF heat-map + RRT* tree + path."""
    cdf.plot_cdf(ax, obj_list)
    for node in nodes:
        if node.parent is None:
            continue
        parent = nodes[node.parent]
        ax.plot(
            [parent.q[0], node.q[0]],
            [parent.q[1], node.q[1]],
            color="dimgray",
            linewidth=0.7,
            alpha=0.5,
        )
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.0, label="Path")
    ax.plot(start[0], start[1], "go", markersize=8, label="Start")
    if not hide_goal_dot:
        ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)


def plot_task_space(
    ax,
    cdf: CDF2D,
    obj_list,
    path: Optional[np.ndarray],
    start_q: np.ndarray,
    goal: np.ndarray,
    is_ee_goal: bool,
    title: str,
):
    """Task-space view: obstacles + planned arm motion + goal marker."""
    cdf.plot_objects(ax, obj_list)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("x", size=13)
    ax.set_ylabel("y", size=13)
    ax.set_title(title, size=13)

    # Robot base marker
    ax.plot(0, 0, "ko", markersize=5, zorder=10)

    if path is not None and len(path) >= 2:
        # All intermediate poses in light blue, start/end highlighted in orange
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=path,
            ax=ax,
            color="tab:blue",
            show_start_end=False,
            show_eef_traj=True,
        )
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.vstack([path[0], path[-1]]),
            ax=ax,
            color="tab:orange",
            show_start_end=True,
            show_eef_traj=False,
        )
    else:
        # No path — show start configuration only
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.atleast_2d(start_q),
            ax=ax,
            color="tab:orange",
            show_start_end=True,
            show_eef_traj=False,
        )

    if is_ee_goal:
        # Mark the task-space goal as a red star
        ax.plot(goal[0], goal[1], "r*", markersize=18, label="EE Goal", zorder=11)
        ax.legend(loc="upper right", fontsize=9)
    else:
        # Show the arm at the goal joint configuration in dark green
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.atleast_2d(goal),
            ax=ax,
            color="darkgreen",
            show_start_end=True,
            show_eef_traj=False,
        )


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_stats(scene_name: str, planner_name: str, stats: dict):
    print(f"\n[{scene_name}] {planner_name}")
    print(f"success:                 {stats['success']}")
    print(f"time_to_first_path_sec:  {stats['time_to_first_path_sec']}")
    print(f"nodes_to_first_path:     {stats['nodes_to_first_path']}")
    print(f"final_path_cost:         {stats['final_path_cost']}")
    print(f"accepted_nodes:          {stats['accepted_nodes']}")
    print(f"discarded_samples:       {stats['discarded_samples']}")
    print(f"rejection_rate:          {stats['rejection_rate']:.4f}")
    print(f"first_goal_iteration:    {stats['first_goal_iteration']}")
    print(f"planning_time_sec:       {stats['planning_time_sec']:.4f}")


# ---------------------------------------------------------------------------
# IK helper
# ---------------------------------------------------------------------------

def _pick_ik_goal(
    cdf: CDF2D,
    obj_list,
    p_goal: np.ndarray,
    start: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find IK solutions for *p_goal*, return (best_q, all_valid_qs)."""
    p_t = torch.tensor(p_goal, dtype=torch.float32, device=device)
    _, q_solutions, _ = cdf.find_q_ee(p_t)
    q_np = q_solutions.detach().cpu().numpy().astype(np.float32)

    valid = []
    for q in q_np:
        q_t = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
        sdf = cdf.inference_sdf(q_t, obj_list).item()
        if sdf > 0.0:
            valid.append(q)

    if len(valid) == 0:
        raise RuntimeError(
            f"No collision-free IK solution found for EE goal {p_goal}. "
            "Try a different goal or increase the IK batch size."
        )

    valid_np = np.array(valid, dtype=np.float32)
    dists = np.linalg.norm(valid_np - start, axis=1)
    best_idx = int(np.argmin(dists))
    return valid_np[best_idx], valid_np


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(scene_name: str, args, ee_goal: Optional[np.ndarray]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    obj_list = make_scene(scene_name, device)
    q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
    q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)

    if ee_goal is not None:
        start = SCENE_START_EE_GOAL[scene_name][0]
        p_goal = ee_goal
    else:
        start, goal_q = SCENE_START_GOAL[scene_name]
        p_goal = None

    shared_kwargs = dict(
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
    )

    ik_solutions: Optional[np.ndarray] = None

    if p_goal is not None:
        q_ik_goal, ik_solutions = _pick_ik_goal(cdf, obj_list, p_goal, start, device)
        print(f"  IK for vanilla: {len(ik_solutions)} collision-free solutions, "
              f"best q = [{q_ik_goal[0]:.3f}, {q_ik_goal[1]:.3f}]")

        vanilla = Vanilla_RRTStar(
            cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
            goal_bias=args.goal_bias, **shared_kwargs,
        )
        vanilla_goal = q_ik_goal

        cdf_rrt = CDF_RRTStar_EE(
            cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
            goal_bias=0.10,
            safety_margin_task_space=args.safety_margin_task_space,
            safety_margin_c_space=args.safety_margin_c_space,
            softmin_beta=args.softmin_beta,
            ee_model_path=args.ee_model_path,
            ee_goal_threshold=args.ee_goal_threshold,
            **shared_kwargs,
        )
        cdf_goal = p_goal  # task-space
    else:
        vanilla = Vanilla_RRTStar(
            cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
            goal_bias=args.goal_bias, **shared_kwargs,
        )
        vanilla_goal = goal_q

        cdf_rrt = CDF_RRTStar(
            cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
            goal_bias=0.10,
            safety_margin_task_space=args.safety_margin_task_space,
            safety_margin_c_space=args.safety_margin_c_space,
            softmin_beta=args.softmin_beta,
            **shared_kwargs,
        )
        cdf_goal = goal_q

    vanilla_nodes, vanilla_path, vanilla_stats = vanilla.plan(
        start=start, goal=vanilla_goal, max_iters=args.max_iters, seed=args.seed
    )
    cdf_nodes, cdf_path, cdf_stats = cdf_rrt.plan(
        start=start, goal=cdf_goal, max_iters=args.max_iters, seed=args.seed
    )

    print_stats(scene_name, "Vanilla_RRTStar", vanilla_stats)
    print_stats(scene_name, "CDF_RRTStar" + ("_EE" if p_goal is not None else ""), cdf_stats)

    is_ee = p_goal is not None

    # --- Figure 1: C-space obstacle CDF + tree ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig1.suptitle(f"{scene_name}: Configuration Space", fontsize=14)
    plot_tree(
        ax1, cdf=cdf, obj_list=obj_list,
        nodes=vanilla_nodes, path=vanilla_path,
        start=start, goal=vanilla_goal,
        title="Vanilla RRT*",
        hide_goal_dot=False,
    )
    plot_tree(
        ax2, cdf=cdf, obj_list=obj_list,
        nodes=cdf_nodes, path=cdf_path,
        start=start,
        # For EE mode cdf_goal is a task-space point — don't mark it in C-space
        goal=vanilla_goal if is_ee else cdf_goal,
        title="CDF-guided RRT*" + (" (EE)" if is_ee else ""),
        hide_goal_dot=is_ee,
    )
    fig1.tight_layout()

    # --- Figure 2: Task-space motion + obstacles ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(f"{scene_name}: Task Space", fontsize=14)
    plot_task_space(
        ax3, cdf=cdf, obj_list=obj_list,
        path=vanilla_path,
        start_q=start,
        goal=vanilla_goal if not is_ee else p_goal,
        is_ee_goal=is_ee,
        title="Vanilla RRT*" + (" (IK goal)" if is_ee else ""),
    )
    plot_task_space(
        ax4, cdf=cdf, obj_list=obj_list,
        path=cdf_path,
        start_q=start,
        goal=p_goal if is_ee else cdf_goal,
        is_ee_goal=is_ee,
        title="CDF-guided RRT*" + (" (EE goal)" if is_ee else ""),
    )
    fig2.tight_layout()

    # --- Figure 3 (EE mode only): EE goal CDF ---
    if is_ee:
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 7))
        fig3.suptitle(f"{scene_name}: EE Goal CDF", fontsize=14)
        cdf.plot_ee_goal_cdf(ax5, p_goal, ik_solutions=ik_solutions)
        ax5.plot(start[0], start[1], "go", markersize=8, label="Start")
        ax5.set_title("Vanilla (IK solutions as red ★)")
        ax5.legend(loc="upper right", fontsize=9)

        cdf.plot_ee_goal_cdf(ax6, p_goal)
        ax6.plot(start[0], start[1], "go", markersize=8, label="Start")
        ax6.set_title("CDF-guided (goal manifold)")
        ax6.legend(loc="upper right", fontsize=9)
        fig3.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenes: List[str]
    if args.scene == "both":
        scenes = ["scene_1", "scene_2"]
    elif args.scene == "all":
        scenes = ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"]
    else:
        scenes = [args.scene]

    ee_goal: Optional[np.ndarray] = None
    if args.ee_goal is not None:
        ee_goal = np.array(args.ee_goal, dtype=np.float32)

    mode_str = "Task-space EE goal" if ee_goal is not None else "Joint-space goal"
    print(f"=== CDF-Guided RRT* Evaluation ({mode_str}, Single Seed) ===")
    print(f"seed: {args.seed}, max_iters: {args.max_iters}, scenes: {', '.join(scenes)}")
    if ee_goal is not None:
        print(f"ee_goal: [{ee_goal[0]:.3f}, {ee_goal[1]:.3f}]")

    for scene_name in scenes:
        evaluate_scene(scene_name, args, ee_goal)

    plt.show()


if __name__ == "__main__":
    main()
