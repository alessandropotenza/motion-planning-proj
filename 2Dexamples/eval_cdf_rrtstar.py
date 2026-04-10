#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# -----------------------------------------------------------------------------

import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from cdf_guided_rrtstar import (
    CDF_RRTStar,
    SAFETY_MARGIN_TASK_SPACE,
    SAFETY_MARGIN_C_SPACE,
    SOFTMIN_BETA,
    Vanilla_RRTStar,
)
from rrt_star_2d import make_scene


SCENE_START_GOAL: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "scene_1": (np.array([-2.0, -1.0], dtype=np.float32), np.array([1.5, 1.2], dtype=np.float32)),
    "scene_2": (np.array([-2.4, 0.2], dtype=np.float32), np.array([2.3, -0.8], dtype=np.float32)),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-seed evaluation: Vanilla RRT* vs CDF-guided RRT*.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=2500)
    parser.add_argument("--scene", choices=["scene_1", "scene_2", "both"], default="both")

    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--goal-threshold", type=float, default=0.25)
    parser.add_argument("--goal-bias", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--edge-resolution", type=float, default=0.05)

    parser.add_argument("--safety-margin-task-space", type=float, default=SAFETY_MARGIN_TASK_SPACE)
    parser.add_argument("--safety-margin-c-space", type=float, default=SAFETY_MARGIN_C_SPACE)
    parser.add_argument("--softmin-beta", type=float, default=SOFTMIN_BETA)
    return parser.parse_args()


def plot_tree(ax, cdf: CDF2D, obj_list, nodes, path, start, goal, title: str):
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
    ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)


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


def evaluate_scene(scene_name: str, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    obj_list = make_scene(scene_name, device)
    q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
    q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)
    start, goal = SCENE_START_GOAL[scene_name]

    vanilla = Vanilla_RRTStar(
        cdf=cdf,
        obj_list=obj_list,
        q_min=q_min,
        q_max=q_max,
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=args.goal_bias,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
    )

    cdf_rrt = CDF_RRTStar(
        cdf=cdf,
        obj_list=obj_list,
        q_min=q_min,
        q_max=q_max,
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=0.10,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
        safety_margin_task_space=args.safety_margin_task_space,
        safety_margin_c_space=args.safety_margin_c_space,
        softmin_beta=args.softmin_beta,
    )

    vanilla_nodes, vanilla_path, vanilla_stats = vanilla.plan(
        start=start, goal=goal, max_iters=args.max_iters, seed=args.seed
    )
    cdf_nodes, cdf_path, cdf_stats = cdf_rrt.plan(
        start=start, goal=goal, max_iters=args.max_iters, seed=args.seed
    )

    print_stats(scene_name, "Vanilla_RRTStar", vanilla_stats)
    print_stats(scene_name, "CDF_RRTStar", cdf_stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_tree(
        ax1,
        cdf=cdf,
        obj_list=obj_list,
        nodes=vanilla_nodes,
        path=vanilla_path,
        start=start,
        goal=goal,
        title=f"{scene_name}: Vanilla RRT*",
    )
    plot_tree(
        ax2,
        cdf=cdf,
        obj_list=obj_list,
        nodes=cdf_nodes,
        path=cdf_path,
        start=start,
        goal=goal,
        title=f"{scene_name}: CDF-guided RRT*",
    )
    plt.tight_layout()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scenes: List[str]
    if args.scene == "both":
        scenes = ["scene_1", "scene_2"]
    else:
        scenes = [args.scene]

    print("=== CDF-Guided RRT* Evaluation (Single Seed) ===")
    print(f"seed: {args.seed}, max_iters: {args.max_iters}, scenes: {', '.join(scenes)}")
    for scene_name in scenes:
        evaluate_scene(scene_name, args)

    plt.show()


if __name__ == "__main__":
    main()
