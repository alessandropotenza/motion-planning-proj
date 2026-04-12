#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# -----------------------------------------------------------------------------

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from primitives2D_torch import Circle
import robot_plot2D


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int]
    cost: float


def make_scene(scene_name: str, device: torch.device):
    if scene_name == "scene_1":
        return [Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device)]
    if scene_name == "scene_2":
        return [
            Circle(center=torch.tensor([2.3, -2.3]), radius=0.3, device=device),
            Circle(center=torch.tensor([0.0, 2.45]), radius=0.3, device=device),
        ]
    if scene_name == "scene_3":
        # "Dense Upper" — five obstacles filling the upper and lateral workspace.
        # The arm starts in the lower region and must reach a configuration in the
        # upper region, threading between tightly-packed obstacles.  This maps to a
        # heavily fragmented C-space obstacle that is hard for uniform sampling.
        return [
            Circle(center=torch.tensor([2.2,  2.2]), radius=0.8, device=device),
            Circle(center=torch.tensor([0.0,  3.0]), radius=0.7, device=device),
            Circle(center=torch.tensor([-2.2, 2.2]), radius=0.8, device=device),
            Circle(center=torch.tensor([3.0,  0.0]), radius=0.6, device=device),
            Circle(center=torch.tensor([-3.0, 0.0]), radius=0.6, device=device),
        ]
    if scene_name == "scene_4":
        # "Diagonal Block" — two large circles occupy opposite corners (upper-right
        # and lower-left), with a small blocker on the right side.  The arm must
        # navigate along the anti-diagonal band of C-space from the upper-left to
        # the lower-right region.  Verified solvable (~26% rejection rate).
        return [
            Circle(center=torch.tensor([ 2.5,  2.5]), radius=1.3, device=device),
            Circle(center=torch.tensor([-2.5, -2.5]), radius=1.3, device=device),
            Circle(center=torch.tensor([ 2.8, -1.5]), radius=0.6, device=device),
        ]
    if scene_name == "scene_5":
        # "Scattered Perimeter" — five moderate obstacles arranged around the
        # workspace perimeter.  Each obstacle blocks a different arm reach direction,
        # creating a multi-obstacle C-space challenge (~31% rejection rate).
        return [
            Circle(center=torch.tensor([ 2.0,  2.0]), radius=0.7, device=device),
            Circle(center=torch.tensor([-2.0,  2.0]), radius=0.7, device=device),
            Circle(center=torch.tensor([ 0.0, -3.0]), radius=0.7, device=device),
            Circle(center=torch.tensor([ 3.0, -1.0]), radius=0.6, device=device),
            Circle(center=torch.tensor([-3.0, -1.0]), radius=0.6, device=device),
        ]
    if scene_name == "scene_6":
        # "Scattered Perimeter" — five moderate obstacles arranged around the
        # workspace perimeter.  Each obstacle blocks a different arm reach direction,
        # creating a multi-obstacle C-space challenge (~31% rejection rate).
        return [
            # Circle(center=torch.tensor([ 2.2,  0.0]), radius=0.1, device=device),
            # Circle(center=torch.tensor([-2.0,  2.0]), radius=0.7, device=device),
            # Circle(center=torch.tensor([ 0.0, -3.0]), radius=0.7, device=device),
            Circle(center=torch.tensor([2.5 , 0.0]),  radius=0.3, device=device),
            Circle(center=torch.tensor([3.0 , 0.0]),  radius=0.3, device=device),
            Circle(center=torch.tensor([3.5 , 0.0]),  radius=0.3, device=device),
        ]
    raise ValueError(f"Unknown scene: {scene_name}")


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def steer(from_q: np.ndarray, to_q: np.ndarray, step_size: float) -> np.ndarray:
    direction = to_q - from_q
    distance = np.linalg.norm(direction)
    if distance <= step_size:
        return to_q.copy()
    return from_q + direction / distance * step_size


def in_bounds(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> bool:
    return bool(np.all(q >= q_min) and np.all(q <= q_max))


def is_state_collision_free(cdf: CDF2D, obj_list, q: np.ndarray, device: torch.device) -> bool:
    q_t = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
    sdf = cdf.inference_sdf(q_t, obj_list).item()
    return sdf > 0.0


def is_edge_collision_free(
    cdf: CDF2D,
    obj_list,
    q_from: np.ndarray,
    q_to: np.ndarray,
    edge_resolution: float,
    device: torch.device,
) -> bool:
    dist = euclidean(q_from, q_to)
    num_samples = max(2, int(math.ceil(dist / edge_resolution)) + 1)
    alphas = np.linspace(0.0, 1.0, num_samples)
    qs = np.outer(1.0 - alphas, q_from) + np.outer(alphas, q_to)
    q_t = torch.tensor(qs, dtype=torch.float32, device=device)
    sdf = cdf.inference_sdf(q_t, obj_list)
    return bool(torch.all(sdf > 0.0).item())


def nearest_node_index(nodes: List[Node], q_sample: np.ndarray) -> int:
    distances = [euclidean(node.q, q_sample) for node in nodes]
    return int(np.argmin(distances))


def nearby_node_indices(nodes: List[Node], q_new: np.ndarray, radius: float) -> List[int]:
    return [i for i, node in enumerate(nodes) if euclidean(node.q, q_new) <= radius]


def update_descendant_costs(nodes: List[Node], parent_idx: int) -> None:
    stack = [parent_idx]
    while stack:
        current_idx = stack.pop()
        current_q = nodes[current_idx].q
        current_cost = nodes[current_idx].cost
        for child_idx, child in enumerate(nodes):
            if child.parent == current_idx:
                edge_cost = euclidean(current_q, child.q)
                nodes[child_idx].cost = current_cost + edge_cost
                stack.append(child_idx)


def extract_path(nodes: List[Node], goal_idx: int) -> np.ndarray:
    path = []
    idx = goal_idx
    while idx is not None:
        path.append(nodes[idx].q)
        idx = nodes[idx].parent
    path.reverse()
    return np.asarray(path)


def rrt_star(
    cdf: CDF2D,
    obj_list,
    start: np.ndarray,
    goal: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    max_iters: int,
    step_size: float,
    goal_threshold: float,
    goal_bias: float,
    neighbor_radius: float,
    edge_resolution: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[List[Node], Optional[int], dict]:
    nodes = [Node(q=start.copy(), parent=None, cost=0.0)]
    collision_rejects = 0
    rewires = 0
    goal_idx = None
    first_goal_iteration = None

    for it in range(max_iters):
        if rng.random() < goal_bias:
            q_rand = goal.copy()
        else:
            q_rand = rng.uniform(q_min, q_max)

        nearest_idx = nearest_node_index(nodes, q_rand)
        if goal_idx is not None and nearest_idx == goal_idx:
            distances = [
                euclidean(node.q, q_rand) if idx != goal_idx else np.inf
                for idx, node in enumerate(nodes)
            ]
            nearest_idx = int(np.argmin(distances))
        q_new = steer(nodes[nearest_idx].q, q_rand, step_size)

        if not in_bounds(q_new, q_min, q_max):
            collision_rejects += 1
            continue

        if not is_edge_collision_free(
            cdf, obj_list, nodes[nearest_idx].q, q_new, edge_resolution, device
        ):
            collision_rejects += 1
            continue

        near_idxs = nearby_node_indices(nodes, q_new, neighbor_radius)
        if nearest_idx not in near_idxs:
            near_idxs.append(nearest_idx)

        best_parent = nearest_idx
        best_cost = nodes[nearest_idx].cost + euclidean(nodes[nearest_idx].q, q_new)
        for idx in near_idxs:
            if goal_idx is not None and idx == goal_idx:
                continue
            candidate_cost = nodes[idx].cost + euclidean(nodes[idx].q, q_new)
            if candidate_cost >= best_cost:
                continue
            if not is_edge_collision_free(cdf, obj_list, nodes[idx].q, q_new, edge_resolution, device):
                continue
            best_parent = idx
            best_cost = candidate_cost

        new_idx = len(nodes)
        nodes.append(Node(q=q_new, parent=best_parent, cost=best_cost))

        for idx in near_idxs:
            if idx == best_parent:
                continue
            new_cost = nodes[new_idx].cost + euclidean(nodes[new_idx].q, nodes[idx].q)
            if new_cost >= nodes[idx].cost:
                continue
            if not is_edge_collision_free(
                cdf, obj_list, nodes[new_idx].q, nodes[idx].q, edge_resolution, device
            ):
                continue
            nodes[idx].parent = new_idx
            nodes[idx].cost = new_cost
            rewires += 1
            update_descendant_costs(nodes, idx)

        if euclidean(q_new, goal) <= goal_threshold and is_edge_collision_free(
            cdf, obj_list, q_new, goal, edge_resolution, device
        ):
            goal_cost = nodes[new_idx].cost + euclidean(q_new, goal)
            if goal_idx is None:
                goal_idx = len(nodes)
                nodes.append(Node(q=goal.copy(), parent=new_idx, cost=goal_cost))
                first_goal_iteration = it + 1
            elif goal_cost < nodes[goal_idx].cost:
                nodes[goal_idx].parent = new_idx
                nodes[goal_idx].cost = goal_cost
                rewires += 1
                update_descendant_costs(nodes, goal_idx)

    stats = {
        "success": goal_idx is not None,
        "iterations": max_iters,
        "collision_rejects": collision_rejects,
        "rewires": rewires,
        "first_goal_iteration": first_goal_iteration,
    }
    return nodes, goal_idx, stats


def plot_results(
    cdf: CDF2D,
    obj_list,
    nodes: List[Node],
    path: Optional[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
) -> None:
    fig_c, ax_c = plt.subplots(figsize=(9, 8))
    cdf.plot_cdf(ax_c, obj_list)

    for node in nodes:
        if node.parent is None:
            continue
        parent = nodes[node.parent]
        ax_c.plot(
            [parent.q[0], node.q[0]],
            [parent.q[1], node.q[1]],
            color="dimgray",
            linewidth=0.7,
            alpha=0.5,
        )

    if path is not None:
        ax_c.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5, label="Planned path")

    ax_c.plot(start[0], start[1], "go", markersize=9, label="Start")
    ax_c.plot(goal[0], goal[1], "bo", markersize=9, label="Goal")
    ax_c.legend(loc="upper right")
    ax_c.set_title("Configuration Space: RRT* Tree and Path")

    fig_t, ax_t = plt.subplots(figsize=(9, 8))
    cdf.plot_objects(ax_t, obj_list)
    ax_t.set_aspect("equal", "box")
    ax_t.set_xlim((-4.0, 4.0))
    ax_t.set_ylim((-4.0, 4.0))
    ax_t.set_title("Task Space: Obstacles and Planned Motion")

    if path is not None:
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=path,
            ax=ax_t,
            color="tab:blue",
            show_start_end=False,
            show_eef_traj=True,
        )
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.vstack([path[0], path[-1]]),
            ax=ax_t,
            color="tab:orange",
            show_start_end=True,
            show_eef_traj=False,
        )

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Standard RRT* planner in 2D joint space.")
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("Q1", "Q2"))
    parser.add_argument("--goal", nargs=2, type=float, required=True, metavar=("Q1", "Q2"))
    parser.add_argument("--scene", type=str, default="scene_1", choices=["scene_1", "scene_2"])
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--goal-threshold", type=float, default=0.25)
    parser.add_argument("--goal-bias", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--edge-resolution", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    obj_list = make_scene(args.scene, device)

    start = np.asarray(args.start, dtype=np.float32)
    goal = np.asarray(args.goal, dtype=np.float32)
    q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
    q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)

    if not in_bounds(start, q_min, q_max):
        raise SystemExit(
            f"Error: Start is out of bounds. Expected each joint in [{-math.pi:.3f}, {math.pi:.3f}]."
        )
    if not in_bounds(goal, q_min, q_max):
        raise SystemExit(
            f"Error: Goal is out of bounds. Expected each joint in [{-math.pi:.3f}, {math.pi:.3f}]."
        )
    if not is_state_collision_free(cdf, obj_list, start, device):
        raise SystemExit("Error: Start configuration is in collision.")
    if not is_state_collision_free(cdf, obj_list, goal, device):
        raise SystemExit("Error: Goal configuration is in collision.")

    t0 = time.time()
    nodes, goal_idx, stats = rrt_star(
        cdf=cdf,
        obj_list=obj_list,
        start=start,
        goal=goal,
        q_min=q_min,
        q_max=q_max,
        max_iters=args.max_iters,
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=args.goal_bias,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
        rng=rng,
        device=device,
    )
    planning_time = time.time() - t0

    path = extract_path(nodes, goal_idx) if goal_idx is not None else None
    path_cost = None
    if path is not None:
        path_cost = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))

    print("\n=== RRT* Planning Statistics ===")
    print(f"success:            {stats['success']}")
    print(f"iterations:         {stats['iterations']}")
    print(f"nodes_in_tree:      {len(nodes)}")
    print(f"collision_rejects:  {stats['collision_rejects']}")
    print(f"rewires:            {stats['rewires']}")
    if stats["first_goal_iteration"] is not None:
        print(f"first_goal_iter:    {stats['first_goal_iteration']}")
    else:
        print("first_goal_iter:    N/A")
    print(f"planning_time_sec:  {planning_time:.4f}")
    if path is not None:
        print(f"path_cost:          {path_cost:.4f}")
        print(f"path_waypoints:     {len(path)}")
    else:
        print("path_cost:          N/A (no feasible path found)")
        print("path_waypoints:     0")

    plot_results(cdf, obj_list, nodes, path, start, goal)


if __name__ == "__main__":
    main()
