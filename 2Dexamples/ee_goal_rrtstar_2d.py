#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Standalone 2D RRT* planners for task-space end-effector goals.
# -----------------------------------------------------------------------------

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from mlp import MLPRegression  # noqa: F401  # Needed when loading serialized MLP models.
import robot_plot2D
from rrt_star_2d import make_scene


CUR_PATH = os.path.dirname(os.path.realpath(__file__))

# Legacy scene defaults are in joint space. We keep them for start fallback and
# derive task-space goals via FK when --goal-task is not provided.
SCENE_DEFAULT_Q: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "scene_1": (np.array([-2.0, -1.0], dtype=np.float32), np.array([1.5, 1.2], dtype=np.float32)),
    "scene_2": (np.array([-2.4, 0.2], dtype=np.float32), np.array([2.3, -0.8], dtype=np.float32)),
    "scene_3": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5, 2.5], dtype=np.float32)),
    "scene_4": (np.array([-2.0, 1.0], dtype=np.float32), np.array([2.5, 0.5], dtype=np.float32)),
    "scene_5": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5, 2.5], dtype=np.float32)),
}


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int]
    cost: float


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def fk_end_effector(cdf: CDF2D, q: np.ndarray) -> np.ndarray:
    q_t = torch.tensor(q, dtype=torch.float32, device=cdf.device).reshape(1, 2)
    with torch.no_grad():
        ee = cdf.robot.forward_kinematics_eef(q_t).squeeze(0)
    return ee.detach().cpu().numpy().astype(np.float32)


def batched_fk_end_effector(cdf: CDF2D, path_q: np.ndarray) -> np.ndarray:
    q_t = torch.tensor(path_q, dtype=torch.float32, device=cdf.device)
    with torch.no_grad():
        ee = cdf.robot.forward_kinematics_eef(q_t)
    return ee.detach().cpu().numpy().astype(np.float32)


class RRTStarBase:
    """Minimal readable RRT* base loop with hook points for goal semantics."""

    def __init__(
        self,
        cdf: CDF2D,
        obj_list,
        q_min: np.ndarray,
        q_max: np.ndarray,
        step_size: float = 0.25,
        goal_threshold: float = 0.25,
        goal_bias: float = 0.05,
        neighbor_radius: float = 0.5,
        edge_resolution: float = 0.05,
    ) -> None:
        self.cdf = cdf
        self.obj_list = obj_list
        self.device = cdf.device
        self.q_min = np.asarray(q_min, dtype=np.float32)
        self.q_max = np.asarray(q_max, dtype=np.float32)

        self.step_size = float(step_size)
        self.goal_threshold = float(goal_threshold)
        self.goal_bias = float(goal_bias)
        self.neighbor_radius = float(neighbor_radius)
        self.edge_resolution = float(edge_resolution)

    def in_bounds(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.q_min) and np.all(q <= self.q_max))

    def clamp_to_bounds(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max).astype(np.float32)

    def is_state_collision_free(self, q: np.ndarray) -> bool:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 2)
        sdf = self.cdf.inference_sdf(q_t, self.obj_list).item()
        return sdf > 0.0

    def is_edge_collision_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        dist = euclidean(q_from, q_to)
        num_samples = max(2, int(np.ceil(dist / self.edge_resolution)) + 1)
        alphas = np.linspace(0.0, 1.0, num_samples)
        qs = np.outer(1.0 - alphas, q_from) + np.outer(alphas, q_to)
        q_t = torch.tensor(qs, dtype=torch.float32, device=self.device)
        sdf = self.cdf.inference_sdf(q_t, self.obj_list)
        return bool(torch.all(sdf > 0.0).item())

    def nearest_index(self, nodes: List[Node], q_sample: np.ndarray, goal_idx: Optional[int]) -> int:
        best_idx = -1
        best_dist = np.inf
        for idx, node in enumerate(nodes):
            if goal_idx is not None and idx == goal_idx:
                continue
            d = euclidean(node.q, q_sample)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def nearby_indices(self, nodes: List[Node], q: np.ndarray) -> List[int]:
        return [i for i, node in enumerate(nodes) if euclidean(node.q, q) <= self.neighbor_radius]

    def update_descendant_costs(self, nodes: List[Node], parent_idx: int) -> None:
        stack = [parent_idx]
        while stack:
            current = stack.pop()
            current_cost = nodes[current].cost
            current_q = nodes[current].q
            for child_idx, child in enumerate(nodes):
                if child.parent == current:
                    edge_cost = euclidean(current_q, child.q)
                    nodes[child_idx].cost = current_cost + edge_cost
                    stack.append(child_idx)

    def extract_path(self, nodes: List[Node], goal_idx: int) -> np.ndarray:
        path = []
        idx: Optional[int] = goal_idx
        while idx is not None:
            path.append(nodes[idx].q)
            idx = nodes[idx].parent
        path.reverse()
        return np.asarray(path, dtype=np.float32)

    def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.goal_bias:
            return goal.copy().astype(np.float32)
        return rng.uniform(self.q_min, self.q_max).astype(np.float32)

    def rollout_edge(
        self,
        q_from: np.ndarray,
        q_target: np.ndarray,
        max_extension: Optional[float],
    ) -> Tuple[np.ndarray, float, bool, bool]:
        direction = q_target - q_from
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            return q_from.copy(), 0.0, True, True

        travel = dist if max_extension is None else min(dist, max_extension)
        q_end = q_from + (direction / dist) * travel
        q_end = self.clamp_to_bounds(q_end)

        collision_free = self.is_edge_collision_free(q_from, q_end)
        reached_target = abs(travel - dist) <= 1e-8
        return q_end.astype(np.float32), float(travel), collision_free, reached_target

    # Hooks for planner-specific goal semantics.
    def _validate_goal(self, goal: np.ndarray) -> None:
        if not self.in_bounds(goal):
            raise ValueError("Goal is out of bounds.")
        if not self.is_state_collision_free(goal):
            raise ValueError("Goal is in collision.")

    def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
        return euclidean(q_new, goal) <= self.goal_threshold

    def _make_goal_connection(
        self,
        new_idx: int,
        nodes: List[Node],
        goal: np.ndarray,
        goal_idx: Optional[int],
        rewires: int,
    ) -> Tuple[Optional[int], int]:
        _, edge_cost, ok, reached = self.rollout_edge(nodes[new_idx].q, goal, max_extension=None)
        if not (ok and reached):
            return goal_idx, rewires

        goal_cost = nodes[new_idx].cost + edge_cost
        if goal_idx is None:
            goal_idx = len(nodes)
            nodes.append(Node(q=goal.copy(), parent=new_idx, cost=goal_cost))
        elif goal_cost < nodes[goal_idx].cost:
            nodes[goal_idx].parent = new_idx
            nodes[goal_idx].cost = goal_cost
            rewires += 1
            self.update_descendant_costs(nodes, goal_idx)
        return goal_idx, rewires

    def plan(
        self,
        start_q: np.ndarray,
        goal: np.ndarray,
        max_iters: int,
        seed: int,
    ) -> Tuple[List[Node], Optional[np.ndarray], dict]:
        rng = np.random.default_rng(seed)
        start_q = np.asarray(start_q, dtype=np.float32).reshape(2)
        goal = np.asarray(goal, dtype=np.float32).reshape(2)

        if not self.in_bounds(start_q):
            raise ValueError("Start is out of bounds.")
        if not self.is_state_collision_free(start_q):
            raise ValueError("Start is in collision.")
        self._validate_goal(goal)

        nodes = [Node(q=start_q.copy(), parent=None, cost=0.0)]
        goal_idx: Optional[int] = None
        rejected_samples = 0
        rewires = 0
        t0 = time.time()

        for _ in range(max_iters):
            q_target = self.sample_target(goal, rng)
            nearest_idx = self.nearest_index(nodes, q_target, goal_idx)

            q_new, edge_cost, ok, _ = self.rollout_edge(
                nodes[nearest_idx].q,
                q_target,
                max_extension=self.step_size,
            )

            no_progress = euclidean(q_new, nodes[nearest_idx].q) < 1e-8
            if (not ok) or (not self.in_bounds(q_new)) or no_progress:
                rejected_samples += 1
                continue

            near_idxs = self.nearby_indices(nodes, q_new)
            if nearest_idx not in near_idxs:
                near_idxs.append(nearest_idx)

            best_parent = nearest_idx
            best_cost = nodes[nearest_idx].cost + edge_cost

            for idx in near_idxs:
                if goal_idx is not None and idx == goal_idx:
                    continue
                _, conn_cost, conn_ok, reached = self.rollout_edge(nodes[idx].q, q_new, max_extension=None)
                if not (conn_ok and reached):
                    continue
                candidate_cost = nodes[idx].cost + conn_cost
                if candidate_cost < best_cost:
                    best_parent = idx
                    best_cost = candidate_cost

            new_idx = len(nodes)
            nodes.append(Node(q=q_new, parent=best_parent, cost=best_cost))

            for idx in near_idxs:
                if idx == best_parent or (goal_idx is not None and idx == goal_idx):
                    continue
                _, rewire_cost, rewire_ok, reached = self.rollout_edge(nodes[new_idx].q, nodes[idx].q, max_extension=None)
                if not (rewire_ok and reached):
                    continue
                new_cost = nodes[new_idx].cost + rewire_cost
                if new_cost < nodes[idx].cost:
                    nodes[idx].parent = new_idx
                    nodes[idx].cost = new_cost
                    rewires += 1
                    self.update_descendant_costs(nodes, idx)

            if self._is_near_goal(q_new, goal):
                goal_idx, rewires = self._make_goal_connection(new_idx, nodes, goal, goal_idx, rewires)

        planning_time = time.time() - t0
        path = self.extract_path(nodes, goal_idx) if goal_idx is not None else None
        final_path_cost = float(nodes[goal_idx].cost) if goal_idx is not None else None

        accepted_nodes = len(nodes) - 1
        denom = accepted_nodes + rejected_samples
        rejection_rate = (rejected_samples / denom) if denom > 0 else 0.0

        stats = {
            "success": goal_idx is not None,
            "iterations": max_iters,
            "accepted_nodes": accepted_nodes,
            "discarded_samples": rejected_samples,
            "rejection_rate": rejection_rate,
            "rewires": rewires,
            "planning_time_sec": planning_time,
            "final_path_cost": final_path_cost,
        }
        return nodes, path, stats


class VanillaEERRTStar(RRTStarBase):
    """Vanilla RRT* with task-space goal input and private IK-to-joint-goal logic."""

    def _pick_collision_free_ik_goal(
        self,
        start_q: np.ndarray,
        goal_task: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_t = torch.tensor(goal_task, dtype=torch.float32, device=self.device)
        _, q_solutions, _ = self.cdf.find_q_ee(target_t)
        q_np = q_solutions.detach().cpu().numpy().astype(np.float32)

        valid = []
        for q in q_np:
            if self.is_state_collision_free(q):
                valid.append(q)

        if not valid:
            raise RuntimeError(
                f"No collision-free IK solution found for task goal [{goal_task[0]:.3f}, {goal_task[1]:.3f}]"
            )

        valid_np = np.asarray(valid, dtype=np.float32)
        dists = np.linalg.norm(valid_np - start_q.reshape(1, 2), axis=1)
        best_idx = int(np.argmin(dists))
        return valid_np[best_idx], valid_np

    def solve(
        self,
        start_q: np.ndarray,
        goal_task: np.ndarray,
        max_iters: int,
        seed: int,
    ) -> dict:
        goal_q, ik_solutions = self._pick_collision_free_ik_goal(start_q, goal_task)
        nodes, path, stats = self.plan(start_q=start_q, goal=goal_q, max_iters=max_iters, seed=seed)
        return {
            "nodes": nodes,
            "path": path,
            "stats": stats,
            "config_goal_marker": goal_q,
            "ik_solutions": ik_solutions,
        }


class CDFEERRTStar(RRTStarBase):
    """CDF-guided RRT* for task-space EE goals using obstacle and EE CDF models."""

    def __init__(
        self,
        cdf: CDF2D,
        obj_list,
        q_min: np.ndarray,
        q_max: np.ndarray,
        step_size: float = 0.25,
        goal_threshold: float = 0.25,
        goal_bias: float = 0.10,
        neighbor_radius: float = 0.5,
        edge_resolution: float = 0.05,
        safety_margin_c_space: float = 0.2,
        softmin_beta: float = 50.0,
        model_path: Optional[str] = None,
        ee_model_path: Optional[str] = None,
        ee_goal_threshold: float = 0.15,
        oracle_points_per_obj: int = 120,
    ) -> None:
        super().__init__(
            cdf=cdf,
            obj_list=obj_list,
            q_min=q_min,
            q_max=q_max,
            step_size=step_size,
            goal_threshold=goal_threshold,
            goal_bias=goal_bias,
            neighbor_radius=neighbor_radius,
            edge_resolution=edge_resolution,
        )
        self.safety_margin_c_space = float(safety_margin_c_space)
        self.softmin_beta = float(softmin_beta)
        self.ee_goal_threshold = float(ee_goal_threshold)

        if model_path is None:
            model_path = os.path.join(CUR_PATH, "model.pth")
        if ee_model_path is None:
            ee_model_path = os.path.join(CUR_PATH, "model_ee.pth")

        self.net_obs = torch.load(model_path, map_location=self.device, weights_only=False)
        self.net_obs.eval()
        self.net_obs.requires_grad_(False)

        self.net_ee = torch.load(ee_model_path, map_location=self.device, weights_only=False)
        self.net_ee.eval()
        self.net_ee.requires_grad_(False)

        self.oracle_points_per_obj = int(oracle_points_per_obj)
        self.oracle_points = self._build_oracle_points().to(self.device)

        self.goal_task: Optional[np.ndarray] = None

    def _build_oracle_points(self) -> torch.Tensor:
        point_sets = []
        for obj in self.obj_list:
            if hasattr(obj, "center") and hasattr(obj, "radius"):
                center = obj.center.detach().cpu().float()
                radius = float(obj.radius)
                theta = torch.linspace(0.0, 2.0 * np.pi, self.oracle_points_per_obj + 1)[:-1]
                pts = torch.stack(
                    [
                        center[0] + radius * torch.cos(theta),
                        center[1] + radius * torch.sin(theta),
                    ],
                    dim=1,
                )
            elif hasattr(obj, "sample_surface"):
                pts = obj.sample_surface(self.oracle_points_per_obj).detach().cpu().float()
            else:
                raise ValueError("Unsupported obstacle type for oracle point generation.")
            point_sets.append(pts)
        return torch.cat(point_sets, dim=0)

    def _softmin(self, distances: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(-self.softmin_beta * distances, dim=0) / self.softmin_beta

    def get_cdf_distance(self, q: np.ndarray) -> float:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 2)
        with torch.no_grad():
            q_rep = q_t.repeat(self.oracle_points.size(0), 1)
            net_input = torch.cat([self.oracle_points, q_rep], dim=1)
            d_all = self.net_obs.forward(net_input).squeeze()
            fused_dist = self._softmin(d_all)

            sdf_val = self.cdf.inference_sdf(q_t, self.obj_list)
            signed_dist = float(np.sign(sdf_val.item())) * float(fused_dist.item())
        return signed_dist

    def get_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 2).requires_grad_(True)
        q_rep = q_t.repeat(self.oracle_points.size(0), 1)
        net_input = torch.cat([self.oracle_points, q_rep], dim=1)

        d_all = self.net_obs.forward(net_input).squeeze()
        fused_dist = self._softmin(d_all)
        grad = torch.autograd.grad(fused_dist, q_t, retain_graph=False, create_graph=False)[0].squeeze(0)

        q_sdf = q_t.detach().clone().requires_grad_(True)
        sdf_val, sdf_grad = self.cdf.inference_sdf(q_sdf, self.obj_list, return_grad=True)
        signed_dist = float(np.sign(sdf_val.item())) * float(fused_dist.detach().item())

        grad_np = grad.detach().cpu().numpy()
        sdf_grad_np = sdf_grad.squeeze(0).detach().cpu().numpy()
        if np.dot(grad_np, sdf_grad_np) < 0.0:
            grad_np = -grad_np

        return signed_dist, safe_normalize(grad_np)

    def get_ee_cdf_data(self, q: np.ndarray, goal_task: np.ndarray) -> Tuple[float, np.ndarray]:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 2).requires_grad_(True)
        p_t = torch.tensor(goal_task, dtype=torch.float32, device=self.device).reshape(1, 2)
        net_input = torch.cat([p_t, q_t], dim=1)
        dist = self.net_ee(net_input).squeeze()
        grad = torch.autograd.grad(dist, q_t, retain_graph=False, create_graph=False)[0].squeeze(0)
        grad_np = grad.detach().cpu().numpy()
        return float(dist.item()), safe_normalize(grad_np)

    def _validate_goal(self, goal: np.ndarray) -> None:
        # In EE-goal mode, "goal" is a dummy placeholder. We validate start only.
        return

    def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
        ee = fk_end_effector(self.cdf, q_new)
        return euclidean(ee, self.goal_task) <= self.ee_goal_threshold

    def _make_goal_connection(
        self,
        new_idx: int,
        nodes: List[Node],
        goal: np.ndarray,
        goal_idx: Optional[int],
        rewires: int,
    ) -> Tuple[Optional[int], int]:
        # The new node itself satisfies the task-space goal condition.
        goal_cost = nodes[new_idx].cost
        if goal_idx is None:
            goal_idx = new_idx
        elif goal_cost < nodes[goal_idx].cost:
            goal_idx = new_idx
            rewires += 1
        return goal_idx, rewires

    def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.goal_bias:
            # Goal-biased EE projection: random q -> project onto EE-goal manifold.
            q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
            d_ee, g_ee = self.get_ee_cdf_data(q_rand, self.goal_task)
            q = q_rand - d_ee * g_ee
            q = self.clamp_to_bounds(q)
            return q

        # Otherwise use obstacle-aware random sampling.
        q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
        cdf_dist = self.get_cdf_distance(q_rand)
        if cdf_dist > self.safety_margin_c_space:
            return q_rand

        d_obs, g_obs = self.get_cdf_data(q_rand)
        q_proj = q_rand + (self.safety_margin_c_space - d_obs) * g_obs
        return self.clamp_to_bounds(q_proj)

    def rollout_edge(
        self,
        q_from: np.ndarray,
        q_target: np.ndarray,
        max_extension: Optional[float],
    ) -> Tuple[np.ndarray, float, bool, bool]:
        q_end, travel, collision_free, reached_target = super().rollout_edge(q_from, q_target, max_extension)

        if not collision_free or travel < 1e-10:
            return q_end, travel, collision_free, reached_target
        if max_extension is None:
            return q_end, travel, collision_free, reached_target

        d_obs, g_obs = self.get_cdf_data(q_end)
        if d_obs >= self.safety_margin_c_space:
            return q_end, travel, collision_free, reached_target

        q_projected = q_end + (self.safety_margin_c_space - d_obs) * g_obs
        q_projected = self.clamp_to_bounds(q_projected)
        if not self.is_edge_collision_free(q_from, q_projected):
            new_travel = euclidean(q_from, q_projected)
            return q_projected, new_travel, False, False

        new_travel = euclidean(q_from, q_projected)
        reached_target_proj = euclidean(q_projected, q_target) <= 1e-6
        return q_projected, new_travel, True, reached_target_proj

    def solve(
        self,
        start_q: np.ndarray,
        goal_task: np.ndarray,
        max_iters: int,
        seed: int,
    ) -> dict:
        self.goal_task = np.asarray(goal_task, dtype=np.float32).reshape(2)
        dummy_goal = np.asarray(start_q, dtype=np.float32).reshape(2).copy()
        nodes, path, stats = self.plan(start_q=start_q, goal=dummy_goal, max_iters=max_iters, seed=seed)

        # For CDF planner, C-space goal marker is the final path configuration.
        config_goal_marker = None
        if path is not None and len(path) > 0:
            config_goal_marker = path[-1].copy()

        return {
            "nodes": nodes,
            "path": path,
            "stats": stats,
            "config_goal_marker": config_goal_marker,
            "ik_solutions": None,
        }


def print_stats(planner_name: str, stats: dict, path: Optional[np.ndarray]) -> None:
    print("\n=== RRT* Planning Statistics ===")
    print(f"planner:                 {planner_name}")
    print(f"success:                 {stats['success']}")
    print(f"iterations:              {stats['iterations']}")
    print(f"accepted_nodes:          {stats['accepted_nodes']}")
    print(f"discarded_samples:       {stats['discarded_samples']}")
    print(f"rejection_rate:          {stats['rejection_rate']:.4f}")
    print(f"rewires:                 {stats['rewires']}")
    print(f"planning_time_sec:       {stats['planning_time_sec']:.4f}")

    if stats["final_path_cost"] is not None:
        print(f"final_path_cost:         {stats['final_path_cost']:.4f}")
    else:
        print("final_path_cost:         N/A")

    if path is None:
        print("path_waypoints:          0")
    else:
        print(f"path_waypoints:          {len(path)}")


def plot_configuration_space(
    cdf: CDF2D,
    obj_list,
    nodes: List[Node],
    start_q: np.ndarray,
    config_goal_marker: Optional[np.ndarray],
    path: Optional[np.ndarray],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    cdf.plot_cdf(ax, obj_list)

    # Draw the final planning tree first so the path stays visually on top.
    tree_label_added = False
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
            label="Tree" if not tree_label_added else None,
        )
        tree_label_added = True

    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5, label="Path")

    ax.plot(start_q[0], start_q[1], "go", markersize=9, label="Start (q)")
    if config_goal_marker is not None:
        ax.plot(
            config_goal_marker[0],
            config_goal_marker[1],
            "bo",
            markersize=9,
            label="Goal (q)",
        )

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()


def plot_task_space(
    cdf: CDF2D,
    obj_list,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    path_q: Optional[np.ndarray],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    cdf.plot_objects(ax, obj_list)
    ax.set_aspect("equal", "box")
    ax.set_xlim((-4.0, 4.0))
    ax.set_ylim((-4.0, 4.0))

    start_ee = fk_end_effector(cdf, start_q)
    ax.plot(start_ee[0], start_ee[1], "go", markersize=9, label="Start (task)")
    ax.plot(goal_task[0], goal_task[1], "bo", markersize=9, label="Goal (task)")

    if path_q is not None and len(path_q) > 0:
        ee_path = batched_fk_end_effector(cdf, path_q)
        ax.plot(
            ee_path[:, 0],
            ee_path[:, 1],
            linestyle="--",
            color="tab:red",
            linewidth=2.0,
            label="EE path",
        )

        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=path_q,
            ax=ax,
            color="tab:blue",
            show_start_end=False,
            show_eef_traj=False,
        )

        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.vstack([path_q[0], path_q[-1]]),
            ax=ax,
            color="tab:orange",
            show_start_end=True,
            show_eef_traj=False,
        )
    else:
        robot_plot2D.plot_2d_manipulators(
            joint_angles_batch=np.asarray([start_q], dtype=np.float32),
            ax=ax,
            color="tab:orange",
            show_start_end=True,
            show_eef_traj=False,
        )

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple 2D EE-goal RRT*: choose Vanilla (with internal IK) or CDF-guided mode."
    )

    parser.add_argument("--planner", choices=["vanilla", "cdf"], default="cdf")
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default="scene_1",
    )
    parser.add_argument("--start", nargs=2, type=float, metavar=("Q1", "Q2"))
    parser.add_argument("--goal-task", nargs=2, type=float, metavar=("X", "Y"))

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=2500)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--goal-threshold", type=float, default=0.25)
    parser.add_argument("--goal-bias", type=float, default=0.10)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--edge-resolution", type=float, default=0.05)

    parser.add_argument("--safety-margin-c-space", type=float, default=0.2)
    parser.add_argument("--softmin-beta", type=float, default=50.0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ee-model-path", type=str, default=None)
    parser.add_argument("--ee-goal-threshold", type=float, default=0.15)

    return parser.parse_args()


def resolve_start_and_goal_task(args, cdf: CDF2D) -> Tuple[np.ndarray, np.ndarray]:
    default_start_q, default_goal_q = SCENE_DEFAULT_Q[args.scene]

    if args.start is None:
        start_q = default_start_q.copy()
    else:
        start_q = np.asarray(args.start, dtype=np.float32)

    if args.goal_task is None:
        goal_task = fk_end_effector(cdf, default_goal_q)
    else:
        goal_task = np.asarray(args.goal_task, dtype=np.float32)

    return start_q, goal_task


def validate_start(planner: RRTStarBase, start_q: np.ndarray) -> None:
    if not planner.in_bounds(start_q):
        q_min = planner.q_min
        q_max = planner.q_max
        raise SystemExit(
            "Error: Start is out of bounds. "
            f"Expected each joint in [{q_min[0]:.3f}, {q_max[0]:.3f}]."
        )
    if not planner.is_state_collision_free(start_q):
        raise SystemExit("Error: Start configuration is in collision.")


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    obj_list = make_scene(args.scene, device)

    q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
    q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)
    start_q, goal_task = resolve_start_and_goal_task(args, cdf)

    planner_kwargs = dict(
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

    if args.planner == "vanilla":
        planner_name = "VanillaEERRTStar"
        planner = VanillaEERRTStar(**planner_kwargs)
    else:
        planner_name = "CDFEERRTStar"
        planner = CDFEERRTStar(
            **planner_kwargs,
            safety_margin_c_space=args.safety_margin_c_space,
            softmin_beta=args.softmin_beta,
            model_path=args.model_path,
            ee_model_path=args.ee_model_path,
            ee_goal_threshold=args.ee_goal_threshold,
        )

    validate_start(planner, start_q)

    print("=== EE-Goal 2D RRT* ===")
    print(f"planner: {planner_name}")
    print(f"scene: {args.scene}")
    print(f"seed: {args.seed}, max_iters: {args.max_iters}")
    print(f"start_q: [{start_q[0]:.3f}, {start_q[1]:.3f}]")
    print(f"goal_task: [{goal_task[0]:.3f}, {goal_task[1]:.3f}]")

    try:
        result = planner.solve(
            start_q=start_q,
            goal_task=goal_task,
            max_iters=args.max_iters,
            seed=args.seed,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    path_q = result["path"]
    print_stats(planner_name, result["stats"], path_q)

    plot_configuration_space(
        cdf=cdf,
        obj_list=obj_list,
        nodes=result["nodes"],
        start_q=start_q,
        config_goal_marker=result["config_goal_marker"],
        path=path_q,
        title=f"{args.scene}: {planner_name} (Configuration Space)",
    )

    plot_task_space(
        cdf=cdf,
        obj_list=obj_list,
        start_q=start_q,
        goal_task=goal_task,
        path_q=path_q,
        title=f"{args.scene}: {planner_name} (Task Space)",
    )

    plt.show()


if __name__ == "__main__":
    main()
