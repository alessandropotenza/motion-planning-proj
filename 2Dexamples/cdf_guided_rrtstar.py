#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# -----------------------------------------------------------------------------

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from mlp import MLPRegression  # Needed for torch.load() of model.pth
from rrt_star_2d import make_scene


SAFETY_MARGIN_TASK_SPACE = 0.1
SAFETY_MARGIN_C_SPACE = 0.2
SOFTMIN_BETA = 50.0

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
SCENE_START_GOAL: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "scene_1": (np.array([-2.0, -1.0], dtype=np.float32), np.array([1.5,  1.2], dtype=np.float32)),
    "scene_2": (np.array([-2.4,  0.2], dtype=np.float32), np.array([2.3, -0.8], dtype=np.float32)),
    # scene_3 — "Dense Upper": navigate through tightly-packed upper obstacles.
    "scene_3": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5, 2.5], dtype=np.float32)),
    # scene_4 — "Diagonal Block": two large corner obstacles + right-side blocker.
    "scene_4": (np.array([-2.0,  1.0], dtype=np.float32), np.array([2.5,  0.5], dtype=np.float32)),
    # scene_5 — "Scattered Perimeter": five moderate perimeter obstacles.
    "scene_5": (np.array([-0.5, -2.5], dtype=np.float32), np.array([0.5,  2.5], dtype=np.float32)),
}

StepCallback = Callable[[Dict[str, Any]], None]


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int]
    cost: float


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


class RRTStarBase:
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
        return np.clip(q, self.q_min, self.q_max)

    def is_state_collision_free(self, q: np.ndarray) -> bool:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)
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
        best_idx = None
        best_dist = np.inf
        for idx, node in enumerate(nodes):
            if goal_idx is not None and idx == goal_idx:
                continue
            d = euclidean(node.q, q_sample)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return int(best_idx)

    def nearby_indices(self, nodes: List[Node], q: np.ndarray) -> List[int]:
        return [i for i, node in enumerate(nodes) if euclidean(node.q, q) <= self.neighbor_radius]

    def update_descendant_costs(self, nodes: List[Node], parent_idx: int) -> None:
        stack = [parent_idx]
        while stack:
            current = stack.pop()
            for child_idx, child in enumerate(nodes):
                if child.parent == current:
                    edge = euclidean(nodes[current].q, child.q)
                    nodes[child_idx].cost = nodes[current].cost + edge
                    stack.append(child_idx)

    def extract_path(self, nodes: List[Node], goal_idx: int) -> np.ndarray:
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(nodes[idx].q)
            idx = nodes[idx].parent
        path.reverse()
        return np.asarray(path)

    def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.goal_bias:
            return goal.copy()
        return rng.uniform(self.q_min, self.q_max).astype(np.float32)

    def sample_target_with_info(
        self, goal: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        q_target = self.sample_target(goal, rng).astype(np.float32)
        return q_target, {
            "mode": "direct",
            "raw_sample": q_target.copy(),
            "used_sample": q_target.copy(),
            "projected": False,
        }

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
        q_end = q_from + direction / dist * travel
        q_end = self.clamp_to_bounds(q_end)
        collision_free = self.is_edge_collision_free(q_from, q_end)
        reached_target = abs(travel - dist) <= 1e-8
        return q_end.astype(np.float32), float(travel), collision_free, reached_target

    # ------------------------------------------------------------------
    # Overridable goal hooks (used by plan())
    # ------------------------------------------------------------------

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
        it: int,
        t0: float,
    ) -> Tuple[Optional[int], int, Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]:
        """Try to connect q_new to the goal.

        Returns (goal_idx, rewires, first_goal_iteration, nodes_to_first_path,
                 time_to_first_path_sec, goal_event).
        Unchanged values should be passed through from the caller.
        """
        _, goal_conn_cost, ok, reached = self.rollout_edge(
            nodes[new_idx].q, goal, max_extension=None
        )
        if not (ok and reached):
            return goal_idx, rewires, None, None, None, None

        goal_cost = nodes[new_idx].cost + goal_conn_cost
        goal_event: Optional[Dict[str, Any]] = None
        first_goal_iteration: Optional[int] = None
        nodes_to_first_path: Optional[int] = None
        time_to_first_path_sec: Optional[float] = None

        if goal_idx is None:
            goal_idx = len(nodes)
            nodes.append(Node(q=goal.copy(), parent=new_idx, cost=goal_cost))
            first_goal_iteration = it + 1
            nodes_to_first_path = len(nodes)
            time_to_first_path_sec = time.time() - t0
            goal_event = {
                "type": "added",
                "goal_idx": goal_idx,
                "parent_idx": new_idx,
                "goal_q": goal.copy(),
                "parent_q": nodes[new_idx].q.copy(),
            }
        elif goal_cost < nodes[goal_idx].cost:
            nodes[goal_idx].parent = new_idx
            nodes[goal_idx].cost = goal_cost
            rewires += 1
            goal_event = {
                "type": "rewired",
                "goal_idx": goal_idx,
                "parent_idx": new_idx,
                "goal_q": goal.copy(),
                "parent_q": nodes[new_idx].q.copy(),
            }
            self.update_descendant_costs(nodes, goal_idx)

        return goal_idx, rewires, first_goal_iteration, nodes_to_first_path, time_to_first_path_sec, goal_event

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iters: int,
        seed: int,
        callback: Optional[StepCallback] = None,
    ) -> Tuple[List[Node], Optional[np.ndarray], dict]:
        rng = np.random.default_rng(seed)
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        if not self.in_bounds(start):
            raise ValueError("Start is out of bounds.")
        if not self.is_state_collision_free(start):
            raise ValueError("Start is in collision.")
        self._validate_goal(goal)

        nodes = [Node(q=start.copy(), parent=None, cost=0.0)]
        goal_idx = None
        first_goal_iteration = None
        time_to_first_path_sec = None
        nodes_to_first_path = None

        rejected_samples = 0
        rewires = 0
        t0 = time.time()

        for it in range(max_iters):
            q_target, sampling_info = self.sample_target_with_info(goal, rng)
            nearest_idx = self.nearest_index(nodes, q_target, goal_idx)

            q_new, edge_cost, collision_free, _ = self.rollout_edge(
                nodes[nearest_idx].q, q_target, max_extension=self.step_size
            )
            event: Dict[str, Any] = {
                "iteration": it + 1,
                "max_iters": max_iters,
                "q_target": q_target.copy(),
                "sampling": sampling_info,
                "nearest_idx": nearest_idx,
                "accepted": False,
                "rejection_reason": None,
                "new_node_idx": None,
                "new_node_parent": None,
                "new_node_q": None,
                "new_parent_q": None,
                "rewired_nodes": [],
                "goal_event": None,
            }
            if (
                (not collision_free)
                or (not self.in_bounds(q_new))
                or euclidean(q_new, nodes[nearest_idx].q) < 1e-8
            ):
                rejected_samples += 1
                if not collision_free:
                    event["rejection_reason"] = "edge_collision"
                elif not self.in_bounds(q_new):
                    event["rejection_reason"] = "out_of_bounds"
                else:
                    event["rejection_reason"] = "no_progress"
                event["q_new"] = q_new.copy()
                event["tree_size"] = len(nodes)
                event["discarded_samples"] = rejected_samples
                event["accepted_nodes"] = len(nodes) - 1
                event["total_rewires"] = rewires
                event["elapsed_sec"] = time.time() - t0
                if callback is not None:
                    callback(event)
                continue

            near_idxs = self.nearby_indices(nodes, q_new)
            if nearest_idx not in near_idxs:
                near_idxs.append(nearest_idx)

            best_parent = nearest_idx
            best_cost = nodes[nearest_idx].cost + edge_cost
            for idx in near_idxs:
                if goal_idx is not None and idx == goal_idx:
                    continue
                _, conn_cost, ok, reached = self.rollout_edge(nodes[idx].q, q_new, max_extension=None)
                if not ok or not reached:
                    continue
                candidate_cost = nodes[idx].cost + conn_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = idx

            new_idx = len(nodes)
            nodes.append(Node(q=q_new, parent=best_parent, cost=best_cost))

            rewired_nodes = []
            for idx in near_idxs:
                if idx == best_parent or (goal_idx is not None and idx == goal_idx):
                    continue
                _, rewire_cost, ok, reached = self.rollout_edge(nodes[new_idx].q, nodes[idx].q, max_extension=None)
                if not ok or not reached:
                    continue
                new_cost = nodes[new_idx].cost + rewire_cost
                if new_cost < nodes[idx].cost:
                    old_parent = nodes[idx].parent
                    nodes[idx].parent = new_idx
                    nodes[idx].cost = new_cost
                    rewires += 1
                    rewired_nodes.append(
                        {
                            "node_idx": idx,
                            "old_parent": old_parent,
                            "new_parent": new_idx,
                            "node_q": nodes[idx].q.copy(),
                            "new_parent_q": nodes[new_idx].q.copy(),
                        }
                    )
                    self.update_descendant_costs(nodes, idx)

            goal_event = None
            if self._is_near_goal(q_new, goal):
                (
                    goal_idx, rewires,
                    _fgi, _nfp, _ttfp, goal_event,
                ) = self._make_goal_connection(
                    new_idx, nodes, goal, goal_idx, rewires, it, t0,
                )
                if _fgi is not None:
                    first_goal_iteration = _fgi
                if _nfp is not None:
                    nodes_to_first_path = _nfp
                if _ttfp is not None:
                    time_to_first_path_sec = _ttfp

            event["accepted"] = True
            event["q_new"] = q_new.copy()
            event["new_node_idx"] = new_idx
            event["new_node_parent"] = best_parent
            event["new_node_q"] = nodes[new_idx].q.copy()
            event["new_parent_q"] = nodes[best_parent].q.copy()
            event["rewired_nodes"] = rewired_nodes
            event["goal_event"] = goal_event
            event["tree_size"] = len(nodes)
            event["discarded_samples"] = rejected_samples
            event["accepted_nodes"] = len(nodes) - 1
            event["total_rewires"] = rewires
            event["elapsed_sec"] = time.time() - t0
            if callback is not None:
                callback(event)

        planning_time_sec = time.time() - t0
        path = self.extract_path(nodes, goal_idx) if goal_idx is not None else None
        final_path_cost = float(nodes[goal_idx].cost) if goal_idx is not None else None

        accepted_nodes = len(nodes) - 1
        denom = accepted_nodes + rejected_samples
        rejection_rate = (rejected_samples / denom) if denom > 0 else 0.0

        stats = {
            "success": goal_idx is not None,
            "iterations": max_iters,
            "first_goal_iteration": first_goal_iteration,
            "time_to_first_path_sec": time_to_first_path_sec,
            "nodes_to_first_path": nodes_to_first_path,
            "final_path_cost": final_path_cost,
            "accepted_nodes": accepted_nodes,
            "discarded_samples": rejected_samples,
            "rejection_rate": rejection_rate,
            "rewires": rewires,
            "planning_time_sec": planning_time_sec,
        }
        return nodes, path, stats


class Vanilla_RRTStar(RRTStarBase):
    pass


class CDF_RRTStar(RRTStarBase):
    """CDF-guided RRT* using one-step projection sampling.

    Uses the neural CDF to repair samples that are near or inside obstacles
    via single-step gradient projection onto a safety shell.  Edge extension
    and collision checking are inherited from the vanilla RRTStarBase
    (straight-line edges with analytic SDF).
    """

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
        safety_margin_task_space: float = SAFETY_MARGIN_TASK_SPACE,
        safety_margin_c_space: float = SAFETY_MARGIN_C_SPACE,
        softmin_beta: float = SOFTMIN_BETA,
        model_path: Optional[str] = None,
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
        self.safety_margin_task_space = float(safety_margin_task_space)
        self.safety_margin_c_space = float(safety_margin_c_space)
        self.softmin_beta = float(softmin_beta)
        self.oracle_points_per_obj = int(oracle_points_per_obj)

        if model_path is None:
            model_path = os.path.join(CUR_PATH, "model.pth")
        self.model_path = model_path
        self.net = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.net.eval()
        self.net.requires_grad_(False)

        self.oracle_points = self._build_oracle_points().to(self.device)

    # ------------------------------------------------------------------
    # Oracle points & softmin (unchanged)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # CDF queries
    # ------------------------------------------------------------------

    def _quick_sdf(self, q: np.ndarray) -> float:
        """Cheap analytic workspace SDF (FK + primitive SDF, no neural net)."""
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            sdf = self.cdf.inference_sdf(q_t, self.obj_list)
        return float(sdf.item())

    def get_cdf_distance(self, q: np.ndarray) -> float:
        """Forward-only CDF query (no gradient). Returns signed distance."""
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)
        pts = self.oracle_points
        with torch.no_grad():
            q_rep = q_t.repeat(pts.size(0), 1)
            net_input = torch.cat([pts, q_rep], dim=1)
            d_all = self.net.forward(net_input).squeeze()
            fused_dist = self._softmin(d_all)

            q_sdf = q_t.clone()
            sdf_val = self.cdf.inference_sdf(q_sdf, self.obj_list)
            signed_dist = float(np.sign(sdf_val.item())) * float(fused_dist.item())
        return signed_dist

    def get_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """CDF query returning signed distance and unit gradient."""
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
        pts = self.oracle_points
        q_rep = q_t.repeat(pts.size(0), 1)
        net_input = torch.cat([pts, q_rep], dim=1)

        d_all = self.net.forward(net_input).squeeze()
        fused_dist = self._softmin(d_all)
        grad = torch.autograd.grad(fused_dist, q_t, retain_graph=False, create_graph=False)[0].squeeze(0)

        q_sdf = q_t.detach().clone().requires_grad_(True)
        sdf_val, sdf_grad = self.cdf.inference_sdf(q_sdf, self.obj_list, return_grad=True)
        signed_dist = float(np.sign(sdf_val.item())) * float(fused_dist.detach().item())

        grad_np = grad.detach().cpu().numpy()
        sdf_grad_np = sdf_grad.squeeze(0).detach().cpu().numpy()
        if np.dot(grad_np, sdf_grad_np) < 0.0:
            grad_np = -grad_np
        grad_np = safe_normalize(grad_np)
        return signed_dist, grad_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Sampling (one-step projection, SDF-gated)
    # ------------------------------------------------------------------

    def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q_target, _ = self.sample_target_with_info(goal, rng)
        return q_target

    def sample_target_with_info(
        self, goal: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng.random() < self.goal_bias:
            goal_sample = goal.copy().astype(np.float32)
            return goal_sample, {
                "mode": "goal_bias",
                "raw_sample": goal_sample.copy(),
                "used_sample": goal_sample.copy(),
                "projected": False,
            }

        q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)

        cdf_dist = self.get_cdf_distance(q_rand)
        if cdf_dist > self.safety_margin_c_space:
            return q_rand, {
                "mode": "cdf_keep",
                "raw_sample": q_rand.copy(),
                "used_sample": q_rand.copy(),
                "projected": False,
                "cdf_distance": float(cdf_dist),
            }

        dist, grad = self.get_cdf_data(q_rand)
        q_proj = q_rand + (self.safety_margin_c_space - dist) * grad
        q_proj = self.clamp_to_bounds(q_proj)
        q_proj = q_proj.astype(np.float32)
        return q_proj, {
            "mode": "cdf_project",
            "raw_sample": q_rand.copy(),
            "used_sample": q_proj.copy(),
            "projected": True,
            "cdf_distance": float(dist),
        }

    # ------------------------------------------------------------------
    # Edge extension with safety-shell endpoint projection
    # ------------------------------------------------------------------

    def rollout_edge(
        self,
        q_from: np.ndarray,
        q_target: np.ndarray,
        max_extension: Optional[float],
    ) -> Tuple[np.ndarray, float, bool, bool]:
        # Compute the straight-line endpoint first (base logic: steer + bounds clamp).
        q_end, travel, collision_free, reached_target = super().rollout_edge(
            q_from, q_target, max_extension
        )

        # If the base already rejected the edge, or the endpoint didn't move, there
        # is nothing to project.
        if not collision_free or travel < 1e-10:
            return q_end, travel, collision_free, reached_target

        # Rewiring calls (max_extension is None) operate on already-accepted nodes,
        # so we skip the projection there to avoid displacing known-good positions.
        if max_extension is None:
            return q_end, travel, collision_free, reached_target

        # Check whether the new endpoint lands inside the safety shell.
        cdf_dist, grad = self.get_cdf_data(q_end)
        if cdf_dist >= self.safety_margin_c_space:
            # Already outside the shell -- nothing to do.
            return q_end, travel, collision_free, reached_target

        # Project q_end outward to the safety shell boundary.
        # grad points away from the nearest collision boundary (unit vector),
        # so adding (margin - dist) * grad moves the point to the shell surface.
        q_projected = q_end + (self.safety_margin_c_space - cdf_dist) * grad
        q_projected = self.clamp_to_bounds(q_projected).astype(np.float32)

        # Re-validate the modified chord q_from → q_projected.
        collision_free_proj = self.is_edge_collision_free(q_from, q_projected)
        if not collision_free_proj:
            # The projected chord cuts an obstacle; discard the extension entirely.
            return q_projected, float(euclidean(q_from, q_projected)), False, False

        new_travel = float(euclidean(q_from, q_projected))
        reached_target_proj = euclidean(q_projected, q_target) <= 1e-6
        return q_projected, new_travel, True, reached_target_proj


class CDF_RRTStar_EE(CDF_RRTStar):
    """CDF-guided RRT* with a task-space end-effector goal.

    Loads a second neural CDF (model_ee.pth) that maps [p, q] -> distance
    to the IK manifold of workspace point *p*.  Goal-biased samples are
    projected onto the goal manifold via the EE CDF gradient, then repaired
    with the obstacle CDF if needed.  Goal checking uses FK distance
    instead of joint-space Euclidean distance.
    """

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
        safety_margin_task_space: float = SAFETY_MARGIN_TASK_SPACE,
        safety_margin_c_space: float = SAFETY_MARGIN_C_SPACE,
        softmin_beta: float = SOFTMIN_BETA,
        model_path: Optional[str] = None,
        oracle_points_per_obj: int = 120,
        ee_model_path: Optional[str] = None,
        ee_goal_threshold: float = 0.15,
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
            safety_margin_task_space=safety_margin_task_space,
            safety_margin_c_space=safety_margin_c_space,
            softmin_beta=softmin_beta,
            model_path=model_path,
            oracle_points_per_obj=oracle_points_per_obj,
        )
        self.ee_goal_threshold = float(ee_goal_threshold)
        if ee_model_path is None:
            ee_model_path = os.path.join(CUR_PATH, "model_ee.pth")
        self.net_ee = torch.load(ee_model_path, map_location=self.device, weights_only=False)
        self.net_ee.eval()
        self.net_ee.requires_grad_(False)
        self.p_goal_task: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # EE CDF query
    # ------------------------------------------------------------------

    def get_ee_cdf_data(
        self, q: np.ndarray, p_goal: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Single-oracle EE CDF: input [p_goal | q] -> (dist, unit_grad_wrt_q)."""
        q_t = (
            torch.tensor(q, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        p_t = torch.tensor(p_goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        net_input = torch.cat([p_t, q_t], dim=1)
        dist = self.net_ee(net_input).squeeze()
        grad = torch.autograd.grad(dist, q_t, retain_graph=False, create_graph=False)[0].squeeze(0)
        return float(dist.item()), safe_normalize(grad.detach().cpu().numpy()).astype(np.float32)

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _validate_goal(self, goal: np.ndarray) -> None:
        pass

    def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
        q_t = torch.tensor(q_new, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            ee = self.cdf.robot.forward_kinematics_eef(q_t).squeeze()
        ee_np = ee.cpu().numpy()
        return float(np.linalg.norm(ee_np - self.p_goal_task)) <= self.ee_goal_threshold

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
        """The newly added tree node *is* the goal — zero-length connection."""
        goal_cost = nodes[new_idx].cost
        goal_event: Optional[Dict[str, Any]] = None
        first_goal_iteration: Optional[int] = None
        nodes_to_first_path: Optional[int] = None
        time_to_first_path_sec: Optional[float] = None

        if goal_idx is None:
            goal_idx = new_idx
            first_goal_iteration = it + 1
            nodes_to_first_path = len(nodes)
            time_to_first_path_sec = time.time() - t0
            goal_event = {
                "type": "added",
                "goal_idx": goal_idx,
                "parent_idx": nodes[new_idx].parent,
                "goal_q": nodes[new_idx].q.copy(),
                "parent_q": nodes[nodes[new_idx].parent].q.copy(),
            }
        elif goal_cost < nodes[goal_idx].cost:
            goal_idx = new_idx
            rewires += 1
            goal_event = {
                "type": "rewired",
                "goal_idx": goal_idx,
                "parent_idx": nodes[new_idx].parent,
                "goal_q": nodes[new_idx].q.copy(),
                "parent_q": nodes[nodes[new_idx].parent].q.copy(),
            }

        return goal_idx, rewires, first_goal_iteration, nodes_to_first_path, time_to_first_path_sec, goal_event

    # ------------------------------------------------------------------
    # Sampling: goal bias uses EE CDF projection + obstacle repair
    # ------------------------------------------------------------------

    def sample_target_with_info(
        self, goal: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng.random() < self.goal_bias:
            q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
            d_ee, g_ee = self.get_ee_cdf_data(q_rand, self.p_goal_task)
            q = q_rand - d_ee * g_ee
            q = self.clamp_to_bounds(q).astype(np.float32)

            cdf_dist = self.get_cdf_distance(q)
            if cdf_dist < self.safety_margin_c_space:
                d_obs, g_obs = self.get_cdf_data(q)
                q = q + (self.safety_margin_c_space - d_obs) * g_obs
                q = self.clamp_to_bounds(q).astype(np.float32)

            return q, {
                "mode": "ee_goal_project",
                "raw_sample": q_rand.copy(),
                "used_sample": q.copy(),
                "projected": True,
                "ee_cdf_distance": float(d_ee),
            }

        return super().sample_target_with_info(goal, rng)

    # ------------------------------------------------------------------
    # Plan (stores task-space goal, passes dummy joint-space goal)
    # ------------------------------------------------------------------

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iters: int,
        seed: int,
        callback: Optional[StepCallback] = None,
    ) -> Tuple[List[Node], Optional[np.ndarray], dict]:
        self.p_goal_task = np.asarray(goal, dtype=np.float32)
        dummy_goal = np.asarray(start, dtype=np.float32).copy()
        return super(CDF_RRTStar, self).plan(
            start, dummy_goal, max_iters, seed, callback
        )


def _matplotlib_backend_is_interactive() -> bool:
    """True if the GUI can be shown/updated (not file-only or notebook inline)."""
    name = plt.get_backend().lower()
    if name in ("agg", "svg", "pdf", "ps", "pgf", "template"):
        return False
    if "inline" in name:
        return False
    return True


class LiveTreeVisualizer:
    def __init__(
        self,
        cdf: CDF2D,
        obj_list,
        start: np.ndarray,
        goal: np.ndarray,
        planner_name: str,
        update_every: int = 1,
        pause_sec: float = 0.001,
        p_goal_task_space: Optional[np.ndarray] = None,
        ik_solutions: Optional[np.ndarray] = None,
    ) -> None:
        self.update_every = max(1, int(update_every))
        self.pause_sec = max(0.0, float(pause_sec))
        self.title_prefix = planner_name
        self.backend = plt.get_backend().lower()
        self.interactive_backend = _matplotlib_backend_is_interactive()
        self.edge_artists: Dict[int, Any] = {}

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        cdf.plot_cdf(self.ax, obj_list)

        self.ax.plot(start[0], start[1], "go", markersize=8, label="Start")
        self.ax.plot(goal[0], goal[1], "bo", markersize=8, label="Goal")
        self.sample_artist, = self.ax.plot(
            [], [], marker="x", markersize=8, linestyle="None", color="tab:red", label="Current sample"
        )
        self.raw_sample_artist, = self.ax.plot(
            [],
            [],
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="darkorange",
            linestyle="None",
            label="Raw sample",
        )
        self.raw_link_artist, = self.ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=1.0,
            color="darkorange",
            alpha=0.8,
            label="Projection",
        )
        self.ax.legend(loc="upper right", fontsize=9)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        if self.interactive_backend:
            self.fig.canvas.flush_events()

        self.fig_goal: Optional[plt.Figure] = None
        if p_goal_task_space is not None:
            self.fig_goal, ax_goal = plt.subplots(figsize=(9, 8))
            cdf.plot_ee_goal_cdf(ax_goal, p_goal_task_space, ik_solutions=ik_solutions)
            ax_goal.plot(start[0], start[1], "go", markersize=8, label="Start")
            ax_goal.set_title(f"{planner_name}: EE Goal CDF")
            ax_goal.legend(loc="upper right", fontsize=9)
            self.fig_goal.tight_layout()
            self.fig_goal.canvas.draw_idle()
            if self.interactive_backend:
                self.fig_goal.canvas.flush_events()

    def _maybe_draw_raw_projection(self, q_target: np.ndarray, sampling: Dict[str, Any]) -> None:
        raw_sample = sampling.get("raw_sample")
        if raw_sample is None:
            self.raw_sample_artist.set_data([], [])
            self.raw_link_artist.set_data([], [])
            return
        raw_sample = np.asarray(raw_sample, dtype=np.float32)
        if np.linalg.norm(raw_sample - q_target) < 1e-7:
            self.raw_sample_artist.set_data([], [])
            self.raw_link_artist.set_data([], [])
            return
        self.raw_sample_artist.set_data([raw_sample[0]], [raw_sample[1]])
        self.raw_link_artist.set_data([raw_sample[0], q_target[0]], [raw_sample[1], q_target[1]])

    def on_step(self, event: Dict[str, Any]) -> None:
        should_draw = (event["iteration"] % self.update_every) == 0
        if not should_draw and event.get("goal_event") is None:
            return

        if event["accepted"]:
            new_idx = int(event["new_node_idx"])
            parent_q = np.asarray(event["new_parent_q"], dtype=np.float32)
            node_q = np.asarray(event["new_node_q"], dtype=np.float32)
            edge, = self.ax.plot(
                [parent_q[0], node_q[0]],
                [parent_q[1], node_q[1]],
                color="dimgray",
                linewidth=0.7,
                alpha=0.5,
            )
            self.edge_artists[new_idx] = edge

            for rw in event["rewired_nodes"]:
                node_idx = int(rw["node_idx"])
                node_q = np.asarray(rw["node_q"], dtype=np.float32)
                parent_q = np.asarray(rw["new_parent_q"], dtype=np.float32)
                edge = self.edge_artists.get(node_idx)
                if edge is None:
                    edge, = self.ax.plot([], [], color="tab:orange", linewidth=1.0, alpha=0.85)
                    self.edge_artists[node_idx] = edge
                edge.set_data([parent_q[0], node_q[0]], [parent_q[1], node_q[1]])
                edge.set_color("tab:orange")
                edge.set_linewidth(1.0)
                edge.set_alpha(0.85)

        goal_event = event.get("goal_event")
        if goal_event is not None:
            goal_idx = int(goal_event["goal_idx"])
            parent_q = np.asarray(goal_event["parent_q"], dtype=np.float32)
            goal_q = np.asarray(goal_event["goal_q"], dtype=np.float32)
            edge = self.edge_artists.get(goal_idx)
            if edge is None:
                edge, = self.ax.plot([], [], color="tab:green", linewidth=1.1, alpha=0.9)
                self.edge_artists[goal_idx] = edge
            edge.set_data([parent_q[0], goal_q[0]], [parent_q[1], goal_q[1]])
            edge.set_color("tab:green")
            edge.set_linewidth(1.1)
            edge.set_alpha(0.9)

        q_target = np.asarray(event["q_target"], dtype=np.float32)
        sample_color = "tab:green" if event["accepted"] else "tab:red"
        self.sample_artist.set_data([q_target[0]], [q_target[1]])
        self.sample_artist.set_color(sample_color)
        self._maybe_draw_raw_projection(q_target, event["sampling"])

        self.ax.set_title(
            f"{self.title_prefix} | iter {event['iteration']}/{event['max_iters']} | "
            f"nodes={event['tree_size']} | rejected={event['discarded_samples']} | "
            f"rewires={event['total_rewires']}"
        )
        self.fig.canvas.draw_idle()
        if self.interactive_backend:
            plt.pause(self.pause_sec)


def plot_tree(
    ax,
    cdf: CDF2D,
    obj_list,
    nodes: List[Node],
    path: Optional[np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
) -> None:
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


def print_stats(planner_name: str, stats: dict) -> None:
    print("\n=== RRT* Planning Statistics ===")
    print(f"planner:                 {planner_name}")
    print(f"success:                 {stats['success']}")
    print(f"time_to_first_path_sec:  {stats['time_to_first_path_sec']}")
    print(f"nodes_to_first_path:     {stats['nodes_to_first_path']}")
    print(f"final_path_cost:         {stats['final_path_cost']}")
    print(f"accepted_nodes:          {stats['accepted_nodes']}")
    print(f"discarded_samples:       {stats['discarded_samples']}")
    print(f"rejection_rate:          {stats['rejection_rate']:.4f}")
    print(f"rewires:                 {stats['rewires']}")
    print(f"first_goal_iteration:    {stats['first_goal_iteration']}")
    print(f"planning_time_sec:       {stats['planning_time_sec']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Vanilla or CDF-guided RRT* with optional live tree/sampling visualization."
    )
    parser.add_argument("--planner", choices=["vanilla", "cdf"], default="cdf")
    parser.add_argument("--scene", choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"], default="scene_1")
    parser.add_argument("--start", nargs=2, type=float, metavar=("Q1", "Q2"))
    parser.add_argument("--goal", nargs=2, type=float, metavar=("Q1", "Q2"))

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=2500)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--goal-threshold", type=float, default=0.25)
    parser.add_argument("--goal-bias", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=float, default=0.5)
    parser.add_argument("--edge-resolution", type=float, default=0.05)

    parser.add_argument("--safety-margin-task-space", type=float, default=SAFETY_MARGIN_TASK_SPACE)
    parser.add_argument("--safety-margin-c-space", type=float, default=SAFETY_MARGIN_C_SPACE)
    parser.add_argument("--softmin-beta", type=float, default=SOFTMIN_BETA)
    parser.add_argument("--model-path", type=str, default=None)

    parser.add_argument("--ee-goal", nargs=2, type=float, metavar=("X", "Y"),
                        help="Task-space EE goal (uses CDF_RRTStar_EE; overrides --goal).")
    parser.add_argument("--ee-model-path", type=str, default=None,
                        help="Path to trained EE CDF model (model_ee.pth).")
    parser.add_argument("--ee-goal-threshold", type=float, default=0.15,
                        help="EE distance threshold for task-space goal reaching.")

    parser.add_argument("--live-viz", action="store_true")
    parser.add_argument("--viz-update-every", type=int, default=1)
    parser.add_argument("--viz-pause-sec", type=float, default=0.001)
    return parser.parse_args()


def _resolve_start_goal(args) -> Tuple[np.ndarray, np.ndarray]:
    default_start, default_goal = SCENE_START_GOAL[args.scene]
    if args.start is None:
        start = default_start.copy()
    else:
        start = np.asarray(args.start, dtype=np.float32)
    if args.goal is None:
        goal = default_goal.copy()
    else:
        goal = np.asarray(args.goal, dtype=np.float32)
    return start, goal


def _validate_query_points(planner: RRTStarBase, start: np.ndarray, goal: np.ndarray) -> None:
    q_min = planner.q_min
    q_max = planner.q_max
    if not planner.in_bounds(start):
        raise SystemExit(
            f"Error: Start is out of bounds. Expected each joint in [{q_min[0]:.3f}, {q_max[0]:.3f}]."
        )
    if not planner.in_bounds(goal):
        raise SystemExit(
            f"Error: Goal is out of bounds. Expected each joint in [{q_min[0]:.3f}, {q_max[0]:.3f}]."
        )
    if not planner.is_state_collision_free(start):
        raise SystemExit("Error: Start configuration is in collision.")
    if not planner.is_state_collision_free(goal):
        raise SystemExit("Error: Goal configuration is in collision.")


def _pick_ik_goal(
    cdf_obj: CDF2D,
    obj_list,
    p_goal: np.ndarray,
    start: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find IK solutions for *p_goal*, return (best_q, all_valid_qs)."""
    p_t = torch.tensor(p_goal, dtype=torch.float32, device=device)
    _, q_solutions, _ = cdf_obj.find_q_ee(p_t)
    q_np = q_solutions.detach().cpu().numpy().astype(np.float32)

    valid = []
    for q in q_np:
        q_t = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
        sdf = cdf_obj.inference_sdf(q_t, obj_list).item()
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


def main():
    args = parse_args()
    if args.viz_update_every < 1:
        raise SystemExit("Error: --viz-update-every must be >= 1.")
    if args.viz_pause_sec < 0.0:
        raise SystemExit("Error: --viz-pause-sec must be >= 0.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    obj_list = make_scene(args.scene, device)
    q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
    q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)
    start, _ = _resolve_start_goal(args)

    ee_mode = args.ee_goal is not None
    p_goal_task: Optional[np.ndarray] = None
    ik_solutions: Optional[np.ndarray] = None

    if ee_mode:
        p_goal_task = np.asarray(args.ee_goal, dtype=np.float32)

    planner_kwargs = dict(
        step_size=args.step_size,
        goal_threshold=args.goal_threshold,
        goal_bias=args.goal_bias,
        neighbor_radius=args.neighbor_radius,
        edge_resolution=args.edge_resolution,
    )

    if args.planner == "vanilla":
        if ee_mode:
            q_ik_goal, ik_solutions = _pick_ik_goal(cdf, obj_list, p_goal_task, start, device)
            goal = q_ik_goal
            print(f"IK for vanilla: {len(ik_solutions)} collision-free solutions, "
                  f"best q = [{q_ik_goal[0]:.3f}, {q_ik_goal[1]:.3f}]")
        else:
            _, goal = _resolve_start_goal(args)

        planner_name = "Vanilla_RRTStar"
        planner: RRTStarBase = Vanilla_RRTStar(
            cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
            **planner_kwargs,
        )
        _validate_query_points(planner, start, goal)
    else:
        if ee_mode:
            planner_name = "CDF_RRTStar_EE"
            goal = p_goal_task
            planner = CDF_RRTStar_EE(
                cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
                safety_margin_task_space=args.safety_margin_task_space,
                safety_margin_c_space=args.safety_margin_c_space,
                softmin_beta=args.softmin_beta,
                model_path=args.model_path,
                ee_model_path=args.ee_model_path,
                ee_goal_threshold=args.ee_goal_threshold,
                **planner_kwargs,
            )
        else:
            _, goal = _resolve_start_goal(args)
            planner_name = "CDF_RRTStar"
            planner = CDF_RRTStar(
                cdf=cdf, obj_list=obj_list, q_min=q_min, q_max=q_max,
                safety_margin_task_space=args.safety_margin_task_space,
                safety_margin_c_space=args.safety_margin_c_space,
                softmin_beta=args.softmin_beta,
                model_path=args.model_path,
                **planner_kwargs,
            )
            _validate_query_points(planner, start, goal)

    callback = None
    if args.live_viz:
        viz_goal = goal if not ee_mode else start.copy()
        viz = LiveTreeVisualizer(
            cdf=cdf,
            obj_list=obj_list,
            start=start,
            goal=viz_goal,
            planner_name=planner_name,
            update_every=args.viz_update_every,
            pause_sec=args.viz_pause_sec,
            p_goal_task_space=p_goal_task,
            ik_solutions=ik_solutions,
        )
        callback = viz.on_step

    print("=== Live RRT* Runner ===")
    print(f"planner: {planner_name}")
    print(f"scene: {args.scene}")
    print(f"seed: {args.seed}, max_iters: {args.max_iters}")
    if ee_mode:
        print(f"start: [{start[0]:.3f}, {start[1]:.3f}] | "
              f"ee_goal: [{p_goal_task[0]:.3f}, {p_goal_task[1]:.3f}] | "
              f"live_viz: {args.live_viz}")
    else:
        print(f"start: [{start[0]:.3f}, {start[1]:.3f}] | "
              f"goal: [{goal[0]:.3f}, {goal[1]:.3f}] | "
              f"live_viz: {args.live_viz}")

    nodes, path, stats = planner.plan(
        start=start,
        goal=goal,
        max_iters=args.max_iters,
        seed=args.seed,
        callback=callback,
    )

    print_stats(planner_name, stats)

    fig, ax = plt.subplots(figsize=(9, 8))
    plot_tree(
        ax=ax,
        cdf=cdf,
        obj_list=obj_list,
        nodes=nodes,
        path=path,
        start=start,
        goal=goal if not ee_mode else start.copy(),
        title=f"{args.scene}: {planner_name} (Final)",
    )

    if ee_mode:
        fig2, ax2 = plt.subplots(figsize=(9, 8))
        cdf.plot_ee_goal_cdf(ax2, p_goal_task, ik_solutions=ik_solutions)
        ax2.plot(start[0], start[1], "go", markersize=8, label="Start")
        ax2.set_title(f"{args.scene}: {planner_name} - EE Goal CDF")
        ax2.legend(loc="upper right", fontsize=9)
        fig2.tight_layout()

    plt.tight_layout()
    plt.ioff()
    if _matplotlib_backend_is_interactive():
        plt.show()
    else:
        plt.show(block=False)
        plt.close("all")


if __name__ == "__main__":
    main()
