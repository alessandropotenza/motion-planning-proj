#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# -----------------------------------------------------------------------------

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from cdf import CDF2D
from mlp import MLPRegression  # Needed for torch.load() of model.pth


SAFETY_MARGIN = 1.0
SOFTMIN_BETA = 50.0

CUR_PATH = os.path.dirname(os.path.realpath(__file__))


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

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iters: int,
        seed: int,
    ) -> Tuple[List[Node], Optional[np.ndarray], dict]:
        rng = np.random.default_rng(seed)
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)

        if not self.in_bounds(start):
            raise ValueError("Start is out of bounds.")
        if not self.in_bounds(goal):
            raise ValueError("Goal is out of bounds.")
        if not self.is_state_collision_free(start):
            raise ValueError("Start is in collision.")
        if not self.is_state_collision_free(goal):
            raise ValueError("Goal is in collision.")

        nodes = [Node(q=start.copy(), parent=None, cost=0.0)]
        goal_idx = None
        first_goal_iteration = None
        time_to_first_path_sec = None
        nodes_to_first_path = None

        rejected_samples = 0
        rewires = 0
        t0 = time.time()

        for it in range(max_iters):
            q_target = self.sample_target(goal, rng)
            nearest_idx = self.nearest_index(nodes, q_target, goal_idx)

            q_new, edge_cost, collision_free, _ = self.rollout_edge(
                nodes[nearest_idx].q, q_target, max_extension=self.step_size
            )
            if (
                (not collision_free)
                or (not self.in_bounds(q_new))
                or euclidean(q_new, nodes[nearest_idx].q) < 1e-8
            ):
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
                _, conn_cost, ok, reached = self.rollout_edge(nodes[idx].q, q_new, max_extension=None)
                if not ok or not reached:
                    continue
                candidate_cost = nodes[idx].cost + conn_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = idx

            new_idx = len(nodes)
            nodes.append(Node(q=q_new, parent=best_parent, cost=best_cost))

            for idx in near_idxs:
                if idx == best_parent or (goal_idx is not None and idx == goal_idx):
                    continue
                _, rewire_cost, ok, reached = self.rollout_edge(nodes[new_idx].q, nodes[idx].q, max_extension=None)
                if not ok or not reached:
                    continue
                new_cost = nodes[new_idx].cost + rewire_cost
                if new_cost < nodes[idx].cost:
                    nodes[idx].parent = new_idx
                    nodes[idx].cost = new_cost
                    rewires += 1
                    self.update_descendant_costs(nodes, idx)

            if euclidean(q_new, goal) <= self.goal_threshold:
                _, goal_conn_cost, ok, reached = self.rollout_edge(nodes[new_idx].q, goal, max_extension=None)
                if ok and reached:
                    goal_cost = nodes[new_idx].cost + goal_conn_cost
                    if goal_idx is None:
                        goal_idx = len(nodes)
                        nodes.append(Node(q=goal.copy(), parent=new_idx, cost=goal_cost))
                        first_goal_iteration = it + 1
                        nodes_to_first_path = len(nodes)
                        time_to_first_path_sec = time.time() - t0
                    elif goal_cost < nodes[goal_idx].cost:
                        nodes[goal_idx].parent = new_idx
                        nodes[goal_idx].cost = goal_cost
                        rewires += 1
                        self.update_descendant_costs(nodes, goal_idx)

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
        safety_margin: float = SAFETY_MARGIN,
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
        self.safety_margin = float(safety_margin)
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
        if rng.random() < self.goal_bias:
            return goal.copy()

        q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)

        sdf_val = self._quick_sdf(q_rand)
        if sdf_val > self.safety_margin:
            return q_rand

        dist, grad = self.get_cdf_data(q_rand)
        q_proj = q_rand + (self.safety_margin - dist) * grad
        q_proj = self.clamp_to_bounds(q_proj)
        return q_proj.astype(np.float32)
