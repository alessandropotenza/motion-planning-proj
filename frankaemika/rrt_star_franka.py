# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""7-DOF RRT* in joint space for the Franka Panda (vanilla + shared base class).

Collision checking is injected via ``SphereArmCollisionChecker`` (analytic FK +
obstacle SDFs). No PyBullet dependency in this module.

The planning loop mirrors ``2Dexamples/cdf_guided_rrtstar.RRTStarBase`` API and
statistics keys used by ``eval_cdf_rrtstar.print_stats``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class CollisionChecker(Protocol):
    q_min: np.ndarray
    q_max: np.ndarray

    def is_state_free(self, q: np.ndarray) -> bool:
        ...

    def is_edge_free(self, q_from: np.ndarray, q_to: np.ndarray, edge_resolution: float) -> bool:
        ...


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


def path_to_tree_node_nearest_goal(
    nodes: List[Node], goal: np.ndarray
) -> Tuple[np.ndarray, int, float]:
    """Tree path from the root to the node whose ``q`` is closest to ``goal`` (L2 in joint space).

    Useful for visualizing exploration progress when no goal connection was found.
    """
    goal = np.asarray(goal, dtype=np.float32).reshape(7)
    if not nodes:
        raise ValueError("nodes must be non-empty")
    best_idx = int(np.argmin([euclidean(n.q, goal) for n in nodes]))
    best_dist = euclidean(nodes[best_idx].q, goal)
    path: List[np.ndarray] = []
    idx: Optional[int] = best_idx
    while idx is not None:
        path.append(nodes[idx].q)
        idx = nodes[idx].parent
    path.reverse()
    return np.asarray(path, dtype=np.float32), best_idx, best_dist


StepCallback = Callable[[Dict[str, Any]], None]


class RRTStarFrankaBase:
    def __init__(
        self,
        checker: CollisionChecker,
        step_size: float = 0.15,
        goal_threshold: float = 0.25,
        goal_bias: float = 0.05,
        neighbor_radius: float = 0.55,
        edge_resolution: float = 0.08,
    ) -> None:
        self.checker = checker
        self.q_min = np.asarray(checker.q_min, dtype=np.float32)
        self.q_max = np.asarray(checker.q_max, dtype=np.float32)
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
        return self.checker.is_state_free(np.asarray(q, dtype=np.float64))

    def is_edge_collision_free(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        return self.checker.is_edge_free(
            np.asarray(q_from, dtype=np.float64),
            np.asarray(q_to, dtype=np.float64),
            self.edge_resolution,
        )

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
        return np.asarray(path, dtype=np.float32)

    def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q_target, _ = self.sample_target_with_info(goal, rng)
        return q_target

    def sample_target_with_info(
        self, goal: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if rng.random() < self.goal_bias:
            g = goal.copy().astype(np.float32)
            return g, {
                "mode": "goal_bias",
                "raw_sample": g.copy(),
                "used_sample": g.copy(),
                "projected": False,
            }
        q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
        return q_rand, {
            "mode": "direct",
            "raw_sample": q_rand.copy(),
            "used_sample": q_rand.copy(),
            "projected": False,
        }

    def rollout_edge(
        self,
        q_from: np.ndarray,
        q_target: np.ndarray,
        max_extension: Optional[float],
    ) -> Tuple[np.ndarray, float, bool, bool]:
        direction = q_target - q_from
        dist = float(np.linalg.norm(direction))
        if dist < 1e-10:
            return q_from.copy(), 0.0, True, True
        travel = dist if max_extension is None else min(dist, max_extension)
        q_end = q_from + direction / dist * travel
        q_end = self.clamp_to_bounds(q_end)
        collision_free = self.is_edge_collision_free(q_from, q_end)
        reached_target = abs(travel - dist) <= 1e-8
        return q_end.astype(np.float32), float(travel), collision_free, reached_target

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
        _, goal_conn_cost, ok, reached = self.rollout_edge(nodes[new_idx].q, goal, max_extension=None)
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
        log_every_iters: Optional[int] = None,
        log_prefix: str = "",
    ) -> Tuple[List[Node], Optional[np.ndarray], dict]:
        rng = np.random.default_rng(seed)
        start = np.asarray(start, dtype=np.float32).reshape(7)
        goal = np.asarray(goal, dtype=np.float32).reshape(7)

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

        def _progress_log(it_done: int) -> None:
            if not log_every_iters or log_every_iters <= 0:
                return
            if (it_done + 1) % log_every_iters != 0:
                return
            elapsed = time.time() - t0
            print(
                f"{log_prefix}iter {it_done + 1}/{max_iters}  "
                f"tree_nodes={len(nodes)}  rejected={rejected_samples}  "
                f"rewires={rewires}  goal={'yes' if goal_idx is not None else 'no'}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        log_every_iters = 10

        for it in range(max_iters):
            if it % log_every_iters == 0:
                print(f"iter {it + 1}/{max_iters}  tree_nodes={len(nodes)}  rejected={rejected_samples}  rewires={rewires}", flush=True)
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
                _progress_log(it)
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
                    goal_idx,
                    rewires,
                    _fgi,
                    _nfp,
                    _ttfp,
                    goal_event,
                ) = self._make_goal_connection(new_idx, nodes, goal, goal_idx, rewires, it, t0)
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
            _progress_log(it)

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


class VanillaFrankaRRTStar(RRTStarFrankaBase):
    """Uniform joint-space sampling + straight-line edges (standard RRT*)."""
    pass


def path_waypoint_cost(path: Optional[np.ndarray]) -> Optional[float]:
    if path is None or len(path) < 2:
        return None
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))


def print_stats_franka(scene_name: str, planner_name: str, stats: dict) -> None:
    """Same fields as ``2Dexamples/eval_cdf_rrtstar.print_stats``."""
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
