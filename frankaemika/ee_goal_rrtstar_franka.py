#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# EE-goal planning for Franka Panda (7-DOF):
#   - Vanilla RRT* (C-space start + task-space goal via numerical IK)
#   - CDF-guided RRT* (CDFEERRTStar — mirrors 2D CDFEERRTStar)
#   - Pull-and-slide  (PullAndSlide — mirrors 2D PullAndSlide)
#
# Distance models loaded from model_dict_ee.pt and model_dict_wb.pt.
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
    from mlp import MLPRegression
    from nn_cdf import CDF
else:  # pragma: no cover
    MLPRegression = None  # type: ignore[assignment,misc]
    CDF = None  # type: ignore[assignment,misc]


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

SCENE_DEFAULTS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "demo_table": (
        np.array([2.0, -0.25, 0.0, -2.15, 0.0, 1.85, 0.785], dtype=np.float32),
        # np.array([0.55, 0.10, 0.48], dtype=np.float32),
        np.array([0.55, -0.35, 0.48], dtype=np.float32),
    ),
    "sparse": (
        np.array([2.0, -0.25, 0.0, -2.15, 0.0, 1.85, 0.785], dtype=np.float32),
        np.array([0.6, 0.0, 0.5], dtype=np.float32),
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
        np.array([0.0, 0, 1.0], dtype=np.float32),
    ),
    "cluttered_crossing": (
        np.array([0.205, -0.395, 0.173, -1.906, 0.041, 1.939, 1.097], dtype=np.float32),
        np.array([0.554, 0.307, 0.576], dtype=np.float32),
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _arr_str(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=3, floatmode="fixed")


def default_start_and_goal_task(scene: str) -> Tuple[np.ndarray, np.ndarray]:
    if scene in SCENE_DEFAULTS:
        s, g = SCENE_DEFAULTS[scene]
        return s.copy(), g.copy()
    return DEFAULT_START_Q.copy(), DEFAULT_GOAL_TASK.copy()


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def _ee_pos(q: np.ndarray) -> np.ndarray:
    return fk_flange_position(np.asarray(q, dtype=np.float64).reshape(7)).astype(np.float32)


def _ee_err_norm(q: np.ndarray, goal_task: np.ndarray) -> float:
    return float(np.linalg.norm(_ee_pos(q) - goal_task))


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


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------

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
    rng = np.random.default_rng(seed)
    q_min = checker.q_min.astype(np.float32)
    q_max = checker.q_max.astype(np.float32)
    start_q = np.asarray(start_q, dtype=np.float32).reshape(7)
    goal_task = np.asarray(goal_task, dtype=np.float32).reshape(3)

    def objective(q: np.ndarray) -> float:
        return _ee_err_norm(q, goal_task) + reg_to_start * float(np.linalg.norm(q - start_q))

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
    # return [True, True, True, True, True, True, True], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if not checker.is_state_free(start_q):
        raise SystemExit("Start configuration is invalid (out-of-bounds or in collision).")
    try:
        goal_q, ik_solutions = solve_task_goal_ik(
            checker, start_q, goal_task, seed=seed, n_restarts=26, max_steps=100,
        )
    except Exception as exc:
        raise SystemExit(
            "Task goal is not collision-feasible: no collision-free IK solution found. "
            "Adjust scene/defaults or pass --goal-task."
        ) from exc
    return goal_q, ik_solutions


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_model_dict(path: str, device, checkpoint_iter: int = 49900) -> "MLPRegression":
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and checkpoint_iter in raw:
        state = raw[checkpoint_iter]
    elif isinstance(raw, dict):
        last_key = max(k for k in raw if isinstance(k, int))
        state = raw[last_key]
    else:
        state = raw
    net = MLPRegression(
        input_dims=10, output_dims=1,
        mlp_layers=[1024, 512, 256, 128, 128],
        skips=[], act_fn=torch.nn.ReLU, nerf=True,
    )
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net



# ---------------------------------------------------------------------------
# CDFEERRTStar — mirrors 2D CDFEERRTStar
# ---------------------------------------------------------------------------

if torch is not None:
    class CDFEERRTStar(RRTStarFrankaBase):
        """CDF-guided RRT* for task-space EE goals (7-DOF).

        Uses obstacle CDF net (model_dict_wb.pt) and EE CDF net (model_dict_ee.pt).
        Distance / gradient queries go through ``cdf.inference_d_wrt_q(x, q, model)``.
        """

        def __init__(
            self,
            checker: SphereArmCollisionChecker,
            obstacles: List[Obstacle],
            device: "torch.device",
            step_size: float = 0.12,
            goal_threshold: float = 0.35,
            goal_bias: float = 0.10,
            neighbor_radius: float = 0.65,
            edge_resolution: float = 0.06,
            safety_margin_c_space: float = 0.12,
            ee_model_path: Optional[str] = None,
            wb_model_path: Optional[str] = None,
            ee_goal_threshold: float = 0.06,
            oracle_points_per_obj: int = 48,
            model_checkpoint_iter: int = 49900,
        ) -> None:
            super().__init__(
                checker=checker,
                step_size=step_size,
                goal_threshold=goal_threshold,
                goal_bias=goal_bias,
                neighbor_radius=neighbor_radius,
                edge_resolution=edge_resolution,
            )
            self.obstacles = obstacles
            self.device = device
            self.safety_margin_c_space = float(safety_margin_c_space)
            self.ee_goal_threshold = float(ee_goal_threshold)

            self.cdf = CDF(device)

            _ee_path = ee_model_path or os.path.join(CUR_DIR, "model_dict_ee.pt")
            _wb_path = wb_model_path or os.path.join(CUR_DIR, "model_dict_wb.pt")

            self.net_ee = _load_model_dict(_ee_path, device, model_checkpoint_iter)
            self.net_obs = _load_model_dict(_wb_path, device, model_checkpoint_iter)

            rng = np.random.default_rng(0)
            pts = [o.sample_surface(oracle_points_per_obj, rng) for o in obstacles]
            self.oracle_points = torch.tensor(
                np.concatenate(pts, axis=0), dtype=torch.float32, device=device,
            )

            self.goal_task: Optional[np.ndarray] = None

        def get_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
            """Whole-body CDF distance + gradient via cdf.inference_d_wrt_q."""
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
            d, grad = self.cdf.inference_d_wrt_q(self.oracle_points, q_t, self.net_obs)
            d_val = float(d.squeeze().detach().cpu().item())
            grad_np = grad.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return d_val, safe_normalize(grad_np)

        def get_cdf_distance(self, q: np.ndarray) -> float:
            """Scalar whole-body CDF distance (no gradient)."""
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
            d = self.cdf.inference_d_wrt_q(self.oracle_points, q_t, self.net_obs, return_grad=False)
            return float(d.squeeze().detach().cpu().item())

        def get_ee_cdf_data(self, q: np.ndarray, goal_task: np.ndarray) -> Tuple[float, np.ndarray]:
            """EE CDF distance + gradient via cdf.inference_d_wrt_q."""
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
            p_t = torch.tensor(goal_task, dtype=torch.float32, device=self.device).reshape(1, 3)
            d, grad = self.cdf.inference_d_wrt_q(p_t, q_t, self.net_ee)
            d_val = float(d.squeeze().detach().cpu().item())
            grad_np = grad.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return d_val, safe_normalize(grad_np)

        def _validate_goal(self, goal: np.ndarray) -> None:
            pass

        def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
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
            goal_cost = nodes[new_idx].cost
            first_goal_iteration = None
            nodes_to_first_path = None
            time_to_first_path_sec = None
            goal_event: Dict[str, Any] = {"type": "task_goal", "node_idx": new_idx}
            if goal_idx is None:
                goal_idx = new_idx
                first_goal_iteration = it + 1
                nodes_to_first_path = len(nodes)
                time_to_first_path_sec = time.time() - t0
            elif goal_cost < nodes[goal_idx].cost:
                goal_idx = new_idx
                rewires += 1
            return goal_idx, rewires, first_goal_iteration, nodes_to_first_path, time_to_first_path_sec, goal_event

        def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            if rng.random() < self.goal_bias:
                q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
                d_ee, g_ee = self.get_ee_cdf_data(q_rand, self.goal_task)
                q = q_rand - d_ee * g_ee
                return self.clamp_to_bounds(q)

            q_rand = rng.uniform(self.q_min, self.q_max).astype(np.float32)
            cdf_dist = self.get_cdf_distance(q_rand)
            if cdf_dist > self.safety_margin_c_space:
                return q_rand

            d_obs, g_obs = self.get_cdf_data(q_rand)
            q_proj = q_rand + (self.safety_margin_c_space - d_obs) * g_obs
            return self.clamp_to_bounds(q_proj)

        def sample_target_with_info(
            self, goal: np.ndarray, rng: np.random.Generator,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            q = self.sample_target(goal, rng)
            return q, {"mode": "cdf_ee"}

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
            q_projected = self.clamp_to_bounds(q_projected).astype(np.float32)
            collision_free_proj = self.is_edge_collision_free(q_from, q_projected)
            if not collision_free_proj:
                return q_projected, float(np.linalg.norm(q_projected - q_from)), False, False
            new_travel = float(np.linalg.norm(q_projected - q_from))
            reached_proj = np.linalg.norm(q_projected - q_target) <= 1e-6
            return q_projected, new_travel, True, reached_proj

        def solve(
            self,
            start_q: np.ndarray,
            goal_task: np.ndarray,
            max_iters: int,
            seed: int,
        ) -> dict:
            self.goal_task = np.asarray(goal_task, dtype=np.float32).reshape(3)
            start = np.asarray(start_q, dtype=np.float32).reshape(7)
            dummy_goal = start.copy()
            nodes, path, stats = self.plan(start, dummy_goal, max_iters=max_iters, seed=seed)
            config_goal_marker = path[-1].copy() if path is not None and len(path) > 0 else None
            return {
                "nodes": nodes,
                "path": path,
                "stats": stats,
                "config_goal_marker": config_goal_marker,
                "ik_solutions": None,
            }


    # -------------------------------------------------------------------
    # PullAndSlide — mirrors 2D PullAndSlide
    # -------------------------------------------------------------------

    class PullAndSlide(RRTStarFrankaBase):
        """Pull-and-slide sampler for task-space EE goals (7-DOF).

        Uses WB CDF net (model_dict_wb.pt) and EE CDF net (model_dict_ee.pt).
        Distance / gradient queries go through ``cdf.inference_d_wrt_q(x, q, model)``.
        Mirrors the 2D PullAndSlide API.
        """

        def __init__(
            self,
            checker: SphereArmCollisionChecker,
            obstacles: List[Obstacle],
            device: "torch.device",
            step_size: float = 0.12,
            goal_threshold: float = 0.35,
            goal_bias: float = 0.12,
            neighbor_radius: float = 0.65,
            edge_resolution: float = 0.06,
            ee_model_path: Optional[str] = None,
            wb_model_path: Optional[str] = None,
            ee_goal_threshold: float = 0.06,
            oracle_points_per_obj: int = 48,
            model_checkpoint_iter: int = 49900,
        ) -> None:
            super().__init__(
                checker=checker,
                step_size=step_size,
                goal_threshold=goal_threshold,
                goal_bias=goal_bias,
                neighbor_radius=neighbor_radius,
                edge_resolution=edge_resolution,
            )
            self.obstacles = obstacles
            self.device = device
            self.ee_goal_threshold = float(ee_goal_threshold)

            self.cdf = CDF(device)

            _ee_path = ee_model_path or os.path.join(CUR_DIR, "model_dict_ee.pt")
            _wb_path = wb_model_path or os.path.join(CUR_DIR, "model_dict_wb.pt")

            self.net_ee = _load_model_dict(_ee_path, device, model_checkpoint_iter)
            self.net_wb = _load_model_dict(_wb_path, device, model_checkpoint_iter)

            rng = np.random.default_rng(0)
            oracle_points_per_obj = 1000
            pts = [o.sample_surface(oracle_points_per_obj, rng) for o in obstacles]
            self.obstacle_points = torch.tensor(
                np.concatenate(pts, axis=0), dtype=torch.float32, device=device,
            )

            self.goal_task: Optional[np.ndarray] = None

        def get_ee_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
            """EE CDF distance + gradient via cdf.inference_d_wrt_q."""
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
            p_t = torch.tensor(self.goal_task, dtype=torch.float32, device=self.device).reshape(1, 3)
            d, grad = self.cdf.inference_d_wrt_q(p_t, q_t, self.net_ee)
            d_val = float(d.squeeze().detach().cpu().item())
            d_val = max(d_val, 0)
            grad_np = grad.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return d_val, safe_normalize(grad_np)

        def get_wb_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
            """Whole-body CDF distance + gradient via cdf.inference_d_wrt_q."""
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
            d, grad = self.cdf.inference_d_wrt_q(self.obstacle_points, q_t, self.net_wb)
            d_val = float(d.squeeze().detach().cpu().item())
            d_val = max(d_val, 0)
            grad_np = grad.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return d_val, safe_normalize(grad_np)

        def _validate_goal(self, goal: np.ndarray) -> None:
            pass

        def _is_near_goal(self, q_new: np.ndarray, goal: np.ndarray) -> bool:
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
            goal_cost = nodes[new_idx].cost
            first_goal_iteration = None
            nodes_to_first_path = None
            time_to_first_path_sec = None
            goal_event: Dict[str, Any] = {"type": "task_goal", "node_idx": new_idx}
            if goal_idx is None:
                goal_idx = new_idx
                first_goal_iteration = it + 1
                nodes_to_first_path = len(nodes)
                time_to_first_path_sec = time.time() - t0
            elif goal_cost < nodes[goal_idx].cost:
                goal_idx = new_idx
                rewires += 1
            return goal_idx, rewires, first_goal_iteration, nodes_to_first_path, time_to_first_path_sec, goal_event

        def sample_target(self, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            q_sample = rng.uniform(self.q_min, self.q_max).astype(np.float32)

            ee_d, ee_grad = self.get_ee_cdf_data(q_sample)

            if rng.random() < self.goal_bias or ee_d < self.goal_threshold:
            # if True:
                for _ in range(2):
                    q_sample = q_sample - ee_d * ee_grad
                    ee_d, ee_grad = self.get_ee_cdf_data(q_sample)
                q_sample = q_sample - ee_d * ee_grad
                return self.clamp_to_bounds(q_sample)

            wb_d, wb_grad = self.get_wb_cdf_data(q_sample)
            ee_d, ee_grad = self.get_ee_cdf_data(q_sample)

            collided = not self.is_state_collision_free(q_sample)

            # SAFETY_MARGIN = 0.06

            if collided:
                for _ in range(2):
                    q_sample = q_sample - (wb_d) * wb_grad
                    wb_d, wb_grad = self.get_wb_cdf_data(q_sample)
                q_sample = q_sample - (wb_d) * wb_grad
                q_sample = self.clamp_to_bounds(q_sample)

            if wb_d > self.step_size and not collided:
                pull_nudge = -self.step_size * ee_grad
                q_sample = q_sample + pull_nudge
            else:
                v_perp = ee_grad - float(np.dot(ee_grad, wb_grad)) * wb_grad
                n = float(np.linalg.norm(v_perp))
                if n > 1e-8:
                    new_grad = v_perp / n
                else:
                    new_grad = ee_grad
                nudge = -self.step_size * new_grad
                q_sample = q_sample + nudge

            return q_sample

        def sample_target_with_info(
            self, goal: np.ndarray, rng: np.random.Generator,
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            q = self.sample_target(goal, rng)
            return q, {"mode": "pull_and_slide"}

        def solve(
            self,
            start_q: np.ndarray,
            goal_task: np.ndarray,
            max_iters: int,
            seed: int,
        ) -> dict:
            self.goal_task = np.asarray(goal_task, dtype=np.float32).reshape(3)
            start = np.asarray(start_q, dtype=np.float32).reshape(7)
            dummy_goal = start.copy()
            nodes, path, stats = self.plan(start, dummy_goal, max_iters=max_iters, seed=seed)
            config_goal_marker = path[-1].copy() if path is not None and len(path) > 0 else None
            return {
                "nodes": nodes,
                "path": path,
                "stats": stats,
                "config_goal_marker": config_goal_marker,
                "ik_solutions": None,
            }

else:  # pragma: no cover — torch unavailable stubs
    class CDFEERRTStar(RRTStarFrankaBase):  # type: ignore[no-redef]
        pass

    class PullAndSlide(RRTStarFrankaBase):  # type: ignore[no-redef]
        pass

# Keep old names for backward compat with eval script imports
TaskGoalCDFFrankaRRTStar = CDFEERRTStar
PullAndSlideFrankaRRTStar = PullAndSlide


# ---------------------------------------------------------------------------
# Result / printing
# ---------------------------------------------------------------------------

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


def _best_effort_path(
    results: List[PlannerResult],
    goal_task: np.ndarray,
) -> Optional[PlannerResult]:
    """Find the planner result whose tree node is closest to goal_task in EE space.

    When no planner found a valid path, we scan every node in every tree and
    extract the path to the node whose end-effector is nearest the goal.
    Returns a PlannerResult with that path, or None if all trees are empty.
    """
    best_err = float("inf")
    best_result: Optional[PlannerResult] = None

    for r in results:
        nodes = r.stats.get("_nodes")
        if nodes is None or len(nodes) == 0:
            continue
        for idx, node in enumerate(nodes):
            err = _ee_err_norm(node.q, goal_task)
            if err < best_err:
                best_err = err
                best_node_idx = idx
                best_nodes = nodes

    if best_result is None and best_err < float("inf"):
        path_segs: List[np.ndarray] = []
        nid: Optional[int] = best_node_idx
        while nid is not None:
            path_segs.append(best_nodes[nid].q)
            nid = best_nodes[nid].parent
        path_segs.reverse()
        path = np.asarray(path_segs, dtype=np.float32)
        best_result = PlannerResult(
            planner_name="best_effort",
            path=path,
            stats={},
            task_goal_error=best_err,
        )
    return best_result


# ---------------------------------------------------------------------------
# PyBullet visualization
# ---------------------------------------------------------------------------

def _spawn_obstacle_visuals(p, obstacles: List[Obstacle]) -> List[int]:
    bodies: List[int] = []
    for o in obstacles:
        if isinstance(o, SphereObstacle):
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=float(o.radius),
                rgbaColor=[0.85, 0.35, 0.1, 1.0], specularColor=[0, 0, 0, 1],
            )
            bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=o.center.tolist())
            bodies.append(bid)
        elif isinstance(o, BoxObstacle):
            he = o.half_extents.astype(float).tolist()
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=he,
                rgbaColor=[0.2, 0.45, 0.9, 1.0], specularColor=[0, 0, 0, 1],
            )
            bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=o.center.tolist())
            bodies.append(bid)
    return bodies


def _spawn_task_trace(p, path_q: np.ndarray, color=(0.95, 0.1, 0.1), width: float = 2.5) -> None:
    if path_q is None or len(path_q) < 2:
        return
    ee = np.stack([_ee_pos(q) for q in path_q], axis=0)
    for i in range(len(ee) - 1):
        p.addUserDebugLine(
            lineFromXYZ=ee[i].tolist(), lineToXYZ=ee[i + 1].tolist(),
            lineColorRGB=list(color), lineWidth=float(width), lifeTime=0.0,
        )


def visualize_task_space_pybullet(
    path: np.ndarray,
    obstacles: List[Obstacle],
    goal_task: np.ndarray,
    hz: float = 20.0,
    hold_last_sec: float = 0.0,
    start_hold_sec: float = 2.0,
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
        p.GEOM_SPHERE, radius=0.035,
        rgbaColor=[0.08, 0.82, 0.18, 1.0], specularColor=[0.2, 0.2, 0.2, 1],
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

    if start_hold_sec > 0.0:
        robot.set_joint_positions(dense_q[0])
        t_pause_end = time.time() + float(start_hold_sec)
        while time.time() < t_pause_end and p.isConnected():
            p.stepSimulation()
            time.sleep(dt)

    _spawn_task_trace(p, dense_q)
    for qq in dense_q:
        robot.set_joint_positions(qq)
        p.stepSimulation()
        time.sleep(dt * 0.5)

    if hold_last_sec <= 0.0:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    ap.add_argument("--ee-goal-threshold", type=float, default=0.1)
    # ap.add_argument("--ee-goal-threshold", type=float, default=0.06)
    ap.add_argument("--demo", action="store_true", help="Animate best successful plan in PyBullet.")
    ap.add_argument("--demo-hold-sec", type=float, default=0.0,
                    help="Seconds to keep PyBullet open after playback. <= 0 means keep open.")
    ap.add_argument("--demo-start-hold-sec", type=float, default=2.0,
                    help="Seconds to hold at start configuration before playback (0 to skip).")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (default auto)")
    ap.add_argument("--log-every", type=int, default=0, help="Print planner progress every N iterations.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# main — uses solve() for all planners
# ---------------------------------------------------------------------------

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
        checker=checker, start_q=start_q, goal_task=goal_task, seed=args.seed,
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
            print(f"\n[vanilla] IK solutions: {len(ik_solutions)}")
            print(f"[vanilla] selected goal_q = {_arr_str(goal_q_feasible)}")
            planner_v = VanillaFrankaRRTStar(
                checker=checker, step_size=args.step_size, goal_threshold=args.goal_threshold,
                goal_bias=args.goal_bias, neighbor_radius=args.neighbor_radius,
                edge_resolution=args.edge_resolution,
            )
            log_kw = {"log_every_iters": args.log_every, "log_prefix": "[vanilla] "} if args.log_every > 0 else {}
            nodes_v, path, stats = planner_v.plan(start_q, goal_q_feasible, max_iters=args.max_iters, seed=args.seed, **log_kw)
            stats["_nodes"] = nodes_v
            task_err = _ee_err_norm(path[-1], goal_task) if path is not None and len(path) > 0 else float("nan")
            res_v = PlannerResult("vanilla", path, stats, task_err)
            print_stats("VanillaFrankaRRTStar (task-goal via IK)", res_v.stats, res_v.path, res_v.task_goal_error)
            results.append(res_v)
        except Exception as exc:
            print(f"\n[vanilla] failed: {exc}")

    if args.planner in ("cdf", "all"):
        cdf_planner = CDFEERRTStar(
            checker=checker, obstacles=obstacles, device=device,
            step_size=args.step_size, goal_threshold=args.goal_threshold,
            goal_bias=max(args.goal_bias, 0.10), neighbor_radius=args.neighbor_radius,
            edge_resolution=args.edge_resolution, ee_goal_threshold=args.ee_goal_threshold,
        )
        result_c = cdf_planner.solve(start_q=start_q, goal_task=goal_task, max_iters=args.max_iters, seed=args.seed)
        path_c = result_c["path"]
        stats_c = result_c["stats"]
        stats_c["_nodes"] = result_c["nodes"]
        task_err_c = _ee_err_norm(path_c[-1], goal_task) if path_c is not None and len(path_c) > 0 else float("nan")
        res_c = PlannerResult("cdf", path_c, stats_c, task_err_c)
        print_stats("CDFEERRTStar (task-goal)", res_c.stats, res_c.path, res_c.task_goal_error)
        results.append(res_c)

    if args.planner in ("pullandslide", "all"):
        pull_planner = PullAndSlide(
            checker=checker, obstacles=obstacles, device=device,
            step_size=args.step_size, goal_threshold=args.goal_threshold,
            goal_bias=max(args.goal_bias, 0.12), neighbor_radius=args.neighbor_radius,
            edge_resolution=args.edge_resolution, ee_goal_threshold=args.ee_goal_threshold,
        )
        result_p = pull_planner.solve(start_q=start_q, goal_task=goal_task, max_iters=args.max_iters, seed=args.seed)
        path_p = result_p["path"]
        stats_p = result_p["stats"]
        stats_p["_nodes"] = result_p["nodes"]
        task_err_p = _ee_err_norm(path_p[-1], goal_task) if path_p is not None and len(path_p) > 0 else float("nan")
        res_p = PlannerResult("pullandslide", path_p, stats_p, task_err_p)
        print_stats("PullAndSlide (task-goal)", res_p.stats, res_p.path, res_p.task_goal_error)
        results.append(res_p)

    success = [r for r in results if r.path is not None and bool(r.stats.get("success"))]
    if not success:
        best_effort = _best_effort_path(results, goal_task)
        if best_effort is not None and best_effort.path is not None and len(best_effort.path) > 1:
            print(
                f"\nNo planner found a valid path. Showing best-effort path "
                f"(EE error to goal: {best_effort.task_goal_error:.4f} m, "
                f"{len(best_effort.path)} waypoints)."
            )
            if args.demo:
                visualize_task_space_pybullet(
                    path=best_effort.path, obstacles=obstacles, goal_task=goal_task,
                    hold_last_sec=args.demo_hold_sec, start_hold_sec=args.demo_start_hold_sec,
                    label=f"best-effort | ee_err={best_effort.task_goal_error:.3f}m",
                )
        else:
            print("\nNo planner produced a valid path and no tree nodes reached near the goal.")
            if args.demo:
                visualize_task_space_pybullet(
                    path=start_q.reshape(1, 7), obstacles=obstacles, goal_task=goal_task,
                    hold_last_sec=args.demo_hold_sec, start_hold_sec=args.demo_start_hold_sec,
                    label="No feasible path (scene-only view)",
                )
        raise SystemExit("No planner produced a valid path. Try increasing --max-iters or relaxing scene/goal.")

    best = min(success, key=lambda r: float(r.stats.get("final_path_cost", float("inf"))))
    print(f"\nBest successful planner: {best.planner_name} (cost={best.stats.get('final_path_cost')})")

    if args.demo:
        print(f"Launching PyBullet task-space animation for planner: {best.planner_name}")
        visualize_task_space_pybullet(
            path=best.path, obstacles=obstacles, goal_task=goal_task,
            hold_last_sec=args.demo_hold_sec, start_hold_sec=args.demo_start_hold_sec,
            label=f"{best.planner_name} | task-goal",
        )


if __name__ == "__main__":
    main()
