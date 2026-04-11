# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""CDF-guided RRT* for the Franka Panda (7-DOF), mirroring the 2D logic.

Uses the pretrained joint-space CDF MLP (``model_dict.pt``) with soft-min fusion
over workspace oracle points sampled on obstacle surfaces, plus a one-step
projection sampler and optional safety-shell endpoint projection on edges.

Collision / clearance uses ``SphereArmCollisionChecker.workspace_margin`` (analytic),
not PyBullet.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mlp import MLPRegression
from rrt_star_franka import RRTStarFrankaBase, safe_normalize
from workspace_obstacles import Obstacle

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

SAFETY_MARGIN_TASK_SPACE = 0.05
SAFETY_MARGIN_C_SPACE = 0.12
SOFTMIN_BETA = 40.0


class WorkspaceMarginChecker:
    """Protocol subset used by CDF planner."""

    def workspace_margin(self, q: np.ndarray) -> float:
        ...


class CDFGuidedFrankaRRTStar(RRTStarFrankaBase):
    def __init__(
        self,
        checker: Any,
        obstacles: List[Obstacle],
        device: torch.device,
        step_size: float = 0.15,
        goal_threshold: float = 0.25,
        goal_bias: float = 0.10,
        neighbor_radius: float = 0.55,
        edge_resolution: float = 0.08,
        safety_margin_task_space: float = SAFETY_MARGIN_TASK_SPACE,
        safety_margin_c_space: float = SAFETY_MARGIN_C_SPACE,
        softmin_beta: float = SOFTMIN_BETA,
        model_dict_path: Optional[str] = None,
        model_checkpoint_iter: int = 49900,
        oracle_points_per_primitive: int = 48,
    ) -> None:
        super().__init__(
            checker=checker,
            step_size=step_size,
            goal_threshold=goal_threshold,
            goal_bias=goal_bias,
            neighbor_radius=neighbor_radius,
            edge_resolution=edge_resolution,
        )
        if not hasattr(checker, "workspace_margin"):
            raise TypeError("checker must implement workspace_margin(q) for CDF-guided mode")
        self._margin_checker: WorkspaceMarginChecker = checker  # type: ignore[assignment]
        self.obstacles = obstacles
        self.device = device
        self.safety_margin_task_space = float(safety_margin_task_space)
        self.safety_margin_c_space = float(safety_margin_c_space)
        self.softmin_beta = float(softmin_beta)

        path = model_dict_path or os.path.join(CUR_DIR, "model_dict.pt")
        raw = torch.load(path, map_location=device, weights_only=False)
        state = raw[model_checkpoint_iter] if isinstance(raw, dict) else raw
        self.net = MLPRegression(
            input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128], skips=[], act_fn=torch.nn.ReLU, nerf=True
        )
        self.net.load_state_dict(state)
        self.net.to(device)
        self.net.eval()

        rng = np.random.default_rng(0)
        pts_list = []
        for o in obstacles:
            pts_list.append(o.sample_surface(oracle_points_per_primitive, rng))
        oracle = np.concatenate(pts_list, axis=0)
        self.oracle_points = torch.tensor(oracle, dtype=torch.float32, device=device)

    def _softmin(self, d: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(-self.softmin_beta * d, dim=0) / self.softmin_beta

    def _quick_margin(self, q: np.ndarray) -> float:
        return float(self._margin_checker.workspace_margin(q))

    def _grad_margin_fd(self, q: np.ndarray, eps: float = 2e-3) -> np.ndarray:
        m0 = self._quick_margin(q)
        g = np.zeros(7, dtype=np.float64)
        for i in range(7):
            dq = np.zeros(7, dtype=np.float64)
            dq[i] = eps
            g[i] = (self._quick_margin(q + dq) - m0) / eps
        return safe_normalize(g.astype(np.float32))

    def get_cdf_distance(self, q: np.ndarray) -> float:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7)
        pts = self.oracle_points
        q_rep = q_t.repeat(pts.shape[0], 1)
        inp = torch.cat([pts, q_rep], dim=1)
        with torch.no_grad():
            d_all = self.net.forward(inp).squeeze(-1)
            fused = self._softmin(d_all)
            margin = self._quick_margin(q)
            signed = float(np.sign(margin)) * float(fused.item())
        return signed

    def get_cdf_data(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device).reshape(1, 7).requires_grad_(True)
        pts = self.oracle_points
        q_rep = q_t.repeat(pts.shape[0], 1)
        inp = torch.cat([pts, q_rep], dim=1)
        d_all = self.net.forward(inp).squeeze(-1)
        fused = self._softmin(d_all)
        grad = torch.autograd.grad(fused, q_t, retain_graph=False, create_graph=False)[0].squeeze(0)
        grad_np = grad.detach().cpu().numpy().astype(np.float32)
        margin = self._quick_margin(q)
        g_m = self._grad_margin_fd(q)
        if float(np.dot(grad_np, g_m)) < 0.0:
            grad_np = -grad_np
        grad_np = safe_normalize(grad_np)
        signed = float(np.sign(margin)) * float(fused.detach().item())
        return signed, grad_np

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
        return q_proj.astype(np.float32), {
            "mode": "cdf_project",
            "raw_sample": q_rand.copy(),
            "used_sample": q_proj.copy(),
            "projected": True,
            "cdf_distance": float(dist),
        }

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

        cdf_dist, grad = self.get_cdf_data(q_end)
        if cdf_dist >= self.safety_margin_c_space:
            return q_end, travel, collision_free, reached_target

        q_projected = q_end + (self.safety_margin_c_space - cdf_dist) * grad
        q_projected = self.clamp_to_bounds(q_projected).astype(np.float32)
        collision_free_proj = self.is_edge_collision_free(q_from, q_projected)
        if not collision_free_proj:
            return q_projected, float(np.linalg.norm(q_projected - q_from)), False, False
        new_travel = float(np.linalg.norm(q_projected - q_from))
        reached_proj = np.linalg.norm(q_projected - q_target) <= 1e-6
        return q_projected, new_travel, True, reached_proj
