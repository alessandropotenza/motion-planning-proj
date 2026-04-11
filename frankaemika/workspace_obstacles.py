# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""Task-space obstacles for Franka demos: analytic SDFs (no PyBullet).

Obstacles are used only by the collision checker and CDF oracle sampling.
PyBullet should mirror the same parameters for visualization only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch


def sdf_sphere_np(p: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance to sphere boundary (positive outside). p: (N,3)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    return np.linalg.norm(p - c, axis=1) - float(radius)


def sdf_box_aa_np(p: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Axis-aligned box SDF (positive outside). p: (N,3)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    h = np.asarray(half_extents, dtype=np.float64).reshape(1, 3)
    q = np.abs(p - c) - h
    o = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    i = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
    return o + i


@dataclass
class SphereObstacle:
    center: np.ndarray  # (3,)
    radius: float

    def sdf_np(self, p: np.ndarray) -> np.ndarray:
        return sdf_sphere_np(p, self.center, self.radius)

    def sdf_torch(self, p: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(self.center, dtype=p.dtype, device=p.device).view(1, 3)
        return torch.norm(p - c, dim=1) - float(self.radius)

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Approximately uniform points on sphere."""
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        return self.center.reshape(1, 3) + self.radius * u


@dataclass
class BoxObstacle:
    center: np.ndarray  # (3,)
    half_extents: np.ndarray  # (3,)

    def sdf_np(self, p: np.ndarray) -> np.ndarray:
        return sdf_box_aa_np(p, self.center, self.half_extents)

    def sdf_torch(self, p: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(self.center, dtype=p.dtype, device=p.device).view(1, 3)
        h = torch.as_tensor(self.half_extents, dtype=p.dtype, device=p.device).view(1, 3)
        q = torch.abs(p - c) - h
        o = torch.norm(torch.clamp(q, min=0.0), dim=1)
        i = torch.minimum(torch.maximum(q[:, 0], torch.maximum(q[:, 1], q[:, 2])), torch.zeros_like(q[:, 0]))
        return o + i

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Stratified samples on the six faces."""
        c = np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.half_extents, dtype=np.float64)
        faces = []
        per_face = max(1, n // 6)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                pts = rng.uniform(-1, 1, size=(per_face, 3))
                pts[:, axis] = sign
                corner = c + h * pts
                faces.append(corner)
        pts = np.vstack(faces)
        if len(pts) < n:
            extra = self.sample_surface(n - len(pts) + per_face, rng)
            pts = np.vstack([pts, extra])
        idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
        return pts[idx]


Obstacle = Union[SphereObstacle, BoxObstacle]


def union_sdf_np(p: np.ndarray, obstacles: Sequence[Obstacle]) -> np.ndarray:
    """Minimum SDF (union of primitives)."""
    if not obstacles:
        return np.full(p.shape[0], 1e9, dtype=np.float64)
    d = obstacles[0].sdf_np(p)
    for o in obstacles[1:]:
        d = np.minimum(d, o.sdf_np(p))
    return d


def union_sdf_torch(p: torch.Tensor, obstacles: Sequence[Obstacle]) -> torch.Tensor:
    if not obstacles:
        return torch.full((p.shape[0],), 1e9, dtype=p.dtype, device=p.device)
    d = obstacles[0].sdf_torch(p)
    for o in obstacles[1:]:
        d = torch.minimum(d, o.sdf_torch(p))
    return d


def build_demo_obstacles(scene: str = "demo_table", rng: np.random.Generator | None = None) -> List[Obstacle]:
    """Named scenes: analytic obstacles in robot base frame (meters)."""
    rng = rng or np.random.default_rng(0)
    if scene == "demo_table":
        # Horizontal plate the arm must reach over / around
        return [
            BoxObstacle(
                center=np.array([0.45, 0.0, 0.35], dtype=np.float64),
                half_extents=np.array([0.12, 0.35, 0.04], dtype=np.float64),
            ),
            SphereObstacle(center=np.array([0.25, -0.35, 0.55], dtype=np.float64), radius=0.12),
            SphereObstacle(center=np.array([0.55, 0.35, 0.45], dtype=np.float64), radius=0.10),
        ]
    if scene == "sparse":
        return [
            SphereObstacle(center=np.array([0.5, 0.2, 0.4], dtype=np.float64), radius=0.15),
        ]
    if scene == "pillar_and_box":
        return [
            BoxObstacle(
                center=np.array([0.4, -0.2, 0.25]),
                half_extents=np.array([0.08, 0.25, 0.25]),
            ),
            SphereObstacle(center=np.array([0.35, 0.35, 0.5]), radius=0.11),
        ]
    raise ValueError(f"Unknown scene {scene}")
