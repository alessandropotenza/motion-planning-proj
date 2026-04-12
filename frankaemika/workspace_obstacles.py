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
    if scene == "cluttered_gate":
        # Broad clutter around the arm with a narrow but usable central passage.
        return [
            BoxObstacle(center=np.array([0.42, 0.00, 0.30]), half_extents=np.array([0.10, 0.30, 0.03])),
            BoxObstacle(center=np.array([0.52, -0.25, 0.42]), half_extents=np.array([0.045, 0.055, 0.14])),
            BoxObstacle(center=np.array([0.52, 0.25, 0.42]), half_extents=np.array([0.045, 0.055, 0.14])),
            BoxObstacle(center=np.array([0.66, -0.05, 0.34]), half_extents=np.array([0.05, 0.10, 0.06])),
            BoxObstacle(center=np.array([0.33, 0.22, 0.44]), half_extents=np.array([0.045, 0.06, 0.10])),
            SphereObstacle(center=np.array([0.60, -0.30, 0.52]), radius=0.075),
            SphereObstacle(center=np.array([0.60, 0.30, 0.52]), radius=0.075),
            SphereObstacle(center=np.array([0.72, 0.10, 0.46]), radius=0.07),
            SphereObstacle(center=np.array([0.30, -0.24, 0.50]), radius=0.07),
            SphereObstacle(center=np.array([0.44, 0.00, 0.58]), radius=0.07),
            SphereObstacle(center=np.array([0.50, 0.14, 0.64]), radius=0.06),
            SphereObstacle(center=np.array([0.50, -0.14, 0.64]), radius=0.06),
        ]
    if scene == "cluttered_shelf":
        # Multi-level shelf and side clutter spread across the workspace.
        return [
            BoxObstacle(center=np.array([0, 0.5, 0]), half_extents=np.array([1, 0.1, 1])),
            BoxObstacle(center=np.array([0, -0.5, 0]), half_extents=np.array([1, 0.1, 1])),
            # BoxObstacle(center=np.array([0.5, 0, 0]), half_extents=np.array([0.1, 1, 1])),
            # BoxObstacle(center=np.array([-0.5, 0, 0]), half_extents=np.array([0.1, 1, 1])),
            # BoxObstacle(center=np.array([0.52, 0.00, 0.39]), half_extents=np.array([0.18, 0.24, 0.03])),
            # BoxObstacle(center=np.array([0.42, 0.00, 0.30]), half_extents=np.array([0.03, 0.24, 0.12])),
            # BoxObstacle(center=np.array([0.62, 0.00, 0.30]), half_extents=np.array([0.03, 0.24, 0.12])),
            # BoxObstacle(center=np.array([0.72, -0.16, 0.38]), half_extents=np.array([0.04, 0.08, 0.12])),
            # BoxObstacle(center=np.array([0.32, 0.20, 0.36]), half_extents=np.array([0.04, 0.08, 0.10])),
            # SphereObstacle(center=np.array([0.49, -0.20, 0.31]), radius=0.065),
            # SphereObstacle(center=np.array([0.56, 0.20, 0.32]), radius=0.065),
            # SphereObstacle(center=np.array([0.44, 0.00, 0.53]), radius=0.075),
            # SphereObstacle(center=np.array([0.60, 0.02, 0.53]), radius=0.075),
            # SphereObstacle(center=np.array([0.50, -0.30, 0.50]), radius=0.06),
            # SphereObstacle(center=np.array([0.66, 0.28, 0.48]), radius=0.06),
        ]
    if scene == "cluttered_crossing":
        # Interleaved obstacles all around the arm to force weaving paths.
        return [
            BoxObstacle(center=np.array([0.46, -0.16, 0.38]), half_extents=np.array([0.055, 0.06, 0.14])),
            BoxObstacle(center=np.array([0.46, 0.16, 0.38]), half_extents=np.array([0.055, 0.06, 0.14])),
            BoxObstacle(center=np.array([0.58, 0.00, 0.33]), half_extents=np.array([0.05, 0.13, 0.05])),
            BoxObstacle(center=np.array([0.36, 0.00, 0.33]), half_extents=np.array([0.05, 0.13, 0.05])),
            BoxObstacle(center=np.array([0.66, 0.22, 0.42]), half_extents=np.array([0.04, 0.07, 0.12])),
            BoxObstacle(center=np.array([0.30, -0.20, 0.42]), half_extents=np.array([0.04, 0.07, 0.12])),
            SphereObstacle(center=np.array([0.54, -0.26, 0.48]), radius=0.075),
            SphereObstacle(center=np.array([0.54, 0.26, 0.48]), radius=0.075),
            SphereObstacle(center=np.array([0.42, 0.00, 0.56]), radius=0.08),
            SphereObstacle(center=np.array([0.62, 0.00, 0.56]), radius=0.08),
            SphereObstacle(center=np.array([0.72, -0.08, 0.52]), radius=0.065),
            SphereObstacle(center=np.array([0.28, 0.08, 0.52]), radius=0.065),
        ]
    raise ValueError(f"Unknown scene {scene}")
