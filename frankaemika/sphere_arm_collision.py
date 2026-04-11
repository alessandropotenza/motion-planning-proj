# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""Conservative sphere-inflation collision model for the Franka arm (no PyBullet).

Uses FK from `franka_kinematics` plus task-space obstacle SDFs from
`workspace_obstacles`. This is intentionally simple; tighten radii or add
spheres if you see false negatives in tight scenes.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from franka_kinematics import fk_link_origins, fk_flange_position, Q_MAX_DEFAULT, Q_MIN_DEFAULT
from workspace_obstacles import Obstacle, union_sdf_np


class SphereArmCollisionChecker:
    """Sphere soup approximating the arm vs analytic workspace obstacles."""

    def __init__(
        self,
        obstacles: Sequence[Obstacle],
        margin: float = 0.02,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ) -> None:
        self.obstacles = list(obstacles)
        self.margin = float(margin)
        self.q_min = np.asarray(q_min if q_min is not None else Q_MIN_DEFAULT, dtype=np.float32)
        self.q_max = np.asarray(q_max if q_max is not None else Q_MAX_DEFAULT, dtype=np.float32)
        # Base + 7 link origins + flange (conservative radii, meters)
        self._radii = np.array(
            [0.14, 0.11, 0.11, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06],
            dtype=np.float64,
        )

    def _sphere_centers(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        link_o = fk_link_origins(q)
        flange = fk_flange_position(q).reshape(1, 3)
        base = np.array([[0.0, 0.0, 0.06]], dtype=np.float64)
        return np.vstack([base, link_o, flange])

    def workspace_margin(self, q: np.ndarray) -> float:
        """Scalar clearance: positive if all spheres are outside obstacles (+ margin)."""
        centers = self._sphere_centers(q)
        d = union_sdf_np(centers, self.obstacles) - self._radii
        return float(np.min(d) - self.margin)

    def is_state_free(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        if not (np.all(q >= self.q_min) and np.all(q <= self.q_max)):
            return False
        return self.workspace_margin(q) > 0.0

    def is_edge_free(self, q_from: np.ndarray, q_to: np.ndarray, edge_resolution: float) -> bool:
        q_from = np.asarray(q_from, dtype=np.float64).reshape(7)
        q_to = np.asarray(q_to, dtype=np.float64).reshape(7)
        dist = float(np.linalg.norm(q_to - q_from))
        n = max(2, int(math.ceil(dist / edge_resolution)) + 1)
        for a in np.linspace(0.0, 1.0, n):
            q = (1.0 - a) * q_from + a * q_to
            if not self.is_state_free(q):
                return False
        return True
