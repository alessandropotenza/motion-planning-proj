# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""Optional collision checking with Pinocchio + hpp-fcl (FCL backend).

This backend loads collision meshes from the Panda URDF and measures distances
to analytic workspace obstacles (spheres / axis-aligned boxes). It satisfies the
same interface as ``SphereArmCollisionChecker`` (including ``workspace_margin`` for
CDF-guided RRT*).

Installation (pick one that matches your platform; wheels vary):

- **Conda (recommended):** ``conda install pinocchio -c conda-forge``
- **ROS:** use the ``python3-pinocchio`` / workspace packages for your distro.
- **pip:** sometimes ``pip install pin`` (Pinocchio); ensure it was built with
  hpp-fcl / collision support.

If ``import pinocchio`` fails, keep ``--collision-backend sphere`` (default).

If you use **ROS and conda**, avoid loading Pinocchio from ``/opt/ros/...`` (see
``requirements-franka-pinocchio.txt``); this module prepends the active env's
``site-packages`` before import to prefer conda-forge builds.
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

from workspace_obstacles import BoxObstacle, Obstacle, SphereObstacle

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_URDF = os.path.join(CUR_DIR, "panda_urdf", "panda.urdf")


def _prepend_active_env_site_packages() -> None:
    """Put this interpreter's ``site-packages`` first so conda/venv wins over ROS.

    Sourcing ROS (``setup.bash``) often prepends ``/opt/ros/<distro>/.../site-packages``,
    which shadows a conda-forge Pinocchio and pulls in a build linked against NumPy 1,
    breaking under NumPy 2 with ``_ARRAY_API`` / segfault errors.
    """
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    site = os.path.join(sys.prefix, "lib", f"python{ver}", "site-packages")
    if not os.path.isdir(site):
        return
    norm_site = os.path.normpath(site)
    sys.path[:] = [p for p in sys.path if os.path.normpath(p) != norm_site]
    sys.path.insert(0, site)


def _import_pin_and_fcl():
    _prepend_active_env_site_packages()
    try:
        import pinocchio as pin  # type: ignore
        import hppfcl as fcl  # type: ignore
    except Exception as e:  # pragma: no cover
        msg = str(e)
        extra = []
        if "_ARRAY_API" in msg or "numpy" in msg.lower():
            extra.append(
                "If the traceback shows ``/opt/ros/`` inside pinocchio, ROS is still "
                "shadowing conda: open a shell **without** ``source /opt/ros/.../setup.bash`` "
                "or run ``unset PYTHONPATH`` before ``conda activate``.\n"
                "If pinocchio loads from conda but NumPy errors remain, try: "
                "``conda install 'numpy<2' -c conda-forge`` in the same env.\n"
            )
        elif "opt/ros" in msg:
            extra.append(
                "ROS is ahead of conda on ``sys.path``. Avoid sourcing ROS in this shell, "
                "or ``unset PYTHONPATH``, then re-run with your conda env active.\n"
            )
        raise ImportError(
            "Could not import Pinocchio + hpp-fcl for ``--collision-backend pin``.\n"
            + "".join(extra)
            + "Install (e.g. ``conda install pinocchio -c conda-forge``) or use "
            "``--collision-backend sphere``.\n"
            f"Original error: {msg}"
        ) from e

    pin_file = getattr(pin, "__file__", "") or ""
    if "/opt/ros/" in pin_file.replace("\\", "/"):
        raise ImportError(
            "Pinocchio is still loading from ROS (see ``pinocchio.__file__``), not from "
            "your conda env. Do not ``source /opt/ros/.../setup.bash`` in this shell, or "
            "run ``unset PYTHONPATH`` after conda activate, then retry.\n"
            f"Resolved module file: {pin_file}"
        )
    return pin, fcl


def _arm_joint_limits(pin, model) -> Tuple[np.ndarray, np.ndarray]:
    lowers: List[float] = []
    uppers: List[float] = []
    for i in range(7):
        jid = model.getJointId(f"panda_joint{i + 1}")
        if jid == 0:
            raise RuntimeError(f"panda_joint{i + 1} missing from URDF model.")
        j = model.joints[jid]
        iq = int(j.idx_q)
        lowers.append(float(model.lowerPositionLimit[iq]))
        uppers.append(float(model.upperPositionLimit[iq]))
    return np.asarray(lowers, dtype=np.float64), np.asarray(uppers, dtype=np.float64)


def _q7_to_full_configuration(pin, model, q7: np.ndarray, finger_open: float = 0.04) -> np.ndarray:
    """Map 7 arm joints to the full ``model.nq`` vector (fills finger joints)."""
    q = pin.neutral(model).copy()
    q7 = np.asarray(q7, dtype=np.float64).reshape(7)
    for i in range(7):
        name = f"panda_joint{i + 1}"
        jid = model.getJointId(name)
        if jid == 0:
            raise RuntimeError(f"Joint {name!r} not found in URDF model.")
        joint = model.joints[jid]
        if joint.nq != 1:
            raise RuntimeError(f"Joint {name!r} has nq={joint.nq}, expected 1 (revolute).")
        q[joint.idx_q] = q7[i]
    for jn in ("panda_finger_joint1", "panda_finger_joint2"):
        jid = model.getJointId(jn)
        if jid != 0:
            j = model.joints[jid]
            if j.nq >= 1:
                q[j.idx_q] = finger_open
    return q


def _se3_to_transform3f(fcl, M) -> "object":
    """Convert pin.SE3 to hppfcl.Transform3f."""
    R = np.asarray(M.rotation, dtype=np.float64)
    t = np.asarray(M.translation, dtype=np.float64).reshape(3)
    return fcl.Transform3f(R, t)


def _obstacle_to_fcl(fcl, o: Obstacle) -> Tuple[object, object]:
    """Return (CollisionGeometry, Transform3f in world frame) for primitive ``o``."""
    if isinstance(o, SphereObstacle):
        geom = fcl.Sphere(float(o.radius))
        T = fcl.Transform3f()
        c = np.asarray(o.center, dtype=np.float64).reshape(3)
        T.setTranslation(c)
        return geom, T
    if isinstance(o, BoxObstacle):
        he = np.asarray(o.half_extents, dtype=np.float64).reshape(3)
        # hppfcl.Box expects **half-extents** (same convention as our BoxObstacle).
        geom = fcl.Box(float(he[0]), float(he[1]), float(he[2]))
        T = fcl.Transform3f()
        T.setTranslation(np.asarray(o.center, dtype=np.float64).reshape(3))
        return geom, T
    raise TypeError(f"Unsupported obstacle type: {type(o)}")


def _distance_min(fcl, ga, Ta, gb, Tb) -> float:
    req = fcl.DistanceRequest()
    res = fcl.DistanceResult()
    out = fcl.distance(ga, Ta, gb, Tb, req, res)
    if isinstance(out, (float, int, np.floating)):
        return float(out)
    d = getattr(res, "min_distance", None)
    if d is not None:
        return float(d)
    return float(getattr(res, "distance", 0.0))


class PinFclCollisionChecker:
    """Pinocchio kinematics + hpp-fcl distances from URDF collision meshes to obstacles."""

    def __init__(
        self,
        obstacles: Sequence[Obstacle],
        margin: float = 0.02,
        urdf_path: str = DEFAULT_URDF,
        finger_open: float = 0.04,
    ) -> None:
        pin, fcl = _import_pin_and_fcl()
        self._pin = pin
        self._fcl = fcl
        self.margin = float(margin)
        self.urdf_path = urdf_path
        self.package_dir = os.path.dirname(os.path.abspath(urdf_path))
        self.finger_open = float(finger_open)

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.model = pin.buildModelFromUrdf(urdf_path)
        geom_type = getattr(pin, "GeometryType", None)
        if geom_type is not None:
            coll_type = getattr(geom_type, "COLLISION", None)
        else:
            coll_type = None
        if coll_type is None:
            coll_type = getattr(pin, "COLLISION", 1)

        try:
            self.geom_model = pin.buildGeomFromUrdf(
                self.model, urdf_path, coll_type, package_dirs=[self.package_dir]
            )
        except (TypeError, AttributeError):
            try:
                self.geom_model = pin.buildGeomFromUrdf(self.model, urdf_path, coll_type, self.package_dir)
            except TypeError:
                self.geom_model = pin.buildGeomFromUrdf(self.model, urdf_path, coll_type)

        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)

        lo, hi = _arm_joint_limits(pin, self.model)
        self.q_min = lo.astype(np.float32)
        self.q_max = hi.astype(np.float32)

        self._obstacles_fcl: List[Tuple[object, object]] = [_obstacle_to_fcl(fcl, o) for o in obstacles]

        if len(self.geom_model.geometryObjects) == 0:
            raise RuntimeError(
                "Pinocchio loaded zero collision geometries from the URDF. "
                "Check mesh paths (package://…) and package_dirs."
            )

    def _update_placements(self, q7: np.ndarray) -> None:
        pin = self._pin
        q = _q7_to_full_configuration(pin, self.model, q7, finger_open=self.finger_open)
        pin.forwardKinematics(self.model, self.data, q)
        try:
            pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        except TypeError:
            pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data)

    def _min_distance_to_obstacles(self, q7: np.ndarray) -> float:
        """Minimum distance between any robot collision geometry and any obstacle."""
        self._update_placements(q7)
        fcl = self._fcl
        d_min = np.inf
        for k, go in enumerate(self.geom_model.geometryObjects):
            oMg = self.geom_data.oMg[k]
            Ta = _se3_to_transform3f(fcl, oMg)
            ga = go.geometry
            for obs_geom, obs_T in self._obstacles_fcl:
                d = _distance_min(fcl, ga, Ta, obs_geom, obs_T)
                if d < d_min:
                    d_min = d
        if not np.isfinite(d_min):
            return 1e9
        return float(d_min)

    def workspace_margin(self, q: np.ndarray) -> float:
        """Minimum geometry-obstacle distance minus ``margin`` (positive when free)."""
        return self._min_distance_to_obstacles(np.asarray(q, dtype=np.float64).reshape(7)) - self.margin

    def is_state_free(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        if not (np.all(q >= self.q_min) and np.all(q <= self.q_max)):
            return False
        return self.workspace_margin(q) > 0.0

    def is_edge_free(self, q_from: np.ndarray, q_to: np.ndarray, edge_resolution: float) -> bool:
        import math

        q_from = np.asarray(q_from, dtype=np.float64).reshape(7)
        q_to = np.asarray(q_to, dtype=np.float64).reshape(7)
        dist = float(np.linalg.norm(q_to - q_from))
        n = max(2, int(math.ceil(dist / edge_resolution)) + 1)
        for a in np.linspace(0.0, 1.0, n):
            q = (1.0 - a) * q_from + a * q_to
            if not self.is_state_free(q):
                return False
        return True
