# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Part of the CDF project motion-planning examples.
# -----------------------------------------------------------------------------
"""Minimal Franka Panda arm FK from `panda_urdf/panda.urdf` (7-DOF chain only).

Used for analytic collision proxies (spheres) without PyBullet. Base is fixed at
the world origin matching the PyBullet demo.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PANDA_URDF = os.path.join(CUR_DIR, "panda_urdf", "panda.urdf")


def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


def _T_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_R(np.asarray(rpy, dtype=np.float64))
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _parse_vector(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split()]
    return np.asarray(parts, dtype=np.float64)


def load_arm_chain_from_urdf(
    urdf_path: str = PANDA_URDF,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Return joint specs in serial order, q_min, q_max (7,)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        # URDF stores joint type as an attribute, not a <type> child element.
        if j.get("type") != "revolute":
            continue
        if not name.startswith("panda_joint") or name == "panda_joint8":
            continue
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        o = j.find("origin")
        xyz = _parse_vector(o.get("xyz", "0 0 0"))
        rpy = _parse_vector(o.get("rpy", "0 0 0"))
        axis_el = j.find("axis")
        axis = _parse_vector(axis_el.get("xyz", "0 0 1")) if axis_el is not None else np.array([0, 0, 1.0])
        lim = j.find("limit")
        lower = float(lim.get("lower"))
        upper = float(lim.get("upper"))
        joints[name] = {
            "name": name,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }

    order = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    chain = [joints[k] for k in order]
    q_min = np.array([j["lower"] for j in chain], dtype=np.float32)
    q_max = np.array([j["upper"] for j in chain], dtype=np.float32)
    return chain, q_min, q_max


_ARM_CHAIN, Q_MIN_DEFAULT, Q_MAX_DEFAULT = load_arm_chain_from_urdf()


def fk_link_origins(q: np.ndarray, chain: Optional[List[dict]] = None) -> np.ndarray:
    """World positions of each revolute child link origin (7,) after joints 1..7.

    Args:
        q: shape (7,) joint angles in radians.

    Returns:
        (7, 3) array of link-frame origins in world coordinates (matches URDF chain).
    """
    q = np.asarray(q, dtype=np.float64).reshape(7)
    chain = chain or _ARM_CHAIN
    T = np.eye(4, dtype=np.float64)
    origins = np.zeros((7, 3), dtype=np.float64)
    for i, spec in enumerate(chain):
        Ti = _T_from_xyz_rpy(spec["xyz"], spec["rpy"])
        Rq = _rot_axis(spec["axis"], q[i])
        T = T @ Ti @ Rq
        origins[i] = T[:3, 3]
    return origins


def fk_flange_position(q: np.ndarray, chain: Optional[List[dict]] = None) -> np.ndarray:
    """panda_link8 origin (fixed offset from link7) — matches URDF fixed joint."""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    chain = chain or _ARM_CHAIN
    T = np.eye(4, dtype=np.float64)
    for i, spec in enumerate(chain):
        Ti = _T_from_xyz_rpy(spec["xyz"], spec["rpy"])
        Rq = _rot_axis(spec["axis"], q[i])
        T = T @ Ti @ Rq
    T = T @ _T_from_xyz_rpy(np.array([0, 0, 0.107]), np.array([0, 0, 0]))
    return T[:3, 3]
