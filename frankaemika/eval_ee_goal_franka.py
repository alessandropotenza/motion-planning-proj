#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Checkpointed evaluator for EE-goal Franka planners (7-DOF).
# Mirrors 2Dexamples/eval_ee_goal_rrtstar.py for the Franka robot.
# Writes detailed_log.csv consumed by data_analysis_franka*.py scripts.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from franka_kinematics import fk_flange_position
from rrt_star_franka import Node, VanillaFrankaRRTStar, path_waypoint_cost
from sphere_arm_collision import SphereArmCollisionChecker
from workspace_obstacles import build_demo_obstacles

EE_IMPORT_ERROR: Optional[Exception] = None
try:
    from ee_goal_rrtstar_franka import (
        SCENE_CHOICES,
        PullAndSlideFrankaRRTStar,
        TaskGoalCDFFrankaRRTStar,
        default_start_and_goal_task,
        solve_task_goal_ik,
    )
except Exception as exc:
    EE_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "frankaemika" / "outputs" / "eval_ee_goal_franka"


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    start_q: Tuple[float, ...]
    goal_task: Tuple[float, float, float]


@dataclass
class EvalConfig:
    planners: List[str]
    scenes: List[str]
    scene_queries: Dict[str, List[BenchmarkQuery]]
    log_start_iter: int
    max_iters: int
    seed: int

    step_size: float = 0.12
    goal_threshold: float = 0.35
    goal_bias: float = 0.08
    neighbor_radius: float = 0.65
    edge_resolution: float = 0.06
    ee_goal_threshold: float = 0.06

    ik_restarts: int = 40
    ik_max_steps: int = 140

    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: Optional[str] = None


@dataclass
class RunSnapshot:
    iteration_budget: int
    path_found: bool
    nodes: List[Node]
    path: Optional[np.ndarray]
    stats: Dict[str, Any]
    error: Optional[str]


def _default_scene_queries() -> Dict[str, List[BenchmarkQuery]]:
    queries: Dict[str, List[BenchmarkQuery]] = {}
    for scene in SCENE_CHOICES:
        sq, gt = default_start_and_goal_task(scene)
        queries[scene] = [
            BenchmarkQuery("q1", tuple(float(x) for x in sq), tuple(float(x) for x in gt))
        ]
    return queries


def _make_default_config() -> EvalConfig:
    return EvalConfig(
        planners=["vanilla", "cdf", "pullandslide"],
        scenes=list(SCENE_CHOICES),
        scene_queries=_default_scene_queries(),
        log_start_iter=50,
        max_iters=800,
        seed=1,
    )


def build_checkpoints(start_iter: int, max_iters: int) -> List[int]:
    if start_iter <= 0:
        raise ValueError("log_start_iter must be > 0")
    checkpoints: List[int] = []
    current = int(start_iter)
    while current < max_iters:
        checkpoints.append(current)
        current *= 2
    checkpoints.append(max_iters)
    return sorted(set(checkpoints))


def path_length(path: Optional[np.ndarray]) -> float:
    if path is None or len(path) < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))


def ee_goal_error(path: Optional[np.ndarray], goal_task: np.ndarray) -> float:
    if path is None or len(path) == 0:
        return float("nan")
    ee = fk_flange_position(path[-1].astype(np.float64)).astype(np.float32)
    return float(np.linalg.norm(ee - goal_task))


def _make_vanilla(checker: SphereArmCollisionChecker, cfg: EvalConfig) -> VanillaFrankaRRTStar:
    return VanillaFrankaRRTStar(
        checker=checker,
        step_size=cfg.step_size,
        goal_threshold=cfg.goal_threshold,
        goal_bias=cfg.goal_bias,
        neighbor_radius=cfg.neighbor_radius,
        edge_resolution=cfg.edge_resolution,
    )


def _make_cdf(checker, obstacles, device, cfg: EvalConfig):
    return TaskGoalCDFFrankaRRTStar(
        checker=checker,
        obstacles=obstacles,
        device=device,
        step_size=cfg.step_size,
        goal_threshold=cfg.goal_threshold,
        goal_bias=max(cfg.goal_bias, 0.10),
        neighbor_radius=cfg.neighbor_radius,
        edge_resolution=cfg.edge_resolution,
        ee_goal_threshold=cfg.ee_goal_threshold,
    )


def _make_pullandslide(checker, obstacles, device, cfg: EvalConfig):
    return PullAndSlideFrankaRRTStar(
        checker=checker,
        obstacles=obstacles,
        device=device,
        step_size=cfg.step_size,
        goal_threshold=cfg.goal_threshold,
        goal_bias=max(cfg.goal_bias, 0.12),
        neighbor_radius=cfg.neighbor_radius,
        edge_resolution=cfg.edge_resolution,
        ee_goal_threshold=cfg.ee_goal_threshold,
    )


def ensure_run_dir(cfg: EvalConfig) -> Path:
    run_name = cfg.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_case_with_cache(
    planner_name: str,
    scene_name: str,
    query: BenchmarkQuery,
    checker: SphereArmCollisionChecker,
    obstacles,
    device,
    goal_q: Optional[np.ndarray],
    ik_solutions: Optional[np.ndarray],
    checkpoints: List[int],
    cfg: EvalConfig,
) -> List[Dict[str, Any]]:
    cache: Dict[int, RunSnapshot] = {}
    start_q = np.asarray(query.start_q, dtype=np.float32)
    goal_task = np.asarray(query.goal_task, dtype=np.float32)

    def run_at_iters(iter_budget: int) -> RunSnapshot:
        if iter_budget in cache:
            return cache[iter_budget]
        try:
            if planner_name == "vanilla":
                planner = _make_vanilla(checker, cfg)
                nodes, path_out, stats = planner.plan(
                    start_q, goal_q, max_iters=iter_budget, seed=cfg.seed,
                )
            elif planner_name == "cdf":
                planner = _make_cdf(checker, obstacles, device, cfg)
                planner.set_goal_task(goal_task)
                nodes, path_out, stats = planner.plan(
                    start_q, start_q.copy(), max_iters=iter_budget, seed=cfg.seed,
                )
            elif planner_name == "pullandslide":
                planner = _make_pullandslide(checker, obstacles, device, cfg)
                planner.set_goal_task(goal_task)
                nodes, path_out, stats = planner.plan(
                    start_q, start_q.copy(), max_iters=iter_budget, seed=cfg.seed,
                )
            else:
                raise ValueError(f"Unknown planner '{planner_name}'")
            snap = RunSnapshot(iter_budget, bool(stats["success"]), nodes, path_out, stats, None)
        except Exception as exc:
            snap = RunSnapshot(iter_budget, False, [], None, {}, str(exc))
        cache[iter_budget] = snap
        return snap

    for cp in checkpoints:
        run_at_iters(cp)

    first_success_cp: Optional[int] = None
    for cp in checkpoints:
        if cache[cp].path_found:
            first_success_cp = cp
            break

    first_path_iteration: Optional[int] = None
    if first_success_cp is not None:
        lo, hi = 1, first_success_cp
        while lo < hi:
            mid = (lo + hi) // 2
            if run_at_iters(mid).path_found:
                hi = mid
            else:
                lo = mid + 1
        first_path_iteration = lo

    events: List[Tuple[int, str]] = [(cp, "checkpoint") for cp in checkpoints]
    if first_path_iteration is not None:
        if first_path_iteration in checkpoints:
            events = [
                (it, "checkpoint_first_path" if it == first_path_iteration else kind)
                for it, kind in events
            ]
        else:
            events.append((first_path_iteration, "first_path"))
    events.sort(key=lambda x: (x[0], x[1]))

    rows: List[Dict[str, Any]] = []
    for iter_budget, event_type in events:
        snap = run_at_iters(iter_budget)

        nodes_total = float("nan")
        accepted_nodes = float("nan")
        discarded_nodes = float("nan")
        rewires_val = float("nan")
        rejection_rate = float("nan")
        planning_time_sec = float("nan")
        final_path_cost = float("nan")
        path_waypoints = 0
        q_path_length = float("nan")
        ee_err = float("nan")
        config_goal_vals = [float("nan")] * 7

        if snap.stats:
            stats = snap.stats
            nodes_total = float(len(snap.nodes))
            accepted_nodes = float(stats.get("accepted_nodes", float("nan")))
            discarded_nodes = float(stats.get("discarded_samples", float("nan")))
            rewires_val = float(stats.get("rewires", float("nan")))
            rejection_rate = float(stats.get("rejection_rate", float("nan")))
            planning_time_sec = float(stats.get("planning_time_sec", float("nan")))
            fpc = stats.get("final_path_cost")
            final_path_cost = float(fpc) if fpc is not None else float("nan")
            path_waypoints = int(len(snap.path)) if snap.path is not None else 0
            q_path_length = path_length(snap.path)
            ee_err = ee_goal_error(snap.path, goal_task)
            if goal_q is not None:
                config_goal_vals = [float(goal_q[i]) for i in range(7)]

        row: Dict[str, Any] = {
            "planner": planner_name,
            "scene": scene_name,
            "query_id": query.query_id,
            "seed": cfg.seed,
            "event_type": event_type,
            "iteration_budget": iter_budget,
            "path_found": int(snap.path_found),
            "first_path_iteration": first_path_iteration if first_path_iteration is not None else "",
            "nodes_total": nodes_total,
            "accepted_nodes": accepted_nodes,
            "discarded_nodes": discarded_nodes,
            "rewires": rewires_val,
            "rejection_rate": rejection_rate,
            "path_length": q_path_length,
            "path_waypoints": path_waypoints,
            "final_path_cost": final_path_cost,
            "ee_goal_error": ee_err,
            "planning_time_sec": planning_time_sec,
            "error": snap.error or "",
        }
        for i in range(7):
            row[f"config_goal_q{i + 1}"] = config_goal_vals[i]
        rows.append(row)

    return rows


CSV_FIELDNAMES: List[str] = [
    "planner", "scene", "query_id", "seed", "event_type", "iteration_budget",
    "path_found", "first_path_iteration",
    "nodes_total", "accepted_nodes", "discarded_nodes", "rewires", "rejection_rate",
    "path_length", "path_waypoints", "final_path_cost", "ee_goal_error",
    "planning_time_sec",
] + [f"config_goal_q{i + 1}" for i in range(7)] + ["error"]


def write_detailed_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr))


def write_planner_summaries(rows: List[Dict[str, Any]], cfg: EvalConfig, run_dir: Path) -> None:
    if not rows:
        return
    for planner_name in cfg.planners:
        planner_rows = [
            r for r in rows
            if r["planner"] == planner_name
            and int(r["iteration_budget"]) == int(cfg.max_iters)
            and r["event_type"] in ("checkpoint", "checkpoint_first_path")
        ]
        summary_rows: List[Dict[str, Any]] = []
        scenes = sorted({r["scene"] for r in planner_rows})
        for scene in scenes:
            scene_rows = [r for r in planner_rows if r["scene"] == scene]
            n_q = len(scene_rows)
            success_vals = [int(r["path_found"]) for r in scene_rows]
            success_rate = float(np.mean(success_vals)) if n_q > 0 else float("nan")
            planning_time = [float(r["planning_time_sec"]) for r in scene_rows]
            path_lengths = [float(r["path_length"]) for r in scene_rows]
            nodes_tot = [float(r["nodes_total"]) for r in scene_rows]
            discarded = [float(r["discarded_nodes"]) for r in scene_rows]
            ee_errors = [float(r["ee_goal_error"]) for r in scene_rows]
            fpi_success = [
                float(r["first_path_iteration"])
                for r in scene_rows
                if int(r["path_found"]) == 1 and str(r["first_path_iteration"]) != ""
            ]
            summary_rows.append({
                "planner": planner_name,
                "scene": scene,
                "num_queries": n_q,
                "success_rate": success_rate,
                "planning_time_mean": _nanmean(planning_time),
                "planning_time_std": _nanstd(planning_time),
                "path_length_mean": _nanmean(path_lengths),
                "path_length_std": _nanstd(path_lengths),
                "nodes_total_mean": _nanmean(nodes_tot),
                "nodes_total_std": _nanstd(nodes_tot),
                "discarded_nodes_mean": _nanmean(discarded),
                "discarded_nodes_std": _nanstd(discarded),
                "ee_goal_error_mean": _nanmean(ee_errors),
                "ee_goal_error_std": _nanstd(ee_errors),
                "first_path_iteration_mean_success_only": _nanmean(fpi_success),
            })
        fieldnames = [
            "planner", "scene", "num_queries", "success_rate",
            "planning_time_mean", "planning_time_std",
            "path_length_mean", "path_length_std",
            "nodes_total_mean", "nodes_total_std",
            "discarded_nodes_mean", "discarded_nodes_std",
            "ee_goal_error_mean", "ee_goal_error_std",
            "first_path_iteration_mean_success_only",
        ]
        summary_path = run_dir / f"summary_{planner_name}.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)


def main(cfg: Optional[EvalConfig] = None) -> None:
    if EE_IMPORT_ERROR is not None:
        raise SystemExit(
            f"Failed to import planners from ee_goal_rrtstar_franka.py.\n"
            f"Original import error: {EE_IMPORT_ERROR}"
        )
    if cfg is None:
        cfg = _make_default_config()

    need_torch = any(p in ("cdf", "pullandslide") for p in cfg.planners)
    if need_torch and torch is None:
        raise SystemExit("PyTorch required for CDF/Pull-and-slide. Install torch or remove them from config.")

    if torch is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    checkpoints = build_checkpoints(cfg.log_start_iter, cfg.max_iters)
    run_dir = ensure_run_dir(cfg)
    all_rows: List[Dict[str, Any]] = []

    np.random.seed(cfg.seed)
    if torch is not None:
        torch.manual_seed(cfg.seed)

    print("=== Franka EE-Goal Planner Evaluation (7-DOF) ===")
    print(f"run_dir:      {run_dir}")
    print(f"planners:     {cfg.planners}")
    print(f"scenes:       {cfg.scenes}")
    print(f"checkpoints:  {checkpoints}")
    print(f"seed={cfg.seed}, max_iters={cfg.max_iters}")

    t0_total = time.time()
    for scene_name in cfg.scenes:
        rng = np.random.default_rng(cfg.seed)
        obstacles = build_demo_obstacles(scene_name, rng)
        checker = SphereArmCollisionChecker(obstacles, margin=0.02)

        queries = cfg.scene_queries.get(scene_name, [])
        if not queries:
            print(f"[WARN] No queries for scene={scene_name}, skipping.")
            continue

        for query in queries:
            start_q = np.asarray(query.start_q, dtype=np.float32)
            goal_task = np.asarray(query.goal_task, dtype=np.float32)

            if not checker.is_state_free(start_q):
                print(f"[WARN] Skipping {scene_name}/{query.query_id}: start in collision.")
                continue

            goal_q: Optional[np.ndarray] = None
            ik_solutions: Optional[np.ndarray] = None
            if "vanilla" in cfg.planners:
                try:
                    goal_q, ik_solutions = solve_task_goal_ik(
                        checker, start_q, goal_task,
                        seed=cfg.seed, n_restarts=cfg.ik_restarts, max_steps=cfg.ik_max_steps,
                    )
                except Exception as exc:
                    print(f"[WARN] IK failed for {scene_name}/{query.query_id}: {exc}")

            for planner_name in cfg.planners:
                if planner_name == "vanilla" and goal_q is None:
                    print(f"[SKIP] vanilla/{scene_name}/{query.query_id}: no IK solution.")
                    continue
                print(
                    f"[RUN] planner={planner_name}  scene={scene_name}  "
                    f"query={query.query_id}  checkpoints={len(checkpoints)}"
                )
                t0 = time.time()
                rows = run_case_with_cache(
                    planner_name=planner_name,
                    scene_name=scene_name,
                    query=query,
                    checker=checker,
                    obstacles=obstacles,
                    device=device,
                    goal_q=goal_q,
                    ik_solutions=ik_solutions,
                    checkpoints=checkpoints,
                    cfg=cfg,
                )
                elapsed = time.time() - t0
                cp_row = next((r for r in rows if int(r["iteration_budget"]) == cfg.max_iters), None)
                success_flag = bool(cp_row and int(cp_row.get("path_found", 0))) if cp_row else False
                print(
                    f"       done in {elapsed:.1f}s — "
                    f"{'SUCCESS' if success_flag else 'no path'} "
                    f"({len(rows)} log rows)"
                )
                all_rows.extend(rows)

    detailed_path = run_dir / "detailed_log.csv"
    write_detailed_csv(all_rows, detailed_path)
    write_planner_summaries(all_rows, cfg, run_dir)

    print(f"\nWrote detailed CSV: {detailed_path}")
    for pn in cfg.planners:
        print(f"Wrote summary CSV:  {run_dir / f'summary_{pn}.csv'}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Elapsed: {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
