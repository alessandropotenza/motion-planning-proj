#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Checkpointed evaluator for EE-goal planners in ee_goal_rrtstar_2d.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cdf import CDF2D
from rrt_star_2d import make_scene


# Fail fast with a clear message if ee_goal_rrtstar_2d imports are unavailable.
EE_IMPORT_ERROR: Optional[Exception] = None
try:
    from ee_goal_rrtstar_2d import (
        CDFEERRTStar,
        PullAndSlide,
        VanillaEERRTStar,
        fk_end_effector,
        plot_configuration_space,
        plot_task_space,
    )
except Exception as exc:  # pragma: no cover - this is for environment diagnosis.
    EE_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "2Dexamples" / "outputs" / "eval_ee_goal_rrtstar"
DEFAULT_WB_MODEL = PROJECT_ROOT / "2Dexamples" / "model_wb.pth"
DEFAULT_EE_MODEL = PROJECT_ROOT / "2Dexamples" / "model_ee.pth"


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    start_q: Tuple[float, float]
    goal_task: Tuple[float, float]


@dataclass
class EvalConfig:
    planners: List[str]
    scenes: List[str]
    scene_queries: Dict[str, List[BenchmarkQuery]]
    log_start_iter: int
    max_iters: int
    seed: int

    # Shared planner settings
    step_size: float = 0.25
    goal_threshold: float = 0.25
    goal_bias: float = 0.10
    neighbor_radius: float = 0.5
    edge_resolution: float = 0.05
    ee_goal_threshold: float = 0.15

    # CDF / Pull-and-slide settings
    safety_margin_c_space: float = 0.0
    softmin_beta: float = 50.0
    wb_model_path: Path = DEFAULT_WB_MODEL
    ee_model_path: Path = DEFAULT_EE_MODEL

    # Outputs
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: Optional[str] = None
    save_plots: bool = True


@dataclass
class RunSnapshot:
    iteration_budget: int
    path_found: bool
    result: Optional[Dict[str, Any]]
    error: Optional[str]


def default_scene_queries() -> Dict[str, List[BenchmarkQuery]]:
    """Predefined query sets (different starts + task-space goals) per scene."""
    return {
        "scene_1": [
            BenchmarkQuery("q1", (-2.0, -1.0), (-1.0, 3.0)),
            # BenchmarkQuery("q2", (-1.6, -1.2), (-1.5, 2.8)),
        ],
        "scene_2": [
            BenchmarkQuery("q1", (-1.1, 0.5), (0.0, 1.0)),
            # BenchmarkQuery("q2", (-2.0, 0.0), (-1.0, 3.3)),
        ],
        "scene_3": [
            BenchmarkQuery("q1", (-0.4, 0.4), (1.0, 3.2)),
            # BenchmarkQuery("q2", (-0.8, -2.2), (-0.2, 1.2)),
        ],
        "scene_4": [
            BenchmarkQuery("q1", (-0.4, 0.4), (1.0, 3.2)),
            # BenchmarkQuery("q2", (-1.6, 1.2), (-3.5, 1.5)),
        ],
        "scene_5": [
            BenchmarkQuery("q1", (-1.57, -1.57), (-2.0, 2.0)),
            # BenchmarkQuery("q2", (-0.2, -2.2), (-0.2, 1.2)),
        ],
        "scene_6": [
            BenchmarkQuery("q1", (-math.pi / 12.0, 0.0), (3.864, 1.035)),
            # BenchmarkQuery("q2", (-0.10, -0.20), (3.864, 1.035)),
        ],
    }


# Light defaults for quick smoke testing.
CONFIG = EvalConfig(
    # planners=["rrt", "cdf", "pullandslide"],
    planners=["rrt", "pullandslide"],
    scenes=["scene_3", "scene_5", "scene_6"],
    scene_queries=default_scene_queries(),
    log_start_iter=50,
    max_iters=1600,
    seed=1,
)


def build_checkpoints(start_iter: int, max_iters: int) -> List[int]:
    if start_iter <= 0:
        raise ValueError("log_start_iter must be > 0")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")

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
    diffs = path[1:] - path[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def ee_goal_error(cdf: CDF2D, path: Optional[np.ndarray], goal_task: np.ndarray) -> float:
    if path is None or len(path) == 0:
        return float("nan")
    ee_end = fk_end_effector(cdf, path[-1])
    return float(np.linalg.norm(ee_end - goal_task))


def planner_factory(
    planner_name: str,
    cdf: CDF2D,
    obj_list,
    q_min: np.ndarray,
    q_max: np.ndarray,
    cfg: EvalConfig,
):
    common_kwargs = dict(
        cdf=cdf,
        obj_list=obj_list,
        q_min=q_min,
        q_max=q_max,
        step_size=cfg.step_size,
        goal_threshold=cfg.goal_threshold,
        goal_bias=cfg.goal_bias,
        neighbor_radius=cfg.neighbor_radius,
        edge_resolution=cfg.edge_resolution,
    )
    if planner_name == "rrt":
        return VanillaEERRTStar(**common_kwargs)
    if planner_name == "cdf":
        return CDFEERRTStar(
            **common_kwargs,
            safety_margin_c_space=cfg.safety_margin_c_space,
            softmin_beta=cfg.softmin_beta,
            model_path=str(cfg.wb_model_path),
            ee_model_path=str(cfg.ee_model_path),
            ee_goal_threshold=cfg.ee_goal_threshold,
        )
    if planner_name == "pullandslide":
        return PullAndSlide(
            **common_kwargs,
            wb_model_path=str(cfg.wb_model_path),
            ee_model_path=str(cfg.ee_model_path),
            ee_goal_threshold=cfg.ee_goal_threshold,
        )
    raise ValueError(f"Unknown planner '{planner_name}'")


def ensure_run_dir(cfg: EvalConfig) -> Path:
    run_name = cfg.run_name
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def plot_ee_goal_cdf_snapshot(
    cdf: CDF2D,
    start_q: np.ndarray,
    goal_task: np.ndarray,
    path_q: Optional[np.ndarray],
    ik_solutions: Optional[np.ndarray],
    title: str,
    file_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    cdf.plot_ee_goal_cdf(ax, goal_task, ik_solutions=ik_solutions)

    ax.plot(start_q[0], start_q[1], "go", markersize=8, label="Start (q)")
    if path_q is not None and len(path_q) > 0:
        ax.plot(path_q[:, 0], path_q[:, 1], "r-", linewidth=2.0, label="Path (q)")
        ax.plot(path_q[-1, 0], path_q[-1, 1], "bo", markersize=8, label="Current end (q)")

    ax.text(
        0.02,
        0.02,
        f"task goal = ({goal_task[0]:.2f}, {goal_task[1]:.2f})",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def run_case_with_cache(
    planner_name: str,
    scene_name: str,
    query: BenchmarkQuery,
    planner,
    cdf: CDF2D,
    checkpoints: List[int],
    cfg: EvalConfig,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    cache: Dict[int, RunSnapshot] = {}
    start_q = np.asarray(query.start_q, dtype=np.float32)
    goal_task = np.asarray(query.goal_task, dtype=np.float32)

    def run_at_iters(iter_budget: int) -> RunSnapshot:
        if iter_budget in cache:
            return cache[iter_budget]
        try:
            result = planner.solve(
                start_q=start_q,
                goal_task=goal_task,
                max_iters=iter_budget,
                seed=cfg.seed,
            )
            snap = RunSnapshot(
                iteration_budget=iter_budget,
                path_found=bool(result["stats"]["success"]),
                result=result,
                error=None,
            )
        except Exception as exc:  # Keep evaluation running for other cases.
            snap = RunSnapshot(
                iteration_budget=iter_budget,
                path_found=False,
                result=None,
                error=str(exc),
            )
        cache[iter_budget] = snap
        return snap

    # Always evaluate checkpoint budgets first.
    for cp in checkpoints:
        run_at_iters(cp)

    # Find first successful checkpoint, then binary search for exact first success.
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

    # Build log events:
    # - checkpoints at fixed budgets
    # - one extra first-path event only if it does not coincide with a checkpoint
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
        result = snap.result

        nodes_total = float("nan")
        accepted_nodes = float("nan")
        discarded_nodes = float("nan")
        rewires = float("nan")
        rejection_rate = float("nan")
        planning_time_sec = float("nan")
        final_path_cost = float("nan")
        path_waypoints = 0
        q_path_length = float("nan")
        ee_err = float("nan")
        config_goal_q1 = float("nan")
        config_goal_q2 = float("nan")
        path_q: Optional[np.ndarray] = None
        nodes = []
        ik_solutions = None

        if result is not None:
            stats = result["stats"]
            nodes = result["nodes"]
            path_q = result["path"]
            ik_solutions = result.get("ik_solutions")

            nodes_total = float(len(nodes))
            accepted_nodes = float(stats.get("accepted_nodes", float("nan")))
            discarded_nodes = float(stats.get("discarded_samples", float("nan")))
            rewires = float(stats.get("rewires", float("nan")))
            rejection_rate = float(stats.get("rejection_rate", float("nan")))
            planning_time_sec = float(stats.get("planning_time_sec", float("nan")))
            final_path_cost = float(stats["final_path_cost"]) if stats.get("final_path_cost") is not None else float("nan")
            path_waypoints = int(len(path_q)) if path_q is not None else 0
            q_path_length = path_length(path_q)
            ee_err = ee_goal_error(cdf, path_q, goal_task)

            config_goal_marker = result.get("config_goal_marker")
            if config_goal_marker is not None:
                config_goal_q1 = float(config_goal_marker[0])
                config_goal_q2 = float(config_goal_marker[1])
        else:
            config_goal_marker = None

        row = {
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
            "rewires": rewires,
            "rejection_rate": rejection_rate,
            "path_length": q_path_length,
            "path_waypoints": path_waypoints,
            "final_path_cost": final_path_cost,
            "ee_goal_error": ee_err,
            "planning_time_sec": planning_time_sec,
            "config_goal_q1": config_goal_q1,
            "config_goal_q2": config_goal_q2,
            "error": snap.error or "",
        }
        rows.append(row)

        if cfg.save_plots:
            plot_dir = run_dir / "plots" / planner_name / scene_name / query.query_id
            plot_dir.mkdir(parents=True, exist_ok=True)

            base_name = f"{planner_name}_{scene_name}_{query.query_id}_iter{iter_budget}"
            plot_configuration_space(
                cdf=cdf,
                obj_list=planner.obj_list,
                nodes=nodes,
                start_q=start_q,
                config_goal_marker=config_goal_marker,
                path=path_q,
                title=f"{scene_name} | {planner_name} | {query.query_id} | {event_type} @ {iter_budget}",
                file_path=plot_dir / f"cspace_{base_name}.png",
            )
            plot_task_space(
                cdf=cdf,
                obj_list=planner.obj_list,
                start_q=start_q,
                goal_task=goal_task,
                path_q=path_q,
                title=f"{scene_name} | {planner_name} | {query.query_id} | {event_type} @ {iter_budget}",
                file_path=plot_dir / f"task_{base_name}.png",
            )
            plot_ee_goal_cdf_snapshot(
                cdf=cdf,
                start_q=start_q,
                goal_task=goal_task,
                path_q=path_q,
                ik_solutions=ik_solutions,
                title=f"{scene_name} | {planner_name} | {query.query_id} | EE Goal CDF @ {iter_budget}",
                file_path=plot_dir / f"ee_cdf_{base_name}.png",
            )
            plt.close("all")

    return rows


def write_detailed_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "planner",
        "scene",
        "query_id",
        "seed",
        "event_type",
        "iteration_budget",
        "path_found",
        "first_path_iteration",
        "nodes_total",
        "accepted_nodes",
        "discarded_nodes",
        "rewires",
        "rejection_rate",
        "path_length",
        "path_waypoints",
        "final_path_cost",
        "ee_goal_error",
        "planning_time_sec",
        "config_goal_q1",
        "config_goal_q2",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
            r
            for r in rows
            if r["planner"] == planner_name
            and int(r["iteration_budget"]) == int(cfg.max_iters)
            and r["event_type"] in ("checkpoint", "checkpoint_first_path")
        ]

        summary_rows: List[Dict[str, Any]] = []
        scenes = sorted({r["scene"] for r in planner_rows})
        for scene in scenes:
            scene_rows = [r for r in planner_rows if r["scene"] == scene]
            num_queries = len(scene_rows)
            success_values = [int(r["path_found"]) for r in scene_rows]
            success_rate = float(np.mean(success_values)) if num_queries > 0 else float("nan")

            planning_time = [float(r["planning_time_sec"]) for r in scene_rows]
            path_lengths = [float(r["path_length"]) for r in scene_rows]
            nodes_total = [float(r["nodes_total"]) for r in scene_rows]
            discarded = [float(r["discarded_nodes"]) for r in scene_rows]
            ee_errors = [float(r["ee_goal_error"]) for r in scene_rows]

            first_path_iters_success = [
                float(r["first_path_iteration"])
                for r in scene_rows
                if int(r["path_found"]) == 1 and str(r["first_path_iteration"]) != ""
            ]

            summary_rows.append(
                {
                    "planner": planner_name,
                    "scene": scene,
                    "num_queries": num_queries,
                    "success_rate": success_rate,
                    "planning_time_mean": _nanmean(planning_time),
                    "planning_time_std": _nanstd(planning_time),
                    "path_length_mean": _nanmean(path_lengths),
                    "path_length_std": _nanstd(path_lengths),
                    "nodes_total_mean": _nanmean(nodes_total),
                    "nodes_total_std": _nanstd(nodes_total),
                    "discarded_nodes_mean": _nanmean(discarded),
                    "discarded_nodes_std": _nanstd(discarded),
                    "ee_goal_error_mean": _nanmean(ee_errors),
                    "ee_goal_error_std": _nanstd(ee_errors),
                    "first_path_iteration_mean_success_only": _nanmean(first_path_iters_success),
                }
            )

        summary_path = run_dir / f"summary_{planner_name}.csv"
        fieldnames = [
            "planner",
            "scene",
            "num_queries",
            "success_rate",
            "planning_time_mean",
            "planning_time_std",
            "path_length_mean",
            "path_length_std",
            "nodes_total_mean",
            "nodes_total_std",
            "discarded_nodes_mean",
            "discarded_nodes_std",
            "ee_goal_error_mean",
            "ee_goal_error_std",
            "first_path_iteration_mean_success_only",
        ]
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)


def main(cfg: EvalConfig = CONFIG) -> None:
    if EE_IMPORT_ERROR is not None:
        raise SystemExit(
            "Failed to import planners from ee_goal_rrtstar_2d.py. "
            "This environment is missing a required dependency (commonly 'tqdm' via nn_cdf).\n"
            f"Original import error: {EE_IMPORT_ERROR}"
        )

    unknown_planners = [p for p in cfg.planners if p not in {"rrt", "cdf", "pullandslide"}]
    if unknown_planners:
        raise SystemExit(f"Unknown planner names in config: {unknown_planners}")

    missing_scene_queries = [s for s in cfg.scenes if s not in cfg.scene_queries]
    if missing_scene_queries:
        raise SystemExit(f"Missing scene query sets for scenes: {missing_scene_queries}")

    checkpoints = build_checkpoints(cfg.log_start_iter, cfg.max_iters)
    run_dir = ensure_run_dir(cfg)
    all_rows: List[Dict[str, Any]] = []

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=== EE-Goal Planner Evaluation ===")
    print(f"run_dir: {run_dir}")
    print(f"planners: {cfg.planners}")
    print(f"scenes: {cfg.scenes}")
    print(f"checkpoints: {checkpoints}")
    print(f"seed: {cfg.seed}, max_iters: {cfg.max_iters}")

    t0 = time.time()
    for scene_name in cfg.scenes:
        cdf = CDF2D(device)
        obj_list = make_scene(scene_name, device)
        q_min = cdf.q_min.detach().cpu().numpy().astype(np.float32)
        q_max = cdf.q_max.detach().cpu().numpy().astype(np.float32)

        queries = cfg.scene_queries[scene_name]
        for query in queries:
            start_q = np.asarray(query.start_q, dtype=np.float32)
            if not (np.all(start_q >= q_min) and np.all(start_q <= q_max)):
                print(f"[WARN] Skipping {scene_name}/{query.query_id}: start out of bounds.")
                continue

            for planner_name in cfg.planners:
                print(f"[RUN] planner={planner_name} scene={scene_name} query={query.query_id}")
                planner = planner_factory(
                    planner_name=planner_name,
                    cdf=cdf,
                    obj_list=obj_list,
                    q_min=q_min,
                    q_max=q_max,
                    cfg=cfg,
                )
                rows = run_case_with_cache(
                    planner_name=planner_name,
                    scene_name=scene_name,
                    query=query,
                    planner=planner,
                    cdf=cdf,
                    checkpoints=checkpoints,
                    cfg=cfg,
                    run_dir=run_dir,
                )
                all_rows.extend(rows)

    detailed_path = run_dir / "detailed_log.csv"
    write_detailed_csv(all_rows, detailed_path)
    write_planner_summaries(all_rows, cfg, run_dir)

    print(f"\nWrote detailed CSV: {detailed_path}")
    for planner_name in cfg.planners:
        print(f"Wrote summary CSV: {run_dir / f'summary_{planner_name}.csv'}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Elapsed sec: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
