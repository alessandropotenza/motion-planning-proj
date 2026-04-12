#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Bar plots from eval_ee_goal_rrtstar detailed_log.csv (checkpoint rows):
#   (1) mean iterations to first feasible path (first_path_iteration)
#   (2) mean final path cost at the checkpoint (final_path_cost)
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = "/home/joshua/Packages/mp_proj2/2Dexamples/outputs/eval_ee_goal_rrtstar/run_20260412_052039/detailed_log.csv"

PLANNERS: Tuple[str, ...] = ("rrt", "cdf", "pullandslide")
DISPLAY_LABELS = {
    "rrt": "Vanilla EE-RRT*",
    "cdf": "CDF EE-RRT*",
    "pullandslide": "Pull-and-slide",
}


def _natural_scene_key(scene: str) -> Tuple:
    """Sort scene_2 before scene_10."""
    parts = re.split(r"(\d+)", scene)
    key: List = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def parse_optional_float(value: str | None) -> float:
    """Parse optional numeric CSV field; missing / invalid → NaN."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_checkpoint_rows(csv_path: Path, iteration_budget: int) -> List[dict]:
    rows: List[dict] = []
    target = int(iteration_budget)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("event_type") or "").strip() != "checkpoint":
                continue
            try:
                it = int(float(r["iteration_budget"]))
            except (KeyError, TypeError, ValueError):
                continue
            if it != target:
                continue
            rows.append(r)
    return rows


def list_available_checkpoints(csv_path: Path) -> List[int]:
    seen: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("event_type") or "").strip() != "checkpoint":
                continue
            try:
                it = int(float(r["iteration_budget"]))
            except (KeyError, TypeError, ValueError):
                continue
            seen.add(it)
    return sorted(seen)


def mean_field_by_scene_planner(
    rows: Iterable[dict],
    field: str,
) -> Dict[Tuple[str, str], float]:
    """Mean of `field` over queries for each (scene, planner)."""
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        scene = (r.get("scene") or "").strip()
        planner = (r.get("planner") or "").strip().lower()
        if not scene or not planner:
            continue
        buckets[(scene, planner)].append(parse_optional_float(r.get(field)))
    out: Dict[Tuple[str, str], float] = {}
    for key, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        if not np.any(np.isfinite(arr)):
            out[key] = float("nan")
        else:
            out[key] = float(np.nanmean(arr))
    return out


ACROSS_SCENES_LABEL = "Mean (scenes)"


def build_bar_matrix(
    scenes: Sequence[str], means: Dict[Tuple[str, str], float]
) -> Tuple[np.ndarray, List[str]]:
    """
    Rows: one per scene, then one row that averages each planner's metric across scenes.
    Columns: one per planner (grouped bars at each x).
    """
    n = len(scenes)
    n_planners = len(PLANNERS)
    vals = np.full((n + 1, n_planners), np.nan, dtype=float)
    for i, sc in enumerate(scenes):
        for j, pl in enumerate(PLANNERS):
            vals[i, j] = means.get((sc, pl), float("nan"))
    for j in range(n_planners):
        col = vals[:n, j]
        if not np.any(np.isfinite(col)):
            vals[n, j] = float("nan")
        else:
            vals[n, j] = float(np.nanmean(col))
    x_labels = list(scenes) + [ACROSS_SCENES_LABEL]
    return vals, x_labels


def plot_grouped_planner_bars(
    x_labels: Sequence[str],
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    out_path: Path | None,
) -> None:
    n_groups = len(x_labels)
    x = np.arange(n_groups, dtype=float)
    n_bars = len(PLANNERS)
    width = 0.22
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    labels = [DISPLAY_LABELS[p] for p in PLANNERS]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * n_groups + 4), 5.0))
    cmap = plt.cm.get_cmap("tab10", n_bars)

    for b in range(n_bars):
        heights = values[:, b].copy()
        plot_heights = np.nan_to_num(heights, nan=0.0)
        bars = ax.bar(
            x + offsets[b],
            plot_heights,
            width,
            label=labels[b],
            color=cmap(b),
            edgecolor="black",
            linewidth=0.4,
        )
        for rect, h_orig in zip(bars, heights):
            if not np.isfinite(h_orig):
                rect.set_hatch("//")
                rect.set_facecolor("0.85")
                rect.set_edgecolor("0.4")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scene")
    ax.set_xticks(x)
    ax.set_xticklabels(list(x_labels), rotation=0)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bar charts from detailed_log.csv: iterations to first feasibility "
        "and final path cost (checkpoint rows only)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to detailed_log.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--checkpoint",
        "--iteration-budget",
        type=int,
        dest="checkpoint",
        metavar="N",
        default=None,
        help="Iteration budget (must match a checkpoint row in the log). "
        "If omitted, uses the largest checkpoint present in the CSV.",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="Print available iteration_budget values from checkpoint rows and exit.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: same directory as the CSV).",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv.expanduser().resolve()

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.list_checkpoints:
        avail = list_available_checkpoints(csv_path)
        print("Available checkpoint iteration budgets:", ", ".join(map(str, avail)))
        return

    checkpoint = args.checkpoint
    avail = list_available_checkpoints(csv_path)
    if checkpoint is None:
        if not avail:
            raise SystemExit("No checkpoint rows found in CSV.")
        checkpoint = avail[-1]

    rows = load_checkpoint_rows(csv_path, checkpoint)
    if not rows:
        raise SystemExit(
            f"No checkpoint rows for iteration_budget={checkpoint}. "
            f"Available: {avail}"
        )

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = csv_path.parent
    else:
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    suffix = csv_path.parent.name

    means_iter = mean_field_by_scene_planner(rows, "first_path_iteration")
    means_cost = mean_field_by_scene_planner(rows, "final_path_cost")
    scenes = sorted(
        {k[0] for k in means_iter} | {k[0] for k in means_cost},
        key=_natural_scene_key,
    )
    values_iter, x_labels = build_bar_matrix(scenes, means_iter)
    path_feasible = out_dir / f"first_feasible_iter_checkpoint_{checkpoint}.png"
    plot_grouped_planner_bars(
        x_labels,
        values_iter,
        ylabel="Mean iterations to first feasible path",
        title=f"Iterations to first feasibility — checkpoint {checkpoint} — {suffix}",
        out_path=path_feasible,
    )

    values_cost, x_labels_c = build_bar_matrix(scenes, means_cost)
    path_cost = out_dir / f"final_path_cost_checkpoint_{checkpoint}.png"
    plot_grouped_planner_bars(
        x_labels_c,
        values_cost,
        ylabel="Mean final path cost (at checkpoint)",
        title=f"Best path cost at checkpoint — {checkpoint} iters — {suffix}",
        out_path=path_cost,
    )

    print(f"Wrote {path_feasible}")
    print(f"Wrote {path_cost}")


if __name__ == "__main__":
    main()
