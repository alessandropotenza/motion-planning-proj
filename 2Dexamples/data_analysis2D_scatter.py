#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Two plot types per scene (checkpoint + first-path rows):
#   - planning_time_sec vs final_path_cost
#   - iteration_budget vs final_path_cost
# Solid lines connect points in sorted-x order; scatter markers on each point.
# Color = planner, marker = query.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = "/home/joshua/Packages/mp_proj2/2Dexamples/outputs/eval_ee_goal_rrtstar/run_20260412_052039/detailed_log.csv"

# Rows to include (matches eval_ee_goal_rrtstar event types).
EVENT_TYPES = frozenset({"checkpoint", "checkpoint_first_path", "first_path"})

PLANNERS_ORDER: Tuple[str, ...] = ("rrt", "cdf", "pullandslide")
DISPLAY_LABELS: Dict[str, str] = {
    "rrt": "Vanilla EE-RRT*",
    "cdf": "CDF EE-RRT*",
    "pullandslide": "Pull-and-slide",
}

# Distinct markers per query_id so multiple benchmarks stay visible.
QUERY_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


def _natural_scene_key(scene: str) -> Tuple:
    parts = re.split(r"(\d+)", scene)
    key: List = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def parse_optional_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def x_value_from_row(row: dict, x_field: str) -> float:
    """Numeric x for plotting (time or iteration budget)."""
    if x_field == "iteration_budget":
        return parse_optional_float(row.get("iteration_budget"))
    return parse_optional_float(row.get(x_field))


def load_rows(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            et = (r.get("event_type") or "").strip()
            if et not in EVENT_TYPES:
                continue
            out.append(r)
    return out


def scenes_in_rows(rows: Sequence[dict]) -> List[str]:
    s = {(r.get("scene") or "").strip() for r in rows if (r.get("scene") or "").strip()}
    return sorted(s, key=_natural_scene_key)


def planners_in_rows(rows: Sequence[dict]) -> List[str]:
    seen = {(r.get("planner") or "").strip().lower() for r in rows if (r.get("planner") or "").strip()}
    return [p for p in PLANNERS_ORDER if p in seen] + sorted(seen - set(PLANNERS_ORDER))


def query_ids_in_scene(rows: Sequence[dict], scene: str) -> List[str]:
    q = {(r.get("query_id") or "").strip() for r in rows if (r.get("scene") or "").strip() == scene}
    return sorted(q)


def plot_scene_x_vs_path_cost(
    rows: Sequence[dict],
    scene: str,
    planners: Sequence[str],
    out_path: Path,
    *,
    x_field: str,
    xlabel: str,
    title_xy: str,
    title_suffix: str = "",
) -> None:
    scene_rows = [r for r in rows if (r.get("scene") or "").strip() == scene]
    queries = query_ids_in_scene(rows, scene)
    q_to_m = {q: QUERY_MARKERS[i % len(QUERY_MARKERS)] for i, q in enumerate(queries)}

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    cmap = plt.cm.get_cmap("tab10", max(len(planners), 1))

    for pi, planner in enumerate(planners):
        pr = [r for r in scene_rows if (r.get("planner") or "").strip().lower() == planner]
        color = cmap(pi)
        for qid in queries:
            xs: List[float] = []
            cs: List[float] = []
            for r in pr:
                if ((r.get("query_id") or "").strip() or "default") != qid:
                    continue
                x = x_value_from_row(r, x_field)
                c = parse_optional_float(r.get("final_path_cost"))
                if not np.isfinite(x) or not np.isfinite(c):
                    continue
                xs.append(x)
                cs.append(c)
            if not xs:
                continue
            order = np.argsort(xs, kind="mergesort")
            xs_arr = np.asarray(xs, dtype=float)[order]
            cs_arr = np.asarray(cs, dtype=float)[order]
            if xs_arr.size >= 2:
                ax.plot(
                    xs_arr,
                    cs_arr,
                    "-",
                    color=color,
                    linewidth=1.65,
                    solid_capstyle="round",
                    alpha=0.95,
                    zorder=1,
                )
            mk = q_to_m.get(qid, "o")
            ax.scatter(
                xs_arr,
                cs_arr,
                s=42,
                color=color,
                marker=mk,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.85,
                zorder=2,
            )

        # One legend entry per planner (proxy artist).
        ax.scatter([], [], color=color, s=42, edgecolors="black", linewidths=0.35, label=DISPLAY_LABELS.get(planner, planner))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Path cost (final_path_cost)")
    ttl = f"{scene}: {title_xy}"
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    ax.set_title(ttl)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="best", fontsize=9)

    if len(queries) > 1:
        mq_lines = [f"  {q}: {q_to_m[q]}" for q in queries]
        fig.text(0.02, 0.02, "Markers: query\n" + "\n".join(mq_lines), fontsize=7, va="bottom", family="monospace")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-scene lines + markers: time vs path cost and iterations vs path cost "
        "(checkpoints + first-path rows)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to detailed_log.csv",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files (default: same folder as CSV).",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No checkpoint / first-path rows found in CSV.")

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = scenes_in_rows(rows)
    planners = planners_in_rows(rows)
    suffix = csv_path.parent.name

    for scene in scenes:
        safe = scene.replace("/", "_")
        out_time = out_dir / f"time_vs_pathcost_{safe}.png"
        plot_scene_x_vs_path_cost(
            rows,
            scene,
            planners,
            out_time,
            x_field="planning_time_sec",
            xlabel="Planning time (s)",
            title_xy="time vs path cost",
            title_suffix=suffix,
        )
        print(f"Wrote {out_time}")
        out_iter = out_dir / f"iter_vs_pathcost_{safe}.png"
        plot_scene_x_vs_path_cost(
            rows,
            scene,
            planners,
            out_iter,
            x_field="iteration_budget",
            xlabel="Iteration budget",
            title_xy="iterations vs path cost",
            title_suffix=suffix,
        )
        print(f"Wrote {out_iter}")


if __name__ == "__main__":
    main()
