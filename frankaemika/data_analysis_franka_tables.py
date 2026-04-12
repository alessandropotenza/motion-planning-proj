#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Tabular summaries from eval_ee_goal_franka detailed_log.csv.
# Per-checkpoint tables, single-checkpoint slices, first-path events, and
# optional LaTeX snippets (booktabs).
# Franka (7-DOF) counterpart of 2Dexamples/data_analysis2D_tables.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CHECKPOINT_EVENTS = frozenset({"checkpoint", "checkpoint_first_path"})
SCATTER_EVENTS = frozenset({"checkpoint", "checkpoint_first_path", "first_path"})

PLANNERS_ORDER: Tuple[str, ...] = ("vanilla", "cdf", "pullandslide")

PLANNER_LATEX_SHORT: Dict[str, str] = {
    "vanilla": "Vanilla",
    "cdf": "CDF",
    "pullandslide": "Pull-and-slide",
}

AGG_FLOAT_FIELDS: Tuple[str, ...] = (
    "planning_time_sec",
    "path_length",
    "final_path_cost",
    "ee_goal_error",
    "nodes_total",
    "accepted_nodes",
    "discarded_nodes",
    "rewires",
    "rejection_rate",
    "path_waypoints",
    "first_path_iteration",
) + tuple(f"config_goal_q{i + 1}" for i in range(7))


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


def parse_path_found(value: str | None) -> int:
    v = parse_optional_float(value)
    return int(v) if np.isfinite(v) else 0


def list_available_checkpoints(csv_path: Path) -> List[int]:
    seen: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("event_type") or "").strip() != "checkpoint":
                continue
            try:
                it = int(float(r["iteration_budget"]))
            except (KeyError, TypeError, ValueError):
                continue
            seen.add(it)
    return sorted(seen)


def load_all_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_at_iteration_budget(
    rows: Iterable[dict], iteration_budget: int, event_types: frozenset[str],
) -> List[dict]:
    target = int(iteration_budget)
    out: List[dict] = []
    for r in rows:
        et = (r.get("event_type") or "").strip()
        if et not in event_types:
            continue
        try:
            it = int(float(r["iteration_budget"]))
        except (KeyError, TypeError, ValueError):
            continue
        if it != target:
            continue
        out.append(r)
    return out


def rows_event_first_path(rows: Iterable[dict]) -> List[dict]:
    return [r for r in rows if (r.get("event_type") or "").strip() == "first_path"]


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


def group_key_scene_planner(rows: Sequence[dict]) -> Dict[Tuple[str, str], List[dict]]:
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        scene = (r.get("scene") or "").strip()
        planner = (r.get("planner") or "").strip().lower()
        if not scene or not planner:
            continue
        buckets[(scene, planner)].append(r)
    return dict(buckets)


def aggregate_rows(group_rows: List[dict]) -> Dict[str, Any]:
    n = len(group_rows)
    pf = [parse_path_found(r.get("path_found")) for r in group_rows]
    success_rate = float(np.mean(pf)) if n else float("nan")

    first_path_success = [
        parse_optional_float(r.get("first_path_iteration"))
        for r in group_rows
        if parse_path_found(r.get("path_found")) == 1
    ]

    row: Dict[str, Any] = {
        "num_queries": n,
        "success_rate": success_rate,
        "first_path_iteration_mean_success_only": _nanmean(first_path_success),
    }
    for field in AGG_FLOAT_FIELDS:
        vals = [parse_optional_float(r.get(field)) for r in group_rows]
        row[f"{field}_mean"] = _nanmean(vals)
        row[f"{field}_std"] = _nanstd(vals)

    err_n = sum(1 for r in group_rows if str(r.get("error") or "").strip())
    row["num_nonempty_errors"] = err_n
    return row


def sort_planners(planners: Iterable[str]) -> List[str]:
    seen = {p.lower() for p in planners}
    return [p for p in PLANNERS_ORDER if p in seen] + sorted(seen - set(PLANNERS_ORDER))


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def latex_escape(text: str) -> str:
    out: List[str] = []
    for ch in str(text):
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append(f"\\{ch}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def fmt_latex_float(x: float, *, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "---"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 100:
        return f"{x:.{max(1, nd - 1)}f}"
    if ax >= 10:
        return f"{x:.{nd}f}"
    if ax >= 1:
        return f"{x:.{nd}f}"
    return f"{x:.{nd + 1}g}"


def fmt_latex_success(rate: float) -> str:
    if not np.isfinite(rate):
        return "---"
    return f"{100.0 * rate:.0f}\\%"


def planner_latex_name(planner: str) -> str:
    return latex_escape(PLANNER_LATEX_SHORT.get(planner.lower(), planner))


def iter_scene_planner_aggregates(rows_at_budget: List[dict]) -> List[Tuple[str, str, Dict[str, Any]]]:
    buckets = group_key_scene_planner(rows_at_budget)
    scenes = sorted({k[0] for k in buckets}, key=_natural_scene_key)
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for scene in scenes:
        for planner in sort_planners({k[1] for k in buckets if k[0] == scene}):
            grp = buckets.get((scene, planner), [])
            if grp:
                out.append((scene, planner, aggregate_rows(grp)))
    return out


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

def write_latex_snippets(
    rows_at_budget: List[dict],
    iteration_budget: int,
    out_dir: Path,
    tag: str,
) -> List[Path]:
    rows_data = iter_scene_planner_aggregates(rows_at_budget)
    by_planner: Dict[str, List[dict]] = defaultdict(list)
    for r in rows_at_budget:
        pl = (r.get("planner") or "").strip().lower()
        if pl:
            by_planner[pl].append(r)

    written: List[Path] = []
    hdr = (
        "% Tables generated by data_analysis_franka_tables.py\n"
        f"% iteration\\_budget = {iteration_budget}\n"
        "% Requires: \\usepackage{booktabs}\n"
    )

    def emit(lines: List[str]) -> str:
        return hdr + "\n".join(lines) + "\n"

    # Table 1: outcomes
    t1 = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Scene & Planner & Success & Path cost \\",
        r"\midrule",
    ]
    for scene, planner, agg in rows_data:
        t1.append(
            f"{latex_escape(scene)} & {planner_latex_name(planner)} & "
            f"{fmt_latex_success(agg['success_rate'])} & "
            f"{fmt_latex_float(agg['final_path_cost_mean'])} \\\\"
        )
    t1.extend([r"\bottomrule", r"\end{tabular}"])
    p1 = out_dir / f"latex_scene_planner_outcomes_{tag}.tex"
    p1.write_text(emit(t1), encoding="utf-8")
    written.append(p1)

    # Table 2: timing
    t2 = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Scene & Planner & Time (s) & EE error (m) \\",
        r"\midrule",
    ]
    for scene, planner, agg in rows_data:
        t2.append(
            f"{latex_escape(scene)} & {planner_latex_name(planner)} & "
            f"{fmt_latex_float(agg['planning_time_sec_mean'])} & "
            f"{fmt_latex_float(agg['ee_goal_error_mean'])} \\\\"
        )
    t2.extend([r"\bottomrule", r"\end{tabular}"])
    p2 = out_dir / f"latex_scene_planner_timing_{tag}.tex"
    p2.write_text(emit(t2), encoding="utf-8")
    written.append(p2)

    # Table 3: tree size
    t3 = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Scene & Planner & Nodes & Rewires \\",
        r"\midrule",
    ]
    for scene, planner, agg in rows_data:
        t3.append(
            f"{latex_escape(scene)} & {planner_latex_name(planner)} & "
            f"{fmt_latex_float(agg['nodes_total_mean'])} & "
            f"{fmt_latex_float(agg['rewires_mean'])} \\\\"
        )
    t3.extend([r"\bottomrule", r"\end{tabular}"])
    p3 = out_dir / f"latex_scene_planner_tree_{tag}.tex"
    p3.write_text(emit(t3), encoding="utf-8")
    written.append(p3)

    # Table 4: global planner summary
    t4 = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Planner & Success & Path cost & Time (s) \\",
        r"\midrule",
    ]
    for planner in sort_planners(by_planner.keys()):
        grp = by_planner[planner]
        agg = aggregate_rows(grp)
        t4.append(
            f"{planner_latex_name(planner)} & "
            f"{fmt_latex_success(agg['success_rate'])} & "
            f"{fmt_latex_float(agg['final_path_cost_mean'])} & "
            f"{fmt_latex_float(agg['planning_time_sec_mean'])} \\\\"
        )
    t4.extend([r"\bottomrule", r"\end{tabular}"])
    p4 = out_dir / f"latex_planner_global_{tag}.tex"
    p4.write_text(emit(t4), encoding="utf-8")
    written.append(p4)

    return written


def write_latex_pathcost_by_checkpoint(
    all_rows: List[dict],
    checkpoints: Sequence[int],
    out_dir: Path,
    csv_path: Path,
) -> List[Path]:
    scenes: set[str] = set()
    planners_seen: set[str] = set()
    for r in all_rows:
        et = (r.get("event_type") or "").strip()
        if et not in CHECKPOINT_EVENTS:
            continue
        sc = (r.get("scene") or "").strip()
        pl = (r.get("planner") or "").strip().lower()
        if sc:
            scenes.add(sc)
        if pl:
            planners_seen.add(pl)
    scenes_sorted = sorted(scenes, key=_natural_scene_key)
    planners = sort_planners(planners_seen)
    if not scenes_sorted or not planners:
        return []

    hdr = (
        "% Path cost by checkpoint — generated by data_analysis_franka_tables.py\n"
        f"% Source: {latex_escape(csv_path.name)}\n"
        "% Requires: \\usepackage{booktabs}\n"
    )

    written: List[Path] = []
    for scene in scenes_sorted:
        col_spec = "r" * len(planners)
        col_headers = " & ".join(planner_latex_name(p) for p in planners)
        lines = [
            f"\\begin{{tabular}}{{@{{}}r{col_spec}@{{}}}}",
            r"\toprule",
            f"Iters & {col_headers} \\\\",
            r"\midrule",
        ]
        for cp in checkpoints:
            cp_rows = [
                r for r in all_rows
                if (r.get("event_type") or "").strip() in CHECKPOINT_EVENTS
                and (r.get("scene") or "").strip() == scene
                and int(float(r.get("iteration_budget", "0"))) == cp
            ]
            cells: List[str] = []
            for pl in planners:
                vals = [
                    parse_optional_float(r.get("final_path_cost"))
                    for r in cp_rows
                    if (r.get("planner") or "").strip().lower() == pl
                ]
                arr = np.asarray(vals, dtype=float)
                if arr.size == 0 or not np.any(np.isfinite(arr)):
                    cells.append("---")
                else:
                    cells.append(fmt_latex_float(float(np.nanmean(arr))))
            lines.append(f"{cp} & {' & '.join(cells)} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        safe = scene.replace("/", "_")
        out_path = out_dir / f"latex_pathcost_checkpoints_{safe}.tex"
        out_path.write_text(
            hdr + f"% Scene: {latex_escape(scene)}\n" + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# CSV table writers
# ---------------------------------------------------------------------------

def _agg_fieldnames() -> List[str]:
    fieldnames = [
        "iteration_budget", "planner", "scene",
        "num_queries", "success_rate",
        "first_path_iteration_mean_success_only", "num_nonempty_errors",
    ]
    for f in AGG_FLOAT_FIELDS:
        fieldnames.append(f"{f}_mean")
        fieldnames.append(f"{f}_std")
    return fieldnames


def write_scene_planner_table(rows_at_budget: List[dict], iteration_budget: int, out_path: Path) -> None:
    buckets = group_key_scene_planner(rows_at_budget)
    scenes = sorted({k[0] for k in buckets}, key=_natural_scene_key)
    fieldnames = _agg_fieldnames()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for scene in scenes:
            for planner in sort_planners({k[1] for k in buckets if k[0] == scene}):
                grp = buckets.get((scene, planner), [])
                if not grp:
                    continue
                agg = aggregate_rows(grp)
                w.writerow({"iteration_budget": iteration_budget, "planner": planner, "scene": scene, **agg})


def write_per_query_table(rows_at_budget: List[dict], iteration_budget: int, out_path: Path) -> None:
    fieldnames = [
        "iteration_budget", "planner", "scene", "query_id", "seed", "event_type",
        "path_found", "first_path_iteration", "planning_time_sec",
        "path_length", "path_waypoints", "final_path_cost", "ee_goal_error",
        "nodes_total", "accepted_nodes", "discarded_nodes", "rewires", "rejection_rate",
    ] + [f"config_goal_q{i + 1}" for i in range(7)] + ["error"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows_at_budget,
        key=lambda r: (
            _natural_scene_key((r.get("scene") or "").strip()),
            (r.get("planner") or "").strip().lower(),
            (r.get("query_id") or "").strip(),
        ),
    )
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            rec = {k: r.get(k, "") for k in fieldnames}
            rec["iteration_budget"] = iteration_budget
            w.writerow(rec)


def write_first_path_table(rows: List[dict], out_path: Path) -> None:
    fp_rows = rows_event_first_path(rows)
    fieldnames = [
        "planner", "scene", "query_id", "seed", "iteration_budget",
        "path_found", "first_path_iteration", "planning_time_sec",
        "path_length", "path_waypoints", "final_path_cost", "ee_goal_error",
        "nodes_total", "accepted_nodes", "discarded_nodes", "rewires", "rejection_rate",
    ] + [f"config_goal_q{i + 1}" for i in range(7)] + ["error"]
    fp_rows.sort(
        key=lambda r: (
            _natural_scene_key((r.get("scene") or "").strip()),
            (r.get("planner") or "").strip().lower(),
            (r.get("query_id") or "").strip(),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in fp_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_planner_global_table(rows_at_budget: List[dict], iteration_budget: int, out_path: Path) -> None:
    by_planner: Dict[str, List[dict]] = defaultdict(list)
    for r in rows_at_budget:
        pl = (r.get("planner") or "").strip().lower()
        if pl:
            by_planner[pl].append(r)
    fieldnames = [
        "iteration_budget", "planner", "num_queries", "num_scenes",
        "success_rate", "first_path_iteration_mean_success_only", "num_nonempty_errors",
    ]
    for f in AGG_FLOAT_FIELDS:
        fieldnames.append(f"{f}_mean")
        fieldnames.append(f"{f}_std")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for planner in sort_planners(by_planner.keys()):
            grp = by_planner[planner]
            scenes = {(r.get("scene") or "").strip() for r in grp if (r.get("scene") or "").strip()}
            agg = aggregate_rows(grp)
            w.writerow({
                "iteration_budget": iteration_budget, "planner": planner,
                "num_scenes": len(scenes), **agg,
            })


# ---------------------------------------------------------------------------
# All-checkpoint CSV writers
# ---------------------------------------------------------------------------

def write_meta_all_checkpoints(csv_path: Path, checkpoints: List[int], n_log_rows: int, out_path: Path) -> None:
    fieldnames = ["source_csv", "checkpoint_iteration_budgets", "n_checkpoints", "n_rows_in_log"]
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "source_csv": str(csv_path),
            "checkpoint_iteration_budgets": " ".join(map(str, checkpoints)),
            "n_checkpoints": len(checkpoints),
            "n_rows_in_log": n_log_rows,
        })


def write_scene_planner_all_checkpoints(all_rows: List[dict], checkpoints: Sequence[int], out_path: Path) -> None:
    fieldnames = _agg_fieldnames()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for cp in checkpoints:
            slice_rows = rows_at_iteration_budget(all_rows, cp, CHECKPOINT_EVENTS)
            if not slice_rows:
                continue
            buckets = group_key_scene_planner(slice_rows)
            scenes = sorted({k[0] for k in buckets}, key=_natural_scene_key)
            for scene in scenes:
                for planner in sort_planners({k[1] for k in buckets if k[0] == scene}):
                    grp = buckets.get((scene, planner), [])
                    if grp:
                        agg = aggregate_rows(grp)
                        w.writerow({"iteration_budget": cp, "planner": planner, "scene": scene, **agg})


def write_per_query_all_checkpoints(all_rows: List[dict], checkpoints: Sequence[int], out_path: Path) -> None:
    fieldnames = [
        "iteration_budget", "planner", "scene", "query_id", "seed", "event_type",
        "path_found", "first_path_iteration", "planning_time_sec",
        "path_length", "path_waypoints", "final_path_cost", "ee_goal_error",
        "nodes_total", "accepted_nodes", "discarded_nodes", "rewires", "rejection_rate",
    ] + [f"config_goal_q{i + 1}" for i in range(7)] + ["error"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for cp in checkpoints:
            slice_rows = rows_at_iteration_budget(all_rows, cp, CHECKPOINT_EVENTS)
            rows_sorted = sorted(
                slice_rows,
                key=lambda r: (
                    _natural_scene_key((r.get("scene") or "").strip()),
                    (r.get("planner") or "").strip().lower(),
                    (r.get("query_id") or "").strip(),
                ),
            )
            for r in rows_sorted:
                rec = {k: r.get(k, "") for k in fieldnames}
                rec["iteration_budget"] = cp
                w.writerow(rec)


def write_planner_global_all_checkpoints(all_rows: List[dict], checkpoints: Sequence[int], out_path: Path) -> None:
    fieldnames = [
        "iteration_budget", "planner", "num_queries", "num_scenes",
        "success_rate", "first_path_iteration_mean_success_only", "num_nonempty_errors",
    ]
    for f in AGG_FLOAT_FIELDS:
        fieldnames.append(f"{f}_mean")
        fieldnames.append(f"{f}_std")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for cp in checkpoints:
            slice_rows = rows_at_iteration_budget(all_rows, cp, CHECKPOINT_EVENTS)
            if not slice_rows:
                continue
            by_planner: Dict[str, List[dict]] = defaultdict(list)
            for r in slice_rows:
                pl = (r.get("planner") or "").strip().lower()
                if pl:
                    by_planner[pl].append(r)
            for planner in sort_planners(by_planner.keys()):
                grp = by_planner[planner]
                scenes = {(r.get("scene") or "").strip() for r in grp if (r.get("scene") or "").strip()}
                agg = aggregate_rows(grp)
                w.writerow({
                    "iteration_budget": cp, "planner": planner,
                    "num_scenes": len(scenes), **agg,
                })


def write_meta_row(csv_path: Path, iteration_budget: int, all_rows: int, out_path: Path) -> None:
    fieldnames = ["source_csv", "iteration_budget_rows", "n_rows_in_log"]
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({
            "source_csv": str(csv_path),
            "iteration_budget_rows": iteration_budget,
            "n_rows_in_log": all_rows,
        })


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export metric tables (CSV + optional LaTeX) from Franka eval detailed_log.csv."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to detailed_log.csv")
    parser.add_argument(
        "--checkpoint", "--iteration-budget",
        type=int, dest="checkpoint", default=None, metavar="N",
        help="Aggregate checkpoint rows at this budget (default: largest checkpoint).",
    )
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="Print checkpoint iteration_budget values and exit.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: directory of input CSV).")
    parser.add_argument("--no-all-checkpoints", action="store_true",
                        help="Skip writing *_all_checkpoints.csv files.")
    parser.add_argument("--latex", action="store_true",
                        help="Write LaTeX snippets (.tex, booktabs).")
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    avail = list_available_checkpoints(csv_path)
    if args.list_checkpoints:
        print("Checkpoint iteration budgets:", ", ".join(map(str, avail)))
        return

    checkpoint = args.checkpoint
    if checkpoint is None:
        if not avail:
            raise SystemExit("No checkpoint rows found in CSV.")
        checkpoint = avail[-1]

    all_rows = load_all_rows(csv_path)
    rows_final = rows_at_iteration_budget(all_rows, checkpoint, CHECKPOINT_EVENTS)
    if not rows_final:
        raise SystemExit(
            f"No rows for iteration_budget={checkpoint} with event in {sorted(CHECKPOINT_EVENTS)}. "
            f"Available checkpoints: {avail}"
        )

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"checkpoint_{checkpoint}"

    if not args.no_all_checkpoints:
        write_meta_all_checkpoints(csv_path, avail, len(all_rows), out_dir / "metrics_meta_all_checkpoints.csv")
        write_scene_planner_all_checkpoints(all_rows, avail, out_dir / "metrics_by_scene_planner_all_checkpoints.csv")
        write_per_query_all_checkpoints(all_rows, avail, out_dir / "metrics_per_query_all_checkpoints.csv")
        write_planner_global_all_checkpoints(all_rows, avail, out_dir / "metrics_by_planner_global_all_checkpoints.csv")

    write_meta_row(csv_path, checkpoint, len(all_rows), out_dir / f"metrics_meta_{tag}.csv")
    write_scene_planner_table(rows_final, checkpoint, out_dir / f"metrics_by_scene_planner_{tag}.csv")
    write_per_query_table(rows_final, checkpoint, out_dir / f"metrics_per_query_{tag}.csv")
    write_planner_global_table(rows_final, checkpoint, out_dir / f"metrics_by_planner_global_{tag}.csv")
    write_first_path_table(all_rows, out_dir / "metrics_first_path_events.csv")

    print(f"Wrote tables under {out_dir} (single-checkpoint slice: {checkpoint}):")
    if not args.no_all_checkpoints:
        print(f"  metrics_meta_all_checkpoints.csv  (checkpoints: {' '.join(map(str, avail))})")
        print("  metrics_by_scene_planner_all_checkpoints.csv")
        print("  metrics_per_query_all_checkpoints.csv")
        print("  metrics_by_planner_global_all_checkpoints.csv")
    print(f"  metrics_meta_{tag}.csv")
    print(f"  metrics_by_scene_planner_{tag}.csv")
    print(f"  metrics_per_query_{tag}.csv")
    print(f"  metrics_by_planner_global_{tag}.csv")
    print("  metrics_first_path_events.csv")

    if args.latex:
        latex_paths = write_latex_snippets(rows_final, checkpoint, out_dir, tag)
        latex_paths.extend(
            write_latex_pathcost_by_checkpoint(all_rows, avail, out_dir, csv_path)
        )
        print("LaTeX snippets (booktabs; copy tabular into table environment):")
        for lp in latex_paths:
            print(f"  {lp.name}")


if __name__ == "__main__":
    main()
