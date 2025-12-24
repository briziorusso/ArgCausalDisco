#!/usr/bin/env python
"""Post-process benchmark_solvers CSV output.

Usage examples:
  python -m ArgCausalDisco.bench_postprocess --in /tmp/attrib.csv
  python -m ArgCausalDisco.bench_postprocess --in /tmp/run1.csv /tmp/run2.csv --out /tmp/summary.csv

This script:
- concatenates one or more benchmark CSVs
- prints a per-(n_nodes, solver) summary (mean/std)
- prints speedups relative to baseline
- prints a simple attribution view:
    search_speedup = baseline / binsearch
    grounding_speedup = binsearch / incremental
  (when those solvers are present)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Iterable

import pandas as pd


def _bytes_to_mib(x: float | int) -> float:
    return float(x) / (1024.0 * 1024.0)


def _pick_first_present(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for c in columns:
        if c in df.columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="One or more CSVs from benchmark_solvers")
    ap.add_argument("--out", type=str, default=None, help="Optional path to write the summary CSV")
    args = ap.parse_args()

    dfs = []
    for p in args.inputs:
        df = pd.read_csv(p)
        df["_source"] = Path(p).name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Normalize legacy timeout column.
    # Expected current column: `timed_out`.
    timeout_col = _pick_first_present(["timed_out", "timeout", "timedout"], data)
    if timeout_col is None:
        data["timed_out"] = False
    else:
        data["timed_out"] = data[timeout_col]
    data["timed_out"] = data["timed_out"].fillna(False).astype(bool)

    # Normalize error column
    if "error" not in data.columns:
        data["error"] = ""

    # Ensure readable memory columns exist.
    if "peak_mem_mb" not in data.columns and "peak_mem_bytes" in data.columns:
        data["peak_mem_mb"] = data["peak_mem_bytes"].map(_bytes_to_mib)
    if "rss_delta_mb" not in data.columns and "rss_delta" in data.columns:
        # rss_delta is KiB on Linux in our harness
        data["rss_delta_mb"] = data["rss_delta"].map(lambda v: float(v) / 1024.0)

    # Normalize time metric name.
    # Prefer `total_sec` when present (includes solver+postprocess), else fall back.
    time_col = _pick_first_present(["total_sec", "elapsed_sec", "solver_sec"], data)
    if time_col is None:
        data["elapsed_sec"] = float("nan")
        time_col = "elapsed_sec"
    if time_col != "elapsed_sec":
        data["elapsed_sec"] = data[time_col]

    metrics = [
        "elapsed_sec",
        "peak_mem_mb",
        "rss_delta_mb",
        "sid",
        "solve_calls",
        "ground_calls",
        "timed_out",
    ]
    existing = [m for m in metrics if m in data.columns]

    summary = (
        data.groupby(["n_nodes", "solver"], as_index=False)
        .agg({m: ["mean", "std"] for m in existing if m != "timed_out"} | ({"timed_out": ["mean"]} if "timed_out" in existing else {}))
    )

    print("\nSummary (mean/std):")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(summary)

    def _print_speedups(metric: str, *, label: str) -> None:
        if metric not in data.columns:
            return
        pivot = data.pivot_table(index=["n_nodes", "run"], columns="solver", values=metric, aggfunc="first")
        if "baseline" not in pivot.columns:
            return

        speed = pivot.reset_index().copy()
        for solver in sorted([c for c in pivot.columns if c != "baseline"]):
            # Larger ratio means baseline uses more (slower / more memory).
            speed[f"baseline_over_{solver}"] = speed["baseline"] / speed[solver]

        cols = [c for c in speed.columns if c.startswith("baseline_over_")]
        if not cols:
            return
        speedups = speed.groupby("n_nodes", as_index=False).agg({c: ["mean", "std"] for c in cols})
        print(f"\nSpeedups for {label} (baseline / solver):")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
            print(speedups)

        if all(s in pivot.columns for s in ("baseline", "binsearch", "incremental")):
            attr = pivot.reset_index().copy()
            attr["search_baseline_over_binsearch"] = attr["baseline"] / attr["binsearch"]
            attr["grounding_binsearch_over_incremental"] = attr["binsearch"] / attr["incremental"]
            attr_sum = attr.groupby("n_nodes", as_index=False).agg(
                {
                    "search_baseline_over_binsearch": ["mean", "std"],
                    "grounding_binsearch_over_incremental": ["mean", "std"],
                }
            )
            print(f"\nAttribution view for {label}:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
                print(attr_sum)

    # Speedups + attribution for time and memory
    _print_speedups("elapsed_sec", label="time (elapsed_sec)")
    _print_speedups("peak_mem_mb", label="peak memory (MiB)")
    _print_speedups("rss_delta_mb", label="RSS delta (MiB)")

    # --- Stage timing / failure interpretation ---
    stage_cols = [
        "compile_sec_total",
        "ground_sec_total",
        "solve_sec_total",
        "postprocess_sec",
        "total_sec",
    ]
    have_stage = [c for c in stage_cols if c in data.columns]

    def _classify_timeout_row(row) -> str:
        err = str(row.get("error", "") or "")
        if "compile_and_ground" in err:
            return "compile/ground"
        # If stage columns exist, guess by max contributor.
        contrib = {}
        for c in ("compile_sec_total", "ground_sec_total", "solve_sec_total", "postprocess_sec"):
            v = row.get(c)
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if not math.isnan(v) and v >= 0:
                contrib[c] = v
        if contrib:
            best = max(contrib.items(), key=lambda kv: kv[1])[0]
            return best.replace("_sec_total", "")
        return "unknown"

    if have_stage:
        # A compact summary table that is easier to read than the raw MultiIndex aggregation.
        agg = (
            data.groupby(["n_nodes", "solver"], as_index=False)
            .agg(
                {
                    "compile_sec_total": "mean",
                    "ground_sec_total": "mean",
                    "solve_sec_total": "mean",
                    "postprocess_sec": "mean" if "postprocess_sec" in data.columns else "mean",
                    "total_sec": "mean" if "total_sec" in data.columns else "mean",
                    "timed_out": "mean",
                }
            )
        )
        # If total_sec is missing in the CSV but stage cols exist, derive a coarse total.
        if "total_sec" not in data.columns:
            agg["total_sec"] = (
                agg.get("compile_sec_total", 0.0)
                + agg.get("ground_sec_total", 0.0)
                + agg.get("solve_sec_total", 0.0)
                + agg.get("postprocess_sec", 0.0)
            )

        def _safe_div(num: float, den: float) -> float:
            try:
                if den and not math.isnan(den):
                    return float(num) / float(den)
            except Exception:
                pass
            return float("nan")

        agg["compile_share"] = agg.apply(lambda r: _safe_div(r.get("compile_sec_total", float("nan")), r.get("total_sec", float("nan"))), axis=1)
        agg["ground_share"] = agg.apply(lambda r: _safe_div(r.get("ground_sec_total", float("nan")), r.get("total_sec", float("nan"))), axis=1)
        agg["solve_share"] = agg.apply(lambda r: _safe_div(r.get("solve_sec_total", float("nan")), r.get("total_sec", float("nan"))), axis=1)
        agg["post_share"] = agg.apply(lambda r: _safe_div(r.get("postprocess_sec", float("nan")), r.get("total_sec", float("nan"))), axis=1)

        def _dominant_stage(row) -> str:
            items = {
                "compile": row.get("compile_sec_total", float("nan")),
                "ground": row.get("ground_sec_total", float("nan")),
                "solve": row.get("solve_sec_total", float("nan")),
                "post": row.get("postprocess_sec", float("nan")),
            }
            items = {k: float(v) for k, v in items.items() if v is not None and not math.isnan(float(v))}
            if not items:
                return "unknown"
            return max(items.items(), key=lambda kv: kv[1])[0]

        agg["dominant_stage"] = agg.apply(_dominant_stage, axis=1)

        # Round for printing.
        printable = agg.copy()
        for c in [
            "compile_sec_total",
            "ground_sec_total",
            "solve_sec_total",
            "postprocess_sec",
            "total_sec",
            "compile_share",
            "ground_share",
            "solve_share",
            "post_share",
            "timed_out",
        ]:
            if c in printable.columns:
                printable[c] = printable[c].map(lambda v: round(float(v), 4) if pd.notna(v) else v)

        print("\nStage breakdown (means) with dominant stage + timeout rate:")
        cols = [
            "n_nodes",
            "solver",
            "total_sec",
            "compile_sec_total",
            "ground_sec_total",
            "solve_sec_total",
            "postprocess_sec",
            "dominant_stage",
            "timed_out",
            "compile_share",
            "ground_share",
            "solve_share",
            "post_share",
        ]
        cols = [c for c in cols if c in printable.columns]
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 220):
            print(printable[cols].sort_values(["n_nodes", "solver"]))

        # Timeout classification table
        td = data[data["timed_out"] == True].copy()  # noqa: E712
        if not td.empty:
            td["timeout_stage"] = td.apply(_classify_timeout_row, axis=1)
            by_stage = (
                td.groupby(["n_nodes", "solver", "timeout_stage"], as_index=False)
                .size()
                .rename(columns={"size": "n_timeouts"})
            )
            print("\nTimeouts classified by stage:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 220):
                print(by_stage)
        else:
            print("\nNo timeouts detected in the input.")
    else:
        print(
            "\nStage timing columns not found in this CSV. "
            "Re-run the benchmark after updating benchmark_solvers to get compile/ground/solve/postprocess timings."
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Flatten multiindex columns
        flat = summary.copy()
        flat.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in flat.columns]
        flat.to_csv(out_path, index=False)
        print(f"\nWrote summary CSV to {out_path}")


if __name__ == "__main__":
    main()
