#!/usr/bin/env python
"""Post-process benchmark_solvers CSV output.

This script:
- Concatenates and merges one or more benchmark CSVs from benchmark_solvers.py.
- Computes per-(n_nodes, solver) summary statistics (mean/std) across runs.
- Calculates speedups relative to baseline and provides attribution views
  showing contributions from search strategy, grounding, and solve phases.
- Supports multi-run merging, flexible filtering by run labels and solver names,
  and per-source exclusion mappings.
- Displays stage timing breakdowns (compile/ground/solve) when available.

Usage examples:
  # Single CSV summary
  python bench_postprocess.py --in results/benchmarks/bench_baseline.csv

  # Merge multiple runs and write combined CSV
  python bench_postprocess.py \
    --in results/benchmarks/bench_run1.csv results/benchmarks/bench_run2.csv \
    --merge-out results/benchmarks/merged.csv

  # Filter by run label tokens (filename stem or CSV 'run' column)
  python bench_postprocess.py \
    --in results/benchmarks/bench_depfirst.csv results/benchmarks/bench_increm_only.csv \
    --select-runs depfirst increm_only \
    --merge-out results/benchmarks/filtered.csv

  # Include only specific solvers
  python bench_postprocess.py \
    --in results/benchmarks/merged.csv \
    --include-solvers baseline incremental dep_first

  # Exclude solvers per source file
  python bench_postprocess.py \
    --in results/benchmarks/bench_depfirst.csv results/benchmarks/bench_increm_only.csv \
    --exclude-map bench_depfirst:incremental,dep_first \
    --merge-out results/benchmarks/merged_filtered.csv

  # Combined: multi-run merge with filtering and summary output
  python bench_postprocess.py \
    --in results/benchmarks/*.csv \
    --select-runs baseline increm \
    --include-solvers baseline incremental binsearch \
    --merge-out results/benchmarks/all_merged.csv \
    --out results/benchmarks/summary.csv
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
    ap.add_argument(
        "--merge-out",
        type=str,
        default=None,
        help="Optional path to write the merged raw CSV (after any run filtering)",
    )
    ap.add_argument(
        "--select-runs",
        nargs="+",
        default=None,
        help="Only include rows whose inferred run label (filename stem or CSV 'run') contains any of these tokens",
    )
    ap.add_argument(
        "--include-solvers",
        nargs="+",
        default=None,
        help="If provided, keep only these solver labels (e.g., incremental baseline dep_first)",
    )
    ap.add_argument(
        "--exclude-solvers",
        nargs="+",
        default=None,
        help="If provided, drop these solver labels across all inputs",
    )
    ap.add_argument(
        "--exclude-map",
        nargs="+",
        default=None,
        help="Per-source solver exclusion, format: source_stem:solver1,solver2 (e.g., bench_depfirst:incremental,dep_first)",
    )
    args = ap.parse_args()

    dfs = []
    for p in args.inputs:
        df = pd.read_csv(p)
        src_path = Path(p)
        df["_source"] = src_path.name
        df["_stem"] = src_path.stem
        # If the CSV does not already label a run, infer it from the filename.
        # This makes it easy to aggregate multiple independent runs like
        # bench_depfirst.csv, increm_only.csv, depfirst_only.csv without
        # modifying the benchmark harness.
        if "run" not in df.columns:
            df["run"] = src_path.stem
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Normalize solver naming to avoid missing blocks due to variants
    if "solver" in data.columns:
        data["solver"] = data["solver"].replace({
            "depfirst": "dep_first",
            "first_dep": "dep_first",
            "baseline_incr": "incremental",
        })

    # Optional run filtering: include only rows whose run label contains any of the provided tokens.
    if args.select_runs:
        tokens = [str(t).strip() for t in args.select_runs if str(t).strip()]
        if tokens:
            def _row_matches(row) -> bool:
                try:
                    s_run = str(row.get("run", ""))
                    s_src = str(row.get("_source", ""))
                    s_stem = str(row.get("_stem", ""))
                    blob = " ".join([s_run, s_src, s_stem])
                    return any(tok in blob for tok in tokens)
                except Exception:
                    return False
            before = len(data)
            data = data[data.apply(_row_matches, axis=1)]
            after = len(data)
            if after == 0:
                print("Warning: run selection filtered out all rows. Check your --select-runs tokens against file stems or pass no filter.")

    # Save merged raw if requested (handy for downstream analysis)
    if args.merge_out:
        out_path = Path(args.merge_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out_path, index=False)
        print(f"\nWrote merged CSV to {out_path}")

    # Global solver include/exclude
    if args.include_solvers:
        keep = set(args.include_solvers)
        if "solver" in data.columns:
            data = data[data["solver"].isin(keep)]
    if args.exclude_solvers:
        drop = set(args.exclude_solvers)
        if "solver" in data.columns:
            data = data[~data["solver"].isin(drop)]

    # Per-source solver exclusion map: tokens like "bench_depfirst:incremental,dep_first"
    if args.exclude_map:
        rules = {}
        for tok in args.exclude_map:
            try:
                src, solvers_csv = tok.split(":", 1)
                solvers = [s.strip() for s in solvers_csv.split(",") if s.strip()]
                if solvers:
                    rules[src.strip()] = set(solvers)
            except Exception:
                print(f"Warning: could not parse exclude-map token '{tok}', expected source:solver1,solver2")
        if rules and {"_stem", "solver"}.issubset(set(data.columns)):
            mask = pd.Series([True] * len(data))
            for src, solvers in rules.items():
                drop_rows = (data["_stem"] == src) & (data["solver"].isin(solvers))
                mask = mask & (~drop_rows)
            before = len(data)
            data = data[mask]
            after = len(data)
            if before != after:
                print(f"Applied exclude-map: removed {before-after} rows per source rules")

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

    # Pretty per-solver summary akin to prior screenshots
    def _pretty_solver_summary(df: pd.DataFrame) -> None:
        cols_needed = [
            "total_sec", "elapsed_sec", "timed_out", "n_models", "sid",
            "compile_sec_total", "ground_sec_total", "solve_sec_total",
        ]
        have = [c for c in cols_needed if c in df.columns]
        if not have:
            return
        for n in sorted(df["n_nodes"].dropna().unique()):
            sub = df[df["n_nodes"] == n]
            if sub.empty:
                continue
            try:
                timeout_val = sub.get("timeout", None)
                timeout_str = f", {int(float(timeout_val.iloc[0]))}s timeout" if timeout_val is not None else ""
            except Exception:
                timeout_str = ""
            print("\n" + "=" * 74)
            print(f"BENCHMARK RESULTS SUMMARY ({int(n)} nodes, {sub['run'].nunique()} runs{timeout_str})")
            print("=" * 74)
            for solver in sorted(sub["solver"].unique()):
                block = sub[sub["solver"] == solver]
                if block.empty:
                    continue
                mean_total = float(block.get("total_sec", block.get("elapsed_sec", pd.Series([float('nan')]))).mean())
                std_total = float(block.get("total_sec", block.get("elapsed_sec", pd.Series([float('nan')]))).std())
                timeout_rate = float(block["timed_out"].mean()) if "timed_out" in block.columns else float('nan')
                models_mean = float(block.get("n_models", pd.Series([float('nan')])).mean())
                sid_mean = float(block.get("sid", pd.Series([float('nan')])).mean())
                comp_mean = float(block.get("compile_sec_total", pd.Series([float('nan')])).mean())
                ground_mean = float(block.get("ground_sec_total", pd.Series([float('nan')])).mean())
                solve_mean = float(block.get("solve_sec_total", pd.Series([float('nan')])).mean())

                def _fmt_sec(v: float) -> str:
                    try:
                        if v is None or pd.isna(v) or not pd.notna(v):
                            return "n/a"
                        if v == float('inf') or v != v:  # inf or NaN
                            return "n/a"
                        return f"{round(float(v), 1)}s"
                    except Exception:
                        return "n/a"

                print(f"\n{solver.upper()}:")
                print(f"  Total time (mean):  {_fmt_sec(mean_total)}")
                print(f"  Total time (std):   {_fmt_sec(std_total)}")
                try:
                    to_str = f"{round(timeout_rate, 1)} runs" if pd.notna(timeout_rate) else "n/a"
                except Exception:
                    to_str = "n/a"
                print(f"  Timed out:          {to_str}")
                try:
                    models_str = f"{round(models_mean, 1)}" if pd.notna(models_mean) else "n/a"
                except Exception:
                    models_str = "n/a"
                print(f"  Models found (mean): {models_str}")
                try:
                    sid_str = f"{round(sid_mean, 1)}" if pd.notna(sid_mean) else "n/a"
                except Exception:
                    sid_str = "n/a"
                print(f"  SID (mean):         {sid_str}")
                print("  Time breakdown:")
                print(f"    - Compile: {_fmt_sec(comp_mean)}")
                print(f"    - Ground:  {_fmt_sec(ground_mean)}")
                print(f"    - Solve:   {_fmt_sec(solve_mean)}")

            # Speedup comparison relative to baseline using per-solver means
            pv = sub.groupby("solver", as_index=False)["elapsed_sec"].mean().set_index("solver")["elapsed_sec"]
            if "baseline" in pv.index:
                print("\nSPEEDUP COMPARISON (relative to baseline):")
                base = float(pv["baseline"]) if pd.notna(pv.get("baseline")) else float("nan")
                for s in sorted([c for c in pv.index if c != "baseline"]):
                    val = float(pv.get(s))
                    ratio = base / val if pd.notna(base) and pd.notna(val) and val != 0.0 else float("nan")
                    if pd.notna(ratio) and ratio != float('inf'):
                        print(f"  {s}: {round(ratio, 1)}x faster")

    _pretty_solver_summary(data)

    def _print_speedups(metric: str, *, label: str) -> None:
        if metric not in data.columns:
            return
        # Compare per-n_nodes means across solvers (no run alignment required)
        by = data.groupby(["n_nodes", "solver"], as_index=False)[metric].mean()
        pv = by.pivot_table(index="n_nodes", columns="solver", values=metric, aggfunc="first")
        if "baseline" not in pv.columns:
            return

        speed_rows = []
        for n in pv.index:
            base = pv.loc[n, "baseline"]
            for solver in sorted([c for c in pv.columns if c != "baseline"]):
                val = pv.loc[n, solver]
                ratio = float(base) / float(val) if pd.notna(base) and pd.notna(val) and float(val) != 0.0 else float("nan")
                speed_rows.append({"n_nodes": n, f"baseline_over_{solver}": ratio})
        speed_df = pd.DataFrame(speed_rows)
        if not speed_df.empty:
            speedups = speed_df.groupby("n_nodes", as_index=False).agg({c: ["mean", "std"] for c in speed_df.columns if c != "n_nodes"})
        else:
            speedups = pd.DataFrame()
        print(f"\nSpeedups for {label} (baseline / solver):")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
            print(speedups)

        if all(s in pv.columns for s in ("baseline", "binsearch", "incremental")):
            attr = []
            for n in pv.index:
                s_base = pv.loc[n, "baseline"]
                s_bin = pv.loc[n, "binsearch"]
                s_incr = pv.loc[n, "incremental"]
                attr.append({
                    "n_nodes": n,
                    "search_baseline_over_binsearch": float(s_base) / float(s_bin) if pd.notna(s_base) and pd.notna(s_bin) and float(s_bin) != 0.0 else float("nan"),
                    "grounding_binsearch_over_incremental": float(s_bin) / float(s_incr) if pd.notna(s_bin) and pd.notna(s_incr) and float(s_incr) != 0.0 else float("nan"),
                })
            attr_sum = pd.DataFrame(attr).groupby("n_nodes", as_index=False).agg(
                {
                    "search_baseline_over_binsearch": ["mean", "std"],
                    "grounding_binsearch_over_incremental": ["mean", "std"],
                }
            )
            print(f"\nAttribution view for {label}:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
                print(attr_sum)

        if all(s in pv.columns for s in ("binsearch", "dep_first")):
            attr2 = []
            for n in pv.index:
                s_bin = pv.loc[n, "binsearch"]
                s_dep = pv.loc[n, "dep_first"]
                attr2.append({
                    "n_nodes": n,
                    "reground_binsearch_over_dep_first": float(s_bin) / float(s_dep) if pd.notna(s_bin) and pd.notna(s_dep) and float(s_dep) != 0.0 else float("nan"),
                })
            attr2_sum = pd.DataFrame(attr2).groupby("n_nodes", as_index=False).agg(
                {
                    "reground_binsearch_over_dep_first": ["mean", "std"],
                }
            )
            print(f"\nReground cost for {label} (binsearch / dep_first):")
            print("  dep_first avoids regrounding by only toggling dep facts via externals")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
                print(attr2_sum)

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
