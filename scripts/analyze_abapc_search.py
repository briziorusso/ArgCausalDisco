#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

RUN_SUMMARY_RE = re.compile(
    r"\[run-summary\]\s+dataset=(?P<dataset>\S+)\s+model=(?P<model>.+?)\s+"
    r"run=(?P<run>[^ ]+)\s+seed=(?P<seed>\d+)\s+elapsed=(?P<elapsed>[0-9.]+)s\s+"
    r"build=(?P<build>[0-9.]+)s\s+ground=(?P<ground>[0-9.]+)s\s+first_solve=(?P<first>[0-9.]+)s\s+"
    r"satchecks=(?P<calls>\d+)\s+total=(?P<total>[0-9.]+)s\s+avg=(?P<avg>[0-9.]+)s\s+"
    r"min=(?P<min>[0-9.]+)s\s+max=(?P<max>[0-9.]+)s\s+sat=(?P<sat>\d+)\s+unsat=(?P<unsat>\d+)\s+"
    r"unknown=(?P<unknown>\d+)\s+final_solve=(?P<final>[0-9.]+)s\s+remove_n=(?P<remove>[^ ]+)\s+"
    r"bracket=\[(?P<lo>[^,]+),(?P<hi>[^\]]+)\]\s+approximate=(?P<approx>\w+)\s+resumed=(?P<resume>\w+)"
)

LABELS_RE = re.compile(r"\[run-summary\]\s+satcheck_labels=(?P<labels>\{.*\})")


def _to_bool(text: str) -> bool:
    return str(text).strip().lower() == "true"


def _to_num(text: str):
    if text in {"None", "n/a", "nan"}:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def load_run_summaries(version: str) -> list[dict]:
    log_path = RESULTS_DIR / f"log_{version}.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    summaries: list[dict] = []
    pending_index: int | None = None
    for raw_line in log_path.read_text(errors="replace").splitlines():
        match = RUN_SUMMARY_RE.search(raw_line)
        if match:
            row = match.groupdict()
            run_text = row["run"]
            if "/" in run_text:
                run_idx_1, run_total = run_text.split("/", 1)
                run_idx = int(run_idx_1) - 1
                run_total = int(run_total)
            else:
                run_idx = int(run_text)
                run_total = None
            best_unsat = _to_num(row["lo"])
            best_sat = _to_num(row["hi"])
            summaries.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "run_idx": run_idx,
                    "run_total": run_total,
                    "seed": int(row["seed"]),
                    "elapsed_sec": float(row["elapsed"]),
                    "build_sec": float(row["build"]),
                    "ground_sec": float(row["ground"]),
                    "first_solve_sec": float(row["first"]),
                    "satcheck_calls": int(row["calls"]),
                    "satcheck_total_sec": float(row["total"]),
                    "satcheck_avg_sec": float(row["avg"]),
                    "satcheck_min_sec": float(row["min"]),
                    "satcheck_max_sec": float(row["max"]),
                    "sat": int(row["sat"]),
                    "unsat": int(row["unsat"]),
                    "unknown": int(row["unknown"]),
                    "final_solve_sec": float(row["final"]),
                    "remove_n": _to_num(row["remove"]),
                    "best_unsat_removed": best_unsat,
                    "best_sat_removed": best_sat,
                    "approximate": _to_bool(row["approx"]),
                    "resumed": _to_bool(row["resume"]),
                    "bracket_width": (
                        int(best_sat) - int(best_unsat)
                        if isinstance(best_sat, int) and isinstance(best_unsat, int)
                        else None
                    ),
                    "satcheck_labels": {},
                }
            )
            pending_index = len(summaries) - 1
            continue

        label_match = LABELS_RE.search(raw_line)
        if label_match and pending_index is not None:
            try:
                labels = ast.literal_eval(label_match.group("labels"))
            except Exception:
                labels = {}
            if isinstance(labels, dict):
                summaries[pending_index]["satcheck_labels"] = labels
    return summaries


def load_active_search_state(version: str, dataset: str) -> dict | None:
    scenario_dir = RESULTS_DIR / f"abapc_{version}_{dataset}"
    state_path = scenario_dir / "satcheck_search.json"
    if not state_path.exists():
        return None
    data = json.loads(state_path.read_text())
    records = list(data.get("satcheck_records") or [])
    statuses = Counter()
    labels = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        statuses[str(record.get("status", "")).lower()] += 1
        labels[str(record.get("label", ""))] += 1
    best_unsat = data.get("best_unsat_removed")
    best_sat = data.get("best_sat_removed")
    return {
        "path": str(state_path),
        "search_complete": bool(data.get("search_complete", False)),
        "approximate": bool(data.get("approximate_remove_search", False)),
        "remove_n": data.get("remove_n"),
        "best_unsat_removed": best_unsat,
        "best_sat_removed": best_sat,
        "bracket_width": (
            int(best_sat) - int(best_unsat)
            if isinstance(best_sat, int) and isinstance(best_unsat, int)
            else None
        ),
        "calls": len(records),
        "cache_size": len(data.get("cache") or {}),
        "status_counts": dict(statuses),
        "label_counts": dict(labels),
    }


def fmt_sec(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}s"


def fmt_ratio(numer: float, denom: float) -> str:
    if denom == 0:
        return "n/a"
    return f"{(100.0 * numer / denom):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise ABAPC remove-search behaviour from experiment logs and checkpoints.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--dataset", default="child")
    args = parser.parse_args()

    summaries = load_run_summaries(args.version)
    if summaries:
        print(f"Completed runs for version={args.version}: {len(summaries)}")
        exact = [row for row in summaries if not row["approximate"]]
        approx = [row for row in summaries if row["approximate"]]
        print(f"  exact={len(exact)} approximate={len(approx)}")

        total_calls = sum(row["satcheck_calls"] for row in summaries)
        total_unknown = sum(row["unknown"] for row in summaries)
        total_sat_time = sum(row["satcheck_total_sec"] for row in summaries)
        total_final_time = sum(row["final_solve_sec"] for row in summaries)
        widths = [row["bracket_width"] for row in summaries if row["bracket_width"] is not None]
        label_counts = Counter()
        for row in summaries:
            label_counts.update(row.get("satcheck_labels") or {})

        print(f"  satcheck calls total={total_calls} avg={total_calls / len(summaries):.1f}")
        print(
            f"  satcheck time total={fmt_sec(total_sat_time)} avg/run={fmt_sec(total_sat_time / len(summaries))} "
            f"unknown_share={fmt_ratio(total_unknown, total_calls)}"
        )
        print(
            f"  final solve total={fmt_sec(total_final_time)} avg/run={fmt_sec(total_final_time / len(summaries))} "
            f"search_share={fmt_ratio(total_sat_time, total_sat_time + total_final_time)}"
        )
        if widths:
            print(
                "  terminal bracket width "
                f"min={min(widths)} median={int(statistics.median(widths))} max={max(widths)} "
                f"avg={statistics.mean(widths):.1f}"
            )
        print(f"  satcheck labels aggregate={dict(label_counts)}")
        print()
        print("Per-run:")
        for row in summaries:
            run_no = row["run_idx"] + 1
            total = row["run_total"]
            run_label = f"{run_no}/{total}" if total is not None else str(run_no)
            print(
                f"  run={run_label} seed={row['seed']} approx={row['approximate']} "
                f"bracket=[{row['best_unsat_removed']},{row['best_sat_removed']}] width={row['bracket_width']} "
                f"calls={row['satcheck_calls']} unknown={row['unknown']} "
                f"satcheck_total={fmt_sec(row['satcheck_total_sec'])} final={fmt_sec(row['final_solve_sec'])}"
            )
    else:
        print(f"No completed run summaries found for version={args.version}")

    active = load_active_search_state(args.version, args.dataset)
    if active:
        print()
        print(f"Active checkpoint for dataset={args.dataset}:")
        print(f"  path={active['path']}")
        print(
            f"  complete={active['search_complete']} approximate={active['approximate']} "
            f"remove_n={active['remove_n']} bracket=[{active['best_unsat_removed']},{active['best_sat_removed']}] "
            f"width={active['bracket_width']}"
        )
        print(
            f"  calls={active['calls']} cache={active['cache_size']} "
            f"statuses={active['status_counts']}"
        )
        print(f"  labels={active['label_counts']}")


if __name__ == "__main__":
    main()
