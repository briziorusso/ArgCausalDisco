#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_support import (  # noqa: E402
    CPDAG_METRIC_MAP,
    CPDAG_PROGRESS_COLUMNS,
    CPDAG_SUMMARY_COLUMNS,
    DAG_METRIC_MAP,
    DAG_PROGRESS_COLUMNS,
    DAG_SUMMARY_COLUMNS,
    save_summary_tables,
    summarise_results,
)


RESULTS_DIR = REPO_ROOT / "results"
DATASET_ORDER = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]


def _clean_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _cpdag_nnz(B: np.ndarray) -> int:
    directed = 0
    undirected = 0
    n = int(B.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            left = float(B[i, j])
            right = float(B[j, i])
            is_undirected = (left == -1.0) or (right == -1.0) or ((left != 0.0) and (right != 0.0))
            is_directed = (left != 0.0) ^ (right != 0.0)
            if is_undirected:
                undirected += 1
            elif is_directed:
                directed += 1
    return directed + undirected


def _load_run_rows(run_dir: Path, dataset: str, pretty_name: str) -> tuple[dict[str, object], dict[str, object]]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    dag_summary = dict(summary.get("dag") or {})
    cpdag_summary = dict(summary.get("cpdag") or {})
    elapsed = _clean_float(summary.get("elapsed_sec"))
    run_idx = int(summary.get("run_idx", 0))
    seed = int(summary.get("seed", 0))

    dag_eval = np.load(run_dir / "graph_est_dag_eval.npy")
    cpdag_eval = np.load(run_dir / "graph_est_cpdag_eval.npy")

    dag_row = {
        "dataset": dataset,
        "model": pretty_name,
        "elapsed": elapsed,
        "nnz": int(np.count_nonzero(dag_eval)),
        "fdr": np.nan,
        "tpr": np.nan,
        "fpr": np.nan,
        "precision": _clean_float(dag_summary.get("precision")),
        "recall": _clean_float(dag_summary.get("recall")),
        "F1": _clean_float(dag_summary.get("F1")),
        "adjacency_precision": _clean_float(dag_summary.get("adjacency_precision")),
        "adjacency_recall": _clean_float(dag_summary.get("adjacency_recall")),
        "adjacency_F1": _clean_float(dag_summary.get("adjacency_F1")),
        "arrowhead_precision": _clean_float(dag_summary.get("arrowhead_precision")),
        "arrowhead_recall": _clean_float(dag_summary.get("arrowhead_recall")),
        "arrowhead_F1": _clean_float(dag_summary.get("arrowhead_F1")),
        "shd": _clean_float(dag_summary.get("shd")),
        "sid": _clean_float(dag_summary.get("sid")),
        "run_idx": run_idx,
        "seed": seed,
    }

    cpdag_row = {
        "dataset": dataset,
        "model": pretty_name,
        "elapsed": elapsed,
        "nnz": int(_cpdag_nnz(cpdag_eval)),
        "fdr": np.nan,
        "tpr": np.nan,
        "fpr": np.nan,
        "precision": _clean_float(cpdag_summary.get("precision")),
        "recall": _clean_float(cpdag_summary.get("recall")),
        "F1": _clean_float(cpdag_summary.get("F1")),
        "adjacency_precision": _clean_float(cpdag_summary.get("adjacency_precision")),
        "adjacency_recall": _clean_float(cpdag_summary.get("adjacency_recall")),
        "adjacency_F1": _clean_float(cpdag_summary.get("adjacency_F1")),
        "arrowhead_precision": _clean_float(cpdag_summary.get("arrowhead_precision")),
        "arrowhead_recall": _clean_float(cpdag_summary.get("arrowhead_recall")),
        "arrowhead_F1": _clean_float(cpdag_summary.get("arrowhead_F1")),
        "shd": _clean_float(cpdag_summary.get("shd")),
        "sid_low": _clean_float(cpdag_summary.get("sid_low")),
        "sid_high": _clean_float(cpdag_summary.get("sid_high")),
        "run_idx": run_idx,
        "seed": seed,
    }
    return dag_row, cpdag_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild progress and summary files for one model version from archived run folders.")
    parser.add_argument("--source-version", required=True, help="Version prefix already present in run artifact directories.")
    parser.add_argument("--target-version", required=True, help="Version name to write under results/progress and stored_results_*.npy.")
    parser.add_argument("--model-key", required=True, help="Scenario key prefix, e.g. fgs, mpc.")
    parser.add_argument("--model-name", required=True, help="Pretty model label to store in progress/summary rows.")
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    args = parser.parse_args()

    progress_dir = RESULTS_DIR / "progress" / args.target_version
    progress_dir.mkdir(parents=True, exist_ok=True)

    dag_rows: list[dict[str, object]] = []
    cpdag_rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        runs_dir = RESULTS_DIR / f"{args.model_key}_{args.source_version}_{dataset}" / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.glob("run_*")):
            summary_path = run_dir / "run_summary.json"
            dag_path = run_dir / "graph_est_dag_eval.npy"
            cpdag_path = run_dir / "graph_est_cpdag_eval.npy"
            if not (summary_path.exists() and dag_path.exists() and cpdag_path.exists()):
                continue
            dag_row, cpdag_row = _load_run_rows(run_dir, dataset, args.model_name)
            dag_rows.append(dag_row)
            cpdag_rows.append(cpdag_row)

    dag_df = pd.DataFrame(dag_rows, columns=DAG_PROGRESS_COLUMNS).sort_values(["dataset", "run_idx"]).reset_index(drop=True)
    cpdag_df = pd.DataFrame(cpdag_rows, columns=CPDAG_PROGRESS_COLUMNS).sort_values(["dataset", "run_idx"]).reset_index(drop=True)

    for dataset in sorted(set(dag_df["dataset"])) if not dag_df.empty else []:
        dag_df[dag_df["dataset"] == dataset].to_csv(
            progress_dir / f"{dataset}__{args.model_key}_dag.csv",
            index=False,
        )
    for dataset in sorted(set(cpdag_df["dataset"])) if not cpdag_df.empty else []:
        cpdag_df[cpdag_df["dataset"] == dataset].to_csv(
            progress_dir / f"{dataset}__{args.model_key}_cpdag.csv",
            index=False,
        )

    dag_summary = summarise_results(dag_df[DAG_PROGRESS_COLUMNS[:-2]], DAG_METRIC_MAP).reindex(columns=DAG_SUMMARY_COLUMNS)
    cpdag_summary = summarise_results(cpdag_df[CPDAG_PROGRESS_COLUMNS[:-2]], CPDAG_METRIC_MAP).reindex(columns=CPDAG_SUMMARY_COLUMNS)
    save_summary_tables(RESULTS_DIR, args.target_version, dag_summary, cpdag_summary)

    print(f"Wrote {args.target_version}: {len(dag_rows)} DAG rows, {len(cpdag_rows)} CPDAG rows")


if __name__ == "__main__":
    main()
