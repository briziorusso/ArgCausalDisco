#!/usr/bin/env python3
"""Audit whether each saved MPC CPDAG enforces the CI tests it received.

MPC (Majority-PC in this repository) uses CI tests procedurally and returns one
CPDAG; it does not expose a retained/released fact set or a repair objective.
This script therefore measures an *enforcement gap*, not a contestability
margin: it enumerates a consistent DAG extension of each returned CPDAG and
counts recorded test outcomes contradicted by the represented Markov class.

All DAGs in a CPDAG have the same d-separation model, so one consistent
extension is sufficient for the CI audit.  Invalid endpoint graphs remain
explicit and count as not fully enforcing their trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baseline_compatible_extensions import (  # noqa: E402
    enumerate_consistent_extensions,
    parse_ci_facts,
)
from scripts.collect_matched_baseline_tables import _load_mcs_fact_records  # noqa: E402
from scripts.run_matched_baseline_experiments import (  # noqa: E402
    _estimate_to_cpdag_for_metrics,
)


DEFAULT_GRAPH_RECORDS = (
    REPO_ROOT / "results" / "tables" / "paper_final_matched_50rep" / "final_graph_records.csv"
)
DEFAULT_DENSE_MCS_ROOT = (
    REPO_ROOT / "results" / "final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked"
)
DEFAULT_SPARSE_MCS_ROOT = (
    REPO_ROOT / "results" / "final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "tables" / "paper_final_matched_50rep" / "baseline_contestability"
)

BNLEARN_DATASETS = ("cancer", "earthquake", "survey", "asia")
SYNTHETIC_DATASETS = ("er5", "er8", "sf5", "sf8")
DATASET_ORDER = BNLEARN_DATASETS + SYNTHETIC_DATASETS
DATASET_LABELS = {
    "cancer": "Cancer (5)",
    "earthquake": "Earthquake (5)",
    "survey": "Survey (6)",
    "asia": "Asia (8)",
    "er5": "ER (5)",
    "er8": "ER (8)",
    "sf5": "SF (5)",
    "sf8": "SF (8)",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _raw_fact_paths(dense_root: Path, sparse_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    dense = _load_mcs_fact_records(
        dense_root, include_partial=False, allow_injected_wrong_facts=False
    )
    sparse = _load_mcs_fact_records(
        sparse_root, include_partial=False, allow_injected_wrong_facts=False
    )
    frames = [
        dense[(dense["method"] == "Raw PC facts") & dense["dataset"].isin(BNLEARN_DATASETS)],
        sparse[(sparse["method"] == "Raw PC facts") & sparse["dataset"].isin(SYNTHETIC_DATASETS)],
    ]
    facts = pd.concat(frames, ignore_index=True)
    if facts.duplicated(["dataset", "seed"]).any():
        raise RuntimeError("Raw matched CI records contain duplicate dataset/seed identities")

    paths: dict[tuple[str, int], dict[str, Any]] = {}
    for row in facts.itertuples(index=False):
        source = _resolve(str(row.source))
        facts_path = source.parent / f"rep{int(row.rep_local)}" / "facts.lp"
        if not facts_path.exists():
            raise FileNotFoundError(f"Missing matched CI trace: {facts_path}")
        paths[(str(row.dataset), int(row.seed))] = {
            "path": facts_path,
            "expected": int(row.total_facts),
        }
    return paths


def audit_dag_against_facts(
    dag: np.ndarray,
    facts: Iterable[tuple[bool, int, int, frozenset[int]]],
) -> dict[str, int]:
    """Count CI outcomes contradicted by a DAG."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(dag)))
    graph.add_edges_from(
        (int(parent), int(child))
        for parent, child in zip(*np.where(np.asarray(dag) == 1))
    )
    total = 0
    contradicted_independence = 0
    contradicted_dependence = 0
    for tested_independent, x, y, conditioning in facts:
        total += 1
        graph_independent = bool(
            nx.is_d_separator(graph, {int(x)}, {int(y)}, set(conditioning))
        )
        if graph_independent == bool(tested_independent):
            continue
        if tested_independent:
            contradicted_independence += 1
        else:
            contradicted_dependence += 1
    return {
        "total_tests": total,
        "contradicted_independence_tests": contradicted_independence,
        "contradicted_dependence_tests": contradicted_dependence,
        "contradicted_tests": contradicted_independence + contradicted_dependence,
    }


def summarise_audit(records: pd.DataFrame) -> pd.DataFrame:
    """Return dataset summaries without imputing counts for invalid graphs."""

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        group = records[records["dataset"] == dataset].copy()
        if group.empty:
            continue
        valid = group[group["endpoint_valid"]].copy()
        fully = int(group["all_tests_enforced"].sum())
        seeds = int(group["seed"].nunique())
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "outputs": seeds,
                "valid_outputs": int(valid["seed"].nunique()),
                "fully_enforcing_outputs": fully,
                "enforcement_failure_outputs": seeds - fully,
                "enforcement_failure_proportion": (seeds - fully) / seeds,
                "tests_mean_valid": valid["total_tests"].mean(),
                "contradicted_tests_mean_valid": valid["contradicted_tests"].mean(),
                "contradicted_tests_median_valid": valid["contradicted_tests"].median(),
                "contradicted_tests_max_valid": valid["contradicted_tests"].max(),
                "contradiction_rate_mean_valid": valid["contradiction_rate"].mean(),
                "contradicted_independence_mean_valid": valid[
                    "contradicted_independence_tests"
                ].mean(),
                "contradicted_dependence_mean_valid": valid[
                    "contradicted_dependence_tests"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    *,
    graph_records_path: Path,
    dense_root: Path,
    sparse_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph_records = pd.read_csv(graph_records_path)
    mpc = graph_records[graph_records["method"] == "MPC"].copy()
    if mpc.empty:
        raise RuntimeError(f"No MPC records found in {graph_records_path}")
    if mpc.duplicated(["dataset", "seed"]).any():
        raise RuntimeError("MPC graph records contain duplicate dataset/seed identities")

    fact_paths = _raw_fact_paths(dense_root, sparse_root)
    rows: list[dict[str, Any]] = []
    for row in mpc.itertuples(index=False):
        dataset = str(row.dataset)
        seed = int(row.seed)
        identity = (dataset, seed)
        if identity not in fact_paths:
            raise RuntimeError(f"No matched CI trace for MPC {dataset}/{seed}")
        fact_entry = fact_paths[identity]
        facts_path = Path(fact_entry["path"])
        facts = parse_ci_facts(facts_path)
        if len(facts) != int(fact_entry["expected"]):
            raise RuntimeError(
                f"Incomplete CI trace for {dataset}/{seed}: "
                f"expected {fact_entry['expected']}, found {len(facts)}"
            )

        artifact_path = _resolve(str(row.raw_graph_path))
        result: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "seed": seed,
            "method": "MPC",
            "endpoint_valid": False,
            "endpoint_invalid_reason": "",
            "n_dag_extensions": math.nan,
            "total_tests": len(facts),
            "contradicted_tests": math.nan,
            "contradicted_independence_tests": math.nan,
            "contradicted_dependence_tests": math.nan,
            "contradiction_rate": math.nan,
            "all_tests_enforced": False,
            "raw_graph_path": str(artifact_path),
            "facts_path": str(facts_path),
        }
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                cpdag = _estimate_to_cpdag_for_metrics(np.asarray(artifact["W_est"]))
            extensions = enumerate_consistent_extensions(cpdag)
            if not extensions:
                raise ValueError("Returned CPDAG has no consistent DAG extension")
            counts = audit_dag_against_facts(extensions[0], facts)
            result.update(counts)
            result["endpoint_valid"] = True
            result["n_dag_extensions"] = len(extensions)
            result["contradiction_rate"] = counts["contradicted_tests"] / counts["total_tests"]
            result["all_tests_enforced"] = counts["contradicted_tests"] == 0
        except Exception as exc:
            result["endpoint_invalid_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(result)

    audit = pd.DataFrame(rows).sort_values(
        ["dataset", "seed"],
        key=lambda series: series.map({name: i for i, name in enumerate(DATASET_ORDER)})
        if series.name == "dataset"
        else series,
    )
    return audit, summarise_audit(audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-records", default=str(DEFAULT_GRAPH_RECORDS))
    parser.add_argument("--dense-mcs-root", default=str(DEFAULT_DENSE_MCS_ROOT))
    parser.add_argument("--sparse-mcs-root", default=str(DEFAULT_SPARSE_MCS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_records = _resolve(args.graph_records)
    dense_root = _resolve(args.dense_mcs_root)
    sparse_root = _resolve(args.sparse_mcs_root)
    output_dir = _resolve(args.output_dir)
    audit, summary = run_audit(
        graph_records_path=graph_records,
        dense_root=dense_root,
        sparse_root=sparse_root,
    )
    audit_path = output_dir / "mpc_enforcement_audit.csv"
    summary_path = output_dir / "mpc_enforcement_summary.csv"
    manifest_path = output_dir / "mpc_enforcement_manifest.json"
    _atomic_dataframe(audit_path, audit)
    _atomic_dataframe(summary_path, summary)
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "script": str(Path(__file__).resolve()),
            "semantics": (
                "Enforcement gap: contradictions between the returned MPC Markov class and "
                "the full recorded matched G2 CI trace; this is not a repair margin."
            ),
            "inputs": {
                "graph_records": {"path": str(graph_records), "sha256": _sha256(graph_records)},
                "dense_mcs_root": str(dense_root),
                "sparse_mcs_root": str(sparse_root),
            },
            "outputs": {
                "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
                "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            },
            "counts": {
                "outputs": int(len(audit)),
                "valid_outputs": int(audit["endpoint_valid"].sum()),
                "fully_enforcing_outputs": int(audit["all_tests_enforced"].sum()),
                "outputs_with_enforcement_gap": int((~audit["all_tests_enforced"]).sum()),
            },
        },
    )
    print(summary.to_string(index=False))
    print(f"Wrote MPC enforcement audit: {audit_path}")
    print(f"Wrote MPC enforcement summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
