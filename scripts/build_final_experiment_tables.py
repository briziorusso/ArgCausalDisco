#!/usr/bin/env python3
"""Build the final matched 50-run paper tables and empirical narrative.

ASPCR-DAG is included dataset-by-dataset only after all 50 corrected DAG runs
are present and validated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import collect_matched_baseline_tables as collector  # noqa: E402
from scripts.baseline_compatible_extensions import (  # noqa: E402
    dag_satisfies_ci_facts,
    enrich_progress_records,
    enumerate_consistent_extensions,
    parse_ci_facts,
)
from scripts.run_matched_baseline_experiments import _estimate_to_cpdag_for_metrics  # noqa: E402


VERSION = "paper_final_matched_50rep"
EXPECTED_SEEDS = tuple(range(2026, 2076))
METHODS = ("ABA-PC", "OptABA-PC", "MPC", "FGS", "ASPCR-DAG")
METHOD_LABELS = {
    "ABA-PC": "ABA-PC",
    "OptABA-PC": "OptABA-PC",
    "MPC": "MPC",
    "FGS": "FGS",
    "ASPCR-DAG": r"ASPCR-DAG",
}
DATASET_LABELS = {spec.key: spec.label for spec in collector.DATASETS}
DATASET_ORDER = tuple(spec.key for spec in collector.DATASETS)
DATASET_NODES = {
    "cancer": 5, "earthquake": 5, "survey": 6, "asia": 8,
    "er5": 5, "er8": 8, "sf5": 5, "sf8": 8,
}
BNLEARN_DATASETS = DATASET_ORDER[:4]
SYNTHETIC_DATASETS = DATASET_ORDER[4:]
BASELINE_VERSIONS = (
    "paper_bnlearn_alpha001_nowrong_noweight_50rep_mpc_fgs",
    "paper_er_sf_alpha001_nowrong_noweight_50rep_mpc",
    "paper_er_sf_alpha001_nowrong_noweight_50rep_fgs",
    "paper_fgs_bdeu_alpha001_n5000_50rep",
)
CORRECTED_ASPCR_VERSIONS = (
    "paper_aspcr_dag_alpha001_n5000_50rep",
    "paper_aspcr_dag_er_sf_e1_alpha001_n5000_50rep",
)
CORRECTED_ASPCR_VERSION = CORRECTED_ASPCR_VERSIONS[0]
FGS_BDEU_VERSION = "paper_fgs_bdeu_alpha001_n5000_50rep"
SPARSE_MCS_DIRNAME = "final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep"
SPARSE_BASELINE_VERSION = "paper_er_sf_sparse_alpha001_noweight_50rep_mpc_fgs"
CONTESTABILITY_TABLE_DIRNAME = "paper_current_alpha001_nowrong_noweight_50rep_preview"
GRAPH_METRICS = (
    "cpdag_shd_avg", "cpdag_shd_best", "cpdag_shd_worst",
    "cpdag_f1_avg", "cpdag_f1_best", "cpdag_f1_worst",
    "dag_shd_avg", "dag_shd_best", "dag_shd_worst",
    "dag_f1_avg", "dag_f1_best", "dag_f1_worst",
    "time_sec", "n_cpdags_compat", "n_dags_compat",
    "n_cpdags_returned", "n_dags_returned", "endpoint_valid",
)
FACT_METRICS = (
    "total_facts", "correct_facts", "wrong_facts", "accepted_facts",
    "accepted_correct_facts", "accepted_wrong_facts", "removed_facts",
    "removed_wrong_facts", "fact_precision", "fact_recall", "fact_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final and supplementary tables from the matched primary experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--mcs-results-dir",
        default="results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked",
    )
    parser.add_argument(
        "--mcs-recovery-dir",
        default="results/recovery_optaba_runs",
        help="Optional completed ABA/OptABA-PC recovery runs, overlaid by dataset and seed.",
    )
    parser.add_argument("--out-dir", default=f"results/tables/{VERSION}")
    return parser.parse_args()


def _normalise_method(method: Any) -> str:
    value = str(method)
    if value in {"ASPCR-log", "ASPCR-DAG-log", "ASPCR-DAG"}:
        return "ASPCR-DAG"
    return value


def _portable_path(path: Path) -> str:
    """Represent repository artefacts without embedding a workstation path."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sort_records(frame: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    return (
        frame.assign(
            _dataset_order=frame["dataset"].map(dataset_order).fillna(len(dataset_order)),
            _method_order=frame["method"].map(method_order).fillna(len(method_order)),
            _seed_order=pd.to_numeric(frame["seed"], errors="coerce"),
        )
        .sort_values(
            ["_dataset_order", "_method_order", "_seed_order"],
            kind="mergesort",
        )
        .drop(columns=["_dataset_order", "_method_order", "_seed_order"])
        .reset_index(drop=True)
    )


def _load_progress(results_dir: Path, versions: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for priority, version in enumerate(versions):
        frame = collector._load_progress_records(results_dir, [version])
        if frame.empty:
            continue
        frame = frame.copy()
        frame["method"] = frame["method"].map(_normalise_method)
        frame["source_version"] = version
        frame["_priority"] = priority
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["_priority", "source_version"]).drop_duplicates(
        ["dataset", "method", "seed"], keep="last"
    )
    return out.drop(columns=["_priority"])


def _corrected_aspcr_counts(results_dir: Path, version: str) -> dict[str, int]:
    progress_dir = results_dir / "progress" / version
    counts: dict[str, int] = {}
    for dataset in DATASET_ORDER:
        path = progress_dir / f"{dataset}__aspcr_log_dag_dag.csv"
        if not path.exists():
            counts[dataset] = 0
            continue
        frame = pd.read_csv(path)
        ok = frame[frame.get("status", "") == "ok"]
        counts[dataset] = int(pd.to_numeric(ok.get("seed"), errors="coerce").dropna().nunique())
    return counts


def _select_aspcr(results_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    corrected = _load_progress(results_dir, CORRECTED_ASPCR_VERSIONS)
    counts_by_version = {
        version: _corrected_aspcr_counts(results_dir, version)
        for version in CORRECTED_ASPCR_VERSIONS
    }
    selected: list[pd.DataFrame] = []
    manifest: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        eligible_versions = (
            {CORRECTED_ASPCR_VERSIONS[0]}
            if dataset in BNLEARN_DATASETS
            else {CORRECTED_ASPCR_VERSIONS[1]}
        )
        corrected_dataset = (
            corrected[
                (corrected.get("dataset", "") == dataset)
                & corrected.get("source_version", pd.Series(index=corrected.index, dtype=str)).isin(eligible_versions)
            ].copy()
            if not corrected.empty
            else pd.DataFrame()
        )
        corrected_seeds = set(pd.to_numeric(corrected_dataset.get("seed"), errors="coerce").dropna().astype(int))
        if corrected_seeds == set(EXPECTED_SEEDS):
            chosen = corrected_dataset
            status = "corrected_dag_complete"
            sources = sorted(set(chosen.get("source_version", pd.Series(dtype=str)).dropna().astype(str)))
            if len(sources) != 1:
                raise RuntimeError(f"ASPCR {dataset} rows mix source versions: {sources}")
            source = sources[0]
        else:
            chosen = pd.DataFrame()
            status = "unavailable"
            source = None
        if not chosen.empty:
            chosen["aspcr_status"] = status
            selected.append(chosen)
        manifest[dataset] = {
            "status": status,
            "source_version": source,
            "corrected_successful_seeds": (
                counts_by_version.get(str(source), {}).get(dataset, 0) if source else 0
            ),
            "selected_rows": int(len(chosen)),
        }
    return (pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()), manifest


def _load_corrected_aspcr_facts(results_dir: Path, selected_manifest: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, entry in selected_manifest.items():
        if entry["status"] != "corrected_dag_complete":
            continue
        progress_dir = results_dir / "progress" / str(entry["source_version"])
        path = progress_dir / f"{dataset}__aspcr_log_dag_dag.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["status"] == "ok"].copy()
        for row in frame.to_dict("records"):
            rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "method": "ASPCR-DAG",
                "seed": int(row["seed"]),
                "total_facts": row["fact_total"],
                "correct_facts": row["fact_true"],
                "wrong_facts": row["fact_false"],
                "accepted_facts": row["fact_retained"],
                "accepted_correct_facts": row["fact_retained_true"],
                "accepted_wrong_facts": row["fact_retained_false"],
                "removed_facts": row["fact_removed"],
                "removed_wrong_facts": row["fact_removed_false"],
                "fact_precision": row["fact_precision"],
                "fact_recall": row["fact_recall"],
                "fact_f1": row["fact_f1"],
                "source": _portable_path(path),
            })
    return pd.DataFrame(rows)


def _load_repair_records(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only directly comparable ABA-PC/OptABA-PC records."""
    graph = collector._load_mcs_records(root, include_partial=False, allow_injected_wrong_facts=False)
    facts = collector._load_mcs_fact_records(root, include_partial=False, allow_injected_wrong_facts=False)
    graph = graph[graph["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
    facts = facts[facts["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
    return graph, facts


def _overlay_repair_recoveries(
    primary: pd.DataFrame,
    recoveries: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay explicitly rerun repair rows without duplicating an identity."""

    if recoveries.empty:
        return primary
    recoveries = recoveries[recoveries["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
    combined = pd.concat([primary, recoveries], ignore_index=True)
    return combined.drop_duplicates(["dataset", "method", "seed"], keep="last")


def _load_raw_fact_records(root: Path) -> pd.DataFrame:
    facts = collector._load_mcs_fact_records(root, include_partial=False, allow_injected_wrong_facts=False)
    return facts[facts["method"] == "Raw PC facts"].copy()


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _audit_test_compatibility(
    graph_records: pd.DataFrame,
    raw_fact_records: pd.DataFrame,
) -> pd.DataFrame:
    """Separate returned-extension size from compatibility with enforced CI facts."""

    output = graph_records.copy()
    for column in ("n_cpdags_returned", "n_dags_returned", "compatibility_basis"):
        if column not in output:
            output[column] = np.nan if column != "compatibility_basis" else ""

    # CausalABA enumerates graphs satisfying the retained (post-repair) CI set.
    repair_mask = output["method"].isin(("ABA-PC", "OptABA-PC"))
    output.loc[repair_mask, "n_cpdags_returned"] = output.loc[repair_mask, "n_cpdags_compat"]
    output.loc[repair_mask, "n_dags_returned"] = output.loc[repair_mask, "n_dags_compat"]
    output.loc[repair_mask, "compatibility_basis"] = "retained CI facts"

    # Score-based FGS has no tested/enforced CI set, so compatibility is undefined.
    fgs_mask = output["method"] == "FGS"
    output.loc[fgs_mask, ["n_cpdags_compat", "n_dags_compat"]] = np.nan
    output.loc[fgs_mask, "compatibility_basis"] = "not applicable (score based)"

    fact_paths: dict[tuple[str, int], tuple[Path, int]] = {}
    for row in raw_fact_records.itertuples(index=False):
        source = _resolve_repo_path(row.source)
        fact_paths[(str(row.dataset), int(row.seed))] = (
            source.parent / f"rep{int(row.rep_local)}" / "facts.lp",
            int(row.total_facts),
        )

    fact_cache: dict[Path, list[tuple[bool, int, int, frozenset[int]]]] = {}
    for index, row in output[output["method"] == "MPC"].iterrows():
        identity = (str(row["dataset"]), int(row["seed"]))
        fact_entry = fact_paths.get(identity)
        if fact_entry is None:
            raise RuntimeError(f"Missing matched CI trace for {row['method']} {identity}")
        facts_path, expected_facts = fact_entry
        if not facts_path.exists():
            raise RuntimeError(f"Missing matched CI trace for {row['method']} {identity}")
        facts = fact_cache.setdefault(facts_path, parse_ci_facts(facts_path))
        if len(facts) != expected_facts:
            raise RuntimeError(
                f"Incomplete matched CI trace for {row['method']} {identity}: "
                f"expected {expected_facts}, found {len(facts)}"
            )
        artifact_path = _resolve_repo_path(row.get("raw_graph_path", ""))
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                cpdag = _estimate_to_cpdag_for_metrics(np.asarray(artifact["W_est"]))
            extensions = enumerate_consistent_extensions(cpdag)
            compatible = dag_satisfies_ci_facts(extensions[0], facts)
        except Exception:
            extensions = []
            compatible = False
        output.at[index, "n_cpdags_returned"] = 1.0 if extensions else 0.0
        output.at[index, "n_dags_returned"] = float(len(extensions))
        output.at[index, "n_cpdags_compat"] = 1.0 if compatible else 0.0
        output.at[index, "n_dags_compat"] = float(len(extensions)) if compatible else 0.0
        output.at[index, "compatibility_basis"] = "full recorded matched G2 CI trace"

    for index, row in output[output["method"] == "ASPCR-DAG"].iterrows():
        constraints_path = _resolve_repo_path(row.get("aspcr_constraints_path", ""))
        if not constraints_path.exists():
            raise RuntimeError(
                f"Missing ASPCR constraint trace for {row['dataset']}/{int(row['seed'])}"
            )
        constraints = pd.read_csv(constraints_path)
        n_nodes = DATASET_NODES[str(row["dataset"])]
        expected_facts = math.comb(n_nodes, 2) * (2 ** (n_nodes - 2))
        if len(constraints) != expected_facts:
            raise RuntimeError(
                f"Incomplete ASPCR CI trace for {row['dataset']}/{int(row['seed'])}: "
                f"expected {expected_facts}, found {len(constraints)}"
            )
        compatible = bool(
            len(constraints)
            and (constraints["tested_relation"] == constraints["final_graph_relation"]).all()
        )
        output.at[index, "n_cpdags_returned"] = 1.0
        output.at[index, "n_dags_returned"] = 1.0
        output.at[index, "n_cpdags_compat"] = 1.0 if compatible else 0.0
        output.at[index, "n_dags_compat"] = 1.0 if compatible else 0.0
        output.at[index, "compatibility_basis"] = "complete native ASPCR CI trace"

    return output


def _validate_inputs(graph_records: pd.DataFrame, fact_records: pd.DataFrame) -> None:
    expected = set(EXPECTED_SEEDS)
    if graph_records.duplicated(["dataset", "method", "seed"]).any():
        raise RuntimeError("Final graph records contain duplicate dataset/method/seed identities")
    if fact_records.duplicated(["dataset", "method", "seed"]).any():
        raise RuntimeError("Final fact records contain duplicate dataset/method/seed identities")
    contaminated = pd.to_numeric(graph_records.get("pct_wrong_facts"), errors="coerce").fillna(0) > 0
    if contaminated.any():
        raise RuntimeError("Final graph records include injected wrong facts")

    required_compatibility_columns = (
        "n_cpdags_returned", "n_dags_returned",
        "n_cpdags_compat", "n_dags_compat", "compatibility_basis",
    )
    missing_columns = [name for name in required_compatibility_columns if name not in graph_records]
    if missing_columns:
        raise RuntimeError(f"Missing compatibility-audit columns: {missing_columns}")

    # Structural output size and satisfaction of CI evidence are distinct.
    # These assertions run on every production-table build, preventing the old
    # extension-count-as-compatibility error from silently recurring.
    fgs = graph_records[graph_records["method"] == "FGS"]
    if not fgs.empty and (
        pd.to_numeric(fgs["n_cpdags_compat"], errors="coerce").notna().any()
        or pd.to_numeric(fgs["n_dags_compat"], errors="coerce").notna().any()
    ):
        raise RuntimeError("FGS is score based: CI compatibility must be undefined")
    audited = graph_records[graph_records["method"] != "FGS"]
    if audited["compatibility_basis"].fillna("").str.strip().eq("").any():
        raise RuntimeError("Every CI-based result must declare its compatibility basis")
    for compatible_name, returned_name in (
        ("n_cpdags_compat", "n_cpdags_returned"),
        ("n_dags_compat", "n_dags_returned"),
    ):
        compatible = pd.to_numeric(audited[compatible_name], errors="coerce")
        returned = pd.to_numeric(audited[returned_name], errors="coerce")
        if compatible.isna().any() or returned.isna().any():
            raise RuntimeError(f"Incomplete compatibility audit: {compatible_name}/{returned_name}")
        if (compatible < 0).any() or (returned < 0).any():
            raise RuntimeError("Negative returned/compatible graph count")
        if (compatible > returned).any():
            raise RuntimeError(f"{compatible_name} exceeds structural {returned_name}")
    single_cpdag = audited[audited["method"].isin(("MPC", "ASPCR-DAG"))]
    compatible_cpdags = set(
        pd.to_numeric(single_cpdag["n_cpdags_compat"], errors="coerce").dropna().unique()
    )
    if not compatible_cpdags.issubset({0.0, 1.0}):
        raise RuntimeError("Single-output baseline has an invalid compatible-CPDAG count")
    fact_identities = {
        "total = correct + false": fact_records["total_facts"] - fact_records["correct_facts"] - fact_records["wrong_facts"],
        "retained = retained-correct + retained-false": fact_records["accepted_facts"] - fact_records["accepted_correct_facts"] - fact_records["accepted_wrong_facts"],
        "removed = total - retained": fact_records["removed_facts"] - fact_records["total_facts"] + fact_records["accepted_facts"],
    }
    for name, residual in fact_identities.items():
        values = pd.to_numeric(residual, errors="coerce").dropna()
        if not np.allclose(values, 0):
            raise RuntimeError(f"Inconsistent fact accounting: {name}")
    for dataset in DATASET_ORDER:
        for method in ("ABA-PC", "MPC"):
            rows = graph_records[(graph_records["dataset"] == dataset) & (graph_records["method"] == method)]
            seeds = set(pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int))
            if seeds != expected:
                raise RuntimeError(f"Expected 50 matched {dataset}/{method} seeds; got {len(seeds)}")
        rows = fact_records[(fact_records["dataset"] == dataset) & (fact_records["method"] == "ABA-PC")]
        seeds = set(pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int))
        if seeds != expected:
            raise RuntimeError(f"Expected 50 fact-audit {dataset}/ABA-PC seeds; got {len(seeds)}")

        # ABA-PC and OptABA-PC must be audited only on the identical tested CI
        # facts.  This is the comparison for which retained-fact metrics have a
        # common meaning; graph-only baselines are deliberately excluded.
        repairs = fact_records[
            (fact_records["dataset"] == dataset)
            & fact_records["method"].isin(("ABA-PC", "OptABA-PC"))
        ]
        for metric in ("total_facts", "correct_facts", "wrong_facts"):
            pivot = repairs.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            if {"ABA-PC", "OptABA-PC"}.issubset(pivot.columns):
                paired = pivot[["ABA-PC", "OptABA-PC"]].dropna()
                if not np.allclose(paired["ABA-PC"], paired["OptABA-PC"]):
                    raise RuntimeError(f"Mismatched tested CI inputs for {dataset}: {metric}")


def _summarise(records: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in records.groupby(["dataset", "method"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset),
            "method": method,
            "n_seeds": int(pd.to_numeric(group["seed"], errors="coerce").dropna().nunique()),
        }
        for metric in metrics:
            raw_values = group[metric] if metric in group else pd.Series(dtype=float)
            values = pd.to_numeric(raw_values, errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else float("nan"))
            row[f"{metric}_n"] = int(len(values))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    dataset_order = {name: index for index, name in enumerate(DATASET_ORDER)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    out["_dataset"] = out["dataset"].map(dataset_order)
    out["_method"] = out["method"].map(method_order).fillna(999)
    return out.sort_values(["_dataset", "_method"]).drop(columns=["_dataset", "_method"]).reset_index(drop=True)


def _bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(adjusted, index=values.index)
    order = valid[np.argsort(p[valid])]
    running = 1.0
    m = len(order)
    for rank_index in range(m - 1, -1, -1):
        idx = order[rank_index]
        rank = rank_index + 1
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def _paired_test_rows(graph_records: pd.DataFrame, fact_records: pd.DataFrame) -> pd.DataFrame:
    if collector.stats is None:
        raise RuntimeError("scipy is required for paired Wilcoxon tests")
    metric_specs = (
        ("fact_f1", "Fact F1", True, fact_records, ("ABA-PC",)),
        ("fact_precision", "Fact precision", True, fact_records, ("ABA-PC",)),
        ("fact_recall", "Fact recall", True, fact_records, ("ABA-PC",)),
        ("accepted_correct_facts", "Correct facts kept", True, fact_records, ("ABA-PC",)),
        ("accepted_wrong_facts", "Incorrect facts kept", False, fact_records, ("ABA-PC",)),
        ("cpdag_shd_avg", "CPDAG SHD avg", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("cpdag_shd_best", "CPDAG SHD best", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("cpdag_shd_worst", "CPDAG SHD worst", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("cpdag_f1_avg", "CPDAG F1 avg", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("cpdag_f1_best", "CPDAG F1 best", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("cpdag_f1_worst", "CPDAG F1 worst", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_shd_avg", "DAG SHD avg", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_shd_best", "DAG SHD best", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_shd_worst", "DAG SHD worst", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_f1_avg", "DAG F1 avg", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_f1_best", "DAG F1 best", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("dag_f1_worst", "DAG F1 worst", True, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
        ("n_cpdags_compat", "Compatible CPDAGs", False, graph_records, ("ABA-PC",)),
        ("n_dags_compat", "Compatible DAGs", False, graph_records, ("ABA-PC",)),
        ("time_sec", "Time", False, graph_records, ("ABA-PC", "MPC", "FGS", "ASPCR-DAG")),
    )
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for metric, label, higher_better, source, comparators in metric_specs:
            subset = source[source["dataset"] == dataset]
            pivot = subset.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            for comparator in comparators:
                if "OptABA-PC" not in pivot or comparator not in pivot:
                    continue
                paired = pivot[["OptABA-PC", comparator]].dropna()
                if paired.empty:
                    continue
                subject = paired["OptABA-PC"].to_numpy(dtype=float)
                other = paired[comparator].to_numpy(dtype=float)
                delta = subject - other
                if np.allclose(delta, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = collector.stats.wilcoxon(subject, other, alternative="two-sided", zero_method="wilcox")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                wins = delta > 0 if higher_better else delta < 0
                rows.append({
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "metric": metric,
                    "metric_label": label,
                    "higher_better": higher_better,
                    "subject": "OptABA-PC",
                    "comparator": comparator,
                    "n": len(paired),
                    "subject_mean": float(subject.mean()),
                    "comparator_mean": float(other.mean()),
                    "delta_mean": float(delta.mean()),
                    "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "wins": int(wins.sum()),
                    "wilcoxon_statistic": statistic,
                    "p": p_value,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_bh"] = out.groupby(["dataset", "metric"], sort=False)["p"].transform(_bh_adjust)
    out["favourable"] = np.where(out["higher_better"], out["delta_mean"] > 0, out["delta_mean"] < 0)
    out["significant_favourable"] = out["favourable"] & (out["p_bh"] < 0.05)
    return out


def _num(value: Any, digits: int = 2) -> str:
    if not collector._is_num(value):
        return "--"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def _math(value: Any, *, bold: bool = False, digits: int = 2, n: Any | None = None) -> str:
    if not collector._is_num(value):
        return "--"
    body = _num(value, digits)
    body = rf"\mathbf{{{body}}}" if bold else body
    if n is not None and collector._is_num(n) and int(n) < len(EXPECTED_SEEDS):
        body += rf"_{{{int(n)}}}"
    return rf"${body}$"


def _mean_std(
    mean: Any,
    std: Any,
    n: Any | None = None,
    *,
    digits: int = 2,
    bold: bool = False,
    marker: str = "",
) -> str:
    if not collector._is_num(mean):
        return "--"
    body = rf"{_num(mean, digits)}\pm{_num(std, digits)}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    if marker:
        body += rf"^{{{marker}}}"
    if n is not None and collector._is_num(n) and int(n) < len(EXPECTED_SEEDS):
        body += rf"_{{{int(n)}}}"
    return rf"${body}$"


def _summary_lookup(summary: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    return {(str(row.dataset), str(row.method)): row for row in summary.itertuples(index=False)}


def _method_label(method: str, dataset: str, aspcr_manifest: dict[str, dict[str, Any]]) -> str:
    del dataset, aspcr_manifest
    if method != "ASPCR-DAG":
        return METHOD_LABELS.get(method, method)
    return METHOD_LABELS[method]


def _best_methods(graph_summary: pd.DataFrame, dataset: str, metric: str, *, higher: bool) -> set[str]:
    subset = graph_summary[graph_summary["dataset"] == dataset]
    values = {
        str(row["method"]): float(row[f"{metric}_mean"])
        for _, row in subset.iterrows() if collector._is_num(row.get(f"{metric}_mean"))
    }
    if not values:
        return set()
    best = max(values.values()) if higher else min(values.values())
    return {method for method, value in values.items() if math.isclose(value, best, abs_tol=1e-12)}


def _test_row(tests: pd.DataFrame, dataset: str, comparator: str, metric: str) -> pd.Series | None:
    rows = tests[
        (tests["dataset"] == dataset)
        & (tests["comparator"] == comparator)
        & (tests["metric"] == metric)
    ]
    return None if rows.empty else rows.iloc[0]


def _star_marker(row: pd.Series | None) -> str:
    if row is None:
        return ""
    p_value = float(row["p_bh"])
    level = 3 if p_value < 0.001 else (2 if p_value < 0.01 else (1 if p_value < 0.05 else 0))
    return "*" * level


def _annotated_mean_std(
    row: Any,
    metric: str,
    method: str,
    dataset: str,
    tests: pd.DataFrame,
    *,
    digits: int = 2,
) -> str:
    """Absolute appendix value with embedded paired significance indicators."""

    marker = ""
    bold = False
    if method in {"ABA-PC", "OptABA-PC"}:
        test = _test_row(tests, dataset, "ABA-PC", metric)
        significance = _star_marker(test)
        winner = None
        if significance and test is not None:
            winner = "OptABA-PC" if bool(test["favourable"]) else "ABA-PC"
        marker = significance if method == winner else ""
        bold = method == winner
    else:
        test = _test_row(tests, dataset, method, metric)
        level = len(_star_marker(test))
        marker = r"\dagger" * level
        bold = bool(marker and test is not None and not test["favourable"])
    return _mean_std(
        getattr(row, f"{metric}_mean"),
        getattr(row, f"{metric}_std"),
        getattr(row, f"{metric}_n"),
        digits=digits,
        bold=bold,
        marker=marker,
    )


def _delta_mean_std(row: pd.Series | None, *, digits: int = 2) -> str:
    if row is None:
        return "--"
    mean = float(row["delta_mean"])
    std = float(row["delta_std"])
    marker = _star_marker(row)
    body = rf"{mean:+.{digits}f}\pm{_num(std, digits)}"
    if marker and bool(row["favourable"]):
        body = rf"\mathbf{{{body}}}^{{{marker}}}"
    elif marker:
        body += rf"^{{{marker}}}"
    if int(row["n"]) < len(EXPECTED_SEEDS):
        body += rf"_{{{int(row['n'])}}}"
    return rf"${body}$"


def _value_with_aba_delta(
    mean: Any,
    std: Any,
    n: Any,
    tests: pd.DataFrame,
    dataset: str,
    metric: str,
    *,
    digits: int,
) -> str:
    absolute = _mean_std(mean, std, n, digits=digits)
    delta = _delta_mean_std(_test_row(tests, dataset, "ABA-PC", metric), digits=digits)
    return rf"\shortstack{{{absolute}\\[-1pt]{{\scriptsize $\Delta_{{\mathrm{{ABA}}}}$: {delta}}}}}"


def _render_main(
    graph_summary: pd.DataFrame,
    fact_summary: pd.DataFrame,
    tests: pd.DataFrame,
    aspcr_manifest: dict[str, dict[str, Any]],
) -> str:
    del aspcr_manifest
    graph = _summary_lookup(graph_summary)
    facts = _summary_lookup(fact_summary)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Method & Correct retained $\uparrow$ & Fact F1 $\uparrow$ & \#CPDAGs $\downarrow$ & CPDAG F1 $\uparrow$ & CPDAG SHD $\downarrow$ & Time (s) $\downarrow$\\",
        r"\midrule",
    ]
    metric_specs = (
        ("accepted_correct_facts", "fact", 1, True),
        ("fact_f1", "fact", 3, True),
        ("n_cpdags_compat", "graph", 1, False),
        ("cpdag_f1_avg", "graph", 3, True),
        ("cpdag_shd_avg", "graph", 2, False),
        ("time_sec", "graph", 2, False),
    )
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        for method_index, method in enumerate(("ABA-PC", "OptABA-PC")):
            grow = graph[(dataset, method)]
            frow = facts[(dataset, method)]
            cells: list[str] = []
            for metric, source, digits, _ in metric_specs:
                row = frow if source == "fact" else grow
                test = _test_row(tests, dataset, "ABA-PC", metric)
                significance = _star_marker(test)
                winner = None
                if significance and test is not None:
                    winner = "OptABA-PC" if bool(test["favourable"]) else "ABA-PC"
                marker = significance if method == winner else ""
                bold = method == winner
                cells.append(_mean_std(
                    getattr(row, f"{metric}_mean"),
                    getattr(row, f"{metric}_std"),
                    digits=digits,
                    bold=bold,
                    marker=marker,
                ))
            lines.append(" & ".join([
                DATASET_LABELS[dataset] if method_index == 0 else "",
                method,
                *cells,
            ]) + r"\\")
        if dataset_index != len(DATASET_ORDER) - 1:
            lines.append(r"\addlinespace[1pt]")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}%",
        r"\caption{Complete absolute scale-and-variability view of the compact primary paired comparison (mean $\pm$ standard deviation over 50 matched seeds, $N=5000$, $\alpha=0.01$). This table reports all eight primary datasets, including Earthquake, which is omitted from the compact main table because the two repair methods do not differ significantly on its core fact, compatibility, or graph-accuracy outcomes. Both methods repair the same recorded CI tests. Correct retained is the number of facts agreeing with $d$-separation in the generating DAG; Fact F1 scores retained-fact classification; \#CPDAGs counts equivalence classes satisfying every retained post-repair fact. Arrows give the preferred direction. Only a statistically significant winner is bold and marked: one, two, and three stars denote BH-adjusted $p<0.05$, $0.01$, and $0.001$ from paired two-sided Wilcoxon tests. Non-significant pairs are plain.}",
        r"\label{tab:absolute-results}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _delta_relative_cell(
    test: pd.Series | None,
    *,
    digits: int,
    time_ratio: bool = False,
) -> str:
    if test is None:
        return "--"
    delta = float(test["delta_mean"])
    comparator = float(test["comparator_mean"])
    display_digits = max(digits, 2) if 0 < abs(delta) < 1 and digits < 2 else digits
    delta_text = f"{delta:+.{display_digits}f}"
    if time_ratio:
        relative_text = "--" if math.isclose(comparator, 0.0) else rf"\times{float(test['subject_mean']) / comparator:.2f}"
    else:
        relative_text = "--" if math.isclose(comparator, 0.0) else rf"{100.0 * delta / comparator:+.1f}\%"
    body = rf"{delta_text}\ ({relative_text})"
    marker = _star_marker(test)
    if marker:
        body = rf"\mathbf{{{body}}}^{{{marker}}}"
    return rf"${body}$"


def _render_main_delta(tests: pd.DataFrame) -> str:
    metric_specs = (
        ("accepted_correct_facts", r"$\Delta n_{\mathrm{acc}}^{T}\uparrow$", 1, False),
        ("fact_f1", r"$\Delta\mathrm{FactF1}\uparrow$", 3, False),
        ("n_cpdags_compat", r"$\Delta\#\mathrm{CPDAG}\downarrow$", 1, False),
        ("cpdag_f1_avg", r"$\Delta\mathrm{F1}_{\mathrm{CPDAG}}\uparrow$", 3, False),
        ("cpdag_shd_avg", r"$\Delta\mathrm{SHD}\downarrow$", 3, False),
        ("time_sec", r"$\Delta t\downarrow$", 3, True),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        "Dataset & " + " & ".join(label for _, label, _, _ in metric_specs) + r"\\",
        r"\midrule",
    ]
    for dataset in (dataset for dataset in DATASET_ORDER if dataset != "earthquake"):
        lines.append(" & ".join([
            DATASET_LABELS[dataset],
            *[
                _delta_relative_cell(
                    _test_row(tests, dataset, "ABA-PC", metric),
                    digits=digits,
                    time_ratio=time_ratio,
                )
                for metric, _, digits, time_ratio in metric_specs
            ],
        ]) + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Paired OptABA-PC--ABA-PC effects on seven fixed-network and synthetic benchmarks (number of nodes in brackets). Entries are mean within-seed differences $\Delta$; parentheses give change relative to ABA-PC, except runtime ($t_{\mathrm{Opt}}/t_{\mathrm{ABA}}$). Positive values favour upward metrics and negative values favour \#CPDAG, SHD, and time. $\Delta n_{\mathrm{acc}}^{T}$ counts additional retained true CI facts; \#CPDAG counts MECs satisfying all retained facts. Bold effects are statistically significant: one, two, and three stars denote BH-adjusted paired $p<0.05$, $0.01$, and $0.001$; the sign identifies the favoured method. The supplementary absolute-results table gives absolute mean $\pm$ standard deviation for all eight datasets; Earthquake is omitted here because its core outcomes do not differ significantly.}",
        r"\label{tab:main-results}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_graph_metrics(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    aspcr_manifest: dict[str, dict[str, Any]],
) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Dataset & Method & CPDAG SHD $\downarrow$ & CPDAG F1 $\uparrow$ & DAG SHD $\downarrow$ & DAG F1 $\uparrow$ & Compat. CPDAGs $\downarrow$\\", r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        subset = summary[summary["dataset"] == dataset]
        for method_index, row in enumerate(subset.itertuples(index=False)):
            lines.append(" & ".join([
                DATASET_LABELS[dataset] if method_index == 0 else "",
                _method_label(row.method, dataset, aspcr_manifest),
                _annotated_mean_std(row, "cpdag_shd_avg", row.method, dataset, tests),
                _annotated_mean_std(row, "cpdag_f1_avg", row.method, dataset, tests, digits=3),
                _annotated_mean_std(row, "dag_shd_avg", row.method, dataset, tests),
                _annotated_mean_std(row, "dag_f1_avg", row.method, dataset, tests, digits=3),
                _annotated_mean_std(row, "n_cpdags_compat", row.method, dataset, tests, digits=1),
            ]) + r"\\")
        if dataset_index != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Graph reconstruction and CI-compatible equivalence classes on the matched primary experiments (mean $\pm$ standard deviation over seeds). ABA-PC and OptABA-PC CPDAG values average over every equivalence class admitted by the selected repair; MPC and FGS return one CPDAG when their output is structurally valid, and ASPCR-DAG returns one DAG whose equivalence class supplies its CPDAG score. DAG columns average every compatible DAG for ABA-PC/OptABA-PC and every consistent extension of a valid MPC/FGS CPDAG; ASPCR-DAG is scored on its returned DAG. The final column counts CPDAGs satisfying the enforced evidence: retained post-repair facts for ABA-PC/OptABA-PC and every recorded test for MPC/ASPCR-DAG; it is undefined for score-based FGS (`--'). Stars identify the significant ABA-PC--OptABA-PC winner. Daggers on an external method indicate a paired difference from OptABA-PC; bold daggers favour that external method, whereas unbold daggers favour OptABA-PC. One, two, and three symbols denote BH-adjusted $p<0.05$, $0.01$, and $0.001$. Subscripts occur only for MPC DAG-extension metrics when its endpoint output is invalid: $n=38$ on Survey and $n=49$ on ER(8).}",
        r"\label{tab:final-graph-metrics}", r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_fact_metrics(summary: pd.DataFrame, tests: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccccc}", r"\toprule",
        r"Dataset & Method & Correct available & Correct kept & False kept & Precision & Recall & Fact F1 & $n$\\", r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        subset = summary[summary["dataset"] == dataset]
        for method_index, row in enumerate(subset.itertuples(index=False)):
            lines.append(" & ".join([
                DATASET_LABELS[dataset] if method_index == 0 else "",
                METHOD_LABELS.get(row.method, row.method),
                _mean_std(row.correct_facts_mean, row.correct_facts_std, digits=1),
                _annotated_mean_std(row, "accepted_correct_facts", row.method, dataset, tests, digits=1),
                _annotated_mean_std(row, "accepted_wrong_facts", row.method, dataset, tests, digits=1),
                _annotated_mean_std(row, "fact_precision", row.method, dataset, tests, digits=3),
                _annotated_mean_std(row, "fact_recall", row.method, dataset, tests, digits=3),
                _annotated_mean_std(row, "fact_f1", row.method, dataset, tests, digits=3),
                str(int(row.n_seeds)),
            ]) + r"\\")
        if dataset_index != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{CI-fact accounting on the matched primary experiments (mean $\pm$ standard deviation). Correct available is the number of recorded tests agreeing with $d$-separation in the generating DAG; correct/false kept split the facts retained after repair. Precision, recall, and Fact F1 treat a retained correct test as a true positive, a retained false test as a false positive, and a released correct test as a false negative. The final column $n$ is the number of successful runs contributing to that row. ABA-PC and OptABA-PC repair the identical test set, so stars identify their significant paired winner at BH-adjusted $p<0.05$, $0.01$, and $0.001$; only the winner is bold and marked. For ASPCR-DAG only, \emph{kept} means that the returned DAG satisfies a native soft test relation; this is descriptive graph--test agreement, not an explicit accepted or retained CI set. MPC and FGS are omitted because neither exposes comparable fact accounting.}",
        r"\label{tab:final-fact-metrics}", r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_range(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    aspcr_manifest: dict[str, dict[str, Any]],
    *,
    prefix: str,
) -> str:
    is_cpdag = prefix == "cpdag"
    label = "CPDAG" if is_cpdag else "DAG"
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}", r"\toprule",
        rf"Dataset & Method & SHD avg & SHD best & SHD worst & F1 avg & F1 best & F1 worst & Compat. {label}s\\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        subset = summary[summary["dataset"] == dataset]
        for method_index, row in enumerate(subset.itertuples(index=False)):
            metric_names = [
                f"{prefix}_shd_avg", f"{prefix}_shd_best", f"{prefix}_shd_worst",
                f"{prefix}_f1_avg", f"{prefix}_f1_best", f"{prefix}_f1_worst",
                "n_cpdags_compat" if is_cpdag else "n_dags_compat",
            ]
            lines.append(" & ".join([
                DATASET_LABELS[dataset] if method_index == 0 else "",
                _method_label(row.method, dataset, aspcr_manifest),
                *[
                    _annotated_mean_std(
                        row,
                        name,
                        row.method,
                        dataset,
                        tests,
                        digits=3 if 3 <= index <= 5 else (1 if index == 6 else 2),
                    )
                    for index, name in enumerate(metric_names)
                ],
            ]) + r"\\")
        if dataset_index != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}%",
        rf"\caption{{Average, best, and worst {label} reconstruction within each run of the matched primary experiments (mean $\pm$ standard deviation over seeds). "
        + (
            r"ABA-PC and OptABA-PC can return several CPDAG equivalence classes satisfying their selected facts; MPC and FGS return one CPDAG when structurally valid, and ASPCR-DAG's returned DAG determines one CPDAG, so their average, best, and worst CPDAG values coincide within a run. Compat. CPDAGs counts classes satisfying the retained facts for ABA-PC/OptABA-PC or every recorded native CI test for MPC/ASPCR-DAG; compatibility is undefined for score-based FGS (`--')."
            if is_cpdag else
            r"ABA-PC and OptABA-PC are scored over every DAG satisfying their selected facts. For MPC and FGS, we enumerate the structurally consistent extensions of each valid returned CPDAG to assess orientation quality throughout its equivalence class; ASPCR-DAG is scored on its single returned DAG, so its average, best, and worst values coincide. DAGs in one CPDAG share the same CI model but can have different orientation error against the generating DAG. Compat. DAGs counts extensions also satisfying the method's recorded CI evidence; compatibility is undefined for score-based FGS (`--')."
        )
        + r" Stars identify the significant ABA-PC--OptABA-PC winner. Daggers on an external method indicate a paired difference from OptABA-PC; bold daggers favour that external method and unbold daggers favour OptABA-PC. One, two, and three symbols denote BH-adjusted $p<0.05$, $0.01$, and $0.001$."
        + (
            "}" if is_cpdag else
            r" Subscripts occur only for MPC's endpoint-valid DAG extensions: $n=38$ on Survey and $n=49$ on ER(8).}"
        ),
        rf"\label{{tab:final-{prefix}-ranges}}", r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_runtime_completion(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    aspcr_manifest: dict[str, dict[str, Any]],
) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Dataset & Method & Time (s) & Returned CPDAGs & Returned DAGs & CI-compat. CPDAGs & CI-compat. DAGs\\", r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        subset = summary[summary["dataset"] == dataset]
        for method_index, row in enumerate(subset.itertuples(index=False)):
            lines.append(" & ".join([
                DATASET_LABELS[dataset] if method_index == 0 else "",
                _method_label(row.method, dataset, aspcr_manifest),
                _annotated_mean_std(row, "time_sec", row.method, dataset, tests),
                _mean_std(row.n_cpdags_returned_mean, row.n_cpdags_returned_std, row.n_cpdags_returned_n, digits=1),
                _mean_std(row.n_dags_returned_mean, row.n_dags_returned_std, row.n_dags_returned_n, digits=1),
                _annotated_mean_std(row, "n_cpdags_compat", row.method, dataset, tests, digits=1),
                _annotated_mean_std(row, "n_dags_compat", row.method, dataset, tests, digits=1),
            ]) + r"\\")
        if dataset_index != len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}%",
        r"\caption{Timing, structural output size, and CI compatibility for the matched primary experiments (mean $\pm$ standard deviation over 50 seeds for each reported method--dataset pair). Returned CPDAGs and DAGs are structural output/extension counts. CI-compatible counts additionally require all retained post-repair facts for ABA-PC/OptABA-PC or every recorded native test for MPC/ASPCR-DAG; compatibility is undefined for score-based FGS (`--'). ABA-PC and OptABA-PC times include construction, solving, and compatible-set evaluation and are directly paired. MPC/FGS time the learner call, while ASPCR-DAG reports its native end-to-end R run; their values and paired daggers describe implementations with different timing scopes and are not a controlled cross-family speed ranking. Stars identify the significant ABA-PC--OptABA-PC winner. Bold daggers favour the external method and unbold daggers favour OptABA-PC. One, two, and three symbols denote BH-adjusted $p<0.05$, $0.01$, and $0.001$.}",
        r"\label{tab:final-runtime-completion}", r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_sensitivity(
    dense_graph: pd.DataFrame,
    dense_facts: pd.DataFrame,
    sparse_graph: pd.DataFrame,
    sparse_facts: pd.DataFrame,
) -> str:
    scenario_inputs = {
        "dense": (dense_graph, dense_facts),
        "sparse": (sparse_graph, sparse_facts),
    }
    scenarios: dict[
        str,
        tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any], pd.DataFrame],
    ] = {}
    for name, (graph, facts) in scenario_inputs.items():
        graph = graph[graph["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
        facts = facts[facts["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
        scenarios[name] = (
            _summary_lookup(_summarise(graph, GRAPH_METRICS)),
            _summary_lookup(_summarise(facts, FACT_METRICS)),
            _paired_test_rows(graph, facts),
        )

    def add_panel(lines: list[str], datasets: tuple[str, ...], settings: tuple[tuple[str, str], ...]) -> None:
        for dataset_index, dataset in enumerate(datasets):
            first_dataset_row = True
            for setting, scenario_name in settings:
                graph, facts, tests = scenarios[scenario_name]
                for method_index, method in enumerate(("ABA-PC", "OptABA-PC")):
                    grow = graph[(dataset, method)]
                    frow = facts[(dataset, method)]
                    lines.append(" & ".join([
                        DATASET_LABELS[dataset] if first_dataset_row else "",
                        setting if method_index == 0 else "",
                        method,
                        _annotated_mean_std(frow, "accepted_correct_facts", method, dataset, tests, digits=1),
                        _annotated_mean_std(frow, "fact_f1", method, dataset, tests, digits=3),
                        _annotated_mean_std(grow, "cpdag_f1_avg", method, dataset, tests, digits=3),
                        _annotated_mean_std(grow, "cpdag_shd_avg", method, dataset, tests),
                        _annotated_mean_std(grow, "time_sec", method, dataset, tests),
                    ]) + r"\\")
                    first_dataset_row = False
            if dataset_index != len(datasets) - 1:
                lines.append(r"\addlinespace[2pt]")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\textbf{Density sensitivity at fixed $N=5000$ and $\alpha=0.01$}\\[2pt]",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Dataset & Setting & Method & Correct retained & Fact F1 & CPDAG F1 & CPDAG SHD & Time (s)\\",
        r"\midrule",
    ]
    add_panel(
        lines,
        ("er5", "er8", "sf5", "sf8"),
        (("Sparse (primary)", "sparse"), ("Dense", "dense")),
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}%",
        r"\caption{Graph-density sensitivity of the ABA-PC--OptABA-PC comparison (absolute mean $\pm$ standard deviation over seeds 2026--2075). Sparse graphs use one edge per node and are the primary synthetic setting; dense graphs use two edges per node. Sample size ($N=5000$), CI-test threshold ($\alpha=0.01$), data generator, and all other method settings are held fixed. Stars identify the statistically significant winner at BH-adjusted $p<0.05$, $0.01$, and $0.001$; only that value is bold and marked.}",
        r"\label{tab:repair-sensitivity}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _story(
    graph_summary: pd.DataFrame,
    fact_summary: pd.DataFrame,
    tests: pd.DataFrame,
    sparse_tests: pd.DataFrame,
    manifest: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    del graph_summary, fact_summary  # The narrative is based on paired, not marginal, evidence.

    def datasets(
        frame: pd.DataFrame,
        metric: str,
        comparator: str = "ABA-PC",
        *,
        significant: bool = False,
        favourable: bool | None = None,
    ) -> list[str]:
        rows = frame[(frame["metric"] == metric) & (frame["comparator"] == comparator)].copy()
        if significant:
            rows = rows[pd.to_numeric(rows["p_bh"], errors="coerce") < 0.05]
        if favourable is not None:
            rows = rows[rows["favourable"] == favourable]
        return [dataset for dataset in DATASET_ORDER if dataset in set(rows["dataset"])]

    label = lambda names: ", ".join(DATASET_LABELS[name] for name in names)  # noqa: E731
    correct_sig = datasets(tests, "accepted_correct_facts", significant=True, favourable=True)
    recall_sig = datasets(tests, "fact_recall", significant=True, favourable=True)
    fact_f1_gain = datasets(tests, "fact_f1", significant=True, favourable=True)
    fact_f1_loss = datasets(tests, "fact_f1", significant=True, favourable=False)
    shd_aba_sig = datasets(tests, "cpdag_shd_avg", significant=True, favourable=True)
    f1_aba_sig = datasets(tests, "cpdag_f1_avg", significant=True, favourable=True)
    shd_fgs_sig = datasets(tests, "cpdag_shd_avg", "FGS", significant=True, favourable=True)
    f1_fgs_sig = datasets(tests, "cpdag_f1_avg", "FGS", significant=True, favourable=True)
    shd_asp_sig = datasets(tests, "cpdag_shd_avg", "ASPCR-DAG", significant=True, favourable=True)
    f1_asp_sig = datasets(tests, "cpdag_f1_avg", "ASPCR-DAG", significant=True, favourable=True)
    asp_available = datasets(tests, "cpdag_f1_avg", "ASPCR-DAG")
    sparse_fact_sig = datasets(sparse_tests, "fact_f1", significant=True, favourable=True)
    sparse_shd_sig = datasets(sparse_tests, "cpdag_shd_avg", significant=True, favourable=True)
    unavailable = [DATASET_LABELS[d] for d, item in manifest.items() if item["status"] == "unavailable"]

    markdown = f"""# Experimental story

- **Primary claim: optimal repair improves heuristic repair.** The primary study uses the four fixed bnlearn networks and sparse one-edge-per-node ER/SF graphs. OptABA-PC retains significantly more correct facts on {len(correct_sig)}/8 datasets ({label(correct_sig)}) and has significantly higher fact recall on {len(recall_sig)}/8. It significantly lowers CPDAG SHD and raises CPDAG F1 on {len(shd_aba_sig)}/8 and {len(f1_aba_sig)}/8 datasets, respectively; no dataset significantly favours ABA-PC structurally.
- **Why ABA-PC can have higher Fact F1.** Fact F1 treats a retained correct test as a true positive, a retained wrong test as a false positive, and a released correct test as a false negative. ABA-PC's more aggressive release raises precision but sacrifices recall. OptABA-PC optimises release cost and coherence, not truth-labelled Fact F1, so its recall gain can be offset by retaining more false tests. In the primary results, {label(fact_f1_loss)} significantly favours ABA-PC on Fact F1, whereas {label(fact_f1_gain)} significantly favours OptABA-PC; the remaining {8 - len(fact_f1_gain) - len(fact_f1_loss)} differences are not significant.
- **Sparsity matters.** On all {len(sparse_fact_sig)}/4 primary sparse ER/SF datasets ({label(sparse_fact_sig)}), OptABA-PC significantly improves Fact F1 and lowers CPDAG SHD. The recall benefit dominates the precision cost in this regime. The two-edge-per-node results remain in the appendix as a density sensitivity analysis rather than being mixed into the primary table.
- **External baselines are graph-only references.** OptABA-PC significantly beats FGS in both CPDAG SHD and F1 on {len(shd_fgs_sig)}/8 and {len(f1_fgs_sig)}/8 datasets. Corrected ASPCR-DAG is matched to {len(asp_available)} primary datasets; OptABA-PC significantly lowers SHD on {len(shd_asp_sig)}/{len(asp_available)} ({label(shd_asp_sig)}) and raises F1 on {len(f1_asp_sig)}/{len(asp_available)} ({label(f1_asp_sig)}). MPC is especially strong on the sparse synthetic graphs because Majority-PC targets one CPDAG using majority evidence across separating sets, whereas OptABA-PC optimises coherent fact retention and can admit several CPDAGs. MPC is not significantly better than OptABA-PC's best class on ER(5), ER(8), or SF(5), and most synthetic MPC outputs do not satisfy their complete recorded trace; its structural margins should therefore not be turned into a fact-level claim.
- **ASPCR is appendix-only.** Validated corrected ASPCR-DAG results are included on the {len(asp_available)} matched primary datasets for which they are available; no result is reported for {', '.join(unavailable) if unavailable else 'none'}. The completed ER(5)/SF(5) artifacts use the matched one-edge-per-node synthetic setting and are included in the primary appendix comparison.
- **Recommended framing.** The paper should claim optimal, contestable CI-fact repair that improves correct-fact retention and heuristic-repair reconstruction, with especially clear gains on sparse graphs. Native discrete FGS is a competitive graph-only reference, while OptABA-PC has strong gains over validated ASPCR-DAG. The paper should not claim uniform graph-reconstruction dominance over MPC or FGS, or equate either method's structural accuracy with repair of the tested CI facts.
"""
    latex = (
        rf"Relative to heuristic ABA-PC, OptABA-PC retains significantly more correct facts on {len(correct_sig)}/8 datasets and significantly improves CPDAG SHD and F1 on {len(shd_aba_sig)}/8 and {len(f1_aba_sig)}/8, with no significant structural loss. "
        rf"Its primary Fact F1 is significantly higher on {label(fact_f1_gain)} and lower on {label(fact_f1_loss)} because optimal cost repair raises recall while sometimes retaining additional false tests. "
        rf"On the sparse ER/SF graphs, this recall gain dominates: Fact F1 and CPDAG SHD improve significantly on all four datasets; denser synthetic results are reported as sensitivity evidence. "
        rf"At graph level, native discrete FGS is competitive: it significantly outperforms OptABA-PC on at least one mean CPDAG metric on six datasets, whereas OptABA-PC is significantly better on both metrics for Earthquake and on SHD for Asia. OptABA-PC improves F1 over validated ASPCR-DAG on all {len(asp_available)} matched primary datasets. Majority-PC is especially strong on the sparse synthetic graphs because it targets one CPDAG using majority evidence across separating sets; OptABA-PC instead optimises coherent fact retention and may admit several CPDAGs. MPC and FGS remain graph-only references because neither defines a comparable retained CI-fact set."
    )
    return markdown, latex + "\n"


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    mcs_dir = Path(args.mcs_results_dir).expanduser().resolve()
    mcs_recovery_dir = Path(args.mcs_recovery_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mcs_graph = collector._load_mcs_records(mcs_dir, include_partial=False, allow_injected_wrong_facts=False)
    recovery_graph = collector._load_mcs_records(
        mcs_recovery_dir, include_partial=False, allow_injected_wrong_facts=False
    )
    mcs_graph = _overlay_repair_recoveries(mcs_graph, recovery_graph)
    mcs_graph["method"] = mcs_graph["method"].map(_normalise_method)
    baseline_graph = _load_progress(results_dir, BASELINE_VERSIONS)
    baseline_graph = enrich_progress_records(baseline_graph, repo_root=REPO_ROOT)
    aspcr_graph, aspcr_manifest = _select_aspcr(results_dir)
    dense_graph_records = pd.concat(
        [frame for frame in (mcs_graph, baseline_graph, aspcr_graph) if not frame.empty],
        ignore_index=True,
    )
    dense_graph_records = dense_graph_records[
        dense_graph_records["dataset"].isin(DATASET_ORDER)
        & dense_graph_records["method"].isin(METHODS)
    ].copy()

    mcs_facts = collector._load_mcs_fact_records(mcs_dir, include_partial=False, allow_injected_wrong_facts=False)
    recovery_facts = collector._load_mcs_fact_records(
        mcs_recovery_dir, include_partial=False, allow_injected_wrong_facts=False
    )
    recovery_facts = recovery_facts[recovery_facts["method"] != "Raw PC facts"].copy()
    mcs_facts = _overlay_repair_recoveries(mcs_facts, recovery_facts)
    repair_facts = mcs_facts[mcs_facts["method"].isin(("ABA-PC", "OptABA-PC"))].copy()
    aspcr_facts = _load_corrected_aspcr_facts(results_dir, aspcr_manifest)
    dense_fact_records = pd.concat(
        [frame for frame in (repair_facts, aspcr_facts) if not frame.empty],
        ignore_index=True,
    )
    dense_fact_records = dense_fact_records[dense_fact_records["dataset"].isin(DATASET_ORDER)].copy()

    # The primary synthetic setting uses one edge per node.  A matched
    # two-edge-per-node configuration is reported separately as a density
    # sensitivity analysis.
    sparse_graph, sparse_facts = _load_repair_records(results_dir / SPARSE_MCS_DIRNAME)
    sparse_raw_facts = _load_raw_fact_records(results_dir / SPARSE_MCS_DIRNAME)
    sparse_baseline_graph = _load_progress(
        results_dir, (SPARSE_BASELINE_VERSION, FGS_BDEU_VERSION)
    )
    sparse_baseline_graph = enrich_progress_records(sparse_baseline_graph, repo_root=REPO_ROOT)
    primary_synthetic_aspcr = aspcr_graph[
        aspcr_graph["dataset"].isin(SYNTHETIC_DATASETS)
    ].copy() if not aspcr_graph.empty else pd.DataFrame()
    graph_records = pd.concat([
        dense_graph_records[dense_graph_records["dataset"].isin(BNLEARN_DATASETS)],
        sparse_graph[sparse_graph["dataset"].isin(SYNTHETIC_DATASETS)],
        sparse_baseline_graph[sparse_baseline_graph["dataset"].isin(SYNTHETIC_DATASETS)],
        primary_synthetic_aspcr,
    ], ignore_index=True)
    graph_records = graph_records[
        graph_records["dataset"].isin(DATASET_ORDER)
        & graph_records["method"].isin(METHODS)
    ].copy()
    fact_records = pd.concat([
        dense_fact_records[dense_fact_records["dataset"].isin(BNLEARN_DATASETS)],
        sparse_facts[sparse_facts["dataset"].isin(SYNTHETIC_DATASETS)],
        dense_fact_records[
            (dense_fact_records["method"] == "ASPCR-DAG")
            & dense_fact_records["dataset"].isin(SYNTHETIC_DATASETS)
        ],
    ], ignore_index=True)
    raw_fact_records = pd.concat([
        mcs_facts[
            (mcs_facts["method"] == "Raw PC facts")
            & mcs_facts["dataset"].isin(BNLEARN_DATASETS)
        ],
        sparse_raw_facts[sparse_raw_facts["dataset"].isin(SYNTHETIC_DATASETS)],
    ], ignore_index=True)
    graph_records = _audit_test_compatibility(graph_records, raw_fact_records)
    graph_records = _sort_records(graph_records)
    fact_records = _sort_records(fact_records)

    primary_aspcr_manifest = json.loads(json.dumps(aspcr_manifest))
    for dataset in SYNTHETIC_DATASETS:
        if primary_aspcr_manifest[dataset]["status"] != "corrected_dag_complete":
            primary_aspcr_manifest[dataset]["note"] = (
                "No complete corrected ASPCR-DAG artifact is available for the primary synthetic setting."
            )

    _validate_inputs(graph_records, fact_records)
    graph_summary = _summarise(graph_records, GRAPH_METRICS)
    fact_summary = _summarise(fact_records, FACT_METRICS)
    tests = _paired_test_rows(graph_records, fact_records)

    dense_repair_tests = _paired_test_rows(
        dense_graph_records[dense_graph_records["method"].isin(("ABA-PC", "OptABA-PC"))],
        dense_fact_records[dense_fact_records["method"].isin(("ABA-PC", "OptABA-PC"))],
    )
    sparse_tests = _paired_test_rows(sparse_graph, sparse_facts)
    story_md, story_tex = _story(
        graph_summary,
        fact_summary,
        tests,
        sparse_tests,
        primary_aspcr_manifest,
    )

    graph_records.to_csv(out_dir / "final_graph_records.csv", index=False)
    fact_records.to_csv(out_dir / "final_fact_records.csv", index=False)
    compatibility_columns = [
        "dataset", "method", "seed", "compatibility_basis",
        "n_cpdags_returned", "n_dags_returned",
        "n_cpdags_compat", "n_dags_compat",
    ]
    compatibility_audit = graph_records[compatibility_columns].copy()
    compatibility_audit.to_csv(out_dir / "final_compatibility_audit.csv", index=False)
    def count_compatible_seeds(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce")
        return float("nan") if numeric.notna().sum() == 0 else float((numeric > 0).sum())

    compatibility_summary = (
        compatibility_audit.groupby(["dataset", "method"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            compatibility_basis=("compatibility_basis", "first"),
            returned_cpdags_mean=("n_cpdags_returned", "mean"),
            returned_dags_mean=("n_dags_returned", "mean"),
            ci_compatible_cpdag_seeds=("n_cpdags_compat", count_compatible_seeds),
            ci_compatible_dag_seeds=("n_dags_compat", count_compatible_seeds),
        )
        .reset_index()
    )
    compatibility_summary.to_csv(out_dir / "final_compatibility_summary.csv", index=False)
    graph_summary.to_csv(out_dir / "final_graph_summary.csv", index=False)
    fact_summary.to_csv(out_dir / "final_fact_summary.csv", index=False)
    tests.to_csv(out_dir / "final_paired_wilcoxon_tests.csv", index=False)
    sensitivity_tests = pd.concat([
        dense_repair_tests.assign(scenario="dense_alpha001_n5000"),
        sparse_tests.assign(scenario="sparse_alpha001_n5000"),
    ], ignore_index=True)
    sensitivity_tests.to_csv(out_dir / "final_sensitivity_paired_tests.csv", index=False)
    (out_dir / "table_final_main.tex").write_text(_render_main(graph_summary, fact_summary, tests, primary_aspcr_manifest))
    (out_dir / "table_final_main_delta.tex").write_text(_render_main_delta(tests))
    (out_dir / "table_final_graph_metrics.tex").write_text(_render_graph_metrics(graph_summary, tests, primary_aspcr_manifest))
    (out_dir / "table_final_fact_metrics.tex").write_text(_render_fact_metrics(fact_summary, tests))
    (out_dir / "table_final_cpdag_ranges.tex").write_text(_render_range(graph_summary, tests, primary_aspcr_manifest, prefix="cpdag"))
    (out_dir / "table_final_dag_ranges.tex").write_text(_render_range(graph_summary, tests, primary_aspcr_manifest, prefix="dag"))
    (out_dir / "table_final_runtime_completion.tex").write_text(_render_runtime_completion(graph_summary, tests, primary_aspcr_manifest))
    (out_dir / "table_final_paired_tests.tex").write_text(
        "% Inferential details are exported to final_paired_wilcoxon_tests.csv; "
        "significance indicators are retained in both the compact main table and "
        "the absolute appendix tables.\n"
    )
    (out_dir / "table_final_sensitivity.tex").write_text(
        _render_sensitivity(
            dense_graph_records, dense_fact_records,
            sparse_graph, sparse_facts,
        )
    )
    contestability_table_source = (
        results_dir / "tables" / CONTESTABILITY_TABLE_DIRNAME / "table_empirical_contestability.tex"
    )
    if not contestability_table_source.exists():
        raise FileNotFoundError(
            f"Missing validated contestability table: {contestability_table_source}"
        )
    (out_dir / "table_empirical_contestability.tex").write_text(
        contestability_table_source.read_text()
    )
    (out_dir / "experiment_story.md").write_text(story_md)
    (out_dir / "experiment_story.tex").write_text(story_tex)
    completion: dict[str, dict[str, Any]] = {}
    for (dataset, method), group in graph_records.groupby(["dataset", "method"], sort=False):
        seed_values = pd.to_numeric(group["seed"], errors="coerce")
        saved_seeds = set(seed_values.dropna().astype(int))
        evaluated = group[pd.to_numeric(group["cpdag_shd_avg"], errors="coerce").notna() & pd.to_numeric(group["dag_shd_avg"], errors="coerce").notna()]
        evaluated_seeds = set(pd.to_numeric(evaluated["seed"], errors="coerce").dropna().astype(int))
        completion[f"{dataset}/{method}"] = {
            "saved_seeds": len(saved_seeds),
            "graph_evaluations": len(evaluated_seeds),
            "missing_saved_seed_ids": sorted(set(EXPECTED_SEEDS) - saved_seeds),
            "missing_graph_evaluation_seed_ids": sorted(set(EXPECTED_SEEDS) - evaluated_seeds),
        }
    fact_completion: dict[str, dict[str, Any]] = {}
    for (dataset, method), group in fact_records.groupby(["dataset", "method"], sort=False):
        fact_seeds = set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int))
        fact_completion[f"{dataset}/{method}"] = {
            "fact_audits": len(fact_seeds),
            "missing_fact_audit_seed_ids": sorted(set(EXPECTED_SEEDS) - fact_seeds),
        }
    manifest = {
        "version": VERSION,
        "expected_seeds": list(EXPECTED_SEEDS),
        "sample_size": 5000,
        "ci_alpha": 0.01,
        "mcs_results_dir": _portable_path(mcs_dir),
        "mcs_recovery_dir": _portable_path(mcs_recovery_dir),
        "baseline_versions": list(BASELINE_VERSIONS),
        "corrected_aspcr_versions": list(CORRECTED_ASPCR_VERSIONS),
        "aspcr_selection": primary_aspcr_manifest,
        "all_corrected_aspcr_artifacts": aspcr_manifest,
        "primary_synthetic_setting": {
            "edge_per_node": 1,
            "repair_source": _portable_path(results_dir / SPARSE_MCS_DIRNAME),
            "baseline_version": SPARSE_BASELINE_VERSION,
        },
        "sensitivity_sources": {
            "dense_alpha001_n5000": _portable_path(mcs_dir),
            "sparse_alpha001_n5000": _portable_path(results_dir / SPARSE_MCS_DIRNAME),
        },
        "graph_completion": completion,
        "fact_completion": fact_completion,
        "fact_accounting_identities_validated": True,
        "compatibility_semantics": {
            "returned": "Structural CPDAG output and its consistent DAG extensions.",
            "ABA-PC/OptABA-PC": "Satisfies the retained post-repair CI facts.",
            "MPC": "Satisfies every test in the full recorded matched G2 CI-test trace.",
            "ASPCR-DAG": "Satisfies the complete native Bayesian CI-test trace.",
            "FGS": "Not applicable because FGS is score based.",
        },
        "compatibility_identities_validated": True,
        "compatibility_audit_csv": _portable_path(out_dir / "final_compatibility_audit.csv"),
        "compatibility_summary_csv": _portable_path(out_dir / "final_compatibility_summary.csv"),
        "graph_rows": len(graph_records),
        "fact_rows": len(fact_records),
    }
    (out_dir / "table_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(story_md)
    print(f"Wrote final tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
