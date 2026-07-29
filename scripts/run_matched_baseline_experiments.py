#!/usr/bin/env python3
"""Matched baseline runner for the MCS/OptABA-PC paper experiments.

This is a lean adaptation of the UAI 2026 `experiments.py` runner:
it keeps per-run progress files, resumability, raw graph artifacts, and separate
DAG/CPDAG summaries, but narrows the interface to the baselines and datasets
needed for the paper appendix.

Final paper protocol:
  - bnlearn: cancer, earthquake, survey, asia
  - synthetic: ER(5), ER(8), SF(5), SF(8); edge_per_node=1 is primary
  - repetitions/seeds: 50 runs, seeds 2026..2075
  - sample size: 5000
  - MPC CI test: gsq, alpha=0.01

ASPCR-log is supported as a separate slow method and retains its native
Bayesian log-weighted CI trace. ``aspcr_log_dag`` uses the acyclic, causally
sufficient ASP encoding and requires complete per-constraint and per-component
artifacts before a run is recorded as successful.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DAG_BASE_COLUMNS = [
    "dataset", "model", "elapsed", "nnz", "fdr", "tpr", "fpr",
    "precision", "recall", "F1", "shd",
]
CPDAG_BASE_COLUMNS = DAG_BASE_COLUMNS.copy()
FACT_COUNT_COLUMNS = [
    "fact_total", "fact_true", "fact_false", "fact_retained",
    "fact_retained_true", "fact_retained_false", "fact_removed",
    "fact_removed_true", "fact_removed_false",
]
FACT_METRIC_COLUMNS = ["fact_precision", "fact_recall", "fact_f1"]
ASPCR_PROVENANCE_COLUMNS = [
    "aspcr_model_space", "aspcr_encoding", "aspcr_solver_objective",
    "aspcr_recomputed_objective", "aspcr_graph_is_dag", "aspcr_n_directed",
    "aspcr_n_bidirected", "aspcr_n_tailtail", "aspcr_testing_time",
    "aspcr_encoding_time", "aspcr_solving_time", "aspcr_data_path",
    "aspcr_directed_path", "aspcr_bidirected_path", "aspcr_tailtail_path",
    "aspcr_constraints_path", "aspcr_diagnostics_path", "aspcr_time_path",
]
PROGRESS_COLUMNS = DAG_BASE_COLUMNS + [
    "run_idx", "rep", "seed", "status", "error",
    "raw_graph_path", "true_graph_path",
] + FACT_COUNT_COLUMNS + FACT_METRIC_COLUMNS + ASPCR_PROVENANCE_COLUMNS
SUMMARY_METRICS = [
    "elapsed", "nnz", "fdr", "tpr", "fpr", "precision", "recall", "F1", "shd",
] + FACT_COUNT_COLUMNS + FACT_METRIC_COLUMNS
PAPER_DATASETS = ("cancer", "earthquake", "survey", "asia", "er5", "er8", "sf5", "sf8")
DEFAULT_EXPERIMENT_DATASETS = ("cancer", "earthquake", "survey", "er5", "sf5")
DEFAULT_METHODS = ("aspcr_log_dag",)
ASPCR_METHODS = {
    "aspcr_log": {
        "algorithm": "log-weights",
        "model_space": "general",
        "encoding": "new_wmaxsat.pl",
    },
    "aspcr_log_dag": {
        "algorithm": "log-weights",
        "model_space": "dag_sufficient",
        "encoding": "new_wmaxsat_acyclic_sufficient.pl",
    },
}
DISPLAY_NAMES = {
    "mpc": "MPC",
    "spc": "Shapley-PC",
    "fgs": "FGS",
    "aspcr_log": "ASPCR-log",
    "aspcr_log_dag": "ASPCR-DAG-log",
    "random_edge": "Random (match |E|)",
}
DEFAULT_RSCRIPT = Path(os.environ.get("RSCRIPT") or shutil.which("Rscript") or "Rscript")
DEFAULT_CLINGO_BIN_DIR = Path(
    os.environ.get("CLINGO_BIN_DIR") or shutil.which("clingo") or sys.executable
).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    n_nodes: int
    graph_type: str | None = None


@dataclass(frozen=True)
class ASPCRRunResult:
    W_est: np.ndarray
    G_directed: np.ndarray
    G_bidirected: np.ndarray
    G_tailtail: np.ndarray
    elapsed: float
    diagnostics: dict[str, Any]
    paths: dict[str, Path]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _install_notears_stub() -> None:
    """Let cd_algorithms.models import without optional NOTEARS installed."""

    if "notears.nonlinear" in sys.modules:
        return
    notears_module = types.ModuleType("notears")
    notears_nonlinear_module = types.ModuleType("notears.nonlinear")

    class _DummyMLP:
        pass

    def _dummy_notears_nonlinear(*args: Any, **kwargs: Any) -> None:
        raise ImportError("notears is not installed; this runner only uses non-NOTEARS baselines")

    notears_nonlinear_module.NotearsMLP = _DummyMLP
    notears_nonlinear_module.notears_nonlinear = _dummy_notears_nonlinear
    notears_module.nonlinear = notears_nonlinear_module
    sys.modules["notears"] = notears_module
    sys.modules["notears.nonlinear"] = notears_nonlinear_module


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _empty_metrics() -> dict[str, float]:
    return {k: float("nan") for k in SUMMARY_METRICS if k != "elapsed"}


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Use F1=0 for finite graph estimates with no true-positive edges."""

    out = dict(metrics)
    if _is_number(out.get("shd")):
        if not _is_number(out.get("F1")):
            out["F1"] = 0.0
        if not _is_number(out.get("precision")) and _is_number(out.get("recall")) and float(out["recall"]) == 0.0:
            out["precision"] = 0.0
    return out


def _normalise_endpoint_matrix(W_est: np.ndarray) -> np.ndarray:
    """Convert causal-learn style endpoint matrices to one edge per pair."""

    B = np.asarray(W_est)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Estimated graph must be a square matrix.")
    if not ((B == 0) | (B == 1) | (B == -1)).all():
        return (B != 0).astype(int)
    if not (B == -1).any():
        return (B > 0).astype(int)

    C = np.zeros(B.shape, dtype=int)
    d = B.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            left = B[i, j]
            right = B[j, i]
            if left == 0 and right == 0:
                continue
            if left == 1 and right == -1:
                C[i, j] = 1
            elif left == -1 and right == 1:
                C[j, i] = 1
            elif left == -1 and right == -1:
                C[i, j] = -1
            elif left == 1 and right == 1:
                C[i, j] = -1
            elif left == 1 and right == 0:
                C[i, j] = 1
            elif left == 0 and right == 1:
                C[j, i] = 1
            elif left == -1 and right == 0:
                C[i, j] = -1
            elif left == 0 and right == -1:
                C[j, i] = -1
    return C


def _accuracy_from_adjacency(W_est: np.ndarray, B_true: np.ndarray) -> dict[str, Any]:
    """Count edge-level metrics even when an estimate is not a valid DAG."""

    B = _normalise_endpoint_matrix(np.asarray(W_est))
    T = np.asarray(B_true)
    if B.shape != T.shape or B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Estimated and true graphs must be square matrices with the same shape.")

    B_unique = B.copy()
    if ((B_unique == 1) & (B_unique.T == 1)).any():
        for i in range(B_unique.shape[0]):
            for j in range(B_unique.shape[1]):
                if B_unique[i, j] == B_unique[j, i] == 1:
                    B_unique[i, j] = -1
                    B_unique[j, i] = 0
    if ((B_unique == -1) & (B_unique.T == -1)).any():
        for i in range(B_unique.shape[0]):
            for j in range(B_unique.shape[1]):
                if B_unique[i, j] == B_unique[j, i] == -1:
                    B_unique[i, j] = -1
                    B_unique[j, i] = 0

    d = T.shape[0]
    pred_und = np.flatnonzero(B_unique == -1)
    pred = np.flatnonzero(B_unique == 1)
    cond = np.flatnonzero(T)
    cond_reversed = np.flatnonzero(T.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    true_pos = np.intersect1d(pred, cond, assume_unique=True) if len(pred) else np.array([])
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True) if len(pred_und) else np.array([])
    true_pos = np.concatenate([true_pos, true_pos_und])

    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True) if len(extra) else np.array([])
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)

    pred_skel = ((B != 0) | (B.T != 0)).astype(int)
    true_skel = ((T != 0) | (T.T != 0)).astype(int)
    lower = np.tril_indices(d, k=-1)
    extra_skel = int(((pred_skel[lower] == 1) & (true_skel[lower] == 0)).sum())
    missing_skel = int(((pred_skel[lower] == 0) & (true_skel[lower] == 1)).sum())
    shd = extra_skel + missing_skel + len(reverse)

    precision = float(len(true_pos)) / max(pred_size, 1)
    recall = float(len(true_pos)) / max(len(cond), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    metrics = {
        "nnz": pred_size,
        "fdr": float(len(reverse) + len(false_pos)) / max(pred_size, 1),
        "tpr": float(len(true_pos)) / max(len(cond), 1),
        "fpr": float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "shd": shd,
    }
    return {key: round(value, 4) for key, value in metrics.items()}


def _read_progress(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    df = pd.read_csv(path)
    for col in PROGRESS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[PROGRESS_COLUMNS]


def _upsert_progress(path: Path, row: dict[str, Any]) -> None:
    """Store one canonical row per dataset/model/seed identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_progress(path)
    if not current.empty:
        seed_values = pd.to_numeric(current["seed"], errors="coerce")
        duplicate = (
            (current["dataset"].astype(str) == str(row["dataset"]))
            & (current["model"].astype(str) == str(row["model"]))
            & (seed_values == int(row["seed"]))
        )
        current = current.loc[~duplicate]
    updated = pd.concat(
        [current, pd.DataFrame([row]).reindex(columns=PROGRESS_COLUMNS)],
        ignore_index=True,
    )
    updated["run_idx"] = pd.to_numeric(updated["run_idx"], errors="coerce")
    updated = updated.sort_values(["run_idx", "seed"], kind="stable")
    temporary = path.with_suffix(path.suffix + ".tmp")
    updated.to_csv(temporary, index=False)
    temporary.replace(path)


def _successful_seed_ids(dag_progress: pd.DataFrame, cpdag_progress: pd.DataFrame) -> set[int]:
    def successful(frame: pd.DataFrame) -> set[int]:
        ok = frame.loc[frame["status"].astype(str) == "ok", "seed"]
        return set(pd.to_numeric(ok, errors="coerce").dropna().astype(int))

    return successful(dag_progress) & successful(cpdag_progress)


def _write_summary(path: Path, progress: pd.DataFrame) -> None:
    ok = progress[progress["status"] == "ok"].copy()
    if ok.empty:
        cols = ["dataset", "model", "n_runs"]
        for metric in SUMMARY_METRICS:
            cols += [f"{metric}_mean", f"{metric}_std"]
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return

    rows: list[dict[str, Any]] = []
    for (dataset, model), group in ok.groupby(["dataset", "model"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "n_runs": int(pd.to_numeric(group["seed"], errors="coerce").nunique()),
        }
        for metric in SUMMARY_METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_npz(
    path: Path,
    *,
    W_est: np.ndarray,
    B_true: np.ndarray,
    metadata: dict[str, Any],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "W_est": np.asarray(W_est),
        "B_true": np.asarray(B_true),
        "metadata": np.array(json.dumps(metadata, sort_keys=True, default=_json_default)),
    }
    if extra_arrays:
        arrays.update({name: np.asarray(value) for name, value in extra_arrays.items()})
    np.savez_compressed(path, **arrays)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialise {type(value).__name__}")


def _estimate_to_cpdag_for_metrics(W_est: np.ndarray):
    """Compatibility copy of UAI runner's endpoint-preserving CPDAG conversion."""

    from utils.graph_utils import dag2cpdag, is_dag

    W = np.asarray(W_est)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Estimated graph must be a square matrix.")
    if not ((W == 0) | (W == 1) | (W == -1)).all():
        return dag2cpdag((W > 0).astype(int))
    if not (W == -1).any():
        B = (W > 0).astype(int)
        if is_dag(B):
            return dag2cpdag(B)
        return B

    C = np.zeros(W.shape, dtype=int)
    d = W.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            left = W[i, j]
            right = W[j, i]
            if left == 0 and right == 0:
                continue
            if left == 1 and right == -1:
                C[i, j] = 1
            elif left == -1 and right == 1:
                C[j, i] = 1
            elif left == -1 and right == -1:
                C[i, j] = -1
                C[j, i] = -1
            elif left == 1 and right == 1:
                C[i, j] = 1
                C[j, i] = 1
            elif left == 1 and right == 0:
                C[i, j] = 1
            elif left == 0 and right == 1:
                C[j, i] = 1
            elif left == -1 and right == 0:
                C[i, j] = -1
            elif left == 0 and right == -1:
                C[j, i] = -1
    return C


def _row(
    *,
    spec: DatasetSpec,
    model: str,
    elapsed: float,
    metrics: dict[str, Any],
    run_idx: int,
    seed: int,
    status: str,
    error: str,
    raw_graph_path: Path | None,
    true_graph_path: Path | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": spec.name,
        "model": model,
        "elapsed": elapsed,
        "run_idx": run_idx,
        "rep": run_idx + 1,
        "seed": seed,
        "status": status,
        "error": error,
        "raw_graph_path": str(raw_graph_path or ""),
        "true_graph_path": str(true_graph_path or ""),
    }
    for metric in SUMMARY_METRICS:
        if metric == "elapsed":
            continue
        out[metric] = metrics.get(metric, float("nan"))
    if extra:
        out.update(extra)
    return out


def _dataset_specs(names: list[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for name in names:
        n = name.lower()
        if n == "cancer":
            specs.append(DatasetSpec("cancer", "bnlearn", 5))
        elif n == "earthquake":
            specs.append(DatasetSpec("earthquake", "bnlearn", 5))
        elif n == "survey":
            specs.append(DatasetSpec("survey", "bnlearn", 6))
        elif n == "asia":
            specs.append(DatasetSpec("asia", "bnlearn", 8))
        elif n in {"er5", "er_5"}:
            specs.append(DatasetSpec("er5", "synthetic", 5, "ER"))
        elif n in {"er8", "er_8"}:
            specs.append(DatasetSpec("er8", "synthetic", 8, "ER"))
        elif n in {"sf5", "sf_5"}:
            specs.append(DatasetSpec("sf5", "synthetic", 5, "SF"))
        elif n in {"sf8", "sf_8"}:
            specs.append(DatasetSpec("sf8", "synthetic", 8, "SF"))
        else:
            raise SystemExit(f"Unknown dataset '{name}'. Use one of: {', '.join(PAPER_DATASETS)}")
    return specs


def _load_data(spec: DatasetSpec, *, sample_size: int, seed: int, args: argparse.Namespace):
    from sklearn.preprocessing import StandardScaler
    from utils.data_utils import load_bnlearn_data_dag, simulate_dag, simulate_discrete_data, simulate_linear_continuous_data
    from utils.helpers import random_stability

    if spec.source == "bnlearn":
        std = bool(args.bn_standardise)
        return load_bnlearn_data_dag(
            spec.name,
            args.bn_data_path,
            sample_size,
            seed=seed,
            standardise=std,
            print_info=False,
        )

    random_stability(seed)
    n_edges = int(args.edge_per_node) * int(spec.n_nodes)
    B_true = simulate_dag(d=spec.n_nodes, s0=n_edges, graph_type=str(spec.graph_type))
    truth_edges = set((int(i), int(j)) for i, j in zip(*np.where(B_true == 1)))
    if args.synthetic_sim_type == "discrete":
        X = simulate_discrete_data(
            num_of_nodes=spec.n_nodes,
            sample_size=sample_size,
            truth_DAG_directed_edges=truth_edges,
            random_seed=seed,
        )
        if args.synthetic_standardise:
            X = StandardScaler().fit_transform(X)
    else:
        X = simulate_linear_continuous_data(
            num_of_nodes=spec.n_nodes,
            sample_size=sample_size,
            truth_DAG_directed_edges=truth_edges,
            noise_type=args.noise_type,
            random_seed=seed,
        )
        if args.synthetic_standardise:
            X = StandardScaler().fit_transform(X)
    return X, B_true


def _write_aspcr_inputs(aspcr_data_dir: Path, spec: DatasetSpec, seed: int, X: np.ndarray, B_true: np.ndarray) -> tuple[Path, Path]:
    aspcr_data_dir.mkdir(parents=True, exist_ok=True)
    data_path = aspcr_data_dir / f"data_{spec.name}_{seed}.csv"
    true_path = aspcr_data_dir / f"true_graph_{spec.name}_{seed}.csv"
    pd.DataFrame(B_true).to_csv(true_path, index=False, header=False)
    pd.DataFrame(X).to_csv(data_path, index=False, header=False)
    return data_path, true_path


def expected_constraint_count(n_nodes: int) -> int:
    return math.comb(int(n_nodes), 2) * (2 ** (int(n_nodes) - 2))


def _is_dag_adjacency(adjacency: np.ndarray) -> bool:
    graph = np.asarray(adjacency)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        return False
    graph = graph != 0
    if np.diag(graph).any():
        return False
    indegree = graph.sum(axis=0).astype(int)
    queue = list(np.flatnonzero(indegree == 0))
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return visited == graph.shape[0]


def _aspcr_output_paths(data_path: Path, *, algorithm: str, model_space: str) -> dict[str, Path]:
    data_path = data_path.expanduser().resolve()
    if data_path.parent.name != "data":
        raise ValueError(f"ASPCR input must be inside a data directory: {data_path}")
    run_id = data_path.stem.removeprefix("data_")
    prefix = data_path.parent.parent / "results" / (
        f"aspcr_{safe_filename(algorithm)}_{safe_filename(model_space)}_{run_id}"
    )
    return {
        "directed": Path(f"{prefix}_directed.csv"),
        "bidirected": Path(f"{prefix}_bidirected.csv"),
        "tailtail": Path(f"{prefix}_tailtail.csv"),
        "constraints": Path(f"{prefix}_constraints.csv"),
        "diagnostics": Path(f"{prefix}_diagnostics.csv"),
        "time": Path(f"{prefix}_time.csv"),
    }


def _read_graph_component(path: Path, n_nodes: int, name: str) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"Missing ASPCR {name} artifact: {path}")
    component = pd.read_csv(path, header=None).to_numpy()
    if component.shape != (n_nodes, n_nodes):
        raise RuntimeError(
            f"ASPCR {name} component has shape {component.shape}, expected {(n_nodes, n_nodes)}"
        )
    if not np.isin(component, [0, 1]).all():
        raise RuntimeError(f"ASPCR {name} component is not binary: {path}")
    component = component.astype(int)
    if np.diag(component).any():
        raise RuntimeError(f"ASPCR {name} component contains a self-edge: {path}")
    return component


def _bool_series(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapping = {
        "true": True, "t": True, "1": True, "yes": True,
        "false": False, "f": False, "0": False, "no": False,
    }
    converted = values.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(values.loc[converted.isna()].astype(str).unique())
        raise RuntimeError(f"Invalid boolean values in ASPCR {column}: {bad}")
    return converted.astype(bool)


def _parse_index_set(value: Any) -> tuple[int, ...]:
    if pd.isna(value) or str(value).strip() == "":
        return ()
    return tuple(sorted(int(item) for item in str(value).split(";") if item != ""))


def _validate_constraint_coverage(constraints: pd.DataFrame, n_nodes: int) -> None:
    observed: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for row in constraints.itertuples(index=False):
        x, y = int(row.x), int(row.y)
        conditioning = _parse_index_set(row.conditioning_set)
        intervention = _parse_index_set(row.intervention_set)
        if not (1 <= x < y <= n_nodes):
            raise RuntimeError(f"Invalid ASPCR constraint pair ({x}, {y})")
        if intervention:
            raise RuntimeError("Matched ASPCR run contains a non-observational intervention set")
        if x in conditioning or y in conditioning or len(set(conditioning)) != len(conditioning):
            raise RuntimeError(f"Invalid conditioning set for ASPCR pair ({x}, {y}): {conditioning}")
        key = (x, y, conditioning, intervention)
        if key in observed:
            raise RuntimeError(f"Duplicate ASPCR constraint: {key}")
        observed.add(key)

    expected: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for x in range(1, n_nodes + 1):
        for y in range(x + 1, n_nodes + 1):
            available = [node for node in range(1, n_nodes + 1) if node not in {x, y}]
            for size in range(len(available) + 1):
                for conditioning in itertools.combinations(available, size):
                    expected.add((x, y, tuple(conditioning), ()))
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise RuntimeError(f"Incomplete ASPCR constraint coverage: missing={missing}, extra={extra}")


def _same_number(actual: Any, expected: Any, *, atol: float = 1e-9) -> bool:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError):
        return False
    if math.isnan(actual_float) and math.isnan(expected_float):
        return True
    return math.isclose(actual_float, expected_float, rel_tol=0.0, abs_tol=atol)


def load_and_validate_aspcr_outputs(
    data_path: Path,
    *,
    algorithm: str,
    model_space: str,
    n_nodes: int,
) -> ASPCRRunResult:
    paths = _aspcr_output_paths(data_path, algorithm=algorithm, model_space=model_space)
    G_hej = _read_graph_component(paths["directed"], n_nodes, "directed")
    G_bidirected = _read_graph_component(paths["bidirected"], n_nodes, "bidirected")
    G_tailtail = _read_graph_component(paths["tailtail"], n_nodes, "tail-tail")
    if not np.array_equal(G_bidirected, G_bidirected.T):
        raise RuntimeError("ASPCR bidirected component is not symmetric")
    if not np.array_equal(G_tailtail, G_tailtail.T):
        raise RuntimeError("ASPCR tail-tail component is not symmetric")

    W_est = G_hej.T
    graph_is_dag = _is_dag_adjacency(W_est)
    n_bidirected = int(G_bidirected.sum() // 2)
    n_tailtail = int(G_tailtail.sum() // 2)
    if model_space == "dag_sufficient":
        if not graph_is_dag:
            raise RuntimeError("DAG-mode ASPCR directed result contains a cycle")
        if n_bidirected:
            raise RuntimeError(f"DAG-mode ASPCR result has {n_bidirected} bidirected edges")
        if n_tailtail:
            raise RuntimeError(f"DAG-mode ASPCR result has {n_tailtail} tail-tail edges")

    if not paths["constraints"].exists():
        raise RuntimeError(f"Missing ASPCR constraint trace: {paths['constraints']}")
    constraints = pd.read_csv(paths["constraints"])
    required_columns = {
        "constraint_id", "x", "y", "conditioning_set", "intervention_set",
        "test_independent", "truth_independent", "final_independent",
        "tested_relation", "truth_relation", "final_graph_relation",
        "test_correct", "retained", "retained_true", "retained_false",
        "raw_weight", "asp_weight",
    }
    missing_columns = sorted(required_columns - set(constraints.columns))
    if missing_columns:
        raise RuntimeError(f"ASPCR constraint trace lacks columns: {missing_columns}")
    expected_total = expected_constraint_count(n_nodes)
    if len(constraints) != expected_total:
        raise RuntimeError(f"Expected {expected_total} ASPCR constraints, got {len(constraints)}")
    ids = pd.to_numeric(constraints["constraint_id"], errors="raise").astype(int)
    if ids.tolist() != list(range(1, expected_total + 1)):
        raise RuntimeError("ASPCR constraint IDs are not the complete ordered sequence")
    _validate_constraint_coverage(constraints, n_nodes)

    for column in [
        "test_independent", "truth_independent", "final_independent",
        "test_correct", "retained", "retained_true", "retained_false",
    ]:
        constraints[column] = _bool_series(constraints[column], column=column)
    tested = constraints["test_independent"]
    truth = constraints["truth_independent"]
    final = constraints["final_independent"]
    correct = tested == truth
    retained = tested == final
    if not (constraints["test_correct"] == correct).all():
        raise RuntimeError("ASPCR row-level test-correct accounting is inconsistent")
    if not (constraints["retained"] == retained).all():
        raise RuntimeError("ASPCR row-level retained accounting is inconsistent")
    if not (constraints["retained_true"] == (retained & correct)).all():
        raise RuntimeError("ASPCR row-level retained-true accounting is inconsistent")
    if not (constraints["retained_false"] == (retained & ~correct)).all():
        raise RuntimeError("ASPCR row-level retained-false accounting is inconsistent")
    relation = {True: "independent", False: "dependent"}
    for bool_column, relation_column in [
        ("test_independent", "tested_relation"),
        ("truth_independent", "truth_relation"),
        ("final_independent", "final_graph_relation"),
    ]:
        expected_relations = constraints[bool_column].map(relation)
        if not (constraints[relation_column].astype(str) == expected_relations).all():
            raise RuntimeError(f"ASPCR {relation_column} labels disagree with {bool_column}")

    counts: dict[str, int] = {
        "fact_total": expected_total,
        "fact_true": int(correct.sum()),
        "fact_false": int((~correct).sum()),
        "fact_retained": int(retained.sum()),
        "fact_retained_true": int((retained & correct).sum()),
        "fact_retained_false": int((retained & ~correct).sum()),
    }
    counts["fact_removed"] = counts["fact_total"] - counts["fact_retained"]
    counts["fact_removed_true"] = counts["fact_true"] - counts["fact_retained_true"]
    counts["fact_removed_false"] = counts["fact_false"] - counts["fact_retained_false"]
    fact_precision = (
        counts["fact_retained_true"] / counts["fact_retained"]
        if counts["fact_retained"] else float("nan")
    )
    fact_recall = counts["fact_retained_true"] / counts["fact_true"] if counts["fact_true"] else float("nan")
    fact_f1 = (
        0.0 if not math.isfinite(fact_precision) or not math.isfinite(fact_recall) or fact_precision + fact_recall == 0
        else 2 * fact_precision * fact_recall / (fact_precision + fact_recall)
    )
    asp_weights = pd.to_numeric(constraints["asp_weight"], errors="raise")
    recomputed_objective = int(asp_weights.loc[~retained].sum())

    if not paths["diagnostics"].exists():
        raise RuntimeError(f"Missing ASPCR diagnostics: {paths['diagnostics']}")
    diagnostic_table = pd.read_csv(paths["diagnostics"])
    if len(diagnostic_table) != 1:
        raise RuntimeError("ASPCR diagnostics must contain exactly one row")
    diagnostics = diagnostic_table.iloc[0].to_dict()
    required_diagnostics = set(counts) | {
        "fact_precision", "fact_recall", "fact_f1", "solver_objective",
        "recomputed_objective", "graph_is_dag", "n_directed", "n_bidirected",
        "n_tailtail", "elapsed_total", "testing_time", "encoding_time",
        "solving_time", "algorithm", "model_space", "encoding", "expected_constraints",
        "test", "weight", "prior_independence", "alpha",
    }
    missing_diagnostics = sorted(required_diagnostics - set(diagnostics))
    if missing_diagnostics:
        raise RuntimeError(f"ASPCR diagnostics lack fields: {missing_diagnostics}")
    for column, expected in counts.items():
        if not _same_number(diagnostics[column], expected):
            raise RuntimeError(f"ASPCR aggregate {column} disagrees with its constraint trace")
    for column, expected in {
        "fact_precision": fact_precision,
        "fact_recall": fact_recall,
        "fact_f1": fact_f1,
        "recomputed_objective": recomputed_objective,
        "solver_objective": recomputed_objective,
        "n_directed": int(W_est.sum()),
        "n_bidirected": n_bidirected,
        "n_tailtail": n_tailtail,
        "expected_constraints": expected_total,
    }.items():
        if not _same_number(diagnostics[column], expected):
            raise RuntimeError(f"ASPCR diagnostic {column} is inconsistent: {diagnostics[column]} != {expected}")
    if _bool_series(pd.Series([diagnostics["graph_is_dag"]]), column="graph_is_dag").iloc[0] != graph_is_dag:
        raise RuntimeError("ASPCR graph_is_dag diagnostic is inconsistent")
    if str(diagnostics["model_space"]) != model_space:
        raise RuntimeError("ASPCR model-space provenance is inconsistent")
    matching_configs = [
        config for config in ASPCR_METHODS.values()
        if config["algorithm"] == algorithm and config["model_space"] == model_space
    ]
    if str(diagnostics["algorithm"]) != algorithm:
        raise RuntimeError("ASPCR algorithm provenance is inconsistent")
    if algorithm == "log-weights":
        native_config = {
            "test": "bayes",
            "weight": "log",
            "prior_independence": 0.4,
            "alpha": 20.0,
        }
        for field, expected in native_config.items():
            actual = diagnostics[field]
            if isinstance(expected, str):
                matches = str(actual) == expected
            else:
                matches = _same_number(actual, expected)
            if not matches:
                raise RuntimeError(
                    f"ASPCR native {field} provenance is inconsistent: {actual!r} != {expected!r}"
                )
    if matching_configs and str(diagnostics["encoding"]) != str(matching_configs[0]["encoding"]):
        raise RuntimeError("ASPCR encoding provenance is inconsistent")

    if not paths["time"].exists():
        raise RuntimeError(f"Missing ASPCR total-time artifact: {paths['time']}")
    elapsed = float(pd.read_csv(paths["time"], header=None).iloc[0, 0])
    if not math.isfinite(elapsed) or elapsed < 0 or not _same_number(elapsed, diagnostics["elapsed_total"], atol=1e-6):
        raise RuntimeError("ASPCR elapsed-time artifact is invalid or inconsistent")

    diagnostics.update(counts)
    diagnostics.update({
        "fact_precision": fact_precision,
        "fact_recall": fact_recall,
        "fact_f1": fact_f1,
        "recomputed_objective": recomputed_objective,
        "graph_is_dag": graph_is_dag,
        "n_directed": int(W_est.sum()),
        "n_bidirected": n_bidirected,
        "n_tailtail": n_tailtail,
    })
    return ASPCRRunResult(
        W_est=W_est,
        G_directed=W_est,
        G_bidirected=G_bidirected,
        G_tailtail=G_tailtail,
        elapsed=elapsed,
        diagnostics=diagnostics,
        paths=paths,
    )


def _aspcr_progress_fields(
    *,
    data_path: Path,
    method: str,
    result: ASPCRRunResult | None = None,
) -> dict[str, Any]:
    config = ASPCR_METHODS[method]
    paths = _aspcr_output_paths(
        data_path,
        algorithm=str(config["algorithm"]),
        model_space=str(config["model_space"]),
    )
    fields: dict[str, Any] = {
        "aspcr_model_space": config["model_space"],
        "aspcr_encoding": config["encoding"],
        "aspcr_data_path": str(data_path.resolve()),
        "aspcr_directed_path": str(paths["directed"]),
        "aspcr_bidirected_path": str(paths["bidirected"]),
        "aspcr_tailtail_path": str(paths["tailtail"]),
        "aspcr_constraints_path": str(paths["constraints"]),
        "aspcr_diagnostics_path": str(paths["diagnostics"]),
        "aspcr_time_path": str(paths["time"]),
    }
    if result is None:
        return fields
    diagnostics = result.diagnostics
    fields.update({column: diagnostics[column] for column in FACT_COUNT_COLUMNS + FACT_METRIC_COLUMNS})
    fields.update({
        "aspcr_solver_objective": diagnostics["solver_objective"],
        "aspcr_recomputed_objective": diagnostics["recomputed_objective"],
        "aspcr_graph_is_dag": diagnostics["graph_is_dag"],
        "aspcr_n_directed": diagnostics["n_directed"],
        "aspcr_n_bidirected": diagnostics["n_bidirected"],
        "aspcr_n_tailtail": diagnostics["n_tailtail"],
        "aspcr_testing_time": diagnostics["testing_time"],
        "aspcr_encoding_time": diagnostics["encoding_time"],
        "aspcr_solving_time": diagnostics["solving_time"],
    })
    return fields


def _run_aspcr(data_path: Path, *, method: str, n_nodes: int, args: argparse.Namespace) -> ASPCRRunResult:
    if not args.aspcr_r_dir:
        raise RuntimeError("ASPCR requested but --aspcr-r-dir was not supplied")
    config = ASPCR_METHODS[method]
    aspcr_r_dir = Path(args.aspcr_r_dir).expanduser().resolve()
    if not (aspcr_r_dir / "load.R").exists():
        raise RuntimeError(f"ASPCR R directory has no load.R: {aspcr_r_dir}")
    wrapper = REPO_ROOT / "cd_algorithms" / "run_aspcr_csvdata.R"
    if not wrapper.exists():
        raise RuntimeError(f"Missing ASPCR wrapper: {wrapper}")

    rscript_value = str(args.rscript)
    rscript_found = shutil.which(rscript_value)
    rscript = Path(rscript_found or rscript_value).expanduser()
    if not rscript.exists():
        raise RuntimeError(
            f"Rscript not found: {rscript_value}. Set RSCRIPT or pass --rscript."
        )

    cmd = [
        str(rscript),
        "--vanilla",
        "-e",
        (
            f"setwd({json.dumps(str(aspcr_r_dir))}); "
            "source('load.R'); loud(); library(stringr); "
            f"source({json.dumps(str(wrapper))}); "
            f"runpipeexternal({json.dumps(str(data_path.resolve()))}, "
            f"{json.dumps(config['algorithm'])}, N_override={int(args.sample_size)}, "
            f"model_space={json.dumps(config['model_space'])})"
        ),
    ]
    env = os.environ.copy()
    executable_bin = str(Path(sys.executable).resolve().parent)
    path_prefixes = [executable_bin]
    if args.clingo_bin_dir:
        path_prefixes.insert(0, str(Path(args.clingo_bin_dir).expanduser().resolve()))
    r_home = env.get("R_HOME")
    if r_home:
        r_home_bin = Path(r_home).expanduser() / "bin"
        if r_home_bin.exists():
            path_prefixes.insert(0, str(r_home_bin.resolve()))
    env["PATH"] = os.pathsep.join(path_prefixes + [env.get("PATH", "")])
    if not shutil.which("clingo", path=env["PATH"]):
        raise RuntimeError("clingo is not available on the ASPCR subprocess PATH")
    subprocess.run(cmd, check=True, env=env)
    return load_and_validate_aspcr_outputs(
        data_path,
        algorithm=str(config["algorithm"]),
        model_space=str(config["model_space"]),
        n_nodes=n_nodes,
    )


def _compute_metrics(W_est: np.ndarray, B_true: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    from utils.graph_utils import DAGMetrics, is_dag

    B_true_arr = np.asarray(B_true)
    try:
        cpdag_est = _estimate_to_cpdag_for_metrics(W_est)
        try:
            cpdag_metrics = _sanitize_metrics(DAGMetrics(cpdag_est, B_true_arr, sid=False).metrics)
        except Exception:
            cpdag_metrics = _sanitize_metrics(_accuracy_from_adjacency(cpdag_est, B_true_arr))
    except Exception:
        cpdag_metrics = _sanitize_metrics(_accuracy_from_adjacency(np.asarray(W_est), B_true_arr))

    B_dag = (np.asarray(W_est) > 0).astype(int)
    bidirected = (B_dag == 1) & (B_dag.T == 1)
    if bidirected.any():
        B_dag[bidirected] = 0
    if is_dag(B_dag):
        dag_metrics = _sanitize_metrics(DAGMetrics(B_dag, B_true_arr, sid=False).metrics)
    else:
        dag_metrics = _sanitize_metrics(_accuracy_from_adjacency(B_dag, B_true_arr))
    return dag_metrics, cpdag_metrics


def _metadata(args: argparse.Namespace, specs: list[DatasetSpec], out_dir: Path) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "version": args.version,
        "methods": args.methods,
        "aspcr_method_configs": {
            method: ASPCR_METHODS[method] for method in args.methods if method in ASPCR_METHODS
        },
        "datasets": [s.__dict__ for s in specs],
        "n_runs": args.n_runs,
        "seeds": [args.seed_start + i for i in range(args.n_runs)],
        "sample_size": args.sample_size,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
        "edge_per_node": args.edge_per_node,
        "synthetic_sim_type": args.synthetic_sim_type,
        "noise_type": args.noise_type,
        "bn_data_path": args.bn_data_path,
        "bn_standardise": args.bn_standardise,
        "synthetic_standardise": args.synthetic_standardise,
        "aspcr_max_nodes": args.aspcr_max_nodes,
        "progress_columns": PROGRESS_COLUMNS,
        "aspcr_graph_convention": (
            "NPZ W_est/G_directed use row->column; R directed CSV uses G[child,parent]"
        ),
        "outputs": {
            "progress_dir": str(out_dir / "progress" / args.version),
            "estimated_graphs_dir": str(out_dir / "estimated_graphs" / args.version),
            "stored_dag_summary": str(out_dir / f"stored_results_{args.version}.csv"),
            "stored_cpdag_summary": str(out_dir / f"stored_results_{args.version}_cpdag.csv"),
            "metadata": str(out_dir / f"metadata_{args.version}.json"),
        },
    }


def _assert_metadata_compatible(existing: dict[str, Any], proposed: dict[str, Any]) -> None:
    identity_fields = [
        "version", "methods", "aspcr_method_configs", "datasets", "n_runs", "seeds",
        "sample_size", "test_alpha", "test_name", "edge_per_node",
        "synthetic_sim_type", "noise_type", "bn_data_path", "bn_standardise",
        "synthetic_standardise", "aspcr_max_nodes",
    ]
    mismatches = [
        field for field in identity_fields if existing.get(field) != proposed.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: existing={existing.get(field)!r}, requested={proposed.get(field)!r}"
            for field in mismatches
        )
        raise SystemExit(
            "Refusing to mix experiment configurations in one result version. "
            f"Use a new --version (or explicit --fresh) for: {details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched MPC/Shapley-PC/FGS/ASPCR baseline experiments for the MCS paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="paper_aspcr_dag_alpha001_n5000_50rep")
    parser.add_argument(
        "--methods", nargs="+", default=list(DEFAULT_METHODS),
        choices=["mpc", "spc", "fgs", "aspcr_log", "aspcr_log_dag", "random_edge"],
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_EXPERIMENT_DATASETS),
        help=f"Datasets from: {', '.join(PAPER_DATASETS)}",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--test-alpha", type=float, default=0.001)
    parser.add_argument("--test-name", choices=["fisherz", "chisq", "gsq", "kci", "fastkci", "rcit"], default="gsq")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bn-data-path", default="datasets")
    parser.add_argument("--bn-standardise", type=str_to_bool, default=True)
    parser.add_argument("--synthetic-standardise", type=str_to_bool, default=False)
    parser.add_argument("--synthetic-sim-type", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--noise-type", choices=["gaussian", "exponential"], default="gaussian")
    parser.add_argument("--edge-per-node", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Remove this version's progress/artifacts before running")
    parser.add_argument("--save-graphs", type=str_to_bool, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--hard-exit", action="store_true", help="Use os._exit at the end; useful after FGS/pycausal JVM use")
    parser.add_argument("--aspcr-r-dir", default=os.environ.get("ASPCR_R_DIR"), help="Directory containing ASPCR load.R")
    parser.add_argument("--aspcr-max-nodes", type=int, default=6)
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", str(DEFAULT_RSCRIPT)), help="Rscript executable")
    parser.add_argument(
        "--clingo-bin-dir",
        default=os.environ.get("CLINGO_BIN_DIR", str(DEFAULT_CLINGO_BIN_DIR)),
        help="Directory containing the clingo executable used by ASPCR",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fresh and args.resume:
        raise SystemExit("Use either --fresh or --resume, not both.")
    specs = _dataset_specs(args.datasets)
    out_dir = Path(args.results_dir)
    progress_dir = out_dir / "progress" / args.version
    graph_dir = out_dir / "estimated_graphs" / args.version
    aspcr_data_dir = out_dir / "aspcr_matched" / args.version / "data"
    aspcr_results_dir = out_dir / "aspcr_matched" / args.version / "results"
    aspcr_results_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for path in (progress_dir, graph_dir, aspcr_data_dir.parent):
            if path.exists():
                shutil.rmtree(path)
    progress_dir.mkdir(parents=True, exist_ok=True)
    if args.save_graphs:
        graph_dir.mkdir(parents=True, exist_ok=True)

    metadata = _metadata(args, specs, out_dir)
    metadata_path = out_dir / f"metadata_{args.version}.json"
    if metadata_path.exists() and not args.fresh:
        existing_metadata = json.loads(metadata_path.read_text())
        _assert_metadata_compatible(existing_metadata, metadata)
        metadata = existing_metadata
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    run_method = None
    if any(method not in ASPCR_METHODS and method != "random_edge" for method in args.methods):
        _install_notears_stub()
        from cd_algorithms.models import run_method as imported_run_method

        run_method = imported_run_method
    simulate_dag = None
    random_stability = None
    if "random_edge" in args.methods:
        from utils.data_utils import simulate_dag as imported_simulate_dag
        from utils.helpers import random_stability as imported_random_stability

        simulate_dag = imported_simulate_dag
        random_stability = imported_random_stability

    seeds = [args.seed_start + i for i in range(args.n_runs)]

    for spec in specs:
        for method in args.methods:
            display = DISPLAY_NAMES.get(method, method)
            ds_safe = safe_filename(spec.name)
            method_safe = safe_filename(method)
            dag_path = progress_dir / f"{ds_safe}__{method_safe}_dag.csv"
            cpdag_path = progress_dir / f"{ds_safe}__{method_safe}_cpdag.csv"
            dag_runs = _read_progress(dag_path)
            cpdag_runs = _read_progress(cpdag_path)
            successful_seeds = _successful_seed_ids(dag_runs, cpdag_runs) if args.resume else set()
            if args.resume and successful_seeds:
                print(
                    f"[resume] {spec.name}/{method}: {len(successful_seeds)}/{args.n_runs} "
                    "successful seed identities already complete",
                    flush=True,
                )

            for run_idx, seed in enumerate(seeds):
                if seed in successful_seeds:
                    continue
                raw_path = graph_dir / ds_safe / safe_filename(display) / f"run_{run_idx:04d}_seed_{seed}.npz"
                true_path: Path | None = None
                progress_extra: dict[str, Any] = {}
                aspcr_result: ASPCRRunResult | None = None
                extra_arrays: dict[str, np.ndarray] | None = None
                try:
                    X, B_true = _load_data(spec, sample_size=args.sample_size, seed=seed, args=args)
                    if method == "random_edge":
                        assert random_stability is not None and simulate_dag is not None
                        random_stability(seed)
                        start = datetime.now()
                        W_est = simulate_dag(d=B_true.shape[0], s0=int(np.asarray(B_true).sum()), graph_type="ER")
                        elapsed = (datetime.now() - start).total_seconds()
                    elif method in ASPCR_METHODS:
                        data_path, true_path = _write_aspcr_inputs(aspcr_data_dir, spec, seed, X, B_true)
                        progress_extra = _aspcr_progress_fields(data_path=data_path, method=method)
                        if spec.n_nodes > int(args.aspcr_max_nodes):
                            raise RuntimeError(
                                f"ASPCR skipped by policy: n_nodes={spec.n_nodes} > aspcr_max_nodes={args.aspcr_max_nodes}"
                            )
                        aspcr_result = _run_aspcr(data_path, method=method, n_nodes=spec.n_nodes, args=args)
                        W_est = aspcr_result.W_est
                        elapsed = aspcr_result.elapsed
                        progress_extra = _aspcr_progress_fields(
                            data_path=data_path,
                            method=method,
                            result=aspcr_result,
                        )
                        extra_arrays = {
                            "G_directed": aspcr_result.G_directed,
                            "G_bidirected": aspcr_result.G_bidirected,
                            "G_tailtail": aspcr_result.G_tailtail,
                        }
                    else:
                        assert run_method is not None
                        W_est, elapsed = run_method(
                            X,
                            method,
                            seed,
                            test_alpha=args.test_alpha,
                            test_name=args.test_name,
                            device=args.device,
                            scenario=f"{method}_{args.version}_{spec.name}",
                        )
                        if "Tensor" in str(type(W_est)):
                            W_est = np.asarray([list(i) for i in W_est])
                        if W_est is None:
                            raise RuntimeError("run_method returned W_est=None")

                    if args.save_graphs:
                        artifact_metadata: dict[str, Any] = {
                            "dataset": spec.name,
                            "source": spec.source,
                            "graph_type": spec.graph_type,
                            "n_nodes": spec.n_nodes,
                            "method": method,
                            "model": display,
                            "run_idx": run_idx,
                            "rep": run_idx + 1,
                            "seed": seed,
                            "sample_size": args.sample_size,
                            "test_alpha": args.test_alpha,
                            "test_name": args.test_name,
                            "edge_per_node": args.edge_per_node,
                            "synthetic_sim_type": args.synthetic_sim_type,
                        }
                        if aspcr_result is not None:
                            artifact_metadata["aspcr"] = {
                                "diagnostics": aspcr_result.diagnostics,
                                "paths": {name: str(path) for name, path in aspcr_result.paths.items()},
                                "directed_component_convention": "row->column",
                            }
                        _save_npz(
                            raw_path,
                            W_est=np.asarray(W_est),
                            B_true=np.asarray(B_true),
                            metadata=artifact_metadata,
                            extra_arrays=extra_arrays,
                        )
                    dag_metrics, cpdag_metrics = _compute_metrics(np.asarray(W_est), np.asarray(B_true))
                    raw_ref = raw_path if args.save_graphs else None
                    true_ref = true_path
                    dag_row = _row(
                        spec=spec,
                        model=display,
                        elapsed=float(elapsed),
                        metrics=dag_metrics,
                        run_idx=run_idx,
                        seed=seed,
                        status="ok",
                        error="",
                        raw_graph_path=raw_ref,
                        true_graph_path=true_ref,
                        extra=progress_extra,
                    )
                    cpdag_row = _row(
                        spec=spec,
                        model=display,
                        elapsed=float(elapsed),
                        metrics=cpdag_metrics,
                        run_idx=run_idx,
                        seed=seed,
                        status="ok",
                        error="",
                        raw_graph_path=raw_ref,
                        true_graph_path=true_ref,
                        extra=progress_extra,
                    )
                except Exception as exc:  # noqa: BLE001 - keep overnight jobs resumable.
                    if args.fail_fast:
                        raise
                    err = f"{type(exc).__name__}: {exc}"
                    dag_row = _row(
                        spec=spec,
                        model=display,
                        elapsed=float("nan"),
                        metrics=_empty_metrics(),
                        run_idx=run_idx,
                        seed=seed,
                        status="error",
                        error=err,
                        raw_graph_path=None,
                        true_graph_path=true_path,
                        extra=progress_extra,
                    )
                    cpdag_row = dict(dag_row)

                _upsert_progress(dag_path, dag_row)
                _upsert_progress(cpdag_path, cpdag_row)
                print(
                    f"[saved] dataset={spec.name} method={method} rep={run_idx + 1}/{args.n_runs} "
                    f"status={dag_row['status']}",
                    flush=True,
                )

            dag_runs = _read_progress(dag_path)
            cpdag_runs = _read_progress(cpdag_path)
            _write_summary(progress_dir / f"{ds_safe}__{method_safe}_dag_summary.csv", dag_runs)
            _write_summary(progress_dir / f"{ds_safe}__{method_safe}_cpdag_summary.csv", cpdag_runs)

    all_dag = pd.concat([_read_progress(p) for p in sorted(progress_dir.glob("*__*_dag.csv"))], ignore_index=True)
    all_cpdag = pd.concat([_read_progress(p) for p in sorted(progress_dir.glob("*__*_cpdag.csv"))], ignore_index=True)
    _write_summary(out_dir / f"stored_results_{args.version}.csv", all_dag)
    _write_summary(out_dir / f"stored_results_{args.version}_cpdag.csv", all_cpdag)
    print(f"[done] progress: {progress_dir}", flush=True)
    print(f"[done] graphs: {graph_dir}", flush=True)
    print(f"[done] DAG summary: {out_dir / f'stored_results_{args.version}.csv'}", flush=True)
    print(f"[done] CPDAG summary: {out_dir / f'stored_results_{args.version}_cpdag.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    if "--hard-exit" in sys.argv:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
