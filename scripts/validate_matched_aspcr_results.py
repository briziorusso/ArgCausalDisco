#!/usr/bin/env python3
"""Validate a completed matched ASPCR-DAG experiment version.

The checks intentionally reload every R trace and NPZ graph artifact, then
regenerate the matched input for each seed.  A report is only valid when both
DAG and CPDAG progress files contain one successful row for every requested
seed and all provenance agrees.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_matched_baseline_experiments as runner  # noqa: E402


DEFAULT_VERSION = "paper_aspcr_dag_alpha001_n5000_50rep"
DEFAULT_DATASETS = ("cancer", "earthquake", "survey", "er5", "sf5")
DEFAULT_METHOD = "aspcr_log_dag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all progress, graph, fact, and matched-input artifacts for ASPCR-DAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--method", choices=sorted(runner.ASPCR_METHODS), default=DEFAULT_METHOD)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--test-alpha", type=float, default=0.001)
    parser.add_argument("--no-regenerate", action="store_true", help="Skip deterministic input regeneration")
    parser.add_argument("--no-write", action="store_true", help="Print checks without writing the JSON report")
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _load_metadata(results_dir: Path, version: str) -> dict[str, Any]:
    path = results_dir / f"metadata_{version}.json"
    if not path.exists():
        raise RuntimeError(f"Missing experiment metadata: {path}")
    return json.loads(path.read_text())


def _regeneration_args(metadata: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        bn_standardise=bool(metadata.get("bn_standardise", True)),
        bn_data_path=metadata.get("bn_data_path", "datasets"),
        edge_per_node=int(metadata.get("edge_per_node", 2)),
        synthetic_sim_type=metadata.get("synthetic_sim_type", "discrete"),
        synthetic_standardise=bool(metadata.get("synthetic_standardise", False)),
        noise_type=metadata.get("noise_type", "gaussian"),
    )


def _seed_rows(frame: pd.DataFrame) -> dict[int, pd.Series]:
    rows: dict[int, pd.Series] = {}
    for _, row in frame.iterrows():
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        rows[seed] = row
    return rows


def _duplicate_seeds(frame: pd.DataFrame) -> list[int]:
    seeds = pd.to_numeric(frame.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    return sorted(int(seed) for seed in seeds.loc[seeds.duplicated(keep=False)].unique())


def _finite_stats(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return {"min": None, "mean": None, "max": None, "total": None}
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "total": float(finite.sum()),
    }


def _validate_npz(
    path: Path,
    *,
    result: runner.ASPCRRunResult,
    B_true: np.ndarray,
    dataset: str,
    seed: int,
    sample_size: int,
    method: str,
) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing Python graph artifact: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"W_est", "B_true", "G_directed", "G_bidirected", "G_tailtail", "metadata"}
        missing = required - set(artifact.files)
        if missing:
            raise RuntimeError(f"NPZ graph artifact lacks arrays: {sorted(missing)}")
        for name, expected in {
            "W_est": result.W_est,
            "G_directed": result.G_directed,
            "G_bidirected": result.G_bidirected,
            "G_tailtail": result.G_tailtail,
            "B_true": B_true,
        }.items():
            if not np.array_equal(artifact[name], expected):
                raise RuntimeError(f"NPZ {name} disagrees with its source artifact")
        metadata = json.loads(str(artifact["metadata"].item()))
    expected_metadata = {
        "dataset": dataset,
        "seed": seed,
        "sample_size": sample_size,
        "method": method,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"NPZ metadata {key}={metadata.get(key)!r}, expected {expected!r}")
    if "aspcr" not in metadata or "diagnostics" not in metadata["aspcr"]:
        raise RuntimeError("NPZ metadata lacks ASPCR provenance")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    metadata = _load_metadata(results_dir, args.version)
    config_errors: list[str] = []
    for key, expected in {
        "version": args.version,
        "n_runs": args.n_runs,
        "sample_size": args.sample_size,
        "test_alpha": args.test_alpha,
    }.items():
        if metadata.get(key) != expected:
            config_errors.append(f"metadata {key}={metadata.get(key)!r}, expected {expected!r}")
    expected_seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    if metadata.get("seeds") != expected_seeds:
        config_errors.append("metadata seeds do not match the requested seed identities")
    if args.method not in metadata.get("methods", []):
        config_errors.append(f"metadata does not list method {args.method}")
    metadata_datasets = [item.get("name") for item in metadata.get("datasets", [])]
    selected_metadata_datasets = [name for name in metadata_datasets if name in set(args.datasets)]
    if selected_metadata_datasets != args.datasets:
        config_errors.append(
            f"requested datasets={args.datasets!r} are not an ordered subset of "
            f"metadata datasets={metadata_datasets!r}"
        )
    if metadata.get("aspcr_method_configs", {}).get(args.method) != runner.ASPCR_METHODS[args.method]:
        config_errors.append(f"metadata ASPCR configuration for {args.method} is inconsistent")

    progress_dir = results_dir / "progress" / args.version
    method_config = runner.ASPCR_METHODS[args.method]
    regeneration_args = _regeneration_args(metadata)
    specs = {spec.name: spec for spec in runner._dataset_specs(args.datasets)}
    dataset_reports: dict[str, Any] = {}
    all_valid = not config_errors

    for dataset in args.datasets:
        dag_path = progress_dir / f"{runner.safe_filename(dataset)}__{runner.safe_filename(args.method)}_dag.csv"
        cpdag_path = progress_dir / f"{runner.safe_filename(dataset)}__{runner.safe_filename(args.method)}_cpdag.csv"
        errors: list[str] = []
        failed_rows: list[dict[str, Any]] = []
        if not dag_path.exists() or not cpdag_path.exists():
            missing_paths = [str(path) for path in (dag_path, cpdag_path) if not path.exists()]
            errors.append(f"missing progress files: {missing_paths}")
            dag = pd.DataFrame(columns=runner.PROGRESS_COLUMNS)
            cpdag = pd.DataFrame(columns=runner.PROGRESS_COLUMNS)
        else:
            dag = runner._read_progress(dag_path)
            cpdag = runner._read_progress(cpdag_path)

        dag_duplicates = _duplicate_seeds(dag)
        cpdag_duplicates = _duplicate_seeds(cpdag)
        if dag_duplicates or cpdag_duplicates:
            errors.append(f"duplicate seeds: DAG={dag_duplicates}, CPDAG={cpdag_duplicates}")
        dag_rows = _seed_rows(dag)
        cpdag_rows = _seed_rows(cpdag)
        unexpected_dag_seeds = sorted(set(dag_rows) - set(expected_seeds))
        unexpected_cpdag_seeds = sorted(set(cpdag_rows) - set(expected_seeds))
        if unexpected_dag_seeds or unexpected_cpdag_seeds:
            errors.append(
                f"unexpected seeds: DAG={unexpected_dag_seeds}, CPDAG={unexpected_cpdag_seeds}"
            )
        successful: list[int] = []
        for seed in expected_seeds:
            dag_row = dag_rows.get(seed)
            cpdag_row = cpdag_rows.get(seed)
            if dag_row is None or cpdag_row is None:
                continue
            if str(dag_row["status"]) == "ok" and str(cpdag_row["status"]) == "ok":
                successful.append(seed)
                for column in [
                    "elapsed", "raw_graph_path", "true_graph_path",
                    *runner.FACT_COUNT_COLUMNS, *runner.FACT_METRIC_COLUMNS,
                    *runner.ASPCR_PROVENANCE_COLUMNS,
                ]:
                    dag_value, cpdag_value = dag_row[column], cpdag_row[column]
                    if runner._same_number(dag_value, cpdag_value):
                        continue
                    if (pd.isna(dag_value) and pd.isna(cpdag_value)) or str(dag_value) == str(cpdag_value):
                        continue
                    errors.append(f"seed {seed}: DAG/CPDAG progress mismatch in {column}")
                    break
            else:
                failed_rows.append({
                    "seed": seed,
                    "dag_status": str(dag_row["status"]),
                    "cpdag_status": str(cpdag_row["status"]),
                    "error": str(dag_row.get("error", "")),
                })
        missing_seeds = sorted(set(expected_seeds) - set(successful) - {row["seed"] for row in failed_rows})

        run_diagnostics: list[dict[str, Any]] = []
        for seed in successful:
            row = dag_rows[seed]
            try:
                data_path = Path(str(row["aspcr_data_path"]))
                result = runner.load_and_validate_aspcr_outputs(
                    data_path,
                    algorithm=str(method_config["algorithm"]),
                    model_space=str(method_config["model_space"]),
                    n_nodes=specs[dataset].n_nodes,
                )
                saved_X = pd.read_csv(data_path, header=None).to_numpy()
                true_path = Path(str(row["true_graph_path"]))
                if not true_path.exists():
                    raise RuntimeError(f"Missing matched true graph: {true_path}")
                saved_B = pd.read_csv(true_path, header=None).to_numpy().astype(int)
                if not args.no_regenerate:
                    regenerated_X, regenerated_B = runner._load_data(
                        specs[dataset],
                        sample_size=args.sample_size,
                        seed=seed,
                        args=regeneration_args,
                    )
                    if not np.array_equal(np.asarray(regenerated_B).astype(int), saved_B):
                        raise RuntimeError("Regenerated true graph disagrees with the matched input")
                    if not np.allclose(np.asarray(regenerated_X), saved_X, rtol=1e-12, atol=1e-12):
                        raise RuntimeError("Regenerated sample disagrees with the matched input")
                raw_path = Path(str(row["raw_graph_path"]))
                _validate_npz(
                    raw_path,
                    result=result,
                    B_true=saved_B,
                    dataset=dataset,
                    seed=seed,
                    sample_size=args.sample_size,
                    method=args.method,
                )
                for column in runner.FACT_COUNT_COLUMNS + runner.FACT_METRIC_COLUMNS:
                    if not runner._same_number(row[column], result.diagnostics[column]):
                        raise RuntimeError(f"Progress field {column} disagrees with ASPCR diagnostics")
                run_diagnostics.append(result.diagnostics)
            except Exception as exc:  # noqa: BLE001 - report every invalid artifact.
                errors.append(f"seed {seed}: {type(exc).__name__}: {exc}")

        runtimes = [float(item["elapsed_total"]) for item in run_diagnostics]
        directed_counts = [float(item["n_directed"]) for item in run_diagnostics]
        fact_totals = {column: int(sum(int(item[column]) for item in run_diagnostics)) for column in runner.FACT_COUNT_COLUMNS}
        retained_total = fact_totals.get("fact_retained", 0)
        true_total = fact_totals.get("fact_true", 0)
        retained_true_total = fact_totals.get("fact_retained_true", 0)
        fact_precision = retained_true_total / retained_total if retained_total else None
        fact_recall = retained_true_total / true_total if true_total else None
        fact_f1 = (
            2 * fact_precision * fact_recall / (fact_precision + fact_recall)
            if fact_precision is not None and fact_recall is not None and fact_precision + fact_recall else 0.0
        )
        expected_constraints = runner.expected_constraint_count(specs[dataset].n_nodes)
        report = {
            "expected_runs": args.n_runs,
            "successful_progress_seeds": successful,
            "n_successful_progress_seeds": len(successful),
            "n_validated_artifacts": len(run_diagnostics),
            "missing_seeds": missing_seeds,
            "failed_rows": failed_rows,
            "duplicate_dag_seeds": dag_duplicates,
            "duplicate_cpdag_seeds": cpdag_duplicates,
            "unexpected_dag_seeds": unexpected_dag_seeds,
            "unexpected_cpdag_seeds": unexpected_cpdag_seeds,
            "expected_constraints_per_run": expected_constraints,
            "runtime_seconds": _finite_stats(runtimes),
            "directed_edges": _finite_stats(directed_counts),
            "all_graphs_dag": all(bool(item["graph_is_dag"]) for item in run_diagnostics),
            "total_bidirected_edges": int(sum(int(item["n_bidirected"]) for item in run_diagnostics)),
            "total_tailtail_edges": int(sum(int(item["n_tailtail"]) for item in run_diagnostics)),
            "fact_counts": fact_totals,
            "aggregate_fact_precision": fact_precision,
            "aggregate_fact_recall": fact_recall,
            "aggregate_fact_f1": fact_f1,
            "errors": errors,
        }
        report["valid"] = (
            len(successful) == args.n_runs
            and len(run_diagnostics) == args.n_runs
            and not missing_seeds
            and not failed_rows
            and not dag_duplicates
            and not cpdag_duplicates
            and not errors
            and report["all_graphs_dag"]
            and report["total_bidirected_edges"] == 0
            and report["total_tailtail_edges"] == 0
            and fact_totals.get("fact_total", 0) == args.n_runs * expected_constraints
        )
        all_valid = all_valid and bool(report["valid"])
        dataset_reports[dataset] = report

    return {
        "version": args.version,
        "method": args.method,
        "expected_seeds": expected_seeds,
        "sample_size": args.sample_size,
        "regenerated_inputs": not args.no_regenerate,
        "configuration_errors": config_errors,
        "datasets": dataset_reports,
        "valid": all_valid,
    }


def main() -> int:
    args = parse_args()
    report = validate(args)
    for dataset, item in report["datasets"].items():
        runtime = item["runtime_seconds"]
        print(
            f"{dataset}: progress={item['n_successful_progress_seeds']}/{item['expected_runs']} "
            f"validated={item['n_validated_artifacts']}/{item['expected_runs']} "
            f"constraints/run={item['expected_constraints_per_run']} "
            f"runtime_mean={runtime['mean']} valid={item['valid']}",
            flush=True,
        )
        for error in item["errors"]:
            print(f"  ERROR {error}", flush=True)
    if report["configuration_errors"]:
        for error in report["configuration_errors"]:
            print(f"CONFIG ERROR {error}", flush=True)
    if not args.no_write:
        report_path = Path(args.report_path) if args.report_path else Path(args.results_dir) / f"validation_{args.version}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report={report_path}", flush=True)
    print(f"VALID={report['valid']}", flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
