import argparse
import faulthandler
import fnmatch
import json
import logging
import os
import re
import signal
import socket
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


DAG_BASE_COLUMNS = [
    "dataset", "model", "elapsed", "nnz", "fdr", "tpr", "fpr",
    "precision", "recall", "F1",
    "adjacency_precision", "adjacency_recall", "adjacency_F1",
    "arrowhead_precision", "arrowhead_recall", "arrowhead_F1",
    "shd", "sid",
]
CPDAG_BASE_COLUMNS = [
    "dataset", "model", "elapsed", "nnz", "fdr", "tpr", "fpr",
    "precision", "recall", "F1",
    "adjacency_precision", "adjacency_recall", "adjacency_F1",
    "arrowhead_precision", "arrowhead_recall", "arrowhead_F1",
    "shd", "sid_low", "sid_high",
]
DAG_PROGRESS_COLUMNS = DAG_BASE_COLUMNS + ["run_idx", "seed"]
CPDAG_PROGRESS_COLUMNS = CPDAG_BASE_COLUMNS + ["run_idx", "seed"]

DAG_METRIC_MAP = [
    ("elapsed", "elapsed"),
    ("nnz", "nnz"),
    ("fdr", "fdr"),
    ("tpr", "tpr"),
    ("fpr", "fpr"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("F1", "F1"),
    ("adjacency_precision", "adjacency_precision"),
    ("adjacency_recall", "adjacency_recall"),
    ("adjacency_F1", "adjacency_F1"),
    ("arrowhead_precision", "arrowhead_precision"),
    ("arrowhead_recall", "arrowhead_recall"),
    ("arrowhead_F1", "arrowhead_F1"),
    ("shd", "shd"),
    ("sid", "SID"),
]
CPDAG_METRIC_MAP = [
    ("elapsed", "elapsed"),
    ("nnz", "nnz"),
    ("fdr", "fdr"),
    ("tpr", "tpr"),
    ("fpr", "fpr"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("F1", "F1"),
    ("adjacency_precision", "adjacency_precision"),
    ("adjacency_recall", "adjacency_recall"),
    ("adjacency_F1", "adjacency_F1"),
    ("arrowhead_precision", "arrowhead_precision"),
    ("arrowhead_recall", "arrowhead_recall"),
    ("arrowhead_F1", "arrowhead_F1"),
    ("shd", "shd"),
    ("sid_low", "SID_low"),
    ("sid_high", "SID_high"),
]

DAG_SUMMARY_COLUMNS = ["dataset", "model"] + [
    f"{dst}_{stat}" for _, dst in DAG_METRIC_MAP for stat in ("mean", "std")
]
CPDAG_SUMMARY_COLUMNS = ["dataset", "model"] + [
    f"{dst}_{stat}" for _, dst in CPDAG_METRIC_MAP for stat in ("mean", "std")
]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_existing_summary(path: Path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and hasattr(arr, "item"):
        arr = arr.item()
    if isinstance(arr, pd.DataFrame):
        df = arr
    elif isinstance(arr, np.ndarray):
        try:
            df = pd.DataFrame(arr, columns=columns)
        except Exception:
            try:
                df = pd.DataFrame(arr.tolist(), columns=columns)
            except Exception:
                logging.warning(
                    "Could not interpret summary file at %s; starting with empty frame.",
                    path,
                )
                return pd.DataFrame(columns=columns)
    else:
        logging.warning(
            "Unexpected data type in summary file at %s; starting with empty frame.",
            path,
        )
        return pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    return df[columns]


def load_progress_df(path: Path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    return df[columns]


def append_progress_row(path: Path, row: dict, columns):
    df_row = pd.DataFrame([row]).reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row.to_csv(path, mode="a", header=not path.exists(), index=False)


def format_run_indicator(run_idx: int, total_runs: int | None = None) -> str:
    run_no = int(run_idx) + 1
    if total_runs is None:
        return str(run_no)
    return f"{run_no}/{int(total_runs)}"


def log_run_start(
    *,
    dataset_name: str,
    model_name: str,
    run_idx: int,
    total_runs: int,
    seed: int,
    scenario: str | None = None,
) -> None:
    if scenario:
        logging.info(
            "[run] dataset=%s model=%s run=%s seed=%s scenario=%s",
            dataset_name,
            model_name,
            format_run_indicator(run_idx, total_runs),
            int(seed),
            scenario,
        )
    else:
        logging.info(
            "[run] dataset=%s model=%s run=%s seed=%s",
            dataset_name,
            model_name,
            format_run_indicator(run_idx, total_runs),
            int(seed),
        )


def summarise_results(df: pd.DataFrame, metric_pairs):
    if df.empty:
        return pd.DataFrame(columns=["dataset", "model"] + [
            f"{dst}_{stat}" for _, dst in metric_pairs for stat in ("mean", "std")
        ])
    agg_dict = {src: ["mean", "std"] for src, _ in metric_pairs}
    summary = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(agg_dict)
        .round(2)
    )
    summary.columns = ["dataset", "model"] + [
        f"{dst}_{stat}" for _, dst in metric_pairs for stat in ("mean", "std")
    ]
    return summary


def save_summary_tables(base_path: Path, version: str, dag_df: pd.DataFrame, cpdag_df: pd.DataFrame):
    np.save(
        base_path / f"stored_results_{version}.npy",
        dag_df.reindex(columns=DAG_SUMMARY_COLUMNS).to_numpy(),
    )
    np.save(
        base_path / f"stored_results_{version}_cpdag.npy",
        cpdag_df.reindex(columns=CPDAG_SUMMARY_COLUMNS).to_numpy(),
    )


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _diff_dicts(old: dict, new: dict, *, ignore_keys: set[str] | None = None) -> dict[str, dict[str, object]]:
    ignore = set(ignore_keys or set())
    diffs: dict[str, dict[str, object]] = {}
    for key in sorted(set(old) | set(new)):
        if key in ignore:
            continue
        left = old.get(key)
        right = new.get(key)
        if left != right:
            diffs[str(key)] = {"old": _json_safe(left), "new": _json_safe(right)}
    return diffs


def record_run_manifest(*, results_path: Path, version: str, args: argparse.Namespace, script_path: str) -> Path:
    manifest_path = results_path / f"run_config_{version}.json"
    args_payload = _json_safe(vars(args))
    attempt = {
        "started_at": datetime.now().isoformat(),
        "pid": int(os.getpid()),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "argv": [str(x) for x in sys.argv],
        "args": args_payload,
    }
    runtime_only_keys = {"resume", "load_res", "save_res"}
    manifest: dict[str, Any]
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                manifest = {}
        except Exception:
            manifest = {}
    else:
        manifest = {}

    baseline_args = manifest.get("baseline_args")
    if not isinstance(baseline_args, dict):
        baseline_args = dict(args_payload)

    drift_from_baseline = _diff_dicts(baseline_args, args_payload)
    meaningful_drift = _diff_dicts(
        baseline_args,
        args_payload,
        ignore_keys=runtime_only_keys,
    )
    attempt["diff_from_baseline"] = drift_from_baseline
    attempt["meaningful_diff_from_baseline"] = meaningful_drift
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    attempts.append(attempt)

    manifest = {
        "version": version,
        "script_path": str(Path(script_path).resolve()),
        "results_dir": str(results_path.resolve()),
        "created_at": manifest.get("created_at") or attempt["started_at"],
        "updated_at": attempt["started_at"],
        "baseline_args": baseline_args,
        "latest_args": args_payload,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    _write_json_atomic(manifest_path, manifest)

    if meaningful_drift:
        logging.warning(
            "[run-config] current invocation differs from baseline for version=%s; changed_keys=%s manifest=%s",
            version,
            sorted(meaningful_drift.keys()),
            manifest_path,
        )
    else:
        logging.info(
            "[run-config] recorded invocation %d for version=%s at %s",
            len(attempts),
            version,
            manifest_path,
        )
    return manifest_path


def _clean_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_metric_subset(metrics: dict, keys: list[str]) -> dict:
    return {key: _clean_scalar(metrics.get(key)) for key in keys}


def _profile_float(profile: dict, key: str) -> float:
    try:
        return float(profile.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _fmt_sec(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}s"
    except Exception:
        return "n/a"


def build_run_summary(
    *,
    dataset_name: str,
    model_name: str,
    scenario: str | None,
    run_idx: int,
    total_runs: int | None,
    seed: int,
    elapsed: float,
    run_details: dict | None,
    dag_metrics: dict,
    cpdag_metrics: dict,
    dag_eval_status: dict | None = None,
    cpdag_eval_status: dict | None = None,
) -> dict:
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "run_idx": int(run_idx),
        "run_total": (int(total_runs) if total_runs is not None else None),
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "scenario": scenario,
        "dag": _clean_metric_subset(
            dag_metrics,
            [
                "precision", "recall", "F1",
                "adjacency_precision", "adjacency_recall", "adjacency_F1",
                "arrowhead_precision", "arrowhead_recall", "arrowhead_F1",
                "shd", "sid",
            ],
        ),
        "cpdag": _clean_metric_subset(
            cpdag_metrics,
            [
                "precision", "recall", "F1",
                "adjacency_precision", "adjacency_recall", "adjacency_F1",
                "arrowhead_precision", "arrowhead_recall", "arrowhead_F1",
                "shd", "sid_low", "sid_high",
            ],
        ),
    }
    if isinstance(dag_eval_status, dict) and dag_eval_status:
        summary["dag_eval_status"] = _json_safe(dag_eval_status)
    if isinstance(cpdag_eval_status, dict) and cpdag_eval_status:
        summary["cpdag_eval_status"] = _json_safe(cpdag_eval_status)
    if not isinstance(run_details, dict):
        return summary

    profile = run_details.get("solver_profile")
    if not isinstance(profile, dict):
        summary["remove_n"] = _clean_scalar(run_details.get("remove_n"))
        return summary

    satcheck_records = list(profile.get("satcheck_records") or [])
    satcheck_secs: list[float] = []
    satcheck_statuses: Counter[str] = Counter()
    satcheck_labels: Counter[str] = Counter()
    for record in satcheck_records:
        if not isinstance(record, dict):
            continue
        try:
            satcheck_secs.append(float(record.get("sec", 0.0) or 0.0))
        except Exception:
            pass
        status = str(record.get("status", "")).lower()
        label = str(record.get("label", ""))
        if status:
            satcheck_statuses[status] += 1
        if label:
            satcheck_labels[label] += 1

    compile_sec = _profile_float(profile, "compile_sec_total")
    gen_sec = _profile_float(profile, "write_gen_lp_sec")
    load_encoding_sec = _profile_float(profile, "load_encoding_sec")
    load_facts_sec = _profile_float(profile, "load_facts_sec")
    load_wc_sec = _profile_float(profile, "load_wc_sec")
    add_bayesball_sec = _profile_float(profile, "add_bayesball_sec")
    build_sec = compile_sec + gen_sec + load_encoding_sec + load_facts_sec + load_wc_sec + add_bayesball_sec
    ground_sec = _profile_float(profile, "ground_sec_total")
    final_solve_sec = _profile_float(profile, "final_opt_sec")
    satcheck_total_sec = sum(satcheck_secs) if satcheck_secs else _profile_float(profile, "sat_check_sec_total")
    first_solve_sec = max(
        0.0,
        _profile_float(profile, "solve_sec_total") - _profile_float(profile, "sat_check_sec_total") - final_solve_sec,
    )

    summary.update(
        {
            "scenario": run_details.get("scenario"),
            "remove_n": _clean_scalar(profile.get("remove_n", run_details.get("remove_n"))),
            "best_unsat_removed": _clean_scalar(profile.get("best_unsat_removed")),
            "best_sat_removed": _clean_scalar(profile.get("best_sat_removed")),
            "approximate_remove_search": bool(profile.get("remove_search_approximate", False)),
            "satcheck_resume": bool(profile.get("satcheck_resume", False)),
            "solver_backend": profile.get("solver_backend"),
            "build_sec": build_sec,
            "build_breakdown": {
                "compile_sec": compile_sec,
                "gen_facts_sec": gen_sec,
                "load_encoding_sec": load_encoding_sec,
                "load_facts_sec": load_facts_sec,
                "load_wc_sec": load_wc_sec,
                "add_bayesball_sec": add_bayesball_sec,
            },
            "ground_sec": ground_sec,
            "first_solve_sec": first_solve_sec,
            "final_solve_sec": final_solve_sec,
            "final_solve_config": {
                "timeout_sec": _clean_scalar(profile.get("final_solve_timeout")),
                "opt_mode": _clean_scalar(profile.get("final_solve_opt_mode")),
                "n_models": _clean_scalar(profile.get("final_solve_n_models")),
            },
            "satchecks": {
                "calls": int(len(satcheck_records)),
                "total_sec": satcheck_total_sec,
                "avg_sec": (satcheck_total_sec / len(satcheck_records)) if satcheck_records else 0.0,
                "min_sec": min(satcheck_secs) if satcheck_secs else 0.0,
                "max_sec": max(satcheck_secs) if satcheck_secs else 0.0,
                "sat": int(satcheck_statuses.get("sat", 0)),
                "unsat": int(satcheck_statuses.get("unsat", 0)),
                "unknown": int(satcheck_statuses.get("unknown", 0)),
                "labels": dict(sorted(satcheck_labels.items())),
            },
        }
    )
    fact_keys = [
        "facts_total", "removed_fact_count", "kept_fact_count",
        "removed_indep_count", "removed_dep_count",
        "kept_indep_count", "kept_dep_count",
        "removed_truth_counts", "kept_truth_counts",
        "fully_released_pair_count", "fully_released_pairs",
        "removed_fact_keys", "removed_true_fact_keys",
        "removed_false_fact_keys", "removed_unknown_fact_keys",
        "removed_pair_counts", "pair_fact_counts_total",
    ]
    fact_summary = {k: profile.get(k) for k in fact_keys if k in profile}
    if fact_summary:
        summary["facts"] = fact_summary

    reground_keys = [
        "pre_grounding", "disable_reground", "skeleton_rules_reduction",
        "block_edge_mode", "block_edge_initial_true_count", "block_edge_frozen_pair_count",
        "release_event_count", "release_events",
        "last_of_kind_release_count", "last_indep_release_count",
        "reground_eligible_count", "reground_performed_count", "reground_skipped_count",
        "reground_eligible_indep_count", "reground_performed_indep_count", "reground_skipped_indep_count",
        "reground_performed_pairs", "reground_skipped_pairs",
        "reground_compile_profiles", "reground_compile_count",
        "paths_added_initial", "pairs_considered_initial",
    ]
    reground_summary = {k: profile.get(k) for k in reground_keys if k in profile}
    if reground_summary:
        summary["reground"] = reground_summary
    return summary


def log_run_summary(summary: dict) -> None:
    run_str = format_run_indicator(
        int(summary.get("run_idx", 0) or 0),
        int(summary["run_total"]) if summary.get("run_total") is not None else None,
    )
    satchecks = summary.get("satchecks")
    if isinstance(satchecks, dict):
        logging.info(
            "[run-summary] dataset=%s model=%s run=%s seed=%s elapsed=%s build=%s ground=%s first_solve=%s "
            "satchecks=%d total=%s avg=%s min=%s max=%s sat=%d unsat=%d unknown=%d final_solve=%s "
            "remove_n=%s bracket=[%s,%s] approximate=%s resumed=%s",
            summary.get("dataset"),
            summary.get("model"),
            run_str,
            summary.get("seed"),
            _fmt_sec(summary.get("elapsed_sec")),
            _fmt_sec(summary.get("build_sec")),
            _fmt_sec(summary.get("ground_sec")),
            _fmt_sec(summary.get("first_solve_sec")),
            int(satchecks.get("calls", 0) or 0),
            _fmt_sec(satchecks.get("total_sec")),
            _fmt_sec(satchecks.get("avg_sec")),
            _fmt_sec(satchecks.get("min_sec")),
            _fmt_sec(satchecks.get("max_sec")),
            int(satchecks.get("sat", 0) or 0),
            int(satchecks.get("unsat", 0) or 0),
            int(satchecks.get("unknown", 0) or 0),
            _fmt_sec(summary.get("final_solve_sec")),
            summary.get("remove_n"),
            summary.get("best_unsat_removed"),
            summary.get("best_sat_removed"),
            summary.get("approximate_remove_search"),
            summary.get("satcheck_resume"),
        )
        labels = satchecks.get("labels")
        if labels:
            logging.info("[run-summary] satcheck_labels=%s", labels)
    facts = summary.get('facts')
    if isinstance(facts, dict):
        logging.info(
            "[run-summary] facts total=%s removed=%s kept=%s removed_indep=%s removed_dep=%s fully_released_pairs=%s removed_truth=%s",
            facts.get('facts_total'),
            facts.get('removed_fact_count'),
            facts.get('kept_fact_count'),
            facts.get('removed_indep_count'),
            facts.get('removed_dep_count'),
            facts.get('fully_released_pair_count'),
            facts.get('removed_truth_counts'),
        )
    reground = summary.get('reground')
    if isinstance(reground, dict):
        logging.info(
            "[run-summary] reground pre_grounding=%s disable_reground=%s eligible=%s performed=%s skipped=%s indep_skipped=%s",
            reground.get('pre_grounding'),
            reground.get('disable_reground'),
            reground.get('reground_eligible_count'),
            reground.get('reground_performed_count'),
            reground.get('reground_skipped_count'),
            reground.get('reground_skipped_indep_count'),
        )
        build_breakdown = summary.get("build_breakdown")
        if build_breakdown:
            logging.info("[run-summary] build_breakdown=%s", build_breakdown)
        final_solve_config = summary.get("final_solve_config")
        if final_solve_config:
            logging.info("[run-summary] final_solve_config=%s", final_solve_config)
    else:
        logging.info(
            "[run-summary] dataset=%s model=%s run=%s seed=%s elapsed=%s",
            summary.get("dataset"),
            summary.get("model"),
            run_str,
            summary.get("seed"),
            _fmt_sec(summary.get("elapsed_sec")),
        )
    logging.info("[run-summary] dag=%s cpdag=%s", summary.get("dag"), summary.get("cpdag"))
    for graph_kind in ("dag", "cpdag"):
        eval_status = summary.get(f"{graph_kind}_eval_status")
        if not isinstance(eval_status, dict):
            continue
        for metric_name, status in sorted(eval_status.items()):
            if not isinstance(status, dict):
                continue
            if status.get("status") == "timeout":
                logging.warning(
                    "[run-summary] %s metric %s timed out after %ss (elapsed=%ss)",
                    graph_kind,
                    metric_name,
                    status.get("timeout_sec"),
                    status.get("elapsed_sec"),
                )
            elif status.get("status") == "error":
                logging.warning(
                    "[run-summary] %s metric %s errored: %s",
                    graph_kind,
                    metric_name,
                    status.get("error"),
                )


def archive_run_artifacts(
    *,
    results_path: Path,
    summary: dict,
    graph_artifacts: dict[str, np.ndarray] | None = None,
) -> Path | None:
    scenario = summary.get("scenario")
    if not scenario:
        return None
    seed = summary.get("seed")
    run_idx = int(summary.get("run_idx", 0) or 0)
    scenario_dir = results_path / str(scenario)

    archive_dir = scenario_dir / "runs" / f"run_{run_idx + 1:04d}_seed_{int(seed)}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "facts.lp",
        "facts_I.lp",
        "facts_wc.lp",
        "log.log",
        "satcheck_search.json",
        "satcheck_autotune.json",
        "heartbeat_final-solve.status",
    ]:
        src = scenario_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)

    summary_payload = dict(summary)
    if graph_artifacts:
        saved_graph_artifacts: dict[str, dict[str, Any]] = {}
        for name, array in graph_artifacts.items():
            if array is None:
                continue
            arr = np.asarray(array)
            filename = f"{name}.npy"
            np.save(archive_dir / filename, arr)
            saved_graph_artifacts[name] = {
                "file": filename,
                "shape": [int(x) for x in arr.shape],
                "dtype": str(arr.dtype),
            }
        if saved_graph_artifacts:
            summary_payload["graph_artifacts"] = saved_graph_artifacts

    with open(archive_dir / "run_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    return archive_dir


def install_signal_breadcrumbs(*, results_path: Path, version: str) -> Path:
    signal_log_path = results_path / f"signal_{version}.log"

    def _write_signal_breadcrumb(sig_name: str) -> None:
        try:
            if resource is None:
                maxrss_kib = None
            else:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                maxrss_kib = getattr(ru, "ru_maxrss", None)
        except Exception:
            maxrss_kib = None
        try:
            with open(signal_log_path, "a") as f:
                f.write(
                    f"\n--- {datetime.now().isoformat()} received {sig_name} "
                    f"(pid={os.getpid()}) ---\n"
                )
                if maxrss_kib is not None:
                    f.write(f"ru_maxrss_kib={maxrss_kib}\n")
                f.write("Stack trace (all threads):\n")
                faulthandler.dump_traceback(file=f, all_threads=True)
        except Exception:
            pass

    def _handle_sigterm(signum, frame):  # noqa: ARG001
        _write_signal_breadcrumb("SIGTERM")
        raise SystemExit(143)

    def _handle_sigint(signum, frame):  # noqa: ARG001
        _write_signal_breadcrumb("SIGINT")
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigint)
    return signal_log_path


def edge_class(d, e):
    if e == d:
        return "d"
    target = 1.5 * d
    if abs(e - round(target)) <= 1:
        return "1.5d"
    return None


def parse_causenet_meta(fn):
    match = re.match(r"dag_(\d+)_nodes_(\d+)_edges_(.*)\.bifxml$", fn)
    if not match:
        return None
    nodes = int(match.group(1))
    edges = int(match.group(2))
    tail = match.group(3)
    parts = tail.split("_") if "_" in tail else [tail]
    heur = parts[0] if parts else None
    gtype = parts[-1].lower() if parts else None
    return nodes, edges, heur, gtype


def keep_causenet_file(path, args):
    basename = os.path.basename(path)
    if args.names and basename not in args.names:
        return False
    if args.include and not any(s in basename for s in args.include):
        return False
    if args.glob and not any(fnmatch.fnmatch(basename, pat) for pat in args.glob):
        return False
    if args.regex and not any(re.search(rx, basename) for rx in args.regex):
        return False
    meta = parse_causenet_meta(basename)
    if meta is None:
        return True
    n_nodes, n_edges, heur, gtype = meta
    if args.nodes and n_nodes != args.nodes:
        return False
    if args.edges_class and edge_class(n_nodes, n_edges) != args.edges_class:
        return False
    if args.heur and heur != args.heur:
        return False
    if args.gtype and gtype != args.gtype:
        return False
    return True
