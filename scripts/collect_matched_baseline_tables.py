#!/usr/bin/env python3
"""Collect matched baseline runs into paper-ready LaTeX tables.

The script combines:
  * matched MPC/FGS/ASPCR progress files from scripts/run_matched_baseline_experiments.py
  * ABA-PC and OptABA-PC per-repetition summaries from final_mcs_experiments

It writes a compact comparison table plus the Welch-test tables used for bolding.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from scipy import stats
except Exception:
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)

METRICS: tuple[tuple[str, str, bool], ...] = (
    ("cpdag_shd", "CPDAG SHD $\\downarrow$", False),
    ("cpdag_f1", "CPDAG F1 $\\uparrow$", True),
    ("dag_shd", "DAG SHD $\\downarrow$", False),
    ("dag_f1", "DAG F1 $\\uparrow$", True),
    ("time_sec", "Time (s) $\\downarrow$", False),
)

CPDAG_RANGE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("cpdag_shd_avg", "Avg SHD $\\downarrow$", False),
    ("cpdag_shd_best", "Best SHD $\\downarrow$", False),
    ("cpdag_shd_worst", "Worst SHD $\\downarrow$", False),
    ("cpdag_f1_avg", "Avg F1 $\\uparrow$", True),
    ("cpdag_f1_best", "Best F1 $\\uparrow$", True),
    ("cpdag_f1_worst", "Worst F1 $\\uparrow$", True),
)

DAG_RANGE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("dag_shd_avg", "Avg SHD $\\downarrow$", False),
    ("dag_shd_best", "Best SHD $\\downarrow$", False),
    ("dag_shd_worst", "Worst SHD $\\downarrow$", False),
    ("dag_f1_avg", "Avg F1 $\\uparrow$", True),
    ("dag_f1_best", "Best F1 $\\uparrow$", True),
    ("dag_f1_worst", "Worst F1 $\\uparrow$", True),
)

MCS_STRUCTURE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("n_cpdags_compat", "Compat. CPDAGs $\\downarrow$", False),
    ("cpdag_adjacency_f1_avg", "CPDAG skel. F1 $\\uparrow$", True),
    ("cpdag_arrowhead_f1_avg", "CPDAG arrow F1 $\\uparrow$", True),
    ("adjacency_f1_avg", "DAG skel. F1 $\\uparrow$", True),
    ("arrowhead_f1_avg", "DAG arrow F1 $\\uparrow$", True),
    ("true_cpdag_in_compat", "True CPDAG in set $\\uparrow$", True),
    ("true_dag_in_compat", "True DAG in set $\\uparrow$", True),
)

OPT_VS_MPC_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("cpdag_shd_avg", "$\\Delta$ CPDAG SHD avg", False),
    ("cpdag_shd_best", "$\\Delta$ CPDAG SHD best", False),
    ("cpdag_f1_avg", "$\\Delta$ CPDAG F1 avg", True),
    ("cpdag_f1_best", "$\\Delta$ CPDAG F1 best", True),
    ("dag_shd_avg", "$\\Delta$ DAG SHD avg", False),
    ("dag_shd_best", "$\\Delta$ DAG SHD best", False),
    ("dag_f1_avg", "$\\Delta$ DAG F1 avg", True),
    ("dag_f1_best", "$\\Delta$ DAG F1 best", True),
)

METHOD_ORDER = ("ABA-PC", "OptABA-PC", "MPC", "FGS", "ASPCR-log")
FACT_METHOD_ORDER = ("Raw PC facts", "ABA-PC", "OptABA-PC")
FACT_METRICS: tuple[tuple[str, str], ...] = (
    ("total_facts", "Facts"),
    ("wrong_facts", "Wrong"),
    ("accepted_facts", "Kept"),
    ("accepted_wrong_facts", "Kept wrong"),
    ("removed_wrong_facts", "Removed wrong"),
    ("fact_precision", "Precision"),
    ("fact_recall", "Recall"),
    ("fact_f1", "F1"),
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    mcs_dir_match: re.Pattern[str] | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("cancer", "Cancer (5)", re.compile(r"^wc_sweep_bnlearn_cancer(?:_|$)")),
    DatasetSpec("earthquake", "Earthquake (5)", re.compile(r"^wc_sweep_bnlearn_earthquake(?:_|$)")),
    DatasetSpec("survey", "Survey (6)", re.compile(r"^wc_sweep_bnlearn_survey(?:_|$)")),
    DatasetSpec("asia", "Asia (8)", re.compile(r"^wc_sweep_bnlearn_asia(?:_|$)")),
    DatasetSpec("er5", "ER (5)", re.compile(r"^wc_sweep_5_2026_")),
    DatasetSpec("er8", "ER (8)", re.compile(r"^wc_sweep_8_2026_")),
    DatasetSpec("sf5", "SF (5)", re.compile(r"^wc_sweep_sf5_2026_")),
    DatasetSpec("sf8", "SF (8)", re.compile(r"^wc_sweep_sf8_2026_")),
)


def _is_num(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _fmt_num(x: float, *, digits: int = 3) -> str:
    if not _is_num(x):
        return "--"
    x = float(x)
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def _latex_sig_marker(marker: str | None) -> str:
    if marker == "***":
        return r"\ast\ast\ast"
    if marker == "**":
        return r"\ast\ast"
    if marker == "*":
        return r"\ast"
    if marker == ".":
        return r"\cdot"
    return ""


def _fmt_mean_std(mean: float, std: float, *, bold: bool = False, marker: str | None = None) -> str:
    if not _is_num(mean):
        return "--"
    body = rf"{_fmt_num(mean)}\pm{_fmt_num(std)}"
    if bold:
        body = rf"\mathbf{{{body}}}"
        marker_text = _latex_sig_marker(marker)
        if marker_text:
            body = rf"{body}^{{{marker_text}}}"
    return rf"${body}$"


def _fmt_t(x: float) -> str:
    if not _is_num(x):
        return "--"
    return f"${float(x):.2f}$"


def _fmt_p(x: float) -> str:
    if not _is_num(x):
        return "--"
    x = max(0.0, min(1.0, float(x)))
    if x < 0.001:
        return "$<0.001$"
    return f"${x:.3f}$"


def _sig_symbol(p: float) -> str:
    if not _is_num(p):
        return "--"
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return "ns"


def _mean(xs: Iterable[Any]) -> float:
    vals = [float(x) for x in xs if _is_num(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(xs: Iterable[Any]) -> float:
    vals = [float(x) for x in xs if _is_num(x)]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    return float(np.std(vals, ddof=1))


def _safe_div(num: Any, den: Any) -> float:
    if not (_is_num(num) and _is_num(den)):
        return float("nan")
    den_f = float(den)
    if den_f == 0.0:
        return float("nan")
    return float(num) / den_f


def _harmonic_f1(precision: Any, recall: Any) -> float:
    if not (_is_num(precision) and _is_num(recall)):
        return float("nan")
    p = float(precision)
    r = float(recall)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _time_value(row: dict[str, Any]) -> float:
    for key in ("method_time_sec", "wall_time_sec", "solve_time_sec"):
        if _is_num(row.get(key)):
            return float(row[key])
    return float("nan")


def _f1_value(value: Any, shd_value: Any) -> float:
    if _is_num(value):
        return float(value)
    if _is_num(shd_value):
        return 0.0
    return float("nan")


def _filter_aba_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if isinstance(r, dict) and r.get("solver") == "causalaba_increm" and r.get("opt_mode") == "optN"
    ]


def _filter_opt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if isinstance(r, dict)
        and r.get("encoding") == "inc"
        and r.get("objective") == "lex"
        and r.get("opt_strategy") == "bb"
        and r.get("opt_mode") == "optN"
        and r.get("reification") == "mus"
        and not str(r.get("status", "")).upper().startswith(("SKIPPED", "ERROR", "FAILED"))
        and (not _is_num(r.get("n_tests_total")) or float(r["n_tests_total"]) > 0)
    ]


def _find_run_dirs(root: Path, pattern: re.Pattern[str], *, include_partial: bool = False) -> list[Path]:
    if not root.exists():
        return []
    matches = sorted(p for p in root.iterdir() if p.is_dir() and pattern.search(p.name))
    completed = [
        p
        for p in matches
        if (p / "summary.json").exists() or (include_partial and (p / "summary.partial.json").exists())
    ]
    return sorted(completed, key=lambda p: (p.stat().st_mtime, p.name))


def _summary_path_for_run(run_dir: Path, *, include_partial: bool = False) -> Path | None:
    summary = run_dir / "summary.json"
    if summary.exists():
        return summary
    partial = run_dir / "summary.partial.json"
    if include_partial and partial.exists():
        return partial
    return None


def _seed_for_row(summary: dict[str, Any], row: dict[str, Any]) -> float:
    if not (_is_num(summary.get("seed")) and _is_num(row.get("rep"))):
        return float("nan")
    return float(int(summary["seed"]) + int(row["rep"]) - 1)


def _injected_wrong_fact_rate(summary: dict[str, Any]) -> float:
    value = summary.get("pct_wrong_facts")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid pct_wrong_facts value in MCS summary: {value!r}") from None


def _iter_mcs_method_rows(
    root: Path,
    *,
    include_partial: bool = False,
    allow_injected_wrong_facts: bool = False,
) -> Iterable[dict[str, Any]]:
    for spec in DATASETS:
        if spec.mcs_dir_match is None:
            continue
        for run_dir in _find_run_dirs(root, spec.mcs_dir_match, include_partial=include_partial):
            summary_path = _summary_path_for_run(run_dir, include_partial=include_partial)
            if summary_path is None:
                continue
            try:
                obj = json.loads(summary_path.read_text())
            except Exception:
                continue
            injected_rate = _injected_wrong_fact_rate(obj)
            if injected_rate > 0 and not allow_injected_wrong_facts:
                warnings.warn(
                    f"Excluding {summary_path}: pct_wrong_facts={injected_rate:g} altered the "
                    "ABA-PC/OptABA-PC inputs and is not matched by the MPC baseline.",
                    RuntimeWarning,
                )
                continue
            source_mtime = float(summary_path.stat().st_mtime)
            for method, rows in (
                ("ABA-PC", _filter_aba_rows(obj.get("baseline_results", []) or [])),
                ("OptABA-PC", _filter_opt_rows(obj.get("wc_results", []) or [])),
            ):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    rep_local = int(row["rep"]) if _is_num(row.get("rep")) else 0
                    yield {
                        "spec": spec,
                        "method": method,
                        "row": row,
                        "summary": obj,
                        "rep_local": rep_local,
                        "seed": _seed_for_row(obj, row),
                        "source": _portable_path(summary_path),
                        "source_mtime": source_mtime,
                        "pct_wrong_facts": injected_rate,
                    }


def _load_mcs_records(
    root: Path,
    *,
    include_partial: bool = False,
    allow_injected_wrong_facts: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for item in _iter_mcs_method_rows(
        root,
        include_partial=include_partial,
        allow_injected_wrong_facts=allow_injected_wrong_facts,
    ):
        spec = item["spec"]
        row = item["row"]
        cpdag_shd_avg = row.get("cpdag_shd_avg", np.nan)
        cpdag_f1_avg = _f1_value(row.get("cpdag_f1_avg", np.nan), cpdag_shd_avg)
        dag_shd_avg = row.get("shd_avg", np.nan)
        dag_f1_avg = _f1_value(row.get("f1_avg", np.nan), dag_shd_avg)
        records.append(
            {
                "dataset": spec.key,
                "dataset_label": spec.label,
                "method": item["method"],
                "rep": int(item["rep_local"]),
                "rep_local": int(item["rep_local"]),
                "seed": item["seed"],
                "cpdag_shd": cpdag_shd_avg,
                "cpdag_f1": cpdag_f1_avg,
                "dag_shd": dag_shd_avg,
                "dag_f1": dag_f1_avg,
                "cpdag_shd_avg": cpdag_shd_avg,
                "cpdag_shd_best": row.get("cpdag_shd_best", np.nan),
                "cpdag_shd_worst": row.get("cpdag_shd_worst", np.nan),
                "cpdag_f1_avg": cpdag_f1_avg,
                "cpdag_f1_best": _f1_value(row.get("cpdag_f1_best", np.nan), row.get("cpdag_shd_best", np.nan)),
                "cpdag_f1_worst": _f1_value(row.get("cpdag_f1_worst", np.nan), row.get("cpdag_shd_worst", np.nan)),
                "dag_shd_avg": dag_shd_avg,
                "dag_shd_best": row.get("shd_best", np.nan),
                "dag_shd_worst": row.get("shd_worst", np.nan),
                "dag_f1_avg": dag_f1_avg,
                "dag_f1_best": _f1_value(row.get("f1_best", np.nan), row.get("shd_best", np.nan)),
                "dag_f1_worst": _f1_value(row.get("f1_worst", np.nan), row.get("shd_worst", np.nan)),
                "n_cpdags_compat": row.get("n_cpdags_compat", np.nan),
                "n_dags_compat": row.get("n_dags_compat", np.nan),
                "adjacency_f1_avg": row.get("adjacency_f1_avg", np.nan),
                "adjacency_f1_best": row.get("adjacency_f1_best", np.nan),
                "adjacency_f1_worst": row.get("adjacency_f1_worst", np.nan),
                "arrowhead_f1_avg": row.get("arrowhead_f1_avg", np.nan),
                "arrowhead_f1_best": row.get("arrowhead_f1_best", np.nan),
                "arrowhead_f1_worst": row.get("arrowhead_f1_worst", np.nan),
                "cpdag_adjacency_f1_avg": row.get("cpdag_adjacency_f1_avg", np.nan),
                "cpdag_adjacency_f1_best": row.get("cpdag_adjacency_f1_best", np.nan),
                "cpdag_adjacency_f1_worst": row.get("cpdag_adjacency_f1_worst", np.nan),
                "cpdag_arrowhead_f1_avg": row.get("cpdag_arrowhead_f1_avg", np.nan),
                "cpdag_arrowhead_f1_best": row.get("cpdag_arrowhead_f1_best", np.nan),
                "cpdag_arrowhead_f1_worst": row.get("cpdag_arrowhead_f1_worst", np.nan),
                "true_cpdag_in_compat": row.get("true_cpdag_in_compat", np.nan),
                "true_dag_in_compat": row.get("true_dag_in_compat", np.nan),
                "time_sec": _time_value(row),
                "pct_wrong_facts": item["pct_wrong_facts"],
                "source": item["source"],
                "_source_mtime": item["source_mtime"],
            }
        )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values(["_source_mtime", "source"]).drop_duplicates(
            ["dataset", "method", "seed"], keep="last"
        )
        out = out.drop(columns=["_source_mtime"])
    return out


def _fact_counts_from_row(row: dict[str, Any], *, method: str) -> dict[str, float]:
    total = float(row.get("n_tests_total")) if _is_num(row.get("n_tests_total")) else float("nan")
    correct = float(row.get("n_tests_true")) if _is_num(row.get("n_tests_true")) else float("nan")
    wrong = total - correct if _is_num(total) and _is_num(correct) else float("nan")

    if method == "Raw PC facts":
        accepted = total
        accepted_correct = correct
        precision = _safe_div(correct, total)
        recall = 1.0 if _is_num(total) else float("nan")
        f1 = _harmonic_f1(precision, recall)
    else:
        accepted = float(row.get("n_tests_accepted")) if _is_num(row.get("n_tests_accepted")) else float("nan")
        accepted_correct = (
            float(row.get("n_tests_accepted_true")) if _is_num(row.get("n_tests_accepted_true")) else float("nan")
        )
        precision = _safe_div(accepted_correct, accepted)
        recall = _safe_div(accepted_correct, correct)
        f1 = float(row["accepted_fact_f1"]) if _is_num(row.get("accepted_fact_f1")) else _harmonic_f1(precision, recall)

    accepted_wrong = accepted - accepted_correct if _is_num(accepted) and _is_num(accepted_correct) else float("nan")
    removed = total - accepted if _is_num(total) and _is_num(accepted) else float("nan")
    removed_wrong = wrong - accepted_wrong if _is_num(wrong) and _is_num(accepted_wrong) else float("nan")
    return {
        "total_facts": total,
        "correct_facts": correct,
        "wrong_facts": wrong,
        "accepted_facts": accepted,
        "accepted_correct_facts": accepted_correct,
        "accepted_wrong_facts": accepted_wrong,
        "removed_facts": removed,
        "removed_wrong_facts": removed_wrong,
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": f1,
    }


def _load_mcs_fact_records(
    root: Path,
    *,
    include_partial: bool = False,
    allow_injected_wrong_facts: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for item in _iter_mcs_method_rows(
        root,
        include_partial=include_partial,
        allow_injected_wrong_facts=allow_injected_wrong_facts,
    ):
        spec = item["spec"]
        row = item["row"]
        base = {
            "dataset": spec.key,
            "dataset_label": spec.label,
            "seed": item["seed"],
            "rep_local": int(item["rep_local"]),
            "source": item["source"],
            "pct_wrong_facts": item["pct_wrong_facts"],
            "_source_mtime": item["source_mtime"],
        }
        method = item["method"]
        records.append({**base, "method": method, **_fact_counts_from_row(row, method=method)})

        raw_key = (spec.key, float(item["seed"]))
        raw_record = {**base, "method": "Raw PC facts", **_fact_counts_from_row(row, method="Raw PC facts")}
        old = raw_by_key.get(raw_key)
        if old is None or float(raw_record["_source_mtime"]) >= float(old["_source_mtime"]):
            raw_by_key[raw_key] = raw_record

    records.extend(raw_by_key.values())
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values(["_source_mtime", "source"]).drop_duplicates(
            ["dataset", "method", "seed"], keep="last"
        )
        out = out.drop(columns=["_source_mtime"])
    return out


def _load_progress_pair(cpdag_path: Path, dag_path: Path, *, source_version: str) -> pd.DataFrame:
    if not dag_path.exists():
        return pd.DataFrame()
    cpdag = pd.read_csv(cpdag_path)
    dag = pd.read_csv(dag_path)
    key_cols = ["dataset", "model", "run_idx", "rep", "seed"]
    keep = key_cols + [
        "status", "elapsed", "F1", "shd", "error", "raw_graph_path",
        "aspcr_constraints_path",
    ]
    cpdag = cpdag[[c for c in keep if c in cpdag.columns]].copy()
    dag = dag[[c for c in keep if c in dag.columns]].copy()
    cpdag = cpdag[cpdag.get("status", "") == "ok"].copy()
    dag = dag[dag.get("status", "") == "ok"].copy()
    if cpdag.empty or dag.empty:
        return pd.DataFrame()
    merged = cpdag.merge(dag, on=key_cols, suffixes=("_cpdag", "_dag"), how="inner")
    if merged.empty:
        return pd.DataFrame()

    dataset_labels = {s.key: s.label for s in DATASETS}
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        dataset = str(row["dataset"])
        method = str(row["model"])
        records.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_labels.get(dataset, dataset),
                "method": method,
                "rep": int(row["rep"]),
                "seed": int(row["seed"]) if _is_num(row.get("seed")) else np.nan,
                "cpdag_shd": row.get("shd_cpdag", np.nan),
                "cpdag_f1": _f1_value(row.get("F1_cpdag", np.nan), row.get("shd_cpdag", np.nan)),
                "dag_shd": row.get("shd_dag", np.nan),
                "dag_f1": _f1_value(row.get("F1_dag", np.nan), row.get("shd_dag", np.nan)),
                "cpdag_shd_avg": row.get("shd_cpdag", np.nan),
                "cpdag_shd_best": row.get("shd_cpdag", np.nan),
                "cpdag_shd_worst": row.get("shd_cpdag", np.nan),
                "cpdag_f1_avg": _f1_value(row.get("F1_cpdag", np.nan), row.get("shd_cpdag", np.nan)),
                "cpdag_f1_best": _f1_value(row.get("F1_cpdag", np.nan), row.get("shd_cpdag", np.nan)),
                "cpdag_f1_worst": _f1_value(row.get("F1_cpdag", np.nan), row.get("shd_cpdag", np.nan)),
                "dag_shd_avg": row.get("shd_dag", np.nan),
                "dag_shd_best": row.get("shd_dag", np.nan),
                "dag_shd_worst": row.get("shd_dag", np.nan),
                "dag_f1_avg": _f1_value(row.get("F1_dag", np.nan), row.get("shd_dag", np.nan)),
                "dag_f1_best": _f1_value(row.get("F1_dag", np.nan), row.get("shd_dag", np.nan)),
                "dag_f1_worst": _f1_value(row.get("F1_dag", np.nan), row.get("shd_dag", np.nan)),
                "n_cpdags_compat": 1.0,
                "n_dags_compat": 1.0,
                "time_sec": row.get("elapsed_cpdag", row.get("elapsed_dag", np.nan)),
                "raw_graph_path": row.get("raw_graph_path_cpdag", row.get("raw_graph_path_dag", "")),
                "aspcr_constraints_path": row.get(
                    "aspcr_constraints_path_dag",
                    row.get("aspcr_constraints_path_cpdag", ""),
                ),
                "source": source_version,
            }
        )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values(["dataset", "method", "seed", "rep"]).drop_duplicates(
            ["dataset", "method", "seed"], keep="last"
        )
    return out


def _load_progress_records(results_dir: Path, versions: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for version in versions:
        progress_dir = results_dir / "progress" / version
        if not progress_dir.exists():
            continue
        for cpdag_path in sorted(progress_dir.glob("*__*_cpdag.csv")):
            if cpdag_path.name.endswith("_summary.csv"):
                continue
            dag_path = cpdag_path.with_name(cpdag_path.name.replace("_cpdag.csv", "_dag.csv"))
            frames.append(_load_progress_pair(cpdag_path, dag_path, source_version=version))
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_records(args: argparse.Namespace) -> pd.DataFrame:
    results_dir = Path(args.results_dir).expanduser().resolve()
    mcs_dir = Path(args.mcs_results_dir).expanduser().resolve()
    progress = _load_progress_records(results_dir, args.progress_version)
    mcs = _load_mcs_records(
        mcs_dir,
        include_partial=bool(args.include_partial_mcs),
        allow_injected_wrong_facts=bool(args.allow_injected_wrong_facts),
    )
    frames = [df for df in (mcs, progress) if not df.empty]
    if not frames:
        raise RuntimeError("No records found.")
    records = pd.concat(frames, ignore_index=True)
    metric_names = {
        metric
        for metric, _, _ in (
            METRICS
            + CPDAG_RANGE_METRICS
            + DAG_RANGE_METRICS
            + MCS_STRUCTURE_METRICS
            + OPT_VS_MPC_METRICS
        )
    }
    for metric in metric_names:
        if metric not in records.columns:
            records[metric] = np.nan
        records[metric] = pd.to_numeric(records[metric], errors="coerce")
    return records


def _collect_fact_records(args: argparse.Namespace) -> pd.DataFrame:
    mcs_dir = Path(args.mcs_results_dir).expanduser().resolve()
    records = _load_mcs_fact_records(
        mcs_dir,
        include_partial=bool(args.include_partial_mcs),
        allow_injected_wrong_facts=bool(args.allow_injected_wrong_facts),
    )
    for metric, _ in FACT_METRICS:
        if metric in records.columns:
            records[metric] = pd.to_numeric(records[metric], errors="coerce")
    return records


def _n_unique_runs(group: pd.DataFrame) -> int:
    if "seed" in group.columns:
        seeds = pd.to_numeric(group["seed"], errors="coerce").dropna()
        if not seeds.empty:
            return int(seeds.nunique())
    return int(group["rep"].nunique()) if "rep" in group.columns else int(len(group))


def _summarise_for_metrics(records: pd.DataFrame, metrics: tuple[tuple[str, str, bool], ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, dataset_label, method), group in records.groupby(["dataset", "dataset_label", "method"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": dataset_label,
            "method": method,
            "n_reps": _n_unique_runs(group),
        }
        for metric, _, _ in metrics:
            if metric not in group.columns:
                group_metric = pd.Series(dtype=float)
            else:
                group_metric = group[metric]
            vals = pd.to_numeric(group_metric, errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
            row[f"{metric}_n"] = int(len(vals))
        graph_count_metrics = [metric for metric in ("cpdag_shd", "cpdag_f1", "dag_shd", "dag_f1") if f"{metric}_n" in row]
        if graph_count_metrics:
            row["n_graph"] = int(min(row[f"{metric}_n"] for metric in graph_count_metrics))
        else:
            row["n_graph"] = int(min((row[f"{metric}_n"] for metric, _, _ in metrics), default=0))
        row["n"] = row["n_graph"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    dataset_order = {s.key: i for i, s in enumerate(DATASETS)}
    method_order = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary["_dataset_order"] = summary["dataset"].map(dataset_order).fillna(999)
    summary["_method_order"] = summary["method"].map(method_order).fillna(999)
    summary = summary.sort_values(["_dataset_order", "_method_order", "method"]).drop(columns=["_dataset_order", "_method_order"])
    return summary.reset_index(drop=True)


def _summarise(records: pd.DataFrame) -> pd.DataFrame:
    return _summarise_for_metrics(records, METRICS)


def _summarise_fact_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, dataset_label, method), group in records.groupby(["dataset", "dataset_label", "method"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": dataset_label,
            "method": method,
            "n_reps": _n_unique_runs(group),
        }
        for metric, _ in FACT_METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
            row[f"{metric}_n"] = int(len(vals))
        rows.append(row)
    summary = pd.DataFrame(rows)
    dataset_order = {s.key: i for i, s in enumerate(DATASETS)}
    method_order = {m: i for i, m in enumerate(FACT_METHOD_ORDER)}
    summary["_dataset_order"] = summary["dataset"].map(dataset_order).fillna(999)
    summary["_method_order"] = summary["method"].map(method_order).fillna(999)
    summary = summary.sort_values(["_dataset_order", "_method_order", "method"]).drop(columns=["_dataset_order", "_method_order"])
    return summary.reset_index(drop=True)


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    if (
        len(a) == len(b)
        and np.allclose(a, b)
        and np.allclose(np.var(a, ddof=1), 0.0)
        and np.allclose(np.var(b, ddof=1), 0.0)
    ):
        return 0.0, 1.0
    if np.allclose(np.var(a, ddof=1), 0.0) and np.allclose(np.var(b, ddof=1), 0.0):
        if np.allclose(np.mean(a), np.mean(b)):
            return 0.0, 1.0
        return math.copysign(float("inf"), float(np.mean(a) - np.mean(b))), 0.0
    if stats is not None:
        res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        t = float(res.statistic) if _is_num(res.statistic) else float("nan")
        p = float(res.pvalue) if _is_num(res.pvalue) else float("nan")
        return t, p

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se2 = var_a / len(a) + var_b / len(b)
    if se2 <= 0 or not math.isfinite(se2):
        return float("nan"), float("nan")
    t = (mean_a - mean_b) / math.sqrt(se2)
    denom = ((var_a / len(a)) ** 2 / (len(a) - 1)) + ((var_b / len(b)) ** 2 / (len(b) - 1))
    if denom <= 0 or not math.isfinite(denom):
        return float("nan"), float("nan")
    df = (se2**2) / denom
    p = _student_t_two_sided_p(t, df)
    return t, p


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: float) -> float:
    if not (_is_num(t) and _is_num(df)) or df <= 0:
        return float("nan")
    x = df / (df + float(t) ** 2)
    return max(0.0, min(1.0, _regularized_incomplete_beta(0.5 * df, 0.5, x)))


def _bh_adjust(pvals: list[float]) -> list[float]:
    qvals = [float("nan")] * len(pvals)
    indexed = [(i, float(p)) for i, p in enumerate(pvals) if _is_num(p)]
    if not indexed:
        return qvals
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    running = 1.0
    adjusted_sorted: list[tuple[int, float]] = []
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted_sorted.append((idx, min(running, 1.0)))
    for idx, q in adjusted_sorted:
        qvals[idx] = q
    return qvals


def _values(records: pd.DataFrame, dataset: str, method: str, metric: str) -> np.ndarray:
    mask = (records["dataset"] == dataset) & (records["method"] == method)
    return pd.to_numeric(records.loc[mask, metric], errors="coerce").dropna().to_numpy(dtype=float)


def _build_tests(
    records: pd.DataFrame,
    *,
    alpha: float,
    metrics: tuple[tuple[str, str, bool], ...] = METRICS,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], bool]]:
    bold: dict[tuple[str, str, str], bool] = {}
    tests: list[dict[str, Any]] = []
    label_by_dataset = {s.key: s.label for s in DATASETS}

    for spec in DATASETS:
        present = [m for m in method_order if not records[(records["dataset"] == spec.key) & (records["method"] == m)].empty]
        if not present:
            continue
        for metric, metric_label, higher_better in metrics:
            stats_by_method: dict[str, dict[str, Any]] = {}
            for method in present:
                vals = _values(records, spec.key, method, metric)
                if len(vals):
                    stats_by_method[method] = {
                        "values": vals,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    }
            if len(stats_by_method) < 2:
                for method in stats_by_method:
                    bold[(spec.key, method, metric)] = True
                continue

            best_method = sorted(
                stats_by_method,
                key=lambda m: (
                    -stats_by_method[m]["mean"] if higher_better else stats_by_method[m]["mean"],
                    method_order.index(m) if m in method_order else 999,
                ),
            )[0]
            best = stats_by_method[best_method]
            bold[(spec.key, best_method, metric)] = True

            family_rows: list[dict[str, Any]] = []
            for method, data in stats_by_method.items():
                if method == best_method:
                    continue
                t_stat, p_value = _welch(best["values"], data["values"])
                family_rows.append(
                    {
                        "dataset": spec.key,
                        "dataset_label": label_by_dataset.get(spec.key, spec.label),
                        "metric": metric,
                        "metric_label": metric_label,
                        "best_method": best_method,
                        "comparator": method,
                        "best_mean": best["mean"],
                        "best_std": best["std"],
                        "best_n": best["n"],
                        "comparator_mean": data["mean"],
                        "comparator_std": data["std"],
                        "comparator_n": data["n"],
                        "t": t_stat,
                        "p": p_value,
                        "higher_better": higher_better,
                    }
                )
            qvals = _bh_adjust([r["p"] for r in family_rows])
            for row, q in zip(family_rows, qvals, strict=True):
                row["p_bh"] = q
                row["sig_bh"] = _sig_symbol(q)
                same_mean = np.isclose(float(row["best_mean"]), float(row["comparator_mean"]), rtol=1e-9, atol=1e-12)
                is_tied = bool(same_mean or (_is_num(q) and q >= alpha))
                row["tied_with_best"] = is_tied
                if is_tied:
                    bold[(spec.key, row["comparator"], metric)] = True
                tests.append(row)

    return pd.DataFrame.from_records(tests), bold


def _build_indicator_tests(
    records: pd.DataFrame,
    bold: dict[tuple[str, str, str], bool],
    *,
    metrics: tuple[tuple[str, str, bool], ...] = METRICS,
    method_order: tuple[str, ...] = METHOD_ORDER,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], str]]:
    tests: list[dict[str, Any]] = []
    markers: dict[tuple[str, str, str], str] = {}
    label_by_dataset = {s.key: s.label for s in DATASETS}

    for spec in DATASETS:
        present = [m for m in method_order if not records[(records["dataset"] == spec.key) & (records["method"] == m)].empty]
        if not present:
            continue
        for metric, metric_label, higher_better in metrics:
            stats_by_method: dict[str, dict[str, Any]] = {}
            for method in present:
                vals = _values(records, spec.key, method, metric)
                if len(vals):
                    stats_by_method[method] = {
                        "values": vals,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    }
            if len(stats_by_method) < 2:
                continue

            ranked = sorted(
                stats_by_method,
                key=lambda m: (
                    -stats_by_method[m]["mean"] if higher_better else stats_by_method[m]["mean"],
                    method_order.index(m) if m in method_order else 999,
                ),
            )
            top_methods: list[str] = []
            next_method: str | None = None
            for method in ranked:
                if bold.get((spec.key, method, metric), False) and next_method is None:
                    top_methods.append(method)
                else:
                    next_method = method
                    break
            if not top_methods or next_method is None:
                continue

            family_rows: list[dict[str, Any]] = []
            comparator = stats_by_method[next_method]
            for method in top_methods:
                subject = stats_by_method[method]
                t_stat, p_value = _welch(subject["values"], comparator["values"])
                family_rows.append(
                    {
                        "dataset": spec.key,
                        "dataset_label": label_by_dataset.get(spec.key, spec.label),
                        "metric": metric,
                        "metric_label": metric_label,
                        "best_method": method,
                        "comparator": next_method,
                        "best_mean": subject["mean"],
                        "best_std": subject["std"],
                        "best_n": subject["n"],
                        "comparator_mean": comparator["mean"],
                        "comparator_std": comparator["std"],
                        "comparator_n": comparator["n"],
                        "t": t_stat,
                        "p": p_value,
                        "higher_better": higher_better,
                        "tied_with_best": False,
                        "test_role": "marker_top_vs_next_nonbold",
                        "marker_printed_on": method,
                    }
                )
            qvals = _bh_adjust([r["p"] for r in family_rows])
            for row, q in zip(family_rows, qvals, strict=True):
                row["p_bh"] = q
                row["sig_bh"] = _sig_symbol(q)
                markers[(spec.key, row["best_method"], metric)] = row["sig_bh"]
                tests.append(row)

    return pd.DataFrame.from_records(tests), markers


def _render_main_table(
    summary: pd.DataFrame,
    bold: dict[tuple[str, str, str], bool],
    markers: dict[tuple[str, str, str], str],
    *,
    out_rel_note: str,
    label: str,
    caption_note: str = "",
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    caption = (
        r"Matched baseline comparison for the current experimental runs. Values are mean $\pm$ standard deviation "
        r"over successful matched repetitions; $n_g$ reports the minimum number of graph-metric repetitions "
        r"available across the CPDAG and DAG columns. Runtime is averaged over all available runtime records. "
        r"Lower SHD/runtime and higher F1 are better. Bold entries are the best mean and methods not significantly "
        r"different from it under Welch tests with Benjamini--Hochberg correction at $\alpha=0.05$; superscripts "
        r"on bold top entries show the corrected top-vs-next-nonbold indicator test."
    )
    if caption_note:
        caption = caption + " " + str(caption_note).strip()
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrccccc}")
    lines.append(r"\toprule")
    header = ["Dataset", "Method", "$n_g$"] + [label for _, label, _ in METRICS]
    lines.append(" & ".join(header) + r"\\")
    lines.append(r"\midrule")
    first_dataset = True
    for spec in DATASETS:
        sub = summary[summary["dataset"] == spec.key]
        if sub.empty:
            continue
        if not first_dataset:
            lines.append(r"\midrule")
        first_dataset = False
        for _, row in sub.iterrows():
            cells = [_latex_escape(str(row["dataset_label"])), _latex_escape(str(row["method"])), str(int(row["n"]))]
            for metric, _, _ in METRICS:
                cell_bold = bool(bold.get((str(row["dataset"]), str(row["method"]), metric), False))
                marker = markers.get((str(row["dataset"]), str(row["method"]), metric))
                cells.append(_fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"], bold=cell_bold, marker=marker))
            lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append("")
    lines.append(r"% Generated by " + _latex_escape(out_rel_note) + ".")
    return "\n".join(lines) + "\n"


def _render_metric_table(
    summary: pd.DataFrame,
    bold: dict[tuple[str, str, str], bool],
    markers: dict[tuple[str, str, str], str],
    *,
    metrics: tuple[tuple[str, str, bool], ...],
    label: str,
    caption: str,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    align = "llr" + "c" * len(metrics)
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    header = ["Dataset", "Method", "$n$"] + [metric_label for _, metric_label, _ in metrics]
    lines.append(" & ".join(header) + r"\\")
    lines.append(r"\midrule")
    if summary.empty:
        lines.append(rf"\multicolumn{{{3 + len(metrics)}}}{{c}}{{No records available.}}\\")
    else:
        first_dataset = True
        for spec in DATASETS:
            sub = summary[summary["dataset"] == spec.key]
            if sub.empty:
                continue
            if not first_dataset:
                lines.append(r"\midrule")
            first_dataset = False
            for _, row in sub.iterrows():
                n_vals = [int(row.get(f"{metric}_n", 0) or 0) for metric, _, _ in metrics]
                n_display = min([n for n in n_vals if n > 0], default=0)
                cells = [_latex_escape(str(row["dataset_label"])), _latex_escape(str(row["method"])), str(n_display)]
                for metric, _, _ in metrics:
                    cell_bold = bool(bold.get((str(row["dataset"]), str(row["method"]), metric), False))
                    marker = markers.get((str(row["dataset"]), str(row["method"]), metric))
                    cells.append(_fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"], bold=cell_bold, marker=marker))
                lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _build_opt_vs_mpc_deltas(
    records: pd.DataFrame,
    *,
    subject: str = "OptABA-PC",
    comparator: str = "MPC",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        sub = records[records["dataset"] == spec.key].copy()
        if sub.empty:
            continue
        for metric, metric_label, higher_better in OPT_VS_MPC_METRICS:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            if subject not in pivot.columns or comparator not in pivot.columns:
                continue
            paired = pivot[[subject, comparator]].apply(pd.to_numeric, errors="coerce").dropna()
            if paired.empty:
                continue
            subj_vals = paired[subject].to_numpy(dtype=float)
            comp_vals = paired[comparator].to_numpy(dtype=float)
            delta = subj_vals - comp_vals
            better = delta > 0 if higher_better else delta < 0
            t_stat, p_value = _welch(subj_vals, comp_vals)
            rows.append(
                {
                    "dataset": spec.key,
                    "dataset_label": spec.label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "higher_better": higher_better,
                    "subject": subject,
                    "comparator": comparator,
                    "n": int(len(delta)),
                    "subject_mean": float(np.mean(subj_vals)),
                    "comparator_mean": float(np.mean(comp_vals)),
                    "delta_mean": float(np.mean(delta)),
                    "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                    "subject_better_n": int(np.sum(better)),
                    "t": t_stat,
                    "p": p_value,
                }
            )
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        out["p_bh"] = _bh_adjust(out["p"].tolist())
        out["sig_bh"] = [_sig_symbol(p) for p in out["p_bh"]]
        out["subject_significantly_better"] = [
            bool(
                _is_num(row["p_bh"])
                and float(row["p_bh"]) < 0.05
                and ((float(row["delta_mean"]) > 0) if bool(row["higher_better"]) else (float(row["delta_mean"]) < 0))
            )
            for _, row in out.iterrows()
        ]
    return out


def _fmt_delta_cell(row: pd.Series | None) -> str:
    if row is None or not _is_num(row.get("delta_mean")):
        return "--"
    delta = float(row["delta_mean"])
    body = _fmt_num(delta)
    marker = _latex_sig_marker(str(row.get("sig_bh")) if row.get("subject_significantly_better") else None)
    if bool(row.get("subject_significantly_better")):
        body = rf"\mathbf{{{body}}}"
        if marker:
            body = rf"{body}^{{{marker}}}"
    n = int(row.get("n", 0) or 0)
    better = int(row.get("subject_better_n", 0) or 0)
    return rf"${body}$ ({better}/{n})"


def _render_opt_vs_mpc_delta_table(deltas: pd.DataFrame, *, label: str, caption_note: str = "") -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    caption = (
        r"Matched OptABA-PC minus MPC differences. Negative SHD deltas and positive F1 deltas favour OptABA-PC; "
        r"parentheses report the number of matched seeds where OptABA-PC is better. Bold entries are favourable "
        r"OptABA-PC differences under Welch tests with Benjamini--Hochberg correction at $\alpha=0.05$."
    )
    if caption_note:
        caption = caption + " " + str(caption_note).strip()
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(OPT_VS_MPC_METRICS) + r"}")
    lines.append(r"\toprule")
    lines.append("Dataset & " + " & ".join(label for _, label, _ in OPT_VS_MPC_METRICS) + r"\\")
    lines.append(r"\midrule")
    if deltas.empty:
        lines.append(rf"\multicolumn{{{1 + len(OPT_VS_MPC_METRICS)}}}{{c}}{{No matched OptABA-PC/MPC records available.}}\\")
    else:
        for spec in DATASETS:
            sub = deltas[deltas["dataset"] == spec.key]
            if sub.empty:
                continue
            cells = [_latex_escape(spec.label)]
            for metric, _, _ in OPT_VS_MPC_METRICS:
                match = sub[sub["metric"] == metric]
                cells.append(_fmt_delta_cell(match.iloc[0] if not match.empty else None))
            lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _test_means_cell(row: pd.Series) -> str:
    left = _fmt_mean_std(float(row["best_mean"]), float(row["best_std"]))
    right = _fmt_mean_std(float(row["comparator_mean"]), float(row["comparator_std"]))
    return rf"{left} vs {right}"


def _render_tests_table(rows: pd.DataFrame, *, label: str, caption: str, max_rows: int = 36) -> str:
    if rows.empty:
        chunks = [rows]
    else:
        chunks = [rows.iloc[i : i + max_rows].copy() for i in range(0, len(rows), max_rows)]

    rendered: list[str] = []
    for chunk_idx, chunk in enumerate(chunks):
        rendered.append(_render_tests_table_chunk(chunk, label=label, caption=caption, chunk_idx=chunk_idx, n_chunks=len(chunks)))
    return "\n".join(rendered) + "\n"


def _render_tests_table_chunk(
    rows: pd.DataFrame,
    *,
    label: str,
    caption: str,
    chunk_idx: int,
    n_chunks: int,
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    if n_chunks == 1:
        lines.append(r"\caption{" + caption + r"}")
    else:
        suffix = f" Part {chunk_idx + 1} of {n_chunks}." if chunk_idx == 0 else f" Continued, part {chunk_idx + 1} of {n_chunks}."
        lines.append(r"\caption{" + caption + _latex_escape(suffix) + r"}")
    if chunk_idx == 0:
        lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllrcrcc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Metric & Methods & $n$ & Means $\pm$ Std & $t$ & $p$ & $p_{\mathrm{BH}}$\\"
    )
    lines.append(r"\midrule")
    if rows.empty:
        lines.append(r"\multicolumn{8}{c}{No applicable tests.}\\")
    else:
        for _, row in rows.iterrows():
            methods = _latex_escape(f"{row['best_method']} vs {row['comparator']}")
            ns = f"{int(row['best_n'])}/{int(row['comparator_n'])}"
            cells = [
                _latex_escape(str(row["dataset_label"])),
                str(row["metric_label"]),
                methods,
                ns,
                _test_means_cell(row),
                _fmt_t(float(row["t"])),
                rf"{_fmt_p(float(row['p']))}\,{_sig_symbol(float(row['p']))}",
                rf"{_fmt_p(float(row['p_bh']))}\,{_sig_symbol(float(row['p_bh']))}",
            ]
            lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _render_fact_table(summary: pd.DataFrame, *, label: str, caption_note: str = "") -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    caption = (
        r"Fact-level diagnostics for the ABA-PC inputs and accepted fact sets. Raw PC facts are the complete "
        r"CI fact set produced by the same PC/MPC configuration before ABA-PC removal. Values are mean $\pm$ "
        r"standard deviation over unique seeds. Higher precision, recall, and F1 are better; lower wrong and "
        r"kept-wrong counts are better."
    )
    if caption_note:
        caption = caption + " " + str(caption_note).strip()
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrcccccccc}")
    lines.append(r"\toprule")
    header = ["Dataset", "Method", "$n$"] + [label for _, label in FACT_METRICS]
    lines.append(" & ".join(header) + r"\\")
    lines.append(r"\midrule")
    if summary.empty:
        lines.append(r"\multicolumn{11}{c}{No fact diagnostics available.}\\")
    else:
        first_dataset = True
        for spec in DATASETS:
            sub = summary[summary["dataset"] == spec.key]
            if sub.empty:
                continue
            if not first_dataset:
                lines.append(r"\midrule")
            first_dataset = False
            for _, row in sub.iterrows():
                cells = [_latex_escape(str(row["dataset_label"])), _latex_escape(str(row["method"])), str(int(row["n_reps"]))]
                for metric, _ in FACT_METRICS:
                    cells.append(_fmt_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"]))
                lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _write_outputs(
    args: argparse.Namespace,
    records: pd.DataFrame,
    summary: pd.DataFrame,
    fact_records: pd.DataFrame,
    fact_summary: pd.DataFrame,
    tests: pd.DataFrame,
    indicator: pd.DataFrame,
    bold: dict[tuple[str, str, str], bool],
    markers: dict[tuple[str, str, str], str],
) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records.to_csv(out_dir / "matched_baseline_records.csv", index=False)
    summary.to_csv(out_dir / "matched_baseline_summary.csv", index=False)
    fact_records.to_csv(out_dir / "matched_baseline_fact_records.csv", index=False)
    fact_summary.to_csv(out_dir / "matched_baseline_fact_summary.csv", index=False)
    tests.to_csv(out_dir / "matched_baseline_welch_tests.csv", index=False)

    rel_note = str(Path(__file__).resolve())
    (out_dir / "table_matched_baseline_comparison.tex").write_text(
        _render_main_table(
            summary,
            bold,
            markers,
            out_rel_note=rel_note,
            label=str(args.table_label),
            caption_note=str(args.caption_note or ""),
        )
    )
    (out_dir / "table_matched_baseline_fact_diagnostics.tex").write_text(
        _render_fact_table(
            fact_summary,
            label=str(args.fact_table_label),
            caption_note=str(args.caption_note or ""),
        )
    )

    def _derived_label(suffix: str) -> str:
        table_label = str(args.table_label)
        if "comparison" in table_label:
            return table_label.replace("comparison", suffix)
        return f"{table_label}-{suffix}"

    comparison_methods = ("ABA-PC", "OptABA-PC", "MPC")
    comparison_records = records[records["method"].isin(comparison_methods)].copy()
    mcs_records = records[records["method"].isin(("ABA-PC", "OptABA-PC"))].copy()

    cpdag_range_summary = _summarise_for_metrics(comparison_records, CPDAG_RANGE_METRICS)
    cpdag_range_tests, cpdag_range_bold = _build_tests(
        comparison_records,
        alpha=float(args.alpha),
        metrics=CPDAG_RANGE_METRICS,
        method_order=comparison_methods,
    )
    cpdag_range_indicator, cpdag_range_markers = _build_indicator_tests(
        comparison_records,
        cpdag_range_bold,
        metrics=CPDAG_RANGE_METRICS,
        method_order=comparison_methods,
    )
    cpdag_range_summary.to_csv(out_dir / "matched_baseline_cpdag_range_summary.csv", index=False)
    cpdag_range_tests.to_csv(out_dir / "matched_baseline_cpdag_range_welch_tests.csv", index=False)
    cpdag_range_indicator.to_csv(out_dir / "matched_baseline_cpdag_range_indicator_tests.csv", index=False)
    (out_dir / "table_matched_baseline_cpdag_range.tex").write_text(
        _render_metric_table(
            cpdag_range_summary,
            cpdag_range_bold,
            cpdag_range_markers,
            metrics=CPDAG_RANGE_METRICS,
            label=_derived_label("cpdag-range"),
            caption=(
                r"CPDAG compatible-set view for ABA-PC and OptABA-PC against MPC. For MPC, average, best, "
                r"and worst coincide because the method returns one graph. For ABA-PC and OptABA-PC, average, "
                r"best, and worst are computed over the compatible graph set returned by graph evaluation. "
                r"Bold entries are best/tied under Welch tests with Benjamini--Hochberg correction."
            ),
        )
    )

    dag_range_summary = _summarise_for_metrics(comparison_records, DAG_RANGE_METRICS)
    dag_range_tests, dag_range_bold = _build_tests(
        comparison_records,
        alpha=float(args.alpha),
        metrics=DAG_RANGE_METRICS,
        method_order=comparison_methods,
    )
    dag_range_indicator, dag_range_markers = _build_indicator_tests(
        comparison_records,
        dag_range_bold,
        metrics=DAG_RANGE_METRICS,
        method_order=comparison_methods,
    )
    dag_range_summary.to_csv(out_dir / "matched_baseline_dag_range_summary.csv", index=False)
    dag_range_tests.to_csv(out_dir / "matched_baseline_dag_range_welch_tests.csv", index=False)
    dag_range_indicator.to_csv(out_dir / "matched_baseline_dag_range_indicator_tests.csv", index=False)
    (out_dir / "table_matched_baseline_dag_range.tex").write_text(
        _render_metric_table(
            dag_range_summary,
            dag_range_bold,
            dag_range_markers,
            metrics=DAG_RANGE_METRICS,
            label=_derived_label("dag-range"),
            caption=(
                r"DAG compatible-set view for ABA-PC and OptABA-PC against MPC. For MPC, average, best, "
                r"and worst coincide because the method returns one graph. For ABA-PC and OptABA-PC, average, "
                r"best, and worst are computed over the compatible graph set returned by graph evaluation. "
                r"Bold entries are best/tied under Welch tests with Benjamini--Hochberg correction."
            ),
        )
    )

    mcs_structure_summary = _summarise_for_metrics(mcs_records, MCS_STRUCTURE_METRICS)
    mcs_structure_tests, mcs_structure_bold = _build_tests(
        mcs_records,
        alpha=float(args.alpha),
        metrics=MCS_STRUCTURE_METRICS,
        method_order=("ABA-PC", "OptABA-PC"),
    )
    mcs_structure_indicator, mcs_structure_markers = _build_indicator_tests(
        mcs_records,
        mcs_structure_bold,
        metrics=MCS_STRUCTURE_METRICS,
        method_order=("ABA-PC", "OptABA-PC"),
    )
    mcs_structure_summary.to_csv(out_dir / "matched_baseline_mcs_structure_summary.csv", index=False)
    mcs_structure_tests.to_csv(out_dir / "matched_baseline_mcs_structure_welch_tests.csv", index=False)
    mcs_structure_indicator.to_csv(out_dir / "matched_baseline_mcs_structure_indicator_tests.csv", index=False)
    (out_dir / "table_matched_baseline_mcs_structure.tex").write_text(
        _render_metric_table(
            mcs_structure_summary,
            mcs_structure_bold,
            mcs_structure_markers,
            metrics=MCS_STRUCTURE_METRICS,
            label=_derived_label("mcs-structure"),
            caption=(
                r"ABA-PC and OptABA-PC compatible-set diagnostics. Skeleton F1 evaluates adjacency recovery, "
                r"arrow F1 evaluates directed arrow recovery, and true-in-set columns report the fraction of "
                r"runs where the ground-truth graph/equivalence class is present among compatible outputs. "
                r"Bold entries are best/tied under Welch tests with Benjamini--Hochberg correction."
            ),
        )
    )

    opt_vs_mpc = _build_opt_vs_mpc_deltas(records)
    opt_vs_mpc.to_csv(out_dir / "matched_baseline_optabapc_vs_mpc_deltas.csv", index=False)
    (out_dir / "table_matched_baseline_optabapc_vs_mpc_deltas.tex").write_text(
        _render_opt_vs_mpc_delta_table(
            opt_vs_mpc,
            label=_derived_label("optabapc-vs-mpc-deltas"),
            caption_note=str(args.caption_note or ""),
        )
    )

    tied = tests[tests["tied_with_best"] == True].copy() if not tests.empty else pd.DataFrame()
    tied.to_csv(out_dir / "matched_baseline_tied_bolding_tests.csv", index=False)
    indicator.to_csv(out_dir / "matched_baseline_indicator_tests.csv", index=False)

    (out_dir / "table_matched_baseline_tied_bolding_tests.tex").write_text(
        _render_tests_table(
            tied,
            label=str(args.tied_tests_label),
            caption=(
                r"Non-significant best-vs-comparator Welch tests used to extend bolding in "
                rf"Table~\ref{{{args.table_label}}}. Significance symbols are based on "
                r"Benjamini--Hochberg corrected p-values within each dataset--metric family "
                r"($^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, $^{.}p<0.1$)."
            ),
        )
    )
    (out_dir / "table_matched_baseline_indicator_tests.tex").write_text(
        _render_tests_table(
            indicator,
            label=str(args.indicator_tests_label),
            caption=(
                rf"Welch tests for Table~\ref{{{args.table_label}}}. Each row compares a bolded "
                r"top entry with the next non-bold method in that dataset--metric ranking. Significance "
                r"symbols are based on Benjamini--Hochberg corrected p-values within each dataset--metric "
                r"family and are printed as superscripts on the corresponding bold table entries."
            ),
        )
    )
    print(f"Wrote records: {out_dir / 'matched_baseline_records.csv'}")
    print(f"Wrote summary: {out_dir / 'matched_baseline_summary.csv'}")
    print(f"Wrote fact diagnostics: {out_dir / 'matched_baseline_fact_summary.csv'}")
    print(f"Wrote tests: {out_dir / 'matched_baseline_welch_tests.csv'}")
    print(f"Wrote LaTeX tables under: {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--mcs-results-dir", default=str(REPO_ROOT / "results" / "final_mcs_experiments"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "tables" / "paper_matched_baselines"))
    parser.add_argument("--table-label", default="tab:matched-baseline-comparison")
    parser.add_argument("--fact-table-label", default=None)
    parser.add_argument("--tied-tests-label", default="tab:matched-baseline-tied-bolding-tests")
    parser.add_argument("--indicator-tests-label", default="tab:matched-baseline-indicator-tests")
    parser.add_argument("--caption-note", default="")
    parser.add_argument(
        "--progress-version",
        action="append",
        default=None,
        help="Matched baseline progress version to include. Can be repeated.",
    )
    parser.add_argument(
        "--include-partial-mcs",
        action="store_true",
        help="Also read summary.partial.json files for progress previews. Final paper tables should omit this.",
    )
    parser.add_argument(
        "--allow-injected-wrong-facts",
        action="store_true",
        help=(
            "Include diagnostic MCS runs with pct_wrong_facts > 0. Disabled by default because "
            "their ABA-PC/OptABA-PC inputs are not matched by the MPC baseline."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.progress_version is None:
        args.progress_version = ["paper_mpc_fgs_10rep", "paper_aspcr_10rep"]
    if args.fact_table_label is None:
        table_label = str(args.table_label)
        if "comparison" in table_label:
            args.fact_table_label = table_label.replace("comparison", "fact-diagnostics")
        else:
            args.fact_table_label = f"{table_label}-fact-diagnostics"
    return args


def main() -> int:
    args = _parse_args()
    records = _collect_records(args)
    fact_records = _collect_fact_records(args)
    summary = _summarise(records)
    fact_summary = _summarise_fact_records(fact_records)
    tests, bold = _build_tests(records, alpha=float(args.alpha))
    indicator, markers = _build_indicator_tests(records, bold)
    _write_outputs(args, records, summary, fact_records, fact_summary, tests, indicator, bold, markers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
