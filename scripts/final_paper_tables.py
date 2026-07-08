"""Write the compact, numbered UAI paper table set.

This is a presentation layer over the generated artifacts in ``results/tables``.
It does not run experiments or recompute graph metrics from raw predictions.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import paper_tables as pt  # noqa: E402


MetricList = list[pt.MetricSpec]
GroupSpec = tuple[str, Callable[[pd.DataFrame], pd.Series]]

RAW_SHD = pt.MetricSpec("SHD", "shd_mean", "shd_std", False)
RAW_SID = pt.MetricSpec("SID", "SID_mean", "SID_std", False)

CPDAG_MAIN_METRICS = [pt.STRUCTURAL_METRICS["NSHD"], pt.STRUCTURAL_METRICS["F1"]]
CPDAG_EXTRA_METRICS = [
    pt.STRUCTURAL_METRICS["NSID-low"],
    pt.STRUCTURAL_METRICS["NSID-high"],
    pt.STRUCTURAL_METRICS["Precision"],
    pt.STRUCTURAL_METRICS["Recall"],
]
DAG_MAIN_METRICS = [RAW_SHD, pt.STRUCTURAL_METRICS["F1"]]
DAG_RAW_METRICS = [
    RAW_SHD,
    pt.STRUCTURAL_METRICS["Precision"],
    pt.STRUCTURAL_METRICS["Recall"],
    pt.STRUCTURAL_METRICS["F1"],
]
DAG_ALL_METRICS = [
    RAW_SHD,
    RAW_SID,
    pt.STRUCTURAL_METRICS["Precision"],
    pt.STRUCTURAL_METRICS["Recall"],
    pt.STRUCTURAL_METRICS["F1"],
]
DAG_GPT_METRICS = [
    RAW_SHD,
    pt.STRUCTURAL_METRICS["F1"],
]

DISPLAY_ONLY_METHODS = {"LLM-BFS"}
CAUSENET_DAG_METHODS = [
    "Random",
    "FGS",
    "NOTEARS-MLP",
    "GRaSP",
    "BOSS",
    "MPC",
    "MPC-LLM",
    "ABAPC",
    "ABAPC-LLM",
    "LLM-BFS",
]
BNLEARN_CPDAG_METHODS = ["GRaSP", "BOSS", "MPC", "MPC-LLM", "ABAPC", "ABAPC-LLM"]
BNLEARN_DAG_METHODS = CAUSENET_DAG_METHODS
GPT_ABLATION_METHODS = [
    "ABAPC-LLM",
    "ABAPC-LLM-gpt5mini",
    "ABAPC",
    "MPC-LLM",
    "MPC-LLM-gpt5mini",
    "MPC",
]
MAIN_WITH_GPT_METHODS = [
    "Random",
    "FGS",
    "NOTEARS-MLP",
    "GRaSP",
    "BOSS",
    "MPC",
    "MPC-LLM",
    "MPC-LLM-gpt5mini",
    "ABAPC",
    "ABAPC-LLM",
    "ABAPC-LLM-gpt5mini",
    "LLM-BFS",
]
LLM_SOURCE_LABELS = {
    "ABAPC-LLM": "ABAPC-LLM (Gemini)",
    "ABAPC-LLM-gpt5mini": "ABAPC-LLM (GPT)",
    "MPC-LLM": "MPC-LLM (Gemini)",
    "MPC-LLM-gpt5mini": "MPC-LLM (GPT)",
    "LLM-BFS": "LLM-BFS (Gemini)",
}
RUNS_PER_DATASET = 50
BOLD_ALPHA = 0.05
RESULT_SIG_NOTE = (
    r"Among repeated-run methods, bold marks the best mean and statistically tied methods under BH-corrected Welch tests within each "
    r"metric/column. Significance levels for the $p$-values against the next non-bold method in the ranking are: "
    r"\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, \textnormal{*} $p<0.05$, "
    r"\textnormal{.} $p<0.1$."
)
TEST_SIG_NOTE = (
    r" Significance levels for both reported $p$-values are: "
    r"\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, \textnormal{*} $p<0.05$, "
    r"\textnormal{.} $p<0.1$."
)
LLM_BFS_NOTE = r" LLM-BFS is shown from one saved run and is excluded from Welch tests and bolding."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final numbered markdown tables for the UAI paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", default="g2a01")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--synthetic-dir", default="synthetic")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def archive_current_final_files(final_dir: Path) -> tuple[Path | None, Path | None]:
    files = [path for path in final_dir.iterdir() if path.is_file()]
    if not files:
        return None, None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = final_dir / "archive" / f"tables_{timestamp}"
    legacy_check_dir = final_dir / "check" / f"legacy_{timestamp}"
    archived_any = False
    checked_any = False
    for path in files:
        if path.name.startswith("check_"):
            legacy_check_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), legacy_check_dir / path.name)
            checked_any = True
        else:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), archive_dir / path.name)
            archived_any = True
    return archive_dir if archived_any else None, legacy_check_dir if checked_any else None


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [max([len(columns[i])] + [len(row[i]) for row in rows]) for i in range(len(columns))]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [fmt_row(columns), fmt_row(["-" * width for width in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def write_table(path: Path, title: str, note: str, table: pd.DataFrame) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                note,
                "",
                markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )


def metric_label(spec: pt.MetricSpec) -> str:
    direction = "higher" if spec.higher_is_better else "lower"
    return f"{spec.label} ({direction})"


def numeric(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def group_metric_stats(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    methods: list[str],
) -> dict[str, tuple[float, float]]:
    group = summary[group_mask]
    stats = {}
    for method in methods:
        sub = group[group["model"].eq(method)]
        if sub.empty or spec.mean_col not in sub or spec.std_col not in sub:
            stats[method] = (np.nan, np.nan)
            continue
        stats[method] = (
            float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean()),
            float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean()),
        )
    return stats


def best_method(stats: dict[str, tuple[float, float]], spec: pt.MetricSpec) -> str | None:
    ranked = ranked_methods(stats, spec)
    return ranked[0] if ranked else None


def ranked_methods(stats: dict[str, tuple[float, float]], spec: pt.MetricSpec) -> list[str]:
    finite = [(method, values[0]) for method, values in stats.items() if np.isfinite(values[0])]
    if not finite:
        return []
    order = method_order()
    return [
        method
        for method, _ in sorted(
            finite,
            key=lambda item: ((-item[1]) if spec.higher_is_better else item[1], order.get(item[0], 999)),
        )
    ]


def runner_up_method(stats: dict[str, tuple[float, float]], spec: pt.MetricSpec) -> str | None:
    ranked = ranked_methods(stats, spec)
    if len(ranked) < 2:
        return None
    return ranked[1]


def method_order() -> dict[str, int]:
    ordered = []
    for method in [
        *CAUSENET_DAG_METHODS,
        *pt.SYNTH_METHOD_ORDER,
        *MAIN_WITH_GPT_METHODS,
        *GPT_ABLATION_METHODS,
        *BNLEARN_CPDAG_METHODS,
    ]:
        if method not in ordered:
            ordered.append(method)
    return {method: index for index, method in enumerate(ordered)}


def method_display(method: str, method_labels: dict[str, str] | None = None) -> str:
    if method_labels and method in method_labels:
        return method_labels[method]
    return method.replace("gpt5mini", "GPT")


def testable_methods(methods: list[str]) -> list[str]:
    return [method for method in methods if method not in DISPLAY_ONLY_METHODS]


def present_methods(summary: pd.DataFrame, methods: list[str], mask: pd.Series | None = None) -> list[str]:
    if mask is None:
        mask = pd.Series(True, index=summary.index)
    return [method for method in methods if (mask & summary["model"].eq(method)).any()]


def carry_display_only_rows(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Carry single-run display baselines into comparison views without changing tests."""
    rows = []
    for method in DISPLAY_ONLY_METHODS:
        if source["model"].eq(method).any() and not target["model"].eq(method).any():
            rows.append(source[source["model"].eq(method)])
    if not rows:
        return target
    return pd.concat([target, *rows], ignore_index=True, sort=False)


def method_test_stats(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    method: str,
) -> tuple[float, float, int]:
    sub = summary[group_mask & summary["model"].eq(method)]
    if sub.empty or spec.mean_col not in sub or spec.std_col not in sub:
        return np.nan, np.nan, 0
    means = pd.to_numeric(sub[spec.mean_col], errors="coerce").dropna()
    if means.empty:
        return np.nan, np.nan, 0
    mean = float(means.mean())
    n_datasets = int(sub.loc[means.index, "dataset"].nunique()) if "dataset" in sub else len(means)
    nobs = max(1, n_datasets * RUNS_PER_DATASET)
    if len(means) > 1:
        std = float(means.std(ddof=1))
    else:
        std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
    if not np.isfinite(std):
        std = 0.0
    return mean, std, nobs


def welch_p_from_stats(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    if not all(np.isfinite(v) for v in [mean_a, std_a, mean_b, std_b]) or min(n_a, n_b) < 1:
        return np.nan
    if std_a == 0 and std_b == 0:
        return 1.0 if np.isclose(mean_a, mean_b) else 0.0
    _, p_value = pt.welch_from_stats(mean_a, std_a, n_a, mean_b, std_b, n_b)
    return float(p_value)


def statistically_tied_methods(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    methods: list[str],
    *,
    alpha: float = BOLD_ALPHA,
) -> set[str]:
    tests = best_vs_comparator_tests(summary, group_mask, spec, methods)
    stats = group_metric_stats(summary, group_mask, spec, methods)
    best = best_method(stats, spec)
    if best is None:
        return set()
    bold = {best}
    if not tests.empty:
        for row in tests.itertuples(index=False):
            if np.isfinite(row.p_bh) and row.p_bh >= alpha:
                bold.add(str(row.comparator))
    return bold


def best_vs_comparator_tests(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    methods: list[str],
    *,
    table_name: str = "",
    group_label: str = "",
) -> pd.DataFrame:
    stats = group_metric_stats(summary, group_mask, spec, methods)
    best = best_method(stats, spec)
    runner = runner_up_method(stats, spec)
    if best is None:
        return pd.DataFrame()
    best_display_mean, best_display_std = stats[best]
    best_test_mean, best_test_std, best_nobs = method_test_stats(summary, group_mask, spec, best)
    records = []
    for method in methods:
        if method == best:
            continue
        display_mean, display_std = stats.get(method, (np.nan, np.nan))
        mean, std, nobs = method_test_stats(summary, group_mask, spec, method)
        if not np.isfinite(mean):
            continue
        if not all(np.isfinite(v) for v in [best_test_mean, best_test_std, mean, std]):
            continue
        if best_test_std == 0 and std == 0:
            t_stat = 0.0 if np.isclose(best_test_mean, mean) else np.inf
            p_value = 1.0 if np.isclose(best_test_mean, mean) else 0.0
        else:
            t_stat, p_value = pt.welch_from_stats(best_test_mean, best_test_std, best_nobs, mean, std, nobs)
        records.append(
            {
                "table": table_name,
                "group": group_label,
                "metric": spec.label,
                "best": best,
                "comparator": method,
                "best_display_mean": best_display_mean,
                "best_display_std": best_display_std,
                "comparator_display_mean": display_mean,
                "comparator_display_std": display_std,
                "best_test_mean": best_test_mean,
                "best_test_std": best_test_std,
                "best_nobs": best_nobs,
                "comparator_test_mean": mean,
                "comparator_test_std": std,
                "comparator_nobs": nobs,
                "t": float(t_stat),
                "p_value": float(p_value),
                "is_runner_up": method == runner,
                "test_role": "bolding_best_vs_comparator",
                "marker_printed_on": "",
            }
        )
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests["p_bh"] = np.nan
    finite_idx = tests[np.isfinite(pd.to_numeric(tests["p_value"], errors="coerce"))].index
    if len(finite_idx):
        _, p_bh = pt.bh_adjust(tests.loc[finite_idx, "p_value"], alpha=BOLD_ALPHA)
        tests.loc[finite_idx, "p_bh"] = p_bh
    tests["marker_bh"] = tests["p_bh"].map(pt.sig_marker)
    tests["comparator_tied"] = tests["p_bh"].ge(BOLD_ALPHA)
    return tests


def method_pair_test_record(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    subject: str,
    comparator: str,
    *,
    table_name: str = "",
    group_label: str = "",
    test_role: str = "",
    marker_printed_on: str = "",
) -> dict[str, object] | None:
    stats = group_metric_stats(summary, group_mask, spec, [subject, comparator])
    subject_display_mean, subject_display_std = stats.get(subject, (np.nan, np.nan))
    comparator_display_mean, comparator_display_std = stats.get(comparator, (np.nan, np.nan))
    subject_mean, subject_std, subject_nobs = method_test_stats(summary, group_mask, spec, subject)
    comparator_mean, comparator_std, comparator_nobs = method_test_stats(summary, group_mask, spec, comparator)
    if not all(np.isfinite(v) for v in [subject_mean, subject_std, comparator_mean, comparator_std]):
        return None
    if subject_std == 0 and comparator_std == 0:
        t_stat = 0.0 if np.isclose(subject_mean, comparator_mean) else np.inf
        p_value = 1.0 if np.isclose(subject_mean, comparator_mean) else 0.0
    else:
        t_stat, p_value = pt.welch_from_stats(
            subject_mean,
            subject_std,
            subject_nobs,
            comparator_mean,
            comparator_std,
            comparator_nobs,
        )
    return {
        "table": table_name,
        "group": group_label,
        "metric": spec.label,
        "best": subject,
        "comparator": comparator,
        "best_display_mean": subject_display_mean,
        "best_display_std": subject_display_std,
        "comparator_display_mean": comparator_display_mean,
        "comparator_display_std": comparator_display_std,
        "best_test_mean": subject_mean,
        "best_test_std": subject_std,
        "best_nobs": subject_nobs,
        "comparator_test_mean": comparator_mean,
        "comparator_test_std": comparator_std,
        "comparator_nobs": comparator_nobs,
        "t": float(t_stat),
        "p_value": float(p_value),
        "is_runner_up": False,
        "test_role": test_role,
        "marker_printed_on": marker_printed_on,
    }


def top_vs_next_tests(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    methods: list[str],
    *,
    table_name: str = "",
    group_label: str = "",
) -> pd.DataFrame:
    stats = group_metric_stats(summary, group_mask, spec, methods)
    ranked = ranked_methods(stats, spec)
    if len(ranked) < 2:
        return pd.DataFrame()
    bold = statistically_tied_methods(summary, group_mask, spec, methods)
    top_methods = []
    next_method = None
    for method in ranked:
        if method in bold and next_method is None:
            top_methods.append(method)
        else:
            next_method = method
            break
    if not top_methods or next_method is None:
        return pd.DataFrame()
    records = []
    for method in top_methods:
        record = method_pair_test_record(
            summary,
            group_mask,
            spec,
            method,
            next_method,
            table_name=table_name,
            group_label=group_label,
            test_role="marker_top_vs_next_nonbold",
            marker_printed_on=method,
        )
        if record is not None:
            records.append(record)
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests["p_bh"] = np.nan
    finite_idx = tests[np.isfinite(pd.to_numeric(tests["p_value"], errors="coerce"))].index
    if len(finite_idx):
        _, p_bh = pt.bh_adjust(tests.loc[finite_idx, "p_value"], alpha=BOLD_ALPHA)
        tests.loc[finite_idx, "p_bh"] = p_bh
    tests["marker_bh"] = tests["p_bh"].map(pt.sig_marker)
    tests["comparator_tied"] = False
    return tests


def top_cell_markers(
    summary: pd.DataFrame,
    group_mask: pd.Series,
    spec: pt.MetricSpec,
    methods: list[str],
) -> dict[str, str]:
    tests = top_vs_next_tests(summary, group_mask, spec, methods)
    if tests.empty:
        return {}
    return {
        str(row.marker_printed_on): str(row.marker_bh)
        for row in tests.itertuples(index=False)
    }


def array_best_vs_comparator_tests(
    values: dict[str, np.ndarray],
    *,
    higher_is_better: bool,
    table_name: str = "",
    group_label: str = "",
    metric: str = "",
) -> pd.DataFrame:
    means = {
        label: float(np.nanmean(vals))
        for label, vals in values.items()
        if len(np.asarray(vals)) and np.isfinite(np.asarray(vals, dtype=float)).any()
    }
    if not means:
        return pd.DataFrame()
    order = {label: index for index, label in enumerate(values)}
    ranked = sorted(
        means,
        key=lambda label: ((-means[label]) if higher_is_better else means[label], order[label]),
    )
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    best_vals = np.asarray(values[best], dtype=float)
    best_vals = best_vals[np.isfinite(best_vals)]
    records = []
    for label, vals in values.items():
        if label == best:
            continue
        comp_vals = np.asarray(vals, dtype=float)
        comp_vals = comp_vals[np.isfinite(comp_vals)]
        if len(best_vals) < 2 or len(comp_vals) < 2:
            continue
        t_stat, p_value = pt.welch_from_arrays(best_vals, comp_vals)
        records.append(
            {
                "table": table_name,
                "group": group_label,
                "metric": metric,
                "best": best,
                "comparator": label,
                "best_display_mean": float(np.nanmean(best_vals)),
                "best_display_std": float(np.nanstd(best_vals, ddof=1)),
                "comparator_display_mean": float(np.nanmean(comp_vals)),
                "comparator_display_std": float(np.nanstd(comp_vals, ddof=1)),
                "best_test_mean": float(np.nanmean(best_vals)),
                "best_test_std": float(np.nanstd(best_vals, ddof=1)),
                "best_nobs": len(best_vals),
                "comparator_test_mean": float(np.nanmean(comp_vals)),
                "comparator_test_std": float(np.nanstd(comp_vals, ddof=1)),
                "comparator_nobs": len(comp_vals),
                "t": float(t_stat),
                "p_value": float(p_value),
                "is_runner_up": label == runner,
                "test_role": "bolding_best_vs_comparator",
                "marker_printed_on": "",
            }
        )
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests["p_bh"] = np.nan
    finite_idx = tests[np.isfinite(pd.to_numeric(tests["p_value"], errors="coerce"))].index
    if len(finite_idx):
        _, p_bh = pt.bh_adjust(tests.loc[finite_idx, "p_value"], alpha=BOLD_ALPHA)
        tests.loc[finite_idx, "p_bh"] = p_bh
    tests["marker_bh"] = tests["p_bh"].map(pt.sig_marker)
    tests["comparator_tied"] = tests["p_bh"].ge(BOLD_ALPHA)
    return tests


def array_top_vs_next_tests(
    values: dict[str, np.ndarray],
    *,
    higher_is_better: bool,
    table_name: str = "",
    group_label: str = "",
    metric: str = "",
) -> pd.DataFrame:
    means = {
        label: float(np.nanmean(vals))
        for label, vals in values.items()
        if len(np.asarray(vals)) and np.isfinite(np.asarray(vals, dtype=float)).any()
    }
    if len(means) < 2:
        return pd.DataFrame()
    order = {label: index for index, label in enumerate(values)}
    ranked = sorted(
        means,
        key=lambda label: ((-means[label]) if higher_is_better else means[label], order[label]),
    )
    bold = statistically_tied_arrays(values, higher_is_better=higher_is_better)
    top_labels = []
    next_label = None
    for label in ranked:
        if label in bold and next_label is None:
            top_labels.append(label)
        else:
            next_label = label
            break
    if not top_labels or next_label is None:
        return pd.DataFrame()
    next_vals = np.asarray(values[next_label], dtype=float)
    next_vals = next_vals[np.isfinite(next_vals)]
    records = []
    for label in top_labels:
        subject_vals = np.asarray(values[label], dtype=float)
        subject_vals = subject_vals[np.isfinite(subject_vals)]
        if len(subject_vals) < 2 or len(next_vals) < 2:
            continue
        t_stat, p_value = pt.welch_from_arrays(subject_vals, next_vals)
        records.append(
            {
                "table": table_name,
                "group": group_label,
                "metric": metric,
                "best": label,
                "comparator": next_label,
                "best_display_mean": float(np.nanmean(subject_vals)),
                "best_display_std": float(np.nanstd(subject_vals, ddof=1)),
                "comparator_display_mean": float(np.nanmean(next_vals)),
                "comparator_display_std": float(np.nanstd(next_vals, ddof=1)),
                "best_test_mean": float(np.nanmean(subject_vals)),
                "best_test_std": float(np.nanstd(subject_vals, ddof=1)),
                "best_nobs": len(subject_vals),
                "comparator_test_mean": float(np.nanmean(next_vals)),
                "comparator_test_std": float(np.nanstd(next_vals, ddof=1)),
                "comparator_nobs": len(next_vals),
                "t": float(t_stat),
                "p_value": float(p_value),
                "is_runner_up": False,
                "test_role": "marker_top_vs_next_nonbold",
                "marker_printed_on": label,
            }
        )
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests["p_bh"] = np.nan
    finite_idx = tests[np.isfinite(pd.to_numeric(tests["p_value"], errors="coerce"))].index
    if len(finite_idx):
        _, p_bh = pt.bh_adjust(tests.loc[finite_idx, "p_value"], alpha=BOLD_ALPHA)
        tests.loc[finite_idx, "p_bh"] = p_bh
    tests["marker_bh"] = tests["p_bh"].map(pt.sig_marker)
    tests["comparator_tied"] = False
    return tests


def statistically_tied_arrays(
    values: dict[str, np.ndarray],
    *,
    higher_is_better: bool,
    alpha: float = BOLD_ALPHA,
) -> set[str]:
    tests = array_best_vs_comparator_tests(values, higher_is_better=higher_is_better)
    means = {
        label: float(np.nanmean(vals))
        for label, vals in values.items()
        if len(np.asarray(vals)) and np.isfinite(np.asarray(vals, dtype=float)).any()
    }
    if not means:
        return set()
    order = {label: index for index, label in enumerate(values)}
    best = sorted(
        means,
        key=lambda label: ((-means[label]) if higher_is_better else means[label], order[label]),
    )[0]
    bold = {best}
    if not tests.empty:
        for row in tests.itertuples(index=False):
            if np.isfinite(row.p_bh) and row.p_bh >= alpha:
                bold.add(str(row.comparator))
    return bold


def array_top_cell_markers(values: dict[str, np.ndarray], *, higher_is_better: bool) -> dict[str, str]:
    tests = array_top_vs_next_tests(values, higher_is_better=higher_is_better)
    if tests.empty:
        return {}
    return {
        str(row.marker_printed_on): str(row.marker_bh)
        for row in tests.itertuples(index=False)
    }


def markdown_marker(marker: str | None) -> str:
    if not marker or marker == "ns":
        return ""
    return " " + marker.replace("*", r"\*")


def fmt_cell(mean: float, std: float, *, bold: bool = False, marker: str | None = None) -> str:
    if not np.isfinite(mean):
        return "--"
    cell = pt.fmt_num(mean, ndigits=3) if not np.isfinite(std) else pt.fmt_pm(mean, std, ndigits=3)
    if bold:
        return f"**{cell}**{markdown_marker(marker)}"
    return cell


def metric_by_group_table(
    summary: pd.DataFrame,
    metrics: MetricList,
    groups: list[GroupSpec],
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    rows = []
    for spec in metrics:
        group_masks = {group_label: group_fn(summary) for group_label, group_fn in groups}
        group_stats = {
            group_label: group_metric_stats(summary, group_mask, spec, methods)
            for group_label, group_mask in group_masks.items()
        }
        group_bold = {
            group_label: statistically_tied_methods(summary, group_mask, spec, tested_methods)
            for group_label, group_mask in group_masks.items()
        }
        group_marker = {
            group_label: top_cell_markers(summary, group_mask, spec, tested_methods)
            for group_label, group_mask in group_masks.items()
        }
        for method in methods:
            row = {"Metric": metric_label(spec), "Method": method_display(method, method_labels)}
            for group_label, _ in groups:
                mean, std = group_stats[group_label].get(method, (np.nan, np.nan))
                marker = group_marker[group_label].get(method)
                row[group_label] = fmt_cell(mean, std, bold=method in group_bold[group_label], marker=marker)
            rows.append(row)
    return pd.DataFrame(rows)


def node_groups() -> list[GroupSpec]:
    return [(f"V={nodes}", lambda df, nodes=nodes: df["n_nodes"].eq(nodes)) for nodes in [5, 10, 15]]


def graph_type_groups() -> list[GroupSpec]:
    return [(graph_type, lambda df, graph_type=graph_type: df["graph_type_label"].eq(graph_type)) for graph_type in ["ER", "SF", "LT"]]


def v_graph_type_groups(summary: pd.DataFrame | None = None) -> list[GroupSpec]:
    graph_types = ["ER", "SF", "LT"]
    if summary is not None and "graph_type_label" in summary:
        present = set(summary["graph_type_label"].dropna())
        graph_types = [graph_type for graph_type in graph_types if graph_type in present]
    groups = [
        (
            f"{graph_type} V={nodes}",
            lambda df, graph_type=graph_type, nodes=nodes: df["graph_type_label"].eq(graph_type) & df["n_nodes"].eq(nodes),
        )
        for graph_type in graph_types
        for nodes in [5, 10, 15]
    ]
    groups.extend(
        [
            (
                f"Average V={nodes}",
                lambda df, nodes=nodes: df["n_nodes"].eq(nodes),
            )
            for nodes in [5, 10, 15]
        ]
    )
    groups.extend(
        [
            (
                f"{graph_type} Average",
                lambda df, graph_type=graph_type: df["graph_type_label"].eq(graph_type),
            )
            for graph_type in graph_types
        ]
    )
    groups.append(("Average Average", lambda df: pd.Series(True, index=df.index)))
    return groups


def cross_table_cell_mask(summary: pd.DataFrame, nodes: int | str, graph_type: str) -> pd.Series:
    mask = pd.Series(True, index=summary.index)
    if nodes != "Average":
        mask &= summary["n_nodes"].eq(nodes)
    if graph_type != "Average":
        mask &= summary["graph_type_label"].eq(graph_type)
    return mask


def load_optional_alpha05_summary(tables_dir: Path) -> pd.DataFrame | None:
    candidates = [
        tables_dir / "synthetic_structural_summary_g2a05.csv",
        tables_dir / "archive" / "current_20260705_120330" / "synthetic_structural_summary_g2a05.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def ensure_intermediate_structural_tables(args: argparse.Namespace) -> None:
    """Create summary CSVs needed by the final presentation layer if absent."""

    tables_dir = Path(args.tables_dir)
    required = [
        tables_dir / f"synthetic_structural_summary_{args.tag}.csv",
    ]
    optional_alpha = tables_dir / "synthetic_structural_summary_g2a05.csv"
    if all(path.exists() for path in required) and optional_alpha.exists():
        return

    tables_dir.mkdir(parents=True, exist_ok=True)
    source_log = pt.SourceLog(rows=[])

    def write_summary(tag: str, graph_kind: str, *, allow_missing: bool) -> None:
        pt_args = argparse.Namespace(
            tag=tag,
            results_dir=str(args.results_dir),
            synthetic_dir=str(args.synthetic_dir),
            out_dir=str(args.tables_dir),
            graph_kind=graph_kind,
            allow_missing=allow_missing,
        )
        summary = pt.structural_summary(pt_args, source_log, graph_kind=graph_kind)
        if summary is None:
            return
        stem = (
            f"synthetic_structural_summary_{tag}"
            if graph_kind == "dag"
            else f"synthetic_{graph_kind}_structural_summary_{tag}"
        )
        summary_path = tables_dir / f"{stem}.csv"
        summary.to_csv(summary_path, index=False)
        source_log.add(stem, summary_path, "written")
        pt.build_structural_metric_tables(summary, tables_dir, tag, graph_kind=graph_kind)
        pt.build_tests_table(summary, tables_dir, tag, graph_kind=graph_kind)
        if graph_kind == "dag":
            pt.build_runtime_table(summary, tables_dir, tag)

    if not required[0].exists():
        write_summary(args.tag, "dag", allow_missing=False)
    cpdag_path = tables_dir / f"synthetic_cpdag_structural_summary_{args.tag}.csv"
    if not cpdag_path.exists():
        write_summary(args.tag, "cpdag", allow_missing=True)
    if not optional_alpha.exists():
        write_summary("g2a05", "dag", allow_missing=True)

    source_log.write(tables_dir / f"paper_table_sources_{args.tag}.csv")


def load_structural_summary(tables_dir: Path, stem: str) -> pd.DataFrame:
    path = tables_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}")
    return pd.read_csv(path)


def append_llm_bfs_dag_rows(summary: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or summary["model"].eq("LLM-BFS").any():
        return summary
    raw = pd.read_csv(csv_path)
    raw = raw[raw["impl"].astype(str).str.lower().eq("llm-bfs")].copy()
    if raw.empty:
        return summary

    metadata_columns = [
        "n_nodes",
        "filename_n_edges",
        "n_edges",
        "graph_type",
        "graph_type_label",
        "graph_kind",
    ]
    metadata = {
        dataset: sub.iloc[0].to_dict()
        for dataset, sub in summary.groupby("dataset", sort=False)
    }
    rows = []
    for row in raw.itertuples(index=False):
        dataset = str(row.dataset)
        meta = metadata.get(dataset, {})
        n_edges = pd.to_numeric(meta.get("n_edges"), errors="coerce")
        out = {
            "dataset": dataset,
            "model": "LLM-BFS",
            "source_kind": "csv_single_run",
        }
        for col in metadata_columns:
            if col in meta:
                out[col] = meta[col]
        for metric, raw_col in [
            ("elapsed", "time"),
            ("nnz", "dag_nnz"),
            ("fdr", "dag_fdr"),
            ("tpr", "dag_tpr"),
            ("fpr", "dag_fpr"),
            ("precision", "dag_precision"),
            ("recall", "dag_recall"),
            ("F1", "dag_F1"),
            ("shd", "dag_shd"),
            ("SID", "dag_sid"),
        ]:
            out[f"{metric}_mean"] = float(pd.to_numeric(getattr(row, raw_col), errors="coerce"))
            out[f"{metric}_std"] = np.nan
        out["SHD_mean"] = out["shd_mean"]
        out["SHD_std"] = out["shd_std"]
        out["SID_low_mean"] = out["SID_mean"]
        out["SID_low_std"] = out["SID_std"]
        out["SID_high_mean"] = out["SID_mean"]
        out["SID_high_std"] = out["SID_std"]
        if np.isfinite(n_edges) and n_edges:
            out["NSHD_mean"] = out["shd_mean"] / n_edges
            out["NSHD_std"] = np.nan
            out["NSID_mean"] = out["SID_mean"] / n_edges
            out["NSID_std"] = np.nan
            out["NSID_low_mean"] = out["SID_low_mean"] / n_edges
            out["NSID_low_std"] = np.nan
            out["NSID_high_mean"] = out["SID_high_mean"] / n_edges
            out["NSID_high_std"] = np.nan
        rows.append(out)
    if not rows:
        return summary
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False)


def abapc_row_n_nodes(dataset: str, sub: pd.DataFrame) -> int:
    if str(dataset).startswith("dag_"):
        return pt.n_nodes(dataset)
    return int(sub["num_nodes"].iloc[0])


def aggregate_abapc_cpdag_rows(
    csv_path: Path,
    *,
    label_by_impl: dict[str, str],
    consensus_path: Path | None = None,
    fallback_csv: Path | None = None,
) -> pd.DataFrame:
    fallback_paths = [fallback_csv] if fallback_csv is not None else []
    df = pt.load_abapc_csv(csv_path, consensus_path=consensus_path, fallback_paths=fallback_paths)
    rows = []
    for (dataset, impl), sub in df.groupby(["dataset", "impl"]):
        if impl not in label_by_impl:
            continue
        n_edges = float(sub["num_edges"].iloc[0])
        row = {
            "dataset": dataset,
            "model": label_by_impl[impl],
            "n_nodes": abapc_row_n_nodes(dataset, sub),
            "n_edges": n_edges,
            "F1_mean": float(sub["cpdag_F1"].mean()),
            "F1_std": float(sub["cpdag_F1"].std(ddof=1)),
            "NSHD_mean": float(sub["cpdag_shd"].mean() / n_edges),
            "NSHD_std": float(sub["cpdag_shd"].std(ddof=1) / n_edges),
            "precision_mean": float(sub["cpdag_precision"].mean()),
            "precision_std": float(sub["cpdag_precision"].std(ddof=1)),
            "recall_mean": float(sub["cpdag_recall"].mean()),
            "recall_std": float(sub["cpdag_recall"].std(ddof=1)),
            "NSID_low_mean": float(sub["cpdag_sid_low"].mean() / n_edges),
            "NSID_low_std": float(sub["cpdag_sid_low"].std(ddof=1) / n_edges),
            "NSID_high_mean": float(sub["cpdag_sid_high"].mean() / n_edges),
            "NSID_high_std": float(sub["cpdag_sid_high"].std(ddof=1) / n_edges),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_abapc_dag_rows(
    csv_path: Path,
    *,
    label_by_impl: dict[str, str],
    consensus_path: Path | None = None,
    fallback_csv: Path | None = None,
) -> pd.DataFrame:
    fallback_paths = [fallback_csv] if fallback_csv is not None else []
    df = pt.load_abapc_csv(csv_path, consensus_path=consensus_path, fallback_paths=fallback_paths)
    rows = []
    for (dataset, impl), sub in df.groupby(["dataset", "impl"]):
        if impl not in label_by_impl:
            continue
        n_edges = float(sub["num_edges"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "model": label_by_impl[impl],
                "n_nodes": abapc_row_n_nodes(dataset, sub),
                "n_edges": n_edges,
                "F1_mean": float(sub["dag_F1"].mean()),
                "F1_std": float(sub["dag_F1"].std(ddof=1)),
                "shd_mean": float(sub["dag_shd"].mean()),
                "shd_std": float(sub["dag_shd"].std(ddof=1)),
                "SHD_mean": float(sub["dag_shd"].mean()),
                "SHD_std": float(sub["dag_shd"].std(ddof=1)),
                "SID_mean": float(sub["dag_sid"].mean()),
                "SID_std": float(sub["dag_sid"].std(ddof=1)),
                "NSHD_mean": float(sub["dag_shd"].mean() / n_edges),
                "NSHD_std": float(sub["dag_shd"].std(ddof=1) / n_edges),
                "precision_mean": float(sub["dag_precision"].mean()),
                "precision_std": float(sub["dag_precision"].std(ddof=1)),
                "recall_mean": float(sub["dag_recall"].mean()),
                "recall_std": float(sub["dag_recall"].std(ddof=1)),
                "NSID_mean": float(sub["dag_sid"].mean() / n_edges),
                "NSID_std": float(sub["dag_sid"].std(ddof=1) / n_edges),
            }
        )
    return pd.DataFrame(rows)


def cpdag_npy_rows(path: Path, label_by_model: dict[str, str], edge_counts: dict[str, float] | None = None) -> pd.DataFrame:
    df = pt.load_npy_summary(path, graph_kind="cpdag")
    df = df[df["model"].isin(label_by_model)].copy()
    df["model"] = df["model"].map(label_by_model)
    df["n_nodes"] = df["dataset"].map(lambda dataset: pt.n_nodes(dataset) if str(dataset).startswith("dag_") else np.nan)
    if edge_counts is None:
        df["n_edges"] = df["dataset"].map(lambda dataset: pt.structural_n_edges(dataset, REPO_ROOT / "synthetic"))
    else:
        df["n_edges"] = df["dataset"].map(edge_counts)
    for col in ["F1", "precision", "recall", "shd", "SID_low", "SID_high"]:
        for stat in ["mean", "std"]:
            df[f"{col}_{stat}"] = pd.to_numeric(df[f"{col}_{stat}"], errors="coerce")
    df["NSHD_mean"] = df["shd_mean"] / df["n_edges"]
    df["NSHD_std"] = df["shd_std"] / df["n_edges"]
    df["NSID_low_mean"] = df["SID_low_mean"] / df["n_edges"]
    df["NSID_low_std"] = df["SID_low_std"] / df["n_edges"]
    df["NSID_high_mean"] = df["SID_high_mean"] / df["n_edges"]
    df["NSID_high_std"] = df["SID_high_std"] / df["n_edges"]
    return df


def dag_npy_rows(path: Path, label_by_model: dict[str, str], edge_counts: dict[str, float] | None = None) -> pd.DataFrame:
    df = pt.load_npy_summary(path, graph_kind="dag")
    df = df[df["model"].isin(label_by_model)].copy()
    df["model"] = df["model"].map(label_by_model)
    df["n_nodes"] = df["dataset"].map(lambda dataset: pt.n_nodes(dataset) if str(dataset).startswith("dag_") else np.nan)
    if edge_counts is None:
        df["n_edges"] = df["dataset"].map(
            lambda dataset: pt.structural_n_edges(dataset, REPO_ROOT / "synthetic")
            if str(dataset).startswith("dag_")
            else np.nan
        )
    else:
            df["n_edges"] = df["dataset"].map(edge_counts)
    for col in ["F1", "shd", "precision", "recall", "SID"]:
        for stat in ["mean", "std"]:
            df[f"{col}_{stat}"] = pd.to_numeric(df[f"{col}_{stat}"], errors="coerce")
    df = df.rename(columns={"shd_mean": "SHD_mean", "shd_std": "SHD_std"})
    df["shd_mean"] = df["SHD_mean"]
    df["shd_std"] = df["SHD_std"]
    df["NSHD_mean"] = df["SHD_mean"] / df["n_edges"]
    df["NSHD_std"] = df["SHD_std"] / df["n_edges"]
    df["NSID_mean"] = df["SID_mean"] / df["n_edges"]
    df["NSID_std"] = df["SID_std"] / df["n_edges"]
    return df


def build_bnlearn_cpdag_summary(results_dir: Path) -> pd.DataFrame:
    abapc_csv = results_dir / "ABAPC-LLM" / "merged_bnlearn-desc_g2a01.csv"
    fallback_csv = results_dir / "ABAPC-LLM" / "merged_bnlearn_g2a01.csv"
    consensus_path = results_dir / "llm_constraints" / "bnlearn-desc-consensus.json"
    abapc = aggregate_abapc_cpdag_rows(
        abapc_csv,
        label_by_impl={"org": "ABAPC", "new": "ABAPC-LLM"},
        consensus_path=consensus_path,
        fallback_csv=fallback_csv,
    )
    edge_counts = dict(zip(abapc["dataset"], abapc["n_edges"]))
    frames = [
        cpdag_npy_rows(
            results_dir / "stored_results_bnlearn_boss_grasp_cpdag.npy",
            {"GRaSP": "GRaSP", "BOSS": "BOSS"},
            edge_counts,
        ),
        cpdag_npy_rows(
            results_dir / "stored_results_bnlearn_mpc_desc_g2a01_cpdag.npy",
            {"MPC": "MPC", "MPC-LLM": "MPC-LLM"},
            edge_counts,
        ),
        abapc,
    ]
    return pd.concat(frames, ignore_index=True)


def build_bnlearn_dag_summary(results_dir: Path) -> pd.DataFrame:
    abapc_csv = results_dir / "ABAPC-LLM" / "merged_bnlearn-desc_g2a01.csv"
    fallback_csv = results_dir / "ABAPC-LLM" / "merged_bnlearn_g2a01.csv"
    consensus_path = results_dir / "llm_constraints" / "bnlearn-desc-consensus.json"
    abapc = aggregate_abapc_dag_rows(
        abapc_csv,
        label_by_impl={"org": "ABAPC", "new": "ABAPC-LLM"},
        consensus_path=consensus_path,
        fallback_csv=fallback_csv,
    )
    edge_counts = dict(zip(abapc["dataset"], abapc["n_edges"]))
    frames = [
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_big_rnd_mpc2.npy",
            {"Random": "Random"},
            edge_counts,
        ),
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_child_base.npy",
            {"Random": "Random"},
            edge_counts,
        ),
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_big_fgs_nt.npy",
            {"FGS": "FGS", "NOTEARS-MLP": "NOTEARS-MLP"},
            edge_counts,
        ),
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_child_base2.npy",
            {"FGS": "FGS", "NOTEARS-MLP": "NOTEARS-MLP"},
            edge_counts,
        ),
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_boss_grasp.npy",
            {"GRaSP": "GRaSP", "BOSS": "BOSS"},
            edge_counts,
        ),
        dag_npy_rows(
            results_dir / "stored_results_bnlearn_mpc_desc_g2a01.npy",
            {"MPC": "MPC", "MPC-LLM": "MPC-LLM"},
            edge_counts,
        ),
        abapc,
    ]
    summary = pd.concat(frames, ignore_index=True)
    return append_llm_bfs_dag_rows(summary, results_dir / "causal-bfs-bnlearn-results.csv")


def bnlearn_dataset_label(summary: pd.DataFrame, dataset: str, *, latex: bool = False) -> str:
    sub = summary[summary["dataset"].eq(dataset)]
    if sub.empty:
        return tex_escape(dataset) if latex else dataset
    nodes = int(pd.to_numeric(sub["n_nodes"], errors="coerce").dropna().iloc[0])
    edges = int(pd.to_numeric(sub["n_edges"], errors="coerce").dropna().iloc[0])
    if latex:
        return rf"\shortstack[l]{{\texttt{{{tex_escape(dataset)}}}\\$|\nodeSet|={nodes}$\\$|\edgSet|={edges}$}}"
    return f"{dataset} (|V|={nodes}, |E|={edges})"


def build_causenet_dag_full_table(
    summary: pd.DataFrame,
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    rows = []
    for nodes in [5, 10, 15]:
        group_mask = summary["n_nodes"].eq(nodes)
        bold_by_metric = {
            spec.label: statistically_tied_methods(summary, group_mask, spec, tested_methods)
            for spec in DAG_RAW_METRICS
        }
        marker_by_metric = {
            spec.label: top_cell_markers(summary, group_mask, spec, tested_methods)
            for spec in DAG_RAW_METRICS
        }
        for method in methods:
            sub = summary[group_mask & summary["model"].eq(method)]
            if sub.empty:
                continue
            row = {"V": nodes, "Method": method_display(method, method_labels)}
            for spec in DAG_RAW_METRICS:
                mean = float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean())
                std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
                marker = marker_by_metric[spec.label].get(method)
                row[metric_label(spec)] = fmt_cell(
                    mean,
                    std,
                    bold=method in bold_by_metric[spec.label],
                    marker=marker,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_causenet_dag_v_graph_type_table(summary: pd.DataFrame) -> pd.DataFrame:
    methods = present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    graph_types = [graph_type for graph_type in ["ER", "SF", "LT"] if summary["graph_type_label"].eq(graph_type).any()]
    columns = [*graph_types, "Average"]
    rows = []
    for nodes in [5, 10, 15, "Average"]:
        for spec in DAG_MAIN_METRICS:
            group_masks = {
                graph_type: cross_table_cell_mask(summary, nodes, graph_type)
                for graph_type in columns
            }
            group_stats = {
                graph_type: group_metric_stats(summary, mask, spec, methods)
                for graph_type, mask in group_masks.items()
            }
            group_bold = {
                graph_type: statistically_tied_methods(summary, mask, spec, tested_methods)
                for graph_type, mask in group_masks.items()
            }
            group_marker = {
                graph_type: top_cell_markers(summary, mask, spec, tested_methods)
                for graph_type, mask in group_masks.items()
            }
            for method in methods:
                row = {"V": nodes, "Metric": metric_label(spec), "Method": method}
                for graph_type in columns:
                    mean, std = group_stats[graph_type].get(method, (np.nan, np.nan))
                    marker = group_marker[graph_type].get(method)
                    row[graph_type] = fmt_cell(mean, std, bold=method in group_bold[graph_type], marker=marker)
                rows.append(row)
    return pd.DataFrame(rows)


def build_alpha_ablation_table(summary01: pd.DataFrame, summary05: pd.DataFrame) -> pd.DataFrame:
    summary05 = carry_display_only_rows(summary01, summary05)
    methods = present_methods(summary01, CAUSENET_DAG_METHODS)
    rows = []
    for alpha_label, summary in [("0.01", summary01), ("0.05", summary05)]:
        methods_present = [method for method in methods if summary["model"].eq(method).any()]
        tested_methods = testable_methods(methods_present)
        for spec in DAG_MAIN_METRICS:
            group_masks = {nodes: summary["n_nodes"].eq(nodes) for nodes in [5, 10, 15]}
            group_stats = {
                nodes: group_metric_stats(summary, group_mask, spec, methods_present)
                for nodes, group_mask in group_masks.items()
            }
            group_bold = {
                nodes: statistically_tied_methods(summary, group_mask, spec, tested_methods)
                for nodes, group_mask in group_masks.items()
            }
            group_marker = {
                nodes: top_cell_markers(summary, group_mask, spec, tested_methods)
                for nodes, group_mask in group_masks.items()
            }
            for method in methods_present:
                row = {"Alpha": alpha_label, "Metric": metric_label(spec), "Method": method}
                for nodes in [5, 10, 15]:
                    mean, std = group_stats[nodes].get(method, (np.nan, np.nan))
                    row[f"$|\\nodeSet|={nodes}$"] = fmt_cell(
                        mean,
                        std,
                        bold=method in group_bold[nodes],
                        marker=group_marker[nodes].get(method),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_bnlearn_cpdag_table(results_dir: Path) -> pd.DataFrame:
    summary = build_bnlearn_cpdag_summary(results_dir)
    metrics = [
        pt.STRUCTURAL_METRICS["NSHD"],
        pt.STRUCTURAL_METRICS["F1"],
        pt.STRUCTURAL_METRICS["NSID-low"],
        pt.STRUCTURAL_METRICS["NSID-high"],
        pt.STRUCTURAL_METRICS["Precision"],
        pt.STRUCTURAL_METRICS["Recall"],
    ]
    rows = []
    dataset_order = ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
    for dataset in [d for d in dataset_order if summary["dataset"].eq(d).any()]:
        sub_dataset = summary[summary["dataset"].eq(dataset)]
        dataset_mask = summary["dataset"].eq(dataset)
        bold_by_metric = {}
        marker_by_metric = {}
        for spec in metrics:
            bold_by_metric[spec.label] = statistically_tied_methods(
                summary,
                dataset_mask,
                spec,
                BNLEARN_CPDAG_METHODS,
            )
            marker_by_metric[spec.label] = top_cell_markers(summary, dataset_mask, spec, BNLEARN_CPDAG_METHODS)
        for method in BNLEARN_CPDAG_METHODS:
            sub = sub_dataset[sub_dataset["model"].eq(method)]
            if sub.empty:
                continue
            row = {"Dataset": dataset, "Method": method}
            for spec in metrics:
                mean = float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean())
                std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
                marker = marker_by_metric[spec.label].get(method)
                row[metric_label(spec)] = fmt_cell(
                    mean,
                    std,
                    bold=method in bold_by_metric[spec.label],
                    marker=marker,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_bnlearn_dag_table(results_dir: Path) -> pd.DataFrame:
    summary = build_bnlearn_dag_summary(results_dir)
    metrics = DAG_ALL_METRICS
    rows = []
    dataset_order = ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
    for dataset in [d for d in dataset_order if summary["dataset"].eq(d).any()]:
        sub_dataset = summary[summary["dataset"].eq(dataset)]
        dataset_mask = summary["dataset"].eq(dataset)
        tested_methods = testable_methods(BNLEARN_DAG_METHODS)
        bold_by_metric = {}
        marker_by_metric = {}
        for spec in metrics:
            bold_by_metric[spec.label] = statistically_tied_methods(
                summary,
                dataset_mask,
                spec,
                tested_methods,
            )
            marker_by_metric[spec.label] = top_cell_markers(summary, dataset_mask, spec, tested_methods)
        for method in BNLEARN_DAG_METHODS:
            sub = sub_dataset[sub_dataset["model"].eq(method)]
            if sub.empty:
                continue
            row = {"Dataset": bnlearn_dataset_label(summary, dataset), "Method": method}
            for spec in metrics:
                mean = float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean())
                std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
                marker = marker_by_metric[spec.label].get(method)
                row[metric_label(spec)] = fmt_cell(
                    mean,
                    std,
                    bold=method in bold_by_metric[spec.label],
                    marker=marker,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def constraint_source_values(
    path: Path,
    *,
    side: str,
    metric: str,
    exclude_zero_constraints: bool = False,
) -> np.ndarray:
    df = pd.read_json(path)
    if exclude_zero_constraints:
        length_col = f"{side}_length"
        df = df[pd.to_numeric(df[length_col], errors="coerce").fillna(0) > 0]
    return pd.to_numeric(df[f"{side}_{metric}"], errors="coerce").to_numpy(dtype=float)


def constraint_source_stats(path: Path, *, side: str, metric: str) -> tuple[float, float]:
    values = constraint_source_values(path, side=side, metric=metric)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def build_desc_consensus_table(results_dir: Path, *, exclude_zero_constraints: bool = False) -> pd.DataFrame:
    variants = [
        ("Average no desc", "Average", "No", "base"),
        ("Average desc", "Average", "Yes", "desc"),
        ("Consensus no desc", "Consensus", "No", "consensus"),
        ("Consensus desc", "Consensus", "Yes", "desc-consensus"),
    ]
    sources = {
        ("bnlearn", "base"): results_dir / "llm_constraints" / "bnlearn.json",
        ("bnlearn", "desc"): results_dir / "llm_constraints" / "bnlearn-desc.json",
        ("bnlearn", "consensus"): results_dir / "llm_constraints" / "bnlearn-consensus.json",
        ("bnlearn", "desc-consensus"): results_dir / "llm_constraints" / "bnlearn-desc-consensus.json",
        ("CauseNet", "base"): results_dir / "llm_constraints" / "synthetic.json",
        ("CauseNet", "desc"): results_dir / "llm_constraints" / "synthetic-desc.json",
        ("CauseNet", "consensus"): results_dir / "llm_constraints" / "synthetic-consensus.json",
        ("CauseNet", "desc-consensus"): results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
    }
    rows = []
    for dataset in ["bnlearn", "CauseNet"]:
        for side_label, side in [("Forbidden", "forbidden"), ("Required", "required")]:
            for metric_label_text, metric, higher in [
                ("Count", "length", None),
                ("Precision", "Precision", True),
                ("Recall", "Recall", True),
                ("F1", "F1", True),
            ]:
                values = {}
                for col_label, _, _, key in variants:
                    values[col_label] = constraint_source_values(
                        sources[(dataset, key)],
                        side=side,
                        metric=metric,
                        exclude_zero_constraints=exclude_zero_constraints,
                    )
                bold_cols = set()
                markers = {}
                if higher is not None:
                    bold_cols = statistically_tied_arrays(values, higher_is_better=higher)
                    markers = array_top_cell_markers(values, higher_is_better=higher)
                row = {"Dataset": dataset, "Side": side_label, "Metric": metric_label_text}
                for col_label, _, _, _ in variants:
                    mean = float(np.nanmean(values[col_label]))
                    std = float(np.nanstd(values[col_label], ddof=1))
                    row[col_label] = fmt_cell(
                        mean,
                        std,
                        bold=col_label in bold_cols,
                        marker=markers.get(col_label),
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_gpt_ablation_summary(results_dir: Path, tag: str) -> pd.DataFrame:
    fallback_csv = results_dir / "ABAPC-LLM" / f"merged_synthetic_{tag}.csv"
    gemini = aggregate_abapc_cpdag_rows(
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc_{tag}.csv",
        label_by_impl={"org": "ABAPC", "new": "ABAPC-LLM"},
        consensus_path=results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
        fallback_csv=fallback_csv,
    )
    gpt = aggregate_abapc_cpdag_rows(
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc-gpt5mini_{tag}.csv",
        label_by_impl={"new": "ABAPC-LLM-gpt5mini"},
    )
    mpc = cpdag_npy_rows(
        results_dir / f"stored_results_causenet_mpc_desc_{tag}_cpdag.npy",
        {"MPC": "MPC", "MPC-LLM": "MPC-LLM"},
    )
    gpt_mpc = cpdag_npy_rows(
        results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}_cpdag.npy",
        {"MPC-LLM": "MPC-LLM-gpt5mini"},
    )
    summary = pd.concat([gemini, gpt, mpc, gpt_mpc], ignore_index=True)
    valid_sets = []
    for method in GPT_ABLATION_METHODS:
        sub = summary[summary["model"].eq(method)]
        finite = sub[np.isfinite(pd.to_numeric(sub["F1_mean"], errors="coerce")) & np.isfinite(pd.to_numeric(sub["NSHD_mean"], errors="coerce"))]
        valid_sets.append(set(finite["dataset"]))
    common = set.intersection(*valid_sets)
    return summary[summary["dataset"].isin(common)].copy()


def build_cpdag_main_with_gpt_summary(cpdag_summary: pd.DataFrame, gpt_summary: pd.DataFrame) -> pd.DataFrame:
    common_datasets = set(gpt_summary["dataset"])
    standard_methods = ["Random", "FGS", "NOTEARS-MLP", "GRaSP", "BOSS"]
    standard = cpdag_summary[
        cpdag_summary["dataset"].isin(common_datasets) & cpdag_summary["model"].isin(standard_methods)
    ].copy()
    gpt_rows = gpt_summary[gpt_summary["model"].isin(GPT_ABLATION_METHODS)].copy()
    return pd.concat([standard, gpt_rows], ignore_index=True)


def build_dag_main_with_gpt_summary(dag_summary: pd.DataFrame, gpt_dag_summary: pd.DataFrame) -> pd.DataFrame:
    common_datasets = set(gpt_dag_summary["dataset"])
    standard_methods = ["Random", "FGS", "NOTEARS-MLP", "GRaSP", "BOSS", "LLM-BFS"]
    standard = dag_summary[
        dag_summary["dataset"].isin(common_datasets) & dag_summary["model"].isin(standard_methods)
    ].copy()
    gpt_rows = gpt_dag_summary[gpt_dag_summary["model"].isin(GPT_ABLATION_METHODS)].copy()
    return pd.concat([standard, gpt_rows], ignore_index=True)


def build_gpt_ablation_dag_summary(results_dir: Path, tag: str) -> pd.DataFrame:
    fallback_csv = results_dir / "ABAPC-LLM" / f"merged_synthetic_{tag}.csv"
    gemini = aggregate_abapc_dag_rows(
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc_{tag}.csv",
        label_by_impl={"org": "ABAPC", "new": "ABAPC-LLM"},
        consensus_path=results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
        fallback_csv=fallback_csv,
    )
    gpt = aggregate_abapc_dag_rows(
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc-gpt5mini_{tag}.csv",
        label_by_impl={"new": "ABAPC-LLM-gpt5mini"},
    )
    mpc = dag_npy_rows(
        results_dir / f"stored_results_causenet_mpc_desc_{tag}.npy",
        {"MPC": "MPC", "MPC-LLM": "MPC-LLM"},
    )
    gpt_mpc = dag_npy_rows(
        results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}.npy",
        {"MPC-LLM": "MPC-LLM-gpt5mini"},
    )
    summary = pd.concat([gemini, gpt, mpc, gpt_mpc], ignore_index=True)
    valid_sets = []
    for method in GPT_ABLATION_METHODS:
        sub = summary[summary["model"].eq(method)]
        finite = sub[
            np.isfinite(pd.to_numeric(sub["F1_mean"], errors="coerce"))
            & np.isfinite(pd.to_numeric(sub["NSHD_mean"], errors="coerce"))
        ]
        valid_sets.append(set(finite["dataset"]))
    common = set.intersection(*valid_sets)
    return summary[summary["dataset"].isin(common)].copy()


def build_gpt_ablation_dag_summary_from_base(
    results_dir: Path,
    tag: str,
    dag_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the matched DAG GPT table using complete Gemini/plain rows.

    The refreshed MPC/MPC-LLM NPY artifact can be a partial rerun while
    ``synthetic_structural_summary`` already contains the complete structural
    summary used for the other final tables.  Use that complete base for
    Gemini/plain rows and only read the GPT-specific artifacts for GPT rows.
    """

    base = dag_summary[
        dag_summary["model"].isin(["ABAPC", "ABAPC-LLM", "MPC", "MPC-LLM"])
    ].copy()
    gpt = aggregate_abapc_dag_rows(
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc-gpt5mini_{tag}.csv",
        label_by_impl={"new": "ABAPC-LLM-gpt5mini"},
    )
    gpt_mpc = dag_npy_rows(
        results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}.npy",
        {"MPC-LLM": "MPC-LLM-gpt5mini"},
    )
    summary = pd.concat([base, gpt, gpt_mpc], ignore_index=True, sort=False)
    valid_sets = []
    for method in GPT_ABLATION_METHODS:
        sub = summary[summary["model"].eq(method)]
        finite = sub[
            np.isfinite(pd.to_numeric(sub["F1_mean"], errors="coerce"))
            & np.isfinite(pd.to_numeric(sub["NSHD_mean"], errors="coerce"))
        ]
        valid_sets.append(set(finite["dataset"]))
    common = set.intersection(*valid_sets)
    return summary[summary["dataset"].isin(common)].copy()


def gpt_mpc_difference_counts(results_dir: Path, tag: str) -> pd.DataFrame:
    rows = []
    for graph_kind, suffix, metrics in [
        ("DAG", "", ["F1_mean", "shd_mean", "precision_mean", "recall_mean"]),
        ("CPDAG", "_cpdag", ["F1_mean", "shd_mean", "precision_mean", "recall_mean"]),
    ]:
        kind = graph_kind.lower()
        gemini = pt.load_npy_summary(results_dir / f"stored_results_causenet_mpc_desc_{tag}{suffix}.npy", graph_kind=kind)
        gpt = pt.load_npy_summary(
            results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}{suffix}.npy",
            graph_kind=kind,
        )
        for frame in [gemini, gpt]:
            numeric_cols = [col for col in frame.columns if col not in {"dataset", "model"}]
            frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        variants = {
            "MPC": gemini[gemini["model"].eq("MPC")].set_index("dataset"),
            "MPC-LLM": gemini[gemini["model"].eq("MPC-LLM")].set_index("dataset"),
            "MPC-LLM-GPT": gpt[gpt["model"].eq("MPC-LLM")].set_index("dataset"),
        }
        common = set.intersection(*(set(frame.index) for frame in variants.values()))
        for left, right in [("MPC", "MPC-LLM"), ("MPC", "MPC-LLM-GPT"), ("MPC-LLM", "MPC-LLM-GPT")]:
            row = {"Graph": graph_kind, "Comparison": f"{left} vs {right}", "Datasets": len(common)}
            for metric in metrics:
                differences = (
                    variants[left].loc[list(common), metric].astype(float)
                    - variants[right].loc[list(common), metric].astype(float)
                ).abs()
                row[metric] = int((differences > 1e-12).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def write_gpt_ablation_audit(final_dir: Path, results_dir: Path, tag: str) -> None:
    sources = [
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc_{tag}.csv",
        results_dir / "ABAPC-LLM" / f"merged_synthetic-desc-gpt5mini_{tag}.csv",
        results_dir / f"stored_results_causenet_mpc_desc_{tag}.npy",
        results_dir / f"stored_results_causenet_mpc_desc_{tag}_cpdag.npy",
        results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}.npy",
        results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}_cpdag.npy",
        results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
        results_dir / "llm_constraints" / "synthetic-desc-gpt5mini-consensus.json",
        results_dir / "run_causenet_mpc_llm_gpt5mini_desc_g2a01.out",
        Path("results/tables/gpt5mini_llm_comparison_desc_summary.csv"),
    ]
    source_rows = []
    for path in sources:
        full_path = path if path.is_absolute() else REPO_ROOT / path
        if full_path.exists():
            stat = full_path.stat()
            source_rows.append(
                {
                    "Path": str(path),
                    "Exists": "yes",
                    "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "Bytes": stat.st_size,
                }
            )
        else:
            source_rows.append({"Path": str(path), "Exists": "no", "Modified": "--", "Bytes": "--"})
    diff_table = gpt_mpc_difference_counts(results_dir, tag)
    audit_text = "\n".join(
        [
            "# GPT Ablation Audit",
            "",
            "The old screenshot-style DAG table matches `results/tables/gpt5mini_llm_comparison_desc_summary.csv`, which was written before the refreshed MPC results. The current DAG check table is regenerated from the current saved inputs.",
            "",
            "Config notes:",
            "",
            "- `experiments.py` maps `--test_name g2` to causal-learn's `gsq` name before logging, so `test_name=gsq` in the run log is expected for the `g2a01` run.",
            "- The GPT MPC-LLM run log loads `results/llm_constraints/synthetic-desc-gpt5mini-consensus.json` and reports 54 constraint rows.",
            "- The currently saved CPDAG MPC-variant files were produced by the old `dag2cpdag((W_est != 0).astype(int))` path; that removes the signed endpoint encoding that separates MPC from MPC-LLM in the DAG view when the skeleton is unchanged. `experiments.py` now uses `estimate_to_cpdag_for_metrics(W_est)`, but those CPDAG summaries need rerunning before use.",
            "- The final-style check tables average the per-dataset run standard deviations. The old `gpt5mini_llm_comparison_desc_summary.csv` and `gemini_gpt5_structural_g2a01.csv` tables use standard deviation across dataset-level means, so compare the means directly but do not compare their `+/-` columns as the same quantity.",
            "",
            "## Source Files",
            "",
            markdown_table(pd.DataFrame(source_rows)),
            "",
            "## MPC Variant Difference Counts",
            "",
            "Each metric column reports how many of the 54 matched datasets differ between the two methods.",
            "",
            markdown_table(diff_table),
            "",
        ]
    )
    check_dir = final_dir / "check"
    check_dir.mkdir(parents=True, exist_ok=True)
    (check_dir / f"gpt_ablation_audit_{tag}.md").write_text(audit_text, encoding="utf-8")


def build_runtime_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in pt.SYNTH_METHOD_ORDER:
        sub_model = summary[summary["model"].eq(method)]
        if sub_model.empty:
            continue
        row = {"Method": method}
        for nodes in [5, 10, 15]:
            sub = sub_model[sub_model["n_nodes"].eq(nodes)]
            row[f"V={nodes}"] = pt.fmt_pm(
                float(pd.to_numeric(sub["elapsed_mean"], errors="coerce").mean()),
                float(pd.to_numeric(sub["elapsed_mean"], errors="coerce").std(ddof=1)),
                ndigits=2,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def tests_stem(tag: str, graph_kind: str) -> str:
    if graph_kind == "cpdag":
        return f"structural_cpdag_tests_abapcllm_bh_{tag}"
    if graph_kind == "dag":
        return f"structural_tests_abapcllm_bh_{tag}"
    raise ValueError(f"Unknown graph kind: {graph_kind}")


def build_tests_table(tables_dir: Path, tag: str, graph_kind: str = "cpdag") -> pd.DataFrame:
    tests = pd.read_csv(tables_dir / f"{tests_stem(tag, graph_kind)}.csv")
    rows = []
    for row in tests.itertuples(index=False):
        rows.append(
            {
                "V": int(row.nodes),
                "Metric": row.metric,
                "Comparison": row.comparison,
                "ABAPC-LLM": pt.fmt_pm(row.ref_mean, row.ref_std, ndigits=3),
                "Other": pt.fmt_pm(row.other_mean, row.other_std, ndigits=3),
                "t": f"{row.t:.2f}",
                "p_BH": fmt_p(row.p_bh),
                "Sig.": row.marker_bh,
            }
        )
    return pd.DataFrame(rows)


def metric_source_columns(spec: pt.MetricSpec) -> str:
    return f"{spec.mean_col}/{spec.std_col}"


def final_test_provenance(table_name: str, metric: str, tag: str) -> str:
    if table_name in {"CauseNet structural", "CauseNet type-size", "Matched LLM source"}:
        return (
            f"results/tables/synthetic_structural_summary_{tag}.csv; metric columns follow the table metric "
            "definition; display std is the mean of per-dataset run stds; Welch std is the std of per-dataset means."
        )
    if table_name == "Alpha ablation":
        return (
            f"results/tables/synthetic_structural_summary_{tag}.csv and synthetic_structural_summary_g2a05.csv; "
            "display std is the mean of per-dataset run stds; Welch std is the std of per-dataset means."
        )
    if table_name == "bnlearn":
        return (
            "bnlearn DAG rows reconstructed from saved ABAPC CSVs and baseline NPY result bundles; "
            "display std is the mean of per-dataset run stds; Welch std uses the available per-dataset/test std."
        )
    return "Derived from the same summary frame used by the corresponding final table."


def add_test_provenance(tests: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    if tests.empty:
        return tests
    tests = tests.copy()
    tests["provenance"] = [
        final_test_provenance(str(row.table), str(row.metric), tag)
        for row in tests.itertuples(index=False)
    ]
    return tests


def add_bh_by_family(tests: pd.DataFrame, family_cols: list[str]) -> pd.DataFrame:
    if tests.empty:
        return tests
    tests = tests.copy()
    tests["p_bh"] = np.nan
    for _, sub in tests.groupby(family_cols, dropna=False):
        finite_idx = sub[np.isfinite(pd.to_numeric(sub["p_value"], errors="coerce"))].index
        if len(finite_idx):
            _, p_bh = pt.bh_adjust(tests.loc[finite_idx, "p_value"], alpha=BOLD_ALPHA)
            tests.loc[finite_idx, "p_bh"] = p_bh
    tests["marker_bh"] = tests["p_bh"].map(pt.sig_marker)
    tests["comparator_tied"] = tests["p_bh"].ge(BOLD_ALPHA)
    return tests


def collect_subject_vs_comparator_tests(
    table_name: str,
    summary: pd.DataFrame,
    metrics: MetricList,
    groups: list[GroupSpec],
    methods: list[str],
    *,
    subject: str,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    records = []
    for spec in metrics:
        for group_label, group_fn in groups:
            group_mask = group_fn(summary)
            for comparator in methods:
                if comparator == subject or not summary["model"].eq(comparator).any():
                    continue
                record = method_pair_test_record(
                    summary,
                    group_mask,
                    spec,
                    subject,
                    comparator,
                    table_name=table_name,
                    group_label=group_label,
                    test_role="subject_vs_comparator",
                    marker_printed_on="",
                )
                if record is None:
                    continue
                record["metric_source_columns"] = metric_source_columns(spec)
                records.append(record)
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests = add_bh_by_family(tests, ["table", "group", "metric"])
    tests["marker_printed_on_result_cell"] = "no"
    tests["comparator_bolded_as_tied"] = tests["comparator_tied"].map(lambda value: "yes" if value else "no")
    if method_labels:
        tests["best"] = tests["best"].map(lambda method: method_display(method, method_labels))
        tests["comparator"] = tests["comparator"].map(lambda method: method_display(method, method_labels))
    return tests


def build_causenet_main_abapcllm_tests(dag_summary: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    methods = present_methods(dag_summary, CAUSENET_DAG_METHODS)
    tests = collect_subject_vs_comparator_tests(
        "CauseNet structural",
        dag_summary,
        DAG_RAW_METRICS,
        node_groups(),
        methods,
        subject="ABAPC-LLM",
    )
    return add_test_provenance(tests, tag=tag)


def collect_metric_group_tests(
    table_name: str,
    summary: pd.DataFrame,
    metrics: MetricList,
    groups: list[GroupSpec],
    methods: list[str],
    *,
    method_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    frames = []
    for spec in metrics:
        for group_label, group_fn in groups:
            tests = best_vs_comparator_tests(
                summary,
                group_fn(summary),
                spec,
                methods,
                table_name=table_name,
                group_label=group_label,
            )
            if tests.empty:
                continue
            tests = tests.copy()
            tests["best"] = tests["best"].map(lambda method: method_display(method, method_labels))
            tests["comparator"] = tests["comparator"].map(lambda method: method_display(method, method_labels))
            tests["marker_printed_on"] = tests["marker_printed_on"].map(
                lambda method: method_display(method, method_labels) if method else ""
            )
            frames.append(tests)
            marker_tests = top_vs_next_tests(
                summary,
                group_fn(summary),
                spec,
                methods,
                table_name=table_name,
                group_label=group_label,
            )
            if not marker_tests.empty:
                marker_tests = marker_tests.copy()
                marker_tests["best"] = marker_tests["best"].map(lambda method: method_display(method, method_labels))
                marker_tests["comparator"] = marker_tests["comparator"].map(lambda method: method_display(method, method_labels))
                marker_tests["marker_printed_on"] = marker_tests["marker_printed_on"].map(
                    lambda method: method_display(method, method_labels) if method else ""
                )
                frames.append(marker_tests)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_desc_consensus_tests(results_dir: Path) -> pd.DataFrame:
    variants = [
        ("Average no desc", "base"),
        ("Average desc", "desc"),
        ("Consensus no desc", "consensus"),
        ("Consensus desc", "desc-consensus"),
    ]
    sources = {
        ("bnlearn", "base"): results_dir / "llm_constraints" / "bnlearn.json",
        ("bnlearn", "desc"): results_dir / "llm_constraints" / "bnlearn-desc.json",
        ("bnlearn", "consensus"): results_dir / "llm_constraints" / "bnlearn-consensus.json",
        ("bnlearn", "desc-consensus"): results_dir / "llm_constraints" / "bnlearn-desc-consensus.json",
        ("CauseNet", "base"): results_dir / "llm_constraints" / "synthetic.json",
        ("CauseNet", "desc"): results_dir / "llm_constraints" / "synthetic-desc.json",
        ("CauseNet", "consensus"): results_dir / "llm_constraints" / "synthetic-consensus.json",
        ("CauseNet", "desc-consensus"): results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
    }
    frames = []
    for dataset in ["bnlearn", "CauseNet"]:
        for side_label, side in [("Forbidden", "forbidden"), ("Required", "required")]:
            for metric_label_text, metric in [
                ("Precision", "Precision"),
                ("Recall", "Recall"),
                ("F1", "F1"),
            ]:
                values = {
                    label: constraint_source_values(sources[(dataset, key)], side=side, metric=metric)
                    for label, key in variants
                }
                tests = array_best_vs_comparator_tests(
                    values,
                    higher_is_better=True,
                    table_name="Table 6 desc/consensus ablation",
                    group_label=f"{dataset} {side_label}",
                    metric=metric_label_text,
                )
                if not tests.empty:
                    frames.append(tests)
                marker_tests = array_top_vs_next_tests(
                    values,
                    higher_is_better=True,
                    table_name="Table 6 desc/consensus ablation",
                    group_label=f"{dataset} {side_label}",
                    metric=metric_label_text,
                )
                if not marker_tests.empty:
                    frames.append(marker_tests)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_all_final_tests(
    *,
    results_dir: Path,
    cpdag_summary: pd.DataFrame,
    main_with_gpt_summary: pd.DataFrame,
    dag_summary: pd.DataFrame,
    dag_main_with_gpt_summary: pd.DataFrame,
    gpt_summary: pd.DataFrame,
    gpt_dag_summary: pd.DataFrame,
) -> pd.DataFrame:
    bnlearn_summary = build_bnlearn_cpdag_summary(results_dir)
    bnlearn_dag_summary = build_bnlearn_dag_summary(results_dir)
    bnlearn_metrics = [
        pt.STRUCTURAL_METRICS["NSHD"],
        pt.STRUCTURAL_METRICS["F1"],
        pt.STRUCTURAL_METRICS["NSID-low"],
        pt.STRUCTURAL_METRICS["NSID-high"],
        pt.STRUCTURAL_METRICS["Precision"],
        pt.STRUCTURAL_METRICS["Recall"],
    ]
    bnlearn_dataset_groups = [
        (dataset, lambda df, dataset=dataset: df["dataset"].eq(dataset))
        for dataset in ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
        if bnlearn_summary["dataset"].eq(dataset).any()
    ]
    bnlearn_dag_dataset_groups = [
        (dataset, lambda df, dataset=dataset: df["dataset"].eq(dataset))
        for dataset in ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
        if bnlearn_dag_summary["dataset"].eq(dataset).any()
    ]
    frames = [
        collect_metric_group_tests(
            "Table 1 CauseNet CPDAG main",
            cpdag_summary,
            CPDAG_MAIN_METRICS,
            node_groups(),
            [method for method in pt.SYNTH_METHOD_ORDER if cpdag_summary["model"].eq(method).any()],
        ),
        collect_metric_group_tests(
            "Table 1b matched CPDAG main with LLM sources",
            main_with_gpt_summary,
            CPDAG_MAIN_METRICS,
            node_groups(),
            testable_methods(MAIN_WITH_GPT_METHODS),
            method_labels=LLM_SOURCE_LABELS,
        ),
        collect_metric_group_tests(
            "Table 1c matched DAG main with LLM sources",
            dag_main_with_gpt_summary,
            DAG_GPT_METRICS,
            node_groups(),
            testable_methods(MAIN_WITH_GPT_METHODS),
            method_labels=LLM_SOURCE_LABELS,
        ),
        collect_metric_group_tests(
            "Table 2 CauseNet CPDAG additional",
            cpdag_summary,
            CPDAG_EXTRA_METRICS,
            node_groups(),
            [method for method in pt.SYNTH_METHOD_ORDER if cpdag_summary["model"].eq(method).any()],
        ),
        collect_metric_group_tests(
            "Table 3 CauseNet CPDAG graph type",
            cpdag_summary,
            [*CPDAG_MAIN_METRICS, *CPDAG_EXTRA_METRICS],
            graph_type_groups(),
            [method for method in pt.SYNTH_METHOD_ORDER if cpdag_summary["model"].eq(method).any()],
        ),
        collect_metric_group_tests(
            "Table 4 CauseNet DAG all metrics",
            dag_summary,
            DAG_ALL_METRICS,
            node_groups(),
            [method for method in pt.SYNTH_METHOD_ORDER if dag_summary["model"].eq(method).any()],
        ),
        collect_metric_group_tests(
            "Table 4b CauseNet DAG graph type",
            dag_summary,
            DAG_ALL_METRICS,
            graph_type_groups(),
            [method for method in pt.SYNTH_METHOD_ORDER if dag_summary["model"].eq(method).any()],
        ),
        collect_metric_group_tests(
            "Table 5 bnlearn CPDAG",
            bnlearn_summary,
            bnlearn_metrics,
            bnlearn_dataset_groups,
            BNLEARN_CPDAG_METHODS,
        ),
        collect_metric_group_tests(
            "Table 5b bnlearn DAG",
            bnlearn_dag_summary,
            DAG_ALL_METRICS,
            bnlearn_dag_dataset_groups,
            BNLEARN_DAG_METHODS,
        ),
        collect_desc_consensus_tests(results_dir),
        collect_metric_group_tests(
            "Table 7 Gemini/GPT CPDAG ablation",
            gpt_summary,
            CPDAG_MAIN_METRICS,
            [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
            GPT_ABLATION_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
        collect_metric_group_tests(
            "Table 7b Gemini/GPT DAG ablation",
            gpt_dag_summary,
            DAG_GPT_METRICS,
            [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
            GPT_ABLATION_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    ]
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    tests = pd.concat(nonempty, ignore_index=True)
    tests["marker_printed_on_result_cell"] = tests["marker_printed_on"].map(lambda value: value if value else "no")
    tests["comparator_bolded_as_tied"] = tests["comparator_tied"].map(lambda value: "yes" if value else "no")
    return tests


def display_all_final_tests(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tests.itertuples(index=False):
        provenance = getattr(row, "provenance", "")
        rows.append(
            {
                "Table": row.table,
                "Test role": row.test_role,
                "Group": row.group,
                "Metric": row.metric,
                "Subject": row.best,
                "Comparator": row.comparator,
                "Subject display": pt.fmt_pm(row.best_display_mean, row.best_display_std, ndigits=3),
                "Comparator display": pt.fmt_pm(row.comparator_display_mean, row.comparator_display_std, ndigits=3),
                "Subject test sd/n": f"{pt.fmt_num(row.best_test_std, 3)} / {int(row.best_nobs)}",
                "Comparator test sd/n": f"{pt.fmt_num(row.comparator_test_std, 3)} / {int(row.comparator_nobs)}",
                "t": f"{row.t:.2f}",
                "p_BH": fmt_p(row.p_bh),
                "Sig.": row.marker_bh,
                "Marker printed on": row.marker_printed_on_result_cell,
                "Comparator tied": row.comparator_bolded_as_tied,
                "Provenance": provenance,
            }
        )
    return pd.DataFrame(rows)


def display_compact_final_tests(tests: pd.DataFrame, *, tied: bool = False) -> pd.DataFrame:
    rows = []
    for row in tests.itertuples(index=False):
        provenance = getattr(row, "provenance", "")
        if tied:
            test_label = f"{row.comparator} vs {row.best}"
            subject_label = row.comparator
        else:
            test_label = f"{row.best} vs {row.comparator}"
            subject_label = row.best
        rows.append(
            {
                "Table": row.table,
                "Group": row.group,
                "Metric": row.metric,
                "Subject": subject_label,
                "Test": test_label,
                "Subject value": pt.fmt_pm(row.best_display_mean, row.best_display_std, ndigits=3)
                if not tied
                else pt.fmt_pm(row.comparator_display_mean, row.comparator_display_std, ndigits=3),
                "Comparator value": pt.fmt_pm(row.comparator_display_mean, row.comparator_display_std, ndigits=3)
                if not tied
                else pt.fmt_pm(row.best_display_mean, row.best_display_std, ndigits=3),
                "t": f"{row.t:.2f}",
                "p_BH": fmt_p(row.p_bh),
                "Sig.": row.marker_bh,
                "Provenance": provenance,
            }
        )
    return pd.DataFrame(rows)


def tex_test_group_label(label: str, *, compact: bool = False) -> str:
    if label.startswith("alpha=") and " V=" in label:
        alpha, nodes = label.split(" V=", 1)
        alpha_label = rf"$\alpha={tex_escape(alpha.removeprefix('alpha='))}$"
        node_label = rf"$|\nodeSet|={tex_escape(nodes)}$"
        return rf"\shortstack[l]{{{alpha_label}\\{node_label}}}" if compact else rf"{alpha_label}, {node_label}"
    if label == "Average Average":
        return r"Avg."
    if label.startswith("Average V="):
        node_label = rf"$|\nodeSet|={tex_escape(label.removeprefix('Average V='))}$"
        return rf"\shortstack[l]{{Avg.\\{node_label}}}" if compact else rf"Avg., {node_label}"
    if label.endswith(" Average"):
        graph_type = tex_escape(label.removesuffix(" Average"))
        return rf"\shortstack[l]{{{graph_type}\\Avg.}}" if compact else rf"{graph_type}, Avg."
    if " V=" in label:
        graph_type, nodes = label.split(" V=", 1)
        graph_type = tex_escape(graph_type)
        node_label = rf"$|\nodeSet|={tex_escape(nodes)}$"
        return rf"\shortstack[l]{{{graph_type}\\{node_label}}}" if compact else rf"{graph_type}, {node_label}"
    if label.startswith("V="):
        return rf"$|\nodeSet|={tex_escape(label.removeprefix('V='))}$"
    return tex_escape(label)


def tex_compact_test_name(left: str, right: str) -> str:
    return rf"{tex_escape(left)} vs {tex_escape(right)}"


def tex_result_table_ref(table_name: str) -> str:
    labels = {
        "CauseNet structural": "tab:causenet_dag_full_metrics",
        "CauseNet type-size": "tab:causenet_dag_v_graph_type",
        "Alpha ablation": "tab:alpha_ablation_dag",
        "Matched LLM source": "tab:gpt_model_ablation_dag",
        "bnlearn": "tab:bnlearn_dag_all_metrics",
    }
    label = labels.get(table_name)
    return rf"\ref{{{label}}}" if label else "--"


def final_test_table_order(table_name: str) -> int:
    order = {
        "CauseNet structural": 1,
        "CauseNet type-size": 2,
        "Alpha ablation": 3,
        "bnlearn": 4,
        "Matched LLM source": 8,
    }
    return order.get(table_name, 99)


def final_test_group_order(label: str) -> tuple[int, int, int, str]:
    datasets = ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
    graph_types = ["ER", "SF", "LT", "Average"]
    if label.startswith("alpha=") and " V=" in label:
        alpha, nodes = label.split(" V=", 1)
        alpha_order = {"alpha=0.01": 0, "alpha=0.05": 1}.get(alpha, 99)
        return (2, alpha_order, int(nodes), "")
    if label.startswith("V="):
        return (0, int(label.removeprefix("V=")), 0, "")
    if " V=" in label:
        graph_type, nodes = label.split(" V=", 1)
        return (1, int(nodes), graph_types.index(graph_type) if graph_type in graph_types else 99, "")
    if label.startswith("Average V="):
        return (1, int(label.removeprefix("Average V=")), graph_types.index("Average"), "")
    if label.endswith(" Average"):
        graph_type = label.removesuffix(" Average")
        return (1, 99, graph_types.index(graph_type) if graph_type in graph_types else 99, "")
    if label == "Average Average":
        return (1, 99, graph_types.index("Average"), "")
    if label in datasets:
        return (3, datasets.index(label), 0, "")
    return (4, 0, 0, label)


def final_test_metric_order(metric: str) -> int:
    order = {"SHD": 0, "NSHD": 0, "Precision": 1, "Recall": 2, "F1": 3, "SID": 4, "NSID": 4}
    return order.get(metric, 99)


def latex_final_tests_table(tests: pd.DataFrame, *, tied: bool = False, table_ref_col: bool = False) -> str:
    tests = tests.copy()
    tests["_table_order"] = tests["table"].map(final_test_table_order)
    tests["_group_order"] = tests["group"].map(final_test_group_order)
    tests["_metric_order"] = tests["metric"].map(final_test_metric_order)
    tests["_row_order"] = np.arange(len(tests))
    sort_columns = ["_group_order", "_metric_order", "_row_order"]
    if table_ref_col:
        sort_columns = ["_table_order", *sort_columns]
    tests = tests.sort_values(sort_columns)
    column_spec = r"cll l|c|r|rl|rl" if table_ref_col else r"ll l|c|r|rl|rl"
    header = (
        r"\textbf{Table} & \textbf{Group} & \textbf{Metric} & \textbf{Methods} & \textbf{Means$\pm$Std} & "
        r"\textbf{t} & \multicolumn{2}{c|}{\textbf{p-value}} & \multicolumn{2}{c}{\textbf{$p_{\mathrm{BH}}$}} \\"
        if table_ref_col
        else r"\textbf{Group} & \textbf{Metric} & \textbf{Methods} & \textbf{Means$\pm$Std} & \textbf{t} & "
        r"\multicolumn{2}{c|}{\textbf{p-value}} & \multicolumn{2}{c}{\textbf{$p_{\mathrm{BH}}$}} \\"
    )
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    previous_group = None
    for row in tests.itertuples(index=False):
        if previous_group is not None and row.group != previous_group:
            lines.append(r"\midrule")
        previous_group = row.group
        left = row.best
        right = row.comparator
        means = (
            rf"{tex_pm(row.best_display_mean, row.best_display_std)} vs "
            rf"{tex_pm(row.comparator_display_mean, row.comparator_display_std)}"
        )
        cells = [
            tex_test_group_label(row.group, compact=table_ref_col),
            tex_escape(row.metric),
            tex_compact_test_name(left, right),
            means,
            rf"${row.t:.2f}$",
            tex_p(row.p_value),
            rf"\!\!\!{pt.sig_marker(row.p_value)}",
            tex_p(row.p_bh),
            rf"\!\!\!{row.marker_bh}",
        ]
        if table_ref_col:
            cells.insert(0, tex_result_table_ref(row.table))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_subject_tests_table(tests: pd.DataFrame) -> str:
    tests = tests.copy()
    method_order_map = {method: index for index, method in enumerate(pt.SYNTH_METHOD_ORDER)}
    tests["_group_order"] = tests["group"].map(final_test_group_order)
    tests["_metric_order"] = tests["metric"].map(final_test_metric_order)
    tests["_method_order"] = tests["comparator"].map(lambda method: method_order_map.get(method, 999))
    tests = tests.sort_values(["_group_order", "_metric_order", "_method_order"])
    lines = [
        r"\begin{tabular}{ccl|c|r|rl|rl}",
        r"\toprule",
        r"\textbf{$|\nodeSet|$} & \textbf{Metric} & \textbf{Methods} & \textbf{Means$\pm$Std} & \textbf{t} & \multicolumn{2}{c|}{\textbf{p-value}} & \multicolumn{2}{c}{\textbf{$p_{\mathrm{BH}}$}} \\",
        r"\midrule",
    ]
    groups = [label for label in ["V=5", "V=10", "V=15"] if tests["group"].eq(label).any()]
    metrics = [spec.label for spec in DAG_RAW_METRICS]
    for group_index, group in enumerate(groups):
        sub_group = tests[tests["group"].eq(group)]
        group_span = len(sub_group)
        first_group_row = True
        for metric_index, metric in enumerate(metrics):
            sub_metric = sub_group[sub_group["metric"].eq(metric)]
            if sub_metric.empty:
                continue
            first_metric_row = True
            for row in sub_metric.itertuples(index=False):
                group_cell = (
                    rf"\multirow{{{group_span}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{tex_test_group_label(group)}}}}}}}"
                    if first_group_row
                    else ""
                )
                metric_cell = rf"\multirow{{{len(sub_metric)}}}{{*}}{{\textbf{{{tex_escape(metric)}}}}}" if first_metric_row else ""
                means = (
                    rf"{tex_pm(row.best_display_mean, row.best_display_std)} vs "
                    rf"{tex_pm(row.comparator_display_mean, row.comparator_display_std)}"
                )
                cells = [
                    group_cell,
                    metric_cell,
                    tex_compact_test_name(row.best, row.comparator),
                    means,
                    rf"${row.t:.2f}$",
                    tex_p(row.p_value),
                    rf"\!\!\!{pt.sig_marker(row.p_value)}",
                    tex_p(row.p_bh),
                    rf"\!\!\!{row.marker_bh}",
                ]
                lines.append(" & ".join(cells) + r" \\")
                first_group_row = False
                first_metric_row = False
            if metric_index != len(metrics) - 1:
                lines.append(r"\cmidrule(lr){2-9}")
        if group_index != len(groups) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def build_dag_final_tests(
    *,
    results_dir: Path,
    tag: str,
    dag_summary: pd.DataFrame,
    dag_main_with_gpt_summary: pd.DataFrame,
    gpt_dag_summary: pd.DataFrame,
    alpha05_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    methods = [method for method in pt.SYNTH_METHOD_ORDER if dag_summary["model"].eq(method).any()]
    bnlearn_dag_summary = build_bnlearn_dag_summary(results_dir)
    bnlearn_dag_dataset_groups = [
        (dataset, lambda df, dataset=dataset: df["dataset"].eq(dataset))
        for dataset in ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
        if bnlearn_dag_summary["dataset"].eq(dataset).any()
    ]
    frames = [
        collect_metric_group_tests(
            "CauseNet structural",
            dag_summary,
            DAG_RAW_METRICS,
            node_groups(),
            methods,
        ),
        collect_metric_group_tests(
            "CauseNet type-size",
            dag_summary,
            DAG_MAIN_METRICS,
            v_graph_type_groups(dag_summary),
            methods,
        ),
        collect_metric_group_tests(
            "Matched LLM source",
            dag_main_with_gpt_summary,
            DAG_RAW_METRICS,
            node_groups(),
            testable_methods(MAIN_WITH_GPT_METHODS),
            method_labels=LLM_SOURCE_LABELS,
        ),
        collect_metric_group_tests(
            "bnlearn",
            bnlearn_dag_summary,
            DAG_ALL_METRICS,
            bnlearn_dag_dataset_groups,
            testable_methods(BNLEARN_DAG_METHODS),
        ),
    ]
    if alpha05_summary is not None:
        alpha_methods = [method for method in pt.SYNTH_METHOD_ORDER if dag_summary["model"].eq(method).any()]
        for alpha_label, summary in [("0.01", dag_summary), ("0.05", alpha05_summary)]:
            frames.append(
                collect_metric_group_tests(
                    "Alpha ablation",
                    summary,
                    DAG_MAIN_METRICS,
                    [
                        (
                            f"alpha={alpha_label} V={nodes}",
                            lambda df, nodes=nodes: df["n_nodes"].eq(nodes),
                        )
                        for nodes in [5, 10, 15]
                    ],
                    [method for method in alpha_methods if summary["model"].eq(method).any()],
                )
            )
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    tests = pd.concat(nonempty, ignore_index=True)
    tests["marker_printed_on_result_cell"] = tests["marker_printed_on"].map(lambda value: value if value else "no")
    tests["comparator_bolded_as_tied"] = tests["comparator_tied"].map(lambda value: "yes" if value else "no")
    return add_test_provenance(tests, tag=tag)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tex_method(method: str, method_labels: dict[str, str] | None = None) -> str:
    return tex_escape(method_display(method, method_labels))


def tex_metric_label(spec: pt.MetricSpec) -> str:
    arrow = r"$\uparrow$" if spec.higher_is_better else r"$\downarrow$"
    return rf"{tex_escape(spec.label)} {arrow}"


def tex_metric_label_compact(spec: pt.MetricSpec) -> str:
    arrow = r"$\uparrow$" if spec.higher_is_better else r"$\downarrow$"
    return rf"{tex_escape(spec.label)}{arrow}"


def tex_group_label(label: str) -> str:
    if label.startswith("V="):
        return rf"$|\nodeSet|={tex_escape(label.removeprefix('V='))}$"
    return tex_escape(label)


def tex_marker(marker: str | None) -> str:
    if not marker or marker == "ns":
        return ""
    return rf"\,\textnormal{{{tex_escape(marker)}}}"


def tex_pm(mean: float, std: float, *, bold: bool = False, ndigits: int = 3, marker: str | None = None) -> str:
    if not np.isfinite(mean):
        return "--"
    if np.isfinite(std):
        cell = rf"\num{{{pt.fmt_num(mean, ndigits)} +- {pt.fmt_num(std, ndigits)}}}"
    else:
        cell = rf"\num{{{pt.fmt_num(mean, ndigits)}}}"
    if bold:
        cell = rf"{{\bfseries {cell}}}"
    return cell + tex_marker(marker)


def tex_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value == 0:
        return "$0$"
    if abs(value) < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        exp = int(exponent)
        sign = "-" if exp < 0 else "+"
        return rf"${float(mantissa):.2f}\text{{e{sign}}}{abs(exp)}$"
    return rf"${value:.3f}$"


def tex_table(
    *,
    caption: str,
    label: str,
    body: str,
    table_env: str = "table*",
    placement: str = "t",
    size: str = r"\scriptsize",
    tabcolsep: str = "3pt",
    resize: str | None = r"\textwidth",
    arraystretch: str | None = None,
) -> str:
    lines = [
        rf"\begin{{{table_env}}}[{placement}]",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \centering",
        rf"    {size}",
        rf"    \setlength{{\tabcolsep}}{{{tabcolsep}}}",
    ]
    if arraystretch is not None:
        lines.append(rf"    \renewcommand{{\arraystretch}}{{{arraystretch}}}")
    if resize:
        lines.append(rf"    \resizebox{{{resize}}}{{!}}{{%")
        lines.extend("    " + line for line in body.splitlines())
        lines.append("    }")
    else:
        lines.extend("    " + line for line in body.splitlines())
    lines.append(rf"\end{{{table_env}}}")
    return "\n".join(lines) + "\n"


def latex_metric_by_group_table(
    summary: pd.DataFrame,
    metrics: MetricList,
    groups: list[GroupSpec],
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> str:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    group_labels = [label for label, _ in groups]
    group_masks = {group_label: group_fn(summary) for group_label, group_fn in groups}
    align = "ll" + "c" * len(group_labels)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Method} & "
        + " & ".join(rf"\textbf{{{tex_group_label(label)}}}" for label in group_labels)
        + r" \\",
        r"\midrule",
    ]
    for metric_index, spec in enumerate(metrics):
        group_stats = {
            group_label: group_metric_stats(summary, group_mask, spec, methods)
            for group_label, group_mask in group_masks.items()
        }
        group_bold = {
            group_label: statistically_tied_methods(summary, group_mask, spec, tested_methods)
            for group_label, group_mask in group_masks.items()
        }
        group_marker = {
            group_label: top_cell_markers(summary, group_mask, spec, tested_methods)
            for group_label, group_mask in group_masks.items()
        }
        for method_index, method in enumerate(methods):
            metric_cell = (
                rf"\multirow{{{len(methods)}}}{{*}}{{\textbf{{{tex_metric_label(spec)}}}}}"
                if method_index == 0
                else ""
            )
            cells = [metric_cell, tex_method(method, method_labels)]
            for group_label in group_labels:
                mean, std = group_stats[group_label].get(method, (np.nan, np.nan))
                marker = group_marker[group_label].get(method)
                cells.append(tex_pm(mean, std, bold=method in group_bold[group_label], marker=marker))
            lines.append(" & ".join(cells) + r" \\")
        if metric_index != len(metrics) - 1:
            lines.append(r"\cmidrule(lr){2-" + str(2 + len(group_labels)) + "}")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_causenet_dag_full_table(
    summary: pd.DataFrame,
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> str:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    align = "l|l" + "c" * len(DAG_RAW_METRICS)
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        r"& \textbf{Method} & "
        + " & ".join(rf"\textbf{{{tex_metric_label(spec)}}}" for spec in DAG_RAW_METRICS)
        + r" \\",
        r"\midrule",
    ]
    for node_index, nodes in enumerate([5, 10, 15]):
        group_mask = summary["n_nodes"].eq(nodes)
        group_methods = [method for method in methods if (group_mask & summary["model"].eq(method)).any()]
        tested_methods = testable_methods(group_methods)
        bold_by_metric = {
            spec.label: statistically_tied_methods(summary, group_mask, spec, tested_methods)
            for spec in DAG_RAW_METRICS
        }
        marker_by_metric = {
            spec.label: top_cell_markers(summary, group_mask, spec, tested_methods)
            for spec in DAG_RAW_METRICS
        }
        for method_index, method in enumerate(group_methods):
            node_cell = (
                rf"\multirow{{{len(group_methods)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{$|\nodeSet|={nodes}$}}}}}}"
                if method_index == 0
                else ""
            )
            sub = summary[group_mask & summary["model"].eq(method)]
            cells = [node_cell, tex_method(method, method_labels)]
            for spec in DAG_RAW_METRICS:
                mean = float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean())
                std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
                marker = marker_by_metric[spec.label].get(method)
                cells.append(tex_pm(mean, std, bold=method in bold_by_metric[spec.label], marker=marker))
            lines.append(" & ".join(cells) + r" \\")
        if node_index != 2:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_metric_by_v_graph_type_table(
    summary: pd.DataFrame,
    metrics: MetricList,
    *,
    methods: list[str] | None = None,
) -> str:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    graph_types = [graph_type for graph_type in ["ER", "SF", "LT"] if summary["graph_type_label"].eq(graph_type).any()]
    columns = [*graph_types, "Average"]
    lines = [
        r"\begin{tabular}{@{}ccl" + ("c" * len(columns)) + r"@{}}",
        r"\toprule",
        r" & & \textbf{Method} & " + " & ".join(rf"\textbf{{{tex_escape(graph_type)}}}" for graph_type in columns) + r" \\",
        r"\midrule",
    ]
    present_nodes: list[int | str] = [nodes for nodes in [5, 10, 15] if summary["n_nodes"].eq(nodes).any()]
    present_nodes.append("Average")
    for node_index, nodes in enumerate(present_nodes):
        node_span = len(metrics) * len(methods)
        for metric_index, spec in enumerate(metrics):
            group_masks = {
                graph_type: cross_table_cell_mask(summary, nodes, graph_type)
                for graph_type in columns
            }
            group_stats = {
                graph_type: group_metric_stats(summary, mask, spec, methods)
                for graph_type, mask in group_masks.items()
            }
            group_bold = {
                graph_type: statistically_tied_methods(summary, mask, spec, tested_methods)
                for graph_type, mask in group_masks.items()
            }
            group_marker = {
                graph_type: top_cell_markers(summary, mask, spec, tested_methods)
                for graph_type, mask in group_masks.items()
            }
            for method_index, method in enumerate(methods):
                node_label = r"\textbf{Avg.}" if nodes == "Average" else rf"\textbf{{$|\nodeSet|={nodes}$}}"
                node_cell = (
                    rf"\multirow{{{node_span}}}{{*}}{{\rotatebox{{90}}{{{node_label}}}}}"
                    if metric_index == 0 and method_index == 0
                    else ""
                )
                metric_cell = (
                    rf"\multirow{{{len(methods)}}}{{*}}{{\textbf{{{tex_metric_label(spec)}}}}}"
                    if method_index == 0
                    else ""
                )
                cells = [node_cell, metric_cell, tex_method(method)]
                for graph_type in columns:
                    mean, std = group_stats[graph_type].get(method, (np.nan, np.nan))
                    marker = group_marker[graph_type].get(method)
                    cells.append(tex_pm(mean, std, bold=method in group_bold[graph_type], marker=marker))
                lines.append(" & ".join(cells) + r" \\")
            if metric_index != len(metrics) - 1:
                lines.append(r"\cmidrule(lr){2-" + str(3 + len(columns)) + "}")
        if node_index != len(present_nodes) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_alpha_ablation_table(summary01: pd.DataFrame, summary05: pd.DataFrame) -> str:
    summary05 = carry_display_only_rows(summary01, summary05)
    methods = present_methods(summary01, CAUSENET_DAG_METHODS)
    lines = [
        r"\begin{tabular}{cllccc}",
        r"\toprule",
        r"\textbf{$\alpha$} & \textbf{Metric} & \textbf{Method} & "
        r"\textbf{$|\nodeSet|=5$} & \textbf{$|\nodeSet|=10$} & \textbf{$|\nodeSet|=15$} \\",
        r"\midrule",
    ]
    for alpha_index, (alpha_label, summary) in enumerate([("0.01", summary01), ("0.05", summary05)]):
        methods_present = [method for method in methods if summary["model"].eq(method).any()]
        tested_methods = testable_methods(methods_present)
        alpha_span = len(DAG_MAIN_METRICS) * len(methods_present)
        for metric_index, spec in enumerate(DAG_MAIN_METRICS):
            group_masks = {nodes: summary["n_nodes"].eq(nodes) for nodes in [5, 10, 15]}
            group_stats = {
                nodes: group_metric_stats(summary, mask, spec, methods_present)
                for nodes, mask in group_masks.items()
            }
            group_bold = {
                nodes: statistically_tied_methods(summary, mask, spec, tested_methods)
                for nodes, mask in group_masks.items()
            }
            group_marker = {
                nodes: top_cell_markers(summary, mask, spec, tested_methods)
                for nodes, mask in group_masks.items()
            }
            for method_index, method in enumerate(methods_present):
                alpha_cell = (
                    rf"\multirow{{{alpha_span}}}{{*}}{{\rotatebox{{90}}{{\textbf{{$\alpha={alpha_label}$}}}}}}"
                    if metric_index == 0 and method_index == 0
                    else ""
                )
                metric_cell = (
                    rf"\multirow{{{len(methods_present)}}}{{*}}{{\textbf{{{tex_metric_label(spec)}}}}}"
                    if method_index == 0
                    else ""
                )
                cells = [alpha_cell, metric_cell, tex_method(method)]
                for nodes in [5, 10, 15]:
                    mean, std = group_stats[nodes].get(method, (np.nan, np.nan))
                    marker = group_marker[nodes].get(method)
                    cells.append(tex_pm(mean, std, bold=method in group_bold[nodes], marker=marker))
                lines.append(" & ".join(cells) + r" \\")
            if metric_index != len(DAG_MAIN_METRICS) - 1:
                lines.append(r"\cmidrule(lr){2-6}")
        if alpha_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_compact_metric_grid_table(
    summary: pd.DataFrame,
    metrics: MetricList,
    groups: list[GroupSpec],
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> str:
    methods = methods or present_methods(summary, CAUSENET_DAG_METHODS)
    tested_methods = testable_methods(methods)
    group_labels = [label for label, _ in groups]
    group_masks = {group_label: group_fn(summary) for group_label, group_fn in groups}
    align = "l" + "c" * (len(metrics) * len(group_labels))
    metric_headers = []
    cmidrules = []
    for metric_index, spec in enumerate(metrics):
        start = 2 + metric_index * len(group_labels)
        end = start + len(group_labels) - 1
        metric_headers.append(rf"\multicolumn{{{len(group_labels)}}}{{c}}{{\textbf{{{tex_metric_label(spec)}}}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    group_header = " & ".join(
        rf"\textbf{{{tex_group_label(group_label)}}}"
        for _ in metrics
        for group_label in group_labels
    )
    metric_group_stats = {
        spec.label: {
            group_label: group_metric_stats(summary, group_mask, spec, methods)
            for group_label, group_mask in group_masks.items()
        }
        for spec in metrics
    }
    metric_group_bold = {
        spec.label: {
            group_label: statistically_tied_methods(summary, group_mask, spec, tested_methods)
            for group_label, group_mask in group_masks.items()
        }
        for spec in metrics
    }
    metric_group_marker = {
        spec.label: {
            group_label: top_cell_markers(summary, group_masks[group_label], spec, tested_methods)
            for group_label in group_labels
        }
        for spec in metrics
    }
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(metric_headers) + r" \\",
        " ".join(cmidrules),
        r" & " + group_header + r" \\",
        r"\midrule",
    ]
    for method in methods:
        cells = [tex_method(method, method_labels)]
        for spec in metrics:
            for group_label in group_labels:
                mean, std = metric_group_stats[spec.label][group_label].get(method, (np.nan, np.nan))
                marker = metric_group_marker[spec.label][group_label].get(method)
                cells.append(
                    tex_pm(mean, std, bold=method in metric_group_bold[spec.label][group_label], marker=marker)
                )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_split_metric_grid_table(
    summary: pd.DataFrame,
    metric_chunks: list[MetricList],
    groups: list[GroupSpec],
    *,
    methods: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
) -> str:
    panels = []
    for metrics in metric_chunks:
        panels.append(
            "\n".join(
                [
                    r"\resizebox{\textwidth}{!}{%",
                    latex_compact_metric_grid_table(
                        summary,
                        metrics,
                        groups,
                        methods=methods,
                        method_labels=method_labels,
                    ),
                    r"}",
                ]
            )
        )
    return ("\n\n" + r"\vspace{0.45em}" + "\n").join(panels)


def latex_desc_consensus_table(results_dir: Path, *, exclude_zero_constraints: bool = False) -> str:
    sources = {
        ("bnlearn", "Average", "w/o desc"): results_dir / "llm_constraints" / "bnlearn.json",
        ("bnlearn", "Average", "with desc"): results_dir / "llm_constraints" / "bnlearn-desc.json",
        ("bnlearn", "Consensus", "w/o desc"): results_dir / "llm_constraints" / "bnlearn-consensus.json",
        ("bnlearn", "Consensus", "with desc"): results_dir / "llm_constraints" / "bnlearn-desc-consensus.json",
        ("CauseNet", "Average", "w/o desc"): results_dir / "llm_constraints" / "synthetic.json",
        ("CauseNet", "Average", "with desc"): results_dir / "llm_constraints" / "synthetic-desc.json",
        ("CauseNet", "Consensus", "w/o desc"): results_dir / "llm_constraints" / "synthetic-consensus.json",
        ("CauseNet", "Consensus", "with desc"): results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
    }
    row_defs = [
        ("Forbidden constraints", "forbidden", "length", r"\# constraints", False),
        ("Forbidden constraints", "forbidden", "Precision", "Precision", True),
        ("Forbidden constraints", "forbidden", "Recall", "Recall", True),
        ("Forbidden constraints", "forbidden", "F1", "F1", True),
        ("Required constraints", "required", "length", r"\# constraints", False),
        ("Required constraints", "required", "Precision", "Precision", True),
        ("Required constraints", "required", "Recall", "Recall", True),
        ("Required constraints", "required", "F1", "F1", True),
    ]
    col_defs = list(sources.keys())

    def desc_cell(mean: float, std: float, *, bold: bool, best: bool, arrow: str | None, ndigits: int) -> str:
        if not np.isfinite(mean):
            return "--"
        if np.isfinite(std):
            cell = rf"\num{{{pt.fmt_num(mean, ndigits)} +- {pt.fmt_num(std, ndigits)}}}"
        else:
            cell = rf"\num{{{pt.fmt_num(mean, ndigits)}}}"
        if best:
            cell = rf"\best{{{cell}}}"
        elif bold:
            cell = rf"{{\bfseries {cell}}}"
        if arrow == "up":
            cell += r"\,\up"
        elif arrow == "down":
            cell += r"\,\down"
        return cell

    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{bnlearn}} & \multicolumn{4}{c}{\textbf{CauseNet}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"\textbf{Metric} & \multicolumn{2}{c}{\textbf{Average}} & \multicolumn{2}{c}{\textbf{Consensus}} & \multicolumn{2}{c}{\textbf{Average}} & \multicolumn{2}{c}{\textbf{Consensus}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & \textbf{w/o desc} & \textbf{with desc} & \textbf{w/o desc} & \textbf{with desc} & \textbf{w/o desc} & \textbf{with desc} & \textbf{w/o desc} & \textbf{with desc} \\",
        r"\midrule",
    ]
    last_section = None
    for section, side, metric, label, bold_best in row_defs:
        if section != last_section:
            if last_section is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{9}}{{c}}{{\textbf{{{section}}}}} \\")
            lines.append(r"\midrule")
            last_section = section
        values = []
        value_arrays = {}
        for key in col_defs:
            vals = constraint_source_values(
                sources[key],
                side=side,
                metric=metric,
                exclude_zero_constraints=exclude_zero_constraints,
            )
            value_arrays[key] = vals
            values.append((float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))))
        means_by_key = {key: mean for key, (mean, _) in zip(col_defs, values)}
        best_key = max(means_by_key, key=means_by_key.get) if bold_best else None
        bold_keys = set()
        arrows: dict[tuple[str, str, str], str] = {}
        if bold_best:
            for dataset in ["bnlearn", "CauseNet"]:
                for desc in ["w/o desc", "with desc"]:
                    average_key = (dataset, "Average", desc)
                    consensus_key = (dataset, "Consensus", desc)
                    if means_by_key[average_key] >= means_by_key[consensus_key]:
                        bold_keys.add(average_key)
                    else:
                        bold_keys.add(consensus_key)
                for source_kind in ["Average", "Consensus"]:
                    no_desc_key = (dataset, source_kind, "w/o desc")
                    with_desc_key = (dataset, source_kind, "with desc")
                    arrows[with_desc_key] = "up" if means_by_key[with_desc_key] > means_by_key[no_desc_key] else "down"
        cells = [label]
        for key, (mean, std) in zip(col_defs, values):
            cells.append(
                desc_cell(
                    mean,
                    std,
                    bold=key in bold_keys,
                    best=key == best_key,
                    arrow=arrows.get(key),
                    ndigits=2 if metric == "length" else 3,
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_bnlearn_metric_table(summary: pd.DataFrame, metrics: MetricList, methods_order: list[str]) -> str:
    dataset_order = ["asia", "cancer", "earthquake", "survey", "sachs", "child"]
    tested_methods = testable_methods(methods_order)
    lines = [
        r"\begin{tabular}{ll" + ("c" * len(metrics)) + r"}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Method} & "
        + " & ".join(rf"\textbf{{{tex_metric_label(spec)}}}" for spec in metrics)
        + r" \\",
        r"\midrule",
    ]
    present_datasets = [d for d in dataset_order if summary["dataset"].eq(d).any()]
    for dataset_index, dataset in enumerate(present_datasets):
        sub_dataset = summary[summary["dataset"].eq(dataset)]
        dataset_mask = summary["dataset"].eq(dataset)
        bold_by_metric = {}
        marker_by_metric = {}
        for spec in metrics:
            bold_by_metric[spec.label] = statistically_tied_methods(
                summary,
                dataset_mask,
                spec,
                tested_methods,
            )
            marker_by_metric[spec.label] = top_cell_markers(summary, dataset_mask, spec, tested_methods)
        methods = [m for m in methods_order if sub_dataset["model"].eq(m).any()]
        for method_index, method in enumerate(methods):
            dataset_cell = (
                rf"\multirow{{{len(methods)}}}{{*}}{{{bnlearn_dataset_label(summary, dataset, latex=True)}}}"
                if method_index == 0
                else ""
            )
            sub = sub_dataset[sub_dataset["model"].eq(method)]
            cells = [dataset_cell, tex_method(method)]
            for spec in metrics:
                mean = float(pd.to_numeric(sub[spec.mean_col], errors="coerce").mean())
                std = float(pd.to_numeric(sub[spec.std_col], errors="coerce").mean())
                marker = marker_by_metric[spec.label].get(method)
                cells.append(tex_pm(mean, std, bold=method in bold_by_metric[spec.label], marker=marker))
            lines.append(" & ".join(cells) + r" \\")
        if dataset_index != len(present_datasets) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_bnlearn_cpdag_table(summary: pd.DataFrame) -> str:
    metrics = [
        pt.STRUCTURAL_METRICS["NSHD"],
        pt.STRUCTURAL_METRICS["F1"],
        pt.STRUCTURAL_METRICS["NSID-low"],
        pt.STRUCTURAL_METRICS["NSID-high"],
        pt.STRUCTURAL_METRICS["Precision"],
        pt.STRUCTURAL_METRICS["Recall"],
    ]
    return latex_bnlearn_metric_table(summary, metrics, BNLEARN_CPDAG_METHODS)


def latex_bnlearn_dag_table(summary: pd.DataFrame) -> str:
    return latex_bnlearn_metric_table(summary, DAG_ALL_METRICS, BNLEARN_DAG_METHODS)


def latex_runtime_table(table: pd.DataFrame) -> str:
    def tex_from_pm(cell: str) -> str:
        if "+/-" not in str(cell):
            return r"\textemdash"
        mean, std = [float(x.strip()) for x in cell.split("+/-")]
        return tex_pm(mean, std, ndigits=2)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{$|\nodeSet|=5$} & \textbf{$|\nodeSet|=10$} & \textbf{$|\nodeSet|=15$} \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{tex_method(row['Method'])} & {tex_from_pm(row['V=5'])} & "
            f"{tex_from_pm(row['V=10'])} & {tex_from_pm(row['V=15'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def latex_tests_table(tables_dir: Path, tag: str, graph_kind: str = "cpdag") -> str:
    tests = pd.read_csv(tables_dir / f"{tests_stem(tag, graph_kind)}.csv")
    methods_order = ["Random", "FGS", "NOTEARS-MLP", "GRaSP", "BOSS", "MPC", "MPC-LLM", "ABAPC"]
    tests["other"] = tests["comparison"].str.replace("ABAPC-LLM vs ", "", regex=False)
    tests["method_order"] = tests["other"].map({m: i for i, m in enumerate(methods_order)})
    tests["metric_order"] = tests["metric"].map({"F1": 0, "SHD": 1, "NSHD": 1})
    tests = tests.sort_values(["nodes", "metric_order", "method_order"])
    lines = [
        r"\begin{tabular}{ccl|c|r|rl|rl}",
        r"\toprule",
        r"\textbf{$|\nodeSet|$} & \textbf{Metric} & \textbf{Methods} & \textbf{Means$\pm$Std} & \textbf{t} & \multicolumn{2}{c|}{\textbf{p-value}} & \multicolumn{2}{c}{\textbf{$p_{\mathrm{BH}}$}} \\",
        r"\midrule",
    ]
    for node_index, nodes in enumerate([5, 10, 15]):
        sub_nodes = tests[tests["nodes"].eq(nodes)]
        node_span = len(sub_nodes)
        first_node_row = True
        for metric_index, metric in enumerate(["F1", "SHD"]):
            sub_metric = sub_nodes[sub_nodes["metric"].eq(metric)]
            first_metric_row = True
            for row in sub_metric.itertuples(index=False):
                node_cell = (
                    rf"\multirow{{{node_span}}}{{*}}{{\rotatebox{{90}}{{\textbf{{$|\nodeSet|={nodes}$}}}}}}"
                    if first_node_row
                    else ""
                )
                metric_cell = rf"\multirow{{{len(sub_metric)}}}{{*}}{{\textbf{{{metric}}}}}" if first_metric_row else ""
                means = rf"{tex_pm(row.ref_mean, row.ref_std)} vs {tex_pm(row.other_mean, row.other_std)}"
                cells = [
                    node_cell,
                    metric_cell,
                    tex_escape(row.comparison),
                    means,
                    rf"${row.t:.2f}$",
                    tex_p(row.p_value),
                    rf"\!\!\!{row.marker}",
                    tex_p(row.p_bh),
                    rf"\!\!\!{row.marker_bh}",
                ]
                lines.append(" & ".join(cells) + r" \\")
                first_node_row = False
                first_metric_row = False
            if metric_index == 0:
                lines.append(r"\cmidrule(lr){2-9}")
        if node_index != 2:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_tex(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_final_tex_tables(
    *,
    final_dir: Path,
    results_dir: Path,
    tables_dir: Path,
    tag: str,
    cpdag_summary: pd.DataFrame,
    main_with_gpt_summary: pd.DataFrame,
    dag_summary: pd.DataFrame,
    gpt_summary: pd.DataFrame,
    gpt_dag_summary: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> None:
    bold_test_note = RESULT_SIG_NOTE
    write_tex(
        final_dir / "table_01_causenet_cpdag_main.tex",
        tex_table(
            caption=(
                r"Main CPDAG structural performance on \texttt{CauseNet} synthetic datasets. "
                r"Values are mean$\pm$std over datasets/repetitions, grouped by $|\nodeSet|$. "
                + bold_test_note
            ),
            label="tab:causenet_cpdag_main",
            body=latex_metric_by_group_table(cpdag_summary, CPDAG_MAIN_METRICS, node_groups()),
            table_env="table*",
            resize=r"\textwidth",
        ),
    )
    write_tex(
        final_dir / "table_01b_causenet_cpdag_main_llm_sources.tex",
        tex_table(
            caption=(
                rf"Matched main CPDAG structural performance on the {main_with_gpt_summary['dataset'].nunique()} "
                r"\texttt{CauseNet} synthetic datasets where both Gemini and GPT consensus runs are available. "
                r"LLM-constraint variants are labelled by source. "
                + bold_test_note
            ),
            label="tab:causenet_cpdag_main_llm_sources",
            body=latex_metric_by_group_table(
                main_with_gpt_summary,
                CPDAG_MAIN_METRICS,
                node_groups(),
                methods=MAIN_WITH_GPT_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="3pt",
        ),
    )
    dag_main_with_gpt_summary = build_dag_main_with_gpt_summary(dag_summary, gpt_dag_summary)
    write_tex(
        final_dir / "table_01c_causenet_dag_main_llm_sources.tex",
        tex_table(
            caption=(
                rf"Matched main structural performance on the {dag_main_with_gpt_summary['dataset'].nunique()} "
                r"\texttt{CauseNet} synthetic datasets where both Gemini and GPT consensus runs are available. "
                r"LLM-constraint variants are labelled by source. "
                + bold_test_note
            ),
            label="tab:causenet_dag_main_llm_sources",
            body=latex_metric_by_group_table(
                dag_main_with_gpt_summary,
                DAG_GPT_METRICS,
                node_groups(),
                methods=MAIN_WITH_GPT_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="3pt",
        ),
    )
    write_tex(
        final_dir / "table_02_causenet_cpdag_additional_metrics.tex",
        tex_table(
            caption=(
                r"Additional CPDAG metrics on \texttt{CauseNet} synthetic datasets, grouped by $|\nodeSet|$. "
                + bold_test_note
            ),
            label="tab:causenet_cpdag_additional",
            body=latex_metric_by_group_table(cpdag_summary, CPDAG_EXTRA_METRICS, node_groups()),
            table_env="table*",
            resize=r"\textwidth",
        ),
    )
    write_tex(
        final_dir / "table_03_causenet_cpdag_graph_type.tex",
        tex_table(
            caption=(
                r"CPDAG metrics on \texttt{CauseNet} synthetic datasets by graph type. "
                + bold_test_note
            ),
            label="tab:causenet_cpdag_graph_type",
            body=latex_split_metric_grid_table(
                cpdag_summary,
                [
                    [
                        pt.STRUCTURAL_METRICS["NSHD"],
                        pt.STRUCTURAL_METRICS["F1"],
                        pt.STRUCTURAL_METRICS["NSID-low"],
                    ],
                    [
                        pt.STRUCTURAL_METRICS["NSID-high"],
                        pt.STRUCTURAL_METRICS["Precision"],
                        pt.STRUCTURAL_METRICS["Recall"],
                    ],
                ],
                graph_type_groups(),
            ),
            table_env="table*",
            resize=None,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "table_04_causenet_dag_all_metrics.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{CauseNet} synthetic datasets, grouped by $|\nodeSet|$. "
                + bold_test_note
            ),
            label="tab:causenet_dag_all_metrics",
            body=latex_split_metric_grid_table(
                dag_summary,
                [
                    [
                        pt.STRUCTURAL_METRICS["NSHD"],
                        pt.STRUCTURAL_METRICS["F1"],
                        pt.STRUCTURAL_METRICS["NSID"],
                    ],
                    [
                        pt.STRUCTURAL_METRICS["Precision"],
                        pt.STRUCTURAL_METRICS["Recall"],
                    ],
                ],
                node_groups(),
            ),
            table_env="table*",
            resize=None,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "table_04b_causenet_dag_graph_type.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{CauseNet} synthetic datasets by graph type. "
                + bold_test_note
            ),
            label="tab:causenet_dag_graph_type",
            body=latex_split_metric_grid_table(
                dag_summary,
                [
                    [
                        pt.STRUCTURAL_METRICS["NSHD"],
                        pt.STRUCTURAL_METRICS["F1"],
                        pt.STRUCTURAL_METRICS["NSID"],
                    ],
                    [
                        pt.STRUCTURAL_METRICS["Precision"],
                        pt.STRUCTURAL_METRICS["Recall"],
                    ],
                ],
                graph_type_groups(),
            ),
            table_env="table*",
            resize=None,
            size=r"\scriptsize",
            tabcolsep="2pt",
        ),
    )
    bnlearn_summary = build_bnlearn_cpdag_summary(results_dir)
    write_tex(
        final_dir / "table_05_bnlearn_cpdag_all_metrics.tex",
        tex_table(
            caption=(
                r"CPDAG metrics on \texttt{bnlearn} benchmarks by dataset. "
                r"Bold marks the best mean and methods not significantly different from it under "
                r"Welch tests with Benjamini--Hochberg correction within each dataset/metric. "
                r"Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking "
                r"(\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, \textnormal{*} $p<0.05$, "
                r"\textnormal{.} $p<0.1$). "
                r"Only methods with saved CPDAG metrics are shown."
            ),
            label="tab:bnlearn_cpdag_all_metrics",
            body=latex_bnlearn_cpdag_table(bnlearn_summary),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2.2pt",
        ),
    )
    bnlearn_dag_summary = build_bnlearn_dag_summary(results_dir)
    write_tex(
        final_dir / "table_05b_bnlearn_dag_all_metrics.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{bnlearn} benchmarks by dataset. "
                r"Bold marks the best mean and methods not significantly different from it under "
                r"Welch tests with Benjamini--Hochberg correction within each dataset/metric. "
                r"Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking "
                r"(\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, \textnormal{*} $p<0.05$, "
                r"\textnormal{.} $p<0.1$)."
            ),
            label="tab:bnlearn_dag_all_metrics",
            body=latex_bnlearn_dag_table(bnlearn_dag_summary),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "table_06_desc_consensus_ablation.tex",
        tex_table(
            caption=(
                r"LLM-derived constraint quality under different elicitation strategies. "
                r"Bold marks the best mean and variants not significantly different from it under "
                r"Welch tests with Benjamini--Hochberg correction across strategies in each row; "
                r"counts are not bolded. Markers on bolded top cells denote corrected tests against "
                r"the next non-bold variant in the ranking (\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, "
                r"\textnormal{*} $p<0.05$, \textnormal{.} $p<0.1$)."
            ),
            label="tab:desc_consensus_ablation",
            body=latex_desc_consensus_table(results_dir),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="3.5pt",
        ),
    )
    write_tex(
        final_dir / "table_07_gpt_model_ablation.tex",
        tex_table(
            caption=(
                rf"Matched Gemini/GPT-5-mini CPDAG comparison on the {gpt_summary['dataset'].nunique()} "
                r"\texttt{CauseNet} synthetic datasets where both consensus runs are available. "
                + bold_test_note
            ),
            label="tab:gpt_model_ablation_cpdag",
            body=latex_metric_by_group_table(
                gpt_summary,
                CPDAG_MAIN_METRICS,
                [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
                methods=GPT_ABLATION_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"\textwidth",
        ),
    )
    write_tex(
        final_dir / "table_07b_gpt_model_ablation_dag.tex",
        tex_table(
            caption=(
                rf"Matched Gemini/GPT-5-mini structural comparison on the {gpt_dag_summary['dataset'].nunique()} "
                r"\texttt{CauseNet} synthetic datasets where both consensus runs are available. "
                + bold_test_note
            ),
            label="tab:gpt_model_ablation_dag",
            body=latex_metric_by_group_table(
                gpt_dag_summary,
                DAG_GPT_METRICS,
                [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
                methods=GPT_ABLATION_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"\textwidth",
        ),
    )
    write_tex(
        final_dir / "table_08_runtime_by_node_count.tex",
        tex_table(
            caption=(
                r"Average runtime in seconds on synthetic \texttt{CauseNet} datasets "
                r"(mean$\pm$std over repetitions). LLM API latency is excluded."
            ),
            label="tab:runtime_scaling_final",
            body=latex_runtime_table(runtime_table),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="6pt",
        ),
    )
    write_tex(
        final_dir / "table_09_cpdag_main_tests.tex",
        tex_table(
            caption=(
                r"Two-sample, unequal variance $t$-tests comparing ABAPC-LLM against other methods "
                r"on the main CPDAG metrics for \texttt{CauseNet} synthetic datasets. "
                r"BH-corrected $p$-values account for multiple comparisons."
                + TEST_SIG_NOTE
            ),
            label="tab:cpdag_main_tests",
            body=latex_tests_table(tables_dir, tag),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "legacy_table_09b_dag_main_tests.tex",
        tex_table(
            caption=(
                r"Two-sample, unequal variance $t$-tests comparing ABAPC-LLM against other methods "
                r"on the main structural metrics for \texttt{CauseNet} synthetic datasets. "
                r"BH-corrected $p$-values account for multiple comparisons."
                + TEST_SIG_NOTE
            ),
            label="tab:dag_main_tests",
            body=latex_tests_table(tables_dir, tag, graph_kind="dag"),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2pt",
        ),
    )


def write_dag_final_tex_tables(
    *,
    final_dir: Path,
    results_dir: Path,
    tables_dir: Path,
    tag: str,
    dag_summary: pd.DataFrame,
    gpt_dag_summary: pd.DataFrame,
    runtime_table: pd.DataFrame,
    alpha05_summary: pd.DataFrame | None = None,
    all_tests: pd.DataFrame | None = None,
    abapcllm_main_tests: pd.DataFrame | None = None,
) -> None:
    bold_test_note = RESULT_SIG_NOTE
    dag_main_with_gpt_summary = build_dag_main_with_gpt_summary(dag_summary, gpt_dag_summary)
    methods = present_methods(dag_summary, CAUSENET_DAG_METHODS)
    if all_tests is None:
        all_tests = build_dag_final_tests(
            results_dir=results_dir,
            tag=tag,
            dag_summary=dag_summary,
            dag_main_with_gpt_summary=dag_main_with_gpt_summary,
            gpt_dag_summary=gpt_dag_summary,
            alpha05_summary=alpha05_summary,
        )
    if abapcllm_main_tests is None:
        abapcllm_main_tests = build_causenet_main_abapcllm_tests(dag_summary, tag=tag)

    write_tex(
        final_dir / "table_01_causenet_dag_full_metrics.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{CauseNet} synthetic datasets, grouped by node count $|\nodeSet|$. "
                r"Among repeated-run methods, bold marks the best mean and statistically tied methods under BH-corrected Welch tests within each "
                r"metric/column (Tabs.~\ref{tab:causenet_main_indicator_tests}-\ref{tab:structural_tied_bolding_tests}, "
                r"App.~\ref{sec:appendix_statistical_tests}). Significance levels for the $p$-values against the next "
                r"non-bold method in the ranking are: \textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, "
                r"\textnormal{*} $p<0.05$."
                + LLM_BFS_NOTE
            ),
            label="tab:causenet_dag_full_metrics",
            body=latex_causenet_dag_full_table(dag_summary),
            table_env="table*",
            resize=r"0.83\textwidth",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "table_02_causenet_dag_v_graph_type.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{CauseNet} by graph type and $|\nodeSet|$, with marginal averages. "
                + bold_test_note
                + LLM_BFS_NOTE
                + r" Tests: Table~\ref{tab:causenet_type_size_indicator_tests}; "
                + r"tied bolding: Table~\ref{tab:structural_tied_bolding_tests}."
            ),
            label="tab:causenet_dag_v_graph_type",
            body=latex_metric_by_v_graph_type_table(dag_summary, DAG_MAIN_METRICS, methods=methods),
            table_env="table*",
            resize=None,
            size=r"\scriptsize",
            tabcolsep="2pt",
            arraystretch="0.68",
        ),
    )
    write_tex(
        final_dir / "table_06_desc_consensus_ablation.tex",
        tex_table(
            caption=(
                r"LLM-derived constraint quality by dataset, consensus strategy, and description use. "
                r"Bold compares Average vs. Consensus within each setting; green marks the row best; arrows show the description effect."
            ),
            label="tab:desc_consensus_ablation",
            body=latex_desc_consensus_table(results_dir),
            table_env="table*",
            placement="h!b",
            resize=r"1\textwidth",
            tabcolsep="3.5pt",
        ),
    )
    write_tex(
        final_dir / "table_07_desc_consensus_nonzero_ablation.tex",
        tex_table(
            caption=(
                r"LLM-derived constraint quality excluding zero-constraint cases. "
                r"Formatting follows Table~\ref{tab:desc_consensus_ablation}: bold compares Average vs. Consensus within each setting; "
                r"green marks the row best; arrows show the description effect."
            ),
            label="tab:desc_consensus_nonzero_ablation",
            body=latex_desc_consensus_table(results_dir, exclude_zero_constraints=True),
            table_env="table*",
            resize=r"1\textwidth",
            tabcolsep="3.5pt",
        ),
    )
    write_tex(
        final_dir / "table_08_gpt_model_ablation_dag.tex",
        tex_table(
            caption=(
                rf"Structural metrics on the {dag_main_with_gpt_summary['dataset'].nunique()} matched \texttt{{CauseNet}} datasets "
                r"with Gemini and GPT-5-mini constraints, grouped by $|\nodeSet|$. LLM-constraint variants are labelled by source. "
                + bold_test_note
                + LLM_BFS_NOTE
                + r" Tests: Table~\ref{tab:matched_llm_indicator_tests}; "
                + r"tied bolding: Table~\ref{tab:structural_tied_bolding_tests}."
            ),
            label="tab:gpt_model_ablation_dag",
            body=latex_causenet_dag_full_table(
                dag_main_with_gpt_summary,
                methods=MAIN_WITH_GPT_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"0.92\textwidth",
            tabcolsep="1.8pt",
        ),
    )
    if alpha05_summary is not None:
        write_tex(
            final_dir / "table_03_alpha_ablation_dag.tex",
            tex_table(
                caption=(
                    r"Significance threshold ablation on \texttt{CauseNet}. "
                    r"The main runs use $\alpha=0.01$; the comparison uses $\alpha=0.05$ runs. "
                    + bold_test_note
                    + LLM_BFS_NOTE
                    + r" Tests: Table~\ref{tab:alpha_indicator_tests}; "
                    + r"tied bolding: Table~\ref{tab:structural_tied_bolding_tests}."
                ),
                label="tab:alpha_ablation_dag",
                body=latex_alpha_ablation_table(dag_summary, alpha05_summary),
                table_env="table*",
                resize=r"0.8\textwidth",
                tabcolsep="2pt",
            ),
        )
    bnlearn_dag_summary = build_bnlearn_dag_summary(results_dir)
    write_tex(
        final_dir / "table_04_bnlearn_dag_all_metrics.tex",
        tex_table(
            caption=(
                r"Structural metrics on \texttt{bnlearn} benchmarks; dataset labels report $|\nodeSet|$ and $|\edgSet|$. "
                + bold_test_note
                + LLM_BFS_NOTE
                + r" Tests: Tables~\ref{tab:bnlearn_earthquake_indicator_tests}, "
                + r"\ref{tab:bnlearn_other_indicator_tests}, and \ref{tab:bnlearn_child_indicator_tests}; "
                + r"tied bolding: Table~\ref{tab:bnlearn_tied_bolding_tests}."
            ),
            label="tab:bnlearn_dag_all_metrics",
            body=latex_bnlearn_dag_table(bnlearn_dag_summary),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2pt",
        ),
    )
    write_tex(
        final_dir / "table_05_runtime_by_node_count.tex",
        tex_table(
            caption=(
                r"Runtime in seconds on \texttt{CauseNet} synthetic datasets (mean$\pm$std; LLM API latency excluded)."
            ),
            label="tab:runtime_scaling_final",
            body=latex_runtime_table(runtime_table),
            table_env="table*",
            resize=None,
            size=r"\normalsize",
            tabcolsep="8pt",
        ),
    )
    if all_tests is not None and not all_tests.empty:
        def write_compact_test_tex(
            filename: str,
            *,
            caption: str,
            label: str,
            rows: pd.DataFrame,
            tied: bool = False,
            table_ref_col: bool = False,
        ) -> None:
            if rows.empty:
                return
            write_tex(
                final_dir / filename,
                tex_table(
                    caption=caption + TEST_SIG_NOTE,
                    label=label,
                    body=latex_final_tests_table(rows, tied=tied, table_ref_col=table_ref_col),
                    table_env="table*",
                    resize=r"\textwidth",
                    size=r"\scriptsize",
                    tabcolsep="2pt",
                ),
            )

        marker_tests = all_tests[all_tests["test_role"].eq("marker_top_vs_next_nonbold")].copy()
        tied_tests = all_tests[
            all_tests["test_role"].eq("bolding_best_vs_comparator") & all_tests["comparator_tied"].astype(bool)
        ].copy()
        bnlearn_markers = marker_tests[marker_tests["table"].eq("bnlearn")].copy()
        write_compact_test_tex(
            "table_09_causenet_main_indicator_tests.tex",
            caption=(
                r"Welch tests for Table~\ref{tab:causenet_dag_full_metrics}. "
                r"Each row compares a bolded top entry with the next non-bold method in that column."
            ),
            label="tab:causenet_main_indicator_tests",
            rows=marker_tests[marker_tests["table"].eq("CauseNet structural")],
        )
        write_compact_test_tex(
            "table_11_causenet_type_size_indicator_tests.tex",
            caption=(
                r"Welch tests for Table~\ref{tab:causenet_dag_v_graph_type}, including marginal averages."
            ),
            label="tab:causenet_type_size_indicator_tests",
            rows=marker_tests[marker_tests["table"].eq("CauseNet type-size")],
        )
        write_compact_test_tex(
            "table_17_matched_llm_indicator_tests.tex",
            caption=(
                r"Welch tests for Table~\ref{tab:gpt_model_ablation_dag}."
            ),
            label="tab:matched_llm_indicator_tests",
            rows=marker_tests[marker_tests["table"].eq("Matched LLM source")],
        )
        write_compact_test_tex(
            "table_12_alpha_indicator_tests.tex",
            caption=(
                r"Welch tests for Table~\ref{tab:alpha_ablation_dag}."
            ),
            label="tab:alpha_indicator_tests",
            rows=marker_tests[marker_tests["table"].eq("Alpha ablation")],
        )
        write_compact_test_tex(
            "table_13_bnlearn_earthquake_indicator_tests.tex",
            caption=(
                r"Welch tests for the \texttt{earthquake} rows in Table~\ref{tab:bnlearn_dag_all_metrics}."
            ),
            label="tab:bnlearn_earthquake_indicator_tests",
            rows=bnlearn_markers[bnlearn_markers["group"].eq("earthquake")],
        )
        write_compact_test_tex(
            "table_14_bnlearn_other_indicator_tests.tex",
            caption=(
                r"Welch tests for Table~\ref{tab:bnlearn_dag_all_metrics}, excluding the \texttt{earthquake} and \texttt{child} rows."
            ),
            label="tab:bnlearn_other_indicator_tests",
            rows=bnlearn_markers[~bnlearn_markers["group"].isin(["earthquake", "child"])],
        )
        write_compact_test_tex(
            "table_15_bnlearn_child_indicator_tests.tex",
            caption=(
                r"Welch tests for the \texttt{child} rows in Table~\ref{tab:bnlearn_dag_all_metrics}."
            ),
            label="tab:bnlearn_child_indicator_tests",
            rows=bnlearn_markers[bnlearn_markers["group"].eq("child")],
        )
        write_compact_test_tex(
            "table_10_structural_tied_bolding_tests.tex",
            caption=(
                r"Non-significant best-vs-comparator tests that justify additional bolded methods in Tables~\ref{tab:causenet_dag_full_metrics}, \ref{tab:causenet_dag_v_graph_type}, \ref{tab:alpha_ablation_dag}, and \ref{tab:gpt_model_ablation_dag}."
            ),
            label="tab:structural_tied_bolding_tests",
            rows=tied_tests[~tied_tests["table"].eq("bnlearn")],
            tied=True,
            table_ref_col=True,
        )
        write_compact_test_tex(
            "table_16_bnlearn_tied_bolding_tests.tex",
            caption=(
                r"Non-significant best-vs-comparator tests that justify additional bolded methods in Table~\ref{tab:bnlearn_dag_all_metrics}."
            ),
            label="tab:bnlearn_tied_bolding_tests",
            rows=tied_tests[tied_tests["table"].eq("bnlearn")],
            tied=True,
        )


def write_final_tables(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    ensure_intermediate_structural_tables(args)
    final_dir = tables_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    check_dir = final_dir / "check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)

    dag_summary = load_structural_summary(tables_dir, f"synthetic_structural_summary_{args.tag}")
    dag_summary = append_llm_bfs_dag_rows(dag_summary, results_dir / "causal-bfs-synthetic-results.csv")
    alpha05_summary = load_optional_alpha05_summary(tables_dir)
    gpt_dag_summary = build_gpt_ablation_dag_summary_from_base(results_dir, args.tag, dag_summary)
    dag_main_with_gpt_summary = build_dag_main_with_gpt_summary(dag_summary, gpt_dag_summary)
    runtime_table = build_runtime_table(dag_summary)

    standard_note = (
        "Key: among repeated-run methods, **bold** marks the best mean and statistically tied methods under BH-corrected Welch tests "
        "within each metric/column. Significance levels for the p-values against the next non-bold method in the ranking are: "
        "*** p<0.001, ** p<0.01, * p<0.05, . p<0.1. Values are mean +/- std."
    )
    llm_bfs_md_note = " LLM-BFS is shown from one saved run and is excluded from Welch tests and bolding."
    md_test_sig_note = (
        " Significance levels for both reported p-values are: "
        "*** p<0.001, ** p<0.01, * p<0.05, . p<0.1."
    )

    write_table(
        final_dir / "table_01_causenet_dag_full_metrics.md",
        "Table 1. CauseNet structural metrics by node count",
        standard_note + llm_bfs_md_note + " Metrics are ordered as SHD, precision, recall, and F1.",
        build_causenet_dag_full_table(dag_summary),
    )
    write_table(
        final_dir / "table_02_causenet_dag_v_graph_type.md",
        "Table 2. CauseNet structural F1 and SHD by node count and graph type",
        standard_note + llm_bfs_md_note + " Average column/row report the corresponding marginal groups.",
        build_causenet_dag_v_graph_type_table(dag_summary),
    )
    write_table(
        final_dir / "table_06_desc_consensus_ablation.md",
        "Table 6. Description and consensus constraint ablation",
        "Key: green marks the overall best setting in each quality row; arrows on with-description cells indicate whether descriptions improve or reduce the corresponding no-description setting. Values are mean +/- std. Counts are not bolded because more constraints is not necessarily better.",
        build_desc_consensus_table(results_dir),
    )
    write_table(
        final_dir / "table_07_desc_consensus_nonzero_ablation.md",
        "Table 7. Description and consensus constraint ablation excluding zero-constraint cases",
        "Key: same formatting as Table 4, but each side/metric excludes rows where the corresponding required/forbidden constraint set is empty.",
        build_desc_consensus_table(results_dir, exclude_zero_constraints=True),
    )
    write_table(
        final_dir / "table_08_gpt_model_ablation_dag.md",
        "Table 8. Matched CauseNet structural metrics with Gemini/GPT baselines",
        standard_note
        + f" Strict matched dataset count: {dag_main_with_gpt_summary['dataset'].nunique()}. "
        + "LLM-constraint variants are labelled by source. Metrics are ordered as SHD, precision, recall, and F1."
        + llm_bfs_md_note,
        build_causenet_dag_full_table(
            dag_main_with_gpt_summary,
            methods=MAIN_WITH_GPT_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_table(
        final_dir / "table_04_bnlearn_dag_all_metrics.md",
        "Table 4. bnlearn structural metrics by dataset",
        "Key: among repeated-run methods, **bold** marks the best mean and any method not significantly different from it by Welch tests with Benjamini-Hochberg correction within each dataset/metric. Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking (*** p<0.001, ** p<0.01, * p<0.05, . p<0.1). Values are mean +/- std. Dataset labels report |V| and |E|."
        + llm_bfs_md_note,
        build_bnlearn_dag_table(results_dir),
    )
    write_table(
        final_dir / "table_05_runtime_by_node_count.md",
        "Table 5. Runtime by node count",
        "Values are mean +/- std seconds over CauseNet datasets.",
        runtime_table,
    )
    if alpha05_summary is not None:
        write_table(
            final_dir / "table_03_alpha_ablation_dag.md",
            "Table 3. CauseNet structural alpha ablation",
            standard_note
            + " Compares the current G^2 alpha=0.01 run with the archived alpha=0.05 run on the structural metrics."
            + llm_bfs_md_note,
            build_alpha_ablation_table(dag_summary, alpha05_summary),
        )

    all_tests = build_dag_final_tests(
        results_dir=results_dir,
        tag=args.tag,
        dag_summary=dag_summary,
        dag_main_with_gpt_summary=dag_main_with_gpt_summary,
        gpt_dag_summary=gpt_dag_summary,
        alpha05_summary=alpha05_summary,
    )
    abapcllm_main_tests = build_causenet_main_abapcllm_tests(dag_summary, tag=args.tag)
    abapcllm_main_tests.to_csv(check_dir / f"table_08_causenet_main_all_baseline_tests_{args.tag}.csv", index=False)
    write_table(
        check_dir / "table_08_causenet_main_all_baseline_tests.md",
        "Check. CauseNet structural tests for ABAPC-LLM against every Table 1 baseline",
        (
            "Canonical tests for Table 1. Display values match Table 1 exactly; `Subject test sd/n` and "
            "`Comparator test sd/n` show the standard deviations and sample sizes actually used in Welch tests."
        ),
        display_all_final_tests(abapcllm_main_tests),
    )
    if not all_tests.empty:
        all_tests.to_csv(check_dir / f"all_final_tests_{args.tag}.csv", index=False)
        write_table(
            check_dir / f"all_final_tests_{args.tag}.md",
            "Check. Complete final structural table tests",
            "Rows labelled `bolding_best_vs_comparator` compare the numeric best with each comparator and drive tied bolding. Rows labelled `marker_top_vs_next_nonbold` compare each bolded top method/variant with the next non-bold entry in the ranking and supply the significance symbols."
            + md_test_sig_note,
            display_all_final_tests(all_tests),
        )
        marker_tests = all_tests[all_tests["test_role"].eq("marker_top_vs_next_nonbold")].copy()
        marker_tests.to_csv(check_dir / f"welch_indicator_tests_{args.tag}.csv", index=False)
        structural_markers = marker_tests[~marker_tests["table"].eq("bnlearn")].copy()
        bnlearn_markers = marker_tests[marker_tests["table"].eq("bnlearn")].copy()
        if not structural_markers.empty:
            write_table(
                check_dir / f"structural_indicator_tests_{args.tag}.md",
                "Table 9. CauseNet Welch indicator tests",
                "These Welch tests produce the significance symbols in the CauseNet structural tables."
                + md_test_sig_note,
                display_compact_final_tests(structural_markers),
            )
        if not bnlearn_markers.empty:
            write_table(
                check_dir / f"bnlearn_indicator_tests_{args.tag}.md",
                "Table 10. bnlearn Welch indicator tests",
                "These Welch tests produce the significance symbols in the bnlearn table." + md_test_sig_note,
                display_compact_final_tests(bnlearn_markers),
            )
        write_table(
            check_dir / f"welch_indicator_tests_{args.tag}.md",
            "Table 9. Welch indicator tests",
            "These are the exact Welch tests used for significance symbols in the final structural tables. Each bolded top method/variant is compared against the next non-bold entry in that metric/group ranking, with BH correction applied within that marker-test family."
            + md_test_sig_note,
            display_compact_final_tests(marker_tests),
        )
        tied_tests = all_tests[
            all_tests["test_role"].eq("bolding_best_vs_comparator") & all_tests["comparator_tied"].astype(bool)
        ].copy()
        tied_tests.to_csv(check_dir / f"tied_bolding_tests_{args.tag}.csv", index=False)
        write_table(
            check_dir / f"tied_bolding_tests_{args.tag}.md",
            "Table 11. Tests behind additional bolded ties",
            "These are the non-significant best-vs-comparator tests that justify bolding methods other than the numeric best."
            + md_test_sig_note,
            display_compact_final_tests(tied_tests, tied=True),
        )

    write_dag_final_tex_tables(
        final_dir=final_dir,
        results_dir=results_dir,
        tables_dir=tables_dir,
        tag=args.tag,
        dag_summary=dag_summary,
        gpt_dag_summary=gpt_dag_summary,
        runtime_table=runtime_table,
        alpha05_summary=alpha05_summary,
        all_tests=all_tests,
        abapcllm_main_tests=abapcllm_main_tests,
    )
    return

    cpdag_summary = load_structural_summary(tables_dir, f"synthetic_cpdag_structural_summary_{args.tag}")
    dag_summary = load_structural_summary(tables_dir, f"synthetic_structural_summary_{args.tag}")
    gpt_summary = build_gpt_ablation_summary(results_dir, args.tag)
    gpt_dag_summary = build_gpt_ablation_dag_summary(results_dir, args.tag)
    main_with_gpt_summary = build_cpdag_main_with_gpt_summary(cpdag_summary, gpt_summary)
    dag_main_with_gpt_summary = build_dag_main_with_gpt_summary(dag_summary, gpt_dag_summary)

    standard_note = (
        "Key: **bold** marks the best mean and any method not significantly different from it "
        "by Welch tests with Benjamini-Hochberg correction within each metric/column. "
        "Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking "
        "(*** p<0.001, ** p<0.01, * p<0.05, . p<0.1). Values are mean +/- std."
    )
    cpdag_note = standard_note + " CPDAG metrics use the skeleton plus compelled v-structure convention."

    write_table(
        final_dir / "table_01_causenet_cpdag_main.md",
        "Table 1. CauseNet CPDAG main metrics by node count",
        cpdag_note,
        metric_by_group_table(cpdag_summary, CPDAG_MAIN_METRICS, node_groups()),
    )
    write_table(
        final_dir / "table_01b_causenet_cpdag_main_llm_sources.md",
        "Table 1b. Matched CauseNet CPDAG main metrics with Gemini/GPT baselines",
        cpdag_note
        + f" Strict matched dataset count: {main_with_gpt_summary['dataset'].nunique()}. "
        + "LLM-constraint variants are labelled by source.",
        metric_by_group_table(
            main_with_gpt_summary,
            CPDAG_MAIN_METRICS,
            node_groups(),
            methods=MAIN_WITH_GPT_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_table(
        final_dir / "table_01c_causenet_dag_main_llm_sources.md",
        "Table 1c. Matched CauseNet DAG main metrics with Gemini/GPT baselines",
        standard_note
        + f" Strict matched dataset count: {dag_main_with_gpt_summary['dataset'].nunique()}. "
        + "LLM-constraint variants are labelled by source.",
        metric_by_group_table(
            dag_main_with_gpt_summary,
            DAG_GPT_METRICS,
            node_groups(),
            methods=MAIN_WITH_GPT_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_table(
        final_dir / "table_02_causenet_cpdag_additional_metrics.md",
        "Table 2. CauseNet CPDAG additional metrics by node count",
        cpdag_note,
        metric_by_group_table(cpdag_summary, CPDAG_EXTRA_METRICS, node_groups()),
    )
    write_table(
        final_dir / "table_03_causenet_cpdag_graph_type.md",
        "Table 3. CauseNet CPDAG metrics by graph type",
        cpdag_note,
        metric_by_group_table(cpdag_summary, [*CPDAG_MAIN_METRICS, *CPDAG_EXTRA_METRICS], graph_type_groups()),
    )
    write_table(
        final_dir / "table_04_causenet_dag_all_metrics.md",
        "Table 4. CauseNet structural metrics by node count",
        standard_note,
        metric_by_group_table(dag_summary, DAG_ALL_METRICS, node_groups()),
    )
    write_table(
        final_dir / "table_04b_causenet_dag_graph_type.md",
        "Table 4b. CauseNet structural metrics by graph type",
        standard_note,
        metric_by_group_table(dag_summary, DAG_ALL_METRICS, graph_type_groups()),
    )
    write_table(
        final_dir / "table_05_bnlearn_cpdag_all_metrics.md",
        "Table 5. bnlearn CPDAG metrics by dataset",
        "Key: **bold** marks the best mean and any method not significantly different from it by Welch tests with Benjamini-Hochberg correction within each dataset/metric. Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking (*** p<0.001, ** p<0.01, * p<0.05, . p<0.1). Values are mean +/- std. CPDAG metrics use the skeleton plus compelled v-structure convention. Results are shown by dataset for methods with saved CPDAG metrics.",
        build_bnlearn_cpdag_table(results_dir),
    )
    write_table(
        final_dir / "table_05b_bnlearn_dag_all_metrics.md",
        "Table 5b. bnlearn structural metrics by dataset",
        "Key: **bold** marks the best mean and any method not significantly different from it by Welch tests with Benjamini-Hochberg correction within each dataset/metric. Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking (*** p<0.001, ** p<0.01, * p<0.05, . p<0.1). Values are mean +/- std.",
        build_bnlearn_dag_table(results_dir),
    )
    write_table(
        final_dir / "table_06_desc_consensus_ablation.md",
        "Table 6. Description and consensus constraint ablation",
        "Key: **bold** marks the best variant and any variant not significantly different from it by Welch tests with Benjamini-Hochberg correction within each dataset/side/metric row. Markers on bolded top cells denote corrected tests against the next non-bold variant in the ranking (*** p<0.001, ** p<0.01, * p<0.05, . p<0.1). Values are mean +/- std. Counts are not bolded because more constraints is not necessarily better.",
        build_desc_consensus_table(results_dir),
    )

    all_tests = build_all_final_tests(
        results_dir=results_dir,
        cpdag_summary=cpdag_summary,
        main_with_gpt_summary=main_with_gpt_summary,
        dag_summary=dag_summary,
        dag_main_with_gpt_summary=dag_main_with_gpt_summary,
        gpt_summary=gpt_summary,
        gpt_dag_summary=gpt_dag_summary,
    )
    if not all_tests.empty:
        all_tests.to_csv(final_dir / f"check_all_final_table_tests_{args.tag}.csv", index=False)
        write_table(
            final_dir / f"check_all_final_table_tests_{args.tag}.md",
            "Check. All final table best-vs-comparator tests",
            "Rows labelled `bolding_best_vs_comparator` compare the numeric best with each comparator and drive tied bolding. Rows labelled `marker_top_vs_next_nonbold` compare each bolded top method/variant with the next non-bold entry in the ranking and supply the significance symbols.",
            display_all_final_tests(all_tests),
        )
        marker_tests = all_tests[all_tests["test_role"].eq("marker_top_vs_next_nonbold")].copy()
        marker_tests.to_csv(final_dir / f"check_all_final_indicator_tests_{args.tag}.csv", index=False)
        write_table(
            final_dir / f"check_all_final_indicator_tests_{args.tag}.md",
            "Check. All final Welch significance indicators",
            "These are the exact tests used to print markers in the final tables. Each bolded top method/variant is compared against the next non-bold entry in that metric/group ranking, with BH correction applied within that marker-test family.",
            display_all_final_tests(marker_tests),
        )
        bolding_tests = all_tests[all_tests["test_role"].eq("bolding_best_vs_comparator")].copy()
        bolding_tests.to_csv(final_dir / f"check_all_final_bolding_tests_{args.tag}.csv", index=False)
        write_table(
            final_dir / f"check_all_final_bolding_tests_{args.tag}.md",
            "Check. All final tied-bolding tests",
            "These tests compare the numeric best with every comparator in a metric/group. Comparators with BH-corrected p >= 0.05 are bolded as statistically tied with the best.",
            display_all_final_tests(bolding_tests),
        )
    write_table(
        final_dir / "table_07_gpt_model_ablation.md",
        "Table 7. LLM model ablation on CPDAG metrics",
        cpdag_note + f" Strict matched dataset count: {gpt_summary['dataset'].nunique()}.",
        metric_by_group_table(
            gpt_summary,
            CPDAG_MAIN_METRICS,
            [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
            methods=GPT_ABLATION_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_table(
        final_dir / "table_07b_gpt_model_ablation_dag.md",
        "Table 7b. LLM model ablation on structural metrics",
        standard_note + f" Strict matched dataset count: {gpt_dag_summary['dataset'].nunique()}.",
        metric_by_group_table(
            gpt_dag_summary,
            DAG_GPT_METRICS,
            [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
            methods=GPT_ABLATION_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_table(
        final_dir / f"check_gpt_model_ablation_dag_{args.tag}.md",
        "Check. LLM model ablation on structural metrics",
        f"Current structural check from refreshed saved inputs. Strict matched dataset count: {gpt_dag_summary['dataset'].nunique()}. The +/- values use the final-table convention: mean of per-dataset run standard deviations. Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking.",
        metric_by_group_table(
            gpt_dag_summary,
            DAG_GPT_METRICS,
            [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
            methods=GPT_ABLATION_METHODS,
            method_labels=LLM_SOURCE_LABELS,
        ),
    )
    write_gpt_ablation_audit(final_dir, results_dir, args.tag)
    runtime_table = build_runtime_table(dag_summary)
    write_table(
        final_dir / "table_08_runtime_by_node_count.md",
        "Table 8. Runtime by node count",
        "Values are mean +/- std seconds over CauseNet datasets.",
        runtime_table,
    )
    write_table(
        final_dir / "table_09_cpdag_main_tests.md",
        "Table 9. CPDAG main metric tests for ABAPC-LLM",
        "Benjamini-Hochberg adjusted p-values are shown for ABAPC-LLM versus each comparator on the main skeleton plus compelled v-structure CPDAG metrics.",
        build_tests_table(tables_dir, args.tag),
    )
    write_table(
        final_dir / f"check_dag_main_tests_abapcllm_{args.tag}.md",
        "Check. DAG main metric tests for ABAPC-LLM",
        "Benjamini-Hochberg adjusted p-values are shown for ABAPC-LLM versus each comparator on the main structural F1 and SHD metrics.",
        build_tests_table(tables_dir, args.tag, graph_kind="dag"),
    )
    write_table(
        final_dir / "legacy_table_09b_dag_main_tests.md",
        "Legacy Table 9b. DAG main metric tests for ABAPC-LLM",
        "Benjamini-Hochberg adjusted p-values are shown for ABAPC-LLM versus each comparator on the main structural F1 and SHD metrics.",
        build_tests_table(tables_dir, args.tag, graph_kind="dag"),
    )
    write_final_tex_tables(
        final_dir=final_dir,
        results_dir=results_dir,
        tables_dir=tables_dir,
        tag=args.tag,
        cpdag_summary=cpdag_summary,
        main_with_gpt_summary=main_with_gpt_summary,
        dag_summary=dag_summary,
        gpt_summary=gpt_summary,
        gpt_dag_summary=gpt_dag_summary,
        runtime_table=runtime_table,
    )
    write_tex(
        final_dir / f"check_gpt_model_ablation_dag_{args.tag}.tex",
        tex_table(
            caption=(
                rf"Current structural Gemini/GPT-5-mini comparison on the {gpt_dag_summary['dataset'].nunique()} "
                r"\texttt{CauseNet} synthetic datasets where both consensus runs are available. "
                r"Bold marks the best mean and methods not significantly different from it under "
                r"Welch tests with Benjamini--Hochberg correction within each metric/column. "
                r"Markers on bolded top cells denote corrected tests against the next non-bold method in the ranking "
                r"(\textnormal{***} $p<0.001$, \textnormal{**} $p<0.01$, \textnormal{*} $p<0.05$, "
                r"\textnormal{.} $p<0.1$)."
            ),
            label="tab:gpt_model_ablation_dag_check",
            body=latex_metric_by_group_table(
                gpt_dag_summary,
                DAG_GPT_METRICS,
                [("Overall", lambda df: pd.Series(True, index=df.index)), *node_groups()],
                methods=GPT_ABLATION_METHODS,
                method_labels=LLM_SOURCE_LABELS,
            ),
            table_env="table*",
            resize=r"\textwidth",
        ),
    )
    write_tex(
        final_dir / f"check_dag_main_tests_abapcllm_{args.tag}.tex",
        tex_table(
            caption=(
                r"Two-sample, unequal variance $t$-tests comparing ABAPC-LLM against other methods "
                r"on the main structural metrics for \texttt{CauseNet} synthetic datasets. "
                r"BH-corrected $p$-values account for multiple comparisons."
            ),
            label="tab:dag_main_tests_check",
            body=latex_tests_table(tables_dir, args.tag, graph_kind="dag"),
            table_env="table*",
            resize=r"\textwidth",
            tabcolsep="2pt",
        ),
    )


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    final_dir = tables_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_archive:
        archive_dir, legacy_check_dir = archive_current_final_files(final_dir)
        if archive_dir is not None:
            print(f"Archived current final tables to {archive_dir}")
        if legacy_check_dir is not None:
            print(f"Moved current check files to {legacy_check_dir}")
    write_final_tables(args)
    print(f"Wrote final numbered tables to {tables_dir / 'final'}")


if __name__ == "__main__":
    main()
