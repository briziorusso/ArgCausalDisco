"""Generate UAI paper tables from refreshed experiment artifacts.

This script is intentionally file-based: after the long experiment runs finish,
rerun it to regenerate CSV, Markdown, LaTeX snippets, and a source manifest under
``results/tables``.  It does not run experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DAG_SUMMARY_COLUMNS = ["dataset", "model"] + [
    f"{metric}_{stat}"
    for metric in [
        "elapsed",
        "nnz",
        "fdr",
        "tpr",
        "fpr",
        "precision",
        "recall",
        "F1",
        "shd",
        "SID",
    ]
    for stat in ["mean", "std"]
]

SYNTH_METHOD_ORDER = [
    "Random",
    "FGS",
    "NOTEARS-MLP",
    "GRaSP",
    "BOSS",
    "MPC",
    "MPC-LLM",
    "ABAPC",
    "ABAPC-LLM",
]

GPT_METHOD_ORDER = [
    "ABAPC-LLM",
    "ABAPC-LLM-gpt5mini",
    "ABAPC",
    "MPC-LLM",
    "MPC-LLM-gpt5mini",
    "MPC",
]


@dataclass
class SourceLog:
    rows: list[dict[str, str]]

    def add(self, artifact: str, path: Path, status: str) -> None:
        self.rows.append({"artifact": artifact, "path": str(path), "status": status})

    def write(self, path: Path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate UAI table artifacts from refreshed g2/alpha results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", default="g2a01", help="Result suffix/tag to use.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip tables whose refreshed inputs are still missing.",
    )
    return parser.parse_args()


def require_path(path: Path, source_log: SourceLog, artifact: str, allow_missing: bool) -> bool:
    if path.exists():
        source_log.add(artifact, path, "found")
        return True
    source_log.add(artifact, path, "missing")
    if allow_missing:
        return False
    raise FileNotFoundError(f"Missing required {artifact}: {path}")


def require_all(
    items: Iterable[tuple[str, Path]],
    source_log: SourceLog,
    allow_missing: bool,
) -> bool:
    ok = True
    for name, path in items:
        ok = require_path(path, source_log, name, allow_missing) and ok
    return ok


def load_npy_summary(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    return pd.DataFrame(arr, columns=DAG_SUMMARY_COLUMNS)


def n_nodes(dataset: str) -> int:
    match = re.search(r"dag_(\d+)_nodes_", str(dataset))
    if not match:
        raise ValueError(f"Cannot parse node count from dataset name: {dataset}")
    return int(match.group(1))


def n_edges(dataset: str) -> int:
    match = re.search(r"_nodes_(\d+)_edges_", str(dataset))
    if not match:
        raise ValueError(f"Cannot parse edge count from dataset name: {dataset}")
    return int(match.group(1))


def fmt_num(value: float, ndigits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{ndigits}f}"


def fmt_pm(mean: float, std: float, ndigits: int = 3, latex: bool = False) -> str:
    if pd.isna(mean):
        return "--"
    text = f"{fmt_num(mean, ndigits)}+-{fmt_num(std, ndigits)}"
    if latex:
        return rf"\num{{{text}}}"
    return text.replace("+-", " +/- ")


def sig_marker(p_value: float) -> str:
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return "ns"


def p_fmt(p_value: float) -> str:
    if pd.isna(p_value):
        return "--"
    if p_value == 0:
        return "0"
    if abs(p_value) < 1e-3:
        mantissa, exponent = f"{p_value:.2e}".split("e")
        return rf"{float(mantissa):.2f}\mathrm{{e}}{{{int(exponent)}}}"
    return f"{p_value:.3f}"


def bh_adjust(p_values: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg correction without a statsmodels dependency."""

    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=bool), np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running_min = min(running_min, ranked[i] * n / rank)
        adjusted_ranked[i] = running_min
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted <= alpha, adjusted


def normal_two_sided_p(t_stat: float) -> float:
    return math.erfc(abs(float(t_stat)) / math.sqrt(2.0))


def welch_from_arrays(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    try:
        from scipy.stats import ttest_ind  # type: ignore

        t_stat, p_value = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(t_stat), float(p_value)
    except Exception:
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        se = math.sqrt(var_a / len(a) + var_b / len(b))
        if se == 0:
            return 0.0, 1.0
        t_stat = (mean_a - mean_b) / se
        return t_stat, normal_two_sided_p(t_stat)


def welch_from_stats(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> tuple[float, float]:
    try:
        from scipy.stats import ttest_ind_from_stats  # type: ignore

        t_stat, p_value = ttest_ind_from_stats(mean_a, std_a, n_a, mean_b, std_b, n_b, equal_var=False)
        return float(t_stat), float(p_value)
    except Exception:
        se = math.sqrt((std_a**2) / n_a + (std_b**2) / n_b)
        if se == 0:
            return 0.0, 1.0
        t_stat = (mean_a - mean_b) / se
        return t_stat, normal_two_sided_p(t_stat)


def write_text_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(markdown_table(df) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max([len(columns[i])] + [len(row[i]) for row in rows])
        for i in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [fmt_row(columns), fmt_row(["-" * width for width in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def write_latex(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def load_empty_prior_names(path: Path) -> set[str]:
    priors = pd.read_json(path)
    return {
        row.filename
        for row in priors.itertuples(index=False)
        if not ((row.priors or {}).get("forbidden") or (row.priors or {}).get("required"))
    }


def aggregate_abapc_runs(df: pd.DataFrame, impl: str, label: str) -> pd.DataFrame:
    sub = df[df["impl"].eq(impl)].copy()
    return (
        sub.groupby("dataset", as_index=False)
        .agg(
            **{
                f"{label} F1": ("dag_F1", "mean"),
                f"{label} SHD": ("dag_shd", "mean"),
                f"{label} elapsed": ("time", "mean"),
            }
        )
    )


def complete_empty_prior_abapc_rows(
    primary: pd.DataFrame,
    empty_prior_names: Iterable[str],
    fallback_frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """Fill empty-prior datasets with plain ABAPC rows.

    When consensus priors are empty, ABAPC-LLM should reduce to the same solver
    without semantic constraints.  Some experiment files skip those datasets
    entirely, so we materialise explicit fallback rows for table/plot refreshes.
    """

    frames = [primary.copy()]
    fallback = pd.concat([f.copy() for f in fallback_frames if f is not None and not f.empty], ignore_index=True)
    if fallback.empty:
        return primary

    for dataset in empty_prior_names:
        org_rows = fallback[(fallback["dataset"].eq(dataset)) & (fallback["impl"].eq("org"))].copy()
        if org_rows.empty:
            continue
        for impl in ["org", "new"]:
            exists = (primary["dataset"].eq(dataset) & primary["impl"].eq(impl)).any()
            if exists:
                continue
            rows = org_rows.copy()
            rows["impl"] = impl
            frames.append(rows)

    return pd.concat(frames, ignore_index=True)


def load_abapc_csv(
    path: Path,
    *,
    consensus_path: Path | None = None,
    fallback_paths: Iterable[Path] = (),
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if consensus_path is None:
        return df
    fallback_frames = [pd.read_csv(p) for p in fallback_paths if p.exists()]
    return complete_empty_prior_abapc_rows(df, load_empty_prior_names(consensus_path), fallback_frames)


def load_mpc(path: Path, model: str, label: str) -> pd.DataFrame:
    return (
        load_npy_summary(path)
        .query("model == @model")
        [["dataset", "F1_mean", "shd_mean", "elapsed_mean"]]
        .rename(
            columns={
                "F1_mean": f"{label} F1",
                "shd_mean": f"{label} SHD",
                "elapsed_mean": f"{label} elapsed",
            }
        )
    )


def merge_frames(frames: list[pd.DataFrame], how: str = "inner") -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="dataset", how=how)
    return merged


def finite_metric_mask(df: pd.DataFrame) -> pd.Series:
    metric_cols = [c for c in df.columns if c != "dataset"]
    return df[metric_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def gpt_summary_table(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.copy()
    valid["n_nodes"] = valid["dataset"].map(n_nodes)
    rows = []
    groups = [("Overall", valid)] + [
        (f"|V|={n}", valid[valid["n_nodes"].eq(n)]) for n in sorted(valid["n_nodes"].unique())
    ]
    for group_label, group in groups:
        row = {"Group": group_label, "Datasets": len(group)}
        for method in GPT_METHOD_ORDER:
            row[f"{method} F1"] = fmt_pm(group[f"{method} F1"].mean(), group[f"{method} F1"].std(ddof=1))
        for method in GPT_METHOD_ORDER:
            row[f"{method} SHD"] = fmt_pm(group[f"{method} SHD"].mean(), group[f"{method} SHD"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def should_bold(values: dict[str, np.ndarray], higher_is_better: bool) -> set[str]:
    means = {m: np.nanmean(v) for m, v in values.items() if len(v) > 1 and np.isfinite(v).any()}
    if not means:
        return set()
    best = max(means, key=means.get) if higher_is_better else min(means, key=means.get)
    p_values = []
    methods = []
    for method, vals in values.items():
        if method == best or len(vals) < 2:
            continue
        _, p_value = welch_from_arrays(values[best], vals)
        p_values.append(p_value)
        methods.append(method)
    bold = {best}
    if p_values:
        reject, _ = bh_adjust(p_values, alpha=0.05)
        for method, reject_one in zip(methods, reject):
            if not reject_one:
                bold.add(method)
    return bold


def gpt_latex_table(valid: pd.DataFrame) -> str:
    valid = valid.copy()
    valid["n_nodes"] = valid["dataset"].map(n_nodes)
    groups = [("Overall", valid)] + [
        (rf"$|\nodeSet|={n}$", valid[valid["n_nodes"].eq(n)]) for n in sorted(valid["n_nodes"].unique())
    ]

    def metric_table(metric: str, higher_is_better: bool) -> str:
        arrow = r"\uparrow" if higher_is_better else r"\downarrow"
        lines = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            rf"\multicolumn{{5}}{{c}}{{\textbf{{{metric} {arrow}}}}} \\",
            r"\midrule",
            r"\textbf{Method} & \textbf{Overall} & $\boldsymbol{|\nodeSet|=5}$ & $\boldsymbol{|\nodeSet|=10}$ & $\boldsymbol{|\nodeSet|=15}$ \\",
            r"\midrule",
        ]
        for method in GPT_METHOD_ORDER:
            cells = [method.replace("gpt5mini", "GPT")]
            for _, group in groups:
                col = f"{method} {metric}"
                values = {
                    m: group[f"{m} {metric}"].dropna().astype(float).to_numpy()
                    for m in GPT_METHOD_ORDER
                    if f"{m} {metric}" in group.columns
                }
                bold = should_bold(values, higher_is_better)
                mean = group[col].mean()
                std = group[col].std(ddof=1)
                cell = rf"${mean:.3f}\pm{std:.3f}$"
                if method in bold:
                    cell = rf"$\mathbf{{{mean:.3f}\pm{std:.3f}}}$"
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines)

    return (
        metric_table("F1", higher_is_better=True)
        + "\n\n\\vspace{0.15cm}\n"
        + metric_table("SHD", higher_is_better=False)
    )


def build_gpt_tables(args: argparse.Namespace, source_log: SourceLog) -> None:
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    tag = args.tag
    allow_missing = args.allow_missing

    gemini_csv = results_dir / "ABAPC-LLM" / f"merged_synthetic-desc_{tag}.csv"
    gpt_csv = results_dir / "ABAPC-LLM" / f"merged_synthetic-desc-gpt5mini_{tag}.csv"
    fallback_csv = results_dir / "ABAPC-LLM" / f"merged_synthetic_{tag}.csv"
    gemini_consensus = results_dir / "llm_constraints" / "synthetic-desc-consensus.json"
    gpt_consensus = results_dir / "llm_constraints" / "synthetic-desc-gpt5mini-consensus.json"
    mpc_path = results_dir / f"stored_results_causenet_mpc_desc_{tag}.npy"
    gpt_mpc_path = results_dir / f"stored_results_causenet_mpc_llm_gpt5mini_desc_{tag}.npy"

    required = [
        ("gemini_abapc_csv", gemini_csv),
        ("gpt_abapc_csv", gpt_csv),
        ("plain_abapc_fallback_csv", fallback_csv),
        ("gemini_consensus", gemini_consensus),
        ("gpt_consensus", gpt_consensus),
        ("mpc_mpc_llm_summary", mpc_path),
        ("gpt_mpc_llm_summary", gpt_mpc_path),
    ]
    if not require_all(required, source_log, allow_missing):
        print("Skipping GPT structural table because refreshed inputs are incomplete.")
        return

    gemini = load_abapc_csv(gemini_csv, consensus_path=gemini_consensus, fallback_paths=[fallback_csv])
    gpt = load_abapc_csv(gpt_csv)
    abapc = aggregate_abapc_runs(gemini, "org", "ABAPC")
    abapc_llm = aggregate_abapc_runs(gemini, "new", "ABAPC-LLM")
    abapc_llm_gpt = aggregate_abapc_runs(gpt, "new", "ABAPC-LLM-gpt5mini")
    mpc = load_mpc(mpc_path, "MPC", "MPC")
    mpc_llm = load_mpc(mpc_path, "MPC-LLM", "MPC-LLM")
    mpc_llm_gpt = load_mpc(gpt_mpc_path, "MPC-LLM", "MPC-LLM-gpt5mini")

    strict = merge_frames([abapc_llm, abapc_llm_gpt, abapc, mpc_llm, mpc_llm_gpt, mpc], how="inner")
    strict = strict[finite_metric_mask(strict)].copy()
    summary = gpt_summary_table(strict)
    stem = f"gemini_gpt5_structural_{tag}"
    write_text_table(summary, out_dir, stem)
    write_latex(out_dir / f"{stem}.tex", gpt_latex_table(strict))
    print(f"Wrote {stem}: {len(strict)} matched datasets")


def constraint_stats(path: Path, *, filtered: bool, prefix: str) -> dict[str, tuple[float, float]]:
    df = pd.read_json(path)
    metrics = {}
    for side in ["forbidden", "required"]:
        sub = df
        if filtered:
            sub = df[df[f"{side}_length"] > 0]
        for metric in ["length", "Precision", "Recall", "F1"]:
            col = f"{side}_{metric}"
            metrics[f"{prefix}_{side}_{metric}"] = (float(sub[col].mean()), float(sub[col].std()))
    return metrics


def build_constraint_table(results_dir: Path, out_dir: Path, *, filtered: bool, source_log: SourceLog, allow_missing: bool) -> None:
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
    if not require_all(
        [(f"constraints_{key}", path) for key, path in sources.items()],
        source_log,
        allow_missing,
    ):
        print("Skipping constraint quality table because inputs are incomplete.")
        return

    row_defs = [
        ("Forbidden constraints", "forbidden", "length", "# constraints"),
        ("Forbidden constraints", "forbidden", "Precision", "Precision"),
        ("Forbidden constraints", "forbidden", "Recall", "Recall"),
        ("Forbidden constraints", "forbidden", "F1", "F1"),
        ("Required constraints", "required", "length", "# constraints"),
        ("Required constraints", "required", "Precision", "Precision"),
        ("Required constraints", "required", "Recall", "Recall"),
        ("Required constraints", "required", "F1", "F1"),
    ]
    col_defs = list(sources.keys())
    rows = []
    latex_lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{bnlearn}} & \multicolumn{4}{c}{\textbf{CauseNet}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Metric & \multicolumn{2}{c}{Average} & \multicolumn{2}{c}{Consensus} & \multicolumn{2}{c}{Average} & \multicolumn{2}{c}{Consensus} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & w/o desc & with desc & w/o desc & with desc & w/o desc & with desc & w/o desc & with desc \\",
        r"\midrule",
    ]
    last_section = None
    for section, side, metric, label in row_defs:
        if section != last_section:
            latex_lines.extend([rf"\multicolumn{{9}}{{c}}{{\textbf{{{section}}}}} \\", r"\midrule"])
            last_section = section
        row = {"Section": section, "Metric": label}
        latex_cells = [label]
        for key in col_defs:
            path = sources[key]
            df = pd.read_json(path)
            sub = df[df[f"{side}_length"] > 0] if filtered else df
            mean = float(sub[f"{side}_{metric}"].mean())
            std = float(sub[f"{side}_{metric}"].std())
            row[" / ".join(key)] = fmt_pm(mean, std)
            latex_cells.append(fmt_pm(mean, std, latex=True))
        rows.append(row)
        latex_lines.append(" & ".join(latex_cells) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])

    stem = "llm_constraints_quality_filtered" if filtered else "llm_constraints_quality"
    table = pd.DataFrame(rows)
    write_text_table(table, out_dir, stem)
    write_latex(out_dir / f"{stem}.tex", "\n".join(latex_lines))
    print(f"Wrote {stem}")


def summary_rows_from_npy(path: Path, include: set[str] | None = None) -> pd.DataFrame:
    df = load_npy_summary(path)
    if include is not None:
        df = df[df["model"].isin(include)].copy()
    df["source_kind"] = "npy"
    return df


def summary_rows_from_abapc(csv_path: Path, consensus_path: Path, fallback_csv: Path) -> pd.DataFrame:
    df = load_abapc_csv(csv_path, consensus_path=consensus_path, fallback_paths=[fallback_csv])
    mapping = {"org": "ABAPC", "new": "ABAPC-LLM"}
    rows = []
    for (dataset, impl), sub in df.groupby(["dataset", "impl"]):
        model = mapping.get(impl, impl)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "elapsed_mean": sub["time"].mean(),
                "elapsed_std": sub["time"].std(ddof=1),
                "F1_mean": sub["dag_F1"].mean(),
                "F1_std": sub["dag_F1"].std(ddof=1),
                "shd_mean": sub["dag_shd"].mean(),
                "shd_std": sub["dag_shd"].std(ddof=1),
                "precision_mean": sub["dag_precision"].mean(),
                "precision_std": sub["dag_precision"].std(ddof=1),
                "recall_mean": sub["dag_recall"].mean(),
                "recall_std": sub["dag_recall"].std(ddof=1),
                "SID_mean": sub["dag_sid"].mean(),
                "SID_std": sub["dag_sid"].std(ddof=1),
                "source_kind": "csv",
            }
        )
    return pd.DataFrame(rows)


def structural_summary(args: argparse.Namespace, source_log: SourceLog) -> pd.DataFrame | None:
    results_dir = Path(args.results_dir)
    tag = args.tag
    allow_missing = args.allow_missing
    inputs = {
        "random_summary": results_dir / "stored_results_test_rnd_spc.npy",
        "fgs_notears_summary": results_dir / "stored_results_causenet_base.npy",
        "boss_grasp_summary": results_dir / "stored_results_causenet_boss_grasp.npy",
        "mpc_mpc_llm_summary": results_dir / f"stored_results_causenet_mpc_desc_{tag}.npy",
        "abapc_csv": results_dir / "ABAPC-LLM" / f"merged_synthetic-desc_{tag}.csv",
        "abapc_fallback_csv": results_dir / "ABAPC-LLM" / f"merged_synthetic_{tag}.csv",
        "abapc_consensus": results_dir / "llm_constraints" / "synthetic-desc-consensus.json",
    }
    if not require_all(inputs.items(), source_log, allow_missing):
        print("Skipping structural summaries because inputs are incomplete.")
        return None

    frames = [
        summary_rows_from_npy(inputs["random_summary"], {"Random"}),
        summary_rows_from_npy(inputs["fgs_notears_summary"], {"FGS", "NOTEARS-MLP"}),
        summary_rows_from_npy(inputs["boss_grasp_summary"], {"GRaSP", "BOSS"}),
        summary_rows_from_npy(inputs["mpc_mpc_llm_summary"], {"MPC", "MPC-LLM"}),
        summary_rows_from_abapc(inputs["abapc_csv"], inputs["abapc_consensus"], inputs["abapc_fallback_csv"]),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["n_nodes"] = df["dataset"].map(n_nodes)
    df["n_edges"] = df["dataset"].map(n_edges)
    df["NSHD_mean"] = df["shd_mean"].astype(float) / df["n_edges"].astype(float)
    df["NSHD_std"] = df["shd_std"].astype(float) / df["n_edges"].astype(float)
    return df[df["model"].isin(SYNTH_METHOD_ORDER)].copy()


def build_runtime_table(summary: pd.DataFrame, out_dir: Path, tag: str) -> None:
    rows = []
    for model in SYNTH_METHOD_ORDER:
        row = {"Method": model}
        sub_model = summary[summary["model"].eq(model)]
        for nodes in [5, 10, 15]:
            sub = sub_model[sub_model["n_nodes"].eq(nodes)]
            row[f"|V|={nodes}"] = fmt_pm(sub["elapsed_mean"].mean(), sub["elapsed_mean"].std(ddof=1), ndigits=2)
        rows.append(row)
    table = pd.DataFrame(rows)
    stem = f"runtime_scaling_{tag}"
    write_text_table(table, out_dir, stem)

    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & $|\nodeSet|=5$ & $|\nodeSet|=10$ & $|\nodeSet|=15$ \\", r"\midrule"]
    for row in rows:
        lines.append(
            f"{row['Method']} & {row['|V|=5']} & {row['|V|=10']} & {row['|V|=15']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(out_dir / f"{stem}.tex", "\n".join(lines))
    print(f"Wrote {stem}")


def build_tests_table(summary: pd.DataFrame, out_dir: Path, tag: str) -> None:
    comparisons = [m for m in SYNTH_METHOD_ORDER if m != "ABAPC-LLM" and m in set(summary["model"])]
    records = []
    for nodes in [5, 10, 15]:
        sub_nodes = summary[summary["n_nodes"].eq(nodes)]
        nobs = max(1, int(sub_nodes["dataset"].nunique() * 50))
        ref = sub_nodes[sub_nodes["model"].eq("ABAPC-LLM")]
        if ref.empty:
            continue
        for metric, higher in [("F1", True), ("NSHD", False)]:
            ref_mean = float(ref[f"{metric}_mean"].mean())
            ref_std = float(ref[f"{metric}_mean"].std(ddof=1))
            for other in comparisons:
                comp = sub_nodes[sub_nodes["model"].eq(other)]
                if comp.empty:
                    continue
                comp_mean = float(comp[f"{metric}_mean"].mean())
                comp_std = float(comp[f"{metric}_mean"].std(ddof=1))
                t_stat, p_value = welch_from_stats(
                    ref_mean,
                    ref_std if not math.isnan(ref_std) else 0.0,
                    nobs,
                    comp_mean,
                    comp_std if not math.isnan(comp_std) else 0.0,
                    nobs,
                )
                records.append(
                    {
                        "nodes": nodes,
                        "metric": metric,
                        "comparison": f"ABAPC-LLM vs {other}",
                        "ref_mean": ref_mean,
                        "ref_std": ref_std,
                        "other_mean": comp_mean,
                        "other_std": comp_std,
                        "t": t_stat,
                        "p_value": p_value,
                    }
                )
    table = pd.DataFrame(records)
    if table.empty:
        return
    table["p_bh"] = np.nan
    for (_, metric), idx in table.groupby(["nodes", "metric"]).groups.items():
        _, p_bh = bh_adjust(table.loc[idx, "p_value"], alpha=0.05)
        table.loc[idx, "p_bh"] = p_bh
    table["marker"] = table["p_value"].map(sig_marker)
    table["marker_bh"] = table["p_bh"].map(sig_marker)
    stem = f"structural_tests_abapcllm_bh_{tag}"
    write_text_table(table, out_dir, stem)

    lines = [
        r"\begin{tabular}{c|c|l|c|r|rl|rl}",
        r"\toprule",
        r"Dataset & Metric & Methods & Means$\pm$Std & $t$ & \multicolumn{2}{c|}{$p$} & \multicolumn{2}{c}{$p_{\mathrm{BH}}$} \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        means = rf"${row.ref_mean:.3f}\pm{row.ref_std:.3f}$ vs ${row.other_mean:.3f}\pm{row.other_std:.3f}$"
        lines.append(
            rf"$|\nodeSet|={row.nodes}$ & {row.metric} & {row.comparison} & {means} & "
            rf"${row.t:.2f}$ & ${p_fmt(row.p_value)}$ & {row.marker} & ${p_fmt(row.p_bh)}$ & {row.marker_bh} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(out_dir / f"{stem}.tex", "\n".join(lines))
    print(f"Wrote {stem}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_log = SourceLog(rows=[])

    build_gpt_tables(args, source_log)
    build_constraint_table(Path(args.results_dir), out_dir, filtered=False, source_log=source_log, allow_missing=args.allow_missing)
    build_constraint_table(Path(args.results_dir), out_dir, filtered=True, source_log=source_log, allow_missing=args.allow_missing)

    summary = structural_summary(args, source_log)
    if summary is not None:
        summary_path = out_dir / f"synthetic_structural_summary_{args.tag}.csv"
        summary.to_csv(summary_path, index=False)
        source_log.add("synthetic_structural_summary", summary_path, "written")
        build_runtime_table(summary, out_dir, args.tag)
        build_tests_table(summary, out_dir, args.tag)

    source_log.write(out_dir / f"paper_table_sources_{args.tag}.csv")
    print(f"Wrote source manifest: {out_dir / f'paper_table_sources_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
