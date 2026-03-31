#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RESULTS_DIR = REPO_ROOT / "results"
DATASET_ORDER = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]
DATASET_TITLE = {
    "cancer": "Cancer",
    "earthquake": "Earthquake",
    "survey": "Survey",
    "asia": "Asia",
    "sachs": "Sachs",
    "child": "Child",
}

METHOD_ORDER = [
    "Random",
    "rnd-dir",
    "Random (match |E|)",
    "FGS",
    "NOTEARS-MLP",
    "MPC",
    "ABAPC (orig)",
    "ABAPC (nor)",
    "ABAPC (bb)",
    "ABAPC (bb-nor)",
]

METRIC_TITLE = {
    "precision": "Precision",
    "recall": "Recall",
    "F1": "F1",
    "shd": "SHD",
    "sid": "SID",
    "sid_low": "SID (low)",
    "sid_high": "SID (high)",
    "adjacency_precision": "Skeleton Precision",
    "adjacency_recall": "Skeleton Recall",
    "adjacency_F1": "Skeleton F1",
    "arrowhead_precision": "Arrowhead Precision",
    "arrowhead_recall": "Arrowhead Recall",
    "arrowhead_F1": "Arrowhead F1",
}


def _version_has_artifacts(version: str) -> bool:
    return any(
        path.exists()
        for path in [
            RESULTS_DIR / f"stored_results_{version}.npy",
            RESULTS_DIR / f"stored_results_{version}_cpdag.npy",
            RESULTS_DIR / "progress" / version,
        ]
    )


def _resolve_version(primary: str, *fallbacks: str) -> str:
    for candidate in (primary, *fallbacks):
        if candidate and _version_has_artifacts(candidate):
            return candidate
    return primary


def _parse_aliases(items: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method-alias entry {item!r}; expected Pretty Name=Alias")
        src, dst = item.split("=", 1)
        aliases[src.strip()] = dst.strip()
    return aliases


def _significance_suffix(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    if p_value <= 0.1:
        return "."
    return ""


def _metric_label(metric: str) -> str:
    return METRIC_TITLE.get(metric, metric.replace("_", " ").title())


def _dataset_title(dataset: str) -> str:
    return DATASET_TITLE.get(dataset, dataset.title())


def _latex_row(
    method1: str,
    mean1: float,
    std1: float,
    method2: str,
    mean2: float,
    std2: float,
    stat: float,
    p_value: float,
) -> str:
    return (
        rf"\!\!\!{method1} $({mean1:.1f}\pm{std1:.1f})$ "
        rf"\!v\! {method2} $({mean2:.1f}\pm{std2:.1f})$ "
        rf"\!\!\!&\!\!\! {stat:.3f} \!\!\!&\!\!\! {p_value:.3f}{_significance_suffix(p_value)} \\"
    )


def _load_progress_rows(version: str, kind: str) -> pd.DataFrame:
    progress_dir = RESULTS_DIR / "progress" / version
    suffix = "_cpdag.csv" if kind == "cpdag" else "_dag.csv"
    if not progress_dir.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(progress_dir.glob(f"*{suffix}")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"dataset", "model"}.issubset(frame.columns):
            frame["dataset"] = frame["dataset"].astype(str).str.lower()
            frame["model"] = frame["model"].astype(str)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_combined_progress(kind: str, run_specs: list[dict[str, object]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in run_specs:
        if spec["kind"] != kind:
            continue
        frame = _load_progress_rows(str(spec["version"]), kind)
        if frame.empty:
            continue
        include = spec.get("include")
        if include:
            frame = frame[frame["model"].isin(include)].copy()
        replace_models = spec.get("replace_models") or {}
        if replace_models:
            frame["model"] = frame["model"].replace(replace_models)
        include_datasets = spec.get("include_datasets")
        if include_datasets:
            frame = frame[frame["dataset"].isin(include_datasets)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_run_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    abapc_orig_others_version = _resolve_version(
        args.abapc_orig_others_version,
        "abapc_orig_problem3_matched10_gsq",
    )
    abapc_orig_child_version = _resolve_version(args.abapc_orig_child_version)

    run_specs = [
        {"version": args.random_version or args.random_fgs_version, "kind": "dag", "include": ["Random"]},
        {"version": args.fgs_version or args.random_fgs_version, "kind": "dag", "include": ["FGS"]},
        {"version": args.nt_version, "kind": "dag", "include": ["NOTEARS-MLP"]},
        {"version": args.mpc_version, "kind": "dag", "include": ["MPC"]},
        {
            "version": args.abapc_nor_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_nor_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_bb_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_nor_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {
            "version": args.abapc_bb_nor_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {"version": args.random_version or args.random_fgs_version, "kind": "cpdag", "include": ["Random"]},
        {"version": args.fgs_version or args.random_fgs_version, "kind": "cpdag", "include": ["FGS"]},
        {"version": args.nt_version, "kind": "cpdag", "include": ["NOTEARS-MLP"]},
        {"version": args.mpc_version, "kind": "cpdag", "include": ["MPC"]},
        {
            "version": args.abapc_nor_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_nor_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_bb_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_nor_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {
            "version": args.abapc_bb_nor_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
    ]

    if args.include_orig:
        run_specs.extend(
            [
                {
                    "version": abapc_orig_others_version,
                    "kind": "dag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_child_version,
                    "kind": "dag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["child"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_others_version,
                    "kind": "cpdag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_child_version,
                    "kind": "cpdag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["child"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
            ]
        )

    return run_specs


def _render_dataset_table(
    frame: pd.DataFrame,
    dataset: str,
    metrics: list[str],
    methods: list[str],
    aliases: dict[str, str],
) -> str:
    dataset_rows = frame[frame["dataset"] == dataset].copy()
    available_methods = [method for method in methods if method in dataset_rows["model"].unique()]

    lines = [
        r"\begin{table}[ht]",
        rf"    \caption{{t-tests for difference in means for {_dataset_title(dataset)} dataset. \\ Significance levels: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1.}}",
        rf"    \label{{tab:{dataset}_tests}}",
        r"    \centering",
        r"    \begin{tabular}{rrl}",
        r"        \!\!\! Method (mean$\pm$std)  & \!\!\!t & \!\!\!p-value \\",
        r"        \hline",
    ]

    for metric_index, metric in enumerate(metrics):
        pretty_metric = _metric_label(metric)
        lines.append(rf"        \multicolumn{{1}}{{l}}{{{pretty_metric}}}&& \\")
        lines.append(r"        \hline        \\[\dimexpr-\normalbaselineskip+2pt]")

        metric_rows_added = 0
        for method1, method2 in combinations(available_methods, 2):
            vals1 = pd.to_numeric(
                dataset_rows.loc[dataset_rows["model"] == method1, metric],
                errors="coerce",
            ).dropna()
            vals2 = pd.to_numeric(
                dataset_rows.loc[dataset_rows["model"] == method2, metric],
                errors="coerce",
            ).dropna()
            if len(vals1) == 0 or len(vals2) == 0:
                continue

            arr1 = vals1.to_numpy(dtype=float)
            arr2 = vals2.to_numpy(dtype=float)
            if len(arr1) == len(arr2) and np.allclose(arr1, arr2, equal_nan=True):
                stat, p_value = 0.0, 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    stat, p_value = ttest_ind(arr1, arr2, equal_var=False, nan_policy="omit")
            label1 = aliases.get(method1, method1)
            label2 = aliases.get(method2, method2)
            lines.append(
                _latex_row(
                    label1,
                    float(vals1.mean()),
                    float(vals1.std(ddof=1)) if len(vals1) > 1 else 0.0,
                    label2,
                    float(vals2.mean()),
                    float(vals2.std(ddof=1)) if len(vals2) > 1 else 0.0,
                    float(stat),
                    float(p_value),
                )
            )
            metric_rows_added += 1

        if metric_rows_added == 0:
            lines.append(r"        \multicolumn{3}{l}{No comparable rows available.} \\")

        if metric_index < len(metrics) - 1:
            lines.append(r"        \\[\dimexpr-\normalbaselineskip+6pt]")
            lines.append(r"        \hline \\[\dimexpr-\normalbaselineskip+2pt]")

    lines.extend(
        [
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-dataset matched-10 Welch t-test LaTeX tables from run-level progress files."
    )
    parser.add_argument("--kind", choices=["dag", "cpdag"], default="cpdag")
    parser.add_argument("--datasets", nargs="+", default=["asia", "cancer", "earthquake", "survey", "sachs", "child"])
    parser.add_argument("--metrics", nargs="+", default=["sid_low", "sid_high"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ABAPC (bb-nor)", "FGS", "MPC", "NOTEARS-MLP", "Random"],
        help="Pretty-name methods to compare, in the order used for pairwise rows.",
    )
    parser.add_argument(
        "--method-alias",
        nargs="*",
        default=["ABAPC (bb-nor)=APC", "NOTEARS-MLP=NT", "Random=RND"],
        help="Optional Pretty Name=Alias mappings used in the LaTeX rows.",
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "tables" / "matched10_ttests"))
    parser.add_argument("--random-version", default="bnlearn_random_matched10_gsq_graphmetrics")
    parser.add_argument("--fgs-version", default="bnlearn_fgs_matched10_gsq_graphmetrics")
    parser.add_argument("--random-fgs-version", default=None)
    parser.add_argument("--nt-version", default="bnlearn_nt_matched10_gsq_graphmetrics")
    parser.add_argument("--mpc-version", default="bnlearn_mpc_matched10_gsq_graphmetrics")
    parser.add_argument("--abapc-orig-others-version", default="bnlearn_abapc_orig_matched10_others_gsq")
    parser.add_argument("--abapc-orig-child-version", default="child_abapc_orig_matched10_gsq")
    parser.add_argument("--abapc-nor-others-version", default="bnlearn_abapc_nor_matched10_others_gsq_graphmetrics")
    parser.add_argument("--abapc-nor-child-version", default="child_abapc_nor_matched10_gsq_graphmetrics")
    parser.add_argument("--abapc-bb-others-version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-child-version", default="child_abapc_bb_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-nor-others-version", default="bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-nor-child-version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument("--include-orig", action="store_true")
    args = parser.parse_args()

    args.random_version = _resolve_version(
        args.random_version or args.random_fgs_version,
        "bnlearn_random_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    args.fgs_version = _resolve_version(
        args.fgs_version or args.random_fgs_version,
        "bnlearn_fgs_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    args.nt_version = _resolve_version(
        args.nt_version,
        "bnlearn_nt_matched10_gsq_graphmetrics",
        "bnlearn_nt_matched10_gsq",
    )
    args.mpc_version = _resolve_version(
        args.mpc_version,
        "bnlearn_mpc_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )

    aliases = _parse_aliases(args.method_alias)
    datasets = [d.lower() for d in args.datasets]
    metrics = [m.lower() for m in args.metrics]
    kind = args.kind

    run_specs = _build_run_specs(args)
    all_rows = _load_combined_progress(kind, run_specs)
    if all_rows.empty:
        raise SystemExit("No run-level matched-10 progress rows were found for the requested versions.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_parts: list[str] = []
    for dataset in DATASET_ORDER:
        if dataset not in datasets:
            continue
        tex = _render_dataset_table(all_rows, dataset, metrics, args.methods, aliases)
        out_path = out_dir / f"{dataset}_tests.tex"
        out_path.write_text(tex, encoding="utf-8")
        combined_parts.append(tex)
        print(f"Wrote {out_path}")

    combined_path = out_dir / "all_datasets_tests.tex"
    combined_path.write_text("\n".join(combined_parts), encoding="utf-8")
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
