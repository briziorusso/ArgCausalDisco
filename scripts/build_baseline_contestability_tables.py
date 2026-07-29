#!/usr/bin/env python3
"""Build the empirical contestability comparison tables and summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "tables" / "paper_final_matched_50rep" / "baseline_contestability"
)
DEFAULT_MPC_SUMMARY = DEFAULT_OUTPUT_DIR / "mpc_enforcement_summary.csv"
DEFAULT_ASPCR_DIR = DEFAULT_OUTPUT_DIR / "aspcr_native"
DEFAULT_ASPCR_SUMMARY = DEFAULT_ASPCR_DIR / "aspcr_contestability_summary.csv"
DEFAULT_ASPCR_CHALLENGES = DEFAULT_ASPCR_DIR / "aspcr_contestability_challenges.csv"
DEFAULT_ASPCR_MANIFEST = DEFAULT_ASPCR_DIR / "aspcr_contestability_manifest.json"
DEFAULT_OPTABA_MANIFEST = (
    REPO_ROOT
    / "results"
    / "tables"
    / "paper_current_alpha001_nowrong_noweight_50rep_preview"
    / "contestability_manifest.json"
)
DEFAULT_OPTABA_SUMMARY = (
    REPO_ROOT
    / "results"
    / "tables"
    / "paper_current_alpha001_nowrong_noweight_50rep_preview"
    / "contestability_dataset_summary.csv"
)
DEFAULT_OPTABA_CHALLENGES = (
    REPO_ROOT
    / "results"
    / "tables"
    / "paper_current_alpha001_nowrong_noweight_50rep_preview"
    / "contestability_challenges.csv"
)

DATASET_ORDER = ("cancer", "earthquake", "survey", "asia", "er5", "er8", "sf5", "sf8")
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
LABEL_TO_DATASET = {label: dataset for dataset, label in DATASET_LABELS.items()}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "--" if number is None else f"{number:.{digits}f}"


def _count_prop(count: Any, total: Any) -> str:
    count_number = _number(count)
    total_number = _number(total)
    if count_number is None or total_number is None or total_number == 0:
        return "--"
    return f"{int(count_number)}/{int(total_number)} ({100 * count_number / total_number:.0f}\\%)"


def _zero_prop(row: pd.Series) -> str:
    return _count_prop(row.get("zero_margin_challenges"), row.get("proved_challenges"))


def _median_iqr(row: pd.Series) -> str:
    median = _number(row.get("normalized_margin_median"))
    q1 = _number(row.get("normalized_margin_q1"))
    q3 = _number(row.get("normalized_margin_q3"))
    if median is None or q1 is None or q3 is None:
        return "--"
    return f"{median:.3f} [{q1:.3f},{q3:.3f}]"


def render_mpc_table(summary: pd.DataFrame) -> str:
    rows = {str(row.dataset): row for row in summary.itertuples(index=False)}
    caption = r"\caption{Empirical enforcement gap of Majority-PC (MPC). Full enforcement means that the returned CPDAG's Markov-equivalence class satisfies every outcome in the complete recorded matched G$^2$ trace. Invalid endpoint outputs count as failures of full enforcement; contradiction counts and rates are averaged only over structurally valid outputs. Indep./dep. split the contradicted tests by their recorded relation. This is an output-consistency audit, not an optimisation margin.}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Outputs (valid) & Full enforcement & Mean contrad. & Mean rate & Indep. & Dep. \\",
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        row = rows.get(dataset)
        if row is None:
            continue
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS[dataset],
                    f"{int(row.outputs)} ({int(row.valid_outputs)})",
                    _count_prop(row.fully_enforcing_outputs, row.outputs),
                    _fmt(row.contradicted_tests_mean_valid),
                    f"{100 * float(row.contradiction_rate_mean_valid):.1f}\\%",
                    _fmt(row.contradicted_independence_mean_valid),
                    _fmt(row.contradicted_dependence_mean_valid),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            caption,
            r"\label{tab:mpc-enforcement-gap}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _keyed_summary(frame: pd.DataFrame) -> dict[str, pd.Series]:
    keyed: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        raw = str(row.get("dataset", ""))
        dataset = raw if raw in DATASET_LABELS else LABEL_TO_DATASET.get(raw)
        if dataset:
            keyed[dataset] = row
    return keyed


def render_repair_table(optaba: pd.DataFrame, aspcr: pd.DataFrame) -> str:
    opt_rows = _keyed_summary(optaba)
    asp_rows = _keyed_summary(aspcr)
    datasets = [
        dataset for dataset in DATASET_ORDER if dataset in opt_rows or dataset in asp_rows
    ]
    if not any(dataset in opt_rows and dataset in asp_rows for dataset in datasets):
        raise RuntimeError("OptABA-PC and ASPCR summaries have no overlapping datasets")
    caption = r"\caption{Dataset-level comparison of forced-fact responses. Baselines are analysable seeds; each target is one fact released by OptABA-PC or failed by ASPCR-DAG and made mandatory in a hard-retention re-solve. For OptABA-PC, each re-solve instantiates a user contestation and $\mu(f)$ empirically instantiates the proved contestability threshold. For ASPCR-DAG, each re-solve is a post-hoc sensitivity probe and $\mu_{\mathrm{ASP}}(f)$ is a sensitivity gap, not evidence that the method offers redress. Zero $\mu$/gap counts proved re-solves for which hard retention does not increase the optimum. The normalised quantities are $\mu(f)/w_f$ and $\mu_{\mathrm{ASP}}(f)/w_f$; $Q_1$ and $Q_3$ denote their first and third quartiles. The last column gives infeasible targets/timeouts. ASPCR-DAG uses its exhaustive native Bayesian log-weighted trace, whereas OptABA-PC uses the matched adaptive G$^2$ trace and corresponding release weights, so target counts and normalised values are not paired cross-method effect sizes.}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrl}",
        r"\toprule",
        r"Dataset & Method & Baselines & Targets & Proved & Zero $\mu$/gap & Normalised $\mu$/gap median [$Q_1,Q_3$] & Infeas./timeout \\",
        r"\midrule",
    ]
    for dataset in datasets:
        method_rows = [
            ("OptABA-PC", opt_rows.get(dataset)),
            ("ASPCR-DAG", asp_rows.get(dataset)),
        ]
        first = True
        for method, row in method_rows:
            if row is None:
                continue
            label = DATASET_LABELS[dataset] if first else ""
            first = False
            lines.append(
                f"{label} & {method} & {int(row['analyzable_seeds'])} & "
                f"{int(row['challenges'])} & {int(row['proved_challenges'])} & "
                f"{_zero_prop(row)} & {_median_iqr(row)} & "
                f"{int(row['infeasible_challenges'])}/{int(row['timeouts'])} \\\\"
            )
        if dataset != datasets[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            caption,
            r"\label{tab:contestability-margin-comparison}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_abapc_optimality_summary(optaba_manifest: dict[str, Any]) -> pd.DataFrame:
    """Compare ABA-PC with each independently proved OptABA-PC baseline optimum."""
    records: list[dict[str, Any]] = []
    summary_cache: dict[Path, dict[str, Any]] = {}
    for instance in optaba_manifest.get("baseline_instances", []):
        if not instance.get("analyzable") or not instance.get("optimality_proven"):
            continue
        dataset = str(instance.get("dataset_key", ""))
        if dataset not in DATASET_LABELS:
            continue
        summary_path = _resolve(str(instance["summary_path"]))
        if summary_path not in summary_cache:
            summary_cache[summary_path] = json.loads(summary_path.read_text())
        summary = summary_cache[summary_path]
        rep = int(instance["rep"])
        candidates = [row for row in summary.get("baseline_results", []) if int(row.get("rep", -1)) == rep]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ABA-PC result for {dataset} rep {rep} in {summary_path}")
        baseline = candidates[0]
        total_weight = int(baseline["weight_total"])
        aba_release_cost = total_weight - int(baseline["weight_accepted"])
        opt_release_cost = int(instance["expected_baseline_cost"])
        if aba_release_cost < opt_release_cost:
            raise RuntimeError(
                f"ABA-PC cost {aba_release_cost} is below proved optimum {opt_release_cost} "
                f"for {dataset} rep {rep}"
            )
        relative_reduction = (
            (aba_release_cost - opt_release_cost) / aba_release_cost
            if aba_release_cost > 0
            else 0.0
        )
        records.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "seed": int(instance["seed"]),
                "aba_release_cost": aba_release_cost,
                "optaba_release_cost": opt_release_cost,
                "absolute_cost_reduction": aba_release_cost - opt_release_cost,
                "relative_cost_reduction": relative_reduction,
                "aba_strictly_suboptimal": aba_release_cost > opt_release_cost,
            }
        )
    if not records:
        raise RuntimeError("OptABA-PC manifest contains no proved baseline instances")
    raw = pd.DataFrame.from_records(records)
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        group = raw[raw["dataset"] == dataset]
        if group.empty:
            continue
        reduction = group["relative_cost_reduction"]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "proved_pairs": len(group),
                "aba_strictly_suboptimal": int(group["aba_strictly_suboptimal"].sum()),
                "full_trace_enforced": int((group["optaba_release_cost"] == 0).sum()),
                "relative_cost_reduction_median": float(reduction.median()),
                "relative_cost_reduction_q1": float(reduction.quantile(0.25)),
                "relative_cost_reduction_q3": float(reduction.quantile(0.75)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _pooled_margin(frame: pd.DataFrame) -> str:
    values = pd.to_numeric(frame["normalized_margin"], errors="coerce").dropna()
    if values.empty:
        return "--"
    return f"{values.median():.3f} [{values.quantile(0.25):.3f},{values.quantile(0.75):.3f}]"


def render_method_overview(
    abapc: pd.DataFrame,
    optaba: pd.DataFrame,
    optaba_challenges: pd.DataFrame,
    aspcr: pd.DataFrame,
    aspcr_challenges: pd.DataFrame,
    mpc: pd.DataFrame,
) -> str:
    aba_pairs = int(abapc["proved_pairs"].sum())
    aba_suboptimal = int(abapc["aba_strictly_suboptimal"].sum())
    repaired_full_trace = int(abapc["full_trace_enforced"].sum())
    opt_challenges = int(pd.to_numeric(optaba["proved_challenges"], errors="coerce").sum())
    opt_zero = int(pd.to_numeric(optaba["zero_margin_challenges"], errors="coerce").sum())
    asp_challenges = int(pd.to_numeric(aspcr["proved_challenges"], errors="coerce").sum())
    asp_zero = int(pd.to_numeric(aspcr["zero_margin_challenges"], errors="coerce").sum())
    mpc_outputs = int(pd.to_numeric(mpc["outputs"], errors="coerce").sum())
    mpc_full = int(pd.to_numeric(mpc["fully_enforcing_outputs"], errors="coerce").sum())
    asp_baselines = int(pd.to_numeric(aspcr["analyzable_seeds"], errors="coerce").sum())
    # A baseline enforces its complete native trace iff it produces no
    # failed fact targeted by a sensitivity probe.
    asp_failed_by_seed = (
        aspcr_challenges.groupby(["dataset", "seed"], dropna=False).size()
        if not aspcr_challenges.empty
        else pd.Series(dtype=int)
    )
    asp_full_trace = asp_baselines - int(len(asp_failed_by_seed))
    caption = r"\caption{Comparison of four empirical properties: satisfaction of the complete input trace, satisfaction of the subset declared retained (or not failed), objective optimality, and the response of the optimised objective after one rejected fact is made mandatory. \emph{Full-trace compatibility} permits no exclusions as released or failed; \emph{declared set} tests the method's stated post-repair commitments. ABA-PC and OptABA-PC share the 250 proved-audit G$^2$ instances, so their objective comparison is paired: ABA-PC matches the proved minimum on 142 and OptABA-PC is strictly cheaper on the other 108. MPC is audited on all 400 matched G$^2$ outputs; its bold compatibility rate is called out for discussion and does not denote a winner. ASPCR-DAG uses the same saved samples on three datasets but a different exhaustive Bayesian trace, log weights, and single-DAG graph-disagreement objective. Each hard-retention re-solve targets one rejected fact. For OptABA-PC, the re-solve instantiates a user contestation and $\mu(f)$ is the proved contestability response; for ASPCR-DAG, $\mu_{\mathrm{ASP}}(f)$ is a post-hoc sensitivity gap, not a method-level redress guarantee. Zero $\mu$/gap counts proved re-solves for which hard retention does not increase the optimum. The normalised quantities are $\mu(f)/w_f$ and $\mu_{\mathrm{ASP}}(f)/w_f$; $Q_1$ and $Q_3$ denote their first and third quartiles. The two methods' target counts and normalised values are not paired or numerically commensurate cross-method effects.}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Method & Full-trace compat. & Declared set & Objective evidence & Hard re-solves & Zero $\mu$/gap & Normalised $\mu$/gap median [$Q_1,Q_3$] \\",
        r"\midrule",
        f"ABA-PC & {_count_prop(repaired_full_trace, aba_pairs)} & "
        f"{_count_prop(aba_pairs, aba_pairs)} & Opt lower {_count_prop(aba_suboptimal, aba_pairs)} "
        f"& -- & -- & --" + r" \\",
        f"MPC & \\textbf{{{_count_prop(mpc_full, mpc_outputs)}}} & -- & -- & -- & -- & --" + r" \\",
        f"ASPCR-DAG & {_count_prop(asp_full_trace, asp_baselines)} & "
        f"{_count_prop(asp_baselines, asp_baselines)} & min. {_count_prop(asp_baselines, asp_baselines)} "
        f"& {_count_prop(asp_challenges, asp_challenges)} & "
        f"{_count_prop(asp_zero, asp_challenges)} & {_pooled_margin(aspcr_challenges)}" + r" \\",
        f"OptABA-PC & {_count_prop(repaired_full_trace, aba_pairs)} & "
        f"{_count_prop(aba_pairs, aba_pairs)} & min. {_count_prop(aba_pairs, aba_pairs)} "
        f"& {_count_prop(opt_challenges, opt_challenges)} & "
        f"{_count_prop(opt_zero, opt_challenges)} & {_pooled_margin(optaba_challenges)}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:contestability-overview}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mpc-summary", default=str(DEFAULT_MPC_SUMMARY))
    parser.add_argument("--aspcr-summary", default=str(DEFAULT_ASPCR_SUMMARY))
    parser.add_argument("--aspcr-challenges", default=str(DEFAULT_ASPCR_CHALLENGES))
    parser.add_argument("--aspcr-manifest", default=str(DEFAULT_ASPCR_MANIFEST))
    parser.add_argument("--optaba-summary", default=str(DEFAULT_OPTABA_SUMMARY))
    parser.add_argument("--optaba-challenges", default=str(DEFAULT_OPTABA_CHALLENGES))
    parser.add_argument("--optaba-manifest", default=str(DEFAULT_OPTABA_MANIFEST))
    parser.add_argument("--allow-partial-aspcr", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    mpc_path = _resolve(args.mpc_summary)
    aspcr_path = _resolve(args.aspcr_summary)
    aspcr_challenges_path = _resolve(args.aspcr_challenges)
    aspcr_manifest_path = _resolve(args.aspcr_manifest)
    optaba_path = _resolve(args.optaba_summary)
    optaba_challenges_path = _resolve(args.optaba_challenges)
    optaba_manifest_path = _resolve(args.optaba_manifest)
    for path in (
        mpc_path,
        aspcr_path,
        aspcr_challenges_path,
        aspcr_manifest_path,
        optaba_path,
        optaba_challenges_path,
        optaba_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    aspcr_manifest = json.loads(aspcr_manifest_path.read_text())
    optaba_manifest = json.loads(optaba_manifest_path.read_text())
    if aspcr_manifest.get("status") != "complete" and not args.allow_partial_aspcr:
        raise RuntimeError(
            f"ASPCR contestability manifest is {aspcr_manifest.get('status')!r}; "
            "finish it or pass --allow-partial-aspcr for a diagnostic table"
        )

    mpc = pd.read_csv(mpc_path)
    aspcr = pd.read_csv(aspcr_path)
    aspcr_challenges = pd.read_csv(aspcr_challenges_path)
    optaba = pd.read_csv(optaba_path)
    optaba_challenges = pd.read_csv(optaba_challenges_path)
    abapc = build_abapc_optimality_summary(optaba_manifest)
    mpc_tex = output_dir / "table_mpc_enforcement_gap.tex"
    repair_tex = output_dir / "table_contestability_margin_comparison.tex"
    overview_tex = output_dir / "table_contestability_method_overview.tex"
    abapc_summary = output_dir / "abapc_optimality_summary.csv"
    _atomic_text(mpc_tex, render_mpc_table(mpc))
    _atomic_text(repair_tex, render_repair_table(optaba, aspcr))
    _atomic_text(
        overview_tex,
        render_method_overview(
            abapc, optaba, optaba_challenges, aspcr, aspcr_challenges, mpc
        ),
    )
    _atomic_text(abapc_summary, abapc.to_csv(index=False))
    manifest_path = output_dir / "baseline_contestability_table_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "aspcr_status": aspcr_manifest.get("status"),
            "allow_partial_aspcr": bool(args.allow_partial_aspcr),
            "inputs": {
                "mpc": {"path": str(mpc_path), "sha256": _sha256(mpc_path)},
                "aspcr": {"path": str(aspcr_path), "sha256": _sha256(aspcr_path)},
                "aspcr_challenges": {
                    "path": str(aspcr_challenges_path), "sha256": _sha256(aspcr_challenges_path)
                },
                "optaba": {"path": str(optaba_path), "sha256": _sha256(optaba_path)},
                "optaba_challenges": {
                    "path": str(optaba_challenges_path), "sha256": _sha256(optaba_challenges_path)
                },
                "optaba_manifest": {
                    "path": str(optaba_manifest_path), "sha256": _sha256(optaba_manifest_path)
                },
            },
            "outputs": {
                "mpc_table": {"path": str(mpc_tex), "sha256": _sha256(mpc_tex)},
                "repair_table": {"path": str(repair_tex), "sha256": _sha256(repair_tex)},
                "overview_table": {"path": str(overview_tex), "sha256": _sha256(overview_tex)},
                "abapc_summary": {"path": str(abapc_summary), "sha256": _sha256(abapc_summary)},
            },
        },
    )
    print(f"Wrote MPC enforcement table: {mpc_tex}")
    print(f"Wrote dataset-level margin comparison: {repair_tex}")
    print(f"Wrote contestability overview table: {overview_tex}")
    print(f"Wrote ABA-PC optimality summary: {abapc_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
