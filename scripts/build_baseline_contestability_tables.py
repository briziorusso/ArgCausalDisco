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
DEFAULT_FINAL_FACT_RECORDS = DEFAULT_OUTPUT_DIR.parent / "final_fact_records.csv"
DEFAULT_FINAL_GRAPH_RECORDS = DEFAULT_OUTPUT_DIR.parent / "final_graph_records.csv"
DEFAULT_MPC_SUMMARY = DEFAULT_OUTPUT_DIR / "mpc_matched_250" / "mpc_enforcement_summary.csv"
DEFAULT_MPC_MATCHED_AUDIT = DEFAULT_OUTPUT_DIR / "mpc_matched_250" / "mpc_enforcement_audit.csv"
DEFAULT_MPC_PRIMARY_AUDIT = DEFAULT_OUTPUT_DIR / "mpc_enforcement_audit.csv"
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


def render_mpc_table(
    summary: pd.DataFrame, abapc: pd.DataFrame, primary: dict[str, Any]
) -> str:
    caption = r"\caption{Dataset-level Majority-PC enforcement audit over all 400 primary $G^2$ traces. Feasible counts are established by the exact ABA consistency check before any fact is released. Full compatibility requires the returned MPC Markov class to satisfy every input test; the avoidable gap counts feasible traces for which it does not. A dash indicates that no avoidable-gap rate exists because no trace is feasible. Invalid endpoint outputs count as incompatible.}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Inputs & Feasible traces & Full compatibility & Avoidable gap \\",
        r"\midrule",
    ]
    for item in primary["per_dataset"]:
        dataset = str(item["dataset"])
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS[dataset],
                    str(int(item["inputs"])),
                    _count_prop(item["feasible"], item["inputs"]),
                    _count_prop(item["mpc_compatible"], item["inputs"]),
                    _count_prop(item["avoidable_gaps"], item["feasible"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            " & ".join(
                [
                    r"\textbf{Total}",
                    str(int(primary["inputs"])),
                    _count_prop(primary["feasible"], primary["inputs"]),
                    _count_prop(primary["mpc_compatible"], primary["inputs"]),
                    _count_prop(primary["avoidable_gaps"], primary["feasible"]),
                ]
            )
            + r" \\",
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


def render_repair_table(
    abapc: pd.DataFrame,
    optaba_challenges: pd.DataFrame,
    aspcr: pd.DataFrame,
    aspcr_challenges: pd.DataFrame,
    mpc_matched: dict[str, int],
) -> str:
    pairs = int(abapc["proved_pairs"].sum())
    feasible = int(abapc["full_trace_enforced"].sum())
    inconsistent = pairs - feasible
    aba_higher = int(abapc["aba_strictly_suboptimal"].sum())
    aba_minimum = inconsistent - aba_higher
    opt_values = pd.to_numeric(optaba_challenges["normalized_margin"], errors="raise")
    opt_proved = int(pd.to_numeric(optaba_challenges["optimality_proven"], errors="raise").sum())
    opt_zero = int(pd.to_numeric(optaba_challenges["zero_margin"], errors="raise").sum())
    opt_near = int((opt_values <= 0.1).sum())
    opt_large = int((opt_values > 1.0).sum())
    asp_baselines = int(pd.to_numeric(aspcr["analyzable_seeds"], errors="raise").sum())
    asp_values = pd.to_numeric(aspcr_challenges["normalized_margin"], errors="raise")
    asp_proved = int(pd.to_numeric(aspcr_challenges["optimality_proven"], errors="raise").sum())
    asp_zero = int(pd.to_numeric(aspcr_challenges["zero_margin"], errors="raise").sum())
    asp_failed_seeds = int(aspcr_challenges[["dataset", "seed"]].drop_duplicates().shape[0])
    asp_full = asp_baselines - asp_failed_seeds
    mpc_feasible = int(mpc_matched["full_trace_feasible"])
    mpc_compatible = int(mpc_matched["full_trace_returned"])
    mpc_gaps = int(mpc_matched["avoidable_gaps"])
    if pairs != 250 or feasible != mpc_feasible or opt_proved != len(optaba_challenges):
        raise RuntimeError("Exact decision-audit accounting is inconsistent")
    if asp_proved != len(aspcr_challenges):
        raise RuntimeError("ASPCR sensitivity audit contains unproved targets")
    caption = (
        r"\caption{Decision and forced-fact evidence. OptABA-PC, ABA-PC, and MPC use the exact "
        r"250-trace $G^2$ audit; ASPCR-DAG uses a separate 250-run native Bayesian audit. A forced "
        r"OptABA-PC solve is a user contestation and $\mu(f)/w_f$ is its exact threshold. ASPCR-DAG's "
        r"forced solve is a post-hoc sensitivity test of a soft graph-disagreement objective, so its "
        r"$\mathrm{gap}/w_f$ is not numerically comparable. ABA-PC and MPC define neither category optima nor "
        r"a forced-fact certificate. Brackets give $[Q_1,Q_3]$; a dash means that the object is undefined. "
        r"Bold marks OptABA-PC's exact certified result.}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{4.0cm}>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{3.1cm}@{}}",
        r"\toprule",
        r"Method & Scope & Baseline decision evidence & Forced-fact evidence & Certificate supplied \\",
        r"\midrule",
        rf"OptABA-PC & $G^2$, {pairs} & Minimum repair on \textbf{{{inconsistent}/{inconsistent}}} inconsistent traces; lower cost than ABA-PC on \textbf{{{aba_higher}}}, tied on {aba_minimum}. & "
        rf"\textbf{{{opt_proved}/{len(optaba_challenges)}}} proved; zero gaps {opt_zero}/{len(optaba_challenges)}; median $\mu/w_f={opt_values.median():.3f}$ [{opt_values.quantile(0.25):.3f},{opt_values.quantile(0.75):.3f}]. {100.0 * opt_near / len(opt_values):.1f}\% are $\leq0.1$; {100.0 * opt_large / len(opt_values):.1f}\% are $>1$. & \textbf{{Exact user-contestation threshold.}} \\",
        rf"ABA-PC & $G^2$, {pairs} & Selected repair enforced on {pairs}/{pairs}; minimum cost on only {aba_minimum}/{inconsistent} inconsistent traces ({aba_higher} higher). & -- & Heuristic repair; no exact response threshold. \\",
        rf"MPC & $G^2$, {pairs} & Compatible on {mpc_compatible}/{mpc_feasible} feasible traces; {mpc_gaps} avoidable gaps; no selected repair. & -- & No repair or contestation object. \\",
        rf"ASPCR-DAG & native, {asp_baselines} & Full trace on {asp_full}/{asp_baselines}; graph optimum proved on {asp_baselines}/{asp_baselines}. & "
        rf"{asp_proved}/{len(aspcr_challenges)} proved; zero gaps {asp_zero}/{len(aspcr_challenges)}; median $\mathrm{{gap}}/w_f={asp_values.median():.3f}$ [{asp_values.quantile(0.25):.3f},{asp_values.quantile(0.75):.3f}]. & Post-hoc graph-objective sensitivity. \\",
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:contestability-margin-comparison}",
        r"\end{table*}",
        "",
    ]
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
        improved = reduction[group["aba_strictly_suboptimal"]]
        feasible = int((group["optaba_release_cost"] == 0).sum())
        inconsistent = int(len(group) - feasible)
        lower = int(group["aba_strictly_suboptimal"].sum())
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "proved_pairs": len(group),
                "aba_strictly_suboptimal": lower,
                "full_trace_enforced": feasible,
                "inconsistent_traces": inconsistent,
                "optaba_lower_cost": lower,
                "equal_cost_inconsistent": inconsistent - lower,
                "relative_cost_reduction_median": float(reduction.median()),
                "relative_cost_reduction_q1": float(reduction.quantile(0.25)),
                "relative_cost_reduction_q3": float(reduction.quantile(0.75)),
                "improved_relative_cost_reduction_median": (
                    float(improved.median()) if len(improved) else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _pooled_margin(frame: pd.DataFrame) -> str:
    values = pd.to_numeric(frame["normalized_margin"], errors="coerce").dropna()
    if values.empty:
        return "--"
    return f"{values.median():.3f} [{values.quantile(0.25):.3f},{values.quantile(0.75):.3f}]"


def render_primary_repair_enforcement_table(primary: dict[str, Any]) -> str:
    """Merge repair-cost and MPC-enforcement evidence over all 400 inputs."""

    inputs = int(primary["inputs"])
    feasible = int(primary["feasible"])
    inconsistent = int(primary["inconsistent"])
    lower = int(primary["opt_lower_cost"])
    equal = int(primary["same_cost_inconsistent"])
    compatible = int(primary["mpc_compatible"])
    gaps = int(primary["avoidable_gaps"])
    if (inputs, feasible, inconsistent, lower, equal, compatible, gaps) != (
        400, 88, 312, 294, 18, 80, 8
    ):
        raise RuntimeError("Primary repair/enforcement table accounting is inconsistent")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& Input status & \multicolumn{3}{c}{Saved OptABA-PC repair vs. ABA-PC} & \multicolumn{2}{c}{MPC output} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(l){6-7}",
        r"Dataset & Feasible / inputs & Lower / inconsistent & Same cost & Median saving & Full trace / inputs & Gap / feasible \\",
        r"\midrule",
    ]
    for row in primary["per_dataset"]:
        n_inconsistent = int(row["inconsistent"])
        saving_value = row["improved_relative_cost_reduction_median"]
        saving = (
            f"{100.0 * float(saving_value):.1f}\\%" if pd.notna(saving_value) else "--"
        )
        lower_text = (
            f"{int(row['opt_lower_cost'])}/{n_inconsistent}" if n_inconsistent else "--"
        )
        same_text = str(int(row["same_cost_inconsistent"])) if n_inconsistent else "--"
        feasible_n = int(row["feasible"])
        gap_text = (
            f"{int(row['avoidable_gaps'])}/{feasible_n}" if feasible_n else "--"
        )
        lines.append(
            f"{DATASET_LABELS[str(row['dataset'])]} & {feasible_n}/{int(row['inputs'])} & "
            f"{lower_text} & {same_text} & {saving} & "
            f"{int(row['mpc_compatible'])}/{int(row['inputs'])} & {gap_text} \\\\"
        )
    total_saving = 100.0 * float(primary["improved_relative_cost_reduction_median"])
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {feasible}/{inputs} & \textbf{{{lower}/{inconsistent}}} & {equal} & {total_saving:.1f}\% & {compatible}/{inputs} & \textbf{{{gaps}/{feasible}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Repair quality and MPC enforcement on the same 400 primary $G^2$ traces. Feasible inputs admit a DAG satisfying the entire trace; all other inputs require a repair. \emph{Lower/inconsistent} counts saved OptABA-PC repairs with lower release cost than ABA-PC among the inconsistent inputs, and \emph{same cost} is the complementary count; ABA-PC is never cheaper. Median saving is conditional on a lower saved cost. Thirty-two OptABA-PC rows are valid timeout incumbents, so this full-coverage comparison is descriptive; Table~\ref{tab:contestability-margin-comparison} separately gives the 250-instance exact audit with proved optima. MPC \emph{full trace} counts returned classes satisfying every input test, and \emph{gap} counts feasible traces that MPC nevertheless fails to satisfy.}",
            r"\label{tab:primary-repair-enforcement}",
            r"\label{tab:abapc-optimality-by-dataset}",
            r"\label{tab:mpc-enforcement-gap}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_margin_by_dataset_table(
    optaba: pd.DataFrame,
    optaba_challenges: pd.DataFrame,
    aspcr: pd.DataFrame,
    aspcr_challenges: pd.DataFrame,
) -> str:
    """Restore the dataset-level OptABA-PC/ASPCR forced-fact margin view."""

    opt_rows = _keyed_summary(optaba)
    asp_rows = _keyed_summary(aspcr)
    datasets = ("cancer", "earthquake", "survey", "er5", "sf5")
    if any(dataset not in opt_rows or dataset not in asp_rows for dataset in datasets):
        raise RuntimeError("Margin table requires five validated datasets for both methods")
    opt_total = int(pd.to_numeric(optaba["analyzable_seeds"], errors="raise").sum())
    asp_total = int(pd.to_numeric(aspcr["analyzable_seeds"], errors="raise").sum())
    if opt_total != 250 or asp_total != 250:
        raise RuntimeError("Margin table requires 250 baselines per method")

    def repair_relative(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "--"
        margin = pd.to_numeric(frame["margin"], errors="raise")
        baseline = pd.to_numeric(frame["baseline_optimum_cost"], errors="raise")
        if (baseline <= 0).any():
            raise RuntimeError("Non-positive OptABA-PC baseline cost in margin table")
        return f"{(margin / baseline).median():.3f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{OptABA-PC contestations} & \multicolumn{2}{c}{ASPCR-DAG sensitivity probes} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
        r"Dataset & Proved & Median $\mu/w_f$ $[Q_1,Q_3]$ & Median $\mu/c^*$ & Proved & Median $\mu_{\rm ASP}/w_f$ $[Q_1,Q_3]$ \\",
        r"\midrule",
    ]
    for index, dataset in enumerate(datasets):
        opt_row = opt_rows[dataset]
        asp_row = asp_rows[dataset]
        opt_challenges = int(opt_row["challenges"])
        asp_challenges = int(asp_row["challenges"])
        opt_dataset_challenges = optaba_challenges[
            optaba_challenges["dataset"].astype(str).isin(
                (dataset, DATASET_LABELS[dataset])
            )
        ]
        lines.append(
            f"{DATASET_LABELS[dataset]} & "
            f"{int(opt_row['proved_challenges'])}/{opt_challenges} & {_median_iqr(opt_row)} & "
            f"{repair_relative(opt_dataset_challenges)} & "
            f"{int(asp_row['proved_challenges'])}/{asp_challenges} & {_median_iqr(asp_row)} \\\\"
        )
    opt_challenges_n = len(optaba_challenges)
    asp_challenges_n = len(aspcr_challenges)
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {opt_challenges_n}/{opt_challenges_n} & {_pooled_margin(optaba_challenges)} & {repair_relative(optaba_challenges)} & {asp_challenges_n}/{asp_challenges_n} & {_pooled_margin(aspcr_challenges)} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Dataset-level forced-fact evidence over 250 validated baselines per method (50 per dataset). Each target is a fact rejected by the baseline optimum; all re-solves prove optimality and none has zero gap. OptABA-PC's $\mu(f)/w_f$ is an exact user-contestation threshold; $c^*$ denotes the complete baseline optimum repair cost. ASPCR-DAG's $\mu_{\rm ASP}(f)/w_f$ is instead a post-hoc sensitivity gap for its soft graph-disagreement objective. These are not paired effect sizes: the methods use different traces, weights, and objectives, and their synthetic audits use different graph densities.}",
            r"\label{tab:margin-by-dataset}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_margin_insights(
    optaba_challenges: pd.DataFrame, aspcr_challenges: pd.DataFrame
) -> pd.DataFrame:
    """Export the pooled quantities used to interpret Table 6 margins."""

    rows: list[dict[str, Any]] = []
    for method, frame in (
        ("OptABA-PC", optaba_challenges),
        ("ASPCR-DAG sensitivity", aspcr_challenges),
    ):
        normalized = pd.to_numeric(frame["normalized_margin"], errors="raise")
        margin = pd.to_numeric(frame["margin"], errors="raise")
        baseline = pd.to_numeric(frame["baseline_optimum_cost"], errors="raise")
        if (baseline <= 0).any():
            raise RuntimeError(f"Non-positive baseline cost in {method} margin audit")
        repair_relative = margin / baseline
        row = {
            "method": method,
            "targets": len(frame),
            "zero_gaps": int(pd.to_numeric(frame["zero_margin"], errors="raise").sum()),
            "normalized_median": float(normalized.median()),
            "normalized_q1": float(normalized.quantile(0.25)),
            "normalized_q3": float(normalized.quantile(0.75)),
            "normalized_min": float(normalized.min()),
            "normalized_max": float(normalized.max()),
            "within_10pct_weight": int((normalized <= 0.1).sum()),
            "within_10pct_weight_proportion": float((normalized <= 0.1).mean()),
            "more_than_double_weight": int((normalized > 1.0).sum()),
            "more_than_double_weight_proportion": float((normalized > 1.0).mean()),
            "repair_relative_median": float(repair_relative.median()),
            "repair_relative_q1": float(repair_relative.quantile(0.25)),
            "repair_relative_q3": float(repair_relative.quantile(0.75)),
        }
        if method.startswith("ASPCR"):
            row["other_penalty_shift_median_weight_units"] = 1.0 + row[
                "normalized_median"
            ]
        else:
            row["other_penalty_shift_median_weight_units"] = math.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_aspcr_trace_accounting(manifest: dict[str, Any]) -> pd.DataFrame:
    """Count native soft tests satisfied and failed by each saved ASPCR DAG."""

    artifacts = manifest.get("inputs", {}).get("artifacts", [])
    if not artifacts:
        raise RuntimeError("ASPCR manifest contains no constraint artifacts")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for artifact in artifacts:
        dataset = str(artifact["dataset"])
        seed = int(artifact["seed"])
        identity = (dataset, seed)
        if identity in seen:
            raise RuntimeError(f"Duplicate ASPCR constraint artifact: {identity}")
        seen.add(identity)
        constraints_path = _resolve(artifact["constraints"]["path"])
        constraints = pd.read_csv(constraints_path)
        if "constraint_id" not in constraints or "retained" not in constraints:
            raise RuntimeError(f"Incomplete ASPCR constraint trace: {constraints_path}")
        if constraints["constraint_id"].duplicated().any():
            raise RuntimeError(f"Duplicate ASPCR constraint ids: {constraints_path}")
        retained_text = constraints["retained"].astype(str).str.strip().str.lower()
        if not set(retained_text.unique()).issubset({"true", "false"}):
            raise RuntimeError(f"Invalid ASPCR retained flags: {constraints_path}")
        total = int(len(constraints))
        satisfied = int((retained_text == "true").sum())
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "trace_facts": total,
            "satisfied_facts": satisfied,
            "failed_facts": total - satisfied,
        })
    per_run = pd.DataFrame.from_records(rows)
    summary = (
        per_run.groupby("dataset", sort=False)
        .agg(
            baselines=("seed", "nunique"),
            trace_facts=("trace_facts", "sum"),
            satisfied_facts=("satisfied_facts", "sum"),
            failed_facts=("failed_facts", "sum"),
        )
        .reset_index()
    )
    summary["satisfied_proportion"] = summary["satisfied_facts"] / summary["trace_facts"]
    return summary


def build_aspcr_primary_accounting(
    fact_records_path: Path, graph_records_path: Path
) -> dict[str, int]:
    """Summarise every corrected ASPCR baseline in the primary result set."""

    facts = pd.read_csv(fact_records_path)
    graphs = pd.read_csv(graph_records_path)
    facts = facts[facts["method"] == "ASPCR-DAG"].copy()
    graphs = graphs[graphs["method"] == "ASPCR-DAG"].copy()
    if facts.empty or graphs.empty:
        raise RuntimeError("Primary result records contain no ASPCR-DAG rows")
    identity = ["dataset", "seed"]
    if facts.duplicated(identity).any() or graphs.duplicated(identity).any():
        raise RuntimeError("Duplicate ASPCR-DAG dataset/seed rows in primary records")
    fact_ids = set(map(tuple, facts[identity].itertuples(index=False, name=None)))
    graph_ids = set(map(tuple, graphs[identity].itertuples(index=False, name=None)))
    if fact_ids != graph_ids:
        raise RuntimeError("ASPCR-DAG fact and graph primary records do not align")
    total = int(pd.to_numeric(facts["total_facts"], errors="raise").sum())
    satisfied = int(pd.to_numeric(facts["accepted_facts"], errors="raise").sum())
    failed = int(pd.to_numeric(facts["removed_facts"], errors="raise").sum())
    if satisfied + failed != total:
        raise RuntimeError("Inconsistent ASPCR-DAG primary fact accounting")
    compatible = pd.to_numeric(graphs["n_cpdags_compat"], errors="raise") > 0
    return {
        "baselines": len(fact_ids),
        "full_trace": int(compatible.sum()),
        "trace_facts": total,
        "satisfied_facts": satisfied,
        "failed_facts": failed,
    }


def build_mpc_matched_accounting(
    audit_path: Path, optaba_manifest: dict[str, Any]
) -> dict[str, int]:
    """Align MPC output compatibility with the exact 250 OptABA audit traces."""

    audit = pd.read_csv(audit_path)
    audit = audit[["dataset", "seed", "all_tests_enforced"]].copy()
    if audit.duplicated(["dataset", "seed"]).any():
        raise RuntimeError("Duplicate MPC rows in the matched 250-output audit")
    baselines = pd.DataFrame(optaba_manifest.get("baseline_instances", []))
    if baselines.empty:
        raise RuntimeError("OptABA-PC manifest has no baseline instances")
    baselines["dataset"] = baselines["dataset_key"].astype(str)
    baselines["full_trace_feasible"] = (
        pd.to_numeric(baselines["expected_baseline_cost"], errors="raise") == 0
    )
    aligned = baselines[["dataset", "seed", "full_trace_feasible"]].merge(
        audit,
        on=["dataset", "seed"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(aligned) != 250 or not (aligned["_merge"] == "both").all():
        raise RuntimeError("MPC audit does not match the 250 OptABA-PC baseline identities")
    if (aligned["all_tests_enforced"] & ~aligned["full_trace_feasible"]).any():
        raise RuntimeError("MPC allegedly enforces a trace proved infeasible by OptABA-PC")
    feasible = int(aligned["full_trace_feasible"].sum())
    returned = int(aligned["all_tests_enforced"].sum())
    return {
        "baselines": len(aligned),
        "full_trace_feasible": feasible,
        "full_trace_returned": returned,
        "avoidable_gaps": feasible - returned,
    }


def build_primary_decision_accounting(
    fact_records_path: Path,
    graph_records_path: Path,
    mpc_audit_path: Path,
) -> dict[str, Any]:
    """Audit all 400 primary traces without conflating infeasibility with failure.

    ABA-PC performs an exact consistency check before releasing any fact, so a
    zero-release ABA row identifies a trace that is DAG-feasible.  OptABA-PC's
    zero-release classification agrees on all 400 recovered primary outputs.
    """

    facts = pd.read_csv(fact_records_path)
    graphs = pd.read_csv(graph_records_path)
    mpc = pd.read_csv(mpc_audit_path)
    repair_columns = ["dataset", "seed", "removed_facts", "rep_local", "source"]
    aba = facts[facts["method"] == "ABA-PC"][repair_columns].copy()
    opt = facts[facts["method"] == "OptABA-PC"][repair_columns].copy()
    mpc = mpc[["dataset", "seed", "all_tests_enforced"]].copy()
    for name, frame in (("ABA-PC", aba), ("OptABA-PC", opt), ("MPC", mpc)):
        if frame.duplicated(["dataset", "seed"]).any():
            raise RuntimeError(f"Duplicate {name} primary dataset/seed identities")
    if len(aba) != 400 or len(opt) != 400 or len(mpc) != 400:
        raise RuntimeError(
            "Primary audit requires 400 ABA/OptABA-PC/MPC rows; "
            f"got {len(aba)}/{len(opt)}/{len(mpc)}"
        )
    aba_ids = set(map(tuple, aba[["dataset", "seed"]].itertuples(index=False, name=None)))
    mpc_ids = set(map(tuple, mpc[["dataset", "seed"]].itertuples(index=False, name=None)))
    opt_ids = set(map(tuple, opt[["dataset", "seed"]].itertuples(index=False, name=None)))
    if aba_ids != opt_ids or aba_ids != mpc_ids:
        raise RuntimeError("Primary ABA, OptABA-PC, and MPC identities do not align")
    aligned_opt = aba.merge(
        opt,
        on=["dataset", "seed"],
        how="left",
        suffixes=("_aba", "_opt"),
        indicator=True,
        validate="one_to_one",
    )
    both = aligned_opt[aligned_opt["_merge"] == "both"]
    if not (
        both["removed_facts_aba"].eq(0)
        == both["removed_facts_opt"].eq(0)
    ).all():
        raise RuntimeError("ABA and OptABA-PC disagree on primary full-trace feasibility")
    # Read the selected source row for each identity.  This both distinguishes
    # completed optimisation from saved timeout incumbents and recovers the
    # exact integer release costs used by the paired 400-instance comparison.
    summary_cache: dict[Path, dict[str, Any]] = {}

    def selected_row(record: Any, method: str) -> dict[str, Any]:
        summary_path = _resolve(record.source)
        summary = summary_cache.setdefault(
            summary_path, json.loads(summary_path.read_text())
        )
        key = "baseline_results" if method == "ABA-PC" else "wc_results"
        candidates = [
            row
            for row in summary.get(key, [])
            if int(row.get("rep", -1)) == int(record.rep_local)
            and (
                method == "ABA-PC"
                or (
                    row.get("encoding") == "inc"
                    and row.get("objective") == "lex"
                    and row.get("opt_strategy") == "bb"
                    and row.get("opt_mode") == "optN"
                    and row.get("reification") == "mus"
                )
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one selected {method} row for {record.dataset}/{int(record.seed)} "
                f"in {summary_path}; found {len(candidates)}"
            )
        return candidates[0]

    opt_outcomes: list[dict[str, Any]] = []
    repair_costs: list[dict[str, Any]] = []
    aba_index = {
        (str(row.dataset), int(row.seed)): row for row in aba.itertuples(index=False)
    }
    for opt_record in opt.itertuples(index=False):
        identity = (str(opt_record.dataset), int(opt_record.seed))
        aba_record = aba_index[identity]
        aba_row = selected_row(aba_record, "ABA-PC")
        opt_row = selected_row(opt_record, "OptABA-PC")
        timed_out = bool(opt_row.get("timed_out", False))
        opt_outcomes.append(
            {
                "dataset": identity[0],
                "seed": identity[1],
                "completed": not timed_out,
                "timed_out": timed_out,
                "skipped": False,
            }
        )
        aba_cost = int(aba_row["weight_total"]) - int(aba_row["weight_accepted"])
        opt_cost = int(opt_row["weight_total"]) - int(opt_row["weight_accepted"])
        if opt_cost > aba_cost:
            raise RuntimeError(
                f"Saved OptABA-PC cost exceeds ABA-PC for {identity}: {opt_cost}>{aba_cost}"
            )
        repair_costs.append(
            {
                "dataset": identity[0],
                "seed": identity[1],
                "aba_cost": aba_cost,
                "opt_cost": opt_cost,
                "strictly_lower": opt_cost < aba_cost,
                "relative_saving": (
                    (aba_cost - opt_cost) / aba_cost if aba_cost > 0 else 0.0
                ),
            }
        )
    outcomes = pd.DataFrame.from_records(opt_outcomes)
    if len(outcomes) != 400:
        raise RuntimeError(f"Primary audit requires 400 OptABA-PC outcomes; got {len(outcomes)}")
    opt_completed = int(outcomes["completed"].sum())
    opt_timed_out = int(outcomes["timed_out"].sum())
    opt_skipped = int(outcomes["skipped"].sum())
    if (opt_completed, opt_timed_out, opt_skipped) != (368, 32, 0):
        raise RuntimeError(
            "Unexpected OptABA-PC run outcomes: "
            f"{opt_completed} completed, {opt_timed_out} timed out, {opt_skipped} skipped"
        )

    selected = graphs[graphs["method"].isin(["ABA-PC", "OptABA-PC"])].copy()
    if selected.duplicated(["dataset", "seed", "method"]).any():
        raise RuntimeError("Duplicate primary selected-repair graph rows")
    selected["selected_set_enforced"] = (
        pd.to_numeric(selected["n_cpdags_compat"], errors="raise") > 0
    )
    for method, expected in (("ABA-PC", len(aba)), ("OptABA-PC", len(opt))):
        group = selected[selected["method"] == method]
        if len(group) != expected or not group["selected_set_enforced"].all():
            raise RuntimeError(f"Incomplete selected-set enforcement for {method}")

    audit = aba.rename(columns={"removed_facts": "aba_removed_facts"}).merge(
        mpc,
        on=["dataset", "seed"],
        validate="one_to_one",
    )
    audit["full_trace_feasible"] = audit["aba_removed_facts"].eq(0)
    audit["opt_output_available"] = list(
        map(tuple, audit[["dataset", "seed"]].itertuples(index=False, name=None))
    )
    audit["opt_output_available"] = audit["opt_output_available"].isin(opt_ids)
    if (audit["all_tests_enforced"] & ~audit["full_trace_feasible"]).any():
        raise RuntimeError("MPC allegedly enforces an infeasible primary trace")

    costs = pd.DataFrame.from_records(repair_costs)
    audit = audit.merge(costs, on=["dataset", "seed"], validate="one_to_one")
    per_dataset: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        group = audit[audit["dataset"] == dataset]
        feasible = int(group["full_trace_feasible"].sum())
        compatible = int(group["all_tests_enforced"].sum())
        inconsistent_group = group[~group["full_trace_feasible"]]
        improved = inconsistent_group[inconsistent_group["strictly_lower"]]
        lower = int(inconsistent_group["strictly_lower"].sum())
        per_dataset.append(
            {
                "dataset": dataset,
                "inputs": int(len(group)),
                "feasible": feasible,
                "mpc_compatible": compatible,
                "avoidable_gaps": feasible - compatible,
                "opt_outputs": int(group["opt_output_available"].sum()),
                "inconsistent": int(len(inconsistent_group)),
                "opt_lower_cost": lower,
                "same_cost_inconsistent": int(len(inconsistent_group) - lower),
                "improved_relative_cost_reduction_median": (
                    float(improved["relative_saving"].median())
                    if len(improved)
                    else float("nan")
                ),
            }
        )
    feasible = int(audit["full_trace_feasible"].sum())
    mpc_compatible = int(audit["all_tests_enforced"].sum())
    opt_outputs = int(audit["opt_output_available"].sum())
    inconsistent = len(audit) - feasible
    return {
        "inputs": int(len(audit)),
        "feasible": feasible,
        "inconsistent": inconsistent,
        "mpc_compatible": mpc_compatible,
        "avoidable_gaps": feasible - mpc_compatible,
        "aba_outputs": int(len(aba)),
        "opt_outputs": opt_outputs,
        "opt_missing": int(len(audit) - opt_outputs),
        "opt_completed": opt_completed,
        "opt_solver_timeouts": opt_timed_out,
        "opt_skipped": opt_skipped,
        "opt_solver_timeouts_by_dataset": {
            dataset: int(group["timed_out"].sum())
            for dataset, group in outcomes.groupby("dataset", sort=False)
        },
        "aba_inconsistent_repairs": inconsistent,
        "opt_inconsistent_repairs": inconsistent - int(len(audit) - opt_outputs),
        "opt_lower_cost": int(audit["strictly_lower"].sum()),
        "same_cost": int((~audit["strictly_lower"]).sum()),
        "same_cost_inconsistent": int(
            ((~audit["strictly_lower"]) & (~audit["full_trace_feasible"])).sum()
        ),
        "improved_relative_cost_reduction_median": float(
            audit.loc[audit["strictly_lower"], "relative_saving"].median()
        ),
        "per_dataset": per_dataset,
    }


def render_method_overview(
    abapc: pd.DataFrame,
    optaba: pd.DataFrame,
    optaba_challenges: pd.DataFrame,
    aspcr: pd.DataFrame,
    aspcr_challenges: pd.DataFrame,
    aspcr_trace: pd.DataFrame,
    aspcr_primary: dict[str, int],
    mpc_matched: dict[str, int],
    primary: dict[str, Any],
) -> str:
    primary_inputs = int(primary["inputs"])
    primary_feasible = int(primary["feasible"])
    primary_inconsistent = int(primary["inconsistent"])
    primary_mpc_compatible = int(primary["mpc_compatible"])
    primary_aba_outputs = int(primary["aba_outputs"])
    primary_opt_outputs = int(primary["opt_outputs"])
    primary_opt_repairs = int(primary["opt_inconsistent_repairs"])
    asp_baselines = int(aspcr_primary["baselines"])
    asp_full_trace = int(aspcr_primary["full_trace"])
    if (
        primary_inputs != 400
        or primary_feasible != 88
        or primary_inconsistent != 312
        or primary_mpc_compatible != 80
        or asp_baselines != 250
    ):
        raise RuntimeError("Primary evidence-overview accounting is inconsistent")
    opt_completed = int(primary["opt_completed"])
    opt_timed_out = int(primary["opt_solver_timeouts"])
    opt_skipped = int(primary["opt_skipped"])
    caption = (
        r"\caption{Evidence correspondence, run completion, and repair coverage. A trace is feasible if some DAG satisfies every recorded CI outcome; a full-trace output is the method's returned graph or class satisfying all of them. Thus MPC has 88 feasible inputs but only 80 matching outputs. A timeout incumbent is a valid saved repair whose optimality was not proved before the limit. An explicit repair names excluded facts, and selected-repair enforcement verifies every retained fact against the output. ASPCR-DAG uses its primary matched run set; FGS has no CI trace. A dash denotes an undefined object; bold marks complete audited coverage, not significance.}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{4.1cm}>{\centering\arraybackslash}p{1.7cm}>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.7cm}>{\centering\arraybackslash}p{2.2cm}@{}}",
        r"\toprule",
        r"Method & Run status & Feasible traces & Full-trace matches & Explicit repairs & Selected repair enforced \\",
        r"\midrule",
        rf"OptABA-PC & \textbf{{{primary_opt_outputs}/{primary_inputs} saved}}: {opt_completed} completed, {opt_timed_out} timeout incumbents & {primary_feasible}/{primary_inputs} & {primary_feasible}/{primary_opt_outputs} & \textbf{{{primary_opt_repairs}/{primary_inconsistent}}} & \textbf{{{primary_opt_outputs}/{primary_opt_outputs}}} \\",
        rf"ABA-PC & \textbf{{{primary_aba_outputs}/{primary_inputs} saved and completed}} & {primary_feasible}/{primary_inputs} & {primary_feasible}/{primary_inputs} & \textbf{{{primary_inconsistent}/{primary_inconsistent}}} & \textbf{{{primary_inputs}/{primary_inputs}}} \\",
        rf"ASPCR-DAG & \textbf{{{asp_baselines}/{asp_baselines} saved; all optima proved}} & {asp_full_trace}/{asp_baselines} & {asp_full_trace}/{asp_baselines} & -- & -- \\",
        rf"MPC & \textbf{{{primary_inputs}/{primary_inputs} saved and completed}} & {primary_feasible}/{primary_inputs} & {primary_mpc_compatible}/{primary_inputs} & -- & -- \\",
        rf"FGS & \textbf{{{primary_inputs}/{primary_inputs} saved and completed}} & -- & -- & -- & -- \\",
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
    parser.add_argument("--mpc-matched-audit", default=str(DEFAULT_MPC_MATCHED_AUDIT))
    parser.add_argument("--mpc-primary-audit", default=str(DEFAULT_MPC_PRIMARY_AUDIT))
    parser.add_argument("--aspcr-summary", default=str(DEFAULT_ASPCR_SUMMARY))
    parser.add_argument("--aspcr-challenges", default=str(DEFAULT_ASPCR_CHALLENGES))
    parser.add_argument("--aspcr-manifest", default=str(DEFAULT_ASPCR_MANIFEST))
    parser.add_argument("--final-fact-records", default=str(DEFAULT_FINAL_FACT_RECORDS))
    parser.add_argument("--final-graph-records", default=str(DEFAULT_FINAL_GRAPH_RECORDS))
    parser.add_argument("--optaba-summary", default=str(DEFAULT_OPTABA_SUMMARY))
    parser.add_argument("--optaba-challenges", default=str(DEFAULT_OPTABA_CHALLENGES))
    parser.add_argument("--optaba-manifest", default=str(DEFAULT_OPTABA_MANIFEST))
    parser.add_argument("--allow-partial-aspcr", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    mpc_path = _resolve(args.mpc_summary)
    mpc_matched_audit_path = _resolve(args.mpc_matched_audit)
    mpc_primary_audit_path = _resolve(args.mpc_primary_audit)
    aspcr_path = _resolve(args.aspcr_summary)
    aspcr_challenges_path = _resolve(args.aspcr_challenges)
    aspcr_manifest_path = _resolve(args.aspcr_manifest)
    final_fact_records_path = _resolve(args.final_fact_records)
    final_graph_records_path = _resolve(args.final_graph_records)
    optaba_path = _resolve(args.optaba_summary)
    optaba_challenges_path = _resolve(args.optaba_challenges)
    optaba_manifest_path = _resolve(args.optaba_manifest)
    for path in (
        mpc_path,
        mpc_matched_audit_path,
        mpc_primary_audit_path,
        aspcr_path,
        aspcr_challenges_path,
        aspcr_manifest_path,
        final_fact_records_path,
        final_graph_records_path,
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
    mpc_matched = build_mpc_matched_accounting(mpc_matched_audit_path, optaba_manifest)
    primary = build_primary_decision_accounting(
        final_fact_records_path, final_graph_records_path, mpc_primary_audit_path
    )
    aspcr = pd.read_csv(aspcr_path)
    aspcr_challenges = pd.read_csv(aspcr_challenges_path)
    aspcr_trace = build_aspcr_trace_accounting(aspcr_manifest)
    aspcr_primary = build_aspcr_primary_accounting(
        final_fact_records_path, final_graph_records_path
    )
    optaba = pd.read_csv(optaba_path)
    optaba_challenges = pd.read_csv(optaba_challenges_path)
    abapc = build_abapc_optimality_summary(optaba_manifest)
    repair_tex = output_dir / "table_contestability_margin_comparison.tex"
    overview_tex = output_dir / "table_contestability_method_overview.tex"
    primary_comparison_tex = output_dir / "table_primary_repair_enforcement.tex"
    margin_dataset_tex = output_dir / "table_margin_by_dataset.tex"
    abapc_summary = output_dir / "abapc_optimality_summary.csv"
    margin_insights_path = output_dir / "margin_insight_summary.csv"
    _atomic_text(
        repair_tex,
        render_repair_table(
            abapc, optaba_challenges, aspcr, aspcr_challenges, mpc_matched
        ),
    )
    _atomic_text(
        overview_tex,
        render_method_overview(
            abapc, optaba, optaba_challenges, aspcr, aspcr_challenges,
            aspcr_trace, aspcr_primary, mpc_matched, primary
        ),
    )
    _atomic_text(
        primary_comparison_tex,
        render_primary_repair_enforcement_table(primary),
    )
    _atomic_text(
        margin_dataset_tex,
        render_margin_by_dataset_table(
            optaba, optaba_challenges, aspcr, aspcr_challenges
        ),
    )
    _atomic_text(
        output_dir / "aspcr_trace_accounting.csv",
        aspcr_trace.to_csv(index=False),
    )
    _atomic_text(abapc_summary, abapc.to_csv(index=False))
    _atomic_text(
        margin_insights_path,
        build_margin_insights(optaba_challenges, aspcr_challenges).to_csv(index=False),
    )
    manifest_path = output_dir / "baseline_contestability_table_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "aspcr_status": aspcr_manifest.get("status"),
            "aspcr_trace_accounting": aspcr_trace.to_dict("records"),
            "aspcr_primary_accounting": aspcr_primary,
            "mpc_matched_accounting": mpc_matched,
            "primary_decision_accounting": primary,
            "allow_partial_aspcr": bool(args.allow_partial_aspcr),
            "inputs": {
                "mpc": {"path": str(mpc_path), "sha256": _sha256(mpc_path)},
                "mpc_matched_audit": {
                    "path": str(mpc_matched_audit_path),
                    "sha256": _sha256(mpc_matched_audit_path),
                },
                "mpc_primary_audit": {
                    "path": str(mpc_primary_audit_path),
                    "sha256": _sha256(mpc_primary_audit_path),
                },
                "aspcr": {"path": str(aspcr_path), "sha256": _sha256(aspcr_path)},
                "aspcr_challenges": {
                    "path": str(aspcr_challenges_path), "sha256": _sha256(aspcr_challenges_path)
                },
                "final_fact_records": {
                    "path": str(final_fact_records_path),
                    "sha256": _sha256(final_fact_records_path),
                },
                "final_graph_records": {
                    "path": str(final_graph_records_path),
                    "sha256": _sha256(final_graph_records_path),
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
                "repair_table": {"path": str(repair_tex), "sha256": _sha256(repair_tex)},
                "overview_table": {"path": str(overview_tex), "sha256": _sha256(overview_tex)},
                "primary_repair_enforcement_table": {
                    "path": str(primary_comparison_tex),
                    "sha256": _sha256(primary_comparison_tex),
                },
                "margin_dataset_table": {
                    "path": str(margin_dataset_tex),
                    "sha256": _sha256(margin_dataset_tex),
                },
                "abapc_summary": {"path": str(abapc_summary), "sha256": _sha256(abapc_summary)},
                "margin_insights": {
                    "path": str(margin_insights_path),
                    "sha256": _sha256(margin_insights_path),
                },
            },
        },
    )
    print(f"Wrote dataset-level margin comparison: {repair_tex}")
    print(f"Wrote contestability overview table: {overview_tex}")
    print(f"Wrote primary repair/enforcement table: {primary_comparison_tex}")
    print(f"Wrote dataset-level margin table: {margin_dataset_tex}")
    print(f"Wrote ABA-PC optimality summary: {abapc_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
