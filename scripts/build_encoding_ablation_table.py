#!/usr/bin/env python3
"""Validate and tabulate the matched active-path/Bayes-ball ablation.

The script requires 50 seed-matched runs for every reported dataset.  It checks
that every completed pair produces the same ABA-PC repair and then compares
end-to-end method time (program construction plus solving).  Timed-out pairs do
not enter the time summary or paired test; completion is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DATASETS = (
    ("cancer", "Cancer (5)"),
    ("earthquake", "Earthquake (5)"),
    ("survey", "Survey (6)"),
    ("asia", "Asia (8)"),
    ("er5", "ER (5)"),
    ("er8", "ER (8)"),
    ("sf5", "SF (5)"),
    ("sf8", "SF (8)"),
)
SEMANTIC_FIELDS = (
    "removed",
    "n_tests_total",
    "n_tests_accepted",
    "n_tests_accepted_true",
    "accepted_weight",
    "weight_accepted_true",
    "accepted_fact_f1",
    "models_after",
)


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini--Hochberg adjusted p-values in input order."""
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = [min(1.0, p_values[index] * count / rank)
              for rank, index in enumerate(order, start=1)]
    for position in range(count - 2, -1, -1):
        ranked[position] = min(ranked[position], ranked[position + 1])
    adjusted = [0.0] * count
    for index, value in zip(order, ranked):
        adjusted[int(index)] = float(value)
    return adjusted


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_time(mean: float, standard_deviation: float) -> str:
    digits = 3 if mean < 1.0 else 2
    return f"${mean:.{digits}f}\\pm{standard_deviation:.{digits}f}$"


def read_dataset(root: Path, key: str, label: str) -> dict[str, object]:
    summaries = sorted(root.glob(f"{key}_*/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one completed summary for {key}, found {summaries}")
    summary_path = summaries[0]
    payload = json.loads(summary_path.read_text())
    if payload.get("seed") != 2026 or payload.get("reps") != 50:
        raise RuntimeError(f"Unexpected seed/repetition configuration in {summary_path}")
    if set(payload.get("encodings", [])) != {"base", "inc"}:
        raise RuntimeError(f"Unexpected encodings in {summary_path}")

    by_rep: dict[int, dict[str, dict[str, object]]] = {}
    for row in payload["baseline_results"]:
        by_rep.setdefault(int(row["rep"]), {})[str(row["solver"])] = row
    if set(by_rep) != set(range(1, 51)):
        raise RuntimeError(f"Missing or duplicate repetitions in {summary_path}")

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for repetition, methods in sorted(by_rep.items()):
        if set(methods) != {"causalaba", "causalaba_increm"}:
            raise RuntimeError(f"Missing encoding for {key}, repetition {repetition}")
        pairs.append((methods["causalaba"], methods["causalaba_increm"]))

    original_complete = sum(not bool(original["timed_out"]) for original, _ in pairs)
    bayes_complete = sum(not bool(bayes["timed_out"]) for _, bayes in pairs)
    if bayes_complete != 50:
        raise RuntimeError(f"Bayes-ball did not complete all {key} repetitions")
    completed_pairs = [
        (original, bayes)
        for original, bayes in pairs
        if not bool(original["timed_out"]) and not bool(bayes["timed_out"])
    ]

    for original, bayes in completed_pairs:
        mismatches = [field for field in SEMANTIC_FIELDS if original.get(field) != bayes.get(field)]
        if mismatches:
            raise RuntimeError(
                f"Repair mismatch for {key}, repetition {original['rep']}: {mismatches}"
            )

    original_time = np.asarray(
        [float(original["method_time_sec"]) for original, _ in completed_pairs]
    )
    bayes_time = np.asarray(
        [float(bayes["method_time_sec"]) for _, bayes in completed_pairs]
    )
    difference = bayes_time - original_time
    raw_p = 1.0 if np.all(difference == 0) else float(wilcoxon(difference).pvalue)
    return {
        "dataset": label,
        "key": key,
        "paired_n": len(completed_pairs),
        "original_complete": original_complete,
        "bayes_complete": bayes_complete,
        "original_mean_sec": float(original_time.mean()),
        "original_sd_sec": float(original_time.std(ddof=1)),
        "bayes_mean_sec": float(bayes_time.mean()),
        "bayes_sd_sec": float(bayes_time.std(ddof=1)),
        "bayes_over_original": float(bayes_time.mean() / original_time.mean()),
        "raw_p": raw_p,
        "summary_path": str(summary_path),
    }


def render_tex(rows: list[dict[str, object]]) -> str:
    body = []
    for row in rows:
        marker = stars(float(row["adjusted_p"]))
        original = format_time(float(row["original_mean_sec"]), float(row["original_sd_sec"]))
        bayes = format_time(float(row["bayes_mean_sec"]), float(row["bayes_sd_sec"]))
        if marker:
            if float(row["original_mean_sec"]) < float(row["bayes_mean_sec"]):
                original = f"$\\mathbf{{{original[1:-1]}}}^{{{marker}}}$"
            else:
                bayes = f"$\\mathbf{{{bayes[1:-1]}}}^{{{marker}}}$"
        body.append(
            f"{row['dataset']} & {row['original_complete']}/50 & {row['bayes_complete']}/50 "
            f"& {row['paired_n']} & {original} & {bayes} "
            f"& ${float(row['bayes_over_original']):.2f}\\times$\\\\"
        )
    source_comments = "\n".join(
        f"% {row['dataset']}: {row['summary_path']}" for row in rows
    )
    return rf"""% Generated by scripts/build_encoding_ablation_table.py.
% Do not edit numerical cells manually. Source summaries:
{source_comments}
\begin{{table*}}[t]
\centering
\small
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
Dataset & \multicolumn{{2}}{{c}}{{Completed}} & Paired $n$ & \multicolumn{{2}}{{c}}{{End-to-end time (s)}} & BB/original \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{5-6}}
 & Active paths & Bayes-ball & & Active paths & Bayes-ball & \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\caption{{Matched encoding-efficiency ablation over seeds 2026--2075, with $N=5000$, $G^2$ tests at $\alpha=0.01$, identical test traces and weights, and the same sequential ABA-PC repair policy. Only the causal-consistency encoding changes. Times are mean $\pm$ standard deviation for completed seed pairs and include program construction and solving; ``BB/original'' is the ratio of those means, so values below one favour Bayes-ball. Completion is over all 50 runs. Bold marks the significantly faster implementation under paired two-sided Wilcoxon tests with BH adjustment across datasets ($^*$, $^{{**}}$, and $^{{***}}$ for adjusted $p<0.05$, $0.01$, and $0.001$). Every completed pair returned identical retained-fact counts, weights, and model counts.}}
\label{{tab:bayesball-runtime}}
\end{{table*}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    rows = [read_dataset(args.results_root, key, label) for key, label in DATASETS]
    adjusted = bh_adjust([float(row["raw_p"]) for row in rows])
    for row, p_value in zip(rows, adjusted):
        row["adjusted_p"] = p_value

    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(render_tex(rows))
    csv_output = args.csv_output or args.results_root / "encoding_ablation_summary.csv"
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows) * 50} matched runs; wrote {args.tex_output} and {csv_output}")


if __name__ == "__main__":
    main()
