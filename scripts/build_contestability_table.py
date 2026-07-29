#!/usr/bin/env python3
"""Build the contestability dataset-summary CSV and paper-ready TeX table.

Only proved forced-retention optima contribute numerical margins.  Timeout,
unproved, and infeasible rows remain in the challenge CSV and are counted
explicitly here.  The ER (5) zero-challenge row is always retained when it is
present in the experiment manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "tables"
    / "paper_current_alpha001_nowrong_noweight_50rep_preview"
)

DATASET_SPECS: tuple[tuple[str, str], ...] = (
    ("cancer", "Cancer (5)"),
    ("earthquake", "Earthquake (5)"),
    ("survey", "Survey (6)"),
    ("asia", "Asia (8)"),
    ("er5", "ER (5)"),
    ("er8", "ER (8)"),
    ("sf5", "SF (5)"),
    ("sf8", "SF (8)"),
)
DATASET_ORDER = {label: i for i, (_key, label) in enumerate(DATASET_SPECS)}

SUMMARY_FIELDS = (
    "dataset",
    "analyzable_seeds",
    "challenges",
    "proved_challenges",
    "zero_margin_challenges",
    "zero_margin_proportion",
    "normalized_margin_median",
    "normalized_margin_q1",
    "normalized_margin_q3",
    "normalized_margin_iqr",
    "infeasible_challenges",
    "timeouts",
    "unproved_challenges",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    """NumPy-compatible linear quantile without a NumPy dependency."""

    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def load_challenges(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Contestability challenge CSV not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "seed",
            "fact_id",
            "normalized_margin",
            "zero_margin",
            "timeout",
            "optimality_proven",
            "infeasible",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing challenge columns in {path}: {sorted(missing)}")
        rows = list(reader)

    seen: set[str] = set()
    for row in rows:
        fact_id = str(row.get("fact_id", ""))
        if not fact_id:
            raise ValueError(f"Challenge row without fact_id in {path}")
        if fact_id in seen:
            raise ValueError(f"Duplicate challenge fact_id {fact_id!r} in {path}")
        seen.add(fact_id)

        proved = _as_bool(row.get("optimality_proven"))
        timed_out = _as_bool(row.get("timeout"))
        infeasible = _as_bool(row.get("infeasible"))
        forced_cost = _as_float(row.get("forced_retention_optimum_cost"))
        margin = _as_float(row.get("margin"))
        normalized = _as_float(row.get("normalized_margin"))
        if proved:
            if timed_out or infeasible:
                raise ValueError(f"Proved challenge {fact_id!r} is also timeout/infeasible")
            if forced_cost is None or margin is None or normalized is None:
                raise ValueError(f"Proved challenge {fact_id!r} has unavailable numerical fields")
            if _as_bool(row.get("zero_margin")) != (margin == 0.0):
                raise ValueError(f"zero_margin disagrees with margin for {fact_id!r}")
        elif forced_cost is not None or margin is not None or normalized is not None:
            raise ValueError(
                f"Unproved challenge {fact_id!r} reports a forced optimum or derived margin"
            )
    return rows


def summarise_contestability(
    challenges: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = list(challenges)
    manifest_instances = [
        item for item in (manifest.get("baseline_instances", []) or []) if isinstance(item, dict)
    ]
    labels = {
        str(item.get("dataset"))
        for item in manifest_instances
        if item.get("dataset")
    }
    labels.update(str(row.get("dataset")) for row in rows if row.get("dataset"))
    ordered_labels = sorted(labels, key=lambda label: (DATASET_ORDER.get(label, 999), label))

    summaries: list[dict[str, Any]] = []
    for label in ordered_labels:
        dataset_rows = [row for row in rows if str(row.get("dataset")) == label]
        instances = [item for item in manifest_instances if str(item.get("dataset")) == label]
        analyzable_seeds = {
            int(item["seed"])
            for item in instances
            if item.get("analyzable") is True and item.get("seed") is not None
        }
        proved_rows = [
            row
            for row in dataset_rows
            if _as_bool(row.get("optimality_proven"))
            and _as_float(row.get("normalized_margin")) is not None
        ]
        normalized = [
            float(_as_float(row.get("normalized_margin")))
            for row in proved_rows
            if _as_float(row.get("normalized_margin")) is not None
        ]
        zero_count = sum(_as_bool(row.get("zero_margin")) for row in proved_rows)
        q1 = _quantile(normalized, 0.25)
        median = _quantile(normalized, 0.5)
        q3 = _quantile(normalized, 0.75)
        summaries.append(
            {
                "dataset": label,
                "analyzable_seeds": len(analyzable_seeds),
                "challenges": len(dataset_rows),
                "proved_challenges": len(proved_rows),
                "zero_margin_challenges": zero_count,
                "zero_margin_proportion": zero_count / len(proved_rows) if proved_rows else None,
                "normalized_margin_median": median,
                "normalized_margin_q1": q1,
                "normalized_margin_q3": q3,
                "normalized_margin_iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
                "infeasible_challenges": sum(_as_bool(row.get("infeasible")) for row in dataset_rows),
                "timeouts": sum(_as_bool(row.get("timeout")) for row in dataset_rows),
                "unproved_challenges": len(dataset_rows) - len(proved_rows),
            }
        )
    return summaries


def write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    temporary.replace(path)


def _latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "--"
    return f"{number:.{digits}f}"


def render_tex_table(
    summaries: Sequence[dict[str, Any]],
    *,
    label: str,
    caption: str,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & Seeds & Contestations & Proved & Zero margin & Median [Q1,Q3] & Infeasible & Timeouts \\",
        r"\midrule",
    ]
    for row in summaries:
        proportion = _fmt_number(row.get("zero_margin_proportion"))
        if proportion != "--":
            proportion = f"{proportion} ({int(row.get('zero_margin_challenges', 0))})"
        median = _fmt_number(row.get("normalized_margin_median"))
        q1 = _fmt_number(row.get("normalized_margin_q1"))
        q3 = _fmt_number(row.get("normalized_margin_q3"))
        median_iqr = "--" if median == "--" else f"{median} [{q1},{q3}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.get("dataset", "")),
                    str(int(row.get("analyzable_seeds", 0))),
                    str(int(row.get("challenges", 0))),
                    str(int(row.get("proved_challenges", 0))),
                    proportion,
                    median_iqr,
                    str(int(row.get("infeasible_challenges", 0))),
                    str(int(row.get("timeouts", 0))),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize",
            (
                r"Zero margin is reported as a proportion (count) among contestations with both optima proved. "
                r"The normalised-margin column reports the median [first quartile, third quartile]. "
                r"Unavailable numerical summaries are shown as --. ER (5) is retained explicitly even when "
                r"its frozen optima release no CI facts."
            ),
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--challenges", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--tex", default=None)
    parser.add_argument("--label", default="tab:empirical-contestability")
    parser.add_argument(
        "--caption",
        default=(
            "Empirical contestability of CI facts released by the reported OptABA-PC optimum "
            "over the frozen 50 seeds."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Build a diagnostic preview even when the runner manifest is not complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    challenge_path = Path(args.challenges).expanduser().resolve() if args.challenges else output_dir / "contestability_challenges.csv"
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir / "contestability_manifest.json"
    summary_path = Path(args.summary_csv).expanduser().resolve() if args.summary_csv else output_dir / "contestability_dataset_summary.csv"
    tex_path = Path(args.tex).expanduser().resolve() if args.tex else output_dir / "table_empirical_contestability.tex"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Contestability manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" and not args.allow_partial:
        raise RuntimeError(
            f"Manifest status is {manifest.get('status')!r}; finish the runner or pass --allow-partial for a preview."
        )

    recorded_challenge = ((manifest.get("outputs", {}) or {}).get("challenges_csv", {}) or {})
    recorded_sha256 = recorded_challenge.get("sha256")
    if recorded_sha256 and recorded_sha256 != _sha256_path(challenge_path):
        raise ValueError(
            f"Challenge CSV hash does not match the runner manifest: {challenge_path}"
        )
    challenges = load_challenges(challenge_path)
    if manifest.get("status") == "complete":
        expected_rows = int((manifest.get("counts", {}) or {}).get("challenge_rows", -1))
        if expected_rows != len(challenges):
            raise ValueError(
                f"Complete manifest records {expected_rows} challenge rows but CSV contains {len(challenges)}"
            )
    summaries = summarise_contestability(challenges, manifest)
    write_summary_csv(summary_path, summaries)
    tex = render_tex_table(summaries, label=str(args.label), caption=str(args.caption))
    _atomic_write_text(tex_path, tex)

    outputs = dict(manifest.get("outputs", {}) or {})
    for name, path in (
        ("challenges_csv", challenge_path),
        ("dataset_summary_csv", summary_path),
        ("tex_table", tex_path),
    ):
        outputs[name] = {"path": str(path), "sha256": _sha256_path(path)}
    manifest["outputs"] = outputs
    manifest["table_builder"] = {
        "script": str(Path(__file__).resolve()),
        "updated_at_utc": _utc_now(),
        "allow_partial": bool(args.allow_partial),
        "quantile_method": "linear interpolation at positions (n-1)q",
        "proved_row_rule": "optimality_proven=true and normalized_margin is finite",
    }
    _atomic_write_json(manifest_path, manifest)

    print(f"Wrote dataset summary: {summary_path}")
    print(f"Wrote TeX table: {tex_path}")
    print(f"Updated manifest output hashes: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
