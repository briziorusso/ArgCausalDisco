#!/usr/bin/env python3
"""Recreate the paper LaTeX table for the main WC sweep results (10 reps).

This script expects the 5 required run directories to be present as children of
this folder (e.g., after collecting/moving them here).

It reads each run's `summary.json` produced by `scripts/wc_opt_strategy_sweep.py`
and renders the OptABA-PC -- ABA-PC deltas used in the paper table.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TableSpec:
    dataset_label: str
    dir_match: re.Pattern[str]


SPECS: list[TableSpec] = [
    TableSpec("Cancer (5)", re.compile(r"^wc_sweep_bnlearn_cancer_")),
    TableSpec("Survey (6)", re.compile(r"^wc_sweep_bnlearn_survey_")),
    TableSpec("Asia (8)", re.compile(r"^wc_sweep_bnlearn_asia_")),
    TableSpec("ER (5)", re.compile(r"^wc_sweep_5_2026_")),
    TableSpec("ER (8)", re.compile(r"^wc_sweep_8_2026_")),
]


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _mean(xs: Iterable[Any]) -> float:
    ys = [float(x) for x in xs if _is_num(x)]
    if not ys:
        return float("nan")
    return float(sum(ys) / len(ys))


def _pct(delta: float, base: float) -> float:
    if not (_is_num(delta) and _is_num(base)):
        return float("nan")
    if abs(float(base)) <= 1e-12:
        return float("nan")
    return float(delta) / float(base) * 100.0


def _fmt_delta_pct(delta: float, base: float, *, prec: int) -> str:
    pct = _pct(delta, base)
    if _is_num(pct):
        return rf"\ensuremath{{{delta:+.{prec}f}\,({pct:+0.1f}\%)}}"
    return rf"\ensuremath{{{delta:+.{prec}f}}}"


def _fmt_time(delta: float, ratio: float) -> str:
    if _is_num(delta) and _is_num(ratio):
        return rf"\ensuremath{{{delta:+.3f}\,(\times {ratio:0.2f})}}"
    if _is_num(delta):
        return rf"\ensuremath{{{delta:+.3f}}}"
    return r"\ensuremath{NA}"


def _require_reps_1_to_10(rows: list[dict[str, Any]], *, label: str, run_dir: Path) -> None:
    reps = sorted({int(r.get("rep")) for r in rows if isinstance(r.get("rep"), int)})
    want = list(range(1, 11))
    if reps != want:
        raise RuntimeError(f"{run_dir.name}: expected 10 reps (1..10) for {label}, got {reps!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path}") from e


def _pick_time_key(base_rows: list[dict[str, Any]], opt_rows: list[dict[str, Any]]) -> str:
    def has_any_solve(rows: list[dict[str, Any]]) -> bool:
        return any(_is_num(r.get("solve_time_sec")) for r in rows)

    # The paper table uses solve-time when available; some reps may time out and
    # omit solve_time_sec, so we only require at least one numeric entry.
    if has_any_solve(base_rows) and has_any_solve(opt_rows):
        return "solve_time_sec"

    # Fallback when solve-time is not recorded (e.g., hard timeouts).
    return "wall_time_sec"


def _row_from_run(run_dir: Path, *, dataset_label: str) -> tuple[str, list[str]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(str(summary_path))
    obj = _load_json(summary_path)

    baseline_rows = [
        r
        for r in (obj.get("baseline_results", []) or [])
        if isinstance(r, dict) and r.get("solver") == "causalaba_increm" and r.get("opt_mode") == "optN"
    ]
    opt_rows = [
        r
        for r in (obj.get("wc_results", []) or [])
        if isinstance(r, dict)
        and r.get("encoding") == "inc"
        and r.get("objective") == "lex"
        and r.get("opt_strategy") == "bb"
        and r.get("opt_mode") == "optN"
        and r.get("reification") == "mus"
    ]

    _require_reps_1_to_10(baseline_rows, label="baseline (causalaba_increm,optN)", run_dir=run_dir)
    _require_reps_1_to_10(opt_rows, label="OptABA-PC (inc,lex,bb,optN,mus)", run_dir=run_dir)

    def mu(rows: list[dict[str, Any]], key: str) -> float:
        return _mean(r.get(key) for r in rows)

    # Core metrics (means over 10 repetitions).
    b_naccT = mu(baseline_rows, "n_tests_accepted_true")
    o_naccT = mu(opt_rows, "n_tests_accepted_true")
    d_naccT = o_naccT - b_naccT

    b_fact = mu(baseline_rows, "accepted_fact_f1")
    o_fact = mu(opt_rows, "accepted_fact_f1")
    d_fact = o_fact - b_fact

    b_ncp = mu(baseline_rows, "n_cpdags_compat")
    o_ncp = mu(opt_rows, "n_cpdags_compat")
    d_ncp = o_ncp - b_ncp

    b_shd = mu(baseline_rows, "cpdag_shd_avg")
    o_shd = mu(opt_rows, "cpdag_shd_avg")
    d_shd = o_shd - b_shd

    b_shdw = mu(baseline_rows, "cpdag_shd_worst")
    o_shdw = mu(opt_rows, "cpdag_shd_worst")
    d_shdw = o_shdw - b_shdw

    time_key = _pick_time_key(baseline_rows, opt_rows)
    b_t = mu(baseline_rows, time_key)
    o_t = mu(opt_rows, time_key)
    d_t = o_t - b_t
    t_ratio = (o_t / b_t) if _is_num(o_t) and _is_num(b_t) and abs(float(b_t)) > 1e-12 else float("nan")

    cells = [
        _fmt_delta_pct(d_naccT, b_naccT, prec=1),
        _fmt_delta_pct(d_fact, b_fact, prec=3),
        _fmt_delta_pct(d_ncp, b_ncp, prec=1),
        _fmt_delta_pct(d_shd, b_shd, prec=3),
        _fmt_delta_pct(d_shdw, b_shdw, prec=3),
        _fmt_time(d_t, t_ratio),
    ]
    return dataset_label, cells


def _find_run_dir(root: Path, spec: TableSpec) -> Path:
    matches = [p for p in root.iterdir() if p.is_dir() and spec.dir_match.search(p.name)]
    if not matches:
        raise FileNotFoundError(f"No run dir matching {spec.dir_match.pattern!r} under {root}")
    if len(matches) > 1:
        names = ", ".join(sorted(p.name for p in matches))
        raise RuntimeError(f"Multiple run dirs match {spec.dir_match.pattern!r} under {root}: {names}")
    return matches[0]


def _render_table(rows: list[tuple[str, list[str]]]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset (nodes) & $\Delta n_{\mathrm{acc}}^{T}\uparrow$ & $\Delta \mathrm{FactF1}\uparrow$ & $\Delta \#\mathrm{CPDAG}\downarrow$ & $\Delta \mathrm{SHD}\downarrow$ & $\Delta \mathrm{SHD}_{w}\downarrow$ & $\Delta t\downarrow$\\"
    )
    lines.append(r"\midrule")
    for dataset_label, cells in rows:
        lines.append(dataset_label + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Results on bnlearn and synthetic benchmarks (10 repetitions). Entries report OptABA-PC -- ABA-PC as $\Delta$ (relative change in parentheses). $\Delta n_{\mathrm{acc}}^{T}$ is the change in the number of accepted true constraints and $\Delta \mathrm{FactF1}$ is the change in the F1 of fact classification. All graphical metrics ($\#\mathrm{CPDAG}$, SHD, and worst-case $\mathrm{SHD}_w$) are computed on the returned CPDAG solution sets. Runtime $\Delta t$ is the absolute increase in seconds (OptABA-PC time divided by ABA-PC time in parentheses).}"
    )
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parent

    rows: list[tuple[str, list[str]]] = []
    for spec in SPECS:
        run_dir = _find_run_dir(root, spec)
        rows.append(_row_from_run(run_dir, dataset_label=spec.dataset_label))

    out_tex = _render_table(rows)

    out_path = root / "main_results_table.tex"
    out_path.write_text(out_tex)
    print(out_tex, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
