#!/usr/bin/env python3

"""Render run-level and overall summaries from saved wc sweep results.

This script is tmux-friendly: use `--pager` to view output via `less`.

It reads `summary.json` files produced by `scripts/wc_opt_strategy_sweep.py`.

Examples:
  python scripts/wc_sweep_report.py --results-dir results/wc_sweep_5_2004_20260204_231905 --pager
  python scripts/wc_sweep_report.py --summary results/wc_sweep_5_2004_20260204_231905/summary.json
  python scripts/wc_sweep_report.py --glob 'results/wc_sweep_*_*/summary.json' --pager
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable


# NOTE: When running as `python scripts/wc_sweep_report.py`, Python places the
# `scripts/` directory (not the repo root) on sys.path. Some optional
# recomputation paths import modules from the repo root (e.g., `tests_mus.py`),
# so we ensure the repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)


def _is_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _stats(xs: Iterable[Any]) -> tuple[float, float, float]:
    ys: list[float] = []
    for x in xs:
        if _is_number(x):
            ys.append(float(x))
    if not ys:
        nan = float("nan")
        return nan, nan, nan
    return _mean(ys), float(min(ys)), float(max(ys))


def _stdev(xs: Iterable[Any]) -> float:
    """Sample standard deviation (ddof=1); returns 0.0 for <=1 value.

    Ignores None and non-numeric values.
    """

    ys: list[float] = []
    for x in xs:
        if _is_number(x):
            ys.append(float(x))
    if len(ys) <= 1:
        return 0.0
    try:
        import statistics

        return float(statistics.stdev(ys))
    except Exception:
        mu = float(sum(ys) / len(ys))
        var = sum((y - mu) ** 2 for y in ys) / float(max(1, len(ys) - 1))
        return float(var**0.5)


def _fmt_float(x: Any, width: int, prec: int = 3) -> str:
    if not _is_number(x):
        return "".rjust(width)
    return f"{float(x):{width}.{prec}f}"


def _fmt_int(x: Any, width: int) -> str:
    try:
        if x is None:
            return "".rjust(width)
        return f"{int(x):{width}d}"
    except Exception:
        return "".rjust(width)


def _fmt_range(mn: float, mx: float, width: int, prec: int = 3) -> str:
    if not (_is_number(mn) and _is_number(mx)):
        return "".rjust(width)
    s = f"[{mn:.{prec}f},{mx:.{prec}f}]"
    return s.rjust(width)


def _fmt_range1(mn: float, mx: float, width: int) -> str:
    if not (_is_number(mn) and _is_number(mx)):
        return "".rjust(width)
    s = f"[{mn:.1f},{mx:.1f}]"
    return s.rjust(width)


def _mean_key(rows: list[Any], key: str, *, exclude_timeouts: bool) -> float:
    xs: list[float] = []
    for r in rows:
        if exclude_timeouts and bool(getattr(r, "timed_out", False)):
            continue
        v = getattr(r, key, None)
        if _is_number(v):
            xs.append(float(v))
    return _mean(xs)


def _delta(a: float, b: float) -> tuple[float, float]:
    """Return (b-a, pct_of_a)."""

    if not (_is_number(a) and _is_number(b)):
        nan = float("nan")
        return nan, nan
    da = float(b) - float(a)
    if abs(float(a)) <= 1e-12:
        return da, float("nan")
    return da, (da / float(a) * 100.0)


def _fmt_delta(da: float, pct: float, *, prec: int) -> str:
    if not _is_number(da):
        return "NA"
    if _is_number(pct):
        return f"{float(da):+.{prec}f} ({float(pct):+0.1f}%)"
    return f"{float(da):+.{prec}f}"


def _ratio(a: float, b: float) -> float:
    if not (_is_number(a) and _is_number(b)):
        return float("nan")
    if abs(float(a)) <= 1e-12:
        return float("nan")
    return float(b) / float(a)


def _mean_time_sec(rows: list[Any]) -> float:
    """Mean wall-clock time for a method over runs.

    Prefers `method_time_sec` (if present), otherwise falls back to `wall_time_sec`
    per-row so timeouts (which often omit method_time_sec decomposition) are
    still accounted for.
    """

    xs: list[float] = []
    for r in rows:
        v = getattr(r, "method_time_sec", None)
        if not _is_number(v):
            v = getattr(r, "wall_time_sec", None)
        if _is_number(v):
            xs.append(float(v))
    return _mean(xs)


def _pick_baseline_group(rows: list[Any], *, solver: str) -> tuple[str | None, list[Any]]:
    """Pick the most common opt_mode group for a given baseline solver."""

    by_optm: dict[str, list[Any]] = {}
    for r in rows:
        if str(getattr(r, "solver", "")) != solver:
            continue
        optm = str(getattr(r, "opt_mode", ""))
        by_optm.setdefault(optm, []).append(r)
    if not by_optm:
        return None, []
    optm_best = max(by_optm.keys(), key=lambda k: len(by_optm[k]))
    return optm_best, by_optm[optm_best]


def _pick_wc_group(rows: list[Any], *, encoding: str) -> tuple[tuple[str, str, str, str] | None, list[Any]]:
    """Pick the most common WC method group for an encoding.

    Group key = (objective, opt_strategy, opt_mode, reification).
    """

    groups: dict[tuple[str, str, str, str], list[Any]] = {}
    for r in rows:
        if str(getattr(r, "encoding", "")) != encoding:
            continue
        key = (
            str(getattr(r, "objective", "")),
            str(getattr(r, "opt_strategy", "")),
            str(getattr(r, "opt_mode", "")),
            str(getattr(r, "reification", "")),
        )
        groups.setdefault(key, []).append(r)
    if not groups:
        return None, []
    best_key = max(groups.keys(), key=lambda k: len(groups[k]))
    return best_key, groups[best_key]


def _fmt_wc_group(key: tuple[str, str, str, str] | None) -> str:
    if not key:
        return "NA"
    obj, strat, optm, reif = key
    return f"{strat}/{reif}({obj})[{optm}]"


def _load_json(path: str) -> dict[str, Any]:
    # The sweep may be concurrently writing summary.partial.json; tolerate a
    # transient partial write by retrying a few times.
    import time

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            last_err = e
        except OSError as e:
            last_err = e
        time.sleep(0.05 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _try_load_saved_metric_ranks(results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / "metric_ranks.json"
    if not path.exists():
        return None
    try:
        d = _load_json(str(path))
        if not isinstance(d, dict):
            return None
        if not isinstance(d.get("ranks_avg", None), dict):
            return None
        if not isinstance(d.get("methods", None), list):
            return None
        return d
    except Exception:
        return None


def _csv_sanitize(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        try:
            return ",".join(str(x) for x in v)
        except Exception:
            return str(v)
    return v


def _write_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_sanitize(r.get(k, "")) for k in fieldnames})


def _infer_sweep_stem(*, results_dir: Path, src_path: str) -> str:
    name = (results_dir.name or "").strip()
    if name:
        return name
    p = Path(src_path)
    if p.name in {"summary.json", "summary.partial.json"}:
        return (p.parent.name or "sweep").strip() or "sweep"
    return (p.stem or "sweep").strip() or "sweep"


def _export_csvs(
    *,
    csv_out_dir: Path,
    results_dir: Path,
    src_path: str,
    argv: str | None,
    run_meta: dict[str, Any] | None,
    baseline_rows: list[_BaselineRow],
    wc_rows: list[_WcRow],
    baseline_src_path: str | None = None,
    wc_src_path: str | None = None,
) -> None:
    """Export a single combined per-run CSV for easy sorting/filtering."""

    out_dir = csv_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_stem = _infer_sweep_stem(results_dir=results_dir, src_path=src_path)
    csv_path = out_dir / f"{sweep_stem}.csv"
    summary_csv_path = out_dir / f"{sweep_stem}.summary.csv"

    run_meta = run_meta or {}

    def _pipe_join(xs: Any) -> str:
        if xs is None:
            return ""
        if isinstance(xs, (list, tuple, set)):
            try:
                return "|".join(str(x) for x in xs)
            except Exception:
                return str(xs)
        return str(xs)

    # Run configuration columns repeated on every row, so multiple runs can be stacked.
    run_cfg = {
        "run_id": sweep_stem,
        "run_src": src_path,
        "run_results_dir": str(results_dir),
        "run_n_nodes": run_meta.get("n_nodes"),
        "run_seed": run_meta.get("seed"),
        "run_reps": run_meta.get("reps"),
        "run_timeout_sec": run_meta.get("timeout_sec"),
        "run_pct_wrong_facts": run_meta.get("pct_wrong_facts"),
        "run_total_weight": run_meta.get("total_weight"),
        "run_strategies": _pipe_join(run_meta.get("strategies")),
        "run_objectives": _pipe_join(run_meta.get("objectives")),
        "run_encodings": _pipe_join(run_meta.get("encodings")),
        "run_opt_modes": _pipe_join(run_meta.get("opt_modes")),
        "run_reifications": _pipe_join(run_meta.get("reifications")),
    }

    # Optional ranking variables (average per-method ranks). Prefer recomputing
    # when mixing sources so baselines align with the merged data.
    rank_names: list[str] = []
    ranks_avg: dict[str, Any] = {}

    def _rank_map(
        method_to_value: dict[tuple[str, str, str, str, str], Any], *, higher_is_better: bool
    ) -> dict[tuple[str, str, str, str, str], float]:
        methods_all = list(method_to_value.keys())
        n_methods = len(methods_all)
        present: list[tuple[tuple[str, str, str, str, str], float]] = []
        for m, v in method_to_value.items():
            if _is_number(v):
                present.append((m, float(v)))
        present.sort(key=lambda kv: kv[1], reverse=higher_is_better)
        ranks: dict[tuple[str, str, str, str, str], float] = {m: float(n_methods) for m in methods_all}
        i = 0
        while i < len(present):
            j = i + 1
            while j < len(present) and present[j][1] == present[i][1]:
                j += 1
            rank_lo = i + 1
            rank_hi = j
            rank_avg = (rank_lo + rank_hi) / 2.0
            for k in range(i, j):
                ranks[present[k][0]] = float(rank_avg)
            i = j
        return ranks

    def _compute_ranks_from_rows() -> tuple[list[str], dict[str, Any]]:
        methods: list[tuple[str, str, str, str, str]] = []
        rep_set: set[int] = set()
        method_rep: dict[tuple[str, str, str, str, str], dict[int, Any]] = {}

        for r in baseline_rows:
            enc = "inc" if r.solver == "causalaba_increm" else "base"
            optm = str(r.opt_mode or "optN")
            mkey = (enc, "-", "baseline", optm, "-")
            md = method_rep.setdefault(mkey, {})
            md[int(r.rep)] = r
            rep_set.add(int(r.rep))

        for r in wc_rows:
            mkey = (r.encoding, r.objective, r.opt_strategy, r.opt_mode, r.reification)
            md = method_rep.setdefault(mkey, {})
            md[int(r.rep)] = r
            rep_set.add(int(r.rep))

        methods = sorted(method_rep.keys())
        reps_sorted = sorted(rep_set)
        if not methods or not reps_sorted:
            return [], {}

        def _time_for_rank(it: Any) -> float | None:
            mt = getattr(it, "method_time_sec", None)
            if mt is not None:
                return mt
            b = getattr(it, "build_time_sec", None)
            s = getattr(it, "solve_time_sec", None)
            e = getattr(it, "eval_time_sec", None)
            if b is not None and s is not None and e is not None:
                return float(b) + float(s) + float(e)
            if b is not None and s is not None:
                return float(b) + float(s)
            return getattr(it, "wall_time_sec", None)

        def _solve_for_rank(it: Any) -> float | None:
            s = getattr(it, "solve_time_sec", None)
            return float(s) if s is not None else None

        metric_defs: list[tuple[str, bool, Any]] = [
            ("timeR", False, _time_for_rank),
            ("solveR", False, _solve_for_rank),
            ("toR", False, lambda it: int(bool(getattr(it, "timed_out", False))) if hasattr(it, "timed_out") else None),
            ("nATR", True, lambda it: getattr(it, "n_accT", None)),
            ("wATR", True, lambda it: getattr(it, "w_accT", None)),
            ("nDR", False, lambda it: getattr(it, "n_dags", None)),
            ("shdR", False, lambda it: getattr(it, "shd", None)),
            ("F1R", True, lambda it: getattr(it, "f1", None)),
            ("adjR", True, lambda it: getattr(it, "adj_f1", None)),
            ("ahR", True, lambda it: getattr(it, "ah_f1", None)),
            ("factR", True, lambda it: getattr(it, "fact_f1", None)),
            ("cshdR", False, lambda it: getattr(it, "cshd", None)),
            ("cF1R", True, lambda it: getattr(it, "cf1", None)),
            ("cadjR", True, lambda it: getattr(it, "cadj_f1", None)),
            ("cahR", True, lambda it: getattr(it, "cah_f1", None)),
        ]

        ranks_acc: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {
            m: {name: [] for (name, _hib, _g) in metric_defs} for m in methods
        }

        for rep in reps_sorted:
            for name, higher_is_better, getter in metric_defs:
                vals: dict[tuple[str, str, str, str, str], Any] = {}
                for m in methods:
                    it = method_rep.get(m, {}).get(rep)
                    if it is None:
                        vals[m] = None
                        continue
                    try:
                        vals[m] = getter(it)
                    except Exception:
                        vals[m] = None
                rm = _rank_map(vals, higher_is_better=higher_is_better)
                for m, rnk in rm.items():
                    ranks_acc[m][name].append(float(rnk))

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("nan")

        def _method_id(m: tuple[str, str, str, str, str]) -> str:
            return "|".join(m)

        ranks_avg_local: dict[str, Any] = {
            _method_id(m): {name: _mean(ranks_acc[m][name]) for (name, _hib, _g) in metric_defs}
            for m in methods
        }
        rank_names_local = [name for (name, _hib, _g) in metric_defs]
        return rank_names_local, ranks_avg_local

    if baseline_src_path is not None or wc_src_path is not None:
        rank_names, ranks_avg = _compute_ranks_from_rows()
    else:
        try:
            saved_ranks = _try_load_saved_metric_ranks(results_dir)
            if isinstance(saved_ranks, dict):
                ranks_avg_raw = saved_ranks.get("ranks_avg", {})
                if isinstance(ranks_avg_raw, dict):
                    ranks_avg = ranks_avg_raw
                metric_defs_raw = saved_ranks.get("metric_defs", []) or []
                if isinstance(metric_defs_raw, list):
                    for md in metric_defs_raw:
                        if not isinstance(md, dict):
                            continue
                        name = str(md.get("name", "") or "").strip()
                        if name:
                            rank_names.append(name)
        except Exception:
            rank_names = []
            ranks_avg = {}

    rank_fields: list[str] = [f"rank_{n}" for n in rank_names]

    def _method_id5(d: dict[str, Any]) -> str:
        return "|".join(
            (
                str(d.get("encoding") or ""),
                str(d.get("objective") or ""),
                str(d.get("opt_strategy") or ""),
                str(d.get("opt_mode") or ""),
                str(d.get("reification") or ""),
            )
        )

    def _method_id4(d: dict[str, Any]) -> str:
        return "|".join(
            (
                str(d.get("encoding") or ""),
                str(d.get("objective") or ""),
                str(d.get("opt_strategy") or ""),
                str(d.get("reification") or ""),
            )
        )

    def _rank_cols_for(method_row: dict[str, Any]) -> dict[str, Any]:
        if not rank_names or not ranks_avg:
            return {k: "" for k in rank_fields}
        mid5 = _method_id5(method_row)
        mid4 = _method_id4(method_row)
        row = ranks_avg.get(mid5, ranks_avg.get(mid4, {}))
        if not isinstance(row, dict):
            row = {}
        out: dict[str, Any] = {}
        for n in rank_names:
            out[f"rank_{n}"] = row.get(n, "")
        return out

    # Merged (baseline + WC).
    merged_fields = [
        # run config (for stacking across runs)
        "run_id",
        "run_src",
        "run_results_dir",
        "run_n_nodes",
        "run_seed",
        "run_reps",
        "run_timeout_sec",
        "run_pct_wrong_facts",
        "run_total_weight",
        "run_strategies",
        "run_objectives",
        "run_encodings",
        "run_opt_modes",
        "run_reifications",
        # ranking variables (optional; from metric_ranks.json)
        *rank_fields,
        # identifiers
        "rep",
        "kind",  # baseline|wc
        "encoding",
        "objective",
        "opt_strategy",
        "opt_mode",
        "reification",
        "solver",
        # run status/timing
        "status",
        "status_norm",
        "wall_time_sec",
        "build_time_sec",
        "solve_time_sec",
        "eval_time_sec",
        "method_time_sec",
        "timed_out",
        "graph_eval_timed_out",
        # baseline-only
        "removed",
        "models_after",
        # shared eval metrics
        "n_total",
        "n_true",
        "n_acc",
        "n_accT",
        "w_total",
        "w_true",
        "w_acc",
        "w_accT",
        "fact_f1",
        "n_dags",
        "true_in",
        "n_cpd",
        "true_cp_in",
        "shd",
        "f1",
        "adj_f1",
        "ah_f1",
        "shd_best",
        "shd_worst",
        "f1_best",
        "f1_worst",
        "adj_f1_best",
        "adj_f1_worst",
        "ah_f1_best",
        "ah_f1_worst",
        "cshd",
        "cf1",
        "cadj_f1",
        "cah_f1",
        "cshd_best",
        "cshd_worst",
        "cf1_best",
        "cf1_worst",
        "cadj_f1_best",
        "cadj_f1_worst",
        "cah_f1_best",
        "cah_f1_worst",
        # wc-only diagnostics
        "objective_costs",
        "cut_weight",
    ]

    merged_rows: list[dict[str, Any]] = []

    # Baseline rows: map into the same method-key shape used in tables.
    for r in baseline_rows:
        enc = ""
        if r.solver == "causalaba":
            enc = "base"
        elif r.solver == "causalaba_increm":
            enc = "inc"
        method_time_sec: Any = ""
        try:
            if r.build_time_sec is not None and r.solve_time_sec is not None and r.eval_time_sec is not None:
                method_time_sec = float(r.build_time_sec) + float(r.solve_time_sec) + float(r.eval_time_sec)
            elif r.build_time_sec is not None and r.solve_time_sec is not None:
                method_time_sec = float(r.build_time_sec) + float(r.solve_time_sec)
            else:
                method_time_sec = float(r.wall_time_sec)
        except Exception:
            method_time_sec = ""
        merged_rows.append(
            {
                **run_cfg,
                **_rank_cols_for(
                    {
                        "encoding": enc,
                        "objective": "-",
                        "opt_strategy": "baseline",
                        "opt_mode": r.opt_mode,
                        "reification": "-",
                    }
                ),
                "rep": r.rep,
                "kind": "baseline",
                "encoding": enc,
                "objective": "-",
                "opt_strategy": "baseline",
                "opt_mode": r.opt_mode,
                "reification": "-",
                "solver": r.solver,
                "status": "",
                "status_norm": "TIMEOUT" if bool(r.timed_out) else "",
                "wall_time_sec": r.wall_time_sec,
                "build_time_sec": r.build_time_sec,
                "solve_time_sec": r.solve_time_sec,
                "eval_time_sec": r.eval_time_sec,
                "method_time_sec": method_time_sec,
                "timed_out": r.timed_out,
                "graph_eval_timed_out": r.graph_eval_timed_out,
                "removed": r.removed,
                "models_after": r.models_after,
                "n_total": r.n_total,
                "n_true": r.n_true,
                "n_acc": r.n_acc,
                "n_accT": r.n_accT,
                "w_total": r.w_total,
                "w_true": r.w_true,
                "w_acc": r.w_acc,
                "w_accT": r.w_accT,
                "fact_f1": r.fact_f1,
                "n_dags": r.n_dags,
                "true_in": r.true_in,
                "n_cpd": r.n_cpd,
                "true_cp_in": r.true_cp_in,
                "shd": r.shd,
                "f1": r.f1,
                "adj_f1": r.adj_f1,
                "ah_f1": r.ah_f1,
                "shd_best": r.shd_best,
                "shd_worst": r.shd_worst,
                "f1_best": r.f1_best,
                "f1_worst": r.f1_worst,
                "adj_f1_best": r.adj_f1_best,
                "adj_f1_worst": r.adj_f1_worst,
                "ah_f1_best": r.ah_f1_best,
                "ah_f1_worst": r.ah_f1_worst,
                "cshd": r.cshd,
                "cf1": r.cf1,
                "cadj_f1": r.cadj_f1,
                "cah_f1": r.cah_f1,
                "cshd_best": r.cshd_best,
                "cshd_worst": r.cshd_worst,
                "cf1_best": r.cf1_best,
                "cf1_worst": r.cf1_worst,
                "cadj_f1_best": r.cadj_f1_best,
                "cadj_f1_worst": r.cadj_f1_worst,
                "cah_f1_best": r.cah_f1_best,
                "cah_f1_worst": r.cah_f1_worst,
                "objective_costs": "",
                "cut_weight": "",
            }
        )

    for r in wc_rows:
        method_time_sec: Any = ""
        try:
            if r.build_time_sec is not None and r.solve_time_sec is not None and r.eval_time_sec is not None:
                method_time_sec = float(r.build_time_sec) + float(r.solve_time_sec) + float(r.eval_time_sec)
            elif r.build_time_sec is not None and r.solve_time_sec is not None:
                method_time_sec = float(r.build_time_sec) + float(r.solve_time_sec)
            else:
                method_time_sec = float(r.wall_time_sec)
        except Exception:
            method_time_sec = ""
        merged_rows.append(
            {
                **run_cfg,
                **_rank_cols_for(
                    {
                        "encoding": r.encoding,
                        "objective": r.objective,
                        "opt_strategy": r.opt_strategy,
                        "opt_mode": r.opt_mode,
                        "reification": r.reification,
                    }
                ),
                "rep": r.rep,
                "kind": "wc",
                "encoding": r.encoding,
                "objective": r.objective,
                "opt_strategy": r.opt_strategy,
                "opt_mode": r.opt_mode,
                "reification": r.reification,
                "solver": "",
                "status": r.status,
                "status_norm": _norm_status(r.status, timed_out=bool(r.timed_out)),
                "wall_time_sec": r.wall_time_sec,
                "build_time_sec": r.build_time_sec,
                "solve_time_sec": r.solve_time_sec,
                "eval_time_sec": r.eval_time_sec,
                "method_time_sec": method_time_sec,
                "timed_out": r.timed_out,
                "graph_eval_timed_out": r.graph_eval_timed_out,
                "removed": "",
                "models_after": "",
                "n_total": r.n_total,
                "n_true": r.n_true,
                "n_acc": r.n_acc,
                "n_accT": r.n_accT,
                "w_total": r.w_total,
                "w_true": r.w_true,
                "w_acc": r.w_acc,
                "w_accT": r.w_accT,
                "fact_f1": r.fact_f1,
                "n_dags": r.n_dags,
                "true_in": r.true_in,
                "n_cpd": r.n_cpd,
                "true_cp_in": r.true_cp_in,
                "shd": r.shd,
                "f1": r.f1,
                "adj_f1": r.adj_f1,
                "ah_f1": r.ah_f1,
                "shd_best": r.shd_best,
                "shd_worst": r.shd_worst,
                "f1_best": r.f1_best,
                "f1_worst": r.f1_worst,
                "adj_f1_best": r.adj_f1_best,
                "adj_f1_worst": r.adj_f1_worst,
                "ah_f1_best": r.ah_f1_best,
                "ah_f1_worst": r.ah_f1_worst,
                "cshd": r.cshd,
                "cf1": r.cf1,
                "cadj_f1": r.cadj_f1,
                "cah_f1": r.cah_f1,
                "cshd_best": r.cshd_best,
                "cshd_worst": r.cshd_worst,
                "cf1_best": r.cf1_best,
                "cf1_worst": r.cf1_worst,
                "cadj_f1_best": r.cadj_f1_best,
                "cadj_f1_worst": r.cadj_f1_worst,
                "cah_f1_best": r.cah_f1_best,
                "cah_f1_worst": r.cah_f1_worst,
                "objective_costs": r.objective_costs,
                "cut_weight": r.cut_weight,
            }
        )

    merged_rows.sort(
        key=lambda d: (
            int(d.get("rep") or 0),
            str(d.get("kind") or ""),
            str(d.get("encoding") or ""),
            str(d.get("objective") or ""),
            str(d.get("opt_strategy") or ""),
            str(d.get("opt_mode") or ""),
            str(d.get("reification") or ""),
            str(d.get("solver") or ""),
        )
    )
    _write_csv(csv_path, rows=merged_rows, fieldnames=merged_fields)

    # Summary-tables CSV: one row per method group (baseline + wc) with
    # avg/min/max for key metrics, suitable for ranking/sorting.
    def _stats_fields(prefix: str, xs: Iterable[Any]) -> dict[str, Any]:
        avg, mn, mx = _stats(xs)
        return {
            f"{prefix}_avg": "" if not _is_number(avg) else avg,
            f"{prefix}_min": "" if not _is_number(mn) else mn,
            f"{prefix}_max": "" if not _is_number(mx) else mx,
        }

    def _group_row(
        *,
        kind: str,
        encoding: str,
        objective: str,
        opt_strategy: str,
        opt_mode: str,
        reification: str,
        solver: str,
        items: list[Any],
        use_method_time: bool,
    ) -> dict[str, Any]:
        timeouts = sum(1 for i in items if bool(getattr(i, "timed_out", False)))
        ge_to = sum(1 for i in items if bool(getattr(i, "graph_eval_timed_out", False)))
        # "g_ok" should reflect that graph-eval produced at least one compatible DAG
        # (i.e., actual metrics can be computed), not merely that the field exists.
        g_ok = 0
        for i in items:
            n_d = getattr(i, "n_dags", None)
            try:
                if n_d is not None and int(n_d) > 0:
                    g_ok += 1
            except Exception:
                continue

        # Runtime: for WC rows prefer (build+solve) where available.
        times: list[float] = []
        for i in items:
            try:
                if use_method_time and getattr(i, "build_time_sec", None) is not None and getattr(i, "solve_time_sec", None) is not None:
                    e = getattr(i, "eval_time_sec", None)
                    if e is not None:
                        times.append(float(getattr(i, "build_time_sec")) + float(getattr(i, "solve_time_sec")) + float(e))
                    else:
                        times.append(float(getattr(i, "build_time_sec")) + float(getattr(i, "solve_time_sec")))
                else:
                    times.append(float(getattr(i, "wall_time_sec")))
            except Exception:
                continue

        out: dict[str, Any] = {
            **run_cfg,
            **_rank_cols_for(
                {
                    "encoding": encoding,
                    "objective": objective,
                    "opt_strategy": opt_strategy,
                    "opt_mode": opt_mode,
                    "reification": reification,
                }
            ),
            "kind": kind,
            "encoding": encoding,
            "objective": objective,
            "opt_strategy": opt_strategy,
            "opt_mode": opt_mode,
            "reification": reification,
            "solver": solver,
            "runs": int(len(items)),
            "timeouts": int(timeouts),
            "ge_to": int(ge_to),
            "g_ok": int(g_ok),
        }
        out.update(_stats_fields("time_sec", times))

        # Build/solve times: only meaningful for WC rows, and only for non-timeout runs.
        build_vals: list[Any] = [getattr(i, "build_time_sec", None) for i in items if (not bool(getattr(i, "timed_out", False)))]
        solve_vals: list[Any] = [getattr(i, "solve_time_sec", None) for i in items if (not bool(getattr(i, "timed_out", False)))]
        out.update(_stats_fields("build_sec", [x for x in build_vals if x is not None]))
        out.update(_stats_fields("solve_sec", [x for x in solve_vals if x is not None]))
        eval_vals: list[Any] = [getattr(i, "eval_time_sec", None) for i in items if (not bool(getattr(i, "timed_out", False)))]
        out.update(_stats_fields("eval_sec", [x for x in eval_vals if x is not None]))

        # Acceptance and graph metrics: compute stats over available (non-missing) values.
        out.update(_stats_fields("n_total", [getattr(i, "n_total", None) for i in items]))
        out.update(_stats_fields("n_true", [getattr(i, "n_true", None) for i in items]))
        out.update(_stats_fields("n_acc", [getattr(i, "n_acc", None) for i in items]))
        out.update(_stats_fields("n_accT", [getattr(i, "n_accT", None) for i in items]))
        out.update(_stats_fields("w_total", [getattr(i, "w_total", None) for i in items]))
        out.update(_stats_fields("w_true", [getattr(i, "w_true", None) for i in items]))
        out.update(_stats_fields("w_acc", [getattr(i, "w_acc", None) for i in items]))
        out.update(_stats_fields("w_accT", [getattr(i, "w_accT", None) for i in items]))
        out.update(_stats_fields("fact_f1", [getattr(i, "fact_f1", None) for i in items]))

        out.update(_stats_fields("n_dags", [getattr(i, "n_dags", None) for i in items]))
        out.update(_stats_fields("shd", [getattr(i, "shd", None) for i in items]))
        out.update(_stats_fields("f1", [getattr(i, "f1", None) for i in items]))
        out.update(_stats_fields("adj_f1", [getattr(i, "adj_f1", None) for i in items]))
        out.update(_stats_fields("ah_f1", [getattr(i, "ah_f1", None) for i in items]))
        out.update(_stats_fields("shd_best", [getattr(i, "shd_best", None) for i in items]))
        out.update(_stats_fields("shd_worst", [getattr(i, "shd_worst", None) for i in items]))
        out.update(_stats_fields("f1_best", [getattr(i, "f1_best", None) for i in items]))
        out.update(_stats_fields("f1_worst", [getattr(i, "f1_worst", None) for i in items]))
        out.update(_stats_fields("adj_f1_best", [getattr(i, "adj_f1_best", None) for i in items]))
        out.update(_stats_fields("adj_f1_worst", [getattr(i, "adj_f1_worst", None) for i in items]))
        out.update(_stats_fields("ah_f1_best", [getattr(i, "ah_f1_best", None) for i in items]))
        out.update(_stats_fields("ah_f1_worst", [getattr(i, "ah_f1_worst", None) for i in items]))

        out.update(_stats_fields("cshd", [getattr(i, "cshd", None) for i in items]))
        out.update(_stats_fields("cf1", [getattr(i, "cf1", None) for i in items]))
        out.update(_stats_fields("cadj_f1", [getattr(i, "cadj_f1", None) for i in items]))
        out.update(_stats_fields("cah_f1", [getattr(i, "cah_f1", None) for i in items]))
        out.update(_stats_fields("cshd_best", [getattr(i, "cshd_best", None) for i in items]))
        out.update(_stats_fields("cshd_worst", [getattr(i, "cshd_worst", None) for i in items]))
        out.update(_stats_fields("cf1_best", [getattr(i, "cf1_best", None) for i in items]))
        out.update(_stats_fields("cf1_worst", [getattr(i, "cf1_worst", None) for i in items]))
        out.update(_stats_fields("cadj_f1_best", [getattr(i, "cadj_f1_best", None) for i in items]))
        out.update(_stats_fields("cadj_f1_worst", [getattr(i, "cadj_f1_worst", None) for i in items]))
        out.update(_stats_fields("cah_f1_best", [getattr(i, "cah_f1_best", None) for i in items]))
        out.update(_stats_fields("cah_f1_worst", [getattr(i, "cah_f1_worst", None) for i in items]))
        return out

    summary_fields: list[str] = [
        # run config (for stacking across runs)
        "run_id",
        "run_src",
        "run_results_dir",
        "run_n_nodes",
        "run_seed",
        "run_reps",
        "run_timeout_sec",
        "run_pct_wrong_facts",
        "run_total_weight",
        "run_strategies",
        "run_objectives",
        "run_encodings",
        "run_opt_modes",
        "run_reifications",
        # ranking variables (optional; from metric_ranks.json)
        *rank_fields,
        # group keys
        "kind",
        "encoding",
        "objective",
        "opt_strategy",
        "opt_mode",
        "reification",
        "solver",
        "runs",
        "timeouts",
        "ge_to",
        "g_ok",
        "time_sec_avg",
        "time_sec_min",
        "time_sec_max",
        "build_sec_avg",
        "build_sec_min",
        "build_sec_max",
        "solve_sec_avg",
        "solve_sec_min",
        "solve_sec_max",
        "eval_sec_avg",
        "eval_sec_min",
        "eval_sec_max",
        "n_total_avg",
        "n_total_min",
        "n_total_max",
        "n_true_avg",
        "n_true_min",
        "n_true_max",
        "n_acc_avg",
        "n_acc_min",
        "n_acc_max",
        "n_accT_avg",
        "n_accT_min",
        "n_accT_max",
        "w_total_avg",
        "w_total_min",
        "w_total_max",
        "w_true_avg",
        "w_true_min",
        "w_true_max",
        "w_acc_avg",
        "w_acc_min",
        "w_acc_max",
        "w_accT_avg",
        "w_accT_min",
        "w_accT_max",
        "fact_f1_avg",
        "fact_f1_min",
        "fact_f1_max",
        "n_dags_avg",
        "n_dags_min",
        "n_dags_max",
        "shd_avg",
        "shd_min",
        "shd_max",
        "f1_avg",
        "f1_min",
        "f1_max",
        "adj_f1_avg",
        "adj_f1_min",
        "adj_f1_max",
        "ah_f1_avg",
        "ah_f1_min",
        "ah_f1_max",
        "shd_best_avg",
        "shd_best_min",
        "shd_best_max",
        "shd_worst_avg",
        "shd_worst_min",
        "shd_worst_max",
        "f1_best_avg",
        "f1_best_min",
        "f1_best_max",
        "f1_worst_avg",
        "f1_worst_min",
        "f1_worst_max",
        "adj_f1_best_avg",
        "adj_f1_best_min",
        "adj_f1_best_max",
        "adj_f1_worst_avg",
        "adj_f1_worst_min",
        "adj_f1_worst_max",
        "ah_f1_best_avg",
        "ah_f1_best_min",
        "ah_f1_best_max",
        "ah_f1_worst_avg",
        "ah_f1_worst_min",
        "ah_f1_worst_max",
        "cshd_avg",
        "cshd_min",
        "cshd_max",
        "cf1_avg",
        "cf1_min",
        "cf1_max",
        "cadj_f1_avg",
        "cadj_f1_min",
        "cadj_f1_max",
        "cah_f1_avg",
        "cah_f1_min",
        "cah_f1_max",
        "cshd_best_avg",
        "cshd_best_min",
        "cshd_best_max",
        "cshd_worst_avg",
        "cshd_worst_min",
        "cshd_worst_max",
        "cf1_best_avg",
        "cf1_best_min",
        "cf1_best_max",
        "cf1_worst_avg",
        "cf1_worst_min",
        "cf1_worst_max",
        "cadj_f1_best_avg",
        "cadj_f1_best_min",
        "cadj_f1_best_max",
        "cadj_f1_worst_avg",
        "cadj_f1_worst_min",
        "cadj_f1_worst_max",
        "cah_f1_best_avg",
        "cah_f1_best_min",
        "cah_f1_best_max",
        "cah_f1_worst_avg",
        "cah_f1_worst_min",
        "cah_f1_worst_max",
    ]

    summary_rows: list[dict[str, Any]] = []
    # Baseline groups (one per solver)
    by_solver: dict[str, list[_BaselineRow]] = {}
    for r in baseline_rows:
        by_solver.setdefault(r.solver, []).append(r)
    for solver, items in sorted(by_solver.items()):
        enc = ""
        if solver == "causalaba":
            enc = "base"
        elif solver == "causalaba_increm":
            enc = "inc"
        summary_rows.append(
            _group_row(
                kind="baseline",
                encoding=enc,
                objective="-",
                opt_strategy="baseline",
                opt_mode="-",
                reification="-",
                solver=solver,
                items=items,
                use_method_time=True,
            )
        )

    # WC groups
    grouped: dict[tuple[str, str, str, str, str], list[_WcRow]] = {}
    for r in wc_rows:
        key = (r.encoding, r.objective, r.opt_strategy, r.opt_mode, r.reification)
        grouped.setdefault(key, []).append(r)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        summary_rows.append(
            _group_row(
                kind="wc",
                encoding=enc,
                objective=obj,
                opt_strategy=strat,
                opt_mode=optm,
                reification=reif,
                solver="",
                items=items,
                use_method_time=True,
            )
        )

    _write_csv(summary_csv_path, rows=summary_rows, fieldnames=summary_fields)

    # Provenance README (append one section per sweep so --glob can share an output dir).
    try:
        readme_path = out_dir / "readme_csv.md"
        is_new = (not readme_path.exists()) or (readme_path.stat().st_size == 0)
        with open(readme_path, "a") as f:
            if is_new:
                f.write("# WC sweep CSV exports\n\n")
                f.write("This file is generated by scripts/wc_sweep_report.py.\n\n")
            f.write(f"## {sweep_stem}\n\n")
            f.write(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}\n")
            if argv:
                f.write(f"- command: {argv}\n")
            f.write(f"- source: {src_path}\n")
            f.write(f"- results_dir: {results_dir}\n")
            if baseline_src_path and baseline_src_path != src_path:
                f.write(f"- baseline_source: {baseline_src_path}\n")
            if wc_src_path and wc_src_path != src_path:
                f.write(f"- wc_source: {wc_src_path}\n")
            f.write(f"- output_csv: {csv_path.name}\n")
            f.write(f"- output_summary_csv: {summary_csv_path.name}\n")
            f.write("\n")
            f.write("### Schema notes\n\n")
            f.write("- One row per rep/run (baseline and wc).\n")
            f.write("- The summary CSV contains one row per method-group with avg/min/max statistics for key columns.\n")
            f.write("- rank_* columns (if present) are loaded from metric_ranks.json (average ranks; lower is better).\n")
            f.write("- timed_out: solver timeout (solve phase only).\n")
            f.write("- graph_eval_timed_out: graph evaluation timeout (separate budget).\n")
            f.write("  - Blank means this field was not recorded in the source summary (legacy/unknown).\n")
            f.write("- kind: baseline|wc. Baseline rows use opt_strategy=baseline and objective/reification placeholders.\n")
            f.write("\n")
            f.write("### Columns\n\n")
            f.write(", ".join(merged_fields) + "\n\n")
            f.write("summary columns:\n\n")
            f.write(", ".join(summary_fields) + "\n\n")
    except Exception:
        pass


def _resolve_inputs(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.summary:
        paths.append(args.summary)
    if args.results_dir:
        paths.append(args.results_dir)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))

    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)

    missing_files: list[str] = []
    missing_dirs: list[str] = []
    for p in out:
        if os.path.isdir(p):
            if not os.path.isdir(p):
                missing_dirs.append(p)
        else:
            if not os.path.isfile(p):
                missing_files.append(p)

    if missing_dirs:
        raise FileNotFoundError("Missing results dir(s):\n  " + "\n  ".join(missing_dirs))
    if missing_files:
        raise FileNotFoundError("Missing summary.json path(s):\n  " + "\n  ".join(missing_files))

    if not out:
        raise ValueError("No inputs given. Use --results-dir, --summary, or --glob.")
    return out


def _render_partial_results_dir(
    results_dir: str,
) -> str:
    out_lines: list[str] = []

    def _cap_print(*a: Any, **k: Any) -> None:
        sep = k.get("sep", " ")
        end = k.get("end", "\n")
        out_lines.append(sep.join(str(x) for x in a) + end)

    global print  # type: ignore[no-redef]
    _real_print = print
    try:
        print = _cap_print  # type: ignore[assignment]

        _print_header("WC SWEEP REPORT (partial)")
        print(f"results_dir={os.path.abspath(results_dir)}")

        cfg_path = os.path.join(results_dir, "0.Config")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg_txt = f.read().strip()
                if cfg_txt:
                    print("-- 0.Config --")
                    print(cfg_txt)
            except Exception:
                pass

        # Progress scan.
        try:
            entries = os.listdir(results_dir)
        except Exception:
            entries = []

        rep_dirs: list[int] = []
        for e in entries:
            if e.startswith("rep") and e[3:].isdigit() and os.path.isdir(os.path.join(results_dir, e)):
                rep_dirs.append(int(e[3:]))
        rep_dirs.sort()

        # Try to infer expected counts from 0.Config (best-effort).
        expected_wc_per_rep: int | None = None
        expected_abapc_per_rep: int | None = None
        try:
            import ast

            cfg_txt = ""
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg_txt = f.read()
            cfg: dict[str, Any] = {}
            for line in (cfg_txt or "").splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
            encs = ast.literal_eval(cfg.get("encodings", "[]")) if cfg.get("encodings") else []
            objs = ast.literal_eval(cfg.get("objectives", "[]")) if cfg.get("objectives") else []
            reifs = ast.literal_eval(cfg.get("reifications", "[]")) if cfg.get("reifications") else []
            strats = ast.literal_eval(cfg.get("strategies", "[]")) if cfg.get("strategies") else []
            if isinstance(encs, list) and isinstance(objs, list) and isinstance(reifs, list) and isinstance(strats, list):
                expected_wc_per_rep = int(len(encs) * len(objs) * len(reifs) * len(strats))
                # baseline dumps: typically one base + one inc program per rep
                expected_abapc_per_rep = 2 if len(encs) >= 1 else None
        except Exception:
            expected_wc_per_rep = None
            expected_abapc_per_rep = None

        # Count LP files at top-level AND under rep*/ and rep*/lps/ (reorg layout).
        top_lp_files = [e for e in entries if e.endswith(".lp") and os.path.isfile(os.path.join(results_dir, e))]
        top_n_abapc = sum(1 for f in top_lp_files if f.startswith("abapc_"))
        top_n_wc = sum(1 for f in top_lp_files if f.startswith("wc_"))

        per_rep_scan: dict[int, dict[str, int]] = {}
        for rep in rep_dirs:
            rep_path = os.path.join(results_dir, f"rep{rep}")
            d = per_rep_scan.setdefault(rep, {"abapc": 0, "wc": 0, "facts": 0, "other": 0, "lp": 0})
            for sub in (".", "lps"):
                scan_dir = rep_path if sub == "." else os.path.join(rep_path, sub)
                if not os.path.isdir(scan_dir):
                    continue
                try:
                    for fn in os.listdir(scan_dir):
                        if not fn.endswith(".lp"):
                            continue
                        fp = os.path.join(scan_dir, fn)
                        if not os.path.isfile(fp):
                            continue
                        d["lp"] += 1
                        if fn.startswith("wc_"):
                            d["wc"] += 1
                        elif fn.startswith("abapc_"):
                            d["abapc"] += 1
                        elif fn.startswith("facts"):
                            d["facts"] += 1
                        else:
                            d["other"] += 1
                except Exception:
                    continue

        total_rep_abapc = sum(v.get("abapc", 0) for v in per_rep_scan.values())
        total_rep_wc = sum(v.get("wc", 0) for v in per_rep_scan.values())
        total_rep_lp = sum(v.get("lp", 0) for v in per_rep_scan.values())

        print("\nProgress")
        print(f"rep_dirs={rep_dirs} (n={len(rep_dirs)})")
        print(
            f"top_level_lp: total={len(top_lp_files)}, abapc={top_n_abapc}, wc={top_n_wc} | "
            f"under_rep_dirs: total={total_rep_lp}, abapc={total_rep_abapc}, wc={total_rep_wc}"
        )

        if per_rep_scan:
            print("\nPer-rep LP coverage (rep*/ and rep*/lps/)")
            hdr = f"{'rep':>4} {'abapc':>6} {'wc':>6} {'facts':>6} {'other':>6}"
            if expected_wc_per_rep is not None:
                hdr += f" {'wc/exp':>9}"
            if expected_abapc_per_rep is not None:
                hdr += f" {'ab/exp':>9}"
            print(hdr)
            for rep in rep_dirs:
                c = per_rep_scan.get(rep, {"abapc": 0, "wc": 0, "facts": 0, "other": 0})
                line = f"{rep:4d} {c.get('abapc',0):6d} {c.get('wc',0):6d} {c.get('facts',0):6d} {c.get('other',0):6d}"
                if expected_wc_per_rep is not None:
                    line += f" {c.get('wc',0):3d}/{expected_wc_per_rep:<5d}"
                if expected_abapc_per_rep is not None:
                    line += f" {c.get('abapc',0):3d}/{expected_abapc_per_rep:<5d}"
                print(line)

        print("\nNOTE: summary.json/summary.partial.json not found yet; this is a progress-only view.")
        print("      New sweeps write summary.partial.json incrementally; use the reporter again once it appears.")

    finally:
        print = _real_print  # type: ignore[assignment]

    return "".join(out_lines)


@dataclass(frozen=True)
class _BaselineRow:
    rep: int
    solver: str
    opt_mode: str
    removed: int
    models_after: int
    wall_time_sec: float
    build_time_sec: float | None
    solve_time_sec: float | None
    eval_time_sec: float | None
    timed_out: bool
    graph_eval_timed_out: bool | None
    n_total: int | None
    n_true: int | None
    n_acc: int | None
    n_accT: int | None
    w_total: int | None
    w_true: int | None
    w_acc: int | None
    w_accT: int | None
    fact_f1: float | None
    n_dags: int | None
    true_in: int | None
    n_cpd: int | None
    true_cp_in: int | None
    shd: float | None
    f1: float | None
    adj_f1: float | None
    ah_f1: float | None
    shd_best: float | None
    shd_worst: float | None
    f1_best: float | None
    f1_worst: float | None
    adj_f1_best: float | None
    adj_f1_worst: float | None
    ah_f1_best: float | None
    ah_f1_worst: float | None
    cshd: float | None
    cf1: float | None
    cadj_f1: float | None
    cah_f1: float | None
    cshd_best: float | None
    cshd_worst: float | None
    cf1_best: float | None
    cf1_worst: float | None
    cadj_f1_best: float | None
    cadj_f1_worst: float | None
    cah_f1_best: float | None
    cah_f1_worst: float | None


@dataclass(frozen=True)
class _WcRow:
    rep: int
    encoding: str
    objective: str
    opt_strategy: str
    opt_mode: str
    reification: str
    wall_time_sec: float
    build_time_sec: float | None
    solve_time_sec: float | None
    eval_time_sec: float | None
    timed_out: bool
    graph_eval_timed_out: bool | None
    status: str
    n_total: int | None
    n_true: int | None
    n_acc: int | None
    n_accT: int | None
    w_total: int | None
    w_true: int | None
    w_acc: int | None
    w_accT: int | None
    fact_f1: float | None
    n_dags: int | None
    true_in: int | None
    n_cpd: int | None
    true_cp_in: int | None
    shd: float | None
    f1: float | None
    adj_f1: float | None
    ah_f1: float | None
    shd_best: float | None
    shd_worst: float | None
    f1_best: float | None
    f1_worst: float | None
    adj_f1_best: float | None
    adj_f1_worst: float | None
    ah_f1_best: float | None
    ah_f1_worst: float | None
    cshd: float | None
    cf1: float | None
    cadj_f1: float | None
    cah_f1: float | None
    cshd_best: float | None
    cshd_worst: float | None
    cf1_best: float | None
    cf1_worst: float | None
    cadj_f1_best: float | None
    cadj_f1_worst: float | None
    cah_f1_best: float | None
    cah_f1_worst: float | None
    # Optional diagnostic fields (may be absent in legacy summaries).
    objective_costs: tuple[int, ...] | None = None
    cut_weight: int | None = None


def _parse_baseline_rows(d: dict[str, Any]) -> list[_BaselineRow]:
    rows = []
    for r in d.get("baseline_results", []) or []:
        rows.append(
            _BaselineRow(
                rep=int(r.get("rep", 0) or 0),
                solver=str(r.get("solver", "")),
                opt_mode=str(r.get("opt_mode", "optN") or "optN"),
                removed=int(r.get("removed", 0) or 0),
                models_after=int(r.get("models_after", 0) or 0),
                wall_time_sec=float(r.get("wall_time_sec", float("nan")) or float("nan")),
                build_time_sec=(float(r["build_time_sec"]) if r.get("build_time_sec") is not None else None),
                solve_time_sec=(float(r["solve_time_sec"]) if r.get("solve_time_sec") is not None else None),
                eval_time_sec=(float(r["eval_time_sec"]) if r.get("eval_time_sec") is not None else None),
                timed_out=bool(r.get("timed_out", False)),
                graph_eval_timed_out=(
                    bool(r.get("graph_eval_timed_out")) if r.get("graph_eval_timed_out") is not None else None
                ),
                n_total=(int(r["n_tests_total"]) if r.get("n_tests_total") is not None else None),
                n_true=(int(r["n_tests_true"]) if r.get("n_tests_true") is not None else None),
                n_acc=(int(r["n_tests_accepted"]) if r.get("n_tests_accepted") is not None else None),
                n_accT=(int(r["n_tests_accepted_true"]) if r.get("n_tests_accepted_true") is not None else None),
                w_total=(int(r["weight_total"]) if r.get("weight_total") is not None else None),
                w_true=(int(r["weight_true"]) if r.get("weight_true") is not None else None),
                w_acc=(int(r["weight_accepted"]) if r.get("weight_accepted") is not None else None),
                w_accT=(int(r["weight_accepted_true"]) if r.get("weight_accepted_true") is not None else None),
                fact_f1=(float(r["accepted_fact_f1"]) if r.get("accepted_fact_f1") is not None else None),
                n_dags=(int(r["n_dags_compat"]) if r.get("n_dags_compat") is not None else None),
                true_in=(int(r["true_dag_in_compat"]) if r.get("true_dag_in_compat") is not None else None),
                n_cpd=(int(r["n_cpdags_compat"]) if r.get("n_cpdags_compat") is not None else None),
                true_cp_in=(int(r["true_cpdag_in_compat"]) if r.get("true_cpdag_in_compat") is not None else None),
                shd=(float(r["shd_avg"]) if r.get("shd_avg") is not None else None),
                f1=(float(r["f1_avg"]) if r.get("f1_avg") is not None else None),
                adj_f1=(float(r["adjacency_f1_avg"]) if r.get("adjacency_f1_avg") is not None else None),
                ah_f1=(float(r["arrowhead_f1_avg"]) if r.get("arrowhead_f1_avg") is not None else None),
                shd_best=(float(r["shd_best"]) if r.get("shd_best") is not None else None),
                shd_worst=(float(r["shd_worst"]) if r.get("shd_worst") is not None else None),
                f1_best=(float(r["f1_best"]) if r.get("f1_best") is not None else None),
                f1_worst=(float(r["f1_worst"]) if r.get("f1_worst") is not None else None),
                adj_f1_best=(float(r["adjacency_f1_best"]) if r.get("adjacency_f1_best") is not None else None),
                adj_f1_worst=(float(r["adjacency_f1_worst"]) if r.get("adjacency_f1_worst") is not None else None),
                ah_f1_best=(float(r["arrowhead_f1_best"]) if r.get("arrowhead_f1_best") is not None else None),
                ah_f1_worst=(float(r["arrowhead_f1_worst"]) if r.get("arrowhead_f1_worst") is not None else None),
                cshd=(float(r["cpdag_shd_avg"]) if r.get("cpdag_shd_avg") is not None else None),
                cf1=(float(r["cpdag_f1_avg"]) if r.get("cpdag_f1_avg") is not None else None),
                cadj_f1=(float(r["cpdag_adjacency_f1_avg"]) if r.get("cpdag_adjacency_f1_avg") is not None else None),
                cah_f1=(float(r["cpdag_arrowhead_f1_avg"]) if r.get("cpdag_arrowhead_f1_avg") is not None else None),
                cshd_best=(float(r["cpdag_shd_best"]) if r.get("cpdag_shd_best") is not None else None),
                cshd_worst=(float(r["cpdag_shd_worst"]) if r.get("cpdag_shd_worst") is not None else None),
                cf1_best=(float(r["cpdag_f1_best"]) if r.get("cpdag_f1_best") is not None else None),
                cf1_worst=(float(r["cpdag_f1_worst"]) if r.get("cpdag_f1_worst") is not None else None),
                cadj_f1_best=(float(r["cpdag_adjacency_f1_best"]) if r.get("cpdag_adjacency_f1_best") is not None else None),
                cadj_f1_worst=(float(r["cpdag_adjacency_f1_worst"]) if r.get("cpdag_adjacency_f1_worst") is not None else None),
                cah_f1_best=(float(r["cpdag_arrowhead_f1_best"]) if r.get("cpdag_arrowhead_f1_best") is not None else None),
                cah_f1_worst=(float(r["cpdag_arrowhead_f1_worst"]) if r.get("cpdag_arrowhead_f1_worst") is not None else None),
            )
        )
    rows.sort(key=lambda x: (x.rep, x.solver, x.opt_mode))
    return rows


def _parse_wc_rows(d: dict[str, Any]) -> list[_WcRow]:
    rows = []
    for r in d.get("wc_results", []) or []:
        oc_raw = r.get("objective_costs")
        oc: tuple[int, ...] | None = None
        if isinstance(oc_raw, list):
            try:
                oc = tuple(int(x) for x in oc_raw)
            except Exception:
                oc = None
        rows.append(
            _WcRow(
                rep=int(r.get("rep", 0) or 0),
                encoding=str(r.get("encoding", "")),
                objective=str(r.get("objective", "")),
                opt_strategy=str(r.get("opt_strategy", "")),
                opt_mode=str(r.get("opt_mode", "optN") or "optN"),
                reification=str(r.get("reification", "")),
                wall_time_sec=float(r.get("wall_time_sec", float("nan")) or float("nan")),
                build_time_sec=(float(r["build_time_sec"]) if r.get("build_time_sec") is not None else None),
                solve_time_sec=(float(r["solve_time_sec"]) if r.get("solve_time_sec") is not None else None),
                eval_time_sec=(float(r["eval_time_sec"]) if r.get("eval_time_sec") is not None else None),
                timed_out=bool(r.get("timed_out", False)),
                graph_eval_timed_out=(
                    bool(r.get("graph_eval_timed_out")) if r.get("graph_eval_timed_out") is not None else None
                ),
                status=str(r.get("status", "")),
                n_total=(int(r["n_tests_total"]) if r.get("n_tests_total") is not None else None),
                n_true=(int(r["n_tests_true"]) if r.get("n_tests_true") is not None else None),
                n_acc=(int(r["n_tests_accepted"]) if r.get("n_tests_accepted") is not None else None),
                n_accT=(int(r["n_tests_accepted_true"]) if r.get("n_tests_accepted_true") is not None else None),
                w_total=(int(r["weight_total"]) if r.get("weight_total") is not None else None),
                w_true=(int(r["weight_true"]) if r.get("weight_true") is not None else None),
                w_acc=(int(r["weight_accepted"]) if r.get("weight_accepted") is not None else None),
                w_accT=(int(r["weight_accepted_true"]) if r.get("weight_accepted_true") is not None else None),
                fact_f1=(float(r["accepted_fact_f1"]) if r.get("accepted_fact_f1") is not None else None),
                n_dags=(int(r["n_dags_compat"]) if r.get("n_dags_compat") is not None else None),
                true_in=(int(r["true_dag_in_compat"]) if r.get("true_dag_in_compat") is not None else None),
                n_cpd=(int(r["n_cpdags_compat"]) if r.get("n_cpdags_compat") is not None else None),
                true_cp_in=(int(r["true_cpdag_in_compat"]) if r.get("true_cpdag_in_compat") is not None else None),
                shd=(float(r["shd_avg"]) if r.get("shd_avg") is not None else None),
                f1=(float(r["f1_avg"]) if r.get("f1_avg") is not None else None),
                adj_f1=(float(r["adjacency_f1_avg"]) if r.get("adjacency_f1_avg") is not None else None),
                ah_f1=(float(r["arrowhead_f1_avg"]) if r.get("arrowhead_f1_avg") is not None else None),
                shd_best=(float(r["shd_best"]) if r.get("shd_best") is not None else None),
                shd_worst=(float(r["shd_worst"]) if r.get("shd_worst") is not None else None),
                f1_best=(float(r["f1_best"]) if r.get("f1_best") is not None else None),
                f1_worst=(float(r["f1_worst"]) if r.get("f1_worst") is not None else None),
                adj_f1_best=(float(r["adjacency_f1_best"]) if r.get("adjacency_f1_best") is not None else None),
                adj_f1_worst=(float(r["adjacency_f1_worst"]) if r.get("adjacency_f1_worst") is not None else None),
                ah_f1_best=(float(r["arrowhead_f1_best"]) if r.get("arrowhead_f1_best") is not None else None),
                ah_f1_worst=(float(r["arrowhead_f1_worst"]) if r.get("arrowhead_f1_worst") is not None else None),
                cshd=(float(r["cpdag_shd_avg"]) if r.get("cpdag_shd_avg") is not None else None),
                cf1=(float(r["cpdag_f1_avg"]) if r.get("cpdag_f1_avg") is not None else None),
                cadj_f1=(float(r["cpdag_adjacency_f1_avg"]) if r.get("cpdag_adjacency_f1_avg") is not None else None),
                cah_f1=(float(r["cpdag_arrowhead_f1_avg"]) if r.get("cpdag_arrowhead_f1_avg") is not None else None),
                cshd_best=(float(r["cpdag_shd_best"]) if r.get("cpdag_shd_best") is not None else None),
                cshd_worst=(float(r["cpdag_shd_worst"]) if r.get("cpdag_shd_worst") is not None else None),
                cf1_best=(float(r["cpdag_f1_best"]) if r.get("cpdag_f1_best") is not None else None),
                cf1_worst=(float(r["cpdag_f1_worst"]) if r.get("cpdag_f1_worst") is not None else None),
                cadj_f1_best=(float(r["cpdag_adjacency_f1_best"]) if r.get("cpdag_adjacency_f1_best") is not None else None),
                cadj_f1_worst=(float(r["cpdag_adjacency_f1_worst"]) if r.get("cpdag_adjacency_f1_worst") is not None else None),
                cah_f1_best=(float(r["cpdag_arrowhead_f1_best"]) if r.get("cpdag_arrowhead_f1_best") is not None else None),
                cah_f1_worst=(float(r["cpdag_arrowhead_f1_worst"]) if r.get("cpdag_arrowhead_f1_worst") is not None else None),
                objective_costs=oc,
                cut_weight=(int(r["cut_weight"]) if r.get("cut_weight") is not None else None),
            )
        )
    rows.sort(key=lambda x: (x.rep, x.encoding, x.objective, x.opt_strategy, x.opt_mode, x.reification))
    return rows


def _norm_status(status: str, *, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    s = (status or "").strip().upper()
    if not s:
        return "UNKNOWN"
    if s.startswith("ERROR"):
        return "ERROR"
    if "UNSAT" in s:
        return "UNSAT"
    if "OPT" in s:
        return "OPT"
    if "SAT" in s:
        return "SAT"
    if "UNKNOWN" in s or "INTERRUPT" in s:
        return "UNKNOWN"
    return "UNKNOWN"


def _assert_direct_mus_opt_equivalence(
    wc_rows: list[_WcRow],
    *,
    total_weight: int | None,
    results_dir: Path,
    src_path: str,
) -> None:
    """Raise if any (direct,mus) pair is OPT in both but differs on objective cost.

    Uses objective_costs/cut_weight when available; otherwise falls back to
    cut_weight := total_weight - w_acc when possible.
    """

    by_key: dict[tuple[int, str, str, str, str], dict[str, _WcRow]] = {}
    for r in wc_rows:
        k = (int(r.rep), str(r.encoding), str(r.objective), str(r.opt_strategy), str(r.opt_mode))
        by_key.setdefault(k, {})[str(r.reification)] = r

    def _obj_sig(r: _WcRow) -> tuple[int, ...] | None:
        if r.objective_costs is not None:
            return tuple(int(x) for x in r.objective_costs)
        if r.cut_weight is not None:
            return (int(r.cut_weight),)
        if total_weight is not None and r.w_acc is not None:
            try:
                return (int(total_weight) - int(r.w_acc),)
            except Exception:
                return None
        return None

    hard: list[tuple[tuple[int, str, str, str, str], _WcRow, _WcRow]] = []
    soft: list[tuple[tuple[int, str, str, str, str], _WcRow, _WcRow, str, bool, bool]] = []
    soft_status_mismatches = 0
    soft_weight_mismatches = 0
    for k, d in sorted(by_key.items()):
        if "direct" not in d or "mus" not in d:
            continue
        r_dir = d["direct"]
        r_mus = d["mus"]

        s_dir = _norm_status(r_dir.status, timed_out=bool(r_dir.timed_out))
        s_mus = _norm_status(r_mus.status, timed_out=bool(r_mus.timed_out))

        sig_dir = _obj_sig(r_dir)
        sig_mus = _obj_sig(r_mus)

        # Hard mismatch: both claim optimality proven but costs disagree.
        if s_dir == "OPT" and s_mus == "OPT" and sig_dir != sig_mus:
            hard.append((k, r_dir, r_mus))
            continue

        # Soft mismatch: anything else where outcomes differ.
        status_diff = (s_dir != s_mus)
        w_diff = (
            (r_dir.w_acc is not None and r_mus.w_acc is not None)
            and (int(r_dir.w_acc) != int(r_mus.w_acc))
        )

        reasons: list[str] = []
        if status_diff:
            reasons.append(f"status {s_dir} vs {s_mus}")
        if w_diff:
            reasons.append(f"wA {r_dir.w_acc} vs {r_mus.w_acc}")
        if sig_dir is not None and sig_mus is not None and sig_dir != sig_mus:
            reasons.append(f"obj_sig {sig_dir} vs {sig_mus}")
        if reasons:
            if status_diff:
                soft_status_mismatches += 1
            if w_diff:
                soft_weight_mismatches += 1
            soft.append((k, r_dir, r_mus, "; ".join(reasons), status_diff, w_diff))

    if not hard and not soft:
        return

    def _resolve_wc_lp_path(
        root: Path,
        *,
        rep: int,
        encoding: str,
        reification: str,
        objective: str,
        opt_strategy: str,
        opt_mode: str,
    ) -> Path:
        fn = f"wc_{encoding}_{reification}_{objective}_{opt_strategy}_{opt_mode}_r{rep}.lp"
        candidates = [
            root / f"rep{rep}" / "lps" / fn,
            root / f"rep{rep}" / fn,
            root / fn,
        ]
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                pass
        return candidates[0]

    lines: list[str] = []
    lines.append("\n[assert-direct-mus-opt-eq] direct vs mus divergence report")
    lines.append(f"source={src_path}")
    lines.append(f"results_dir={results_dir}")
    lines.append(f"total_weight={total_weight}")
    lines.append(f"hard_mismatches={len(hard)}  (OPT/OPT but objective differs)")
    lines.append(f"soft_mismatches={len(soft)}  (any divergence outside OPT/OPT cost mismatch)")
    lines.append(f"soft_status_mismatches={soft_status_mismatches}  (status differs)")
    lines.append(f"soft_weight_mismatches={soft_weight_mismatches}  (accepted_weight differs)")
    lines.append("\nEach row shows the expected .lp files to inspect:")

    if hard:
        lines.append("\n=== HARD (OPT/OPT objective mismatch) ===")
        for (rep, enc, obj, strat, optm), r_dir, r_mus in hard:
            sig_dir = _obj_sig(r_dir)
            sig_mus = _obj_sig(r_mus)
            lp_dir = _resolve_wc_lp_path(
                results_dir,
                rep=rep,
                encoding=enc,
                reification="direct",
                objective=obj,
                opt_strategy=strat,
                opt_mode=optm,
            )
            lp_mus = _resolve_wc_lp_path(
                results_dir,
                rep=rep,
                encoding=enc,
                reification="mus",
                objective=obj,
                opt_strategy=strat,
                opt_mode=optm,
            )
            lines.append(
                "\n"
                + f"rep={rep} enc={enc} obj={obj} strategy={strat} optm={optm}\n"
                + f"  direct: status=OPT wA={r_dir.w_acc} sig={sig_dir}\n"
                + f"  mus:    status=OPT wA={r_mus.w_acc} sig={sig_mus}\n"
                + f"  lp_direct={lp_dir}\n"
                + f"  lp_mus   ={lp_mus}"
            )

    if soft:
        lines.append("\n=== SOFT (printing only weight mismatches) ===")
        soft_print = [t for t in soft if bool(t[5])]
        if not soft_print:
            lines.append("\n(no soft weight mismatches to print)")
        for (rep, enc, obj, strat, optm), r_dir, r_mus, reason, _sd, _wd in soft_print[:200]:
            # cap to avoid spewing thousands of lines
            sig_dir = _obj_sig(r_dir)
            sig_mus = _obj_sig(r_mus)
            s_dir = _norm_status(r_dir.status, timed_out=bool(r_dir.timed_out))
            s_mus = _norm_status(r_mus.status, timed_out=bool(r_mus.timed_out))
            lp_dir = _resolve_wc_lp_path(
                results_dir,
                rep=rep,
                encoding=enc,
                reification="direct",
                objective=obj,
                opt_strategy=strat,
                opt_mode=optm,
            )
            lp_mus = _resolve_wc_lp_path(
                results_dir,
                rep=rep,
                encoding=enc,
                reification="mus",
                objective=obj,
                opt_strategy=strat,
                opt_mode=optm,
            )
            lines.append(
                "\n"
                + f"rep={rep} enc={enc} obj={obj} strategy={strat} optm={optm}  ({reason})\n"
                + f"  direct: status={s_dir} to={int(bool(r_dir.timed_out))} wA={r_dir.w_acc} sig={sig_dir} time={r_dir.wall_time_sec:.3f}\n"
                + f"  mus:    status={s_mus} to={int(bool(r_mus.timed_out))} wA={r_mus.w_acc} sig={sig_mus} time={r_mus.wall_time_sec:.3f}\n"
                + f"  lp_direct={lp_dir}\n"
                + f"  lp_mus   ={lp_mus}"
            )
        if len(soft_print) > 200:
            lines.append(f"\n... truncated soft weight mismatches: showing 200 of {len(soft_print)}")

    raise SystemExit("\n".join(lines))


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_config(d: dict[str, Any], *, label: str, src_path: str) -> None:
    print(f"source={src_path}")
    for k in ("n_nodes", "seed", "reps", "timeout_sec", "pct_wrong_facts"):
        if k in d:
            print(f"{k}={d.get(k)}")
    for k in ("encodings", "objectives", "reifications", "strategies", "opt_modes"):
        if k in d:
            print(f"{k}={d.get(k)}")
    if d.get("wrong_fact_counts") is not None:
        print(f"wrong_fact_counts={d.get('wrong_fact_counts')}")


def _print_baseline_runs(rows: list[_BaselineRow]) -> None:
    if not rows:
        print("(no baseline_results)")
        return

    # Keep this compact and aligned so it compares well with WC runs.
    mode_w = 16

    def _bw_pair(best: float | None, worst: float | None, *, prec: int = 3, width: int = 13) -> str:
        if best is None or worst is None:
            return "".rjust(width)
        return f"[{float(best):.{prec}f},{float(worst):.{prec}f}]".rjust(width)

    header = (
        f"{'rep':>3} {'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'mode':<{mode_w}} "
        f"{'rm':>2} {'time':>7} {'to':>3} {'ge':>3} "
        f"{'nT':>3} {'nTr':>3} {'nA':>4} {'nAT':>4} "
        f"{'wA':>11} {'wAT':>11} {'nD':>4} {'shd':>7} {'F1':>6} {'adjF1':>7} {'ahF1':>7} "
        f"{'shd[b,w]':>13} {'F1[b,w]':>13} {'adj[b,w]':>13} {'ah[b,w]':>13} "
        f"{'cshd[b,w]':>13} {'cF1[b,w]':>13} {'cadj[b,w]':>13} {'cah[b,w]':>13}"
    )
    print(header)
    for r in rows:
        enc = "inc" if r.solver == "causalaba_increm" else "base"
        ge_to = (1 if bool(r.graph_eval_timed_out) else 0) if r.graph_eval_timed_out is not None else None
        shd_bw = _bw_pair(r.shd_best, r.shd_worst)
        f1_bw = _bw_pair(r.f1_best, r.f1_worst)
        adj_bw = _bw_pair(r.adj_f1_best, r.adj_f1_worst)
        ah_bw = _bw_pair(r.ah_f1_best, r.ah_f1_worst)
        cshd_bw = _bw_pair(r.cshd_best, r.cshd_worst)
        cf1_bw = _bw_pair(r.cf1_best, r.cf1_worst)
        cadj_bw = _bw_pair(r.cadj_f1_best, r.cadj_f1_worst)
        cah_bw = _bw_pair(r.cah_f1_best, r.cah_f1_worst)
        print(
            f"{r.rep:>3d} {enc:<4} {'-':<4} {'baseline':<12} {r.opt_mode:<4} {'-':<8} {r.solver:<{mode_w}} "
            f"{_fmt_int(r.removed,2)} {r.wall_time_sec:7.3f} {int(r.timed_out):>3d} {_fmt_int(ge_to,3)} "
            f"{_fmt_int(r.n_total,3)} {_fmt_int(r.n_true,3)} {_fmt_int(r.n_acc,4)} {_fmt_int(r.n_accT,4)} "
            f"{_fmt_int(r.w_acc,11)} {_fmt_int(r.w_accT,11)} "
            f"{_fmt_int(r.n_dags,4)} {_fmt_float(r.shd,7,3)} {_fmt_float(r.f1,6,3)} {_fmt_float(r.adj_f1,7,3)} {_fmt_float(r.ah_f1,7,3)} "
            f"{shd_bw} {f1_bw} {adj_bw} {ah_bw} {cshd_bw} {cf1_bw} {cadj_bw} {cah_bw}"
        )


def _baseline_summary(rows: list[_BaselineRow]) -> None:
    if not rows:
        return

    # Group by solver + opt_mode (so opt vs optN baselines are visible)
    by_solver: dict[tuple[str, str], list[_BaselineRow]] = {}
    for r in rows:
        by_solver.setdefault((r.solver, r.opt_mode), []).append(r)

    def _enc_label(solver: str) -> str:
        return "base" if solver == "causalaba" else "inc"

    def _sort_key(k: tuple[str, str]) -> tuple[str, str]:
        solver, optm = k
        return (_enc_label(solver), str(optm))

    _print_header("Baseline summary (avg)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time_sec':>10} {'build':>10} {'solve':>10} {'eval':>10} {'timeouts':>9} "
        f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
        f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
    )

    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        t_avg, _, _ = _stats([i.wall_time_sec for i in items])
        build_avg, _, _ = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])
        solve_avg, _, _ = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])
        eval_avg, _, _ = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])
        timeouts = sum(1 for i in items if i.timed_out)
        n_total_avg, _, _ = _stats([i.n_total for i in items])
        n_true_avg, _, _ = _stats([i.n_true for i in items])
        n_acc_avg, _, _ = _stats([i.n_acc for i in items])
        n_accT_avg, _, _ = _stats([i.n_accT for i in items])

        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        w_acc_avg, _, _ = _stats(w_acc_vals)
        w_accT_avg, _, _ = _stats(w_accT_vals)

        n_dags_avg, _, _ = _stats([i.n_dags for i in items if i.n_dags is not None])
        shd_avg, _, _ = _stats([i.shd for i in items if i.shd is not None])
        f1_avg, _, _ = _stats([i.f1 for i in items if i.f1 is not None])
        adj_f1_avg, _, _ = _stats([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_avg, _, _ = _stats([i.ah_f1 for i in items if i.ah_f1 is not None])

        nd_disp = "" if not _is_number(n_dags_avg) else f"{n_dags_avg:0.1f}"
        shd_disp = "" if not _is_number(shd_avg) else f"{shd_avg:0.3f}"
        f1_disp = "" if not _is_number(f1_avg) else f"{f1_avg:0.3f}"
        adj_f1_disp = "" if not _is_number(adj_f1_avg) else f"{adj_f1_avg:0.3f}"
        ah_f1_disp = "" if not _is_number(ah_f1_avg) else f"{ah_f1_avg:0.3f}"

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{t_avg:>10.3f} {(build_avg if _is_number(build_avg) else 0.0):>10.3f} {(solve_avg if _is_number(solve_avg) else 0.0):>10.3f} {(eval_avg if _is_number(eval_avg) else 0.0):>10.3f} {timeouts:>9} "
            f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
            f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)) if w_accT_vals else '':>12} "
            f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
        )

    # Facts + CPDAG (avg)
    print("\n(facts + CPDAG) (avg)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'factF1':>10} {'inT':>6} {'inCP':>6} {'nCPD':>8} {'cshd':>10} {'cF1':>10} {'cadjF1':>10} {'cahF1':>10}"
    )

    def _disp(x: float, fmt: str) -> str:
        return "" if not _is_number(x) else format(float(x), fmt)

    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        fact_avg, _, _ = _stats([i.fact_f1 for i in items if i.fact_f1 is not None])
        inT_avg, _, _ = _stats([float(i.true_in) for i in items if i.true_in is not None])
        inCP_avg, _, _ = _stats([float(i.true_cp_in) for i in items if i.true_cp_in is not None])
        ncpd_avg, _, _ = _stats([float(i.n_cpd) for i in items if i.n_cpd is not None])
        cshd_avg, _, _ = _stats([i.cshd for i in items if i.cshd is not None])
        cf1_avg, _, _ = _stats([i.cf1 for i in items if i.cf1 is not None])
        cadj_avg, _, _ = _stats([i.cadj_f1 for i in items if i.cadj_f1 is not None])
        cah_avg, _, _ = _stats([i.cah_f1 for i in items if i.cah_f1 is not None])
        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_disp(fact_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f')):>6} {(_disp(inCP_avg,'0.3f')):>6} {(_disp(ncpd_avg,'0.1f')):>8} "
            f"{_disp(cshd_avg,'0.3f'):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
        )

    _print_header("Baseline summary (std)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time_sd':>10} {'build_sd':>10} {'solve_sd':>10} {'eval_sd':>10} {'timeouts':>9} "
        f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
        f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        t_sd = _stdev([i.wall_time_sec for i in items])
        build_sd = _stdev([i.build_time_sec for i in items if i.build_time_sec is not None])
        solve_sd = _stdev([i.solve_time_sec for i in items if i.solve_time_sec is not None])
        eval_sd = _stdev([i.eval_time_sec for i in items if i.eval_time_sec is not None])
        timeouts = sum(1 for i in items if i.timed_out)
        n_total_sd = _stdev([i.n_total for i in items])
        n_true_sd = _stdev([i.n_true for i in items])
        n_acc_sd = _stdev([i.n_acc for i in items])
        n_accT_sd = _stdev([i.n_accT for i in items])

        w_acc_sd = _stdev([i.w_acc for i in items if i.w_acc is not None])
        w_accT_sd = _stdev([i.w_accT for i in items if i.w_accT is not None])
        n_dags_sd = _stdev([i.n_dags for i in items if i.n_dags is not None])
        shd_sd = _stdev([i.shd for i in items if i.shd is not None])
        f1_sd = _stdev([i.f1 for i in items if i.f1 is not None])
        adj_f1_sd = _stdev([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_sd = _stdev([i.ah_f1 for i in items if i.ah_f1 is not None])

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_float(t_sd,10,3)} {_fmt_float(build_sd,10,3)} {_fmt_float(solve_sd,10,3)} {_fmt_float(eval_sd,10,3)} {timeouts:>9d} "
            f"{_fmt_float(n_total_sd,10,1)} {_fmt_float(n_true_sd,10,1)} {_fmt_float(n_acc_sd,10,1)} {_fmt_float(n_accT_sd,10,1)} "
            f"{_fmt_float(w_acc_sd,12,1)} {_fmt_float(w_accT_sd,12,1)} "
            f"{_fmt_float(n_dags_sd,10,1)} {_fmt_float(shd_sd,8,3)} {_fmt_float(f1_sd,8,3)} {_fmt_float(adj_f1_sd,8,3)} {_fmt_float(ah_f1_sd,8,3)}"
        )

    print("\n(facts + CPDAG) (std)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'factF1_sd':>10} {'inT_sd':>8} {'inCP_sd':>8} {'nCPD_sd':>10} {'cshd_sd':>10} {'cF1_sd':>10} {'cadjF1_sd':>10} {'cahF1_sd':>10}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        fact_sd = _stdev([i.fact_f1 for i in items if i.fact_f1 is not None])
        inT_sd = _stdev([float(i.true_in) for i in items if i.true_in is not None])
        inCP_sd = _stdev([float(i.true_cp_in) for i in items if i.true_cp_in is not None])
        ncpd_sd = _stdev([float(i.n_cpd) for i in items if i.n_cpd is not None])
        cshd_sd = _stdev([i.cshd for i in items if i.cshd is not None])
        cf1_sd = _stdev([i.cf1 for i in items if i.cf1 is not None])
        cadj_sd = _stdev([i.cadj_f1 for i in items if i.cadj_f1 is not None])
        cah_sd = _stdev([i.cah_f1 for i in items if i.cah_f1 is not None])
        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_float(fact_sd,10,3)} {_fmt_float(inT_sd,8,3)} {_fmt_float(inCP_sd,8,3)} {_fmt_float(ncpd_sd,10,1)} "
            f"{_fmt_float(cshd_sd,10,3)} {_fmt_float(cf1_sd,10,3)} {_fmt_float(cadj_sd,10,3)} {_fmt_float(cah_sd,10,3)}"
        )

    _print_header("Baseline summary (min/max)")
    print("(times + counts)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time':>14} {'build':>14} {'solve':>14} {'eval':>14} {'to':>4} {'nT':>12} {'nTr':>12} {'nA':>12} {'nAT':>12}"
    )

    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        _, t_min, t_max = _stats([i.wall_time_sec for i in items])
        _, b_min, b_max = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])
        _, s_min, s_max = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])
        _, e_min, e_max = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])
        timeouts = sum(1 for i in items if i.timed_out)
        _, n_total_min, n_total_max = _stats([i.n_total for i in items])
        _, n_true_min, n_true_max = _stats([i.n_true for i in items])
        _, n_acc_min, n_acc_max = _stats([i.n_acc for i in items])
        _, n_accT_min, n_accT_max = _stats([i.n_accT for i in items])

        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        _, w_acc_min, w_acc_max = _stats(w_acc_vals)
        _, w_accT_min, w_accT_max = _stats(w_accT_vals)

        nd_vals = [i.n_dags for i in items if i.n_dags is not None]
        shd_vals = [i.shd for i in items if i.shd is not None]
        f1_vals = [i.f1 for i in items if i.f1 is not None]
        adj_f1_vals = [i.adj_f1 for i in items if i.adj_f1 is not None]
        ah_f1_vals = [i.ah_f1 for i in items if i.ah_f1 is not None]
        _, nd_min, nd_max = _stats(nd_vals)
        _, shd_min, shd_max = _stats(shd_vals)
        _, f1_min, f1_max = _stats(f1_vals)

        time_range = _fmt_range(t_min, t_max, 14, 3)
        build_range = _fmt_range(b_min, b_max, 14, 3)
        solve_range = _fmt_range(s_min, s_max, 14, 3)
        eval_range = _fmt_range(e_min, e_max, 14, 3)
        n_total_range = _fmt_range1(n_total_min, n_total_max, 12)
        n_true_range = _fmt_range1(n_true_min, n_true_max, 12)
        n_acc_range = _fmt_range1(n_acc_min, n_acc_max, 12)
        n_accT_range = _fmt_range1(n_accT_min, n_accT_max, 12)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{time_range} {build_range} {solve_range} {eval_range} {timeouts:>4d} {n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
        )

    print("\n(weights + graph)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'wA':>22} {'wAT':>22} {'nD':>12} {'shd':>14} {'F1':>12} {'adjF1':>12} {'ahF1':>12}"
    )

    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        timeouts = sum(1 for i in items if i.timed_out)

        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        _, w_acc_min, w_acc_max = _stats(w_acc_vals)
        _, w_accT_min, w_accT_max = _stats(w_accT_vals)
        w_acc_range = (f"[{w_acc_min:.0f},{w_acc_max:.0f}]".rjust(22) if w_acc_vals else "".rjust(22))
        w_accT_range = (f"[{w_accT_min:.0f},{w_accT_max:.0f}]".rjust(22) if w_accT_vals else "".rjust(22))

        nd_vals = [i.n_dags for i in items if i.n_dags is not None]
        shd_vals = [i.shd for i in items if i.shd is not None]
        f1_vals = [i.f1 for i in items if i.f1 is not None]
        adj_f1_vals = [i.adj_f1 for i in items if i.adj_f1 is not None]
        ah_f1_vals = [i.ah_f1 for i in items if i.ah_f1 is not None]
        _, nd_min, nd_max = _stats(nd_vals)
        _, shd_min, shd_max = _stats(shd_vals)
        _, f1_min, f1_max = _stats(f1_vals)
        _, adj_f1_min, adj_f1_max = _stats(adj_f1_vals)
        _, ah_f1_min, ah_f1_max = _stats(ah_f1_vals)
        nd_range = (f"[{nd_min:.1f},{nd_max:.1f}]".rjust(12) if nd_vals else "".rjust(12))
        shd_range = _fmt_range(shd_min, shd_max, 14, 3)
        f1_range = _fmt_range(f1_min, f1_max, 12, 3)
        adj_f1_range = _fmt_range(adj_f1_min, adj_f1_max, 12, 3)
        ah_f1_range = _fmt_range(ah_f1_min, ah_f1_max, 12, 3)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {'-':<8} {len(items):>6} "
            f"{w_acc_range} {w_accT_range} {nd_range} {shd_range} {f1_range} {adj_f1_range} {ah_f1_range}"
        )

    print("\n(facts + CPDAG)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'reif':<8} {'runs':>6} "
        f"{'factF1':>14} {'inT':>10} {'inCP':>10} {'nCPD':>12} {'cshd':>14} {'cF1':>12} {'cadjF1':>12} {'cahF1':>12}"
    )
    for solver in sorted(by_solver.keys(), key=_enc_label):
        items = by_solver[solver]
        fact_vals = [i.fact_f1 for i in items if i.fact_f1 is not None]
        true_in_vals = [float(i.true_in) for i in items if i.true_in is not None]
        true_cp_in_vals = [float(i.true_cp_in) for i in items if i.true_cp_in is not None]
        n_cpd_vals = [float(i.n_cpd) for i in items if i.n_cpd is not None]
        cshd_vals = [i.cshd for i in items if i.cshd is not None]
        cf1_vals = [i.cf1 for i in items if i.cf1 is not None]
        cadj_vals = [i.cadj_f1 for i in items if i.cadj_f1 is not None]
        cah_vals = [i.cah_f1 for i in items if i.cah_f1 is not None]

        _, fact_min, fact_max = _stats(fact_vals)
        _, ti_min, ti_max = _stats(true_in_vals)
        _, tci_min, tci_max = _stats(true_cp_in_vals)
        _, ncpd_min, ncpd_max = _stats(n_cpd_vals)
        _, cshd_min, cshd_max = _stats(cshd_vals)
        _, cf1_min, cf1_max = _stats(cf1_vals)
        _, cadj_min, cadj_max = _stats(cadj_vals)
        _, cah_min, cah_max = _stats(cah_vals)

        fact_range = _fmt_range(fact_min, fact_max, 14, 3)
        inT_range = (f"[{ti_min:.0f},{ti_max:.0f}]".rjust(10) if true_in_vals else "".rjust(10))
        inCP_range = (f"[{tci_min:.0f},{tci_max:.0f}]".rjust(10) if true_cp_in_vals else "".rjust(10))
        nCPD_range = (f"[{ncpd_min:.0f},{ncpd_max:.0f}]".rjust(12) if n_cpd_vals else "".rjust(12))
        cshd_range = _fmt_range(cshd_min, cshd_max, 14, 3)
        cf1_range = _fmt_range(cf1_min, cf1_max, 12, 3)
        cadj_range = _fmt_range(cadj_min, cadj_max, 12, 3)
        cah_range = _fmt_range(cah_min, cah_max, 12, 3)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {'-':<8} {len(items):>6} "
            f"{fact_range} {inT_range} {inCP_range} {nCPD_range} {cshd_range} {cf1_range} {cadj_range} {cah_range}"
        )

    print("\n(graph best/worst over compatible DAGs) (min/max over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
        f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        shd_best_vals = [i.shd_best for i in items if i.shd_best is not None]
        shd_worst_vals = [i.shd_worst for i in items if i.shd_worst is not None]
        f1_best_vals = [i.f1_best for i in items if i.f1_best is not None]
        f1_worst_vals = [i.f1_worst for i in items if i.f1_worst is not None]
        adj_best_vals = [i.adj_f1_best for i in items if i.adj_f1_best is not None]
        adj_worst_vals = [i.adj_f1_worst for i in items if i.adj_f1_worst is not None]
        ah_best_vals = [i.ah_f1_best for i in items if i.ah_f1_best is not None]
        ah_worst_vals = [i.ah_f1_worst for i in items if i.ah_f1_worst is not None]

        _, shd_best_min, shd_best_max = _stats(shd_best_vals)
        _, shd_worst_min, shd_worst_max = _stats(shd_worst_vals)
        _, f1_best_min, f1_best_max = _stats(f1_best_vals)
        _, f1_worst_min, f1_worst_max = _stats(f1_worst_vals)
        _, adj_best_min, adj_best_max = _stats(adj_best_vals)
        _, adj_worst_min, adj_worst_max = _stats(adj_worst_vals)
        _, ah_best_min, ah_best_max = _stats(ah_best_vals)
        _, ah_worst_min, ah_worst_max = _stats(ah_worst_vals)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_range(shd_best_min, shd_best_max, 14, 3)} {_fmt_range(shd_worst_min, shd_worst_max, 14, 3)} "
            f"{_fmt_range(f1_best_min, f1_best_max, 14, 3)} {_fmt_range(f1_worst_min, f1_worst_max, 14, 3)} "
            f"{_fmt_range(adj_best_min, adj_best_max, 14, 3)} {_fmt_range(adj_worst_min, adj_worst_max, 14, 3)} "
            f"{_fmt_range(ah_best_min, ah_best_max, 14, 3)} {_fmt_range(ah_worst_min, ah_worst_max, 14, 3)}"
        )

    print("\n(graph best/worst over compatible DAGs) (avg over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
        f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        shd_best_vals = [i.shd_best for i in items if i.shd_best is not None]
        shd_worst_vals = [i.shd_worst for i in items if i.shd_worst is not None]
        f1_best_vals = [i.f1_best for i in items if i.f1_best is not None]
        f1_worst_vals = [i.f1_worst for i in items if i.f1_worst is not None]
        adj_best_vals = [i.adj_f1_best for i in items if i.adj_f1_best is not None]
        adj_worst_vals = [i.adj_f1_worst for i in items if i.adj_f1_worst is not None]
        ah_best_vals = [i.ah_f1_best for i in items if i.ah_f1_best is not None]
        ah_worst_vals = [i.ah_f1_worst for i in items if i.ah_f1_worst is not None]

        shd_best_avg, _, _ = _stats(shd_best_vals)
        shd_worst_avg, _, _ = _stats(shd_worst_vals)
        f1_best_avg, _, _ = _stats(f1_best_vals)
        f1_worst_avg, _, _ = _stats(f1_worst_vals)
        adj_best_avg, _, _ = _stats(adj_best_vals)
        adj_worst_avg, _, _ = _stats(adj_worst_vals)
        ah_best_avg, _, _ = _stats(ah_best_vals)
        ah_worst_avg, _, _ = _stats(ah_worst_vals)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_float(shd_best_avg,14,3)} {_fmt_float(shd_worst_avg,14,3)} "
            f"{_fmt_float(f1_best_avg,14,3)} {_fmt_float(f1_worst_avg,14,3)} "
            f"{_fmt_float(adj_best_avg,14,3)} {_fmt_float(adj_worst_avg,14,3)} "
            f"{_fmt_float(ah_best_avg,14,3)} {_fmt_float(ah_worst_avg,14,3)}"
        )

    print("\n(CPDAG best/worst over compatible CPDAGs) (min/max over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
        f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        cshd_best_vals = [i.cshd_best for i in items if i.cshd_best is not None]
        cshd_worst_vals = [i.cshd_worst for i in items if i.cshd_worst is not None]
        cf1_best_vals = [i.cf1_best for i in items if i.cf1_best is not None]
        cf1_worst_vals = [i.cf1_worst for i in items if i.cf1_worst is not None]
        cadj_best_vals = [i.cadj_f1_best for i in items if i.cadj_f1_best is not None]
        cadj_worst_vals = [i.cadj_f1_worst for i in items if i.cadj_f1_worst is not None]
        cah_best_vals = [i.cah_f1_best for i in items if i.cah_f1_best is not None]
        cah_worst_vals = [i.cah_f1_worst for i in items if i.cah_f1_worst is not None]

        _, cshd_best_min, cshd_best_max = _stats(cshd_best_vals)
        _, cshd_worst_min, cshd_worst_max = _stats(cshd_worst_vals)
        _, cf1_best_min, cf1_best_max = _stats(cf1_best_vals)
        _, cf1_worst_min, cf1_worst_max = _stats(cf1_worst_vals)
        _, cadj_best_min, cadj_best_max = _stats(cadj_best_vals)
        _, cadj_worst_min, cadj_worst_max = _stats(cadj_worst_vals)
        _, cah_best_min, cah_best_max = _stats(cah_best_vals)
        _, cah_worst_min, cah_worst_max = _stats(cah_worst_vals)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_range(cshd_best_min, cshd_best_max, 14, 3)} {_fmt_range(cshd_worst_min, cshd_worst_max, 14, 3)} "
            f"{_fmt_range(cf1_best_min, cf1_best_max, 14, 3)} {_fmt_range(cf1_worst_min, cf1_worst_max, 14, 3)} "
            f"{_fmt_range(cadj_best_min, cadj_best_max, 14, 3)} {_fmt_range(cadj_worst_min, cadj_worst_max, 14, 3)} "
            f"{_fmt_range(cah_best_min, cah_best_max, 14, 3)} {_fmt_range(cah_worst_min, cah_worst_max, 14, 3)}"
        )

    print("\n(CPDAG best/worst over compatible CPDAGs) (avg over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
        f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
    )
    for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
        items = by_solver[(solver, optm)]
        cshd_best_vals = [i.cshd_best for i in items if i.cshd_best is not None]
        cshd_worst_vals = [i.cshd_worst for i in items if i.cshd_worst is not None]
        cf1_best_vals = [i.cf1_best for i in items if i.cf1_best is not None]
        cf1_worst_vals = [i.cf1_worst for i in items if i.cf1_worst is not None]
        cadj_best_vals = [i.cadj_f1_best for i in items if i.cadj_f1_best is not None]
        cadj_worst_vals = [i.cadj_f1_worst for i in items if i.cadj_f1_worst is not None]
        cah_best_vals = [i.cah_f1_best for i in items if i.cah_f1_best is not None]
        cah_worst_vals = [i.cah_f1_worst for i in items if i.cah_f1_worst is not None]

        cshd_best_avg, _, _ = _stats(cshd_best_vals)
        cshd_worst_avg, _, _ = _stats(cshd_worst_vals)
        cf1_best_avg, _, _ = _stats(cf1_best_vals)
        cf1_worst_avg, _, _ = _stats(cf1_worst_vals)
        cadj_best_avg, _, _ = _stats(cadj_best_vals)
        cadj_worst_avg, _, _ = _stats(cadj_worst_vals)
        cah_best_avg, _, _ = _stats(cah_best_vals)
        cah_worst_avg, _, _ = _stats(cah_worst_vals)

        print(
            f"{_enc_label(solver):<4} {'-':<4} {'baseline':<12} {str(optm):<4} {'-':<8} {len(items):>6} "
            f"{_fmt_float(cshd_best_avg,14,3)} {_fmt_float(cshd_worst_avg,14,3)} "
            f"{_fmt_float(cf1_best_avg,14,3)} {_fmt_float(cf1_worst_avg,14,3)} "
            f"{_fmt_float(cadj_best_avg,14,3)} {_fmt_float(cadj_worst_avg,14,3)} "
            f"{_fmt_float(cah_best_avg,14,3)} {_fmt_float(cah_worst_avg,14,3)}"
        )


def _print_wc_runs(rows: list[_WcRow]) -> None:
    if not rows:
        print("(no wc_results)")
        return

    def _bw_pair(best: float | None, worst: float | None, *, prec: int = 3, width: int = 13) -> str:
        if best is None or worst is None:
            return "".rjust(width)
        return f"[{float(best):.{prec}f},{float(worst):.{prec}f}]".rjust(width)

    # Keep this compact and aligned so it compares well with baseline runs.
    print(
        " rep enc  obj  strategy     optm reif     mode         rm    time   to  ge "
        " nT  nTr   nA  nAT         wA        wAT  nD    shd    F1  adjF1   ahF1"
        "      shd[b,w]       F1[b,w]      adj[b,w]       ah[b,w]     cshd[b,w]      cF1[b,w]     cadj[b,w]      cah[b,w]"
    )
    for r in rows:
        disp_status = _norm_status(r.status, timed_out=bool(r.timed_out))
        ge_to = (1 if bool(r.graph_eval_timed_out) else 0) if r.graph_eval_timed_out is not None else None
        run_time = r.wall_time_sec
        try:
            if r.build_time_sec is not None and r.solve_time_sec is not None:
                run_time = float(r.build_time_sec) + float(r.solve_time_sec)
        except Exception:
            run_time = r.wall_time_sec
        shd_bw = _bw_pair(r.shd_best, r.shd_worst)
        f1_bw = _bw_pair(r.f1_best, r.f1_worst)
        adj_bw = _bw_pair(r.adj_f1_best, r.adj_f1_worst)
        ah_bw = _bw_pair(r.ah_f1_best, r.ah_f1_worst)
        cshd_bw = _bw_pair(r.cshd_best, r.cshd_worst)
        cf1_bw = _bw_pair(r.cf1_best, r.cf1_worst)
        cadj_bw = _bw_pair(r.cadj_f1_best, r.cadj_f1_worst)
        cah_bw = _bw_pair(r.cah_f1_best, r.cah_f1_worst)

        print(
            f"{r.rep:4d} {r.encoding:<4} {r.objective:<4} {r.opt_strategy:<12} {r.opt_mode:<4} {r.reification:<8} {disp_status:<12} "
            f"{'':>2} {run_time:7.3f} {int(r.timed_out):>3d} {_fmt_int(ge_to,3)} "
            f"{_fmt_int(r.n_total,3)} {_fmt_int(r.n_true,3)} {_fmt_int(r.n_acc,4)} {_fmt_int(r.n_accT,4)} "
            f"{_fmt_int(r.w_acc,11)} {_fmt_int(r.w_accT,11)} "
            f"{_fmt_int(r.n_dags,3)} {_fmt_float(r.shd,6,3)} {_fmt_float(r.f1,5,3)} {_fmt_float(r.adj_f1,6,3)} {_fmt_float(r.ah_f1,6,3)} "
            f"{shd_bw} {f1_bw} {adj_bw} {ah_bw} {cshd_bw} {cf1_bw} {cadj_bw} {cah_bw}"
        )


def _wc_summary(
    rows: list[_WcRow],
    baseline_rows: list[_BaselineRow] | None = None,
    *,
    results_dir: Path | None = None,
    run_info: str | None = None,
    force_recompute_ranks: bool = False,
) -> None:
    if not rows:
        return

    grouped: dict[tuple[str, str, str, str, str], list[_WcRow]] = {}
    for r in rows:
        key = (r.encoding, r.objective, r.opt_strategy, r.opt_mode, r.reification)
        grouped.setdefault(key, []).append(r)

    def _iter_baseline_groups() -> list[tuple[str, str, str, str, str, list[_BaselineRow]]]:
        if not baseline_rows:
            return []
        by_solver: dict[tuple[str, str], list[_BaselineRow]] = {}
        for r in baseline_rows:
            by_solver.setdefault((r.solver, r.opt_mode), []).append(r)
        out: list[tuple[str, str, str, str, str, list[_BaselineRow]]] = []
        # Match the labels used in the sweep output, split by opt_mode.
        def _enc_label(solver: str) -> str:
            return "inc" if solver == "causalaba_increm" else "base"

        def _sort_key(k: tuple[str, str]) -> tuple[str, str]:
            solver, optm = k
            return (_enc_label(solver), str(optm))

        for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
            out.append((_enc_label(solver), "-", "baseline", optm, "-", by_solver[(solver, optm)]))
        return out

    _print_header("WC summary (avg)")
    if run_info:
        print(run_info)
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time_sec':>10} {'build':>10} {'solve':>10} {'timeouts':>9} {'ge_to':>6} {'g_ok':>6} "
        f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
        f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
    )

    # Prepend baseline rows for easy comparison.
    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        # Baselines: prefer solve-only time (build+solve) when present; otherwise fall back to wall_time_sec.
        method_times: list[float] = []
        for i in items:
            try:
                if i.build_time_sec is not None and i.solve_time_sec is not None:
                    method_times.append(float(i.build_time_sec) + float(i.solve_time_sec))
                else:
                    method_times.append(float(i.wall_time_sec))
            except Exception:
                continue
        t_avg, _, _ = _stats(method_times)
        b_avg, _, _ = _stats([i.build_time_sec for i in items if (i.build_time_sec is not None and (not i.timed_out))])
        s_avg, _, _ = _stats([i.solve_time_sec for i in items if (i.solve_time_sec is not None and (not i.timed_out))])
        timeouts = sum(1 for i in items if i.timed_out)
        ge_timeouts = sum(1 for i in items if bool(i.graph_eval_timed_out))
        graph_ok = 0
        for i in items:
            try:
                if i.n_dags is not None and int(i.n_dags) > 0:
                    graph_ok += 1
            except Exception:
                continue
        n_total_avg, _, _ = _stats([i.n_total for i in items])
        n_true_avg, _, _ = _stats([i.n_true for i in items])
        n_acc_avg, _, _ = _stats([i.n_acc for i in items])
        n_accT_avg, _, _ = _stats([i.n_accT for i in items])

        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        w_acc_avg, _, _ = _stats(w_acc_vals)
        w_accT_avg, _, _ = _stats(w_accT_vals)

        n_dags_avg, _, _ = _stats([i.n_dags for i in items if i.n_dags is not None])
        shd_avg, _, _ = _stats([i.shd for i in items if i.shd is not None])
        f1_avg, _, _ = _stats([i.f1 for i in items if i.f1 is not None])
        adj_f1_avg, _, _ = _stats([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_avg, _, _ = _stats([i.ah_f1 for i in items if i.ah_f1 is not None])

        nd_disp = "" if not _is_number(n_dags_avg) else f"{n_dags_avg:0.1f}"
        shd_disp = "" if not _is_number(shd_avg) else f"{shd_avg:0.3f}"
        f1_disp = "" if not _is_number(f1_avg) else f"{f1_avg:0.3f}"
        adj_f1_disp = "" if not _is_number(adj_f1_avg) else f"{adj_f1_avg:0.3f}"
        ah_f1_disp = "" if not _is_number(ah_f1_avg) else f"{ah_f1_avg:0.3f}"

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{t_avg:>10.3f} {b_avg:>10.3f} {s_avg:>10.3f} {timeouts:>9} {ge_timeouts:>6} {graph_ok:>6} "
            f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
            f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)) if w_accT_vals else '':>12} "
            f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
        )

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        # Report method runtime excluding graph-eval when build+solve are available.
        method_times: list[float] = []
        for i in items:
            try:
                if i.build_time_sec is not None and i.solve_time_sec is not None:
                    method_times.append(float(i.build_time_sec) + float(i.solve_time_sec))
                else:
                    method_times.append(float(i.wall_time_sec))
            except Exception:
                continue
        t_avg, _, _ = _stats(method_times)
        b_avg, _, _ = _stats([i.build_time_sec for i in items if (not i.timed_out)])
        s_avg, _, _ = _stats([i.solve_time_sec for i in items if (not i.timed_out)])
        timeouts = sum(1 for i in items if i.timed_out)
        ge_timeouts = sum(1 for i in items if bool(i.graph_eval_timed_out))
        graph_ok = 0
        for i in items:
            try:
                if i.n_dags is not None and int(i.n_dags) > 0:
                    graph_ok += 1
            except Exception:
                continue
        n_total_avg, _, _ = _stats([i.n_total for i in items])
        n_true_avg, _, _ = _stats([i.n_true for i in items])
        n_acc_avg, _, _ = _stats([i.n_acc for i in items])
        n_accT_avg, _, _ = _stats([i.n_accT for i in items])

        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        w_acc_avg, _, _ = _stats(w_acc_vals)
        w_accT_avg, _, _ = _stats(w_accT_vals)

        n_dags_avg, _, _ = _stats([i.n_dags for i in items if i.n_dags is not None])
        shd_avg, _, _ = _stats([i.shd for i in items if i.shd is not None])
        f1_avg, _, _ = _stats([i.f1 for i in items if i.f1 is not None])
        adj_f1_avg, _, _ = _stats([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_avg, _, _ = _stats([i.ah_f1 for i in items if i.ah_f1 is not None])

        nd_disp = "" if not _is_number(n_dags_avg) else f"{n_dags_avg:0.1f}"
        shd_disp = "" if not _is_number(shd_avg) else f"{shd_avg:0.3f}"
        f1_disp = "" if not _is_number(f1_avg) else f"{f1_avg:0.3f}"
        adj_f1_disp = "" if not _is_number(adj_f1_avg) else f"{adj_f1_avg:0.3f}"
        ah_f1_disp = "" if not _is_number(ah_f1_avg) else f"{ah_f1_avg:0.3f}"

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{t_avg:>10.3f} {b_avg:>10.3f} {s_avg:>10.3f} {timeouts:>9} {ge_timeouts:>6} {graph_ok:>6} "
            f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
            f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)) if w_accT_vals else '':>12} "
            f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
        )

    _print_header("WC summary (std)")
    if run_info:
        print(run_info)
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time_sd':>10} {'build_sd':>10} {'solve_sd':>10} {'timeouts':>9} {'ge_to':>6} {'g_ok':>6} "
        f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
        f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
    )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        method_times: list[float] = []
        for i in items:
            try:
                if i.build_time_sec is not None and i.solve_time_sec is not None:
                    method_times.append(float(i.build_time_sec) + float(i.solve_time_sec))
                else:
                    method_times.append(float(i.wall_time_sec))
            except Exception:
                continue
        t_sd = _stdev(method_times)
        b_sd = _stdev([i.build_time_sec for i in items if (i.build_time_sec is not None and (not i.timed_out))])
        s_sd = _stdev([i.solve_time_sec for i in items if (i.solve_time_sec is not None and (not i.timed_out))])
        timeouts = sum(1 for i in items if i.timed_out)
        ge_timeouts = sum(1 for i in items if bool(i.graph_eval_timed_out))
        graph_ok = sum(1 for i in items if i.n_dags is not None)

        n_total_sd = _stdev([i.n_total for i in items])
        n_true_sd = _stdev([i.n_true for i in items])
        n_acc_sd = _stdev([i.n_acc for i in items])
        n_accT_sd = _stdev([i.n_accT for i in items])
        w_acc_sd = _stdev([i.w_acc for i in items if i.w_acc is not None])
        w_accT_sd = _stdev([i.w_accT for i in items if i.w_accT is not None])
        n_dags_sd = _stdev([i.n_dags for i in items if i.n_dags is not None])
        shd_sd = _stdev([i.shd for i in items if i.shd is not None])
        f1_sd = _stdev([i.f1 for i in items if i.f1 is not None])
        adj_f1_sd = _stdev([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_sd = _stdev([i.ah_f1 for i in items if i.ah_f1 is not None])

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_float(t_sd,10,3)} {_fmt_float(b_sd,10,3)} {_fmt_float(s_sd,10,3)} {timeouts:>9d} {ge_timeouts:>6d} {graph_ok:>6d} "
            f"{_fmt_float(n_total_sd,10,1)} {_fmt_float(n_true_sd,10,1)} {_fmt_float(n_acc_sd,10,1)} {_fmt_float(n_accT_sd,10,1)} "
            f"{_fmt_float(w_acc_sd,12,1)} {_fmt_float(w_accT_sd,12,1)} { _fmt_float(n_dags_sd,10,1)} {_fmt_float(shd_sd,8,3)} {_fmt_float(f1_sd,8,3)} {_fmt_float(adj_f1_sd,8,3)} {_fmt_float(ah_f1_sd,8,3)}"
        )

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        method_times = []
        for i in items:
            try:
                if i.build_time_sec is not None and i.solve_time_sec is not None:
                    method_times.append(float(i.build_time_sec) + float(i.solve_time_sec))
                else:
                    method_times.append(float(i.wall_time_sec))
            except Exception:
                continue
        t_sd = _stdev(method_times)
        b_sd = _stdev([i.build_time_sec for i in items if (not i.timed_out)])
        s_sd = _stdev([i.solve_time_sec for i in items if (not i.timed_out)])
        timeouts = sum(1 for i in items if i.timed_out)
        ge_timeouts = sum(1 for i in items if bool(i.graph_eval_timed_out))
        graph_ok = sum(1 for i in items if i.n_dags is not None)

        n_total_sd = _stdev([i.n_total for i in items])
        n_true_sd = _stdev([i.n_true for i in items])
        n_acc_sd = _stdev([i.n_acc for i in items])
        n_accT_sd = _stdev([i.n_accT for i in items])
        w_acc_sd = _stdev([i.w_acc for i in items if i.w_acc is not None])
        w_accT_sd = _stdev([i.w_accT for i in items if i.w_accT is not None])
        n_dags_sd = _stdev([i.n_dags for i in items if i.n_dags is not None])
        shd_sd = _stdev([i.shd for i in items if i.shd is not None])
        f1_sd = _stdev([i.f1 for i in items if i.f1 is not None])
        adj_f1_sd = _stdev([i.adj_f1 for i in items if i.adj_f1 is not None])
        ah_f1_sd = _stdev([i.ah_f1 for i in items if i.ah_f1 is not None])

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_float(t_sd,10,3)} {_fmt_float(b_sd,10,3)} {_fmt_float(s_sd,10,3)} {timeouts:>9d} {ge_timeouts:>6d} {graph_ok:>6d} "
            f"{_fmt_float(n_total_sd,10,1)} {_fmt_float(n_true_sd,10,1)} {_fmt_float(n_acc_sd,10,1)} {_fmt_float(n_accT_sd,10,1)} "
            f"{_fmt_float(w_acc_sd,12,1)} {_fmt_float(w_accT_sd,12,1)} { _fmt_float(n_dags_sd,10,1)} {_fmt_float(shd_sd,8,3)} {_fmt_float(f1_sd,8,3)} {_fmt_float(adj_f1_sd,8,3)} {_fmt_float(ah_f1_sd,8,3)}"
        )

    # Facts + CPDAG (avg)
    print("\n(facts + CPDAG) (avg)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'factF1':>10} {'inT':>6} {'inCP':>6} {'nCPD':>8} {'cshd':>10} {'cF1':>10} {'cadjF1':>10} {'cahF1':>10}"
    )

    def _disp(x: float, fmt: str) -> str:
        return "" if not _is_number(x) else format(float(x), fmt)

    # Baselines first
    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        fact_avg, _, _ = _stats([i.fact_f1 for i in items if i.fact_f1 is not None])
        inT_avg, _, _ = _stats([float(i.true_in) for i in items if i.true_in is not None])
        inCP_avg, _, _ = _stats([float(i.true_cp_in) for i in items if i.true_cp_in is not None])
        ncpd_avg, _, _ = _stats([float(i.n_cpd) for i in items if i.n_cpd is not None])
        cshd_avg, _, _ = _stats([i.cshd for i in items if i.cshd is not None])
        cf1_avg, _, _ = _stats([i.cf1 for i in items if i.cf1 is not None])
        cadj_avg, _, _ = _stats([i.cadj_f1 for i in items if i.cadj_f1 is not None])
        cah_avg, _, _ = _stats([i.cah_f1 for i in items if i.cah_f1 is not None])
        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_disp(fact_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f')):>6} {(_disp(inCP_avg,'0.3f')):>6} {(_disp(ncpd_avg,'0.1f')):>8} "
            f"{_disp(cshd_avg,'0.3f'):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
        )

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        fact_avg, _, _ = _stats([i.fact_f1 for i in items if i.fact_f1 is not None])
        inT_avg, _, _ = _stats([float(i.true_in) for i in items if i.true_in is not None])
        inCP_avg, _, _ = _stats([float(i.true_cp_in) for i in items if i.true_cp_in is not None])
        ncpd_avg, _, _ = _stats([float(i.n_cpd) for i in items if i.n_cpd is not None])
        cshd_avg, _, _ = _stats([i.cshd for i in items if i.cshd is not None])
        cf1_avg, _, _ = _stats([i.cf1 for i in items if i.cf1 is not None])
        cadj_avg, _, _ = _stats([i.cadj_f1 for i in items if i.cadj_f1 is not None])
        cah_avg, _, _ = _stats([i.cah_f1 for i in items if i.cah_f1 is not None])
        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_disp(fact_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f')):>6} {(_disp(inCP_avg,'0.3f')):>6} {(_disp(ncpd_avg,'0.1f')):>8} "
            f"{_disp(cshd_avg,'0.3f'):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
        )

    print("\n(facts + CPDAG) (std)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'factF1_sd':>10} {'inT_sd':>8} {'inCP_sd':>8} {'nCPD_sd':>10} {'cshd_sd':>10} {'cF1_sd':>10} {'cadjF1_sd':>10} {'cahF1_sd':>10}"
    )

    def _print_cpdag_std_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        fact_sd = _stdev([it.fact_f1 for it in items if getattr(it, "fact_f1", None) is not None])
        inT_sd = _stdev([float(it.true_in) for it in items if getattr(it, "true_in", None) is not None])
        inCP_sd = _stdev([float(it.true_cp_in) for it in items if getattr(it, "true_cp_in", None) is not None])
        ncpd_sd = _stdev([float(it.n_cpd) for it in items if getattr(it, "n_cpd", None) is not None])
        cshd_sd = _stdev([it.cshd for it in items if getattr(it, "cshd", None) is not None])
        cf1_sd = _stdev([it.cf1 for it in items if getattr(it, "cf1", None) is not None])
        cadj_sd = _stdev([it.cadj_f1 for it in items if getattr(it, "cadj_f1", None) is not None])
        cah_sd = _stdev([it.cah_f1 for it in items if getattr(it, "cah_f1", None) is not None])
        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_float(fact_sd,10,3)} {_fmt_float(inT_sd,8,3)} {_fmt_float(inCP_sd,8,3)} {_fmt_float(ncpd_sd,10,1)} "
            f"{_fmt_float(cshd_sd,10,3)} {_fmt_float(cf1_sd,10,3)} {_fmt_float(cadj_sd,10,3)} {_fmt_float(cah_sd,10,3)}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_cpdag_std_row(enc, obj, strat, optm, reif, items)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_cpdag_std_row(enc, obj, strat, optm, reif, items)

    def _rank_map(
        method_to_value: dict[tuple[str, str, str, str, str], Any], *, higher_is_better: bool
    ) -> dict[tuple[str, str, str, str, str], float]:
        """Return per-method ranks (1..N, lower=better). Missing/non-numeric values get worst rank."""
        methods_all = list(method_to_value.keys())
        n_methods = len(methods_all)

        present: list[tuple[tuple[str, str, str, str, str], float]] = []
        for m, v in method_to_value.items():
            if _is_number(v):
                present.append((m, float(v)))
        present.sort(key=lambda kv: kv[1], reverse=higher_is_better)

        ranks: dict[tuple[str, str, str, str, str], float] = {m: float(n_methods) for m in methods_all}
        i = 0
        while i < len(present):
            j = i + 1
            while j < len(present) and present[j][1] == present[i][1]:
                j += 1
            rank_lo = i + 1
            rank_hi = j
            rank_avg = (rank_lo + rank_hi) / 2.0
            for k in range(i, j):
                ranks[present[k][0]] = float(rank_avg)
            i = j
        return ranks

    def _print_metric_ranks() -> None:
        saved_ranks = None
        if not force_recompute_ranks:
            saved_ranks = _try_load_saved_metric_ranks(results_dir) if results_dir is not None else None
        if saved_ranks is not None:
            methods_raw = saved_ranks.get("methods", []) or []
            ranks_avg = saved_ranks.get("ranks_avg", {}) or {}
            metric_defs_raw = saved_ranks.get("metric_defs", []) or []
            reps_raw = saved_ranks.get("reps", []) or []

            methods_saved: list[tuple[str, str, str, str, str]] = []
            for m in methods_raw:
                if not isinstance(m, dict):
                    continue
                enc = str(m.get("enc", ""))
                obj = str(m.get("obj", ""))
                strat = str(m.get("strategy", ""))
                optm = str(m.get("opt_mode", m.get("optm", "optN")) or "optN")
                reif = str(m.get("reif", ""))
                if enc and strat:
                    methods_saved.append((enc, obj, strat, optm, reif))

            metric_names: list[str] = []
            for md in metric_defs_raw:
                if not isinstance(md, dict):
                    continue
                name = str(md.get("name", "") or "").strip()
                if name:
                    metric_names.append(name)

            runs = 0
            if isinstance(reps_raw, list):
                runs = len(reps_raw)

            if methods_saved and metric_names and isinstance(ranks_avg, dict):
                _print_header("WC metric ranks (avg)  (lower is better)  [loaded metric_ranks.json]")
                print(
                    f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
                    + " ".join(f"{name:>6}" for name in metric_names)
                )
                for enc, obj, strat, optm, reif in methods_saved:
                    mid5 = "|".join((enc, obj, strat, optm, reif))
                    mid4 = "|".join((enc, obj, strat, reif))
                    row = ranks_avg.get(mid5, ranks_avg.get(mid4, {}))
                    out_fields: list[str] = []
                    for name in metric_names:
                        v = row.get(name, None) if isinstance(row, dict) else None
                        out_fields.append(f"{float(v):6.2f}" if _is_number(v) else f"{float('nan'):6.2f}")
                    print(f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {runs:>6d} " + " ".join(out_fields))
                return

        # Methods = baselines (if present) + each WC (enc,obj,strategy,opt_mode,reif)
        methods: list[tuple[str, str, str, str, str]] = []
        rep_set: set[int] = set()
        method_rep: dict[tuple[str, str, str, str, str], dict[int, Any]] = {}

        for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
            mkey = (enc, obj, strat, optm, reif)
            methods.append(mkey)
            md: dict[int, Any] = {}
            for it in items:
                md[int(it.rep)] = it
                rep_set.add(int(it.rep))
            method_rep[mkey] = md

        for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
            mkey = (enc, obj, strat, optm, reif)
            methods.append(mkey)
            md = method_rep.setdefault(mkey, {})
            for it in items:
                md[int(it.rep)] = it
                rep_set.add(int(it.rep))

        reps_sorted = sorted(rep_set)
        if not methods or not reps_sorted:
            return

        def _time_for_rank(it: Any) -> float | None:
            mt = getattr(it, "method_time_sec", None)
            if mt is not None:
                return mt
            b = getattr(it, "build_time_sec", None)
            s = getattr(it, "solve_time_sec", None)
            e = getattr(it, "eval_time_sec", None)
            if b is not None and s is not None and e is not None:
                return float(b) + float(s) + float(e)
            if b is not None and s is not None:
                return float(b) + float(s)
            return getattr(it, "wall_time_sec", None)

        metric_defs: list[tuple[str, bool, Any]] = [
            ("timeR", False, _time_for_rank),
            ("toR", False, lambda it: int(bool(getattr(it, "timed_out", False))) if hasattr(it, "timed_out") else None),
            ("nATR", True, lambda it: getattr(it, "n_accT", None)),
            ("wATR", True, lambda it: getattr(it, "w_accT", None)),
            ("nDR", False, lambda it: getattr(it, "n_dags", None)),
            ("shdR", False, lambda it: getattr(it, "shd", None)),
            ("F1R", True, lambda it: getattr(it, "f1", None)),
            ("adjR", True, lambda it: getattr(it, "adj_f1", None)),
            ("ahR", True, lambda it: getattr(it, "ah_f1", None)),
            ("factR", True, lambda it: getattr(it, "fact_f1", None)),
            ("cshdR", False, lambda it: getattr(it, "cshd", None)),
            ("cF1R", True, lambda it: getattr(it, "cf1", None)),
            ("cadjR", True, lambda it: getattr(it, "cadj_f1", None)),
            ("cahR", True, lambda it: getattr(it, "cah_f1", None)),
        ]

        ranks_acc: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {
            m: {name: [] for (name, _, _) in metric_defs} for m in methods
        }

        for rep in reps_sorted:
            for name, higher_is_better, getter in metric_defs:
                vals: dict[tuple[str, str, str, str, str], Any] = {}
                for m in methods:
                    it = method_rep.get(m, {}).get(rep, None)
                    v = None
                    if it is not None:
                        try:
                            v = getter(it)
                        except Exception:
                            v = None
                    vals[m] = v
                rm = _rank_map(vals, higher_is_better=higher_is_better)
                for m, rnk in rm.items():
                    ranks_acc[m][name].append(float(rnk))

        _print_header("WC metric ranks (avg)  (lower is better)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            + " ".join(f"{name:>6}" for (name, _, _) in metric_defs)
        )
        for m in methods:
            enc, obj, strat, optm, reif = m
            out_fields: list[str] = []
            for name, _, _ in metric_defs:
                xs = ranks_acc[m][name]
                out_fields.append(f"{(sum(xs)/len(xs)) if xs else float('nan'):6.2f}")
            print(f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(reps_sorted):>6} " + " ".join(out_fields))

    _print_metric_ranks()

    _print_header("WC summary (min/max)")
    print("(times + counts)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'time':>14} {'build':>14} {'solve':>14} {'to':>4} {'nT':>12} {'nTr':>12} {'nA':>12} {'nAT':>12}"
    )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _, t_min, t_max = _stats([i.wall_time_sec for i in items])
        timeouts = sum(1 for i in items if i.timed_out)
        _, n_total_min, n_total_max = _stats([i.n_total for i in items])
        _, n_true_min, n_true_max = _stats([i.n_true for i in items])
        _, n_acc_min, n_acc_max = _stats([i.n_acc for i in items])
        _, n_accT_min, n_accT_max = _stats([i.n_accT for i in items])

        time_range = _fmt_range(t_min, t_max, 14, 3)
        build_range = _fmt_range(0.0, 0.0, 14, 3)
        solve_range = _fmt_range(t_min, t_max, 14, 3)
        n_total_range = _fmt_range1(n_total_min, n_total_max, 12)
        n_true_range = _fmt_range1(n_true_min, n_true_max, 12)
        n_acc_range = _fmt_range1(n_acc_min, n_acc_max, 12)
        n_accT_range = _fmt_range1(n_accT_min, n_accT_max, 12)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{time_range} {build_range} {solve_range} {timeouts:>4d} "
            f"{n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
        )

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _, t_min, t_max = _stats([i.wall_time_sec for i in items])
        _, b_min, b_max = _stats([i.build_time_sec for i in items if (not i.timed_out)])
        _, s_min, s_max = _stats([i.solve_time_sec for i in items if (not i.timed_out)])
        timeouts = sum(1 for i in items if i.timed_out)
        _, n_total_min, n_total_max = _stats([i.n_total for i in items])
        _, n_true_min, n_true_max = _stats([i.n_true for i in items])
        _, n_acc_min, n_acc_max = _stats([i.n_acc for i in items])
        _, n_accT_min, n_accT_max = _stats([i.n_accT for i in items])

        time_range = _fmt_range(t_min, t_max, 14, 3)
        build_range = _fmt_range(b_min, b_max, 14, 3)
        solve_range = _fmt_range(s_min, s_max, 14, 3)
        n_total_range = _fmt_range1(n_total_min, n_total_max, 12)
        n_true_range = _fmt_range1(n_true_min, n_true_max, 12)
        n_acc_range = _fmt_range1(n_acc_min, n_acc_max, 12)
        n_accT_range = _fmt_range1(n_accT_min, n_accT_max, 12)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{time_range} {build_range} {solve_range} {timeouts:>4d} "
            f"{n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
        )

    print("\n(weights + graph)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'wA':>22} {'wAT':>22} {'nD':>12} {'shd':>14} {'F1':>12} {'adjF1':>12} {'ahF1':>12}"
    )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        _, w_acc_min, w_acc_max = _stats(w_acc_vals)
        _, w_accT_min, w_accT_max = _stats(w_accT_vals)
        w_acc_range = (f"[{w_acc_min:.0f},{w_acc_max:.0f}]".rjust(22) if w_acc_vals else "".rjust(22))
        w_accT_range = (f"[{w_accT_min:.0f},{w_accT_max:.0f}]".rjust(22) if w_accT_vals else "".rjust(22))

        nd_vals = [i.n_dags for i in items if i.n_dags is not None]
        shd_vals = [i.shd for i in items if i.shd is not None]
        f1_vals = [i.f1 for i in items if i.f1 is not None]
        adj_f1_vals = [i.adj_f1 for i in items if i.adj_f1 is not None]
        ah_f1_vals = [i.ah_f1 for i in items if i.ah_f1 is not None]
        _, nd_min, nd_max = _stats(nd_vals)
        _, shd_min, shd_max = _stats(shd_vals)
        _, f1_min, f1_max = _stats(f1_vals)
        _, adj_f1_min, adj_f1_max = _stats(adj_f1_vals)
        _, ah_f1_min, ah_f1_max = _stats(ah_f1_vals)
        nd_range = (f"[{nd_min:.1f},{nd_max:.1f}]".rjust(12) if nd_vals else "".rjust(12))
        shd_range = _fmt_range(shd_min, shd_max, 14, 3)
        f1_range = _fmt_range(f1_min, f1_max, 12, 3)
        adj_f1_range = _fmt_range(adj_f1_min, adj_f1_max, 12, 3)
        ah_f1_range = _fmt_range(ah_f1_min, ah_f1_max, 12, 3)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{w_acc_range} {w_accT_range} {nd_range} {shd_range} {f1_range} {adj_f1_range} {ah_f1_range}"
        )

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        w_acc_vals = [i.w_acc for i in items if i.w_acc is not None]
        w_accT_vals = [i.w_accT for i in items if i.w_accT is not None]
        _, w_acc_min, w_acc_max = _stats(w_acc_vals)
        _, w_accT_min, w_accT_max = _stats(w_accT_vals)
        w_acc_range = (f"[{w_acc_min:.0f},{w_acc_max:.0f}]".rjust(22) if w_acc_vals else "".rjust(22))
        w_accT_range = (f"[{w_accT_min:.0f},{w_accT_max:.0f}]".rjust(22) if w_accT_vals else "".rjust(22))

        nd_vals = [i.n_dags for i in items if i.n_dags is not None]
        shd_vals = [i.shd for i in items if i.shd is not None]
        f1_vals = [i.f1 for i in items if i.f1 is not None]
        adj_f1_vals = [i.adj_f1 for i in items if i.adj_f1 is not None]
        ah_f1_vals = [i.ah_f1 for i in items if i.ah_f1 is not None]
        _, nd_min, nd_max = _stats(nd_vals)
        _, shd_min, shd_max = _stats(shd_vals)
        _, f1_min, f1_max = _stats(f1_vals)
        _, adj_f1_min, adj_f1_max = _stats(adj_f1_vals)
        _, ah_f1_min, ah_f1_max = _stats(ah_f1_vals)
        nd_range = (f"[{nd_min:.1f},{nd_max:.1f}]".rjust(12) if nd_vals else "".rjust(12))
        shd_range = _fmt_range(shd_min, shd_max, 14, 3)
        f1_range = _fmt_range(f1_min, f1_max, 12, 3)
        adj_f1_range = _fmt_range(adj_f1_min, adj_f1_max, 12, 3)
        ah_f1_range = _fmt_range(ah_f1_min, ah_f1_max, 12, 3)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{w_acc_range} {w_accT_range} {nd_range} {shd_range} {f1_range} {adj_f1_range} {ah_f1_range}"
        )

    print("\n(facts + CPDAG)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'factF1':>14} {'inT':>10} {'inCP':>10} {'nCPD':>12} {'cshd':>14} {'cF1':>12} {'cadjF1':>12} {'cahF1':>12}"
    )

    def _print_cpdag_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        fact_vals = [it.fact_f1 for it in items if getattr(it, "fact_f1", None) is not None]
        true_in_vals = [float(it.true_in) for it in items if getattr(it, "true_in", None) is not None]
        true_cp_in_vals = [float(it.true_cp_in) for it in items if getattr(it, "true_cp_in", None) is not None]
        n_cpd_vals = [float(it.n_cpd) for it in items if getattr(it, "n_cpd", None) is not None]
        cshd_vals = [it.cshd for it in items if getattr(it, "cshd", None) is not None]
        cf1_vals = [it.cf1 for it in items if getattr(it, "cf1", None) is not None]
        cadj_vals = [it.cadj_f1 for it in items if getattr(it, "cadj_f1", None) is not None]
        cah_vals = [it.cah_f1 for it in items if getattr(it, "cah_f1", None) is not None]

        _, fact_min, fact_max = _stats(fact_vals)
        _, ti_min, ti_max = _stats(true_in_vals)
        _, tci_min, tci_max = _stats(true_cp_in_vals)
        _, ncpd_min, ncpd_max = _stats(n_cpd_vals)
        _, cshd_min, cshd_max = _stats(cshd_vals)
        _, cf1_min, cf1_max = _stats(cf1_vals)
        _, cadj_min, cadj_max = _stats(cadj_vals)
        _, cah_min, cah_max = _stats(cah_vals)

        fact_range = _fmt_range(fact_min, fact_max, 14, 3)
        inT_range = (f"[{ti_min:.0f},{ti_max:.0f}]".rjust(10) if true_in_vals else "".rjust(10))
        inCP_range = (f"[{tci_min:.0f},{tci_max:.0f}]".rjust(10) if true_cp_in_vals else "".rjust(10))
        nCPD_range = (f"[{ncpd_min:.0f},{ncpd_max:.0f}]".rjust(12) if n_cpd_vals else "".rjust(12))
        cshd_range = _fmt_range(cshd_min, cshd_max, 14, 3)
        cf1_range = _fmt_range(cf1_min, cf1_max, 12, 3)
        cadj_range = _fmt_range(cadj_min, cadj_max, 12, 3)
        cah_range = _fmt_range(cah_min, cah_max, 12, 3)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{fact_range} {inT_range} {inCP_range} {nCPD_range} {cshd_range} {cf1_range} {cadj_range} {cah_range}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_cpdag_row(enc, obj, strat, optm, reif, items)

    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_cpdag_row(enc, obj, strat, optm, reif, items)

    print("\n(graph best/worst over compatible DAGs) (min/max over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
        f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
    )

    def _print_dag_best_worst_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        shd_best_vals = [float(it.shd_best) for it in items if getattr(it, "shd_best", None) is not None]
        shd_worst_vals = [float(it.shd_worst) for it in items if getattr(it, "shd_worst", None) is not None]
        f1_best_vals = [float(it.f1_best) for it in items if getattr(it, "f1_best", None) is not None]
        f1_worst_vals = [float(it.f1_worst) for it in items if getattr(it, "f1_worst", None) is not None]
        adj_best_vals = [float(it.adj_f1_best) for it in items if getattr(it, "adj_f1_best", None) is not None]
        adj_worst_vals = [float(it.adj_f1_worst) for it in items if getattr(it, "adj_f1_worst", None) is not None]
        ah_best_vals = [float(it.ah_f1_best) for it in items if getattr(it, "ah_f1_best", None) is not None]
        ah_worst_vals = [float(it.ah_f1_worst) for it in items if getattr(it, "ah_f1_worst", None) is not None]

        _, shd_best_min, shd_best_max = _stats(shd_best_vals)
        _, shd_worst_min, shd_worst_max = _stats(shd_worst_vals)
        _, f1_best_min, f1_best_max = _stats(f1_best_vals)
        _, f1_worst_min, f1_worst_max = _stats(f1_worst_vals)
        _, adj_best_min, adj_best_max = _stats(adj_best_vals)
        _, adj_worst_min, adj_worst_max = _stats(adj_worst_vals)
        _, ah_best_min, ah_best_max = _stats(ah_best_vals)
        _, ah_worst_min, ah_worst_max = _stats(ah_worst_vals)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_range(shd_best_min, shd_best_max, 14, 3)} {_fmt_range(shd_worst_min, shd_worst_max, 14, 3)} "
            f"{_fmt_range(f1_best_min, f1_best_max, 14, 3)} {_fmt_range(f1_worst_min, f1_worst_max, 14, 3)} "
            f"{_fmt_range(adj_best_min, adj_best_max, 14, 3)} {_fmt_range(adj_worst_min, adj_worst_max, 14, 3)} "
            f"{_fmt_range(ah_best_min, ah_best_max, 14, 3)} {_fmt_range(ah_worst_min, ah_worst_max, 14, 3)}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_dag_best_worst_row(enc, obj, strat, optm, reif, items)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_dag_best_worst_row(enc, obj, strat, optm, reif, items)

    print("\n(graph best/worst over compatible DAGs) (avg over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
        f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
    )

    def _print_dag_best_worst_avg_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        shd_best_vals = [float(it.shd_best) for it in items if getattr(it, "shd_best", None) is not None]
        shd_worst_vals = [float(it.shd_worst) for it in items if getattr(it, "shd_worst", None) is not None]
        f1_best_vals = [float(it.f1_best) for it in items if getattr(it, "f1_best", None) is not None]
        f1_worst_vals = [float(it.f1_worst) for it in items if getattr(it, "f1_worst", None) is not None]
        adj_best_vals = [float(it.adj_f1_best) for it in items if getattr(it, "adj_f1_best", None) is not None]
        adj_worst_vals = [float(it.adj_f1_worst) for it in items if getattr(it, "adj_f1_worst", None) is not None]
        ah_best_vals = [float(it.ah_f1_best) for it in items if getattr(it, "ah_f1_best", None) is not None]
        ah_worst_vals = [float(it.ah_f1_worst) for it in items if getattr(it, "ah_f1_worst", None) is not None]

        shd_best_avg, _, _ = _stats(shd_best_vals)
        shd_worst_avg, _, _ = _stats(shd_worst_vals)
        f1_best_avg, _, _ = _stats(f1_best_vals)
        f1_worst_avg, _, _ = _stats(f1_worst_vals)
        adj_best_avg, _, _ = _stats(adj_best_vals)
        adj_worst_avg, _, _ = _stats(adj_worst_vals)
        ah_best_avg, _, _ = _stats(ah_best_vals)
        ah_worst_avg, _, _ = _stats(ah_worst_vals)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_float(shd_best_avg,14,3)} {_fmt_float(shd_worst_avg,14,3)} "
            f"{_fmt_float(f1_best_avg,14,3)} {_fmt_float(f1_worst_avg,14,3)} "
            f"{_fmt_float(adj_best_avg,14,3)} {_fmt_float(adj_worst_avg,14,3)} "
            f"{_fmt_float(ah_best_avg,14,3)} {_fmt_float(ah_worst_avg,14,3)}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_dag_best_worst_avg_row(enc, obj, strat, optm, reif, items)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_dag_best_worst_avg_row(enc, obj, strat, optm, reif, items)

    print("\n(CPDAG best/worst over compatible CPDAGs) (min/max over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
        f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
    )

    def _print_cpdag_best_worst_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        cshd_best_vals = [float(it.cshd_best) for it in items if getattr(it, "cshd_best", None) is not None]
        cshd_worst_vals = [float(it.cshd_worst) for it in items if getattr(it, "cshd_worst", None) is not None]
        cf1_best_vals = [float(it.cf1_best) for it in items if getattr(it, "cf1_best", None) is not None]
        cf1_worst_vals = [float(it.cf1_worst) for it in items if getattr(it, "cf1_worst", None) is not None]
        cadj_best_vals = [float(it.cadj_f1_best) for it in items if getattr(it, "cadj_f1_best", None) is not None]
        cadj_worst_vals = [float(it.cadj_f1_worst) for it in items if getattr(it, "cadj_f1_worst", None) is not None]
        cah_best_vals = [float(it.cah_f1_best) for it in items if getattr(it, "cah_f1_best", None) is not None]
        cah_worst_vals = [float(it.cah_f1_worst) for it in items if getattr(it, "cah_f1_worst", None) is not None]

        _, cshd_best_min, cshd_best_max = _stats(cshd_best_vals)
        _, cshd_worst_min, cshd_worst_max = _stats(cshd_worst_vals)
        _, cf1_best_min, cf1_best_max = _stats(cf1_best_vals)
        _, cf1_worst_min, cf1_worst_max = _stats(cf1_worst_vals)
        _, cadj_best_min, cadj_best_max = _stats(cadj_best_vals)
        _, cadj_worst_min, cadj_worst_max = _stats(cadj_worst_vals)
        _, cah_best_min, cah_best_max = _stats(cah_best_vals)
        _, cah_worst_min, cah_worst_max = _stats(cah_worst_vals)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_range(cshd_best_min, cshd_best_max, 14, 3)} {_fmt_range(cshd_worst_min, cshd_worst_max, 14, 3)} "
            f"{_fmt_range(cf1_best_min, cf1_best_max, 14, 3)} {_fmt_range(cf1_worst_min, cf1_worst_max, 14, 3)} "
            f"{_fmt_range(cadj_best_min, cadj_best_max, 14, 3)} {_fmt_range(cadj_worst_min, cadj_worst_max, 14, 3)} "
            f"{_fmt_range(cah_best_min, cah_best_max, 14, 3)} {_fmt_range(cah_worst_min, cah_worst_max, 14, 3)}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_cpdag_best_worst_row(enc, obj, strat, optm, reif, items)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_cpdag_best_worst_row(enc, obj, strat, optm, reif, items)

    print("\n(CPDAG best/worst over compatible CPDAGs) (avg over runs)")
    print(
        f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
        f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
        f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
    )

    def _print_cpdag_best_worst_avg_row(enc: str, obj: str, strat: str, optm: str, reif: str, items: list[Any]) -> None:
        cshd_best_vals = [float(it.cshd_best) for it in items if getattr(it, "cshd_best", None) is not None]
        cshd_worst_vals = [float(it.cshd_worst) for it in items if getattr(it, "cshd_worst", None) is not None]
        cf1_best_vals = [float(it.cf1_best) for it in items if getattr(it, "cf1_best", None) is not None]
        cf1_worst_vals = [float(it.cf1_worst) for it in items if getattr(it, "cf1_worst", None) is not None]
        cadj_best_vals = [float(it.cadj_f1_best) for it in items if getattr(it, "cadj_f1_best", None) is not None]
        cadj_worst_vals = [float(it.cadj_f1_worst) for it in items if getattr(it, "cadj_f1_worst", None) is not None]
        cah_best_vals = [float(it.cah_f1_best) for it in items if getattr(it, "cah_f1_best", None) is not None]
        cah_worst_vals = [float(it.cah_f1_worst) for it in items if getattr(it, "cah_f1_worst", None) is not None]

        cshd_best_avg, _, _ = _stats(cshd_best_vals)
        cshd_worst_avg, _, _ = _stats(cshd_worst_vals)
        cf1_best_avg, _, _ = _stats(cf1_best_vals)
        cf1_worst_avg, _, _ = _stats(cf1_worst_vals)
        cadj_best_avg, _, _ = _stats(cadj_best_vals)
        cadj_worst_avg, _, _ = _stats(cadj_worst_vals)
        cah_best_avg, _, _ = _stats(cah_best_vals)
        cah_worst_avg, _, _ = _stats(cah_worst_vals)

        print(
            f"{enc:<4} {obj:<4} {strat:<12} {optm:<4} {reif:<8} {len(items):>6} "
            f"{_fmt_float(cshd_best_avg,14,3)} {_fmt_float(cshd_worst_avg,14,3)} "
            f"{_fmt_float(cf1_best_avg,14,3)} {_fmt_float(cf1_worst_avg,14,3)} "
            f"{_fmt_float(cadj_best_avg,14,3)} {_fmt_float(cadj_worst_avg,14,3)} "
            f"{_fmt_float(cah_best_avg,14,3)} {_fmt_float(cah_worst_avg,14,3)}"
        )

    for enc, obj, strat, optm, reif, items in _iter_baseline_groups():
        _print_cpdag_best_worst_avg_row(enc, obj, strat, optm, reif, items)
    for (enc, obj, strat, optm, reif), items in sorted(grouped.items()):
        _print_cpdag_best_worst_avg_row(enc, obj, strat, optm, reif, items)

def _render_one(path: str, args: argparse.Namespace) -> str:
    def _resolve_summary_path(p: str) -> str:
        # Allow passing a results directory (in-progress runs may not have summary.json yet).
        if os.path.isdir(p):
            summary_path = os.path.join(p, "summary.json")
            partial_path = os.path.join(p, "summary.partial.json")
            if os.path.isfile(summary_path):
                return summary_path
            if os.path.isfile(partial_path):
                return partial_path
            # Caller will handle rendering a partial results dir view.
        return p

    # Allow passing a results directory (in-progress runs may not have summary.json yet).
    path = _resolve_summary_path(path)
    if os.path.isdir(path):
        return _render_partial_results_dir(path)

    d = _load_json(path)
    results_dir = Path(path).resolve().parent
    out_lines: list[str] = []

    baseline_rows: list[_BaselineRow] = []
    wc_rows: list[_WcRow] = []

    def _cap_print(*a: Any, **k: Any) -> None:
        sep = k.get("sep", " ")
        end = k.get("end", "\n")
        out_lines.append(sep.join(str(x) for x in a) + end)

    # Temporarily swap print function
    global print  # type: ignore[no-redef]
    _real_print = print
    try:
        print = _cap_print  # type: ignore[assignment]

        is_partial = os.path.basename(path) == "summary.partial.json"
        _print_header("WC SWEEP REPORT" + (" (partial snapshot)" if is_partial else ""))
        _print_config(d, label=os.path.basename(os.path.dirname(path)), src_path=path)
        if is_partial:
            # If present, show progress metadata.
            if d.get("reps_completed") is not None or d.get("reps_planned") is not None:
                print(f"progress: reps_completed={d.get('reps_completed')} / reps_planned={d.get('reps_planned')}")

        baseline_d: dict[str, Any] = d
        wc_d: dict[str, Any] = d

        baseline_src: str | None = None
        wc_src: str | None = None

        # Optional overrides for mixing sources.
        baseline_override = getattr(args, "baseline_results_dir", None) or getattr(args, "baseline_summary", None)
        wc_override = getattr(args, "wc_results_dir", None) or getattr(args, "wc_summary", None)

        if baseline_override:
            baseline_src = _resolve_summary_path(str(baseline_override))
            if os.path.isdir(baseline_src):
                print(f"[warn] baseline source dir has no summary.json yet: {baseline_src}")
                baseline_src = None
            else:
                baseline_d = _load_json(baseline_src)

        if wc_override:
            wc_src = _resolve_summary_path(str(wc_override))
            if os.path.isdir(wc_src):
                print(f"[warn] wc source dir has no summary.json yet: {wc_src}")
                wc_src = None
            else:
                wc_d = _load_json(wc_src)

        # Lightweight config sanity checks when mixing sources.
        def _cfg_get(dd: dict[str, Any], k: str) -> Any:
            return dd.get(k) if isinstance(dd, dict) else None

        def _assert_match(k: str) -> None:
            a = _cfg_get(d, k)
            b = _cfg_get(baseline_d, k)
            c = _cfg_get(wc_d, k)
            if baseline_override and (a is not None) and (b is not None) and (str(a) != str(b)):
                if k == "reps":
                    print(f"[warn] baseline override mismatch for reps: primary={a} baseline={b} (continuing)")
                else:
                    raise SystemExit(f"baseline override mismatch for {k}: primary={a} baseline={b}")
            if wc_override and (a is not None) and (c is not None) and (str(a) != str(c)):
                if k == "reps":
                    print(f"[warn] wc override mismatch for reps: primary={a} wc={c} (continuing)")
                else:
                    raise SystemExit(f"wc override mismatch for {k}: primary={a} wc={c}")

        for key in ("n_nodes", "seed", "reps", "pct_wrong_facts"):
            _assert_match(key)

        baseline_rows = _parse_baseline_rows(baseline_d)
        wc_rows = _parse_wc_rows(wc_d)

        if baseline_src or wc_src:
            print("sources:")
            if baseline_src:
                print(f"  baselines: {baseline_src}")
            if wc_src:
                print(f"  wc:        {wc_src}")

        if bool(getattr(args, "assert_direct_mus_opt_eq", False)):
            _assert_direct_mus_opt_equivalence(
                wc_rows,
                total_weight=(int(d["total_weight"]) if d.get("total_weight") is not None else None),
                results_dir=results_dir,
                src_path=path,
            )

        _print_header("Baseline removal runs")
        _print_baseline_runs(baseline_rows)
        _baseline_summary(baseline_rows)

        _print_header("WC opt-strategy sweep")
        _print_wc_runs(wc_rows)
        run_info = (
            f"run: n_nodes={d.get('n_nodes')}, seed={d.get('seed')}, reps={d.get('reps')}, "
            f"timeout_sec={d.get('timeout_sec')}, pct_wrong_facts={d.get('pct_wrong_facts')}"
        )
        force_recompute = bool(
            (baseline_src if "baseline_src" in locals() else None)
            or (wc_src if "wc_src" in locals() else None)
        )
        _wc_summary(
            wc_rows,
            baseline_rows,
            results_dir=results_dir,
            run_info=run_info,
            force_recompute_ranks=force_recompute,
        )

        # ------------------------------------------------------------------
        # Quick analysis block: WC improvements vs baseline and base vs inc.
        # ------------------------------------------------------------------
        _print_header("WC vs baseline deltas (quick)")
        bb_optm, bb_rows = _pick_baseline_group(baseline_rows, solver="causalaba")
        bi_optm, bi_rows = _pick_baseline_group(baseline_rows, solver="causalaba_increm")
        wb_key, wb_rows = _pick_wc_group(wc_rows, encoding="base")
        wi_key, wi_rows = _pick_wc_group(wc_rows, encoding="inc")
        print(f"baseline(base) opt_mode={bb_optm or 'NA'}")
        print(f"baseline(inc)  opt_mode={bi_optm or 'NA'}")
        print(f"WC(base) group: {_fmt_wc_group(wb_key)}")
        print(f"WC(inc)  group: {_fmt_wc_group(wi_key)}")

        def _summary_line(tag: str, rows: list[Any]) -> None:
            solve = _mean_key(rows, "solve_time_sec", exclude_timeouts=True)
            total = _mean_key(rows, "method_time_sec", exclude_timeouts=False)
            if not _is_number(total):
                total = _mean_key(rows, "wall_time_sec", exclude_timeouts=False)

            n_accT = _mean_key(rows, "n_accT", exclude_timeouts=False)
            if not _is_number(n_accT):
                n_accT = _mean_key(rows, "n_tests_accepted_true", exclude_timeouts=False)
            w_accT = _mean_key(rows, "w_accT", exclude_timeouts=False)
            if not _is_number(w_accT):
                w_accT = _mean_key(rows, "weight_accepted_true", exclude_timeouts=False)

            n_cpd = _mean_key(rows, "n_cpd", exclude_timeouts=False)
            if not _is_number(n_cpd):
                n_cpd = _mean_key(rows, "n_cpdags_compat", exclude_timeouts=False)
            cshd = _mean_key(rows, "cshd", exclude_timeouts=False)
            if not _is_number(cshd):
                cshd = _mean_key(rows, "cpdag_shd_avg", exclude_timeouts=False)
            cF1 = _mean_key(rows, "cf1", exclude_timeouts=False)
            if not _is_number(cF1):
                cF1 = _mean_key(rows, "cpdag_f1_avg", exclude_timeouts=False)
            cadj = _mean_key(rows, "cadj_f1", exclude_timeouts=False)
            if not _is_number(cadj):
                cadj = _mean_key(rows, "cpdag_adjacency_f1_avg", exclude_timeouts=False)

            print(
                f"{tag:<14} runs={len(rows):>2d} to={sum(1 for r in rows if bool(getattr(r,'timed_out',False))):>1d} "
                f"solve={solve:>8.3f}s total={total:>8.3f}s "
                f"n_accT={n_accT:>6.1f} w_accT={w_accT:>12.1f} "
                f"nCPD={n_cpd:>8.1f} cSHD={cshd:>7.3f} cF1={cF1:>6.3f} cadj={cadj:>6.3f}"
            )

        if bb_rows:
            _summary_line("baseline base", bb_rows)
        if bi_rows:
            _summary_line("baseline inc", bi_rows)
        if wb_rows:
            _summary_line("WC base", wb_rows)
        if wi_rows:
            _summary_line("WC inc", wi_rows)

        def _compare(label: str, a_rows: list[Any], b_rows: list[Any]) -> None:
            def _m(rows: list[Any], keys: str | tuple[str, ...], *, excl_to: bool) -> float:
                # Prefer parsed row-field names (e.g. n_accT, cshd) but allow
                # raw summary aliases for robustness.
                ks = (keys,) if isinstance(keys, str) else keys
                for k in ks:
                    v = _mean_key(rows, k, exclude_timeouts=excl_to)
                    if _is_number(v):
                        return v
                return float("nan")

            # Metrics (wc - baseline)
            pairs = [
                ("n_accT", ("n_accT", "n_tests_accepted_true"), 1, True),
                ("factF1", ("fact_f1", "accepted_fact_f1"), 3, True),
                ("w_accT", ("w_accT", "weight_accepted_true"), 1, True),
                ("nCPD", ("n_cpd", "n_cpdags_compat"), 1, False),
                ("cSHD", ("cshd", "cpdag_shd_avg"), 3, False),
                ("cF1", ("cf1", "cpdag_f1_avg"), 3, True),
                ("cadj", ("cadj_f1", "cpdag_adjacency_f1_avg"), 3, True),
                ("SHD_w", ("shd_worst",), 3, False),
                ("F1_w", ("f1_worst",), 3, True),
                ("adj_w", ("adj_f1_worst",), 3, True),
                ("ah_w", ("ah_f1_worst",), 3, True),
                ("cSHD_w", ("cshd_worst", "cpdag_shd_worst"), 3, False),
                ("cF1_w", ("cf1_worst", "cpdag_f1_worst"), 3, True),
                ("cadj_w", ("cadj_f1_worst", "cpdag_adjacency_f1_worst"), 3, True),
                ("cah_w", ("cah_f1_worst", "cpdag_arrowhead_f1_worst"), 3, True),
            ]

            print(f"\nWC − baseline ({label}):")
            for short, keys, prec, higher_better in pairs:
                a = _m(a_rows, keys, excl_to=False)
                b = _m(b_rows, keys, excl_to=False)
                da, pct = _delta(a, b)
                arrow = "↑" if higher_better else "↓"
                print(f"  {short:<5} ({arrow} better): {_fmt_delta(da, pct, prec=prec)}")

            # Solve-time delta and ratio
            a_s = _m(a_rows, "solve_time_sec", excl_to=True)
            b_s = _m(b_rows, "solve_time_sec", excl_to=True)
            if _is_number(a_s) and _is_number(b_s):
                da_s, pct_s = _delta(a_s, b_s)
                r = _ratio(a_s, b_s)
                r_s = f"x{r:0.2f}" if _is_number(r) else "NA"
                print(f"  solve (sec): {_fmt_delta(da_s, pct_s, prec=3)}  ratio={r_s}")
            else:
                # If solve-time isn't available (e.g. all timeouts), fall back to
                # overall time_sec and include timeout counts for context.
                a_t = _mean_time_sec(a_rows)
                b_t = _mean_time_sec(b_rows)
                da_t, pct_t = _delta(a_t, b_t)
                r = _ratio(a_t, b_t)
                r_s = f"x{r:0.2f}" if _is_number(r) else "NA"
                to_a = sum(1 for r0 in a_rows if bool(getattr(r0, "timed_out", False)))
                to_b = sum(1 for r0 in b_rows if bool(getattr(r0, "timed_out", False)))
                print(
                    f"  time_sec : {_fmt_delta(da_t, pct_t, prec=3)}  ratio={r_s}  "
                    f"timeouts: baseline={to_a}/{len(a_rows)} WC={to_b}/{len(b_rows)}"
                )

        if bb_rows and wb_rows:
            _compare("base", bb_rows, wb_rows)
        if bi_rows and wi_rows:
            _compare("inc", bi_rows, wi_rows)

        # Base vs inc timing comparisons (baseline and WC)
        if bb_rows and bi_rows:
            bb_s = _mean_key(bb_rows, "solve_time_sec", exclude_timeouts=True)
            bi_s = _mean_key(bi_rows, "solve_time_sec", exclude_timeouts=True)
            if _is_number(bb_s) and _is_number(bi_s):
                da, pct = _delta(bb_s, bi_s)
                r = _ratio(bb_s, bi_s)
                print(
                    f"\ninc − base baseline solve: {_fmt_delta(da, pct, prec=3)}  "
                    f"ratio={(f'x{r:0.2f}' if _is_number(r) else 'NA')}"
                )
            else:
                bb_t = _mean_time_sec(bb_rows)
                bi_t = _mean_time_sec(bi_rows)
                da, pct = _delta(bb_t, bi_t)
                r = _ratio(bb_t, bi_t)
                to_bb = sum(1 for r0 in bb_rows if bool(getattr(r0, "timed_out", False)))
                to_bi = sum(1 for r0 in bi_rows if bool(getattr(r0, "timed_out", False)))
                print(
                    f"\ninc − base baseline time_sec: {_fmt_delta(da, pct, prec=3)}  "
                    f"ratio={(f'x{r:0.2f}' if _is_number(r) else 'NA')}  "
                    f"timeouts: base={to_bb}/{len(bb_rows)} inc={to_bi}/{len(bi_rows)}"
                )

        if wb_rows and wi_rows:
            wb_s = _mean_key(wb_rows, "solve_time_sec", exclude_timeouts=True)
            wi_s = _mean_key(wi_rows, "solve_time_sec", exclude_timeouts=True)
            if _is_number(wb_s) and _is_number(wi_s):
                da, pct = _delta(wb_s, wi_s)
                r = _ratio(wb_s, wi_s)
                print(
                    f"inc − base WC solve:       {_fmt_delta(da, pct, prec=3)}  "
                    f"ratio={(f'x{r:0.2f}' if _is_number(r) else 'NA')}"
                )
            else:
                wb_t = _mean_time_sec(wb_rows)
                wi_t = _mean_time_sec(wi_rows)
                da, pct = _delta(wb_t, wi_t)
                r = _ratio(wb_t, wi_t)
                to_wb = sum(1 for r0 in wb_rows if bool(getattr(r0, "timed_out", False)))
                to_wi = sum(1 for r0 in wi_rows if bool(getattr(r0, "timed_out", False)))
                print(
                    f"inc − base WC time_sec:    {_fmt_delta(da, pct, prec=3)}  "
                    f"ratio={(f'x{r:0.2f}' if _is_number(r) else 'NA')}  "
                    f"timeouts: base={to_wb}/{len(wb_rows)} inc={to_wi}/{len(wi_rows)}"
                )

    finally:
        print = _real_print  # type: ignore[assignment]

    csv_out = getattr(args, "csv_out", None)
    if csv_out:
        out_dir = Path(str(csv_out)).expanduser()
        _export_csvs(
            csv_out_dir=out_dir,
            results_dir=results_dir,
            src_path=path,
            argv=" ".join(sys.argv),
            run_meta=d if isinstance(d, dict) else None,
            baseline_rows=baseline_rows,
            wc_rows=wc_rows,
            baseline_src_path=(baseline_src if 'baseline_src' in locals() else None),
            wc_src_path=(wc_src if 'wc_src' in locals() else None),
        )

    return "".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render run-level and summary tables from saved wc sweep summary.json")
    ap.add_argument("--results-dir", default=None, help="Path to results/wc_sweep_... folder containing summary.json")
    ap.add_argument("--summary", default=None, help="Path to summary.json")
    ap.add_argument("--glob", default=None, help="Glob for summary.json files (quote it), e.g. 'results/wc_sweep_*_*/summary.json'")
    ap.add_argument(
        "--baseline-results-dir",
        default=None,
        help=(
            "If set, load baseline_results from this results directory (its summary.json), while WC methods come from the primary --results-dir/--summary. "
            "Useful to rerun baselines only and combine reporting."
        ),
    )
    ap.add_argument(
        "--baseline-summary",
        default=None,
        help="If set, load baseline_results from this explicit summary.json path (overrides --baseline-results-dir).",
    )
    ap.add_argument(
        "--wc-results-dir",
        default=None,
        help=(
            "If set, load wc_results from this results directory (its summary.json), while baselines come from the primary --results-dir/--summary."
        ),
    )
    ap.add_argument(
        "--wc-summary",
        default=None,
        help="If set, load wc_results from this explicit summary.json path (overrides --wc-results-dir).",
    )
    ap.add_argument("--pager", action="store_true", help="Pipe output through a pager (less) for scrolling")
    ap.add_argument("--no-pager", action="store_true", help="Never use a pager")
    ap.add_argument(
        "--csv-out",
        default=None,
        help=(
            "If set, writes one combined per-run CSV per sweep into this directory (no per-sweep subdir), "
            "named like the results folder (e.g., wc_sweep_7_2004_20260207_105033.csv), plus readme_csv.md provenance."
        ),
    )
    ap.add_argument(
        "--assert-direct-mus-opt-eq",
        action="store_true",
        dest="assert_direct_mus_opt_eq",
        help=(
            "Exit non-zero if any (direct,mus) pair is OPT in both reifications but differs on objective cost. "
            "Prints the per-run wc_... .lp filenames to inspect in the results directory."
        ),
    )
    args = ap.parse_args()

    if args.glob and (args.baseline_results_dir or args.baseline_summary or args.wc_results_dir or args.wc_summary):
        raise SystemExit("Mixing-source overrides (--baseline-*/--wc-*) are not supported with --glob; run per-sweep instead.")

    if args.csv_out:
        # Avoid accidental overwrite when using a single CSV path.
        p = Path(str(args.csv_out)).expanduser()
        if p.suffix.lower() == ".csv":
            raise ValueError("--csv-out must be a directory path (not a .csv file).")

    paths = _resolve_inputs(args)
    text = "\n\n".join(_render_one(p, args) for p in paths)

    use_pager = bool(args.pager) and (not args.no_pager)
    if use_pager:
        import pydoc

        pydoc.pager(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
