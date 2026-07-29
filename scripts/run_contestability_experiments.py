#!/usr/bin/env python3
"""Run the empirical CI-fact contestability sidecar experiments.

For every CI fact released by the selected OptABA-PC optimum, this script
re-solves the emitted optimization program with that fact mandatory.  It never
modifies the canonical matched-baseline CSV/TeX files.

The original sweep summaries did not persist the selected fact identities.
Consequently, this runner first reconstructs and freezes a baseline witness
from each emitted LP, using the original clingo configuration, and validates
its accepted count and weight against summary.json.  The frozen witness and
all solver evidence are written to a JSONL sidecar so resumed runs do not pick
a different tied optimum.

Run this script from the pinned reproduction environment, which provides the
clingo Python package, for example:

  python scripts/run_contestability_experiments.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "paper_aaai2027"
    / "recomputed"
    / "contestability_optabapc"
)
DEFAULT_FACT_RECORDS = (
    REPO_ROOT
    / "results"
    / "paper_aaai2027"
    / "frozen"
    / "results"
    / "tables"
    / "paper_current_alpha001_nowrong_noweight_50rep_preview"
    / "matched_baseline_fact_records.csv"
)
DEFAULT_HANDOFF = REPO_ROOT / "results" / "paper_aaai2027" / "CONTESTABILITY_SCOPE.md"

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
DATASET_ORDER = {key: i for i, (key, _label) in enumerate(DATASET_SPECS)}
DATASET_LABELS = dict(DATASET_SPECS)
DEFAULT_DATASET_KEYS = {"cancer", "earthquake", "survey", "er5", "sf5"}
EIGHT_NODE_DATASET_KEYS = {"asia", "er8", "sf8"}
FROZEN_SEEDS = tuple(range(2026, 2076))

CHALLENGE_FIELDS = (
    "dataset",
    "seed",
    "fact_id",
    "fact_weight",
    "baseline_optimum_cost",
    "forced_retention_optimum_cost",
    "margin",
    "normalized_margin",
    "zero_margin",
    "solve_time_sec",
    "timeout",
    "optimality_proven",
    "optimality_evidence",
    # Extra diagnostic/provenance columns needed to count exceptional cases.
    "infeasible",
    "solver_status",
    "fact_atom",
    "fact_index",
    "source_lp",
)

FACT_RULE_RE = re.compile(
    r"^(?P<atom>ext_(?:indep|dep)\([^\n]+?\))\s*:-\s*mus\((?P<index>\d+)\)\s*\.\s*$"
)
WEIGHT_RE = re.compile(
    r"^__optimum_mcs_objective_literal__\(\s*(?P<weight>\d+)\s*,\s*mus\((?P<index>\d+)\)\s*\)\s*\.\s*$"
)


@dataclass(frozen=True)
class BaselineInstance:
    dataset_key: str
    dataset_label: str
    seed: int
    rep: int
    summary_path: Path
    lp_path: Path
    expected_total: int
    expected_accepted: int
    expected_accepted_weight: int
    expected_total_weight: int

    @property
    def expected_released(self) -> int:
        return self.expected_total - self.expected_accepted

    @property
    def expected_cost(self) -> int:
        return self.expected_total_weight - self.expected_accepted_weight


@dataclass(frozen=True)
class ParsedProgram:
    text: str
    facts: dict[int, str]
    weights: dict[int, int]
    sha256: str


@dataclass(frozen=True)
class SolveOutcome:
    status: str
    timed_out: bool
    infeasible: bool
    optimality_proven: bool
    solve_time_sec: float
    selected_indices: tuple[int, ...]
    objective_cost: int | None
    cost_vector: tuple[int, ...]
    models_seen: int
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _as_int(value: Any, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected integer {field}, got {value!r}") from None
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Expected integer {field}, got {value!r}")
    return int(number)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_source_path(value: str, *, relative_to: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def stable_fact_id(dataset_key: str, seed: int, fact_atom: str) -> str:
    """Return a readable, globally unique identifier stable across reruns."""

    atom = fact_atom.strip().rstrip(".")
    return f"{dataset_key}:{int(seed)}:{atom}"


def parse_emitted_program(path: Path) -> ParsedProgram:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    facts: dict[int, str] = {}
    weights: dict[int, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        fact_match = FACT_RULE_RE.match(line)
        if fact_match:
            index = int(fact_match.group("index"))
            atom = fact_match.group("atom")
            if index in facts and facts[index] != atom:
                raise ValueError(f"Conflicting fact definitions for mus({index}) in {path}")
            facts[index] = atom
            continue
        weight_match = WEIGHT_RE.match(line)
        if weight_match:
            index = int(weight_match.group("index"))
            weight = int(weight_match.group("weight"))
            if weight <= 0:
                raise ValueError(f"Non-positive release weight for mus({index}) in {path}")
            if index in weights and weights[index] != weight:
                raise ValueError(f"Conflicting weights for mus({index}) in {path}")
            weights[index] = weight

    if not facts:
        raise ValueError(f"No guarded ext_dep/ext_indep facts found in {path}")
    expected_indices = set(range(1, len(facts) + 1))
    if set(facts) != expected_indices:
        raise ValueError(f"Non-contiguous mus fact indices in {path}: {sorted(facts)}")
    if set(weights) != expected_indices:
        missing = sorted(expected_indices - set(weights))
        extra = sorted(set(weights) - expected_indices)
        raise ValueError(f"Fact/weight index mismatch in {path}; missing={missing}, extra={extra}")
    if len(set(facts.values())) != len(facts):
        raise ValueError(f"Duplicate CI fact atoms in {path}")
    return ParsedProgram(text=text, facts=facts, weights=weights, sha256=_sha256_bytes(raw))


def _canonical_wc_row(summary: dict[str, Any], rep: int) -> dict[str, Any]:
    matches = [
        row
        for row in (summary.get("wc_results", []) or [])
        if isinstance(row, dict)
        and _as_int(row.get("rep"), field="rep") == rep
        and row.get("encoding") == "inc"
        and row.get("objective") == "lex"
        and row.get("opt_strategy") == "bb"
        and row.get("opt_mode") == "optN"
        and row.get("reification") == "mus"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one canonical OptABA-PC row for rep {rep}; found {len(matches)}")
    return matches[0]


def load_inventory(
    fact_records_path: Path,
    *,
    dataset_keys: set[str],
    seed_start: int,
    seed_end: int,
) -> list[BaselineInstance]:
    if not fact_records_path.exists():
        raise FileNotFoundError(f"Matched fact records not found: {fact_records_path}")

    candidates: dict[tuple[str, int], dict[str, str]] = {}
    with fact_records_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "dataset_label", "seed", "rep_local", "source", "method"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {fact_records_path}: {sorted(missing)}")
        for row in reader:
            key = str(row.get("dataset", ""))
            if row.get("method") != "OptABA-PC" or key not in dataset_keys:
                continue
            seed = _as_int(row.get("seed"), field="seed")
            if seed < seed_start or seed > seed_end:
                continue
            identity = (key, seed)
            if identity in candidates:
                raise ValueError(f"Duplicate OptABA-PC fact record for {identity}")
            candidates[identity] = row

    inventory: list[BaselineInstance] = []
    for (dataset_key, seed), record in candidates.items():
        if seed not in FROZEN_SEEDS:
            raise ValueError(f"Seed {seed} is outside the frozen 2026..2075 range")
        expected_label = DATASET_LABELS[dataset_key]
        if record.get("dataset_label") != expected_label:
            raise ValueError(
                f"Dataset label mismatch for {dataset_key}: {record.get('dataset_label')!r} != {expected_label!r}"
            )
        rep = _as_int(record.get("rep_local"), field="rep_local")
        summary_path = _resolve_source_path(str(record.get("source", "")))
        if not summary_path.exists():
            raise FileNotFoundError(f"Source summary does not exist: {summary_path}")
        summary = json.loads(summary_path.read_text())
        wc_row = _canonical_wc_row(summary, rep)

        lp_path = summary_path.parent / f"rep{rep}" / "lps" / f"wc_inc_mus_lex_bb_optN_r{rep}.lp"
        if not lp_path.exists():
            raise FileNotFoundError(f"Emitted OptABA-PC LP does not exist: {lp_path}")

        expected_total = _as_int(wc_row.get("n_tests_total"), field="n_tests_total")
        expected_accepted = _as_int(wc_row.get("n_tests_accepted"), field="n_tests_accepted")
        expected_total_weight = _as_int(wc_row.get("weight_total"), field="weight_total")
        weight_accepted_value = wc_row.get("weight_accepted", wc_row.get("accepted_weight"))
        expected_accepted_weight = _as_int(weight_accepted_value, field="weight_accepted")

        csv_total = _as_int(record.get("total_facts"), field="total_facts")
        csv_accepted = _as_int(record.get("accepted_facts"), field="accepted_facts")
        if (csv_total, csv_accepted) != (expected_total, expected_accepted):
            raise ValueError(
                f"CSV/summary fact-count mismatch for {dataset_key} seed {seed}: "
                f"CSV={(csv_total, csv_accepted)}, summary={(expected_total, expected_accepted)}"
            )
        if expected_accepted > expected_total or expected_accepted_weight > expected_total_weight:
            raise ValueError(f"Invalid accepted totals for {dataset_key} seed {seed}")

        inventory.append(
            BaselineInstance(
                dataset_key=dataset_key,
                dataset_label=expected_label,
                seed=seed,
                rep=rep,
                summary_path=summary_path,
                lp_path=lp_path,
                expected_total=expected_total,
                expected_accepted=expected_accepted,
                expected_accepted_weight=expected_accepted_weight,
                expected_total_weight=expected_total_weight,
            )
        )

    inventory.sort(key=lambda item: (DATASET_ORDER[item.dataset_key], item.seed))
    return inventory


def _import_clingo() -> Any:
    try:
        import clingo  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The clingo Python package is required. Activate the pinned "
            f"reproduction environment (import error: {exc!r})."
        ) from exc
    return clingo


def solve_program(
    parsed: ParsedProgram,
    *,
    force_index: int | None,
    timeout_sec: float | None,
    threads: int,
    opt_strategy: str | None,
) -> SolveOutcome:
    """Optimize one emitted program and retain explicit proof-state evidence."""

    clingo = _import_clingo()
    if force_index is not None and force_index not in parsed.facts:
        raise ValueError(f"Unknown forced fact index: {force_index}")
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive or None")

    arguments = ["--opt-mode=optN", "-n", "1", "--warn=none", "-t", str(max(1, threads))]
    if opt_strategy:
        arguments.append(f"--opt-strategy={opt_strategy}")

    program_text = parsed.text
    if force_index is not None:
        program_text = f"{program_text.rstrip()}\n\n% Contestability forced-retention constraint.\n:- not mus({force_index}).\n"

    last_selected: tuple[int, ...] = ()
    last_cost_vector: tuple[int, ...] = ()
    last_optimality_proven = False
    models_seen = 0

    def on_model(model: Any) -> None:
        nonlocal last_selected, last_cost_vector, last_optimality_proven, models_seen
        models_seen += 1
        selected: set[int] = set()
        for symbol in model.symbols(shown=True):
            if getattr(symbol, "name", None) != "mus":
                continue
            arguments_ = getattr(symbol, "arguments", ())
            if len(arguments_) != 1:
                continue
            try:
                index = int(str(arguments_[0]))
            except (TypeError, ValueError):
                continue
            if index in parsed.facts:
                selected.add(index)
        last_selected = tuple(sorted(selected))
        last_cost_vector = tuple(int(value) for value in (model.cost or ()))
        last_optimality_proven = bool(model.optimality_proven)

    start = time.perf_counter()
    result = None
    timed_out = False
    error: str | None = None
    try:
        control = clingo.Control(arguments)
        control.add("base", [], program_text)
        control.ground([("base", [])])
        handle = control.solve(async_=True, on_model=on_model)
        finished = handle.wait(timeout=timeout_sec)
        if not finished:
            timed_out = True
            handle.cancel()
            try:
                finished = bool(handle.wait(timeout=2.0))
            except Exception:
                finished = False
        if finished:
            try:
                result = handle.get()
            except Exception as exc:
                error = f"solve handle failed: {exc!r}"
    except Exception as exc:
        error = f"clingo solve failed: {exc!r}"
    elapsed = time.perf_counter() - start

    if result is not None and bool(getattr(result, "interrupted", False)):
        timed_out = True
    infeasible = bool(result is not None and getattr(result, "unsatisfiable", False) and not timed_out)
    optimality_proven = bool(last_optimality_proven and not timed_out and error is None)
    if timed_out:
        status = "TIMEOUT"
    elif infeasible:
        status = "UNSATISFIABLE"
    elif optimality_proven:
        status = "OPTIMUM FOUND"
    elif result is not None and bool(getattr(result, "satisfiable", False)):
        status = "SATISFIABLE"
    elif error:
        status = "ERROR"
    else:
        status = "UNKNOWN"

    objective_cost: int | None = None
    if models_seen:
        selected_set = set(last_selected)
        objective_cost = sum(weight for index, weight in parsed.weights.items() if index not in selected_set)
        if last_cost_vector and len(last_cost_vector) == 1 and last_cost_vector[0] != objective_cost:
            raise RuntimeError(
                f"clingo cost {last_cost_vector[0]} != reconstructed release cost {objective_cost}"
            )

    return SolveOutcome(
        status=status,
        timed_out=timed_out,
        infeasible=infeasible,
        optimality_proven=optimality_proven,
        solve_time_sec=elapsed,
        selected_indices=last_selected,
        objective_cost=objective_cost,
        cost_vector=last_cost_vector,
        models_seen=models_seen,
        error=error,
    )


def _evidence_id_baseline(instance: BaselineInstance) -> str:
    return f"baseline:{instance.dataset_key}:{instance.seed}"


def _evidence_id_challenge(fact_id: str) -> str:
    return f"challenge:{fact_id}"


def _load_evidence(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        evidence_id = str(record.get("evidence_id", ""))
        if not evidence_id:
            raise ValueError(f"Missing evidence_id at {path}:{line_number}")
        records[evidence_id] = record
    return records


def _write_evidence(path: Path, records: dict[str, dict[str, Any]]) -> None:
    def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            DATASET_ORDER.get(str(record.get("dataset_key")), 999),
            int(record.get("seed", 0) or 0),
            0 if record.get("record_type") == "baseline" else 1,
            int(record.get("fact_index", 0) or 0),
        )

    lines = [json.dumps(record, sort_keys=True) for record in sorted(records.values(), key=sort_key)]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _load_challenge_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CHALLENGE_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Existing challenge CSV lacks columns: {sorted(missing)}")
        for row in reader:
            fact_id = str(row.get("fact_id", ""))
            if not fact_id:
                raise ValueError(f"Existing challenge CSV has a row without fact_id: {path}")
            rows[fact_id] = row
    return rows


def _write_challenge_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        dataset_key = str(row.get("fact_id", "")).split(":", 1)[0]
        return (
            DATASET_ORDER.get(dataset_key, 999),
            _as_int(row.get("seed"), field="seed"),
            _as_int(row.get("fact_index"), field="fact_index"),
        )

    ordered = sorted(rows.values(), key=sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHALLENGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def _baseline_evidence_is_usable(
    record: dict[str, Any] | None,
    *,
    instance: BaselineInstance,
    parsed: ParsedProgram,
    baseline_reference: str,
) -> bool:
    if not record or record.get("record_type") != "baseline":
        return False
    try:
        selected = {int(value) for value in (record.get("selected_fact_indices", []) or [])}
        release_cost = sum(weight for index, weight in parsed.weights.items() if index not in selected)
        common = bool(
            record.get("source_lp_sha256") == parsed.sha256
            and record.get("optimality_proven") is True
            and record.get("baseline_reference", "published") == baseline_reference
            and int(record.get("objective_cost", -1)) == release_cost
            and selected <= set(parsed.facts)
        )
        if baseline_reference == "reoptimize":
            return common
        return bool(
            common
            and int(record.get("expected_accepted", -1)) == instance.expected_accepted
            and len(selected) == instance.expected_accepted
            and release_cost == instance.expected_cost
        )
    except (TypeError, ValueError):
        return False


def _challenge_evidence_is_usable(
    record: dict[str, Any] | None,
    *,
    parsed: ParsedProgram,
    fact_id: str,
    fact_index: int,
) -> bool:
    return bool(
        record
        and record.get("record_type") == "challenge"
        and record.get("source_lp_sha256") == parsed.sha256
        and record.get("fact_id") == fact_id
        and int(record.get("fact_index", -1)) == fact_index
        and record.get("forced_constraint") == f":- not mus({fact_index})."
    )


def _git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        metadata = {"commit": commit, "dirty": dirty}
    except Exception as exc:
        metadata = {"error": repr(exc)}
    return metadata


def _build_manifest(
    *,
    args: argparse.Namespace,
    inventory: Sequence[BaselineInstance],
    evidence: dict[str, dict[str, Any]],
    challenge_rows: dict[str, dict[str, Any]],
    challenge_path: Path,
    evidence_path: Path,
    clingo_version: str,
    started_at: str,
    stopped_early: bool,
) -> dict[str, Any]:
    baseline_instances: list[dict[str, Any]] = []
    expected_fact_ids: set[str] = set()
    for instance in inventory:
        evidence_id = _evidence_id_baseline(instance)
        record = evidence.get(evidence_id)
        analyzable = bool(
            record
            and record.get("optimality_proven") is True
            and not (record.get("validation_errors", []) or [])
        )
        released_indices = [int(value) for value in ((record or {}).get("released_fact_indices", []) or [])]
        released_ids = [str(value) for value in ((record or {}).get("released_fact_ids", []) or [])]
        if analyzable:
            expected_fact_ids.update(released_ids)
        baseline_instances.append(
            {
                "dataset_key": instance.dataset_key,
                "dataset": instance.dataset_label,
                "seed": instance.seed,
                "rep": instance.rep,
                "summary_path": str(instance.summary_path),
                "source_lp": str(instance.lp_path),
                "expected_total_facts": instance.expected_total,
                "expected_accepted_facts": instance.expected_accepted,
                "expected_released_facts": instance.expected_released,
                "expected_baseline_cost": instance.expected_cost,
                "baseline_optimum_cost": (record or {}).get("objective_cost"),
                "baseline_reference": (record or {}).get(
                    "baseline_reference", args.baseline_reference
                ),
                "analyzable": analyzable,
                "solver_status": (record or {}).get("status", "NOT_RUN"),
                "optimality_proven": bool((record or {}).get("optimality_proven", False)),
                "released_fact_indices": released_indices,
                "released_fact_ids": released_ids,
                "evidence_id": evidence_id if record else None,
            }
        )

    present_fact_ids = set(challenge_rows) & expected_fact_ids
    complete = bool(
        not stopped_early
        and len(baseline_instances) == len(inventory)
        and all(item["analyzable"] for item in baseline_instances)
        and present_fact_ids == expected_fact_ids
        and all(_evidence_id_challenge(fact_id) in evidence for fact_id in expected_fact_ids)
    )
    row_values = [challenge_rows[fact_id] for fact_id in present_fact_ids]
    proved = sum(_as_bool(row.get("optimality_proven")) for row in row_values)
    timed_out = sum(_as_bool(row.get("timeout")) for row in row_values)
    infeasible = sum(_as_bool(row.get("infeasible")) for row in row_values)

    outputs: dict[str, Any] = {}
    for name, path in (("challenges_csv", challenge_path), ("solve_evidence_jsonl", evidence_path)):
        if path.exists():
            outputs[name] = {"path": str(path), "sha256": _sha256_path(path)}

    handoff = Path(args.handoff).expanduser().resolve()
    return {
        "schema_version": 1,
        "experiment": "OptABA-PC empirical CI-fact contestability",
        "status": "complete" if complete else "partial",
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "repository": {"path": str(REPO_ROOT), **_git_metadata()},
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "handoff": {
            "path": str(handoff),
            "sha256": _sha256_path(handoff) if handoff.exists() else None,
        },
        "scope": {
            "datasets": [DATASET_LABELS[key] for key in sorted({item.dataset_key for item in inventory}, key=DATASET_ORDER.get)],
            "dataset_keys": sorted({item.dataset_key for item in inventory}, key=DATASET_ORDER.get),
            "seed_start": int(args.seed_start),
            "seed_end": int(args.seed_end),
            "frozen_seed_range": [min(FROZEN_SEEDS), max(FROZEN_SEEDS)],
            "eight_node_cases_excluded": not any(
                item.dataset_key in EIGHT_NODE_DATASET_KEYS for item in inventory
            ),
            "eight_node_cases_included": any(
                item.dataset_key in EIGHT_NODE_DATASET_KEYS for item in inventory
            ),
            "baseline_method": "OptABA-PC",
            "baseline_reference": args.baseline_reference,
            "baseline_configuration": {
                "encoding": "inc",
                "objective": "lex",
                "opt_strategy": "bb",
                "opt_mode": "optN",
                "reification": "mus",
            },
        },
        "solver": {
            "name": "clingo",
            "version": clingo_version,
            "threads": int(args.threads),
            "opt_mode": "optN",
            "models": 1,
            "opt_strategy": args.opt_strategy or None,
            "baseline_timeout_sec": float(args.baseline_timeout_sec),
            "forced_timeout_sec": float(args.timeout_sec),
            "proof_criterion": "last clingo model has model.optimality_proven=true",
        },
        "baseline_witness": {
            "origin": (
                "reoptimized_from_emitted_lp"
                if args.baseline_reference == "reoptimize"
                else "reconstructed_from_emitted_lp"
            ),
            "reason": (
                "Re-optimize mode proves the emitted objective and adopts the proved optimum, even when it improves "
                "an unproved historical incumbent."
                if args.baseline_reference == "reoptimize"
                else "The frozen summary.json files persisted accepted fact counts/weights but not selected fact "
                "identities. Each reconstructed witness is re-proved and must match both published accepted count "
                "and weight."
            ),
            "resume_policy": "Reuse the frozen JSONL baseline witness when its source-LP SHA-256 matches.",
        },
        "input_fact_records": {
            "path": str(Path(args.fact_records).expanduser().resolve()),
            "sha256": _sha256_path(Path(args.fact_records).expanduser().resolve()),
        },
        "counts": {
            "baseline_instances": len(inventory),
            "analyzable_seeds": sum(bool(item["analyzable"]) for item in baseline_instances),
            "expected_challenges_from_summaries": sum(item.expected_released for item in inventory),
            "challenges_from_frozen_witnesses": len(expected_fact_ids),
            "challenge_rows": len(present_fact_ids),
            "proved_challenges": proved,
            "unproved_challenges": len(row_values) - proved,
            "timeouts": timed_out,
            "infeasible_challenges": infeasible,
            "missing_challenge_rows": len(expected_fact_ids - present_fact_ids),
        },
        "baseline_instances": baseline_instances,
        "outputs": outputs,
        "command": [sys.executable, *sys.argv],
    }


def _checkpoint(
    *,
    challenge_path: Path,
    evidence_path: Path,
    challenge_rows: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
) -> None:
    _write_evidence(evidence_path, evidence)
    _write_challenge_rows(challenge_path, challenge_rows)


def _parse_dataset_keys(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_DATASET_KEYS)
    keys: set[str] = set()
    for value in values:
        keys.update(part.strip().lower() for part in str(value).split(",") if part.strip())
    unknown = keys - set(DATASET_LABELS)
    if unknown:
        raise ValueError(f"Unknown dataset keys: {sorted(unknown)}; expected {sorted(DATASET_LABELS)}")
    return keys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-records", default=str(DEFAULT_FACT_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--handoff", default=str(DEFAULT_HANDOFF))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset key to run (cancer, earthquake, survey, asia, er5, er8, sf5, sf8). "
            "Repeatable/comma-separated; the default remains the five-dataset handoff scope."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=min(FROZEN_SEEDS))
    parser.add_argument("--seed-end", type=int, default=max(FROZEN_SEEDS))
    parser.add_argument("--timeout-sec", type=float, default=300.0, help="Wall timeout for each forced solve.")
    parser.add_argument("--baseline-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--baseline-reference",
        choices=("published", "reoptimize"),
        default="published",
        help=(
            "published requires the proved witness to match the frozen accepted count/weight; "
            "reoptimize adopts the newly proved optimum and is required when the historical row was only an incumbent."
        ),
    )
    parser.add_argument("--threads", type=int, default=8, help="Clingo threads; 8 reproduces the frozen sweep setting.")
    parser.add_argument("--opt-strategy", default="bb")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--max-challenges",
        type=int,
        default=None,
        help="Stop after this many newly solved challenges (for smoke tests; manifest remains partial).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate/inventory inputs without importing clingo or writing outputs.")
    parser.add_argument(
        "--retry-unproved",
        action="store_true",
        help="Re-run existing timeout/unproved challenge rows; proved rows are always reused.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing contestability CSV/evidence/manifest sidecars in output-dir.",
    )
    args = parser.parse_args()
    if args.seed_start > args.seed_end:
        parser.error("--seed-start must be <= --seed-end")
    if args.seed_start < min(FROZEN_SEEDS) or args.seed_end > max(FROZEN_SEEDS):
        parser.error("Seed bounds must remain within the frozen 2026..2075 range")
    if args.timeout_sec <= 0 or args.baseline_timeout_sec <= 0:
        parser.error("Timeouts must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.max_challenges is not None and args.max_challenges < 0:
        parser.error("--max-challenges must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    dataset_keys = _parse_dataset_keys(args.dataset)
    fact_records_path = Path(args.fact_records).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    challenge_path = output_dir / "contestability_challenges.csv"
    evidence_path = output_dir / "contestability_solve_evidence.jsonl"
    manifest_path = output_dir / "contestability_manifest.json"

    inventory = load_inventory(
        fact_records_path,
        dataset_keys=dataset_keys,
        seed_start=int(args.seed_start),
        seed_end=int(args.seed_end),
    )
    expected_by_dataset = {
        key: sum(item.expected_released for item in inventory if item.dataset_key == key)
        for key in sorted(dataset_keys, key=DATASET_ORDER.get)
    }
    print(f"Inventory: {len(inventory)} frozen baseline instances")
    print(
        "Expected released-fact challenges from summaries: "
        + ", ".join(f"{DATASET_LABELS[key]}={expected_by_dataset[key]}" for key in expected_by_dataset)
        + f" (total={sum(expected_by_dataset.values())})"
    )
    if args.dry_run:
        return 0

    clingo = _import_clingo()
    clingo_version = str(getattr(clingo, "__version__", "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (challenge_path, evidence_path, manifest_path):
            if path.exists():
                path.unlink()

    evidence = _load_evidence(evidence_path)
    challenge_rows = _load_challenge_rows(challenge_path)
    started_at = _utc_now()
    newly_solved = 0
    stopped_early = False

    for instance_number, instance in enumerate(inventory, 1):
        parsed = parse_emitted_program(instance.lp_path)
        if len(parsed.facts) != instance.expected_total:
            raise ValueError(
                f"LP fact count {len(parsed.facts)} != summary total {instance.expected_total} "
                f"for {instance.dataset_key} seed {instance.seed}"
            )
        if sum(parsed.weights.values()) != instance.expected_total_weight:
            raise ValueError(
                f"LP total weight {sum(parsed.weights.values())} != summary total weight "
                f"{instance.expected_total_weight} for {instance.dataset_key} seed {instance.seed}"
            )

        baseline_id = _evidence_id_baseline(instance)
        baseline_record = evidence.get(baseline_id)
        if not _baseline_evidence_is_usable(
            baseline_record,
            instance=instance,
            parsed=parsed,
            baseline_reference=str(args.baseline_reference),
        ):
            print(
                f"[{instance_number}/{len(inventory)}] proving baseline "
                f"{instance.dataset_label} seed {instance.seed} (expected releases={instance.expected_released})"
            )
            baseline_outcome = solve_program(
                parsed,
                force_index=None,
                timeout_sec=float(args.baseline_timeout_sec),
                threads=int(args.threads),
                opt_strategy=str(args.opt_strategy) if args.opt_strategy else None,
            )
            selected = set(baseline_outcome.selected_indices)
            released_indices = sorted(set(parsed.facts) - selected)
            validation_errors: list[str] = []
            if not baseline_outcome.optimality_proven:
                validation_errors.append("baseline optimality was not proven")
            selected_weight = sum(parsed.weights[index] for index in selected)
            if args.baseline_reference == "published":
                if len(selected) != instance.expected_accepted:
                    validation_errors.append(
                        f"accepted count {len(selected)} != published {instance.expected_accepted}"
                    )
                if selected_weight != instance.expected_accepted_weight:
                    validation_errors.append(
                        f"accepted weight {selected_weight} != published {instance.expected_accepted_weight}"
                    )
                if baseline_outcome.objective_cost != instance.expected_cost:
                    validation_errors.append(
                        f"release cost {baseline_outcome.objective_cost} != published {instance.expected_cost}"
                    )

            released_fact_ids = [
                stable_fact_id(instance.dataset_key, instance.seed, parsed.facts[index])
                for index in released_indices
            ]
            baseline_record = {
                "schema_version": 1,
                "record_type": "baseline",
                "baseline_reference": str(args.baseline_reference),
                "evidence_id": baseline_id,
                "dataset_key": instance.dataset_key,
                "dataset": instance.dataset_label,
                "seed": instance.seed,
                "rep": instance.rep,
                "summary_path": str(instance.summary_path),
                "source_lp": str(instance.lp_path),
                "source_lp_sha256": parsed.sha256,
                "status": baseline_outcome.status,
                "timeout": baseline_outcome.timed_out,
                "infeasible": baseline_outcome.infeasible,
                "optimality_proven": baseline_outcome.optimality_proven,
                "solve_time_sec": baseline_outcome.solve_time_sec,
                "objective_cost": baseline_outcome.objective_cost,
                "cost_vector": list(baseline_outcome.cost_vector),
                "models_seen": baseline_outcome.models_seen,
                "selected_fact_indices": sorted(selected),
                "released_fact_indices": released_indices,
                "released_fact_ids": released_fact_ids,
                "expected_total": instance.expected_total,
                "expected_accepted": instance.expected_accepted,
                "expected_accepted_weight": instance.expected_accepted_weight,
                "expected_total_weight": instance.expected_total_weight,
                "expected_objective_cost": instance.expected_cost,
                "validation_errors": validation_errors,
                "error": baseline_outcome.error,
                "clingo_version": clingo_version,
                "solver_options": {
                    "opt_mode": "optN",
                    "models": 1,
                    "threads": int(args.threads),
                    "opt_strategy": args.opt_strategy or None,
                },
                "recorded_at_utc": _utc_now(),
            }
            evidence[baseline_id] = baseline_record
            _write_evidence(evidence_path, evidence)
            if validation_errors:
                print(
                    f"WARNING: unanalyzable baseline {instance.dataset_label} seed {instance.seed}: "
                    + "; ".join(validation_errors),
                    file=sys.stderr,
                )
                continue
        else:
            print(
                f"[{instance_number}/{len(inventory)}] reusing frozen baseline witness "
                f"{instance.dataset_label} seed {instance.seed}"
            )

        released_indices = [int(index) for index in (baseline_record.get("released_fact_indices", []) or [])]
        for fact_index in released_indices:
            fact_atom = parsed.facts[fact_index]
            fact_id = stable_fact_id(instance.dataset_key, instance.seed, fact_atom)
            challenge_id = _evidence_id_challenge(fact_id)
            existing_row = challenge_rows.get(fact_id)
            existing_evidence = evidence.get(challenge_id)
            reusable = bool(
                existing_row
                and _challenge_evidence_is_usable(
                    existing_evidence,
                    parsed=parsed,
                    fact_id=fact_id,
                    fact_index=fact_index,
                )
            )
            if reusable and args.retry_unproved and not _as_bool(existing_row.get("optimality_proven")):
                reusable = False
            if reusable:
                continue
            if args.max_challenges is not None and newly_solved >= args.max_challenges:
                stopped_early = True
                break

            forced = solve_program(
                parsed,
                force_index=fact_index,
                timeout_sec=float(args.timeout_sec),
                threads=int(args.threads),
                opt_strategy=str(args.opt_strategy) if args.opt_strategy else None,
            )
            newly_solved += 1
            baseline_cost = int(baseline_record["objective_cost"])
            forced_cost = forced.objective_cost if forced.optimality_proven else None
            margin = forced_cost - baseline_cost if forced_cost is not None else None
            if margin is not None and margin < 0:
                raise RuntimeError(
                    f"Negative contestability margin for {fact_id}: forced={forced_cost}, baseline={baseline_cost}"
                )
            normalized_margin = margin / parsed.weights[fact_index] if margin is not None else None

            baseline_evidence_ref = f"{evidence_path.name}#{baseline_id}"
            forced_evidence_ref = f"{evidence_path.name}#{challenge_id}"
            challenge_rows[fact_id] = {
                "dataset": instance.dataset_label,
                "seed": instance.seed,
                "fact_id": fact_id,
                "fact_weight": parsed.weights[fact_index],
                "baseline_optimum_cost": baseline_cost,
                "forced_retention_optimum_cost": forced_cost,
                "margin": margin,
                "normalized_margin": normalized_margin,
                "zero_margin": (margin == 0) if margin is not None else "",
                "solve_time_sec": forced.solve_time_sec,
                "timeout": forced.timed_out,
                "optimality_proven": forced.optimality_proven,
                "optimality_evidence": (
                    f"baseline={baseline_evidence_ref}; forced={forced_evidence_ref}; "
                    f"status={forced.status}; "
                    f"model.optimality_proven={str(forced.optimality_proven).lower()}"
                ),
                "infeasible": forced.infeasible,
                "solver_status": forced.status,
                "fact_atom": fact_atom,
                "fact_index": fact_index,
                "source_lp": str(instance.lp_path),
            }
            evidence[challenge_id] = {
                "schema_version": 1,
                "record_type": "challenge",
                "evidence_id": challenge_id,
                "dataset_key": instance.dataset_key,
                "dataset": instance.dataset_label,
                "seed": instance.seed,
                "rep": instance.rep,
                "fact_id": fact_id,
                "fact_atom": fact_atom,
                "fact_index": fact_index,
                "fact_weight": parsed.weights[fact_index],
                "forced_constraint": f":- not mus({fact_index}).",
                "source_lp": str(instance.lp_path),
                "source_lp_sha256": parsed.sha256,
                "baseline_evidence_id": baseline_id,
                "baseline_optimum_cost": baseline_cost,
                "status": forced.status,
                "timeout": forced.timed_out,
                "infeasible": forced.infeasible,
                "optimality_proven": forced.optimality_proven,
                "solve_time_sec": forced.solve_time_sec,
                "objective_cost": forced.objective_cost,
                "reported_forced_retention_optimum_cost": forced_cost,
                "cost_vector": list(forced.cost_vector),
                "models_seen": forced.models_seen,
                "selected_fact_indices": list(forced.selected_indices),
                "error": forced.error,
                "clingo_version": clingo_version,
                "solver_options": {
                    "opt_mode": "optN",
                    "models": 1,
                    "threads": int(args.threads),
                    "opt_strategy": args.opt_strategy or None,
                    "timeout_sec": float(args.timeout_sec),
                },
                "recorded_at_utc": _utc_now(),
            }
            _checkpoint(
                challenge_path=challenge_path,
                evidence_path=evidence_path,
                challenge_rows=challenge_rows,
                evidence=evidence,
            )
            if args.progress_every > 0 and (newly_solved == 1 or newly_solved % args.progress_every == 0):
                print(
                    f"  solved {newly_solved} new challenge(s); latest={fact_id}; "
                    f"status={forced.status}; margin={margin if margin is not None else 'NA'}"
                )

        if stopped_early:
            break

    _checkpoint(
        challenge_path=challenge_path,
        evidence_path=evidence_path,
        challenge_rows=challenge_rows,
        evidence=evidence,
    )
    manifest = _build_manifest(
        args=args,
        inventory=inventory,
        evidence=evidence,
        challenge_rows=challenge_rows,
        challenge_path=challenge_path,
        evidence_path=evidence_path,
        clingo_version=clingo_version,
        started_at=started_at,
        stopped_early=stopped_early,
    )
    _atomic_write_json(manifest_path, manifest)
    print(
        f"Wrote {manifest['counts']['challenge_rows']}/{manifest['counts']['challenges_from_frozen_witnesses']} "
        f"challenge rows; manifest status={manifest['status']}"
    )
    print(f"Challenge CSV: {challenge_path}")
    print(f"Manifest: {manifest_path}")
    return 0 if manifest["status"] == "complete" or stopped_early else 2


if __name__ == "__main__":
    raise SystemExit(main())
