#!/usr/bin/env python3
"""Compute exact forced-fact contestability margins for saved ASPCR-DAG runs.

The matched ASPCR-DAG encoding minimizes the total weight of ``fail/6`` atoms,
where a failure is a tested CI relation contradicted by the returned DAG.  For
each fact failed by the saved optimum, this runner adds the integrity
constraint

    :- constraint(F,X,Y,C,J,M,W), fail(X,Y,C,J,M,W).

with ``F`` fixed to the challenged constraint id, re-solves the same native
Bayesian log-weighted DAG problem, and records the exact optimum-cost margin.
All other facts remain soft.  The saved constraint CSVs are sufficient to
reconstruct the ASP inputs; Bayesian tests are not re-run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "results"
    / "aspcr_matched"
    / "paper_aspcr_dag_alpha001_n5000_50rep"
    / "results"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "tables"
    / "paper_final_matched_50rep"
    / "baseline_contestability"
    / "aspcr_native"
)
DEFAULT_ASPCR_ROOT = Path(os.environ.get("ASPCR_ROOT", REPO_ROOT / "external" / "aspcr-hyttinen2014uai"))
DEFAULT_ENCODING = DEFAULT_ASPCR_ROOT / "ASP" / "new_wmaxsat_acyclic_sufficient.pl"
DEFAULT_DATASETS = ("cancer", "earthquake", "survey")
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
ARTIFACT_RE = re.compile(
    r"^aspcr_log-weights_dag_sufficient_(?P<dataset>[a-z0-9]+)_(?P<seed>\d+)_constraints\.csv$"
)

CHALLENGE_COLUMNS = (
    "dataset",
    "dataset_label",
    "seed",
    "constraint_id",
    "fact_id",
    "tested_relation",
    "x",
    "y",
    "conditioning_set",
    "cset",
    "jset",
    "mset",
    "fact_weight",
    "baseline_optimum_cost",
    "forced_optimum_cost",
    "margin",
    "normalized_margin",
    "zero_margin",
    "optimality_proven",
    "target_enforced",
    "infeasible",
    "timeout",
    "solver_status",
    "solve_time_sec",
    "failed_facts_after_force",
    "source_constraints",
    "source_diagnostics",
    "program_constraints",
    "error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_dataframe(path: Path, frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output:
                output[column] = None
        output = output[list(columns)]
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "t", "1", "yes"}


def _dataset_sort(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    return frame.assign(_order=frame["dataset"].map(order).fillna(999)).sort_values(
        ["_order", "seed", "constraint_id"]
    ).drop(columns="_order")


def build_base_program(n_nodes: int, triples: Iterable[tuple[int, int, int]]) -> str:
    """Build deterministic set and transformation facts for the HEJ encoding."""

    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise ValueError("ASPCR needs at least two nodes")
    lines = [f"node(1..{n_nodes})."]
    for mask in range(2**n_nodes):
        for node in range(1, n_nodes + 1):
            if mask & (1 << (node - 1)):
                lines.append(f"ismember({mask},{node}).")

    seen = {(0, 0, 0)}

    def connect(cset: int, jset: int, mset: int) -> None:
        key = (int(cset), int(jset), int(mset))
        if key in seen:
            return
        cset, jset, mset = key
        if cset:
            bit = cset & -cset
            node = bit.bit_length()
            lower = (cset - bit, jset, mset)
            rule = f"condition({lower[0]},{node},{cset},{jset},{mset})."
        elif mset:
            bit = mset & -mset
            node = bit.bit_length()
            lower = (cset, jset, mset - bit)
            rule = f"marginalize({cset},{jset},{lower[2]},{node},{mset})."
        elif jset:
            bit = jset & -jset
            node = bit.bit_length()
            lower = (cset, jset - bit, mset)
            rule = f"intervene({cset},{lower[1]},{node},{jset},{mset})."
        else:  # pragma: no cover - guarded by the (0,0,0) seed
            raise AssertionError("Unexpected empty transformation key")
        connect(*lower)
        lines.append(rule)
        seen.add(key)

    for triple in sorted({tuple(map(int, triple)) for triple in triples}):
        connect(*triple)
    return "\n".join(lines) + "\n"


def build_constraint_program(constraints: pd.DataFrame) -> str:
    """Reconstruct weighted indep/dep atoms and stable constraint identifiers."""

    lines: list[str] = []
    for row in constraints.sort_values("constraint_id").itertuples(index=False):
        relation = "indep" if _bool_value(row.test_independent) else "dep"
        args = ",".join(
            str(int(value))
            for value in (row.x, row.y, row.cset, row.jset, row.mset, row.asp_weight)
        )
        lines.append(f"{relation}({args}).")
        lines.append(f"constraint({int(row.constraint_id)},{args}).")
    return "\n".join(lines) + "\n"


HARDENING_PROGRAM = """#const forced_id=0.
:- forced_id > 0, constraint(forced_id,X,Y,C,J,M,W), fail(X,Y,C,J,M,W).
#show fail/6.
"""


def _parse_solver_json(raw: str, *, forced_id: int, target_fail: str | None) -> dict[str, Any]:
    payload = json.loads(raw)
    status = str(payload.get("Result", "UNKNOWN"))
    models = payload.get("Models", {}) or {}
    optimum = status == "OPTIMUM FOUND" and str(models.get("Optimum", "")).lower() == "yes"
    witnesses = ((payload.get("Call", []) or [{}])[0].get("Witnesses", []) or [])
    witness = witnesses[-1] if witnesses else {}
    costs = witness.get("Costs", models.get("Costs", [])) or []
    cost = int(costs[0]) if optimum and len(costs) == 1 else None
    atoms = [str(value) for value in (witness.get("Value", []) or [])]
    fail_atoms = [atom for atom in atoms if atom.startswith("fail(")]
    return {
        "solver_status": status,
        "optimality_proven": optimum,
        "infeasible": status == "UNSATISFIABLE",
        "timeout": status in {"UNKNOWN", "SATISFIABLE"},
        "objective_cost": cost,
        "failed_facts": fail_atoms,
        "failed_fact_count": len(fail_atoms),
        "target_enforced": forced_id == 0 or (target_fail is not None and target_fail not in fail_atoms),
        "solver_total_time": float((payload.get("Time", {}) or {}).get("Total", math.nan)),
    }


def solve_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated clingo optimization; suitable for a thread executor."""

    forced_id = int(task["forced_id"])
    timeout_sec = float(task["timeout_sec"])
    command = [
        str(task["clingo"]),
        "--outf=2",
        "--quiet=1,0",
        "--warn=none",
        "--configuration=crafty",
        f"--time-limit={max(1, math.ceil(timeout_sec))}",
        f"--const=forced_id={forced_id}",
        str(task["base_program"]),
        str(task["constraint_program"]),
        str(task["encoding"]),
        str(task["hardening_program"]),
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec + 30,
        )
        elapsed = time.monotonic() - start
        if not completed.stdout.strip():
            raise RuntimeError(
                f"clingo produced no JSON (exit={completed.returncode}): {completed.stderr[-1000:]}"
            )
        result = _parse_solver_json(
            completed.stdout,
            forced_id=forced_id,
            target_fail=task.get("target_fail"),
        )
        result.update(
            {
                "solve_time_sec": elapsed,
                "returncode": completed.returncode,
                # clingo uses 10/20/30 for SAT/UNSAT/OPTIMUM at the CLI.
                "error": (
                    ""
                    if completed.returncode in {0, 10, 20, 30}
                    else completed.stderr[-1000:]
                ),
            }
        )
        return {**task, **result}
    except subprocess.TimeoutExpired as exc:
        return {
            **task,
            "solver_status": "EXTERNAL TIMEOUT",
            "optimality_proven": False,
            "infeasible": False,
            "timeout": True,
            "objective_cost": None,
            "failed_facts": [],
            "failed_fact_count": None,
            "target_enforced": False,
            "solve_time_sec": time.monotonic() - start,
            "returncode": None,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            **task,
            "solver_status": "ERROR",
            "optimality_proven": False,
            "infeasible": False,
            "timeout": False,
            "objective_cost": None,
            "failed_facts": [],
            "failed_fact_count": None,
            "target_enforced": False,
            "solve_time_sec": time.monotonic() - start,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_tasks(tasks: Sequence[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [solve_task(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(solve_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _load_instance(constraints_path: Path, diagnostics_path: Path) -> dict[str, Any]:
    constraints = pd.read_csv(constraints_path)
    diagnostics_table = pd.read_csv(diagnostics_path)
    if len(diagnostics_table) != 1:
        raise RuntimeError(f"Expected one ASPCR diagnostic row: {diagnostics_path}")
    diagnostics = diagnostics_table.iloc[0].to_dict()
    required = {
        "constraint_id", "x", "y", "conditioning_set", "cset", "jset", "mset",
        "test_independent", "tested_relation", "retained", "asp_weight",
    }
    missing = required - set(constraints.columns)
    if missing:
        raise RuntimeError(f"ASPCR constraint trace lacks columns {sorted(missing)}: {constraints_path}")
    ids = pd.to_numeric(constraints["constraint_id"], errors="raise").astype(int)
    if ids.tolist() != list(range(1, len(constraints) + 1)):
        raise RuntimeError(f"Non-contiguous ASPCR constraint ids: {constraints_path}")
    weights = pd.to_numeric(constraints["asp_weight"], errors="raise")
    if (weights < 0).any():
        raise RuntimeError(f"Negative ASPCR fact weight: {constraints_path}")
    expected = math.comb(int(diagnostics["n_nodes"]), 2) * 2 ** (int(diagnostics["n_nodes"]) - 2)
    if len(constraints) != expected or int(diagnostics["expected_constraints"]) != expected:
        raise RuntimeError(f"Incomplete ASPCR trace: {constraints_path}")
    failed = ~constraints["retained"].map(_bool_value)
    if (weights.loc[failed] <= 0).any():
        raise RuntimeError(
            f"A baseline-failed ASPCR fact has zero weight, so its normalized margin is undefined: "
            f"{constraints_path}"
        )
    recomputed = int(pd.to_numeric(constraints.loc[failed, "asp_weight"]).sum())
    if recomputed != int(diagnostics["solver_objective"]):
        raise RuntimeError(f"ASPCR saved objective mismatch: {constraints_path}")
    return {
        "constraints": constraints,
        "diagnostics": diagnostics,
        "failed_mask": failed,
    }


def inventory_instances(
    artifact_dir: Path,
    datasets: set[str],
    program_dir: Path,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    base_inputs: dict[int, list[tuple[int, int, int]]] = {}
    pending: list[dict[str, Any]] = []
    for constraints_path in sorted(artifact_dir.glob("*_constraints.csv")):
        match = ARTIFACT_RE.match(constraints_path.name)
        if match is None or match.group("dataset") not in datasets:
            continue
        dataset = match.group("dataset")
        seed = int(match.group("seed"))
        diagnostics_path = constraints_path.with_name(
            constraints_path.name.replace("_constraints.csv", "_diagnostics.csv")
        )
        if not diagnostics_path.exists():
            raise FileNotFoundError(f"Missing ASPCR diagnostics: {diagnostics_path}")
        loaded = _load_instance(constraints_path, diagnostics_path)
        constraints = loaded["constraints"]
        n_nodes = int(loaded["diagnostics"]["n_nodes"])
        triples = [
            (int(row.cset), int(row.jset), int(row.mset))
            for row in constraints.itertuples(index=False)
        ]
        base_inputs.setdefault(n_nodes, []).extend(triples)
        constraint_program = program_dir / f"{dataset}_{seed}_constraints.lp"
        _atomic_text(constraint_program, build_constraint_program(constraints))
        pending.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "seed": seed,
                "n_nodes": n_nodes,
                "constraints_path": constraints_path,
                "diagnostics_path": diagnostics_path,
                "constraint_program": constraint_program,
                **loaded,
            }
        )

    if not pending:
        raise RuntimeError(f"No selected ASPCR artifacts found under {artifact_dir}")
    for n_nodes, triples in base_inputs.items():
        base_path = program_dir / f"n{n_nodes}_base.lp"
        _atomic_text(base_path, build_base_program(n_nodes, triples))
    hardening_path = program_dir / "hardening.lp"
    _atomic_text(hardening_path, HARDENING_PROGRAM)
    for instance in pending:
        instance["base_program"] = program_dir / f"n{instance['n_nodes']}_base.lp"
        instance["hardening_program"] = hardening_path
        instances.append(instance)
    return sorted(instances, key=lambda item: (DATASET_ORDER.index(item["dataset"]), item["seed"]))


def summarise_challenges(
    challenges: pd.DataFrame,
    instances: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        selected_instances = [item for item in instances if item["dataset"] == dataset]
        if not selected_instances:
            continue
        group = challenges[challenges["dataset"] == dataset].copy()
        proved = group[group["optimality_proven"].map(_bool_value)]
        normalized = pd.to_numeric(proved["normalized_margin"], errors="coerce").dropna()
        zero = int(proved["zero_margin"].map(_bool_value).sum())
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "analyzable_seeds": len({item["seed"] for item in selected_instances}),
                "challenges": len(group),
                "proved_challenges": len(proved),
                "zero_margin_challenges": zero,
                "zero_margin_proportion": zero / len(proved) if len(proved) else math.nan,
                "normalized_margin_median": normalized.quantile(0.5) if len(normalized) else math.nan,
                "normalized_margin_q1": normalized.quantile(0.25) if len(normalized) else math.nan,
                "normalized_margin_q3": normalized.quantile(0.75) if len(normalized) else math.nan,
                "infeasible_challenges": int(group["infeasible"].map(_bool_value).sum()),
                "timeouts": int(group["timeout"].map(_bool_value).sum()),
                "target_enforcement_failures": int((~group["target_enforced"].map(_bool_value)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _existing_challenges(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHALLENGE_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(CHALLENGE_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Existing ASPCR challenge CSV lacks columns: {sorted(missing)}")
    if frame["fact_id"].duplicated().any():
        raise RuntimeError("Existing ASPCR challenge CSV contains duplicate fact ids")
    return frame[list(CHALLENGE_COLUMNS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--encoding",
        default=str(DEFAULT_ENCODING),
        help="ASPCR DAG/sufficiency encoding (or set ASPCR_ROOT).",
    )
    parser.add_argument("--clingo", default=shutil.which("clingo") or str(Path(sys.executable).parent / "clingo"))
    parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--max-challenges",
        type=int,
        default=None,
        help="Run only the first N pending challenges (for smoke tests; manifest remains partial).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.timeout_sec <= 0:
        raise ValueError("--workers and --timeout-sec must be positive")
    selected = set(args.dataset or DEFAULT_DATASETS)
    artifact_dir = _resolve(args.artifact_dir)
    output_dir = _resolve(args.output_dir)
    program_dir = output_dir / "programs"
    encoding = _resolve(args.encoding)
    clingo = _resolve(args.clingo)
    if not encoding.exists():
        raise FileNotFoundError(f"ASPCR encoding not found: {encoding}")
    if not clingo.exists():
        raise FileNotFoundError(f"clingo not found: {clingo}")

    challenge_path = output_dir / "aspcr_contestability_challenges.csv"
    baseline_path = output_dir / "aspcr_baseline_validations.csv"
    summary_path = output_dir / "aspcr_contestability_summary.csv"
    manifest_path = output_dir / "aspcr_contestability_manifest.json"
    instances = inventory_instances(artifact_dir, selected, program_dir)
    print(f"Inventory: {len(instances)} ASPCR-DAG instances", flush=True)

    common = {
        "clingo": str(clingo),
        "encoding": str(encoding),
        "timeout_sec": float(args.timeout_sec),
    }
    baseline_tasks = [
        {
            **common,
            "dataset": item["dataset"],
            "seed": item["seed"],
            "forced_id": 0,
            "target_fail": None,
            "base_program": str(item["base_program"]),
            "constraint_program": str(item["constraint_program"]),
            "hardening_program": str(item["hardening_program"]),
            "expected_cost": int(item["diagnostics"]["solver_objective"]),
        }
        for item in instances
    ]
    baseline_results = _run_tasks(baseline_tasks, args.workers)
    baseline_rows: list[dict[str, Any]] = []
    for result in baseline_results:
        matches = result["optimality_proven"] and result["objective_cost"] == result["expected_cost"]
        baseline_rows.append(
            {
                "dataset": result["dataset"],
                "seed": result["seed"],
                "expected_cost": result["expected_cost"],
                "recomputed_cost": result["objective_cost"],
                "optimality_proven": result["optimality_proven"],
                "objective_matches": matches,
                "solver_status": result["solver_status"],
                "solve_time_sec": result["solve_time_sec"],
                "error": result["error"],
            }
        )
        if not matches:
            raise RuntimeError(
                f"ASPCR baseline validation failed for {result['dataset']}/{result['seed']}: "
                f"expected {result['expected_cost']}, got {result['objective_cost']} "
                f"({result['solver_status']})"
            )
    baseline_frame = _dataset_sort(pd.DataFrame(baseline_rows).assign(constraint_id=0)).drop(
        columns="constraint_id"
    )
    _atomic_dataframe(baseline_path, baseline_frame)
    print("All regenerated ASPCR baselines match their saved proved objectives.", flush=True)

    existing = _existing_challenges(challenge_path)
    if len(existing):
        existing = existing[existing["dataset"].isin(selected)].copy()
    completed_ids = set(existing["fact_id"].astype(str))
    tasks: list[dict[str, Any]] = []
    task_rows: dict[str, dict[str, Any]] = {}
    for instance in instances:
        failed = instance["constraints"].loc[instance["failed_mask"]]
        baseline_cost = int(instance["diagnostics"]["solver_objective"])
        for row in failed.itertuples(index=False):
            constraint_id = int(row.constraint_id)
            fact_id = f"aspcr:{instance['dataset']}:{instance['seed']}:{constraint_id}"
            if fact_id in completed_ids:
                continue
            weight = int(row.asp_weight)
            fail_args = ",".join(
                str(int(value))
                for value in (row.x, row.y, row.cset, row.jset, row.mset, weight)
            )
            task_rows[fact_id] = {
                "dataset": instance["dataset"],
                "dataset_label": instance["dataset_label"],
                "seed": instance["seed"],
                "constraint_id": constraint_id,
                "fact_id": fact_id,
                "tested_relation": str(row.tested_relation),
                "x": int(row.x),
                "y": int(row.y),
                "conditioning_set": "" if pd.isna(row.conditioning_set) else str(row.conditioning_set),
                "cset": int(row.cset),
                "jset": int(row.jset),
                "mset": int(row.mset),
                "fact_weight": weight,
                "baseline_optimum_cost": baseline_cost,
                "source_constraints": str(instance["constraints_path"]),
                "source_diagnostics": str(instance["diagnostics_path"]),
                "program_constraints": str(instance["constraint_program"]),
            }
            tasks.append(
                {
                    **common,
                    "fact_id": fact_id,
                    "dataset": instance["dataset"],
                    "seed": instance["seed"],
                    "forced_id": constraint_id,
                    "target_fail": f"fail({fail_args})",
                    "base_program": str(instance["base_program"]),
                    "constraint_program": str(instance["constraint_program"]),
                    "hardening_program": str(instance["hardening_program"]),
                }
            )

    tasks.sort(key=lambda task: (DATASET_ORDER.index(task["dataset"]), task["seed"], task["forced_id"]))
    total_expected = sum(int(item["failed_mask"].sum()) for item in instances)
    if args.max_challenges is not None:
        tasks = tasks[: max(0, int(args.max_challenges))]
    manifest = {
        "schema_version": 1,
        "status": "running",
        "updated_at_utc": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
        "semantics": (
            "Exact ASPCR native-trace margin: force one baseline-failed fact hard while all "
            "other Bayesian log-weighted facts remain soft."
        ),
        "datasets": sorted(selected, key=DATASET_ORDER.index),
        "instances": len(instances),
        "expected_challenges": total_expected,
        "already_complete": len(existing),
        "pending_selected": len(tasks),
        "workers": args.workers,
        "timeout_sec": args.timeout_sec,
        "max_challenges": args.max_challenges,
        "inputs": {
            "artifact_dir": str(artifact_dir),
            "encoding": {"path": str(encoding), "sha256": _sha256(encoding)},
            "artifacts": [
                {
                    "dataset": item["dataset"],
                    "seed": item["seed"],
                    "constraints": {
                        "path": str(item["constraints_path"]),
                        "sha256": _sha256(item["constraints_path"]),
                    },
                    "diagnostics": {
                        "path": str(item["diagnostics_path"]),
                        "sha256": _sha256(item["diagnostics_path"]),
                    },
                    "program": {
                        "path": str(item["constraint_program"]),
                        "sha256": _sha256(item["constraint_program"]),
                    },
                }
                for item in instances
            ],
        },
        "solver": {
            "path": str(clingo),
            "version": subprocess.run(
                [str(clingo), "--version"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0],
        },
    }
    _atomic_json(manifest_path, manifest)
    print(
        f"Challenges: expected={total_expected}, existing={len(existing)}, pending={len(tasks)}",
        flush=True,
    )

    output_rows = existing.to_dict("records")
    if tasks:
        if args.workers <= 1:
            iterator = enumerate((solve_task(task) for task in tasks), start=1)
            for index, solved in iterator:
                base = task_rows[solved["fact_id"]]
                forced_cost = solved["objective_cost"] if solved["optimality_proven"] else None
                margin = forced_cost - base["baseline_optimum_cost"] if forced_cost is not None else None
                output_rows.append(
                    {
                        **base,
                        "forced_optimum_cost": forced_cost,
                        "margin": margin,
                        "normalized_margin": margin / base["fact_weight"] if margin is not None else None,
                        "zero_margin": margin == 0 if margin is not None else False,
                        "optimality_proven": solved["optimality_proven"],
                        "target_enforced": solved["target_enforced"],
                        "infeasible": solved["infeasible"],
                        "timeout": solved["timeout"],
                        "solver_status": solved["solver_status"],
                        "solve_time_sec": solved["solve_time_sec"],
                        "failed_facts_after_force": solved["failed_fact_count"],
                        "error": solved["error"],
                    }
                )
                if index % 10 == 0 or index == len(tasks):
                    frame = _dataset_sort(pd.DataFrame(output_rows))
                    _atomic_dataframe(challenge_path, frame, columns=CHALLENGE_COLUMNS)
                    print(f"Solved {index}/{len(tasks)} pending challenges", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(solve_task, task): task for task in tasks}
                for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    solved = future.result()
                    base = task_rows[solved["fact_id"]]
                    forced_cost = solved["objective_cost"] if solved["optimality_proven"] else None
                    margin = forced_cost - base["baseline_optimum_cost"] if forced_cost is not None else None
                    output_rows.append(
                        {
                            **base,
                            "forced_optimum_cost": forced_cost,
                            "margin": margin,
                            "normalized_margin": margin / base["fact_weight"] if margin is not None else None,
                            "zero_margin": margin == 0 if margin is not None else False,
                            "optimality_proven": solved["optimality_proven"],
                            "target_enforced": solved["target_enforced"],
                            "infeasible": solved["infeasible"],
                            "timeout": solved["timeout"],
                            "solver_status": solved["solver_status"],
                            "solve_time_sec": solved["solve_time_sec"],
                            "failed_facts_after_force": solved["failed_fact_count"],
                            "error": solved["error"],
                        }
                    )
                    if index % 10 == 0 or index == len(tasks):
                        frame = _dataset_sort(pd.DataFrame(output_rows))
                        _atomic_dataframe(challenge_path, frame, columns=CHALLENGE_COLUMNS)
                        print(f"Solved {index}/{len(tasks)} pending challenges", flush=True)

    challenges = _dataset_sort(pd.DataFrame(output_rows)) if output_rows else pd.DataFrame(columns=CHALLENGE_COLUMNS)
    _atomic_dataframe(challenge_path, challenges, columns=CHALLENGE_COLUMNS)
    summary = summarise_challenges(challenges, instances)
    _atomic_dataframe(summary_path, summary)
    complete = len(challenges) == total_expected and args.max_challenges is None
    manifest.update(
        {
            "status": "complete" if complete else "partial",
            "updated_at_utc": _utc_now(),
            "counts": {
                "challenge_rows": len(challenges),
                "proved_challenges": int(challenges["optimality_proven"].map(_bool_value).sum()) if len(challenges) else 0,
                "zero_margin_challenges": int(challenges["zero_margin"].map(_bool_value).sum()) if len(challenges) else 0,
                "timeouts": int(challenges["timeout"].map(_bool_value).sum()) if len(challenges) else 0,
                "infeasible": int(challenges["infeasible"].map(_bool_value).sum()) if len(challenges) else 0,
                "target_enforcement_failures": int((~challenges["target_enforced"].map(_bool_value)).sum()) if len(challenges) else 0,
            },
            "outputs": {
                "baseline_validations": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
                "challenges": {"path": str(challenge_path), "sha256": _sha256(challenge_path)},
                "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            },
        }
    )
    _atomic_json(manifest_path, manifest)
    print(summary.to_string(index=False), flush=True)
    print(f"Manifest status: {manifest['status']} ({manifest_path})", flush=True)
    return 0 if complete or args.max_challenges is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
