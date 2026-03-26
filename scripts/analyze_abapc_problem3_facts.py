from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pgmpy.readwrite import BIFReader

FACT_RE = re.compile(r"^(?:ext_)?(indep|dep)\((\d+),(\d+),([^)]+)\)\.?$")
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc fact-set and truth comparison for ABAPC runs")
    parser.add_argument("--datasets", nargs="+", default=["asia", "sachs", "survey"])
    parser.add_argument("--bb-version", default="abapc_bb_problem3_matched10_gsq_searchv3")
    parser.add_argument("--nor-version", default="abapc_nor_problem3_matched10_gsq")
    parser.add_argument("--orig-version", default="abapc_orig_problem3_matched10_gsq")
    parser.add_argument("--output-prefix", default="abapc_problem3_fact_compare")
    return parser.parse_args()


def parse_fact_key(fact_key: str) -> tuple[str, int, int, tuple[int, ...]]:
    match = FACT_RE.match(str(fact_key).strip())
    if not match:
        raise ValueError(f"Unsupported fact key format: {fact_key}")
    kind = match.group(1)
    x = int(match.group(2))
    y = int(match.group(3))
    cond_term = match.group(4)
    if cond_term == "empty":
        cond = ()
    elif cond_term.startswith("s"):
        cond = tuple(sorted(int(part) for part in cond_term[1:].split("y") if part))
    else:
        raise ValueError(f"Unsupported conditioning set: {cond_term}")
    return kind, x, y, cond


_TRUE_BN_CACHE: dict[str, tuple[nx.DiGraph, tuple[str, ...]]] = {}
_DSEP_CACHE: dict[tuple[str, int, int, tuple[int, ...]], bool] = {}


def _find_bif_path(dataset: str) -> Path:
    candidates = [
        REPO_ROOT / 'datasets' / 'bayesian',
        REPO_ROOT / 'bayesian',
        REPO_ROOT.parent / 'ABAPC-LLM-1' / 'datasets' / 'bayesian',
        REPO_ROOT.parent / 'ABAPC-LLM-1' / 'bayesian',
    ]
    for base in candidates:
        if not base.exists():
            continue
        direct = [p for p in base.rglob(f'{dataset}.bif') if p.is_file()]
        if direct:
            return direct[0]
        nested = [p for p in base.rglob(f'{dataset}.bif/{dataset}.bif') if p.is_file()]
        if nested:
            return nested[0]
    raise FileNotFoundError(f'Could not locate {dataset}.bif under known data roots')


def load_true_bn(dataset: str) -> tuple[nx.DiGraph, tuple[str, ...]]:
    cached = _TRUE_BN_CACHE.get(dataset)
    if cached is not None:
        return cached
    bif_path = _find_bif_path(dataset)
    model = BIFReader(str(bif_path)).get_model()
    sorted_nodes = tuple(sorted(model.nodes()))
    idx = {name: i for i, name in enumerate(sorted_nodes)}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(sorted_nodes)))
    graph.add_edges_from((idx[src], idx[dst]) for src, dst in model.edges())
    _TRUE_BN_CACHE[dataset] = (graph, sorted_nodes)
    return graph, sorted_nodes


def fact_truth_label(dataset: str, fact_key: str) -> str:
    kind, x, y, cond = parse_fact_key(fact_key)
    cache_key = (dataset, x, y, cond)
    if cache_key in _DSEP_CACHE:
        separated = _DSEP_CACHE[cache_key]
    else:
        graph, _ = load_true_bn(dataset)
        separated = bool(nx.algorithms.d_separated(graph, {x}, {y}, set(cond)))
        _DSEP_CACHE[cache_key] = separated
    if kind == "indep":
        return "true" if separated else "false"
    return "true" if not separated else "false"


def read_fact_keys(run_dir: Path) -> list[str]:
    facts_path = run_dir / "facts.lp"
    if not facts_path.exists():
        return []
    fact_keys: list[str] = []
    for raw_line in facts_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("#external "):
            fact_keys.append(line[len("#external ") :].strip())
    return fact_keys


def scenario_dir(version: str, dataset: str) -> Path:
    return RESULTS_DIR / f"abapc_{version}_{dataset}"


def load_runs(version: str, dataset: str) -> dict[int, dict]:
    runs_dir = scenario_dir(version, dataset) / "runs"
    rows: dict[int, dict] = {}
    if not runs_dir.exists():
        return rows
    for run_summary_path in sorted(runs_dir.glob("run_*_seed_*/run_summary.json")):
        data = json.loads(run_summary_path.read_text())
        run_dir = run_summary_path.parent
        removed = set(data.get("facts", {}).get("removed_fact_keys", []))
        total_facts = set(read_fact_keys(run_dir))
        kept = total_facts - removed if total_facts else set()
        rows[int(data["seed"])] = {
            "run_idx": int(data.get("run_idx", 0)),
            "run_dir": str(run_dir),
            "summary": data,
            "removed": removed,
            "kept": kept,
            "total": total_facts,
        }
    return rows


def truth_counts(dataset: str, fact_keys: set[str]) -> dict[str, int]:
    counts = {"true": 0, "false": 0}
    for fact_key in fact_keys:
        counts[fact_truth_label(dataset, fact_key)] += 1
    return counts


def _num(value, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except Exception:
        return float(default)


def per_method_rows(dataset: str, method_name: str, seed_map: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for seed, payload in sorted(seed_map.items()):
        summary = payload["summary"]
        removed = payload["removed"]
        kept = payload["kept"]
        removed_truth = truth_counts(dataset, removed)
        kept_truth = truth_counts(dataset, kept)
        reground = summary.get("reground", {})
        rows.append({
            "dataset": dataset,
            "method": method_name,
            "seed": seed,
            "run_idx": int(summary.get("run_idx", 0)),
            "elapsed_sec": _num(summary.get("elapsed_sec", 0.0)),
            "dag_F1": _num(summary.get("dag", {}).get("F1", 0.0)),
            "dag_SHD": _num(summary.get("dag", {}).get("shd", 0.0)),
            "cpdag_F1": _num(summary.get("cpdag", {}).get("F1", 0.0)),
            "cpdag_SHD": _num(summary.get("cpdag", {}).get("shd", 0.0)),
            "remove_n": int(summary.get("remove_n", 0) or 0),
            "removed_fact_count": len(removed),
            "kept_fact_count": len(kept),
            "removed_true_count": removed_truth["true"],
            "removed_false_count": removed_truth["false"],
            "kept_true_count": kept_truth["true"],
            "kept_false_count": kept_truth["false"],
            "removed_indep_count": int(summary.get("facts", {}).get("removed_indep_count", 0)),
            "removed_dep_count": int(summary.get("facts", {}).get("removed_dep_count", 0)),
            "fully_released_pair_count": int(summary.get("facts", {}).get("fully_released_pair_count", 0)),
            "approximate": bool(summary.get("approximate_remove_search", False)),
            "solver_backend": summary.get("solver_backend", ""),
            "pre_grounding": reground.get("pre_grounding"),
            "disable_reground": reground.get("disable_reground"),
            "reground_eligible_count": int(reground.get("reground_eligible_count", 0) or 0),
            "reground_performed_count": int(reground.get("reground_performed_count", 0) or 0),
            "reground_skipped_count": int(reground.get("reground_skipped_count", 0) or 0),
            "reground_skipped_indep_count": int(reground.get("reground_skipped_indep_count", 0) or 0),
            "run_dir": payload["run_dir"],
        })
    return rows


def pairwise_rows(dataset: str, left_name: str, left_map: dict[int, dict], right_name: str, right_map: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for seed in sorted(set(left_map) & set(right_map)):
        left = left_map[seed]
        right = right_map[seed]
        only_left = left["removed"] - right["removed"]
        only_right = right["removed"] - left["removed"]
        overlap = left["removed"] & right["removed"]
        left_truth = truth_counts(dataset, only_left)
        right_truth = truth_counts(dataset, only_right)
        overlap_truth = truth_counts(dataset, overlap)
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "left_method": left_name,
            "right_method": right_name,
            "left_removed_count": len(left["removed"]),
            "right_removed_count": len(right["removed"]),
            "overlap_removed_count": len(overlap),
            "only_left_removed_count": len(only_left),
            "only_right_removed_count": len(only_right),
            "only_left_true_count": left_truth["true"],
            "only_left_false_count": left_truth["false"],
            "only_right_true_count": right_truth["true"],
            "only_right_false_count": right_truth["false"],
            "overlap_true_count": overlap_truth["true"],
            "overlap_false_count": overlap_truth["false"],
            "cpdag_F1_delta": _num(left["summary"].get("cpdag", {}).get("F1", 0.0)) - _num(right["summary"].get("cpdag", {}).get("F1", 0.0)),
            "cpdag_SHD_delta": _num(left["summary"].get("cpdag", {}).get("shd", 0.0)) - _num(right["summary"].get("cpdag", {}).get("shd", 0.0)),
            "fully_released_pair_delta": int(left["summary"].get("facts", {}).get("fully_released_pair_count", 0)) - int(right["summary"].get("facts", {}).get("fully_released_pair_count", 0)),
        })
    return rows


def aggregate_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    out = df.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
    return out


def print_summary(method_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    print("\n=== Method Means ===")
    cols = [
        "dataset", "method", "elapsed_sec", "removed_fact_count", "removed_true_count", "removed_false_count",
        "fully_released_pair_count", "reground_skipped_count", "cpdag_F1", "cpdag_SHD",
    ]
    print(method_df[cols].to_string(index=False))
    print("\n=== Pairwise Means ===")
    cols = [
        "dataset", "left_method", "right_method", "only_left_removed_count", "only_left_true_count",
        "only_left_false_count", "only_right_removed_count", "only_right_true_count",
        "only_right_false_count", "cpdag_F1_delta", "cpdag_SHD_delta", "fully_released_pair_delta",
    ]
    print(pairwise_df[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    methods = {
        "ABAPC (bb)": args.bb_version,
        "ABAPC (nor)": args.nor_version,
        "ABAPC (orig)": args.orig_version,
    }

    method_rows: list[dict] = []
    pairwise_rows_all: list[dict] = []
    all_runs: dict[tuple[str, str], dict[int, dict]] = {}

    for dataset in args.datasets:
        for method_name, version in methods.items():
            runs = load_runs(version, dataset)
            all_runs[(dataset, method_name)] = runs
            method_rows.extend(per_method_rows(dataset, method_name, runs))

        pairwise_rows_all.extend(pairwise_rows(dataset, "ABAPC (nor)", all_runs[(dataset, "ABAPC (nor)")], "ABAPC (bb)", all_runs[(dataset, "ABAPC (bb)")]))
        pairwise_rows_all.extend(pairwise_rows(dataset, "ABAPC (orig)", all_runs[(dataset, "ABAPC (orig)")], "ABAPC (bb)", all_runs[(dataset, "ABAPC (bb)")]))
        pairwise_rows_all.extend(pairwise_rows(dataset, "ABAPC (orig)", all_runs[(dataset, "ABAPC (orig)")], "ABAPC (nor)", all_runs[(dataset, "ABAPC (nor)")]))

    seed_df = pd.DataFrame(method_rows).sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    pairwise_seed_df = pd.DataFrame(pairwise_rows_all).sort_values(["dataset", "left_method", "right_method", "seed"]).reset_index(drop=True)
    method_mean_df = aggregate_numeric(seed_df, ["dataset", "method"])
    pairwise_mean_df = aggregate_numeric(pairwise_seed_df, ["dataset", "left_method", "right_method"])

    out_dir = RESULTS_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    seed_df.to_csv(out_dir / f"{prefix}_seedwise.csv", index=False)
    pairwise_seed_df.to_csv(out_dir / f"{prefix}_pairwise_seedwise.csv", index=False)
    method_mean_df.to_csv(out_dir / f"{prefix}_method_means.csv", index=False)
    pairwise_mean_df.to_csv(out_dir / f"{prefix}_pairwise_means.csv", index=False)

    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(
            {
                "datasets": args.datasets,
                "versions": methods,
                "method_means": method_mean_df.to_dict(orient="records"),
                "pairwise_means": pairwise_mean_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print_summary(method_mean_df, pairwise_mean_df)
    print(f"\nWrote analysis files to {out_dir}")


if __name__ == "__main__":
    main()
