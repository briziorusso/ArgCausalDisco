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
DATASET_ORDER = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]
ALPHA_ORDER = ["alpha005", "alpha001"]
METHOD_ORDER = ["ABAPC (nor)", "ABAPC (bb)", "ABAPC (bb-nor)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched10 ABAPC fact universes and method behaviour between alpha=0.05 and alpha=0.01"
    )
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    parser.add_argument("--alpha005-nor-others-version", default="bnlearn_abapc_nor_matched10_others_gsq_graphmetrics")
    parser.add_argument("--alpha005-nor-child-version", default="child_abapc_nor_matched10_gsq_graphmetrics")
    parser.add_argument("--alpha005-bb-others-version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--alpha005-bb-child-version", default="child_abapc_bb_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument("--alpha005-bb-nor-others-version", default="bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--alpha005-bb-nor-child-version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument("--alpha001-nor-others-version", default="bnlearn_abapc_nor_matched10_others_gsq_alpha001")
    parser.add_argument("--alpha001-nor-child-version", default="child_abapc_nor_matched10_gsq_alpha001")
    parser.add_argument("--alpha001-bb-others-version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3_alpha001")
    parser.add_argument("--alpha001-bb-child-version", default="child_abapc_bb_matched10_gsq_searchv3_alpha001")
    parser.add_argument("--alpha001-bb-nor-others-version", default="bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_alpha001")
    parser.add_argument("--alpha001-bb-nor-child-version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3_alpha001")
    parser.add_argument("--output-prefix", default="matched10_alpha_fact_compare")
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
        REPO_ROOT / "datasets" / "bayesian",
        REPO_ROOT / "bayesian",
        REPO_ROOT.parent / "ABAPC-LLM-1" / "datasets" / "bayesian",
        REPO_ROOT.parent / "ABAPC-LLM-1" / "bayesian",
    ]
    for base in candidates:
        if not base.exists():
            continue
        direct = [p for p in base.rglob(f"{dataset}.bif") if p.is_file()]
        if direct:
            return direct[0]
        nested = [p for p in base.rglob(f"{dataset}.bif/{dataset}.bif") if p.is_file()]
        if nested:
            return nested[0]
    raise FileNotFoundError(f"Could not locate {dataset}.bif under known data roots")


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


def _method_label(method_key: str) -> str:
    return {
        "nor": "ABAPC (nor)",
        "bb": "ABAPC (bb)",
        "bb_nor": "ABAPC (bb-nor)",
    }[method_key]


def _version_for_dataset(args: argparse.Namespace, alpha_label: str, method_key: str, dataset: str) -> str:
    suffix = "child" if dataset == "child" else "others"
    attr = f"{alpha_label}-{method_key.replace('_', '-')}-{suffix}-version".replace("-", "_")
    return str(getattr(args, attr))


def _num(value, default: float = 0.0) -> float:
    try:
        return float(default if value is None else value)
    except Exception:
        return float(default)


def _truth_counts(dataset: str, fact_keys: list[str] | set[str]) -> dict[str, int]:
    counts = {"true": 0, "false": 0}
    for fact_key in fact_keys:
        counts[fact_truth_label(dataset, fact_key)] += 1
    return counts


def _kind_counts(dataset: str, fact_keys: list[str] | set[str]) -> dict[str, int]:
    counts = {
        "indep_true": 0,
        "indep_false": 0,
        "dep_true": 0,
        "dep_false": 0,
    }
    for fact_key in fact_keys:
        kind, _, _, _ = parse_fact_key(fact_key)
        truth = fact_truth_label(dataset, fact_key)
        counts[f"{kind}_{truth}"] += 1
    return counts


def _edge_pair_set(dataset: str) -> set[tuple[int, int]]:
    graph, _ = load_true_bn(dataset)
    return {tuple(sorted((int(src), int(dst)))) for src, dst in graph.edges()}


def _pair_to_text(pair: tuple[int, int]) -> str:
    return f"{pair[0]},{pair[1]}"


def _indep_pairs(fact_keys: list[str] | set[str]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for fact_key in fact_keys:
        kind, x, y, _ = parse_fact_key(fact_key)
        if kind == "indep":
            pairs.add(tuple(sorted((x, y))))
    return pairs


def _contradictory_pairs(fact_keys: list[str] | set[str]) -> set[tuple[int, int]]:
    indep: set[tuple[int, int]] = set()
    dep: set[tuple[int, int]] = set()
    for fact_key in fact_keys:
        kind, x, y, _ = parse_fact_key(fact_key)
        pair = tuple(sorted((x, y)))
        if kind == "indep":
            indep.add(pair)
        else:
            dep.add(pair)
    return indep & dep


def _reconstruct_removed_keys(fact_keys_in_order: list[str], remove_n: int) -> list[str]:
    remove_n = max(0, min(int(remove_n or 0), len(fact_keys_in_order)))
    return fact_keys_in_order[-remove_n:] if remove_n else []


def load_runs(version: str, dataset: str) -> dict[int, dict]:
    runs_dir = scenario_dir(version, dataset) / "runs"
    rows: dict[int, dict] = {}
    if not runs_dir.exists():
        return rows
    for run_summary_path in sorted(runs_dir.glob("run_*_seed_*/run_summary.json")):
        data = json.loads(run_summary_path.read_text())
        run_dir = run_summary_path.parent
        fact_keys_in_order = read_fact_keys(run_dir)
        removed_fact_keys = list(data.get("facts", {}).get("removed_fact_keys", []) or [])
        if not removed_fact_keys and fact_keys_in_order:
            removed_fact_keys = _reconstruct_removed_keys(fact_keys_in_order, int(data.get("remove_n", 0) or 0))
        removed = set(removed_fact_keys)
        kept = [fact_key for fact_key in fact_keys_in_order if fact_key not in removed]
        rows[int(data["seed"])] = {
            "run_idx": int(data.get("run_idx", 0)),
            "run_dir": str(run_dir),
            "summary": data,
            "fact_keys_in_order": fact_keys_in_order,
            "removed_keys": removed,
            "kept_keys": kept,
        }
    return rows


def initial_fact_profile(dataset: str, fact_keys_in_order: list[str]) -> dict[str, object]:
    edge_pairs = _edge_pair_set(dataset)
    kind_counts = _kind_counts(dataset, fact_keys_in_order)
    truth_counts = _truth_counts(dataset, fact_keys_in_order)
    indep_pairs = _indep_pairs(fact_keys_in_order)
    contradictory_pairs = _contradictory_pairs(fact_keys_in_order)
    wrong_block_pairs = indep_pairs & edge_pairs
    correct_block_pairs = indep_pairs - edge_pairs
    return {
        "facts_total": int(len(fact_keys_in_order)),
        "true_fact_count": int(truth_counts["true"]),
        "false_fact_count": int(truth_counts["false"]),
        "true_indep_count": int(kind_counts["indep_true"]),
        "false_indep_count": int(kind_counts["indep_false"]),
        "true_dep_count": int(kind_counts["dep_true"]),
        "false_dep_count": int(kind_counts["dep_false"]),
        "indep_pair_count": int(len(indep_pairs)),
        "wrong_block_pair_count": int(len(wrong_block_pairs)),
        "correct_block_pair_count": int(len(correct_block_pairs)),
        "contradictory_pair_count": int(len(contradictory_pairs)),
        "wrong_block_pairs": sorted(_pair_to_text(pair) for pair in wrong_block_pairs),
        "correct_block_pairs": sorted(_pair_to_text(pair) for pair in correct_block_pairs),
    }


def method_effect_profile(dataset: str, payload: dict, method_name: str) -> dict[str, object]:
    edge_pairs = _edge_pair_set(dataset)
    fact_keys_in_order = list(payload["fact_keys_in_order"])
    removed_keys = set(payload["removed_keys"])
    kept_keys = list(payload["kept_keys"])
    removed_truth = _truth_counts(dataset, removed_keys)
    kept_truth = _truth_counts(dataset, kept_keys)
    removed_kind = _kind_counts(dataset, removed_keys)
    kept_kind = _kind_counts(dataset, kept_keys)
    initial_indep_pairs = _indep_pairs(fact_keys_in_order)
    remaining_indep_pairs = _indep_pairs(kept_keys)
    released_block_pairs = initial_indep_pairs - remaining_indep_pairs
    wrong_block_pairs_initial = initial_indep_pairs & edge_pairs
    remaining_wrong_block_pairs = remaining_indep_pairs & edge_pairs
    released_wrong_block_pairs = released_block_pairs & edge_pairs
    released_correct_block_pairs = released_block_pairs - edge_pairs
    effective_wrong_block_pairs = wrong_block_pairs_initial if method_name == "ABAPC (bb-nor)" else remaining_wrong_block_pairs
    summary = payload["summary"]
    return {
        "removed_fact_count": int(len(removed_keys)),
        "kept_fact_count": int(len(kept_keys)),
        "removed_true_count": int(removed_truth["true"]),
        "removed_false_count": int(removed_truth["false"]),
        "kept_true_count": int(kept_truth["true"]),
        "kept_false_count": int(kept_truth["false"]),
        "removed_true_indep_count": int(removed_kind["indep_true"]),
        "removed_false_indep_count": int(removed_kind["indep_false"]),
        "removed_true_dep_count": int(removed_kind["dep_true"]),
        "removed_false_dep_count": int(removed_kind["dep_false"]),
        "kept_true_indep_count": int(kept_kind["indep_true"]),
        "kept_false_indep_count": int(kept_kind["indep_false"]),
        "kept_true_dep_count": int(kept_kind["dep_true"]),
        "kept_false_dep_count": int(kept_kind["dep_false"]),
        "released_block_pair_count": int(len(released_block_pairs)),
        "released_wrong_block_pair_count": int(len(released_wrong_block_pairs)),
        "released_correct_block_pair_count": int(len(released_correct_block_pairs)),
        "remaining_wrong_block_pair_count": int(len(remaining_wrong_block_pairs)),
        "effective_wrong_block_pair_count": int(len(effective_wrong_block_pairs)),
        "dag_F1": _num(summary.get("dag", {}).get("F1")),
        "dag_SHD": _num(summary.get("dag", {}).get("shd")),
        "dag_SID": _num(summary.get("dag", {}).get("sid")),
        "cpdag_F1": _num(summary.get("cpdag", {}).get("F1")),
        "cpdag_SHD": _num(summary.get("cpdag", {}).get("shd")),
        "cpdag_SID_low": _num(summary.get("cpdag", {}).get("sid_low")),
        "cpdag_SID_high": _num(summary.get("cpdag", {}).get("sid_high")),
        "elapsed_sec": _num(summary.get("elapsed_sec")),
        "remove_n": int(summary.get("remove_n", 0) or 0),
        "fully_released_pair_count_reported": int(summary.get("facts", {}).get("fully_released_pair_count", 0) or 0),
        "block_edge_mode": str(summary.get("reground", {}).get("block_edge_mode", "")),
    }


def compare_fact_sets(dataset: str, old_keys: list[str], new_keys: list[str]) -> dict[str, object]:
    old_set = set(old_keys)
    new_set = set(new_keys)
    old_only = old_set - new_set
    new_only = new_set - old_set
    old_only_truth = _truth_counts(dataset, old_only)
    new_only_truth = _truth_counts(dataset, new_only)
    old_only_kind = _kind_counts(dataset, old_only)
    new_only_kind = _kind_counts(dataset, new_only)
    return {
        "old_only_count": int(len(old_only)),
        "new_only_count": int(len(new_only)),
        "old_only_true_count": int(old_only_truth["true"]),
        "old_only_false_count": int(old_only_truth["false"]),
        "new_only_true_count": int(new_only_truth["true"]),
        "new_only_false_count": int(new_only_truth["false"]),
        "old_only_true_indep_count": int(old_only_kind["indep_true"]),
        "old_only_false_indep_count": int(old_only_kind["indep_false"]),
        "old_only_true_dep_count": int(old_only_kind["dep_true"]),
        "old_only_false_dep_count": int(old_only_kind["dep_false"]),
        "new_only_true_indep_count": int(new_only_kind["indep_true"]),
        "new_only_false_indep_count": int(new_only_kind["indep_false"]),
        "new_only_true_dep_count": int(new_only_kind["dep_true"]),
        "new_only_false_dep_count": int(new_only_kind["dep_false"]),
    }


def aggregate_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if col not in set(group_cols) and pd.api.types.is_numeric_dtype(df[col])]
    return df.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()


def _sort_dataset_alpha_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset"] = pd.Categorical(out["dataset"], DATASET_ORDER, ordered=True)
    if "alpha" in out.columns:
        out["alpha"] = pd.Categorical(out["alpha"], ALPHA_ORDER, ordered=True)
    if "method" in out.columns:
        out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    sort_cols = [col for col in ["dataset", "alpha", "method", "seed"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def build_versions(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for alpha_label in ALPHA_ORDER:
        versions[alpha_label] = {}
        for dataset in args.datasets:
            for method_key in ("nor", "bb", "bb_nor"):
                versions[alpha_label][f"{dataset}:{method_key}"] = _version_for_dataset(args, alpha_label, method_key, dataset)
    return versions


def method_gap_rows(method_seed_df: pd.DataFrame) -> pd.DataFrame:
    if method_seed_df.empty:
        return pd.DataFrame()
    bb = method_seed_df[method_seed_df["method"] == "ABAPC (bb)"].copy()
    bb_nor = method_seed_df[method_seed_df["method"] == "ABAPC (bb-nor)"].copy()
    merge_cols = ["dataset", "alpha", "seed"]
    use_cols = merge_cols + [
        "dag_F1",
        "dag_SHD",
        "cpdag_F1",
        "cpdag_SHD",
        "elapsed_sec",
        "released_wrong_block_pair_count",
        "released_correct_block_pair_count",
        "effective_wrong_block_pair_count",
        "remove_n",
    ]
    merged = bb[use_cols].merge(bb_nor[use_cols], on=merge_cols, suffixes=("_bb", "_bb_nor"))
    for metric in ["dag_F1", "dag_SHD", "cpdag_F1", "cpdag_SHD", "elapsed_sec", "effective_wrong_block_pair_count", "remove_n"]:
        merged[f"delta_{metric}_bb_minus_bb_nor"] = merged[f"{metric}_bb"] - merged[f"{metric}_bb_nor"]
    return _sort_dataset_alpha_method(merged)


def main() -> None:
    args = parse_args()
    versions = build_versions(args)

    initial_rows: list[dict] = []
    method_rows: list[dict] = []
    alpha_delta_rows: list[dict] = []
    consistency_rows: list[dict] = []

    canonical_payloads: dict[tuple[str, str], dict[int, dict]] = {}
    all_payloads: dict[tuple[str, str, str], dict[int, dict]] = {}

    for alpha_label in ALPHA_ORDER:
        for dataset in args.datasets:
            method_payloads: dict[str, dict[int, dict]] = {}
            for method_key in ("nor", "bb", "bb_nor"):
                version = versions[alpha_label][f"{dataset}:{method_key}"]
                payloads = load_runs(version, dataset)
                method_name = _method_label(method_key)
                all_payloads[(alpha_label, dataset, method_name)] = payloads
                method_payloads[method_name] = payloads
                for seed, payload in payloads.items():
                    method_rows.append(
                        {
                            "dataset": dataset,
                            "alpha": alpha_label,
                            "method": method_name,
                            "seed": seed,
                            "version": version,
                            **method_effect_profile(dataset, payload, method_name),
                        }
                    )

            canonical_runs = method_payloads["ABAPC (nor)"] or method_payloads["ABAPC (bb)"] or method_payloads["ABAPC (bb-nor)"]
            canonical_payloads[(alpha_label, dataset)] = canonical_runs
            for seed, payload in canonical_runs.items():
                base_facts = payload["fact_keys_in_order"]
                consistency = {"ABAPC (nor)": True, "ABAPC (bb)": True, "ABAPC (bb-nor)": True}
                for method_name, runs in method_payloads.items():
                    other = runs.get(seed)
                    if other is not None and other["fact_keys_in_order"] != base_facts:
                        consistency[method_name] = False
                consistency_rows.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha_label,
                        "seed": seed,
                        **{f"facts_match_{method.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}": value for method, value in consistency.items()},
                    }
                )
                initial_rows.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha_label,
                        "seed": seed,
                        "version": versions[alpha_label][f"{dataset}:nor"],
                        **initial_fact_profile(dataset, base_facts),
                    }
                )

    for dataset in args.datasets:
        old_runs = canonical_payloads.get(("alpha005", dataset), {})
        new_runs = canonical_payloads.get(("alpha001", dataset), {})
        for seed in sorted(set(old_runs) & set(new_runs)):
            old_payload = old_runs[seed]
            new_payload = new_runs[seed]
            old_profile = initial_fact_profile(dataset, old_payload["fact_keys_in_order"])
            new_profile = initial_fact_profile(dataset, new_payload["fact_keys_in_order"])
            row = {
                "dataset": dataset,
                "seed": seed,
                "alpha_old": "alpha005",
                "alpha_new": "alpha001",
                **compare_fact_sets(dataset, old_payload["fact_keys_in_order"], new_payload["fact_keys_in_order"]),
            }
            for key in [
                "facts_total",
                "true_fact_count",
                "false_fact_count",
                "true_indep_count",
                "false_indep_count",
                "true_dep_count",
                "false_dep_count",
                "indep_pair_count",
                "wrong_block_pair_count",
                "correct_block_pair_count",
                "contradictory_pair_count",
            ]:
                row[f"{key}_old"] = old_profile[key]
                row[f"{key}_new"] = new_profile[key]
                row[f"delta_{key}"] = float(new_profile[key]) - float(old_profile[key])
            alpha_delta_rows.append(row)

    initial_seed_df = _sort_dataset_alpha_method(pd.DataFrame(initial_rows))
    method_seed_df = _sort_dataset_alpha_method(pd.DataFrame(method_rows))
    alpha_delta_seed_df = _sort_dataset_alpha_method(pd.DataFrame(alpha_delta_rows))
    consistency_df = _sort_dataset_alpha_method(pd.DataFrame(consistency_rows))
    gap_seed_df = method_gap_rows(method_seed_df)

    initial_mean_df = aggregate_numeric(initial_seed_df, ["dataset", "alpha"])
    method_mean_df = aggregate_numeric(method_seed_df, ["dataset", "alpha", "method"])
    alpha_delta_mean_df = aggregate_numeric(alpha_delta_seed_df, ["dataset"])
    gap_mean_df = aggregate_numeric(gap_seed_df, ["dataset", "alpha"])

    out_dir = RESULTS_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    initial_seed_df.to_csv(out_dir / f"{prefix}_initial_seedwise.csv", index=False)
    method_seed_df.to_csv(out_dir / f"{prefix}_method_seedwise.csv", index=False)
    alpha_delta_seed_df.to_csv(out_dir / f"{prefix}_alpha_delta_seedwise.csv", index=False)
    consistency_df.to_csv(out_dir / f"{prefix}_fact_consistency_seedwise.csv", index=False)
    gap_seed_df.to_csv(out_dir / f"{prefix}_bb_gap_seedwise.csv", index=False)
    initial_mean_df.to_csv(out_dir / f"{prefix}_initial_means.csv", index=False)
    method_mean_df.to_csv(out_dir / f"{prefix}_method_means.csv", index=False)
    alpha_delta_mean_df.to_csv(out_dir / f"{prefix}_alpha_delta_means.csv", index=False)
    gap_mean_df.to_csv(out_dir / f"{prefix}_bb_gap_means.csv", index=False)

    summary_payload = {
        "datasets": args.datasets,
        "versions": versions,
        "initial_means": initial_mean_df.to_dict(orient="records"),
        "method_means": method_mean_df.to_dict(orient="records"),
        "alpha_delta_means": alpha_delta_mean_df.to_dict(orient="records"),
        "bb_gap_means": gap_mean_df.to_dict(orient="records"),
    }
    with open(out_dir / f"{prefix}_summary.json", "w") as handle:
        json.dump(summary_payload, handle, indent=2)

    print("\n=== Initial Fact Deltas (alpha001 - alpha005) ===")
    print(
        alpha_delta_mean_df[
            [
                "dataset",
                "delta_false_fact_count",
                "delta_false_indep_count",
                "delta_false_dep_count",
                "delta_wrong_block_pair_count",
                "new_only_false_indep_count",
                "new_only_false_dep_count",
                "old_only_false_indep_count",
                "old_only_false_dep_count",
            ]
        ].to_string(index=False)
    )
    print("\n=== Method Means ===")
    print(
        method_mean_df[
            [
                "dataset",
                "alpha",
                "method",
                "removed_false_indep_count",
                "released_wrong_block_pair_count",
                "effective_wrong_block_pair_count",
                "dag_F1",
                "cpdag_F1",
                "elapsed_sec",
            ]
        ].to_string(index=False)
    )
    print("\n=== BB vs BB-NOR Gap Means (bb - bb-nor) ===")
    print(
        gap_mean_df[
            [
                "dataset",
                "alpha",
                "delta_dag_F1_bb_minus_bb_nor",
                "delta_cpdag_F1_bb_minus_bb_nor",
                "delta_dag_SHD_bb_minus_bb_nor",
                "delta_cpdag_SHD_bb_minus_bb_nor",
                "delta_effective_wrong_block_pair_count_bb_minus_bb_nor",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote analysis files to {out_dir}")


if __name__ == "__main__":
    main()
