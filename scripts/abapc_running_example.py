#!/usr/bin/env python
"""Search or replay a 4-node running example where ABAPC improves over MPC.

Default setup follows the paper's running example:

- Variables: Education (E), Occupation (O), Race (R), Income (I)
- Ground truth: E->O, R->O, O->I, E->I
- CI test: G^2 (`gsq`)
- Initial strength: gamma(p, alpha) only (`S_weight=False`)

The script searches for an instance where:

1. MPC returns an output graph from an inconsistent fact set
   (no compatible DAG exists for all MPC facts), and
2. ABAPC using the incremental solver removes a weakest suffix of facts,
   restores compatibility, and returns a better witness DAG than MPC.

Use `--seed` and `--sample-size` to replay one concrete instance after search.

python scripts/abapc_running_example.py --seed 3354 --sample-size 12 --alpha 0.05 --min-ab-true-arrows 3 --min-ab-edge-count 3 --min-mpc-edge-count 1
ABAPC running example: seed=3354 sample_size=12 alpha=0.05 indep_test=gsq S_weight=False
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as e:  # pragma: no cover
    if e.name == "numpy":
        sys.stderr.write(
            "ERROR: Missing dependency 'numpy'. Run this inside the project's Python environment.\n"
        )
    raise
import networkx as nx
import pandas as pd

warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causalaba import compile_and_ground
from causalaba_increm import CausalABA as CausalABA_INC
from causalaba_mus import _normalize_ext_fact_key
from utils.graph_eval import graph_eval_from_accepted
from utils.graph_utils import (
    DAGMetrics,
    extract_test_elements_from_symbol,
    find_all_d_separations_sets,
    initial_strength,
    model_to_set_of_arrows,
)
from utils.data_utils import simulate_data_and_run_PC


N_NODES = 4
DISPLAY_LABELS = ["E", "O", "R", "I"]
INTERNAL_LABELS = [f"X{i + 1}" for i in range(N_NODES)]
B_TRUE = np.array(
    [
        [0, 1, 0, 1],  # E
        [0, 0, 0, 1],  # O
        [0, 1, 0, 0],  # R
        [0, 0, 0, 0],  # I
    ],
    dtype=int,
)


@contextlib.contextmanager
def _quiet(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _build_true_graph() -> tuple[nx.DiGraph, nx.DiGraph, dict[tuple[int, int, tuple[int, ...]], str]]:
    graph = nx.DiGraph(
        pd.DataFrame(B_TRUE.copy(), columns=INTERNAL_LABELS, index=INTERNAL_LABELS)
    )
    graph_int = nx.from_numpy_array(B_TRUE.copy(), create_using=nx.DiGraph)
    true_rel: dict[tuple[int, int, tuple[int, ...]], str] = {}
    for test in find_all_d_separations_sets(graph, verbose=False):
        x, S, y, dep_type = extract_test_elements_from_symbol(test)
        i, j = (x, y) if x < y else (y, x)
        true_rel[(int(i), int(j), tuple(sorted(int(s) for s in S)))] = str(dep_type)
    return graph, graph_int, true_rel


def _fact_sort_key(fact: tuple[str, float, float, bool]) -> tuple[float, str]:
    try:
        strength = float(fact[2])
    except Exception:
        strength = float("-inf")
    if math.isnan(strength):
        strength = float("-inf")
    return (-strength, str(fact[0]))


def _write_fact_files(
    out_dir: Path,
    facts: list[tuple[str, float, float, bool]],
) -> tuple[Path, Path, Path]:
    facts_path = out_dir / "facts.lp"
    facts_I_path = out_dir / "facts_I.lp"
    facts_wc_path = out_dir / "facts_wc.lp"
    with open(facts_path, "w") as f:
        for fact_str, _p, _I, _ok in facts:
            f.write(f"#external ext_{fact_str}\n")
    with open(facts_I_path, "w") as fI:
        for fact_str, _p, I, _ok in facts:
            fI.write(f"ext_{fact_str} I={I}, NA\n")
    with open(facts_wc_path, "w") as fWC:
        for fact_str, _p, I, _ok in facts:
            try:
                w = int(round(float(I) * 9_999_999))
            except Exception:
                w = 1
            w = max(1, min(9_999_999, w))
            fWC.write(f":~ ext_{fact_str} [-{w}]\n")
    return facts_path, facts_I_path, facts_wc_path


def _var_guard_fact_keys(n_nodes: int) -> list[str]:
    return [f"var({i})" for i in range(max(0, int(n_nodes)))]


def _augment_eval_with_var_guards(
    *,
    keys_in_file_order: list[str],
    accepted_keys: set[str],
    n_nodes: int,
) -> tuple[list[str], set[str]]:
    keys_aug = [str(k) for k in (keys_in_file_order or []) if k]
    accepted_aug = {str(k) for k in (accepted_keys or set()) if k}
    present = set(keys_aug)
    for vk in _var_guard_fact_keys(n_nodes):
        if vk not in present:
            keys_aug.append(vk)
            present.add(vk)
        accepted_aug.add(vk)
    return keys_aug, accepted_aug


def _graph_eval(
    *,
    n_nodes: int,
    G_true_int: nx.DiGraph,
    keys_in_file_order: list[str],
    accepted_keys: set[str],
    timeout_sec: float,
    cache: dict[frozenset[str], tuple[Any, ...]],
) -> tuple[Any, ...] | None:
    keys_eval, accepted_eval = _augment_eval_with_var_guards(
        keys_in_file_order=keys_in_file_order,
        accepted_keys=accepted_keys,
        n_nodes=n_nodes,
    )
    try:
        return graph_eval_from_accepted(
            n_nodes=n_nodes,
            G_true1=G_true_int,
            keys_in_file_order=keys_eval,
            accepted_keys=accepted_eval,
            timeout_sec=float(timeout_sec),
            threads=1,
            cache=cache,
        )
    except Exception:
        return None


def _extract_pc_edges(cg_local) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    directed: list[tuple[int, int]] = []
    undirected: list[tuple[int, int]] = []

    try:
        directed = [(int(a), int(b)) for (a, b) in (cg_local.find_fully_directed() or [])]
    except Exception:
        directed = []

    try:
        undirected = [
            (int(a), int(b))
            for (a, b) in (cg_local.find_undirected() or [])
            if int(a) < int(b)
        ]
    except Exception:
        undirected = []

    if not directed and not undirected:
        try:
            Gobj = getattr(cg_local, "G", None)
            mat = getattr(Gobj, "graph", None)
            if mat is not None:
                n = int(mat.shape[0])
                for i in range(n):
                    for j in range(i + 1, n):
                        a = int(mat[i, j])
                        b = int(mat[j, i])
                        if a == 0 and b == 0:
                            continue
                        if a == -1 and b == 1:
                            directed.append((i, j))
                            continue
                        if a == 1 and b == -1:
                            directed.append((j, i))
                            continue
                        undirected.append((i, j))
        except Exception:
            pass

    return sorted(set(directed)), sorted(set(undirected))


def _cpdag_matrix_from_edges(
    *,
    n_nodes: int,
    directed: list[tuple[int, int]],
    undirected: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((n_nodes, n_nodes), dtype=int)
    for u, v in directed:
        out[int(u), int(v)] = 1
    for u, v in undirected:
        out[int(u), int(v)] = 1
        out[int(v), int(u)] = 1
    return out


def _adj_matrix_from_edges(*, n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((n_nodes, n_nodes), dtype=int)
    for u, v in edges:
        out[int(u), int(v)] = 1
    return out


def _count_oriented_correct(edges: list[tuple[int, int]]) -> int:
    true_edges = {
        (int(i), int(j))
        for i in range(N_NODES)
        for j in range(N_NODES)
        if int(B_TRUE[i, j]) == 1
    }
    return int(sum(1 for (u, v) in edges if (int(u), int(v)) in true_edges))


def _select_best_ab_model(
    models_after: list[Any],
) -> tuple[list[tuple[int, int]], dict[str, Any], int, list[list[tuple[int, int]]]]:
    """Stabilize witness selection across nondeterministic optimal-model order."""
    best_edges: list[tuple[int, int]] = []
    best_metrics: dict[str, Any] = {}
    best_correct = 0
    best_score: tuple[float, int, float, int, tuple[tuple[int, int], ...]] | None = None
    all_edge_sets: list[list[tuple[int, int]]] = []

    for model in list(models_after or []):
        try:
            edges = sorted((int(u), int(v)) for (u, v) in model_to_set_of_arrows(model))
        except Exception:
            continue
        if not edges:
            continue
        all_edge_sets.append(list(edges))
        adj = _adj_matrix_from_edges(n_nodes=N_NODES, edges=list(edges))
        try:
            metrics = DAGMetrics(adj.copy(), B_TRUE.copy(), sid=False).metrics
        except Exception:
            metrics = {}
        shd = float(metrics.get("shd", 10**9))
        if math.isnan(shd):
            shd = float(10**9)
        f1 = float(metrics.get("F1", 0.0))
        if math.isnan(f1):
            f1 = 0.0
        correct = int(_count_oriented_correct(list(edges)))
        score = (float(shd), -int(correct), -float(f1), -len(edges), tuple(edges))
        if best_score is None or score < best_score:
            best_score = score
            best_edges = list(edges)
            best_metrics = dict(metrics)
            best_correct = int(correct)

    return best_edges, best_metrics, best_correct, all_edge_sets


def _pretty_fact_key(k: str) -> str:
    key = str(k or "").strip()
    key = _normalize_ext_fact_key(key)
    if key.startswith("ext_"):
        key = key[4:]
    m = re.match(r"^(dep|indep)\((\d+),(\d+),(empty|s[0-9y]+)\)$", key)
    if not m:
        return str(k)
    pred, xs, ys, sarg = m.groups()
    x = int(xs)
    y = int(ys)
    rel = "_||_" if pred == "indep" else "_|/|_"
    if sarg == "empty":
        S_fmt = "{}"
    else:
        idxs = [int(t) for t in sarg[1:].split("y") if t]
        S_fmt = "{" + ",".join(DISPLAY_LABELS[i] for i in idxs) + "}" if idxs else "{}"
    return f"{DISPLAY_LABELS[x]} {rel} {DISPLAY_LABELS[y]} | {S_fmt}"


def _representative_cpdag_edges(
    *,
    n_nodes: int,
    accepted_keys: set[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    indep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    dep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for k in accepted_keys:
        kk = _normalize_ext_fact_key(k)
        if kk.startswith("ext_"):
            kk = kk[4:]
        try:
            x, S, y, dep_type = extract_test_elements_from_symbol(f"{kk}.")
        except Exception:
            continue
        pair = (int(x), int(y))
        cond = tuple(sorted(int(s) for s in S))
        if "indep" in str(dep_type):
            indep_facts.setdefault(pair, set()).add(cond)
        else:
            dep_facts.setdefault(pair, set()).add(cond)

    def _cp_key(C: np.ndarray) -> tuple[int, ...]:
        Cc = np.array(C, dtype=int, copy=True)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                a = int(Cc[i, j])
                b = int(Cc[j, i])
                if a == 1 and b == 0:
                    continue
                if b == 1 and a == 0:
                    continue
                if a != 0 or b != 0:
                    Cc[i, j] = -1
                    Cc[j, i] = 0
        return tuple(int(x) for x in Cc.flatten())

    with tempfile.TemporaryDirectory() as td:
        facts_path = Path(td) / "accepted.lp"
        with open(facts_path, "w") as f:
            for vk in _var_guard_fact_keys(n_nodes):
                f.write(f"{vk}.\n")
            for k in sorted(accepted_keys):
                kk = _normalize_ext_fact_key(k)
                if not kk.endswith("."):
                    kk = kk + "."
                f.write(f"{kk}\n")

        ctl = compile_and_ground(
            n_nodes,
            str(facts_path),
            skeleton_rules_reduction=True,
            weak_constraints=False,
            indep_facts=indep_facts,
            dep_facts=dep_facts,
            opt_mode="ignore",
            out_n=0,
            show=["arrow"],
            pre_grounding=False,
            ext_flag=False,
            threads=1,
            deadline=None,
            timing_recorder=None,
        )

        counts: Counter[tuple[int, ...]] = Counter()
        mats: dict[tuple[int, ...], np.ndarray] = {}
        from utils.graph_utils import dag2cpdag

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                try:
                    arrows = model_to_set_of_arrows(model.symbols(shown=True))
                except Exception:
                    continue
                B = np.zeros((n_nodes, n_nodes), dtype=int)
                for (u, v) in arrows:
                    B[int(u), int(v)] = 1
                try:
                    C = dag2cpdag(B.copy())
                except Exception:
                    continue
                key = _cp_key(C)
                counts[key] += 1
                mats.setdefault(key, C)

        if not counts:
            return [], []

        best_key = counts.most_common(1)[0][0]
        C = mats[best_key]
        directed: list[tuple[int, int]] = []
        undirected: list[tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                if int(C[i, j]) == 1 and int(C[j, i]) == 0:
                    directed.append((i, j))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if int(C[i, j]) != 0 and int(C[j, i]) != 0:
                    undirected.append((i, j))
        return sorted(set(directed)), sorted(set(undirected))


def _pc_pvalue_map(cg_local) -> dict[tuple[int, int, tuple[int, ...]], float]:
    out: dict[tuple[int, int, tuple[int, ...]], float] = {}
    for x, y in combinations(range(N_NODES), 2):
        for S, p in list(cg_local.sepset[x, y]):
            out[(int(x), int(y), tuple(sorted(int(s) for s in S)))] = float(p)
    return out


def _build_facts_from_pc(
    *,
    cg_local,
    alpha: float,
    true_rel: dict[tuple[int, int, tuple[int, ...]], str],
) -> list[tuple[str, float, float, bool]]:
    p_map = _pc_pvalue_map(cg_local)
    facts: list[tuple[str, float, float, bool]] = []
    for (x, y, S_tup), p in p_map.items():
        dep_type_pc = "indep" if float(p) > float(alpha) else "dep"
        I = initial_strength(
            float(p),
            len(S_tup),
            float(alpha),
            0.5,
            N_NODES,
            S_weight=False,
        )
        s_str = "empty" if not S_tup else "s" + "y".join(str(i) for i in S_tup)
        fact_str = f"{dep_type_pc}({x},{y},{s_str})."
        truth_dep = true_rel.get((int(x), int(y), tuple(S_tup)))
        is_correct = bool(truth_dep == dep_type_pc)
        facts.append((fact_str, float(p), float(I), is_correct))
    return sorted(facts, key=_fact_sort_key)


def _evaluate_instance(
    *,
    seed: int,
    sample_size: int,
    alpha: float,
    indep_test: str,
    graph_eval_timeout: float,
    solve_timeout: float,
    quiet_solvers: bool,
    true_graph: nx.DiGraph,
    true_graph_int: nx.DiGraph,
    true_rel: dict[tuple[int, int, tuple[int, ...]], str],
) -> dict[str, Any] | None:
    try:
        with _quiet(quiet_solvers):
            data_local, cg_local = simulate_data_and_run_PC(
                true_graph,
                alpha,
                indep_test=indep_test,
                seed=int(seed),
                uc_rule=5,
                uc_priority=2,
                stable=True,
                sample_size=int(sample_size),
            )
    except Exception:
        return None

    facts_local = _build_facts_from_pc(cg_local=cg_local, alpha=alpha, true_rel=true_rel)
    if not facts_local:
        return None

    wrong = int(sum(1 for _fact, _p, _I, is_correct in facts_local if not is_correct))
    if wrong <= 0:
        return None

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        facts_path, facts_I_path, facts_wc_path = _write_fact_files(tmpdir, facts_local)
        keys_in_order = [
            _normalize_ext_fact_key(f"ext_{fact_str}")
            for fact_str, _p, _I, _ok in facts_local
        ]
        all_keys = [k for k in keys_in_order if k]
        all_set = set(all_keys)
        truth_map = {
            _normalize_ext_fact_key(f"ext_{fact_str}"): bool(is_correct)
            for fact_str, _p, _I, is_correct in facts_local
        }
        p_by_key = {
            _normalize_ext_fact_key(f"ext_{fact_str}"): float(p)
            for fact_str, p, _I, _ok in facts_local
        }
        I_by_key = {
            _normalize_ext_fact_key(f"ext_{fact_str}"): float(I)
            for fact_str, _p, I, _ok in facts_local
        }
        w_by_key: dict[str, int] = {}
        for fact_str, _p, I, _ok in facts_local:
            try:
                w = int(round(float(I) * 9_999_999))
            except Exception:
                w = 1
            w_by_key[_normalize_ext_fact_key(f"ext_{fact_str}")] = max(1, min(9_999_999, w))

        ge_cache: dict[frozenset[str], tuple[Any, ...]] = {}
        ge_all = _graph_eval(
            n_nodes=N_NODES,
            G_true_int=true_graph_int,
            keys_in_file_order=all_keys,
            accepted_keys=all_set,
            timeout_sec=graph_eval_timeout,
            cache=ge_cache,
        )
        if ge_all is None or bool(ge_all[-1]):
            return None
        full_incompatible = int(ge_all[0] or 0) == 0
        if not full_incompatible:
            return None

        with _quiet(quiet_solvers):
            solver_out = CausalABA_INC(
                N_NODES,
                str(facts_path),
                weak_constraints=True,
                search_for_models="first",
                opt_mode="optN",
                out_n=0,
                skeleton_rules_reduction=True,
                print_models=False,
                return_statistics=True,
                solve_timeout=float(solve_timeout),
            )

        if not isinstance(solver_out, (list, tuple)) or len(solver_out) < 5:
            return None

        models_after = list(solver_out[0] or [])
        remove_n = int(solver_out[3] or 0)
        profile = solver_out[4] if isinstance(solver_out[4], dict) else {}
        if remove_n <= 0 or not models_after:
            return None

        removed_keys = {
            _normalize_ext_fact_key(str(k))
            for k in list(profile.get("removed_fact_keys") or [])
            if str(k).strip()
        }
        if not removed_keys:
            removed_keys = set(all_keys[-remove_n:])
        accepted_keys = set(all_set) - set(removed_keys)
        if not accepted_keys:
            return None

        ge_ab = _graph_eval(
            n_nodes=N_NODES,
            G_true_int=true_graph_int,
            keys_in_file_order=all_keys,
            accepted_keys=accepted_keys,
            timeout_sec=graph_eval_timeout,
            cache=ge_cache,
        )
        if ge_ab is None or bool(ge_ab[-1]) or int(ge_ab[0] or 0) <= 0:
            return None

        pc_dir, pc_undir = _extract_pc_edges(cg_local)
        pc_cpdag = _cpdag_matrix_from_edges(
            n_nodes=N_NODES,
            directed=pc_dir,
            undirected=pc_undir,
        )
        try:
            pc_metrics = DAGMetrics(pc_cpdag.copy(), B_TRUE.copy(), sid=False).metrics
        except Exception:
            pc_metrics = {}

        try:
            ab_edges, ab_metrics, ab_oriented_correct, ab_optimal_models = _select_best_ab_model(
                models_after
            )
        except Exception:
            ab_edges, ab_metrics, ab_oriented_correct, ab_optimal_models = [], {}, 0, []
        if not ab_edges:
            return None

        try:
            ab_cpdag_dir, ab_cpdag_undir = _representative_cpdag_edges(
                n_nodes=N_NODES,
                accepted_keys=accepted_keys,
            )
            ab_cpdag = _cpdag_matrix_from_edges(
                n_nodes=N_NODES,
                directed=ab_cpdag_dir,
                undirected=ab_cpdag_undir,
            )
            ab_cpdag_metrics = DAGMetrics(ab_cpdag.copy(), B_TRUE.copy(), sid=False).metrics
        except Exception:
            ab_cpdag_dir, ab_cpdag_undir, ab_cpdag_metrics = [], [], {}

        pc_shd = float(pc_metrics.get("shd", 10**9))
        pc_f1 = float(pc_metrics.get("F1", 0.0))
        ab_shd = float(ab_metrics.get("shd", 10**9))
        ab_f1 = float(ab_metrics.get("F1", 0.0))
        ab_better_than_pc = (ab_shd < pc_shd) or (ab_f1 > pc_f1)
        if not ab_better_than_pc:
            return None

        return {
            "seed": int(seed),
            "sample_size": int(sample_size),
            "alpha": float(alpha),
            "indep_test": str(indep_test),
            "facts_local": list(facts_local),
            "facts_count": int(len(facts_local)),
            "wrong_count": int(wrong),
            "keys_in_order": list(all_keys),
            "truth_map": dict(truth_map),
            "p_by_key": dict(p_by_key),
            "I_by_key": dict(I_by_key),
            "w_by_key": dict(w_by_key),
            "all_incompatible": bool(full_incompatible),
            "all_eval": ge_all,
            "ab_eval": ge_ab,
            "remove_n": int(remove_n),
            "removed_keys": sorted(removed_keys),
            "accepted_keys": sorted(accepted_keys),
            "pc_dir": list(pc_dir),
            "pc_undir": list(pc_undir),
            "pc_metrics": dict(pc_metrics),
            "pc_oriented_correct": int(_count_oriented_correct(pc_dir)),
            "ab_edges": list(ab_edges),
            "ab_metrics": dict(ab_metrics),
            "ab_oriented_correct": int(ab_oriented_correct),
            "ab_optimal_model_count": int(len(ab_optimal_models)),
            "ab_optimal_models": list(ab_optimal_models),
            "ab_cpdag_dir": list(ab_cpdag_dir),
            "ab_cpdag_undir": list(ab_cpdag_undir),
            "ab_cpdag_metrics": dict(ab_cpdag_metrics),
            "solver_profile": dict(profile),
            "tmp_facts_I": str(facts_I_path),
            "tmp_facts_wc": str(facts_wc_path),
        }


def _candidate_score(case: dict[str, Any]) -> tuple[int, int, int, float, float, int, int]:
    ab_shd = float(case.get("ab_metrics", {}).get("shd", 10**9))
    ab_f1 = float(case.get("ab_metrics", {}).get("F1", 0.0))
    if math.isnan(ab_f1):
        ab_f1 = 0.0
    pc_shd = float(case.get("pc_metrics", {}).get("shd", 10**9))
    exact = 1 if ab_shd == 0 else 0
    ab_true = int(case.get("ab_oriented_correct", 0))
    ab_edges = len(list(case.get("ab_edges") or []))
    delta_shd = pc_shd - ab_shd
    return (
        -exact,
        -ab_true,
        -ab_edges,
        -float(delta_shd),
        -float(ab_f1),
        int(case.get("remove_n", 10**9)),
        int(case.get("wrong_count", 10**9)),
    )


def _mpc_edge_count(case: dict[str, Any]) -> int:
    return int(len(list(case.get("pc_dir") or [])) + len(list(case.get("pc_undir") or [])))


def _alpha_tag(alpha: float) -> str:
    return f"a{float(alpha):.3f}".replace(".", "p")


def _save_case(case: dict[str, Any], out_root: Path) -> Path:
    scenario_dir = out_root / (
        f"seed{int(case['seed'])}_ss{int(case['sample_size'])}_{_alpha_tag(float(case['alpha']))}_"
        f"{case['indep_test']}_noweight"
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)

    facts_path, facts_I_path, facts_wc_path = _write_fact_files(
        scenario_dir,
        list(case["facts_local"]),
    )
    summary = {
        "seed": int(case["seed"]),
        "sample_size": int(case["sample_size"]),
        "alpha": float(case["alpha"]),
        "indep_test": str(case["indep_test"]),
        "strength_S_weight": False,
        "facts_count": int(case["facts_count"]),
        "wrong_count": int(case["wrong_count"]),
        "all_incompatible": bool(case["all_incompatible"]),
        "all_eval": list(case["all_eval"]) if isinstance(case["all_eval"], (list, tuple)) else case["all_eval"],
        "ab_eval": list(case["ab_eval"]) if isinstance(case["ab_eval"], (list, tuple)) else case["ab_eval"],
        "remove_n": int(case["remove_n"]),
        "removed_keys": list(case["removed_keys"]),
        "accepted_keys": list(case["accepted_keys"]),
        "pc_dir": list(case["pc_dir"]),
        "pc_undir": list(case["pc_undir"]),
        "pc_metrics": dict(case["pc_metrics"]),
        "ab_edges": list(case["ab_edges"]),
        "ab_metrics": dict(case["ab_metrics"]),
        "ab_optimal_model_count": int(case.get("ab_optimal_model_count", 0)),
        "ab_optimal_models": list(case.get("ab_optimal_models") or []),
        "ab_cpdag_dir": list(case["ab_cpdag_dir"]),
        "ab_cpdag_undir": list(case["ab_cpdag_undir"]),
        "ab_cpdag_metrics": dict(case["ab_cpdag_metrics"]),
        "solver_profile": dict(case["solver_profile"]),
        "facts_files": {
            "facts": str(facts_path),
            "facts_I": str(facts_I_path),
            "facts_wc": str(facts_wc_path),
        },
    }
    with open(scenario_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return scenario_dir


def _fmt_metric(metrics: dict[str, Any], key: str) -> str:
    if not isinstance(metrics, dict):
        return "n/a"
    value = metrics.get(key)
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "n/a"
        if isinstance(value, int):
            return str(int(value))
        value_f = float(value)
        if math.isnan(value_f):
            return "n/a"
        return f"{value_f:.3f}"
    except Exception:
        return str(value)


def _print_edges(title: str, edges: list[tuple[int, int]], *, undirected: bool = False) -> None:
    print(f"{title} (n={len(edges)}):")
    for u, v in edges:
        if undirected:
            print(f"  {DISPLAY_LABELS[int(u)]} -- {DISPLAY_LABELS[int(v)]}")
        else:
            print(f"  {DISPLAY_LABELS[int(u)]} -> {DISPLAY_LABELS[int(v)]}")


def _print_fact_block(title: str, keys: list[str], case: dict[str, Any]) -> None:
    print(f"{title} (n={len(keys)}):")
    for key in keys:
        p = case["p_by_key"].get(key)
        I = case["I_by_key"].get(key)
        truth = case["truth_map"].get(key)
        truth_label = "true" if truth is True else "false" if truth is False else "unknown"
        p_str = f"{float(p):.4f}" if p is not None else "NA"
        I_str = f"{float(I):.4f}" if I is not None else "NA"
        print(f"  {_pretty_fact_key(key):<22} p={p_str:<7} I={I_str:<7} {truth_label}")


def _print_case(case: dict[str, Any], saved_dir: Path) -> None:
    removed = list(case["removed_keys"])
    accepted = list(case["accepted_keys"])
    facts_all = list(case["keys_in_order"])

    print(
        f"ABAPC running example: seed={case['seed']} sample_size={case['sample_size']} "
        f"alpha={case['alpha']} indep_test={case['indep_test']} S_weight=False"
    )
    print("Ground truth edges: E->O, R->O, O->I, E->I")
    print(
        f"All MPC facts incompatible: {case['all_incompatible']} "
        f"(compatible_dags={case['all_eval'][0]}, compatible_cpdags={case['all_eval'][6]})"
    )
    print(
        f"ABAPC accepted subset compatible: dags={case['ab_eval'][0]} cpdags={case['ab_eval'][6]} "
        f"true_dag_in_compat={case['ab_eval'][5]} true_cpdag_in_compat={case['ab_eval'][7]}"
    )
    print(
        f"MPC CPDAG metrics: shd={_fmt_metric(case['pc_metrics'], 'shd')} "
        f"F1={_fmt_metric(case['pc_metrics'], 'F1')} "
        f"oriented_correct={case['pc_oriented_correct']}"
    )
    print(
        f"ABAPC witness DAG metrics: shd={_fmt_metric(case['ab_metrics'], 'shd')} "
        f"F1={_fmt_metric(case['ab_metrics'], 'F1')} "
        f"oriented_correct={case['ab_oriented_correct']}"
    )
    print(f"ABAPC optimal models enumerated: {case.get('ab_optimal_model_count', 0)}")
    print(
        f"ABAPC representative CPDAG metrics: shd={_fmt_metric(case['ab_cpdag_metrics'], 'shd')} "
        f"F1={_fmt_metric(case['ab_cpdag_metrics'], 'F1')}"
    )
    print(f"Facts removed by ABAPC: {case['remove_n']} of {case['facts_count']}")
    print(f"Saved instance: {saved_dir}")
    print()

    _print_edges("MPC directed edges", list(case["pc_dir"]))
    _print_edges("MPC undirected edges", list(case["pc_undir"]), undirected=True)
    _print_edges("ABAPC witness DAG edges", list(case["ab_edges"]))
    if case["ab_cpdag_dir"] or case["ab_cpdag_undir"]:
        _print_edges("ABAPC representative CPDAG directed edges", list(case["ab_cpdag_dir"]))
        _print_edges(
            "ABAPC representative CPDAG undirected edges",
            list(case["ab_cpdag_undir"]),
            undirected=True,
        )

    print()
    _print_fact_block("All executed MPC facts", facts_all, case)
    print()
    _print_fact_block("ABAPC removed facts", removed, case)
    print()
    _print_fact_block("ABAPC accepted facts", accepted, case)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep-test", default="gsq", choices=["fisherz", "chisq", "gsq", "kci", "fastkci", "rcit"])
    parser.add_argument("--min-ab-true-arrows", type=int, default=0)
    parser.add_argument("--min-ab-edge-count", type=int, default=0)
    parser.add_argument("--min-mpc-edge-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Replay a single seed.")
    parser.add_argument("--sample-size", type=int, default=None, help="Replay a single sample size.")
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--seed-end", type=int, default=8000)
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default="10,15,20,30,50,100,200",
        help="Comma-separated sample sizes to search.",
    )
    parser.add_argument("--max-evals", type=int, default=400)
    parser.add_argument("--graph-eval-timeout", type=float, default=8.0)
    parser.add_argument("--solve-timeout", type=float, default=20.0)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "abapc_running_example",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print solver logs instead of suppressing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.verbose):
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        for name in ("pgmpy", "causalaba_increm", "causalaba", "root"):
            logging.getLogger(name).setLevel(logging.ERROR)
    true_graph, true_graph_int, true_rel = _build_true_graph()
    quiet_solvers = not bool(args.verbose)

    if args.seed is not None or args.sample_size is not None:
        if args.seed is None or args.sample_size is None:
            raise SystemExit("Provide both --seed and --sample-size to replay a single instance.")
        case = _evaluate_instance(
            seed=int(args.seed),
            sample_size=int(args.sample_size),
            alpha=float(args.alpha),
            indep_test=str(args.indep_test),
            graph_eval_timeout=float(args.graph_eval_timeout),
            solve_timeout=float(args.solve_timeout),
            quiet_solvers=quiet_solvers,
            true_graph=true_graph,
            true_graph_int=true_graph_int,
            true_rel=true_rel,
        )
        if case is None:
            print(
                f"No suitable ABAPC>MPC instance for seed={args.seed} sample_size={args.sample_size}."
            )
            return 1
        if int(case.get("ab_oriented_correct", 0)) < int(args.min_ab_true_arrows):
            print(
                f"Instance fails --min-ab-true-arrows={args.min_ab_true_arrows}: "
                f"got {case.get('ab_oriented_correct', 0)}."
            )
            return 1
        if len(list(case.get("ab_edges") or [])) < int(args.min_ab_edge_count):
            print(
                f"Instance fails --min-ab-edge-count={args.min_ab_edge_count}: "
                f"got {len(list(case.get('ab_edges') or []))}."
            )
            return 1
        if _mpc_edge_count(case) < int(args.min_mpc_edge_count):
            print(
                f"Instance fails --min-mpc-edge-count={args.min_mpc_edge_count}: "
                f"got {_mpc_edge_count(case)}."
            )
            return 1
        saved_dir = _save_case(case, Path(args.out_root))
        _print_case(case, saved_dir)
        return 0

    sample_sizes = [
        int(part.strip())
        for part in str(args.sample_sizes).split(",")
        if part.strip()
    ]
    evaluated = 0
    best: dict[str, Any] | None = None
    best_score: tuple[int, float, float, int, int] | None = None

    for sample_size in sample_sizes:
        for seed in range(int(args.seed_start), int(args.seed_end)):
            if evaluated >= int(args.max_evals):
                break
            evaluated += 1
            case = _evaluate_instance(
                seed=int(seed),
                sample_size=int(sample_size),
                alpha=float(args.alpha),
                indep_test=str(args.indep_test),
                graph_eval_timeout=float(args.graph_eval_timeout),
                solve_timeout=float(args.solve_timeout),
                quiet_solvers=quiet_solvers,
                true_graph=true_graph,
                true_graph_int=true_graph_int,
                true_rel=true_rel,
            )
            if case is None:
                continue
            if int(case.get("ab_oriented_correct", 0)) < int(args.min_ab_true_arrows):
                continue
            if len(list(case.get("ab_edges") or [])) < int(args.min_ab_edge_count):
                continue
            if _mpc_edge_count(case) < int(args.min_mpc_edge_count):
                continue
            score = _candidate_score(case)
            if best is None or score < best_score:
                best = case
                best_score = score
                print(
                    "[search] candidate "
                    f"seed={case['seed']} sample_size={case['sample_size']} "
                    f"wrong={case['wrong_count']} remove_n={case['remove_n']} "
                    f"mpc_edges={_mpc_edge_count(case)} "
                    f"ab_true={case['ab_oriented_correct']} "
                    f"ab_edges={len(list(case.get('ab_edges') or []))} "
                    f"mpc_shd={_fmt_metric(case['pc_metrics'], 'shd')} "
                    f"ab_shd={_fmt_metric(case['ab_metrics'], 'shd')} "
                    f"ab_f1={_fmt_metric(case['ab_metrics'], 'F1')}"
                )
                if (
                    float(case.get("ab_metrics", {}).get("shd", 10**9)) == 0.0
                    or int(case.get("ab_oriented_correct", 0)) >= 4
                ):
                    break
        if best is not None and (
            float(best.get("ab_metrics", {}).get("shd", 10**9)) == 0.0
            or int(best.get("ab_oriented_correct", 0)) >= 4
        ):
            break

    if best is None:
        print(
            f"No suitable instance found in budget (evals={evaluated}, seeds={args.seed_start}:{args.seed_end}, sample_sizes={sample_sizes})."
        )
        return 1

    saved_dir = _save_case(best, Path(args.out_root))
    print()
    _print_case(best, saved_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
