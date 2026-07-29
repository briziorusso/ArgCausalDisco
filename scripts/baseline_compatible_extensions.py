#!/usr/bin/env python3
"""Enumerate DAG extensions of saved single-output endpoint graphs.

The matched baseline runner stores PC-family and FGS endpoint matrices.  An
undirected endpoint graph represents a set of DAGs, so treating its directed
part as one DAG is not a set-valued comparison.  This module constructs one
consistent extension with the Dor--Tarsi sink-removal algorithm, enumerates
the remaining Markov-equivalent DAGs through covered-edge reversals, and
summarises their reconstruction metrics.

The datasets in this paper have at most eight nodes, so exhaustive MEC
enumeration is bounded by 8! DAGs for a complete undirected component.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
from typing import Any

import networkx as nx
import numpy as np

from scripts.run_matched_baseline_experiments import _estimate_to_cpdag_for_metrics
from utils.graph_utils import DAGMetrics, is_dag


_CI_FACT_RE = re.compile(
    r"^#external\s+ext_(indep|dep)\((\d+),(\d+),([A-Za-z0-9_]+)\)\."
)


def parse_ci_facts(path: str | Path) -> list[tuple[bool, int, int, frozenset[int]]]:
    """Parse the complete tested CI assignment from a CausalABA facts file."""

    facts: list[tuple[bool, int, int, frozenset[int]]] = []
    for line in Path(path).read_text().splitlines():
        match = _CI_FACT_RE.match(line.strip())
        if match is None:
            continue
        relation, x_text, y_text, set_name = match.groups()
        conditioning = frozenset(
            int(value) for value in re.findall(r"\d+", set_name)
        ) if set_name != "empty" else frozenset()
        facts.append((relation == "indep", int(x_text), int(y_text), conditioning))
    if not facts:
        raise ValueError(f"No CI facts found in {path}")
    return facts


def dag_satisfies_ci_facts(
    dag: np.ndarray,
    facts: list[tuple[bool, int, int, frozenset[int]]],
) -> bool:
    """Return whether a DAG satisfies every tested independence/dependence."""

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(dag)))
    graph.add_edges_from((int(x), int(y)) for x, y in zip(*np.where(np.asarray(dag) == 1)))
    for test_independent, x, y, conditioning in facts:
        separated = bool(nx.is_d_separator(graph, {x}, {y}, set(conditioning)))
        if separated != test_independent:
            return False
    return True


def _directed_and_undirected(cpdag: np.ndarray) -> tuple[set[tuple[int, int]], set[frozenset[int]]]:
    graph = np.asarray(cpdag)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Endpoint graph must be square")
    directed: set[tuple[int, int]] = set()
    undirected: set[frozenset[int]] = set()
    for i in range(len(graph)):
        if graph[i, i] != 0:
            raise ValueError("Endpoint graph has a self-loop")
        for j in range(i + 1, len(graph)):
            ij = graph[i, j] != 0
            ji = graph[j, i] != 0
            if not ij and not ji:
                continue
            if ij and ji:
                undirected.add(frozenset((i, j)))
            elif ij:
                directed.add((i, j))
            else:
                directed.add((j, i))
    return directed, undirected


def consistent_extension(cpdag: np.ndarray) -> np.ndarray:
    """Return one DAG extension, or raise when the endpoint graph is invalid."""

    directed, undirected = _directed_and_undirected(cpdag)
    n_nodes = len(cpdag)
    remaining = set(range(n_nodes))
    work_directed = set(directed)
    work_undirected = set(undirected)
    extension = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for parent, child in directed:
        extension[parent, child] = 1

    while remaining:
        selected: int | None = None
        selected_neighbours: set[int] = set()
        for node in sorted(remaining):
            children = {child for parent, child in work_directed if parent == node and child in remaining}
            if children:
                continue
            neighbours = {
                next(iter(edge - {node}))
                for edge in work_undirected
                if node in edge and (edge - {node}) <= remaining
            }
            parents = {parent for parent, child in work_directed if child == node and parent in remaining}
            adjacent = neighbours | parents
            clique_ok = all(
                a == b
                or (a, b) in work_directed
                or (b, a) in work_directed
                or frozenset((a, b)) in work_undirected
                for a in neighbours
                for b in adjacent
            )
            if clique_ok:
                selected = node
                selected_neighbours = neighbours
                break
        if selected is None:
            raise ValueError("Endpoint graph has no consistent DAG extension")
        for neighbour in selected_neighbours:
            extension[neighbour, selected] = 1
        remaining.remove(selected)
        work_directed = {
            (parent, child) for parent, child in work_directed
            if parent != selected and child != selected
        }
        work_undirected = {edge for edge in work_undirected if selected not in edge}

    if not is_dag(extension):
        raise ValueError("Constructed extension is cyclic")
    return extension


def enumerate_consistent_extensions(cpdag: np.ndarray) -> list[np.ndarray]:
    """Enumerate the Markov-equivalent DAGs consistent with a valid CPDAG."""

    directed, _ = _directed_and_undirected(cpdag)
    first = consistent_extension(cpdag)
    queue: deque[np.ndarray] = deque([first])
    seen = {first.tobytes()}
    extensions: list[np.ndarray] = []
    while queue:
        dag = queue.popleft()
        extensions.append(dag)
        parents = [set(np.flatnonzero(dag[:, node])) for node in range(len(dag))]
        for parent, child in zip(*np.where(dag == 1)):
            parent = int(parent)
            child = int(child)
            if parents[child] != parents[parent] | {parent}:
                continue
            candidate = dag.copy()
            candidate[parent, child] = 0
            candidate[child, parent] = 1
            if any(candidate[a, b] != 1 for a, b in directed):
                continue
            identity = candidate.tobytes()
            if identity not in seen:
                seen.add(identity)
                queue.append(candidate)
    return extensions


def summarise_artifact(path: str | Path, *, repo_root: Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    with np.load(artifact_path, allow_pickle=False) as artifact:
        estimate = np.asarray(artifact["W_est"])
        truth = np.asarray(artifact["B_true"])
    cpdag = _estimate_to_cpdag_for_metrics(estimate)
    try:
        extensions = enumerate_consistent_extensions(cpdag)
    except Exception as exc:  # invalid/cyclic endpoint outputs remain explicit
        return {
            "endpoint_valid": 0.0,
            "endpoint_invalid_reason": f"{type(exc).__name__}: {exc}",
            "n_cpdags_returned": 0.0,
            "n_dags_returned": 0.0,
            "n_cpdags_compat": float("nan"),
            "n_dags_compat": float("nan"),
            "dag_shd_avg": float("nan"),
            "dag_shd_best": float("nan"),
            "dag_shd_worst": float("nan"),
            "dag_f1_avg": float("nan"),
            "dag_f1_best": float("nan"),
            "dag_f1_worst": float("nan"),
            "true_dag_in_compat": float("nan"),
        }

    shd: list[float] = []
    f1: list[float] = []
    for dag in extensions:
        metrics = DAGMetrics(np.asarray(dag), np.asarray(truth), sid=False).metrics
        shd.append(float(metrics["shd"]))
        f1.append(float(metrics["F1"]) if np.isfinite(metrics["F1"]) else 0.0)
    truth_identity = np.asarray(truth, dtype=np.int8).tobytes()
    return {
        "endpoint_valid": 1.0,
        "endpoint_invalid_reason": "",
        "n_cpdags_returned": 1.0,
        "n_dags_returned": float(len(extensions)),
        # Test compatibility is audited separately against the actual CI trace.
        "n_cpdags_compat": float("nan"),
        "n_dags_compat": float("nan"),
        "dag_shd_avg": float(np.mean(shd)),
        "dag_shd_best": float(np.min(shd)),
        "dag_shd_worst": float(np.max(shd)),
        "dag_f1_avg": float(np.mean(f1)),
        "dag_f1_best": float(np.max(f1)),
        "dag_f1_worst": float(np.min(f1)),
        "true_dag_in_compat": float(any(dag.tobytes() == truth_identity for dag in extensions)),
    }


def enrich_progress_records(records, *, repo_root: Path):
    """Replace placeholder baseline DAG metrics with compatible-set metrics."""

    output = records.copy()
    if "endpoint_valid" not in output:
        output["endpoint_valid"] = np.nan
    if "endpoint_invalid_reason" not in output:
        output["endpoint_invalid_reason"] = ""
    if "n_cpdags_returned" not in output:
        output["n_cpdags_returned"] = np.nan
    if "n_dags_returned" not in output:
        output["n_dags_returned"] = np.nan
    eligible = output["method"].isin({"MPC", "FGS", "Shapley-PC"})
    for index, row in output.loc[eligible].iterrows():
        raw_path = str(row.get("raw_graph_path", "") or "")
        if not raw_path:
            output.at[index, "endpoint_valid"] = 0.0
            output.at[index, "endpoint_invalid_reason"] = "Missing raw graph artifact path"
            continue
        summary = summarise_artifact(raw_path, repo_root=repo_root)
        for name, value in summary.items():
            output.at[index, name] = value
    return output
