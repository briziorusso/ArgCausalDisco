#!/usr/bin/env python
"""
Compare baseline causalaba vs. incremental causalaba on the same instance.

Usage examples:
  python scripts/compare_solvers.py --n-nodes 5 --facts encodings/test_lps/five_node_sprinkler_PC_facts.lp --weak-constraints --fact-pct 0.27
  python scripts/compare_solvers.py --n-nodes 4 --facts results/test_abapc_four_node_example/facts.lp --weak-constraints --fact-pct 1.0
"""

from __future__ import annotations

import argparse
import sys
import resource
import time
import tracemalloc
from typing import Any, Callable
from pathlib import Path

# Ensure project root on path so local imports work when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .causalaba import CausalABA as CausalABA_Base
    from .causalaba_increm import CausalABA as CausalABA_Incr
except ImportError:  # pragma: no cover
    from causalaba import CausalABA as CausalABA_Base
    from causalaba_increm import CausalABA as CausalABA_Incr


def _run_with_metrics(
    label: str,
    solver: Callable[..., Any],
    n_nodes: int,
    facts: str | None,
    fact_pct: float,
    weak_constraints: bool,
    search_for_models: str,
    opt_mode: str,
    max_path_length: int | None,
    max_conditioning_size: int | None,
    collider_tree_depth: int | None,
    cycle_length: int | None,
) -> dict[str, Any]:
    """Run a solver and collect time and memory statistics."""
    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    models, _ = solver(
        n_nodes,
        facts,
        weak_constraints=weak_constraints,
        fact_pct=fact_pct,
        search_for_models=search_for_models,
        opt_mode=opt_mode,
        max_path_length=max_path_length,
        max_conditioning_size=max_conditioning_size,
        collider_tree_depth=collider_tree_depth,
        cycle_length=cycle_length,
        print_models=False,
    )
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in kilobytes on Linux, bytes on macOS; keep raw value for consistency
    return {
        "label": label,
        "elapsed_sec": elapsed,
        "models": len(models) if isinstance(models, list) else 0,
        "rss_delta": rss_after - rss_before,
        "rss_after": rss_after,
        "peak_tracemalloc_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs. incremental CausalABA")
    parser.add_argument("--n-nodes", type=int, required=True, help="Number of variables in the instance")
    parser.add_argument("--facts", type=str, required=True, help="Path to facts .lp file")
    parser.add_argument(
        "--weak-constraints", action="store_true", help="Enable weak constraints (loads *_wc.lp if present)"
    )
    parser.add_argument("--fact-pct", type=float, default=1.0, help="Fraction of facts to activate (default: 1.0)")
    parser.add_argument(
        "--search-for-models",
        type=str,
        default="No",
        choices=["No", "first", "all_subsets", "first_subsets"],
        help="Model search strategy (mirrors solver argument)",
    )
    parser.add_argument(
        "--opt-mode",
        type=str,
        default="optN",
        choices=["optN", "opt", "enum"],
        help="Optimization mode passed to the solver",
    )
    parser.add_argument("--max-path-length", type=int, default=None, help="Optional path length cutoff")
    parser.add_argument("--max-conditioning-size", type=int, default=None, help="Optional conditioning size cutoff")
    parser.add_argument("--collider-tree-depth", type=int, default=None, help="Optional collider depth bound")
    parser.add_argument("--cycle-length", type=int, default=None, help="Optional cycle length bound")
    args = parser.parse_args()

    runs = []
    runs.append(
        _run_with_metrics(
            "baseline",
            CausalABA_Base,
            args.n_nodes,
            args.facts,
            args.fact_pct,
            args.weak_constraints,
            args.search_for_models,
            args.opt_mode,
            args.max_path_length,
            args.max_conditioning_size,
            args.collider_tree_depth,
            args.cycle_length,
        )
    )
    runs.append(
        _run_with_metrics(
            "incremental",
            CausalABA_Incr,
            args.n_nodes,
            args.facts,
            args.fact_pct,
            args.weak_constraints,
            args.search_for_models,
            args.opt_mode,
            args.max_path_length,
            args.max_conditioning_size,
            args.collider_tree_depth,
            args.cycle_length,
        )
    )

    print("label,elapsed_sec,models,rss_delta,rss_after,peak_tracemalloc_bytes")
    for r in runs:
        print(
            f"{r['label']},{r['elapsed_sec']:.6f},{r['models']},"
            f"{r['rss_delta']},{r['rss_after']},{r['peak_tracemalloc_bytes']}"
        )


if __name__ == "__main__":
    main()
