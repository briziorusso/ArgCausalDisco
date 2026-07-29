from __future__ import annotations

import numpy as np
import pytest

from scripts.baseline_compatible_extensions import (
    consistent_extension,
    dag_satisfies_ci_facts,
    enumerate_consistent_extensions,
    parse_ci_facts,
)
from utils.graph_utils import is_dag


def test_three_node_undirected_chain_has_three_extensions() -> None:
    cpdag = np.array([
        [0, -1, 0],
        [-1, 0, -1],
        [0, -1, 0],
    ])
    extensions = enumerate_consistent_extensions(cpdag)
    assert len(extensions) == 3
    assert all(is_dag(dag) for dag in extensions)


def test_compelled_collider_has_one_extension() -> None:
    cpdag = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    extensions = enumerate_consistent_extensions(cpdag)
    assert len(extensions) == 1
    assert np.array_equal(extensions[0], cpdag)


def test_directed_cycle_has_no_extension() -> None:
    cyclic = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    with pytest.raises(ValueError, match="no consistent DAG extension"):
        consistent_extension(cyclic)


def test_ci_trace_compatibility_is_not_extension_count(tmp_path) -> None:
    facts_path = tmp_path / "facts.lp"
    facts_path.write_text(
        "#external ext_dep(0,1,empty).\n"
        "#external ext_indep(0,2,s1).\n"
    )
    facts = parse_ci_facts(facts_path)
    chain = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    assert dag_satisfies_ci_facts(chain, facts)

    contradictory_path = tmp_path / "contradictory.lp"
    contradictory_path.write_text(
        "#external ext_dep(0,1,empty).\n"
        "#external ext_indep(0,1,empty).\n"
    )
    assert not dag_satisfies_ci_facts(chain, parse_ci_facts(contradictory_path))
