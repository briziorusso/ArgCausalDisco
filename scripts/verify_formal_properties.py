#!/usr/bin/env python3
"""Dependency-free finite checks for the OptABA-PC formal claims.

The checker mirrors the implemented Bayes-ball transition rules, compares them
with a separate active-path definition of d-connection, evaluates stable models
of finite ground normal programs from the GL reduct, and audits the manuscript's
two four-node examples, including every challenge in the 14-fact trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import FrozenSet, Iterable, Iterator, Optional


Edge = tuple[int, int]
Graph = FrozenSet[Edge]


def powerset(items: Iterable[int]) -> Iterator[FrozenSet[int]]:
    values = tuple(items)
    for mask in range(1 << len(values)):
        yield frozenset(values[i] for i in range(len(values)) if mask & (1 << i))


def is_dag(n: int, edges: Graph) -> bool:
    children = [set() for _ in range(n)]
    indegree = [0] * n
    for parent, child in edges:
        children[parent].add(child)
        indegree[child] += 1
    frontier = [node for node in range(n) if indegree[node] == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    return visited == n


def labelled_dags(n: int) -> Iterator[Graph]:
    """Enumerate each simple labelled DAG exactly once."""
    pairs = tuple(combinations(range(n), 2))
    for choices in product((0, 1, 2), repeat=len(pairs)):
        edges = frozenset(
            (left, right) if choice == 1 else (right, left)
            for (left, right), choice in zip(pairs, choices)
            if choice
        )
        if is_dag(n, edges):
            yield edges


def graph_relations(n: int, edges: Graph) -> tuple[list[set[int]], list[set[int]]]:
    parents = [set() for _ in range(n)]
    children = [set() for _ in range(n)]
    for parent, child in edges:
        parents[child].add(parent)
        children[parent].add(child)
    return parents, children


def ancestors_of(n: int, edges: Graph, observed: FrozenSet[int]) -> set[int]:
    parents, _ = graph_relations(n, edges)
    ancestors = set(observed)
    frontier = list(observed)
    while frontier:
        node = frontier.pop()
        for parent in parents[node]:
            if parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)
    return ancestors


def implemented_bayes_ball(
    n: int, edges: Graph, start: int, target: int, observed: FrozenSet[int]
) -> bool:
    """Least closure of the four rules currently emitted by the implementation."""
    parents, children = graph_relations(n, edges)
    anc_observed = ancestors_of(n, edges, observed)
    global_colliders = {node for node in range(n) if len(parents[node]) >= 2}
    states = {("u", start), ("d", start)}
    frontier = list(states)
    while frontier:
        direction, node = frontier.pop()
        successors: list[tuple[str, int]] = []
        if direction == "u" and node not in observed:
            successors.extend(("u", parent) for parent in parents[node])
            successors.extend(("d", child) for child in children[node])
        elif direction == "d":
            if node in anc_observed and node in global_colliders:
                successors.extend(("u", parent) for parent in parents[node])
            if node not in observed:
                successors.extend(("d", child) for child in children[node])
        for state in successors:
            if state not in states:
                states.add(state)
                frontier.append(state)
    return ("u", target) in states or ("d", target) in states


def shachter_bottom_marks(
    n: int, edges: Graph, start: int, observed: FrozenSet[int]
) -> set[int]:
    """Shachter's Algorithm 2 specialized to probabilistic DAG nodes."""
    parents, children = graph_relations(n, edges)
    top_marked: set[int] = set()
    bottom_marked: set[int] = set()
    scheduled = {("u", start)}  # Shachter initializes as if from a child.
    frontier = list(scheduled)
    while frontier:
        direction, node = frontier.pop()
        successors: list[tuple[str, int]] = []
        if direction == "u" and node not in observed:
            if node not in top_marked:
                top_marked.add(node)
                successors.extend(("u", parent) for parent in parents[node])
            if node not in bottom_marked:
                bottom_marked.add(node)
                successors.extend(("d", child) for child in children[node])
        elif direction == "d":
            if node in observed and node not in top_marked:
                top_marked.add(node)
                successors.extend(("u", parent) for parent in parents[node])
            if node not in observed and node not in bottom_marked:
                bottom_marked.add(node)
                successors.extend(("d", child) for child in children[node])
        for state in successors:
            if state not in scheduled:
                scheduled.add(state)
                frontier.append(state)
    return bottom_marked


def reference_d_connected(
    n: int, edges: Graph, start: int, target: int, observed: FrozenSet[int]
) -> bool:
    """Independent reference: enumerate simple paths and apply the definition."""
    parents, children = graph_relations(n, edges)
    adjacent = [parents[node] | children[node] for node in range(n)]
    anc_observed = ancestors_of(n, edges, observed)

    def paths(node: int, path: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if node == target:
            yield path
            return
        for neighbour in adjacent[node]:
            if neighbour not in path:
                yield from paths(neighbour, path + (neighbour,))

    for path in paths(start, (start,)):
        active = True
        for previous, node, following in zip(path, path[1:], path[2:]):
            collider = (previous, node) in edges and (following, node) in edges
            if collider and node not in anc_observed:
                active = False
                break
            if not collider and node in observed:
                active = False
                break
        if active:
            return True
    return False


def check_bayes_ball() -> tuple[int, int]:
    dag_count = 0
    query_count = 0
    for n in range(2, 5):
        for edges in labelled_dags(n):
            dag_count += 1
            for start in range(n):
                for target in range(n):
                    if start == target:
                        continue
                    remaining = set(range(n)) - {start, target}
                    for observed in powerset(remaining):
                        query_count += 1
                        actual = implemented_bayes_ball(n, edges, start, target, observed)
                        shachter = target in shachter_bottom_marks(n, edges, start, observed)
                        expected = reference_d_connected(n, edges, start, target, observed)
                        assert actual == shachter, (
                            "ASP/Shachter simulation mismatch",
                            n,
                            edges,
                            start,
                            target,
                            observed,
                            actual,
                            shachter,
                        )
                        assert actual == expected, (
                            "Bayes-ball mismatch",
                            n,
                            edges,
                            start,
                            target,
                            observed,
                            actual,
                            expected,
                        )
    return dag_count, query_count


@dataclass(frozen=True)
class Rule:
    head: Optional[str]
    positive: FrozenSet[str] = frozenset()
    negative: FrozenSet[str] = frozenset()


def choice_rules(atom: str) -> tuple[Rule, Rule]:
    complement = f"__not_{atom}"
    return Rule(atom, negative=frozenset({complement})), Rule(
        complement, negative=frozenset({atom})
    )


def stable_models(program: tuple[Rule, ...]) -> list[FrozenSet[str]]:
    atoms = sorted(
        {rule.head for rule in program if rule.head is not None}
        | {atom for rule in program for atom in rule.positive | rule.negative}
    )
    models: list[FrozenSet[str]] = []
    for candidate in powerset(range(len(atoms))):
        interpretation = frozenset(atoms[index] for index in candidate)
        reduct = [
            Rule(rule.head, rule.positive)
            for rule in program
            if not (rule.negative & interpretation)
        ]
        closure: set[str] = set()
        changed = True
        while changed:
            changed = False
            for rule in reduct:
                if rule.head is not None and rule.positive <= closure and rule.head not in closure:
                    closure.add(rule.head)
                    changed = True
        if frozenset(closure) != interpretation:
            continue
        if any(rule.head is None and rule.positive <= closure for rule in reduct):
            continue
        models.append(interpretation)
    return models


def correction_sets(
    models: list[FrozenSet[str]], objectives: FrozenSet[str]
) -> list[FrozenSet[str]]:
    repairs = []
    for repair in powerset(range(len(objectives))):
        ordered = tuple(sorted(objectives))
        released = frozenset(ordered[index] for index in repair)
        retained = objectives - released
        if any(retained <= model for model in models):
            repairs.append(released)
    return repairs


def minimal_sets(sets: Iterable[FrozenSet[str]]) -> list[FrozenSet[str]]:
    values = list(sets)
    return [value for value in values if not any(other < value for other in values)]


def check_projection_theorem() -> tuple[int, int]:
    objectives = frozenset({"o1", "o2", "o3", "o4"})
    program = tuple(rule for atom in objectives for rule in choice_rules(atom)) + (
        Rule("a", frozenset({"o1"}), frozenset({"b"})),
        Rule("b", frozenset({"o2"}), frozenset({"c"})),
        Rule("c", frozenset({"o3"}), frozenset({"a"})),
        Rule(None, frozenset({"o4"}), frozenset({"d"})),
    )
    models = stable_models(program)
    repairs = correction_sets(models, objectives)
    mcses = minimal_sets(repairs)
    weights = {"o1": 5, "o2": 10, "o3": 15, "o4": 20}
    repair_cost = lambda repair: sum(weights[atom] for atom in repair)
    optimum = min(map(repair_cost, repairs))
    optimal_mcses = {repair for repair in mcses if repair_cost(repair) == optimum}
    model_cost = lambda model: sum(weights[atom] for atom in objectives - model)
    optimal_models = [model for model in models if model_cost(model) == optimum]
    projections = {objectives - model for model in optimal_models}
    assert projections == optimal_mcses == {frozenset({"o1", "o4"})}
    assert all(repair in mcses for repair in projections)
    assert all(repair_cost(objectives - model) == model_cost(model) for model in models)

    equal_weights = {atom: 5 for atom in objectives}
    equal_cost = lambda repair: sum(equal_weights[atom] for atom in repair)
    equal_optimum = min(map(equal_cost, repairs))
    equal_mcses = {repair for repair in mcses if equal_cost(repair) == equal_optimum}
    equal_models = [
        model
        for model in models
        if sum(equal_weights[atom] for atom in objectives - model) == equal_optimum
    ]
    assert {objectives - model for model in equal_models} == equal_mcses
    assert equal_optimum == 10  # two equal-weight facts are counted separately
    active_tuples = {(equal_weights[atom], 1, atom) for atom in {"o1", "o4"}}
    assert len(active_tuples) == 2 and sum(item[0] for item in active_tuples) == 10

    multi_objectives = frozenset({"o"})
    multi_program = choice_rules("o") + (
        Rule("x", negative=frozenset({"y"})),
        Rule("y", negative=frozenset({"x"})),
        Rule(None, positive=frozenset({"o"})),
    )
    multi_models = stable_models(multi_program)
    assert len(multi_models) == 2
    assert {multi_objectives - model for model in multi_models} == {
        frozenset({"o"})
    }
    return len(models), len(mcses)


@dataclass(frozen=True)
class Fact:
    name: str
    independent: bool
    left: int
    right: int
    observed: FrozenSet[int]
    weight: int


def running_example_facts() -> tuple[Fact, ...]:
    e, r, o, i = range(4)
    return (
        Fact("indep(E,O,{I})", True, e, o, frozenset({i}), 1000),
        Fact("indep(R,I,{O})", True, r, i, frozenset({o}), 1000),
        Fact("dep(O,R,{})", False, o, r, frozenset(), 990),
        Fact("dep(O,R,{E})", False, o, r, frozenset({e}), 950),
        Fact("dep(E,O,{})", False, e, o, frozenset(), 920),
        Fact("dep(E,O,{R})", False, e, o, frozenset({r}), 920),
        Fact("dep(E,I,{})", False, e, i, frozenset(), 840),
        Fact("dep(E,I,{R})", False, e, i, frozenset({r}), 560),
        Fact("dep(R,I,{})", False, r, i, frozenset(), 540),
        Fact("indep(E,I,{O})", True, e, i, frozenset({o}), 530),
        Fact("indep(R,I,{E})", True, r, i, frozenset({e}), 530),
        Fact("dep(O,R,{I})", False, o, r, frozenset({i}), 520),
        Fact("indep(O,I,{})", True, o, i, frozenset(), 510),
        Fact("indep(E,R,{})", True, e, r, frozenset(), 500),
    )


def check_running_example() -> tuple[int, int]:
    facts = running_example_facts()
    all_indices = frozenset(range(len(facts)))
    dag_satisfaction: list[tuple[Graph, FrozenSet[int]]] = []
    for edges in labelled_dags(4):
        satisfied = frozenset(
            index
            for index, fact in enumerate(facts)
            if (not reference_d_connected(4, edges, fact.left, fact.right, fact.observed))
            == fact.independent
        )
        dag_satisfaction.append((edges, satisfied))

    def coherent(enforced: FrozenSet[int]) -> bool:
        return any(enforced <= satisfied for _, satisfied in dag_satisfaction)

    correction_masks = [
        released for released in powerset(all_indices) if coherent(all_indices - released)
    ]
    mcses = [
        released
        for released in correction_masks
        if not any(other < released for other in correction_masks)
    ]
    cost = lambda released, delta=None: sum(
        fact.weight + ((delta or {}).get(index, 0))
        for index, fact in enumerate(facts)
        if index in released
    )
    baseline_cost = min(map(cost, correction_masks))
    optima = {released for released in correction_masks if cost(released) == baseline_cost}
    baseline = frozenset({0, 10, 12})
    assert baseline_cost == 2040 and optima == {baseline} and baseline in mcses
    witness_graphs = [
        edges for edges, satisfied in dag_satisfaction if all_indices - baseline <= satisfied
    ]
    assert witness_graphs == [frozenset({(0, 2), (1, 2), (2, 3)})]

    muses = []
    for enforced in powerset(all_indices):
        if coherent(enforced):
            continue
        if all(coherent(enforced - {fact}) for fact in enforced):
            muses.append(enforced)
    expected_private = {
        0: frozenset({0, 4, 9}),
        10: frozenset({8, 10, 13}),
        12: frozenset({1, 8, 12}),
    }
    expected_keep_cost = {0: 2070, 10: 2540, 12: 2910}
    expected_margin = {0: 30, 10: 500, 12: 870}
    for challenged in baseline:
        retained_plus_fact = (all_indices - baseline) | {challenged}
        assert not coherent(retained_plus_fact)
        assert expected_private[challenged] in muses
        assert expected_private[challenged] & baseline == {challenged}

        keeping = [repair for repair in correction_masks if challenged not in repair]
        releasing = [repair for repair in correction_masks if challenged in repair]
        keep_cost = min(map(cost, keeping))
        release_cost = min(map(cost, releasing))
        margin = keep_cost - release_cost
        assert keep_cost == expected_keep_cost[challenged]
        assert release_cost == baseline_cost
        assert margin == expected_margin[challenged]
        hard_optima = {repair for repair in keeping if cost(repair) == keep_cost}
        assert hard_optima and baseline not in hard_optima
        assert all(challenged not in repair for repair in hard_optima)

        tie_delta = {challenged: margin}
        tie_cost = min(cost(other, tie_delta) for other in correction_masks)
        at_tie = {
            repair
            for repair in correction_masks
            if cost(repair, tie_delta) == tie_cost
        }
        assert any(challenged in repair for repair in at_tie)
        assert any(challenged not in repair for repair in at_tie)
        forcing_delta = {challenged: margin + 1}
        forcing_cost = min(cost(other, forcing_delta) for other in correction_masks)
        after_tie = {
            repair
            for repair in correction_masks
            if cost(repair, forcing_delta) == forcing_cost
        }
        assert all(challenged not in repair for repair in after_tie)

    return len(muses), len(mcses)


def check_mixed_truth_challenge() -> tuple[int, int, int, int, int, int]:
    """Audit Example 8, where one correct and two wrong facts are released."""
    facts = running_example_facts()
    all_indices = frozenset(range(len(facts)))
    # The second trace omits only indep(E,R,{}), which need not be queried.
    objectives = all_indices - {13}
    dag_satisfaction: list[tuple[Graph, FrozenSet[int]]] = []
    for edges in labelled_dags(4):
        satisfied = frozenset(
            index
            for index, fact in enumerate(facts)
            if (not reference_d_connected(4, edges, fact.left, fact.right, fact.observed))
            == fact.independent
        )
        dag_satisfaction.append((edges, satisfied))

    def coherent(enforced: FrozenSet[int]) -> bool:
        return any(enforced <= satisfied for _, satisfied in dag_satisfaction)

    correction_masks = [
        released
        for released in powerset(objectives)
        if coherent(objectives - released)
    ]
    mcses = [
        released
        for released in correction_masks
        if not any(other < released for other in correction_masks)
    ]

    def cost(released: FrozenSet[int], delta: Optional[dict[int, int]] = None) -> int:
        increases = delta or {}
        return sum(facts[index].weight + increases.get(index, 0) for index in released)

    challenged = 9  # indep(E,I,{O})
    baseline = frozenset({challenged, 10, 12})
    hard_repair = frozenset({0, 10, 12})
    ground_truth = frozenset({(0, 2), (1, 2), (2, 3)})
    correct = frozenset(
        index
        for index, fact in enumerate(facts)
        if (not reference_d_connected(
            4, ground_truth, fact.left, fact.right, fact.observed
        ))
        == fact.independent
    )
    assert baseline & correct == {challenged}
    assert baseline - correct == {10, 12}

    baseline_cost = min(map(cost, correction_masks))
    optima = {
        released for released in correction_masks if cost(released) == baseline_cost
    }
    assert baseline in mcses
    assert baseline_cost == 1570 and optima == {baseline}
    baseline_graphs = {
        edges
        for edges, satisfied in dag_satisfaction
        if objectives - baseline <= satisfied
    }
    assert len(baseline_graphs) == 4
    assert all(
        challenged not in satisfied
        for edges, satisfied in dag_satisfaction
        if edges in baseline_graphs
    )

    private_mus = frozenset({0, 4, challenged})
    assert not coherent(private_mus)
    assert all(coherent(private_mus - {fact}) for fact in private_mus)
    assert private_mus & baseline == {challenged}

    keeping = [repair for repair in correction_masks if challenged not in repair]
    releasing = [repair for repair in correction_masks if challenged in repair]
    keep_cost = min(map(cost, keeping))
    release_cost = min(map(cost, releasing))
    margin = keep_cost - release_cost
    hard_optima = {repair for repair in keeping if cost(repair) == keep_cost}
    assert release_cost == baseline_cost == 1570
    assert keep_cost == 2040 and margin == 470
    assert hard_optima == {hard_repair}
    hard_graphs = {
        edges
        for edges, satisfied in dag_satisfaction
        if objectives - hard_repair <= satisfied
    }
    assert len(hard_graphs) == 13
    assert ground_truth in hard_graphs
    assert baseline_graphs.isdisjoint(hard_graphs)

    at_tie_cost = min(
        cost(repair, {challenged: margin}) for repair in correction_masks
    )
    at_tie = {
        repair
        for repair in correction_masks
        if cost(repair, {challenged: margin}) == at_tie_cost
    }
    assert at_tie == {baseline, hard_repair}

    after_tie_cost = min(
        cost(repair, {challenged: margin + 1}) for repair in correction_masks
    )
    after_tie = {
        repair
        for repair in correction_masks
        if cost(repair, {challenged: margin + 1}) == after_tie_cost
    }
    assert after_tie == {hard_repair}
    return (
        baseline_cost,
        keep_cost,
        margin,
        len(mcses),
        len(baseline_graphs),
        len(hard_graphs),
    )


def main() -> None:
    dags, queries = check_bayes_ball()
    stable_count, toy_mcs_count = check_projection_theorem()
    mus_count, example_mcs_count = check_running_example()
    (
        mixed_cost,
        mixed_keep_cost,
        mixed_margin,
        mixed_mcs_count,
        mixed_graph_count,
        mixed_keep_graph_count,
    ) = check_mixed_truth_challenge()
    print(
        "Bayes-ball/Shachter simulation: "
        f"{dags} labelled DAGs, {queries} valid ordered queries: PASS"
    )
    print(
        "Projection/equal-weight/multiple-model toy checks: "
        f"{stable_count} stable models, {toy_mcs_count} MCSes: PASS"
    )
    print(
        "Four-node challenge audit: "
        f"{mus_count} MUSes, {example_mcs_count} MCSes, 3 released facts: PASS"
    )
    print(
        "Mixed-truth challenge audit: "
        f"{mixed_mcs_count} MCSes, optimum={mixed_cost}, "
        f"hard-retention optimum={mixed_keep_cost}, margin={mixed_margin}, "
        f"graphs={mixed_graph_count}->{mixed_keep_graph_count}, "
        "disjoint graph sets, ground truth restored, private MUS size=3: PASS"
    )


if __name__ == "__main__":
    main()
