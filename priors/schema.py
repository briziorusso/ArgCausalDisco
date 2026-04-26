from collections.abc import Iterable

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from pydantic import BaseModel


class Constraints(BaseModel):
    forbidden: set[tuple[str, str]] = set()
    required: set[tuple[str, str]] = set()


def build_mpc_background_knowledge(
    variables: Iterable[str],
    constraints: Constraints,
) -> tuple[BackgroundKnowledge, list[str]]:
    sorted_variables = sorted(variables)
    variable_set = set(sorted_variables)
    if len(sorted_variables) != len(variable_set):
        raise ValueError("MPC background knowledge variables must be unique.")

    prior_variables = {
        node
        for edge in constraints.forbidden | constraints.required
        for node in edge
    }
    missing = sorted(prior_variables - variable_set)
    if missing:
        raise ValueError(
            "MPC background knowledge contains variables not present in the dataset: "
            + ", ".join(missing)
        )

    nodes = {variable: GraphNode(variable) for variable in sorted_variables}
    bk = BackgroundKnowledge()
    for source, target in sorted(constraints.forbidden):
        bk.add_forbidden_by_node(nodes[source], nodes[target])
    for source, target in sorted(constraints.required):
        bk.add_required_by_node(nodes[source], nodes[target])

    return bk, sorted_variables


class PriorKnowledge:
    def __init__(self, variables: list[str], constraints: Constraints) -> None:
        self.constraints = constraints
        self.var_id = {var: i for i, var in enumerate(sorted(variables))}
        self.forbidden = {
            (self.var_id[edge[0]], self.var_id[edge[1]])
            for edge in constraints.forbidden
        }
        self.required = {
            (self.var_id[edge[0]], self.var_id[edge[1]])
            for edge in constraints.required
        }
        self.background_knowledge = self.get_background_knowledge()

    def get_background_knowledge(self) -> BackgroundKnowledge:
        """
        Convert constraints from the Constraints model to BackgroundKnowledge object.

        Args:
            constraints: Constraints object containing forbidden and required edges and tiers

        Returns:
            BackgroundKnowledge object with the specified constraints
        """
        # Initialize BackgroundKnowledge object
        bk = BackgroundKnowledge()

        # Add forbidden edges (edges that cannot exist)
        for source, target in self.forbidden:
            bk.add_forbidden_by_node(GraphNode(source), GraphNode(target))

        return bk
