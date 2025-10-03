from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from pydantic import BaseModel


class Constraints(BaseModel):
    forbidden: set[tuple[str, str]] = set()
    required: set[tuple[str, str]] = set()


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
