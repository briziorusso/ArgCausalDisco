import pandas as pd
from pydantic import BaseModel

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

from priors.llm import extract


class Constraints(BaseModel):
    forbidden: list[tuple[str, str]]


class PriorKnowledge:
    def __init__(self, variables: list[str], constraints: Constraints) -> None:
        self.constraints = constraints
        self.var_id = {var: i for i, var in enumerate(sorted(variables))}
        self.forbidden = {
            (self.var_id[edge[0]], self.var_id[edge[1]])
            for edge in constraints.forbidden
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


def generate_priors(dataset_path: str, show_prompt: bool = False) -> PriorKnowledge:
    """
    Generate prior knowledge from the dataset.

    dataset_path (str): Path to the dataset CSV file.
    """
    variables = pd.read_csv(dataset_path).to_dict("records")

    var_desc = ""
    sorted_vars = []
    for i, var in enumerate(sorted(variables, key=lambda x: x["symbol"])):
        var_desc += f"{i}. {var['symbol']}: {var['name']}, {var['description']}\n"
        sorted_vars.append(var["symbol"])

    prompt = (
        "Your goal is to find out causal constraints that you're very confident"
        "based on expert knowledge, among the given variables:\n"
        f"{var_desc}"
    )
    if show_prompt:
        print(prompt)

    constraint = extract(prompt, Constraints)
    return PriorKnowledge(variables=sorted_vars, constraints=constraint)
