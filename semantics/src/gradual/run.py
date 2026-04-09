from dataclasses import dataclass
from typing import List

from src.gradual.model_wrapper import ModelWrapper, ModelEnum
from src.utils.enums import Fact
from src.causal_aba.core_factory import CoreABASPSolverFactory

from GradualABA.semantics.modular.SetProductAggregation import SetProductAggregation
from GradualABA.BSAF import BSAF
from src.gradual.abaf_opt import ABAFOptimised
from src.utils.enums import RelationEnum
from GradualABA.ABAF import ABAF, Assumption
from typing import Union, Dict
from GradualABA.constants import DEFAULT_WEIGHT
import src.causal_aba.assumptions as assums
from src.gradual.extra.abaf_factory_v0 import FactoryV0

from logger import logger


@dataclass
class GradualCausalABAOutput:
    arrow_strengths: dict
    indep_strengths: dict
    graph_data: dict
    has_converged_map: dict


def run_get_bsaf_and_assum_dict(factory: CoreABASPSolverFactory,
                                facts: List[Fact],
                                set_aggregation=SetProductAggregation(),
                                abaf_class: Union[ABAFOptimised, ABAF] = ABAFOptimised) -> BSAF:
    """    Run the factory to create a BSAF with assums strengths based on the given facts.
    """
    abaf_builder = factory.create_solver(facts=facts)
    abaf = abaf_builder.get_abaf(abaf_class=abaf_class)
    bsaf = abaf.to_bsaf(weight_agg=set_aggregation)

    return bsaf, abaf_builder.name_to_assumption


def run_get_bsaf(factory: CoreABASPSolverFactory,
                 facts: List[Fact],
                 set_aggregation=SetProductAggregation(),
                 abaf_class: Union[ABAFOptimised, ABAF] = ABAFOptimised) -> BSAF:
    """    Run the factory to create a BSAF with assums strengths based on the given facts.
    """
    bsaf, _ = run_get_bsaf_and_assum_dict(factory, facts, set_aggregation, abaf_class)
    return bsaf


def run_model(n_nodes: int,
              bsaf: BSAF,
              model_name: ModelEnum = ModelEnum.DF_QUAD,
              set_aggregation=SetProductAggregation(),
              aggregation=None,
              influence=None,
              conservativeness: float = 1.0,
              iterations: int = 10,
              conv_epsilon: float = 1e-3,
              conv_last_n: int = 5,) -> GradualCausalABAOutput:
    """
    Run the gradual causal ABA solver with the given parameters.

    :param X_s: Input data for the model.
    :param factory: Factory to create the ABAF solver.
    :param model_name: Name of the model to use.
    :param set_aggregation: Set aggregation method to use.
    :param aggregation: Aggregation method to use.
    :param influence: Influence method to use.
    :param conservativeness: Conservativeness factor for the model.

    :return: GradualCausalABAOutput containing arrow strengths, indep strengths.
    """
    logger.info("solving BSAF with GradualCausalABA")
    model_wrapper = ModelWrapper(
        bsaf=bsaf,
        n_nodes=n_nodes,
        model_name=model_name,
        set_aggregation=set_aggregation,
        aggregation=aggregation,
        influence=influence,
        conservativeness=conservativeness)
    graph_data = model_wrapper.solve(iterations=iterations,
                                     verbose=True)

    return GradualCausalABAOutput(
        arrow_strengths=model_wrapper.get_arrow_strengths(),
        indep_strengths=model_wrapper.get_indep_strengths(),
        graph_data=graph_data,
        has_converged_map=model_wrapper.model.has_converged(epsilon=conv_epsilon,
                                                            last_n=conv_last_n)
    )


def reset_weights(assum_dict: Dict[str, Assumption], default_weight=DEFAULT_WEIGHT):
    """
    Reset the weights of the assumptions in BSAF to their initial values.
    
    :param assum_dict: Dictionary mapping assumption names to their instances.
    """
    for _, assum in assum_dict.items():
        assum.update_weight(default_weight)


def get_indep_assum_strength(fact: Fact) -> float:
        """
        Get the strength of the assumption based on the fact.
         Strengths of independence assumptions are assigned as follows:
         0.5 if nothing is known about the independence assumption
         0.5 + 0.5 * fact_score if we have an independence fact
         1 - (0.5 + 0.5 * fact_score) if we have a dependency fact
        """
        if fact.relation == RelationEnum.indep:
            return 0.5 + 0.5 * fact.score
        elif fact.relation == RelationEnum.dep:
            return 1 - (0.5 + 0.5 * fact.score)
        else:
            raise ValueError(f"Unknown relation type: {fact.relation}")


def set_weights_according_to_facts(assum_dict: Dict[str, Assumption], facts: List[Fact]):
    """
    Set the weights of the assumptions in BSAF according to the provided facts.
    
    :param bsaf: The BSAF instance whose weights are to be updated.
    :param assum_dict: Dictionary mapping assumption names to their instances.
    :param facts: List of facts containing the new weights for the assumptions.
    """
    for fact in facts:
        assumption_name = assums.indep(fact.node1, fact.node2, fact.node_set)
        new_weight = get_indep_assum_strength(fact)
        if assumption_name in assum_dict:
            assum_dict[assumption_name].update_weight(new_weight)


if __name__ == "__main__":
    # Example usage
    factory = FactoryV0(n_nodes=5)
    facts = [Fact(node1=1,
                  node2=2,
                  relation=Fact.RelationEnum.dep,
                  score=0.8,
                  S={3, 4}),
             Fact(node1=2,
                  node2=3,
                  relation=Fact.RelationEnum.indep,
                  score=0.5,
                  S={4})]
    bsaf = run_get_bsaf(factory, facts)

    output = run_model(n_nodes=5, bsaf=bsaf)
