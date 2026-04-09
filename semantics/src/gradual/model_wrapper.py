from enum import Enum
import numpy as np
from typing import Union, Dict
from GradualABA.semantics.modular import (
    ProductAggregation,
    SumAggregation,
    QuadraticMaximumInfluence,
    EulerBasedInfluence,
    LinearInfluence,
    MLPBasedInfluence
)
from GradualABA.semantics.bsafDiscreteModular import DiscreteModular
from GradualABA.semantics.modular.SetMaxAggregation import SetMaxAggregation
from GradualABA.semantics.modular.SetSumAggregation import SetSumAggregation
from GradualABA.semantics.modular.SetMeanAggregation import SetMeanAggregation
from GradualABA.semantics.modular.SetProductAggregation import SetProductAggregation
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation
from GradualABA.ABAF.Assumption import Assumption
from ArgCausalDisco.utils.graph_utils import is_dag
from src.utils.utils import get_matrix_from_arrow_set, get_arrows_from_assumptions

import src.causal_aba.assumptions as asm
from logger import logger


class ModelEnum(str, Enum):
    """
    Enum for different model types.
    """
    QE = "Quadratic Energy"
    DF_QUAD = "DF-QuAD"


class ModelWrapper:
    """
    Given a BSAF, will run Naive Gradual Causal ABA with provided aggregation and influence functions.
    """

    def __init__(self,
                 bsaf,
                 n_nodes: int,
                 model_name: Union[ModelEnum, None] = ModelEnum.DF_QUAD,
                 set_aggregation: Union[SetMaxAggregation,
                                        SetSumAggregation,
                                        SetMeanAggregation,
                                        SetProductAggregation,
                                        SetMinAggregation] = SetProductAggregation,
                 aggregation: Union[ProductAggregation,
                                    SumAggregation,
                                    None] = None,
                 influence: Union[QuadraticMaximumInfluence,
                                  EulerBasedInfluence,
                                  LinearInfluence,
                                  MLPBasedInfluence,
                                  None] = None,
                 conservativeness: float = 1,
                 ):
        """
        Initialize the model wrapper with a BSAF instance.
        
        Args:
            bsaf: An instance of BSAF containing the assumptions and arguments.
            model_name: The name of the model to use. If None, set_aggregation, aggregation, and influence must be provided.
            set_aggregation: An instance of SetMaxAggregation, SetSumAggregation, SetMeanAggregation,
                             SetProductAggregation, or SetMinAggregation.
                             Can be None if model_name is provided.
            aggregation: An instance of ProductAggregation or SumAggregation.
                                Can be None if model_name is provided.
            influence: An instance of QuadraticMaximumInfluence, EulerBasedInfluence, LinearInfluence, or MLPBasedInfluence.
                                Can be None if model_name is provided.
        """
        self.bsaf = bsaf
        self.n_nodes = n_nodes
        self.solved = False

        # dictionary (assumption_name: strength) for arr and noe assumptions
        self.arrow_strengths = None
        self.indep_strengths = None

        if set_aggregation is None:
            raise ValueError("Please provide a set aggregation method.")

        if ((aggregation is None
             or influence is None)
                and model_name is None):
            raise ValueError(
                "Please provide either model_name or all three of (set_aggregation, aggregation, influence).")

        if model_name == ModelEnum.QE:
            self.model = DiscreteModular(
                BSAF=bsaf,
                set_aggregation=set_aggregation,
                aggregation=SumAggregation() if aggregation is None else aggregation,
                influence=QuadraticMaximumInfluence(conservativeness=conservativeness) if influence is None else influence
            )
        elif model_name == ModelEnum.DF_QUAD:
            self.model = DiscreteModular(
                BSAF=bsaf,
                set_aggregation=set_aggregation,
                aggregation=ProductAggregation() if aggregation is None else aggregation,
                influence=LinearInfluence(conservativeness=conservativeness) if influence is None else influence
            )
        elif model_name is None:
            self.model = DiscreteModular(
                BSAF=bsaf,
                set_aggregation=set_aggregation,
                aggregation=aggregation,
                influence=influence
            )
        else:
            raise ValueError(f"Model {model_name} is not supported. "
                             f"Please use one of {list(ModelEnum)}.")

    def solve(self,
              iterations: int = 10,
              verbose: bool = False) -> Dict[Assumption, float]:
        self.solved = True
        self.model.solve(iterations=iterations,
                         generate_plot=True,
                         verbose=verbose)

        graph_data = self.model.graph_data

        self.arrow_strengths = {
            asm_name: strength_evo[-1][1]
            for asm_name, strength_evo in graph_data.items()
            if asm_name.startswith('arr')
        }
        self.indep_strengths = {
            asm_name: strength_evo[-1][1]
            for asm_name, strength_evo in graph_data.items()
            if asm_name.startswith(('indep', 'dep'))
        }

        return graph_data

    @staticmethod
    def get_graph_strength_from_arrow_strengths(graph: np.ndarray, arrow_strengths: dict) -> float:
        """ Score provided graph based on arrow strengths."""
        score = 0.0
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                try:
                    if graph[i, j] > 0:
                        score += arrow_strengths[asm.arr(i, j)]
                    else:  # else if its 0
                        score += arrow_strengths[asm.noe(i, j)]
                except KeyError:
                    logger.warning(f"Arrow assumption {asm.arr(i, j)} not found in arrow strengths.")
                    return None
        return score

    def get_graph_strength(self, graph: np.ndarray) -> float:
        """ Score provided graph based on arrow strengths."""
        if not self.solved:
            raise RuntimeError("Model has not been solved yet. Please call solve() first.")

        if self.arrow_strengths is None:
            raise RuntimeError("Arrow strengths have not been computed. Please call solve() first.")

        return self.get_graph_strength_from_arrow_strengths(graph, self.arrow_strengths)

    def _get_nodes_from_arr_or_noe(self, asm_name: str) -> tuple:
        """
        Extracts the nodes from an assumption name that starts with 'arr_' or 'noe_'.
        
        :param asm_name: The assumption name to parse.
        :return: A tuple of integers representing the nodes in the assumption.
        """
        if not asm_name.startswith(('arr_', 'noe_')):
            raise ValueError(f"Assumption name {asm_name} must start with 'arr_' or 'noe_'.")

        try:
            _, node1, node2 = asm_name.split("_")
            node1 = int(node1)
            node2 = int(node2)
        except ValueError as e:
            raise ValueError(f"Arrow must be of form arr_<node1>_<node2>, got {asm_name}. Here is the exception {e}")

        return node1, node2

    def build_greedy_graph(self) -> np.ndarray:
        """Build a greedy graph based on the arrow strengths."""
        if not self.solved:
            raise RuntimeError("Model has not been solved yet. Please call solve() first.")

        if self.arrow_strengths is None:
            raise RuntimeError("Arrow strengths have not been computed. Please call solve() first.")

        sorted_arrow_strengths = sorted(
            self.arrow_strengths.items(),
            key=lambda item: item[1],
            reverse=True
        )

        accepted_assumptions = set()
        for asm_name, _ in sorted_arrow_strengths:
            if len(accepted_assumptions) == self.n_nodes * (self.n_nodes - 1) // 2:
                # all possible edges are accepted, no need to check further
                break

            node1, node2 = self._get_nodes_from_arr_or_noe(asm_name)

            # enforce mutual exclusivity of arr and noe assumptions
            if asm.arr(node1, node2) in accepted_assumptions or \
               asm.arr(node2, node1) in accepted_assumptions or \
               asm.noe(node1, node2) in accepted_assumptions:
                # is not accepted, moving on to next assumption
                continue

            # enforce acyclicity
            matrix = get_matrix_from_arrow_set(
                get_arrows_from_assumptions(
                    accepted_assumptions),
                self.n_nodes)
            if not is_dag(matrix):
                # is not accepted, moving on to next assumption
                continue
            accepted_assumptions.add(asm_name)

        greedy_matrix = get_matrix_from_arrow_set(
            get_arrows_from_assumptions(
                accepted_assumptions),
            self.n_nodes)
        return greedy_matrix

    def get_arrow_strengths(self) -> Dict[str, float]:
        if not self.solved:
            raise RuntimeError("Model has not been solved yet. Please call solve() first.")

        if self.arrow_strengths is None:
            raise RuntimeError("Arrow strengths have not been computed. Please call solve() first.")
        return self.arrow_strengths

    def get_indep_strengths(self) -> Dict[str, float]:
        """
        Get the strengths of independence assumptions.
        
        Returns:
            A dictionary mapping independence assumption names to their strengths.
        """
        if not self.solved:
            raise RuntimeError("Model has not been solved yet. Please call solve() first.")
        if self.indep_strengths is None:
            raise RuntimeError("Arrow strengths have not been computed. Please call solve() first.")

        return self.indep_strengths
