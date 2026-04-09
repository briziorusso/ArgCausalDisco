# This factory produces ABAF with original ABA rules and nothing more

from typing import List
import networkx as nx
from itertools import combinations
from tqdm import tqdm

import src.causal_aba.assumptions as assums
from src.causal_aba.factory import ABASPSolverFactory
from src.gradual.abaf_builder import ABAFBuilder
from src.utils.enums import Fact, RelationEnum
from src.utils.utils import powerset


class FactoryV0(ABASPSolverFactory):
    """
    Factory class for creating ABAF with original ABA rules and nothing more.
    This factory is used to create a basic ABAF without any additional assumptions or rules.
    """

    def __init__(self, n_nodes: int):
        super().__init__(n_nodes, abaf_class=ABAFBuilder)

    @staticmethod
    def _add_fact(solver, fact: Fact):
        raise NotImplementedError("This factory does not support adding facts.")

    @staticmethod
    def _get_indep_assum_strength(fact: Fact) -> float:
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

    @staticmethod
    def _add_fact_strengths(solver: ABAFBuilder, facts: List[Fact]):
        """
        Add strength of facts as strengths of independence assumptions corresponding to them.
        """
        for fact in facts:
            # Update the weight of the independence assumption
            assumption_name = assums.indep(fact.node1, fact.node2, fact.node_set)
            new_weight = FactoryV0._get_indep_assum_strength(fact)
            solver.update_assumption_weight(assumption_name, new_weight)

    def create_solver(self, facts: List[Fact], debug_mode: bool = False) -> ABAFBuilder:
        '''
        Create ABASP solver with active/blocked path rules, 
        independence assumptions and facts.

        NOTE 1: add ap assumption and rule only for all possible dep and indep present in facts
        NOTE 2: consider all paths
        '''
        graph = nx.complete_graph(self.n_nodes)
        solver = self.create_core_solver({})
        all_paths = dict()

        for X, Y in tqdm(combinations(range(self.n_nodes), 2),
                         total=self.n_nodes*(self.n_nodes-1) // 2,
                         desc="iterating through node combinations"):
            if X > Y:
                X, Y = Y, X   # ensure X < Y for consistency
            paths = [tuple(p) for p in nx.all_simple_paths(graph, source=X, target=Y)]
            all_paths[(X, Y)] = paths

            self._add_path_definition_rules(solver, paths, X, Y)

            for S in powerset(set(range(self.n_nodes)) - {X, Y}):
                self._add_indep_assumptions(solver, X, Y, S)

                for path_id, my_path in enumerate(paths):
                    # active path definition
                    self._add_blocked_path_assumptions(solver, path_id, X, Y, S)
                    self._add_active_path_rules(solver, path_id, my_path, X, Y, S)
                    self._add_dependence_rules(solver, path_id, X, Y, S)

                self._add_independence_rules(solver, paths, X, Y, S)
        
        self._add_fact_strengths(solver, facts)

        if debug_mode:
            return solver, all_paths

        return solver
