from typing import Set, FrozenSet
from itertools import combinations
import src.causal_aba.atoms as atoms
import src.causal_aba.assumptions as assums
from aspforaba.src.aspforaba import ABASolver
from src.utils.utils import unique_product, powerset


class CoreABASPSolverFactory:
    """
    Factory class for creating ABASP solvers with core rules and assumptions.


    It contains all graph edge assumptions, colliders and corresponding rules.
    NOTE: It does not contain active/blocked path rules and independence assumptions.

    NOTE: Variables corresponding to graph nodes are capital letters everywhere (e.g., X, Y, Z).
    """

    def __init__(self,
                 n_nodes: int,
                 abaf_class=ABASolver):
        self.abaf_class = abaf_class
        self.n_nodes = n_nodes

    @staticmethod
    def _add_graph_edge_assumptions(solver, X, Y):
        for assumption in [assums.arr(X, Y), assums.arr(Y, X), assums.noe(X, Y)]:
            solver.add_assumption(assumption)
            solver.add_contrary(assumption, assums.contrary(assumption))

        for assumption1, assumption2 in unique_product([assums.arr(X, Y),
                                                        assums.arr(Y, X),
                                                        assums.noe(X, Y)], repeat=2):
            solver.add_rule(assums.contrary(assumption2), [assumption1])

        solver.add_rule(atoms.dpath(X, Y), [assums.arr(X, Y)])
        solver.add_rule(atoms.dpath(Y, X), [assums.arr(Y, X)])

        solver.add_rule(atoms.edge(X, Y), [assums.arr(X, Y)])
        solver.add_rule(atoms.edge(X, Y), [assums.arr(Y, X)])

    @staticmethod
    def _add_acyclicity_rules(solver, X, Y):
        solver.add_rule(assums.contrary(assums.arr(Y, X)), [atoms.dpath(X, Y)])

    @staticmethod
    def _add_non_blocking_rules(solver, X, Y, S, n_nodes):
        for N in S:
            if N not in {X, Y}:  # unique X, Y, N
                # 1) N doesn't block the S-active path between its neighbours X and Y
                #    if N is a collider and belongs to the set S
                solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.collider(X, N, Y)])

        for N in set(range(n_nodes)) - set(S):  # nodes not in S
            if N not in {X, Y}:  # unique X, Y, N
                # 2) N doesn't block the S-active path between its neighbours X and Y
                #    if N is not a collider and doesn't belong to the set S
                solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.not_collider(X, N, Y)])

                # 3) N doesn't block the S-active path between its neighbours X and Y
                #    if N doesn't belong to the set S and has descendant that belongs to S
                for Z in S:
                    if Z not in {X, Y, N}:
                        solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.collider(
                            X, N, Y), atoms.descendant_of_collider(Z, X, N, Y)])

    @staticmethod
    def _add_direct_path_definition_rules(solver, X, Y, Z):
        solver.add_rule(atoms.dpath(X, Y), [assums.arr(X, Z), atoms.dpath(Z, Y)])

    @staticmethod
    def _add_collider_definition_rules(solver, X, Y, Z):
        # collider on middle node: X->Y<-Z
        solver.add_rule(atoms.collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Z, Y)])

        # all not collider cases
        # X->Y->Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Y, Z)])
        # X<-Y->Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Y, X), assums.arr(Y, Z)])
        # X<-Y<-Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Z, Y), assums.arr(Y, X)])

    @staticmethod
    def _add_collider_descendant_definition_rules(solver, X, Y, Z, N):
        solver.add_rule(atoms.descendant_of_collider(N, X, Y, Z), [atoms.collider(X, Y, Z), atoms.dpath(Y, N)])

    def create_core_solver(self, edges_to_remove: Set[FrozenSet[int]]):
        """
        Create a core solver for the ABASP problem.
        It contains all graph edge assumptions, colliders and corresponding rules.
        NOTE: It does not contain active/blocked path rules and independence assumptions.

        """
        # do not consider arr assumptions for paths that have edges with nodes that are independent (for any set S)

        solver = self.abaf_class()

        for X, Y in unique_product(range(self.n_nodes), repeat=2):
            if frozenset({X, Y}) in edges_to_remove:
                continue

            if X < Y:  # for X, Y unique combinations
                self._add_graph_edge_assumptions(solver, X, Y)

                
            self._add_acyclicity_rules(solver, X, Y)

        for X, Y, Z in unique_product(range(self.n_nodes), repeat=3):

            if frozenset({X, Z}) not in edges_to_remove:
                self._add_direct_path_definition_rules(solver, X, Y, Z)

            if frozenset({Y, X}) not in edges_to_remove and frozenset({Y, Z}) not in edges_to_remove:
                if X < Z:
                    # X < Z is to avoid duplicates as colliders are symmetric
                    self._add_collider_definition_rules(solver, X, Y, Z)

                    for N in range(self.n_nodes):
                        if N not in {X, Y, Z}:  # X, Y, Z, N unique
                            self._add_collider_descendant_definition_rules(solver, X, Y, Z, N)
        
        for X, Y in combinations(range(self.n_nodes), 2):
            for S in powerset(range(self.n_nodes)):
                self._add_non_blocking_rules(solver, X, Y, S, self.n_nodes)


        return solver

    def create_solver(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
