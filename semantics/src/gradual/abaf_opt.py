from GradualABA.ABAF import ABAF, Sentence, Assumption
from GradualABA.BSAF import Argument
from itertools import product
from typing import Set, FrozenSet
from tqdm import tqdm
from logger import logger


class ABAFOptimised(ABAF):
    def _get_derivations(self,
                         sent: Sentence,
                         deriv_table: dict,
                         visited: set = set()):
        """
        Recursive function to get all derivations of a sentence
        from the rules in the ABAF.
        
        Returns: set of frozen sets of assumptions"""
        if sent in deriv_table:
            # base case of the recursion
            return deriv_table[sent]
        else:
            # recursive case
            all_derivations = set()
            move_on_to_next_rule = False
            for rule in self.rules:
                if rule.head != sent:
                    continue  # irrelevant rule, skip to next
                derivations = dict()
                added_assumptions = set()
                for body_element in rule.body:
                    if body_element in visited:
                        # if we have already visited this element,
                        # so we reached a cycle
                        # we stop the route and do not continue
                        move_on_to_next_rule = True
                        break
                    elif isinstance(body_element, Assumption):
                        # if is an assumption, add to current premises
                        added_assumptions.add(body_element)
                    else:
                        # if is a sentence, recursively get derivations
                        derivations[body_element] = self._get_derivations(
                            body_element,
                            deriv_table,
                            visited=visited.union({body_element, sent}))
                        if len(derivations[body_element]) == 0:
                            # if no derivations for given literal, continue to next rule
                            move_on_to_next_rule = True
                            break
                if move_on_to_next_rule:
                    move_on_to_next_rule = False
                    continue
                # mix and match all derivations of literals
                if len(derivations) == 0:
                    # if no derivations, then we have a direct derivation from assumptions
                    all_derivations.add(frozenset(added_assumptions))
                else:
                    for possible_derivations in product(*derivations.values()):
                        deriv_candidate = set().union(added_assumptions, *possible_derivations)
                        all_derivations.add(frozenset(deriv_candidate))
            # store the result in the derivation table
            deriv_table[sent] = all_derivations
            return all_derivations

    def _prune_supersets(self, my_set: Set[FrozenSet]) -> Set[FrozenSet]:
        """
        Prune the nested set by removing supersets.

        INFO:
        In the context of derivations this ensures that we only keep the minimal sets of assumptions
        that can derive a given sentence.
        """
        pruned_set = set()  # create a copy to avoid modifying the set while iterating
        for item in my_set:
            remove_set = set()
            add_item = True
            for other_item in pruned_set:
                if item.issuperset(other_item):
                    # if item is a superset of other_item, we do not add it
                    add_item = False
                    break
                elif item.issubset(other_item):
                    # if item is a subset of other_item, we remove other_item
                    remove_set.add(other_item)
            pruned_set.difference_update(remove_set)
            if add_item:
                pruned_set.add(item)
        return pruned_set

    def _create_argument(self, sent: Sentence, deriv: FrozenSet[Assumption], weight_agg):
        """
        Create an argument from a sentence and a set of assumptions.

        Args:
            sent (Sentence): The sentence that is being argued for.
            deriv (FrozenSet[Assumption]): The set of assumptions that support the sentence.
            weight_agg (function): A function to aggregate weights of assumptions.
        Returns:
            Argument: An argument object containing the sentence, assumptions, and aggregated weight.
        """
        state = {asm.name: asm.initial_weight for asm in deriv}
        prem_names = {asm.name for asm in deriv}
        init_w = weight_agg.aggregate_set(
            state=state,
            set=prem_names
        )

        return Argument(
            claim=sent,
            premise=list(deriv),
            initial_weight=init_w
        )

    def build_arguments_procedure(self, weight_agg, prune_supersets=True):
        """
        Overrides the funtion in the parent class to
        perform a procedure that results in smaller set of arguments.

        This method ensures that the argument set is optimal
        regarding redundancy notions introduced in 
        "Instantiations and Computational Aspects of Non-Flat Assumption-based Argumentation"
        Lehtonen et al. (2024).

        Goal is for each assumption and each assumption contrary,
        get a list of sets of assumptions
        where each set can derive the assumption or the contrary.

        we do derivation starting from assumptions and their contraries
        tracking back untill we reach another assumption, then process is stoped.
        if we reach a literal, then we track back from that literal untill we reach assumptions.
        in routes where cycle is encountered, we stop the route and do not continue.
        This method ensures that there is no "assumption redundancy".

        there is an option to prune supersets which would
        ensure there is no "derivation redundancy"
        """

        deriv_table = dict()
        contraries = set()

        for assumption in tqdm(self.assumptions, desc="Building arguments"):
            # get all derivations for the assumption
            self._get_derivations(assumption, deriv_table, visited={assumption})
            # get all derivations for the contrary of the assumption
            contrary = next(a for a in self.sentences if a.name == assumption.contrary)
            contraries.add(contrary)
            self._get_derivations(contrary, deriv_table, visited={contrary})

        asm_and_contrary_derivs = {
            sent: derivs for sent, derivs in deriv_table.items()
            if sent in self.assumptions or sent in contraries
        }

        if prune_supersets:
            logger.info("Pruning supersets from derivations")
            asm_and_contrary_derivs = {
                sent: self._prune_supersets(derivs)
                for sent, derivs in asm_and_contrary_derivs.items()
            }

        # convert into arguments:
        self.arguments = []
        for sent, derivs in asm_and_contrary_derivs.items():
            for deriv in derivs:
                self.arguments.append(self._create_argument(sent, deriv, weight_agg))

        return self.arguments
