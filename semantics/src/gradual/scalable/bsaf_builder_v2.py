import sys
sys.path.append("GradualABA/")  # Adjust the path as necessary to import modules

from typing import List, Dict, Tuple, Set, FrozenSet, Union
from src.constants import DEFAULT_WEIGHT  # 0.5
import src.causal_aba.assumptions as asm
from GradualABA.ABAF.Assumption import Assumption, Sentence
from GradualABA.BSAF.Argument import Argument
from GradualABA.ABAF.Rule import Rule
from GradualABA.BSAF.BSAF import BSAF
from itertools import combinations, permutations
from tqdm import tqdm
from src.utils.resource_utils import MemoryUsageExceededException, check_memory_usage_gb
from src.utils.utils import check_arrows_dag
from src.constants import MEMORY_THRESHOLD

from logger import logger


def active_path(path_nodes: Tuple, S: set):
    """Generate a unique name for the active path assumption based on the path nodes and conditioning set.
    The conditioning set is sorted to ensure the name is unique and deterministic.

    Args:
        path_nodes (Tuple): A tuple of node IDs representing the path.
        S (set): A set of node IDs representing the conditioning set.
    """
    S = sorted(list(S))
    return f"active_path_" + '_'.join([str(i) for i in path_nodes]) + "__" + '_'.join([str(i) for i in S])


def is_contrary(sentence: Sentence):
    return sentence.name.startswith('-')


def get_reindex_map(n_nodes, node1: int, node2: int, new_node1: int, new_node2: int) -> Dict[int, int]:
    """Swap node1 and node2 with new_node1 and new_node2 in the given node.
     Provides a deterministic reindexing of nodes in the solution.
     Useful when solution is symetric regarding the nodes"""
    reindex_map = {node1: new_node1, node2: new_node2}
    nodes_left = set(range(n_nodes)) - {node1, node2}

    # reassign remaining nodes to remaining new nodes randomly
    nodes_left_new = set(range(n_nodes)) - {new_node1, new_node2}
    for old_node, new_node in zip(nodes_left, nodes_left_new):
        reindex_map[old_node] = new_node
    return reindex_map


class BSAFBuilderV2:
    def __init__(self,
                 n_nodes: int,
                 include_collider_tree_arguments: bool = True,
                 default_weight: float = DEFAULT_WEIGHT,
                 neighbourhood_n_nodes: Union[int, None] = None,
                 max_cycle_size: int = 5,
                 max_collider_tree_depth: int = 5,
                 max_path_length: int = 5,
                 max_conditioning_set_size: int = 5):
        """
        Initialize the BSAFBuilderV2 with the given parameters.
        Args:
            n_nodes (int): Number of nodes in the BSAF.
            include_collider_tree_arguments (bool): Whether to include collider tree arguments.
            default_weight (float): Default weight for assumptions.
            neighbourhood_n_nodes (int, optional): Number of nodes in the neighbourhood for collider trees.
                This parameter overrides max_cycle_size, max_collider_tree_depth, and max_path_length.
            max_cycle_size (int): Maximum size of cycles to consider.
            max_collider_tree_depth (int): Maximum depth of collider trees.
            max_path_length (int): Maximum length of paths to consider.
            max_conditioning_set_size (int): Maximum size of conditioning sets to consider.
        """
        self.n_nodes = n_nodes
        self.default_weight = default_weight
        self.arguments = set()
        Rule.reset_identifiers()
        Sentence.reset_identifiers()
        Assumption.reset_identifiers()
        self.name_to_assumption: Dict[str, Assumption] = dict()  # Maps assumption names to Assumption objects
        # Maps names to Sentence objects
        # Sentences can be assumptions or assumption contrary literals
        self.name_to_sentence: Dict[str, Sentence] = dict()
        self.contrary_to_assumption: Dict[Sentence, Assumption] = dict()


        if neighbourhood_n_nodes is not None:
            assert isinstance(neighbourhood_n_nodes, int) and neighbourhood_n_nodes >= 3, \
                "neighbourhood_n_nodes must be a positive integer greater or equal to 3."
            self.max_cycle_size = neighbourhood_n_nodes
            self.max_collider_tree_depth = neighbourhood_n_nodes - 3  # longest collider tree can be of size n_nodes-3
            self.max_path_length = neighbourhood_n_nodes - 1  # longest path can be of size n_nodes-1
        else:
            # Longest cycle can be of size n_nodes, so we set reasonable limit
            self.max_cycle_size = min(max_cycle_size, n_nodes)
            # deepest collider tree can be of size n_nodes-3, so we set reasonable limit
            # to see this consider the longest path in the collider tree, which can be of size n_nodes-3
            # as it can't pass through the nodes in the v-structure, because there would be cycle
            self.max_collider_tree_depth = min(max_collider_tree_depth, n_nodes-3)  # Ensure depth does not exceed n_nodes-3
            # longest path can be of size n_nodes-1, so we set reasonable limit
            self.max_path_length = min(max_path_length, n_nodes-1)
            # At most n_nodes-2 conditioning variables

        self.max_conditioning_set_size = min(max_conditioning_set_size, n_nodes-2)
        # If collider tree arguments are not included, we skip their generation, which can speed up the process
        self.include_collider_tree_arguments = include_collider_tree_arguments

    def _get_assumption_from_contrary(self, sentence: Sentence):
        try:
            assumption_name = sentence.name[1:]  # remove the leading -
            assumption = self.name_to_assumption[assumption_name]
            return assumption
        except KeyError as e:
            logger.error(f"Error finding assumption of contrary {sentence}.")
            raise ValueError(f"Error finding assumption of contrary {sentence}.")

    def _add_assumption(self, name: str, initial_weight: float):
        if name not in self.name_to_assumption:
            self.name_to_sentence[name] = Assumption(name=name,
                                                     contrary=asm.contrary(name),
                                                     initial_weight=initial_weight)
            self.name_to_sentence[asm.contrary(name)] = Sentence(name=asm.contrary(name))
            self.name_to_assumption[name] = self.name_to_sentence[name]
        else:
            logger.warning(f"Assumption {name} already exists, skipping creation.")

    def _add_argument(self, claim: Sentence, premise: List[Sentence]):
        try:
            self.arguments.add(Argument(claim=self.name_to_sentence[claim],
                                        premise=[self.name_to_sentence[p] for p in premise]))
        except KeyError as e:
            logger.error(f"Error adding argument: {e}. Claim or premise not found in name_to_sentence.")
            raise ValueError(f"Claim or premise not found in name_to_sentence: {claim}, {premise}")

    def _add_arr_and_mutual_exclusion_arguments(self):
        """Add arrow and no-edge assumptions for all pairs of nodes.
        Also add mutual exclusion arguments for arrows and no-edge assumptions for each pair of nodes.
        """
        for node1, node2 in tqdm(combinations(range(self.n_nodes), 2), 
                                 desc="Adding arrow and no-edge assumptions",
                                 total=self.n_nodes * (self.n_nodes - 1) // 2):
            # Add arrow assumptions
            self._add_assumption(asm.arr(node1, node2), initial_weight=self.default_weight)
            self._add_assumption(asm.arr(node2, node1), initial_weight=self.default_weight)
            # Add no-edge assumption
            self._add_assumption(asm.noe(node1, node2), initial_weight=self.default_weight)

            # Add mutual exclusion arguments for arrows
            assums = [asm.arr(node1, node2), asm.arr(node2, node1), asm.noe(node1, node2)]
            for assum1 in assums:
                for assum2 in assums:
                    if assum1 != assum2:
                        self._add_argument(claim=asm.contrary(assum2), premise=[assum1])

    def _add_cycle_arguments(self):
        """Add cycle arguments for all cycles of size 3 to max_cycle_size.
        Cycles are added as arguments with the assumption that they are mutually exclusive.
        """
        if self.max_cycle_size < 3:
            logger.warning("Max cycle size is less than 3, skipping cycle arguments.")
            return

        # get all unique arrangements of nodes of size cycle-size
        # to get arrangements, first get all combinations, then consdider all permutations
        # each arrangement is considered a cycle, attaks arrow that goes from last node to first node
        for cycle_size in tqdm(range(3, self.max_cycle_size + 1),
                               desc="Adding cycle arguments",
                               total=self.max_cycle_size - 2):
            for perm in permutations(range(self.n_nodes), cycle_size):
                # Generate all permutations corresponding to all possible cycles
                self._add_argument(claim=asm.contrary(asm.arr(perm[-1], perm[0])),
                                    premise=[asm.arr(perm[i], perm[i+1]) for i in range(cycle_size - 1)])

    def _add_indep_assums_and_indep_noe_arguments(self):
        """Add independence assumptions and argument that independence implies no-edge.
        Independence assumptions are added for all pairs of nodes and all conditioning sets 
        up to self.max_conditioning_set_size.
        """
        if self.max_conditioning_set_size < 0:
            logger.error("Max conditioning set size is less than 0, exitting.")
            raise ValueError("Max conditioning set size must be non-negative.")

        for node1, node2 in tqdm(combinations(range(self.n_nodes), 2),
                                 desc="Adding independence assumptions and arguments",
                                 total=self.n_nodes * (self.n_nodes - 1) // 2):
            for conditioning_set_size in range(self.max_conditioning_set_size + 1):
                for conditioning_set in combinations(
                        set(range(self.n_nodes)) - {node1, node2},
                        conditioning_set_size):
                    # Add independence assumption
                    self._add_assumption(asm.indep(node1, node2, conditioning_set),
                                         initial_weight=self.default_weight)
                    # Add argument that independence implies no-edge
                    self._add_argument(claim=asm.noe(node1, node2),
                                       premise=[asm.indep(node1, node2, conditioning_set)])

    def _get_all_paths(self, source: int, target: int, max_path_length: int) -> List[List[int]]:
        """Get all paths between node1 and node2 with length up to given max_path_length.
        Paths are represented as tuples of node IDs.
        """
        paths = []
        intermediate_nodes_all = set(range(self.n_nodes)) - {source, target}
        for num_intermediate_nodes in range(0, max_path_length):
            for intermediate_nodes in permutations(intermediate_nodes_all, num_intermediate_nodes):
                paths.append((source, *intermediate_nodes, target))
        return paths

    def _get_all_descendant_branches(self, central_collider_node: int,
                                     left_collider_node: int,
                                     right_collider_node: int,
                                     conditioning_set: FrozenSet[int]) -> List[Set[Tuple]]:
        if central_collider_node in conditioning_set:
            logger.error("When building collider trees: parent node is in the conditioning set, cannot find descendants.")
            raise ValueError("Parent node cannot be in the conditioning set.")

        if self.max_collider_tree_depth == 0:
            return []

        intermediate_nodes_all = set(range(self.n_nodes)) - {central_collider_node, 
                                                             left_collider_node, 
                                                             right_collider_node, 
                                                             *conditioning_set}
        branches = []
        # if len(intermediate_nodes_all) is 1 then possible max depth is 2
        max_depth = min(self.max_collider_tree_depth, len(intermediate_nodes_all)+1)
        for descendant in conditioning_set:
            for num_intermediate_nodes in range(0, max_depth):
                for intermediate_nodes in permutations(intermediate_nodes_all, num_intermediate_nodes):
                    branch_nodes_sequence = (central_collider_node, *intermediate_nodes, descendant)
                    branch_arrows = frozenset({
                        (branch_nodes_sequence[i], branch_nodes_sequence[i + 1])
                        for i in range(len(branch_nodes_sequence) - 1)
                    })
                    branches.append(branch_arrows)
        return branches

    def _get_path_solutions(self, path_nodes: Tuple, conditioning_set: FrozenSet[int]) -> List[FrozenSet[Tuple[int, int]]]:
        """Generate all possible active collider trees on path_nodes.
        Uses a backtracking approach to find all valid combinations of arrows

        Rough sketch of the algorithm:
        1. if we reached the end of the path, terminate
        2. try adding one of the 2 candidate arrows
            - Is mutual exclusivity satisfied?, no: terminate, yes go to next check
            - If the current node is in the conditioning set
                - if is collider: good, move on to recursive call
                - if not collider: bad, terminate this solution
            - If the current node is not in the conditioning set
                - if not collider: good, move on to recursive call
                - if collider: branch out to all descendants that are in the conditioning set, up to certain distance
                    Each possible branch continues as separate recursive call.
        3. try adding the other candidate arrow.
        4. Finnish.
        """
        solutions = []

        def dfs(previous_arrow_on_path=None, current_node_id=0, arrows_so_far=set()):
            if current_node_id == len(path_nodes) - 1:
                # Base case of recursion: we reached the end of the path, so we can add the solution
                solutions.append((frozenset(arrows_so_far)))  # essentially just copying the arrows_so_far
                return
            else:
                # Recursive case: consider candidate arrows, see if they form a valid partial solution
                for candidate_arrow in [
                    (path_nodes[current_node_id], path_nodes[current_node_id + 1]),
                    (path_nodes[current_node_id + 1], path_nodes[current_node_id])
                ]:
                    reversed_candidate_arrow = (candidate_arrow[1], candidate_arrow[0])
                    if reversed_candidate_arrow in arrows_so_far:  # mutual exclusivity check
                        continue
                    elif previous_arrow_on_path is None:  # equivalent to current_node_id == 0:
                        # case 0: no previous arrow, just add the candidate arrow and continue solving
                        diff_set = {candidate_arrow} - arrows_so_far  # add candidate to the set if not already there
                        arrows_so_far.update(diff_set)
                        dfs(previous_arrow_on_path=candidate_arrow,
                            current_node_id=current_node_id + 1,
                            arrows_so_far=arrows_so_far)
                        arrows_so_far.difference_update(diff_set)  # backtrack
                    elif path_nodes[current_node_id] in conditioning_set:
                        # case 1: current node is in the conditioning set
                        if (previous_arrow_on_path[1] == path_nodes[current_node_id]
                                and candidate_arrow[1] == path_nodes[current_node_id]):
                            # collider case, good, add candidate arrow and continue solving
                            diff_set = {candidate_arrow} - arrows_so_far  # add candidate to the set if not already there
                            arrows_so_far.update(diff_set)
                            dfs(previous_arrow_on_path=candidate_arrow,
                                current_node_id=current_node_id + 1,
                                arrows_so_far=arrows_so_far)
                            arrows_so_far.difference_update(diff_set)  # backtrack
                        else:  # bad, terminate this solution, path is blocked
                            continue
                    else:
                        # case 2: current node is not in the conditioning set
                        if (previous_arrow_on_path[1] == path_nodes[current_node_id]
                                and candidate_arrow[1] == path_nodes[current_node_id]):  # collider case
                            # check out all possible descendants and paths to them that can help avoid blocking

                            branches = self._get_all_descendant_branches(central_collider_node=path_nodes[current_node_id], 
                                                                         left_collider_node=path_nodes[current_node_id - 1], 
                                                                         right_collider_node=path_nodes[current_node_id + 1],
                                                                         conditioning_set=conditioning_set)
                            for branch in branches:  # check mutual exclusivity
                                if any((arr[1], arr[0]) in arrows_so_far for arr in branch):
                                    continue
                                diff_set = {candidate_arrow, *branch} - arrows_so_far  # add branch arrows to the set
                                arrows_so_far.update(diff_set)
                                if check_arrows_dag(arrows_so_far):
                                    # if the arrows so far are acyclic, continue solving with the branch arrows added
                                    dfs(previous_arrow_on_path=candidate_arrow,  # continue solving with the branch arrows added
                                        current_node_id=current_node_id + 1,
                                        arrows_so_far=arrows_so_far)
                                arrows_so_far.difference_update(diff_set)  # backtrack, remove the branch arrows
                        else:  # not collider case, good, add and move on
                            diff_set = {candidate_arrow} - arrows_so_far  # add candidate to the set if not already there
                            arrows_so_far.update(diff_set)
                            dfs(previous_arrow_on_path=candidate_arrow,
                                current_node_id=current_node_id + 1,
                                arrows_so_far=arrows_so_far)
                            arrows_so_far.difference_update(diff_set)  # backtrack
        dfs()
        return solutions

    def _get_generic_solution(self, node1=0, node2=1) -> Dict[Tuple[int], Dict[FrozenSet[int], List[FrozenSet[Tuple[int, int]]]]]:
        """Add active path assumptions and arguments.
        Active path assumptions are added for all pairs of nodes and all paths with length up to self.max_path.
        Z-activr collider tree arrows that support active paths are considered up to self.max_collider_tree_depth.
        Active path conditioning node sets are considered up to self.max_conditioning_set_size.
        
        Using the symmetry of nodes with each other.
        We can find one solutions for given pair of nodes and then use symmetry to find solutions for other nodes.
        Using symmetry would be basically swapping places in the paths, conditioning sets and active collider trees.

        Returns:
            A dictionary: {path_tuple: {conditioning_set: List of active collider trees}}
            Each active collider tree is represented as a set of arrows.
            Each arrow is a tuple of node IDs (from, to).
        """
        if self.max_path_length < 2:
            logger.error("Max path length is less than 2, cannot add active path assumptions.")
            raise ValueError("Max path length must be at least 2.")

        # get generic solution for node1 = 0 and node2 = 1
        all_solutions = dict()

        paths = self._get_all_paths(node1, node2, self.max_path_length)
        conditioning_nodes_all = set(range(self.n_nodes)) - {node1, node2}
        for path in tqdm(paths,
                         desc=f"Finding active paths for dummy nodes {node1} and {node2}",
                         total=len(paths)):
            if check_memory_usage_gb() > MEMORY_THRESHOLD:
                logger.error("Memory usage exceeded threshold, stopping path generation.")
                raise MemoryUsageExceededException("Memory usage exceeded threshold, stopping path generation.")
            for conditioning_set_size in range(self.max_conditioning_set_size + 1):
                for conditioning_set in combinations(conditioning_nodes_all, conditioning_set_size):
                    conditioning_set = frozenset(conditioning_set)  # make it immutable for use in assumptions
                    # Get all active collider trees for the path
                    solutions = self._get_path_solutions(path, conditioning_set)
                    all_solutions[path] = all_solutions.get(path, dict())
                    all_solutions[path][conditioning_set] = solutions
        return all_solutions

    def _swap_generic_solution(self, solution: Dict[Tuple[int], Dict[FrozenSet[int], List[FrozenSet[Tuple[int, int]]]]],
                               node1: int, node2: int,
                               new_node1: int, new_node2) -> Set[Tuple[int, int]]:
        if solution is None:
            logger.error("Solution is None, cannot swap generic solution.")
            raise ValueError("Solution is None, cannot swap generic solution.")
        reindex_map = get_reindex_map(self.n_nodes, node1, node2, new_node1, new_node2)
        swapped_solution = dict()
        for path, conditioning_set_dict in solution.items():
            new_path = tuple(reindex_map[node] for node in path)
            swapped_solution[new_path] = dict()
            for conditioning_set, active_collider_trees in conditioning_set_dict.items():
                new_conditioning_set = frozenset({reindex_map[node] for node in conditioning_set})
                swapped_solution[new_path][new_conditioning_set] = [
                    frozenset({
                        (reindex_map[arr[0]], reindex_map[arr[1]])
                        for arr in active_collider_tree
                    })
                    for active_collider_tree in active_collider_trees
                ]

        return swapped_solution

    def _add_active_path_assums_and_arguments(self):
        """Add active path assumptions and arguments.
        - active path attacks no-edge and is attacked back
        - active path attacks independence assumption and is attacked back
        - active path supports collider structure around nodes in conditioning set
        - active path attacks and is attacked by any structure (other than collider) at nodes in conditioning set.
        - active path is supported by collider structure.
        """
        generic_node1, generic_node2 = 0, 1
        generic_solution = self._get_generic_solution(generic_node1, generic_node2)
        for node1, node2 in tqdm(combinations(range(self.n_nodes), 2),
                                 desc="Adding active path assumptions and arguments",
                                 total=self.n_nodes * (self.n_nodes - 1) // 2):
            if check_memory_usage_gb() > MEMORY_THRESHOLD:
                logger.error("Memory usage exceeded threshold, stopping active path generation.")
                raise MemoryUsageExceededException("Memory usage exceeded threshold, stopping active path generation.")
            if node1 > node2:  # Swap nodes to ensure node1 < node2 for symmetry
                node1, node2 = node2, node1

            if node1 == generic_node1 and node2 == generic_node2:
                solution = generic_solution
            else:  # Swap the generic solution to match the current node pair
                solution = self._swap_generic_solution(generic_solution, generic_node1, generic_node2, node1, node2)

            for path_id, (path, conditioning_set_dict) in enumerate(solution.items()):
                for conditioning_set, active_collider_trees in conditioning_set_dict.items():
                    # Add active path assumption
                    path_assumption_name = active_path(path, conditioning_set)
                    self._add_assumption(path_assumption_name, initial_weight=self.default_weight)

                    # Add argument that active path attacks no-edge assumption and vice versa
                    self._add_argument(claim=asm.contrary(asm.noe(node1, node2)), premise=[path_assumption_name])
                    self._add_argument(claim=asm.contrary(path_assumption_name), premise=[asm.noe(node1, node2)])

                    # Add argument that active path attacks independence assumption and vice versa
                    indep_assumption_name = asm.indep(node1, node2, conditioning_set)
                    self._add_argument(claim=asm.contrary(indep_assumption_name), premise=[path_assumption_name])
                    self._add_argument(claim=asm.contrary(path_assumption_name), premise=[indep_assumption_name])

                    # supporting collider structure for nodes in conditioning set,  mutually attacking anything else
                    for pos in range(1, len(path) - 1):
                        if path[pos] in conditioning_set:
                            # Add argument that active path supports collider structure
                            self._add_argument(claim=asm.arr(path[pos-1], path[pos]), premise=[path_assumption_name])
                            self._add_argument(claim=asm.arr(path[pos+1], path[pos]), premise=[path_assumption_name])
                            # attack and get attacked by anything else
                            self._add_argument(claim=asm.contrary(asm.arr(path[pos], path[pos-1])),
                                               premise=[path_assumption_name])
                            self._add_argument(claim=asm.contrary(asm.arr(path[pos], path[pos+1])),
                                               premise=[path_assumption_name])
                            self._add_argument(claim=asm.contrary(path_assumption_name),
                                               premise=[asm.arr(path[pos], path[pos+1])])
                            self._add_argument(claim=asm.contrary(path_assumption_name),
                                               premise=[asm.arr(path[pos], path[pos-1])])

                    # active collider trees supporting the active path
                    for active_collider_tree in active_collider_trees:
                        # Add argument that active path is supported by collider structure
                        self._add_argument(claim=path_assumption_name,
                                           premise=[asm.arr(arr[0], arr[1]) for arr in active_collider_tree])

    def _add_active_path_assums_and_arguments_wo_collider_trees(self):
        """Add active path assumptions and arguments without collider trees.
        This is a simplified version that does not consider collider trees, which can speed up the process.
        """
        for node1, node2 in tqdm(combinations(range(self.n_nodes), 2),
                                 desc="Adding active path assumptions and arguments without collider trees",
                                 total=self.n_nodes * (self.n_nodes - 1) // 2):
            if check_memory_usage_gb() > MEMORY_THRESHOLD:
                logger.error("Memory usage exceeded threshold, stopping active path generation.")
                raise MemoryUsageExceededException("Memory usage exceeded threshold, stopping active path generation.")

            if node1 > node2:
                node1, node2 = node2, node1
            
            paths = self._get_all_paths(node1, node2, self.max_path_length)
            conditioning_nodes_all = set(range(self.n_nodes)) - {node1, node2}
            for path in paths:
                for conditioning_set_size in range(self.max_conditioning_set_size + 1):
                    for conditioning_set in combinations(conditioning_nodes_all, conditioning_set_size):
                        # TODO: code repetition with self._add_active_path_assums_and_arguments(), refactor
                        conditioning_set = frozenset(conditioning_set)
                        path_assumption_name = active_path(path, conditioning_set)
                        self._add_assumption(path_assumption_name, initial_weight=self.default_weight)

                        # Add argument that active path attacks no-edge assumption and vice versa
                        self._add_argument(claim=asm.contrary(asm.noe(node1, node2)), premise=[path_assumption_name])
                        self._add_argument(claim=asm.contrary(path_assumption_name), premise=[asm.noe(node1, node2)])

                        # Add argument that active path attacks independence assumption and vice versa
                        indep_assumption_name = asm.indep(node1, node2, conditioning_set)
                        self._add_argument(claim=asm.contrary(indep_assumption_name), premise=[path_assumption_name])
                        self._add_argument(claim=asm.contrary(path_assumption_name), premise=[indep_assumption_name])

                        # supporting collider structure for nodes in conditioning set,  mutually attacking anything else
                        for pos in range(1, len(path) - 1):
                            if path[pos] in conditioning_set:
                                # Add argument that active path supports collider structure
                                self._add_argument(claim=asm.arr(path[pos-1], path[pos]), premise=[path_assumption_name])
                                self._add_argument(claim=asm.arr(path[pos+1], path[pos]), premise=[path_assumption_name])
                                # attack and get attacked by anything else
                                self._add_argument(claim=asm.contrary(asm.arr(path[pos], path[pos-1])),
                                                premise=[path_assumption_name])
                                self._add_argument(claim=asm.contrary(asm.arr(path[pos], path[pos+1])),
                                                premise=[path_assumption_name])
                                self._add_argument(claim=asm.contrary(path_assumption_name),
                                                premise=[asm.arr(path[pos], path[pos+1])])
                                self._add_argument(claim=asm.contrary(path_assumption_name),
                                                premise=[asm.arr(path[pos], path[pos-1])])

    def build_arguments(self):
        """Build all arguments for the BSAF.
        This method should be called after all assumptions are added.
        """
        self._add_arr_and_mutual_exclusion_arguments()
        self._add_cycle_arguments()
        self._add_indep_assums_and_indep_noe_arguments()
        if self.include_collider_tree_arguments:
            self._add_active_path_assums_and_arguments()
        else:
            self._add_active_path_assums_and_arguments_wo_collider_trees()

    def create_bsaf(self):
        """
        Create the BSAF with the assumptions and arguments built from the facts.
        """
        self.build_arguments()

        # Create BSAF with the assumptions and arguments
        self.bsaf = BSAF(assumptions=set(self.name_to_assumption.values()),
                         arguments=set())
        i = 0
        for argument in tqdm(self.arguments,
                             desc="Adding arguments to BSAF",
                             total=len(self.arguments)):
            i += 1
            if i % 1000 == 0:
                if check_memory_usage_gb() > MEMORY_THRESHOLD:
                    logger.error("Memory usage exceeded threshold, stopping BSAF creation.")
                    raise MemoryUsageExceededException("Memory usage exceeded threshold, stopping BSAF creation.")

            self.bsaf.arguments.add(argument)
            if is_contrary(argument.claim):  # add attack
                assumption = self._get_assumption_from_contrary(argument.claim)
                self.bsaf.attacks[assumption].add(frozenset(argument.premise))
            else:  # add support
                self.bsaf.supports[argument.claim].add(frozenset(argument.premise))
        return self.bsaf


if __name__ == "__main__":
    # Example usage
    n_nodes = 11
    bsaf_builder = BSAFBuilderV2(n_nodes=n_nodes,
                                 neighbourhood_n_nodes=4,
                                 max_conditioning_set_size=3)
    bsaf = bsaf_builder.create_bsaf()
    # print(bsaf)
