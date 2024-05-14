"""Copyright 2024 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2024 Fabrizio Russo"

from itertools import combinations
import time
import warnings
from typing import List
from tqdm.auto import tqdm
import logging
import numpy as np
from numpy import ndarray
from copy import deepcopy

try:
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.graph.GraphClass import CausalGraph    
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.utils.PCUtils.Helper import append_value
    from causallearn.utils.PCUtils import Meek, UCSepset
    from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
        orient_by_background_knowledge
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.utils.PCUtils.Helper import append_value, sort_dict_ascending
except:
    import sys
    sys.path.append("../causal-learn/")
    from causallearn.graph.GraphClass import CausalGraph    
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.utils.PCUtils.Helper import append_value
    from causallearn.utils.PCUtils import Meek, UCSepset
    from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
        orient_by_background_knowledge
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.utils.PCUtils.Helper import append_value, sort_dict_ascending
from utils.cit import *

#This function overrides causal-learn/causallearn/search/ConstraintBased/PC.py
#It is the same as the original pc function, but with the uc_rule parameter allowing value of 3 for spc
def pc(
    data: ndarray, 
    alpha=0.05, 
    indep_test=fisherz, 
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    selection: str = 'bot',
    extra_tests: bool = False,
    mvpc: bool = False, 
    correction_name: str = 'MV_Crtn_Fisher_Z',
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is larger than the sample size!")

    if mvpc:  # missing value PC
        from causallearn.search.graph.PC import mvpc_alg
        if indep_test == fisherz:
            indep_test = mv_fisherz
        return mvpc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, correction_name=correction_name, stable=stable,
                        uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge,
                        verbose=verbose,
                        show_progress=show_progress, **kwargs)
    else:
        return pc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, 
                      uc_rule=uc_rule, uc_priority=uc_priority, background_knowledge=background_knowledge,
                        selection=selection, extra_tests=extra_tests, verbose=verbose, show_progress=show_progress, 
                        **kwargs)

#This function overrides causal-learn/causallearn/search/ConstraintBased/PC.py
#It is the same as the original pc function, but with the uc_rule parameter allowing value of 3 for spc
def pc_alg(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test: str,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    selection: str = "bot",
    extra_tests: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: original PC algorithm (default)
           4: Conservative PC (Ramsey et al., 2006)
           5: Majority PC (Colombo et al., 2014)
           1: run maxP (Ramsey, J. (2016). Improving accuracy and scalability of the pc algorithm by maximizing p-value. arXiv preprint arXiv:1610.00378.)
           2: run definiteMaxP (Ramsey, J. (2016))
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    indep_test = CIT(data, indep_test, **kwargs)
    cg_1 = skeleton_discovery(data, alpha, indep_test, stable,
                            background_knowledge=background_knowledge, verbose=verbose,
                            show_progress=show_progress, node_names=node_names, uc_rule=uc_rule)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        logging.info("   Starting Orientation by Separating Set (Original PC)")
        if uc_priority != -1:
            cg_1 = uc_sepset(cg_1,  alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=0)
        else:
            cg_1 = uc_sepset(cg_1,  alpha, background_knowledge=background_knowledge, uc_rule=0)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 4:
        logging.info("   Starting Orientation by Separating Set (Conservative)")
        if uc_priority != -1:
            cg_1 = uc_sepset(cg_1, alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=1)
        else:
            cg_1 = uc_sepset(cg_1, alpha,  background_knowledge=background_knowledge, uc_rule=1)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 5:
        logging.info("   Starting Orientation by Separating Set (Majority)")
        if uc_priority != -1:
            cg_1 = uc_sepset(cg_1,  alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=2)
        else:
            cg_1 = uc_sepset(cg_1,  alpha, background_knowledge=background_knowledge, uc_rule=2)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        logging.info("   Starting Orientation by MaxP (Ramsey, 2016)")
        if uc_priority != -1:
            cg_1 = maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_1 = maxp(cg_1, background_knowledge=background_knowledge)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        logging.info("   Starting Orientation by DefiniteMaxP (Ramsey, 2016)")
        if uc_priority != -1:
            cg_1 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_1 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_1 = Meek.definite_meek(cg_1, background_knowledge=background_knowledge)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    end = time.time()

    cg_1.PC_elapsed = end - start

    return cg_1


def skeleton_discovery(
    data: ndarray, 
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
    uc_rule: str = None,
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    tests_set = set()
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    if (x, y, S) in tests_set:
                        continue
                    tests_set.add((x, y, S))
                    tests_set.add((y, x, S)) # to avoid duplicate tests
                    p = cg.ci_test(x, y, S)
                    append_value(cg.sepset, x, y, (S,p))
                    append_value(cg.sepset, y, x, (S,p))                    
                    if p > alpha:
                        if verbose:
                            logging.info('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                    else:
                        if verbose:
                            logging.info('%d dep %d | %s with p-value %f\n' % (x, y, S, p))


        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress:
        pbar.close()
    logging.info("   Skeleton discovery done. Number of independence tests: %d" % (len(tests_set)/2))
    # logging.info("   Number of unique tests: %d" % (len(tests_set)/2))
    logging.info("   Number of edges in the skeleton: %d" % len(cg.find_undirected()))

    return cg

def uc_sepset(cg: CausalGraph, alpha: float = 0.05, priority: int = 3, uc_rule: int = 1, 
              background_knowledge: BackgroundKnowledge | None = None, verbose:bool = False) -> CausalGraph:
    """
    Run (UC_sepset) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    uc_rule : rule of chosing if trusting the sepset (default = 0)
              0: Original PC (Spirtes et al., 2000)
              1: Conservative PC (Ramsey et al., 2006)
              2: Majority PC (Colombo et al., 2014)
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)

    R0 = []  # Records of possible orientations
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        if uc_rule == 0:
            ### No additional tests carried out, only decided based on the tests from the skeleton phase
            condition = all(y not in S[0] for S in cg_new.sepset[x, z] if S[1] > alpha) ## S[1] > alpha are the separating sets (p-value > alpha)
            if verbose:
                logging.debug(f"sepsets: {[S for S in cg_new.sepset[x, z] if S[1] > alpha]}")
                logging.debug(f"condition y not in sepsets: {condition}")
        elif uc_rule == 1: 
            ## from https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf page 5
            # the algorithm determines all subsets of Adj(a) \ c (where Adj(a) are all nodes adjecent to a) and Adj(c) \ a that make a and c
            # conditionally independent. They are called separating sets. 
            cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
            cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
            [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_with_y_p]
            if verbose:
                logging.debug(f"cond_with_y: {cond_with_y_p}")

            cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
            cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
            [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_without_y_p]
            if verbose:
                logging.debug(f"cond_without_y:{cond_without_y_p}")

            # In the conservative version x−y−z is oriented as x → y ← z if y is in none of the separating sets.
            condition = all(y not in S[0] for S in cg_new.sepset[x, z] if S[1] > alpha) ## S[1] > alpha are the separating sets (p-value > alpha)
            if verbose:
                logging.debug(f"sepsets: {[S for S in cg_new.sepset[x, z] if S[1] > alpha]}")
                logging.debug(f"condition y not in any sepsets of adj nodes: {condition}")
        elif uc_rule == 2: 
            ## from https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf page 5
            # the algorithm determines all subsets of Adj(a) \ c (where Adj(a) are all nodes adjecent to a) and Adj(c) \ a that make a and c
            # conditionally independent. They are called separating sets. 
            cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
            cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
            [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_with_y_p]
            if verbose:
                logging.debug(f"cond_with_y: {cond_with_y_p}")

            cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
            cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
            [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_without_y_p]
            if verbose:
                logging.debug(f"cond_without_y:{cond_without_y_p}")

            # In the majority rule version the triple x − y − z is marked as "ambiguous"
            # if and only if y is in exactly 50 percent of such separating sets or no separating set was found. 
            # If y is in less than 50 percent of the separating sets it is set as a
            # v-structure, and if in more than 50 percent it is set as a non v-structure.
            condition = len(set().union([S for S in cg_new.sepset[x, z] if y in S[0] and S[1]>alpha])) < \
                            len(set().union([S for S in cg_new.sepset[x,z] if S[1] > alpha])) / 2
            if verbose:
                logging.debug(f"sepsets: {[S for S in cg_new.sepset[x, z] if S[1] > alpha]}")
                logging.debug(f"condition y not in majority of sepsets of adj nodes: {condition}")

        if condition:
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            else:
                R0.append((x, y, z))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            for (x, y, z) in R0:
                UC_dict[(x, y, z)] = max([S[1] for S in cg_new.sepset[x, z]])
            UC_dict = sort_dict_ascending(UC_dict)

        else:  # 4. Order colliders by p_{xy|not y} in descending order
            for (x, y, z) in R0:
                UC_dict[(x, y, z)] = max([S[1] for S in cg_new.sepset[x, z]])
            UC_dict = sort_dict_ascending(UC_dict, descending=True)

        for (x, y, z) in UC_dict.keys():
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue
            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

        return cg_new

def maxp(cg: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None, verbose: bool = True) -> CausalGraph:
    """
    Run (MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            logging.debug(str((x, y, z)))
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
        [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_with_y_p]
        if verbose:
            logging.debug(f"cond_with_y: {cond_with_y_p}")

        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
        [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_without_y_p]
        if verbose:
            logging.debug(f"cond_without_y:{cond_without_y_p}")

        max_p_contain_y = max([S[1] for S in cond_with_y_p])
        max_p_not_contain_y = max([S[1] for S in cond_without_y_p])

        if verbose:
            logging.debug(f"max_p_contain_y: {max_p_contain_y}")
            logging.debug(f"max_p_not_contain_y: {max_p_not_contain_y}")
            logging.debug(f"Tentatively orienting? {max_p_not_contain_y > max_p_contain_y}")

        if max_p_not_contain_y > max_p_contain_y:
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)

        if verbose:
            logging.debug(f"UC_dict: {UC_dict}")

        for (x, y, z) in UC_dict.keys():
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue

            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    logging.debug(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    logging.debug(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

        return cg_new