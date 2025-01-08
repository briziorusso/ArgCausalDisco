from __future__ import annotations
from itertools import combinations

import time
import warnings
from typing import List
from tqdm.auto import tqdm
import logging

import numpy as np
from numpy import ndarray


try:
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.utils.cit import *
    from causallearn.utils.PCUtils.Helper import append_value
    from causallearn.utils.PCUtils import Helper, Meek, UCSepset
    from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
        orient_by_background_knowledge
except:
    import sys
    sys.path.append("../causal-learn/")
    from causallearn.graph.GraphClass import CausalGraph    
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.utils.cit import *
    from causallearn.utils.PCUtils.Helper import append_value
    from causallearn.utils.PCUtils import Helper, Meek, UCSepset
    from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
        orient_by_background_knowledge
import cd_algorithms.spc as spc

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
           3: run ShaplePC (Shapley causal discovery. arXiv preprint arXiv:XXXX.XXXXXX.)
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
            cg_1 = spc.uc_sepset(cg_1,  alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=0)
        else:
            cg_1 = spc.uc_sepset(cg_1,  alpha, background_knowledge=background_knowledge, uc_rule=0)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 4:
        logging.info("   Starting Orientation by Separating Set (Conservative)")
        if uc_priority != -1:
            cg_1 = spc.uc_sepset(cg_1, alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=1)
        else:
            cg_1 = spc.uc_sepset(cg_1, alpha,  background_knowledge=background_knowledge, uc_rule=1)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 5:
        logging.info("   Starting Orientation by Separating Set (Majority)")
        if uc_priority != -1:
            cg_1 = spc.uc_sepset(cg_1,  alpha, uc_priority, background_knowledge=background_knowledge, uc_rule=2)
        else:
            cg_1 = spc.uc_sepset(cg_1,  alpha, background_knowledge=background_knowledge, uc_rule=2)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        logging.info("   Starting Orientation by MaxP (Ramsey, 2016)")
        if uc_priority != -1:
            cg_1 = spc.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_1 = spc.maxp(cg_1, background_knowledge=background_knowledge)
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

    elif uc_rule == 3:
        logging.info("   Starting Orientation by Shapley PC")
        if uc_priority != -1:
            cg_1 = spc.shapley_cs(cg_1, uc_priority, background_knowledge=background_knowledge, selection=selection, extra_tests=extra_tests, verbose=verbose)
        else:
            cg_1 = spc.shapley_cs(cg_1, background_knowledge=background_knowledge, selection=selection, extra_tests=extra_tests, verbose=verbose)
        logging.info("   Starting propagation by Meek Rules (Meek, 1995)")
        cg_1 = Meek.meek(cg_1, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2, 3, 4, 5]")
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
        
    if verbose:
        logging.info("   Skeleton discovery done. Number of independence tests: %d" % (len(tests_set)/2))
        logging.info("   Number of edges in the skeleton: %d" % len(cg.find_undirected()))
    else:
        logging.debug("   Skeleton discovery done. Number of independence tests: %d" % (len(tests_set)/2))
        logging.debug("   Number of edges in the skeleton: %d" % len(cg.find_undirected()))

    return cg
