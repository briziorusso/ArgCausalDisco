from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value, sort_dict_ascending
import math, statistics
import numpy as np
import pandas as pd
from operator import itemgetter
from typing import List
import networkx as nx
from itertools import chain, combinations, groupby, compress
import logging
from collections import defaultdict
import time
# from multiprocessing import Pool, Process, Manager
from joblib import Parallel, delayed

# import multiprocessing as mp
# mp.set_start_method("spawn")

import igraph as ig
def is_dag(W):
    # G = ig.Graph.Weighted_Adjacency(W.tolist())
    # return G.is_dag()
    return nx.is_directed_acyclic_graph(nx.from_numpy_array((W > 0).astype(int), create_using=nx.DiGraph))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    # manager = Manager()
    # return_dict = manager.dict()
    # jobs = []
    # for i in range(0, len(sv_params)):
    #     p = Process(target=calculate_sv, args=sv_params[i])
    #     jobs.append(p)
    #     p.start()
    # for proc in jobs:
    #     proc.join()
    # results = return_dict.values()

## Define function to calculate shapley values
def calculate_sv(i, t_key, t_val, s_key, s_val, w_i):
    # i, t_key, t_val, s_key, s_val, w_i = args
    if set(s_key).issubset(set(t_key)) and set(t_key) - set(s_key) == {i}:
        v_i = t_val - s_val
        sv_i = v_i * w_i
        return i, sv_i

group_tups = lambda tu : [(k, sum(v2[1] for v2 in v)) for k, v in groupby(tu, lambda x: x[0])]

from functools import reduce

def expand_grid(*arrs):
    ncols = len(arrs)
    nrows = reduce(lambda x, y: x * y, map(len, arrs), 1)
    
    return np.array(np.meshgrid(*arrs)).reshape(ncols, nrows).T


def shapley(x: int, y: int, num_of_nodes: int, sepset: np.ndarray, prec_lev: int = 10, rebase: bool = False, verbose: bool = False) -> list:
    s_p = [i for i in list(set(sepset[x, y])) if len(i) > 0 and type(i[0]) == tuple]
    ## coalitions
    sets_S = set().union(*[i[0] for i in s_p])
    ## number of players corresponds to the size of the biggest coalition
    max_set_size = max([len(i[0]) for i in s_p])
    n_factorial = math.factorial(max_set_size)
    ## weights for the shapley value
    w_i_len = {l: math.factorial(l) * math.factorial(max_set_size - l - 1) / n_factorial for l in range(0, max_set_size)}

    # start = time.time()
    # # sv_list = []
    # sv_is = []
    # subtime = time.time()
    # sv_params = []
    # for i in sets_S - {x, y}:
    #     ## select p-values
    #     s_p_i = [t for t in s_p if i in t[0]]
    #     without_i = [t for t in s_p if i not in t[0]]
    #     ## calculate shapley value
    #     for t_key, t_val in s_p_i:
    #         for s_key, s_val in without_i:
    #             if len(s_key) == len(t_key) - 1:
    #                 sv_params.append((i, t_key, t_val, s_key, s_val, w_i_len[len(s_key)]))
    # # sv_params = [(i, t_key, t_val, s_key, s_val, w_i_len[len(s_key)]) for i in sets_S for t_key, t_val in [t for t in s_p if i in t[0]] for s_key, s_val in [sp for sp in [t for t in s_p if i not in t[0]] if len(sp[0]) == len(t_key)-1]]    
    # print(f"sv_params: {time.time()-subtime}")
    # subtime = time.time()
    # mask = [set(a)-set(b)=={i} for i,a,_,b,_,_ in sv_params]
    # print(f"mask: {time.time()-subtime}")
    # subtime = time.time()
    # sv_params = list(compress(sv_params, mask))
    # print(f"compress: {time.time()-subtime}")
    # subtime = time.time()
    # for i, t_key, t_val, s_key, s_val, w_i in sv_params:
    #     v_i = t_val - s_val ## marginal contribution of i 
    #     w_i = w_i_len[len(s_key)]
    #     sv_i = v_i * w_i
    #     sv_is.append((i,sv_i))
    # print(f"sv_is: {time.time()-subtime}")
    # subtime = time.time()
    # sv_list = group_tups(sv_is)
    # print(f"group_tups: {time.time()-subtime}")

    # with Pool() as pool:
    #     results = pool.starmap(calculate_sv, sv_params)
    # pool.close()
    # pool.join()
             # results = map(calculate_sv, sv_params)
    # sv_is = [x for x in results if x is not None]
    ## sum up the shapley values for each player i
    # sv_list = group_tups(sv_is)
    # m1 = time.time()-start

    start = time.time()
    sv_list2 = []
    for i in sets_S - {x, y}:
        avg_sv_i = 0
        ## select p-values
        s_p_i = [t for t in s_p if i in t[0]]
        without_i = [t for t in s_p if i not in t[0]]

        len_cs = max_set_size
        # while len_cs >= 1:
        #     for t in [sp for sp in s_p_i if len(sp[0]) == len_cs]:
        #         # s = [wi for wi in without_i if set(wi[0])==set(t[0])-{i}]     
        #         for s in [wi for wi in without_i if len(wi[0]) == len_cs-1]:
        #             if set(t[0]) - set(s[0]) == {i}:
        #                 avg_sv_i += (t[1] - s[1]) * w_i_len[len(s[0])]
        #     len_cs -= 1

        while len_cs >= 1:
            t_filtered = [sp for sp in s_p_i if len(sp[0]) == len_cs]
            s_filtered = [wi for wi in without_i if len(wi[0]) == len_cs - 1]
            
            t_dict = {tuple(set(t[0]) - {i}): t[1] for t in t_filtered}
            for s in s_filtered:
                if s[0] in t_dict:
                    avg_sv_i += (t_dict[s[0]] - s[1]) * w_i_len[len(s[0])]

            len_cs -= 1
            
        # avg_sv_i = round(sum(sv_is), prec_lev)
        # if verbose:
            # logging.debug("SV of {} = {}".format(i, avg_sv_i))
        sv_list2.append((i, avg_sv_i))

    # m2 = time.time()-start
    # print(f"M2 is quicker than M1: {m2 < m1}: {m2} vs {m1}")

    # print(sv_list == sv_list2)

    return sv_list2


## Define function to calculate shapley values
def shapley_loop(x:int, y:int, num_of_nodes:int, sepset:np.ndarray, prec_lev:int=10, rebase:bool=False, verbose:bool=False) -> list:
    """
    Calculate the Shapley value for all subsets of conditioning sets for the pair of variables x and y
    
    Parameters
    ----------
        x: variable x
        y: variable y
        num_of_nodes: number of nodes in the graph
        sepset: matrix of conditioning sets and p-values
        prec_lev: precision level
        verbose: print intermediate results

    Returns
    -------
        sv_list: list of tuples (i, sv_i) where i is the variable and sv_i is the Shapley value of i
    """

    ## Extract conditioning sets and p-values from the skeleton
    s_p = [i for i in list(set(sepset[x,y])) if len(i) > 0 and type(i[0]) == tuple]
    sets_S = set().union(*[i[0] for i in s_p])
    max_set_size = max([len(i[0]) for i in s_p])
    n_factorial = math.factorial(max_set_size)

    w_i_len = {l: math.factorial(l) * math.factorial(max_set_size - l - 1) / n_factorial for l in range(0, max_set_size)}

    sv_list = []
    for i in sets_S - {x, y}:
        sv_is = []
        ## select p-values
        s_p_i = [t for t in s_p if i in t[0]]
        # if len(s_p_i) == 0:
        #     ## No conditioning sets contain i
        #     sv_list.append((i, np.nan))
        #     continue
        without_i = [t for t in s_p if i not in t[0]]

        if rebase:
            ## Rebase the p-values around the p-value of the marginal test
            eps = 1e-10
            p_marginal = [t[1] for t in s_p if len(t[0]) == 0][0]
            s_p_i = [(t[0], t[1] / ((p_marginal+eps) if p_marginal==0 else p_marginal)) for t in s_p_i]
            without_i = [(t[0], t[1] / ((p_marginal+eps) if p_marginal==0 else p_marginal)) for t in without_i]

        # start = time.time()
        len_cs = max_set_size
        while len_cs >= 1:
            for t in [sp for sp in s_p_i if len(sp[0]) == len_cs]:
                # s = [wi for wi in without_i if set(wi[0])==set(t[0])-{i}]     
                for s in [s for s in without_i if len(s[0]) == len_cs-1]:
                    if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}:
                # if len(s) > 0:
                    # s = s[0]
                        v_i = t[1] - s[1] ## marginal contribution of i 
                        w_i = w_i_len[len(s[0])]
                        sv_i = v_i * w_i
                        sv_is.append(sv_i)
            len_cs -= 1
        # m2 = time.time()-start

        # ## calculate shapley value
        # start = time.time()
        # for t in s_p_i:
        #     for s in without_i:
        #         if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}: # if s is the only difference between t and t-{i}
        #             v_i = t[1] - s[1] ## marginal contribution of i 
        #             w_i = w_i_len[len(s[0])]
        #             sv_i = v_i * w_i
        #             sv_is.append(sv_i)
        #             if verbose:
        #                 logging.debug((t[0],s[0]), sv_i)
        # m1 = time.time()-start

        # print(f"M2 is quicker than M1: {m2 < m1}: {m2} vs {m1}")
            
        avg_sv_i = round(sum(sv_is), prec_lev)
        # if verbose:
            # logging.debug("SV of {} = {}".format(i, avg_sv_i))
        sv_list.append((i, avg_sv_i))

    return sv_list


## Define function to calculate shapley values
def shapley_ori(x:int, y:int, num_of_nodes:int, sepset:np.ndarray, prec_lev:int=10, rebase:bool=False, verbose:bool=False) -> list:
    """
    Calculate the Shapley value for all subsets of conditioning sets for the pair of variables x and y
    
    Parameters
    ----------
        x: variable x
        y: variable y
        num_of_nodes: number of nodes in the graph
        sepset: matrix of conditioning sets and p-values
        prec_lev: precision level
        verbose: print intermediate results

    Returns
    -------
        sv_list: list of tuples (i, sv_i) where i is the variable and sv_i is the Shapley value of i
    """

    ## Extract conditioning sets and p-values from the skeleton
    s_p = [i for i in list(set(sepset[x,y])) if len(i) > 0 and type(i[0]) == tuple]
    sets_S = set().union(*[i[0] for i in s_p])
    max_set_size = max([len(i[0]) for i in s_p])
    n_factorial = math.factorial(max_set_size)

    w_i_len = {l: math.factorial(l) * math.factorial(max_set_size - l - 1) / n_factorial for l in range(0, max_set_size)}

    sv_list = []
    for i in sets_S - {x, y}:
        sv_is = []
        ## select p-values
        s_p_i = [t for t in s_p if i in t[0]]
        # if len(s_p_i) == 0:
        #     ## No conditioning sets contain i
        #     sv_list.append((i, np.nan))
        #     continue
        without_i = [t for t in s_p if i not in t[0]]

        if rebase:
            ## Rebase the p-values around the p-value of the marginal test
            eps = 1e-10
            p_marginal = [t[1] for t in s_p if len(t[0]) == 0][0]
            s_p_i = [(t[0], t[1] / ((p_marginal+eps) if p_marginal==0 else p_marginal)) for t in s_p_i]
            without_i = [(t[0], t[1] / ((p_marginal+eps) if p_marginal==0 else p_marginal)) for t in without_i]

        start = time.time()
        len_cs = max_set_size
        while len_cs >= 1:
            for t in [sp for sp in s_p_i if len(sp[0]) == len_cs]:
                s = [wi for wi in without_i if set(wi[0])==set(t[0])-{i}]     
                # for s in [s for s in without_i if len(s[0]) == len_cs-1]:
                #     if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}:
                if len(s) > 0:
                    s = s[0]
                    v_i = t[1] - s[1] ## marginal contribution of i 
                    w_i = w_i_len[len(s[0])]
                    sv_i = v_i * w_i
                    sv_is.append(sv_i)
            len_cs -= 1
        m2 = time.time()-start

        ## calculate shapley value
        start = time.time()
        for t in s_p_i:
            for s in without_i:
                if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}: # if s is the only difference between t and t-{i}
                    v_i = t[1] - s[1] ## marginal contribution of i 
                    w_i = w_i_len[len(s[0])]
                    sv_i = v_i * w_i
                    sv_is.append(sv_i)
                    if verbose:
                        logging.debug((t[0],s[0]), sv_i)
        m1 = time.time()-start

        print(f"M2 is quicker than M1: {m2 < m1}: {m2} vs {m1}")
            
        avg_sv_i = round(sum(sv_is), prec_lev)
        # if verbose:
            # logging.debug("SV of {} = {}".format(i, avg_sv_i))
        sv_list.append((i, avg_sv_i))

    return sv_list


def check_UTs(cg, UT_gamma=0.0015, DAG_check=True, verbose=False):

    cg_new = deepcopy(cg)

    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples
    edge_to_remove = {}
    edges_to_add = {}
    # edge_removal = defaultdict(list)
    # confounding_edges = defaultdict(list)
    # confounders = defaultdict(list)
    for (x, y, z) in UT:
        ## find neighbors of x and y that could be confounders
        neigh_y = set(cg_new.neighbors(y))
        neigh_x = set(cg_new.neighbors(x))
        ## second degree neighbors of x and y
        neigh_x_2 = set()
        neigh_y_2 = set()
        # for i in neigh_x:
        #     neigh_x_2 = neigh_x_2.union(cg_new.neighbors(i))
        # for i in neigh_y:
        #     neigh_y_2 = neigh_y_2.union(cg_new.neighbors(i))
        neigh_x_y = neigh_x.union(neigh_y).union(neigh_x_2).union(neigh_y_2) - {x, y, z}
        if len(neigh_x_y) > 0:
            cond_b_x_y = powerset(neigh_x_y)
            cond_b_x_y_p = [(S,cg_new.ci_test(x, y, S)) for S in cond_b_x_y]
            [append_value(cg_new.sepset, x, y, (S,p)) for S,p in cond_b_x_y_p]
            if verbose:
                logging.debug(f"cond_b_{x}_{y}: {cond_b_x_y_p}")
        
        ## find neighbors of z and y that could be confounders
        neigh_z = set(cg_new.neighbors(z))
        ## second degree neighbors of z and y
        neigh_z_2 = set()
        # for i in neigh_z:
        #     neigh_z_2 = neigh_z_2.union(cg_new.neighbors(i))
        neigh_z_y = neigh_z.union(neigh_y).union(neigh_z_2).union(neigh_y_2) - {z, y, x}
        if len(neigh_z_y) > 0:
            cond_b_z_y = powerset(neigh_z_y)
            cond_b_z_y_p = [(S,cg_new.ci_test(z, y, S)) for S in cond_b_z_y]
            [append_value(cg_new.sepset, z, y, (S,p)) for S,p in cond_b_z_y_p]
            if verbose:
                logging.debug(f"cond_b_{z}_{y}: {cond_b_z_y_p}")
                
        ## Calculate the shapley value of the neighbors of x and z
        sv_list_x = shapley(x, y, len(cg_new.G.nodes), cg_new.sepset, verbose=verbose)
        sv_list_z = shapley(z, y, len(cg_new.G.nodes), cg_new.sepset, verbose=verbose)
        ## remove nan values (no conditioning sets contain i)
        sv_list_x = [sv for sv in sv_list_x if not np.isnan(sv[1]) and sv[0] not in [x, y, z]]
        sv_list_z = [sv for sv in sv_list_z if not np.isnan(sv[1]) and sv[0] not in [x, y, z]]
        ## identify possible confounders for the UT by maximizing shapley value
        confounder_x_y, confounders_z_y = 0, 0
        if len([sv[1] for sv in sv_list_x if sv[1] > 0]) != 0:
            confounder_x_y = max([sv[1] for sv in sv_list_x if sv[1] > 0])
        if len([sv[1] for sv in sv_list_z if sv[1] > 0]) != 0:
            confounders_z_y = max([sv[1] for sv in sv_list_z if sv[1] > 0])
        if verbose:
            logging.debug(f"max confounder_x_y: {[sv for sv in sv_list_x if sv[1] == confounder_x_y]}")
            logging.debug(f"max confounder_z_y: {[sv for sv in sv_list_z if sv[1] == confounders_z_y]}")
        
        ## Found counfounder, removing edge and reconsidering the UT
        if confounder_x_y > UT_gamma:
            confounder = [sv for sv in sv_list_x if sv[1] == confounder_x_y][0][0]
            logging.debug(f"Found confounder for {x} -- {y}: {confounder}")
            # edge_removal[(x,y,z)].append((x, y))
            # confounding_edges[(x,y,z)].append((confounder, x))
            # confounding_edges[(x,y,z)].append((confounder, y))
            # # confounders[(x,y,z)].append(((x,y), confounder, confounder_SIV))
            # confounders[(f"X{x+1}",f"X{y+1}",f"X{z+1}")].append([(f"X{x+1}",f"X{y+1}"), f"X{confounder+1}", confounder_x_y, [(f"X{confounder+1}",f"X{x+1}"), (f"X{confounder+1}",f"X{y+1}")]])
            edge = (x,y) if x < y else (y,x)
            edge_to_remove[edge] = confounder_x_y
            edges_to_add[edge] = [(confounder, x), (confounder, y)]
            # if verbose:
            #     logging.debug(f"Oriented: {confounder} --> {x} ({cg_new.G.nodes[confounder].get_name()} --> {cg_new.G.nodes[x].get_name()})")
            #     logging.debug(f"Oriented: {confounder} --> {y} ({cg_new.G.nodes[confounder].get_name()} --> {cg_new.G.nodes[y].get_name()})")
        if confounders_z_y > UT_gamma:
            confounder = [sv for sv in sv_list_z if sv[1] == confounders_z_y][0][0]
            logging.debug(f"Found confounder for {z} -- {y}: {confounder}")
            # edge_removal[(x,y,z)].append((z, y))
            # confounding_edges[(x,y,z)].append((confounder, z))
            # confounding_edges[(x,y,z)].append((confounder, y))
            # # confounders[(x,y,z)].append(((z,y), confounder, confounder_SIV))
            # confounders[(f"X{x+1}",f"X{y+1}",f"X{z+1}")].append([(f"X{z+1}",f"X{y+1}"), f"X{confounder+1}", confounders_z_y, [(f"X{confounder+1}",f"X{z+1}"), (f"X{confounder+1}",f"X{y+1}")]])
            edge = (z,y) if z < y else (y,z)
            edge_to_remove[edge] = confounders_z_y
            edges_to_add[edge] = [(confounder, z), (confounder, y)]
            # if verbose:
            #     logging.debug(f"Oriented: {confounder} --> {z} ({cg_new.G.nodes[confounder].get_name()} --> {cg_new.G.nodes[z].get_name()})")
            #     logging.debug(f"Oriented: {confounder} --> {y} ({cg_new.G.nodes[confounder].get_name()} --> {cg_new.G.nodes[y].get_name()})")

        ## TODO: think about SIV=0
    ## aggregate edges independently of direction
    for k,v in edge_to_remove.items():
        edge_to_remove[k] += v

    ## sort the edges to remove by the shapley value
    edge_to_remove = sort_dict_ascending(edge_to_remove, True)

    edge_removed = []
    for (x, y) in edge_to_remove.keys():
        if (x, y) not in edge_removed: ## TODO: think about added edges
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
                edge_removed.append((x, y))
                if verbose:
                    logging.debug(f"Removed: {x} --- {y} ({cg_new.G.nodes[x].get_name()} --- {cg_new.G.nodes[y].get_name()})")
                ## add confounding edges
                for (i, j) in edges_to_add[(x, y)]:
                    edge = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[j])
                    if edge is not None:
                        cg_new.G.remove_edge(edge)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW)) ## TODO: check if not better to add as tail-tail
                    if verbose:
                        logging.debug(f"Oriented: {i} --> {j} ({cg_new.G.nodes[i].get_name()} --> {cg_new.G.nodes[j].get_name()})")
                if not is_dag(cg_new.G.graph > 0) and DAG_check:
                    ## Disorient the confounding edges
                    for (i, j) in edges_to_add[(x, y)]: 
                        cg_new.G.remove_edge(cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[i]))
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[j], Endpoint.TAIL, Endpoint.TAIL))
                        if verbose:
                            logging.debug(f'Disoriented because of DAG condition: {j} --- {i} ({cg_new.G.nodes[j].get_name()} --- {cg_new.G.nodes[i].get_name()})')      
                    ## Reinstate the original edge
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))
                    if verbose:
                        logging.debug(f'Reinstated: {x} --- {y} ({cg_new.G.nodes[x].get_name()} --- {cg_new.G.nodes[y].get_name()})')   

    revised_UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    return cg_new, revised_UT

def shapley_cs(cg: CausalGraph, priority: int = 2, background_knowledge: BackgroundKnowledge = None, 
                selection: str = 'neg', extra_tests: bool = False, UT_check: bool = False, UT_gamma: float = 0.01, DAG_check: bool = True,
                return_svs:bool=False, verbose: bool = False ) -> CausalGraph:
    """
    Run (ShapleyPC) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 2)
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

    if UT_check:
        cg_new, UT = check_UTs(cg_new, UT_gamma=UT_gamma, verbose=verbose)
        logging.info(f"Number of undirected edges: {len(cg_new.find_undirected())}")
        logging.info(f"Number of directed edges: {len(cg_new.find_fully_directed())}")
    else:
        UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    UC_dict = {}
    est_svs = []

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
        if verbose:
            logging.debug(f"cond_with_y: {cond_with_y_p}")
        [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_with_y_p]

        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
        if verbose:
            logging.debug(f"cond_without_y:{cond_without_y_p}")
        ## Add additional test to sepset_list if not already there
        [append_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_without_y_p]

        if extra_tests:
            logging.debug("Calculating additional tests...")
            ## Fetch missing sets for shapley calculation
            union_of_sepsets = set().union(*[set(f) for f in cond_without_y + cond_with_y])
            power_set = list(powerset(union_of_sepsets))

            if len(power_set) > len(cond_without_y + cond_with_y):
                missing_sets = [s for s in power_set if s not in cond_without_y + cond_with_y]
                missing_sets_p = [(S,cg_new.ci_test(x, z, S)) for S in missing_sets]
                [append_value(cg_new.sepset, x, z, (S,p)) for S,p in missing_sets_p]
                if verbose:
                    logging.debug(f"missing_sets: {missing_sets_p}")

        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                    background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        sv_list = shapley(x, z, len(cg_new.G.nodes), cg_new.sepset, verbose=verbose)
        ## remove nan values (no conditioning sets contain i)
        sv_list = [sv for sv in sv_list if not np.isnan(sv[1])]
        ## sort shapley values
        sv_list.sort(key=lambda x: x[1], reverse=False)
        est_svs.append({(x,z):sv_list})
        if verbose:
            logging.debug(str(sv_list))

        ## Base Option: deem collider if SIV is negative
        if selection == 'neg':
            if y not in [sv[0] for sv in sv_list if sv[1] < 0]:
                logging.debug("Tentatively orienting? False")
                continue
            else:
                if verbose:
                    logging.debug("Tentatively orienting? True")
            
        ## Option 1: only take the lowest shapley value as a v-structure candidate
        elif selection == 'bot':
            ## select the top candidate for dependency (the lowest contribution to the p-value for rejecting the null) 
            y_star = [sv[0] for sv in sv_list if sv[1] == min(sv_list, key=itemgetter(1))[1] and sv[1] < 0]
            if verbose:
                logging.debug(f"y*={y_star}")
            if y not in y_star:
                continue

        ## Option 2: accept all the candidates that are in the 2 lowest SVs
        elif selection == 'bot2':
            if len(sv_list) >= 2:
                if y not in [s[0] for s in sv_list[:1] if s[1] < 0]:
                    continue
            elif len(sv_list) == 1:
                if y not in [s[0] for s in sv_list if s[1] < 0]:
                    continue
            else:
                continue            

        ## Option 3: take the lowest shapley value as a v-structure candidate if it is lower than the median
        elif selection == 'median':
            median_sv = statistics.median([sv[1] for sv in sv_list])
            node_sv = [sv for sv in sv_list if sv[0]==y][1]
            if node_sv > 0 or node_sv > median_sv:
                continue

        ## Option 4: accept all the candidates that are lower than the one with the biggest increment
        elif selection == 'top_change':
            arr = pd.DataFrame([sv for sv in sv_list if sv[1]<0], columns=['Var', 'SV']).sort_values('SV', ascending=True)
            arr['change'] = arr['SV'].diff()
            if len(arr) == 0:
                continue # no negative SVs, all contribute positively to the p-value
            if y not in arr.loc[np.where(max(arr['change']))[0][0]:,'Var'].values:
                continue # y is not in the top change        

        else:
            raise ValueError(f"Selection method {selection} not recognized")

        # if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)): ## if not already oriented in the opposite direction
        #     if verbose:
        #         logging.debug(f"{y} -- {x} and {y} -- {z}")

        if priority == 0:  # 0: overwrite
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            # Fully orient the edge irrespective of what have been oriented
            cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            if verbose:
                logging.debug(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
            if edge3 is not None:
                cg_new.G.remove_edge(edge3)
            edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge4 is not None:
                cg_new.G.remove_edge(edge4)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            if verbose:
                logging.debug(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

        ### Conclusions: Orient V-structure
        elif priority == 1:  # 1: orient bi-directed
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                    cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        logging.debug(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                    cg_new.G.remove_edge(edge1)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                    if verbose:
                        logging.debug(f'Oriented: x={x} <--> y={y} ({cg_new.G.nodes[x].get_name()} <--> {cg_new.G.nodes[y].get_name()})')
            else:
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    logging.debug(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                    cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        logging.debug(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                    cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                    if verbose:
                        logging.debug(f'Oriented: z={z} <--> y={y} ({cg_new.G.nodes[z].get_name()} <--> {cg_new.G.nodes[y].get_name()})')
            else:
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    logging.debug(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

        elif priority == 2:  # 2: prioritize existing
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

        elif priority == 3: ## 3. Prioritize stronger colliders
            UC_dict[(x, y, z)] = [sv[1] for sv in sv_list if sv[0] == y][0]

        if priority in [0, 1, 2] and not is_dag(cg_new.G.graph > 0) and DAG_check:
            ## Disorient the v-structure
            cg_new.G.remove_edge(cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y]))
            cg_new.G.add_edge(Edge(cg_new.G.nodes[y], cg_new.G.nodes[z], Endpoint.TAIL, Endpoint.TAIL))
            ## Disorient the first edge
            cg_new.G.remove_edge(cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y]))
            cg_new.G.add_edge(Edge(cg_new.G.nodes[y], cg_new.G.nodes[x], Endpoint.TAIL, Endpoint.TAIL))
            if verbose:
                logging.debug(f'Disoriented because of DAG condition: z={z} --- y={y} ({cg_new.G.nodes[z].get_name()} --- {cg_new.G.nodes[y].get_name()})')
                logging.debug(f'Disoriented because of DAG condition: x={x} --- y={y} ({cg_new.G.nodes[x].get_name()} --- {cg_new.G.nodes[y].get_name()})')

    if priority in [0, 1, 2]:
        return cg_new

    else:
        ## 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sort_dict_ascending(UC_dict)
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
            
                if not is_dag(cg_new.G.graph > 0) and DAG_check:
                    ## Disorient the second edge
                    cg_new.G.remove_edge(cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y]))
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[y], cg_new.G.nodes[z], Endpoint.TAIL, Endpoint.TAIL))
                    ## Disorient the first edge
                    cg_new.G.remove_edge(cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y]))
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[y], cg_new.G.nodes[x], Endpoint.TAIL, Endpoint.TAIL))
                    if verbose:
                        logging.debug(f'Disoriented because of DAG condition: z={z} --- y={y} ({cg_new.G.nodes[z].get_name()} --- {cg_new.G.nodes[y].get_name()})')
                        logging.debug(f'Disoriented because of DAG condition: x={x} --- y={y} ({cg_new.G.nodes[x].get_name()} --- {cg_new.G.nodes[y].get_name()})')
    if return_svs:
        return cg_new, UT, est_svs
    else:
        return cg_new

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