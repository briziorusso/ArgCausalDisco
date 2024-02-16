from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
import math, statistics
import numpy as np
import pandas as pd
from operator import itemgetter
from typing import List

import igraph as ig
def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ##TODO: compare to nx.is_directed_acyclic_graph(G)
    return G.is_dag()

def add_value(array, i, j, value):
    """
    Append value to the list at array[i, j]
    """
    if array[i, j] is None:
        array[i, j] = [value]
    else:
        array[i, j].append(value) if value not in array[i, j] else None

## Define function to calculate shapley values
def shapley(x:int, y:int, cg_1:CausalGraph, verbose:bool=False) -> list:
    """
    Calculate the Shapley value for all subsets of conditioning sets for the pair of variables x and y
    
    Parameters
    ----------
        x: variable x
        y: variable y
        cg_1: CausalGraph object
        verbose: print intermediate results

    Returns
    -------
        sv_list: list of tuples (i, sv_i) where i is the variable and sv_i is the Shapley value of i
    """

    num_of_nodes = len(cg_1.G.nodes)
    ## Extract conditioning sets and p-values from the skeleton  ##TODO: check if we have all the tests at all calls
    s_p = list(set(cg_1.sepset[x,y]))
    # s_p = [t for t in cg_1.sepset[x,y] if len(t) == 2 and type(t[1])==np.float64]
    max_set_size = max([len(i[0]) for i in s_p])
    n_factorial = math.factorial(max_set_size)

    sv_list = []
    for i in range(num_of_nodes):
        sv_is = []
        if i not in {x, y}:
            ## calculate marginal contribution of i
            without_i = [t for t in s_p if i not in t[0]]
            ## select p-values
            s_p_i = [t for t in s_p if i in t[0]]
            if len(s_p_i) == 0:
                ## No conditioning sets contain i
                sv_list.append((i, np.nan))
                continue

            ## calculate shapley value
            for t in s_p_i:
                for s in without_i:
                    if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}: # if s is the only difference between t and t-{i}
                        v_i = t[1] - s[1] ## marginal contribution of i 
                        w_i = math.factorial(len(s[0])) * math.factorial(max_set_size - len(s[0]) - 1) / n_factorial
                        sv_i = v_i * w_i
                        sv_is.append(sv_i)
                        if verbose:
                            print((t[0],s[0]), sv_i)
            avg_sv_i = round(sum(sv_is),6)
            if verbose:
                print("SV of {} = {}".format(i, avg_sv_i))
            sv_list.append((i, avg_sv_i))

    return sv_list

def shapley_cs(cg_new: CausalGraph, priority: int = 2, background_knowledge: BackgroundKnowledge = None, 
                verbose: bool = False, selection: str = 'top') -> CausalGraph:
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

    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
        if verbose:
            print(f"cond_with_y: {cond_with_y_p}")
        [add_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_with_y_p]

        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
        if verbose:
            print(f"cond_without_y:{cond_without_y_p}")
        ## Add additional test to sepset_list if not already there
        [add_value(cg_new.sepset, x, z, (S,p)) for S,p in cond_without_y_p]

        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                    background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        sv_list = shapley(x, z, cg_new)
        ## remove nan values (no conditioning sets contain i)
        sv_list = [sv for sv in sv_list if not np.isnan(sv[1])]
        ## sort shapley values
        sv_list.sort(key=lambda x: x[1], reverse=False)
        if verbose:
            print(sv_list)

        ## Option 1: only take the lowest shapley value as a v-structure candidate
        if selection == 'bot':
            ## select the top candidate for dependency (the lowest contribution to the p-value for rejecting the null) 
            y_star = [sv[0] for sv in sv_list if sv[1] == max(sv_list, key=itemgetter(1))[1]]
            
            if verbose:
                print(y_star)
            if y not in y_star:
                continue

        ## Option 2: accept all the candidates that are in the 2 lowest SVs
        elif selection == 'bot2':
            if len(sv_list) >= 2:
                if y not in [s[0] for s in sv_list[0:2]]:
                    continue
            elif len(sv_list) == 1:
                if y not in [s[0] for s in sv_list]:
                    continue
            else:
                continue            

        ## Option 3: take the highest shapley value as a v-structure candidate if it is lower than the median
        elif selection == 'median':
            median_sv = statistics.median([sv[1] for sv in sv_list])
            if [sv[1] for sv in sv_list if sv[0]==y][0] < median_sv:
                continue

        ## Option 4: accept all the candidates that are higher than the one with the biggest increment
        elif selection == 'top_change':
            arr = pd.DataFrame([sv for sv in sv_list], columns=['Var', 'SV']).sort_values('SV', ascending=True)
            arr['change'] = arr['SV'].diff()
            if len(arr) == 0:
                continue # no negative SVs, all contribute positively to the p-value
            if y not in arr.loc[np.where(max(arr['change']))[0][0]:,'Var'].values:
                continue # y is not in the top change        

        else:
            raise ValueError(f"Selection method {selection} not recognized")

        if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
            if verbose:
                print(f"{y} -- {x} and {y} -- {z}")

            ### Conclusions: Orient V-structure               
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            if verbose:
                print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                print("Is DAG?", is_dag(cg_new.G.graph > 0))
            
            ##TODO: we can check which one has the lowest p-value and remove the other one
            
            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            if verbose:
                print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                print("Is DAG?", is_dag(cg_new.G.graph > 0))

            if priority == 0: ## 0. Overwrite
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 0 to be implemented")
            if priority == 1: ## 1. Orient bi-directed
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 1 to be implemented")
            if priority == 2: ## 2. Prioritize existing colliders
                if not is_dag(cg_new.G.graph > 0):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))
                    if verbose:
                        print(f'Removed: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                        print(f'Removed: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
            elif priority == 3: ## 3. Prioritize stronger colliders
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 3 to be implemented")

    return cg_new
