from itertools import chain, combinations
import numpy as np
import networkx as nx

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def model_to_adjacency_matrix(model:list, num_of_nodes:int)->np.ndarray:
    adj_mat = np.zeros((num_of_nodes,num_of_nodes))
    for atom in model:
        if atom.name == 'arrow':
            adj_mat[int(atom.arguments[0].name[1])][int(atom.arguments[1].name[1])] = 1
    return adj_mat

def model_to_set_of_arrows(model:list, num_of_nodes:int)->set:
    arrows = set()
    for atom in model:
        if atom.name == 'arrow':
            arrows.add((atom.arguments[0].number,atom.arguments[1].number))
    return arrows

def find_all_d_separations_sets(G, verbose=True, debug=False):
    no_of_var = len(G.nodes)
    septests = []
    for comb in combinations(range(no_of_var), 2):
        if debug:
            print(comb)
        if comb[0] != comb[1]:
            x = comb[0]
            y = comb[1]
            if debug:
                print(x,y)
            depth = 0
            while no_of_var-1 > depth:
                Neigh_x_noy = [f"X{k+1}" for k in range(no_of_var) if k != x and k != y]
                if debug:
                    print(Neigh_x_noy)
                for S in combinations(Neigh_x_noy, depth):
                    if debug:
                        print(S)
                    s = set([int(s.replace('X',''))-1 for s in S])
                    s_str = 'empty' if len(S)==0 else 's'+''.join([str(i) for i in s])
                    if nx.algorithms.d_separated(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        if verbose:
                            print(f"X{x+1} and X{y+1} are d-separated by {S}")
                        septests.append(f"indep({x},{y},{s_str}).")
                    else:
                        # print(f"X{x+1} and X{y+1} are not d-separated by {S}")
                        septests.append(f"dep({x},{y},{s_str}).")
                depth += 1
    return septests