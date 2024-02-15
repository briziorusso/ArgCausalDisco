from itertools import chain, combinations
import numpy as np

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