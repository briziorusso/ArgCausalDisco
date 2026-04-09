# Atoms

def dpath(X, Y):
    return f"dpath_{X}_{Y}"


def collider(X, Y, Z):
    # colliding on middle node
    # collider_X_Y_Z equivalent to X->Y<-Z
    # is simmetric with respect to X and Z
    if X > Z:
        X, Z = Z, X
    return f"collider_{X}_{Y}_{Z}"


def not_collider(X, Y, Z):
    # symmetric with respect to X and Z
    if X > Z:
        X, Z = Z, X
    return f"not_collider_{X}_{Y}_{Z}"


def descendant_of_collider(Z, X, N, Y):
    # descendant of collider
    # Z is the descendant node
    # X and Y are the colliding nodes
    # N is the middle node
    # symmetric with respect to X and Y
    if X > Y:
        X, Y = Y, X
    return f"desc_{Z}_of_collider_{X}_{N}_{Y}"


def non_blocking(N: int, X: int, Y: int, S: set):
    # N is non-blocking node for the path X-Y
    # X and Y immediate neighbours of N
    # S is the set of nodes for which the path is S-active

    # is symmetric with respect to X and Y
    if X > Y:
        X, Y = Y, X
    S = sorted(list(S))
    return f"nb_{N}__{X}_{Y}__" + '_'.join([str(i) for i in S])


def path(source: int, target: int, path_id: int):
    # path is symmetric with respect to source and target
    if source > target:
        source, target = target, source
    return f'path_{source}_{target}__{path_id}'


def edge(X, Y):
    """WARNING: This one is not an assumption.
       It is an atom that is the contrary of no-edge (noe) assumption.
    """
    # edge is symmetric
    if X > Y:
        X, Y = Y, X
    return f"edge_{X}_{Y}"
