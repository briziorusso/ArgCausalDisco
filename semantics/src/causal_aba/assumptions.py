# Assumptions and their contraries

def contrary(assumption):
    return f"-{assumption}"


def arr(X, Y):
    return f"arr_{X}_{Y}"


def noe(X, Y):
    # no-edge is symmetric
    if X > Y:
        X, Y = Y, X
    return f"noe_{X}_{Y}"


def indep(X, Y, S):
    # is symmetric with respect to X and Y
    if X > Y:
        X, Y = Y, X
    S = sorted(list(S))
    return f"indep_{X}_{Y}__" + '_'.join([str(i) for i in S])


def blocked_path(source, target, path_id: int, S: set):
    S = sorted(list(S))
    # is symmetric with respect to source and target
    if source > target:
        source, target = target, source
    return f"blocked_path_{source}_{target}__{path_id}__" + '_'.join([str(i) for i in S])


# Assumptions below this point are not present in the original Causal ABAF.
# However they are used in other experiments.

def dep(X, Y, S):
    # is symmetric with respect to X and Y
    if X > Y:
        X, Y = Y, X
    S = sorted(list(S))
    return f"dep_{X}_{Y}__" + '_'.join([str(i) for i in S])


def active_path(source, target, path_id: int, S: set):
    S = sorted(list(S))
    # is symmetric with respect to source and target
    if source > target:
        source, target = target, source
    return f"active_path_{source}_{target}__{path_id}__" + '_'.join([str(i) for i in S])
