from clingo.control import Control
import networkx as nx
from itertools import chain, combinations
from datetime import datetime

num_of_nodes = 3
verbose = False

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
ctl = Control(["0"])
### Load encodings
ctl.load("encodings/causalaba.lp")
### Add vars
ctl.add("base", [], f"var(0..{num_of_nodes-1}).")
if verbose:
    print(f"var(0..{num_of_nodes-1}).")
### add indep rules
for (X,Y) in combinations(range(num_of_nodes),2):
    G = nx.complete_graph(num_of_nodes)
    paths = nx.all_simple_paths(G, source=X, target=Y)
    indep_rule_body = []
    for path in paths:
        indep_rule_body.append(f" not ap({','.join([str(p) for p in path])},S)")
    indep_rule = f"indep({X},{Y},S) :-" + ','.join(indep_rule_body) + f", not in({X},S), not in({Y},S), set(S)."
    ctl.add("base", [], indep_rule)
    if verbose:
        print(indep_rule)
### add nonblocker rules
for Z in range(num_of_nodes):
    ctl.add("base", [], f"nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in({Z},S), collider_desc({Z},N,X,Y).")
    print(f"nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in({Z},S), collider_desc({Z},N,X,Y).")
### Active and blocked paths rules
bps = []
for path_len in range(2,num_of_nodes+1):
    if path_len>2:
        Xs = [f"X{idx}" for idx in range(path_len)]
        vars = [f"var({X})" for X in Xs]
        edges = [f"edge(X{idx},X{idx+1})" for idx in range(len(Xs)-1)]
        nbs = [f"nb(X{idx},X{idx-1},X{idx+1},S)" for idx in range(1,len(Xs)-1)]
        uneq_const = []
        for comb in combinations(Xs, 2):
            uneq_const.append(f"{comb[0]}!={comb[1]}")
        if len(nbs)>0:
            body = f"{','.join(edges)}, {','.join(vars)}, {','.join(uneq_const)}, {','.join(nbs)}"
        else:
            body = f"{','.join(edges)}, {','.join(vars)}, {','.join(uneq_const)}"
        ### Active paths
        ctl.add("base", [], f"ap({','.join(Xs)},S) :- {body}, not in({Xs[0]},S), not in({Xs[-1]},S), set(S).")
        if verbose:
            print(f"ap({','.join(Xs)},S) :- {body}, not in({Xs[0]},S), not in({Xs[-1]},S), set(S).") 
for S in powerset(range(num_of_nodes)):
    for s in S:
        ctl.add("base", [], f"in({s},{'s'+''.join([str(i) for i in S])}).")

## Add Facts
ctl.add("base", [], """\
%indep(0,1,empty).
%dep(0,1,2).
%arrow(0,1).
%arrow(1,2).
%arrow(2,3).
%arrow(0,2).
%:- edge(0,3).
%:- edge(1,3).
""")

### Ground & Solve
ctl.ground([("base", [])])
# print(ctl.solve(on_model=print))

start = datetime.now()
ctl.solve(on_model=lambda m: print(f"Answer: {m}"))
end = datetime.now()
print("RunTime:", end-start)
print("Number of models:", int(ctl.statistics['summary']['models']['enumerated']))
