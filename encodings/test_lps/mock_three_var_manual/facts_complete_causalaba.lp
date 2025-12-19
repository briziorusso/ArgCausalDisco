%%% COMPLETE CAUSALABA ENCODING FOR 3-NODE EXAMPLE
%%% This file contains all the facts and rules that would be generated
%%% by causalaba.py for the mock_three_var test case

%%% Facts
ext_indep(1,2,s0).
ext_dep(1,2,empty).
ext_indep(0,1,empty).

%%% Main encoding
var(0). var(1). var(2).
set(empty).
set(s0).
in(0,s0).

%%% Convert ext_* to regular predicates
indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.

%%% Symmetry
indep(X,Y,S) :- indep(Y,X,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- dep(Y,X,S), var(X), var(Y), set(S), X!=Y.
:- dep(X,Y,S), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).

%%% Generate arrows (DAG structure)
arrow(X,Y) | arrow(Y,X) | not edge(X,Y) :- var(X), var(Y), X!=Y.
edge(X,Y) :- arrow(X,Y), X!=Y, var(X), var(Y).
edge(X,Y) :- edge(Y,X), X!=Y, var(X), var(Y).

%%% Test constraints
:- edge(X,Y), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).
:- arrow(X,Y), not edge(X,Y), var(X), var(Y).
:- edge(X,Y), not arrow(X,Y), not arrow(Y,X), var(X), var(Y).
:- arrow(Y,X), arrow(X,Y), X!=Y, var(X), var(Y).

%%% Acyclicity
dpath(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
dpath(X,Y):- arrow(X,Z), dpath(Z,Y), var(X). 
:- dpath(X,X), var(X).

%%% Colliders
collider(Y,X,Z) :- arrow(X,Y), arrow(Z,Y), X!=Z, X!=Y, Y!=Z, var(X), var(Y), var(Z).
collider_desc(N,Y,X,Z) :- collider(Y,X,Z), dpath(Y,N), var(N), X!=N, Y!=N, Z!=N.

%%% Nonblocker rules
nb(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc(Z,N,X,Y), var(Z).

%%% GENERATED ACTIVE PATH RULES
%%% For complete graph on {0,1,2}, all simple paths:

%%% Paths from 0 to 1
p1 :- edge(0,1).
ap(0,1,p1,S) :- p1, not in(0,S), not in(1,S), set(S).

%%% Paths from 0 to 2
p2 :- edge(0,2).
ap(0,2,p2,S) :- p2, not in(0,S), not in(2,S), set(S).

p3 :- edge(0,1), edge(1,2).
ap(0,2,p3,S) :- p3, nb(1,0,2,S), not in(0,S), not in(2,S), set(S).

%%% Paths from 1 to 0
p4 :- edge(1,0).
ap(1,0,p4,S) :- p4, not in(1,S), not in(0,S), set(S).

%%% Paths from 1 to 2
p5 :- edge(1,2).
ap(1,2,p5,S) :- p5, not in(1,S), not in(2,S), set(S).

p6 :- edge(1,0), edge(0,2).
ap(1,2,p6,S) :- p6, nb(0,1,2,S), not in(1,S), not in(2,S), set(S).

%%% Paths from 2 to 0
p7 :- edge(2,0).
ap(2,0,p7,S) :- p7, not in(2,S), not in(0,S), set(S).

p8 :- edge(2,1), edge(1,0).
ap(2,0,p8,S) :- p8, nb(1,2,0,S), not in(2,S), not in(0,S), set(S).

%%% Paths from 2 to 1
p9 :- edge(2,1).
ap(2,1,p9,S) :- p9, not in(2,S), not in(1,S), set(S).

p10 :- edge(2,0), edge(0,1).
ap(2,1,p10,S) :- p10, nb(0,2,1,S), not in(2,S), not in(1,S), set(S).

%%% Active path implies dependence (when there's an independence claim)
dep(0,1,S) :- ap(0,1,_,S), set(S).
dep(0,2,S) :- ap(0,2,_,S), set(S).
dep(1,2,S) :- ap(1,2,_,S), set(S).

%%% No active path implies independence (when there's a dependence claim)
indep(1,2,empty) :- not ap(1,2,_,empty).

%%% Show
#show arrow/2.
