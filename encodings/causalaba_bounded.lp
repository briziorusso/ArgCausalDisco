#program main(n_vars).
% Boundedness constants (override with -c l_cyc=.., -c l_b=..)
#const l_cyc=0.
#const l_b=0.
#const l_max=l_cyc+l_b.

%%% sets and elements
var(0..n_vars).
set(X) :- in(_,X).
set(empty).

%%% dependency and independency
%%% symmetry
indep(X,Y,S) :- indep(Y,X,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- dep(Y,X,S), var(X), var(Y), set(S), X!=Y.
%%% test
:- dep(X,Y,S), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).

%%%%% All DAGs
%%% generate arrows
arrow(X,Y) | arrow(Y,X) | not edge(X,Y) :- var(X), var(Y), X!=Y.
edge(X,Y) :- arrow(X,Y), X!=Y, var(X), var(Y).
%%% symmetry
edge(X,Y) :- edge(Y,X), X!=Y, var(X), var(Y).
%%% test
:- edge(X,Y), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).
:- arrow(X,Y), not edge(X,Y), var(X), var(Y).
:- edge(X,Y), not arrow(X,Y), not arrow(Y,X), var(X), var(Y).
:- arrow(Y,X), arrow(X,Y), X!=Y, var(X), var(Y).

%%% acyclicity
% Unbounded transitive closure
dpath(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
dpath(X,Y):- arrow(X,Z), dpath(Z,Y), var(X).

% Bounded path of exact length K up to l_max
dpath_k(X,Y,1) :- arrow(X,Y), X!=Y, var(X), var(Y).
dpath_k(X,Y,K) :- arrow(X,Z), dpath_k(Z,Y,K-1), var(X), var(Y), var(Z), K=2..l_max.

% Acyclicity constraint: if l_cyc>0, only consider cycles up to l_cyc; otherwise use unbounded
:- dpath_k(X,X,K), var(X), K=1..l_cyc, l_cyc > 0.
:- dpath(X,X), var(X), l_cyc = 0.

%%% Y is a collider between X and Z 
collider(Y,X,Z) :- arrow(X,Y), arrow(Z,Y), X!=Z, X!=Y, Y!=Z, var(X), var(Y), var(Z).
%%% Y is a collider between X and Z and there is a directed path from Y to N
collider_desc(N,Y,X,Z) :- collider(Y,X,Z), dpath(Y,N), var(N), X!=N, Y!=N, Z!=N.

% Bounded collider descendant using bounded path up to l_b
collider_desc_b(N,Y,X,Z) :- collider(Y,X,Z), dpath_k(Y,N,K), var(N), X!=N, Y!=N, Z!=N, K=1..l_b.

%%% Nonblocker rules
nb(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc(Z,N,X,Y), var(Z).

% Bounded nonblocker variant that uses collider_desc_b for the descendant case
nb_b(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb_b(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb_b(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc_b(Z,N,X,Y), var(Z).
