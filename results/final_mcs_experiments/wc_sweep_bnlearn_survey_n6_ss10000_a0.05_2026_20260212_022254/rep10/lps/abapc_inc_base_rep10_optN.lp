% source dump (pre-ground) from causalaba_increm.CausalABA
% note: this is NOT a grounded observer dump
% note: facts are intentionally omitted (base-only dump).
%       Load facts separately (e.g. add your facts .lp on the clingo command line).
% note: flattened one-shot dump (no #program parts).
%%% sets and elements
var(0..n_vars).
set(X) :- in(N,X).
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
dpath(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
dpath(X,Y):- arrow(X,Z), dpath(Z,Y), var(X). 
:- dpath(X,X), var(X).

%%% Y is a collider between X and Z 
collider(Y,X,Z) :- arrow(X,Y), arrow(Z,Y), X!=Z, X!=Y, Y!=Z, var(X), var(Y), var(Z).
%%% Y is a collider between X and Z and there is a directed path from Y to N
collider_desc(N,Y,X,Z) :- collider(Y,X,Z), dpath(Y,N), var(N), X!=N, Y!=N, Z!=N.

%%% Nonblocker rules
nb(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc(Z,N,X,Y), var(Z).

% generated facts (in/2, qpair/3, etc)
in(2,s2y4).
in(4,s2y4).
in(2,s2).
in(0,s0y4).
in(4,s0y4).
in(3,s3y4).
in(4,s3y4).
in(5,s5).
in(1,s1y5).
in(5,s1y5).
in(0,s0y2y3).
in(2,s0y2y3).
in(3,s0y2y3).
in(0,s0y2).
in(2,s0y2).
in(0,s0).
in(3,s3).
in(0,s0y2y4).
in(2,s0y2y4).
in(4,s0y2y4).
in(0,s0y3y4).
in(3,s0y3y4).
in(4,s0y3y4).
in(0,s0y3).
in(3,s0y3).
in(4,s4).
in(2,s2y3).
in(3,s2y3).
in(1,s1).
in(0,s0y2y3y4).
in(2,s0y2y3y4).
in(3,s0y2y3y4).
in(4,s0y2y3y4).
in(2,s2y3y4).
in(3,s2y3y4).
in(4,s2y3y4).
qpair(0,1,s2y4).
qpair(0,1,s2).
qpair(0,1,s3y4).
qpair(0,1,s5).
qpair(0,1,s4).
qpair(0,1,s2y3).
qpair(0,1,s2y3y4).
qpair(0,1,empty).
qpair(0,1,s3).
qpair(0,4,s1).
qpair(0,4,s1y5).
qpair(1,2,s0y4).
qpair(1,2,s3y4).
qpair(1,2,s5).
qpair(1,2,s0y3y4).
qpair(1,2,s0y3).
qpair(1,2,s4).
qpair(1,2,s0).
qpair(1,2,empty).
qpair(1,2,s3).
qpair(1,3,s2y4).
qpair(1,3,s2).
qpair(1,3,s0y4).
qpair(1,3,s0y2y4).
qpair(1,3,s5).
qpair(1,3,s4).
qpair(1,3,s0y2).
qpair(1,3,s0).
qpair(1,3,empty).
qpair(1,4,s2).
qpair(1,4,s5).
qpair(1,4,s0y3).
qpair(1,4,s0y2y3).
qpair(1,4,s2y3).
qpair(1,4,s0y2).
qpair(1,4,s0).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,5,empty).
qpair(2,3,s5).
qpair(2,3,s1y5).
qpair(2,5,s3y4).
qpair(2,5,s4).
qpair(2,5,s1).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(3,5,s2y4).
qpair(3,5,s2).
qpair(3,5,s4).
qpair(3,5,s1).
qpair(3,5,empty).
qpair(4,5,s2).
qpair(4,5,s2y3).
qpair(4,5,s1).
qpair(4,5,empty).
qpair(4,5,s3).
qpair(0,2,s1).
qpair(0,2,empty).
qpair(0,2,s5).
qpair(0,2,s1y5).
qpair(0,3,s1).
qpair(0,3,empty).
qpair(0,3,s5).
qpair(0,3,s1y5).
qpair(0,4,empty).
qpair(0,4,s5).
qpair(0,5,empty).
qpair(1,5,s2y4).
qpair(1,5,s2).
qpair(1,5,s0y4).
qpair(1,5,s0y2y4).
qpair(1,5,s3y4).
qpair(1,5,s0y3y4).
qpair(1,5,s0y3).
qpair(1,5,s0y2y3).
qpair(1,5,s4).
qpair(1,5,s2y3).
qpair(1,5,s0y2y3y4).
qpair(1,5,s0y2).
qpair(1,5,s2y3y4).
qpair(1,5,s0).
qpair(1,5,s3).
qpair(2,3,s1).
qpair(2,3,empty).
qpair(2,4,s1).
qpair(2,4,empty).
qpair(2,4,s5).
qpair(2,4,s1y5).
qpair(3,4,s1).
qpair(3,4,empty).
qpair(3,4,s5).
qpair(3,4,s1y5).

indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.
:- dep(X,Y,S), not ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).
:- indep(X,Y,S), ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).
% Query triples to evaluate (restrict to one orientation to match constraints).

bb_start(X,S) :- qpair(X,_,S), set(S), var(X).

% Ancestor-of-observed (incl. observed itself): used to open colliders.
anc_obs(N,S) :- in(N,S), set(S), var(N).
anc_obs(N,S) :- in(Z,S), dpath(N,Z), set(S), var(N), var(Z), N!=Z.

is_collider(N) :- arrow(P,N), arrow(Q,N), P<Q, P!=Q, var(P), var(Q), var(N).

% Bayes-ball states: bb_u(start,node,set) = reached node with ball arriving from a child.
%                  bb_d(start,node,set) = reached node with ball arriving from a parent.
bb_u(X,X,S) :- bb_start(X,S).
bb_d(X,X,S) :- bb_start(X,S).

% Move to parents.
% - If the ball arrives from a child (bb_u), only propagate further if the
%   current node is NOT observed.
% - If the ball arrives from a parent (bb_d), propagate to parents iff the
%   current node is an ancestor of an observed node (incl. observed itself).
bb_u(X,P,S) :- bb_u(X,N,S), not in(N,S), arrow(P,N), var(P), var(N), set(S).
bb_u(X,P,S) :- bb_d(X,N,S), is_collider(N), anc_obs(N,S), arrow(P,N), var(P), var(N), set(S).

% Move to children (only if current node is not observed).
bb_d(X,C,S) :- bb_u(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).
bb_d(X,C,S) :- bb_d(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).

% Active connection exists between X and Y given S iff Bayes-ball can reach Y.
ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_u(X,Y,S).
ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_d(X,Y,S).

% Backwards-compatible alias for callers/tests that expect ap/4 atoms.
ap(X,Y,bb,S) :- ap3(X,Y,S), qpair(X,Y,S).
:- block_edge(X,Y), edge(X,Y), X<Y, var(X), var(Y).
% materialized external assignments (block_edge/2)
block_edge(0,2).
block_edge(0,3).
block_edge(0,4).
block_edge(0,5).
block_edge(1,5).
block_edge(2,3).
block_edge(2,4).
block_edge(3,4).

var(0).
var(1).
var(2).
var(3).
var(4).
var(5).
#show arrow/2.
