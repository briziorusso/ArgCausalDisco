% source dump (pre-ground) from causalaba_increm.CausalABA
% note: this is NOT a grounded observer dump
% note: facts are intentionally omitted (base-only dump).
%       Load facts separately (e.g. add your facts .lp on the clingo command line).
%%% sets and elements
var(0..7).
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
in(0,s0).
in(1,s1).
in(2,s2).
in(3,s3).
in(4,s4).
in(5,s5).
in(6,s6).
in(7,s7).
in(0,s0y1).
in(1,s0y1).
in(0,s0y2).
in(2,s0y2).
in(0,s0y3).
in(3,s0y3).
in(0,s0y4).
in(4,s0y4).
in(0,s0y5).
in(5,s0y5).
in(0,s0y6).
in(6,s0y6).
in(0,s0y7).
in(7,s0y7).
in(1,s1y2).
in(2,s1y2).
in(1,s1y3).
in(3,s1y3).
in(1,s1y4).
in(4,s1y4).
in(1,s1y5).
in(5,s1y5).
in(1,s1y6).
in(6,s1y6).
in(1,s1y7).
in(7,s1y7).
in(2,s2y3).
in(3,s2y3).
in(2,s2y4).
in(4,s2y4).
in(2,s2y5).
in(5,s2y5).
in(2,s2y6).
in(6,s2y6).
in(2,s2y7).
in(7,s2y7).
in(3,s3y4).
in(4,s3y4).
in(3,s3y5).
in(5,s3y5).
in(3,s3y6).
in(6,s3y6).
in(3,s3y7).
in(7,s3y7).
in(4,s4y5).
in(5,s4y5).
in(4,s4y6).
in(6,s4y6).
in(4,s4y7).
in(7,s4y7).
in(5,s5y6).
in(6,s5y6).
in(5,s5y7).
in(7,s5y7).
in(6,s6y7).
in(7,s6y7).
qpair(0,1,s6).
qpair(0,1,s2).
qpair(0,1,s6y7).
qpair(0,1,s7).
qpair(0,1,empty).
qpair(0,1,s3).
qpair(0,2,s1).
qpair(0,2,s7).
qpair(0,2,empty).
qpair(1,6,s0y7).
qpair(1,6,s7).
qpair(1,6,s0).
qpair(1,6,empty).
qpair(1,6,s3).
qpair(1,7,s6).
qpair(1,7,s2).
qpair(1,7,s0y6).
qpair(1,7,s2y3).
qpair(1,7,s0).
qpair(1,7,empty).
qpair(1,7,s3).
qpair(2,3,s0y7).
qpair(2,3,s7).
qpair(2,3,s6y7).
qpair(2,7,s1y3).
qpair(2,7,s1).
qpair(2,7,s0).
qpair(2,7,empty).
qpair(2,7,s3).
qpair(3,6,s1).
qpair(3,6,s7).
qpair(3,6,empty).
qpair(3,7,s6).
qpair(3,7,s1y2).
qpair(3,7,s2).
qpair(3,7,s1).
qpair(3,7,empty).
qpair(4,5,empty).
qpair(0,3,empty).
qpair(0,4,empty).
qpair(0,5,empty).
qpair(0,6,s1y3).
qpair(0,6,s1y2).
qpair(0,6,s2).
qpair(0,6,s1).
qpair(0,6,s3).
qpair(1,4,empty).
qpair(1,5,empty).
qpair(2,3,s0).
qpair(2,3,s6).
qpair(2,4,empty).
qpair(2,5,empty).
qpair(2,6,empty).
qpair(3,4,empty).
qpair(3,5,empty).
qpair(4,6,empty).
qpair(4,7,empty).
qpair(5,6,empty).
qpair(5,7,empty).

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

var(0).
var(1).
var(2).
var(3).
var(4).
var(5).
var(6).
var(7).

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep/wc_sweep_8_2026_er_sparse_alpha001_noweight_seed2026_2075_20260715_175946/rep3/lps/abapc_inc_full_base_rep3_optN.lp =====
% ===== Base program sha256 (raw): 1bfe52dff25de0e872971a9feb2750e781aafe7f2595b9477baa5a1dab2a9a12 =====
% ===== Base program sha256 (normalized): 8919c54452709104a288f5b4e159a31c59a75861593b663fa50db3b0b1a771a7 =====
% ===== Assumption Layer for MUS =====

% ===== Explicit var/1 facts =====
var(0).
var(1).
var(2).
var(3).
var(4).
var(5).
var(6).
var(7).
{mus(1)}.
{mus(2)}.
{mus(3)}.
{mus(4)}.
{mus(5)}.
{mus(6)}.
{mus(7)}.
{mus(8)}.
{mus(9)}.
{mus(10)}.
{mus(11)}.
{mus(12)}.
{mus(13)}.
{mus(14)}.
{mus(15)}.
{mus(16)}.
{mus(17)}.
{mus(18)}.
{mus(19)}.
{mus(20)}.
{mus(21)}.
{mus(22)}.
{mus(23)}.
{mus(24)}.
{mus(25)}.
{mus(26)}.
{mus(27)}.
{mus(28)}.
{mus(29)}.
{mus(30)}.
{mus(31)}.
{mus(32)}.
{mus(33)}.
{mus(34)}.
{mus(35)}.
{mus(36)}.
{mus(37)}.
{mus(38)}.
{mus(39)}.
{mus(40)}.
{mus(41)}.
{mus(42)}.
{mus(43)}.
{mus(44)}.
{mus(45)}.
{mus(46)}.
{mus(47)}.
{mus(48)}.
{mus(49)}.
{mus(50)}.
{mus(51)}.
{mus(52)}.
{mus(53)}.
{mus(54)}.
{mus(55)}.
{mus(56)}.
{mus(57)}.
{mus(58)}.
{mus(59)}.

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s2):-mus(2).
ext_dep(0,1,s3):-mus(3).
ext_dep(0,1,s6):-mus(4).
ext_dep(0,1,s7):-mus(5).
ext_dep(0,1,s6y7):-mus(6).
ext_dep(0,2,empty):-mus(7).
ext_dep(0,2,s1):-mus(8).
ext_dep(0,2,s7):-mus(9).
ext_indep(0,3,empty):-mus(10).
ext_indep(0,4,empty):-mus(11).
ext_indep(0,5,empty):-mus(12).
ext_indep(0,6,s1):-mus(13).
ext_indep(0,6,s2):-mus(14).
ext_indep(0,6,s3):-mus(15).
ext_indep(0,6,s1y2):-mus(16).
ext_indep(0,6,s1y3):-mus(17).
ext_indep(1,4,empty):-mus(18).
ext_indep(1,5,empty):-mus(19).
ext_dep(1,6,empty):-mus(20).
ext_dep(1,6,s0):-mus(21).
ext_dep(1,6,s3):-mus(22).
ext_dep(1,6,s7):-mus(23).
ext_dep(1,6,s0y7):-mus(24).
ext_dep(1,7,empty):-mus(25).
ext_dep(1,7,s0):-mus(26).
ext_dep(1,7,s2):-mus(27).
ext_dep(1,7,s3):-mus(28).
ext_dep(1,7,s6):-mus(29).
ext_dep(1,7,s0y6):-mus(30).
ext_dep(1,7,s2y3):-mus(31).
ext_indep(2,3,s0):-mus(32).
ext_indep(2,3,s6):-mus(33).
ext_dep(2,3,s7):-mus(34).
ext_dep(2,3,s0y7):-mus(35).
ext_dep(2,3,s6y7):-mus(36).
ext_indep(2,4,empty):-mus(37).
ext_indep(2,5,empty):-mus(38).
ext_indep(2,6,empty):-mus(39).
ext_dep(2,7,empty):-mus(40).
ext_dep(2,7,s0):-mus(41).
ext_dep(2,7,s1):-mus(42).
ext_dep(2,7,s3):-mus(43).
ext_dep(2,7,s1y3):-mus(44).
ext_indep(3,4,empty):-mus(45).
ext_indep(3,5,empty):-mus(46).
ext_dep(3,6,empty):-mus(47).
ext_dep(3,6,s1):-mus(48).
ext_dep(3,6,s7):-mus(49).
ext_dep(3,7,empty):-mus(50).
ext_dep(3,7,s1):-mus(51).
ext_dep(3,7,s2):-mus(52).
ext_dep(3,7,s6):-mus(53).
ext_dep(3,7,s1y2):-mus(54).
ext_dep(4,5,empty):-mus(55).
ext_indep(4,6,empty):-mus(56).
ext_indep(4,7,empty):-mus(57).
ext_indep(5,6,empty):-mus(58).
ext_indep(5,7,empty):-mus(59).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(9999999, mus(1)).
__optimum_mcs_objective_literal__(9999999, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(9999999, mus(4)).
__optimum_mcs_objective_literal__(9999999, mus(5)).
__optimum_mcs_objective_literal__(9999999, mus(6)).
__optimum_mcs_objective_literal__(9999999, mus(7)).
__optimum_mcs_objective_literal__(9999999, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9959650, mus(10)).
__optimum_mcs_objective_literal__(6187413, mus(11)).
__optimum_mcs_objective_literal__(5308867, mus(12)).
__optimum_mcs_objective_literal__(5580440, mus(13)).
__optimum_mcs_objective_literal__(7665161, mus(14)).
__optimum_mcs_objective_literal__(7876205, mus(15)).
__optimum_mcs_objective_literal__(6202865, mus(16)).
__optimum_mcs_objective_literal__(5188975, mus(17)).
__optimum_mcs_objective_literal__(6743975, mus(18)).
__optimum_mcs_objective_literal__(7184666, mus(19)).
__optimum_mcs_objective_literal__(9999999, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(9999999, mus(22)).
__optimum_mcs_objective_literal__(9999999, mus(23)).
__optimum_mcs_objective_literal__(9999999, mus(24)).
__optimum_mcs_objective_literal__(9979393, mus(25)).
__optimum_mcs_objective_literal__(9999999, mus(26)).
__optimum_mcs_objective_literal__(9997284, mus(27)).
__optimum_mcs_objective_literal__(9693611, mus(28)).
__optimum_mcs_objective_literal__(9999307, mus(29)).
__optimum_mcs_objective_literal__(9999999, mus(30)).
__optimum_mcs_objective_literal__(9308559, mus(31)).
__optimum_mcs_objective_literal__(9336955, mus(32)).
__optimum_mcs_objective_literal__(9196614, mus(33)).
__optimum_mcs_objective_literal__(9999999, mus(34)).
__optimum_mcs_objective_literal__(9999999, mus(35)).
__optimum_mcs_objective_literal__(9999999, mus(36)).
__optimum_mcs_objective_literal__(7550090, mus(37)).
__optimum_mcs_objective_literal__(8293276, mus(38)).
__optimum_mcs_objective_literal__(6813522, mus(39)).
__optimum_mcs_objective_literal__(9999999, mus(40)).
__optimum_mcs_objective_literal__(9999999, mus(41)).
__optimum_mcs_objective_literal__(9999999, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(6571216, mus(45)).
__optimum_mcs_objective_literal__(8177061, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9999999, mus(48)).
__optimum_mcs_objective_literal__(9999999, mus(49)).
__optimum_mcs_objective_literal__(9999999, mus(50)).
__optimum_mcs_objective_literal__(9999999, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(9999999, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9999999, mus(55)).
__optimum_mcs_objective_literal__(5301472, mus(56)).
__optimum_mcs_objective_literal__(7375760, mus(57)).
__optimum_mcs_objective_literal__(8636543, mus(58)).
__optimum_mcs_objective_literal__(6895680, mus(59)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
