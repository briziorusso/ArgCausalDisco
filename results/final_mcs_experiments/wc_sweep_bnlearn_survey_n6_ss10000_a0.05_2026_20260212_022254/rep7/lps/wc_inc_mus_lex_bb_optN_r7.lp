% source dump (pre-ground) from causalaba_increm.CausalABA
% note: this is NOT a grounded observer dump
% note: facts are intentionally omitted (base-only dump).
%       Load facts separately (e.g. add your facts .lp on the clingo command line).
%%% sets and elements
var(0..5).
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
in(1,s1y2).
in(2,s1y2).
in(1,s1y3).
in(3,s1y3).
in(1,s1y4).
in(4,s1y4).
in(1,s1y5).
in(5,s1y5).
in(2,s2y3).
in(3,s2y3).
in(2,s2y4).
in(4,s2y4).
in(2,s2y5).
in(5,s2y5).
in(3,s3y4).
in(4,s3y4).
in(3,s3y5).
in(5,s3y5).
in(4,s4y5).
in(5,s4y5).
in(0,s0y1y2).
in(1,s0y1y2).
in(2,s0y1y2).
in(0,s0y1y3).
in(1,s0y1y3).
in(3,s0y1y3).
in(0,s0y1y4).
in(1,s0y1y4).
in(4,s0y1y4).
in(0,s0y1y5).
in(1,s0y1y5).
in(5,s0y1y5).
in(0,s0y2y3).
in(2,s0y2y3).
in(3,s0y2y3).
in(0,s0y2y4).
in(2,s0y2y4).
in(4,s0y2y4).
in(0,s0y2y5).
in(2,s0y2y5).
in(5,s0y2y5).
in(0,s0y3y4).
in(3,s0y3y4).
in(4,s0y3y4).
in(0,s0y3y5).
in(3,s0y3y5).
in(5,s0y3y5).
in(0,s0y4y5).
in(4,s0y4y5).
in(5,s0y4y5).
in(1,s1y2y3).
in(2,s1y2y3).
in(3,s1y2y3).
in(1,s1y2y4).
in(2,s1y2y4).
in(4,s1y2y4).
in(1,s1y2y5).
in(2,s1y2y5).
in(5,s1y2y5).
in(1,s1y3y4).
in(3,s1y3y4).
in(4,s1y3y4).
in(1,s1y3y5).
in(3,s1y3y5).
in(5,s1y3y5).
in(1,s1y4y5).
in(4,s1y4y5).
in(5,s1y4y5).
in(2,s2y3y4).
in(3,s2y3y4).
in(4,s2y3y4).
in(2,s2y3y5).
in(3,s2y3y5).
in(5,s2y3y5).
in(2,s2y4y5).
in(4,s2y4y5).
in(5,s2y4y5).
in(3,s3y4y5).
in(4,s3y4y5).
in(5,s3y4y5).
in(0,s0y1y2y3).
in(1,s0y1y2y3).
in(2,s0y1y2y3).
in(3,s0y1y2y3).
in(0,s0y1y2y4).
in(1,s0y1y2y4).
in(2,s0y1y2y4).
in(4,s0y1y2y4).
in(0,s0y1y2y5).
in(1,s0y1y2y5).
in(2,s0y1y2y5).
in(5,s0y1y2y5).
in(0,s0y1y3y4).
in(1,s0y1y3y4).
in(3,s0y1y3y4).
in(4,s0y1y3y4).
in(0,s0y1y3y5).
in(1,s0y1y3y5).
in(3,s0y1y3y5).
in(5,s0y1y3y5).
in(0,s0y1y4y5).
in(1,s0y1y4y5).
in(4,s0y1y4y5).
in(5,s0y1y4y5).
in(0,s0y2y3y4).
in(2,s0y2y3y4).
in(3,s0y2y3y4).
in(4,s0y2y3y4).
in(0,s0y2y3y5).
in(2,s0y2y3y5).
in(3,s0y2y3y5).
in(5,s0y2y3y5).
in(0,s0y2y4y5).
in(2,s0y2y4y5).
in(4,s0y2y4y5).
in(5,s0y2y4y5).
in(0,s0y3y4y5).
in(3,s0y3y4y5).
in(4,s0y3y4y5).
in(5,s0y3y4y5).
in(1,s1y2y3y4).
in(2,s1y2y3y4).
in(3,s1y2y3y4).
in(4,s1y2y3y4).
in(1,s1y2y3y5).
in(2,s1y2y3y5).
in(3,s1y2y3y5).
in(5,s1y2y3y5).
in(1,s1y2y4y5).
in(2,s1y2y4y5).
in(4,s1y2y4y5).
in(5,s1y2y4y5).
in(1,s1y3y4y5).
in(3,s1y3y4y5).
in(4,s1y3y4y5).
in(5,s1y3y4y5).
in(2,s2y3y4y5).
in(3,s2y3y4y5).
in(4,s2y3y4y5).
in(5,s2y3y4y5).
qpair(0,1,s2y4).
qpair(0,1,s2).
qpair(0,1,s3y4).
qpair(0,1,s4).
qpair(0,1,s2y3).
qpair(0,1,s2y3y4).
qpair(0,1,empty).
qpair(0,1,s3).
qpair(0,4,s1).
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
qpair(1,4,s0y3).
qpair(1,4,s0y2y3).
qpair(1,4,s2y3).
qpair(1,4,s0y2).
qpair(1,4,s0).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(2,3,s5).
qpair(2,3,s1y5).
qpair(2,5,s1).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(3,5,s1).
qpair(3,5,empty).
qpair(3,5,s2).
qpair(0,2,s1).
qpair(0,2,empty).
qpair(0,2,s5).
qpair(0,2,s1y5).
qpair(0,3,s1).
qpair(0,3,empty).
qpair(0,3,s5).
qpair(0,3,s1y5).
qpair(0,4,empty).
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
qpair(1,5,empty).
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
qpair(4,5,empty).

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

% ===== Base program loaded from: results/wc_sweep_bnlearn_survey_n6_ss10000_a0.05_2026_20260212_022254/rep7/lps/abapc_inc_full_base_rep7_optN.lp =====
% ===== Base program sha256 (raw): 0e32e9a76c3c619851697de6895d8b497d984f800f1b3aaa435d98d7b6e63661 =====
% ===== Base program sha256 (normalized): 4beac95e1aa5f642dc0477e2d7f6766c3b790b49dd9eba0a734a108dd898a0bd =====
% ===== Assumption Layer for MUS =====

% ===== Explicit var/1 facts =====
var(0).
var(1).
var(2).
var(3).
var(4).
var(5).
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
{mus(60)}.
{mus(61)}.
{mus(62)}.
{mus(63)}.
{mus(64)}.
{mus(65)}.
{mus(66)}.
{mus(67)}.
{mus(68)}.
{mus(69)}.
{mus(70)}.
{mus(71)}.
{mus(72)}.
{mus(73)}.
{mus(74)}.
{mus(75)}.
{mus(76)}.
{mus(77)}.
{mus(78)}.
{mus(79)}.
{mus(80)}.

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s2):-mus(2).
ext_dep(0,1,s3):-mus(3).
ext_dep(0,1,s4):-mus(4).
ext_dep(0,1,s2y3):-mus(5).
ext_dep(0,1,s2y4):-mus(6).
ext_dep(0,1,s3y4):-mus(7).
ext_dep(0,1,s2y3y4):-mus(8).
ext_indep(0,2,empty):-mus(9).
ext_indep(0,2,s1):-mus(10).
ext_indep(0,2,s1y5):-mus(11).
ext_indep(0,2,s5):-mus(12).
ext_indep(0,3,empty):-mus(13).
ext_indep(0,3,s1):-mus(14).
ext_indep(0,3,s1y5):-mus(15).
ext_indep(0,3,s5):-mus(16).
ext_indep(0,4,empty):-mus(17).
ext_dep(0,4,s1):-mus(18).
ext_indep(0,5,empty):-mus(19).
ext_dep(1,2,empty):-mus(20).
ext_dep(1,2,s0):-mus(21).
ext_dep(1,2,s3):-mus(22).
ext_dep(1,2,s4):-mus(23).
ext_dep(1,2,s5):-mus(24).
ext_dep(1,2,s0y3):-mus(25).
ext_dep(1,2,s0y4):-mus(26).
ext_dep(1,2,s3y4):-mus(27).
ext_dep(1,2,s0y3y4):-mus(28).
ext_dep(1,3,empty):-mus(29).
ext_dep(1,3,s0):-mus(30).
ext_dep(1,3,s2):-mus(31).
ext_dep(1,3,s4):-mus(32).
ext_dep(1,3,s5):-mus(33).
ext_dep(1,3,s0y2):-mus(34).
ext_dep(1,3,s0y4):-mus(35).
ext_dep(1,3,s2y4):-mus(36).
ext_dep(1,3,s0y2y4):-mus(37).
ext_dep(1,4,empty):-mus(38).
ext_dep(1,4,s0):-mus(39).
ext_dep(1,4,s2):-mus(40).
ext_dep(1,4,s3):-mus(41).
ext_dep(1,4,s0y2):-mus(42).
ext_dep(1,4,s0y3):-mus(43).
ext_dep(1,4,s2y3):-mus(44).
ext_dep(1,4,s0y2y3):-mus(45).
ext_indep(1,5,empty):-mus(46).
ext_indep(1,5,s2y4):-mus(47).
ext_indep(1,5,s2):-mus(48).
ext_indep(1,5,s0y2y4):-mus(49).
ext_indep(1,5,s0y2y3):-mus(50).
ext_indep(1,5,s2y3):-mus(51).
ext_indep(1,5,s0y2y3y4):-mus(52).
ext_indep(1,5,s0y2):-mus(53).
ext_indep(1,5,s2y3y4):-mus(54).
ext_indep(1,5,s0y4):-mus(55).
ext_indep(1,5,s3y4):-mus(56).
ext_indep(1,5,s0y3y4):-mus(57).
ext_indep(1,5,s0y3):-mus(58).
ext_indep(1,5,s4):-mus(59).
ext_indep(1,5,s0):-mus(60).
ext_indep(1,5,s3):-mus(61).
ext_indep(2,3,empty):-mus(62).
ext_indep(2,3,s1):-mus(63).
ext_dep(2,3,s1y5):-mus(64).
ext_dep(2,3,s5):-mus(65).
ext_indep(2,4,empty):-mus(66).
ext_indep(2,4,s1):-mus(67).
ext_indep(2,4,s1y5):-mus(68).
ext_indep(2,4,s5):-mus(69).
ext_dep(2,5,empty):-mus(70).
ext_dep(2,5,s1):-mus(71).
ext_dep(2,5,s3):-mus(72).
ext_indep(3,4,empty):-mus(73).
ext_indep(3,4,s1):-mus(74).
ext_indep(3,4,s1y5):-mus(75).
ext_indep(3,4,s5):-mus(76).
ext_dep(3,5,empty):-mus(77).
ext_dep(3,5,s1):-mus(78).
ext_dep(3,5,s2):-mus(79).
ext_indep(4,5,empty):-mus(80).

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
__optimum_mcs_objective_literal__(5326266, mus(9)).
__optimum_mcs_objective_literal__(8740863, mus(10)).
__optimum_mcs_objective_literal__(6645186, mus(11)).
__optimum_mcs_objective_literal__(5290205, mus(12)).
__optimum_mcs_objective_literal__(5220160, mus(13)).
__optimum_mcs_objective_literal__(6888011, mus(14)).
__optimum_mcs_objective_literal__(6918192, mus(15)).
__optimum_mcs_objective_literal__(5614552, mus(16)).
__optimum_mcs_objective_literal__(7277864, mus(17)).
__optimum_mcs_objective_literal__(9999996, mus(18)).
__optimum_mcs_objective_literal__(6891420, mus(19)).
__optimum_mcs_objective_literal__(9999999, mus(20)).
__optimum_mcs_objective_literal__(9999998, mus(21)).
__optimum_mcs_objective_literal__(9999999, mus(22)).
__optimum_mcs_objective_literal__(9999999, mus(23)).
__optimum_mcs_objective_literal__(9999999, mus(24)).
__optimum_mcs_objective_literal__(9999970, mus(25)).
__optimum_mcs_objective_literal__(9999979, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(9999381, mus(28)).
__optimum_mcs_objective_literal__(9993172, mus(29)).
__optimum_mcs_objective_literal__(9823596, mus(30)).
__optimum_mcs_objective_literal__(9976975, mus(31)).
__optimum_mcs_objective_literal__(9975974, mus(32)).
__optimum_mcs_objective_literal__(9935879, mus(33)).
__optimum_mcs_objective_literal__(8805984, mus(34)).
__optimum_mcs_objective_literal__(9244749, mus(35)).
__optimum_mcs_objective_literal__(9850281, mus(36)).
__optimum_mcs_objective_literal__(6301425, mus(37)).
__optimum_mcs_objective_literal__(9984234, mus(38)).
__optimum_mcs_objective_literal__(9999999, mus(39)).
__optimum_mcs_objective_literal__(9978210, mus(40)).
__optimum_mcs_objective_literal__(9947074, mus(41)).
__optimum_mcs_objective_literal__(9999999, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9857511, mus(44)).
__optimum_mcs_objective_literal__(9999998, mus(45)).
__optimum_mcs_objective_literal__(8287095, mus(46)).
__optimum_mcs_objective_literal__(7123454, mus(47)).
__optimum_mcs_objective_literal__(8535289, mus(48)).
__optimum_mcs_objective_literal__(8559268, mus(49)).
__optimum_mcs_objective_literal__(7074092, mus(50)).
__optimum_mcs_objective_literal__(8808183, mus(51)).
__optimum_mcs_objective_literal__(8701178, mus(52)).
__optimum_mcs_objective_literal__(8517226, mus(53)).
__optimum_mcs_objective_literal__(8475726, mus(54)).
__optimum_mcs_objective_literal__(8342199, mus(55)).
__optimum_mcs_objective_literal__(6770130, mus(56)).
__optimum_mcs_objective_literal__(8875194, mus(57)).
__optimum_mcs_objective_literal__(8933828, mus(58)).
__optimum_mcs_objective_literal__(5775890, mus(59)).
__optimum_mcs_objective_literal__(9350981, mus(60)).
__optimum_mcs_objective_literal__(8478897, mus(61)).
__optimum_mcs_objective_literal__(5563709, mus(62)).
__optimum_mcs_objective_literal__(5973564, mus(63)).
__optimum_mcs_objective_literal__(8310582, mus(64)).
__optimum_mcs_objective_literal__(9567813, mus(65)).
__optimum_mcs_objective_literal__(7609305, mus(66)).
__optimum_mcs_objective_literal__(5941306, mus(67)).
__optimum_mcs_objective_literal__(8698783, mus(68)).
__optimum_mcs_objective_literal__(9724501, mus(69)).
__optimum_mcs_objective_literal__(9999999, mus(70)).
__optimum_mcs_objective_literal__(9999996, mus(71)).
__optimum_mcs_objective_literal__(9999999, mus(72)).
__optimum_mcs_objective_literal__(7778646, mus(73)).
__optimum_mcs_objective_literal__(7744567, mus(74)).
__optimum_mcs_objective_literal__(8727373, mus(75)).
__optimum_mcs_objective_literal__(8382747, mus(76)).
__optimum_mcs_objective_literal__(9999999, mus(77)).
__optimum_mcs_objective_literal__(9999999, mus(78)).
__optimum_mcs_objective_literal__(9999999, mus(79)).
__optimum_mcs_objective_literal__(6531806, mus(80)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
