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
qpair(0,1,s3).
qpair(0,1,empty).
qpair(0,1,s2).
qpair(0,1,s4).
qpair(0,2,s1).
qpair(0,2,empty).
qpair(0,2,s5).
qpair(1,2,s5).
qpair(1,2,s4).
qpair(1,2,s0y5).
qpair(1,2,s0).
qpair(1,2,empty).
qpair(1,2,s3).
qpair(1,3,empty).
qpair(1,3,s2).
qpair(1,3,s4).
qpair(1,3,s5).
qpair(1,4,s0).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(2,5,s0y1).
qpair(2,5,s1).
qpair(2,5,s0).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(3,5,s1).
qpair(3,5,empty).
qpair(3,5,s2).
qpair(0,2,s1y5).
qpair(0,3,empty).
qpair(0,4,empty).
qpair(0,5,empty).
qpair(1,3,s0).
qpair(1,4,s2).
qpair(1,5,s2).
qpair(1,5,s2y3).
qpair(1,5,s0y2).
qpair(1,5,s0).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(2,3,s1).
qpair(2,3,empty).
qpair(2,3,s5).
qpair(2,3,s1y5).
qpair(2,4,empty).
qpair(3,4,empty).
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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked/wc_sweep_bnlearn_survey_alpha001_nowrong_noweight_seed2026_2075_20260715_155405/rep50/lps/abapc_inc_full_base_rep50_optN.lp =====
% ===== Base program sha256 (raw): 56c8f459d03578afcb271d5ffb0c5d8319eceed2e42d80d4ead8495d6a1ce3d6 =====
% ===== Base program sha256 (normalized): 194b5d9eae4f5a9280e7882c64ef281a796d531d546cee6c9667304a5c8a77fc =====
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

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s2):-mus(2).
ext_dep(0,1,s3):-mus(3).
ext_dep(0,1,s4):-mus(4).
ext_dep(0,2,empty):-mus(5).
ext_dep(0,2,s1):-mus(6).
ext_dep(0,2,s5):-mus(7).
ext_indep(0,2,s1y5):-mus(8).
ext_indep(0,3,empty):-mus(9).
ext_indep(0,4,empty):-mus(10).
ext_indep(0,5,empty):-mus(11).
ext_dep(1,2,empty):-mus(12).
ext_dep(1,2,s0):-mus(13).
ext_dep(1,2,s3):-mus(14).
ext_dep(1,2,s4):-mus(15).
ext_dep(1,2,s5):-mus(16).
ext_dep(1,2,s0y5):-mus(17).
ext_dep(1,3,empty):-mus(18).
ext_indep(1,3,s0):-mus(19).
ext_dep(1,3,s2):-mus(20).
ext_dep(1,3,s4):-mus(21).
ext_dep(1,3,s5):-mus(22).
ext_dep(1,4,empty):-mus(23).
ext_dep(1,4,s0):-mus(24).
ext_indep(1,4,s2):-mus(25).
ext_dep(1,4,s3):-mus(26).
ext_indep(1,5,empty):-mus(27).
ext_indep(1,5,s2):-mus(28).
ext_indep(1,5,s2y3):-mus(29).
ext_indep(1,5,s0y2):-mus(30).
ext_indep(1,5,s0):-mus(31).
ext_indep(1,5,s3):-mus(32).
ext_indep(2,3,empty):-mus(33).
ext_indep(2,3,s1y5):-mus(34).
ext_indep(2,3,s5):-mus(35).
ext_indep(2,3,s1):-mus(36).
ext_indep(2,4,empty):-mus(37).
ext_dep(2,5,empty):-mus(38).
ext_dep(2,5,s0):-mus(39).
ext_dep(2,5,s1):-mus(40).
ext_dep(2,5,s3):-mus(41).
ext_dep(2,5,s0y1):-mus(42).
ext_indep(3,4,empty):-mus(43).
ext_dep(3,5,empty):-mus(44).
ext_dep(3,5,s1):-mus(45).
ext_dep(3,5,s2):-mus(46).
ext_indep(4,5,empty):-mus(47).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(9999999, mus(1)).
__optimum_mcs_objective_literal__(9999999, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(9999999, mus(4)).
__optimum_mcs_objective_literal__(9892949, mus(5)).
__optimum_mcs_objective_literal__(6972506, mus(6)).
__optimum_mcs_objective_literal__(9471189, mus(7)).
__optimum_mcs_objective_literal__(5130905, mus(8)).
__optimum_mcs_objective_literal__(7809636, mus(9)).
__optimum_mcs_objective_literal__(8557817, mus(10)).
__optimum_mcs_objective_literal__(5144913, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(9999991, mus(13)).
__optimum_mcs_objective_literal__(9999999, mus(14)).
__optimum_mcs_objective_literal__(9999998, mus(15)).
__optimum_mcs_objective_literal__(9999998, mus(16)).
__optimum_mcs_objective_literal__(9999781, mus(17)).
__optimum_mcs_objective_literal__(9576192, mus(18)).
__optimum_mcs_objective_literal__(5013184, mus(19)).
__optimum_mcs_objective_literal__(9117148, mus(20)).
__optimum_mcs_objective_literal__(9314469, mus(21)).
__optimum_mcs_objective_literal__(8096798, mus(22)).
__optimum_mcs_objective_literal__(8477562, mus(23)).
__optimum_mcs_objective_literal__(9808115, mus(24)).
__optimum_mcs_objective_literal__(5057930, mus(25)).
__optimum_mcs_objective_literal__(7776178, mus(26)).
__optimum_mcs_objective_literal__(5500102, mus(27)).
__optimum_mcs_objective_literal__(5472312, mus(28)).
__optimum_mcs_objective_literal__(5743052, mus(29)).
__optimum_mcs_objective_literal__(5955392, mus(30)).
__optimum_mcs_objective_literal__(5799922, mus(31)).
__optimum_mcs_objective_literal__(5714057, mus(32)).
__optimum_mcs_objective_literal__(8317418, mus(33)).
__optimum_mcs_objective_literal__(7201315, mus(34)).
__optimum_mcs_objective_literal__(7472803, mus(35)).
__optimum_mcs_objective_literal__(7076271, mus(36)).
__optimum_mcs_objective_literal__(5119464, mus(37)).
__optimum_mcs_objective_literal__(9999927, mus(38)).
__optimum_mcs_objective_literal__(9999090, mus(39)).
__optimum_mcs_objective_literal__(9999739, mus(40)).
__optimum_mcs_objective_literal__(9999567, mus(41)).
__optimum_mcs_objective_literal__(9986198, mus(42)).
__optimum_mcs_objective_literal__(7438151, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(9999999, mus(46)).
__optimum_mcs_objective_literal__(5320444, mus(47)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
