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
qpair(1,2,s6).
qpair(1,2,s5).
qpair(1,2,s4).
qpair(1,2,s7).
qpair(1,2,empty).
qpair(1,2,s3).
qpair(1,3,s6).
qpair(1,3,empty).
qpair(1,3,s2).
qpair(1,4,s2y5).
qpair(1,4,empty).
qpair(1,4,s2).
qpair(1,5,s2).
qpair(1,5,s3y4).
qpair(1,5,s4).
qpair(1,5,s7).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(1,7,s6).
qpair(1,7,empty).
qpair(1,7,s2).
qpair(2,3,s6).
qpair(2,3,s5).
qpair(2,3,s4).
qpair(2,3,s5y7).
qpair(2,3,s1).
qpair(2,3,s7).
qpair(2,3,s4y5).
qpair(2,3,s6y7).
qpair(2,3,s5y6).
qpair(2,3,empty).
qpair(2,3,s4y7).
qpair(2,4,s6).
qpair(2,4,s5).
qpair(2,4,s1).
qpair(2,4,s7).
qpair(2,4,empty).
qpair(2,5,s6).
qpair(2,5,s4).
qpair(2,5,s7).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(2,6,s1).
qpair(2,6,empty).
qpair(2,6,s4).
qpair(2,6,s5).
qpair(2,7,s6).
qpair(2,7,s1).
qpair(2,7,empty).
qpair(2,7,s5).
qpair(3,4,s6).
qpair(3,4,s2).
qpair(3,4,s2y7).
qpair(3,4,s5).
qpair(3,4,s5y7).
qpair(3,4,s1).
qpair(3,4,s7).
qpair(3,4,s6y7).
qpair(3,4,s2y5).
qpair(3,4,s2y6).
qpair(3,4,s5y6).
qpair(3,4,empty).
qpair(3,5,s6).
qpair(3,5,s2).
qpair(3,5,s2y7).
qpair(3,5,s4).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s6y7).
qpair(3,5,s2y6).
qpair(3,5,empty).
qpair(3,5,s4y7).
qpair(3,6,s2y4).
qpair(3,6,s2).
qpair(3,6,s2y7).
qpair(3,6,s5).
qpair(3,6,s4).
qpair(3,6,s5y7).
qpair(3,6,s1).
qpair(3,6,s7).
qpair(3,6,s4y5).
qpair(3,6,s2y5).
qpair(3,6,empty).
qpair(3,6,s4y7).
qpair(3,7,s6).
qpair(3,7,s2y4).
qpair(3,7,s2).
qpair(3,7,s5).
qpair(3,7,s4).
qpair(3,7,s1).
qpair(3,7,s4y5).
qpair(3,7,s2y5).
qpair(3,7,s2y6).
qpair(3,7,s5y6).
qpair(3,7,empty).
qpair(4,5,s1y3).
qpair(4,5,s2).
qpair(4,5,s1).
qpair(4,5,s7).
qpair(4,5,empty).
qpair(4,5,s3).
qpair(4,6,s3).
qpair(4,6,s3y5).
qpair(4,7,s6).
qpair(4,7,s2).
qpair(4,7,s5).
qpair(4,7,s1).
qpair(4,7,empty).
qpair(5,7,s6).
qpair(5,7,s1).
qpair(5,7,empty).
qpair(5,7,s2).
qpair(6,7,s2).
qpair(6,7,s5).
qpair(6,7,s4).
qpair(6,7,s1).
qpair(6,7,empty).
qpair(0,1,empty).
qpair(0,2,empty).
qpair(0,3,empty).
qpair(0,4,empty).
qpair(0,5,empty).
qpair(0,6,empty).
qpair(0,7,empty).
qpair(1,3,s7).
qpair(1,3,s4).
qpair(1,3,s5).
qpair(1,4,s7).
qpair(1,4,s3).
qpair(1,4,s3y5).
qpair(1,4,s5).
qpair(1,6,empty).
qpair(1,7,s3).
qpair(1,7,s4).
qpair(1,7,s5).
qpair(2,3,s4y6).
qpair(2,4,s3).
qpair(2,5,s1).
qpair(2,5,s1y4).
qpair(2,6,s7).
qpair(2,6,s3).
qpair(2,7,s3).
qpair(2,7,s4).
qpair(3,5,s2y4).
qpair(3,5,s4y6).
qpair(3,5,s1y4).
qpair(3,7,s4y6).
qpair(4,6,empty).
qpair(4,6,s5).
qpair(4,7,s3).
qpair(5,6,empty).
qpair(5,7,s3).
qpair(5,7,s4).
qpair(6,7,s3).

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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked/wc_sweep_bnlearn_asia_alpha001_nowrong_noweight_seed2026_2075_20260715_161105/rep9/lps/abapc_inc_full_base_rep9_optN.lp =====
% ===== Base program sha256 (raw): 5fae3bb971c1354d00c7cc95ed0b4ca842a1aff81ad954a9dea3920715d951e8 =====
% ===== Base program sha256 (normalized): 2405af8160a09669060538108c51717771bc95f7b6dd69964b37431fd7c3a95e =====
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
{mus(81)}.
{mus(82)}.
{mus(83)}.
{mus(84)}.
{mus(85)}.
{mus(86)}.
{mus(87)}.
{mus(88)}.
{mus(89)}.
{mus(90)}.
{mus(91)}.
{mus(92)}.
{mus(93)}.
{mus(94)}.
{mus(95)}.
{mus(96)}.
{mus(97)}.
{mus(98)}.
{mus(99)}.
{mus(100)}.
{mus(101)}.
{mus(102)}.
{mus(103)}.
{mus(104)}.
{mus(105)}.
{mus(106)}.
{mus(107)}.
{mus(108)}.
{mus(109)}.
{mus(110)}.
{mus(111)}.
{mus(112)}.
{mus(113)}.
{mus(114)}.
{mus(115)}.
{mus(116)}.
{mus(117)}.
{mus(118)}.
{mus(119)}.
{mus(120)}.
{mus(121)}.
{mus(122)}.
{mus(123)}.
{mus(124)}.
{mus(125)}.
{mus(126)}.
{mus(127)}.
{mus(128)}.
{mus(129)}.
{mus(130)}.
{mus(131)}.
{mus(132)}.
{mus(133)}.
{mus(134)}.
{mus(135)}.
{mus(136)}.
{mus(137)}.
{mus(138)}.
{mus(139)}.
{mus(140)}.
{mus(141)}.
{mus(142)}.
{mus(143)}.
{mus(144)}.
{mus(145)}.
{mus(146)}.
{mus(147)}.
{mus(148)}.
{mus(149)}.
{mus(150)}.
{mus(151)}.
{mus(152)}.
{mus(153)}.
{mus(154)}.

% ===== Adorned Facts =====
ext_indep(0,1,empty):-mus(1).
ext_indep(0,2,empty):-mus(2).
ext_indep(0,3,empty):-mus(3).
ext_indep(0,4,empty):-mus(4).
ext_indep(0,5,empty):-mus(5).
ext_indep(0,6,empty):-mus(6).
ext_indep(0,7,empty):-mus(7).
ext_dep(1,2,empty):-mus(8).
ext_dep(1,2,s3):-mus(9).
ext_dep(1,2,s4):-mus(10).
ext_dep(1,2,s5):-mus(11).
ext_dep(1,2,s7):-mus(12).
ext_dep(1,2,s6):-mus(13).
ext_dep(1,3,empty):-mus(14).
ext_dep(1,3,s2):-mus(15).
ext_indep(1,3,s4):-mus(16).
ext_indep(1,3,s5):-mus(17).
ext_indep(1,3,s7):-mus(18).
ext_dep(1,3,s6):-mus(19).
ext_dep(1,4,empty):-mus(20).
ext_dep(1,4,s2):-mus(21).
ext_indep(1,4,s3):-mus(22).
ext_indep(1,4,s5):-mus(23).
ext_indep(1,4,s7):-mus(24).
ext_dep(1,4,s2y5):-mus(25).
ext_indep(1,4,s3y5):-mus(26).
ext_dep(1,5,empty):-mus(27).
ext_dep(1,5,s2):-mus(28).
ext_dep(1,5,s3):-mus(29).
ext_dep(1,5,s4):-mus(30).
ext_dep(1,5,s7):-mus(31).
ext_dep(1,5,s3y4):-mus(32).
ext_indep(1,6,empty):-mus(33).
ext_dep(1,7,empty):-mus(34).
ext_dep(1,7,s2):-mus(35).
ext_indep(1,7,s3):-mus(36).
ext_indep(1,7,s4):-mus(37).
ext_indep(1,7,s5):-mus(38).
ext_dep(1,7,s6):-mus(39).
ext_dep(2,3,empty):-mus(40).
ext_dep(2,3,s1):-mus(41).
ext_dep(2,3,s4):-mus(42).
ext_dep(2,3,s5):-mus(43).
ext_dep(2,3,s6):-mus(44).
ext_dep(2,3,s7):-mus(45).
ext_dep(2,3,s4y5):-mus(46).
ext_indep(2,3,s4y6):-mus(47).
ext_dep(2,3,s4y7):-mus(48).
ext_dep(2,3,s5y6):-mus(49).
ext_dep(2,3,s5y7):-mus(50).
ext_dep(2,3,s6y7):-mus(51).
ext_dep(2,4,empty):-mus(52).
ext_dep(2,4,s1):-mus(53).
ext_indep(2,4,s3):-mus(54).
ext_dep(2,4,s5):-mus(55).
ext_dep(2,4,s6):-mus(56).
ext_dep(2,4,s7):-mus(57).
ext_dep(2,5,empty):-mus(58).
ext_indep(2,5,s1):-mus(59).
ext_dep(2,5,s3):-mus(60).
ext_dep(2,5,s4):-mus(61).
ext_dep(2,5,s6):-mus(62).
ext_dep(2,5,s7):-mus(63).
ext_indep(2,5,s1y4):-mus(64).
ext_dep(2,6,empty):-mus(65).
ext_dep(2,6,s1):-mus(66).
ext_indep(2,6,s3):-mus(67).
ext_dep(2,6,s4):-mus(68).
ext_dep(2,6,s5):-mus(69).
ext_indep(2,6,s7):-mus(70).
ext_dep(2,7,empty):-mus(71).
ext_dep(2,7,s1):-mus(72).
ext_indep(2,7,s3):-mus(73).
ext_indep(2,7,s4):-mus(74).
ext_dep(2,7,s5):-mus(75).
ext_dep(2,7,s6):-mus(76).
ext_dep(3,4,empty):-mus(77).
ext_dep(3,4,s1):-mus(78).
ext_dep(3,4,s2):-mus(79).
ext_dep(3,4,s5):-mus(80).
ext_dep(3,4,s6):-mus(81).
ext_dep(3,4,s7):-mus(82).
ext_dep(3,4,s2y5):-mus(83).
ext_dep(3,4,s2y6):-mus(84).
ext_dep(3,4,s2y7):-mus(85).
ext_dep(3,4,s5y6):-mus(86).
ext_dep(3,4,s5y7):-mus(87).
ext_dep(3,4,s6y7):-mus(88).
ext_dep(3,5,empty):-mus(89).
ext_dep(3,5,s1):-mus(90).
ext_dep(3,5,s2):-mus(91).
ext_dep(3,5,s4):-mus(92).
ext_dep(3,5,s6):-mus(93).
ext_dep(3,5,s7):-mus(94).
ext_indep(3,5,s2y4):-mus(95).
ext_dep(3,5,s2y6):-mus(96).
ext_dep(3,5,s2y7):-mus(97).
ext_indep(3,5,s4y6):-mus(98).
ext_dep(3,5,s4y7):-mus(99).
ext_dep(3,5,s6y7):-mus(100).
ext_indep(3,5,s1y4):-mus(101).
ext_dep(3,6,empty):-mus(102).
ext_dep(3,6,s1):-mus(103).
ext_dep(3,6,s2):-mus(104).
ext_dep(3,6,s4):-mus(105).
ext_dep(3,6,s5):-mus(106).
ext_dep(3,6,s7):-mus(107).
ext_dep(3,6,s2y4):-mus(108).
ext_dep(3,6,s2y5):-mus(109).
ext_dep(3,6,s2y7):-mus(110).
ext_dep(3,6,s4y5):-mus(111).
ext_dep(3,6,s4y7):-mus(112).
ext_dep(3,6,s5y7):-mus(113).
ext_dep(3,7,empty):-mus(114).
ext_dep(3,7,s1):-mus(115).
ext_dep(3,7,s2):-mus(116).
ext_dep(3,7,s4):-mus(117).
ext_dep(3,7,s5):-mus(118).
ext_dep(3,7,s6):-mus(119).
ext_dep(3,7,s2y4):-mus(120).
ext_dep(3,7,s2y5):-mus(121).
ext_dep(3,7,s2y6):-mus(122).
ext_dep(3,7,s4y5):-mus(123).
ext_indep(3,7,s4y6):-mus(124).
ext_dep(3,7,s5y6):-mus(125).
ext_dep(4,5,empty):-mus(126).
ext_dep(4,5,s1):-mus(127).
ext_dep(4,5,s2):-mus(128).
ext_dep(4,5,s3):-mus(129).
ext_dep(4,5,s7):-mus(130).
ext_dep(4,5,s1y3):-mus(131).
ext_indep(4,6,empty):-mus(132).
ext_dep(4,6,s3):-mus(133).
ext_dep(4,6,s3y5):-mus(134).
ext_indep(4,6,s5):-mus(135).
ext_dep(4,7,empty):-mus(136).
ext_dep(4,7,s1):-mus(137).
ext_dep(4,7,s2):-mus(138).
ext_indep(4,7,s3):-mus(139).
ext_dep(4,7,s5):-mus(140).
ext_dep(4,7,s6):-mus(141).
ext_indep(5,6,empty):-mus(142).
ext_dep(5,7,empty):-mus(143).
ext_dep(5,7,s1):-mus(144).
ext_dep(5,7,s2):-mus(145).
ext_indep(5,7,s3):-mus(146).
ext_indep(5,7,s4):-mus(147).
ext_dep(5,7,s6):-mus(148).
ext_dep(6,7,empty):-mus(149).
ext_dep(6,7,s2):-mus(150).
ext_indep(6,7,s3):-mus(151).
ext_dep(6,7,s1):-mus(152).
ext_dep(6,7,s4):-mus(153).
ext_dep(6,7,s5):-mus(154).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(7374025, mus(1)).
__optimum_mcs_objective_literal__(8784711, mus(2)).
__optimum_mcs_objective_literal__(5838381, mus(3)).
__optimum_mcs_objective_literal__(6369517, mus(4)).
__optimum_mcs_objective_literal__(9434669, mus(5)).
__optimum_mcs_objective_literal__(6516407, mus(6)).
__optimum_mcs_objective_literal__(6534982, mus(7)).
__optimum_mcs_objective_literal__(9999999, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(9999999, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(9999999, mus(13)).
__optimum_mcs_objective_literal__(9739964, mus(14)).
__optimum_mcs_objective_literal__(9999999, mus(15)).
__optimum_mcs_objective_literal__(7901141, mus(16)).
__optimum_mcs_objective_literal__(8363666, mus(17)).
__optimum_mcs_objective_literal__(6341035, mus(18)).
__optimum_mcs_objective_literal__(9788907, mus(19)).
__optimum_mcs_objective_literal__(9824443, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(6512698, mus(22)).
__optimum_mcs_objective_literal__(8135114, mus(23)).
__optimum_mcs_objective_literal__(5738564, mus(24)).
__optimum_mcs_objective_literal__(9999999, mus(25)).
__optimum_mcs_objective_literal__(9461466, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(9999999, mus(28)).
__optimum_mcs_objective_literal__(9999999, mus(29)).
__optimum_mcs_objective_literal__(9999999, mus(30)).
__optimum_mcs_objective_literal__(9999999, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(7686180, mus(33)).
__optimum_mcs_objective_literal__(9538793, mus(34)).
__optimum_mcs_objective_literal__(9999998, mus(35)).
__optimum_mcs_objective_literal__(7319693, mus(36)).
__optimum_mcs_objective_literal__(6889067, mus(37)).
__optimum_mcs_objective_literal__(9413297, mus(38)).
__optimum_mcs_objective_literal__(9487364, mus(39)).
__optimum_mcs_objective_literal__(9999999, mus(40)).
__optimum_mcs_objective_literal__(9999999, mus(41)).
__optimum_mcs_objective_literal__(9997682, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(9917254, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9958605, mus(48)).
__optimum_mcs_objective_literal__(9999999, mus(49)).
__optimum_mcs_objective_literal__(9999988, mus(50)).
__optimum_mcs_objective_literal__(9999999, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(9999999, mus(53)).
__optimum_mcs_objective_literal__(6109038, mus(54)).
__optimum_mcs_objective_literal__(9999999, mus(55)).
__optimum_mcs_objective_literal__(9999999, mus(56)).
__optimum_mcs_objective_literal__(9999999, mus(57)).
__optimum_mcs_objective_literal__(9999999, mus(58)).
__optimum_mcs_objective_literal__(5271384, mus(59)).
__optimum_mcs_objective_literal__(9999999, mus(60)).
__optimum_mcs_objective_literal__(9999999, mus(61)).
__optimum_mcs_objective_literal__(9999999, mus(62)).
__optimum_mcs_objective_literal__(9999999, mus(63)).
__optimum_mcs_objective_literal__(6021722, mus(64)).
__optimum_mcs_objective_literal__(9995016, mus(65)).
__optimum_mcs_objective_literal__(9999972, mus(66)).
__optimum_mcs_objective_literal__(6271622, mus(67)).
__optimum_mcs_objective_literal__(9988615, mus(68)).
__optimum_mcs_objective_literal__(9832239, mus(69)).
__optimum_mcs_objective_literal__(5525725, mus(70)).
__optimum_mcs_objective_literal__(9999999, mus(71)).
__optimum_mcs_objective_literal__(9999999, mus(72)).
__optimum_mcs_objective_literal__(9137173, mus(73)).
__optimum_mcs_objective_literal__(5225720, mus(74)).
__optimum_mcs_objective_literal__(9999999, mus(75)).
__optimum_mcs_objective_literal__(9999999, mus(76)).
__optimum_mcs_objective_literal__(9999999, mus(77)).
__optimum_mcs_objective_literal__(9999999, mus(78)).
__optimum_mcs_objective_literal__(9999999, mus(79)).
__optimum_mcs_objective_literal__(9999999, mus(80)).
__optimum_mcs_objective_literal__(9999999, mus(81)).
__optimum_mcs_objective_literal__(9999999, mus(82)).
__optimum_mcs_objective_literal__(9999999, mus(83)).
__optimum_mcs_objective_literal__(9999999, mus(84)).
__optimum_mcs_objective_literal__(9999999, mus(85)).
__optimum_mcs_objective_literal__(9999999, mus(86)).
__optimum_mcs_objective_literal__(9999999, mus(87)).
__optimum_mcs_objective_literal__(9999999, mus(88)).
__optimum_mcs_objective_literal__(9999999, mus(89)).
__optimum_mcs_objective_literal__(9999999, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(6902069, mus(92)).
__optimum_mcs_objective_literal__(9999999, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(5659810, mus(95)).
__optimum_mcs_objective_literal__(9999999, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(9999999, mus(98)).
__optimum_mcs_objective_literal__(7104949, mus(99)).
__optimum_mcs_objective_literal__(9999999, mus(100)).
__optimum_mcs_objective_literal__(5082066, mus(101)).
__optimum_mcs_objective_literal__(9999999, mus(102)).
__optimum_mcs_objective_literal__(9999999, mus(103)).
__optimum_mcs_objective_literal__(9999999, mus(104)).
__optimum_mcs_objective_literal__(9999999, mus(105)).
__optimum_mcs_objective_literal__(9999999, mus(106)).
__optimum_mcs_objective_literal__(9999999, mus(107)).
__optimum_mcs_objective_literal__(9999999, mus(108)).
__optimum_mcs_objective_literal__(9999999, mus(109)).
__optimum_mcs_objective_literal__(9999999, mus(110)).
__optimum_mcs_objective_literal__(9999999, mus(111)).
__optimum_mcs_objective_literal__(9999999, mus(112)).
__optimum_mcs_objective_literal__(9999999, mus(113)).
__optimum_mcs_objective_literal__(9999999, mus(114)).
__optimum_mcs_objective_literal__(9999999, mus(115)).
__optimum_mcs_objective_literal__(9999999, mus(116)).
__optimum_mcs_objective_literal__(9999999, mus(117)).
__optimum_mcs_objective_literal__(9999999, mus(118)).
__optimum_mcs_objective_literal__(9999999, mus(119)).
__optimum_mcs_objective_literal__(9999999, mus(120)).
__optimum_mcs_objective_literal__(9999999, mus(121)).
__optimum_mcs_objective_literal__(9999999, mus(122)).
__optimum_mcs_objective_literal__(9999999, mus(123)).
__optimum_mcs_objective_literal__(9999999, mus(124)).
__optimum_mcs_objective_literal__(9999999, mus(125)).
__optimum_mcs_objective_literal__(9999999, mus(126)).
__optimum_mcs_objective_literal__(9999999, mus(127)).
__optimum_mcs_objective_literal__(9999999, mus(128)).
__optimum_mcs_objective_literal__(9997751, mus(129)).
__optimum_mcs_objective_literal__(9999999, mus(130)).
__optimum_mcs_objective_literal__(9979803, mus(131)).
__optimum_mcs_objective_literal__(6187286, mus(132)).
__optimum_mcs_objective_literal__(9999999, mus(133)).
__optimum_mcs_objective_literal__(9999999, mus(134)).
__optimum_mcs_objective_literal__(6353474, mus(135)).
__optimum_mcs_objective_literal__(9999999, mus(136)).
__optimum_mcs_objective_literal__(9999999, mus(137)).
__optimum_mcs_objective_literal__(9999999, mus(138)).
__optimum_mcs_objective_literal__(5478140, mus(139)).
__optimum_mcs_objective_literal__(9999999, mus(140)).
__optimum_mcs_objective_literal__(9999999, mus(141)).
__optimum_mcs_objective_literal__(5005796, mus(142)).
__optimum_mcs_objective_literal__(9999999, mus(143)).
__optimum_mcs_objective_literal__(9999999, mus(144)).
__optimum_mcs_objective_literal__(9999999, mus(145)).
__optimum_mcs_objective_literal__(6591110, mus(146)).
__optimum_mcs_objective_literal__(7404410, mus(147)).
__optimum_mcs_objective_literal__(9999999, mus(148)).
__optimum_mcs_objective_literal__(9999999, mus(149)).
__optimum_mcs_objective_literal__(9999999, mus(150)).
__optimum_mcs_objective_literal__(5459560, mus(151)).
__optimum_mcs_objective_literal__(9999999, mus(152)).
__optimum_mcs_objective_literal__(9999999, mus(153)).
__optimum_mcs_objective_literal__(9999999, mus(154)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
