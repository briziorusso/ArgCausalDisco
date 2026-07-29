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
in(0,s0y1y6).
in(1,s0y1y6).
in(6,s0y1y6).
in(0,s0y1y7).
in(1,s0y1y7).
in(7,s0y1y7).
in(0,s0y2y3).
in(2,s0y2y3).
in(3,s0y2y3).
in(0,s0y2y4).
in(2,s0y2y4).
in(4,s0y2y4).
in(0,s0y2y5).
in(2,s0y2y5).
in(5,s0y2y5).
in(0,s0y2y6).
in(2,s0y2y6).
in(6,s0y2y6).
in(0,s0y2y7).
in(2,s0y2y7).
in(7,s0y2y7).
in(0,s0y3y4).
in(3,s0y3y4).
in(4,s0y3y4).
in(0,s0y3y5).
in(3,s0y3y5).
in(5,s0y3y5).
in(0,s0y3y6).
in(3,s0y3y6).
in(6,s0y3y6).
in(0,s0y3y7).
in(3,s0y3y7).
in(7,s0y3y7).
in(0,s0y4y5).
in(4,s0y4y5).
in(5,s0y4y5).
in(0,s0y4y6).
in(4,s0y4y6).
in(6,s0y4y6).
in(0,s0y4y7).
in(4,s0y4y7).
in(7,s0y4y7).
in(0,s0y5y6).
in(5,s0y5y6).
in(6,s0y5y6).
in(0,s0y5y7).
in(5,s0y5y7).
in(7,s0y5y7).
in(0,s0y6y7).
in(6,s0y6y7).
in(7,s0y6y7).
in(1,s1y2y3).
in(2,s1y2y3).
in(3,s1y2y3).
in(1,s1y2y4).
in(2,s1y2y4).
in(4,s1y2y4).
in(1,s1y2y5).
in(2,s1y2y5).
in(5,s1y2y5).
in(1,s1y2y6).
in(2,s1y2y6).
in(6,s1y2y6).
in(1,s1y2y7).
in(2,s1y2y7).
in(7,s1y2y7).
in(1,s1y3y4).
in(3,s1y3y4).
in(4,s1y3y4).
in(1,s1y3y5).
in(3,s1y3y5).
in(5,s1y3y5).
in(1,s1y3y6).
in(3,s1y3y6).
in(6,s1y3y6).
in(1,s1y3y7).
in(3,s1y3y7).
in(7,s1y3y7).
in(1,s1y4y5).
in(4,s1y4y5).
in(5,s1y4y5).
in(1,s1y4y6).
in(4,s1y4y6).
in(6,s1y4y6).
in(1,s1y4y7).
in(4,s1y4y7).
in(7,s1y4y7).
in(1,s1y5y6).
in(5,s1y5y6).
in(6,s1y5y6).
in(1,s1y5y7).
in(5,s1y5y7).
in(7,s1y5y7).
in(1,s1y6y7).
in(6,s1y6y7).
in(7,s1y6y7).
in(2,s2y3y4).
in(3,s2y3y4).
in(4,s2y3y4).
in(2,s2y3y5).
in(3,s2y3y5).
in(5,s2y3y5).
in(2,s2y3y6).
in(3,s2y3y6).
in(6,s2y3y6).
in(2,s2y3y7).
in(3,s2y3y7).
in(7,s2y3y7).
in(2,s2y4y5).
in(4,s2y4y5).
in(5,s2y4y5).
in(2,s2y4y6).
in(4,s2y4y6).
in(6,s2y4y6).
in(2,s2y4y7).
in(4,s2y4y7).
in(7,s2y4y7).
in(2,s2y5y6).
in(5,s2y5y6).
in(6,s2y5y6).
in(2,s2y5y7).
in(5,s2y5y7).
in(7,s2y5y7).
in(2,s2y6y7).
in(6,s2y6y7).
in(7,s2y6y7).
in(3,s3y4y5).
in(4,s3y4y5).
in(5,s3y4y5).
in(3,s3y4y6).
in(4,s3y4y6).
in(6,s3y4y6).
in(3,s3y4y7).
in(4,s3y4y7).
in(7,s3y4y7).
in(3,s3y5y6).
in(5,s3y5y6).
in(6,s3y5y6).
in(3,s3y5y7).
in(5,s3y5y7).
in(7,s3y5y7).
in(3,s3y6y7).
in(6,s3y6y7).
in(7,s3y6y7).
in(4,s4y5y6).
in(5,s4y5y6).
in(6,s4y5y6).
in(4,s4y5y7).
in(5,s4y5y7).
in(7,s4y5y7).
in(4,s4y6y7).
in(6,s4y6y7).
in(7,s4y6y7).
in(5,s5y6y7).
in(6,s5y6y7).
in(7,s5y6y7).
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
in(0,s0y1y2y6).
in(1,s0y1y2y6).
in(2,s0y1y2y6).
in(6,s0y1y2y6).
in(0,s0y1y2y7).
in(1,s0y1y2y7).
in(2,s0y1y2y7).
in(7,s0y1y2y7).
in(0,s0y1y3y4).
in(1,s0y1y3y4).
in(3,s0y1y3y4).
in(4,s0y1y3y4).
in(0,s0y1y3y5).
in(1,s0y1y3y5).
in(3,s0y1y3y5).
in(5,s0y1y3y5).
in(0,s0y1y3y6).
in(1,s0y1y3y6).
in(3,s0y1y3y6).
in(6,s0y1y3y6).
in(0,s0y1y3y7).
in(1,s0y1y3y7).
in(3,s0y1y3y7).
in(7,s0y1y3y7).
in(0,s0y1y4y5).
in(1,s0y1y4y5).
in(4,s0y1y4y5).
in(5,s0y1y4y5).
in(0,s0y1y4y6).
in(1,s0y1y4y6).
in(4,s0y1y4y6).
in(6,s0y1y4y6).
in(0,s0y1y4y7).
in(1,s0y1y4y7).
in(4,s0y1y4y7).
in(7,s0y1y4y7).
in(0,s0y1y5y6).
in(1,s0y1y5y6).
in(5,s0y1y5y6).
in(6,s0y1y5y6).
in(0,s0y1y5y7).
in(1,s0y1y5y7).
in(5,s0y1y5y7).
in(7,s0y1y5y7).
in(0,s0y1y6y7).
in(1,s0y1y6y7).
in(6,s0y1y6y7).
in(7,s0y1y6y7).
in(0,s0y2y3y4).
in(2,s0y2y3y4).
in(3,s0y2y3y4).
in(4,s0y2y3y4).
in(0,s0y2y3y5).
in(2,s0y2y3y5).
in(3,s0y2y3y5).
in(5,s0y2y3y5).
in(0,s0y2y3y6).
in(2,s0y2y3y6).
in(3,s0y2y3y6).
in(6,s0y2y3y6).
in(0,s0y2y3y7).
in(2,s0y2y3y7).
in(3,s0y2y3y7).
in(7,s0y2y3y7).
in(0,s0y2y4y5).
in(2,s0y2y4y5).
in(4,s0y2y4y5).
in(5,s0y2y4y5).
in(0,s0y2y4y6).
in(2,s0y2y4y6).
in(4,s0y2y4y6).
in(6,s0y2y4y6).
in(0,s0y2y4y7).
in(2,s0y2y4y7).
in(4,s0y2y4y7).
in(7,s0y2y4y7).
in(0,s0y2y5y6).
in(2,s0y2y5y6).
in(5,s0y2y5y6).
in(6,s0y2y5y6).
in(0,s0y2y5y7).
in(2,s0y2y5y7).
in(5,s0y2y5y7).
in(7,s0y2y5y7).
in(0,s0y2y6y7).
in(2,s0y2y6y7).
in(6,s0y2y6y7).
in(7,s0y2y6y7).
in(0,s0y3y4y5).
in(3,s0y3y4y5).
in(4,s0y3y4y5).
in(5,s0y3y4y5).
in(0,s0y3y4y6).
in(3,s0y3y4y6).
in(4,s0y3y4y6).
in(6,s0y3y4y6).
in(0,s0y3y4y7).
in(3,s0y3y4y7).
in(4,s0y3y4y7).
in(7,s0y3y4y7).
in(0,s0y3y5y6).
in(3,s0y3y5y6).
in(5,s0y3y5y6).
in(6,s0y3y5y6).
in(0,s0y3y5y7).
in(3,s0y3y5y7).
in(5,s0y3y5y7).
in(7,s0y3y5y7).
in(0,s0y3y6y7).
in(3,s0y3y6y7).
in(6,s0y3y6y7).
in(7,s0y3y6y7).
in(0,s0y4y5y6).
in(4,s0y4y5y6).
in(5,s0y4y5y6).
in(6,s0y4y5y6).
in(0,s0y4y5y7).
in(4,s0y4y5y7).
in(5,s0y4y5y7).
in(7,s0y4y5y7).
in(0,s0y4y6y7).
in(4,s0y4y6y7).
in(6,s0y4y6y7).
in(7,s0y4y6y7).
in(0,s0y5y6y7).
in(5,s0y5y6y7).
in(6,s0y5y6y7).
in(7,s0y5y6y7).
in(1,s1y2y3y4).
in(2,s1y2y3y4).
in(3,s1y2y3y4).
in(4,s1y2y3y4).
in(1,s1y2y3y5).
in(2,s1y2y3y5).
in(3,s1y2y3y5).
in(5,s1y2y3y5).
in(1,s1y2y3y6).
in(2,s1y2y3y6).
in(3,s1y2y3y6).
in(6,s1y2y3y6).
in(1,s1y2y3y7).
in(2,s1y2y3y7).
in(3,s1y2y3y7).
in(7,s1y2y3y7).
in(1,s1y2y4y5).
in(2,s1y2y4y5).
in(4,s1y2y4y5).
in(5,s1y2y4y5).
in(1,s1y2y4y6).
in(2,s1y2y4y6).
in(4,s1y2y4y6).
in(6,s1y2y4y6).
in(1,s1y2y4y7).
in(2,s1y2y4y7).
in(4,s1y2y4y7).
in(7,s1y2y4y7).
in(1,s1y2y5y6).
in(2,s1y2y5y6).
in(5,s1y2y5y6).
in(6,s1y2y5y6).
in(1,s1y2y5y7).
in(2,s1y2y5y7).
in(5,s1y2y5y7).
in(7,s1y2y5y7).
in(1,s1y2y6y7).
in(2,s1y2y6y7).
in(6,s1y2y6y7).
in(7,s1y2y6y7).
in(1,s1y3y4y5).
in(3,s1y3y4y5).
in(4,s1y3y4y5).
in(5,s1y3y4y5).
in(1,s1y3y4y6).
in(3,s1y3y4y6).
in(4,s1y3y4y6).
in(6,s1y3y4y6).
in(1,s1y3y4y7).
in(3,s1y3y4y7).
in(4,s1y3y4y7).
in(7,s1y3y4y7).
in(1,s1y3y5y6).
in(3,s1y3y5y6).
in(5,s1y3y5y6).
in(6,s1y3y5y6).
in(1,s1y3y5y7).
in(3,s1y3y5y7).
in(5,s1y3y5y7).
in(7,s1y3y5y7).
in(1,s1y3y6y7).
in(3,s1y3y6y7).
in(6,s1y3y6y7).
in(7,s1y3y6y7).
in(1,s1y4y5y6).
in(4,s1y4y5y6).
in(5,s1y4y5y6).
in(6,s1y4y5y6).
in(1,s1y4y5y7).
in(4,s1y4y5y7).
in(5,s1y4y5y7).
in(7,s1y4y5y7).
in(1,s1y4y6y7).
in(4,s1y4y6y7).
in(6,s1y4y6y7).
in(7,s1y4y6y7).
in(1,s1y5y6y7).
in(5,s1y5y6y7).
in(6,s1y5y6y7).
in(7,s1y5y6y7).
in(2,s2y3y4y5).
in(3,s2y3y4y5).
in(4,s2y3y4y5).
in(5,s2y3y4y5).
in(2,s2y3y4y6).
in(3,s2y3y4y6).
in(4,s2y3y4y6).
in(6,s2y3y4y6).
in(2,s2y3y4y7).
in(3,s2y3y4y7).
in(4,s2y3y4y7).
in(7,s2y3y4y7).
in(2,s2y3y5y6).
in(3,s2y3y5y6).
in(5,s2y3y5y6).
in(6,s2y3y5y6).
in(2,s2y3y5y7).
in(3,s2y3y5y7).
in(5,s2y3y5y7).
in(7,s2y3y5y7).
in(2,s2y3y6y7).
in(3,s2y3y6y7).
in(6,s2y3y6y7).
in(7,s2y3y6y7).
in(2,s2y4y5y6).
in(4,s2y4y5y6).
in(5,s2y4y5y6).
in(6,s2y4y5y6).
in(2,s2y4y5y7).
in(4,s2y4y5y7).
in(5,s2y4y5y7).
in(7,s2y4y5y7).
in(2,s2y4y6y7).
in(4,s2y4y6y7).
in(6,s2y4y6y7).
in(7,s2y4y6y7).
in(2,s2y5y6y7).
in(5,s2y5y6y7).
in(6,s2y5y6y7).
in(7,s2y5y6y7).
in(3,s3y4y5y6).
in(4,s3y4y5y6).
in(5,s3y4y5y6).
in(6,s3y4y5y6).
in(3,s3y4y5y7).
in(4,s3y4y5y7).
in(5,s3y4y5y7).
in(7,s3y4y5y7).
in(3,s3y4y6y7).
in(4,s3y4y6y7).
in(6,s3y4y6y7).
in(7,s3y4y6y7).
in(3,s3y5y6y7).
in(5,s3y5y6y7).
in(6,s3y5y6y7).
in(7,s3y5y6y7).
in(4,s4y5y6y7).
in(5,s4y5y6y7).
in(6,s4y5y6y7).
in(7,s4y5y6y7).
qpair(0,4,s2).
qpair(0,5,s3y6y7).
qpair(0,5,s3y7).
qpair(0,5,s1y3y6y7).
qpair(0,5,s6y7).
qpair(0,5,s7).
qpair(0,5,s1y3y7).
qpair(0,5,s1y7).
qpair(0,5,s1y6y7).
qpair(0,7,s5).
qpair(0,7,s4).
qpair(0,7,s1).
qpair(0,7,s4y5).
qpair(0,7,empty).
qpair(1,3,s5).
qpair(1,5,s6).
qpair(1,5,s3y6y7).
qpair(1,5,s3y7).
qpair(1,5,s6y7).
qpair(1,5,s7).
qpair(1,5,s3y6).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(1,6,s5).
qpair(1,7,s0y4y5).
qpair(2,4,s7).
qpair(2,4,empty).
qpair(2,7,s4y5).
qpair(2,7,s0y4).
qpair(2,7,s0y4y5).
qpair(2,7,s4).
qpair(3,5,s6).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s1y7).
qpair(3,5,s6y7).
qpair(3,5,s1y6).
qpair(3,5,empty).
qpair(3,5,s1y6y7).
qpair(3,6,s5).
qpair(3,7,s0y4y5).
qpair(4,5,s1y6y7).
qpair(4,7,s2).
qpair(4,7,s5).
qpair(4,7,s1).
qpair(4,7,s0y5).
qpair(4,7,s0).
qpair(4,7,empty).
qpair(5,6,s1y3).
qpair(5,6,s3y7).
qpair(5,6,s1).
qpair(5,6,s7).
qpair(5,6,s1y7).
qpair(5,6,s1y3y7).
qpair(5,6,empty).
qpair(5,6,s3).
qpair(5,7,s6).
qpair(5,7,s1y3).
qpair(5,7,s1y3y6).
qpair(5,7,s0y4).
qpair(5,7,s4).
qpair(5,7,s1).
qpair(5,7,s3y6).
qpair(5,7,s0).
qpair(5,7,s1y6).
qpair(5,7,empty).
qpair(5,7,s3).
qpair(6,7,s0y4y5).
qpair(0,1,empty).
qpair(0,2,empty).
qpair(0,3,empty).
qpair(0,4,s7).
qpair(0,4,s2y7).
qpair(0,5,s6).
qpair(0,5,s1y3).
qpair(0,5,s1y3y6).
qpair(0,5,s1).
qpair(0,5,s3y6).
qpair(0,5,s1y6).
qpair(0,5,s3).
qpair(0,6,empty).
qpair(1,2,empty).
qpair(1,4,empty).
qpair(1,7,s4y5).
qpair(1,7,s0y4).
qpair(1,7,s0y5).
qpair(2,3,empty).
qpair(2,5,empty).
qpair(2,6,empty).
qpair(2,7,s0).
qpair(2,7,s0y5).
qpair(2,7,s5).
qpair(3,4,empty).
qpair(3,7,s0y4).
qpair(3,7,s5).
qpair(3,7,s4).
qpair(3,7,s4y5).
qpair(3,7,s0y5).
qpair(3,7,s0).
qpair(4,5,s6).
qpair(4,5,s1y3).
qpair(4,5,s1y3y6).
qpair(4,5,s2).
qpair(4,5,s3y6y7).
qpair(4,5,s2y7).
qpair(4,5,s3y7).
qpair(4,5,s1y3y6y7).
qpair(4,5,s1).
qpair(4,5,s7).
qpair(4,5,s1y7).
qpair(4,5,s6y7).
qpair(4,5,s1y3y7).
qpair(4,5,s3y6).
qpair(4,5,s1y6).
qpair(4,5,s3).
qpair(4,6,empty).
qpair(6,7,s0y4).
qpair(6,7,s5).
qpair(6,7,s4).
qpair(6,7,s4y5).
qpair(6,7,s0y5).
qpair(6,7,s0).

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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep/wc_sweep_sf8_2026_sparse_alpha001_noweight_seed2026_2075_20260715_180034/rep10/lps/abapc_inc_full_base_rep10_optN.lp =====
% ===== Base program sha256 (raw): bc30bff898f75d347da34b6f14468c047aff2b7337fcd4c982f4e4b73446fba4 =====
% ===== Base program sha256 (normalized): c0723dcd2a2bac56951a345567c9bc55311f262bc02aa4be1a97e23e142250cb =====
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

% ===== Adorned Facts =====
ext_indep(0,1,empty):-mus(1).
ext_indep(0,2,empty):-mus(2).
ext_indep(0,3,empty):-mus(3).
ext_dep(0,4,s2):-mus(4).
ext_indep(0,4,s7):-mus(5).
ext_indep(0,4,s2y7):-mus(6).
ext_indep(0,5,s1):-mus(7).
ext_indep(0,5,s3):-mus(8).
ext_indep(0,5,s6):-mus(9).
ext_dep(0,5,s7):-mus(10).
ext_indep(0,5,s1y3):-mus(11).
ext_indep(0,5,s1y6):-mus(12).
ext_dep(0,5,s1y7):-mus(13).
ext_indep(0,5,s3y6):-mus(14).
ext_dep(0,5,s3y7):-mus(15).
ext_dep(0,5,s6y7):-mus(16).
ext_indep(0,5,s1y3y6):-mus(17).
ext_dep(0,5,s1y3y7):-mus(18).
ext_dep(0,5,s1y6y7):-mus(19).
ext_dep(0,5,s3y6y7):-mus(20).
ext_dep(0,5,s1y3y6y7):-mus(21).
ext_indep(0,6,empty):-mus(22).
ext_dep(0,7,empty):-mus(23).
ext_dep(0,7,s1):-mus(24).
ext_dep(0,7,s4):-mus(25).
ext_dep(0,7,s5):-mus(26).
ext_dep(0,7,s4y5):-mus(27).
ext_indep(1,2,empty):-mus(28).
ext_dep(1,3,s5):-mus(29).
ext_indep(1,4,empty):-mus(30).
ext_dep(1,5,empty):-mus(31).
ext_dep(1,5,s3):-mus(32).
ext_dep(1,5,s6):-mus(33).
ext_dep(1,5,s7):-mus(34).
ext_dep(1,5,s3y6):-mus(35).
ext_dep(1,5,s3y7):-mus(36).
ext_dep(1,5,s6y7):-mus(37).
ext_dep(1,5,s3y6y7):-mus(38).
ext_dep(1,6,s5):-mus(39).
ext_indep(1,7,s0y4):-mus(40).
ext_indep(1,7,s0y5):-mus(41).
ext_indep(1,7,s4y5):-mus(42).
ext_dep(1,7,s0y4y5):-mus(43).
ext_indep(2,3,empty):-mus(44).
ext_dep(2,4,empty):-mus(45).
ext_dep(2,4,s7):-mus(46).
ext_indep(2,5,empty):-mus(47).
ext_indep(2,6,empty):-mus(48).
ext_indep(2,7,s0):-mus(49).
ext_dep(2,7,s4):-mus(50).
ext_indep(2,7,s5):-mus(51).
ext_dep(2,7,s0y4):-mus(52).
ext_indep(2,7,s0y5):-mus(53).
ext_dep(2,7,s4y5):-mus(54).
ext_dep(2,7,s0y4y5):-mus(55).
ext_indep(3,4,empty):-mus(56).
ext_dep(3,5,empty):-mus(57).
ext_dep(3,5,s1):-mus(58).
ext_dep(3,5,s6):-mus(59).
ext_dep(3,5,s7):-mus(60).
ext_dep(3,5,s1y6):-mus(61).
ext_dep(3,5,s1y7):-mus(62).
ext_dep(3,5,s6y7):-mus(63).
ext_dep(3,5,s1y6y7):-mus(64).
ext_dep(3,6,s5):-mus(65).
ext_indep(3,7,s0):-mus(66).
ext_indep(3,7,s4):-mus(67).
ext_indep(3,7,s5):-mus(68).
ext_indep(3,7,s0y4):-mus(69).
ext_indep(3,7,s0y5):-mus(70).
ext_indep(3,7,s4y5):-mus(71).
ext_dep(3,7,s0y4y5):-mus(72).
ext_indep(4,5,s1):-mus(73).
ext_indep(4,5,s2):-mus(74).
ext_indep(4,5,s3):-mus(75).
ext_indep(4,5,s6):-mus(76).
ext_indep(4,5,s7):-mus(77).
ext_indep(4,5,s1y3):-mus(78).
ext_indep(4,5,s1y6):-mus(79).
ext_indep(4,5,s1y7):-mus(80).
ext_indep(4,5,s2y7):-mus(81).
ext_indep(4,5,s3y6):-mus(82).
ext_indep(4,5,s3y7):-mus(83).
ext_indep(4,5,s6y7):-mus(84).
ext_indep(4,5,s1y3y6):-mus(85).
ext_indep(4,5,s1y3y7):-mus(86).
ext_dep(4,5,s1y6y7):-mus(87).
ext_indep(4,5,s3y6y7):-mus(88).
ext_indep(4,5,s1y3y6y7):-mus(89).
ext_indep(4,6,empty):-mus(90).
ext_dep(4,7,empty):-mus(91).
ext_dep(4,7,s0):-mus(92).
ext_dep(4,7,s1):-mus(93).
ext_dep(4,7,s2):-mus(94).
ext_dep(4,7,s5):-mus(95).
ext_dep(4,7,s0y5):-mus(96).
ext_dep(5,6,empty):-mus(97).
ext_dep(5,6,s1):-mus(98).
ext_dep(5,6,s3):-mus(99).
ext_dep(5,6,s7):-mus(100).
ext_dep(5,6,s1y3):-mus(101).
ext_dep(5,6,s1y7):-mus(102).
ext_dep(5,6,s3y7):-mus(103).
ext_dep(5,6,s1y3y7):-mus(104).
ext_dep(5,7,empty):-mus(105).
ext_dep(5,7,s0):-mus(106).
ext_dep(5,7,s1):-mus(107).
ext_dep(5,7,s3):-mus(108).
ext_dep(5,7,s4):-mus(109).
ext_dep(5,7,s6):-mus(110).
ext_dep(5,7,s0y4):-mus(111).
ext_dep(5,7,s1y3):-mus(112).
ext_dep(5,7,s1y6):-mus(113).
ext_dep(5,7,s3y6):-mus(114).
ext_dep(5,7,s1y3y6):-mus(115).
ext_indep(6,7,s0):-mus(116).
ext_indep(6,7,s4):-mus(117).
ext_indep(6,7,s5):-mus(118).
ext_indep(6,7,s0y4):-mus(119).
ext_indep(6,7,s0y5):-mus(120).
ext_indep(6,7,s4y5):-mus(121).
ext_dep(6,7,s0y4y5):-mus(122).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(9829231, mus(1)).
__optimum_mcs_objective_literal__(9075563, mus(2)).
__optimum_mcs_objective_literal__(8558471, mus(3)).
__optimum_mcs_objective_literal__(9998044, mus(4)).
__optimum_mcs_objective_literal__(6216990, mus(5)).
__optimum_mcs_objective_literal__(5036176, mus(6)).
__optimum_mcs_objective_literal__(7423531, mus(7)).
__optimum_mcs_objective_literal__(6132115, mus(8)).
__optimum_mcs_objective_literal__(7950494, mus(9)).
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(5857383, mus(11)).
__optimum_mcs_objective_literal__(6953292, mus(12)).
__optimum_mcs_objective_literal__(9999999, mus(13)).
__optimum_mcs_objective_literal__(6574444, mus(14)).
__optimum_mcs_objective_literal__(9999999, mus(15)).
__optimum_mcs_objective_literal__(9999999, mus(16)).
__optimum_mcs_objective_literal__(5850979, mus(17)).
__optimum_mcs_objective_literal__(9999999, mus(18)).
__optimum_mcs_objective_literal__(9999999, mus(19)).
__optimum_mcs_objective_literal__(9999999, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(7879359, mus(22)).
__optimum_mcs_objective_literal__(9999999, mus(23)).
__optimum_mcs_objective_literal__(9999999, mus(24)).
__optimum_mcs_objective_literal__(9999999, mus(25)).
__optimum_mcs_objective_literal__(9999999, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(6379149, mus(28)).
__optimum_mcs_objective_literal__(9999999, mus(29)).
__optimum_mcs_objective_literal__(6528919, mus(30)).
__optimum_mcs_objective_literal__(9999999, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(9999999, mus(33)).
__optimum_mcs_objective_literal__(9999999, mus(34)).
__optimum_mcs_objective_literal__(9999999, mus(35)).
__optimum_mcs_objective_literal__(9999999, mus(36)).
__optimum_mcs_objective_literal__(9999999, mus(37)).
__optimum_mcs_objective_literal__(9999999, mus(38)).
__optimum_mcs_objective_literal__(9928161, mus(39)).
__optimum_mcs_objective_literal__(5141317, mus(40)).
__optimum_mcs_objective_literal__(6607777, mus(41)).
__optimum_mcs_objective_literal__(5162900, mus(42)).
__optimum_mcs_objective_literal__(9613441, mus(43)).
__optimum_mcs_objective_literal__(9831615, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(9999999, mus(46)).
__optimum_mcs_objective_literal__(7584274, mus(47)).
__optimum_mcs_objective_literal__(5349960, mus(48)).
__optimum_mcs_objective_literal__(8109410, mus(49)).
__optimum_mcs_objective_literal__(9999999, mus(50)).
__optimum_mcs_objective_literal__(7102126, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(7643876, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9999999, mus(55)).
__optimum_mcs_objective_literal__(9579222, mus(56)).
__optimum_mcs_objective_literal__(9999999, mus(57)).
__optimum_mcs_objective_literal__(9999999, mus(58)).
__optimum_mcs_objective_literal__(9999999, mus(59)).
__optimum_mcs_objective_literal__(9999999, mus(60)).
__optimum_mcs_objective_literal__(9999999, mus(61)).
__optimum_mcs_objective_literal__(9999999, mus(62)).
__optimum_mcs_objective_literal__(9999999, mus(63)).
__optimum_mcs_objective_literal__(9999999, mus(64)).
__optimum_mcs_objective_literal__(9999999, mus(65)).
__optimum_mcs_objective_literal__(5088527, mus(66)).
__optimum_mcs_objective_literal__(9604807, mus(67)).
__optimum_mcs_objective_literal__(7267285, mus(68)).
__optimum_mcs_objective_literal__(6283984, mus(69)).
__optimum_mcs_objective_literal__(5566362, mus(70)).
__optimum_mcs_objective_literal__(7568697, mus(71)).
__optimum_mcs_objective_literal__(9995444, mus(72)).
__optimum_mcs_objective_literal__(7983382, mus(73)).
__optimum_mcs_objective_literal__(5271983, mus(74)).
__optimum_mcs_objective_literal__(9785038, mus(75)).
__optimum_mcs_objective_literal__(6304191, mus(76)).
__optimum_mcs_objective_literal__(7959412, mus(77)).
__optimum_mcs_objective_literal__(8872021, mus(78)).
__optimum_mcs_objective_literal__(7712957, mus(79)).
__optimum_mcs_objective_literal__(5445112, mus(80)).
__optimum_mcs_objective_literal__(5004501, mus(81)).
__optimum_mcs_objective_literal__(8047231, mus(82)).
__optimum_mcs_objective_literal__(8091323, mus(83)).
__optimum_mcs_objective_literal__(5766136, mus(84)).
__optimum_mcs_objective_literal__(8900118, mus(85)).
__optimum_mcs_objective_literal__(5290261, mus(86)).
__optimum_mcs_objective_literal__(8224252, mus(87)).
__optimum_mcs_objective_literal__(5188441, mus(88)).
__optimum_mcs_objective_literal__(5003392, mus(89)).
__optimum_mcs_objective_literal__(7713990, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(9999999, mus(92)).
__optimum_mcs_objective_literal__(9999999, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(9999999, mus(95)).
__optimum_mcs_objective_literal__(9999999, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(9999999, mus(98)).
__optimum_mcs_objective_literal__(9999999, mus(99)).
__optimum_mcs_objective_literal__(9999999, mus(100)).
__optimum_mcs_objective_literal__(9999999, mus(101)).
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
__optimum_mcs_objective_literal__(5555242, mus(116)).
__optimum_mcs_objective_literal__(7171025, mus(117)).
__optimum_mcs_objective_literal__(5848648, mus(118)).
__optimum_mcs_objective_literal__(5706295, mus(119)).
__optimum_mcs_objective_literal__(6886969, mus(120)).
__optimum_mcs_objective_literal__(5520814, mus(121)).
__optimum_mcs_objective_literal__(9634640, mus(122)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
