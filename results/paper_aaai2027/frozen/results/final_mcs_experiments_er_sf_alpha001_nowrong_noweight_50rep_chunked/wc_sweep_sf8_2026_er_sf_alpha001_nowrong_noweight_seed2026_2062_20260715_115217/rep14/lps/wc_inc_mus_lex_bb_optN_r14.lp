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
qpair(0,1,s2y5).
qpair(0,2,s1y4y7).
qpair(0,2,s5).
qpair(0,2,s4).
qpair(0,2,s1y4).
qpair(0,2,s1).
qpair(0,2,s7).
qpair(0,2,s1y7).
qpair(0,2,empty).
qpair(0,2,s4y7).
qpair(0,4,s2y3y6).
qpair(0,4,s3y5y6).
qpair(0,4,s2y3).
qpair(0,4,s3y6).
qpair(0,4,s2y5).
qpair(1,2,s0y7).
qpair(1,2,s0y4).
qpair(1,2,s5).
qpair(1,2,s0y4y7).
qpair(1,2,s4).
qpair(1,2,s7).
qpair(1,2,s0).
qpair(1,2,empty).
qpair(1,2,s4y7).
qpair(1,3,s4y5).
qpair(1,3,s2y5).
qpair(1,3,s4).
qpair(1,3,s5).
qpair(1,4,s7).
qpair(1,5,s6).
qpair(1,5,s2).
qpair(1,5,s3y4).
qpair(1,5,s3y4y6).
qpair(1,5,s4y6).
qpair(1,5,s4).
qpair(1,5,s3y6).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(1,6,s2).
qpair(1,6,s4).
qpair(1,7,s2).
qpair(1,7,s2y5).
qpair(2,3,s0y1y4).
qpair(2,3,s0y4).
qpair(2,3,s1y4y7).
qpair(2,3,s5).
qpair(2,3,s0y4y7).
qpair(2,3,s1y4).
qpair(2,3,s4).
qpair(2,3,s0y1y4y7).
qpair(2,3,s4y5).
qpair(2,3,s4y7).
qpair(2,4,s6).
qpair(2,4,s0y1).
qpair(2,4,s0y7).
qpair(2,4,s5).
qpair(2,4,s0y1y7).
qpair(2,4,s3y5y6).
qpair(2,4,s1).
qpair(2,4,s7).
qpair(2,4,s1y7).
qpair(2,4,s5y6).
qpair(2,4,s3y6).
qpair(2,4,s0).
qpair(2,4,empty).
qpair(2,4,s3).
qpair(2,4,s3y5).
qpair(2,6,s0y1y4y7).
qpair(2,7,s0y1).
qpair(2,7,s0y1y4).
qpair(2,7,s0y4).
qpair(2,7,s5).
qpair(2,7,s4).
qpair(2,7,s1y4).
qpair(2,7,s1).
qpair(2,7,s0).
qpair(2,7,empty).
qpair(3,4,s6).
qpair(3,4,s2).
qpair(3,4,s5).
qpair(3,4,s1).
qpair(3,4,s7).
qpair(3,4,s2y5).
qpair(3,4,s2y6).
qpair(3,4,s5y6).
qpair(3,4,empty).
qpair(3,4,s2y5y6).
qpair(3,5,s6).
qpair(3,5,s2).
qpair(3,5,s4y6).
qpair(3,5,s1y4y6).
qpair(3,5,s4).
qpair(3,5,s1y4).
qpair(3,5,s1).
qpair(3,5,s1y6).
qpair(3,5,empty).
qpair(4,5,s6).
qpair(4,5,s1y3).
qpair(4,5,s1y3y6).
qpair(4,5,s2).
qpair(4,5,s2y3y6).
qpair(4,5,s2y3).
qpair(4,5,s1).
qpair(4,5,s7).
qpair(4,5,s2y6).
qpair(4,5,s3y6).
qpair(4,5,s1y6).
qpair(4,5,empty).
qpair(4,5,s3).
qpair(4,6,s2).
qpair(4,6,s5).
qpair(4,6,s2y3y5).
qpair(4,6,s2y3).
qpair(4,6,s1).
qpair(4,6,s7).
qpair(4,6,s2y5).
qpair(4,6,empty).
qpair(4,6,s3).
qpair(4,6,s3y5).
qpair(4,7,s2y3y6).
qpair(4,7,s3y5y6).
qpair(4,7,s2y3y5).
qpair(4,7,s2y3).
qpair(4,7,s2y3y5y6).
qpair(4,7,s3y6).
qpair(4,7,s2y5).
qpair(4,7,s3y5).
qpair(5,6,s1y3).
qpair(5,6,s2).
qpair(5,6,s3y4).
qpair(5,6,s4).
qpair(5,6,s1y4).
qpair(5,6,s1y3y4).
qpair(5,6,s1).
qpair(5,6,empty).
qpair(5,6,s3).
qpair(0,1,s2).
qpair(0,1,s5).
qpair(0,3,empty).
qpair(0,4,s6).
qpair(0,4,s2).
qpair(0,4,s5).
qpair(0,4,s2y5y6).
qpair(0,4,s2y3y5).
qpair(0,4,s2y3y5y6).
qpair(0,4,s2y6).
qpair(0,4,s5y6).
qpair(0,4,s3).
qpair(0,4,s3y5).
qpair(0,5,empty).
qpair(0,6,empty).
qpair(0,7,s2).
qpair(1,3,s2).
qpair(1,6,s4y5).
qpair(1,6,s2y5).
qpair(1,6,s5).
qpair(1,7,s5).
qpair(2,3,s0y1).
qpair(2,3,s0y7).
qpair(2,3,s0y1y7).
qpair(2,3,s1).
qpair(2,3,s7).
qpair(2,3,s1y7).
qpair(2,3,s0).
qpair(2,6,s0y1).
qpair(2,6,s0y7).
qpair(2,6,s0y1y4).
qpair(2,6,s0y4).
qpair(2,6,s1y4y7).
qpair(2,6,s5).
qpair(2,6,s0y1y7).
qpair(2,6,s0y4y7).
qpair(2,6,s4).
qpair(2,6,s1y4).
qpair(2,6,s1).
qpair(2,6,s7).
qpair(2,6,s1y7).
qpair(2,6,s4y5).
qpair(2,6,s0).
qpair(2,6,s4y7).
qpair(3,7,empty).
qpair(4,7,s1).
qpair(4,7,s2y6).
qpair(4,7,s5y6).
qpair(4,7,s2y5y6).
qpair(5,7,empty).
qpair(6,7,empty).

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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked/wc_sweep_sf8_2026_er_sf_alpha001_nowrong_noweight_seed2026_2062_20260715_115217/rep14/lps/abapc_inc_full_base_rep14_optN.lp =====
% ===== Base program sha256 (raw): a90536676c3f6537b56373ec344e17847a2658d56c4f16086aab0d9365e47a1f =====
% ===== Base program sha256 (normalized): 9b6867d81b55bf0c1d7c1106ddd4345b0403fb2e0f316816eecb2304d956c43b =====
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
{mus(155)}.
{mus(156)}.
{mus(157)}.
{mus(158)}.
{mus(159)}.
{mus(160)}.
{mus(161)}.
{mus(162)}.
{mus(163)}.
{mus(164)}.
{mus(165)}.
{mus(166)}.
{mus(167)}.
{mus(168)}.
{mus(169)}.
{mus(170)}.
{mus(171)}.
{mus(172)}.
{mus(173)}.
{mus(174)}.
{mus(175)}.
{mus(176)}.
{mus(177)}.
{mus(178)}.
{mus(179)}.
{mus(180)}.
{mus(181)}.
{mus(182)}.
{mus(183)}.
{mus(184)}.
{mus(185)}.
{mus(186)}.
{mus(187)}.

% ===== Adorned Facts =====
ext_indep(0,1,s2):-mus(1).
ext_indep(0,1,s5):-mus(2).
ext_dep(0,1,s2y5):-mus(3).
ext_dep(0,2,empty):-mus(4).
ext_dep(0,2,s1):-mus(5).
ext_dep(0,2,s4):-mus(6).
ext_dep(0,2,s5):-mus(7).
ext_dep(0,2,s7):-mus(8).
ext_dep(0,2,s1y4):-mus(9).
ext_dep(0,2,s1y7):-mus(10).
ext_dep(0,2,s4y7):-mus(11).
ext_dep(0,2,s1y4y7):-mus(12).
ext_indep(0,3,empty):-mus(13).
ext_indep(0,4,s2):-mus(14).
ext_indep(0,4,s3):-mus(15).
ext_indep(0,4,s5):-mus(16).
ext_indep(0,4,s6):-mus(17).
ext_dep(0,4,s2y3):-mus(18).
ext_dep(0,4,s2y5):-mus(19).
ext_indep(0,4,s2y6):-mus(20).
ext_indep(0,4,s3y5):-mus(21).
ext_dep(0,4,s3y6):-mus(22).
ext_indep(0,4,s5y6):-mus(23).
ext_indep(0,4,s2y3y5):-mus(24).
ext_dep(0,4,s2y3y6):-mus(25).
ext_indep(0,4,s2y5y6):-mus(26).
ext_dep(0,4,s3y5y6):-mus(27).
ext_indep(0,4,s2y3y5y6):-mus(28).
ext_indep(0,5,empty):-mus(29).
ext_indep(0,6,empty):-mus(30).
ext_indep(0,7,s2):-mus(31).
ext_dep(1,2,empty):-mus(32).
ext_dep(1,2,s0):-mus(33).
ext_dep(1,2,s4):-mus(34).
ext_dep(1,2,s5):-mus(35).
ext_dep(1,2,s7):-mus(36).
ext_dep(1,2,s0y4):-mus(37).
ext_dep(1,2,s0y7):-mus(38).
ext_dep(1,2,s4y7):-mus(39).
ext_dep(1,2,s0y4y7):-mus(40).
ext_indep(1,3,s2):-mus(41).
ext_dep(1,3,s4):-mus(42).
ext_dep(1,3,s5):-mus(43).
ext_dep(1,3,s2y5):-mus(44).
ext_dep(1,3,s4y5):-mus(45).
ext_dep(1,4,s7):-mus(46).
ext_dep(1,5,empty):-mus(47).
ext_dep(1,5,s2):-mus(48).
ext_dep(1,5,s3):-mus(49).
ext_dep(1,5,s4):-mus(50).
ext_dep(1,5,s6):-mus(51).
ext_dep(1,5,s3y4):-mus(52).
ext_dep(1,5,s3y6):-mus(53).
ext_dep(1,5,s4y6):-mus(54).
ext_dep(1,5,s3y4y6):-mus(55).
ext_dep(1,6,s2):-mus(56).
ext_dep(1,6,s4):-mus(57).
ext_indep(1,6,s5):-mus(58).
ext_indep(1,6,s2y5):-mus(59).
ext_indep(1,6,s4y5):-mus(60).
ext_dep(1,7,s2):-mus(61).
ext_indep(1,7,s5):-mus(62).
ext_dep(1,7,s2y5):-mus(63).
ext_indep(2,3,s0):-mus(64).
ext_indep(2,3,s1):-mus(65).
ext_dep(2,3,s4):-mus(66).
ext_dep(2,3,s5):-mus(67).
ext_indep(2,3,s7):-mus(68).
ext_indep(2,3,s0y1):-mus(69).
ext_dep(2,3,s0y4):-mus(70).
ext_indep(2,3,s0y7):-mus(71).
ext_dep(2,3,s1y4):-mus(72).
ext_indep(2,3,s1y7):-mus(73).
ext_dep(2,3,s4y5):-mus(74).
ext_dep(2,3,s4y7):-mus(75).
ext_dep(2,3,s0y1y4):-mus(76).
ext_indep(2,3,s0y1y7):-mus(77).
ext_dep(2,3,s0y4y7):-mus(78).
ext_dep(2,3,s1y4y7):-mus(79).
ext_dep(2,3,s0y1y4y7):-mus(80).
ext_dep(2,4,empty):-mus(81).
ext_dep(2,4,s0):-mus(82).
ext_dep(2,4,s1):-mus(83).
ext_dep(2,4,s3):-mus(84).
ext_dep(2,4,s5):-mus(85).
ext_dep(2,4,s6):-mus(86).
ext_dep(2,4,s7):-mus(87).
ext_dep(2,4,s0y1):-mus(88).
ext_dep(2,4,s0y7):-mus(89).
ext_dep(2,4,s1y7):-mus(90).
ext_dep(2,4,s3y5):-mus(91).
ext_dep(2,4,s3y6):-mus(92).
ext_dep(2,4,s5y6):-mus(93).
ext_dep(2,4,s0y1y7):-mus(94).
ext_dep(2,4,s3y5y6):-mus(95).
ext_indep(2,6,s0):-mus(96).
ext_indep(2,6,s1):-mus(97).
ext_indep(2,6,s4):-mus(98).
ext_indep(2,6,s5):-mus(99).
ext_indep(2,6,s7):-mus(100).
ext_indep(2,6,s0y1):-mus(101).
ext_indep(2,6,s0y4):-mus(102).
ext_indep(2,6,s0y7):-mus(103).
ext_indep(2,6,s1y4):-mus(104).
ext_indep(2,6,s1y7):-mus(105).
ext_indep(2,6,s4y5):-mus(106).
ext_indep(2,6,s4y7):-mus(107).
ext_indep(2,6,s0y1y4):-mus(108).
ext_indep(2,6,s0y1y7):-mus(109).
ext_indep(2,6,s0y4y7):-mus(110).
ext_indep(2,6,s1y4y7):-mus(111).
ext_dep(2,6,s0y1y4y7):-mus(112).
ext_dep(2,7,empty):-mus(113).
ext_dep(2,7,s0):-mus(114).
ext_dep(2,7,s1):-mus(115).
ext_dep(2,7,s4):-mus(116).
ext_dep(2,7,s5):-mus(117).
ext_dep(2,7,s0y1):-mus(118).
ext_dep(2,7,s0y4):-mus(119).
ext_dep(2,7,s1y4):-mus(120).
ext_dep(2,7,s0y1y4):-mus(121).
ext_dep(3,4,empty):-mus(122).
ext_dep(3,4,s1):-mus(123).
ext_dep(3,4,s2):-mus(124).
ext_dep(3,4,s5):-mus(125).
ext_dep(3,4,s6):-mus(126).
ext_dep(3,4,s7):-mus(127).
ext_dep(3,4,s2y5):-mus(128).
ext_dep(3,4,s2y6):-mus(129).
ext_dep(3,4,s5y6):-mus(130).
ext_dep(3,4,s2y5y6):-mus(131).
ext_dep(3,5,empty):-mus(132).
ext_dep(3,5,s1):-mus(133).
ext_dep(3,5,s2):-mus(134).
ext_dep(3,5,s4):-mus(135).
ext_dep(3,5,s6):-mus(136).
ext_dep(3,5,s1y4):-mus(137).
ext_dep(3,5,s1y6):-mus(138).
ext_dep(3,5,s4y6):-mus(139).
ext_dep(3,5,s1y4y6):-mus(140).
ext_indep(3,7,empty):-mus(141).
ext_dep(4,5,empty):-mus(142).
ext_dep(4,5,s1):-mus(143).
ext_dep(4,5,s2):-mus(144).
ext_dep(4,5,s3):-mus(145).
ext_dep(4,5,s6):-mus(146).
ext_dep(4,5,s7):-mus(147).
ext_dep(4,5,s1y3):-mus(148).
ext_dep(4,5,s1y6):-mus(149).
ext_dep(4,5,s2y3):-mus(150).
ext_dep(4,5,s2y6):-mus(151).
ext_dep(4,5,s3y6):-mus(152).
ext_dep(4,5,s1y3y6):-mus(153).
ext_dep(4,5,s2y3y6):-mus(154).
ext_dep(4,6,empty):-mus(155).
ext_dep(4,6,s1):-mus(156).
ext_dep(4,6,s2):-mus(157).
ext_dep(4,6,s3):-mus(158).
ext_dep(4,6,s5):-mus(159).
ext_dep(4,6,s7):-mus(160).
ext_dep(4,6,s2y3):-mus(161).
ext_dep(4,6,s2y5):-mus(162).
ext_dep(4,6,s3y5):-mus(163).
ext_dep(4,6,s2y3y5):-mus(164).
ext_indep(4,7,s1):-mus(165).
ext_dep(4,7,s2y3):-mus(166).
ext_dep(4,7,s2y5):-mus(167).
ext_indep(4,7,s2y6):-mus(168).
ext_dep(4,7,s3y5):-mus(169).
ext_dep(4,7,s3y6):-mus(170).
ext_indep(4,7,s5y6):-mus(171).
ext_dep(4,7,s2y3y5):-mus(172).
ext_dep(4,7,s2y3y6):-mus(173).
ext_indep(4,7,s2y5y6):-mus(174).
ext_dep(4,7,s3y5y6):-mus(175).
ext_dep(4,7,s2y3y5y6):-mus(176).
ext_dep(5,6,empty):-mus(177).
ext_dep(5,6,s1):-mus(178).
ext_dep(5,6,s2):-mus(179).
ext_dep(5,6,s3):-mus(180).
ext_dep(5,6,s4):-mus(181).
ext_dep(5,6,s1y3):-mus(182).
ext_dep(5,6,s1y4):-mus(183).
ext_dep(5,6,s3y4):-mus(184).
ext_dep(5,6,s1y3y4):-mus(185).
ext_indep(5,7,empty):-mus(186).
ext_indep(6,7,empty):-mus(187).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(5000990, mus(1)).
__optimum_mcs_objective_literal__(8698983, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(9999999, mus(4)).
__optimum_mcs_objective_literal__(9999999, mus(5)).
__optimum_mcs_objective_literal__(9999999, mus(6)).
__optimum_mcs_objective_literal__(9999999, mus(7)).
__optimum_mcs_objective_literal__(9999999, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(9999999, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(8988622, mus(13)).
__optimum_mcs_objective_literal__(9109934, mus(14)).
__optimum_mcs_objective_literal__(7443996, mus(15)).
__optimum_mcs_objective_literal__(8020950, mus(16)).
__optimum_mcs_objective_literal__(5115148, mus(17)).
__optimum_mcs_objective_literal__(5694798, mus(18)).
__optimum_mcs_objective_literal__(8120683, mus(19)).
__optimum_mcs_objective_literal__(5213843, mus(20)).
__optimum_mcs_objective_literal__(6886005, mus(21)).
__optimum_mcs_objective_literal__(8744203, mus(22)).
__optimum_mcs_objective_literal__(6984897, mus(23)).
__optimum_mcs_objective_literal__(8824469, mus(24)).
__optimum_mcs_objective_literal__(9611141, mus(25)).
__optimum_mcs_objective_literal__(5252187, mus(26)).
__optimum_mcs_objective_literal__(5510567, mus(27)).
__optimum_mcs_objective_literal__(5124178, mus(28)).
__optimum_mcs_objective_literal__(6808660, mus(29)).
__optimum_mcs_objective_literal__(9189235, mus(30)).
__optimum_mcs_objective_literal__(5119449, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(9999999, mus(33)).
__optimum_mcs_objective_literal__(9999999, mus(34)).
__optimum_mcs_objective_literal__(9999999, mus(35)).
__optimum_mcs_objective_literal__(9999999, mus(36)).
__optimum_mcs_objective_literal__(9999999, mus(37)).
__optimum_mcs_objective_literal__(9999999, mus(38)).
__optimum_mcs_objective_literal__(9999999, mus(39)).
__optimum_mcs_objective_literal__(9999999, mus(40)).
__optimum_mcs_objective_literal__(5365453, mus(41)).
__optimum_mcs_objective_literal__(8633381, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(5981686, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9999999, mus(48)).
__optimum_mcs_objective_literal__(9999999, mus(49)).
__optimum_mcs_objective_literal__(9999999, mus(50)).
__optimum_mcs_objective_literal__(9999999, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(9999999, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9999999, mus(55)).
__optimum_mcs_objective_literal__(5840349, mus(56)).
__optimum_mcs_objective_literal__(9453965, mus(57)).
__optimum_mcs_objective_literal__(6890190, mus(58)).
__optimum_mcs_objective_literal__(6411730, mus(59)).
__optimum_mcs_objective_literal__(5423303, mus(60)).
__optimum_mcs_objective_literal__(9991777, mus(61)).
__optimum_mcs_objective_literal__(5282062, mus(62)).
__optimum_mcs_objective_literal__(9901134, mus(63)).
__optimum_mcs_objective_literal__(5914743, mus(64)).
__optimum_mcs_objective_literal__(5073859, mus(65)).
__optimum_mcs_objective_literal__(9999999, mus(66)).
__optimum_mcs_objective_literal__(9975201, mus(67)).
__optimum_mcs_objective_literal__(6174599, mus(68)).
__optimum_mcs_objective_literal__(5177097, mus(69)).
__optimum_mcs_objective_literal__(9999999, mus(70)).
__optimum_mcs_objective_literal__(6931618, mus(71)).
__optimum_mcs_objective_literal__(9999999, mus(72)).
__optimum_mcs_objective_literal__(5729670, mus(73)).
__optimum_mcs_objective_literal__(9999999, mus(74)).
__optimum_mcs_objective_literal__(9999999, mus(75)).
__optimum_mcs_objective_literal__(9999999, mus(76)).
__optimum_mcs_objective_literal__(6463384, mus(77)).
__optimum_mcs_objective_literal__(9999999, mus(78)).
__optimum_mcs_objective_literal__(9999999, mus(79)).
__optimum_mcs_objective_literal__(9999999, mus(80)).
__optimum_mcs_objective_literal__(9999999, mus(81)).
__optimum_mcs_objective_literal__(9999999, mus(82)).
__optimum_mcs_objective_literal__(9999999, mus(83)).
__optimum_mcs_objective_literal__(9999999, mus(84)).
__optimum_mcs_objective_literal__(9999999, mus(85)).
__optimum_mcs_objective_literal__(9999372, mus(86)).
__optimum_mcs_objective_literal__(9999999, mus(87)).
__optimum_mcs_objective_literal__(9999999, mus(88)).
__optimum_mcs_objective_literal__(9999999, mus(89)).
__optimum_mcs_objective_literal__(9999999, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(9999999, mus(92)).
__optimum_mcs_objective_literal__(9999999, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(9999999, mus(95)).
__optimum_mcs_objective_literal__(8516199, mus(96)).
__optimum_mcs_objective_literal__(6735279, mus(97)).
__optimum_mcs_objective_literal__(9975166, mus(98)).
__optimum_mcs_objective_literal__(8910596, mus(99)).
__optimum_mcs_objective_literal__(7329902, mus(100)).
__optimum_mcs_objective_literal__(6130126, mus(101)).
__optimum_mcs_objective_literal__(9401806, mus(102)).
__optimum_mcs_objective_literal__(6443909, mus(103)).
__optimum_mcs_objective_literal__(6566118, mus(104)).
__optimum_mcs_objective_literal__(5250661, mus(105)).
__optimum_mcs_objective_literal__(9987513, mus(106)).
__optimum_mcs_objective_literal__(9454881, mus(107)).
__optimum_mcs_objective_literal__(5108284, mus(108)).
__optimum_mcs_objective_literal__(5071733, mus(109)).
__optimum_mcs_objective_literal__(6940014, mus(110)).
__optimum_mcs_objective_literal__(5072185, mus(111)).
__optimum_mcs_objective_literal__(9581197, mus(112)).
__optimum_mcs_objective_literal__(9999999, mus(113)).
__optimum_mcs_objective_literal__(9999999, mus(114)).
__optimum_mcs_objective_literal__(9999999, mus(115)).
__optimum_mcs_objective_literal__(9999998, mus(116)).
__optimum_mcs_objective_literal__(9999999, mus(117)).
__optimum_mcs_objective_literal__(9999999, mus(118)).
__optimum_mcs_objective_literal__(9999991, mus(119)).
__optimum_mcs_objective_literal__(9999999, mus(120)).
__optimum_mcs_objective_literal__(9999999, mus(121)).
__optimum_mcs_objective_literal__(9999999, mus(122)).
__optimum_mcs_objective_literal__(9999999, mus(123)).
__optimum_mcs_objective_literal__(9999999, mus(124)).
__optimum_mcs_objective_literal__(9999999, mus(125)).
__optimum_mcs_objective_literal__(9999999, mus(126)).
__optimum_mcs_objective_literal__(9999999, mus(127)).
__optimum_mcs_objective_literal__(9999999, mus(128)).
__optimum_mcs_objective_literal__(9999999, mus(129)).
__optimum_mcs_objective_literal__(9690677, mus(130)).
__optimum_mcs_objective_literal__(9999999, mus(131)).
__optimum_mcs_objective_literal__(9999999, mus(132)).
__optimum_mcs_objective_literal__(9999999, mus(133)).
__optimum_mcs_objective_literal__(9999999, mus(134)).
__optimum_mcs_objective_literal__(9999999, mus(135)).
__optimum_mcs_objective_literal__(9999999, mus(136)).
__optimum_mcs_objective_literal__(9999999, mus(137)).
__optimum_mcs_objective_literal__(9999999, mus(138)).
__optimum_mcs_objective_literal__(9999995, mus(139)).
__optimum_mcs_objective_literal__(9999999, mus(140)).
__optimum_mcs_objective_literal__(5029329, mus(141)).
__optimum_mcs_objective_literal__(9999999, mus(142)).
__optimum_mcs_objective_literal__(9999999, mus(143)).
__optimum_mcs_objective_literal__(9999999, mus(144)).
__optimum_mcs_objective_literal__(9999999, mus(145)).
__optimum_mcs_objective_literal__(9999999, mus(146)).
__optimum_mcs_objective_literal__(9999999, mus(147)).
__optimum_mcs_objective_literal__(9999999, mus(148)).
__optimum_mcs_objective_literal__(9999999, mus(149)).
__optimum_mcs_objective_literal__(9999999, mus(150)).
__optimum_mcs_objective_literal__(9999999, mus(151)).
__optimum_mcs_objective_literal__(9999999, mus(152)).
__optimum_mcs_objective_literal__(9999999, mus(153)).
__optimum_mcs_objective_literal__(9999999, mus(154)).
__optimum_mcs_objective_literal__(9999999, mus(155)).
__optimum_mcs_objective_literal__(9999999, mus(156)).
__optimum_mcs_objective_literal__(9999999, mus(157)).
__optimum_mcs_objective_literal__(9999999, mus(158)).
__optimum_mcs_objective_literal__(9999999, mus(159)).
__optimum_mcs_objective_literal__(9999999, mus(160)).
__optimum_mcs_objective_literal__(9999999, mus(161)).
__optimum_mcs_objective_literal__(9999999, mus(162)).
__optimum_mcs_objective_literal__(9999999, mus(163)).
__optimum_mcs_objective_literal__(9999999, mus(164)).
__optimum_mcs_objective_literal__(5002051, mus(165)).
__optimum_mcs_objective_literal__(9999995, mus(166)).
__optimum_mcs_objective_literal__(9999999, mus(167)).
__optimum_mcs_objective_literal__(6512226, mus(168)).
__optimum_mcs_objective_literal__(9999999, mus(169)).
__optimum_mcs_objective_literal__(9156363, mus(170)).
__optimum_mcs_objective_literal__(6361621, mus(171)).
__optimum_mcs_objective_literal__(9999999, mus(172)).
__optimum_mcs_objective_literal__(7293111, mus(173)).
__optimum_mcs_objective_literal__(5000057, mus(174)).
__optimum_mcs_objective_literal__(8979945, mus(175)).
__optimum_mcs_objective_literal__(9968298, mus(176)).
__optimum_mcs_objective_literal__(9999999, mus(177)).
__optimum_mcs_objective_literal__(9999999, mus(178)).
__optimum_mcs_objective_literal__(9999999, mus(179)).
__optimum_mcs_objective_literal__(9999999, mus(180)).
__optimum_mcs_objective_literal__(9999999, mus(181)).
__optimum_mcs_objective_literal__(9999999, mus(182)).
__optimum_mcs_objective_literal__(9999999, mus(183)).
__optimum_mcs_objective_literal__(9999999, mus(184)).
__optimum_mcs_objective_literal__(9999999, mus(185)).
__optimum_mcs_objective_literal__(7354232, mus(186)).
__optimum_mcs_objective_literal__(9302320, mus(187)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
