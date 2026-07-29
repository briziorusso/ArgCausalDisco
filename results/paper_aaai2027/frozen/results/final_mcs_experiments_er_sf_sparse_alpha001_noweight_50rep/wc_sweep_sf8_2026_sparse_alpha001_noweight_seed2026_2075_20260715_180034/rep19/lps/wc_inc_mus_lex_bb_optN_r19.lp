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
qpair(0,1,s2).
qpair(0,1,s3y4).
qpair(0,1,s5).
qpair(0,1,s4).
qpair(0,1,s4y5).
qpair(0,1,s3y4y5).
qpair(0,1,empty).
qpair(0,1,s3).
qpair(0,1,s3y5).
qpair(0,2,s1).
qpair(0,2,empty).
qpair(0,3,s1).
qpair(0,3,s1y2).
qpair(0,4,s1).
qpair(0,5,s6y7).
qpair(0,5,s7).
qpair(1,2,s3y4).
qpair(1,2,s4y5).
qpair(1,2,s3y4y5).
qpair(1,2,s0y5).
qpair(1,2,s0y3y4y5).
qpair(1,2,s0y4y5).
qpair(1,2,s3y5).
qpair(1,3,s2).
qpair(1,3,s0y4).
qpair(1,3,s5).
qpair(1,3,s4).
qpair(1,3,s4y5).
qpair(1,3,s0y5).
qpair(1,3,s0).
qpair(1,3,empty).
qpair(1,3,s0y4y5).
qpair(1,4,s2).
qpair(1,4,s5).
qpair(1,4,s0y3).
qpair(1,4,s0y5).
qpair(1,4,s0).
qpair(1,4,s0y3y5).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,4,s3y5).
qpair(1,5,s6).
qpair(1,5,s2).
qpair(1,5,s0y4).
qpair(1,5,s3y4).
qpair(1,5,s0y3y4).
qpair(1,5,s0y3).
qpair(1,5,s4).
qpair(1,5,s6y7).
qpair(1,5,s7).
qpair(1,5,s0).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(1,6,s5).
qpair(1,6,s4y5).
qpair(1,6,s3y4y5).
qpair(1,6,s0y5).
qpair(1,6,s0y3y4y5).
qpair(1,6,s0y3y5).
qpair(1,6,s0y4y5).
qpair(1,6,s3y5).
qpair(1,7,s5).
qpair(1,7,s4y5).
qpair(1,7,s3y4y5).
qpair(1,7,s0y5).
qpair(1,7,s0y3y4y5).
qpair(1,7,s0y3y5).
qpair(1,7,s0y4y5).
qpair(1,7,s3y5).
qpair(3,4,s1).
qpair(5,6,s1).
qpair(5,6,s7).
qpair(5,6,empty).
qpair(5,6,s1y7).
qpair(5,7,s6).
qpair(5,7,s1).
qpair(5,7,s1y6).
qpair(5,7,empty).
qpair(6,7,s5).
qpair(0,3,s2).
qpair(0,4,s1y2).
qpair(0,4,s2).
qpair(0,5,s6).
qpair(0,5,s1y2).
qpair(0,5,s2).
qpair(0,5,s1).
qpair(0,5,s1y7).
qpair(0,5,s1y6).
qpair(0,5,s1y6y7).
qpair(0,6,empty).
qpair(0,7,empty).
qpair(1,2,s0y3y4).
qpair(1,2,s0y3y5).
qpair(1,2,s0y3).
qpair(1,2,s0y4).
qpair(1,6,s0y4).
qpair(1,6,s3y4).
qpair(1,6,s0y3y4).
qpair(1,6,s0y3).
qpair(1,6,s4).
qpair(1,6,s0).
qpair(1,6,s3).
qpair(1,7,s0y4).
qpair(1,7,s3y4).
qpair(1,7,s0y3y4).
qpair(1,7,s0y3).
qpair(1,7,s4).
qpair(1,7,s0).
qpair(1,7,s3).
qpair(2,3,empty).
qpair(2,4,empty).
qpair(2,5,empty).
qpair(2,6,empty).
qpair(2,7,empty).
qpair(3,5,s6).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s1y7).
qpair(3,5,s6y7).
qpair(3,5,s1y6).
qpair(3,5,s1y6y7).
qpair(3,6,empty).
qpair(3,7,empty).
qpair(4,5,s6).
qpair(4,5,s1).
qpair(4,5,s7).
qpair(4,5,s1y7).
qpair(4,5,s6y7).
qpair(4,5,s1y6).
qpair(4,5,s1y6y7).
qpair(4,6,empty).
qpair(4,7,empty).

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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep/wc_sweep_sf8_2026_sparse_alpha001_noweight_seed2026_2075_20260715_180034/rep19/lps/abapc_inc_full_base_rep19_optN.lp =====
% ===== Base program sha256 (raw): ad011690805741a076382652e33d6e1f05c87e7a99f26ba32ba51804b2f2762e =====
% ===== Base program sha256 (normalized): 49ce52af36c37fbcff0c4ec554d105952e6d17ba8b2c93b749d6b2e652a88514 =====
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

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s2):-mus(2).
ext_dep(0,1,s3):-mus(3).
ext_dep(0,1,s4):-mus(4).
ext_dep(0,1,s5):-mus(5).
ext_dep(0,1,s3y4):-mus(6).
ext_dep(0,1,s3y5):-mus(7).
ext_dep(0,1,s4y5):-mus(8).
ext_dep(0,1,s3y4y5):-mus(9).
ext_dep(0,2,empty):-mus(10).
ext_dep(0,2,s1):-mus(11).
ext_dep(0,3,s1):-mus(12).
ext_indep(0,3,s2):-mus(13).
ext_dep(0,3,s1y2):-mus(14).
ext_dep(0,4,s1):-mus(15).
ext_indep(0,4,s2):-mus(16).
ext_indep(0,4,s1y2):-mus(17).
ext_indep(0,5,s1):-mus(18).
ext_indep(0,5,s2):-mus(19).
ext_indep(0,5,s6):-mus(20).
ext_dep(0,5,s7):-mus(21).
ext_indep(0,5,s1y2):-mus(22).
ext_indep(0,5,s1y6):-mus(23).
ext_indep(0,5,s1y7):-mus(24).
ext_dep(0,5,s6y7):-mus(25).
ext_indep(0,5,s1y6y7):-mus(26).
ext_indep(0,6,empty):-mus(27).
ext_indep(0,7,empty):-mus(28).
ext_indep(1,2,s0y3):-mus(29).
ext_indep(1,2,s0y4):-mus(30).
ext_dep(1,2,s0y5):-mus(31).
ext_dep(1,2,s3y4):-mus(32).
ext_dep(1,2,s3y5):-mus(33).
ext_dep(1,2,s4y5):-mus(34).
ext_indep(1,2,s0y3y4):-mus(35).
ext_indep(1,2,s0y3y5):-mus(36).
ext_dep(1,2,s0y4y5):-mus(37).
ext_dep(1,2,s3y4y5):-mus(38).
ext_dep(1,2,s0y3y4y5):-mus(39).
ext_dep(1,3,empty):-mus(40).
ext_dep(1,3,s0):-mus(41).
ext_dep(1,3,s2):-mus(42).
ext_dep(1,3,s4):-mus(43).
ext_dep(1,3,s5):-mus(44).
ext_dep(1,3,s0y4):-mus(45).
ext_dep(1,3,s0y5):-mus(46).
ext_dep(1,3,s4y5):-mus(47).
ext_dep(1,3,s0y4y5):-mus(48).
ext_dep(1,4,empty):-mus(49).
ext_dep(1,4,s0):-mus(50).
ext_dep(1,4,s2):-mus(51).
ext_dep(1,4,s3):-mus(52).
ext_dep(1,4,s5):-mus(53).
ext_dep(1,4,s0y3):-mus(54).
ext_dep(1,4,s0y5):-mus(55).
ext_dep(1,4,s3y5):-mus(56).
ext_dep(1,4,s0y3y5):-mus(57).
ext_dep(1,5,empty):-mus(58).
ext_dep(1,5,s0):-mus(59).
ext_dep(1,5,s2):-mus(60).
ext_dep(1,5,s3):-mus(61).
ext_dep(1,5,s4):-mus(62).
ext_dep(1,5,s6):-mus(63).
ext_dep(1,5,s7):-mus(64).
ext_dep(1,5,s0y3):-mus(65).
ext_dep(1,5,s0y4):-mus(66).
ext_dep(1,5,s3y4):-mus(67).
ext_dep(1,5,s6y7):-mus(68).
ext_dep(1,5,s0y3y4):-mus(69).
ext_indep(1,6,s0):-mus(70).
ext_indep(1,6,s3):-mus(71).
ext_indep(1,6,s4):-mus(72).
ext_dep(1,6,s5):-mus(73).
ext_indep(1,6,s0y3):-mus(74).
ext_indep(1,6,s0y4):-mus(75).
ext_dep(1,6,s0y5):-mus(76).
ext_indep(1,6,s3y4):-mus(77).
ext_dep(1,6,s3y5):-mus(78).
ext_dep(1,6,s4y5):-mus(79).
ext_indep(1,6,s0y3y4):-mus(80).
ext_dep(1,6,s0y3y5):-mus(81).
ext_dep(1,6,s0y4y5):-mus(82).
ext_dep(1,6,s3y4y5):-mus(83).
ext_dep(1,6,s0y3y4y5):-mus(84).
ext_indep(1,7,s0):-mus(85).
ext_indep(1,7,s3):-mus(86).
ext_indep(1,7,s4):-mus(87).
ext_dep(1,7,s5):-mus(88).
ext_indep(1,7,s0y3):-mus(89).
ext_indep(1,7,s0y4):-mus(90).
ext_dep(1,7,s0y5):-mus(91).
ext_indep(1,7,s3y4):-mus(92).
ext_dep(1,7,s3y5):-mus(93).
ext_dep(1,7,s4y5):-mus(94).
ext_indep(1,7,s0y3y4):-mus(95).
ext_dep(1,7,s0y3y5):-mus(96).
ext_dep(1,7,s0y4y5):-mus(97).
ext_dep(1,7,s3y4y5):-mus(98).
ext_dep(1,7,s0y3y4y5):-mus(99).
ext_indep(2,3,empty):-mus(100).
ext_indep(2,4,empty):-mus(101).
ext_indep(2,5,empty):-mus(102).
ext_indep(2,6,empty):-mus(103).
ext_indep(2,7,empty):-mus(104).
ext_dep(3,4,s1):-mus(105).
ext_indep(3,5,s1):-mus(106).
ext_indep(3,5,s6):-mus(107).
ext_indep(3,5,s7):-mus(108).
ext_indep(3,5,s1y6):-mus(109).
ext_indep(3,5,s1y7):-mus(110).
ext_indep(3,5,s6y7):-mus(111).
ext_indep(3,5,s1y6y7):-mus(112).
ext_indep(3,6,empty):-mus(113).
ext_indep(3,7,empty):-mus(114).
ext_indep(4,5,s1):-mus(115).
ext_indep(4,5,s6):-mus(116).
ext_indep(4,5,s7):-mus(117).
ext_indep(4,5,s1y6):-mus(118).
ext_indep(4,5,s1y7):-mus(119).
ext_indep(4,5,s6y7):-mus(120).
ext_indep(4,5,s1y6y7):-mus(121).
ext_indep(4,6,empty):-mus(122).
ext_indep(4,7,empty):-mus(123).
ext_dep(5,6,empty):-mus(124).
ext_dep(5,6,s1):-mus(125).
ext_dep(5,6,s7):-mus(126).
ext_dep(5,6,s1y7):-mus(127).
ext_dep(5,7,empty):-mus(128).
ext_dep(5,7,s1):-mus(129).
ext_dep(5,7,s6):-mus(130).
ext_dep(5,7,s1y6):-mus(131).
ext_dep(6,7,s5):-mus(132).

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
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(9999999, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(5113280, mus(13)).
__optimum_mcs_objective_literal__(9999749, mus(14)).
__optimum_mcs_objective_literal__(8694610, mus(15)).
__optimum_mcs_objective_literal__(9838371, mus(16)).
__optimum_mcs_objective_literal__(8926358, mus(17)).
__optimum_mcs_objective_literal__(5825318, mus(18)).
__optimum_mcs_objective_literal__(6161834, mus(19)).
__optimum_mcs_objective_literal__(5115860, mus(20)).
__optimum_mcs_objective_literal__(9370169, mus(21)).
__optimum_mcs_objective_literal__(6287079, mus(22)).
__optimum_mcs_objective_literal__(7460101, mus(23)).
__optimum_mcs_objective_literal__(6168883, mus(24)).
__optimum_mcs_objective_literal__(8740531, mus(25)).
__optimum_mcs_objective_literal__(8052955, mus(26)).
__optimum_mcs_objective_literal__(7212798, mus(27)).
__optimum_mcs_objective_literal__(9466222, mus(28)).
__optimum_mcs_objective_literal__(6886824, mus(29)).
__optimum_mcs_objective_literal__(6211589, mus(30)).
__optimum_mcs_objective_literal__(8223140, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(9999989, mus(33)).
__optimum_mcs_objective_literal__(9999983, mus(34)).
__optimum_mcs_objective_literal__(7709192, mus(35)).
__optimum_mcs_objective_literal__(5030512, mus(36)).
__optimum_mcs_objective_literal__(8907249, mus(37)).
__optimum_mcs_objective_literal__(9999757, mus(38)).
__optimum_mcs_objective_literal__(9534266, mus(39)).
__optimum_mcs_objective_literal__(9999993, mus(40)).
__optimum_mcs_objective_literal__(9999999, mus(41)).
__optimum_mcs_objective_literal__(9999916, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999906, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(9999999, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9999999, mus(48)).
__optimum_mcs_objective_literal__(9999953, mus(49)).
__optimum_mcs_objective_literal__(9999998, mus(50)).
__optimum_mcs_objective_literal__(9995333, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(9986477, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9994262, mus(55)).
__optimum_mcs_objective_literal__(9999999, mus(56)).
__optimum_mcs_objective_literal__(9999999, mus(57)).
__optimum_mcs_objective_literal__(9999999, mus(58)).
__optimum_mcs_objective_literal__(9999999, mus(59)).
__optimum_mcs_objective_literal__(9999999, mus(60)).
__optimum_mcs_objective_literal__(9999999, mus(61)).
__optimum_mcs_objective_literal__(9999999, mus(62)).
__optimum_mcs_objective_literal__(9999999, mus(63)).
__optimum_mcs_objective_literal__(9999999, mus(64)).
__optimum_mcs_objective_literal__(9999999, mus(65)).
__optimum_mcs_objective_literal__(9999999, mus(66)).
__optimum_mcs_objective_literal__(9999999, mus(67)).
__optimum_mcs_objective_literal__(9999999, mus(68)).
__optimum_mcs_objective_literal__(9999999, mus(69)).
__optimum_mcs_objective_literal__(8168749, mus(70)).
__optimum_mcs_objective_literal__(9939518, mus(71)).
__optimum_mcs_objective_literal__(8398859, mus(72)).
__optimum_mcs_objective_literal__(9999999, mus(73)).
__optimum_mcs_objective_literal__(6679196, mus(74)).
__optimum_mcs_objective_literal__(8391651, mus(75)).
__optimum_mcs_objective_literal__(9999992, mus(76)).
__optimum_mcs_objective_literal__(8733936, mus(77)).
__optimum_mcs_objective_literal__(9999998, mus(78)).
__optimum_mcs_objective_literal__(9999995, mus(79)).
__optimum_mcs_objective_literal__(7621074, mus(80)).
__optimum_mcs_objective_literal__(9999935, mus(81)).
__optimum_mcs_objective_literal__(9999992, mus(82)).
__optimum_mcs_objective_literal__(9999583, mus(83)).
__optimum_mcs_objective_literal__(9999717, mus(84)).
__optimum_mcs_objective_literal__(8695199, mus(85)).
__optimum_mcs_objective_literal__(7437754, mus(86)).
__optimum_mcs_objective_literal__(7091591, mus(87)).
__optimum_mcs_objective_literal__(9999999, mus(88)).
__optimum_mcs_objective_literal__(7621579, mus(89)).
__optimum_mcs_objective_literal__(9077942, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(7258192, mus(92)).
__optimum_mcs_objective_literal__(9999999, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(8457794, mus(95)).
__optimum_mcs_objective_literal__(9999999, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(9999999, mus(98)).
__optimum_mcs_objective_literal__(9999978, mus(99)).
__optimum_mcs_objective_literal__(8157114, mus(100)).
__optimum_mcs_objective_literal__(9888384, mus(101)).
__optimum_mcs_objective_literal__(9971651, mus(102)).
__optimum_mcs_objective_literal__(8061121, mus(103)).
__optimum_mcs_objective_literal__(9538921, mus(104)).
__optimum_mcs_objective_literal__(9999999, mus(105)).
__optimum_mcs_objective_literal__(5846209, mus(106)).
__optimum_mcs_objective_literal__(7807373, mus(107)).
__optimum_mcs_objective_literal__(6964142, mus(108)).
__optimum_mcs_objective_literal__(6038334, mus(109)).
__optimum_mcs_objective_literal__(7478063, mus(110)).
__optimum_mcs_objective_literal__(6941674, mus(111)).
__optimum_mcs_objective_literal__(7310236, mus(112)).
__optimum_mcs_objective_literal__(5073124, mus(113)).
__optimum_mcs_objective_literal__(8603052, mus(114)).
__optimum_mcs_objective_literal__(5737349, mus(115)).
__optimum_mcs_objective_literal__(5107187, mus(116)).
__optimum_mcs_objective_literal__(5255587, mus(117)).
__optimum_mcs_objective_literal__(6306421, mus(118)).
__optimum_mcs_objective_literal__(5705911, mus(119)).
__optimum_mcs_objective_literal__(5108926, mus(120)).
__optimum_mcs_objective_literal__(6274374, mus(121)).
__optimum_mcs_objective_literal__(7281661, mus(122)).
__optimum_mcs_objective_literal__(5203462, mus(123)).
__optimum_mcs_objective_literal__(9999999, mus(124)).
__optimum_mcs_objective_literal__(9999999, mus(125)).
__optimum_mcs_objective_literal__(9999999, mus(126)).
__optimum_mcs_objective_literal__(9999999, mus(127)).
__optimum_mcs_objective_literal__(9999999, mus(128)).
__optimum_mcs_objective_literal__(9999999, mus(129)).
__optimum_mcs_objective_literal__(9999999, mus(130)).
__optimum_mcs_objective_literal__(9999999, mus(131)).
__optimum_mcs_objective_literal__(9999999, mus(132)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
