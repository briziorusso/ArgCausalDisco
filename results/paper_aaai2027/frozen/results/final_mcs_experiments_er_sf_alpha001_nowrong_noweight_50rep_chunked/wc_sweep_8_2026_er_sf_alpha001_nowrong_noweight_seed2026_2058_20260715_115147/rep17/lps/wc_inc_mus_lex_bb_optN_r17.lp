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
qpair(0,1,s6).
qpair(0,1,s3y4).
qpair(0,1,s4).
qpair(0,1,s3y6).
qpair(0,1,empty).
qpair(0,1,s3).
qpair(0,3,s6).
qpair(0,3,s2).
qpair(0,3,s1y2).
qpair(0,3,s5).
qpair(0,3,s1y5).
qpair(0,3,s1y2y6).
qpair(0,3,s4).
qpair(0,3,s1y5y6).
qpair(0,3,s1).
qpair(0,3,s2y5).
qpair(0,3,s2y6).
qpair(0,3,s5y6).
qpair(0,3,s1y2y5).
qpair(0,3,s1y2y5y6).
qpair(0,3,s1y6).
qpair(0,3,empty).
qpair(0,3,s2y5y6).
qpair(0,4,s2y7).
qpair(0,4,s3y6).
qpair(0,5,s6).
qpair(0,5,s3y7).
qpair(0,5,s3).
qpair(0,5,s3y6).
qpair(0,5,s1y3y6).
qpair(0,5,s1y3).
qpair(0,6,s1y3).
qpair(0,6,s2).
qpair(0,6,s2y7).
qpair(0,6,s3y7).
qpair(0,6,s4).
qpair(0,6,s2y3).
qpair(0,6,s1).
qpair(0,6,s7).
qpair(0,6,s2y3y7).
qpair(0,6,empty).
qpair(0,6,s3).
qpair(0,7,s6).
qpair(0,7,s1y3y6).
qpair(0,7,s4y6).
qpair(0,7,s4).
qpair(0,7,s4y5).
qpair(0,7,s5y6).
qpair(0,7,s4y5y6).
qpair(0,7,s3y6).
qpair(0,7,s1y6).
qpair(1,3,s6).
qpair(1,3,s2).
qpair(1,3,s0y4).
qpair(1,3,s5).
qpair(1,3,s0y2y5).
qpair(1,3,s0y5y6).
qpair(1,3,s4).
qpair(1,3,s0y6).
qpair(1,3,s0y2y6).
qpair(1,3,s0y2).
qpair(1,3,s0y2y5y6).
qpair(1,3,s2y5).
qpair(1,3,s2y6).
qpair(1,3,s0y5).
qpair(1,3,s5y6).
qpair(1,3,s0).
qpair(1,3,empty).
qpair(1,3,s2y5y6).
qpair(1,4,s6).
qpair(1,4,s2).
qpair(1,4,s2y7).
qpair(1,4,s0y3).
qpair(1,4,s7).
qpair(1,4,s0).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,5,s3y4).
qpair(1,5,s0y3y4).
qpair(1,5,s3y7).
qpair(1,5,s0y3).
qpair(1,5,s3).
qpair(1,7,s6).
qpair(1,7,s0y4).
qpair(1,7,s3y4).
qpair(1,7,s0y3y4).
qpair(1,7,s4y6).
qpair(1,7,s4).
qpair(1,7,s4y5).
qpair(1,7,s5y6).
qpair(1,7,s4y5y6).
qpair(2,3,s5).
qpair(2,3,s4y6).
qpair(2,3,s0y1y6).
qpair(2,3,s0y5).
qpair(2,3,s1y6).
qpair(2,3,s0y5y6).
qpair(2,3,s4).
qpair(2,3,s1y5y6).
qpair(2,3,s1).
qpair(2,3,s7).
qpair(2,3,s0y1y5y6).
qpair(2,3,s5y6).
qpair(2,3,s0y1y5).
qpair(2,3,s0y1).
qpair(2,3,s1y5).
qpair(2,3,s0).
qpair(2,3,s6).
qpair(2,3,s0y6).
qpair(2,3,empty).
qpair(2,4,s6).
qpair(2,4,s1).
qpair(2,4,s7).
qpair(2,4,s1y7).
qpair(2,4,s3y6).
qpair(2,4,s0).
qpair(2,4,empty).
qpair(2,4,s3).
qpair(2,5,s3y4).
qpair(2,5,s3y4y6).
qpair(2,5,s3y7).
qpair(2,5,s4y6).
qpair(2,5,s4).
qpair(2,5,s7).
qpair(2,5,s3y6).
qpair(2,5,s3).
qpair(2,6,s0y7).
qpair(2,6,s3y4).
qpair(2,6,s0y3y7).
qpair(2,6,s3y7).
qpair(2,6,s0y3).
qpair(2,6,s4).
qpair(2,6,s1).
qpair(2,6,s7).
qpair(2,6,s0).
qpair(2,6,empty).
qpair(2,6,s3).
qpair(3,5,s6).
qpair(3,5,s0y1).
qpair(3,5,s2).
qpair(3,5,s1y2).
qpair(3,5,s0y1y2).
qpair(3,5,s1y2y6).
qpair(3,5,s0y1y6).
qpair(3,5,s0y6).
qpair(3,5,s0y2y6).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s0y2).
qpair(3,5,s2y6).
qpair(3,5,s0y1y2y6).
qpair(3,5,s0).
qpair(3,5,s1y6).
qpair(3,5,empty).
qpair(3,6,s2).
qpair(3,6,s5).
qpair(3,6,s0y2).
qpair(3,6,s0y5).
qpair(3,6,s1y2y5).
qpair(3,6,s2y5).
qpair(3,6,s0y2y7).
qpair(3,6,s1).
qpair(3,6,s7).
qpair(3,6,s0y1y5).
qpair(3,6,s0y1y2).
qpair(3,6,s0y1).
qpair(3,6,s0y7).
qpair(3,6,s1y2).
qpair(3,6,s2y7).
qpair(3,6,s1y5).
qpair(3,6,s0).
qpair(3,6,s0y2y5).
qpair(3,6,empty).
qpair(3,6,s0y1y2y5).
qpair(4,5,s1y2y7).
qpair(4,5,s1y2).
qpair(4,5,s2).
qpair(4,5,s2y7).
qpair(4,5,s7).
qpair(4,5,s1y7).
qpair(4,7,s6).
qpair(4,7,s2).
qpair(4,7,s1y2).
qpair(4,7,s5).
qpair(4,7,s1).
qpair(4,7,s5y6).
qpair(4,7,s0).
qpair(4,7,empty).
qpair(5,7,s6).
qpair(5,7,s2).
qpair(5,7,s4y6).
qpair(5,7,s4).
qpair(5,7,empty).
qpair(5,7,s3).
qpair(6,7,s2).
qpair(6,7,s5).
qpair(6,7,s0y3).
qpair(6,7,s0y2y3).
qpair(6,7,s4).
qpair(6,7,s2y3).
qpair(6,7,s1).
qpair(6,7,s0y2).
qpair(6,7,s4y5).
qpair(6,7,s0).
qpair(6,7,empty).
qpair(6,7,s3).
qpair(0,4,s1y2y7).
qpair(0,4,s1y3y6).
qpair(0,4,s1y2).
qpair(0,4,s1y7).
qpair(0,4,s1y6).
qpair(0,4,s1y3).
qpair(0,5,s1).
qpair(0,5,s7).
qpair(0,5,s1y6).
qpair(0,7,s1y3).
qpair(0,7,s1).
qpair(0,7,s3).
qpair(0,7,s5).
qpair(1,5,s0).
qpair(1,5,s7).
qpair(1,5,s0y4).
qpair(1,5,s4).
qpair(1,7,s0).
qpair(1,7,s0y3).
qpair(1,7,s3).
qpair(1,7,s5).
qpair(2,5,s6).
qpair(4,5,s1).
qpair(4,5,s3).
qpair(4,5,s3y7).

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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked/wc_sweep_8_2026_er_sf_alpha001_nowrong_noweight_seed2026_2058_20260715_115147/rep17/lps/abapc_inc_full_base_rep17_optN.lp =====
% ===== Base program sha256 (raw): ba243522bc009f8b6e43e1f26e171963445bfffe2ac8098e84730791e36e8c7a =====
% ===== Base program sha256 (normalized): a82be97739908864e7e92d82dde0f727f07aca505e2113e6a0552ab825a742eb =====
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
{mus(188)}.
{mus(189)}.
{mus(190)}.
{mus(191)}.
{mus(192)}.
{mus(193)}.
{mus(194)}.
{mus(195)}.
{mus(196)}.
{mus(197)}.
{mus(198)}.
{mus(199)}.
{mus(200)}.
{mus(201)}.
{mus(202)}.
{mus(203)}.
{mus(204)}.
{mus(205)}.
{mus(206)}.
{mus(207)}.
{mus(208)}.
{mus(209)}.
{mus(210)}.
{mus(211)}.
{mus(212)}.
{mus(213)}.
{mus(214)}.
{mus(215)}.
{mus(216)}.
{mus(217)}.
{mus(218)}.
{mus(219)}.
{mus(220)}.
{mus(221)}.
{mus(222)}.
{mus(223)}.
{mus(224)}.
{mus(225)}.
{mus(226)}.
{mus(227)}.
{mus(228)}.
{mus(229)}.
{mus(230)}.
{mus(231)}.

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s3):-mus(2).
ext_dep(0,1,s4):-mus(3).
ext_dep(0,1,s6):-mus(4).
ext_dep(0,1,s3y4):-mus(5).
ext_dep(0,1,s3y6):-mus(6).
ext_dep(0,3,empty):-mus(7).
ext_dep(0,3,s1):-mus(8).
ext_dep(0,3,s2):-mus(9).
ext_dep(0,3,s4):-mus(10).
ext_dep(0,3,s5):-mus(11).
ext_dep(0,3,s6):-mus(12).
ext_dep(0,3,s1y2):-mus(13).
ext_dep(0,3,s1y5):-mus(14).
ext_dep(0,3,s1y6):-mus(15).
ext_dep(0,3,s2y5):-mus(16).
ext_dep(0,3,s2y6):-mus(17).
ext_dep(0,3,s5y6):-mus(18).
ext_dep(0,3,s1y2y5):-mus(19).
ext_dep(0,3,s1y2y6):-mus(20).
ext_dep(0,3,s1y5y6):-mus(21).
ext_dep(0,3,s2y5y6):-mus(22).
ext_dep(0,3,s1y2y5y6):-mus(23).
ext_indep(0,4,s1y2):-mus(24).
ext_indep(0,4,s1y3):-mus(25).
ext_indep(0,4,s1y6):-mus(26).
ext_indep(0,4,s1y7):-mus(27).
ext_dep(0,4,s2y7):-mus(28).
ext_dep(0,4,s3y6):-mus(29).
ext_indep(0,4,s1y2y7):-mus(30).
ext_indep(0,4,s1y3y6):-mus(31).
ext_indep(0,5,s1):-mus(32).
ext_dep(0,5,s3):-mus(33).
ext_dep(0,5,s6):-mus(34).
ext_indep(0,5,s7):-mus(35).
ext_dep(0,5,s1y3):-mus(36).
ext_indep(0,5,s1y6):-mus(37).
ext_dep(0,5,s3y6):-mus(38).
ext_dep(0,5,s3y7):-mus(39).
ext_dep(0,5,s1y3y6):-mus(40).
ext_dep(0,6,empty):-mus(41).
ext_dep(0,6,s1):-mus(42).
ext_dep(0,6,s2):-mus(43).
ext_dep(0,6,s3):-mus(44).
ext_dep(0,6,s4):-mus(45).
ext_dep(0,6,s7):-mus(46).
ext_dep(0,6,s1y3):-mus(47).
ext_dep(0,6,s2y3):-mus(48).
ext_dep(0,6,s2y7):-mus(49).
ext_dep(0,6,s3y7):-mus(50).
ext_dep(0,6,s2y3y7):-mus(51).
ext_indep(0,7,s1):-mus(52).
ext_indep(0,7,s3):-mus(53).
ext_dep(0,7,s4):-mus(54).
ext_indep(0,7,s5):-mus(55).
ext_dep(0,7,s6):-mus(56).
ext_indep(0,7,s1y3):-mus(57).
ext_dep(0,7,s1y6):-mus(58).
ext_dep(0,7,s3y6):-mus(59).
ext_dep(0,7,s4y5):-mus(60).
ext_dep(0,7,s4y6):-mus(61).
ext_dep(0,7,s5y6):-mus(62).
ext_dep(0,7,s1y3y6):-mus(63).
ext_dep(0,7,s4y5y6):-mus(64).
ext_dep(1,3,empty):-mus(65).
ext_dep(1,3,s0):-mus(66).
ext_dep(1,3,s2):-mus(67).
ext_dep(1,3,s4):-mus(68).
ext_dep(1,3,s5):-mus(69).
ext_dep(1,3,s6):-mus(70).
ext_dep(1,3,s0y2):-mus(71).
ext_dep(1,3,s0y4):-mus(72).
ext_dep(1,3,s0y5):-mus(73).
ext_dep(1,3,s0y6):-mus(74).
ext_dep(1,3,s2y5):-mus(75).
ext_dep(1,3,s2y6):-mus(76).
ext_dep(1,3,s5y6):-mus(77).
ext_dep(1,3,s0y2y5):-mus(78).
ext_dep(1,3,s0y2y6):-mus(79).
ext_dep(1,3,s0y5y6):-mus(80).
ext_dep(1,3,s2y5y6):-mus(81).
ext_dep(1,3,s0y2y5y6):-mus(82).
ext_dep(1,4,empty):-mus(83).
ext_dep(1,4,s0):-mus(84).
ext_dep(1,4,s2):-mus(85).
ext_dep(1,4,s3):-mus(86).
ext_dep(1,4,s6):-mus(87).
ext_dep(1,4,s7):-mus(88).
ext_dep(1,4,s0y3):-mus(89).
ext_dep(1,4,s2y7):-mus(90).
ext_indep(1,5,s0):-mus(91).
ext_dep(1,5,s3):-mus(92).
ext_indep(1,5,s4):-mus(93).
ext_indep(1,5,s7):-mus(94).
ext_dep(1,5,s0y3):-mus(95).
ext_indep(1,5,s0y4):-mus(96).
ext_dep(1,5,s3y4):-mus(97).
ext_dep(1,5,s3y7):-mus(98).
ext_dep(1,5,s0y3y4):-mus(99).
ext_indep(1,7,s0):-mus(100).
ext_indep(1,7,s3):-mus(101).
ext_dep(1,7,s4):-mus(102).
ext_indep(1,7,s5):-mus(103).
ext_dep(1,7,s6):-mus(104).
ext_indep(1,7,s0y3):-mus(105).
ext_dep(1,7,s0y4):-mus(106).
ext_dep(1,7,s3y4):-mus(107).
ext_dep(1,7,s4y5):-mus(108).
ext_dep(1,7,s4y6):-mus(109).
ext_dep(1,7,s5y6):-mus(110).
ext_dep(1,7,s0y3y4):-mus(111).
ext_dep(1,7,s4y5y6):-mus(112).
ext_dep(2,3,empty):-mus(113).
ext_dep(2,3,s0):-mus(114).
ext_dep(2,3,s1):-mus(115).
ext_dep(2,3,s4):-mus(116).
ext_dep(2,3,s5):-mus(117).
ext_dep(2,3,s6):-mus(118).
ext_dep(2,3,s7):-mus(119).
ext_dep(2,3,s0y1):-mus(120).
ext_dep(2,3,s0y5):-mus(121).
ext_dep(2,3,s0y6):-mus(122).
ext_dep(2,3,s1y5):-mus(123).
ext_dep(2,3,s1y6):-mus(124).
ext_dep(2,3,s4y6):-mus(125).
ext_dep(2,3,s5y6):-mus(126).
ext_dep(2,3,s0y1y5):-mus(127).
ext_dep(2,3,s0y1y6):-mus(128).
ext_dep(2,3,s0y5y6):-mus(129).
ext_dep(2,3,s1y5y6):-mus(130).
ext_dep(2,3,s0y1y5y6):-mus(131).
ext_dep(2,4,empty):-mus(132).
ext_dep(2,4,s0):-mus(133).
ext_dep(2,4,s1):-mus(134).
ext_dep(2,4,s3):-mus(135).
ext_dep(2,4,s6):-mus(136).
ext_dep(2,4,s7):-mus(137).
ext_dep(2,4,s1y7):-mus(138).
ext_dep(2,4,s3y6):-mus(139).
ext_dep(2,5,s3):-mus(140).
ext_dep(2,5,s4):-mus(141).
ext_indep(2,5,s6):-mus(142).
ext_dep(2,5,s7):-mus(143).
ext_dep(2,5,s3y4):-mus(144).
ext_dep(2,5,s3y6):-mus(145).
ext_dep(2,5,s3y7):-mus(146).
ext_dep(2,5,s4y6):-mus(147).
ext_dep(2,5,s3y4y6):-mus(148).
ext_dep(2,6,empty):-mus(149).
ext_dep(2,6,s0):-mus(150).
ext_dep(2,6,s1):-mus(151).
ext_dep(2,6,s3):-mus(152).
ext_dep(2,6,s4):-mus(153).
ext_dep(2,6,s7):-mus(154).
ext_dep(2,6,s0y3):-mus(155).
ext_dep(2,6,s0y7):-mus(156).
ext_dep(2,6,s3y4):-mus(157).
ext_dep(2,6,s3y7):-mus(158).
ext_dep(2,6,s0y3y7):-mus(159).
ext_dep(3,5,empty):-mus(160).
ext_dep(3,5,s0):-mus(161).
ext_dep(3,5,s1):-mus(162).
ext_dep(3,5,s2):-mus(163).
ext_dep(3,5,s6):-mus(164).
ext_dep(3,5,s7):-mus(165).
ext_dep(3,5,s0y1):-mus(166).
ext_dep(3,5,s0y2):-mus(167).
ext_dep(3,5,s0y6):-mus(168).
ext_dep(3,5,s1y2):-mus(169).
ext_dep(3,5,s1y6):-mus(170).
ext_dep(3,5,s2y6):-mus(171).
ext_dep(3,5,s0y1y2):-mus(172).
ext_dep(3,5,s0y1y6):-mus(173).
ext_dep(3,5,s0y2y6):-mus(174).
ext_dep(3,5,s1y2y6):-mus(175).
ext_dep(3,5,s0y1y2y6):-mus(176).
ext_dep(3,6,empty):-mus(177).
ext_dep(3,6,s0):-mus(178).
ext_dep(3,6,s1):-mus(179).
ext_dep(3,6,s2):-mus(180).
ext_dep(3,6,s5):-mus(181).
ext_dep(3,6,s7):-mus(182).
ext_dep(3,6,s0y1):-mus(183).
ext_dep(3,6,s0y2):-mus(184).
ext_dep(3,6,s0y5):-mus(185).
ext_dep(3,6,s0y7):-mus(186).
ext_dep(3,6,s1y2):-mus(187).
ext_dep(3,6,s1y5):-mus(188).
ext_dep(3,6,s2y5):-mus(189).
ext_dep(3,6,s2y7):-mus(190).
ext_dep(3,6,s0y1y2):-mus(191).
ext_dep(3,6,s0y1y5):-mus(192).
ext_dep(3,6,s0y2y5):-mus(193).
ext_dep(3,6,s0y2y7):-mus(194).
ext_dep(3,6,s1y2y5):-mus(195).
ext_dep(3,6,s0y1y2y5):-mus(196).
ext_indep(4,5,s1):-mus(197).
ext_dep(4,5,s2):-mus(198).
ext_indep(4,5,s3):-mus(199).
ext_dep(4,5,s7):-mus(200).
ext_dep(4,5,s1y2):-mus(201).
ext_dep(4,5,s1y7):-mus(202).
ext_dep(4,5,s2y7):-mus(203).
ext_indep(4,5,s3y7):-mus(204).
ext_dep(4,5,s1y2y7):-mus(205).
ext_dep(4,7,empty):-mus(206).
ext_dep(4,7,s0):-mus(207).
ext_dep(4,7,s1):-mus(208).
ext_dep(4,7,s2):-mus(209).
ext_dep(4,7,s5):-mus(210).
ext_dep(4,7,s6):-mus(211).
ext_dep(4,7,s1y2):-mus(212).
ext_dep(4,7,s5y6):-mus(213).
ext_dep(5,7,empty):-mus(214).
ext_dep(5,7,s2):-mus(215).
ext_dep(5,7,s3):-mus(216).
ext_dep(5,7,s4):-mus(217).
ext_dep(5,7,s6):-mus(218).
ext_dep(5,7,s4y6):-mus(219).
ext_dep(6,7,empty):-mus(220).
ext_dep(6,7,s0):-mus(221).
ext_dep(6,7,s1):-mus(222).
ext_dep(6,7,s2):-mus(223).
ext_dep(6,7,s3):-mus(224).
ext_dep(6,7,s4):-mus(225).
ext_dep(6,7,s5):-mus(226).
ext_dep(6,7,s0y2):-mus(227).
ext_dep(6,7,s0y3):-mus(228).
ext_dep(6,7,s2y3):-mus(229).
ext_dep(6,7,s4y5):-mus(230).
ext_dep(6,7,s0y2y3):-mus(231).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(9999999, mus(1)).
__optimum_mcs_objective_literal__(9999999, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(9999999, mus(4)).
__optimum_mcs_objective_literal__(9999999, mus(5)).
__optimum_mcs_objective_literal__(9999999, mus(6)).
__optimum_mcs_objective_literal__(9999999, mus(7)).
__optimum_mcs_objective_literal__(9999931, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(9999999, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(9999999, mus(13)).
__optimum_mcs_objective_literal__(9999999, mus(14)).
__optimum_mcs_objective_literal__(9999999, mus(15)).
__optimum_mcs_objective_literal__(9999999, mus(16)).
__optimum_mcs_objective_literal__(9999999, mus(17)).
__optimum_mcs_objective_literal__(9999999, mus(18)).
__optimum_mcs_objective_literal__(9999999, mus(19)).
__optimum_mcs_objective_literal__(9999999, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(9999999, mus(22)).
__optimum_mcs_objective_literal__(9999999, mus(23)).
__optimum_mcs_objective_literal__(6411766, mus(24)).
__optimum_mcs_objective_literal__(5024707, mus(25)).
__optimum_mcs_objective_literal__(7165803, mus(26)).
__optimum_mcs_objective_literal__(6434947, mus(27)).
__optimum_mcs_objective_literal__(9999999, mus(28)).
__optimum_mcs_objective_literal__(6163559, mus(29)).
__optimum_mcs_objective_literal__(5587108, mus(30)).
__optimum_mcs_objective_literal__(5016348, mus(31)).
__optimum_mcs_objective_literal__(7847335, mus(32)).
__optimum_mcs_objective_literal__(9999944, mus(33)).
__optimum_mcs_objective_literal__(9928435, mus(34)).
__optimum_mcs_objective_literal__(9972731, mus(35)).
__optimum_mcs_objective_literal__(9980705, mus(36)).
__optimum_mcs_objective_literal__(5030455, mus(37)).
__optimum_mcs_objective_literal__(9999999, mus(38)).
__optimum_mcs_objective_literal__(9990662, mus(39)).
__optimum_mcs_objective_literal__(9999990, mus(40)).
__optimum_mcs_objective_literal__(9999998, mus(41)).
__optimum_mcs_objective_literal__(9995208, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(9999994, mus(45)).
__optimum_mcs_objective_literal__(9999999, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9999999, mus(48)).
__optimum_mcs_objective_literal__(9999999, mus(49)).
__optimum_mcs_objective_literal__(9999999, mus(50)).
__optimum_mcs_objective_literal__(9999999, mus(51)).
__optimum_mcs_objective_literal__(7823138, mus(52)).
__optimum_mcs_objective_literal__(5420675, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9358479, mus(55)).
__optimum_mcs_objective_literal__(9999999, mus(56)).
__optimum_mcs_objective_literal__(5448751, mus(57)).
__optimum_mcs_objective_literal__(9999999, mus(58)).
__optimum_mcs_objective_literal__(9999999, mus(59)).
__optimum_mcs_objective_literal__(9999999, mus(60)).
__optimum_mcs_objective_literal__(9999999, mus(61)).
__optimum_mcs_objective_literal__(9999999, mus(62)).
__optimum_mcs_objective_literal__(9999999, mus(63)).
__optimum_mcs_objective_literal__(9999999, mus(64)).
__optimum_mcs_objective_literal__(9999999, mus(65)).
__optimum_mcs_objective_literal__(9999912, mus(66)).
__optimum_mcs_objective_literal__(9999999, mus(67)).
__optimum_mcs_objective_literal__(9999999, mus(68)).
__optimum_mcs_objective_literal__(9999999, mus(69)).
__optimum_mcs_objective_literal__(9999999, mus(70)).
__optimum_mcs_objective_literal__(9999999, mus(71)).
__optimum_mcs_objective_literal__(9999900, mus(72)).
__optimum_mcs_objective_literal__(9999999, mus(73)).
__optimum_mcs_objective_literal__(9999997, mus(74)).
__optimum_mcs_objective_literal__(9999999, mus(75)).
__optimum_mcs_objective_literal__(9999999, mus(76)).
__optimum_mcs_objective_literal__(9999999, mus(77)).
__optimum_mcs_objective_literal__(9999999, mus(78)).
__optimum_mcs_objective_literal__(9999999, mus(79)).
__optimum_mcs_objective_literal__(9999999, mus(80)).
__optimum_mcs_objective_literal__(9999999, mus(81)).
__optimum_mcs_objective_literal__(9999999, mus(82)).
__optimum_mcs_objective_literal__(9999999, mus(83)).
__optimum_mcs_objective_literal__(9999876, mus(84)).
__optimum_mcs_objective_literal__(9999999, mus(85)).
__optimum_mcs_objective_literal__(9999949, mus(86)).
__optimum_mcs_objective_literal__(9999999, mus(87)).
__optimum_mcs_objective_literal__(9999999, mus(88)).
__optimum_mcs_objective_literal__(9989198, mus(89)).
__optimum_mcs_objective_literal__(9999999, mus(90)).
__optimum_mcs_objective_literal__(6745383, mus(91)).
__optimum_mcs_objective_literal__(9999999, mus(92)).
__optimum_mcs_objective_literal__(6516458, mus(93)).
__optimum_mcs_objective_literal__(6766790, mus(94)).
__optimum_mcs_objective_literal__(9999999, mus(95)).
__optimum_mcs_objective_literal__(7653748, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(9999999, mus(98)).
__optimum_mcs_objective_literal__(9999993, mus(99)).
__optimum_mcs_objective_literal__(8211423, mus(100)).
__optimum_mcs_objective_literal__(6048680, mus(101)).
__optimum_mcs_objective_literal__(9999999, mus(102)).
__optimum_mcs_objective_literal__(7871070, mus(103)).
__optimum_mcs_objective_literal__(9999999, mus(104)).
__optimum_mcs_objective_literal__(5896415, mus(105)).
__optimum_mcs_objective_literal__(9999999, mus(106)).
__optimum_mcs_objective_literal__(9999999, mus(107)).
__optimum_mcs_objective_literal__(9999999, mus(108)).
__optimum_mcs_objective_literal__(9999999, mus(109)).
__optimum_mcs_objective_literal__(9999999, mus(110)).
__optimum_mcs_objective_literal__(9999999, mus(111)).
__optimum_mcs_objective_literal__(9999999, mus(112)).
__optimum_mcs_objective_literal__(9999994, mus(113)).
__optimum_mcs_objective_literal__(9999999, mus(114)).
__optimum_mcs_objective_literal__(9999999, mus(115)).
__optimum_mcs_objective_literal__(9985192, mus(116)).
__optimum_mcs_objective_literal__(9999999, mus(117)).
__optimum_mcs_objective_literal__(9999999, mus(118)).
__optimum_mcs_objective_literal__(9999938, mus(119)).
__optimum_mcs_objective_literal__(9999999, mus(120)).
__optimum_mcs_objective_literal__(9999999, mus(121)).
__optimum_mcs_objective_literal__(9999999, mus(122)).
__optimum_mcs_objective_literal__(9999999, mus(123)).
__optimum_mcs_objective_literal__(9999999, mus(124)).
__optimum_mcs_objective_literal__(9999966, mus(125)).
__optimum_mcs_objective_literal__(9999999, mus(126)).
__optimum_mcs_objective_literal__(9999999, mus(127)).
__optimum_mcs_objective_literal__(9999999, mus(128)).
__optimum_mcs_objective_literal__(9999999, mus(129)).
__optimum_mcs_objective_literal__(9999999, mus(130)).
__optimum_mcs_objective_literal__(9999999, mus(131)).
__optimum_mcs_objective_literal__(9999999, mus(132)).
__optimum_mcs_objective_literal__(9999999, mus(133)).
__optimum_mcs_objective_literal__(9999999, mus(134)).
__optimum_mcs_objective_literal__(9999999, mus(135)).
__optimum_mcs_objective_literal__(9999999, mus(136)).
__optimum_mcs_objective_literal__(9999999, mus(137)).
__optimum_mcs_objective_literal__(9999999, mus(138)).
__optimum_mcs_objective_literal__(9999999, mus(139)).
__optimum_mcs_objective_literal__(9999999, mus(140)).
__optimum_mcs_objective_literal__(9999999, mus(141)).
__optimum_mcs_objective_literal__(5025255, mus(142)).
__optimum_mcs_objective_literal__(9692922, mus(143)).
__optimum_mcs_objective_literal__(9999999, mus(144)).
__optimum_mcs_objective_literal__(9999999, mus(145)).
__optimum_mcs_objective_literal__(9999998, mus(146)).
__optimum_mcs_objective_literal__(9999999, mus(147)).
__optimum_mcs_objective_literal__(9999999, mus(148)).
__optimum_mcs_objective_literal__(8745383, mus(149)).
__optimum_mcs_objective_literal__(9999946, mus(150)).
__optimum_mcs_objective_literal__(9956961, mus(151)).
__optimum_mcs_objective_literal__(9999931, mus(152)).
__optimum_mcs_objective_literal__(9998927, mus(153)).
__optimum_mcs_objective_literal__(9885616, mus(154)).
__optimum_mcs_objective_literal__(9999999, mus(155)).
__optimum_mcs_objective_literal__(9999994, mus(156)).
__optimum_mcs_objective_literal__(9999921, mus(157)).
__optimum_mcs_objective_literal__(9999935, mus(158)).
__optimum_mcs_objective_literal__(9999999, mus(159)).
__optimum_mcs_objective_literal__(9999999, mus(160)).
__optimum_mcs_objective_literal__(9999999, mus(161)).
__optimum_mcs_objective_literal__(9999999, mus(162)).
__optimum_mcs_objective_literal__(9999999, mus(163)).
__optimum_mcs_objective_literal__(9999999, mus(164)).
__optimum_mcs_objective_literal__(9999999, mus(165)).
__optimum_mcs_objective_literal__(9999999, mus(166)).
__optimum_mcs_objective_literal__(9999999, mus(167)).
__optimum_mcs_objective_literal__(9999999, mus(168)).
__optimum_mcs_objective_literal__(9999999, mus(169)).
__optimum_mcs_objective_literal__(9999999, mus(170)).
__optimum_mcs_objective_literal__(9999999, mus(171)).
__optimum_mcs_objective_literal__(9999999, mus(172)).
__optimum_mcs_objective_literal__(9999999, mus(173)).
__optimum_mcs_objective_literal__(9999999, mus(174)).
__optimum_mcs_objective_literal__(9999999, mus(175)).
__optimum_mcs_objective_literal__(9999999, mus(176)).
__optimum_mcs_objective_literal__(9999999, mus(177)).
__optimum_mcs_objective_literal__(9999999, mus(178)).
__optimum_mcs_objective_literal__(9999999, mus(179)).
__optimum_mcs_objective_literal__(9999999, mus(180)).
__optimum_mcs_objective_literal__(9999999, mus(181)).
__optimum_mcs_objective_literal__(9999992, mus(182)).
__optimum_mcs_objective_literal__(9999999, mus(183)).
__optimum_mcs_objective_literal__(9999999, mus(184)).
__optimum_mcs_objective_literal__(9999999, mus(185)).
__optimum_mcs_objective_literal__(9999999, mus(186)).
__optimum_mcs_objective_literal__(9999999, mus(187)).
__optimum_mcs_objective_literal__(9999999, mus(188)).
__optimum_mcs_objective_literal__(9999999, mus(189)).
__optimum_mcs_objective_literal__(9999999, mus(190)).
__optimum_mcs_objective_literal__(9999999, mus(191)).
__optimum_mcs_objective_literal__(9999999, mus(192)).
__optimum_mcs_objective_literal__(9999999, mus(193)).
__optimum_mcs_objective_literal__(9999999, mus(194)).
__optimum_mcs_objective_literal__(9999999, mus(195)).
__optimum_mcs_objective_literal__(9999999, mus(196)).
__optimum_mcs_objective_literal__(6786826, mus(197)).
__optimum_mcs_objective_literal__(9999999, mus(198)).
__optimum_mcs_objective_literal__(7994675, mus(199)).
__optimum_mcs_objective_literal__(9999994, mus(200)).
__optimum_mcs_objective_literal__(9999999, mus(201)).
__optimum_mcs_objective_literal__(9999999, mus(202)).
__optimum_mcs_objective_literal__(9999999, mus(203)).
__optimum_mcs_objective_literal__(5008871, mus(204)).
__optimum_mcs_objective_literal__(9999999, mus(205)).
__optimum_mcs_objective_literal__(9999999, mus(206)).
__optimum_mcs_objective_literal__(9999999, mus(207)).
__optimum_mcs_objective_literal__(9999999, mus(208)).
__optimum_mcs_objective_literal__(9999999, mus(209)).
__optimum_mcs_objective_literal__(9999999, mus(210)).
__optimum_mcs_objective_literal__(9999999, mus(211)).
__optimum_mcs_objective_literal__(9999999, mus(212)).
__optimum_mcs_objective_literal__(9999999, mus(213)).
__optimum_mcs_objective_literal__(9999999, mus(214)).
__optimum_mcs_objective_literal__(9999999, mus(215)).
__optimum_mcs_objective_literal__(9999999, mus(216)).
__optimum_mcs_objective_literal__(9999999, mus(217)).
__optimum_mcs_objective_literal__(9999999, mus(218)).
__optimum_mcs_objective_literal__(9999999, mus(219)).
__optimum_mcs_objective_literal__(9999944, mus(220)).
__optimum_mcs_objective_literal__(9999999, mus(221)).
__optimum_mcs_objective_literal__(9999999, mus(222)).
__optimum_mcs_objective_literal__(9999991, mus(223)).
__optimum_mcs_objective_literal__(9973749, mus(224)).
__optimum_mcs_objective_literal__(9999989, mus(225)).
__optimum_mcs_objective_literal__(9997888, mus(226)).
__optimum_mcs_objective_literal__(9999999, mus(227)).
__optimum_mcs_objective_literal__(9999999, mus(228)).
__optimum_mcs_objective_literal__(9989384, mus(229)).
__optimum_mcs_objective_literal__(9999990, mus(230)).
__optimum_mcs_objective_literal__(9999999, mus(231)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
