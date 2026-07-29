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
in(0,s0y1y2y3y4).
in(1,s0y1y2y3y4).
in(2,s0y1y2y3y4).
in(3,s0y1y2y3y4).
in(4,s0y1y2y3y4).
in(0,s0y1y2y3y5).
in(1,s0y1y2y3y5).
in(2,s0y1y2y3y5).
in(3,s0y1y2y3y5).
in(5,s0y1y2y3y5).
in(0,s0y1y2y3y6).
in(1,s0y1y2y3y6).
in(2,s0y1y2y3y6).
in(3,s0y1y2y3y6).
in(6,s0y1y2y3y6).
in(0,s0y1y2y3y7).
in(1,s0y1y2y3y7).
in(2,s0y1y2y3y7).
in(3,s0y1y2y3y7).
in(7,s0y1y2y3y7).
in(0,s0y1y2y4y5).
in(1,s0y1y2y4y5).
in(2,s0y1y2y4y5).
in(4,s0y1y2y4y5).
in(5,s0y1y2y4y5).
in(0,s0y1y2y4y6).
in(1,s0y1y2y4y6).
in(2,s0y1y2y4y6).
in(4,s0y1y2y4y6).
in(6,s0y1y2y4y6).
in(0,s0y1y2y4y7).
in(1,s0y1y2y4y7).
in(2,s0y1y2y4y7).
in(4,s0y1y2y4y7).
in(7,s0y1y2y4y7).
in(0,s0y1y2y5y6).
in(1,s0y1y2y5y6).
in(2,s0y1y2y5y6).
in(5,s0y1y2y5y6).
in(6,s0y1y2y5y6).
in(0,s0y1y2y5y7).
in(1,s0y1y2y5y7).
in(2,s0y1y2y5y7).
in(5,s0y1y2y5y7).
in(7,s0y1y2y5y7).
in(0,s0y1y2y6y7).
in(1,s0y1y2y6y7).
in(2,s0y1y2y6y7).
in(6,s0y1y2y6y7).
in(7,s0y1y2y6y7).
in(0,s0y1y3y4y5).
in(1,s0y1y3y4y5).
in(3,s0y1y3y4y5).
in(4,s0y1y3y4y5).
in(5,s0y1y3y4y5).
in(0,s0y1y3y4y6).
in(1,s0y1y3y4y6).
in(3,s0y1y3y4y6).
in(4,s0y1y3y4y6).
in(6,s0y1y3y4y6).
in(0,s0y1y3y4y7).
in(1,s0y1y3y4y7).
in(3,s0y1y3y4y7).
in(4,s0y1y3y4y7).
in(7,s0y1y3y4y7).
in(0,s0y1y3y5y6).
in(1,s0y1y3y5y6).
in(3,s0y1y3y5y6).
in(5,s0y1y3y5y6).
in(6,s0y1y3y5y6).
in(0,s0y1y3y5y7).
in(1,s0y1y3y5y7).
in(3,s0y1y3y5y7).
in(5,s0y1y3y5y7).
in(7,s0y1y3y5y7).
in(0,s0y1y3y6y7).
in(1,s0y1y3y6y7).
in(3,s0y1y3y6y7).
in(6,s0y1y3y6y7).
in(7,s0y1y3y6y7).
in(0,s0y1y4y5y6).
in(1,s0y1y4y5y6).
in(4,s0y1y4y5y6).
in(5,s0y1y4y5y6).
in(6,s0y1y4y5y6).
in(0,s0y1y4y5y7).
in(1,s0y1y4y5y7).
in(4,s0y1y4y5y7).
in(5,s0y1y4y5y7).
in(7,s0y1y4y5y7).
in(0,s0y1y4y6y7).
in(1,s0y1y4y6y7).
in(4,s0y1y4y6y7).
in(6,s0y1y4y6y7).
in(7,s0y1y4y6y7).
in(0,s0y1y5y6y7).
in(1,s0y1y5y6y7).
in(5,s0y1y5y6y7).
in(6,s0y1y5y6y7).
in(7,s0y1y5y6y7).
in(0,s0y2y3y4y5).
in(2,s0y2y3y4y5).
in(3,s0y2y3y4y5).
in(4,s0y2y3y4y5).
in(5,s0y2y3y4y5).
in(0,s0y2y3y4y6).
in(2,s0y2y3y4y6).
in(3,s0y2y3y4y6).
in(4,s0y2y3y4y6).
in(6,s0y2y3y4y6).
in(0,s0y2y3y4y7).
in(2,s0y2y3y4y7).
in(3,s0y2y3y4y7).
in(4,s0y2y3y4y7).
in(7,s0y2y3y4y7).
in(0,s0y2y3y5y6).
in(2,s0y2y3y5y6).
in(3,s0y2y3y5y6).
in(5,s0y2y3y5y6).
in(6,s0y2y3y5y6).
in(0,s0y2y3y5y7).
in(2,s0y2y3y5y7).
in(3,s0y2y3y5y7).
in(5,s0y2y3y5y7).
in(7,s0y2y3y5y7).
in(0,s0y2y3y6y7).
in(2,s0y2y3y6y7).
in(3,s0y2y3y6y7).
in(6,s0y2y3y6y7).
in(7,s0y2y3y6y7).
in(0,s0y2y4y5y6).
in(2,s0y2y4y5y6).
in(4,s0y2y4y5y6).
in(5,s0y2y4y5y6).
in(6,s0y2y4y5y6).
in(0,s0y2y4y5y7).
in(2,s0y2y4y5y7).
in(4,s0y2y4y5y7).
in(5,s0y2y4y5y7).
in(7,s0y2y4y5y7).
in(0,s0y2y4y6y7).
in(2,s0y2y4y6y7).
in(4,s0y2y4y6y7).
in(6,s0y2y4y6y7).
in(7,s0y2y4y6y7).
in(0,s0y2y5y6y7).
in(2,s0y2y5y6y7).
in(5,s0y2y5y6y7).
in(6,s0y2y5y6y7).
in(7,s0y2y5y6y7).
in(0,s0y3y4y5y6).
in(3,s0y3y4y5y6).
in(4,s0y3y4y5y6).
in(5,s0y3y4y5y6).
in(6,s0y3y4y5y6).
in(0,s0y3y4y5y7).
in(3,s0y3y4y5y7).
in(4,s0y3y4y5y7).
in(5,s0y3y4y5y7).
in(7,s0y3y4y5y7).
in(0,s0y3y4y6y7).
in(3,s0y3y4y6y7).
in(4,s0y3y4y6y7).
in(6,s0y3y4y6y7).
in(7,s0y3y4y6y7).
in(0,s0y3y5y6y7).
in(3,s0y3y5y6y7).
in(5,s0y3y5y6y7).
in(6,s0y3y5y6y7).
in(7,s0y3y5y6y7).
in(0,s0y4y5y6y7).
in(4,s0y4y5y6y7).
in(5,s0y4y5y6y7).
in(6,s0y4y5y6y7).
in(7,s0y4y5y6y7).
in(1,s1y2y3y4y5).
in(2,s1y2y3y4y5).
in(3,s1y2y3y4y5).
in(4,s1y2y3y4y5).
in(5,s1y2y3y4y5).
in(1,s1y2y3y4y6).
in(2,s1y2y3y4y6).
in(3,s1y2y3y4y6).
in(4,s1y2y3y4y6).
in(6,s1y2y3y4y6).
in(1,s1y2y3y4y7).
in(2,s1y2y3y4y7).
in(3,s1y2y3y4y7).
in(4,s1y2y3y4y7).
in(7,s1y2y3y4y7).
in(1,s1y2y3y5y6).
in(2,s1y2y3y5y6).
in(3,s1y2y3y5y6).
in(5,s1y2y3y5y6).
in(6,s1y2y3y5y6).
in(1,s1y2y3y5y7).
in(2,s1y2y3y5y7).
in(3,s1y2y3y5y7).
in(5,s1y2y3y5y7).
in(7,s1y2y3y5y7).
in(1,s1y2y3y6y7).
in(2,s1y2y3y6y7).
in(3,s1y2y3y6y7).
in(6,s1y2y3y6y7).
in(7,s1y2y3y6y7).
in(1,s1y2y4y5y6).
in(2,s1y2y4y5y6).
in(4,s1y2y4y5y6).
in(5,s1y2y4y5y6).
in(6,s1y2y4y5y6).
in(1,s1y2y4y5y7).
in(2,s1y2y4y5y7).
in(4,s1y2y4y5y7).
in(5,s1y2y4y5y7).
in(7,s1y2y4y5y7).
in(1,s1y2y4y6y7).
in(2,s1y2y4y6y7).
in(4,s1y2y4y6y7).
in(6,s1y2y4y6y7).
in(7,s1y2y4y6y7).
in(1,s1y2y5y6y7).
in(2,s1y2y5y6y7).
in(5,s1y2y5y6y7).
in(6,s1y2y5y6y7).
in(7,s1y2y5y6y7).
in(1,s1y3y4y5y6).
in(3,s1y3y4y5y6).
in(4,s1y3y4y5y6).
in(5,s1y3y4y5y6).
in(6,s1y3y4y5y6).
in(1,s1y3y4y5y7).
in(3,s1y3y4y5y7).
in(4,s1y3y4y5y7).
in(5,s1y3y4y5y7).
in(7,s1y3y4y5y7).
in(1,s1y3y4y6y7).
in(3,s1y3y4y6y7).
in(4,s1y3y4y6y7).
in(6,s1y3y4y6y7).
in(7,s1y3y4y6y7).
in(1,s1y3y5y6y7).
in(3,s1y3y5y6y7).
in(5,s1y3y5y6y7).
in(6,s1y3y5y6y7).
in(7,s1y3y5y6y7).
in(1,s1y4y5y6y7).
in(4,s1y4y5y6y7).
in(5,s1y4y5y6y7).
in(6,s1y4y5y6y7).
in(7,s1y4y5y6y7).
in(2,s2y3y4y5y6).
in(3,s2y3y4y5y6).
in(4,s2y3y4y5y6).
in(5,s2y3y4y5y6).
in(6,s2y3y4y5y6).
in(2,s2y3y4y5y7).
in(3,s2y3y4y5y7).
in(4,s2y3y4y5y7).
in(5,s2y3y4y5y7).
in(7,s2y3y4y5y7).
in(2,s2y3y4y6y7).
in(3,s2y3y4y6y7).
in(4,s2y3y4y6y7).
in(6,s2y3y4y6y7).
in(7,s2y3y4y6y7).
in(2,s2y3y5y6y7).
in(3,s2y3y5y6y7).
in(5,s2y3y5y6y7).
in(6,s2y3y5y6y7).
in(7,s2y3y5y6y7).
in(2,s2y4y5y6y7).
in(4,s2y4y5y6y7).
in(5,s2y4y5y6y7).
in(6,s2y4y5y6y7).
in(7,s2y4y5y6y7).
in(3,s3y4y5y6y7).
in(4,s3y4y5y6y7).
in(5,s3y4y5y6y7).
in(6,s3y4y5y6y7).
in(7,s3y4y5y6y7).
in(0,s0y1y2y3y4y5).
in(1,s0y1y2y3y4y5).
in(2,s0y1y2y3y4y5).
in(3,s0y1y2y3y4y5).
in(4,s0y1y2y3y4y5).
in(5,s0y1y2y3y4y5).
in(0,s0y1y2y3y4y6).
in(1,s0y1y2y3y4y6).
in(2,s0y1y2y3y4y6).
in(3,s0y1y2y3y4y6).
in(4,s0y1y2y3y4y6).
in(6,s0y1y2y3y4y6).
in(0,s0y1y2y3y4y7).
in(1,s0y1y2y3y4y7).
in(2,s0y1y2y3y4y7).
in(3,s0y1y2y3y4y7).
in(4,s0y1y2y3y4y7).
in(7,s0y1y2y3y4y7).
in(0,s0y1y2y3y5y6).
in(1,s0y1y2y3y5y6).
in(2,s0y1y2y3y5y6).
in(3,s0y1y2y3y5y6).
in(5,s0y1y2y3y5y6).
in(6,s0y1y2y3y5y6).
in(0,s0y1y2y3y5y7).
in(1,s0y1y2y3y5y7).
in(2,s0y1y2y3y5y7).
in(3,s0y1y2y3y5y7).
in(5,s0y1y2y3y5y7).
in(7,s0y1y2y3y5y7).
in(0,s0y1y2y3y6y7).
in(1,s0y1y2y3y6y7).
in(2,s0y1y2y3y6y7).
in(3,s0y1y2y3y6y7).
in(6,s0y1y2y3y6y7).
in(7,s0y1y2y3y6y7).
in(0,s0y1y2y4y5y6).
in(1,s0y1y2y4y5y6).
in(2,s0y1y2y4y5y6).
in(4,s0y1y2y4y5y6).
in(5,s0y1y2y4y5y6).
in(6,s0y1y2y4y5y6).
in(0,s0y1y2y4y5y7).
in(1,s0y1y2y4y5y7).
in(2,s0y1y2y4y5y7).
in(4,s0y1y2y4y5y7).
in(5,s0y1y2y4y5y7).
in(7,s0y1y2y4y5y7).
in(0,s0y1y2y4y6y7).
in(1,s0y1y2y4y6y7).
in(2,s0y1y2y4y6y7).
in(4,s0y1y2y4y6y7).
in(6,s0y1y2y4y6y7).
in(7,s0y1y2y4y6y7).
in(0,s0y1y2y5y6y7).
in(1,s0y1y2y5y6y7).
in(2,s0y1y2y5y6y7).
in(5,s0y1y2y5y6y7).
in(6,s0y1y2y5y6y7).
in(7,s0y1y2y5y6y7).
in(0,s0y1y3y4y5y6).
in(1,s0y1y3y4y5y6).
in(3,s0y1y3y4y5y6).
in(4,s0y1y3y4y5y6).
in(5,s0y1y3y4y5y6).
in(6,s0y1y3y4y5y6).
in(0,s0y1y3y4y5y7).
in(1,s0y1y3y4y5y7).
in(3,s0y1y3y4y5y7).
in(4,s0y1y3y4y5y7).
in(5,s0y1y3y4y5y7).
in(7,s0y1y3y4y5y7).
in(0,s0y1y3y4y6y7).
in(1,s0y1y3y4y6y7).
in(3,s0y1y3y4y6y7).
in(4,s0y1y3y4y6y7).
in(6,s0y1y3y4y6y7).
in(7,s0y1y3y4y6y7).
in(0,s0y1y3y5y6y7).
in(1,s0y1y3y5y6y7).
in(3,s0y1y3y5y6y7).
in(5,s0y1y3y5y6y7).
in(6,s0y1y3y5y6y7).
in(7,s0y1y3y5y6y7).
in(0,s0y1y4y5y6y7).
in(1,s0y1y4y5y6y7).
in(4,s0y1y4y5y6y7).
in(5,s0y1y4y5y6y7).
in(6,s0y1y4y5y6y7).
in(7,s0y1y4y5y6y7).
in(0,s0y2y3y4y5y6).
in(2,s0y2y3y4y5y6).
in(3,s0y2y3y4y5y6).
in(4,s0y2y3y4y5y6).
in(5,s0y2y3y4y5y6).
in(6,s0y2y3y4y5y6).
in(0,s0y2y3y4y5y7).
in(2,s0y2y3y4y5y7).
in(3,s0y2y3y4y5y7).
in(4,s0y2y3y4y5y7).
in(5,s0y2y3y4y5y7).
in(7,s0y2y3y4y5y7).
in(0,s0y2y3y4y6y7).
in(2,s0y2y3y4y6y7).
in(3,s0y2y3y4y6y7).
in(4,s0y2y3y4y6y7).
in(6,s0y2y3y4y6y7).
in(7,s0y2y3y4y6y7).
in(0,s0y2y3y5y6y7).
in(2,s0y2y3y5y6y7).
in(3,s0y2y3y5y6y7).
in(5,s0y2y3y5y6y7).
in(6,s0y2y3y5y6y7).
in(7,s0y2y3y5y6y7).
in(0,s0y2y4y5y6y7).
in(2,s0y2y4y5y6y7).
in(4,s0y2y4y5y6y7).
in(5,s0y2y4y5y6y7).
in(6,s0y2y4y5y6y7).
in(7,s0y2y4y5y6y7).
in(0,s0y3y4y5y6y7).
in(3,s0y3y4y5y6y7).
in(4,s0y3y4y5y6y7).
in(5,s0y3y4y5y6y7).
in(6,s0y3y4y5y6y7).
in(7,s0y3y4y5y6y7).
in(1,s1y2y3y4y5y6).
in(2,s1y2y3y4y5y6).
in(3,s1y2y3y4y5y6).
in(4,s1y2y3y4y5y6).
in(5,s1y2y3y4y5y6).
in(6,s1y2y3y4y5y6).
in(1,s1y2y3y4y5y7).
in(2,s1y2y3y4y5y7).
in(3,s1y2y3y4y5y7).
in(4,s1y2y3y4y5y7).
in(5,s1y2y3y4y5y7).
in(7,s1y2y3y4y5y7).
in(1,s1y2y3y4y6y7).
in(2,s1y2y3y4y6y7).
in(3,s1y2y3y4y6y7).
in(4,s1y2y3y4y6y7).
in(6,s1y2y3y4y6y7).
in(7,s1y2y3y4y6y7).
in(1,s1y2y3y5y6y7).
in(2,s1y2y3y5y6y7).
in(3,s1y2y3y5y6y7).
in(5,s1y2y3y5y6y7).
in(6,s1y2y3y5y6y7).
in(7,s1y2y3y5y6y7).
in(1,s1y2y4y5y6y7).
in(2,s1y2y4y5y6y7).
in(4,s1y2y4y5y6y7).
in(5,s1y2y4y5y6y7).
in(6,s1y2y4y5y6y7).
in(7,s1y2y4y5y6y7).
in(1,s1y3y4y5y6y7).
in(3,s1y3y4y5y6y7).
in(4,s1y3y4y5y6y7).
in(5,s1y3y4y5y6y7).
in(6,s1y3y4y5y6y7).
in(7,s1y3y4y5y6y7).
in(2,s2y3y4y5y6y7).
in(3,s2y3y4y5y6y7).
in(4,s2y3y4y5y6y7).
in(5,s2y3y4y5y6y7).
in(6,s2y3y4y5y6y7).
in(7,s2y3y4y5y6y7).
qpair(0,1,s3y5y7).
qpair(0,1,s3y4).
qpair(0,1,s5).
qpair(0,1,s3y7).
qpair(0,1,s4).
qpair(0,1,s2y3).
qpair(0,1,s7).
qpair(0,1,s2y3y4).
qpair(0,1,s3).
qpair(0,1,s3y5).
qpair(0,2,s6).
qpair(0,3,s2).
qpair(0,3,s5).
qpair(0,3,s1y5y7).
qpair(0,3,s5y7).
qpair(0,3,s2y5y7).
qpair(0,3,s1y2y5).
qpair(0,3,s2y4y5y7).
qpair(0,3,s2y5).
qpair(0,3,s1y4y5).
qpair(0,3,s2y4y5).
qpair(0,3,s1y2y5y7).
qpair(0,3,s4).
qpair(0,3,s1).
qpair(0,3,s7).
qpair(0,3,s4y5).
qpair(0,3,s1y2y4y5y7).
qpair(0,3,s1y2y4).
qpair(0,3,s1y2y4y5).
qpair(0,3,s1y4y5y7).
qpair(0,3,s1y2y7).
qpair(0,3,s2y4).
qpair(0,3,s1y2).
qpair(0,3,s2y7).
qpair(0,3,s1y5).
qpair(0,3,s1y4y7).
qpair(0,3,s2y4y7).
qpair(0,3,s4y7).
qpair(0,3,s6).
qpair(0,3,s4y5y7).
qpair(0,3,s1y2y4y7).
qpair(0,3,s1y4).
qpair(0,3,s1y7).
qpair(0,3,empty).
qpair(0,5,s6).
qpair(0,5,s2).
qpair(0,5,s2y3y6).
qpair(0,5,s3y7).
qpair(0,5,s4).
qpair(0,5,s2y3).
qpair(0,5,s1).
qpair(0,5,s7).
qpair(0,5,s2y6).
qpair(0,5,s3y6).
qpair(0,5,empty).
qpair(0,5,s3).
qpair(0,6,s3y5y7).
qpair(0,6,s2).
qpair(0,6,s5y7).
qpair(0,6,s1).
qpair(0,6,s3y5).
qpair(0,7,s6).
qpair(0,7,s2).
qpair(0,7,s3y4).
qpair(0,7,s5).
qpair(0,7,s4).
qpair(0,7,s1).
qpair(0,7,empty).
qpair(0,7,s3).
qpair(0,7,s3y5).
qpair(1,2,s6).
qpair(1,2,s3y4).
qpair(1,2,s5).
qpair(1,2,s4y6).
qpair(1,2,s4).
qpair(1,2,s7).
qpair(1,2,s4y5).
qpair(1,2,s3y4y5).
qpair(1,2,s3y6).
qpair(1,2,s0).
qpair(1,2,empty).
qpair(1,2,s3).
qpair(1,2,s3y5).
qpair(1,3,s2).
qpair(1,3,s0y5y7).
qpair(1,3,s5).
qpair(1,3,s4y6).
qpair(1,3,s5y7).
qpair(1,3,s2y5y7).
qpair(1,3,s0y2).
qpair(1,3,s0y5).
qpair(1,3,s2y4y5y7).
qpair(1,3,s2y5).
qpair(1,3,s0y2y7).
qpair(1,3,s2y4y5).
qpair(1,3,s0y2y4).
qpair(1,3,s0y4y7).
qpair(1,3,s4).
qpair(1,3,s7).
qpair(1,3,s4y5).
qpair(1,3,s0y7).
qpair(1,3,s2y4).
qpair(1,3,s0y4).
qpair(1,3,s2y7).
qpair(1,3,s0y2y4y5).
qpair(1,3,s0y2y4y5y7).
qpair(1,3,s0y2y5y7).
qpair(1,3,s2y4y7).
qpair(1,3,s0).
qpair(1,3,s4y7).
qpair(1,3,s6).
qpair(1,3,s0y4y5y7).
qpair(1,3,s4y5y7).
qpair(1,3,s2y6).
qpair(1,3,s0y2y5).
qpair(1,3,s0y2y4y7).
qpair(1,3,empty).
qpair(1,3,s0y4y5).
qpair(1,4,s6).
qpair(1,4,s2).
qpair(1,4,s2y7).
qpair(1,4,s5).
qpair(1,4,s3y7).
qpair(1,4,s2y3).
qpair(1,4,s7).
qpair(1,4,s2y6).
qpair(1,4,s3y6).
qpair(1,4,s2y3y7).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,5,s7).
qpair(1,6,s2y4).
qpair(1,6,s2).
qpair(1,6,s3y4).
qpair(1,6,s5).
qpair(1,6,s4).
qpair(1,6,s7).
qpair(1,6,s0).
qpair(1,6,empty).
qpair(1,6,s3).
qpair(1,7,s6).
qpair(1,7,s5).
qpair(2,3,s0y1y5y7).
qpair(2,3,s0y5y7).
qpair(2,3,s0y1y4y5).
qpair(2,3,s5).
qpair(2,3,s1y5y7).
qpair(2,3,s5y7).
qpair(2,3,s0y5).
qpair(2,3,s0y1y4y5y7).
qpair(2,3,s1y4y5).
qpair(2,3,s0y4y7).
qpair(2,3,s4).
qpair(2,3,s0y1y4y7).
qpair(2,3,s1).
qpair(2,3,s7).
qpair(2,3,s4y5).
qpair(2,3,s0y1y5).
qpair(2,3,s1y4y5y7).
qpair(2,3,s0y1).
qpair(2,3,s0y7).
qpair(2,3,s0y4).
qpair(2,3,s1y4y7).
qpair(2,3,s1y5).
qpair(2,3,s0).
qpair(2,3,s4y7).
qpair(2,3,s6).
qpair(2,3,s0y1y4).
qpair(2,3,s0y1y7).
qpair(2,3,s4y5y7).
qpair(2,3,s0y4y5y7).
qpair(2,3,s1y4).
qpair(2,3,s1y7).
qpair(2,3,empty).
qpair(2,3,s0y4y5).
qpair(2,4,s1y3).
qpair(2,4,s5).
qpair(2,4,s1y5).
qpair(2,4,s3y7).
qpair(2,4,s1y3y5).
qpair(2,4,s1).
qpair(2,4,s7).
qpair(2,4,s1y7).
qpair(2,4,s1y3y7).
qpair(2,4,s0).
qpair(2,4,empty).
qpair(2,4,s3).
qpair(2,4,s3y5).
qpair(2,5,s6).
qpair(2,5,s1y3).
qpair(2,5,s3y4).
qpair(2,5,s0y3).
qpair(2,5,s4).
qpair(2,5,s0y6).
qpair(2,5,s1y4).
qpair(2,5,s1).
qpair(2,5,s7).
qpair(2,5,s0y3y6).
qpair(2,5,s1y3y4).
qpair(2,5,s3y6).
qpair(2,5,s0).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(2,6,s1y3y4y5).
qpair(2,6,s1y3y5).
qpair(2,6,s1y5).
qpair(2,6,s5).
qpair(2,6,s4y5).
qpair(2,6,s3y4y5).
qpair(2,6,s1y4y5).
qpair(2,6,s3y5).
qpair(3,4,s0y1y5y7).
qpair(3,4,s2).
qpair(3,4,s0y5y7).
qpair(3,4,s0y1y2y5y7).
qpair(3,4,s5).
qpair(3,4,s1y5y7).
qpair(3,4,s5y7).
qpair(3,4,s2y5y7).
qpair(3,4,s0y2).
qpair(3,4,s0y5).
qpair(3,4,s1y2y5).
qpair(3,4,s2y5).
qpair(3,4,s0y2y7).
qpair(3,4,s0y1y2y7).
qpair(3,4,s1y2y5y7).
qpair(3,4,s1).
qpair(3,4,s7).
qpair(3,4,s0y1y5).
qpair(3,4,s0y1y2).
qpair(3,4,s0y1).
qpair(3,4,s0y7).
qpair(3,4,s1y2y7).
qpair(3,4,s1y2).
qpair(3,4,s2y7).
qpair(3,4,s1y5).
qpair(3,4,s0y2y5y7).
qpair(3,4,s0).
qpair(3,4,s6).
qpair(3,4,s0y1y7).
qpair(3,4,s1y7).
qpair(3,4,s0y2y5).
qpair(3,4,empty).
qpair(3,4,s0y1y2y5).
qpair(3,5,s2).
qpair(3,5,s0y2).
qpair(3,5,s0y2y7).
qpair(3,5,s0y2y4).
qpair(3,5,s0y4y7).
qpair(3,5,s0y1y2y7).
qpair(3,5,s0y1y2y4).
qpair(3,5,s4).
qpair(3,5,s0y1y4y7).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s0y1y2).
qpair(3,5,s1y2y4).
qpair(3,5,s0y1).
qpair(3,5,s0y7).
qpair(3,5,s2y4).
qpair(3,5,s1y2).
qpair(3,5,s0y4).
qpair(3,5,s1y2y7).
qpair(3,5,s2y7).
qpair(3,5,s1y4y7).
qpair(3,5,s0y1y2y4y7).
qpair(3,5,s2y4y7).
qpair(3,5,s0y2y6).
qpair(3,5,s0).
qpair(3,5,s4y7).
qpair(3,5,s6).
qpair(3,5,s0y1y4).
qpair(3,5,s0y1y7).
qpair(3,5,s1y2y4y7).
qpair(3,5,s1y4).
qpair(3,5,s0y6).
qpair(3,5,s1y7).
qpair(3,5,s2y6).
qpair(3,5,s0y2y4y7).
qpair(3,5,empty).
qpair(3,6,s0y1y5y7).
qpair(3,6,s0y1y4y5).
qpair(3,6,s0y1y2y5y7).
qpair(3,6,s5y7).
qpair(3,6,s2y5y7).
qpair(3,6,s0y2).
qpair(3,6,s0y1y2y4y5).
qpair(3,6,s0y1y4y5y7).
qpair(3,6,s2y4y5y7).
qpair(3,6,s2y5).
qpair(3,6,s2y4y5).
qpair(3,6,s0y2y4).
qpair(3,6,s0y1y2y7).
qpair(3,6,s0y1y2y4).
qpair(3,6,s0y1y4y7).
qpair(3,6,s4y5).
qpair(3,6,s0y1y2).
qpair(3,6,s1y2y4y5y7).
qpair(3,6,s0y1y2y4y5y7).
qpair(3,6,s0y1).
qpair(3,6,s1y2y7).
qpair(3,6,s2y4).
qpair(3,6,s1y2).
qpair(3,6,s0y1y2y4y7).
qpair(3,6,s2y7).
qpair(3,6,s1y5).
qpair(3,6,s1y4y7).
qpair(3,6,s0y2y4y5).
qpair(3,6,s0y2y4y5y7).
qpair(3,6,s2y4y7).
qpair(3,6,s4y7).
qpair(3,6,s0y1y4).
qpair(3,6,s0y1y7).
qpair(3,6,s4y5y7).
qpair(3,6,s1y2y4y7).
qpair(3,6,s1y4).
qpair(3,6,s1y7).
qpair(3,6,s0y2y5).
qpair(3,6,s0y2y4y7).
qpair(3,6,s0y1y2y5).
qpair(3,7,s2).
qpair(3,7,s0y1y4y5).
qpair(3,7,s5).
qpair(3,7,s0y2).
qpair(3,7,s0y5).
qpair(3,7,s1y2y5).
qpair(3,7,s0y1y2y4y5).
qpair(3,7,s2y5).
qpair(3,7,s1y4y5).
qpair(3,7,s2y4y5).
qpair(3,7,s0y2y4).
qpair(3,7,s0y1y2y4).
qpair(3,7,s4).
qpair(3,7,s1).
qpair(3,7,s4y5).
qpair(3,7,s0y1y5).
qpair(3,7,s0y1y2).
qpair(3,7,s1y2y4).
qpair(3,7,s1y2y4y5).
qpair(3,7,s0y1).
qpair(3,7,s2y4).
qpair(3,7,s1y2).
qpair(3,7,s0y4).
qpair(3,7,s0y2y4y5).
qpair(3,7,s1y5).
qpair(3,7,s0).
qpair(3,7,s6).
qpair(3,7,s0y1y4).
qpair(3,7,s1y4).
qpair(3,7,s0y2y5).
qpair(3,7,empty).
qpair(3,7,s0y4y5).
qpair(3,7,s0y1y2y5).
qpair(4,7,s1y3).
qpair(4,7,s2).
qpair(4,7,s1y2).
qpair(4,7,s5).
qpair(4,7,s0y3).
qpair(4,7,s1y2y3).
qpair(4,7,s2y3).
qpair(4,7,s1).
qpair(4,7,s0).
qpair(4,7,empty).
qpair(4,7,s3).
qpair(5,6,s2).
qpair(5,6,s0y3).
qpair(5,6,s0y2y3).
qpair(5,6,s4).
qpair(5,6,s2y3).
qpair(5,6,s1).
qpair(5,6,s0y2).
qpair(5,6,s0).
qpair(5,6,empty).
qpair(5,6,s3).
qpair(0,1,s2y4).
qpair(0,1,s2).
qpair(0,1,s5y7).
qpair(0,6,s3y7).
qpair(1,6,s2y3).
qpair(2,6,s1y3).
qpair(2,6,s3y4).
qpair(2,6,s1y4).
qpair(2,6,s4).
qpair(2,6,s1y3y4).
qpair(2,6,s1).
qpair(2,6,s3).
qpair(3,6,s0y7).
qpair(3,6,s0y2y7).
qpair(3,6,s0y5y7).
qpair(3,6,s0y4).
qpair(3,6,s1y4y5y7).
qpair(3,6,s0y4y5y7).
qpair(3,6,s0y4y7).
qpair(3,6,s1y5y7).
qpair(3,6,s1y2y5y7).
qpair(3,6,s0y2y5y7).
qpair(3,6,s0y1y5).
qpair(3,6,s0y5).
qpair(3,6,s1y2y5).
qpair(3,6,s1y2y4y5).
qpair(3,6,s1y4y5).
qpair(3,6,s0y4y5).
qpair(3,6,s1y2y4).
qpair(4,6,empty).
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

% ===== Base program loaded from: results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked/wc_sweep_8_2026_er_sf_alpha001_nowrong_noweight_seed2026_2058_20260715_115147/rep29/lps/abapc_inc_full_base_rep29_optN.lp =====
% ===== Base program sha256 (raw): e2083c291d1981d04e9f8bc6711ba6adb028176ac910bc6fd8ad07047afbd756 =====
% ===== Base program sha256 (normalized): aa0dac80eddc42f8aa54c5821edd23ad03fb86e25cd87d7e885b7b441dfa2173 =====
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
{mus(232)}.
{mus(233)}.
{mus(234)}.
{mus(235)}.
{mus(236)}.
{mus(237)}.
{mus(238)}.
{mus(239)}.
{mus(240)}.
{mus(241)}.
{mus(242)}.
{mus(243)}.
{mus(244)}.
{mus(245)}.
{mus(246)}.
{mus(247)}.
{mus(248)}.
{mus(249)}.
{mus(250)}.
{mus(251)}.
{mus(252)}.
{mus(253)}.
{mus(254)}.
{mus(255)}.
{mus(256)}.
{mus(257)}.
{mus(258)}.
{mus(259)}.
{mus(260)}.
{mus(261)}.
{mus(262)}.
{mus(263)}.
{mus(264)}.
{mus(265)}.
{mus(266)}.
{mus(267)}.
{mus(268)}.
{mus(269)}.
{mus(270)}.
{mus(271)}.
{mus(272)}.
{mus(273)}.
{mus(274)}.
{mus(275)}.
{mus(276)}.
{mus(277)}.
{mus(278)}.
{mus(279)}.
{mus(280)}.
{mus(281)}.
{mus(282)}.
{mus(283)}.
{mus(284)}.
{mus(285)}.
{mus(286)}.
{mus(287)}.
{mus(288)}.
{mus(289)}.
{mus(290)}.
{mus(291)}.
{mus(292)}.
{mus(293)}.
{mus(294)}.
{mus(295)}.
{mus(296)}.
{mus(297)}.
{mus(298)}.
{mus(299)}.
{mus(300)}.
{mus(301)}.
{mus(302)}.
{mus(303)}.
{mus(304)}.
{mus(305)}.
{mus(306)}.
{mus(307)}.
{mus(308)}.
{mus(309)}.
{mus(310)}.
{mus(311)}.
{mus(312)}.
{mus(313)}.
{mus(314)}.
{mus(315)}.
{mus(316)}.
{mus(317)}.
{mus(318)}.
{mus(319)}.
{mus(320)}.
{mus(321)}.
{mus(322)}.
{mus(323)}.
{mus(324)}.
{mus(325)}.
{mus(326)}.
{mus(327)}.
{mus(328)}.
{mus(329)}.
{mus(330)}.
{mus(331)}.
{mus(332)}.
{mus(333)}.
{mus(334)}.
{mus(335)}.
{mus(336)}.
{mus(337)}.
{mus(338)}.
{mus(339)}.
{mus(340)}.
{mus(341)}.
{mus(342)}.
{mus(343)}.
{mus(344)}.
{mus(345)}.
{mus(346)}.
{mus(347)}.
{mus(348)}.
{mus(349)}.
{mus(350)}.
{mus(351)}.
{mus(352)}.
{mus(353)}.
{mus(354)}.
{mus(355)}.
{mus(356)}.
{mus(357)}.
{mus(358)}.
{mus(359)}.
{mus(360)}.
{mus(361)}.
{mus(362)}.
{mus(363)}.
{mus(364)}.
{mus(365)}.
{mus(366)}.
{mus(367)}.
{mus(368)}.
{mus(369)}.
{mus(370)}.
{mus(371)}.
{mus(372)}.
{mus(373)}.
{mus(374)}.
{mus(375)}.
{mus(376)}.
{mus(377)}.
{mus(378)}.
{mus(379)}.
{mus(380)}.
{mus(381)}.
{mus(382)}.
{mus(383)}.
{mus(384)}.
{mus(385)}.
{mus(386)}.
{mus(387)}.
{mus(388)}.
{mus(389)}.
{mus(390)}.
{mus(391)}.
{mus(392)}.
{mus(393)}.
{mus(394)}.
{mus(395)}.
{mus(396)}.
{mus(397)}.
{mus(398)}.
{mus(399)}.
{mus(400)}.
{mus(401)}.
{mus(402)}.
{mus(403)}.
{mus(404)}.
{mus(405)}.

% ===== Adorned Facts =====
ext_indep(0,1,s2):-mus(1).
ext_dep(0,1,s3):-mus(2).
ext_dep(0,1,s4):-mus(3).
ext_dep(0,1,s5):-mus(4).
ext_dep(0,1,s7):-mus(5).
ext_dep(0,1,s2y3):-mus(6).
ext_indep(0,1,s2y4):-mus(7).
ext_dep(0,1,s3y4):-mus(8).
ext_dep(0,1,s3y5):-mus(9).
ext_dep(0,1,s3y7):-mus(10).
ext_indep(0,1,s5y7):-mus(11).
ext_dep(0,1,s2y3y4):-mus(12).
ext_dep(0,1,s3y5y7):-mus(13).
ext_dep(0,2,s6):-mus(14).
ext_dep(0,3,empty):-mus(15).
ext_dep(0,3,s1):-mus(16).
ext_dep(0,3,s2):-mus(17).
ext_dep(0,3,s4):-mus(18).
ext_dep(0,3,s5):-mus(19).
ext_dep(0,3,s6):-mus(20).
ext_dep(0,3,s7):-mus(21).
ext_dep(0,3,s1y2):-mus(22).
ext_dep(0,3,s1y4):-mus(23).
ext_dep(0,3,s1y5):-mus(24).
ext_dep(0,3,s1y7):-mus(25).
ext_dep(0,3,s2y4):-mus(26).
ext_dep(0,3,s2y5):-mus(27).
ext_dep(0,3,s2y7):-mus(28).
ext_dep(0,3,s4y5):-mus(29).
ext_dep(0,3,s4y7):-mus(30).
ext_dep(0,3,s5y7):-mus(31).
ext_dep(0,3,s1y2y4):-mus(32).
ext_dep(0,3,s1y2y5):-mus(33).
ext_dep(0,3,s1y2y7):-mus(34).
ext_dep(0,3,s1y4y5):-mus(35).
ext_dep(0,3,s1y4y7):-mus(36).
ext_dep(0,3,s1y5y7):-mus(37).
ext_dep(0,3,s2y4y5):-mus(38).
ext_dep(0,3,s2y4y7):-mus(39).
ext_dep(0,3,s2y5y7):-mus(40).
ext_dep(0,3,s4y5y7):-mus(41).
ext_dep(0,3,s1y2y4y5):-mus(42).
ext_dep(0,3,s1y2y4y7):-mus(43).
ext_dep(0,3,s1y2y5y7):-mus(44).
ext_dep(0,3,s1y4y5y7):-mus(45).
ext_dep(0,3,s2y4y5y7):-mus(46).
ext_dep(0,3,s1y2y4y5y7):-mus(47).
ext_dep(0,5,empty):-mus(48).
ext_dep(0,5,s1):-mus(49).
ext_dep(0,5,s2):-mus(50).
ext_dep(0,5,s3):-mus(51).
ext_dep(0,5,s4):-mus(52).
ext_dep(0,5,s6):-mus(53).
ext_dep(0,5,s7):-mus(54).
ext_dep(0,5,s2y3):-mus(55).
ext_dep(0,5,s2y6):-mus(56).
ext_dep(0,5,s3y6):-mus(57).
ext_dep(0,5,s3y7):-mus(58).
ext_dep(0,5,s2y3y6):-mus(59).
ext_dep(0,6,s1):-mus(60).
ext_dep(0,6,s2):-mus(61).
ext_dep(0,6,s3y5):-mus(62).
ext_indep(0,6,s3y7):-mus(63).
ext_dep(0,6,s5y7):-mus(64).
ext_dep(0,6,s3y5y7):-mus(65).
ext_dep(0,7,empty):-mus(66).
ext_dep(0,7,s1):-mus(67).
ext_dep(0,7,s2):-mus(68).
ext_dep(0,7,s3):-mus(69).
ext_dep(0,7,s4):-mus(70).
ext_dep(0,7,s5):-mus(71).
ext_dep(0,7,s6):-mus(72).
ext_dep(0,7,s3y4):-mus(73).
ext_dep(0,7,s3y5):-mus(74).
ext_dep(1,2,empty):-mus(75).
ext_dep(1,2,s0):-mus(76).
ext_dep(1,2,s3):-mus(77).
ext_dep(1,2,s4):-mus(78).
ext_dep(1,2,s5):-mus(79).
ext_dep(1,2,s6):-mus(80).
ext_dep(1,2,s7):-mus(81).
ext_dep(1,2,s3y4):-mus(82).
ext_dep(1,2,s3y5):-mus(83).
ext_dep(1,2,s3y6):-mus(84).
ext_dep(1,2,s4y5):-mus(85).
ext_dep(1,2,s4y6):-mus(86).
ext_dep(1,2,s3y4y5):-mus(87).
ext_dep(1,3,empty):-mus(88).
ext_dep(1,3,s0):-mus(89).
ext_dep(1,3,s2):-mus(90).
ext_dep(1,3,s4):-mus(91).
ext_dep(1,3,s5):-mus(92).
ext_dep(1,3,s6):-mus(93).
ext_dep(1,3,s7):-mus(94).
ext_dep(1,3,s0y2):-mus(95).
ext_dep(1,3,s0y4):-mus(96).
ext_dep(1,3,s0y5):-mus(97).
ext_dep(1,3,s0y7):-mus(98).
ext_dep(1,3,s2y4):-mus(99).
ext_dep(1,3,s2y5):-mus(100).
ext_dep(1,3,s2y6):-mus(101).
ext_dep(1,3,s2y7):-mus(102).
ext_dep(1,3,s4y5):-mus(103).
ext_dep(1,3,s4y6):-mus(104).
ext_dep(1,3,s4y7):-mus(105).
ext_dep(1,3,s5y7):-mus(106).
ext_dep(1,3,s0y2y4):-mus(107).
ext_dep(1,3,s0y2y5):-mus(108).
ext_dep(1,3,s0y2y7):-mus(109).
ext_dep(1,3,s0y4y5):-mus(110).
ext_dep(1,3,s0y4y7):-mus(111).
ext_dep(1,3,s0y5y7):-mus(112).
ext_dep(1,3,s2y4y5):-mus(113).
ext_dep(1,3,s2y4y7):-mus(114).
ext_dep(1,3,s2y5y7):-mus(115).
ext_dep(1,3,s4y5y7):-mus(116).
ext_dep(1,3,s0y2y4y5):-mus(117).
ext_dep(1,3,s0y2y4y7):-mus(118).
ext_dep(1,3,s0y2y5y7):-mus(119).
ext_dep(1,3,s0y4y5y7):-mus(120).
ext_dep(1,3,s2y4y5y7):-mus(121).
ext_dep(1,3,s0y2y4y5y7):-mus(122).
ext_dep(1,4,empty):-mus(123).
ext_dep(1,4,s2):-mus(124).
ext_dep(1,4,s3):-mus(125).
ext_dep(1,4,s5):-mus(126).
ext_dep(1,4,s6):-mus(127).
ext_dep(1,4,s7):-mus(128).
ext_dep(1,4,s2y3):-mus(129).
ext_dep(1,4,s2y6):-mus(130).
ext_dep(1,4,s2y7):-mus(131).
ext_dep(1,4,s3y6):-mus(132).
ext_dep(1,4,s3y7):-mus(133).
ext_dep(1,4,s2y3y7):-mus(134).
ext_dep(1,5,s7):-mus(135).
ext_dep(1,6,empty):-mus(136).
ext_dep(1,6,s0):-mus(137).
ext_dep(1,6,s2):-mus(138).
ext_dep(1,6,s3):-mus(139).
ext_dep(1,6,s4):-mus(140).
ext_dep(1,6,s5):-mus(141).
ext_dep(1,6,s7):-mus(142).
ext_indep(1,6,s2y3):-mus(143).
ext_dep(1,6,s2y4):-mus(144).
ext_dep(1,6,s3y4):-mus(145).
ext_dep(1,7,s5):-mus(146).
ext_dep(1,7,s6):-mus(147).
ext_dep(2,3,empty):-mus(148).
ext_dep(2,3,s0):-mus(149).
ext_dep(2,3,s1):-mus(150).
ext_dep(2,3,s4):-mus(151).
ext_dep(2,3,s5):-mus(152).
ext_dep(2,3,s6):-mus(153).
ext_dep(2,3,s7):-mus(154).
ext_dep(2,3,s0y1):-mus(155).
ext_dep(2,3,s0y4):-mus(156).
ext_dep(2,3,s0y5):-mus(157).
ext_dep(2,3,s0y7):-mus(158).
ext_dep(2,3,s1y4):-mus(159).
ext_dep(2,3,s1y5):-mus(160).
ext_dep(2,3,s1y7):-mus(161).
ext_dep(2,3,s4y5):-mus(162).
ext_dep(2,3,s4y7):-mus(163).
ext_dep(2,3,s5y7):-mus(164).
ext_dep(2,3,s0y1y4):-mus(165).
ext_dep(2,3,s0y1y5):-mus(166).
ext_dep(2,3,s0y1y7):-mus(167).
ext_dep(2,3,s0y4y5):-mus(168).
ext_dep(2,3,s0y4y7):-mus(169).
ext_dep(2,3,s0y5y7):-mus(170).
ext_dep(2,3,s1y4y5):-mus(171).
ext_dep(2,3,s1y4y7):-mus(172).
ext_dep(2,3,s1y5y7):-mus(173).
ext_dep(2,3,s4y5y7):-mus(174).
ext_dep(2,3,s0y1y4y5):-mus(175).
ext_dep(2,3,s0y1y4y7):-mus(176).
ext_dep(2,3,s0y1y5y7):-mus(177).
ext_dep(2,3,s0y4y5y7):-mus(178).
ext_dep(2,3,s1y4y5y7):-mus(179).
ext_dep(2,3,s0y1y4y5y7):-mus(180).
ext_dep(2,4,empty):-mus(181).
ext_dep(2,4,s0):-mus(182).
ext_dep(2,4,s1):-mus(183).
ext_dep(2,4,s3):-mus(184).
ext_dep(2,4,s5):-mus(185).
ext_dep(2,4,s7):-mus(186).
ext_dep(2,4,s1y3):-mus(187).
ext_dep(2,4,s1y5):-mus(188).
ext_dep(2,4,s1y7):-mus(189).
ext_dep(2,4,s3y5):-mus(190).
ext_dep(2,4,s3y7):-mus(191).
ext_dep(2,4,s1y3y5):-mus(192).
ext_dep(2,4,s1y3y7):-mus(193).
ext_dep(2,5,empty):-mus(194).
ext_dep(2,5,s0):-mus(195).
ext_dep(2,5,s1):-mus(196).
ext_dep(2,5,s3):-mus(197).
ext_dep(2,5,s4):-mus(198).
ext_dep(2,5,s6):-mus(199).
ext_dep(2,5,s7):-mus(200).
ext_dep(2,5,s0y3):-mus(201).
ext_dep(2,5,s0y6):-mus(202).
ext_dep(2,5,s1y3):-mus(203).
ext_dep(2,5,s1y4):-mus(204).
ext_dep(2,5,s3y4):-mus(205).
ext_dep(2,5,s3y6):-mus(206).
ext_dep(2,5,s0y3y6):-mus(207).
ext_dep(2,5,s1y3y4):-mus(208).
ext_indep(2,6,s1):-mus(209).
ext_indep(2,6,s3):-mus(210).
ext_indep(2,6,s4):-mus(211).
ext_dep(2,6,s5):-mus(212).
ext_indep(2,6,s1y3):-mus(213).
ext_indep(2,6,s1y4):-mus(214).
ext_dep(2,6,s1y5):-mus(215).
ext_indep(2,6,s3y4):-mus(216).
ext_dep(2,6,s3y5):-mus(217).
ext_dep(2,6,s4y5):-mus(218).
ext_indep(2,6,s1y3y4):-mus(219).
ext_dep(2,6,s1y3y5):-mus(220).
ext_dep(2,6,s1y4y5):-mus(221).
ext_dep(2,6,s3y4y5):-mus(222).
ext_dep(2,6,s1y3y4y5):-mus(223).
ext_dep(3,4,empty):-mus(224).
ext_dep(3,4,s0):-mus(225).
ext_dep(3,4,s1):-mus(226).
ext_dep(3,4,s2):-mus(227).
ext_dep(3,4,s5):-mus(228).
ext_dep(3,4,s6):-mus(229).
ext_dep(3,4,s7):-mus(230).
ext_dep(3,4,s0y1):-mus(231).
ext_dep(3,4,s0y2):-mus(232).
ext_dep(3,4,s0y5):-mus(233).
ext_dep(3,4,s0y7):-mus(234).
ext_dep(3,4,s1y2):-mus(235).
ext_dep(3,4,s1y5):-mus(236).
ext_dep(3,4,s1y7):-mus(237).
ext_dep(3,4,s2y5):-mus(238).
ext_dep(3,4,s2y7):-mus(239).
ext_dep(3,4,s5y7):-mus(240).
ext_dep(3,4,s0y1y2):-mus(241).
ext_dep(3,4,s0y1y5):-mus(242).
ext_dep(3,4,s0y1y7):-mus(243).
ext_dep(3,4,s0y2y5):-mus(244).
ext_dep(3,4,s0y2y7):-mus(245).
ext_dep(3,4,s0y5y7):-mus(246).
ext_dep(3,4,s1y2y5):-mus(247).
ext_dep(3,4,s1y2y7):-mus(248).
ext_dep(3,4,s1y5y7):-mus(249).
ext_dep(3,4,s2y5y7):-mus(250).
ext_dep(3,4,s0y1y2y5):-mus(251).
ext_dep(3,4,s0y1y2y7):-mus(252).
ext_dep(3,4,s0y1y5y7):-mus(253).
ext_dep(3,4,s0y2y5y7):-mus(254).
ext_dep(3,4,s1y2y5y7):-mus(255).
ext_dep(3,4,s0y1y2y5y7):-mus(256).
ext_dep(3,5,empty):-mus(257).
ext_dep(3,5,s0):-mus(258).
ext_dep(3,5,s1):-mus(259).
ext_dep(3,5,s2):-mus(260).
ext_dep(3,5,s4):-mus(261).
ext_dep(3,5,s6):-mus(262).
ext_dep(3,5,s7):-mus(263).
ext_dep(3,5,s0y1):-mus(264).
ext_dep(3,5,s0y2):-mus(265).
ext_dep(3,5,s0y4):-mus(266).
ext_dep(3,5,s0y6):-mus(267).
ext_dep(3,5,s0y7):-mus(268).
ext_dep(3,5,s1y2):-mus(269).
ext_dep(3,5,s1y4):-mus(270).
ext_dep(3,5,s1y7):-mus(271).
ext_dep(3,5,s2y4):-mus(272).
ext_dep(3,5,s2y6):-mus(273).
ext_dep(3,5,s2y7):-mus(274).
ext_dep(3,5,s4y7):-mus(275).
ext_dep(3,5,s0y1y2):-mus(276).
ext_dep(3,5,s0y1y4):-mus(277).
ext_dep(3,5,s0y1y7):-mus(278).
ext_dep(3,5,s0y2y4):-mus(279).
ext_dep(3,5,s0y2y6):-mus(280).
ext_dep(3,5,s0y2y7):-mus(281).
ext_dep(3,5,s0y4y7):-mus(282).
ext_dep(3,5,s1y2y4):-mus(283).
ext_dep(3,5,s1y2y7):-mus(284).
ext_dep(3,5,s1y4y7):-mus(285).
ext_dep(3,5,s2y4y7):-mus(286).
ext_dep(3,5,s0y1y2y4):-mus(287).
ext_dep(3,5,s0y1y2y7):-mus(288).
ext_dep(3,5,s0y1y4y7):-mus(289).
ext_dep(3,5,s0y2y4y7):-mus(290).
ext_dep(3,5,s1y2y4y7):-mus(291).
ext_dep(3,5,s0y1y2y4y7):-mus(292).
ext_dep(3,6,s0y1):-mus(293).
ext_dep(3,6,s0y2):-mus(294).
ext_indep(3,6,s0y4):-mus(295).
ext_indep(3,6,s0y5):-mus(296).
ext_indep(3,6,s0y7):-mus(297).
ext_dep(3,6,s1y2):-mus(298).
ext_dep(3,6,s1y4):-mus(299).
ext_dep(3,6,s1y5):-mus(300).
ext_dep(3,6,s1y7):-mus(301).
ext_dep(3,6,s2y4):-mus(302).
ext_dep(3,6,s2y5):-mus(303).
ext_dep(3,6,s2y7):-mus(304).
ext_dep(3,6,s4y5):-mus(305).
ext_dep(3,6,s4y7):-mus(306).
ext_dep(3,6,s5y7):-mus(307).
ext_dep(3,6,s0y1y2):-mus(308).
ext_dep(3,6,s0y1y4):-mus(309).
ext_indep(3,6,s0y1y5):-mus(310).
ext_dep(3,6,s0y1y7):-mus(311).
ext_dep(3,6,s0y2y4):-mus(312).
ext_dep(3,6,s0y2y5):-mus(313).
ext_indep(3,6,s0y2y7):-mus(314).
ext_indep(3,6,s0y4y5):-mus(315).
ext_indep(3,6,s0y4y7):-mus(316).
ext_indep(3,6,s0y5y7):-mus(317).
ext_indep(3,6,s1y2y4):-mus(318).
ext_indep(3,6,s1y2y5):-mus(319).
ext_dep(3,6,s1y2y7):-mus(320).
ext_indep(3,6,s1y4y5):-mus(321).
ext_dep(3,6,s1y4y7):-mus(322).
ext_indep(3,6,s1y5y7):-mus(323).
ext_dep(3,6,s2y4y5):-mus(324).
ext_dep(3,6,s2y4y7):-mus(325).
ext_dep(3,6,s2y5y7):-mus(326).
ext_dep(3,6,s4y5y7):-mus(327).
ext_dep(3,6,s0y1y2y4):-mus(328).
ext_dep(3,6,s0y1y2y5):-mus(329).
ext_dep(3,6,s0y1y2y7):-mus(330).
ext_dep(3,6,s0y1y4y5):-mus(331).
ext_dep(3,6,s0y1y4y7):-mus(332).
ext_dep(3,6,s0y1y5y7):-mus(333).
ext_dep(3,6,s0y2y4y5):-mus(334).
ext_dep(3,6,s0y2y4y7):-mus(335).
ext_indep(3,6,s0y2y5y7):-mus(336).
ext_indep(3,6,s0y4y5y7):-mus(337).
ext_indep(3,6,s1y2y4y5):-mus(338).
ext_dep(3,6,s1y2y4y7):-mus(339).
ext_indep(3,6,s1y2y5y7):-mus(340).
ext_indep(3,6,s1y4y5y7):-mus(341).
ext_dep(3,6,s2y4y5y7):-mus(342).
ext_dep(3,6,s0y1y2y4y5):-mus(343).
ext_dep(3,6,s0y1y2y4y7):-mus(344).
ext_dep(3,6,s0y1y2y5y7):-mus(345).
ext_dep(3,6,s0y1y4y5y7):-mus(346).
ext_dep(3,6,s0y2y4y5y7):-mus(347).
ext_dep(3,6,s1y2y4y5y7):-mus(348).
ext_dep(3,6,s0y1y2y4y5y7):-mus(349).
ext_dep(3,7,empty):-mus(350).
ext_dep(3,7,s0):-mus(351).
ext_dep(3,7,s1):-mus(352).
ext_dep(3,7,s2):-mus(353).
ext_dep(3,7,s4):-mus(354).
ext_dep(3,7,s5):-mus(355).
ext_dep(3,7,s6):-mus(356).
ext_dep(3,7,s0y1):-mus(357).
ext_dep(3,7,s0y2):-mus(358).
ext_dep(3,7,s0y4):-mus(359).
ext_dep(3,7,s0y5):-mus(360).
ext_dep(3,7,s1y2):-mus(361).
ext_dep(3,7,s1y4):-mus(362).
ext_dep(3,7,s1y5):-mus(363).
ext_dep(3,7,s2y4):-mus(364).
ext_dep(3,7,s2y5):-mus(365).
ext_dep(3,7,s4y5):-mus(366).
ext_dep(3,7,s0y1y2):-mus(367).
ext_dep(3,7,s0y1y4):-mus(368).
ext_dep(3,7,s0y1y5):-mus(369).
ext_dep(3,7,s0y2y4):-mus(370).
ext_dep(3,7,s0y2y5):-mus(371).
ext_dep(3,7,s0y4y5):-mus(372).
ext_dep(3,7,s1y2y4):-mus(373).
ext_dep(3,7,s1y2y5):-mus(374).
ext_dep(3,7,s1y4y5):-mus(375).
ext_dep(3,7,s2y4y5):-mus(376).
ext_dep(3,7,s0y1y2y4):-mus(377).
ext_dep(3,7,s0y1y2y5):-mus(378).
ext_dep(3,7,s0y1y4y5):-mus(379).
ext_dep(3,7,s0y2y4y5):-mus(380).
ext_dep(3,7,s1y2y4y5):-mus(381).
ext_dep(3,7,s0y1y2y4y5):-mus(382).
ext_indep(4,6,empty):-mus(383).
ext_dep(4,7,empty):-mus(384).
ext_dep(4,7,s0):-mus(385).
ext_dep(4,7,s1):-mus(386).
ext_dep(4,7,s2):-mus(387).
ext_dep(4,7,s3):-mus(388).
ext_dep(4,7,s5):-mus(389).
ext_dep(4,7,s0y3):-mus(390).
ext_dep(4,7,s1y2):-mus(391).
ext_dep(4,7,s1y3):-mus(392).
ext_dep(4,7,s2y3):-mus(393).
ext_dep(4,7,s1y2y3):-mus(394).
ext_dep(5,6,empty):-mus(395).
ext_dep(5,6,s0):-mus(396).
ext_dep(5,6,s1):-mus(397).
ext_dep(5,6,s2):-mus(398).
ext_dep(5,6,s3):-mus(399).
ext_dep(5,6,s4):-mus(400).
ext_dep(5,6,s0y2):-mus(401).
ext_dep(5,6,s0y3):-mus(402).
ext_dep(5,6,s2y3):-mus(403).
ext_dep(5,6,s0y2y3):-mus(404).
ext_indep(6,7,empty):-mus(405).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(5229768, mus(1)).
__optimum_mcs_objective_literal__(9999999, mus(2)).
__optimum_mcs_objective_literal__(9198941, mus(3)).
__optimum_mcs_objective_literal__(5332590, mus(4)).
__optimum_mcs_objective_literal__(8328870, mus(5)).
__optimum_mcs_objective_literal__(9999999, mus(6)).
__optimum_mcs_objective_literal__(6160012, mus(7)).
__optimum_mcs_objective_literal__(9999999, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9999999, mus(10)).
__optimum_mcs_objective_literal__(8228563, mus(11)).
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
__optimum_mcs_objective_literal__(9999999, mus(24)).
__optimum_mcs_objective_literal__(9999999, mus(25)).
__optimum_mcs_objective_literal__(9999999, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(9999999, mus(28)).
__optimum_mcs_objective_literal__(9999999, mus(29)).
__optimum_mcs_objective_literal__(9999999, mus(30)).
__optimum_mcs_objective_literal__(9999999, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(9999999, mus(33)).
__optimum_mcs_objective_literal__(9999999, mus(34)).
__optimum_mcs_objective_literal__(9999999, mus(35)).
__optimum_mcs_objective_literal__(9999999, mus(36)).
__optimum_mcs_objective_literal__(9999999, mus(37)).
__optimum_mcs_objective_literal__(9999999, mus(38)).
__optimum_mcs_objective_literal__(9999999, mus(39)).
__optimum_mcs_objective_literal__(9999999, mus(40)).
__optimum_mcs_objective_literal__(9999999, mus(41)).
__optimum_mcs_objective_literal__(9999999, mus(42)).
__optimum_mcs_objective_literal__(9999999, mus(43)).
__optimum_mcs_objective_literal__(9999999, mus(44)).
__optimum_mcs_objective_literal__(9999999, mus(45)).
__optimum_mcs_objective_literal__(9999999, mus(46)).
__optimum_mcs_objective_literal__(9999999, mus(47)).
__optimum_mcs_objective_literal__(9999999, mus(48)).
__optimum_mcs_objective_literal__(9999999, mus(49)).
__optimum_mcs_objective_literal__(9999999, mus(50)).
__optimum_mcs_objective_literal__(9999999, mus(51)).
__optimum_mcs_objective_literal__(9999999, mus(52)).
__optimum_mcs_objective_literal__(9999999, mus(53)).
__optimum_mcs_objective_literal__(9999999, mus(54)).
__optimum_mcs_objective_literal__(9999999, mus(55)).
__optimum_mcs_objective_literal__(9999999, mus(56)).
__optimum_mcs_objective_literal__(9999999, mus(57)).
__optimum_mcs_objective_literal__(9999999, mus(58)).
__optimum_mcs_objective_literal__(9999999, mus(59)).
__optimum_mcs_objective_literal__(9989814, mus(60)).
__optimum_mcs_objective_literal__(9999999, mus(61)).
__optimum_mcs_objective_literal__(9999999, mus(62)).
__optimum_mcs_objective_literal__(5017139, mus(63)).
__optimum_mcs_objective_literal__(9999999, mus(64)).
__optimum_mcs_objective_literal__(9999999, mus(65)).
__optimum_mcs_objective_literal__(9797591, mus(66)).
__optimum_mcs_objective_literal__(9925782, mus(67)).
__optimum_mcs_objective_literal__(9999999, mus(68)).
__optimum_mcs_objective_literal__(9999998, mus(69)).
__optimum_mcs_objective_literal__(9961127, mus(70)).
__optimum_mcs_objective_literal__(9999999, mus(71)).
__optimum_mcs_objective_literal__(9831063, mus(72)).
__optimum_mcs_objective_literal__(9999999, mus(73)).
__optimum_mcs_objective_literal__(9999999, mus(74)).
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
__optimum_mcs_objective_literal__(9999999, mus(129)).
__optimum_mcs_objective_literal__(9999999, mus(130)).
__optimum_mcs_objective_literal__(9999999, mus(131)).
__optimum_mcs_objective_literal__(9999999, mus(132)).
__optimum_mcs_objective_literal__(9999999, mus(133)).
__optimum_mcs_objective_literal__(9999999, mus(134)).
__optimum_mcs_objective_literal__(9999999, mus(135)).
__optimum_mcs_objective_literal__(9984659, mus(136)).
__optimum_mcs_objective_literal__(9959363, mus(137)).
__optimum_mcs_objective_literal__(9929681, mus(138)).
__optimum_mcs_objective_literal__(9804395, mus(139)).
__optimum_mcs_objective_literal__(9621535, mus(140)).
__optimum_mcs_objective_literal__(9333305, mus(141)).
__optimum_mcs_objective_literal__(9881817, mus(142)).
__optimum_mcs_objective_literal__(5001916, mus(143)).
__optimum_mcs_objective_literal__(5618647, mus(144)).
__optimum_mcs_objective_literal__(7573386, mus(145)).
__optimum_mcs_objective_literal__(9999999, mus(146)).
__optimum_mcs_objective_literal__(9999999, mus(147)).
__optimum_mcs_objective_literal__(9989567, mus(148)).
__optimum_mcs_objective_literal__(9999999, mus(149)).
__optimum_mcs_objective_literal__(9999999, mus(150)).
__optimum_mcs_objective_literal__(9999448, mus(151)).
__optimum_mcs_objective_literal__(9979938, mus(152)).
__optimum_mcs_objective_literal__(9994542, mus(153)).
__optimum_mcs_objective_literal__(9999998, mus(154)).
__optimum_mcs_objective_literal__(9999999, mus(155)).
__optimum_mcs_objective_literal__(9999999, mus(156)).
__optimum_mcs_objective_literal__(9999999, mus(157)).
__optimum_mcs_objective_literal__(9999999, mus(158)).
__optimum_mcs_objective_literal__(9999999, mus(159)).
__optimum_mcs_objective_literal__(9999999, mus(160)).
__optimum_mcs_objective_literal__(9999999, mus(161)).
__optimum_mcs_objective_literal__(9966351, mus(162)).
__optimum_mcs_objective_literal__(9999991, mus(163)).
__optimum_mcs_objective_literal__(9999937, mus(164)).
__optimum_mcs_objective_literal__(9999999, mus(165)).
__optimum_mcs_objective_literal__(9999999, mus(166)).
__optimum_mcs_objective_literal__(9999999, mus(167)).
__optimum_mcs_objective_literal__(9999999, mus(168)).
__optimum_mcs_objective_literal__(9999999, mus(169)).
__optimum_mcs_objective_literal__(9999999, mus(170)).
__optimum_mcs_objective_literal__(9999999, mus(171)).
__optimum_mcs_objective_literal__(9999999, mus(172)).
__optimum_mcs_objective_literal__(9999999, mus(173)).
__optimum_mcs_objective_literal__(9999998, mus(174)).
__optimum_mcs_objective_literal__(9999999, mus(175)).
__optimum_mcs_objective_literal__(9999999, mus(176)).
__optimum_mcs_objective_literal__(9999999, mus(177)).
__optimum_mcs_objective_literal__(9999999, mus(178)).
__optimum_mcs_objective_literal__(9999999, mus(179)).
__optimum_mcs_objective_literal__(9999999, mus(180)).
__optimum_mcs_objective_literal__(9999999, mus(181)).
__optimum_mcs_objective_literal__(9999999, mus(182)).
__optimum_mcs_objective_literal__(9999999, mus(183)).
__optimum_mcs_objective_literal__(9999999, mus(184)).
__optimum_mcs_objective_literal__(9999999, mus(185)).
__optimum_mcs_objective_literal__(9999999, mus(186)).
__optimum_mcs_objective_literal__(9999999, mus(187)).
__optimum_mcs_objective_literal__(9999999, mus(188)).
__optimum_mcs_objective_literal__(9999999, mus(189)).
__optimum_mcs_objective_literal__(9999999, mus(190)).
__optimum_mcs_objective_literal__(9999998, mus(191)).
__optimum_mcs_objective_literal__(9999999, mus(192)).
__optimum_mcs_objective_literal__(9999999, mus(193)).
__optimum_mcs_objective_literal__(9999999, mus(194)).
__optimum_mcs_objective_literal__(9999999, mus(195)).
__optimum_mcs_objective_literal__(9999999, mus(196)).
__optimum_mcs_objective_literal__(9999999, mus(197)).
__optimum_mcs_objective_literal__(9999999, mus(198)).
__optimum_mcs_objective_literal__(9999999, mus(199)).
__optimum_mcs_objective_literal__(9999999, mus(200)).
__optimum_mcs_objective_literal__(9999999, mus(201)).
__optimum_mcs_objective_literal__(9999999, mus(202)).
__optimum_mcs_objective_literal__(9999999, mus(203)).
__optimum_mcs_objective_literal__(9999999, mus(204)).
__optimum_mcs_objective_literal__(9999999, mus(205)).
__optimum_mcs_objective_literal__(9999999, mus(206)).
__optimum_mcs_objective_literal__(9999999, mus(207)).
__optimum_mcs_objective_literal__(9999999, mus(208)).
__optimum_mcs_objective_literal__(6391995, mus(209)).
__optimum_mcs_objective_literal__(5096269, mus(210)).
__optimum_mcs_objective_literal__(5136695, mus(211)).
__optimum_mcs_objective_literal__(9999998, mus(212)).
__optimum_mcs_objective_literal__(5893129, mus(213)).
__optimum_mcs_objective_literal__(6034150, mus(214)).
__optimum_mcs_objective_literal__(9999999, mus(215)).
__optimum_mcs_objective_literal__(5266465, mus(216)).
__optimum_mcs_objective_literal__(9998781, mus(217)).
__optimum_mcs_objective_literal__(9999991, mus(218)).
__optimum_mcs_objective_literal__(7559974, mus(219)).
__optimum_mcs_objective_literal__(9984543, mus(220)).
__optimum_mcs_objective_literal__(9999992, mus(221)).
__optimum_mcs_objective_literal__(9944246, mus(222)).
__optimum_mcs_objective_literal__(9932703, mus(223)).
__optimum_mcs_objective_literal__(9999232, mus(224)).
__optimum_mcs_objective_literal__(9999999, mus(225)).
__optimum_mcs_objective_literal__(9999947, mus(226)).
__optimum_mcs_objective_literal__(9999948, mus(227)).
__optimum_mcs_objective_literal__(9999999, mus(228)).
__optimum_mcs_objective_literal__(9999527, mus(229)).
__optimum_mcs_objective_literal__(9999999, mus(230)).
__optimum_mcs_objective_literal__(9999999, mus(231)).
__optimum_mcs_objective_literal__(9999999, mus(232)).
__optimum_mcs_objective_literal__(9999999, mus(233)).
__optimum_mcs_objective_literal__(9999999, mus(234)).
__optimum_mcs_objective_literal__(9999999, mus(235)).
__optimum_mcs_objective_literal__(9999999, mus(236)).
__optimum_mcs_objective_literal__(9999999, mus(237)).
__optimum_mcs_objective_literal__(9999999, mus(238)).
__optimum_mcs_objective_literal__(9999998, mus(239)).
__optimum_mcs_objective_literal__(9999999, mus(240)).
__optimum_mcs_objective_literal__(9999999, mus(241)).
__optimum_mcs_objective_literal__(9999999, mus(242)).
__optimum_mcs_objective_literal__(9999999, mus(243)).
__optimum_mcs_objective_literal__(9999999, mus(244)).
__optimum_mcs_objective_literal__(9999999, mus(245)).
__optimum_mcs_objective_literal__(9999999, mus(246)).
__optimum_mcs_objective_literal__(9999999, mus(247)).
__optimum_mcs_objective_literal__(9999999, mus(248)).
__optimum_mcs_objective_literal__(9999999, mus(249)).
__optimum_mcs_objective_literal__(9999999, mus(250)).
__optimum_mcs_objective_literal__(9999999, mus(251)).
__optimum_mcs_objective_literal__(9999999, mus(252)).
__optimum_mcs_objective_literal__(9999999, mus(253)).
__optimum_mcs_objective_literal__(9999999, mus(254)).
__optimum_mcs_objective_literal__(9999999, mus(255)).
__optimum_mcs_objective_literal__(9999999, mus(256)).
__optimum_mcs_objective_literal__(9999999, mus(257)).
__optimum_mcs_objective_literal__(9999999, mus(258)).
__optimum_mcs_objective_literal__(9999999, mus(259)).
__optimum_mcs_objective_literal__(9999999, mus(260)).
__optimum_mcs_objective_literal__(9999999, mus(261)).
__optimum_mcs_objective_literal__(9999999, mus(262)).
__optimum_mcs_objective_literal__(9999999, mus(263)).
__optimum_mcs_objective_literal__(9999999, mus(264)).
__optimum_mcs_objective_literal__(9999999, mus(265)).
__optimum_mcs_objective_literal__(9999999, mus(266)).
__optimum_mcs_objective_literal__(9999999, mus(267)).
__optimum_mcs_objective_literal__(9999999, mus(268)).
__optimum_mcs_objective_literal__(9999999, mus(269)).
__optimum_mcs_objective_literal__(9999999, mus(270)).
__optimum_mcs_objective_literal__(9999999, mus(271)).
__optimum_mcs_objective_literal__(9999999, mus(272)).
__optimum_mcs_objective_literal__(9999999, mus(273)).
__optimum_mcs_objective_literal__(9999999, mus(274)).
__optimum_mcs_objective_literal__(9999999, mus(275)).
__optimum_mcs_objective_literal__(9999999, mus(276)).
__optimum_mcs_objective_literal__(9999999, mus(277)).
__optimum_mcs_objective_literal__(9999999, mus(278)).
__optimum_mcs_objective_literal__(9999999, mus(279)).
__optimum_mcs_objective_literal__(9999999, mus(280)).
__optimum_mcs_objective_literal__(9999999, mus(281)).
__optimum_mcs_objective_literal__(9999999, mus(282)).
__optimum_mcs_objective_literal__(9999999, mus(283)).
__optimum_mcs_objective_literal__(9999999, mus(284)).
__optimum_mcs_objective_literal__(9999999, mus(285)).
__optimum_mcs_objective_literal__(9999999, mus(286)).
__optimum_mcs_objective_literal__(9999999, mus(287)).
__optimum_mcs_objective_literal__(9999999, mus(288)).
__optimum_mcs_objective_literal__(9999999, mus(289)).
__optimum_mcs_objective_literal__(9999999, mus(290)).
__optimum_mcs_objective_literal__(9999999, mus(291)).
__optimum_mcs_objective_literal__(9999999, mus(292)).
__optimum_mcs_objective_literal__(8621655, mus(293)).
__optimum_mcs_objective_literal__(7623774, mus(294)).
__optimum_mcs_objective_literal__(5002030, mus(295)).
__optimum_mcs_objective_literal__(5293518, mus(296)).
__optimum_mcs_objective_literal__(5711431, mus(297)).
__optimum_mcs_objective_literal__(6417948, mus(298)).
__optimum_mcs_objective_literal__(9261679, mus(299)).
__optimum_mcs_objective_literal__(7734867, mus(300)).
__optimum_mcs_objective_literal__(5489822, mus(301)).
__optimum_mcs_objective_literal__(8815973, mus(302)).
__optimum_mcs_objective_literal__(8115819, mus(303)).
__optimum_mcs_objective_literal__(8952846, mus(304)).
__optimum_mcs_objective_literal__(9107757, mus(305)).
__optimum_mcs_objective_literal__(6770753, mus(306)).
__optimum_mcs_objective_literal__(9270219, mus(307)).
__optimum_mcs_objective_literal__(7641496, mus(308)).
__optimum_mcs_objective_literal__(9913667, mus(309)).
__optimum_mcs_objective_literal__(5189882, mus(310)).
__optimum_mcs_objective_literal__(9200202, mus(311)).
__optimum_mcs_objective_literal__(9626109, mus(312)).
__optimum_mcs_objective_literal__(6745891, mus(313)).
__optimum_mcs_objective_literal__(5019014, mus(314)).
__optimum_mcs_objective_literal__(5405197, mus(315)).
__optimum_mcs_objective_literal__(5045567, mus(316)).
__optimum_mcs_objective_literal__(6054102, mus(317)).
__optimum_mcs_objective_literal__(5036332, mus(318)).
__optimum_mcs_objective_literal__(5074637, mus(319)).
__optimum_mcs_objective_literal__(5421773, mus(320)).
__optimum_mcs_objective_literal__(5223169, mus(321)).
__optimum_mcs_objective_literal__(5755157, mus(322)).
__optimum_mcs_objective_literal__(5001931, mus(323)).
__optimum_mcs_objective_literal__(6749128, mus(324)).
__optimum_mcs_objective_literal__(9785230, mus(325)).
__optimum_mcs_objective_literal__(9782927, mus(326)).
__optimum_mcs_objective_literal__(8008902, mus(327)).
__optimum_mcs_objective_literal__(9816867, mus(328)).
__optimum_mcs_objective_literal__(5804875, mus(329)).
__optimum_mcs_objective_literal__(9397975, mus(330)).
__optimum_mcs_objective_literal__(5710810, mus(331)).
__optimum_mcs_objective_literal__(9951312, mus(332)).
__optimum_mcs_objective_literal__(5316799, mus(333)).
__optimum_mcs_objective_literal__(8440903, mus(334)).
__optimum_mcs_objective_literal__(9500390, mus(335)).
__optimum_mcs_objective_literal__(5040801, mus(336)).
__optimum_mcs_objective_literal__(5612963, mus(337)).
__optimum_mcs_objective_literal__(5177125, mus(338)).
__optimum_mcs_objective_literal__(8828258, mus(339)).
__optimum_mcs_objective_literal__(5042650, mus(340)).
__optimum_mcs_objective_literal__(5165103, mus(341)).
__optimum_mcs_objective_literal__(9794532, mus(342)).
__optimum_mcs_objective_literal__(9767019, mus(343)).
__optimum_mcs_objective_literal__(9997265, mus(344)).
__optimum_mcs_objective_literal__(7830302, mus(345)).
__optimum_mcs_objective_literal__(9583466, mus(346)).
__optimum_mcs_objective_literal__(9742065, mus(347)).
__optimum_mcs_objective_literal__(6078367, mus(348)).
__optimum_mcs_objective_literal__(9991735, mus(349)).
__optimum_mcs_objective_literal__(9847323, mus(350)).
__optimum_mcs_objective_literal__(9999999, mus(351)).
__optimum_mcs_objective_literal__(9998981, mus(352)).
__optimum_mcs_objective_literal__(9999989, mus(353)).
__optimum_mcs_objective_literal__(9999989, mus(354)).
__optimum_mcs_objective_literal__(9999999, mus(355)).
__optimum_mcs_objective_literal__(9040561, mus(356)).
__optimum_mcs_objective_literal__(9999999, mus(357)).
__optimum_mcs_objective_literal__(9999999, mus(358)).
__optimum_mcs_objective_literal__(9999999, mus(359)).
__optimum_mcs_objective_literal__(9999999, mus(360)).
__optimum_mcs_objective_literal__(9999999, mus(361)).
__optimum_mcs_objective_literal__(9999993, mus(362)).
__optimum_mcs_objective_literal__(9999999, mus(363)).
__optimum_mcs_objective_literal__(9999999, mus(364)).
__optimum_mcs_objective_literal__(9999999, mus(365)).
__optimum_mcs_objective_literal__(9999999, mus(366)).
__optimum_mcs_objective_literal__(9999999, mus(367)).
__optimum_mcs_objective_literal__(9999999, mus(368)).
__optimum_mcs_objective_literal__(9999999, mus(369)).
__optimum_mcs_objective_literal__(9999999, mus(370)).
__optimum_mcs_objective_literal__(9999999, mus(371)).
__optimum_mcs_objective_literal__(9999999, mus(372)).
__optimum_mcs_objective_literal__(9999999, mus(373)).
__optimum_mcs_objective_literal__(9999999, mus(374)).
__optimum_mcs_objective_literal__(9999999, mus(375)).
__optimum_mcs_objective_literal__(9999999, mus(376)).
__optimum_mcs_objective_literal__(9999999, mus(377)).
__optimum_mcs_objective_literal__(9999999, mus(378)).
__optimum_mcs_objective_literal__(9999999, mus(379)).
__optimum_mcs_objective_literal__(9999999, mus(380)).
__optimum_mcs_objective_literal__(9999999, mus(381)).
__optimum_mcs_objective_literal__(9999999, mus(382)).
__optimum_mcs_objective_literal__(5118166, mus(383)).
__optimum_mcs_objective_literal__(9999999, mus(384)).
__optimum_mcs_objective_literal__(9999999, mus(385)).
__optimum_mcs_objective_literal__(9999999, mus(386)).
__optimum_mcs_objective_literal__(9999999, mus(387)).
__optimum_mcs_objective_literal__(9999999, mus(388)).
__optimum_mcs_objective_literal__(9999999, mus(389)).
__optimum_mcs_objective_literal__(9999999, mus(390)).
__optimum_mcs_objective_literal__(9999999, mus(391)).
__optimum_mcs_objective_literal__(9999999, mus(392)).
__optimum_mcs_objective_literal__(9999999, mus(393)).
__optimum_mcs_objective_literal__(9999999, mus(394)).
__optimum_mcs_objective_literal__(9999999, mus(395)).
__optimum_mcs_objective_literal__(9999999, mus(396)).
__optimum_mcs_objective_literal__(9999999, mus(397)).
__optimum_mcs_objective_literal__(9999999, mus(398)).
__optimum_mcs_objective_literal__(9999999, mus(399)).
__optimum_mcs_objective_literal__(9999999, mus(400)).
__optimum_mcs_objective_literal__(9999999, mus(401)).
__optimum_mcs_objective_literal__(9999999, mus(402)).
__optimum_mcs_objective_literal__(9999999, mus(403)).
__optimum_mcs_objective_literal__(9999999, mus(404)).
__optimum_mcs_objective_literal__(5290201, mus(405)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
