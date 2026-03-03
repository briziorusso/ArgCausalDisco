% source dump (pre-ground) from causalaba_increm.CausalABA
% note: this is NOT a grounded observer dump
% note: facts are intentionally omitted (base-only dump).
%       Load facts separately (e.g. add your facts .lp on the clingo command line).
%%% sets and elements
var(0..9).
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
in(8,s8).
in(9,s9).
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
in(0,s0y8).
in(8,s0y8).
in(0,s0y9).
in(9,s0y9).
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
in(1,s1y8).
in(8,s1y8).
in(1,s1y9).
in(9,s1y9).
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
in(2,s2y8).
in(8,s2y8).
in(2,s2y9).
in(9,s2y9).
in(3,s3y4).
in(4,s3y4).
in(3,s3y5).
in(5,s3y5).
in(3,s3y6).
in(6,s3y6).
in(3,s3y7).
in(7,s3y7).
in(3,s3y8).
in(8,s3y8).
in(3,s3y9).
in(9,s3y9).
in(4,s4y5).
in(5,s4y5).
in(4,s4y6).
in(6,s4y6).
in(4,s4y7).
in(7,s4y7).
in(4,s4y8).
in(8,s4y8).
in(4,s4y9).
in(9,s4y9).
in(5,s5y6).
in(6,s5y6).
in(5,s5y7).
in(7,s5y7).
in(5,s5y8).
in(8,s5y8).
in(5,s5y9).
in(9,s5y9).
in(6,s6y7).
in(7,s6y7).
in(6,s6y8).
in(8,s6y8).
in(6,s6y9).
in(9,s6y9).
in(7,s7y8).
in(8,s7y8).
in(7,s7y9).
in(9,s7y9).
in(8,s8y9).
in(9,s8y9).
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
in(0,s0y1y8).
in(1,s0y1y8).
in(8,s0y1y8).
in(0,s0y1y9).
in(1,s0y1y9).
in(9,s0y1y9).
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
in(0,s0y2y8).
in(2,s0y2y8).
in(8,s0y2y8).
in(0,s0y2y9).
in(2,s0y2y9).
in(9,s0y2y9).
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
in(0,s0y3y8).
in(3,s0y3y8).
in(8,s0y3y8).
in(0,s0y3y9).
in(3,s0y3y9).
in(9,s0y3y9).
in(0,s0y4y5).
in(4,s0y4y5).
in(5,s0y4y5).
in(0,s0y4y6).
in(4,s0y4y6).
in(6,s0y4y6).
in(0,s0y4y7).
in(4,s0y4y7).
in(7,s0y4y7).
in(0,s0y4y8).
in(4,s0y4y8).
in(8,s0y4y8).
in(0,s0y4y9).
in(4,s0y4y9).
in(9,s0y4y9).
in(0,s0y5y6).
in(5,s0y5y6).
in(6,s0y5y6).
in(0,s0y5y7).
in(5,s0y5y7).
in(7,s0y5y7).
in(0,s0y5y8).
in(5,s0y5y8).
in(8,s0y5y8).
in(0,s0y5y9).
in(5,s0y5y9).
in(9,s0y5y9).
in(0,s0y6y7).
in(6,s0y6y7).
in(7,s0y6y7).
in(0,s0y6y8).
in(6,s0y6y8).
in(8,s0y6y8).
in(0,s0y6y9).
in(6,s0y6y9).
in(9,s0y6y9).
in(0,s0y7y8).
in(7,s0y7y8).
in(8,s0y7y8).
in(0,s0y7y9).
in(7,s0y7y9).
in(9,s0y7y9).
in(0,s0y8y9).
in(8,s0y8y9).
in(9,s0y8y9).
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
in(1,s1y2y8).
in(2,s1y2y8).
in(8,s1y2y8).
in(1,s1y2y9).
in(2,s1y2y9).
in(9,s1y2y9).
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
in(1,s1y3y8).
in(3,s1y3y8).
in(8,s1y3y8).
in(1,s1y3y9).
in(3,s1y3y9).
in(9,s1y3y9).
in(1,s1y4y5).
in(4,s1y4y5).
in(5,s1y4y5).
in(1,s1y4y6).
in(4,s1y4y6).
in(6,s1y4y6).
in(1,s1y4y7).
in(4,s1y4y7).
in(7,s1y4y7).
in(1,s1y4y8).
in(4,s1y4y8).
in(8,s1y4y8).
in(1,s1y4y9).
in(4,s1y4y9).
in(9,s1y4y9).
in(1,s1y5y6).
in(5,s1y5y6).
in(6,s1y5y6).
in(1,s1y5y7).
in(5,s1y5y7).
in(7,s1y5y7).
in(1,s1y5y8).
in(5,s1y5y8).
in(8,s1y5y8).
in(1,s1y5y9).
in(5,s1y5y9).
in(9,s1y5y9).
in(1,s1y6y7).
in(6,s1y6y7).
in(7,s1y6y7).
in(1,s1y6y8).
in(6,s1y6y8).
in(8,s1y6y8).
in(1,s1y6y9).
in(6,s1y6y9).
in(9,s1y6y9).
in(1,s1y7y8).
in(7,s1y7y8).
in(8,s1y7y8).
in(1,s1y7y9).
in(7,s1y7y9).
in(9,s1y7y9).
in(1,s1y8y9).
in(8,s1y8y9).
in(9,s1y8y9).
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
in(2,s2y3y8).
in(3,s2y3y8).
in(8,s2y3y8).
in(2,s2y3y9).
in(3,s2y3y9).
in(9,s2y3y9).
in(2,s2y4y5).
in(4,s2y4y5).
in(5,s2y4y5).
in(2,s2y4y6).
in(4,s2y4y6).
in(6,s2y4y6).
in(2,s2y4y7).
in(4,s2y4y7).
in(7,s2y4y7).
in(2,s2y4y8).
in(4,s2y4y8).
in(8,s2y4y8).
in(2,s2y4y9).
in(4,s2y4y9).
in(9,s2y4y9).
in(2,s2y5y6).
in(5,s2y5y6).
in(6,s2y5y6).
in(2,s2y5y7).
in(5,s2y5y7).
in(7,s2y5y7).
in(2,s2y5y8).
in(5,s2y5y8).
in(8,s2y5y8).
in(2,s2y5y9).
in(5,s2y5y9).
in(9,s2y5y9).
in(2,s2y6y7).
in(6,s2y6y7).
in(7,s2y6y7).
in(2,s2y6y8).
in(6,s2y6y8).
in(8,s2y6y8).
in(2,s2y6y9).
in(6,s2y6y9).
in(9,s2y6y9).
in(2,s2y7y8).
in(7,s2y7y8).
in(8,s2y7y8).
in(2,s2y7y9).
in(7,s2y7y9).
in(9,s2y7y9).
in(2,s2y8y9).
in(8,s2y8y9).
in(9,s2y8y9).
in(3,s3y4y5).
in(4,s3y4y5).
in(5,s3y4y5).
in(3,s3y4y6).
in(4,s3y4y6).
in(6,s3y4y6).
in(3,s3y4y7).
in(4,s3y4y7).
in(7,s3y4y7).
in(3,s3y4y8).
in(4,s3y4y8).
in(8,s3y4y8).
in(3,s3y4y9).
in(4,s3y4y9).
in(9,s3y4y9).
in(3,s3y5y6).
in(5,s3y5y6).
in(6,s3y5y6).
in(3,s3y5y7).
in(5,s3y5y7).
in(7,s3y5y7).
in(3,s3y5y8).
in(5,s3y5y8).
in(8,s3y5y8).
in(3,s3y5y9).
in(5,s3y5y9).
in(9,s3y5y9).
in(3,s3y6y7).
in(6,s3y6y7).
in(7,s3y6y7).
in(3,s3y6y8).
in(6,s3y6y8).
in(8,s3y6y8).
in(3,s3y6y9).
in(6,s3y6y9).
in(9,s3y6y9).
in(3,s3y7y8).
in(7,s3y7y8).
in(8,s3y7y8).
in(3,s3y7y9).
in(7,s3y7y9).
in(9,s3y7y9).
in(3,s3y8y9).
in(8,s3y8y9).
in(9,s3y8y9).
in(4,s4y5y6).
in(5,s4y5y6).
in(6,s4y5y6).
in(4,s4y5y7).
in(5,s4y5y7).
in(7,s4y5y7).
in(4,s4y5y8).
in(5,s4y5y8).
in(8,s4y5y8).
in(4,s4y5y9).
in(5,s4y5y9).
in(9,s4y5y9).
in(4,s4y6y7).
in(6,s4y6y7).
in(7,s4y6y7).
in(4,s4y6y8).
in(6,s4y6y8).
in(8,s4y6y8).
in(4,s4y6y9).
in(6,s4y6y9).
in(9,s4y6y9).
in(4,s4y7y8).
in(7,s4y7y8).
in(8,s4y7y8).
in(4,s4y7y9).
in(7,s4y7y9).
in(9,s4y7y9).
in(4,s4y8y9).
in(8,s4y8y9).
in(9,s4y8y9).
in(5,s5y6y7).
in(6,s5y6y7).
in(7,s5y6y7).
in(5,s5y6y8).
in(6,s5y6y8).
in(8,s5y6y8).
in(5,s5y6y9).
in(6,s5y6y9).
in(9,s5y6y9).
in(5,s5y7y8).
in(7,s5y7y8).
in(8,s5y7y8).
in(5,s5y7y9).
in(7,s5y7y9).
in(9,s5y7y9).
in(5,s5y8y9).
in(8,s5y8y9).
in(9,s5y8y9).
in(6,s6y7y8).
in(7,s6y7y8).
in(8,s6y7y8).
in(6,s6y7y9).
in(7,s6y7y9).
in(9,s6y7y9).
in(6,s6y8y9).
in(8,s6y8y9).
in(9,s6y8y9).
in(7,s7y8y9).
in(8,s7y8y9).
in(9,s7y8y9).
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
in(0,s0y1y2y8).
in(1,s0y1y2y8).
in(2,s0y1y2y8).
in(8,s0y1y2y8).
in(0,s0y1y2y9).
in(1,s0y1y2y9).
in(2,s0y1y2y9).
in(9,s0y1y2y9).
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
in(0,s0y1y3y8).
in(1,s0y1y3y8).
in(3,s0y1y3y8).
in(8,s0y1y3y8).
in(0,s0y1y3y9).
in(1,s0y1y3y9).
in(3,s0y1y3y9).
in(9,s0y1y3y9).
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
in(0,s0y1y4y8).
in(1,s0y1y4y8).
in(4,s0y1y4y8).
in(8,s0y1y4y8).
in(0,s0y1y4y9).
in(1,s0y1y4y9).
in(4,s0y1y4y9).
in(9,s0y1y4y9).
in(0,s0y1y5y6).
in(1,s0y1y5y6).
in(5,s0y1y5y6).
in(6,s0y1y5y6).
in(0,s0y1y5y7).
in(1,s0y1y5y7).
in(5,s0y1y5y7).
in(7,s0y1y5y7).
in(0,s0y1y5y8).
in(1,s0y1y5y8).
in(5,s0y1y5y8).
in(8,s0y1y5y8).
in(0,s0y1y5y9).
in(1,s0y1y5y9).
in(5,s0y1y5y9).
in(9,s0y1y5y9).
in(0,s0y1y6y7).
in(1,s0y1y6y7).
in(6,s0y1y6y7).
in(7,s0y1y6y7).
in(0,s0y1y6y8).
in(1,s0y1y6y8).
in(6,s0y1y6y8).
in(8,s0y1y6y8).
in(0,s0y1y6y9).
in(1,s0y1y6y9).
in(6,s0y1y6y9).
in(9,s0y1y6y9).
in(0,s0y1y7y8).
in(1,s0y1y7y8).
in(7,s0y1y7y8).
in(8,s0y1y7y8).
in(0,s0y1y7y9).
in(1,s0y1y7y9).
in(7,s0y1y7y9).
in(9,s0y1y7y9).
in(0,s0y1y8y9).
in(1,s0y1y8y9).
in(8,s0y1y8y9).
in(9,s0y1y8y9).
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
in(0,s0y2y3y8).
in(2,s0y2y3y8).
in(3,s0y2y3y8).
in(8,s0y2y3y8).
in(0,s0y2y3y9).
in(2,s0y2y3y9).
in(3,s0y2y3y9).
in(9,s0y2y3y9).
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
in(0,s0y2y4y8).
in(2,s0y2y4y8).
in(4,s0y2y4y8).
in(8,s0y2y4y8).
in(0,s0y2y4y9).
in(2,s0y2y4y9).
in(4,s0y2y4y9).
in(9,s0y2y4y9).
in(0,s0y2y5y6).
in(2,s0y2y5y6).
in(5,s0y2y5y6).
in(6,s0y2y5y6).
in(0,s0y2y5y7).
in(2,s0y2y5y7).
in(5,s0y2y5y7).
in(7,s0y2y5y7).
in(0,s0y2y5y8).
in(2,s0y2y5y8).
in(5,s0y2y5y8).
in(8,s0y2y5y8).
in(0,s0y2y5y9).
in(2,s0y2y5y9).
in(5,s0y2y5y9).
in(9,s0y2y5y9).
in(0,s0y2y6y7).
in(2,s0y2y6y7).
in(6,s0y2y6y7).
in(7,s0y2y6y7).
in(0,s0y2y6y8).
in(2,s0y2y6y8).
in(6,s0y2y6y8).
in(8,s0y2y6y8).
in(0,s0y2y6y9).
in(2,s0y2y6y9).
in(6,s0y2y6y9).
in(9,s0y2y6y9).
in(0,s0y2y7y8).
in(2,s0y2y7y8).
in(7,s0y2y7y8).
in(8,s0y2y7y8).
in(0,s0y2y7y9).
in(2,s0y2y7y9).
in(7,s0y2y7y9).
in(9,s0y2y7y9).
in(0,s0y2y8y9).
in(2,s0y2y8y9).
in(8,s0y2y8y9).
in(9,s0y2y8y9).
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
in(0,s0y3y4y8).
in(3,s0y3y4y8).
in(4,s0y3y4y8).
in(8,s0y3y4y8).
in(0,s0y3y4y9).
in(3,s0y3y4y9).
in(4,s0y3y4y9).
in(9,s0y3y4y9).
in(0,s0y3y5y6).
in(3,s0y3y5y6).
in(5,s0y3y5y6).
in(6,s0y3y5y6).
in(0,s0y3y5y7).
in(3,s0y3y5y7).
in(5,s0y3y5y7).
in(7,s0y3y5y7).
in(0,s0y3y5y8).
in(3,s0y3y5y8).
in(5,s0y3y5y8).
in(8,s0y3y5y8).
in(0,s0y3y5y9).
in(3,s0y3y5y9).
in(5,s0y3y5y9).
in(9,s0y3y5y9).
in(0,s0y3y6y7).
in(3,s0y3y6y7).
in(6,s0y3y6y7).
in(7,s0y3y6y7).
in(0,s0y3y6y8).
in(3,s0y3y6y8).
in(6,s0y3y6y8).
in(8,s0y3y6y8).
in(0,s0y3y6y9).
in(3,s0y3y6y9).
in(6,s0y3y6y9).
in(9,s0y3y6y9).
in(0,s0y3y7y8).
in(3,s0y3y7y8).
in(7,s0y3y7y8).
in(8,s0y3y7y8).
in(0,s0y3y7y9).
in(3,s0y3y7y9).
in(7,s0y3y7y9).
in(9,s0y3y7y9).
in(0,s0y3y8y9).
in(3,s0y3y8y9).
in(8,s0y3y8y9).
in(9,s0y3y8y9).
in(0,s0y4y5y6).
in(4,s0y4y5y6).
in(5,s0y4y5y6).
in(6,s0y4y5y6).
in(0,s0y4y5y7).
in(4,s0y4y5y7).
in(5,s0y4y5y7).
in(7,s0y4y5y7).
in(0,s0y4y5y8).
in(4,s0y4y5y8).
in(5,s0y4y5y8).
in(8,s0y4y5y8).
in(0,s0y4y5y9).
in(4,s0y4y5y9).
in(5,s0y4y5y9).
in(9,s0y4y5y9).
in(0,s0y4y6y7).
in(4,s0y4y6y7).
in(6,s0y4y6y7).
in(7,s0y4y6y7).
in(0,s0y4y6y8).
in(4,s0y4y6y8).
in(6,s0y4y6y8).
in(8,s0y4y6y8).
in(0,s0y4y6y9).
in(4,s0y4y6y9).
in(6,s0y4y6y9).
in(9,s0y4y6y9).
in(0,s0y4y7y8).
in(4,s0y4y7y8).
in(7,s0y4y7y8).
in(8,s0y4y7y8).
in(0,s0y4y7y9).
in(4,s0y4y7y9).
in(7,s0y4y7y9).
in(9,s0y4y7y9).
in(0,s0y4y8y9).
in(4,s0y4y8y9).
in(8,s0y4y8y9).
in(9,s0y4y8y9).
in(0,s0y5y6y7).
in(5,s0y5y6y7).
in(6,s0y5y6y7).
in(7,s0y5y6y7).
in(0,s0y5y6y8).
in(5,s0y5y6y8).
in(6,s0y5y6y8).
in(8,s0y5y6y8).
in(0,s0y5y6y9).
in(5,s0y5y6y9).
in(6,s0y5y6y9).
in(9,s0y5y6y9).
in(0,s0y5y7y8).
in(5,s0y5y7y8).
in(7,s0y5y7y8).
in(8,s0y5y7y8).
in(0,s0y5y7y9).
in(5,s0y5y7y9).
in(7,s0y5y7y9).
in(9,s0y5y7y9).
in(0,s0y5y8y9).
in(5,s0y5y8y9).
in(8,s0y5y8y9).
in(9,s0y5y8y9).
in(0,s0y6y7y8).
in(6,s0y6y7y8).
in(7,s0y6y7y8).
in(8,s0y6y7y8).
in(0,s0y6y7y9).
in(6,s0y6y7y9).
in(7,s0y6y7y9).
in(9,s0y6y7y9).
in(0,s0y6y8y9).
in(6,s0y6y8y9).
in(8,s0y6y8y9).
in(9,s0y6y8y9).
in(0,s0y7y8y9).
in(7,s0y7y8y9).
in(8,s0y7y8y9).
in(9,s0y7y8y9).
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
in(1,s1y2y3y8).
in(2,s1y2y3y8).
in(3,s1y2y3y8).
in(8,s1y2y3y8).
in(1,s1y2y3y9).
in(2,s1y2y3y9).
in(3,s1y2y3y9).
in(9,s1y2y3y9).
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
in(1,s1y2y4y8).
in(2,s1y2y4y8).
in(4,s1y2y4y8).
in(8,s1y2y4y8).
in(1,s1y2y4y9).
in(2,s1y2y4y9).
in(4,s1y2y4y9).
in(9,s1y2y4y9).
in(1,s1y2y5y6).
in(2,s1y2y5y6).
in(5,s1y2y5y6).
in(6,s1y2y5y6).
in(1,s1y2y5y7).
in(2,s1y2y5y7).
in(5,s1y2y5y7).
in(7,s1y2y5y7).
in(1,s1y2y5y8).
in(2,s1y2y5y8).
in(5,s1y2y5y8).
in(8,s1y2y5y8).
in(1,s1y2y5y9).
in(2,s1y2y5y9).
in(5,s1y2y5y9).
in(9,s1y2y5y9).
in(1,s1y2y6y7).
in(2,s1y2y6y7).
in(6,s1y2y6y7).
in(7,s1y2y6y7).
in(1,s1y2y6y8).
in(2,s1y2y6y8).
in(6,s1y2y6y8).
in(8,s1y2y6y8).
in(1,s1y2y6y9).
in(2,s1y2y6y9).
in(6,s1y2y6y9).
in(9,s1y2y6y9).
in(1,s1y2y7y8).
in(2,s1y2y7y8).
in(7,s1y2y7y8).
in(8,s1y2y7y8).
in(1,s1y2y7y9).
in(2,s1y2y7y9).
in(7,s1y2y7y9).
in(9,s1y2y7y9).
in(1,s1y2y8y9).
in(2,s1y2y8y9).
in(8,s1y2y8y9).
in(9,s1y2y8y9).
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
in(1,s1y3y4y8).
in(3,s1y3y4y8).
in(4,s1y3y4y8).
in(8,s1y3y4y8).
in(1,s1y3y4y9).
in(3,s1y3y4y9).
in(4,s1y3y4y9).
in(9,s1y3y4y9).
in(1,s1y3y5y6).
in(3,s1y3y5y6).
in(5,s1y3y5y6).
in(6,s1y3y5y6).
in(1,s1y3y5y7).
in(3,s1y3y5y7).
in(5,s1y3y5y7).
in(7,s1y3y5y7).
in(1,s1y3y5y8).
in(3,s1y3y5y8).
in(5,s1y3y5y8).
in(8,s1y3y5y8).
in(1,s1y3y5y9).
in(3,s1y3y5y9).
in(5,s1y3y5y9).
in(9,s1y3y5y9).
in(1,s1y3y6y7).
in(3,s1y3y6y7).
in(6,s1y3y6y7).
in(7,s1y3y6y7).
in(1,s1y3y6y8).
in(3,s1y3y6y8).
in(6,s1y3y6y8).
in(8,s1y3y6y8).
in(1,s1y3y6y9).
in(3,s1y3y6y9).
in(6,s1y3y6y9).
in(9,s1y3y6y9).
in(1,s1y3y7y8).
in(3,s1y3y7y8).
in(7,s1y3y7y8).
in(8,s1y3y7y8).
in(1,s1y3y7y9).
in(3,s1y3y7y9).
in(7,s1y3y7y9).
in(9,s1y3y7y9).
in(1,s1y3y8y9).
in(3,s1y3y8y9).
in(8,s1y3y8y9).
in(9,s1y3y8y9).
in(1,s1y4y5y6).
in(4,s1y4y5y6).
in(5,s1y4y5y6).
in(6,s1y4y5y6).
in(1,s1y4y5y7).
in(4,s1y4y5y7).
in(5,s1y4y5y7).
in(7,s1y4y5y7).
in(1,s1y4y5y8).
in(4,s1y4y5y8).
in(5,s1y4y5y8).
in(8,s1y4y5y8).
in(1,s1y4y5y9).
in(4,s1y4y5y9).
in(5,s1y4y5y9).
in(9,s1y4y5y9).
in(1,s1y4y6y7).
in(4,s1y4y6y7).
in(6,s1y4y6y7).
in(7,s1y4y6y7).
in(1,s1y4y6y8).
in(4,s1y4y6y8).
in(6,s1y4y6y8).
in(8,s1y4y6y8).
in(1,s1y4y6y9).
in(4,s1y4y6y9).
in(6,s1y4y6y9).
in(9,s1y4y6y9).
in(1,s1y4y7y8).
in(4,s1y4y7y8).
in(7,s1y4y7y8).
in(8,s1y4y7y8).
in(1,s1y4y7y9).
in(4,s1y4y7y9).
in(7,s1y4y7y9).
in(9,s1y4y7y9).
in(1,s1y4y8y9).
in(4,s1y4y8y9).
in(8,s1y4y8y9).
in(9,s1y4y8y9).
in(1,s1y5y6y7).
in(5,s1y5y6y7).
in(6,s1y5y6y7).
in(7,s1y5y6y7).
in(1,s1y5y6y8).
in(5,s1y5y6y8).
in(6,s1y5y6y8).
in(8,s1y5y6y8).
in(1,s1y5y6y9).
in(5,s1y5y6y9).
in(6,s1y5y6y9).
in(9,s1y5y6y9).
in(1,s1y5y7y8).
in(5,s1y5y7y8).
in(7,s1y5y7y8).
in(8,s1y5y7y8).
in(1,s1y5y7y9).
in(5,s1y5y7y9).
in(7,s1y5y7y9).
in(9,s1y5y7y9).
in(1,s1y5y8y9).
in(5,s1y5y8y9).
in(8,s1y5y8y9).
in(9,s1y5y8y9).
in(1,s1y6y7y8).
in(6,s1y6y7y8).
in(7,s1y6y7y8).
in(8,s1y6y7y8).
in(1,s1y6y7y9).
in(6,s1y6y7y9).
in(7,s1y6y7y9).
in(9,s1y6y7y9).
in(1,s1y6y8y9).
in(6,s1y6y8y9).
in(8,s1y6y8y9).
in(9,s1y6y8y9).
in(1,s1y7y8y9).
in(7,s1y7y8y9).
in(8,s1y7y8y9).
in(9,s1y7y8y9).
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
in(2,s2y3y4y8).
in(3,s2y3y4y8).
in(4,s2y3y4y8).
in(8,s2y3y4y8).
in(2,s2y3y4y9).
in(3,s2y3y4y9).
in(4,s2y3y4y9).
in(9,s2y3y4y9).
in(2,s2y3y5y6).
in(3,s2y3y5y6).
in(5,s2y3y5y6).
in(6,s2y3y5y6).
in(2,s2y3y5y7).
in(3,s2y3y5y7).
in(5,s2y3y5y7).
in(7,s2y3y5y7).
in(2,s2y3y5y8).
in(3,s2y3y5y8).
in(5,s2y3y5y8).
in(8,s2y3y5y8).
in(2,s2y3y5y9).
in(3,s2y3y5y9).
in(5,s2y3y5y9).
in(9,s2y3y5y9).
in(2,s2y3y6y7).
in(3,s2y3y6y7).
in(6,s2y3y6y7).
in(7,s2y3y6y7).
in(2,s2y3y6y8).
in(3,s2y3y6y8).
in(6,s2y3y6y8).
in(8,s2y3y6y8).
in(2,s2y3y6y9).
in(3,s2y3y6y9).
in(6,s2y3y6y9).
in(9,s2y3y6y9).
in(2,s2y3y7y8).
in(3,s2y3y7y8).
in(7,s2y3y7y8).
in(8,s2y3y7y8).
in(2,s2y3y7y9).
in(3,s2y3y7y9).
in(7,s2y3y7y9).
in(9,s2y3y7y9).
in(2,s2y3y8y9).
in(3,s2y3y8y9).
in(8,s2y3y8y9).
in(9,s2y3y8y9).
in(2,s2y4y5y6).
in(4,s2y4y5y6).
in(5,s2y4y5y6).
in(6,s2y4y5y6).
in(2,s2y4y5y7).
in(4,s2y4y5y7).
in(5,s2y4y5y7).
in(7,s2y4y5y7).
in(2,s2y4y5y8).
in(4,s2y4y5y8).
in(5,s2y4y5y8).
in(8,s2y4y5y8).
in(2,s2y4y5y9).
in(4,s2y4y5y9).
in(5,s2y4y5y9).
in(9,s2y4y5y9).
in(2,s2y4y6y7).
in(4,s2y4y6y7).
in(6,s2y4y6y7).
in(7,s2y4y6y7).
in(2,s2y4y6y8).
in(4,s2y4y6y8).
in(6,s2y4y6y8).
in(8,s2y4y6y8).
in(2,s2y4y6y9).
in(4,s2y4y6y9).
in(6,s2y4y6y9).
in(9,s2y4y6y9).
in(2,s2y4y7y8).
in(4,s2y4y7y8).
in(7,s2y4y7y8).
in(8,s2y4y7y8).
in(2,s2y4y7y9).
in(4,s2y4y7y9).
in(7,s2y4y7y9).
in(9,s2y4y7y9).
in(2,s2y4y8y9).
in(4,s2y4y8y9).
in(8,s2y4y8y9).
in(9,s2y4y8y9).
in(2,s2y5y6y7).
in(5,s2y5y6y7).
in(6,s2y5y6y7).
in(7,s2y5y6y7).
in(2,s2y5y6y8).
in(5,s2y5y6y8).
in(6,s2y5y6y8).
in(8,s2y5y6y8).
in(2,s2y5y6y9).
in(5,s2y5y6y9).
in(6,s2y5y6y9).
in(9,s2y5y6y9).
in(2,s2y5y7y8).
in(5,s2y5y7y8).
in(7,s2y5y7y8).
in(8,s2y5y7y8).
in(2,s2y5y7y9).
in(5,s2y5y7y9).
in(7,s2y5y7y9).
in(9,s2y5y7y9).
in(2,s2y5y8y9).
in(5,s2y5y8y9).
in(8,s2y5y8y9).
in(9,s2y5y8y9).
in(2,s2y6y7y8).
in(6,s2y6y7y8).
in(7,s2y6y7y8).
in(8,s2y6y7y8).
in(2,s2y6y7y9).
in(6,s2y6y7y9).
in(7,s2y6y7y9).
in(9,s2y6y7y9).
in(2,s2y6y8y9).
in(6,s2y6y8y9).
in(8,s2y6y8y9).
in(9,s2y6y8y9).
in(2,s2y7y8y9).
in(7,s2y7y8y9).
in(8,s2y7y8y9).
in(9,s2y7y8y9).
in(3,s3y4y5y6).
in(4,s3y4y5y6).
in(5,s3y4y5y6).
in(6,s3y4y5y6).
in(3,s3y4y5y7).
in(4,s3y4y5y7).
in(5,s3y4y5y7).
in(7,s3y4y5y7).
in(3,s3y4y5y8).
in(4,s3y4y5y8).
in(5,s3y4y5y8).
in(8,s3y4y5y8).
in(3,s3y4y5y9).
in(4,s3y4y5y9).
in(5,s3y4y5y9).
in(9,s3y4y5y9).
in(3,s3y4y6y7).
in(4,s3y4y6y7).
in(6,s3y4y6y7).
in(7,s3y4y6y7).
in(3,s3y4y6y8).
in(4,s3y4y6y8).
in(6,s3y4y6y8).
in(8,s3y4y6y8).
in(3,s3y4y6y9).
in(4,s3y4y6y9).
in(6,s3y4y6y9).
in(9,s3y4y6y9).
in(3,s3y4y7y8).
in(4,s3y4y7y8).
in(7,s3y4y7y8).
in(8,s3y4y7y8).
in(3,s3y4y7y9).
in(4,s3y4y7y9).
in(7,s3y4y7y9).
in(9,s3y4y7y9).
in(3,s3y4y8y9).
in(4,s3y4y8y9).
in(8,s3y4y8y9).
in(9,s3y4y8y9).
in(3,s3y5y6y7).
in(5,s3y5y6y7).
in(6,s3y5y6y7).
in(7,s3y5y6y7).
in(3,s3y5y6y8).
in(5,s3y5y6y8).
in(6,s3y5y6y8).
in(8,s3y5y6y8).
in(3,s3y5y6y9).
in(5,s3y5y6y9).
in(6,s3y5y6y9).
in(9,s3y5y6y9).
in(3,s3y5y7y8).
in(5,s3y5y7y8).
in(7,s3y5y7y8).
in(8,s3y5y7y8).
in(3,s3y5y7y9).
in(5,s3y5y7y9).
in(7,s3y5y7y9).
in(9,s3y5y7y9).
in(3,s3y5y8y9).
in(5,s3y5y8y9).
in(8,s3y5y8y9).
in(9,s3y5y8y9).
in(3,s3y6y7y8).
in(6,s3y6y7y8).
in(7,s3y6y7y8).
in(8,s3y6y7y8).
in(3,s3y6y7y9).
in(6,s3y6y7y9).
in(7,s3y6y7y9).
in(9,s3y6y7y9).
in(3,s3y6y8y9).
in(6,s3y6y8y9).
in(8,s3y6y8y9).
in(9,s3y6y8y9).
in(3,s3y7y8y9).
in(7,s3y7y8y9).
in(8,s3y7y8y9).
in(9,s3y7y8y9).
in(4,s4y5y6y7).
in(5,s4y5y6y7).
in(6,s4y5y6y7).
in(7,s4y5y6y7).
in(4,s4y5y6y8).
in(5,s4y5y6y8).
in(6,s4y5y6y8).
in(8,s4y5y6y8).
in(4,s4y5y6y9).
in(5,s4y5y6y9).
in(6,s4y5y6y9).
in(9,s4y5y6y9).
in(4,s4y5y7y8).
in(5,s4y5y7y8).
in(7,s4y5y7y8).
in(8,s4y5y7y8).
in(4,s4y5y7y9).
in(5,s4y5y7y9).
in(7,s4y5y7y9).
in(9,s4y5y7y9).
in(4,s4y5y8y9).
in(5,s4y5y8y9).
in(8,s4y5y8y9).
in(9,s4y5y8y9).
in(4,s4y6y7y8).
in(6,s4y6y7y8).
in(7,s4y6y7y8).
in(8,s4y6y7y8).
in(4,s4y6y7y9).
in(6,s4y6y7y9).
in(7,s4y6y7y9).
in(9,s4y6y7y9).
in(4,s4y6y8y9).
in(6,s4y6y8y9).
in(8,s4y6y8y9).
in(9,s4y6y8y9).
in(4,s4y7y8y9).
in(7,s4y7y8y9).
in(8,s4y7y8y9).
in(9,s4y7y8y9).
in(5,s5y6y7y8).
in(6,s5y6y7y8).
in(7,s5y6y7y8).
in(8,s5y6y7y8).
in(5,s5y6y7y9).
in(6,s5y6y7y9).
in(7,s5y6y7y9).
in(9,s5y6y7y9).
in(5,s5y6y8y9).
in(6,s5y6y8y9).
in(8,s5y6y8y9).
in(9,s5y6y8y9).
in(5,s5y7y8y9).
in(7,s5y7y8y9).
in(8,s5y7y8y9).
in(9,s5y7y8y9).
in(6,s6y7y8y9).
in(7,s6y7y8y9).
in(8,s6y7y8y9).
in(9,s6y7y8y9).
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
in(0,s0y1y2y3y8).
in(1,s0y1y2y3y8).
in(2,s0y1y2y3y8).
in(3,s0y1y2y3y8).
in(8,s0y1y2y3y8).
in(0,s0y1y2y3y9).
in(1,s0y1y2y3y9).
in(2,s0y1y2y3y9).
in(3,s0y1y2y3y9).
in(9,s0y1y2y3y9).
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
in(0,s0y1y2y4y8).
in(1,s0y1y2y4y8).
in(2,s0y1y2y4y8).
in(4,s0y1y2y4y8).
in(8,s0y1y2y4y8).
in(0,s0y1y2y4y9).
in(1,s0y1y2y4y9).
in(2,s0y1y2y4y9).
in(4,s0y1y2y4y9).
in(9,s0y1y2y4y9).
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
in(0,s0y1y2y5y8).
in(1,s0y1y2y5y8).
in(2,s0y1y2y5y8).
in(5,s0y1y2y5y8).
in(8,s0y1y2y5y8).
in(0,s0y1y2y5y9).
in(1,s0y1y2y5y9).
in(2,s0y1y2y5y9).
in(5,s0y1y2y5y9).
in(9,s0y1y2y5y9).
in(0,s0y1y2y6y7).
in(1,s0y1y2y6y7).
in(2,s0y1y2y6y7).
in(6,s0y1y2y6y7).
in(7,s0y1y2y6y7).
in(0,s0y1y2y6y8).
in(1,s0y1y2y6y8).
in(2,s0y1y2y6y8).
in(6,s0y1y2y6y8).
in(8,s0y1y2y6y8).
in(0,s0y1y2y6y9).
in(1,s0y1y2y6y9).
in(2,s0y1y2y6y9).
in(6,s0y1y2y6y9).
in(9,s0y1y2y6y9).
in(0,s0y1y2y7y8).
in(1,s0y1y2y7y8).
in(2,s0y1y2y7y8).
in(7,s0y1y2y7y8).
in(8,s0y1y2y7y8).
in(0,s0y1y2y7y9).
in(1,s0y1y2y7y9).
in(2,s0y1y2y7y9).
in(7,s0y1y2y7y9).
in(9,s0y1y2y7y9).
in(0,s0y1y2y8y9).
in(1,s0y1y2y8y9).
in(2,s0y1y2y8y9).
in(8,s0y1y2y8y9).
in(9,s0y1y2y8y9).
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
in(0,s0y1y3y4y8).
in(1,s0y1y3y4y8).
in(3,s0y1y3y4y8).
in(4,s0y1y3y4y8).
in(8,s0y1y3y4y8).
in(0,s0y1y3y4y9).
in(1,s0y1y3y4y9).
in(3,s0y1y3y4y9).
in(4,s0y1y3y4y9).
in(9,s0y1y3y4y9).
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
in(0,s0y1y3y5y8).
in(1,s0y1y3y5y8).
in(3,s0y1y3y5y8).
in(5,s0y1y3y5y8).
in(8,s0y1y3y5y8).
in(0,s0y1y3y5y9).
in(1,s0y1y3y5y9).
in(3,s0y1y3y5y9).
in(5,s0y1y3y5y9).
in(9,s0y1y3y5y9).
in(0,s0y1y3y6y7).
in(1,s0y1y3y6y7).
in(3,s0y1y3y6y7).
in(6,s0y1y3y6y7).
in(7,s0y1y3y6y7).
in(0,s0y1y3y6y8).
in(1,s0y1y3y6y8).
in(3,s0y1y3y6y8).
in(6,s0y1y3y6y8).
in(8,s0y1y3y6y8).
in(0,s0y1y3y6y9).
in(1,s0y1y3y6y9).
in(3,s0y1y3y6y9).
in(6,s0y1y3y6y9).
in(9,s0y1y3y6y9).
in(0,s0y1y3y7y8).
in(1,s0y1y3y7y8).
in(3,s0y1y3y7y8).
in(7,s0y1y3y7y8).
in(8,s0y1y3y7y8).
in(0,s0y1y3y7y9).
in(1,s0y1y3y7y9).
in(3,s0y1y3y7y9).
in(7,s0y1y3y7y9).
in(9,s0y1y3y7y9).
in(0,s0y1y3y8y9).
in(1,s0y1y3y8y9).
in(3,s0y1y3y8y9).
in(8,s0y1y3y8y9).
in(9,s0y1y3y8y9).
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
in(0,s0y1y4y5y8).
in(1,s0y1y4y5y8).
in(4,s0y1y4y5y8).
in(5,s0y1y4y5y8).
in(8,s0y1y4y5y8).
in(0,s0y1y4y5y9).
in(1,s0y1y4y5y9).
in(4,s0y1y4y5y9).
in(5,s0y1y4y5y9).
in(9,s0y1y4y5y9).
in(0,s0y1y4y6y7).
in(1,s0y1y4y6y7).
in(4,s0y1y4y6y7).
in(6,s0y1y4y6y7).
in(7,s0y1y4y6y7).
in(0,s0y1y4y6y8).
in(1,s0y1y4y6y8).
in(4,s0y1y4y6y8).
in(6,s0y1y4y6y8).
in(8,s0y1y4y6y8).
in(0,s0y1y4y6y9).
in(1,s0y1y4y6y9).
in(4,s0y1y4y6y9).
in(6,s0y1y4y6y9).
in(9,s0y1y4y6y9).
in(0,s0y1y4y7y8).
in(1,s0y1y4y7y8).
in(4,s0y1y4y7y8).
in(7,s0y1y4y7y8).
in(8,s0y1y4y7y8).
in(0,s0y1y4y7y9).
in(1,s0y1y4y7y9).
in(4,s0y1y4y7y9).
in(7,s0y1y4y7y9).
in(9,s0y1y4y7y9).
in(0,s0y1y4y8y9).
in(1,s0y1y4y8y9).
in(4,s0y1y4y8y9).
in(8,s0y1y4y8y9).
in(9,s0y1y4y8y9).
in(0,s0y1y5y6y7).
in(1,s0y1y5y6y7).
in(5,s0y1y5y6y7).
in(6,s0y1y5y6y7).
in(7,s0y1y5y6y7).
in(0,s0y1y5y6y8).
in(1,s0y1y5y6y8).
in(5,s0y1y5y6y8).
in(6,s0y1y5y6y8).
in(8,s0y1y5y6y8).
in(0,s0y1y5y6y9).
in(1,s0y1y5y6y9).
in(5,s0y1y5y6y9).
in(6,s0y1y5y6y9).
in(9,s0y1y5y6y9).
in(0,s0y1y5y7y8).
in(1,s0y1y5y7y8).
in(5,s0y1y5y7y8).
in(7,s0y1y5y7y8).
in(8,s0y1y5y7y8).
in(0,s0y1y5y7y9).
in(1,s0y1y5y7y9).
in(5,s0y1y5y7y9).
in(7,s0y1y5y7y9).
in(9,s0y1y5y7y9).
in(0,s0y1y5y8y9).
in(1,s0y1y5y8y9).
in(5,s0y1y5y8y9).
in(8,s0y1y5y8y9).
in(9,s0y1y5y8y9).
in(0,s0y1y6y7y8).
in(1,s0y1y6y7y8).
in(6,s0y1y6y7y8).
in(7,s0y1y6y7y8).
in(8,s0y1y6y7y8).
in(0,s0y1y6y7y9).
in(1,s0y1y6y7y9).
in(6,s0y1y6y7y9).
in(7,s0y1y6y7y9).
in(9,s0y1y6y7y9).
in(0,s0y1y6y8y9).
in(1,s0y1y6y8y9).
in(6,s0y1y6y8y9).
in(8,s0y1y6y8y9).
in(9,s0y1y6y8y9).
in(0,s0y1y7y8y9).
in(1,s0y1y7y8y9).
in(7,s0y1y7y8y9).
in(8,s0y1y7y8y9).
in(9,s0y1y7y8y9).
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
in(0,s0y2y3y4y8).
in(2,s0y2y3y4y8).
in(3,s0y2y3y4y8).
in(4,s0y2y3y4y8).
in(8,s0y2y3y4y8).
in(0,s0y2y3y4y9).
in(2,s0y2y3y4y9).
in(3,s0y2y3y4y9).
in(4,s0y2y3y4y9).
in(9,s0y2y3y4y9).
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
in(0,s0y2y3y5y8).
in(2,s0y2y3y5y8).
in(3,s0y2y3y5y8).
in(5,s0y2y3y5y8).
in(8,s0y2y3y5y8).
in(0,s0y2y3y5y9).
in(2,s0y2y3y5y9).
in(3,s0y2y3y5y9).
in(5,s0y2y3y5y9).
in(9,s0y2y3y5y9).
in(0,s0y2y3y6y7).
in(2,s0y2y3y6y7).
in(3,s0y2y3y6y7).
in(6,s0y2y3y6y7).
in(7,s0y2y3y6y7).
in(0,s0y2y3y6y8).
in(2,s0y2y3y6y8).
in(3,s0y2y3y6y8).
in(6,s0y2y3y6y8).
in(8,s0y2y3y6y8).
in(0,s0y2y3y6y9).
in(2,s0y2y3y6y9).
in(3,s0y2y3y6y9).
in(6,s0y2y3y6y9).
in(9,s0y2y3y6y9).
in(0,s0y2y3y7y8).
in(2,s0y2y3y7y8).
in(3,s0y2y3y7y8).
in(7,s0y2y3y7y8).
in(8,s0y2y3y7y8).
in(0,s0y2y3y7y9).
in(2,s0y2y3y7y9).
in(3,s0y2y3y7y9).
in(7,s0y2y3y7y9).
in(9,s0y2y3y7y9).
in(0,s0y2y3y8y9).
in(2,s0y2y3y8y9).
in(3,s0y2y3y8y9).
in(8,s0y2y3y8y9).
in(9,s0y2y3y8y9).
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
in(0,s0y2y4y5y8).
in(2,s0y2y4y5y8).
in(4,s0y2y4y5y8).
in(5,s0y2y4y5y8).
in(8,s0y2y4y5y8).
in(0,s0y2y4y5y9).
in(2,s0y2y4y5y9).
in(4,s0y2y4y5y9).
in(5,s0y2y4y5y9).
in(9,s0y2y4y5y9).
in(0,s0y2y4y6y7).
in(2,s0y2y4y6y7).
in(4,s0y2y4y6y7).
in(6,s0y2y4y6y7).
in(7,s0y2y4y6y7).
in(0,s0y2y4y6y8).
in(2,s0y2y4y6y8).
in(4,s0y2y4y6y8).
in(6,s0y2y4y6y8).
in(8,s0y2y4y6y8).
in(0,s0y2y4y6y9).
in(2,s0y2y4y6y9).
in(4,s0y2y4y6y9).
in(6,s0y2y4y6y9).
in(9,s0y2y4y6y9).
in(0,s0y2y4y7y8).
in(2,s0y2y4y7y8).
in(4,s0y2y4y7y8).
in(7,s0y2y4y7y8).
in(8,s0y2y4y7y8).
in(0,s0y2y4y7y9).
in(2,s0y2y4y7y9).
in(4,s0y2y4y7y9).
in(7,s0y2y4y7y9).
in(9,s0y2y4y7y9).
in(0,s0y2y4y8y9).
in(2,s0y2y4y8y9).
in(4,s0y2y4y8y9).
in(8,s0y2y4y8y9).
in(9,s0y2y4y8y9).
in(0,s0y2y5y6y7).
in(2,s0y2y5y6y7).
in(5,s0y2y5y6y7).
in(6,s0y2y5y6y7).
in(7,s0y2y5y6y7).
in(0,s0y2y5y6y8).
in(2,s0y2y5y6y8).
in(5,s0y2y5y6y8).
in(6,s0y2y5y6y8).
in(8,s0y2y5y6y8).
in(0,s0y2y5y6y9).
in(2,s0y2y5y6y9).
in(5,s0y2y5y6y9).
in(6,s0y2y5y6y9).
in(9,s0y2y5y6y9).
in(0,s0y2y5y7y8).
in(2,s0y2y5y7y8).
in(5,s0y2y5y7y8).
in(7,s0y2y5y7y8).
in(8,s0y2y5y7y8).
in(0,s0y2y5y7y9).
in(2,s0y2y5y7y9).
in(5,s0y2y5y7y9).
in(7,s0y2y5y7y9).
in(9,s0y2y5y7y9).
in(0,s0y2y5y8y9).
in(2,s0y2y5y8y9).
in(5,s0y2y5y8y9).
in(8,s0y2y5y8y9).
in(9,s0y2y5y8y9).
in(0,s0y2y6y7y8).
in(2,s0y2y6y7y8).
in(6,s0y2y6y7y8).
in(7,s0y2y6y7y8).
in(8,s0y2y6y7y8).
in(0,s0y2y6y7y9).
in(2,s0y2y6y7y9).
in(6,s0y2y6y7y9).
in(7,s0y2y6y7y9).
in(9,s0y2y6y7y9).
in(0,s0y2y6y8y9).
in(2,s0y2y6y8y9).
in(6,s0y2y6y8y9).
in(8,s0y2y6y8y9).
in(9,s0y2y6y8y9).
in(0,s0y2y7y8y9).
in(2,s0y2y7y8y9).
in(7,s0y2y7y8y9).
in(8,s0y2y7y8y9).
in(9,s0y2y7y8y9).
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
in(0,s0y3y4y5y8).
in(3,s0y3y4y5y8).
in(4,s0y3y4y5y8).
in(5,s0y3y4y5y8).
in(8,s0y3y4y5y8).
in(0,s0y3y4y5y9).
in(3,s0y3y4y5y9).
in(4,s0y3y4y5y9).
in(5,s0y3y4y5y9).
in(9,s0y3y4y5y9).
in(0,s0y3y4y6y7).
in(3,s0y3y4y6y7).
in(4,s0y3y4y6y7).
in(6,s0y3y4y6y7).
in(7,s0y3y4y6y7).
in(0,s0y3y4y6y8).
in(3,s0y3y4y6y8).
in(4,s0y3y4y6y8).
in(6,s0y3y4y6y8).
in(8,s0y3y4y6y8).
in(0,s0y3y4y6y9).
in(3,s0y3y4y6y9).
in(4,s0y3y4y6y9).
in(6,s0y3y4y6y9).
in(9,s0y3y4y6y9).
in(0,s0y3y4y7y8).
in(3,s0y3y4y7y8).
in(4,s0y3y4y7y8).
in(7,s0y3y4y7y8).
in(8,s0y3y4y7y8).
in(0,s0y3y4y7y9).
in(3,s0y3y4y7y9).
in(4,s0y3y4y7y9).
in(7,s0y3y4y7y9).
in(9,s0y3y4y7y9).
in(0,s0y3y4y8y9).
in(3,s0y3y4y8y9).
in(4,s0y3y4y8y9).
in(8,s0y3y4y8y9).
in(9,s0y3y4y8y9).
in(0,s0y3y5y6y7).
in(3,s0y3y5y6y7).
in(5,s0y3y5y6y7).
in(6,s0y3y5y6y7).
in(7,s0y3y5y6y7).
in(0,s0y3y5y6y8).
in(3,s0y3y5y6y8).
in(5,s0y3y5y6y8).
in(6,s0y3y5y6y8).
in(8,s0y3y5y6y8).
in(0,s0y3y5y6y9).
in(3,s0y3y5y6y9).
in(5,s0y3y5y6y9).
in(6,s0y3y5y6y9).
in(9,s0y3y5y6y9).
in(0,s0y3y5y7y8).
in(3,s0y3y5y7y8).
in(5,s0y3y5y7y8).
in(7,s0y3y5y7y8).
in(8,s0y3y5y7y8).
in(0,s0y3y5y7y9).
in(3,s0y3y5y7y9).
in(5,s0y3y5y7y9).
in(7,s0y3y5y7y9).
in(9,s0y3y5y7y9).
in(0,s0y3y5y8y9).
in(3,s0y3y5y8y9).
in(5,s0y3y5y8y9).
in(8,s0y3y5y8y9).
in(9,s0y3y5y8y9).
in(0,s0y3y6y7y8).
in(3,s0y3y6y7y8).
in(6,s0y3y6y7y8).
in(7,s0y3y6y7y8).
in(8,s0y3y6y7y8).
in(0,s0y3y6y7y9).
in(3,s0y3y6y7y9).
in(6,s0y3y6y7y9).
in(7,s0y3y6y7y9).
in(9,s0y3y6y7y9).
in(0,s0y3y6y8y9).
in(3,s0y3y6y8y9).
in(6,s0y3y6y8y9).
in(8,s0y3y6y8y9).
in(9,s0y3y6y8y9).
in(0,s0y3y7y8y9).
in(3,s0y3y7y8y9).
in(7,s0y3y7y8y9).
in(8,s0y3y7y8y9).
in(9,s0y3y7y8y9).
in(0,s0y4y5y6y7).
in(4,s0y4y5y6y7).
in(5,s0y4y5y6y7).
in(6,s0y4y5y6y7).
in(7,s0y4y5y6y7).
in(0,s0y4y5y6y8).
in(4,s0y4y5y6y8).
in(5,s0y4y5y6y8).
in(6,s0y4y5y6y8).
in(8,s0y4y5y6y8).
in(0,s0y4y5y6y9).
in(4,s0y4y5y6y9).
in(5,s0y4y5y6y9).
in(6,s0y4y5y6y9).
in(9,s0y4y5y6y9).
in(0,s0y4y5y7y8).
in(4,s0y4y5y7y8).
in(5,s0y4y5y7y8).
in(7,s0y4y5y7y8).
in(8,s0y4y5y7y8).
in(0,s0y4y5y7y9).
in(4,s0y4y5y7y9).
in(5,s0y4y5y7y9).
in(7,s0y4y5y7y9).
in(9,s0y4y5y7y9).
in(0,s0y4y5y8y9).
in(4,s0y4y5y8y9).
in(5,s0y4y5y8y9).
in(8,s0y4y5y8y9).
in(9,s0y4y5y8y9).
in(0,s0y4y6y7y8).
in(4,s0y4y6y7y8).
in(6,s0y4y6y7y8).
in(7,s0y4y6y7y8).
in(8,s0y4y6y7y8).
in(0,s0y4y6y7y9).
in(4,s0y4y6y7y9).
in(6,s0y4y6y7y9).
in(7,s0y4y6y7y9).
in(9,s0y4y6y7y9).
in(0,s0y4y6y8y9).
in(4,s0y4y6y8y9).
in(6,s0y4y6y8y9).
in(8,s0y4y6y8y9).
in(9,s0y4y6y8y9).
in(0,s0y4y7y8y9).
in(4,s0y4y7y8y9).
in(7,s0y4y7y8y9).
in(8,s0y4y7y8y9).
in(9,s0y4y7y8y9).
in(0,s0y5y6y7y8).
in(5,s0y5y6y7y8).
in(6,s0y5y6y7y8).
in(7,s0y5y6y7y8).
in(8,s0y5y6y7y8).
in(0,s0y5y6y7y9).
in(5,s0y5y6y7y9).
in(6,s0y5y6y7y9).
in(7,s0y5y6y7y9).
in(9,s0y5y6y7y9).
in(0,s0y5y6y8y9).
in(5,s0y5y6y8y9).
in(6,s0y5y6y8y9).
in(8,s0y5y6y8y9).
in(9,s0y5y6y8y9).
in(0,s0y5y7y8y9).
in(5,s0y5y7y8y9).
in(7,s0y5y7y8y9).
in(8,s0y5y7y8y9).
in(9,s0y5y7y8y9).
in(0,s0y6y7y8y9).
in(6,s0y6y7y8y9).
in(7,s0y6y7y8y9).
in(8,s0y6y7y8y9).
in(9,s0y6y7y8y9).
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
in(1,s1y2y3y4y8).
in(2,s1y2y3y4y8).
in(3,s1y2y3y4y8).
in(4,s1y2y3y4y8).
in(8,s1y2y3y4y8).
in(1,s1y2y3y4y9).
in(2,s1y2y3y4y9).
in(3,s1y2y3y4y9).
in(4,s1y2y3y4y9).
in(9,s1y2y3y4y9).
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
in(1,s1y2y3y5y8).
in(2,s1y2y3y5y8).
in(3,s1y2y3y5y8).
in(5,s1y2y3y5y8).
in(8,s1y2y3y5y8).
in(1,s1y2y3y5y9).
in(2,s1y2y3y5y9).
in(3,s1y2y3y5y9).
in(5,s1y2y3y5y9).
in(9,s1y2y3y5y9).
in(1,s1y2y3y6y7).
in(2,s1y2y3y6y7).
in(3,s1y2y3y6y7).
in(6,s1y2y3y6y7).
in(7,s1y2y3y6y7).
in(1,s1y2y3y6y8).
in(2,s1y2y3y6y8).
in(3,s1y2y3y6y8).
in(6,s1y2y3y6y8).
in(8,s1y2y3y6y8).
in(1,s1y2y3y6y9).
in(2,s1y2y3y6y9).
in(3,s1y2y3y6y9).
in(6,s1y2y3y6y9).
in(9,s1y2y3y6y9).
in(1,s1y2y3y7y8).
in(2,s1y2y3y7y8).
in(3,s1y2y3y7y8).
in(7,s1y2y3y7y8).
in(8,s1y2y3y7y8).
in(1,s1y2y3y7y9).
in(2,s1y2y3y7y9).
in(3,s1y2y3y7y9).
in(7,s1y2y3y7y9).
in(9,s1y2y3y7y9).
in(1,s1y2y3y8y9).
in(2,s1y2y3y8y9).
in(3,s1y2y3y8y9).
in(8,s1y2y3y8y9).
in(9,s1y2y3y8y9).
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
in(1,s1y2y4y5y8).
in(2,s1y2y4y5y8).
in(4,s1y2y4y5y8).
in(5,s1y2y4y5y8).
in(8,s1y2y4y5y8).
in(1,s1y2y4y5y9).
in(2,s1y2y4y5y9).
in(4,s1y2y4y5y9).
in(5,s1y2y4y5y9).
in(9,s1y2y4y5y9).
in(1,s1y2y4y6y7).
in(2,s1y2y4y6y7).
in(4,s1y2y4y6y7).
in(6,s1y2y4y6y7).
in(7,s1y2y4y6y7).
in(1,s1y2y4y6y8).
in(2,s1y2y4y6y8).
in(4,s1y2y4y6y8).
in(6,s1y2y4y6y8).
in(8,s1y2y4y6y8).
in(1,s1y2y4y6y9).
in(2,s1y2y4y6y9).
in(4,s1y2y4y6y9).
in(6,s1y2y4y6y9).
in(9,s1y2y4y6y9).
in(1,s1y2y4y7y8).
in(2,s1y2y4y7y8).
in(4,s1y2y4y7y8).
in(7,s1y2y4y7y8).
in(8,s1y2y4y7y8).
in(1,s1y2y4y7y9).
in(2,s1y2y4y7y9).
in(4,s1y2y4y7y9).
in(7,s1y2y4y7y9).
in(9,s1y2y4y7y9).
in(1,s1y2y4y8y9).
in(2,s1y2y4y8y9).
in(4,s1y2y4y8y9).
in(8,s1y2y4y8y9).
in(9,s1y2y4y8y9).
in(1,s1y2y5y6y7).
in(2,s1y2y5y6y7).
in(5,s1y2y5y6y7).
in(6,s1y2y5y6y7).
in(7,s1y2y5y6y7).
in(1,s1y2y5y6y8).
in(2,s1y2y5y6y8).
in(5,s1y2y5y6y8).
in(6,s1y2y5y6y8).
in(8,s1y2y5y6y8).
in(1,s1y2y5y6y9).
in(2,s1y2y5y6y9).
in(5,s1y2y5y6y9).
in(6,s1y2y5y6y9).
in(9,s1y2y5y6y9).
in(1,s1y2y5y7y8).
in(2,s1y2y5y7y8).
in(5,s1y2y5y7y8).
in(7,s1y2y5y7y8).
in(8,s1y2y5y7y8).
in(1,s1y2y5y7y9).
in(2,s1y2y5y7y9).
in(5,s1y2y5y7y9).
in(7,s1y2y5y7y9).
in(9,s1y2y5y7y9).
in(1,s1y2y5y8y9).
in(2,s1y2y5y8y9).
in(5,s1y2y5y8y9).
in(8,s1y2y5y8y9).
in(9,s1y2y5y8y9).
in(1,s1y2y6y7y8).
in(2,s1y2y6y7y8).
in(6,s1y2y6y7y8).
in(7,s1y2y6y7y8).
in(8,s1y2y6y7y8).
in(1,s1y2y6y7y9).
in(2,s1y2y6y7y9).
in(6,s1y2y6y7y9).
in(7,s1y2y6y7y9).
in(9,s1y2y6y7y9).
in(1,s1y2y6y8y9).
in(2,s1y2y6y8y9).
in(6,s1y2y6y8y9).
in(8,s1y2y6y8y9).
in(9,s1y2y6y8y9).
in(1,s1y2y7y8y9).
in(2,s1y2y7y8y9).
in(7,s1y2y7y8y9).
in(8,s1y2y7y8y9).
in(9,s1y2y7y8y9).
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
in(1,s1y3y4y5y8).
in(3,s1y3y4y5y8).
in(4,s1y3y4y5y8).
in(5,s1y3y4y5y8).
in(8,s1y3y4y5y8).
in(1,s1y3y4y5y9).
in(3,s1y3y4y5y9).
in(4,s1y3y4y5y9).
in(5,s1y3y4y5y9).
in(9,s1y3y4y5y9).
in(1,s1y3y4y6y7).
in(3,s1y3y4y6y7).
in(4,s1y3y4y6y7).
in(6,s1y3y4y6y7).
in(7,s1y3y4y6y7).
in(1,s1y3y4y6y8).
in(3,s1y3y4y6y8).
in(4,s1y3y4y6y8).
in(6,s1y3y4y6y8).
in(8,s1y3y4y6y8).
in(1,s1y3y4y6y9).
in(3,s1y3y4y6y9).
in(4,s1y3y4y6y9).
in(6,s1y3y4y6y9).
in(9,s1y3y4y6y9).
in(1,s1y3y4y7y8).
in(3,s1y3y4y7y8).
in(4,s1y3y4y7y8).
in(7,s1y3y4y7y8).
in(8,s1y3y4y7y8).
in(1,s1y3y4y7y9).
in(3,s1y3y4y7y9).
in(4,s1y3y4y7y9).
in(7,s1y3y4y7y9).
in(9,s1y3y4y7y9).
in(1,s1y3y4y8y9).
in(3,s1y3y4y8y9).
in(4,s1y3y4y8y9).
in(8,s1y3y4y8y9).
in(9,s1y3y4y8y9).
in(1,s1y3y5y6y7).
in(3,s1y3y5y6y7).
in(5,s1y3y5y6y7).
in(6,s1y3y5y6y7).
in(7,s1y3y5y6y7).
in(1,s1y3y5y6y8).
in(3,s1y3y5y6y8).
in(5,s1y3y5y6y8).
in(6,s1y3y5y6y8).
in(8,s1y3y5y6y8).
in(1,s1y3y5y6y9).
in(3,s1y3y5y6y9).
in(5,s1y3y5y6y9).
in(6,s1y3y5y6y9).
in(9,s1y3y5y6y9).
in(1,s1y3y5y7y8).
in(3,s1y3y5y7y8).
in(5,s1y3y5y7y8).
in(7,s1y3y5y7y8).
in(8,s1y3y5y7y8).
in(1,s1y3y5y7y9).
in(3,s1y3y5y7y9).
in(5,s1y3y5y7y9).
in(7,s1y3y5y7y9).
in(9,s1y3y5y7y9).
in(1,s1y3y5y8y9).
in(3,s1y3y5y8y9).
in(5,s1y3y5y8y9).
in(8,s1y3y5y8y9).
in(9,s1y3y5y8y9).
in(1,s1y3y6y7y8).
in(3,s1y3y6y7y8).
in(6,s1y3y6y7y8).
in(7,s1y3y6y7y8).
in(8,s1y3y6y7y8).
in(1,s1y3y6y7y9).
in(3,s1y3y6y7y9).
in(6,s1y3y6y7y9).
in(7,s1y3y6y7y9).
in(9,s1y3y6y7y9).
in(1,s1y3y6y8y9).
in(3,s1y3y6y8y9).
in(6,s1y3y6y8y9).
in(8,s1y3y6y8y9).
in(9,s1y3y6y8y9).
in(1,s1y3y7y8y9).
in(3,s1y3y7y8y9).
in(7,s1y3y7y8y9).
in(8,s1y3y7y8y9).
in(9,s1y3y7y8y9).
in(1,s1y4y5y6y7).
in(4,s1y4y5y6y7).
in(5,s1y4y5y6y7).
in(6,s1y4y5y6y7).
in(7,s1y4y5y6y7).
in(1,s1y4y5y6y8).
in(4,s1y4y5y6y8).
in(5,s1y4y5y6y8).
in(6,s1y4y5y6y8).
in(8,s1y4y5y6y8).
in(1,s1y4y5y6y9).
in(4,s1y4y5y6y9).
in(5,s1y4y5y6y9).
in(6,s1y4y5y6y9).
in(9,s1y4y5y6y9).
in(1,s1y4y5y7y8).
in(4,s1y4y5y7y8).
in(5,s1y4y5y7y8).
in(7,s1y4y5y7y8).
in(8,s1y4y5y7y8).
in(1,s1y4y5y7y9).
in(4,s1y4y5y7y9).
in(5,s1y4y5y7y9).
in(7,s1y4y5y7y9).
in(9,s1y4y5y7y9).
in(1,s1y4y5y8y9).
in(4,s1y4y5y8y9).
in(5,s1y4y5y8y9).
in(8,s1y4y5y8y9).
in(9,s1y4y5y8y9).
in(1,s1y4y6y7y8).
in(4,s1y4y6y7y8).
in(6,s1y4y6y7y8).
in(7,s1y4y6y7y8).
in(8,s1y4y6y7y8).
in(1,s1y4y6y7y9).
in(4,s1y4y6y7y9).
in(6,s1y4y6y7y9).
in(7,s1y4y6y7y9).
in(9,s1y4y6y7y9).
in(1,s1y4y6y8y9).
in(4,s1y4y6y8y9).
in(6,s1y4y6y8y9).
in(8,s1y4y6y8y9).
in(9,s1y4y6y8y9).
in(1,s1y4y7y8y9).
in(4,s1y4y7y8y9).
in(7,s1y4y7y8y9).
in(8,s1y4y7y8y9).
in(9,s1y4y7y8y9).
in(1,s1y5y6y7y8).
in(5,s1y5y6y7y8).
in(6,s1y5y6y7y8).
in(7,s1y5y6y7y8).
in(8,s1y5y6y7y8).
in(1,s1y5y6y7y9).
in(5,s1y5y6y7y9).
in(6,s1y5y6y7y9).
in(7,s1y5y6y7y9).
in(9,s1y5y6y7y9).
in(1,s1y5y6y8y9).
in(5,s1y5y6y8y9).
in(6,s1y5y6y8y9).
in(8,s1y5y6y8y9).
in(9,s1y5y6y8y9).
in(1,s1y5y7y8y9).
in(5,s1y5y7y8y9).
in(7,s1y5y7y8y9).
in(8,s1y5y7y8y9).
in(9,s1y5y7y8y9).
in(1,s1y6y7y8y9).
in(6,s1y6y7y8y9).
in(7,s1y6y7y8y9).
in(8,s1y6y7y8y9).
in(9,s1y6y7y8y9).
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
in(2,s2y3y4y5y8).
in(3,s2y3y4y5y8).
in(4,s2y3y4y5y8).
in(5,s2y3y4y5y8).
in(8,s2y3y4y5y8).
in(2,s2y3y4y5y9).
in(3,s2y3y4y5y9).
in(4,s2y3y4y5y9).
in(5,s2y3y4y5y9).
in(9,s2y3y4y5y9).
in(2,s2y3y4y6y7).
in(3,s2y3y4y6y7).
in(4,s2y3y4y6y7).
in(6,s2y3y4y6y7).
in(7,s2y3y4y6y7).
in(2,s2y3y4y6y8).
in(3,s2y3y4y6y8).
in(4,s2y3y4y6y8).
in(6,s2y3y4y6y8).
in(8,s2y3y4y6y8).
in(2,s2y3y4y6y9).
in(3,s2y3y4y6y9).
in(4,s2y3y4y6y9).
in(6,s2y3y4y6y9).
in(9,s2y3y4y6y9).
in(2,s2y3y4y7y8).
in(3,s2y3y4y7y8).
in(4,s2y3y4y7y8).
in(7,s2y3y4y7y8).
in(8,s2y3y4y7y8).
in(2,s2y3y4y7y9).
in(3,s2y3y4y7y9).
in(4,s2y3y4y7y9).
in(7,s2y3y4y7y9).
in(9,s2y3y4y7y9).
in(2,s2y3y4y8y9).
in(3,s2y3y4y8y9).
in(4,s2y3y4y8y9).
in(8,s2y3y4y8y9).
in(9,s2y3y4y8y9).
in(2,s2y3y5y6y7).
in(3,s2y3y5y6y7).
in(5,s2y3y5y6y7).
in(6,s2y3y5y6y7).
in(7,s2y3y5y6y7).
in(2,s2y3y5y6y8).
in(3,s2y3y5y6y8).
in(5,s2y3y5y6y8).
in(6,s2y3y5y6y8).
in(8,s2y3y5y6y8).
in(2,s2y3y5y6y9).
in(3,s2y3y5y6y9).
in(5,s2y3y5y6y9).
in(6,s2y3y5y6y9).
in(9,s2y3y5y6y9).
in(2,s2y3y5y7y8).
in(3,s2y3y5y7y8).
in(5,s2y3y5y7y8).
in(7,s2y3y5y7y8).
in(8,s2y3y5y7y8).
in(2,s2y3y5y7y9).
in(3,s2y3y5y7y9).
in(5,s2y3y5y7y9).
in(7,s2y3y5y7y9).
in(9,s2y3y5y7y9).
in(2,s2y3y5y8y9).
in(3,s2y3y5y8y9).
in(5,s2y3y5y8y9).
in(8,s2y3y5y8y9).
in(9,s2y3y5y8y9).
in(2,s2y3y6y7y8).
in(3,s2y3y6y7y8).
in(6,s2y3y6y7y8).
in(7,s2y3y6y7y8).
in(8,s2y3y6y7y8).
in(2,s2y3y6y7y9).
in(3,s2y3y6y7y9).
in(6,s2y3y6y7y9).
in(7,s2y3y6y7y9).
in(9,s2y3y6y7y9).
in(2,s2y3y6y8y9).
in(3,s2y3y6y8y9).
in(6,s2y3y6y8y9).
in(8,s2y3y6y8y9).
in(9,s2y3y6y8y9).
in(2,s2y3y7y8y9).
in(3,s2y3y7y8y9).
in(7,s2y3y7y8y9).
in(8,s2y3y7y8y9).
in(9,s2y3y7y8y9).
in(2,s2y4y5y6y7).
in(4,s2y4y5y6y7).
in(5,s2y4y5y6y7).
in(6,s2y4y5y6y7).
in(7,s2y4y5y6y7).
in(2,s2y4y5y6y8).
in(4,s2y4y5y6y8).
in(5,s2y4y5y6y8).
in(6,s2y4y5y6y8).
in(8,s2y4y5y6y8).
in(2,s2y4y5y6y9).
in(4,s2y4y5y6y9).
in(5,s2y4y5y6y9).
in(6,s2y4y5y6y9).
in(9,s2y4y5y6y9).
in(2,s2y4y5y7y8).
in(4,s2y4y5y7y8).
in(5,s2y4y5y7y8).
in(7,s2y4y5y7y8).
in(8,s2y4y5y7y8).
in(2,s2y4y5y7y9).
in(4,s2y4y5y7y9).
in(5,s2y4y5y7y9).
in(7,s2y4y5y7y9).
in(9,s2y4y5y7y9).
in(2,s2y4y5y8y9).
in(4,s2y4y5y8y9).
in(5,s2y4y5y8y9).
in(8,s2y4y5y8y9).
in(9,s2y4y5y8y9).
in(2,s2y4y6y7y8).
in(4,s2y4y6y7y8).
in(6,s2y4y6y7y8).
in(7,s2y4y6y7y8).
in(8,s2y4y6y7y8).
in(2,s2y4y6y7y9).
in(4,s2y4y6y7y9).
in(6,s2y4y6y7y9).
in(7,s2y4y6y7y9).
in(9,s2y4y6y7y9).
in(2,s2y4y6y8y9).
in(4,s2y4y6y8y9).
in(6,s2y4y6y8y9).
in(8,s2y4y6y8y9).
in(9,s2y4y6y8y9).
in(2,s2y4y7y8y9).
in(4,s2y4y7y8y9).
in(7,s2y4y7y8y9).
in(8,s2y4y7y8y9).
in(9,s2y4y7y8y9).
in(2,s2y5y6y7y8).
in(5,s2y5y6y7y8).
in(6,s2y5y6y7y8).
in(7,s2y5y6y7y8).
in(8,s2y5y6y7y8).
in(2,s2y5y6y7y9).
in(5,s2y5y6y7y9).
in(6,s2y5y6y7y9).
in(7,s2y5y6y7y9).
in(9,s2y5y6y7y9).
in(2,s2y5y6y8y9).
in(5,s2y5y6y8y9).
in(6,s2y5y6y8y9).
in(8,s2y5y6y8y9).
in(9,s2y5y6y8y9).
in(2,s2y5y7y8y9).
in(5,s2y5y7y8y9).
in(7,s2y5y7y8y9).
in(8,s2y5y7y8y9).
in(9,s2y5y7y8y9).
in(2,s2y6y7y8y9).
in(6,s2y6y7y8y9).
in(7,s2y6y7y8y9).
in(8,s2y6y7y8y9).
in(9,s2y6y7y8y9).
in(3,s3y4y5y6y7).
in(4,s3y4y5y6y7).
in(5,s3y4y5y6y7).
in(6,s3y4y5y6y7).
in(7,s3y4y5y6y7).
in(3,s3y4y5y6y8).
in(4,s3y4y5y6y8).
in(5,s3y4y5y6y8).
in(6,s3y4y5y6y8).
in(8,s3y4y5y6y8).
in(3,s3y4y5y6y9).
in(4,s3y4y5y6y9).
in(5,s3y4y5y6y9).
in(6,s3y4y5y6y9).
in(9,s3y4y5y6y9).
in(3,s3y4y5y7y8).
in(4,s3y4y5y7y8).
in(5,s3y4y5y7y8).
in(7,s3y4y5y7y8).
in(8,s3y4y5y7y8).
in(3,s3y4y5y7y9).
in(4,s3y4y5y7y9).
in(5,s3y4y5y7y9).
in(7,s3y4y5y7y9).
in(9,s3y4y5y7y9).
in(3,s3y4y5y8y9).
in(4,s3y4y5y8y9).
in(5,s3y4y5y8y9).
in(8,s3y4y5y8y9).
in(9,s3y4y5y8y9).
in(3,s3y4y6y7y8).
in(4,s3y4y6y7y8).
in(6,s3y4y6y7y8).
in(7,s3y4y6y7y8).
in(8,s3y4y6y7y8).
in(3,s3y4y6y7y9).
in(4,s3y4y6y7y9).
in(6,s3y4y6y7y9).
in(7,s3y4y6y7y9).
in(9,s3y4y6y7y9).
in(3,s3y4y6y8y9).
in(4,s3y4y6y8y9).
in(6,s3y4y6y8y9).
in(8,s3y4y6y8y9).
in(9,s3y4y6y8y9).
in(3,s3y4y7y8y9).
in(4,s3y4y7y8y9).
in(7,s3y4y7y8y9).
in(8,s3y4y7y8y9).
in(9,s3y4y7y8y9).
in(3,s3y5y6y7y8).
in(5,s3y5y6y7y8).
in(6,s3y5y6y7y8).
in(7,s3y5y6y7y8).
in(8,s3y5y6y7y8).
in(3,s3y5y6y7y9).
in(5,s3y5y6y7y9).
in(6,s3y5y6y7y9).
in(7,s3y5y6y7y9).
in(9,s3y5y6y7y9).
in(3,s3y5y6y8y9).
in(5,s3y5y6y8y9).
in(6,s3y5y6y8y9).
in(8,s3y5y6y8y9).
in(9,s3y5y6y8y9).
in(3,s3y5y7y8y9).
in(5,s3y5y7y8y9).
in(7,s3y5y7y8y9).
in(8,s3y5y7y8y9).
in(9,s3y5y7y8y9).
in(3,s3y6y7y8y9).
in(6,s3y6y7y8y9).
in(7,s3y6y7y8y9).
in(8,s3y6y7y8y9).
in(9,s3y6y7y8y9).
in(4,s4y5y6y7y8).
in(5,s4y5y6y7y8).
in(6,s4y5y6y7y8).
in(7,s4y5y6y7y8).
in(8,s4y5y6y7y8).
in(4,s4y5y6y7y9).
in(5,s4y5y6y7y9).
in(6,s4y5y6y7y9).
in(7,s4y5y6y7y9).
in(9,s4y5y6y7y9).
in(4,s4y5y6y8y9).
in(5,s4y5y6y8y9).
in(6,s4y5y6y8y9).
in(8,s4y5y6y8y9).
in(9,s4y5y6y8y9).
in(4,s4y5y7y8y9).
in(5,s4y5y7y8y9).
in(7,s4y5y7y8y9).
in(8,s4y5y7y8y9).
in(9,s4y5y7y8y9).
in(4,s4y6y7y8y9).
in(6,s4y6y7y8y9).
in(7,s4y6y7y8y9).
in(8,s4y6y7y8y9).
in(9,s4y6y7y8y9).
in(5,s5y6y7y8y9).
in(6,s5y6y7y8y9).
in(7,s5y6y7y8y9).
in(8,s5y6y7y8y9).
in(9,s5y6y7y8y9).
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
in(0,s0y1y2y3y4y8).
in(1,s0y1y2y3y4y8).
in(2,s0y1y2y3y4y8).
in(3,s0y1y2y3y4y8).
in(4,s0y1y2y3y4y8).
in(8,s0y1y2y3y4y8).
in(0,s0y1y2y3y4y9).
in(1,s0y1y2y3y4y9).
in(2,s0y1y2y3y4y9).
in(3,s0y1y2y3y4y9).
in(4,s0y1y2y3y4y9).
in(9,s0y1y2y3y4y9).
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
in(0,s0y1y2y3y5y8).
in(1,s0y1y2y3y5y8).
in(2,s0y1y2y3y5y8).
in(3,s0y1y2y3y5y8).
in(5,s0y1y2y3y5y8).
in(8,s0y1y2y3y5y8).
in(0,s0y1y2y3y5y9).
in(1,s0y1y2y3y5y9).
in(2,s0y1y2y3y5y9).
in(3,s0y1y2y3y5y9).
in(5,s0y1y2y3y5y9).
in(9,s0y1y2y3y5y9).
in(0,s0y1y2y3y6y7).
in(1,s0y1y2y3y6y7).
in(2,s0y1y2y3y6y7).
in(3,s0y1y2y3y6y7).
in(6,s0y1y2y3y6y7).
in(7,s0y1y2y3y6y7).
in(0,s0y1y2y3y6y8).
in(1,s0y1y2y3y6y8).
in(2,s0y1y2y3y6y8).
in(3,s0y1y2y3y6y8).
in(6,s0y1y2y3y6y8).
in(8,s0y1y2y3y6y8).
in(0,s0y1y2y3y6y9).
in(1,s0y1y2y3y6y9).
in(2,s0y1y2y3y6y9).
in(3,s0y1y2y3y6y9).
in(6,s0y1y2y3y6y9).
in(9,s0y1y2y3y6y9).
in(0,s0y1y2y3y7y8).
in(1,s0y1y2y3y7y8).
in(2,s0y1y2y3y7y8).
in(3,s0y1y2y3y7y8).
in(7,s0y1y2y3y7y8).
in(8,s0y1y2y3y7y8).
in(0,s0y1y2y3y7y9).
in(1,s0y1y2y3y7y9).
in(2,s0y1y2y3y7y9).
in(3,s0y1y2y3y7y9).
in(7,s0y1y2y3y7y9).
in(9,s0y1y2y3y7y9).
in(0,s0y1y2y3y8y9).
in(1,s0y1y2y3y8y9).
in(2,s0y1y2y3y8y9).
in(3,s0y1y2y3y8y9).
in(8,s0y1y2y3y8y9).
in(9,s0y1y2y3y8y9).
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
in(0,s0y1y2y4y5y8).
in(1,s0y1y2y4y5y8).
in(2,s0y1y2y4y5y8).
in(4,s0y1y2y4y5y8).
in(5,s0y1y2y4y5y8).
in(8,s0y1y2y4y5y8).
in(0,s0y1y2y4y5y9).
in(1,s0y1y2y4y5y9).
in(2,s0y1y2y4y5y9).
in(4,s0y1y2y4y5y9).
in(5,s0y1y2y4y5y9).
in(9,s0y1y2y4y5y9).
in(0,s0y1y2y4y6y7).
in(1,s0y1y2y4y6y7).
in(2,s0y1y2y4y6y7).
in(4,s0y1y2y4y6y7).
in(6,s0y1y2y4y6y7).
in(7,s0y1y2y4y6y7).
in(0,s0y1y2y4y6y8).
in(1,s0y1y2y4y6y8).
in(2,s0y1y2y4y6y8).
in(4,s0y1y2y4y6y8).
in(6,s0y1y2y4y6y8).
in(8,s0y1y2y4y6y8).
in(0,s0y1y2y4y6y9).
in(1,s0y1y2y4y6y9).
in(2,s0y1y2y4y6y9).
in(4,s0y1y2y4y6y9).
in(6,s0y1y2y4y6y9).
in(9,s0y1y2y4y6y9).
in(0,s0y1y2y4y7y8).
in(1,s0y1y2y4y7y8).
in(2,s0y1y2y4y7y8).
in(4,s0y1y2y4y7y8).
in(7,s0y1y2y4y7y8).
in(8,s0y1y2y4y7y8).
in(0,s0y1y2y4y7y9).
in(1,s0y1y2y4y7y9).
in(2,s0y1y2y4y7y9).
in(4,s0y1y2y4y7y9).
in(7,s0y1y2y4y7y9).
in(9,s0y1y2y4y7y9).
in(0,s0y1y2y4y8y9).
in(1,s0y1y2y4y8y9).
in(2,s0y1y2y4y8y9).
in(4,s0y1y2y4y8y9).
in(8,s0y1y2y4y8y9).
in(9,s0y1y2y4y8y9).
in(0,s0y1y2y5y6y7).
in(1,s0y1y2y5y6y7).
in(2,s0y1y2y5y6y7).
in(5,s0y1y2y5y6y7).
in(6,s0y1y2y5y6y7).
in(7,s0y1y2y5y6y7).
in(0,s0y1y2y5y6y8).
in(1,s0y1y2y5y6y8).
in(2,s0y1y2y5y6y8).
in(5,s0y1y2y5y6y8).
in(6,s0y1y2y5y6y8).
in(8,s0y1y2y5y6y8).
in(0,s0y1y2y5y6y9).
in(1,s0y1y2y5y6y9).
in(2,s0y1y2y5y6y9).
in(5,s0y1y2y5y6y9).
in(6,s0y1y2y5y6y9).
in(9,s0y1y2y5y6y9).
in(0,s0y1y2y5y7y8).
in(1,s0y1y2y5y7y8).
in(2,s0y1y2y5y7y8).
in(5,s0y1y2y5y7y8).
in(7,s0y1y2y5y7y8).
in(8,s0y1y2y5y7y8).
in(0,s0y1y2y5y7y9).
in(1,s0y1y2y5y7y9).
in(2,s0y1y2y5y7y9).
in(5,s0y1y2y5y7y9).
in(7,s0y1y2y5y7y9).
in(9,s0y1y2y5y7y9).
in(0,s0y1y2y5y8y9).
in(1,s0y1y2y5y8y9).
in(2,s0y1y2y5y8y9).
in(5,s0y1y2y5y8y9).
in(8,s0y1y2y5y8y9).
in(9,s0y1y2y5y8y9).
in(0,s0y1y2y6y7y8).
in(1,s0y1y2y6y7y8).
in(2,s0y1y2y6y7y8).
in(6,s0y1y2y6y7y8).
in(7,s0y1y2y6y7y8).
in(8,s0y1y2y6y7y8).
in(0,s0y1y2y6y7y9).
in(1,s0y1y2y6y7y9).
in(2,s0y1y2y6y7y9).
in(6,s0y1y2y6y7y9).
in(7,s0y1y2y6y7y9).
in(9,s0y1y2y6y7y9).
in(0,s0y1y2y6y8y9).
in(1,s0y1y2y6y8y9).
in(2,s0y1y2y6y8y9).
in(6,s0y1y2y6y8y9).
in(8,s0y1y2y6y8y9).
in(9,s0y1y2y6y8y9).
in(0,s0y1y2y7y8y9).
in(1,s0y1y2y7y8y9).
in(2,s0y1y2y7y8y9).
in(7,s0y1y2y7y8y9).
in(8,s0y1y2y7y8y9).
in(9,s0y1y2y7y8y9).
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
in(0,s0y1y3y4y5y8).
in(1,s0y1y3y4y5y8).
in(3,s0y1y3y4y5y8).
in(4,s0y1y3y4y5y8).
in(5,s0y1y3y4y5y8).
in(8,s0y1y3y4y5y8).
in(0,s0y1y3y4y5y9).
in(1,s0y1y3y4y5y9).
in(3,s0y1y3y4y5y9).
in(4,s0y1y3y4y5y9).
in(5,s0y1y3y4y5y9).
in(9,s0y1y3y4y5y9).
in(0,s0y1y3y4y6y7).
in(1,s0y1y3y4y6y7).
in(3,s0y1y3y4y6y7).
in(4,s0y1y3y4y6y7).
in(6,s0y1y3y4y6y7).
in(7,s0y1y3y4y6y7).
in(0,s0y1y3y4y6y8).
in(1,s0y1y3y4y6y8).
in(3,s0y1y3y4y6y8).
in(4,s0y1y3y4y6y8).
in(6,s0y1y3y4y6y8).
in(8,s0y1y3y4y6y8).
in(0,s0y1y3y4y6y9).
in(1,s0y1y3y4y6y9).
in(3,s0y1y3y4y6y9).
in(4,s0y1y3y4y6y9).
in(6,s0y1y3y4y6y9).
in(9,s0y1y3y4y6y9).
in(0,s0y1y3y4y7y8).
in(1,s0y1y3y4y7y8).
in(3,s0y1y3y4y7y8).
in(4,s0y1y3y4y7y8).
in(7,s0y1y3y4y7y8).
in(8,s0y1y3y4y7y8).
in(0,s0y1y3y4y7y9).
in(1,s0y1y3y4y7y9).
in(3,s0y1y3y4y7y9).
in(4,s0y1y3y4y7y9).
in(7,s0y1y3y4y7y9).
in(9,s0y1y3y4y7y9).
in(0,s0y1y3y4y8y9).
in(1,s0y1y3y4y8y9).
in(3,s0y1y3y4y8y9).
in(4,s0y1y3y4y8y9).
in(8,s0y1y3y4y8y9).
in(9,s0y1y3y4y8y9).
in(0,s0y1y3y5y6y7).
in(1,s0y1y3y5y6y7).
in(3,s0y1y3y5y6y7).
in(5,s0y1y3y5y6y7).
in(6,s0y1y3y5y6y7).
in(7,s0y1y3y5y6y7).
in(0,s0y1y3y5y6y8).
in(1,s0y1y3y5y6y8).
in(3,s0y1y3y5y6y8).
in(5,s0y1y3y5y6y8).
in(6,s0y1y3y5y6y8).
in(8,s0y1y3y5y6y8).
in(0,s0y1y3y5y6y9).
in(1,s0y1y3y5y6y9).
in(3,s0y1y3y5y6y9).
in(5,s0y1y3y5y6y9).
in(6,s0y1y3y5y6y9).
in(9,s0y1y3y5y6y9).
in(0,s0y1y3y5y7y8).
in(1,s0y1y3y5y7y8).
in(3,s0y1y3y5y7y8).
in(5,s0y1y3y5y7y8).
in(7,s0y1y3y5y7y8).
in(8,s0y1y3y5y7y8).
in(0,s0y1y3y5y7y9).
in(1,s0y1y3y5y7y9).
in(3,s0y1y3y5y7y9).
in(5,s0y1y3y5y7y9).
in(7,s0y1y3y5y7y9).
in(9,s0y1y3y5y7y9).
in(0,s0y1y3y5y8y9).
in(1,s0y1y3y5y8y9).
in(3,s0y1y3y5y8y9).
in(5,s0y1y3y5y8y9).
in(8,s0y1y3y5y8y9).
in(9,s0y1y3y5y8y9).
in(0,s0y1y3y6y7y8).
in(1,s0y1y3y6y7y8).
in(3,s0y1y3y6y7y8).
in(6,s0y1y3y6y7y8).
in(7,s0y1y3y6y7y8).
in(8,s0y1y3y6y7y8).
in(0,s0y1y3y6y7y9).
in(1,s0y1y3y6y7y9).
in(3,s0y1y3y6y7y9).
in(6,s0y1y3y6y7y9).
in(7,s0y1y3y6y7y9).
in(9,s0y1y3y6y7y9).
in(0,s0y1y3y6y8y9).
in(1,s0y1y3y6y8y9).
in(3,s0y1y3y6y8y9).
in(6,s0y1y3y6y8y9).
in(8,s0y1y3y6y8y9).
in(9,s0y1y3y6y8y9).
in(0,s0y1y3y7y8y9).
in(1,s0y1y3y7y8y9).
in(3,s0y1y3y7y8y9).
in(7,s0y1y3y7y8y9).
in(8,s0y1y3y7y8y9).
in(9,s0y1y3y7y8y9).
in(0,s0y1y4y5y6y7).
in(1,s0y1y4y5y6y7).
in(4,s0y1y4y5y6y7).
in(5,s0y1y4y5y6y7).
in(6,s0y1y4y5y6y7).
in(7,s0y1y4y5y6y7).
in(0,s0y1y4y5y6y8).
in(1,s0y1y4y5y6y8).
in(4,s0y1y4y5y6y8).
in(5,s0y1y4y5y6y8).
in(6,s0y1y4y5y6y8).
in(8,s0y1y4y5y6y8).
in(0,s0y1y4y5y6y9).
in(1,s0y1y4y5y6y9).
in(4,s0y1y4y5y6y9).
in(5,s0y1y4y5y6y9).
in(6,s0y1y4y5y6y9).
in(9,s0y1y4y5y6y9).
in(0,s0y1y4y5y7y8).
in(1,s0y1y4y5y7y8).
in(4,s0y1y4y5y7y8).
in(5,s0y1y4y5y7y8).
in(7,s0y1y4y5y7y8).
in(8,s0y1y4y5y7y8).
in(0,s0y1y4y5y7y9).
in(1,s0y1y4y5y7y9).
in(4,s0y1y4y5y7y9).
in(5,s0y1y4y5y7y9).
in(7,s0y1y4y5y7y9).
in(9,s0y1y4y5y7y9).
in(0,s0y1y4y5y8y9).
in(1,s0y1y4y5y8y9).
in(4,s0y1y4y5y8y9).
in(5,s0y1y4y5y8y9).
in(8,s0y1y4y5y8y9).
in(9,s0y1y4y5y8y9).
in(0,s0y1y4y6y7y8).
in(1,s0y1y4y6y7y8).
in(4,s0y1y4y6y7y8).
in(6,s0y1y4y6y7y8).
in(7,s0y1y4y6y7y8).
in(8,s0y1y4y6y7y8).
in(0,s0y1y4y6y7y9).
in(1,s0y1y4y6y7y9).
in(4,s0y1y4y6y7y9).
in(6,s0y1y4y6y7y9).
in(7,s0y1y4y6y7y9).
in(9,s0y1y4y6y7y9).
in(0,s0y1y4y6y8y9).
in(1,s0y1y4y6y8y9).
in(4,s0y1y4y6y8y9).
in(6,s0y1y4y6y8y9).
in(8,s0y1y4y6y8y9).
in(9,s0y1y4y6y8y9).
in(0,s0y1y4y7y8y9).
in(1,s0y1y4y7y8y9).
in(4,s0y1y4y7y8y9).
in(7,s0y1y4y7y8y9).
in(8,s0y1y4y7y8y9).
in(9,s0y1y4y7y8y9).
in(0,s0y1y5y6y7y8).
in(1,s0y1y5y6y7y8).
in(5,s0y1y5y6y7y8).
in(6,s0y1y5y6y7y8).
in(7,s0y1y5y6y7y8).
in(8,s0y1y5y6y7y8).
in(0,s0y1y5y6y7y9).
in(1,s0y1y5y6y7y9).
in(5,s0y1y5y6y7y9).
in(6,s0y1y5y6y7y9).
in(7,s0y1y5y6y7y9).
in(9,s0y1y5y6y7y9).
in(0,s0y1y5y6y8y9).
in(1,s0y1y5y6y8y9).
in(5,s0y1y5y6y8y9).
in(6,s0y1y5y6y8y9).
in(8,s0y1y5y6y8y9).
in(9,s0y1y5y6y8y9).
in(0,s0y1y5y7y8y9).
in(1,s0y1y5y7y8y9).
in(5,s0y1y5y7y8y9).
in(7,s0y1y5y7y8y9).
in(8,s0y1y5y7y8y9).
in(9,s0y1y5y7y8y9).
in(0,s0y1y6y7y8y9).
in(1,s0y1y6y7y8y9).
in(6,s0y1y6y7y8y9).
in(7,s0y1y6y7y8y9).
in(8,s0y1y6y7y8y9).
in(9,s0y1y6y7y8y9).
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
in(0,s0y2y3y4y5y8).
in(2,s0y2y3y4y5y8).
in(3,s0y2y3y4y5y8).
in(4,s0y2y3y4y5y8).
in(5,s0y2y3y4y5y8).
in(8,s0y2y3y4y5y8).
in(0,s0y2y3y4y5y9).
in(2,s0y2y3y4y5y9).
in(3,s0y2y3y4y5y9).
in(4,s0y2y3y4y5y9).
in(5,s0y2y3y4y5y9).
in(9,s0y2y3y4y5y9).
in(0,s0y2y3y4y6y7).
in(2,s0y2y3y4y6y7).
in(3,s0y2y3y4y6y7).
in(4,s0y2y3y4y6y7).
in(6,s0y2y3y4y6y7).
in(7,s0y2y3y4y6y7).
in(0,s0y2y3y4y6y8).
in(2,s0y2y3y4y6y8).
in(3,s0y2y3y4y6y8).
in(4,s0y2y3y4y6y8).
in(6,s0y2y3y4y6y8).
in(8,s0y2y3y4y6y8).
in(0,s0y2y3y4y6y9).
in(2,s0y2y3y4y6y9).
in(3,s0y2y3y4y6y9).
in(4,s0y2y3y4y6y9).
in(6,s0y2y3y4y6y9).
in(9,s0y2y3y4y6y9).
in(0,s0y2y3y4y7y8).
in(2,s0y2y3y4y7y8).
in(3,s0y2y3y4y7y8).
in(4,s0y2y3y4y7y8).
in(7,s0y2y3y4y7y8).
in(8,s0y2y3y4y7y8).
in(0,s0y2y3y4y7y9).
in(2,s0y2y3y4y7y9).
in(3,s0y2y3y4y7y9).
in(4,s0y2y3y4y7y9).
in(7,s0y2y3y4y7y9).
in(9,s0y2y3y4y7y9).
in(0,s0y2y3y4y8y9).
in(2,s0y2y3y4y8y9).
in(3,s0y2y3y4y8y9).
in(4,s0y2y3y4y8y9).
in(8,s0y2y3y4y8y9).
in(9,s0y2y3y4y8y9).
in(0,s0y2y3y5y6y7).
in(2,s0y2y3y5y6y7).
in(3,s0y2y3y5y6y7).
in(5,s0y2y3y5y6y7).
in(6,s0y2y3y5y6y7).
in(7,s0y2y3y5y6y7).
in(0,s0y2y3y5y6y8).
in(2,s0y2y3y5y6y8).
in(3,s0y2y3y5y6y8).
in(5,s0y2y3y5y6y8).
in(6,s0y2y3y5y6y8).
in(8,s0y2y3y5y6y8).
in(0,s0y2y3y5y6y9).
in(2,s0y2y3y5y6y9).
in(3,s0y2y3y5y6y9).
in(5,s0y2y3y5y6y9).
in(6,s0y2y3y5y6y9).
in(9,s0y2y3y5y6y9).
in(0,s0y2y3y5y7y8).
in(2,s0y2y3y5y7y8).
in(3,s0y2y3y5y7y8).
in(5,s0y2y3y5y7y8).
in(7,s0y2y3y5y7y8).
in(8,s0y2y3y5y7y8).
in(0,s0y2y3y5y7y9).
in(2,s0y2y3y5y7y9).
in(3,s0y2y3y5y7y9).
in(5,s0y2y3y5y7y9).
in(7,s0y2y3y5y7y9).
in(9,s0y2y3y5y7y9).
in(0,s0y2y3y5y8y9).
in(2,s0y2y3y5y8y9).
in(3,s0y2y3y5y8y9).
in(5,s0y2y3y5y8y9).
in(8,s0y2y3y5y8y9).
in(9,s0y2y3y5y8y9).
in(0,s0y2y3y6y7y8).
in(2,s0y2y3y6y7y8).
in(3,s0y2y3y6y7y8).
in(6,s0y2y3y6y7y8).
in(7,s0y2y3y6y7y8).
in(8,s0y2y3y6y7y8).
in(0,s0y2y3y6y7y9).
in(2,s0y2y3y6y7y9).
in(3,s0y2y3y6y7y9).
in(6,s0y2y3y6y7y9).
in(7,s0y2y3y6y7y9).
in(9,s0y2y3y6y7y9).
in(0,s0y2y3y6y8y9).
in(2,s0y2y3y6y8y9).
in(3,s0y2y3y6y8y9).
in(6,s0y2y3y6y8y9).
in(8,s0y2y3y6y8y9).
in(9,s0y2y3y6y8y9).
in(0,s0y2y3y7y8y9).
in(2,s0y2y3y7y8y9).
in(3,s0y2y3y7y8y9).
in(7,s0y2y3y7y8y9).
in(8,s0y2y3y7y8y9).
in(9,s0y2y3y7y8y9).
in(0,s0y2y4y5y6y7).
in(2,s0y2y4y5y6y7).
in(4,s0y2y4y5y6y7).
in(5,s0y2y4y5y6y7).
in(6,s0y2y4y5y6y7).
in(7,s0y2y4y5y6y7).
in(0,s0y2y4y5y6y8).
in(2,s0y2y4y5y6y8).
in(4,s0y2y4y5y6y8).
in(5,s0y2y4y5y6y8).
in(6,s0y2y4y5y6y8).
in(8,s0y2y4y5y6y8).
in(0,s0y2y4y5y6y9).
in(2,s0y2y4y5y6y9).
in(4,s0y2y4y5y6y9).
in(5,s0y2y4y5y6y9).
in(6,s0y2y4y5y6y9).
in(9,s0y2y4y5y6y9).
in(0,s0y2y4y5y7y8).
in(2,s0y2y4y5y7y8).
in(4,s0y2y4y5y7y8).
in(5,s0y2y4y5y7y8).
in(7,s0y2y4y5y7y8).
in(8,s0y2y4y5y7y8).
in(0,s0y2y4y5y7y9).
in(2,s0y2y4y5y7y9).
in(4,s0y2y4y5y7y9).
in(5,s0y2y4y5y7y9).
in(7,s0y2y4y5y7y9).
in(9,s0y2y4y5y7y9).
in(0,s0y2y4y5y8y9).
in(2,s0y2y4y5y8y9).
in(4,s0y2y4y5y8y9).
in(5,s0y2y4y5y8y9).
in(8,s0y2y4y5y8y9).
in(9,s0y2y4y5y8y9).
in(0,s0y2y4y6y7y8).
in(2,s0y2y4y6y7y8).
in(4,s0y2y4y6y7y8).
in(6,s0y2y4y6y7y8).
in(7,s0y2y4y6y7y8).
in(8,s0y2y4y6y7y8).
in(0,s0y2y4y6y7y9).
in(2,s0y2y4y6y7y9).
in(4,s0y2y4y6y7y9).
in(6,s0y2y4y6y7y9).
in(7,s0y2y4y6y7y9).
in(9,s0y2y4y6y7y9).
in(0,s0y2y4y6y8y9).
in(2,s0y2y4y6y8y9).
in(4,s0y2y4y6y8y9).
in(6,s0y2y4y6y8y9).
in(8,s0y2y4y6y8y9).
in(9,s0y2y4y6y8y9).
in(0,s0y2y4y7y8y9).
in(2,s0y2y4y7y8y9).
in(4,s0y2y4y7y8y9).
in(7,s0y2y4y7y8y9).
in(8,s0y2y4y7y8y9).
in(9,s0y2y4y7y8y9).
in(0,s0y2y5y6y7y8).
in(2,s0y2y5y6y7y8).
in(5,s0y2y5y6y7y8).
in(6,s0y2y5y6y7y8).
in(7,s0y2y5y6y7y8).
in(8,s0y2y5y6y7y8).
in(0,s0y2y5y6y7y9).
in(2,s0y2y5y6y7y9).
in(5,s0y2y5y6y7y9).
in(6,s0y2y5y6y7y9).
in(7,s0y2y5y6y7y9).
in(9,s0y2y5y6y7y9).
in(0,s0y2y5y6y8y9).
in(2,s0y2y5y6y8y9).
in(5,s0y2y5y6y8y9).
in(6,s0y2y5y6y8y9).
in(8,s0y2y5y6y8y9).
in(9,s0y2y5y6y8y9).
in(0,s0y2y5y7y8y9).
in(2,s0y2y5y7y8y9).
in(5,s0y2y5y7y8y9).
in(7,s0y2y5y7y8y9).
in(8,s0y2y5y7y8y9).
in(9,s0y2y5y7y8y9).
in(0,s0y2y6y7y8y9).
in(2,s0y2y6y7y8y9).
in(6,s0y2y6y7y8y9).
in(7,s0y2y6y7y8y9).
in(8,s0y2y6y7y8y9).
in(9,s0y2y6y7y8y9).
in(0,s0y3y4y5y6y7).
in(3,s0y3y4y5y6y7).
in(4,s0y3y4y5y6y7).
in(5,s0y3y4y5y6y7).
in(6,s0y3y4y5y6y7).
in(7,s0y3y4y5y6y7).
in(0,s0y3y4y5y6y8).
in(3,s0y3y4y5y6y8).
in(4,s0y3y4y5y6y8).
in(5,s0y3y4y5y6y8).
in(6,s0y3y4y5y6y8).
in(8,s0y3y4y5y6y8).
in(0,s0y3y4y5y6y9).
in(3,s0y3y4y5y6y9).
in(4,s0y3y4y5y6y9).
in(5,s0y3y4y5y6y9).
in(6,s0y3y4y5y6y9).
in(9,s0y3y4y5y6y9).
in(0,s0y3y4y5y7y8).
in(3,s0y3y4y5y7y8).
in(4,s0y3y4y5y7y8).
in(5,s0y3y4y5y7y8).
in(7,s0y3y4y5y7y8).
in(8,s0y3y4y5y7y8).
in(0,s0y3y4y5y7y9).
in(3,s0y3y4y5y7y9).
in(4,s0y3y4y5y7y9).
in(5,s0y3y4y5y7y9).
in(7,s0y3y4y5y7y9).
in(9,s0y3y4y5y7y9).
in(0,s0y3y4y5y8y9).
in(3,s0y3y4y5y8y9).
in(4,s0y3y4y5y8y9).
in(5,s0y3y4y5y8y9).
in(8,s0y3y4y5y8y9).
in(9,s0y3y4y5y8y9).
in(0,s0y3y4y6y7y8).
in(3,s0y3y4y6y7y8).
in(4,s0y3y4y6y7y8).
in(6,s0y3y4y6y7y8).
in(7,s0y3y4y6y7y8).
in(8,s0y3y4y6y7y8).
in(0,s0y3y4y6y7y9).
in(3,s0y3y4y6y7y9).
in(4,s0y3y4y6y7y9).
in(6,s0y3y4y6y7y9).
in(7,s0y3y4y6y7y9).
in(9,s0y3y4y6y7y9).
in(0,s0y3y4y6y8y9).
in(3,s0y3y4y6y8y9).
in(4,s0y3y4y6y8y9).
in(6,s0y3y4y6y8y9).
in(8,s0y3y4y6y8y9).
in(9,s0y3y4y6y8y9).
in(0,s0y3y4y7y8y9).
in(3,s0y3y4y7y8y9).
in(4,s0y3y4y7y8y9).
in(7,s0y3y4y7y8y9).
in(8,s0y3y4y7y8y9).
in(9,s0y3y4y7y8y9).
in(0,s0y3y5y6y7y8).
in(3,s0y3y5y6y7y8).
in(5,s0y3y5y6y7y8).
in(6,s0y3y5y6y7y8).
in(7,s0y3y5y6y7y8).
in(8,s0y3y5y6y7y8).
in(0,s0y3y5y6y7y9).
in(3,s0y3y5y6y7y9).
in(5,s0y3y5y6y7y9).
in(6,s0y3y5y6y7y9).
in(7,s0y3y5y6y7y9).
in(9,s0y3y5y6y7y9).
in(0,s0y3y5y6y8y9).
in(3,s0y3y5y6y8y9).
in(5,s0y3y5y6y8y9).
in(6,s0y3y5y6y8y9).
in(8,s0y3y5y6y8y9).
in(9,s0y3y5y6y8y9).
in(0,s0y3y5y7y8y9).
in(3,s0y3y5y7y8y9).
in(5,s0y3y5y7y8y9).
in(7,s0y3y5y7y8y9).
in(8,s0y3y5y7y8y9).
in(9,s0y3y5y7y8y9).
in(0,s0y3y6y7y8y9).
in(3,s0y3y6y7y8y9).
in(6,s0y3y6y7y8y9).
in(7,s0y3y6y7y8y9).
in(8,s0y3y6y7y8y9).
in(9,s0y3y6y7y8y9).
in(0,s0y4y5y6y7y8).
in(4,s0y4y5y6y7y8).
in(5,s0y4y5y6y7y8).
in(6,s0y4y5y6y7y8).
in(7,s0y4y5y6y7y8).
in(8,s0y4y5y6y7y8).
in(0,s0y4y5y6y7y9).
in(4,s0y4y5y6y7y9).
in(5,s0y4y5y6y7y9).
in(6,s0y4y5y6y7y9).
in(7,s0y4y5y6y7y9).
in(9,s0y4y5y6y7y9).
in(0,s0y4y5y6y8y9).
in(4,s0y4y5y6y8y9).
in(5,s0y4y5y6y8y9).
in(6,s0y4y5y6y8y9).
in(8,s0y4y5y6y8y9).
in(9,s0y4y5y6y8y9).
in(0,s0y4y5y7y8y9).
in(4,s0y4y5y7y8y9).
in(5,s0y4y5y7y8y9).
in(7,s0y4y5y7y8y9).
in(8,s0y4y5y7y8y9).
in(9,s0y4y5y7y8y9).
in(0,s0y4y6y7y8y9).
in(4,s0y4y6y7y8y9).
in(6,s0y4y6y7y8y9).
in(7,s0y4y6y7y8y9).
in(8,s0y4y6y7y8y9).
in(9,s0y4y6y7y8y9).
in(0,s0y5y6y7y8y9).
in(5,s0y5y6y7y8y9).
in(6,s0y5y6y7y8y9).
in(7,s0y5y6y7y8y9).
in(8,s0y5y6y7y8y9).
in(9,s0y5y6y7y8y9).
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
in(1,s1y2y3y4y5y8).
in(2,s1y2y3y4y5y8).
in(3,s1y2y3y4y5y8).
in(4,s1y2y3y4y5y8).
in(5,s1y2y3y4y5y8).
in(8,s1y2y3y4y5y8).
in(1,s1y2y3y4y5y9).
in(2,s1y2y3y4y5y9).
in(3,s1y2y3y4y5y9).
in(4,s1y2y3y4y5y9).
in(5,s1y2y3y4y5y9).
in(9,s1y2y3y4y5y9).
in(1,s1y2y3y4y6y7).
in(2,s1y2y3y4y6y7).
in(3,s1y2y3y4y6y7).
in(4,s1y2y3y4y6y7).
in(6,s1y2y3y4y6y7).
in(7,s1y2y3y4y6y7).
in(1,s1y2y3y4y6y8).
in(2,s1y2y3y4y6y8).
in(3,s1y2y3y4y6y8).
in(4,s1y2y3y4y6y8).
in(6,s1y2y3y4y6y8).
in(8,s1y2y3y4y6y8).
in(1,s1y2y3y4y6y9).
in(2,s1y2y3y4y6y9).
in(3,s1y2y3y4y6y9).
in(4,s1y2y3y4y6y9).
in(6,s1y2y3y4y6y9).
in(9,s1y2y3y4y6y9).
in(1,s1y2y3y4y7y8).
in(2,s1y2y3y4y7y8).
in(3,s1y2y3y4y7y8).
in(4,s1y2y3y4y7y8).
in(7,s1y2y3y4y7y8).
in(8,s1y2y3y4y7y8).
in(1,s1y2y3y4y7y9).
in(2,s1y2y3y4y7y9).
in(3,s1y2y3y4y7y9).
in(4,s1y2y3y4y7y9).
in(7,s1y2y3y4y7y9).
in(9,s1y2y3y4y7y9).
in(1,s1y2y3y4y8y9).
in(2,s1y2y3y4y8y9).
in(3,s1y2y3y4y8y9).
in(4,s1y2y3y4y8y9).
in(8,s1y2y3y4y8y9).
in(9,s1y2y3y4y8y9).
in(1,s1y2y3y5y6y7).
in(2,s1y2y3y5y6y7).
in(3,s1y2y3y5y6y7).
in(5,s1y2y3y5y6y7).
in(6,s1y2y3y5y6y7).
in(7,s1y2y3y5y6y7).
in(1,s1y2y3y5y6y8).
in(2,s1y2y3y5y6y8).
in(3,s1y2y3y5y6y8).
in(5,s1y2y3y5y6y8).
in(6,s1y2y3y5y6y8).
in(8,s1y2y3y5y6y8).
in(1,s1y2y3y5y6y9).
in(2,s1y2y3y5y6y9).
in(3,s1y2y3y5y6y9).
in(5,s1y2y3y5y6y9).
in(6,s1y2y3y5y6y9).
in(9,s1y2y3y5y6y9).
in(1,s1y2y3y5y7y8).
in(2,s1y2y3y5y7y8).
in(3,s1y2y3y5y7y8).
in(5,s1y2y3y5y7y8).
in(7,s1y2y3y5y7y8).
in(8,s1y2y3y5y7y8).
in(1,s1y2y3y5y7y9).
in(2,s1y2y3y5y7y9).
in(3,s1y2y3y5y7y9).
in(5,s1y2y3y5y7y9).
in(7,s1y2y3y5y7y9).
in(9,s1y2y3y5y7y9).
in(1,s1y2y3y5y8y9).
in(2,s1y2y3y5y8y9).
in(3,s1y2y3y5y8y9).
in(5,s1y2y3y5y8y9).
in(8,s1y2y3y5y8y9).
in(9,s1y2y3y5y8y9).
in(1,s1y2y3y6y7y8).
in(2,s1y2y3y6y7y8).
in(3,s1y2y3y6y7y8).
in(6,s1y2y3y6y7y8).
in(7,s1y2y3y6y7y8).
in(8,s1y2y3y6y7y8).
in(1,s1y2y3y6y7y9).
in(2,s1y2y3y6y7y9).
in(3,s1y2y3y6y7y9).
in(6,s1y2y3y6y7y9).
in(7,s1y2y3y6y7y9).
in(9,s1y2y3y6y7y9).
in(1,s1y2y3y6y8y9).
in(2,s1y2y3y6y8y9).
in(3,s1y2y3y6y8y9).
in(6,s1y2y3y6y8y9).
in(8,s1y2y3y6y8y9).
in(9,s1y2y3y6y8y9).
in(1,s1y2y3y7y8y9).
in(2,s1y2y3y7y8y9).
in(3,s1y2y3y7y8y9).
in(7,s1y2y3y7y8y9).
in(8,s1y2y3y7y8y9).
in(9,s1y2y3y7y8y9).
in(1,s1y2y4y5y6y7).
in(2,s1y2y4y5y6y7).
in(4,s1y2y4y5y6y7).
in(5,s1y2y4y5y6y7).
in(6,s1y2y4y5y6y7).
in(7,s1y2y4y5y6y7).
in(1,s1y2y4y5y6y8).
in(2,s1y2y4y5y6y8).
in(4,s1y2y4y5y6y8).
in(5,s1y2y4y5y6y8).
in(6,s1y2y4y5y6y8).
in(8,s1y2y4y5y6y8).
in(1,s1y2y4y5y6y9).
in(2,s1y2y4y5y6y9).
in(4,s1y2y4y5y6y9).
in(5,s1y2y4y5y6y9).
in(6,s1y2y4y5y6y9).
in(9,s1y2y4y5y6y9).
in(1,s1y2y4y5y7y8).
in(2,s1y2y4y5y7y8).
in(4,s1y2y4y5y7y8).
in(5,s1y2y4y5y7y8).
in(7,s1y2y4y5y7y8).
in(8,s1y2y4y5y7y8).
in(1,s1y2y4y5y7y9).
in(2,s1y2y4y5y7y9).
in(4,s1y2y4y5y7y9).
in(5,s1y2y4y5y7y9).
in(7,s1y2y4y5y7y9).
in(9,s1y2y4y5y7y9).
in(1,s1y2y4y5y8y9).
in(2,s1y2y4y5y8y9).
in(4,s1y2y4y5y8y9).
in(5,s1y2y4y5y8y9).
in(8,s1y2y4y5y8y9).
in(9,s1y2y4y5y8y9).
in(1,s1y2y4y6y7y8).
in(2,s1y2y4y6y7y8).
in(4,s1y2y4y6y7y8).
in(6,s1y2y4y6y7y8).
in(7,s1y2y4y6y7y8).
in(8,s1y2y4y6y7y8).
in(1,s1y2y4y6y7y9).
in(2,s1y2y4y6y7y9).
in(4,s1y2y4y6y7y9).
in(6,s1y2y4y6y7y9).
in(7,s1y2y4y6y7y9).
in(9,s1y2y4y6y7y9).
in(1,s1y2y4y6y8y9).
in(2,s1y2y4y6y8y9).
in(4,s1y2y4y6y8y9).
in(6,s1y2y4y6y8y9).
in(8,s1y2y4y6y8y9).
in(9,s1y2y4y6y8y9).
in(1,s1y2y4y7y8y9).
in(2,s1y2y4y7y8y9).
in(4,s1y2y4y7y8y9).
in(7,s1y2y4y7y8y9).
in(8,s1y2y4y7y8y9).
in(9,s1y2y4y7y8y9).
in(1,s1y2y5y6y7y8).
in(2,s1y2y5y6y7y8).
in(5,s1y2y5y6y7y8).
in(6,s1y2y5y6y7y8).
in(7,s1y2y5y6y7y8).
in(8,s1y2y5y6y7y8).
in(1,s1y2y5y6y7y9).
in(2,s1y2y5y6y7y9).
in(5,s1y2y5y6y7y9).
in(6,s1y2y5y6y7y9).
in(7,s1y2y5y6y7y9).
in(9,s1y2y5y6y7y9).
in(1,s1y2y5y6y8y9).
in(2,s1y2y5y6y8y9).
in(5,s1y2y5y6y8y9).
in(6,s1y2y5y6y8y9).
in(8,s1y2y5y6y8y9).
in(9,s1y2y5y6y8y9).
in(1,s1y2y5y7y8y9).
in(2,s1y2y5y7y8y9).
in(5,s1y2y5y7y8y9).
in(7,s1y2y5y7y8y9).
in(8,s1y2y5y7y8y9).
in(9,s1y2y5y7y8y9).
in(1,s1y2y6y7y8y9).
in(2,s1y2y6y7y8y9).
in(6,s1y2y6y7y8y9).
in(7,s1y2y6y7y8y9).
in(8,s1y2y6y7y8y9).
in(9,s1y2y6y7y8y9).
in(1,s1y3y4y5y6y7).
in(3,s1y3y4y5y6y7).
in(4,s1y3y4y5y6y7).
in(5,s1y3y4y5y6y7).
in(6,s1y3y4y5y6y7).
in(7,s1y3y4y5y6y7).
in(1,s1y3y4y5y6y8).
in(3,s1y3y4y5y6y8).
in(4,s1y3y4y5y6y8).
in(5,s1y3y4y5y6y8).
in(6,s1y3y4y5y6y8).
in(8,s1y3y4y5y6y8).
in(1,s1y3y4y5y6y9).
in(3,s1y3y4y5y6y9).
in(4,s1y3y4y5y6y9).
in(5,s1y3y4y5y6y9).
in(6,s1y3y4y5y6y9).
in(9,s1y3y4y5y6y9).
in(1,s1y3y4y5y7y8).
in(3,s1y3y4y5y7y8).
in(4,s1y3y4y5y7y8).
in(5,s1y3y4y5y7y8).
in(7,s1y3y4y5y7y8).
in(8,s1y3y4y5y7y8).
in(1,s1y3y4y5y7y9).
in(3,s1y3y4y5y7y9).
in(4,s1y3y4y5y7y9).
in(5,s1y3y4y5y7y9).
in(7,s1y3y4y5y7y9).
in(9,s1y3y4y5y7y9).
in(1,s1y3y4y5y8y9).
in(3,s1y3y4y5y8y9).
in(4,s1y3y4y5y8y9).
in(5,s1y3y4y5y8y9).
in(8,s1y3y4y5y8y9).
in(9,s1y3y4y5y8y9).
in(1,s1y3y4y6y7y8).
in(3,s1y3y4y6y7y8).
in(4,s1y3y4y6y7y8).
in(6,s1y3y4y6y7y8).
in(7,s1y3y4y6y7y8).
in(8,s1y3y4y6y7y8).
in(1,s1y3y4y6y7y9).
in(3,s1y3y4y6y7y9).
in(4,s1y3y4y6y7y9).
in(6,s1y3y4y6y7y9).
in(7,s1y3y4y6y7y9).
in(9,s1y3y4y6y7y9).
in(1,s1y3y4y6y8y9).
in(3,s1y3y4y6y8y9).
in(4,s1y3y4y6y8y9).
in(6,s1y3y4y6y8y9).
in(8,s1y3y4y6y8y9).
in(9,s1y3y4y6y8y9).
in(1,s1y3y4y7y8y9).
in(3,s1y3y4y7y8y9).
in(4,s1y3y4y7y8y9).
in(7,s1y3y4y7y8y9).
in(8,s1y3y4y7y8y9).
in(9,s1y3y4y7y8y9).
in(1,s1y3y5y6y7y8).
in(3,s1y3y5y6y7y8).
in(5,s1y3y5y6y7y8).
in(6,s1y3y5y6y7y8).
in(7,s1y3y5y6y7y8).
in(8,s1y3y5y6y7y8).
in(1,s1y3y5y6y7y9).
in(3,s1y3y5y6y7y9).
in(5,s1y3y5y6y7y9).
in(6,s1y3y5y6y7y9).
in(7,s1y3y5y6y7y9).
in(9,s1y3y5y6y7y9).
in(1,s1y3y5y6y8y9).
in(3,s1y3y5y6y8y9).
in(5,s1y3y5y6y8y9).
in(6,s1y3y5y6y8y9).
in(8,s1y3y5y6y8y9).
in(9,s1y3y5y6y8y9).
in(1,s1y3y5y7y8y9).
in(3,s1y3y5y7y8y9).
in(5,s1y3y5y7y8y9).
in(7,s1y3y5y7y8y9).
in(8,s1y3y5y7y8y9).
in(9,s1y3y5y7y8y9).
in(1,s1y3y6y7y8y9).
in(3,s1y3y6y7y8y9).
in(6,s1y3y6y7y8y9).
in(7,s1y3y6y7y8y9).
in(8,s1y3y6y7y8y9).
in(9,s1y3y6y7y8y9).
in(1,s1y4y5y6y7y8).
in(4,s1y4y5y6y7y8).
in(5,s1y4y5y6y7y8).
in(6,s1y4y5y6y7y8).
in(7,s1y4y5y6y7y8).
in(8,s1y4y5y6y7y8).
in(1,s1y4y5y6y7y9).
in(4,s1y4y5y6y7y9).
in(5,s1y4y5y6y7y9).
in(6,s1y4y5y6y7y9).
in(7,s1y4y5y6y7y9).
in(9,s1y4y5y6y7y9).
in(1,s1y4y5y6y8y9).
in(4,s1y4y5y6y8y9).
in(5,s1y4y5y6y8y9).
in(6,s1y4y5y6y8y9).
in(8,s1y4y5y6y8y9).
in(9,s1y4y5y6y8y9).
in(1,s1y4y5y7y8y9).
in(4,s1y4y5y7y8y9).
in(5,s1y4y5y7y8y9).
in(7,s1y4y5y7y8y9).
in(8,s1y4y5y7y8y9).
in(9,s1y4y5y7y8y9).
in(1,s1y4y6y7y8y9).
in(4,s1y4y6y7y8y9).
in(6,s1y4y6y7y8y9).
in(7,s1y4y6y7y8y9).
in(8,s1y4y6y7y8y9).
in(9,s1y4y6y7y8y9).
in(1,s1y5y6y7y8y9).
in(5,s1y5y6y7y8y9).
in(6,s1y5y6y7y8y9).
in(7,s1y5y6y7y8y9).
in(8,s1y5y6y7y8y9).
in(9,s1y5y6y7y8y9).
in(2,s2y3y4y5y6y7).
in(3,s2y3y4y5y6y7).
in(4,s2y3y4y5y6y7).
in(5,s2y3y4y5y6y7).
in(6,s2y3y4y5y6y7).
in(7,s2y3y4y5y6y7).
in(2,s2y3y4y5y6y8).
in(3,s2y3y4y5y6y8).
in(4,s2y3y4y5y6y8).
in(5,s2y3y4y5y6y8).
in(6,s2y3y4y5y6y8).
in(8,s2y3y4y5y6y8).
in(2,s2y3y4y5y6y9).
in(3,s2y3y4y5y6y9).
in(4,s2y3y4y5y6y9).
in(5,s2y3y4y5y6y9).
in(6,s2y3y4y5y6y9).
in(9,s2y3y4y5y6y9).
in(2,s2y3y4y5y7y8).
in(3,s2y3y4y5y7y8).
in(4,s2y3y4y5y7y8).
in(5,s2y3y4y5y7y8).
in(7,s2y3y4y5y7y8).
in(8,s2y3y4y5y7y8).
in(2,s2y3y4y5y7y9).
in(3,s2y3y4y5y7y9).
in(4,s2y3y4y5y7y9).
in(5,s2y3y4y5y7y9).
in(7,s2y3y4y5y7y9).
in(9,s2y3y4y5y7y9).
in(2,s2y3y4y5y8y9).
in(3,s2y3y4y5y8y9).
in(4,s2y3y4y5y8y9).
in(5,s2y3y4y5y8y9).
in(8,s2y3y4y5y8y9).
in(9,s2y3y4y5y8y9).
in(2,s2y3y4y6y7y8).
in(3,s2y3y4y6y7y8).
in(4,s2y3y4y6y7y8).
in(6,s2y3y4y6y7y8).
in(7,s2y3y4y6y7y8).
in(8,s2y3y4y6y7y8).
in(2,s2y3y4y6y7y9).
in(3,s2y3y4y6y7y9).
in(4,s2y3y4y6y7y9).
in(6,s2y3y4y6y7y9).
in(7,s2y3y4y6y7y9).
in(9,s2y3y4y6y7y9).
in(2,s2y3y4y6y8y9).
in(3,s2y3y4y6y8y9).
in(4,s2y3y4y6y8y9).
in(6,s2y3y4y6y8y9).
in(8,s2y3y4y6y8y9).
in(9,s2y3y4y6y8y9).
in(2,s2y3y4y7y8y9).
in(3,s2y3y4y7y8y9).
in(4,s2y3y4y7y8y9).
in(7,s2y3y4y7y8y9).
in(8,s2y3y4y7y8y9).
in(9,s2y3y4y7y8y9).
in(2,s2y3y5y6y7y8).
in(3,s2y3y5y6y7y8).
in(5,s2y3y5y6y7y8).
in(6,s2y3y5y6y7y8).
in(7,s2y3y5y6y7y8).
in(8,s2y3y5y6y7y8).
in(2,s2y3y5y6y7y9).
in(3,s2y3y5y6y7y9).
in(5,s2y3y5y6y7y9).
in(6,s2y3y5y6y7y9).
in(7,s2y3y5y6y7y9).
in(9,s2y3y5y6y7y9).
in(2,s2y3y5y6y8y9).
in(3,s2y3y5y6y8y9).
in(5,s2y3y5y6y8y9).
in(6,s2y3y5y6y8y9).
in(8,s2y3y5y6y8y9).
in(9,s2y3y5y6y8y9).
in(2,s2y3y5y7y8y9).
in(3,s2y3y5y7y8y9).
in(5,s2y3y5y7y8y9).
in(7,s2y3y5y7y8y9).
in(8,s2y3y5y7y8y9).
in(9,s2y3y5y7y8y9).
in(2,s2y3y6y7y8y9).
in(3,s2y3y6y7y8y9).
in(6,s2y3y6y7y8y9).
in(7,s2y3y6y7y8y9).
in(8,s2y3y6y7y8y9).
in(9,s2y3y6y7y8y9).
in(2,s2y4y5y6y7y8).
in(4,s2y4y5y6y7y8).
in(5,s2y4y5y6y7y8).
in(6,s2y4y5y6y7y8).
in(7,s2y4y5y6y7y8).
in(8,s2y4y5y6y7y8).
in(2,s2y4y5y6y7y9).
in(4,s2y4y5y6y7y9).
in(5,s2y4y5y6y7y9).
in(6,s2y4y5y6y7y9).
in(7,s2y4y5y6y7y9).
in(9,s2y4y5y6y7y9).
in(2,s2y4y5y6y8y9).
in(4,s2y4y5y6y8y9).
in(5,s2y4y5y6y8y9).
in(6,s2y4y5y6y8y9).
in(8,s2y4y5y6y8y9).
in(9,s2y4y5y6y8y9).
in(2,s2y4y5y7y8y9).
in(4,s2y4y5y7y8y9).
in(5,s2y4y5y7y8y9).
in(7,s2y4y5y7y8y9).
in(8,s2y4y5y7y8y9).
in(9,s2y4y5y7y8y9).
in(2,s2y4y6y7y8y9).
in(4,s2y4y6y7y8y9).
in(6,s2y4y6y7y8y9).
in(7,s2y4y6y7y8y9).
in(8,s2y4y6y7y8y9).
in(9,s2y4y6y7y8y9).
in(2,s2y5y6y7y8y9).
in(5,s2y5y6y7y8y9).
in(6,s2y5y6y7y8y9).
in(7,s2y5y6y7y8y9).
in(8,s2y5y6y7y8y9).
in(9,s2y5y6y7y8y9).
in(3,s3y4y5y6y7y8).
in(4,s3y4y5y6y7y8).
in(5,s3y4y5y6y7y8).
in(6,s3y4y5y6y7y8).
in(7,s3y4y5y6y7y8).
in(8,s3y4y5y6y7y8).
in(3,s3y4y5y6y7y9).
in(4,s3y4y5y6y7y9).
in(5,s3y4y5y6y7y9).
in(6,s3y4y5y6y7y9).
in(7,s3y4y5y6y7y9).
in(9,s3y4y5y6y7y9).
in(3,s3y4y5y6y8y9).
in(4,s3y4y5y6y8y9).
in(5,s3y4y5y6y8y9).
in(6,s3y4y5y6y8y9).
in(8,s3y4y5y6y8y9).
in(9,s3y4y5y6y8y9).
in(3,s3y4y5y7y8y9).
in(4,s3y4y5y7y8y9).
in(5,s3y4y5y7y8y9).
in(7,s3y4y5y7y8y9).
in(8,s3y4y5y7y8y9).
in(9,s3y4y5y7y8y9).
in(3,s3y4y6y7y8y9).
in(4,s3y4y6y7y8y9).
in(6,s3y4y6y7y8y9).
in(7,s3y4y6y7y8y9).
in(8,s3y4y6y7y8y9).
in(9,s3y4y6y7y8y9).
in(3,s3y5y6y7y8y9).
in(5,s3y5y6y7y8y9).
in(6,s3y5y6y7y8y9).
in(7,s3y5y6y7y8y9).
in(8,s3y5y6y7y8y9).
in(9,s3y5y6y7y8y9).
in(4,s4y5y6y7y8y9).
in(5,s4y5y6y7y8y9).
in(6,s4y5y6y7y8y9).
in(7,s4y5y6y7y8y9).
in(8,s4y5y6y7y8y9).
in(9,s4y5y6y7y8y9).
qpair(0,1,s2).
qpair(0,1,s2y8y9).
qpair(0,1,s3y4).
qpair(0,1,s2y3y7y8).
qpair(0,1,s5).
qpair(0,1,s4y9).
qpair(0,1,s3y7).
qpair(0,1,s2y7y8y9).
qpair(0,1,s8).
qpair(0,1,s3y7y9).
qpair(0,1,s8y9).
qpair(0,1,s4y7y9).
qpair(0,1,s2y3y8y9).
qpair(0,1,s3y4y7).
qpair(0,1,s2y8).
qpair(0,1,s2y3y9).
qpair(0,1,s2y4y8).
qpair(0,1,s4).
qpair(0,1,s7).
qpair(0,1,s3y9).
qpair(0,1,s4y8).
qpair(0,1,s3y7y8).
qpair(0,1,s7y8y9).
qpair(0,1,s2y7y9).
qpair(0,1,s2y4).
qpair(0,1,s4y7y8).
qpair(0,1,s3y4y9).
qpair(0,1,s2y4y7).
qpair(0,1,s7y9).
qpair(0,1,s2y3y8).
qpair(0,1,s3y8y9).
qpair(0,1,s2y3y7y9).
qpair(0,1,s4y8y9).
qpair(0,1,s4y7).
qpair(0,1,s9).
qpair(0,1,s6).
qpair(0,1,s3y8).
qpair(0,1,s2y3y7y8y9).
qpair(0,1,s2y7y8).
qpair(0,1,s2y3).
qpair(0,1,s2y9).
qpair(0,1,s3y4y8).
qpair(0,1,s3y7y8y9).
qpair(0,1,s2y3y4).
qpair(0,1,s2y4y9).
qpair(0,1,s2y3y7).
qpair(0,1,empty).
qpair(0,1,s7y8).
qpair(0,2,s3y4).
qpair(0,2,s5).
qpair(0,2,s4y9).
qpair(0,2,s3y7).
qpair(0,2,s8).
qpair(0,2,s3y7y9).
qpair(0,2,s8y9).
qpair(0,2,s4y5y9).
qpair(0,2,s4y7y9).
qpair(0,2,s1y4y5).
qpair(0,2,s1y3).
qpair(0,2,s1y9).
qpair(0,2,s1y3y9).
qpair(0,2,s3y4y7).
qpair(0,2,s1y3y7y8).
qpair(0,2,s1y7y8y9).
qpair(0,2,s4).
qpair(0,2,s1).
qpair(0,2,s7).
qpair(0,2,s4y5).
qpair(0,2,s3y9).
qpair(0,2,s1y5y9).
qpair(0,2,s4y8).
qpair(0,2,s5y9).
qpair(0,2,s1y7y9).
qpair(0,2,s7y8y9).
qpair(0,2,s1y3y8y9).
qpair(0,2,s1y4y7).
qpair(0,2,s1y5).
qpair(0,2,s4y7y8).
qpair(0,2,s3y4y9).
qpair(0,2,s1y8).
qpair(0,2,s7y9).
qpair(0,2,s3y8y9).
qpair(0,2,s4y8y9).
qpair(0,2,s3).
qpair(0,2,s9).
qpair(0,2,s4y7).
qpair(0,2,s1y7y8).
qpair(0,2,s3y8).
qpair(0,2,s1y4y5y9).
qpair(0,2,s1y4).
qpair(0,2,s1y3y4).
qpair(0,2,s3y4y8).
qpair(0,2,s1y4y9).
qpair(0,2,s1y7).
qpair(0,2,s1y3y7).
qpair(0,2,s3y7y8y9).
qpair(0,2,empty).
qpair(0,2,s1y8y9).
qpair(0,2,s7y8).
qpair(0,3,s2).
qpair(0,3,s2y8y9).
qpair(0,3,s2y7y8y9).
qpair(0,3,s5).
qpair(0,3,s4y9).
qpair(0,3,s1y2y7y8).
qpair(0,3,s1y2y7y8y9).
qpair(0,3,s8).
qpair(0,3,s8y9).
qpair(0,3,s4y7y9).
qpair(0,3,s1y2y8).
qpair(0,3,s1y9).
qpair(0,3,s1y2y8y9).
qpair(0,3,s1y4y8).
qpair(0,3,s1y7y8y9).
qpair(0,3,s2y4y8).
qpair(0,3,s4).
qpair(0,3,s1).
qpair(0,3,s7).
qpair(0,3,s4y8).
qpair(0,3,s5y9).
qpair(0,3,s1y7y9).
qpair(0,3,s7y8y9).
qpair(0,3,s1y2y4).
qpair(0,3,s1y2y7).
qpair(0,3,s2y4).
qpair(0,3,s1y2).
qpair(0,3,s2y7).
qpair(0,3,s1y4y7).
qpair(0,3,s4y7y8).
qpair(0,3,s1y8).
qpair(0,3,s2y4y7).
qpair(0,3,s7y9).
qpair(0,3,s4y8y9).
qpair(0,3,s1y2y7y9).
qpair(0,3,s4y7).
qpair(0,3,s1y7y8).
qpair(0,3,s1y2y9).
qpair(0,3,s1y4).
qpair(0,3,s2y7y8).
qpair(0,3,s2y9).
qpair(0,3,s1y7).
qpair(0,3,s1y4y9).
qpair(0,3,s2y4y9).
qpair(0,3,empty).
qpair(0,3,s1y8y9).
qpair(0,3,s7y8).
qpair(0,4,s5).
qpair(0,5,s4).
qpair(0,7,s2).
qpair(0,7,s2y8y9).
qpair(0,7,s3y4).
qpair(0,7,s5).
qpair(0,7,s4y9).
qpair(0,7,s8).
qpair(0,7,s8y9).
qpair(0,7,s2y3y8y9).
qpair(0,7,s1y2y8).
qpair(0,7,s1y3).
qpair(0,7,s1y9).
qpair(0,7,s1y3y9).
qpair(0,7,s2y8).
qpair(0,7,s1y4y8).
qpair(0,7,s1y2y8y9).
qpair(0,7,s1y2y3y8y9).
qpair(0,7,s2y3y9).
qpair(0,7,s2y4y8).
qpair(0,7,s4).
qpair(0,7,s1).
qpair(0,7,s3y9).
qpair(0,7,s4y8).
qpair(0,7,s1y3y8y9).
qpair(0,7,s1y2y4).
qpair(0,7,s1y2y3y9).
qpair(0,7,s2y4).
qpair(0,7,s1y2).
qpair(0,7,s1y3y8).
qpair(0,7,s3y4y9).
qpair(0,7,s1y8).
qpair(0,7,s2y3y8).
qpair(0,7,s3y8y9).
qpair(0,7,s4y8y9).
qpair(0,7,s3).
qpair(0,7,s9).
qpair(0,7,s3y8).
qpair(0,7,s1y2y3).
qpair(0,7,s1y2y9).
qpair(0,7,s1y2y3y8).
qpair(0,7,s1y4).
qpair(0,7,s1y3y4).
qpair(0,7,s2y3).
qpair(0,7,s2y9).
qpair(0,7,s1y4y9).
qpair(0,7,s3y4y8).
qpair(0,7,s2y3y4).
qpair(0,7,s2y4y9).
qpair(0,7,empty).
qpair(0,7,s1y8y9).
qpair(0,8,s2).
qpair(0,8,s3y4).
qpair(0,8,s5).
qpair(0,8,s4y9).
qpair(0,8,s3y7).
qpair(0,8,s4y6).
qpair(0,8,s5y7).
qpair(0,8,s3y7y9).
qpair(0,8,s4y7y9).
qpair(0,8,s4y6y7).
qpair(0,8,s1y3).
qpair(0,8,s1y9).
qpair(0,8,s1y3y9).
qpair(0,8,s3y4y7).
qpair(0,8,s2y3y9).
qpair(0,8,s4).
qpair(0,8,s1).
qpair(0,8,s7).
qpair(0,8,s3y9).
qpair(0,8,s5y6).
qpair(0,8,s1y2y4).
qpair(0,8,s1y2y3y9).
qpair(0,8,s2y7y9).
qpair(0,8,s1y2y7).
qpair(0,8,s2y4).
qpair(0,8,s1y2).
qpair(0,8,s2y7).
qpair(0,8,s1y4y7).
qpair(0,8,s3y4y9).
qpair(0,8,s2y4y7).
qpair(0,8,s7y9).
qpair(0,8,s6y7).
qpair(0,8,s2y3y7y9).
qpair(0,8,s1y2y7y9).
qpair(0,8,s3).
qpair(0,8,s4y7).
qpair(0,8,s6).
qpair(0,8,s1y2y3).
qpair(0,8,s1y2y9).
qpair(0,8,s1y4).
qpair(0,8,s1y3y4).
qpair(0,8,s2y3).
qpair(0,8,s2y9).
qpair(0,8,s1y7).
qpair(0,8,s1y3y7).
qpair(0,8,s1y4y9).
qpair(0,8,s2y3y4).
qpair(0,8,s1y2y3y7y9).
qpair(0,8,s2y4y9).
qpair(0,8,s2y3y7).
qpair(0,8,s1y3y7y9).
qpair(0,8,empty).
qpair(0,9,s2).
qpair(0,9,s2y3y7y8).
qpair(0,9,s3y4).
qpair(0,9,s5).
qpair(0,9,s3y5y6).
qpair(0,9,s3y7).
qpair(0,9,s1y2y7y8).
qpair(0,9,s8).
qpair(0,9,s1y2y3y7).
qpair(0,9,s1y2y5).
qpair(0,9,s1y6).
qpair(0,9,s1y3y6).
qpair(0,9,s2y5).
qpair(0,9,s1y3).
qpair(0,9,s1y2y3y7y8).
qpair(0,9,s3y4y7).
qpair(0,9,s2y8).
qpair(0,9,s1y4y8).
qpair(0,9,s1y3y7y8).
qpair(0,9,s2y4y8).
qpair(0,9,s2y3y6).
qpair(0,9,s4).
qpair(0,9,s1y5y6).
qpair(0,9,s1).
qpair(0,9,s7).
qpair(0,9,s5y6).
qpair(0,9,s4y8).
qpair(0,9,s3y6).
qpair(0,9,s1y2y3y6).
qpair(0,9,s1y2y4).
qpair(0,9,s2y5y6).
qpair(0,9,s1y2y7).
qpair(0,9,s2y4).
qpair(0,9,s1y2).
qpair(0,9,s1y3y8).
qpair(0,9,s2y7).
qpair(0,9,s1y5).
qpair(0,9,s1y3y5).
qpair(0,9,s1y4y7).
qpair(0,9,s4y7y8).
qpair(0,9,s1y8).
qpair(0,9,s2y4y7).
qpair(0,9,s2y3y5).
qpair(0,9,s2y3y8).
qpair(0,9,s2y3y5y6).
qpair(0,9,s1y2y3y5y6).
qpair(0,9,s3).
qpair(0,9,s4y7).
qpair(0,9,s3y5).
qpair(0,9,s1y3y5y6).
qpair(0,9,s6).
qpair(0,9,s1y7y8).
qpair(0,9,s3y8).
qpair(0,9,s1y2y6).
qpair(0,9,s1y2y3y5).
qpair(0,9,s1y2y3y8).
qpair(0,9,s1y4).
qpair(0,9,s1y3y4).
qpair(0,9,s2y3).
qpair(0,9,s2y7y8).
qpair(0,9,s1y7).
qpair(0,9,s1y3y7).
qpair(0,9,s2y6).
qpair(0,9,s2y3y4).
qpair(0,9,s3y4y8).
qpair(0,9,s2y3y7).
qpair(0,9,empty).
qpair(0,9,s7y8).
qpair(1,2,s6).
qpair(1,2,s0y4).
qpair(1,2,s5).
qpair(1,2,s4y9).
qpair(1,2,s8).
qpair(1,2,s0y9).
qpair(1,2,s0y4y5).
qpair(1,2,s4).
qpair(1,2,s0y5y9).
qpair(1,2,s7).
qpair(1,2,s4y5).
qpair(1,2,s4y5y9).
qpair(1,2,s0y4y5y9).
qpair(1,2,s0y5).
qpair(1,2,s0).
qpair(1,2,s0y4y9).
qpair(1,2,empty).
qpair(1,2,s9).
qpair(1,4,s6).
qpair(1,4,s5).
qpair(1,4,s0y9).
qpair(1,4,s0y2).
qpair(1,4,s7y8).
qpair(1,5,s7).
qpair(1,5,s8).
qpair(1,6,s5y8y9).
qpair(1,6,s5y8).
qpair(1,6,s0y9).
qpair(1,6,s2y9).
qpair(1,6,s0y2).
qpair(1,6,s8y9).
qpair(1,6,s0y2y9).
qpair(1,7,s6).
qpair(1,7,s0y4y8).
qpair(1,7,s5).
qpair(1,7,s4y8).
qpair(1,7,s3).
qpair(1,8,s5).
qpair(1,8,s0y4y7).
qpair(1,8,s0y4y6y7).
qpair(1,8,s4y6).
qpair(1,8,s0y6y7).
qpair(1,8,s2y9).
qpair(1,8,s6y7).
qpair(1,8,s4y6y7).
qpair(1,8,s3).
qpair(1,8,s4y7).
qpair(1,9,s2).
qpair(1,9,s0y2y3y5).
qpair(1,9,s5).
qpair(1,9,s3y5y6).
qpair(1,9,s8).
qpair(1,9,s0y2).
qpair(1,9,s0y3y6).
qpair(1,9,s0y5).
qpair(1,9,s2y5).
qpair(1,9,s0y2y3y5y6).
qpair(1,9,s2y3y6).
qpair(1,9,s0y5y6).
qpair(1,9,s4).
qpair(1,9,s7).
qpair(1,9,s3y6).
qpair(1,9,s0y3y5).
qpair(1,9,s2y5y6).
qpair(1,9,s0y3y5y6).
qpair(1,9,s0y2y3).
qpair(1,9,s2y3y5).
qpair(1,9,s0y2y6).
qpair(1,9,s2y3y5y6).
qpair(1,9,s0).
qpair(1,9,s3).
qpair(1,9,s0y2y3y6).
qpair(1,9,s3y5).
qpair(1,9,s6).
qpair(1,9,s0y3).
qpair(1,9,s0y6).
qpair(1,9,s2y3).
qpair(1,9,s0y2y5y6).
qpair(1,9,s2y6).
qpair(1,9,s0y2y5).
qpair(1,9,empty).
qpair(2,4,s5).
qpair(2,4,s8).
qpair(2,4,s0y1y9).
qpair(2,4,s0y5).
qpair(2,4,s0y8).
qpair(2,4,s1y9).
qpair(2,4,s0y5y9).
qpair(2,4,s1).
qpair(2,4,s7).
qpair(2,4,s1y5y9).
qpair(2,4,s0y1y5).
qpair(2,4,s0y1y5y9).
qpair(2,4,s5y9).
qpair(2,4,s0y7).
qpair(2,4,s1y5).
qpair(2,4,s0).
qpair(2,4,s0y7y8).
qpair(2,4,s9).
qpair(2,4,s0y9).
qpair(2,4,empty).
qpair(2,4,s7y8).
qpair(2,5,s4y9).
qpair(2,5,s8).
qpair(2,5,s0y1y9).
qpair(2,5,s8y9).
qpair(2,5,s1y9).
qpair(2,5,s6y8).
qpair(2,5,s4).
qpair(2,5,s1).
qpair(2,5,s3y9).
qpair(2,5,s3y6).
qpair(2,5,s0y1).
qpair(2,5,s0y4).
qpair(2,5,s3y6y9).
qpair(2,5,s0).
qpair(2,5,s0y1y4y9).
qpair(2,5,s3).
qpair(2,5,s9).
qpair(2,5,s6).
qpair(2,5,s0y1y4).
qpair(2,5,s0y9).
qpair(2,5,s1y4).
qpair(2,5,s1y4y9).
qpair(2,5,empty).
qpair(2,5,s6y9).
qpair(2,7,s3).
qpair(2,8,s3).
qpair(2,9,s0y1y4y5).
qpair(2,9,s5).
qpair(2,9,s3y5y6).
qpair(2,9,s0y1y3).
qpair(2,9,s8).
qpair(2,9,s0y1y6).
qpair(2,9,s0y3y6).
qpair(2,9,s0y5).
qpair(2,9,s1y6).
qpair(2,9,s1y3y6).
qpair(2,9,s1y4y5).
qpair(2,9,s1y3).
qpair(2,9,s0y5y6).
qpair(2,9,s0y1y3y6).
qpair(2,9,s4).
qpair(2,9,s1y5y6).
qpair(2,9,s1).
qpair(2,9,s7).
qpair(2,9,s4y5).
qpair(2,9,s0y1y5y6).
qpair(2,9,s5y6).
qpair(2,9,s0y1y5).
qpair(2,9,s3y6).
qpair(2,9,s0y1y3y5y6).
qpair(2,9,s0y3y5).
qpair(2,9,s0y1).
qpair(2,9,s0y4).
qpair(2,9,s0y3y5y6).
qpair(2,9,s1y5).
qpair(2,9,s1y3y5).
qpair(2,9,s0y1y3y5).
qpair(2,9,s0).
qpair(2,9,s3).
qpair(2,9,s3y5).
qpair(2,9,s6).
qpair(2,9,s0y1y4).
qpair(2,9,s0y3).
qpair(2,9,s1y4).
qpair(2,9,s0y6).
qpair(2,9,empty).
qpair(3,5,s6).
qpair(3,5,s2).
qpair(3,5,s6y8).
qpair(3,5,s8).
qpair(3,5,s2y6y9).
qpair(3,5,s4).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s2y9).
qpair(3,5,s2y6).
qpair(3,5,s8y9).
qpair(3,5,s0).
qpair(3,5,empty).
qpair(3,5,s6y9).
qpair(3,5,s2y8).
qpair(3,7,s0y4y8).
qpair(3,7,s2).
qpair(3,7,s0y4).
qpair(3,7,s1).
qpair(3,7,s4y8).
qpair(3,7,s5y9).
qpair(3,7,s0y8).
qpair(3,8,s0y7).
qpair(3,8,s2).
qpair(3,8,s0y4y7).
qpair(3,8,s4y6).
qpair(3,8,s0y9).
qpair(3,8,s0y6y7).
qpair(3,8,s4y6y7).
qpair(3,8,s1).
qpair(3,8,s6y7).
qpair(3,8,s5y9).
qpair(3,8,s4y7).
qpair(3,9,s2).
qpair(3,9,s5).
qpair(3,9,s8).
qpair(3,9,s0y1y6).
qpair(3,9,s0y5).
qpair(3,9,s1y2y5).
qpair(3,9,s1y6).
qpair(3,9,s2y5).
qpair(3,9,s0y5y6).
qpair(3,9,s0y1y2y5y6).
qpair(3,9,s4).
qpair(3,9,s1y5y6).
qpair(3,9,s1).
qpair(3,9,s7).
qpair(3,9,s0y1y5y6).
qpair(3,9,s5y6).
qpair(3,9,s2y5y6).
qpair(3,9,s0y1).
qpair(3,9,s1y2).
qpair(3,9,s1y5).
qpair(3,9,s0y2y6).
qpair(3,9,s0y1y2y6).
qpair(3,9,s0).
qpair(3,9,s1y2y5y6).
qpair(3,9,s6).
qpair(3,9,s1y2y6).
qpair(3,9,s0y6).
qpair(3,9,s0y2y5y6).
qpair(3,9,s2y6).
qpair(3,9,s0y2y5).
qpair(3,9,s0y1y2y5).
qpair(4,5,s2y7y8).
qpair(4,5,s3y6y9).
qpair(4,5,s3y9).
qpair(4,5,s3y6).
qpair(4,5,s0).
qpair(4,5,s6y9).
qpair(4,5,s2y8).
qpair(4,6,s5y8y9).
qpair(4,6,s5y8).
qpair(4,6,s5).
qpair(4,6,s7).
qpair(4,6,s5y9).
qpair(4,7,s2).
qpair(4,7,s5).
qpair(4,7,s8).
qpair(4,7,s1).
qpair(4,7,s0y2).
qpair(4,7,s0).
qpair(4,7,s0y8).
qpair(4,7,s0y2y8).
qpair(4,7,s3).
qpair(4,7,s9).
qpair(4,7,s2y8).
qpair(4,8,s6).
qpair(4,8,s0y7).
qpair(4,8,s0y2y7).
qpair(4,8,s2y7).
qpair(4,8,s5).
qpair(4,8,s0y6y7).
qpair(4,8,s5y7).
qpair(4,8,s0y6).
qpair(4,8,s1).
qpair(4,8,s7).
qpair(4,8,s0y2).
qpair(4,8,s5y6).
qpair(4,8,s0y5).
qpair(4,8,s0).
qpair(4,8,empty).
qpair(4,8,s3).
qpair(4,8,s9).
qpair(4,9,s1y2).
qpair(4,9,s1y5).
qpair(4,9,s1y3y5).
qpair(4,9,s1y5y6).
qpair(4,9,s5y6).
qpair(4,9,s1y2y3y5y6).
qpair(4,9,s1y6).
qpair(4,9,s2y5).
qpair(4,9,s1y3).
qpair(4,9,s3y5).
qpair(4,9,s1y3y5y6).
qpair(5,6,s3y8).
qpair(5,6,s2).
qpair(5,6,s2y3y9).
qpair(5,6,s8).
qpair(5,6,s4).
qpair(5,6,s1).
qpair(5,6,s8y9).
qpair(5,6,s3y9).
qpair(5,6,empty).
qpair(5,6,s3).
qpair(5,6,s9).
qpair(5,6,s2y8).
qpair(5,8,s2y3y6y9).
qpair(5,8,s2y3y9).
qpair(5,8,s0y4y7).
qpair(5,8,s0y4y6y7).
qpair(5,8,s2y6y9).
qpair(5,8,s3y6y9).
qpair(5,8,s1).
qpair(5,8,s4y6y7).
qpair(5,9,s2).
qpair(5,9,s0y1y3).
qpair(5,9,s8).
qpair(5,9,s0y1y6).
qpair(5,9,s0y2).
qpair(5,9,s0y3y6).
qpair(5,9,s1y6).
qpair(5,9,s1y3y6).
qpair(5,9,s1y3).
qpair(5,9,s2y8).
qpair(5,9,s0y1y2y3y6).
qpair(5,9,s2y3y6).
qpair(5,9,s6y8).
qpair(5,9,s0y1y3y6).
qpair(5,9,s4).
qpair(5,9,s1).
qpair(5,9,s0y1y2).
qpair(5,9,s1y2y3y6).
qpair(5,9,s0y1).
qpair(5,9,s1y2).
qpair(5,9,s0y2y6).
qpair(5,9,s0y1y2y3).
qpair(5,9,s0y1y2y6).
qpair(5,9,s0).
qpair(5,9,s3).
qpair(5,9,s0y2y3y6).
qpair(5,9,s6).
qpair(5,9,s3y8).
qpair(5,9,s1y2y6).
qpair(5,9,s0y3).
qpair(5,9,s1y2y3).
qpair(5,9,s0y6).
qpair(5,9,s2y3).
qpair(5,9,s2y6).
qpair(5,9,empty).
qpair(6,7,s0y4).
qpair(6,7,s4).
qpair(6,8,s0y7).
qpair(6,8,s0y4).
qpair(6,8,s5).
qpair(6,8,s4).
qpair(6,8,s5y7).
qpair(6,8,s4y7).
qpair(6,8,s1).
qpair(6,8,s7).
qpair(6,8,s4y5).
qpair(6,8,s0y5).
qpair(6,8,s0).
qpair(6,8,s5y9).
qpair(6,8,empty).
qpair(6,8,s3).
qpair(6,8,s9).
qpair(6,9,s2).
qpair(6,9,s0y2y3y5).
qpair(6,9,s0y1y3).
qpair(6,9,s8).
qpair(6,9,s0y2).
qpair(6,9,s0y5).
qpair(6,9,s1y2y5).
qpair(6,9,s2y5).
qpair(6,9,s1y3).
qpair(6,9,s4).
qpair(6,9,s1).
qpair(6,9,s0y1y5).
qpair(6,9,s0y1y2).
qpair(6,9,s0y3y5).
qpair(6,9,s0y1).
qpair(6,9,s1y2).
qpair(6,9,s1y3y5).
qpair(6,9,s0y2y3).
qpair(6,9,s2y3y5).
qpair(6,9,s0y1y3y5).
qpair(6,9,s0y1y2y3y5).
qpair(6,9,s0).
qpair(6,9,s3).
qpair(6,9,s3y5).
qpair(6,9,s5y8).
qpair(6,9,s1y2y3y5).
qpair(6,9,s0y3).
qpair(6,9,s1y2y3).
qpair(6,9,s2y3).
qpair(6,9,s0y2y5).
qpair(6,9,empty).
qpair(6,9,s0y1y2y5).
qpair(7,8,s6).
qpair(7,8,s2).
qpair(7,8,s0y4).
qpair(7,8,s5).
qpair(7,8,s4y6).
qpair(7,8,s4).
qpair(7,8,s0y6).
qpair(7,8,s1).
qpair(7,8,s4y5).
qpair(7,8,s5y6).
qpair(7,8,s0y5).
qpair(7,8,s0y4y6).
qpair(7,8,s0).
qpair(7,8,s3).
qpair(7,9,s5).
qpair(7,9,s3y5y6).
qpair(7,9,s0y1y6).
qpair(7,9,s0y1y2y3y5y6).
qpair(7,9,s0y8).
qpair(7,9,s2y5).
qpair(7,9,s2y3y6).
qpair(7,9,s0y1y3y6).
qpair(7,9,s1y5y6).
qpair(7,9,s5y6).
qpair(7,9,s0y3y5y6).
qpair(7,9,s1y5).
qpair(7,9,s1y3y5).
qpair(7,9,s2y3y5).
qpair(7,9,s2y3y5y6).
qpair(7,9,s1y2y3y5y6).
qpair(7,9,s3y5).
qpair(7,9,s1y3y5y6).
qpair(7,9,s1y2y3y5).
qpair(0,1,s3).
qpair(0,1,s2y7).
qpair(0,2,s1y3y8).
qpair(0,2,s1y3y7y8y9).
qpair(0,2,s3y7y8).
qpair(0,2,s1y3y7y9).
qpair(0,2,s1y4y8).
qpair(0,3,s2y7y9).
qpair(0,3,s9).
qpair(0,3,s2y8).
qpair(0,8,s1y7y9).
qpair(0,8,s4y5).
qpair(0,8,s1y2y3y7).
qpair(0,8,s9).
qpair(0,9,s3y7y8).
qpair(0,9,s1y2y5y6).
qpair(0,9,s1y2y8).
qpair(0,9,s1y2y3).
qpair(1,2,s5y9).
qpair(1,4,s2y7).
qpair(1,4,s2y7y8).
qpair(1,4,s2y9).
qpair(1,4,s0y2y9).
qpair(1,4,s2y8).
qpair(1,5,s4).
qpair(1,6,s5y9).
qpair(1,6,s7).
qpair(1,6,s4).
qpair(1,7,s0y4).
qpair(1,7,s0y9).
qpair(1,7,s2y9).
qpair(1,7,s0y2).
qpair(1,7,s0y2y9).
qpair(1,7,s0y8).
qpair(1,8,s0y7).
qpair(1,8,s0y4).
qpair(1,8,s0y9).
qpair(1,8,s0y6).
qpair(1,8,s0y2).
qpair(1,8,s0y2y9).
qpair(1,8,s0y4y6).
qpair(1,9,s5y6).
qpair(2,4,s0y1).
qpair(2,5,s7).
qpair(2,5,s0y4y9).
qpair(2,5,s3y8).
qpair(2,9,s0y4y5).
qpair(2,9,s1y3y5y6).
qpair(3,4,empty).
qpair(3,5,s0y9).
qpair(3,5,s9).
qpair(3,7,s0y5y9).
qpair(3,7,s0y9).
qpair(3,7,s0y5).
qpair(3,8,s0y4).
qpair(3,8,s0y4y6y7).
qpair(3,8,s0y5y9).
qpair(3,8,s0y6).
qpair(3,8,s0y4y6).
qpair(3,8,s0y5).
qpair(3,9,s0y1y2).
qpair(3,9,s0y2).
qpair(3,9,empty).
qpair(3,9,s0y1y5).
qpair(4,5,s2y3y6y9).
qpair(4,5,s2y3y9).
qpair(4,5,s2y7).
qpair(4,5,s2y3y6).
qpair(4,5,s2y6y9).
qpair(4,5,s2y3).
qpair(4,5,s1).
qpair(4,5,s2y9).
qpair(4,5,s2y6).
qpair(4,5,s7y8).
qpair(4,6,s2).
qpair(4,6,s2y7).
qpair(4,6,s8).
qpair(4,6,s2y7y8).
qpair(4,6,s8y9).
qpair(4,6,s9).
qpair(4,6,s7y8).
qpair(4,6,s2y8).
qpair(4,7,empty).
qpair(4,8,s6y7).
qpair(4,8,s2).
qpair(4,9,s0y2y3y5).
qpair(4,9,s3y5y6).
qpair(4,9,s0y1y3).
qpair(4,9,s0y1y6).
qpair(4,9,s0y2).
qpair(4,9,s0y3y6).
qpair(4,9,s0y5).
qpair(4,9,s1y2y5).
qpair(4,9,s0y1y2y3y5y6).
qpair(4,9,s1y3y6).
qpair(4,9,s2y8).
qpair(4,9,s0y2y3y5y6).
qpair(4,9,s0y1y2y3y6).
qpair(4,9,s2y3y6).
qpair(4,9,s0y5y6).
qpair(4,9,s0y1y3y6).
qpair(4,9,s0y1y2y5y6).
qpair(4,9,s0y1y5y6).
qpair(4,9,s0y1y5).
qpair(4,9,s3y6).
qpair(4,9,s0y1y2).
qpair(4,9,s0y1y3y5y6).
qpair(4,9,s0y3y5).
qpair(4,9,s1y2y3y6).
qpair(4,9,s2y5y6).
qpair(4,9,s0y1).
qpair(4,9,s0y3y5y6).
qpair(4,9,s2y7).
qpair(4,9,s0y2y3).
qpair(4,9,s2y3y5).
qpair(4,9,s0y2y6).
qpair(4,9,s0y1y3y5).
qpair(4,9,s0y1y2y3).
qpair(4,9,s2y3y5y6).
qpair(4,9,s0y1y2y3y5).
qpair(4,9,s0y1y2y6).
qpair(4,9,s1y2y5y6).
qpair(4,9,s0y1y2y5).
qpair(4,9,s0y2y3y6).
qpair(4,9,s1y2y6).
qpair(4,9,s1y2y3y5).
qpair(4,9,s0y3).
qpair(4,9,s1y2y3).
qpair(4,9,s2y7y8).
qpair(4,9,s0y6).
qpair(4,9,s2y3).
qpair(4,9,s0y2y5y6).
qpair(4,9,s2y6).
qpair(4,9,s0y2y5).
qpair(4,9,s7y8).
qpair(5,6,s0).
qpair(5,6,s2y9).
qpair(5,6,s2y3).
qpair(5,7,empty).
qpair(5,8,s0y6y7).
qpair(5,8,s0y4y6).
qpair(5,8,s2y3y6).
qpair(5,9,s0y2y3).
qpair(5,9,s3y6).
qpair(6,7,s0y4y8).
qpair(6,7,s5y8y9).
qpair(6,7,s5).
qpair(6,7,s5y8).
qpair(6,7,s8).
qpair(6,7,s8y9).
qpair(6,7,s4y8).
qpair(6,7,s0).
qpair(6,7,s5y9).
qpair(6,7,s0y8).
qpair(6,7,s9).
qpair(6,8,s0y4y7).
qpair(6,8,s2).
qpair(6,9,s1y5).
qpair(6,9,s0y1y2y3).
qpair(6,9,s5).
qpair(7,8,empty).
qpair(7,9,s0y4y8).
qpair(7,9,s2).
qpair(7,9,s0y2y3y5).
qpair(7,9,s0y1y3).
qpair(7,9,s8).
qpair(7,9,s0y2).
qpair(7,9,s0y3y6).
qpair(7,9,s0y5).
qpair(7,9,s1y2y5).
qpair(7,9,s1y6).
qpair(7,9,s1y3y6).
qpair(7,9,s1y3).
qpair(7,9,s0y2y3y5y6).
qpair(7,9,s0y1y2y3y6).
qpair(7,9,s0y5y6).
qpair(7,9,s0y1y2y5y6).
qpair(7,9,s4).
qpair(7,9,s0y1y5y6).
qpair(7,9,s1).
qpair(7,9,s0y1y5).
qpair(7,9,s4y8).
qpair(7,9,s3y6).
qpair(7,9,s0y1y2).
qpair(7,9,s0y1y3y5y6).
qpair(7,9,s0y3y5).
qpair(7,9,s1y2y3y6).
qpair(7,9,s2y5y6).
qpair(7,9,s0y1).
qpair(7,9,s1y2).
qpair(7,9,s0y4).
qpair(7,9,s0y2y3).
qpair(7,9,s0y2y6).
qpair(7,9,s0y1y3y5).
qpair(7,9,s0y1y2y3).
qpair(7,9,s0y1y2y3y5).
qpair(7,9,s0y1y2y6).
qpair(7,9,s0).
qpair(7,9,s1y2y5y6).
qpair(7,9,s3).
qpair(7,9,s0y2y3y6).
qpair(7,9,s6).
qpair(7,9,s1y2y6).
qpair(7,9,s0y3).
qpair(7,9,s1y2y3).
qpair(7,9,s0y6).
qpair(7,9,s2y3).
qpair(7,9,s0y2y5y6).
qpair(7,9,s2y6).
qpair(7,9,s0y2y5).
qpair(7,9,s0y1y2y5).

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
var(8).
var(9).

% ===== Augmented in/2 facts (required by ext_* facts) =====
in(0,s0y8y2).
in(0,s0y8y4).
in(0,s0y8y7).
in(0,s0y9y2).
in(0,s0y9y4).
in(0,s0y9y4y5).
in(0,s0y9y5).
in(1,s1y2y9y7).
in(1,s1y3y9y7).
in(1,s1y9y7).
in(1,s8y1).
in(1,s8y1y2).
in(1,s8y1y2y3).
in(1,s8y1y2y7).
in(1,s8y1y2y9).
in(1,s8y1y3).
in(1,s8y1y3y7).
in(1,s8y1y3y9).
in(1,s8y1y4).
in(1,s8y1y7).
in(1,s8y1y9).
in(1,s8y1y9y7).
in(2,s0y8y2).
in(2,s0y9y2).
in(2,s1y2y9y7).
in(2,s8y1y2).
in(2,s8y1y2y3).
in(2,s8y1y2y7).
in(2,s8y1y2y9).
in(2,s8y2).
in(2,s8y2y3).
in(2,s8y2y3y7).
in(2,s8y2y4).
in(2,s8y2y7).
in(2,s8y9y2).
in(2,s8y9y2y3).
in(2,s8y9y2y7).
in(2,s9y2).
in(2,s9y2y3).
in(2,s9y2y3y6).
in(2,s9y2y3y7).
in(2,s9y2y4).
in(2,s9y2y6).
in(2,s9y2y7).
in(3,s1y3y9y7).
in(3,s8y1y2y3).
in(3,s8y1y3).
in(3,s8y1y3y7).
in(3,s8y1y3y9).
in(3,s8y2y3).
in(3,s8y2y3y7).
in(3,s8y3).
in(3,s8y3y4).
in(3,s8y3y7).
in(3,s8y9y2y3).
in(3,s8y9y3).
in(3,s8y9y3y7).
in(3,s9y2y3).
in(3,s9y2y3y6).
in(3,s9y2y3y7).
in(3,s9y3).
in(3,s9y3y4).
in(3,s9y3y6).
in(3,s9y3y7).
in(4,s0y8y4).
in(4,s0y9y4).
in(4,s0y9y4y5).
in(4,s8y1y4).
in(4,s8y2y4).
in(4,s8y3y4).
in(4,s8y4).
in(4,s8y4y7).
in(4,s8y9y4).
in(4,s9y2y4).
in(4,s9y3y4).
in(4,s9y4).
in(4,s9y4y5).
in(4,s9y4y7).
in(5,s0y9y4y5).
in(5,s0y9y5).
in(5,s8y5).
in(5,s8y9y5).
in(5,s9y4y5).
in(5,s9y5).
in(6,s8y6).
in(6,s9y2y3y6).
in(6,s9y2y6).
in(6,s9y3y6).
in(6,s9y6).
in(7,s0y8y7).
in(7,s1y2y9y7).
in(7,s1y3y9y7).
in(7,s1y9y7).
in(7,s8y1y2y7).
in(7,s8y1y3y7).
in(7,s8y1y7).
in(7,s8y1y9y7).
in(7,s8y2y3y7).
in(7,s8y2y7).
in(7,s8y3y7).
in(7,s8y4y7).
in(7,s8y7).
in(7,s8y9y2y7).
in(7,s8y9y3y7).
in(7,s8y9y7).
in(7,s9y2y3y7).
in(7,s9y2y7).
in(7,s9y3y7).
in(7,s9y4y7).
in(7,s9y7).
in(8,s0y8y2).
in(8,s0y8y4).
in(8,s0y8y7).
in(8,s8y1).
in(8,s8y1y2).
in(8,s8y1y2y3).
in(8,s8y1y2y7).
in(8,s8y1y2y9).
in(8,s8y1y3).
in(8,s8y1y3y7).
in(8,s8y1y3y9).
in(8,s8y1y4).
in(8,s8y1y7).
in(8,s8y1y9).
in(8,s8y1y9y7).
in(8,s8y2).
in(8,s8y2y3).
in(8,s8y2y3y7).
in(8,s8y2y4).
in(8,s8y2y7).
in(8,s8y3).
in(8,s8y3y4).
in(8,s8y3y7).
in(8,s8y4).
in(8,s8y4y7).
in(8,s8y5).
in(8,s8y6).
in(8,s8y7).
in(8,s8y9y2).
in(8,s8y9y2y3).
in(8,s8y9y2y7).
in(8,s8y9y3).
in(8,s8y9y3y7).
in(8,s8y9y4).
in(8,s8y9y5).
in(8,s8y9y7).
in(9,s0y9y2).
in(9,s0y9y4).
in(9,s0y9y4y5).
in(9,s0y9y5).
in(9,s1y2y9y7).
in(9,s1y3y9y7).
in(9,s1y9y7).
in(9,s8y1y2y9).
in(9,s8y1y3y9).
in(9,s8y1y9).
in(9,s8y1y9y7).
in(9,s8y9y2).
in(9,s8y9y2y3).
in(9,s8y9y2y7).
in(9,s8y9y3).
in(9,s8y9y3y7).
in(9,s8y9y4).
in(9,s8y9y5).
in(9,s8y9y7).
in(9,s9y2).
in(9,s9y2y3).
in(9,s9y2y3y6).
in(9,s9y2y3y7).
in(9,s9y2y4).
in(9,s9y2y6).
in(9,s9y2y7).
in(9,s9y3).
in(9,s9y3y4).
in(9,s9y3y6).
in(9,s9y3y7).
in(9,s9y4).
in(9,s9y4y5).
in(9,s9y4y7).
in(9,s9y5).
in(9,s9y6).
in(9,s9y7).


% ===== Base program loaded from: results/wc_sweep_10_2026_20260212_123442/rep1/lps/abapc_inc_full_base_rep1_optN.lp =====
% ===== Base program sha256 (raw): 668b092735a05ce9b11a2c7fb7100a7d6c26cabe822b939bcf27be917ddc7876 =====
% ===== Base program sha256 (normalized): 5d0ffcb22526891b605f38a4dd37ae263a2eab42b9ddef3479250f2196311093 =====
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
var(8).
var(9).
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
{mus(406)}.
{mus(407)}.
{mus(408)}.
{mus(409)}.
{mus(410)}.
{mus(411)}.
{mus(412)}.
{mus(413)}.
{mus(414)}.
{mus(415)}.
{mus(416)}.
{mus(417)}.
{mus(418)}.
{mus(419)}.
{mus(420)}.
{mus(421)}.
{mus(422)}.
{mus(423)}.
{mus(424)}.
{mus(425)}.
{mus(426)}.
{mus(427)}.
{mus(428)}.
{mus(429)}.
{mus(430)}.
{mus(431)}.
{mus(432)}.
{mus(433)}.
{mus(434)}.
{mus(435)}.
{mus(436)}.
{mus(437)}.
{mus(438)}.
{mus(439)}.
{mus(440)}.
{mus(441)}.
{mus(442)}.
{mus(443)}.
{mus(444)}.
{mus(445)}.
{mus(446)}.
{mus(447)}.
{mus(448)}.
{mus(449)}.
{mus(450)}.
{mus(451)}.
{mus(452)}.
{mus(453)}.
{mus(454)}.
{mus(455)}.
{mus(456)}.
{mus(457)}.
{mus(458)}.
{mus(459)}.
{mus(460)}.
{mus(461)}.
{mus(462)}.
{mus(463)}.
{mus(464)}.
{mus(465)}.
{mus(466)}.
{mus(467)}.
{mus(468)}.
{mus(469)}.
{mus(470)}.
{mus(471)}.
{mus(472)}.
{mus(473)}.
{mus(474)}.
{mus(475)}.
{mus(476)}.
{mus(477)}.
{mus(478)}.
{mus(479)}.
{mus(480)}.
{mus(481)}.
{mus(482)}.
{mus(483)}.
{mus(484)}.
{mus(485)}.
{mus(486)}.
{mus(487)}.
{mus(488)}.
{mus(489)}.
{mus(490)}.
{mus(491)}.
{mus(492)}.
{mus(493)}.
{mus(494)}.
{mus(495)}.
{mus(496)}.
{mus(497)}.
{mus(498)}.
{mus(499)}.
{mus(500)}.
{mus(501)}.
{mus(502)}.
{mus(503)}.
{mus(504)}.
{mus(505)}.
{mus(506)}.
{mus(507)}.
{mus(508)}.
{mus(509)}.
{mus(510)}.
{mus(511)}.
{mus(512)}.
{mus(513)}.
{mus(514)}.
{mus(515)}.
{mus(516)}.
{mus(517)}.
{mus(518)}.
{mus(519)}.
{mus(520)}.
{mus(521)}.
{mus(522)}.
{mus(523)}.
{mus(524)}.
{mus(525)}.
{mus(526)}.
{mus(527)}.
{mus(528)}.
{mus(529)}.
{mus(530)}.
{mus(531)}.
{mus(532)}.
{mus(533)}.
{mus(534)}.
{mus(535)}.
{mus(536)}.
{mus(537)}.
{mus(538)}.
{mus(539)}.
{mus(540)}.
{mus(541)}.
{mus(542)}.
{mus(543)}.
{mus(544)}.
{mus(545)}.
{mus(546)}.
{mus(547)}.
{mus(548)}.
{mus(549)}.
{mus(550)}.
{mus(551)}.
{mus(552)}.
{mus(553)}.
{mus(554)}.
{mus(555)}.
{mus(556)}.
{mus(557)}.
{mus(558)}.
{mus(559)}.
{mus(560)}.
{mus(561)}.
{mus(562)}.
{mus(563)}.
{mus(564)}.
{mus(565)}.
{mus(566)}.
{mus(567)}.
{mus(568)}.
{mus(569)}.
{mus(570)}.
{mus(571)}.
{mus(572)}.
{mus(573)}.
{mus(574)}.
{mus(575)}.
{mus(576)}.
{mus(577)}.
{mus(578)}.
{mus(579)}.
{mus(580)}.
{mus(581)}.
{mus(582)}.
{mus(583)}.
{mus(584)}.
{mus(585)}.
{mus(586)}.
{mus(587)}.
{mus(588)}.
{mus(589)}.
{mus(590)}.
{mus(591)}.
{mus(592)}.
{mus(593)}.
{mus(594)}.
{mus(595)}.
{mus(596)}.
{mus(597)}.
{mus(598)}.
{mus(599)}.
{mus(600)}.
{mus(601)}.
{mus(602)}.
{mus(603)}.
{mus(604)}.
{mus(605)}.
{mus(606)}.
{mus(607)}.
{mus(608)}.
{mus(609)}.
{mus(610)}.
{mus(611)}.
{mus(612)}.
{mus(613)}.
{mus(614)}.
{mus(615)}.
{mus(616)}.
{mus(617)}.
{mus(618)}.
{mus(619)}.
{mus(620)}.
{mus(621)}.
{mus(622)}.
{mus(623)}.
{mus(624)}.
{mus(625)}.
{mus(626)}.
{mus(627)}.
{mus(628)}.
{mus(629)}.
{mus(630)}.
{mus(631)}.
{mus(632)}.
{mus(633)}.
{mus(634)}.
{mus(635)}.
{mus(636)}.
{mus(637)}.
{mus(638)}.
{mus(639)}.
{mus(640)}.
{mus(641)}.
{mus(642)}.
{mus(643)}.
{mus(644)}.
{mus(645)}.
{mus(646)}.
{mus(647)}.
{mus(648)}.
{mus(649)}.
{mus(650)}.
{mus(651)}.
{mus(652)}.
{mus(653)}.
{mus(654)}.
{mus(655)}.
{mus(656)}.
{mus(657)}.
{mus(658)}.
{mus(659)}.
{mus(660)}.
{mus(661)}.
{mus(662)}.
{mus(663)}.
{mus(664)}.
{mus(665)}.
{mus(666)}.
{mus(667)}.
{mus(668)}.
{mus(669)}.
{mus(670)}.
{mus(671)}.
{mus(672)}.
{mus(673)}.
{mus(674)}.
{mus(675)}.
{mus(676)}.
{mus(677)}.
{mus(678)}.
{mus(679)}.
{mus(680)}.
{mus(681)}.
{mus(682)}.
{mus(683)}.
{mus(684)}.
{mus(685)}.
{mus(686)}.
{mus(687)}.
{mus(688)}.
{mus(689)}.
{mus(690)}.
{mus(691)}.
{mus(692)}.
{mus(693)}.
{mus(694)}.
{mus(695)}.
{mus(696)}.
{mus(697)}.
{mus(698)}.
{mus(699)}.
{mus(700)}.
{mus(701)}.
{mus(702)}.
{mus(703)}.
{mus(704)}.
{mus(705)}.
{mus(706)}.
{mus(707)}.
{mus(708)}.
{mus(709)}.
{mus(710)}.
{mus(711)}.
{mus(712)}.
{mus(713)}.
{mus(714)}.
{mus(715)}.
{mus(716)}.
{mus(717)}.
{mus(718)}.
{mus(719)}.
{mus(720)}.
{mus(721)}.
{mus(722)}.
{mus(723)}.
{mus(724)}.
{mus(725)}.
{mus(726)}.
{mus(727)}.
{mus(728)}.
{mus(729)}.
{mus(730)}.
{mus(731)}.
{mus(732)}.
{mus(733)}.
{mus(734)}.
{mus(735)}.
{mus(736)}.
{mus(737)}.
{mus(738)}.
{mus(739)}.
{mus(740)}.
{mus(741)}.
{mus(742)}.
{mus(743)}.
{mus(744)}.
{mus(745)}.
{mus(746)}.
{mus(747)}.
{mus(748)}.
{mus(749)}.
{mus(750)}.
{mus(751)}.
{mus(752)}.
{mus(753)}.
{mus(754)}.
{mus(755)}.
{mus(756)}.
{mus(757)}.
{mus(758)}.
{mus(759)}.
{mus(760)}.
{mus(761)}.
{mus(762)}.
{mus(763)}.
{mus(764)}.
{mus(765)}.
{mus(766)}.
{mus(767)}.
{mus(768)}.
{mus(769)}.
{mus(770)}.
{mus(771)}.
{mus(772)}.
{mus(773)}.
{mus(774)}.
{mus(775)}.
{mus(776)}.
{mus(777)}.
{mus(778)}.
{mus(779)}.
{mus(780)}.
{mus(781)}.
{mus(782)}.
{mus(783)}.
{mus(784)}.
{mus(785)}.
{mus(786)}.
{mus(787)}.
{mus(788)}.
{mus(789)}.
{mus(790)}.
{mus(791)}.
{mus(792)}.
{mus(793)}.
{mus(794)}.
{mus(795)}.
{mus(796)}.
{mus(797)}.
{mus(798)}.
{mus(799)}.
{mus(800)}.
{mus(801)}.
{mus(802)}.
{mus(803)}.
{mus(804)}.
{mus(805)}.
{mus(806)}.
{mus(807)}.
{mus(808)}.
{mus(809)}.
{mus(810)}.
{mus(811)}.
{mus(812)}.
{mus(813)}.
{mus(814)}.
{mus(815)}.
{mus(816)}.
{mus(817)}.
{mus(818)}.
{mus(819)}.
{mus(820)}.
{mus(821)}.
{mus(822)}.
{mus(823)}.
{mus(824)}.
{mus(825)}.
{mus(826)}.
{mus(827)}.
{mus(828)}.
{mus(829)}.
{mus(830)}.
{mus(831)}.
{mus(832)}.
{mus(833)}.
{mus(834)}.
{mus(835)}.
{mus(836)}.
{mus(837)}.
{mus(838)}.
{mus(839)}.
{mus(840)}.
{mus(841)}.
{mus(842)}.
{mus(843)}.
{mus(844)}.
{mus(845)}.
{mus(846)}.
{mus(847)}.
{mus(848)}.
{mus(849)}.
{mus(850)}.
{mus(851)}.
{mus(852)}.
{mus(853)}.
{mus(854)}.
{mus(855)}.
{mus(856)}.
{mus(857)}.
{mus(858)}.
{mus(859)}.
{mus(860)}.
{mus(861)}.
{mus(862)}.
{mus(863)}.
{mus(864)}.
{mus(865)}.
{mus(866)}.
{mus(867)}.
{mus(868)}.
{mus(869)}.
{mus(870)}.
{mus(871)}.
{mus(872)}.
{mus(873)}.
{mus(874)}.
{mus(875)}.
{mus(876)}.
{mus(877)}.
{mus(878)}.
{mus(879)}.
{mus(880)}.
{mus(881)}.
{mus(882)}.
{mus(883)}.
{mus(884)}.
{mus(885)}.
{mus(886)}.
{mus(887)}.
{mus(888)}.
{mus(889)}.
{mus(890)}.
{mus(891)}.
{mus(892)}.
{mus(893)}.
{mus(894)}.
{mus(895)}.
{mus(896)}.
{mus(897)}.
{mus(898)}.
{mus(899)}.
{mus(900)}.
{mus(901)}.
{mus(902)}.
{mus(903)}.
{mus(904)}.
{mus(905)}.
{mus(906)}.
{mus(907)}.
{mus(908)}.
{mus(909)}.
{mus(910)}.
{mus(911)}.
{mus(912)}.
{mus(913)}.
{mus(914)}.
{mus(915)}.
{mus(916)}.
{mus(917)}.
{mus(918)}.
{mus(919)}.
{mus(920)}.
{mus(921)}.
{mus(922)}.
{mus(923)}.
{mus(924)}.
{mus(925)}.
{mus(926)}.
{mus(927)}.
{mus(928)}.
{mus(929)}.
{mus(930)}.
{mus(931)}.
{mus(932)}.
{mus(933)}.
{mus(934)}.
{mus(935)}.
{mus(936)}.
{mus(937)}.
{mus(938)}.
{mus(939)}.
{mus(940)}.
{mus(941)}.
{mus(942)}.
{mus(943)}.
{mus(944)}.
{mus(945)}.
{mus(946)}.
{mus(947)}.
{mus(948)}.

% ===== Adorned Facts =====
ext_dep(0,1,empty):-mus(1).
ext_dep(0,1,s2):-mus(2).
ext_indep(0,1,s3):-mus(3).
ext_dep(0,1,s4):-mus(4).
ext_dep(0,1,s5):-mus(5).
ext_dep(0,1,s6):-mus(6).
ext_dep(0,1,s7):-mus(7).
ext_dep(0,1,s8):-mus(8).
ext_dep(0,1,s9):-mus(9).
ext_dep(0,1,s2y3):-mus(10).
ext_dep(0,1,s2y4):-mus(11).
ext_indep(0,1,s2y7):-mus(12).
ext_dep(0,1,s8y2):-mus(13).
ext_dep(0,1,s9y2):-mus(14).
ext_dep(0,1,s3y4):-mus(15).
ext_dep(0,1,s3y7):-mus(16).
ext_dep(0,1,s8y3):-mus(17).
ext_dep(0,1,s9y3):-mus(18).
ext_dep(0,1,s4y7):-mus(19).
ext_dep(0,1,s8y4):-mus(20).
ext_dep(0,1,s9y4):-mus(21).
ext_dep(0,1,s8y7):-mus(22).
ext_dep(0,1,s9y7):-mus(23).
ext_dep(0,1,s8y9):-mus(24).
ext_dep(0,1,s2y3y4):-mus(25).
ext_dep(0,1,s2y3y7):-mus(26).
ext_dep(0,1,s8y2y3):-mus(27).
ext_dep(0,1,s9y2y3):-mus(28).
ext_dep(0,1,s2y4y7):-mus(29).
ext_dep(0,1,s8y2y4):-mus(30).
ext_dep(0,1,s9y2y4):-mus(31).
ext_dep(0,1,s8y2y7):-mus(32).
ext_dep(0,1,s9y2y7):-mus(33).
ext_dep(0,1,s8y9y2):-mus(34).
ext_dep(0,1,s3y4y7):-mus(35).
ext_dep(0,1,s8y3y4):-mus(36).
ext_dep(0,1,s9y3y4):-mus(37).
ext_dep(0,1,s8y3y7):-mus(38).
ext_dep(0,1,s9y3y7):-mus(39).
ext_dep(0,1,s8y9y3):-mus(40).
ext_dep(0,1,s8y4y7):-mus(41).
ext_dep(0,1,s9y4y7):-mus(42).
ext_dep(0,1,s8y9y4):-mus(43).
ext_dep(0,1,s8y9y7):-mus(44).
ext_dep(0,1,s8y2y3y7):-mus(45).
ext_dep(0,1,s9y2y3y7):-mus(46).
ext_dep(0,1,s8y9y2y3):-mus(47).
ext_dep(0,1,s8y9y2y7):-mus(48).
ext_dep(0,1,s8y9y3y7):-mus(49).
ext_dep(0,1,s2y3y7y8y9):-mus(50).
ext_dep(0,2,empty):-mus(51).
ext_dep(0,2,s1):-mus(52).
ext_dep(0,2,s3):-mus(53).
ext_dep(0,2,s4):-mus(54).
ext_dep(0,2,s5):-mus(55).
ext_dep(0,2,s7):-mus(56).
ext_dep(0,2,s8):-mus(57).
ext_dep(0,2,s9):-mus(58).
ext_dep(0,2,s1y3):-mus(59).
ext_dep(0,2,s1y4):-mus(60).
ext_dep(0,2,s1y5):-mus(61).
ext_dep(0,2,s1y7):-mus(62).
ext_dep(0,2,s8y1):-mus(63).
ext_dep(0,2,s1y9):-mus(64).
ext_dep(0,2,s3y4):-mus(65).
ext_dep(0,2,s3y7):-mus(66).
ext_dep(0,2,s8y3):-mus(67).
ext_dep(0,2,s9y3):-mus(68).
ext_dep(0,2,s4y5):-mus(69).
ext_dep(0,2,s4y7):-mus(70).
ext_dep(0,2,s8y4):-mus(71).
ext_dep(0,2,s9y4):-mus(72).
ext_dep(0,2,s9y5):-mus(73).
ext_dep(0,2,s8y7):-mus(74).
ext_dep(0,2,s9y7):-mus(75).
ext_dep(0,2,s8y9):-mus(76).
ext_dep(0,2,s1y3y4):-mus(77).
ext_dep(0,2,s1y3y7):-mus(78).
ext_indep(0,2,s8y1y3):-mus(79).
ext_dep(0,2,s1y3y9):-mus(80).
ext_dep(0,2,s1y4y5):-mus(81).
ext_dep(0,2,s1y4y7):-mus(82).
ext_indep(0,2,s8y1y4):-mus(83).
ext_dep(0,2,s1y4y9):-mus(84).
ext_dep(0,2,s1y5y9):-mus(85).
ext_dep(0,2,s8y1y7):-mus(86).
ext_dep(0,2,s1y9y7):-mus(87).
ext_dep(0,2,s8y1y9):-mus(88).
ext_dep(0,2,s3y4y7):-mus(89).
ext_dep(0,2,s8y3y4):-mus(90).
ext_dep(0,2,s9y3y4):-mus(91).
ext_indep(0,2,s8y3y7):-mus(92).
ext_dep(0,2,s9y3y7):-mus(93).
ext_dep(0,2,s8y9y3):-mus(94).
ext_dep(0,2,s9y4y5):-mus(95).
ext_dep(0,2,s8y4y7):-mus(96).
ext_dep(0,2,s9y4y7):-mus(97).
ext_dep(0,2,s8y9y4):-mus(98).
ext_dep(0,2,s8y9y7):-mus(99).
ext_dep(0,2,s8y1y3y7):-mus(100).
ext_indep(0,2,s1y3y9y7):-mus(101).
ext_dep(0,2,s8y1y3y9):-mus(102).
ext_dep(0,2,s1y4y5y9):-mus(103).
ext_dep(0,2,s8y1y9y7):-mus(104).
ext_dep(0,2,s8y9y3y7):-mus(105).
ext_indep(0,2,s1y3y7y8y9):-mus(106).
ext_dep(0,3,empty):-mus(107).
ext_dep(0,3,s1):-mus(108).
ext_dep(0,3,s2):-mus(109).
ext_dep(0,3,s4):-mus(110).
ext_dep(0,3,s5):-mus(111).
ext_dep(0,3,s7):-mus(112).
ext_dep(0,3,s8):-mus(113).
ext_indep(0,3,s9):-mus(114).
ext_dep(0,3,s1y2):-mus(115).
ext_dep(0,3,s1y4):-mus(116).
ext_dep(0,3,s1y7):-mus(117).
ext_dep(0,3,s8y1):-mus(118).
ext_dep(0,3,s1y9):-mus(119).
ext_dep(0,3,s2y4):-mus(120).
ext_dep(0,3,s2y7):-mus(121).
ext_indep(0,3,s8y2):-mus(122).
ext_dep(0,3,s9y2):-mus(123).
ext_dep(0,3,s4y7):-mus(124).
ext_dep(0,3,s8y4):-mus(125).
ext_dep(0,3,s9y4):-mus(126).
ext_dep(0,3,s9y5):-mus(127).
ext_dep(0,3,s8y7):-mus(128).
ext_dep(0,3,s9y7):-mus(129).
ext_dep(0,3,s8y9):-mus(130).
ext_dep(0,3,s1y2y4):-mus(131).
ext_dep(0,3,s1y2y7):-mus(132).
ext_dep(0,3,s8y1y2):-mus(133).
ext_dep(0,3,s1y2y9):-mus(134).
ext_dep(0,3,s1y4y7):-mus(135).
ext_dep(0,3,s8y1y4):-mus(136).
ext_dep(0,3,s1y4y9):-mus(137).
ext_dep(0,3,s8y1y7):-mus(138).
ext_dep(0,3,s1y9y7):-mus(139).
ext_dep(0,3,s8y1y9):-mus(140).
ext_dep(0,3,s2y4y7):-mus(141).
ext_dep(0,3,s8y2y4):-mus(142).
ext_dep(0,3,s9y2y4):-mus(143).
ext_dep(0,3,s8y2y7):-mus(144).
ext_indep(0,3,s9y2y7):-mus(145).
ext_dep(0,3,s8y9y2):-mus(146).
ext_dep(0,3,s8y4y7):-mus(147).
ext_dep(0,3,s9y4y7):-mus(148).
ext_dep(0,3,s8y9y4):-mus(149).
ext_dep(0,3,s8y9y7):-mus(150).
ext_dep(0,3,s8y1y2y7):-mus(151).
ext_dep(0,3,s1y2y9y7):-mus(152).
ext_dep(0,3,s8y1y2y9):-mus(153).
ext_dep(0,3,s8y1y9y7):-mus(154).
ext_dep(0,3,s8y9y2y7):-mus(155).
ext_dep(0,3,s1y2y7y8y9):-mus(156).
ext_dep(0,4,s5):-mus(157).
ext_dep(0,5,s4):-mus(158).
ext_dep(0,7,empty):-mus(159).
ext_dep(0,7,s1):-mus(160).
ext_dep(0,7,s2):-mus(161).
ext_dep(0,7,s3):-mus(162).
ext_dep(0,7,s4):-mus(163).
ext_dep(0,7,s5):-mus(164).
ext_dep(0,7,s8):-mus(165).
ext_dep(0,7,s9):-mus(166).
ext_dep(0,7,s1y2):-mus(167).
ext_dep(0,7,s1y3):-mus(168).
ext_dep(0,7,s1y4):-mus(169).
ext_dep(0,7,s8y1):-mus(170).
ext_dep(0,7,s1y9):-mus(171).
ext_dep(0,7,s2y3):-mus(172).
ext_dep(0,7,s2y4):-mus(173).
ext_dep(0,7,s8y2):-mus(174).
ext_dep(0,7,s9y2):-mus(175).
ext_dep(0,7,s3y4):-mus(176).
ext_dep(0,7,s8y3):-mus(177).
ext_dep(0,7,s9y3):-mus(178).
ext_dep(0,7,s8y4):-mus(179).
ext_dep(0,7,s9y4):-mus(180).
ext_dep(0,7,s8y9):-mus(181).
ext_dep(0,7,s1y2y3):-mus(182).
ext_dep(0,7,s1y2y4):-mus(183).
ext_dep(0,7,s8y1y2):-mus(184).
ext_dep(0,7,s1y2y9):-mus(185).
ext_dep(0,7,s1y3y4):-mus(186).
ext_dep(0,7,s8y1y3):-mus(187).
ext_dep(0,7,s1y3y9):-mus(188).
ext_dep(0,7,s8y1y4):-mus(189).
ext_dep(0,7,s1y4y9):-mus(190).
ext_dep(0,7,s8y1y9):-mus(191).
ext_dep(0,7,s2y3y4):-mus(192).
ext_dep(0,7,s8y2y3):-mus(193).
ext_dep(0,7,s9y2y3):-mus(194).
ext_dep(0,7,s8y2y4):-mus(195).
ext_dep(0,7,s9y2y4):-mus(196).
ext_dep(0,7,s8y9y2):-mus(197).
ext_dep(0,7,s8y3y4):-mus(198).
ext_dep(0,7,s9y3y4):-mus(199).
ext_dep(0,7,s8y9y3):-mus(200).
ext_dep(0,7,s8y9y4):-mus(201).
ext_dep(0,7,s8y1y2y3):-mus(202).
ext_dep(0,7,s1y2y3y9):-mus(203).
ext_dep(0,7,s8y1y2y9):-mus(204).
ext_dep(0,7,s8y1y3y9):-mus(205).
ext_dep(0,7,s8y9y2y3):-mus(206).
ext_dep(0,7,s1y2y3y8y9):-mus(207).
ext_dep(0,8,empty):-mus(208).
ext_dep(0,8,s1):-mus(209).
ext_dep(0,8,s2):-mus(210).
ext_dep(0,8,s3):-mus(211).
ext_dep(0,8,s4):-mus(212).
ext_dep(0,8,s5):-mus(213).
ext_dep(0,8,s6):-mus(214).
ext_dep(0,8,s7):-mus(215).
ext_indep(0,8,s9):-mus(216).
ext_dep(0,8,s1y2):-mus(217).
ext_dep(0,8,s1y3):-mus(218).
ext_dep(0,8,s1y4):-mus(219).
ext_dep(0,8,s1y7):-mus(220).
ext_dep(0,8,s1y9):-mus(221).
ext_dep(0,8,s2y3):-mus(222).
ext_dep(0,8,s2y4):-mus(223).
ext_dep(0,8,s2y7):-mus(224).
ext_dep(0,8,s9y2):-mus(225).
ext_dep(0,8,s3y4):-mus(226).
ext_dep(0,8,s3y7):-mus(227).
ext_dep(0,8,s9y3):-mus(228).
ext_indep(0,8,s4y5):-mus(229).
ext_dep(0,8,s4y6):-mus(230).
ext_dep(0,8,s4y7):-mus(231).
ext_dep(0,8,s9y4):-mus(232).
ext_dep(0,8,s5y6):-mus(233).
ext_dep(0,8,s5y7):-mus(234).
ext_dep(0,8,s6y7):-mus(235).
ext_dep(0,8,s9y7):-mus(236).
ext_dep(0,8,s1y2y3):-mus(237).
ext_dep(0,8,s1y2y4):-mus(238).
ext_dep(0,8,s1y2y7):-mus(239).
ext_dep(0,8,s1y2y9):-mus(240).
ext_dep(0,8,s1y3y4):-mus(241).
ext_dep(0,8,s1y3y7):-mus(242).
ext_dep(0,8,s1y3y9):-mus(243).
ext_dep(0,8,s1y4y7):-mus(244).
ext_dep(0,8,s1y4y9):-mus(245).
ext_indep(0,8,s1y9y7):-mus(246).
ext_dep(0,8,s2y3y4):-mus(247).
ext_dep(0,8,s2y3y7):-mus(248).
ext_dep(0,8,s9y2y3):-mus(249).
ext_dep(0,8,s2y4y7):-mus(250).
ext_dep(0,8,s9y2y4):-mus(251).
ext_dep(0,8,s9y2y7):-mus(252).
ext_dep(0,8,s3y4y7):-mus(253).
ext_dep(0,8,s9y3y4):-mus(254).
ext_dep(0,8,s9y3y7):-mus(255).
ext_dep(0,8,s4y6y7):-mus(256).
ext_dep(0,8,s9y4y7):-mus(257).
ext_indep(0,8,s1y2y3y7):-mus(258).
ext_dep(0,8,s1y2y3y9):-mus(259).
ext_dep(0,8,s1y2y9y7):-mus(260).
ext_dep(0,8,s1y3y9y7):-mus(261).
ext_dep(0,8,s9y2y3y7):-mus(262).
ext_dep(0,8,s1y2y3y7y9):-mus(263).
ext_dep(0,9,empty):-mus(264).
ext_dep(0,9,s1):-mus(265).
ext_dep(0,9,s2):-mus(266).
ext_dep(0,9,s3):-mus(267).
ext_dep(0,9,s4):-mus(268).
ext_dep(0,9,s5):-mus(269).
ext_dep(0,9,s6):-mus(270).
ext_dep(0,9,s7):-mus(271).
ext_dep(0,9,s8):-mus(272).
ext_dep(0,9,s1y2):-mus(273).
ext_dep(0,9,s1y3):-mus(274).
ext_dep(0,9,s1y4):-mus(275).
ext_dep(0,9,s1y5):-mus(276).
ext_dep(0,9,s1y6):-mus(277).
ext_dep(0,9,s1y7):-mus(278).
ext_dep(0,9,s8y1):-mus(279).
ext_dep(0,9,s2y3):-mus(280).
ext_dep(0,9,s2y4):-mus(281).
ext_dep(0,9,s2y5):-mus(282).
ext_dep(0,9,s2y6):-mus(283).
ext_dep(0,9,s2y7):-mus(284).
ext_dep(0,9,s8y2):-mus(285).
ext_dep(0,9,s3y4):-mus(286).
ext_dep(0,9,s3y5):-mus(287).
ext_dep(0,9,s3y6):-mus(288).
ext_dep(0,9,s3y7):-mus(289).
ext_dep(0,9,s8y3):-mus(290).
ext_dep(0,9,s4y7):-mus(291).
ext_dep(0,9,s8y4):-mus(292).
ext_dep(0,9,s5y6):-mus(293).
ext_dep(0,9,s8y7):-mus(294).
ext_indep(0,9,s1y2y3):-mus(295).
ext_dep(0,9,s1y2y4):-mus(296).
ext_dep(0,9,s1y2y5):-mus(297).
ext_dep(0,9,s1y2y6):-mus(298).
ext_dep(0,9,s1y2y7):-mus(299).
ext_indep(0,9,s8y1y2):-mus(300).
ext_dep(0,9,s1y3y4):-mus(301).
ext_dep(0,9,s1y3y5):-mus(302).
ext_dep(0,9,s1y3y6):-mus(303).
ext_dep(0,9,s1y3y7):-mus(304).
ext_dep(0,9,s8y1y3):-mus(305).
ext_dep(0,9,s1y4y7):-mus(306).
ext_dep(0,9,s8y1y4):-mus(307).
ext_dep(0,9,s1y5y6):-mus(308).
ext_dep(0,9,s8y1y7):-mus(309).
ext_dep(0,9,s2y3y4):-mus(310).
ext_dep(0,9,s2y3y5):-mus(311).
ext_dep(0,9,s2y3y6):-mus(312).
ext_dep(0,9,s2y3y7):-mus(313).
ext_dep(0,9,s8y2y3):-mus(314).
ext_dep(0,9,s2y4y7):-mus(315).
ext_dep(0,9,s8y2y4):-mus(316).
ext_dep(0,9,s2y5y6):-mus(317).
ext_dep(0,9,s8y2y7):-mus(318).
ext_dep(0,9,s3y4y7):-mus(319).
ext_dep(0,9,s8y3y4):-mus(320).
ext_dep(0,9,s3y5y6):-mus(321).
ext_indep(0,9,s8y3y7):-mus(322).
ext_dep(0,9,s8y4y7):-mus(323).
ext_dep(0,9,s1y2y3y5):-mus(324).
ext_dep(0,9,s1y2y3y6):-mus(325).
ext_dep(0,9,s1y2y3y7):-mus(326).
ext_dep(0,9,s8y1y2y3):-mus(327).
ext_indep(0,9,s1y2y5y6):-mus(328).
ext_dep(0,9,s8y1y2y7):-mus(329).
ext_dep(0,9,s1y3y5y6):-mus(330).
ext_dep(0,9,s8y1y3y7):-mus(331).
ext_dep(0,9,s2y3y5y6):-mus(332).
ext_dep(0,9,s8y2y3y7):-mus(333).
ext_dep(0,9,s1y2y3y5y6):-mus(334).
ext_dep(0,9,s1y2y3y7y8):-mus(335).
ext_dep(1,2,empty):-mus(336).
ext_dep(1,2,s0):-mus(337).
ext_dep(1,2,s4):-mus(338).
ext_dep(1,2,s5):-mus(339).
ext_dep(1,2,s6):-mus(340).
ext_dep(1,2,s7):-mus(341).
ext_dep(1,2,s8):-mus(342).
ext_dep(1,2,s9):-mus(343).
ext_dep(1,2,s0y4):-mus(344).
ext_dep(1,2,s0y5):-mus(345).
ext_dep(1,2,s0y9):-mus(346).
ext_dep(1,2,s4y5):-mus(347).
ext_dep(1,2,s9y4):-mus(348).
ext_indep(1,2,s9y5):-mus(349).
ext_dep(1,2,s0y4y5):-mus(350).
ext_dep(1,2,s0y9y4):-mus(351).
ext_dep(1,2,s0y9y5):-mus(352).
ext_dep(1,2,s9y4y5):-mus(353).
ext_dep(1,2,s0y9y4y5):-mus(354).
ext_dep(1,4,s5):-mus(355).
ext_dep(1,4,s6):-mus(356).
ext_dep(1,4,s0y2):-mus(357).
ext_dep(1,4,s0y9):-mus(358).
ext_indep(1,4,s2y7):-mus(359).
ext_indep(1,4,s8y2):-mus(360).
ext_indep(1,4,s9y2):-mus(361).
ext_dep(1,4,s8y7):-mus(362).
ext_indep(1,4,s0y9y2):-mus(363).
ext_indep(1,4,s8y2y7):-mus(364).
ext_indep(1,5,s4):-mus(365).
ext_dep(1,5,s7):-mus(366).
ext_dep(1,5,s8):-mus(367).
ext_indep(1,6,s4):-mus(368).
ext_indep(1,6,s7):-mus(369).
ext_dep(1,6,s0y2):-mus(370).
ext_dep(1,6,s0y9):-mus(371).
ext_dep(1,6,s9y2):-mus(372).
ext_dep(1,6,s8y5):-mus(373).
ext_indep(1,6,s9y5):-mus(374).
ext_dep(1,6,s8y9):-mus(375).
ext_dep(1,6,s0y9y2):-mus(376).
ext_dep(1,6,s8y9y5):-mus(377).
ext_dep(1,7,s3):-mus(378).
ext_dep(1,7,s5):-mus(379).
ext_dep(1,7,s6):-mus(380).
ext_indep(1,7,s0y2):-mus(381).
ext_indep(1,7,s0y4):-mus(382).
ext_indep(1,7,s0y8):-mus(383).
ext_indep(1,7,s0y9):-mus(384).
ext_indep(1,7,s9y2):-mus(385).
ext_dep(1,7,s8y4):-mus(386).
ext_indep(1,7,s0y9y2):-mus(387).
ext_dep(1,7,s0y8y4):-mus(388).
ext_dep(1,8,s3):-mus(389).
ext_dep(1,8,s5):-mus(390).
ext_indep(1,8,s0y2):-mus(391).
ext_indep(1,8,s0y4):-mus(392).
ext_indep(1,8,s0y6):-mus(393).
ext_indep(1,8,s0y7):-mus(394).
ext_indep(1,8,s0y9):-mus(395).
ext_dep(1,8,s9y2):-mus(396).
ext_dep(1,8,s4y6):-mus(397).
ext_dep(1,8,s4y7):-mus(398).
ext_dep(1,8,s6y7):-mus(399).
ext_indep(1,8,s0y9y2):-mus(400).
ext_indep(1,8,s0y4y6):-mus(401).
ext_dep(1,8,s0y4y7):-mus(402).
ext_dep(1,8,s0y6y7):-mus(403).
ext_dep(1,8,s4y6y7):-mus(404).
ext_dep(1,8,s0y4y6y7):-mus(405).
ext_dep(1,9,empty):-mus(406).
ext_dep(1,9,s0):-mus(407).
ext_dep(1,9,s2):-mus(408).
ext_dep(1,9,s3):-mus(409).
ext_dep(1,9,s4):-mus(410).
ext_dep(1,9,s5):-mus(411).
ext_dep(1,9,s6):-mus(412).
ext_dep(1,9,s7):-mus(413).
ext_dep(1,9,s8):-mus(414).
ext_dep(1,9,s0y2):-mus(415).
ext_dep(1,9,s0y3):-mus(416).
ext_dep(1,9,s0y5):-mus(417).
ext_dep(1,9,s0y6):-mus(418).
ext_dep(1,9,s2y3):-mus(419).
ext_dep(1,9,s2y5):-mus(420).
ext_dep(1,9,s2y6):-mus(421).
ext_dep(1,9,s3y5):-mus(422).
ext_dep(1,9,s3y6):-mus(423).
ext_indep(1,9,s5y6):-mus(424).
ext_dep(1,9,s0y2y3):-mus(425).
ext_dep(1,9,s0y2y5):-mus(426).
ext_dep(1,9,s0y2y6):-mus(427).
ext_dep(1,9,s0y3y5):-mus(428).
ext_dep(1,9,s0y3y6):-mus(429).
ext_dep(1,9,s0y5y6):-mus(430).
ext_dep(1,9,s2y3y5):-mus(431).
ext_dep(1,9,s2y3y6):-mus(432).
ext_dep(1,9,s2y5y6):-mus(433).
ext_dep(1,9,s3y5y6):-mus(434).
ext_dep(1,9,s0y2y3y5):-mus(435).
ext_dep(1,9,s0y2y3y6):-mus(436).
ext_dep(1,9,s0y2y5y6):-mus(437).
ext_dep(1,9,s0y3y5y6):-mus(438).
ext_dep(1,9,s2y3y5y6):-mus(439).
ext_dep(1,9,s0y2y3y5y6):-mus(440).
ext_dep(2,4,empty):-mus(441).
ext_dep(2,4,s0):-mus(442).
ext_dep(2,4,s1):-mus(443).
ext_dep(2,4,s5):-mus(444).
ext_dep(2,4,s7):-mus(445).
ext_dep(2,4,s8):-mus(446).
ext_dep(2,4,s9):-mus(447).
ext_indep(2,4,s0y1):-mus(448).
ext_dep(2,4,s0y5):-mus(449).
ext_dep(2,4,s0y7):-mus(450).
ext_dep(2,4,s0y8):-mus(451).
ext_dep(2,4,s0y9):-mus(452).
ext_dep(2,4,s1y5):-mus(453).
ext_dep(2,4,s1y9):-mus(454).
ext_dep(2,4,s9y5):-mus(455).
ext_dep(2,4,s8y7):-mus(456).
ext_dep(2,4,s0y1y5):-mus(457).
ext_dep(2,4,s0y1y9):-mus(458).
ext_dep(2,4,s0y9y5):-mus(459).
ext_dep(2,4,s0y8y7):-mus(460).
ext_dep(2,4,s1y5y9):-mus(461).
ext_dep(2,4,s0y1y5y9):-mus(462).
ext_dep(2,5,empty):-mus(463).
ext_dep(2,5,s0):-mus(464).
ext_dep(2,5,s1):-mus(465).
ext_dep(2,5,s3):-mus(466).
ext_dep(2,5,s4):-mus(467).
ext_dep(2,5,s6):-mus(468).
ext_indep(2,5,s7):-mus(469).
ext_dep(2,5,s8):-mus(470).
ext_dep(2,5,s9):-mus(471).
ext_dep(2,5,s0y1):-mus(472).
ext_dep(2,5,s0y4):-mus(473).
ext_dep(2,5,s0y9):-mus(474).
ext_dep(2,5,s1y4):-mus(475).
ext_dep(2,5,s1y9):-mus(476).
ext_dep(2,5,s3y6):-mus(477).
ext_indep(2,5,s8y3):-mus(478).
ext_dep(2,5,s9y3):-mus(479).
ext_dep(2,5,s9y4):-mus(480).
ext_dep(2,5,s8y6):-mus(481).
ext_dep(2,5,s9y6):-mus(482).
ext_dep(2,5,s8y9):-mus(483).
ext_dep(2,5,s0y1y4):-mus(484).
ext_dep(2,5,s0y1y9):-mus(485).
ext_indep(2,5,s0y9y4):-mus(486).
ext_dep(2,5,s1y4y9):-mus(487).
ext_dep(2,5,s9y3y6):-mus(488).
ext_dep(2,5,s0y1y4y9):-mus(489).
ext_dep(2,7,s3):-mus(490).
ext_dep(2,8,s3):-mus(491).
ext_dep(2,9,empty):-mus(492).
ext_dep(2,9,s0):-mus(493).
ext_dep(2,9,s1):-mus(494).
ext_dep(2,9,s3):-mus(495).
ext_dep(2,9,s4):-mus(496).
ext_dep(2,9,s5):-mus(497).
ext_dep(2,9,s6):-mus(498).
ext_dep(2,9,s7):-mus(499).
ext_dep(2,9,s8):-mus(500).
ext_dep(2,9,s0y1):-mus(501).
ext_dep(2,9,s0y3):-mus(502).
ext_dep(2,9,s0y4):-mus(503).
ext_dep(2,9,s0y5):-mus(504).
ext_dep(2,9,s0y6):-mus(505).
ext_dep(2,9,s1y3):-mus(506).
ext_dep(2,9,s1y4):-mus(507).
ext_dep(2,9,s1y5):-mus(508).
ext_dep(2,9,s1y6):-mus(509).
ext_dep(2,9,s3y5):-mus(510).
ext_dep(2,9,s3y6):-mus(511).
ext_dep(2,9,s4y5):-mus(512).
ext_dep(2,9,s5y6):-mus(513).
ext_dep(2,9,s0y1y3):-mus(514).
ext_dep(2,9,s0y1y4):-mus(515).
ext_dep(2,9,s0y1y5):-mus(516).
ext_dep(2,9,s0y1y6):-mus(517).
ext_dep(2,9,s0y3y5):-mus(518).
ext_dep(2,9,s0y3y6):-mus(519).
ext_indep(2,9,s0y4y5):-mus(520).
ext_dep(2,9,s0y5y6):-mus(521).
ext_dep(2,9,s1y3y5):-mus(522).
ext_dep(2,9,s1y3y6):-mus(523).
ext_dep(2,9,s1y4y5):-mus(524).
ext_dep(2,9,s1y5y6):-mus(525).
ext_dep(2,9,s3y5y6):-mus(526).
ext_dep(2,9,s0y1y3y5):-mus(527).
ext_dep(2,9,s0y1y3y6):-mus(528).
ext_dep(2,9,s0y1y4y5):-mus(529).
ext_dep(2,9,s0y1y5y6):-mus(530).
ext_dep(2,9,s0y3y5y6):-mus(531).
ext_indep(2,9,s1y3y5y6):-mus(532).
ext_dep(2,9,s0y1y3y5y6):-mus(533).
ext_indep(3,4,empty):-mus(534).
ext_dep(3,5,empty):-mus(535).
ext_dep(3,5,s0):-mus(536).
ext_dep(3,5,s1):-mus(537).
ext_dep(3,5,s2):-mus(538).
ext_dep(3,5,s4):-mus(539).
ext_dep(3,5,s6):-mus(540).
ext_dep(3,5,s7):-mus(541).
ext_dep(3,5,s8):-mus(542).
ext_indep(3,5,s9):-mus(543).
ext_indep(3,5,s0y9):-mus(544).
ext_dep(3,5,s2y6):-mus(545).
ext_dep(3,5,s8y2):-mus(546).
ext_dep(3,5,s9y2):-mus(547).
ext_dep(3,5,s8y6):-mus(548).
ext_dep(3,5,s9y6):-mus(549).
ext_dep(3,5,s8y9):-mus(550).
ext_dep(3,5,s9y2y6):-mus(551).
ext_dep(3,7,s1):-mus(552).
ext_dep(3,7,s2):-mus(553).
ext_dep(3,7,s0y4):-mus(554).
ext_indep(3,7,s0y5):-mus(555).
ext_dep(3,7,s0y8):-mus(556).
ext_indep(3,7,s0y9):-mus(557).
ext_dep(3,7,s8y4):-mus(558).
ext_dep(3,7,s9y5):-mus(559).
ext_dep(3,7,s0y8y4):-mus(560).
ext_indep(3,7,s0y9y5):-mus(561).
ext_dep(3,8,s1):-mus(562).
ext_dep(3,8,s2):-mus(563).
ext_indep(3,8,s0y4):-mus(564).
ext_indep(3,8,s0y5):-mus(565).
ext_indep(3,8,s0y6):-mus(566).
ext_dep(3,8,s0y7):-mus(567).
ext_dep(3,8,s0y9):-mus(568).
ext_dep(3,8,s4y6):-mus(569).
ext_dep(3,8,s4y7):-mus(570).
ext_dep(3,8,s9y5):-mus(571).
ext_dep(3,8,s6y7):-mus(572).
ext_indep(3,8,s0y4y6):-mus(573).
ext_dep(3,8,s0y4y7):-mus(574).
ext_indep(3,8,s0y9y5):-mus(575).
ext_dep(3,8,s0y6y7):-mus(576).
ext_dep(3,8,s4y6y7):-mus(577).
ext_indep(3,8,s0y4y6y7):-mus(578).
ext_indep(3,9,empty):-mus(579).
ext_dep(3,9,s0):-mus(580).
ext_dep(3,9,s1):-mus(581).
ext_dep(3,9,s2):-mus(582).
ext_dep(3,9,s4):-mus(583).
ext_dep(3,9,s5):-mus(584).
ext_dep(3,9,s6):-mus(585).
ext_dep(3,9,s7):-mus(586).
ext_dep(3,9,s8):-mus(587).
ext_dep(3,9,s0y1):-mus(588).
ext_indep(3,9,s0y2):-mus(589).
ext_dep(3,9,s0y5):-mus(590).
ext_dep(3,9,s0y6):-mus(591).
ext_dep(3,9,s1y2):-mus(592).
ext_dep(3,9,s1y5):-mus(593).
ext_dep(3,9,s1y6):-mus(594).
ext_dep(3,9,s2y5):-mus(595).
ext_dep(3,9,s2y6):-mus(596).
ext_dep(3,9,s5y6):-mus(597).
ext_indep(3,9,s0y1y2):-mus(598).
ext_indep(3,9,s0y1y5):-mus(599).
ext_dep(3,9,s0y1y6):-mus(600).
ext_dep(3,9,s0y2y5):-mus(601).
ext_dep(3,9,s0y2y6):-mus(602).
ext_dep(3,9,s0y5y6):-mus(603).
ext_dep(3,9,s1y2y5):-mus(604).
ext_dep(3,9,s1y2y6):-mus(605).
ext_dep(3,9,s1y5y6):-mus(606).
ext_dep(3,9,s2y5y6):-mus(607).
ext_dep(3,9,s0y1y2y5):-mus(608).
ext_dep(3,9,s0y1y2y6):-mus(609).
ext_dep(3,9,s0y1y5y6):-mus(610).
ext_dep(3,9,s0y2y5y6):-mus(611).
ext_dep(3,9,s1y2y5y6):-mus(612).
ext_dep(3,9,s0y1y2y5y6):-mus(613).
ext_dep(4,5,s0):-mus(614).
ext_indep(4,5,s1):-mus(615).
ext_indep(4,5,s2y3):-mus(616).
ext_indep(4,5,s2y6):-mus(617).
ext_indep(4,5,s2y7):-mus(618).
ext_dep(4,5,s8y2):-mus(619).
ext_indep(4,5,s9y2):-mus(620).
ext_dep(4,5,s3y6):-mus(621).
ext_dep(4,5,s9y3):-mus(622).
ext_dep(4,5,s9y6):-mus(623).
ext_indep(4,5,s8y7):-mus(624).
ext_indep(4,5,s2y3y6):-mus(625).
ext_indep(4,5,s9y2y3):-mus(626).
ext_indep(4,5,s9y2y6):-mus(627).
ext_dep(4,5,s8y2y7):-mus(628).
ext_dep(4,5,s9y3y6):-mus(629).
ext_indep(4,5,s9y2y3y6):-mus(630).
ext_indep(4,6,s2):-mus(631).
ext_dep(4,6,s5):-mus(632).
ext_dep(4,6,s7):-mus(633).
ext_indep(4,6,s8):-mus(634).
ext_indep(4,6,s9):-mus(635).
ext_indep(4,6,s2y7):-mus(636).
ext_indep(4,6,s8y2):-mus(637).
ext_dep(4,6,s8y5):-mus(638).
ext_dep(4,6,s9y5):-mus(639).
ext_indep(4,6,s8y7):-mus(640).
ext_indep(4,6,s8y9):-mus(641).
ext_indep(4,6,s8y2y7):-mus(642).
ext_dep(4,6,s8y9y5):-mus(643).
ext_indep(4,7,empty):-mus(644).
ext_dep(4,7,s0):-mus(645).
ext_dep(4,7,s1):-mus(646).
ext_dep(4,7,s2):-mus(647).
ext_dep(4,7,s3):-mus(648).
ext_dep(4,7,s5):-mus(649).
ext_dep(4,7,s8):-mus(650).
ext_dep(4,7,s9):-mus(651).
ext_dep(4,7,s0y2):-mus(652).
ext_dep(4,7,s0y8):-mus(653).
ext_dep(4,7,s8y2):-mus(654).
ext_dep(4,7,s0y8y2):-mus(655).
ext_dep(4,8,empty):-mus(656).
ext_dep(4,8,s0):-mus(657).
ext_dep(4,8,s1):-mus(658).
ext_indep(4,8,s2):-mus(659).
ext_dep(4,8,s3):-mus(660).
ext_dep(4,8,s5):-mus(661).
ext_dep(4,8,s6):-mus(662).
ext_dep(4,8,s7):-mus(663).
ext_dep(4,8,s9):-mus(664).
ext_dep(4,8,s0y2):-mus(665).
ext_dep(4,8,s0y5):-mus(666).
ext_dep(4,8,s0y6):-mus(667).
ext_dep(4,8,s0y7):-mus(668).
ext_dep(4,8,s2y7):-mus(669).
ext_dep(4,8,s5y6):-mus(670).
ext_dep(4,8,s5y7):-mus(671).
ext_indep(4,8,s6y7):-mus(672).
ext_dep(4,8,s0y2y7):-mus(673).
ext_dep(4,8,s0y6y7):-mus(674).
ext_indep(4,9,s0y1):-mus(675).
ext_indep(4,9,s0y2):-mus(676).
ext_indep(4,9,s0y3):-mus(677).
ext_indep(4,9,s0y5):-mus(678).
ext_indep(4,9,s0y6):-mus(679).
ext_dep(4,9,s1y2):-mus(680).
ext_dep(4,9,s1y3):-mus(681).
ext_dep(4,9,s1y5):-mus(682).
ext_dep(4,9,s1y6):-mus(683).
ext_indep(4,9,s2y3):-mus(684).
ext_dep(4,9,s2y5):-mus(685).
ext_indep(4,9,s2y6):-mus(686).
ext_indep(4,9,s2y7):-mus(687).
ext_indep(4,9,s8y2):-mus(688).
ext_dep(4,9,s3y5):-mus(689).
ext_indep(4,9,s3y6):-mus(690).
ext_dep(4,9,s5y6):-mus(691).
ext_indep(4,9,s8y7):-mus(692).
ext_indep(4,9,s0y1y2):-mus(693).
ext_indep(4,9,s0y1y3):-mus(694).
ext_indep(4,9,s0y1y5):-mus(695).
ext_indep(4,9,s0y1y6):-mus(696).
ext_indep(4,9,s0y2y3):-mus(697).
ext_indep(4,9,s0y2y5):-mus(698).
ext_indep(4,9,s0y2y6):-mus(699).
ext_indep(4,9,s0y3y5):-mus(700).
ext_indep(4,9,s0y3y6):-mus(701).
ext_indep(4,9,s0y5y6):-mus(702).
ext_indep(4,9,s1y2y3):-mus(703).
ext_indep(4,9,s1y2y5):-mus(704).
ext_indep(4,9,s1y2y6):-mus(705).
ext_dep(4,9,s1y3y5):-mus(706).
ext_indep(4,9,s1y3y6):-mus(707).
ext_dep(4,9,s1y5y6):-mus(708).
ext_indep(4,9,s2y3y5):-mus(709).
ext_indep(4,9,s2y3y6):-mus(710).
ext_indep(4,9,s2y5y6):-mus(711).
ext_indep(4,9,s8y2y7):-mus(712).
ext_indep(4,9,s3y5y6):-mus(713).
ext_indep(4,9,s0y1y2y3):-mus(714).
ext_indep(4,9,s0y1y2y5):-mus(715).
ext_indep(4,9,s0y1y2y6):-mus(716).
ext_indep(4,9,s0y1y3y5):-mus(717).
ext_indep(4,9,s0y1y3y6):-mus(718).
ext_indep(4,9,s0y1y5y6):-mus(719).
ext_indep(4,9,s0y2y3y5):-mus(720).
ext_indep(4,9,s0y2y3y6):-mus(721).
ext_indep(4,9,s0y2y5y6):-mus(722).
ext_indep(4,9,s0y3y5y6):-mus(723).
ext_indep(4,9,s1y2y3y5):-mus(724).
ext_indep(4,9,s1y2y3y6):-mus(725).
ext_indep(4,9,s1y2y5y6):-mus(726).
ext_dep(4,9,s1y3y5y6):-mus(727).
ext_indep(4,9,s2y3y5y6):-mus(728).
ext_indep(4,9,s0y1y2y3y5):-mus(729).
ext_indep(4,9,s0y1y2y3y6):-mus(730).
ext_indep(4,9,s0y1y2y5y6):-mus(731).
ext_indep(4,9,s0y1y3y5y6):-mus(732).
ext_indep(4,9,s0y2y3y5y6):-mus(733).
ext_dep(4,9,s1y2y3y5y6):-mus(734).
ext_indep(4,9,s0y1y2y3y5y6):-mus(735).
ext_dep(5,6,empty):-mus(736).
ext_indep(5,6,s0):-mus(737).
ext_dep(5,6,s1):-mus(738).
ext_dep(5,6,s2):-mus(739).
ext_dep(5,6,s3):-mus(740).
ext_dep(5,6,s4):-mus(741).
ext_dep(5,6,s8):-mus(742).
ext_dep(5,6,s9):-mus(743).
ext_indep(5,6,s2y3):-mus(744).
ext_dep(5,6,s8y2):-mus(745).
ext_indep(5,6,s9y2):-mus(746).
ext_dep(5,6,s8y3):-mus(747).
ext_dep(5,6,s9y3):-mus(748).
ext_dep(5,6,s8y9):-mus(749).
ext_dep(5,6,s9y2y3):-mus(750).
ext_indep(5,7,empty):-mus(751).
ext_dep(5,8,s1):-mus(752).
ext_indep(5,8,s0y4y6):-mus(753).
ext_dep(5,8,s0y4y7):-mus(754).
ext_indep(5,8,s0y6y7):-mus(755).
ext_indep(5,8,s2y3y6):-mus(756).
ext_dep(5,8,s9y2y3):-mus(757).
ext_dep(5,8,s9y2y6):-mus(758).
ext_dep(5,8,s9y3y6):-mus(759).
ext_dep(5,8,s4y6y7):-mus(760).
ext_dep(5,8,s0y4y6y7):-mus(761).
ext_dep(5,8,s9y2y3y6):-mus(762).
ext_dep(5,9,empty):-mus(763).
ext_dep(5,9,s0):-mus(764).
ext_dep(5,9,s1):-mus(765).
ext_dep(5,9,s2):-mus(766).
ext_dep(5,9,s3):-mus(767).
ext_dep(5,9,s4):-mus(768).
ext_dep(5,9,s6):-mus(769).
ext_dep(5,9,s8):-mus(770).
ext_dep(5,9,s0y1):-mus(771).
ext_dep(5,9,s0y2):-mus(772).
ext_dep(5,9,s0y3):-mus(773).
ext_dep(5,9,s0y6):-mus(774).
ext_dep(5,9,s1y2):-mus(775).
ext_dep(5,9,s1y3):-mus(776).
ext_dep(5,9,s1y6):-mus(777).
ext_dep(5,9,s2y3):-mus(778).
ext_dep(5,9,s2y6):-mus(779).
ext_dep(5,9,s8y2):-mus(780).
ext_indep(5,9,s3y6):-mus(781).
ext_dep(5,9,s8y3):-mus(782).
ext_dep(5,9,s8y6):-mus(783).
ext_dep(5,9,s0y1y2):-mus(784).
ext_dep(5,9,s0y1y3):-mus(785).
ext_dep(5,9,s0y1y6):-mus(786).
ext_indep(5,9,s0y2y3):-mus(787).
ext_dep(5,9,s0y2y6):-mus(788).
ext_dep(5,9,s0y3y6):-mus(789).
ext_dep(5,9,s1y2y3):-mus(790).
ext_dep(5,9,s1y2y6):-mus(791).
ext_dep(5,9,s1y3y6):-mus(792).
ext_dep(5,9,s2y3y6):-mus(793).
ext_dep(5,9,s0y1y2y3):-mus(794).
ext_dep(5,9,s0y1y2y6):-mus(795).
ext_dep(5,9,s0y1y3y6):-mus(796).
ext_dep(5,9,s0y2y3y6):-mus(797).
ext_dep(5,9,s1y2y3y6):-mus(798).
ext_dep(5,9,s0y1y2y3y6):-mus(799).
ext_indep(6,7,s0):-mus(800).
ext_dep(6,7,s4):-mus(801).
ext_indep(6,7,s5):-mus(802).
ext_indep(6,7,s8):-mus(803).
ext_indep(6,7,s9):-mus(804).
ext_dep(6,7,s0y4):-mus(805).
ext_indep(6,7,s0y8):-mus(806).
ext_indep(6,7,s8y4):-mus(807).
ext_indep(6,7,s8y5):-mus(808).
ext_indep(6,7,s9y5):-mus(809).
ext_indep(6,7,s8y9):-mus(810).
ext_indep(6,7,s0y8y4):-mus(811).
ext_indep(6,7,s8y9y5):-mus(812).
ext_dep(6,8,empty):-mus(813).
ext_dep(6,8,s0):-mus(814).
ext_dep(6,8,s1):-mus(815).
ext_indep(6,8,s2):-mus(816).
ext_dep(6,8,s3):-mus(817).
ext_dep(6,8,s4):-mus(818).
ext_dep(6,8,s5):-mus(819).
ext_dep(6,8,s7):-mus(820).
ext_dep(6,8,s9):-mus(821).
ext_dep(6,8,s0y4):-mus(822).
ext_dep(6,8,s0y5):-mus(823).
ext_dep(6,8,s0y7):-mus(824).
ext_dep(6,8,s4y5):-mus(825).
ext_dep(6,8,s4y7):-mus(826).
ext_dep(6,8,s5y7):-mus(827).
ext_dep(6,8,s9y5):-mus(828).
ext_indep(6,8,s0y4y7):-mus(829).
ext_dep(6,9,empty):-mus(830).
ext_dep(6,9,s0):-mus(831).
ext_dep(6,9,s1):-mus(832).
ext_dep(6,9,s2):-mus(833).
ext_dep(6,9,s3):-mus(834).
ext_dep(6,9,s4):-mus(835).
ext_indep(6,9,s5):-mus(836).
ext_dep(6,9,s8):-mus(837).
ext_dep(6,9,s0y1):-mus(838).
ext_dep(6,9,s0y2):-mus(839).
ext_dep(6,9,s0y3):-mus(840).
ext_dep(6,9,s0y5):-mus(841).
ext_dep(6,9,s1y2):-mus(842).
ext_dep(6,9,s1y3):-mus(843).
ext_indep(6,9,s1y5):-mus(844).
ext_dep(6,9,s2y3):-mus(845).
ext_dep(6,9,s2y5):-mus(846).
ext_dep(6,9,s3y5):-mus(847).
ext_dep(6,9,s8y5):-mus(848).
ext_dep(6,9,s0y1y2):-mus(849).
ext_dep(6,9,s0y1y3):-mus(850).
ext_dep(6,9,s0y1y5):-mus(851).
ext_dep(6,9,s0y2y3):-mus(852).
ext_dep(6,9,s0y2y5):-mus(853).
ext_dep(6,9,s0y3y5):-mus(854).
ext_dep(6,9,s1y2y3):-mus(855).
ext_dep(6,9,s1y2y5):-mus(856).
ext_dep(6,9,s1y3y5):-mus(857).
ext_dep(6,9,s2y3y5):-mus(858).
ext_indep(6,9,s0y1y2y3):-mus(859).
ext_dep(6,9,s0y1y2y5):-mus(860).
ext_dep(6,9,s0y1y3y5):-mus(861).
ext_dep(6,9,s0y2y3y5):-mus(862).
ext_dep(6,9,s1y2y3y5):-mus(863).
ext_dep(6,9,s0y1y2y3y5):-mus(864).
ext_indep(7,8,empty):-mus(865).
ext_dep(7,8,s0):-mus(866).
ext_dep(7,8,s1):-mus(867).
ext_dep(7,8,s2):-mus(868).
ext_dep(7,8,s3):-mus(869).
ext_dep(7,8,s4):-mus(870).
ext_dep(7,8,s5):-mus(871).
ext_dep(7,8,s6):-mus(872).
ext_dep(7,8,s0y4):-mus(873).
ext_dep(7,8,s0y5):-mus(874).
ext_dep(7,8,s0y6):-mus(875).
ext_dep(7,8,s4y5):-mus(876).
ext_dep(7,8,s4y6):-mus(877).
ext_dep(7,8,s5y6):-mus(878).
ext_dep(7,8,s0y4y6):-mus(879).
ext_indep(7,9,s0):-mus(880).
ext_indep(7,9,s1):-mus(881).
ext_indep(7,9,s2):-mus(882).
ext_indep(7,9,s3):-mus(883).
ext_indep(7,9,s4):-mus(884).
ext_dep(7,9,s5):-mus(885).
ext_indep(7,9,s6):-mus(886).
ext_indep(7,9,s8):-mus(887).
ext_indep(7,9,s0y1):-mus(888).
ext_indep(7,9,s0y2):-mus(889).
ext_indep(7,9,s0y3):-mus(890).
ext_indep(7,9,s0y4):-mus(891).
ext_indep(7,9,s0y5):-mus(892).
ext_indep(7,9,s0y6):-mus(893).
ext_dep(7,9,s0y8):-mus(894).
ext_indep(7,9,s1y2):-mus(895).
ext_indep(7,9,s1y3):-mus(896).
ext_dep(7,9,s1y5):-mus(897).
ext_indep(7,9,s1y6):-mus(898).
ext_indep(7,9,s2y3):-mus(899).
ext_dep(7,9,s2y5):-mus(900).
ext_indep(7,9,s2y6):-mus(901).
ext_dep(7,9,s3y5):-mus(902).
ext_indep(7,9,s3y6):-mus(903).
ext_indep(7,9,s8y4):-mus(904).
ext_dep(7,9,s5y6):-mus(905).
ext_indep(7,9,s0y1y2):-mus(906).
ext_indep(7,9,s0y1y3):-mus(907).
ext_indep(7,9,s0y1y5):-mus(908).
ext_dep(7,9,s0y1y6):-mus(909).
ext_indep(7,9,s0y2y3):-mus(910).
ext_indep(7,9,s0y2y5):-mus(911).
ext_indep(7,9,s0y2y6):-mus(912).
ext_indep(7,9,s0y3y5):-mus(913).
ext_indep(7,9,s0y3y6):-mus(914).
ext_indep(7,9,s0y8y4):-mus(915).
ext_indep(7,9,s0y5y6):-mus(916).
ext_indep(7,9,s1y2y3):-mus(917).
ext_indep(7,9,s1y2y5):-mus(918).
ext_indep(7,9,s1y2y6):-mus(919).
ext_dep(7,9,s1y3y5):-mus(920).
ext_indep(7,9,s1y3y6):-mus(921).
ext_dep(7,9,s1y5y6):-mus(922).
ext_dep(7,9,s2y3y5):-mus(923).
ext_dep(7,9,s2y3y6):-mus(924).
ext_indep(7,9,s2y5y6):-mus(925).
ext_dep(7,9,s3y5y6):-mus(926).
ext_indep(7,9,s0y1y2y3):-mus(927).
ext_indep(7,9,s0y1y2y5):-mus(928).
ext_indep(7,9,s0y1y2y6):-mus(929).
ext_indep(7,9,s0y1y3y5):-mus(930).
ext_dep(7,9,s0y1y3y6):-mus(931).
ext_indep(7,9,s0y1y5y6):-mus(932).
ext_indep(7,9,s0y2y3y5):-mus(933).
ext_indep(7,9,s0y2y3y6):-mus(934).
ext_indep(7,9,s0y2y5y6):-mus(935).
ext_dep(7,9,s0y3y5y6):-mus(936).
ext_dep(7,9,s1y2y3y5):-mus(937).
ext_indep(7,9,s1y2y3y6):-mus(938).
ext_indep(7,9,s1y2y5y6):-mus(939).
ext_dep(7,9,s1y3y5y6):-mus(940).
ext_dep(7,9,s2y3y5y6):-mus(941).
ext_indep(7,9,s0y1y2y3y5):-mus(942).
ext_indep(7,9,s0y1y2y3y6):-mus(943).
ext_indep(7,9,s0y1y2y5y6):-mus(944).
ext_indep(7,9,s0y1y3y5y6):-mus(945).
ext_indep(7,9,s0y2y3y5y6):-mus(946).
ext_dep(7,9,s1y2y3y5y6):-mus(947).
ext_dep(7,9,s0y1y2y3y5y6):-mus(948).

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
__optimum_mcs_objective_literal__(9999999, mus(70)).
__optimum_mcs_objective_literal__(9999999, mus(71)).
__optimum_mcs_objective_literal__(9999999, mus(72)).
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
__optimum_mcs_objective_literal__(9999999, mus(136)).
__optimum_mcs_objective_literal__(9999999, mus(137)).
__optimum_mcs_objective_literal__(9999999, mus(138)).
__optimum_mcs_objective_literal__(9999999, mus(139)).
__optimum_mcs_objective_literal__(9999999, mus(140)).
__optimum_mcs_objective_literal__(9999999, mus(141)).
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
__optimum_mcs_objective_literal__(9985714, mus(158)).
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
__optimum_mcs_objective_literal__(9999999, mus(182)).
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
__optimum_mcs_objective_literal__(9999999, mus(220)).
__optimum_mcs_objective_literal__(9999999, mus(221)).
__optimum_mcs_objective_literal__(9999999, mus(222)).
__optimum_mcs_objective_literal__(9999999, mus(223)).
__optimum_mcs_objective_literal__(9999999, mus(224)).
__optimum_mcs_objective_literal__(9999999, mus(225)).
__optimum_mcs_objective_literal__(9999999, mus(226)).
__optimum_mcs_objective_literal__(9999999, mus(227)).
__optimum_mcs_objective_literal__(9999999, mus(228)).
__optimum_mcs_objective_literal__(9999999, mus(229)).
__optimum_mcs_objective_literal__(9999999, mus(230)).
__optimum_mcs_objective_literal__(9999999, mus(231)).
__optimum_mcs_objective_literal__(9999999, mus(232)).
__optimum_mcs_objective_literal__(9999999, mus(233)).
__optimum_mcs_objective_literal__(9999999, mus(234)).
__optimum_mcs_objective_literal__(9999999, mus(235)).
__optimum_mcs_objective_literal__(9999999, mus(236)).
__optimum_mcs_objective_literal__(9999999, mus(237)).
__optimum_mcs_objective_literal__(9999999, mus(238)).
__optimum_mcs_objective_literal__(9999999, mus(239)).
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
__optimum_mcs_objective_literal__(9999999, mus(293)).
__optimum_mcs_objective_literal__(9999999, mus(294)).
__optimum_mcs_objective_literal__(9999999, mus(295)).
__optimum_mcs_objective_literal__(9999999, mus(296)).
__optimum_mcs_objective_literal__(9999999, mus(297)).
__optimum_mcs_objective_literal__(9999999, mus(298)).
__optimum_mcs_objective_literal__(9999999, mus(299)).
__optimum_mcs_objective_literal__(9999999, mus(300)).
__optimum_mcs_objective_literal__(9999999, mus(301)).
__optimum_mcs_objective_literal__(9999999, mus(302)).
__optimum_mcs_objective_literal__(9999999, mus(303)).
__optimum_mcs_objective_literal__(9999999, mus(304)).
__optimum_mcs_objective_literal__(9999999, mus(305)).
__optimum_mcs_objective_literal__(9999999, mus(306)).
__optimum_mcs_objective_literal__(9999999, mus(307)).
__optimum_mcs_objective_literal__(9999999, mus(308)).
__optimum_mcs_objective_literal__(9999999, mus(309)).
__optimum_mcs_objective_literal__(9999999, mus(310)).
__optimum_mcs_objective_literal__(9999999, mus(311)).
__optimum_mcs_objective_literal__(9999999, mus(312)).
__optimum_mcs_objective_literal__(9999999, mus(313)).
__optimum_mcs_objective_literal__(9999999, mus(314)).
__optimum_mcs_objective_literal__(9999999, mus(315)).
__optimum_mcs_objective_literal__(9999999, mus(316)).
__optimum_mcs_objective_literal__(9999999, mus(317)).
__optimum_mcs_objective_literal__(9999999, mus(318)).
__optimum_mcs_objective_literal__(9999999, mus(319)).
__optimum_mcs_objective_literal__(9999999, mus(320)).
__optimum_mcs_objective_literal__(9999999, mus(321)).
__optimum_mcs_objective_literal__(9999999, mus(322)).
__optimum_mcs_objective_literal__(9999994, mus(323)).
__optimum_mcs_objective_literal__(9999999, mus(324)).
__optimum_mcs_objective_literal__(9999999, mus(325)).
__optimum_mcs_objective_literal__(9999999, mus(326)).
__optimum_mcs_objective_literal__(9999999, mus(327)).
__optimum_mcs_objective_literal__(9999999, mus(328)).
__optimum_mcs_objective_literal__(9999999, mus(329)).
__optimum_mcs_objective_literal__(9999999, mus(330)).
__optimum_mcs_objective_literal__(9999999, mus(331)).
__optimum_mcs_objective_literal__(9999999, mus(332)).
__optimum_mcs_objective_literal__(9999999, mus(333)).
__optimum_mcs_objective_literal__(9999999, mus(334)).
__optimum_mcs_objective_literal__(9999999, mus(335)).
__optimum_mcs_objective_literal__(9999999, mus(336)).
__optimum_mcs_objective_literal__(9999999, mus(337)).
__optimum_mcs_objective_literal__(9999999, mus(338)).
__optimum_mcs_objective_literal__(9999999, mus(339)).
__optimum_mcs_objective_literal__(9999999, mus(340)).
__optimum_mcs_objective_literal__(9999999, mus(341)).
__optimum_mcs_objective_literal__(9999999, mus(342)).
__optimum_mcs_objective_literal__(9999999, mus(343)).
__optimum_mcs_objective_literal__(9999999, mus(344)).
__optimum_mcs_objective_literal__(9999999, mus(345)).
__optimum_mcs_objective_literal__(9999999, mus(346)).
__optimum_mcs_objective_literal__(9999999, mus(347)).
__optimum_mcs_objective_literal__(9999999, mus(348)).
__optimum_mcs_objective_literal__(9999999, mus(349)).
__optimum_mcs_objective_literal__(9999999, mus(350)).
__optimum_mcs_objective_literal__(9999999, mus(351)).
__optimum_mcs_objective_literal__(9999999, mus(352)).
__optimum_mcs_objective_literal__(9999999, mus(353)).
__optimum_mcs_objective_literal__(9999999, mus(354)).
__optimum_mcs_objective_literal__(9999996, mus(355)).
__optimum_mcs_objective_literal__(9999997, mus(356)).
__optimum_mcs_objective_literal__(5695844, mus(357)).
__optimum_mcs_objective_literal__(9999963, mus(358)).
__optimum_mcs_objective_literal__(6014414, mus(359)).
__optimum_mcs_objective_literal__(6270058, mus(360)).
__optimum_mcs_objective_literal__(6259781, mus(361)).
__optimum_mcs_objective_literal__(9999999, mus(362)).
__optimum_mcs_objective_literal__(9529074, mus(363)).
__optimum_mcs_objective_literal__(5861644, mus(364)).
__optimum_mcs_objective_literal__(5048782, mus(365)).
__optimum_mcs_objective_literal__(6465073, mus(366)).
__optimum_mcs_objective_literal__(9995101, mus(367)).
__optimum_mcs_objective_literal__(5319033, mus(368)).
__optimum_mcs_objective_literal__(5346581, mus(369)).
__optimum_mcs_objective_literal__(7356067, mus(370)).
__optimum_mcs_objective_literal__(9999999, mus(371)).
__optimum_mcs_objective_literal__(9999999, mus(372)).
__optimum_mcs_objective_literal__(9999999, mus(373)).
__optimum_mcs_objective_literal__(9999999, mus(374)).
__optimum_mcs_objective_literal__(9999999, mus(375)).
__optimum_mcs_objective_literal__(9999999, mus(376)).
__optimum_mcs_objective_literal__(9999999, mus(377)).
__optimum_mcs_objective_literal__(9610137, mus(378)).
__optimum_mcs_objective_literal__(9508400, mus(379)).
__optimum_mcs_objective_literal__(9594993, mus(380)).
__optimum_mcs_objective_literal__(8763456, mus(381)).
__optimum_mcs_objective_literal__(5200028, mus(382)).
__optimum_mcs_objective_literal__(7200846, mus(383)).
__optimum_mcs_objective_literal__(9110758, mus(384)).
__optimum_mcs_objective_literal__(6221903, mus(385)).
__optimum_mcs_objective_literal__(9997744, mus(386)).
__optimum_mcs_objective_literal__(8862451, mus(387)).
__optimum_mcs_objective_literal__(9877687, mus(388)).
__optimum_mcs_objective_literal__(9983389, mus(389)).
__optimum_mcs_objective_literal__(9999298, mus(390)).
__optimum_mcs_objective_literal__(5104755, mus(391)).
__optimum_mcs_objective_literal__(7175964, mus(392)).
__optimum_mcs_objective_literal__(6193123, mus(393)).
__optimum_mcs_objective_literal__(6630208, mus(394)).
__optimum_mcs_objective_literal__(7648815, mus(395)).
__optimum_mcs_objective_literal__(9999644, mus(396)).
__optimum_mcs_objective_literal__(9999982, mus(397)).
__optimum_mcs_objective_literal__(9999999, mus(398)).
__optimum_mcs_objective_literal__(9975291, mus(399)).
__optimum_mcs_objective_literal__(5387512, mus(400)).
__optimum_mcs_objective_literal__(7221785, mus(401)).
__optimum_mcs_objective_literal__(9223514, mus(402)).
__optimum_mcs_objective_literal__(5973191, mus(403)).
__optimum_mcs_objective_literal__(9999999, mus(404)).
__optimum_mcs_objective_literal__(6818750, mus(405)).
__optimum_mcs_objective_literal__(9999999, mus(406)).
__optimum_mcs_objective_literal__(9999999, mus(407)).
__optimum_mcs_objective_literal__(9999999, mus(408)).
__optimum_mcs_objective_literal__(9999999, mus(409)).
__optimum_mcs_objective_literal__(9999999, mus(410)).
__optimum_mcs_objective_literal__(9999999, mus(411)).
__optimum_mcs_objective_literal__(9999999, mus(412)).
__optimum_mcs_objective_literal__(9999999, mus(413)).
__optimum_mcs_objective_literal__(9999999, mus(414)).
__optimum_mcs_objective_literal__(9999999, mus(415)).
__optimum_mcs_objective_literal__(9999999, mus(416)).
__optimum_mcs_objective_literal__(9999999, mus(417)).
__optimum_mcs_objective_literal__(9999999, mus(418)).
__optimum_mcs_objective_literal__(9999999, mus(419)).
__optimum_mcs_objective_literal__(9999999, mus(420)).
__optimum_mcs_objective_literal__(9999999, mus(421)).
__optimum_mcs_objective_literal__(9999999, mus(422)).
__optimum_mcs_objective_literal__(9999999, mus(423)).
__optimum_mcs_objective_literal__(9999999, mus(424)).
__optimum_mcs_objective_literal__(9999999, mus(425)).
__optimum_mcs_objective_literal__(9999999, mus(426)).
__optimum_mcs_objective_literal__(9999999, mus(427)).
__optimum_mcs_objective_literal__(9999999, mus(428)).
__optimum_mcs_objective_literal__(9999999, mus(429)).
__optimum_mcs_objective_literal__(9999999, mus(430)).
__optimum_mcs_objective_literal__(9999999, mus(431)).
__optimum_mcs_objective_literal__(9999999, mus(432)).
__optimum_mcs_objective_literal__(9999999, mus(433)).
__optimum_mcs_objective_literal__(9999999, mus(434)).
__optimum_mcs_objective_literal__(9999999, mus(435)).
__optimum_mcs_objective_literal__(9999999, mus(436)).
__optimum_mcs_objective_literal__(9999999, mus(437)).
__optimum_mcs_objective_literal__(9999999, mus(438)).
__optimum_mcs_objective_literal__(9999999, mus(439)).
__optimum_mcs_objective_literal__(9999999, mus(440)).
__optimum_mcs_objective_literal__(9999999, mus(441)).
__optimum_mcs_objective_literal__(9999999, mus(442)).
__optimum_mcs_objective_literal__(9999999, mus(443)).
__optimum_mcs_objective_literal__(9999999, mus(444)).
__optimum_mcs_objective_literal__(9999999, mus(445)).
__optimum_mcs_objective_literal__(9999999, mus(446)).
__optimum_mcs_objective_literal__(9999999, mus(447)).
__optimum_mcs_objective_literal__(9999999, mus(448)).
__optimum_mcs_objective_literal__(9999999, mus(449)).
__optimum_mcs_objective_literal__(9999999, mus(450)).
__optimum_mcs_objective_literal__(9999999, mus(451)).
__optimum_mcs_objective_literal__(9999999, mus(452)).
__optimum_mcs_objective_literal__(9999999, mus(453)).
__optimum_mcs_objective_literal__(9999999, mus(454)).
__optimum_mcs_objective_literal__(9999999, mus(455)).
__optimum_mcs_objective_literal__(9999999, mus(456)).
__optimum_mcs_objective_literal__(9999999, mus(457)).
__optimum_mcs_objective_literal__(9999999, mus(458)).
__optimum_mcs_objective_literal__(9999999, mus(459)).
__optimum_mcs_objective_literal__(9999999, mus(460)).
__optimum_mcs_objective_literal__(9999999, mus(461)).
__optimum_mcs_objective_literal__(9999999, mus(462)).
__optimum_mcs_objective_literal__(9999999, mus(463)).
__optimum_mcs_objective_literal__(9999999, mus(464)).
__optimum_mcs_objective_literal__(9999999, mus(465)).
__optimum_mcs_objective_literal__(9999999, mus(466)).
__optimum_mcs_objective_literal__(9999999, mus(467)).
__optimum_mcs_objective_literal__(9999999, mus(468)).
__optimum_mcs_objective_literal__(9999999, mus(469)).
__optimum_mcs_objective_literal__(9999999, mus(470)).
__optimum_mcs_objective_literal__(9999999, mus(471)).
__optimum_mcs_objective_literal__(9999999, mus(472)).
__optimum_mcs_objective_literal__(9999999, mus(473)).
__optimum_mcs_objective_literal__(9999999, mus(474)).
__optimum_mcs_objective_literal__(9999999, mus(475)).
__optimum_mcs_objective_literal__(9999999, mus(476)).
__optimum_mcs_objective_literal__(9999999, mus(477)).
__optimum_mcs_objective_literal__(9999999, mus(478)).
__optimum_mcs_objective_literal__(9999999, mus(479)).
__optimum_mcs_objective_literal__(9999999, mus(480)).
__optimum_mcs_objective_literal__(9999999, mus(481)).
__optimum_mcs_objective_literal__(9999999, mus(482)).
__optimum_mcs_objective_literal__(9999999, mus(483)).
__optimum_mcs_objective_literal__(9999999, mus(484)).
__optimum_mcs_objective_literal__(9999999, mus(485)).
__optimum_mcs_objective_literal__(9999999, mus(486)).
__optimum_mcs_objective_literal__(9999999, mus(487)).
__optimum_mcs_objective_literal__(9999999, mus(488)).
__optimum_mcs_objective_literal__(9999999, mus(489)).
__optimum_mcs_objective_literal__(9839723, mus(490)).
__optimum_mcs_objective_literal__(9999999, mus(491)).
__optimum_mcs_objective_literal__(9999999, mus(492)).
__optimum_mcs_objective_literal__(9999999, mus(493)).
__optimum_mcs_objective_literal__(9999999, mus(494)).
__optimum_mcs_objective_literal__(9999999, mus(495)).
__optimum_mcs_objective_literal__(9999999, mus(496)).
__optimum_mcs_objective_literal__(9999999, mus(497)).
__optimum_mcs_objective_literal__(9999999, mus(498)).
__optimum_mcs_objective_literal__(9999999, mus(499)).
__optimum_mcs_objective_literal__(9999999, mus(500)).
__optimum_mcs_objective_literal__(9999999, mus(501)).
__optimum_mcs_objective_literal__(9999999, mus(502)).
__optimum_mcs_objective_literal__(9999999, mus(503)).
__optimum_mcs_objective_literal__(9999999, mus(504)).
__optimum_mcs_objective_literal__(9999999, mus(505)).
__optimum_mcs_objective_literal__(9999999, mus(506)).
__optimum_mcs_objective_literal__(9999999, mus(507)).
__optimum_mcs_objective_literal__(9999999, mus(508)).
__optimum_mcs_objective_literal__(9999999, mus(509)).
__optimum_mcs_objective_literal__(9999999, mus(510)).
__optimum_mcs_objective_literal__(9999999, mus(511)).
__optimum_mcs_objective_literal__(9999999, mus(512)).
__optimum_mcs_objective_literal__(9999999, mus(513)).
__optimum_mcs_objective_literal__(9999999, mus(514)).
__optimum_mcs_objective_literal__(9999999, mus(515)).
__optimum_mcs_objective_literal__(9999999, mus(516)).
__optimum_mcs_objective_literal__(9999999, mus(517)).
__optimum_mcs_objective_literal__(9999999, mus(518)).
__optimum_mcs_objective_literal__(9999999, mus(519)).
__optimum_mcs_objective_literal__(9999999, mus(520)).
__optimum_mcs_objective_literal__(9999999, mus(521)).
__optimum_mcs_objective_literal__(9999999, mus(522)).
__optimum_mcs_objective_literal__(9999999, mus(523)).
__optimum_mcs_objective_literal__(9999999, mus(524)).
__optimum_mcs_objective_literal__(9999999, mus(525)).
__optimum_mcs_objective_literal__(9999999, mus(526)).
__optimum_mcs_objective_literal__(9999999, mus(527)).
__optimum_mcs_objective_literal__(9999999, mus(528)).
__optimum_mcs_objective_literal__(9999999, mus(529)).
__optimum_mcs_objective_literal__(9999999, mus(530)).
__optimum_mcs_objective_literal__(9999999, mus(531)).
__optimum_mcs_objective_literal__(9999999, mus(532)).
__optimum_mcs_objective_literal__(9999999, mus(533)).
__optimum_mcs_objective_literal__(6376991, mus(534)).
__optimum_mcs_objective_literal__(9999999, mus(535)).
__optimum_mcs_objective_literal__(9999999, mus(536)).
__optimum_mcs_objective_literal__(9999999, mus(537)).
__optimum_mcs_objective_literal__(9999999, mus(538)).
__optimum_mcs_objective_literal__(9999999, mus(539)).
__optimum_mcs_objective_literal__(9999999, mus(540)).
__optimum_mcs_objective_literal__(9999999, mus(541)).
__optimum_mcs_objective_literal__(9999999, mus(542)).
__optimum_mcs_objective_literal__(9999999, mus(543)).
__optimum_mcs_objective_literal__(9999999, mus(544)).
__optimum_mcs_objective_literal__(9999998, mus(545)).
__optimum_mcs_objective_literal__(9999999, mus(546)).
__optimum_mcs_objective_literal__(9999999, mus(547)).
__optimum_mcs_objective_literal__(9999999, mus(548)).
__optimum_mcs_objective_literal__(9999999, mus(549)).
__optimum_mcs_objective_literal__(9999999, mus(550)).
__optimum_mcs_objective_literal__(9999999, mus(551)).
__optimum_mcs_objective_literal__(9907297, mus(552)).
__optimum_mcs_objective_literal__(9951189, mus(553)).
__optimum_mcs_objective_literal__(9419932, mus(554)).
__optimum_mcs_objective_literal__(7594751, mus(555)).
__optimum_mcs_objective_literal__(9829799, mus(556)).
__optimum_mcs_objective_literal__(9759071, mus(557)).
__optimum_mcs_objective_literal__(9981211, mus(558)).
__optimum_mcs_objective_literal__(9369371, mus(559)).
__optimum_mcs_objective_literal__(9982492, mus(560)).
__optimum_mcs_objective_literal__(7710763, mus(561)).
__optimum_mcs_objective_literal__(9999999, mus(562)).
__optimum_mcs_objective_literal__(9999999, mus(563)).
__optimum_mcs_objective_literal__(5035136, mus(564)).
__optimum_mcs_objective_literal__(8692376, mus(565)).
__optimum_mcs_objective_literal__(8777964, mus(566)).
__optimum_mcs_objective_literal__(9690269, mus(567)).
__optimum_mcs_objective_literal__(9111210, mus(568)).
__optimum_mcs_objective_literal__(9999946, mus(569)).
__optimum_mcs_objective_literal__(9998883, mus(570)).
__optimum_mcs_objective_literal__(9999999, mus(571)).
__optimum_mcs_objective_literal__(9999984, mus(572)).
__optimum_mcs_objective_literal__(6176138, mus(573)).
__optimum_mcs_objective_literal__(9847157, mus(574)).
__optimum_mcs_objective_literal__(5089997, mus(575)).
__optimum_mcs_objective_literal__(7460505, mus(576)).
__optimum_mcs_objective_literal__(9975740, mus(577)).
__optimum_mcs_objective_literal__(5055563, mus(578)).
__optimum_mcs_objective_literal__(9999999, mus(579)).
__optimum_mcs_objective_literal__(9999999, mus(580)).
__optimum_mcs_objective_literal__(9999999, mus(581)).
__optimum_mcs_objective_literal__(9999999, mus(582)).
__optimum_mcs_objective_literal__(9999999, mus(583)).
__optimum_mcs_objective_literal__(9999999, mus(584)).
__optimum_mcs_objective_literal__(9999999, mus(585)).
__optimum_mcs_objective_literal__(9999999, mus(586)).
__optimum_mcs_objective_literal__(9999999, mus(587)).
__optimum_mcs_objective_literal__(9999999, mus(588)).
__optimum_mcs_objective_literal__(9999999, mus(589)).
__optimum_mcs_objective_literal__(9999999, mus(590)).
__optimum_mcs_objective_literal__(9999999, mus(591)).
__optimum_mcs_objective_literal__(9999999, mus(592)).
__optimum_mcs_objective_literal__(9999999, mus(593)).
__optimum_mcs_objective_literal__(9999999, mus(594)).
__optimum_mcs_objective_literal__(9999999, mus(595)).
__optimum_mcs_objective_literal__(9999999, mus(596)).
__optimum_mcs_objective_literal__(9999999, mus(597)).
__optimum_mcs_objective_literal__(9999999, mus(598)).
__optimum_mcs_objective_literal__(9999999, mus(599)).
__optimum_mcs_objective_literal__(9999999, mus(600)).
__optimum_mcs_objective_literal__(9999999, mus(601)).
__optimum_mcs_objective_literal__(9999999, mus(602)).
__optimum_mcs_objective_literal__(9999999, mus(603)).
__optimum_mcs_objective_literal__(9999999, mus(604)).
__optimum_mcs_objective_literal__(9999999, mus(605)).
__optimum_mcs_objective_literal__(9999999, mus(606)).
__optimum_mcs_objective_literal__(9999999, mus(607)).
__optimum_mcs_objective_literal__(9999999, mus(608)).
__optimum_mcs_objective_literal__(9999999, mus(609)).
__optimum_mcs_objective_literal__(9999999, mus(610)).
__optimum_mcs_objective_literal__(9999999, mus(611)).
__optimum_mcs_objective_literal__(9999999, mus(612)).
__optimum_mcs_objective_literal__(9999999, mus(613)).
__optimum_mcs_objective_literal__(9996875, mus(614)).
__optimum_mcs_objective_literal__(8596069, mus(615)).
__optimum_mcs_objective_literal__(5816854, mus(616)).
__optimum_mcs_objective_literal__(5268657, mus(617)).
__optimum_mcs_objective_literal__(5153087, mus(618)).
__optimum_mcs_objective_literal__(8715582, mus(619)).
__optimum_mcs_objective_literal__(5512742, mus(620)).
__optimum_mcs_objective_literal__(9999998, mus(621)).
__optimum_mcs_objective_literal__(9827566, mus(622)).
__optimum_mcs_objective_literal__(9999789, mus(623)).
__optimum_mcs_objective_literal__(9938269, mus(624)).
__optimum_mcs_objective_literal__(5674512, mus(625)).
__optimum_mcs_objective_literal__(5701405, mus(626)).
__optimum_mcs_objective_literal__(6887488, mus(627)).
__optimum_mcs_objective_literal__(8637430, mus(628)).
__optimum_mcs_objective_literal__(9999052, mus(629)).
__optimum_mcs_objective_literal__(5668809, mus(630)).
__optimum_mcs_objective_literal__(9778836, mus(631)).
__optimum_mcs_objective_literal__(9999988, mus(632)).
__optimum_mcs_objective_literal__(7117897, mus(633)).
__optimum_mcs_objective_literal__(6832979, mus(634)).
__optimum_mcs_objective_literal__(5742743, mus(635)).
__optimum_mcs_objective_literal__(5480214, mus(636)).
__optimum_mcs_objective_literal__(6482630, mus(637)).
__optimum_mcs_objective_literal__(9999989, mus(638)).
__optimum_mcs_objective_literal__(9937535, mus(639)).
__optimum_mcs_objective_literal__(8278089, mus(640)).
__optimum_mcs_objective_literal__(6291857, mus(641)).
__optimum_mcs_objective_literal__(5870562, mus(642)).
__optimum_mcs_objective_literal__(9856835, mus(643)).
__optimum_mcs_objective_literal__(9999999, mus(644)).
__optimum_mcs_objective_literal__(9999999, mus(645)).
__optimum_mcs_objective_literal__(9999999, mus(646)).
__optimum_mcs_objective_literal__(9999999, mus(647)).
__optimum_mcs_objective_literal__(9999999, mus(648)).
__optimum_mcs_objective_literal__(9999999, mus(649)).
__optimum_mcs_objective_literal__(9999999, mus(650)).
__optimum_mcs_objective_literal__(9999999, mus(651)).
__optimum_mcs_objective_literal__(9999999, mus(652)).
__optimum_mcs_objective_literal__(9999999, mus(653)).
__optimum_mcs_objective_literal__(9999999, mus(654)).
__optimum_mcs_objective_literal__(9999999, mus(655)).
__optimum_mcs_objective_literal__(9999999, mus(656)).
__optimum_mcs_objective_literal__(9999999, mus(657)).
__optimum_mcs_objective_literal__(9999999, mus(658)).
__optimum_mcs_objective_literal__(9999999, mus(659)).
__optimum_mcs_objective_literal__(9999999, mus(660)).
__optimum_mcs_objective_literal__(9999999, mus(661)).
__optimum_mcs_objective_literal__(9999999, mus(662)).
__optimum_mcs_objective_literal__(9999999, mus(663)).
__optimum_mcs_objective_literal__(9999999, mus(664)).
__optimum_mcs_objective_literal__(9999999, mus(665)).
__optimum_mcs_objective_literal__(9999999, mus(666)).
__optimum_mcs_objective_literal__(9999999, mus(667)).
__optimum_mcs_objective_literal__(9999999, mus(668)).
__optimum_mcs_objective_literal__(9999999, mus(669)).
__optimum_mcs_objective_literal__(9999999, mus(670)).
__optimum_mcs_objective_literal__(9999999, mus(671)).
__optimum_mcs_objective_literal__(9999999, mus(672)).
__optimum_mcs_objective_literal__(9999999, mus(673)).
__optimum_mcs_objective_literal__(9999999, mus(674)).
__optimum_mcs_objective_literal__(5015640, mus(675)).
__optimum_mcs_objective_literal__(5451235, mus(676)).
__optimum_mcs_objective_literal__(5432793, mus(677)).
__optimum_mcs_objective_literal__(5280870, mus(678)).
__optimum_mcs_objective_literal__(5166652, mus(679)).
__optimum_mcs_objective_literal__(7178787, mus(680)).
__optimum_mcs_objective_literal__(8771687, mus(681)).
__optimum_mcs_objective_literal__(9876314, mus(682)).
__optimum_mcs_objective_literal__(9902068, mus(683)).
__optimum_mcs_objective_literal__(5107626, mus(684)).
__optimum_mcs_objective_literal__(8100335, mus(685)).
__optimum_mcs_objective_literal__(8643322, mus(686)).
__optimum_mcs_objective_literal__(5453775, mus(687)).
__optimum_mcs_objective_literal__(5612726, mus(688)).
__optimum_mcs_objective_literal__(9541830, mus(689)).
__optimum_mcs_objective_literal__(5800946, mus(690)).
__optimum_mcs_objective_literal__(8433430, mus(691)).
__optimum_mcs_objective_literal__(5960461, mus(692)).
__optimum_mcs_objective_literal__(8148826, mus(693)).
__optimum_mcs_objective_literal__(6048316, mus(694)).
__optimum_mcs_objective_literal__(5463253, mus(695)).
__optimum_mcs_objective_literal__(5386920, mus(696)).
__optimum_mcs_objective_literal__(6042028, mus(697)).
__optimum_mcs_objective_literal__(6725339, mus(698)).
__optimum_mcs_objective_literal__(6096334, mus(699)).
__optimum_mcs_objective_literal__(5112576, mus(700)).
__optimum_mcs_objective_literal__(6333865, mus(701)).
__optimum_mcs_objective_literal__(6993486, mus(702)).
__optimum_mcs_objective_literal__(5782424, mus(703)).
__optimum_mcs_objective_literal__(5613613, mus(704)).
__optimum_mcs_objective_literal__(5128277, mus(705)).
__optimum_mcs_objective_literal__(9919975, mus(706)).
__optimum_mcs_objective_literal__(5284866, mus(707)).
__optimum_mcs_objective_literal__(8518505, mus(708)).
__optimum_mcs_objective_literal__(5082220, mus(709)).
__optimum_mcs_objective_literal__(6091874, mus(710)).
__optimum_mcs_objective_literal__(5116372, mus(711)).
__optimum_mcs_objective_literal__(6296677, mus(712)).
__optimum_mcs_objective_literal__(5038372, mus(713)).
__optimum_mcs_objective_literal__(8897740, mus(714)).
__optimum_mcs_objective_literal__(9552134, mus(715)).
__optimum_mcs_objective_literal__(8881933, mus(716)).
__optimum_mcs_objective_literal__(5084640, mus(717)).
__optimum_mcs_objective_literal__(8262089, mus(718)).
__optimum_mcs_objective_literal__(6429168, mus(719)).
__optimum_mcs_objective_literal__(7867347, mus(720)).
__optimum_mcs_objective_literal__(6396631, mus(721)).
__optimum_mcs_objective_literal__(7548567, mus(722)).
__optimum_mcs_objective_literal__(6166666, mus(723)).
__optimum_mcs_objective_literal__(5586838, mus(724)).
__optimum_mcs_objective_literal__(7769299, mus(725)).
__optimum_mcs_objective_literal__(5144596, mus(726)).
__optimum_mcs_objective_literal__(7593306, mus(727)).
__optimum_mcs_objective_literal__(5698080, mus(728)).
__optimum_mcs_objective_literal__(8793059, mus(729)).
__optimum_mcs_objective_literal__(8349782, mus(730)).
__optimum_mcs_objective_literal__(6849868, mus(731)).
__optimum_mcs_objective_literal__(5888204, mus(732)).
__optimum_mcs_objective_literal__(5917843, mus(733)).
__optimum_mcs_objective_literal__(5471565, mus(734)).
__optimum_mcs_objective_literal__(5338689, mus(735)).
__optimum_mcs_objective_literal__(9999999, mus(736)).
__optimum_mcs_objective_literal__(9999999, mus(737)).
__optimum_mcs_objective_literal__(9999999, mus(738)).
__optimum_mcs_objective_literal__(9999999, mus(739)).
__optimum_mcs_objective_literal__(9999999, mus(740)).
__optimum_mcs_objective_literal__(9999999, mus(741)).
__optimum_mcs_objective_literal__(9999999, mus(742)).
__optimum_mcs_objective_literal__(9999999, mus(743)).
__optimum_mcs_objective_literal__(9999999, mus(744)).
__optimum_mcs_objective_literal__(9999999, mus(745)).
__optimum_mcs_objective_literal__(9999999, mus(746)).
__optimum_mcs_objective_literal__(9999999, mus(747)).
__optimum_mcs_objective_literal__(9999999, mus(748)).
__optimum_mcs_objective_literal__(9999999, mus(749)).
__optimum_mcs_objective_literal__(9999999, mus(750)).
__optimum_mcs_objective_literal__(6037921, mus(751)).
__optimum_mcs_objective_literal__(9999380, mus(752)).
__optimum_mcs_objective_literal__(6829458, mus(753)).
__optimum_mcs_objective_literal__(9927094, mus(754)).
__optimum_mcs_objective_literal__(8333748, mus(755)).
__optimum_mcs_objective_literal__(8927199, mus(756)).
__optimum_mcs_objective_literal__(9999996, mus(757)).
__optimum_mcs_objective_literal__(9673293, mus(758)).
__optimum_mcs_objective_literal__(9999896, mus(759)).
__optimum_mcs_objective_literal__(9999999, mus(760)).
__optimum_mcs_objective_literal__(9985061, mus(761)).
__optimum_mcs_objective_literal__(9786841, mus(762)).
__optimum_mcs_objective_literal__(9999999, mus(763)).
__optimum_mcs_objective_literal__(9999999, mus(764)).
__optimum_mcs_objective_literal__(9999999, mus(765)).
__optimum_mcs_objective_literal__(9999999, mus(766)).
__optimum_mcs_objective_literal__(9999999, mus(767)).
__optimum_mcs_objective_literal__(9999999, mus(768)).
__optimum_mcs_objective_literal__(9999999, mus(769)).
__optimum_mcs_objective_literal__(9999999, mus(770)).
__optimum_mcs_objective_literal__(9999999, mus(771)).
__optimum_mcs_objective_literal__(9999999, mus(772)).
__optimum_mcs_objective_literal__(9999999, mus(773)).
__optimum_mcs_objective_literal__(9999999, mus(774)).
__optimum_mcs_objective_literal__(9999999, mus(775)).
__optimum_mcs_objective_literal__(9999999, mus(776)).
__optimum_mcs_objective_literal__(9999999, mus(777)).
__optimum_mcs_objective_literal__(9999999, mus(778)).
__optimum_mcs_objective_literal__(9999999, mus(779)).
__optimum_mcs_objective_literal__(9999999, mus(780)).
__optimum_mcs_objective_literal__(9999999, mus(781)).
__optimum_mcs_objective_literal__(9999999, mus(782)).
__optimum_mcs_objective_literal__(9999999, mus(783)).
__optimum_mcs_objective_literal__(9999999, mus(784)).
__optimum_mcs_objective_literal__(9999999, mus(785)).
__optimum_mcs_objective_literal__(9999999, mus(786)).
__optimum_mcs_objective_literal__(9999999, mus(787)).
__optimum_mcs_objective_literal__(9999999, mus(788)).
__optimum_mcs_objective_literal__(9999999, mus(789)).
__optimum_mcs_objective_literal__(9999999, mus(790)).
__optimum_mcs_objective_literal__(9999999, mus(791)).
__optimum_mcs_objective_literal__(9999999, mus(792)).
__optimum_mcs_objective_literal__(9999999, mus(793)).
__optimum_mcs_objective_literal__(9999999, mus(794)).
__optimum_mcs_objective_literal__(9999999, mus(795)).
__optimum_mcs_objective_literal__(9999999, mus(796)).
__optimum_mcs_objective_literal__(9999999, mus(797)).
__optimum_mcs_objective_literal__(9999999, mus(798)).
__optimum_mcs_objective_literal__(9999999, mus(799)).
__optimum_mcs_objective_literal__(6472177, mus(800)).
__optimum_mcs_objective_literal__(9204251, mus(801)).
__optimum_mcs_objective_literal__(7508574, mus(802)).
__optimum_mcs_objective_literal__(7059852, mus(803)).
__optimum_mcs_objective_literal__(6192736, mus(804)).
__optimum_mcs_objective_literal__(9995875, mus(805)).
__optimum_mcs_objective_literal__(6465127, mus(806)).
__optimum_mcs_objective_literal__(8757305, mus(807)).
__optimum_mcs_objective_literal__(6166905, mus(808)).
__optimum_mcs_objective_literal__(6370977, mus(809)).
__optimum_mcs_objective_literal__(6418298, mus(810)).
__optimum_mcs_objective_literal__(8094269, mus(811)).
__optimum_mcs_objective_literal__(7408418, mus(812)).
__optimum_mcs_objective_literal__(9999999, mus(813)).
__optimum_mcs_objective_literal__(9999999, mus(814)).
__optimum_mcs_objective_literal__(9999999, mus(815)).
__optimum_mcs_objective_literal__(9999999, mus(816)).
__optimum_mcs_objective_literal__(9999999, mus(817)).
__optimum_mcs_objective_literal__(9999999, mus(818)).
__optimum_mcs_objective_literal__(9999999, mus(819)).
__optimum_mcs_objective_literal__(9999999, mus(820)).
__optimum_mcs_objective_literal__(9999999, mus(821)).
__optimum_mcs_objective_literal__(9999999, mus(822)).
__optimum_mcs_objective_literal__(9999999, mus(823)).
__optimum_mcs_objective_literal__(9999999, mus(824)).
__optimum_mcs_objective_literal__(9999999, mus(825)).
__optimum_mcs_objective_literal__(9999999, mus(826)).
__optimum_mcs_objective_literal__(9999999, mus(827)).
__optimum_mcs_objective_literal__(9999999, mus(828)).
__optimum_mcs_objective_literal__(9999999, mus(829)).
__optimum_mcs_objective_literal__(9999999, mus(830)).
__optimum_mcs_objective_literal__(9999999, mus(831)).
__optimum_mcs_objective_literal__(9999999, mus(832)).
__optimum_mcs_objective_literal__(9999999, mus(833)).
__optimum_mcs_objective_literal__(9999999, mus(834)).
__optimum_mcs_objective_literal__(9999999, mus(835)).
__optimum_mcs_objective_literal__(9999999, mus(836)).
__optimum_mcs_objective_literal__(9999999, mus(837)).
__optimum_mcs_objective_literal__(9999999, mus(838)).
__optimum_mcs_objective_literal__(9999999, mus(839)).
__optimum_mcs_objective_literal__(9999999, mus(840)).
__optimum_mcs_objective_literal__(9999999, mus(841)).
__optimum_mcs_objective_literal__(9999999, mus(842)).
__optimum_mcs_objective_literal__(9999999, mus(843)).
__optimum_mcs_objective_literal__(9999999, mus(844)).
__optimum_mcs_objective_literal__(9999999, mus(845)).
__optimum_mcs_objective_literal__(9999999, mus(846)).
__optimum_mcs_objective_literal__(9999999, mus(847)).
__optimum_mcs_objective_literal__(9999999, mus(848)).
__optimum_mcs_objective_literal__(9999999, mus(849)).
__optimum_mcs_objective_literal__(9999999, mus(850)).
__optimum_mcs_objective_literal__(9999999, mus(851)).
__optimum_mcs_objective_literal__(9999999, mus(852)).
__optimum_mcs_objective_literal__(9999999, mus(853)).
__optimum_mcs_objective_literal__(9999999, mus(854)).
__optimum_mcs_objective_literal__(9999999, mus(855)).
__optimum_mcs_objective_literal__(9999999, mus(856)).
__optimum_mcs_objective_literal__(9999999, mus(857)).
__optimum_mcs_objective_literal__(9999999, mus(858)).
__optimum_mcs_objective_literal__(9999999, mus(859)).
__optimum_mcs_objective_literal__(9999999, mus(860)).
__optimum_mcs_objective_literal__(9999999, mus(861)).
__optimum_mcs_objective_literal__(9999999, mus(862)).
__optimum_mcs_objective_literal__(9999999, mus(863)).
__optimum_mcs_objective_literal__(9999999, mus(864)).
__optimum_mcs_objective_literal__(9999999, mus(865)).
__optimum_mcs_objective_literal__(9999999, mus(866)).
__optimum_mcs_objective_literal__(9999999, mus(867)).
__optimum_mcs_objective_literal__(9999999, mus(868)).
__optimum_mcs_objective_literal__(9999999, mus(869)).
__optimum_mcs_objective_literal__(9999999, mus(870)).
__optimum_mcs_objective_literal__(9999999, mus(871)).
__optimum_mcs_objective_literal__(9999999, mus(872)).
__optimum_mcs_objective_literal__(9999999, mus(873)).
__optimum_mcs_objective_literal__(9999999, mus(874)).
__optimum_mcs_objective_literal__(9999999, mus(875)).
__optimum_mcs_objective_literal__(9999999, mus(876)).
__optimum_mcs_objective_literal__(9999999, mus(877)).
__optimum_mcs_objective_literal__(9999999, mus(878)).
__optimum_mcs_objective_literal__(9999999, mus(879)).
__optimum_mcs_objective_literal__(5974791, mus(880)).
__optimum_mcs_objective_literal__(7032022, mus(881)).
__optimum_mcs_objective_literal__(6519265, mus(882)).
__optimum_mcs_objective_literal__(6337056, mus(883)).
__optimum_mcs_objective_literal__(8690973, mus(884)).
__optimum_mcs_objective_literal__(8343413, mus(885)).
__optimum_mcs_objective_literal__(6040737, mus(886)).
__optimum_mcs_objective_literal__(6922178, mus(887)).
__optimum_mcs_objective_literal__(7837861, mus(888)).
__optimum_mcs_objective_literal__(7422507, mus(889)).
__optimum_mcs_objective_literal__(8519586, mus(890)).
__optimum_mcs_objective_literal__(9722609, mus(891)).
__optimum_mcs_objective_literal__(6689668, mus(892)).
__optimum_mcs_objective_literal__(5283331, mus(893)).
__optimum_mcs_objective_literal__(6146926, mus(894)).
__optimum_mcs_objective_literal__(7748273, mus(895)).
__optimum_mcs_objective_literal__(7349784, mus(896)).
__optimum_mcs_objective_literal__(7483093, mus(897)).
__optimum_mcs_objective_literal__(6788846, mus(898)).
__optimum_mcs_objective_literal__(5233000, mus(899)).
__optimum_mcs_objective_literal__(6762805, mus(900)).
__optimum_mcs_objective_literal__(5532801, mus(901)).
__optimum_mcs_objective_literal__(8421051, mus(902)).
__optimum_mcs_objective_literal__(5519886, mus(903)).
__optimum_mcs_objective_literal__(7602960, mus(904)).
__optimum_mcs_objective_literal__(7493288, mus(905)).
__optimum_mcs_objective_literal__(8263178, mus(906)).
__optimum_mcs_objective_literal__(9783282, mus(907)).
__optimum_mcs_objective_literal__(5957711, mus(908)).
__optimum_mcs_objective_literal__(6817235, mus(909)).
__optimum_mcs_objective_literal__(9100432, mus(910)).
__optimum_mcs_objective_literal__(7392279, mus(911)).
__optimum_mcs_objective_literal__(6979514, mus(912)).
__optimum_mcs_objective_literal__(7077687, mus(913)).
__optimum_mcs_objective_literal__(5524354, mus(914)).
__optimum_mcs_objective_literal__(7666130, mus(915)).
__optimum_mcs_objective_literal__(5958039, mus(916)).
__optimum_mcs_objective_literal__(5657680, mus(917)).
__optimum_mcs_objective_literal__(5053857, mus(918)).
__optimum_mcs_objective_literal__(6635215, mus(919)).
__optimum_mcs_objective_literal__(7985654, mus(920)).
__optimum_mcs_objective_literal__(6513423, mus(921)).
__optimum_mcs_objective_literal__(5490494, mus(922)).
__optimum_mcs_objective_literal__(8321896, mus(923)).
__optimum_mcs_objective_literal__(8496251, mus(924)).
__optimum_mcs_objective_literal__(5106771, mus(925)).
__optimum_mcs_objective_literal__(9756373, mus(926)).
__optimum_mcs_objective_literal__(9790273, mus(927)).
__optimum_mcs_objective_literal__(6954127, mus(928)).
__optimum_mcs_objective_literal__(7344074, mus(929)).
__optimum_mcs_objective_literal__(7767012, mus(930)).
__optimum_mcs_objective_literal__(8162186, mus(931)).
__optimum_mcs_objective_literal__(5330125, mus(932)).
__optimum_mcs_objective_literal__(7702821, mus(933)).
__optimum_mcs_objective_literal__(6898275, mus(934)).
__optimum_mcs_objective_literal__(7704760, mus(935)).
__optimum_mcs_objective_literal__(6129993, mus(936)).
__optimum_mcs_objective_literal__(7074206, mus(937)).
__optimum_mcs_objective_literal__(5399731, mus(938)).
__optimum_mcs_objective_literal__(5063627, mus(939)).
__optimum_mcs_objective_literal__(9038562, mus(940)).
__optimum_mcs_objective_literal__(9364875, mus(941)).
__optimum_mcs_objective_literal__(8541392, mus(942)).
__optimum_mcs_objective_literal__(7733203, mus(943)).
__optimum_mcs_objective_literal__(5968546, mus(944)).
__optimum_mcs_objective_literal__(5084824, mus(945)).
__optimum_mcs_objective_literal__(5441215, mus(946)).
__optimum_mcs_objective_literal__(9721238, mus(947)).
__optimum_mcs_objective_literal__(6693446, mus(948)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
