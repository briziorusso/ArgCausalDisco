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
qpair(0,2,s5).
qpair(0,2,s4y9).
qpair(0,2,s8).
qpair(0,2,s4y6).
qpair(0,2,s4y5y6).
qpair(0,2,s1y6).
qpair(0,2,s1y4y5).
qpair(0,2,s1y3).
qpair(0,2,s1y9).
qpair(0,2,s6y8).
qpair(0,2,s4).
qpair(0,2,s1y5y6).
qpair(0,2,s1).
qpair(0,2,s4y5).
qpair(0,2,s3y9).
qpair(0,2,s5y6).
qpair(0,2,s4y8).
qpair(0,2,s3y6).
qpair(0,2,s5y9).
qpair(0,2,s1y5).
qpair(0,2,s1y8).
qpair(0,2,s3).
qpair(0,2,s9).
qpair(0,2,s3y5).
qpair(0,2,s6).
qpair(0,2,s3y8).
qpair(0,2,s1y4y5y6).
qpair(0,2,s5y8).
qpair(0,2,s1y4).
qpair(0,2,empty).
qpair(0,2,s6y9).
qpair(0,3,s6).
qpair(0,3,s2).
qpair(0,3,s1y2).
qpair(0,3,s5).
qpair(0,3,s1y5).
qpair(0,3,s4).
qpair(0,3,s1).
qpair(0,3,s7).
qpair(0,3,s2y6).
qpair(0,3,s5y6).
qpair(0,3,s2y5).
qpair(0,5,s4).
qpair(0,6,s2).
qpair(0,6,s5).
qpair(0,6,s3y7).
qpair(0,6,s2y4y7y8).
qpair(0,6,s8).
qpair(0,6,s1y3).
qpair(0,6,s2y8).
qpair(0,6,s2y4y8).
qpair(0,6,s4).
qpair(0,6,s1).
qpair(0,6,s7).
qpair(0,6,s4y8).
qpair(0,6,s2y4).
qpair(0,6,s1y2).
qpair(0,6,s2y7).
qpair(0,6,s4y7y8).
qpair(0,6,s2y4y7).
qpair(0,6,s3).
qpair(0,6,s3y8).
qpair(0,6,s2y3).
qpair(0,6,empty).
qpair(0,6,s7y8).
qpair(0,7,s6).
qpair(0,7,s2).
qpair(0,7,s2y3y6).
qpair(0,7,s2y6).
qpair(0,7,s3y6).
qpair(0,8,s1y6y9).
qpair(0,8,s4y6).
qpair(0,8,s1y5y6y9).
qpair(0,8,s4y5y6).
qpair(0,8,s1y4y5y6y9).
qpair(0,8,s1y6).
qpair(0,8,s4y5y6y9).
qpair(0,8,s2y3y6).
qpair(0,8,s1y5y6).
qpair(0,8,s1y5y9).
qpair(0,8,s5y6).
qpair(0,8,s3y6).
qpair(0,8,s4y6y9).
qpair(0,8,s6).
qpair(0,8,s5y6y9).
qpair(0,8,s1y4y5y6).
qpair(0,8,s1y4y6).
qpair(0,8,s2y6).
qpair(0,8,s6y9).
qpair(1,2,s3y4).
qpair(1,2,s4y9).
qpair(1,2,s8).
qpair(1,2,s4y6).
qpair(1,2,s3y5y9).
qpair(1,2,s0y4y5y6).
qpair(1,2,s0y5).
qpair(1,2,s4y5y6).
qpair(1,2,s0y8).
qpair(1,2,s6y8).
qpair(1,2,s0y5y6).
qpair(1,2,s4y5).
qpair(1,2,s3y5y8).
qpair(1,2,s3y9).
qpair(1,2,s5y6).
qpair(1,2,s4y8).
qpair(1,2,s3y6).
qpair(1,2,s5y9).
qpair(1,2,s0y4).
qpair(1,2,s3y8y9).
qpair(1,2,s0).
qpair(1,2,s3).
qpair(1,2,s9).
qpair(1,2,s3y5).
qpair(1,2,s6).
qpair(1,2,s5y8y9).
qpair(1,2,s3y8).
qpair(1,2,s3y5y8y9).
qpair(1,2,s5y8).
qpair(1,2,s0y3).
qpair(1,2,s0y9).
qpair(1,2,s0y6).
qpair(1,2,empty).
qpair(1,2,s6y9).
qpair(1,2,s0y4y5).
qpair(1,3,s2).
qpair(1,3,s2y8y9).
qpair(1,3,s5).
qpair(1,3,s8).
qpair(1,3,s0y2).
qpair(1,3,s8y9).
qpair(1,3,s0y5).
qpair(1,3,s0y8).
qpair(1,3,s2y5).
qpair(1,3,s2y8).
qpair(1,3,s4).
qpair(1,3,s7).
qpair(1,3,s5y6).
qpair(1,3,s5y9).
qpair(1,3,s2y5y9).
qpair(1,3,s0).
qpair(1,3,s9).
qpair(1,3,s6).
qpair(1,3,s5y8y9).
qpair(1,3,s5y8).
qpair(1,3,s2y5y8).
qpair(1,3,s0y9).
qpair(1,3,s0y6).
qpair(1,3,s2y5y8y9).
qpair(1,3,s2y9).
qpair(1,3,empty).
qpair(1,5,s2).
qpair(1,5,s2y8y9).
qpair(1,5,s8).
qpair(1,5,s0y2).
qpair(1,5,s8y9).
qpair(1,5,s2y3y8y9).
qpair(1,5,s0y8).
qpair(1,5,s2y8).
qpair(1,5,s2y3y9).
qpair(1,5,s4).
qpair(1,5,s3y9).
qpair(1,5,s2y3y8).
qpair(1,5,s3y8y9).
qpair(1,5,s0).
qpair(1,5,s3).
qpair(1,5,s6).
qpair(1,5,s3y8).
qpair(1,5,s0y9).
qpair(1,5,s2y3).
qpair(1,5,empty).
qpair(1,7,empty).
qpair(1,8,s2).
qpair(1,8,s5).
qpair(1,8,s4y9).
qpair(1,8,s3y5y9).
qpair(1,8,s0y2).
qpair(1,8,s0y5).
qpair(1,8,s2y5).
qpair(1,8,s2y3y9).
qpair(1,8,s4y5y6y9).
qpair(1,8,s4).
qpair(1,8,s4y5).
qpair(1,8,s3y9).
qpair(1,8,s5y6).
qpair(1,8,s5y9).
qpair(1,8,s2y4).
qpair(1,8,s4y6y9).
qpair(1,8,s2y3y5).
qpair(1,8,s2y3y5y9).
qpair(1,8,s0).
qpair(1,8,s3).
qpair(1,8,s9).
qpair(1,8,s3y5).
qpair(1,8,s6).
qpair(1,8,s5y6y9).
qpair(1,8,s0y3).
qpair(1,8,s0y9).
qpair(1,8,s2y3).
qpair(1,8,s2y9).
qpair(1,8,s2y6).
qpair(1,8,empty).
qpair(1,8,s6y9).
qpair(1,9,s5).
qpair(1,9,s8).
qpair(1,9,s0y2).
qpair(1,9,s0y5).
qpair(1,9,s0y8).
qpair(1,9,s2y5).
qpair(1,9,s2y8).
qpair(1,9,s3y5y8).
qpair(1,9,s2y3y5).
qpair(1,9,s2y3y8).
qpair(1,9,s0).
qpair(1,9,s3).
qpair(1,9,s3y5).
qpair(1,9,s3y8).
qpair(1,9,s5y8).
qpair(1,9,s0y3).
qpair(1,9,s2y3).
qpair(1,9,empty).
qpair(1,9,s2y3y5y8).
qpair(2,3,s5y8).
qpair(2,3,s4y9).
qpair(2,3,s8).
qpair(2,3,s0y9).
qpair(2,3,s1y8).
qpair(2,3,s7).
qpair(2,3,s4y8).
qpair(2,3,s5y9).
qpair(2,3,s0y8).
qpair(2,3,s6y9).
qpair(2,3,s9).
qpair(2,4,s5).
qpair(2,4,s8).
qpair(2,4,s0y1y6).
qpair(2,4,s8y9).
qpair(2,4,s1y6).
qpair(2,4,s0y8).
qpair(2,4,s1y3).
qpair(2,4,s1y9).
qpair(2,4,s6y8).
qpair(2,4,s1y5y6).
qpair(2,4,s1).
qpair(2,4,s0y1y5y6).
qpair(2,4,s3y9).
qpair(2,4,s5y6).
qpair(2,4,s0y1y5).
qpair(2,4,s3y6).
qpair(2,4,s5y9).
qpair(2,4,s0y1).
qpair(2,4,s1y8).
qpair(2,4,s0).
qpair(2,4,s9).
qpair(2,4,s3y5).
qpair(2,4,s6).
qpair(2,4,s3y8).
qpair(2,4,s5y8).
qpair(2,4,s0y3).
qpair(2,4,s0y9).
qpair(2,4,s0y6).
qpair(2,4,empty).
qpair(2,4,s6y9).
qpair(2,5,s3y4).
qpair(2,5,s4y9).
qpair(2,5,s4y6).
qpair(2,5,s0y1y6).
qpair(2,5,s1y6).
qpair(2,5,s0y8).
qpair(2,5,s1y3).
qpair(2,5,s1y9).
qpair(2,5,s6y8).
qpair(2,5,s4).
qpair(2,5,s4y8).
qpair(2,5,s0y1).
qpair(2,5,s0y4).
qpair(2,5,s1y3y8).
qpair(2,5,s1y8).
qpair(2,5,s0y4y6).
qpair(2,5,s0).
qpair(2,5,s3).
qpair(2,5,s9).
qpair(2,5,s0y1y4y6).
qpair(2,5,s6).
qpair(2,5,s3y8).
qpair(2,5,s0y1y4).
qpair(2,5,s0y3).
qpair(2,5,s1y4).
qpair(2,5,s0y6).
qpair(2,5,empty).
qpair(2,5,s6y9).
qpair(2,6,s0y4y8).
qpair(2,6,s0y1y4y5).
qpair(2,6,s3y4).
qpair(2,6,s5).
qpair(2,6,s4y9).
qpair(2,6,s3y7).
qpair(2,6,s8).
qpair(2,6,s8y9).
qpair(2,6,s0y5).
qpair(2,6,s0y8).
qpair(2,6,s1y4y5).
qpair(2,6,s1y3).
qpair(2,6,s0y4y7).
qpair(2,6,s4).
qpair(2,6,s1).
qpair(2,6,s7).
qpair(2,6,s4y5).
qpair(2,6,s3y9).
qpair(2,6,s0y1y5).
qpair(2,6,s4y8).
qpair(2,6,s5y9).
qpair(2,6,s0y1).
qpair(2,6,s0y7).
qpair(2,6,s0y4).
qpair(2,6,s4y7y8).
qpair(2,6,s1y5).
qpair(2,6,s1y8).
qpair(2,6,s0).
qpair(2,6,s0y7y8).
qpair(2,6,s3).
qpair(2,6,s9).
qpair(2,6,s3y5).
qpair(2,6,s4y7).
qpair(2,6,s3y8).
qpair(2,6,s0y1y4).
qpair(2,6,s5y8).
qpair(2,6,s0y3).
qpair(2,6,s0y9).
qpair(2,6,s1y4).
qpair(2,6,empty).
qpair(2,6,s0y4y5).
qpair(2,6,s7y8).
qpair(2,7,s0y1y4y5).
qpair(2,7,s4y6).
qpair(2,7,s0y1y6).
qpair(2,7,s0y4y5y6).
qpair(2,7,s4y5y6).
qpair(2,7,s1y6).
qpair(2,7,s1y4y5).
qpair(2,7,s0y5y6).
qpair(2,7,s1y5y6).
qpair(2,7,s1).
qpair(2,7,s4y5).
qpair(2,7,s0y1y5y6).
qpair(2,7,s5y6).
qpair(2,7,s0y1y4y5y6).
qpair(2,7,s1y5).
qpair(2,7,s0y4y6).
qpair(2,7,s0y1y4y6).
qpair(2,7,s6).
qpair(2,7,s0y1y4).
qpair(2,7,s1y4y5y6).
qpair(2,7,s1y4y6).
qpair(2,7,s1y4).
qpair(2,7,s0y6).
qpair(2,8,s3y4).
qpair(2,8,s0y9).
qpair(2,8,s3).
qpair(2,8,s3y9).
qpair(2,8,s3y6).
qpair(2,8,s1y3).
qpair(2,8,s3y5).
qpair(2,9,s3y8).
qpair(2,9,s0y1y4).
qpair(2,9,s0y1y4y6).
qpair(2,9,s3y4).
qpair(2,9,s1y4y5y6).
qpair(2,9,s5y8).
qpair(2,9,s6y8).
qpair(2,9,s0y3).
qpair(2,9,s0y5y6).
qpair(2,9,s1y4y6).
qpair(2,9,s0y1y6).
qpair(2,9,s3).
qpair(2,9,s4y8).
qpair(2,9,s3y6).
qpair(2,9,s0y8).
qpair(2,9,s1y3).
qpair(2,9,s0y4y5).
qpair(3,4,s6).
qpair(3,4,s1).
qpair(3,4,s0).
qpair(3,4,empty).
qpair(3,4,s9).
qpair(3,5,s6).
qpair(3,5,s0y1).
qpair(3,5,s2).
qpair(3,5,s1y2).
qpair(3,5,s8).
qpair(3,5,s1y8).
qpair(3,5,s4).
qpair(3,5,s0y6).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s0y2).
qpair(3,5,s2y6).
qpair(3,5,s0).
qpair(3,5,s1y6).
qpair(3,5,s1y2y8).
qpair(3,5,s9).
qpair(3,5,s2y8).
qpair(3,6,s1y2).
qpair(3,6,s2y4y8).
qpair(3,6,s4y7y8).
qpair(3,6,s0y4y7).
qpair(3,6,s2y4y7y8).
qpair(3,6,s2y4y7).
qpair(3,6,s2y7y8).
qpair(3,6,s0y4y7y8).
qpair(3,6,s0y7y8).
qpair(3,6,s2y5).
qpair(3,6,s9).
qpair(3,7,s6).
qpair(3,7,s1).
qpair(3,7,empty).
qpair(3,9,s6).
qpair(3,9,s4).
qpair(3,9,s7).
qpair(3,9,s0y1y5).
qpair(3,9,s0y5).
qpair(4,6,s2).
qpair(4,6,s5).
qpair(4,6,s3y7).
qpair(4,6,s8).
qpair(4,6,s0y8).
qpair(4,6,s2y8).
qpair(4,6,s0y2y7).
qpair(4,6,s7).
qpair(4,6,s0y7).
qpair(4,6,s2y7).
qpair(4,6,empty).
qpair(4,6,s0).
qpair(4,6,s0y7y8).
qpair(4,6,s3).
qpair(4,6,s3y8).
qpair(4,6,s0y3).
qpair(4,6,s2y7y8).
qpair(4,6,s2y3).
qpair(4,6,s0y2y8).
qpair(4,6,s7y8).
qpair(4,7,s6).
qpair(4,7,s6y8).
qpair(4,7,s2y6).
qpair(4,7,s2y6y8).
qpair(4,8,s2).
qpair(4,8,s5).
qpair(4,8,s1y6y9).
qpair(4,8,s1y5y6y9).
qpair(4,8,s1y6).
qpair(4,8,s2y5).
qpair(4,8,s1y9).
qpair(4,8,s1y5y6).
qpair(4,8,s1).
qpair(4,8,s1y5y9).
qpair(4,8,s5y6).
qpair(4,8,s5y9).
qpair(4,8,s1y5).
qpair(4,8,s9).
qpair(4,8,s6).
qpair(4,8,s5y6y9).
qpair(4,8,s2y9).
qpair(4,8,s2y6).
qpair(4,8,empty).
qpair(4,8,s6y9).
qpair(4,9,s6y8).
qpair(5,7,empty).
qpair(5,8,s2).
qpair(5,8,s4y9).
qpair(5,8,s1y6y9).
qpair(5,8,s4y6).
qpair(5,8,s1y6).
qpair(5,8,s1y3).
qpair(5,8,s1y9).
qpair(5,8,s4).
qpair(5,8,s1y4y6y9).
qpair(5,8,s2y4).
qpair(5,8,s1y2).
qpair(5,8,s4y6y9).
qpair(5,8,s0).
qpair(5,8,s3).
qpair(5,8,s6).
qpair(5,8,s1y2y3).
qpair(5,8,s1y4y6).
qpair(5,8,s1y4).
qpair(5,8,s2y3).
qpair(5,8,s2y9).
qpair(5,8,s1y4y9).
qpair(5,8,s2y6).
qpair(5,8,empty).
qpair(5,8,s6y9).
qpair(6,7,s0y4y8).
qpair(6,7,s2).
qpair(6,7,s3y4).
qpair(6,7,s5).
qpair(6,7,s8).
qpair(6,7,s0y2).
qpair(6,7,s0y8).
qpair(6,7,s2y8).
qpair(6,7,s0y2y4).
qpair(6,7,s2y4y8).
qpair(6,7,s4).
qpair(6,7,s4y8).
qpair(6,7,s0y4).
qpair(6,7,s0y2y4y8).
qpair(6,7,empty).
qpair(6,7,s0).
qpair(6,7,s3).
qpair(6,7,s3y8).
qpair(6,7,s0y3).
qpair(6,7,s0y2y8).
qpair(6,8,s2).
qpair(6,8,s3y4).
qpair(6,8,s5).
qpair(6,8,s3y7).
qpair(6,8,s0y2).
qpair(6,8,s4y5y9).
qpair(6,8,s2y5).
qpair(6,8,s1y4y5).
qpair(6,8,s1y9).
qpair(6,8,s0y2y4).
qpair(6,8,s4).
qpair(6,8,s1).
qpair(6,8,s7).
qpair(6,8,s4y5).
qpair(6,8,s1y5y9).
qpair(6,8,s5y9).
qpair(6,8,s0y7).
qpair(6,8,s2y4).
qpair(6,8,s0y4).
qpair(6,8,s2y7).
qpair(6,8,s1y5).
qpair(6,8,s2y4y7).
qpair(6,8,s0).
qpair(6,8,s3).
qpair(6,8,s9).
qpair(6,8,s1y4y5y9).
qpair(6,8,s0y3).
qpair(6,8,s1y4).
qpair(6,8,s2y3).
qpair(6,8,s2y9).
qpair(6,8,s1y4y9).
qpair(6,8,s0y2y4y7).
qpair(6,8,empty).
qpair(6,9,s0y7).
qpair(6,9,s0y2y7y8).
qpair(6,9,s2y7).
qpair(6,9,s4y7y8).
qpair(6,9,s0y4y7y8).
qpair(6,9,s7).
qpair(6,9,s4y8).
qpair(6,9,s4y7).
qpair(7,8,s5).
qpair(7,8,s4y9).
qpair(7,8,s1y6y9).
qpair(7,8,s4y6).
qpair(7,8,s1y5y6y9).
qpair(7,8,s4y5y9).
qpair(7,8,s4y5y6).
qpair(7,8,s1y4y5y6y9).
qpair(7,8,s1y6).
qpair(7,8,s1y4y5).
qpair(7,8,s4y5y6y9).
qpair(7,8,s1).
qpair(7,8,s1y5y9).
qpair(7,8,s5y6).
qpair(7,8,s1y4y6y9).
qpair(7,8,s4y6y9).
qpair(7,8,s1y5).
qpair(7,8,s9).
qpair(7,8,s6).
qpair(7,8,s5y6y9).
qpair(7,8,s1y4y5y9).
qpair(7,8,s1y4y6).
qpair(7,8,s1y4y9).
qpair(7,8,s6y9).
qpair(8,9,s2).
qpair(8,9,s5).
qpair(8,9,s4y6).
qpair(8,9,s4y5y6).
qpair(8,9,s1y6).
qpair(8,9,s1y4y5).
qpair(8,9,s2y5).
qpair(8,9,s4).
qpair(8,9,s1y5y6).
qpair(8,9,s1).
qpair(8,9,s4y5).
qpair(8,9,s5y6).
qpair(8,9,s2y4).
qpair(8,9,s1y2).
qpair(8,9,s1y5).
qpair(8,9,s3).
qpair(8,9,s6).
qpair(8,9,s1y4y5y6).
qpair(8,9,s1y4y6).
qpair(8,9,s1y4).
qpair(8,9,s2y6).
qpair(8,9,empty).
qpair(0,2,s8y9).
qpair(0,2,s1y4y6).
qpair(0,2,s3y4).
qpair(0,3,s1y6).
qpair(0,3,empty).
qpair(0,3,s9).
qpair(0,6,s4y7).
qpair(0,6,s3y4).
qpair(0,6,s2y7y8).
qpair(0,7,s2y3).
qpair(0,7,s3).
qpair(0,8,s2).
qpair(0,8,s5).
qpair(0,8,s1y5).
qpair(0,8,s4y9).
qpair(0,8,s1y9).
qpair(0,8,s1y4y5y9).
qpair(0,8,s4).
qpair(0,8,s1y4).
qpair(0,8,s2y3).
qpair(0,8,s1).
qpair(0,8,s4y5).
qpair(0,8,s1y4y9).
qpair(0,8,s4y5y9).
qpair(0,8,s5y9).
qpair(0,8,s1y4y5).
qpair(0,8,s3).
qpair(0,8,s9).
qpair(0,8,s1y4y6y9).
qpair(0,9,empty).
qpair(1,2,s8y9).
qpair(1,2,s0y4y6).
qpair(1,2,s4).
qpair(1,2,s5).
qpair(1,3,s2y6).
qpair(1,5,s2y9).
qpair(1,5,s0y3).
qpair(1,5,s9).
qpair(1,8,s4y6).
qpair(1,8,s4y5y9).
qpair(1,8,s2y5y9).
qpair(1,8,s4y5y6).
qpair(1,9,s2y5y8).
qpair(1,9,s2).
qpair(2,3,s6y8).
qpair(2,3,s8y9).
qpair(2,3,s1y9).
qpair(2,4,s0y5y6).
qpair(2,4,s3).
qpair(2,4,s0y5).
qpair(2,4,s1y5).
qpair(2,5,s8).
qpair(2,5,s0y9).
qpair(2,5,s1y4y6).
qpair(2,5,s1).
qpair(2,5,s8y9).
qpair(2,5,s3y9).
qpair(2,5,s3y6).
qpair(2,6,s0y4y7y8).
qpair(2,6,s1y9).
qpair(2,7,s0y1).
qpair(2,7,s0y4).
qpair(2,7,s5).
qpair(2,7,s4).
qpair(2,7,s0y1y5).
qpair(2,7,s0y5).
qpair(2,7,s0).
qpair(2,7,s0y4y5).
qpair(2,8,s0y3).
qpair(2,9,s0y1y4y5).
qpair(2,9,s0y4y5y6).
qpair(2,9,s1y5y6).
qpair(2,9,s0y1y5y6).
qpair(2,9,s0y1y5).
qpair(2,9,s0y4y6).
qpair(2,9,s4y5y6).
qpair(2,9,s1y4y5).
qpair(2,9,s3y5).
qpair(2,9,s0y1y4y5y6).
qpair(3,4,s7).
qpair(3,4,s2).
qpair(3,4,s8).
qpair(3,4,s5).
qpair(3,5,empty).
qpair(3,6,s0y4y8).
qpair(3,6,s0y2y7).
qpair(3,6,s0y2y7y8).
qpair(3,6,s0y2y4).
qpair(3,6,s0y2y4y8).
qpair(3,6,s0y2y4y7y8).
qpair(3,6,s0y1y5).
qpair(3,6,s0y2y4y7).
qpair(3,6,s0y2y8).
qpair(3,7,s2).
qpair(3,7,s5).
qpair(3,7,s4).
qpair(3,7,s0).
qpair(3,7,s9).
qpair(3,9,s0y1).
qpair(3,9,s1y8).
qpair(3,9,s2).
qpair(3,9,s1y5).
qpair(4,5,s0).
qpair(4,6,s0y2).
qpair(4,6,s0y2y7y8).
qpair(4,7,s2).
qpair(4,7,s8).
qpair(4,7,s2y8).
qpair(4,8,s1y2).
qpair(4,8,s3).
qpair(4,9,s6).
qpair(4,9,s2).
qpair(4,9,s1y8).
qpair(4,9,s8).
qpair(4,9,s1).
qpair(4,9,s2y6).
qpair(4,9,s2y6y8).
qpair(4,9,s2y8).
qpair(5,8,s1).
qpair(5,8,s9).
qpair(6,7,s2y3).
qpair(6,7,s2y4).
qpair(6,8,s0y2y7).
qpair(6,8,s1y2).
qpair(6,8,s4y9).
qpair(6,8,s0y4y7).
qpair(6,8,s4y7).
qpair(6,9,s0y4y8).
qpair(6,9,s2).
qpair(6,9,s2y4y7y8).
qpair(6,9,s8).
qpair(6,9,s0y2y4y7y8).
qpair(6,9,s0y2).
qpair(6,9,s0y8).
qpair(6,9,s2y8).
qpair(6,9,s0y2y7).
qpair(6,9,s0y2y4).
qpair(6,9,s2y4y8).
qpair(6,9,s0y4y7).
qpair(6,9,s4).
qpair(6,9,s1).
qpair(6,9,s2y4).
qpair(6,9,s0y4).
qpair(6,9,s0y2y4y8).
qpair(6,9,s1y8).
qpair(6,9,s2y4y7).
qpair(6,9,s0).
qpair(6,9,s0y7y8).
qpair(6,9,s2y7y8).
qpair(6,9,s0y2y4y7).
qpair(6,9,s0y2y8).
qpair(6,9,s7y8).
qpair(7,8,s1y4y5y6).
qpair(7,8,s1y4).
qpair(7,8,s4).
qpair(7,8,s1y5y6).
qpair(7,8,s4y5).
qpair(7,8,s5y9).
qpair(7,8,s1y9).
qpair(7,9,empty).

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
in(0,s0y8y2y4).
in(0,s0y8y2y7).
in(0,s0y8y4).
in(0,s0y8y4y7).
in(0,s0y8y7).
in(1,s8y1).
in(1,s8y1y2).
in(1,s8y1y3).
in(2,s0y8y2).
in(2,s0y8y2y4).
in(2,s0y8y2y7).
in(2,s8y1y2).
in(2,s8y2).
in(2,s8y2y3).
in(2,s8y2y3y5).
in(2,s8y2y4).
in(2,s8y2y4y7).
in(2,s8y2y5).
in(2,s8y2y6).
in(2,s8y2y7).
in(2,s8y9y2).
in(2,s8y9y2y3).
in(2,s8y9y2y5).
in(2,s9y2).
in(2,s9y2y3).
in(2,s9y2y3y5).
in(2,s9y2y5).
in(3,s8y1y3).
in(3,s8y2y3).
in(3,s8y2y3y5).
in(3,s8y3).
in(3,s8y3y5).
in(3,s8y9y2y3).
in(3,s8y9y3).
in(3,s8y9y3y5).
in(3,s9y2y3).
in(3,s9y2y3y5).
in(3,s9y3).
in(3,s9y3y5).
in(4,s0y8y2y4).
in(4,s0y8y4).
in(4,s0y8y4y7).
in(4,s8y2y4).
in(4,s8y2y4y7).
in(4,s8y4).
in(4,s8y4y7).
in(4,s9y4).
in(4,s9y4y5).
in(4,s9y4y5y6).
in(4,s9y4y6).
in(5,s8y2y3y5).
in(5,s8y2y5).
in(5,s8y3y5).
in(5,s8y5).
in(5,s8y9y2y5).
in(5,s8y9y3y5).
in(5,s8y9y5).
in(5,s9y2y3y5).
in(5,s9y2y5).
in(5,s9y3y5).
in(5,s9y4y5).
in(5,s9y4y5y6).
in(5,s9y5).
in(5,s9y5y6).
in(6,s8y2y6).
in(6,s8y6).
in(6,s9y4y5y6).
in(6,s9y4y6).
in(6,s9y5y6).
in(6,s9y6).
in(7,s0y8y2y7).
in(7,s0y8y4y7).
in(7,s0y8y7).
in(7,s8y2y4y7).
in(7,s8y2y7).
in(7,s8y4y7).
in(7,s8y7).
in(8,s0y8y2).
in(8,s0y8y2y4).
in(8,s0y8y2y7).
in(8,s0y8y4).
in(8,s0y8y4y7).
in(8,s0y8y7).
in(8,s8y1).
in(8,s8y1y2).
in(8,s8y1y3).
in(8,s8y2).
in(8,s8y2y3).
in(8,s8y2y3y5).
in(8,s8y2y4).
in(8,s8y2y4y7).
in(8,s8y2y5).
in(8,s8y2y6).
in(8,s8y2y7).
in(8,s8y3).
in(8,s8y3y5).
in(8,s8y4).
in(8,s8y4y7).
in(8,s8y5).
in(8,s8y6).
in(8,s8y7).
in(8,s8y9y2).
in(8,s8y9y2y3).
in(8,s8y9y2y5).
in(8,s8y9y3).
in(8,s8y9y3y5).
in(8,s8y9y5).
in(9,s8y9y2).
in(9,s8y9y2y3).
in(9,s8y9y2y5).
in(9,s8y9y3).
in(9,s8y9y3y5).
in(9,s8y9y5).
in(9,s9y2).
in(9,s9y2y3).
in(9,s9y2y3y5).
in(9,s9y2y5).
in(9,s9y3).
in(9,s9y3y5).
in(9,s9y4).
in(9,s9y4y5).
in(9,s9y4y5y6).
in(9,s9y4y6).
in(9,s9y5).
in(9,s9y5y6).
in(9,s9y6).


% ===== Base program loaded from: results/wc_sweep_10_2026_20260212_123442/rep10/lps/abapc_inc_full_base_rep10_optN.lp =====
% ===== Base program sha256 (raw): 31db5447446e7ede871e58a6ece3b9792526d7d25eb1310ad1967675fca75122 =====
% ===== Base program sha256 (normalized): 9c9d774c66851c34eab4fa5924fb85e67229047f16feb296106e9dd682d86a18 =====
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

% ===== Adorned Facts =====
ext_dep(0,2,empty):-mus(1).
ext_dep(0,2,s1):-mus(2).
ext_dep(0,2,s3):-mus(3).
ext_dep(0,2,s4):-mus(4).
ext_dep(0,2,s5):-mus(5).
ext_dep(0,2,s6):-mus(6).
ext_dep(0,2,s8):-mus(7).
ext_dep(0,2,s9):-mus(8).
ext_dep(0,2,s1y3):-mus(9).
ext_dep(0,2,s1y4):-mus(10).
ext_dep(0,2,s1y5):-mus(11).
ext_dep(0,2,s1y6):-mus(12).
ext_dep(0,2,s8y1):-mus(13).
ext_dep(0,2,s1y9):-mus(14).
ext_indep(0,2,s3y4):-mus(15).
ext_dep(0,2,s3y5):-mus(16).
ext_dep(0,2,s3y6):-mus(17).
ext_dep(0,2,s8y3):-mus(18).
ext_dep(0,2,s9y3):-mus(19).
ext_dep(0,2,s4y5):-mus(20).
ext_dep(0,2,s4y6):-mus(21).
ext_dep(0,2,s8y4):-mus(22).
ext_dep(0,2,s9y4):-mus(23).
ext_dep(0,2,s5y6):-mus(24).
ext_dep(0,2,s8y5):-mus(25).
ext_dep(0,2,s9y5):-mus(26).
ext_dep(0,2,s8y6):-mus(27).
ext_dep(0,2,s9y6):-mus(28).
ext_indep(0,2,s8y9):-mus(29).
ext_dep(0,2,s1y4y5):-mus(30).
ext_indep(0,2,s1y4y6):-mus(31).
ext_dep(0,2,s1y5y6):-mus(32).
ext_dep(0,2,s4y5y6):-mus(33).
ext_dep(0,2,s1y4y5y6):-mus(34).
ext_indep(0,3,empty):-mus(35).
ext_dep(0,3,s1):-mus(36).
ext_dep(0,3,s2):-mus(37).
ext_dep(0,3,s4):-mus(38).
ext_dep(0,3,s5):-mus(39).
ext_dep(0,3,s6):-mus(40).
ext_dep(0,3,s7):-mus(41).
ext_indep(0,3,s9):-mus(42).
ext_dep(0,3,s1y2):-mus(43).
ext_dep(0,3,s1y5):-mus(44).
ext_indep(0,3,s1y6):-mus(45).
ext_dep(0,3,s2y5):-mus(46).
ext_dep(0,3,s2y6):-mus(47).
ext_dep(0,3,s5y6):-mus(48).
ext_dep(0,5,s4):-mus(49).
ext_dep(0,6,empty):-mus(50).
ext_dep(0,6,s1):-mus(51).
ext_dep(0,6,s2):-mus(52).
ext_dep(0,6,s3):-mus(53).
ext_dep(0,6,s4):-mus(54).
ext_dep(0,6,s5):-mus(55).
ext_dep(0,6,s7):-mus(56).
ext_dep(0,6,s8):-mus(57).
ext_dep(0,6,s1y2):-mus(58).
ext_dep(0,6,s1y3):-mus(59).
ext_dep(0,6,s2y3):-mus(60).
ext_dep(0,6,s2y4):-mus(61).
ext_dep(0,6,s2y7):-mus(62).
ext_dep(0,6,s8y2):-mus(63).
ext_indep(0,6,s3y4):-mus(64).
ext_dep(0,6,s3y7):-mus(65).
ext_dep(0,6,s8y3):-mus(66).
ext_indep(0,6,s4y7):-mus(67).
ext_dep(0,6,s8y4):-mus(68).
ext_dep(0,6,s8y7):-mus(69).
ext_dep(0,6,s2y4y7):-mus(70).
ext_dep(0,6,s8y2y4):-mus(71).
ext_indep(0,6,s8y2y7):-mus(72).
ext_dep(0,6,s8y4y7):-mus(73).
ext_dep(0,6,s8y2y4y7):-mus(74).
ext_dep(0,7,s2):-mus(75).
ext_indep(0,7,s3):-mus(76).
ext_dep(0,7,s6):-mus(77).
ext_indep(0,7,s2y3):-mus(78).
ext_dep(0,7,s2y6):-mus(79).
ext_dep(0,7,s3y6):-mus(80).
ext_dep(0,7,s2y3y6):-mus(81).
ext_indep(0,8,s1):-mus(82).
ext_indep(0,8,s2):-mus(83).
ext_indep(0,8,s3):-mus(84).
ext_indep(0,8,s4):-mus(85).
ext_indep(0,8,s5):-mus(86).
ext_dep(0,8,s6):-mus(87).
ext_indep(0,8,s9):-mus(88).
ext_indep(0,8,s1y4):-mus(89).
ext_indep(0,8,s1y5):-mus(90).
ext_dep(0,8,s1y6):-mus(91).
ext_indep(0,8,s1y9):-mus(92).
ext_indep(0,8,s2y3):-mus(93).
ext_dep(0,8,s2y6):-mus(94).
ext_dep(0,8,s3y6):-mus(95).
ext_indep(0,8,s4y5):-mus(96).
ext_dep(0,8,s4y6):-mus(97).
ext_indep(0,8,s9y4):-mus(98).
ext_dep(0,8,s5y6):-mus(99).
ext_indep(0,8,s9y5):-mus(100).
ext_dep(0,8,s9y6):-mus(101).
ext_indep(0,8,s1y4y5):-mus(102).
ext_dep(0,8,s1y4y6):-mus(103).
ext_indep(0,8,s1y4y9):-mus(104).
ext_dep(0,8,s1y5y6):-mus(105).
ext_dep(0,8,s1y5y9):-mus(106).
ext_dep(0,8,s1y6y9):-mus(107).
ext_dep(0,8,s2y3y6):-mus(108).
ext_dep(0,8,s4y5y6):-mus(109).
ext_indep(0,8,s9y4y5):-mus(110).
ext_dep(0,8,s9y4y6):-mus(111).
ext_dep(0,8,s9y5y6):-mus(112).
ext_dep(0,8,s1y4y5y6):-mus(113).
ext_indep(0,8,s1y4y5y9):-mus(114).
ext_indep(0,8,s1y4y6y9):-mus(115).
ext_dep(0,8,s1y5y6y9):-mus(116).
ext_dep(0,8,s9y4y5y6):-mus(117).
ext_dep(0,8,s1y4y5y6y9):-mus(118).
ext_indep(0,9,empty):-mus(119).
ext_dep(1,2,empty):-mus(120).
ext_dep(1,2,s0):-mus(121).
ext_dep(1,2,s3):-mus(122).
ext_indep(1,2,s4):-mus(123).
ext_indep(1,2,s5):-mus(124).
ext_dep(1,2,s6):-mus(125).
ext_dep(1,2,s8):-mus(126).
ext_dep(1,2,s9):-mus(127).
ext_dep(1,2,s0y3):-mus(128).
ext_dep(1,2,s0y4):-mus(129).
ext_dep(1,2,s0y5):-mus(130).
ext_dep(1,2,s0y6):-mus(131).
ext_dep(1,2,s0y8):-mus(132).
ext_dep(1,2,s0y9):-mus(133).
ext_dep(1,2,s3y4):-mus(134).
ext_dep(1,2,s3y5):-mus(135).
ext_dep(1,2,s3y6):-mus(136).
ext_dep(1,2,s8y3):-mus(137).
ext_dep(1,2,s9y3):-mus(138).
ext_dep(1,2,s4y5):-mus(139).
ext_dep(1,2,s4y6):-mus(140).
ext_dep(1,2,s8y4):-mus(141).
ext_dep(1,2,s9y4):-mus(142).
ext_dep(1,2,s5y6):-mus(143).
ext_dep(1,2,s8y5):-mus(144).
ext_dep(1,2,s9y5):-mus(145).
ext_dep(1,2,s8y6):-mus(146).
ext_dep(1,2,s9y6):-mus(147).
ext_indep(1,2,s8y9):-mus(148).
ext_dep(1,2,s0y4y5):-mus(149).
ext_indep(1,2,s0y4y6):-mus(150).
ext_dep(1,2,s0y5y6):-mus(151).
ext_dep(1,2,s8y3y5):-mus(152).
ext_dep(1,2,s9y3y5):-mus(153).
ext_dep(1,2,s8y9y3):-mus(154).
ext_dep(1,2,s4y5y6):-mus(155).
ext_dep(1,2,s8y9y5):-mus(156).
ext_dep(1,2,s0y4y5y6):-mus(157).
ext_dep(1,2,s8y9y3y5):-mus(158).
ext_dep(1,3,empty):-mus(159).
ext_dep(1,3,s0):-mus(160).
ext_dep(1,3,s2):-mus(161).
ext_dep(1,3,s4):-mus(162).
ext_dep(1,3,s5):-mus(163).
ext_dep(1,3,s6):-mus(164).
ext_dep(1,3,s7):-mus(165).
ext_dep(1,3,s8):-mus(166).
ext_dep(1,3,s9):-mus(167).
ext_dep(1,3,s0y2):-mus(168).
ext_dep(1,3,s0y5):-mus(169).
ext_dep(1,3,s0y6):-mus(170).
ext_dep(1,3,s0y8):-mus(171).
ext_dep(1,3,s0y9):-mus(172).
ext_dep(1,3,s2y5):-mus(173).
ext_indep(1,3,s2y6):-mus(174).
ext_dep(1,3,s8y2):-mus(175).
ext_dep(1,3,s9y2):-mus(176).
ext_dep(1,3,s5y6):-mus(177).
ext_dep(1,3,s8y5):-mus(178).
ext_dep(1,3,s9y5):-mus(179).
ext_dep(1,3,s8y9):-mus(180).
ext_dep(1,3,s8y2y5):-mus(181).
ext_dep(1,3,s9y2y5):-mus(182).
ext_dep(1,3,s8y9y2):-mus(183).
ext_dep(1,3,s8y9y5):-mus(184).
ext_dep(1,3,s8y9y2y5):-mus(185).
ext_dep(1,5,empty):-mus(186).
ext_dep(1,5,s0):-mus(187).
ext_dep(1,5,s2):-mus(188).
ext_dep(1,5,s3):-mus(189).
ext_dep(1,5,s4):-mus(190).
ext_dep(1,5,s6):-mus(191).
ext_dep(1,5,s8):-mus(192).
ext_indep(1,5,s9):-mus(193).
ext_dep(1,5,s0y2):-mus(194).
ext_indep(1,5,s0y3):-mus(195).
ext_dep(1,5,s0y8):-mus(196).
ext_dep(1,5,s0y9):-mus(197).
ext_dep(1,5,s2y3):-mus(198).
ext_dep(1,5,s8y2):-mus(199).
ext_indep(1,5,s9y2):-mus(200).
ext_dep(1,5,s8y3):-mus(201).
ext_dep(1,5,s9y3):-mus(202).
ext_dep(1,5,s8y9):-mus(203).
ext_dep(1,5,s8y2y3):-mus(204).
ext_dep(1,5,s9y2y3):-mus(205).
ext_dep(1,5,s8y9y2):-mus(206).
ext_dep(1,5,s8y9y3):-mus(207).
ext_dep(1,5,s8y9y2y3):-mus(208).
ext_dep(1,7,empty):-mus(209).
ext_dep(1,8,empty):-mus(210).
ext_dep(1,8,s0):-mus(211).
ext_dep(1,8,s2):-mus(212).
ext_dep(1,8,s3):-mus(213).
ext_dep(1,8,s4):-mus(214).
ext_dep(1,8,s5):-mus(215).
ext_dep(1,8,s6):-mus(216).
ext_dep(1,8,s9):-mus(217).
ext_dep(1,8,s0y2):-mus(218).
ext_dep(1,8,s0y3):-mus(219).
ext_dep(1,8,s0y5):-mus(220).
ext_dep(1,8,s0y9):-mus(221).
ext_dep(1,8,s2y3):-mus(222).
ext_dep(1,8,s2y4):-mus(223).
ext_dep(1,8,s2y5):-mus(224).
ext_dep(1,8,s2y6):-mus(225).
ext_dep(1,8,s9y2):-mus(226).
ext_dep(1,8,s3y5):-mus(227).
ext_dep(1,8,s9y3):-mus(228).
ext_dep(1,8,s4y5):-mus(229).
ext_indep(1,8,s4y6):-mus(230).
ext_dep(1,8,s9y4):-mus(231).
ext_dep(1,8,s5y6):-mus(232).
ext_dep(1,8,s9y5):-mus(233).
ext_dep(1,8,s9y6):-mus(234).
ext_dep(1,8,s2y3y5):-mus(235).
ext_dep(1,8,s9y2y3):-mus(236).
ext_indep(1,8,s9y2y5):-mus(237).
ext_dep(1,8,s9y3y5):-mus(238).
ext_indep(1,8,s4y5y6):-mus(239).
ext_indep(1,8,s9y4y5):-mus(240).
ext_dep(1,8,s9y4y6):-mus(241).
ext_dep(1,8,s9y5y6):-mus(242).
ext_dep(1,8,s9y2y3y5):-mus(243).
ext_dep(1,8,s9y4y5y6):-mus(244).
ext_dep(1,9,empty):-mus(245).
ext_dep(1,9,s0):-mus(246).
ext_indep(1,9,s2):-mus(247).
ext_dep(1,9,s3):-mus(248).
ext_dep(1,9,s5):-mus(249).
ext_dep(1,9,s8):-mus(250).
ext_dep(1,9,s0y2):-mus(251).
ext_dep(1,9,s0y3):-mus(252).
ext_dep(1,9,s0y5):-mus(253).
ext_dep(1,9,s0y8):-mus(254).
ext_dep(1,9,s2y3):-mus(255).
ext_dep(1,9,s2y5):-mus(256).
ext_dep(1,9,s8y2):-mus(257).
ext_dep(1,9,s3y5):-mus(258).
ext_dep(1,9,s8y3):-mus(259).
ext_dep(1,9,s8y5):-mus(260).
ext_dep(1,9,s2y3y5):-mus(261).
ext_dep(1,9,s8y2y3):-mus(262).
ext_indep(1,9,s8y2y5):-mus(263).
ext_dep(1,9,s8y3y5):-mus(264).
ext_dep(1,9,s8y2y3y5):-mus(265).
ext_dep(2,3,s7):-mus(266).
ext_dep(2,3,s8):-mus(267).
ext_dep(2,3,s9):-mus(268).
ext_dep(2,3,s0y8):-mus(269).
ext_dep(2,3,s0y9):-mus(270).
ext_dep(2,3,s8y1):-mus(271).
ext_indep(2,3,s1y9):-mus(272).
ext_dep(2,3,s8y4):-mus(273).
ext_dep(2,3,s9y4):-mus(274).
ext_dep(2,3,s8y5):-mus(275).
ext_dep(2,3,s9y5):-mus(276).
ext_indep(2,3,s8y6):-mus(277).
ext_dep(2,3,s9y6):-mus(278).
ext_indep(2,3,s8y9):-mus(279).
ext_dep(2,4,empty):-mus(280).
ext_dep(2,4,s0):-mus(281).
ext_dep(2,4,s1):-mus(282).
ext_indep(2,4,s3):-mus(283).
ext_dep(2,4,s5):-mus(284).
ext_dep(2,4,s6):-mus(285).
ext_dep(2,4,s8):-mus(286).
ext_dep(2,4,s9):-mus(287).
ext_dep(2,4,s0y1):-mus(288).
ext_dep(2,4,s0y3):-mus(289).
ext_indep(2,4,s0y5):-mus(290).
ext_dep(2,4,s0y6):-mus(291).
ext_dep(2,4,s0y8):-mus(292).
ext_dep(2,4,s0y9):-mus(293).
ext_dep(2,4,s1y3):-mus(294).
ext_indep(2,4,s1y5):-mus(295).
ext_dep(2,4,s1y6):-mus(296).
ext_dep(2,4,s8y1):-mus(297).
ext_dep(2,4,s1y9):-mus(298).
ext_dep(2,4,s3y5):-mus(299).
ext_dep(2,4,s3y6):-mus(300).
ext_dep(2,4,s8y3):-mus(301).
ext_dep(2,4,s9y3):-mus(302).
ext_dep(2,4,s5y6):-mus(303).
ext_dep(2,4,s8y5):-mus(304).
ext_dep(2,4,s9y5):-mus(305).
ext_dep(2,4,s8y6):-mus(306).
ext_dep(2,4,s9y6):-mus(307).
ext_dep(2,4,s8y9):-mus(308).
ext_dep(2,4,s0y1y5):-mus(309).
ext_dep(2,4,s0y1y6):-mus(310).
ext_indep(2,4,s0y5y6):-mus(311).
ext_dep(2,4,s1y5y6):-mus(312).
ext_dep(2,4,s0y1y5y6):-mus(313).
ext_dep(2,5,empty):-mus(314).
ext_dep(2,5,s0):-mus(315).
ext_indep(2,5,s1):-mus(316).
ext_dep(2,5,s3):-mus(317).
ext_dep(2,5,s4):-mus(318).
ext_dep(2,5,s6):-mus(319).
ext_indep(2,5,s8):-mus(320).
ext_dep(2,5,s9):-mus(321).
ext_dep(2,5,s0y1):-mus(322).
ext_dep(2,5,s0y3):-mus(323).
ext_dep(2,5,s0y4):-mus(324).
ext_dep(2,5,s0y6):-mus(325).
ext_dep(2,5,s0y8):-mus(326).
ext_indep(2,5,s0y9):-mus(327).
ext_dep(2,5,s1y3):-mus(328).
ext_dep(2,5,s1y4):-mus(329).
ext_dep(2,5,s1y6):-mus(330).
ext_dep(2,5,s8y1):-mus(331).
ext_dep(2,5,s1y9):-mus(332).
ext_dep(2,5,s3y4):-mus(333).
ext_indep(2,5,s3y6):-mus(334).
ext_dep(2,5,s8y3):-mus(335).
ext_indep(2,5,s9y3):-mus(336).
ext_dep(2,5,s4y6):-mus(337).
ext_dep(2,5,s8y4):-mus(338).
ext_dep(2,5,s9y4):-mus(339).
ext_dep(2,5,s8y6):-mus(340).
ext_dep(2,5,s9y6):-mus(341).
ext_indep(2,5,s8y9):-mus(342).
ext_dep(2,5,s0y1y4):-mus(343).
ext_dep(2,5,s0y1y6):-mus(344).
ext_dep(2,5,s0y4y6):-mus(345).
ext_dep(2,5,s8y1y3):-mus(346).
ext_indep(2,5,s1y4y6):-mus(347).
ext_dep(2,5,s0y1y4y6):-mus(348).
ext_dep(2,6,empty):-mus(349).
ext_dep(2,6,s0):-mus(350).
ext_dep(2,6,s1):-mus(351).
ext_dep(2,6,s3):-mus(352).
ext_dep(2,6,s4):-mus(353).
ext_dep(2,6,s5):-mus(354).
ext_dep(2,6,s7):-mus(355).
ext_dep(2,6,s8):-mus(356).
ext_dep(2,6,s9):-mus(357).
ext_dep(2,6,s0y1):-mus(358).
ext_dep(2,6,s0y3):-mus(359).
ext_dep(2,6,s0y4):-mus(360).
ext_dep(2,6,s0y5):-mus(361).
ext_dep(2,6,s0y7):-mus(362).
ext_dep(2,6,s0y8):-mus(363).
ext_dep(2,6,s0y9):-mus(364).
ext_dep(2,6,s1y3):-mus(365).
ext_dep(2,6,s1y4):-mus(366).
ext_dep(2,6,s1y5):-mus(367).
ext_dep(2,6,s8y1):-mus(368).
ext_indep(2,6,s1y9):-mus(369).
ext_dep(2,6,s3y4):-mus(370).
ext_dep(2,6,s3y5):-mus(371).
ext_dep(2,6,s3y7):-mus(372).
ext_dep(2,6,s8y3):-mus(373).
ext_dep(2,6,s9y3):-mus(374).
ext_dep(2,6,s4y5):-mus(375).
ext_dep(2,6,s4y7):-mus(376).
ext_dep(2,6,s8y4):-mus(377).
ext_dep(2,6,s9y4):-mus(378).
ext_dep(2,6,s8y5):-mus(379).
ext_dep(2,6,s9y5):-mus(380).
ext_dep(2,6,s8y7):-mus(381).
ext_dep(2,6,s8y9):-mus(382).
ext_dep(2,6,s0y1y4):-mus(383).
ext_dep(2,6,s0y1y5):-mus(384).
ext_dep(2,6,s0y4y5):-mus(385).
ext_dep(2,6,s0y4y7):-mus(386).
ext_dep(2,6,s0y8y4):-mus(387).
ext_dep(2,6,s0y8y7):-mus(388).
ext_dep(2,6,s1y4y5):-mus(389).
ext_dep(2,6,s8y4y7):-mus(390).
ext_dep(2,6,s0y1y4y5):-mus(391).
ext_indep(2,6,s0y8y4y7):-mus(392).
ext_indep(2,7,s0):-mus(393).
ext_dep(2,7,s1):-mus(394).
ext_indep(2,7,s4):-mus(395).
ext_indep(2,7,s5):-mus(396).
ext_dep(2,7,s6):-mus(397).
ext_indep(2,7,s0y1):-mus(398).
ext_indep(2,7,s0y4):-mus(399).
ext_indep(2,7,s0y5):-mus(400).
ext_dep(2,7,s0y6):-mus(401).
ext_dep(2,7,s1y4):-mus(402).
ext_dep(2,7,s1y5):-mus(403).
ext_dep(2,7,s1y6):-mus(404).
ext_dep(2,7,s4y5):-mus(405).
ext_dep(2,7,s4y6):-mus(406).
ext_dep(2,7,s5y6):-mus(407).
ext_dep(2,7,s0y1y4):-mus(408).
ext_indep(2,7,s0y1y5):-mus(409).
ext_dep(2,7,s0y1y6):-mus(410).
ext_indep(2,7,s0y4y5):-mus(411).
ext_dep(2,7,s0y4y6):-mus(412).
ext_dep(2,7,s0y5y6):-mus(413).
ext_dep(2,7,s1y4y5):-mus(414).
ext_dep(2,7,s1y4y6):-mus(415).
ext_dep(2,7,s1y5y6):-mus(416).
ext_dep(2,7,s4y5y6):-mus(417).
ext_dep(2,7,s0y1y4y5):-mus(418).
ext_dep(2,7,s0y1y4y6):-mus(419).
ext_dep(2,7,s0y1y5y6):-mus(420).
ext_dep(2,7,s0y4y5y6):-mus(421).
ext_dep(2,7,s1y4y5y6):-mus(422).
ext_dep(2,7,s0y1y4y5y6):-mus(423).
ext_dep(2,8,s3):-mus(424).
ext_indep(2,8,s0y3):-mus(425).
ext_dep(2,8,s0y9):-mus(426).
ext_dep(2,8,s1y3):-mus(427).
ext_dep(2,8,s3y4):-mus(428).
ext_dep(2,8,s3y5):-mus(429).
ext_dep(2,8,s3y6):-mus(430).
ext_dep(2,8,s9y3):-mus(431).
ext_dep(2,9,s3):-mus(432).
ext_dep(2,9,s0y3):-mus(433).
ext_dep(2,9,s0y8):-mus(434).
ext_dep(2,9,s1y3):-mus(435).
ext_dep(2,9,s3y4):-mus(436).
ext_indep(2,9,s3y5):-mus(437).
ext_dep(2,9,s3y6):-mus(438).
ext_dep(2,9,s8y3):-mus(439).
ext_dep(2,9,s8y4):-mus(440).
ext_dep(2,9,s8y5):-mus(441).
ext_dep(2,9,s8y6):-mus(442).
ext_dep(2,9,s0y1y4):-mus(443).
ext_indep(2,9,s0y1y5):-mus(444).
ext_dep(2,9,s0y1y6):-mus(445).
ext_dep(2,9,s0y4y5):-mus(446).
ext_indep(2,9,s0y4y6):-mus(447).
ext_dep(2,9,s0y5y6):-mus(448).
ext_indep(2,9,s1y4y5):-mus(449).
ext_dep(2,9,s1y4y6):-mus(450).
ext_indep(2,9,s1y5y6):-mus(451).
ext_indep(2,9,s4y5y6):-mus(452).
ext_indep(2,9,s0y1y4y5):-mus(453).
ext_dep(2,9,s0y1y4y6):-mus(454).
ext_indep(2,9,s0y1y5y6):-mus(455).
ext_indep(2,9,s0y4y5y6):-mus(456).
ext_dep(2,9,s1y4y5y6):-mus(457).
ext_indep(2,9,s0y1y4y5y6):-mus(458).
ext_dep(3,4,empty):-mus(459).
ext_dep(3,4,s0):-mus(460).
ext_dep(3,4,s1):-mus(461).
ext_indep(3,4,s2):-mus(462).
ext_indep(3,4,s5):-mus(463).
ext_dep(3,4,s6):-mus(464).
ext_indep(3,4,s7):-mus(465).
ext_indep(3,4,s8):-mus(466).
ext_dep(3,4,s9):-mus(467).
ext_indep(3,5,empty):-mus(468).
ext_dep(3,5,s0):-mus(469).
ext_dep(3,5,s1):-mus(470).
ext_dep(3,5,s2):-mus(471).
ext_dep(3,5,s4):-mus(472).
ext_dep(3,5,s6):-mus(473).
ext_dep(3,5,s7):-mus(474).
ext_dep(3,5,s8):-mus(475).
ext_dep(3,5,s9):-mus(476).
ext_dep(3,5,s0y1):-mus(477).
ext_dep(3,5,s0y2):-mus(478).
ext_dep(3,5,s0y6):-mus(479).
ext_dep(3,5,s1y2):-mus(480).
ext_dep(3,5,s1y6):-mus(481).
ext_dep(3,5,s8y1):-mus(482).
ext_dep(3,5,s2y6):-mus(483).
ext_dep(3,5,s8y2):-mus(484).
ext_dep(3,5,s8y1y2):-mus(485).
ext_dep(3,6,s9):-mus(486).
ext_dep(3,6,s1y2):-mus(487).
ext_dep(3,6,s2y5):-mus(488).
ext_indep(3,6,s0y1y5):-mus(489).
ext_indep(3,6,s0y2y4):-mus(490).
ext_indep(3,6,s0y2y7):-mus(491).
ext_indep(3,6,s0y8y2):-mus(492).
ext_dep(3,6,s0y4y7):-mus(493).
ext_indep(3,6,s0y8y4):-mus(494).
ext_dep(3,6,s0y8y7):-mus(495).
ext_dep(3,6,s2y4y7):-mus(496).
ext_dep(3,6,s8y2y4):-mus(497).
ext_dep(3,6,s8y2y7):-mus(498).
ext_dep(3,6,s8y4y7):-mus(499).
ext_indep(3,6,s0y2y4y7):-mus(500).
ext_indep(3,6,s0y8y2y4):-mus(501).
ext_indep(3,6,s0y8y2y7):-mus(502).
ext_dep(3,6,s0y8y4y7):-mus(503).
ext_dep(3,6,s8y2y4y7):-mus(504).
ext_indep(3,6,s0y2y4y7y8):-mus(505).
ext_dep(3,7,empty):-mus(506).
ext_indep(3,7,s0):-mus(507).
ext_dep(3,7,s1):-mus(508).
ext_indep(3,7,s2):-mus(509).
ext_indep(3,7,s4):-mus(510).
ext_indep(3,7,s5):-mus(511).
ext_dep(3,7,s6):-mus(512).
ext_indep(3,7,s9):-mus(513).
ext_indep(3,9,s2):-mus(514).
ext_dep(3,9,s4):-mus(515).
ext_dep(3,9,s6):-mus(516).
ext_dep(3,9,s7):-mus(517).
ext_indep(3,9,s0y1):-mus(518).
ext_dep(3,9,s0y5):-mus(519).
ext_indep(3,9,s1y5):-mus(520).
ext_indep(3,9,s8y1):-mus(521).
ext_dep(3,9,s0y1y5):-mus(522).
ext_indep(4,5,s0):-mus(523).
ext_dep(4,6,empty):-mus(524).
ext_dep(4,6,s0):-mus(525).
ext_dep(4,6,s2):-mus(526).
ext_dep(4,6,s3):-mus(527).
ext_dep(4,6,s5):-mus(528).
ext_dep(4,6,s7):-mus(529).
ext_dep(4,6,s8):-mus(530).
ext_indep(4,6,s0y2):-mus(531).
ext_dep(4,6,s0y3):-mus(532).
ext_dep(4,6,s0y7):-mus(533).
ext_dep(4,6,s0y8):-mus(534).
ext_dep(4,6,s2y3):-mus(535).
ext_dep(4,6,s2y7):-mus(536).
ext_dep(4,6,s8y2):-mus(537).
ext_dep(4,6,s3y7):-mus(538).
ext_dep(4,6,s8y3):-mus(539).
ext_dep(4,6,s8y7):-mus(540).
ext_dep(4,6,s0y2y7):-mus(541).
ext_dep(4,6,s0y8y2):-mus(542).
ext_dep(4,6,s0y8y7):-mus(543).
ext_dep(4,6,s8y2y7):-mus(544).
ext_indep(4,6,s0y8y2y7):-mus(545).
ext_indep(4,7,s2):-mus(546).
ext_dep(4,7,s6):-mus(547).
ext_indep(4,7,s8):-mus(548).
ext_dep(4,7,s2y6):-mus(549).
ext_indep(4,7,s8y2):-mus(550).
ext_dep(4,7,s8y6):-mus(551).
ext_dep(4,7,s8y2y6):-mus(552).
ext_dep(4,8,empty):-mus(553).
ext_dep(4,8,s1):-mus(554).
ext_dep(4,8,s2):-mus(555).
ext_indep(4,8,s3):-mus(556).
ext_dep(4,8,s5):-mus(557).
ext_dep(4,8,s6):-mus(558).
ext_dep(4,8,s9):-mus(559).
ext_indep(4,8,s1y2):-mus(560).
ext_dep(4,8,s1y5):-mus(561).
ext_dep(4,8,s1y6):-mus(562).
ext_dep(4,8,s1y9):-mus(563).
ext_dep(4,8,s2y5):-mus(564).
ext_dep(4,8,s2y6):-mus(565).
ext_dep(4,8,s9y2):-mus(566).
ext_dep(4,8,s5y6):-mus(567).
ext_dep(4,8,s9y5):-mus(568).
ext_dep(4,8,s9y6):-mus(569).
ext_dep(4,8,s1y5y6):-mus(570).
ext_dep(4,8,s1y5y9):-mus(571).
ext_dep(4,8,s1y6y9):-mus(572).
ext_dep(4,8,s9y5y6):-mus(573).
ext_dep(4,8,s1y5y6y9):-mus(574).
ext_indep(4,9,s1):-mus(575).
ext_indep(4,9,s2):-mus(576).
ext_indep(4,9,s6):-mus(577).
ext_indep(4,9,s8):-mus(578).
ext_indep(4,9,s8y1):-mus(579).
ext_indep(4,9,s2y6):-mus(580).
ext_indep(4,9,s8y2):-mus(581).
ext_dep(4,9,s8y6):-mus(582).
ext_indep(4,9,s8y2y6):-mus(583).
ext_dep(5,7,empty):-mus(584).
ext_dep(5,8,empty):-mus(585).
ext_dep(5,8,s0):-mus(586).
ext_indep(5,8,s1):-mus(587).
ext_dep(5,8,s2):-mus(588).
ext_dep(5,8,s3):-mus(589).
ext_dep(5,8,s4):-mus(590).
ext_dep(5,8,s6):-mus(591).
ext_indep(5,8,s9):-mus(592).
ext_dep(5,8,s1y2):-mus(593).
ext_dep(5,8,s1y3):-mus(594).
ext_dep(5,8,s1y4):-mus(595).
ext_dep(5,8,s1y6):-mus(596).
ext_dep(5,8,s1y9):-mus(597).
ext_dep(5,8,s2y3):-mus(598).
ext_dep(5,8,s2y4):-mus(599).
ext_dep(5,8,s2y6):-mus(600).
ext_dep(5,8,s9y2):-mus(601).
ext_dep(5,8,s4y6):-mus(602).
ext_dep(5,8,s9y4):-mus(603).
ext_dep(5,8,s9y6):-mus(604).
ext_dep(5,8,s1y2y3):-mus(605).
ext_dep(5,8,s1y4y6):-mus(606).
ext_dep(5,8,s1y4y9):-mus(607).
ext_dep(5,8,s1y6y9):-mus(608).
ext_dep(5,8,s9y4y6):-mus(609).
ext_dep(5,8,s1y4y6y9):-mus(610).
ext_dep(6,7,empty):-mus(611).
ext_dep(6,7,s0):-mus(612).
ext_dep(6,7,s2):-mus(613).
ext_dep(6,7,s3):-mus(614).
ext_dep(6,7,s4):-mus(615).
ext_dep(6,7,s5):-mus(616).
ext_dep(6,7,s8):-mus(617).
ext_dep(6,7,s0y2):-mus(618).
ext_dep(6,7,s0y3):-mus(619).
ext_dep(6,7,s0y4):-mus(620).
ext_dep(6,7,s0y8):-mus(621).
ext_indep(6,7,s2y3):-mus(622).
ext_indep(6,7,s2y4):-mus(623).
ext_dep(6,7,s8y2):-mus(624).
ext_dep(6,7,s3y4):-mus(625).
ext_dep(6,7,s8y3):-mus(626).
ext_dep(6,7,s8y4):-mus(627).
ext_dep(6,7,s0y2y4):-mus(628).
ext_dep(6,7,s0y8y2):-mus(629).
ext_dep(6,7,s0y8y4):-mus(630).
ext_dep(6,7,s8y2y4):-mus(631).
ext_dep(6,7,s0y8y2y4):-mus(632).
ext_dep(6,8,empty):-mus(633).
ext_dep(6,8,s0):-mus(634).
ext_dep(6,8,s1):-mus(635).
ext_dep(6,8,s2):-mus(636).
ext_dep(6,8,s3):-mus(637).
ext_dep(6,8,s4):-mus(638).
ext_dep(6,8,s5):-mus(639).
ext_dep(6,8,s7):-mus(640).
ext_dep(6,8,s9):-mus(641).
ext_dep(6,8,s0y2):-mus(642).
ext_dep(6,8,s0y3):-mus(643).
ext_dep(6,8,s0y4):-mus(644).
ext_dep(6,8,s0y7):-mus(645).
ext_indep(6,8,s1y2):-mus(646).
ext_dep(6,8,s1y4):-mus(647).
ext_dep(6,8,s1y5):-mus(648).
ext_dep(6,8,s1y9):-mus(649).
ext_dep(6,8,s2y3):-mus(650).
ext_dep(6,8,s2y4):-mus(651).
ext_dep(6,8,s2y5):-mus(652).
ext_dep(6,8,s2y7):-mus(653).
ext_dep(6,8,s9y2):-mus(654).
ext_dep(6,8,s3y4):-mus(655).
ext_dep(6,8,s3y7):-mus(656).
ext_dep(6,8,s4y5):-mus(657).
ext_indep(6,8,s4y7):-mus(658).
ext_indep(6,8,s9y4):-mus(659).
ext_dep(6,8,s9y5):-mus(660).
ext_dep(6,8,s0y2y4):-mus(661).
ext_indep(6,8,s0y2y7):-mus(662).
ext_indep(6,8,s0y4y7):-mus(663).
ext_dep(6,8,s1y4y5):-mus(664).
ext_dep(6,8,s1y4y9):-mus(665).
ext_dep(6,8,s1y5y9):-mus(666).
ext_dep(6,8,s2y4y7):-mus(667).
ext_dep(6,8,s9y4y5):-mus(668).
ext_dep(6,8,s0y2y4y7):-mus(669).
ext_dep(6,8,s1y4y5y9):-mus(670).
ext_indep(6,9,s0):-mus(671).
ext_indep(6,9,s1):-mus(672).
ext_indep(6,9,s2):-mus(673).
ext_indep(6,9,s4):-mus(674).
ext_dep(6,9,s7):-mus(675).
ext_indep(6,9,s8):-mus(676).
ext_indep(6,9,s0y2):-mus(677).
ext_indep(6,9,s0y4):-mus(678).
ext_dep(6,9,s0y7):-mus(679).
ext_indep(6,9,s0y8):-mus(680).
ext_indep(6,9,s8y1):-mus(681).
ext_indep(6,9,s2y4):-mus(682).
ext_dep(6,9,s2y7):-mus(683).
ext_indep(6,9,s8y2):-mus(684).
ext_dep(6,9,s4y7):-mus(685).
ext_dep(6,9,s8y4):-mus(686).
ext_indep(6,9,s8y7):-mus(687).
ext_indep(6,9,s0y2y4):-mus(688).
ext_indep(6,9,s0y2y7):-mus(689).
ext_indep(6,9,s0y8y2):-mus(690).
ext_indep(6,9,s0y4y7):-mus(691).
ext_indep(6,9,s0y8y4):-mus(692).
ext_indep(6,9,s0y8y7):-mus(693).
ext_indep(6,9,s2y4y7):-mus(694).
ext_indep(6,9,s8y2y4):-mus(695).
ext_indep(6,9,s8y2y7):-mus(696).
ext_dep(6,9,s8y4y7):-mus(697).
ext_indep(6,9,s0y2y4y7):-mus(698).
ext_indep(6,9,s0y8y2y4):-mus(699).
ext_dep(6,9,s0y8y2y7):-mus(700).
ext_dep(6,9,s0y8y4y7):-mus(701).
ext_indep(6,9,s8y2y4y7):-mus(702).
ext_indep(6,9,s0y2y4y7y8):-mus(703).
ext_dep(7,8,s1):-mus(704).
ext_indep(7,8,s4):-mus(705).
ext_dep(7,8,s5):-mus(706).
ext_dep(7,8,s6):-mus(707).
ext_dep(7,8,s9):-mus(708).
ext_indep(7,8,s1y4):-mus(709).
ext_dep(7,8,s1y5):-mus(710).
ext_dep(7,8,s1y6):-mus(711).
ext_indep(7,8,s1y9):-mus(712).
ext_indep(7,8,s4y5):-mus(713).
ext_dep(7,8,s4y6):-mus(714).
ext_dep(7,8,s9y4):-mus(715).
ext_dep(7,8,s5y6):-mus(716).
ext_indep(7,8,s9y5):-mus(717).
ext_dep(7,8,s9y6):-mus(718).
ext_dep(7,8,s1y4y5):-mus(719).
ext_dep(7,8,s1y4y6):-mus(720).
ext_dep(7,8,s1y4y9):-mus(721).
ext_indep(7,8,s1y5y6):-mus(722).
ext_dep(7,8,s1y5y9):-mus(723).
ext_dep(7,8,s1y6y9):-mus(724).
ext_dep(7,8,s4y5y6):-mus(725).
ext_dep(7,8,s9y4y5):-mus(726).
ext_dep(7,8,s9y4y6):-mus(727).
ext_dep(7,8,s9y5y6):-mus(728).
ext_indep(7,8,s1y4y5y6):-mus(729).
ext_dep(7,8,s1y4y5y9):-mus(730).
ext_dep(7,8,s1y4y6y9):-mus(731).
ext_dep(7,8,s1y5y6y9):-mus(732).
ext_dep(7,8,s9y4y5y6):-mus(733).
ext_dep(7,8,s1y4y5y6y9):-mus(734).
ext_indep(7,9,empty):-mus(735).
ext_dep(8,9,empty):-mus(736).
ext_dep(8,9,s1):-mus(737).
ext_dep(8,9,s2):-mus(738).
ext_dep(8,9,s3):-mus(739).
ext_dep(8,9,s4):-mus(740).
ext_dep(8,9,s5):-mus(741).
ext_dep(8,9,s6):-mus(742).
ext_dep(8,9,s1y2):-mus(743).
ext_dep(8,9,s1y4):-mus(744).
ext_dep(8,9,s1y5):-mus(745).
ext_dep(8,9,s1y6):-mus(746).
ext_dep(8,9,s2y4):-mus(747).
ext_dep(8,9,s2y5):-mus(748).
ext_dep(8,9,s2y6):-mus(749).
ext_dep(8,9,s4y5):-mus(750).
ext_dep(8,9,s4y6):-mus(751).
ext_dep(8,9,s5y6):-mus(752).
ext_dep(8,9,s1y4y5):-mus(753).
ext_dep(8,9,s1y4y6):-mus(754).
ext_dep(8,9,s1y5y6):-mus(755).
ext_dep(8,9,s4y5y6):-mus(756).
ext_dep(8,9,s1y4y5y6):-mus(757).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(9935065, mus(1)).
__optimum_mcs_objective_literal__(9998912, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(9710138, mus(4)).
__optimum_mcs_objective_literal__(9806242, mus(5)).
__optimum_mcs_objective_literal__(9999999, mus(6)).
__optimum_mcs_objective_literal__(9797089, mus(7)).
__optimum_mcs_objective_literal__(9956452, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(9988131, mus(10)).
__optimum_mcs_objective_literal__(9999960, mus(11)).
__optimum_mcs_objective_literal__(9999999, mus(12)).
__optimum_mcs_objective_literal__(9995213, mus(13)).
__optimum_mcs_objective_literal__(9994156, mus(14)).
__optimum_mcs_objective_literal__(9999999, mus(15)).
__optimum_mcs_objective_literal__(9999999, mus(16)).
__optimum_mcs_objective_literal__(9999999, mus(17)).
__optimum_mcs_objective_literal__(9999999, mus(18)).
__optimum_mcs_objective_literal__(9999999, mus(19)).
__optimum_mcs_objective_literal__(8975167, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(7514534, mus(22)).
__optimum_mcs_objective_literal__(9549396, mus(23)).
__optimum_mcs_objective_literal__(9999999, mus(24)).
__optimum_mcs_objective_literal__(8805681, mus(25)).
__optimum_mcs_objective_literal__(9979245, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(9999999, mus(28)).
__optimum_mcs_objective_literal__(9868107, mus(29)).
__optimum_mcs_objective_literal__(9998784, mus(30)).
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
__optimum_mcs_objective_literal__(8080466, mus(49)).
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
__optimum_mcs_objective_literal__(8495153, mus(75)).
__optimum_mcs_objective_literal__(7999284, mus(76)).
__optimum_mcs_objective_literal__(9999999, mus(77)).
__optimum_mcs_objective_literal__(9426396, mus(78)).
__optimum_mcs_objective_literal__(9999999, mus(79)).
__optimum_mcs_objective_literal__(9999999, mus(80)).
__optimum_mcs_objective_literal__(9999999, mus(81)).
__optimum_mcs_objective_literal__(8049546, mus(82)).
__optimum_mcs_objective_literal__(7676843, mus(83)).
__optimum_mcs_objective_literal__(6774343, mus(84)).
__optimum_mcs_objective_literal__(5125175, mus(85)).
__optimum_mcs_objective_literal__(5670762, mus(86)).
__optimum_mcs_objective_literal__(9999999, mus(87)).
__optimum_mcs_objective_literal__(8720146, mus(88)).
__optimum_mcs_objective_literal__(5643036, mus(89)).
__optimum_mcs_objective_literal__(8107129, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(9317420, mus(92)).
__optimum_mcs_objective_literal__(7054018, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(9999999, mus(95)).
__optimum_mcs_objective_literal__(5087247, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(5560401, mus(98)).
__optimum_mcs_objective_literal__(9999999, mus(99)).
__optimum_mcs_objective_literal__(6144665, mus(100)).
__optimum_mcs_objective_literal__(9999999, mus(101)).
__optimum_mcs_objective_literal__(6234217, mus(102)).
__optimum_mcs_objective_literal__(9999999, mus(103)).
__optimum_mcs_objective_literal__(7235336, mus(104)).
__optimum_mcs_objective_literal__(9999999, mus(105)).
__optimum_mcs_objective_literal__(9042395, mus(106)).
__optimum_mcs_objective_literal__(9999999, mus(107)).
__optimum_mcs_objective_literal__(9999999, mus(108)).
__optimum_mcs_objective_literal__(9999999, mus(109)).
__optimum_mcs_objective_literal__(5137914, mus(110)).
__optimum_mcs_objective_literal__(9999999, mus(111)).
__optimum_mcs_objective_literal__(9999999, mus(112)).
__optimum_mcs_objective_literal__(9999999, mus(113)).
__optimum_mcs_objective_literal__(7009768, mus(114)).
__optimum_mcs_objective_literal__(9999999, mus(115)).
__optimum_mcs_objective_literal__(9999999, mus(116)).
__optimum_mcs_objective_literal__(9999999, mus(117)).
__optimum_mcs_objective_literal__(9999999, mus(118)).
__optimum_mcs_objective_literal__(6254045, mus(119)).
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
__optimum_mcs_objective_literal__(9999999, mus(158)).
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
__optimum_mcs_objective_literal__(9510312, mus(209)).
__optimum_mcs_objective_literal__(9961002, mus(210)).
__optimum_mcs_objective_literal__(9857865, mus(211)).
__optimum_mcs_objective_literal__(9993615, mus(212)).
__optimum_mcs_objective_literal__(9993549, mus(213)).
__optimum_mcs_objective_literal__(9999964, mus(214)).
__optimum_mcs_objective_literal__(9999999, mus(215)).
__optimum_mcs_objective_literal__(9929377, mus(216)).
__optimum_mcs_objective_literal__(9989088, mus(217)).
__optimum_mcs_objective_literal__(9972745, mus(218)).
__optimum_mcs_objective_literal__(9943892, mus(219)).
__optimum_mcs_objective_literal__(9999999, mus(220)).
__optimum_mcs_objective_literal__(9919097, mus(221)).
__optimum_mcs_objective_literal__(9999659, mus(222)).
__optimum_mcs_objective_literal__(9700862, mus(223)).
__optimum_mcs_objective_literal__(9999999, mus(224)).
__optimum_mcs_objective_literal__(9989620, mus(225)).
__optimum_mcs_objective_literal__(9986062, mus(226)).
__optimum_mcs_objective_literal__(9999999, mus(227)).
__optimum_mcs_objective_literal__(9998412, mus(228)).
__optimum_mcs_objective_literal__(9999999, mus(229)).
__optimum_mcs_objective_literal__(9999997, mus(230)).
__optimum_mcs_objective_literal__(9999988, mus(231)).
__optimum_mcs_objective_literal__(9999999, mus(232)).
__optimum_mcs_objective_literal__(9999999, mus(233)).
__optimum_mcs_objective_literal__(9951037, mus(234)).
__optimum_mcs_objective_literal__(9999999, mus(235)).
__optimum_mcs_objective_literal__(9999139, mus(236)).
__optimum_mcs_objective_literal__(9999999, mus(237)).
__optimum_mcs_objective_literal__(9999999, mus(238)).
__optimum_mcs_objective_literal__(9999999, mus(239)).
__optimum_mcs_objective_literal__(9999999, mus(240)).
__optimum_mcs_objective_literal__(9999998, mus(241)).
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
__optimum_mcs_objective_literal__(9995096, mus(266)).
__optimum_mcs_objective_literal__(9997837, mus(267)).
__optimum_mcs_objective_literal__(9997271, mus(268)).
__optimum_mcs_objective_literal__(9999999, mus(269)).
__optimum_mcs_objective_literal__(9999999, mus(270)).
__optimum_mcs_objective_literal__(9357840, mus(271)).
__optimum_mcs_objective_literal__(5274569, mus(272)).
__optimum_mcs_objective_literal__(9933802, mus(273)).
__optimum_mcs_objective_literal__(9961502, mus(274)).
__optimum_mcs_objective_literal__(9996203, mus(275)).
__optimum_mcs_objective_literal__(9975843, mus(276)).
__optimum_mcs_objective_literal__(9933241, mus(277)).
__optimum_mcs_objective_literal__(9898439, mus(278)).
__optimum_mcs_objective_literal__(9994437, mus(279)).
__optimum_mcs_objective_literal__(9999999, mus(280)).
__optimum_mcs_objective_literal__(9999999, mus(281)).
__optimum_mcs_objective_literal__(9999999, mus(282)).
__optimum_mcs_objective_literal__(9999999, mus(283)).
__optimum_mcs_objective_literal__(9999999, mus(284)).
__optimum_mcs_objective_literal__(9999999, mus(285)).
__optimum_mcs_objective_literal__(9999999, mus(286)).
__optimum_mcs_objective_literal__(9999999, mus(287)).
__optimum_mcs_objective_literal__(9999999, mus(288)).
__optimum_mcs_objective_literal__(9999998, mus(289)).
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
__optimum_mcs_objective_literal__(9999999, mus(323)).
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
__optimum_mcs_objective_literal__(9999999, mus(355)).
__optimum_mcs_objective_literal__(9999999, mus(356)).
__optimum_mcs_objective_literal__(9999999, mus(357)).
__optimum_mcs_objective_literal__(9999999, mus(358)).
__optimum_mcs_objective_literal__(9999999, mus(359)).
__optimum_mcs_objective_literal__(9999999, mus(360)).
__optimum_mcs_objective_literal__(9999999, mus(361)).
__optimum_mcs_objective_literal__(9999999, mus(362)).
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
__optimum_mcs_objective_literal__(9999999, mus(383)).
__optimum_mcs_objective_literal__(9999999, mus(384)).
__optimum_mcs_objective_literal__(9999999, mus(385)).
__optimum_mcs_objective_literal__(9999999, mus(386)).
__optimum_mcs_objective_literal__(9999999, mus(387)).
__optimum_mcs_objective_literal__(9999999, mus(388)).
__optimum_mcs_objective_literal__(9999999, mus(389)).
__optimum_mcs_objective_literal__(9999999, mus(390)).
__optimum_mcs_objective_literal__(9999999, mus(391)).
__optimum_mcs_objective_literal__(9999999, mus(392)).
__optimum_mcs_objective_literal__(9106743, mus(393)).
__optimum_mcs_objective_literal__(8399758, mus(394)).
__optimum_mcs_objective_literal__(9439503, mus(395)).
__optimum_mcs_objective_literal__(9267014, mus(396)).
__optimum_mcs_objective_literal__(9999999, mus(397)).
__optimum_mcs_objective_literal__(9615234, mus(398)).
__optimum_mcs_objective_literal__(6394907, mus(399)).
__optimum_mcs_objective_literal__(9934162, mus(400)).
__optimum_mcs_objective_literal__(9999999, mus(401)).
__optimum_mcs_objective_literal__(7938536, mus(402)).
__optimum_mcs_objective_literal__(9525257, mus(403)).
__optimum_mcs_objective_literal__(9999996, mus(404)).
__optimum_mcs_objective_literal__(8858676, mus(405)).
__optimum_mcs_objective_literal__(9999999, mus(406)).
__optimum_mcs_objective_literal__(9999996, mus(407)).
__optimum_mcs_objective_literal__(7134322, mus(408)).
__optimum_mcs_objective_literal__(9962158, mus(409)).
__optimum_mcs_objective_literal__(9999999, mus(410)).
__optimum_mcs_objective_literal__(7801963, mus(411)).
__optimum_mcs_objective_literal__(9999999, mus(412)).
__optimum_mcs_objective_literal__(9999999, mus(413)).
__optimum_mcs_objective_literal__(9952935, mus(414)).
__optimum_mcs_objective_literal__(9999999, mus(415)).
__optimum_mcs_objective_literal__(9999568, mus(416)).
__optimum_mcs_objective_literal__(9999999, mus(417)).
__optimum_mcs_objective_literal__(9477328, mus(418)).
__optimum_mcs_objective_literal__(9999999, mus(419)).
__optimum_mcs_objective_literal__(9999996, mus(420)).
__optimum_mcs_objective_literal__(9999999, mus(421)).
__optimum_mcs_objective_literal__(9999999, mus(422)).
__optimum_mcs_objective_literal__(9999999, mus(423)).
__optimum_mcs_objective_literal__(9679307, mus(424)).
__optimum_mcs_objective_literal__(9027675, mus(425)).
__optimum_mcs_objective_literal__(9222629, mus(426)).
__optimum_mcs_objective_literal__(9987137, mus(427)).
__optimum_mcs_objective_literal__(9999999, mus(428)).
__optimum_mcs_objective_literal__(9999719, mus(429)).
__optimum_mcs_objective_literal__(9998656, mus(430)).
__optimum_mcs_objective_literal__(9578127, mus(431)).
__optimum_mcs_objective_literal__(9022650, mus(432)).
__optimum_mcs_objective_literal__(9999999, mus(433)).
__optimum_mcs_objective_literal__(9563742, mus(434)).
__optimum_mcs_objective_literal__(9999998, mus(435)).
__optimum_mcs_objective_literal__(5090173, mus(436)).
__optimum_mcs_objective_literal__(5543434, mus(437)).
__optimum_mcs_objective_literal__(6463762, mus(438)).
__optimum_mcs_objective_literal__(8880454, mus(439)).
__optimum_mcs_objective_literal__(5683960, mus(440)).
__optimum_mcs_objective_literal__(5413150, mus(441)).
__optimum_mcs_objective_literal__(8485380, mus(442)).
__optimum_mcs_objective_literal__(9999993, mus(443)).
__optimum_mcs_objective_literal__(8605323, mus(444)).
__optimum_mcs_objective_literal__(9999998, mus(445)).
__optimum_mcs_objective_literal__(7711166, mus(446)).
__optimum_mcs_objective_literal__(5127216, mus(447)).
__optimum_mcs_objective_literal__(5945821, mus(448)).
__optimum_mcs_objective_literal__(5294405, mus(449)).
__optimum_mcs_objective_literal__(9999997, mus(450)).
__optimum_mcs_objective_literal__(5139568, mus(451)).
__optimum_mcs_objective_literal__(6887479, mus(452)).
__optimum_mcs_objective_literal__(7261903, mus(453)).
__optimum_mcs_objective_literal__(9999938, mus(454)).
__optimum_mcs_objective_literal__(5227186, mus(455)).
__optimum_mcs_objective_literal__(5379527, mus(456)).
__optimum_mcs_objective_literal__(7429551, mus(457)).
__optimum_mcs_objective_literal__(5879179, mus(458)).
__optimum_mcs_objective_literal__(8520851, mus(459)).
__optimum_mcs_objective_literal__(9757421, mus(460)).
__optimum_mcs_objective_literal__(5081528, mus(461)).
__optimum_mcs_objective_literal__(5141369, mus(462)).
__optimum_mcs_objective_literal__(5020548, mus(463)).
__optimum_mcs_objective_literal__(9941376, mus(464)).
__optimum_mcs_objective_literal__(5003716, mus(465)).
__optimum_mcs_objective_literal__(5027444, mus(466)).
__optimum_mcs_objective_literal__(5272784, mus(467)).
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
__optimum_mcs_objective_literal__(9876633, mus(486)).
__optimum_mcs_objective_literal__(7462698, mus(487)).
__optimum_mcs_objective_literal__(9173915, mus(488)).
__optimum_mcs_objective_literal__(5096826, mus(489)).
__optimum_mcs_objective_literal__(9131459, mus(490)).
__optimum_mcs_objective_literal__(9713091, mus(491)).
__optimum_mcs_objective_literal__(9939459, mus(492)).
__optimum_mcs_objective_literal__(9434556, mus(493)).
__optimum_mcs_objective_literal__(5010870, mus(494)).
__optimum_mcs_objective_literal__(9998944, mus(495)).
__optimum_mcs_objective_literal__(9902764, mus(496)).
__optimum_mcs_objective_literal__(9557796, mus(497)).
__optimum_mcs_objective_literal__(9996581, mus(498)).
__optimum_mcs_objective_literal__(9999921, mus(499)).
__optimum_mcs_objective_literal__(9589932, mus(500)).
__optimum_mcs_objective_literal__(9268337, mus(501)).
__optimum_mcs_objective_literal__(9956881, mus(502)).
__optimum_mcs_objective_literal__(9994424, mus(503)).
__optimum_mcs_objective_literal__(9999996, mus(504)).
__optimum_mcs_objective_literal__(9889557, mus(505)).
__optimum_mcs_objective_literal__(5555162, mus(506)).
__optimum_mcs_objective_literal__(5298905, mus(507)).
__optimum_mcs_objective_literal__(6567888, mus(508)).
__optimum_mcs_objective_literal__(5264069, mus(509)).
__optimum_mcs_objective_literal__(5427485, mus(510)).
__optimum_mcs_objective_literal__(5304507, mus(511)).
__optimum_mcs_objective_literal__(8749505, mus(512)).
__optimum_mcs_objective_literal__(5269540, mus(513)).
__optimum_mcs_objective_literal__(9845766, mus(514)).
__optimum_mcs_objective_literal__(9661494, mus(515)).
__optimum_mcs_objective_literal__(9879018, mus(516)).
__optimum_mcs_objective_literal__(9719941, mus(517)).
__optimum_mcs_objective_literal__(5488292, mus(518)).
__optimum_mcs_objective_literal__(9999999, mus(519)).
__optimum_mcs_objective_literal__(5092363, mus(520)).
__optimum_mcs_objective_literal__(7819293, mus(521)).
__optimum_mcs_objective_literal__(9926354, mus(522)).
__optimum_mcs_objective_literal__(5136833, mus(523)).
__optimum_mcs_objective_literal__(9896948, mus(524)).
__optimum_mcs_objective_literal__(9999999, mus(525)).
__optimum_mcs_objective_literal__(9999999, mus(526)).
__optimum_mcs_objective_literal__(9995432, mus(527)).
__optimum_mcs_objective_literal__(9999958, mus(528)).
__optimum_mcs_objective_literal__(9999999, mus(529)).
__optimum_mcs_objective_literal__(9997563, mus(530)).
__optimum_mcs_objective_literal__(9999999, mus(531)).
__optimum_mcs_objective_literal__(9999999, mus(532)).
__optimum_mcs_objective_literal__(9999999, mus(533)).
__optimum_mcs_objective_literal__(9999999, mus(534)).
__optimum_mcs_objective_literal__(9999999, mus(535)).
__optimum_mcs_objective_literal__(9999999, mus(536)).
__optimum_mcs_objective_literal__(9999999, mus(537)).
__optimum_mcs_objective_literal__(9999999, mus(538)).
__optimum_mcs_objective_literal__(9999589, mus(539)).
__optimum_mcs_objective_literal__(9999999, mus(540)).
__optimum_mcs_objective_literal__(9999999, mus(541)).
__optimum_mcs_objective_literal__(9999999, mus(542)).
__optimum_mcs_objective_literal__(9999999, mus(543)).
__optimum_mcs_objective_literal__(9999999, mus(544)).
__optimum_mcs_objective_literal__(9999999, mus(545)).
__optimum_mcs_objective_literal__(9980839, mus(546)).
__optimum_mcs_objective_literal__(9999999, mus(547)).
__optimum_mcs_objective_literal__(9083566, mus(548)).
__optimum_mcs_objective_literal__(9999999, mus(549)).
__optimum_mcs_objective_literal__(9476678, mus(550)).
__optimum_mcs_objective_literal__(9999999, mus(551)).
__optimum_mcs_objective_literal__(9994271, mus(552)).
__optimum_mcs_objective_literal__(9999999, mus(553)).
__optimum_mcs_objective_literal__(9999999, mus(554)).
__optimum_mcs_objective_literal__(9999999, mus(555)).
__optimum_mcs_objective_literal__(9999999, mus(556)).
__optimum_mcs_objective_literal__(9999999, mus(557)).
__optimum_mcs_objective_literal__(9999999, mus(558)).
__optimum_mcs_objective_literal__(9999999, mus(559)).
__optimum_mcs_objective_literal__(9999999, mus(560)).
__optimum_mcs_objective_literal__(9999999, mus(561)).
__optimum_mcs_objective_literal__(9999999, mus(562)).
__optimum_mcs_objective_literal__(9999999, mus(563)).
__optimum_mcs_objective_literal__(9999999, mus(564)).
__optimum_mcs_objective_literal__(9999999, mus(565)).
__optimum_mcs_objective_literal__(9999999, mus(566)).
__optimum_mcs_objective_literal__(9999999, mus(567)).
__optimum_mcs_objective_literal__(9999999, mus(568)).
__optimum_mcs_objective_literal__(9999999, mus(569)).
__optimum_mcs_objective_literal__(9999999, mus(570)).
__optimum_mcs_objective_literal__(9999999, mus(571)).
__optimum_mcs_objective_literal__(9999999, mus(572)).
__optimum_mcs_objective_literal__(9999999, mus(573)).
__optimum_mcs_objective_literal__(9999999, mus(574)).
__optimum_mcs_objective_literal__(6807229, mus(575)).
__optimum_mcs_objective_literal__(9071089, mus(576)).
__optimum_mcs_objective_literal__(7513236, mus(577)).
__optimum_mcs_objective_literal__(7697903, mus(578)).
__optimum_mcs_objective_literal__(6173689, mus(579)).
__optimum_mcs_objective_literal__(8809721, mus(580)).
__optimum_mcs_objective_literal__(9292006, mus(581)).
__optimum_mcs_objective_literal__(6553911, mus(582)).
__optimum_mcs_objective_literal__(6232857, mus(583)).
__optimum_mcs_objective_literal__(9774719, mus(584)).
__optimum_mcs_objective_literal__(8737215, mus(585)).
__optimum_mcs_objective_literal__(9036858, mus(586)).
__optimum_mcs_objective_literal__(9999999, mus(587)).
__optimum_mcs_objective_literal__(9999427, mus(588)).
__optimum_mcs_objective_literal__(7407744, mus(589)).
__optimum_mcs_objective_literal__(9999179, mus(590)).
__optimum_mcs_objective_literal__(7789827, mus(591)).
__optimum_mcs_objective_literal__(9999999, mus(592)).
__optimum_mcs_objective_literal__(9999999, mus(593)).
__optimum_mcs_objective_literal__(9999999, mus(594)).
__optimum_mcs_objective_literal__(9999999, mus(595)).
__optimum_mcs_objective_literal__(9999999, mus(596)).
__optimum_mcs_objective_literal__(9999999, mus(597)).
__optimum_mcs_objective_literal__(9998023, mus(598)).
__optimum_mcs_objective_literal__(9997873, mus(599)).
__optimum_mcs_objective_literal__(9996914, mus(600)).
__optimum_mcs_objective_literal__(9999999, mus(601)).
__optimum_mcs_objective_literal__(9999806, mus(602)).
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
__optimum_mcs_objective_literal__(9999999, mus(614)).
__optimum_mcs_objective_literal__(9999999, mus(615)).
__optimum_mcs_objective_literal__(9999999, mus(616)).
__optimum_mcs_objective_literal__(9999999, mus(617)).
__optimum_mcs_objective_literal__(9999999, mus(618)).
__optimum_mcs_objective_literal__(9999999, mus(619)).
__optimum_mcs_objective_literal__(9999999, mus(620)).
__optimum_mcs_objective_literal__(9999999, mus(621)).
__optimum_mcs_objective_literal__(9999999, mus(622)).
__optimum_mcs_objective_literal__(9999999, mus(623)).
__optimum_mcs_objective_literal__(9999999, mus(624)).
__optimum_mcs_objective_literal__(9999999, mus(625)).
__optimum_mcs_objective_literal__(9999999, mus(626)).
__optimum_mcs_objective_literal__(9999999, mus(627)).
__optimum_mcs_objective_literal__(9999999, mus(628)).
__optimum_mcs_objective_literal__(9999999, mus(629)).
__optimum_mcs_objective_literal__(9999999, mus(630)).
__optimum_mcs_objective_literal__(9999999, mus(631)).
__optimum_mcs_objective_literal__(9999999, mus(632)).
__optimum_mcs_objective_literal__(9999230, mus(633)).
__optimum_mcs_objective_literal__(9999999, mus(634)).
__optimum_mcs_objective_literal__(9998736, mus(635)).
__optimum_mcs_objective_literal__(9999997, mus(636)).
__optimum_mcs_objective_literal__(9993204, mus(637)).
__optimum_mcs_objective_literal__(9999975, mus(638)).
__optimum_mcs_objective_literal__(9997964, mus(639)).
__optimum_mcs_objective_literal__(9999999, mus(640)).
__optimum_mcs_objective_literal__(9995308, mus(641)).
__optimum_mcs_objective_literal__(9999999, mus(642)).
__optimum_mcs_objective_literal__(9999999, mus(643)).
__optimum_mcs_objective_literal__(9999999, mus(644)).
__optimum_mcs_objective_literal__(9999999, mus(645)).
__optimum_mcs_objective_literal__(9999996, mus(646)).
__optimum_mcs_objective_literal__(9999998, mus(647)).
__optimum_mcs_objective_literal__(9948263, mus(648)).
__optimum_mcs_objective_literal__(9986274, mus(649)).
__optimum_mcs_objective_literal__(9999967, mus(650)).
__optimum_mcs_objective_literal__(9999999, mus(651)).
__optimum_mcs_objective_literal__(9999991, mus(652)).
__optimum_mcs_objective_literal__(9999999, mus(653)).
__optimum_mcs_objective_literal__(9999986, mus(654)).
__optimum_mcs_objective_literal__(9999407, mus(655)).
__optimum_mcs_objective_literal__(9999999, mus(656)).
__optimum_mcs_objective_literal__(9999994, mus(657)).
__optimum_mcs_objective_literal__(9999999, mus(658)).
__optimum_mcs_objective_literal__(9999992, mus(659)).
__optimum_mcs_objective_literal__(9999781, mus(660)).
__optimum_mcs_objective_literal__(9999999, mus(661)).
__optimum_mcs_objective_literal__(9999999, mus(662)).
__optimum_mcs_objective_literal__(9999999, mus(663)).
__optimum_mcs_objective_literal__(9998901, mus(664)).
__optimum_mcs_objective_literal__(9999999, mus(665)).
__optimum_mcs_objective_literal__(9974497, mus(666)).
__optimum_mcs_objective_literal__(9999999, mus(667)).
__optimum_mcs_objective_literal__(9999999, mus(668)).
__optimum_mcs_objective_literal__(9999999, mus(669)).
__optimum_mcs_objective_literal__(9999983, mus(670)).
__optimum_mcs_objective_literal__(6624185, mus(671)).
__optimum_mcs_objective_literal__(6867084, mus(672)).
__optimum_mcs_objective_literal__(6276447, mus(673)).
__optimum_mcs_objective_literal__(6665647, mus(674)).
__optimum_mcs_objective_literal__(9436275, mus(675)).
__optimum_mcs_objective_literal__(6764820, mus(676)).
__optimum_mcs_objective_literal__(6652025, mus(677)).
__optimum_mcs_objective_literal__(7166844, mus(678)).
__optimum_mcs_objective_literal__(5724721, mus(679)).
__optimum_mcs_objective_literal__(7745187, mus(680)).
__optimum_mcs_objective_literal__(7508766, mus(681)).
__optimum_mcs_objective_literal__(7104294, mus(682)).
__optimum_mcs_objective_literal__(7438653, mus(683)).
__optimum_mcs_objective_literal__(6062535, mus(684)).
__optimum_mcs_objective_literal__(8857213, mus(685)).
__optimum_mcs_objective_literal__(6549009, mus(686)).
__optimum_mcs_objective_literal__(5053409, mus(687)).
__optimum_mcs_objective_literal__(6935084, mus(688)).
__optimum_mcs_objective_literal__(5714322, mus(689)).
__optimum_mcs_objective_literal__(7719428, mus(690)).
__optimum_mcs_objective_literal__(5022866, mus(691)).
__optimum_mcs_objective_literal__(5486614, mus(692)).
__optimum_mcs_objective_literal__(5424166, mus(693)).
__optimum_mcs_objective_literal__(5413853, mus(694)).
__optimum_mcs_objective_literal__(5310837, mus(695)).
__optimum_mcs_objective_literal__(6402471, mus(696)).
__optimum_mcs_objective_literal__(7011830, mus(697)).
__optimum_mcs_objective_literal__(6157018, mus(698)).
__optimum_mcs_objective_literal__(6574181, mus(699)).
__optimum_mcs_objective_literal__(7825566, mus(700)).
__optimum_mcs_objective_literal__(5584022, mus(701)).
__optimum_mcs_objective_literal__(5256478, mus(702)).
__optimum_mcs_objective_literal__(5989483, mus(703)).
__optimum_mcs_objective_literal__(9999999, mus(704)).
__optimum_mcs_objective_literal__(6412134, mus(705)).
__optimum_mcs_objective_literal__(9999999, mus(706)).
__optimum_mcs_objective_literal__(9999991, mus(707)).
__optimum_mcs_objective_literal__(9999999, mus(708)).
__optimum_mcs_objective_literal__(9999999, mus(709)).
__optimum_mcs_objective_literal__(9999999, mus(710)).
__optimum_mcs_objective_literal__(9999999, mus(711)).
__optimum_mcs_objective_literal__(9999999, mus(712)).
__optimum_mcs_objective_literal__(9999999, mus(713)).
__optimum_mcs_objective_literal__(9999999, mus(714)).
__optimum_mcs_objective_literal__(9999999, mus(715)).
__optimum_mcs_objective_literal__(9999999, mus(716)).
__optimum_mcs_objective_literal__(9999999, mus(717)).
__optimum_mcs_objective_literal__(9999999, mus(718)).
__optimum_mcs_objective_literal__(9999999, mus(719)).
__optimum_mcs_objective_literal__(9999999, mus(720)).
__optimum_mcs_objective_literal__(9999999, mus(721)).
__optimum_mcs_objective_literal__(9999999, mus(722)).
__optimum_mcs_objective_literal__(9999999, mus(723)).
__optimum_mcs_objective_literal__(9999999, mus(724)).
__optimum_mcs_objective_literal__(9999999, mus(725)).
__optimum_mcs_objective_literal__(9999999, mus(726)).
__optimum_mcs_objective_literal__(9999999, mus(727)).
__optimum_mcs_objective_literal__(9999999, mus(728)).
__optimum_mcs_objective_literal__(9999999, mus(729)).
__optimum_mcs_objective_literal__(9999999, mus(730)).
__optimum_mcs_objective_literal__(9999999, mus(731)).
__optimum_mcs_objective_literal__(9999999, mus(732)).
__optimum_mcs_objective_literal__(9999999, mus(733)).
__optimum_mcs_objective_literal__(9999999, mus(734)).
__optimum_mcs_objective_literal__(5175506, mus(735)).
__optimum_mcs_objective_literal__(9307377, mus(736)).
__optimum_mcs_objective_literal__(9846516, mus(737)).
__optimum_mcs_objective_literal__(9129616, mus(738)).
__optimum_mcs_objective_literal__(7717590, mus(739)).
__optimum_mcs_objective_literal__(8137774, mus(740)).
__optimum_mcs_objective_literal__(9999999, mus(741)).
__optimum_mcs_objective_literal__(6922858, mus(742)).
__optimum_mcs_objective_literal__(8853264, mus(743)).
__optimum_mcs_objective_literal__(9728052, mus(744)).
__optimum_mcs_objective_literal__(9999999, mus(745)).
__optimum_mcs_objective_literal__(8646994, mus(746)).
__optimum_mcs_objective_literal__(6544179, mus(747)).
__optimum_mcs_objective_literal__(9999999, mus(748)).
__optimum_mcs_objective_literal__(7753137, mus(749)).
__optimum_mcs_objective_literal__(9999999, mus(750)).
__optimum_mcs_objective_literal__(9751795, mus(751)).
__optimum_mcs_objective_literal__(9999999, mus(752)).
__optimum_mcs_objective_literal__(9999999, mus(753)).
__optimum_mcs_objective_literal__(9961227, mus(754)).
__optimum_mcs_objective_literal__(9999999, mus(755)).
__optimum_mcs_objective_literal__(9999999, mus(756)).
__optimum_mcs_objective_literal__(9999999, mus(757)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
