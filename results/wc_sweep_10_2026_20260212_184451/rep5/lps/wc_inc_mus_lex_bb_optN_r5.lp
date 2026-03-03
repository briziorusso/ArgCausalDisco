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
qpair(0,1,s6y7y9).
qpair(0,1,s2y8y9).
qpair(0,1,s2y7y8y9).
qpair(0,1,s3y4).
qpair(0,1,s5).
qpair(0,1,s3y5y6).
qpair(0,1,s8).
qpair(0,1,s4y6).
qpair(0,1,s8y9).
qpair(0,1,s4y5y6).
qpair(0,1,s2y6y8).
qpair(0,1,s6y7y8y9).
qpair(0,1,s6y8).
qpair(0,1,s4).
qpair(0,1,s6y7y8).
qpair(0,1,s4y5).
qpair(0,1,s5y6).
qpair(0,1,s7y8y9).
qpair(0,1,s2y6y7).
qpair(0,1,s3y4y6).
qpair(0,1,s7y9).
qpair(0,1,s6y7).
qpair(0,1,s9).
qpair(0,1,s3y5).
qpair(0,1,s6).
qpair(0,1,s2y6y7y8).
qpair(0,1,s2y6y7y8y9).
qpair(0,1,s2y6y9).
qpair(0,1,s2y7y8).
qpair(0,1,s2y9).
qpair(0,1,s3y4y5).
qpair(0,1,s2y6).
qpair(0,1,s2y6y8y9).
qpair(0,1,s7y8).
qpair(0,2,s6).
qpair(0,2,s6y7y9).
qpair(0,2,s5).
qpair(0,2,s8).
qpair(0,2,s6y7y8).
qpair(0,2,s7y9).
qpair(0,2,s6y7).
qpair(0,2,s7).
qpair(0,2,s8y9).
qpair(0,2,s6y8y9).
qpair(0,2,s7y8y9).
qpair(0,2,empty).
qpair(0,2,s3).
qpair(0,2,s9).
qpair(0,3,s6y7y9).
qpair(0,3,s2y8y9).
qpair(0,3,s2y7y8y9).
qpair(0,3,s2y6y8).
qpair(0,3,s2y8).
qpair(0,3,s6y7y8y9).
qpair(0,3,s6y8).
qpair(0,3,s6y7y8).
qpair(0,3,s6y8y9).
qpair(0,3,s4y8).
qpair(0,3,s7y8y9).
qpair(0,3,s2y6y7y9).
qpair(0,3,s2y7y9).
qpair(0,3,s2y6y7).
qpair(0,3,s2y7).
qpair(0,3,s1y8).
qpair(0,3,s6y7).
qpair(0,3,s2y6y7y8).
qpair(0,3,s2y6y7y8y9).
qpair(0,3,s2y6y9).
qpair(0,3,s1y4).
qpair(0,3,s2y9).
qpair(0,3,s2y6).
qpair(0,3,s2y6y8y9).
qpair(0,3,s6y9).
qpair(0,3,s7y8).
qpair(0,5,s2y6y7).
qpair(0,5,s6y7y9).
qpair(0,5,s2y6y7y8y9).
qpair(0,5,s2y7).
qpair(0,5,s6y9).
qpair(0,5,s6y8).
qpair(0,5,s2y6y9).
qpair(0,5,s1y4).
qpair(0,5,s2y7y8).
qpair(0,5,s7y9).
qpair(0,5,s6y7).
qpair(0,5,s6y7y8).
qpair(0,5,s2y6y7y9).
qpair(0,5,s2y6).
qpair(0,5,s4y8).
qpair(0,5,s2y6y8).
qpair(0,5,s3).
qpair(0,5,s1y4y8).
qpair(0,6,s2).
qpair(0,6,s2y8y9).
qpair(0,6,s2y7y8y9).
qpair(0,6,s5).
qpair(0,6,s3y7).
qpair(0,6,s8).
qpair(0,6,s5y7).
qpair(0,6,s3y7y9).
qpair(0,6,s8y9).
qpair(0,6,s1y9).
qpair(0,6,s1y3y9).
qpair(0,6,s2y8).
qpair(0,6,s1y7y8y9).
qpair(0,6,s1).
qpair(0,6,s7).
qpair(0,6,s3y9).
qpair(0,6,s3y7y8).
qpair(0,6,s5y9).
qpair(0,6,s1y7y9).
qpair(0,6,s7y8y9).
qpair(0,6,s2y7y9).
qpair(0,6,s1y3y8).
qpair(0,6,s2y7).
qpair(0,6,s1y5).
qpair(0,6,s1y8).
qpair(0,6,s3y8y9).
qpair(0,6,s9).
qpair(0,6,s3y5).
qpair(0,6,s1y7y8).
qpair(0,6,s3y8).
qpair(0,6,s5y8).
qpair(0,6,s2y7y8).
qpair(0,6,s2y9).
qpair(0,6,s1y3y7).
qpair(0,6,empty).
qpair(0,6,s1y8y9).
qpair(0,7,s6).
qpair(0,7,s2).
qpair(0,7,s2y8y9).
qpair(0,7,s5).
qpair(0,7,s6y9).
qpair(0,7,s8).
qpair(0,7,s2y6y9).
qpair(0,7,s2y9).
qpair(0,7,s6y8y9).
qpair(0,7,s8y9).
qpair(0,7,s2y6).
qpair(0,7,s2y6y8y9).
qpair(0,7,s2y6y8).
qpair(0,7,empty).
qpair(0,7,s3).
qpair(0,7,s2y8).
qpair(0,8,s6y7y9).
qpair(0,8,s2y3y6y9).
qpair(0,8,s2).
qpair(0,8,s3y5y6).
qpair(0,8,s3y7).
qpair(0,8,s3y5y9).
qpair(0,8,s5y7).
qpair(0,8,s2y5).
qpair(0,8,s2y3y9).
qpair(0,8,s2y3y6).
qpair(0,8,s4).
qpair(0,8,s7).
qpair(0,8,s3y9).
qpair(0,8,s5y6).
qpair(0,8,s3y6).
qpair(0,8,s5y9).
qpair(0,8,s2y6y7y9).
qpair(0,8,s2y5y6).
qpair(0,8,s2y7y9).
qpair(0,8,s2y6y7).
qpair(0,8,s2y7).
qpair(0,8,s2y3y5y6y9).
qpair(0,8,s2y3y5).
qpair(0,8,s3y6y9).
qpair(0,8,s7y9).
qpair(0,8,s6y7).
qpair(0,8,s2y3y5y9).
qpair(0,8,s2y3y5y6).
qpair(0,8,s2y5y6y9).
qpair(0,8,s3).
qpair(0,8,s9).
qpair(0,8,s3y5).
qpair(0,8,s6).
qpair(0,8,s5y6y9).
qpair(0,8,s2y9).
qpair(0,8,s2y6).
qpair(0,8,empty).
qpair(0,9,s6).
qpair(0,9,s2y6y7).
qpair(0,9,s2y6y7y8).
qpair(0,9,s5).
qpair(0,9,s6y8).
qpair(0,9,s8).
qpair(0,9,s2y7y8).
qpair(0,9,s6y7y8).
qpair(0,9,s1).
qpair(0,9,s6y7).
qpair(0,9,s2y6).
qpair(0,9,empty).
qpair(0,9,s3).
qpair(0,9,s7y8).
qpair(0,9,s2y8).
qpair(1,2,empty).
qpair(1,3,s6).
qpair(1,3,s4y5y8).
qpair(1,3,s2).
qpair(1,3,s5).
qpair(1,3,s5y8).
qpair(1,3,s5y6y8).
qpair(1,3,s6y8).
qpair(1,3,s8).
qpair(1,3,s4y6).
qpair(1,3,s4y6y8).
qpair(1,3,s7).
qpair(1,3,s4y5).
qpair(1,3,s5y6).
qpair(1,3,s4y8).
qpair(1,3,s4y5y6).
qpair(1,3,s0).
qpair(1,3,s9).
qpair(1,4,s6).
qpair(1,4,s5).
qpair(1,4,s3y5y6).
qpair(1,4,s8).
qpair(1,4,s5y6).
qpair(1,4,s3y6).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,4,s9).
qpair(1,4,s3y5).
qpair(1,5,s6).
qpair(1,5,s3y8).
qpair(1,5,s2).
qpair(1,5,s3y4).
qpair(1,5,s3y4y6).
qpair(1,5,s8).
qpair(1,5,s4).
qpair(1,5,s3y4y8).
qpair(1,5,s7).
qpair(1,5,s4y8).
qpair(1,5,s0).
qpair(1,5,empty).
qpair(1,5,s3).
qpair(1,5,s9).
qpair(1,6,s3y4).
qpair(1,6,s5).
qpair(1,6,s3y7).
qpair(1,6,s8).
qpair(1,6,s5y7).
qpair(1,6,s3y7y9).
qpair(1,6,s0y5).
qpair(1,6,s0y3y9).
qpair(1,6,s0y8).
qpair(1,6,s7).
qpair(1,6,s4y5).
qpair(1,6,s3y9).
qpair(1,6,s0y7y9).
qpair(1,6,s3y7y8).
qpair(1,6,s5y9).
qpair(1,6,s0y3y8).
qpair(1,6,s0y7).
qpair(1,6,s7y9).
qpair(1,6,s3y8y9).
qpair(1,6,s0).
qpair(1,6,s0y7y8).
qpair(1,6,s3).
qpair(1,6,s3y5).
qpair(1,6,s3y8).
qpair(1,6,s0y3y7).
qpair(1,6,s5y8).
qpair(1,6,s0y3).
qpair(1,6,s3y4y5).
qpair(1,6,s0y7y8y9).
qpair(1,6,empty).
qpair(1,6,s7y8).
qpair(1,7,s6).
qpair(1,7,s3y4y5y6).
qpair(1,7,s3y4y6).
qpair(1,7,s5).
qpair(1,7,s3y5y6).
qpair(1,7,s4y6).
qpair(1,7,s4).
qpair(1,7,s0y6).
qpair(1,7,s2y6).
qpair(1,7,s5y6).
qpair(1,7,s4y5y6).
qpair(1,7,s3y6).
qpair(1,9,s3y4y5y6).
qpair(1,9,s3y4).
qpair(1,9,s0y6y8).
qpair(1,9,s3y4y6).
qpair(1,9,s3y5y6).
qpair(1,9,s6y8).
qpair(1,9,s4y6).
qpair(1,9,s7).
qpair(1,9,s4y5).
qpair(1,9,s3y4y5).
qpair(1,9,s5y6).
qpair(1,9,s4y5y6).
qpair(1,9,s3y6).
qpair(1,9,s0y8).
qpair(1,9,s3y5).
qpair(2,3,s6).
qpair(2,3,s0y7).
qpair(2,3,s5).
qpair(2,3,s1y4).
qpair(2,3,s1y4y8).
qpair(2,5,s6).
qpair(2,5,s0y7y8).
qpair(2,5,s3).
qpair(2,5,s1y4).
qpair(2,7,s6).
qpair(2,7,s5).
qpair(2,7,s6y8).
qpair(2,7,s8).
qpair(2,7,s0y6).
qpair(2,7,s0).
qpair(2,7,s0y8).
qpair(2,7,empty).
qpair(2,7,s3).
qpair(2,8,s0y3y6y9).
qpair(2,8,s5).
qpair(2,8,s3y5y6).
qpair(2,8,s0y3y5y6y9).
qpair(2,8,s3y5y9).
qpair(2,8,s5y7).
qpair(2,8,s0y3y6).
qpair(2,8,s0y3y9).
qpair(2,8,s0y5y6).
qpair(2,8,s4).
qpair(2,8,s7).
qpair(2,8,s3y9).
qpair(2,8,s5y6).
qpair(2,8,s0y3y5).
qpair(2,8,s0y7).
qpair(2,8,s0y3y5y9).
qpair(2,8,s3y5y6y9).
qpair(2,8,s3y6y9).
qpair(2,8,s6y7).
qpair(2,8,s0).
qpair(2,8,s0y6y9).
qpair(2,8,s3).
qpair(2,8,s9).
qpair(2,8,s3y5).
qpair(2,8,s0y5y6y9).
qpair(2,8,s5y6y9).
qpair(2,8,s0y3).
qpair(2,8,empty).
qpair(2,8,s6y9).
qpair(2,9,s1).
qpair(3,4,s6).
qpair(3,4,s2).
qpair(3,4,s5).
qpair(3,4,s1y5).
qpair(3,4,s5y8).
qpair(3,4,s6y8).
qpair(3,4,s1y8).
qpair(3,4,s5y6y8).
qpair(3,4,s1y5y6).
qpair(3,4,s1).
qpair(3,4,s7).
qpair(3,4,s5y6).
qpair(3,4,s1y6y8).
qpair(3,4,s0).
qpair(3,4,s1y6).
qpair(3,4,empty).
qpair(3,4,s1y5y8).
qpair(3,5,s6).
qpair(3,5,s2).
qpair(3,5,s6y8).
qpair(3,5,s4y6).
qpair(3,5,s4y6y8).
qpair(3,5,s1y4y6).
qpair(3,5,s7).
qpair(3,5,s1y6).
qpair(3,6,s4y5y8).
qpair(3,6,s2).
qpair(3,6,s5).
qpair(3,6,s5y8).
qpair(3,6,s5y7).
qpair(3,6,s4y5).
qpair(3,6,s0y5).
qpair(3,6,s5y9).
qpair(3,6,s1y4y5).
qpair(3,6,s1y5y8).
qpair(3,7,s6).
qpair(3,7,s2).
qpair(3,7,s5).
qpair(3,7,s1).
qpair(3,7,empty).
qpair(3,8,s5).
qpair(3,8,s4y6).
qpair(3,8,s5y7).
qpair(3,8,s0y2).
qpair(3,8,s0y5).
qpair(3,8,s4y5y6).
qpair(3,8,s1y6).
qpair(3,8,s2y5).
qpair(3,8,s1y4y5).
qpair(3,8,s0y5y6).
qpair(3,8,s0y2y5y6y9).
qpair(3,8,s4).
qpair(3,8,s0y5y9).
qpair(3,8,s1y5y6).
qpair(3,8,s1).
qpair(3,8,s4y5).
qpair(3,8,s5y6).
qpair(3,8,s5y9).
qpair(3,8,s2y5y9).
qpair(3,8,s0y7).
qpair(3,8,s0y2y5y9).
qpair(3,8,s7y9).
qpair(3,8,s6y7).
qpair(3,8,s0y2y6).
qpair(3,8,s0y2y9).
qpair(3,8,s0).
qpair(3,8,s0y6y9).
qpair(3,8,s2y5y6y9).
qpair(3,8,s0y5y6y9).
qpair(3,8,s6).
qpair(3,8,s5y6y9).
qpair(3,8,s0y9).
qpair(3,8,s1y4y6).
qpair(3,8,s1y4).
qpair(3,8,s0y6).
qpair(3,8,s2y6y9).
qpair(3,8,s2y9).
qpair(3,8,s2y6).
qpair(3,8,s0y2y5).
qpair(3,8,empty).
qpair(3,9,s1).
qpair(3,9,s1y4).
qpair(3,9,s4).
qpair(4,5,s6).
qpair(4,5,s3y8).
qpair(4,5,s2).
qpair(4,5,s1y3y8).
qpair(4,5,s6y8).
qpair(4,5,s8).
qpair(4,5,s1y8).
qpair(4,5,s1).
qpair(4,5,s7).
qpair(4,5,s0).
qpair(4,5,s1y6).
qpair(4,5,empty).
qpair(4,5,s3).
qpair(4,5,s9).
qpair(4,6,s0y1y9).
qpair(4,6,s8y9).
qpair(4,6,s1y3).
qpair(4,6,s1y9).
qpair(4,6,s1y7y8y9).
qpair(4,6,s0y7y9).
qpair(4,6,s0y1y7y9).
qpair(4,6,s0y1y7y8y9).
qpair(4,6,s0y7).
qpair(4,6,s1y8).
qpair(4,6,s7y9).
qpair(4,6,s0y7y8).
qpair(4,6,s3y5).
qpair(4,6,s1y7y8).
qpair(4,6,s0y1y7y8).
qpair(4,6,s0y9).
qpair(4,6,s0y1y8y9).
qpair(4,6,s0y7y8y9).
qpair(4,6,s1y8y9).
qpair(4,6,s7y8).
qpair(5,6,s3y8).
qpair(5,6,s2).
qpair(5,6,s3y4).
qpair(5,6,s3y7).
qpair(5,6,s0y3).
qpair(5,6,s3).
qpair(5,6,s3y9).
qpair(5,6,s1y3).
qpair(5,7,s6).
qpair(5,7,s2).
qpair(5,7,s4).
qpair(5,7,s1).
qpair(5,7,s0).
qpair(5,7,empty).
qpair(5,7,s3).
qpair(5,7,s9).
qpair(5,8,s2y3y6y9).
qpair(5,8,s2).
qpair(5,8,s0y3y6y9).
qpair(5,8,s3y4).
qpair(5,8,s3y7).
qpair(5,8,s4y6).
qpair(5,8,s0y3y6).
qpair(5,8,s1y6).
qpair(5,8,s1y3).
qpair(5,8,s2y3y9).
qpair(5,8,s2y3y6).
qpair(5,8,s0y2y6y9).
qpair(5,8,s4).
qpair(5,8,s1).
qpair(5,8,s7).
qpair(5,8,s3y9).
qpair(5,8,s0y7).
qpair(5,8,s0y2y3).
qpair(5,8,s3y6y9).
qpair(5,8,s7y9).
qpair(5,8,s6y7).
qpair(5,8,s0y2y6).
qpair(5,8,s0y2y3y6y9).
qpair(5,8,s0y2y9).
qpair(5,8,s0).
qpair(5,8,s3).
qpair(5,8,s9).
qpair(5,8,s0y2y3y6).
qpair(5,8,s6).
qpair(5,8,s0y2y3y9).
qpair(5,8,s0y3).
qpair(5,8,s0y9).
qpair(5,8,s2y6y9).
qpair(5,8,s1y4).
qpair(5,8,s0y6).
qpair(5,8,s2y3).
qpair(5,8,s2y9).
qpair(5,8,s1y3y4).
qpair(5,8,empty).
qpair(5,8,s6y9).
qpair(5,9,s7).
qpair(5,9,s2).
qpair(5,9,s3).
qpair(5,9,s1y4).
qpair(6,7,s2).
qpair(6,7,s5).
qpair(6,7,s0y1y3).
qpair(6,7,s8).
qpair(6,7,s0y2).
qpair(6,7,s8y9).
qpair(6,7,s0y1y9).
qpair(6,7,s0y5).
qpair(6,7,s0y3y9).
qpair(6,7,s1y3).
qpair(6,7,s1y9).
qpair(6,7,s1y3y9).
qpair(6,7,s2y8).
qpair(6,7,s1).
qpair(6,7,s3y9).
qpair(6,7,s5y9).
qpair(6,7,s0y1y8).
qpair(6,7,s0y3y8).
qpair(6,7,s0y1).
qpair(6,7,s1y3y8).
qpair(6,7,s1y5).
qpair(6,7,s1y8).
qpair(6,7,s3y8y9).
qpair(6,7,s0).
qpair(6,7,s3).
qpair(6,7,s9).
qpair(6,7,s3y5).
qpair(6,7,s3y8).
qpair(6,7,s0y3).
qpair(6,7,s0y9).
qpair(6,7,s0y1y8y9).
qpair(6,7,s0y8y9).
qpair(6,7,empty).
qpair(6,7,s1y8y9).
qpair(6,8,s2).
qpair(6,8,s0y2y3y5).
qpair(6,8,s5).
qpair(6,8,s3y7).
qpair(6,8,s0y1y3).
qpair(6,8,s3y5y9).
qpair(6,8,s5y7).
qpair(6,8,s3y7y9).
qpair(6,8,s0y2).
qpair(6,8,s0y1y9).
qpair(6,8,s0y5).
qpair(6,8,s2y5).
qpair(6,8,s1y9).
qpair(6,8,s1y3y9).
qpair(6,8,s2y3y9).
qpair(6,8,s0y2y3y5y9).
qpair(6,8,s4).
qpair(6,8,s0y5y9).
qpair(6,8,s1).
qpair(6,8,s7).
qpair(6,8,s0y7y9).
qpair(6,8,s3y9).
qpair(6,8,s5y9).
qpair(6,8,s0y3y5).
qpair(6,8,s1y7y9).
qpair(6,8,s2y5y9).
qpair(6,8,s0y1y7y9).
qpair(6,8,s0y1).
qpair(6,8,s0y7).
qpair(6,8,s0y3y5y9).
qpair(6,8,s2y7).
qpair(6,8,s1y5).
qpair(6,8,s0y2y5y9).
qpair(6,8,s0y2y3).
qpair(6,8,s7y9).
qpair(6,8,s2y3y5y9).
qpair(6,8,s0y2y9).
qpair(6,8,s0).
qpair(6,8,s9).
qpair(6,8,s3y5).
qpair(6,8,s0y2y3y9).
qpair(6,8,s0y1y7).
qpair(6,8,s0y9).
qpair(6,8,s2y9).
qpair(6,8,s1y3y7).
qpair(6,8,s0y2y5).
qpair(6,8,empty).
qpair(6,9,s2).
qpair(6,9,s5).
qpair(6,9,s0y1y3).
qpair(6,9,s8).
qpair(6,9,s5y7).
qpair(6,9,s0y5).
qpair(6,9,s0y8).
qpair(6,9,s1y3).
qpair(6,9,s4).
qpair(6,9,s1).
qpair(6,9,s7).
qpair(6,9,s3y7y8).
qpair(6,9,s0y1y8).
qpair(6,9,s0y1).
qpair(6,9,s0y7).
qpair(6,9,s1y3y8).
qpair(6,9,s1y5).
qpair(6,9,s1y8).
qpair(6,9,s0).
qpair(6,9,s0y7y8).
qpair(6,9,s3).
qpair(6,9,s3y5).
qpair(6,9,s1y7y8).
qpair(6,9,s3y8).
qpair(6,9,s0y3y7).
qpair(6,9,s5y8).
qpair(6,9,s0y3).
qpair(6,9,s1y7).
qpair(6,9,s1y3y7).
qpair(6,9,empty).
qpair(6,9,s7y8).
qpair(7,8,s4).
qpair(7,9,s1).
qpair(7,9,s5).
qpair(8,9,s2).
qpair(8,9,s0y2y3y5).
qpair(8,9,s5).
qpair(8,9,s3y7).
qpair(8,9,s5y7).
qpair(8,9,s0y2).
qpair(8,9,s0y5).
qpair(8,9,s0y2y3y5y6).
qpair(8,9,s2y3y6).
qpair(8,9,s0y5y6).
qpair(8,9,s4).
qpair(8,9,s1).
qpair(8,9,s7).
qpair(8,9,s5y6).
qpair(8,9,s3y6).
qpair(8,9,s0y3y5).
qpair(8,9,s0y3y5y6).
qpair(8,9,s2y7).
qpair(8,9,s0y2y3).
qpair(8,9,s2y3y5).
qpair(8,9,s0y2y6).
qpair(8,9,s6y7).
qpair(8,9,s2y3y5y6).
qpair(8,9,s0).
qpair(8,9,s0y2y3y6).
qpair(8,9,s3y5).
qpair(8,9,s6).
qpair(8,9,s0y3).
qpair(8,9,s2y3).
qpair(8,9,s0y2y5y6).
qpair(8,9,s2y6).
qpair(8,9,s0y2y5).
qpair(8,9,empty).
qpair(0,1,s2y7y9).
qpair(0,1,s3y4y5y6).
qpair(0,1,s2).
qpair(0,1,s2y7).
qpair(0,1,s6y9).
qpair(0,1,s7).
qpair(0,1,s6y8y9).
qpair(0,1,s3y6).
qpair(0,1,s2y6y7y9).
qpair(0,1,s3).
qpair(0,1,s2y8).
qpair(0,2,s6y7y8y9).
qpair(0,2,s6y8).
qpair(0,2,s6y9).
qpair(0,2,s7y8).
qpair(0,3,s5).
qpair(0,3,s2y7y8).
qpair(0,3,s7y9).
qpair(0,3,s8y9).
qpair(0,3,s1y4y8).
qpair(0,4,empty).
qpair(0,5,s2y7y9).
qpair(0,5,s2y6y7y8).
qpair(0,5,s2y8y9).
qpair(0,5,s2y7y8y9).
qpair(0,5,s6y7y8y9).
qpair(0,5,s1y8).
qpair(0,5,s2y9).
qpair(0,5,s6y8y9).
qpair(0,5,s8y9).
qpair(0,5,s2y6y8y9).
qpair(0,5,s7y8y9).
qpair(0,5,s7y8).
qpair(0,5,s2y8).
qpair(0,6,s4).
qpair(0,6,s7y9).
qpair(0,6,s1y7).
qpair(0,6,s3).
qpair(0,6,s1y3).
qpair(0,6,s7y8).
qpair(0,7,s6y8).
qpair(0,7,s9).
qpair(0,8,s5).
qpair(0,8,s3y5y6y9).
qpair(0,8,s2y6y9).
qpair(0,8,s2y3).
qpair(0,8,s2y5y9).
qpair(0,8,s6y9).
qpair(0,9,s7).
qpair(0,9,s2).
qpair(0,9,s2y7).
qpair(0,9,s2y6y8).
qpair(1,3,empty).
qpair(1,3,s4).
qpair(1,5,s6y8).
qpair(1,5,s4y6).
qpair(1,5,s3y6).
qpair(1,6,s0y9).
qpair(1,6,s4).
qpair(1,6,s8y9).
qpair(1,6,s0y8y9).
qpair(1,6,s7y8y9).
qpair(1,6,s9).
qpair(1,7,s2).
qpair(1,7,s3y4).
qpair(1,7,s0y2y6).
qpair(1,7,s0y2).
qpair(1,7,s4y5).
qpair(1,7,s3y4y5).
qpair(1,7,s0).
qpair(1,7,s3).
qpair(1,7,s3y5).
qpair(1,9,s2).
qpair(1,9,s0y6).
qpair(2,3,s0y7y8).
qpair(2,3,s1y8).
qpair(2,3,s4y8).
qpair(2,3,s0y8).
qpair(2,3,s9).
qpair(2,3,s7y8).
qpair(2,4,empty).
qpair(2,5,s0y7).
qpair(2,5,s1y8).
qpair(2,5,s4y8).
qpair(2,5,s0y8).
qpair(2,5,s9).
qpair(2,5,s7y8).
qpair(2,5,s1y4y8).
qpair(2,7,s9).
qpair(2,8,s6).
qpair(2,8,s0y3y5y6).
qpair(2,8,s3y7).
qpair(2,8,s0y9).
qpair(2,8,s0y5y9).
qpair(2,8,s0y6).
qpair(2,8,s7y9).
qpair(2,8,s0y5).
qpair(2,8,s3y6).
qpair(2,8,s5y9).
qpair(2,9,s3).
qpair(2,9,s5).
qpair(3,4,s8).
qpair(3,5,s0).
qpair(3,5,s9).
qpair(3,5,s1y6y8).
qpair(3,6,s1y5).
qpair(3,7,s0).
qpair(3,7,s8).
qpair(3,7,s9).
qpair(3,7,s4).
qpair(3,8,s2).
qpair(3,8,s2y7).
qpair(3,8,s1y5).
qpair(3,8,s0y2y6y9).
qpair(3,8,s7).
qpair(3,8,s0y2y5y6).
qpair(3,8,s6y9).
qpair(3,8,s9).
qpair(3,8,s2y5y6).
qpair(3,9,s6).
qpair(3,9,s0y6y8).
qpair(3,9,s6y8).
qpair(3,9,s8).
qpair(3,9,s1y8).
qpair(3,9,s0y6).
qpair(3,9,s4y8).
qpair(3,9,s0).
qpair(3,9,s0y8).
qpair(3,9,s1y4y8).
qpair(4,5,s1y3).
qpair(4,5,s3y6).
qpair(4,6,s0y1).
qpair(4,6,s0y1y7).
qpair(4,6,s1y5).
qpair(4,6,s1y3y5).
qpair(4,6,s7y8y9).
qpair(4,6,s1y7).
qpair(4,6,s0y8y9).
qpair(4,6,s0y1y8).
qpair(4,6,s0y8).
qpair(4,6,s1y7y9).
qpair(4,7,empty).
qpair(4,8,s7).
qpair(4,9,empty).
qpair(5,7,s8).
qpair(5,8,s2y7).
qpair(5,8,s0y2).
qpair(5,8,s2y6).
qpair(5,8,s3y6).
qpair(5,8,s0y3y9).
qpair(5,8,s0y6y9).
qpair(5,9,s0y6y8).
qpair(5,9,s6y8).
qpair(5,9,s1y8).
qpair(5,9,s0y6).
qpair(5,9,s4y8).
qpair(5,9,s0y8).
qpair(5,9,s1y4y8).
qpair(6,7,s0y8).
qpair(6,7,s4).
qpair(6,7,s5y8).
qpair(6,8,s0y3y7).
qpair(6,8,s0y3).
qpair(6,8,s2y3y5).
qpair(6,8,s2y3).
qpair(6,8,s1y7).
qpair(6,8,s3).
qpair(6,8,s0y3y9).
qpair(6,8,s1y3).
qpair(6,9,s3y7).
qpair(6,9,s0y1y7y8).
qpair(6,9,s0y3y8).
qpair(6,9,s0y1y7).
qpair(7,9,s3).
qpair(8,9,s0y7).
qpair(8,9,s3y5y6).
qpair(8,9,s0y6).
qpair(8,9,s0y3y6).
qpair(8,9,s2y5).
qpair(8,9,s3).
qpair(8,9,s2y5y6).

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
in(0,s0y1y8y7).
in(0,s0y1y9y7).
in(0,s0y8y3).
in(0,s0y8y6).
in(0,s0y8y7).
in(0,s0y8y9y7).
in(0,s0y9y2).
in(0,s0y9y2y3).
in(0,s0y9y2y5).
in(0,s0y9y2y6).
in(0,s0y9y3).
in(0,s0y9y3y5).
in(0,s0y9y3y6).
in(0,s0y9y5).
in(0,s0y9y5y6).
in(0,s0y9y6).
in(0,s0y9y7).
in(1,s0y1y8y7).
in(1,s0y1y9y7).
in(1,s1y9y7).
in(1,s8y1).
in(1,s8y1y3).
in(1,s8y1y4).
in(1,s8y1y5).
in(1,s8y1y6).
in(1,s8y1y7).
in(1,s8y1y9).
in(1,s8y1y9y7).
in(2,s0y9y2).
in(2,s0y9y2y3).
in(2,s0y9y2y5).
in(2,s0y9y2y6).
in(2,s8y2).
in(2,s8y2y6).
in(2,s8y2y6y7).
in(2,s8y2y7).
in(2,s8y9y2).
in(2,s8y9y2y6).
in(2,s8y9y2y7).
in(2,s9y2).
in(2,s9y2y3).
in(2,s9y2y3y5).
in(2,s9y2y3y6).
in(2,s9y2y5).
in(2,s9y2y5y6).
in(2,s9y2y6).
in(2,s9y2y6y7).
in(2,s9y2y7).
in(3,s0y8y3).
in(3,s0y9y2y3).
in(3,s0y9y3).
in(3,s0y9y3y5).
in(3,s0y9y3y6).
in(3,s8y1y3).
in(3,s8y3).
in(3,s8y3y4).
in(3,s8y3y7).
in(3,s8y9y3).
in(3,s9y2y3).
in(3,s9y2y3y5).
in(3,s9y2y3y6).
in(3,s9y3).
in(3,s9y3y5).
in(3,s9y3y5y6).
in(3,s9y3y6).
in(3,s9y3y7).
in(4,s8y1y4).
in(4,s8y3y4).
in(4,s8y4).
in(4,s8y4y5).
in(4,s8y4y6).
in(5,s0y9y2y5).
in(5,s0y9y3y5).
in(5,s0y9y5).
in(5,s0y9y5y6).
in(5,s8y1y5).
in(5,s8y4y5).
in(5,s8y5).
in(5,s8y5y6).
in(5,s9y2y3y5).
in(5,s9y2y5).
in(5,s9y2y5y6).
in(5,s9y3y5).
in(5,s9y3y5y6).
in(5,s9y5).
in(5,s9y5y6).
in(6,s0y8y6).
in(6,s0y9y2y6).
in(6,s0y9y3y6).
in(6,s0y9y5y6).
in(6,s0y9y6).
in(6,s8y1y6).
in(6,s8y2y6).
in(6,s8y2y6y7).
in(6,s8y4y6).
in(6,s8y5y6).
in(6,s8y6).
in(6,s8y6y7).
in(6,s8y9y2y6).
in(6,s8y9y6).
in(6,s8y9y6y7).
in(6,s9y2y3y6).
in(6,s9y2y5y6).
in(6,s9y2y6).
in(6,s9y2y6y7).
in(6,s9y3y5y6).
in(6,s9y3y6).
in(6,s9y5y6).
in(6,s9y6).
in(6,s9y6y7).
in(7,s0y1y8y7).
in(7,s0y1y9y7).
in(7,s0y8y7).
in(7,s0y8y9y7).
in(7,s0y9y7).
in(7,s1y9y7).
in(7,s8y1y7).
in(7,s8y1y9y7).
in(7,s8y2y6y7).
in(7,s8y2y7).
in(7,s8y3y7).
in(7,s8y6y7).
in(7,s8y7).
in(7,s8y9y2y7).
in(7,s8y9y6y7).
in(7,s8y9y7).
in(7,s9y2y6y7).
in(7,s9y2y7).
in(7,s9y3y7).
in(7,s9y6y7).
in(7,s9y7).
in(8,s0y1y8y7).
in(8,s0y8y3).
in(8,s0y8y6).
in(8,s0y8y7).
in(8,s0y8y9y7).
in(8,s8y1).
in(8,s8y1y3).
in(8,s8y1y4).
in(8,s8y1y5).
in(8,s8y1y6).
in(8,s8y1y7).
in(8,s8y1y9).
in(8,s8y1y9y7).
in(8,s8y2).
in(8,s8y2y6).
in(8,s8y2y6y7).
in(8,s8y2y7).
in(8,s8y3).
in(8,s8y3y4).
in(8,s8y3y7).
in(8,s8y4).
in(8,s8y4y5).
in(8,s8y4y6).
in(8,s8y5).
in(8,s8y5y6).
in(8,s8y6).
in(8,s8y6y7).
in(8,s8y7).
in(8,s8y9y2).
in(8,s8y9y2y6).
in(8,s8y9y2y7).
in(8,s8y9y3).
in(8,s8y9y6).
in(8,s8y9y6y7).
in(8,s8y9y7).
in(9,s0y1y9y7).
in(9,s0y8y9y7).
in(9,s0y9y2).
in(9,s0y9y2y3).
in(9,s0y9y2y5).
in(9,s0y9y2y6).
in(9,s0y9y3).
in(9,s0y9y3y5).
in(9,s0y9y3y6).
in(9,s0y9y5).
in(9,s0y9y5y6).
in(9,s0y9y6).
in(9,s0y9y7).
in(9,s1y9y7).
in(9,s8y1y9).
in(9,s8y1y9y7).
in(9,s8y9y2).
in(9,s8y9y2y6).
in(9,s8y9y2y7).
in(9,s8y9y3).
in(9,s8y9y6).
in(9,s8y9y6y7).
in(9,s8y9y7).
in(9,s9y2).
in(9,s9y2y3).
in(9,s9y2y3y5).
in(9,s9y2y3y6).
in(9,s9y2y5).
in(9,s9y2y5y6).
in(9,s9y2y6).
in(9,s9y2y6y7).
in(9,s9y2y7).
in(9,s9y3).
in(9,s9y3y5).
in(9,s9y3y5y6).
in(9,s9y3y6).
in(9,s9y3y7).
in(9,s9y5).
in(9,s9y5y6).
in(9,s9y6).
in(9,s9y6y7).
in(9,s9y7).


% ===== Base program loaded from: results/wc_sweep_10_2026_20260212_184451/rep5/lps/abapc_inc_full_base_rep5_optN.lp =====
% ===== Base program sha256 (raw): 4b1053ab743466404df449665292175ba89ed0006ce96c58561d1f5e8aef84b5 =====
% ===== Base program sha256 (normalized): 877200c70dfd0555ce487e74247846309a9a840564a15dd734fe6920b12dac98 =====
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

% ===== Adorned Facts =====
ext_indep(0,1,s2):-mus(1).
ext_indep(0,1,s3):-mus(2).
ext_dep(0,1,s4):-mus(3).
ext_dep(0,1,s5):-mus(4).
ext_dep(0,1,s6):-mus(5).
ext_indep(0,1,s7):-mus(6).
ext_dep(0,1,s8):-mus(7).
ext_dep(0,1,s9):-mus(8).
ext_dep(0,1,s2y6):-mus(9).
ext_indep(0,1,s2y7):-mus(10).
ext_indep(0,1,s8y2):-mus(11).
ext_dep(0,1,s9y2):-mus(12).
ext_dep(0,1,s3y4):-mus(13).
ext_dep(0,1,s3y5):-mus(14).
ext_indep(0,1,s3y6):-mus(15).
ext_dep(0,1,s4y5):-mus(16).
ext_dep(0,1,s4y6):-mus(17).
ext_dep(0,1,s5y6):-mus(18).
ext_dep(0,1,s6y7):-mus(19).
ext_dep(0,1,s8y6):-mus(20).
ext_indep(0,1,s9y6):-mus(21).
ext_dep(0,1,s8y7):-mus(22).
ext_dep(0,1,s9y7):-mus(23).
ext_dep(0,1,s8y9):-mus(24).
ext_dep(0,1,s2y6y7):-mus(25).
ext_dep(0,1,s8y2y6):-mus(26).
ext_dep(0,1,s9y2y6):-mus(27).
ext_dep(0,1,s8y2y7):-mus(28).
ext_indep(0,1,s9y2y7):-mus(29).
ext_dep(0,1,s8y9y2):-mus(30).
ext_dep(0,1,s3y4y5):-mus(31).
ext_dep(0,1,s3y4y6):-mus(32).
ext_dep(0,1,s3y5y6):-mus(33).
ext_dep(0,1,s4y5y6):-mus(34).
ext_dep(0,1,s8y6y7):-mus(35).
ext_dep(0,1,s9y6y7):-mus(36).
ext_indep(0,1,s8y9y6):-mus(37).
ext_dep(0,1,s8y9y7):-mus(38).
ext_dep(0,1,s8y2y6y7):-mus(39).
ext_indep(0,1,s9y2y6y7):-mus(40).
ext_dep(0,1,s8y9y2y6):-mus(41).
ext_dep(0,1,s8y9y2y7):-mus(42).
ext_indep(0,1,s3y4y5y6):-mus(43).
ext_dep(0,1,s8y9y6y7):-mus(44).
ext_dep(0,1,s2y6y7y8y9):-mus(45).
ext_dep(0,2,empty):-mus(46).
ext_dep(0,2,s3):-mus(47).
ext_dep(0,2,s5):-mus(48).
ext_dep(0,2,s6):-mus(49).
ext_dep(0,2,s7):-mus(50).
ext_dep(0,2,s8):-mus(51).
ext_dep(0,2,s9):-mus(52).
ext_dep(0,2,s6y7):-mus(53).
ext_indep(0,2,s8y6):-mus(54).
ext_indep(0,2,s9y6):-mus(55).
ext_indep(0,2,s8y7):-mus(56).
ext_dep(0,2,s9y7):-mus(57).
ext_dep(0,2,s8y9):-mus(58).
ext_dep(0,2,s8y6y7):-mus(59).
ext_dep(0,2,s9y6y7):-mus(60).
ext_dep(0,2,s8y9y6):-mus(61).
ext_dep(0,2,s8y9y7):-mus(62).
ext_indep(0,2,s8y9y6y7):-mus(63).
ext_indep(0,3,s5):-mus(64).
ext_dep(0,3,s1y4):-mus(65).
ext_dep(0,3,s8y1):-mus(66).
ext_dep(0,3,s2y6):-mus(67).
ext_dep(0,3,s2y7):-mus(68).
ext_dep(0,3,s8y2):-mus(69).
ext_dep(0,3,s9y2):-mus(70).
ext_dep(0,3,s8y4):-mus(71).
ext_dep(0,3,s6y7):-mus(72).
ext_dep(0,3,s8y6):-mus(73).
ext_dep(0,3,s9y6):-mus(74).
ext_dep(0,3,s8y7):-mus(75).
ext_indep(0,3,s9y7):-mus(76).
ext_indep(0,3,s8y9):-mus(77).
ext_indep(0,3,s8y1y4):-mus(78).
ext_dep(0,3,s2y6y7):-mus(79).
ext_dep(0,3,s8y2y6):-mus(80).
ext_dep(0,3,s9y2y6):-mus(81).
ext_indep(0,3,s8y2y7):-mus(82).
ext_dep(0,3,s9y2y7):-mus(83).
ext_dep(0,3,s8y9y2):-mus(84).
ext_dep(0,3,s8y6y7):-mus(85).
ext_dep(0,3,s9y6y7):-mus(86).
ext_dep(0,3,s8y9y6):-mus(87).
ext_dep(0,3,s8y9y7):-mus(88).
ext_dep(0,3,s8y2y6y7):-mus(89).
ext_dep(0,3,s9y2y6y7):-mus(90).
ext_dep(0,3,s8y9y2y6):-mus(91).
ext_dep(0,3,s8y9y2y7):-mus(92).
ext_dep(0,3,s8y9y6y7):-mus(93).
ext_dep(0,3,s2y6y7y8y9):-mus(94).
ext_indep(0,4,empty):-mus(95).
ext_dep(0,5,s3):-mus(96).
ext_dep(0,5,s1y4):-mus(97).
ext_indep(0,5,s8y1):-mus(98).
ext_dep(0,5,s2y6):-mus(99).
ext_dep(0,5,s2y7):-mus(100).
ext_indep(0,5,s8y2):-mus(101).
ext_indep(0,5,s9y2):-mus(102).
ext_dep(0,5,s8y4):-mus(103).
ext_dep(0,5,s6y7):-mus(104).
ext_dep(0,5,s8y6):-mus(105).
ext_dep(0,5,s9y6):-mus(106).
ext_indep(0,5,s8y7):-mus(107).
ext_dep(0,5,s9y7):-mus(108).
ext_indep(0,5,s8y9):-mus(109).
ext_dep(0,5,s8y1y4):-mus(110).
ext_dep(0,5,s2y6y7):-mus(111).
ext_dep(0,5,s8y2y6):-mus(112).
ext_dep(0,5,s9y2y6):-mus(113).
ext_dep(0,5,s8y2y7):-mus(114).
ext_indep(0,5,s9y2y7):-mus(115).
ext_indep(0,5,s8y9y2):-mus(116).
ext_dep(0,5,s8y6y7):-mus(117).
ext_dep(0,5,s9y6y7):-mus(118).
ext_indep(0,5,s8y9y6):-mus(119).
ext_indep(0,5,s8y9y7):-mus(120).
ext_indep(0,5,s8y2y6y7):-mus(121).
ext_dep(0,5,s9y2y6y7):-mus(122).
ext_indep(0,5,s8y9y2y6):-mus(123).
ext_indep(0,5,s8y9y2y7):-mus(124).
ext_indep(0,5,s8y9y6y7):-mus(125).
ext_dep(0,5,s2y6y7y8y9):-mus(126).
ext_dep(0,6,empty):-mus(127).
ext_dep(0,6,s1):-mus(128).
ext_dep(0,6,s2):-mus(129).
ext_indep(0,6,s3):-mus(130).
ext_indep(0,6,s4):-mus(131).
ext_dep(0,6,s5):-mus(132).
ext_dep(0,6,s7):-mus(133).
ext_dep(0,6,s8):-mus(134).
ext_dep(0,6,s9):-mus(135).
ext_indep(0,6,s1y3):-mus(136).
ext_dep(0,6,s1y5):-mus(137).
ext_indep(0,6,s1y7):-mus(138).
ext_dep(0,6,s8y1):-mus(139).
ext_dep(0,6,s1y9):-mus(140).
ext_dep(0,6,s2y7):-mus(141).
ext_dep(0,6,s8y2):-mus(142).
ext_dep(0,6,s9y2):-mus(143).
ext_dep(0,6,s3y5):-mus(144).
ext_dep(0,6,s3y7):-mus(145).
ext_dep(0,6,s8y3):-mus(146).
ext_dep(0,6,s9y3):-mus(147).
ext_dep(0,6,s5y7):-mus(148).
ext_dep(0,6,s8y5):-mus(149).
ext_dep(0,6,s9y5):-mus(150).
ext_indep(0,6,s8y7):-mus(151).
ext_indep(0,6,s9y7):-mus(152).
ext_dep(0,6,s8y9):-mus(153).
ext_dep(0,6,s1y3y7):-mus(154).
ext_dep(0,6,s8y1y3):-mus(155).
ext_dep(0,6,s1y3y9):-mus(156).
ext_dep(0,6,s8y1y7):-mus(157).
ext_dep(0,6,s1y9y7):-mus(158).
ext_dep(0,6,s8y1y9):-mus(159).
ext_dep(0,6,s8y2y7):-mus(160).
ext_dep(0,6,s9y2y7):-mus(161).
ext_dep(0,6,s8y9y2):-mus(162).
ext_dep(0,6,s8y3y7):-mus(163).
ext_dep(0,6,s9y3y7):-mus(164).
ext_dep(0,6,s8y9y3):-mus(165).
ext_dep(0,6,s8y9y7):-mus(166).
ext_dep(0,6,s8y1y9y7):-mus(167).
ext_dep(0,6,s8y9y2y7):-mus(168).
ext_dep(0,7,empty):-mus(169).
ext_dep(0,7,s2):-mus(170).
ext_dep(0,7,s3):-mus(171).
ext_dep(0,7,s5):-mus(172).
ext_dep(0,7,s6):-mus(173).
ext_dep(0,7,s8):-mus(174).
ext_indep(0,7,s9):-mus(175).
ext_dep(0,7,s2y6):-mus(176).
ext_dep(0,7,s8y2):-mus(177).
ext_dep(0,7,s9y2):-mus(178).
ext_indep(0,7,s8y6):-mus(179).
ext_dep(0,7,s9y6):-mus(180).
ext_dep(0,7,s8y9):-mus(181).
ext_dep(0,7,s8y2y6):-mus(182).
ext_dep(0,7,s9y2y6):-mus(183).
ext_dep(0,7,s8y9y2):-mus(184).
ext_dep(0,7,s8y9y6):-mus(185).
ext_dep(0,7,s8y9y2y6):-mus(186).
ext_dep(0,8,empty):-mus(187).
ext_dep(0,8,s2):-mus(188).
ext_dep(0,8,s3):-mus(189).
ext_dep(0,8,s4):-mus(190).
ext_indep(0,8,s5):-mus(191).
ext_dep(0,8,s6):-mus(192).
ext_dep(0,8,s7):-mus(193).
ext_dep(0,8,s9):-mus(194).
ext_indep(0,8,s2y3):-mus(195).
ext_dep(0,8,s2y5):-mus(196).
ext_dep(0,8,s2y6):-mus(197).
ext_dep(0,8,s2y7):-mus(198).
ext_dep(0,8,s9y2):-mus(199).
ext_dep(0,8,s3y5):-mus(200).
ext_dep(0,8,s3y6):-mus(201).
ext_dep(0,8,s3y7):-mus(202).
ext_dep(0,8,s9y3):-mus(203).
ext_dep(0,8,s5y6):-mus(204).
ext_dep(0,8,s5y7):-mus(205).
ext_dep(0,8,s9y5):-mus(206).
ext_dep(0,8,s6y7):-mus(207).
ext_indep(0,8,s9y6):-mus(208).
ext_dep(0,8,s9y7):-mus(209).
ext_dep(0,8,s2y3y5):-mus(210).
ext_dep(0,8,s2y3y6):-mus(211).
ext_dep(0,8,s9y2y3):-mus(212).
ext_dep(0,8,s2y5y6):-mus(213).
ext_indep(0,8,s9y2y5):-mus(214).
ext_dep(0,8,s2y6y7):-mus(215).
ext_indep(0,8,s9y2y6):-mus(216).
ext_dep(0,8,s9y2y7):-mus(217).
ext_dep(0,8,s3y5y6):-mus(218).
ext_dep(0,8,s9y3y5):-mus(219).
ext_dep(0,8,s9y3y6):-mus(220).
ext_dep(0,8,s9y5y6):-mus(221).
ext_dep(0,8,s9y6y7):-mus(222).
ext_dep(0,8,s2y3y5y6):-mus(223).
ext_dep(0,8,s9y2y3y5):-mus(224).
ext_dep(0,8,s9y2y3y6):-mus(225).
ext_dep(0,8,s9y2y5y6):-mus(226).
ext_dep(0,8,s9y2y6y7):-mus(227).
ext_indep(0,8,s9y3y5y6):-mus(228).
ext_dep(0,8,s2y3y5y6y9):-mus(229).
ext_dep(0,9,empty):-mus(230).
ext_dep(0,9,s1):-mus(231).
ext_indep(0,9,s2):-mus(232).
ext_dep(0,9,s3):-mus(233).
ext_dep(0,9,s5):-mus(234).
ext_dep(0,9,s6):-mus(235).
ext_indep(0,9,s7):-mus(236).
ext_dep(0,9,s8):-mus(237).
ext_dep(0,9,s2y6):-mus(238).
ext_indep(0,9,s2y7):-mus(239).
ext_dep(0,9,s8y2):-mus(240).
ext_dep(0,9,s6y7):-mus(241).
ext_dep(0,9,s8y6):-mus(242).
ext_dep(0,9,s8y7):-mus(243).
ext_dep(0,9,s2y6y7):-mus(244).
ext_indep(0,9,s8y2y6):-mus(245).
ext_dep(0,9,s8y2y7):-mus(246).
ext_dep(0,9,s8y6y7):-mus(247).
ext_dep(0,9,s8y2y6y7):-mus(248).
ext_dep(1,2,empty):-mus(249).
ext_indep(1,3,empty):-mus(250).
ext_dep(1,3,s0):-mus(251).
ext_dep(1,3,s2):-mus(252).
ext_indep(1,3,s4):-mus(253).
ext_dep(1,3,s5):-mus(254).
ext_dep(1,3,s6):-mus(255).
ext_dep(1,3,s7):-mus(256).
ext_dep(1,3,s8):-mus(257).
ext_dep(1,3,s9):-mus(258).
ext_dep(1,3,s4y5):-mus(259).
ext_dep(1,3,s4y6):-mus(260).
ext_dep(1,3,s8y4):-mus(261).
ext_dep(1,3,s5y6):-mus(262).
ext_dep(1,3,s8y5):-mus(263).
ext_dep(1,3,s8y6):-mus(264).
ext_dep(1,3,s4y5y6):-mus(265).
ext_dep(1,3,s8y4y5):-mus(266).
ext_dep(1,3,s8y4y6):-mus(267).
ext_dep(1,3,s8y5y6):-mus(268).
ext_dep(1,4,empty):-mus(269).
ext_dep(1,4,s3):-mus(270).
ext_dep(1,4,s5):-mus(271).
ext_dep(1,4,s6):-mus(272).
ext_dep(1,4,s8):-mus(273).
ext_dep(1,4,s9):-mus(274).
ext_dep(1,4,s3y5):-mus(275).
ext_dep(1,4,s3y6):-mus(276).
ext_dep(1,4,s5y6):-mus(277).
ext_dep(1,4,s3y5y6):-mus(278).
ext_dep(1,5,empty):-mus(279).
ext_dep(1,5,s0):-mus(280).
ext_dep(1,5,s2):-mus(281).
ext_dep(1,5,s3):-mus(282).
ext_dep(1,5,s4):-mus(283).
ext_dep(1,5,s6):-mus(284).
ext_dep(1,5,s7):-mus(285).
ext_dep(1,5,s8):-mus(286).
ext_dep(1,5,s9):-mus(287).
ext_dep(1,5,s3y4):-mus(288).
ext_indep(1,5,s3y6):-mus(289).
ext_dep(1,5,s8y3):-mus(290).
ext_indep(1,5,s4y6):-mus(291).
ext_dep(1,5,s8y4):-mus(292).
ext_indep(1,5,s8y6):-mus(293).
ext_dep(1,5,s3y4y6):-mus(294).
ext_dep(1,5,s8y3y4):-mus(295).
ext_dep(1,6,empty):-mus(296).
ext_dep(1,6,s0):-mus(297).
ext_dep(1,6,s3):-mus(298).
ext_indep(1,6,s4):-mus(299).
ext_dep(1,6,s5):-mus(300).
ext_dep(1,6,s7):-mus(301).
ext_dep(1,6,s8):-mus(302).
ext_indep(1,6,s9):-mus(303).
ext_dep(1,6,s0y3):-mus(304).
ext_dep(1,6,s0y5):-mus(305).
ext_dep(1,6,s0y7):-mus(306).
ext_dep(1,6,s0y8):-mus(307).
ext_indep(1,6,s0y9):-mus(308).
ext_dep(1,6,s3y4):-mus(309).
ext_dep(1,6,s3y5):-mus(310).
ext_dep(1,6,s3y7):-mus(311).
ext_dep(1,6,s8y3):-mus(312).
ext_dep(1,6,s9y3):-mus(313).
ext_dep(1,6,s4y5):-mus(314).
ext_dep(1,6,s5y7):-mus(315).
ext_dep(1,6,s8y5):-mus(316).
ext_dep(1,6,s9y5):-mus(317).
ext_dep(1,6,s8y7):-mus(318).
ext_dep(1,6,s9y7):-mus(319).
ext_indep(1,6,s8y9):-mus(320).
ext_dep(1,6,s0y3y7):-mus(321).
ext_dep(1,6,s0y8y3):-mus(322).
ext_dep(1,6,s0y9y3):-mus(323).
ext_dep(1,6,s0y8y7):-mus(324).
ext_dep(1,6,s0y9y7):-mus(325).
ext_indep(1,6,s0y8y9):-mus(326).
ext_dep(1,6,s3y4y5):-mus(327).
ext_dep(1,6,s8y3y7):-mus(328).
ext_dep(1,6,s9y3y7):-mus(329).
ext_dep(1,6,s8y9y3):-mus(330).
ext_indep(1,6,s8y9y7):-mus(331).
ext_dep(1,6,s0y8y9y7):-mus(332).
ext_indep(1,7,s0):-mus(333).
ext_indep(1,7,s2):-mus(334).
ext_indep(1,7,s3):-mus(335).
ext_dep(1,7,s4):-mus(336).
ext_dep(1,7,s5):-mus(337).
ext_dep(1,7,s6):-mus(338).
ext_indep(1,7,s0y2):-mus(339).
ext_dep(1,7,s0y6):-mus(340).
ext_dep(1,7,s2y6):-mus(341).
ext_indep(1,7,s3y4):-mus(342).
ext_indep(1,7,s3y5):-mus(343).
ext_dep(1,7,s3y6):-mus(344).
ext_indep(1,7,s4y5):-mus(345).
ext_dep(1,7,s4y6):-mus(346).
ext_dep(1,7,s5y6):-mus(347).
ext_indep(1,7,s0y2y6):-mus(348).
ext_indep(1,7,s3y4y5):-mus(349).
ext_dep(1,7,s3y4y6):-mus(350).
ext_dep(1,7,s3y5y6):-mus(351).
ext_dep(1,7,s4y5y6):-mus(352).
ext_dep(1,7,s3y4y5y6):-mus(353).
ext_indep(1,9,s2):-mus(354).
ext_dep(1,9,s7):-mus(355).
ext_indep(1,9,s0y6):-mus(356).
ext_dep(1,9,s0y8):-mus(357).
ext_dep(1,9,s3y4):-mus(358).
ext_dep(1,9,s3y5):-mus(359).
ext_dep(1,9,s3y6):-mus(360).
ext_dep(1,9,s4y5):-mus(361).
ext_dep(1,9,s4y6):-mus(362).
ext_dep(1,9,s5y6):-mus(363).
ext_dep(1,9,s8y6):-mus(364).
ext_dep(1,9,s0y8y6):-mus(365).
ext_dep(1,9,s3y4y5):-mus(366).
ext_dep(1,9,s3y4y6):-mus(367).
ext_dep(1,9,s3y5y6):-mus(368).
ext_dep(1,9,s4y5y6):-mus(369).
ext_dep(1,9,s3y4y5y6):-mus(370).
ext_dep(2,3,s5):-mus(371).
ext_dep(2,3,s6):-mus(372).
ext_indep(2,3,s9):-mus(373).
ext_dep(2,3,s0y7):-mus(374).
ext_indep(2,3,s0y8):-mus(375).
ext_dep(2,3,s1y4):-mus(376).
ext_indep(2,3,s8y1):-mus(377).
ext_indep(2,3,s8y4):-mus(378).
ext_indep(2,3,s8y7):-mus(379).
ext_indep(2,3,s0y8y7):-mus(380).
ext_dep(2,3,s8y1y4):-mus(381).
ext_indep(2,4,empty):-mus(382).
ext_dep(2,5,s3):-mus(383).
ext_dep(2,5,s6):-mus(384).
ext_indep(2,5,s9):-mus(385).
ext_indep(2,5,s0y7):-mus(386).
ext_indep(2,5,s0y8):-mus(387).
ext_dep(2,5,s1y4):-mus(388).
ext_indep(2,5,s8y1):-mus(389).
ext_indep(2,5,s8y4):-mus(390).
ext_indep(2,5,s8y7):-mus(391).
ext_dep(2,5,s0y8y7):-mus(392).
ext_indep(2,5,s8y1y4):-mus(393).
ext_dep(2,7,empty):-mus(394).
ext_dep(2,7,s0):-mus(395).
ext_dep(2,7,s3):-mus(396).
ext_dep(2,7,s5):-mus(397).
ext_dep(2,7,s6):-mus(398).
ext_dep(2,7,s8):-mus(399).
ext_indep(2,7,s9):-mus(400).
ext_dep(2,7,s0y6):-mus(401).
ext_dep(2,7,s0y8):-mus(402).
ext_dep(2,7,s8y6):-mus(403).
ext_dep(2,8,empty):-mus(404).
ext_dep(2,8,s0):-mus(405).
ext_dep(2,8,s3):-mus(406).
ext_dep(2,8,s4):-mus(407).
ext_dep(2,8,s5):-mus(408).
ext_indep(2,8,s6):-mus(409).
ext_dep(2,8,s7):-mus(410).
ext_dep(2,8,s9):-mus(411).
ext_dep(2,8,s0y3):-mus(412).
ext_indep(2,8,s0y5):-mus(413).
ext_indep(2,8,s0y6):-mus(414).
ext_dep(2,8,s0y7):-mus(415).
ext_indep(2,8,s0y9):-mus(416).
ext_dep(2,8,s3y5):-mus(417).
ext_indep(2,8,s3y6):-mus(418).
ext_indep(2,8,s3y7):-mus(419).
ext_dep(2,8,s9y3):-mus(420).
ext_dep(2,8,s5y6):-mus(421).
ext_dep(2,8,s5y7):-mus(422).
ext_indep(2,8,s9y5):-mus(423).
ext_dep(2,8,s6y7):-mus(424).
ext_dep(2,8,s9y6):-mus(425).
ext_indep(2,8,s9y7):-mus(426).
ext_dep(2,8,s0y3y5):-mus(427).
ext_dep(2,8,s0y3y6):-mus(428).
ext_dep(2,8,s0y9y3):-mus(429).
ext_dep(2,8,s0y5y6):-mus(430).
ext_indep(2,8,s0y9y5):-mus(431).
ext_dep(2,8,s0y9y6):-mus(432).
ext_dep(2,8,s3y5y6):-mus(433).
ext_dep(2,8,s9y3y5):-mus(434).
ext_dep(2,8,s9y3y6):-mus(435).
ext_dep(2,8,s9y5y6):-mus(436).
ext_indep(2,8,s0y3y5y6):-mus(437).
ext_dep(2,8,s0y9y3y5):-mus(438).
ext_dep(2,8,s0y9y3y6):-mus(439).
ext_dep(2,8,s0y9y5y6):-mus(440).
ext_dep(2,8,s9y3y5y6):-mus(441).
ext_dep(2,8,s0y3y5y6y9):-mus(442).
ext_dep(2,9,s1):-mus(443).
ext_indep(2,9,s3):-mus(444).
ext_indep(2,9,s5):-mus(445).
ext_dep(3,4,empty):-mus(446).
ext_dep(3,4,s0):-mus(447).
ext_dep(3,4,s1):-mus(448).
ext_dep(3,4,s2):-mus(449).
ext_dep(3,4,s5):-mus(450).
ext_dep(3,4,s6):-mus(451).
ext_dep(3,4,s7):-mus(452).
ext_indep(3,4,s8):-mus(453).
ext_dep(3,4,s1y5):-mus(454).
ext_dep(3,4,s1y6):-mus(455).
ext_dep(3,4,s8y1):-mus(456).
ext_dep(3,4,s5y6):-mus(457).
ext_dep(3,4,s8y5):-mus(458).
ext_dep(3,4,s8y6):-mus(459).
ext_dep(3,4,s1y5y6):-mus(460).
ext_dep(3,4,s8y1y5):-mus(461).
ext_dep(3,4,s8y1y6):-mus(462).
ext_dep(3,4,s8y5y6):-mus(463).
ext_indep(3,5,s0):-mus(464).
ext_dep(3,5,s2):-mus(465).
ext_dep(3,5,s6):-mus(466).
ext_dep(3,5,s7):-mus(467).
ext_indep(3,5,s9):-mus(468).
ext_dep(3,5,s1y6):-mus(469).
ext_dep(3,5,s4y6):-mus(470).
ext_dep(3,5,s8y6):-mus(471).
ext_dep(3,5,s1y4y6):-mus(472).
ext_indep(3,5,s8y1y6):-mus(473).
ext_dep(3,5,s8y4y6):-mus(474).
ext_dep(3,6,s2):-mus(475).
ext_dep(3,6,s5):-mus(476).
ext_dep(3,6,s0y5):-mus(477).
ext_indep(3,6,s1y5):-mus(478).
ext_dep(3,6,s4y5):-mus(479).
ext_dep(3,6,s5y7):-mus(480).
ext_dep(3,6,s8y5):-mus(481).
ext_dep(3,6,s9y5):-mus(482).
ext_dep(3,6,s1y4y5):-mus(483).
ext_dep(3,6,s8y1y5):-mus(484).
ext_dep(3,6,s8y4y5):-mus(485).
ext_dep(3,7,empty):-mus(486).
ext_indep(3,7,s0):-mus(487).
ext_dep(3,7,s1):-mus(488).
ext_dep(3,7,s2):-mus(489).
ext_indep(3,7,s4):-mus(490).
ext_dep(3,7,s5):-mus(491).
ext_dep(3,7,s6):-mus(492).
ext_indep(3,7,s8):-mus(493).
ext_indep(3,7,s9):-mus(494).
ext_dep(3,8,empty):-mus(495).
ext_dep(3,8,s0):-mus(496).
ext_dep(3,8,s1):-mus(497).
ext_indep(3,8,s2):-mus(498).
ext_dep(3,8,s4):-mus(499).
ext_dep(3,8,s5):-mus(500).
ext_dep(3,8,s6):-mus(501).
ext_indep(3,8,s7):-mus(502).
ext_indep(3,8,s9):-mus(503).
ext_dep(3,8,s0y2):-mus(504).
ext_dep(3,8,s0y5):-mus(505).
ext_dep(3,8,s0y6):-mus(506).
ext_dep(3,8,s0y7):-mus(507).
ext_dep(3,8,s0y9):-mus(508).
ext_dep(3,8,s1y4):-mus(509).
ext_indep(3,8,s1y5):-mus(510).
ext_dep(3,8,s1y6):-mus(511).
ext_dep(3,8,s2y5):-mus(512).
ext_dep(3,8,s2y6):-mus(513).
ext_indep(3,8,s2y7):-mus(514).
ext_dep(3,8,s9y2):-mus(515).
ext_dep(3,8,s4y5):-mus(516).
ext_dep(3,8,s4y6):-mus(517).
ext_dep(3,8,s5y6):-mus(518).
ext_dep(3,8,s5y7):-mus(519).
ext_dep(3,8,s9y5):-mus(520).
ext_dep(3,8,s6y7):-mus(521).
ext_indep(3,8,s9y6):-mus(522).
ext_dep(3,8,s9y7):-mus(523).
ext_dep(3,8,s0y2y5):-mus(524).
ext_dep(3,8,s0y2y6):-mus(525).
ext_dep(3,8,s0y9y2):-mus(526).
ext_dep(3,8,s0y5y6):-mus(527).
ext_dep(3,8,s0y9y5):-mus(528).
ext_dep(3,8,s0y9y6):-mus(529).
ext_dep(3,8,s1y4y5):-mus(530).
ext_dep(3,8,s1y4y6):-mus(531).
ext_dep(3,8,s1y5y6):-mus(532).
ext_indep(3,8,s2y5y6):-mus(533).
ext_dep(3,8,s9y2y5):-mus(534).
ext_dep(3,8,s9y2y6):-mus(535).
ext_dep(3,8,s4y5y6):-mus(536).
ext_dep(3,8,s9y5y6):-mus(537).
ext_indep(3,8,s0y2y5y6):-mus(538).
ext_dep(3,8,s0y9y2y5):-mus(539).
ext_indep(3,8,s0y9y2y6):-mus(540).
ext_dep(3,8,s0y9y5y6):-mus(541).
ext_dep(3,8,s9y2y5y6):-mus(542).
ext_dep(3,8,s0y2y5y6y9):-mus(543).
ext_indep(3,9,s0):-mus(544).
ext_dep(3,9,s1):-mus(545).
ext_dep(3,9,s4):-mus(546).
ext_indep(3,9,s6):-mus(547).
ext_indep(3,9,s8):-mus(548).
ext_indep(3,9,s0y6):-mus(549).
ext_indep(3,9,s0y8):-mus(550).
ext_dep(3,9,s1y4):-mus(551).
ext_indep(3,9,s8y1):-mus(552).
ext_indep(3,9,s8y4):-mus(553).
ext_indep(3,9,s8y6):-mus(554).
ext_indep(3,9,s0y8y6):-mus(555).
ext_indep(3,9,s8y1y4):-mus(556).
ext_dep(4,5,empty):-mus(557).
ext_dep(4,5,s0):-mus(558).
ext_dep(4,5,s1):-mus(559).
ext_dep(4,5,s2):-mus(560).
ext_dep(4,5,s3):-mus(561).
ext_dep(4,5,s6):-mus(562).
ext_dep(4,5,s7):-mus(563).
ext_dep(4,5,s8):-mus(564).
ext_dep(4,5,s9):-mus(565).
ext_indep(4,5,s1y3):-mus(566).
ext_dep(4,5,s1y6):-mus(567).
ext_dep(4,5,s8y1):-mus(568).
ext_indep(4,5,s3y6):-mus(569).
ext_dep(4,5,s8y3):-mus(570).
ext_dep(4,5,s8y6):-mus(571).
ext_dep(4,5,s8y1y3):-mus(572).
ext_indep(4,6,s0y1):-mus(573).
ext_dep(4,6,s0y7):-mus(574).
ext_indep(4,6,s0y8):-mus(575).
ext_dep(4,6,s0y9):-mus(576).
ext_dep(4,6,s1y3):-mus(577).
ext_indep(4,6,s1y5):-mus(578).
ext_indep(4,6,s1y7):-mus(579).
ext_dep(4,6,s8y1):-mus(580).
ext_dep(4,6,s1y9):-mus(581).
ext_dep(4,6,s3y5):-mus(582).
ext_dep(4,6,s8y7):-mus(583).
ext_dep(4,6,s9y7):-mus(584).
ext_dep(4,6,s8y9):-mus(585).
ext_indep(4,6,s0y1y7):-mus(586).
ext_indep(4,6,s0y1y8):-mus(587).
ext_dep(4,6,s0y1y9):-mus(588).
ext_dep(4,6,s0y8y7):-mus(589).
ext_dep(4,6,s0y9y7):-mus(590).
ext_indep(4,6,s0y8y9):-mus(591).
ext_indep(4,6,s1y3y5):-mus(592).
ext_dep(4,6,s8y1y7):-mus(593).
ext_indep(4,6,s1y9y7):-mus(594).
ext_dep(4,6,s8y1y9):-mus(595).
ext_indep(4,6,s8y9y7):-mus(596).
ext_dep(4,6,s0y1y8y7):-mus(597).
ext_dep(4,6,s0y1y9y7):-mus(598).
ext_dep(4,6,s0y1y8y9):-mus(599).
ext_dep(4,6,s0y8y9y7):-mus(600).
ext_dep(4,6,s8y1y9y7):-mus(601).
ext_dep(4,6,s0y1y7y8y9):-mus(602).
ext_indep(4,7,empty):-mus(603).
ext_indep(4,8,s7):-mus(604).
ext_indep(4,9,empty):-mus(605).
ext_dep(5,6,s2):-mus(606).
ext_dep(5,6,s3):-mus(607).
ext_dep(5,6,s0y3):-mus(608).
ext_dep(5,6,s1y3):-mus(609).
ext_dep(5,6,s3y4):-mus(610).
ext_dep(5,6,s3y7):-mus(611).
ext_dep(5,6,s8y3):-mus(612).
ext_dep(5,6,s9y3):-mus(613).
ext_dep(5,7,empty):-mus(614).
ext_dep(5,7,s0):-mus(615).
ext_dep(5,7,s1):-mus(616).
ext_dep(5,7,s2):-mus(617).
ext_dep(5,7,s3):-mus(618).
ext_dep(5,7,s4):-mus(619).
ext_dep(5,7,s6):-mus(620).
ext_indep(5,7,s8):-mus(621).
ext_dep(5,7,s9):-mus(622).
ext_dep(5,8,empty):-mus(623).
ext_dep(5,8,s0):-mus(624).
ext_dep(5,8,s1):-mus(625).
ext_dep(5,8,s2):-mus(626).
ext_dep(5,8,s3):-mus(627).
ext_dep(5,8,s4):-mus(628).
ext_dep(5,8,s6):-mus(629).
ext_dep(5,8,s7):-mus(630).
ext_dep(5,8,s9):-mus(631).
ext_indep(5,8,s0y2):-mus(632).
ext_dep(5,8,s0y3):-mus(633).
ext_dep(5,8,s0y6):-mus(634).
ext_dep(5,8,s0y7):-mus(635).
ext_dep(5,8,s0y9):-mus(636).
ext_dep(5,8,s1y3):-mus(637).
ext_dep(5,8,s1y4):-mus(638).
ext_dep(5,8,s1y6):-mus(639).
ext_dep(5,8,s2y3):-mus(640).
ext_indep(5,8,s2y6):-mus(641).
ext_indep(5,8,s2y7):-mus(642).
ext_dep(5,8,s9y2):-mus(643).
ext_dep(5,8,s3y4):-mus(644).
ext_indep(5,8,s3y6):-mus(645).
ext_dep(5,8,s3y7):-mus(646).
ext_dep(5,8,s9y3):-mus(647).
ext_dep(5,8,s4y6):-mus(648).
ext_dep(5,8,s6y7):-mus(649).
ext_dep(5,8,s9y6):-mus(650).
ext_dep(5,8,s9y7):-mus(651).
ext_dep(5,8,s0y2y3):-mus(652).
ext_dep(5,8,s0y2y6):-mus(653).
ext_dep(5,8,s0y9y2):-mus(654).
ext_dep(5,8,s0y3y6):-mus(655).
ext_indep(5,8,s0y9y3):-mus(656).
ext_indep(5,8,s0y9y6):-mus(657).
ext_dep(5,8,s1y3y4):-mus(658).
ext_dep(5,8,s2y3y6):-mus(659).
ext_dep(5,8,s9y2y3):-mus(660).
ext_dep(5,8,s9y2y6):-mus(661).
ext_dep(5,8,s9y3y6):-mus(662).
ext_dep(5,8,s0y2y3y6):-mus(663).
ext_dep(5,8,s0y9y2y3):-mus(664).
ext_dep(5,8,s0y9y2y6):-mus(665).
ext_dep(5,8,s0y9y3y6):-mus(666).
ext_dep(5,8,s9y2y3y6):-mus(667).
ext_dep(5,8,s0y2y3y6y9):-mus(668).
ext_dep(5,9,s2):-mus(669).
ext_dep(5,9,s3):-mus(670).
ext_dep(5,9,s7):-mus(671).
ext_indep(5,9,s0y6):-mus(672).
ext_indep(5,9,s0y8):-mus(673).
ext_dep(5,9,s1y4):-mus(674).
ext_indep(5,9,s8y1):-mus(675).
ext_indep(5,9,s8y4):-mus(676).
ext_indep(5,9,s8y6):-mus(677).
ext_indep(5,9,s0y8y6):-mus(678).
ext_indep(5,9,s8y1y4):-mus(679).
ext_dep(6,7,empty):-mus(680).
ext_dep(6,7,s0):-mus(681).
ext_dep(6,7,s1):-mus(682).
ext_dep(6,7,s2):-mus(683).
ext_dep(6,7,s3):-mus(684).
ext_indep(6,7,s4):-mus(685).
ext_dep(6,7,s5):-mus(686).
ext_dep(6,7,s8):-mus(687).
ext_dep(6,7,s9):-mus(688).
ext_dep(6,7,s0y1):-mus(689).
ext_dep(6,7,s0y2):-mus(690).
ext_dep(6,7,s0y3):-mus(691).
ext_dep(6,7,s0y5):-mus(692).
ext_indep(6,7,s0y8):-mus(693).
ext_dep(6,7,s0y9):-mus(694).
ext_dep(6,7,s1y3):-mus(695).
ext_dep(6,7,s1y5):-mus(696).
ext_dep(6,7,s8y1):-mus(697).
ext_dep(6,7,s1y9):-mus(698).
ext_dep(6,7,s8y2):-mus(699).
ext_dep(6,7,s3y5):-mus(700).
ext_dep(6,7,s8y3):-mus(701).
ext_dep(6,7,s9y3):-mus(702).
ext_indep(6,7,s8y5):-mus(703).
ext_dep(6,7,s9y5):-mus(704).
ext_dep(6,7,s8y9):-mus(705).
ext_dep(6,7,s0y1y3):-mus(706).
ext_dep(6,7,s0y1y8):-mus(707).
ext_dep(6,7,s0y1y9):-mus(708).
ext_dep(6,7,s0y8y3):-mus(709).
ext_dep(6,7,s0y9y3):-mus(710).
ext_dep(6,7,s0y8y9):-mus(711).
ext_dep(6,7,s8y1y3):-mus(712).
ext_dep(6,7,s1y3y9):-mus(713).
ext_dep(6,7,s8y1y9):-mus(714).
ext_dep(6,7,s8y9y3):-mus(715).
ext_dep(6,7,s0y1y8y9):-mus(716).
ext_dep(6,8,empty):-mus(717).
ext_dep(6,8,s0):-mus(718).
ext_dep(6,8,s1):-mus(719).
ext_dep(6,8,s2):-mus(720).
ext_indep(6,8,s3):-mus(721).
ext_dep(6,8,s4):-mus(722).
ext_dep(6,8,s5):-mus(723).
ext_dep(6,8,s7):-mus(724).
ext_dep(6,8,s9):-mus(725).
ext_dep(6,8,s0y1):-mus(726).
ext_dep(6,8,s0y2):-mus(727).
ext_indep(6,8,s0y3):-mus(728).
ext_dep(6,8,s0y5):-mus(729).
ext_dep(6,8,s0y7):-mus(730).
ext_dep(6,8,s0y9):-mus(731).
ext_indep(6,8,s1y3):-mus(732).
ext_dep(6,8,s1y5):-mus(733).
ext_indep(6,8,s1y7):-mus(734).
ext_dep(6,8,s1y9):-mus(735).
ext_indep(6,8,s2y3):-mus(736).
ext_dep(6,8,s2y5):-mus(737).
ext_dep(6,8,s2y7):-mus(738).
ext_dep(6,8,s9y2):-mus(739).
ext_dep(6,8,s3y5):-mus(740).
ext_dep(6,8,s3y7):-mus(741).
ext_dep(6,8,s9y3):-mus(742).
ext_dep(6,8,s5y7):-mus(743).
ext_dep(6,8,s9y5):-mus(744).
ext_dep(6,8,s9y7):-mus(745).
ext_dep(6,8,s0y1y3):-mus(746).
ext_dep(6,8,s0y1y7):-mus(747).
ext_dep(6,8,s0y1y9):-mus(748).
ext_dep(6,8,s0y2y3):-mus(749).
ext_dep(6,8,s0y2y5):-mus(750).
ext_dep(6,8,s0y9y2):-mus(751).
ext_dep(6,8,s0y3y5):-mus(752).
ext_indep(6,8,s0y3y7):-mus(753).
ext_indep(6,8,s0y9y3):-mus(754).
ext_dep(6,8,s0y9y5):-mus(755).
ext_dep(6,8,s0y9y7):-mus(756).
ext_dep(6,8,s1y3y7):-mus(757).
ext_dep(6,8,s1y3y9):-mus(758).
ext_dep(6,8,s1y9y7):-mus(759).
ext_indep(6,8,s2y3y5):-mus(760).
ext_dep(6,8,s9y2y3):-mus(761).
ext_dep(6,8,s9y2y5):-mus(762).
ext_dep(6,8,s9y3y5):-mus(763).
ext_dep(6,8,s9y3y7):-mus(764).
ext_dep(6,8,s0y1y9y7):-mus(765).
ext_dep(6,8,s0y2y3y5):-mus(766).
ext_dep(6,8,s0y9y2y3):-mus(767).
ext_dep(6,8,s0y9y2y5):-mus(768).
ext_dep(6,8,s0y9y3y5):-mus(769).
ext_dep(6,8,s9y2y3y5):-mus(770).
ext_dep(6,8,s0y2y3y5y9):-mus(771).
ext_dep(6,9,empty):-mus(772).
ext_dep(6,9,s0):-mus(773).
ext_dep(6,9,s1):-mus(774).
ext_dep(6,9,s2):-mus(775).
ext_dep(6,9,s3):-mus(776).
ext_dep(6,9,s4):-mus(777).
ext_dep(6,9,s5):-mus(778).
ext_dep(6,9,s7):-mus(779).
ext_dep(6,9,s8):-mus(780).
ext_dep(6,9,s0y1):-mus(781).
ext_dep(6,9,s0y3):-mus(782).
ext_dep(6,9,s0y5):-mus(783).
ext_dep(6,9,s0y7):-mus(784).
ext_dep(6,9,s0y8):-mus(785).
ext_dep(6,9,s1y3):-mus(786).
ext_dep(6,9,s1y5):-mus(787).
ext_dep(6,9,s1y7):-mus(788).
ext_dep(6,9,s8y1):-mus(789).
ext_dep(6,9,s3y5):-mus(790).
ext_indep(6,9,s3y7):-mus(791).
ext_dep(6,9,s8y3):-mus(792).
ext_dep(6,9,s5y7):-mus(793).
ext_dep(6,9,s8y5):-mus(794).
ext_dep(6,9,s8y7):-mus(795).
ext_dep(6,9,s0y1y3):-mus(796).
ext_indep(6,9,s0y1y7):-mus(797).
ext_dep(6,9,s0y1y8):-mus(798).
ext_dep(6,9,s0y3y7):-mus(799).
ext_indep(6,9,s0y8y3):-mus(800).
ext_dep(6,9,s0y8y7):-mus(801).
ext_dep(6,9,s1y3y7):-mus(802).
ext_dep(6,9,s8y1y3):-mus(803).
ext_dep(6,9,s8y1y7):-mus(804).
ext_dep(6,9,s8y3y7):-mus(805).
ext_indep(6,9,s0y1y8y7):-mus(806).
ext_dep(7,8,s4):-mus(807).
ext_dep(7,9,s1):-mus(808).
ext_indep(7,9,s3):-mus(809).
ext_dep(7,9,s5):-mus(810).
ext_dep(8,9,empty):-mus(811).
ext_dep(8,9,s0):-mus(812).
ext_dep(8,9,s1):-mus(813).
ext_dep(8,9,s2):-mus(814).
ext_indep(8,9,s3):-mus(815).
ext_dep(8,9,s4):-mus(816).
ext_dep(8,9,s5):-mus(817).
ext_dep(8,9,s6):-mus(818).
ext_dep(8,9,s7):-mus(819).
ext_dep(8,9,s0y2):-mus(820).
ext_dep(8,9,s0y3):-mus(821).
ext_dep(8,9,s0y5):-mus(822).
ext_indep(8,9,s0y6):-mus(823).
ext_indep(8,9,s0y7):-mus(824).
ext_dep(8,9,s2y3):-mus(825).
ext_indep(8,9,s2y5):-mus(826).
ext_dep(8,9,s2y6):-mus(827).
ext_dep(8,9,s2y7):-mus(828).
ext_dep(8,9,s3y5):-mus(829).
ext_dep(8,9,s3y6):-mus(830).
ext_dep(8,9,s3y7):-mus(831).
ext_dep(8,9,s5y6):-mus(832).
ext_dep(8,9,s5y7):-mus(833).
ext_dep(8,9,s6y7):-mus(834).
ext_dep(8,9,s0y2y3):-mus(835).
ext_dep(8,9,s0y2y5):-mus(836).
ext_dep(8,9,s0y2y6):-mus(837).
ext_dep(8,9,s0y3y5):-mus(838).
ext_indep(8,9,s0y3y6):-mus(839).
ext_dep(8,9,s0y5y6):-mus(840).
ext_dep(8,9,s2y3y5):-mus(841).
ext_dep(8,9,s2y3y6):-mus(842).
ext_indep(8,9,s2y5y6):-mus(843).
ext_indep(8,9,s3y5y6):-mus(844).
ext_dep(8,9,s0y2y3y5):-mus(845).
ext_dep(8,9,s0y2y3y6):-mus(846).
ext_dep(8,9,s0y2y5y6):-mus(847).
ext_dep(8,9,s0y3y5y6):-mus(848).
ext_dep(8,9,s2y3y5y6):-mus(849).
ext_dep(8,9,s0y2y3y5y6):-mus(850).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(8620058, mus(1)).
__optimum_mcs_objective_literal__(9999999, mus(2)).
__optimum_mcs_objective_literal__(9999999, mus(3)).
__optimum_mcs_objective_literal__(7934456, mus(4)).
__optimum_mcs_objective_literal__(9999999, mus(5)).
__optimum_mcs_objective_literal__(5064356, mus(6)).
__optimum_mcs_objective_literal__(6036798, mus(7)).
__optimum_mcs_objective_literal__(9999978, mus(8)).
__optimum_mcs_objective_literal__(9999999, mus(9)).
__optimum_mcs_objective_literal__(5175497, mus(10)).
__optimum_mcs_objective_literal__(6195608, mus(11)).
__optimum_mcs_objective_literal__(9894621, mus(12)).
__optimum_mcs_objective_literal__(9999999, mus(13)).
__optimum_mcs_objective_literal__(9999997, mus(14)).
__optimum_mcs_objective_literal__(9999999, mus(15)).
__optimum_mcs_objective_literal__(9999999, mus(16)).
__optimum_mcs_objective_literal__(9999999, mus(17)).
__optimum_mcs_objective_literal__(9999999, mus(18)).
__optimum_mcs_objective_literal__(9999999, mus(19)).
__optimum_mcs_objective_literal__(9999999, mus(20)).
__optimum_mcs_objective_literal__(9999999, mus(21)).
__optimum_mcs_objective_literal__(9717305, mus(22)).
__optimum_mcs_objective_literal__(9991721, mus(23)).
__optimum_mcs_objective_literal__(9999944, mus(24)).
__optimum_mcs_objective_literal__(9999999, mus(25)).
__optimum_mcs_objective_literal__(9999999, mus(26)).
__optimum_mcs_objective_literal__(9999999, mus(27)).
__optimum_mcs_objective_literal__(5151796, mus(28)).
__optimum_mcs_objective_literal__(6074022, mus(29)).
__optimum_mcs_objective_literal__(9950795, mus(30)).
__optimum_mcs_objective_literal__(9999999, mus(31)).
__optimum_mcs_objective_literal__(9999999, mus(32)).
__optimum_mcs_objective_literal__(9999999, mus(33)).
__optimum_mcs_objective_literal__(9999999, mus(34)).
__optimum_mcs_objective_literal__(9999999, mus(35)).
__optimum_mcs_objective_literal__(9999999, mus(36)).
__optimum_mcs_objective_literal__(9999999, mus(37)).
__optimum_mcs_objective_literal__(9967594, mus(38)).
__optimum_mcs_objective_literal__(9999999, mus(39)).
__optimum_mcs_objective_literal__(9999960, mus(40)).
__optimum_mcs_objective_literal__(9999999, mus(41)).
__optimum_mcs_objective_literal__(7926420, mus(42)).
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
__optimum_mcs_objective_literal__(9998556, mus(64)).
__optimum_mcs_objective_literal__(9999999, mus(65)).
__optimum_mcs_objective_literal__(9933544, mus(66)).
__optimum_mcs_objective_literal__(9999999, mus(67)).
__optimum_mcs_objective_literal__(9213394, mus(68)).
__optimum_mcs_objective_literal__(5069196, mus(69)).
__optimum_mcs_objective_literal__(9904776, mus(70)).
__optimum_mcs_objective_literal__(9855100, mus(71)).
__optimum_mcs_objective_literal__(9999998, mus(72)).
__optimum_mcs_objective_literal__(9999808, mus(73)).
__optimum_mcs_objective_literal__(9998633, mus(74)).
__optimum_mcs_objective_literal__(5188628, mus(75)).
__optimum_mcs_objective_literal__(9837384, mus(76)).
__optimum_mcs_objective_literal__(6397689, mus(77)).
__optimum_mcs_objective_literal__(8915925, mus(78)).
__optimum_mcs_objective_literal__(9968852, mus(79)).
__optimum_mcs_objective_literal__(9991854, mus(80)).
__optimum_mcs_objective_literal__(9999999, mus(81)).
__optimum_mcs_objective_literal__(5281971, mus(82)).
__optimum_mcs_objective_literal__(9999998, mus(83)).
__optimum_mcs_objective_literal__(9999826, mus(84)).
__optimum_mcs_objective_literal__(9966126, mus(85)).
__optimum_mcs_objective_literal__(9999990, mus(86)).
__optimum_mcs_objective_literal__(9877451, mus(87)).
__optimum_mcs_objective_literal__(7678282, mus(88)).
__optimum_mcs_objective_literal__(9850896, mus(89)).
__optimum_mcs_objective_literal__(9999999, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(9999999, mus(92)).
__optimum_mcs_objective_literal__(9999988, mus(93)).
__optimum_mcs_objective_literal__(9999999, mus(94)).
__optimum_mcs_objective_literal__(5432019, mus(95)).
__optimum_mcs_objective_literal__(9999999, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(8248275, mus(98)).
__optimum_mcs_objective_literal__(9999999, mus(99)).
__optimum_mcs_objective_literal__(9999999, mus(100)).
__optimum_mcs_objective_literal__(9984645, mus(101)).
__optimum_mcs_objective_literal__(9999999, mus(102)).
__optimum_mcs_objective_literal__(9999999, mus(103)).
__optimum_mcs_objective_literal__(9999999, mus(104)).
__optimum_mcs_objective_literal__(9911165, mus(105)).
__optimum_mcs_objective_literal__(9999999, mus(106)).
__optimum_mcs_objective_literal__(9923672, mus(107)).
__optimum_mcs_objective_literal__(9999999, mus(108)).
__optimum_mcs_objective_literal__(7063787, mus(109)).
__optimum_mcs_objective_literal__(9999999, mus(110)).
__optimum_mcs_objective_literal__(9999999, mus(111)).
__optimum_mcs_objective_literal__(9744708, mus(112)).
__optimum_mcs_objective_literal__(9999999, mus(113)).
__optimum_mcs_objective_literal__(9907446, mus(114)).
__optimum_mcs_objective_literal__(9999999, mus(115)).
__optimum_mcs_objective_literal__(6935192, mus(116)).
__optimum_mcs_objective_literal__(7881199, mus(117)).
__optimum_mcs_objective_literal__(9999999, mus(118)).
__optimum_mcs_objective_literal__(5558374, mus(119)).
__optimum_mcs_objective_literal__(6751801, mus(120)).
__optimum_mcs_objective_literal__(5640775, mus(121)).
__optimum_mcs_objective_literal__(9999999, mus(122)).
__optimum_mcs_objective_literal__(5343771, mus(123)).
__optimum_mcs_objective_literal__(5357809, mus(124)).
__optimum_mcs_objective_literal__(5749505, mus(125)).
__optimum_mcs_objective_literal__(9535860, mus(126)).
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
__optimum_mcs_objective_literal__(5813027, mus(249)).
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
__optimum_mcs_objective_literal__(5024537, mus(333)).
__optimum_mcs_objective_literal__(8451538, mus(334)).
__optimum_mcs_objective_literal__(6035699, mus(335)).
__optimum_mcs_objective_literal__(7203774, mus(336)).
__optimum_mcs_objective_literal__(5633778, mus(337)).
__optimum_mcs_objective_literal__(6754118, mus(338)).
__optimum_mcs_objective_literal__(5155071, mus(339)).
__optimum_mcs_objective_literal__(9470507, mus(340)).
__optimum_mcs_objective_literal__(9562426, mus(341)).
__optimum_mcs_objective_literal__(6062226, mus(342)).
__optimum_mcs_objective_literal__(8258886, mus(343)).
__optimum_mcs_objective_literal__(9744606, mus(344)).
__optimum_mcs_objective_literal__(8927727, mus(345)).
__optimum_mcs_objective_literal__(9238328, mus(346)).
__optimum_mcs_objective_literal__(9999657, mus(347)).
__optimum_mcs_objective_literal__(5510358, mus(348)).
__optimum_mcs_objective_literal__(8376905, mus(349)).
__optimum_mcs_objective_literal__(9520950, mus(350)).
__optimum_mcs_objective_literal__(8736707, mus(351)).
__optimum_mcs_objective_literal__(9999747, mus(352)).
__optimum_mcs_objective_literal__(6348807, mus(353)).
__optimum_mcs_objective_literal__(5812457, mus(354)).
__optimum_mcs_objective_literal__(9633421, mus(355)).
__optimum_mcs_objective_literal__(9999102, mus(356)).
__optimum_mcs_objective_literal__(9999999, mus(357)).
__optimum_mcs_objective_literal__(7184949, mus(358)).
__optimum_mcs_objective_literal__(5618647, mus(359)).
__optimum_mcs_objective_literal__(9976527, mus(360)).
__optimum_mcs_objective_literal__(9592768, mus(361)).
__optimum_mcs_objective_literal__(9995799, mus(362)).
__optimum_mcs_objective_literal__(9999402, mus(363)).
__optimum_mcs_objective_literal__(9980546, mus(364)).
__optimum_mcs_objective_literal__(5728314, mus(365)).
__optimum_mcs_objective_literal__(9890854, mus(366)).
__optimum_mcs_objective_literal__(9999258, mus(367)).
__optimum_mcs_objective_literal__(9999812, mus(368)).
__optimum_mcs_objective_literal__(9998801, mus(369)).
__optimum_mcs_objective_literal__(9999999, mus(370)).
__optimum_mcs_objective_literal__(9519074, mus(371)).
__optimum_mcs_objective_literal__(9987582, mus(372)).
__optimum_mcs_objective_literal__(6210449, mus(373)).
__optimum_mcs_objective_literal__(9578357, mus(374)).
__optimum_mcs_objective_literal__(5780049, mus(375)).
__optimum_mcs_objective_literal__(9999999, mus(376)).
__optimum_mcs_objective_literal__(9062089, mus(377)).
__optimum_mcs_objective_literal__(7063211, mus(378)).
__optimum_mcs_objective_literal__(8691994, mus(379)).
__optimum_mcs_objective_literal__(7871141, mus(380)).
__optimum_mcs_objective_literal__(8461636, mus(381)).
__optimum_mcs_objective_literal__(7546973, mus(382)).
__optimum_mcs_objective_literal__(9999999, mus(383)).
__optimum_mcs_objective_literal__(9999999, mus(384)).
__optimum_mcs_objective_literal__(9999999, mus(385)).
__optimum_mcs_objective_literal__(9999999, mus(386)).
__optimum_mcs_objective_literal__(9483796, mus(387)).
__optimum_mcs_objective_literal__(9999999, mus(388)).
__optimum_mcs_objective_literal__(8538241, mus(389)).
__optimum_mcs_objective_literal__(5643124, mus(390)).
__optimum_mcs_objective_literal__(9406649, mus(391)).
__optimum_mcs_objective_literal__(9821350, mus(392)).
__optimum_mcs_objective_literal__(7595064, mus(393)).
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
__optimum_mcs_objective_literal__(9999999, mus(405)).
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
__optimum_mcs_objective_literal__(7643493, mus(443)).
__optimum_mcs_objective_literal__(5280508, mus(444)).
__optimum_mcs_objective_literal__(5113578, mus(445)).
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
__optimum_mcs_objective_literal__(9476898, mus(473)).
__optimum_mcs_objective_literal__(9999999, mus(474)).
__optimum_mcs_objective_literal__(9999995, mus(475)).
__optimum_mcs_objective_literal__(9999999, mus(476)).
__optimum_mcs_objective_literal__(9999999, mus(477)).
__optimum_mcs_objective_literal__(9951296, mus(478)).
__optimum_mcs_objective_literal__(9999999, mus(479)).
__optimum_mcs_objective_literal__(9999999, mus(480)).
__optimum_mcs_objective_literal__(9999999, mus(481)).
__optimum_mcs_objective_literal__(9999987, mus(482)).
__optimum_mcs_objective_literal__(9803164, mus(483)).
__optimum_mcs_objective_literal__(8619138, mus(484)).
__optimum_mcs_objective_literal__(9999999, mus(485)).
__optimum_mcs_objective_literal__(7670521, mus(486)).
__optimum_mcs_objective_literal__(5044285, mus(487)).
__optimum_mcs_objective_literal__(7467978, mus(488)).
__optimum_mcs_objective_literal__(5478588, mus(489)).
__optimum_mcs_objective_literal__(9995150, mus(490)).
__optimum_mcs_objective_literal__(5776890, mus(491)).
__optimum_mcs_objective_literal__(8956231, mus(492)).
__optimum_mcs_objective_literal__(5255592, mus(493)).
__optimum_mcs_objective_literal__(7295611, mus(494)).
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
__optimum_mcs_objective_literal__(9999999, mus(534)).
__optimum_mcs_objective_literal__(9999999, mus(535)).
__optimum_mcs_objective_literal__(9999999, mus(536)).
__optimum_mcs_objective_literal__(9999999, mus(537)).
__optimum_mcs_objective_literal__(9999999, mus(538)).
__optimum_mcs_objective_literal__(9999999, mus(539)).
__optimum_mcs_objective_literal__(9999999, mus(540)).
__optimum_mcs_objective_literal__(9999999, mus(541)).
__optimum_mcs_objective_literal__(9999999, mus(542)).
__optimum_mcs_objective_literal__(9999999, mus(543)).
__optimum_mcs_objective_literal__(5099144, mus(544)).
__optimum_mcs_objective_literal__(5808252, mus(545)).
__optimum_mcs_objective_literal__(6631985, mus(546)).
__optimum_mcs_objective_literal__(5108254, mus(547)).
__optimum_mcs_objective_literal__(8661952, mus(548)).
__optimum_mcs_objective_literal__(8729745, mus(549)).
__optimum_mcs_objective_literal__(7999950, mus(550)).
__optimum_mcs_objective_literal__(9280695, mus(551)).
__optimum_mcs_objective_literal__(8956187, mus(552)).
__optimum_mcs_objective_literal__(6428214, mus(553)).
__optimum_mcs_objective_literal__(7006117, mus(554)).
__optimum_mcs_objective_literal__(5326041, mus(555)).
__optimum_mcs_objective_literal__(5163335, mus(556)).
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
__optimum_mcs_objective_literal__(9998772, mus(572)).
__optimum_mcs_objective_literal__(6343863, mus(573)).
__optimum_mcs_objective_literal__(9999300, mus(574)).
__optimum_mcs_objective_literal__(9999999, mus(575)).
__optimum_mcs_objective_literal__(9962080, mus(576)).
__optimum_mcs_objective_literal__(5937627, mus(577)).
__optimum_mcs_objective_literal__(6210230, mus(578)).
__optimum_mcs_objective_literal__(5903371, mus(579)).
__optimum_mcs_objective_literal__(9157244, mus(580)).
__optimum_mcs_objective_literal__(9632390, mus(581)).
__optimum_mcs_objective_literal__(9782684, mus(582)).
__optimum_mcs_objective_literal__(9999998, mus(583)).
__optimum_mcs_objective_literal__(9474970, mus(584)).
__optimum_mcs_objective_literal__(9999981, mus(585)).
__optimum_mcs_objective_literal__(7437128, mus(586)).
__optimum_mcs_objective_literal__(8316522, mus(587)).
__optimum_mcs_objective_literal__(7770913, mus(588)).
__optimum_mcs_objective_literal__(9999998, mus(589)).
__optimum_mcs_objective_literal__(9819928, mus(590)).
__optimum_mcs_objective_literal__(9993395, mus(591)).
__optimum_mcs_objective_literal__(5395758, mus(592)).
__optimum_mcs_objective_literal__(5314989, mus(593)).
__optimum_mcs_objective_literal__(5307791, mus(594)).
__optimum_mcs_objective_literal__(9933531, mus(595)).
__optimum_mcs_objective_literal__(9997313, mus(596)).
__optimum_mcs_objective_literal__(9650144, mus(597)).
__optimum_mcs_objective_literal__(5076216, mus(598)).
__optimum_mcs_objective_literal__(5234574, mus(599)).
__optimum_mcs_objective_literal__(9995217, mus(600)).
__optimum_mcs_objective_literal__(6778069, mus(601)).
__optimum_mcs_objective_literal__(8090316, mus(602)).
__optimum_mcs_objective_literal__(9841415, mus(603)).
__optimum_mcs_objective_literal__(9971065, mus(604)).
__optimum_mcs_objective_literal__(6119077, mus(605)).
__optimum_mcs_objective_literal__(9999827, mus(606)).
__optimum_mcs_objective_literal__(9999866, mus(607)).
__optimum_mcs_objective_literal__(9997991, mus(608)).
__optimum_mcs_objective_literal__(9973706, mus(609)).
__optimum_mcs_objective_literal__(9980243, mus(610)).
__optimum_mcs_objective_literal__(9999999, mus(611)).
__optimum_mcs_objective_literal__(9801907, mus(612)).
__optimum_mcs_objective_literal__(9999997, mus(613)).
__optimum_mcs_objective_literal__(9999999, mus(614)).
__optimum_mcs_objective_literal__(9999998, mus(615)).
__optimum_mcs_objective_literal__(9999999, mus(616)).
__optimum_mcs_objective_literal__(9999999, mus(617)).
__optimum_mcs_objective_literal__(9999999, mus(618)).
__optimum_mcs_objective_literal__(9999999, mus(619)).
__optimum_mcs_objective_literal__(9999999, mus(620)).
__optimum_mcs_objective_literal__(5979090, mus(621)).
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
__optimum_mcs_objective_literal__(9999999, mus(633)).
__optimum_mcs_objective_literal__(9999999, mus(634)).
__optimum_mcs_objective_literal__(9999999, mus(635)).
__optimum_mcs_objective_literal__(9999999, mus(636)).
__optimum_mcs_objective_literal__(9999999, mus(637)).
__optimum_mcs_objective_literal__(9999999, mus(638)).
__optimum_mcs_objective_literal__(9999999, mus(639)).
__optimum_mcs_objective_literal__(9999999, mus(640)).
__optimum_mcs_objective_literal__(9999999, mus(641)).
__optimum_mcs_objective_literal__(9999999, mus(642)).
__optimum_mcs_objective_literal__(9999999, mus(643)).
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
__optimum_mcs_objective_literal__(8705827, mus(673)).
__optimum_mcs_objective_literal__(9999999, mus(674)).
__optimum_mcs_objective_literal__(7405805, mus(675)).
__optimum_mcs_objective_literal__(8476114, mus(676)).
__optimum_mcs_objective_literal__(7750688, mus(677)).
__optimum_mcs_objective_literal__(8027129, mus(678)).
__optimum_mcs_objective_literal__(6363714, mus(679)).
__optimum_mcs_objective_literal__(9023984, mus(680)).
__optimum_mcs_objective_literal__(9890696, mus(681)).
__optimum_mcs_objective_literal__(9867623, mus(682)).
__optimum_mcs_objective_literal__(9959807, mus(683)).
__optimum_mcs_objective_literal__(8176976, mus(684)).
__optimum_mcs_objective_literal__(9020845, mus(685)).
__optimum_mcs_objective_literal__(9999999, mus(686)).
__optimum_mcs_objective_literal__(9999999, mus(687)).
__optimum_mcs_objective_literal__(9999567, mus(688)).
__optimum_mcs_objective_literal__(9984460, mus(689)).
__optimum_mcs_objective_literal__(9985516, mus(690)).
__optimum_mcs_objective_literal__(5405547, mus(691)).
__optimum_mcs_objective_literal__(9999999, mus(692)).
__optimum_mcs_objective_literal__(9999999, mus(693)).
__optimum_mcs_objective_literal__(8146011, mus(694)).
__optimum_mcs_objective_literal__(9985805, mus(695)).
__optimum_mcs_objective_literal__(9999999, mus(696)).
__optimum_mcs_objective_literal__(9999999, mus(697)).
__optimum_mcs_objective_literal__(9968762, mus(698)).
__optimum_mcs_objective_literal__(9999999, mus(699)).
__optimum_mcs_objective_literal__(9999999, mus(700)).
__optimum_mcs_objective_literal__(9999999, mus(701)).
__optimum_mcs_objective_literal__(9867682, mus(702)).
__optimum_mcs_objective_literal__(9999999, mus(703)).
__optimum_mcs_objective_literal__(9999999, mus(704)).
__optimum_mcs_objective_literal__(9999999, mus(705)).
__optimum_mcs_objective_literal__(9856231, mus(706)).
__optimum_mcs_objective_literal__(9999999, mus(707)).
__optimum_mcs_objective_literal__(9682738, mus(708)).
__optimum_mcs_objective_literal__(9999999, mus(709)).
__optimum_mcs_objective_literal__(9998892, mus(710)).
__optimum_mcs_objective_literal__(9999999, mus(711)).
__optimum_mcs_objective_literal__(9999999, mus(712)).
__optimum_mcs_objective_literal__(9999975, mus(713)).
__optimum_mcs_objective_literal__(9999999, mus(714)).
__optimum_mcs_objective_literal__(9999999, mus(715)).
__optimum_mcs_objective_literal__(9999999, mus(716)).
__optimum_mcs_objective_literal__(9995998, mus(717)).
__optimum_mcs_objective_literal__(9999995, mus(718)).
__optimum_mcs_objective_literal__(9999999, mus(719)).
__optimum_mcs_objective_literal__(9999241, mus(720)).
__optimum_mcs_objective_literal__(9999999, mus(721)).
__optimum_mcs_objective_literal__(9999997, mus(722)).
__optimum_mcs_objective_literal__(9742160, mus(723)).
__optimum_mcs_objective_literal__(9999999, mus(724)).
__optimum_mcs_objective_literal__(9999999, mus(725)).
__optimum_mcs_objective_literal__(9999999, mus(726)).
__optimum_mcs_objective_literal__(9999999, mus(727)).
__optimum_mcs_objective_literal__(9999999, mus(728)).
__optimum_mcs_objective_literal__(9999998, mus(729)).
__optimum_mcs_objective_literal__(9999999, mus(730)).
__optimum_mcs_objective_literal__(9999999, mus(731)).
__optimum_mcs_objective_literal__(9999999, mus(732)).
__optimum_mcs_objective_literal__(9999999, mus(733)).
__optimum_mcs_objective_literal__(9999999, mus(734)).
__optimum_mcs_objective_literal__(9999999, mus(735)).
__optimum_mcs_objective_literal__(9999999, mus(736)).
__optimum_mcs_objective_literal__(9643981, mus(737)).
__optimum_mcs_objective_literal__(9999999, mus(738)).
__optimum_mcs_objective_literal__(9999999, mus(739)).
__optimum_mcs_objective_literal__(9999992, mus(740)).
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
__optimum_mcs_objective_literal__(9999999, mus(751)).
__optimum_mcs_objective_literal__(9999998, mus(752)).
__optimum_mcs_objective_literal__(9999999, mus(753)).
__optimum_mcs_objective_literal__(9999999, mus(754)).
__optimum_mcs_objective_literal__(9999999, mus(755)).
__optimum_mcs_objective_literal__(9999999, mus(756)).
__optimum_mcs_objective_literal__(9999999, mus(757)).
__optimum_mcs_objective_literal__(9999999, mus(758)).
__optimum_mcs_objective_literal__(9999999, mus(759)).
__optimum_mcs_objective_literal__(9983755, mus(760)).
__optimum_mcs_objective_literal__(9999999, mus(761)).
__optimum_mcs_objective_literal__(9999999, mus(762)).
__optimum_mcs_objective_literal__(9999999, mus(763)).
__optimum_mcs_objective_literal__(9999999, mus(764)).
__optimum_mcs_objective_literal__(9999999, mus(765)).
__optimum_mcs_objective_literal__(9999997, mus(766)).
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
__optimum_mcs_objective_literal__(9999999, mus(800)).
__optimum_mcs_objective_literal__(9999999, mus(801)).
__optimum_mcs_objective_literal__(9999999, mus(802)).
__optimum_mcs_objective_literal__(9999999, mus(803)).
__optimum_mcs_objective_literal__(9999999, mus(804)).
__optimum_mcs_objective_literal__(9999999, mus(805)).
__optimum_mcs_objective_literal__(9999999, mus(806)).
__optimum_mcs_objective_literal__(9999999, mus(807)).
__optimum_mcs_objective_literal__(9999730, mus(808)).
__optimum_mcs_objective_literal__(5344822, mus(809)).
__optimum_mcs_objective_literal__(9999389, mus(810)).
__optimum_mcs_objective_literal__(9999999, mus(811)).
__optimum_mcs_objective_literal__(9999999, mus(812)).
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

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
