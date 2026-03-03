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
in(0,s0y1y2y3y4y5y6).
in(1,s0y1y2y3y4y5y6).
in(2,s0y1y2y3y4y5y6).
in(3,s0y1y2y3y4y5y6).
in(4,s0y1y2y3y4y5y6).
in(5,s0y1y2y3y4y5y6).
in(6,s0y1y2y3y4y5y6).
in(0,s0y1y2y3y4y5y7).
in(1,s0y1y2y3y4y5y7).
in(2,s0y1y2y3y4y5y7).
in(3,s0y1y2y3y4y5y7).
in(4,s0y1y2y3y4y5y7).
in(5,s0y1y2y3y4y5y7).
in(7,s0y1y2y3y4y5y7).
in(0,s0y1y2y3y4y5y8).
in(1,s0y1y2y3y4y5y8).
in(2,s0y1y2y3y4y5y8).
in(3,s0y1y2y3y4y5y8).
in(4,s0y1y2y3y4y5y8).
in(5,s0y1y2y3y4y5y8).
in(8,s0y1y2y3y4y5y8).
in(0,s0y1y2y3y4y5y9).
in(1,s0y1y2y3y4y5y9).
in(2,s0y1y2y3y4y5y9).
in(3,s0y1y2y3y4y5y9).
in(4,s0y1y2y3y4y5y9).
in(5,s0y1y2y3y4y5y9).
in(9,s0y1y2y3y4y5y9).
in(0,s0y1y2y3y4y6y7).
in(1,s0y1y2y3y4y6y7).
in(2,s0y1y2y3y4y6y7).
in(3,s0y1y2y3y4y6y7).
in(4,s0y1y2y3y4y6y7).
in(6,s0y1y2y3y4y6y7).
in(7,s0y1y2y3y4y6y7).
in(0,s0y1y2y3y4y6y8).
in(1,s0y1y2y3y4y6y8).
in(2,s0y1y2y3y4y6y8).
in(3,s0y1y2y3y4y6y8).
in(4,s0y1y2y3y4y6y8).
in(6,s0y1y2y3y4y6y8).
in(8,s0y1y2y3y4y6y8).
in(0,s0y1y2y3y4y6y9).
in(1,s0y1y2y3y4y6y9).
in(2,s0y1y2y3y4y6y9).
in(3,s0y1y2y3y4y6y9).
in(4,s0y1y2y3y4y6y9).
in(6,s0y1y2y3y4y6y9).
in(9,s0y1y2y3y4y6y9).
in(0,s0y1y2y3y4y7y8).
in(1,s0y1y2y3y4y7y8).
in(2,s0y1y2y3y4y7y8).
in(3,s0y1y2y3y4y7y8).
in(4,s0y1y2y3y4y7y8).
in(7,s0y1y2y3y4y7y8).
in(8,s0y1y2y3y4y7y8).
in(0,s0y1y2y3y4y7y9).
in(1,s0y1y2y3y4y7y9).
in(2,s0y1y2y3y4y7y9).
in(3,s0y1y2y3y4y7y9).
in(4,s0y1y2y3y4y7y9).
in(7,s0y1y2y3y4y7y9).
in(9,s0y1y2y3y4y7y9).
in(0,s0y1y2y3y4y8y9).
in(1,s0y1y2y3y4y8y9).
in(2,s0y1y2y3y4y8y9).
in(3,s0y1y2y3y4y8y9).
in(4,s0y1y2y3y4y8y9).
in(8,s0y1y2y3y4y8y9).
in(9,s0y1y2y3y4y8y9).
in(0,s0y1y2y3y5y6y7).
in(1,s0y1y2y3y5y6y7).
in(2,s0y1y2y3y5y6y7).
in(3,s0y1y2y3y5y6y7).
in(5,s0y1y2y3y5y6y7).
in(6,s0y1y2y3y5y6y7).
in(7,s0y1y2y3y5y6y7).
in(0,s0y1y2y3y5y6y8).
in(1,s0y1y2y3y5y6y8).
in(2,s0y1y2y3y5y6y8).
in(3,s0y1y2y3y5y6y8).
in(5,s0y1y2y3y5y6y8).
in(6,s0y1y2y3y5y6y8).
in(8,s0y1y2y3y5y6y8).
in(0,s0y1y2y3y5y6y9).
in(1,s0y1y2y3y5y6y9).
in(2,s0y1y2y3y5y6y9).
in(3,s0y1y2y3y5y6y9).
in(5,s0y1y2y3y5y6y9).
in(6,s0y1y2y3y5y6y9).
in(9,s0y1y2y3y5y6y9).
in(0,s0y1y2y3y5y7y8).
in(1,s0y1y2y3y5y7y8).
in(2,s0y1y2y3y5y7y8).
in(3,s0y1y2y3y5y7y8).
in(5,s0y1y2y3y5y7y8).
in(7,s0y1y2y3y5y7y8).
in(8,s0y1y2y3y5y7y8).
in(0,s0y1y2y3y5y7y9).
in(1,s0y1y2y3y5y7y9).
in(2,s0y1y2y3y5y7y9).
in(3,s0y1y2y3y5y7y9).
in(5,s0y1y2y3y5y7y9).
in(7,s0y1y2y3y5y7y9).
in(9,s0y1y2y3y5y7y9).
in(0,s0y1y2y3y5y8y9).
in(1,s0y1y2y3y5y8y9).
in(2,s0y1y2y3y5y8y9).
in(3,s0y1y2y3y5y8y9).
in(5,s0y1y2y3y5y8y9).
in(8,s0y1y2y3y5y8y9).
in(9,s0y1y2y3y5y8y9).
in(0,s0y1y2y3y6y7y8).
in(1,s0y1y2y3y6y7y8).
in(2,s0y1y2y3y6y7y8).
in(3,s0y1y2y3y6y7y8).
in(6,s0y1y2y3y6y7y8).
in(7,s0y1y2y3y6y7y8).
in(8,s0y1y2y3y6y7y8).
in(0,s0y1y2y3y6y7y9).
in(1,s0y1y2y3y6y7y9).
in(2,s0y1y2y3y6y7y9).
in(3,s0y1y2y3y6y7y9).
in(6,s0y1y2y3y6y7y9).
in(7,s0y1y2y3y6y7y9).
in(9,s0y1y2y3y6y7y9).
in(0,s0y1y2y3y6y8y9).
in(1,s0y1y2y3y6y8y9).
in(2,s0y1y2y3y6y8y9).
in(3,s0y1y2y3y6y8y9).
in(6,s0y1y2y3y6y8y9).
in(8,s0y1y2y3y6y8y9).
in(9,s0y1y2y3y6y8y9).
in(0,s0y1y2y3y7y8y9).
in(1,s0y1y2y3y7y8y9).
in(2,s0y1y2y3y7y8y9).
in(3,s0y1y2y3y7y8y9).
in(7,s0y1y2y3y7y8y9).
in(8,s0y1y2y3y7y8y9).
in(9,s0y1y2y3y7y8y9).
in(0,s0y1y2y4y5y6y7).
in(1,s0y1y2y4y5y6y7).
in(2,s0y1y2y4y5y6y7).
in(4,s0y1y2y4y5y6y7).
in(5,s0y1y2y4y5y6y7).
in(6,s0y1y2y4y5y6y7).
in(7,s0y1y2y4y5y6y7).
in(0,s0y1y2y4y5y6y8).
in(1,s0y1y2y4y5y6y8).
in(2,s0y1y2y4y5y6y8).
in(4,s0y1y2y4y5y6y8).
in(5,s0y1y2y4y5y6y8).
in(6,s0y1y2y4y5y6y8).
in(8,s0y1y2y4y5y6y8).
in(0,s0y1y2y4y5y6y9).
in(1,s0y1y2y4y5y6y9).
in(2,s0y1y2y4y5y6y9).
in(4,s0y1y2y4y5y6y9).
in(5,s0y1y2y4y5y6y9).
in(6,s0y1y2y4y5y6y9).
in(9,s0y1y2y4y5y6y9).
in(0,s0y1y2y4y5y7y8).
in(1,s0y1y2y4y5y7y8).
in(2,s0y1y2y4y5y7y8).
in(4,s0y1y2y4y5y7y8).
in(5,s0y1y2y4y5y7y8).
in(7,s0y1y2y4y5y7y8).
in(8,s0y1y2y4y5y7y8).
in(0,s0y1y2y4y5y7y9).
in(1,s0y1y2y4y5y7y9).
in(2,s0y1y2y4y5y7y9).
in(4,s0y1y2y4y5y7y9).
in(5,s0y1y2y4y5y7y9).
in(7,s0y1y2y4y5y7y9).
in(9,s0y1y2y4y5y7y9).
in(0,s0y1y2y4y5y8y9).
in(1,s0y1y2y4y5y8y9).
in(2,s0y1y2y4y5y8y9).
in(4,s0y1y2y4y5y8y9).
in(5,s0y1y2y4y5y8y9).
in(8,s0y1y2y4y5y8y9).
in(9,s0y1y2y4y5y8y9).
in(0,s0y1y2y4y6y7y8).
in(1,s0y1y2y4y6y7y8).
in(2,s0y1y2y4y6y7y8).
in(4,s0y1y2y4y6y7y8).
in(6,s0y1y2y4y6y7y8).
in(7,s0y1y2y4y6y7y8).
in(8,s0y1y2y4y6y7y8).
in(0,s0y1y2y4y6y7y9).
in(1,s0y1y2y4y6y7y9).
in(2,s0y1y2y4y6y7y9).
in(4,s0y1y2y4y6y7y9).
in(6,s0y1y2y4y6y7y9).
in(7,s0y1y2y4y6y7y9).
in(9,s0y1y2y4y6y7y9).
in(0,s0y1y2y4y6y8y9).
in(1,s0y1y2y4y6y8y9).
in(2,s0y1y2y4y6y8y9).
in(4,s0y1y2y4y6y8y9).
in(6,s0y1y2y4y6y8y9).
in(8,s0y1y2y4y6y8y9).
in(9,s0y1y2y4y6y8y9).
in(0,s0y1y2y4y7y8y9).
in(1,s0y1y2y4y7y8y9).
in(2,s0y1y2y4y7y8y9).
in(4,s0y1y2y4y7y8y9).
in(7,s0y1y2y4y7y8y9).
in(8,s0y1y2y4y7y8y9).
in(9,s0y1y2y4y7y8y9).
in(0,s0y1y2y5y6y7y8).
in(1,s0y1y2y5y6y7y8).
in(2,s0y1y2y5y6y7y8).
in(5,s0y1y2y5y6y7y8).
in(6,s0y1y2y5y6y7y8).
in(7,s0y1y2y5y6y7y8).
in(8,s0y1y2y5y6y7y8).
in(0,s0y1y2y5y6y7y9).
in(1,s0y1y2y5y6y7y9).
in(2,s0y1y2y5y6y7y9).
in(5,s0y1y2y5y6y7y9).
in(6,s0y1y2y5y6y7y9).
in(7,s0y1y2y5y6y7y9).
in(9,s0y1y2y5y6y7y9).
in(0,s0y1y2y5y6y8y9).
in(1,s0y1y2y5y6y8y9).
in(2,s0y1y2y5y6y8y9).
in(5,s0y1y2y5y6y8y9).
in(6,s0y1y2y5y6y8y9).
in(8,s0y1y2y5y6y8y9).
in(9,s0y1y2y5y6y8y9).
in(0,s0y1y2y5y7y8y9).
in(1,s0y1y2y5y7y8y9).
in(2,s0y1y2y5y7y8y9).
in(5,s0y1y2y5y7y8y9).
in(7,s0y1y2y5y7y8y9).
in(8,s0y1y2y5y7y8y9).
in(9,s0y1y2y5y7y8y9).
in(0,s0y1y2y6y7y8y9).
in(1,s0y1y2y6y7y8y9).
in(2,s0y1y2y6y7y8y9).
in(6,s0y1y2y6y7y8y9).
in(7,s0y1y2y6y7y8y9).
in(8,s0y1y2y6y7y8y9).
in(9,s0y1y2y6y7y8y9).
in(0,s0y1y3y4y5y6y7).
in(1,s0y1y3y4y5y6y7).
in(3,s0y1y3y4y5y6y7).
in(4,s0y1y3y4y5y6y7).
in(5,s0y1y3y4y5y6y7).
in(6,s0y1y3y4y5y6y7).
in(7,s0y1y3y4y5y6y7).
in(0,s0y1y3y4y5y6y8).
in(1,s0y1y3y4y5y6y8).
in(3,s0y1y3y4y5y6y8).
in(4,s0y1y3y4y5y6y8).
in(5,s0y1y3y4y5y6y8).
in(6,s0y1y3y4y5y6y8).
in(8,s0y1y3y4y5y6y8).
in(0,s0y1y3y4y5y6y9).
in(1,s0y1y3y4y5y6y9).
in(3,s0y1y3y4y5y6y9).
in(4,s0y1y3y4y5y6y9).
in(5,s0y1y3y4y5y6y9).
in(6,s0y1y3y4y5y6y9).
in(9,s0y1y3y4y5y6y9).
in(0,s0y1y3y4y5y7y8).
in(1,s0y1y3y4y5y7y8).
in(3,s0y1y3y4y5y7y8).
in(4,s0y1y3y4y5y7y8).
in(5,s0y1y3y4y5y7y8).
in(7,s0y1y3y4y5y7y8).
in(8,s0y1y3y4y5y7y8).
in(0,s0y1y3y4y5y7y9).
in(1,s0y1y3y4y5y7y9).
in(3,s0y1y3y4y5y7y9).
in(4,s0y1y3y4y5y7y9).
in(5,s0y1y3y4y5y7y9).
in(7,s0y1y3y4y5y7y9).
in(9,s0y1y3y4y5y7y9).
in(0,s0y1y3y4y5y8y9).
in(1,s0y1y3y4y5y8y9).
in(3,s0y1y3y4y5y8y9).
in(4,s0y1y3y4y5y8y9).
in(5,s0y1y3y4y5y8y9).
in(8,s0y1y3y4y5y8y9).
in(9,s0y1y3y4y5y8y9).
in(0,s0y1y3y4y6y7y8).
in(1,s0y1y3y4y6y7y8).
in(3,s0y1y3y4y6y7y8).
in(4,s0y1y3y4y6y7y8).
in(6,s0y1y3y4y6y7y8).
in(7,s0y1y3y4y6y7y8).
in(8,s0y1y3y4y6y7y8).
in(0,s0y1y3y4y6y7y9).
in(1,s0y1y3y4y6y7y9).
in(3,s0y1y3y4y6y7y9).
in(4,s0y1y3y4y6y7y9).
in(6,s0y1y3y4y6y7y9).
in(7,s0y1y3y4y6y7y9).
in(9,s0y1y3y4y6y7y9).
in(0,s0y1y3y4y6y8y9).
in(1,s0y1y3y4y6y8y9).
in(3,s0y1y3y4y6y8y9).
in(4,s0y1y3y4y6y8y9).
in(6,s0y1y3y4y6y8y9).
in(8,s0y1y3y4y6y8y9).
in(9,s0y1y3y4y6y8y9).
in(0,s0y1y3y4y7y8y9).
in(1,s0y1y3y4y7y8y9).
in(3,s0y1y3y4y7y8y9).
in(4,s0y1y3y4y7y8y9).
in(7,s0y1y3y4y7y8y9).
in(8,s0y1y3y4y7y8y9).
in(9,s0y1y3y4y7y8y9).
in(0,s0y1y3y5y6y7y8).
in(1,s0y1y3y5y6y7y8).
in(3,s0y1y3y5y6y7y8).
in(5,s0y1y3y5y6y7y8).
in(6,s0y1y3y5y6y7y8).
in(7,s0y1y3y5y6y7y8).
in(8,s0y1y3y5y6y7y8).
in(0,s0y1y3y5y6y7y9).
in(1,s0y1y3y5y6y7y9).
in(3,s0y1y3y5y6y7y9).
in(5,s0y1y3y5y6y7y9).
in(6,s0y1y3y5y6y7y9).
in(7,s0y1y3y5y6y7y9).
in(9,s0y1y3y5y6y7y9).
in(0,s0y1y3y5y6y8y9).
in(1,s0y1y3y5y6y8y9).
in(3,s0y1y3y5y6y8y9).
in(5,s0y1y3y5y6y8y9).
in(6,s0y1y3y5y6y8y9).
in(8,s0y1y3y5y6y8y9).
in(9,s0y1y3y5y6y8y9).
in(0,s0y1y3y5y7y8y9).
in(1,s0y1y3y5y7y8y9).
in(3,s0y1y3y5y7y8y9).
in(5,s0y1y3y5y7y8y9).
in(7,s0y1y3y5y7y8y9).
in(8,s0y1y3y5y7y8y9).
in(9,s0y1y3y5y7y8y9).
in(0,s0y1y3y6y7y8y9).
in(1,s0y1y3y6y7y8y9).
in(3,s0y1y3y6y7y8y9).
in(6,s0y1y3y6y7y8y9).
in(7,s0y1y3y6y7y8y9).
in(8,s0y1y3y6y7y8y9).
in(9,s0y1y3y6y7y8y9).
in(0,s0y1y4y5y6y7y8).
in(1,s0y1y4y5y6y7y8).
in(4,s0y1y4y5y6y7y8).
in(5,s0y1y4y5y6y7y8).
in(6,s0y1y4y5y6y7y8).
in(7,s0y1y4y5y6y7y8).
in(8,s0y1y4y5y6y7y8).
in(0,s0y1y4y5y6y7y9).
in(1,s0y1y4y5y6y7y9).
in(4,s0y1y4y5y6y7y9).
in(5,s0y1y4y5y6y7y9).
in(6,s0y1y4y5y6y7y9).
in(7,s0y1y4y5y6y7y9).
in(9,s0y1y4y5y6y7y9).
in(0,s0y1y4y5y6y8y9).
in(1,s0y1y4y5y6y8y9).
in(4,s0y1y4y5y6y8y9).
in(5,s0y1y4y5y6y8y9).
in(6,s0y1y4y5y6y8y9).
in(8,s0y1y4y5y6y8y9).
in(9,s0y1y4y5y6y8y9).
in(0,s0y1y4y5y7y8y9).
in(1,s0y1y4y5y7y8y9).
in(4,s0y1y4y5y7y8y9).
in(5,s0y1y4y5y7y8y9).
in(7,s0y1y4y5y7y8y9).
in(8,s0y1y4y5y7y8y9).
in(9,s0y1y4y5y7y8y9).
in(0,s0y1y4y6y7y8y9).
in(1,s0y1y4y6y7y8y9).
in(4,s0y1y4y6y7y8y9).
in(6,s0y1y4y6y7y8y9).
in(7,s0y1y4y6y7y8y9).
in(8,s0y1y4y6y7y8y9).
in(9,s0y1y4y6y7y8y9).
in(0,s0y1y5y6y7y8y9).
in(1,s0y1y5y6y7y8y9).
in(5,s0y1y5y6y7y8y9).
in(6,s0y1y5y6y7y8y9).
in(7,s0y1y5y6y7y8y9).
in(8,s0y1y5y6y7y8y9).
in(9,s0y1y5y6y7y8y9).
in(0,s0y2y3y4y5y6y7).
in(2,s0y2y3y4y5y6y7).
in(3,s0y2y3y4y5y6y7).
in(4,s0y2y3y4y5y6y7).
in(5,s0y2y3y4y5y6y7).
in(6,s0y2y3y4y5y6y7).
in(7,s0y2y3y4y5y6y7).
in(0,s0y2y3y4y5y6y8).
in(2,s0y2y3y4y5y6y8).
in(3,s0y2y3y4y5y6y8).
in(4,s0y2y3y4y5y6y8).
in(5,s0y2y3y4y5y6y8).
in(6,s0y2y3y4y5y6y8).
in(8,s0y2y3y4y5y6y8).
in(0,s0y2y3y4y5y6y9).
in(2,s0y2y3y4y5y6y9).
in(3,s0y2y3y4y5y6y9).
in(4,s0y2y3y4y5y6y9).
in(5,s0y2y3y4y5y6y9).
in(6,s0y2y3y4y5y6y9).
in(9,s0y2y3y4y5y6y9).
in(0,s0y2y3y4y5y7y8).
in(2,s0y2y3y4y5y7y8).
in(3,s0y2y3y4y5y7y8).
in(4,s0y2y3y4y5y7y8).
in(5,s0y2y3y4y5y7y8).
in(7,s0y2y3y4y5y7y8).
in(8,s0y2y3y4y5y7y8).
in(0,s0y2y3y4y5y7y9).
in(2,s0y2y3y4y5y7y9).
in(3,s0y2y3y4y5y7y9).
in(4,s0y2y3y4y5y7y9).
in(5,s0y2y3y4y5y7y9).
in(7,s0y2y3y4y5y7y9).
in(9,s0y2y3y4y5y7y9).
in(0,s0y2y3y4y5y8y9).
in(2,s0y2y3y4y5y8y9).
in(3,s0y2y3y4y5y8y9).
in(4,s0y2y3y4y5y8y9).
in(5,s0y2y3y4y5y8y9).
in(8,s0y2y3y4y5y8y9).
in(9,s0y2y3y4y5y8y9).
in(0,s0y2y3y4y6y7y8).
in(2,s0y2y3y4y6y7y8).
in(3,s0y2y3y4y6y7y8).
in(4,s0y2y3y4y6y7y8).
in(6,s0y2y3y4y6y7y8).
in(7,s0y2y3y4y6y7y8).
in(8,s0y2y3y4y6y7y8).
in(0,s0y2y3y4y6y7y9).
in(2,s0y2y3y4y6y7y9).
in(3,s0y2y3y4y6y7y9).
in(4,s0y2y3y4y6y7y9).
in(6,s0y2y3y4y6y7y9).
in(7,s0y2y3y4y6y7y9).
in(9,s0y2y3y4y6y7y9).
in(0,s0y2y3y4y6y8y9).
in(2,s0y2y3y4y6y8y9).
in(3,s0y2y3y4y6y8y9).
in(4,s0y2y3y4y6y8y9).
in(6,s0y2y3y4y6y8y9).
in(8,s0y2y3y4y6y8y9).
in(9,s0y2y3y4y6y8y9).
in(0,s0y2y3y4y7y8y9).
in(2,s0y2y3y4y7y8y9).
in(3,s0y2y3y4y7y8y9).
in(4,s0y2y3y4y7y8y9).
in(7,s0y2y3y4y7y8y9).
in(8,s0y2y3y4y7y8y9).
in(9,s0y2y3y4y7y8y9).
in(0,s0y2y3y5y6y7y8).
in(2,s0y2y3y5y6y7y8).
in(3,s0y2y3y5y6y7y8).
in(5,s0y2y3y5y6y7y8).
in(6,s0y2y3y5y6y7y8).
in(7,s0y2y3y5y6y7y8).
in(8,s0y2y3y5y6y7y8).
in(0,s0y2y3y5y6y7y9).
in(2,s0y2y3y5y6y7y9).
in(3,s0y2y3y5y6y7y9).
in(5,s0y2y3y5y6y7y9).
in(6,s0y2y3y5y6y7y9).
in(7,s0y2y3y5y6y7y9).
in(9,s0y2y3y5y6y7y9).
in(0,s0y2y3y5y6y8y9).
in(2,s0y2y3y5y6y8y9).
in(3,s0y2y3y5y6y8y9).
in(5,s0y2y3y5y6y8y9).
in(6,s0y2y3y5y6y8y9).
in(8,s0y2y3y5y6y8y9).
in(9,s0y2y3y5y6y8y9).
in(0,s0y2y3y5y7y8y9).
in(2,s0y2y3y5y7y8y9).
in(3,s0y2y3y5y7y8y9).
in(5,s0y2y3y5y7y8y9).
in(7,s0y2y3y5y7y8y9).
in(8,s0y2y3y5y7y8y9).
in(9,s0y2y3y5y7y8y9).
in(0,s0y2y3y6y7y8y9).
in(2,s0y2y3y6y7y8y9).
in(3,s0y2y3y6y7y8y9).
in(6,s0y2y3y6y7y8y9).
in(7,s0y2y3y6y7y8y9).
in(8,s0y2y3y6y7y8y9).
in(9,s0y2y3y6y7y8y9).
in(0,s0y2y4y5y6y7y8).
in(2,s0y2y4y5y6y7y8).
in(4,s0y2y4y5y6y7y8).
in(5,s0y2y4y5y6y7y8).
in(6,s0y2y4y5y6y7y8).
in(7,s0y2y4y5y6y7y8).
in(8,s0y2y4y5y6y7y8).
in(0,s0y2y4y5y6y7y9).
in(2,s0y2y4y5y6y7y9).
in(4,s0y2y4y5y6y7y9).
in(5,s0y2y4y5y6y7y9).
in(6,s0y2y4y5y6y7y9).
in(7,s0y2y4y5y6y7y9).
in(9,s0y2y4y5y6y7y9).
in(0,s0y2y4y5y6y8y9).
in(2,s0y2y4y5y6y8y9).
in(4,s0y2y4y5y6y8y9).
in(5,s0y2y4y5y6y8y9).
in(6,s0y2y4y5y6y8y9).
in(8,s0y2y4y5y6y8y9).
in(9,s0y2y4y5y6y8y9).
in(0,s0y2y4y5y7y8y9).
in(2,s0y2y4y5y7y8y9).
in(4,s0y2y4y5y7y8y9).
in(5,s0y2y4y5y7y8y9).
in(7,s0y2y4y5y7y8y9).
in(8,s0y2y4y5y7y8y9).
in(9,s0y2y4y5y7y8y9).
in(0,s0y2y4y6y7y8y9).
in(2,s0y2y4y6y7y8y9).
in(4,s0y2y4y6y7y8y9).
in(6,s0y2y4y6y7y8y9).
in(7,s0y2y4y6y7y8y9).
in(8,s0y2y4y6y7y8y9).
in(9,s0y2y4y6y7y8y9).
in(0,s0y2y5y6y7y8y9).
in(2,s0y2y5y6y7y8y9).
in(5,s0y2y5y6y7y8y9).
in(6,s0y2y5y6y7y8y9).
in(7,s0y2y5y6y7y8y9).
in(8,s0y2y5y6y7y8y9).
in(9,s0y2y5y6y7y8y9).
in(0,s0y3y4y5y6y7y8).
in(3,s0y3y4y5y6y7y8).
in(4,s0y3y4y5y6y7y8).
in(5,s0y3y4y5y6y7y8).
in(6,s0y3y4y5y6y7y8).
in(7,s0y3y4y5y6y7y8).
in(8,s0y3y4y5y6y7y8).
in(0,s0y3y4y5y6y7y9).
in(3,s0y3y4y5y6y7y9).
in(4,s0y3y4y5y6y7y9).
in(5,s0y3y4y5y6y7y9).
in(6,s0y3y4y5y6y7y9).
in(7,s0y3y4y5y6y7y9).
in(9,s0y3y4y5y6y7y9).
in(0,s0y3y4y5y6y8y9).
in(3,s0y3y4y5y6y8y9).
in(4,s0y3y4y5y6y8y9).
in(5,s0y3y4y5y6y8y9).
in(6,s0y3y4y5y6y8y9).
in(8,s0y3y4y5y6y8y9).
in(9,s0y3y4y5y6y8y9).
in(0,s0y3y4y5y7y8y9).
in(3,s0y3y4y5y7y8y9).
in(4,s0y3y4y5y7y8y9).
in(5,s0y3y4y5y7y8y9).
in(7,s0y3y4y5y7y8y9).
in(8,s0y3y4y5y7y8y9).
in(9,s0y3y4y5y7y8y9).
in(0,s0y3y4y6y7y8y9).
in(3,s0y3y4y6y7y8y9).
in(4,s0y3y4y6y7y8y9).
in(6,s0y3y4y6y7y8y9).
in(7,s0y3y4y6y7y8y9).
in(8,s0y3y4y6y7y8y9).
in(9,s0y3y4y6y7y8y9).
in(0,s0y3y5y6y7y8y9).
in(3,s0y3y5y6y7y8y9).
in(5,s0y3y5y6y7y8y9).
in(6,s0y3y5y6y7y8y9).
in(7,s0y3y5y6y7y8y9).
in(8,s0y3y5y6y7y8y9).
in(9,s0y3y5y6y7y8y9).
in(0,s0y4y5y6y7y8y9).
in(4,s0y4y5y6y7y8y9).
in(5,s0y4y5y6y7y8y9).
in(6,s0y4y5y6y7y8y9).
in(7,s0y4y5y6y7y8y9).
in(8,s0y4y5y6y7y8y9).
in(9,s0y4y5y6y7y8y9).
in(1,s1y2y3y4y5y6y7).
in(2,s1y2y3y4y5y6y7).
in(3,s1y2y3y4y5y6y7).
in(4,s1y2y3y4y5y6y7).
in(5,s1y2y3y4y5y6y7).
in(6,s1y2y3y4y5y6y7).
in(7,s1y2y3y4y5y6y7).
in(1,s1y2y3y4y5y6y8).
in(2,s1y2y3y4y5y6y8).
in(3,s1y2y3y4y5y6y8).
in(4,s1y2y3y4y5y6y8).
in(5,s1y2y3y4y5y6y8).
in(6,s1y2y3y4y5y6y8).
in(8,s1y2y3y4y5y6y8).
in(1,s1y2y3y4y5y6y9).
in(2,s1y2y3y4y5y6y9).
in(3,s1y2y3y4y5y6y9).
in(4,s1y2y3y4y5y6y9).
in(5,s1y2y3y4y5y6y9).
in(6,s1y2y3y4y5y6y9).
in(9,s1y2y3y4y5y6y9).
in(1,s1y2y3y4y5y7y8).
in(2,s1y2y3y4y5y7y8).
in(3,s1y2y3y4y5y7y8).
in(4,s1y2y3y4y5y7y8).
in(5,s1y2y3y4y5y7y8).
in(7,s1y2y3y4y5y7y8).
in(8,s1y2y3y4y5y7y8).
in(1,s1y2y3y4y5y7y9).
in(2,s1y2y3y4y5y7y9).
in(3,s1y2y3y4y5y7y9).
in(4,s1y2y3y4y5y7y9).
in(5,s1y2y3y4y5y7y9).
in(7,s1y2y3y4y5y7y9).
in(9,s1y2y3y4y5y7y9).
in(1,s1y2y3y4y5y8y9).
in(2,s1y2y3y4y5y8y9).
in(3,s1y2y3y4y5y8y9).
in(4,s1y2y3y4y5y8y9).
in(5,s1y2y3y4y5y8y9).
in(8,s1y2y3y4y5y8y9).
in(9,s1y2y3y4y5y8y9).
in(1,s1y2y3y4y6y7y8).
in(2,s1y2y3y4y6y7y8).
in(3,s1y2y3y4y6y7y8).
in(4,s1y2y3y4y6y7y8).
in(6,s1y2y3y4y6y7y8).
in(7,s1y2y3y4y6y7y8).
in(8,s1y2y3y4y6y7y8).
in(1,s1y2y3y4y6y7y9).
in(2,s1y2y3y4y6y7y9).
in(3,s1y2y3y4y6y7y9).
in(4,s1y2y3y4y6y7y9).
in(6,s1y2y3y4y6y7y9).
in(7,s1y2y3y4y6y7y9).
in(9,s1y2y3y4y6y7y9).
in(1,s1y2y3y4y6y8y9).
in(2,s1y2y3y4y6y8y9).
in(3,s1y2y3y4y6y8y9).
in(4,s1y2y3y4y6y8y9).
in(6,s1y2y3y4y6y8y9).
in(8,s1y2y3y4y6y8y9).
in(9,s1y2y3y4y6y8y9).
in(1,s1y2y3y4y7y8y9).
in(2,s1y2y3y4y7y8y9).
in(3,s1y2y3y4y7y8y9).
in(4,s1y2y3y4y7y8y9).
in(7,s1y2y3y4y7y8y9).
in(8,s1y2y3y4y7y8y9).
in(9,s1y2y3y4y7y8y9).
in(1,s1y2y3y5y6y7y8).
in(2,s1y2y3y5y6y7y8).
in(3,s1y2y3y5y6y7y8).
in(5,s1y2y3y5y6y7y8).
in(6,s1y2y3y5y6y7y8).
in(7,s1y2y3y5y6y7y8).
in(8,s1y2y3y5y6y7y8).
in(1,s1y2y3y5y6y7y9).
in(2,s1y2y3y5y6y7y9).
in(3,s1y2y3y5y6y7y9).
in(5,s1y2y3y5y6y7y9).
in(6,s1y2y3y5y6y7y9).
in(7,s1y2y3y5y6y7y9).
in(9,s1y2y3y5y6y7y9).
in(1,s1y2y3y5y6y8y9).
in(2,s1y2y3y5y6y8y9).
in(3,s1y2y3y5y6y8y9).
in(5,s1y2y3y5y6y8y9).
in(6,s1y2y3y5y6y8y9).
in(8,s1y2y3y5y6y8y9).
in(9,s1y2y3y5y6y8y9).
in(1,s1y2y3y5y7y8y9).
in(2,s1y2y3y5y7y8y9).
in(3,s1y2y3y5y7y8y9).
in(5,s1y2y3y5y7y8y9).
in(7,s1y2y3y5y7y8y9).
in(8,s1y2y3y5y7y8y9).
in(9,s1y2y3y5y7y8y9).
in(1,s1y2y3y6y7y8y9).
in(2,s1y2y3y6y7y8y9).
in(3,s1y2y3y6y7y8y9).
in(6,s1y2y3y6y7y8y9).
in(7,s1y2y3y6y7y8y9).
in(8,s1y2y3y6y7y8y9).
in(9,s1y2y3y6y7y8y9).
in(1,s1y2y4y5y6y7y8).
in(2,s1y2y4y5y6y7y8).
in(4,s1y2y4y5y6y7y8).
in(5,s1y2y4y5y6y7y8).
in(6,s1y2y4y5y6y7y8).
in(7,s1y2y4y5y6y7y8).
in(8,s1y2y4y5y6y7y8).
in(1,s1y2y4y5y6y7y9).
in(2,s1y2y4y5y6y7y9).
in(4,s1y2y4y5y6y7y9).
in(5,s1y2y4y5y6y7y9).
in(6,s1y2y4y5y6y7y9).
in(7,s1y2y4y5y6y7y9).
in(9,s1y2y4y5y6y7y9).
in(1,s1y2y4y5y6y8y9).
in(2,s1y2y4y5y6y8y9).
in(4,s1y2y4y5y6y8y9).
in(5,s1y2y4y5y6y8y9).
in(6,s1y2y4y5y6y8y9).
in(8,s1y2y4y5y6y8y9).
in(9,s1y2y4y5y6y8y9).
in(1,s1y2y4y5y7y8y9).
in(2,s1y2y4y5y7y8y9).
in(4,s1y2y4y5y7y8y9).
in(5,s1y2y4y5y7y8y9).
in(7,s1y2y4y5y7y8y9).
in(8,s1y2y4y5y7y8y9).
in(9,s1y2y4y5y7y8y9).
in(1,s1y2y4y6y7y8y9).
in(2,s1y2y4y6y7y8y9).
in(4,s1y2y4y6y7y8y9).
in(6,s1y2y4y6y7y8y9).
in(7,s1y2y4y6y7y8y9).
in(8,s1y2y4y6y7y8y9).
in(9,s1y2y4y6y7y8y9).
in(1,s1y2y5y6y7y8y9).
in(2,s1y2y5y6y7y8y9).
in(5,s1y2y5y6y7y8y9).
in(6,s1y2y5y6y7y8y9).
in(7,s1y2y5y6y7y8y9).
in(8,s1y2y5y6y7y8y9).
in(9,s1y2y5y6y7y8y9).
in(1,s1y3y4y5y6y7y8).
in(3,s1y3y4y5y6y7y8).
in(4,s1y3y4y5y6y7y8).
in(5,s1y3y4y5y6y7y8).
in(6,s1y3y4y5y6y7y8).
in(7,s1y3y4y5y6y7y8).
in(8,s1y3y4y5y6y7y8).
in(1,s1y3y4y5y6y7y9).
in(3,s1y3y4y5y6y7y9).
in(4,s1y3y4y5y6y7y9).
in(5,s1y3y4y5y6y7y9).
in(6,s1y3y4y5y6y7y9).
in(7,s1y3y4y5y6y7y9).
in(9,s1y3y4y5y6y7y9).
in(1,s1y3y4y5y6y8y9).
in(3,s1y3y4y5y6y8y9).
in(4,s1y3y4y5y6y8y9).
in(5,s1y3y4y5y6y8y9).
in(6,s1y3y4y5y6y8y9).
in(8,s1y3y4y5y6y8y9).
in(9,s1y3y4y5y6y8y9).
in(1,s1y3y4y5y7y8y9).
in(3,s1y3y4y5y7y8y9).
in(4,s1y3y4y5y7y8y9).
in(5,s1y3y4y5y7y8y9).
in(7,s1y3y4y5y7y8y9).
in(8,s1y3y4y5y7y8y9).
in(9,s1y3y4y5y7y8y9).
in(1,s1y3y4y6y7y8y9).
in(3,s1y3y4y6y7y8y9).
in(4,s1y3y4y6y7y8y9).
in(6,s1y3y4y6y7y8y9).
in(7,s1y3y4y6y7y8y9).
in(8,s1y3y4y6y7y8y9).
in(9,s1y3y4y6y7y8y9).
in(1,s1y3y5y6y7y8y9).
in(3,s1y3y5y6y7y8y9).
in(5,s1y3y5y6y7y8y9).
in(6,s1y3y5y6y7y8y9).
in(7,s1y3y5y6y7y8y9).
in(8,s1y3y5y6y7y8y9).
in(9,s1y3y5y6y7y8y9).
in(1,s1y4y5y6y7y8y9).
in(4,s1y4y5y6y7y8y9).
in(5,s1y4y5y6y7y8y9).
in(6,s1y4y5y6y7y8y9).
in(7,s1y4y5y6y7y8y9).
in(8,s1y4y5y6y7y8y9).
in(9,s1y4y5y6y7y8y9).
in(2,s2y3y4y5y6y7y8).
in(3,s2y3y4y5y6y7y8).
in(4,s2y3y4y5y6y7y8).
in(5,s2y3y4y5y6y7y8).
in(6,s2y3y4y5y6y7y8).
in(7,s2y3y4y5y6y7y8).
in(8,s2y3y4y5y6y7y8).
in(2,s2y3y4y5y6y7y9).
in(3,s2y3y4y5y6y7y9).
in(4,s2y3y4y5y6y7y9).
in(5,s2y3y4y5y6y7y9).
in(6,s2y3y4y5y6y7y9).
in(7,s2y3y4y5y6y7y9).
in(9,s2y3y4y5y6y7y9).
in(2,s2y3y4y5y6y8y9).
in(3,s2y3y4y5y6y8y9).
in(4,s2y3y4y5y6y8y9).
in(5,s2y3y4y5y6y8y9).
in(6,s2y3y4y5y6y8y9).
in(8,s2y3y4y5y6y8y9).
in(9,s2y3y4y5y6y8y9).
in(2,s2y3y4y5y7y8y9).
in(3,s2y3y4y5y7y8y9).
in(4,s2y3y4y5y7y8y9).
in(5,s2y3y4y5y7y8y9).
in(7,s2y3y4y5y7y8y9).
in(8,s2y3y4y5y7y8y9).
in(9,s2y3y4y5y7y8y9).
in(2,s2y3y4y6y7y8y9).
in(3,s2y3y4y6y7y8y9).
in(4,s2y3y4y6y7y8y9).
in(6,s2y3y4y6y7y8y9).
in(7,s2y3y4y6y7y8y9).
in(8,s2y3y4y6y7y8y9).
in(9,s2y3y4y6y7y8y9).
in(2,s2y3y5y6y7y8y9).
in(3,s2y3y5y6y7y8y9).
in(5,s2y3y5y6y7y8y9).
in(6,s2y3y5y6y7y8y9).
in(7,s2y3y5y6y7y8y9).
in(8,s2y3y5y6y7y8y9).
in(9,s2y3y5y6y7y8y9).
in(2,s2y4y5y6y7y8y9).
in(4,s2y4y5y6y7y8y9).
in(5,s2y4y5y6y7y8y9).
in(6,s2y4y5y6y7y8y9).
in(7,s2y4y5y6y7y8y9).
in(8,s2y4y5y6y7y8y9).
in(9,s2y4y5y6y7y8y9).
in(3,s3y4y5y6y7y8y9).
in(4,s3y4y5y6y7y8y9).
in(5,s3y4y5y6y7y8y9).
in(6,s3y4y5y6y7y8y9).
in(7,s3y4y5y6y7y8y9).
in(8,s3y4y5y6y7y8y9).
in(9,s3y4y5y6y7y8y9).
qpair(0,1,s2).
qpair(0,1,s4y6y9).
qpair(0,1,s4y9).
qpair(0,1,s4y6y8y9).
qpair(0,1,s4y6y8).
qpair(0,1,s2y3).
qpair(0,1,s7).
qpair(0,1,s2y3y7).
qpair(0,1,s4y8y9).
qpair(0,2,s3y8).
qpair(0,2,s5).
qpair(0,2,s3y7).
qpair(0,2,s6y8).
qpair(0,2,s8).
qpair(0,2,s4).
qpair(0,2,s1).
qpair(0,2,s6y7).
qpair(0,2,s3y6).
qpair(0,2,s3y7y8).
qpair(0,2,empty).
qpair(0,2,s3).
qpair(0,2,s9).
qpair(0,2,s7y8).
qpair(0,3,s2).
qpair(0,3,s8).
qpair(0,3,s8y9).
qpair(0,3,s1y9).
qpair(0,3,s2y8).
qpair(0,3,s6y8).
qpair(0,3,s4).
qpair(0,3,s1).
qpair(0,3,s7).
qpair(0,3,s5y9).
qpair(0,3,s2y7).
qpair(0,3,s1y5).
qpair(0,3,s6y7).
qpair(0,3,s6).
qpair(0,3,s5y8y9).
qpair(0,3,s5y8).
qpair(0,3,s2y7y8).
qpair(0,3,s2y6).
qpair(0,3,empty).
qpair(0,3,s7y8).
qpair(0,5,s6).
qpair(0,5,s9).
qpair(0,6,s1y5).
qpair(0,6,s5).
qpair(0,6,s5y8).
qpair(0,6,s1y4).
qpair(0,6,s4).
qpair(0,6,s4y7).
qpair(0,6,s5y7).
qpair(0,6,s4y5).
qpair(0,6,s4y8).
qpair(0,6,s9).
qpair(0,7,s6).
qpair(0,7,s2).
qpair(0,7,s5y8).
qpair(0,7,s6y8).
qpair(0,7,s8).
qpair(0,7,s4).
qpair(0,7,s2y3).
qpair(0,7,s1).
qpair(0,7,s2y3y8).
qpair(0,7,s2y6).
qpair(0,7,s5y6).
qpair(0,7,empty).
qpair(0,7,s3).
qpair(0,7,s9).
qpair(0,7,s2y8).
qpair(0,8,s4y9).
qpair(0,8,s1y5y7).
qpair(0,8,s3y5y9).
qpair(0,8,s1y3y4y5y9).
qpair(0,8,s1y2y3y4y9).
qpair(0,8,s1y2y4y6).
qpair(0,8,s1y3y9).
qpair(0,8,s3y4y5y6).
qpair(0,8,s2y4y5).
qpair(0,8,s4y5y6y9).
qpair(0,8,s1y2y3y4y5y6y9).
qpair(0,8,s1y5y9).
qpair(0,8,s3y6).
qpair(0,8,s2y3y4y5y6).
qpair(0,8,s1y4y6y9).
qpair(0,8,s1y2y3y9).
qpair(0,8,s2y6y7).
qpair(0,8,s2y4).
qpair(0,8,s4y6y9).
qpair(0,8,s2y4y5y9).
qpair(0,8,s2y4y7).
qpair(0,8,s1y2y3y5y6).
qpair(0,8,s1y3y5y6).
qpair(0,8,s6).
qpair(0,8,s2y6y9).
qpair(0,8,s1y3y4).
qpair(0,8,s1y2y5y6y9).
qpair(0,8,s1y2y3y4y5y6).
qpair(0,8,s3y4y5).
qpair(0,8,s1y2y4y5y6y9).
qpair(0,8,s2y4y9).
qpair(0,8,empty).
qpair(0,8,s3y5y6).
qpair(0,8,s5y7).
qpair(0,8,s1y5y6y9).
qpair(0,8,s1y2y3y4).
qpair(0,8,s4y5y6).
qpair(0,8,s1y3y6).
qpair(0,8,s1y4y5).
qpair(0,8,s1y3).
qpair(0,8,s3y4y7).
qpair(0,8,s1y3y4y5y6y9).
qpair(0,8,s5y7y9).
qpair(0,8,s1y2y4y6y9).
qpair(0,8,s2y3y6).
qpair(0,8,s1y5y6).
qpair(0,8,s1).
qpair(0,8,s5y6y7).
qpair(0,8,s2y3y4y5y6y9).
qpair(0,8,s1y3y6y9).
qpair(0,8,s5y9).
qpair(0,8,s1y2y3y6).
qpair(0,8,s2y5y6).
qpair(0,8,s1y5).
qpair(0,8,s2y4y5y6).
qpair(0,8,s3y4y9).
qpair(0,8,s2y3y5y6y9).
qpair(0,8,s7y9).
qpair(0,8,s2y3y5y9).
qpair(0,8,s1y6y7).
qpair(0,8,s3y5).
qpair(0,8,s1y2y3).
qpair(0,8,s2y4y5y6y9).
qpair(0,8,s1y4y9).
qpair(0,8,s1y7).
qpair(0,8,s6y7y9).
qpair(0,8,s1y2y3y5y6y9).
qpair(0,8,s2y3y4y6y9).
qpair(0,8,s5).
qpair(0,8,s1y6y9).
qpair(0,8,s3y7).
qpair(0,8,s4y6).
qpair(0,8,s1y2y4y9).
qpair(0,8,s1y2y3y4y6).
qpair(0,8,s1y4y5y6y9).
qpair(0,8,s4y7y9).
qpair(0,8,s1y3y4y6).
qpair(0,8,s1y9).
qpair(0,8,s3y6y7).
qpair(0,8,s1y2y6y9).
qpair(0,8,s7).
qpair(0,8,s3y9).
qpair(0,8,s5y6).
qpair(0,8,s2y3y4y5).
qpair(0,8,s2y7y9).
qpair(0,8,s1y2y7).
qpair(0,8,s1y2).
qpair(0,8,s3y4y6).
qpair(0,8,s2y3y5).
qpair(0,8,s3y6y9).
qpair(0,8,s1y3y4y6y9).
qpair(0,8,s1y2y3y5y9).
qpair(0,8,s1y2y5y6).
qpair(0,8,s9).
qpair(0,8,s5y6y9).
qpair(0,8,s1y2y4y5y9).
qpair(0,8,s4y5y7).
qpair(0,8,s1y4y5y9).
qpair(0,8,s1y2y9).
qpair(0,8,s1y4y6).
qpair(0,8,s1y4).
qpair(0,8,s1y2y3y4y5y9).
qpair(0,8,s2y3).
qpair(0,8,s2y3y7).
qpair(0,8,s2y4y6).
qpair(0,8,s2y3y6y9).
qpair(0,8,s3y4).
qpair(0,8,s2y3y4y9).
qpair(0,8,s2y4y6y9).
qpair(0,8,s2y5y7).
qpair(0,8,s3y7y9).
qpair(0,8,s1y6).
qpair(0,8,s2y5).
qpair(0,8,s2y3y9).
qpair(0,8,s3y4y5y9).
qpair(0,8,s4).
qpair(0,8,s4y5).
qpair(0,8,s3y4y5y6y9).
qpair(0,8,s1y7y9).
qpair(0,8,s2y5y9).
qpair(0,8,s1y2y3y4y5).
qpair(0,8,s2y3y4y5y9).
qpair(0,8,s2y7).
qpair(0,8,s1y3y5).
qpair(0,8,s6y7).
qpair(0,8,s2y3y5y6).
qpair(0,8,s4y7).
qpair(0,8,s3y5y7).
qpair(0,8,s1y3y5y9).
qpair(0,8,s1y4y5y6).
qpair(0,8,s1y2y3y5).
qpair(0,8,s2y9).
qpair(0,8,s1y3y7).
qpair(0,8,s2y3y4).
qpair(0,8,s6y9).
qpair(0,9,s6).
qpair(0,9,s5).
qpair(1,2,s4y6y9).
qpair(1,2,s6y9).
qpair(1,2,s4y9).
qpair(1,2,s4y6y8y9).
qpair(1,2,s4y6).
qpair(1,2,s7).
qpair(1,2,s8y9).
qpair(1,2,s4y8y9).
qpair(1,2,s3).
qpair(1,4,s6).
qpair(1,4,s3y8).
qpair(1,4,s6y9).
qpair(1,4,s6y8).
qpair(1,4,s8).
qpair(1,4,s7).
qpair(1,4,s8y9).
qpair(1,4,s3y9).
qpair(1,4,s3y6).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,4,s9).
qpair(1,5,s4y8).
qpair(1,5,s4).
qpair(1,6,s2).
qpair(1,6,s3y4).
qpair(1,6,s5).
qpair(1,6,s4y9).
qpair(1,6,s8).
qpair(1,6,s5y7).
qpair(1,6,s8y9).
qpair(1,6,s0y5).
qpair(1,6,s0y8).
qpair(1,6,s4).
qpair(1,6,s7).
qpair(1,6,s4y5).
qpair(1,6,s3y9).
qpair(1,6,s4y8).
qpair(1,6,s0y4).
qpair(1,6,s0).
qpair(1,6,s4y8y9).
qpair(1,6,s3).
qpair(1,6,s4y7).
qpair(1,6,s9).
qpair(1,6,s3y8).
qpair(1,6,s5y8).
qpair(1,6,s7y8).
qpair(1,7,s3).
qpair(1,7,s0y5).
qpair(1,7,s6y9).
qpair(1,8,s4y9).
qpair(1,8,s0y5).
qpair(1,8,s4y6y7).
qpair(1,8,s0y2y4y6).
qpair(1,8,s3y4y5y6).
qpair(1,8,s0y2y4).
qpair(1,8,s3y6).
qpair(1,8,s2y3y4y5y6).
qpair(1,8,s0y7).
qpair(1,8,s2y6y7).
qpair(1,8,s0y3y5y9).
qpair(1,8,s4y6y9).
qpair(1,8,s2y4y5y9).
qpair(1,8,s2y4y7).
qpair(1,8,s0y2y6).
qpair(1,8,s0y3y4y5y9).
qpair(1,8,s0y4y9).
qpair(1,8,s6).
qpair(1,8,s0y2y3y9).
qpair(1,8,s0y3y7).
qpair(1,8,s0y9).
qpair(1,8,s2y6y9).
qpair(1,8,s3y4y5).
qpair(1,8,s0y2y4y5y9).
qpair(1,8,s0y2y5y9).
qpair(1,8,s2y4y9).
qpair(1,8,s3y4y6y9).
qpair(1,8,empty).
qpair(1,8,s0y3y5y6y9).
qpair(1,8,s5y7).
qpair(1,8,s2y3y4y6).
qpair(1,8,s0y2).
qpair(1,8,s0y4y5y9).
qpair(1,8,s0y2y3y4y6y9).
qpair(1,8,s4y5y6).
qpair(1,8,s3y4y7).
qpair(1,8,s5y7y9).
qpair(1,8,s2y3y6).
qpair(1,8,s0y2y3y5y9).
qpair(1,8,s0y2y5y6y9).
qpair(1,8,s5y6y7).
qpair(1,8,s0y2y3y4).
qpair(1,8,s2y3y4y5y6y9).
qpair(1,8,s5y9).
qpair(1,8,s0y3y4y5y6y9).
qpair(1,8,s0y2y4y5).
qpair(1,8,s2y4y5y6).
qpair(1,8,s3y4y9).
qpair(1,8,s3y5y6y9).
qpair(1,8,s0y2y3).
qpair(1,8,s0y3y4y5y6).
qpair(1,8,s2y3y5y9).
qpair(1,8,s0y2y3y6y9).
qpair(1,8,s0y4y6).
qpair(1,8,s0y2y3y6).
qpair(1,8,s2y4y5y6y9).
qpair(1,8,s0y2y4y5y6).
qpair(1,8,s2y6).
qpair(1,8,s0y2y3y4y5).
qpair(1,8,s6y7y9).
qpair(1,8,s0y5y7).
qpair(1,8,s5).
qpair(1,8,s3y7).
qpair(1,8,s0y4y5y6).
qpair(1,8,s0y3y9).
qpair(1,8,s4y7y9).
qpair(1,8,s0y2y7).
qpair(1,8,s3y6y7).
qpair(1,8,s0y2y3y5y6).
qpair(1,8,s0y3y4y6y9).
qpair(1,8,s0y6y7).
qpair(1,8,s0y5y9).
qpair(1,8,s7).
qpair(1,8,s3y9).
qpair(1,8,s5y6).
qpair(1,8,s2y3y4y5).
qpair(1,8,s2y7y9).
qpair(1,8,s0y4).
qpair(1,8,s0y3y5y6).
qpair(1,8,s0y2y3y4y5y6y9).
qpair(1,8,s2y3y5).
qpair(1,8,s3y6y9).
qpair(1,8,s0).
qpair(1,8,s0y6y9).
qpair(1,8,s2y5y6y9).
qpair(1,8,s0y5y6y9).
qpair(1,8,s0y3y4).
qpair(1,8,s4y5y7).
qpair(1,8,s0y6).
qpair(1,8,s2y3).
qpair(1,8,s2y3y7).
qpair(1,8,s2y4y6).
qpair(1,8,s0y4y5).
qpair(1,8,s2y3y6y9).
qpair(1,8,s2).
qpair(1,8,s0y2y3y5).
qpair(1,8,s3y4).
qpair(1,8,s0y3y6y9).
qpair(1,8,s0y2y4y5y6y9).
qpair(1,8,s2y3y4y9).
qpair(1,8,s2y4y6y9).
qpair(1,8,s2y5y7).
qpair(1,8,s3y7y9).
qpair(1,8,s0y3y6).
qpair(1,8,s4y5y9).
qpair(1,8,s2y5).
qpair(1,8,s0y2y4y9).
qpair(1,8,s2y3y9).
qpair(1,8,s3y4y5y9).
qpair(1,8,s0y4y7).
qpair(1,8,s4y5).
qpair(1,8,s3y4y5y6y9).
qpair(1,8,s0y2y4y6y9).
qpair(1,8,s2y5y9).
qpair(1,8,s2y3y4y5y9).
qpair(1,8,s2y7).
qpair(1,8,s6y7).
qpair(1,8,s0y4y6y9).
qpair(1,8,s0y2y3y4y9).
qpair(1,8,s4y7).
qpair(1,8,s3y5y7).
qpair(1,8,s0y2y3y4y5y9).
qpair(1,8,s2y9).
qpair(1,8,s2y3y4).
qpair(1,8,s6y9).
qpair(1,9,s6).
qpair(1,9,s3y8).
qpair(1,9,s2).
qpair(1,9,s5).
qpair(1,9,s6y8).
qpair(1,9,s8).
qpair(1,9,s4y6).
qpair(1,9,s4y6y8).
qpair(1,9,s4).
qpair(1,9,s7).
qpair(1,9,s4y8).
qpair(1,9,s3y6).
qpair(1,9,empty).
qpair(2,3,s6).
qpair(2,3,s1).
qpair(2,3,s7).
qpair(2,3,s4).
qpair(2,4,s6).
qpair(2,4,s5).
qpair(2,5,s6).
qpair(2,5,s1).
qpair(2,5,s3y7y8).
qpair(2,5,s3y7).
qpair(2,6,s1y7y8).
qpair(2,6,s8).
qpair(2,6,s1y8).
qpair(2,6,s1).
qpair(2,6,s1y7).
qpair(2,6,s0).
qpair(2,6,s0y8).
qpair(2,6,s7y8).
qpair(2,7,s6).
qpair(2,7,s5).
qpair(2,7,s0y5y6).
qpair(2,7,s5y6).
qpair(2,7,s0).
qpair(2,7,s0y8).
qpair(2,8,s0y1y4y5).
qpair(2,8,s4y9).
qpair(2,8,s1y5y7).
qpair(2,8,s0y1y6).
qpair(2,8,s1y3y4y5y9).
qpair(2,8,s0y5).
qpair(2,8,s4y6y7).
qpair(2,8,s1y3y9).
qpair(2,8,s3y4y5y6).
qpair(2,8,s4y5y6y9).
qpair(2,8,s0y1y3y9).
qpair(2,8,s0y1y5y6).
qpair(2,8,s1y5y9).
qpair(2,8,s0y1y3y5y9).
qpair(2,8,s3y6).
qpair(2,8,s0y3y5).
qpair(2,8,s1y4y6y9).
qpair(2,8,s0y1y4y5y6).
qpair(2,8,s0y7).
qpair(2,8,s0y3y5y9).
qpair(2,8,s0y3y4y5y9).
qpair(2,8,s0y1y4y9).
qpair(2,8,s0y4y9).
qpair(2,8,s1y3y5y6).
qpair(2,8,s6).
qpair(2,8,s0y3y7).
qpair(2,8,s0y1y3y4y6).
qpair(2,8,s0y9).
qpair(2,8,s1y3y4).
qpair(2,8,s3y4y6y9).
qpair(2,8,empty).
qpair(2,8,s0y1y4y5y6y9).
qpair(2,8,s0y1y3y4).
qpair(2,8,s0y1y3y4y5y9).
qpair(2,8,s3y5y6).
qpair(2,8,s0y1y3).
qpair(2,8,s0y3y5y6y9).
qpair(2,8,s0y1y5y6y9).
qpair(2,8,s5y7).
qpair(2,8,s1y5y6y9).
qpair(2,8,s0y4y5y9).
qpair(2,8,s4y5y6).
qpair(2,8,s1y3y6).
qpair(2,8,s1y4y5).
qpair(2,8,s1y3).
qpair(2,8,s3y4y7).
qpair(2,8,s1y3y4y9).
qpair(2,8,s1y3y4y5y6y9).
qpair(2,8,s0y1y3y6).
qpair(2,8,s1y5y6).
qpair(2,8,s1).
qpair(2,8,s5y6y7).
qpair(2,8,s0y7y9).
qpair(2,8,s0y1y5).
qpair(2,8,s1y3y6y9).
qpair(2,8,s5y9).
qpair(2,8,s0y3y4y5y6y9).
qpair(2,8,s1y4y7).
qpair(2,8,s1y5).
qpair(2,8,s7y9).
qpair(2,8,s0y4y6).
qpair(2,8,s3).
qpair(2,8,s1y6y7).
qpair(2,8,s3y5).
qpair(2,8,s1y4y9).
qpair(2,8,s0y1y6y9).
qpair(2,8,s0y1y3y4y5y6).
qpair(2,8,s6y7y9).
qpair(2,8,s0y5y7).
qpair(2,8,s0y1y4y6y9).
qpair(2,8,s5).
qpair(2,8,s3y7).
qpair(2,8,s4y6).
qpair(2,8,s0y1y3y4y5).
qpair(2,8,s0y4y5y6).
qpair(2,8,s1y3y4y5y6).
qpair(2,8,s0y3y9).
qpair(2,8,s4y7y9).
qpair(2,8,s0y3y4y5).
qpair(2,8,s1y4y5y6y9).
qpair(2,8,s1y9).
qpair(2,8,s3y6y7).
qpair(2,8,s0y3y4y6y9).
qpair(2,8,s0y6y7).
qpair(2,8,s0y5y9).
qpair(2,8,s7).
qpair(2,8,s3y9).
qpair(2,8,s5y6).
qpair(2,8,s0y4).
qpair(2,8,s3y4y6).
qpair(2,8,s0y3y5y6).
qpair(2,8,s3y6y9).
qpair(2,8,s0).
qpair(2,8,s0y6y9).
qpair(2,8,s0y1y4y6).
qpair(2,8,s0y5y6y9).
qpair(2,8,s5y6y9).
qpair(2,8,s0y1y3y4y9).
qpair(2,8,s0y1y7).
qpair(2,8,s0y3y4).
qpair(2,8,s4y5y7).
qpair(2,8,s1y4y5y9).
qpair(2,8,s1y4y6).
qpair(2,8,s1y4).
qpair(2,8,s0y6).
qpair(2,8,s0y3y4y9).
qpair(2,8,s0y1y3y6y9).
qpair(2,8,s0y3y6y9).
qpair(2,8,s3y4).
qpair(2,8,s3y7y9).
qpair(2,8,s0y1y9).
qpair(2,8,s0y3y6).
qpair(2,8,s1y6).
qpair(2,8,s1y3y5y6y9).
qpair(2,8,s3y4y5y9).
qpair(2,8,s0y4y7).
qpair(2,8,s0y5y6).
qpair(2,8,s3y4y5y6y9).
qpair(2,8,s0y1y5y9).
qpair(2,8,s0y1).
qpair(2,8,s1y3y4y5).
qpair(2,8,s0y1y4y5y9).
qpair(2,8,s0y1y3y4y6y9).
qpair(2,8,s1y3y5).
qpair(2,8,s6y7).
qpair(2,8,s0y4y6y9).
qpair(2,8,s4y7).
qpair(2,8,s3y5y7).
qpair(2,8,s0y1y4).
qpair(2,8,s1y4y5y6).
qpair(2,8,s0y3y4y6).
qpair(2,8,s0y3).
qpair(2,8,s1y3y7).
qpair(2,8,s0y4y5y6y9).
qpair(2,9,s6).
qpair(2,9,s3y8).
qpair(2,9,s1y3y8).
qpair(2,9,s5).
qpair(2,9,s1y8).
qpair(2,9,s1y3).
qpair(3,4,s6).
qpair(3,4,s2).
qpair(3,4,s0y9).
qpair(3,4,s0y5y9).
qpair(3,4,s7).
qpair(3,4,s0y5y8).
qpair(3,4,s0y5).
qpair(3,5,s6).
qpair(3,5,s2).
qpair(3,5,s1y9).
qpair(3,5,s8).
qpair(3,5,s0y9).
qpair(3,5,s1y8).
qpair(3,5,s6y8).
qpair(3,5,s1).
qpair(3,5,s7).
qpair(3,5,s6y7).
qpair(3,5,s8y9).
qpair(3,5,s0).
qpair(3,5,s0y8).
qpair(3,5,empty).
qpair(3,5,s9).
qpair(3,5,s7y8).
qpair(3,6,s5y8y9).
qpair(3,6,s2).
qpair(3,6,s0y9).
qpair(3,6,s4).
qpair(3,6,s0y5y9).
qpair(3,6,s1y7).
qpair(3,6,s8y9).
qpair(3,6,s0y5).
qpair(3,6,s0y8y9).
qpair(3,6,s5y9).
qpair(3,6,s0y8).
qpair(3,7,s1).
qpair(3,7,s2).
qpair(3,7,s4).
qpair(3,8,s0y1y4y5).
qpair(3,8,s4y9).
qpair(3,8,s1y5y7).
qpair(3,8,s0y1y6).
qpair(3,8,s0y5).
qpair(3,8,s1y2y4y6).
qpair(3,8,s0y1y2y4y5).
qpair(3,8,s4y6y7).
qpair(3,8,s2y4y5).
qpair(3,8,s0y2y4).
qpair(3,8,s4y5y6y9).
qpair(3,8,s0y1y5y6).
qpair(3,8,s1y4y6y9).
qpair(3,8,s0y1y4y5y6).
qpair(3,8,s0y7).
qpair(3,8,s2y4).
qpair(3,8,s2y6y7).
qpair(3,8,s2y4y5y9).
qpair(3,8,s2y4y7).
qpair(3,8,s1y2y5y9).
qpair(3,8,s6).
qpair(3,8,s0y1y2y4y5y9).
qpair(3,8,s0y9).
qpair(3,8,s0y1y2y4y9).
qpair(3,8,s0y1y2y4y5y6y9).
qpair(3,8,s1y2y5y6y9).
qpair(3,8,s0y2y4y5y9).
qpair(3,8,s1y2y4y5y6y9).
qpair(3,8,s0y2y5y9).
qpair(3,8,s2y4y9).
qpair(3,8,s0y1y4y5y6y9).
qpair(3,8,s0y1y5y6y9).
qpair(3,8,s5y7).
qpair(3,8,s1y5y6y9).
qpair(3,8,s0y2).
qpair(3,8,s0y4y5y9).
qpair(3,8,s1y4y5).
qpair(3,8,s5y7y9).
qpair(3,8,s1y2y4y6y9).
qpair(3,8,s0y2y5y6y9).
qpair(3,8,s0y1y2y5y9).
qpair(3,8,s1y5y6).
qpair(3,8,s5y6y7).
qpair(3,8,s0y1y5).
qpair(3,8,s5y9).
qpair(3,8,s1y2y4y5).
qpair(3,8,s2y5y6).
qpair(3,8,s1y4y7).
qpair(3,8,s1y5).
qpair(3,8,s2y4y5y6).
qpair(3,8,s7y9).
qpair(3,8,s0y4y6).
qpair(3,8,s0y1y2y4y6).
qpair(3,8,s2y4y5y6y9).
qpair(3,8,s0y2y4y5y6).
qpair(3,8,s1y4y9).
qpair(3,8,s0y2y5y6).
qpair(3,8,s2y6).
qpair(3,8,s0y2y5).
qpair(3,8,s0y1y6y9).
qpair(3,8,s0y5y7).
qpair(3,8,s0y1y4y6y9).
qpair(3,8,s5).
qpair(3,8,s1y6y9).
qpair(3,8,s4y6).
qpair(3,8,s0y4y5y6).
qpair(3,8,s1y2y4y9).
qpair(3,8,s1y2y5).
qpair(3,8,s1y4y5y6y9).
qpair(3,8,s1y9).
qpair(3,8,s0y2y6y9).
qpair(3,8,s1y2y6y9).
qpair(3,8,s0y6y7).
qpair(3,8,s0y5y9).
qpair(3,8,s7).
qpair(3,8,s2y7y9).
qpair(3,8,s1y2y7).
qpair(3,8,s1y2).
qpair(3,8,s0y4).
qpair(3,8,s0).
qpair(3,8,s0y6y9).
qpair(3,8,s1y2y5y6).
qpair(3,8,s0y1y2y9).
qpair(3,8,s0y1y4y6).
qpair(3,8,s9).
qpair(3,8,s0y5y6y9).
qpair(3,8,s1y2y4y5y9).
qpair(3,8,s0y1y7).
qpair(3,8,s4y5y7).
qpair(3,8,s0y1y2y4y5y6).
qpair(3,8,s1y2y9).
qpair(3,8,s1y4y6).
qpair(3,8,s1y4).
qpair(3,8,s0y6).
qpair(3,8,s2y4y6).
qpair(3,8,s0y4y5).
qpair(3,8,s2).
qpair(3,8,s0y2y4y5y6y9).
qpair(3,8,s2y4y6y9).
qpair(3,8,s0y1y9).
qpair(3,8,s4y5y9).
qpair(3,8,s1y6).
qpair(3,8,s0y4y7).
qpair(3,8,s0y5y6).
qpair(3,8,s0y1y2y4).
qpair(3,8,s4).
qpair(3,8,s0y1y5y9).
qpair(3,8,s0y2y4y6y9).
qpair(3,8,s0y1y2).
qpair(3,8,s1y7y9).
qpair(3,8,s2y5y9).
qpair(3,8,s1y2y4).
qpair(3,8,s0y1y2y4y6y9).
qpair(3,8,s0y1).
qpair(3,8,s2y7).
qpair(3,8,s6y7).
qpair(3,8,s0y2y9).
qpair(3,8,s0y4y6y9).
qpair(3,8,s0y1y2y6).
qpair(3,8,s4y7).
qpair(3,8,s1y2y4y5y6).
qpair(3,8,s0y1y4).
qpair(3,8,s1y2y6).
qpair(3,8,s0y1y2y5y6y9).
qpair(3,8,s2y9).
qpair(3,8,s0y4y5y6y9).
qpair(3,8,s6y9).
qpair(3,9,s6).
qpair(3,9,s2).
qpair(3,9,s5).
qpair(3,9,s1y5).
qpair(3,9,s8).
qpair(3,9,s1y4).
qpair(3,9,s1).
qpair(3,9,s7).
qpair(3,9,s0y5y8).
qpair(3,9,s0y5).
qpair(3,9,s4y8).
qpair(3,9,s0).
qpair(3,9,s0y8).
qpair(3,9,empty).
qpair(4,5,s1).
qpair(4,5,s3).
qpair(4,5,s3y7).
qpair(4,6,s0y1).
qpair(4,6,s0y7).
qpair(4,6,s2).
qpair(4,6,s5).
qpair(4,6,s1y5).
qpair(4,6,s5y7).
qpair(4,6,s8y9).
qpair(4,6,s0y5).
qpair(4,6,s0).
qpair(4,6,s3).
qpair(4,6,s9).
qpair(4,8,s1y5y7).
qpair(4,8,s3y5y9).
qpair(4,8,s0y1y6).
qpair(4,8,s0y5).
qpair(4,8,s1y3y9).
qpair(4,8,s0y1y3y9).
qpair(4,8,s0y1y5y6).
qpair(4,8,s0y3y5).
qpair(4,8,s1y2y3y9).
qpair(4,8,s0y7).
qpair(4,8,s2y6y7).
qpair(4,8,s0y3y5y9).
qpair(4,8,s0y2y6).
qpair(4,8,s0y1y2y3).
qpair(4,8,s1y2y5y9).
qpair(4,8,s1y2y3y5y6).
qpair(4,8,s1y3y5y6).
qpair(4,8,s6).
qpair(4,8,s0y2y3y9).
qpair(4,8,s0y3y7).
qpair(4,8,s0y9).
qpair(4,8,s2y6y9).
qpair(4,8,s1y2y5y6y9).
qpair(4,8,s0y2y5y9).
qpair(4,8,empty).
qpair(4,8,s0y1y2y5).
qpair(4,8,s0y1y3).
qpair(4,8,s0y3y5y6y9).
qpair(4,8,s0y1y5y6y9).
qpair(4,8,s5y7).
qpair(4,8,s1y5y6y9).
qpair(4,8,s0y1y2y3y5y6y9).
qpair(4,8,s0y2).
qpair(4,8,s1y3y6).
qpair(4,8,s0y1y2y3y5y9).
qpair(4,8,s1y3).
qpair(4,8,s0y2y3y5y6y9).
qpair(4,8,s5y7y9).
qpair(4,8,s2y3y6).
qpair(4,8,s0y2y3y5y9).
qpair(4,8,s0y1y3y6).
qpair(4,8,s0y2y5y6y9).
qpair(4,8,s0y1y2y5y9).
qpair(4,8,s1y5y6).
qpair(4,8,s1).
qpair(4,8,s5y6y7).
qpair(4,8,s0y7y9).
qpair(4,8,s1y2y3y6y9).
qpair(4,8,s0y1y5).
qpair(4,8,s1y3y6y9).
qpair(4,8,s1y2y3y6).
qpair(4,8,s2y5y6).
qpair(4,8,s1y5).
qpair(4,8,s3y5y6y9).
qpair(4,8,s0y2y3).
qpair(4,8,s2y3y5y9).
qpair(4,8,s0y2y3y6y9).
qpair(4,8,s0y1y2y3y5).
qpair(4,8,s3).
qpair(4,8,s1y6y7).
qpair(4,8,s3y5).
qpair(4,8,s0y2y3y6).
qpair(4,8,s0y1y2y3y6y9).
qpair(4,8,s1y2y3).
qpair(4,8,s1y7).
qpair(4,8,s0y2y5y6).
qpair(4,8,s2y6).
qpair(4,8,s1y2y3y5y6y9).
qpair(4,8,s0y5y7).
qpair(4,8,s5).
qpair(4,8,s1y6y9).
qpair(4,8,s3y7).
qpair(4,8,s1y2y5).
qpair(4,8,s1y9).
qpair(4,8,s0y2y7).
qpair(4,8,s3y6y7).
qpair(4,8,s0y2y3y5y6).
qpair(4,8,s0y2y6y9).
qpair(4,8,s1y2y6y9).
qpair(4,8,s0y1y2y5y6).
qpair(4,8,s0y6y7).
qpair(4,8,s0y5y9).
qpair(4,8,s7).
qpair(4,8,s3y9).
qpair(4,8,s5y6).
qpair(4,8,s0y1y3y5y6).
qpair(4,8,s2y7y9).
qpair(4,8,s1y2y7).
qpair(4,8,s0y3y5y6).
qpair(4,8,s0y1y3y5y6y9).
qpair(4,8,s2y3y5).
qpair(4,8,s3y6y9).
qpair(4,8,s0y1y3y5).
qpair(4,8,s1y2y3y5y9).
qpair(4,8,s0y6y9).
qpair(4,8,s1y2y5y6).
qpair(4,8,s2y5y6y9).
qpair(4,8,s0y1y2y9).
qpair(4,8,s0y5y6y9).
qpair(4,8,s9).
qpair(4,8,s0y1y7).
qpair(4,8,s1y2y9).
qpair(4,8,s0y6).
qpair(4,8,s2y3).
qpair(4,8,s2y3y7).
qpair(4,8,s0y1y3y6y9).
qpair(4,8,s2y3y6y9).
qpair(4,8,s2).
qpair(4,8,s0y2y3y5).
qpair(4,8,s0y3y6y9).
qpair(4,8,s2y5y7).
qpair(4,8,s3y7y9).
qpair(4,8,s0y1y9).
qpair(4,8,s0y3y6).
qpair(4,8,s2y5).
qpair(4,8,s1y3y5y6y9).
qpair(4,8,s2y3y9).
qpair(4,8,s0y5y6).
qpair(4,8,s0y1y5y9).
qpair(4,8,s0y1y2).
qpair(4,8,s2y5y9).
qpair(4,8,s0y1).
qpair(4,8,s2y7).
qpair(4,8,s1y3y5).
qpair(4,8,s6y7).
qpair(4,8,s2y3y5y6).
qpair(4,8,s0y2y9).
qpair(4,8,s3y5y7).
qpair(4,8,s1y2y6).
qpair(4,8,s1y2y3y5).
qpair(4,8,s0y3).
qpair(4,8,s0y1y2y5y6y9).
qpair(4,8,s2y9).
qpair(4,8,s0y1y2y6y9).
qpair(4,8,s6y9).
qpair(4,9,s6).
qpair(4,9,s5).
qpair(4,9,s6y8).
qpair(4,9,s0).
qpair(4,9,s1y6).
qpair(5,6,s0y1).
qpair(5,6,s0y7).
qpair(5,6,s2).
qpair(5,6,s4).
qpair(5,6,s4y8).
qpair(5,6,s0y8).
qpair(5,6,s4y7).
qpair(5,7,s6).
qpair(5,7,s3y8).
qpair(5,7,s2).
qpair(5,7,s0y6y8).
qpair(5,7,s6y8).
qpair(5,7,s8).
qpair(5,7,s0y6).
qpair(5,7,s1).
qpair(5,7,s3y6).
qpair(5,7,s0).
qpair(5,7,s0y8).
qpair(5,7,s3).
qpair(5,7,s9).
qpair(5,8,s4y9).
qpair(5,8,s0y1y6).
qpair(5,8,s1y2y3y4y9).
qpair(5,8,s1y2y4y6).
qpair(5,8,s4y6y7).
qpair(5,8,s1y3y9).
qpair(5,8,s0y2y4y6).
qpair(5,8,s0y1y3y9).
qpair(5,8,s3y6).
qpair(5,8,s1y4y6y9).
qpair(5,8,s1y2y3y9).
qpair(5,8,s2y6y7).
qpair(5,8,s2y4).
qpair(5,8,s4y6y9).
qpair(5,8,s2y4y7).
qpair(5,8,s0y2y6).
qpair(5,8,s0y1y2y3).
qpair(5,8,s0y1y4y9).
qpair(5,8,s0y4y9).
qpair(5,8,s6).
qpair(5,8,s0y2y3y9).
qpair(5,8,s0y3y7).
qpair(5,8,s0y1y3y4y6).
qpair(5,8,s2y6y9).
qpair(5,8,s0y9).
qpair(5,8,s1y3y4).
qpair(5,8,s0y1y2y4y9).
qpair(5,8,s2y4y9).
qpair(5,8,s3y4y6y9).
qpair(5,8,empty).
qpair(5,8,s0y1y3y4).
qpair(5,8,s0y1y3).
qpair(5,8,s2y3y4y6).
qpair(5,8,s0y2).
qpair(5,8,s0y2y3y4y6y9).
qpair(5,8,s1y3y6).
qpair(5,8,s3y4y7).
qpair(5,8,s1y3y4y9).
qpair(5,8,s1y2y4y6y9).
qpair(5,8,s2y3y6).
qpair(5,8,s0y1y3y6).
qpair(5,8,s1y2y3y6y9).
qpair(5,8,s1).
qpair(5,8,s0y2y3y4).
qpair(5,8,s0y7y9).
qpair(5,8,s1y2y3y4y6y9).
qpair(5,8,s1y3y6y9).
qpair(5,8,s1y2y3y6).
qpair(5,8,s0y1y2y3y4y6y9).
qpair(5,8,s1y4y7).
qpair(5,8,s3y4y9).
qpair(5,8,s0y2y3).
qpair(5,8,s7y9).
qpair(5,8,s0y2y3y6y9).
qpair(5,8,s0y4y6).
qpair(5,8,s3).
qpair(5,8,s1y6y7).
qpair(5,8,s0y2y3y6).
qpair(5,8,s0y1y2y3y6y9).
qpair(5,8,s0y1y2y4y6).
qpair(5,8,s1y4y9).
qpair(5,8,s1y7).
qpair(5,8,s2y6).
qpair(5,8,s6y7y9).
qpair(5,8,s2y3y4y6y9).
qpair(5,8,s0y1y4y6y9).
qpair(5,8,s1y6y9).
qpair(5,8,s3y7).
qpair(5,8,s4y6).
qpair(5,8,s1y2y4y9).
qpair(5,8,s1y2y3y4y6).
qpair(5,8,s4y7y9).
qpair(5,8,s1y9).
qpair(5,8,s3y6y7).
qpair(5,8,s0y3y4y6y9).
qpair(5,8,s0y2y6y9).
qpair(5,8,s1y2y6y9).
qpair(5,8,s0y1y2y3y9).
qpair(5,8,s0y6y7).
qpair(5,8,s7).
qpair(5,8,s3y9).
qpair(5,8,s2y7y9).
qpair(5,8,s1y2).
qpair(5,8,s0y4).
qpair(5,8,s3y4y6).
qpair(5,8,s3y6y9).
qpair(5,8,s1y3y4y6y9).
qpair(5,8,s0y1y2y3y4y9).
qpair(5,8,s0).
qpair(5,8,s0y1y2y9).
qpair(5,8,s0y1y4y6).
qpair(5,8,s9).
qpair(5,8,s0y1y7).
qpair(5,8,s1y4y6).
qpair(5,8,s1y4).
qpair(5,8,s0y6).
qpair(5,8,s2y3).
qpair(5,8,s0y3y4y9).
qpair(5,8,s2y3y7).
qpair(5,8,s0y1y3y6y9).
qpair(5,8,s2y4y6).
qpair(5,8,s0y1y2y3y4).
qpair(5,8,s2).
qpair(5,8,s0y3y6y9).
qpair(5,8,s3y4).
qpair(5,8,s2y3y4y9).
qpair(5,8,s3y7y9).
qpair(5,8,s0y3y6).
qpair(5,8,s1y6).
qpair(5,8,s0y2y4y9).
qpair(5,8,s2y3y9).
qpair(5,8,s0y4y7).
qpair(5,8,s0y1y2y4).
qpair(5,8,s4).
qpair(5,8,s0y2y4y6y9).
qpair(5,8,s1y2y4).
qpair(5,8,s0y1y2y4y6y9).
qpair(5,8,s0y1).
qpair(5,8,s0y1y3y4y6y9).
qpair(5,8,s2y7).
qpair(5,8,s6y7).
qpair(5,8,s0y1y2y3y4y6).
qpair(5,8,s0y2y9).
qpair(5,8,s0y4y6y9).
qpair(5,8,s0y2y3y4y9).
qpair(5,8,s0y1y2y6).
qpair(5,8,s4y7).
qpair(5,8,s0y1y4).
qpair(5,8,s1y2y6).
qpair(5,8,s0y3y4y6).
qpair(5,8,s0y3).
qpair(5,8,s2y9).
qpair(5,8,s0y1y2y6y9).
qpair(5,8,s2y3y4).
qpair(5,8,s6y9).
qpair(5,9,s0).
qpair(5,9,s2).
qpair(5,9,s6).
qpair(5,9,s4).
qpair(6,7,s0y1).
qpair(6,7,s0y4).
qpair(6,7,s5).
qpair(6,7,s1y5).
qpair(6,7,s8).
qpair(6,7,s1y8).
qpair(6,7,s4).
qpair(6,7,s1y4).
qpair(6,7,s1).
qpair(6,7,s4y8).
qpair(6,7,empty).
qpair(6,7,s3).
qpair(6,7,s9).
qpair(6,8,s0y1y4y5).
qpair(6,8,s4y9).
qpair(6,8,s1y3y4y5y9).
qpair(6,8,s1y2y3y4y9).
qpair(6,8,s0y1y2y4y5).
qpair(6,8,s1y3y9).
qpair(6,8,s0y2y4).
qpair(6,8,s0y1y3y5y9).
qpair(6,8,s0y3y5).
qpair(6,8,s1y2y3y9).
qpair(6,8,s0y7).
qpair(6,8,s2y4).
qpair(6,8,s2y4y7).
qpair(6,8,s0y3y4y5y9).
qpair(6,8,s0y1y2y3).
qpair(6,8,s1y2y5y9).
qpair(6,8,s0y1y4y9).
qpair(6,8,s0y4y9).
qpair(6,8,s0y2y3y9).
qpair(6,8,s0y1y2y4y5y9).
qpair(6,8,s0y3y7).
qpair(6,8,s0y9).
qpair(6,8,s0y1y2y4y9).
qpair(6,8,s1y3y4).
qpair(6,8,s3y4y5).
qpair(6,8,s0y2y4y5y9).
qpair(6,8,s0y2y5y9).
qpair(6,8,s2y4y9).
qpair(6,8,empty).
qpair(6,8,s0y1y2y5).
qpair(6,8,s0y1y3y4).
qpair(6,8,s0y1y3y4y5y9).
qpair(6,8,s0y1y3).
qpair(6,8,s5y7).
qpair(6,8,s1y2y3y4).
qpair(6,8,s0y2).
qpair(6,8,s0y4y5y9).
qpair(6,8,s0y1y2y3y5y9).
qpair(6,8,s1y4y5).
qpair(6,8,s1y3).
qpair(6,8,s3y4y7).
qpair(6,8,s1y3y4y9).
qpair(6,8,s5y7y9).
qpair(6,8,s0y2y3y5y9).
qpair(6,8,s0y1y2y5y9).
qpair(6,8,s1).
qpair(6,8,s0y2y3y4).
qpair(6,8,s0y7y9).
qpair(6,8,s0y1y5).
qpair(6,8,s5y9).
qpair(6,8,s1y2y4y5).
qpair(6,8,s1y4y7).
qpair(6,8,s0y2y4y5).
qpair(6,8,s3y4y9).
qpair(6,8,s0y2y3).
qpair(6,8,s7y9).
qpair(6,8,s2y3y5y9).
qpair(6,8,s0y1y2y3y5).
qpair(6,8,s3).
qpair(6,8,s3y5).
qpair(6,8,s1y2y3).
qpair(6,8,s1y4y9).
qpair(6,8,s1y7).
qpair(6,8,s0y2y5).
qpair(6,8,s0y1y2y3y4y5).
qpair(6,8,s5).
qpair(6,8,s3y7).
qpair(6,8,s0y1y3y4y5).
qpair(6,8,s1y2y4y9).
qpair(6,8,s1y2y5).
qpair(6,8,s0y3y9).
qpair(6,8,s4y7y9).
qpair(6,8,s1y9).
qpair(6,8,s0y2y7).
qpair(6,8,s0y1y2y3y9).
qpair(6,8,s0y5y9).
qpair(6,8,s7).
qpair(6,8,s3y9).
qpair(6,8,s2y3y4y5).
qpair(6,8,s2y7y9).
qpair(6,8,s1y2y7).
qpair(6,8,s0y4).
qpair(6,8,s2y3y5).
qpair(6,8,s0y1y3y5).
qpair(6,8,s1y2y3y5y9).
qpair(6,8,s0y1y2y3y4y9).
qpair(6,8,s0).
qpair(6,8,s0y1y2y9).
qpair(6,8,s9).
qpair(6,8,s0y1y3y4y9).
qpair(6,8,s1y2y4y5y9).
qpair(6,8,s0y1y7).
qpair(6,8,s0y3y4).
qpair(6,8,s4y5y7).
qpair(6,8,s1y4y5y9).
qpair(6,8,s1y2y9).
qpair(6,8,s0y3y4y9).
qpair(6,8,s1y4).
qpair(6,8,s1y2y3y4y5y9).
qpair(6,8,s2y3).
qpair(6,8,s2y3y7).
qpair(6,8,s0y1y2y3y4).
qpair(6,8,s0y4y5).
qpair(6,8,s2).
qpair(6,8,s0y2y3y5).
qpair(6,8,s3y4).
qpair(6,8,s2y5y7).
qpair(6,8,s3y7y9).
qpair(6,8,s0y1y9).
qpair(6,8,s4y5y9).
qpair(6,8,s2y5).
qpair(6,8,s0y2y4y9).
qpair(6,8,s3y4y5y9).
qpair(6,8,s0y1y2y4).
qpair(6,8,s4).
qpair(6,8,s4y5).
qpair(6,8,s0y1y5y9).
qpair(6,8,s1y7y9).
qpair(6,8,s1y2y4).
qpair(6,8,s2y3y4y5y9).
qpair(6,8,s1y3y4y5).
qpair(6,8,s0y1y4y5y9).
qpair(6,8,s2y7).
qpair(6,8,s1y3y5).
qpair(6,8,s0y2y9).
qpair(6,8,s0y2y3y4y9).
qpair(6,8,s4y7).
qpair(6,8,s0y1y4).
qpair(6,8,s1y3y5y9).
qpair(6,8,s0y2y3y4y5y9).
qpair(6,8,s1y2y3y5).
qpair(6,8,s0y3).
qpair(6,8,s0y1y2y3y4y5y9).
qpair(6,8,s1y3y7).
qpair(6,8,s2y3y4).
qpair(6,9,s0).
qpair(6,9,s2).
qpair(6,9,s4).
qpair(6,9,s5).
qpair(8,9,s1y5y7).
qpair(8,9,s0y1y6).
qpair(8,9,s0y5).
qpair(8,9,s1y2y4y6).
qpair(8,9,s0y1y2y4y5).
qpair(8,9,s4y6y7).
qpair(8,9,s0y2y4y6).
qpair(8,9,s3y4y5y6).
qpair(8,9,s0y2y4).
qpair(8,9,s0y1y5y6).
qpair(8,9,s0y3y5).
qpair(8,9,s2y3y4y5y6).
qpair(8,9,s0y1y4y5y6).
qpair(8,9,s0y7).
qpair(8,9,s2y4).
qpair(8,9,s2y6y7).
qpair(8,9,s2y4y7).
qpair(8,9,s0y1y2y3).
qpair(8,9,s1y2y3y5y6).
qpair(8,9,s1y3y5y6).
qpair(8,9,s6).
qpair(8,9,s0y3y7).
qpair(8,9,s0y1y3y4y6).
qpair(8,9,s0y1y2y3y4y5y6).
qpair(8,9,s1y3y4).
qpair(8,9,s1y2y3y4y5y6).
qpair(8,9,s3y4y5).
qpair(8,9,empty).
qpair(8,9,s0y1y2y5).
qpair(8,9,s0y1y3y4).
qpair(8,9,s3y5y6).
qpair(8,9,s0y1y3).
qpair(8,9,s1y2y3y4).
qpair(8,9,s0y2).
qpair(8,9,s1y3y6).
qpair(8,9,s1y4y5).
qpair(8,9,s1y3).
qpair(8,9,s3y4y7).
qpair(8,9,s0y1y3y6).
qpair(8,9,s1y5y6).
qpair(8,9,s0y2y3y4).
qpair(8,9,s0y1y5).
qpair(8,9,s1y2y3y6).
qpair(8,9,s1y2y4y5).
qpair(8,9,s2y5y6).
qpair(8,9,s1y4y7).
qpair(8,9,s1y5).
qpair(8,9,s0y2y4y5).
qpair(8,9,s2y4y5y6).
qpair(8,9,s0y2y3).
qpair(8,9,s0y3y4y5y6).
qpair(8,9,s0y1y2y3y5).
qpair(8,9,s0y4y6).
qpair(8,9,s3).
qpair(8,9,s3y5).
qpair(8,9,s0y2y3y4y5y6).
qpair(8,9,s0y1y2y4y6).
qpair(8,9,s1y2y3).
qpair(8,9,s0y2y4y5y6).
qpair(8,9,s1y7).
qpair(8,9,s0y2y5y6).
qpair(8,9,s2y6).
qpair(8,9,s0y2y5).
qpair(8,9,s0y2y3y4y5).
qpair(8,9,s0y1y3y4y5y6).
qpair(8,9,s0y1y2y3y4y5).
qpair(8,9,s0y5y7).
qpair(8,9,s5).
qpair(8,9,s3y7).
qpair(8,9,s0y4y5y6).
qpair(8,9,s1y3y4y5y6).
qpair(8,9,s1y2y5).
qpair(8,9,s0y3y4y5).
qpair(8,9,s0y1y2y3y5y6).
qpair(8,9,s1y3y4y6).
qpair(8,9,s0y2y7).
qpair(8,9,s3y6y7).
qpair(8,9,s0y2y3y5y6).
qpair(8,9,s0y1y2y5y6).
qpair(8,9,s0y6y7).
qpair(8,9,s7).
qpair(8,9,s5y6).
qpair(8,9,s2y3y4y5).
qpair(8,9,s0y1y3y5y6).
qpair(8,9,s1y2y7).
qpair(8,9,s0y4).
qpair(8,9,s3y4y6).
qpair(8,9,s0y3y5y6).
qpair(8,9,s2y3y5).
qpair(8,9,s0y1y3y5).
qpair(8,9,s0).
qpair(8,9,s1y2y5y6).
qpair(8,9,s0y1y4y6).
qpair(8,9,s0y1y7).
qpair(8,9,s0y3y4).
qpair(8,9,s4y5y7).
qpair(8,9,s0y1y2y4y5y6).
qpair(8,9,s1y4y6).
qpair(8,9,s1y4).
qpair(8,9,s0y6).
qpair(8,9,s2y3).
qpair(8,9,s2y4y6).
qpair(8,9,s0y1y2y3y4).
qpair(8,9,s2).
qpair(8,9,s0y2y3y5).
qpair(8,9,s3y4).
qpair(8,9,s2y5y7).
qpair(8,9,s0y3y6).
qpair(8,9,s1y6).
qpair(8,9,s2y5).
qpair(8,9,s0y1y2y3y6).
qpair(8,9,s0y4y7).
qpair(8,9,s0y1y2y4).
qpair(8,9,s4).
qpair(8,9,s0y1y2).
qpair(8,9,s1y2y3y4y5).
qpair(8,9,s1y2y4).
qpair(8,9,s0y1).
qpair(8,9,s1y3y4y5).
qpair(8,9,s1y3y5).
qpair(8,9,s6y7).
qpair(8,9,s2y3y5y6).
qpair(8,9,s0y1y2y3y4y6).
qpair(8,9,s0y1y2y6).
qpair(8,9,s4y7).
qpair(8,9,s1y2y4y5y6).
qpair(8,9,s3y5y7).
qpair(8,9,s0y1y4).
qpair(8,9,s1y4y5y6).
qpair(8,9,s0y3y4y6).
qpair(8,9,s1y2y3y5).
qpair(8,9,s0y3).
qpair(8,9,s1y3y7).
qpair(8,9,s2y3y4).
qpair(0,1,s2y3y7y8).
qpair(0,1,s3y7).
qpair(0,1,s8).
qpair(0,1,s4y6).
qpair(0,1,s8y9).
qpair(0,1,s2y8).
qpair(0,1,s6y8).
qpair(0,1,s4).
qpair(0,1,s6y8y9).
qpair(0,1,s4y8).
qpair(0,1,s3y7y8).
qpair(0,1,s2y7).
qpair(0,1,s2y3y8).
qpair(0,1,s3).
qpair(0,1,s9).
qpair(0,1,s6).
qpair(0,1,s3y8).
qpair(0,1,s2y7y8).
qpair(0,1,s6y9).
qpair(0,1,s7y8).
qpair(0,2,s6).
qpair(0,2,s7).
qpair(0,3,s1y8).
qpair(0,3,s9).
qpair(0,3,s5).
qpair(0,4,s6).
qpair(0,4,s3y8).
qpair(0,4,s2y3y7y8).
qpair(0,4,s2y7).
qpair(0,4,s5).
qpair(0,4,s3y7).
qpair(0,4,s1y8).
qpair(0,4,s2y7y8).
qpair(0,4,s2y3).
qpair(0,4,s2y3y8).
qpair(0,4,s3y7y8).
qpair(0,4,s2y3y7).
qpair(0,4,s9).
qpair(0,4,s7y8).
qpair(0,4,s2y8).
qpair(0,5,s4).
qpair(0,7,s5y6y8).
qpair(0,7,s3y8).
qpair(0,7,s3y6).
qpair(0,7,s5).
qpair(0,8,s2).
qpair(0,8,s2y3y4y6).
qpair(0,8,s1y3y4y5y6).
qpair(0,8,s4y5y9).
qpair(0,8,s1y2y5).
qpair(0,8,s4y6y7).
qpair(0,8,s1y3y4y9).
qpair(0,8,s1y3y5y6y9).
qpair(0,8,s1y2y3y6y9).
qpair(0,8,s1y2y3y4y6y9).
qpair(0,8,s1y2y4y5).
qpair(0,8,s1y2y4).
qpair(0,8,s1y3y4y5).
qpair(0,8,s1y4y7).
qpair(0,8,s3y5y6y9).
qpair(0,8,s1y2y5y9).
qpair(0,8,s2y5y6y9).
qpair(0,8,s3).
qpair(0,8,s1y2y4y5y6).
qpair(0,8,s1y2y6).
qpair(0,8,s2y6).
qpair(0,8,s3y4y6y9).
qpair(0,9,s4).
qpair(1,2,s5).
qpair(1,2,s6y8).
qpair(1,2,s4y6y8).
qpair(1,2,s6y8y9).
qpair(1,2,s4y8).
qpair(1,2,s0y8).
qpair(1,3,s7).
qpair(1,3,s2).
qpair(1,4,s0).
qpair(1,4,s6y8y9).
qpair(1,4,s2).
qpair(1,5,s4y9).
qpair(1,5,s3y7).
qpair(1,5,s8).
qpair(1,5,s4y6).
qpair(1,5,s8y9).
qpair(1,5,s6y8).
qpair(1,5,s7).
qpair(1,5,s6y8y9).
qpair(1,5,s3y7y8).
qpair(1,5,s4y6y9).
qpair(1,5,s4y6y8y9).
qpair(1,5,s4y8y9).
qpair(1,5,s3).
qpair(1,5,s9).
qpair(1,5,s6).
qpair(1,5,s3y8).
qpair(1,5,s4y6y8).
qpair(1,5,s6y9).
qpair(1,5,s7y8).
qpair(1,6,s0y7).
qpair(1,6,empty).
qpair(1,7,s2).
qpair(1,7,s4y6y9).
qpair(1,7,s4y9).
qpair(1,7,s4y6y8y9).
qpair(1,7,s6y8).
qpair(1,7,s4y6).
qpair(1,7,s0y5y6).
qpair(1,7,s4y6y8).
qpair(1,7,s0y6).
qpair(1,7,s6y8y9).
qpair(1,7,s8y9).
qpair(1,7,s5y6).
qpair(1,7,s4y8).
qpair(1,7,s4y8y9).
qpair(1,8,s2y3y4y6y9).
qpair(1,8,s3y5y6).
qpair(1,8,s4y6).
qpair(1,8,s3y5y9).
qpair(1,8,s0y3y4y5).
qpair(1,8,s2y4y5).
qpair(1,8,s0y2y3y5y6y9).
qpair(1,8,s4y5y6y9).
qpair(1,8,s0y2y6y9).
qpair(1,8,s0y5y6).
qpair(1,8,s4).
qpair(1,8,s0y7y9).
qpair(1,8,s0y3y5).
qpair(1,8,s2y5y6).
qpair(1,8,s2y4).
qpair(1,8,s3y4y6).
qpair(1,8,s2y3y5y6y9).
qpair(1,8,s7y9).
qpair(1,8,s0y2y3y4y6).
qpair(1,8,s2y3y5y6).
qpair(1,8,s0y2y9).
qpair(1,8,s3).
qpair(1,8,s9).
qpair(1,8,s3y5).
qpair(1,8,s5y6y9).
qpair(1,8,s0y2y3y4y5y6).
qpair(1,8,s0y3y4y6).
qpair(1,8,s0y3).
qpair(1,8,s0y3y4y9).
qpair(1,8,s0y2y5y6).
qpair(1,8,s0y2y5).
qpair(1,8,s0y4y5y6y9).
qpair(1,9,s0).
qpair(1,9,s3).
qpair(1,9,s3y4).
qpair(2,4,s0y8).
qpair(2,4,s1y8).
qpair(2,4,s3).
qpair(2,4,s9).
qpair(2,5,s3y8).
qpair(2,5,s4).
qpair(2,5,s0y8).
qpair(2,5,s9).
qpair(2,5,s7y8).
qpair(2,6,s7).
qpair(2,7,s8).
qpair(2,7,s0y5).
qpair(2,7,s0y6).
qpair(2,8,s1y6y9).
qpair(2,8,s3y5y9).
qpair(2,8,s4y5y9).
qpair(2,8,s1y3y4y6).
qpair(2,8,s5y7y9).
qpair(2,8,s0y1y3y4y5y6y9).
qpair(2,8,s4).
qpair(2,8,s4y5).
qpair(2,8,s0y1y3y5y6).
qpair(2,8,s1y7y9).
qpair(2,8,s4y6y9).
qpair(2,8,s3y4y9).
qpair(2,8,s3y5y6y9).
qpair(2,8,s0y1y3y5y6y9).
qpair(2,8,s0y3y4y5y6).
qpair(2,8,s0y1y3y5).
qpair(2,8,s1y3y4y6y9).
qpair(2,8,s9).
qpair(2,8,s1y3y5y9).
qpair(2,8,s1y7).
qpair(2,8,s3y4y5).
qpair(2,8,s6y9).
qpair(2,8,s0y4y5).
qpair(2,9,s0y8).
qpair(2,9,s4).
qpair(3,4,s5y8y9).
qpair(3,4,s0y5y8y9).
qpair(3,4,s5y8).
qpair(3,4,s1y8).
qpair(3,4,s8y9).
qpair(3,4,s0y8y9).
qpair(3,4,s5y9).
qpair(3,4,s0y8).
qpair(3,5,s0y1).
qpair(3,5,s0y8y9).
qpair(3,5,s4).
qpair(3,6,s1y7y8).
qpair(3,6,s0y5y8y9).
qpair(3,6,s5y8).
qpair(3,6,s1y8).
qpair(3,6,s0y5y8).
qpair(3,6,s7y8).
qpair(3,8,s6y7y9).
qpair(3,8,s2y5y7).
qpair(3,8,s4y5y6).
qpair(3,8,s4y7y9).
qpair(3,8,s2y5).
qpair(3,8,s0y2y4y6).
qpair(3,8,s0y2y7).
qpair(3,8,s0y2y4y9).
qpair(3,8,s0y1y2y5y6).
qpair(3,8,s1).
qpair(3,8,s4y5).
qpair(3,8,s0y7y9).
qpair(3,8,s1y5y9).
qpair(3,8,s5y6).
qpair(3,8,s0y1y4y5y9).
qpair(3,8,s4y6y9).
qpair(3,8,s0y2y4y5).
qpair(3,8,s0y2y6).
qpair(3,8,s0y1y4y9).
qpair(3,8,s0y4y9).
qpair(3,8,s2y5y6y9).
qpair(3,8,s1y6y7).
qpair(3,8,s5y6y9).
qpair(3,8,s1y4y5y6).
qpair(3,8,s1y4y5y9).
qpair(3,8,s2y6y9).
qpair(3,8,s1y7).
qpair(3,8,s0y1y2y6y9).
qpair(3,8,empty).
qpair(3,8,s0y1y2y5).
qpair(3,9,s0y1).
qpair(3,9,s1y8).
qpair(3,9,s4).
qpair(3,9,s5y8).
qpair(4,5,s3y8).
qpair(4,5,s1y8).
qpair(4,5,s8).
qpair(4,5,s7).
qpair(4,5,s3y7y8).
qpair(4,5,s7y8).
qpair(4,6,s0y8).
qpair(4,6,s1y9).
qpair(4,6,s5y8).
qpair(4,7,empty).
qpair(4,8,s6y7y9).
qpair(4,8,s3y5y6).
qpair(4,8,s0y1y2y3y5y6).
qpair(4,8,s0y3y9).
qpair(4,8,s1y6).
qpair(4,8,s0y2y5).
qpair(4,8,s0y1y2y3y6).
qpair(4,8,s0y1y2y3y9).
qpair(4,8,s1y5y9).
qpair(4,8,s0y1y3y5y9).
qpair(4,8,s3y6).
qpair(4,8,s1y7y9).
qpair(4,8,s5y9).
qpair(4,8,s1y2).
qpair(4,8,s2y3y5y6y9).
qpair(4,8,s7y9).
qpair(4,8,s0y1y2y6).
qpair(4,8,s0).
qpair(4,8,s5y6y9).
qpair(4,8,s1y3y5y9).
qpair(4,8,s1y3y7).
qpair(4,8,s0y1y6y9).
qpair(4,9,s2).
qpair(5,6,s0).
qpair(5,6,s0y4).
qpair(5,6,s9).
qpair(5,6,s1y4).
qpair(5,7,empty).
qpair(5,8,s2y3y6y9).
qpair(5,8,s2y4y6y9).
qpair(5,8,s1y2y3y4).
qpair(5,8,s0y1y9).
qpair(5,8,s0y3y9).
qpair(5,8,s1y3y4y6).
qpair(5,8,s1y3).
qpair(5,8,s0y2y7).
qpair(5,8,s0y2y4).
qpair(5,8,s0y1y2y3y6).
qpair(5,8,s0y1y2).
qpair(5,8,s1y7y9).
qpair(5,8,s0y7).
qpair(5,8,s1y2y7).
qpair(5,8,s0y2y3y4y6).
qpair(5,8,s0y6y9).
qpair(5,8,s0y1y3y4y9).
qpair(5,8,s0y3y4).
qpair(5,8,s1y2y3).
qpair(5,8,s1y2y9).
qpair(5,8,s1y3y7).
qpair(5,8,s0y1y6y9).
qpair(6,7,s5y8).
qpair(6,7,s4y5).
qpair(6,7,s0y5y8).
qpair(6,7,s0y5).
qpair(6,7,s0).
qpair(6,7,s0y8).
qpair(6,8,s0y5y7).
qpair(6,8,s1y5y7).
qpair(6,8,s3y5y9).
qpair(6,8,s2y3y4y9).
qpair(6,8,s0y5).
qpair(6,8,s0y3y4y5).
qpair(6,8,s2y4y5).
qpair(6,8,s2y3y9).
qpair(6,8,s0y4y7).
qpair(6,8,s0y1y3y9).
qpair(6,8,s1y5y9).
qpair(6,8,s0y1y2).
qpair(6,8,s2y5y9).
qpair(6,8,s1y2y3y4y5).
qpair(6,8,s0y1).
qpair(6,8,s0y3y5y9).
qpair(6,8,s1y2).
qpair(6,8,s1y5).
qpair(6,8,s2y4y5y9).
qpair(6,8,s3y5y7).
qpair(6,8,s2y9).
qpair(6,8,s0y2y3y4y5).
qpair(7,9,empty).
qpair(8,9,s0y1y4y5).
qpair(8,9,s4y6).
qpair(8,9,s0y1y3y4y5).
qpair(8,9,s5y7).
qpair(8,9,s2y3y4y6).
qpair(8,9,s1y2y3y4y6).
qpair(8,9,s4y5y6).
qpair(8,9,s2y4y5).
qpair(8,9,s2y3y6).
qpair(8,9,s0y5y6).
qpair(8,9,s5y6y7).
qpair(8,9,s1).
qpair(8,9,s4y5).
qpair(8,9,s3y6).
qpair(8,9,s1y2).
qpair(8,9,s2y7).
qpair(8,9,s0y2y6).
qpair(8,9,s0y2y3y4y6).
qpair(8,9,s0y2y3y6).
qpair(8,9,s1y6y7).
qpair(8,9,s1y2y6).
qpair(8,9,s2y3y7).
qpair(8,9,s0y4y5).

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
in(0,s0y8y5).
in(0,s0y8y5y9).
in(0,s0y8y6).
in(0,s0y9y2).
in(0,s0y9y2y3).
in(0,s0y9y2y4).
in(0,s0y9y2y5).
in(0,s0y9y2y6).
in(0,s0y9y3).
in(0,s0y9y3y4).
in(0,s0y9y3y5).
in(0,s0y9y3y6).
in(0,s0y9y4).
in(0,s0y9y4y5).
in(0,s0y9y4y6).
in(0,s0y9y5).
in(0,s0y9y5y6).
in(0,s0y9y6).
in(0,s0y9y7).
in(1,s1y9y7).
in(1,s8y1).
in(1,s8y1y3).
in(1,s8y1y7).
in(2,s0y9y2).
in(2,s0y9y2y3).
in(2,s0y9y2y4).
in(2,s0y9y2y5).
in(2,s0y9y2y6).
in(2,s8y2).
in(2,s8y2y3).
in(2,s8y2y3y7).
in(2,s8y2y7).
in(2,s9y2).
in(2,s9y2y3).
in(2,s9y2y3y4).
in(2,s9y2y3y5).
in(2,s9y2y3y6).
in(2,s9y2y4).
in(2,s9y2y4y5).
in(2,s9y2y4y6).
in(2,s9y2y5).
in(2,s9y2y5y6).
in(2,s9y2y6).
in(2,s9y2y7).
in(3,s0y9y2y3).
in(3,s0y9y3).
in(3,s0y9y3y4).
in(3,s0y9y3y5).
in(3,s0y9y3y6).
in(3,s8y1y3).
in(3,s8y2y3).
in(3,s8y2y3y7).
in(3,s8y3).
in(3,s8y3y7).
in(3,s9y2y3).
in(3,s9y2y3y4).
in(3,s9y2y3y5).
in(3,s9y2y3y6).
in(3,s9y3).
in(3,s9y3y4).
in(3,s9y3y4y5).
in(3,s9y3y4y6).
in(3,s9y3y5).
in(3,s9y3y5y6).
in(3,s9y3y6).
in(3,s9y3y7).
in(4,s0y9y2y4).
in(4,s0y9y3y4).
in(4,s0y9y4).
in(4,s0y9y4y5).
in(4,s0y9y4y6).
in(4,s8y4).
in(4,s8y4y6).
in(4,s8y9y4).
in(4,s8y9y4y6).
in(4,s9y2y3y4).
in(4,s9y2y4).
in(4,s9y2y4y5).
in(4,s9y2y4y6).
in(4,s9y3y4).
in(4,s9y3y4y5).
in(4,s9y3y4y6).
in(4,s9y4).
in(4,s9y4y5).
in(4,s9y4y5y6).
in(4,s9y4y6).
in(4,s9y4y7).
in(5,s0y8y5).
in(5,s0y8y5y9).
in(5,s0y9y2y5).
in(5,s0y9y3y5).
in(5,s0y9y4y5).
in(5,s0y9y5).
in(5,s0y9y5y6).
in(5,s8y5).
in(5,s8y5y6).
in(5,s8y9y5).
in(5,s9y2y3y5).
in(5,s9y2y4y5).
in(5,s9y2y5).
in(5,s9y2y5y6).
in(5,s9y3y4y5).
in(5,s9y3y5).
in(5,s9y3y5y6).
in(5,s9y4y5).
in(5,s9y4y5y6).
in(5,s9y5).
in(5,s9y5y6).
in(5,s9y5y7).
in(6,s0y8y6).
in(6,s0y9y2y6).
in(6,s0y9y3y6).
in(6,s0y9y4y6).
in(6,s0y9y5y6).
in(6,s0y9y6).
in(6,s8y4y6).
in(6,s8y5y6).
in(6,s8y6).
in(6,s8y9y4y6).
in(6,s8y9y6).
in(6,s9y2y3y6).
in(6,s9y2y4y6).
in(6,s9y2y5y6).
in(6,s9y2y6).
in(6,s9y3y4y6).
in(6,s9y3y5y6).
in(6,s9y3y6).
in(6,s9y4y5y6).
in(6,s9y4y6).
in(6,s9y5y6).
in(6,s9y6).
in(6,s9y6y7).
in(7,s0y9y7).
in(7,s1y9y7).
in(7,s8y1y7).
in(7,s8y2y3y7).
in(7,s8y2y7).
in(7,s8y3y7).
in(7,s8y7).
in(7,s9y2y7).
in(7,s9y3y7).
in(7,s9y4y7).
in(7,s9y5y7).
in(7,s9y6y7).
in(7,s9y7).
in(8,s0y8y5).
in(8,s0y8y5y9).
in(8,s0y8y6).
in(8,s8y1).
in(8,s8y1y3).
in(8,s8y1y7).
in(8,s8y2).
in(8,s8y2y3).
in(8,s8y2y3y7).
in(8,s8y2y7).
in(8,s8y3).
in(8,s8y3y7).
in(8,s8y4).
in(8,s8y4y6).
in(8,s8y5).
in(8,s8y5y6).
in(8,s8y6).
in(8,s8y7).
in(8,s8y9y4).
in(8,s8y9y4y6).
in(8,s8y9y5).
in(8,s8y9y6).
in(9,s0y8y5y9).
in(9,s0y9y2).
in(9,s0y9y2y3).
in(9,s0y9y2y4).
in(9,s0y9y2y5).
in(9,s0y9y2y6).
in(9,s0y9y3).
in(9,s0y9y3y4).
in(9,s0y9y3y5).
in(9,s0y9y3y6).
in(9,s0y9y4).
in(9,s0y9y4y5).
in(9,s0y9y4y6).
in(9,s0y9y5).
in(9,s0y9y5y6).
in(9,s0y9y6).
in(9,s0y9y7).
in(9,s1y9y7).
in(9,s8y9y4).
in(9,s8y9y4y6).
in(9,s8y9y5).
in(9,s8y9y6).
in(9,s9y2).
in(9,s9y2y3).
in(9,s9y2y3y4).
in(9,s9y2y3y5).
in(9,s9y2y3y6).
in(9,s9y2y4).
in(9,s9y2y4y5).
in(9,s9y2y4y6).
in(9,s9y2y5).
in(9,s9y2y5y6).
in(9,s9y2y6).
in(9,s9y2y7).
in(9,s9y3).
in(9,s9y3y4).
in(9,s9y3y4y5).
in(9,s9y3y4y6).
in(9,s9y3y5).
in(9,s9y3y5y6).
in(9,s9y3y6).
in(9,s9y3y7).
in(9,s9y4).
in(9,s9y4y5).
in(9,s9y4y5y6).
in(9,s9y4y6).
in(9,s9y4y7).
in(9,s9y5).
in(9,s9y5y6).
in(9,s9y5y7).
in(9,s9y6).
in(9,s9y6y7).
in(9,s9y7).


% ===== Base program loaded from: results/wc_sweep_10_2026_20260213_095633/rep8/lps/abapc_inc_full_base_rep8_optN.lp =====
% ===== Base program sha256 (raw): caf9de62b2762b3563307b0dde0d1c611327834f7f4f72ec7181aa2ba8dffb1c =====
% ===== Base program sha256 (normalized): 4ff0a71c9f9dc58bd0943e30c90f3959905815f0ae45ab525eda19f0785bda06 =====
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
{mus(949)}.
{mus(950)}.
{mus(951)}.
{mus(952)}.
{mus(953)}.
{mus(954)}.
{mus(955)}.
{mus(956)}.
{mus(957)}.
{mus(958)}.
{mus(959)}.
{mus(960)}.
{mus(961)}.
{mus(962)}.
{mus(963)}.
{mus(964)}.
{mus(965)}.
{mus(966)}.
{mus(967)}.
{mus(968)}.
{mus(969)}.
{mus(970)}.
{mus(971)}.
{mus(972)}.
{mus(973)}.
{mus(974)}.
{mus(975)}.
{mus(976)}.
{mus(977)}.
{mus(978)}.
{mus(979)}.
{mus(980)}.
{mus(981)}.
{mus(982)}.
{mus(983)}.
{mus(984)}.
{mus(985)}.
{mus(986)}.
{mus(987)}.
{mus(988)}.
{mus(989)}.
{mus(990)}.
{mus(991)}.
{mus(992)}.
{mus(993)}.
{mus(994)}.
{mus(995)}.
{mus(996)}.
{mus(997)}.
{mus(998)}.
{mus(999)}.
{mus(1000)}.
{mus(1001)}.
{mus(1002)}.
{mus(1003)}.
{mus(1004)}.
{mus(1005)}.
{mus(1006)}.
{mus(1007)}.
{mus(1008)}.
{mus(1009)}.
{mus(1010)}.
{mus(1011)}.
{mus(1012)}.
{mus(1013)}.
{mus(1014)}.
{mus(1015)}.
{mus(1016)}.
{mus(1017)}.
{mus(1018)}.
{mus(1019)}.
{mus(1020)}.
{mus(1021)}.
{mus(1022)}.
{mus(1023)}.
{mus(1024)}.
{mus(1025)}.
{mus(1026)}.
{mus(1027)}.
{mus(1028)}.
{mus(1029)}.
{mus(1030)}.
{mus(1031)}.
{mus(1032)}.
{mus(1033)}.
{mus(1034)}.
{mus(1035)}.
{mus(1036)}.
{mus(1037)}.
{mus(1038)}.
{mus(1039)}.
{mus(1040)}.
{mus(1041)}.
{mus(1042)}.
{mus(1043)}.
{mus(1044)}.
{mus(1045)}.
{mus(1046)}.
{mus(1047)}.
{mus(1048)}.
{mus(1049)}.
{mus(1050)}.
{mus(1051)}.
{mus(1052)}.
{mus(1053)}.
{mus(1054)}.
{mus(1055)}.
{mus(1056)}.
{mus(1057)}.
{mus(1058)}.
{mus(1059)}.
{mus(1060)}.
{mus(1061)}.
{mus(1062)}.
{mus(1063)}.
{mus(1064)}.
{mus(1065)}.
{mus(1066)}.
{mus(1067)}.
{mus(1068)}.
{mus(1069)}.
{mus(1070)}.
{mus(1071)}.
{mus(1072)}.
{mus(1073)}.
{mus(1074)}.
{mus(1075)}.
{mus(1076)}.
{mus(1077)}.
{mus(1078)}.
{mus(1079)}.
{mus(1080)}.
{mus(1081)}.
{mus(1082)}.
{mus(1083)}.
{mus(1084)}.
{mus(1085)}.
{mus(1086)}.
{mus(1087)}.
{mus(1088)}.
{mus(1089)}.
{mus(1090)}.
{mus(1091)}.
{mus(1092)}.
{mus(1093)}.
{mus(1094)}.
{mus(1095)}.
{mus(1096)}.
{mus(1097)}.
{mus(1098)}.
{mus(1099)}.
{mus(1100)}.
{mus(1101)}.
{mus(1102)}.
{mus(1103)}.
{mus(1104)}.
{mus(1105)}.
{mus(1106)}.
{mus(1107)}.
{mus(1108)}.
{mus(1109)}.
{mus(1110)}.
{mus(1111)}.
{mus(1112)}.
{mus(1113)}.
{mus(1114)}.
{mus(1115)}.
{mus(1116)}.
{mus(1117)}.
{mus(1118)}.
{mus(1119)}.
{mus(1120)}.
{mus(1121)}.
{mus(1122)}.
{mus(1123)}.
{mus(1124)}.
{mus(1125)}.
{mus(1126)}.
{mus(1127)}.
{mus(1128)}.
{mus(1129)}.
{mus(1130)}.
{mus(1131)}.
{mus(1132)}.
{mus(1133)}.
{mus(1134)}.
{mus(1135)}.
{mus(1136)}.
{mus(1137)}.
{mus(1138)}.
{mus(1139)}.
{mus(1140)}.
{mus(1141)}.
{mus(1142)}.
{mus(1143)}.
{mus(1144)}.
{mus(1145)}.
{mus(1146)}.
{mus(1147)}.
{mus(1148)}.
{mus(1149)}.
{mus(1150)}.
{mus(1151)}.
{mus(1152)}.
{mus(1153)}.
{mus(1154)}.
{mus(1155)}.
{mus(1156)}.
{mus(1157)}.
{mus(1158)}.
{mus(1159)}.
{mus(1160)}.
{mus(1161)}.
{mus(1162)}.
{mus(1163)}.
{mus(1164)}.
{mus(1165)}.
{mus(1166)}.
{mus(1167)}.
{mus(1168)}.
{mus(1169)}.
{mus(1170)}.
{mus(1171)}.
{mus(1172)}.
{mus(1173)}.
{mus(1174)}.
{mus(1175)}.
{mus(1176)}.
{mus(1177)}.
{mus(1178)}.
{mus(1179)}.
{mus(1180)}.
{mus(1181)}.
{mus(1182)}.
{mus(1183)}.
{mus(1184)}.
{mus(1185)}.
{mus(1186)}.
{mus(1187)}.
{mus(1188)}.
{mus(1189)}.
{mus(1190)}.
{mus(1191)}.
{mus(1192)}.
{mus(1193)}.
{mus(1194)}.
{mus(1195)}.
{mus(1196)}.
{mus(1197)}.
{mus(1198)}.
{mus(1199)}.
{mus(1200)}.
{mus(1201)}.
{mus(1202)}.
{mus(1203)}.
{mus(1204)}.
{mus(1205)}.
{mus(1206)}.
{mus(1207)}.
{mus(1208)}.
{mus(1209)}.
{mus(1210)}.
{mus(1211)}.
{mus(1212)}.
{mus(1213)}.
{mus(1214)}.
{mus(1215)}.
{mus(1216)}.
{mus(1217)}.
{mus(1218)}.
{mus(1219)}.
{mus(1220)}.
{mus(1221)}.
{mus(1222)}.
{mus(1223)}.
{mus(1224)}.
{mus(1225)}.
{mus(1226)}.
{mus(1227)}.
{mus(1228)}.
{mus(1229)}.
{mus(1230)}.
{mus(1231)}.
{mus(1232)}.
{mus(1233)}.
{mus(1234)}.
{mus(1235)}.
{mus(1236)}.
{mus(1237)}.
{mus(1238)}.
{mus(1239)}.
{mus(1240)}.
{mus(1241)}.
{mus(1242)}.
{mus(1243)}.
{mus(1244)}.
{mus(1245)}.
{mus(1246)}.
{mus(1247)}.
{mus(1248)}.
{mus(1249)}.
{mus(1250)}.
{mus(1251)}.
{mus(1252)}.
{mus(1253)}.
{mus(1254)}.
{mus(1255)}.
{mus(1256)}.
{mus(1257)}.
{mus(1258)}.
{mus(1259)}.
{mus(1260)}.
{mus(1261)}.
{mus(1262)}.
{mus(1263)}.
{mus(1264)}.
{mus(1265)}.
{mus(1266)}.
{mus(1267)}.
{mus(1268)}.
{mus(1269)}.
{mus(1270)}.
{mus(1271)}.
{mus(1272)}.
{mus(1273)}.
{mus(1274)}.
{mus(1275)}.
{mus(1276)}.
{mus(1277)}.
{mus(1278)}.
{mus(1279)}.
{mus(1280)}.
{mus(1281)}.
{mus(1282)}.
{mus(1283)}.
{mus(1284)}.
{mus(1285)}.
{mus(1286)}.
{mus(1287)}.
{mus(1288)}.
{mus(1289)}.
{mus(1290)}.
{mus(1291)}.
{mus(1292)}.
{mus(1293)}.
{mus(1294)}.
{mus(1295)}.
{mus(1296)}.
{mus(1297)}.
{mus(1298)}.
{mus(1299)}.
{mus(1300)}.
{mus(1301)}.
{mus(1302)}.
{mus(1303)}.
{mus(1304)}.
{mus(1305)}.
{mus(1306)}.
{mus(1307)}.
{mus(1308)}.
{mus(1309)}.
{mus(1310)}.
{mus(1311)}.
{mus(1312)}.
{mus(1313)}.
{mus(1314)}.
{mus(1315)}.
{mus(1316)}.
{mus(1317)}.
{mus(1318)}.
{mus(1319)}.
{mus(1320)}.
{mus(1321)}.
{mus(1322)}.
{mus(1323)}.
{mus(1324)}.
{mus(1325)}.
{mus(1326)}.
{mus(1327)}.
{mus(1328)}.
{mus(1329)}.
{mus(1330)}.
{mus(1331)}.
{mus(1332)}.
{mus(1333)}.
{mus(1334)}.
{mus(1335)}.
{mus(1336)}.
{mus(1337)}.
{mus(1338)}.
{mus(1339)}.
{mus(1340)}.
{mus(1341)}.
{mus(1342)}.
{mus(1343)}.
{mus(1344)}.
{mus(1345)}.
{mus(1346)}.
{mus(1347)}.
{mus(1348)}.
{mus(1349)}.
{mus(1350)}.
{mus(1351)}.
{mus(1352)}.
{mus(1353)}.
{mus(1354)}.
{mus(1355)}.
{mus(1356)}.
{mus(1357)}.
{mus(1358)}.
{mus(1359)}.
{mus(1360)}.
{mus(1361)}.
{mus(1362)}.
{mus(1363)}.
{mus(1364)}.
{mus(1365)}.
{mus(1366)}.
{mus(1367)}.
{mus(1368)}.
{mus(1369)}.
{mus(1370)}.
{mus(1371)}.
{mus(1372)}.
{mus(1373)}.
{mus(1374)}.
{mus(1375)}.
{mus(1376)}.
{mus(1377)}.
{mus(1378)}.
{mus(1379)}.
{mus(1380)}.
{mus(1381)}.
{mus(1382)}.
{mus(1383)}.
{mus(1384)}.
{mus(1385)}.
{mus(1386)}.
{mus(1387)}.
{mus(1388)}.
{mus(1389)}.
{mus(1390)}.
{mus(1391)}.
{mus(1392)}.
{mus(1393)}.
{mus(1394)}.
{mus(1395)}.
{mus(1396)}.
{mus(1397)}.
{mus(1398)}.
{mus(1399)}.
{mus(1400)}.
{mus(1401)}.
{mus(1402)}.
{mus(1403)}.
{mus(1404)}.
{mus(1405)}.
{mus(1406)}.
{mus(1407)}.
{mus(1408)}.
{mus(1409)}.
{mus(1410)}.
{mus(1411)}.
{mus(1412)}.
{mus(1413)}.
{mus(1414)}.
{mus(1415)}.
{mus(1416)}.
{mus(1417)}.
{mus(1418)}.
{mus(1419)}.
{mus(1420)}.
{mus(1421)}.
{mus(1422)}.
{mus(1423)}.
{mus(1424)}.
{mus(1425)}.
{mus(1426)}.
{mus(1427)}.
{mus(1428)}.
{mus(1429)}.
{mus(1430)}.
{mus(1431)}.
{mus(1432)}.
{mus(1433)}.
{mus(1434)}.
{mus(1435)}.
{mus(1436)}.
{mus(1437)}.
{mus(1438)}.
{mus(1439)}.
{mus(1440)}.
{mus(1441)}.
{mus(1442)}.
{mus(1443)}.
{mus(1444)}.
{mus(1445)}.
{mus(1446)}.
{mus(1447)}.
{mus(1448)}.
{mus(1449)}.
{mus(1450)}.
{mus(1451)}.
{mus(1452)}.
{mus(1453)}.
{mus(1454)}.
{mus(1455)}.
{mus(1456)}.
{mus(1457)}.
{mus(1458)}.
{mus(1459)}.
{mus(1460)}.
{mus(1461)}.
{mus(1462)}.
{mus(1463)}.
{mus(1464)}.
{mus(1465)}.
{mus(1466)}.
{mus(1467)}.
{mus(1468)}.
{mus(1469)}.
{mus(1470)}.
{mus(1471)}.
{mus(1472)}.
{mus(1473)}.
{mus(1474)}.
{mus(1475)}.
{mus(1476)}.
{mus(1477)}.
{mus(1478)}.
{mus(1479)}.
{mus(1480)}.
{mus(1481)}.
{mus(1482)}.
{mus(1483)}.
{mus(1484)}.
{mus(1485)}.
{mus(1486)}.
{mus(1487)}.
{mus(1488)}.
{mus(1489)}.
{mus(1490)}.
{mus(1491)}.
{mus(1492)}.
{mus(1493)}.
{mus(1494)}.
{mus(1495)}.
{mus(1496)}.
{mus(1497)}.
{mus(1498)}.
{mus(1499)}.
{mus(1500)}.
{mus(1501)}.
{mus(1502)}.
{mus(1503)}.
{mus(1504)}.
{mus(1505)}.
{mus(1506)}.
{mus(1507)}.
{mus(1508)}.
{mus(1509)}.
{mus(1510)}.
{mus(1511)}.
{mus(1512)}.
{mus(1513)}.
{mus(1514)}.
{mus(1515)}.
{mus(1516)}.
{mus(1517)}.
{mus(1518)}.
{mus(1519)}.
{mus(1520)}.
{mus(1521)}.
{mus(1522)}.
{mus(1523)}.
{mus(1524)}.
{mus(1525)}.
{mus(1526)}.
{mus(1527)}.
{mus(1528)}.
{mus(1529)}.
{mus(1530)}.
{mus(1531)}.
{mus(1532)}.
{mus(1533)}.
{mus(1534)}.
{mus(1535)}.
{mus(1536)}.
{mus(1537)}.
{mus(1538)}.
{mus(1539)}.
{mus(1540)}.
{mus(1541)}.
{mus(1542)}.
{mus(1543)}.
{mus(1544)}.
{mus(1545)}.
{mus(1546)}.
{mus(1547)}.
{mus(1548)}.
{mus(1549)}.
{mus(1550)}.
{mus(1551)}.
{mus(1552)}.
{mus(1553)}.
{mus(1554)}.
{mus(1555)}.
{mus(1556)}.
{mus(1557)}.
{mus(1558)}.
{mus(1559)}.
{mus(1560)}.
{mus(1561)}.
{mus(1562)}.
{mus(1563)}.
{mus(1564)}.
{mus(1565)}.
{mus(1566)}.
{mus(1567)}.
{mus(1568)}.
{mus(1569)}.
{mus(1570)}.
{mus(1571)}.
{mus(1572)}.
{mus(1573)}.
{mus(1574)}.
{mus(1575)}.
{mus(1576)}.
{mus(1577)}.
{mus(1578)}.
{mus(1579)}.
{mus(1580)}.
{mus(1581)}.
{mus(1582)}.
{mus(1583)}.
{mus(1584)}.
{mus(1585)}.
{mus(1586)}.
{mus(1587)}.
{mus(1588)}.
{mus(1589)}.
{mus(1590)}.
{mus(1591)}.
{mus(1592)}.
{mus(1593)}.
{mus(1594)}.
{mus(1595)}.
{mus(1596)}.
{mus(1597)}.
{mus(1598)}.
{mus(1599)}.
{mus(1600)}.
{mus(1601)}.
{mus(1602)}.
{mus(1603)}.
{mus(1604)}.
{mus(1605)}.
{mus(1606)}.
{mus(1607)}.
{mus(1608)}.
{mus(1609)}.
{mus(1610)}.
{mus(1611)}.
{mus(1612)}.
{mus(1613)}.
{mus(1614)}.
{mus(1615)}.
{mus(1616)}.
{mus(1617)}.
{mus(1618)}.
{mus(1619)}.
{mus(1620)}.
{mus(1621)}.
{mus(1622)}.
{mus(1623)}.
{mus(1624)}.
{mus(1625)}.
{mus(1626)}.
{mus(1627)}.
{mus(1628)}.
{mus(1629)}.
{mus(1630)}.
{mus(1631)}.
{mus(1632)}.
{mus(1633)}.
{mus(1634)}.
{mus(1635)}.
{mus(1636)}.
{mus(1637)}.
{mus(1638)}.
{mus(1639)}.
{mus(1640)}.
{mus(1641)}.
{mus(1642)}.
{mus(1643)}.
{mus(1644)}.
{mus(1645)}.
{mus(1646)}.
{mus(1647)}.
{mus(1648)}.
{mus(1649)}.
{mus(1650)}.
{mus(1651)}.
{mus(1652)}.
{mus(1653)}.
{mus(1654)}.
{mus(1655)}.
{mus(1656)}.
{mus(1657)}.
{mus(1658)}.
{mus(1659)}.
{mus(1660)}.
{mus(1661)}.
{mus(1662)}.
{mus(1663)}.
{mus(1664)}.
{mus(1665)}.
{mus(1666)}.
{mus(1667)}.
{mus(1668)}.
{mus(1669)}.
{mus(1670)}.
{mus(1671)}.
{mus(1672)}.
{mus(1673)}.
{mus(1674)}.
{mus(1675)}.
{mus(1676)}.
{mus(1677)}.
{mus(1678)}.
{mus(1679)}.
{mus(1680)}.
{mus(1681)}.
{mus(1682)}.
{mus(1683)}.
{mus(1684)}.
{mus(1685)}.

% ===== Adorned Facts =====
ext_dep(0,1,s2):-mus(1).
ext_indep(0,1,s3):-mus(2).
ext_indep(0,1,s4):-mus(3).
ext_indep(0,1,s6):-mus(4).
ext_dep(0,1,s7):-mus(5).
ext_indep(0,1,s8):-mus(6).
ext_indep(0,1,s9):-mus(7).
ext_dep(0,1,s2y3):-mus(8).
ext_indep(0,1,s2y7):-mus(9).
ext_indep(0,1,s8y2):-mus(10).
ext_indep(0,1,s3y7):-mus(11).
ext_indep(0,1,s8y3):-mus(12).
ext_indep(0,1,s4y6):-mus(13).
ext_indep(0,1,s8y4):-mus(14).
ext_dep(0,1,s9y4):-mus(15).
ext_indep(0,1,s8y6):-mus(16).
ext_indep(0,1,s9y6):-mus(17).
ext_indep(0,1,s8y7):-mus(18).
ext_indep(0,1,s8y9):-mus(19).
ext_dep(0,1,s2y3y7):-mus(20).
ext_indep(0,1,s8y2y3):-mus(21).
ext_indep(0,1,s8y2y7):-mus(22).
ext_indep(0,1,s8y3y7):-mus(23).
ext_dep(0,1,s8y4y6):-mus(24).
ext_dep(0,1,s9y4y6):-mus(25).
ext_dep(0,1,s8y9y4):-mus(26).
ext_indep(0,1,s8y9y6):-mus(27).
ext_indep(0,1,s8y2y3y7):-mus(28).
ext_dep(0,1,s8y9y4y6):-mus(29).
ext_dep(0,2,empty):-mus(30).
ext_dep(0,2,s1):-mus(31).
ext_dep(0,2,s3):-mus(32).
ext_dep(0,2,s4):-mus(33).
ext_dep(0,2,s5):-mus(34).
ext_indep(0,2,s6):-mus(35).
ext_indep(0,2,s7):-mus(36).
ext_dep(0,2,s8):-mus(37).
ext_dep(0,2,s9):-mus(38).
ext_dep(0,2,s3y6):-mus(39).
ext_dep(0,2,s3y7):-mus(40).
ext_dep(0,2,s8y3):-mus(41).
ext_dep(0,2,s6y7):-mus(42).
ext_dep(0,2,s8y6):-mus(43).
ext_dep(0,2,s8y7):-mus(44).
ext_dep(0,2,s8y3y7):-mus(45).
ext_dep(0,3,empty):-mus(46).
ext_dep(0,3,s1):-mus(47).
ext_dep(0,3,s2):-mus(48).
ext_dep(0,3,s4):-mus(49).
ext_indep(0,3,s5):-mus(50).
ext_dep(0,3,s6):-mus(51).
ext_dep(0,3,s7):-mus(52).
ext_dep(0,3,s8):-mus(53).
ext_indep(0,3,s9):-mus(54).
ext_dep(0,3,s1y5):-mus(55).
ext_indep(0,3,s8y1):-mus(56).
ext_dep(0,3,s1y9):-mus(57).
ext_dep(0,3,s2y6):-mus(58).
ext_dep(0,3,s2y7):-mus(59).
ext_dep(0,3,s8y2):-mus(60).
ext_dep(0,3,s8y5):-mus(61).
ext_dep(0,3,s9y5):-mus(62).
ext_dep(0,3,s6y7):-mus(63).
ext_dep(0,3,s8y6):-mus(64).
ext_dep(0,3,s8y7):-mus(65).
ext_dep(0,3,s8y9):-mus(66).
ext_dep(0,3,s8y2y7):-mus(67).
ext_dep(0,3,s8y9y5):-mus(68).
ext_indep(0,4,s5):-mus(69).
ext_indep(0,4,s6):-mus(70).
ext_indep(0,4,s9):-mus(71).
ext_indep(0,4,s8y1):-mus(72).
ext_indep(0,4,s2y3):-mus(73).
ext_indep(0,4,s2y7):-mus(74).
ext_indep(0,4,s8y2):-mus(75).
ext_indep(0,4,s3y7):-mus(76).
ext_indep(0,4,s8y3):-mus(77).
ext_indep(0,4,s8y7):-mus(78).
ext_indep(0,4,s2y3y7):-mus(79).
ext_indep(0,4,s8y2y3):-mus(80).
ext_indep(0,4,s8y2y7):-mus(81).
ext_indep(0,4,s8y3y7):-mus(82).
ext_indep(0,4,s8y2y3y7):-mus(83).
ext_indep(0,5,s4):-mus(84).
ext_dep(0,5,s6):-mus(85).
ext_dep(0,5,s9):-mus(86).
ext_dep(0,6,s4):-mus(87).
ext_dep(0,6,s5):-mus(88).
ext_dep(0,6,s9):-mus(89).
ext_dep(0,6,s1y4):-mus(90).
ext_dep(0,6,s1y5):-mus(91).
ext_dep(0,6,s4y5):-mus(92).
ext_dep(0,6,s4y7):-mus(93).
ext_dep(0,6,s8y4):-mus(94).
ext_dep(0,6,s5y7):-mus(95).
ext_dep(0,6,s8y5):-mus(96).
ext_dep(0,7,empty):-mus(97).
ext_dep(0,7,s1):-mus(98).
ext_dep(0,7,s2):-mus(99).
ext_dep(0,7,s3):-mus(100).
ext_dep(0,7,s4):-mus(101).
ext_indep(0,7,s5):-mus(102).
ext_dep(0,7,s6):-mus(103).
ext_dep(0,7,s8):-mus(104).
ext_dep(0,7,s9):-mus(105).
ext_dep(0,7,s2y3):-mus(106).
ext_dep(0,7,s2y6):-mus(107).
ext_dep(0,7,s8y2):-mus(108).
ext_indep(0,7,s3y6):-mus(109).
ext_indep(0,7,s8y3):-mus(110).
ext_dep(0,7,s5y6):-mus(111).
ext_dep(0,7,s8y5):-mus(112).
ext_dep(0,7,s8y6):-mus(113).
ext_dep(0,7,s8y2y3):-mus(114).
ext_indep(0,7,s8y5y6):-mus(115).
ext_dep(0,8,empty):-mus(116).
ext_dep(0,8,s1):-mus(117).
ext_indep(0,8,s2):-mus(118).
ext_indep(0,8,s3):-mus(119).
ext_dep(0,8,s4):-mus(120).
ext_dep(0,8,s5):-mus(121).
ext_dep(0,8,s6):-mus(122).
ext_dep(0,8,s7):-mus(123).
ext_dep(0,8,s9):-mus(124).
ext_dep(0,8,s1y2):-mus(125).
ext_dep(0,8,s1y3):-mus(126).
ext_dep(0,8,s1y4):-mus(127).
ext_dep(0,8,s1y5):-mus(128).
ext_dep(0,8,s1y6):-mus(129).
ext_dep(0,8,s1y7):-mus(130).
ext_dep(0,8,s1y9):-mus(131).
ext_dep(0,8,s2y3):-mus(132).
ext_dep(0,8,s2y4):-mus(133).
ext_dep(0,8,s2y5):-mus(134).
ext_indep(0,8,s2y6):-mus(135).
ext_dep(0,8,s2y7):-mus(136).
ext_dep(0,8,s9y2):-mus(137).
ext_dep(0,8,s3y4):-mus(138).
ext_dep(0,8,s3y5):-mus(139).
ext_dep(0,8,s3y6):-mus(140).
ext_dep(0,8,s3y7):-mus(141).
ext_dep(0,8,s9y3):-mus(142).
ext_dep(0,8,s4y5):-mus(143).
ext_dep(0,8,s4y6):-mus(144).
ext_dep(0,8,s4y7):-mus(145).
ext_dep(0,8,s9y4):-mus(146).
ext_dep(0,8,s5y6):-mus(147).
ext_dep(0,8,s5y7):-mus(148).
ext_dep(0,8,s9y5):-mus(149).
ext_dep(0,8,s6y7):-mus(150).
ext_dep(0,8,s9y6):-mus(151).
ext_dep(0,8,s9y7):-mus(152).
ext_dep(0,8,s1y2y3):-mus(153).
ext_indep(0,8,s1y2y4):-mus(154).
ext_indep(0,8,s1y2y5):-mus(155).
ext_indep(0,8,s1y2y6):-mus(156).
ext_dep(0,8,s1y2y7):-mus(157).
ext_dep(0,8,s1y2y9):-mus(158).
ext_dep(0,8,s1y3y4):-mus(159).
ext_dep(0,8,s1y3y5):-mus(160).
ext_dep(0,8,s1y3y6):-mus(161).
ext_dep(0,8,s1y3y7):-mus(162).
ext_dep(0,8,s1y3y9):-mus(163).
ext_dep(0,8,s1y4y5):-mus(164).
ext_dep(0,8,s1y4y6):-mus(165).
ext_indep(0,8,s1y4y7):-mus(166).
ext_dep(0,8,s1y4y9):-mus(167).
ext_dep(0,8,s1y5y6):-mus(168).
ext_dep(0,8,s1y5y7):-mus(169).
ext_dep(0,8,s1y5y9):-mus(170).
ext_dep(0,8,s1y6y7):-mus(171).
ext_dep(0,8,s1y6y9):-mus(172).
ext_dep(0,8,s1y9y7):-mus(173).
ext_dep(0,8,s2y3y4):-mus(174).
ext_dep(0,8,s2y3y5):-mus(175).
ext_dep(0,8,s2y3y6):-mus(176).
ext_dep(0,8,s2y3y7):-mus(177).
ext_dep(0,8,s9y2y3):-mus(178).
ext_dep(0,8,s2y4y5):-mus(179).
ext_dep(0,8,s2y4y6):-mus(180).
ext_dep(0,8,s2y4y7):-mus(181).
ext_dep(0,8,s9y2y4):-mus(182).
ext_dep(0,8,s2y5y6):-mus(183).
ext_dep(0,8,s2y5y7):-mus(184).
ext_dep(0,8,s9y2y5):-mus(185).
ext_dep(0,8,s2y6y7):-mus(186).
ext_dep(0,8,s9y2y6):-mus(187).
ext_dep(0,8,s9y2y7):-mus(188).
ext_dep(0,8,s3y4y5):-mus(189).
ext_dep(0,8,s3y4y6):-mus(190).
ext_dep(0,8,s3y4y7):-mus(191).
ext_dep(0,8,s9y3y4):-mus(192).
ext_dep(0,8,s3y5y6):-mus(193).
ext_dep(0,8,s3y5y7):-mus(194).
ext_dep(0,8,s9y3y5):-mus(195).
ext_dep(0,8,s3y6y7):-mus(196).
ext_dep(0,8,s9y3y6):-mus(197).
ext_dep(0,8,s9y3y7):-mus(198).
ext_dep(0,8,s4y5y6):-mus(199).
ext_dep(0,8,s4y5y7):-mus(200).
ext_indep(0,8,s9y4y5):-mus(201).
ext_indep(0,8,s4y6y7):-mus(202).
ext_dep(0,8,s9y4y6):-mus(203).
ext_dep(0,8,s9y4y7):-mus(204).
ext_dep(0,8,s5y6y7):-mus(205).
ext_dep(0,8,s9y5y6):-mus(206).
ext_dep(0,8,s9y5y7):-mus(207).
ext_dep(0,8,s9y6y7):-mus(208).
ext_dep(0,8,s1y2y3y4):-mus(209).
ext_dep(0,8,s1y2y3y5):-mus(210).
ext_dep(0,8,s1y2y3y6):-mus(211).
ext_dep(0,8,s1y2y3y9):-mus(212).
ext_indep(0,8,s1y2y4y5):-mus(213).
ext_dep(0,8,s1y2y4y6):-mus(214).
ext_dep(0,8,s1y2y4y9):-mus(215).
ext_dep(0,8,s1y2y5y6):-mus(216).
ext_indep(0,8,s1y2y5y9):-mus(217).
ext_dep(0,8,s1y2y6y9):-mus(218).
ext_indep(0,8,s1y3y4y5):-mus(219).
ext_dep(0,8,s1y3y4y6):-mus(220).
ext_indep(0,8,s1y3y4y9):-mus(221).
ext_dep(0,8,s1y3y5y6):-mus(222).
ext_dep(0,8,s1y3y5y9):-mus(223).
ext_dep(0,8,s1y3y6y9):-mus(224).
ext_dep(0,8,s1y4y5y6):-mus(225).
ext_dep(0,8,s1y4y5y9):-mus(226).
ext_dep(0,8,s1y4y6y9):-mus(227).
ext_dep(0,8,s1y5y6y9):-mus(228).
ext_dep(0,8,s2y3y4y5):-mus(229).
ext_indep(0,8,s2y3y4y6):-mus(230).
ext_dep(0,8,s9y2y3y4):-mus(231).
ext_dep(0,8,s2y3y5y6):-mus(232).
ext_dep(0,8,s9y2y3y5):-mus(233).
ext_dep(0,8,s9y2y3y6):-mus(234).
ext_dep(0,8,s2y4y5y6):-mus(235).
ext_dep(0,8,s9y2y4y5):-mus(236).
ext_dep(0,8,s9y2y4y6):-mus(237).
ext_indep(0,8,s9y2y5y6):-mus(238).
ext_dep(0,8,s3y4y5y6):-mus(239).
ext_dep(0,8,s9y3y4y5):-mus(240).
ext_indep(0,8,s9y3y4y6):-mus(241).
ext_indep(0,8,s9y3y5y6):-mus(242).
ext_dep(0,8,s9y4y5y6):-mus(243).
ext_dep(0,8,s1y2y3y4y5):-mus(244).
ext_dep(0,8,s1y2y3y4y6):-mus(245).
ext_dep(0,8,s1y2y3y4y9):-mus(246).
ext_dep(0,8,s1y2y3y5y6):-mus(247).
ext_dep(0,8,s1y2y3y5y9):-mus(248).
ext_indep(0,8,s1y2y3y6y9):-mus(249).
ext_indep(0,8,s1y2y4y5y6):-mus(250).
ext_dep(0,8,s1y2y4y5y9):-mus(251).
ext_dep(0,8,s1y2y4y6y9):-mus(252).
ext_dep(0,8,s1y2y5y6y9):-mus(253).
ext_indep(0,8,s1y3y4y5y6):-mus(254).
ext_dep(0,8,s1y3y4y5y9):-mus(255).
ext_dep(0,8,s1y3y4y6y9):-mus(256).
ext_indep(0,8,s1y3y5y6y9):-mus(257).
ext_dep(0,8,s1y4y5y6y9):-mus(258).
ext_dep(0,8,s2y3y4y5y6):-mus(259).
ext_dep(0,8,s2y3y4y5y9):-mus(260).
ext_dep(0,8,s2y3y4y6y9):-mus(261).
ext_dep(0,8,s2y3y5y6y9):-mus(262).
ext_dep(0,8,s2y4y5y6y9):-mus(263).
ext_dep(0,8,s3y4y5y6y9):-mus(264).
ext_dep(0,8,s1y2y3y4y5y6):-mus(265).
ext_dep(0,8,s1y2y3y4y5y9):-mus(266).
ext_indep(0,8,s1y2y3y4y6y9):-mus(267).
ext_dep(0,8,s1y2y3y5y6y9):-mus(268).
ext_dep(0,8,s1y2y4y5y6y9):-mus(269).
ext_dep(0,8,s1y3y4y5y6y9):-mus(270).
ext_dep(0,8,s2y3y4y5y6y9):-mus(271).
ext_dep(0,8,s1y2y3y4y5y6y9):-mus(272).
ext_indep(0,9,s4):-mus(273).
ext_dep(0,9,s5):-mus(274).
ext_dep(0,9,s6):-mus(275).
ext_dep(1,2,s3):-mus(276).
ext_indep(1,2,s5):-mus(277).
ext_dep(1,2,s7):-mus(278).
ext_indep(1,2,s0y8):-mus(279).
ext_dep(1,2,s4y6):-mus(280).
ext_indep(1,2,s8y4):-mus(281).
ext_dep(1,2,s9y4):-mus(282).
ext_indep(1,2,s8y6):-mus(283).
ext_dep(1,2,s9y6):-mus(284).
ext_dep(1,2,s8y9):-mus(285).
ext_indep(1,2,s8y4y6):-mus(286).
ext_dep(1,2,s9y4y6):-mus(287).
ext_dep(1,2,s8y9y4):-mus(288).
ext_indep(1,2,s8y9y6):-mus(289).
ext_dep(1,2,s8y9y4y6):-mus(290).
ext_indep(1,3,s2):-mus(291).
ext_indep(1,3,s7):-mus(292).
ext_dep(1,4,empty):-mus(293).
ext_indep(1,4,s0):-mus(294).
ext_indep(1,4,s2):-mus(295).
ext_dep(1,4,s3):-mus(296).
ext_dep(1,4,s6):-mus(297).
ext_dep(1,4,s7):-mus(298).
ext_dep(1,4,s8):-mus(299).
ext_dep(1,4,s9):-mus(300).
ext_dep(1,4,s3y6):-mus(301).
ext_dep(1,4,s8y3):-mus(302).
ext_dep(1,4,s9y3):-mus(303).
ext_dep(1,4,s8y6):-mus(304).
ext_dep(1,4,s9y6):-mus(305).
ext_dep(1,4,s8y9):-mus(306).
ext_indep(1,4,s8y9y6):-mus(307).
ext_indep(1,5,s3):-mus(308).
ext_dep(1,5,s4):-mus(309).
ext_indep(1,5,s6):-mus(310).
ext_indep(1,5,s7):-mus(311).
ext_indep(1,5,s8):-mus(312).
ext_indep(1,5,s9):-mus(313).
ext_indep(1,5,s3y7):-mus(314).
ext_indep(1,5,s8y3):-mus(315).
ext_indep(1,5,s4y6):-mus(316).
ext_dep(1,5,s8y4):-mus(317).
ext_indep(1,5,s9y4):-mus(318).
ext_indep(1,5,s8y6):-mus(319).
ext_indep(1,5,s9y6):-mus(320).
ext_indep(1,5,s8y7):-mus(321).
ext_indep(1,5,s8y9):-mus(322).
ext_indep(1,5,s8y3y7):-mus(323).
ext_indep(1,5,s8y4y6):-mus(324).
ext_indep(1,5,s9y4y6):-mus(325).
ext_indep(1,5,s8y9y4):-mus(326).
ext_indep(1,5,s8y9y6):-mus(327).
ext_indep(1,5,s8y9y4y6):-mus(328).
ext_indep(1,6,empty):-mus(329).
ext_dep(1,6,s0):-mus(330).
ext_dep(1,6,s2):-mus(331).
ext_dep(1,6,s3):-mus(332).
ext_dep(1,6,s4):-mus(333).
ext_dep(1,6,s5):-mus(334).
ext_dep(1,6,s7):-mus(335).
ext_dep(1,6,s8):-mus(336).
ext_dep(1,6,s9):-mus(337).
ext_dep(1,6,s0y4):-mus(338).
ext_dep(1,6,s0y5):-mus(339).
ext_indep(1,6,s0y7):-mus(340).
ext_dep(1,6,s0y8):-mus(341).
ext_dep(1,6,s3y4):-mus(342).
ext_dep(1,6,s8y3):-mus(343).
ext_dep(1,6,s9y3):-mus(344).
ext_dep(1,6,s4y5):-mus(345).
ext_dep(1,6,s4y7):-mus(346).
ext_dep(1,6,s8y4):-mus(347).
ext_dep(1,6,s9y4):-mus(348).
ext_dep(1,6,s5y7):-mus(349).
ext_dep(1,6,s8y5):-mus(350).
ext_dep(1,6,s8y7):-mus(351).
ext_dep(1,6,s8y9):-mus(352).
ext_dep(1,6,s8y9y4):-mus(353).
ext_indep(1,7,s2):-mus(354).
ext_dep(1,7,s3):-mus(355).
ext_dep(1,7,s0y5):-mus(356).
ext_indep(1,7,s0y6):-mus(357).
ext_indep(1,7,s4y6):-mus(358).
ext_indep(1,7,s8y4):-mus(359).
ext_indep(1,7,s9y4):-mus(360).
ext_indep(1,7,s5y6):-mus(361).
ext_indep(1,7,s8y6):-mus(362).
ext_dep(1,7,s9y6):-mus(363).
ext_indep(1,7,s8y9):-mus(364).
ext_indep(1,7,s0y5y6):-mus(365).
ext_indep(1,7,s8y4y6):-mus(366).
ext_indep(1,7,s9y4y6):-mus(367).
ext_indep(1,7,s8y9y4):-mus(368).
ext_indep(1,7,s8y9y6):-mus(369).
ext_indep(1,7,s8y9y4y6):-mus(370).
ext_dep(1,8,empty):-mus(371).
ext_dep(1,8,s0):-mus(372).
ext_dep(1,8,s2):-mus(373).
ext_indep(1,8,s3):-mus(374).
ext_indep(1,8,s4):-mus(375).
ext_dep(1,8,s5):-mus(376).
ext_dep(1,8,s6):-mus(377).
ext_dep(1,8,s7):-mus(378).
ext_indep(1,8,s9):-mus(379).
ext_dep(1,8,s0y2):-mus(380).
ext_indep(1,8,s0y3):-mus(381).
ext_dep(1,8,s0y4):-mus(382).
ext_dep(1,8,s0y5):-mus(383).
ext_dep(1,8,s0y6):-mus(384).
ext_dep(1,8,s0y7):-mus(385).
ext_dep(1,8,s0y9):-mus(386).
ext_dep(1,8,s2y3):-mus(387).
ext_indep(1,8,s2y4):-mus(388).
ext_dep(1,8,s2y5):-mus(389).
ext_dep(1,8,s2y6):-mus(390).
ext_dep(1,8,s2y7):-mus(391).
ext_dep(1,8,s9y2):-mus(392).
ext_dep(1,8,s3y4):-mus(393).
ext_indep(1,8,s3y5):-mus(394).
ext_dep(1,8,s3y6):-mus(395).
ext_dep(1,8,s3y7):-mus(396).
ext_dep(1,8,s9y3):-mus(397).
ext_dep(1,8,s4y5):-mus(398).
ext_indep(1,8,s4y6):-mus(399).
ext_dep(1,8,s4y7):-mus(400).
ext_dep(1,8,s9y4):-mus(401).
ext_dep(1,8,s5y6):-mus(402).
ext_dep(1,8,s5y7):-mus(403).
ext_dep(1,8,s9y5):-mus(404).
ext_dep(1,8,s6y7):-mus(405).
ext_dep(1,8,s9y6):-mus(406).
ext_indep(1,8,s9y7):-mus(407).
ext_dep(1,8,s0y2y3):-mus(408).
ext_dep(1,8,s0y2y4):-mus(409).
ext_indep(1,8,s0y2y5):-mus(410).
ext_dep(1,8,s0y2y6):-mus(411).
ext_dep(1,8,s0y2y7):-mus(412).
ext_indep(1,8,s0y9y2):-mus(413).
ext_dep(1,8,s0y3y4):-mus(414).
ext_indep(1,8,s0y3y5):-mus(415).
ext_dep(1,8,s0y3y6):-mus(416).
ext_dep(1,8,s0y3y7):-mus(417).
ext_dep(1,8,s0y9y3):-mus(418).
ext_dep(1,8,s0y4y5):-mus(419).
ext_dep(1,8,s0y4y6):-mus(420).
ext_dep(1,8,s0y4y7):-mus(421).
ext_dep(1,8,s0y9y4):-mus(422).
ext_indep(1,8,s0y5y6):-mus(423).
ext_dep(1,8,s0y5y7):-mus(424).
ext_dep(1,8,s0y9y5):-mus(425).
ext_dep(1,8,s0y6y7):-mus(426).
ext_dep(1,8,s0y9y6):-mus(427).
ext_indep(1,8,s0y9y7):-mus(428).
ext_dep(1,8,s2y3y4):-mus(429).
ext_dep(1,8,s2y3y5):-mus(430).
ext_dep(1,8,s2y3y6):-mus(431).
ext_dep(1,8,s2y3y7):-mus(432).
ext_dep(1,8,s9y2y3):-mus(433).
ext_indep(1,8,s2y4y5):-mus(434).
ext_dep(1,8,s2y4y6):-mus(435).
ext_dep(1,8,s2y4y7):-mus(436).
ext_dep(1,8,s9y2y4):-mus(437).
ext_indep(1,8,s2y5y6):-mus(438).
ext_dep(1,8,s2y5y7):-mus(439).
ext_dep(1,8,s9y2y5):-mus(440).
ext_dep(1,8,s2y6y7):-mus(441).
ext_dep(1,8,s9y2y6):-mus(442).
ext_dep(1,8,s9y2y7):-mus(443).
ext_dep(1,8,s3y4y5):-mus(444).
ext_indep(1,8,s3y4y6):-mus(445).
ext_dep(1,8,s3y4y7):-mus(446).
ext_dep(1,8,s9y3y4):-mus(447).
ext_indep(1,8,s3y5y6):-mus(448).
ext_dep(1,8,s3y5y7):-mus(449).
ext_indep(1,8,s9y3y5):-mus(450).
ext_dep(1,8,s3y6y7):-mus(451).
ext_dep(1,8,s9y3y6):-mus(452).
ext_dep(1,8,s9y3y7):-mus(453).
ext_dep(1,8,s4y5y6):-mus(454).
ext_dep(1,8,s4y5y7):-mus(455).
ext_dep(1,8,s9y4y5):-mus(456).
ext_dep(1,8,s4y6y7):-mus(457).
ext_dep(1,8,s9y4y6):-mus(458).
ext_dep(1,8,s9y4y7):-mus(459).
ext_dep(1,8,s5y6y7):-mus(460).
ext_indep(1,8,s9y5y6):-mus(461).
ext_dep(1,8,s9y5y7):-mus(462).
ext_dep(1,8,s9y6y7):-mus(463).
ext_dep(1,8,s0y2y3y4):-mus(464).
ext_dep(1,8,s0y2y3y5):-mus(465).
ext_dep(1,8,s0y2y3y6):-mus(466).
ext_dep(1,8,s0y9y2y3):-mus(467).
ext_dep(1,8,s0y2y4y5):-mus(468).
ext_dep(1,8,s0y2y4y6):-mus(469).
ext_dep(1,8,s0y9y2y4):-mus(470).
ext_indep(1,8,s0y2y5y6):-mus(471).
ext_dep(1,8,s0y9y2y5):-mus(472).
ext_indep(1,8,s0y9y2y6):-mus(473).
ext_indep(1,8,s0y3y4y5):-mus(474).
ext_indep(1,8,s0y3y4y6):-mus(475).
ext_indep(1,8,s0y9y3y4):-mus(476).
ext_dep(1,8,s0y3y5y6):-mus(477).
ext_dep(1,8,s0y9y3y5):-mus(478).
ext_dep(1,8,s0y9y3y6):-mus(479).
ext_dep(1,8,s0y4y5y6):-mus(480).
ext_dep(1,8,s0y9y4y5):-mus(481).
ext_dep(1,8,s0y9y4y6):-mus(482).
ext_dep(1,8,s0y9y5y6):-mus(483).
ext_dep(1,8,s2y3y4y5):-mus(484).
ext_dep(1,8,s2y3y4y6):-mus(485).
ext_dep(1,8,s9y2y3y4):-mus(486).
ext_indep(1,8,s2y3y5y6):-mus(487).
ext_dep(1,8,s9y2y3y5):-mus(488).
ext_dep(1,8,s9y2y3y6):-mus(489).
ext_dep(1,8,s2y4y5y6):-mus(490).
ext_dep(1,8,s9y2y4y5):-mus(491).
ext_dep(1,8,s9y2y4y6):-mus(492).
ext_dep(1,8,s9y2y5y6):-mus(493).
ext_dep(1,8,s3y4y5y6):-mus(494).
ext_dep(1,8,s9y3y4y5):-mus(495).
ext_dep(1,8,s9y3y4y6):-mus(496).
ext_dep(1,8,s9y3y5y6):-mus(497).
ext_indep(1,8,s9y4y5y6):-mus(498).
ext_dep(1,8,s0y2y3y4y5):-mus(499).
ext_indep(1,8,s0y2y3y4y6):-mus(500).
ext_dep(1,8,s0y2y3y4y9):-mus(501).
ext_dep(1,8,s0y2y3y5y6):-mus(502).
ext_dep(1,8,s0y2y3y5y9):-mus(503).
ext_dep(1,8,s0y2y3y6y9):-mus(504).
ext_dep(1,8,s0y2y4y5y6):-mus(505).
ext_dep(1,8,s0y2y4y5y9):-mus(506).
ext_dep(1,8,s0y2y4y6y9):-mus(507).
ext_dep(1,8,s0y2y5y6y9):-mus(508).
ext_dep(1,8,s0y3y4y5y6):-mus(509).
ext_dep(1,8,s0y3y4y5y9):-mus(510).
ext_dep(1,8,s0y3y4y6y9):-mus(511).
ext_dep(1,8,s0y3y5y6y9):-mus(512).
ext_indep(1,8,s0y4y5y6y9):-mus(513).
ext_dep(1,8,s2y3y4y5y6):-mus(514).
ext_dep(1,8,s2y3y4y5y9):-mus(515).
ext_indep(1,8,s2y3y4y6y9):-mus(516).
ext_indep(1,8,s2y3y5y6y9):-mus(517).
ext_dep(1,8,s2y4y5y6y9):-mus(518).
ext_dep(1,8,s3y4y5y6y9):-mus(519).
ext_indep(1,8,s0y2y3y4y5y6):-mus(520).
ext_dep(1,8,s0y2y3y4y5y9):-mus(521).
ext_dep(1,8,s0y2y3y4y6y9):-mus(522).
ext_indep(1,8,s0y2y3y5y6y9):-mus(523).
ext_dep(1,8,s0y2y4y5y6y9):-mus(524).
ext_dep(1,8,s0y3y4y5y6y9):-mus(525).
ext_dep(1,8,s2y3y4y5y6y9):-mus(526).
ext_dep(1,8,s0y2y3y4y5y6y9):-mus(527).
ext_dep(1,9,empty):-mus(528).
ext_indep(1,9,s0):-mus(529).
ext_dep(1,9,s2):-mus(530).
ext_indep(1,9,s3):-mus(531).
ext_dep(1,9,s4):-mus(532).
ext_dep(1,9,s5):-mus(533).
ext_dep(1,9,s6):-mus(534).
ext_dep(1,9,s7):-mus(535).
ext_dep(1,9,s8):-mus(536).
ext_indep(1,9,s3y4):-mus(537).
ext_dep(1,9,s3y6):-mus(538).
ext_dep(1,9,s8y3):-mus(539).
ext_dep(1,9,s4y6):-mus(540).
ext_dep(1,9,s8y4):-mus(541).
ext_dep(1,9,s8y6):-mus(542).
ext_dep(1,9,s8y4y6):-mus(543).
ext_dep(2,3,s1):-mus(544).
ext_dep(2,3,s4):-mus(545).
ext_dep(2,3,s6):-mus(546).
ext_dep(2,3,s7):-mus(547).
ext_indep(2,4,s3):-mus(548).
ext_dep(2,4,s5):-mus(549).
ext_dep(2,4,s6):-mus(550).
ext_indep(2,4,s9):-mus(551).
ext_indep(2,4,s0y8):-mus(552).
ext_indep(2,4,s8y1):-mus(553).
ext_dep(2,5,s1):-mus(554).
ext_indep(2,5,s4):-mus(555).
ext_dep(2,5,s6):-mus(556).
ext_indep(2,5,s9):-mus(557).
ext_indep(2,5,s0y8):-mus(558).
ext_dep(2,5,s3y7):-mus(559).
ext_indep(2,5,s8y3):-mus(560).
ext_indep(2,5,s8y7):-mus(561).
ext_dep(2,5,s8y3y7):-mus(562).
ext_dep(2,6,s0):-mus(563).
ext_dep(2,6,s1):-mus(564).
ext_indep(2,6,s7):-mus(565).
ext_dep(2,6,s8):-mus(566).
ext_dep(2,6,s0y8):-mus(567).
ext_dep(2,6,s1y7):-mus(568).
ext_dep(2,6,s8y1):-mus(569).
ext_dep(2,6,s8y7):-mus(570).
ext_dep(2,6,s8y1y7):-mus(571).
ext_dep(2,7,s0):-mus(572).
ext_dep(2,7,s5):-mus(573).
ext_dep(2,7,s6):-mus(574).
ext_indep(2,7,s8):-mus(575).
ext_indep(2,7,s0y5):-mus(576).
ext_indep(2,7,s0y6):-mus(577).
ext_dep(2,7,s0y8):-mus(578).
ext_dep(2,7,s5y6):-mus(579).
ext_dep(2,7,s0y5y6):-mus(580).
ext_dep(2,8,empty):-mus(581).
ext_dep(2,8,s0):-mus(582).
ext_dep(2,8,s1):-mus(583).
ext_dep(2,8,s3):-mus(584).
ext_indep(2,8,s4):-mus(585).
ext_dep(2,8,s5):-mus(586).
ext_dep(2,8,s6):-mus(587).
ext_dep(2,8,s7):-mus(588).
ext_indep(2,8,s9):-mus(589).
ext_dep(2,8,s0y1):-mus(590).
ext_dep(2,8,s0y3):-mus(591).
ext_dep(2,8,s0y4):-mus(592).
ext_dep(2,8,s0y5):-mus(593).
ext_dep(2,8,s0y6):-mus(594).
ext_dep(2,8,s0y7):-mus(595).
ext_dep(2,8,s0y9):-mus(596).
ext_dep(2,8,s1y3):-mus(597).
ext_dep(2,8,s1y4):-mus(598).
ext_dep(2,8,s1y5):-mus(599).
ext_dep(2,8,s1y6):-mus(600).
ext_indep(2,8,s1y7):-mus(601).
ext_dep(2,8,s1y9):-mus(602).
ext_dep(2,8,s3y4):-mus(603).
ext_dep(2,8,s3y5):-mus(604).
ext_dep(2,8,s3y6):-mus(605).
ext_dep(2,8,s3y7):-mus(606).
ext_dep(2,8,s9y3):-mus(607).
ext_indep(2,8,s4y5):-mus(608).
ext_dep(2,8,s4y6):-mus(609).
ext_dep(2,8,s4y7):-mus(610).
ext_dep(2,8,s9y4):-mus(611).
ext_dep(2,8,s5y6):-mus(612).
ext_dep(2,8,s5y7):-mus(613).
ext_dep(2,8,s9y5):-mus(614).
ext_dep(2,8,s6y7):-mus(615).
ext_indep(2,8,s9y6):-mus(616).
ext_dep(2,8,s9y7):-mus(617).
ext_dep(2,8,s0y1y3):-mus(618).
ext_dep(2,8,s0y1y4):-mus(619).
ext_dep(2,8,s0y1y5):-mus(620).
ext_dep(2,8,s0y1y6):-mus(621).
ext_dep(2,8,s0y1y7):-mus(622).
ext_dep(2,8,s0y1y9):-mus(623).
ext_dep(2,8,s0y3y4):-mus(624).
ext_dep(2,8,s0y3y5):-mus(625).
ext_dep(2,8,s0y3y6):-mus(626).
ext_dep(2,8,s0y3y7):-mus(627).
ext_dep(2,8,s0y9y3):-mus(628).
ext_indep(2,8,s0y4y5):-mus(629).
ext_dep(2,8,s0y4y6):-mus(630).
ext_dep(2,8,s0y4y7):-mus(631).
ext_dep(2,8,s0y9y4):-mus(632).
ext_dep(2,8,s0y5y6):-mus(633).
ext_dep(2,8,s0y5y7):-mus(634).
ext_dep(2,8,s0y9y5):-mus(635).
ext_dep(2,8,s0y6y7):-mus(636).
ext_dep(2,8,s0y9y6):-mus(637).
ext_dep(2,8,s0y9y7):-mus(638).
ext_dep(2,8,s1y3y4):-mus(639).
ext_dep(2,8,s1y3y5):-mus(640).
ext_dep(2,8,s1y3y6):-mus(641).
ext_dep(2,8,s1y3y7):-mus(642).
ext_dep(2,8,s1y3y9):-mus(643).
ext_dep(2,8,s1y4y5):-mus(644).
ext_dep(2,8,s1y4y6):-mus(645).
ext_dep(2,8,s1y4y7):-mus(646).
ext_dep(2,8,s1y4y9):-mus(647).
ext_dep(2,8,s1y5y6):-mus(648).
ext_dep(2,8,s1y5y7):-mus(649).
ext_dep(2,8,s1y5y9):-mus(650).
ext_dep(2,8,s1y6y7):-mus(651).
ext_indep(2,8,s1y6y9):-mus(652).
ext_indep(2,8,s1y9y7):-mus(653).
ext_indep(2,8,s3y4y5):-mus(654).
ext_dep(2,8,s3y4y6):-mus(655).
ext_dep(2,8,s3y4y7):-mus(656).
ext_indep(2,8,s9y3y4):-mus(657).
ext_dep(2,8,s3y5y6):-mus(658).
ext_dep(2,8,s3y5y7):-mus(659).
ext_indep(2,8,s9y3y5):-mus(660).
ext_dep(2,8,s3y6y7):-mus(661).
ext_dep(2,8,s9y3y6):-mus(662).
ext_dep(2,8,s9y3y7):-mus(663).
ext_dep(2,8,s4y5y6):-mus(664).
ext_dep(2,8,s4y5y7):-mus(665).
ext_indep(2,8,s9y4y5):-mus(666).
ext_dep(2,8,s4y6y7):-mus(667).
ext_indep(2,8,s9y4y6):-mus(668).
ext_dep(2,8,s9y4y7):-mus(669).
ext_dep(2,8,s5y6y7):-mus(670).
ext_dep(2,8,s9y5y6):-mus(671).
ext_indep(2,8,s9y5y7):-mus(672).
ext_dep(2,8,s9y6y7):-mus(673).
ext_dep(2,8,s0y1y3y4):-mus(674).
ext_indep(2,8,s0y1y3y5):-mus(675).
ext_dep(2,8,s0y1y3y6):-mus(676).
ext_dep(2,8,s0y1y3y9):-mus(677).
ext_dep(2,8,s0y1y4y5):-mus(678).
ext_dep(2,8,s0y1y4y6):-mus(679).
ext_dep(2,8,s0y1y4y9):-mus(680).
ext_dep(2,8,s0y1y5y6):-mus(681).
ext_dep(2,8,s0y1y5y9):-mus(682).
ext_dep(2,8,s0y1y6y9):-mus(683).
ext_dep(2,8,s0y3y4y5):-mus(684).
ext_dep(2,8,s0y3y4y6):-mus(685).
ext_dep(2,8,s0y9y3y4):-mus(686).
ext_dep(2,8,s0y3y5y6):-mus(687).
ext_dep(2,8,s0y9y3y5):-mus(688).
ext_dep(2,8,s0y9y3y6):-mus(689).
ext_dep(2,8,s0y4y5y6):-mus(690).
ext_dep(2,8,s0y9y4y5):-mus(691).
ext_dep(2,8,s0y9y4y6):-mus(692).
ext_dep(2,8,s0y9y5y6):-mus(693).
ext_dep(2,8,s1y3y4y5):-mus(694).
ext_indep(2,8,s1y3y4y6):-mus(695).
ext_dep(2,8,s1y3y4y9):-mus(696).
ext_dep(2,8,s1y3y5y6):-mus(697).
ext_indep(2,8,s1y3y5y9):-mus(698).
ext_dep(2,8,s1y3y6y9):-mus(699).
ext_dep(2,8,s1y4y5y6):-mus(700).
ext_dep(2,8,s1y4y5y9):-mus(701).
ext_dep(2,8,s1y4y6y9):-mus(702).
ext_dep(2,8,s1y5y6y9):-mus(703).
ext_dep(2,8,s3y4y5y6):-mus(704).
ext_dep(2,8,s9y3y4y5):-mus(705).
ext_dep(2,8,s9y3y4y6):-mus(706).
ext_indep(2,8,s9y3y5y6):-mus(707).
ext_dep(2,8,s9y4y5y6):-mus(708).
ext_dep(2,8,s0y1y3y4y5):-mus(709).
ext_dep(2,8,s0y1y3y4y6):-mus(710).
ext_dep(2,8,s0y1y3y4y9):-mus(711).
ext_indep(2,8,s0y1y3y5y6):-mus(712).
ext_dep(2,8,s0y1y3y5y9):-mus(713).
ext_dep(2,8,s0y1y3y6y9):-mus(714).
ext_dep(2,8,s0y1y4y5y6):-mus(715).
ext_dep(2,8,s0y1y4y5y9):-mus(716).
ext_dep(2,8,s0y1y4y6y9):-mus(717).
ext_dep(2,8,s0y1y5y6y9):-mus(718).
ext_indep(2,8,s0y3y4y5y6):-mus(719).
ext_dep(2,8,s0y3y4y5y9):-mus(720).
ext_dep(2,8,s0y3y4y6y9):-mus(721).
ext_dep(2,8,s0y3y5y6y9):-mus(722).
ext_dep(2,8,s0y4y5y6y9):-mus(723).
ext_dep(2,8,s1y3y4y5y6):-mus(724).
ext_dep(2,8,s1y3y4y5y9):-mus(725).
ext_indep(2,8,s1y3y4y6y9):-mus(726).
ext_dep(2,8,s1y3y5y6y9):-mus(727).
ext_dep(2,8,s1y4y5y6y9):-mus(728).
ext_dep(2,8,s3y4y5y6y9):-mus(729).
ext_dep(2,8,s0y1y3y4y5y6):-mus(730).
ext_dep(2,8,s0y1y3y4y5y9):-mus(731).
ext_dep(2,8,s0y1y3y4y6y9):-mus(732).
ext_indep(2,8,s0y1y3y5y6y9):-mus(733).
ext_dep(2,8,s0y1y4y5y6y9):-mus(734).
ext_dep(2,8,s0y3y4y5y6y9):-mus(735).
ext_dep(2,8,s1y3y4y5y6y9):-mus(736).
ext_indep(2,8,s0y1y3y4y5y6y9):-mus(737).
ext_indep(2,9,s4):-mus(738).
ext_dep(2,9,s5):-mus(739).
ext_dep(2,9,s6):-mus(740).
ext_indep(2,9,s0y8):-mus(741).
ext_dep(2,9,s1y3):-mus(742).
ext_dep(2,9,s8y1):-mus(743).
ext_dep(2,9,s8y3):-mus(744).
ext_dep(2,9,s8y1y3):-mus(745).
ext_dep(3,4,s2):-mus(746).
ext_dep(3,4,s6):-mus(747).
ext_dep(3,4,s7):-mus(748).
ext_dep(3,4,s0y5):-mus(749).
ext_indep(3,4,s0y8):-mus(750).
ext_dep(3,4,s0y9):-mus(751).
ext_indep(3,4,s8y1):-mus(752).
ext_indep(3,4,s8y5):-mus(753).
ext_indep(3,4,s9y5):-mus(754).
ext_indep(3,4,s8y9):-mus(755).
ext_dep(3,4,s0y8y5):-mus(756).
ext_dep(3,4,s0y9y5):-mus(757).
ext_indep(3,4,s0y8y9):-mus(758).
ext_indep(3,4,s8y9y5):-mus(759).
ext_indep(3,4,s0y8y5y9):-mus(760).
ext_dep(3,5,empty):-mus(761).
ext_dep(3,5,s0):-mus(762).
ext_dep(3,5,s1):-mus(763).
ext_dep(3,5,s2):-mus(764).
ext_indep(3,5,s4):-mus(765).
ext_dep(3,5,s6):-mus(766).
ext_dep(3,5,s7):-mus(767).
ext_dep(3,5,s8):-mus(768).
ext_dep(3,5,s9):-mus(769).
ext_indep(3,5,s0y1):-mus(770).
ext_dep(3,5,s0y8):-mus(771).
ext_dep(3,5,s0y9):-mus(772).
ext_dep(3,5,s8y1):-mus(773).
ext_dep(3,5,s1y9):-mus(774).
ext_dep(3,5,s6y7):-mus(775).
ext_dep(3,5,s8y6):-mus(776).
ext_dep(3,5,s8y7):-mus(777).
ext_dep(3,5,s8y9):-mus(778).
ext_indep(3,5,s0y8y9):-mus(779).
ext_dep(3,6,s2):-mus(780).
ext_dep(3,6,s4):-mus(781).
ext_dep(3,6,s0y5):-mus(782).
ext_dep(3,6,s0y8):-mus(783).
ext_dep(3,6,s0y9):-mus(784).
ext_dep(3,6,s1y7):-mus(785).
ext_indep(3,6,s8y1):-mus(786).
ext_indep(3,6,s8y5):-mus(787).
ext_dep(3,6,s9y5):-mus(788).
ext_indep(3,6,s8y7):-mus(789).
ext_dep(3,6,s8y9):-mus(790).
ext_indep(3,6,s0y8y5):-mus(791).
ext_dep(3,6,s0y9y5):-mus(792).
ext_dep(3,6,s0y8y9):-mus(793).
ext_indep(3,6,s8y1y7):-mus(794).
ext_dep(3,6,s8y9y5):-mus(795).
ext_indep(3,6,s0y8y5y9):-mus(796).
ext_dep(3,7,s1):-mus(797).
ext_dep(3,7,s2):-mus(798).
ext_dep(3,7,s4):-mus(799).
ext_indep(3,8,empty):-mus(800).
ext_dep(3,8,s0):-mus(801).
ext_indep(3,8,s1):-mus(802).
ext_dep(3,8,s2):-mus(803).
ext_dep(3,8,s4):-mus(804).
ext_dep(3,8,s5):-mus(805).
ext_dep(3,8,s6):-mus(806).
ext_dep(3,8,s7):-mus(807).
ext_dep(3,8,s9):-mus(808).
ext_dep(3,8,s0y1):-mus(809).
ext_dep(3,8,s0y2):-mus(810).
ext_dep(3,8,s0y4):-mus(811).
ext_dep(3,8,s0y5):-mus(812).
ext_dep(3,8,s0y6):-mus(813).
ext_dep(3,8,s0y7):-mus(814).
ext_dep(3,8,s0y9):-mus(815).
ext_dep(3,8,s1y2):-mus(816).
ext_dep(3,8,s1y4):-mus(817).
ext_dep(3,8,s1y5):-mus(818).
ext_dep(3,8,s1y6):-mus(819).
ext_indep(3,8,s1y7):-mus(820).
ext_dep(3,8,s1y9):-mus(821).
ext_dep(3,8,s2y4):-mus(822).
ext_indep(3,8,s2y5):-mus(823).
ext_dep(3,8,s2y6):-mus(824).
ext_dep(3,8,s2y7):-mus(825).
ext_dep(3,8,s9y2):-mus(826).
ext_indep(3,8,s4y5):-mus(827).
ext_dep(3,8,s4y6):-mus(828).
ext_dep(3,8,s4y7):-mus(829).
ext_dep(3,8,s9y4):-mus(830).
ext_indep(3,8,s5y6):-mus(831).
ext_dep(3,8,s5y7):-mus(832).
ext_dep(3,8,s9y5):-mus(833).
ext_dep(3,8,s6y7):-mus(834).
ext_dep(3,8,s9y6):-mus(835).
ext_dep(3,8,s9y7):-mus(836).
ext_dep(3,8,s0y1y2):-mus(837).
ext_dep(3,8,s0y1y4):-mus(838).
ext_dep(3,8,s0y1y5):-mus(839).
ext_dep(3,8,s0y1y6):-mus(840).
ext_dep(3,8,s0y1y7):-mus(841).
ext_dep(3,8,s0y1y9):-mus(842).
ext_dep(3,8,s0y2y4):-mus(843).
ext_dep(3,8,s0y2y5):-mus(844).
ext_indep(3,8,s0y2y6):-mus(845).
ext_indep(3,8,s0y2y7):-mus(846).
ext_dep(3,8,s0y9y2):-mus(847).
ext_dep(3,8,s0y4y5):-mus(848).
ext_dep(3,8,s0y4y6):-mus(849).
ext_dep(3,8,s0y4y7):-mus(850).
ext_indep(3,8,s0y9y4):-mus(851).
ext_dep(3,8,s0y5y6):-mus(852).
ext_dep(3,8,s0y5y7):-mus(853).
ext_dep(3,8,s0y9y5):-mus(854).
ext_dep(3,8,s0y6y7):-mus(855).
ext_dep(3,8,s0y9y6):-mus(856).
ext_indep(3,8,s0y9y7):-mus(857).
ext_dep(3,8,s1y2y4):-mus(858).
ext_dep(3,8,s1y2y5):-mus(859).
ext_dep(3,8,s1y2y6):-mus(860).
ext_dep(3,8,s1y2y7):-mus(861).
ext_dep(3,8,s1y2y9):-mus(862).
ext_dep(3,8,s1y4y5):-mus(863).
ext_dep(3,8,s1y4y6):-mus(864).
ext_dep(3,8,s1y4y7):-mus(865).
ext_dep(3,8,s1y4y9):-mus(866).
ext_dep(3,8,s1y5y6):-mus(867).
ext_dep(3,8,s1y5y7):-mus(868).
ext_indep(3,8,s1y5y9):-mus(869).
ext_indep(3,8,s1y6y7):-mus(870).
ext_dep(3,8,s1y6y9):-mus(871).
ext_dep(3,8,s1y9y7):-mus(872).
ext_dep(3,8,s2y4y5):-mus(873).
ext_dep(3,8,s2y4y6):-mus(874).
ext_dep(3,8,s2y4y7):-mus(875).
ext_dep(3,8,s9y2y4):-mus(876).
ext_dep(3,8,s2y5y6):-mus(877).
ext_indep(3,8,s2y5y7):-mus(878).
ext_dep(3,8,s9y2y5):-mus(879).
ext_dep(3,8,s2y6y7):-mus(880).
ext_indep(3,8,s9y2y6):-mus(881).
ext_dep(3,8,s9y2y7):-mus(882).
ext_indep(3,8,s4y5y6):-mus(883).
ext_dep(3,8,s4y5y7):-mus(884).
ext_dep(3,8,s9y4y5):-mus(885).
ext_dep(3,8,s4y6y7):-mus(886).
ext_indep(3,8,s9y4y6):-mus(887).
ext_indep(3,8,s9y4y7):-mus(888).
ext_dep(3,8,s5y6y7):-mus(889).
ext_indep(3,8,s9y5y6):-mus(890).
ext_dep(3,8,s9y5y7):-mus(891).
ext_indep(3,8,s9y6y7):-mus(892).
ext_dep(3,8,s0y1y2y4):-mus(893).
ext_indep(3,8,s0y1y2y5):-mus(894).
ext_dep(3,8,s0y1y2y6):-mus(895).
ext_dep(3,8,s0y1y2y9):-mus(896).
ext_dep(3,8,s0y1y4y5):-mus(897).
ext_dep(3,8,s0y1y4y6):-mus(898).
ext_indep(3,8,s0y1y4y9):-mus(899).
ext_dep(3,8,s0y1y5y6):-mus(900).
ext_dep(3,8,s0y1y5y9):-mus(901).
ext_dep(3,8,s0y1y6y9):-mus(902).
ext_indep(3,8,s0y2y4y5):-mus(903).
ext_indep(3,8,s0y2y4y6):-mus(904).
ext_indep(3,8,s0y9y2y4):-mus(905).
ext_dep(3,8,s0y2y5y6):-mus(906).
ext_dep(3,8,s0y9y2y5):-mus(907).
ext_dep(3,8,s0y9y2y6):-mus(908).
ext_dep(3,8,s0y4y5y6):-mus(909).
ext_dep(3,8,s0y9y4y5):-mus(910).
ext_dep(3,8,s0y9y4y6):-mus(911).
ext_dep(3,8,s0y9y5y6):-mus(912).
ext_dep(3,8,s1y2y4y5):-mus(913).
ext_dep(3,8,s1y2y4y6):-mus(914).
ext_dep(3,8,s1y2y4y9):-mus(915).
ext_dep(3,8,s1y2y5y6):-mus(916).
ext_dep(3,8,s1y2y5y9):-mus(917).
ext_dep(3,8,s1y2y6y9):-mus(918).
ext_indep(3,8,s1y4y5y6):-mus(919).
ext_indep(3,8,s1y4y5y9):-mus(920).
ext_dep(3,8,s1y4y6y9):-mus(921).
ext_dep(3,8,s1y5y6y9):-mus(922).
ext_dep(3,8,s2y4y5y6):-mus(923).
ext_dep(3,8,s9y2y4y5):-mus(924).
ext_dep(3,8,s9y2y4y6):-mus(925).
ext_indep(3,8,s9y2y5y6):-mus(926).
ext_dep(3,8,s9y4y5y6):-mus(927).
ext_dep(3,8,s0y1y2y4y5):-mus(928).
ext_dep(3,8,s0y1y2y4y6):-mus(929).
ext_dep(3,8,s0y1y2y4y9):-mus(930).
ext_indep(3,8,s0y1y2y5y6):-mus(931).
ext_dep(3,8,s0y1y2y5y9):-mus(932).
ext_indep(3,8,s0y1y2y6y9):-mus(933).
ext_dep(3,8,s0y1y4y5y6):-mus(934).
ext_indep(3,8,s0y1y4y5y9):-mus(935).
ext_dep(3,8,s0y1y4y6y9):-mus(936).
ext_dep(3,8,s0y1y5y6y9):-mus(937).
ext_dep(3,8,s0y2y4y5y6):-mus(938).
ext_dep(3,8,s0y2y4y5y9):-mus(939).
ext_dep(3,8,s0y2y4y6y9):-mus(940).
ext_dep(3,8,s0y2y5y6y9):-mus(941).
ext_dep(3,8,s0y4y5y6y9):-mus(942).
ext_dep(3,8,s1y2y4y5y6):-mus(943).
ext_dep(3,8,s1y2y4y5y9):-mus(944).
ext_dep(3,8,s1y2y4y6y9):-mus(945).
ext_dep(3,8,s1y2y5y6y9):-mus(946).
ext_dep(3,8,s1y4y5y6y9):-mus(947).
ext_dep(3,8,s2y4y5y6y9):-mus(948).
ext_dep(3,8,s0y1y2y4y5y6):-mus(949).
ext_dep(3,8,s0y1y2y4y5y9):-mus(950).
ext_dep(3,8,s0y1y2y4y6y9):-mus(951).
ext_dep(3,8,s0y1y2y5y6y9):-mus(952).
ext_dep(3,8,s0y1y4y5y6y9):-mus(953).
ext_dep(3,8,s0y2y4y5y6y9):-mus(954).
ext_dep(3,8,s1y2y4y5y6y9):-mus(955).
ext_dep(3,8,s0y1y2y4y5y6y9):-mus(956).
ext_dep(3,9,empty):-mus(957).
ext_dep(3,9,s0):-mus(958).
ext_dep(3,9,s1):-mus(959).
ext_dep(3,9,s2):-mus(960).
ext_indep(3,9,s4):-mus(961).
ext_dep(3,9,s5):-mus(962).
ext_dep(3,9,s6):-mus(963).
ext_dep(3,9,s7):-mus(964).
ext_dep(3,9,s8):-mus(965).
ext_indep(3,9,s0y1):-mus(966).
ext_dep(3,9,s0y5):-mus(967).
ext_dep(3,9,s0y8):-mus(968).
ext_dep(3,9,s1y4):-mus(969).
ext_dep(3,9,s1y5):-mus(970).
ext_indep(3,9,s8y1):-mus(971).
ext_dep(3,9,s8y4):-mus(972).
ext_indep(3,9,s8y5):-mus(973).
ext_dep(3,9,s0y8y5):-mus(974).
ext_dep(4,5,s1):-mus(975).
ext_dep(4,5,s3):-mus(976).
ext_indep(4,5,s7):-mus(977).
ext_indep(4,5,s8):-mus(978).
ext_indep(4,5,s8y1):-mus(979).
ext_dep(4,5,s3y7):-mus(980).
ext_indep(4,5,s8y3):-mus(981).
ext_indep(4,5,s8y7):-mus(982).
ext_indep(4,5,s8y3y7):-mus(983).
ext_dep(4,6,s0):-mus(984).
ext_dep(4,6,s2):-mus(985).
ext_dep(4,6,s3):-mus(986).
ext_dep(4,6,s5):-mus(987).
ext_dep(4,6,s9):-mus(988).
ext_dep(4,6,s0y1):-mus(989).
ext_dep(4,6,s0y5):-mus(990).
ext_dep(4,6,s0y7):-mus(991).
ext_indep(4,6,s0y8):-mus(992).
ext_dep(4,6,s1y5):-mus(993).
ext_indep(4,6,s1y9):-mus(994).
ext_dep(4,6,s5y7):-mus(995).
ext_indep(4,6,s8y5):-mus(996).
ext_dep(4,6,s8y9):-mus(997).
ext_indep(4,7,empty):-mus(998).
ext_dep(4,8,empty):-mus(999).
ext_indep(4,8,s0):-mus(1000).
ext_dep(4,8,s1):-mus(1001).
ext_dep(4,8,s2):-mus(1002).
ext_dep(4,8,s3):-mus(1003).
ext_dep(4,8,s5):-mus(1004).
ext_dep(4,8,s6):-mus(1005).
ext_dep(4,8,s7):-mus(1006).
ext_dep(4,8,s9):-mus(1007).
ext_dep(4,8,s0y1):-mus(1008).
ext_dep(4,8,s0y2):-mus(1009).
ext_dep(4,8,s0y3):-mus(1010).
ext_dep(4,8,s0y5):-mus(1011).
ext_dep(4,8,s0y6):-mus(1012).
ext_dep(4,8,s0y7):-mus(1013).
ext_dep(4,8,s0y9):-mus(1014).
ext_indep(4,8,s1y2):-mus(1015).
ext_dep(4,8,s1y3):-mus(1016).
ext_dep(4,8,s1y5):-mus(1017).
ext_indep(4,8,s1y6):-mus(1018).
ext_dep(4,8,s1y7):-mus(1019).
ext_dep(4,8,s1y9):-mus(1020).
ext_dep(4,8,s2y3):-mus(1021).
ext_dep(4,8,s2y5):-mus(1022).
ext_dep(4,8,s2y6):-mus(1023).
ext_dep(4,8,s2y7):-mus(1024).
ext_dep(4,8,s9y2):-mus(1025).
ext_dep(4,8,s3y5):-mus(1026).
ext_indep(4,8,s3y6):-mus(1027).
ext_dep(4,8,s3y7):-mus(1028).
ext_dep(4,8,s9y3):-mus(1029).
ext_dep(4,8,s5y6):-mus(1030).
ext_dep(4,8,s5y7):-mus(1031).
ext_indep(4,8,s9y5):-mus(1032).
ext_dep(4,8,s6y7):-mus(1033).
ext_dep(4,8,s9y6):-mus(1034).
ext_indep(4,8,s9y7):-mus(1035).
ext_dep(4,8,s0y1y2):-mus(1036).
ext_dep(4,8,s0y1y3):-mus(1037).
ext_dep(4,8,s0y1y5):-mus(1038).
ext_dep(4,8,s0y1y6):-mus(1039).
ext_dep(4,8,s0y1y7):-mus(1040).
ext_dep(4,8,s0y1y9):-mus(1041).
ext_dep(4,8,s0y2y3):-mus(1042).
ext_indep(4,8,s0y2y5):-mus(1043).
ext_dep(4,8,s0y2y6):-mus(1044).
ext_dep(4,8,s0y2y7):-mus(1045).
ext_dep(4,8,s0y9y2):-mus(1046).
ext_dep(4,8,s0y3y5):-mus(1047).
ext_dep(4,8,s0y3y6):-mus(1048).
ext_dep(4,8,s0y3y7):-mus(1049).
ext_indep(4,8,s0y9y3):-mus(1050).
ext_dep(4,8,s0y5y6):-mus(1051).
ext_dep(4,8,s0y5y7):-mus(1052).
ext_dep(4,8,s0y9y5):-mus(1053).
ext_dep(4,8,s0y6y7):-mus(1054).
ext_dep(4,8,s0y9y6):-mus(1055).
ext_dep(4,8,s0y9y7):-mus(1056).
ext_dep(4,8,s1y2y3):-mus(1057).
ext_dep(4,8,s1y2y5):-mus(1058).
ext_dep(4,8,s1y2y6):-mus(1059).
ext_dep(4,8,s1y2y7):-mus(1060).
ext_dep(4,8,s1y2y9):-mus(1061).
ext_dep(4,8,s1y3y5):-mus(1062).
ext_dep(4,8,s1y3y6):-mus(1063).
ext_indep(4,8,s1y3y7):-mus(1064).
ext_dep(4,8,s1y3y9):-mus(1065).
ext_dep(4,8,s1y5y6):-mus(1066).
ext_dep(4,8,s1y5y7):-mus(1067).
ext_indep(4,8,s1y5y9):-mus(1068).
ext_dep(4,8,s1y6y7):-mus(1069).
ext_dep(4,8,s1y6y9):-mus(1070).
ext_indep(4,8,s1y9y7):-mus(1071).
ext_dep(4,8,s2y3y5):-mus(1072).
ext_dep(4,8,s2y3y6):-mus(1073).
ext_dep(4,8,s2y3y7):-mus(1074).
ext_dep(4,8,s9y2y3):-mus(1075).
ext_dep(4,8,s2y5y6):-mus(1076).
ext_dep(4,8,s2y5y7):-mus(1077).
ext_dep(4,8,s9y2y5):-mus(1078).
ext_dep(4,8,s2y6y7):-mus(1079).
ext_dep(4,8,s9y2y6):-mus(1080).
ext_dep(4,8,s9y2y7):-mus(1081).
ext_indep(4,8,s3y5y6):-mus(1082).
ext_dep(4,8,s3y5y7):-mus(1083).
ext_dep(4,8,s9y3y5):-mus(1084).
ext_dep(4,8,s3y6y7):-mus(1085).
ext_dep(4,8,s9y3y6):-mus(1086).
ext_dep(4,8,s9y3y7):-mus(1087).
ext_dep(4,8,s5y6y7):-mus(1088).
ext_indep(4,8,s9y5y6):-mus(1089).
ext_dep(4,8,s9y5y7):-mus(1090).
ext_indep(4,8,s9y6y7):-mus(1091).
ext_dep(4,8,s0y1y2y3):-mus(1092).
ext_dep(4,8,s0y1y2y5):-mus(1093).
ext_indep(4,8,s0y1y2y6):-mus(1094).
ext_dep(4,8,s0y1y2y9):-mus(1095).
ext_dep(4,8,s0y1y3y5):-mus(1096).
ext_dep(4,8,s0y1y3y6):-mus(1097).
ext_dep(4,8,s0y1y3y9):-mus(1098).
ext_dep(4,8,s0y1y5y6):-mus(1099).
ext_dep(4,8,s0y1y5y9):-mus(1100).
ext_indep(4,8,s0y1y6y9):-mus(1101).
ext_dep(4,8,s0y2y3y5):-mus(1102).
ext_dep(4,8,s0y2y3y6):-mus(1103).
ext_dep(4,8,s0y9y2y3):-mus(1104).
ext_dep(4,8,s0y2y5y6):-mus(1105).
ext_dep(4,8,s0y9y2y5):-mus(1106).
ext_dep(4,8,s0y9y2y6):-mus(1107).
ext_dep(4,8,s0y3y5y6):-mus(1108).
ext_dep(4,8,s0y9y3y5):-mus(1109).
ext_dep(4,8,s0y9y3y6):-mus(1110).
ext_dep(4,8,s0y9y5y6):-mus(1111).
ext_dep(4,8,s1y2y3y5):-mus(1112).
ext_dep(4,8,s1y2y3y6):-mus(1113).
ext_dep(4,8,s1y2y3y9):-mus(1114).
ext_dep(4,8,s1y2y5y6):-mus(1115).
ext_dep(4,8,s1y2y5y9):-mus(1116).
ext_dep(4,8,s1y2y6y9):-mus(1117).
ext_dep(4,8,s1y3y5y6):-mus(1118).
ext_indep(4,8,s1y3y5y9):-mus(1119).
ext_dep(4,8,s1y3y6y9):-mus(1120).
ext_dep(4,8,s1y5y6y9):-mus(1121).
ext_dep(4,8,s2y3y5y6):-mus(1122).
ext_dep(4,8,s9y2y3y5):-mus(1123).
ext_dep(4,8,s9y2y3y6):-mus(1124).
ext_dep(4,8,s9y2y5y6):-mus(1125).
ext_dep(4,8,s9y3y5y6):-mus(1126).
ext_dep(4,8,s0y1y2y3y5):-mus(1127).
ext_indep(4,8,s0y1y2y3y6):-mus(1128).
ext_indep(4,8,s0y1y2y3y9):-mus(1129).
ext_dep(4,8,s0y1y2y5y6):-mus(1130).
ext_dep(4,8,s0y1y2y5y9):-mus(1131).
ext_dep(4,8,s0y1y2y6y9):-mus(1132).
ext_dep(4,8,s0y1y3y5y6):-mus(1133).
ext_indep(4,8,s0y1y3y5y9):-mus(1134).
ext_dep(4,8,s0y1y3y6y9):-mus(1135).
ext_dep(4,8,s0y1y5y6y9):-mus(1136).
ext_dep(4,8,s0y2y3y5y6):-mus(1137).
ext_dep(4,8,s0y2y3y5y9):-mus(1138).
ext_dep(4,8,s0y2y3y6y9):-mus(1139).
ext_dep(4,8,s0y2y5y6y9):-mus(1140).
ext_dep(4,8,s0y3y5y6y9):-mus(1141).
ext_dep(4,8,s1y2y3y5y6):-mus(1142).
ext_dep(4,8,s1y2y3y5y9):-mus(1143).
ext_dep(4,8,s1y2y3y6y9):-mus(1144).
ext_dep(4,8,s1y2y5y6y9):-mus(1145).
ext_dep(4,8,s1y3y5y6y9):-mus(1146).
ext_indep(4,8,s2y3y5y6y9):-mus(1147).
ext_indep(4,8,s0y1y2y3y5y6):-mus(1148).
ext_dep(4,8,s0y1y2y3y5y9):-mus(1149).
ext_dep(4,8,s0y1y2y3y6y9):-mus(1150).
ext_dep(4,8,s0y1y2y5y6y9):-mus(1151).
ext_dep(4,8,s0y1y3y5y6y9):-mus(1152).
ext_dep(4,8,s0y2y3y5y6y9):-mus(1153).
ext_dep(4,8,s1y2y3y5y6y9):-mus(1154).
ext_dep(4,8,s0y1y2y3y5y6y9):-mus(1155).
ext_dep(4,9,s0):-mus(1156).
ext_indep(4,9,s2):-mus(1157).
ext_dep(4,9,s5):-mus(1158).
ext_dep(4,9,s6):-mus(1159).
ext_dep(4,9,s1y6):-mus(1160).
ext_dep(4,9,s8y6):-mus(1161).
ext_indep(5,6,s0):-mus(1162).
ext_dep(5,6,s2):-mus(1163).
ext_dep(5,6,s4):-mus(1164).
ext_indep(5,6,s9):-mus(1165).
ext_dep(5,6,s0y1):-mus(1166).
ext_indep(5,6,s0y4):-mus(1167).
ext_dep(5,6,s0y7):-mus(1168).
ext_dep(5,6,s0y8):-mus(1169).
ext_indep(5,6,s1y4):-mus(1170).
ext_dep(5,6,s4y7):-mus(1171).
ext_dep(5,6,s8y4):-mus(1172).
ext_indep(5,7,empty):-mus(1173).
ext_dep(5,7,s0):-mus(1174).
ext_dep(5,7,s1):-mus(1175).
ext_dep(5,7,s2):-mus(1176).
ext_dep(5,7,s3):-mus(1177).
ext_dep(5,7,s6):-mus(1178).
ext_dep(5,7,s8):-mus(1179).
ext_dep(5,7,s9):-mus(1180).
ext_dep(5,7,s0y6):-mus(1181).
ext_dep(5,7,s0y8):-mus(1182).
ext_dep(5,7,s3y6):-mus(1183).
ext_dep(5,7,s8y3):-mus(1184).
ext_dep(5,7,s8y6):-mus(1185).
ext_dep(5,7,s0y8y6):-mus(1186).
ext_dep(5,8,empty):-mus(1187).
ext_dep(5,8,s0):-mus(1188).
ext_dep(5,8,s1):-mus(1189).
ext_dep(5,8,s2):-mus(1190).
ext_dep(5,8,s3):-mus(1191).
ext_dep(5,8,s4):-mus(1192).
ext_dep(5,8,s6):-mus(1193).
ext_dep(5,8,s7):-mus(1194).
ext_dep(5,8,s9):-mus(1195).
ext_dep(5,8,s0y1):-mus(1196).
ext_dep(5,8,s0y2):-mus(1197).
ext_dep(5,8,s0y3):-mus(1198).
ext_dep(5,8,s0y4):-mus(1199).
ext_dep(5,8,s0y6):-mus(1200).
ext_indep(5,8,s0y7):-mus(1201).
ext_dep(5,8,s0y9):-mus(1202).
ext_dep(5,8,s1y2):-mus(1203).
ext_indep(5,8,s1y3):-mus(1204).
ext_dep(5,8,s1y4):-mus(1205).
ext_dep(5,8,s1y6):-mus(1206).
ext_dep(5,8,s1y7):-mus(1207).
ext_dep(5,8,s1y9):-mus(1208).
ext_dep(5,8,s2y3):-mus(1209).
ext_dep(5,8,s2y4):-mus(1210).
ext_dep(5,8,s2y6):-mus(1211).
ext_dep(5,8,s2y7):-mus(1212).
ext_dep(5,8,s9y2):-mus(1213).
ext_dep(5,8,s3y4):-mus(1214).
ext_dep(5,8,s3y6):-mus(1215).
ext_dep(5,8,s3y7):-mus(1216).
ext_dep(5,8,s9y3):-mus(1217).
ext_dep(5,8,s4y6):-mus(1218).
ext_dep(5,8,s4y7):-mus(1219).
ext_dep(5,8,s9y4):-mus(1220).
ext_dep(5,8,s6y7):-mus(1221).
ext_dep(5,8,s9y6):-mus(1222).
ext_dep(5,8,s9y7):-mus(1223).
ext_indep(5,8,s0y1y2):-mus(1224).
ext_dep(5,8,s0y1y3):-mus(1225).
ext_dep(5,8,s0y1y4):-mus(1226).
ext_dep(5,8,s0y1y6):-mus(1227).
ext_dep(5,8,s0y1y7):-mus(1228).
ext_indep(5,8,s0y1y9):-mus(1229).
ext_dep(5,8,s0y2y3):-mus(1230).
ext_indep(5,8,s0y2y4):-mus(1231).
ext_dep(5,8,s0y2y6):-mus(1232).
ext_indep(5,8,s0y2y7):-mus(1233).
ext_dep(5,8,s0y9y2):-mus(1234).
ext_indep(5,8,s0y3y4):-mus(1235).
ext_dep(5,8,s0y3y6):-mus(1236).
ext_dep(5,8,s0y3y7):-mus(1237).
ext_indep(5,8,s0y9y3):-mus(1238).
ext_dep(5,8,s0y4y6):-mus(1239).
ext_dep(5,8,s0y4y7):-mus(1240).
ext_dep(5,8,s0y9y4):-mus(1241).
ext_dep(5,8,s0y6y7):-mus(1242).
ext_indep(5,8,s0y9y6):-mus(1243).
ext_dep(5,8,s0y9y7):-mus(1244).
ext_indep(5,8,s1y2y3):-mus(1245).
ext_dep(5,8,s1y2y4):-mus(1246).
ext_dep(5,8,s1y2y6):-mus(1247).
ext_indep(5,8,s1y2y7):-mus(1248).
ext_indep(5,8,s1y2y9):-mus(1249).
ext_dep(5,8,s1y3y4):-mus(1250).
ext_dep(5,8,s1y3y6):-mus(1251).
ext_indep(5,8,s1y3y7):-mus(1252).
ext_dep(5,8,s1y3y9):-mus(1253).
ext_dep(5,8,s1y4y6):-mus(1254).
ext_dep(5,8,s1y4y7):-mus(1255).
ext_dep(5,8,s1y4y9):-mus(1256).
ext_dep(5,8,s1y6y7):-mus(1257).
ext_dep(5,8,s1y6y9):-mus(1258).
ext_indep(5,8,s1y9y7):-mus(1259).
ext_dep(5,8,s2y3y4):-mus(1260).
ext_dep(5,8,s2y3y6):-mus(1261).
ext_dep(5,8,s2y3y7):-mus(1262).
ext_dep(5,8,s9y2y3):-mus(1263).
ext_dep(5,8,s2y4y6):-mus(1264).
ext_dep(5,8,s2y4y7):-mus(1265).
ext_dep(5,8,s9y2y4):-mus(1266).
ext_dep(5,8,s2y6y7):-mus(1267).
ext_dep(5,8,s9y2y6):-mus(1268).
ext_dep(5,8,s9y2y7):-mus(1269).
ext_dep(5,8,s3y4y6):-mus(1270).
ext_dep(5,8,s3y4y7):-mus(1271).
ext_dep(5,8,s9y3y4):-mus(1272).
ext_dep(5,8,s3y6y7):-mus(1273).
ext_dep(5,8,s9y3y6):-mus(1274).
ext_dep(5,8,s9y3y7):-mus(1275).
ext_dep(5,8,s4y6y7):-mus(1276).
ext_dep(5,8,s9y4y6):-mus(1277).
ext_dep(5,8,s9y4y7):-mus(1278).
ext_dep(5,8,s9y6y7):-mus(1279).
ext_dep(5,8,s0y1y2y3):-mus(1280).
ext_dep(5,8,s0y1y2y4):-mus(1281).
ext_dep(5,8,s0y1y2y6):-mus(1282).
ext_dep(5,8,s0y1y2y9):-mus(1283).
ext_dep(5,8,s0y1y3y4):-mus(1284).
ext_dep(5,8,s0y1y3y6):-mus(1285).
ext_dep(5,8,s0y1y3y9):-mus(1286).
ext_dep(5,8,s0y1y4y6):-mus(1287).
ext_dep(5,8,s0y1y4y9):-mus(1288).
ext_indep(5,8,s0y1y6y9):-mus(1289).
ext_dep(5,8,s0y2y3y4):-mus(1290).
ext_dep(5,8,s0y2y3y6):-mus(1291).
ext_dep(5,8,s0y9y2y3):-mus(1292).
ext_dep(5,8,s0y2y4y6):-mus(1293).
ext_dep(5,8,s0y9y2y4):-mus(1294).
ext_dep(5,8,s0y9y2y6):-mus(1295).
ext_dep(5,8,s0y3y4y6):-mus(1296).
ext_dep(5,8,s0y9y3y4):-mus(1297).
ext_dep(5,8,s0y9y3y6):-mus(1298).
ext_dep(5,8,s0y9y4y6):-mus(1299).
ext_indep(5,8,s1y2y3y4):-mus(1300).
ext_dep(5,8,s1y2y3y6):-mus(1301).
ext_dep(5,8,s1y2y3y9):-mus(1302).
ext_dep(5,8,s1y2y4y6):-mus(1303).
ext_dep(5,8,s1y2y4y9):-mus(1304).
ext_dep(5,8,s1y2y6y9):-mus(1305).
ext_indep(5,8,s1y3y4y6):-mus(1306).
ext_dep(5,8,s1y3y4y9):-mus(1307).
ext_dep(5,8,s1y3y6y9):-mus(1308).
ext_dep(5,8,s1y4y6y9):-mus(1309).
ext_dep(5,8,s2y3y4y6):-mus(1310).
ext_dep(5,8,s9y2y3y4):-mus(1311).
ext_indep(5,8,s9y2y3y6):-mus(1312).
ext_indep(5,8,s9y2y4y6):-mus(1313).
ext_dep(5,8,s9y3y4y6):-mus(1314).
ext_dep(5,8,s0y1y2y3y4):-mus(1315).
ext_indep(5,8,s0y1y2y3y6):-mus(1316).
ext_dep(5,8,s0y1y2y3y9):-mus(1317).
ext_dep(5,8,s0y1y2y4y6):-mus(1318).
ext_dep(5,8,s0y1y2y4y9):-mus(1319).
ext_dep(5,8,s0y1y2y6y9):-mus(1320).
ext_dep(5,8,s0y1y3y4y6):-mus(1321).
ext_indep(5,8,s0y1y3y4y9):-mus(1322).
ext_dep(5,8,s0y1y3y6y9):-mus(1323).
ext_dep(5,8,s0y1y4y6y9):-mus(1324).
ext_indep(5,8,s0y2y3y4y6):-mus(1325).
ext_dep(5,8,s0y2y3y4y9):-mus(1326).
ext_dep(5,8,s0y2y3y6y9):-mus(1327).
ext_dep(5,8,s0y2y4y6y9):-mus(1328).
ext_dep(5,8,s0y3y4y6y9):-mus(1329).
ext_dep(5,8,s1y2y3y4y6):-mus(1330).
ext_dep(5,8,s1y2y3y4y9):-mus(1331).
ext_dep(5,8,s1y2y3y6y9):-mus(1332).
ext_dep(5,8,s1y2y4y6y9):-mus(1333).
ext_dep(5,8,s1y3y4y6y9):-mus(1334).
ext_dep(5,8,s2y3y4y6y9):-mus(1335).
ext_dep(5,8,s0y1y2y3y4y6):-mus(1336).
ext_dep(5,8,s0y1y2y3y4y9):-mus(1337).
ext_dep(5,8,s0y1y2y3y6y9):-mus(1338).
ext_dep(5,8,s0y1y2y4y6y9):-mus(1339).
ext_dep(5,8,s0y1y3y4y6y9):-mus(1340).
ext_dep(5,8,s0y2y3y4y6y9):-mus(1341).
ext_dep(5,8,s1y2y3y4y6y9):-mus(1342).
ext_dep(5,8,s0y1y2y3y4y6y9):-mus(1343).
ext_dep(5,9,s0):-mus(1344).
ext_dep(5,9,s2):-mus(1345).
ext_dep(5,9,s4):-mus(1346).
ext_dep(5,9,s6):-mus(1347).
ext_dep(6,7,empty):-mus(1348).
ext_indep(6,7,s0):-mus(1349).
ext_dep(6,7,s1):-mus(1350).
ext_dep(6,7,s3):-mus(1351).
ext_dep(6,7,s4):-mus(1352).
ext_dep(6,7,s5):-mus(1353).
ext_dep(6,7,s8):-mus(1354).
ext_dep(6,7,s9):-mus(1355).
ext_dep(6,7,s0y1):-mus(1356).
ext_dep(6,7,s0y4):-mus(1357).
ext_indep(6,7,s0y5):-mus(1358).
ext_indep(6,7,s0y8):-mus(1359).
ext_dep(6,7,s1y4):-mus(1360).
ext_dep(6,7,s1y5):-mus(1361).
ext_dep(6,7,s8y1):-mus(1362).
ext_indep(6,7,s4y5):-mus(1363).
ext_dep(6,7,s8y4):-mus(1364).
ext_indep(6,7,s8y5):-mus(1365).
ext_indep(6,7,s0y8y5):-mus(1366).
ext_dep(6,8,empty):-mus(1367).
ext_dep(6,8,s0):-mus(1368).
ext_dep(6,8,s1):-mus(1369).
ext_dep(6,8,s2):-mus(1370).
ext_dep(6,8,s3):-mus(1371).
ext_dep(6,8,s4):-mus(1372).
ext_dep(6,8,s5):-mus(1373).
ext_dep(6,8,s7):-mus(1374).
ext_dep(6,8,s9):-mus(1375).
ext_indep(6,8,s0y1):-mus(1376).
ext_dep(6,8,s0y2):-mus(1377).
ext_dep(6,8,s0y3):-mus(1378).
ext_dep(6,8,s0y4):-mus(1379).
ext_indep(6,8,s0y5):-mus(1380).
ext_dep(6,8,s0y7):-mus(1381).
ext_dep(6,8,s0y9):-mus(1382).
ext_indep(6,8,s1y2):-mus(1383).
ext_dep(6,8,s1y3):-mus(1384).
ext_dep(6,8,s1y4):-mus(1385).
ext_indep(6,8,s1y5):-mus(1386).
ext_dep(6,8,s1y7):-mus(1387).
ext_dep(6,8,s1y9):-mus(1388).
ext_dep(6,8,s2y3):-mus(1389).
ext_dep(6,8,s2y4):-mus(1390).
ext_dep(6,8,s2y5):-mus(1391).
ext_dep(6,8,s2y7):-mus(1392).
ext_indep(6,8,s9y2):-mus(1393).
ext_dep(6,8,s3y4):-mus(1394).
ext_dep(6,8,s3y5):-mus(1395).
ext_dep(6,8,s3y7):-mus(1396).
ext_dep(6,8,s9y3):-mus(1397).
ext_dep(6,8,s4y5):-mus(1398).
ext_dep(6,8,s4y7):-mus(1399).
ext_dep(6,8,s9y4):-mus(1400).
ext_dep(6,8,s5y7):-mus(1401).
ext_dep(6,8,s9y5):-mus(1402).
ext_dep(6,8,s9y7):-mus(1403).
ext_indep(6,8,s0y1y2):-mus(1404).
ext_dep(6,8,s0y1y3):-mus(1405).
ext_dep(6,8,s0y1y4):-mus(1406).
ext_dep(6,8,s0y1y5):-mus(1407).
ext_dep(6,8,s0y1y7):-mus(1408).
ext_dep(6,8,s0y1y9):-mus(1409).
ext_dep(6,8,s0y2y3):-mus(1410).
ext_dep(6,8,s0y2y4):-mus(1411).
ext_dep(6,8,s0y2y5):-mus(1412).
ext_dep(6,8,s0y2y7):-mus(1413).
ext_dep(6,8,s0y9y2):-mus(1414).
ext_dep(6,8,s0y3y4):-mus(1415).
ext_dep(6,8,s0y3y5):-mus(1416).
ext_dep(6,8,s0y3y7):-mus(1417).
ext_dep(6,8,s0y9y3):-mus(1418).
ext_dep(6,8,s0y4y5):-mus(1419).
ext_indep(6,8,s0y4y7):-mus(1420).
ext_dep(6,8,s0y9y4):-mus(1421).
ext_indep(6,8,s0y5y7):-mus(1422).
ext_dep(6,8,s0y9y5):-mus(1423).
ext_dep(6,8,s0y9y7):-mus(1424).
ext_dep(6,8,s1y2y3):-mus(1425).
ext_dep(6,8,s1y2y4):-mus(1426).
ext_dep(6,8,s1y2y5):-mus(1427).
ext_dep(6,8,s1y2y7):-mus(1428).
ext_dep(6,8,s1y2y9):-mus(1429).
ext_dep(6,8,s1y3y4):-mus(1430).
ext_dep(6,8,s1y3y5):-mus(1431).
ext_dep(6,8,s1y3y7):-mus(1432).
ext_dep(6,8,s1y3y9):-mus(1433).
ext_dep(6,8,s1y4y5):-mus(1434).
ext_dep(6,8,s1y4y7):-mus(1435).
ext_dep(6,8,s1y4y9):-mus(1436).
ext_indep(6,8,s1y5y7):-mus(1437).
ext_indep(6,8,s1y5y9):-mus(1438).
ext_dep(6,8,s1y9y7):-mus(1439).
ext_dep(6,8,s2y3y4):-mus(1440).
ext_dep(6,8,s2y3y5):-mus(1441).
ext_dep(6,8,s2y3y7):-mus(1442).
ext_indep(6,8,s9y2y3):-mus(1443).
ext_indep(6,8,s2y4y5):-mus(1444).
ext_dep(6,8,s2y4y7):-mus(1445).
ext_dep(6,8,s9y2y4):-mus(1446).
ext_dep(6,8,s2y5y7):-mus(1447).
ext_indep(6,8,s9y2y5):-mus(1448).
ext_dep(6,8,s9y2y7):-mus(1449).
ext_dep(6,8,s3y4y5):-mus(1450).
ext_dep(6,8,s3y4y7):-mus(1451).
ext_dep(6,8,s9y3y4):-mus(1452).
ext_indep(6,8,s3y5y7):-mus(1453).
ext_indep(6,8,s9y3y5):-mus(1454).
ext_dep(6,8,s9y3y7):-mus(1455).
ext_dep(6,8,s4y5y7):-mus(1456).
ext_dep(6,8,s9y4y5):-mus(1457).
ext_dep(6,8,s9y4y7):-mus(1458).
ext_dep(6,8,s9y5y7):-mus(1459).
ext_dep(6,8,s0y1y2y3):-mus(1460).
ext_dep(6,8,s0y1y2y4):-mus(1461).
ext_dep(6,8,s0y1y2y5):-mus(1462).
ext_dep(6,8,s0y1y2y9):-mus(1463).
ext_dep(6,8,s0y1y3y4):-mus(1464).
ext_dep(6,8,s0y1y3y5):-mus(1465).
ext_indep(6,8,s0y1y3y9):-mus(1466).
ext_dep(6,8,s0y1y4y5):-mus(1467).
ext_dep(6,8,s0y1y4y9):-mus(1468).
ext_dep(6,8,s0y1y5y9):-mus(1469).
ext_dep(6,8,s0y2y3y4):-mus(1470).
ext_dep(6,8,s0y2y3y5):-mus(1471).
ext_dep(6,8,s0y9y2y3):-mus(1472).
ext_dep(6,8,s0y2y4y5):-mus(1473).
ext_dep(6,8,s0y9y2y4):-mus(1474).
ext_dep(6,8,s0y9y2y5):-mus(1475).
ext_indep(6,8,s0y3y4y5):-mus(1476).
ext_dep(6,8,s0y9y3y4):-mus(1477).
ext_indep(6,8,s0y9y3y5):-mus(1478).
ext_dep(6,8,s0y9y4y5):-mus(1479).
ext_dep(6,8,s1y2y3y4):-mus(1480).
ext_dep(6,8,s1y2y3y5):-mus(1481).
ext_dep(6,8,s1y2y3y9):-mus(1482).
ext_dep(6,8,s1y2y4y5):-mus(1483).
ext_dep(6,8,s1y2y4y9):-mus(1484).
ext_dep(6,8,s1y2y5y9):-mus(1485).
ext_dep(6,8,s1y3y4y5):-mus(1486).
ext_dep(6,8,s1y3y4y9):-mus(1487).
ext_dep(6,8,s1y3y5y9):-mus(1488).
ext_dep(6,8,s1y4y5y9):-mus(1489).
ext_dep(6,8,s2y3y4y5):-mus(1490).
ext_indep(6,8,s9y2y3y4):-mus(1491).
ext_dep(6,8,s9y2y3y5):-mus(1492).
ext_indep(6,8,s9y2y4y5):-mus(1493).
ext_dep(6,8,s9y3y4y5):-mus(1494).
ext_dep(6,8,s0y1y2y3y4):-mus(1495).
ext_dep(6,8,s0y1y2y3y5):-mus(1496).
ext_dep(6,8,s0y1y2y3y9):-mus(1497).
ext_dep(6,8,s0y1y2y4y5):-mus(1498).
ext_dep(6,8,s0y1y2y4y9):-mus(1499).
ext_dep(6,8,s0y1y2y5y9):-mus(1500).
ext_dep(6,8,s0y1y3y4y5):-mus(1501).
ext_dep(6,8,s0y1y3y4y9):-mus(1502).
ext_dep(6,8,s0y1y3y5y9):-mus(1503).
ext_dep(6,8,s0y1y4y5y9):-mus(1504).
ext_indep(6,8,s0y2y3y4y5):-mus(1505).
ext_dep(6,8,s0y2y3y4y9):-mus(1506).
ext_dep(6,8,s0y2y3y5y9):-mus(1507).
ext_dep(6,8,s0y2y4y5y9):-mus(1508).
ext_dep(6,8,s0y3y4y5y9):-mus(1509).
ext_indep(6,8,s1y2y3y4y5):-mus(1510).
ext_dep(6,8,s1y2y3y4y9):-mus(1511).
ext_dep(6,8,s1y2y3y5y9):-mus(1512).
ext_dep(6,8,s1y2y4y5y9):-mus(1513).
ext_dep(6,8,s1y3y4y5y9):-mus(1514).
ext_dep(6,8,s2y3y4y5y9):-mus(1515).
ext_dep(6,8,s0y1y2y3y4y5):-mus(1516).
ext_dep(6,8,s0y1y2y3y4y9):-mus(1517).
ext_dep(6,8,s0y1y2y3y5y9):-mus(1518).
ext_dep(6,8,s0y1y2y4y5y9):-mus(1519).
ext_dep(6,8,s0y1y3y4y5y9):-mus(1520).
ext_dep(6,8,s0y2y3y4y5y9):-mus(1521).
ext_dep(6,8,s1y2y3y4y5y9):-mus(1522).
ext_dep(6,8,s0y1y2y3y4y5y9):-mus(1523).
ext_dep(6,9,s0):-mus(1524).
ext_dep(6,9,s2):-mus(1525).
ext_dep(6,9,s4):-mus(1526).
ext_dep(6,9,s5):-mus(1527).
ext_indep(7,9,empty):-mus(1528).
ext_dep(8,9,empty):-mus(1529).
ext_dep(8,9,s0):-mus(1530).
ext_indep(8,9,s1):-mus(1531).
ext_dep(8,9,s2):-mus(1532).
ext_dep(8,9,s3):-mus(1533).
ext_dep(8,9,s4):-mus(1534).
ext_dep(8,9,s5):-mus(1535).
ext_dep(8,9,s6):-mus(1536).
ext_dep(8,9,s7):-mus(1537).
ext_dep(8,9,s0y1):-mus(1538).
ext_dep(8,9,s0y2):-mus(1539).
ext_dep(8,9,s0y3):-mus(1540).
ext_dep(8,9,s0y4):-mus(1541).
ext_dep(8,9,s0y5):-mus(1542).
ext_dep(8,9,s0y6):-mus(1543).
ext_dep(8,9,s0y7):-mus(1544).
ext_indep(8,9,s1y2):-mus(1545).
ext_dep(8,9,s1y3):-mus(1546).
ext_dep(8,9,s1y4):-mus(1547).
ext_dep(8,9,s1y5):-mus(1548).
ext_dep(8,9,s1y6):-mus(1549).
ext_dep(8,9,s1y7):-mus(1550).
ext_dep(8,9,s2y3):-mus(1551).
ext_dep(8,9,s2y4):-mus(1552).
ext_dep(8,9,s2y5):-mus(1553).
ext_dep(8,9,s2y6):-mus(1554).
ext_indep(8,9,s2y7):-mus(1555).
ext_dep(8,9,s3y4):-mus(1556).
ext_dep(8,9,s3y5):-mus(1557).
ext_indep(8,9,s3y6):-mus(1558).
ext_dep(8,9,s3y7):-mus(1559).
ext_indep(8,9,s4y5):-mus(1560).
ext_indep(8,9,s4y6):-mus(1561).
ext_dep(8,9,s4y7):-mus(1562).
ext_dep(8,9,s5y6):-mus(1563).
ext_indep(8,9,s5y7):-mus(1564).
ext_dep(8,9,s6y7):-mus(1565).
ext_dep(8,9,s0y1y2):-mus(1566).
ext_dep(8,9,s0y1y3):-mus(1567).
ext_dep(8,9,s0y1y4):-mus(1568).
ext_dep(8,9,s0y1y5):-mus(1569).
ext_dep(8,9,s0y1y6):-mus(1570).
ext_dep(8,9,s0y1y7):-mus(1571).
ext_dep(8,9,s0y2y3):-mus(1572).
ext_dep(8,9,s0y2y4):-mus(1573).
ext_dep(8,9,s0y2y5):-mus(1574).
ext_indep(8,9,s0y2y6):-mus(1575).
ext_dep(8,9,s0y2y7):-mus(1576).
ext_dep(8,9,s0y3y4):-mus(1577).
ext_dep(8,9,s0y3y5):-mus(1578).
ext_dep(8,9,s0y3y6):-mus(1579).
ext_dep(8,9,s0y3y7):-mus(1580).
ext_indep(8,9,s0y4y5):-mus(1581).
ext_dep(8,9,s0y4y6):-mus(1582).
ext_dep(8,9,s0y4y7):-mus(1583).
ext_indep(8,9,s0y5y6):-mus(1584).
ext_dep(8,9,s0y5y7):-mus(1585).
ext_dep(8,9,s0y6y7):-mus(1586).
ext_dep(8,9,s1y2y3):-mus(1587).
ext_dep(8,9,s1y2y4):-mus(1588).
ext_dep(8,9,s1y2y5):-mus(1589).
ext_indep(8,9,s1y2y6):-mus(1590).
ext_dep(8,9,s1y2y7):-mus(1591).
ext_dep(8,9,s1y3y4):-mus(1592).
ext_dep(8,9,s1y3y5):-mus(1593).
ext_dep(8,9,s1y3y6):-mus(1594).
ext_dep(8,9,s1y3y7):-mus(1595).
ext_dep(8,9,s1y4y5):-mus(1596).
ext_dep(8,9,s1y4y6):-mus(1597).
ext_dep(8,9,s1y4y7):-mus(1598).
ext_dep(8,9,s1y5y6):-mus(1599).
ext_dep(8,9,s1y5y7):-mus(1600).
ext_indep(8,9,s1y6y7):-mus(1601).
ext_dep(8,9,s2y3y4):-mus(1602).
ext_dep(8,9,s2y3y5):-mus(1603).
ext_indep(8,9,s2y3y6):-mus(1604).
ext_indep(8,9,s2y3y7):-mus(1605).
ext_indep(8,9,s2y4y5):-mus(1606).
ext_dep(8,9,s2y4y6):-mus(1607).
ext_dep(8,9,s2y4y7):-mus(1608).
ext_dep(8,9,s2y5y6):-mus(1609).
ext_dep(8,9,s2y5y7):-mus(1610).
ext_dep(8,9,s2y6y7):-mus(1611).
ext_dep(8,9,s3y4y5):-mus(1612).
ext_dep(8,9,s3y4y6):-mus(1613).
ext_dep(8,9,s3y4y7):-mus(1614).
ext_dep(8,9,s3y5y6):-mus(1615).
ext_dep(8,9,s3y5y7):-mus(1616).
ext_dep(8,9,s3y6y7):-mus(1617).
ext_indep(8,9,s4y5y6):-mus(1618).
ext_dep(8,9,s4y5y7):-mus(1619).
ext_dep(8,9,s4y6y7):-mus(1620).
ext_indep(8,9,s5y6y7):-mus(1621).
ext_dep(8,9,s0y1y2y3):-mus(1622).
ext_dep(8,9,s0y1y2y4):-mus(1623).
ext_dep(8,9,s0y1y2y5):-mus(1624).
ext_dep(8,9,s0y1y2y6):-mus(1625).
ext_dep(8,9,s0y1y3y4):-mus(1626).
ext_dep(8,9,s0y1y3y5):-mus(1627).
ext_dep(8,9,s0y1y3y6):-mus(1628).
ext_indep(8,9,s0y1y4y5):-mus(1629).
ext_dep(8,9,s0y1y4y6):-mus(1630).
ext_dep(8,9,s0y1y5y6):-mus(1631).
ext_dep(8,9,s0y2y3y4):-mus(1632).
ext_dep(8,9,s0y2y3y5):-mus(1633).
ext_indep(8,9,s0y2y3y6):-mus(1634).
ext_dep(8,9,s0y2y4y5):-mus(1635).
ext_dep(8,9,s0y2y4y6):-mus(1636).
ext_dep(8,9,s0y2y5y6):-mus(1637).
ext_dep(8,9,s0y3y4y5):-mus(1638).
ext_dep(8,9,s0y3y4y6):-mus(1639).
ext_dep(8,9,s0y3y5y6):-mus(1640).
ext_dep(8,9,s0y4y5y6):-mus(1641).
ext_dep(8,9,s1y2y3y4):-mus(1642).
ext_dep(8,9,s1y2y3y5):-mus(1643).
ext_dep(8,9,s1y2y3y6):-mus(1644).
ext_dep(8,9,s1y2y4y5):-mus(1645).
ext_dep(8,9,s1y2y4y6):-mus(1646).
ext_dep(8,9,s1y2y5y6):-mus(1647).
ext_dep(8,9,s1y3y4y5):-mus(1648).
ext_dep(8,9,s1y3y4y6):-mus(1649).
ext_dep(8,9,s1y3y5y6):-mus(1650).
ext_dep(8,9,s1y4y5y6):-mus(1651).
ext_dep(8,9,s2y3y4y5):-mus(1652).
ext_indep(8,9,s2y3y4y6):-mus(1653).
ext_dep(8,9,s2y3y5y6):-mus(1654).
ext_dep(8,9,s2y4y5y6):-mus(1655).
ext_dep(8,9,s3y4y5y6):-mus(1656).
ext_dep(8,9,s0y1y2y3y4):-mus(1657).
ext_dep(8,9,s0y1y2y3y5):-mus(1658).
ext_dep(8,9,s0y1y2y3y6):-mus(1659).
ext_dep(8,9,s0y1y2y4y5):-mus(1660).
ext_dep(8,9,s0y1y2y4y6):-mus(1661).
ext_dep(8,9,s0y1y2y5y6):-mus(1662).
ext_indep(8,9,s0y1y3y4y5):-mus(1663).
ext_dep(8,9,s0y1y3y4y6):-mus(1664).
ext_dep(8,9,s0y1y3y5y6):-mus(1665).
ext_dep(8,9,s0y1y4y5y6):-mus(1666).
ext_dep(8,9,s0y2y3y4y5):-mus(1667).
ext_indep(8,9,s0y2y3y4y6):-mus(1668).
ext_dep(8,9,s0y2y3y5y6):-mus(1669).
ext_dep(8,9,s0y2y4y5y6):-mus(1670).
ext_dep(8,9,s0y3y4y5y6):-mus(1671).
ext_dep(8,9,s1y2y3y4y5):-mus(1672).
ext_indep(8,9,s1y2y3y4y6):-mus(1673).
ext_dep(8,9,s1y2y3y5y6):-mus(1674).
ext_dep(8,9,s1y2y4y5y6):-mus(1675).
ext_dep(8,9,s1y3y4y5y6):-mus(1676).
ext_dep(8,9,s2y3y4y5y6):-mus(1677).
ext_dep(8,9,s0y1y2y3y4y5):-mus(1678).
ext_dep(8,9,s0y1y2y3y4y6):-mus(1679).
ext_dep(8,9,s0y1y2y3y5y6):-mus(1680).
ext_dep(8,9,s0y1y2y4y5y6):-mus(1681).
ext_dep(8,9,s0y1y3y4y5y6):-mus(1682).
ext_dep(8,9,s0y2y3y4y5y6):-mus(1683).
ext_dep(8,9,s1y2y3y4y5y6):-mus(1684).
ext_dep(8,9,s0y1y2y3y4y5y6):-mus(1685).

% ===== Objective Literals for Optimum MCS =====
% __optimum_mcs_objective_literal__(W, mus(I)).
__optimum_mcs_objective_literal__(8619952, mus(1)).
__optimum_mcs_objective_literal__(5378843, mus(2)).
__optimum_mcs_objective_literal__(7364572, mus(3)).
__optimum_mcs_objective_literal__(6943802, mus(4)).
__optimum_mcs_objective_literal__(6292283, mus(5)).
__optimum_mcs_objective_literal__(7717856, mus(6)).
__optimum_mcs_objective_literal__(5188059, mus(7)).
__optimum_mcs_objective_literal__(9195963, mus(8)).
__optimum_mcs_objective_literal__(9294232, mus(9)).
__optimum_mcs_objective_literal__(7168780, mus(10)).
__optimum_mcs_objective_literal__(5030305, mus(11)).
__optimum_mcs_objective_literal__(9419113, mus(12)).
__optimum_mcs_objective_literal__(5860121, mus(13)).
__optimum_mcs_objective_literal__(6526372, mus(14)).
__optimum_mcs_objective_literal__(7949366, mus(15)).
__optimum_mcs_objective_literal__(6041565, mus(16)).
__optimum_mcs_objective_literal__(7725304, mus(17)).
__optimum_mcs_objective_literal__(8178105, mus(18)).
__optimum_mcs_objective_literal__(8714503, mus(19)).
__optimum_mcs_objective_literal__(8983263, mus(20)).
__optimum_mcs_objective_literal__(9514880, mus(21)).
__optimum_mcs_objective_literal__(7293807, mus(22)).
__optimum_mcs_objective_literal__(9290976, mus(23)).
__optimum_mcs_objective_literal__(5378362, mus(24)).
__optimum_mcs_objective_literal__(6016115, mus(25)).
__optimum_mcs_objective_literal__(8989851, mus(26)).
__optimum_mcs_objective_literal__(7768890, mus(27)).
__optimum_mcs_objective_literal__(8556058, mus(28)).
__optimum_mcs_objective_literal__(9989508, mus(29)).
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
__optimum_mcs_objective_literal__(5577804, mus(69)).
__optimum_mcs_objective_literal__(5025627, mus(70)).
__optimum_mcs_objective_literal__(5757357, mus(71)).
__optimum_mcs_objective_literal__(6741132, mus(72)).
__optimum_mcs_objective_literal__(6235641, mus(73)).
__optimum_mcs_objective_literal__(6538887, mus(74)).
__optimum_mcs_objective_literal__(6125516, mus(75)).
__optimum_mcs_objective_literal__(5402341, mus(76)).
__optimum_mcs_objective_literal__(7273879, mus(77)).
__optimum_mcs_objective_literal__(8488820, mus(78)).
__optimum_mcs_objective_literal__(5773135, mus(79)).
__optimum_mcs_objective_literal__(5864017, mus(80)).
__optimum_mcs_objective_literal__(6397698, mus(81)).
__optimum_mcs_objective_literal__(7469722, mus(82)).
__optimum_mcs_objective_literal__(5076271, mus(83)).
__optimum_mcs_objective_literal__(5190365, mus(84)).
__optimum_mcs_objective_literal__(9999806, mus(85)).
__optimum_mcs_objective_literal__(9927123, mus(86)).
__optimum_mcs_objective_literal__(9999999, mus(87)).
__optimum_mcs_objective_literal__(9999999, mus(88)).
__optimum_mcs_objective_literal__(9999999, mus(89)).
__optimum_mcs_objective_literal__(9999998, mus(90)).
__optimum_mcs_objective_literal__(9999999, mus(91)).
__optimum_mcs_objective_literal__(9999997, mus(92)).
__optimum_mcs_objective_literal__(9999999, mus(93)).
__optimum_mcs_objective_literal__(9999794, mus(94)).
__optimum_mcs_objective_literal__(9999999, mus(95)).
__optimum_mcs_objective_literal__(9999999, mus(96)).
__optimum_mcs_objective_literal__(9999999, mus(97)).
__optimum_mcs_objective_literal__(9999999, mus(98)).
__optimum_mcs_objective_literal__(9999999, mus(99)).
__optimum_mcs_objective_literal__(9999999, mus(100)).
__optimum_mcs_objective_literal__(9999916, mus(101)).
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
__optimum_mcs_objective_literal__(9999986, mus(127)).
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
__optimum_mcs_objective_literal__(9999596, mus(144)).
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
__optimum_mcs_objective_literal__(9998497, mus(164)).
__optimum_mcs_objective_literal__(9998018, mus(165)).
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
__optimum_mcs_objective_literal__(9993897, mus(199)).
__optimum_mcs_objective_literal__(9999999, mus(200)).
__optimum_mcs_objective_literal__(9999998, mus(201)).
__optimum_mcs_objective_literal__(9999999, mus(202)).
__optimum_mcs_objective_literal__(9997543, mus(203)).
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
__optimum_mcs_objective_literal__(9999976, mus(225)).
__optimum_mcs_objective_literal__(9999999, mus(226)).
__optimum_mcs_objective_literal__(9999997, mus(227)).
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
__optimum_mcs_objective_literal__(9999972, mus(243)).
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
__optimum_mcs_objective_literal__(5220999, mus(273)).
__optimum_mcs_objective_literal__(9946611, mus(274)).
__optimum_mcs_objective_literal__(9989133, mus(275)).
__optimum_mcs_objective_literal__(9987228, mus(276)).
__optimum_mcs_objective_literal__(9996838, mus(277)).
__optimum_mcs_objective_literal__(9981852, mus(278)).
__optimum_mcs_objective_literal__(6452636, mus(279)).
__optimum_mcs_objective_literal__(9999999, mus(280)).
__optimum_mcs_objective_literal__(6724074, mus(281)).
__optimum_mcs_objective_literal__(9999999, mus(282)).
__optimum_mcs_objective_literal__(6926796, mus(283)).
__optimum_mcs_objective_literal__(9999999, mus(284)).
__optimum_mcs_objective_literal__(9999999, mus(285)).
__optimum_mcs_objective_literal__(6063752, mus(286)).
__optimum_mcs_objective_literal__(9999999, mus(287)).
__optimum_mcs_objective_literal__(9999214, mus(288)).
__optimum_mcs_objective_literal__(7553125, mus(289)).
__optimum_mcs_objective_literal__(5255867, mus(290)).
__optimum_mcs_objective_literal__(9988956, mus(291)).
__optimum_mcs_objective_literal__(9993340, mus(292)).
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
__optimum_mcs_objective_literal__(5670497, mus(308)).
__optimum_mcs_objective_literal__(8649480, mus(309)).
__optimum_mcs_objective_literal__(5801210, mus(310)).
__optimum_mcs_objective_literal__(5182190, mus(311)).
__optimum_mcs_objective_literal__(7667839, mus(312)).
__optimum_mcs_objective_literal__(6142612, mus(313)).
__optimum_mcs_objective_literal__(8461189, mus(314)).
__optimum_mcs_objective_literal__(5398594, mus(315)).
__optimum_mcs_objective_literal__(5229875, mus(316)).
__optimum_mcs_objective_literal__(8529391, mus(317)).
__optimum_mcs_objective_literal__(5153390, mus(318)).
__optimum_mcs_objective_literal__(8522348, mus(319)).
__optimum_mcs_objective_literal__(7650982, mus(320)).
__optimum_mcs_objective_literal__(8294025, mus(321)).
__optimum_mcs_objective_literal__(7986597, mus(322)).
__optimum_mcs_objective_literal__(5828800, mus(323)).
__optimum_mcs_objective_literal__(8579530, mus(324)).
__optimum_mcs_objective_literal__(5755513, mus(325)).
__optimum_mcs_objective_literal__(5368555, mus(326)).
__optimum_mcs_objective_literal__(9708555, mus(327)).
__optimum_mcs_objective_literal__(5213755, mus(328)).
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
__optimum_mcs_objective_literal__(5179186, mus(354)).
__optimum_mcs_objective_literal__(6081911, mus(355)).
__optimum_mcs_objective_literal__(8172490, mus(356)).
__optimum_mcs_objective_literal__(5098247, mus(357)).
__optimum_mcs_objective_literal__(8869228, mus(358)).
__optimum_mcs_objective_literal__(6803969, mus(359)).
__optimum_mcs_objective_literal__(5631673, mus(360)).
__optimum_mcs_objective_literal__(5350487, mus(361)).
__optimum_mcs_objective_literal__(7662384, mus(362)).
__optimum_mcs_objective_literal__(9298969, mus(363)).
__optimum_mcs_objective_literal__(8461990, mus(364)).
__optimum_mcs_objective_literal__(5193334, mus(365)).
__optimum_mcs_objective_literal__(6448185, mus(366)).
__optimum_mcs_objective_literal__(7872064, mus(367)).
__optimum_mcs_objective_literal__(6352180, mus(368)).
__optimum_mcs_objective_literal__(6151583, mus(369)).
__optimum_mcs_objective_literal__(5664624, mus(370)).
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
__optimum_mcs_objective_literal__(9999999, mus(490)).
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
__optimum_mcs_objective_literal__(9997505, mus(544)).
__optimum_mcs_objective_literal__(7774359, mus(545)).
__optimum_mcs_objective_literal__(9999999, mus(546)).
__optimum_mcs_objective_literal__(9998541, mus(547)).
__optimum_mcs_objective_literal__(9885273, mus(548)).
__optimum_mcs_objective_literal__(9629560, mus(549)).
__optimum_mcs_objective_literal__(9999999, mus(550)).
__optimum_mcs_objective_literal__(9964933, mus(551)).
__optimum_mcs_objective_literal__(5757053, mus(552)).
__optimum_mcs_objective_literal__(7147192, mus(553)).
__optimum_mcs_objective_literal__(9391046, mus(554)).
__optimum_mcs_objective_literal__(6719643, mus(555)).
__optimum_mcs_objective_literal__(9983286, mus(556)).
__optimum_mcs_objective_literal__(5263830, mus(557)).
__optimum_mcs_objective_literal__(9998464, mus(558)).
__optimum_mcs_objective_literal__(9997749, mus(559)).
__optimum_mcs_objective_literal__(6561498, mus(560)).
__optimum_mcs_objective_literal__(8591824, mus(561)).
__optimum_mcs_objective_literal__(8409209, mus(562)).
__optimum_mcs_objective_literal__(9998392, mus(563)).
__optimum_mcs_objective_literal__(9999999, mus(564)).
__optimum_mcs_objective_literal__(9378631, mus(565)).
__optimum_mcs_objective_literal__(9999999, mus(566)).
__optimum_mcs_objective_literal__(9999999, mus(567)).
__optimum_mcs_objective_literal__(9999999, mus(568)).
__optimum_mcs_objective_literal__(9999999, mus(569)).
__optimum_mcs_objective_literal__(9999999, mus(570)).
__optimum_mcs_objective_literal__(9999999, mus(571)).
__optimum_mcs_objective_literal__(9999999, mus(572)).
__optimum_mcs_objective_literal__(8830494, mus(573)).
__optimum_mcs_objective_literal__(7449145, mus(574)).
__optimum_mcs_objective_literal__(8862989, mus(575)).
__optimum_mcs_objective_literal__(9999999, mus(576)).
__optimum_mcs_objective_literal__(9999999, mus(577)).
__optimum_mcs_objective_literal__(9999999, mus(578)).
__optimum_mcs_objective_literal__(9984184, mus(579)).
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
__optimum_mcs_objective_literal__(9999999, mus(673)).
__optimum_mcs_objective_literal__(9999999, mus(674)).
__optimum_mcs_objective_literal__(9999999, mus(675)).
__optimum_mcs_objective_literal__(9999999, mus(676)).
__optimum_mcs_objective_literal__(9999999, mus(677)).
__optimum_mcs_objective_literal__(9999999, mus(678)).
__optimum_mcs_objective_literal__(9999999, mus(679)).
__optimum_mcs_objective_literal__(9999999, mus(680)).
__optimum_mcs_objective_literal__(9999999, mus(681)).
__optimum_mcs_objective_literal__(9999999, mus(682)).
__optimum_mcs_objective_literal__(9999999, mus(683)).
__optimum_mcs_objective_literal__(9999999, mus(684)).
__optimum_mcs_objective_literal__(9999999, mus(685)).
__optimum_mcs_objective_literal__(9999999, mus(686)).
__optimum_mcs_objective_literal__(9999999, mus(687)).
__optimum_mcs_objective_literal__(9999999, mus(688)).
__optimum_mcs_objective_literal__(9999999, mus(689)).
__optimum_mcs_objective_literal__(9999999, mus(690)).
__optimum_mcs_objective_literal__(9999999, mus(691)).
__optimum_mcs_objective_literal__(9999999, mus(692)).
__optimum_mcs_objective_literal__(9999999, mus(693)).
__optimum_mcs_objective_literal__(9999999, mus(694)).
__optimum_mcs_objective_literal__(9999999, mus(695)).
__optimum_mcs_objective_literal__(9999999, mus(696)).
__optimum_mcs_objective_literal__(9999999, mus(697)).
__optimum_mcs_objective_literal__(9999999, mus(698)).
__optimum_mcs_objective_literal__(9999999, mus(699)).
__optimum_mcs_objective_literal__(9999999, mus(700)).
__optimum_mcs_objective_literal__(9999999, mus(701)).
__optimum_mcs_objective_literal__(9999999, mus(702)).
__optimum_mcs_objective_literal__(9999999, mus(703)).
__optimum_mcs_objective_literal__(9999999, mus(704)).
__optimum_mcs_objective_literal__(9999999, mus(705)).
__optimum_mcs_objective_literal__(9999999, mus(706)).
__optimum_mcs_objective_literal__(9999999, mus(707)).
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
__optimum_mcs_objective_literal__(9999999, mus(735)).
__optimum_mcs_objective_literal__(9999999, mus(736)).
__optimum_mcs_objective_literal__(9999999, mus(737)).
__optimum_mcs_objective_literal__(9999999, mus(738)).
__optimum_mcs_objective_literal__(9999999, mus(739)).
__optimum_mcs_objective_literal__(9999999, mus(740)).
__optimum_mcs_objective_literal__(6483130, mus(741)).
__optimum_mcs_objective_literal__(9999999, mus(742)).
__optimum_mcs_objective_literal__(9999999, mus(743)).
__optimum_mcs_objective_literal__(9999999, mus(744)).
__optimum_mcs_objective_literal__(9999999, mus(745)).
__optimum_mcs_objective_literal__(9999749, mus(746)).
__optimum_mcs_objective_literal__(9990264, mus(747)).
__optimum_mcs_objective_literal__(9999992, mus(748)).
__optimum_mcs_objective_literal__(9945234, mus(749)).
__optimum_mcs_objective_literal__(6786343, mus(750)).
__optimum_mcs_objective_literal__(9992981, mus(751)).
__optimum_mcs_objective_literal__(7553748, mus(752)).
__optimum_mcs_objective_literal__(5615792, mus(753)).
__optimum_mcs_objective_literal__(9830316, mus(754)).
__optimum_mcs_objective_literal__(9894189, mus(755)).
__optimum_mcs_objective_literal__(8033900, mus(756)).
__optimum_mcs_objective_literal__(9698013, mus(757)).
__optimum_mcs_objective_literal__(8334756, mus(758)).
__optimum_mcs_objective_literal__(8004497, mus(759)).
__optimum_mcs_objective_literal__(9837606, mus(760)).
__optimum_mcs_objective_literal__(9999996, mus(761)).
__optimum_mcs_objective_literal__(9999924, mus(762)).
__optimum_mcs_objective_literal__(9999976, mus(763)).
__optimum_mcs_objective_literal__(9999999, mus(764)).
__optimum_mcs_objective_literal__(9998415, mus(765)).
__optimum_mcs_objective_literal__(9999999, mus(766)).
__optimum_mcs_objective_literal__(9999999, mus(767)).
__optimum_mcs_objective_literal__(9999999, mus(768)).
__optimum_mcs_objective_literal__(9999999, mus(769)).
__optimum_mcs_objective_literal__(9999994, mus(770)).
__optimum_mcs_objective_literal__(9999999, mus(771)).
__optimum_mcs_objective_literal__(9999998, mus(772)).
__optimum_mcs_objective_literal__(9999999, mus(773)).
__optimum_mcs_objective_literal__(9999998, mus(774)).
__optimum_mcs_objective_literal__(9999999, mus(775)).
__optimum_mcs_objective_literal__(9999999, mus(776)).
__optimum_mcs_objective_literal__(9999999, mus(777)).
__optimum_mcs_objective_literal__(9999999, mus(778)).
__optimum_mcs_objective_literal__(9999999, mus(779)).
__optimum_mcs_objective_literal__(9999999, mus(780)).
__optimum_mcs_objective_literal__(9999999, mus(781)).
__optimum_mcs_objective_literal__(9999999, mus(782)).
__optimum_mcs_objective_literal__(7878600, mus(783)).
__optimum_mcs_objective_literal__(9999999, mus(784)).
__optimum_mcs_objective_literal__(9999999, mus(785)).
__optimum_mcs_objective_literal__(7723114, mus(786)).
__optimum_mcs_objective_literal__(8778786, mus(787)).
__optimum_mcs_objective_literal__(9999999, mus(788)).
__optimum_mcs_objective_literal__(8019212, mus(789)).
__optimum_mcs_objective_literal__(9997456, mus(790)).
__optimum_mcs_objective_literal__(6686432, mus(791)).
__optimum_mcs_objective_literal__(9999999, mus(792)).
__optimum_mcs_objective_literal__(9672013, mus(793)).
__optimum_mcs_objective_literal__(8133221, mus(794)).
__optimum_mcs_objective_literal__(9455878, mus(795)).
__optimum_mcs_objective_literal__(5456277, mus(796)).
__optimum_mcs_objective_literal__(9421646, mus(797)).
__optimum_mcs_objective_literal__(9216255, mus(798)).
__optimum_mcs_objective_literal__(5986171, mus(799)).
__optimum_mcs_objective_literal__(9999999, mus(800)).
__optimum_mcs_objective_literal__(9999999, mus(801)).
__optimum_mcs_objective_literal__(9999999, mus(802)).
__optimum_mcs_objective_literal__(9999999, mus(803)).
__optimum_mcs_objective_literal__(9999999, mus(804)).
__optimum_mcs_objective_literal__(9999999, mus(805)).
__optimum_mcs_objective_literal__(9999999, mus(806)).
__optimum_mcs_objective_literal__(9999999, mus(807)).
__optimum_mcs_objective_literal__(9999999, mus(808)).
__optimum_mcs_objective_literal__(9999999, mus(809)).
__optimum_mcs_objective_literal__(9999999, mus(810)).
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
__optimum_mcs_objective_literal__(9999999, mus(880)).
__optimum_mcs_objective_literal__(9999999, mus(881)).
__optimum_mcs_objective_literal__(9999999, mus(882)).
__optimum_mcs_objective_literal__(9999999, mus(883)).
__optimum_mcs_objective_literal__(9999999, mus(884)).
__optimum_mcs_objective_literal__(9999999, mus(885)).
__optimum_mcs_objective_literal__(9999999, mus(886)).
__optimum_mcs_objective_literal__(9999999, mus(887)).
__optimum_mcs_objective_literal__(9999999, mus(888)).
__optimum_mcs_objective_literal__(9999999, mus(889)).
__optimum_mcs_objective_literal__(9999999, mus(890)).
__optimum_mcs_objective_literal__(9999999, mus(891)).
__optimum_mcs_objective_literal__(9999999, mus(892)).
__optimum_mcs_objective_literal__(9999999, mus(893)).
__optimum_mcs_objective_literal__(9999999, mus(894)).
__optimum_mcs_objective_literal__(9999999, mus(895)).
__optimum_mcs_objective_literal__(9999999, mus(896)).
__optimum_mcs_objective_literal__(9999999, mus(897)).
__optimum_mcs_objective_literal__(9999999, mus(898)).
__optimum_mcs_objective_literal__(9999999, mus(899)).
__optimum_mcs_objective_literal__(9999999, mus(900)).
__optimum_mcs_objective_literal__(9999999, mus(901)).
__optimum_mcs_objective_literal__(9999999, mus(902)).
__optimum_mcs_objective_literal__(9999999, mus(903)).
__optimum_mcs_objective_literal__(9999999, mus(904)).
__optimum_mcs_objective_literal__(9999999, mus(905)).
__optimum_mcs_objective_literal__(9999999, mus(906)).
__optimum_mcs_objective_literal__(9999999, mus(907)).
__optimum_mcs_objective_literal__(9999999, mus(908)).
__optimum_mcs_objective_literal__(9999999, mus(909)).
__optimum_mcs_objective_literal__(9999999, mus(910)).
__optimum_mcs_objective_literal__(9999999, mus(911)).
__optimum_mcs_objective_literal__(9999999, mus(912)).
__optimum_mcs_objective_literal__(9999999, mus(913)).
__optimum_mcs_objective_literal__(9999999, mus(914)).
__optimum_mcs_objective_literal__(9999999, mus(915)).
__optimum_mcs_objective_literal__(9999999, mus(916)).
__optimum_mcs_objective_literal__(9999999, mus(917)).
__optimum_mcs_objective_literal__(9999999, mus(918)).
__optimum_mcs_objective_literal__(9999999, mus(919)).
__optimum_mcs_objective_literal__(9999999, mus(920)).
__optimum_mcs_objective_literal__(9999999, mus(921)).
__optimum_mcs_objective_literal__(9999999, mus(922)).
__optimum_mcs_objective_literal__(9999999, mus(923)).
__optimum_mcs_objective_literal__(9999999, mus(924)).
__optimum_mcs_objective_literal__(9999999, mus(925)).
__optimum_mcs_objective_literal__(9999999, mus(926)).
__optimum_mcs_objective_literal__(9999999, mus(927)).
__optimum_mcs_objective_literal__(9999999, mus(928)).
__optimum_mcs_objective_literal__(9999999, mus(929)).
__optimum_mcs_objective_literal__(9999999, mus(930)).
__optimum_mcs_objective_literal__(9999999, mus(931)).
__optimum_mcs_objective_literal__(9999999, mus(932)).
__optimum_mcs_objective_literal__(9999999, mus(933)).
__optimum_mcs_objective_literal__(9999999, mus(934)).
__optimum_mcs_objective_literal__(9999999, mus(935)).
__optimum_mcs_objective_literal__(9999999, mus(936)).
__optimum_mcs_objective_literal__(9999999, mus(937)).
__optimum_mcs_objective_literal__(9999999, mus(938)).
__optimum_mcs_objective_literal__(9999999, mus(939)).
__optimum_mcs_objective_literal__(9999999, mus(940)).
__optimum_mcs_objective_literal__(9999999, mus(941)).
__optimum_mcs_objective_literal__(9999999, mus(942)).
__optimum_mcs_objective_literal__(9999999, mus(943)).
__optimum_mcs_objective_literal__(9999999, mus(944)).
__optimum_mcs_objective_literal__(9999999, mus(945)).
__optimum_mcs_objective_literal__(9999999, mus(946)).
__optimum_mcs_objective_literal__(9999999, mus(947)).
__optimum_mcs_objective_literal__(9999999, mus(948)).
__optimum_mcs_objective_literal__(9999999, mus(949)).
__optimum_mcs_objective_literal__(9999999, mus(950)).
__optimum_mcs_objective_literal__(9999999, mus(951)).
__optimum_mcs_objective_literal__(9999999, mus(952)).
__optimum_mcs_objective_literal__(9999999, mus(953)).
__optimum_mcs_objective_literal__(9999999, mus(954)).
__optimum_mcs_objective_literal__(9999999, mus(955)).
__optimum_mcs_objective_literal__(9999999, mus(956)).
__optimum_mcs_objective_literal__(9999995, mus(957)).
__optimum_mcs_objective_literal__(9999999, mus(958)).
__optimum_mcs_objective_literal__(9999999, mus(959)).
__optimum_mcs_objective_literal__(9999999, mus(960)).
__optimum_mcs_objective_literal__(8688354, mus(961)).
__optimum_mcs_objective_literal__(9999999, mus(962)).
__optimum_mcs_objective_literal__(9999999, mus(963)).
__optimum_mcs_objective_literal__(9999963, mus(964)).
__optimum_mcs_objective_literal__(9999999, mus(965)).
__optimum_mcs_objective_literal__(9999999, mus(966)).
__optimum_mcs_objective_literal__(9999999, mus(967)).
__optimum_mcs_objective_literal__(9999999, mus(968)).
__optimum_mcs_objective_literal__(9699634, mus(969)).
__optimum_mcs_objective_literal__(9999999, mus(970)).
__optimum_mcs_objective_literal__(9999999, mus(971)).
__optimum_mcs_objective_literal__(9999999, mus(972)).
__optimum_mcs_objective_literal__(9999999, mus(973)).
__optimum_mcs_objective_literal__(9999999, mus(974)).
__optimum_mcs_objective_literal__(9443951, mus(975)).
__optimum_mcs_objective_literal__(7588763, mus(976)).
__optimum_mcs_objective_literal__(7290339, mus(977)).
__optimum_mcs_objective_literal__(9187043, mus(978)).
__optimum_mcs_objective_literal__(5163017, mus(979)).
__optimum_mcs_objective_literal__(9479301, mus(980)).
__optimum_mcs_objective_literal__(7262608, mus(981)).
__optimum_mcs_objective_literal__(9958381, mus(982)).
__optimum_mcs_objective_literal__(6853662, mus(983)).
__optimum_mcs_objective_literal__(9999999, mus(984)).
__optimum_mcs_objective_literal__(9999999, mus(985)).
__optimum_mcs_objective_literal__(9999999, mus(986)).
__optimum_mcs_objective_literal__(9999999, mus(987)).
__optimum_mcs_objective_literal__(9999999, mus(988)).
__optimum_mcs_objective_literal__(9999999, mus(989)).
__optimum_mcs_objective_literal__(9999999, mus(990)).
__optimum_mcs_objective_literal__(9999999, mus(991)).
__optimum_mcs_objective_literal__(5175538, mus(992)).
__optimum_mcs_objective_literal__(9999999, mus(993)).
__optimum_mcs_objective_literal__(9999999, mus(994)).
__optimum_mcs_objective_literal__(9999999, mus(995)).
__optimum_mcs_objective_literal__(8253702, mus(996)).
__optimum_mcs_objective_literal__(9999999, mus(997)).
__optimum_mcs_objective_literal__(7556884, mus(998)).
__optimum_mcs_objective_literal__(9999999, mus(999)).
__optimum_mcs_objective_literal__(9999999, mus(1000)).
__optimum_mcs_objective_literal__(9999999, mus(1001)).
__optimum_mcs_objective_literal__(9999999, mus(1002)).
__optimum_mcs_objective_literal__(9999999, mus(1003)).
__optimum_mcs_objective_literal__(9999999, mus(1004)).
__optimum_mcs_objective_literal__(9999999, mus(1005)).
__optimum_mcs_objective_literal__(9999999, mus(1006)).
__optimum_mcs_objective_literal__(9999999, mus(1007)).
__optimum_mcs_objective_literal__(9999999, mus(1008)).
__optimum_mcs_objective_literal__(9999999, mus(1009)).
__optimum_mcs_objective_literal__(9999999, mus(1010)).
__optimum_mcs_objective_literal__(9999999, mus(1011)).
__optimum_mcs_objective_literal__(9999999, mus(1012)).
__optimum_mcs_objective_literal__(9999999, mus(1013)).
__optimum_mcs_objective_literal__(9999999, mus(1014)).
__optimum_mcs_objective_literal__(9999999, mus(1015)).
__optimum_mcs_objective_literal__(9999999, mus(1016)).
__optimum_mcs_objective_literal__(9999999, mus(1017)).
__optimum_mcs_objective_literal__(9999999, mus(1018)).
__optimum_mcs_objective_literal__(9999999, mus(1019)).
__optimum_mcs_objective_literal__(9999999, mus(1020)).
__optimum_mcs_objective_literal__(9999999, mus(1021)).
__optimum_mcs_objective_literal__(9999999, mus(1022)).
__optimum_mcs_objective_literal__(9999999, mus(1023)).
__optimum_mcs_objective_literal__(9999999, mus(1024)).
__optimum_mcs_objective_literal__(9999999, mus(1025)).
__optimum_mcs_objective_literal__(9999999, mus(1026)).
__optimum_mcs_objective_literal__(9999999, mus(1027)).
__optimum_mcs_objective_literal__(9999999, mus(1028)).
__optimum_mcs_objective_literal__(9999999, mus(1029)).
__optimum_mcs_objective_literal__(9999999, mus(1030)).
__optimum_mcs_objective_literal__(9999999, mus(1031)).
__optimum_mcs_objective_literal__(9999999, mus(1032)).
__optimum_mcs_objective_literal__(9999999, mus(1033)).
__optimum_mcs_objective_literal__(9999999, mus(1034)).
__optimum_mcs_objective_literal__(9999999, mus(1035)).
__optimum_mcs_objective_literal__(9999999, mus(1036)).
__optimum_mcs_objective_literal__(9999999, mus(1037)).
__optimum_mcs_objective_literal__(9999999, mus(1038)).
__optimum_mcs_objective_literal__(9999999, mus(1039)).
__optimum_mcs_objective_literal__(9999999, mus(1040)).
__optimum_mcs_objective_literal__(9999999, mus(1041)).
__optimum_mcs_objective_literal__(9999999, mus(1042)).
__optimum_mcs_objective_literal__(9999999, mus(1043)).
__optimum_mcs_objective_literal__(9999999, mus(1044)).
__optimum_mcs_objective_literal__(9999999, mus(1045)).
__optimum_mcs_objective_literal__(9999999, mus(1046)).
__optimum_mcs_objective_literal__(9999999, mus(1047)).
__optimum_mcs_objective_literal__(9999999, mus(1048)).
__optimum_mcs_objective_literal__(9999999, mus(1049)).
__optimum_mcs_objective_literal__(9999999, mus(1050)).
__optimum_mcs_objective_literal__(9999999, mus(1051)).
__optimum_mcs_objective_literal__(9999999, mus(1052)).
__optimum_mcs_objective_literal__(9999999, mus(1053)).
__optimum_mcs_objective_literal__(9999999, mus(1054)).
__optimum_mcs_objective_literal__(9999999, mus(1055)).
__optimum_mcs_objective_literal__(9999999, mus(1056)).
__optimum_mcs_objective_literal__(9999999, mus(1057)).
__optimum_mcs_objective_literal__(9999999, mus(1058)).
__optimum_mcs_objective_literal__(9999999, mus(1059)).
__optimum_mcs_objective_literal__(9999999, mus(1060)).
__optimum_mcs_objective_literal__(9999999, mus(1061)).
__optimum_mcs_objective_literal__(9999999, mus(1062)).
__optimum_mcs_objective_literal__(9999999, mus(1063)).
__optimum_mcs_objective_literal__(9999999, mus(1064)).
__optimum_mcs_objective_literal__(9999999, mus(1065)).
__optimum_mcs_objective_literal__(9999999, mus(1066)).
__optimum_mcs_objective_literal__(9999999, mus(1067)).
__optimum_mcs_objective_literal__(9999999, mus(1068)).
__optimum_mcs_objective_literal__(9999999, mus(1069)).
__optimum_mcs_objective_literal__(9999999, mus(1070)).
__optimum_mcs_objective_literal__(9999999, mus(1071)).
__optimum_mcs_objective_literal__(9999999, mus(1072)).
__optimum_mcs_objective_literal__(9999999, mus(1073)).
__optimum_mcs_objective_literal__(9999999, mus(1074)).
__optimum_mcs_objective_literal__(9999999, mus(1075)).
__optimum_mcs_objective_literal__(9999999, mus(1076)).
__optimum_mcs_objective_literal__(9999999, mus(1077)).
__optimum_mcs_objective_literal__(9999999, mus(1078)).
__optimum_mcs_objective_literal__(9999999, mus(1079)).
__optimum_mcs_objective_literal__(9999999, mus(1080)).
__optimum_mcs_objective_literal__(9999999, mus(1081)).
__optimum_mcs_objective_literal__(9999999, mus(1082)).
__optimum_mcs_objective_literal__(9999999, mus(1083)).
__optimum_mcs_objective_literal__(9999999, mus(1084)).
__optimum_mcs_objective_literal__(9999999, mus(1085)).
__optimum_mcs_objective_literal__(9999999, mus(1086)).
__optimum_mcs_objective_literal__(9999999, mus(1087)).
__optimum_mcs_objective_literal__(9999999, mus(1088)).
__optimum_mcs_objective_literal__(9999999, mus(1089)).
__optimum_mcs_objective_literal__(9999999, mus(1090)).
__optimum_mcs_objective_literal__(9999999, mus(1091)).
__optimum_mcs_objective_literal__(9999999, mus(1092)).
__optimum_mcs_objective_literal__(9999999, mus(1093)).
__optimum_mcs_objective_literal__(9999999, mus(1094)).
__optimum_mcs_objective_literal__(9999999, mus(1095)).
__optimum_mcs_objective_literal__(9999999, mus(1096)).
__optimum_mcs_objective_literal__(9999999, mus(1097)).
__optimum_mcs_objective_literal__(9999999, mus(1098)).
__optimum_mcs_objective_literal__(9999999, mus(1099)).
__optimum_mcs_objective_literal__(9999999, mus(1100)).
__optimum_mcs_objective_literal__(9999999, mus(1101)).
__optimum_mcs_objective_literal__(9999999, mus(1102)).
__optimum_mcs_objective_literal__(9999999, mus(1103)).
__optimum_mcs_objective_literal__(9999999, mus(1104)).
__optimum_mcs_objective_literal__(9999999, mus(1105)).
__optimum_mcs_objective_literal__(9999999, mus(1106)).
__optimum_mcs_objective_literal__(9999999, mus(1107)).
__optimum_mcs_objective_literal__(9999999, mus(1108)).
__optimum_mcs_objective_literal__(9999999, mus(1109)).
__optimum_mcs_objective_literal__(9999999, mus(1110)).
__optimum_mcs_objective_literal__(9999999, mus(1111)).
__optimum_mcs_objective_literal__(9999999, mus(1112)).
__optimum_mcs_objective_literal__(9999999, mus(1113)).
__optimum_mcs_objective_literal__(9999999, mus(1114)).
__optimum_mcs_objective_literal__(9999999, mus(1115)).
__optimum_mcs_objective_literal__(9999999, mus(1116)).
__optimum_mcs_objective_literal__(9999999, mus(1117)).
__optimum_mcs_objective_literal__(9999999, mus(1118)).
__optimum_mcs_objective_literal__(9999999, mus(1119)).
__optimum_mcs_objective_literal__(9999999, mus(1120)).
__optimum_mcs_objective_literal__(9999999, mus(1121)).
__optimum_mcs_objective_literal__(9999999, mus(1122)).
__optimum_mcs_objective_literal__(9999999, mus(1123)).
__optimum_mcs_objective_literal__(9999999, mus(1124)).
__optimum_mcs_objective_literal__(9999999, mus(1125)).
__optimum_mcs_objective_literal__(9999999, mus(1126)).
__optimum_mcs_objective_literal__(9999999, mus(1127)).
__optimum_mcs_objective_literal__(9999999, mus(1128)).
__optimum_mcs_objective_literal__(9999999, mus(1129)).
__optimum_mcs_objective_literal__(9999999, mus(1130)).
__optimum_mcs_objective_literal__(9999999, mus(1131)).
__optimum_mcs_objective_literal__(9999999, mus(1132)).
__optimum_mcs_objective_literal__(9999999, mus(1133)).
__optimum_mcs_objective_literal__(9999999, mus(1134)).
__optimum_mcs_objective_literal__(9999999, mus(1135)).
__optimum_mcs_objective_literal__(9999999, mus(1136)).
__optimum_mcs_objective_literal__(9999999, mus(1137)).
__optimum_mcs_objective_literal__(9999999, mus(1138)).
__optimum_mcs_objective_literal__(9999999, mus(1139)).
__optimum_mcs_objective_literal__(9999999, mus(1140)).
__optimum_mcs_objective_literal__(9999999, mus(1141)).
__optimum_mcs_objective_literal__(9999999, mus(1142)).
__optimum_mcs_objective_literal__(9999999, mus(1143)).
__optimum_mcs_objective_literal__(9999999, mus(1144)).
__optimum_mcs_objective_literal__(9999999, mus(1145)).
__optimum_mcs_objective_literal__(9999999, mus(1146)).
__optimum_mcs_objective_literal__(9999999, mus(1147)).
__optimum_mcs_objective_literal__(9999999, mus(1148)).
__optimum_mcs_objective_literal__(9999999, mus(1149)).
__optimum_mcs_objective_literal__(9999999, mus(1150)).
__optimum_mcs_objective_literal__(9999999, mus(1151)).
__optimum_mcs_objective_literal__(9999999, mus(1152)).
__optimum_mcs_objective_literal__(9999999, mus(1153)).
__optimum_mcs_objective_literal__(9999999, mus(1154)).
__optimum_mcs_objective_literal__(9999999, mus(1155)).
__optimum_mcs_objective_literal__(9999993, mus(1156)).
__optimum_mcs_objective_literal__(9999999, mus(1157)).
__optimum_mcs_objective_literal__(9999999, mus(1158)).
__optimum_mcs_objective_literal__(9999999, mus(1159)).
__optimum_mcs_objective_literal__(9999999, mus(1160)).
__optimum_mcs_objective_literal__(9999999, mus(1161)).
__optimum_mcs_objective_literal__(9999857, mus(1162)).
__optimum_mcs_objective_literal__(9999966, mus(1163)).
__optimum_mcs_objective_literal__(9867057, mus(1164)).
__optimum_mcs_objective_literal__(9970444, mus(1165)).
__optimum_mcs_objective_literal__(9820949, mus(1166)).
__optimum_mcs_objective_literal__(5056598, mus(1167)).
__optimum_mcs_objective_literal__(9998959, mus(1168)).
__optimum_mcs_objective_literal__(9881152, mus(1169)).
__optimum_mcs_objective_literal__(5029499, mus(1170)).
__optimum_mcs_objective_literal__(7727180, mus(1171)).
__optimum_mcs_objective_literal__(5193571, mus(1172)).
__optimum_mcs_objective_literal__(9999999, mus(1173)).
__optimum_mcs_objective_literal__(9999999, mus(1174)).
__optimum_mcs_objective_literal__(9999999, mus(1175)).
__optimum_mcs_objective_literal__(9999999, mus(1176)).
__optimum_mcs_objective_literal__(9999999, mus(1177)).
__optimum_mcs_objective_literal__(9999999, mus(1178)).
__optimum_mcs_objective_literal__(9999999, mus(1179)).
__optimum_mcs_objective_literal__(9999999, mus(1180)).
__optimum_mcs_objective_literal__(9999999, mus(1181)).
__optimum_mcs_objective_literal__(9999999, mus(1182)).
__optimum_mcs_objective_literal__(9999999, mus(1183)).
__optimum_mcs_objective_literal__(9999999, mus(1184)).
__optimum_mcs_objective_literal__(9999999, mus(1185)).
__optimum_mcs_objective_literal__(9999999, mus(1186)).
__optimum_mcs_objective_literal__(9999999, mus(1187)).
__optimum_mcs_objective_literal__(9999999, mus(1188)).
__optimum_mcs_objective_literal__(9999999, mus(1189)).
__optimum_mcs_objective_literal__(9999999, mus(1190)).
__optimum_mcs_objective_literal__(9999999, mus(1191)).
__optimum_mcs_objective_literal__(9999999, mus(1192)).
__optimum_mcs_objective_literal__(9999999, mus(1193)).
__optimum_mcs_objective_literal__(9999999, mus(1194)).
__optimum_mcs_objective_literal__(9999999, mus(1195)).
__optimum_mcs_objective_literal__(9999999, mus(1196)).
__optimum_mcs_objective_literal__(9999999, mus(1197)).
__optimum_mcs_objective_literal__(9999999, mus(1198)).
__optimum_mcs_objective_literal__(9999984, mus(1199)).
__optimum_mcs_objective_literal__(9999999, mus(1200)).
__optimum_mcs_objective_literal__(9999999, mus(1201)).
__optimum_mcs_objective_literal__(9999999, mus(1202)).
__optimum_mcs_objective_literal__(9999999, mus(1203)).
__optimum_mcs_objective_literal__(9999999, mus(1204)).
__optimum_mcs_objective_literal__(9999997, mus(1205)).
__optimum_mcs_objective_literal__(9999999, mus(1206)).
__optimum_mcs_objective_literal__(9999999, mus(1207)).
__optimum_mcs_objective_literal__(9999999, mus(1208)).
__optimum_mcs_objective_literal__(9999999, mus(1209)).
__optimum_mcs_objective_literal__(9999999, mus(1210)).
__optimum_mcs_objective_literal__(9999999, mus(1211)).
__optimum_mcs_objective_literal__(9999999, mus(1212)).
__optimum_mcs_objective_literal__(9999999, mus(1213)).
__optimum_mcs_objective_literal__(9999999, mus(1214)).
__optimum_mcs_objective_literal__(9999999, mus(1215)).
__optimum_mcs_objective_literal__(9999999, mus(1216)).
__optimum_mcs_objective_literal__(9999999, mus(1217)).
__optimum_mcs_objective_literal__(9999992, mus(1218)).
__optimum_mcs_objective_literal__(9999999, mus(1219)).
__optimum_mcs_objective_literal__(9999999, mus(1220)).
__optimum_mcs_objective_literal__(9999999, mus(1221)).
__optimum_mcs_objective_literal__(9999999, mus(1222)).
__optimum_mcs_objective_literal__(9999999, mus(1223)).
__optimum_mcs_objective_literal__(9999999, mus(1224)).
__optimum_mcs_objective_literal__(9999999, mus(1225)).
__optimum_mcs_objective_literal__(9998304, mus(1226)).
__optimum_mcs_objective_literal__(9999999, mus(1227)).
__optimum_mcs_objective_literal__(9999999, mus(1228)).
__optimum_mcs_objective_literal__(9999999, mus(1229)).
__optimum_mcs_objective_literal__(9999999, mus(1230)).
__optimum_mcs_objective_literal__(9999965, mus(1231)).
__optimum_mcs_objective_literal__(9999999, mus(1232)).
__optimum_mcs_objective_literal__(9999999, mus(1233)).
__optimum_mcs_objective_literal__(9999999, mus(1234)).
__optimum_mcs_objective_literal__(9999999, mus(1235)).
__optimum_mcs_objective_literal__(9999999, mus(1236)).
__optimum_mcs_objective_literal__(9999999, mus(1237)).
__optimum_mcs_objective_literal__(9999999, mus(1238)).
__optimum_mcs_objective_literal__(9998302, mus(1239)).
__optimum_mcs_objective_literal__(9999999, mus(1240)).
__optimum_mcs_objective_literal__(9999900, mus(1241)).
__optimum_mcs_objective_literal__(9999999, mus(1242)).
__optimum_mcs_objective_literal__(9999999, mus(1243)).
__optimum_mcs_objective_literal__(9999999, mus(1244)).
__optimum_mcs_objective_literal__(9999999, mus(1245)).
__optimum_mcs_objective_literal__(9999991, mus(1246)).
__optimum_mcs_objective_literal__(9999999, mus(1247)).
__optimum_mcs_objective_literal__(9999999, mus(1248)).
__optimum_mcs_objective_literal__(9999999, mus(1249)).
__optimum_mcs_objective_literal__(9999999, mus(1250)).
__optimum_mcs_objective_literal__(9999999, mus(1251)).
__optimum_mcs_objective_literal__(9999999, mus(1252)).
__optimum_mcs_objective_literal__(9999999, mus(1253)).
__optimum_mcs_objective_literal__(9999901, mus(1254)).
__optimum_mcs_objective_literal__(9999999, mus(1255)).
__optimum_mcs_objective_literal__(9999286, mus(1256)).
__optimum_mcs_objective_literal__(9999999, mus(1257)).
__optimum_mcs_objective_literal__(9999999, mus(1258)).
__optimum_mcs_objective_literal__(9999999, mus(1259)).
__optimum_mcs_objective_literal__(9999999, mus(1260)).
__optimum_mcs_objective_literal__(9999999, mus(1261)).
__optimum_mcs_objective_literal__(9999999, mus(1262)).
__optimum_mcs_objective_literal__(9999999, mus(1263)).
__optimum_mcs_objective_literal__(9997368, mus(1264)).
__optimum_mcs_objective_literal__(9999999, mus(1265)).
__optimum_mcs_objective_literal__(9999708, mus(1266)).
__optimum_mcs_objective_literal__(9999999, mus(1267)).
__optimum_mcs_objective_literal__(9999999, mus(1268)).
__optimum_mcs_objective_literal__(9999999, mus(1269)).
__optimum_mcs_objective_literal__(9999999, mus(1270)).
__optimum_mcs_objective_literal__(9999999, mus(1271)).
__optimum_mcs_objective_literal__(9999999, mus(1272)).
__optimum_mcs_objective_literal__(9999999, mus(1273)).
__optimum_mcs_objective_literal__(9999999, mus(1274)).
__optimum_mcs_objective_literal__(9999999, mus(1275)).
__optimum_mcs_objective_literal__(9999999, mus(1276)).
__optimum_mcs_objective_literal__(9997247, mus(1277)).
__optimum_mcs_objective_literal__(9999999, mus(1278)).
__optimum_mcs_objective_literal__(9999999, mus(1279)).
__optimum_mcs_objective_literal__(9999999, mus(1280)).
__optimum_mcs_objective_literal__(9999999, mus(1281)).
__optimum_mcs_objective_literal__(9999999, mus(1282)).
__optimum_mcs_objective_literal__(9999999, mus(1283)).
__optimum_mcs_objective_literal__(9999999, mus(1284)).
__optimum_mcs_objective_literal__(9999999, mus(1285)).
__optimum_mcs_objective_literal__(9999999, mus(1286)).
__optimum_mcs_objective_literal__(9999996, mus(1287)).
__optimum_mcs_objective_literal__(9999999, mus(1288)).
__optimum_mcs_objective_literal__(9999998, mus(1289)).
__optimum_mcs_objective_literal__(9999999, mus(1290)).
__optimum_mcs_objective_literal__(9999999, mus(1291)).
__optimum_mcs_objective_literal__(9999999, mus(1292)).
__optimum_mcs_objective_literal__(9999986, mus(1293)).
__optimum_mcs_objective_literal__(9999996, mus(1294)).
__optimum_mcs_objective_literal__(9999999, mus(1295)).
__optimum_mcs_objective_literal__(9999999, mus(1296)).
__optimum_mcs_objective_literal__(9999999, mus(1297)).
__optimum_mcs_objective_literal__(9999999, mus(1298)).
__optimum_mcs_objective_literal__(9999965, mus(1299)).
__optimum_mcs_objective_literal__(9999999, mus(1300)).
__optimum_mcs_objective_literal__(9999999, mus(1301)).
__optimum_mcs_objective_literal__(9999999, mus(1302)).
__optimum_mcs_objective_literal__(9999996, mus(1303)).
__optimum_mcs_objective_literal__(9999893, mus(1304)).
__optimum_mcs_objective_literal__(9999924, mus(1305)).
__optimum_mcs_objective_literal__(9999999, mus(1306)).
__optimum_mcs_objective_literal__(9999999, mus(1307)).
__optimum_mcs_objective_literal__(9999999, mus(1308)).
__optimum_mcs_objective_literal__(9971177, mus(1309)).
__optimum_mcs_objective_literal__(9999999, mus(1310)).
__optimum_mcs_objective_literal__(9999999, mus(1311)).
__optimum_mcs_objective_literal__(9999999, mus(1312)).
__optimum_mcs_objective_literal__(9983788, mus(1313)).
__optimum_mcs_objective_literal__(9999999, mus(1314)).
__optimum_mcs_objective_literal__(9999999, mus(1315)).
__optimum_mcs_objective_literal__(9999999, mus(1316)).
__optimum_mcs_objective_literal__(9999999, mus(1317)).
__optimum_mcs_objective_literal__(9999999, mus(1318)).
__optimum_mcs_objective_literal__(9999999, mus(1319)).
__optimum_mcs_objective_literal__(9999991, mus(1320)).
__optimum_mcs_objective_literal__(9999999, mus(1321)).
__optimum_mcs_objective_literal__(9999999, mus(1322)).
__optimum_mcs_objective_literal__(9999999, mus(1323)).
__optimum_mcs_objective_literal__(9999999, mus(1324)).
__optimum_mcs_objective_literal__(9999999, mus(1325)).
__optimum_mcs_objective_literal__(9999999, mus(1326)).
__optimum_mcs_objective_literal__(9999999, mus(1327)).
__optimum_mcs_objective_literal__(9999999, mus(1328)).
__optimum_mcs_objective_literal__(9999999, mus(1329)).
__optimum_mcs_objective_literal__(9999999, mus(1330)).
__optimum_mcs_objective_literal__(9999999, mus(1331)).
__optimum_mcs_objective_literal__(9999999, mus(1332)).
__optimum_mcs_objective_literal__(9999957, mus(1333)).
__optimum_mcs_objective_literal__(9999999, mus(1334)).
__optimum_mcs_objective_literal__(9999999, mus(1335)).
__optimum_mcs_objective_literal__(9999999, mus(1336)).
__optimum_mcs_objective_literal__(9999999, mus(1337)).
__optimum_mcs_objective_literal__(9999999, mus(1338)).
__optimum_mcs_objective_literal__(9999999, mus(1339)).
__optimum_mcs_objective_literal__(9999999, mus(1340)).
__optimum_mcs_objective_literal__(9999999, mus(1341)).
__optimum_mcs_objective_literal__(9999999, mus(1342)).
__optimum_mcs_objective_literal__(9999999, mus(1343)).
__optimum_mcs_objective_literal__(9994631, mus(1344)).
__optimum_mcs_objective_literal__(9999045, mus(1345)).
__optimum_mcs_objective_literal__(9955337, mus(1346)).
__optimum_mcs_objective_literal__(9999282, mus(1347)).
__optimum_mcs_objective_literal__(9999999, mus(1348)).
__optimum_mcs_objective_literal__(9999999, mus(1349)).
__optimum_mcs_objective_literal__(9999999, mus(1350)).
__optimum_mcs_objective_literal__(9999999, mus(1351)).
__optimum_mcs_objective_literal__(9999999, mus(1352)).
__optimum_mcs_objective_literal__(9999999, mus(1353)).
__optimum_mcs_objective_literal__(9999999, mus(1354)).
__optimum_mcs_objective_literal__(9999999, mus(1355)).
__optimum_mcs_objective_literal__(9999999, mus(1356)).
__optimum_mcs_objective_literal__(9999999, mus(1357)).
__optimum_mcs_objective_literal__(9999999, mus(1358)).
__optimum_mcs_objective_literal__(9999999, mus(1359)).
__optimum_mcs_objective_literal__(9999999, mus(1360)).
__optimum_mcs_objective_literal__(9999999, mus(1361)).
__optimum_mcs_objective_literal__(9999999, mus(1362)).
__optimum_mcs_objective_literal__(9999999, mus(1363)).
__optimum_mcs_objective_literal__(9999999, mus(1364)).
__optimum_mcs_objective_literal__(9999999, mus(1365)).
__optimum_mcs_objective_literal__(9999999, mus(1366)).
__optimum_mcs_objective_literal__(9999999, mus(1367)).
__optimum_mcs_objective_literal__(9999999, mus(1368)).
__optimum_mcs_objective_literal__(9999999, mus(1369)).
__optimum_mcs_objective_literal__(9999999, mus(1370)).
__optimum_mcs_objective_literal__(9999999, mus(1371)).
__optimum_mcs_objective_literal__(9999999, mus(1372)).
__optimum_mcs_objective_literal__(9999999, mus(1373)).
__optimum_mcs_objective_literal__(9999999, mus(1374)).
__optimum_mcs_objective_literal__(9999999, mus(1375)).
__optimum_mcs_objective_literal__(9999999, mus(1376)).
__optimum_mcs_objective_literal__(9999999, mus(1377)).
__optimum_mcs_objective_literal__(9999999, mus(1378)).
__optimum_mcs_objective_literal__(9999999, mus(1379)).
__optimum_mcs_objective_literal__(9999999, mus(1380)).
__optimum_mcs_objective_literal__(9999999, mus(1381)).
__optimum_mcs_objective_literal__(9999999, mus(1382)).
__optimum_mcs_objective_literal__(9999999, mus(1383)).
__optimum_mcs_objective_literal__(9999999, mus(1384)).
__optimum_mcs_objective_literal__(9999999, mus(1385)).
__optimum_mcs_objective_literal__(9999999, mus(1386)).
__optimum_mcs_objective_literal__(9999999, mus(1387)).
__optimum_mcs_objective_literal__(9999999, mus(1388)).
__optimum_mcs_objective_literal__(9999999, mus(1389)).
__optimum_mcs_objective_literal__(9999999, mus(1390)).
__optimum_mcs_objective_literal__(9999999, mus(1391)).
__optimum_mcs_objective_literal__(9999999, mus(1392)).
__optimum_mcs_objective_literal__(9999999, mus(1393)).
__optimum_mcs_objective_literal__(9999999, mus(1394)).
__optimum_mcs_objective_literal__(9999999, mus(1395)).
__optimum_mcs_objective_literal__(9999999, mus(1396)).
__optimum_mcs_objective_literal__(9999999, mus(1397)).
__optimum_mcs_objective_literal__(9999999, mus(1398)).
__optimum_mcs_objective_literal__(9999999, mus(1399)).
__optimum_mcs_objective_literal__(9999999, mus(1400)).
__optimum_mcs_objective_literal__(9999999, mus(1401)).
__optimum_mcs_objective_literal__(9999999, mus(1402)).
__optimum_mcs_objective_literal__(9999999, mus(1403)).
__optimum_mcs_objective_literal__(9999999, mus(1404)).
__optimum_mcs_objective_literal__(9999999, mus(1405)).
__optimum_mcs_objective_literal__(9999999, mus(1406)).
__optimum_mcs_objective_literal__(9999999, mus(1407)).
__optimum_mcs_objective_literal__(9999999, mus(1408)).
__optimum_mcs_objective_literal__(9999999, mus(1409)).
__optimum_mcs_objective_literal__(9999999, mus(1410)).
__optimum_mcs_objective_literal__(9999999, mus(1411)).
__optimum_mcs_objective_literal__(9999999, mus(1412)).
__optimum_mcs_objective_literal__(9999999, mus(1413)).
__optimum_mcs_objective_literal__(9999999, mus(1414)).
__optimum_mcs_objective_literal__(9999999, mus(1415)).
__optimum_mcs_objective_literal__(9999999, mus(1416)).
__optimum_mcs_objective_literal__(9999999, mus(1417)).
__optimum_mcs_objective_literal__(9999999, mus(1418)).
__optimum_mcs_objective_literal__(9999999, mus(1419)).
__optimum_mcs_objective_literal__(9999999, mus(1420)).
__optimum_mcs_objective_literal__(9999999, mus(1421)).
__optimum_mcs_objective_literal__(9999999, mus(1422)).
__optimum_mcs_objective_literal__(9999999, mus(1423)).
__optimum_mcs_objective_literal__(9999999, mus(1424)).
__optimum_mcs_objective_literal__(9999999, mus(1425)).
__optimum_mcs_objective_literal__(9999999, mus(1426)).
__optimum_mcs_objective_literal__(9999999, mus(1427)).
__optimum_mcs_objective_literal__(9999999, mus(1428)).
__optimum_mcs_objective_literal__(9999999, mus(1429)).
__optimum_mcs_objective_literal__(9999999, mus(1430)).
__optimum_mcs_objective_literal__(9999999, mus(1431)).
__optimum_mcs_objective_literal__(9999999, mus(1432)).
__optimum_mcs_objective_literal__(9999999, mus(1433)).
__optimum_mcs_objective_literal__(9999999, mus(1434)).
__optimum_mcs_objective_literal__(9999999, mus(1435)).
__optimum_mcs_objective_literal__(9999999, mus(1436)).
__optimum_mcs_objective_literal__(9999999, mus(1437)).
__optimum_mcs_objective_literal__(9999999, mus(1438)).
__optimum_mcs_objective_literal__(9999999, mus(1439)).
__optimum_mcs_objective_literal__(9999999, mus(1440)).
__optimum_mcs_objective_literal__(9999999, mus(1441)).
__optimum_mcs_objective_literal__(9999999, mus(1442)).
__optimum_mcs_objective_literal__(9999999, mus(1443)).
__optimum_mcs_objective_literal__(9999999, mus(1444)).
__optimum_mcs_objective_literal__(9999999, mus(1445)).
__optimum_mcs_objective_literal__(9999999, mus(1446)).
__optimum_mcs_objective_literal__(9999999, mus(1447)).
__optimum_mcs_objective_literal__(9999999, mus(1448)).
__optimum_mcs_objective_literal__(9999999, mus(1449)).
__optimum_mcs_objective_literal__(9999999, mus(1450)).
__optimum_mcs_objective_literal__(9999999, mus(1451)).
__optimum_mcs_objective_literal__(9999999, mus(1452)).
__optimum_mcs_objective_literal__(9999999, mus(1453)).
__optimum_mcs_objective_literal__(9999999, mus(1454)).
__optimum_mcs_objective_literal__(9999999, mus(1455)).
__optimum_mcs_objective_literal__(9999999, mus(1456)).
__optimum_mcs_objective_literal__(9999999, mus(1457)).
__optimum_mcs_objective_literal__(9999999, mus(1458)).
__optimum_mcs_objective_literal__(9999999, mus(1459)).
__optimum_mcs_objective_literal__(9999999, mus(1460)).
__optimum_mcs_objective_literal__(9999999, mus(1461)).
__optimum_mcs_objective_literal__(9999999, mus(1462)).
__optimum_mcs_objective_literal__(9999999, mus(1463)).
__optimum_mcs_objective_literal__(9999999, mus(1464)).
__optimum_mcs_objective_literal__(9999999, mus(1465)).
__optimum_mcs_objective_literal__(9999999, mus(1466)).
__optimum_mcs_objective_literal__(9999999, mus(1467)).
__optimum_mcs_objective_literal__(9999999, mus(1468)).
__optimum_mcs_objective_literal__(9999999, mus(1469)).
__optimum_mcs_objective_literal__(9999999, mus(1470)).
__optimum_mcs_objective_literal__(9999999, mus(1471)).
__optimum_mcs_objective_literal__(9999999, mus(1472)).
__optimum_mcs_objective_literal__(9999999, mus(1473)).
__optimum_mcs_objective_literal__(9999999, mus(1474)).
__optimum_mcs_objective_literal__(9999999, mus(1475)).
__optimum_mcs_objective_literal__(9999999, mus(1476)).
__optimum_mcs_objective_literal__(9999999, mus(1477)).
__optimum_mcs_objective_literal__(9999999, mus(1478)).
__optimum_mcs_objective_literal__(9999999, mus(1479)).
__optimum_mcs_objective_literal__(9999999, mus(1480)).
__optimum_mcs_objective_literal__(9999999, mus(1481)).
__optimum_mcs_objective_literal__(9999999, mus(1482)).
__optimum_mcs_objective_literal__(9999999, mus(1483)).
__optimum_mcs_objective_literal__(9999999, mus(1484)).
__optimum_mcs_objective_literal__(9999999, mus(1485)).
__optimum_mcs_objective_literal__(9999999, mus(1486)).
__optimum_mcs_objective_literal__(9999999, mus(1487)).
__optimum_mcs_objective_literal__(9999999, mus(1488)).
__optimum_mcs_objective_literal__(9999999, mus(1489)).
__optimum_mcs_objective_literal__(9999999, mus(1490)).
__optimum_mcs_objective_literal__(9999999, mus(1491)).
__optimum_mcs_objective_literal__(9999999, mus(1492)).
__optimum_mcs_objective_literal__(9999999, mus(1493)).
__optimum_mcs_objective_literal__(9999999, mus(1494)).
__optimum_mcs_objective_literal__(9999999, mus(1495)).
__optimum_mcs_objective_literal__(9999999, mus(1496)).
__optimum_mcs_objective_literal__(9999999, mus(1497)).
__optimum_mcs_objective_literal__(9999999, mus(1498)).
__optimum_mcs_objective_literal__(9999999, mus(1499)).
__optimum_mcs_objective_literal__(9999999, mus(1500)).
__optimum_mcs_objective_literal__(9999999, mus(1501)).
__optimum_mcs_objective_literal__(9999999, mus(1502)).
__optimum_mcs_objective_literal__(9999999, mus(1503)).
__optimum_mcs_objective_literal__(9999999, mus(1504)).
__optimum_mcs_objective_literal__(9999999, mus(1505)).
__optimum_mcs_objective_literal__(9999999, mus(1506)).
__optimum_mcs_objective_literal__(9999999, mus(1507)).
__optimum_mcs_objective_literal__(9999999, mus(1508)).
__optimum_mcs_objective_literal__(9999999, mus(1509)).
__optimum_mcs_objective_literal__(9999999, mus(1510)).
__optimum_mcs_objective_literal__(9999999, mus(1511)).
__optimum_mcs_objective_literal__(9999999, mus(1512)).
__optimum_mcs_objective_literal__(9999999, mus(1513)).
__optimum_mcs_objective_literal__(9999999, mus(1514)).
__optimum_mcs_objective_literal__(9999999, mus(1515)).
__optimum_mcs_objective_literal__(9999999, mus(1516)).
__optimum_mcs_objective_literal__(9999999, mus(1517)).
__optimum_mcs_objective_literal__(9999999, mus(1518)).
__optimum_mcs_objective_literal__(9999999, mus(1519)).
__optimum_mcs_objective_literal__(9999999, mus(1520)).
__optimum_mcs_objective_literal__(9999999, mus(1521)).
__optimum_mcs_objective_literal__(9999999, mus(1522)).
__optimum_mcs_objective_literal__(9999999, mus(1523)).
__optimum_mcs_objective_literal__(9999999, mus(1524)).
__optimum_mcs_objective_literal__(9999999, mus(1525)).
__optimum_mcs_objective_literal__(9999999, mus(1526)).
__optimum_mcs_objective_literal__(9999999, mus(1527)).
__optimum_mcs_objective_literal__(5114409, mus(1528)).
__optimum_mcs_objective_literal__(9999999, mus(1529)).
__optimum_mcs_objective_literal__(9999999, mus(1530)).
__optimum_mcs_objective_literal__(9999999, mus(1531)).
__optimum_mcs_objective_literal__(9999999, mus(1532)).
__optimum_mcs_objective_literal__(9999999, mus(1533)).
__optimum_mcs_objective_literal__(9999999, mus(1534)).
__optimum_mcs_objective_literal__(9999999, mus(1535)).
__optimum_mcs_objective_literal__(9999999, mus(1536)).
__optimum_mcs_objective_literal__(9999999, mus(1537)).
__optimum_mcs_objective_literal__(9999999, mus(1538)).
__optimum_mcs_objective_literal__(9999999, mus(1539)).
__optimum_mcs_objective_literal__(9999999, mus(1540)).
__optimum_mcs_objective_literal__(9999999, mus(1541)).
__optimum_mcs_objective_literal__(9999999, mus(1542)).
__optimum_mcs_objective_literal__(9999999, mus(1543)).
__optimum_mcs_objective_literal__(9999999, mus(1544)).
__optimum_mcs_objective_literal__(9999999, mus(1545)).
__optimum_mcs_objective_literal__(9999999, mus(1546)).
__optimum_mcs_objective_literal__(9999999, mus(1547)).
__optimum_mcs_objective_literal__(9999999, mus(1548)).
__optimum_mcs_objective_literal__(9999999, mus(1549)).
__optimum_mcs_objective_literal__(9999999, mus(1550)).
__optimum_mcs_objective_literal__(9999999, mus(1551)).
__optimum_mcs_objective_literal__(9999999, mus(1552)).
__optimum_mcs_objective_literal__(9999999, mus(1553)).
__optimum_mcs_objective_literal__(9999999, mus(1554)).
__optimum_mcs_objective_literal__(9999999, mus(1555)).
__optimum_mcs_objective_literal__(9999999, mus(1556)).
__optimum_mcs_objective_literal__(9999999, mus(1557)).
__optimum_mcs_objective_literal__(9999999, mus(1558)).
__optimum_mcs_objective_literal__(9999999, mus(1559)).
__optimum_mcs_objective_literal__(9999999, mus(1560)).
__optimum_mcs_objective_literal__(9999999, mus(1561)).
__optimum_mcs_objective_literal__(9999999, mus(1562)).
__optimum_mcs_objective_literal__(9999999, mus(1563)).
__optimum_mcs_objective_literal__(9999999, mus(1564)).
__optimum_mcs_objective_literal__(9999999, mus(1565)).
__optimum_mcs_objective_literal__(9999999, mus(1566)).
__optimum_mcs_objective_literal__(9999999, mus(1567)).
__optimum_mcs_objective_literal__(9999999, mus(1568)).
__optimum_mcs_objective_literal__(9999999, mus(1569)).
__optimum_mcs_objective_literal__(9999999, mus(1570)).
__optimum_mcs_objective_literal__(9999999, mus(1571)).
__optimum_mcs_objective_literal__(9999999, mus(1572)).
__optimum_mcs_objective_literal__(9999999, mus(1573)).
__optimum_mcs_objective_literal__(9999999, mus(1574)).
__optimum_mcs_objective_literal__(9999999, mus(1575)).
__optimum_mcs_objective_literal__(9999999, mus(1576)).
__optimum_mcs_objective_literal__(9999999, mus(1577)).
__optimum_mcs_objective_literal__(9999999, mus(1578)).
__optimum_mcs_objective_literal__(9999999, mus(1579)).
__optimum_mcs_objective_literal__(9999999, mus(1580)).
__optimum_mcs_objective_literal__(9999999, mus(1581)).
__optimum_mcs_objective_literal__(9999999, mus(1582)).
__optimum_mcs_objective_literal__(9999999, mus(1583)).
__optimum_mcs_objective_literal__(9999999, mus(1584)).
__optimum_mcs_objective_literal__(9999999, mus(1585)).
__optimum_mcs_objective_literal__(9999999, mus(1586)).
__optimum_mcs_objective_literal__(9999999, mus(1587)).
__optimum_mcs_objective_literal__(9999999, mus(1588)).
__optimum_mcs_objective_literal__(9999999, mus(1589)).
__optimum_mcs_objective_literal__(9999999, mus(1590)).
__optimum_mcs_objective_literal__(9999999, mus(1591)).
__optimum_mcs_objective_literal__(9999999, mus(1592)).
__optimum_mcs_objective_literal__(9999999, mus(1593)).
__optimum_mcs_objective_literal__(9999999, mus(1594)).
__optimum_mcs_objective_literal__(9999999, mus(1595)).
__optimum_mcs_objective_literal__(9999999, mus(1596)).
__optimum_mcs_objective_literal__(9999999, mus(1597)).
__optimum_mcs_objective_literal__(9999999, mus(1598)).
__optimum_mcs_objective_literal__(9999999, mus(1599)).
__optimum_mcs_objective_literal__(9999999, mus(1600)).
__optimum_mcs_objective_literal__(9999999, mus(1601)).
__optimum_mcs_objective_literal__(9999999, mus(1602)).
__optimum_mcs_objective_literal__(9999999, mus(1603)).
__optimum_mcs_objective_literal__(9999999, mus(1604)).
__optimum_mcs_objective_literal__(9999999, mus(1605)).
__optimum_mcs_objective_literal__(9999999, mus(1606)).
__optimum_mcs_objective_literal__(9999999, mus(1607)).
__optimum_mcs_objective_literal__(9999999, mus(1608)).
__optimum_mcs_objective_literal__(9999999, mus(1609)).
__optimum_mcs_objective_literal__(9999999, mus(1610)).
__optimum_mcs_objective_literal__(9999999, mus(1611)).
__optimum_mcs_objective_literal__(9999999, mus(1612)).
__optimum_mcs_objective_literal__(9999999, mus(1613)).
__optimum_mcs_objective_literal__(9999999, mus(1614)).
__optimum_mcs_objective_literal__(9999999, mus(1615)).
__optimum_mcs_objective_literal__(9999999, mus(1616)).
__optimum_mcs_objective_literal__(9999999, mus(1617)).
__optimum_mcs_objective_literal__(9999999, mus(1618)).
__optimum_mcs_objective_literal__(9999999, mus(1619)).
__optimum_mcs_objective_literal__(9999999, mus(1620)).
__optimum_mcs_objective_literal__(9999999, mus(1621)).
__optimum_mcs_objective_literal__(9999999, mus(1622)).
__optimum_mcs_objective_literal__(9999999, mus(1623)).
__optimum_mcs_objective_literal__(9999999, mus(1624)).
__optimum_mcs_objective_literal__(9999999, mus(1625)).
__optimum_mcs_objective_literal__(9999999, mus(1626)).
__optimum_mcs_objective_literal__(9999999, mus(1627)).
__optimum_mcs_objective_literal__(9999999, mus(1628)).
__optimum_mcs_objective_literal__(9999999, mus(1629)).
__optimum_mcs_objective_literal__(9999999, mus(1630)).
__optimum_mcs_objective_literal__(9999999, mus(1631)).
__optimum_mcs_objective_literal__(9999999, mus(1632)).
__optimum_mcs_objective_literal__(9999999, mus(1633)).
__optimum_mcs_objective_literal__(9999999, mus(1634)).
__optimum_mcs_objective_literal__(9999999, mus(1635)).
__optimum_mcs_objective_literal__(9999999, mus(1636)).
__optimum_mcs_objective_literal__(9999999, mus(1637)).
__optimum_mcs_objective_literal__(9999999, mus(1638)).
__optimum_mcs_objective_literal__(9999999, mus(1639)).
__optimum_mcs_objective_literal__(9999999, mus(1640)).
__optimum_mcs_objective_literal__(9999999, mus(1641)).
__optimum_mcs_objective_literal__(9999999, mus(1642)).
__optimum_mcs_objective_literal__(9999999, mus(1643)).
__optimum_mcs_objective_literal__(9999999, mus(1644)).
__optimum_mcs_objective_literal__(9999999, mus(1645)).
__optimum_mcs_objective_literal__(9999999, mus(1646)).
__optimum_mcs_objective_literal__(9999999, mus(1647)).
__optimum_mcs_objective_literal__(9999999, mus(1648)).
__optimum_mcs_objective_literal__(9999999, mus(1649)).
__optimum_mcs_objective_literal__(9999999, mus(1650)).
__optimum_mcs_objective_literal__(9999999, mus(1651)).
__optimum_mcs_objective_literal__(9999999, mus(1652)).
__optimum_mcs_objective_literal__(9999999, mus(1653)).
__optimum_mcs_objective_literal__(9999999, mus(1654)).
__optimum_mcs_objective_literal__(9999999, mus(1655)).
__optimum_mcs_objective_literal__(9999999, mus(1656)).
__optimum_mcs_objective_literal__(9999999, mus(1657)).
__optimum_mcs_objective_literal__(9999999, mus(1658)).
__optimum_mcs_objective_literal__(9999999, mus(1659)).
__optimum_mcs_objective_literal__(9999999, mus(1660)).
__optimum_mcs_objective_literal__(9999999, mus(1661)).
__optimum_mcs_objective_literal__(9999999, mus(1662)).
__optimum_mcs_objective_literal__(9999999, mus(1663)).
__optimum_mcs_objective_literal__(9999999, mus(1664)).
__optimum_mcs_objective_literal__(9999999, mus(1665)).
__optimum_mcs_objective_literal__(9999999, mus(1666)).
__optimum_mcs_objective_literal__(9999999, mus(1667)).
__optimum_mcs_objective_literal__(9999999, mus(1668)).
__optimum_mcs_objective_literal__(9999999, mus(1669)).
__optimum_mcs_objective_literal__(9999999, mus(1670)).
__optimum_mcs_objective_literal__(9999999, mus(1671)).
__optimum_mcs_objective_literal__(9999999, mus(1672)).
__optimum_mcs_objective_literal__(9999999, mus(1673)).
__optimum_mcs_objective_literal__(9999999, mus(1674)).
__optimum_mcs_objective_literal__(9999999, mus(1675)).
__optimum_mcs_objective_literal__(9999999, mus(1676)).
__optimum_mcs_objective_literal__(9999999, mus(1677)).
__optimum_mcs_objective_literal__(9999999, mus(1678)).
__optimum_mcs_objective_literal__(9999999, mus(1679)).
__optimum_mcs_objective_literal__(9999999, mus(1680)).
__optimum_mcs_objective_literal__(9999999, mus(1681)).
__optimum_mcs_objective_literal__(9999999, mus(1682)).
__optimum_mcs_objective_literal__(9999999, mus(1683)).
__optimum_mcs_objective_literal__(9999999, mus(1684)).
__optimum_mcs_objective_literal__(9999999, mus(1685)).

:~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]

#show mus/1.
