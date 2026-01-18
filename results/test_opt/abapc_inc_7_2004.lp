% source dump (pre-ground) from causalaba_increm.CausalABA
% note: this is NOT a grounded observer dump
#program main(n_vars).
%%% sets and elements
var(0..n_vars).
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
% facts (as loaded)
#external ext_indep(0,1,empty).
#external ext_indep(0,2,empty).
#external ext_indep(0,3,empty).
#external ext_dep(0,4,empty).
#external ext_indep(0,4,s1).
#external ext_dep(0,4,s2).
#external ext_dep(0,4,s3).
#external ext_dep(0,4,s6).
#external ext_indep(0,5,empty).
#external ext_indep(0,6,empty).
#external ext_indep(1,2,s3).
#external ext_indep(1,2,s4).
#external ext_indep(1,2,s5).
#external ext_indep(1,2,s6).
#external ext_indep(1,2,s3y4).
#external ext_indep(1,2,s3y5).
#external ext_indep(1,2,s4y5).
#external ext_indep(1,2,s4y6).
#external ext_indep(1,2,s3y4y5).
#external ext_indep(1,3,s2).
#external ext_indep(1,3,s4).
#external ext_indep(1,3,s5).
#external ext_indep(1,3,s6).
#external ext_indep(1,3,s2y5).
#external ext_indep(1,3,s2y6).
#external ext_indep(1,3,s4y6).
#external ext_indep(1,3,s5y6).
#external ext_indep(1,3,s2y5y6).
#external ext_dep(1,4,empty).
#external ext_dep(1,4,s0).
#external ext_dep(1,4,s2).
#external ext_dep(1,4,s3).
#external ext_dep(1,4,s6).
#external ext_dep(1,4,s2y3).
#external ext_dep(1,4,s2y6).
#external ext_dep(1,4,s3y6).
#external ext_dep(1,4,s2y3y6).
#external ext_indep(1,5,s2).
#external ext_indep(1,5,s3).
#external ext_indep(1,5,s4).
#external ext_indep(1,5,s6).
#external ext_indep(1,5,s2y3).
#external ext_indep(1,5,s2y6).
#external ext_indep(1,5,s3y6).
#external ext_indep(1,5,s4y6).
#external ext_indep(1,5,s2y3y6).
#external ext_dep(1,6,empty).
#external ext_dep(1,6,s2).
#external ext_dep(1,6,s3).
#external ext_dep(1,6,s4).
#external ext_dep(1,6,s5).
#external ext_dep(1,6,s2y3).
#external ext_dep(1,6,s2y4).
#external ext_dep(1,6,s2y5).
#external ext_dep(1,6,s3y4).
#external ext_dep(1,6,s3y5).
#external ext_dep(1,6,s4y5).
#external ext_dep(1,6,s3y4y5).
#external ext_dep(2,3,empty).
#external ext_dep(2,3,s4).
#external ext_dep(2,3,s5).
#external ext_dep(2,3,s6).
#external ext_dep(2,3,s4y5).
#external ext_dep(2,3,s4y6).
#external ext_dep(2,3,s5y6).
#external ext_dep(2,3,s4y5y6).
#external ext_dep(2,4,empty).
#external ext_dep(2,4,s0).
#external ext_dep(2,4,s1).
#external ext_dep(2,4,s3).
#external ext_dep(2,4,s5).
#external ext_dep(2,4,s6).
#external ext_dep(2,4,s1y3).
#external ext_dep(2,4,s1y6).
#external ext_dep(2,4,s3y5).
#external ext_dep(2,4,s3y6).
#external ext_dep(2,4,s5y6).
#external ext_dep(2,4,s1y3y6).
#external ext_dep(2,5,empty).
#external ext_dep(2,5,s3).
#external ext_dep(2,5,s4).
#external ext_dep(2,5,s6).
#external ext_dep(2,5,s3y4).
#external ext_dep(2,5,s3y6).
#external ext_dep(2,5,s4y6).
#external ext_dep(3,4,s0).
#external ext_dep(3,5,empty).
#external ext_dep(3,5,s2).
#external ext_dep(3,5,s4).
#external ext_dep(3,5,s6).
#external ext_dep(3,5,s2y4).
#external ext_dep(3,5,s2y6).
#external ext_dep(3,5,s4y6).
#external ext_dep(3,5,s2y4y6).
#external ext_dep(3,6,empty).
#external ext_dep(3,6,s1).
#external ext_dep(3,6,s2).
#external ext_dep(3,6,s4).
#external ext_dep(3,6,s5).
#external ext_dep(3,6,s1y2).
#external ext_dep(3,6,s1y4).
#external ext_dep(3,6,s1y5).
#external ext_dep(3,6,s2y4).
#external ext_dep(3,6,s2y5).
#external ext_dep(3,6,s4y5).
#external ext_dep(3,6,s1y4y5).
#external ext_dep(3,6,s2y4y5).
#external ext_dep(4,6,empty).
#external ext_dep(4,6,s0).
#external ext_dep(4,6,s1).
#external ext_dep(4,6,s2).
#external ext_dep(4,6,s3).
#external ext_dep(4,6,s5).
#external ext_dep(4,6,s1y2).
#external ext_dep(4,6,s1y3).
#external ext_dep(4,6,s1y5).
#external ext_dep(4,6,s2y3).
#external ext_dep(4,6,s2y5).
#external ext_dep(4,6,s3y5).
#external ext_dep(4,6,s1y2y3).
#external ext_dep(4,6,s1y3y5).
#external ext_dep(5,6,empty).
#external ext_dep(5,6,s1).
#external ext_dep(5,6,s2).
#external ext_dep(5,6,s3).
#external ext_dep(5,6,s4).
#external ext_dep(5,6,s1y2).
#external ext_dep(5,6,s1y3).
#external ext_dep(5,6,s1y4).
#external ext_dep(5,6,s2y3).
#external ext_dep(5,6,s2y4).
#external ext_dep(5,6,s3y4).
#external ext_dep(5,6,s1y3y4).

% weak constraints (as loaded)
:~ ext_indep(0,1,empty). [-5742849]
:~ ext_indep(0,2,empty). [-9937293]
:~ ext_indep(0,3,empty). [-9919858]
:~ ext_dep(0,4,empty). [-6100651]
:~ ext_indep(0,4,s1). [-4138770]
:~ ext_dep(0,4,s2). [-4881937]
:~ ext_dep(0,4,s3). [-4881220]
:~ ext_dep(0,4,s6). [-5356736]
:~ ext_indep(0,5,empty). [-8016820]
:~ ext_indep(0,6,empty). [-8085754]
:~ ext_indep(1,2,s3). [-7816731]
:~ ext_indep(1,2,s4). [-5890583]
:~ ext_indep(1,2,s5). [-7161207]
:~ ext_indep(1,2,s6). [-7080379]
:~ ext_indep(1,2,s3y4). [-4360732]
:~ ext_indep(1,2,s3y5). [-5324170]
:~ ext_indep(1,2,s4y5). [-3341592]
:~ ext_indep(1,2,s4y6). [-4755167]
:~ ext_indep(1,2,s3y4y5). [-2233973]
:~ ext_indep(1,3,s2). [-5164343]
:~ ext_indep(1,3,s4). [-7436144]
:~ ext_indep(1,3,s5). [-5892251]
:~ ext_indep(1,3,s6). [-4939139]
:~ ext_indep(1,3,s2y5). [-4404433]
:~ ext_indep(1,3,s2y6). [-3742963]
:~ ext_indep(1,3,s4y6). [-5138970]
:~ ext_indep(1,3,s5y6). [-3901511]
:~ ext_indep(1,3,s2y5y6). [-2602373]
:~ ext_dep(1,4,empty). [-9999999]
:~ ext_dep(1,4,s0). [-7999999]
:~ ext_dep(1,4,s2). [-7999999]
:~ ext_dep(1,4,s3). [-7999999]
:~ ext_dep(1,4,s6). [-7999999]
:~ ext_dep(1,4,s2y3). [-5999999]
:~ ext_dep(1,4,s2y6). [-5999999]
:~ ext_dep(1,4,s3y6). [-5999999]
:~ ext_dep(1,4,s2y3y6). [-4000000]
:~ ext_indep(1,5,s2). [-4785705]
:~ ext_indep(1,5,s3). [-5310465]
:~ ext_indep(1,5,s4). [-4114379]
:~ ext_indep(1,5,s6). [-6148857]
:~ ext_indep(1,5,s2y3). [-3920975]
:~ ext_indep(1,5,s2y6). [-4754400]
:~ ext_indep(1,5,s3y6). [-5362415]
:~ ext_indep(1,5,s4y6). [-4682432]
:~ ext_indep(1,5,s2y3y6). [-3627029]
:~ ext_dep(1,6,empty). [-9990840]
:~ ext_dep(1,6,s2). [-7992826]
:~ ext_dep(1,6,s3). [-7993425]
:~ ext_dep(1,6,s4). [-7999999]
:~ ext_dep(1,6,s5). [-7987312]
:~ ext_dep(1,6,s2y3). [-5995102]
:~ ext_dep(1,6,s2y4). [-5999999]
:~ ext_dep(1,6,s2y5). [-5990172]
:~ ext_dep(1,6,s3y4). [-5999999]
:~ ext_dep(1,6,s3y5). [-5992586]
:~ ext_dep(1,6,s4y5). [-5999999]
:~ ext_dep(1,6,s3y4y5). [-4000000]
:~ ext_dep(2,3,empty). [-9999999]
:~ ext_dep(2,3,s4). [-7999999]
:~ ext_dep(2,3,s5). [-7624472]
:~ ext_dep(2,3,s6). [-7999999]
:~ ext_dep(2,3,s4y5). [-5612503]
:~ ext_dep(2,3,s4y6). [-5999999]
:~ ext_dep(2,3,s5y6). [-4351046]
:~ ext_dep(2,3,s4y5y6). [-2733590]
:~ ext_dep(2,4,empty). [-8919276]
:~ ext_dep(2,4,s0). [-7135687]
:~ ext_dep(2,4,s1). [-7317305]
:~ ext_dep(2,4,s3). [-5902805]
:~ ext_dep(2,4,s5). [-7930773]
:~ ext_dep(2,4,s6). [-6310164]
:~ ext_dep(2,4,s1y3). [-4815923]
:~ ext_dep(2,4,s1y6). [-4862656]
:~ ext_dep(2,4,s3y5). [-5929120]
:~ ext_dep(2,4,s3y6). [-3364033]
:~ ext_dep(2,4,s5y6). [-5027577]
:~ ext_dep(2,4,s1y3y6). [-2531326]
:~ ext_dep(2,5,empty). [-9999999]
:~ ext_dep(2,5,s3). [-7999999]
:~ ext_dep(2,5,s4). [-7999999]
:~ ext_dep(2,5,s6). [-7999999]
:~ ext_dep(2,5,s3y4). [-5999999]
:~ ext_dep(2,5,s3y6). [-5999999]
:~ ext_dep(2,5,s4y6). [-5999999]
:~ ext_dep(3,4,s0). [-7214988]
:~ ext_dep(3,5,empty). [-9999999]
:~ ext_dep(3,5,s2). [-7999999]
:~ ext_dep(3,5,s4). [-7999999]
:~ ext_dep(3,5,s6). [-7999999]
:~ ext_dep(3,5,s2y4). [-5999999]
:~ ext_dep(3,5,s2y6). [-5999999]
:~ ext_dep(3,5,s4y6). [-5999999]
:~ ext_dep(3,5,s2y4y6). [-4000000]
:~ ext_dep(3,6,empty). [-8729590]
:~ ext_dep(3,6,s1). [-7095082]
:~ ext_dep(3,6,s2). [-5812854]
:~ ext_dep(3,6,s4). [-5853136]
:~ ext_dep(3,6,s5). [-7999999]
:~ ext_dep(3,6,s1y2). [-4520810]
:~ ext_dep(3,6,s1y4). [-4473881]
:~ ext_dep(3,6,s1y5). [-5999999]
:~ ext_dep(3,6,s2y4). [-3029810]
:~ ext_dep(3,6,s2y5). [-5999999]
:~ ext_dep(3,6,s4y5). [-5999999]
:~ ext_dep(3,6,s1y4y5). [-4000000]
:~ ext_dep(3,6,s2y4y5). [-4000000]
:~ ext_dep(4,6,empty). [-9999999]
:~ ext_dep(4,6,s0). [-7999999]
:~ ext_dep(4,6,s1). [-7999999]
:~ ext_dep(4,6,s2). [-7999999]
:~ ext_dep(4,6,s3). [-7999999]
:~ ext_dep(4,6,s5). [-7999999]
:~ ext_dep(4,6,s1y2). [-5999999]
:~ ext_dep(4,6,s1y3). [-5999999]
:~ ext_dep(4,6,s1y5). [-5999999]
:~ ext_dep(4,6,s2y3). [-5999999]
:~ ext_dep(4,6,s2y5). [-5999999]
:~ ext_dep(4,6,s3y5). [-5999999]
:~ ext_dep(4,6,s1y2y3). [-4000000]
:~ ext_dep(4,6,s1y3y5). [-4000000]
:~ ext_dep(5,6,empty). [-9999999]
:~ ext_dep(5,6,s1). [-7999999]
:~ ext_dep(5,6,s2). [-7999999]
:~ ext_dep(5,6,s3). [-7999999]
:~ ext_dep(5,6,s4). [-7999999]
:~ ext_dep(5,6,s1y2). [-5999999]
:~ ext_dep(5,6,s1y3). [-5999999]
:~ ext_dep(5,6,s1y4). [-5999999]
:~ ext_dep(5,6,s2y3). [-5999999]
:~ ext_dep(5,6,s2y4). [-5999999]
:~ ext_dep(5,6,s3y4). [-5999999]
:~ ext_dep(5,6,s1y3y4). [-4000000]

% generated facts (in/2, qpair/3, etc)
#program specific.
in(2,s2).
in(3,s3y4).
in(4,s3y4).
in(5,s5).
in(4,s4y6).
in(6,s4y6).
in(4,s4y5y6).
in(5,s4y5y6).
in(6,s4y5y6).
in(1,s1y3y6).
in(3,s1y3y6).
in(6,s1y3y6).
in(1,s1y6).
in(6,s1y6).
in(2,s2y5).
in(5,s2y5).
in(1,s1y3).
in(3,s1y3).
in(1,s1y4y5).
in(4,s1y4y5).
in(5,s1y4y5).
in(2,s2y4y5).
in(4,s2y4y5).
in(5,s2y4y5).
in(2,s2y3y6).
in(3,s2y3y6).
in(6,s2y3y6).
in(4,s4).
in(1,s1).
in(4,s4y5).
in(5,s4y5).
in(5,s5y6).
in(6,s5y6).
in(3,s3y6).
in(6,s3y6).
in(2,s2y5y6).
in(5,s2y5y6).
in(6,s2y5y6).
in(2,s2y4).
in(4,s2y4).
in(1,s1y2).
in(2,s1y2).
in(1,s1y5).
in(5,s1y5).
in(1,s1y3y5).
in(3,s1y3y5).
in(5,s1y3y5).
in(0,s0).
in(3,s3).
in(3,s3y5).
in(5,s3y5).
in(6,s6).
in(1,s1y2y3).
in(2,s1y2y3).
in(3,s1y2y3).
in(1,s1y4).
in(4,s1y4).
in(1,s1y3y4).
in(3,s1y3y4).
in(4,s1y3y4).
in(2,s2y3).
in(3,s2y3).
in(3,s3y4y5).
in(4,s3y4y5).
in(5,s3y4y5).
in(2,s2y6).
in(6,s2y6).
in(2,s2y4y6).
in(4,s2y4y6).
in(6,s2y4y6).
qpair(0,4,s6).
qpair(0,4,s3).
qpair(0,4,empty).
qpair(0,4,s2).
qpair(1,4,s6).
qpair(1,4,s2).
qpair(1,4,s2y3y6).
qpair(1,4,s2y3).
qpair(1,4,s2y6).
qpair(1,4,s3y6).
qpair(1,4,s0).
qpair(1,4,empty).
qpair(1,4,s3).
qpair(1,6,s2y4).
qpair(1,6,s2).
qpair(1,6,s3y4).
qpair(1,6,s5).
qpair(1,6,s4).
qpair(1,6,s2y3).
qpair(1,6,s4y5).
qpair(1,6,s3y4y5).
qpair(1,6,s2y5).
qpair(1,6,empty).
qpair(1,6,s3).
qpair(1,6,s3y5).
qpair(2,3,s6).
qpair(2,3,s5).
qpair(2,3,s4y6).
qpair(2,3,s4).
qpair(2,3,s4y5).
qpair(2,3,s5y6).
qpair(2,3,s4y5y6).
qpair(2,3,empty).
qpair(2,4,s6).
qpair(2,4,s1y3).
qpair(2,4,s1y3y6).
qpair(2,4,s5).
qpair(2,4,s1).
qpair(2,4,s5y6).
qpair(2,4,s3y6).
qpair(2,4,s0).
qpair(2,4,s1y6).
qpair(2,4,empty).
qpair(2,4,s3).
qpair(2,4,s3y5).
qpair(2,5,s6).
qpair(2,5,s3y4).
qpair(2,5,s4y6).
qpair(2,5,s4).
qpair(2,5,s3y6).
qpair(2,5,empty).
qpair(2,5,s3).
qpair(3,4,s0).
qpair(3,5,s6).
qpair(3,5,s2y4).
qpair(3,5,s2).
qpair(3,5,s2y4y6).
qpair(3,5,s4y6).
qpair(3,5,s4).
qpair(3,5,s2y6).
qpair(3,5,empty).
qpair(3,6,s2y4).
qpair(3,6,s2).
qpair(3,6,s1y2).
qpair(3,6,s2y4y5).
qpair(3,6,s5).
qpair(3,6,s1y5).
qpair(3,6,s4).
qpair(3,6,s1y4).
qpair(3,6,s1).
qpair(3,6,s4y5).
qpair(3,6,s2y5).
qpair(3,6,empty).
qpair(3,6,s1y4y5).
qpair(4,6,s1y3).
qpair(4,6,s2).
qpair(4,6,s1y2).
qpair(4,6,s5).
qpair(4,6,s1y5).
qpair(4,6,s1y3y5).
qpair(4,6,s1y2y3).
qpair(4,6,s2y3).
qpair(4,6,s1).
qpair(4,6,s2y5).
qpair(4,6,s0).
qpair(4,6,empty).
qpair(4,6,s3).
qpair(4,6,s3y5).
qpair(5,6,s1y3).
qpair(5,6,s2y4).
qpair(5,6,s2).
qpair(5,6,s1y2).
qpair(5,6,s3y4).
qpair(5,6,s4).
qpair(5,6,s1y4).
qpair(5,6,s2y3).
qpair(5,6,s1).
qpair(5,6,s1y3y4).
qpair(5,6,empty).
qpair(5,6,s3).
qpair(0,1,empty).
qpair(0,2,empty).
qpair(0,3,empty).
qpair(0,4,s1).
qpair(0,5,empty).
qpair(0,6,empty).
qpair(1,2,s6).
qpair(1,2,s3y4).
qpair(1,2,s5).
qpair(1,2,s4y6).
qpair(1,2,s4).
qpair(1,2,s4y5).
qpair(1,2,s3y4y5).
qpair(1,2,s3).
qpair(1,2,s3y5).
qpair(1,3,s6).
qpair(1,3,s2).
qpair(1,3,s5).
qpair(1,3,s4y6).
qpair(1,3,s4).
qpair(1,3,s2y6).
qpair(1,3,s5y6).
qpair(1,3,s2y5).
qpair(1,3,s2y5y6).
qpair(1,5,s6).
qpair(1,5,s2).
qpair(1,5,s2y3y6).
qpair(1,5,s4y6).
qpair(1,5,s4).
qpair(1,5,s2y3).
qpair(1,5,s2y6).
qpair(1,5,s3y6).
qpair(1,5,s3).

#program specific.
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
:- block_edge(X,Y), edge(X,Y), X<Y, var(X), var(Y).
#external block_edge(0,1).
#external block_edge(0,2).
#external block_edge(0,3).
#external block_edge(0,4).
#external block_edge(0,5).
#external block_edge(0,6).
#external block_edge(1,2).
#external block_edge(1,3).
#external block_edge(1,5).

#program base.
var(0).
var(1).
var(2).
var(3).
var(4).
var(5).
var(6).
#show arrow/2.
