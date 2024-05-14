#const n_vars=2.

in(0,s0).
in(1,s1).
in(2,s2).
in(0,s0y1).
in(1,s0y1).
in(0,s0y2).
in(2,s0y2).
in(1,s1y2).
in(2,s1y2).
in(0,s0y1y2).
in(1,s0y1y2).
in(2,s0y1y2).

#external ext_indep(0,1,empty).
%indep(0,1,s2).
%dep(0,2,empty).
%dep(0,2,s1).
%dep(1,2,empty).
%dep(1,2,s0).

%#external ext_ar.
a :- not b. 
#external ext_b.

%ar :- ext_ar.
%ar :- a.
b :- ext_b.
b :- not a.

indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.

p1 :- edge(0,1).
ap(0,1,p1,S) :- p1,  not in(0,S), not in(1,S), set(S).
p2 :- edge(0,2),edge(2,1).
ap(0,1,p2,S) :- p2, nb(2,0,1,S), not in(0,S), not in(1,S), set(S).
indep(0,1,S) :-  not ap(0,1,p1,S), not ap(0,1,p2,S), not in(0,S), not in(1,S), set(S).
dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).

p3 :- edge(0,1),edge(1,2).
ap(0,2,p3,S) :- p3, nb(1,0,2,S), not in(0,S), not in(2,S), set(S).
p4 :- edge(0,2).
ap(0,2,p4,S) :- p4,  not in(0,S), not in(2,S), set(S).
indep(0,2,S) :-  not ap(0,2,p3,S), not ap(0,2,p4,S), not in(0,S), not in(2,S), set(S).
dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).

p5 :- edge(1,0),edge(0,2).
ap(1,2,p5,S) :- p5, nb(0,1,2,S), not in(1,S), not in(2,S), set(S).
p6 :- edge(1,2).
ap(1,2,p6,S) :- p6,  not in(1,S), not in(2,S), set(S).
indep(1,2,S) :-  not ap(1,2,p5,S), not ap(1,2,p6,S), not in(1,S), not in(2,S), set(S).
dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).

%%% sets and elements
var(0..n_vars).
set(X) :- in(_,X).
set(empty).

%%% dependency and independency
%%% symmetry
indep(X,Y,S) :- indep(Y,X,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- dep(Y,X,S), var(X), var(Y), set(S), X!=Y.
%%% test
:- dep(X,Y,S), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).
:- edge(X,Y), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).

%%%%% All DAGs
%%% generate arrows
arrow(X,Y) | arrow(Y,X) | not edge(X,Y) :- var(X), var(Y), X!=Y.
%%% symmetry
edge(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
edge(X,Y):- edge(Y,X), X!=Y, var(X), var(Y).
%%% test
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

%%% Nonblocker rules, third rule is in python since it is path dependent
nb(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc(Z,N,X,Y), var(Z).

%#show arrow/2.
%#show ext_indep/3.
%#show ext_dep/3.
#show ar/0.
#show a/0.
#show b/0.

%#show collider/3.
%#show collider_desc/4.
%#show dep/3. 
%#show indep/3. 
%#show ap/3.
%#show ap/4.
%#show ap/5.
%#show dpath/2.
%#show nb/4.
