%%% sets and elements
var(0..n_vars).
set(X) :- in(_,X).
set(empty).

%%% dependency and independency
indep(X,Y,S) :- not dep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).
:- dep(X,Y,S), indep(X,Y,S).
%%% symmetry
indep(X,Y,S) :- indep(Y,X,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- dep(Y,X,S), var(X), var(Y), set(S), X!=Y.

%%%%% All DAGs
%generate arrows
arrow(X,Y) | arrow(Y,X) | not edge(X,Y) :- var(X), var(Y), X!=Y.
%%% edge and arrow conditions
edge(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
edge(X,Y):- edge(Y,X), X!=Y, var(X), var(Y).
not arrow(X,Y) :- not edge(X,Y), var(X), var(Y).
arrow(X,Y) :- edge(X,Y), not arrow(Y,X), var(X), var(Y).
not edge(X,Y) :- not arrow(X,Y), not arrow(Y,X), var(X), var(Y).
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

%%% Active and Blocked path rules are in python, here only the one for edges that does not change with paths
ap(X,Y,S) :- edge(X,Y), var(X),var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).
dep(X,Y,S) :- ap(X,Y,S).

%#show arrow/2.
%#show collider/3.
%#show collider_desc/4.
%#show dep/3. 
%#show indep/3. 
%#show ap/3.
%#show ap/4.
%#show ap/5.
%#show dpath/2.
%#show nb/4.
