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
nb(N,X,Y,S) :- not in(N,S), collider_desc(Z,N,X,Y), var(N), var(X), var(Y), var(Z), N!=X, N!=Y, X!=Y, in(Z,S).

%%%%% Counting paths
num(N):- {var(X)} == N. % count nodes
step(1..N) :- num(N). % mark possible steps

path(X,Y,2,cons(X,cons(Y,empty))) :- edge(X,Y).
path(A,C,NN+1,cons(A,L)) :- edge(A,B), path(B,C,NN,L), step(NN+1).

member(X,path(X,Y,N,cons(X,L))) :- path(X,Y,N,cons(X,L)).
member(Y,path(X,Y,N,cons(X,L))) :- path(X,Y,N,cons(X,L)).   
member(M,path(S,T,NN+1,cons(S,cons(Z,L)))) :- member(M,path(Z,T,NN,cons(Z,L))), path(S,T,NN+1,cons(S,cons(Z,L))).   

track(Y,Z,N,L):- {member(X,path(Y,Z,N,L)):var(X)} == N, path(Y,Z,N,L).
n_tracks(Y,Z,T):- {track(Y,Z,_,_)} == T, var(Y), var(Z), Y!=Z, T!=0.
max_tracks(Y,Z,T) :- n_tracks(Y,Z,T), T = #max{T_i,1:n_tracks(Y,Z,T_i)}.

ap(Y,Z,N,L,S):- track(Y,Z,N,L), N-2{member(M,path(Y,Z,N,L)):nb(M,X,Z,S)}N-2, M!=Y, M!=Z, M!=X, var(M), var(X), set(S), not in(Y,S), not in(Z,S).

dep(Y,Z,S) :- ap(Y,Z,_,_,S).
%indep(X,Y,S) :-  T{not ap(X,Y,_,_,S)}T, max_tracks(X,Y,T), var(X), var(Y), set(S), not in(X,S), not in(Y,S).

%#show track/4.
%#show arrow/2.
%#show activetrack/5.
%#show collider/3.
%#show indep/3.
%#show dep/3.