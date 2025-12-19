% Assumption-layer MUS for the semantic (full) encoding
% We expose three assumption atoms a(1..3) and link them to facts via o/1

{mus(1)}.
{mus(2)}.
{mus(3)}.

% Conditional facts activated by o/1
ext_indep(1,2,s0) :- mus(1).
ext_dep(1,2,empty) :- mus(2).
ext_indep(0,1,empty) :- mus(3).

% Core declarations (same as extended file)
var(0). var(1). var(2).
set(empty).
set(s0).
in(0,s0).

indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.

indep(X,Y,S) :- indep(Y,X,S), var(X), var(Y), set(S), X!=Y.
dep(X,Y,S) :- dep(Y,X,S), var(X), var(Y), set(S), X!=Y.
:- dep(X,Y,S), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).

arrow(X,Y) | arrow(Y,X) | not edge(X,Y) :- var(X), var(Y), X!=Y.
edge(X,Y) :- arrow(X,Y), X!=Y, var(X), var(Y).
edge(X,Y) :- edge(Y,X), X!=Y, var(X), var(Y).

:- edge(X,Y), indep(X,Y,S), set(S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S).
:- arrow(X,Y), not edge(X,Y), var(X), var(Y).
:- edge(X,Y), not arrow(X,Y), not arrow(Y,X), var(X), var(Y).
:- arrow(Y,X), arrow(X,Y), X!=Y, var(X), var(Y).

dpath(X,Y):- arrow(X,Y), X!=Y, var(X), var(Y).
dpath(X,Y):- arrow(X,Z), dpath(Z,Y), var(X). 
:- dpath(X,X), var(X).

collider(Y,X,Z) :- arrow(X,Y), arrow(Z,Y), X!=Z, X!=Y, Y!=Z, var(X), var(Y), var(Z).
collider_desc(N,Y,X,Z) :- collider(Y,X,Z), dpath(Y,N), var(N), X!=N, Y!=N, Z!=N.

nb(N,X,Y,S) :- in(N,S), collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), not collider(N,X,Y), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, set(S).
nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in(Z,S), collider_desc(Z,N,X,Y), var(Z).

p1 :- edge(0,1).
ap(0,1,p1,S) :- p1, not in(0,S), not in(1,S), set(S).

p2 :- edge(0,2).
ap(0,2,p2,S) :- p2, not in(0,S), not in(2,S), set(S).

p3 :- edge(0,1), edge(1,2).
ap(0,2,p3,S) :- p3, nb(1,0,2,S), not in(0,S), not in(2,S), set(S).

p4 :- edge(1,0).
ap(1,0,p4,S) :- p4, not in(1,S), not in(0,S), set(S).

p5 :- edge(1,2).
ap(1,2,p5,S) :- p5, not in(1,S), not in(2,S), set(S).

p6 :- edge(1,0), edge(0,2).
ap(1,2,p6,S) :- p6, nb(0,1,2,S), not in(1,S), not in(2,S), set(S).

p7 :- edge(2,0).
ap(2,0,p7,S) :- p7, not in(2,S), not in(0,S), set(S).

p8 :- edge(2,1), edge(1,0).
ap(2,0,p8,S) :- p8, nb(1,2,0,S), not in(2,S), not in(0,S), set(S).

p9 :- edge(2,1).
ap(2,1,p9,S) :- p9, not in(2,S), not in(1,S), set(S).

p10 :- edge(2,0), edge(0,1).
ap(2,1,p10,S) :- p10, nb(0,2,1,S), not in(2,S), not in(1,S), set(S).

dep(0,1,S) :- ap(0,1,_,S), set(S).
dep(0,2,S) :- ap(0,2,_,S), set(S).
dep(1,2,S) :- ap(1,2,_,S), set(S).

indep(1,2,empty) :- not ap(1,2,_,empty).