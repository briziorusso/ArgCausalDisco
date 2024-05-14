#const n_vars=3.

:~ indep(0,1,empty). [-855]
:~ indep(0,1,s2). [-330]
:~ indep(0,1,s3). [-438]
:~ indep(0,1,s2y3). [-0]
:~ dep(0,2,empty). [-996]
:~ dep(0,2,s1). [-498]
:~ dep(0,2,s3). [-498]
:~ dep(0,3,empty). [-999]
:~ dep(0,3,s2). [-499]
:~ dep(1,2,empty). [-1000]
:~ dep(1,2,s0). [-500]
:~ indep(1,3,empty). [-653]
:~ indep(2,3,empty). [-842]
:~ indep(2,3,s0). [-460]
:~ indep(2,3,s1). [-486]
:~ indep(2,3,s0y1). [-0]

in(0,s0).
in(1,s1).
in(2,s2).
in(3,s3).
in(0,s0y1).
in(1,s0y1).
in(0,s0y2).
in(2,s0y2).
in(0,s0y3).
in(3,s0y3).
in(1,s1y2).
in(2,s1y2).
in(1,s1y3).
in(3,s1y3).
in(2,s2y3).
in(3,s2y3).
in(0,s0y1y2).
in(1,s0y1y2).
in(2,s0y1y2).
in(0,s0y1y3).
in(1,s0y1y3).
in(3,s0y1y3).
in(0,s0y2y3).
in(2,s0y2y3).
in(3,s0y2y3).
in(1,s1y2y3).
in(2,s1y2y3).
in(3,s1y2y3).
in(0,s0y1y2y3).
in(1,s0y1y2y3).
in(2,s0y1y2y3).
in(3,s0y1y2y3).

p1 :- edge(0,2),edge(2,1).
ap(0,1,p1,S) :- p1, nb(2,0,1,S), not in(0,S), not in(1,S), set(S).
p2 :- edge(0,3),edge(3,1).
ap(0,1,p2,S) :- p2, nb(3,0,1,S), not in(0,S), not in(1,S), set(S).
p3 :- edge(0,3),edge(3,2),edge(2,1).
ap(0,1,p3,S) :- p3, nb(3,0,2,S),nb(2,3,1,S), not in(0,S), not in(1,S), set(S).

p4 :- edge(0,2).
ap(0,2,p4,S) :- p4,  not in(0,S), not in(2,S), set(S).
p5 :- edge(0,3),edge(3,1),edge(1,2).
ap(0,2,p5,S) :- p5, nb(3,0,1,S),nb(1,3,2,S), not in(0,S), not in(2,S), set(S).
p6 :- edge(0,3),edge(3,2).
ap(0,2,p6,S) :- p6, nb(3,0,2,S), not in(0,S), not in(2,S), set(S).
indep(0,2,S) :-  not ap(0,2,p4,S), not ap(0,2,p5,S), not ap(0,2,p6,S), not in(0,S), not in(2,S), set(S).

p7 :- edge(0,3).
ap(0,3,p7,S) :- p7,  not in(0,S), not in(3,S), set(S).
indep(0,3,S) :-  not ap(0,3,p7,S), not in(0,S), not in(3,S), set(S).

p8 :- edge(1,0),edge(0,2).
ap(1,2,p8,S) :- p8, nb(0,1,2,S), not in(1,S), not in(2,S), set(S).
p9 :- edge(1,0),edge(0,3),edge(3,2).
ap(1,2,p9,S) :- p9, nb(0,1,3,S),nb(3,0,2,S), not in(1,S), not in(2,S), set(S).
p10 :- edge(1,2).
ap(1,2,p10,S) :- p10,  not in(1,S), not in(2,S), set(S).
indep(1,2,S) :-  not ap(1,2,p8,S), not ap(1,2,p9,S), not ap(1,2,p10,S), not in(1,S), not in(2,S), set(S).

p11 :- edge(1,0),edge(0,3).
ap(1,3,p11,S) :- p11, nb(0,1,3,S), not in(1,S), not in(3,S), set(S).
p12 :- edge(1,2),edge(2,0),edge(0,3).
ap(1,3,p12,S) :- p12, nb(2,1,0,S),nb(0,2,3,S), not in(1,S), not in(3,S), set(S).

p13 :- edge(2,0),edge(0,3).
ap(2,3,p13,S) :- p13, nb(0,2,3,S), not in(2,S), not in(3,S), set(S).
p14 :- edge(2,1),edge(1,0),edge(0,3).
ap(2,3,p14,S) :- p14, nb(1,2,0,S),nb(0,1,3,S), not in(2,S), not in(3,S), set(S).
dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).

dep(1,2,empty).
dep(0,3,empty).
dep(0,2,empty).


#show arrow/2.
#show indep/3.
#show dep/3.