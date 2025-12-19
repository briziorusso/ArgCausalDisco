% final after removal 1
% literal mapping
% 1 -> in(0,s0)
% 8 -> arrow(1,0)
% 9 -> arrow(0,1)
% 12 -> arrow(2,0)
% 13 -> arrow(0,2)
% 18 -> arrow(2,1)
% 19 -> arrow(1,2)
% 52 -> ext_indep(1,2,s0)
% 53 -> ext_indep(0,1,empty)
% 54 -> indep(0,1,empty)
% 55 -> indep(1,2,s0)
% 56 -> ext_dep(1,2,empty)
% 57 -> indep(1,2,empty)
% 58 -> indep(2,1,empty)
% 59 -> indep(2,1,s0)
% 60 -> indep(1,0,empty)
% 61 -> dep(1,2,empty)
% 62 -> dep(2,1,empty)
% 63 -> block_edge(1,2)
% 64 -> block_edge(0,1)
% 66 -> active_pair(1,2)
% 68 -> ap(1,2,q1_1,empty)
% 70 -> ap(1,2,q1_1,s0)
% 71 -> dep(1,2,s0)
% symbolic atoms
% 52: ext_indep(1,2,s0)
% 53: ext_indep(0,1,empty)
% 56: ext_dep(1,2,empty)
% 2: set(empty)
% 39: set(s0)
% 3: var(0)
% 4: var(1)
% 5: var(2)
% 54: indep(0,1,empty)
% 55: indep(1,2,s0)
% 57: indep(1,2,empty)
% 58: indep(2,1,empty)
% 59: indep(2,1,s0)
% 60: indep(1,0,empty)
% 61: dep(1,2,empty)
% 62: dep(2,1,empty)
% 71: dep(1,2,s0)
% 7: edge(1,0)
% 11: edge(2,0)
% 15: edge(0,1)
% 17: edge(2,1)
% 21: edge(0,2)
% 23: edge(1,2)
% 63: block_edge(1,2)
% 64: block_edge(0,1)
% 1: in(0,s0)
% 8: arrow(1,0)
% 9: arrow(0,1)
% 12: arrow(2,0)
% 13: arrow(0,2)
% 18: arrow(2,1)
% 19: arrow(1,2)
% 24: dpath(1,0)
% 25: dpath(2,0)
% 26: dpath(0,1)
% 27: dpath(2,1)
% 28: dpath(0,2)
% 29: dpath(1,2)
% 33: collider(1,2,0)
% 34: collider(2,1,0)
% 35: collider(0,2,1)
% 36: collider(2,0,1)
% 37: collider(0,1,2)
% 38: collider(1,0,2)
% 40: nb(2,1,0,empty)
% 41: nb(1,2,0,empty)
% 42: nb(2,0,1,empty)
% 43: nb(0,2,1,empty)
% 44: nb(1,0,2,empty)
% 45: nb(0,1,2,empty)
% 46: nb(2,1,0,s0)
% 47: nb(1,2,0,s0)
% 48: nb(2,0,1,s0)
% 49: nb(1,0,2,s0)
% 50: nb(0,2,1,s0)
% 51: nb(0,1,2,s0)
% 68: ap(1,2,q1_1,empty)
% 70: ap(1,2,q1_1,s0)
% 66: active_pair(1,2)
% 67: q1_1
in(0,s0).
2.
3.
4.
5.
6 :- not 7.
8 | 9 :- not 6.
10 :- not 11.
12 | 13 :- not 10.
14 :- not 15.
9 | 8 :- not 14.
16 :- not 17.
18 | 19 :- not 16.
20 :- not 21.
13 | 12 :- not 20.
22 :- not 23.
19 | 18 :- not 22.
7 :- 8.
11 :- 12.
15 :- 9.
17 :- 18.
21 :- 13.
23 :- 19.
17 :- 23.
11 :- 21.
23 :- 17.
7 :- 15.
21 :- 11.
15 :- 7.
24 :- 8.
25 :- 12.
26 :- 9.
27 :- 18.
28 :- 13.
29 :- 19.
28 :- 29, 9.
30 :- 29, 18.
29 :- 28, 8.
30 :- 28, 12.
26 :- 27, 13.
31 :- 27, 19.
31 :- 26, 8.
27 :- 26, 12.
32 :- 25, 13.
24 :- 25, 19.
32 :- 24, 9.
25 :- 24, 18.
24 :- 32, 8.
25 :- 32, 12.
26 :- 31, 9.
27 :- 31, 18.
28 :- 30, 13.
29 :- 30, 19.
33 :- 9, 18.
34 :- 13, 19.
35 :- 8, 12.
36 :- 19, 13.
37 :- 12, 8.
38 :- 18, 9.
39.
40 :- not 34.
41 :- not 33.
42 :- not 36.
43 :- not 35.
44 :- not 38.
45 :- not 37.
46 :- not 34.
47 :- not 33.
48 :- not 36.
49 :- not 38.
50 :- 35.
51 :- 37.
:- 32.
:- 31.
:- 30.
:- 8, 9.
:- 12, 13.
:- 9, 8.
:- 18, 19.
:- 13, 12.
:- 19, 18.
:- 7, not 8, not 9.
:- 11, not 12, not 13.
:- 15, not 9, not 8.
:- 17, not 18, not 19.
:- 21, not 13, not 12.
:- 23, not 19, not 18.
:- 8, not 7.
:- 12, not 11.
:- 9, not 15.
:- 18, not 17.
:- 13, not 21.
:- 19, not 23.
54 :- 53.
55 :- 52.
57 :- 56.
58 :- 57.
59 :- 55.
60 :- 54.
54 :- 60.
55 :- 59.
57 :- 58.
:- 54, 15.
:- 57, 23.
:- 58, 17.
:- 60, 7.
:- 55, 23.
:- 59, 17.
61 :- 56.
62 :- 61.
61 :- 62.
:- 57, 61.
:- 58, 62.
:- 23, 63.
:- 15, 64.
67 :- 21, 7, 66.
68 :- 45, 66, 67.
69 :- 68.
indep(1,2,empty) :- not 69, ext_dep(1,2,empty).
70 :- 51, 66, 67.
71 :- 70, ext_indep(1,2,s0).
:~ 56=-552894464, 53=-1060142772, 52=1732812800. [0]
% external assignments
% ext_dep(1,2,empty) = True
% ext_indep(1,2,s0) = True
% ext_indep(0,1,empty) = False
% block_edge(0,1) = False
% block_edge(1,2) = True
% active_pair(1,2) = True
