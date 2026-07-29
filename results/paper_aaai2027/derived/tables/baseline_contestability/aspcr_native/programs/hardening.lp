#const forced_id=0.
:- forced_id > 0, constraint(forced_id,X,Y,C,J,M,W), fail(X,Y,C,J,M,W).
#show fail/6.
