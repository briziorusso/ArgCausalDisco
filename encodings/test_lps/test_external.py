import os
from clingo.control import Control
from clingo import Function, Number, String

ctl = Control(['-t %d' % os.cpu_count()])
ctl.configuration.solve.parallel_mode = os.cpu_count()
ctl.configuration.solve.models="0"
ctl.configuration.solver.seed="2024"

ctl.load("encodings/test_lps/test_external.lp")

ctl.ground([("base", [])])
models = []
count_models = 0
with ctl.solve(yield_=True) as handle:
    for model in handle:
        models.append(model.symbols(shown=True))
        count_models += 1
        print(f"Answer {count_models}: {model}")
