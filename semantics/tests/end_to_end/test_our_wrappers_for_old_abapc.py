import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from pathlib import Path
import shutil
import pytest
from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.abapc import ABAPC
from src.utils.gen_random_nx import generate_random_bn_data
from src.abapc import get_cg_and_facts, get_best_model
from src.abapc import get_models_from_facts
from src.abapc import get_best_model
from src.constants import ALPHA, INDEP_TEST
from logger import logger
import logging


N_NODES = 5
TOL = 1e-6

logger.setLevel(logging.ERROR)

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_our_wrappers_yield_same_model_and_ranking_as_old_causalaba_interface(seed):
    n_runs = 5
    sample_size = 5000
    version = f'random_{n_runs}rep'

    # generate random data
    X_s, _ = generate_random_bn_data(
        n_nodes=N_NODES,
        n_edges=N_NODES,
        n_samples=sample_size,
        seed=seed,
        standardise=True
    )

    # run old CausalABA with our wrapping interface
    random_stability(seed)
    cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
    facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score

    Path('test_results').mkdir(exist_ok=True)
    base_location = str(Path('test_results') / f'nodes{N_NODES}_edges{N_NODES}_{seed}_facts.lp')

    new_model_sets = get_models_from_facts(facts, seed, N_NODES, base_location=base_location)
    _, _, best_I_new = get_best_model(new_model_sets, N_NODES, cg, alpha=ALPHA)

    # Now run with old CausalABA interface
    all_models, best_ranking = ABAPC(data=X_s,
                                     alpha=ALPHA,
                                     indep_test=INDEP_TEST,
                                     scenario=f"abapc_{version}_{N_NODES}",
                                     out_mode="optN")
    _, best_I_old = best_ranking[0]

    all_models = set(all_models)
    new_model_sets = set(new_model_sets)

    # compare results

    assert all_models == new_model_sets, "Model sets do not match between old and new implementations."
    assert abs(best_I_old - best_I_new) < TOL, "Best model scores do not match between old and new implementations. There is something wrong with ranking of solutions"

    # clean up, revert diff, remove test results directory
    if Path('test_results').exists():
        shutil.rmtree(Path('test_results'), ignore_errors=True)
    if Path(f"results/abapc_{version}_{N_NODES}").exists():
        shutil.rmtree(Path(f"results/abapc_{version}_{N_NODES}"), ignore_errors=True)


if __name__ == "__main__":
    # apply this custom diff

    # Run the main function
    test_our_wrappers_yield_same_model_and_ranking_as_old_causalaba_interface(42)
    print("Test passed successfully!")
