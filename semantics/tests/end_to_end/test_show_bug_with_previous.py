import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

import shutil
from collections import defaultdict
from ArgCausalDisco.abapc import ABAPC
from src.abapc import get_arrow_sets
from pathlib import Path


def test_causalaba_equal_to_ABASP(mocker):
    """
    NOTE: THIS TEST FAILS

    This test shows difference between old and new implementation of ABAPC.

    More specifically, considering the case when there is 3 node graph with following facts:

    alpha=0.01

    1. independent (p=1   , p>alpha, I=0      ), X=1, Y=2, S={0},
    2. dependent   (p=0   , p<alpha, I=0.995  ), X=1, Y=2, S={},
    3. independent (p=0.02, p>alpha, I=0.505  ), X=0, Y=1, S={}

    The only solution is ignoring the first fact and coming up with following 3 graphs:

    0     2 --> 1     (2, 1)
    0     2 <-- 1     (1, 2)
    0 --> 2 <-- 1     (0, 2), (1, 2)

    The new implementation works correctly and returns the above 3 graphs.
    The old implementation returns 3 different graphs:

    1 <-- 0 --> 2     (0, 1), (0, 2)
    1 <-- 0 <-- 2     (0, 1), (2, 0)
    1 --> 0 --> 2     (1, 0), (0, 2)

    Which is incorrect and is actually the solution where instead of the first fact the last one is ignored.

    """

    # Mock the PC algorithm and its output

    cg = mocker.MagicMock()
    cg.sepset = defaultdict(list)

    cg.sepset.update({
        (1, 2): [((0,), 1), ((), 0)],
        (0, 1): [((), 0.02)],
    })

    mock_pc = mocker.MagicMock(return_value=cg)
    data = mocker.MagicMock()
    data.shape = (5000, 3)
    seed = 42  # doesn't matter for this test

    with mocker.patch('ArgCausalDisco.abapc.pc', mock_pc), mocker.patch('src.abapc.pc', mock_pc):
        models, _ = ABAPC(data=data,
                          seed=seed,
                          alpha=0.01,
                          indep_test='fisherz',
                          scenario='test_cancer_5_nodes',
                          out_mode="optN")

        abasp_models, _, _, _ = get_arrow_sets(data, seed=seed, alpha=0.01)
        abasp_models = {frozenset(model) for model in abasp_models}

    print('Old implementation models: ', models)
    print('New implementation models: ', abasp_models)
    assert abasp_models == models
    print("ABASP and ABAPC models are equal, test passed successfully!")

    # clean up
    if Path('results/test_cancer_5_nodes').exists():
        shutil.rmtree(Path('results/test_cancer_5_nodes'), ignore_errors=True)
