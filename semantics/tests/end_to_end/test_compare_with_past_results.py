
import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from pathlib import Path
import pytest
import shutil

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.abapc import ABAPC

from src.abapc import get_arrow_sets
from src.utils.bn_utils import get_dataset
from src.constants import ALPHA, INDEP_TEST


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_causalaba_equal_to_ABASP(seed):
    data, _ = get_dataset('cancer', seed=seed)

    random_stability(seed)
    models, _ = ABAPC(data=data,
                    alpha=ALPHA,
                    indep_test=INDEP_TEST,
                    scenario='test_cancer_5_nodes',
                    out_mode="optN")
    
    abasp_models, _, _, _ = get_arrow_sets(data, seed=seed)
    abasp_models = {frozenset(model) for model in abasp_models}

    print(seed)
    print(models)
    print(abasp_models)
    assert abasp_models == models
    print("ABASP and ABAPC models are equal, test passed successfully!")

    # clean up
    if Path('results/test_cancer_5_nodes').exists():
        shutil.rmtree(Path('results/test_cancer_5_nodes'), ignore_errors=True)


if __name__ == "__main__":
    test_causalaba_equal_to_ABASP(seed=42)
