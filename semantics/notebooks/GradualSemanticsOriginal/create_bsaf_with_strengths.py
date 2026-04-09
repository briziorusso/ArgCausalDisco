import sys
sys.path.insert(0, '../../')
import sitecustomize
sys.path.insert(0, '../../ArgCausalDisco/')
sys.path.insert(0, '../../notears/')
sys.path.append("../../GradualABA")

from logger import logger
from src.gradual.run import run_get_bsaf
from src.gradual.extra.abaf_factory_v0 import FactoryV0
from src.utils.bn_utils import get_dataset
from src.abapc import get_cg_and_facts
from ArgCausalDisco.utils.helpers import random_stability
from pathlib import Path
import time
import pickle
from GradualABA.ABAF import ABAF
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE


SEED = 2025
RESULT_DIR = Path("./results/")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = 'cancer'
APPENDIX = f"{DATASET_NAME}_with_strengths"


def main():
    X_s, B_true = get_dataset(DATASET_NAME,
                              seed=SEED,
                              sample_size=SAMPLE_SIZE)

    # get facts from pc
    uc_rule = 5
    data = X_s

    random_stability(SEED)
    n_nodes = data.shape[1]
    _, facts = get_cg_and_facts(data, alpha=ALPHA, indep_test=INDEP_TEST, uc_rule=uc_rule)

    pickle.dump(facts, open(RESULT_DIR / f"facts_{APPENDIX}.pkl", "wb"))

    factory = FactoryV0(n_nodes=n_nodes)

    start = time.time()
    bsaf = run_get_bsaf(factory, facts, abaf_class=ABAF)
    end = time.time()
    logger.info(f"Time taken to create BSAF: {end - start} seconds")
    with open(RESULT_DIR / f"time_{APPENDIX}.txt", "w") as f:
        f.write(f"{end - start} seconds taken to create BSAF\n")

    pickle.dump(bsaf, open(RESULT_DIR / f"bsaf_{APPENDIX}.pkl", "wb"))


if __name__ == "__main__":
    main()
