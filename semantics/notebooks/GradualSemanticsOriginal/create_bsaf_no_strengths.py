import sys
sys.path.insert(0, '../../')
import sitecustomize
sys.path.insert(0, '../../ArgCausalDisco/')
sys.path.insert(0, '../../notears/')
sys.path.append("../../GradualABA")

from logger import logger
from src.gradual.run import run_get_bsaf
from src.gradual.extra.abaf_factory_v0 import FactoryV0
from ArgCausalDisco.utils.helpers import random_stability
from pathlib import Path
import time
import pickle
from GradualABA.ABAF import ABAF



RESULT_DIR = Path("./results/")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_NAME = 'cancer'
SEED = 2025
APPENDIX = f"{DATASET_NAME}_no_strengths"


def main():
    random_stability(SEED)
    n_nodes = 5
    facts = []  # no facts so all independence assumptions get strength of 0.5 initially

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
