import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import sitecustomize

ARGCAUSALDISCO_ROOT = PROJECT_ROOT.parent
if not ((ARGCAUSALDISCO_ROOT / "__init__.py").exists() and (ARGCAUSALDISCO_ROOT / "utils").is_dir()):
    ARGCAUSALDISCO_ROOT = PROJECT_ROOT / "ArgCausalDisco"

DIFF_PATH = PROJECT_ROOT / "notebooks" / "DiscrepancyExperiment" / "arg_cd.diff"
os.system(f'cd "{ARGCAUSALDISCO_ROOT}" && git apply "{DIFF_PATH}"')

sys.path.insert(0, 'notears/')

import numpy as np
from tqdm import tqdm
import pandas as pd
import time

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from ArgCausalDisco.abapc import ABAPC

from src.abapc import get_arrow_sets, get_best_model
from src.utils.bn_utils import get_dataset
from logger import logger

ALPHA = 0.01
INDEP_TEST = 'fisherz'
RESULTS_DIR = Path(__file__).resolve().parent
LOAD_SAVED = False
EXPERIMENT_RESULT_PATH = RESULTS_DIR / 'experiment_results.csv'

dataset_list = [
    'cancer',
    'earthquake',
    'survey',
    # 'asia'
]

names_dict = {'pc': 'PC',
              'pc_max': 'Max-PC',
              'fgs': 'FGS',
              'spc': 'Shapley-PC',
              'mpc': 'MPC',
              'cpc': 'CPC',
              'abapc': 'ABAPC (Ours)',
              'pure_abapc': 'ABAPC (ASPforABA)',
              'cam': 'CAM',
              'nt': 'NOTEARS-MLP',
              'mcsl': 'MCSL-MLP',
              'ges': 'GES',
              'random': 'Random'}

model_list = ['abapc', 'pure_abapc',]


def main(n_runs=50,
         sample_size=5000):

    version = f'bnlearn_{n_runs}rep'

    # setup previous causal disco codebase logger
    logger_setup(str(RESULTS_DIR / f'log_{version}.log'))

    experiment_result_data = []

    for dataset_name in tqdm(dataset_list, desc="tqdm Datasets"):
        random_stability(2024)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
        for seed in tqdm(seeds_list, desc="tqdm Seeds"):
            X_s, B_true = get_dataset(dataset_name,
                                            seed=seed,
                                            sample_size=sample_size)
        
            
                

            # pure abapc
            start = time.time()
            stable_arrow_sets, cg, _, _ = get_arrow_sets(X_s,
                                                        seed=seed,
                                                        alpha=ALPHA,
                                                        indep_test=INDEP_TEST)
            _, W_est_pure, best_I_new = get_best_model(stable_arrow_sets,
                                            n_nodes=X_s.shape[1],
                                            cg=cg,
                                            alpha=ALPHA)
            pure_abapc_elapsed = time.time() - start

            # old implementation
            start = time.time()
            W_est_old, best_I_old, all_models = ABAPC(data=X_s,
                                        alpha=ALPHA,
                                        indep_test=INDEP_TEST,
                                        scenario=f"abapc_{version}_{dataset_name}",
                                        out_mode="opt")
            old_elapsed = time.time() - start

            all_models = set(all_models)
            stable_arrow_sets = set(stable_arrow_sets)

            W_est_old = np.array(W_est_old, dtype=int)
            W_est_pure = np.array(W_est_pure, dtype=int)

            experiment_result_data.append({
                'dataset': dataset_name,
                'seed': seed,
                'num_models_old': len(all_models),
                'num_models_new': len(stable_arrow_sets),
                'all_models_sets_equal': all_models == stable_arrow_sets,
                'best_models_equal': np.array_equal(W_est_old, W_est_pure),
                'best_model_old': str(W_est_old.tolist()),
                'best_model_new': str(W_est_pure.tolist()),
                'best_model_score_old': best_I_old,
                'best_model_score_new': best_I_new,
                'time_elapsed_sec_old': old_elapsed,
                'time_elapsed_sec_new': pure_abapc_elapsed,
            })

            experiment_result_df = pd.DataFrame(experiment_result_data)
            experiment_result_df.to_csv(EXPERIMENT_RESULT_PATH, index=False)


if __name__ == "__main__":
    # apply this custom diff

    # Run the main function
    main()


# revert the diff
os.system(f'cd "{ARGCAUSALDISCO_ROOT}" && git checkout .')
