"""Copyright 2024 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2024 Fabrizio Russo"

import logging
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from models import run_method
from utils import load_bnlearn_data_dag, DAGMetrics, random_stability, simulate_dag, logger_setup

import warnings
warnings.filterwarnings("ignore")

version = 'bnlearn_wstd_disco_kci'
logger_setup(f'results/log_{version}.log')
device = 0
print_graphs = False
save_res = True
load_res = False
dataset_list = ['cancer', #'earthquake', 
                'survey', 'asia'
                # , 'sachs' 
                #, 'alarm', 'insurance', 'child', 'hailfinder',  'hepar2'
                ]
model_list = [
            'random'
            ,'spc' 
            # ,'pc_max'
            # ,'cam'
            ,'nt' 
            # ,'mcsl'
            ,'fgs',
            'abapc'
    ]
data_path = '../ShapleyPC-local/datasets'
sample_size = 2000

mt_res = pd.DataFrame()
for dataset_name in dataset_list:
    if load_res:
        mt_res = np.load(f"results/stored_results_{version}.npy", allow_pickle=True)
    else:
        ##Load data
        X_s, B_true = load_bnlearn_data_dag(dataset_name, data_path, sample_size, seed=2023, print_info=True, standardise=True)
        # np.save(f"{dataset_name}_data.npy", X_s)

        names_dict = {'pc_max':'Max-PC', 'fgs':'FGS', 'spc':'Shapley-PC', 'abapc':'ABAPC (Ours)', 'cam':'CAM', 'nt':'NOTEARS-MLP', 'mcsl':'MCSL-MLP', 'ges':'GES', 'random':'Random'}
        # B_true = nx.adjacency_matrix(true_causal_matrix).todense()

        for method in model_list:
            random_stability(2023)
            seeds_list = np.random.randint(0, 10000, (10, )).tolist()
            seeds_list = [seeds_list[0]] if method in ['abapc', 'spc', 'pc_max', 'cam', 'fgs'] else seeds_list
            logging.debug(f'Seeds:{seeds_list}')
            logging.info(f"Running {method}")
            
            method_res = []
            for seed in seeds_list:
                if method=='random':
                    random_stability(seed)
                    start = datetime.now()
                    B_est = simulate_dag(d=B_true.shape[1], s0=B_true.sum().astype(int), graph_type='ER')
                    elapsed = (datetime.now()-start).total_seconds()
                    mt = DAGMetrics(B_est, B_true)
                else:
                    W_est, elapsed = run_method(X_s, method, seed, test_alpha=0.01, test_name='kci', device=device, scenario=f"{method}_{version}_{dataset_name}")
                    logger_setup(f'results/log_{version}.log', continue_logging=True)
                    if W_est is None:
                        mt.metrics = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                    else:
                        B_est = (W_est > 0).astype(int)
                        mt = DAGMetrics(B_est, B_true)

                # calculate metrics
                logging.info({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt.metrics})
                method_res.append({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt.metrics})
            method_sum = pd.DataFrame(method_res).groupby(['dataset','model'], as_index=False).agg(['mean','std']).round(2).reset_index(drop=True)
            method_sum.columns = method_sum.columns.map('_'.join).str.strip('_')
            mt_res = pd.concat([mt_res, method_sum], sort=False)

            if save_res:
                np.save(f"results/stored_results_{version}.npy", mt_res )