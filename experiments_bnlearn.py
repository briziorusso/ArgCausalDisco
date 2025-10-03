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
import numpy as np
import pandas as pd
from datetime import datetime
from cd_algorithms.models import run_method
from utils.graph_utils import DAGMetrics, dag2cpdag, is_dag
from utils.helpers import random_stability, logger_setup
from utils.data_utils import load_bnlearn_data_dag, simulate_dag
import warnings
warnings.filterwarnings("ignore")

version = 'bnlearn_big_rnd_mpc2'
logger_setup(f'results/log_{version}.log')
data_path = 'datasets'
sample_size = 5000
n_runs = 50
device = 0
load_res = False
save_res = True
dataset_list = [
                'cancer', 
                'earthquake', 
                'survey', 
                'asia',
                'sachs',
                ]
model_list = [
            'random'
            ,'mpc'
            # 'abapc'
            # 'fgs'
            # ,'nt'
            # ,'spc'
            ]
            ###Start time 17:25 - Finished 19:45

if load_res:         
    mt_res = pd.DataFrame(np.load(f"results/stored_results_{version}.npy", allow_pickle=True), 
                       columns = ['dataset', 'model', 'elapsed_mean', 'elapsed_std', 'nnz_mean', 'nnz_std', 
                                'fdr_mean', 'fdr_std', 'tpr_mean', 'tpr_std', 'fpr_mean', 'fpr_std', 
                                'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
                                'F1_mean', 'F1_std', 'shd_mean', 'shd_std','sid_mean', 'sid_std'])
    mt_res_cpdag = pd.DataFrame(np.load(f"results/stored_results_{version}_cpdag.npy", allow_pickle=True), 
                       columns = ['dataset', 'model', 'elapsed_mean', 'elapsed_std', 'nnz_mean', 'nnz_std', 
                                'fdr_mean', 'fdr_std', 'tpr_mean', 'tpr_std', 'fpr_mean', 'fpr_std', 
                                'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
                                'F1_mean', 'F1_std', 'shd_mean', 'shd_std', 'sid_low_mean', 'sid_low_std', 'sid_high_mean', 'sid_high_std'])
    ## save backup to npy
    np.save(f"results/stored_results_{version}_bkp.npy", mt_res )
    np.save(f"results/stored_results_{version}_cpdag_bkp.npy", mt_res_cpdag )
else:
    mt_res = pd.DataFrame()
    mt_res_cpdag = pd.DataFrame()

for dataset_name in dataset_list:
    names_dict = {'pc':'PC', 'pc_max':'Max-PC', 'fgs':'FGS', 'spc':'Shapley-PC', 'mpc':'MPC', 'cpc':'CPC', 'abapc':'ABAPC (Ours)', 'cam':'CAM', 'nt':'NOTEARS-MLP', 'mcsl':'MCSL-MLP', 'ges':'GES', 'random':'Random'}
    # B_true = nx.adjacency_matrix(true_causal_matrix).todense()

    for method in model_list:
        random_stability(2024)
        seeds_list = np.random.randint(0, 10000, (n_runs, )).tolist()
        # seeds_list = [seeds_list[0]] if method in ['abapc', 'spc', 'pc_max', 'cam', 'fgs', 'pc', 'mpc', 'cpc'] else seeds_list
        logging.debug(f'Seeds:{seeds_list}')
        logging.info(f"Running {method}")
        
        method_res = []
        method_res_cpdag = []
        for seed in seeds_list:
            ##Load data
            X_s, B_true = load_bnlearn_data_dag(dataset_name, data_path, sample_size, seed=seed, print_info=True if seed == seeds_list[0] else False, standardise=True)
            if method=='random':
                random_stability(seed)
                start = datetime.now()
                B_est = simulate_dag(d=B_true.shape[1], s0=B_true.sum().astype(int), graph_type='ER')
                elapsed = (datetime.now()-start).total_seconds()
                mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                mt_dag = DAGMetrics(B_est, B_true).metrics
            else:
                W_est, elapsed = run_method(X_s, method, seed, test_alpha=0.01, test_name='fisherz', device=device, scenario=f"{method}_{version}_{dataset_name}")
                if 'Tensor' in str(type(W_est)):
                    W_est = np.asarray([list(i) for i in W_est])
                logger_setup(f'results/log_{version}.log', continue_logging=True)
                if W_est is None:
                    mt_cpdag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                    mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                else:
                    B_est = (W_est != 0).astype(int)
                    mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                    B_est = (W_est > 0).astype(int)
                    if not is_dag(B_est):
                        logging.warning(f"Estimated graph not a DAG, skipping run")
                        mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                    else:
                        mt_dag = DAGMetrics(B_est, B_true).metrics
            # calculate metrics
            logging.info({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt_dag})
            logging.info({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt_cpdag})
            
            method_res.append({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt_dag})
            if type(mt_cpdag['sid'])==tuple:
                mt_sid_low = mt_cpdag['sid'][0]
                mt_sid_high = mt_cpdag['sid'][1]
            else:
                mt_sid_low = mt_cpdag['sid']
                mt_sid_high = mt_cpdag['sid']
            mt_cpdag.pop('sid')
            mt_cpdag['sid_low'] = mt_sid_low
            mt_cpdag['sid_high'] = mt_sid_high
            method_res_cpdag.append({'dataset':dataset_name, 'model':names_dict[method], 'elapsed':elapsed , **mt_cpdag})
            
        method_sum = pd.DataFrame(method_res).groupby(['dataset','model'], as_index=False).agg(['mean','std']).round(2).reset_index(drop=True)
        method_sum.columns = method_sum.columns.map('_'.join).str.strip('_')
        mt_res = pd.concat([mt_res, method_sum], sort=False)

        method_sum = pd.DataFrame(method_res_cpdag).groupby(['dataset','model'], as_index=False).agg(['mean','std']).round(2).reset_index(drop=True)
        method_sum.columns = method_sum.columns.map('_'.join).str.strip('_')
        mt_res_cpdag = pd.concat([mt_res_cpdag, method_sum], sort=False)

        if save_res:
            np.save(f"results/stored_results_{version}.npy", mt_res )
            np.save(f"results/stored_results_{version}_cpdag.npy", mt_res_cpdag )
