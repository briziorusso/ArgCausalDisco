"""Run causal discovery experiments on CausaNet-style BIFXML graphs.
This mirrors experiments_bnlearn.py but loads graphs from the bifxmls folder
and simulates discrete data from the DAG structure.
"""

import os
import logging
import re
import fnmatch
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from datetime import datetime
from cd_algorithms.models import run_method
from utils.graph_utils import DAGMetrics, dag2cpdag, is_dag
from utils.helpers import random_stability, logger_setup
from utils.data_utils import load_causenet_data_dag, simulate_dag
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


DAG_BASE_COLUMNS = ['dataset', 'model', 'elapsed', 'nnz', 'fdr', 'tpr', 'fpr',
                    'precision', 'recall', 'F1', 'shd', 'sid']
CPDAG_BASE_COLUMNS = ['dataset', 'model', 'elapsed', 'nnz', 'fdr', 'tpr', 'fpr',
                      'precision', 'recall', 'F1', 'shd', 'sid_low', 'sid_high']
DAG_PROGRESS_COLUMNS = DAG_BASE_COLUMNS + ['run_idx', 'seed']
CPDAG_PROGRESS_COLUMNS = CPDAG_BASE_COLUMNS + ['run_idx', 'seed']
DAG_METRIC_MAP = [
    ('elapsed', 'elapsed'),
    ('nnz', 'nnz'),
    ('fdr', 'fdr'),
    ('tpr', 'tpr'),
    ('fpr', 'fpr'),
    ('precision', 'precision'),
    ('recall', 'recall'),
    ('F1', 'F1'),
    ('shd', 'shd'),
    ('sid', 'SID'),
]
CPDAG_METRIC_MAP = [
    ('elapsed', 'elapsed'),
    ('nnz', 'nnz'),
    ('fdr', 'fdr'),
    ('tpr', 'tpr'),
    ('fpr', 'fpr'),
    ('precision', 'precision'),
    ('recall', 'recall'),
    ('F1', 'F1'),
    ('shd', 'shd'),
    ('sid_low', 'SID_low'),
    ('sid_high', 'SID_high'),
]
DAG_SUMMARY_COLUMNS = ['dataset', 'model'] + [f"{dst}_{stat}" for _, dst in DAG_METRIC_MAP for stat in ('mean', 'std')]
CPDAG_SUMMARY_COLUMNS = ['dataset', 'model'] + [f"{dst}_{stat}" for _, dst in CPDAG_METRIC_MAP for stat in ('mean', 'std')]


def safe_filename(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def load_existing_summary(path: Path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and hasattr(arr, 'item'):
        arr = arr.item()
    if isinstance(arr, pd.DataFrame):
        df = arr
    elif isinstance(arr, np.ndarray):
        try:
            df = pd.DataFrame(arr, columns=columns)
        except Exception:
            try:
                df = pd.DataFrame(arr.tolist(), columns=columns)
            except Exception:
                logging.warning(f'Could not interpret summary file at {path}; starting with empty frame.')
                return pd.DataFrame(columns=columns)
    else:
        logging.warning(f'Unexpected data type in summary file at {path}; starting with empty frame.')
        return pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    return df[columns]


def load_progress_df(path: Path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    return df[columns]


def append_progress_row(path: Path, row: dict, columns):
    df_row = pd.DataFrame([row]).reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row.to_csv(path, mode='a', header=not path.exists(), index=False)


def summarise_results(df: pd.DataFrame, metric_pairs):
    if df.empty:
        return pd.DataFrame(columns=['dataset', 'model'] + [f"{dst}_{stat}" for _, dst in metric_pairs for stat in ('mean', 'std')])
    agg_dict = {src: ['mean', 'std'] for src, _ in metric_pairs}
    summary = (
        df.groupby(['dataset', 'model'], as_index=False)
          .agg(agg_dict)
          .round(2)
    )
    summary.columns = ['dataset', 'model'] + [f"{dst}_{stat}" for _, dst in metric_pairs for stat in ('mean', 'std')]
    return summary


def save_summary_tables(base_path: Path, version: str, dag_df: pd.DataFrame, cpdag_df: pd.DataFrame):
    np.save(base_path / f'stored_results_{version}.npy', dag_df.reindex(columns=DAG_SUMMARY_COLUMNS).to_numpy())
    np.save(base_path / f'stored_results_{version}_cpdag.npy', cpdag_df.reindex(columns=CPDAG_SUMMARY_COLUMNS).to_numpy())

# CLI args for flexible subset selection
parser = argparse.ArgumentParser(description='Run experiments on CausaNet BIFXML graphs')
parser.add_argument('--version', default='causenet_bifxml_50rep')
parser.add_argument('--models', nargs='*', default=['random'], help='List of models to run (default: random only)')
parser.add_argument('--results_dir', default='results')
parser.add_argument('--bifxml_dir', default=os.path.join('datasets','causenet_generator','bifxmls'))
parser.add_argument('--sample_size', type=int, default=5000)
parser.add_argument('--n_runs', type=int, default=50)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_res', action='store_true')
parser.add_argument('--save_res', action='store_true', default=True)
parser.add_argument('--simulate_with', choices=['internal','pyagrum'], default='internal', help='How to simulate data from DAG')
parser.add_argument('--resume', action='store_true', help='Resume from saved progress and summaries (default: start fresh)')
# Subset filters
parser.add_argument('--include', action='append', help='Substring to include (can be used multiple times)')
parser.add_argument('--glob', action='append', help='Glob pattern to include (can be used multiple times)')
parser.add_argument('--regex', action='append', help='Regex to include (can be used multiple times)')
parser.add_argument('--names', nargs='*', help='Exact base names (without path) to include')
parser.add_argument('--nodes', type=int, choices=[5,10,15], help='Only graphs with this number of nodes')
parser.add_argument('--edges_class', choices=['d','1.5d'], help='Only graphs with edges equal to d or ~1.5d')
parser.add_argument('--heur', choices=['none','degrees','semantics'], help='Only graphs with this heuristic')
parser.add_argument('--type', dest='gtype', choices=['random','er','sf'], help='Only graphs with this type')

args = parser.parse_args()

# Config
version = args.version
results_path = Path(args.results_dir)
results_path.mkdir(parents=True, exist_ok=True)
logger_setup(str(results_path / f'log_{version}.log'))

# Folder with .bifxml graphs
bifxml_dir = args.bifxml_dir
sample_size = args.sample_size
n_runs = args.n_runs
device = args.device
load_res = args.load_res
resume = args.resume
save_res = args.save_res
simulate_with = args.simulate_with
include_filter = None  # deprecated

model_list = args.models
## Available models:
## 'random', 'pc', 'pc_max', 'fgs', 'spc', 'mpc', 'cpc', 'abapc', 'cam', 'nt', 'mcsl', 'ges'

load_existing = load_res or resume
if load_existing:
    mt_res = load_existing_summary(results_path / f'stored_results_{version}.npy', DAG_SUMMARY_COLUMNS)
    mt_res_cpdag = load_existing_summary(results_path / f'stored_results_{version}_cpdag.npy', CPDAG_SUMMARY_COLUMNS)
    if load_res and mt_res.empty:
        logging.warning('Requested --load_res but no stored DAG results were found.')
    if load_res and mt_res_cpdag.empty:
        logging.warning('Requested --load_res but no stored CPDAG results were found.')
else:
    mt_res = pd.DataFrame(columns=DAG_SUMMARY_COLUMNS)
    mt_res_cpdag = pd.DataFrame(columns=CPDAG_SUMMARY_COLUMNS)

mt_res = mt_res.reindex(columns=DAG_SUMMARY_COLUMNS, fill_value=np.nan)
mt_res_cpdag = mt_res_cpdag.reindex(columns=CPDAG_SUMMARY_COLUMNS, fill_value=np.nan)

progress_dir = results_path / 'progress' / version
if not resume and progress_dir.exists():
    shutil.rmtree(progress_dir)
progress_dir.mkdir(parents=True, exist_ok=True)

if load_existing and save_res:
    np.save(results_path / f'stored_results_{version}_bkp.npy', mt_res.reindex(columns=DAG_SUMMARY_COLUMNS).to_numpy())
    np.save(results_path / f'stored_results_{version}_cpdag_bkp.npy', mt_res_cpdag.reindex(columns=CPDAG_SUMMARY_COLUMNS).to_numpy())

# enumerate datasets
bifxml_files = [
    os.path.join(bifxml_dir, fn) for fn in sorted(os.listdir(bifxml_dir)) if fn.endswith('.bifxml')
]

def edge_class(d, e):
    if e == d:
        return 'd'
    target = 1.5 * d
    if abs(e - round(target)) <= 1:
        return '1.5d'
    return None

def parse_meta(fn):
    m = re.match(r'dag_(\d+)_nodes_(\d+)_edges_(.*)\.bifxml$', fn)
    if not m:
        return None
    nodes = int(m.group(1)); edges = int(m.group(2)); tail = m.group(3)
    parts = tail.split('_') if '_' in tail else [tail]
    heur = parts[0] if parts else None
    gtype = parts[-1].lower() if parts else None
    return nodes, edges, heur, gtype

def keep_file(path):
    bn = os.path.basename(path)
    if args.names and bn not in args.names:
        return False
    if args.include and not any(s in bn for s in args.include):
        return False
    if args.glob and not any(fnmatch.fnmatch(bn, pat) for pat in args.glob):
        return False
    if args.regex and not any(re.search(rx, bn) for rx in args.regex):
        return False
    meta = parse_meta(bn)
    if meta is None:
        return True
    n, e, heur, gtype = meta
    if args.nodes and n != args.nodes:
        return False
    if args.edges_class and edge_class(n, e) != args.edges_class:
        return False
    if args.heur and heur != args.heur:
        return False
    if args.gtype and gtype != args.gtype:
        return False
    return True

bifxml_files = [p for p in bifxml_files if keep_file(p)]

if not bifxml_files:
    raise SystemExit('No .bifxml files selected by the provided filters.')

names_dict = {'pc':'PC', 'pc_max':'Max-PC', 'fgs':'FGS', 'spc':'Shapley-PC', 'mpc':'MPC', 'cpc':'CPC', 'abapc':'ABAPC (Ours)', 'cam':'CAM', 'nt':'NOTEARS-MLP', 'mcsl':'MCSL-MLP', 'ges':'GES', 'random':'Random'}

for graph_path in bifxml_files:
    dataset_name = os.path.splitext(os.path.basename(graph_path))[0]
    dataset_safe = safe_filename(dataset_name)

    for method in model_list:
        display_name = names_dict.get(method, method)
        if method not in names_dict:
            logging.warning(f'Unknown method key "{method}"; using raw name in outputs.')

        dag_progress_path = progress_dir / f"{dataset_safe}__{safe_filename(method)}_dag.csv"
        cpdag_progress_path = progress_dir / f"{dataset_safe}__{safe_filename(method)}_cpdag.csv"

        dag_runs = load_progress_df(dag_progress_path, DAG_PROGRESS_COLUMNS)
        cpdag_runs = load_progress_df(cpdag_progress_path, CPDAG_PROGRESS_COLUMNS)
        if not dag_runs.empty:
            dag_runs = dag_runs.sort_values('run_idx').reset_index(drop=True)
        if not cpdag_runs.empty:
            cpdag_runs = cpdag_runs.sort_values('run_idx').reset_index(drop=True)

        completed_runs = min(len(dag_runs), len(cpdag_runs))
        if len(dag_runs) != len(cpdag_runs):
            logging.warning(
                f'Mismatch between DAG ({len(dag_runs)}) and CPDAG ({len(cpdag_runs)}) progress counts for '
                f'{display_name} on {dataset_name}; trimming to the smallest count ({completed_runs}).'
            )
            dag_runs = dag_runs.head(completed_runs)
            cpdag_runs = cpdag_runs.head(completed_runs)

        if completed_runs > n_runs:
            logging.warning(
                f'Existing progress for {display_name} on {dataset_name} has {completed_runs} runs; '
                f'truncating to requested n_runs={n_runs}.')
            dag_runs = dag_runs.head(n_runs)
            cpdag_runs = cpdag_runs.head(n_runs)
            completed_runs = n_runs

        random_stability(2024)
        seeds_list = np.random.randint(0, 10000, (n_runs, )).tolist()
        logging.debug(f'Seeds:{seeds_list}')

        if completed_runs:
            logging.info(f"Resuming {method} on {dataset_name}: {completed_runs}/{n_runs} runs already completed.")

        logging.info(f"Running {method} on {dataset_name}")

        for idx in range(completed_runs, n_runs):
            seed = seeds_list[idx]
            # Load data + true DAG from BIFXML (simulate discrete data with random CPDs)
            X_s, B_true = load_causenet_data_dag(
                graph_path,
                sample_size,
                seed=seed,
                print_info=True if idx == 0 and completed_runs == 0 else False,
                standardise=False,
                simulate_with=simulate_with,
            )

            if 'random' in method.lower(): 
                random_stability(seed)
                start = datetime.now()
                if method == 'random_edge':
                    s0 = int(B_true.sum())
                elif method == 'random':
                    ### sample a random edge density
                    s0 = np.random.randint(B_true.shape[1], (B_true.shape[1]*(B_true.shape[1]-1))//2 + 1)
                    logging.info(f'Sampling random graph with {s0} edges')
                else:
                    raise ValueError(f'Unknown random method {method}')
                B_est = simulate_dag(d=B_true.shape[1], s0=s0, graph_type='ER')
                elapsed = (datetime.now()-start).total_seconds()
                mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                mt_dag = DAGMetrics(B_est, B_true).metrics
            else:
                W_est, elapsed = run_method(X_s, method, seed, test_alpha=0.01, test_name='fisherz', device=device, scenario=f"{method}_{version}_{dataset_name}")
                if 'Tensor' in str(type(W_est)):
                    W_est = np.asarray([list(i) for i in W_est])
                logger_setup(str(results_path / f'log_{version}.log'), continue_logging=True)
                if W_est is None:
                    mt_cpdag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                    mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}
                else:
                    B_est_binary = (W_est != 0).astype(int)
                    mt_cpdag = DAGMetrics(dag2cpdag(B_est_binary.copy()), B_true).metrics
                    B_est = (W_est > 0).astype(int)
                    bidirected_mask = (B_est == 1) & (B_est.T == 1)
                    if bidirected_mask.any():
                        logging.warning('Estimated graph contains bidirected edges; removing them before DAG metrics computation.')
                        B_est[bidirected_mask] = 0
                    if is_dag(B_est):
                        mt_dag = DAGMetrics(B_est, B_true).metrics
                    else:
                        logging.warning('Estimated graph is not a DAG after bidirected edge removal; skipping DAG metrics for this run.')
                        mt_dag = {'nnz':np.nan, 'fdr':np.nan, 'tpr':np.nan, 'fpr':np.nan, 'precision':np.nan, 'recall':np.nan, 'F1':np.nan, 'shd':np.nan, 'sid':np.nan}

            logging.info({'dataset':dataset_name, 'model':display_name, 'elapsed':elapsed , **mt_dag})
            logging.info({'dataset':dataset_name, 'model':display_name, 'elapsed':elapsed , **mt_cpdag})

            dag_row = {'dataset':dataset_name, 'model':display_name, 'elapsed':elapsed, **mt_dag, 'run_idx': idx, 'seed': seed}
            if isinstance(mt_cpdag.get('sid'), tuple):
                mt_sid_low, mt_sid_high = mt_cpdag['sid']
            else:
                mt_sid_low = mt_cpdag.get('sid')
                mt_sid_high = mt_cpdag.get('sid')
            mt_cpdag.pop('sid', None)
            mt_cpdag['sid_low'] = mt_sid_low
            mt_cpdag['sid_high'] = mt_sid_high
            cpdag_row = {'dataset':dataset_name, 'model':display_name, 'elapsed':elapsed, **mt_cpdag, 'run_idx': idx, 'seed': seed}

            append_progress_row(dag_progress_path, dag_row, DAG_PROGRESS_COLUMNS)
            append_progress_row(cpdag_progress_path, cpdag_row, CPDAG_PROGRESS_COLUMNS)

            dag_runs = pd.concat([dag_runs, pd.DataFrame([dag_row])], ignore_index=True)
            cpdag_runs = pd.concat([cpdag_runs, pd.DataFrame([cpdag_row])], ignore_index=True)

        if len(dag_runs) < n_runs or len(cpdag_runs) < n_runs:
            logging.info(
                f'Completed {len(dag_runs)}/{n_runs} runs for {display_name} on {dataset_name}; '
                f'partial progress saved for resume.'
            )
            continue

        dag_metrics_df = dag_runs[DAG_BASE_COLUMNS]
        cpdag_metrics_df = cpdag_runs[CPDAG_BASE_COLUMNS]

        dag_summary = summarise_results(dag_metrics_df, DAG_METRIC_MAP).reindex(columns=DAG_SUMMARY_COLUMNS)
        cpdag_summary = summarise_results(cpdag_metrics_df, CPDAG_METRIC_MAP).reindex(columns=CPDAG_SUMMARY_COLUMNS)

        if not dag_summary.empty:
            mask = (mt_res['dataset'] == dataset_name) & (mt_res['model'] == display_name)
            mt_res = mt_res[~mask]
            mt_res = pd.concat([mt_res, dag_summary], ignore_index=True)

        if not cpdag_summary.empty:
            mask_cpdag = (mt_res_cpdag['dataset'] == dataset_name) & (mt_res_cpdag['model'] == display_name)
            mt_res_cpdag = mt_res_cpdag[~mask_cpdag]
            mt_res_cpdag = pd.concat([mt_res_cpdag, cpdag_summary], ignore_index=True)

        mt_res = mt_res.reindex(columns=DAG_SUMMARY_COLUMNS, fill_value=np.nan)
        mt_res_cpdag = mt_res_cpdag.reindex(columns=CPDAG_SUMMARY_COLUMNS, fill_value=np.nan)

        if save_res:
            logging.info(f'Saving results to {results_path}/stored_results_{version}.npy')
            save_summary_tables(results_path, version, mt_res, mt_res_cpdag)

print('Done')
