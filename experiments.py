import os
import logging
import argparse
from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

# Prefer package-relative imports to avoid accidentally importing modules from a
# different checkout (e.g., ArgCausalDisco-1) when PYTHONPATH is set broadly.
try:
    from .cd_algorithms.models import run_method
    from .utils.graph_utils import DAGMetrics, dag2cpdag, is_dag
    from .utils.helpers import random_stability, logger_setup
    from .utils.experiment_support import (
        CPDAG_BASE_COLUMNS,
        CPDAG_METRIC_MAP,
        CPDAG_PROGRESS_COLUMNS,
        CPDAG_SUMMARY_COLUMNS,
        DAG_BASE_COLUMNS,
        DAG_METRIC_MAP,
        DAG_PROGRESS_COLUMNS,
        DAG_SUMMARY_COLUMNS,
        append_progress_row,
        archive_run_artifacts,
        build_run_summary,
        format_run_indicator,
        install_signal_breadcrumbs,
        keep_causenet_file,
        load_existing_summary,
        load_progress_df,
        log_run_start,
        log_run_summary,
        record_run_manifest,
        safe_filename,
        save_summary_tables,
        str_to_bool,
        summarise_results,
    )
    from .utils.data_utils import (
        load_bnlearn_data_dag,
        simulate_dag,
        simulate_discrete_data,
        simulate_linear_continuous_data,
        BIF_FOLDER_MAP,
    )
except ImportError:  # pragma: no cover
    from cd_algorithms.models import run_method
    from utils.graph_utils import DAGMetrics, dag2cpdag, is_dag
    from utils.helpers import random_stability, logger_setup
    from utils.experiment_support import (
        CPDAG_BASE_COLUMNS,
        CPDAG_METRIC_MAP,
        CPDAG_PROGRESS_COLUMNS,
        CPDAG_SUMMARY_COLUMNS,
        DAG_BASE_COLUMNS,
        DAG_METRIC_MAP,
        DAG_PROGRESS_COLUMNS,
        DAG_SUMMARY_COLUMNS,
        append_progress_row,
        archive_run_artifacts,
        build_run_summary,
        format_run_indicator,
        install_signal_breadcrumbs,
        keep_causenet_file,
        load_existing_summary,
        load_progress_df,
        log_run_start,
        log_run_summary,
        record_run_manifest,
        safe_filename,
        save_summary_tables,
        str_to_bool,
        summarise_results,
    )
    from utils.data_utils import (
        load_bnlearn_data_dag,
        simulate_dag,
        simulate_discrete_data,
        simulate_linear_continuous_data,
        BIF_FOLDER_MAP,
    )
import warnings

warnings.filterwarnings("ignore")


def sample_uniform_random_dag(num_nodes: int, edge_prob: float = 0.5) -> np.ndarray:
    """Sample a random DAG on a fixed node set.

    The construction first samples a random topological order, then includes
    each admissible forward edge independently with probability ``edge_prob``.
    This keeps the baseline acyclic while avoiding any calibration to the
    ground-truth edge count.
    """

    if num_nodes <= 0:
        return np.zeros((0, 0), dtype=int)
    lower = (np.random.rand(num_nodes, num_nodes) < edge_prob).astype(int)
    lower = np.tril(lower, k=-1)
    perm = np.random.permutation(np.eye(num_nodes, dtype=int))
    dag = perm.T @ lower @ perm
    assert is_dag(dag)
    return dag.astype(int, copy=False)


# CLI
parser = argparse.ArgumentParser(
    description='Run causal discovery experiments on CauseNet or BNLearn datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--source', choices=['causenet', 'bnlearn'], required=True, help='Source of datasets to use')
parser.add_argument('--version', required=True, help='Version name for this experiment run')
parser.add_argument('--models', nargs='*', default=['random'], help='List of models to run')
parser.add_argument('--results_dir', default='results')

# CauseNet specific
parser.add_argument('--bifxml_dir', default=os.path.join('datasets', 'causenet_generator', 'bifxmls'), help='Folder with .bifxml graphs (for source=causenet)')
parser.add_argument('--simulate_with', choices=['internal', 'pyagrum'], default='internal', help='How to simulate data from DAG (causenet only)')

# BNLearn specific
parser.add_argument('--bn_data_path', default='datasets', help='Root data path for BNLearn .bif files')

# Random synthetic generation
parser.add_argument('--graph_type', choices=['ER', 'SF', 'BP'], default='ER', help='Random graph type (source=random)')
parser.add_argument('--size', action='append', help='Random: comma pair n,e (e=edges); can repeat, e.g., --size 10,15 --size 20,30')
parser.add_argument('--n_nodes_list', dest='n_nodes_list', nargs='*', type=int, help='Random: list of node counts, e.g., --n_nodes_list 5 10 20')
parser.add_argument('--density_list', dest='density_list', nargs='*', type=float, help='Random: list of edge densities (edges ≈ density*n), e.g., --density_list 1 2 4')
parser.add_argument('--sim_type', choices=['discrete', 'continuous'], default='discrete', help='Random: type of data simulation')
parser.add_argument('--noise_type', choices=['gaussian', 'exponential'], default='gaussian', help='Random (continuous): exogenous noise distribution')

# Common
parser.add_argument('--sample_size', type=int, default=5000)
parser.add_argument('--n_runs', type=int, default=50)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_res', action='store_true')
parser.add_argument('--save_res', action='store_true', default=True)
parser.add_argument('--resume', action='store_true', help='Resume from saved progress and summaries')
parser.add_argument('--standardise', type=str_to_bool, default=True, metavar='{true,false}', help='Standardise data (z-score) after label encoding')
parser.add_argument('--test_alpha', type=float, default=0.05, help='Significance level for conditional independence tests')
parser.add_argument('--test_name', choices=['fisherz', 'chisq', 'gsq', 'kci', 'fastkci', 'rcit'], default='gsq', help='Independence test to use')

## ABAPC specific
parser.add_argument('--S_weight', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC: use S_weight')
parser.add_argument('--pre_grounding', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC: use pre_grounding (slower for incremental; use only for baseline)')
parser.add_argument('--skeleton_rules_reduction', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC: use skeleton_rules_reduction')
parser.add_argument('--disable_reground', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC: disable regrounding')
parser.add_argument('--return_statistics', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC: return statistics')
parser.add_argument('--out_n', type=int, default=5, help='ABAPC: number of output models to request')   
parser.add_argument('--threads', type=int, default=None, help='ABAPC/CausalABA: clingo solver threads (-t); smaller uses less memory, default is to let clingo decide based on available cores')
parser.add_argument('--solve_timeout', type=float, default=None, help='ABAPC/CausalABA: optional wall-time limit per clingo solve call (seconds)')
parser.add_argument('--final-solve-timeout', '--final_solve_timeout', dest='final_solve_timeout', type=float, default=None, help='ABAPC/CausalABA incremental: optional wall-time limit for the final solve only (seconds); use 0 to disable even if --solve_timeout is set')
parser.add_argument('--final-solve-opt-mode', '--final_solve_opt_mode', dest='final_solve_opt_mode', choices=['ignore', 'opt', 'optN'], default=None, help='ABAPC/CausalABA incremental: opt_mode to use for the final solve only; defaults to the solver opt_mode')
parser.add_argument('--final-solve-n-models', '--final_solve_n_models', dest='final_solve_n_models', type=int, default=None, help='ABAPC/CausalABA incremental: models limit for the final solve only; defaults to --out_n')
parser.add_argument('--satcheck_timeout', type=float, default=None, help='ABAPC/CausalABA: optional wall-time limit per satcheck during removal search (seconds); defaults to --solve_timeout')
parser.add_argument('--satcheck_threads', type=int, default=None, help='ABAPC/CausalABA: solver threads to use for satchecks during removal search; defaults to --threads')
parser.add_argument('--satcheck_probe_limit', type=int, default=8, help='ABAPC/CausalABA: number of alternate removal counts to probe when a satcheck times out')
parser.add_argument('--satcheck_frontier_crawl', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC/CausalABA: bias plateau search toward the SAT frontier instead of plain midpoint search')
parser.add_argument('--satcheck_promoted_retry', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC/CausalABA: retry frontier-adjacent UNKNOWN satchecks with a larger timeout budget')
parser.add_argument('--satcheck_promoted_retry_timeout_scale', type=float, default=2.0, help='ABAPC/CausalABA: timeout multiplier used for promoted retries of UNKNOWN satchecks')
parser.add_argument('--satcheck_promoted_retry_max_retries', type=int, default=1, help='ABAPC/CausalABA: maximum number of promoted retries per UNKNOWN removal count')
parser.add_argument('--satcheck_retry_frontier_sat', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC/CausalABA: allow promoted retries on the SAT frontier; disabled by default because UNSAT-frontier retries are usually more informative on hard plateaus')
parser.add_argument('--satcheck_portfolio', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC/CausalABA: use a small plateau portfolio of alternative candidates before full probe fallback')
parser.add_argument('--satcheck_portfolio_size', type=int, default=3, help='ABAPC/CausalABA: number of candidates in each plateau portfolio batch')
parser.add_argument('--satcheck_portfolio_timeout_scale', type=float, default=0.5, help='ABAPC/CausalABA: timeout ratio for each plateau portfolio candidate')
parser.add_argument('--satcheck_portfolio_min_timeout', type=float, default=180.0, help='ABAPC/CausalABA: minimum timeout in seconds for each plateau portfolio candidate')
parser.add_argument('--satcheck_portfolio_throttle', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC/CausalABA: gently reduce portfolio batch size after repeated all-UNKNOWN portfolio batches in the same bracket')
parser.add_argument('--satcheck_portfolio_min_size', type=int, default=2, help='ABAPC/CausalABA: smallest portfolio batch size when plateau throttling is active')
parser.add_argument('--satcheck_plateau_stop', type=str_to_bool, default=True, metavar='{true,false}', help='ABAPC/CausalABA: stop early and accept the best SAT bound when the bracket is already narrow and the search is dominated by UNKNOWN outcomes')
parser.add_argument('--satcheck_plateau_stop_width_ratio', type=float, default=0.07, help='ABAPC/CausalABA: maximum certified bracket width as a fraction of total facts for the plateau stop rule')
parser.add_argument('--satcheck_plateau_stop_min_calls', type=int, default=40, help='ABAPC/CausalABA: minimum number of recorded satchecks before the plateau stop rule can trigger')
parser.add_argument('--satcheck_plateau_stop_unknown_ratio', type=float, default=0.80, help='ABAPC/CausalABA: minimum UNKNOWN call ratio needed to trigger the plateau stop rule')
parser.add_argument('--adaptive_satcheck_threads', type=str_to_bool, default=False, metavar='{true,false}', help='ABAPC/CausalABA: adapt satcheck threads using timeout+memory signals and persist recommendations across resumed runs')
parser.add_argument('--satcheck_min_threads', type=int, default=1, help='ABAPC/CausalABA: lower bound for adaptive satcheck thread tuning')
parser.add_argument('--satcheck_increase_step', type=int, default=2, help='ABAPC/CausalABA: additive thread increase used by the adaptive satcheck tuner after low-memory timeouts')
parser.add_argument('--abapc_solver', choices=['incremental', 'baseline'], default='incremental', help='Use incremental or baseline CausalABA inside ABAPC')
parser.add_argument('--eval_timeout', type=float, default=600.0, help='Maximum wall-time budget in seconds for each individual expensive evaluation metric (currently SHD and SID); use 0 to disable')

# Bounded Causal ABA parameters
parser.add_argument('--max_path_length', type=int, default=None, help='Bound: maximum simple path length |p| (lp)')
parser.add_argument('--max_conditioning_size', type=int, default=None, help='Bound: maximum conditioning set size |Z| (sz)')
parser.add_argument('--collider_tree_depth', type=int, default=None, help='Bound: collider-tree depth (lb)')
parser.add_argument('--cycle_length', type=int, default=None, help='Bound: acyclicity checks only cycles up to length (lcyc)')
parser.add_argument('--lp_ratio', type=float, default=None, help='Fraction of (n-1) for max_path_length')
parser.add_argument('--sz_ratio', type=float, default=None, help='Fraction of (n-2) for max_conditioning_size')
parser.add_argument('--lb_ratio', type=float, default=None, help='Fraction of (n-1) for collider_tree_depth')
parser.add_argument('--lcyc_ratio', type=float, default=None, help='Fraction of n for cycle_length')

# Subset filters (interpreted based on source)
parser.add_argument('--include', action='append', help='Substring to include; for causenet: file base name filters, for bnlearn: dataset name filters')
parser.add_argument('--glob', action='append', help='Glob pattern to include (causenet only for filenames)')
parser.add_argument('--regex', action='append', help='Regex to include (causenet only for filenames)')
parser.add_argument('--names', nargs='*', help='Exact names to include (causenet: file base names without path; bnlearn: dataset names)')
parser.add_argument('--nodes', type=int, choices=[5, 10, 15], help='CauseNet filter: only graphs with this number of nodes')
parser.add_argument('--edges_class', choices=['d', '1.5d'], help='CauseNet filter: only graphs with edges equal to d or ~1.5d')
parser.add_argument('--heur', choices=['none', 'degrees', 'semantics'], help='CauseNet filter: only graphs with this heuristic')
parser.add_argument('--type', dest='gtype', choices=['random', 'er', 'sf'], help='CauseNet filter: only graphs with this type')

args = parser.parse_args()

# Config
version = args.version
results_path = Path(args.results_dir)
results_path.mkdir(parents=True, exist_ok=True)
logger_setup(str(results_path / f'log_{version}.log'))
run_manifest_path = record_run_manifest(
    results_path=results_path,
    version=version,
    args=args,
    script_path=__file__,
)

# Log which code is actually executing/imported. This helps detect accidental
# execution from a different checkout (e.g., ArgCausalDisco vs ArgCausalDisco-1).
try:
    import cd_algorithms.models as _models_mod
    import abapc as _abapc_mod
    logging.info(f"experiments.py path: {__file__}")
    logging.info(f"cwd: {os.getcwd()}")
    logging.info(f"cd_algorithms.models path: {_models_mod.__file__}")
    logging.info(f"abapc.py path: {_abapc_mod.__file__}")
except Exception as _e:  # pragma: no cover
    logging.warning(f"Could not log import paths: {_e}")

logging.info(f"run_config manifest: {run_manifest_path}")


install_signal_breadcrumbs(results_path=results_path, version=version)

sample_size = args.sample_size
n_runs = args.n_runs
device = args.device
load_res = args.load_res
resume = args.resume
save_res = args.save_res
simulate_with = args.simulate_with
standardise = args.standardise

test_alpha = args.test_alpha
test_name = args.test_name
S_weight = args.S_weight
pre_grounding = args.pre_grounding
skeleton_rules_reduction = args.skeleton_rules_reduction
disable_reground = args.disable_reground
return_statistics = args.return_statistics
out_n = args.out_n
threads = args.threads
solve_timeout = args.solve_timeout
final_solve_timeout = args.final_solve_timeout
final_solve_opt_mode = args.final_solve_opt_mode
final_solve_n_models = args.final_solve_n_models
satcheck_timeout = args.satcheck_timeout
satcheck_threads = args.satcheck_threads
satcheck_probe_limit = args.satcheck_probe_limit
satcheck_frontier_crawl = args.satcheck_frontier_crawl
satcheck_promoted_retry = args.satcheck_promoted_retry
satcheck_promoted_retry_timeout_scale = args.satcheck_promoted_retry_timeout_scale
satcheck_promoted_retry_max_retries = args.satcheck_promoted_retry_max_retries
satcheck_retry_frontier_sat = args.satcheck_retry_frontier_sat
satcheck_portfolio = args.satcheck_portfolio
satcheck_portfolio_size = args.satcheck_portfolio_size
satcheck_portfolio_timeout_scale = args.satcheck_portfolio_timeout_scale
satcheck_portfolio_min_timeout = args.satcheck_portfolio_min_timeout
satcheck_portfolio_throttle = args.satcheck_portfolio_throttle
satcheck_portfolio_min_size = args.satcheck_portfolio_min_size
satcheck_plateau_stop = args.satcheck_plateau_stop
satcheck_plateau_stop_width_ratio = args.satcheck_plateau_stop_width_ratio
satcheck_plateau_stop_min_calls = args.satcheck_plateau_stop_min_calls
satcheck_plateau_stop_unknown_ratio = args.satcheck_plateau_stop_unknown_ratio
adaptive_satcheck_threads = args.adaptive_satcheck_threads
satcheck_min_threads = args.satcheck_min_threads
satcheck_increase_step = args.satcheck_increase_step
max_path_length = args.max_path_length
max_conditioning_size = args.max_conditioning_size
collider_tree_depth = args.collider_tree_depth
cycle_length = args.cycle_length
lp_ratio = args.lp_ratio
sz_ratio = args.sz_ratio
lb_ratio = args.lb_ratio
lcyc_ratio = args.lcyc_ratio
abapc_solver = args.abapc_solver
eval_timeout = None if args.eval_timeout is None or float(args.eval_timeout) <= 0 else float(args.eval_timeout)

logging.info(f"ABAPC solver selection: {abapc_solver}")
logging.info(f"Evaluation metric timeout: {eval_timeout if eval_timeout is not None else 'disabled'}")

model_list = args.models
names_dict = {
    'pc': 'PC',
    'pc_max': 'Max-PC',
    'fgs': 'FGS',
    'spc': 'Shapley-PC',
    'mpc': 'MPC',
    'cpc': 'CPC',
    'abapc': 'ABAPC (Ours)',
    'cam': 'CAM',
    'nt': 'NOTEARS-MLP',
    'mcsl': 'MCSL-MLP',
    'ges': 'GES',
    'random': 'Random',
    'rnd-dir': 'rnd-dir',
    'random_edge': 'Random (match |E|)'
}

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

GRAPH_METRIC_KEYS = [
    'nnz', 'fdr', 'tpr', 'fpr',
    'precision', 'recall', 'F1',
    'adjacency_precision', 'adjacency_recall', 'adjacency_F1',
    'arrowhead_precision', 'arrowhead_recall', 'arrowhead_F1',
    'shd',
]


def empty_metric_result():
    metrics = {key: np.nan for key in GRAPH_METRIC_KEYS}
    metrics['sid'] = np.nan
    return metrics


def graph_artifact_bundle(
    *,
    B_true,
    W_est=None,
    B_est_binary=None,
    B_est_dag_eval=None,
    B_est_cpdag_eval=None,
):
    artifacts = {'graph_true': np.asarray(B_true)}
    if W_est is not None:
        artifacts['graph_est_raw'] = np.asarray(W_est)
    if B_est_binary is not None:
        artifacts['graph_est_binary'] = np.asarray(B_est_binary)
    if B_est_dag_eval is not None:
        artifacts['graph_est_dag_eval'] = np.asarray(B_est_dag_eval)
    if B_est_cpdag_eval is not None:
        artifacts['graph_est_cpdag_eval'] = np.asarray(B_est_cpdag_eval)
    return artifacts

# Gather datasets according to source
datasets = []  # list of tuples: (dataset_name, loader, loader_kwargs)
if args.source == 'causenet':
    bifxml_dir = args.bifxml_dir
    bifxml_files = [
        os.path.join(bifxml_dir, fn) for fn in sorted(os.listdir(bifxml_dir)) if fn.endswith('.bifxml')
    ]
    bifxml_files = [p for p in bifxml_files if keep_causenet_file(p, args)]
    if not bifxml_files:
        raise SystemExit('No .bifxml files selected by the provided filters.')
    for graph_path in bifxml_files:
        dataset_name = os.path.splitext(os.path.basename(graph_path))[0]
        datasets.append((dataset_name, 'causenet', {'graph_path': graph_path}))
elif args.source == 'bnlearn':
    # BNLearn
    # Determine dataset names: default to all known if none provided; allow include filters
    all_bn = sorted(BIF_FOLDER_MAP.keys())
    if args.names:
        selected = [n for n in args.names if n in all_bn]
    else:
        selected = all_bn
    if args.include:
        selected = [n for n in selected if any(s in n for s in args.include)]
    if not selected:
        raise SystemExit('No BNLearn datasets selected. Use --names or --include to pick datasets.')
    for name in selected:
        datasets.append((name, 'bnlearn', {'dataset_name': name}))
else:
    # Random synthetic graphs
    sizes = []
    if args.size:
        for pair in args.size:
            try:
                n_str, e_str = pair.split(',')
                n = int(n_str.strip()); e = int(e_str.strip())
                sizes.append((n, e))
            except Exception:
                raise SystemExit(f'Invalid --size entry "{pair}". Use format n,e')
    if args.n_nodes_list and args.density_list:
        for n in args.n_nodes_list:
            for dens in args.density_list:
                e = max(n - 1, int(round(dens * n)))  # ensure at least a tree edge count
                sizes.append((n, e))
    if not sizes:
        raise SystemExit('For --source random, provide either --size n,e (can repeat) or both --n_nodes_list and --density_list.')
    for n, e in sizes:
        ds_name = f'random_n{n}_e{e}_{args.graph_type}'
        datasets.append((ds_name, 'random', {'n': n, 'e': e, 'graph_type': args.graph_type}))


# Main loop
for dataset_name, src, info in datasets:
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
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
        logging.debug(f'Seeds:{seeds_list}')

        if completed_runs:
            logging.info(f"Resuming {method} on {dataset_name}: {completed_runs}/{n_runs} runs already completed.")

        logging.info(f"Running {method} on {dataset_name}")

        for idx in range(completed_runs, n_runs):
            seed = seeds_list[idx]
            scenario = f"{method}_{version}_{dataset_name}"
            logging.info(
                "Starting run %s for %s on %s (seed=%s)",
                format_run_indicator(idx, n_runs),
                display_name,
                dataset_name,
                seed,
            )
            log_run_start(
                dataset_name=dataset_name,
                model_name=display_name,
                run_idx=idx,
                total_runs=n_runs,
                seed=seed,
                scenario=scenario if method == 'abapc' else None,
            )
            # Load data + true DAG
            std = True if standardise is None else standardise
            if src == 'bnlearn':
                X_s, B_true = load_bnlearn_data_dag(
                    info['dataset_name'], args.bn_data_path, sample_size,
                    seed=seed,
                    print_info=True if idx == 0 and completed_runs == 0 else False,
                    standardise=std,
                )
            elif src == 'random':
                # Generate a random ground-truth DAG and discrete data
                B_true = simulate_dag(d=info['n'], s0=info['e'], graph_type=info['graph_type'])
                edges = list(zip(*np.where(B_true == 1)))
                truth_edges = set((i, j) for i, j in edges)
                if args.sim_type == 'discrete':
                    X_s = simulate_discrete_data(
                        num_of_nodes=info['n'],
                        sample_size=sample_size,
                        truth_DAG_directed_edges=truth_edges,
                        random_seed=seed,
                    )
                else:
                    X_s = simulate_linear_continuous_data(
                        num_of_nodes=info['n'],
                        sample_size=sample_size,
                        truth_DAG_directed_edges=truth_edges,
                        noise_type=args.noise_type,
                        random_seed=seed,
                    )
            else:
                # For CausaNet (if present), fall back to BNLearn loader or extend here later
                X_s, B_true = load_bnlearn_data_dag(
                    info.get('dataset_name', 'asia'), args.bn_data_path, sample_size,
                    seed=seed,
                    print_info=True if idx == 0 and completed_runs == 0 else False,
                    standardise=std,
                )

            if method in {'random', 'rnd-dir', 'random_edge'}:
                random_stability(seed)
                start = datetime.now()
                run_details = None
                if method == 'random':
                    W_est = sample_uniform_random_dag(B_true.shape[1], edge_prob=0.5)
                    logging.info(f'Sampling uniform random DAG with {int(W_est.sum())} edges')
                elif method == 'rnd-dir':
                    s0 = np.random.randint(B_true.shape[1], (B_true.shape[1] * (B_true.shape[1] - 1)) // 2 + 1)
                    logging.info(f'Sampling random DAG with {s0} directed edges')
                    W_est = simulate_dag(d=B_true.shape[1], s0=s0, graph_type='ER')
                elif method == 'random_edge':
                    s0 = int(B_true.sum())
                    logging.info(f'Sampling random DAG matched to true |E|={s0}')
                    W_est = simulate_dag(d=B_true.shape[1], s0=s0, graph_type='ER')
                else:
                    raise ValueError(f'Unknown random method {method}')
                elapsed = (datetime.now() - start).total_seconds()
            else:
                return_run_details = method == 'abapc'
                run_output = run_method(
                    X_s, method, seed, test_alpha=test_alpha, test_name=test_name,
                    device=device, scenario=scenario,
                    run_label=format_run_indicator(idx, n_runs),
                    S_weight=S_weight, pre_grounding=pre_grounding,
                    skeleton_rules_reduction=skeleton_rules_reduction,
                    disable_reground=disable_reground,
                    return_statistics=return_statistics, out_n=out_n,
                    max_path_length=max_path_length,
                    max_conditioning_size=max_conditioning_size,
                    collider_tree_depth=collider_tree_depth,
                    cycle_length=cycle_length,
                    lp_ratio=lp_ratio,
                    sz_ratio=sz_ratio,
                    lb_ratio=lb_ratio,
                    lcyc_ratio=lcyc_ratio,
                    threads=threads,
                    solve_timeout=solve_timeout,
                    final_solve_timeout=final_solve_timeout,
                    final_solve_opt_mode=final_solve_opt_mode,
                    final_solve_n_models=final_solve_n_models,
                    satcheck_timeout=satcheck_timeout,
                    satcheck_threads=satcheck_threads,
                    satcheck_probe_limit=satcheck_probe_limit,
                    satcheck_frontier_crawl=satcheck_frontier_crawl,
                    satcheck_promoted_retry=satcheck_promoted_retry,
                    satcheck_promoted_retry_timeout_scale=satcheck_promoted_retry_timeout_scale,
                    satcheck_promoted_retry_max_retries=satcheck_promoted_retry_max_retries,
                    satcheck_retry_frontier_sat=satcheck_retry_frontier_sat,
                    satcheck_portfolio=satcheck_portfolio,
                    satcheck_portfolio_size=satcheck_portfolio_size,
                    satcheck_portfolio_timeout_scale=satcheck_portfolio_timeout_scale,
                    satcheck_portfolio_min_timeout=satcheck_portfolio_min_timeout,
                    satcheck_portfolio_throttle=satcheck_portfolio_throttle,
                    satcheck_portfolio_min_size=satcheck_portfolio_min_size,
                    satcheck_plateau_stop=satcheck_plateau_stop,
                    satcheck_plateau_stop_width_ratio=satcheck_plateau_stop_width_ratio,
                    satcheck_plateau_stop_min_calls=satcheck_plateau_stop_min_calls,
                    satcheck_plateau_stop_unknown_ratio=satcheck_plateau_stop_unknown_ratio,
                    adaptive_satcheck_threads=adaptive_satcheck_threads,
                    satcheck_min_threads=satcheck_min_threads,
                    satcheck_increase_step=satcheck_increase_step,
                    abapc_solver=abapc_solver,
                    return_run_details=return_run_details,
                )
                if return_run_details:
                    W_est, elapsed, run_details = run_output
                else:
                    W_est, elapsed = run_output
                    run_details = None
                if 'Tensor' in str(type(W_est)):
                    W_est = np.asarray([list(i) for i in W_est])
                logger_setup(str(results_path / f'log_{version}.log'), continue_logging=True)

            graph_artifacts = graph_artifact_bundle(B_true=B_true)
            B_est_binary = None
            B_est_cpdag_eval = None
            B_est_dag_eval = None
            dag_eval_status = {}
            cpdag_eval_status = {}
            if W_est is not None:
                W_est = np.asarray(W_est)
                graph_artifacts['graph_est_raw'] = W_est.copy()
                B_est_binary = (W_est != 0).astype(int)
                graph_artifacts['graph_est_binary'] = B_est_binary.copy()
                try:
                    B_est_cpdag_eval = dag2cpdag(B_est_binary.copy())
                    graph_artifacts['graph_est_cpdag_eval'] = B_est_cpdag_eval.copy()
                    cpdag_metrics = DAGMetrics(B_est_cpdag_eval, B_true, metric_timeout=eval_timeout)
                    mt_cpdag = cpdag_metrics.metrics
                    cpdag_eval_status = getattr(cpdag_metrics, 'eval_status', {}) or {}
                except Exception as e:
                    logging.error(f'DAGMetrics computation failed for CPDAG: {e}')
                    mt_cpdag = empty_metric_result()
                    cpdag_eval_status = {'graph_eval': {'status': 'error', 'error': str(e), 'timed_out': False}}

                B_est_dag_eval = (W_est > 0).astype(int)
                bidirected_mask = (B_est_dag_eval == 1) & (B_est_dag_eval.T == 1)
                if bidirected_mask.any():
                    logging.warning('Estimated graph contains bidirected edges; removing them before DAG metrics computation.')
                    B_est_dag_eval[bidirected_mask] = 0
                graph_artifacts['graph_est_dag_eval'] = B_est_dag_eval.copy()
                if is_dag(B_est_dag_eval):
                    try:
                        dag_metrics = DAGMetrics(B_est_dag_eval, B_true, metric_timeout=eval_timeout)
                        mt_dag = dag_metrics.metrics
                        dag_eval_status = getattr(dag_metrics, 'eval_status', {}) or {}
                    except Exception as e:
                        logging.error(f'DAGMetrics computation failed for DAG: {e}')
                        mt_dag = empty_metric_result()
                        dag_eval_status = {'graph_eval': {'status': 'error', 'error': str(e), 'timed_out': False}}
                else:
                    logging.warning('Estimated graph is not a DAG after bidirected edge removal; skipping DAG metrics for this run.')
                    mt_dag = empty_metric_result()
                    dag_eval_status = {'graph_eval': {'status': 'skipped_non_dag', 'timed_out': False}}
            else:
                mt_cpdag = empty_metric_result()
                mt_dag = empty_metric_result()
                dag_eval_status = {'graph_eval': {'status': 'skipped_no_estimate', 'timed_out': False}}
                cpdag_eval_status = {'graph_eval': {'status': 'skipped_no_estimate', 'timed_out': False}}

            for graph_kind, eval_status in (('dag', dag_eval_status), ('cpdag', cpdag_eval_status)):
                for metric_name, status in (eval_status or {}).items():
                    if not isinstance(status, dict):
                        continue
                    if status.get('status') == 'timeout':
                        logging.warning(
                            '[eval-timeout] graph=%s metric=%s timeout=%ss elapsed=%ss',
                            graph_kind,
                            metric_name,
                            status.get('timeout_sec'),
                            status.get('elapsed_sec'),
                        )
                    elif status.get('status') == 'error':
                        logging.warning(
                            '[eval-error] graph=%s metric=%s error=%s',
                            graph_kind,
                            metric_name,
                            status.get('error'),
                        )

            logging.info({'dataset': dataset_name, 'model': display_name, 'elapsed': elapsed, **mt_dag})
            logging.info({'dataset': dataset_name, 'model': display_name, 'elapsed': elapsed, **mt_cpdag})

            # Early validation: check for null SID on first run to catch config issues immediately
            if idx == completed_runs and mt_cpdag.get('sid') is None and method not in ['random', 'rnd-dir', 'random_edge']:
                logging.error(
                    f"SID is null for {display_name} on {dataset_name} (first run). "
                    f"This likely means R SID package is not installed or DAGMetrics is failing. "
                    f"Stopping to avoid wasting time on {n_runs} runs with missing metrics."
                )
                raise SystemExit(1)

            dag_row = {'dataset': dataset_name, 'model': display_name, 'elapsed': elapsed, **mt_dag, 'run_idx': idx, 'seed': seed}
            if isinstance(mt_cpdag.get('sid'), tuple):
                mt_sid_low, mt_sid_high = mt_cpdag['sid']
            else:
                mt_sid_low = mt_cpdag.get('sid')
                mt_sid_high = mt_cpdag.get('sid')
            mt_cpdag.pop('sid', None)
            mt_cpdag['sid_low'] = mt_sid_low
            mt_cpdag['sid_high'] = mt_sid_high
            cpdag_row = {'dataset': dataset_name, 'model': display_name, 'elapsed': elapsed, **mt_cpdag, 'run_idx': idx, 'seed': seed}

            run_summary = build_run_summary(
                dataset_name=dataset_name,
                model_name=display_name,
                scenario=scenario,
                run_idx=idx,
                total_runs=n_runs,
                seed=seed,
                elapsed=elapsed,
                run_details=run_details,
                dag_metrics=dag_row,
                cpdag_metrics=cpdag_row,
                dag_eval_status=dag_eval_status,
                cpdag_eval_status=cpdag_eval_status,
            )
            log_run_summary(run_summary)
            archived_run_dir = archive_run_artifacts(
                results_path=results_path,
                summary=run_summary,
                graph_artifacts=graph_artifacts,
            )
            if archived_run_dir is not None:
                logging.info("[run-summary] archived_run_artifacts=%s", archived_run_dir)

            append_progress_row(dag_progress_path, dag_row, DAG_PROGRESS_COLUMNS)
            append_progress_row(cpdag_progress_path, cpdag_row, CPDAG_PROGRESS_COLUMNS)

            dag_runs = pd.concat([dag_runs, pd.DataFrame([dag_row])], ignore_index=True)
            cpdag_runs = pd.concat([cpdag_runs, pd.DataFrame([cpdag_row])], ignore_index=True)

        if len(dag_runs) < n_runs or len(cpdag_runs) < n_runs:
            logging.info(
                f'Completed {len(dag_runs)}/{n_runs} runs for {display_name} on {dataset_name}; partial progress saved for resume.'
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
raise SystemExit(0)
