import bnlearn as bn
from algs.llm.pairwise import llm_pairwise 
from algs.llm.bfs import llm_bfs
import numpy as np
from metrics import compute_metrics
from args import get_args
from data.var_names_and_desc import *
from data.var_names_and_desc import load_heuristic_dataset_varmap
import json
import os
import pandas as pd
try:
  import pyagrum as gum
except Exception:
  gum = None
from data.init_prompts import *

args = get_args()

def _heuristic_bifxml_path(ds, base_dir):
  base = base_dir
  candidate = os.path.join(base, f'{ds}.bifxml')
  if os.path.exists(candidate):
    return candidate
  return None

def _parse_edges_from_bifxml(bifxml_path):
  if gum is None:
    raise ImportError("pyAgrum is required to parse BIFXML files. Please install pyAgrum.")
  bn = gum.loadBN(bifxml_path)
  # Alphabetical order of variable names
  names_sorted = sorted(bn.names())
  id_order = [bn.idFromName(n) for n in names_sorted]
  adj = bn.adjacencyMatrix()
  # reorder to alphabetical
  adj = adj[np.ix_(id_order, id_order)].astype(int)
  return names_sorted, adj


heur_bif = _heuristic_bifxml_path(args.dataset, args.heuristic_dir)
if args.dataset == "neuropathic":
  adj_mat = np.load('./data/neuropathic_dag_gt.npy')
  df = pd.read_csv(f'./data/neuropathic_data_{args.n_samples}.csv')
elif heur_bif is not None:
  # Load variables/descriptions and edges from BIFXML
  if gum is None:
    raise ImportError("pyAgrum is required to use heuristic BIFXML datasets. Please install pyagrum.")
  bn = gum.loadBN(heur_bif)
  # Build adjacency in alphabetical order
  names_sorted = sorted(bn.names())
  id_order = [bn.idFromName(n) for n in names_sorted]
  adj_mat = bn.adjacencyMatrix()
  adj_mat = adj_mat[np.ix_(id_order, id_order)].astype(int)
  # Generate samples using pyAgrum and align columns to alphabetical order
  gum.initRandom(2025)
  df = gum.generateSample(bn, args.n_samples, with_labels=False, random_order=False)[0]
  df = df[names_sorted]
else:
  dag = bn.import_DAG(f'./data/{args.dataset}.bif')
  var_names = dag['adjmat'].columns
  adj_mat = dag['adjmat'].to_numpy().astype(int)
  df = bn.sampling(dag, n=args.n_samples)

data = df.to_numpy()
gaussian_noise = np.random.normal(loc=0.0, scale=0.00001, size=data.shape)

if args.alg == 'llm_pairwise':
  var_map = VAR_NAMES_AND_DESC.get(args.dataset)
  if var_map is None:
    var_map = load_heuristic_dataset_varmap(args.dataset, base_dir=args.heuristic_dir)
  predicted_adj_mat = llm_pairwise(var_map, prompts.get(args.dataset, prompts['asia']), df, include_statistics=False)
elif args.alg == 'llm_pairwise_with_statistics':
  var_map = VAR_NAMES_AND_DESC.get(args.dataset)
  if var_map is None:
    var_map = load_heuristic_dataset_varmap(args.dataset, base_dir=args.heuristic_dir)
  predicted_adj_mat = llm_pairwise(var_map, prompts.get(args.dataset, prompts['asia']), df, include_statistics=True)
elif args.alg == 'llm_bfs':
  var_map = VAR_NAMES_AND_DESC.get(args.dataset)
  if var_map is None:
    var_map = load_heuristic_dataset_varmap(args.dataset, base_dir=args.heuristic_dir)
  predicted_adj_mat = llm_bfs(var_map, args.dataset, df, include_statistics=False)
elif args.alg == 'llm_bfs_with_statistics':
  var_map = VAR_NAMES_AND_DESC.get(args.dataset)
  if var_map is None:
    var_map = load_heuristic_dataset_varmap(args.dataset, base_dir=args.heuristic_dir)
  predicted_adj_mat = llm_bfs(var_map, args.dataset, df, include_statistics=True)
else:
  raise ValueError(f"Unknown algorithm {args.alg}")


metrics = compute_metrics(adj_mat, predicted_adj_mat)
logdir = f"{args.logdir}/{args.dataset}/{args.n_samples}"
os.makedirs(logdir, exist_ok=True)
logfile = f"{logdir}/{args.alg}" 
if args.alg in ['dagma_nonlinear', 'dagma_linear', 'notears']:
  logfile += f"_lambda1={args.lambda1}"
logfile += ".json"
with open(logfile, "w") as outfile: 
    json.dump(metrics, outfile)

# Save adjacency matrices for reference
np.save(f"{logdir}/true_adj.npy", adj_mat)
np.save(f"{logdir}/pred_adj.npy", predicted_adj_mat)

# Print true adjacency and final metrics
print("True adjacency matrix (ground truth):")
print(adj_mat)
print("\nPredicted adjacency matrix:")
print(predicted_adj_mat)
print("\nFinal metrics:")
print(json.dumps(metrics, indent=2, default=str))
