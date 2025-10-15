import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def find_datasets(heuristic_dir: Path):
  bifxml_paths = sorted(heuristic_dir.glob('*.bifxml'))
  return [p.stem for p in bifxml_paths]


def run_single(dataset: str, n_samples: int, alg: str, project_root: Path, heuristic_dir: Path, logdir: str, exclude_desc: bool):
  cmd = [
    sys.executable,
    str(project_root / 'run.py'),
    '--heuristic_dir', str(heuristic_dir),
    '--dataset', dataset,
    '--alg', alg,
    '--n_samples', str(n_samples),
    '--logdir', logdir,
  ]
  if exclude_desc:
    cmd.append('--exclude_desc')
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  print(f"\n===== {dataset} | stdout =====\n{result.stdout}")
  if result.returncode != 0:
    print(f"===== {dataset} | stderr (exit {result.returncode}) =====\n{result.stderr}")
  return result.returncode


def collect_metrics(dataset: str, n_samples: int, alg: str, logdir: Path):
  log_file = logdir / dataset / str(n_samples) / f"{alg}.json"
  if not log_file.exists():
    return None
  try:
    with open(log_file, 'r') as f:
      metrics = json.load(f)
  except Exception:
    return None
  metrics['dataset'] = dataset
  metrics['n_samples'] = n_samples
  metrics['alg'] = alg
  return metrics


def is_run_valid(dataset: str, n_samples: int, alg: str, logdir: Path) -> bool:
  """Check if a run is valid and complete."""
  base_path = logdir / dataset / str(n_samples)
  pred_adj_path = base_path / 'pred_adj.npy'
  true_adj_path = base_path / 'true_adj.npy'
  json_log_path = base_path / f"{alg}.json"

  if not (pred_adj_path.exists() and true_adj_path.exists() and json_log_path.exists()):
    return False

  try:
    json_data = json.loads(json_log_path.read_text())
    # Rerun if the key is missing or the value is False
    return json_data.get("Is estimated graph a DAG?", False)
  except (json.JSONDecodeError, FileNotFoundError):
    return False


def main():
  parser = argparse.ArgumentParser(description='Batch run heuristic datasets and summarize metrics.')
  parser.add_argument('--heuristic_dir', type=str, default='../bnlearn', help='Directory containing .bifxml heuristic datasets')
  parser.add_argument('--alg', type=str, default='llm_bfs_with_statistics', help='Algorithm to run')
  parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to generate/use')
  parser.add_argument('--project_root', type=str, default='.', help='Project root where run.py resides')
  parser.add_argument('--logdir', type=str, default='logs', help='Logs directory used by run.py')
  parser.add_argument('--save_csv', type=str, default=None, help='Optional path to save the aggregated CSV')
  parser.add_argument('--exclude_desc', action='store_true', help='Exclude descriptions when prompting LLMs.')
  args = parser.parse_args()

  heuristic_dir = Path(args.heuristic_dir).resolve()
  project_root = Path(args.project_root).resolve()
  logdir = Path(args.logdir).resolve()

  datasets = find_datasets(heuristic_dir)
  if not datasets:
    print(f"No .bifxml datasets found in {heuristic_dir}")
    sys.exit(1)

  all_rows = []
  
  while True:
    datasets_to_run = [
      ds for ds in datasets
      if not is_run_valid(ds, args.n_samples, args.alg, logdir)
    ]

    if datasets_to_run:
      print(f"Found {len(datasets_to_run)} datasets to run out of {len(datasets)} total.")
      for ds in tqdm(datasets_to_run, desc="Running datasets"):
        run_single(ds, args.n_samples, args.alg, project_root, heuristic_dir, logdir, args.exclude_desc)
    else:
      print("All datasets have already been run.")
      break

  print("\n>>> Collecting all metrics...")
  for ds in tqdm(datasets, desc="Collecting metrics"):
    metrics = collect_metrics(ds, args.n_samples, args.alg, logdir)
    if metrics is not None:
      all_rows.append(metrics)
    else:
      print(f"Warning: metrics not found for {ds}")

  if not all_rows:
    print("No metrics collected.")
    sys.exit(2)

  df = pd.DataFrame(all_rows)
  # Order columns if common keys exist
  preferred = ['dataset', 'n_samples', 'alg', 'precision', 'recall', 'F_score', 'accuracy', 'SID', 'NHD', 'REFERENCE NHD', 'RATIO', 'Number of predicted edges', 'Number of true edges']
  cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
  df = df[cols]

  print("\n===== Aggregated Results =====")
  print(df.to_string(index=False))

  save_path = args.save_csv
  if save_path is None:
    save_path = logdir / f"summary_{heuristic_dir.name}_{args.n_samples}_{args.alg}.csv"
  else:
    save_path = Path(save_path)
  df.to_csv(save_path, index=False)
  print(f"\nSaved summary to {save_path}")


if __name__ == '__main__':
  main()


