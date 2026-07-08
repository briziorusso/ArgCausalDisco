import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.graph_utils import DAGMetrics, estimate_to_cpdag_for_metrics


def dataset_names(source_dir: Path) -> set[str]:
    return {path.stem for path in source_dir.glob("*.bifxml")}


def export_source(
    source_name: str,
    source_dir: Path,
    logs_dir: Path,
    output_path: Path,
    sample_size: int,
    alg: str,
) -> None:
    names = dataset_names(source_dir)
    if not names:
        print(f"Skipping {source_name}: no .bifxml files found in {source_dir}")
        return

    rows = []
    not_dag = []
    missing = []

    for dataset in sorted(names):
        run_dir = logs_dir / dataset / str(sample_size)
        pred_path = run_dir / "pred_adj.npy"
        true_path = run_dir / "true_adj.npy"
        json_path = run_dir / f"{alg}.json"
        if not (pred_path.exists() and true_path.exists() and json_path.exists()):
            missing.append(dataset)
            continue

        B_est = np.load(pred_path)
        B_true = np.load(true_path)
        json_data = json.loads(json_path.read_text())
        pred_is_dag = json_data["Is estimated graph a DAG?"]
        if not pred_is_dag:
            not_dag.append(dataset)
            continue

        mt_dag = DAGMetrics((B_est > 0).astype(int), B_true).metrics

        mt_cpdag = DAGMetrics(estimate_to_cpdag_for_metrics(B_est), B_true).metrics
        cpdag_sid = mt_cpdag.pop("sid")
        if not isinstance(cpdag_sid, tuple):
            cpdag_sid = (cpdag_sid, cpdag_sid)
        mt_cpdag["sid_low"], mt_cpdag["sid_high"] = cpdag_sid

        rows.append(
            {
                "dataset": dataset,
                "impl": "llm-bfs",
                "time": 1,
                **{f"dag_{k}": v for k, v in mt_dag.items()},
                **{f"cpdag_{k}": v for k, v in mt_cpdag.items()},
            }
        )

    if not_dag:
        raise SystemExit(f"{source_name}: estimated graph was not a DAG for {not_dag}")
    if missing:
        print(f"{source_name}: missing {len(missing)} completed BFS logs: {missing}")
    if not rows:
        print(f"{source_name}: no completed BFS rows found; not writing {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Causal-LLM-BFS logs to compact paper CSV files."
    )
    parser.add_argument("--logs_dir", default="causal-llm-bfs/logs")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--alg", default="llm_bfs_with_statistics")
    parser.add_argument("--sources", nargs="*", default=["bnlearn", "synthetic"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = (REPO_ROOT / args.logs_dir).resolve()
    source_map = {
        "bnlearn": (
            REPO_ROOT / "bnlearn",
            REPO_ROOT / "results" / "causal-bfs-bnlearn-results.csv",
        ),
        "synthetic": (
            REPO_ROOT / "synthetic",
            REPO_ROOT / "results" / "causal-bfs-synthetic-results.csv",
        ),
    }

    for source in args.sources:
        if source not in source_map:
            raise SystemExit(f"Unknown source {source!r}; choose from {sorted(source_map)}")
        source_dir, output_path = source_map[source]
        export_source(
            source_name=source,
            source_dir=source_dir,
            logs_dir=logs_dir,
            output_path=output_path,
            sample_size=args.sample_size,
            alg=args.alg,
        )


if __name__ == "__main__":
    main()
