import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils.graph_utils import DAGMetrics, dag2cpdag

# TODO: Add impl for synthetic datasets as well.
cd_metrics = []
not_dag = []
sample_size = 5000
for i, path in enumerate(Path("causal-llm-bfs/logs").glob("*")):
    if not path.is_dir() or path.stem not in {
        "asia",
        "cancer",
        "sachs",
        "earthquake",
        "survey",
        "child",
    }:
        continue
    dataset = path.stem
    path = path / str(sample_size)
    B_est = np.load(path / "pred_adj.npy")
    B_true = np.load(path / "true_adj.npy")
    json_data = json.loads((path / "llm_bfs_with_statistics.json").read_text())
    pred_is_dag = json_data["Is estimated graph a DAG?"]
    if not pred_is_dag:
        print(i, dataset)
        not_dag.append(dataset)
        continue

    # DAG metrics
    mt_dag = DAGMetrics((B_est > 0).astype(int), B_true).metrics if pred_is_dag else {}

    # CPDAG metrics
    B_est_cpdag = (B_est != 0).astype(int)
    mt_cpdag = DAGMetrics(dag2cpdag(B_est_cpdag), B_true).metrics
    cpdag_sid = mt_cpdag.pop("sid")
    if not isinstance(cpdag_sid, tuple):
        cpdag_sid = (cpdag_sid, cpdag_sid)
    mt_cpdag["sid_low"], mt_cpdag["sid_high"] = cpdag_sid
    cd_metrics.append(
        {
            "dataset": dataset,
            **{f"dag_{k}": v for k, v in mt_dag.items()},
            **{f"cpdag_{k}": v for k, v in mt_cpdag.items()},
        }
    )

assert not not_dag
bfs_df = pd.DataFrame(cd_metrics)
bfs_df.insert(1, "time", 1)
bfs_df.insert(1, "impl", "llm-bfs")
bfs_df.to_csv("results/causal-bfs-bnlearn-results.csv", index=False)
