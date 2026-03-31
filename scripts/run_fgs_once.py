#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pydot  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FGS call in an isolated subprocess.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    meta_path = Path(args.meta)

    start = time.time()
    meta: dict[str, object] = {"ok": False, "elapsed": None}

    try:
        from pycausal.pycausal import pycausal as pyc
        from pycausal import search as s

        X = np.load(input_path, allow_pickle=True)
        jm = pyc()
        jm.start_vm()
        try:
            fitted = s.tetradrunner()
            fitted.run(
                algoId="fges",
                dfs=pd.DataFrame(X, columns=[f"X{c}" for c in range(1, X.shape[1] + 1)]),
                scoreId="sem-bic",
                dataType="continuous",
                maxDegree=-1,
                faithfulnessAssumed=True,
                verbose=False,
            )

            graph = fitted.getTetradGraph()
            dot_str = jm.tetradGraphToDot(graph)
            graphs = pydot.graph_from_dot_data(dot_str)
            W_est = nx.adjacency_matrix(nx.nx_pydot.from_pydot(graphs[0])).todense()

            if W_est.shape[0] != X.shape[1]:
                g = nx.nx_pydot.from_pydot(graphs[0])
                g.add_nodes_from([f"X{d}" for d in range(1, X.shape[1] + 1)])
                W_est = nx.adjacency_matrix(g).todense()

            np.save(output_path, np.asarray(W_est))
            meta["ok"] = True
        finally:
            try:
                jm.stop_vm()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover
        meta["error"] = str(exc)
    finally:
        meta["elapsed"] = float(time.time() - start)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main()
