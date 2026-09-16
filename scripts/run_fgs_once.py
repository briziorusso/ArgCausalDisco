#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FGS_BDEU_CONFIG = {
    'algorithm': 'Tetrad FGES', 'score': 'bdeu-score', 'data_type': 'discrete',
    'sample_prior': 15.0, 'structure_prior': 1.0, 'max_degree': -1,
    'faithfulness_assumed': True, 'symmetric_first_step': False,
}


def categorical_frame(values):
    """Encode observed categories, never bin continuous values implicitly."""
    values=np.asarray(values)
    frame=pd.DataFrame(index=range(len(values)))
    for i in range(values.shape[1]):
        codes,_=pd.factorize(values[:,i],sort=True)
        if (codes<0).any():
            raise ValueError('FGS-BDeu does not accept missing categories')
        frame[f'X{i+1}']=codes.astype(int)
    return frame


def tetrad_adjacency(edges, names):
    """Preserve Tetrad endpoints and the input column order; DOT drops both."""
    index={name:i for i,name in enumerate(names)}
    result=np.zeros((len(names),len(names)),dtype=int)
    for edge in edges:
        left,kind,right=str(edge).strip().split()
        i,j=index[left],index[right]
        if kind=='-->':result[i,j]=1
        elif kind=='<--':result[j,i]=1
        elif kind=='---':result[i,j]=result[j,i]=-1
        else:raise ValueError(f'Unexpected FGES edge: {edge}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FGS call in an isolated subprocess.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--score", choices=['sem-bic','bdeu-score'], default='sem-bic')
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
            frame=categorical_frame(X) if args.score=='bdeu-score' else pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
            score_options=({'samplePrior':FGS_BDEU_CONFIG['sample_prior'],
                            'structurePrior':FGS_BDEU_CONFIG['structure_prior'],
                            'symmetricFirstStep':False} if args.score=='bdeu-score' else {})
            search_start=time.perf_counter()
            fitted.run(
                algoId="fges",
                dfs=frame,
                scoreId=args.score,
                dataType='discrete' if args.score=='bdeu-score' else 'continuous',
                maxDegree=-1,
                faithfulnessAssumed=True,
                verbose=False,
                **score_options,
            )
            meta['search_elapsed']=time.perf_counter()-search_start
            meta['score']=args.score

            W_est = tetrad_adjacency(fitted.getEdges(), [f'X{i+1}' for i in range(X.shape[1])])

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
