#!/usr/bin/env python3
"""Reevaluate saved raw graphs; no data simulation or learner execution."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import importlib.metadata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_aij_tables import Sources, load_archived_primary, DATASETS, SPECS, SEEDS, version_for, EDGES
from utils.aij_graph_metrics import decode_estimate, prepare_estimate, complete_cpdag, structural_scores, exact_sid_bounds, dag_sid, InvalidGraph


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sid-timeout',type=float,default=120.)
    args=ap.parse_args()
    out=ROOT/'results/aij_evaluation/v2';out.mkdir(parents=True,exist_ok=True)
    src=Sources(ROOT);old=load_archived_primary(src)
    files=[Path(__file__),ROOT/'utils/aij_graph_metrics.py']
    code={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    versions={x:importlib.metadata.version(x) for x in ('numpy','networkx','causal-learn','gadjid')}
    records=[];run_audit=[];started=time.monotonic()
    for dataset in DATASETS:
        for method,(key,_,_) in SPECS.items():
            version='bnlearn_baselines_matched10_gsq_graphmetrics' if key in ('mpc','fgs') else version_for(method,dataset)
            for seed in SEEDS:
                folder,=(ROOT/f'results/{key}_{version}_{dataset}/runs').glob(f'run_*_seed_{seed}')
                raw_path=folder/'graph_est_raw.npy';truth_path=folder/'graph_true.npy'
                target=out/'runs'/dataset/key/version/str(seed);target.mkdir(parents=True,exist_ok=True)
                identity={'code':code,'dependencies':versions,'raw_sha256':digest(raw_path),'truth_sha256':digest(truth_path),
                          'sid_timeout':args.sid_timeout}
                cache=target/'evaluation.json'
                saved=json.loads(cache.read_text()) if cache.exists() else None
                if saved is not None and saved['identity']==identity:
                    entries=saved['records'];audit=saved['audit']
                else:
                    t=time.monotonic();raw=np.load(raw_path);truth=np.load(truth_path)
                    entries=[];arrays={'truth':truth};audit={'dataset':dataset,'method':method,'seed':seed}
                    try:
                        dag,cpdag=prepare_estimate(raw,key,seed)
                        reference=complete_cpdag(truth)
                        arrays.update(dag=dag,cpdag=cpdag,reference_cpdag=reference)
                        sid=exact_sid_bounds(truth,cpdag,timeout=args.sid_timeout)
                        audit.update(graph_status='valid',sid_status=sid['sid_status'],sid_states=sid.get('sid_states'))
                        for name in ('sid_low_witness','sid_high_witness'):
                            if name in sid:arrays[name]=sid[name]
                        for kind,estimate,ref in [('dag',dag,truth),('cpdag',cpdag,reference)]:
                            row=old[(old.dataset==dataset)&(old.method==method)&(old.seed==seed)&(old.kind==kind)].iloc[0].to_dict()
                            for name in ('sid','sid_low','sid_high','nsid','nsid_low','nsid_high','fdr','tpr','fpr'):row[name]=np.nan
                            row.update(structural_scores(estimate,ref))
                            row.update(graph_status='valid',metric_protocol='aij_v2',evaluation_source=cache.relative_to(ROOT).as_posix())
                            if kind=='dag':row.update(sid=dag_sid(truth,dag),sid_status='exact')
                            else:row.update({k:sid[k] for k in ('sid_low','sid_high','sid_status')})
                            for metric in ('shd','sid','sid_low','sid_high'):row['n'+metric]=row.get(metric,np.nan)/EDGES[dataset]
                            entries.append(row)
                        np.savez_compressed(target/'graphs.npz',**arrays)
                    except InvalidGraph as exc:
                        audit.update(graph_status='invalid',sid_status='invalid_graph',error=str(exc))
                        # Edge comparisons remain defined even when the output
                        # does not represent a Markov equivalence class.
                        pdag=decode_estimate(raw,key)
                        reference=complete_cpdag(truth)
                        arrays.update(pdag=pdag,reference_cpdag=reference)
                        np.savez_compressed(target/'graphs.npz',**arrays)
                        for kind in ('dag','cpdag'):
                            row=old[(old.dataset==dataset)&(old.method==method)&(old.seed==seed)&(old.kind==kind)].iloc[0].to_dict()
                            for name in ('nnz','precision','recall','F1','adjacency_precision','adjacency_recall','adjacency_F1',
                                         'arrowhead_precision','arrowhead_recall','arrowhead_F1','shd','sid','sid_low','sid_high',
                                         'nshd','nsid','nsid_low','nsid_high','fdr','tpr','fpr'):row[name]=np.nan
                            if kind=='cpdag':
                                row.update(structural_scores(pdag,reference))
                                row['nshd']=row['shd']/EDGES[dataset]
                            row.update(graph_status='invalid',sid_status='invalid_graph',metric_protocol='aij_v2',evaluation_source=cache.relative_to(ROOT).as_posix())
                            entries.append(row)
                    audit['evaluation_seconds']=time.monotonic()-t
                    cache.write_text(json.dumps(clean({'identity':identity,'records':entries,'audit':audit}),indent=2,allow_nan=False)+'\n')
                records.extend(entries);run_audit.append(audit)
            print(f'{dataset}/{method}: done; wall {time.monotonic()-started:.1f}s',flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(out/'records.csv',index=False)
    pd.DataFrame(run_audit).to_csv(out/'run_audit.csv',index=False)
    manifest={'protocol':'aij_v2','code':code,'dependencies':versions,'records_sha256':digest(out/'records.csv'),
              'runs':len(run_audit),'graph_status_counts':pd.Series([r['graph_status'] for r in run_audit]).value_counts().to_dict(),
              'sid_status_counts':pd.Series([r['sid_status'] for r in run_audit]).value_counts().to_dict(),
              'fgs_caveat':'Historical FGS export used DOT adjacency; native endpoint/node-label provenance is absent. Reevaluation scores the saved DAG export; rerun FGS with corrected exporter for a native CPDAG comparison.'}
    (out/'manifest.json').write_text(json.dumps(clean(manifest),indent=2)+'\n')
    print(json.dumps(clean(manifest),indent=2))


if __name__=='__main__':main()
