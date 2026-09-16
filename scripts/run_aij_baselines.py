#!/usr/bin/env python3
"""Launch, resume and evaluate matched AIJ ASPCR-DAG or discrete-score FGS.

See README_AIJ.md in this directory. No external solver is invoked with --plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_matched_baseline_experiments as runner
from utils.aij_graph_metrics import complete_cpdag, dag_sid, exact_sid_bounds, prepare_estimate, structural_scores

SEEDS=[7816,3578,2656,2688,2494,183,7977,3199,316,8266]
DATASETS=['cancer','earthquake','survey','asia','sachs','child']
METHODS={'aspcr':'aspcr_log_dag','fgs':'fgs_bdeu'}
VERSIONS={'aspcr':'aij_aspcr_dag_native_n5000_matched10_v2',
          'fgs':'aij_fgs_bdeu_n5000_matched10_v2'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands(args):
    shared=['--version',VERSIONS[args.method],'--results-dir',str(args.results_dir),
            '--datasets',*args.datasets,'--seeds',*map(str,SEEDS),
            '--sample-size','5000','--test-alpha','0.05']
    launch=[sys.executable,'-u',str(ROOT/'scripts/run_matched_baseline_experiments.py'),
            *shared,'--methods',METHODS[args.method],'--resume','--aij-inputs']
    audit=None
    if args.method=='aspcr':
        launch+=['--aspcr-max-nodes','6','--aspcr-r-dir',str(args.aspcr_r_dir),'--rscript',args.rscript]
        if args.clingo_bin_dir: launch+=['--clingo-bin-dir',args.clingo_bin_dir]
        audit=[sys.executable,'-u',str(ROOT/'scripts/validate_matched_aspcr_results.py'),*shared]
    return launch,audit


def check_inputs(datasets):
    """Use stored samples where available; verify labels against archived truth."""
    inputs=[]
    sample_args=SimpleNamespace(bn_standardise=True,bn_data_path='datasets',aij_inputs=True)
    for spec in runner._dataset_specs(datasets):
        for seed in SEEDS:
            X,B=runner._load_data(spec,sample_size=5000,seed=seed,args=sample_args)
            if X.shape!=(5000,spec.n_nodes) or not np.isfinite(X).all():
                raise ValueError(f'Invalid sample: {spec.name}/{seed}')
            folder,=(ROOT/f'results/mpc_bnlearn_baselines_matched10_gsq_graphmetrics_{spec.name}/runs').glob(f'run_*_seed_{seed}')
            truth_path=folder/'graph_true.npy'
            if not np.array_equal(B,np.load(truth_path)):
                raise ValueError(f'True graph/node order changed: {spec.name}/{seed}')
            archived=ROOT/f'datasets/data_npy/data_{spec.name}_{seed}.npy'
            csv=ROOT/f'results/aspcr/data/data_{spec.name}_{seed}.csv'
            if csv.exists() and not np.allclose(X,pd.read_csv(csv,header=None).to_numpy(),rtol=1e-12,atol=1e-12):
                raise ValueError(f'Input differs from archived AIJ CSV: {csv}')
            inputs.append({'dataset':spec.name,'seed':seed,
                           'sample_source':'archived' if archived.exists() else 'regenerated',
                           'sample_sha256':hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()).hexdigest(),
                           'truth_matrix_sha256':hashlib.sha256(np.asarray(B,dtype='<i8').tobytes()).hexdigest(),
                           'truth_sha256':digest(truth_path)})
    return inputs


def preflight(args):
    files=[Path(__file__),ROOT/'scripts/run_matched_baseline_experiments.py',
           ROOT/'utils/aij_graph_metrics.py',ROOT/'utils/graph_utils.py',ROOT/'utils/data_utils.py']
    if args.method=='aspcr':
        if not args.aspcr_r_dir or not (args.aspcr_r_dir/'load.R').is_file():
            raise SystemExit('Set ASPCR_R_DIR to the external ASPCR R directory containing load.R.')
        encoding=args.aspcr_r_dir.parent/'ASP/new_wmaxsat_acyclic_sufficient.pl'
        if not encoding.is_file(): raise SystemExit(f'Missing DAG-only encoding: {encoding}')
        executable=shutil.which(args.rscript)
        if not executable: raise SystemExit(f'Rscript not found: {args.rscript}')
        env=os.environ.copy()
        if args.clingo_bin_dir: env['PATH']=args.clingo_bin_dir+os.pathsep+env.get('PATH','')
        if not shutil.which('clingo',path=env.get('PATH')): raise SystemExit('clingo not found; set CLINGO_BIN_DIR.')
        subprocess.run([executable,'--vanilla','-e',
                        f"setwd({json.dumps(str(args.aspcr_r_dir))}); source('load.R'); library(stringr)"],
                       check=True,env=env,timeout=120)
        files += [encoding,ROOT/'cd_algorithms/run_aspcr_csvdata.R',ROOT/'scripts/validate_matched_aspcr_results.py']
    else:
        if importlib.util.find_spec('pycausal') is None:
            raise SystemExit('pycausal is missing; activate the working FGS/Java environment.')
        # A small discrete search verifies Java and the installed Tetrad score ID.
        x=np.tile(np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]]),(10,1))
        runner._run_fgs_bdeu(x)
        files += [ROOT/'scripts/run_fgs_once.py']
    return check_inputs(args.datasets),files


def evaluate_completed(args,inputs):
    """Fail on incomplete cohorts; save exact SID and edge scores separately."""
    records=[];sources={}
    version=VERSIONS[args.method];method=METHODS[args.method]
    out=args.results_dir/'aij_evaluation/reruns'/version
    out.mkdir(parents=True,exist_ok=True)
    expected={(r['dataset'],r['seed']):r for r in inputs}
    metadata_path=args.results_dir/f'metadata_{version}.json'
    metadata=json.loads(metadata_path.read_text())
    if metadata['graph_metric_protocol']!='aij_v2': raise ValueError('Unexpected metric protocol')
    sources[str(metadata_path)]=digest(metadata_path)
    for dataset in args.datasets:
        prefix=args.results_dir/'progress'/version/f'{dataset}__{method}'
        frames=[runner._read_progress(Path(str(prefix)+f'_{kind}.csv')) for kind in ('dag','cpdag')]
        if runner._successful_seed_ids(*frames)!=set(SEEDS):
            raise RuntimeError(f'{dataset}: incomplete successful seed cohort; repeat the launch command to resume')
        for frame in frames:
            if frame.seed.duplicated().any(): raise ValueError(f'{dataset}: duplicate seed records')
        for row in frames[0].itertuples():
            path=Path(row.raw_graph_path)
            sources[str(path)]=digest(path)
            with np.load(path,allow_pickle=False) as saved:
                info=json.loads(str(saved['metadata'].item()))
                if (info['dataset'],int(info['seed']),info['method'],int(info['sample_size']))!=(dataset,int(row.seed),method,5000):
                    raise ValueError(f'Artifact identity mismatch: {path}')
                if info['sample_sha256']!=expected[dataset,int(row.seed)]['sample_sha256']:
                    raise ValueError(f'Input hash mismatch: {dataset}/{row.seed}')
                if args.method=='fgs' and info['fgs_bdeu']!=runner.FGS_BDEU_CONFIG:
                    raise ValueError('FGS score settings changed')
                raw,truth=saved['W_est'],saved['B_true']
                if hashlib.sha256(np.asarray(truth,dtype='<i8').tobytes()).hexdigest()!=expected[dataset,int(row.seed)]['truth_matrix_sha256']:
                    raise ValueError(f'True graph mismatch: {path}')
                dag,cpdag=prepare_estimate(raw,'fgs' if args.method=='fgs' else 'abapc',seed=int(row.seed))
                reference=complete_cpdag(truth)
                bounds=exact_sid_bounds(truth,cpdag,timeout=120)
                if bounds['sid_status']!='exact':
                    raise RuntimeError(f'Exact SID budget exhausted: {dataset}/{row.seed}')
                witness=out/f'{dataset}_{row.seed}_graphs.npz'
                np.savez_compressed(witness,truth=truth,dag=dag,cpdag=cpdag,
                                    sid_low_witness=bounds['sid_low_witness'],sid_high_witness=bounds['sid_high_witness'])
                for kind,estimate,ref in [('dag',dag,truth),('cpdag',cpdag,reference)]:
                    record={'dataset':dataset,'method':method,'seed':int(row.seed),'kind':kind,
                            'elapsed':float(row.elapsed),'metric_protocol':'aij_v2',**structural_scores(estimate,ref)}
                    if kind=='dag':record['sid']=dag_sid(truth,dag)
                    else:record.update({k:bounds[k] for k in ('sid_low','sid_high')})
                    for metric in ('shd','sid','sid_low','sid_high'):
                        if metric in record:record['n'+metric]=record[metric]/int(truth.sum())
                    records.append(record)
    target=out/'records.csv'
    pd.DataFrame(records).to_csv(target,index=False)
    report={'protocol':'aij_v2','version':version,'runs':len(records)//2,
            'records_sha256':digest(target),'sources':sources,
            'code':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'utils/aij_graph_metrics.py']}}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Validated {len(records)//2} runs; exact evaluation: {target}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--aspcr-r-dir',type=Path,default=os.environ.get('ASPCR_R_DIR'))
    parser.add_argument('--rscript',default=os.environ.get('RSCRIPT','Rscript'))
    parser.add_argument('--clingo-bin-dir',default=os.environ.get('CLINGO_BIN_DIR'))
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    args.datasets=args.datasets or (DATASETS[:3] if args.method=='aspcr' else DATASETS)
    if args.method=='aspcr' and any(d not in DATASETS[:3] for d in args.datasets):
        parser.error('ASPCR launch is limited to Cancer, Earthquake and Survey')
    args.results_dir=args.results_dir.expanduser().resolve()
    if args.aspcr_r_dir:args.aspcr_r_dir=args.aspcr_r_dir.expanduser().resolve()
    os.chdir(ROOT)
    launch,audit=commands(args)
    for dataset in args.datasets:
        prefix=args.results_dir/'progress'/VERSIONS[args.method]/f'{dataset}__{METHODS[args.method]}'
        completed=runner._successful_seed_ids(*(runner._read_progress(Path(str(prefix)+f'_{k}.csv')) for k in ('dag','cpdag')))
        print(f'{dataset}: pending seeds {[s for s in SEEDS if s not in completed]}',flush=True)
    if args.plan:
        print(json.dumps({'launch':launch,'validate':audit},indent=2));return 0
    inputs,files=preflight(args)
    if args.preflight_only:
        print(f'Preflight passed: {len(inputs)} matched inputs');return 0
    provenance={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'commands':[launch,audit],'inputs':inputs,'files':{str(p):digest(p) for p in files}}
    directory=args.results_dir/'matched10';directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (directory/f'{VERSIONS[args.method]}_{stamp}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    subprocess.run(launch,check=True,cwd=ROOT)
    if audit:subprocess.run(audit,check=True,cwd=ROOT)
    evaluate_completed(args,inputs)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
