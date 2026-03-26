from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causalaba import compile_and_ground
from utils.graph_utils import extract_test_elements_from_symbol

RESULTS_DIR = REPO_ROOT / 'results'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Counterfactual reground analysis for ABAPC problem datasets')
    parser.add_argument('--datasets', nargs='+', default=['asia', 'sachs', 'survey'])
    parser.add_argument('--bb-version', default='abapc_bb_problem3_matched10_gsq_searchv3')
    parser.add_argument('--nor-version', default='abapc_nor_problem3_matched10_gsq')
    parser.add_argument('--orig-version', default='abapc_orig_problem3_matched10_gsq')
    parser.add_argument('--output-prefix', default='abapc_problem3_reground_effects')
    parser.add_argument('--threads', type=int, default=1)
    return parser.parse_args()


def scenario_dir(version: str, dataset: str) -> Path:
    return RESULTS_DIR / f'abapc_{version}_{dataset}'


def load_run_summaries(version: str, dataset: str) -> list[tuple[Path, dict]]:
    runs_dir = scenario_dir(version, dataset) / 'runs'
    if not runs_dir.exists():
        return []
    out: list[tuple[Path, dict]] = []
    for summary_path in sorted(runs_dir.glob('run_*_seed_*/run_summary.json')):
        out.append((summary_path.parent, json.loads(summary_path.read_text())))
    return out


def load_fact_entries(facts_i_path: Path) -> list[tuple[int, tuple[int, ...], int, str, str]]:
    entries: list[tuple[int, tuple[int, ...], int, str, str]] = []
    for raw_line in facts_i_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('%'):
            continue
        if ' I=' in line:
            statement = line.split(' I=', 1)[0].strip()
        else:
            statement = line
        if not statement:
            continue
        x, s, y, dep_type = extract_test_elements_from_symbol(statement)
        entries.append((int(x), tuple(sorted(s)), int(y), str(dep_type), statement.strip()))
    return entries


def build_fact_groups(entries: Iterable[tuple[int, tuple[int, ...], int, str, str]]):
    indep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    dep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
    for x, s, y, dep_type, _stmt in entries:
        target = indep_facts if 'indep' in dep_type else dep_facts
        target.setdefault((x, y), set()).add(tuple(sorted(s)))
    return indep_facts, dep_facts


def count_specific_rules(dump_path: Path) -> dict[str, int]:
    counts = Counter()
    for raw_line in dump_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('%'):
            continue
        counts['specific_total'] += 1
        if line.startswith('p') and ':-' in line and line[1:2].isdigit():
            counts['path_rules'] += 1
        elif line.startswith('ap('):
            counts['ap_rules'] += 1
        elif line.startswith('dep(') and ':-' in line and 'ap(' in line:
            counts['dep_support_rules'] += 1
        elif line.startswith('indep(') and ':-' in line and 'ap(' in line:
            counts['indep_support_rules'] += 1
        elif line.startswith(':- edge('):
            counts['blocked_edge_constraints'] += 1
        elif line.startswith('in('):
            counts['set_membership_rules'] += 1
    return {k: int(v) for k, v in counts.items()}


def compile_state(
    *,
    n_nodes: int,
    indep_facts: dict[tuple[int, int], set[tuple[int, ...]]],
    dep_facts: dict[tuple[int, int], set[tuple[int, ...]]],
    pre_grounding: bool,
    skeleton_rules_reduction: bool,
    threads: int,
) -> dict:
    timing: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / 'specific.lp'
        ctl = compile_and_ground(
            n_nodes=n_nodes,
            facts_location='',
            skeleton_rules_reduction=skeleton_rules_reduction,
            weak_constraints=False,
            indep_facts=indep_facts,
            dep_facts=dep_facts,
            opt_mode='ignore',
            out_n=0,
            show=['arrow'],
            pre_grounding=pre_grounding,
            ext_flag=False,
            prior_knowledge=None,
            threads=threads,
            timing_recorder=timing,
            dump_specific=str(dump_path),
        )
        profile = getattr(ctl, '_causalaba_profile', {}) or {}
        rule_counts = count_specific_rules(dump_path)
    return {
        'paths_added': int(profile.get('paths_added', 0) or 0),
        'pairs_considered': int(profile.get('pairs_considered', 0) or 0),
        'compile_sec': float(profile.get('compile_sec', 0.0) or 0.0),
        'ground_sec': float(profile.get('ground_sec', 0.0) or 0.0),
        **rule_counts,
    }


def summarize_method(dataset: str, method: str, version: str, threads: int) -> list[dict]:
    rows: list[dict] = []
    for run_dir, summary in load_run_summaries(version, dataset):
        facts_i = run_dir / 'facts_I.lp'
        if not facts_i.exists():
            continue
        entries = load_fact_entries(facts_i)
        removed_keys = set(summary.get('facts', {}).get('removed_fact_keys', []))
        kept_entries = [entry for entry in entries if entry[4] not in removed_keys]
        indep_init, dep_init = build_fact_groups(entries)
        indep_final, dep_final = build_fact_groups(kept_entries)
        pre_grounding = bool(summary.get('reground', {}).get('pre_grounding', summary.get('pre_grounding', False)))
        skeleton_rules_reduction = bool(summary.get('reground', {}).get('skeleton_rules_reduction', summary.get('skeleton_rules_reduction', True)))
        n_nodes = len(summary.get('cpdag_nodes', [])) if summary.get('cpdag_nodes') else len(set([e[0] for e in entries] + [e[2] for e in entries]))

        initial_counts = compile_state(
            n_nodes=n_nodes,
            indep_facts=indep_init,
            dep_facts=dep_init,
            pre_grounding=pre_grounding,
            skeleton_rules_reduction=skeleton_rules_reduction,
            threads=threads,
        )
        final_counts = compile_state(
            n_nodes=n_nodes,
            indep_facts=indep_final,
            dep_facts=dep_final,
            pre_grounding=pre_grounding,
            skeleton_rules_reduction=skeleton_rules_reduction,
            threads=threads,
        )

        row = {
            'dataset': dataset,
            'method': method,
            'seed': int(summary.get('seed', -1)),
            'run_idx': int(summary.get('run_idx', 0)),
            'solver_backend': summary.get('solver_backend', ''),
            'pre_grounding': pre_grounding,
            'disable_reground': bool(summary.get('reground', {}).get('disable_reground', summary.get('disable_reground', False))),
            'removed_fact_count': int(summary.get('facts', {}).get('removed_fact_count', 0) or 0),
            'fully_released_pair_count': int(summary.get('facts', {}).get('fully_released_pair_count', 0) or 0),
            'reground_skipped_count': int(summary.get('reground', {}).get('reground_skipped_count', 0) or 0),
        }
        for prefix, counts in [('initial', initial_counts), ('final', final_counts)]:
            for key, value in counts.items():
                row[f'{prefix}_{key}'] = value
        for key in ['paths_added', 'pairs_considered', 'specific_total', 'path_rules', 'ap_rules', 'dep_support_rules', 'indep_support_rules', 'blocked_edge_constraints', 'set_membership_rules']:
            row[f'delta_{key}'] = row.get(f'final_{key}', 0) - row.get(f'initial_{key}', 0)
        rows.append(row)
    return rows


def aggregate_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in df.columns if c not in set(group_cols) and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    methods = {
        'ABAPC (bb)': args.bb_version,
        'ABAPC (nor)': args.nor_version,
        'ABAPC (orig)': args.orig_version,
    }
    rows: list[dict] = []
    for dataset in args.datasets:
        for method, version in methods.items():
            rows.extend(summarize_method(dataset, method, version, args.threads))
    seed_df = pd.DataFrame(rows).sort_values(['dataset', 'method', 'seed']).reset_index(drop=True)
    mean_df = aggregate_numeric(seed_df, ['dataset', 'method'])
    out_dir = RESULTS_DIR / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    seed_df.to_csv(out_dir / f'{prefix}_seedwise.csv', index=False)
    mean_df.to_csv(out_dir / f'{prefix}_method_means.csv', index=False)
    print('\n=== Reground Effect Means ===')
    cols = [
        'dataset', 'method', 'removed_fact_count', 'fully_released_pair_count', 'reground_skipped_count',
        'delta_paths_added', 'delta_path_rules', 'delta_ap_rules', 'delta_dep_support_rules',
        'delta_indep_support_rules', 'delta_blocked_edge_constraints', 'delta_pairs_considered'
    ]
    print(mean_df[cols].to_string(index=False))
    print(f'\nWrote analysis files to {out_dir}')


if __name__ == '__main__':
    main()
