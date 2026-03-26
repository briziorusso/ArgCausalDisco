#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runners.run_bnlearn_abapc_bb_matched10 import (
    MATCHED_DIR,
    DEFAULT_PYTHON,
    DEFAULT_PLATEAU_UNKNOWN_RATIO,
    DEFAULT_PLATEAU_WIDTH_RATIO,
    build_command,
    canonical_seed_list,
)

DEFAULT_NAMES = ["asia", "sachs", "survey"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABAPC (bb) on asia/sachs/survey using matched gsq facts.")
    parser.add_argument('--version', default='abapc_bb_problem3_matched10_gsq_searchv3')
    parser.add_argument('--names', nargs='+', default=DEFAULT_NAMES)
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--test-alpha', type=float, default=0.05)
    parser.add_argument('--test-name', default='gsq')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--satcheck-threads', type=int, default=24)
    parser.add_argument('--satcheck-timeout', type=float, default=600.0)
    parser.add_argument('--satcheck-probe-limit', type=int, default=12)
    parser.add_argument('--portfolio-size', type=int, default=3)
    parser.add_argument('--portfolio-timeout-scale', type=float, default=1.0)
    parser.add_argument('--portfolio-min-timeout', type=float, default=180.0)
    parser.add_argument('--satcheck-min-threads', type=int, default=8)
    parser.add_argument('--satcheck-increase-step', type=int, default=2)
    parser.add_argument('--satcheck-plateau-stop-width-ratio', type=float, default=DEFAULT_PLATEAU_WIDTH_RATIO)
    parser.add_argument('--satcheck-plateau-stop-unknown-ratio', type=float, default=DEFAULT_PLATEAU_UNKNOWN_RATIO)
    parser.add_argument('--final-solve-timeout', type=float, default=0.0)
    parser.add_argument('--final-solve-opt-mode', default='opt', choices=['ignore', 'opt', 'optN'])
    parser.add_argument('--final-solve-n-models', type=int, default=1)
    parser.add_argument('--print-only', action='store_true')
    parser.add_argument('--python-bin', default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    args = parser.parse_args()

    seeds = canonical_seed_list()[:args.n_runs]
    command = build_command(
        version=args.version, python_bin=args.python_bin, names=args.names, sample_size=args.sample_size,
        n_runs=args.n_runs, test_alpha=args.test_alpha, test_name=args.test_name, threads=args.threads,
        satcheck_threads=args.satcheck_threads, satcheck_timeout=args.satcheck_timeout,
        satcheck_probe_limit=args.satcheck_probe_limit, portfolio_size=args.portfolio_size,
        portfolio_timeout_scale=args.portfolio_timeout_scale, portfolio_min_timeout=args.portfolio_min_timeout,
        satcheck_min_threads=args.satcheck_min_threads, satcheck_increase_step=args.satcheck_increase_step,
        satcheck_plateau_stop_width_ratio=args.satcheck_plateau_stop_width_ratio,
        satcheck_plateau_stop_unknown_ratio=args.satcheck_plateau_stop_unknown_ratio,
        final_solve_timeout=args.final_solve_timeout, final_solve_opt_mode=args.final_solve_opt_mode,
        final_solve_n_models=args.final_solve_n_models,
    )

    note = {
        'version': args.version, 'names': args.names, 'selected_seeds': seeds, 'command': command,
        'note': 'Three-dataset ABAPC (bb) comparison runner for asia/sachs/survey on matched gsq facts.'
    }
    note_path = MATCHED_DIR / f'{args.version}_launch.json'
    note_path.write_text(json.dumps(note, indent=2) + '\n')
    print("Model: [\'abapc\']")
    print(f'Datasets: {args.names}')
    print('Matched seeds:', seeds)
    print('Launch note:', note_path)
    print('Command:')
    print(' '.join(shlex.quote(part) for part in command))
    if not args.print_only:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
