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

from scripts.runners.run_bnlearn_abapc_nor_matched10 import (
    MATCHED_DIR,
    DEFAULT_PYTHON,
    build_command,
    canonical_seed_list,
)

DEFAULT_NAMES = ["asia", "sachs", "survey"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABAPC (nor) on asia/sachs/survey using matched gsq facts.")
    parser.add_argument('--version', default='abapc_nor_problem3_matched10_gsq')
    parser.add_argument('--names', nargs='+', default=DEFAULT_NAMES)
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test-alpha', type=float, default=0.05)
    parser.add_argument('--test-name', default='gsq')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--S-weight', dest='s_weight', action='store_true')
    parser.add_argument('--no-S-weight', dest='s_weight', action='store_false')
    parser.set_defaults(s_weight=False)
    parser.add_argument('--pre-grounding', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--disable-reground', type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument('--print-only', action='store_true')
    parser.add_argument('--python-bin', default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    args = parser.parse_args()

    seeds = canonical_seed_list()[:args.n_runs]
    command = build_command(
        version=args.version, python_bin=args.python_bin, names=args.names, sample_size=args.sample_size,
        n_runs=args.n_runs, device=args.device, test_alpha=args.test_alpha, test_name=args.test_name,
        threads=args.threads, s_weight=args.s_weight, pre_grounding=args.pre_grounding,
        disable_reground=args.disable_reground,
    )

    note = {
        'version': args.version, 'names': args.names, 'selected_seeds': seeds, 'command': command,
        'note': 'Three-dataset ABAPC (nor) comparison runner for asia/sachs/survey on matched gsq facts.'
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
