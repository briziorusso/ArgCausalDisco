#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.helpers import random_stability  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results"
MATCHED_DIR = RESULTS_DIR / "matched10"
MATCHED_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PYTHON = Path("/vol/bitbucket/fr920/miniconda3/envs/aba-env/bin/python")


def canonical_seed_list(total_runs: int = 50) -> list[int]:
    random_stability(2024)
    return np.random.randint(0, 10000, (total_runs,)).tolist()


def build_command(
    *,
    version: str,
    python_bin: str,
    sample_size: int,
    n_runs: int,
    test_alpha: float,
    test_name: str,
    threads: int,
    s_weight: bool,
    pre_grounding: bool,
    disable_reground: bool,
) -> list[str]:
    return [
        python_bin,
        str(REPO_ROOT / "experiments.py"),
        "--source", "bnlearn",
        "--models", "abapc",
        "--names", "child",
        "--version", version,
        "--sample_size", str(sample_size),
        "--n_runs", str(n_runs),
        "--resume",
        "--test_alpha", str(test_alpha),
        "--test_name", test_name,
        "--abapc_solver", "baseline",
        "--S_weight", str(s_weight).lower(),
        "--pre_grounding", str(pre_grounding).lower(),
        "--disable_reground", str(disable_reground).lower(),
        "--threads", str(threads),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABAPC (orig) on child using the baseline CausalABA solver on the same "
            "matched gsq facts as the other matched-10 comparisons."
        )
    )
    parser.add_argument("--version", default="child_abapc_orig_matched10_gsq")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--S-weight", dest="s_weight", action="store_true", help="Enable conditioning-set weighting in fact strengths.")
    parser.add_argument("--no-S-weight", dest="s_weight", action="store_false", help="Disable conditioning-set weighting in fact strengths.")
    parser.set_defaults(s_weight=False)
    parser.add_argument("--pre-grounding", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--disable-reground", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--print-only", action="store_true", help="Print the command and selected seeds without executing.")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter used to launch experiments.py.",
    )
    args = parser.parse_args()

    seeds = canonical_seed_list()[: args.n_runs]
    command = build_command(
        version=args.version,
        python_bin=args.python_bin,
        sample_size=args.sample_size,
        n_runs=args.n_runs,
        test_alpha=args.test_alpha,
        test_name=args.test_name,
        threads=args.threads,
        s_weight=args.s_weight,
        pre_grounding=args.pre_grounding,
        disable_reground=args.disable_reground,
    )

    launch_note = {
        "version": args.version,
        "selected_seeds": seeds,
        "selection_rule": f"first {args.n_runs} seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "command": command,
        "cwd": str(REPO_ROOT),
        "python": args.python_bin,
        "sample_size": args.sample_size,
        "n_runs": args.n_runs,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
        "threads": args.threads,
        "abapc_solver": "baseline",
        "s_weight": args.s_weight,
        "pre_grounding": args.pre_grounding,
        "disable_reground": args.disable_reground,
        "note": "Baseline/orig comparison on child: same current fact generation as BB, but baseline CausalABA without the incomplete-regrounding shortcuts unless explicitly enabled.",
    }
    note_path = MATCHED_DIR / f"{args.version}_launch.json"
    with open(note_path, "w") as handle:
        json.dump(launch_note, handle, indent=2)
        handle.write("\n")

    print("Matched 10 seeds:", seeds)
    print("Launch note:", note_path)
    print("Command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.print_only:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
