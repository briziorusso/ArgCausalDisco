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
    version: str,
    python_bin: str,
    sample_size: int,
    n_runs: int,
    test_alpha: float,
    test_name: str,
    resume: bool,
) -> list[str]:
    command = [
        python_bin,
        str(REPO_ROOT / "experiments.py"),
        "--source", "bnlearn",
        "--models", "abapc",
        "--names", "child",
        "--version", version,
        "--sample_size", str(sample_size),
        "--n_runs", str(n_runs),
        "--test_alpha", str(test_alpha),
        "--test_name", test_name,
        "--threads", "24",
        "--satcheck_threads", "24",
        "--satcheck_timeout", "600",
        "--satcheck_probe_limit", "12",
        "--satcheck_frontier_crawl", "true",
        "--satcheck_promoted_retry", "true",
        "--satcheck_promoted_retry_timeout_scale", "2.0",
        "--satcheck_promoted_retry_max_retries", "1",
        "--satcheck_retry_frontier_sat", "false",
        "--satcheck_portfolio", "true",
        "--satcheck_portfolio_size", "3",
        "--satcheck_portfolio_timeout_scale", "1.0",
        "--satcheck_portfolio_min_timeout", "180",
        "--satcheck_portfolio_throttle", "true",
        "--satcheck_portfolio_min_size", "2",
        "--satcheck_plateau_stop", "true",
        "--satcheck_plateau_stop_width_ratio", "0.08",
        "--satcheck_plateau_stop_min_calls", "40",
        "--satcheck_plateau_stop_unknown_ratio", "0.75",
        "--adaptive_satcheck_threads", "true",
        "--satcheck_min_threads", "8",
        "--satcheck_increase_step", "2",
        "--final-solve-timeout", "0",
        "--final-solve-opt-mode", "opt",
        "--final-solve-n-models", "1",
    ]
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exact child line-search on the matched 10 canonical seeds.")
    parser.add_argument("--version", default="child_abapc_bb_matched10_gsq_searchv3")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing progress and summaries.")
    parser.add_argument("--fresh", dest="resume", action="store_false", help="Ignore existing progress and rerun this version from scratch.")
    parser.set_defaults(resume=True)
    parser.add_argument("--print-only", action="store_true", help="Print the command and selected seeds without executing.")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter used to launch experiments.py.",
    )
    args = parser.parse_args()

    seeds = canonical_seed_list()[: args.n_runs]
    command = build_command(
        args.version,
        args.python_bin,
        args.sample_size,
        args.n_runs,
        args.test_alpha,
        args.test_name,
        args.resume,
    )

    launch_note = {
        "version": args.version,
        "selected_seeds": seeds,
        "selection_rule": "first 10 seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "command": command,
        "cwd": str(REPO_ROOT),
        "python": args.python_bin,
        "sample_size": args.sample_size,
        "n_runs": args.n_runs,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
        "resume": args.resume,
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
