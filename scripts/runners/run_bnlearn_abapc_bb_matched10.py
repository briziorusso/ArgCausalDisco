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
DEFAULT_NAMES = ["cancer", "earthquake", "survey", "asia", "sachs"]
DEFAULT_PLATEAU_WIDTH_RATIO = 0.08
DEFAULT_PLATEAU_UNKNOWN_RATIO = 0.75


def canonical_seed_list(total_runs: int = 50) -> list[int]:
    random_stability(2024)
    return np.random.randint(0, 10000, (total_runs,)).tolist()


def build_command(
    *,
    version: str,
    python_bin: str,
    names: list[str],
    sample_size: int,
    n_runs: int,
    test_alpha: float,
    test_name: str,
    threads: int,
    satcheck_threads: int,
    satcheck_timeout: float,
    satcheck_probe_limit: int,
    portfolio_size: int,
    portfolio_timeout_scale: float,
    portfolio_min_timeout: float,
    satcheck_min_threads: int,
    satcheck_increase_step: int,
    satcheck_plateau_stop_width_ratio: float,
    satcheck_plateau_stop_unknown_ratio: float,
    final_solve_timeout: float,
    final_solve_opt_mode: str,
    final_solve_n_models: int,
    resume: bool,
) -> list[str]:
    command = [
        python_bin,
        str(REPO_ROOT / "experiments.py"),
        "--source", "bnlearn",
        "--models", "abapc",
        "--names", *names,
        "--version", version,
        "--sample_size", str(sample_size),
        "--n_runs", str(n_runs),
        "--test_alpha", str(test_alpha),
        "--test_name", test_name,
        "--threads", str(threads),
        "--satcheck_threads", str(satcheck_threads),
        "--satcheck_timeout", str(satcheck_timeout),
        "--satcheck_probe_limit", str(satcheck_probe_limit),
        "--satcheck_frontier_crawl", "true",
        "--satcheck_promoted_retry", "true",
        "--satcheck_promoted_retry_timeout_scale", "2.0",
        "--satcheck_promoted_retry_max_retries", "1",
        "--satcheck_retry_frontier_sat", "false",
        "--satcheck_portfolio", "true",
        "--satcheck_portfolio_size", str(portfolio_size),
        "--satcheck_portfolio_timeout_scale", str(portfolio_timeout_scale),
        "--satcheck_portfolio_min_timeout", str(portfolio_min_timeout),
        "--satcheck_portfolio_throttle", "true",
        "--satcheck_portfolio_min_size", "2",
        "--satcheck_plateau_stop", "true",
        "--satcheck_plateau_stop_width_ratio", str(satcheck_plateau_stop_width_ratio),
        "--satcheck_plateau_stop_min_calls", "40",
        "--satcheck_plateau_stop_unknown_ratio", str(satcheck_plateau_stop_unknown_ratio),
        "--adaptive_satcheck_threads", "true",
        "--satcheck_min_threads", str(satcheck_min_threads),
        "--satcheck_increase_step", str(satcheck_increase_step),
        "--final-solve-timeout", str(final_solve_timeout),
        "--final-solve-opt-mode", final_solve_opt_mode,
        "--final-solve-n-models", str(final_solve_n_models),
    ]
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ABAPC (bb) on the bnlearn non-child datasets using the matched canonical seeds."
    )
    parser.add_argument("--version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3")
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--satcheck-threads", type=int, default=24)
    parser.add_argument("--satcheck-timeout", type=float, default=600.0)
    parser.add_argument("--satcheck-probe-limit", type=int, default=12)
    parser.add_argument("--portfolio-size", type=int, default=3)
    parser.add_argument("--portfolio-timeout-scale", type=float, default=1.0)
    parser.add_argument("--portfolio-min-timeout", type=float, default=180.0)
    parser.add_argument("--satcheck-min-threads", type=int, default=8)
    parser.add_argument("--satcheck-increase-step", type=int, default=2)
    parser.add_argument("--satcheck-plateau-stop-width-ratio", type=float, default=DEFAULT_PLATEAU_WIDTH_RATIO)
    parser.add_argument("--satcheck-plateau-stop-unknown-ratio", type=float, default=DEFAULT_PLATEAU_UNKNOWN_RATIO)
    parser.add_argument("--final-solve-timeout", type=float, default=0.0)
    parser.add_argument("--final-solve-opt-mode", default="opt", choices=["ignore", "opt", "optN"])
    parser.add_argument("--final-solve-n-models", type=int, default=1)
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

    seeds = canonical_seed_list()[:args.n_runs]
    command = build_command(
        version=args.version,
        python_bin=args.python_bin,
        names=args.names,
        sample_size=args.sample_size,
        n_runs=args.n_runs,
        test_alpha=args.test_alpha,
        test_name=args.test_name,
        threads=args.threads,
        satcheck_threads=args.satcheck_threads,
        satcheck_timeout=args.satcheck_timeout,
        satcheck_probe_limit=args.satcheck_probe_limit,
        portfolio_size=args.portfolio_size,
        portfolio_timeout_scale=args.portfolio_timeout_scale,
        portfolio_min_timeout=args.portfolio_min_timeout,
        satcheck_min_threads=args.satcheck_min_threads,
        satcheck_increase_step=args.satcheck_increase_step,
        satcheck_plateau_stop_width_ratio=args.satcheck_plateau_stop_width_ratio,
        satcheck_plateau_stop_unknown_ratio=args.satcheck_plateau_stop_unknown_ratio,
        final_solve_timeout=args.final_solve_timeout,
        final_solve_opt_mode=args.final_solve_opt_mode,
        final_solve_n_models=args.final_solve_n_models,
        resume=args.resume,
    )

    launch_note = {
        "version": args.version,
        "names": args.names,
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
        "satcheck_threads": args.satcheck_threads,
        "satcheck_timeout": args.satcheck_timeout,
        "satcheck_probe_limit": args.satcheck_probe_limit,
        "satcheck_plateau_stop_width_ratio": args.satcheck_plateau_stop_width_ratio,
        "satcheck_plateau_stop_unknown_ratio": args.satcheck_plateau_stop_unknown_ratio,
        "final_solve_timeout": args.final_solve_timeout,
        "final_solve_opt_mode": args.final_solve_opt_mode,
        "final_solve_n_models": args.final_solve_n_models,
        "resume": args.resume,
    }
    note_path = MATCHED_DIR / f"{args.version}_launch.json"
    with open(note_path, "w") as handle:
        json.dump(launch_note, handle, indent=2)
        handle.write("\n")

    print("Model: ['abapc']")
    print(f"Datasets: {args.names}")
    print("Matched seeds:", seeds)
    print("Launch note:", note_path)
    print("Command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.print_only:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
