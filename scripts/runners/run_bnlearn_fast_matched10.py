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
DEFAULT_MODELS = ["random", "fgs", "nt"]
DEFAULT_NAMES = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]


def canonical_seed_list(total_runs: int = 50) -> list[int]:
    random_stability(2024)
    return np.random.randint(0, 10000, (total_runs,)).tolist()


def build_command(
    *,
    version: str,
    python_bin: str,
    models: list[str],
    names: list[str],
    sample_size: int,
    n_runs: int,
    device: int,
    test_alpha: float,
    test_name: str,
) -> list[str]:
    return [
        python_bin,
        str(REPO_ROOT / "experiments.py"),
        "--source", "bnlearn",
        "--models", *models,
        "--names", *names,
        "--version", version,
        "--sample_size", str(sample_size),
        "--n_runs", str(n_runs),
        "--resume",
        "--device", str(device),
        "--test_alpha", str(test_alpha),
        "--test_name", test_name,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fast bnlearn baselines on the matched canonical seeds."
    )
    parser.add_argument("--version", default="bnlearn_fast_matched10_gsq")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
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
        models=args.models,
        names=args.names,
        sample_size=args.sample_size,
        n_runs=args.n_runs,
        device=args.device,
        test_alpha=args.test_alpha,
        test_name=args.test_name,
    )

    launch_note = {
        "version": args.version,
        "models": args.models,
        "names": args.names,
        "selected_seeds": seeds,
        "selection_rule": f"first {args.n_runs} seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "command": command,
        "cwd": str(REPO_ROOT),
        "python": args.python_bin,
        "sample_size": args.sample_size,
        "n_runs": args.n_runs,
        "device": args.device,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
    }
    note_path = MATCHED_DIR / f"{args.version}_launch.json"
    with open(note_path, "w") as handle:
        json.dump(launch_note, handle, indent=2)
        handle.write("\n")

    print(f"Models: {args.models}")
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
