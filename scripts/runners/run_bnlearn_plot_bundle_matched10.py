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
ALL_NAMES = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]
OTHER_NAMES = ["cancer", "earthquake", "survey", "asia", "sachs"]
DEFAULT_STEPS = ["bb", "bb-nor", "nor", "baselines", "nt"]


def canonical_seed_list(total_runs: int = 50) -> list[int]:
    random_stability(2024)
    return np.random.randint(0, 10000, (total_runs,)).tolist()


def _runner_command(
    *,
    runner_python: str,
    script_name: str,
    args: list[str],
) -> list[str]:
    return [
        runner_python,
        str(REPO_ROOT / "scripts" / "runners" / script_name),
        *args,
    ]


def build_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    common_fast = [
        "--sample-size",
        str(args.sample_size),
        "--n-runs",
        str(args.n_runs),
        "--device",
        str(args.device),
        "--test-alpha",
        str(args.test_alpha),
        "--test-name",
        args.test_name,
        "--python-bin",
        args.python_bin,
    ]
    common_abapc = [
        "--sample-size",
        str(args.sample_size),
        "--n-runs",
        str(args.n_runs),
        "--test-alpha",
        str(args.test_alpha),
        "--test-name",
        args.test_name,
        "--python-bin",
        args.python_bin,
    ]
    common_child = [
        "--sample-size",
        str(args.sample_size),
        "--n-runs",
        str(args.n_runs),
        "--test-alpha",
        str(args.test_alpha),
        "--test-name",
        args.test_name,
        "--python-bin",
        args.python_bin,
    ]

    commands: list[dict[str, object]] = []
    step_commands: dict[str, list[dict[str, object]]] = {
        "baselines": [
            {
                "label": "Baselines (random, fgs, mpc)",
                "version": args.baselines_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_bnlearn_fast_matched10.py",
                    args=[
                        "--version",
                        args.baselines_version,
                        "--models",
                        "random",
                        "fgs",
                        "mpc",
                        "--names",
                        *ALL_NAMES,
                        *common_fast,
                    ],
                ),
            }
        ],
        "nt": [
            {
                "label": "NOTEARS-MLP",
                "version": args.nt_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_bnlearn_fast_matched10.py",
                    args=[
                        "--version",
                        args.nt_version,
                        "--models",
                        "nt",
                        "--names",
                        *ALL_NAMES,
                        *common_fast,
                    ],
                ),
            }
        ],
        "nor": [
            {
                "label": "ABAPC (nor) others",
                "version": args.nor_others_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_bnlearn_abapc_nor_matched10.py",
                    args=[
                        "--version",
                        args.nor_others_version,
                        "--names",
                        *OTHER_NAMES,
                        *common_abapc,
                    ],
                ),
            },
            {
                "label": "ABAPC (nor) child",
                "version": args.nor_child_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_child_abapc_nor_matched10.py",
                    args=[
                        "--version",
                        args.nor_child_version,
                        *common_child,
                    ],
                ),
            },
        ],
        "bb": [
            {
                "label": "ABAPC (bb) others",
                "version": args.bb_others_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_bnlearn_abapc_bb_matched10.py",
                    args=[
                        "--version",
                        args.bb_others_version,
                        "--names",
                        *OTHER_NAMES,
                        *common_abapc,
                    ],
                ),
            },
            {
                "label": "ABAPC (bb) child",
                "version": args.bb_child_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_child_matched10.py",
                    args=[
                        "--version",
                        args.bb_child_version,
                        *common_child,
                    ],
                ),
            },
        ],
        "bb-nor": [
            {
                "label": "ABAPC (bb-nor) others",
                "version": args.bb_nor_others_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_bnlearn_abapc_bb_norapprox_matched10.py",
                    args=[
                        "--version",
                        args.bb_nor_others_version,
                        "--names",
                        *OTHER_NAMES,
                        *common_abapc,
                    ],
                ),
            },
            {
                "label": "ABAPC (bb-nor) child",
                "version": args.bb_nor_child_version,
                "command": _runner_command(
                    runner_python=args.runner_python,
                    script_name="run_child_abapc_bb_norapprox_matched10.py",
                    args=[
                        "--version",
                        args.bb_nor_child_version,
                        *common_child,
                    ],
                ),
            },
        ],
    }

    for step in args.steps:
        commands.extend(step_commands[step])

    if args.print_only:
        for spec in commands:
            spec["command"] = [*spec["command"], "--print-only"]

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full matched-10 BNLearn plot bundle used by the current "
            "figures, excluding ABAPC (orig): random, fgs, nt, mpc, nor, bb, "
            "and bb-nor across all six datasets."
        )
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=DEFAULT_STEPS,
        default=DEFAULT_STEPS,
        help="Subset of method groups to execute.",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
    parser.add_argument("--baselines-version", default="bnlearn_baselines_matched10_gsq")
    parser.add_argument("--nt-version", default="bnlearn_nt_matched10_gsq")
    parser.add_argument("--nor-others-version", default="bnlearn_abapc_nor_matched10_others_gsq")
    parser.add_argument("--nor-child-version", default="child_abapc_nor_matched10_gsq")
    parser.add_argument("--bb-others-version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3")
    parser.add_argument("--bb-child-version", default="child_abapc_bb_matched10_gsq_searchv3")
    parser.add_argument("--bb-nor-others-version", default="bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3")
    parser.add_argument("--bb-nor-child-version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3")
    parser.add_argument("--print-only", action="store_true", help="Print the delegated commands without executing them.")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter passed through to the delegated runners for experiments.py.",
    )
    parser.add_argument(
        "--runner-python",
        default=None,
        help="Python interpreter used to execute the delegated runner scripts. Defaults to --python-bin.",
    )
    args = parser.parse_args()
    if args.runner_python is None:
        args.runner_python = args.python_bin

    seeds = canonical_seed_list()[: args.n_runs]
    commands = build_commands(args)

    launch_note = {
        "steps": args.steps,
        "selected_seeds": seeds,
        "selection_rule": f"first {args.n_runs} seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "sample_size": args.sample_size,
        "n_runs": args.n_runs,
        "device": args.device,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
        "python_bin": args.python_bin,
        "runner_python": args.runner_python,
        "commands": commands,
        "versions": {
            "baselines": args.baselines_version,
            "nt": args.nt_version,
            "nor_others": args.nor_others_version,
            "nor_child": args.nor_child_version,
            "bb_others": args.bb_others_version,
            "bb_child": args.bb_child_version,
            "bb_nor_others": args.bb_nor_others_version,
            "bb_nor_child": args.bb_nor_child_version,
        },
    }
    note_path = MATCHED_DIR / "bnlearn_plot_bundle_matched10_launch.json"
    with open(note_path, "w") as handle:
        json.dump(launch_note, handle, indent=2)
        handle.write("\n")

    print("Matched seeds:", seeds)
    print("Launch note:", note_path)
    print()
    for idx, spec in enumerate(commands, start=1):
        print(f"[{idx}/{len(commands)}] {spec['label']} -> {spec['version']}")
        print(" ".join(shlex.quote(part) for part in spec["command"]))
        print()

    if args.print_only:
        return

    for idx, spec in enumerate(commands, start=1):
        print(f"Running [{idx}/{len(commands)}] {spec['label']} ({spec['version']})")
        subprocess.run(spec["command"], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
