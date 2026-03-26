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

from scripts.runners.run_child_matched10 import (  # noqa: E402
    DEFAULT_PYTHON,
    MATCHED_DIR,
    build_command as build_bb_command,
    canonical_seed_list,
)


def build_command(
    *,
    version: str,
    python_bin: str,
    test_alpha: float,
    test_name: str,
    disable_reground: bool,
) -> list[str]:
    command = build_bb_command(version, python_bin, test_alpha, test_name)
    command.extend(
        [
            "--abapc_solver",
            "incremental",
            "--pre_grounding",
            "false",
            "--disable_reground",
            str(disable_reground).lower(),
        ]
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABAPC on child with the incremental Bayes-ball encoding but "
            "freeze the initial block_edge skeleton to emulate the old "
            "no-reground approximation."
        )
    )
    parser.add_argument("--version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3")
    parser.add_argument("--test-alpha", type=float, default=0.05)
    parser.add_argument("--test-name", default="gsq")
    parser.add_argument(
        "--disable-reground",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Freeze block_edge assignments after the initial activation step.",
    )
    parser.add_argument("--print-only", action="store_true", help="Print the command and selected seeds without executing.")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter used to launch experiments.py.",
    )
    args = parser.parse_args()

    seeds = canonical_seed_list()[:10]
    command = build_command(
        version=args.version,
        python_bin=args.python_bin,
        test_alpha=args.test_alpha,
        test_name=args.test_name,
        disable_reground=args.disable_reground,
    )

    launch_note = {
        "version": args.version,
        "selected_seeds": seeds,
        "selection_rule": "first 10 seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "command": command,
        "cwd": str(REPO_ROOT),
        "python": args.python_bin,
        "test_alpha": args.test_alpha,
        "test_name": args.test_name,
        "abapc_solver": "incremental",
        "pre_grounding": False,
        "disable_reground": args.disable_reground,
        "note": (
            "Isolation run: incremental Bayes-ball encoding with the child "
            "matched-10 bb search settings, but block_edge assignments frozen "
            "after the initial activation step to emulate the old nor "
            "no-reground approximation."
        ),
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
