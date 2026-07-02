"""Compatibility entry point for the Gemini/GPT-5-mini structural table.

The main UAI table-generation logic now lives in ``scripts/paper_tables.py`` so the
same source paths and formatting are used by the paper tables and notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_tables import SourceLog, build_gpt_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Gemini/GPT-5-mini structural comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", default="g2a01")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    source_log = SourceLog(rows=[])
    build_gpt_tables(args, source_log)
    source_log.write(Path(args.out_dir) / f"gemini_gpt5_structural_sources_{args.tag}.csv")


if __name__ == "__main__":
    main()
