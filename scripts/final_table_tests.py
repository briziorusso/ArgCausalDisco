"""Generate the canonical statistical-test artifacts used by final paper tables.

This script is intentionally separate from ``final_paper_tables.py`` so the
test layer can be audited in one step. It does not recompute graph metrics from
raw predictions; it consumes the same saved summary tables used by the final
table renderer and writes the tests that the renderer imports/regenerates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import final_paper_tables as fpt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write canonical final-table Welch/BH test artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tag", default="g2a01")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--out-dir", default=None, help="Defaults to <tables-dir>/final")
    return parser.parse_args()


def write_outputs(
    *,
    final_dir: Path,
    tag: str,
    all_tests: pd.DataFrame,
    abapcllm_main_tests: pd.DataFrame,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)

    abapcllm_main_tests.to_csv(final_dir / f"table_08_dag_main_tests_{tag}.csv", index=False)
    fpt.write_table(
        final_dir / "table_08_dag_main_tests.md",
        "Table 8. DAG tests for ABAPC-LLM against every Table 1 baseline",
        (
            "Canonical tests for Table 1. Display values match Table 1 exactly; `Subject test sd/n` and "
            "`Comparator test sd/n` show the standard deviations and sample sizes actually used in Welch tests."
        ),
        fpt.display_all_final_tests(abapcllm_main_tests),
    )

    all_tests.to_csv(final_dir / f"table_12_all_final_tests_{tag}.csv", index=False)
    fpt.write_table(
        final_dir / f"table_12_all_final_tests_{tag}.md",
        "Table 12. Complete DAG final table tests",
        (
            "Rows labelled `bolding_best_vs_comparator` compare the numeric best with each comparator and drive tied "
            "bolding. Rows labelled `marker_top_vs_next_nonbold` compare each bolded top method/variant with the next "
            "non-bold entry in the ranking and supply the printed significance markers."
        ),
        fpt.display_all_final_tests(all_tests),
    )

    marker_tests = all_tests[all_tests["test_role"].eq("marker_top_vs_next_nonbold")].copy()
    marker_tests.to_csv(final_dir / f"table_09_printed_indicator_tests_{tag}.csv", index=False)
    fpt.write_table(
        final_dir / f"table_09_printed_indicator_tests_{tag}.md",
        "Table 9. Printed significance indicator tests",
        (
            "These are the exact tests used to print markers in the DAG final tables. Each bolded top method/variant "
            "is compared against the next non-bold entry in that metric/group ranking, with BH correction applied "
            "within that marker-test family."
        ),
        fpt.display_compact_final_tests(marker_tests),
    )

    tied_tests = all_tests[
        all_tests["test_role"].eq("bolding_best_vs_comparator") & all_tests["comparator_tied"].astype(bool)
    ].copy()
    tied_tests.to_csv(final_dir / f"table_11_tied_bolding_tests_{tag}.csv", index=False)
    fpt.write_table(
        final_dir / f"table_11_tied_bolding_tests_{tag}.md",
        "Table 11. Tests behind additional bolded ties",
        "These are the non-significant best-vs-comparator tests that justify bolding methods other than the numeric best.",
        fpt.display_compact_final_tests(tied_tests, tied=True),
    )


def build_tests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    dag_summary = fpt.load_structural_summary(tables_dir, f"synthetic_structural_summary_{args.tag}")
    alpha05_summary = fpt.load_optional_alpha05_summary(tables_dir)
    gpt_dag_summary = fpt.build_gpt_ablation_dag_summary_from_base(results_dir, args.tag, dag_summary)
    dag_main_with_gpt_summary = fpt.build_dag_main_with_gpt_summary(dag_summary, gpt_dag_summary)

    all_tests = fpt.build_dag_final_tests(
        results_dir=results_dir,
        tag=args.tag,
        dag_summary=dag_summary,
        dag_main_with_gpt_summary=dag_main_with_gpt_summary,
        gpt_dag_summary=gpt_dag_summary,
        alpha05_summary=alpha05_summary,
    )
    abapcllm_main_tests = fpt.build_causenet_main_abapcllm_tests(dag_summary, tag=args.tag)
    return all_tests, abapcllm_main_tests


def main() -> None:
    args = parse_args()
    final_dir = Path(args.out_dir) if args.out_dir else Path(args.tables_dir) / "final"
    all_tests, abapcllm_main_tests = build_tests(args)
    write_outputs(
        final_dir=final_dir,
        tag=args.tag,
        all_tests=all_tests,
        abapcllm_main_tests=abapcllm_main_tests,
    )
    print(f"Wrote canonical test artifacts to {final_dir}")
    print(f"  ABAPC-LLM Table 1 tests: {len(abapcllm_main_tests)} rows")
    print(f"  Printed-marker/tied-bolding tests: {len(all_tests)} rows")


if __name__ == "__main__":
    main()
