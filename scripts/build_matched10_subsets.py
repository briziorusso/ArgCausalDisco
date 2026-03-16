#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_support import (  # noqa: E402
    CPDAG_METRIC_MAP,
    CPDAG_PROGRESS_COLUMNS,
    DAG_METRIC_MAP,
    DAG_PROGRESS_COLUMNS,
    save_summary_tables,
    summarise_results,
)
from utils.helpers import random_stability  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results"
PROGRESS_DIR = RESULTS_DIR / "progress"
MATCHED_DIR = RESULTS_DIR / "matched10"
MATCHED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED_COUNT = 10
DEFAULT_SUFFIX = "_matched10"
DEFAULT_SOURCE_VERSIONS = [
    "bnlearn_50rep_abapc",
    "bnlearn_50rep_mpc",
]
SUMMARY_ONLY_VERSIONS = [
    "bnlearn_big_rnd_mpc",
    "bnlearn_big_fgs_nt",
    "bnlearn_child_base",
    "bnlearn_child_base2",
]


@dataclass
class VersionBuild:
    source_version: str
    target_version: str
    dag_files: int
    cpdag_files: int
    dag_rows: int
    cpdag_rows: int
    datasets: list[str]
    models: list[str]


def canonical_seed_list(total_runs: int = 50) -> list[int]:
    random_stability(2024)
    return np.random.randint(0, 10000, (total_runs,)).tolist()


def _load_progress(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    for column in missing:
        frame[column] = np.nan
    return frame[columns].copy()


def _filter_frame(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    filtered = frame[frame["seed"].isin(seeds)].copy()
    filtered["seed_order"] = filtered["seed"].map({seed: idx for idx, seed in enumerate(seeds)})
    filtered = filtered.sort_values(["dataset", "model", "seed_order", "run_idx"]).drop(columns=["seed_order"])
    return filtered


def _reset_target_progress_dir(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale in target_dir.glob("*.csv"):
        stale.unlink()


def build_subset(source_version: str, target_version: str, seeds: list[int]) -> VersionBuild:
    source_dir = PROGRESS_DIR / source_version
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing progress directory for {source_version}: {source_dir}")

    target_dir = PROGRESS_DIR / target_version
    _reset_target_progress_dir(target_dir)

    dag_frames: list[pd.DataFrame] = []
    cpdag_frames: list[pd.DataFrame] = []
    dag_file_count = 0
    cpdag_file_count = 0

    for source_path in sorted(source_dir.glob("*_dag.csv")):
        dag_frame = _filter_frame(_load_progress(source_path, DAG_PROGRESS_COLUMNS), seeds)
        if dag_frame.empty:
            continue
        dag_frame.to_csv(target_dir / source_path.name, index=False)
        dag_frames.append(dag_frame)
        dag_file_count += 1

    for source_path in sorted(source_dir.glob("*_cpdag.csv")):
        cpdag_frame = _filter_frame(_load_progress(source_path, CPDAG_PROGRESS_COLUMNS), seeds)
        if cpdag_frame.empty:
            continue
        cpdag_frame.to_csv(target_dir / source_path.name, index=False)
        cpdag_frames.append(cpdag_frame)
        cpdag_file_count += 1

    dag_progress = pd.concat(dag_frames, ignore_index=True) if dag_frames else pd.DataFrame(columns=DAG_PROGRESS_COLUMNS)
    cpdag_progress = pd.concat(cpdag_frames, ignore_index=True) if cpdag_frames else pd.DataFrame(columns=CPDAG_PROGRESS_COLUMNS)

    dag_summary = summarise_results(dag_progress, DAG_METRIC_MAP)
    cpdag_summary = summarise_results(cpdag_progress, CPDAG_METRIC_MAP)
    save_summary_tables(RESULTS_DIR, target_version, dag_summary, cpdag_summary)

    datasets = sorted(set(dag_progress["dataset"].dropna().astype(str).tolist()))
    models = sorted(set(dag_progress["model"].dropna().astype(str).tolist()))
    return VersionBuild(
        source_version=source_version,
        target_version=target_version,
        dag_files=dag_file_count,
        cpdag_files=cpdag_file_count,
        dag_rows=int(len(dag_progress)),
        cpdag_rows=int(len(cpdag_progress)),
        datasets=datasets,
        models=models,
    )


def write_manifest(*, seeds: list[int], builds: list[VersionBuild]) -> tuple[Path, Path]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    payload = {
        "created_at": timestamp,
        "selection_rule": "first 10 seeds from random_stability(2024) + np.random.randint(0, 10000, (50,))",
        "seed_count": len(seeds),
        "seeds": seeds,
        "builds": [asdict(build) for build in builds],
        "summary_only_versions_not_subsetted": SUMMARY_ONLY_VERSIONS,
        "recommended_exact_child_version": "child_sat24_t600_matched10",
        "recommended_exact_child_command": (
            "python scripts/run_child_matched10.py --version child_sat24_t600_matched10"
        ),
    }
    json_path = MATCHED_DIR / "child_matched10_manifest.json"
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Child matched-10 subset",
        "",
        f"Built at: `{timestamp}`",
        "",
        "Selection rule:",
        "`random_stability(2024); np.random.randint(0, 10000, (50,))[:10]`",
        "",
        f"Selected seeds: `{seeds}`",
        "",
        "Derived versions:",
    ]
    for build in builds:
        lines.append(
            f"- `{build.target_version}` from `{build.source_version}` "
            f"({build.dag_files} DAG files, {build.cpdag_files} CPDAG files, "
            f"{build.dag_rows} DAG rows, {build.cpdag_rows} CPDAG rows)"
        )
    lines.extend(
        [
            "",
            "Versions not subsetted because only aggregated summaries are available:",
        ]
    )
    for version in SUMMARY_ONLY_VERSIONS:
        lines.append(f"- `{version}`")
    lines.extend(
        [
            "",
            "Exact child rerun:",
            "`python scripts/run_child_matched10.py --version child_sat24_t600_matched10`",
            "",
            "This workflow only creates duplicated derived outputs. It does not edit the source progress or summary files.",
        ]
    )
    markdown_path = MATCHED_DIR / "child_matched10_manifest.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build duplicated matched-seed subsets from existing progress CSVs.")
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX, help="Suffix appended to each duplicated version.")
    parser.add_argument(
        "--source-versions",
        nargs="+",
        default=DEFAULT_SOURCE_VERSIONS,
        help="Source result versions with seed-level progress files to subset.",
    )
    args = parser.parse_args()

    seeds = canonical_seed_list()[:args.seed_count]
    builds = [
        build_subset(source_version=source_version, target_version=f"{source_version}{args.suffix}", seeds=seeds)
        for source_version in args.source_versions
    ]
    json_path, markdown_path = write_manifest(seeds=seeds, builds=builds)

    print("Matched 10 seeds:", seeds)
    for build in builds:
        print(
            f"Built {build.target_version} from {build.source_version}: "
            f"dag_rows={build.dag_rows} cpdag_rows={build.cpdag_rows}"
        )
    print(f"Manifest: {json_path}")
    print(f"Note: {markdown_path}")


if __name__ == "__main__":
    main()
