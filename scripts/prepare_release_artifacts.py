#!/usr/bin/env python3
"""Build the curated, portable paper artefact directory from final runs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "results" / "paper_aaai2027"
CONFIG_PATH = REPO_ROOT / "configs" / "paper_aaai2027.json"

BASELINE_VERSIONS = (
    "paper_bnlearn_alpha001_nowrong_noweight_50rep_mpc_fgs",
    "paper_er_sf_sparse_alpha001_noweight_50rep_mpc_fgs",
    "paper_er_sf_alpha001_nowrong_noweight_50rep_mpc",
    "paper_fgs_bdeu_alpha001_n5000_50rep",
)
ASPCR_VERSIONS = {
    "paper_aspcr_dag_alpha001_n5000_50rep": ("cancer", "earthquake", "survey"),
    "paper_aspcr_dag_er_sf_e1_alpha001_n5000_50rep": ("er5", "sf5"),
}
REPORTED_ASPCR_DATASETS = tuple(
    dataset for datasets in ASPCR_VERSIONS.values() for dataset in datasets
)

STATIC_RELEASE_FILES = {"README.md", "CONTESTABILITY_SCOPE.md"}
GENERATED_RELEASE_NAMES = {
    "frozen",
    "derived",
    "manifest.json",
    "FROZEN_SHA256SUMS",
    "SHA256SUMS",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_text(text: str) -> str:
    text = text.replace(str(REPO_ROOT.resolve()), ".")
    workspace_repo = r"/vol/bitbucket/[^/\s\"']+/ArgCausalDisco(?:-worktrees/[^/\s\"']+)?"
    text = re.sub(workspace_repo, ".", text)
    text = re.sub(
        r"/vol/bitbucket/[^/\s\"']+/(?:aspcr-hyttinen2014uai|ASPhyttinen)",
        "${ASPCR_ROOT}",
        text,
    )
    text = re.sub(r"/vol/bitbucket/[^/\s\"']+/R/R-[^/\s\"']+/bin/Rscript", "${RSCRIPT}", text)
    text = re.sub(r"/vol/bitbucket/[^/\s\"']+/R/R-[^/\s\"']+", "${R_HOME}", text)
    text = re.sub(
        r"/vol/bitbucket/[^/\s\"']+/miniconda3/envs/[^/\s\"']+/bin/python",
        "python",
        text,
    )
    text = re.sub(
        r"/vol/bitbucket/[^/\s\"']+/miniconda3/envs/[^/\s\"']+/bin/clingo",
        "clingo",
        text,
    )
    text = re.sub(
        r"/vol/bitbucket/[^/\s\"']+/MUSandMCS---CausalABA",
        "${PAPER_REPOSITORY}",
        text,
    )
    text = re.sub(r"/vol/bitbucket/[^/\s\"']+", "${WORKSPACE}", text)
    frozen_prefix = "results/paper_aaai2027/frozen/results/"
    text = text.replace("./results/", "results/")
    return re.sub(r"(?<![A-Za-z0-9_./])results/", frozen_prefix, text)


def _rewrite_npz(source: Path, destination: Path) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "metadata" in arrays:
        metadata = arrays["metadata"]
        if metadata.shape == ():
            value = str(metadata.item())
            arrays["metadata"] = np.asarray(_portable_text(value))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _copy_portable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".npz":
        _rewrite_npz(source, destination)
        return
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        shutil.copy2(source, destination)
        return
    destination.write_text(_portable_text(text))


def _default_include(path: Path) -> bool:
    return path.name not in {".DS_Store", ".RData", ".Rhistory", ".Rapp.history"} and path.suffix != ".log"


def _aspcr_include_for(datasets: tuple[str, ...]) -> Callable[[Path], bool]:
    """Include generic ASPCR files and only the requested dataset artefacts."""

    known = {"cancer", "earthquake", "survey", "asia", "er5", "er8", "sf5", "sf8"}
    allowed = set(datasets)

    def include(path: Path) -> bool:
        if not _default_include(path):
            return False
        name = f"_{path.name.lower()}_"
        mentioned = {
            dataset
            for dataset in known
            if f"_{dataset}_" in name or any(part.lower() == dataset for part in path.parts)
        }
        return not mentioned or bool(mentioned & allowed)

    return include


def _aspcr_include(path: Path) -> bool:
    """Backward-compatible all-reported-dataset filter used by focused tests."""

    return _aspcr_include_for(REPORTED_ASPCR_DATASETS)(path)


def _copy_entry(
    source: Path,
    destination: Path,
    *,
    include: Callable[[Path], bool] = _default_include,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required final artefact is missing: {source}")
    if source.is_file():
        if include(source):
            _copy_portable(source, destination)
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and include(path):
            _copy_portable(path, destination / path.relative_to(source))


def _copy_result_path(relative: str, *, include: Callable[[Path], bool] = _default_include) -> None:
    source = REPO_ROOT / relative
    destination = RELEASE_ROOT / "frozen" / relative
    _copy_entry(source, destination, include=include)


def _tree_inventory(root: Path) -> tuple[list[tuple[str, str, int]], str]:
    rows: list[tuple[str, str, int]] = []
    tree = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        rows.append((relative, digest, size))
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    return rows, tree.hexdigest()


def _write_checksums(path: Path, rows: list[tuple[str, str, int]], *, base: Path) -> None:
    lines = []
    for relative, digest, _size in rows:
        absolute = REPO_ROOT / relative
        lines.append(f"{digest}  {absolute.relative_to(base).as_posix()}")
    path.write_text("\n".join(lines) + "\n")


def _release_inventory() -> list[tuple[str, str, int]]:
    rows, _tree = _tree_inventory(RELEASE_ROOT)
    return [
        row
        for row in rows
        if row[0] != (RELEASE_ROOT / "SHA256SUMS").relative_to(REPO_ROOT).as_posix()
        and "recomputed" not in (REPO_ROOT / row[0]).relative_to(RELEASE_ROOT).parts
    ]


def _refresh_checksums() -> None:
    if not (RELEASE_ROOT / "manifest.json").exists() or not (RELEASE_ROOT / "frozen").exists():
        raise RuntimeError("Generated release content is missing; run the full preparation first")
    frozen_rows, _tree = _tree_inventory(RELEASE_ROOT / "frozen")
    _write_checksums(RELEASE_ROOT / "FROZEN_SHA256SUMS", frozen_rows, base=RELEASE_ROOT / "frozen")
    _write_checksums(RELEASE_ROOT / "SHA256SUMS", _release_inventory(), base=RELEASE_ROOT)


def _clear_generated(force: bool) -> None:
    existing = [RELEASE_ROOT / name for name in GENERATED_RELEASE_NAMES if (RELEASE_ROOT / name).exists()]
    if existing and not force:
        listed = ", ".join(path.name for path in sorted(existing))
        raise RuntimeError(f"Generated release content already exists ({listed}); pass --force to refresh it")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _final_source_paths() -> list[tuple[str, Callable[[Path], bool]]]:
    paths: list[tuple[str, Callable[[Path], bool]]] = [
        ("results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked", _default_include),
        ("results/final_mcs_experiments_er_sf_sparse_alpha001_noweight_50rep", _default_include),
        ("results/recovery_optaba_runs", _default_include),
        ("results/tables/paper_current_alpha001_nowrong_noweight_50rep_preview", _default_include),
    ]
    for version in BASELINE_VERSIONS:
        paths.extend(
            [
                (f"results/metadata_{version}.json", _default_include),
                (f"results/stored_results_{version}.csv", _default_include),
                (f"results/stored_results_{version}_cpdag.csv", _default_include),
                (f"results/progress/{version}", _default_include),
                (f"results/estimated_graphs/{version}", _default_include),
            ]
        )
    for version, datasets in ASPCR_VERSIONS.items():
        include = _aspcr_include_for(datasets)
        paths.extend(
            [
                (f"results/metadata_{version}.json", _default_include),
                (f"results/stored_results_{version}.csv", include),
                (f"results/stored_results_{version}_cpdag.csv", include),
                (f"results/progress/{version}", include),
                (f"results/estimated_graphs/{version}", include),
                (f"results/aspcr_matched/{version}", include),
            ]
        )
    return paths


def build(force: bool) -> dict[str, Any]:
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    missing_static = [name for name in STATIC_RELEASE_FILES if not (RELEASE_ROOT / name).exists()]
    if missing_static:
        raise RuntimeError(f"Missing static release documentation: {missing_static}")
    _clear_generated(force)

    included: list[str] = []
    for relative, include in _final_source_paths():
        _copy_result_path(relative, include=include)
        included.append(relative)

    final_tables = REPO_ROOT / "results" / "tables" / "paper_final_matched_50rep"

    def final_table_include(path: Path) -> bool:
        relative = path.relative_to(final_tables)
        return (
            _default_include(path)
            and "contestability_er8" not in relative.parts
            and "contestability_sf8" not in relative.parts
        )

    _copy_entry(final_tables, RELEASE_ROOT / "derived" / "tables", include=final_table_include)

    frozen_rows, frozen_tree = _tree_inventory(RELEASE_ROOT / "frozen")
    config = json.loads(CONFIG_PATH.read_text())
    formal_path = REPO_ROOT / "scripts" / "verify_formal_properties.py"
    manifest: dict[str, Any] = {
        "release": "paper_aaai2027",
        "protocol": config,
        "included_source_paths": included,
        "excluded_exploratory_families": config["excluded"],
        "reported_aspcr_datasets": list(REPORTED_ASPCR_DATASETS),
        "frozen_file_count": len(frozen_rows),
        "frozen_bytes": sum(size for _path, _digest, size in frozen_rows),
        "frozen_tree_sha256": frozen_tree,
        "formal_checker": {
            "path": formal_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(formal_path),
        },
        "completion_exceptions": {
            "survey/MPC": {"missing_graph_evaluation_seeds": [2026, 2029, 2035, 2038, 2040, 2048, 2052, 2055, 2056, 2064, 2069, 2070]},
            "er8/MPC": {"missing_graph_evaluation_seeds": [2030]},
        },
    }
    (RELEASE_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_checksums(RELEASE_ROOT / "FROZEN_SHA256SUMS", frozen_rows, base=RELEASE_ROOT / "frozen")

    _write_checksums(RELEASE_ROOT / "SHA256SUMS", _release_inventory(), base=RELEASE_ROOT)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refresh only generated content below results/paper_aaai2027")
    parser.add_argument(
        "--checksums-only",
        action="store_true",
        help="Refresh inventories without copying or deleting any artefact.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checksums_only:
        _refresh_checksums()
        print("Refreshed release checksum inventories")
        return 0
    manifest = build(args.force)
    print(
        f"Prepared {manifest['frozen_file_count']} frozen files "
        f"({manifest['frozen_bytes']} bytes) in {RELEASE_ROOT.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
