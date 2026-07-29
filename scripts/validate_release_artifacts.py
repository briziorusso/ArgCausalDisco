#!/usr/bin/env python3
"""Read-only integrity and scope checks for the curated paper artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "results" / "paper_aaai2027"
LOCAL_PATH_RE = re.compile(r"(?:/vol/bitbucket/|/homes/|[A-Za-z]:[\\\\/]Users[\\\\/])")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksum_file(path: Path, base: Path) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed checksum line {path}:{line_number}") from exc
        target = (base / relative).resolve()
        if target in expected:
            raise RuntimeError(f"Duplicate checksum target: {relative}")
        expected[target] = digest
    return expected


def _verify_checksums(path: Path, base: Path, actual_files: Iterable[Path]) -> None:
    expected = _read_checksum_file(path, base)
    actual = {item.resolve() for item in actual_files}
    if set(expected) != actual:
        missing = sorted(str(item) for item in set(expected) - actual)
        extra = sorted(str(item) for item in actual - set(expected))
        raise RuntimeError(f"Checksum inventory mismatch; missing={missing[:5]}, extra={extra[:5]}")
    mismatched = [item for item, digest in expected.items() if _sha256(item) != digest]
    if mismatched:
        raise RuntimeError(f"Checksum mismatch: {[str(item) for item in mismatched[:10]]}")


def _check_portability(path: Path) -> None:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive.files or archive["metadata"].shape != ():
                return
            text = str(archive["metadata"].item())
    else:
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            return
    if LOCAL_PATH_RE.search(text):
        raise RuntimeError(f"Machine-local path remains in release artefact: {path}")


def _validate_completion(manifest: dict, table_manifest: dict) -> None:
    expected = manifest["completion_exceptions"]
    graph = table_manifest["graph_completion"]
    checks = {
        "asia/OptABA-PC": graph["asia/OptABA-PC"]["missing_saved_seed_ids"],
        "survey/MPC": graph["survey/MPC"]["missing_graph_evaluation_seed_ids"],
        "er8/MPC": graph["er8/MPC"]["missing_graph_evaluation_seed_ids"],
        "sf5/FGS": graph["sf5/FGS"]["missing_saved_seed_ids"],
    }
    expected_values = {
        key: value.get("missing_saved_seeds", value.get("missing_graph_evaluation_seeds", []))
        for key, value in expected.items()
    }
    if checks != expected_values:
        raise RuntimeError(f"Completion exceptions changed: actual={checks}, expected={expected_values}")


def validate() -> dict[str, int]:
    manifest_path = RELEASE_ROOT / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Release manifest is missing; run scripts/prepare_release_artifacts.py")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("release") != "paper_aaai2027":
        raise RuntimeError("Unexpected release identifier")
    if protocol.get("sample_size") != 5000 or protocol.get("shared_ci", {}).get("alpha") != 0.01:
        raise RuntimeError("Frozen protocol does not match N=5000 and alpha=0.01")
    seeds = protocol.get("seeds", {})
    if seeds != {"start": 2026, "end": 2075, "count": 50}:
        raise RuntimeError(f"Unexpected seed protocol: {seeds}")

    frozen_files = sorted(item for item in (RELEASE_ROOT / "frozen").rglob("*") if item.is_file())
    _verify_checksums(RELEASE_ROOT / "FROZEN_SHA256SUMS", RELEASE_ROOT / "frozen", frozen_files)

    all_files = sorted(
        item
        for item in RELEASE_ROOT.rglob("*")
        if item.is_file()
        and item.name != "SHA256SUMS"
        and "recomputed" not in item.relative_to(RELEASE_ROOT).parts
    )
    _verify_checksums(RELEASE_ROOT / "SHA256SUMS", RELEASE_ROOT, all_files)
    for path in all_files:
        _check_portability(path)

    formal = manifest["formal_checker"]
    formal_path = REPO_ROOT / formal["path"]
    if _sha256(formal_path) != formal["sha256"]:
        raise RuntimeError("Formal-property checker differs from the release manifest")

    table_manifest_path = RELEASE_ROOT / "derived" / "tables" / "table_build_manifest.json"
    table_manifest = json.loads(table_manifest_path.read_text())
    if table_manifest.get("sample_size") != 5000 or table_manifest.get("ci_alpha") != 0.01:
        raise RuntimeError("Derived table manifest protocol is inconsistent")
    _validate_completion(manifest, table_manifest)

    forbidden = {"shapleypc", "smoke_aspcr", "10rep"}
    release_names = {path.name.lower() for path in all_files}
    leaked = sorted(token for token in forbidden if any(token in name for name in release_names))
    if leaked:
        raise RuntimeError(f"Exploratory result family leaked into release: {leaked}")

    return {"frozen_files": len(frozen_files), "release_files": len(all_files)}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    counts = validate()
    print(
        f"Release validation PASS: {counts['frozen_files']} frozen files; "
        f"{counts['release_files']} checksummed release files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
