#!/usr/bin/env python3
"""Build the source-only anonymised archive submitted with the paper."""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "OptABA-PC-minimal"
MAX_BYTES = 50 * 1024 * 1024

ROOT_FILES = (
    "verify_formal_properties.py",
    "requirements.txt",
    "requirements-repro.txt",
    "requirements-fgs.txt",
    "environment.yml",
    "causalaba_mus.py",
    "causalaba_increm.py",
    "causalaba_binsearch.py",
    "causalaba.py",
    "abapc.py",
    "__init__.py",
    "REPRODUCIBILITY.md",
    "README.md",
    "LICENSE",
)

TREE_FILES = (
    "utils/prior_knowledge.py",
    "utils/helpers.py",
    "utils/graph_utils.py",
    "utils/graph_eval.py",
    "utils/data_utils.py",
    "utils/__init__.py",
    "tests/test_contestability.py",
    "tests/test_collect_matched_tables.py",
    "tests/test_baseline_contestability.py",
    "tests/test_baseline_compatible_extensions.py",
    "tests/test_aspcr_pipeline.py",
    "scripts/wc_opt_strategy_sweep.py",
    "scripts/verify_formal_properties.py",
    "scripts/validate_matched_aspcr_results.py",
    "scripts/run_matched_baseline_experiments.py",
    "scripts/run_contestability_experiments.py",
    "scripts/run_aspcr_contestability.py",
    "scripts/collect_matched_baseline_tables.py",
    "scripts/build_final_experiment_tables.py",
    "scripts/build_contestability_table.py",
    "scripts/build_baseline_contestability_tables.py",
    "scripts/build_encoding_ablation_table.py",
    "scripts/baseline_compatible_extensions.py",
    "scripts/audit_mpc_test_enforcement.py",
    "encodings/causalaba.lp",
    "datasets/bayesian/small/survey.bif/survey.bif",
    "datasets/bayesian/small/earthquake.bif/earthquake.bif",
    "datasets/bayesian/small/cancer.bif/cancer.bif",
    "datasets/bayesian/small/asia.bif/asia.bif",
    "configs/paper_aaai2027.json",
    "cd_algorithms/spc.py",
    "cd_algorithms/run_aspcr_csvdata.R",
    "cd_algorithms/models.py",
    "cd_algorithms/PC.py",
)

MINIMAL_README = """# Minimal experiment-code archive

This archive contains only the source, fixed network definitions, environment
specifications, and focused tests needed for the experiments reported in the
paper. It is derived from the anonymised code release.

Included experiment paths are:

- ABA-PC and OptABA-PC runs (`scripts/wc_opt_strategy_sweep.py`);
- MPC, discrete-BDeu FGS, and ASPCR-DAG runs;
- OptABA-PC, MPC, and ASPCR contestability/correspondence analyses;
- compatible-extension, encoding-ablation, statistical, and table builders;
- the DAG-restricted ASPCR R wrapper;
- the four reported bnlearn BIF networks;
- the executable formal-property checker and focused pipeline tests.

Synthetic ER and SF graphs and samples are generated from the recorded seeds,
so they require no stored dataset files. The external ASPCR implementation is
not redistributed because its package states no redistribution licence; setup
is documented in `REPRODUCIBILITY.md`.

Generated results and logs, exploratory drivers, injected-corruption and
Shapley-PC experiments, notebooks, plots, unused datasets, packaging utilities,
caches, and version-control metadata are excluded.

Run the source checks from the archive root:

```bash
python -m pytest -q tests
python verify_formal_properties.py
```
"""

IDENTITY_REPLACEMENTS = (
    ("Fabrizio Russo", "XXXX-2 XXXX-4"),
    ("fabrizio@imperial.ac.uk", "XXXX-2@example.invalid"),
    ("Department of Computing, Imperial College London", "Anonymous Institution"),
    ("briziorusso", "anonymous-author"),
)
LOCAL_PATH_RE = re.compile(r"(?:/vol/bitbucket/[^/\s\"']+|/homes/[^/\s\"']+)")
FORBIDDEN_RE = re.compile(
    r"(?:Fabrizio\s+Russo|fabrizio@imperial\.ac\.uk|briziorusso|/vol/bitbucket/|/homes/|\bfr920\b)",
    re.IGNORECASE,
)


def _anonymise(text: str) -> str:
    for source, replacement in IDENTITY_REPLACEMENTS:
        text = text.replace(source, replacement)
    text = re.sub(r"\bfr920\b", "anonymous-user", text, flags=re.IGNORECASE)
    return LOCAL_PATH_RE.sub("${WORKSPACE}", text)


def _zip_info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    if directory and not name.endswith("/"):
        name += "/"
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if directory else 0o644) << 16
    return info


def _payloads() -> dict[str, bytes]:
    files = (*ROOT_FILES, *TREE_FILES)
    missing = [relative for relative in files if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Minimal archive inputs are missing: {missing}")
    payloads: dict[str, bytes] = {}
    for relative in files:
        source = REPO_ROOT / relative
        try:
            text = source.read_text()
        except UnicodeDecodeError:
            payload = source.read_bytes()
        else:
            payload = _anonymise(text).encode("utf-8")
        payloads[relative] = payload
    payloads["MINIMAL_ARCHIVE.md"] = MINIMAL_README.encode("utf-8")
    leaked = [name for name, payload in payloads.items() if FORBIDDEN_RE.search(payload.decode("utf-8", "ignore"))]
    if leaked:
        raise RuntimeError(f"Identity or local path remains in minimal archive inputs: {leaked}")
    return payloads


def build(output: Path) -> tuple[int, int, str]:
    payloads = _payloads()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        directories = {ARCHIVE_ROOT}
        for relative in payloads:
            parts = Path(relative).parts[:-1]
            for index in range(1, len(parts) + 1):
                directories.add(f"{ARCHIVE_ROOT}/{'/'.join(parts[:index])}")
        for directory in sorted(directories):
            archive.writestr(_zip_info(directory, directory=True), b"")
        for relative, payload in sorted(payloads.items()):
            archive.writestr(_zip_info(f"{ARCHIVE_ROOT}/{relative}"), payload)
    size = output.stat().st_size
    if size > MAX_BYTES:
        output.unlink()
        raise RuntimeError(f"Archive exceeds 50 MiB: {size} bytes")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(payloads), size, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "OptABA-PC_AAAI2027_code_minimal_anonymised.zip",
    )
    args = parser.parse_args()
    count, size, digest = build(args.output)
    print(f"Wrote {args.output}: {count} files, {size} bytes, sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
