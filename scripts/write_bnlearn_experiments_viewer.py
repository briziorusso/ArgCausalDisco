#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_bnlearn_matched10_compare import FIGS_DIR, _write_bnlearn_experiments_viewer  # noqa: E402


MATCHED10_FIGURES = [
    "Fig.bn_matched10_dag_SHD_F1.html",
    "Fig.bn_matched10_dag_SID.html",
    "Fig.bn_matched10_dag_prec_rec.html",
    "Fig.bn_matched10_dag_size.html",
    "Fig.bn_matched10_dag_skeleton_arrowhead_F1.html",
    "Fig.bn_matched10_dag_skeleton_prec_rec.html",
    "Fig.bn_matched10_dag_arrowhead_prec_rec.html",
    "Fig.2_SID_cpdag_matched10.html",
    "Fig.bn_matched10_cpdag_SHD_F1.html",
    "Fig.bn_matched10_cpdag_prec_rec.html",
    "Fig.bn_matched10_cpdag_size.html",
    "Fig.bn_matched10_cpdag_skeleton_arrowhead_F1.html",
    "Fig.bn_matched10_cpdag_skeleton_prec_rec.html",
    "Fig.bn_matched10_cpdag_arrowhead_prec_rec.html",
    "Fig.3_runtime_matched10.html",
]


def main() -> None:
    matched10_paths = [FIGS_DIR / name for name in MATCHED10_FIGURES]
    viewer_path = _write_bnlearn_experiments_viewer(matched10_paths, output_suffix="")
    print(f"Wrote {viewer_path}")


if __name__ == "__main__":
    main()
