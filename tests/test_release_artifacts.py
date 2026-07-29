from pathlib import Path

import pandas as pd

from scripts import build_final_experiment_tables as tables
from scripts import prepare_release_artifacts as release


def test_portable_text_rewrites_repository_and_external_paths() -> None:
    text = (
        '"/vol/bitbucket/user/ArgCausalDisco-worktrees/topic/results/progress/x.csv" '
        '"/vol/bitbucket/user/aspcr-hyttinen2014uai/ASP/encoding.pl"'
    )
    portable = release._portable_text(text)
    assert "/vol/bitbucket" not in portable
    assert "results/paper_aaai2027/frozen/results/progress/x.csv" in portable
    assert "${ASPCR_ROOT}/ASP/encoding.pl" in portable


def test_aspcr_release_scope_excludes_unreported_synthetic_runs() -> None:
    assert release._aspcr_include(Path("results/aspcr/results/trace_cancer_2026.csv"))
    assert not release._aspcr_include(Path("results/aspcr/results/trace_er5_2026.csv"))
    assert not release._aspcr_include(Path("results/estimated/er5/run_2026.npz"))


def test_final_record_sort_is_dataset_method_seed_stable() -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "survey", "method": "OptABA-PC", "seed": 2027},
            {"dataset": "cancer", "method": "OptABA-PC", "seed": 2027},
            {"dataset": "cancer", "method": "ABA-PC", "seed": 2028},
            {"dataset": "cancer", "method": "ABA-PC", "seed": 2026},
        ]
    )
    ordered = tables._sort_records(frame)
    assert list(ordered.itertuples(index=False, name=None)) == [
        ("cancer", "ABA-PC", 2026),
        ("cancer", "ABA-PC", 2028),
        ("cancer", "OptABA-PC", 2027),
        ("survey", "OptABA-PC", 2027),
    ]
