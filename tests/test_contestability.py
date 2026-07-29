from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_contestability_table import render_tex_table, summarise_contestability
from scripts.run_contestability_experiments import (
    parse_emitted_program,
    solve_program,
    stable_fact_id,
)


def test_parse_emitted_program_and_stable_fact_id(tmp_path: Path) -> None:
    program = tmp_path / "instance.lp"
    program.write_text(
        "\n".join(
            [
                "{mus(1)}.",
                "{mus(2)}.",
                "ext_dep(0,1,empty):-mus(1).",
                "ext_indep(0,2,s1):-mus(2).",
                "__optimum_mcs_objective_literal__(7, mus(1)).",
                "__optimum_mcs_objective_literal__(11, mus(2)).",
                ":~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]",
                "#show mus/1.",
                "",
            ]
        )
    )

    parsed = parse_emitted_program(program)

    assert parsed.facts == {1: "ext_dep(0,1,empty)", 2: "ext_indep(0,2,s1)"}
    assert parsed.weights == {1: 7, 2: 11}
    assert stable_fact_id("cancer", 2026, parsed.facts[1]) == "cancer:2026:ext_dep(0,1,empty)"


def test_dataset_summary_keeps_zero_challenge_er_and_excludes_unproved_margins() -> None:
    manifest = {
        "baseline_instances": [
            {"dataset": "Cancer (5)", "seed": 2026, "analyzable": True},
            {"dataset": "Cancer (5)", "seed": 2027, "analyzable": True},
            {"dataset": "ER (5)", "seed": 2026, "analyzable": True},
        ]
    }
    challenges = [
        {
            "dataset": "Cancer (5)",
            "optimality_proven": True,
            "normalized_margin": 0.0,
            "zero_margin": True,
            "timeout": False,
            "infeasible": False,
        },
        {
            "dataset": "Cancer (5)",
            "optimality_proven": True,
            "normalized_margin": 0.5,
            "zero_margin": False,
            "timeout": False,
            "infeasible": False,
        },
        {
            "dataset": "Cancer (5)",
            "optimality_proven": False,
            "normalized_margin": "",
            "zero_margin": "",
            "timeout": True,
            "infeasible": False,
        },
        {
            "dataset": "Cancer (5)",
            "optimality_proven": False,
            "normalized_margin": "",
            "zero_margin": "",
            "timeout": False,
            "infeasible": True,
        },
    ]

    summary = summarise_contestability(challenges, manifest)

    assert [row["dataset"] for row in summary] == ["Cancer (5)", "ER (5)"]
    cancer, er5 = summary
    assert cancer["analyzable_seeds"] == 2
    assert cancer["challenges"] == 4
    assert cancer["proved_challenges"] == 2
    assert cancer["zero_margin_proportion"] == pytest.approx(0.5)
    assert cancer["normalized_margin_median"] == pytest.approx(0.25)
    assert cancer["normalized_margin_q1"] == pytest.approx(0.125)
    assert cancer["normalized_margin_q3"] == pytest.approx(0.375)
    assert cancer["normalized_margin_iqr"] == pytest.approx(0.25)
    assert cancer["timeouts"] == 1
    assert cancer["infeasible_challenges"] == 1
    assert cancer["unproved_challenges"] == 2
    assert er5["analyzable_seeds"] == 1
    assert er5["challenges"] == 0
    assert er5["zero_margin_proportion"] is None

    tex = render_tex_table(summary, label="tab:test", caption="Test.")
    assert "ER (5) & 1 & 0 & 0 & -- & -- & 0 & 0" in tex


def test_forced_retention_solver_proves_zero_margin_tie(tmp_path: Path) -> None:
    pytest.importorskip("clingo")
    program = tmp_path / "tie.lp"
    program.write_text(
        "\n".join(
            [
                "{mus(1)}.",
                "{mus(2)}.",
                ":- mus(1), mus(2).",
                ":- not mus(1), not mus(2).",
                "ext_dep(0,1,empty):-mus(1).",
                "ext_indep(0,2,empty):-mus(2).",
                "__optimum_mcs_objective_literal__(5, mus(1)).",
                "__optimum_mcs_objective_literal__(5, mus(2)).",
                ":~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]",
                "#show mus/1.",
                "",
            ]
        )
    )
    parsed = parse_emitted_program(program)
    baseline = solve_program(
        parsed,
        force_index=None,
        timeout_sec=10,
        threads=1,
        opt_strategy="bb",
    )
    assert baseline.optimality_proven
    assert baseline.objective_cost == 5
    released_index = next(iter(set(parsed.facts) - set(baseline.selected_indices)))

    forced = solve_program(
        parsed,
        force_index=released_index,
        timeout_sec=10,
        threads=1,
        opt_strategy="bb",
    )

    assert forced.optimality_proven
    assert forced.objective_cost == baseline.objective_cost
    assert released_index in forced.selected_indices
