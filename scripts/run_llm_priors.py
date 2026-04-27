from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Annotated, Callable

import pandas as pd
import pyagrum as gum
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from priors.llm import extract
from priors.prompt import prepare_priors
from priors.schema import Constraints


def _true_arrows(bn: gum.BayesNet) -> set[tuple[str, str]]:
    return {
        (bn.variable(source).name(), bn.variable(target).name())
        for source, target in bn.arcs()
    }


def _prior_metrics(
    constraints: Constraints,
    true_arrows: set[tuple[str, str]],
    n_nodes: int,
) -> dict[str, float | int]:
    forbidden_length = len(constraints.forbidden)
    forbidden_precision = len(constraints.forbidden - true_arrows) / max(
        forbidden_length, 1
    )
    forbidden_recall = len(constraints.forbidden - true_arrows) / (
        n_nodes * (n_nodes - 1) - len(true_arrows)
    )
    forbidden_f1 = (
        2
        * forbidden_precision
        * forbidden_recall
        / max(forbidden_precision + forbidden_recall, 1e-6)
    )

    required_length = len(constraints.required)
    required_precision = len(constraints.required & true_arrows) / max(
        required_length, 1
    )
    required_recall = len(constraints.required & true_arrows) / len(true_arrows)
    required_f1 = (
        2
        * required_precision
        * required_recall
        / max(required_precision + required_recall, 1e-6)
    )

    return {
        "forbidden_length": forbidden_length,
        "forbidden_Precision": forbidden_precision,
        "forbidden_Recall": forbidden_recall,
        "forbidden_F1": forbidden_f1,
        "required_length": required_length,
        "required_Precision": required_precision,
        "required_Recall": required_recall,
        "required_F1": required_f1,
    }


async def generate_priors(
    bn: gum.BayesNet,
    variable_descriptions: dict[str, str] | None,
    prior_model: str,
    parse_model: str,
) -> dict:
    priors_prompt = prepare_priors(bn, descriptions=variable_descriptions)
    priors_raw = (
        (await extract(priors_prompt, None, model=prior_model)).choices[0].message.content
    )

    valid_var_pattern = r"^(?:" + r"|".join(re.escape(var) for var in bn.names()) + r")$"
    VarType = Annotated[str, Field(pattern=valid_var_pattern)]

    class VarConstraints(BaseModel):
        forbidden: set[tuple[VarType, VarType]] = set()
        required: set[tuple[VarType, VarType]] = set()

    priors = await extract(
        prompt=priors_raw,
        model=parse_model,
        pydantic_model=VarConstraints,
    )
    constraints = Constraints(**priors.model_dump(mode="json"))
    return {
        "priors": constraints.model_dump(mode="json"),
        **_prior_metrics(constraints, _true_arrows(bn), bn.size()),
    }


async def evaluate_priors(
    bif_paths: list[Path],
    prior_model: str,
    parse_model: str,
    exclude_descriptions: bool,
    max_concurrent: int,
    row_context: dict | None = None,
    row_callback: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(max_concurrent)
    row_context = row_context or {}

    async def process_one(bif_path: Path) -> dict:
        async with semaphore:
            bn = gum.loadBN(str(bif_path))
            variable_descriptions = {
                name: bn.variable(name).description() for name in bn.names()
            }
            if exclude_descriptions or not any(variable_descriptions.values()):
                variable_descriptions = None

            row = {
                "title": bn.propertyWithDefault("name", "no_name"),
                "filename": bif_path.stem,
                "num_nodes": bn.size(),
                "num_edges": len(bn.arcs()),
                "variable_descriptions": variable_descriptions,
            }
            row.update(
                await generate_priors(
                    bn=bn,
                    variable_descriptions=variable_descriptions,
                    prior_model=prior_model,
                    parse_model=parse_model,
                )
            )
            row.update(row_context)
            return row

    rows = []
    tasks = [asyncio.create_task(process_one(path)) for path in bif_paths]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        row = await future
        rows.append(row)
        if row_callback is not None:
            row_callback(row)
    return pd.DataFrame(rows)


def write_json_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_json(tmp_path, orient="records", indent=4)
    tmp_path.replace(path)


async def run_repeated_priors(
    bif_paths: list[Path],
    repeats: int,
    prior_model: str,
    parse_model: str,
    exclude_descriptions: bool,
    max_concurrent: int,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    all_rows = []
    completed: set[tuple[int, str]] = set()

    if checkpoint_path is not None and resume and checkpoint_path.exists():
        checkpoint_df = pd.read_json(checkpoint_path)
        all_rows = checkpoint_df.to_dict("records")
        completed = {
            (int(row["run_id"]), str(row["filename"]))
            for row in all_rows
            if "run_id" in row and "filename" in row
        }
        print(f"Loaded checkpoint with {len(all_rows)} rows from {checkpoint_path}")

    def checkpoint_row(row: dict) -> None:
        all_rows.append(row)
        completed.add((int(row["run_id"]), str(row["filename"])))
        if checkpoint_path is not None:
            checkpoint_df = pd.DataFrame(all_rows).sort_values(["run_id", "filename"])
            write_json_atomic(checkpoint_df, checkpoint_path)
            print(
                f"Checkpointed {len(all_rows)} rows to {checkpoint_path}",
                flush=True,
            )

    for run_id in range(repeats):
        pending_paths = [
            path for path in bif_paths if (run_id, path.stem) not in completed
        ]
        if not pending_paths:
            print(f"Prior run {run_id + 1}/{repeats} already complete; skipping.")
            continue
        print(f"Prior run {run_id + 1}/{repeats}")
        await evaluate_priors(
            bif_paths=pending_paths,
            prior_model=prior_model,
            parse_model=parse_model,
            exclude_descriptions=exclude_descriptions,
            max_concurrent=max_concurrent,
            row_context={"run_id": run_id},
            row_callback=checkpoint_row,
        )
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).sort_values(["run_id", "filename"]).reset_index(
        drop=True
    )


def aggregate_priors(priors_path: Path, bifs_path: Path) -> pd.DataFrame:
    rows = []
    df = pd.read_json(priors_path)
    for dataset in df["filename"].unique():
        bn = gum.loadBN(str(bifs_path / f"{dataset}.bifxml"))
        priors = [Constraints(**prior_dict) for prior_dict in df.loc[df["filename"] == dataset, "priors"]]
        consensus = Constraints(
            forbidden=set.intersection(*(prior.forbidden for prior in priors))
            if priors
            else set(),
            required=set.intersection(*(prior.required for prior in priors))
            if priors
            else set(),
        )
        first_row = (
            df.loc[df["filename"] == dataset, ["filename", "num_nodes", "num_edges", "variable_descriptions"]]
            .iloc[0]
            .to_dict()
        )
        rows.append(
            {
                **first_row,
                "priors": consensus.model_dump(mode="json"),
                **_prior_metrics(consensus, _true_arrows(bn), bn.size()),
            }
        )
    return pd.DataFrame(rows)


def select_bif_paths(bifxml_dir: Path, names: list[str] | None) -> list[Path]:
    paths = sorted(bifxml_dir.glob("*.bifxml"))
    if not names:
        return paths
    wanted = {Path(name).stem for name in names}
    selected = [path for path in paths if path.stem in wanted]
    missing = sorted(wanted - {path.stem for path in selected})
    if missing:
        raise SystemExit(f"Requested BIFXML names not found in {bifxml_dir}: {missing}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repeated LLM prior constraints and an intersection consensus JSON."
    )
    parser.add_argument("--bifxml_dir", default="synthetic")
    parser.add_argument("--output_prefix", default="synthetic-desc-gpt55")
    parser.add_argument("--prior_model", default="gpt55")
    parser.add_argument("--parse_model", default="gemini-2.5-flash-lite")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--results_dir", default="results/llm_constraints")
    parser.add_argument("--exclude_descriptions", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--names", nargs="*", help="Optional BIFXML stems or filenames for smoke runs.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    bifxml_dir = Path(args.bifxml_dir)
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bif_paths = select_bif_paths(bifxml_dir, args.names)
    if not bif_paths:
        raise SystemExit(f"No .bifxml files found in {bifxml_dir}")

    raw_path = output_dir / f"{args.output_prefix}.json"
    consensus_path = output_dir / f"{args.output_prefix}-consensus.json"
    checkpoint_path = output_dir / f"{args.output_prefix}.partial.json"

    raw_df = await run_repeated_priors(
        bif_paths=bif_paths,
        repeats=args.repeats,
        prior_model=args.prior_model,
        parse_model=args.parse_model,
        exclude_descriptions=args.exclude_descriptions,
        max_concurrent=args.max_concurrent,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
    )
    raw_df.to_json(raw_path, orient="records", indent=4)

    consensus_df = aggregate_priors(raw_path, bifxml_dir)
    consensus_df.to_json(consensus_path, orient="records", indent=4)

    print(f"Wrote raw priors: {raw_path}")
    print(f"Wrote consensus priors: {consensus_path}")


if __name__ == "__main__":
    asyncio.run(main())
