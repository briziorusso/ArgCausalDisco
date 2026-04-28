import re
from pathlib import Path

import numpy as np
import pandas as pd


base = Path(".")
out_dir = base / "results" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)

summary_columns = ["dataset", "model"] + [
    f"{metric}_{stat}"
    for metric in ["elapsed", "nnz", "fdr", "tpr", "fpr", "precision", "recall", "F1", "shd", "SID"]
    for stat in ["mean", "std"]
]
methods = ["ABAPC-LLM", "ABAPC-LLM-gpt5mini", "ABAPC", "MPC-LLM", "MPC-LLM-gpt5mini", "MPC"]


def load_npy(path: Path) -> pd.DataFrame:
    return pd.DataFrame(np.load(path, allow_pickle=True), columns=summary_columns)


def n_nodes(dataset: str) -> int:
    return int(re.search(r"dag_(\d+)_nodes_", dataset).group(1))


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def aggregate_abapc(csv_path: Path, impl: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return (
        df[df["impl"].eq(impl)]
        .groupby("dataset", as_index=False)
        .agg({"dag_F1": "mean", "dag_shd": "mean"})
        .rename(columns={"dag_F1": f"{label} F1", "dag_shd": f"{label} SHD"})
    )


def combine_preferred(label: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    value_cols = [f"{label} F1", f"{label} SHD"]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=value_cols, how="all")
    return combined.drop_duplicates(subset=["dataset"], keep="first").reset_index(drop=True)


def load_empty_prior_names(path: Path) -> set[str]:
    priors = pd.read_json(path)
    return {
        row.filename
        for row in priors.itertuples(index=False)
        if not ((row.priors or {}).get("forbidden") or (row.priors or {}).get("required"))
    }


def load_mpc(path: Path, model: str, label: str) -> pd.DataFrame:
    return (
        load_npy(path)
        .query("model == @model")
        [["dataset", "F1_mean", "shd_mean"]]
        .rename(columns={"F1_mean": f"{label} F1", "shd_mean": f"{label} SHD"})
    )


def table_from_frame(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.copy()
    valid["n_nodes"] = valid["dataset"].map(n_nodes)
    rows = []
    groups = [("Overall", valid)] + [
        (f"|V|={n}", valid[valid["n_nodes"].eq(n)]) for n in sorted(valid["n_nodes"].unique())
    ]

    for label, group in groups:
        row = {"Group": label, "Datasets": len(group)}
        for method in methods:
            row[f"{method} F1"] = fmt(group[f"{method} F1"].mean(), group[f"{method} F1"].std(ddof=1))
        for method in methods:
            row[f"{method} SHD"] = fmt(group[f"{method} SHD"].mean(), group[f"{method} SHD"].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, stem: str) -> None:
    md = table.to_markdown(index=False).replace("| |V|=", "| \\|V\\|=")
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")
    print(f"\n{stem}")
    print(md)


def merge_all(frames: list[pd.DataFrame], how: str = "inner") -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="dataset", how=how)
    return merged


def finite_metric_mask(df: pd.DataFrame) -> pd.Series:
    metric_cols = [c for c in df.columns if c != "dataset"]
    return df[metric_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def with_abapc_fallback(
    llm: pd.DataFrame,
    abapc: pd.DataFrame,
    label: str,
    empty_prior_names: set[str],
    all_datasets: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    f1_col = f"{label} F1"
    shd_col = f"{label} SHD"
    df = pd.DataFrame({"dataset": all_datasets}).merge(llm, on="dataset", how="left")
    df = df.merge(abapc, on="dataset", how="left")

    llm_missing_or_invalid = df[[f1_col, shd_col]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    empty_prior = df["dataset"].isin(empty_prior_names)
    fallback_mask = llm_missing_or_invalid | empty_prior

    df.loc[fallback_mask, f1_col] = df.loc[fallback_mask, "ABAPC F1"]
    df.loc[fallback_mask, shd_col] = df.loc[fallback_mask, "ABAPC SHD"]

    stats = {
        "empty_prior": int(empty_prior.sum()),
        "missing_or_invalid": int(llm_missing_or_invalid.sum()),
        "filled": int(fallback_mask.sum()),
    }
    return df[["dataset", f1_col, shd_col]], stats


all_datasets = sorted(path.stem for path in (base / "synthetic").glob("*.bifxml"))
gemini_csv = base / "results/ABAPC-LLM/merged_synthetic-desc.csv"
gpt_csv = base / "results/ABAPC-LLM/merged_synthetic-desc-gpt5mini.csv"
abapc_fallback_csv = base / "results/ABAPC-LLM/plain_abapc_fallback_dag_5_nodes_5_edges_semantics_ER.csv"

abapc = combine_preferred(
    "ABAPC",
    [
        aggregate_abapc(gemini_csv, "org", "ABAPC"),
        aggregate_abapc(gpt_csv, "org", "ABAPC"),
        aggregate_abapc(abapc_fallback_csv, "org", "ABAPC"),
    ],
)
abapc_llm = aggregate_abapc(gemini_csv, "new", "ABAPC-LLM")
abapc_llm_gpt = aggregate_abapc(gpt_csv, "new", "ABAPC-LLM-gpt5mini")
mpc = load_mpc(base / "results/stored_results_causenet_base.npy", "MPC", "MPC")
mpc_llm = load_mpc(base / "results/stored_results_causenet_mpc_llm_desc.npy", "MPC-LLM", "MPC-LLM")
mpc_llm_gpt = load_mpc(
    base / "results/stored_results_causenet_mpc_llm_gpt5mini_desc.npy",
    "MPC-LLM",
    "MPC-LLM-gpt5mini",
)

strict = merge_all([abapc_llm, abapc_llm_gpt, abapc, mpc_llm, mpc_llm_gpt, mpc], how="inner")
strict = strict[finite_metric_mask(strict)].copy()
write_table(table_from_frame(strict), "gpt5mini_llm_comparison_desc_summary")
print(f"Strict matched datasets: {len(strict)}")

gemini_empty = load_empty_prior_names(base / "results/llm_constraints/synthetic-desc-consensus.json")
gpt_empty = load_empty_prior_names(base / "results/llm_constraints/synthetic-desc-gpt5mini-consensus.json")
abapc_llm_all, gemini_fallback_stats = with_abapc_fallback(
    abapc_llm,
    abapc,
    "ABAPC-LLM",
    gemini_empty,
    all_datasets,
)
abapc_llm_gpt_all, gpt_fallback_stats = with_abapc_fallback(
    abapc_llm_gpt,
    abapc,
    "ABAPC-LLM-gpt5mini",
    gpt_empty,
    all_datasets,
)

all54 = merge_all(
    [
        pd.DataFrame({"dataset": all_datasets}),
        abapc_llm_all,
        abapc_llm_gpt_all,
        abapc,
        mpc_llm,
        mpc_llm_gpt,
        mpc,
    ],
    how="left",
)
if len(all54) != 54:
    raise SystemExit(f"Expected 54 datasets, got {len(all54)}")
missing_after_fallback = all54.loc[~finite_metric_mask(all54), "dataset"].tolist()
if missing_after_fallback:
    raise SystemExit(f"All-54 fallback table still has missing metrics for: {missing_after_fallback}")

write_table(table_from_frame(all54), "gpt5mini_llm_comparison_desc_summary_all54_fallback")
print(f"All-54 fallback datasets: {len(all54)}")
print(f"Gemini ABAPC-LLM fallback stats: {gemini_fallback_stats}")
print(f"GPT-5-mini ABAPC-LLM fallback stats: {gpt_fallback_stats}")
print(f"MPC-LLM rows already include plain-MPC fallback for empty-prior datasets in experiments.py.")
