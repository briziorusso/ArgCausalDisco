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

def load_npy(path):
    return pd.DataFrame(np.load(path, allow_pickle=True), columns=summary_columns)

def n_nodes(dataset):
    return int(re.search(r"dag_(\d+)_nodes_", dataset).group(1))

def fmt(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"

gemini_csv = pd.read_csv(base / "results/ABAPC-LLM/merged_synthetic-desc.csv")
gpt_csv = pd.read_csv(base / "results/ABAPC-LLM/merged_synthetic-desc-gpt5mini.csv")

abapc = (
    gemini_csv[gemini_csv["impl"].eq("org")]
    .groupby("dataset", as_index=False)
    .agg({"dag_F1": "mean", "dag_shd": "mean"})
    .rename(columns={"dag_F1": "ABAPC F1", "dag_shd": "ABAPC SHD"})
)

abapc_llm = (
    gemini_csv[gemini_csv["impl"].eq("new")]
    .groupby("dataset", as_index=False)
    .agg({"dag_F1": "mean", "dag_shd": "mean"})
    .rename(columns={"dag_F1": "ABAPC-LLM F1", "dag_shd": "ABAPC-LLM SHD"})
)

abapc_llm_gpt = (
    gpt_csv[gpt_csv["impl"].eq("new")]
    .groupby("dataset", as_index=False)
    .agg({"dag_F1": "mean", "dag_shd": "mean"})
    .rename(columns={"dag_F1": "ABAPC-LLM-gpt5mini F1", "dag_shd": "ABAPC-LLM-gpt5mini SHD"})
)

mpc = (
    load_npy(base / "results/stored_results_causenet_base.npy")
    .query("model == 'MPC'")
    [["dataset", "F1_mean", "shd_mean"]]
    .rename(columns={"F1_mean": "MPC F1", "shd_mean": "MPC SHD"})
)

mpc_llm = (
    load_npy(base / "results/stored_results_causenet_mpc_llm_desc.npy")
    .query("model == 'MPC-LLM'")
    [["dataset", "F1_mean", "shd_mean"]]
    .rename(columns={"F1_mean": "MPC-LLM F1", "shd_mean": "MPC-LLM SHD"})
)

mpc_llm_gpt = (
    load_npy(base / "results/stored_results_causenet_mpc_llm_gpt5mini_desc.npy")
    .query("model == 'MPC-LLM'")
    [["dataset", "F1_mean", "shd_mean"]]
    .rename(columns={"F1_mean": "MPC-LLM-gpt5mini F1", "shd_mean": "MPC-LLM-gpt5mini SHD"})
)

merged = abapc_llm.merge(abapc_llm_gpt, on="dataset")
merged = merged.merge(abapc, on="dataset").merge(mpc_llm, on="dataset").merge(mpc_llm_gpt, on="dataset").merge(mpc, on="dataset")

metric_cols = [c for c in merged.columns if c != "dataset"]
valid = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=metric_cols).copy()
valid["n_nodes"] = valid["dataset"].map(n_nodes)

methods = ["ABAPC-LLM", "ABAPC-LLM-gpt5mini", "ABAPC", "MPC-LLM", "MPC-LLM-gpt5mini", "MPC"]
rows = []

for label, group in [("Overall", valid)] + [(f"|V|={n}", valid[valid["n_nodes"].eq(n)]) for n in sorted(valid["n_nodes"].unique())]:
    row = {"Group": label, "Datasets": len(group)}
    for method in methods:
        row[f"{method} F1"] = fmt(group[f"{method} F1"].mean(), group[f"{method} F1"].std(ddof=1))
    for method in methods:
        row[f"{method} SHD"] = fmt(group[f"{method} SHD"].mean(), group[f"{method} SHD"].std(ddof=1))
    rows.append(row)

table = pd.DataFrame(rows)
md = table.to_markdown(index=False).replace("| |V|=", "| \\|V\\|=")

(table).to_csv(out_dir / "gpt5mini_llm_comparison_desc_summary.csv", index=False)
(out_dir / "gpt5mini_llm_comparison_desc_summary.md").write_text(md + "\n", encoding="utf-8")

print(md)
print(f"\nValid matched datasets: {len(valid)}")
