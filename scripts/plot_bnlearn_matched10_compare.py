#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.plotting import (  # noqa: E402
    bar_chart_plotly,
    double_bar_chart_plotly,
    leaf_green,
    main_purple,
    plot_runtime,
    sand_yellow,
    sec_blue,
    sec_orange,
    water_green,
)
from utils.experiment_support import (  # noqa: E402
    CPDAG_BASE_COLUMNS,
    CPDAG_SUMMARY_COLUMNS,
    DAG_BASE_COLUMNS,
    DAG_SUMMARY_COLUMNS,
    load_existing_summary,
)
from utils.graph_utils import dag2cpdag  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


DAG_COLS = DAG_SUMMARY_COLUMNS
CPDAG_COLS = CPDAG_SUMMARY_COLUMNS
DAG_PROGRESS_METRICS = [column for column in DAG_BASE_COLUMNS if column not in {"dataset", "model"}]
CPDAG_PROGRESS_METRICS = [column for column in CPDAG_BASE_COLUMNS if column not in {"dataset", "model"}]
LEGACY_DAG_COLS = [
    "dataset",
    "model",
    "elapsed_mean",
    "elapsed_std",
    "nnz_mean",
    "nnz_std",
    "fdr_mean",
    "fdr_std",
    "tpr_mean",
    "tpr_std",
    "fpr_mean",
    "fpr_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "F1_mean",
    "F1_std",
    "shd_mean",
    "shd_std",
    "sid_mean",
    "sid_std",
]

DATASET_ORDER = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]
NODES_MAP = {"asia": 8, "cancer": 5, "earthquake": 5, "sachs": 11, "survey": 6, "child": 20}
EDGES_MAP = {"asia": 8, "cancer": 4, "earthquake": 4, "sachs": 17, "survey": 6, "child": 25}

METHOD_ORDER = [
    "Random",
    "rnd-dir",
    "FGS",
    "NOTEARS-MLP",
    "MPC",
    "ABAPC (orig)",
    "ABAPC (nor)",
    "ABAPC (bb)",
    "ABAPC (bb-nor)",
]
NAMES_DICT = {
    "random": "Random",
    "rnd-dir": "rnd-dir",
    "random_edge": "Random (match |E|)",
    "fgs": "FGS",
    "nt": "NOTEARS-MLP",
    "mpc": "MPC",
    "abapc_orig": "ABAPC (orig)",
    "abapc_nor": "ABAPC (nor)",
    "abapc_bb": "ABAPC (bb)",
    "abapc_bb_nor": "ABAPC (bb-nor)",
}
COLORS_DICT = {
    "random": "#7f7f7f",
    "rnd-dir": "#4d4d4d",
    "random_edge": "#b3b3b3",
    "fgs": sec_orange,
    "nt": sec_blue,
    "mpc": main_purple,
    "abapc_orig": "#8c564b",
    "abapc_nor": sand_yellow,
    "abapc_bb": water_green,
    "abapc_bb_nor": leaf_green,
}
SYMBOLS_DICT = {
    "random": "x",
    "rnd-dir": "x-thin-open",
    "random_edge": "x-open",
    "fgs": "circle-open-dot",
    "nt": "x",
    "mpc": "diamond-dot",
    "abapc_orig": "square-dot",
    "abapc_nor": "triangle-down-dot",
    "abapc_bb": "triangle-up-dot",
    "abapc_bb_nor": "star",
}

TRUE_GRAPH_LABEL = "True Graph Size"
TRUE_GRAPH_COLOR = "#7f7f7f"


def _version_has_artifacts(version: str) -> bool:
    return any(
        path.exists()
        for path in [
            RESULTS_DIR / f"stored_results_{version}.npy",
            RESULTS_DIR / f"stored_results_{version}_cpdag.npy",
            RESULTS_DIR / "progress" / version,
        ]
    )


def _resolve_version(primary: str, *fallbacks: str) -> str:
    for candidate in (primary, *fallbacks):
        if candidate and _version_has_artifacts(candidate):
            return candidate
    return primary


def _pretty_to_key(pretty_name: str) -> str:
    for key, label in NAMES_DICT.items():
        if label == pretty_name:
            return key
    raise KeyError(pretty_name)


def _load_summary(version: str, kind: str) -> pd.DataFrame:
    suffix = "_cpdag.npy" if kind == "cpdag" else ".npy"
    columns = CPDAG_COLS if kind == "cpdag" else DAG_COLS
    path = RESULTS_DIR / f"stored_results_{version}{suffix}"
    if not path.exists():
        return _load_progress_summary(version, kind)
    frame = load_existing_summary(path, columns)
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["model"] = frame["model"].astype(str)
    return frame


def _load_progress_summary(version: str, kind: str) -> pd.DataFrame:
    progress_dir = RESULTS_DIR / "progress" / version
    columns = CPDAG_COLS if kind == "cpdag" else DAG_COLS
    if not progress_dir.exists():
        return pd.DataFrame(columns=columns)

    suffix = "_cpdag.csv" if kind == "cpdag" else "_dag.csv"
    metrics = CPDAG_PROGRESS_METRICS if kind == "cpdag" else DAG_PROGRESS_METRICS
    frames: list[pd.DataFrame] = []

    for path in sorted(progress_dir.glob(f"*{suffix}")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"dataset", "model", *metrics}
        if not required.issubset(frame.columns):
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (dataset, model), group in combined.groupby(["dataset", "model"], sort=False):
        row: dict[str, object] = {"dataset": str(dataset).lower(), "model": str(model)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if kind == "dag" and "sid_mean" in row:
            row["SID_mean"] = row.pop("sid_mean")
            row["SID_std"] = row.pop("sid_std")
        if kind == "cpdag" and "sid_low_mean" in row:
            row["SID_low_mean"] = row.pop("sid_low_mean")
            row["SID_low_std"] = row.pop("sid_low_std")
            row["SID_high_mean"] = row.pop("sid_high_mean")
            row["SID_high_std"] = row.pop("sid_high_std")
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def _load_combined(kind: str, run_specs: list[dict[str, object]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in run_specs:
        if spec["kind"] != kind:
            continue
        frame = _load_summary(spec["version"], kind)
        if frame.empty:
            continue
        include = spec.get("include")
        if include:
            frame = frame[frame["model"].isin(include)].copy()
        replace_models = spec.get("replace_models") or {}
        if replace_models:
            frame["model"] = frame["model"].replace(replace_models)
        include_datasets = spec.get("include_datasets")
        if include_datasets:
            frame = frame[frame["dataset"].isin(include_datasets)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CPDAG_COLS if kind == "cpdag" else DAG_COLS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["dataset", "model"], keep="last")
    return combined


def _add_dataset_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out[out["dataset"].isin(DATASET_ORDER)].copy()
    out["base_dataset"] = out["dataset"].astype(str).str.lower()
    out["n_nodes"] = out["base_dataset"].map(NODES_MAP).astype(float)
    out["n_edges"] = out["base_dataset"].map(EDGES_MAP).astype(float)
    out["dataset"] = out["base_dataset"].map(
        lambda name: f"{name.upper()}<br> |V|={NODES_MAP[name]}, |E|={EDGES_MAP[name]}"
    )
    return out


def _sort_by_dataset_order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "base_dataset" in out.columns:
        out["_dataset_order"] = out["base_dataset"].map({name: i for i, name in enumerate(DATASET_ORDER)})
        out = out.sort_values(["_dataset_order", "model"]).drop(columns=["_dataset_order"])
    return out


def _build_version_lookup(run_specs: list[dict[str, object]], kind: str = "dag") -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for spec in run_specs:
        if spec["kind"] != kind:
            continue
        artifact_version = str(spec.get("artifact_version") or spec["version"])
        include = list(spec.get("include") or [])
        replace_models = dict(spec.get("replace_models") or {})
        include_datasets = list(spec.get("include_datasets") or DATASET_ORDER)
        for model in include:
            pretty = replace_models.get(model, model)
            for dataset in include_datasets:
                lookup[(pretty, dataset)] = artifact_version
    return lookup


def _scenario_dir_for(version: str, pretty_name: str, dataset: str) -> Path:
    key = _pretty_to_key(pretty_name)
    if key.startswith("abapc_"):
        return RESULTS_DIR / f"abapc_{version}_{dataset}"
    return RESULTS_DIR / f"{key}_{version}_{dataset}"


def _count_cpdag_edges(B: np.ndarray) -> tuple[int, int]:
    directed = 0
    undirected = 0
    n = int(B.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            left = float(B[i, j])
            right = float(B[j, i])
            is_undirected = (left == -1.0) or (right == -1.0) or ((left != 0.0) and (right != 0.0))
            is_directed = (left != 0.0) ^ (right != 0.0)
            if is_undirected:
                undirected += 1
            elif is_directed:
                directed += 1
    return directed, undirected


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _load_true_graphs(version_lookup: dict[tuple[str, str], str], datasets: list[str]) -> dict[str, np.ndarray]:
    true_graphs: dict[str, np.ndarray] = {}
    for dataset in datasets:
        for method in METHOD_ORDER:
            version = version_lookup.get((method, dataset))
            if not version:
                continue
            runs_dir = _scenario_dir_for(version, method, dataset) / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in sorted(runs_dir.glob("run_*")):
                graph_path = run_dir / "graph_true.npy"
                if graph_path.exists():
                    true_graphs[dataset] = np.load(graph_path)
                    break
            if dataset in true_graphs:
                break
    return true_graphs


def _build_size_style_dicts() -> tuple[dict[str, str], dict[str, str]]:
    names = dict(NAMES_DICT)
    colors = dict(COLORS_DICT)
    names["true_graph"] = TRUE_GRAPH_LABEL
    colors["true_graph"] = TRUE_GRAPH_COLOR
    return names, colors


def _build_dag_size_frame(
    dag_df: pd.DataFrame,
    size_methods: list[str],
    true_graphs: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = dag_df[dag_df["model"].isin([m for m in size_methods if m != TRUE_GRAPH_LABEL])].copy()
    rows = rows[
        [
            "dataset",
            "base_dataset",
            "n_nodes",
            "n_edges",
            "model",
            "nnz_mean",
            "nnz_std",
        ]
    ].copy()

    true_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        if dataset not in rows["base_dataset"].unique():
            continue
        B_true = true_graphs.get(dataset)
        dag_edges = int(np.count_nonzero(B_true)) if B_true is not None else int(EDGES_MAP[dataset])
        true_rows.append(
            {
                "dataset": f"{dataset.upper()}<br> |V|={NODES_MAP[dataset]}, |E|={EDGES_MAP[dataset]}",
                "base_dataset": dataset,
                "n_nodes": float(NODES_MAP[dataset]),
                "n_edges": float(EDGES_MAP[dataset]),
                "model": TRUE_GRAPH_LABEL,
                "nnz_mean": float(dag_edges),
                "nnz_std": 0.0,
            }
        )

    out = pd.concat([pd.DataFrame(true_rows), rows], ignore_index=True)
    return _sort_by_dataset_order(out)


def _build_cpdag_size_frame(
    cpdag_df: pd.DataFrame,
    size_methods: list[str],
    version_lookup: dict[tuple[str, str], str],
    true_graphs: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets_present = [dataset for dataset in DATASET_ORDER if dataset in cpdag_df["base_dataset"].unique()]

    for dataset in datasets_present:
        B_true = true_graphs.get(dataset)
        if B_true is not None:
            C_true = dag2cpdag(B_true.copy())
            true_directed, true_undirected = _count_cpdag_edges(C_true)
            rows.append(
                {
                    "dataset": f"{dataset.upper()}<br> |V|={NODES_MAP[dataset]}, |E|={EDGES_MAP[dataset]}",
                    "base_dataset": dataset,
                    "n_nodes": float(NODES_MAP[dataset]),
                    "n_edges": float(EDGES_MAP[dataset]),
                    "model": TRUE_GRAPH_LABEL,
                    "cpdag_directed_edges_mean": float(true_directed),
                    "cpdag_directed_edges_std": 0.0,
                    "cpdag_undirected_edges_mean": float(true_undirected),
                    "cpdag_undirected_edges_std": 0.0,
                }
            )

        for method in size_methods:
            if method == TRUE_GRAPH_LABEL:
                continue
            version = version_lookup.get((method, dataset))
            if not version:
                continue
            runs_dir = _scenario_dir_for(version, method, dataset) / "runs"
            if not runs_dir.exists():
                continue

            directed_vals: list[float] = []
            undirected_vals: list[float] = []
            for run_dir in sorted(runs_dir.glob("run_*")):
                cpdag_path = run_dir / "graph_est_cpdag_eval.npy"
                if not cpdag_path.exists():
                    continue
                C_est = np.load(cpdag_path)
                directed, undirected = _count_cpdag_edges(C_est)
                directed_vals.append(float(directed))
                undirected_vals.append(float(undirected))

            if not directed_vals:
                continue

            directed_mean, directed_std = _mean_std(directed_vals)
            undirected_mean, undirected_std = _mean_std(undirected_vals)
            rows.append(
                {
                    "dataset": f"{dataset.upper()}<br> |V|={NODES_MAP[dataset]}, |E|={EDGES_MAP[dataset]}",
                    "base_dataset": dataset,
                    "n_nodes": float(NODES_MAP[dataset]),
                    "n_edges": float(EDGES_MAP[dataset]),
                    "model": method,
                    "cpdag_directed_edges_mean": directed_mean,
                    "cpdag_directed_edges_std": directed_std,
                    "cpdag_undirected_edges_mean": undirected_mean,
                    "cpdag_undirected_edges_std": undirected_std,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "base_dataset",
                "n_nodes",
                "n_edges",
                "model",
                "cpdag_directed_edges_mean",
                "cpdag_directed_edges_std",
                "cpdag_undirected_edges_mean",
                "cpdag_undirected_edges_std",
            ]
        )
    return _sort_by_dataset_order(out)


def _add_normalised_dag(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_shd_mean"] = out["shd_mean"].astype(float) / out["n_edges"].astype(float)
    out["p_shd_std"] = out["shd_std"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_mean"] = out["SID_mean"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_std"] = out["SID_std"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_mean"] = out["p_SID_mean"].replace(0, 0.03)
    return out


def _add_normalised_cpdag(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_shd_mean"] = out["shd_mean"].astype(float) / out["n_edges"].astype(float)
    out["p_shd_std"] = out["shd_std"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_low_mean"] = out["SID_low_mean"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_high_mean"] = out["SID_high_mean"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_low_std"] = out["SID_low_std"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_high_std"] = out["SID_high_std"].astype(float) / out["n_edges"].astype(float)
    out["p_SID_low_mean"] = out["p_SID_low_mean"].replace(0, 0.03)
    out["p_SID_high_mean"] = out["p_SID_high_mean"].replace(0, 0.03)
    return out


def _metrics_available(frame: pd.DataFrame, metric_names: list[str]) -> bool:
    if frame.empty:
        return False
    for metric in metric_names:
        column = f"{metric}_mean"
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return True
    return False


def _load_orig_runtime_fallback() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for version in ["abapc_orig_problem3_matched10_gsq", "bnlearn_dag_v5_2000", "bnlearn_dag_v5"]:
        path = RESULTS_DIR / f"stored_results_{version}.npy"
        if not path.exists():
            continue
        frame: pd.DataFrame | None = None
        try:
            frame = load_existing_summary(path, DAG_COLS)
        except Exception:
            frame = None
        if frame is None or frame.empty:
            try:
                frame = pd.DataFrame(np.load(path, allow_pickle=True), columns=LEGACY_DAG_COLS)
                if "sid_mean" in frame.columns and "SID_mean" not in frame.columns:
                    frame["SID_mean"] = frame["sid_mean"]
            except Exception:
                continue
        frame["dataset"] = frame["dataset"].astype(str).str.lower()
        frame["model"] = frame["model"].astype(str)
        frame = frame[
            frame["model"].str.contains("ABAPC", case=False, na=False)
            & frame["dataset"].isin(["cancer", "earthquake", "survey", "asia", "sachs"])
        ].copy()
        if frame.empty:
            continue
        frame["model"] = "ABAPC (orig)"
        frames.append(_add_dataset_metadata(frame))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["base_dataset"], keep="first")
    return _sort_by_dataset_order(combined)


def _augment_runtime_with_orig_fallback(dag_df: pd.DataFrame) -> pd.DataFrame:
    runtime_df = dag_df.copy()
    fallback = _load_orig_runtime_fallback()
    if fallback.empty:
        return runtime_df

    existing_orig = runtime_df[runtime_df["model"] == "ABAPC (orig)"].copy()
    existing_datasets = set(existing_orig.get("base_dataset", pd.Series(dtype=str)).astype(str))
    fallback = fallback[~fallback["base_dataset"].isin(existing_datasets)].copy()
    if fallback.empty:
        return runtime_df

    for col in runtime_df.columns:
        if col not in fallback.columns:
            fallback[col] = np.nan
    for col in fallback.columns:
        if col not in runtime_df.columns:
            runtime_df[col] = np.nan
    fallback = fallback[runtime_df.columns]
    runtime_df = pd.concat([runtime_df, fallback], ignore_index=True)
    runtime_df = _sort_by_dataset_order(runtime_df)
    return runtime_df


def _find_png_browser(preferred: str | None) -> str | None:
    if preferred:
        return preferred
    for candidate in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _export_png_via_browser(
    *,
    browser_path: str,
    html_path: Path,
    png_path: Path,
    width: int,
    height: int,
    wait_ms: int,
    device_scale_factor: float,
) -> bool:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        f"--window-size={int(width)},{int(height)}",
        f"--virtual-time-budget={int(wait_ms)}",
        f"--force-device-scale-factor={float(device_scale_factor)}",
        f"--screenshot={str(png_path)}",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return png_path.exists()
    except Exception:
        return False


def _write_viewer_html(html_paths: list[Path], output_path: Path, title: str) -> None:
    def viewer_label(name: str) -> str:
        stem = Path(name).stem
        if stem.endswith("_no_nor"):
            stem = stem[: -len("_no_nor")]
        mapping = {
            "Fig.bn_dag_SHD_F1": "Original DAG - SHD & F1",
            "Fig.bn_dag_SHD_SID": "Original DAG - SHD & F1",
            "Fig.bn_dag_SID": "Original DAG - SID",
            "Fig.bn_dag_F1": "Original DAG - SID",
            "Fig.bn_dag_prec_rec": "Original DAG - Precision & Recall",
            "Fig.bn_cpdag_SID_best_worst": "Original CPDAG - SID Range",
            "Fig.bn_cpdag_SHD_F1": "Original CPDAG - SHD & F1",
            "Fig.bn_abapc_variants_sid_with_aspforaba": "ABAPC Variants - SID",
            "Fig.2_runtime": "Runtime",
            "Fig.bn_matched10_dag_SHD_F1": "DAG - SHD & F1",
            "Fig.bn_matched10_dag_SID": "DAG - SID",
            "Fig.bn_matched10_dag_prec_rec": "DAG - Precision & Recall",
            "Fig.bn_matched10_dag_size": "DAG - Size",
            "Fig.bn_matched10_dag_skeleton_arrowhead_F1": "DAG - Sk-F1 & AH-F1",
            "Fig.bn_matched10_dag_skeleton_prec_rec": "DAG - Skeleton Prec & Rec",
            "Fig.bn_matched10_dag_arrowhead_prec_rec": "DAG - Arrowhead Prec & Rec",
            "Fig.2_SID_cpdag_matched10": "CPDAG - SID Range",
            "Fig.bn_matched10_cpdag_SHD_F1": "CPDAG - SHD & F1",
            "Fig.bn_matched10_cpdag_prec_rec": "CPDAG - Precision & Recall",
            "Fig.bn_matched10_cpdag_size": "CPDAG - Size",
            "Fig.bn_matched10_cpdag_skeleton_arrowhead_F1": "CPDAG - Sk-F1 & AH-F1",
            "Fig.bn_matched10_cpdag_skeleton_prec_rec": "CPDAG - Skeleton Prec & Rec",
            "Fig.bn_matched10_cpdag_arrowhead_prec_rec": "CPDAG - Arrowhead Prec & Rec",
            "Fig.3_runtime_matched10": "Runtime",
        }
        return mapping.get(stem, name)

    figures = []
    seen: set[str] = set()
    for path in html_paths:
        if path.suffix.lower() != ".html":
            continue
        name = path.name
        if name in seen or not path.exists():
            continue
        seen.add(name)
        figures.append(name)

    if not figures:
        return

    nav_items = "\n".join(
        f'<button class="nav-btn" data-target="{name}" data-label="{viewer_label(name)}">{viewer_label(name)}</button>'
        for name in figures
    )
    first = figures[0]
    first_label = viewer_label(first)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #dbe3f0;
      --text: #152033;
      --muted: #506178;
      --accent: #2358d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: var(--bg);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #f9fbff 0%, #eef3fb 100%);
      padding: 20px 18px;
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      line-height: 1.1;
    }}
    .hint {{
      margin: 0 0 18px 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.4;
    }}
    .nav {{
      display: grid;
      gap: 10px;
    }}
    .nav-btn {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 12px;
      padding: 12px 14px;
      text-align: left;
      font-size: 15px;
      cursor: pointer;
    }}
    .nav-btn:hover {{
      border-color: var(--accent);
    }}
    .nav-btn.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(35, 88, 211, 0.12);
      background: #f3f7ff;
    }}
    .viewer {{
      padding: 10px 12px 12px 12px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      min-height: 100vh;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      padding: 0 2px 8px 2px;
    }}
    .toolbar-title {{
      display: grid;
      gap: 2px;
    }}
    .toolbar-label {{
      font-size: 18px;
      font-weight: 700;
    }}
    .toolbar-file {{
      font-size: 13px;
      color: var(--muted);
    }}
    .toolbar-link {{
      color: var(--accent);
      text-decoration: none;
      font-size: 14px;
    }}
    iframe {{
      width: 100%;
      height: calc(100vh - 56px);
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fff;
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      iframe {{
        height: 75vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>{title}</h1>
      <p class="hint">Chose the metrics to display.</p>
      <div class="nav">
        {nav_items}
      </div>
    </aside>
    <main class="viewer">
      <div class="toolbar">
        <div class="toolbar-title">
          <div class="toolbar-label" id="viewer-label">{first_label}</div>
          <div class="toolbar-file" id="viewer-file">{first}</div>
        </div>
        <a class="toolbar-link" id="open-link" href="{first}" target="_blank" rel="noopener">Open selected figure in a new tab</a>
      </div>
      <iframe id="viewer-frame" src="{first}" title="{title} viewer"></iframe>
    </main>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll('.nav-btn'));
    const frame = document.getElementById('viewer-frame');
    const labelEl = document.getElementById('viewer-label');
    const fileEl = document.getElementById('viewer-file');
    const openLink = document.getElementById('open-link');

    function setActive(target) {{
      for (const btn of buttons) {{
        const active = btn.dataset.target === target;
        btn.classList.toggle('active', active);
      }}
      const activeButton = buttons.find((btn) => btn.dataset.target === target);
      frame.src = target;
      labelEl.textContent = activeButton ? activeButton.dataset.label : target;
      fileEl.textContent = target;
      openLink.href = target;
      if (window.location.hash !== '#' + target) {{
        history.replaceState(null, '', '#' + target);
      }}
    }}

    for (const btn of buttons) {{
      btn.addEventListener('click', () => setActive(btn.dataset.target));
    }}

    const initial = decodeURIComponent(window.location.hash.slice(1));
    const available = new Set(buttons.map((btn) => btn.dataset.target));
    setActive(available.has(initial) ? initial : {first!r});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _preferred_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _write_bnlearn_experiments_viewer(generated_html_paths: list[Path], output_suffix: str) -> Path:
    notebook_paths = [
        _preferred_existing_path(
            _apply_output_suffix(FIGS_DIR / "Fig.bn_dag_SHD_F1.html", output_suffix),
            _apply_output_suffix(FIGS_DIR / "Fig.bn_dag_SHD_SID.html", output_suffix),
        ),
        _preferred_existing_path(
            _apply_output_suffix(FIGS_DIR / "Fig.bn_dag_SID.html", output_suffix),
            _apply_output_suffix(FIGS_DIR / "Fig.bn_dag_F1.html", output_suffix),
        ),
        _apply_output_suffix(FIGS_DIR / "Fig.bn_dag_prec_rec.html", output_suffix),
        _apply_output_suffix(FIGS_DIR / "Fig.bn_cpdag_SID_best_worst.html", output_suffix),
        _apply_output_suffix(FIGS_DIR / "Fig.bn_cpdag_SHD_F1.html", output_suffix),
        _apply_output_suffix(FIGS_DIR / "Fig.bn_abapc_variants_sid_with_aspforaba.html", output_suffix),
        _preferred_existing_path(
            _apply_output_suffix(FIGS_DIR / "Fig.2_runtime.html", output_suffix),
            _apply_output_suffix(FIGS_DIR / "Fig.3_runtime_matched10.html", output_suffix),
        ),
    ]
    viewer_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_experiments_viewer.html", output_suffix)
    _write_viewer_html(notebook_paths + list(generated_html_paths), viewer_path, "BNLearn Plot Viewer")
    return viewer_path


def _write_links_html(html_paths: list[Path], output_path: Path, title: str) -> None:
    figures = []
    seen: set[str] = set()
    for path in html_paths:
        if path.suffix.lower() != ".html":
            continue
        name = path.name
        if name in seen or not path.exists():
            continue
        seen.add(name)
        figures.append(name)

    if not figures:
        return

    links = "\n".join(
        f'<li><a href="{name}" target="_blank" rel="noopener">{name}</a></li>'
        for name in figures
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} Links</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #dbe3f0;
      --text: #152033;
      --muted: #506178;
      --accent: #2358d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      font-family: Georgia, "Times New Roman", serif;
      background: var(--bg);
      color: var(--text);
    }}
    .panel {{
      max-width: 980px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px 28px;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 30px;
      line-height: 1.1;
    }}
    p {{
      margin: 0 0 18px 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.45;
    }}
    ul {{
      margin: 0;
      padding-left: 22px;
      columns: 2;
      column-gap: 36px;
    }}
    li {{
      break-inside: avoid;
      margin: 0 0 10px 0;
      padding-right: 10px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 16px;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 900px) {{
      body {{
        padding: 16px;
      }}
      ul {{
        columns: 1;
      }}
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>{title} Links</h1>
    <p>Direct links to the interactive HTML plots. Use this page if the embedded iframe viewer is slow in your browser.</p>
    <ul>
      {links}
    </ul>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _apply_output_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _series_color(pretty_name: str) -> str:
    for key, label in NAMES_DICT.items():
        if label == pretty_name:
            return COLORS_DICT[key]
    raise KeyError(pretty_name)


def _series_symbol(pretty_name: str) -> str:
    for key, label in NAMES_DICT.items():
        if label == pretty_name:
            return SYMBOLS_DICT[key]
    raise KeyError(pretty_name)


def build_cpdag_sid_compare(cpdag_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    present_methods = [method for method in METHOD_ORDER if method in cpdag_df["model"].unique()]
    label_names = ["Best", "Worst"]

    for side_index, metric in enumerate(["p_SID_low", "p_SID_high"]):
        for method_index, method in enumerate(present_methods):
            sub = cpdag_df[cpdag_df["model"] == method].copy()
            sub["_dataset_order"] = sub["dataset"].map({name: i for i, name in enumerate(DATASET_ORDER)})
            sub = sub.sort_values("_dataset_order")
            fig.add_trace(
                go.Bar(
                    x=sub["dataset"],
                    y=sub[f"{metric}_mean"],
                    error_y=dict(type="data", array=sub[f"{metric}_std"], visible=True),
                    name=method,
                    marker_color=_series_color(method),
                    opacity=0.6,
                    offsetgroup=method_index + len(present_methods) * side_index + side_index,
                    yaxis=f"y{side_index + 1}",
                    showlegend=side_index == 0,
                )
            )
        if side_index == 0 and present_methods:
            ref = cpdag_df[cpdag_df["model"] == present_methods[-1]].copy()
            ref["_dataset_order"] = ref["dataset"].map({name: i for i, name in enumerate(DATASET_ORDER)})
            ref = ref.sort_values("_dataset_order")
            fig.add_trace(
                go.Bar(
                    x=ref["dataset"],
                    y=np.zeros(len(ref)),
                    name="",
                    marker_color="white",
                    opacity=1.0,
                    offsetgroup=len(present_methods) + 1,
                    showlegend=False,
                )
            )

    fig.update_layout(
        barmode="group",
        bargap=0.08,
        bargroupgap=0.05,
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="top", y=1.23),
        template="plotly_white",
        width=1600,
        height=700,
        margin=dict(l=40, r=40, b=70, t=100),
        hovermode="x unified",
        font=dict(size=23, family="Serif", color="black"),
        yaxis2=dict(scaleanchor=0, showline=False, showgrid=False, showticklabels=False, zeroline=True),
    )
    fig.update_yaxes(title={"text": "Normalised SID", "font": {"size": 23}}, range=[0, 16.8], secondary_y=False)
    fig.update_yaxes(title={"text": "", "font": {"size": 23}}, range=[0, 16.8], secondary_y=True, showticklabels=False)

    unique_datasets = list(dict.fromkeys(cpdag_df.sort_values("n_nodes")["dataset"].tolist()))
    n_x_cat = max(len(unique_datasets), 1)
    cluster_width = 1.0 / n_x_cat
    total_tile_width = min(cluster_width * 0.55, cluster_width * 0.9)
    tile_width = total_tile_width / 2
    gap = min(cluster_width * 0.18, max(cluster_width - total_tile_width, 0.0))
    cluster_padding = max((cluster_width - (tile_width * 2 + gap)) / 2, 0.0)
    top_y0, top_y1 = 1.04, 1.10
    text_y = (top_y0 + top_y1) / 2

    for dataset_index in range(n_x_cat):
        cluster_left = dataset_index * cluster_width + cluster_padding
        for label_index, label_text in enumerate(label_names):
            left = cluster_left + label_index * (tile_width + gap)
            right = left + tile_width
            fig.add_shape(
                type="rect",
                xref="x domain",
                yref="y domain",
                x0=max(0, left - 0.02),
                x1=min(1, right + 0.02),
                y0=top_y0,
                y1=top_y1,
                line=dict(color="#E5ECF6", width=2),
                fillcolor="#E5ECF6",
                layer="below",
            )
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=(left + right) / 2,
                y=text_y,
                text=label_text,
                showarrow=False,
                font=dict(size=21, family="Serif", color="black"),
            )
    return fig


def build_runtime_compare(dag_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, pretty_name in NAMES_DICT.items():
        sub = dag_df[dag_df["model"] == pretty_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("n_nodes")
        fig.add_trace(
            go.Scatter(
                x=sub["n_nodes"],
                y=sub["elapsed_mean"],
                error_y=dict(type="data", array=sub["elapsed_std"], visible=True),
                mode="lines+markers",
                name=pretty_name,
                line=dict(width=5, color=COLORS_DICT[key]),
                marker=dict(size=18, symbol=SYMBOLS_DICT[key], color=COLORS_DICT[key]),
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=1400,
        height=700,
        margin=dict(l=40, r=40, b=60, t=60),
        legend=dict(orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=1.02),
        font=dict(size=26, family="Serif", color="black"),
        xaxis_title="Number of Nodes (|V|)",
        yaxis_title="log(elapsed time [s])",
    )
    fig.update_xaxes(dtick=1, showgrid=True, gridcolor="#E5ECF6")
    fig.update_yaxes(type="log", showgrid=True, gridcolor="#E5ECF6")
    return fig


def build_child_comparison_table(dag_df: pd.DataFrame, cpdag_df: pd.DataFrame) -> pd.DataFrame:
    dag_child = dag_df[dag_df["base_dataset"] == "child"].copy()
    cpdag_child = cpdag_df[cpdag_df["base_dataset"] == "child"].copy()
    merged = dag_child.merge(
        cpdag_child[["dataset", "model", "SID_low_mean", "SID_high_mean"]],
        on=["dataset", "model"],
        how="left",
    )
    return merged[
        [
            "model", "elapsed_mean", "precision_mean", "recall_mean", "F1_mean",
            "shd_mean", "SID_mean", "SID_low_mean", "SID_high_mean",
        ]
    ].sort_values("model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the full bnlearn matched-10 comparison for the seed-resolved methods.")
    parser.add_argument(
        "--fgs-nt-version",
        default=None,
        help="Legacy combined version containing Random/FGS/NOTEARS. Used as fallback when the split baseline version args are omitted.",
    )
    parser.add_argument("--random-version", default="bnlearn_random_matched10_gsq_graphmetrics")
    parser.add_argument("--fgs-version", default="bnlearn_fgs_matched10_gsq_graphmetrics")
    parser.add_argument("--random-fgs-version", default=None)
    parser.add_argument("--nt-version", default="bnlearn_nt_matched10_gsq_graphmetrics")
    parser.add_argument("--mpc-version", default="bnlearn_mpc_matched10_gsq_graphmetrics")
    parser.add_argument("--abapc-orig-others-version", default="bnlearn_abapc_orig_matched10_others_gsq")
    parser.add_argument("--abapc-orig-child-version", default="child_abapc_orig_matched10_gsq")
    parser.add_argument("--abapc-nor-others-version", default="bnlearn_abapc_nor_matched10_others_gsq_graphmetrics")
    parser.add_argument("--abapc-nor-child-version", default="child_abapc_nor_matched10_gsq_graphmetrics")
    parser.add_argument("--abapc-bb-others-version", default="bnlearn_abapc_bb_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-child-version", default="child_abapc_bb_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-nor-others-version", default="bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_graphmetrics")
    parser.add_argument("--abapc-bb-nor-child-version", default="child_abapc_bb_norapprox_matched10_gsq_searchv3_graphmetrics")
    parser.add_argument(
        "--disable-orig",
        action="store_true",
        help="Do not include ABAPC (orig) in the DAG/CPDAG metric figures, even if fallback archives exist.",
    )
    parser.add_argument(
        "--disable-runtime-orig",
        action="store_true",
        help="Do not include the fallback ABAPC (orig) trace in the runtime figure.",
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="*",
        default=[],
        choices=METHOD_ORDER,
        help="Pretty-name methods to remove from the DAG/CPDAG figures.",
    )
    parser.add_argument(
        "--exclude-runtime-methods",
        nargs="*",
        default=[],
        choices=METHOD_ORDER,
        help="Pretty-name methods to remove from the runtime figure only.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to generated figure filenames, for example '_no_nor'.",
    )
    parser.add_argument("--export-png", action="store_true", help="Export PNG companions for the generated HTML figures using a headless browser.")
    parser.add_argument("--png-browser", default="", help="Explicit browser binary for PNG export; defaults to auto-detecting Chrome/Chromium.")
    parser.add_argument("--png-width", type=int, default=1800, help="Viewport width used for browser-based PNG export.")
    parser.add_argument("--png-height", type=int, default=900, help="Viewport height used for browser-based PNG export.")
    parser.add_argument("--png-wait-ms", type=int, default=4000, help="Virtual-time budget for browser rendering before the screenshot is captured.")
    parser.add_argument("--png-device-scale-factor", type=float, default=2.0, help="Browser device scale factor for higher-resolution PNG screenshots.")
    args = parser.parse_args()

    random_version = _resolve_version(
        args.random_version or args.random_fgs_version or args.fgs_nt_version,
        "bnlearn_random_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    fgs_version = _resolve_version(
        args.fgs_version or args.random_fgs_version or args.fgs_nt_version,
        "bnlearn_fgs_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    nt_version = _resolve_version(
        args.nt_version or args.fgs_nt_version,
        "bnlearn_nt_matched10_gsq_graphmetrics",
        "bnlearn_nt_matched10_gsq",
    )
    baseline_artifact_version = _resolve_version(
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    mpc_version = _resolve_version(
        args.mpc_version,
        "bnlearn_mpc_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq_graphmetrics",
        "bnlearn_baselines_matched10_gsq",
    )
    abapc_nor_others_version = _resolve_version(
        args.abapc_nor_others_version,
        "bnlearn_abapc_nor_matched10_others_gsq_graphmetrics",
        "bnlearn_abapc_nor_matched10_others_gsq",
    )
    abapc_nor_child_version = _resolve_version(
        args.abapc_nor_child_version,
        "child_abapc_nor_matched10_gsq_graphmetrics",
        "child_abapc_nor_matched10_gsq",
    )
    abapc_bb_others_version = _resolve_version(
        args.abapc_bb_others_version,
        "bnlearn_abapc_bb_matched10_others_gsq_searchv3_graphmetrics",
        "bnlearn_abapc_bb_matched10_others_gsq_searchv3",
    )
    abapc_bb_child_version = _resolve_version(
        args.abapc_bb_child_version,
        "child_abapc_bb_matched10_gsq_searchv3_graphmetrics",
        "child_abapc_bb_matched10_gsq_searchv3",
    )
    abapc_bb_nor_others_version = _resolve_version(
        args.abapc_bb_nor_others_version,
        "bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_graphmetrics",
        "bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3",
    )
    abapc_bb_nor_child_version = _resolve_version(
        args.abapc_bb_nor_child_version,
        "child_abapc_bb_norapprox_matched10_gsq_searchv3_graphmetrics",
        "child_abapc_bb_norapprox_matched10_gsq_searchv3",
    )
    abapc_orig_others_version = _resolve_version(
        args.abapc_orig_others_version,
        "abapc_orig_problem3_matched10_gsq",
    )
    abapc_orig_child_version = _resolve_version(args.abapc_orig_child_version)

    # The plotting helpers also try to render interactively and export images.
    # For scripted report generation, keep the HTML output and suppress the rest.
    go.Figure.show = lambda self, *args, **kwargs: None
    go.Figure.write_image = lambda self, *args, **kwargs: None

    run_specs = [
        {"version": random_version, "artifact_version": random_version, "kind": "dag", "include": ["Random"]},
        {"version": fgs_version, "artifact_version": baseline_artifact_version, "kind": "dag", "include": ["FGS"]},
        {"version": nt_version, "kind": "dag", "include": ["NOTEARS-MLP"]},
        {"version": mpc_version, "artifact_version": baseline_artifact_version, "kind": "dag", "include": ["MPC"]},
        {
            "version": abapc_nor_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": abapc_nor_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": abapc_bb_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": abapc_bb_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": abapc_bb_nor_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {
            "version": abapc_bb_nor_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {"version": random_version, "artifact_version": random_version, "kind": "cpdag", "include": ["Random"]},
        {"version": fgs_version, "artifact_version": baseline_artifact_version, "kind": "cpdag", "include": ["FGS"]},
        {"version": nt_version, "kind": "cpdag", "include": ["NOTEARS-MLP"]},
        {"version": mpc_version, "artifact_version": baseline_artifact_version, "kind": "cpdag", "include": ["MPC"]},
        {
            "version": abapc_nor_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": abapc_nor_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": abapc_bb_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": abapc_bb_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": abapc_bb_nor_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
        {
            "version": abapc_bb_nor_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb-nor)"},
        },
    ]

    if not args.disable_orig:
        run_specs.extend(
            [
                {
                    "version": abapc_orig_others_version,
                    "kind": "dag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_child_version,
                    "kind": "dag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["child"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_others_version,
                    "kind": "cpdag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
                {
                    "version": abapc_orig_child_version,
                    "kind": "cpdag",
                    "include": ["ABAPC (Ours)"],
                    "include_datasets": ["child"],
                    "replace_models": {"ABAPC (Ours)": "ABAPC (orig)"},
                },
            ]
        )

    dag_df = _sort_by_dataset_order(_add_normalised_dag(_add_dataset_metadata(_load_combined("dag", run_specs))))
    cpdag_df = _sort_by_dataset_order(_add_normalised_cpdag(_add_dataset_metadata(_load_combined("cpdag", run_specs))))

    if args.exclude_methods:
        exclude_methods = set(args.exclude_methods)
        dag_df = dag_df[~dag_df["model"].isin(exclude_methods)].copy()
        cpdag_df = cpdag_df[~cpdag_df["model"].isin(exclude_methods)].copy()
    else:
        exclude_methods = set()

    if dag_df.empty or cpdag_df.empty:
        raise SystemExit("No matched-10 summaries were found for the requested versions.")

    dag_methods = [method for method in METHOD_ORDER if method in dag_df["model"].unique()]
    cpdag_methods = [method for method in METHOD_ORDER if method in cpdag_df["model"].unique()]
    artifact_version_lookup = _build_version_lookup(run_specs, kind="dag")
    true_graphs = _load_true_graphs(artifact_version_lookup, DATASET_ORDER)
    size_names_dict, size_colors_dict = _build_size_style_dicts()
    size_methods = [
        TRUE_GRAPH_LABEL,
        *[
            method
            for method in METHOD_ORDER
            if method in dag_df["model"].unique() and method not in {"Random", "ABAPC (orig)"}
        ],
    ]
    dag_size_df = _build_dag_size_frame(dag_df, size_methods, true_graphs)
    cpdag_size_df = _build_cpdag_size_frame(cpdag_df, size_methods, artifact_version_lookup, true_graphs)
    runtime_df = dag_df.copy()
    if not args.disable_runtime_orig:
        runtime_df = _augment_runtime_with_orig_fallback(runtime_df)
    if args.exclude_runtime_methods:
        runtime_df = runtime_df[~runtime_df["model"].isin(set(args.exclude_runtime_methods))].copy()
    runtime_method_keys = [
        method
        for method in ["random", "fgs", "nt", "mpc", "abapc_orig", "abapc_nor", "abapc_bb", "abapc_bb_nor"]
        if NAMES_DICT[method] in runtime_df["model"].unique()
    ]

    generated_html_paths: list[Path] = []
    output_suffix = args.output_suffix

    double_bar_chart_plotly(
        dag_df, ["p_shd", "F1"], NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SHD_F1.html", output_suffix)),
        debug=False, range_y1=[0, 2.6], range_y2=[0, 5.6], rect_exp=0.01,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SHD_F1.html", output_suffix))
    bar_chart_plotly(
        dag_df, "p_SID", NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SID.html", output_suffix)),
        debug=False,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SID.html", output_suffix))
    double_bar_chart_plotly(
        dag_df, ["precision", "recall"], NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_prec_rec.html", output_suffix)),
        debug=False,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_prec_rec.html", output_suffix))
    bar_chart_plotly(
        dag_size_df, "nnz", size_names_dict, size_colors_dict, size_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_size.html", output_suffix)),
        debug=False,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_size.html", output_suffix))
    if _metrics_available(dag_df, ["adjacency_F1", "arrowhead_F1"]):
        double_bar_chart_plotly(
            dag_df, ["adjacency_F1", "arrowhead_F1"], NAMES_DICT, COLORS_DICT, dag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_skeleton_arrowhead_F1.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_skeleton_arrowhead_F1.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_dag_skeleton_arrowhead_F1.html: skeleton/arrowhead F1 metrics are missing from the saved summaries.")
    if _metrics_available(dag_df, ["adjacency_precision", "adjacency_recall"]):
        double_bar_chart_plotly(
            dag_df, ["adjacency_precision", "adjacency_recall"], NAMES_DICT, COLORS_DICT, dag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_skeleton_prec_rec.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_skeleton_prec_rec.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_dag_skeleton_prec_rec.html: skeleton precision/recall metrics are missing from the saved summaries.")
    if _metrics_available(dag_df, ["arrowhead_precision", "arrowhead_recall"]):
        double_bar_chart_plotly(
            dag_df, ["arrowhead_precision", "arrowhead_recall"], NAMES_DICT, COLORS_DICT, dag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_arrowhead_prec_rec.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_arrowhead_prec_rec.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_dag_arrowhead_prec_rec.html: arrowhead precision/recall metrics are missing from the saved summaries.")
    double_bar_chart_plotly(
        cpdag_df, ["p_SID_low", "p_SID_high"], NAMES_DICT, COLORS_DICT, cpdag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.2_SID_cpdag_matched10.html", output_suffix)),
        debug=False, range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.2_SID_cpdag_matched10.html", output_suffix))
    double_bar_chart_plotly(
        cpdag_df, ["p_shd", "F1"], NAMES_DICT, COLORS_DICT, cpdag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_SHD_F1.html", output_suffix)),
        debug=False, range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_SHD_F1.html", output_suffix))
    double_bar_chart_plotly(
        cpdag_df, ["precision", "recall"], NAMES_DICT, COLORS_DICT, cpdag_methods,
        save_figs=True, font_size=23,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_prec_rec.html", output_suffix)),
        debug=False,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_prec_rec.html", output_suffix))
    if not cpdag_size_df.empty:
        max_directed = float(np.nanmax(cpdag_size_df["cpdag_directed_edges_mean"].to_numpy(dtype=float)))
        max_undirected = float(np.nanmax(cpdag_size_df["cpdag_undirected_edges_mean"].to_numpy(dtype=float)))
        double_bar_chart_plotly(
            cpdag_size_df,
            ["cpdag_directed_edges", "cpdag_undirected_edges"],
            size_names_dict,
            size_colors_dict,
            size_methods,
            save_figs=True,
            font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_size.html", output_suffix)),
            debug=False,
            range_y1=[0, max_directed + 1.0],
            range_y2=[0, max_undirected + 1.0],
            rect_exp=0.008,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_size.html", output_suffix))
    if _metrics_available(cpdag_df, ["adjacency_F1", "arrowhead_F1"]):
        double_bar_chart_plotly(
            cpdag_df, ["adjacency_F1", "arrowhead_F1"], NAMES_DICT, COLORS_DICT, cpdag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_arrowhead_F1.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_arrowhead_F1.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_cpdag_skeleton_arrowhead_F1.html: skeleton/arrowhead F1 metrics are missing from the saved summaries.")
    if _metrics_available(cpdag_df, ["adjacency_precision", "adjacency_recall"]):
        double_bar_chart_plotly(
            cpdag_df, ["adjacency_precision", "adjacency_recall"], NAMES_DICT, COLORS_DICT, cpdag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_prec_rec.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_prec_rec.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_cpdag_skeleton_prec_rec.html: skeleton precision/recall metrics are missing from the saved summaries.")
    if _metrics_available(cpdag_df, ["arrowhead_precision", "arrowhead_recall"]):
        double_bar_chart_plotly(
            cpdag_df, ["arrowhead_precision", "arrowhead_recall"], NAMES_DICT, COLORS_DICT, cpdag_methods,
            save_figs=True, font_size=23,
            output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_arrowhead_prec_rec.html", output_suffix)),
            debug=False,
        )
        generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_arrowhead_prec_rec.html", output_suffix))
    else:
        print("Skipping Fig.bn_matched10_cpdag_arrowhead_prec_rec.html: arrowhead precision/recall metrics are missing from the saved summaries.")
    plot_runtime(
        runtime_df,
        ["n_nodes"],
        "",
        NAMES_DICT,
        SYMBOLS_DICT,
        COLORS_DICT,
        runtime_method_keys,
        share_y=False,
        save_figs=True,
        output_name=str(_apply_output_suffix(FIGS_DIR / "Fig.3_runtime_matched10.html", output_suffix)),
        debug=False,
        font_size=20,
        plot_height=370,
        plot_width=800,
        model_aliases=NAMES_DICT,
    )
    generated_html_paths.append(_apply_output_suffix(FIGS_DIR / "Fig.3_runtime_matched10.html", output_suffix))

    child_table = build_child_comparison_table(dag_df, cpdag_df)
    child_table_path = _apply_output_suffix(FIGS_DIR / "bnlearn_matched10_child_table.csv", output_suffix)
    child_table.to_csv(child_table_path, index=False)
    viewer_title = "Plot Viewer"
    links_title = "Plot Links"
    viewer_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_viewer.html", output_suffix)
    _write_viewer_html(generated_html_paths, viewer_path, viewer_title)
    experiments_viewer_path = _write_bnlearn_experiments_viewer(generated_html_paths, output_suffix)
    links_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_links.html", output_suffix)
    _write_links_html(generated_html_paths, links_path, links_title)

    dag_shd_f1_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SHD_F1.html", output_suffix)
    dag_sid_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_SID.html", output_suffix)
    dag_prec_rec_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_prec_rec.html", output_suffix)
    dag_size_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_dag_size.html", output_suffix)
    cpdag_sid_path = _apply_output_suffix(FIGS_DIR / "Fig.2_SID_cpdag_matched10.html", output_suffix)
    cpdag_shd_f1_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_SHD_F1.html", output_suffix)
    cpdag_prec_rec_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_prec_rec.html", output_suffix)
    cpdag_size_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_size.html", output_suffix)
    cpdag_sk_ah_f1_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_arrowhead_F1.html", output_suffix)
    cpdag_sk_prec_rec_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_skeleton_prec_rec.html", output_suffix)
    cpdag_ah_prec_rec_path = _apply_output_suffix(FIGS_DIR / "Fig.bn_matched10_cpdag_arrowhead_prec_rec.html", output_suffix)
    runtime_path = _apply_output_suffix(FIGS_DIR / "Fig.3_runtime_matched10.html", output_suffix)

    print(child_table.to_string(index=False))
    print(f"Wrote {dag_shd_f1_path}")
    print(f"Wrote {dag_sid_path}")
    print(f"Wrote {dag_prec_rec_path}")
    print(f"Wrote {dag_size_path}")
    print(f"Wrote {cpdag_sid_path}")
    print(f"Wrote {cpdag_shd_f1_path}")
    print(f"Wrote {cpdag_prec_rec_path}")
    print(f"Wrote {cpdag_size_path}")
    print(f"Wrote {cpdag_sk_ah_f1_path}")
    print(f"Wrote {cpdag_sk_prec_rec_path}")
    print(f"Wrote {cpdag_ah_prec_rec_path}")
    print(f"Wrote {runtime_path}")
    print(f"Wrote {viewer_path}")
    print(f"Wrote {experiments_viewer_path}")
    print(f"Wrote {links_path}")
    print(f"Wrote {child_table_path}")

    if args.export_png:
        browser_path = _find_png_browser(args.png_browser.strip() or None)
        if browser_path is None:
            print("PNG export requested, but no Chrome/Chromium browser was found.")
        else:
            for html_path in generated_html_paths:
                png_path = html_path.with_suffix(".png")
                ok = _export_png_via_browser(
                    browser_path=browser_path,
                    html_path=html_path,
                    png_path=png_path,
                    width=args.png_width,
                    height=args.png_height,
                    wait_ms=args.png_wait_ms,
                    device_scale_factor=args.png_device_scale_factor,
                )
                if ok:
                    print(f"Wrote {png_path}")
                else:
                    print(f"Failed to write {png_path}")


if __name__ == "__main__":
    main()
