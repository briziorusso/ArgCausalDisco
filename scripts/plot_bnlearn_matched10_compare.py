#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    main_green,
    main_purple,
    plot_runtime,
    sec_blue,
    sec_orange,
)


RESULTS_DIR = REPO_ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


DAG_COLS = [
    "dataset", "model", "elapsed_mean", "elapsed_std", "nnz_mean", "nnz_std",
    "fdr_mean", "fdr_std", "tpr_mean", "tpr_std", "fpr_mean", "fpr_std",
    "precision_mean", "precision_std", "recall_mean", "recall_std",
    "F1_mean", "F1_std", "shd_mean", "shd_std", "SID_mean", "SID_std",
]
CPDAG_COLS = [
    "dataset", "model", "elapsed_mean", "elapsed_std", "nnz_mean", "nnz_std",
    "fdr_mean", "fdr_std", "tpr_mean", "tpr_std", "fpr_mean", "fpr_std",
    "precision_mean", "precision_std", "recall_mean", "recall_std",
    "F1_mean", "F1_std", "shd_mean", "shd_std",
    "SID_low_mean", "SID_low_std", "SID_high_mean", "SID_high_std",
]

DATASET_ORDER = ["cancer", "earthquake", "survey", "asia", "sachs", "child"]
NODES_MAP = {"asia": 8, "cancer": 5, "earthquake": 5, "sachs": 11, "survey": 6, "child": 20}
EDGES_MAP = {"asia": 8, "cancer": 4, "earthquake": 4, "sachs": 17, "survey": 6, "child": 25}

METHOD_ORDER = ["Random", "FGS", "NOTEARS-MLP", "MPC", "ABAPC (nor)", "ABAPC (bb)"]
NAMES_DICT = {
    "random": "Random",
    "fgs": "FGS",
    "nt": "NOTEARS-MLP",
    "mpc": "MPC",
    "abapc_nor": "ABAPC (nor)",
    "abapc_bb": "ABAPC (bb)",
}
COLORS_DICT = {
    "random": "#7f7f7f",
    "fgs": sec_orange,
    "nt": sec_blue,
    "mpc": main_green,
    "abapc_nor": "#bcbd22",
    "abapc_bb": main_purple,
}
SYMBOLS_DICT = {
    "random": "x",
    "fgs": "circle-open-dot",
    "nt": "x",
    "mpc": "diamond-dot",
    "abapc_nor": "triangle-down-dot",
    "abapc_bb": "triangle-up-dot",
}


def _load_summary(version: str, kind: str) -> pd.DataFrame:
    suffix = "_cpdag.npy" if kind == "cpdag" else ".npy"
    columns = CPDAG_COLS if kind == "cpdag" else DAG_COLS
    path = RESULTS_DIR / f"stored_results_{version}{suffix}"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(np.load(path, allow_pickle=True), columns=columns)
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["model"] = frame["model"].astype(str)
    return frame


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
    out["n_nodes"] = out["dataset"].map(NODES_MAP).astype(float)
    out["n_edges"] = out["dataset"].map(EDGES_MAP).astype(float)
    out["dataset_label"] = out["dataset"].map(
        lambda name: f"{name.upper()}<br> |V|={NODES_MAP[name]}, |E|={EDGES_MAP[name]}"
    )
    return out


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
                    x=sub["dataset_label"],
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
                    x=ref["dataset_label"],
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

    unique_datasets = list(dict.fromkeys(cpdag_df.sort_values("n_nodes")["dataset_label"].tolist()))
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
    dag_child = dag_df[dag_df["dataset"] == "child"].copy()
    cpdag_child = cpdag_df[cpdag_df["dataset"] == "child"].copy()
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
    parser.add_argument("--fgs-nt-version", default="bnlearn_fgs_nt_matched10")
    parser.add_argument("--mpc-version", default="bnlearn_50rep_mpc_matched10")
    parser.add_argument("--abapc-nor-version", default="bnlearn_50rep_abapc_matched10")
    parser.add_argument("--abapc-bb-others-version", default="bnlearn_abapc_bb_matched10_others")
    parser.add_argument("--abapc-bb-child-version", default="child_sat24_t600_matched10")
    args = parser.parse_args()

    # The plotting helpers also try to render interactively and export images.
    # For scripted report generation, keep the HTML output and suppress the rest.
    go.Figure.show = lambda self, *args, **kwargs: None
    go.Figure.write_image = lambda self, *args, **kwargs: None

    run_specs = [
        {"version": args.fgs_nt_version, "kind": "dag", "include": ["Random", "FGS", "NOTEARS-MLP"]},
        {"version": args.mpc_version, "kind": "dag", "include": ["MPC"]},
        {
            "version": args.abapc_nor_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_bb_others_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_child_version,
            "kind": "dag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {"version": args.fgs_nt_version, "kind": "cpdag", "include": ["Random", "FGS", "NOTEARS-MLP"]},
        {"version": args.mpc_version, "kind": "cpdag", "include": ["MPC"]},
        {
            "version": args.abapc_nor_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (nor)"},
        },
        {
            "version": args.abapc_bb_others_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["cancer", "earthquake", "survey", "asia", "sachs"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
        {
            "version": args.abapc_bb_child_version,
            "kind": "cpdag",
            "include": ["ABAPC (Ours)"],
            "include_datasets": ["child"],
            "replace_models": {"ABAPC (Ours)": "ABAPC (bb)"},
        },
    ]

    dag_df = _add_normalised_dag(_add_dataset_metadata(_load_combined("dag", run_specs)))
    cpdag_df = _add_normalised_cpdag(_add_dataset_metadata(_load_combined("cpdag", run_specs)))

    if dag_df.empty or cpdag_df.empty:
        raise SystemExit("No matched-10 summaries were found for the requested versions.")

    dag_methods = [method for method in METHOD_ORDER if method in dag_df["model"].unique()]
    cpdag_methods = [method for method in METHOD_ORDER if method in cpdag_df["model"].unique()]

    double_bar_chart_plotly(
        dag_df, ["p_shd", "F1"], NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(FIGS_DIR / "Fig.bn_matched10_dag_SHD_F1.html"),
        debug=False, range_y1=[0, 2.6], range_y2=[0, 5.6], rect_exp=0.01,
    )
    bar_chart_plotly(
        dag_df, "p_SID", NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(FIGS_DIR / "Fig.bn_matched10_dag_SID.html"),
        debug=False,
    )
    double_bar_chart_plotly(
        dag_df, ["precision", "recall"], NAMES_DICT, COLORS_DICT, dag_methods,
        save_figs=True, font_size=23,
        output_name=str(FIGS_DIR / "Fig.bn_matched10_dag_prec_rec.html"),
        debug=False,
    )
    double_bar_chart_plotly(
        cpdag_df, ["p_SID_low", "p_SID_high"], NAMES_DICT, COLORS_DICT, cpdag_methods,
        save_figs=True, font_size=23,
        output_name=str(FIGS_DIR / "Fig.2_SID_cpdag_matched10.html"),
        debug=False, range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
    )
    double_bar_chart_plotly(
        cpdag_df, ["p_shd", "F1"], NAMES_DICT, COLORS_DICT, cpdag_methods,
        save_figs=True, font_size=23,
        output_name=str(FIGS_DIR / "Fig.bn_matched10_cpdag_SHD_F1.html"),
        debug=False, range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
    )
    plot_runtime(
        dag_df,
        ["n_nodes"],
        "",
        NAMES_DICT,
        SYMBOLS_DICT,
        COLORS_DICT,
        ["random", "fgs", "nt", "mpc", "abapc_nor", "abapc_bb"],
        share_y=False,
        save_figs=True,
        output_name=str(FIGS_DIR / "Fig.3_runtime_matched10.html"),
        debug=False,
        font_size=20,
        plot_height=370,
        plot_width=800,
        model_aliases=NAMES_DICT,
    )

    child_table = build_child_comparison_table(dag_df, cpdag_df)
    child_table_path = FIGS_DIR / "bnlearn_matched10_child_table.csv"
    child_table.to_csv(child_table_path, index=False)

    print(child_table.to_string(index=False))
    print(f"Wrote {FIGS_DIR / 'Fig.bn_matched10_dag_SHD_F1.html'}")
    print(f"Wrote {FIGS_DIR / 'Fig.bn_matched10_dag_SID.html'}")
    print(f"Wrote {FIGS_DIR / 'Fig.bn_matched10_dag_prec_rec.html'}")
    print(f"Wrote {FIGS_DIR / 'Fig.2_SID_cpdag_matched10.html'}")
    print(f"Wrote {FIGS_DIR / 'Fig.bn_matched10_cpdag_SHD_F1.html'}")
    print(f"Wrote {FIGS_DIR / 'Fig.3_runtime_matched10.html'}")
    print(f"Wrote {child_table_path}")


if __name__ == "__main__":
    main()
