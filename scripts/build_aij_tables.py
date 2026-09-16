#!/usr/bin/env python3
"""Build auditable AIJ LaTeX tables from saved experiments; never run learners.

See scripts/README_AIJ.md for sources, metric definitions and server commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import warnings
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ("cancer", "earthquake", "survey", "asia", "sachs", "child")
NODES = dict(zip(DATASETS, (5, 5, 6, 8, 11, 20)))
EDGES = dict(zip(DATASETS, (4, 4, 6, 8, 17, 25)))
SEEDS = (7816, 3578, 2656, 2688, 2494, 183, 7977, 3199, 316, 8266)
METHODS = ("Random", "FGS", "NOTEARS-MLP", "MPC", "ABAPC (bb)", "ABAPC (bb-nor)")
VARIANTS = ("ABAPC (nor)", "ABAPC (bb)", "ABAPC (bb-nor)")
SPECS = {
    "Random": ("random", "bnlearn_random_matched10_gsq_graphmetrics", None),
    "FGS": ("fgs", "bnlearn_fgs_matched10_gsq_graphmetrics", None),
    "NOTEARS-MLP": ("nt", "bnlearn_nt_matched10_gsq_graphmetrics", None),
    "MPC": ("mpc", "bnlearn_mpc_matched10_gsq_graphmetrics", None),
    "ABAPC (nor)": ("abapc", "bnlearn_abapc_nor_matched10_others_gsq_graphmetrics", "child_abapc_nor_matched10_gsq_graphmetrics"),
    "ABAPC (bb)": ("abapc", "bnlearn_abapc_bb_matched10_others_gsq_searchv3_graphmetrics", "child_abapc_bb_matched10_gsq_searchv3_graphmetrics"),
    "ABAPC (bb-nor)": ("abapc", "bnlearn_abapc_bb_norapprox_matched10_others_gsq_searchv3_graphmetrics", "child_abapc_bb_norapprox_matched10_gsq_searchv3_graphmetrics"),
}
LEGACY_METRICS = ("elapsed", "nnz", "fdr", "tpr", "fpr", "precision", "recall", "F1", "shd", "sid")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


class Sources:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, dict] = {}

    def use(self, relative: str | Path, purpose: str = "data") -> Path:
        path = self.root / relative
        data = path.read_bytes()
        name = path.relative_to(self.root).as_posix()
        self.files[name] = {"sha256": sha(data), "bytes": len(data), "purpose": purpose}
        return path

    def csv(self, relative: str | Path, purpose: str = "data") -> pd.DataFrame:
        return pd.read_csv(self.use(relative, purpose))


def validate_seeds(frame: pd.DataFrame, keys: list[str], expected=SEEDS) -> None:
    if frame.empty:
        raise ValueError(f"No records for {keys}")
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError(f"Duplicate seed records in {keys}")
    for key, group in frame.groupby(keys, dropna=False):
        if set(group.seed) != set(expected):
            raise ValueError(f"Seed mismatch for {key}: {sorted(group.seed.tolist())}")


def version_for(method: str, dataset: str) -> str:
    _, version, child = SPECS[method]
    return child if dataset == "child" and child else version


def validate_launches(src: Sources) -> None:
    versions = {version_for(m, d) for m in SPECS for d in DATASETS}
    # The split FGS/MPC summaries were reconstructed from this shared launch.
    versions.difference_update({SPECS["FGS"][1], SPECS["MPC"][1]})
    versions.add("bnlearn_baselines_matched10_gsq_graphmetrics")
    for version in sorted(versions):
        path = src.use(f"results/matched10/{version}_launch.json", "Sample size, CI settings and seed provenance")
        launch = json.loads(path.read_text(encoding="utf-8"))
        if launch["selected_seeds"] != list(SEEDS):
            raise ValueError(f"Launch seed mismatch: {version}")
        command = launch["command"]
        for flag, expected in (("--sample_size", "5000"), ("--test_alpha", "0.05"), ("--test_name", "gsq")):
            if command[command.index(flag)+1] != expected:
                raise ValueError(f"Launch configuration mismatch: {version}, {flag}")


def load_archived_primary(src: Sources) -> pd.DataFrame:
    frames = []
    for method, (key, _, _) in SPECS.items():
        for dataset in DATASETS:
            version = version_for(method, dataset)
            for kind in ("dag", "cpdag"):
                path = f"results/progress/{version}/{dataset}__{key}_{kind}.csv"
                df = src.csv(path)
                if set(df.dataset) != {dataset}:
                    raise ValueError(f"Wrong dataset in {path}")
                expected_model = "ABAPC (Ours)" if key == "abapc" else method
                if set(df.model) != {expected_model}:
                    raise ValueError(f"Wrong method in {path}")
                df = df.assign(method=method, kind=kind, version=version, source=path)
                for metric in ("shd", "sid", "sid_low", "sid_high"):
                    if metric in df:
                        df[f"n{metric}"] = pd.to_numeric(df[metric], errors="raise") / EDGES[dataset]
                frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    validate_seeds(all_rows, ["dataset", "method", "kind"])
    required = ["elapsed", "nnz", "shd", "adjacency_F1", "arrowhead_F1"]
    if not np.isfinite(all_rows[required].to_numpy(dtype=float)).all():
        raise ValueError("A required primary metric is missing or non-finite")
    if np.isinf(all_rows.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Infinite measurements in primary records")
    return all_rows


def load_primary(src: Sources) -> pd.DataFrame:
    manifest=json.loads(src.use('results/aij_evaluation/v2/manifest.json', 'Corrected graph-evaluation provenance').read_text())
    path=src.use('results/aij_evaluation/v2/records.csv','Corrected per-run metrics from saved raw graphs')
    if sha(path.read_bytes()) != manifest['records_sha256']:
        raise ValueError('Corrected evaluation CSV hash mismatch')
    for name,expected in manifest['code'].items():
        if sha(src.use(name,'Corrected metric implementation').read_bytes()) != expected:
            raise ValueError(f'Rerun scripts/reevaluate_aij_graphs.py after editing {name}')
    frame=pd.read_csv(path)
    validate_seeds(frame,['dataset','method','kind'])
    for row in frame[frame.kind=='dag'].itertuples():
        provenance=json.loads(src.use(row.evaluation_source,'Per-run corrected evaluation and input hashes').read_text())
        key=SPECS[row.method][0]
        version='bnlearn_baselines_matched10_gsq_graphmetrics' if key in ('mpc','fgs') else version_for(row.method,row.dataset)
        folder,=(src.root/f'results/{key}_{version}_{row.dataset}/runs').glob(f'run_*_seed_{row.seed}')
        for filename,field in [('graph_est_raw.npy','raw_sha256'),('graph_true.npy','truth_sha256')]:
            source=src.use((folder/filename).relative_to(src.root),'Original graph used for corrected evaluation')
            if sha(source.read_bytes()) != provenance['identity'][field]:
                raise ValueError(f'Raw graph changed after reevaluation: {source}')
    return frame


def aggregate(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, sort=False)[metrics].agg(["mean", "std", "count"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def paired_effect(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> dict:
    """Join by seed; report complete-pair counts and every omitted seed."""
    if left.seed.duplicated().any() or right.seed.duplicated().any():
        raise ValueError("A paired comparison contains duplicate seeds")
    if set(left.seed) != set(right.seed):
        raise ValueError("A paired comparison has unmatched seeds")
    pair = left[["seed", metric]].merge(right[["seed", metric]], on="seed", validate="one_to_one", suffixes=("_left", "_right"))
    x = pair[f"{metric}_left"].to_numpy(dtype=float)
    y = pair[f"{metric}_right"].to_numpy(dtype=float)
    complete = np.isfinite(x) & np.isfinite(y)
    excluded = pair.loc[~complete, "seed"].astype(int).tolist()
    expected = len(x)
    x, y = x[complete], y[complete]
    delta = x - y
    if len(delta) < 2:
        stat, p = np.nan, np.nan
    elif np.all(delta == 0):
        stat, p = 0.0, 1.0
    elif np.std(delta, ddof=1) < 1e-14:
        stat, p = math.copysign(math.inf, delta.mean()), 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            test = ttest_rel(x, y)
        stat, p = float(test.statistic), float(test.pvalue)
    return {"n": len(delta), "n_expected": expected, "excluded_seeds": json.dumps(excluded),
            "left_mean": x.mean() if len(x) else np.nan, "right_mean": y.mean() if len(y) else np.nan,
            "delta_mean": delta.mean() if len(delta) else np.nan, "delta_std": delta.std(ddof=1) if len(delta)>1 else np.nan,
            "t_statistic": stat, "p_value_unadjusted": p}


def make_tests(primary: pd.DataFrame, noninformative=frozenset()) -> pd.DataFrame:
    rows = []
    metrics = {"cpdag": ("nsid_low", "nsid_high", "nshd", "F1", "adjacency_F1", "arrowhead_F1",
                         "precision", "recall", "adjacency_precision", "adjacency_recall", "arrowhead_precision", "arrowhead_recall"),
               "dag": ("nsid", "nshd", "F1", "elapsed", "precision", "recall")}
    for dataset in DATASETS:
        for kind, names in metrics.items():
            part = primary[(primary.dataset == dataset) & (primary.kind == kind)]
            for reference in METHODS[:-1]:
                for metric in names:
                    effect = paired_effect(part[part.method == "ABAPC (bb-nor)"], part[part.method == reference], metric)
                    applicable = not (kind == "cpdag" and (dataset, metric) in noninformative)
                    if not applicable:
                        for field in ("left_mean", "right_mean", "delta_mean", "delta_std", "t_statistic", "p_value_unadjusted"):
                            effect[field] = np.nan
                    rows.append({"dataset": dataset, "kind": kind, "method": "ABAPC (bb-nor)", "reference": reference, "metric": metric,
                                 "applicable": applicable, "analysis_status": "not_applicable_no_reference_arrowheads" if not applicable else "available" if effect["n"] else "unavailable",
                                 **effect})
    out = pd.DataFrame(rows)
    # Holm correction across the five comparisons within each dataset/metric.
    out["p_value_holm"] = np.nan
    for _, idx in out.groupby(["dataset", "kind", "metric"]).groups.items():
        ordered = out.loc[idx].dropna(subset=["p_value_unadjusted"]).sort_values("p_value_unadjusted").index
        # Keep the declared five-comparison family even when a test is unavailable.
        adjusted = np.maximum.accumulate(out.loc[ordered, "p_value_unadjusted"].to_numpy() * (len(idx)-np.arange(len(ordered))))
        out.loc[ordered, "p_value_holm"] = np.minimum(adjusted, 1.0)
    return out


def count_edges(matrix: np.ndarray) -> tuple[int, int]:
    directed = undirected = 0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            x, y = float(matrix[i, j]), float(matrix[j, i])
            if x == -1 or y == -1 or (x != 0 and y != 0):
                undirected += 1
            elif (x != 0) != (y != 0):
                directed += 1
    return directed, undirected


def load_sizes(src: Sources, primary: pd.DataFrame) -> pd.DataFrame:
    from utils.graph_utils import dag2cpdag
    src.use("utils/graph_utils.py", "Reference graph conversion used by the existing figure")
    rows = []
    for dataset in DATASETS:
        reference = None
        for method in METHODS:
            key = SPECS[method][0]
            version = version_for(method, dataset)
            if method in ("FGS", "MPC"):
                version = "bnlearn_baselines_matched10_gsq_graphmetrics"
            for seed in SEEDS:
                base = Path(f"results/{key}_{version}_{dataset}/runs")
                matches = sorted((src.root / base).glob(f"run_*_seed_{seed}"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one saved run: {base}, seed={seed}")
                rel = matches[0].relative_to(src.root)
                dag = primary[(primary.dataset == dataset) & (primary.method == method) & (primary.kind == "dag") & (primary.seed == seed)].iloc[0]
                graph_path=Path(dag.evaluation_source).parent/'graphs.npz'
                with np.load(src.use(graph_path,'Evaluated graph representation')) as graphs:
                    d,u=count_edges(graphs['cpdag' if dag.graph_status=='valid' else 'pdag'])
                rows.append(dict(dataset=dataset, method=method, seed=seed, directed=d, undirected=u, dag_edges=dag.nnz))
                true = np.load(src.use(rel / "graph_true.npy"))
                if reference is not None and not np.array_equal(reference, true):
                    raise ValueError(f"Reference graph changes within {dataset}")
                reference = true
        if int(np.count_nonzero(reference)) != EDGES[dataset]:
            raise ValueError(f"Unexpected true edge count: {dataset}")
        d, u = count_edges(dag2cpdag(reference.copy()))
        for seed in SEEDS:
            rows.append(dict(dataset=dataset, method="Reference", seed=seed, directed=d, undirected=u, dag_edges=EDGES[dataset]))
    return pd.DataFrame(rows)


class NotebookTable(HTMLParser):
    """Read a saved pandas HTML table without an optional HTML dependency."""
    def __init__(self):
        super().__init__()
        self.rows, self.row = [], []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)


def load_semantics(src: Sources) -> pd.DataFrame:
    path = src.use("semantics/notebooks/OtherExtensionBasedSemantics/demo_co.ipynb", "Historical aggregates from saved output; CO_MAX raw records are absent")
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    matches = []
    for cell in cells:
        for output in cell.get("outputs", []):
            html = "".join(output.get("data", {}).get("text/html", []))
            if "ABAPC (New CO_MAX)" in html and "sid_low_std" in html:
                parser = NotebookTable()
                parser.feed(html)
                table = pd.DataFrame([row[1:] for row in parser.rows[1:]], columns=parser.rows[0][1:])
                if len(table) == 9:
                    matches.append(table)
    if len(matches) != 1:
        raise ValueError("Cannot uniquely recover the historical semantics aggregate table")
    df = matches[0].rename(columns={"model": "method"})
    for name in ("sid_low_mean", "sid_low_std", "sid_high_mean", "sid_high_std"):
        df[name] = pd.to_numeric(df[name], errors="raise")
    df["method"] = df.method.replace({"ABAPC (New ST)": "Causal ABA (Alt. ST)", "ABAPC (New CO)": "Causal ABA (Alt. CO)", "ABAPC (New CO_MAX)": "Causal ABA (Alt. CO-max)"})
    baseline = src.csv("semantics/results/extension_based_semantics/stored_results_bnlearn_50rep_cpdag.csv")
    baseline = baseline[baseline.dataset.isin(DATASETS[:3]) & ~baseline.model.isin(["ABAPC (ASPforABA)"])].copy()
    baseline = baseline.rename(columns={"model": "method"})
    baseline["method"] = baseline.method.replace({"ABAPC (Ours)": "Causal ABA (Original)"})
    fgs = src.csv("semantics/results/existing/bnlearn_graphs/fgs/all_existing_methods_metrics_cpdag.csv")
    fgs = fgs[fgs.dataset.isin(DATASETS[:3])].rename(columns={"model": "method"})
    fgs = aggregate(fgs, ["dataset", "method"], ["sid_low", "sid_high"])
    out = pd.concat([baseline, fgs, df], ignore_index=True)
    for metric in ("sid_low", "sid_high"):
        for stat in ("mean", "std"):
            out[f"n{metric}_{stat}"] = out[f"{metric}_{stat}"] / out.dataset.map(EDGES)
    return out


def legacy_summaries(src: Sources, primary: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the plotting fallback order but explicitly label its cohorts."""
    recent = primary[primary.kind == "dag"]
    out = aggregate(recent, ["dataset", "method"], ["elapsed", "nsid"])
    out["cohort"] = "AIJ / 5000 / 10"
    chosen = set()
    old = []
    for version, cohort in (("abapc_orig_problem3_matched10_gsq", "AIJ / 5000 / 10"), ("bnlearn_dag_v5_2000", "Legacy / 2000 / --"), ("bnlearn_dag_v5", "Legacy / 5000 / --")):
        path = src.use(f"results/stored_results_{version}.npy", "Historical plot fallback; pre-aggregated and rounded")
        a = np.load(path, allow_pickle=True)
        if a.shape[1] != 22:
            raise ValueError(f"Unexpected historical summary schema: {path}")
        columns = ["dataset", "method"] + [f"{m}_{s}" for m in LEGACY_METRICS for s in ("mean", "std")]
        for _, row in pd.DataFrame(a, columns=columns).iterrows():
            if "ABAPC" not in row.method or row.dataset in chosen:
                continue
            chosen.add(row.dataset)
            old.append(dict(dataset=row.dataset, method="ABAPC (orig)", cohort=cohort,
                            elapsed_mean=float(row.elapsed_mean), elapsed_std=float(row.elapsed_std),
                            nsid_mean=float(row.sid_mean)/EDGES[row.dataset], nsid_std=float(row.sid_std)/EDGES[row.dataset]))
    asp = src.csv("semantics/results/extension_based_semantics/stored_results_bnlearn_50rep.csv")
    for _, row in asp[asp.model == "ABAPC (ASPforABA)"].iterrows():
        old.append(dict(dataset=row.dataset, method="ASPforABA", cohort="Legacy / 5000 / 50",
                        elapsed_mean=row.elapsed_mean, elapsed_std=row.elapsed_std,
                        nsid_mean=row.sid_mean/EDGES[row.dataset], nsid_std=row.sid_std/EDGES[row.dataset]))
    return pd.concat([out, pd.DataFrame(old)], ignore_index=True)


def tex_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def cell(mean, std=None, digits=3, signed=False, bold=False, stacked=False, count=None, applicable=True, marker='') -> str:
    if not applicable:
        return r"\textnormal{n/a}"
    if pd.isna(mean):
        return "--"
    value = float(mean)
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    mean_text = format(value, f"{'+' if signed else ''}.{digits}f")
    text = mean_text
    if std is not None and pd.notna(std):
        text += rf"\,\pm\,{float(std):.{digits}f}"
    if bold:
        text = r"\mathbf{" + text + "}"
    if stacked and std is not None and pd.notna(std):
        mean_text = r"\mathbf{" + mean_text + "}" if bold else mean_text
        if count is not None:
            mean_text += rf"_{{{count}}}"
        if marker:
            mean_text += rf"^{{{marker}}}"
        sd_text = rf"\pm\,{float(std):.{digits}f}"
        if bold:
            sd_text = r"\mathbf{" + sd_text + "}"
        return r"\shortstack{$" + mean_text + r"$\\[-1pt]$\scriptstyle " + sd_text + "$}"
    if count is not None:
        text = "(" + text + rf")_{{{count}}}"
    if marker:
        text += rf"^{{{marker}}}"
    return "$" + text + "$"


class Tables:
    def __init__(self, output: Path):
        self.output = output
        self.names: list[str] = []

    def write(self, name: str, headers: list[str], rows: list[list[str] | None], caption: str) -> None:
        lines = ["% Generated by scripts/build_aij_tables.py; do not edit.",
                 "% Source revisions and input/output hashes: table_build_manifest.json",
                 r"\begin{table}[p]", r"\centering", r"\footnotesize",
                 r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
                 r"\begin{tabular}{@{}" + "l" * min(2, len(headers)) + "c" * (len(headers)-min(2,len(headers))) + "@{}}",
                 r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        for row in rows:
            lines.append(r"\midrule" if row is None else " & ".join(row) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}",
                      r"\label{tab:aij-" + name.replace("table_aij_", "").replace("_", "-") + "}", r"\end{table}", ""])
        filename = name + ".tex"
        (self.output / filename).write_text("\n".join(lines), encoding="utf-8")
        self.names.append(filename)


def significance_marker(tests, dataset, kind, method, metric, direction):
    if tests is None or method == 'ABAPC (bb-nor)': return ''
    rows=tests[(tests.dataset==dataset)&(tests.kind==kind)&(tests.reference==method)&(tests.metric==metric)]
    if rows.empty: return ''
    row=rows.iloc[0]
    if not row.applicable or pd.isna(row.p_value_holm) or row.p_value_holm>=.05: return ''
    target_better=row.delta_mean<0 if direction=='min' else row.delta_mean>0
    return r'\dagger' if target_better else r'\ddagger'


def metric_panels(tables: Tables, summary: pd.DataFrame, name: str, specs: list[tuple], caption: str, methods=METHODS, datasets=DATASETS, bold=False, noninformative=frozenset(), tests=None, kind='cpdag') -> None:
    for start in range(0, len(datasets), 3):
        subset = datasets[start:start+3]
        rows = []
        for dataset in subset:
            group = summary[(summary.dataset == dataset) & summary.method.isin(methods)]
            for i, method in enumerate(methods):
                found = group[group.method == method]
                if found.empty:
                    continue
                row = found.iloc[0]
                vals = []
                for metric, _, digits, direction in specs:
                    optimum = group[f"{metric}_mean"].min() if direction == "min" else group[f"{metric}_mean"].max()
                    count = row.get(f"{metric}_count", 10)
                    value = cell(row[f"{metric}_mean"], row[f"{metric}_std"], digits,
                                 bold=bold and pd.notna(optimum) and math.isclose(row[f"{metric}_mean"], optimum, abs_tol=1e-10),
                                 stacked=len(specs)>=5, count=int(count) if 0<count<10 else None,
                                 applicable=(dataset, metric) not in noninformative,
                                 marker=significance_marker(tests,dataset,kind,method,metric,direction))
                    vals.append(value)
                rows.append([dataset.title() if i == 0 else "", tex_escape(method), *vals])
            if dataset != subset[-1]:
                rows.append(None)
        suffix = f"_{start//3+1}" if len(datasets) > 3 else ""
        tables.write(name+suffix, ["Dataset", "Method", *[x[1] for x in specs]], rows,
                     caption
                     + (" Bold indicates the best mean, including ties." if bold else "")
                     + (r" $\dagger$ and $\ddagger$ indicate significantly worse and better performance, respectively, than ABAPC (bb-nor), using paired two-sided $t$-tests with Holm correction across five comparisons per dataset and metric ($p<0.05$)." if tests is not None else ""))


def noninformative_arrowhead_scores(sizes: pd.DataFrame) -> set[tuple[str, str]]:
    """A reference with no arrowheads cannot support an AH recovery comparison.

    This is a reference property, not a rule that hides low/zero method scores.
    Archive values remain unchanged; only their presentation/inference is masked.
    """
    reference = sizes[sizes.method == "Reference"]
    datasets = reference.groupby("dataset").directed.max()
    return {(dataset, metric) for dataset in datasets[datasets == 0].index
            for metric in ("arrowhead_precision", "arrowhead_recall", "arrowhead_F1")}


def make_primary_tables(tables: Tables, primary: pd.DataFrame, tests: pd.DataFrame, sizes: pd.DataFrame) -> None:
    noninformative = noninformative_arrowhead_scores(sizes)
    common = r"Results use ten matched seeds and 5000 observations per run. Entries report means above sample standard deviations; subscripts indicate fewer than ten defined measurements, and -- denotes an unavailable value."
    cp = primary[primary.kind == "cpdag"]
    specs = [("nsid_low", r"NSID$_{\min}\downarrow$", 2, "min"), ("nsid_high", r"NSID$_{\max}\downarrow$", 2, "min"),
             ("adjacency_F1", r"Sk-F1 $\uparrow$", 3, "max"), ("arrowhead_F1", r"AH-F1 $\uparrow$", 3, "max"),
             ("nshd", r"NSHD $\downarrow$", 2, "min"), ("F1", r"F1 $\uparrow$", 3, "max")]
    summary = aggregate(cp, ["dataset", "method"], [s[0] for s in specs])
    metric_panels(tables, summary, "table_aij_cpdag_core", specs, r"Graph reconstruction relative to the true CPDAG. " + common + r" NSID and NSHD divide Structural Interventional (Hamming, respectively) Distance by the number of true DAG edges. Sk-F1 measures skeleton recovery and AH-F1 arrowhead recovery. AH scores are n/a when the true CPDAG has no compelled arrowheads.", bold=True, noninformative=noninformative, tests=tests)
    specs = [(metric, title, 3, "max") for metric, title in (
        ("precision", r"P $\uparrow$"), ("recall", r"R $\uparrow$"),
        ("adjacency_precision", r"Sk-P $\uparrow$"), ("adjacency_recall", r"Sk-R $\uparrow$"),
        ("arrowhead_precision", r"AH-P $\uparrow$"), ("arrowhead_recall", r"AH-R $\uparrow$"))]
    metric_panels(tables, aggregate(cp, ["dataset", "method"], [s[0] for s in specs]), "table_aij_cpdag_precision_recall", specs,
                  r"Precision (P) and recall (R) relative to the true CPDAG, for edges, skeleton adjacencies (Sk), and arrowheads (AH). " + common + r" AH scores are n/a when the true CPDAG has no compelled arrowheads.", noninformative=noninformative, bold=True, tests=tests)
    dag = primary[primary.kind == "dag"]
    specs = [("nsid", r"NSID $\downarrow$", 2, "min"), ("nshd", r"NSHD $\downarrow$", 2, "min"),
             ("F1", r"F1 $\uparrow$", 3, "max"), ("precision", r"P $\uparrow$", 3, "max"), ("recall", r"R $\uparrow$", 3, "max")]
    metric_panels(tables, aggregate(dag, ["dataset", "method"], [s[0] for s in specs]), "table_aij_dag", specs,
                  r"Graph reconstruction relative to the true DAG. " + common + r" NSID and NSHD divide Structural Interventional (Hamming, respectively) Distance by the number of true DAG edges. P and R denote precision and recall.", bold=True, tests=tests, kind='dag')
    specs = [("directed", "Directed", 1, "max"), ("undirected", "Undirected", 1, "max"), ("dag_edges", "DAG edges", 1, "max")]
    metric_panels(tables, aggregate(sizes, ["dataset", "method"], [s[0] for s in specs]), "table_aij_graph_size", specs,
                  r"Graph sizes: numbers of directed and undirected edges in the estimated partially directed graphs, and edges in their DAG representatives. Entries report means $\pm$ sample standard deviations over ten matched runs; subscripts indicate fewer than ten defined measurements. Reference rows give the true CPDAG and DAG sizes.", methods=("Reference", *METHODS))
    rows = []
    for dataset in DATASETS:
        vals = []
        for metric in ("nsid_low", "nsid_high", "adjacency_F1", "arrowhead_F1", "nshd"):
            row = tests[(tests.dataset == dataset) & (tests.kind == "cpdag") & (tests.reference == "ABAPC (bb)") & (tests.metric == metric)].iloc[0]
            vals.append(cell(row.delta_mean, digits=3, signed=True, count=int(row.n) if 0<row.n<10 else None,
                             applicable=(dataset, metric) not in noninformative))
        time = tests[(tests.dataset == dataset) & (tests.kind == "dag") & (tests.reference == "ABAPC (bb)") & (tests.metric == "elapsed")].iloc[0]
        vals.append(rf"$\times {time.left_mean/time.right_mean:.3g}$")
        rows.append([dataset.title(), *vals])
    tables.write("table_aij_main_delta", ["Dataset", r"$\Delta$NSID$_{\min}\downarrow$", r"$\Delta$NSID$_{\max}\downarrow$", r"$\Delta$Sk-F1$\uparrow$", r"$\Delta$AH-F1$\uparrow$", r"$\Delta$NSHD$\downarrow$", r"$t$ ratio$\downarrow$"], rows,
                 r"Mean within-seed differences, ABAPC (bb-nor) minus ABAPC (bb), over ten matched runs with 5000 observations. Distances are normalised by the number of true DAG edges; negative distance differences and positive F1 differences favour bb-nor. The runtime ratio is bb-nor divided by bb. Subscripts indicate fewer than ten defined pairs; n/a marks AH comparisons with no compelled arrowheads in the true CPDAG, and -- denotes an unavailable value.")


def make_legacy_tables(tables: Tables, legacy: pd.DataFrame, semantics: pd.DataFrame) -> None:
    order = (*METHODS[:4], "ASPforABA", "ABAPC (orig)", *VARIANTS)
    for start in (0, 3):
        rows = []
        for dataset in DATASETS[start:start+3]:
            first = True
            for method in order:
                found = legacy[(legacy.dataset == dataset) & (legacy.method == method)]
                if found.empty:
                    continue
                r = found.iloc[0]
                time_digits = 6 if 0 < abs(r.elapsed_mean) < 0.001 else 3
                rows.append([dataset.title() if first else "", tex_escape(method), r.cohort,
                             cell(r.elapsed_mean, r.elapsed_std, time_digits)])
                first = False
            if dataset != DATASETS[start+2]:
                rows.append(None)
        tables.write(f"table_aij_runtime_{start//3+1}", ["Dataset", "Method", "Cohort / samples / runs", "Time (s)"], rows,
                     r"Execution times in seconds, reported as means $\pm$ sample standard deviations. The cohort column gives the experiment collection, sample size and run count; -- denotes an unavailable run count. Comparisons across cohorts are descriptive.")
    rows = []
    for dataset in DATASETS[:3]:
        for i, method in enumerate(("ASPforABA", "ABAPC (orig)", *VARIANTS)):
            r = legacy[(legacy.dataset == dataset) & (legacy.method == method)].iloc[0]
            rows.append([dataset.title() if i == 0 else "", method, r.cohort, cell(r.nsid_mean, r.nsid_std, 2)])
        if dataset != DATASETS[2]:
            rows.append(None)
    tables.write("table_aij_variants", ["Dataset", "Implementation", "Cohort / samples / runs", r"DAG NSID $\downarrow$"], rows,
                 r"DAG reconstruction across implementations and encoding variants. NSID is SID divided by the number of true DAG edges. Entries report means $\pm$ sample standard deviations. The cohort column gives the experiment collection, sample size and run count; -- denotes an unavailable run count. Comparisons across cohorts are descriptive.")
    order = (*METHODS[:4], "Causal ABA (Original)", "Causal ABA (Alt. ST)", "Causal ABA (Alt. CO)", "Causal ABA (Alt. CO-max)")
    specs = [("nsid_low", r"NSID$_{L}\downarrow$", 3, "min"), ("nsid_high", r"NSID$_{U}\downarrow$", 3, "min")]
    metric_panels(tables, semantics, "table_aij_semantics", specs,
                  r"Comparison of argumentation semantics over 50 runs. Entries report means $\pm$ standard deviations of lower and upper SID bounds, normalised by the number of true DAG edges. These historical bounds need not be attained by a compatible DAG.", methods=order, datasets=DATASETS[:3])


def make_fact_tables(tables: Tables, src: Sources, output: Path) -> None:
    base = "results/analysis/matched10_alpha_fact_compare_"
    method = src.csv(base+"method_seedwise.csv")
    validate_seeds(method, ["dataset", "method", "alpha"])
    # Different thresholds are paired within each method, not pooled.
    delta_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = method[(method.dataset == dataset) & (method.method == variant)]
            old, new = group[group.alpha == "alpha005"], group[group.alpha == "alpha001"]
            for metric in ("dag_F1", "dag_SHD"):
                delta_rows.append(dict(dataset=dataset, method=variant, metric=metric, **paired_effect(new, old, metric)))
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output/"alpha_paired_effects.csv", index=False)
    rows = []
    for dataset in DATASETS:
        for i, variant in enumerate(VARIANTS):
            sub = deltas[(deltas.dataset == dataset) & (deltas.method == variant)].set_index("metric")
            rows.append([dataset.title() if i == 0 else "", variant,
                         cell(sub.loc["dag_F1", "delta_mean"], sub.loc["dag_F1", "delta_std"], signed=True,
                              count=int(sub.loc["dag_F1", "n"]) if 0<sub.loc["dag_F1", "n"]<10 else None),
                         cell(sub.loc["dag_SHD", "delta_mean"], sub.loc["dag_SHD", "delta_std"], 2, signed=True,
                              count=int(sub.loc["dag_SHD", "n"]) if 0<sub.loc["dag_SHD", "n"]<10 else None)])
        if dataset != DATASETS[-1]:
            rows.append(None)
    tables.write("table_aij_alpha_methods", ["Dataset", "Method", r"$\Delta$DAG F1 $\uparrow$", r"$\Delta$DAG SHD $\downarrow$"], rows,
                 r"Effect of the CI-test significance threshold: within-seed differences, $\alpha=0.01$ minus $\alpha=0.05$, reported as means $\pm$ sample standard deviations over ten matched runs. Positive F1 and negative SHD differences favour $\alpha=0.01$. SHD is unnormalised; subscripts indicate fewer than ten defined pairs.")


def ranking_table(tables: Tables, src: Sources, csv_path: str | None) -> bool:
    if csv_path is None:
        return False
    df = src.csv(csv_path, "Original ranking ablation measurements supplied separately")
    required = {"dataset", "method", "seed", "AP", "NDCG"}
    if not required.issubset(df):
        raise ValueError(f"Ranking CSV must contain {sorted(required)}")
    if df.duplicated(["dataset", "method", "seed"]).any():
        raise ValueError("Duplicate ranking measurements")
    if not np.isfinite(df[["AP", "NDCG"]].to_numpy()).all():
        raise ValueError("Missing ranking metric")
    if not df.groupby(["dataset", "method"]).size().eq(50).all():
        raise ValueError("The ranking figure requires 50 runs per dataset/method")
    metric_panels(tables, aggregate(df, ["dataset", "method"], ["AP", "NDCG"]), "table_aij_ranking",
                  [("AP", "AP $\\uparrow$", 3, "max"), ("NDCG", "NDCG $\\uparrow$", 3, "max")],
                  r"Ranking ablation: mean $\pm$ sample SD over 50 runs. (P) denotes the conditioning-set penalty.",
                  methods=tuple(df.method.unique()), datasets=tuple(d for d in DATASETS if d in set(df.dataset)))
    return True


def write_preview(output: Path, names: list[str]) -> None:
    preview = [r"\documentclass[preprint,12pt]{elsarticle}", r"\usepackage{booktabs,amsmath,amssymb,times,graphicx}",
               r"\usepackage[small]{caption}", r"\usepackage[hidelinks]{hyperref}", r"\begin{document}",
               r"\begin{center}\Large AIJ results: generated table review\end{center}",
               r"Tables are generated from saved experimental records. Mean $\pm$ SD is distinguished from paired differences and historical summaries. Source hashes and experiment selections are recorded in \texttt{table\_build\_manifest.json}.",
               r"\clearpage"]
    for name in names:
        preview.extend([rf"\input{{{name}}}", r"\clearpage"])
    preview.append(r"\end{document}")
    (output/"preview.tex").write_text("\n".join(preview)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results/tables/aij")
    parser.add_argument("--paper-dir", type=Path, help="Copy generated snapshots into PAPER/tables/aij; do not modify the manuscript")
    parser.add_argument("--ranking-csv", help="Optional repository-relative original AP/NDCG per-run CSV (see plan)")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    src = Sources(ROOT)
    validate_launches(src)
    primary = load_primary(src)
    sizes = load_sizes(src, primary)
    noninformative = noninformative_arrowhead_scores(sizes)
    tests = make_tests(primary, noninformative)
    semantics = load_semantics(src)
    legacy = legacy_summaries(src, primary)
    tables = Tables(output)
    make_primary_tables(tables, primary, tests, sizes)
    make_legacy_tables(tables, legacy, semantics)
    make_fact_tables(tables, src, output)
    have_ranking = ranking_table(tables, src, args.ranking_csv)
    for name, frame in (("primary_records", primary), ("paired_tests", tests), ("graph_size_records", sizes),
                        ("legacy_summaries", legacy), ("semantics_summaries", semantics)):
        frame.to_csv(output/f"{name}.csv", index=False, lineterminator="\n")
    metrics = ["elapsed", "nnz", "precision", "recall", "F1", "adjacency_precision", "adjacency_recall", "adjacency_F1",
               "arrowhead_precision", "arrowhead_recall", "arrowhead_F1", "nshd", "nsid", "nsid_low", "nsid_high"]
    summary = aggregate(primary, ["dataset", "method", "kind"], metrics)
    summary.to_csv(output/"primary_summary.csv", index=False)
    summary[["dataset", "method", "kind", *[c for c in summary if c.endswith("_count")]]].to_csv(output/"metric_coverage.csv", index=False)
    write_preview(output, tables.names)
    for name in ("scripts/build_aij_tables.py", "scripts/plot_bnlearn_matched10_compare.py", "scripts/generate_bnlearn_matched10_ttest_tables.py",
                 "scripts/analyze_bnlearn_matched10_alpha_facts.py", "utils/experiment_support.py", "scripts/README_AIJ.md"):
        src.use(name, "Generator, experiment-selection evidence or source-map documentation")
    generated_names = set(tables.names) | {"preview.tex", "primary_records.csv", "paired_tests.csv", "graph_size_records.csv", "legacy_summaries.csv",
        "semantics_summaries.csv", "primary_summary.csv", "metric_coverage.csv", "alpha_paired_effects.csv"}
    manifest = dict(schema_version=1, generator="scripts/build_aij_tables.py", git_commit=git("rev-parse", "HEAD"),
                    git_branch=git("branch", "--show-current"), generator_sha256=sha(Path(__file__).read_bytes()),
                    generator_tracked_at_commit=subprocess.run(["git", "cat-file", "-e", "HEAD:scripts/build_aij_tables.py"], cwd=ROOT, capture_output=True).returncode == 0,
                    dependencies={"python":sys.version.split()[0], "numpy":np.__version__, "pandas":pd.__version__},
                    datasets=list(DATASETS), seeds=list(SEEDS), sample_size=5000, ci_test="gsq", alpha=0.05,
                    standard_deviation="sample, ddof=1", normalisation="SID and SHD divided by true DAG edge count; exact zeros preserved",
                    display_not_applicable=[{"dataset":d,"kind":"cpdag","metric":m,"reason":"No reference arrowheads; non-informative recovery comparison"} for d,m in sorted(noninformative)],
                    display_legend={"n/a":"Not applicable/non-informative for the reference graph", "--":"No available numerical value (missing, undefined in archived computation, or unverified); see table caption"},
                    evaluation_protocol="aij_v2: complete CPDAG conversion, explicit endpoints, exact edge-type F1, exact SID extrema with DAG witnesses",
                    evaluation_caveats=["Five Survey MPC outputs have no consistent extension: retain edge scores, omit SID and DAG scores", "FGS historical DOT export lacks original endpoint/node-label provenance; native comparison requires rerunning the corrected exporter", "Historical supplementary cohorts remain separately labelled"],
                    tests="two-sided paired t-test on finite matched pairs, omitted seeds recorded; Holm across five reference methods within each dataset/metric; separate from legacy unpaired t-tests",
                    unresolved=[] if have_ranking else ["ranking.png: original 50-run AP/NDCG data and generating script not found; no replacement numbers invented"],
                    historical_caveats=["Original-encoding plot fallbacks mix 2000- and 5000-sample experiments", "ASPforABA uses a historical 50-run summary", "CO-max semantics aggregates recovered from saved notebook output; raw CO_MAX field missing"],
                    inputs=src.files, tables=tables.names,
                    outputs={name:sha((output/name).read_bytes()) for name in sorted(generated_names)})
    (output/"table_build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    commit=manifest['git_commit']
    base=f"https://github.com/briziorusso/ArgCausalDisco/blob/{commit}"
    links=["# AIJ table provenance", "", f"Code branch: `AIJ`; exact revision: [{commit}]({base.replace('/blob/','/commit/')}).", "",
           f"[Build and server instructions]({base}/scripts/README_AIJ.md).", "",
           f"Generator: [build_aij_tables.py]({base}/scripts/build_aij_tables.py).",
           f"Evaluation: [reevaluate_aij_graphs.py]({base}/scripts/reevaluate_aij_graphs.py) and [graph metrics]({base}/utils/aij_graph_metrics.py).", "",
           "The JSON manifest records input and output SHA-256 hashes. Generated CSVs and evaluation caches are reproduced in the code checkout; they are not duplicated in the paper repository.", "",
           "| Table | LaTeX input |", "| --- | --- |"]
    links += [f"| {i} | [{name}]({name}) |" for i,name in enumerate(tables.names,1)]
    links += ["", "Tables 15--18 from the initial review have been retired. The retained set contains 14 tables. Historical cohorts remain labelled; matched ASPCR-DAG and discrete-score FGS reruns are not substituted before validation.", "",
              "Include a table with `\\input{tables/aij/<filename>.tex}`; compile `preview.tex` to review the collection."]
    (output/'SOURCES.md').write_text('\n'.join(links)+'\n',encoding='utf-8')
    if args.paper_dir:
        destination=args.paper_dir.resolve()/'tables/aij'
        destination.mkdir(parents=True,exist_ok=True)
        for name in set(tables.names)|{'preview.tex','table_build_manifest.json','SOURCES.md'}:
            shutil.copy2(output/name,destination/name)
    print(f"Wrote {len(tables.names)} tables to {output}; {len(primary)} primary records; {len(tests)} paired tests.")
    for issue in manifest["unresolved"]:
        print("UNRESOLVED:", issue)


if __name__ == "__main__":
    main()
