# %% [markdown]
# ## Experimental Setup

# %%

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyagrum as gum

from abapc import ABAPC
from utils.helpers import random_stability
from priors.generate_priors import PriorKnowledge, Constraints


def load_dataset(
    bif_path: Path, sample_size: int, seed: int
) -> tuple[gum.BayesNet, np.ndarray]:
    bn = gum.loadBN(str(bif_path))
    bn.name = bif_path.stem

    gum.initRandom(seed=seed)
    df = gum.generateSample(bn, sample_size, with_labels=False, random_order=False)[0]
    sorted_vars = sorted(df.columns)
    data = df[sorted_vars].to_numpy().astype(float)

    return bn, data

def extract_facts(string: str) -> list[dict]:
    pattern = re.compile(r"ext_((in)?dep)\((.+)\). I=([01].\d+), NA\n")
    matches = pattern.findall(string)
    facts = []
    for match in matches:
        cit_type, _, triple, score = match
        X, Y, S = triple.split(",")
        facts.append(
            {
                "cit_type": cit_type,
                "X": int(X),
                "Y": int(Y),
                "S": set() if S == "empty" else {int(var) for var in S[1:].split("y")},
                "score": float(score),
            }
        )
    return pd.DataFrame(facts).sort_values(
        by="score", ascending=False, ignore_index=True
    )

# %% [markdown]
# ### Define Causal ABA Interface to call both implementations

# %%
from clingo import Function, Number

from utils.graph_utils import extract_test_elements_from_symbol, set_of_models_to_set_of_graphs


def CausalABA(
    n_nodes: int,
    compile_and_ground: Callable,
    facts_location: str = "",
    print_models: bool = True,
    skeleton_rules_reduction: bool = False,
    weak_constraints: bool = False,
    opt_mode: str = "optN",
    show: list = ["arrow"],
    pre_grounding: bool = False,
    disable_reground: bool = False,
    prior_knowledge: PriorKnowledge | None = None,
) -> list:
    """
    CausalABA, a function that takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.

    """
    # (X, Y) -> their condition sets S
    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts = []
    ext_flag = False
    if facts_location:
        facts_loc = (
            facts_location.replace(".lp", "_I.lp")
            if weak_constraints
            else facts_location
        )
        logging.debug(f"   Loading facts from {facts_location}")
        with open(facts_loc, "r") as file:
            for line in file:
                if "dep" not in line or line.startswith("%"):
                    continue
                line_clean = line.replace("#external ", "").replace("\n", "")
                if "ext_" in line_clean:
                    ext_flag = True
                if weak_constraints:
                    statement, Is = line_clean.split(" I=")
                    I, truth = Is.split(",")
                    X, S, Y, dep_type = extract_test_elements_from_symbol(statement)
                    facts.append((X, S, Y, dep_type, statement, float(I), truth))
                else:
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line_clean)
                    facts.append((X, S, Y, dep_type, line_clean, np.nan, "unknown"))

                assert (X not in S) and (Y not in S), f"X or Y in S: {line_clean}"
                condition_set = tuple(S)

                facts_group = indep_facts if "indep" in line_clean else dep_facts
                if (X, Y) not in facts_group:
                    facts_group[(X, Y)] = set()
                assert condition_set not in facts_group[(X, Y)], (
                    f"Redundant external fact: {line_clean}"
                )
                facts_group[(X, Y)].add(condition_set)

    ctl = compile_and_ground(
        n_nodes,
        facts_location,
        skeleton_rules_reduction,
        weak_constraints,
        indep_facts,
        dep_facts,
        opt_mode,
        show,
        pre_grounding,
        ext_flag,
        prior_knowledge,
    )

    facts = sorted(facts, key=lambda x: x[5], reverse=True)
    for fact in facts:
        ctl.assign_external(
            Function(
                fact[3],
                [
                    Number(fact[0]),
                    Number(fact[2]),
                    Function(fact[4].replace(").", "").split(",")[-1]),
                ],
            ),
            True,
        )
        logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
    models = []
    logging.info("   Solving...")
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(shown=True))
            if print_models:
                logging.info(f"Answer {len(models)}: {model}")
    n_models = int(ctl.statistics["summary"]["models"]["enumerated"])
    logging.info(f"Number of models: {n_models}")
    times = {
        key: ctl.statistics["summary"]["times"][key]
        for key in ["total", "cpu", "solve"]
    }
    logging.info(f"Times: {times}")
    remove_n = 0
    logging.info(f"Number of facts removed: {remove_n}")

    ## start removing facts if no models are found
    while n_models == 0 and remove_n < len(facts):
        remove_n += 1
        logging.info(f"Number of facts removed: {remove_n}")

        reground = False
        fact_to_remove = facts[-remove_n]
        X, S, Y, dep_type, fact_str = fact_to_remove[:5]
        logging.debug(f"Removing fact {fact_str}")

        facts_group = indep_facts if dep_type == "ext_indep" else dep_facts
        facts_group[(X, Y)].remove(tuple(S))
        if not facts_group[(X, Y)]:
            del facts_group[(X, Y)]
            reground = (
                disable_reground is False
                and skeleton_rules_reduction
                and (ext_flag is False or dep_type == "ext_indep")
            )
        else:
            logging.debug(
                f"   Not removing fact {fact_str} because there are multiple facts with the same X and Y"
            )
        ctl.assign_external(
            Function(
                dep_type,
                [
                    Number(X),
                    Number(Y),
                    Function(fact_str.replace(").", "").split(",")[-1]),
                ],
            ),
            None,
        )

        if reground:
            ### Save external statements
            logging.info("Recompiling and regrounding...")
            ctl = compile_and_ground(
                n_nodes,
                facts_location,
                skeleton_rules_reduction,
                weak_constraints,
                indep_facts,
                dep_facts,
                opt_mode,
                show,
                pre_grounding=pre_grounding,
                ext_flag=ext_flag,
                prior_knowledge=prior_knowledge,
            )
            for fact in facts[:-remove_n]:
                ctl.assign_external(
                    Function(
                        fact[3],
                        [
                            Number(fact[0]),
                            Number(fact[2]),
                            Function(fact[4].replace(").", "").split(",")[-1]),
                        ],
                    ),
                    True,
                )
                logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            for fact in facts[-remove_n:]:
                ctl.assign_external(
                    Function(
                        fact[3],
                        [
                            Number(fact[0]),
                            Number(fact[2]),
                            Function(fact[4].replace(").", "").split(",")[-1]),
                        ],
                    ),
                    None,
                )
                logging.debug(f"   False fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
        models = []
        logging.info("   Solving...")
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if model.optimality_proven:
                    models.append(model.symbols(shown=True))
                if print_models:
                    logging.info(f"Answer {len(models)}: {model}")
        n_models = int(ctl.statistics["summary"]["models"]["enumerated"])
        logging.info(f"Number of models: {n_models}")
        times = {
            key: ctl.statistics["summary"]["times"][key]
            for key in ["total", "cpu", "solve"]
        }
        logging.info(f"Times: {times}")

    models, _ = set_of_models_to_set_of_graphs(models, n_nodes, False)
    return {
        "remove_n": remove_n,
        "statistics": ctl.statistics,
        "models": models,
    }


# %% [markdown]
# ### Define the evaluation function to compare performance of two different implementations

# %%
import timeit
from typing import Any

from tqdm import tqdm
from sklearn.metrics import average_precision_score

import abapc
import causalaba
from causalaba import compile_and_ground

abapc.tqdm = tqdm  # Avoid using progress bar widgets
causalaba.tqdm = tqdm  # Avoid using progress bar widgets

def retained_facts_score(df_ranked: pd.DataFrame, remove_n: int) -> dict[str, float]:
    df_retained = df_ranked.iloc[:-remove_n]
    precision = df_retained["correct"].sum() / len(df_retained)
    recall = df_retained["correct"].sum() / df_ranked["correct"].sum()
    f1 = 2 * precision * recall / (precision + recall)
    return {"CIT_Precision": precision, "CIT_Recall": recall, "CIT_F1": f1}


def compare_performance(
    bif_path: str | Path,
    sample_size: int,
    prior_df: pd.DataFrame | None,
    repeats: int = 1,
) -> tuple[str, int, float, float, list, list, np.ndarray, Any]:
    random_stability(2024)
    seeds = np.random.randint(0, 10000, size=repeats).tolist()

    if isinstance(bif_path, str):
        bif_path = Path(bif_path)

    impl1_return = []
    impl2_return = []
    impl1 = with_optional_prior(prior_df=prior_df)
    impl2 = with_optional_prior(prior_df=None)

    for seed in seeds:
        bn_true, data = load_dataset(bif_path, sample_size=sample_size, seed=seed)
        facts_I_path, _ = ABAPC(
            data,
            seed=seed,
            alpha=0.01,
            indep_test="gsq",
            S_weight=False,
            out_mode="facts_only",
            scenario="prior_comparison",
        )
        facts_path = facts_I_path.replace("_I.lp", ".lp")
        
        sorted_vars = sorted(bn_true.names())
        df = extract_facts(Path(facts_I_path).read_text())
        df["X"] = df["X"].apply(lambda x: sorted_vars[x])
        df["Y"] = df["Y"].apply(lambda y: sorted_vars[y])
        df["S"] = df["S"].apply(lambda s: {sorted_vars[i] for i in s})
        df["correct"] = df.apply(
            lambda row: (
                row["cit_type"] == "indep"
                if bn_true.isIndependent(row["X"], row["Y"], row["S"])
                else (row["cit_type"] == "dep")
            ),
            axis=1,
        )
        average_precision = average_precision_score(
            df["correct"], df["score"]
        )
        df_ranked = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        timer = timeit.timeit(impl1(bn_true, facts_path, impl1_return), number=1)
        impl1_return[-1]["time"] = timer
        impl1_return[-1]["seed"] = seed
        impl1_return[-1]["AP"] = average_precision
        impl1_return[-1].update(retained_facts_score(df_ranked, remove_n=impl1_return[-1]["remove_n"]))

        timer = timeit.timeit(impl2(bn_true, facts_path, impl2_return), number=1)
        impl2_return[-1]["time"] = timer
        impl2_return[-1]["seed"] = seed
        impl2_return[-1]["AP"] = average_precision
        impl2_return[-1].update(retained_facts_score(df_ranked, remove_n=impl2_return[-1]["remove_n"]))

    return bif_path.stem, impl1_return, impl2_return

# %% [markdown]
# ## Compare Causal ABA performance with/without prior knowledge

# %%
def with_optional_prior(prior_df: pd.DataFrame | None = None):
    def impl_wrapper(bn: gum.BayesNet, facts_path: str, return_ref: list) -> Callable:
        prior_knowledge = None
        if prior_df is not None:
            filename = bn.property("name")
            prior_knowledge = PriorKnowledge(
                variables=bn.names(),
                constraints=Constraints(
                    forbidden=prior_df[prior_df["filename"] == filename]["priors"].iloc[
                        0
                    ],
                ),
            )

        def helper():
            return_ref.append(
                CausalABA(
                    n_nodes=bn.size(),
                    compile_and_ground=compile_and_ground,
                    facts_location=facts_path,
                    print_models=False,
                    skeleton_rules_reduction=True,
                    weak_constraints=True,
                    opt_mode="optN",
                    pre_grounding=True,  # Speed up solving by pre-grounding
                    prior_knowledge=prior_knowledge,
                    disable_reground=True,
                )
            )

        return helper

    return impl_wrapper

# %%
from typing import Iterable

import abapc
from utils.graph_utils import DAGMetrics, dag2cpdag


repeats = 10
MAX_NODES = 15  # Skip graphs larger than this for performance reasons
report_cols = [
    "time",
    "remove_n",
    "cpdag_F1",
    "cpdag_shd",
    "cpdag_sid_low",
    "cpdag_sid_high",
    "AP",
    "CIT_Precision",
    "CIT_F1",
]
report_stats = {col: ["mean", "std"] for col in report_cols}
Path("logs").mkdir(exist_ok=True)


def show_report(df: pd.DataFrame):
    groups = df.groupby(["dataset", "impl"])
    stats = groups.agg(report_stats)
    meta = groups.agg({"num_nodes": "first", "num_edges": "first"})
    meta["repeats"] = groups.size()
    meta.columns = pd.MultiIndex.from_product([["meta"], meta.columns])
    index = (
        groups["num_nodes"].first().reset_index().sort_values(["num_nodes", "dataset", "impl"]).set_index(["dataset", "impl"]).index
    )
    return pd.concat([meta, stats], axis=1).loc[index]


def get_sorted_adjacency(bn: gum.BayesNet) -> np.ndarray:
    """Get the adjacency matrix of the Bayesian network sorted by variable names."""
    alphabetical_order = [bn.idFromName(var) for var in sorted(bn.names())]
    adj = bn.adjacencyMatrix()
    return adj[np.ix_(alphabetical_order, alphabetical_order)]

def run_experiment(prior_df: pd.DataFrame | None, datasets: Iterable[Path]):
    all_runs = []
    skipped = []

    for random_graph_path in datasets:
        filename = random_graph_path.stem
        if filename not in prior_df["filename"].values or not prior_df[prior_df["filename"] == filename]["priors"].iloc[0]:
            skipped.append(filename)
            continue

        bn = gum.loadBN(str(random_graph_path))
        B_true = get_sorted_adjacency(bn)
        if bn.size() > MAX_NODES:
            continue

        print(f"Processing {random_graph_path.name}...")
        (
            bif_name,
            impl1_return,
            impl2_return,
        ) = compare_performance(
            bif_path=random_graph_path,
            sample_size=5000,
            prior_df=prior_df,
            repeats=repeats,
        )

        # Flatten results; each 'run' already carries its seed and run_id
        for impl_name, results in [("org", impl2_return), ("new", impl1_return)]:
            for run in results:
                models = run["models"]
                cd_metrics = []
                for model in models:
                    B_est = np.zeros((bn.size(), bn.size()))
                    for edge in model:
                        B_est[edge[0], edge[1]] = 1
                    model = B_est
                    # DAG metrics
                    B_est_dag = (model > 0).astype(int)
                    mt_dag = DAGMetrics(B_est_dag, B_true).metrics

                    # CPDAG metrics
                    B_est_cpdag = (model != 0).astype(int)
                    mt_cpdag = DAGMetrics(dag2cpdag(B_est_cpdag), B_true).metrics
                    cpdag_sid = mt_cpdag.pop("sid")
                    if not isinstance(cpdag_sid, tuple):
                        cpdag_sid = (cpdag_sid, cpdag_sid)
                    mt_cpdag["sid_low"], mt_cpdag["sid_high"] = cpdag_sid
                    cd_metrics.append({
                        **{f"dag_{k}": v for k, v in mt_dag.items()},
                        **{f"cpdag_{k}": v for k, v in mt_cpdag.items()},
                    })
                cd_metrics_df = pd.DataFrame(cd_metrics)
                if len(cd_metrics_df) > 10:
                    cd_metrics_df.to_csv("logs/example.csv", index=False)

                # Save this run's results in a flat dict
                all_runs.append(
                    {
                        "dataset": bif_name,
                        "num_nodes": bn.size(),
                        "num_edges": bn.sizeArcs(),
                        "impl": impl_name,
                        "seed": run["seed"],
                        "time": run["time"],
                        "remove_n": run["remove_n"],
                        # **{f"dag_{k}": v for k, v in mt_dag.items()},
                        # **{f"cpdag_{k}": v for k, v in mt_cpdag.items()},
                        **cd_metrics_df.mean().to_dict(),
                        "AP": run["AP"],
                        "CIT_Precision": run["CIT_Precision"],
                        "CIT_Recall": run["CIT_Recall"],
                        "CIT_F1": run["CIT_F1"],
                    }
                )

    return pd.DataFrame(all_runs), skipped

# # %% [markdown]
# # ### Bnlearn small datasets

# # %%
# bnlearn_small_datasets = list(Path("priors/datasets/bnlearn/").glob("*.bifxml"))
# print("bnlearn_small_datasets:", bnlearn_small_datasets)
# bnlearn_prior_df = pd.read_json("results/2025/llm/bnlearn_with_desc.json")
# bnlearn_prior_df

# # %%
# res_df_bnlearn = run_experiment(
#     prior_df=bnlearn_prior_df, datasets=bnlearn_small_datasets
# )
# res_df_bnlearn.to_csv(
#     "results/2025/prior_integration_bnlearn.csv-approx", index=False
# )

# # %%
# bnlearn_report = show_report(res_df_bnlearn)

# bnlearn_report.to_csv("results/2025/prior_integration_bnlearn_report-approx.csv")


# %% [markdown]
# ### Synthetic datasets

# %%
# synthetic_datasets = list(Path("priors/datasets/random_graphs/heuristic_by_semantic/").glob("*.bifxml"))
synthetic_datasets = list(Path("priors/datasets/causenet_synth_5_10_15/").glob("*.bifxml"))
print("synthetic_datasets:", synthetic_datasets)
synthetic_prior_df = pd.read_json("results/2025/llm/causenet_synth_5_10_15_with_desc-agg.json")
synthetic_prior_df

# %%
synthetic_res_df, skipped = run_experiment(
    prior_df=synthetic_prior_df, datasets=synthetic_datasets
)
synthetic_res_df.to_csv(
    "results/2025/agg.csv", index=False
)


# %%
synthetic_report = show_report(synthetic_res_df)

synthetic_report.to_csv("results/2025/agg-report.csv")

print("Skipped files:", skipped)
