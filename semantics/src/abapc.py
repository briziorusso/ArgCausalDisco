from itertools import combinations
from tqdm import tqdm
import networkx as nx
import time
import pandas as pd
import numpy as np

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.cd_algorithms.PC import pc
from ArgCausalDisco.utils.graph_utils import initial_strength
from ArgCausalDisco.utils.graph_utils import set_of_models_to_set_of_graphs
from ArgCausalDisco.causalaba import CausalABA

from src.utils.fact_utils import get_fact_location
from src.utils.utils import get_arrows_from_model, get_matrix_from_arrow_set
from src.utils.enums import Fact, RelationEnum
from src.causal_aba.factory import ABASPSolverFactory
from src.utils.enums import SemanticEnum
import src.causal_aba.assumptions as assums
from dataclasses import dataclass

from logger import logger


def get_arrow_sets_from_facts(facts, n_nodes, semantics=SemanticEnum.ST):
    # deterministically sort the facts. 
    # Even if same strength, still order is defined uniquely
    sorted_facts = sorted(facts, 
                          key=lambda x: (x.score, 
                                         x.node1, 
                                         x.node2, 
                                         str(sorted(list(x.node_set)))), 
                          reverse=True)

    # remove facts staring from weakest

    factory = ABASPSolverFactory(n_nodes=n_nodes)
    fact_idx = len(sorted_facts)

    while fact_idx >= 0:
        solver = factory.create_solver(sorted_facts[:fact_idx])
        models = solver.enumerate_extensions(semantics.value, k=50000)  # Limit the number of models to 50,000
        only_empty_model = (models is not None
                            and len(models) == 1
                            and len(models[0].assumptions) == 0)
        break_condition = (models is not None
                           and len(models) > 0
                           and not only_empty_model
                           )
        if break_condition:
            break
        fact_idx -= 1
        logger.info(f"Trying with top {fact_idx} facts")
    arrow_sets = [get_arrows_from_model(model) for model in models]
    return arrow_sets, fact_idx


def get_cg_and_facts(data,
                     alpha=0.01,
                     indep_test='fisherz',
                     uc_rule=5,
                     stable=True):
    n_nodes = data.shape[1]
    cg = pc(data=data, alpha=alpha, indep_test=indep_test, uc_rule=uc_rule,
            stable=stable, show_progress=False, verbose=False)
    facts = []

    for node1, node2 in combinations(range(n_nodes), 2):
        test_PC = [t for t in cg.sepset[node1, node2]]
        for sep_set, p in test_PC:
            dep_type_PC = "indep" if p > alpha else "dep"
            init_strength_value = initial_strength(p, len(sep_set), alpha, 0.5, n_nodes)

            fact = Fact(
                relation=RelationEnum(dep_type_PC),
                node1=node1,
                node2=node2,
                node_set=set(sep_set),
                score=init_strength_value
            )

            if fact not in facts:
                facts.append(fact)
    return cg, facts


def get_arrow_sets(data,
                   seed=42,
                   alpha=0.01,
                   indep_test='fisherz',
                   uc_rule=5,
                   stable=True,
                   semantics=SemanticEnum.ST):
    """
    Get the stable models from the ABAPC algorithm
    Args:
        X_s: np.array
            The dataset to be used for the ABAPC algorithm
        seed: int
            The seed to be used for the random number generator
    Returns:
        models: list
            The stable models from the ABAPC algorithm
            in a form of arrow sets.
    """
    random_stability(seed)
    n_nodes = data.shape[1]
    cg, facts = get_cg_and_facts(data,
                                 alpha=alpha,
                                 indep_test=indep_test,
                                 uc_rule=uc_rule,
                                 stable=stable)
    arrow_sets, num_facts = get_arrow_sets_from_facts(facts, n_nodes, semantics=semantics)

    return arrow_sets, cg, num_facts, facts


def score_model_original(model, n_nodes, cg, alpha=0.01, return_only_I=False):
    B_est = get_matrix_from_arrow_set(model, n_nodes)
    G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
    est_I = 0
    for x, y in combinations(range(n_nodes), 2):
        I_from_data = list(set(cg.sepset[x, y]))
        for s, p in I_from_data:
            PC_dep_type = 'indep' if p > alpha else 'dep'
            s_text = [f"X{r+1}" for r in s]
            dep_type = 'indep' if nx.algorithms.d_separated(G_est, {f"X{x+1}"}, {f"X{y+1}"}, set(s_text)) else 'dep'
            I = initial_strength(p, len(s), alpha, 0.5, n_nodes)
            if dep_type != PC_dep_type:
                est_I += -I
            else:
                est_I += I
    if return_only_I:
        return est_I
        
    return est_I, B_est

def get_best_model(models, n_nodes, cg, alpha=0.01):
    if len(models) > 50000:
        logger.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000])  # Limit the number of models to 30,000

    best_model = None
    best_I = None
    best_B_est = None
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        # derive B_est from the model
        est_I, B_est = score_model_original(model, n_nodes, cg, alpha=alpha)
        if best_model is None or best_I < est_I:
            best_model = model
            best_I = est_I
            best_B_est = B_est
        logger.info(f"DAG from d-ABA: {best_B_est}")
    return best_model, best_B_est, best_I


def score_model_by_refined_indep_facts(model, indep_to_strength, n_nodes, cg, alpha=0.01, return_only_I=False):
    # derive B_est from the model
        B_est = get_matrix_from_arrow_set(model, n_nodes)
        G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
        est_I = 0
        for x, y in combinations(range(n_nodes), 2):
            I_from_data = list(set(cg.sepset[x, y]))
            for s, p in I_from_data:
                PC_dep_type_is_indep = p > alpha
                if PC_dep_type_is_indep:
                    s_text = [f"X{r+1}" for r in s]
                    is_d_separated = nx.algorithms.d_separated(G_est, {f"X{x+1}"}, {f"X{y+1}"}, set(s_text))
                    if is_d_separated:
                        I = indep_to_strength.get(assums.indep(x, y, s), 0)
                        est_I += I
        if return_only_I:
            return est_I
        return est_I, B_est

def get_best_model_by_refined_indep_facts(models, indep_to_strength, n_nodes, cg, alpha=0.01):
    if len(models) > 50000:
        logger.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000])  # Limit the number of models to 30,000

    best_model = None
    best_I = None
    best_B_est = None
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        # derive B_est from the model
        est_I, B_est = score_model_by_refined_indep_facts(model, indep_to_strength, n_nodes, cg, alpha=alpha)

        if best_model is None or best_I < est_I:
            best_model = model
            best_I = est_I
            best_B_est = B_est
        logger.info(f"DAG from d-ABA: {best_B_est}")
    return best_model, best_B_est, best_I


def get_best_model_by_arrows_sum(models, arr_strength, n_nodes):
    if len(models) > 50000:
        logger.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000])  # Limit the number of models to 30,000

    best_model = None
    best_I = None
    best_B_est = None
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        est_I = 0
        for x, y in model:
            est_I += arr_strength.get(assums.arr(x, y), 0)

        if best_model is None or best_I < est_I:
            best_model = model
            best_I = est_I
        logger.info(f"DAG from d-ABA: {best_B_est}")
    best_B_est = get_matrix_from_arrow_set(best_model, n_nodes)
    return best_model, best_B_est, best_I


def get_best_model_by_arrows_mean(models, arr_strength, n_nodes):
    if len(models) > 50000:
        logger.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000])  # Limit the number of models to 30,000

    best_model = None
    best_I = None
    best_B_est = None
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        est_I = 0
        num_arrows = 0
        for x, y in model:
            est_I += arr_strength.get(assums.arr(x, y), 0)
            num_arrows += 1
        est_I = est_I / num_arrows if num_arrows > 0 else 0

        if best_model is None or best_I < est_I:
            best_model = model
            best_I = est_I
        logger.info(f"DAG from d-ABA: {best_B_est}")
    best_B_est = get_matrix_from_arrow_set(best_model, n_nodes)
    return best_model, best_B_est, best_I


@dataclass
class BestModel:
    best_model: frozenset  # frozenset of tuples (node1, node2)
    best_B_est: np.ndarray  # numpy array of shape (n_nodes, n_nodes)
    best_I: float  # float, best strength
    elapsed: float  # time taken to rank the models


@dataclass
class BestModelCollection:
    original: BestModel
    refined_indep_facts: BestModel
    arrows_sum: BestModel
    arrows_mean: BestModel


def get_best_model_various_valuations(models, n_nodes, cg, alpha, indep_to_strength, arr_strength):
    """ Get the best model from the set of models based on various valuations.
    Args:
        models: list of models, where each model is a frozenset of node pair tuples 
        representing arrows in the causal graph
        n_nodes: int, number of nodes in the causal graph
        cg: causal graph object
        alpha: float, significance level for independence tests
        indep_to_strength: dict, mapping from independence facts to their strengths
        arr_strength: dict, mapping from arrow sets to their strengths
    Returns:
        best_model: the best model based on the valuation criteria
    """
    start = time.time()
    best_model, best_B_est, best_I = get_best_model(
        models, n_nodes, cg, alpha=alpha)
    elapsed = time.time() - start
    original_best_model = BestModel(
        best_model=frozenset(best_model),
        best_B_est=best_B_est,
        best_I=best_I,
        elapsed=elapsed
    )

    start = time.time()
    best_model, best_B_est, best_I = get_best_model_by_refined_indep_facts(
        models, indep_to_strength, n_nodes, cg, alpha=alpha)
    elapsed = time.time() - start
    refined_indep_facts_best_model = BestModel(
        best_model=frozenset(best_model),
        best_B_est=best_B_est,
        best_I=best_I,
        elapsed=elapsed
    )

    start = time.time()
    best_model, best_B_est, best_I = get_best_model_by_arrows_sum(models, arr_strength, n_nodes)
    elapsed = time.time() - start
    arrows_sum_best_model = BestModel(
        best_model=frozenset(best_model),
        best_B_est=best_B_est,
        best_I=best_I,
        elapsed=elapsed
    )

    start = time.time()
    best_model, best_B_est, best_I = get_best_model_by_arrows_mean(models, arr_strength, n_nodes)
    elapsed = time.time() - start
    arrows_mean_best_model = BestModel(
        best_model=frozenset(best_model),
        best_B_est=best_B_est,
        best_I=best_I,
        elapsed=elapsed
    )

    return BestModelCollection(
        original=original_best_model,
        refined_indep_facts=refined_indep_facts_best_model,
        arrows_sum=arrows_sum_best_model,
        arrows_mean=arrows_mean_best_model
    )


def get_models_from_facts(facts, seed, n_nodes, base_location='./facts.lp'):
    """ Get models from facts using the CausalABA framework.
        NOTE: this uses the old implementation of CausalABA in ArgCausalDisco.
    Args:
        facts: list of Fact objects
        seed: int, random seed for reproducibility
        n_nodes: int, number of nodes in the causal graph
        base_location: str, base path for saving facts
    Returns:
        model_sets: list of models, where each model is a frozenset of node pair tuples 
        representing arrows in the causal graph."""
    facts_location = get_fact_location(facts, base_location=base_location)

    random_stability(seed)
    model_sets, _ = CausalABA(
        n_nodes, facts_location, weak_constraints=True, skeleton_rules_reduction=True,
        fact_pct=1.0, search_for_models='first',
        opt_mode='optN', print_models=False, set_indep_facts=False)
    model_sets, _ = set_of_models_to_set_of_graphs(model_sets, n_nodes, False)
    return model_sets
