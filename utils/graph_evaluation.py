"""Evaluate learner outputs without discarding computable PDAG edge scores."""
from __future__ import annotations

import numpy as np

from utils.aij_graph_metrics import (
    InvalidGraph, complete_cpdag, consistent_extension, decode_estimate,
)
from utils.graph_utils import DAGMetrics


def evaluate_estimate(raw, truth, method, *, seed=None, metric_timeout=None, sid=True):
    """Return evaluation graphs, metrics, and explicit validity statuses.

    Native PDAG edges are retained if the output has no consistent extension or
    is not a completed equivalence class (e.g. it includes prior orientations).
    Such outputs receive descriptive scores against the true CPDAG, but no
    equivalence-class SID bounds. Unexpected backend errors propagate.
    """
    native = decode_estimate(raw, method)
    invalid = None
    try:
        dag = consistent_extension(native, seed=seed)
    except InvalidGraph as exc:
        dag = None
        cpdag = native
        invalid = {'status': 'no_consistent_extension', 'error': str(exc), 'timed_out': False}
    else:
        cpdag = complete_cpdag(dag)
        if (native * native.T).any() and not np.array_equal(native, cpdag):
            cpdag = native
            invalid = {'status': 'noncompleted_pdag', 'timed_out': False,
                       'error': 'Native PDAG does not represent a complete equivalence class'}

    cp = DAGMetrics(cpdag, truth, sid=sid and invalid is None,
                    metric_timeout=metric_timeout, evaluation_kind='cpdag')
    if invalid is not None:
        cp.eval_status['graph_eval'] = dict(invalid)
        if sid:
            cp.metrics['sid'] = (np.nan, np.nan)
            cp.eval_status['sid'] = dict(invalid, exact=False)

    if dag is None:
        dag_metrics = {key: np.nan for key in cp.metrics}
        dag_status = {'graph_eval': dict(invalid)}
    else:
        result = DAGMetrics(dag, truth, sid=sid, metric_timeout=metric_timeout,
                            evaluation_kind='dag')
        dag_metrics, dag_status = result.metrics, result.eval_status
    return dict(dag=dag, cpdag=cpdag, dag_metrics=dag_metrics,
                cpdag_metrics=cp.metrics, dag_status=dag_status,
                cpdag_status=cp.eval_status)
