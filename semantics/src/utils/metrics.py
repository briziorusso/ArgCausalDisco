from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
import numpy as np


def get_metrics(W_est, B_true):
    """
    Calculate metrics for the estimated graph W_est against the true graph B_true.
    Args:
        W_est: np.ndarray, estimated adjacency matrix
        B_true: np.ndarray, true adjacency matrix
    Returns:
        dictionary with metrics for both CPDAG and DAG representations.
    """
    res_columns = ['nnz', 'fdr', 'tpr', 'fpr', 'precision', 'recall', 'F1', 'shd', 'sid']
    try:
        B_est = (W_est != 0).astype(int)
        mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
    except Exception as e:
        mt_cpdag = {col: np.nan for col in res_columns}
    try:
        B_est = (W_est > 0).astype(int)
        mt_dag = DAGMetrics(B_est, B_true).metrics
    except Exception as e:
        mt_dag = {col: np.nan for col in res_columns}

    # Handle the 'sid' metric which may be a tuple
    if type(mt_cpdag['sid']) == tuple:
        mt_sid_low = mt_cpdag['sid'][0]
        mt_sid_high = mt_cpdag['sid'][1]
    else:
        mt_sid_low = mt_cpdag['sid']
        mt_sid_high = mt_cpdag['sid']
    mt_cpdag.pop('sid')
    mt_cpdag['sid_low'] = mt_sid_low
    mt_cpdag['sid_high'] = mt_sid_high

    return mt_cpdag, mt_dag
