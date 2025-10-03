from itertools import combinations
from math import isnan

try:
    # Prefer SciPy if available for accurate p-values
    from scipy.stats import ttest_ind_from_stats
except Exception:  # fallback with no p-value if SciPy missing
    ttest_ind_from_stats = None


def _short_label(name: str) -> str:
    if 'ABAPC' in name and 'LLM' not in name:
        return 'APC'
    if 'ABAPC' in name and 'LLM' in name:
        return 'APC-LLM'
    if 'SPC' in name:
        return 'SPC'
    if 'NOTEARS' in name:
        return 'NT'
    if 'Random' in name:
        return 'RND'
    return name


def print_ttests_for_dag_groups(df, nobs: int = 50):
    """
    Print LaTeX-friendly pairwise Welch t-tests for grouped DAG metrics.

    Expected columns per metric: '<metric>_mean' and '<metric>_std'.
    Groups by 'dataset' and compares across unique 'model' values present.

    Metrics and order:
      1) p_shd, F1
      2) p_SID
      3) precision, recall

    Parameters
    ----------
    df : pandas.DataFrame
        Grouped DAG dataframe (e.g., `dag_grouped`).
    nobs : int
        Number of observations per group used in t-test-from-stats.
    """

    import pandas as pd  # assumed available in the notebook env

    if df is None or len(df) == 0:
        print('No data available for DAG t-tests.')
        return

    print("0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 'ns' 1.")

    models_for_tests = sorted(df['model'].unique())
    print(models_for_tests)
    if not models_for_tests:
        print('No models available for t-tests.')
        return

    var_groups = [
        ('p_shd', 'F1'),
        ('p_SID',),
        ('precision', 'recall'),
    ]
    var_pretty = {
        'p_shd': 'NSHD',
        'p_SID': 'NSID',
        'F1': 'F1',
        'precision': 'Precision',
        'recall': 'Recall',
    }

    for dataset_label, dataset_rows in df.groupby('dataset'):
        print(dataset_label.replace('<br>', ' ').upper())
        print('=' * 50)
        for group in var_groups:
            for var in group:
                mean_col = f'{var}_mean'
                std_col = f'{var}_std'
                # Defensive checks for presence of aggregated columns
                if mean_col not in dataset_rows.columns or std_col not in dataset_rows.columns:
                    continue
                print(var_pretty.get(var, var))
                for method1, method2 in combinations(models_for_tests, 2):
                    rows1 = dataset_rows[dataset_rows['model'] == method1]
                    rows2 = dataset_rows[dataset_rows['model'] == method2]
                    if rows1.empty or rows2.empty:
                        continue
                    a = rows1[mean_col].values[0]
                    b = rows2[mean_col].values[0]
                    a_std = rows1[std_col].values[0]
                    b_std = rows2[std_col].values[0]
                    if pd.isna(a) or pd.isna(b):
                        continue

                    # Compute Welch's t-test from summary stats if SciPy is available
                    if ttest_ind_from_stats is not None and not any(map(isnan, [a, a_std, b, b_std])):
                        t, p = ttest_ind_from_stats(a, a_std, nobs, b, b_std, nobs, equal_var=False)
                        sig = '' if p > 0.1 else '.' if p <= 0.1 and p > 0.05 else '*' if p <= 0.05 and p > 0.01 else '**' if p <= 0.01 and p > 0.001 else '***'
                    else:
                        t, p, sig = float('nan'), float('nan'), ''

                    label1 = _short_label(method1)
                    label2 = _short_label(method2)
                    # Metric-specific formatting for readability
                    if var in ('F1', 'precision', 'recall'):
                        a_str, a_std_str = f"{float(a):.2f}", f"{float(a_std):.2f}"
                        b_str, b_std_str = f"{float(b):.2f}", f"{float(b_std):.2f}"
                    else:
                        a_str, a_std_str = f"{float(a):.1f}", f"{float(a_std):.1f}"
                        b_str, b_std_str = f"{float(b):.1f}", f"{float(b_std):.1f}"

                    print(rf"\!\!\!{label1} $({a_str}\pm{a_std_str})$ \!v\! {label2} $({b_str}\pm{b_std_str})$ \!\!\!&\!\!\! {t:.3f} \!\!\!&\!\!\! {p:.3f}{sig} \\")
            print("")


# Convenience entry point for %run usage inside the notebook
def run(nobs: int = 50):
    try:
        # Prefer the DAG grouping if present in the notebook
        df = dag_grouped.copy()  # type: ignore[name-defined]
    except Exception:
        try:
            df = all_sum.copy()  # type: ignore[name-defined]
        except Exception:
            df = None
    print_ttests_for_dag_groups(df, nobs=nobs)


if __name__ == '__main__':
    run()

