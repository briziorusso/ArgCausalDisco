# Child matched-10 subset

Built at: `2026-03-16T17:38:28`

Selection rule:
`random_stability(2024); np.random.randint(0, 10000, (50,))[:10]`

Selected seeds: `[7816, 3578, 2656, 2688, 2494, 183, 7977, 3199, 316, 8266]`

Derived versions:
- `bnlearn_50rep_abapc_matched10` from `bnlearn_50rep_abapc` (6 DAG files, 6 CPDAG files, 60 DAG rows, 60 CPDAG rows)
- `bnlearn_50rep_mpc_matched10` from `bnlearn_50rep_mpc` (6 DAG files, 6 CPDAG files, 60 DAG rows, 60 CPDAG rows)

Versions not subsetted because only aggregated summaries are available:
- `bnlearn_big_rnd_mpc`
- `bnlearn_big_fgs_nt`
- `bnlearn_child_base`
- `bnlearn_child_base2`

Exact child rerun:
`python scripts/run_child_matched10.py --version child_sat24_t600_matched10`

This workflow only creates duplicated derived outputs. It does not edit the source progress or summary files.
