# Reproducibility Instructions

## Notebooks

Notebooks of experiments conducted in the thesis:

- [Optimisation Commparison](../results/2025/notebooks/scalability.ipynb) (Contains large output logs)
- [Prior Quality Assessment](../results/2025/notebooks/prior_assessment.ipynb)
- [Prior Integration with Causal ABA](../results/2025/notebooks/prior_integration.ipynb)

Their ouputs can be found under `results/2025` and `results/2025/llm/`.

## Datasets & Priors

Priors used for integration with Causal ABA can be found under `priors/datasets/**/prior_df.json`.

Bnlearn & Synthetic Datasets including variables and LLM-generated descriptions can be found under `priors/datasets/`.

## Download concept embeddings

[concept_embeddings.npy](https://huggingface.co/Lazyhope/causenet-precision-embeddings/resolve/main/concept_embeddings.npy) should be downloaded and placed under `priors/`.

## Matplotlib Figures

Scripts to reproduce figures in the thesis can be found under `scripts/`. Only benchmark log plots scripts are not stored, but should be easy to recreate.

## Generate Synthetic Datasets

Scripts to generate synthetic datasets can be found at [generate_causal_graph](dataset.py#generate_causal_graph), along with the heuristic presented in the thesis [heuristic_by_semantics](dataset.py#heuristic_by_semantics).
