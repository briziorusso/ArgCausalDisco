This subrepo is adapted from https://github.com/fisheeve/gradual-sem-causal-aba

# Gradual Semantics Causal ABA

This project explores the application of various extension-based Assumption Based Argumentation (ABA) semantics, as well as gradual ABA semantics within the Causal ABA framework. The repository contains source code implementing proposed approaches along with experimentation scripts and evaluation results.

## Prerequisites

Python 3.12  
R https://cran.rstudio.com/  
[OPTIONAL] conda 25.1 (Miniconda installation would suffice)  


## Installation

Clone the repo from GitHub.
```bash
git clone git@github.com:fisheeve/gradual-sem-causal-aba.git
cd gradual-sem-causal-aba
```

Run the following to create and setup a python virtual environment, clone sub-repositories, and install R packages:
```bash
make
```
When this repository lives inside `ArgCausalDisco/gradual-sem-causal-aba`, the parent `ArgCausalDisco` checkout is used directly instead of cloning a second copy inside the gradual folder.
Running `make` will prompt inputting the location of Rscript and R Library of your setup. The value of those parameters will be stored in environment variables `RPATH` and `R_LIB_DIR` respectively. The values of these variables are then stored in a `.env` file. The resulting `.env` should look similar to the following:
```
RPATH=/your/path/to/R/4.4.2-gfbf-2024a/bin/Rscript

R_LIB_DIR=/your/path/to/miniforge3/envs/aba-env/lib/R/library
```

Finally, activate the virtual environment before running experiments or tests:
```bash
conda activate aba-env
```
or if you don't have conda installed:
```bash
source .venv/bin/activate
```

## Repository Structure

This repository is organised into the following directories:

- **`src/`** - Core source code implementing the main algorithms and utilities

- **`notebooks/`** - Jupyter notebooks for analysis, visualization, and demonstrations. Contains a few small-scale experiment scripts as well.

- **`scripts/`** - Experimental scripts for large-scale evaluation:
  - `python_scripts/` - Python scripts for running experiments
  - `shell_scripts/` - PBS job submission scripts for HPC environments

- **`results/`** - Experimental results and stored data:
  - `gradual/` - Results from gradual semantics experiments
  - `extension_based_semantics/` - Results comparing Causal ABA under different extension based ABA semantics
  - `existing/` - Baseline results from existing causal discovery methods

- **External Dependencies** - Integrated submodules and external tools:
  - Parent `ArgCausalDisco/` repo, or local `ArgCausalDisco/` clone when used standalone - Implementation of Causal ABA, causal discovery baselines, and evaluation utilities
  - `GradualABA/` - Gradual ABA implementation
  - `aspforaba/` - ASP-based ABA solver
  - `notears/` - NOTEARS causal discovery baseline
  - `py-causal` - pycausal causal discovery python toolkit

- **`tests/`** - Test suite with unit and end-to-end tests

## Tests
Run the following in the terminal to run the unit tests:
```bash
export PYTHONPATH=. && pytest tests/unit
```
Run the following for end-to-end tests (it takes around 40 seconds for these tests to complete):
```bash
export PYTHONPATH=. && pytest tests/end_to_end
```

## Experiments
All of the experiments relevant to this project are in folders `scripts/` and `notebooks/`. Experiment results are in the `results/` folder.
The `notebooks/` folder contains mostly demonstration notebooks that plot results yielded from python scripts in `scripts/`.
The `scripts/` contains python scripts to run experiments. It also contains shell scripts used for submitting python scripts as jobs on [Imperial HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/).


## Declarations
### Use of Generative AI
I acknowledge the use of [GitHub Copilot](https://github.com/features/copilot) powered by [GPT-4.1](https://openai.com/index/gpt-4-1/) for code completion and data visualisation script generation.

## Acknowledgements
I would like to thank the authors and contributors of the following projects, whose source code was used in this research:

- [ArgCausalDisco](https://github.com/briziorusso/ArgCausalDisco.git)  
- [GradualABA](https://github.com/briziorusso/GradualABA.git)  
- [ASP for ABA](https://bitbucket.org/coreo-group/aspforaba.git)  
- [NOTEARS](https://github.com/xunzheng/notears.git)  
- [py-causal](https://github.com/bd2kccd/py-causal.git)

This work was supported by [Imperial High Performance Computing (HPC)](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/) resources provided by **Imperial College London**.
