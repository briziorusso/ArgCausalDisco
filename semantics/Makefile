PYTHON_VERSION=3.12
ENV_NAME=aba-env
VENV_DIR=.venv

# Always remove and recreate .env
$(shell rm -f .env; \
	read -p "R_LIB_DIR is not set. Enter R library path (e.g. ~/R/library): " rlibdir; \
	read -p "RPATH is not set. Enter Rscript path (e.g. /usr/bin/Rscript): " rpath; \
	echo "R_LIB_DIR=$$rlibdir" >> .env; \
	echo "RPATH=$$rpath" >> .env)

include .env
export

.PHONY: install clean

install: clean
	@echo "Setting up Python environment..."
	@if command -v conda > /dev/null 2>&1; then \
		echo "Conda found. Creating and activating conda environment..."; \
		conda create -y -n $(ENV_NAME) -c conda-forge python=$(PYTHON_VERSION) r-base; \
		bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install --upgrade pip && pip install -r requirements.txt"; \
	else \
		echo "Conda not found. Falling back to python venv..."; \
		python$(PYTHON_VERSION) -m venv $(VENV_DIR); \
		. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \
	fi

	@echo "Cloning required repositories..."
	git clone https://bitbucket.org/coreo-group/aspforaba.git
	@if [ -f ../__init__.py ] && [ -f ../abapc.py ] && [ -d ../utils ]; then \
		echo "Using embedded parent ArgCausalDisco checkout."; \
	else \
		git clone https://github.com/briziorusso/ArgCausalDisco.git; \
	fi
	git clone https://github.com/xunzheng/notears.git
	git clone https://github.com/briziorusso/GradualABA.git
	git clone https://github.com/bd2kccd/py-causal.git

	cd py-causal && python setup.py install;

	@echo "Applying custom changes to aspforaba sub-repo..."
	cd aspforaba && git apply ../aspforaba.patch

	@echo "Checking out specific commit of ArgCausalDisco..."
	@if [ -d ArgCausalDisco/.git ]; then \
		cd ArgCausalDisco && git reset --hard 579a1c95f576ecb53d97b0c62d4ceebb90368189; \
	else \
		echo "Skipping ArgCausalDisco checkout because the embedded parent repo is being used."; \
	fi

	@echo "Checking out specific commit of GradualABA..."
	cd GradualABA && git reset --hard 592d45db65aab28b634c1f7e73944dc4482a32b7

	@echo "Installing additional dependencies..."
	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	$(RPATH) -e 'install.packages("BiocManager", repos="https://cloud.r-project.org", lib="$(R_LIB_DIR)")'

	R_INTERACTIVE=FALSE R_PAPERSIZE=letter PAGER=cat R_OPTS="--no-save --no-restore --quiet" \
	$(RPATH) -e '.libPaths(c("$(R_LIB_DIR)", .libPaths())); BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'
clean:
	@echo "Removing virtual environment '$(VENV_DIR)' or conda env '$(ENV_NAME)' (if exists)..."
	@if command -v conda > /dev/null 2>&1; then \
		conda remove -y --name $(ENV_NAME) --all || true; \
	fi
	-rm -rf $(VENV_DIR)

	@echo "Removing cloned repositories..."
	-rm -rf aspforaba
	-rm -rf ArgCausalDisco
	-rm -rf notears
	-rm -rf GradualABA
	-rm -rf py-causal
