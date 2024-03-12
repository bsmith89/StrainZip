JUPYTER_PORT=8888
CLEANUP="build .pytest_cache"

.PHONY: test clean install-pre-commit black isort code-format pre-commit
.PHONY: install-git-jupyter-notebook-clean-smudge install-jupyter-kernel
.PHONY: start-jupyter debug-test


test:
	python -m pytest

debug-test:
	python -m pytest --pdb

type-check:
	pyright

clean:
	rm -rf ${CLEANUP}

install-pre-commit:
	pre-commit install

black:
	black src tests typings

isort:
	isort --profile black src tests typings

code-format: black isort

pre-commit:
	pre-commit run --all-files

install-git-jupyter-notebook-clean-smudge:
	git config --local filter.dropoutput_jupyternb.clean scripts/jupyternb_output_filter.py
	git config --local filter.dropoutput_jupyternb.smudge cat

install-jupyter-kernel:
	python -m ipykernel install --user --name=strainzip

start-jupyter:
	jupyter lab --port ${JUPYTER_PORT}
