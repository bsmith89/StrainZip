JUPYTER_PORT=8888
CLEANUP="build .pytest_cache"

.PHONY: test format clean install-pre-commit black isort code-format pre-commit
.PHONY: install-git-jupyter-notebook-clean-smudge install-jupyter-kernel
.PHONY: start-jupyter


test:
	python -m pytest

type-check:
	pyright

format:
	python -m black src tests
	isort --profile black src tests

clean:
	rm -rf ${CLEANUP}

install-pre-commit:
	pre-commit install

black:
	black src tests typings

isort:
	isort src tests typings

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
