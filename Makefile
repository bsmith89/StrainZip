JUPYTER_PORT=8888
CLEANUP="build .pytest_cache"


test:
	python -m pytest

format:
	python -m black src tests
	isort --profile black src tests

clean:
	rm -rf ${CLEANUP}

install-pre-commit:
	pre-commit install

pre-commit:
	pre-commit run --all-files

install-git-jupyter-notebook-clean-smudge:
	git config --local filter.dropoutput_jupyternb.clean scripts/jupyternb_output_filter.py
	git config --local filter.dropoutput_jupyternb.smudge cat

install_jupyter_kernel:
	python -m ipykernel install --user --name=strainzip

start_jupyter:
	jupyter lab --port ${JUPYTER_PORT}
