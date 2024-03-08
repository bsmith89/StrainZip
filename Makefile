test:
	python -m pytest

format:
	python -m black src tests
	isort --profile black src tests

pre-commit:
	pre-commit run --all-files

install-git-jupyter-notebook-clean-smudge:
	git config --local filter.dropoutput_jupyternb.clean scripts/jupyternb_output_filter.py
	git config --local filter.dropoutput_jupyternb.smudge cat
