test:
	python -m pytest

format:
	python -m black src tests
	isort --profile black src tests

pre-commit:
	pre-commit run --all-files
