full:
	src/run.sh

test:
	src/run.sh test

lint:
	flake8 src/
	isort --check --diff src/

requirements:
	command -v pip-compile &> /dev/null || pip install pip-compile
	pip-compile
