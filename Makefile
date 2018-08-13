test: test-all

test-all:
	python tests/runner.py all

test-unit:
	python tests/runner.py unit

test-integration:
	python tests/runner.py integration

lint:
	pylint cluster

security:
	bandit -r cluster
