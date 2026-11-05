.PHONY: help clean check test format-check autoformat
.DEFAULT_GOAL := help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Check syntax and undefined names without changing files."
	@echo "make autoformat"
	@echo "    Apply formatting in place with black and fixable Ruff edits without failing on existing lint backlog."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	ruff check --select E9,F821,F822,F823 mr0 deployment examples tools tests

# Full style checks are separate from runtime correctness checks.
format-check:
	black --check .
	ruff check .

test:
	python -m pytest tests -q

autoformat:
	black .
	ruff check --fix-only --show-fixes .
