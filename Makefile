.PHONY: help install install-dev test lint format type clean docs docs-live build release

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests with pytest"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  type         Run type checking with mypy"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  docs-live    Build docs with live reload"
	@echo "  build        Build distribution packages"
	@echo "  release      Build and upload to PyPI"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,test]"
	pre-commit install

# Testing and quality
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=photonic_neuromorphics --cov-report=html --cov-report=term

lint:
	black --check --diff src tests
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

type:
	mypy src

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

docs-live:
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

# Building and releasing
build: clean
	python -m build

release: build
	python -m twine upload dist/*

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make docs-live' to start documentation server"