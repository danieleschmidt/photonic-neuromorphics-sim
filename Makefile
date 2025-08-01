.PHONY: help install install-dev test lint format type clean docs docs-live build release
.PHONY: docker-build docker-test docker-dev security-scan performance-test
.PHONY: validate-spice validate-verilog validate-layouts benchmark

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install development dependencies"
	@echo "  dev-setup    Complete development environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run tests with pytest"
	@echo "  test-cov     Run tests with coverage reporting"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests"
	@echo "  test-security     Run security tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  type         Run type checking with mypy"
	@echo "  security-scan  Run security scanning"
	@echo ""
	@echo "Photonic Validation:"
	@echo "  validate-spice    Validate SPICE netlists"
	@echo "  validate-verilog  Validate Verilog modules"
	@echo "  validate-layouts  Validate GDS layouts"
	@echo "  benchmark         Run performance benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker images"
	@echo "  docker-test  Run tests in Docker"
	@echo "  docker-dev   Start development environment"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  docs-live    Build docs with live reload"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean        Clean build artifacts"
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

# Additional test targets
test-unit:
	pytest tests/unit/ -v -m "unit"

test-integration:
	pytest tests/integration/ -v -m "integration"

test-performance:
	pytest tests/performance/ -v -m "performance"

test-security:
	pytest tests/security/ -v -m "security"

test-all:
	pytest tests/ -v --cov=photonic_neuromorphics --cov-report=html

# Security scanning
security-scan:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "Security scan complete. Check bandit-report.json and safety-report.json"

# Photonic-specific validation
validate-spice:
	@echo "Validating SPICE netlists..."
	@find . -name "*.sp" -o -name "*.cir" -o -name "*.net" | while read file; do \
		echo "Checking $$file"; \
		grep -q ".end" "$$file" || echo "Warning: $$file missing .end statement"; \
	done

validate-verilog:
	@echo "Validating Verilog modules..."
	@find . -name "*.v" -o -name "*.sv" | while read file; do \
		echo "Checking $$file"; \
		grep -q "module" "$$file" || echo "Warning: $$file missing module declaration"; \
		grep -q "endmodule" "$$file" || echo "Warning: $$file missing endmodule statement"; \
	done

validate-layouts:
	@echo "Validating GDS layouts..."
	@find . -name "*.gds" | while read file; do \
		echo "Checking $$file"; \
		[ -s "$$file" ] || echo "Warning: $$file is empty"; \
	done

# Performance benchmarking
benchmark:
	pytest tests/regression/test_performance_regression.py -v -m "benchmark" --benchmark-only

# Docker targets
docker-build:
	docker build -t photonic-neuromorphics:dev --target development .
	docker build -t photonic-neuromorphics:prod --target production .
	docker build -t photonic-neuromorphics:test --target testing .

docker-test:
	docker-compose -f docker-compose.yml run --rm test

docker-dev:
	docker-compose -f docker-compose.yml up dev

docker-clean:
	docker-compose -f docker-compose.yml down -v
	docker system prune -f

# CI/CD helpers
ci-test: test-all security-scan validate-spice validate-verilog
	@echo "CI test suite complete"

ci-build: clean lint type ci-test build
	@echo "CI build complete"

# Monitoring and profiling
profile:
	python -m cProfile -o profile.stats -m photonic_neuromorphics.cli --help
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile:
	python -m memory_profiler -m photonic_neuromorphics.cli --help

# Cleanup extended
clean-all: clean
	docker-compose -f docker-compose.yml down -v
	docker system prune -f
	rm -rf .tox/
	rm -rf node_modules/
	rm -rf *.prof
	rm -rf profile.stats
	rm -rf bandit-report.json
	rm -rf safety-report.json