# Multi-stage build for photonic neuromorphics simulation
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for EDA tools and photonic simulation
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    gfortran \
    ngspice \
    gtkwave \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev,docs,test]"

# Set up pre-commit hooks
RUN git config --global --add safe.directory /app
RUN pre-commit install || true

CMD ["python", "-m", "photonic_neuromorphics.cli", "--help"]

# Production stage
FROM base as production

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml README.md LICENSE ./

# Install package
RUN pip install .

# Create non-root user with proper permissions
RUN useradd --create-home --shell /bin/bash --uid 1000 photonic \
    && mkdir -p /app/outputs /app/layouts /app/simulation_data \
    && chown -R photonic:photonic /app
USER photonic

# Set up user environment
ENV HOME=/home/photonic \
    PATH=$HOME/.local/bin:$PATH

CMD ["photonic-sim", "--help"]

# Testing stage
FROM development as testing

# Run tests and coverage
RUN pytest tests/ --cov=photonic_neuromorphics --cov-report=xml --cov-report=term

# Documentation stage
FROM development as docs

EXPOSE 8000
CMD ["make", "docs-live"]