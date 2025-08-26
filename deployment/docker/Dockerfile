# Production Dockerfile for Photonic Neuromorphics Simulation Platform
# Multi-stage build for optimized production deployment

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-prod.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml setup.py ./

# Install application
RUN pip install --no-cache-dir .

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Set labels for metadata
LABEL maintainer="Terragon Labs <dev@terragon.ai>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="photonic-neuromorphics-sim" \
      org.label-schema.description="Photonic Neuromorphics Simulation Platform" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/danieleschmidt/photonic-neuromorphics-sim" \
      org.label-schema.vendor="Terragon Labs" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Create non-root user for security
RUN groupadd -r photonic && useradd -r -g photonic photonic

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=photonic:photonic src/ ./src/
COPY --chown=photonic:photonic scripts/ ./scripts/
COPY --chown=photonic:photonic monitoring/ ./monitoring/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/tmp \
    && chown -R photonic:photonic /app

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    PHOTONIC_ENV=production \
    PHOTONIC_LOG_LEVEL=INFO \
    PHOTONIC_DATA_DIR=/app/data \
    PHOTONIC_LOG_DIR=/app/logs \
    PHOTONIC_TMP_DIR=/app/tmp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from photonic_neuromorphics.core import PhotonicCore; PhotonicCore().health_check()" || exit 1

# Expose ports
EXPOSE 8080 8443 9090

# Switch to non-root user
USER photonic

# Default command
CMD ["python", "-m", "photonic_neuromorphics.api.server", "--host", "0.0.0.0", "--port", "8080"]