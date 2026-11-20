FROM python:3.12-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and writable durable AI authoring directory
RUN useradd -m -u 1000 edcraft && \
    mkdir -p /app/.data/authoring && \
    chown -R edcraft:edcraft /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=edcraft:edcraft /app/.venv /app/.venv

# Add venv to PATH so we can use installed executables
ENV PATH="/app/.venv/bin:$PATH"

# Durable AI authoring state (mounted as a volume in docker-compose)
ENV AUTHORING_STORAGE_DIR=/app/.data/authoring

# Copy application code
COPY --chown=edcraft:edcraft pyproject.toml ./
COPY --chown=edcraft:edcraft edcraft_backend ./edcraft_backend
COPY --chown=edcraft:edcraft worker ./worker
COPY --chown=edcraft:edcraft alembic ./alembic
COPY --chown=edcraft:edcraft alembic.ini ./

# Switch to non-root user
USER edcraft

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - can be overridden in docker-compose
CMD ["uvicorn", "edcraft_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
