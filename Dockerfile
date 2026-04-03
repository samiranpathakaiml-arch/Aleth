# ============================================================
# Aleth — Citation Verification Environment
# Docker image for containerized inference runs
# ============================================================

FROM python:3.10-slim

# Metadata
LABEL maintainer="aleth-team"
LABEL version="1.0.0"
LABEL description="Aleth OpenEnv scientific citation verification benchmark"

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Aleth package
COPY aleth/ ./aleth/

# Copy the runner and config at root level
COPY inference.py .
COPY openenv.yaml .

# Anthropic API key (override at runtime with -e ANTHROPIC_API_KEY=...)
ENV ANTHROPIC_API_KEY=""

# Expose port for optional HTTP interface / health checks
EXPOSE 7860

# Default entrypoint: run the baseline inference
CMD ["python", "inference.py"]
