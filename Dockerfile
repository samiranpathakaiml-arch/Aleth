# ============================================================
# Aleth — OpenEnv Submission Dockerfile
# Builds a FastAPI server that the Scaler dashboard can reach
# ============================================================

FROM python:3.10-slim

LABEL maintainer="aleth-team"
LABEL version="1.1.0"
LABEL description="Aleth OpenEnv citation verification benchmark"

WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all root-level source files (flat structure)
COPY models.py       .
COPY grader.py       .
COPY reward.py       .
COPY environment.py  .
COPY main.py         .
COPY inference.py    .
COPY openenv.yaml    .

# Copy benchmark data
COPY data/ ./data/

# Runtime env vars (override with -e at docker run)
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=""
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Port expected by Hugging Face Spaces / Scaler dashboard
EXPOSE 7860

# Start the OpenEnv server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
