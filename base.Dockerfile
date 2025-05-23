FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    dnsutils \
    iputils-ping \
    curl \
    net-tools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements file first (better caching)
COPY requirements.txt .

# Install all Python dependencies in one go using pip cache mount
# This allows pip's resolver to handle all dependencies together
# Use --mount=type=cache for better caching during the base image build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.3.0 transformers==4.41.1 sentencepiece==0.2.0 accelerate hf_xet \
    && pip install --no-cache-dir scikit-learn==1.3.2 pandas==2.2.0 numpy~=1.26.4 \
    && pip install --no-cache-dir docker

# Explicitly install and verify psycopg2-binary and pgvector
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir psycopg2-binary>=2.9.9 pgvector>=0.2.3 \
    && echo "psycopg2-binary and pgvector explicitly installed"

# Install hdbscan separately for better error visibility
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir hdbscan==0.8.36 \
    && echo "HDBSCAN installed successfully" || (echo "HDBSCAN installation failed!" && exit 1)

# Verify installations in a separate RUN command
# Don't try to access __version__ for hdbscan as it doesn't expose it directly
RUN python -c "import transformers; print(f'Successfully installed transformers {transformers.__version__}')" \
    && python -c "import torch; print(f'Successfully installed torch {torch.__version__}')" \
    && python -c "import google.genai; print('Successfully installed google.genai')" \
    && python -c "import hdbscan; print('Successfully installed hdbscan')" \
    && python -c "import sklearn; print(f'Successfully installed scikit-learn {sklearn.__version__}')" \
    && python -c "import pandas; print(f'Successfully installed pandas {pandas.__version__}')" \
    && python -c "import spacy; print('Successfully installed spacy')" \
    && python -c "import psycopg2; print(f'Successfully installed and imported psycopg2 {psycopg2.__version__}')" \
    && python -c "import pgvector; print('Successfully installed and imported pgvector')"

# Download spaCy model - kept separate as it's a different tool
RUN python -m spacy download en_core_web_lg \
    && echo "spaCy model downloaded successfully"

# Verify PyTorch CUDA setup (optional but helpful)
RUN python -c "import torch; print(f'Torch CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}')"

# Create logs directory for error reports
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Set base environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 