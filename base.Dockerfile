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
    && pip install --no-cache-dir torch==2.3.0 transformers==4.41.1 sentencepiece==0.2.0 \
    && pip install --no-cache-dir scikit-learn==1.3.2 pandas==2.2.0 numpy~=1.26.4 \
    && pip install --no-cache-dir docker

# Install hdbscan separately for better error visibility
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir hdbscan==0.8.36 \
    && echo "HDBSCAN installed successfully" || (echo "HDBSCAN installation failed!" && exit 1)

# Verify installations in a separate RUN command
# Don't try to access __version__ for hdbscan as it doesn't expose it directly
RUN python -c "import transformers; print(f'Successfully installed transformers {transformers.__version__}')" \
    && python -c "import torch; print(f'Successfully installed torch {torch.__version__}')" \
    && python -c "import google.generativeai; print('Successfully installed google.generativeai')" \
    && python -c "import hdbscan; print('Successfully installed hdbscan')" \
    && python -c "import sklearn; print(f'Successfully installed scikit-learn {sklearn.__version__}')" \
    && python -c "import pandas; print(f'Successfully installed pandas {pandas.__version__}')" \
    && python -c "import spacy; print('Successfully installed spacy')"

# Download spaCy model - kept separate as it's a different tool
RUN python -m spacy download en_core_web_lg \
    && echo "spaCy model downloaded successfully"

# Create logs directory for error reports
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Set base environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 