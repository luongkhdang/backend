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

# Make a separate stage for non-Google dependencies
# This ensures pip actually installs everything and fails if there are issues
RUN pip install -r requirements.txt \
    && pip install torch transformers sentencepiece docker \
    && pip install hdbscan scikit-learn pandas numpy~=1.26.4 \
    && python -c "import transformers; print(f'Successfully installed transformers {transformers.__version__}')" \
    && python -c "import torch; print(f'Successfully installed torch {torch.__version__}')" \
    && python -c "import google.generativeai; print('Successfully installed google.generativeai')" \
    && python -c "import hdbscan; print(f'Successfully installed hdbscan {hdbscan.__version__}')" \
    && python -c "import sklearn; print(f'Successfully installed scikit-learn {sklearn.__version__}')" \
    && python -c "import pandas; print(f'Successfully installed pandas {pandas.__version__}')"

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Create logs directory for error reports
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Copy application code - AFTER dependencies are installed
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV READER_DB_HOST=postgres
ENV READER_DB_PORT=5432
ENV READER_DB_NAME=reader_db
ENV READER_DB_USER=postgres
ENV READER_DB_PASSWORD=postgres
ENV WAIT_FOR_DB=true
ENV WAIT_SECONDS=10
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Gemini API settings (values will be overridden by docker-compose)
ENV GEMINI_API_KEY="YOUR_API_KEY_HERE"
ENV GEMINI_EMBEDDING_MODEL="models/text-embedding-004"
ENV GEMINI_EMBEDDING_TASK_TYPE="CLUSTERING"
ENV GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT=2048
ENV GEMINI_EMBEDDING_OUTPUT_DIMENSION=768
ENV GEMINI_EMBEDDING_RATE_LIMIT_PER_MINUTE=1500

# Flash model settings (for future use)
ENV GEMINI_FLASH_MODEL="models/gemini-2.0-flash"
ENV GEMINI_INPUT_CONTEXT_WINDOW_TOKEN=1048576
ENV GEMINI_FLASH_OUTPUT_WINDOW_TOKEN=8192
ENV GEMINI_FLASH_RATE_LIMIT_PER_MINUTE=15
ENV GEMINI_FLASH_RATE_LIMIT_PER_DAY=1500

# Step 2: Clustering settings (default values, will be overridden by docker-compose)
ENV RUN_CLUSTERING_STEP=false
ENV MIN_CLUSTER_SIZE=10
ENV HOT_CLUSTER_THRESHOLD=20
ENV INTERPRET_CLUSTERS=true
ENV MAX_CLUSTERS_TO_INTERPRET=10
ENV CLUSTER_SAMPLE_SIZE=10

# Verify module imports work - This will cause build failure if imports don't work!
RUN python -c "import sys; print(sys.path)" && \
    python -c "import google.generativeai; print('Verified google.generativeai')" \
    && python -c "import transformers; print('Verified transformers')" \
    && python -c "import src.gemini.gemini_client; print('Verified gemini_client')" \
    && python -c "from src.steps.step1 import run; print('Verified step1')" \
    && python -c "import hdbscan; print('Verified hdbscan')" \
    && python -c "import sklearn; print('Verified scikit-learn')" \
    && echo "ALL IMPORTS VERIFIED SUCCESSFULLY"

# Command to run the application
CMD ["python", "src/database/db_setup.py"] 