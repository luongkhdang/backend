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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install non-Google dependencies first
RUN pip install --no-cache-dir $(grep -v "google-\|protobuf" requirements.txt)

# Explicitly install Google packages in the correct order to resolve dependencies
RUN pip install --no-cache-dir protobuf==4.25.3 && \
    pip install --no-cache-dir google-api-core==2.18.0 && \
    pip install --no-cache-dir google-auth==2.22.0 && \
    pip install --no-cache-dir google-api-python-client==2.112.0 && \
    pip install --no-cache-dir google-generativeai==0.7.1

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Create logs directory for error reports
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Copy application code
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
ENV GEMINI_EMBEDDING_MODEL="text-embedding-004"
ENV GEMINI_EMBEDDING_TASK_TYPE="CLUSTERING"
ENV GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT=2048
ENV GEMINI_EMBEDDING_OUTPUT_DIMENSION=768

# Flash model settings (for future use)
ENV GEMINI_FLASH_MODEL="gemini-2.0-flash"
ENV GEMINI_INPUT_CONTEXT_WINDOW_TOKEN=1048576
ENV GEMINI_FLASH_OUTPUT_WINDOW_TOKEN=8192
ENV GEMINI_FLASH_RATE_LIMIT_PER_MINUTE=15
ENV GEMINI_FLASH_RATE_LIMIT_PER_DAY=1500

# Command to run the application
CMD ["python", "src/database/db_setup.py"] 