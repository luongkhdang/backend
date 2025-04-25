FROM article-transfer-base:latest

WORKDIR /app

# Use build arg to track when image was built
ARG BUILD_VERSION=1.0.0
ENV BUILD_VERSION=${BUILD_VERSION}

# Copy application code - AFTER dependencies are installed in the base image
COPY . .

# Create output directory for Step 4
RUN mkdir -p /app/src/output && chmod 777 /app/src/output

# Install json5 dependency needed by the copied code
RUN pip install --no-cache-dir json5 hf_xet

# Set environment variables
ENV READER_DB_HOST=postgres
ENV READER_DB_PORT=5432
ENV READER_DB_NAME=reader_db
ENV READER_DB_USER=postgres
ENV READER_DB_PASSWORD=postgres
ENV WAIT_FOR_DB=true
ENV WAIT_SECONDS=10
# Set to DEBUG level by default
ENV LOG_LEVEL=DEBUG
# Enable debug output for Gemini client
ENV DEBUG_GEMINI_CLIENT=true

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

# Step 2: Clustering settings (defaults below are now overridden by docker-compose)
# ENV RUN_CLUSTERING_STEP=false # Removed: Controlled by RUN_STEP2 in compose
ENV MIN_CLUSTER_SIZE=10
ENV HOT_CLUSTER_THRESHOLD=20
# ENV INTERPRET_CLUSTERS=true # Removed: Runs automatically if spaCy available
ENV MAX_CLUSTERS_TO_INTERPRET=10
ENV CLUSTER_SAMPLE_SIZE=10

# Step 3: Entity Extraction Settings (defaults below are now overridden by docker-compose)
# ENV RUN_STEP3=true # Removed: Controlled by RUN_STEP3 in compose
ENV ENTITY_EXTRACTION_BATCH_SIZE=10
ENV ENTITY_MAX_PRIORITY_ARTICLES=100
# ENV CALCULATE_INFLUENCE_SCORES=true # Removed: Runs automatically within Step 3 logic in main.py
# Allow longer wait times for fallback model in emergency case
ENV GEMINI_MAX_WAIT_SECONDS=60
ENV EMERGENCY_FALLBACK_WAIT_SECONDS=120
# ENV CALCULATE_DOMAIN_GOODNESS=true # Removed: Runs automatically after phase 1 in main.py

# Verify only application modules - faster than verifying all dependencies
RUN python -c "import sys; print(sys.path)" && \
    python -c "import src.gemini.gemini_client; print('Verified gemini_client')" && \
    python -c "from src.steps.step1 import run; print('Verified step1')" && \
    python -c "from src.steps.step2 import run; print('Verified step2')" && \
    python -c "from src.steps.step3 import run; print('Verified step3')" && \
    echo "APP IMPORTS VERIFIED SUCCESSFULLY"

# Command to run the application
CMD ["python", "src/database/db_setup.py"] 