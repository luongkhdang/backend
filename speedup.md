Okay, I understand the problem. Re-downloading and reinstalling large libraries like PyTorch, Transformers, spaCy, and their associated models during every Docker rebuild significantly slows down development.

Here's a breakdown of why this happens and the best way to minimize rebuild times using Docker best practices:

**Why Rebuilds Are Slow**

Docker builds images in layers. Each instruction in your `Dockerfile` (like `RUN`, `COPY`, `ADD`) creates a new layer. If any file involved in a layer changes, Docker has to rebuild that layer _and all subsequent layers_.

In your current `Dockerfile`:

1.  You `COPY requirements.txt .` (Layer 1)
2.  You `RUN pip install ...` and `RUN python -m spacy download ...` (Layer 2 - Very Slow)
3.  You `COPY . .` (Layer 3 - Copies all your application code)

If you change _any_ file in your application code (`src/`), the `COPY . .` layer becomes invalid, forcing Docker to re-run that step. Since dependency installation happens _before_ this, it usually stays cached _unless_ `requirements.txt` or an earlier `Dockerfile` line changes.

However, the issue is often that `requirements.txt` _does_ change, or you need to rebuild for other reasons, forcing the slow dependency installation layer to rerun.

**Best Solution: Multi-Stage Build / Dedicated Base Image**

The most effective strategy is to separate the slow, rarely changing dependency installation into a _base stage_ or a separate _base image_. Your application image then builds `FROM` this base.

**Implementation Steps:**

1.  **Create a `base.Dockerfile` (or use multi-stage):**
    This file will install Python, system dependencies, Python packages from `requirements.txt`, and download the spaCy model.

    ```dockerfile
    # base.Dockerfile
    FROM python:3.10-slim

    WORKDIR /app

    # Install system dependencies (same as before)
    RUN apt-get update && apt-get install -y \
        build-essential \
        libpq-dev \
        dnsutils \
        iputils-ping \
        curl \
        net-tools \
        git \
        && rm -rf /var/lib/apt/lists/*

    # Copy ONLY requirements file
    COPY requirements.txt .

    # Install Python dependencies using pip cache mount (requires BuildKit)
    # Use --mount=type=cache for better caching during the base image build
    RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt \
        && pip install torch transformers sentencepiece docker hdbscan scikit-learn pandas numpy~=1.26.4 \
        # Verify installations (optional but good)
        && python -c "import transformers; print(f'Installed transformers {transformers.__version__}')" \
        && python -c "import torch; print(f'Installed torch {torch.__version__}')" \
        && python -c "import google.generativeai; print('Installed google.generativeai')" \
        && python -c "import hdbscan; print(f'Installed hdbscan {hdbscan.__version__}')" \
        && python -c "import sklearn; print(f'Installed scikit-learn {sklearn.__version__}')" \
        && python -c "import pandas; print(f'Installed pandas {pandas.__version__}')" \
        && python -c "import spacy; print(f'Installed spacy {spacy.__version__}')"

    # Download spaCy model
    RUN python -m spacy download en_core_web_lg

    # Clean up pip cache after installation if not mounting cache during runtime
    # RUN rm -rf /root/.cache/pip

    # Basic environment setup needed for any stage using this base
    ENV PYTHONPATH=/app \
        LOG_LEVEL=INFO \
        PYTHONUNBUFFERED=1

    # Create logs directory now so permissions are set correctly
    RUN mkdir -p /app/logs && chmod 777 /app/logs

    # You can add verification steps here if needed, but they run every time the base is built

    ```

2.  **Build the Base Image:**
    Build this image once and tag it. You only need to rebuild it if `requirements.txt` or system dependencies change.

    ```bash
    # Enable BuildKit for cache mounts (if not default)
    export DOCKER_BUILDKIT=1

    # Build and tag the base image
    docker build -t article-transfer-base:latest -f base.Dockerfile .
    ```

3.  **Simplify Your Main `Dockerfile`:**
    Now, your application's `Dockerfile` becomes much smaller and faster to rebuild.

    ```dockerfile
    # Dockerfile (Application Stage)
    FROM article-transfer-base:latest

    WORKDIR /app

    # Copy application code - This is now one of the last steps!
    # Changes here won't trigger dependency re-installation.
    COPY . .

    # Set environment variables specific to the application runtime
    # (Database, API Keys, etc. - keep these as they are)
    ENV READER_DB_HOST=postgres \
        READER_DB_PORT=5432 \
        READER_DB_NAME=reader_db \
        READER_DB_USER=postgres \
        READER_DB_PASSWORD=postgres \
        WAIT_FOR_DB=true \
        WAIT_SECONDS=10 \
        GEMINI_API_KEY="YOUR_API_KEY_HERE" \
        GEMINI_EMBEDDING_MODEL="models/text-embedding-004" \
        GEMINI_EMBEDDING_TASK_TYPE="CLUSTERING" \
        GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT=2048 \
        GEMINI_EMBEDDING_OUTPUT_DIMENSION=768 \
        GEMINI_EMBEDDING_RATE_LIMIT_PER_MINUTE=1500 \
        GEMINI_FLASH_MODEL="models/gemini-2.0-flash" \
        GEMINI_INPUT_CONTEXT_WINDOW_TOKEN=1048576 \
        GEMINI_FLASH_OUTPUT_WINDOW_TOKEN=8192 \
        GEMINI_FLASH_RATE_LIMIT_PER_MINUTE=15 \
        GEMINI_FLASH_RATE_LIMIT_PER_DAY=1500 \
        RUN_CLUSTERING_STEP=false \
        MIN_CLUSTER_SIZE=10 \
        HOT_CLUSTER_THRESHOLD=20 \
        INTERPRET_CLUSTERS=true \
        MAX_CLUSTERS_TO_INTERPRET=10 \
        CLUSTER_SAMPLE_SIZE=10

    # Verify essential application modules are importable from copied code
    # Keep verification minimal here as base image verified core libs
    RUN python -c "import sys; print(sys.path)" && \
        python -c "import src.gemini.gemini_client; print('Verified gemini_client')" && \
        python -c "from src.steps.step1 import run; print('Verified step1')" && \
        python -c "from src.steps.step2 import run; print('Verified step2')" && \
        echo "APP IMPORTS VERIFIED SUCCESSFULLY"

    # Command to run the application (remains the same)
    CMD ["python", "src/database/db_setup.py"]
    ```

4.  **Update `docker-compose.yml`:**
    Ensure the `build` context in your `docker-compose.yml` still points to the directory containing the _main_ `Dockerfile` (not `base.Dockerfile`). Docker Compose will use the `FROM article-transfer-base:latest` instruction.

    ```yaml
    # docker-compose.yml (relevant service part)
    services:
      article-transfer:
        build:
          context: .
          dockerfile: Dockerfile # Explicitly state the main Dockerfile
        image: article-transfer-app:latest # Optional: Tag the final app image
        container_name: article-transfer
        # ... rest of the configuration remains the same ...
        volumes:
          - ./src:/app/src # Keep this for live code changes
          - ./logs:/app/logs
          - huggingface_cache:/root/.cache/huggingface # Keep this for HF models
        # ... rest of the configuration ...

      # ... other services ...

    volumes:
      postgres_data:
      huggingface_cache: # Keep this named volume
    ```

**Benefits:**

- **Faster Rebuilds:** When you change your application code (`src/`), only the final `COPY . .` layer and subsequent `RUN` (verification) layers in the _main_ `Dockerfile` will be rebuilt, which is extremely fast.
- **Dependency Stability:** The heavy dependencies are isolated in the base image, which you only rebuild when necessary (e.g., updating `requirements.txt`).
- **Cleaner Dockerfile:** The main `Dockerfile` becomes much more focused on the application itself.
- **BuildKit Caching:** Using `--mount=type=cache,target=/root/.cache/pip` in the base image build leverages BuildKit's advanced caching for pip downloads, making base image rebuilds faster too.
- **Model Caching:** The spaCy model is now baked into the base image. The Hugging Face cache volume in `docker-compose.yml` still helps cache transformer models downloaded _at runtime_ if any are needed beyond the base installation.

This approach provides the most significant improvement for development workflows involving frequent code changes and occasional dependency updates. Remember to rebuild the base image (`docker build -t article-transfer-base:latest -f base.Dockerfile .`) whenever you modify `requirements.txt`.
