You want to enhance Step 1 in `main.py` to include content processing (noise removal, standardization) before storing the articles in `reader-db`. The processed content will be validated, and the storage logic will handle both valid and invalid content differently.

# Finalized Implementation Plan for Content Processing Enhancement

## 1. Update Dependencies

- **Add to requirements.txt:**

  ```
  beautifulsoup4==4.13.3  # Latest stable version as of April 2024
  spacy==3.7.5           # Latest version compatible with Python 3.10 in Dockerfile
  ```

- **Update Dockerfile:** Add spaCy model download after pip install

  ```dockerfile
  # Copy requirements and install Python dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  # Download spaCy model
  RUN python -m spacy download en_core_web_lg

  # Copy application code
  COPY . .
  ```

## 2. Create Directory Structure

- Ensure `src/refinery/` directory exists
- Create `src/refinery/__init__.py` (empty file for proper module structure)
- Create `src/refinery/content_processor.py` (new module for processing logic)

## 3. Implement Content Processing in `src/refinery/content_processor.py`

**Header:**

```python
"""
content_processor.py - Article content processing module for Data Refinery Pipeline

This module provides functions to process raw article content:
- process_article_content: Cleans article content by removing noise and standardizing text
- validate_and_prepare_for_storage: Validates processed content and prepares article data for storage

Related files:
- src/main.py: Uses these functions in the data pipeline
- src/database/news_api_client.py: Provides the raw article data
- src/database/reader_db_client.py: Handles storage of processed articles
"""
import re
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
    SPACY_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.critical(f"spaCy or model not available: {e}. Sentence boundary detection will be limited.")
    SPACY_AVAILABLE = False
```

**Functions to implement:**

1. `process_article_content(content: str) -> str`
2. `validate_and_prepare_for_storage(article: Dict[str, Any], processed_content: str) -> Dict[str, Any]`

## 4. Modify `src/main.py`

### 4.1 Update Imports

```python
from .database.news_api_client import NewsAPIClient
from .database.reader_db_client import ReaderDBClient
from .refinery.content_processor import process_article_content, validate_and_prepare_for_storage
import logging
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
```

### 4.2 Rename and Update Functions

- Rename `fetch_articles_minimal_data` → `fetch_articles_for_processing`
- Rename `insert_articles_minimal_data` → `insert_processed_articles`
- Update `main()` function to orchestrate the new workflow

### 4.3 Update Header Documentation

```python
"""
main.py - Step 1 of Data Refinery Pipeline

Step 1: Data Collection, Processing and Storage
- Fetches articles with 'ReadyForReview' status from news-db
- Processes article content (noise removal, standardization)
- Validates content and flags invalid articles
- Inserts data into reader-db with complete metadata
- Leaves additional processing for embedding and clustering to future steps

Later steps will:
- Generate embeddings (Step 2)
- Cluster articles (Step 3)
- Generate summaries (Step 4)
"""
```

## 5. Implementation and Testing Steps

1. **Update dependencies in requirements.txt**
2. **Create and implement refinery module**
3. **Update main.py**:
   - Fix imports and linter errors
   - Rename and update functions
   - Update orchestration logic in main()
4. **Test implementation**:
   - Run with Docker: `docker-compose up --build backend`
   - Monitor logs for successful article processing
   - Check database for correct article storage

## 6. Import Fix for Linter Errors

The current linter errors stem from import issues. Since the Docker environment sets `PYTHONPATH=/app`, we should update the imports in `main.py`:

```python
# For Docker environment
from src.database.news_api_client import NewsAPIClient
from src.database.reader_db_client import ReaderDBClient
from src.refinery.content_processor import process_article_content, validate_and_prepare_for_storage

# Alternative approach with path manipulation (if needed)
# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)
# from database.news_api_client import NewsAPIClient
# from database.reader_db_client import ReaderDBClient
# from refinery.content_processor import process_article_content, validate_and_prepare_for_storage
```
