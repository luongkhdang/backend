Part 1: Foundational Code Quality & Structure
This part focuses on immediate improvements to code structure, readability, maintainability, and basic robustness.

Refactor Batch DB Updates (step1.py):

Why: Reduces code duplication [cite: 453-456, 477-488, 514-522], making maintenance easier and less error-prone.
How: Create a single helper function that takes the list of items, lock, DB client method (e.g., reader_db.batch_insert_articles, reader_db_embed.insert_embedding), and potentially counters/timestamps as arguments.
Implementation Plan:
Define a new helper function \_batch_db_update(...) within step1.py.
Inside \_batch_db_update: Acquire lock, check if list is empty, copy list, clear original list, release lock, get DB method using getattr, call method, update counters/timestamps, handle exceptions, return updated values. (Adjust signature/logic slightly if DB methods have very different args).
Refactor update_database_step1_4, update_embedding_database_1_6, and update_embedding_database_1_7 to call \_batch_db_update with the correct parameters.
Define Constants (step1.py, potentially content_processor.py):

Why: Improves readability and makes changes easier than finding literal strings.
How: Define constants for sentinel values like 'ERROR' and potentially status strings or common keys.  
Implementation Plan:
Create src/common/constants.py.
Define constants like ARTICLE_STATUS_ERROR = 'ERROR', EMBEDDING_BATCH_SIZE = 20, EMBEDDING_CHECKPOINT_INTERVAL = 60, SHORT_ARTICLE_CHAR_LIMIT = 8000.
Import and use these constants in step1.py, reader_db_client.py, and content_processor.py where applicable, replacing the literal values.
Comprehensive Pipeline Status Return (step1.run):

Why: Essential for monitoring, debugging, and understanding pipeline performance.
How: Modify run to return a dictionary containing detailed success/failure counts for each significant step.
Implementation Plan:
Change run signature to return Dict[str, Any].
Initialize status_report: Dict[str, Any] = {}.
Populate status_report throughout the function with counts for fetched, skipped, processed, inserted, summaries attempted/success/failed, embeddings attempted/success/failed for both short and long articles.
Return status_report at the end.
Update main.py to handle and log this dictionary.
