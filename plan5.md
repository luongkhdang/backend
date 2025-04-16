# Plan 5: Files Affected by Database Schema Refactoring

Based on the provided correct table schemas, the following files likely need review and modification to align with the updated database structure:

**Core Database Layer:**

- `src/database/reader_db_client.py`: The main client coordinating database operations. Needs updates to reflect new table structures and potentially modified method signatures in modules.
- `src/database/modules/schema.py`: Contains the `CREATE TABLE` statements or schema initialization logic. Must be updated to match the correct schemas exactly.
- `src/database/modules/articles.py`: Handles SQL queries and logic related to the `articles` table.
- `src/database/modules/entities.py`: Handles SQL queries and logic related to the `entities` and `article_entities` tables.
- `src/database/modules/embeddings.py`: Handles SQL queries and logic related to the `embeddings` table.
- `src/database/modules/clusters.py`: Handles SQL queries and logic related to the `clusters` table.
- `src/database/modules/essays.py`: Handles SQL queries and logic related to the `essays` and `essay_entities` tables.
- `src/database/modules/influence.py`: Handles influence score calculations, likely referencing `entities`, `articles`, and `article_entities`. Needs checking against the new schema.
- `src/database/modules/domains.py`: Check if it interacts with tables affected by schema changes (e.g., `articles`).

**Pipeline Steps & Orchestration:**

- `src/steps/step1/__init__.py` (and potentially submodules): Collects and processes articles, generates embeddings, and interacts heavily with `articles` and `embeddings` tables via the DB client.
- `src/steps/step2/__init__.py` (and potentially submodules): Performs clustering, retrieves embeddings, updates `clusters` and `articles` tables via the DB client.
- `src/steps/step3/__init__.py` (and potentially submodules): Performs entity extraction, interacts with `articles`, `entities`, and `article_entities` tables via the DB client. Calculates influence.
- `src/steps/domain_goodness.py`: Calculates domain scores, likely using the `articles` table.
- `src/main.py`: Orchestrates steps and might call specific DB client methods directly (e.g., for influence calculation or domain goodness).

**Note:** This list prioritizes files with direct database interaction or schema definition. Other files might be indirectly affected if data structures passed between functions change as a result of schema modifications. Review the interfaces between these components.
