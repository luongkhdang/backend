Okay, let's outline the implementation plan for incorporating article length with a **25% weight** into the Step 3 prioritization, saving it to `lenght.md`.

This plan assumes Step 3 currently prioritizes articles based on `domain_goodness` and `cluster_hotness`.

````markdown
# Implementation Plan: Article Length Scoring for Step 3 Prioritization

**Goal:** Integrate article character length as a metric into the prioritization logic for selecting articles for entity extraction in `step3.py`, assigning it a 25% weight. This aims to optimize API usage by favoring articles with sufficient substance while deprioritizing extremely short or excessively long ones.

**Approach:** Utilize a continuous, trapezoidal scoring function for article length and integrate it into the existing priority calculation alongside domain goodness and cluster hotness.

---

## 1. Database Layer (`ReaderDBClient` & Modules)

**Objective:** Fetch the character length of article content alongside other article data needed for Step 3.

**Action:**
Modify the function within `src/database/modules/articles.py` that is responsible for fetching articles for Step 3 processing (e.g., `get_recent_unprocessed_articles` or similar).

**Specific Change:**
Add `LENGTH(content) AS content_length` to the `SELECT` statement in the SQL query.

**Example SQL Snippet (Illustrative):**

```sql
-- Inside the relevant function in src/database/modules/articles.py
SELECT
    a.id,
    a.title,
    a.domain,
    a.pub_date,
    a.cluster_id,
    a.content, -- Keep content if needed later, otherwise might remove for efficiency
    LENGTH(content) AS content_length -- Add this line
    -- other necessary columns...
FROM
    articles a
-- JOINs and WHERE clauses for unprocessed articles...
ORDER BY
    a.pub_date DESC -- or existing order
LIMIT %s;
```
````

**`ReaderDBClient`:**
No direct changes needed in `ReaderDBClient` itself, as it delegates calls. However, ensure the method signature and the processing of the returned data in Step 3 reflect the new `content_length` field.

---

## 2. Length Scoring Function (`step3.py` Logic)

**Objective:** Implement a function within the Step 3 logic to calculate a normalized score (0.0-1.0) based on article length.

**Action:**
Define the following Python function within the appropriate module in `src/steps/step3/` (e.g., in `__init__.py` or a dedicated `prioritization.py` module).

**Function:**

```python
def calculate_length_score(length_chars: Optional[int],
                            min_viable: int = 2000,
                            optimal_start: int = 4000,
                            optimal_end: int = 18000,
                            max_reasonable: int = 35000,
                            long_article_min_score: float = 0.3) -> float:
    """
    Calculates a score (0.0-1.0) based on article character length.

    Args:
        length_chars: Character count of the article content.
        min_viable: Below this length, score is 0.
        optimal_start: Score ramps from 0 to 1 between min_viable and optimal_start.
        optimal_end: Score is 1.0 between optimal_start and optimal_end.
        max_reasonable: Score ramps from 1 down to long_article_min_score between optimal_end and max_reasonable.
        long_article_min_score: Minimum score assigned to articles longer than max_reasonable.

    Returns:
        Normalized length score.
    """
    if not isinstance(length_chars, int) or length_chars <= 0:
        # Handle cases where length wasn't fetched or content is empty/null
        return 0.0

    if length_chars < min_viable:
        return 0.0
    elif length_chars < optimal_start:
        # Ensure divisor is not zero
        if optimal_start == min_viable: return 1.0 # Or 0.0 depending on desired edge case handling
        # Linear increase from 0 to 1
        return (length_chars - min_viable) / (optimal_start - min_viable)
    elif length_chars <= optimal_end:
        return 1.0
    elif length_chars <= max_reasonable:
        # Ensure divisor is not zero
        if max_reasonable == optimal_end: return long_article_min_score
        # Linear decrease from 1 down to long_article_min_score
        progress = (length_chars - optimal_end) / (max_reasonable - optimal_end)
        return 1.0 - (1.0 - long_article_min_score) * progress
    else:
        # Score for very long articles
        return long_article_min_score
```

**Configuration:** Make the threshold parameters (`min_viable`, `optimal_start`, etc.) configurable via environment variables if desired.

---

## 3. Prioritization Logic Update (`step3.py`)

**Objective:** Integrate the `length_score` into the final priority calculation used to sort articles before processing.

**Action:**
Modify the main processing loop or prioritization function within `src/steps/step3/__init__.py` or its helper modules.

**Steps:**

1.  **Fetch Data:** Call the updated `ReaderDBClient` method to get the list of unprocessed articles, ensuring `content_length` is included for each.
2.  **Get Supporting Scores:** Retrieve `domain_goodness` scores and `cluster_hotness` scores as currently implemented.
3.  **Calculate Priority:** Iterate through the fetched articles and calculate the combined priority score for each.

**Example Priority Calculation:**

```python
# Assume 'articles' is the list fetched from DB including 'content_length'
# Assume 'domain_scores' and 'cluster_scores' are dictionaries fetched previously

prioritized_articles = []
for article in articles:
    article_id = article['id']
    domain = article.get('domain')
    cluster_id = article.get('cluster_id')
    content_length = article.get('content_length') # Fetched from DB

    # Get component scores (handle missing values gracefully)
    domain_score = domain_scores.get(domain, 0.5) # Default score if domain unknown
    cluster_score = cluster_scores.get(cluster_id, 0.0) # Default score if no cluster/hotness
    length_score = calculate_length_score(content_length) # Use the function defined above

    # Calculate combined priority score with 25% weight for length
    # Adjust other weights (example assumes they were previously 50/50)
    priority_score = (domain_score * 0.375) + \
                     (cluster_score * 0.375) + \
                     (length_score * 0.25) # New weight for length

    prioritized_articles.append((priority_score, article))

# Sort articles by priority score (descending)
prioritized_articles.sort(key=lambda x: x[0], reverse=True)

# Select the top N articles for processing
# processing_batch = [article for score, article in prioritized_articles[:ENTITY_MAX_PRIORITY_ARTICLES]]
```

4.  **Process Batch:** Proceed with processing the selected batch of articles using the Gemini API.

---

## 4. Rationale & Considerations

- **Sophistication:** Uses a continuous scoring function reflecting the nuanced value of different lengths, avoiding arbitrary cutoffs.
- **Simplicity:** Integrates cleanly into existing logic, requires minimal changes to the DB query, and uses a standard trapezoidal function.
- **Configurability:** Thresholds for the scoring function can be externalized for easier tuning.
- **Weighting:** The 25% weight gives length a significant but not dominant influence, balancing it with source quality and topic relevance.
- **Error Handling:** The scoring function includes basic handling for missing or invalid length data. Ensure fetching logic in Step 3 also handles potential `KeyError` for `content_length`.

```

```
