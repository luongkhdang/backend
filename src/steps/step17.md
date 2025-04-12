# Improvement Plan for Step 1.7 Summarization and Embedding

Based on feedback, the current Step 1.7 implementation in `step1.py` and `localnlp_client.py` has limitations regarding handling very long articles for embedding. Specifically, it truncates input to the BART model's 1024-token limit, potentially losing information, and doesn't explicitly handle chunking strategies for articles significantly exceeding the embedding model's token limit (e.g., 2000 tokens for `text-embedding-004`).

This plan outlines modifications to adopt a chunking and concatenation strategy for long articles, ensuring a single, high-quality 768D embedding is generated per article, consistent with the handling of shorter articles.

## Goal

Generate one high-quality 768D embedding per article using `text-embedding-004` (Task: CLUSTERING).

- Articles <= 2000 tokens: Embed directly (Handled by Step 1.6).
- Articles > 2000 tokens: Summarize using a chunking strategy, concatenate summaries, then embed the result (New Step 1.7 logic).

## Proposed Changes

**1. `step1.py` - Modify Step 1.7 Logic (`run` function):**

- **Identify Candidates for Summarization:**
  - Modify the query in `ReaderDBClient.get_articles_needing_summarization` to fetch articles _potentially_ needing summarization (e.g., those without embeddings and exceeding a certain character length threshold, like 8000 chars, as a pre-filter). The final decision will be based on token count.
  - Alternatively, fetch all articles without embeddings and perform the token check entirely within Step 1.7.
- **Token Count Check:**
  - **Import spaCy:** Add `import spacy` at the top.
  - **Load spaCy Model:** Load `en_core_web_lg` (ensure it's available via Dockerfile). `nlp = spacy.load("en_core_web_lg")`.
  - **Iterate Through Candidates:** Loop through articles retrieved for embedding/summarization.
  - **Calculate Token Count:** For each article's `content`, use `len(nlp(content))` or a more precise tokenizer if available to get the actual token count.
  - **Branching Logic:**
    - **If `token_count <= ARTICLE_TOKEN_CHUNK_THRESHOLD` (e.g., 2000):** This article should have been handled by Step 1.6 or can be embedded directly here if Step 1.6 logic is merged/modified. _Ensure no overlap or duplication with Step 1.6._
    - **If `token_count > ARTICLE_TOKEN_CHUNK_THRESHOLD`:** Proceed with chunking and summarization.
- **Chunking Logic (for articles > threshold):**
  - Process the article `content` with spaCy: `doc = nlp(content)`.
  - Initialize an empty list `chunks = []` and `current_chunk_tokens = 0`, `current_chunk_sentences = []`.
  - Iterate through sentences (`sent` in `doc.sents`):
    - Get sentence token count (`len(sent)`).
    - If adding the sentence doesn't exceed `TARGET_CHUNK_TOKEN_SIZE` (e.g., 1000-1024):
      - Append `sent.text` to `current_chunk_sentences`.
      - Add `len(sent)` to `current_chunk_tokens`.
    - Else (adding sentence exceeds chunk size):
      - Join `current_chunk_sentences` into a string and add to `chunks`.
      - Reset `current_chunk_sentences = [sent.text]` and `current_chunk_tokens = len(sent)`.
  - Add the last remaining chunk to `chunks`.
- **Summarize Chunks:**
  - Initialize `chunk_summaries = []`.
  - Loop through each `chunk` string in `chunks`:
    - Call `summary = localnlp_client.summarize_text(chunk, max_summary_tokens=CHUNK_MAX_TOKENS, min_summary_tokens=CHUNK_MIN_TOKENS)`.
    - Use appropriate `CHUNK_MAX_TOKENS` (e.g., 300) and `CHUNK_MIN_TOKENS` (e.g., 75) configured via environment variables or constants.
    - If `summary` is not None, append it to `chunk_summaries`. Handle potential summarization errors per chunk.
- **Concatenate Summaries:**
  - `concatenated_summary = " ".join(chunk_summaries)`.
  - Check the length of `concatenated_summary`. If it _still_ exceeds the embedding model's limit (unlikely but possible), truncate it carefully.
- **Generate Single Embedding:**
  - `embedding = gemini_client.generate_embedding(concatenated_summary)` (or the original content if <= threshold).
- **Store Embedding:**
  - Call `reader_db_embed.insert_embedding(article_id, embedding)` for the article.
- **Refactor Parallel Processing:** The `process_summarization_embedding_task` function needs to be refactored to include this new chunking, summarizing, concatenating, and embedding logic for a single article. The `ThreadPoolExecutor` will now manage tasks executing this complete workflow per article.

**2. `localnlp_client.py` - Modify `LocalNLPClient`:**

- **Input Handling:** The `summarize_text` method will now receive _chunks_ (expected to be around 1000 tokens).
- **Remove Premature Truncation (within `summarize_text`):** The logic that explicitly truncated input text to `MODEL_MAX_INPUT_TOKENS` _before_ passing it to the pipeline might be redundant if chunks are guaranteed to be near/below this limit. However, the `transformers` pipeline _itself_ handles truncation if the input exceeds the model's capacity, so leaving the pipeline's internal truncation (`truncation=True`) enabled is fine. The client no longer needs its own separate truncation step _before_ the pipeline call.
- **Parameter Meaning:** Clarify in comments that `max_summary_tokens` and `min_summary_tokens` now apply to _chunk_ summaries.
- **Retain Thread Safety:** Keep the `threading.Lock` as multiple chunks (potentially from different articles handled by different threads) might still be processed concurrently.

**3. `database/reader_db_client.py` - Modify Queries (Optional but Recommended):**

- **`get_articles_needing_embedding`:** Rename/refactor potentially. This function in Step 1.6 should strictly get articles _below_ the token threshold (e.g., <= 2000 tokens).
- **`get_articles_needing_summarization`:** Rename/refactor. This function should fetch articles _above_ the token threshold (e.g., > 2000 tokens) to be processed by the new Step 1.7 logic.
- **Token Count Integration:** Accurately filtering by token count in SQL is difficult. It's more feasible to fetch candidates based on character length (e.g., `LENGTH(content) > 8000`) and perform the precise token count check in `step1.py`.

**4. Configuration (`.env`, `docker-compose.yml`):**

- **Review/Update `MAX_SUMMARY_TOKENS`:** Default changed to `512` (or maybe `300` if specifically for chunks). This now controls the _max length of individual chunk summaries_.
- **Review/Update `MIN_SUMMARY_TOKENS`:** Default changed to `150` (or maybe `75`). This controls the _min length of individual chunk summaries_.
- **Add `ARTICLE_TOKEN_CHUNK_THRESHOLD`:** New variable. Defines the token limit above which articles are chunked (e.g., `2000`).
- **Add `TARGET_CHUNK_TOKEN_SIZE`:** New variable. Defines the target size for each chunk (e.g., `1000`). Ensure this is <= `LocalNLPClient.MODEL_MAX_INPUT_TOKENS`.

**5. Documentation and Comments:**

- Update header comments and inline comments in `step1.py` and `localnlp_client.py` to reflect the new chunking strategy, the purpose of token limits, and the single-embedding output goal.

## Expected Outcome

- Articles <= 2000 tokens embedded directly via `text-embedding-004`.
- Articles > 2000 tokens are chunked (~1000 tokens/chunk), each chunk summarized (~75-300 tokens/summary), summaries concatenated, and the final text embedded via `text-embedding-004`.
- All articles result in a single 768D embedding stored in the database.
- Information loss from premature truncation is minimized.
- The "max_length too large" warning should be resolved.
- Thread-safety issues ("Already borrowed") should be resolved.

This plan provides a clear path to address the identified issues and implement a more robust summarization and embedding pipeline for long articles.
