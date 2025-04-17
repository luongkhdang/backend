# Sophisticated Tiered Batching Plan

## Goal

Implement a more sophisticated task generation strategy for Step 3 (Entity Extraction) that:

1.  Composes batches with a defined mix of article tiers (based on priority).
2.  Assigns specific Gemini models and fallbacks based on the assigned tier.
3.  Calculates article priority based on cluster hotness and domain goodness.

## Plan Details

### 1. Priority Calculation & Tier Assignment (Run Once Initially)

- **Location**: Modify `_prioritize_articles` function in `src/steps/step3/__init__.py`.
- **Fetch Articles**: Fetch _all_ articles where `extracted_entities = FALSE` using a new method in `ReaderDBClient` (e.g., `get_all_unprocessed_articles()`).
- **Fetch Scores**: For each article, efficiently retrieve:
  - `cluster_hotness_score` from the `clusters` table (via `articles.cluster_id`).
  - `goodness_score` from the `domain_statistics` table (via `articles.domain`).
  - Requires helper methods in `ReaderDBClient` for efficient lookups.
- **Calculate Priority**: `Combined_Priority_Score = (0.65 * cluster_hotness_score) + (0.35 * goodness_score)`. Handle missing scores (default to 0).
- **Rank**: Sort all fetched unprocessed articles by `Combined_Priority_Score` (descending).
- **Assign Tiers**: Assign `processing_tier` based on rank percentages:
  - Tier 0: Top ~30%
  - Tier 1: Next ~50%
  - Tier 2: Remainder ~20%
- **Return**: Function returns the full list of articles, ranked and assigned a `processing_tier`.

### 2. Dynamic Batch Composition

- **Location**: Modify the main batching loop within the `run` function in `src/steps/step3/__init__.py`.
- **Tier Lists**: After the initial call to `_prioritize_articles`, separate the articles into three lists based on their assigned tier (`tier0_articles`, `tier1_articles`, `tier2_articles`).
- **Batch Loop**: While any tier list contains articles:
  - Create an empty `batch` list.
  - Populate the `batch` (up to 10 articles) by drawing:
    - Up to 4 articles from `tier0_articles`.
    - Up to 5 articles from `tier1_articles`.
    - Up to 1 article from `tier2_articles`.
  - Handle list depletion gracefully (take available articles; the batch size may be less than 10).
  - Proceed to the next step (Model Assignment).

### 3. Tier-Based Model Assignment

- **Location**: Inside the batch composition loop in `src/steps/step3/__init__.py`.
- **Assign Models**: For each article added to the current `batch`:
  - Add `model_to_use` and `fallback_model` keys to the article dictionary based on its `processing_tier`:
    - **Tier 0**: `model_to_use='models/gemini-2.0-flash-thinking-exp-01-21'`, `fallback_model='models/gemini-2.0-flash'`
    - **Tier 1**: `model_to_use='models/gemini-2.0-flash-exp'`, `fallback_model='models/gemini-2.0-flash'`
    - **Tier 2**: `model_to_use='models/gemini-2.0-flash'`, `fallback_model='models/gemini-2.0-flash-lite'`
  - Pass the composed `batch` (containing articles with assigned models) to `_extract_entities_batch`.

### 4. Task Definition Update

- **Location**: `_extract_entities_batch` function in `src/steps/step3/__init__.py`.
- **Include Fallback**: Ensure the `task_definition` dictionary created for `TaskManager` includes both `model_to_use` and the new `fallback_model` received from the main loop.

### 5. Gemini Client Fallback Logic Update

- **Location**: `generate_text_with_prompt_async` method in `src/gemini/gemini_client.py`.
- **Accept Parameter**: Modify the method signature to accept `fallback_model: Optional[str] = None`.
- **Update Logic**: The internal model selection/retry logic should attempt models in this order, checking rate limits before each attempt:
  1.  `model_override` (which corresponds to the tier's primary `model_to_use`).
  2.  The passed `fallback_model` (the tier's specific fallback).
  3.  The global `self.FALLBACK_MODEL` (currently `models/gemini-2.0-flash-lite`).

### 6. Database Client and Module Updates

- **ReaderDBClient/Modules**: Implement/verify necessary methods:
  - `get_all_unprocessed_articles()` in `articles.py` / `ReaderDBClient`.
  - Efficient methods to get `cluster_hotness_score` by `cluster_id` (potentially batch lookup).
  - Efficient methods to get `domain_goodness_score` by `domain` name (potentially batch lookup).

### 7. TaskManager Optimizations for Effective Batching

- **Location**: `src/utils/task_manager.py`
- **Model Coordination**: Add shared state to track model availability across tasks:

  ```python
  # Add to TaskManager.__init__
  self.model_usage_counter = {
      'models/gemini-2.0-flash-thinking-exp-01-21': 0,
      'models/gemini-2.0-flash-exp': 0,
      'models/gemini-2.0-flash': 0,
      'models/gemini-2.0-flash-lite': 0
  }
  self.models_in_cooldown = set()
  ```

- **Staggered Task Execution**: Modify `run_tasks` method to launch tasks with delay:

  ```python
  # Instead of gathering all tasks at once with asyncio.gather
  results = []
  for i, task in enumerate(awaitables):
      # Add small delay between task starts to prevent concurrent rate limit hits
      if i > 0:
          await asyncio.sleep(0.5)  # 500ms stagger between task starts
      try:
          result = await asyncio.wait_for(task, timeout=60)
          results.append(result)
      except asyncio.TimeoutError:
          # Handle timeout for individual task
          results.append((task_article_ids[i], {"error": "Task timeout"}))
  ```

- **Rate Limit Tracking**: Add methods to track and manage rate-limited models:

  ```python
  # In _run_single_task, after a rate limit is hit
  self.models_in_cooldown.add(model_name)
  # Schedule removal after the wait time
  asyncio.create_task(self._remove_from_cooldown(model_name, wait_time))

  # Add new method
  async def _remove_from_cooldown(self, model_name, wait_time):
      await asyncio.sleep(wait_time)
      if model_name in self.models_in_cooldown:
          self.models_in_cooldown.remove(model_name)
  ```

- **Smart Model Selection**: Implement pre-task model availability check:
  ```python
  # Add to _run_single_task at the beginning
  # Check if primary model is in cooldown, if so try fallback immediately
  if task_data.get('model_to_use') in self.models_in_cooldown:
      logger.info(f"Primary model {task_data.get('model_to_use')} in cooldown, using fallback")
      model_to_use = task_data.get('fallback_model', gemini_client.FALLBACK_MODEL)
  else:
      model_to_use = task_data.get('model_to_use')
  ```

### 8. Inter-Batch Cooling Period

- **Location**: The `run` function in `src/steps/step3/__init__.py`
- **Add Pause**: Add a significant pause between batches to allow rate limits to reset:
  ```python
  # Add after processing a batch
  logger.info(f"Adding cooling period after batch {batch_index}/{total_batches}")
  await asyncio.sleep(10)  # 10-second pause between batches
  ```
- **Logging**: Add enhanced logging of model rate limit status between batches:
  ```python
  # Log rate limit status before starting next batch
  for model_name in gemini_client.ALL_MODEL_RPMS.keys():
      current_rpm = rate_limiter.get_current_rpm(model_name)
      wait_time = rate_limiter.get_wait_time(model_name)
      logger.info(f"  Model: {model_name}, Current RPM: {current_rpm}, Wait time: {wait_time:.2f}s")
  ```

## Implementation Notes

- Start with the database methods and the single initial prioritization.
- Update the `GeminiClient` fallback logic carefully.
- Test thoroughly, especially the batch composition when tier lists become uneven or empty.
- Implement TaskManager optimizations to minimize rate limit deadlocks.
- Consider using environment variables for tier ratios and model assignments to allow easy tuning.
