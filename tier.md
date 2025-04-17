# Sophisticated Tiered Batching Plan

## Goal

Implement a more sophisticated task generation strategy for Step 3 (Entity Extraction) that:

1.  Composes batches with a defined mix of article tiers (based on priority).
2.  Assigns specific Gemini models and fallbacks based on the assigned tier.
3.  Calculates article priority based on cluster hotness and domain goodness.

## Plan Details

### 5. Gemini Client Fallback Logic Update

- **Location**: `generate_text_with_prompt_async` method in `src/gemini/gemini_client.py`.
- **Accept Parameter**: Modify the method signature to accept `fallback_model: Optional[str] = None`.
- **Update Logic**: The internal model selection/retry logic should attempt models in this order, checking rate limits before each attempt:
  1.  `model_override` (which corresponds to the tier's primary `model_to_use`).
  2.  The passed `fallback_model` (the tier's specific fallback).
  3.  The global `self.FALLBACK_MODEL` (currently `models/gemini-2.0-flash-lite`).

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

## Implementation Notes

- Update the `GeminiClient` fallback logic carefully.
- Implement TaskManager optimizations to minimize rate limit deadlocks.
