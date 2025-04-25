# Concurrency and Rate Limiting Issues in Step 3 Entity Extraction

## Overview of Current Implementation

The Step 3 module performs entity extraction from articles using the Gemini API. The process:

1. Prioritizes articles based on domain goodness and cluster hotness
2. Assigns articles to tiers (0, 1, 2) based on priority
3. Assigns models to articles based on tier:
   - Tier 0: `gemini-2.0-flash-exp` (primary), `gemini-2.0-flash` (fallback)
   - Tier 1: `gemini-2.0-flash` (primary), `gemini-2.0-flash-lite` (fallback)
   - Tier 2: `gemini-2.0-flash-lite` (primary), `gemini-2.0-flash` (fallback)
4. Processes articles in batches with concurrency using `TaskManager` and `asyncio.gather`

## Identified Issues

### 1. Excessive Use of `gemini-2.0-flash-exp` Model

**Problem**: The logs show `gemini-2.0-flash-exp` being used excessively instead of distributing between models.

**Root Causes**:

- **RateLimiter Lock Sharing**: The RateLimiter uses a shared lock for all models, but the `wait_if_needed_async` method in the GeminiClient is called concurrently by multiple tasks from `asyncio.gather`. This could lead to race conditions where:

  - Tasks check rate limits for a model nearly simultaneously
  - Multiple tasks see that the model is available before any task registers its usage
  - All tasks proceed to use the same model (likely `gemini-2.0-flash-exp` as it's the highest priority)
  - Only after API calls complete do they register their usage, by which time the rate limit is exceeded

- **Deceptive Rate Limit Logic**: In `gemini_client.py`, method `generate_text_with_prompt_async` does a rate check followed by a wait, but doesn't re-check if the wait time was substantial. Instead, it assumes the model is now available after waiting, which may not be true in concurrent scenarios.

- **Concurrency without Coordination**: Using `asyncio.gather` creates multiple concurrent tasks that don't coordinate model selection with each other. Each task independently decides which model to use without knowledge of what other tasks are choosing.

### 2. Processing Stuck in Batch 2

**Problem**: Execution frequently gets stuck in batch 2 due to communication issues between concurrent tasks or incorrect tracking of rate limits.

**Root Causes**:

- **Batch Level Cooldown Inadequate**: The inter-batch delay of 10 seconds (`INTER_BATCH_DELAY_SECONDS`) doesn't adequately account for model rate limits, especially when previous batch substantially used high-priority models.

- **No Batch-Level Model Distribution**: There's no mechanism to distribute models across tasks within a batch. The system lets each task independently decide which model to use without coordination.

- **Rate Limit Window Sliding**: The rate limiter uses a sliding 60-second window for limiting requests. This creates a complex interaction in a concurrent environment where the availability calculation can change rapidly.

- **EmergencyWait Logic Issues**: When rate limits are reached, the emergency fallback logic may not be effective because all tasks might be waiting for the same fallback model, causing the entire batch to stall.

## Race Condition in Detail

The core issue appears to be a race condition in how rate limits are checked and enforced in a concurrent environment:

1. The TaskManager launches N concurrent tasks via `asyncio.gather`
2. Each task calls `gemini_client.generate_text_with_prompt_async`
3. Each task checks rate limits for its preferred model almost simultaneously
4. All tasks see the same model as available (e.g., `gemini-2.0-flash-exp`)
5. All tasks proceed to use this model without coordination
6. The API calls complete and they register usage
7. The rate limit for that model is now exceeded
8. Subsequent tasks or batches encounter rate limits and may stall waiting

## Configuration Observations

- The rate limits defined in `docker-compose.yml` are:

  - `gemini-2.0-flash-exp`: 10 RPM
  - `gemini-2.0-flash`: 15 RPM
  - `gemini-2.0-flash-lite`: 30 RPM

- Max rate limit wait time (`GEMINI_MAX_WAIT_SECONDS`) is set to 40 seconds
- Emergency fallback wait time is 120 seconds

These limits mean that coordinating model usage across concurrent tasks is critical to achieve optimal throughput.

## Proposed Solution

To address the identified issues while maintaining the core architecture, I propose the following solution:

### 1. Pre-Allocate Models at Batch Level

Instead of letting each task independently select a model, implement a batch-level model allocation strategy:

```python
async def _extract_entities_batch(gemini_client, articles, task_manager):
    # Group articles by their assigned model_to_use
    model_groups = {}
    for article in articles:
        model = article.get('model_to_use')
        if model:
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(article)

    # Pre-check rate limits and redistribute if necessary
    revised_articles = []
    for model_name, model_articles in model_groups.items():
        # Check how many slots are available for this model
        slots_available = 0
        if gemini_client.rate_limiter:
            with gemini_client.rate_limiter.lock:  # Important: use lock to avoid race conditions
                limit = gemini_client.rate_limiter.model_rpm_limits.get(model_name, 0)
                current_rpm = gemini_client.rate_limiter.get_current_rpm(model_name)
                slots_available = max(0, limit - current_rpm)

        # If we have enough slots, keep the original model assignment
        if slots_available >= len(model_articles):
            revised_articles.extend(model_articles)
        else:
            # Assign slots_available articles to keep their original model
            for i, article in enumerate(model_articles):
                if i < slots_available:
                    revised_articles.append(article)
                else:
                    # For the rest, select an alternative model based on tier
                    # Use a weighted distribution based on available capacity of other models
                    tier = article.get('processing_tier')
                    fallback = article.get('fallback_model')

                    # Try fallback first if it has capacity
                    fallback_slots = 0
                    if fallback and gemini_client.rate_limiter:
                        with gemini_client.rate_limiter.lock:
                            fallback_limit = gemini_client.rate_limiter.model_rpm_limits.get(fallback, 0)
                            fallback_rpm = gemini_client.rate_limiter.get_current_rpm(fallback)
                            fallback_slots = max(0, fallback_limit - fallback_rpm)

                    if fallback_slots > 0:
                        article['model_to_use'] = fallback
                        logger.info(f"Article {article.get('id')} reassigned from {model_name} to fallback {fallback}")
                    else:
                        # Find any model with capacity
                        for alt_model, alt_limit in gemini_client.rate_limiter.model_rpm_limits.items():
                            if alt_model != model_name and alt_model != fallback:
                                with gemini_client.rate_limiter.lock:
                                    alt_rpm = gemini_client.rate_limiter.get_current_rpm(alt_model)
                                    alt_slots = max(0, alt_limit - alt_rpm)

                                if alt_slots > 0:
                                    article['model_to_use'] = alt_model
                                    logger.info(f"Article {article.get('id')} reassigned from {model_name} to alternative {alt_model}")
                                    break

                    revised_articles.append(article)

    # Now run the tasks with the revised model assignments
    results = await task_manager.run_tasks(gemini_client, revised_articles)
    return results
```

### 2. Improve the Rate Limiter for Concurrent Access

Modify the RateLimiter to better handle concurrent task execution:

```python
async def reserve_and_register_call_async(self, model_name: str) -> bool:
    """
    Atomically check if a call is allowed, and if so, immediately register it.
    This prevents the race condition where multiple tasks check and then register.

    Returns:
        bool: True if the call was allowed and registered, False otherwise
    """
    with self.lock:
        # Check if there's capacity
        if self.is_allowed(model_name):
            # Immediately register the call
            self.register_call(model_name)
            return True
        return False
```

### 3. Modify GeminiClient for Atomic Check and Register

Update `generate_text_with_prompt_async` to use the new atomic check:

```python
# In the rate limit check section
if self.rate_limiter:
    # Try to atomically reserve a slot for this model
    slot_reserved = await self.rate_limiter.reserve_and_register_call_async(model_id)
    if not slot_reserved:
        # No slot available, try waiting if within threshold
        wait_time = await self.rate_limiter.get_wait_time_async(model_id)
        if 0 < wait_time <= self.max_rate_limit_wait_seconds:
            logger.debug(f"[{model_id}] Rate limit hit. Waiting {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
            # Try reserving again after waiting
            slot_reserved = await self.rate_limiter.reserve_and_register_call_async(model_id)
            if not slot_reserved:
                logger.warning(f"[{model_id}] Still rate limited after waiting. Skipping.")
                continue  # Try next model
        else:
            logger.warning(f"[{model_id}] Wait time {wait_time:.2f}s exceeds threshold. Skipping.")
            continue  # Try next model

    # If we reach here, we have reserved a slot
    # Make API call, but don't need to register after (already done)
    # ...
```

### 4. Implement Dynamic Batch Sizing

Adjust the batch size based on model availability:

```python
# At the start of each batch processing loop
adjusted_batch_size = BATCH_SIZE
if batch_index > 0:  # Not the first batch
    # Check current rate limit status
    rate_limited_models = []
    for model_name in gemini_client.rate_limiter.model_rpm_limits.keys():
        current_rpm = gemini_client.rate_limiter.get_current_rpm(model_name)
        limit = gemini_client.rate_limiter.model_rpm_limits.get(model_name, 0)
        available_slots = max(0, limit - current_rpm)
        if available_slots < adjusted_batch_size / 3:  # If less than 1/3 of batch size slots available
            rate_limited_models.append(model_name)

    # If most models are rate limited, reduce batch size
    if len(rate_limited_models) >= 2:  # At least 2 models are heavily used
        adjusted_batch_size = max(3, adjusted_batch_size // 2)  # Reduce batch size but keep minimum of 3
        logger.info(f"Reduced batch size to {adjusted_batch_size} due to rate limit pressure")
```

### 5. Improve Inter-Batch Cooldown

Make the inter-batch delay more intelligent:

```python
# After processing a batch, before starting the next one
max_model_wait_time = 0
for model_name in gemini_client.rate_limiter.model_rpm_limits.keys():
    wait_time = gemini_client.rate_limiter.get_wait_time(model_name)
    max_model_wait_time = max(max_model_wait_time, wait_time)

# Dynamically adjust cooldown based on rate limit status
if max_model_wait_time > INTER_BATCH_DELAY_SECONDS:
    # Some model needs more time to cool down
    cooldown_time = min(max_model_wait_time, 30)  # Cap at 30 seconds
    logger.info(f"Extended cooldown to {cooldown_time:.2f}s based on rate limits")
    await asyncio.sleep(cooldown_time)
else:
    # Standard cooldown is sufficient
    logger.info(f"Standard cooldown of {INTER_BATCH_DELAY_SECONDS}s between batches")
    await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)
```

### 6. Reserved Capacity for Emergency Fallback

Ensure the emergency fallback model always has reserved capacity:

```python
# Add this to the GeminiClient initialization
def __init__(self):
    # ... existing code ...
    # Reserved capacity for emergency fallback
    self.reserved_fallback_slots = 2  # Reserve 2 slots for emergency fallback

# Modify is_allowed in RateLimiter
def is_allowed(self, model_name: str) -> bool:
    with self.lock:
        limit = self.model_rpm_limits.get(model_name)
        if limit is None:
            return True

        # Get client reference to check if this is the fallback model
        client = self._client_ref() if hasattr(self, '_client_ref') else None
        is_fallback = client and hasattr(client, 'fallback_model_id') and model_name == client.fallback_model_id

        # Apply reservation for fallback model
        effective_limit = limit
        if is_fallback and hasattr(client, 'reserved_fallback_slots'):
            reserved = client.reserved_fallback_slots
            effective_limit = limit - reserved

        current_time = time.monotonic()
        timestamps = self.call_timestamps[model_name]

        # ... rest of existing code with effective_limit ...
        current_count = len(timestamps)
        allowed = current_count < effective_limit  # Use effective_limit

        # Special case: if this is the fallback model in emergency mode, ignore reservation
        if not allowed and is_fallback and current_count < limit and getattr(client, '_emergency_mode', False):
            allowed = True

        return allowed
```

This solution addresses the core issues while maintaining the existing architecture. It adds coordination at the batch level while still allowing for concurrent execution, reduces race conditions through atomic operations, and adapts to changing rate limit conditions.
