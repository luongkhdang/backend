# Entity Extraction Batching Issue Analysis and Solution Plan

## Current Issues

1. **Uncoordinated Rate Limit Handling**: When using TaskManager to run 10 concurrent API calls, all tasks independently try to use the primary model first. When they all hit the rate limit simultaneously, they all enter the waiting state with no coordination.

2. **Model Selection Deadlock**: Each task tries the same sequence of models (from highest to lowest tier), causing them to simultaneously hit rate limits on each model before trying the next one.

3. **Independent Waits**: Each task independently waits for rate limits to clear without communicating with other tasks, potentially leading to all tasks waiting for the primary model when other models are available.

4. **No Batch Coordination**: There's no coordination or spacing between batches, causing new batches to immediately hit rate limits from the previous batch.

## Solution Plan

### 1. Implement Shared Rate Limit State

```python
# Add to TaskManager class
def __init__(self):
    self.model_availability = {
        'models/gemini-2.0-flash-thinking-exp-01-21': True,
        'models/gemini-2.0-flash-exp': True,
        'models/gemini-2.0-flash': True,
        'models/gemini-2.0-flash-lite': True
    }
    self.model_cooldown_until = {
        'models/gemini-2.0-flash-thinking-exp-01-21': 0,
        'models/gemini-2.0-flash-exp': 0,
        'models/gemini-2.0-flash': 0,
        'models/gemini-2.0-flash-lite': 0
    }
```

### 2. Smarter Task Distribution Across Models

Instead of all tasks trying the same primary model first, distribute tasks across all available models from the start:

```python
# In the _run_single_task method
# Check shared state to pick model
current_time = time.time()
available_models = []
for model, cooldown_time in self.model_cooldown_until.items():
    if current_time > cooldown_time:
        available_models.append(model)

if available_models:
    # Pick model based on task ID or round-robin
    selected_model = available_models[task_id % len(available_models)]
else:
    # All models are in cooldown, use fallback with lowest wait time
    selected_model = min(self.model_cooldown_until.items(), key=lambda x: x[1])[0]
```

### 3. Add Inter-Batch Delay

```python
# Add at the end of each batch processing loop
logger.info("Adding cooling period between batches")
await asyncio.sleep(10)  # 10-second pause between batches
```

### 4. Update Shared State When Rate Limits Are Hit

```python
# When rate limit is hit:
cooldown_time = time.time() + wait_time
self.model_cooldown_until[model_name] = cooldown_time
logger.info(f"Model {model_name} in cooldown until {cooldown_time}")
```

### 5. Simplified Fallback Strategy

Immediately fall back to another model if current model is rate-limited:

```python
# In gemini_client.py
async def select_best_available_model(self, task_manager=None):
    if task_manager and hasattr(task_manager, 'model_cooldown_until'):
        current_time = time.time()
        # Try models in order of preference
        for model in self.GENERATION_MODELS_CONFIG.keys():
            if current_time > task_manager.model_cooldown_until.get(model, 0):
                return model
        # If all models are in cooldown, use the one with earliest availability
        return min(task_manager.model_cooldown_until.items(), key=lambda x: x[1])[0]

    # If no task_manager info, use normal selection
    return list(self.GENERATION_MODELS_CONFIG.keys())[0]
```

## Implementation Priority

1. First, implement the shared model state tracking in TaskManager (points 1 and 4) - this addresses the core coordination issue
2. Add the inter-batch delay (point 3) - simplest change with immediate impact
3. Implement the smarter task distribution (point 2) - more complex but provides better utilization
4. Update the fallback strategy in gemini_client.py (point 5) - completes the coordination flow

## Minimum Viable Solution

If you want the simplest solution with immediate impact:

1. Add a 10-second delay between batches
2. Modify TaskManager to track which models are recently rate-limited
3. Have each task check this state before selecting a model

This provides basic coordination without a complete rewrite of the batching system.
