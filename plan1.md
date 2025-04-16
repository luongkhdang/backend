# Plan 1: Intelligent Rate Limit Handling in GeminiClient

Rate limit:
Requests per minute (RPM)
Requests per day (RPD)

assorted by performance (best on top):
model (RPM) (TPM (RPD) (input) (output):

'models/gemini-2.0-flash-thinking-exp-01-21' 10 1500
'models/gemini-2.0-flash-exp' 10 1500
'models/gemini-2.0-flash' 15 1500

fallback when hit rate limit error:
'models/gemini-2.0-flash-lite' 30 1500

**Objective:** Modify the `GeminiClient` to wait for a preferred model's rate limit cooldown if the wait time is below a configurable threshold, instead of immediately switching to a lower-preference or fallback model.

**Rationale:**

- **Better Model Usage:** Maximizes the use of higher-quality, preferred models by waiting for short cooldowns instead of defaulting to potentially less capable fallback models.
- **Resource Efficiency:** Avoids unnecessary API calls to fallback models when a short wait would allow the preferred model to be used.
- **Configurability:** Allows tuning the maximum acceptable wait time based on performance needs.

**Implementation Steps:**

1.  **Modify `RateLimiter` (`src/utils/rate_limit.py`):**

    - **Add `get_wait_time` Method:**
      - Create a new public method `def get_wait_time(self, model_name: str) -> float:`.
      - This method should perform the same logic as the beginning of `wait_if_needed` (acquire lock, clean old timestamps, check current count against limit).
      - If the current count is _less than_ the limit, return `0.0` (no wait needed).
      - If the current count _meets or exceeds_ the limit:
        - Calculate the time until the oldest timestamp in the current 60-second window expires (`(oldest_call_time + 60) - current_time_monotonic`).
        - Return this calculated wait time (ensure it's non-negative, add a small buffer like 0.1s if desired).
      - Release the lock before returning.
      - Ensure the method handles cases where `model_name` has no configured limit or no timestamps exist gracefully (e.g., return 0.0).

2.  **Modify `GeminiClient` (`src/gemini/gemini_client.py`):**

    - **Add Configuration:**
      - Define a class constant or instance variable for the maximum wait time, e.g., `MAX_RATE_LIMIT_WAIT_SECONDS`.
      - Fetch its value from an environment variable (e.g., `GEMINI_MAX_WAIT_SECONDS`, default to 5-10 seconds) during `__init__`.
    - **Update `generate_text_with_prompt` (Sync):**
      - Inside the loop iterating through `all_models_to_try`.
      - Before attempting an API call, check `is_allowed = self.rate_limiter.is_allowed(model_name)`.
      - **If `is_allowed` is `False`:**
        - Call `wait_time = self.rate_limiter.get_wait_time(model_name)`.
        - **If `0 < wait_time <= self.MAX_RATE_LIMIT_WAIT_SECONDS`:**
          - Log that we are waiting for the specific model (e.g., `f"Rate limit hit for {model_name}. Waiting {wait_time:.2f}s..."`).
          - Call `self.rate_limiter.wait_if_needed(model_name)` to perform the actual wait.
          - _Do not skip the model._ Let the code proceed to attempt the API call within the existing retry logic for that model. The `wait_if_needed` call replaces the need for an immediate retry sleep for _this specific_ rate limit condition.
        - **Else (wait_time is 0 or > threshold):**
          - Log that the wait time is too long or not needed, and we are skipping to the next model (e.g., `f"Rate limit wait for {model_name} ({wait_time:.2f}s) > threshold. Skipping."`).
          - Use `continue` to move to the next model in the `all_models_to_try` loop.
      - **If `is_allowed` is `True`:**
        - Proceed with the API call attempt as usual within the retry loop.
    - **Update `generate_text_with_prompt_async` (Async):**
      - Implement the _exact same logic_ as described above for the synchronous version, but use:
        - `await self.rate_limiter.wait_if_needed_async(model_name)` when waiting.
        - Ensure logs and control flow (`continue`) are identical.

3.  **Update `docker-compose.yml` (Optional but Recommended):**
    - Add the `GEMINI_MAX_WAIT_SECONDS` environment variable with a default value (e.g., `10`) to the `article-transfer` and `backend` service definitions.

**Expected Outcome:**

- When a preferred Gemini model hits its rate limit, the `GeminiClient` will check the estimated cooldown time.
- If the cooldown is less than or equal to `GEMINI_MAX_WAIT_SECONDS`, the client will wait and then retry the _same preferred model_.
- If the cooldown is longer, the client will skip that model for the current request and try the next model in the preference list or the fallback model.
- This should result in more effective use of the preferred models without introducing excessively long delays.

**Testing:**

- Verify logging indicates when the client waits vs. when it skips a model due to rate limits.
- Simulate rate limit scenarios (if possible) or closely monitor logs during runs with high API usage.
- Test with different values for `GEMINI_MAX_WAIT_SECONDS`.
